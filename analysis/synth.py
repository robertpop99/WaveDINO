"""
Synthetic-data analyses (Supplementary Sections S2-S4 and S6). Loads one trained checkpoint and runs the synthetic
test split under a set of controlled variants, writing one CSV row per test sample so any stratification can be done
offline (analysis/synth_summary.py).

Analyses (--analysis), with the Supplementary section each one feeds:
    atm     S4      standard synthetic test (same distribution as Table I); per-sample atmospheric strength and
                    coherence are recorded so errors can be split into quartiles.
    unwrap          adds +-2*pi*k unwrapping errors over random connected regions covering a given fraction of
                    the land pixels, for k in {1, 2} and area in {1, 5, 10, 20} %, plus the clean case.
    pure    S2      deformation only, no atmosphere: one Mogi source per sample with its peak scaled to a fixed
                    amplitude; records source depth, the regression slope/bias of recovered vs true deformation,
                    the far-field error (signal fabricated or removed away from the source) and (WaveDINO) the
                    predicted ramp.
    grid    S3      real short-baseline atmosphere plus one Mogi source placed uniformly at random over the
                    full frame; records the source position so error can be mapped against it.
    interp  S6      NaN filling of the input interferogram: nearest + Gaussian ('new', default), biharmonic
                    ('classic') and zero fill.

The clean variant of `unwrap` is the `atm` test (same seed and distribution), so `unwrap` alone also covers `atm`.

Every variant replays the same samples (same seeds), so variants are paired. `pure` and `grid` use the full
frame resized to 128 (the inference mode used for Table II) and only short-baseline (< 20 days) scenes;
the others use the mixed crop/resize test distribution of Table I (override with --inference-mode).

Output: results/{dataset}_synth_analysis/{run_folder}_{analysis}.csv

Example:
    python -m analysis.synth --analysis unwrap --dataset campi_flegrei --model-type wavedino-4layers \
        --model-name results/campi_flegrei_128/<WaveDino-4layers run folder> \
        --batch-size 35 --num-workers 5
"""
import argparse
import os
import random
import time
import types

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
from scipy.io import loadmat

from main import get_trainer_class
from models.wavemamba.wavedino_trainer import WaveDinoTrainer
from util.datasets import (get_datasets, collate_fn, time_length_filter, AddDeformation, CustomRandomCrop,
                           Interpolate, OutputType, rad_to_mm, DEFAULT_DAY_INTERVAL)
from util.parser import parse_args

ANALYSES = ['atm', 'unwrap', 'pure', 'grid', 'interp']

UNWRAP_AREAS = [0.01, 0.05, 0.10, 0.20]
UNWRAP_JUMPS = [1, 2]
PURE_AMPLITUDES_MM = [10., 20., 40., 80.]
GRID_AMPLITUDES_MM = [10., 20., 40.]
INTERPOLATIONS = ['new', 'classic', 'zero']

# BringToInterval(-100, 100) maps radians x to x / 100
RAD_TO_INPUT = 1. / 100.


def mm_to_rad(x):
    return x / 28 * (2 * np.pi)


def parse_all_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--analysis', required=True, choices=ANALYSES)
    parser.add_argument('--test-limit', type=int, default=-1,
                        help='samples per variant (default: one pass over the test split)')
    parser.add_argument('--inference-mode', choices=['mixed', 'resize', 'crop'], default=None,
                        help='mixed = Table I test distribution, resize = full frame -> 128, crop = 128 crops')
    parser.add_argument('--analysis-seed', type=int, default=10)
    extra, remaining = parser.parse_known_args()
    args = parse_args(remaining)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    if args.inference_mode is None:
        args.inference_mode = 'resize' if args.analysis in ['pure', 'grid'] else 'mixed'
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------------------------------------------
# Test-set control. INSAR.__getitem__ dispatches on `type(t) is AddDeformation`, so the existing transform
# instances are patched in place instead of being replaced by subclasses.
# ---------------------------------------------------------------------------------------------------------------

def dataset_transforms(ds):
    return [ds.safe_transform] + list(ds.all_transforms)


def find_transforms(ds, cls):
    return [t for trs in dataset_transforms(ds) for t in trs.transforms if type(t) is cls]


def set_inference_mode(ds, args, original):
    """Restores the dataset's original transforms, then optionally forces one inference mode."""
    ds.safe_transform, ds.all_transforms, ds.safe_file = original

    if args.inference_mode == 'mixed':
        return

    # reuse the dataset's own ('resize', 128) / ('crop', 128) chain, so preprocessing is identical to Table I
    def is_mode(trs):
        types_ = [type(t) for t in trs.transforms]
        if args.inference_mode == 'resize':
            return Resize in types_ and CustomRandomCrop not in types_ and \
                any(t.size in [128, (128, 128), [128, 128]] for t in trs.transforms if type(t) is Resize)
        return any(t.size == (128, 128) for t in trs.transforms if type(t) is CustomRandomCrop)

    trs = [t for t in original[1] if is_mode(t)]
    assert len(trs) == 1, f'expected one {args.inference_mode} 128 transform, found {len(trs)}'
    trs = trs[0]
    ds.safe_transform = trs
    ds.all_transforms = [trs]
    ds.safe_file = lambda file: True


def set_interpolation(ds, interpolation):
    def zero_fill(self, img):
        return torch.nan_to_num(img, nan=0.)

    for t in find_transforms(ds, Interpolate):
        if interpolation == 'new':
            t.interpolate_function = t.interpolate_nans_new
        elif interpolation == 'classic':
            t.interpolate_function = t.interpolate_classic
        elif interpolation == 'zero':
            t.interpolate_function = types.MethodType(zero_fill, t)
        else:
            raise ValueError(interpolation)


def set_controlled_deformation(ds, placement, amplitudes_mm, with_atmosphere):
    """
    Replaces the random multi-source deformation with a single Mogi source whose peak is scaled to one of
    `amplitudes_mm`. placement: 'train' = same random offset as training, 'uniform' = source centre uniform over
    the frame (10-90 %). Source depth, amplitude and relative position travel in DEFORM_CHARS
    ([depth_km, amplitude_mm, pos_y, pos_x]; INSAR copies DEFORM_COORDS into the last two slots).
    """
    def controlled_forward(self, img):
        sample_name = self.samples[torch.randint(low=0, high=self.size, size=(1,)).item()]
        depth = float(sample_name.split('depth')[-1].replace('.mat', ''))
        grid = loadmat(sample_name)['los_grid']
        amplitude_mm = amplitudes_mm[torch.randint(low=0, high=len(amplitudes_mm), size=(1,)).item()]
        h, w = img.shape[-2:]

        if placement == 'train':
            deform = self.transform(grid)
        else:
            deform = torch.from_numpy(grid).view(1, *grid.shape)
            peak_y, peak_x = np.unravel_index(np.abs(grid).argmax(), grid.shape)
            pos = 0.1 + 0.8 * torch.rand(2)
            target_y, target_x = pos[0].item() * (h - 1), pos[1].item() * (w - 1)
            # align the grid centre with the frame centre, then move the peak to the target
            dy = target_y - peak_y + (grid.shape[0] - h) / 2
            dx = target_x - peak_x + (grid.shape[1] - w) / 2
            deform = TF.affine(deform, angle=0., translate=[int(round(dx)), int(round(dy))], scale=1., shear=[0., 0.],
                               fill=0.)
            deform = TF.center_crop(deform, [h, w])

        deform = deform.to(img.dtype)
        peak = torch.abs(deform).max()
        deform = deform / peak * mm_to_rad(amplitude_mm) if peak > 0 else deform

        peak_pos = np.unravel_index(torch.abs(deform[0]).argmax().item(), (h, w))
        coords = torch.tensor([peak_pos[0] / (h - 1), peak_pos[1] / (w - 1)])

        atmosphere = img if with_atmosphere else torch.zeros_like(img)
        image = atmosphere + deform
        chars = torch.tensor([depth, amplitude_mm, 0., 0.])
        return {
            OutputType.IMAGE: image,
            OutputType.DEFORMATION: deform,
            OutputType.ORIG_IMG: atmosphere,
            OutputType.DEFORM_CHARS: chars,
            OutputType.DEFORM_COORDS: coords,
        }

    for t in find_transforms(ds, AddDeformation):
        t.forward = types.MethodType(controlled_forward, t)


def reset_deformation(ds):
    for t in find_transforms(ds, AddDeformation):
        if 'forward' in vars(t):
            del t.forward


def short_baseline_indices(ds):
    return [i for i in range(len(ds)) if time_length_filter(ds.dataset.get_path(i), DEFAULT_DAY_INTERVAL)]


# ---------------------------------------------------------------------------------------------------------------
# Unwrapping errors
# ---------------------------------------------------------------------------------------------------------------

def unwrapping_error(land, area, jump, generator):
    """+-2*pi*jump over smooth random regions covering `area` of the land pixels, in network input units."""
    B, _, H, W = land.shape
    noise = torch.randn(B, 1, H, W, generator=generator).to(land.device)
    sigma = H / 16
    kernel = int(4 * sigma) * 2 + 1
    score = TF.gaussian_blur(noise, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

    error = torch.zeros_like(noise)
    signs = torch.randint(low=0, high=2, size=(B,), generator=generator) * 2 - 1
    for b in range(B):
        valid = land[b, 0]
        if valid.sum() == 0:
            continue
        threshold = torch.quantile(score[b, 0][valid], 1 - area)
        region = (score[b, 0] >= threshold) & valid
        error[b, 0][region] = signs[b].item() * jump * 2 * np.pi
    return error * RAD_TO_INPUT


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------

def to_mm(x):
    return rad_to_mm(x * 100)


def per_sample_stats(pred, true, atm, land, coherence, ramp):
    """All maps (B, 1, H, W) in mm except land (bool) and coherence."""
    rows = []
    for b in range(pred.shape[0]):
        p, t, a, m = pred[b, 0], true[b, 0], atm[b, 0], land[b, 0]
        err = p - t
        row = {
            'mse': (err ** 2).mean().item(),
            'mae': err.abs().mean().item(),
            'true_max': t.abs().max().item(),
            'pred_max': p.abs().max().item(),
        }
        # signed gain at the true peak (3x3 mean), robust to where the network puts its own maximum
        py, px = np.unravel_index(t.abs().argmax().item(), t.shape)
        ys, xs = slice(max(py - 1, 0), py + 2), slice(max(px - 1, 0), px + 2)
        row['true_at_peak'] = t[ys, xs].mean().item()
        row['pred_at_peak'] = p[ys, xs].mean().item()
        if m.sum() > 10:
            pl, tl, el = p[m], t[m], err[m]
            row.update({
                'mse_land': (el ** 2).mean().item(),
                'mae_land': el.abs().mean().item(),
                'resid_std_land': el.std().item(),
                'atm_std': a[m].std().item(),
                'atm_p98_p2': (torch.quantile(a[m], 0.98) - torch.quantile(a[m], 0.02)).item(),
                'coh_mean': coherence[b, 0][m].mean().item(),
                'land_fraction': m.float().mean().item(),
            })
            # recovered = slope * true + bias, only where there is deformation to regress on
            if tl.std() > 0.1:
                tc = tl - tl.mean()
                slope = (tc * (pl - pl.mean())).sum() / (tc ** 2).sum()
                row['slope'] = slope.item()
                row['bias'] = (pl.mean() - slope * tl.mean()).item()
                # error where the true deformation is < 5 % of its peak: signal fabricated or removed far away
                far = tl.abs() < 0.05 * tl.abs().max()
                if far.sum() > 10:
                    row['far_field_err_abs'] = el[far].abs().mean().item()
                    row['far_field_err_mean'] = el[far].mean().item()
        if ramp is not None:
            r = ramp[b, 0]
            row['ramp_std'] = r.std().item()
            row['ramp_max'] = r.abs().max().item()
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------------------------------------------

def run_variant(trainer, dataset, args, variant, perturb=None):
    seed_everything(args.analysis_seed)
    generator = torch.Generator().manual_seed(args.analysis_seed)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=True, collate_fn=collate_fn)
    test_limit = args.test_limit if args.test_limit > 0 else len(dataset)

    captured = {}
    hook = None
    if isinstance(trainer, WaveDinoTrainer):
        hook = trainer.model.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__('ramp', output[1] if isinstance(output, tuple) else None))

    rows = []
    start = time.time()
    with torch.no_grad():
        while len(rows) < test_limit:
            for data_dict in loader:
                data_dict = trainer.move_batch_to_device(data_dict)
                land = data_dict[OutputType.WATER_MASK] > 0.5

                if perturb is not None:
                    data_dict[OutputType.IMAGE] = data_dict[OutputType.IMAGE] + perturb(land, generator)

                captured.clear()
                pred = trainer.test_step(data_dict)[:, 0:1]
                ramp = captured.get('ramp')

                batch_rows = per_sample_stats(
                    pred=to_mm(pred),
                    true=to_mm(data_dict[OutputType.DEFORMATION]),
                    atm=to_mm(data_dict[OutputType.ORIG_IMG]),
                    land=land,
                    coherence=data_dict[OutputType.COHERENCE],
                    ramp=to_mm(ramp) if ramp is not None else None,
                )
                chars = data_dict[OutputType.DEFORM_CHARS].cpu().numpy()
                for i, row in enumerate(batch_rows):
                    row.update(variant)
                    row['name'] = data_dict[OutputType.NAME][i]
                    row['time_length'] = data_dict[OutputType.TIME_LENGTH][i]
                    if args.analysis in ['pure', 'grid']:
                        row['depth_km'], row['amplitude_mm'], row['pos_y'], row['pos_x'] = chars[i]
                rows.extend(batch_rows)

                if len(rows) >= test_limit:
                    break

    if hook is not None:
        hook.remove()

    df = pd.DataFrame(rows[:test_limit])
    print(f'==> {variant}: {len(df)} samples in {time.time() - start:.0f} s | '
          f'MSE {df["mse"].mean():.2f} mm2 | MAE {df["mae"].mean():.2f} mm | '
          f'MSE land {df["mse_land"].mean():.2f} mm2', flush=True)
    return df


def main():
    args = parse_all_args()
    seed_everything(args.seed)  # same split as main.py
    loaders = get_datasets(args, None, resolution_mode='adaptive')

    trainer = get_trainer_class(args.model_type)(args, loaders, None, None, None)
    model_path = trainer.load_model(args.model_name, load_best_model=True)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    trainer.prepare_test()

    run_folder = [f for f in model_path.split('/') if f][-1]
    out_dir = os.path.join(args.result_dir, f'{args.dataset}_synth_analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{run_folder}_{args.analysis}.csv')
    print(f'==> {args.analysis} analysis for {model_path}, inference mode {args.inference_mode}, writing {out_file}',
          flush=True)

    dataset = loaders['test'].dataset
    original = (dataset.safe_transform, dataset.all_transforms, dataset.safe_file)
    set_inference_mode(dataset, args, original)

    common = {'analysis': args.analysis, 'model': run_folder, 'inference_mode': args.inference_mode}
    results = []

    if args.analysis == 'atm':
        results.append(run_variant(trainer, dataset, args, dict(common)))

    elif args.analysis == 'unwrap':
        results.append(run_variant(trainer, dataset, args, {**common, 'area': 0., 'jump': 0}))
        for jump in UNWRAP_JUMPS:
            for area in UNWRAP_AREAS:
                perturb = lambda land, gen, a=area, k=jump: unwrapping_error(land, a, k, gen)
                results.append(run_variant(trainer, dataset, args, {**common, 'area': area, 'jump': jump}, perturb))

    elif args.analysis == 'interp':
        for interpolation in INTERPOLATIONS:
            set_interpolation(dataset, interpolation)
            results.append(run_variant(trainer, dataset, args, {**common, 'interpolation': interpolation}))
        set_interpolation(dataset, args.interpolation)

    elif args.analysis in ['pure', 'grid']:
        subset = torch.utils.data.Subset(dataset, short_baseline_indices(dataset))
        print(f'==> {len(subset)} / {len(dataset)} test scenes are short-baseline', flush=True)
        if args.analysis == 'pure':
            set_controlled_deformation(dataset, placement='train', amplitudes_mm=PURE_AMPLITUDES_MM,
                                       with_atmosphere=False)
        else:
            set_controlled_deformation(dataset, placement='uniform', amplitudes_mm=GRID_AMPLITUDES_MM,
                                       with_atmosphere=True)
        results.append(run_variant(trainer, subset, args, dict(common)))
        reset_deformation(dataset)

    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_file, index=False)
    print(f'==> Saved {len(df)} rows to {out_file}', flush=True)


if __name__ == '__main__':
    main()
