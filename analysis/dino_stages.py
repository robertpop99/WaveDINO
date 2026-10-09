"""
Which DINOv3 stages carry useful information for WaveDino? (input standardised and clipped to +-3)

The 3-level paper model (wavedino-4layers) uses stage 1 (stride 4, 128 ch) at encoder levels 1-2 and stage 2
(stride 8, 256 ch) at level 3; stage 3 (stride 16, 512 ch) only feeds the 4-level model and stage 4 (stride 32,
1024 ch) is never used. The DINO block pools each stage to 64 tokens for cross-attention and mean-pools those for
FiLM, so both spatial and whole-image information matter:

  spatial   ridge probe from tokens, all stages bilinearly upsampled to 32x32, to the per-image standardised
            input / synthetic deformation / atmosphere (orig_img); held-out R^2
  global    ridge probe from mean-pooled features to per-image scalars: log deformation peak (mm), log atmosphere
            std (mm), sign of the dominant deformation lobe, and the deformation centroid (x, y); held-out R^2
  eff_rank  participation ratio of the token correlation matrix (channels z-scored, so a few very large
            ConvNeXt channels cannot fake a collapse); eff_rank_cov is the plain covariance version for reference

Runs on CPU:  python -m analysis.dino_stages --dataset campi_flegrei --n-images 512
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.dino import Dino
from analysis.dino_input import standardise, per_image_standardise
from util.datasets import get_datasets, OutputType, rad_to_mm
from util.parser import parse_args

FEATURE_SETS = {
    's1': [1], 's2': [2], 's3': [3], 's4': [4],
    's1+s2 (paper)': [1, 2],
    's1+s2+s3': [1, 2, 3],
    's1..s4': [1, 2, 3, 4],
    's3+s4': [3, 4],
}
LAMBDAS = [1e-3, 1e-2, 1e-1, 1, 10]


def to_mm(x):
    # network units -> rad -> mm (BringToInterval divides by 100)
    return rad_to_mm(x * 100.)


def ridge_fit(X, y, lam):
    A = X.T @ X
    A += lam * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=A.dtype)
    return torch.linalg.solve(A, X.T @ y)


def prep(X_tr, X_te):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    one = lambda X: torch.cat([(X - mu) / sd, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    return one(X_tr), one(X_te)


def r2(y, pred):
    return (1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()


def ridge_cv_r2(X_tr, y_tr, X_te, y_te, folds=5):
    """Pick lambda by k-fold CV on the training images, refit, report held-out R^2 per target column."""
    X_tr, X_te = X_tr.double(), X_te.double()
    y_tr, y_te = y_tr.double(), y_te.double()
    idx = torch.arange(len(X_tr))
    best, best_score = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        score = 0.
        for k in range(folds):
            va = idx % folds == k
            a, b = prep(X_tr[~va], X_tr[va])
            score += np.mean(r2(y_tr[va], b @ ridge_fit(a, y_tr[~va], lam)))
        if score > best_score:
            best, best_score = lam, score
    a, b = prep(X_tr, X_te)
    return r2(y_te, b @ ridge_fit(a, y_tr, best)), best


def spatial_probe(hidden, stages, targets, n_train, per_image=128, size=32, seed=0, chunk=64):
    """hidden: {stage: (N, C, h, w)}; targets: (N, T, size, size). Subsamples positions to keep the solve small."""
    N = targets.shape[0]
    g = torch.Generator().manual_seed(seed)
    pos = torch.stack([torch.randperm(size * size, generator=g)[:per_image] for _ in range(N)])
    tok = []
    for i in range(0, N, chunk):
        feats = torch.cat([F.interpolate(hidden[s][i:i + chunk], size=(size, size), mode='bilinear',
                                         align_corners=False) for s in stages], 1)
        p = pos[i:i + chunk, None, :].expand(-1, feats.shape[1], -1)
        tok.append(feats.flatten(2).gather(2, p).permute(0, 2, 1))
    tok = torch.cat(tok)
    C = tok.shape[-1]
    tgt = targets.flatten(2).gather(2, pos[:, None, :].expand(-1, targets.shape[1], -1)).permute(0, 2, 1)
    X_tr, X_te = tok[:n_train].reshape(-1, C), tok[n_train:].reshape(-1, C)
    y_tr, y_te = tgt[:n_train].reshape(-1, tgt.shape[-1]), tgt[n_train:].reshape(-1, tgt.shape[-1])
    # the spatial probe has plenty of samples; a fixed small lambda is enough
    a, b = prep(X_tr.double(), X_te.double())
    return r2(y_te.double(), b @ ridge_fit(a, y_tr.double(), 1e-2))


def eff_rank(feats, correlation):
    tok = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]).double()
    tok = tok - tok.mean(0)
    if correlation:
        tok = tok / (tok.std(0) + 1e-8)
    eig = torch.linalg.eigvalsh(tok.T @ tok / len(tok)).clamp_min(0)
    return (eig.sum() ** 2 / (eig ** 2).sum()).item()


def global_targets(deform, orig, mask):
    N, _, H, W = deform.shape
    d_mm, a_mm = to_mm(deform), to_mm(orig)
    valid = mask > 0.5
    peak = d_mm.abs().flatten(1).max(1).values
    atm_std = torch.stack([a_mm[i][valid[i]].std() if valid[i].sum() > 10 else a_mm[i].std() for i in range(N)])
    dmax, dmin = d_mm.flatten(1).max(1).values, d_mm.flatten(1).min(1).values
    sign = torch.where(dmax.abs() >= dmin.abs(), 1., -1.)
    w = d_mm.abs()[:, 0] + 1e-8
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    cx = (w * xx).sum((1, 2)) / w.sum((1, 2))
    cy = (w * yy).sum((1, 2)) / w.sum((1, 2))
    names = ['log_deform_peak', 'log_atm_std', 'deform_sign', 'deform_cx', 'deform_cy']
    return names, torch.stack([torch.log1p(peak), torch.log(atm_std + 1e-3), sign, cx, cy], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='campi_flegrei')
    parser.add_argument('--n-images', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', default='results/analysis/wp_dino_stages.csv')
    cli = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    args = parse_args(['--dataset', cli.dataset, '--batch-size', str(cli.batch_size),
                       '--num-workers', str(cli.num_workers), '--image-size', '128', '--no-cuda'])
    loader = get_datasets(args, resolution_mode='adaptive')['train']

    keys = [OutputType.IMAGE, OutputType.DEFORMATION, OutputType.ORIG_IMG, OutputType.WATER_MASK]
    data = {k: [] for k in keys}
    for batch in loader:
        for k in keys:
            data[k].append(batch[k])
        if sum(len(x) for x in data[OutputType.IMAGE]) >= cli.n_images:
            break
    image, deform, orig, mask = [torch.cat(data[k])[:cli.n_images].float() for k in keys]
    N = len(image)
    n_train = int(0.7 * N)
    print(f'{N} images from {cli.dataset} ({n_train} probe-train / {N - n_train} held-out)')

    dino = Dino()
    x = standardise(image, mask).clamp(-3, 3)
    with torch.no_grad():
        chunks = [dino(x[i:i + cli.batch_size]) for i in range(0, N, cli.batch_size)]
    hidden = {s: torch.cat([c[s] for c in chunks]).float() for s in range(1, 5)}

    rows = []
    print(f"\n{'stage':6s} {'res':>4s} {'ch':>5s} {'eff_rank_corr':>14s} {'eff_rank_cov':>13s}")
    for s in range(1, 5):
        er_corr, er_cov = eff_rank(hidden[s], True), eff_rank(hidden[s], False)
        print(f'{s:<6d} {hidden[s].shape[-1]:4d} {hidden[s].shape[1]:5d} {er_corr:14.1f} {er_cov:13.1f}')
        rows.append({'dataset': cli.dataset, 'probe': 'eff_rank', 'features': f's{s}', 'target': 'corr', 'value': er_corr})
        rows.append({'dataset': cli.dataset, 'probe': 'eff_rank', 'features': f's{s}', 'target': 'cov', 'value': er_cov})

    spatial_names = ['image', 'deformation', 'atmosphere']
    spatial_t = torch.cat([per_image_standardise(F.adaptive_avg_pool2d(t, (32, 32))) for t in (image, deform, orig)], 1)
    global_names, global_t = global_targets(deform, orig, mask)
    pooled = {s: hidden[s].mean((2, 3)) for s in hidden}

    print(f"\nSpatial probe at 32x32, held-out R^2\n{'features':15s} " + ' '.join(f'{n:>12s}' for n in spatial_names))
    for name, stages in FEATURE_SETS.items():
        res = spatial_probe(hidden, stages, spatial_t, n_train)
        print(f'{name:15s} ' + ' '.join(f'{v:12.3f}' for v in res))
        rows += [{'dataset': cli.dataset, 'probe': 'spatial', 'features': name, 'target': t, 'value': v}
                 for t, v in zip(spatial_names, res)]

    print(f"\nGlobal probe from mean-pooled features, held-out R^2 (lambda by 5-fold CV)\n{'features':15s} "
          + ' '.join(f'{n:>16s}' for n in global_names))
    for name, stages in FEATURE_SETS.items():
        X = torch.cat([pooled[s] for s in stages], 1)
        res, lam = ridge_cv_r2(X[:n_train], global_t[:n_train], X[n_train:], global_t[n_train:])
        print(f'{name:15s} ' + ' '.join(f'{v:16.3f}' for v in res) + f'   (lambda {lam:g})')
        rows += [{'dataset': cli.dataset, 'probe': 'global', 'features': name, 'target': t, 'value': v}
                 for t, v in zip(global_names, res)]

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    write_header = not os.path.exists(cli.out)
    with open(cli.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f'\nappended to {cli.out}')


if __name__ == '__main__':
    main()
