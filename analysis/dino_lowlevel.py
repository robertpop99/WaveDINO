"""
Supplementary S9: what low-level structure do the DINOv3 stages used by WaveDino (stages 1 and 2) encode on
interferograms?

Same setup as analysis/dino_stages.py (training samples with synthetic deformation, input standardised per image and
clipped to +-3; 'current' = the paper model's unstandardised input), but the spatial ridge probe now targets
low-level descriptors of the input, all computed at 128x128 and average-pooled to 32x32:

  image       the (standardised) input itself
  lowpass     Gaussian low-pass of the input (sigma 8 px = 1/16 of the 128-px patch), i.e. broad smooth variations
  highpass    input minus lowpass, i.e. fine-scale texture
  grad_mag    Sobel gradient magnitude of the input (edges / phase gradients)
  lap_mag     |Laplacian| of the input smoothed with sigma 2 px (curvature, localised extrema)
  dem         elevation, and dem_grad its Sobel gradient magnitude (terrain texture)
  water       water mask (land cover)
  deformation the synthetic deformation contained in the input

Held-out R^2 (70/30 split by image). Runs on CPU:
  python -m analysis.dino_lowlevel --dataset campi_flegrei --n-images 512
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.dino import Dino
from analysis.dino_input import standardise, per_image_standardise
from analysis.dino_stages import spatial_probe
from util.datasets import get_datasets, OutputType
from util.parser import parse_args

FEATURE_SETS = {'s1': [1], 's2': [2], 's1+s2 (paper)': [1, 2], 's3': [3], 's4': [4]}


def gaussian_blur(x, sigma):
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def sobel_mag(x):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=x.dtype).view(1, 1, 3, 3) / 8
    xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx, gy = F.conv2d(xp, kx), F.conv2d(xp, kx.transpose(2, 3))
    return torch.sqrt(gx ** 2 + gy ** 2)


def laplacian_mag(x):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k).abs()


def targets(x, dem, water, deform):
    low = gaussian_blur(x, 8.)
    t = {
        'image': x,
        'lowpass': low,
        'highpass': x - low,
        'grad_mag': sobel_mag(x),
        'lap_mag': laplacian_mag(gaussian_blur(x, 2.)),
        'dem': dem,
        'dem_grad': sobel_mag(dem),
        'water': water,
        'deformation': deform,
    }
    names = list(t)
    return names, torch.cat([per_image_standardise(F.adaptive_avg_pool2d(t[n], (32, 32))) for n in names], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='campi_flegrei')
    parser.add_argument('--n-images', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', default='results/analysis/wp_dino_lowlevel.csv')
    cli = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    args = parse_args(['--dataset', cli.dataset, '--batch-size', str(cli.batch_size),
                       '--num-workers', str(cli.num_workers), '--image-size', '128', '--no-cuda'])
    loader = get_datasets(args, resolution_mode='adaptive')['train']

    keys = [OutputType.IMAGE, OutputType.WATER_MASK, OutputType.DEM, OutputType.DEFORMATION]
    data = {k: [] for k in keys}
    for batch in loader:
        for k in keys:
            data[k].append(batch[k])
        if sum(len(x) for x in data[OutputType.IMAGE]) >= cli.n_images:
            break
    image, mask, dem, deform = [torch.cat(data[k])[:cli.n_images].float() for k in keys]
    dem = dem[:, :1]
    N = len(image)
    n_train = int(0.7 * N)
    print(f'{N} images from {cli.dataset} ({n_train} probe-train / {N - n_train} held-out)')

    dino = Dino()
    rows = []
    for variant in ('standardised', 'current'):
        x = standardise(image, mask).clamp(-3, 3) if variant == 'standardised' else image
        with torch.no_grad():
            chunks = [dino(x[i:i + cli.batch_size]) for i in range(0, N, cli.batch_size)]
        hidden = {s: torch.cat([c[s] for c in chunks]).float() for s in range(1, 5)}
        # targets always describe the interferogram itself, independent of how it is fed to DINO
        names, tgt = targets(standardise(image, mask).clamp(-3, 3), dem, mask, deform)

        print(f"\n[{variant}] spatial probe at 32x32, held-out R^2\n{'features':15s} "
              + ' '.join(f'{n:>9s}' for n in names))
        for fname, stages in FEATURE_SETS.items():
            res = spatial_probe(hidden, stages, tgt, n_train)
            print(f'{fname:15s} ' + ' '.join(f'{v:9.3f}' for v in res))
            rows += [{'dataset': cli.dataset, 'input': variant, 'features': fname, 'target': t, 'r2': v}
                     for t, v in zip(names, res)]

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
