"""
Does standardising each interferogram before DINOv3 give richer features?

The trainers feed DINO the network input as-is: unwrapped phase divided by 100 (BringToInterval(-100, 100)),
so real signals only span about +-0.01..0.1 of the [-1, 1] range. This script pushes real training samples
through the actual pipeline (with synthetic deformation, as in training) and compares DINO features for
  current      the input exactly as the trainers pass it
  standardised (x - mean) / std per image, over non-water pixels
  std+clip3    standardised, then clipped to +-3

Metrics per DINO stage (the stages WaveDino uses are 1-3):
  spatial_frac   within-image spatial variance / total variance of the tokens (0 = features are flat in space)
  img_cos        mean cosine similarity between the spatially pooled features of different images (1 = DINO sees
                 every image as the same thing)
  eff_rank       participation ratio of the token covariance (how many feature directions are used)
  r2_image       held-out R^2 of a ridge probe from tokens to the (pooled, per-image standardised) input
  r2_deform      same, to the synthetic deformation, i.e. how linearly visible the target signal is

Runs on CPU:  python -m analysis.dino_input --dataset campi_flegrei --n-images 256
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.dino import Dino
from util.datasets import get_datasets, OutputType
from util.parser import parse_args


def standardise(image, mask):
    valid = mask > 0.5
    out = torch.empty_like(image)
    for i in range(image.shape[0]):
        v = image[i][valid[i]]
        if v.numel() < 10:
            v = image[i].flatten()
        out[i] = (image[i] - v.mean()) / (v.std() + 1e-8)
    return out


VARIANTS = {
    'current': lambda image, mask: image,
    'standardised': standardise,
    'std+clip3': lambda image, mask: standardise(image, mask).clamp(-3, 3),
}


def per_image_standardise(x):
    # x: (N, 1, h, w)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - m) / (s + 1e-8)


def ridge_r2(tokens, target, n_train, lam=1e-2):
    """tokens: (N, h*w, C), target: (N, h*w). Fit on the first n_train images, report R^2 on the rest."""
    X_tr = tokens[:n_train].reshape(-1, tokens.shape[-1]).double()
    X_te = tokens[n_train:].reshape(-1, tokens.shape[-1]).double()
    y_tr = target[:n_train].reshape(-1).double()
    y_te = target[n_train:].reshape(-1).double()

    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    X_tr = torch.cat([X_tr, torch.ones(len(X_tr), 1, dtype=X_tr.dtype)], 1)
    X_te = torch.cat([X_te, torch.ones(len(X_te), 1, dtype=X_te.dtype)], 1)

    A = X_tr.T @ X_tr
    A += lam * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=A.dtype)
    w = torch.linalg.solve(A, X_tr.T @ y_tr)
    pred = X_te @ w
    return 1 - ((y_te - pred) ** 2).sum().item() / ((y_te - y_te.mean()) ** 2).sum().item()


def stage_metrics(feats, image, deform, n_train):
    """feats: (N, C, h, w) for one stage; image/deform: (N, 1, H, W)."""
    N, C, h, w = feats.shape
    tokens = feats.permute(0, 2, 3, 1).reshape(N, h * w, C).float()

    total_var = tokens.reshape(-1, C).var(0).sum()
    spatial_var = tokens.var(1).mean(0).sum()

    pooled = F.normalize(tokens.mean(1), dim=1)
    cos = pooled @ pooled.T
    img_cos = (cos.sum() - cos.diagonal().sum()) / (N * (N - 1))

    flat = tokens.reshape(-1, C)
    flat = flat - flat.mean(0)
    eig = torch.linalg.eigvalsh((flat.T @ flat).double() / len(flat)).clamp_min(0)
    eff_rank = (eig.sum() ** 2 / (eig ** 2).sum()).item()

    img_t = per_image_standardise(F.adaptive_avg_pool2d(image, (h, w))).reshape(N, -1)
    def_t = per_image_standardise(F.adaptive_avg_pool2d(deform, (h, w))).reshape(N, -1)

    return {
        'spatial_frac': (spatial_var / total_var).item(),
        'img_cos': img_cos.item(),
        'eff_rank': eff_rank,
        'r2_image': ridge_r2(tokens, img_t, n_train),
        'r2_deform': ridge_r2(tokens, def_t, n_train),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='campi_flegrei')
    parser.add_argument('--n-images', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', default='results/analysis/wp_dino_input.csv')
    cli = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    args = parse_args(['--dataset', cli.dataset, '--batch-size', str(cli.batch_size),
                       '--num-workers', str(cli.num_workers), '--image-size', '128', '--no-cuda'])
    loader = get_datasets(args, resolution_mode='adaptive')['train']

    images, deforms, masks = [], [], []
    for batch in loader:
        images.append(batch[OutputType.IMAGE])
        deforms.append(batch[OutputType.DEFORMATION])
        masks.append(batch[OutputType.WATER_MASK])
        if sum(len(x) for x in images) >= cli.n_images:
            break
    image = torch.cat(images)[:cli.n_images].float()
    deform = torch.cat(deforms)[:cli.n_images].float()
    mask = torch.cat(masks)[:cli.n_images].float()
    n_train = int(0.7 * len(image))

    q = image.flatten().quantile(torch.tensor([0.01, 0.5, 0.99]))
    print(f'{len(image)} images from {cli.dataset}; input quantiles 1/50/99 % = '
          f'{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f} (network units, 1 mm = {2 * np.pi / 28 / 100:.4f})')

    dino = Dino()
    rows = []
    for name, fn in VARIANTS.items():
        x = fn(image, mask)
        with torch.no_grad():
            hidden = [dino(x[i:i + cli.batch_size]) for i in range(0, len(x), cli.batch_size)]
        for stage in range(1, 5):
            feats = torch.cat([h[stage] for h in hidden])
            m = stage_metrics(feats, image, deform, n_train)
            rows.append({'dataset': cli.dataset, 'variant': name, 'stage': stage,
                         'resolution': feats.shape[-1], **m})

    keys = ['spatial_frac', 'img_cos', 'eff_rank', 'r2_image', 'r2_deform']
    print(f"\n{'variant':13s} {'stage':>5s} " + ' '.join(f'{k:>12s}' for k in keys))
    for r in rows:
        print(f"{r['variant']:13s} {r['stage']:5d} " + ' '.join(f'{r[k]:12.3f}' for k in keys))

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
