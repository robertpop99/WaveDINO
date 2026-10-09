"""Supplementary S7: build LiCSBAS GEOCml-style input dirs (one per correction variant) for the LdM 083D track.

Every variant uses the same interferogram pairs (the WaveDINO full-frame eval set restricted to 083D, all >= 20 days)
and the same NaN mask (union of raw and GACOS NaNs), so the LiCSBAS runs differ only in the phase values.
The metadata (EQA.dem_par, slc.mli.par, hgt, E/N/U, baselines) comes from an existing LiCSBAS clip of the frame
($TS_ROOT/Laguna_del_Maule_083D_CLIP, LiCSBAS steps 01-05), whose 501x501 grid shares the dataset tifs' upper-left
corner (checked: offset 0, same sign). Then run analysis/timeseries_licsbas.sh for each variant.

Denoised outputs live in the model frame: sign-flipped and divided by 100 (BringToInterval(-100, 100)),
so LiCSBAS unw = -100 * denoised.
"""
import argparse
import os
import re

import numpy as np
import tifffile as tiff

# LiCSBAS working directory (input clips and time-series outputs); set TS_ROOT to change it
TS_ROOT = os.path.expanduser(os.environ.get('TS_ROOT', 'timeseries'))
CLIP_META = os.path.join(TS_ROOT, 'Laguna_del_Maule_083D_CLIP')
META_FILES = ['EQA.dem_par', 'slc.mli.par', 'slc.mli', 'hgt', 'E.geo', 'N.geo', 'U.geo', 'baselines']
SIZE = 501

MODEL_DIR = 'results/laguna_del_male_eval/WaveDino-4layers_network_26_05_13_22_14_23_patches_full'
RAW_DIR = 'datasets/laguna_del_male/83D_unw'
GACOS_DIR = 'datasets/laguna_del_male/83D_gacos'
CC_DIR = 'datasets/laguna_del_male/cc_all'


def pair_of(f):
    m = re.search(r'(\d{8}_\d{8})', f)
    return m.group(1) if m else None


def pad(a, fill):
    out = np.full((SIZE, SIZE), fill, dtype=a.dtype)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def main(args):
    model_files = {pair_of(f): f for f in os.listdir(args.model_dir)}
    raw_pairs = {pair_of(f) for f in os.listdir(RAW_DIR)}
    pairs = sorted(set(model_files) & raw_pairs)
    print(f'{len(pairs)} 083D pairs in {args.model_dir}')

    out_dirs = {v: os.path.join(TS_ROOT, f'LdM_083D_{v}_CLIP') for v in args.variants}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
        for m in META_FILES:
            dst = os.path.join(d, m)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(CLIP_META, m), dst)

    for i, p in enumerate(pairs):
        raw = tiff.imread(os.path.join(RAW_DIR, f'laguna_del_maule_{p}.geo.unw.tif')).astype(np.float32)
        gacos = tiff.imread(os.path.join(GACOS_DIR, f'laguna_del_maule_{p}.geo.unw_GACOS.tif')).astype(np.float32)
        mask = ~np.isfinite(raw) | (raw == 0) | ~np.isfinite(gacos) | (gacos == 0)

        cc = np.nan_to_num(tiff.imread(os.path.join(CC_DIR, f'laguna_del_maule_{p}.geo.cc.tif')))
        cc = pad(np.floor(np.clip(cc, 0, 1) * 255).astype(np.uint8), 0)

        values = {
            'unw': raw,
            'gacos': gacos,
            'wavedino': -100 * np.squeeze(tiff.imread(os.path.join(args.model_dir, model_files[p]))).astype(np.float32),
        }
        for v, d in out_dirs.items():
            unw = values[v].copy()
            unw[mask] = np.nan
            ifg_dir = os.path.join(d, p)
            os.makedirs(ifg_dir, exist_ok=True)
            pad(unw, np.float32(np.nan)).tofile(os.path.join(ifg_dir, f'{p}.unw'))
            cc.tofile(os.path.join(ifg_dir, f'{p}.cc'))

        if i % 200 == 0:
            print(f'{i}/{len(pairs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=MODEL_DIR)
    parser.add_argument('--variants', nargs='+', default=['unw', 'gacos', 'wavedino'])
    main(parser.parse_args())
