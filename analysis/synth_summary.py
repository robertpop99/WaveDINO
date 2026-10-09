"""
Summary tables for the analysis/synth.py outputs (results/{dataset}_synth_analysis/*.csv), one CSV per analysis in
--out-dir. MSE / MAE are full-image values in mm^2 / mm, as in Table I.

    atm_quartiles.csv   S4  clean synthetic test split into quartiles of atmospheric std (per site)
    unwrapping.csv          MSE per unwrapping-error area / jump, and the increase over the clean case
    interpolation.csv   S6  MSE per NaN-filling method
    position.csv        S3  MSE against source distance from the frame centre (fraction of half-width)
    pure.csv            S2  pure deformation: slope, far-field error and ramp size per source depth
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

DATASETS = ['laguna_del_male', 'campi_flegrei']
NAMES = {'WaveDino-4layers': 'WaveDINO', 'WaveDino-ramp-4layers': 'WaveDINO no-ramp', 'ClassicUnet': 'Unet',
         'Diffusion': 'Diffusion', 'SwinIR': 'SwinIR', 'Restormer': 'Restormer', 'WaveMamba': 'WaveMamba'}


def method_name(run_folder):
    return NAMES.get(run_folder.split('_network_')[0], run_folder.split('_network_')[0])


def load(results_dir, analysis):
    frames = []
    for dataset in DATASETS:
        for path in sorted(glob.glob(os.path.join(results_dir, f'{dataset}_synth_analysis', f'*_{analysis}.csv'))):
            df = pd.read_csv(path)
            run_folder = os.path.basename(path)[:-len(f'_{analysis}.csv')]
            df['dataset'] = dataset
            df['method'] = method_name(run_folder)
            df['run'] = run_folder
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def atm_quartiles(unwrap):
    clean = unwrap[unwrap['area'] == 0].copy()
    # quartile edges per site, shared by all methods (same samples, same seed)
    clean['atm_quartile'] = clean.groupby('dataset')['atm_std'].transform(
        lambda x: pd.qcut(x, 4, labels=['Q1 (weakest)', 'Q2', 'Q3', 'Q4 (strongest)']))
    table = clean.groupby(['dataset', 'method', 'atm_quartile'], observed=True).agg(
        n=('mse', 'size'), atm_std_mm=('atm_std', 'median'), mse=('mse', 'mean'), median_mse=('mse', 'median'),
        mae=('mae', 'mean')).reset_index()
    return table


def unwrapping(unwrap):
    table = unwrap.groupby(['dataset', 'method', 'jump', 'area']).agg(
        n=('mse', 'size'), mse=('mse', 'mean'), mae=('mae', 'mean')).reset_index()
    clean = table[table['area'] == 0].set_index(['dataset', 'method'])['mse']
    table['mse_increase_pct'] = 100 * (table['mse'] / table.set_index(['dataset', 'method']).index.map(clean) - 1)
    return table


def interpolation(interp):
    return interp.groupby(['dataset', 'method', 'interpolation'], sort=False).agg(
        n=('mse', 'size'), mse=('mse', 'mean'), mae=('mae', 'mean'), mse_land=('mse_land', 'mean')).reset_index()


def position(grid):
    grid = grid.copy()
    # distance of the source from the frame centre, in units of the half-width (0 = centre, 0.8 = 10 % from edge)
    grid['centre_distance'] = 2 * np.sqrt((grid['pos_y'] - 0.5) ** 2 + (grid['pos_x'] - 0.5) ** 2)
    grid['distance_bin'] = pd.cut(grid['centre_distance'], [0, 0.3, 0.6, 0.9, 1.2],
                                  labels=['0-0.3 (centre)', '0.3-0.6', '0.6-0.9', '0.9-1.2 (corners)'])
    return grid.groupby(['dataset', 'method', 'distance_bin'], observed=True).agg(
        n=('mse', 'size'), mse=('mse', 'mean'), median_mse=('mse', 'median'), slope=('slope', 'median'),
        far_field_err=('far_field_err_mean', 'median')).reset_index()


def pure(pure_df):
    pure_df = pure_df.copy()
    pure_df['depth_bin'] = pd.cut(pure_df['depth_km'], [0, 3, 5, 9], labels=['<3 km', '3-5 km', '>5 km'])
    return pure_df.groupby(['dataset', 'method', 'depth_bin'], observed=True).agg(
        n=('mse', 'size'), mse=('mse', 'mean'), slope=('slope', 'median'), bias=('bias', 'median'),
        far_field_err=('far_field_err_mean', 'median'), ramp_max_mm=('ramp_max', 'median')).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir', default='results/analysis')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    unwrap = load(args.results_dir, 'unwrap')
    tables = {
        'atm_quartiles': atm_quartiles(unwrap),
        'unwrapping': unwrapping(unwrap),
        'interpolation': interpolation(load(args.results_dir, 'interp')),
        'position': position(load(args.results_dir, 'grid')),
        'pure': pure(load(args.results_dir, 'pure')),
    }
    for name, table in tables.items():
        path = os.path.join(args.out_dir, f'{name}.csv')
        table.round(3).to_csv(path, index=False)
        print(f'==> {path}: {len(table)} rows')


if __name__ == '__main__':
    main()
