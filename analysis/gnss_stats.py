"""
GNSS comparison statistics (Table II real-data rows, Supplementary S5). Reads the `*_data.pkl` files written by
util/test.py:get_model_stats (one per method) and reports, per displacement bin:

    n               number of (interferogram, station) pairs
    mse, median_se  mean / median squared InSAR - GNSS difference (mm^2); mse reproduces Table II
    mae, median_ae  mean / median absolute difference (mm)
    bias            mean(InSAR - GNSS) (mm); negative = the correction removes real deformation
    slope, intercept, resid_std, r
                    OLS fit InSAR = slope * GNSS + intercept, its residual std (mm) and Pearson r
    improvement     1 - mse / mse(Unw) on the same pairs (%)

Two pair sets are reported:
    own     each method on its own pairs (what evaluate_model / Table II does)
    common  only pairs present, and finite, for every method, so all rows are compared on the same data

Bins follow evaluate_model: (lo, hi) on the *signed* GNSS displacement, so pairs with negative GNSS displacement
fall outside every bin. --abs-bins bins |GNSS| instead. The 'Mean' column is the unweighted mean of the four bin
values, as in Table II; 'All' pools every pair in the bins.

Unw and GACOS: --rebuild-baselines recomputes their pickles on exactly the interferograms in the WaveDINO output
folder (symlinked into results/gnss_stats/{dataset}_{Unw,GACOS}/, evaluated with the same get_model_stats call
as evaluate_model) and writes results/gnss_stats/{dataset}_{Unw,GACOS}_data.pkl. Run it once before the first table.
Model rows: each method's resize output pickle in --eval-dir ({run}_resize_data.pkl, written by main.py --evaluate).
The run is the paper's folder if present, else the newest '{name}_network_*' folder; --runs METHOD=FOLDER overrides.
--with-patches adds each model's full-frame sliding-window output ({run}_patches_full_data.pkl) as an extra row
'<method> patches' (Supplementary S5); --methods restricts the rows. Missing pickles are skipped with a message.
--max-days N keeps only interferograms with a temporal baseline <= N days (evaluate_model keeps everything >= 20 days).
--reevaluate METHOD [...] re-scores a model's existing output folder with the current get_model_stats (same call
as evaluate_model), writing results/gnss_stats/{dataset}_{METHOD}_data.pkl; the original pickle is not touched.

Example:
    python -m analysis.gnss_stats --dataset campi_flegrei --rebuild-baselines
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from util.datasets import AddCoherence
from util.gps_data_loaders import get_gps_data_loader
from util.test import Stats, get_model_stats
from util.test_data_loader import ClassicDataLoader

BINS = [(0, 10), (10, 20), (20, 30), (30, np.inf)]
BIN_NAMES = ['0-10', '10-20', '20-30', '>30']

# folders holding the uncorrected and GACOS-corrected interferograms, and their filename suffixes
BASELINE_SOURCES = {
    'Unw': ('train', '.geo.unw.tif'),
    'GACOS': ('gacos_all', '.geo.unw_GACOS.tif'),
}

DATASETS = ['laguna_del_male', 'campi_flegrei']

# Table II model rows: method -> trainer.name (the run folders are '{name}_network_{timestamp}')
MODEL_NAMES = {
    'Unet': 'ClassicUnet',
    'Diffusion': 'Diffusion',
    'SwinIR': 'SwinIR',
    'Restormer': 'Restormer',
    'WaveMamba': 'WaveMamba',
    'WaveDINO': 'WaveDino-4layers',
}

# the runs behind the paper's Table II, used when their outputs are in --eval-dir
PAPER_RUNS = {
    'laguna_del_male': {
        'Unet': 'ClassicUnet_network_26_03_23_20_54_48',
        'Diffusion': 'Diffusion_network_26_03_23_20_54_29',
        'SwinIR': 'SwinIR_network_26_03_23_20_54_48',
        'Restormer': 'Restormer_network_26_03_23_20_54_29',
        'WaveMamba': 'WaveMamba_network_26_03_23_20_54_39',
        'WaveDINO': 'WaveDino-4layers_network_26_05_13_22_14_23',
    },
    'campi_flegrei': {
        'Unet': 'ClassicUnet_network_26_03_23_20_54_45',
        'Diffusion': 'Diffusion_network_26_03_23_20_54_48',
        'SwinIR': 'SwinIR_network_26_03_23_20_54_50',
        'Restormer': 'Restormer_network_26_03_23_20_54_52',
        'WaveMamba': 'WaveMamba_network_26_03_23_20_54_53',
        'WaveDINO': 'WaveDino-4layers_network_26_05_13_22_14_07',
    },
}


def model_run(dataset, eval_dir, method, overrides=None):
    """Run folder name of a method: --runs override, else the paper's run if present, else the newest run."""
    if overrides and method in overrides:
        return overrides[method]
    paper_run = PAPER_RUNS[dataset][method]
    if os.path.isfile(os.path.join(eval_dir, f'{paper_run}_resize_data.pkl')):
        return paper_run
    candidates = sorted(glob.glob(os.path.join(eval_dir, f'{MODEL_NAMES[method]}_network_*_resize_data.pkl')))
    if candidates:
        return os.path.basename(candidates[-1])[:-len('_resize_data.pkl')]
    return paper_run


def rebuild_baseline_pickles(dataset, eval_dir, out_dir, overrides=None):
    """Unw / GACOS pickles on exactly the interferograms of the WaveDINO output folder."""
    reference = f'{model_run(dataset, eval_dir, "WaveDINO", overrides)}_resize'
    names = sorted(f.split('.geo.')[0] for f in os.listdir(os.path.join(eval_dir, reference)))
    print(f'==> {len(names)} interferograms in {reference}')

    gps_data = get_gps_data_loader(dataset)
    coherence_extractor = AddCoherence(dataset)

    for method, (source, suffix) in BASELINE_SOURCES.items():
        folder = os.path.join(out_dir, f'{dataset}_{method}')
        os.makedirs(folder, exist_ok=True)
        missing = 0
        for name in names:
            target = os.path.abspath(os.path.join('datasets', dataset, source, name + suffix))
            link = os.path.join(folder, name + suffix)
            if not os.path.isfile(target):
                missing += 1
            elif not os.path.lexists(link):
                os.symlink(target, link)
        print(f'==> {method}: {len(os.listdir(folder))} interferograms linked from datasets/{dataset}/{source}, '
              f'{missing} missing', flush=True)

        # the pickle lands next to the folder
        get_model_stats(folder, ClassicDataLoader(folder), gps_data, lower_time_filter=20, time_filter=1000000,
                        gps_stations=None, norm_interf=True, denorm_interf=False,
                        coherence_extractor=coherence_extractor)


def reevaluate_model_pickles(dataset, eval_dir, out_dir, methods, overrides=None):
    """Re-scores existing model outputs; the folder is symlinked so the pickle is written to out_dir."""
    gps_data = get_gps_data_loader(dataset)
    coherence_extractor = AddCoherence(dataset)
    data_loader = ClassicDataLoader(f'datasets/{dataset}/train/', max_number_files=5000)  # as run_on_test_data

    for method in methods:
        source = os.path.join(eval_dir, f'{model_run(dataset, eval_dir, method, overrides)}_resize')
        folder = os.path.join(out_dir, f'{dataset}_{method}')
        if not os.path.lexists(folder):
            os.symlink(os.path.abspath(source), folder)
        print(f'==> Re-evaluating {method}: {source}', flush=True)
        get_model_stats(folder, data_loader, gps_data, lower_time_filter=20, time_filter=1000000, gps_stations=None,
                        norm_interf=False, denorm_interf=True, coherence_extractor=coherence_extractor)


def load_pairs(path):
    """One row per (interferogram, station) with a model value, as used by evaluate_model."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    stations = data[Stats.args]['gps_stations']
    rows = []
    for img_stats in data[Stats.all_data].values():
        for station in stations:
            if f'{station}_model_diff' in img_stats and f'{station}_real' in img_stats:
                rows.append({
                    'name': img_stats['name'],
                    'station': station,
                    'gnss': float(img_stats[f'{station}_real']),
                    'insar': float(img_stats[f'{station}_model_diff']),
                    'time_interval': img_stats['time_interval'],
                })
    return pd.DataFrame(rows)


def assign_bins(df, abs_bins):
    value = df['gnss'].abs() if abs_bins else df['gnss']
    df['bin'] = None
    for (lo, hi), name in zip(BINS, BIN_NAMES):
        # strict inequalities, as in DistanceQualityMetrics
        df.loc[(value > lo) & (value < hi), 'bin'] = name
    return df


def bin_stats(df):
    err = df['insar'] - df['gnss']
    stats = {
        'n': len(df),
        'mse': (err ** 2).mean(),
        'median_se': (err ** 2).median(),
        'mae': err.abs().mean(),
        'median_ae': err.abs().median(),
        'bias': err.mean(),
    }
    if len(df) > 2 and df['gnss'].std() > 0:
        slope, intercept = np.polyfit(df['gnss'], df['insar'], 1)
        stats.update({
            'slope': slope,
            'intercept': intercept,
            'resid_std': (df['insar'] - (slope * df['gnss'] + intercept)).std(),
            'r': np.corrcoef(df['gnss'], df['insar'])[0, 1],
        })
    return stats


def method_table(pairs, pair_set):
    rows = []
    for method, df in pairs.items():
        df = df[df['bin'].notna()]
        for name in BIN_NAMES + ['All']:
            sub = df if name == 'All' else df[df['bin'] == name]
            if len(sub) == 0:
                continue
            rows.append({'pairs': pair_set, 'method': method, 'bin': name, **bin_stats(sub)})
    table = pd.DataFrame(rows)

    # 'Mean' as in Table II: unweighted mean of the four bin values
    means = table[table['bin'].isin(BIN_NAMES)].groupby(['pairs', 'method'], sort=False)[
        ['mse', 'median_se', 'mae', 'median_ae', 'bias']].mean().reset_index()
    means['bin'] = 'Mean'
    table = pd.concat([table, means], ignore_index=True)

    if 'Unw' in pairs:
        unw = table[table['method'] == 'Unw'].set_index('bin')['mse']
        table['improvement'] = 100 * (1 - table['mse'] / table['bin'].map(unw))
    return table


def print_table(table, value, pair_set, fmt='{:.1f}'):
    sub = table[table['pairs'] == pair_set]
    wide = sub.pivot(index='method', columns='bin', values=value)
    columns = [c for c in BIN_NAMES + ['Mean', 'All'] if c in wide.columns]
    wide = wide.loc[sub['method'].unique(), columns]
    print(f'\n--- {value} ({pair_set} pairs)')
    print(wide.map(lambda x: fmt.format(x) if pd.notna(x) else '-').to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--eval-dir', default=None, help='default: results/{dataset}_eval')
    parser.add_argument('--abs-bins', action='store_true', help='bin on |GNSS| instead of signed GNSS')
    parser.add_argument('--out-dir', default='results/gnss_stats')
    parser.add_argument('--rebuild-baselines', action='store_true',
                        help='recompute the Unw / GACOS pickles on the WaveDINO interferograms first')
    parser.add_argument('--runs', nargs='+', default=[], metavar='METHOD=FOLDER',
                        help='run folder per method (default: the paper run if present, else the newest)')
    parser.add_argument('--methods', nargs='+', default=None, help='only these rows (default: all)')
    parser.add_argument('--with-patches', action='store_true',
                        help="add '<method> patches' rows from the {run}_patches_full outputs")
    parser.add_argument('--max-days', type=int, default=None,
                        help='only interferograms with a temporal baseline <= this many days')
    parser.add_argument('--reevaluate', nargs='+', default=[], metavar='METHOD',
                        help='re-score these models from their output folders into --out-dir first')
    args = parser.parse_args()

    eval_dir = os.path.expanduser(args.eval_dir or f'results/{args.dataset}_eval')
    overrides = dict(r.split('=', 1) for r in args.runs)

    if args.rebuild_baselines:
        rebuild_baseline_pickles(args.dataset, eval_dir, args.out_dir, overrides)
    if args.reevaluate:
        reevaluate_model_pickles(args.dataset, eval_dir, args.out_dir, args.reevaluate, overrides)

    sources = {}
    for method in list(BASELINE_SOURCES) + list(MODEL_NAMES):
        if args.methods is not None and method not in args.methods:
            continue
        rebuilt = os.path.join(args.out_dir, f'{args.dataset}_{method}_data.pkl')
        if os.path.isfile(rebuilt) or method in BASELINE_SOURCES:
            path = rebuilt
        else:
            path = os.path.join(eval_dir, f'{model_run(args.dataset, eval_dir, method, overrides)}_resize_data.pkl')
        sources[method] = path
        if args.with_patches and path.endswith('_resize_data.pkl'):
            sources[f'{method} patches'] = path.replace('_resize_data.pkl', '_patches_full_data.pkl')

    pairs = {}
    for method, path in sources.items():
        if not os.path.isfile(path):
            print(f'{method:18s} SKIPPED, missing {path}')
            continue
        print(f'{method:18s} {path}')
        df = load_pairs(path)
        df = df[np.isfinite(df['insar']) & np.isfinite(df['gnss'])]
        if args.max_days is not None:
            df = df[df['time_interval'] <= args.max_days]
        pairs[method] = assign_bins(df, args.abs_bins)

    binning = '|GNSS|' if args.abs_bins else 'signed GNSS'
    max_days = f', baselines <= {args.max_days} days' if args.max_days is not None else ''
    print(f'==> {args.dataset}, bins on {binning}{max_days}')
    for method, df in pairs.items():
        print(f'{method:10s} pairs {len(df):6d} | in a bin {df["bin"].notna().sum():6d} | '
              f'GNSS < 0: {(df["gnss"] < 0).mean():5.1%}')

    keys = [set(zip(df['name'], df['station'])) for df in pairs.values()]
    common = set.intersection(*keys)
    common_pairs = {m: df[[k in common for k in zip(df['name'], df['station'])]] for m, df in pairs.items()}
    print(f'==> {len(common)} (interferogram, station) pairs are common to all methods')

    table = pd.concat([method_table(pairs, 'own'), method_table(common_pairs, 'common')], ignore_index=True)

    for pair_set in ['own', 'common']:
        print_table(table, 'n', pair_set, fmt='{:.0f}')
        print_table(table, 'mse', pair_set)
        print_table(table, 'median_se', pair_set)
        if pair_set == 'common':
            print_table(table, 'improvement', pair_set, fmt='{:+.0f}%')
            print_table(table, 'bias', pair_set, fmt='{:+.2f}')
            print_table(table, 'slope', pair_set, fmt='{:.2f}')
            print_table(table, 'resid_std', pair_set)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f'{"abs" if args.abs_bins else "signed"}_bins'
    suffix += f'_max{args.max_days}d' if args.max_days is not None else ''
    suffix += '_with_patches' if args.with_patches else ''
    suffix += '_' + '-'.join(args.methods) if args.methods is not None else ''
    out_file = os.path.join(args.out_dir, f'{args.dataset}_{suffix}.csv')
    table.to_csv(out_file, index=False)
    print(f'\n==> Saved {out_file}')


if __name__ == '__main__':
    main()
