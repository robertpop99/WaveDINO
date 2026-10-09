"""
Cross-site tables (Supplementary S8): each site's WaveDINO evaluated on the other site, against the in-domain model.

GNSS: the resize outputs of the in-domain and cross-site runs plus the Unw baseline (results/gnss_stats, see
analysis/gnss_stats.py --rebuild-baselines), binned on |GNSS|, on each method's own pairs and on the common pairs.
Synthetic: the clean synthetic test of each model on the target site (in-domain = the clean `unwrap` variant,
cross-site = `atm`; same samples and seed).

Inputs, for a model trained on SOURCE and evaluated on TARGET (outputs are named after the source run):
    results/{TARGET}_eval/{SOURCE run}_resize_data.pkl        main.py --evaluate --dataset TARGET --model-name <SOURCE run>
    results/{TARGET}_synth_analysis/{SOURCE run}_atm.csv     python -m analysis.synth --analysis atm ...
    results/{TARGET}_synth_analysis/{TARGET run}_unwrap.csv  python -m analysis.synth --analysis unwrap ...

    python -m analysis.cross_site_tables [--runs campi_flegrei=<run folder> laguna_del_male=<run folder>]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from analysis.gnss_stats import load_pairs, assign_bins, method_table

PAPER_RUNS = {'campi_flegrei': 'WaveDino-4layers_network_26_05_13_22_14_07',
              'laguna_del_male': 'WaveDino-4layers_network_26_05_13_22_14_23'}
SHORT = {'campi_flegrei': 'CF', 'laguna_del_male': 'LdM'}


def default_run(results_dir, dataset):
    """The paper's run if it was trained here, else the newest WaveDino-4layers run of the dataset."""
    if os.path.isdir(os.path.join(results_dir, f'{dataset}_128', PAPER_RUNS[dataset])):
        return PAPER_RUNS[dataset]
    runs = sorted(glob.glob(os.path.join(results_dir, f'{dataset}_128', 'WaveDino-4layers_network_*')))
    return os.path.basename(runs[-1]) if runs else PAPER_RUNS[dataset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir', default='results/analysis')
    parser.add_argument('--runs', nargs='+', default=[], metavar='DATASET=FOLDER',
                        help='WaveDINO run folder per site (default: the paper run if present, else the newest)')
    args = parser.parse_args()
    R = args.results_dir
    runs = {d: default_run(R, d) for d in PAPER_RUNS}
    runs.update(dict(r.split('=', 1) for r in args.runs))

    gnss_tables, synth_rows = [], []
    for target, own in runs.items():
        source = [d for d in runs if d != target][0]
        other = runs[source]
        src = {'Unw': f'{R}/gnss_stats/{target}_Unw_data.pkl',
               'WaveDINO in-domain': f'{R}/{target}_eval/{own}_resize_data.pkl',
               f'WaveDINO from {SHORT[source]}': f'{R}/{target}_eval/{other}_resize_data.pkl'}
        loaded = {}
        for m, p in src.items():
            df = load_pairs(p)
            loaded[m] = df[np.isfinite(df.insar) & np.isfinite(df.gnss)]
        for max_days in (None, 60):
            pairs = {m: assign_bins(d if max_days is None else d[d.time_interval <= max_days].copy(), abs_bins=True)
                     for m, d in loaded.items()}
            common = set.intersection(*[set(zip(d.name, d.station)) for d in pairs.values()])
            cp = {m: d[[k in common for k in zip(d.name, d.station)]] for m, d in pairs.items()}
            t = pd.concat([method_table(pairs, 'own'), method_table(cp, 'common')], ignore_index=True)
            t.insert(0, 'baselines', 'all' if max_days is None else f'<={max_days}d')
            t.insert(0, 'dataset', target)
            gnss_tables.append(t)

        # synthetic clean test: in-domain = unwrap area 0 (identical to atm), cross-site = atm; same samples and seed
        sd = f'{R}/{target}_synth_analysis'
        ind = pd.read_csv(f'{sd}/{own}_unwrap.csv').query('area == 0')
        cross = pd.read_csv(f'{sd}/{other}_atm.csv')
        assert list(ind['name']) == list(cross['name'])
        for label, df in [('WaveDINO in-domain', ind), (f'WaveDINO from {SHORT[source]}', cross)]:
            synth_rows.append({'dataset': target, 'method': label, 'n': len(df), 'mse': df.mse.mean(),
                               'median_mse': df.mse.median(), 'mae': df.mae.mean(), 'mse_land': df.mse_land.mean(),
                               'slope': df.slope.median(), 'bias': df.bias.mean()})

    os.makedirs(args.out_dir, exist_ok=True)
    pd.concat(gnss_tables, ignore_index=True).to_csv(f'{args.out_dir}/cross_site_gnss_abs_bins.csv', index=False)
    synth = pd.DataFrame(synth_rows)
    synth.round(3).to_csv(f'{args.out_dir}/cross_site_synth.csv', index=False)
    print(synth.round(2).to_string())


if __name__ == '__main__':
    main()
