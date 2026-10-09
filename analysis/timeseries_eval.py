"""Supplementary S7: compare LiCSBAS time series (per correction variant) against GNSS at the LdM stations.

Inputs: $TS_ROOT/LdM_083D_{variant}_TS/{cum,cum_filt}.h5 (see analysis/timeseries_prep.py and
analysis/timeseries_licsbas.sh) and the daily GNSS LOS series in datasets/laguna_del_male/GPS_Data_csv.

Per station and variant:
  - InSAR is sampled at the station pixel, re-referenced to pixel (213, 460), and aligned to GNSS by removing the
    mean InSAR-GNSS offset over the common epochs (the InSAR datum is arbitrary).
  - RMS misfit and Pearson r on the aligned series, plus r after removing a linear trend from both (the LdM uplift
    trend dominates r otherwise).
  - Magnitude-squared coherence and Welch power spectra on both series linearly resampled to a 12-day grid and
    linearly detrended.
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

from util.gps_data_loaders import LagunaDelMaleGPSData

# LiCSBAS working directory (input clips and time-series outputs); set TS_ROOT to change it
TS_ROOT = os.path.expanduser(os.environ.get('TS_ROOT', 'timeseries'))
GNSS_DIR = 'datasets/laguna_del_male/GPS_Data_csv'
REF = (213, 460)
STEP_DAYS = 12
FS = 365.25 / STEP_DAYS  # samples per year
BANDS = {'all': (0, FS / 2), 'annual': (0.7, 1.5), 'sub-annual': (1.5, FS / 2)}

VARIANTS = {
    'unw': ('LdM_083D_unw_TS', 'Unw'),
    'gacos': ('LdM_083D_gacos_TS', 'GACOS'),
    'wavedino': ('LdM_083D_wavedino_TS', 'WaveDINO'),
    'licsbas_all': ('Laguna_del_Maule_083D_TS', 'LiCSBAS (all pairs, old run)'),
}


def load_cum(ts_dir, fname):
    path = os.path.join(TS_ROOT, ts_dir, fname)
    if not os.path.exists(path):
        return None
    with h5py.File(path, 'r') as h:
        dates = pd.to_datetime(h['imdates'][()].astype(str), format='%Y%m%d')
        cum = h['cum'][()]
    return dates, cum - cum[:, REF[0]:REF[0] + 1, REF[1]:REF[1] + 1]


def load_gnss(station):
    df = pd.read_csv(os.path.join(GNSS_DIR, f'export_{station}_los.csv'))
    return pd.Series(df['0'].to_numpy() * 10, index=pd.to_datetime(df['fecha']))  # cm -> mm


def sample(cum, rc):
    v = cum[:, rc[0], rc[1]]
    if np.isnan(v).all():  # masked station pixel: fall back to the 3x3 neighbourhood
        v = np.nanmean(cum[:, rc[0] - 1:rc[0] + 2, rc[1] - 1:rc[1] + 2], axis=(1, 2))
    return v


def resample(t_days, y, grid):
    return np.interp(grid, t_days, y)


def band_mean(f, x, band):
    m = (f > band[0]) & (f <= band[1])
    return float(np.mean(x[m])) if m.any() else np.nan


def compare(dates, insar, gnss):
    g = gnss.reindex(dates)
    ok = np.isfinite(insar) & np.isfinite(g.to_numpy())
    if ok.sum() < 20:
        return None
    d, x, y = dates[ok], insar[ok], g.to_numpy()[ok]
    x = x - np.mean(x - y)

    t = (d - d[0]).days.to_numpy().astype(float)
    detr = lambda a: a - np.polyval(np.polyfit(t, a, 1), t)

    grid = np.arange(0, t[-1] + 1, STEP_DAYS)
    xr, yr = signal.detrend(resample(t, x, grid)), signal.detrend(resample(t, y, grid))
    nperseg = min(64, len(grid) // 2)  # ~2.1 yr segments
    f, coh = signal.coherence(xr, yr, fs=FS, nperseg=nperseg)
    _, px = signal.welch(xr, fs=FS, nperseg=nperseg)
    _, py = signal.welch(yr, fs=FS, nperseg=nperseg)

    res = {
        'n_epochs': int(ok.sum()),
        'rms_mm': float(np.sqrt(np.mean((x - y) ** 2))),
        'r': float(np.corrcoef(x, y)[0, 1]),
        'r_detrended': float(np.corrcoef(detr(x), detr(y))[0, 1]),
        'rms_detrended_mm': float(np.sqrt(np.mean((detr(x) - detr(y)) ** 2))),
        'rate_ratio': float(np.polyfit(t, x, 1)[0] / np.polyfit(t, y, 1)[0]),  # InSAR / GNSS LOS velocity
        'gnss_rate_mm_yr': float(np.polyfit(t, y, 1)[0] * 365.25),
    }
    for name, band in BANDS.items():
        res[f'coh_{name}'] = band_mean(f, coh, band)
        res[f'power_ratio_{name}'] = band_mean(f, px, band) / band_mean(f, py, band)
    return res, (d, x, y), (f, coh, px, py)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    stations = LagunaDelMaleGPSData(GNSS_DIR).gps_data
    gnss = {s: load_gnss(s) for s in stations}

    rows, series, spectra = [], {}, {}
    for key, (ts_dir, label) in VARIANTS.items():
        for fname in ['cum.h5', 'cum_filt.h5']:
            loaded = load_cum(ts_dir, fname)
            if loaded is None:
                print(f'missing {ts_dir}/{fname}')
                continue
            dates, cum = loaded
            for s, info in stations.items():
                out = compare(dates, sample(cum, info['coords']), gnss[s])
                if out is None:
                    continue
                res, series[(key, fname, s)], spectra[(key, fname, s)] = out
                rows.append({'variant': label, 'file': fname, 'station': s, **res})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'timeseries_gnss_metrics.csv'), index=False)
    cols = ['n_epochs', 'rate_ratio', 'rms_mm', 'r', 'r_detrended', 'rms_detrended_mm', 'coh_all', 'coh_annual',
            'coh_sub-annual', 'power_ratio_annual', 'power_ratio_sub-annual']
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.precision', 3)
    for fname in ['cum.h5', 'cum_filt.h5']:
        sub = df[df['file'] == fname]
        print(f'\n===== {fname}: per station =====')
        print(sub.set_index(['station', 'variant'])[cols].sort_index())
        print(f'\n===== {fname}: mean over stations =====')
        print(sub.groupby('variant', sort=False)[cols].mean())
        common = set.intersection(*[set(g['station']) for _, g in sub.groupby('variant')])
        print(f'\n===== {fname}: mean over stations present in every variant ({sorted(common)}) =====')
        print(sub[sub['station'].isin(common)].groupby('variant', sort=False)[cols].mean())

    for fname in ['cum.h5', 'cum_filt.h5']:
        tag = fname.split('.')[0]
        fig, axes = plt.subplots(len(stations), 2, figsize=(16, 3.2 * len(stations)),
                                 gridspec_kw={'width_ratios': [3, 1]})
        for i, s in enumerate(stations):
            ax, axs = axes[i]
            gnss_plotted = False
            for key, (_, label) in VARIANTS.items():
                if (key, fname, s) not in series:
                    continue
                d, x, y = series[(key, fname, s)]
                if not gnss_plotted:
                    g = gnss[s][d[0]:d[-1]]
                    ax.plot(g.index, g - g.reindex(d).mean() + y.mean(), c='0.6', lw=0.8, label='GNSS (daily)')
                    gnss_plotted = True
                ax.plot(d, x, '.-', ms=3, lw=0.6, label=label)
                f, coh, px, py = spectra[(key, fname, s)]
                axs.semilogy(f, px, label=label)
            keys = [k for k in VARIANTS if (k, fname, s) in spectra]
            if keys:
                f, _, _, py = spectra[(keys[0], fname, s)]
                axs.semilogy(f, py, c='k', ls='--', label='GNSS')
            ax.set_title(f'{s} ({fname})')
            ax.set_ylabel('LOS (mm)')
            axs.set_xlabel('frequency (cycles / yr)')
            axs.set_ylabel('PSD, detrended')
            if i == 0:
                ax.legend(fontsize=8)
                axs.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'timeseries_gnss_{tag}.png'), dpi=130)
        plt.close(fig)
    print(f'\nWrote {args.out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='results/analysis/timeseries')
    main(parser.parse_args())
