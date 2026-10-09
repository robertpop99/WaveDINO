"""Supplementary S7 figure: cumulative LOS deformation at each LdM GNSS station — the daily GNSS curve (B-spline used by the GNSS
evaluation) with GNSS and the Unw / GACOS / WaveDINO LiCSBAS time series scattered at the SAR epochs.

Every series is zeroed at the station's first epoch that is valid in all three variants and in GNSS, so the gap at the
end of the record is the total-displacement error. GNSS scatter = ±3-day median of the daily series.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis.timeseries_eval import load_cum, load_gnss, sample, GNSS_DIR
from util.gps_data_loaders import LagunaDelMaleGPSData

SERIES = [  # fixed categorical order (dataviz reference palette slots 1-3) + marker as secondary encoding
    ('unw', 'LdM_083D_unw_TS', 'Unw', '#2a78d6', 'o'),
    ('gacos', 'LdM_083D_gacos_TS', 'GACOS', '#eb6834', 's'),
    ('wavedino', 'LdM_083D_wavedino_TS', 'WaveDINO', '#1baf7a', '^'),
]
GNSS_COLOR = '#3d3d3a'
TEXT_SECONDARY = '#5f5e5a'
GRID = '#e4e3dd'


def gnss_at(daily, d):
    w = daily[d - pd.Timedelta(days=3): d + pd.Timedelta(days=3)]
    return w.median() if len(w) else np.nan


def main(args):
    loader = LagunaDelMaleGPSData(GNSS_DIR)
    stations = loader.gps_data
    data = {k: load_cum(ts, args.file) for k, ts, *_ in SERIES}

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': TEXT_SECONDARY, 'axes.labelcolor': TEXT_SECONDARY,
                         'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY})
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.6 * len(stations)), sharex=True)

    for ax, (s, info) in zip(axes, stations.items()):
        daily = load_gnss(s)
        spline = pd.Series(info['data']).rename(index=lambda d: pd.Timestamp(d)).sort_index()
        ser = {k: pd.Series(sample(data[k][1], info['coords']), index=data[k][0]) for k, *_ in SERIES}
        masked = [label for k, _, label, *_ in SERIES if ser[k].notna().sum() < 2]  # station pixel masked (cum_filt)
        ser = {k: v for k, v in ser.items() if v.notna().sum() >= 2}
        common = sorted(set.intersection(*[set(v.dropna().index) for v in ser.values()]))
        common = [d for d in common if np.isfinite(gnss_at(daily, d))]
        d0, d1 = common[0], common[-1]
        g0 = gnss_at(daily, d0)

        curve = spline[d0:d1]
        ax.plot(curve.index, curve - spline.asof(d0), color=GNSS_COLOR, lw=2, zorder=3, label='GNSS (daily, spline)')
        g_pts = pd.Series([gnss_at(daily, d) for d in common], index=common) - g0
        ax.scatter(g_pts.index, g_pts, s=14, marker='D', facecolor='white', edgecolor=GNSS_COLOR, lw=0.8, zorder=4,
                   label='GNSS at SAR epochs')

        ends = []
        for k, _, label, color, marker in SERIES:
            if k not in ser:
                continue
            v = ser[k][d0:d1].dropna()
            v = v - v.iloc[0]
            ax.scatter(v.index, v, s=12, marker=marker, color=color, edgecolor='white', lw=0.4, alpha=0.9,
                       zorder=5, label=label)
            ends.append((label, v.iloc[-1], color))
        g_end = g_pts.iloc[-1]

        # direct end labels: total displacement over the period; text in text ink, the colored dot carries identity.
        # Labels are pushed apart vertically so they never overlap; the dot stays at the true value.
        labels = sorted([('GNSS', g_end, GNSS_COLOR)] + ends, key=lambda e: -e[1])
        lo, hi = ax.get_ylim()
        gap = 0.1 * (hi - lo)
        y_prev = np.inf
        for label, val, color in labels:
            y_lab = min(val, y_prev - gap)
            ax.plot([d1 + pd.Timedelta(days=30)], [val], marker='o', ms=4, color=color, clip_on=False)
            ax.annotate(f'{label} {val:.0f} mm', xy=(d1 + pd.Timedelta(days=30), val), xytext=(d1 + pd.Timedelta(days=90), y_lab),
                        va='center', fontsize=8, color=TEXT_SECONDARY, annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color=GRID, lw=0.6))
            y_prev = y_lab

        note = f'; {", ".join(masked)} masked at this pixel' if masked else ''
        ax.set_title(f'{s}  ({d0:%Y-%m-%d} to {d1:%Y-%m-%d}{note})', loc='left', fontsize=10, color='#1f1f1d')
        ax.set_ylabel('LOS displacement (mm)')
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    legend = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(l, h)
    order = ['GNSS (daily, spline)', 'GNSS at SAR epochs'] + [label for _, _, label, *_ in SERIES]
    labels = [l for l in order if l in legend]
    handles = [legend[l] for l in labels]
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'LdM 083D: cumulative LOS deformation, GNSS vs LiCSBAS time series ({args.file})',
                 y=1.025, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 0.98))
    out = os.path.join(args.out_dir, f'timeseries_cumulative_{args.file.split(".")[0]}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='cum.h5', choices=['cum.h5', 'cum_filt.h5'])
    parser.add_argument('--out-dir', default='results/analysis/timeseries')
    main(parser.parse_args())
