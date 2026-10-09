"""
Supplementary S1: how much real deformation the short-duration (<= 18-day) training interferograms contain. For every
short pair and every GNSS station: the GNSS displacement between the two dates, as line-of-sight displacement (what the
interferogram measures; the loaders' *_los.csv series) and as vertical uplift (the U component of the same stations,
smoothed with the same spline_smooth parameters).

    python -m analysis.short_pair_gnss            # LOS and uplift per 6/12/18-day pairs, both sites
    python -m analysis.short_pair_gnss rate       # CF uplift from local linear rates of the daily series (to 2025)
    python -m analysis.short_pair_gnss los        # CF LOS for the pairs after the weekly series end (2023-10)

The 2025 CF variants read the daily INGV series datasets/campi_flegrei/gps_data_zenodo_2025/{STA}_2000_2025.txt.
"""
import datetime as dt
import os
import re

import numpy as np
import pandas as pd

from util.gps_data_loaders import (CampiFlegreiGPSData, LagunaDelMaleGPSData, spline_smooth, CF_SMOOTHING,
                                   LDM_SMOOTHING)

TRAIN = {'laguna_del_male': 'datasets/laguna_del_male/train_all', 'campi_flegrei': 'datasets/campi_flegrei/train_all'}


def short_pairs(directory):
    pairs = []
    for f in os.listdir(directory):
        m = re.findall(r'(\d{8})_(\d{8})', f)
        if not m or not f.endswith('.tif'):
            continue
        a, b = m[0]
        days = (dt.datetime.strptime(b, '%Y%m%d') - dt.datetime.strptime(a, '%Y%m%d')).days
        if days <= 18:
            pairs.append((a, b, days))
    return pairs


def ldm_vertical(station):
    df = pd.read_csv(f'datasets/laguna_del_male/GPS_Data_csv/export_{station}.csv')
    # 'fecha' is YYYY-MM-DD for most stations but DD/MM/YYYY for PUEL; unixtime is unambiguous
    dates = [dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime('%Y%m%d') for t in df['unixtime']]
    return spline_smooth(dates, np.asarray(df['dh']) * 10, *LDM_SMOOTHING)  # cm -> mm


def cf_vertical(station):
    # weekly INGV series: MJD N E U sN sE sU, metres
    arr = np.loadtxt(f'datasets/campi_flegrei/gps_data/{station}_weekly.txt')
    dates = [(dt.date(1858, 11, 17) + dt.timedelta(days=int(m))).strftime('%Y%m%d') for m in arr[:, 0]]
    return spline_smooth(dates, arr[:, 3] * 1000, *CF_SMOOTHING)


def main():
    for dataset, loader_cls, vert_fn in (('laguna_del_male', LagunaDelMaleGPSData, ldm_vertical),
                                         ('campi_flegrei', CampiFlegreiGPSData, cf_vertical)):
        gps = loader_cls()
        pairs = short_pairs(TRAIN[dataset])
        vert = {s: vert_fn(s) for s in gps.stations()}
        rows = []
        for a, b, days in pairs:
            for s in gps.stations():
                la, lb = gps.value(s, a), gps.value(s, b)
                va, vb = vert[s].get(a), vert[s].get(b)
                if None in (la, lb, va, vb):
                    continue
                rows.append((days, s, abs(lb - la), abs(vb - va)))
        df = pd.DataFrame(rows, columns=['days', 'station', 'los', 'up'])
        print(f'{dataset} files {len(pairs)}  (pairs with both LOS and vertical GNSS: {len(df)})')
        for d in (6, 12, 18):
            x = df[df.days == d]
            print(f'  {d:2d} d  n={len(x):6d}  LOS median={x.los.median():4.1f} p95={x.los.quantile(.95):4.1f} '
                  f'max={x.los.max():4.1f} | UP median={x.up.median():4.1f} p95={x.up.quantile(.95):4.1f} '
                  f'max={x.up.max():5.1f} mm')
        print(f'  all   LOS median={df.los.median():.1f} max={df.los.max():.1f} | '
              f'UP median={df.up.median():.1f} p95={df.up.quantile(.95):.1f} max={df.up.max():.1f} mm')
        st = df[df.days == 18].groupby('station')[['los', 'up']].median().sort_values('up', ascending=False)
        print('  fastest stations, 18-day medians (LOS, UP):',
              [(s, round(r.los, 1), round(r.up, 1)) for s, r in st.head(4).iterrows()])


def cf_vertical_2025():
    """CF training pairs run to 2025-09 but the weekly series end in 2023-10; the daily INGV Zenodo series run to 2025.
    Vertical over ALL short CF pairs from the daily series (same 19 stations), and separately for pairs after 2023-10-18."""
    gps = CampiFlegreiGPSData()
    pairs = short_pairs(TRAIN['campi_flegrei'])
    vert = {}
    for s in gps.stations():
        arr = np.loadtxt(f'datasets/campi_flegrei/gps_data_zenodo_2025/{s}_2000_2025.txt', ndmin=2)
        dates = [(dt.date(1858, 11, 17) + dt.timedelta(days=int(m))).strftime('%Y%m%d') for m in arr[:, 0]]
        vert[s] = spline_smooth(dates, arr[:, 3] * 1000, *CF_SMOOTHING)
    rows = [(days, b > '20231018', abs(vert[s][b] - vert[s][a]))
            for a, b, days in pairs for s in gps.stations() if a in vert[s] and b in vert[s]]
    df = pd.DataFrame(rows, columns=['days', 'late', 'up'])
    print(f'campi_flegrei, daily Zenodo series to 2025: {len(pairs)} files, '
          f'{sum(b > "20231018" for _, b, _ in pairs)} after 2023-10-18; station-pairs {len(df)}')
    for name, x in (('all pairs', df), ('after 2023-10-18', df[df.late])):
        for d in (6, 12, 18):
            y = x[x.days == d]
            print(f'  {name:17s} {d:2d} d n={len(y):6d} UP median={y.up.median():4.1f} p95={y.up.quantile(.95):4.1f} '
                  f'max={y.up.max():5.1f} mm')
        print(f'  {name:17s} all    UP median={x.up.median():.1f} p95={x.up.quantile(.95):.1f} max={x.up.max():.1f} mm')


def cf_vertical_rate_2025(half_window=45):
    """The daily Zenodo series is too noisy for the spline at 6-18-day scale (quiet-period 6-day medians of 1.5 mm vs
    0.2 mm weekly), so estimate the uplift over each pair as the local linear rate (least squares over +-half_window
    days around the pair centre) times the pair length. Checked on the pre-2023-10 pairs against the weekly spline."""
    gps = CampiFlegreiGPSData()
    pairs = short_pairs(TRAIN['campi_flegrei'])
    series = {}
    for s in gps.stations():
        arr = np.loadtxt(f'datasets/campi_flegrei/gps_data_zenodo_2025/{s}_2000_2025.txt', ndmin=2)
        series[s] = (arr[:, 0], arr[:, 3] * 1000)
    mjd = lambda d: (dt.datetime.strptime(d, '%Y%m%d').date() - dt.date(1858, 11, 17)).days
    rows = []
    for a, b, days in pairs:
        c = (mjd(a) + mjd(b)) / 2
        for s, (t, u) in series.items():
            m = np.abs(t - c) <= half_window
            if m.sum() < half_window:  # need at least half of the daily samples
                continue
            rate = np.polyfit(t[m], u[m], 1)[0]
            rows.append((days, b > '20231018', abs(rate) * days))
    df = pd.DataFrame(rows, columns=['days', 'late', 'up'])
    print(f'campi_flegrei, local linear rate (+-{half_window} d) from daily series: station-pairs {len(df)}')
    for name, x in (('up to 2023-10-18', df[~df.late]), ('after 2023-10-18', df[df.late])):
        for d in (6, 12, 18):
            y = x[x.days == d]
            print(f'  {name:17s} {d:2d} d n={len(y):6d} UP median={y.up.median():4.1f} p95={y.up.quantile(.95):4.1f} '
                  f'max={y.up.max():5.1f} mm')
        print(f'  {name:17s} all    UP median={x.up.median():.1f} p95={x.up.quantile(.95):.1f} max={x.up.max():.1f} mm')


def cf_los_rate_2025(half_window=45):
    """LOS for the post-2023-10 CF pairs: the per-station LOS coefficients (E, N, U) are recovered by regressing the
    loaders' *_los.csv on the weekly ENU series of the same station, then applied to the local daily-series rates."""
    gps = CampiFlegreiGPSData()
    pairs = short_pairs(TRAIN['campi_flegrei'])
    mjd = lambda d: (dt.datetime.strptime(d, '%Y%m%d').date() - dt.date(1858, 11, 17)).days
    rows = []
    for s in gps.stations():
        wk = np.loadtxt(f'datasets/campi_flegrei/gps_data/{s}_weekly.txt')  # MJD N E U
        los = pd.read_csv(f'datasets/campi_flegrei/gps_data/{s}_los.csv')
        los_mjd = np.array([mjd(str(d)) for d in los['date']])
        common, iw, il = np.intersect1d(wk[:, 0].astype(int), los_mjd, return_indices=True)
        A = np.c_[wk[iw, 2], wk[iw, 1], wk[iw, 3], np.ones(len(iw))]  # E N U const
        coef, res, *_ = np.linalg.lstsq(A, np.asarray(los['deform'])[il], rcond=None)
        fit_rms = np.sqrt(np.mean((A @ coef - np.asarray(los['deform'])[il]) ** 2)) * 1000
        arr = np.loadtxt(f'datasets/campi_flegrei/gps_data_zenodo_2025/{s}_2000_2025.txt', ndmin=2)
        t, enu = arr[:, 0], arr[:, [2, 1, 3]] * 1000
        if s == 'RITE':
            print(f'  {s}: LOS = {coef[0]:.3f} E + {coef[1]:.3f} N + {coef[2]:.3f} U (fit rms {fit_rms:.2f} mm)')
        for a, b, days in pairs:
            if b <= '20231018':
                continue
            c = (mjd(a) + mjd(b)) / 2
            m = np.abs(t - c) <= half_window
            if m.sum() < half_window:
                continue
            rate = np.array([np.polyfit(t[m], enu[m, k], 1)[0] for k in range(3)])
            rows.append((days, abs(rate @ coef[:3]) * days, abs(rate[2]) * days))
    df = pd.DataFrame(rows, columns=['days', 'los', 'up'])
    print(f'campi_flegrei pairs after 2023-10-18, local rates (+-{half_window} d): station-pairs {len(df)}')
    for d in (6, 12, 18):
        y = df[df.days == d]
        print(f'  {d:2d} d n={len(y):5d} LOS median={y.los.median():4.1f} p95={y.los.quantile(.95):4.1f} '
              f'max={y.los.max():5.1f} | UP median={y.up.median():4.1f} max={y.up.max():5.1f} mm')
    print(f'  all   LOS median={df.los.median():.1f} max={df.los.max():.1f} | UP median={df.up.median():.1f} '
          f'max={df.up.max():.1f} mm')


if __name__ == '__main__':
    import sys
    if 'los' in sys.argv:
        cf_los_rate_2025()
        sys.exit()
    if 'rate' in sys.argv:
        cf_vertical_rate_2025()
    else:
        main()
        cf_vertical_2025()
