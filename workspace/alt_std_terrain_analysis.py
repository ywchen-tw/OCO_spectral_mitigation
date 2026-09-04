#!/usr/bin/env python3
"""Terrain-roughness (alt_std) analysis for the near-cloud XCO2 anomaly.

Three questions (2026-08-31, JQSRT review prep):
  A. Confound map — does sub-footprint elevation spread vary with cloud
     distance (orographic cloud) or track the albedo axis, i.e. could
     terrain masquerade as the proximity signal?
  B. Near-cloud modulation — does the land anomaly vary with alt_std once
     the albedo axis is controlled (tercile x tercile)?
  C. Far-field feature response — far from cloud (>= 30 km), do the
     standardized band-k1 / continuum-reflectance departures spread with
     roughness (the cloud-free terrain sensitivity discussed qualitatively
     in the JQSRT discussion section)?

Land only: alt_std is degenerate over ocean (built-in null). Population and
conventions follow bias_sign_conditions.py / spec_sensitivity.py: QF0
snow-free, production land target xco2_bc_anomaly_r15, r15 references,
plain (dual-fit) k1 columns so sigma units match the Fig. 5 numbers.

Outputs (CSV) in results/figures/cld_dist_analysis/alt_std_terrain/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'src'))
from models.pipeline import MAX_ABS_ANOMALY_PPM  # noqa: E402

PARQUET = _REPO / 'results/csv_collection/combined_2016_2020_dates.parquet'
OUT_DIR = _REPO / 'results/figures/cld_dist_analysis/alt_std_terrain'

TARGET = 'xco2_bc_anomaly_r15'
BANDS = ['o2a', 'wco2', 'sco2']
DIST_EDGES = [0, 1, 2, 5, 10, 20, 30, 50, np.inf]
FAR_KM = 30.0


def load() -> pd.DataFrame:
    cols = (['sfc_type', 'cld_dist_km', 'xco2_qf', 'snow_flag',
             'alt_std', 'alb_wco2', 'exp_o2a_intercept', TARGET,
             'r15_exp_int_o2a_mean', 'r15_exp_int_o2a_std'] +
            [f'{b}_k1' for b in BANDS] +
            [f'r15_{b}_k1_mean' for b in BANDS] +
            [f'r15_{b}_k1_std' for b in BANDS])
    df = pd.read_parquet(PARQUET, columns=cols)
    df = df[df.sfc_type == 1].copy()          # land only
    y = df[TARGET]
    df.loc[y.abs() > MAX_ABS_ANOMALY_PPM, TARGET] = np.nan
    for b in BANDS:
        df[f'zk1_{b}'] = ((df[f'{b}_k1'] - df[f'r15_{b}_k1_mean'])
                          / df[f'r15_{b}_k1_std'].replace(0, np.nan))
    df['zexp_o2a'] = ((df.exp_o2a_intercept - df.r15_exp_int_o2a_mean)
                      / df.r15_exp_int_o2a_std.replace(0, np.nan))
    return df


def stats_block(y: pd.Series) -> dict:
    return dict(n=len(y), mean=float(y.mean()), median=float(y.median()),
                frac_pos=float((y > 0).mean()))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    qf0 = df[(df.xco2_qf == 0) & (df.snow_flag == 0)
             & np.isfinite(df.alt_std) & np.isfinite(df.cld_dist_km)].copy()
    print(f'land rows {len(df):,}; QF0 snow-free {len(qf0):,}')

    # --- A. confound map: alt_std vs cloud distance and albedo ------------
    rows = []
    dist_bin = pd.cut(qf0.cld_dist_km, DIST_EDGES, right=False)
    for iv, g in qf0.groupby(dist_bin, observed=True):
        rows.append({'dist_bin_km': f'{iv.left:g}..{iv.right:g}',
                     'n': len(g),
                     'alt_std_mean': float(g.alt_std.mean()),
                     'alt_std_median': float(g.alt_std.median()),
                     'alb_wco2_mean': float(g.alb_wco2.mean())})
    conf = pd.DataFrame(rows)
    lt50 = qf0[qf0.cld_dist_km < 50]
    r_dist, p_dist = sps.spearmanr(lt50.alt_std, lt50.cld_dist_km)
    ok = np.isfinite(qf0.alb_wco2)
    r_alb, p_alb = sps.spearmanr(qf0.alt_std[ok], qf0.alb_wco2[ok])
    conf.to_csv(OUT_DIR / 'alt_std_vs_clddist.csv', index=False)
    print(conf.to_string(index=False))
    print(f'spearman alt_std~cld_dist (<50 km): r={r_dist:+.3f} p={p_dist:.2g}')
    print(f'spearman alt_std~alb_wco2:          r={r_alb:+.3f} p={p_alb:.2g}')

    # --- B. near-cloud anomaly: alt_std tercile x alb_wco2 tercile --------
    tw_rows = []
    for win, dmax in [('near5', 5.0), ('near15', 15.0)]:
        sub = qf0[qf0.cld_dist_km.between(0, dmax)
                  & np.isfinite(qf0[TARGET]) & np.isfinite(qf0.alb_wco2)]
        t_alt = pd.qcut(sub.alt_std, 3, labels=['flat', 'mid', 'rough'],
                        duplicates='drop')
        t_alb = pd.qcut(sub.alb_wco2, 3, labels=['dark', 'mid', 'bright'])
        for ta in t_alt.cat.categories:
            for tb in ['dark', 'mid', 'bright']:
                mm = (t_alt == ta) & (t_alb == tb)
                if mm.sum() > 200:
                    tw_rows.append({'window': win, 'alt_std': ta,
                                    'alb_wco2': tb,
                                    'alt_std_edge': f'{sub.alt_std[t_alt == ta].min():.0f}..'
                                                    f'{sub.alt_std[t_alt == ta].max():.0f}',
                                    **stats_block(sub[TARGET][mm])})
    tw = pd.DataFrame(tw_rows)
    tw.to_csv(OUT_DIR / 'alt_std_twoway.csv', index=False)
    print(tw.to_string(index=False))

    # --- C. far-field feature spread vs alt_std quintile ------------------
    far = qf0[qf0.cld_dist_km >= FAR_KM].copy()
    q = pd.qcut(far.alt_std, 5, duplicates='drop')
    fr_rows = []
    zcols = [f'zk1_{b}' for b in BANDS] + ['zexp_o2a']
    for iv, g in far.groupby(q, observed=True):
        row = {'alt_std_bin_m': f'{iv.left:.3g}..{iv.right:.3g}',
               'n': len(g),
               'alb_wco2_mean': float(g.alb_wco2.mean()),
               'anom_mean': float(g[TARGET].mean()),
               'anom_std': float(g[TARGET].std())}
        for c in zcols:
            row[f'{c}_mean'] = float(g[c].mean())
            row[f'{c}_std'] = float(g[c].std())
        fr_rows.append(row)
    fr = pd.DataFrame(fr_rows)
    fr.to_csv(OUT_DIR / 'alt_std_farfield_features.csv', index=False)
    print(fr.to_string(index=False))
    for c in zcols:
        okc = np.isfinite(far[c])
        r, p = sps.spearmanr(far.alt_std[okc], far[c][okc].abs())
        print(f'far-field spearman alt_std~|{c}|: r={r:+.3f} p={p:.2g} '
              f'n={int(okc.sum()):,}')
    print('written to', OUT_DIR)


if __name__ == '__main__':
    main()
