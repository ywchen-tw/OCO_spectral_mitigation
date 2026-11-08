#!/usr/bin/env python3
"""Median near-cloud XCO2 anomaly by illumination class and WCO2 albedo tercile.

Population and classes follow bias_sign_conditions.py (two-way table): per
surface, snow-free footprints of both quality flags within 5 km of cloud,
target screen |y| <= 100 ppm, illumination class from the standardized O2
A-band continuum-reflectance departure z (shadowed z < -0.5, neutral
|z| <= 0.5, brightened z > +0.5), WCO2 albedo terciles computed per surface.
Adds the pooled row (all terciles) and column (all classes).

FIT SCHEMA (2026-11-02): ``--fit-schema v2`` (default) takes the O2A continuum
reflectance of the agreed-model Doppler refit (results_doppler/), gamma =
o2a_exp_int_am with references r05_/r15_o2a_exp_int_am_{mean,std}, and z as in
analysis.spec_sensitivity._add_v2_deltas (NaN where the reference sigma is not
positive).  ``--fit-schema v1`` reproduces the pre-2026-11-02 edition through
bias_sign_conditions.load() (unsmoothed order-7/3/7 fit, ``_nosg`` sidecar of
results/csv_collection).

Output (v2; v1 under results/ instead of results_doppler/):
  results_doppler/figures/cld_dist_analysis/bias_sign_conditions/illum_albedo_medians.csv
  manuscript/JQSRT_draft/tables/tabB3_illum_albedo.tex   (--tex PATH writes elsewhere)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_sign_conditions import (GAMMA, OUT_DIR, SURFACES, _REPO,  # noqa: E402
                                  gamma_ref, load)
from models.pipeline import MAX_ABS_ANOMALY_PPM  # noqa: E402  (src/ on the path via bias_sign_conditions)

TEX = _REPO / 'manuscript/JQSRT_draft/tables/tabB3_illum_albedo.tex'
_OUT_REL = 'figures/cld_dist_analysis/bias_sign_conditions'
RESULTS = {'v1': _REPO / 'results', 'v2': _REPO / 'results_doppler'}
PARQUETS = {k: v / 'csv_collection/combined_2016_2020_dates.parquet' for k, v in RESULTS.items()}
OUT_DIRS = {'v1': OUT_DIR, 'v2': RESULTS['v2'] / _OUT_REL}
# O2A continuum reflectance gamma of each fit schema: observed column and the
# reference-statistics column of reference prefix ref ('r05'/'r15') and stat
# ('mean'/'std').  v1 = bias_sign_conditions.GAMMA / gamma_ref.
GAMMA_COLS = {
    'v1': (GAMMA, gamma_ref),
    'v2': ('o2a_exp_int_am', lambda ref, stat: f'{ref}_o2a_exp_int_am_{stat}'),
}
NEAR_KM = 5.0
TERCILES = ['dark', 'mid', 'bright']
CLASSES = ['shadowed', 'neutral', 'brightened']


def load_v2(path):
    """Columns of the v2 training parquet for the table, with the training
    target screen |y| <= MAX_ABS_ANOMALY_PPM (as bias_sign_conditions.load)."""
    import pyarrow.parquet as pq
    gamma, ref_col = GAMMA_COLS['v2']
    cols = (['sfc_type', 'cld_dist_km', 'snow_flag', 'alb_wco2', gamma]
            + [s['target'] for s in SURFACES.values()]
            + [ref_col(s['ref'], st) for s in SURFACES.values() for st in ('mean', 'std')])
    df = pq.read_table(path, columns=sorted(set(cols))).to_pandas()
    print(f'{len(df):,} rows read from {path} (fit schema v2)')
    for s in SURFACES.values():
        y = df[s['target']]
        df.loc[y.abs() > MAX_ABS_ANOMALY_PPM, s['target']] = np.nan
    return df


def cells(df, spec, fit_schema='v2'):
    d = df[df.sfc_type == spec['sfc']]
    y = d[spec['target']]
    gamma, ref_col = GAMMA_COLS[fit_schema]
    ref_s = d[ref_col(spec['ref'], 'std')]
    # z of the illumination class: standardized departure of gamma from the
    # clear-sky reference; a non-positive reference sigma gives NaN
    z = (d[gamma] - d[ref_col(spec['ref'], 'mean')]) / ref_s.where(ref_s > 0)
    ok = (np.isfinite(y) & d.cld_dist_km.between(0, NEAR_KM) & (d.snow_flag == 0)
          & np.isfinite(d.alb_wco2) & np.isfinite(z))
    y, z, alb = y[ok], z[ok], d.alb_wco2[ok]
    edges = np.quantile(alb, [0, 1 / 3, 2 / 3, 1])
    terc = pd.qcut(alb, 3, labels=TERCILES)
    cls = pd.Series(np.select([z < -0.5, z > 0.5], ['shadowed', 'brightened'], 'neutral'),
                    index=z.index)
    rows = []
    for t in TERCILES + ['all']:
        for c in CLASSES + ['all']:
            m = ((terc == t) if t != 'all' else True) & ((cls == c) if c != 'all' else True)
            m = pd.Series(m, index=y.index) if np.isscalar(m) else m
            rows.append({'tercile': t, 'class': c, 'n': int(m.sum()),
                         'median': float(y[m].median())})
    return pd.DataFrame(rows), edges


def _signed(v):
    """Two-decimal median with a true minus; 0.00 unsigned."""
    mag = f'{abs(v):.2f}'
    if float(mag) == 0.0:
        return mag
    return ('$-$' if v < 0 else '+') + mag


def tex_table(results, parquet=PARQUETS['v2']):
    # Only the inner tercile edges are printed: the outer edges are the sample
    # extremes, and the retrieved Lambertian albedo (Lite albedo_wco2) has rare
    # outliers above 1 that would otherwise label the brightest tercile.
    min_n = min(int(tab[tab.tercile.isin(TERCILES) & tab['class'].isin(CLASSES)].n.min())
                for tab, _ in results.values())
    lines = [
        '% Generated by workspace/illum_albedo_table.py -- do not hand-edit.',
        f'% Source: {Path(parquet).relative_to(_REPO)} (training dates).',
        r'\begin{table}[t]',
        r'\caption{Median anomaly \delxcobc (ppm) of snow-free footprints of both quality '
        r'flags with a nearest-cloud distance of at most 5~km, by illumination class '
        r'(Sect.~\ref{sec:phenom-illum}) and tercile of the retrieved W\ce{CO2} surface '
        r'albedo. The terciles are computed separately for ocean and land, with the '
        r'tercile edges given in parentheses. Each cell holds at least '
        f'{min_n // 1000 * 1000} footprints.}}',
        r'\label{tab:illum-albedo}',
        r'\centering',
        r'\footnotesize',
        r'\begin{tabular}{lcccc}',
        r'\tophline',
        r' & Shadowed & Neutral & Brightened & All \\',
    ]
    names = {'dark': 'Darkest tercile', 'mid': 'Middle tercile', 'bright': 'Brightest tercile',
             'all': 'All terciles'}
    for surface, (tab, edges) in results.items():
        lo, hi = f'{edges[1]:.2f}', f'{edges[2]:.2f}'
        # pd.qcut bins are right-closed: (min, e1], (e1, e2], (e2, max]
        ranges = {'dark': rf'$\leq$\,{lo}', 'mid': f'{lo}--{hi}', 'bright': rf'$>$\,{hi}'}
        lines.append(r'\middlehline')
        lines.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{surface.capitalize()}}}}} \\')
        for t in TERCILES + ['all']:
            label = names[t]
            if t != 'all':
                label += f' ({ranges[t]})'
            vals = [tab[(tab.tercile == t) & (tab['class'] == c)]['median'].iloc[0]
                    for c in CLASSES + ['all']]
            lines.append(label + ' & ' + ' & '.join(_signed(v) for v in vals) + r' \\')
    lines += [r'\bottomhline', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tex', type=Path, default=TEX, help='output .tex path')
    ap.add_argument('--fit-schema', choices=sorted(RESULTS), default='v2',
                    help='v2 (default): agreed-model o2a_exp_int_am of results_doppler/; '
                         'v1: the pre-2026-11-02 _nosg fit of results/')
    ap.add_argument('--parquet', type=Path, default=None,
                    help="training parquet (default: the fit schema's "
                         'csv_collection/combined_2016_2020_dates.parquet; '
                         'v1 always reads bias_sign_conditions.PARQUET)')
    args = ap.parse_args()
    parquet = args.parquet or PARQUETS[args.fit_schema]
    out_dir = OUT_DIRS[args.fit_schema]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_v2(parquet) if args.fit_schema == 'v2' else load()
    results, frames = {}, []
    for name, spec in SURFACES.items():
        tab, edges = cells(df, spec, args.fit_schema)
        results[name] = (tab, edges)
        frames.append(tab.assign(surface=name))
        print(name, 'tercile edges', np.round(edges, 3))
        print(tab.pivot(index='tercile', columns='class', values='median').round(3))
        print(tab.pivot(index='tercile', columns='class', values='n'))
    pd.concat(frames).to_csv(out_dir / 'illum_albedo_medians.csv', index=False)
    print('written', out_dir / 'illum_albedo_medians.csv')
    args.tex.write_text(tex_table(results, parquet))
    print('written', args.tex)


if __name__ == '__main__':
    main()
