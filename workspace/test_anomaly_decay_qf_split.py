#!/usr/bin/env python3
"""QF-split of the held-out test-set anomaly decay (internal only, 2026-08-31).

Reads the per-date caches written by test_set_anomaly_decay.py (96 held-out
dates, 2014-2021) and repeats the before/after-DE anomaly-decay statistics
separately for QF = 0 and QF = 1 (snow included in each group, matching the
all-flag population of the manuscript figure; a QF0 snow-free variant is
printed to the console as a check but not plotted).

Outputs (internal, not manuscript display items):
  internal_test_anomaly_decay_qf.png    2x2 figure (rows QF0/QF1)
  test_anomaly_decay_binstats_qf.csv    per-bin stats with a qf column
plus printed headline numbers (near-window weighted mean, innermost bin,
far-field IQR at 20-30 km, totals) for the manuscript text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'workspace'))
from models.pipeline import MAX_ABS_ANOMALY_PPM  # noqa: E402

CACHE_DIR = (_REPO / 'results/figures/cld_dist_analysis/'
             'test_set_anomaly_decay/cache')
OUT_DIR = CACHE_DIR.parent
EDGES = np.arange(0.0, 31.0, 1.0)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
SURFACES = [('ocean', 0, 'r05', 5.0, 'tab:blue'),
            ('land', 1, 'r15', 15.0, 'tab:orange')]


def bin_stats(dist: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(dist)
    dist, y = dist[ok], y[ok]
    idx = np.digitize(dist, EDGES) - 1
    out = {k: np.full(len(CENTERS), np.nan) for k in
           ('mean', 'med', 'q25', 'q75')}
    out['n'] = np.zeros(len(CENTERS), int)
    for b in range(len(CENTERS)):
        yy = y[idx == b]
        if len(yy):
            out['n'][b] = len(yy)
            out['mean'][b] = yy.mean()
            out['med'][b] = np.median(yy)
            out['q25'][b], out['q75'][b] = np.percentile(yy, [25, 75])
    return out


def main() -> None:
    caches = sorted(CACHE_DIR.glob('*.parquet'))
    cols = ['date', 'cld_dist_km', 'sfc_type', 'xco2_qf', 'snow_flag',
            'xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15',
            'xco2_de_anomaly_r05', 'xco2_de_anomaly_r15']
    t = pd.concat([pd.read_parquet(p, columns=cols) for p in caches],
                  ignore_index=True)
    print(f'{len(t):,} soundings from {t.date.nunique()} test dates')
    for col in cols[5:]:
        t.loc[t[col].abs() > MAX_ABS_ANOMALY_PPM, col] = np.nan

    groups = [('qf0', t.xco2_qf == 0),
              ('qf1', t.xco2_qf == 1),
              ('qf0snowfree', (t.xco2_qf == 0) & (t.snow_flag == 0))]

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for gi, (gname, gmask) in enumerate(groups):
        plot_row = gi if gi < 2 else None
        for pi, kind in enumerate(('bc', 'de')):
            ax = axes[plot_row, pi] if plot_row is not None else None
            for surf, sfc, r, dmax, color in SURFACES:
                m = gmask & (t.sfc_type == sfc)
                y = t.loc[m, f'xco2_{kind}_anomaly_{r}'].to_numpy(float)
                dist = t.loc[m, 'cld_dist_km'].to_numpy(float)
                st = bin_stats(dist, y)
                for b in range(len(CENTERS)):
                    rows.append(dict(qf=gname, panel=kind, surface=surf,
                                     bin_center_km=CENTERS[b],
                                     n=int(st['n'][b]), mean=st['mean'][b],
                                     median=st['med'][b], q25=st['q25'][b],
                                     q75=st['q75'][b]))
                # headline numbers
                near = CENTERS <= dmax
                far = (CENTERS >= 20) & (CENTERS <= 30)
                w = st['n'][near] / max(st['n'][near].sum(), 1)
                wmean = float(np.nansum(st['mean'][near] * w))
                far_iqr = float(np.nanmean(st['q75'][far] - st['q25'][far]))
                n_tot = int(np.isfinite(y).sum())
                print(f'{gname:12s} {kind} {surf:5s}: n={n_tot:>9,}  '
                      f'near(<={dmax:g}) wmean={wmean:+.3f}  '
                      f'bin0 mean={st["mean"][0]:+.3f} '
                      f'med={st["med"][0]:+.3f}  farIQR={far_iqr:.3f}')
                if ax is not None:
                    ax.plot(CENTERS, st['mean'], color=color, lw=1.5,
                            label=f'{surf} ({r})')
                    ax.plot(CENTERS, st['med'], color=color, lw=1.2, ls='--')
                    ax.fill_between(CENTERS, st['q25'], st['q75'],
                                    color=color, alpha=0.18, lw=0)
            if ax is not None:
                ax.axhline(0, color='0.35', lw=0.8)
                ax.axvline(5, color='tab:blue', lw=0.8, ls=':', alpha=0.7)
                ax.axvline(15, color='tab:orange', lw=0.8, ls=':', alpha=0.7)
                ax.set_title(f'{gname.upper()} — '
                             f'{"before" if kind == "bc" else "after DE"}',
                             fontsize=10, loc='left')
                ax.set_xlim(0, 30)
                if gi == 1:
                    ax.set_xlabel('Nearest-cloud distance (km)')
                if pi == 0:
                    ax.set_ylabel('anomaly (ppm)')
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Held-out test-set anomaly decay, split by quality flag '
                 '(internal)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'internal_test_anomaly_decay_qf.png', dpi=200)
    pd.DataFrame(rows).to_csv(
        OUT_DIR / 'test_anomaly_decay_binstats_qf.csv', index=False)
    print('written to', OUT_DIR)


if __name__ == '__main__':
    main()
