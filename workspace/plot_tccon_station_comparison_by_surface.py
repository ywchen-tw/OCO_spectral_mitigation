"""plot_tccon_station_comparison_by_surface.py — OCO-2 vs TCCON, ocean vs land.

Surface-separated companion to plot_tccon_station_comparison.py.  Reads the
per-surface station-day means written by tccon_correction_policy_stats.py
(tccon_policy_station_means_by_surface.csv) and makes a 4-panel figure:

  row 0 (OCEAN):  A = OCO mean vs TCCON (1:1, OLS)   B = per-station-day bias dumbbell
  row 1 (LAND):   A = OCO mean vs TCCON (1:1, OLS)   B = per-station-day bias dumbbell

Each panel compares uncorrected xco2_bc with the deep-ensemble correction (full μ).
Run once for ALL stations and once with --exclude-sites ny.

Run: PYTHONPATH=src python workspace/plot_tccon_station_comparison_by_surface.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get('CURC_DATA_ROOT')
                 or os.environ.get('OCO2_DATAROOT') or ROOT)
MEANS = DATA_ROOT / 'results/model_comparison/tccon_policy/tccon_policy_station_means_by_surface.csv'
OUTDIR = DATA_ROOT / 'results/model_comparison/tccon_policy'


def _ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]; n = x.size
    if n < 3:
        return None
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm) ** 2)
    if Sxx == 0:
        return None
    slope = np.sum((x - xm) * (y - ym)) / Sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    sse = np.sum(resid ** 2); sst = np.sum((y - ym) ** 2)
    dof = n - 2
    slope_se = np.sqrt((sse / dof) / Sxx) if dof > 0 else np.nan
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return dict(slope=slope, intercept=intercept, slope_se=slope_se, r2=r2, n=n)


_SERIES = [('original xco$_2$_bc', 'uncorrected', 'steelblue'),
           ('corrected (full $\\mu$)', 'full_mu', 'green')]


def _draw_surface(axA, axB, df, sname):
    """Draw the scatter (axA) + bias dumbbell (axB) panels for one surface."""
    if not len(df):
        for ax in (axA, axB):
            ax.text(0.5, 0.5, f'no {sname} station-days', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')
        return None
    df = df.sort_values('full_mu').reset_index(drop=True)
    x = df['tccon_ref'].to_numpy(float); xe = df['tccon_sd'].to_numpy(float)

    # ── panel A: scatter vs TCCON, 1:1, OLS fits ──
    allv = [df['tccon_ref'], df['uncorrected'], df['full_mu']]
    lo = float(np.nanmin([v.min() for v in allv])) - 1
    hi = float(np.nanmax([v.max() for v in allv])) + 1
    xs = np.array([lo, hi]); txt = []
    for lbl, col, color in _SERIES:
        y = df[col].to_numpy(float); ye = df[f'{col}_sd'].to_numpy(float)
        axA.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', ms=5, color=color,
                     alpha=0.55, elinewidth=0.7, label=lbl)
        f = _ols(x, y)
        if f:
            axA.plot(xs, f['intercept'] + f['slope'] * xs, '-', color=color,
                     lw=1.7, alpha=0.9, zorder=4)
            txt.append(f"{lbl}:  slope = {f['slope']:.3f} ± {f['slope_se']:.3f}   "
                       f"R² = {f['r2']:.3f}")
    axA.plot(xs, xs, 'k--', lw=1, label='1:1')
    axA.text(0.04, 0.96, '\n'.join(txt), transform=axA.transAxes, va='top', ha='left',
             fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    axA.set_xlabel('TCCON XCO₂ (ppm)'); axA.set_ylabel('OCO-2 XCO₂ (ppm)')
    axA.set_title(f'{sname.upper()}: OCO-2 vs TCCON  ({len(df)} station-days)')
    axA.legend(loc='lower right', fontsize=8); axA.grid(alpha=0.3)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect('equal')

    # ── panel B: per-station-day bias, original → corrected dumbbell ──
    b_orig = df['uncorrected'].to_numpy(float) - x
    b_corr = df['full_mu'].to_numpy(float) - x
    yv = np.arange(len(df))
    for yi in yv:
        axB.plot([b_orig[yi], b_corr[yi]], [yi, yi], '-', color='lightgray', lw=1.3, zorder=1)
    axB.scatter(b_orig, yv, facecolors='none', edgecolors='steelblue', s=22,
                label='original (no corr)', zorder=2)
    axB.scatter(b_corr, yv, color='green', s=30, label='corrected (full μ)', zorder=3)
    axB.axvline(0, color='k', lw=1)
    axB.set_yticks(yv)
    axB.set_yticklabels([f"{r.site} {r.date}" for r in df.itertuples()], fontsize=6)
    axB.set_xlabel('XCO₂ bias to TCCON (ppm)')

    def _agg(b):
        return np.nanmean(np.abs(b)), np.sqrt(np.nanmean(b ** 2))
    (mb0, rm0), (mb1, rm1) = _agg(b_orig), _agg(b_corr)
    axB.set_title(f'{sname.upper()}: bias to TCCON  '
                  f'(mean|bias| {mb0:.2f}→{mb1:.2f}, RMS {rm0:.2f}→{rm1:.2f})', fontsize=10)
    axB.legend(loc='lower right', fontsize=8); axB.grid(alpha=0.3, axis='x')
    return dict(mean_abs_before=mb0, mean_abs_after=mb1, rms_before=rm0, rms_after=rm1, n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--means', default=str(MEANS))
    ap.add_argument('--output-dir', default=str(OUTDIR))
    ap.add_argument('--exclude-sites', default='',
                    help="Comma-separated TCCON site codes to drop (e.g. 'ny').")
    ap.add_argument('--fname-suffix', default='',
                    help="Appended before the extension of the output figure name.")
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    excl = [s.strip() for s in args.exclude_sites.split(',') if s.strip()]
    tag = ('_excl_' + '_'.join(excl)) if excl else ''

    df = pd.read_csv(args.means)
    df = df.dropna(subset=['tccon_ref', 'full_mu', 'uncorrected']).copy()
    if excl:
        n0 = len(df); df = df[~df['site'].isin(excl)]
        print(f"  excluded sites {excl}: {n0 - len(df)} surface-station-days dropped → {len(df)} remain")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for row, sname in enumerate(('ocean', 'land')):
        sub = df[df['surface'] == sname]
        stats = _draw_surface(axes[row, 0], axes[row, 1], sub, sname)
        if stats:
            print(f"  {sname:5s}: n={stats['n']:3d}  mean|bias| {stats['mean_abs_before']:.3f}"
                  f"→{stats['mean_abs_after']:.3f}  RMSbias {stats['rms_before']:.3f}"
                  f"→{stats['rms_after']:.3f}")
    _sup = 'OCO-2 vs TCCON by surface — original vs corrected'
    if excl:
        _sup += f'  (excl {",".join(excl)})'
    fig.suptitle(_sup, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = out_dir / f'tccon_station_comparison_by_surface{tag}{args.fname_suffix}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  [saved] {out}")


if __name__ == '__main__':
    main()
