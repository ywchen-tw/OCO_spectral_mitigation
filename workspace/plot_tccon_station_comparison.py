"""plot_tccon_station_comparison.py — OCO-2 vs TCCON, original vs corrected.

Station-day view: reads the per-station-day means written by
tccon_correction_policy_stats.py (tccon_policy_station_means.csv) and compares the
uncorrected OCO-2 (xco2_bc) with the deep-ensemble correction (full μ) against
TCCON.  Run once for ALL stations and once with --exclude-sites ny so the verdict
is shown with and without the sole high-latitude outlier (Ny-Ålesund, |lat|>75).

Panel A: OCO-2 mean vs TCCON mean (1:1), 2 series — original xco2_bc and corrected
         — each with an OLS fit + slope/R².
Panel B: per-station-day bias-to-TCCON, original → corrected dumbbell.

(Supersedes plot_latgate_tccon_comparison.py — the |lat|>L gate was dropped after
the radius sweep showed the full correction beats the gate at every radius.)

Run: PYTHONPATH=src python workspace/plot_tccon_station_comparison.py
     PYTHONPATH=src python workspace/plot_tccon_station_comparison.py --exclude-sites ny
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
# On CURC the results/ tree lives under scratch, not the repo checkout.  Honor the
# same data root as tccon_correction_policy_stats.py / get_storage_dir().
DATA_ROOT = Path(os.environ.get('CURC_DATA_ROOT')
                 or os.environ.get('OCO2_DATAROOT') or ROOT)
MEANS = DATA_ROOT / 'results/model_comparison/tccon_policy/tccon_policy_station_means.csv'
OUTDIR = DATA_ROOT / 'results/model_comparison/tccon_policy'


def _ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]; n = x.size
    if n < 3:
        return None
    xm, ym = x.mean(), y.mean()
    Sxx = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - ym)) / Sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    sse = np.sum(resid ** 2); sst = np.sum((y - ym) ** 2)
    dof = n - 2
    slope_se = np.sqrt((sse / dof) / Sxx) if dof > 0 else np.nan
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return dict(slope=slope, intercept=intercept, slope_se=slope_se, r2=r2, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--means', default=str(MEANS))
    ap.add_argument('--output-dir', default=str(OUTDIR))
    ap.add_argument('--exclude-sites', default='',
                    help="Comma-separated TCCON site codes to drop (e.g. 'ny').")
    ap.add_argument('--fname-suffix', default='',
                    help="Appended before the extension of the output figure name "
                         "(e.g. '_r100km') so a parameter sweep's figures coexist.")
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    excl = [s.strip() for s in args.exclude_sites.split(',') if s.strip()]
    tag = ('_excl_' + '_'.join(excl)) if excl else ''

    df = pd.read_csv(args.means)
    df = df.dropna(subset=['tccon_ref', 'full_mu', 'uncorrected']).copy()
    if excl:
        n0 = len(df); df = df[~df['site'].isin(excl)]
        print(f"  excluded sites {excl}: {n0 - len(df)} station-days dropped → {len(df)} remain")
    df = df.sort_values('full_mu').reset_index(drop=True)

    x = df['tccon_ref'].to_numpy(float)
    xe = df['tccon_sd'].to_numpy(float)
    series = [
        ('original xco$_2$_bc', 'uncorrected', 'steelblue'),
        ('corrected (full $\\mu$)', 'full_mu', 'green'),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel A: scatter vs TCCON, 1:1, OLS fits ──────────────────────────────
    allv = [df['tccon_ref'], df['uncorrected'], df['full_mu']]
    lo = float(np.nanmin([v.min() for v in allv])) - 1
    hi = float(np.nanmax([v.max() for v in allv])) + 1
    xs = np.array([lo, hi])
    txt = []
    for lbl, col, color in series:
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
             fontsize=9.5, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    axA.set_xlabel('TCCON XCO₂ (ppm)'); axA.set_ylabel('OCO-2 XCO₂ (ppm)')
    _ttl = 'OCO-2 vs TCCON (mean ± std)  —  original vs corrected'
    if excl:
        _ttl += f'  (excl {",".join(excl)})'
    axA.set_title(_ttl)
    axA.legend(loc='lower right'); axA.grid(alpha=0.3)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.set_aspect('equal')

    # ── Panel B: per-station-day bias, original → corrected dumbbell ────────────
    b_orig = df['uncorrected'].to_numpy(float) - x
    b_corr = df['full_mu'].to_numpy(float) - x
    yv = np.arange(len(df))
    for yi in yv:
        axB.plot([b_orig[yi], b_corr[yi]], [yi, yi], '-', color='lightgray',
                 lw=1.3, zorder=1)
    axB.scatter(b_orig, yv, facecolors='none', edgecolors='steelblue', s=22,
                label='original (no corr)', zorder=2)
    axB.scatter(b_corr, yv, color='green', s=30, label='corrected (full μ)', zorder=3)
    axB.axvline(0, color='k', lw=1)
    axB.set_yticks(yv)
    axB.set_yticklabels([f"{r.site} {r.date}" for r in df.itertuples()], fontsize=6)
    axB.set_xlabel('XCO₂ bias to TCCON (ppm)')
    axB.set_title('Bias to TCCON: original → corrected')
    axB.legend(loc='lower right', fontsize=8); axB.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    out = out_dir / f'tccon_station_comparison{tag}{args.fname_suffix}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)

    # headline numbers
    def agg(b):
        return np.nanmean(np.abs(b)), np.sqrt(np.nanmean(b ** 2))
    for nm, b, col in [('original', b_orig, 'uncorrected'), ('corrected', b_corr, 'full_mu')]:
        mab, rms = agg(b)
        f = _ols(x, df[col])
        print(f"  {nm:9s}  mean|bias|={mab:.3f}  RMSbias={rms:.3f}  "
              f"slope={f['slope']:.4f}  R²={f['r2']:.4f}")
    print(f"\n  {len(df)} station-days ({tag or 'all sites'})")
    print(f"  [saved] {out}")


if __name__ == '__main__':
    main()
