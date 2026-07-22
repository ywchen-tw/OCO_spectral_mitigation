#!/usr/bin/env python3
"""Manuscript Fig. 7b — TCCON significance + coincidence-criteria robustness.

(a) Forest plot of the site-clustered bootstrap deltas (corrected - uncorrected)
    at the primary coincidence criteria (r = 100 km, +/-60 min), AK-harmonized
    TCCON reference, all quality flags: Delta mean |bias|, Delta RMS bias and
    Delta mean per-footprint RMSE, each for all 18 sites and for the
    17-site subset excluding Ny-Alesund.  95% CI whiskers + bootstrap p per row.
(b) Coincidence-sensitivity matrix: the same site-clustered Delta mean |bias|
    re-evaluated on a 3x3 grid of collocation radius (25/50/100 km) x time
    window (+/-30/60/120 min); every cell is significant, tighter radii
    improve more.  The primary cell (100 km, +/-60 min) is outlined.

Inputs (existing CSVs, no recomputation):
  results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/atrain/
      tccon_significance_r100km.csv
      coincidence_sensitivity/coincidence_sensitivity.csv

Output:
  manuscript/figures/fig07b_significance_robustness.png / .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workspace"))  # plot_style
from plot_style import CMAPS, apply_manuscript_style, panel_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ATRAIN = (ROOT / 'results' / 'model_comparison' / 'deep_ensemble'
          / 'de_beta_nll_prof_reg_o05l15_m5' / 'atrain')
SIG_CSV = ATRAIN / 'tccon_significance_r100km.csv'
COIN_CSV = ATRAIN / 'coincidence_sensitivity' / 'coincidence_sensitivity.csv'
OUT_DIR = ROOT / 'manuscript' / 'figures'

# Okabe-Ito
C_ALL = '#0072B2'      # blue    — all sites
C_EXCL = '#E69F00'     # orange  — excluding Ny-Alesund

METRICS = [   # (csv stem, row label)
    ('d_mean_absbias', r'$\Delta$ mean $|$bias$|$'),
    ('d_rms_bias',     r'$\Delta$ RMS bias'),
    ('d_mean_rmse',    r'$\Delta$ fp-RMSE'),
]
SUBSETS = [   # (csv value, legend label, color)
    ('all',     'all sites (18)', C_ALL),
    ('excl_ny', r'excl. Ny-$\mathrm{\AA}$lesund (17)', C_EXCL),
]


def _fmt_p(p: float) -> str:
    """ASCII-only p-value tag (bootstrap floor 2/(B+1) ~ 2e-4 shows as such)."""
    if not np.isfinite(p):
        return ''
    return f'p = {p:.4f}'.rstrip('0') if p < 0.001 else f'p = {p:.3f}'


def panel_forest(ax, sig: pd.DataFrame) -> None:
    rows = sig[sig['qf_group'] == 'all'].set_index('subset')
    ys, gap, step = [], 1.0, 0.72
    y = 0.0
    yticks, yticklabels = [], []
    for mi, (stem, mlabel) in enumerate(METRICS):
        y_group = []
        for subset, slabel, color in SUBSETS:
            r = rows.loc[subset]
            mean = r[f'{stem}_mean']
            lo, hi = r[f'{stem}_ci_lo'], r[f'{stem}_ci_hi']
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                        fmt='o', ms=5, color=color, ecolor=color,
                        elinewidth=1.4, capsize=3, capthick=1.4, zorder=3)
            ax.annotate(_fmt_p(r[f'{stem}_p']), xy=(1.02, y),
                        xycoords=('axes fraction', 'data'),
                        va='center', ha='left', fontsize=8, color='0.25')
            y_group.append(y)
            y -= step
        yticks.append(np.mean(y_group))
        yticklabels.append(mlabel)
        y -= gap - step  # extra space between metric groups
        ys.extend(y_group)
    ax.axvline(0.0, color='0.4', lw=0.8, ls='--', zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    lo_x, _ = ax.get_xlim()
    ax.set_xlim(lo_x, 0.12)   # keep the zero line off the axis edge
    ax.set_xlabel(r'$\Delta$ (corrected $-$ uncorrected)  [ppm]')
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    # column header for the p annotations
    ax.annotate('bootstrap p', xy=(1.02, 1.005), xycoords='axes fraction',
                va='bottom', ha='left', fontsize=8, color='0.25', style='italic')
    ax.legend(handles=[
        plt.Line2D([], [], marker='o', ls='none', ms=5, color=c, label=l)
        for _, l, c in SUBSETS],
        loc='upper left', frameon=False, handletextpad=0.4,
        borderaxespad=0.0, fontsize=8)
    wilc = sig[(sig['qf_group'] == 'all') & (sig['subset'] == 'all')].iloc[0]
    ax.annotate('paired Wilcoxon, station-day $|$bias$|$: '
                + f'p = {wilc["wilcoxon_absbias_p"]:.4f}'
                + f'  (n = {int(wilc["wilcoxon_absbias_n"])} station-days)',
                xy=(0.0, -0.24), xycoords='axes fraction', va='top',
                ha='left', fontsize=8, color='0.25')


def panel_matrix(ax, fig, coin: pd.DataFrame) -> None:
    radii = [25, 50, 100]
    windows = [30, 60, 120]
    grid = np.full((3, 3), np.nan)
    pmat = np.full((3, 3), np.nan)
    for _, r in coin.iterrows():
        i = radii.index(int(r['radius_km']))
        j = windows.index(int(r['window_min']))
        grid[i, j] = r['d_absbias']
        pmat[i, j] = r['d_absbias_p']
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap=CMAPS['mu'], vmin=-vmax, vmax=vmax,
                   origin='upper', aspect='equal')
    for i in range(3):
        for j in range(3):
            rgba = im.cmap(im.norm(grid[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = 'white' if lum < 0.5 else 'black'
            ax.text(j, i - 0.13, f'{grid[i, j]:.2f}', ha='center',
                    va='center', color=tc, fontsize=9, fontweight='bold')
            ax.text(j, i + 0.22, _fmt_p(pmat[i, j]), ha='center',
                    va='center', color=tc, fontsize=7.5)
    # outline the primary configuration (100 km, +/-60 min)
    ip, jp = radii.index(100), windows.index(60)
    ax.add_patch(plt.Rectangle((jp - 0.5, ip - 0.5), 1, 1, fill=False,
                               ec='black', lw=1.8, zorder=4))
    ax.text(jp, ip - 0.37, 'primary', ha='center', va='center',
            fontsize=7, style='italic', color='black')
    ax.set_xticks(range(3))
    ax.set_xticklabels([rf'$\pm${w}' for w in windows])
    ax.set_yticks(range(3))
    ax.set_yticklabels([str(r) for r in radii])
    ax.set_xlabel('coincidence window  [min]')
    ax.set_ylabel('collocation radius  [km]')
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.set_label(r'$\Delta$ mean $|$bias$|$  [ppm]')
    cb.outline.set_linewidth(0.8)


def main() -> None:
    apply_manuscript_style()
    sig = pd.read_csv(SIG_CSV)
    coin = pd.read_csv(COIN_CSV)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.4, 3.1),
        gridspec_kw=dict(width_ratios=[1.30, 1.0], wspace=0.52))
    panel_forest(ax_a, sig)
    panel_matrix(ax_b, fig, coin)
    panel_label(ax_a, '(a)', dx=-0.28)
    panel_label(ax_b, '(b)', dx=-0.22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'fig07b_significance_robustness.{ext}')
    plt.close(fig)
    print('wrote', OUT_DIR / 'fig07b_significance_robustness.png')


if __name__ == '__main__':
    main()
