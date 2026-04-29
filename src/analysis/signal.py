"""
ca_signal.py
============
Signal-hierarchy plot functions extracted from combined_analyze.py.

Contents
--------
- plot_signal_hierarchy           Pearson r(cld_dist) bar chart — k1/k2/k3/exp/exp-alb
- plot_residual_signal_hierarchy  Same after OLS-removing albedo + airmass + cos(SZA)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .utils import _save, bin_by_cld_dist

logger = logging.getLogger(__name__)


def plot_signal_hierarchy(df: pd.DataFrame, outdir: str) -> None:
    """Bar chart of Pearson r(cld_dist_km) for k1, k2, k3, exp_intercept, and
    the exp/alb ratio across all three bands, shown side-by-side for ocean and land.

    The exp/alb ratio bars (rightmost group, separated by a dashed divider) isolate
    the non-albedo component of the cloud-proximity signal.  On ocean the ratio
    r is negative (ratio is higher near clouds); on land it is positive (ratio
    collapses near clouds) — opposite signs expose the two distinct mechanisms.
    """
    # ── compute exp/alb ratio columns ─────────────────────────────────────────
    ratio_pairs = [
        ('exp_o2a_intercept',  'alb_o2a',  '_exp_alb_o2a'),
        ('exp_wco2_intercept', 'alb_wco2', '_exp_alb_wco2'),
        ('exp_sco2_intercept', 'alb_sco2', '_exp_alb_sco2'),
    ]
    new_ratio_cols = {
        ratio_col: df[exp_col] / df[alb_col].replace(0, np.nan)
        for exp_col, alb_col, ratio_col in ratio_pairs
        if exp_col in df.columns and alb_col in df.columns
    }
    df = df.assign(**new_ratio_cols)

    feat_groups = [
        # (column,                band_label,    term_label)
        ('o2a_k1',             'O\u2082A',   'k\u2081'),
        ('o2a_k2',             'O\u2082A',   'k\u2082'),
        ('o2a_k3',             'O\u2082A',   'k\u2083'),
        ('exp_o2a_intercept',  'O\u2082A',   'exp'),
        ('wco2_k1',            'WCO\u2082',  'k\u2081'),
        ('wco2_k2',            'WCO\u2082',  'k\u2082'),
        ('wco2_k3',            'WCO\u2082',  'k\u2083'),
        ('exp_wco2_intercept', 'WCO\u2082',  'exp'),
        ('sco2_k1',            'SCO\u2082',  'k\u2081'),
        ('sco2_k2',            'SCO\u2082',  'k\u2082'),
        ('sco2_k3',            'SCO\u2082',  'k\u2083'),
        ('exp_sco2_intercept', 'SCO\u2082',  'exp'),
        # ── exp/alb ratio group ──────────────────────────────────────────────
        ('_exp_alb_o2a',       'O\u2082A',   'exp/alb'),
        ('_exp_alb_wco2',      'WCO\u2082',  'exp/alb'),
        ('_exp_alb_sco2',      'SCO\u2082',  'exp/alb'),
    ]
    avail = [(col, bl, tl) for col, bl, tl in feat_groups if col in df.columns]
    if not avail:
        return

    # index where the exp/alb group starts (for the separator line)
    sep_idx = next((i for i, (_, _, tl) in enumerate(avail) if tl == 'exp/alb'), None)

    band_colors  = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}
    term_hatches = {
        'k\u2081':   '',
        'k\u2082':   '///',
        'k\u2083':   'xxx',
        'exp':       '...',
        'exp/alb':   '|||',
    }
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    x = np.arange(len(avail))

    for ax, (sfc_name, sdf) in zip(axes, subsets):
        rs = []
        for col, _, _ in avail:
            m = sdf[col].notna() & sdf['cld_dist_km'].notna()
            if m.sum() > 10:
                r, _ = stats.pearsonr(sdf.loc[m, 'cld_dist_km'], sdf.loc[m, col])
            else:
                r = np.nan
            rs.append(r)

        colors  = [band_colors[bl]  for _, bl, _  in avail]
        hatches = [term_hatches[tl] for _, _,  tl in avail]
        bars = ax.bar(x, rs, color=colors, hatch=hatches, edgecolor='white', alpha=0.85)
        ax.axhline(0, color='k', lw=0.8)

        # vertical separator before exp/alb group
        if sep_idx is not None:
            ax.axvline(sep_idx - 0.5, color='dimgray', lw=1.5,
                       linestyle='--', alpha=0.7)
            ax.text(sep_idx - 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.55,
                    'exp/alb\ndivergence →', fontsize=7, color='dimgray',
                    ha='right', va='top', style='italic')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{bl}\n{tl}' for _, bl, tl in avail],
                           fontsize=7.5, rotation=40, ha='right')
        ax.set_ylabel('Pearson r with cld_dist_km', fontsize=9)
        ax.set_title(f'{sfc_name} (n={len(sdf):,})', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, r in zip(bars, rs):
            if np.isfinite(r):
                offset = 0.006 if r >= 0 else -0.006
                ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                        f'{r:.2f}', ha='center',
                        va='bottom' if r >= 0 else 'top', fontsize=6.5)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=b) for b, c in band_colors.items()]
    legend_handles += [Patch(facecolor='gray', hatch=h, label=f'({t})', alpha=0.7)
                       for t, h in term_hatches.items()]
    axes[0].legend(handles=legend_handles, fontsize=7, ncol=2, loc='lower left',
                   title='Band / Term', title_fontsize=7)
    fig.suptitle(
        'Signal hierarchy: Pearson r(cld_dist_km) — k\u2081, k\u2082, k\u2083, exp_intercept, exp/alb ratio'
        '  |  Ocean vs Land',
        fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'signal_hierarchy.png')


def plot_residual_signal_hierarchy(df: pd.DataFrame, outdir: str) -> None:
    """Companion to plot_signal_hierarchy: shows Pearson r(cld_dist, residual)
    after OLS-removing band-matched albedo + airmass + cos(SZA) from each
    spectral coefficient, and airmass + cos(SZA) + AOD from the exp/alb ratio.

    Directly answers: which coefficients carry genuine cloud-proximity signal
    independent of surface reflectance and viewing geometry?

    Key result: SCO₂ k₁/k₂/k₃ on land retain large residual r (≈ +0.20–0.28);
    all other bands collapse to near zero.  Ocean exp/alb O₂A also survives (≈ −0.16).
    """
    # ── compute exp/alb ratios ─────────────────────────────────────────────────
    ratio_assign = {
        f'_ratio_{tag}': df[exp_col] / df[alb_col].replace(0, np.nan)
        for exp_col, alb_col, tag in [
            ('exp_o2a_intercept',  'alb_o2a',  'o2a'),
            ('exp_wco2_intercept', 'alb_wco2', 'wco2'),
            ('exp_sco2_intercept', 'alb_sco2', 'sco2'),
        ]
        if exp_col in df.columns and alb_col in df.columns
    }
    df = df.assign(**ratio_assign)

    # (column, band_label, term_label, [confounder_cols])
    feat_defs = [
        ('o2a_k1',             'O\u2082A',   'k\u2081',   ['alb_o2a',  'airmass', 'mu_sza']),
        ('o2a_k2',             'O\u2082A',   'k\u2082',   ['alb_o2a',  'airmass', 'mu_sza']),
        ('o2a_k3',             'O\u2082A',   'k\u2083',   ['alb_o2a',  'airmass', 'mu_sza']),
        ('exp_o2a_intercept',  'O\u2082A',   'exp',       ['alb_o2a',  'airmass', 'mu_sza']),
        ('wco2_k1',            'WCO\u2082',  'k\u2081',   ['alb_wco2', 'airmass', 'mu_sza']),
        ('wco2_k2',            'WCO\u2082',  'k\u2082',   ['alb_wco2', 'airmass', 'mu_sza']),
        ('wco2_k3',            'WCO\u2082',  'k\u2083',   ['alb_wco2', 'airmass', 'mu_sza']),
        ('exp_wco2_intercept', 'WCO\u2082',  'exp',       ['alb_wco2', 'airmass', 'mu_sza']),
        ('sco2_k1',            'SCO\u2082',  'k\u2081',   ['alb_sco2', 'airmass', 'mu_sza']),
        ('sco2_k2',            'SCO\u2082',  'k\u2082',   ['alb_sco2', 'airmass', 'mu_sza']),
        ('sco2_k3',            'SCO\u2082',  'k\u2083',   ['alb_sco2', 'airmass', 'mu_sza']),
        ('exp_sco2_intercept', 'SCO\u2082',  'exp',       ['alb_sco2', 'airmass', 'mu_sza']),
        # exp/alb ratios — albedo already divided out
        ('_ratio_o2a',         'O\u2082A',   'exp/alb',   ['airmass', 'mu_sza', 'aod_total']),
        ('_ratio_wco2',        'WCO\u2082',  'exp/alb',   ['airmass', 'mu_sza', 'aod_total']),
        ('_ratio_sco2',        'SCO\u2082',  'exp/alb',   ['airmass', 'mu_sza', 'aod_total']),
    ]
    avail = [(col, bl, tl, ctrl) for col, bl, tl, ctrl in feat_defs
             if col in df.columns]
    if not avail:
        return

    sep_idx = next((i for i, (_, _, tl, _) in enumerate(avail) if tl == 'exp/alb'), None)

    band_colors  = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}
    term_hatches = {
        'k\u2081':   '',
        'k\u2082':   '///',
        'k\u2083':   'xxx',
        'exp':       '...',
        'exp/alb':   '|||',
    }
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    x = np.arange(len(avail))

    for ax, (sfc_name, sdf) in zip(axes, subsets):
        rs_resid = []
        for col, _, _, ctrl in avail:
            ctrl_ok = [c for c in ctrl if c in sdf.columns]
            req = [col, 'cld_dist_km'] + ctrl_ok
            m = sdf[req].notna().all(axis=1) & np.isfinite(sdf[col])
            sdf_m = sdf[m]
            if len(sdf_m) < 50:
                rs_resid.append(np.nan)
                continue
            X = np.column_stack([np.ones(len(sdf_m))]
                                + [sdf_m[c].values for c in ctrl_ok])
            y = sdf_m[col].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r, _ = stats.pearsonr(sdf_m['cld_dist_km'].values, y - X @ coef)
            rs_resid.append(r)

        colors  = [band_colors[bl]  for _, bl, _, _  in avail]
        hatches = [term_hatches[tl] for _, _,  tl, _ in avail]
        bars = ax.bar(x, rs_resid, color=colors, hatch=hatches,
                      edgecolor='white', alpha=0.85)
        ax.axhline(0, color='k', lw=0.8)

        if sep_idx is not None:
            ax.axvline(sep_idx - 0.5, color='dimgray', lw=1.5,
                       linestyle='--', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{bl}\n{tl}' for _, bl, tl, _ in avail],
                           fontsize=7.5, rotation=40, ha='right')
        ax.set_ylabel('Pearson r(cld_dist, residual)', fontsize=9)
        ax.set_title(f'{sfc_name} (n={len(sdf):,})', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, r in zip(bars, rs_resid):
            if np.isfinite(r):
                offset = 0.006 if r >= 0 else -0.006
                ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                        f'{r:.2f}', ha='center',
                        va='bottom' if r >= 0 else 'top', fontsize=6.5)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=b) for b, c in band_colors.items()]
    legend_handles += [Patch(facecolor='gray', hatch=h, label=f'({t})', alpha=0.7)
                       for t, h in term_hatches.items()]
    axes[0].legend(handles=legend_handles, fontsize=7, ncol=2, loc='lower left',
                   title='Band / Term', title_fontsize=7)
    fig.suptitle(
        'Residual signal hierarchy: r(cld_dist, residual) after removing alb + airmass + cos(SZA)\n'
        'Genuine cloud-proximity signal: SCO\u2082 k\u2081/k\u2082/k\u2083 (land) '
        'and O\u2082A exp/alb (ocean)',
        fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'residual_signal_hierarchy.png')
