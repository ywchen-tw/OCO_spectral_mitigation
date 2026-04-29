"""
ca_xco2.py
==========
XCO2 anomaly analysis functions extracted from combined_analyze.py.

Contents
--------
- plot_xco2_anomaly_correlations         Pearson r heat-map vs all key predictors
- plot_xco2_anomaly_vs_key_vars          Scatter panels vs top predictors
- plot_xco2_anomaly_vs_cld_dist_binned   Mean ± SEM bar chart by cld_dist bin
- plot_xco2_anomaly_partial              Partial correlation after OLS-removing confounders
- plot_xco2_derived_vs_cld_dist_binned   Binned profile of xco2_raw_minus_apriori etc.
- plot_xco2_derived_vs_bc_anomaly        Scatter of derived XCO2 quantities vs bc anomaly
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from .utils import _save, rolling_median_iqr, bin_by_cld_dist

logger = logging.getLogger(__name__)


def plot_xco2_anomaly_correlations(df, outdir, target_cols=None):
    """Correlation matrix heat-map: xco2 anomaly against all key predictors."""
    if target_cols is None:
        target_cols = ['xco2_bc_anomaly', 'xco2_raw_anomaly']
    predictor_cols = [
        'cld_dist_km',
        'o2a_k1', 'o2a_k2', 'o2a_k2_over_k1',
        'wco2_k1', 'wco2_k2', 'wco2_k2_over_k1',
        'sco2_k1', 'sco2_k2', 'sco2_k2_over_k1',
        'airmass', 'mu_sza', 'mu_vza',
        'alb_o2a', 'alb_wco2', 'alb_sco2',
        'aod_total', 'dp', 'dp_o2a', 'dp_sco2',
        'co2_grad_del', 'fs_rel_0', 'h2o_scale',
        'dpfrac', 'dws', 'glint_angle',
        'snr_o2a', 'snr_wco2', 'snr_sco2',
        'psfc', 'airmass_sq',
        'xco2_bc_minus_apriori', 'xco2_raw_minus_apriori',
        'xco2_bc_minus_raw',
    ]
    cols = [c for c in target_cols + predictor_cols if c in df.columns]
    corr = df[cols].corr(method='pearson')

    # Show correlations with anomaly targets only
    target_corr = corr[target_cols].drop(index=target_cols, errors='ignore')

    fig, ax = plt.subplots(figsize=(6, max(8, len(target_corr) * 0.35)))
    im = ax.imshow(target_corr.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels([c.replace('_', '\n') for c in target_cols], fontsize=9)
    ax.set_yticks(range(len(target_corr)))
    ax.set_yticklabels(target_corr.index, fontsize=8)
    for i in range(len(target_corr)):
        for j in range(len(target_cols)):
            val = target_corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < 0.6 else 'white')
    ax.set_title('Pearson r of predictors with XCO\u2082 anomalies', fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'xco2_anomaly_correlation_heatmap.png')
    return target_corr


def plot_xco2_anomaly_vs_key_vars(df, outdir, target='xco2_bc_anomaly',
                                   target_label=None, max_dist=50, n_roll=200):
    """Scatter panels: target vs top predictors."""
    if target_label is None:
        target_label = 'XCO\u2082 BC anomaly (ppm)'
    key_pairs = [
        ('cld_dist_km',    'Cloud distance (km)'),
        ('o2a_k1',         'O2-A k\u2081'),
        ('o2a_k2',         'O2-A k\u2082'),
        ('o2a_k2_over_k1', 'O2-A k\u2082/k\u2081'),
        ('wco2_k1',        'WCO\u2082 k\u2081'),
        ('wco2_k2',        'WCO\u2082 k\u2082'),
        ('sco2_k1',        'SCO\u2082 k\u2081'),
        ('sco2_k2',        'SCO\u2082 k\u2082'),
        ('airmass',        'Airmass'),
        ('aod_total',      'AOD total'),
        ('dp',             '\u0394P (hPa)'),
        ('alb_o2a',        'Albedo O2-A'),
        ('co2_grad_del',   'CO\u2082 gradient \u0394'),
        ('dpfrac',         'dp fraction'),
        ('glint_angle',    'Glint angle (deg)'),
        ('snr_o2a',        'SNR O2-A'),
    ]
    avail = [(col, lbl) for col, lbl in key_pairs if col in df.columns]

    sub = df[df['cld_dist_km'] <= max_dist] if 'cld_dist_km' in df.columns else df

    ncols = 4
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, (col, lbl) in zip(axes, avail):
        x = sub[col].values
        y = sub[target].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        if len(xm) < 10:
            ax.set_visible(False)
            continue
        ax.hexbin(xm, ym, gridsize=50, cmap='Blues', mincnt=1,
                  norm=mcolors.LogNorm())
        xs, med, q25, q75 = rolling_median_iqr(xm, ym, n_pts=n_roll)
        ax.plot(xs, med, 'r-', lw=1.5, label='median')
        ax.fill_between(xs, q25, q75, color='red', alpha=0.2)
        r, _ = stats.pearsonr(xm, ym)
        ax.set_title(f'vs {lbl}   r={r:.3f}', fontsize=9)
        ax.set_xlabel(lbl, fontsize=8)
        ax.set_ylabel(target_label, fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[len(avail):]:
        ax.set_visible(False)

    fig.suptitle(f'{target_label} vs key predictors', fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, outdir, f'{target}_vs_predictors.png')


def plot_xco2_anomaly_vs_cld_dist_binned(df, bins, labels, outdir, targets=None):
    """Mean XCO2 quantity ± SEM as a function of cloud-distance bin."""
    if targets is None:
        targets = [
            ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'C0'),
            ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'C1'),
        ]
    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    avail = [(t, lbl, c) for t, lbl, c in targets if t in df.columns]

    for col, lbl, c in avail:
        fig, ax = plt.subplots(figsize=(7, 5))
        means = df.groupby(_bin, observed=True)[col].mean().reindex(labels)
        stds  = df.groupby(_bin, observed=True)[col].std().reindex(labels)
        ns    = df.groupby(_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
        sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)

        # reference level = last non-NaN bin
        ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan
        ref_lbl = labels[-1]

        ax.bar(x, means.fillna(0).values, color=c, alpha=0.6, label='mean')
        ax.errorbar(x, means.fillna(0).values, yerr=sems.values, fmt='none',
                    color='k', capsize=4)
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        if np.isfinite(ref_val):
            ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                       label=f'mean @ {ref_lbl} km = {ref_val:.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_xlabel('Cloud distance (km)', fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(f'{lbl}: mean ± SEM by cloud-distance bin', fontsize=10)
        ax.legend(fontsize=8)
        for xi, (m, n) in enumerate(zip(means.values, ns.values)):
            if np.isfinite(m):
                ax.text(xi, m + sems.values[xi] * 1.1,
                        f'n={n:,}', ha='center', fontsize=7, color='gray')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        _save(fig, outdir, f'{col}_vs_cld_dist_binned.png')


def plot_xco2_anomaly_partial(df: pd.DataFrame, bins, labels,
                              outdir: str, target: str = 'xco2_bc_anomaly',
                              target_label: str = None) -> None:
    """Partial correlation of target with cloud distance after OLS-removing
    the main confounders: albedo (all bands), airmass, cos(SZA), AOD, ΔP,
    CO₂ gradient, H₂O scaling, dp fraction.
    """
    if target_label is None:
        target_label = 'XCO\u2082 BC anomaly (ppm)'
    if target not in df.columns:
        logger.warning(f"{target!r} not found — skipping XCO\u2082 partial plot")
        return

    confounders = [
        'alb_o2a', 'alb_wco2', 'alb_sco2',
        'airmass', 'mu_sza', 'aod_total',
        'dp', 'co2_grad_del', 'h2o_scale', 'dpfrac',
    ]
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (sfc_name, sdf) in zip(axes, subsets):
        ctrl  = [c for c in confounders if c in sdf.columns]
        req   = [target, 'cld_dist_km'] + ctrl
        m     = sdf[req].notna().all(axis=1)
        sdf_m = sdf[m].copy()
        if len(sdf_m) < 50:
            ax.set_visible(False)
            continue

        X = np.column_stack([np.ones(len(sdf_m))]
                            + [sdf_m[c].values for c in ctrl])
        y = sdf_m[target].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sdf_m['_resid'] = y - X @ coef

        r_raw, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m[target])
        r_res, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m['_resid'])

        sdf_m['_bin'] = pd.cut(sdf_m['cld_dist_km'], bins=bins,
                               labels=labels, right=False)
        means = sdf_m.groupby('_bin', observed=True)['_resid'].mean().reindex(labels)
        stds  = sdf_m.groupby('_bin', observed=True)['_resid'].std().reindex(labels)
        ns    = sdf_m.groupby('_bin', observed=True)['_resid'].count().reindex(labels).fillna(0).astype(int)
        sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
        xp = np.arange(len(labels))

        ax.fill_between(xp, (means - stds).values, (means + stds).values,
                        color='purple', alpha=0.15, label='\u00b1 1 std')
        ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                    capsize=4, color='purple', lw=1.5,
                    label=f'mean\u00b1SEM  r={r_res:.3f}')
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(f'{target_label} residual', fontsize=9)
        ax.set_title(
            f'{sfc_name} (n={len(sdf_m):,})\n'
            f'r_raw={r_raw:.3f} \u2192 r_resid={r_res:.3f}',
            fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        # sample counts — fixed position just below x-axis (axes-fraction y)
        for xi, n in enumerate(ns.values):
            if n > 0:
                ax.text(xi, -0.14, f'n={n:,}',
                        ha='center', va='top', fontsize=6, color='gray',
                        transform=ax.get_xaxis_transform(), clip_on=False)

    fig.suptitle(
        f'{target_label}: partial correlation with cloud distance\n'
        'after removing albedo + airmass + cos(SZA) + AOD + \u0394P + CO\u2082_grad + H\u2082O',
        fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save(fig, outdir, f'{target}_partial_vs_cld_dist.png')


# ── Section 5b: xco2_raw_minus_apriori & xco2_raw_minus_strong_idp analyses ──

_XCO2_DERIVED_TARGETS = [
    ('xco2_raw_minus_apriori',
     'XCO\u2082 raw \u2212 a priori (ppm)',
     'C2'),
    ('xco2_raw_minus-xco2_strong_idp_minus',
     'XCO\u2082 raw \u2212 strong IDP (ppm)',
     'C3'),
]


def plot_xco2_derived_vs_cld_dist_binned(df: pd.DataFrame, bins, labels,
                                         outdir: str) -> None:
    """Mean ± SEM bar chart of xco2_raw_minus_apriori and
    xco2_raw_minus-xco2_strong_idp_minus as a function of cloud-distance bin.

    Outputs
    -------
    xco2_raw_minus_apriori_vs_cld_dist_binned.png
    xco2_raw_minus_strong_idp_vs_cld_dist_binned.png
    """
    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    for col, lbl, c in _XCO2_DERIVED_TARGETS:
        if col not in df.columns:
            logger.warning(f"Column {col!r} not found — skipping binned profile")
            continue

        means = df.groupby(_bin, observed=True)[col].mean().reindex(labels)
        stds  = df.groupby(_bin, observed=True)[col].std().reindex(labels)
        ns    = df.groupby(_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
        sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)

        ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x, means.fillna(0).values, color=c, alpha=0.6, label='mean')
        ax.errorbar(x, means.fillna(0).values, yerr=sems.values,
                    fmt='none', color='k', capsize=4)
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        if np.isfinite(ref_val):
            ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                       label=f'mean @ {labels[-1]} km = {ref_val:.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.set_xlabel('Cloud distance (km)', fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(f'{lbl}: mean \u00b1 SEM by cloud-distance bin', fontsize=10)
        ax.legend(fontsize=8)
        for xi, (m, n) in enumerate(zip(means.values, ns.values)):
            if np.isfinite(m):
                ax.text(xi, m + sems.values[xi] * 1.1,
                        f'n={n:,}', ha='center', fontsize=7, color='gray')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()

        safe = col.replace('-', '_minus_').replace('__', '_')
        _save(fig, outdir, f'{safe}_vs_cld_dist_binned.png')


def plot_xco2_derived_vs_bc_anomaly(df: pd.DataFrame, bins, labels,
                                    outdir: str, max_dist: float = 50,
                                    n_roll: int = 80,
                                    y_col: str = 'xco2_bc_anomaly',
                                    y_label: str = None) -> None:
    """Scatter of xco2_raw_minus_apriori / xco2_raw_minus-xco2_strong_idp_minus
    vs *y_col* (default: xco2_bc_anomaly).

    Panel A  — hexbin scatter colored by log-density, rolling median overlaid.
    Panel B  — mean ± SEM per cld_dist bin, with one line per bin in Panel A
               to show how the relationship shifts with proximity to cloud.

    Outputs (per derived column)
    ----------------------------
    xco2_raw_minus_apriori_vs_{y_col}.png
    xco2_raw_minus_strong_idp_vs_{y_col}.png
    """
    if y_label is None:
        y_label = y_col.replace('_', ' ')
    if y_col not in df.columns:
        logger.warning(f"{y_col!r} not found — skipping derived vs anomaly plots")
        return

    sub = df[df['cld_dist_km'] <= max_dist].copy() if 'cld_dist_km' in df.columns else df.copy()
    _bin = bin_by_cld_dist(sub, bins, labels)

    bin_colors = plt.cm.get_cmap('plasma', len(labels))

    for col, lbl, _ in _XCO2_DERIVED_TARGETS:
        if col not in df.columns:
            logger.warning(f"Column {col!r} not found — skipping vs {y_col} plot")
            continue

        mask = sub[col].notna() & sub[y_col].notna()
        xv = sub.loc[mask, col].values.astype(float)
        yv = sub.loc[mask, y_col].values.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Panel A: overall hexbin + rolling median ──────────────────────────
        ax = axes[0]
        if len(xv) > 0:
            hb = ax.hexbin(xv, yv, gridsize=60, cmap='YlOrRd',
                           mincnt=1, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='count')
            xs, med, q25, q75 = rolling_median_iqr(xv, yv, n_pts=n_roll)
            ax.plot(xs, med, 'k-', lw=2, label='rolling median')
            ax.fill_between(xs, q25, q75, color='k', alpha=0.15, label='IQR')
            r, p = stats.pearsonr(xv, yv)
            ax.set_title(f'Overall  r={r:.3f}  p={p:.2e}', fontsize=10)
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── Panel B: per-cld_dist-bin mean ± SEM ─────────────────────────────
        ax2 = axes[1]
        bin_means_x, bin_means_y, bin_sems_y = [], [], []
        for bi, lbl_b in enumerate(labels):
            m = (_bin == lbl_b) & mask
            if m.sum() < 5:
                bin_means_x.append(np.nan)
                bin_means_y.append(np.nan)
                bin_sems_y.append(np.nan)
                continue
            xs_b = sub.loc[m, col].values.astype(float)
            ys_b = sub.loc[m, y_col].values.astype(float)
            # scatter colored by bin
            ax2.scatter(xs_b, ys_b, s=2, alpha=0.25,
                        color=bin_colors(bi), label=None)
            # rolling median per bin
            if len(xs_b) >= 10:
                order = np.argsort(xs_b)
                xs_s, med_s, _, _ = rolling_median_iqr(xs_b[order], ys_b[order],
                                                        n_pts=min(30, len(xs_b) // 5))
                ax2.plot(xs_s, med_s, '-', color=bin_colors(bi), lw=2,
                         label=f'{lbl_b} km')
            bin_means_x.append(np.nanmean(xs_b))
            bin_means_y.append(np.nanmean(ys_b))
            bin_sems_y.append(np.nanstd(ys_b) / np.sqrt(len(ys_b)))

        ax2.errorbar(bin_means_x, bin_means_y, yerr=bin_sems_y,
                     fmt='D', color='k', ms=6, capsize=4, zorder=10,
                     label='bin mean \u00b1 SEM')
        ax2.set_xlabel(lbl, fontsize=9)
        ax2.set_ylabel(y_label, fontsize=9)
        ax2.set_title('Grouped by cloud-distance bin', fontsize=10)
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(alpha=0.3)

        y_safe = y_col.replace(' ', '_')
        fig.suptitle(
            f'{lbl} vs {y_label}\n'
            f'(left: all soundings \u2264{max_dist} km from cloud; '
            f'right: colored by cld_dist bin)',
            fontsize=11)
        fig.tight_layout()

        safe = col.replace('-', '_minus_').replace('__', '_')
        _save(fig, outdir, f'{safe}_vs_{y_safe}.png')


# ── Part 3: XCO2 sign-split analyses ──────────────────────────────────────────

def plot_xco2_sign_comparison(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    bins, labels,
    outdir: str,
    split_col: str = 'xco2_bc_anomaly',
    split_label: str = None,
) -> None:
    """Overlay binned-mean profiles of ref-corrected delta variables for
    positive vs negative *split_col* subsets.

    Four panels (one per delta variable type: dk1, dk2, dexp, dalb), each
    showing all three bands (o2a, wco2, sco2).  Solid lines = pos subset,
    dashed lines = neg subset.

    Saved to  {outdir}/xco2_sign_{split_col}/sign_comparison.png
    """
    import os

    if split_label is None:
        split_label = split_col.replace('_', ' ')

    _DELTA_VARS = [
        ('dk1',  'Δk\u2081'),
        ('dk2',  'Δk\u2082'),
        ('dexp', 'Δexp intercept'),
        ('dalb', 'Δalbedo'),
    ]
    _BANDS = [
        ('o2a',  'O2-A',   'tab:blue'),
        ('wco2', 'WCO\u2082', 'tab:orange'),
        ('sco2', 'SCO\u2082', 'tab:green'),
    ]

    # Check that at least one delta column exists in either subset
    all_delta_cols = [f'{v}_{b}' for v, _ in _DELTA_VARS for b, _, _ in _BANDS]
    if not any(c in df_pos.columns for c in all_delta_cols):
        logger.warning("No delta columns found — skipping sign comparison plot")
        return

    sign_dir = os.path.join(outdir, f'xco2_sign_{split_col}')
    os.makedirs(sign_dir, exist_ok=True)

    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (var_prefix, var_label) in zip(axes, _DELTA_VARS):
        plotted = False
        for band_suffix, band_label, color in _BANDS:
            col = f'{var_prefix}_{band_suffix}'

            for subset_df, linestyle, sign_label in [
                (df_pos, '-',  'pos'),
                (df_neg, '--', 'neg'),
            ]:
                if col not in subset_df.columns:
                    continue
                _bin = bin_by_cld_dist(subset_df, bins, labels)
                means = subset_df.groupby(_bin, observed=True)[col].mean().reindex(labels)
                ns    = subset_df.groupby(_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
                stds  = subset_df.groupby(_bin, observed=True)[col].std().reindex(labels)
                sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)

                finite = means.notna()
                if finite.sum() < 2:
                    continue

                ax.plot(x[finite], means.values[finite],
                        linestyle=linestyle, color=color, lw=2,
                        label=f'{band_label} {sign_label}')
                ax.fill_between(x[finite],
                                (means - sems).values[finite],
                                (means + sems).values[finite],
                                color=color, alpha=0.12)
                plotted = True

        ax.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(var_label, fontsize=9)
        ax.set_title(f'{var_label} — pos (solid) vs neg (dashed)', fontsize=10)
        if plotted:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Ref-corrected \u0394variables: {split_label} > 0  vs  < 0\n'
        'Solid = positive subset \u00b7 Dashed = negative subset',
        fontsize=12)
    fig.tight_layout()
    _save(fig, sign_dir, 'sign_comparison.png')


def run_xco2_sign_analysis(
    df: pd.DataFrame,
    bins, labels,
    sfc_outdir: str,
    run_ref: bool = False,
    ref_pairs=None,
    r25_pairs=None,
    split_col: str = 'xco2_bc_anomaly',
    split_label: str = None,
) -> None:
    """Split df by sign of *split_col* and run core plots for each half.

    Subdirectory layout::

        {sfc_outdir}/xco2_sign_{split_col}/
            pos/    (split_col >= 0)
            neg/    (split_col <  0)
            sign_comparison.png

    Parameters
    ----------
    df          : surface-filtered DataFrame (already ocean or land)
    bins/labels : cloud-distance bin edges and labels
    sfc_outdir  : per-surface output directory (e.g. …/ocean)
    run_ref     : if True and ref_* columns are present, run R0–R7 per subset
    ref_pairs   : _REF_PAIRS from ca_ref_corrected (pass-through)
    r25_pairs   : _R25_PAIRS from ca_ref_corrected (unused currently)
    split_col   : column to split on (default: xco2_bc_anomaly)
    split_label : human-readable label for split_col (default: derived from col name)
    """
    import os

    # lazy imports to avoid circular dependencies at module level
    from .k_coeff import (
        plot_distributions_vs_cld_dist, plot_k1_k2_binned_profile,
        plot_k1_k2_vs_cld_dist, plot_k2_over_k1_vs_cld_dist, plot_k1_k2_joint,
    )
    from .albedo import (
        plot_alb_vs_exp_intercept, plot_alb_vs_exp_intercept_cross,
        plot_intercept_binned_profile, plot_alb_binned_profile,
    )

    if split_label is None:
        split_label = split_col.replace('_', ' ')

    if split_col not in df.columns:
        logger.warning(f"{split_col!r} not found — skipping sign-split analysis")
        return

    sign_dir = os.path.join(sfc_outdir, f'xco2_sign_{split_col}')
    subsets = [
        ('pos', df[df[split_col] >= 0]),
        ('neg', df[df[split_col] <  0]),
    ]

    _MIN_N = 500

    for sign_label, sdf in subsets:
        n = len(sdf)
        logger.info(f"  xco2_sign/{sign_label}: {n:,} soundings")
        if n < _MIN_N:
            logger.warning(f"  {sign_label} subset too small ({n} < {_MIN_N}) — skipping")
            continue

        sub_outdir = os.path.join(sign_dir, sign_label)
        os.makedirs(sub_outdir, exist_ok=True)

        # ── core plot suite ───────────────────────────────────────────────────
        logger.info(f"  [{sign_label}] distributions box-plots …")
        plot_distributions_vs_cld_dist(sdf, bins, labels, sub_outdir)

        logger.info(f"  [{sign_label}] exp_intercept binned profiles …")
        plot_intercept_binned_profile(sdf, bins, labels, sub_outdir)

        logger.info(f"  [{sign_label}] albedo vs exp_intercept …")
        plot_alb_vs_exp_intercept(sdf, sub_outdir)
        plot_alb_vs_exp_intercept_cross(sdf, sub_outdir)

        logger.info(f"  [{sign_label}] k1/k2 profiles and scatter …")
        plot_k1_k2_binned_profile(sdf, bins, labels, sub_outdir)
        plot_k1_k2_vs_cld_dist(sdf, sub_outdir)

        logger.info(f"  [{sign_label}] k2/k1 ratio …")
        plot_k2_over_k1_vs_cld_dist(sdf, sub_outdir)

        logger.info(f"  [{sign_label}] k1 vs k2 joint scatter …")
        plot_k1_k2_joint(sdf, sub_outdir)

        logger.info(f"  [{sign_label}] albedo binned profiles …")
        plot_alb_binned_profile(sdf, bins, labels, sub_outdir)

        logger.info(f"  [{sign_label}] XCO2 anomaly binned profiles …")
        plot_xco2_anomaly_vs_cld_dist_binned(sdf, bins, labels, sub_outdir)
        plot_xco2_anomaly_correlations(sdf, sub_outdir)

        # ── ref-corrected R0–R7 ───────────────────────────────────────────────
        if run_ref:
            try:
                from .ref_corrected import (
                    _REF_PAIRS, _has_ref_data, add_ref_anomalies,
                    plot_ref_diff_vs_cld_dist, plot_ref_coverage_bias,
                    plot_ref_std_profiles, plot_ref_corrected_profiles,
                    plot_ref_zscore_profiles, plot_ref_signal_hierarchy,
                    plot_ref_alb_decoupled_exp, plot_obs_vs_ref_scatter,
                )
            except ImportError:
                logger.warning("ca_ref_corrected not available — skipping ref plots")
                run_ref = False

            if run_ref and _has_ref_data(sdf):
                pairs = ref_pairs if ref_pairs is not None else _REF_PAIRS
                ref_sub_outdir = os.path.join(sub_outdir, 'ref_corrected')
                os.makedirs(ref_sub_outdir, exist_ok=True)
                sdf_r = add_ref_anomalies(sdf)
                logger.info(f"  [{sign_label}] ref R0–R7 …")
                plot_ref_diff_vs_cld_dist(sdf_r, ref_sub_outdir, pairs=pairs)
                plot_ref_coverage_bias(sdf_r, bins, labels, ref_sub_outdir, pairs=pairs)
                plot_ref_std_profiles(sdf_r, bins, labels, ref_sub_outdir, pairs=pairs)
                plot_ref_corrected_profiles(sdf_r, bins, labels, ref_sub_outdir, pairs=pairs)
                plot_ref_zscore_profiles(sdf_r, bins, labels, ref_sub_outdir, pairs=pairs)
                plot_ref_signal_hierarchy(sdf_r, ref_sub_outdir, pairs=pairs)
                plot_ref_alb_decoupled_exp(sdf_r, bins, labels, ref_sub_outdir, pairs=pairs)
                plot_obs_vs_ref_scatter(sdf_r, ref_sub_outdir, pairs=pairs)

    # ── comparison overlay (needs ref delta columns) ──────────────────────────
    sdf_pos = subsets[0][1]
    sdf_neg = subsets[1][1]
    if len(sdf_pos) >= _MIN_N and len(sdf_neg) >= _MIN_N:
        logger.info("  sign_comparison overlay …")
        plot_xco2_sign_comparison(sdf_pos, sdf_neg, bins, labels, sfc_outdir,
                                   split_col=split_col, split_label=split_label)


# ── Per-target orchestrator ────────────────────────────────────────────────────

# (col, human label, folder name)
_XCO2_TARGET_CONFIG = [
    ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'xco2_bc_anomaly'),
    ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'xco2_raw_anomaly'),
    ('xco2_bc',          'XCO\u2082 BC (ppm)',           'xco2_bc'),
    ('xco2_raw',         'XCO\u2082 raw (ppm)',          'xco2_raw'),
]


def run_xco2_target_analysis(df: pd.DataFrame, bins, labels,
                              base_outdir: str,
                              target: str,
                              target_label: str) -> None:
    """Run the full XCO2 plot suite for a single target column.

    Saves all figures into  {base_outdir}/{target}/
    """
    import os
    if target not in df.columns:
        logger.warning(f"{target!r} not in DataFrame — skipping")
        return
    outdir = os.path.join(base_outdir, target)
    os.makedirs(outdir, exist_ok=True)

    logger.info(f"  [{target}] correlation heatmap …")
    plot_xco2_anomaly_correlations(df, outdir, target_cols=[target])

    logger.info(f"  [{target}] scatter vs predictors …")
    plot_xco2_anomaly_vs_key_vars(df, outdir, target=target,
                                   target_label=target_label)

    logger.info(f"  [{target}] binned profile vs cld_dist …")
    plot_xco2_anomaly_vs_cld_dist_binned(
        df, bins, labels, outdir,
        targets=[(target, target_label, 'C0')])

    logger.info(f"  [{target}] partial correlation …")
    plot_xco2_anomaly_partial(df, bins, labels, outdir,
                               target=target, target_label=target_label)
