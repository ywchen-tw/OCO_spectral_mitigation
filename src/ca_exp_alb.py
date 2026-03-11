"""
ca_exp_alb.py
=============
exp_intercept / albedo analysis functions extracted from combined_analyze.py.

Contents
--------
- plot_alb_vs_exp_intercept              Within-band scatter + rolling median
- plot_alb_vs_exp_intercept_cross        3×3 cross-band scatter matrix
- plot_intercept_binned_profile          Binned mean ± SEM/std vs cld_dist
- plot_exp_intercept_interband_coherence Pairwise scatter colored by cld_dist
- plot_alb_exp_divergence                % change from far-cloud ref; exp/alb ratio divergence
- plot_exp_intercept_albedo_residuals    OLS residuals after alb+airmass+SZA removal
- plot_exp_alb_ratio_residuals           OLS residuals of exp/alb after airmass+SZA+AOD removal
- plot_alb_binned_profile                Albedo binned profiles
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from ca_utils import _save, rolling_median_iqr, bin_by_cld_dist

logger = logging.getLogger(__name__)


def plot_alb_vs_exp_intercept(df, outdir, n_roll=200):
    """Scatter (hexbin) + rolling median of albedo vs exp_intercept for each band."""
    bands = [
        ('alb_o2a',  'exp_o2a_intercept',  'O2-A',   'C0'),
        ('alb_wco2', 'exp_wco2_intercept', 'WCO\u2082', 'C1'),
        ('alb_sco2', 'exp_sco2_intercept', 'SCO\u2082', 'C2'),
    ]
    avail = [(alb, eint, nm, c) for alb, eint, nm, c in bands
             if alb in df.columns and eint in df.columns]
    if not avail:
        logger.warning("No alb/exp_intercept column pairs found — skipping plot")
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(6 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]

    for ax, (alb_col, eint_col, nm, col) in zip(axes, avail):
        x = df[alb_col].values
        y = df[eint_col].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]

        hb = ax.hexbin(xm, ym, gridsize=60, cmap='YlOrRd',
                       mincnt=1, norm=mcolors.LogNorm())
        plt.colorbar(hb, ax=ax, label='count')

        xs, med, q25, q75 = rolling_median_iqr(xm, ym, n_pts=n_roll)
        ax.plot(xs, med, color=col, lw=1.5, label='rolling median')
        ax.fill_between(xs, q25, q75, color=col, alpha=0.25, label='IQR')

        r, _ = stats.pearsonr(xm, ym)
        ax.set_title(f'{nm}: albedo vs exp_intercept   r={r:.3f}', fontsize=10)
        ax.set_xlabel(f'{nm} albedo', fontsize=9)
        ax.set_ylabel(f'{nm} exp_intercept', fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle('Albedo vs exp_intercept — all bands', fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'alb_vs_exp_intercept.png')


def plot_alb_vs_exp_intercept_cross(df, outdir, n_roll=200):
    """3×3 cross-band scatter: each exp_intercept (rows) vs each albedo (cols)."""
    intercepts = [
        ('exp_o2a_intercept',  'O2-A exp_intercept'),
        ('exp_wco2_intercept', 'WCO\u2082 exp_intercept'),
        ('exp_sco2_intercept', 'SCO\u2082 exp_intercept'),
    ]
    albedos = [
        ('alb_o2a',  'O2-A albedo'),
        ('alb_wco2', 'WCO\u2082 albedo'),
        ('alb_sco2', 'SCO\u2082 albedo'),
    ]
    row_avail = [(col, lbl) for col, lbl in intercepts if col in df.columns]
    col_avail = [(col, lbl) for col, lbl in albedos  if col in df.columns]
    if not row_avail or not col_avail:
        logger.warning("No exp_intercept/albedo columns found for cross-band plot — skipping")
        return

    nrows, ncols = len(row_avail), len(col_avail)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, (int_col, int_lbl) in enumerate(row_avail):
        for ci, (alb_col, alb_lbl) in enumerate(col_avail):
            ax = axes[ri, ci]
            x = df[alb_col].values
            y = df[int_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            xm, ym = x[mask], y[mask]
            if len(xm) < 10:
                ax.set_visible(False)
                continue

            hb = ax.hexbin(xm, ym, gridsize=55, cmap='YlOrRd',
                           mincnt=1, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='count')

            xs, med, q25, q75 = rolling_median_iqr(xm, ym, n_pts=n_roll)
            color = f'C{ri}'
            ax.plot(xs, med, color=color, lw=1.5, label='rolling median')
            ax.fill_between(xs, q25, q75, color=color, alpha=0.25, label='IQR')

            r, _ = stats.pearsonr(xm, ym)
            ax.set_title(f'r={r:.3f}', fontsize=9)
            ax.set_xlabel(alb_lbl, fontsize=9)
            ax.set_ylabel(int_lbl, fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle('exp_intercept vs albedo — cross-band (rows: intercept band, cols: albedo band)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'alb_vs_exp_intercept_cross.png')


def plot_intercept_binned_profile(df, bins, labels, outdir):
    """Mean ± SEM (bars) / ± std (shading) of the spectral exp_intercept per band."""
    bands = [
        ('exp_o2a_intercept',  'O2-A',   'C0'),
        ('exp_wco2_intercept', 'WCO\u2082', 'C1'),
        ('exp_sco2_intercept', 'SCO\u2082', 'C2'),
    ]
    avail = [(col, nm, c) for col, nm, c in bands if col in df.columns]
    if not avail:
        return

    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, color) in zip(axes, avail):
        means = df.groupby(_bin, observed=True)[col].mean().reindex(labels)
        stds  = df.groupby(_bin, observed=True)[col].std().reindex(labels)
        ns    = df.groupby(_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
        sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
        ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan

        ax.fill_between(x, (means - stds).values, (means + stds).values,
                        color=color, alpha=0.15, label='\u00b1 1 std')
        ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                    capsize=4, color=color, lw=1.5, label='mean \u00b1 SEM')
        if np.isfinite(ref_val):
            ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                       label=f'mean @ {labels[-1]} km = {ref_val:.4f}')
        ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(f'{nm} exp_intercept', fontsize=9)
        ax.set_title(f'{nm} exp_intercept: mean \u00b1 SEM (bars) / \u00b1 std (shading)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        finite_means = means.dropna()
        finite_stds  = stds.dropna()
        if finite_means.size:
            spread = max(finite_stds.max() * 1.5, (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
            ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.tight_layout()
    _save(fig, outdir, 'exp_intercept_binned_profile.png')


def plot_exp_intercept_interband_coherence(df: pd.DataFrame, outdir: str) -> None:
    """Pairwise scatter of exp_intercepts across the three bands, colored by
    cld_dist_km.  Quantifies the shared vs band-specific cloud signal.

    Layout: rows = ocean / land;  columns = (O\u2082A vs WCO\u2082), (O\u2082A vs SCO\u2082), (WCO\u2082 vs SCO\u2082).
    """
    pairs = [
        ('exp_o2a_intercept',  'exp_wco2_intercept', 'O\u2082A',  'WCO\u2082'),
        ('exp_o2a_intercept',  'exp_sco2_intercept', 'O\u2082A',  'SCO\u2082'),
        ('exp_wco2_intercept', 'exp_sco2_intercept', 'WCO\u2082', 'SCO\u2082'),
    ]
    avail = [(a, b, na, nb) for a, b, na, nb in pairs
             if a in df.columns and b in df.columns]
    if not avail:
        logger.warning("exp_intercept columns missing — skipping interband coherence plot")
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    nrows, ncols = len(subsets), len(avail)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, (sfc_name, sdf) in enumerate(subsets):
        for ci, (ac, bc, na, nb) in enumerate(avail):
            ax = axes[ri, ci]
            m = sdf[ac].notna() & sdf[bc].notna() & sdf['cld_dist_km'].notna()
            x = sdf.loc[m, ac].values
            y = sdf.loc[m, bc].values
            c = sdf.loc[m, 'cld_dist_km'].values
            if len(x) < 10:
                ax.set_visible(False)
                continue
            r, _ = stats.pearsonr(x, y)
            sc = ax.scatter(x, y, c=c, cmap='viridis_r', s=2, alpha=0.3,
                            vmin=0, vmax=50)
            plt.colorbar(sc, ax=ax, label='cld_dist_km')
            ax.set_xlabel(f'{na} exp_intercept', fontsize=9)
            ax.set_ylabel(f'{nb} exp_intercept', fontsize=9)
            ax.set_title(f'{sfc_name}: {na} vs {nb}  r={r:.3f}', fontsize=9)

    fig.suptitle(
        'exp_intercept inter-band coherence (color = cloud distance km)',
        fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'exp_intercept_interband_coherence.png')


def plot_alb_exp_divergence(df: pd.DataFrame, bins, labels, outdir: str) -> None:
    """Compare how albedo and exp_intercept each change with cloud distance,
    and show their ratio (exp/alb) to isolate the non-albedo component.

    Layout per band (3 rows × 2 surface-type columns):
      • Top pair  — % change from far-cloud reference (alb vs exp, same axes)
      • Bottom    — % change in the exp/alb ratio (divergence from albedo)

    Ocean finding: exp/alb ratio rises near clouds → cloud-edge scattered light
    inflates the spectral baseline beyond what albedo predicts.
    Land finding : exp/alb ratio crashes near clouds despite stable/elevated
    albedo → anomalous suppression of exp_intercept independent of surface reflectance.
    """
    bands = [
        ('alb_o2a',  'exp_o2a_intercept',  'O\u2082A',   'C0'),
        ('alb_wco2', 'exp_wco2_intercept', 'WCO\u2082',  'C1'),
        ('alb_sco2', 'exp_sco2_intercept', 'SCO\u2082',  'C2'),
    ]
    avail = [(alb, exp, nm, c) for alb, exp, nm, c in bands
             if alb in df.columns and exp in df.columns]
    if not avail:
        logger.warning("albedo or exp_intercept columns missing — skipping divergence plot")
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    # ── figure: % change profiles ──────────────────────────────────────────────
    fig1, axes1 = plt.subplots(len(avail), 2,
                               figsize=(13, 4.5 * len(avail)))
    if len(avail) == 1:
        axes1 = axes1[np.newaxis, :]

    # ── figure: exp/alb ratio % change ────────────────────────────────────────
    fig2, axes2 = plt.subplots(len(avail), 2,
                               figsize=(13, 4 * len(avail)))
    if len(avail) == 1:
        axes2 = axes2[np.newaxis, :]

    xp = np.arange(len(labels))

    for row, (alb_col, exp_col, nm, col) in enumerate(avail):
        for ci, (sfc_name, sdf) in enumerate(subsets):
            _sdf_bin = pd.cut(sdf['cld_dist_km'], bins=bins,
                              labels=labels, right=False)

            def _binned(c, _b=_sdf_bin, _s=sdf):
                m = _s.groupby(_b, observed=True)[c].mean().reindex(labels)
                s = _s.groupby(_b, observed=True)[c].std().reindex(labels)
                n = _s.groupby(_b, observed=True)[c].count().reindex(labels).fillna(0).astype(int)
                sem = (s / np.sqrt(n.replace(0, np.nan))).fillna(0)
                return m, s, sem

            alb_m, alb_s, alb_sem = _binned(alb_col)
            exp_m, exp_s, exp_sem = _binned(exp_col)

            # % change from the last (far-cloud) bin
            alb_ref = alb_m.iloc[-1] if np.isfinite(alb_m.iloc[-1]) else np.nan
            exp_ref = exp_m.iloc[-1] if np.isfinite(exp_m.iloc[-1]) else np.nan

            alb_pct = (alb_m - alb_ref) / abs(alb_ref) * 100
            exp_pct = (exp_m - exp_ref) / abs(exp_ref) * 100
            alb_sem_pct = alb_sem / abs(alb_ref) * 100
            exp_sem_pct = exp_sem / abs(exp_ref) * 100

            # exp / alb ratio % change from far-cloud reference
            ratio_m = exp_m / alb_m
            ratio_ref = ratio_m.iloc[-1]
            ratio_pct = (ratio_m - ratio_ref) / abs(ratio_ref) * 100
            # propagate SEM through ratio
            ratio_sem_pct = np.sqrt(
                (exp_sem / exp_m.abs()) ** 2 + (alb_sem / alb_m.abs()) ** 2
            ) * abs(ratio_pct) / 100 * 100   # approx in % units

            r_alb = stats.pearsonr(sdf[alb_col].dropna(),
                                   sdf['cld_dist_km'].reindex(sdf[alb_col].dropna().index))[0]
            r_exp = stats.pearsonr(sdf[exp_col].dropna(),
                                   sdf['cld_dist_km'].reindex(sdf[exp_col].dropna().index))[0]

            # ── plot 1: % change comparison ────────────────────────────────────
            ax = axes1[row, ci]
            ax.errorbar(xp, alb_pct.values, yerr=alb_sem_pct.values, fmt='s-',
                        capsize=3, color=col, lw=1.8, alpha=0.7,
                        label=f'albedo  r={r_alb:.2f}')
            ax.errorbar(xp, exp_pct.values, yerr=exp_sem_pct.values, fmt='o--',
                        capsize=3, color=col, lw=1.8,
                        label=f'exp_int r={r_exp:.2f}')
            ax.fill_between(xp, alb_pct.values, exp_pct.values,
                            color=col, alpha=0.12,
                            label='gap (exp − alb)')
            ax.axhline(0, color='gray', lw=0.8, linestyle='--')
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel('% change from 30–50 km ref', fontsize=9)
            ax.set_title(f'{nm} — {sfc_name}: alb vs exp_intercept', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

            # ── plot 2: exp/alb ratio % change ────────────────────────────────
            ax2 = axes2[row, ci]
            ax2.errorbar(xp, ratio_pct.values, yerr=ratio_sem_pct.values,
                         fmt='o-', capsize=3, color=col, lw=1.8,
                         label='(exp/alb) % change')
            ax2.axhline(0, color='gray', lw=0.8, linestyle='--')
            # annotate near-cloud value
            near_val = ratio_pct.iloc[0]
            if np.isfinite(near_val):
                ax2.annotate(f'{near_val:+.1f}%',
                             xy=(0, near_val),
                             xytext=(0.5, near_val + np.sign(near_val) * 3),
                             fontsize=8, color=col,
                             arrowprops=dict(arrowstyle='->', color=col, lw=1))
            ax2.set_xticks(xp)
            ax2.set_xticklabels(labels, rotation=30, fontsize=8)
            ax2.set_xlabel('Cloud distance (km)', fontsize=9)
            ax2.set_ylabel('exp/alb ratio: % change from ref', fontsize=9)
            ax2.set_title(f'{nm} — {sfc_name}: exp/alb divergence', fontsize=9)
            ax2.legend(fontsize=7)
            ax2.grid(axis='y', alpha=0.3)

    fig1.suptitle(
        'Albedo vs exp_intercept: % change from far-cloud reference (30–50 km)',
        fontsize=11)
    fig1.tight_layout()
    _save(fig1, outdir, 'alb_exp_pct_change_vs_cld_dist.png')

    fig2.suptitle(
        'exp/alb ratio % change from far-cloud reference — isolates non-albedo cloud effect',
        fontsize=11)
    fig2.tight_layout()
    _save(fig2, outdir, 'alb_exp_ratio_divergence_vs_cld_dist.png')


def plot_exp_intercept_albedo_residuals(df: pd.DataFrame, bins, labels,
                                        outdir: str) -> None:
    """OLS-remove albedo + airmass + cos(SZA) from each exp_intercept, then plot
    residuals vs cloud distance (binned mean ± SEM/std).

    Compares r_raw vs r_residual to show how much of the cloud-distance signal
    survives after albedo confound removal.  Ocean and land are shown separately.
    """
    bands = [
        ('exp_o2a_intercept',  'alb_o2a',  'O\u2082A',  'C0'),
        ('exp_wco2_intercept', 'alb_wco2', 'WCO\u2082', 'C1'),
        ('exp_sco2_intercept', 'alb_sco2', 'SCO\u2082', 'C2'),
    ]
    avail = [(ei, alb, nm, c) for ei, alb, nm, c in bands
             if ei in df.columns and alb in df.columns]
    if not avail:
        logger.warning("exp_intercept or albedo columns missing — skipping residual plot")
        return

    control_cols = ['airmass', 'mu_sza']
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(len(avail), 2, figsize=(13, 4.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (ei_col, alb_col, nm, col) in enumerate(avail):
        for ci, (sfc_name, sdf) in enumerate(subsets):
            ax = axes[row, ci]
            ctrl = [c for c in control_cols if c in sdf.columns]
            X_cols = [alb_col] + ctrl
            m = sdf[[ei_col, 'cld_dist_km'] + X_cols].notna().all(axis=1)
            sdf_m = sdf[m].copy()
            if len(sdf_m) < 50:
                ax.set_visible(False)
                continue

            # OLS: exp_intercept ~ alb + airmass + mu_sza
            X = np.column_stack([np.ones(len(sdf_m))]
                                + [sdf_m[c].values for c in X_cols])
            y = sdf_m[ei_col].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            sdf_m['_resid'] = y - X @ coef

            r_raw, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m[ei_col])
            r_res, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m['_resid'])

            # binned residual profile
            sdf_m['_bin'] = pd.cut(sdf_m['cld_dist_km'], bins=bins,
                                   labels=labels, right=False)
            means = sdf_m.groupby('_bin', observed=True)['_resid'].mean().reindex(labels)
            stds  = sdf_m.groupby('_bin', observed=True)['_resid'].std().reindex(labels)
            ns    = sdf_m.groupby('_bin', observed=True)['_resid'].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            xp = np.arange(len(labels))

            ax.fill_between(xp, (means - stds).values, (means + stds).values,
                            color=col, alpha=0.15, label='\u00b1 1 std')
            ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=col, lw=1.5,
                        label=f'mean\u00b1SEM  r={r_res:.3f}')
            ax.axhline(0, color='gray', lw=0.8, linestyle='--')
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'{nm} exp_int residual', fontsize=9)
            ax.set_title(
                f'{nm} — {sfc_name}  |  r_raw={r_raw:.3f} \u2192 r_resid={r_res:.3f}',
                fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size:
                spread = max(finite_stds.max() * 1.5,
                             (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
                ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.suptitle(
        'exp_intercept residuals after removing albedo + airmass + cos(SZA)',
        fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'exp_intercept_albedo_residuals.png')


def plot_exp_alb_ratio_residuals(df: pd.DataFrame, bins, labels,
                                 outdir: str) -> None:
    """OLS-remove airmass + cos(SZA) + AOD from the exp/alb ratio (albedo
    already divided out), then plot residuals vs cloud distance.

    Ocean O₂A: ~67% of the raw signal survives → genuine cloud-edge photon
    enhancement persists even after geometric/aerosol correction.
    Land all bands: sign flip (r_raw >0 → r_resid <0) → once geometry/AOD
    are removed the ratio is actually elevated near clouds, consistent with
    the ocean direction and confirming a real cloud-adjacency effect.
    """
    ratio_defs = [
        ('exp_o2a_intercept',  'alb_o2a',  'O\u2082A',   'C0'),
        ('exp_wco2_intercept', 'alb_wco2', 'WCO\u2082',  'C1'),
        ('exp_sco2_intercept', 'alb_sco2', 'SCO\u2082',  'C2'),
    ]
    avail = [(ei, alb, nm, c) for ei, alb, nm, c in ratio_defs
             if ei in df.columns and alb in df.columns]
    if not avail:
        logger.warning("exp_intercept or albedo columns missing — skipping ratio residual plot")
        return

    ratio_assign = {
        f'_ratio_{alb.split("_")[1]}': df[ei] / df[alb].replace(0, np.nan)
        for ei, alb, nm, c in avail
    }
    df = df.assign(**ratio_assign)

    # confounders: albedo already divided out; control geometry + aerosol
    control_cols = ['airmass', 'mu_sza', 'aod_total']
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(len(avail), 2, figsize=(13, 4.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (ei, alb, nm, col) in enumerate(avail):
        tag       = alb.split('_')[1]
        ratio_col = f'_ratio_{tag}'
        for ci, (sfc_name, sdf) in enumerate(subsets):
            ax = axes[row, ci]
            ctrl  = [c for c in control_cols if c in sdf.columns]
            req   = [ratio_col, 'cld_dist_km'] + ctrl
            m     = sdf[req].notna().all(axis=1) & np.isfinite(sdf[ratio_col])
            sdf_m = sdf[m].copy()
            if len(sdf_m) < 50:
                ax.set_visible(False)
                continue

            X = np.column_stack([np.ones(len(sdf_m))]
                                + [sdf_m[c].values for c in ctrl])
            y = sdf_m[ratio_col].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            sdf_m['_resid'] = y - X @ coef

            r_raw, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m[ratio_col])
            r_res, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m['_resid'])

            sdf_m['_bin'] = pd.cut(sdf_m['cld_dist_km'], bins=bins,
                                   labels=labels, right=False)
            means = sdf_m.groupby('_bin', observed=True)['_resid'].mean().reindex(labels)
            stds  = sdf_m.groupby('_bin', observed=True)['_resid'].std().reindex(labels)
            ns    = sdf_m.groupby('_bin', observed=True)['_resid'].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            xp = np.arange(len(labels))

            ax.fill_between(xp, (means - stds).values, (means + stds).values,
                            color=col, alpha=0.15, label='\u00b1 1 std')
            ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=col, lw=1.5,
                        label=f'mean\u00b1SEM  r={r_res:.3f}')
            ax.axhline(0, color='gray', lw=0.8, linestyle='--')
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'{nm} exp/alb residual', fontsize=9)
            ax.set_title(
                f'{nm} — {sfc_name}  |  r_raw={r_raw:.3f} \u2192 r_resid={r_res:.3f}',
                fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size:
                spread = max(finite_stds.max() * 1.5,
                             (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
                ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.suptitle(
        'exp/alb ratio residuals after removing airmass + cos(SZA) + AOD\n'
        '(albedo already divided out — residual is the pure cloud-adjacency component)',
        fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'exp_alb_ratio_residuals.png')


def plot_alb_binned_profile(df, bins, labels, outdir):
    """Mean ± SEM (bars) / ± std (shading) of albedo per band vs cloud-distance bin."""
    bands = [
        ('alb_o2a',  'O2-A',   'C0'),
        ('alb_wco2', 'WCO\u2082', 'C1'),
        ('alb_sco2', 'SCO\u2082', 'C2'),
    ]
    avail = [(col, nm, c) for col, nm, c in bands if col in df.columns]
    if not avail:
        return

    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, color) in zip(axes, avail):
        means = df.groupby(_bin, observed=True)[col].mean().reindex(labels)
        stds  = df.groupby(_bin, observed=True)[col].std().reindex(labels)
        ns    = df.groupby(_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
        sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
        ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan

        ax.fill_between(x, (means - stds).values, (means + stds).values,
                        color=color, alpha=0.15, label='\u00b1 1 std')
        ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                    capsize=4, color=color, lw=1.5, label='mean \u00b1 SEM')
        if np.isfinite(ref_val):
            ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                       label=f'mean @ {labels[-1]} km = {ref_val:.4f}')
        ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(f'{nm} albedo', fontsize=9)
        ax.set_title(f'{nm} albedo: mean \u00b1 SEM (bars) / \u00b1 std (shading)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        finite_means = means.dropna()
        finite_stds  = stds.dropna()
        if finite_means.size:
            spread = max(finite_stds.max() * 1.5, (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
            ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.tight_layout()
    _save(fig, outdir, 'alb_binned_profile.png')
