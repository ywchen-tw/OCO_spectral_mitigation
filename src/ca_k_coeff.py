"""
ca_k_coeff.py
=============
k1 / k2 / k3 cumulant-coefficient analysis functions extracted from combined_analyze.py.

Contents
--------
- plot_distributions_vs_cld_dist   Box-plots of key variables by cld_dist bin
- plot_k1_k2_vs_cld_dist           Hexbin scatter + rolling median of k1, k2
- plot_k2_over_k1_vs_cld_dist      k2/k1 ratio vs cloud distance
- plot_k1_k2_binned_profile        Binned mean ± SEM/std for k1 and k2
- plot_k1_k2_joint                 k1 vs k2 scatter colored by cld_dist
- plot_higher_order_k_profiles     k3 binned profiles for SCO2 and WCO2
- plot_k_albedo_residuals          OLS residuals of k1/k2/k3 after alb+airmass+SZA
- plot_cross_band_k_combinations   Cross-band kN ratios (binned profiles) + scatter matrix
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from ca_utils import _save, rolling_median_iqr, bin_by_cld_dist

logger = logging.getLogger(__name__)


def plot_distributions_vs_cld_dist(df, bins, labels, outdir):
    """Box-plot key variables split by cloud-distance bin."""
    vars_of_interest = {
        'o2a_k1': 'O2-A  k\u2081',
        'o2a_k2': 'O2-A  k\u2082',
        'wco2_k1': 'WCO\u2082  k\u2081',
        'wco2_k2': 'WCO\u2082  k\u2082',
        'sco2_k1': 'SCO\u2082  k\u2081',
        'sco2_k2': 'SCO\u2082  k\u2082',
        'xco2_bc_anomaly': 'XCO\u2082 BC anomaly (ppm)',
        'xco2_raw_anomaly': 'XCO\u2082 raw anomaly (ppm)',
        'aod_total': 'AOD total',
        'dp': '\u0394P (hPa)',
        'fp_area_km2': 'FP area (km\u00b2)',
    }
    existing = {k: v for k, v in vars_of_interest.items() if k in df.columns}

    _bin = bin_by_cld_dist(df, bins, labels)

    ncols = 2
    nrows = int(np.ceil(len(existing) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, existing.items()):
        groups = [df.loc[_bin == lbl, col].dropna().values for lbl in labels]
        ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='orange', lw=3))
        # reference line: median of the last (40–50 km) bin — matches the box's orange line
        ref_val = np.median(groups[-1]) if len(groups[-1]) > 0 else np.nan
        ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                   label=f'median @ {labels[-1]} km = {ref_val:.4f}')
        ax.legend(fontsize=7)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.set_title(f'{label} vs cloud distance', fontsize=10)
        # overlay sample counts
        for xi, g in enumerate(groups):
            ax.text(xi + 1, ax.get_ylim()[0], f'n={len(g):,}',
                    ha='center', va='bottom', fontsize=6.5, color='gray')

    for ax in axes[len(existing):]:
        ax.set_visible(False)

    fig.suptitle('Variable distributions vs cloud distance (box = IQR, whiskers = 1.5×IQR)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, outdir, 'dist_vs_cld_dist_boxplot.png')


def plot_xco2_anomaly_ocean_land(df, bins, labels, outdir):
    """Two-panel boxplot: xco2_bc_anomaly vs cloud-distance for ocean (left) and land (right)."""
    if 'xco2_bc_anomaly' not in df.columns or 'sfc_type' not in df.columns:
        logger.info("xco2_bc_anomaly or sfc_type missing — skipping ocean/land boxplot")
        return

    _bin = bin_by_cld_dist(df, bins, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    subsets = [('Ocean', df['sfc_type'] == 0),
               ('Land',  df['sfc_type'] == 1)]

    for ax, (title, mask) in zip(axes, subsets):
        sub  = df[mask]
        bsub = _bin[mask]
        groups = [sub.loc[bsub == lbl, 'xco2_bc_anomaly'].dropna().values
                  for lbl in labels]
        ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='orange', lw=3))
        ref_val = np.median(groups[-1]) if len(groups[-1]) > 0 else np.nan
        ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                   label=f'median @ {labels[-1]} km = {ref_val:.4f}')
        ax.axhline(0, color='gray', lw=1, linestyle=':')
        ax.legend(fontsize=7)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel('XCO\u2082 BC anomaly (ppm)', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.set_title(f'{title}: XCO\u2082 BC anomaly vs cloud distance', fontsize=10)
        for xi, g in enumerate(groups):
            ax.text(xi + 1, ax.get_ylim()[0], f'n={len(g):,}',
                    ha='center', va='bottom', fontsize=6.5, color='gray')

    fig.suptitle('XCO\u2082 BC anomaly vs cloud distance — Ocean vs Land\n'
                 '(box = IQR, whiskers = 1.5\u00d7IQR)', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, outdir, 'xco2_anomaly_ocean_land_boxplot.png')


def plot_k1_k2_vs_cld_dist(df, outdir, max_dist=50, n_roll=200):
    """Scatter (hexbin) + rolling median of k1, k2 for each spectral band."""
    bands = [
        ('o2a_k1',  'o2a_k2',  'O2-A',  'C0'),
        ('wco2_k1', 'wco2_k2', 'WCO\u2082', 'C1'),
        ('sco2_k1', 'sco2_k2', 'SCO\u2082', 'C2'),
    ]
    avail = [(k1, k2, nm, c) for k1, k2, nm, c in bands
             if k1 in df.columns and k2 in df.columns]

    sub = df[df['cld_dist_km'] <= max_dist].copy()
    x = sub['cld_dist_km'].values

    fig, axes = plt.subplots(len(avail), 2, figsize=(13, 4.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (k1c, k2c, nm, col) in enumerate(avail):
        for ci, (kcol, klabel) in enumerate([(k1c, 'k\u2081'), (k2c, 'k\u2082')]):
            ax = axes[row, ci]
            y = sub[kcol].values
            mask = np.isfinite(x) & np.isfinite(y)
            xm, ym = x[mask], y[mask]

            # hexbin density
            hb = ax.hexbin(xm, ym, gridsize=60, cmap='YlOrRd',
                           mincnt=1, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='count')

            # rolling median + IQR
            xs, med, q25, q75 = rolling_median_iqr(xm, ym, n_pts=n_roll)
            ax.plot(xs, med, color=col, lw=1.5, label='rolling median')
            ax.fill_between(xs, q25, q75, color=col, alpha=0.25, label='IQR')

            # Pearson r
            r, _ = stats.pearsonr(xm, ym)
            ax.set_title(f'{nm} {klabel} vs cld_dist   r={r:.3f}', fontsize=10)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'{nm} {klabel}', fontsize=9)
            ax.legend(fontsize=8)

    fig.suptitle('k\u2081 and k\u2082 cumulant coefficients vs cloud distance', fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'k1_k2_vs_cld_dist.png')


def plot_k2_over_k1_vs_cld_dist(df, outdir, max_dist=50, n_roll=200):
    """k2/k1 ratio per band vs cloud distance — highlights scattering asymmetry."""
    bands = [
        ('o2a_k2_over_k1',  'O2-A',   'C0'),
        ('wco2_k2_over_k1', 'WCO\u2082', 'C1'),
        ('sco2_k2_over_k1', 'SCO\u2082', 'C2'),
    ]
    avail = [(col, nm, c) for col, nm, c in bands if col in df.columns]
    if not avail:
        return

    sub = df[df['cld_dist_km'] <= max_dist].copy()
    x = sub['cld_dist_km'].values

    fig, axes = plt.subplots(1, len(avail), figsize=(6 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, c) in zip(axes, avail):
        y = sub[col].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        # clip extreme outliers for display
        lo, hi = np.percentile(ym, 1), np.percentile(ym, 99)
        vm = (ym >= lo) & (ym <= hi)
        hb = ax.hexbin(xm[vm], ym[vm], gridsize=55, cmap='plasma',
                       mincnt=1, norm=mcolors.LogNorm())
        plt.colorbar(hb, ax=ax, label='count')
        xs, med, q25, q75 = rolling_median_iqr(xm[vm], ym[vm], n_pts=n_roll)
        ax.plot(xs, med, color=c, lw=2, label='rolling median')
        ax.fill_between(xs, q25, q75, color=c, alpha=0.2)
        r, _ = stats.pearsonr(xm[vm], ym[vm])
        ax.set_title(f'{nm} k\u2082/k\u2081   r={r:.3f}', fontsize=10)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel('k\u2082 / k\u2081', fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle('k\u2082/k\u2081 ratio vs cloud distance (scattering asymmetry proxy)', fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'k2_over_k1_vs_cld_dist.png')


def plot_k1_k2_binned_profile(df, bins, labels, outdir):
    """Mean ± std of k1 and k2 as a function of cloud-distance bin (profile plot)."""
    bands = [
        ('o2a_k1',  'o2a_k2',  'O2-A'),
        ('wco2_k1', 'wco2_k2', 'WCO\u2082'),
        ('sco2_k1', 'sco2_k2', 'SCO\u2082'),
    ]
    avail = [(k1, k2, nm) for k1, k2, nm in bands
             if k1 in df.columns and k2 in df.columns]

    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (k1c, k2c, nm) in enumerate(avail):
        for ci, kcol in enumerate([k1c, k2c]):
            ax = axes[row, ci]
            means = df.groupby(_bin, observed=True)[kcol].mean().reindex(labels)
            stds  = df.groupby(_bin, observed=True)[kcol].std().reindex(labels)
            ns    = df.groupby(_bin, observed=True)[kcol].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan
            color = f'C{row}'
            # shaded ±1 std band (wide spread, light fill)
            ax.fill_between(x, (means - stds).values, (means + stds).values,
                            color=color, alpha=0.15, label='\u00b1 1 std')
            # error bars for SEM (precise uncertainty on the mean)
            ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=color, lw=1.5, label='mean \u00b1 SEM')
            if np.isfinite(ref_val):
                ax.axhline(ref_val, color='tomato', lw=2.5, linestyle='--', zorder=5,
                           label=f'mean @ {labels[-1]} km = {ref_val:.4f}')
            ax.legend(fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            klabel = 'k\u2081' if ci == 0 else 'k\u2082'
            ax.set_ylabel(f'{nm} {klabel}', fontsize=9)
            ax.set_title(f'{nm} {klabel}: mean \u00b1 SEM (bars) / \u00b1 std (shading)', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            # zoom y-axis to show the std spread
            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size:
                spread = max(finite_stds.max() * 1.5, (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
                ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.tight_layout()
    _save(fig, outdir, 'k1_k2_binned_profile.png')


def plot_k1_k2_joint(df, outdir, max_dist=50):
    """k1 vs k2 joint scatter colored by cloud distance — each band."""
    bands = [
        ('o2a_k1',  'o2a_k2',  'O2-A'),
        ('wco2_k1', 'wco2_k2', 'WCO\u2082'),
        ('sco2_k1', 'sco2_k2', 'SCO\u2082'),
    ]
    avail = [(k1, k2, nm) for k1, k2, nm in bands
             if k1 in df.columns and k2 in df.columns]
    if not avail:
        return

    sub = df[df['cld_dist_km'] <= max_dist].copy()
    fig, axes = plt.subplots(1, len(avail), figsize=(6 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]

    for ax, (k1c, k2c, nm) in zip(axes, avail):
        sc = ax.scatter(sub[k1c], sub[k2c], c=sub['cld_dist_km'],
                        cmap='viridis', s=2, alpha=0.4,
                        norm=mcolors.LogNorm(vmin=1, vmax=max_dist))
        plt.colorbar(sc, ax=ax, label='cld_dist_km')
        r, _ = stats.pearsonr(
            sub[k1c].dropna().values,
            sub[k2c].reindex(sub[k1c].dropna().index).values)
        ax.set_xlabel(f'{nm} k\u2081', fontsize=9)
        ax.set_ylabel(f'{nm} k\u2082', fontsize=9)
        ax.set_title(f'{nm}: k\u2081 vs k\u2082 (r={r:.3f})', fontsize=10)

    fig.suptitle('k\u2081 vs k\u2082 colored by cloud distance', fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'k1_vs_k2_joint_cld_dist.png')


def plot_higher_order_k_profiles(df: pd.DataFrame, bins, labels,
                                 outdir: str) -> None:
    """Binned mean \u00b1 SEM/std profile for k3 for SCO\u2082 and WCO\u2082 (meaningful variance);
    O\u2082A k3/k4/k5 are negligibly small and excluded.
    Ocean and land are shown in separate columns.
    """
    cols_info = [
        ('sco2_k3', 'SCO\u2082 k\u2083', 'C2'),
        ('wco2_k3', 'WCO\u2082 k\u2083', 'C1'),
    ]
    avail = [(col, lbl, c) for col, lbl, c in cols_info if col in df.columns]
    if not avail:
        logger.warning("sco2_k3 / wco2_k3 not found — skipping higher-order k plot")
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(len(avail), 2, figsize=(13, 4.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (col, lbl, color) in enumerate(avail):
        for ci, (sfc_name, sdf) in enumerate(subsets):
            ax = axes[row, ci]
            _sdf_bin = bin_by_cld_dist(sdf, bins, labels)
            means = sdf.groupby(_sdf_bin, observed=True)[col].mean().reindex(labels)
            stds  = sdf.groupby(_sdf_bin, observed=True)[col].std().reindex(labels)
            ns    = sdf.groupby(_sdf_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)

            r_val = np.nan
            m = sdf[col].notna() & sdf['cld_dist_km'].notna()
            if m.sum() > 10:
                r_val, _ = stats.pearsonr(sdf.loc[m, 'cld_dist_km'], sdf.loc[m, col])

            ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan
            xp = np.arange(len(labels))

            ax.fill_between(xp, (means - stds).values, (means + stds).values,
                            color=color, alpha=0.15, label='\u00b1 1 std')
            ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=color, lw=1.5,
                        label=f'mean\u00b1SEM  r={r_val:.3f}')
            if np.isfinite(ref_val):
                ax.axhline(ref_val, color='tomato', lw=2, linestyle='--',
                           label=f'ref @ {labels[-1]} km = {ref_val:.4f}')
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f'{lbl} — {sfc_name} (n={ns.sum():,})', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size:
                spread = max(finite_stds.max() * 1.5,
                             (finite_means.max() - finite_means.min()) * 1.5, 1e-9)
                ax.set_ylim(finite_means.min() - spread, finite_means.max() + spread)

    fig.suptitle(
        'Higher-order cumulant k\u2083 vs cloud distance — SCO\u2082 and WCO\u2082',
        fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'higher_order_k_profiles.png')


def plot_k_albedo_residuals(df: pd.DataFrame, bins, labels,
                            outdir: str) -> None:
    """For each of k1, k2, k3 across all three bands: OLS-remove
    alb_{band} + airmass + cos(SZA), then plot residuals vs cloud distance.

    Key finding: SCO₂ k1/k2/k3 on land retain large residual r (+0.20/+0.28/+0.23)
    — a genuine cloud-proximity signal not explained by surface or geometry.
    O₂A and WCO₂ k coefficients collapse to near zero after confounder removal.
    One output file per k term (k1, k2, k3).
    """
    k_terms = [
        ('k1', 'k\u2081'),
        ('k2', 'k\u2082'),
        ('k3', 'k\u2083'),
    ]
    band_defs = [
        ('o2a',  'O\u2082A',  'alb_o2a',  'C0'),
        ('wco2', 'WCO\u2082', 'alb_wco2', 'C1'),
        ('sco2', 'SCO\u2082', 'alb_sco2', 'C2'),
    ]
    control_cols = ['airmass', 'mu_sza']
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for kt, kt_label in k_terms:
        band_info = [(f'{bp}_{kt}', alb, nm, col)
                     for bp, nm, alb, col in band_defs
                     if f'{bp}_{kt}' in df.columns and alb in df.columns]
        if not band_info:
            continue

        fig, axes = plt.subplots(len(band_info), 2,
                                 figsize=(13, 4.5 * len(band_info)))
        if len(band_info) == 1:
            axes = axes[np.newaxis, :]

        for row, (k_col, alb_col, nm, col) in enumerate(band_info):
            for ci, (sfc_name, sdf) in enumerate(subsets):
                ax = axes[row, ci]
                ctrl  = [c for c in control_cols if c in sdf.columns]
                X_cols = [alb_col] + ctrl
                m     = sdf[[k_col, 'cld_dist_km'] + X_cols].notna().all(axis=1)
                sdf_m = sdf[m].copy()
                if len(sdf_m) < 50:
                    ax.set_visible(False)
                    continue

                X = np.column_stack([np.ones(len(sdf_m))]
                                    + [sdf_m[c].values for c in X_cols])
                y = sdf_m[k_col].values
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                sdf_m['_resid'] = y - X @ coef

                r_raw, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m[k_col])
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
                ax.set_ylabel(f'{nm} {kt_label} residual', fontsize=9)
                ax.set_title(
                    f'{nm} {kt_label} — {sfc_name}  |  r_raw={r_raw:.3f} \u2192 r_resid={r_res:.3f}',
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
            f'{kt_label} residuals after removing alb + airmass + cos(SZA)\n'
            f'SCO\u2082 {kt_label} retains genuine cloud-proximity signal on land; '
            f'O\u2082A and WCO\u2082 do not.',
            fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, f'{kt}_albedo_residuals.png')


def plot_cross_band_k_combinations(df: pd.DataFrame, bins, labels,
                                   outdir: str) -> None:
    """Cross-band cumulant-coefficient ratio profiles and scatter matrix.

    Two sub-analyses:
    (A) Binned mean ± SEM profiles of cross-band k ratios
        (e.g. o2a_k2 / wco2_k2) vs cloud-distance bins.
        Ratios that isolate scattering differences across spectral channels.
    (B) Scatter matrix of cross-band k pairs colored by cld_dist bin,
        showing how the joint distribution of kN in two different bands
        shifts with cloud proximity.

    Outputs
    -------
    cross_band_k_ratio_profiles.png   — panel of ratio binned profiles
    cross_band_k_scatter_matrix.png   — N×N scatter grid per k-order
    """
    import itertools

    BANDS = [
        ('o2a',  'O2-A',   'C0'),
        ('wco2', 'WCO\u2082', 'C1'),
        ('sco2', 'SCO\u2082', 'C2'),
    ]
    ORDERS = [
        ('k1', 'k\u2081'),
        ('k2', 'k\u2082'),
        ('k3', 'k\u2083'),
    ]

    # ── (A) Cross-band ratio binned profiles ──────────────────────────────────
    # Build list of (numerator_col, denom_col, label, color) for all
    # (band_i, band_j, kN) combos where band_i != band_j.
    ratio_specs = []
    for (b1, n1, c1), (b2, n2, c2) in itertools.combinations(
            [(b, n, c) for b, n, c in BANDS], 2):
        for kord, klbl in ORDERS:
            col1 = f'{b1}_{kord}'
            col2 = f'{b2}_{kord}'
            if col1 not in df.columns or col2 not in df.columns:
                continue
            ratio_col = f'_xb_{b1}_{b2}_{kord}'
            df[ratio_col] = df[col1] / df[col2].replace(0, np.nan)
            label = f'{n1}/{n2} {klbl}'
            ratio_specs.append((ratio_col, label, c1))

    if ratio_specs:
        _bin = bin_by_cld_dist(df, bins, labels)
        xp = np.arange(len(labels))
        ncols = 3
        nrows = int(np.ceil(len(ratio_specs) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax, (ratio_col, label, color) in zip(axes, ratio_specs):
            # clip extreme outliers to 1–99 percentile for stability
            vals = df[ratio_col].replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            col_clipped = df[ratio_col].clip(lo, hi)
            means = col_clipped.groupby(_bin).mean().reindex(labels)
            stds  = col_clipped.groupby(_bin).std().reindex(labels)
            ns    = col_clipped.groupby(_bin).count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan

            ax.fill_between(xp, (means - stds).values, (means + stds).values,
                            color=color, alpha=0.15, label='\u00b1 1 std')
            ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=color, lw=1.5, label='mean \u00b1 SEM')
            if np.isfinite(ref_val):
                ax.axhline(ref_val, color='tomato', lw=2, linestyle='--',
                           label=f'ref={ref_val:.3f}')

            # Pearson r with cld_dist
            valid = df[ratio_col].replace([np.inf, -np.inf], np.nan).notna() & \
                    df['cld_dist_km'].notna()
            r_val = np.nan
            if valid.sum() > 10:
                r_val, _ = stats.pearsonr(df.loc[valid, 'cld_dist_km'],
                                          df.loc[valid, ratio_col].clip(lo, hi))

            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=7)
            ax.set_xlabel('Cloud distance (km)', fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.set_title(f'{label}   r={r_val:.3f}', fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(axis='y', alpha=0.3)

            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size:
                spread = max(finite_stds.max() * 1.5,
                             (finite_means.max() - finite_means.min()) * 1.5,
                             1e-9)
                ax.set_ylim(finite_means.min() - spread,
                            finite_means.max() + spread)

        for ax in axes[len(ratio_specs):]:
            ax.set_visible(False)

        fig.suptitle(
            'Cross-band k-ratio profiles vs cloud distance\n'
            '(each panel = kN in band A / kN in band B)',
            fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, 'cross_band_k_ratio_profiles.png')

    # ── (B) Cross-band k scatter matrix (per k-order) ────────────────────────
    # For each k-order, draw an N×N grid where cell (i,j) is scatter of
    # band_i kN vs band_j kN, coloured by cld_dist bin.
    cmap_bins = plt.cm.get_cmap('plasma', len(labels))
    _bin_cat = bin_by_cld_dist(df, bins, labels)

    for kord, klbl in ORDERS:
        avail_bands = [(b, n, c) for b, n, c in BANDS
                       if f'{b}_{kord}' in df.columns]
        if len(avail_bands) < 2:
            continue

        nb = len(avail_bands)
        fig, axes = plt.subplots(nb, nb, figsize=(4.5 * nb, 4.5 * nb))
        if nb == 1:
            axes = np.array([[axes]])
        axes = np.array(axes)

        for ri, (b_row, n_row, _) in enumerate(avail_bands):
            for ci, (b_col, n_col, _) in enumerate(avail_bands):
                ax = axes[ri, ci]
                col_x = f'{b_col}_{kord}'
                col_y = f'{b_row}_{kord}'
                if ri == ci:
                    # diagonal: histogram per cld_dist bin
                    for bi, lbl in enumerate(labels):
                        mask = _bin_cat == lbl
                        vals = df.loc[mask, col_x].dropna().values
                        if len(vals) > 5:
                            lo_v = np.percentile(vals, 1)
                            hi_v = np.percentile(vals, 99)
                            ax.hist(vals[(vals >= lo_v) & (vals <= hi_v)],
                                    bins=40, alpha=0.5,
                                    color=cmap_bins(bi),
                                    label=lbl, density=True)
                    ax.set_title(f'{n_row} {klbl}', fontsize=9)
                    ax.legend(fontsize=5, loc='upper right')
                else:
                    for bi, lbl in enumerate(labels):
                        mask = _bin_cat == lbl
                        xs = df.loc[mask, col_x].values
                        ys = df.loc[mask, col_y].values
                        valid = np.isfinite(xs) & np.isfinite(ys)
                        if valid.sum() > 5:
                            # subsample to avoid overplotting
                            idx = np.where(valid)[0]
                            if len(idx) > 3000:
                                idx = np.random.choice(idx, 3000, replace=False)
                            ax.scatter(xs[idx], ys[idx],
                                       s=1.5, alpha=0.4,
                                       color=cmap_bins(bi),
                                       label=lbl if ri == 0 else None)
                    if ci == 0:
                        ax.set_ylabel(f'{n_row} {klbl}', fontsize=8)
                    if ri == nb - 1:
                        ax.set_xlabel(f'{n_col} {klbl}', fontsize=8)

        # shared colorbar legend for cld_dist bins
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap_bins(bi), markersize=7,
                               label=lbl)
                   for bi, lbl in enumerate(labels)]
        fig.legend(handles=handles, title='cld_dist (km)',
                   loc='lower right', fontsize=7, title_fontsize=8,
                   bbox_to_anchor=(1.0, 0.0))
        fig.suptitle(
            f'Cross-band {klbl} scatter matrix colored by cloud-distance bin',
            fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, f'cross_band_{kord}_scatter_matrix.png')

    # clean up temporary ratio columns
    for ratio_col, _, _ in ratio_specs:
        if ratio_col in df.columns:
            df.drop(columns=[ratio_col], inplace=True)


# ── R13: Footprint area vs spectral variables within cloud-distance groups ─────

def plot_fp_area_analysis(df: pd.DataFrame, bins, labels,
                           outdir: str, max_dist: float = 50) -> None:
    """Compare spectral variables vs fp_area_km2 within each cloud-distance bin.

    Skips silently if fp_area_km2 column is absent.

    Five sub-analyses:
      1. Binned profile per cld_dist bin  — mean ± SEM vs fp_area_km2 quantile bins
      2. 2-D hexbin  — fp_area_km2 vs each spectral variable, colored by cld_dist
      3. Partial-r bar chart  — Pearson r(variable, fp_area_km2) per cld_dist bin
      4. Interaction heatmap  — mean xco2_bc_anomaly on (cld_dist × fp_area_km2) grid
      5. fp_area_km2 vs mu_sza — hexbin scatter + rolling median, and binned-mean profile

    Output (under outdir/fp_area/):
      fp_area_binned_{var}.png
      fp_area_hexbin_{var}.png
      fp_area_partial_r.png
      fp_area_xco2_interaction_heatmap.png
      fp_area_vs_mu_sza_hexbin.png
      fp_area_vs_mu_sza_binned_profile.png
    """
    from scipy import stats as _stats
    import os

    if 'fp_area_km2' not in df.columns:
        logger.info("fp_area_km2 not in data — skipping R13 fp_area analysis")
        return

    fp_outdir = os.path.join(outdir, 'fp_area')
    os.makedirs(fp_outdir, exist_ok=True)

    # ── variable registry ─────────────────────────────────────────────────────
    _VARS = [
        ('o2a_k1',           'O\u2082A k\u2081'),
        ('o2a_k2',           'O\u2082A k\u2082'),
        ('wco2_k1',          'WCO\u2082 k\u2081'),
        ('wco2_k2',          'WCO\u2082 k\u2082'),
        ('sco2_k1',          'SCO\u2082 k\u2081'),
        ('sco2_k2',          'SCO\u2082 k\u2082'),
        ('exp_o2a_intercept',  'O\u2082A exp'),
        ('exp_wco2_intercept', 'WCO\u2082 exp'),
        ('exp_sco2_intercept', 'SCO\u2082 exp'),
        ('alb_o2a',          'O\u2082A alb'),
        ('alb_wco2',         'WCO\u2082 alb'),
        ('alb_sco2',         'SCO\u2082 alb'),
        ('xco2_bc_anomaly',  'XCO\u2082 BC anom.'),
    ]
    avail = [(col, lbl) for col, lbl in _VARS if col in df.columns]
    if not avail:
        return

    # Clip fp_area_km2 to 1st–99th percentile
    sub = df[df['cld_dist_km'] <= max_dist].copy()
    lo, hi = np.nanpercentile(sub['fp_area_km2'], [1, 99])
    sub = sub[(sub['fp_area_km2'] >= lo) & (sub['fp_area_km2'] <= hi)].copy()
    if len(sub) < 100:
        logger.warning("R13: too few soundings after fp_area clip — skipping")
        return

    _bin = bin_by_cld_dist(sub, bins, labels)
    FP_AREA_STEP = 0.5   # km² — change here to adjust bin width
    _fp_max = np.nanpercentile(sub['fp_area_km2'], 99)
    fp_edges = np.arange(0, _fp_max + FP_AREA_STEP, FP_AREA_STEP)
    sub['_fp_q'] = pd.cut(sub['fp_area_km2'], bins=fp_edges, right=False)
    sub = sub[sub['_fp_q'].notna()].copy()
    fp_q_labels = [str(iv) for iv in sorted(sub['_fp_q'].cat.categories)]

    # ── 1. Binned profile per fp_area group ───────────────────────────────────
    # Panels = fp_area quintiles + 1 combined; x = cld_dist bins
    sub['_cld_bin'] = _bin
    xi      = np.arange(len(labels))
    fp_cats = sorted(sub['_fp_q'].cat.categories)
    n_fp    = len(fp_cats)
    n_panels = n_fp + 1          # fp_area panels + 1 combined

    for col, lbl in avail:
        ncols = min(n_panels, 4)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4 * nrows),
                                 sharey=True)
        axes = np.array(axes).flatten()

        # ── per-fp_area panels ──────────────────────────────────────────────
        for pi, fq in enumerate(fp_cats):
            ax  = axes[pi]
            grp = sub.loc[sub['_fp_q'] == fq].groupby('_cld_bin', observed=True)[col]
            means = grp.mean().reindex(labels)
            stds  = grp.std().reindex(labels)
            cnts  = grp.count().reindex(labels)
            sems  = stds / np.sqrt(cnts)

            xv = xi
            # STD as shaded band
            ax.fill_between(xv, (means - stds).values, (means + stds).values,
                            alpha=0.18, color=f'C{pi}', label='±1 STD')
            # SEM as error bars
            ax.errorbar(xv, means.values, yerr=sems.values,
                        fmt='o-', capsize=3, color=f'C{pi}', lw=1.8,
                        label='mean ± SEM')
            ax.set_xticks(xv)
            ax.set_xticklabels([str(l) for l in labels], rotation=35, fontsize=6)
            ax.set_title(f'fp_area {fq}', fontsize=8)
            ax.set_xlabel('cld_dist bin (km)', fontsize=7)
            ax.set_ylabel(lbl, fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(fontsize=6, loc='upper left',
                      bbox_to_anchor=(1.01, 1), borderaxespad=0)

        # ── combined panel ──────────────────────────────────────────────────
        ax_all = axes[n_fp]
        for pi, fq in enumerate(fp_cats):
            grp   = sub.loc[sub['_fp_q'] == fq].groupby('_cld_bin', observed=True)[col]
            means = grp.mean().reindex(labels)
            stds  = grp.std().reindex(labels)
            cnts  = grp.count().reindex(labels)
            sems  = stds / np.sqrt(cnts)
            ax_all.errorbar(xi, means.values, yerr=sems.values,
                            fmt='o-', capsize=3, color=f'C{pi}', lw=1.5,
                            label=str(fq))
        ax_all.set_xticks(xi)
        ax_all.set_xticklabels([str(l) for l in labels], rotation=35, fontsize=6)
        ax_all.set_title('All fp_area groups', fontsize=8)
        ax_all.set_xlabel('cld_dist bin (km)', fontsize=7)
        ax_all.set_ylabel(lbl, fontsize=7)
        ax_all.grid(axis='y', alpha=0.3)
        ax_all.legend(title='fp_area bin', fontsize=6,
                      loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        fig.suptitle(f'{lbl} vs cld_dist bin per fp_area group\n'
                     'shaded = ±1 STD,  bars = ±SEM', fontsize=11)
        fig.tight_layout()
        safe = col.replace('_', '')
        _save(fig, fp_outdir, f'fp_area_binned_{safe}.png')

    # ── 2. 2-D hexbin: cld_dist_km vs each spectral variable (binned by fp_area_km2) ──
    fp_area_max = np.nanpercentile(sub['fp_area_km2'].values, 99)
    norm_fp = mcolors.Normalize(vmin=0, vmax=fp_area_max)
    cmap_fp = plt.cm.plasma_r
    xcld = sub['cld_dist_km'].values
    cv   = sub['fp_area_km2'].values

    _fp_q = sub['_fp_q']  # reuse fixed 0.5 km² bins defined above

    for col, lbl in avail:
        yv = sub[col].values.astype(float)
        m  = np.isfinite(xcld) & np.isfinite(yv) & np.isfinite(cv)
        fig, ax = plt.subplots(figsize=(7, 5))
        hb = ax.hexbin(xcld[m], yv[m], C=cv[m], gridsize=50,
                       cmap=cmap_fp, norm=norm_fp, reduce_C_function=np.mean,
                       mincnt=3)
        plt.colorbar(hb, ax=ax, label='mean fp_area (km²)')
        # rolling median per fp_area quintile
        for qi, (qcat, qdf) in enumerate(sub.groupby(_fp_q, observed=True)):
            xq = qdf['cld_dist_km'].values
            yq = qdf[col].values.astype(float)
            mq = np.isfinite(xq) & np.isfinite(yq)
            if mq.sum() < 20:
                continue
            order = np.argsort(xq[mq])
            xs, med, _, _ = rolling_median_iqr(xq[mq][order], yq[mq][order])
            ax.plot(xs, med, lw=1.5, label=str(qcat), color=f'C{qi}')
        ax.set_xlabel('cld_dist_km (km)', fontsize=9)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(f'{lbl} vs cld_dist_km\n(colored by mean fp_area; lines = fp_area quintiles)',
                     fontsize=10)
        fig.legend(title='fp_area bin', fontsize=7, title_fontsize=8,
                   loc='lower center', bbox_to_anchor=(0.5, -0.08),
                   ncol=5, borderaxespad=0)
        fig.tight_layout()
        safe = col.replace('_', '')
        _save(fig, fp_outdir, f'fp_area_hexbin_{safe}.png')

    # ── 3. Partial-r bar chart ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, dist_lbl in zip(axes, labels):
        mask = (_bin == dist_lbl)
        r_vals, bar_labels, bar_cols = [], [], []
        for idx, (col, lbl) in enumerate(avail):
            g = sub.loc[mask, ['fp_area_km2', col]].dropna()
            if len(g) < 10:
                r_vals.append(np.nan)
            else:
                r_v, _ = _stats.pearsonr(g['fp_area_km2'], g[col])
                r_vals.append(r_v)
            bar_labels.append(lbl)
            bar_cols.append(f'C{idx % 10}')
        yi = np.arange(len(r_vals))
        ax.barh(yi, r_vals, color=bar_cols, alpha=0.7, edgecolor='k', lw=0.4)
        ax.set_yticks(yi)
        ax.set_yticklabels(bar_labels, fontsize=7)
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel('Pearson r', fontsize=8)
        ax.set_title(f'{dist_lbl} km', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Pearson r(variable, fp_area_km2) per cloud-distance bin', fontsize=11)
    fig.tight_layout()
    _save(fig, fp_outdir, 'fp_area_partial_r.png')

    # ── 4. Interaction heatmap: mean xco2_bc_anomaly ──────────────────────────
    if 'xco2_bc_anomaly' not in sub.columns:
        return

    grid = sub.groupby([_bin, '_fp_q'], observed=True)['xco2_bc_anomaly'].mean().unstack()
    grid = grid.reindex(labels)

    fig, ax = plt.subplots(figsize=(max(6, len(fp_q_labels)), max(4, len(labels))))
    im = ax.imshow(grid.values.astype(float), aspect='auto', origin='upper',
                   cmap='RdBu_r',
                   vmin=-np.nanpercentile(np.abs(grid.values), 95),
                   vmax= np.nanpercentile(np.abs(grid.values), 95))
    plt.colorbar(im, ax=ax, label='mean XCO\u2082 BC anomaly (ppm)')
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels(fp_q_labels, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('fp_area_km2 (quantile bin)', fontsize=9)
    ax.set_ylabel('Cloud distance (km)', fontsize=9)
    ax.set_title('Mean XCO\u2082 BC anomaly\n(cld_dist \u00d7 fp_area_km2)', fontsize=10)
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            val = grid.values[ri, ci]
            if np.isfinite(val):
                ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.5, color='k')
    fig.tight_layout()
    _save(fig, fp_outdir, 'fp_area_xco2_interaction_heatmap.png')

    # ── 5. fp_area_km2 vs mu_sza relationship ─────────────────────────────────
    if 'mu_sza' not in sub.columns:
        logger.info("R13-5: mu_sza not in data — skipping fp_area vs mu_sza analysis")
        return

    mu    = sub['mu_sza'].values.astype(float)
    farea = sub['fp_area_km2'].values.astype(float)
    m_ok  = np.isfinite(mu) & np.isfinite(farea)
    mu_ok, fa_ok = mu[m_ok], farea[m_ok]

    if m_ok.sum() < 50:
        logger.warning("R13-5: too few valid mu_sza/fp_area pairs — skipping")
        return

    r_val, p_val = _stats.pearsonr(mu_ok, fa_ok)

    # ── 5a. Hexbin scatter ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(mu_ok, fa_ok, gridsize=60, cmap='viridis', mincnt=3,
                   norm=mcolors.LogNorm())
    plt.colorbar(hb, ax=ax, label='count')

    # rolling median overlay
    order = np.argsort(mu_ok)
    xs, med, q25, q75 = rolling_median_iqr(mu_ok[order], fa_ok[order])
    ax.plot(xs, med, 'r-', lw=2, label='rolling median')
    ax.fill_between(xs, q25, q75, color='red', alpha=0.2, label='IQR')

    ax.set_xlabel('mu_sza  [cos(SZA)]', fontsize=10)
    ax.set_ylabel('fp_area_km2  (km²)', fontsize=10)
    ax.set_title(f'Footprint area vs cos(SZA)\nPearson r = {r_val:.3f}  (p = {p_val:.2e})',
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fp_outdir, 'fp_area_vs_mu_sza_hexbin.png')

    # ── 5b. Binned profile: mean fp_area_km2 per mu_sza bin ────────────────
    n_bins  = 20
    mu_edges = np.linspace(np.nanpercentile(mu_ok, 1),
                           np.nanpercentile(mu_ok, 99), n_bins + 1)
    mu_mids  = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mu_bin   = pd.cut(pd.Series(mu_ok), bins=mu_edges)
    fa_ser   = pd.Series(fa_ok)

    grp   = fa_ser.groupby(mu_bin, observed=True)
    means = grp.mean().values
    stds  = grp.std().values
    cnts  = grp.count().values
    sems  = np.where(cnts > 0, stds / np.sqrt(cnts), np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(mu_mids, means - stds, means + stds,
                    alpha=0.18, color='steelblue', label='±1 STD')
    ax.errorbar(mu_mids, means, yerr=sems,
                fmt='o-', capsize=4, color='steelblue', lw=2, label='mean ± SEM')
    ax.set_xlabel('mu_sza  [cos(SZA)]', fontsize=10)
    ax.set_ylabel('fp_area_km2  (km²)', fontsize=10)
    ax.set_title(f'Mean footprint area vs cos(SZA)  (binned)\nPearson r = {r_val:.3f}',
                 fontsize=11)
    # annotate count per bin
    for mx, mn, cnt in zip(mu_mids, means, cnts):
        if np.isfinite(mn) and cnt > 0:
            ax.text(mx, mn, f'n={cnt}', ha='center', va='bottom',
                    fontsize=5.5, color='gray')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, fp_outdir, 'fp_area_vs_mu_sza_binned_profile.png')

    logger.info(f"R13-5: fp_area vs mu_sza  r={r_val:.3f}  p={p_val:.2e}  "
                f"n={m_ok.sum():,}")
