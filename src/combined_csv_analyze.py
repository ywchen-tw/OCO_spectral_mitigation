"""
combined_csv_analyze.py
=======================
Analyze parquet output files from fitting_data_correction.py.

Sections
--------
1. Load & filter data
2. Variable distributions vs cloud distance (cld_dist_km)
3. k1 / k2 cumulant coefficients vs cloud distance — all three bands
4. xco2_anomaly relationships with other variables (scatter + correlation)
5. Save all figures to results/figures/
"""

import os
import sys
import glob
import platform
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy import stats

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_storage_dir() -> Path:
    if platform.system() == "Darwin":
        return Path(Config.get_data_path('local'))
    elif platform.system() == "Linux":
        return Path(Config.get_data_path('curc'))
    return Path(Config.get_data_path('default'))


def load_data(csv_dir: Path, parquet_fname: str | None = None) -> pd.DataFrame:
    """Load combined parquet.  Falls back to all per-date parquets."""
    if parquet_fname:
        path = csv_dir / parquet_fname
        if path.exists():
            logger.info(f"Loading {path}")
            return pd.read_parquet(path)

    files = sorted(glob.glob(str(csv_dir / 'combined_*_all_orbits.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {csv_dir}")
    logger.info(f"Loading {len(files)} per-date files …")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep good-quality soundings: valid xco2_bc, qf==0, no snow."""
    mask = df['xco2_bc'] > 0
    if 'xco2_qf' in df.columns:
        mask &= df['xco2_qf'] == 0
    if 'snow_flag' in df.columns:
        # snow_flag is stored as uint8/int; treat any non-zero value as snowy
        mask &= df['snow_flag'] == 0
    df = df[mask].copy()
    logger.info(f"After QF+snow filter: {len(df):,} soundings")
    return df


def split_by_surface(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'ocean': ..., 'land': ...} subsets based on sfc_type."""
    subsets = {}
    if 'sfc_type' not in df.columns:
        logger.warning("sfc_type column missing — treating all as 'all'")
        subsets['all'] = df
        return subsets
    ocean = df[df['sfc_type'] == 0].copy()
    land  = df[df['sfc_type'] == 1].copy()
    logger.info(f"Ocean soundings: {len(ocean):,}  |  Land soundings: {len(land):,}")
    subsets['ocean'] = ocean
    subsets['land']  = land
    return subsets


def cld_dist_bins(edges=(0, 5, 10, 20, 30, 40, 50)):
    """Return (edges, labels) for cloud-distance bin edges in km."""
    labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges) - 1)]
    return list(edges), labels


def bin_by_cld_dist(df: pd.DataFrame, edges, labels) -> pd.Series:
    return pd.cut(df['cld_dist_km'], bins=edges, labels=labels, right=False)


# ── plotting utilities ────────────────────────────────────────────────────────

def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  saved → {p}")


def rolling_median_iqr(x, y, n_pts=80):
    """Return (bin_centers, median, q25, q75) using fixed-width x bins.

    Replaces the O(n²) sliding-window with an O(n) binned approach:
    divide x into n_pts equal bins, compute percentiles per bin.
    Empty bins are dropped so the curve stays clean.
    """
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    edges = np.linspace(x_min, x_max, n_pts + 1)
    centers, meds, q25s, q75s = [], [], [], []
    bin_idx = np.digitize(x, edges) - 1          # 0-based bin index
    bin_idx = np.clip(bin_idx, 0, n_pts - 1)
    for b in range(n_pts):
        w = y[bin_idx == b]
        if len(w) < 5:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        meds.append(np.median(w))
        q25s.append(np.percentile(w, 25))
        q75s.append(np.percentile(w, 75))
    return (np.array(centers), np.array(meds),
            np.array(q25s), np.array(q75s))


# ── 1. Distribution histograms vs cld_dist bins ───────────────────────────────

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
    }
    existing = {k: v for k, v in vars_of_interest.items() if k in df.columns}

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)

    ncols = 2
    nrows = int(np.ceil(len(existing) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, existing.items()):
        groups = [df.loc[df['_bin'] == lbl, col].dropna().values for lbl in labels]
        ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True,
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


# ── 2. k1 / k2 scatter + rolling stats vs cld_dist ───────────────────────────

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

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (k1c, k2c, nm) in enumerate(avail):
        for ci, kcol in enumerate([k1c, k2c]):
            ax = axes[row, ci]
            means = df.groupby('_bin', observed=True)[kcol].mean().reindex(labels)
            stds  = df.groupby('_bin', observed=True)[kcol].std().reindex(labels)
            ns    = df.groupby('_bin', observed=True)[kcol].count().reindex(labels).fillna(0).astype(int)
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

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, color) in zip(axes, avail):
        means = df.groupby('_bin', observed=True)[col].mean().reindex(labels)
        stds  = df.groupby('_bin', observed=True)[col].std().reindex(labels)
        ns    = df.groupby('_bin', observed=True)[col].count().reindex(labels).fillna(0).astype(int)
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


# ── 3. xco2_anomaly relationships ────────────────────────────────────────────

def plot_xco2_anomaly_correlations(df, outdir):
    """Correlation matrix heat-map: xco2 anomaly against all key predictors."""
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


def plot_xco2_anomaly_vs_key_vars(df, outdir, max_dist=50, n_roll=200):
    """Scatter panels: xco2_bc_anomaly vs top predictors."""
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

    sub = df.copy()
    if 'cld_dist_km' in df.columns:
        sub = sub[sub['cld_dist_km'] <= max_dist]

    target = 'xco2_bc_anomaly'
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
        ax.set_ylabel('XCO\u2082 BC anomaly (ppm)', fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[len(avail):]:
        ax.set_visible(False)

    fig.suptitle('XCO\u2082 BC anomaly vs key predictors', fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, outdir, 'xco2_bc_anomaly_vs_predictors.png')


def plot_xco2_anomaly_vs_cld_dist_binned(df, bins, labels, outdir):
    """Mean XCO2 anomaly ± SEM as a function of cloud-distance bin."""
    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    targets = [
        ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'C0'),
        ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'C1'),
    ]
    avail = [(t, lbl, c) for t, lbl, c in targets if t in df.columns]

    for col, lbl, c in avail:
        fig, ax = plt.subplots(figsize=(7, 5))
        means = df.groupby('_bin', observed=True)[col].mean().reindex(labels)
        stds  = df.groupby('_bin', observed=True)[col].std().reindex(labels)
        ns    = df.groupby('_bin', observed=True)[col].count().reindex(labels).fillna(0).astype(int)
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
        fname = 'xco2_bc_anomaly_vs_cld_dist_binned.png' if 'bc' in col else 'xco2_raw_anomaly_vs_cld_dist_binned.png'
        _save(fig, outdir, fname)


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

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, color) in zip(axes, avail):
        means = df.groupby('_bin', observed=True)[col].mean().reindex(labels)
        stds  = df.groupby('_bin', observed=True)[col].std().reindex(labels)
        ns    = df.groupby('_bin', observed=True)[col].count().reindex(labels).fillna(0).astype(int)
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


def print_summary_stats(df, bins, labels):
    """Print key statistics to stdout."""
    print("=" * 65)
    print("SUMMARY STATISTICS")
    print("=" * 65)
    print(f"Total soundings: {len(df):,}")
    print()

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, bins, labels)

    k_cols = ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2']
    anom_cols = ['xco2_bc_anomaly', 'xco2_raw_anomaly']
    avail_k = [c for c in k_cols if c in df.columns]
    avail_a = [c for c in anom_cols if c in df.columns]

    print("--- k1 / k2 global stats ---")
    print(df[avail_k].describe().round(4).to_string())
    print()
    print("--- XCO2 anomaly global stats ---")
    print(df[avail_a].describe().round(4).to_string())
    print()

    print("--- k1 mean by cloud-distance bin ---")
    for col in avail_k:
        grp = df.groupby('_bin', observed=True)[col].mean().round(4)
        print(f"  {col}: {dict(grp)}")
    print()

    print("--- XCO2 anomaly mean by cloud-distance bin ---")
    for col in avail_a:
        grp = df.groupby('_bin', observed=True)[col].mean().round(4)
        print(f"  {col}: {dict(grp)}")
    print()

    print("--- Pearson r: k1/k2 vs xco2_bc_anomaly ---")
    if 'xco2_bc_anomaly' in df.columns:
        for col in avail_k:
            mask = df[col].notna() & df['xco2_bc_anomaly'].notna()
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, col], df.loc[mask, 'xco2_bc_anomaly'])
                print(f"  {col}: r={r:.4f}  p={p:.3e}")
    print()

    print("--- Pearson r: k1/k2 vs cld_dist_km ---")
    if 'cld_dist_km' in df.columns:
        for col in avail_k:
            mask = df[col].notna() & df['cld_dist_km'].notna()
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, 'cld_dist_km'], df.loc[mask, col])
                print(f"  {col}: r={r:.4f}  p={p:.3e}")
    print("=" * 65)


# ── 4. Stratified analysis ────────────────────────────────────────────────────
#
# For each conditioning variable, the data are split into fixed physical strata
# and the four core plot functions are re-run on each subset.
#
# Edit STRAT_CONFIG to adjust edges or add new variables.
# Keys must match column names in the parquet file.
#
STRAT_CONFIG: dict[str, tuple[list, str]] = {
    # column        : (bin_edges,                   unit_label_for_dir)
    # mu_sza = cos(SZA); higher value = lower SZA (more overhead)
    'mu_sza':      ([0.25, 0.50, 0.65, 0.80, 1.01], 'cosSZA'),
    # Raw SZA in degrees — used if the column exists instead of mu_sza
    'sza':         ([0, 20, 40, 55, 70, 90],         'deg'),
    'alb_o2a':     ([0.00, 0.05, 0.10, 0.25, 0.50, 1.00],  ''),
    'glint_angle': ([0.0,  5.0, 10.0, 20.0, 45.0],   'deg'),
    'aod_total':   ([0.00, 0.025, 0.05, 0.10, 0.25, 1.00],  ''),
    # dp = retrieved – prior surface pressure (hPa); range is roughly -8 to +10
    'dp':          ([-10, -5, -2, 0, 2, 5, 10],             'hPa'),
}


def _safe_label(s: str) -> str:
    """Convert a bin label to a filesystem-safe string."""
    return s.replace('/', '_').replace(' ', '').replace('\u2013', '_')


def _build_strata(df: pd.DataFrame, col: str, edges: list, unit: str
                  ) -> tuple[pd.DataFrame, list, list] | tuple[None, None, None]:
    """Clip *edges* to data range, assign '_strat' column, return (df, clipped, labels).

    Returns (None, None, None) if the column is missing or has too narrow a range.
    """
    if col not in df.columns:
        logger.warning(f"Stratification variable '{col}' not found — skipping")
        return None, None, None

    ser = df[col].dropna()
    lo, hi = ser.min(), ser.max()
    clipped = [e for e in edges if lo <= e <= hi]
    if not clipped or clipped[0] > lo:
        clipped = [lo] + clipped
    if clipped[-1] < hi:
        clipped = clipped + [hi]
    clipped = sorted(set(clipped))
    if len(clipped) < 3:
        logger.warning(f"'{col}' has too narrow a range for stratification — skipping")
        return None, None, None

    n = len(clipped) - 1
    bin_labels = [
        f"{clipped[i]:.3g}{unit}\u2013{clipped[i+1]:.3g}{unit}"
        for i in range(n)
    ]
    df = df.copy()
    df['_strat'] = pd.cut(df[col], bins=clipped, labels=bin_labels, include_lowest=True)
    return df, clipped, bin_labels


def plot_k1_k2_overlay(df: pd.DataFrame, cld_bins, cld_labels,
                       outdir: str, strat_var: str,
                       strat_labels: list) -> None:
    """Mean ± SEM k1/k2 profiles for all strata overlaid on one figure per band."""
    bands = [
        ('o2a_k1',  'o2a_k2',  'O2-A'),
        ('wco2_k1', 'wco2_k2', 'WCO\u2082'),
        ('sco2_k1', 'sco2_k2', 'SCO\u2082'),
    ]
    avail = [(k1, k2, nm) for k1, k2, nm in bands
             if k1 in df.columns and k2 in df.columns]
    if not avail:
        return

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, cld_bins, cld_labels)
    x = np.arange(len(cld_labels))
    colors = plt.colormaps['viridis'](np.linspace(0.1, 0.9, len(strat_labels)))

    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (k1c, k2c, nm) in enumerate(avail):
        for ci, kcol in enumerate([k1c, k2c]):
            ax = axes[row, ci]
            all_means = []
            for si, slabel in enumerate(strat_labels):
                sdf = df[df['_strat'] == slabel]
                if len(sdf) < 100:
                    continue
                means = sdf.groupby('_bin', observed=True)[kcol].mean().reindex(cld_labels)
                stds  = sdf.groupby('_bin', observed=True)[kcol].std().reindex(cld_labels)
                ns    = sdf.groupby('_bin', observed=True)[kcol].count().reindex(cld_labels)
                sems  = stds / np.sqrt(ns)
                ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                            capsize=3, color=colors[si], lw=1.5, label=slabel)
                ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan
                if np.isfinite(ref_val):
                    ax.axhline(ref_val, color=colors[si], lw=1.2,
                               linestyle='--', alpha=0.7, zorder=3)
                all_means.append(means.values[~np.isnan(means.values)])

            ax.set_xticks(x)
            ax.set_xticklabels(cld_labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            klabel = 'k\u2081' if ci == 0 else 'k\u2082'
            ax.set_ylabel(f'{nm} {klabel}', fontsize=9)
            ax.set_title(f'{nm} {klabel} — stratified by {strat_var}', fontsize=10)
            ax.legend(fontsize=7, title=strat_var, title_fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            if all_means:
                all_vals = np.concatenate(all_means)
                if len(all_vals):
                    vmin, vmax = all_vals.min(), all_vals.max()
                    spread = max((vmax - vmin) * 1.5, 1e-9)
                    ax.set_ylim(vmin - spread, vmax + spread)

    fig.suptitle(f'k\u2081 / k\u2082 profiles by {strat_var} stratum', fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'k1_k2_binned_profile_overlay.png')


def plot_intercept_overlay(df: pd.DataFrame, cld_bins, cld_labels,
                           outdir: str, strat_var: str,
                           strat_labels: list) -> None:
    """exp_intercept profiles for all strata overlaid on one figure."""
    bands = [
        ('exp_o2a_intercept',  'O2-A',   'C0'),
        ('exp_wco2_intercept', 'WCO\u2082', 'C1'),
        ('exp_sco2_intercept', 'SCO\u2082', 'C2'),
    ]
    avail = [(col, nm, c) for col, nm, c in bands if col in df.columns]
    if not avail:
        return

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, cld_bins, cld_labels)
    x = np.arange(len(cld_labels))
    colors = plt.colormaps['viridis'](np.linspace(0.1, 0.9, len(strat_labels)))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, _) in zip(axes, avail):
        all_means = []
        for si, slabel in enumerate(strat_labels):
            sdf = df[df['_strat'] == slabel]
            if len(sdf) < 100:
                continue
            means = sdf.groupby('_bin', observed=True)[col].mean().reindex(cld_labels)
            stds  = sdf.groupby('_bin', observed=True)[col].std().reindex(cld_labels)
            ns    = sdf.groupby('_bin', observed=True)[col].count().reindex(cld_labels)
            sems  = stds / np.sqrt(ns)
            ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                        capsize=3, color=colors[si], lw=1.5, label=slabel)
            ref_val = means.dropna().iloc[-1] if means.dropna().size else np.nan
            if np.isfinite(ref_val):
                ax.axhline(ref_val, color=colors[si], lw=1.2,
                           linestyle='--', alpha=0.7, zorder=3)
            all_means.append(means.values[~np.isnan(means.values)])

        ax.set_xticks(x)
        ax.set_xticklabels(cld_labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(f'{nm} exp_intercept', fontsize=9)
        ax.set_title(f'{nm} exp_intercept — stratified by {strat_var}', fontsize=10)
        ax.legend(fontsize=7, title=strat_var, title_fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        if all_means:
            all_vals = np.concatenate(all_means)
            if len(all_vals):
                vmin, vmax = all_vals.min(), all_vals.max()
                spread = max((vmax - vmin) * 1.5, 1e-9)
                ax.set_ylim(vmin - spread, vmax + spread)

    fig.suptitle(f'Spectral exp_intercepts by {strat_var} stratum', fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'exp_intercept_binned_profile_overlay.png')


def plot_xco2_anomaly_binned_overlay(df: pd.DataFrame, cld_bins, cld_labels,
                                     outdir: str, strat_var: str,
                                     strat_labels: list) -> None:
    """Mean XCO2 anomaly profiles for all strata overlaid on one figure."""
    targets = [
        ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'C0'),
        ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'C1'),
    ]
    avail = [(t, lbl, c) for t, lbl, c in targets if t in df.columns]
    if not avail:
        return

    df = df.copy()
    df['_bin'] = bin_by_cld_dist(df, cld_bins, cld_labels)
    x = np.arange(len(cld_labels))
    colors = plt.colormaps['plasma'](np.linspace(0.1, 0.85, len(strat_labels)))

    for col, lbl, _ in avail:
        fig, ax = plt.subplots(figsize=(7, 5))
        all_means = []
        for si, slabel in enumerate(strat_labels):
            sdf = df[df['_strat'] == slabel]
            if len(sdf) < 100:
                continue
            means = sdf.groupby('_bin', observed=True)[col].mean().reindex(cld_labels)
            stds  = sdf.groupby('_bin', observed=True)[col].std().reindex(cld_labels)
            ns    = sdf.groupby('_bin', observed=True)[col].count().reindex(cld_labels)
            sems  = stds / np.sqrt(ns)
            ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                        capsize=3, color=colors[si], lw=1.5, label=slabel)
            all_means.append(means.values[~np.isnan(means.values)])

        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(cld_labels, rotation=30, fontsize=9)
        ax.set_xlabel('Cloud distance (km)', fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(f'{lbl} — stratified by {strat_var}', fontsize=10)
        ax.legend(fontsize=7, title=strat_var, title_fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        if all_means:
            all_vals = np.concatenate(all_means)
            if len(all_vals):
                vmin, vmax = all_vals.min(), all_vals.max()
                spread = max((vmax - vmin) * 1.5, 1e-9)
                ax.set_ylim(vmin - spread, vmax + spread)
        fig.tight_layout()
        fname = ('xco2_bc_anomaly_binned_overlay.png' if 'bc' in col
                 else 'xco2_raw_anomaly_binned_overlay.png')
        _save(fig, outdir, fname)


def run_stratified_analysis(df: pd.DataFrame,
                            cld_bins, cld_labels,
                            base_outdir: str,
                            strat_var: str,
                            edges: list,
                            unit: str = '') -> None:
    """Split df into fixed-edge strata of *strat_var* and run core plots on each.

    Per-stratum plots: {base_outdir}/stratified/by_{strat_var}/{bin_label}/
    Overlay plots:     {base_outdir}/stratified/by_{strat_var}/
    """
    strat_df, _, bin_labels = _build_strata(df, strat_var, edges, unit)
    if strat_df is None:
        return
    assert bin_labels is not None
    df = strat_df

    overlay_dir = os.path.join(base_outdir, 'stratified', f'by_{strat_var}')
    logger.info(f"  Stratifying by '{strat_var}' into {len(bin_labels)} bins")

    for slabel in bin_labels:
        sdf = df[df['_strat'] == slabel].copy()
        if len(sdf) < 100:
            logger.info(f"    {strat_var}={slabel}: {len(sdf)} soundings — skipping (< 100)")
            continue
        logger.info(f"    {strat_var}={slabel}: {len(sdf):,} soundings")
        sdir = os.path.join(overlay_dir, _safe_label(slabel))

        plot_distributions_vs_cld_dist(sdf, cld_bins, cld_labels, sdir)
        plot_xco2_anomaly_vs_cld_dist_binned(sdf, cld_bins, cld_labels, sdir)
        plot_xco2_anomaly_vs_key_vars(sdf, sdir)
        plot_k1_k2_binned_profile(sdf, cld_bins, cld_labels, sdir)
        plot_intercept_binned_profile(sdf, cld_bins, cld_labels, sdir)

    # Overlay comparison figures — all strata on one plot
    logger.info(f"    Generating overlay figures for '{strat_var}' …")
    plot_k1_k2_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
    plot_intercept_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
    plot_xco2_anomaly_binned_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)


# ── NEW Section 2e: Albedo vs exp_intercept divergence with cloud distance ────

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
            sdf = sdf.copy()
            sdf['_bin'] = pd.cut(sdf['cld_dist_km'], bins=bins,
                                 labels=labels, right=False)

            def _binned(c):
                m = sdf.groupby('_bin', observed=True)[c].mean().reindex(labels)
                s = sdf.groupby('_bin', observed=True)[c].std().reindex(labels)
                n = sdf.groupby('_bin', observed=True)[c].count().reindex(labels).fillna(0).astype(int)
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


# ── NEW Section 1: Signal hierarchy ──────────────────────────────────────────

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
    df = df.copy()
    for exp_col, alb_col, ratio_col in ratio_pairs:
        if exp_col in df.columns and alb_col in df.columns:
            df[ratio_col] = df[exp_col] / df[alb_col].replace(0, np.nan)

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


# ── NEW Section 2b: exp_intercept albedo residuals ────────────────────────────

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


# ── NEW Section 2d: exp_intercept inter-band coherence ───────────────────────

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


# ── NEW Section 3c: Higher-order k profiles (k3 for SCO\u2082 and WCO\u2082) ────────────

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
            sdf = sdf.copy()
            sdf['_bin'] = bin_by_cld_dist(sdf, bins, labels)
            means = sdf.groupby('_bin', observed=True)[col].mean().reindex(labels)
            stds  = sdf.groupby('_bin', observed=True)[col].std().reindex(labels)
            ns    = sdf.groupby('_bin', observed=True)[col].count().reindex(labels).fillna(0).astype(int)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    storage_dir = get_storage_dir()
    result_dir  = storage_dir / 'results'
    csv_dir     = result_dir / 'csv_collection'
    # ── load ──────────────────────────────────────────────────────────────────
    df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')

    # ── quality filter (snow excluded, surface split done below) ──────────────
    df = apply_quality_filter(df)

    # ── scale exp_intercept by π ──────────────────────────────────────────────
    # TODO: remove this once the π factor is absorbed into oco_fp_spec_anal.py
    _exp_int_cols = ['exp_o2a_intercept', 'exp_wco2_intercept', 'exp_sco2_intercept']
    for _col in _exp_int_cols:
        if _col in df.columns:
            df[_col] = df[_col] * np.pi

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    # ── Section 1: signal hierarchy (full df, internal ocean/land split) ─────
    overall_outdir = str(result_dir / 'figures' / 'cld_dist_analysis')
    logger.info("Section 1: Plotting signal hierarchy (r vs cld_dist) …")
    plot_signal_hierarchy(df, overall_outdir)

    # ── Section 2b: exp_intercept albedo residuals ────────────────────────────
    logger.info("Section 2b: Plotting exp_intercept albedo residuals …")
    plot_exp_intercept_albedo_residuals(df, bins, labels, overall_outdir)

    # ── Section 2e: albedo vs exp_intercept divergence ───────────────────────
    logger.info("Section 2e: Plotting alb vs exp_intercept divergence …")
    plot_alb_exp_divergence(df, bins, labels, overall_outdir)

    # ── Section 2d: exp_intercept inter-band coherence ────────────────────────
    logger.info("Section 2d: Plotting exp_intercept inter-band coherence …")
    plot_exp_intercept_interband_coherence(df, overall_outdir)

    # ── Section 3c: higher-order k profiles (k3 for SCO₂ and WCO₂) ──────────
    logger.info("Section 3c: Plotting higher-order k3 profiles …")
    plot_higher_order_k_profiles(df, bins, labels, overall_outdir)

    # ── split by surface type ─────────────────────────────────────────────────
    subsets = split_by_surface(df)

    for sfc_name, sdf in subsets.items():
        logger.info(f"\n{'='*55}\nRunning analysis for surface type: {sfc_name.upper()}\n{'='*55}")
        sfc_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / sfc_name)

        print_summary_stats(sdf, bins, labels)

        # ── Section 1 (per surface): distributions ────────────────────────────
        logger.info("Plotting distributions vs cloud distance …")
        plot_distributions_vs_cld_dist(sdf, bins, labels, sfc_outdir)

        # ── Section 2c: exp_intercept binned profiles ─────────────────────────
        logger.info("Plotting intercept binned profiles …")
        plot_intercept_binned_profile(sdf, bins, labels, sfc_outdir)

        # ── Section 2a: albedo vs exp_intercept ──────────────────────────────
        logger.info("Plotting albedo vs exp_intercept …")
        plot_alb_vs_exp_intercept(sdf, sfc_outdir)

        logger.info("Plotting albedo vs exp_intercept cross-band …")
        plot_alb_vs_exp_intercept_cross(sdf, sfc_outdir)

        # ── Section 3a: k1/k2 profiles and scatter ────────────────────────────
        logger.info("Plotting k1/k2 binned profiles …")
        plot_k1_k2_binned_profile(sdf, bins, labels, sfc_outdir)

        logger.info("Plotting k1/k2 scatter vs cloud distance …")
        plot_k1_k2_vs_cld_dist(sdf, sfc_outdir)

        # ── Section 3b: k2/k1 ratio ───────────────────────────────────────────
        logger.info("Plotting k2/k1 ratio vs cloud distance …")
        plot_k2_over_k1_vs_cld_dist(sdf, sfc_outdir)

        # ── Section 3d: k1 vs k2 joint scatter ───────────────────────────────
        logger.info("Plotting k1 vs k2 joint colored by cld_dist …")
        plot_k1_k2_joint(sdf, sfc_outdir)

        # ── Supplementary ─────────────────────────────────────────────────────
        logger.info("Plotting albedo binned profiles …")
        plot_alb_binned_profile(sdf, bins, labels, sfc_outdir)

        # ── Section 5: XCO2 anomaly ───────────────────────────────────────────
        logger.info("Plotting XCO2 anomaly vs predictors …")
        plot_xco2_anomaly_vs_key_vars(sdf, sfc_outdir)

        logger.info("Plotting XCO2 anomaly vs cld_dist binned …")
        plot_xco2_anomaly_vs_cld_dist_binned(sdf, bins, labels, sfc_outdir)

        logger.info("Plotting correlation heat-map …")
        plot_xco2_anomaly_correlations(sdf, sfc_outdir)

        logger.info(f"All figures for {sfc_name} written to {sfc_outdir}")

        # ── Section 4: stratified analyses ────────────────────────────────────
        logger.info(f"Running stratified analyses for {sfc_name.upper()} …")
        for strat_var, (strat_edges, strat_unit) in STRAT_CONFIG.items():
            # Skip mu_sza when sza exists, and vice-versa
            if strat_var == 'mu_sza' and 'sza' in sdf.columns:
                continue
            if strat_var == 'sza' and 'sza' not in sdf.columns:
                continue
            run_stratified_analysis(sdf, bins, labels, sfc_outdir,
                                    strat_var, strat_edges, strat_unit)


if __name__ == '__main__':
    main()
