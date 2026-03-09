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

import gc
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
    # downcast float64 → float32 to halve memory for all numeric columns
    float_cols = df.select_dtypes('float64').columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype('float32')
        logger.info(f"Downcast {len(float_cols)} float64 columns to float32")
    logger.info(f"After QF+snow filter: {len(df):,} soundings")
    return df


def split_by_surface(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'ocean': ..., 'land': ...} subsets based on sfc_type."""
    subsets = {}
    if 'sfc_type' not in df.columns:
        logger.warning("sfc_type column missing — treating all as 'all'")
        subsets['all'] = df
        return subsets
    ocean = df[df['sfc_type'] == 0]
    land  = df[df['sfc_type'] == 1]
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

    _bin = bin_by_cld_dist(df, bins, labels)

    ncols = 2
    nrows = int(np.ceil(len(existing) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, existing.items()):
        groups = [df.loc[_bin == lbl, col].dropna().values for lbl in labels]
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

    sub = df[df['cld_dist_km'] <= max_dist] if 'cld_dist_km' in df.columns else df

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
    _bin = bin_by_cld_dist(df, bins, labels)
    x = np.arange(len(labels))

    targets = [
        ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'C0'),
        ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'C1'),
    ]
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

    _bin = bin_by_cld_dist(df, bins, labels)

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
        grp = df.groupby(_bin, observed=True)[col].mean().round(4)
        print(f"  {col}: {dict(grp)}")
    print()

    print("--- XCO2 anomaly mean by cloud-distance bin ---")
    for col in avail_a:
        grp = df.groupby(_bin, observed=True)[col].mean().round(4)
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

    _bin = bin_by_cld_dist(df, cld_bins, cld_labels)
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
                smask = df['_strat'] == slabel
                sdf = df[smask]
                if len(sdf) < 100:
                    continue
                sdf_bin = _bin[smask]
                means = sdf.groupby(sdf_bin, observed=True)[kcol].mean().reindex(cld_labels)
                stds  = sdf.groupby(sdf_bin, observed=True)[kcol].std().reindex(cld_labels)
                ns    = sdf.groupby(sdf_bin, observed=True)[kcol].count().reindex(cld_labels)
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

    _bin = bin_by_cld_dist(df, cld_bins, cld_labels)
    x = np.arange(len(cld_labels))
    colors = plt.colormaps['viridis'](np.linspace(0.1, 0.9, len(strat_labels)))

    fig, axes = plt.subplots(len(avail), 1, figsize=(7, 4 * len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (col, nm, _) in zip(axes, avail):
        all_means = []
        for si, slabel in enumerate(strat_labels):
            smask = df['_strat'] == slabel
            sdf = df[smask]
            if len(sdf) < 100:
                continue
            sdf_bin = _bin[smask]
            means = sdf.groupby(sdf_bin, observed=True)[col].mean().reindex(cld_labels)
            stds  = sdf.groupby(sdf_bin, observed=True)[col].std().reindex(cld_labels)
            ns    = sdf.groupby(sdf_bin, observed=True)[col].count().reindex(cld_labels)
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

    _bin = bin_by_cld_dist(df, cld_bins, cld_labels)
    x = np.arange(len(cld_labels))
    colors = plt.colormaps['plasma'](np.linspace(0.1, 0.85, len(strat_labels)))

    for col, lbl, _ in avail:
        fig, ax = plt.subplots(figsize=(7, 5))
        all_means = []
        for si, slabel in enumerate(strat_labels):
            smask = df['_strat'] == slabel
            sdf = df[smask]
            if len(sdf) < 100:
                continue
            sdf_bin = _bin[smask]
            means = sdf.groupby(sdf_bin, observed=True)[col].mean().reindex(cld_labels)
            stds  = sdf.groupby(sdf_bin, observed=True)[col].std().reindex(cld_labels)
            ns    = sdf.groupby(sdf_bin, observed=True)[col].count().reindex(cld_labels)
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


# ── NEW Section 1b: Residual signal hierarchy ─────────────────────────────────

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


# ── NEW Section 2f: exp/alb ratio residuals ──────────────────────────────────

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


# ── NEW Section 3e: k1, k2, k3 albedo residuals ───────────────────────────────

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


# ── NEW Section 5: XCO₂ anomaly partial correlation ──────────────────────────

def plot_xco2_anomaly_partial(df: pd.DataFrame, bins, labels,
                              outdir: str) -> None:
    """Partial correlation of xco2_bc_anomaly with cloud distance after
    OLS-removing the main confounders: albedo (all bands), airmass, cos(SZA),
    AOD, ΔP, CO₂ gradient, H₂O scaling, dp fraction.

    Confirms the null result: no meaningful cloud-proximity signal remains in
    XCO₂ once retrieval state variables are controlled for (r_resid < 0.10).
    Also shows the residual vs cloud-distance profile so the direction and
    shape of any residual bias are visible.
    """
    target = 'xco2_bc_anomaly'
    if target not in df.columns:
        logger.warning("xco2_bc_anomaly not found — skipping XCO₂ partial plot")
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
        ax.set_ylabel('XCO\u2082 BC anomaly residual (ppm)', fontsize=9)
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
        'XCO\u2082 BC anomaly: partial correlation with cloud distance\n'
        'after removing albedo + airmass + cos(SZA) + AOD + \u0394P + CO\u2082_grad + H\u2082O\n'
        'Result: r_resid \u2248 0 — no detectable cloud-proximity signal remains in XCO\u2082',
        fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    _save(fig, outdir, 'xco2_anomaly_partial_vs_cld_dist.png')


# ══════════════════════════════════════════════════════════════════════════════
# Ref-corrected analyses  (Sections R1–R7)
# Clear-sky reference pixels: ref_{k1,k2,alb,exp_int}_{band}_{mean,std}
# ══════════════════════════════════════════════════════════════════════════════

# Registry: (obs_col, ref_mean_col, ref_std_col, diff_col, band_label, term_label, band_color)
_REF_PAIRS: list[tuple] = [
    ('o2a_k1',             'ref_o2a_k1_mean',       'ref_o2a_k1_std',       'dk1_o2a',   'O\u2082A',  'k\u2081',   'C0'),
    ('o2a_k2',             'ref_o2a_k2_mean',       'ref_o2a_k2_std',       'dk2_o2a',   'O\u2082A',  'k\u2082',   'C0'),
    ('wco2_k1',            'ref_wco2_k1_mean',      'ref_wco2_k1_std',      'dk1_wco2',  'WCO\u2082', 'k\u2081',   'C1'),
    ('wco2_k2',            'ref_wco2_k2_mean',      'ref_wco2_k2_std',      'dk2_wco2',  'WCO\u2082', 'k\u2082',   'C1'),
    ('sco2_k1',            'ref_sco2_k1_mean',      'ref_sco2_k1_std',      'dk1_sco2',  'SCO\u2082', 'k\u2081',   'C2'),
    ('sco2_k2',            'ref_sco2_k2_mean',      'ref_sco2_k2_std',      'dk2_sco2',  'SCO\u2082', 'k\u2082',   'C2'),
    ('alb_o2a',            'ref_alb_o2a_mean',      'ref_alb_o2a_std',      'dalb_o2a',  'O\u2082A',  'albedo',    'C0'),
    ('alb_wco2',           'ref_alb_wco2_mean',     'ref_alb_wco2_std',     'dalb_wco2', 'WCO\u2082', 'albedo',    'C1'),
    ('alb_sco2',           'ref_alb_sco2_mean',     'ref_alb_sco2_std',     'dalb_sco2', 'SCO\u2082', 'albedo',    'C2'),
    ('exp_o2a_intercept',  'ref_exp_int_o2a_mean',  'ref_exp_int_o2a_std',  'dexp_o2a',  'O\u2082A',  'exp_int',   'C0'),
    ('exp_wco2_intercept', 'ref_exp_int_wco2_mean', 'ref_exp_int_wco2_std', 'dexp_wco2', 'WCO\u2082', 'exp_int',   'C1'),
    ('exp_sco2_intercept', 'ref_exp_int_sco2_mean', 'ref_exp_int_sco2_std', 'dexp_sco2', 'SCO\u2082', 'exp_int',   'C2'),
]

# r25 reference set (min_cld_dist=25 km) — mirrors _REF_PAIRS with r25_ prefix columns
_R25_PAIRS: list[tuple] = [
    ('o2a_k1',             'r25_o2a_k1_mean',       'r25_o2a_k1_std',       'dr25k1_o2a',   'O\u2082A',  'k\u2081',   'C0'),
    ('o2a_k2',             'r25_o2a_k2_mean',       'r25_o2a_k2_std',       'dr25k2_o2a',   'O\u2082A',  'k\u2082',   'C0'),
    ('wco2_k1',            'r25_wco2_k1_mean',      'r25_wco2_k1_std',      'dr25k1_wco2',  'WCO\u2082', 'k\u2081',   'C1'),
    ('wco2_k2',            'r25_wco2_k2_mean',      'r25_wco2_k2_std',      'dr25k2_wco2',  'WCO\u2082', 'k\u2082',   'C1'),
    ('sco2_k1',            'r25_sco2_k1_mean',      'r25_sco2_k1_std',      'dr25k1_sco2',  'SCO\u2082', 'k\u2081',   'C2'),
    ('sco2_k2',            'r25_sco2_k2_mean',      'r25_sco2_k2_std',      'dr25k2_sco2',  'SCO\u2082', 'k\u2082',   'C2'),
    ('alb_o2a',            'r25_alb_o2a_mean',      'r25_alb_o2a_std',      'dr25alb_o2a',  'O\u2082A',  'albedo',    'C0'),
    ('alb_wco2',           'r25_alb_wco2_mean',     'r25_alb_wco2_std',     'dr25alb_wco2', 'WCO\u2082', 'albedo',    'C1'),
    ('alb_sco2',           'r25_alb_sco2_mean',     'r25_alb_sco2_std',     'dr25alb_sco2', 'SCO\u2082', 'albedo',    'C2'),
    ('exp_o2a_intercept',  'r25_exp_int_o2a_mean',  'r25_exp_int_o2a_std',  'dr25exp_o2a',  'O\u2082A',  'exp_int',   'C0'),
    ('exp_wco2_intercept', 'r25_exp_int_wco2_mean', 'r25_exp_int_wco2_std', 'dr25exp_wco2', 'WCO\u2082', 'exp_int',   'C1'),
    ('exp_sco2_intercept', 'r25_exp_int_sco2_mean', 'r25_exp_int_sco2_std', 'dr25exp_sco2', 'SCO\u2082', 'exp_int',   'C2'),
]


def add_ref_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add obs-ref difference and z-score columns for every entry in _REF_PAIRS.

    New columns:
      d{term}_{band}  = obs - ref_mean          e.g. dk1_o2a
      z{term}_{band}  = (obs - ref_mean)/ref_std e.g. zk1_o2a
    Rows without a ref value remain NaN in the new columns.
    """
    new_cols = {}
    for obs, ref_m, ref_s, dcol, _, _, _ in _REF_PAIRS:
        if obs in df.columns and ref_m in df.columns:
            new_cols[dcol] = df[obs] - df[ref_m]
            zcol = 'z' + dcol[1:]   # 'dk1_o2a' → 'zk1_o2a'
            if ref_s in df.columns:
                new_cols[zcol] = new_cols[dcol] / df[ref_s].replace(0, np.nan)
    return df.assign(**new_cols)


def _has_ref_data(df: pd.DataFrame) -> bool:
    """Return True if at least one ref column is present in df."""
    return any(c.startswith('ref_') for c in df.columns)


def add_r25_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add obs-r25 difference and z-score columns for every entry in _R25_PAIRS.

    New columns:
      d{term}_{band}  = obs - r25_mean          e.g. dr25k1_o2a
      z{term}_{band}  = (obs - r25_mean)/r25_std e.g. zr25k1_o2a
    """
    new_cols = {}
    for obs, ref_m, ref_s, dcol, _, _, _ in _R25_PAIRS:
        if obs in df.columns and ref_m in df.columns:
            new_cols[dcol] = df[obs] - df[ref_m]
            zcol = 'z' + dcol[1:]   # 'dr25k1_o2a' → 'zr25k1_o2a'
            if ref_s in df.columns:
                new_cols[zcol] = new_cols[dcol] / df[ref_s].replace(0, np.nan)
    return df.assign(**new_cols)


def _has_r25_data(df: pd.DataFrame) -> bool:
    """Return True if at least one r25 column is present in df."""
    return any(c.startswith('r25_') for c in df.columns)


def _binned_ref_profile(ax, sdf: pd.DataFrame, diff_col: str,
                        bins, labels, color: str, title: str) -> None:
    """Shared helper: plot binned mean ± SEM (errorbar) / ± std (fill) of diff_col."""
    _full_bin = pd.cut(sdf['cld_dist_km'], bins=bins, labels=labels, right=False)
    sub_mask = sdf[[diff_col, 'cld_dist_km']].notna().all(axis=1)
    sub = sdf[sub_mask]
    sub_bin = _full_bin[sub_mask]
    if len(sub) < 10:
        ax.set_visible(False)
        return
    means = sub.groupby(sub_bin, observed=True)[diff_col].mean().reindex(labels)
    stds  = sub.groupby(sub_bin, observed=True)[diff_col].std().reindex(labels)
    ns    = sub.groupby(sub_bin, observed=True)[diff_col].count().reindex(labels).fillna(0).astype(int)
    sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
    xp = np.arange(len(labels))

    ax.fill_between(xp, (means - stds).values, (means + stds).values,
                    color=color, alpha=0.15, label='\u00b1 1 std')
    ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                capsize=4, color=color, lw=1.5, label='mean \u00b1 SEM')
    ax.axhline(0, color='gray', lw=1, linestyle='--')

    m = sub[diff_col].notna() & sub['cld_dist_km'].notna()
    r_val = stats.pearsonr(sub.loc[m, 'cld_dist_km'], sub.loc[m, diff_col])[0] if m.sum() > 10 else np.nan

    ax.set_xticks(xp)
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_xlabel('Cloud distance (km)', fontsize=9)
    ax.set_title(f'{title}   r={r_val:.3f}', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    finite_means = means.dropna()
    finite_stds  = stds.dropna()
    if finite_means.size and finite_stds.size:
        spread = max(finite_stds.max() * 2, abs(finite_means).max() * 2, 1e-9)
        ax.set_ylim(-spread, spread)


# ── R0: fp − ref scatter vs cloud distance ────────────────────────────────────

def plot_ref_diff_vs_cld_dist(df: pd.DataFrame, outdir: str,
                               max_dist: float = 50,
                               n_roll: int = 100,
                               pairs=None, tag: str = 'ref') -> None:
    """Hexbin scatter + rolling median of (obs − ref_mean) vs cld_dist_km.

    One figure per spectral term type (k1, k2, albedo, exp_int).
    Rows  = spectral bands (O₂A, WCO₂, SCO₂).
    Cols  = surface type (ocean / land).
    y = 0 means the footprint matches its clear-sky reference; positive values
    mean the footprint exceeds the reference.  The rolling median reveals any
    systematic trend with cloud proximity.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    term_groups = [
        ('k\u2081',   [p for p in pairs if p[5] == 'k\u2081'],  f'{tag}_diff_scatter_k1.png'),
        ('k\u2082',   [p for p in pairs if p[5] == 'k\u2082'],  f'{tag}_diff_scatter_k2.png'),
        ('albedo', [p for p in pairs if p[5] == 'albedo'],      f'{tag}_diff_scatter_alb.png'),
        ('exp_int',[p for p in pairs if p[5] == 'exp_int'],     f'{tag}_diff_scatter_exp.png'),
    ]
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for term_lbl, pairs, fname in term_groups:
        avail = [p for p in pairs if p[3] in df.columns]
        if not avail:
            logger.warning(f"No diff columns for term '{term_lbl}' — skipping R0 scatter")
            continue

        nrows, ncols = len(avail), len(subsets)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        if nrows == 1:
            axes = axes[np.newaxis, :]
        if ncols == 1:
            axes = axes[:, np.newaxis]

        for row, (_, _, _, dcol, bl, tl, col) in enumerate(avail):
            for ci, (sfc_name, sdf) in enumerate(subsets):
                ax = axes[row, ci]
                sub = sdf[sdf['cld_dist_km'] <= max_dist]
                x = sub['cld_dist_km'].values
                y = sub[dcol].values
                mask = np.isfinite(x) & np.isfinite(y)
                xm, ym = x[mask], y[mask]

                if len(xm) < 10:
                    ax.set_visible(False)
                    continue

                # clip y to 1–99th percentile to suppress outlier whitespace
                ylo, yhi = np.percentile(ym, 1), np.percentile(ym, 99)
                vm = (ym >= ylo) & (ym <= yhi)

                hb = ax.hexbin(xm[vm], ym[vm], gridsize=55, cmap='YlOrRd',
                               mincnt=1, norm=mcolors.LogNorm())
                plt.colorbar(hb, ax=ax, label='count')

                xs, med, q25, q75 = rolling_median_iqr(xm[vm], ym[vm], n_pts=n_roll)
                ax.plot(xs, med, color=col, lw=2, label='rolling median')
                ax.fill_between(xs, q25, q75, color=col, alpha=0.25, label='IQR')

                ax.axhline(0, color='gray', lw=1.2, linestyle='--')

                r, _ = stats.pearsonr(xm[vm], ym[vm])
                ax.set_title(f'{bl} {tl}  obs\u2212ref — {sfc_name}   r={r:.3f}', fontsize=9)
                ax.set_xlabel('Cloud distance (km)', fontsize=9)
                ax.set_ylabel(f'{bl} {tl}\n(obs \u2212 ref)', fontsize=9)
                ax.legend(fontsize=7)

        fig.suptitle(
            f'[{tag}] Footprint \u2212 clear-sky reference  [{term_lbl}]  vs cloud distance\n'
            'y > 0 \u2192 footprint exceeds reference;  y = 0 \u2192 matches clear-sky baseline',
            fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R1: Coverage bias analysis ────────────────────────────────────────────────

def plot_ref_coverage_bias(df: pd.DataFrame, bins, labels, outdir: str,
                           pairs=None, tag: str = 'ref') -> None:
    """Compare soundings that have vs lack a clear-sky reference per cld_dist bin.

    Tests whether the subset with ref data is representative.  Systematic
    differences reveal selection bias that must be considered when interpreting
    ref-corrected anomalies.  The coverage fraction panel (bottom-right) shows
    how ref availability drops sharply near clouds.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    check_vars = {
        'o2a_k1':            'O\u2082A k\u2081',
        'alb_o2a':           'O\u2082A albedo',
        'exp_o2a_intercept': 'O\u2082A exp_int',
        'sco2_k1':           'SCO\u2082 k\u2081',
        'xco2_bc_anomaly':   'XCO\u2082 BC anomaly (ppm)',
        'airmass':           'Airmass',
    }
    avail = {k: v for k, v in check_vars.items() if k in df.columns}
    ref_flag_col = pairs[0][1]   # first pair's ref_mean_col
    if ref_flag_col not in df.columns or not avail:
        return

    _bin     = pd.cut(df['cld_dist_km'], bins=bins, labels=labels, right=False)
    _has_ref = df[ref_flag_col].notna()
    xp    = np.arange(len(labels))
    width = 0.35

    ncols = 2
    nrows = int(np.ceil((len(avail) + 1) / ncols))   # +1 for coverage panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    for ax, (col, lbl) in zip(axes, avail.items()):
        for gi, (flag, color, grp_lbl) in enumerate([(True, 'C0', 'has ref'),
                                                      (False, 'C3', 'no ref')]):
            flag_mask = _has_ref == flag
            sub = df[flag_mask]
            sub_bin = _bin[flag_mask]
            means = sub.groupby(sub_bin, observed=True)[col].mean().reindex(labels)
            stds  = sub.groupby(sub_bin, observed=True)[col].std().reindex(labels)
            ns    = sub.groupby(sub_bin, observed=True)[col].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            ax.bar(xp + (gi - 0.5) * width, means.fillna(0).values, width=width,
                   color=color, alpha=0.75, label=grp_lbl)
            ax.errorbar(xp + (gi - 0.5) * width, means.fillna(0).values,
                        yerr=sems.values, fmt='none', color='k', capsize=3, lw=1)
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(f'{lbl}: has-ref vs no-ref', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Coverage fraction panel
    cov_ax = axes[len(avail)]
    cov_ax.set_visible(True)
    coverage = _has_ref.groupby(_bin, observed=True).mean() * 100
    cov_ax.bar(xp, coverage.reindex(labels).values, color='steelblue', alpha=0.8)
    cov_ax.set_xticks(xp)
    cov_ax.set_xticklabels(labels, rotation=30, fontsize=8)
    cov_ax.set_xlabel('Cloud distance (km)', fontsize=9)
    cov_ax.set_ylabel('% soundings with ref', fontsize=9)
    cov_ax.set_title('Clear-sky ref coverage vs cloud distance', fontsize=10)
    cov_ax.set_ylim(0, 110)
    cov_ax.grid(axis='y', alpha=0.3)
    for xi, cov in enumerate(coverage.reindex(labels).fillna(0).values):
        cov_ax.text(xi, cov + 1.5, f'{cov:.0f}%', ha='center', fontsize=8, color='navy')

    for ax in axes[len(avail) + 1:]:
        ax.set_visible(False)

    fig.suptitle('Selection bias check: soundings with vs without clear-sky reference\n'
                 '(ideally the two groups match within each cld_dist bin)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'ref_coverage_bias.png')


# ── R2: Ref std profiles (scene heterogeneity proxy) ─────────────────────────

def plot_ref_std_profiles(df: pd.DataFrame, bins, labels, outdir: str) -> None:
    """Plot ref_std (within-reference-pool variability) vs cloud-distance bin.

    A decrease in ref_std near clouds can indicate:
      (a) fewer reference pixels → downward-biased std estimate, or
      (b) genuinely more homogeneous clear-sky corridors adjacent to clouds.
    Ocean and land are shown in separate columns.
    """
    std_vars = [
        ('ref_o2a_k1_std',       'O\u2082A k\u2081 ref \u03c3',       'C0'),
        ('ref_sco2_k1_std',      'SCO\u2082 k\u2081 ref \u03c3',      'C2'),
        ('ref_alb_o2a_std',      'O\u2082A alb ref \u03c3',           'C0'),
        ('ref_alb_sco2_std',     'SCO\u2082 alb ref \u03c3',          'C2'),
        ('ref_exp_int_o2a_std',  'O\u2082A exp_int ref \u03c3',       'C0'),
        ('ref_exp_int_sco2_std', 'SCO\u2082 exp_int ref \u03c3',      'C2'),
    ]
    avail = [(c, l, col) for c, l, col in std_vars if c in df.columns]
    if not avail:
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]
    xp = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 3.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (col, lbl, color) in enumerate(avail):
        for ci, (sfc_name, sdf) in enumerate(subsets):
            ax = axes[row, ci]
            _sdf_bin = pd.cut(sdf['cld_dist_km'], bins=bins, labels=labels, right=False)
            means = sdf.groupby(_sdf_bin, observed=True)[col].mean().reindex(labels)
            ns    = sdf.groupby(_sdf_bin, observed=True)[col].count().reindex(labels).replace(0, np.nan)
            sems  = (sdf.groupby(_sdf_bin, observed=True)[col].std().reindex(labels)
                     / np.sqrt(ns)).fillna(0)
            ax.errorbar(xp, means.values, yerr=sems.values,
                        fmt='o-', capsize=3, color=color, lw=1.5)
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(lbl, fontsize=9)
            ax.set_title(f'{lbl} — {sfc_name}', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Clear-sky reference \u03c3 vs cloud distance\n'
                 '(decreasing near clouds may reflect sampling, not homogeneity)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'ref_std_profiles.png')


# ── R3: Ref-corrected anomaly profiles ────────────────────────────────────────

def plot_ref_corrected_profiles(df: pd.DataFrame, bins, labels, outdir: str) -> None:
    """Binned mean ± SEM/std of (obs − ref_mean) for k1, k2, albedo, exp_intercept.

    Four figures (one per variable type), each with 3 rows (bands O2A/WCO2/SCO2)
    × 2 columns (ocean / land).  The y = 0 line marks where obs matches its
    clear-sky baseline; deviations reveal cloud-adjacency effects that survive
    local scene conditioning.
    """
    term_groups = [
        ('k\u2081',   [p for p in _REF_PAIRS if p[5] == 'k\u2081'],   'ref_corrected_k1_profiles.png'),
        ('k\u2082',   [p for p in _REF_PAIRS if p[5] == 'k\u2082'],   'ref_corrected_k2_profiles.png'),
        ('albedo', [p for p in _REF_PAIRS if p[5] == 'albedo'],   'ref_corrected_alb_profiles.png'),
        ('exp_int',[p for p in _REF_PAIRS if p[5] == 'exp_int'],  'ref_corrected_exp_profiles.png'),
    ]
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for term_lbl, pairs, fname in term_groups:
        avail = [p for p in pairs if p[3] in df.columns]
        if not avail:
            continue

        fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4 * len(avail)))
        if len(avail) == 1:
            axes = axes[np.newaxis, :]

        for row, (_, _, _, dcol, bl, tl, col) in enumerate(avail):
            for ci, (sfc_name, sdf) in enumerate(subsets):
                ax = axes[row, ci]
                _binned_ref_profile(ax, sdf, dcol, bins, labels, col,
                                    f'{bl} {tl} \u2212 ref — {sfc_name}')
                if ax.get_visible():
                    ax.set_ylabel(f'{bl} {tl}\nobs \u2212 ref', fontsize=8)

        fig.suptitle(f'Ref-corrected anomaly  [{term_lbl}]:  obs \u2212 clear-sky ref\n'
                     'y = 0 → obs matches clear-sky baseline',
                     fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R4: Ref z-score profiles ──────────────────────────────────────────────────

def plot_ref_zscore_profiles(df: pd.DataFrame, bins, labels, outdir: str) -> None:
    """Binned mean ± SEM of z = (obs − ref_mean) / ref_std vs cloud distance.

    Normalising by ref_std puts all variables on the same scale (units of
    clear-sky natural variability).  A z-score of ±1 means the obs deviates
    by one standard deviation of the local reference distribution.
    """
    term_groups = [
        ('k\u2081',   [p for p in _REF_PAIRS if p[5] == 'k\u2081'],   'ref_zscore_k1_profiles.png'),
        ('k\u2082',   [p for p in _REF_PAIRS if p[5] == 'k\u2082'],   'ref_zscore_k2_profiles.png'),
        ('albedo', [p for p in _REF_PAIRS if p[5] == 'albedo'],   'ref_zscore_alb_profiles.png'),
        ('exp_int',[p for p in _REF_PAIRS if p[5] == 'exp_int'],  'ref_zscore_exp_profiles.png'),
    ]
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for term_lbl, pairs, fname in term_groups:
        avail = [p for p in pairs if p[3] in df.columns]
        if not avail:
            continue

        fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4 * len(avail)))
        if len(avail) == 1:
            axes = axes[np.newaxis, :]

        for row, (_, _, _, dcol, bl, tl, col) in enumerate(avail):
            zcol = 'z' + dcol[1:]   # e.g. 'zk1_o2a'
            if zcol not in df.columns:
                continue
            for ci, (sfc_name, sdf) in enumerate(subsets):
                ax = axes[row, ci]
                _binned_ref_profile(ax, sdf, zcol, bins, labels, col,
                                    f'{bl} {tl} z-score — {sfc_name}')
                if ax.get_visible():
                    ax.set_ylabel(f'{bl} {tl}\n(obs\u2212ref)/\u03c3_ref', fontsize=8)

        fig.suptitle(f'Ref z-score  [{term_lbl}]:  (obs \u2212 ref) / \u03c3_ref\n'
                     'Units of clear-sky natural variability',
                     fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R5: Signal hierarchy with ref normalization ───────────────────────────────

def plot_ref_signal_hierarchy(df: pd.DataFrame, outdir: str) -> None:
    """Bar chart of r(cld_dist, obs − ref) for all ref-corrected variables.

    Companion to plot_signal_hierarchy but using diff columns (obs − ref_mean)
    instead of raw obs.  Variables that retain large |r| after subtracting the
    clear-sky reference carry genuine cloud-adjacency signal not explained by
    local scene co-variation.  Ocean and land are shown side-by-side.
    """
    feat_groups = [
        ('dk1_o2a',   'O\u2082A',  'k\u2081',   ''),
        ('dk2_o2a',   'O\u2082A',  'k\u2082',   '///'),
        ('dk1_wco2',  'WCO\u2082', 'k\u2081',   ''),
        ('dk2_wco2',  'WCO\u2082', 'k\u2082',   '///'),
        ('dk1_sco2',  'SCO\u2082', 'k\u2081',   ''),
        ('dk2_sco2',  'SCO\u2082', 'k\u2082',   '///'),
        ('dalb_o2a',  'O\u2082A',  'alb',       '...'),
        ('dalb_wco2', 'WCO\u2082', 'alb',       '...'),
        ('dalb_sco2', 'SCO\u2082', 'alb',       '...'),
        ('dexp_o2a',  'O\u2082A',  'exp_int',   '|||'),
        ('dexp_wco2', 'WCO\u2082', 'exp_int',   '|||'),
        ('dexp_sco2', 'SCO\u2082', 'exp_int',   '|||'),
    ]
    avail = [(col, bl, tl, ht) for col, bl, tl, ht in feat_groups if col in df.columns]
    if not avail:
        return

    band_colors  = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    x = np.arange(len(avail))

    for ax, (sfc_name, sdf) in zip(axes, subsets):
        rs = []
        for col, _, _, _ in avail:
            m = sdf[col].notna() & sdf['cld_dist_km'].notna()
            r = stats.pearsonr(sdf.loc[m, 'cld_dist_km'], sdf.loc[m, col])[0] if m.sum() > 10 else np.nan
            rs.append(r)

        colors  = [band_colors[bl] for _, bl, _, _  in avail]
        hatches = [ht              for _, _,  _, ht in avail]
        bars = ax.bar(x, rs, color=colors, hatch=hatches, edgecolor='white', alpha=0.85)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{bl}\n{tl}' for _, bl, tl, _ in avail],
                           fontsize=7.5, rotation=40, ha='right')
        ax.set_ylabel('Pearson r(cld_dist, obs\u2212ref)', fontsize=9)
        ax.set_title(f'{sfc_name} (n={sdf["cld_dist_km"].notna().sum():,})', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, r in zip(bars, rs):
            if np.isfinite(r):
                offset = 0.005 if r >= 0 else -0.005
                ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                        f'{r:.2f}', ha='center',
                        va='bottom' if r >= 0 else 'top', fontsize=6.5)

    from matplotlib.patches import Patch
    legend_handles  = [Patch(facecolor=c, label=b) for b, c in band_colors.items()]
    legend_handles += [Patch(facecolor='gray', hatch=h, label=t, alpha=0.7)
                       for t, h in [('k\u2081', ''), ('k\u2082', '///'),
                                    ('alb', '...'), ('exp_int', '|||')]]
    axes[0].legend(handles=legend_handles, fontsize=7, ncol=2, loc='lower left',
                   title='Band / Term', title_fontsize=7)
    fig.suptitle('Ref-corrected signal hierarchy: r(cld_dist, obs \u2212 ref)\n'
                 'Variables retaining large |r| carry genuine cloud-adjacency signal',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, 'ref_signal_hierarchy.png')


# ── R6: Albedo-decoupled exp_intercept in ref-corrected space ─────────────────

def plot_ref_alb_decoupled_exp(df: pd.DataFrame, bins, labels, outdir: str) -> None:
    """OLS-remove dalb from dexp, then plot residual vs cloud distance.

    In ref-corrected space: dexp ~ const + dalb (per band, per surface type).
    The residual isolates the non-albedo component of the cloud-edge signal —
    i.e. photon path-length or scattering effects independent of surface changes.
    Compares r_raw(dexp) vs r_resid to quantify how much cloud signal survives
    after accounting for albedo co-variation with cloud proximity.
    """
    band_defs = [
        ('dexp_o2a',  'dalb_o2a',  'O\u2082A',  'C0'),
        ('dexp_wco2', 'dalb_wco2', 'WCO\u2082', 'C1'),
        ('dexp_sco2', 'dalb_sco2', 'SCO\u2082', 'C2'),
    ]
    avail = [(de, da, nm, col) for de, da, nm, col in band_defs
             if de in df.columns and da in df.columns]
    if not avail:
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]
    xp = np.arange(len(labels))

    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 4.5 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]

    for row, (dexp_col, dalb_col, nm, col) in enumerate(avail):
        for ci, (sfc_name, sdf) in enumerate(subsets):
            ax = axes[row, ci]
            req = [dexp_col, dalb_col, 'cld_dist_km']
            m   = sdf[req].notna().all(axis=1)
            sdf_m = sdf[m].copy()
            if len(sdf_m) < 50:
                ax.set_visible(False)
                continue

            # OLS: dexp ~ const + dalb
            X = np.column_stack([np.ones(len(sdf_m)), sdf_m[dalb_col].values])
            y = sdf_m[dexp_col].values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            sdf_m['_resid'] = y - X @ coef

            r_raw, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m[dexp_col])
            r_res, _ = stats.pearsonr(sdf_m['cld_dist_km'], sdf_m['_resid'])

            sdf_m['_bin'] = pd.cut(sdf_m['cld_dist_km'], bins=bins,
                                   labels=labels, right=False)
            means = sdf_m.groupby('_bin', observed=True)['_resid'].mean().reindex(labels)
            stds  = sdf_m.groupby('_bin', observed=True)['_resid'].std().reindex(labels)
            ns    = sdf_m.groupby('_bin', observed=True)['_resid'].count().reindex(labels).fillna(0).astype(int)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)

            ax.fill_between(xp, (means - stds).values, (means + stds).values,
                            color=col, alpha=0.15, label='\u00b1 1 std')
            ax.errorbar(xp, means.values, yerr=sems.values, fmt='o-',
                        capsize=4, color=col, lw=1.5,
                        label=f'mean \u00b1 SEM  r={r_res:.3f}')
            ax.axhline(0, color='gray', lw=0.8, linestyle='--')
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'{nm} \u0394exp_int | \u0394alb removed', fontsize=9)
            ax.set_title(f'{nm} — {sfc_name}  |  r_raw={r_raw:.3f} \u2192 r_resid={r_res:.3f}',
                         fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            finite_means = means.dropna()
            finite_stds  = stds.dropna()
            if finite_means.size and finite_stds.size:
                spread = max(finite_stds.max() * 2, abs(finite_means).max() * 2, 1e-9)
                ax.set_ylim(-spread, spread)

    fig.suptitle('\u0394exp_int residual after removing \u0394alb (OLS) — ref-corrected space\n'
                 'Surviving signal is independent of surface reflectance co-variation',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, 'ref_alb_decoupled_exp_residuals.png')


# ── R7: Obs vs ref scatter at matched geometry ────────────────────────────────

def plot_obs_vs_ref_scatter(df: pd.DataFrame, outdir: str) -> None:
    """Hexbin scatter of obs vs ref_mean colored by density, with 1:1 line.

    Points above the 1:1 line: obs exceeds clear-sky reference.
    The rolling median (colored by mean cld_dist in each hex) reveals whether
    cloud-adjacent soundings systematically depart from the clear-sky baseline.
    Three variable types (k1, albedo, exp_int) × two surface types (ocean/land).
    """
    var_defs = [
        ('o2a_k1',            'ref_o2a_k1_mean',       'O\u2082A k\u2081',     'C0'),
        ('sco2_k1',           'ref_sco2_k1_mean',      'SCO\u2082 k\u2081',    'C2'),
        ('alb_o2a',           'ref_alb_o2a_mean',      'O\u2082A albedo',       'C0'),
        ('alb_sco2',          'ref_alb_sco2_mean',     'SCO\u2082 albedo',      'C2'),
        ('exp_o2a_intercept', 'ref_exp_int_o2a_mean',  'O\u2082A exp_int',      'C0'),
        ('exp_sco2_intercept','ref_exp_int_sco2_mean', 'SCO\u2082 exp_int',     'C2'),
    ]
    avail = [(oc, rc, lbl, col) for oc, rc, lbl, col in var_defs
             if oc in df.columns and rc in df.columns]
    if not avail:
        return

    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for sfc_name, sdf in subsets:
        ncols = 2
        nrows = int(np.ceil(len(avail) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
        axes = axes.flatten()

        for ax, (obs_col, ref_col, lbl, col) in zip(axes, avail):
            m = sdf[obs_col].notna() & sdf[ref_col].notna()
            x = sdf.loc[m, ref_col].values   # reference (clear-sky)
            y = sdf.loc[m, obs_col].values    # observation
            if len(x) < 10:
                ax.set_visible(False)
                continue

            # clip to 1–99th percentile for display
            xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
            ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
            vm = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)

            hb = ax.hexbin(x[vm], y[vm], gridsize=60, cmap='YlOrRd',
                           mincnt=1, norm=mcolors.LogNorm())
            plt.colorbar(hb, ax=ax, label='count')

            # 1:1 reference line
            lo = min(xlo, ylo)
            hi = max(xhi, yhi)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='1:1 line')

            # rolling median of y vs x
            xs, med, q25, q75 = rolling_median_iqr(x[vm], y[vm], n_pts=60)
            ax.plot(xs, med, color=col, lw=2, label='rolling median')
            ax.fill_between(xs, q25, q75, color=col, alpha=0.25)

            r, _ = stats.pearsonr(x[vm], y[vm])
            bias = np.mean(y[vm] - x[vm])
            ax.set_xlabel(f'ref_mean ({lbl})', fontsize=9)
            ax.set_ylabel(f'obs ({lbl})', fontsize=9)
            ax.set_title(f'{lbl}  |  r={r:.3f}  bias={bias:+.4f}', fontsize=9)
            ax.legend(fontsize=7)

        for ax in axes[len(avail):]:
            ax.set_visible(False)

        fig.suptitle(f'Obs vs clear-sky reference — {sfc_name}\n'
                     'Points above 1:1 line: obs exceeds clear-sky baseline',
                     fontsize=12)
        fig.tight_layout()
        fname = f'obs_vs_ref_scatter_{sfc_name.lower()}.png'
        _save(fig, outdir, fname)


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
    # # TODO: remove this once the π factor is absorbed into oco_fp_spec_anal.py
    # # Scale both obs and ref exp_int so they stay on the same scale for diffs.
    # _exp_int_cols = [
    #     'exp_o2a_intercept',    'exp_wco2_intercept',    'exp_sco2_intercept',
    #     'ref_exp_int_o2a_mean', 'ref_exp_int_wco2_mean', 'ref_exp_int_sco2_mean',
    #     'ref_exp_int_o2a_std',  'ref_exp_int_wco2_std',  'ref_exp_int_sco2_std',
    # ]
    # for _col in _exp_int_cols:
    #     if _col in df.columns:
    #         df[_col] = df[_col]

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    # ── Section 1: signal hierarchy (full df, internal ocean/land split) ─────
    overall_outdir = str(result_dir / 'figures' / 'cld_dist_analysis')
    logger.info("Section 1: Plotting signal hierarchy (r vs cld_dist) …")
    plot_signal_hierarchy(df, overall_outdir)

    # ── Section 1b: residual signal hierarchy ─────────────────────────────────
    logger.info("Section 1b: Plotting residual signal hierarchy …")
    plot_residual_signal_hierarchy(df, overall_outdir)

    # ── Section 2b: exp_intercept albedo residuals ────────────────────────────
    logger.info("Section 2b: Plotting exp_intercept albedo residuals …")
    plot_exp_intercept_albedo_residuals(df, bins, labels, overall_outdir)

    # ── Section 2e: albedo vs exp_intercept divergence ───────────────────────
    logger.info("Section 2e: Plotting alb vs exp_intercept divergence …")
    plot_alb_exp_divergence(df, bins, labels, overall_outdir)

    # ── Section 2f: exp/alb ratio residuals ──────────────────────────────────
    logger.info("Section 2f: Plotting exp/alb ratio residuals …")
    plot_exp_alb_ratio_residuals(df, bins, labels, overall_outdir)

    # ── Section 3e: k1, k2, k3 albedo residuals ──────────────────────────────
    logger.info("Section 3e: Plotting k1/k2/k3 albedo residuals …")
    plot_k_albedo_residuals(df, bins, labels, overall_outdir)

    # ── Section 2d: exp_intercept inter-band coherence ────────────────────────
    logger.info("Section 2d: Plotting exp_intercept inter-band coherence …")
    plot_exp_intercept_interband_coherence(df, overall_outdir)

    # ── Section 3c: higher-order k profiles (k3 for SCO₂ and WCO₂) ──────────
    logger.info("Section 3c: Plotting higher-order k3 profiles …")
    plot_higher_order_k_profiles(df, bins, labels, overall_outdir)

    # ── Sections R1–R7: ref-corrected analyses ────────────────────────────────
    if _has_ref_data(df):
        ref_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / 'ref_corrected')
        logger.info("Adding ref-corrected anomaly columns …")
        df_r = add_ref_anomalies(df)

        logger.info("R0: fp − ref scatter vs cloud distance …")
        plot_ref_diff_vs_cld_dist(df_r, ref_outdir)

        logger.info("R1: Ref coverage bias analysis …")
        plot_ref_coverage_bias(df_r, bins, labels, ref_outdir)

        logger.info("R2: Ref std profiles (scene heterogeneity) …")
        plot_ref_std_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R3: Ref-corrected anomaly profiles …")
        plot_ref_corrected_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R4: Ref z-score profiles …")
        plot_ref_zscore_profiles(df_r, bins, labels, ref_outdir)

        logger.info("R5: Ref-corrected signal hierarchy …")
        plot_ref_signal_hierarchy(df_r, ref_outdir)

        logger.info("R6: Ref albedo-decoupled exp_intercept residuals …")
        plot_ref_alb_decoupled_exp(df_r, bins, labels, ref_outdir)

        logger.info("R7: Obs vs ref scatter …")
        plot_obs_vs_ref_scatter(df_r, ref_outdir)

        logger.info(f"All ref-corrected figures written to {ref_outdir}")
        del df_r
        gc.collect()
    else:
        logger.warning("No ref_* columns found — skipping Sections R1–R7")

    # ── surface-type loop: process ocean then land sequentially ───────────────
    sfc_codes = {'ocean': 0, 'land': 1} if 'sfc_type' in df.columns else {'all': None}

    for sfc_name, sfc_code in sfc_codes.items():
        sdf = df[df['sfc_type'] == sfc_code] if sfc_code is not None else df
        logger.info(f"\n{'='*55}\nRunning analysis for surface type: {sfc_name.upper()}\n{'='*55}")
        logger.info(f"  {sfc_name} soundings: {len(sdf):,}")
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
        logger.info("Plotting XCO2 anomaly partial correlation (cld_dist) …")
        plot_xco2_anomaly_partial(sdf, bins, labels, sfc_outdir)

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

        del sdf
        gc.collect()


if __name__ == '__main__':
    main()
