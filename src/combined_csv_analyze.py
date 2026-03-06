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
from matplotlib.gridspec import GridSpec
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
            r, p = stats.pearsonr(xm, ym)
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
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(strat_labels)))

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
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(strat_labels)))

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
    df, clipped, bin_labels = _build_strata(df, strat_var, edges, unit)
    if df is None:
        return

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

    # Overlay comparison figures — all strata on one plot
    logger.info(f"    Generating overlay figures for '{strat_var}' …")
    plot_k1_k2_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
    plot_xco2_anomaly_binned_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    storage_dir = get_storage_dir()
    result_dir  = storage_dir / 'results'
    csv_dir     = result_dir / 'csv_collection'
    # ── load ──────────────────────────────────────────────────────────────────
    df = load_data(csv_dir, parquet_fname='combined_2016_2020_dates.parquet')

    # ── quality filter (snow excluded, surface split done below) ──────────────
    df = apply_quality_filter(df)

    # ── cloud-distance bins ───────────────────────────────────────────────────
    edges  = [0, 2, 5, 10, 15, 20, 30, 50]
    bins, labels = cld_dist_bins(edges)

    # ── split by surface type ─────────────────────────────────────────────────
    subsets = split_by_surface(df)

    for sfc_name, sdf in subsets.items():
        logger.info(f"\n{'='*55}\nRunning analysis for surface type: {sfc_name.upper()}\n{'='*55}")
        sfc_outdir = str(result_dir / 'figures' / 'cld_dist_analysis' / sfc_name)

        print_summary_stats(sdf, bins, labels)

        logger.info("Plotting distributions vs cloud distance …")
        plot_distributions_vs_cld_dist(sdf, bins, labels, sfc_outdir)

        logger.info("Plotting k1/k2 scatter vs cloud distance …")
        plot_k1_k2_vs_cld_dist(sdf, sfc_outdir)

        logger.info("Plotting k2/k1 ratio vs cloud distance …")
        plot_k2_over_k1_vs_cld_dist(sdf, sfc_outdir)

        logger.info("Plotting k1/k2 binned profiles …")
        plot_k1_k2_binned_profile(sdf, bins, labels, sfc_outdir)

        logger.info("Plotting k1 vs k2 joint colored by cld_dist …")
        plot_k1_k2_joint(sdf, sfc_outdir)

        logger.info("Plotting XCO2 anomaly vs predictors …")
        plot_xco2_anomaly_vs_key_vars(sdf, sfc_outdir)

        logger.info("Plotting XCO2 anomaly vs cld_dist binned …")
        plot_xco2_anomaly_vs_cld_dist_binned(sdf, bins, labels, sfc_outdir)

        logger.info("Plotting correlation heat-map …")
        plot_xco2_anomaly_correlations(sdf, sfc_outdir)

        logger.info(f"All figures for {sfc_name} written to {sfc_outdir}")

        # ── stratified analyses ───────────────────────────────────────────────
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
