"""
ca_pca.py
=========
PCA analysis of spectral / atmospheric feature variables (all non-XCO2 columns).

Contents
--------
- get_pca_features          Select available feature columns (k1/k2/k3, exp_intercept,
                             albedo, ancillary) — excludes all xco2_* columns.
- fit_pca                   StandardScaler + sklearn PCA; returns (pca, pc_scores_df).
- plot_pca_scree             Explained variance ratio bar chart + cumulative line.
- plot_pca_loadings          Heatmap of component loadings (features × PCs).
- plot_pca_scores_vs_cld_dist  Binned mean ± SEM of PC1–PC3 vs cld_dist_km.
- plot_pca_biplot            PC1 vs PC2 scatter colored by cld_dist, loading arrows.
- plot_pca_correlation_with_cld_dist  Pearson r(PC_i, cld_dist_km) bar chart.
- run_pca_analysis           Orchestrates all plots for one surface subset.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ca_utils import _save, bin_by_cld_dist

logger = logging.getLogger(__name__)

# ── feature registry ──────────────────────────────────────────────────────────
# Ordered by band then term; all xco2_* are intentionally absent.
_CANDIDATE_FEATURES = [
    # k1
    'o2a_k1',  'wco2_k1', 'sco2_k1',
    # k2
    'o2a_k2',  'wco2_k2', 'sco2_k2',
    # k3
    'o2a_k3',  'wco2_k3', 'sco2_k3',
    # exp intercepts
    'exp_o2a_intercept', 'exp_wco2_intercept', 'exp_sco2_intercept',
    # albedo
    'alb_o2a', 'alb_wco2', 'alb_sco2',
    # ancillary
    'aod_total', 'dp', 'fp_area_km2',
]

# Human-readable labels for plots
_FEATURE_LABELS = {
    'o2a_k1':  'O₂A k₁',  'wco2_k1': 'WCO₂ k₁', 'sco2_k1': 'SCO₂ k₁',
    'o2a_k2':  'O₂A k₂',  'wco2_k2': 'WCO₂ k₂', 'sco2_k2': 'SCO₂ k₂',
    'o2a_k3':  'O₂A k₃',  'wco2_k3': 'WCO₂ k₃', 'sco2_k3': 'SCO₂ k₃',
    'exp_o2a_intercept':  'exp O₂A',
    'exp_wco2_intercept': 'exp WCO₂',
    'exp_sco2_intercept': 'exp SCO₂',
    'alb_o2a':  'alb O₂A',  'alb_wco2': 'alb WCO₂', 'alb_sco2': 'alb SCO₂',
    'aod_total': 'AOD',
    'dp':        'ΔP',
    'fp_area_km2': 'FP area',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_pca_features(df: pd.DataFrame) -> list[str]:
    """Return available feature columns from _CANDIDATE_FEATURES."""
    present = [c for c in _CANDIDATE_FEATURES if c in df.columns]
    if not present:
        raise ValueError("No PCA feature columns found in DataFrame.")
    logger.info(f"PCA features ({len(present)}): {present}")
    return present


def fit_pca(df: pd.DataFrame, features: list[str]) -> tuple[PCA, pd.DataFrame]:
    """
    Standardise features and fit PCA on rows with no NaN.

    Returns
    -------
    pca : fitted sklearn PCA (n_components = min(n_features, 8))
    pc_scores : DataFrame of PC scores aligned to the clean-row index
    """
    X_raw = df[features].values.astype(np.float32)
    valid = np.isfinite(X_raw).all(axis=1)
    X_clean = X_raw[valid]
    logger.info(f"PCA fit: {valid.sum():,} / {len(X_raw):,} rows after NaN drop")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    n_components = min(len(features), 8)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    pc_scores = pd.DataFrame(scores, columns=pc_cols,
                             index=df.index[valid])
    return pca, pc_scores


# ── plot functions ────────────────────────────────────────────────────────────

def plot_pca_scree(pca: PCA, outdir: str, tag: str = '') -> None:
    """Bar chart of explained variance ratio + cumulative line."""
    evr = pca.explained_variance_ratio_
    n = len(evr)
    pcs = [f'PC{i+1}' for i in range(n)]
    cumulative = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(max(6, n * 0.9), 4))
    bars = ax1.bar(pcs, evr * 100, color='steelblue', alpha=0.8, label='Individual')
    ax1.set_ylabel('Explained variance (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, max(evr) * 130)

    # annotate each bar
    for bar, v in zip(bars, evr):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(pcs, cumulative * 100, color='tomato', marker='o',
             lw=2, label='Cumulative')
    ax2.axhline(90, color='tomato', lw=1, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Cumulative variance (%)', color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2.set_ylim(0, 110)

    fig.suptitle(f'PCA scree plot{" — " + tag if tag else ""}', fontsize=12)
    fig.tight_layout()
    fname = f'pca_scree{"_" + tag if tag else ""}.png'
    _save(fig, outdir, fname)


def plot_pca_loadings(pca: PCA, features: list[str], outdir: str,
                      tag: str = '') -> None:
    """Heatmap of PCA loadings: rows = original features, columns = PCs."""
    labels = [_FEATURE_LABELS.get(f, f) for f in features]
    loadings = pca.components_.T          # shape (n_features, n_components)
    n_pc = loadings.shape[1]
    pc_cols = [f'PC{i+1}' for i in range(n_pc)]

    # annotate PCs with % variance
    pc_labels = [f'PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)'
                 for i in range(n_pc)]

    fig_h = max(4, len(features) * 0.45 + 1.5)
    fig_w = max(6, n_pc * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.abs(loadings).max()
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Loading')

    ax.set_xticks(range(n_pc))
    ax.set_xticklabels(pc_labels, fontsize=9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels, fontsize=9)

    # annotate cells with numeric values
    for i in range(len(features)):
        for j in range(n_pc):
            ax.text(j, i, f'{loadings[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(loadings[i, j]) > 0.4 * vmax else 'black')

    ax.set_title(f'PCA loadings{" — " + tag if tag else ""}', fontsize=11)
    fig.tight_layout()
    fname = f'pca_loadings_heatmap{"_" + tag if tag else ""}.png'
    _save(fig, outdir, fname)


def plot_pca_scores_vs_cld_dist(pc_scores: pd.DataFrame,
                                 df: pd.DataFrame,
                                 bins, labels,
                                 outdir: str,
                                 tag: str = '',
                                 n_pc_show: int = 3) -> None:
    """Binned mean ± SEM of PC1–PC_n vs cld_dist_km."""
    # align cloud distance to the valid-row index of pc_scores
    cld = df.loc[pc_scores.index, 'cld_dist_km']
    bin_col = pd.cut(cld, bins=bins, labels=labels, right=False)

    n_show = min(n_pc_show, pc_scores.shape[1])
    pcs = pc_scores.columns[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4), sharey=False)
    if n_show == 1:
        axes = [axes]

    colors = ['C0', 'C1', 'C2', 'C3']
    for ax, pc, col in zip(axes, pcs, colors):
        grouped = pc_scores[pc].groupby(bin_col)
        means = grouped.mean()
        sems  = grouped.sem()

        ax.bar(range(len(labels)), means.reindex(labels).values,
               yerr=sems.reindex(labels).values,
               color=col, alpha=0.75, capsize=4)
        ax.axhline(0, color='k', lw=0.8, linestyle='--')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_xlabel('Cloud distance (km)', fontsize=9)
        ax.set_ylabel(f'{pc} score', fontsize=9)

        evr_pct = ''
        pc_idx = int(pc[2:]) - 1
        # try to read evr from outer scope if available — just label the title
        ax.set_title(f'{pc} mean score vs cloud distance', fontsize=10)

        # overlay n per bin
        counts = grouped.count().reindex(labels).values
        y_min = ax.get_ylim()[0]
        for xi, cnt in enumerate(counts):
            if cnt and not np.isnan(cnt):
                ax.text(xi, y_min, f'n={int(cnt):,}',
                        ha='center', va='bottom', fontsize=6, color='gray')

    fig.suptitle(f'PC scores vs cloud distance{" — " + tag if tag else ""}',
                 fontsize=11)
    fig.tight_layout()
    fname = f'pca_scores_vs_cld_dist{"_" + tag if tag else ""}.png'
    _save(fig, outdir, fname)


def plot_pca_biplot(pc_scores: pd.DataFrame,
                    pca: PCA,
                    features: list[str],
                    df: pd.DataFrame,
                    outdir: str,
                    tag: str = '',
                    max_points: int = 40_000,
                    n_arrows: int = 6) -> None:
    """
    PC1 vs PC2 scatter colored by cld_dist_km, with loading arrows
    for the top-n variables (largest L2 norm in PC1-PC2 plane).
    """
    cld = df.loc[pc_scores.index, 'cld_dist_km'].values

    # subsample for plotting speed
    n = len(pc_scores)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_points, replace=False)
        x_plot = pc_scores['PC1'].values[idx]
        y_plot = pc_scores['PC2'].values[idx]
        c_plot = cld[idx]
    else:
        x_plot = pc_scores['PC1'].values
        y_plot = pc_scores['PC2'].values
        c_plot = cld

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x_plot, y_plot, c=c_plot, cmap='plasma_r',
                    s=3, alpha=0.4, rasterized=True,
                    norm=mcolors.LogNorm(vmin=max(c_plot.min(), 0.1),
                                         vmax=c_plot.max()))
    plt.colorbar(sc, ax=ax, label='Cloud distance (km)')

    # loading arrows — pick top-n by magnitude in PC1–PC2 plane
    loadings = pca.components_[:2].T          # (n_features, 2)
    magnitudes = np.linalg.norm(loadings, axis=1)
    top_idx = np.argsort(magnitudes)[-n_arrows:]

    # scale arrows to ~30% of axis range
    scale = 0.3 * max(np.abs(x_plot).max(), np.abs(y_plot).max())
    for i in top_idx:
        lx, ly = loadings[i]
        feat_label = _FEATURE_LABELS.get(features[i], features[i])
        ax.annotate('', xy=(lx * scale, ly * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        ax.text(lx * scale * 1.12, ly * scale * 1.12, feat_label,
                ha='center', va='center', fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.axvline(0, color='gray', lw=0.5, linestyle='--')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.set_title(f'PCA biplot (PC1 vs PC2){" — " + tag if tag else ""}', fontsize=11)
    ax.set_facecolor('#1a1a2e')
    fig.tight_layout()
    fname = f'pca_biplot{"_" + tag if tag else ""}.png'
    _save(fig, outdir, fname)


def plot_pca_correlation_with_cld_dist(pca: PCA,
                                        pc_scores: pd.DataFrame,
                                        df: pd.DataFrame,
                                        outdir: str,
                                        tag: str = '') -> None:
    """Bar chart of Pearson r(PC_i, cld_dist_km) for all PCs."""
    from scipy import stats as scipy_stats

    cld = df.loc[pc_scores.index, 'cld_dist_km'].values
    valid = np.isfinite(cld)
    cld_v = cld[valid]

    rs, ps = [], []
    for pc in pc_scores.columns:
        s = pc_scores[pc].values[valid]
        r, p = scipy_stats.pearsonr(s, cld_v)
        rs.append(r)
        ps.append(p)

    n_pc = len(rs)
    pcs = pc_scores.columns.tolist()
    evr = pca.explained_variance_ratio_
    pc_labels = [f'{pc}\n({evr[i]*100:.1f}%)' for i, pc in enumerate(pcs)]

    colors = ['tomato' if r > 0 else 'steelblue' for r in rs]
    fig, ax = plt.subplots(figsize=(max(6, n_pc * 0.9), 4))
    bars = ax.bar(range(n_pc), rs, color=colors, alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(n_pc))
    ax.set_xticklabels(pc_labels, fontsize=9)
    ax.set_ylabel('Pearson r', fontsize=10)
    ax.set_xlabel('Principal component', fontsize=10)

    for bar, r, p in zip(bars, rs, ps):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ypos = r + 0.005 * np.sign(r) if r != 0 else 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{r:.3f}{sig}', ha='center', va='bottom' if r >= 0 else 'top',
                fontsize=8)

    ax.set_title(f'Pearson r(PC, cld_dist_km){" — " + tag if tag else ""}',
                 fontsize=11)
    fig.tight_layout()
    fname = f'pca_correlation_with_cld_dist{"_" + tag if tag else ""}.png'
    _save(fig, outdir, fname)


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_pca_analysis(df: pd.DataFrame, bins, labels,
                     outdir: str, tag: str = '') -> None:
    """
    Run full PCA suite on *df* and write all figures to *outdir*.

    Parameters
    ----------
    df      : filtered DataFrame (one surface type or combined)
    bins    : cld_dist bin edges (list of numbers)
    labels  : cld_dist bin labels (list of strings)
    outdir  : output directory path
    tag     : suffix added to each figure filename (e.g. 'ocean', 'land')
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    features = get_pca_features(df)
    if len(features) < 2:
        logger.warning(f"Too few features for PCA ({len(features)}) — skipping")
        return

    pca, pc_scores = fit_pca(df, features)

    logger.info(f"[PCA{' ' + tag if tag else ''}] Scree plot …")
    plot_pca_scree(pca, outdir, tag=tag)

    logger.info(f"[PCA{' ' + tag if tag else ''}] Loadings heatmap …")
    plot_pca_loadings(pca, features, outdir, tag=tag)

    logger.info(f"[PCA{' ' + tag if tag else ''}] Scores vs cloud distance …")
    plot_pca_scores_vs_cld_dist(pc_scores, df, bins, labels, outdir, tag=tag)

    if pc_scores.shape[1] >= 2:
        logger.info(f"[PCA{' ' + tag if tag else ''}] Biplot …")
        plot_pca_biplot(pc_scores, pca, features, df, outdir, tag=tag)

    logger.info(f"[PCA{' ' + tag if tag else ''}] Correlation with cld_dist …")
    plot_pca_correlation_with_cld_dist(pca, pc_scores, df, outdir, tag=tag)

    logger.info(f"[PCA{' ' + tag if tag else ''}] All figures written to {outdir}")
