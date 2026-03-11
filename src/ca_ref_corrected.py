"""
ca_ref_corrected.py
===================
Ref-corrected analysis functions extracted from combined_analyze.py (Sections R0–R7).

Contents
--------
- _REF_PAIRS               Registry of (obs, ref_mean, ref_std, diff_col, band, term, color) × 15 (k1+k2+k3 × 3 bands + alb × 3 + exp_int × 3)
- _R25_PAIRS               Same structure for r25_* reference (min_cld_dist=25 km)
- add_ref_anomalies        Compute obs-ref diff and z-score columns
- add_r25_anomalies        Compute obs-r25 diff and z-score columns
- _has_ref_data            Presence check for ref_* columns
- _has_r25_data            Presence check for r25_* columns
- _binned_ref_profile      Shared binned-profile subplot helper
- plot_ref_diff_vs_cld_dist    R0: Hexbin scatter of obs−ref vs cld_dist
- plot_ref_coverage_bias       R1: Selection bias — has-ref vs no-ref
- plot_ref_std_profiles        R2: Reference σ vs cld_dist
- plot_ref_corrected_profiles  R3: Binned mean ± SEM of obs−ref
- plot_ref_zscore_profiles     R4: Binned mean ± SEM of (obs−ref)/σ_ref
- plot_ref_signal_hierarchy    R5: r(cld_dist, obs−ref) bar chart
- plot_ref_alb_decoupled_exp   R6: OLS-remove Δalb from Δexp residuals
- plot_obs_vs_ref_scatter      R7: Hexbin obs vs ref_mean with 1:1 line
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from ca_utils import _save, rolling_median_iqr, bin_by_cld_dist

logger = logging.getLogger(__name__)


# Registry: (obs_col, ref_mean_col, ref_std_col, diff_col, band_label, term_label, band_color)
_REF_PAIRS: list[tuple] = [
    ('o2a_k1',             'ref_o2a_k1_mean',       'ref_o2a_k1_std',       'dk1_o2a',   'O\u2082A',  'k\u2081',   'C0'),
    ('o2a_k2',             'ref_o2a_k2_mean',       'ref_o2a_k2_std',       'dk2_o2a',   'O\u2082A',  'k\u2082',   'C0'),
    ('o2a_k3',             'ref_o2a_k3_mean',       'ref_o2a_k3_std',       'dk3_o2a',   'O\u2082A',  'k\u2083',   'C0'),
    ('wco2_k1',            'ref_wco2_k1_mean',      'ref_wco2_k1_std',      'dk1_wco2',  'WCO\u2082', 'k\u2081',   'C1'),
    ('wco2_k2',            'ref_wco2_k2_mean',      'ref_wco2_k2_std',      'dk2_wco2',  'WCO\u2082', 'k\u2082',   'C1'),
    ('wco2_k3',            'ref_wco2_k3_mean',      'ref_wco2_k3_std',      'dk3_wco2',  'WCO\u2082', 'k\u2083',   'C1'),
    ('sco2_k1',            'ref_sco2_k1_mean',      'ref_sco2_k1_std',      'dk1_sco2',  'SCO\u2082', 'k\u2081',   'C2'),
    ('sco2_k2',            'ref_sco2_k2_mean',      'ref_sco2_k2_std',      'dk2_sco2',  'SCO\u2082', 'k\u2082',   'C2'),
    ('sco2_k3',            'ref_sco2_k3_mean',      'ref_sco2_k3_std',      'dk3_sco2',  'SCO\u2082', 'k\u2083',   'C2'),
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
    ('o2a_k3',             'r25_o2a_k3_mean',       'r25_o2a_k3_std',       'dr25k3_o2a',   'O\u2082A',  'k\u2083',   'C0'),
    ('wco2_k1',            'r25_wco2_k1_mean',      'r25_wco2_k1_std',      'dr25k1_wco2',  'WCO\u2082', 'k\u2081',   'C1'),
    ('wco2_k2',            'r25_wco2_k2_mean',      'r25_wco2_k2_std',      'dr25k2_wco2',  'WCO\u2082', 'k\u2082',   'C1'),
    ('wco2_k3',            'r25_wco2_k3_mean',      'r25_wco2_k3_std',      'dr25k3_wco2',  'WCO\u2082', 'k\u2083',   'C1'),
    ('sco2_k1',            'r25_sco2_k1_mean',      'r25_sco2_k1_std',      'dr25k1_sco2',  'SCO\u2082', 'k\u2081',   'C2'),
    ('sco2_k2',            'r25_sco2_k2_mean',      'r25_sco2_k2_std',      'dr25k2_sco2',  'SCO\u2082', 'k\u2082',   'C2'),
    ('sco2_k3',            'r25_sco2_k3_mean',      'r25_sco2_k3_std',      'dr25k3_sco2',  'SCO\u2082', 'k\u2083',   'C2'),
    ('alb_o2a',            'r25_alb_o2a_mean',      'r25_alb_o2a_std',      'dr25alb_o2a',  'O\u2082A',  'albedo',    'C0'),
    ('alb_wco2',           'r25_alb_wco2_mean',     'r25_alb_wco2_std',     'dr25alb_wco2', 'WCO\u2082', 'albedo',    'C1'),
    ('alb_sco2',           'r25_alb_sco2_mean',     'r25_alb_sco2_std',     'dr25alb_sco2', 'SCO\u2082', 'albedo',    'C2'),
    ('exp_o2a_intercept',  'r25_exp_int_o2a_mean',  'r25_exp_int_o2a_std',  'dr25exp_o2a',  'O\u2082A',  'exp_int',   'C0'),
    ('exp_wco2_intercept', 'r25_exp_int_wco2_mean', 'r25_exp_int_wco2_std', 'dr25exp_wco2', 'WCO\u2082', 'exp_int',   'C1'),
    ('exp_sco2_intercept', 'r25_exp_int_sco2_mean', 'r25_exp_int_sco2_std', 'dr25exp_sco2', 'SCO\u2082', 'exp_int',   'C2'),
]


def add_ref_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add obs-ref difference and z-score columns for every entry in _REF_PAIRS.

    Standard diff columns:
      d{term}_{band}  = obs - ref_mean          e.g. dk1_o2a
      z{term}_{band}  = (obs - ref_mean)/ref_std e.g. zk1_o2a

    Derived composite diff columns (computed from standard diffs):
      d_exp_minus_alb_{band}  = dexp_{band} - dalb_{band}
      d_exp_over_alb_{band}   = (obs_exp / obs_alb) - (ref_exp_mean / ref_alb_mean)
      d_k2_over_k1_{band}     = (obs_k2 / obs_k1) - (ref_k2_mean / ref_k1_mean)

    Rows without a ref value remain NaN in the new columns.
    """
    new_cols = {}
    for obs, ref_m, ref_s, dcol, _, _, _ in _REF_PAIRS:
        if obs in df.columns and ref_m in df.columns:
            new_cols[dcol] = df[obs] - df[ref_m]
            zcol = 'z' + dcol[1:]   # 'dk1_o2a' → 'zk1_o2a'
            if ref_s in df.columns:
                new_cols[zcol] = new_cols[dcol] / df[ref_s].replace(0, np.nan)

    # ── derived composite deltas ──────────────────────────────────────────────
    _band_to_exp = {p[4]: p for p in _REF_PAIRS if p[5] == 'exp_int'}
    _band_to_alb = {p[4]: p for p in _REF_PAIRS if p[5] == 'albedo'}
    _band_to_k1  = {p[4]: p for p in _REF_PAIRS if p[5] == 'k\u2081'}
    _band_to_k2  = {p[4]: p for p in _REF_PAIRS if p[5] == 'k\u2082'}

    for band in _band_to_exp:
        exp_p = _band_to_exp[band]
        obs_exp_col, ref_exp_col = exp_p[0], exp_p[1]

        # Δ(exp − alb)
        if band in _band_to_alb:
            alb_p = _band_to_alb[band]
            obs_alb_col, ref_alb_col = alb_p[0], alb_p[1]
            dc_minus = f'd_exp_minus_alb_{band}'
            if obs_exp_col in df.columns and obs_alb_col in df.columns \
                    and ref_exp_col in df.columns and ref_alb_col in df.columns:
                new_cols[dc_minus] = (
                    (df[obs_exp_col] - df[obs_alb_col])
                    - (df[ref_exp_col] - df[ref_alb_col])
                )

            # Δ(exp / alb)
            dc_ratio = f'd_exp_over_alb_{band}'
            if obs_exp_col in df.columns and obs_alb_col in df.columns \
                    and ref_exp_col in df.columns and ref_alb_col in df.columns:
                new_cols[dc_ratio] = (
                    df[obs_exp_col] / df[obs_alb_col].replace(0, np.nan)
                    - df[ref_exp_col] / df[ref_alb_col].replace(0, np.nan)
                )

        # Δ(k2 / k1)
        if band in _band_to_k1 and band in _band_to_k2:
            k1_p = _band_to_k1[band]
            k2_p = _band_to_k2[band]
            obs_k1, ref_k1 = k1_p[0], k1_p[1]
            obs_k2, ref_k2 = k2_p[0], k2_p[1]
            dc_k2k1 = f'd_k2_over_k1_{band}'
            if all(c in df.columns for c in [obs_k1, obs_k2, ref_k1, ref_k2]):
                new_cols[dc_k2k1] = (
                    df[obs_k2] / df[obs_k1].replace(0, np.nan)
                    - df[ref_k2] / df[ref_k1].replace(0, np.nan)
                )

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
        ('k\u2083',   [p for p in pairs if p[5] == 'k\u2083'],  f'{tag}_diff_scatter_k3.png'),
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

    fig.suptitle(f'[{tag}] Selection bias check: soundings with vs without clear-sky reference\n'
                 '(ideally the two groups match within each cld_dist bin)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_coverage_bias.png')


# ── R2: Ref std profiles (scene heterogeneity proxy) ─────────────────────────

def plot_ref_std_profiles(df: pd.DataFrame, bins, labels, outdir: str,
                          pairs=None, tag: str = 'ref') -> None:
    """Plot ref_std (within-reference-pool variability) vs cloud-distance bin.

    A decrease in ref_std near clouds can indicate:
      (a) fewer reference pixels → downward-biased std estimate, or
      (b) genuinely more homogeneous clear-sky corridors adjacent to clouds.
    Ocean and land are shown in separate columns.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    # show k1, albedo, exp_int for O₂A and SCO₂ only
    _show_bands = {'O\u2082A', 'SCO\u2082'}
    _show_terms = {'k\u2081', 'albedo', 'exp_int'}
    std_vars = [
        (ref_s, f'{bl} {tl} {tag} \u03c3', col)
        for _, _, ref_s, _, bl, tl, col in pairs
        if bl in _show_bands and tl in _show_terms
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

    fig.suptitle(f'[{tag}] Clear-sky reference \u03c3 vs cloud distance\n'
                 '(decreasing near clouds may reflect sampling, not homogeneity)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_std_profiles.png')


# ── R3: Ref-corrected anomaly profiles ────────────────────────────────────────

def plot_ref_corrected_profiles(df: pd.DataFrame, bins, labels, outdir: str,
                                pairs=None, tag: str = 'ref') -> None:
    """Binned mean ± SEM/std of (obs − ref_mean) for k1, k2, albedo, exp_intercept.

    Four figures (one per variable type), each with 3 rows (bands O2A/WCO2/SCO2)
    × 2 columns (ocean / land).  The y = 0 line marks where obs matches its
    clear-sky baseline; deviations reveal cloud-adjacency effects that survive
    local scene conditioning.
    """
    if pairs is None:
        pairs = _REF_PAIRS

    _BANDS = ['O\u2082A', 'WCO\u2082', 'SCO\u2082']
    _BAND_COLOR = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}

    # standard pairs groups (7-tuples) + derived groups (4-tuples padded to 7)
    term_groups = [
        ('k\u2081',   [p for p in pairs if p[5] == 'k\u2081'],   f'{tag}_corrected_k1_profiles.png'),
        ('k\u2082',   [p for p in pairs if p[5] == 'k\u2082'],   f'{tag}_corrected_k2_profiles.png'),
        ('k\u2083',   [p for p in pairs if p[5] == 'k\u2083'],   f'{tag}_corrected_k3_profiles.png'),
        ('albedo', [p for p in pairs if p[5] == 'albedo'],       f'{tag}_corrected_alb_profiles.png'),
        ('exp_int',[p for p in pairs if p[5] == 'exp_int'],      f'{tag}_corrected_exp_profiles.png'),
        # derived groups — 4-tuples padded: (None, None, None, dcol, band, term, color)
        ('exp\u2212alb',
         [(None, None, None, f'd_exp_minus_alb_{b}', b, 'exp\u2212alb', _BAND_COLOR[b])
          for b in _BANDS],
         f'{tag}_corrected_exp_minus_alb_profiles.png'),
        ('exp/alb',
         [(None, None, None, f'd_exp_over_alb_{b}', b, 'exp/alb', _BAND_COLOR[b])
          for b in _BANDS],
         f'{tag}_corrected_exp_over_alb_profiles.png'),
        ('k\u2082/k\u2081',
         [(None, None, None, f'd_k2_over_k1_{b}', b, 'k\u2082/k\u2081', _BAND_COLOR[b])
          for b in _BANDS],
         f'{tag}_corrected_k2_over_k1_profiles.png'),
    ]
    subsets = [('Ocean', df[df['sfc_type'] == 0]),
               ('Land',  df[df['sfc_type'] == 1])]

    for term_lbl, grp_pairs, fname in term_groups:
        avail = [p for p in grp_pairs if p[3] in df.columns]
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

        fig.suptitle(f'[{tag}] Ref-corrected anomaly  [{term_lbl}]:  obs \u2212 clear-sky ref\n'
                     'y = 0 \u2192 obs matches clear-sky baseline',
                     fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R4: Ref z-score profiles ──────────────────────────────────────────────────

def plot_ref_zscore_profiles(df: pd.DataFrame, bins, labels, outdir: str,
                             pairs=None, tag: str = 'ref') -> None:
    """Binned mean ± SEM of z = (obs − ref_mean) / ref_std vs cloud distance.

    Normalising by ref_std puts all variables on the same scale (units of
    clear-sky natural variability).  A z-score of ±1 means the obs deviates
    by one standard deviation of the local reference distribution.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    term_groups = [
        ('k\u2081',   [p for p in pairs if p[5] == 'k\u2081'],   f'{tag}_zscore_k1_profiles.png'),
        ('k\u2082',   [p for p in pairs if p[5] == 'k\u2082'],   f'{tag}_zscore_k2_profiles.png'),
        ('k\u2083',   [p for p in pairs if p[5] == 'k\u2083'],   f'{tag}_zscore_k3_profiles.png'),
        ('albedo', [p for p in pairs if p[5] == 'albedo'],       f'{tag}_zscore_alb_profiles.png'),
        ('exp_int',[p for p in pairs if p[5] == 'exp_int'],      f'{tag}_zscore_exp_profiles.png'),
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

        fig.suptitle(f'[{tag}] Ref z-score  [{term_lbl}]:  (obs \u2212 ref) / \u03c3_ref\n'
                     'Units of clear-sky natural variability',
                     fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R5: Signal hierarchy with ref normalization ───────────────────────────────

def plot_ref_signal_hierarchy(df: pd.DataFrame, outdir: str,
                              pairs=None, tag: str = 'ref') -> None:
    """Bar chart of r(cld_dist, obs − ref) for all ref-corrected variables.

    Companion to plot_signal_hierarchy but using diff columns (obs − ref_mean)
    instead of raw obs.  Variables that retain large |r| after subtracting the
    clear-sky reference carry genuine cloud-adjacency signal not explained by
    local scene co-variation.  Ocean and land are shown side-by-side.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    _term_order  = ['k\u2081', 'k\u2082', 'k\u2083', 'albedo', 'exp_int']
    _hatch_map   = {'k\u2081': '', 'k\u2082': '///', 'k\u2083': 'xxx', 'albedo': '...', 'exp_int': '|||'}
    _disp_term   = {'k\u2081': 'k\u2081', 'k\u2082': 'k\u2082', 'k\u2083': 'k\u2083',
                    'albedo': 'alb', 'exp_int': 'exp_int'}
    feat_groups = [
        (dcol, bl, _disp_term[tl], _hatch_map[tl])
        for tl in _term_order
        for _, _, _, dcol, bl, ptl, _ in pairs
        if ptl == tl and dcol in df.columns
    ]
    avail = feat_groups
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
                       for t, h in [('k\u2081', ''), ('k\u2082', '///'), ('k\u2083', 'xxx'),
                                    ('alb', '...'), ('exp_int', '|||')]]
    axes[0].legend(handles=legend_handles, fontsize=7, ncol=2, loc='lower left',
                   title='Band / Term', title_fontsize=7)
    fig.suptitle(f'[{tag}] Ref-corrected signal hierarchy: r(cld_dist, obs \u2212 {tag})\n'
                 'Variables retaining large |r| carry genuine cloud-adjacency signal',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_signal_hierarchy.png')


# ── R6: Albedo-decoupled exp_intercept in ref-corrected space ─────────────────

def plot_ref_alb_decoupled_exp(df: pd.DataFrame, bins, labels, outdir: str,
                               pairs=None, tag: str = 'ref') -> None:
    """OLS-remove dalb from dexp, then plot residual vs cloud distance.

    In ref-corrected space: dexp ~ const + dalb (per band, per surface type).
    The residual isolates the non-albedo component of the cloud-edge signal —
    i.e. photon path-length or scattering effects independent of surface changes.
    Compares r_raw(dexp) vs r_resid to quantify how much cloud signal survives
    after accounting for albedo co-variation with cloud proximity.
    """
    if pairs is None:
        pairs = _REF_PAIRS
    _exp_by_band = {bl: (dcol, col) for _, _, _, dcol, bl, tl, col in pairs if tl == 'exp_int'}
    _alb_by_band = {bl: dcol        for _, _, _, dcol, bl, tl, _   in pairs if tl == 'albedo'}
    band_defs = [
        (_exp_by_band[bl][0], _alb_by_band[bl], bl, _exp_by_band[bl][1])
        for bl in _exp_by_band if bl in _alb_by_band
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

    fig.suptitle(f'[{tag}] \u0394exp_int residual after removing \u0394alb (OLS) — ref-corrected space\n'
                 'Surviving signal is independent of surface reflectance co-variation',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_alb_decoupled_exp_residuals.png')


# ── R7: Obs vs ref scatter at matched geometry ────────────────────────────────

def plot_obs_vs_ref_scatter(df: pd.DataFrame, outdir: str,
                            pairs=None, tag: str = 'ref') -> None:
    """Hexbin scatter of obs vs ref_mean colored by density, with 1:1 line.

    Points above the 1:1 line: obs exceeds clear-sky reference.
    The rolling median (colored by mean cld_dist in each hex) reveals whether
    cloud-adjacent soundings systematically depart from the clear-sky baseline.
    Three variable types (k1, albedo, exp_int) × two surface types (ocean/land).
    """
    if pairs is None:
        pairs = _REF_PAIRS
    _show_bands = {'O\u2082A', 'SCO\u2082'}
    _show_terms = {'k\u2081', 'albedo', 'exp_int'}
    var_defs = [
        (obs, ref_m, f'{bl} {tl}', col)
        for obs, ref_m, _, _, bl, tl, col in pairs
        if bl in _show_bands and tl in _show_terms
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

        fig.suptitle(f'[{tag}] Obs vs clear-sky reference — {sfc_name}\n'
                     'Points above 1:1 line: obs exceeds clear-sky baseline',
                     fontsize=12)
        fig.tight_layout()
        fname = f'obs_vs_{tag}_scatter_{sfc_name.lower()}.png'
        _save(fig, outdir, fname)


# ── R8: Multi-variable delta comparison ───────────────────────────────────────

def plot_ref_delta_multivar(df: pd.DataFrame, bins, labels,
                             outdir: str, pairs=None, tag: str = 'ref') -> None:
    """Overlay binned-mean profiles of all delta columns on one figure per surface.

    One figure per surface type (ocean / land), 4 panels: k1, k2, albedo, exp_int.
    Each panel shows all 3 bands (O2A, WCO2, SCO2) as separate lines.
    Allows direct magnitude / direction comparison across variables.
    Output: {tag}_delta_multivar_{ocean,land}.png
    """
    if pairs is None:
        pairs = _REF_PAIRS

    # Group pairs by term_label → {term: [(dcol, band_label, color), ...]}
    from collections import defaultdict
    term_map: dict = defaultdict(list)
    for _, _, _, dcol, band_lbl, term_lbl, color in pairs:
        if dcol in df.columns:
            term_map[term_lbl].append((dcol, band_lbl, color))

    if not term_map:
        logger.warning(f"[{tag}] R8: no delta columns found — skipping")
        return

    sfc_subsets = [('Ocean', df[df['sfc_type'] == 0]),
                   ('Land',  df[df['sfc_type'] == 1])] if 'sfc_type' in df.columns \
                  else [('All', df)]

    x = np.arange(len(labels))
    term_order = ['k\u2081', 'k\u2082', 'k\u2083', 'albedo', 'exp_int']
    terms = [t for t in term_order if t in term_map] + \
            [t for t in term_map if t not in term_order]

    # derived delta groups: (label, [(dcol, band_lbl, color), ...])
    _BANDS = ['O\u2082A', 'WCO\u2082', 'SCO\u2082']
    _BAND_COLOR = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}
    _derived_terms = []
    for derived_lbl, col_prefix in [
        ('exp\u2212alb',    'd_exp_minus_alb'),
        ('exp/alb',         'd_exp_over_alb'),
        ('k\u2082/k\u2081', 'd_k2_over_k1'),
    ]:
        entries = [(f'{col_prefix}_{b}', b, _BAND_COLOR[b])
                   for b in _BANDS if f'{col_prefix}_{b}' in df.columns]
        if entries:
            _derived_terms.append((derived_lbl, entries))

    all_terms_count = len(terms) + len(_derived_terms)

    for sfc_name, sdf in sfc_subsets:
        if len(sdf) < 50:
            continue
        ncols = min(all_terms_count, 4)
        nrows = int(np.ceil(all_terms_count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 sharey=False)
        axes = np.array(axes).flatten()

        _bin = pd.cut(sdf['cld_dist_km'], bins=bins, labels=labels, right=False)
        ax_idx = 0

        for term in terms:
            ax = axes[ax_idx]; ax_idx += 1
            for dcol, band_lbl, color in term_map[term]:
                sub = sdf[[dcol, 'cld_dist_km']].dropna()
                sub_bin = _bin[sub.index]
                means = sub.groupby(sub_bin, observed=True)[dcol].mean().reindex(labels)
                sems  = (sub.groupby(sub_bin, observed=True)[dcol].std() /
                         np.sqrt(sub.groupby(sub_bin, observed=True)[dcol].count())
                         ).reindex(labels)
                ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                            capsize=3, lw=1.5, label=band_lbl, color=color)
            ax.axhline(0, color='gray', lw=1, linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'\u0394{term} (obs \u2212 ref)', fontsize=9)
            ax.set_title(f'\u0394{term}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

        for derived_lbl, entries in _derived_terms:
            ax = axes[ax_idx]; ax_idx += 1
            for dcol, band_lbl, color in entries:
                sub = sdf[[dcol, 'cld_dist_km']].dropna()
                sub_bin = _bin[sub.index]
                means = sub.groupby(sub_bin, observed=True)[dcol].mean().reindex(labels)
                sems  = (sub.groupby(sub_bin, observed=True)[dcol].std() /
                         np.sqrt(sub.groupby(sub_bin, observed=True)[dcol].count())
                         ).reindex(labels)
                ax.errorbar(x, means.values, yerr=sems.values, fmt='o-',
                            capsize=3, lw=1.5, label=band_lbl, color=color)
            ax.axhline(0, color='gray', lw=1, linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_xlabel('Cloud distance (km)', fontsize=9)
            ax.set_ylabel(f'\u0394({derived_lbl}) (obs \u2212 ref)', fontsize=9)
            ax.set_title(f'\u0394({derived_lbl})', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)

        for ax in axes[ax_idx:]:
            ax.set_visible(False)

        fig.suptitle(f'[{tag}] All delta variables vs cloud distance — {sfc_name}\n'
                     'mean \u00b1 SEM per cld_dist bin', fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, f'{tag}_delta_multivar_{sfc_name.lower()}.png')


# ── R9: Cross-band delta coherence ────────────────────────────────────────────

def plot_ref_cross_band_delta(df: pd.DataFrame, outdir: str,
                               pairs=None, tag: str = 'ref',
                               max_dist: float = 50) -> None:
    """Scatter matrix of delta columns across bands, colored by cld_dist_km.

    One figure per term type (k1, k2, albedo, exp_int).
    3×3 grid: row/col = O2A / WCO2 / SCO2.  Diagonal = histogram.
    Output: {tag}_cross_band_delta_{k1,k2,alb,exp}.png
    """
    if pairs is None:
        pairs = _REF_PAIRS

    from collections import defaultdict
    term_map: dict = defaultdict(list)
    for _, _, _, dcol, band_lbl, term_lbl, _ in pairs:
        if dcol in df.columns:
            term_map[term_lbl].append((dcol, band_lbl))

    sub = df[df['cld_dist_km'] <= max_dist].copy()
    if len(sub) < 50:
        return

    norm = mcolors.Normalize(vmin=0, vmax=max_dist)
    cmap = plt.cm.plasma_r

    term_order = [('k\u2081', f'{tag}_cross_band_delta_k1.png'),
                  ('k\u2082', f'{tag}_cross_band_delta_k2.png'),
                  ('k\u2083', f'{tag}_cross_band_delta_k3.png'),
                  ('albedo', f'{tag}_cross_band_delta_alb.png'),
                  ('exp_int', f'{tag}_cross_band_delta_exp.png')]

    for term_lbl, fname in term_order:
        entries = term_map.get(term_lbl, [])
        if len(entries) < 2:
            continue
        n = len(entries)
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
        if n == 1:
            axes = np.array([[axes]])

        for ri, (dcol_r, band_r) in enumerate(entries):
            for ci, (dcol_c, band_c) in enumerate(entries):
                ax = axes[ri, ci]
                xv = sub[dcol_c].values.astype(float)
                yv = sub[dcol_r].values.astype(float)
                cv = sub['cld_dist_km'].values
                m  = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(cv)
                if ri == ci:
                    ax.hist(yv[m], bins=40, color='steelblue', alpha=0.7)
                    ax.set_xlabel(f'\u0394{band_r}', fontsize=8)
                else:
                    sc = ax.scatter(xv[m], yv[m], c=cv[m], cmap=cmap, norm=norm,
                                    s=3, alpha=0.3, rasterized=True)
                    if ri == 0 and ci == n - 1:
                        plt.colorbar(sc, ax=ax, label='cld_dist (km)')
                    if m.sum() > 10:
                        r_val = stats.pearsonr(xv[m], yv[m])[0]
                        ax.set_title(f'r={r_val:.3f}', fontsize=8)
                    ax.axhline(0, color='gray', lw=0.5)
                    ax.axvline(0, color='gray', lw=0.5)
                if ri == n - 1:
                    ax.set_xlabel(f'\u0394{band_c}', fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f'\u0394{band_r}', fontsize=8)

        fig.suptitle(f'[{tag}] Cross-band delta coherence — \u0394{term_lbl}\n'
                     f'colored by cld_dist_km (0\u2013{max_dist} km)', fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R10: Delta decay length scale ─────────────────────────────────────────────

def plot_ref_delta_decay(df: pd.DataFrame, bins, labels,
                          outdir: str, pairs=None, tag: str = 'ref') -> None:
    """Fit A·exp(−d/τ)+C to binned delta means; report τ and A per variable.

    Produces a grouped bar chart of τ (decay length in km) per delta column,
    separated by surface type.  Also saves a CSV table.
    Output: {tag}_delta_decay_lengths.png + {tag}_delta_decay_table.csv
    """
    from scipy.optimize import curve_fit

    if pairs is None:
        pairs = _REF_PAIRS

    dcols = [(dcol, band_lbl, term_lbl, color)
             for _, _, _, dcol, band_lbl, term_lbl, color in pairs
             if dcol in df.columns]
    if not dcols:
        logger.warning(f"[{tag}] R10: no delta columns — skipping")
        return

    def _exp_decay(d, A, tau, C):
        return A * np.exp(-d / tau) + C

    sfc_subsets = [('Ocean', df[df['sfc_type'] == 0]),
                   ('Land',  df[df['sfc_type'] == 1])] if 'sfc_type' in df.columns \
                  else [('All', df)]

    # bin centres (km) — use midpoint of each edge pair
    _edges = np.array(bins)
    bin_centers = 0.5 * (_edges[:-1] + _edges[1:])

    records = []
    fig, axes = plt.subplots(1, len(sfc_subsets),
                              figsize=(7 * len(sfc_subsets), 5), squeeze=False)

    for col_idx, (sfc_name, sdf) in enumerate(sfc_subsets):
        ax = axes[0, col_idx]
        _bin = pd.cut(sdf['cld_dist_km'], bins=bins, labels=labels, right=False)
        taus, amps, col_labels, bar_colors = [], [], [], []

        for dcol, band_lbl, term_lbl, color in dcols:
            sub = sdf[[dcol, 'cld_dist_km']].dropna()
            if len(sub) < 20:
                continue
            sub_bin = _bin[sub.index]
            means = sub.groupby(sub_bin, observed=True)[dcol].mean().reindex(labels)
            y = means.values.astype(float)
            valid = np.isfinite(y)
            if valid.sum() < 4:
                continue
            xd, yd = bin_centers[valid], y[valid]
            try:
                popt, _ = curve_fit(_exp_decay, xd, yd,
                                    p0=[yd[0] - yd[-1], 10.0, yd[-1]],
                                    maxfev=5000, bounds=([-np.inf, 0.1, -np.inf],
                                                         [np.inf, 200, np.inf]))
                tau_fit = popt[1]
                amp_fit = popt[0]
            except RuntimeError:
                tau_fit, amp_fit = np.nan, np.nan

            taus.append(tau_fit)
            amps.append(amp_fit)
            col_labels.append(f'{band_lbl}\n\u0394{term_lbl}')
            bar_colors.append(color)
            records.append({'surface': sfc_name, 'delta_col': dcol,
                            'band': band_lbl, 'term': term_lbl,
                            'tau_km': tau_fit, 'amplitude': amp_fit})

        xi = np.arange(len(taus))
        bars = ax.bar(xi, taus, color=bar_colors, alpha=0.75, edgecolor='k', lw=0.5)
        ax.set_xticks(xi)
        ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Decay length τ (km)', fontsize=9)
        ax.set_title(f'[{tag}] {sfc_name}', fontsize=10)
        ax.axhline(0, color='gray', lw=1)
        ax.grid(axis='y', alpha=0.3)
        for bar, tau in zip(bars, taus):
            if np.isfinite(tau):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{tau:.1f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'[{tag}] Cloud-proximity decay length τ per delta variable\n'
                 'A·exp(−d/τ)+C fit to binned means', fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_delta_decay_lengths.png')

    if records:
        import os
        csv_path = os.path.join(outdir, f'{tag}_delta_decay_table.csv')
        pd.DataFrame(records).to_csv(csv_path, index=False)
        logger.info(f"  saved → {csv_path}")


# ── R11: Delta vs XCO2 BC anomaly ─────────────────────────────────────────────

def plot_ref_delta_vs_xco2(df: pd.DataFrame, outdir: str,
                            pairs=None, tag: str = 'ref',
                            max_dist: float = 50, n_roll: int = 80) -> None:
    """Scatter of each delta vs xco2_bc_anomaly, colored by cld_dist_km.

    One figure per term type (k1, k2, albedo, exp_int).  Rolling median overlay.
    Tests whether the reference-corrected cloud signal predicts XCO2 bias.
    Output: {tag}_delta_vs_xco2_{k1,k2,alb,exp}.png
    """
    if pairs is None:
        pairs = _REF_PAIRS
    if 'xco2_bc_anomaly' not in df.columns:
        logger.warning(f"[{tag}] R11: xco2_bc_anomaly missing — skipping")
        return

    from collections import defaultdict
    term_map: dict = defaultdict(list)
    for _, _, _, dcol, band_lbl, term_lbl, color in pairs:
        if dcol in df.columns:
            term_map[term_lbl].append((dcol, band_lbl, color))

    sub = df[df['cld_dist_km'] <= max_dist].dropna(subset=['xco2_bc_anomaly']).copy()
    if len(sub) < 50:
        return

    norm = mcolors.Normalize(vmin=0, vmax=max_dist)
    cmap = plt.cm.viridis_r
    yv_xco2 = sub['xco2_bc_anomaly'].values.astype(float)

    term_order = [('k\u2081', f'{tag}_delta_vs_xco2_k1.png'),
                  ('k\u2082', f'{tag}_delta_vs_xco2_k2.png'),
                  ('k\u2083', f'{tag}_delta_vs_xco2_k3.png'),
                  ('albedo', f'{tag}_delta_vs_xco2_alb.png'),
                  ('exp_int', f'{tag}_delta_vs_xco2_exp.png')]

    for term_lbl, fname in term_order:
        entries = term_map.get(term_lbl, [])
        if not entries:
            continue
        ncols = len(entries)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
        axes = axes[0]

        for ax, (dcol, band_lbl, color) in zip(axes, entries):
            xv = sub[dcol].values.astype(float)
            cv = sub['cld_dist_km'].values
            m  = np.isfinite(xv) & np.isfinite(yv_xco2)
            sc = ax.scatter(xv[m], yv_xco2[m], c=cv[m], cmap=cmap, norm=norm,
                            s=4, alpha=0.3, rasterized=True)
            plt.colorbar(sc, ax=ax, label='cld_dist (km)')
            # rolling median
            order = np.argsort(xv[m])
            xs, med, q25, q75 = rolling_median_iqr(xv[m][order], yv_xco2[m][order],
                                                     n_pts=n_roll)
            ax.plot(xs, med, color='tomato', lw=2, label='rolling median')
            ax.fill_between(xs, q25, q75, color='tomato', alpha=0.2)
            ax.axhline(0, color='gray', lw=1, linestyle='--')
            ax.axvline(0, color='gray', lw=1, linestyle='--')
            if m.sum() > 10:
                r_val = stats.pearsonr(xv[m], yv_xco2[m])[0]
                ax.set_title(f'{band_lbl} \u0394{term_lbl}   r={r_val:.3f}', fontsize=9)
            ax.set_xlabel(f'\u0394{term_lbl} ({band_lbl})', fontsize=9)
            ax.set_ylabel('XCO\u2082 BC anomaly (ppm)', fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle(f'[{tag}] \u0394{term_lbl} vs XCO\u2082 BC anomaly\n'
                     f'soundings \u2264{max_dist} km from cloud, colored by cld_dist',
                     fontsize=11)
        fig.tight_layout()
        _save(fig, outdir, fname)


# ── R12: Partial correlation of delta variables with XCO2 anomaly ─────────────

def plot_ref_delta_partial_xco2(df: pd.DataFrame, outdir: str,
                                 pairs=None, tag: str = 'ref') -> None:
    """Pearson partial-r of each delta vs xco2_bc_anomaly after OLS-removing confounders.

    Confounders removed: albedo columns + airmass + cos(SZA).
    Produces a horizontal bar chart; ocean and land shown side-by-side.
    Analogous to plot_residual_signal_hierarchy but for ref-corrected deltas.
    Output: {tag}_delta_partial_xco2.png
    """
    if pairs is None:
        pairs = _REF_PAIRS
    if 'xco2_bc_anomaly' not in df.columns:
        logger.warning(f"[{tag}] R12: xco2_bc_anomaly missing — skipping")
        return

    dcols = [(dcol, band_lbl, term_lbl, color)
             for _, _, _, dcol, band_lbl, term_lbl, color in pairs
             if dcol in df.columns]

    # append derived composite deltas
    _BANDS = ['O\u2082A', 'WCO\u2082', 'SCO\u2082']
    _BAND_COLOR = {'O\u2082A': 'C0', 'WCO\u2082': 'C1', 'SCO\u2082': 'C2'}
    for derived_lbl, col_prefix in [
        ('exp\u2212alb',    'd_exp_minus_alb'),
        ('exp/alb',         'd_exp_over_alb'),
        ('k\u2082/k\u2081', 'd_k2_over_k1'),
    ]:
        for b in _BANDS:
            dc = f'{col_prefix}_{b}'
            if dc in df.columns:
                dcols.append((dc, b, derived_lbl, _BAND_COLOR[b]))

    if not dcols:
        return

    # Build confounder list that actually exist
    _alb_cols = [c for c in df.columns if c.startswith('alb_') and 'ref' not in c and 'r25' not in c]
    _conf_candidates = _alb_cols + ['airmass']
    if 'sza' in df.columns:
        _conf_candidates.append('sza')
    elif 'mu_sza' in df.columns:
        _conf_candidates.append('mu_sza')

    sfc_subsets = [('Ocean', df[df['sfc_type'] == 0]),
                   ('Land',  df[df['sfc_type'] == 1])] if 'sfc_type' in df.columns \
                  else [('All', df)]

    def _partial_r(sdf, dcol, confounders):
        keep = [dcol, 'xco2_bc_anomaly'] + confounders
        sub = sdf[keep].dropna()
        if len(sub) < 30:
            return np.nan
        X_conf = sub[confounders].values.astype(float)
        X_conf = np.column_stack([X_conf, np.ones(len(X_conf))])
        def _resid(y):
            try:
                beta, *_ = np.linalg.lstsq(X_conf, y, rcond=None)
                return y - X_conf @ beta
            except Exception:
                return y
        r_dx = _resid(sub[dcol].values.astype(float))
        r_xco2 = _resid(sub['xco2_bc_anomaly'].values.astype(float))
        m = np.isfinite(r_dx) & np.isfinite(r_xco2)
        if m.sum() < 10:
            return np.nan
        return stats.pearsonr(r_dx[m], r_xco2[m])[0]

    ncols = len(sfc_subsets)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, max(4, len(dcols) * 0.5 + 1)),
                              squeeze=False)

    for col_idx, (sfc_name, sdf) in enumerate(sfc_subsets):
        ax = axes[0, col_idx]
        conf_avail = [c for c in _conf_candidates if c in sdf.columns]
        r_vals, bar_labels, bar_colors = [], [], []

        for dcol, band_lbl, term_lbl, color in dcols:
            r = _partial_r(sdf, dcol, conf_avail)
            r_vals.append(r)
            bar_labels.append(f'{band_lbl} \u0394{term_lbl}')
            bar_colors.append(color)

        yi = np.arange(len(r_vals))
        bars = ax.barh(yi, r_vals, color=bar_colors, alpha=0.75,
                       edgecolor='k', lw=0.5)
        ax.set_yticks(yi)
        ax.set_yticklabels(bar_labels, fontsize=8)
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel('Partial Pearson r', fontsize=9)
        ax.set_title(f'[{tag}] {sfc_name}', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        for bar, r in zip(bars, r_vals):
            if np.isfinite(r):
                xpos = r + (0.005 if r >= 0 else -0.005)
                ha = 'left' if r >= 0 else 'right'
                ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                        f'{r:.3f}', va='center', ha=ha, fontsize=7)

    fig.suptitle(f'[{tag}] Partial r(\u0394var, XCO\u2082 BC anomaly)\n'
                 f'after OLS-removing {", ".join(conf_avail[:3])}…', fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, f'{tag}_delta_partial_xco2.png')


# ── R14: Ref-corrected profiles stratified by footprint area ──────────────────

def plot_ref_corrected_profiles_by_fp_area(
    df: pd.DataFrame, bins, labels, outdir: str,
    pairs=None, tag: str = 'ref', n_fp_bins: int = 5,
) -> None:
    """Ref-corrected anomaly profiles stratified by fp_area_km2 quantile bins.

    Mirrors plot_ref_corrected_profiles (R3): rows = bands, cols = ocean/land.
    Instead of a single mean line per subplot, draws one coloured line per
    fp_area_km2 quintile, showing whether the cloud-adjacency signal in
    (obs − ref) depends on footprint size.

    Six variable groups:
      k1, k2, albedo, exp_int, exp_int − albedo, exp_int / albedo

    The derived groups are computed inline:
      Δ(exp − alb)_{band}  = dexp_{band} − dalb_{band}
      Δ(exp / alb)_{band}  = (obs_exp / obs_alb) − (ref_exp_mean / ref_alb_mean)

    Skips silently if fp_area_km2 column is absent.

    Output (under {outdir}/fp_area/):
        {tag}_corrected_k1_profiles_by_fp_area.png
        {tag}_corrected_k2_profiles_by_fp_area.png
        {tag}_corrected_alb_profiles_by_fp_area.png
        {tag}_corrected_exp_profiles_by_fp_area.png
        {tag}_corrected_exp_minus_alb_profiles_by_fp_area.png
        {tag}_corrected_exp_over_alb_profiles_by_fp_area.png
    """
    import os

    if 'fp_area_km2' not in df.columns:
        logger.info(f"[{tag}] R14: fp_area_km2 not in data — skipping")
        return
    if pairs is None:
        pairs = _REF_PAIRS

    df = df.copy()

    # ── fp_area quantile bins ──────────────────────────────────────────────────
    fp_raw = df['fp_area_km2']
    lo, hi = fp_raw.quantile(0.01), fp_raw.quantile(0.99)
    fp_clipped = fp_raw.clip(lo, hi)
    try:
        _, fp_edges = pd.qcut(fp_clipped, q=n_fp_bins, retbins=True, duplicates='drop')
    except ValueError:
        logger.warning(f"[{tag}] R14: cannot form {n_fp_bins} fp_area bins — skipping")
        return

    actual_n = len(fp_edges) - 1
    fp_qlabels = [f'Q{i+1} ({fp_edges[i]:.2f}–{fp_edges[i+1]:.2f} km²)'
                  for i in range(actual_n)]
    df['_fp_qbin'] = pd.qcut(fp_clipped, q=n_fp_bins, labels=fp_qlabels, duplicates='drop')
    fp_cmap = plt.cm.get_cmap('plasma', actual_n)

    # ── derived diff columns (exp−alb, exp/alb) ──────────────────────────────
    # Must be computed on df BEFORE slicing into sfc_subsets so that the
    # ocean/land slices contain the new columns.
    band_to_exp = {p[4]: p for p in pairs if p[5] == 'exp_int'}
    band_to_alb = {p[4]: p for p in pairs if p[5] == 'albedo'}
    _BAND_ORDER = {'O\u2082A': 0, 'WCO\u2082': 1, 'SCO\u2082': 2}

    derived_minus, derived_ratio = [], []
    for band, exp_p in sorted(band_to_exp.items(), key=lambda kv: _BAND_ORDER.get(kv[0], 99)):
        if band not in band_to_alb:
            continue
        alb_p = band_to_alb[band]
        obs_exp_col, ref_exp_col = exp_p[0], exp_p[1]
        obs_alb_col, ref_alb_col = alb_p[0], alb_p[1]
        dcol_exp, dcol_alb = exp_p[3], alb_p[3]
        color = exp_p[6]

        # Δ(exp − alb) = dexp − dalb
        dc_minus = f'_d_exp_minus_alb_{band}'
        if dcol_exp in df.columns and dcol_alb in df.columns:
            df[dc_minus] = df[dcol_exp] - df[dcol_alb]
            derived_minus.append((dc_minus, band, 'exp\u2212alb', color))

        # Δ(exp / alb) = obs_exp/obs_alb − ref_exp/ref_alb
        dc_ratio = f'_d_exp_over_alb_{band}'
        req = [obs_exp_col, obs_alb_col, ref_exp_col, ref_alb_col]
        if all(c in df.columns for c in req):
            df[dc_ratio] = (
                df[obs_exp_col] / df[obs_alb_col].replace(0, np.nan)
                - df[ref_exp_col] / df[ref_alb_col].replace(0, np.nan)
            )
            derived_ratio.append((dc_ratio, band, 'exp/alb', color))

    # Slice after all new columns are on df so subsets inherit them
    sfc_subsets = [('Ocean', df[df['sfc_type'] == 0]),
                   ('Land',  df[df['sfc_type'] == 1])] if 'sfc_type' in df.columns \
                  else [('All', df)]

    # ── build variable groups (standard + derived) ────────────────────────────
    var_groups = [
        ('k\u2081',          [p for p in pairs if p[5] == 'k\u2081'],
         f'{tag}_corrected_k1_profiles_by_fp_area.png',
         f'[{tag}] Ref-corrected \u0394k\u2081 by footprint area quintile'),
        ('k\u2082',          [p for p in pairs if p[5] == 'k\u2082'],
         f'{tag}_corrected_k2_profiles_by_fp_area.png',
         f'[{tag}] Ref-corrected \u0394k\u2082 by footprint area quintile'),
        ('albedo',         [p for p in pairs if p[5] == 'albedo'],
         f'{tag}_corrected_alb_profiles_by_fp_area.png',
         f'[{tag}] Ref-corrected \u0394albedo by footprint area quintile'),
        ('exp_int',        [p for p in pairs if p[5] == 'exp_int'],
         f'{tag}_corrected_exp_profiles_by_fp_area.png',
         f'[{tag}] Ref-corrected \u0394exp_intercept by footprint area quintile'),
    ]
    # Derived groups use a different row tuple format: (diff_col, band_lbl, term_lbl, color)
    if derived_minus:
        var_groups.append((
            'exp\u2212alb', derived_minus,
            f'{tag}_corrected_exp_minus_alb_profiles_by_fp_area.png',
            f'[{tag}] Ref-corrected \u0394(exp\u2212alb) by footprint area quintile',
        ))
    if derived_ratio:
        var_groups.append((
            'exp/alb', derived_ratio,
            f'{tag}_corrected_exp_over_alb_profiles_by_fp_area.png',
            f'[{tag}] Ref-corrected \u0394(exp/alb) by footprint area quintile',
        ))

    fp_outdir = os.path.join(outdir, 'fp_area')
    os.makedirs(fp_outdir, exist_ok=True)
    xp = np.arange(len(labels))

    def _plot_fp_profile(ax, sdf, diff_col):
        """Draw one coloured line per fp_area quintile on ax."""
        if diff_col not in sdf.columns or '_fp_qbin' not in sdf.columns:
            ax.set_visible(False)
            return
        plotted = False
        for qi, qlbl in enumerate(fp_qlabels):
            mask = (sdf['_fp_qbin'] == qlbl) & sdf[diff_col].notna() & sdf['cld_dist_km'].notna()
            sub_q = sdf[mask]
            if len(sub_q) < 10:
                continue
            _bin_q = pd.cut(sub_q['cld_dist_km'], bins=bins, labels=labels, right=False)
            means = sub_q.groupby(_bin_q, observed=True)[diff_col].mean().reindex(labels)
            ns    = sub_q.groupby(_bin_q, observed=True)[diff_col].count().reindex(labels).fillna(0).astype(int)
            stds  = sub_q.groupby(_bin_q, observed=True)[diff_col].std().reindex(labels)
            sems  = (stds / np.sqrt(ns.replace(0, np.nan))).fillna(0)
            finite = means.notna()
            if finite.sum() < 2:
                continue
            color_q = fp_cmap(qi / max(actual_n - 1, 1))
            ax.plot(xp[finite], means.values[finite],
                    color=color_q, lw=1.8, label=qlbl)
            ax.fill_between(xp[finite],
                            (means - sems).values[finite],
                            (means + sems).values[finite],
                            color=color_q, alpha=0.12)
            plotted = True
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, rotation=30, fontsize=7)
        ax.set_xlabel('Cloud distance (km)', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        if plotted:
            ax.legend(fontsize=6, ncol=1, loc='upper right')
        if not plotted:
            ax.set_visible(False)

    # ── render each variable group ────────────────────────────────────────────
    for term_lbl, rows, fname, suptitle in var_groups:
        # standard pairs  → 7-tuple (obs, ref_m, ref_s, dcol, band_lbl, term_lbl, color)
        # derived tuples  → 4-tuple (dcol, band_lbl, term_lbl, color)
        if rows and len(rows[0]) == 7:
            avail = [(p[3], p[4], p[5], p[6]) for p in rows if p[3] in df.columns]
        else:
            avail = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] in df.columns]

        if not avail:
            continue

        n_rows = len(avail)
        n_cols = len(sfc_subsets)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6.5 * n_cols, 4.5 * n_rows),
                                 squeeze=False)

        for ri, (dcol, band_lbl, tl, _color) in enumerate(avail):
            for ci, (sfc_name, sdf) in enumerate(sfc_subsets):
                ax = axes[ri, ci]
                _plot_fp_profile(ax, sdf, dcol)
                if ax.get_visible():
                    ax.set_ylabel(f'{band_lbl} \u0394{tl}\nobs\u2212ref', fontsize=8)
                    ax.set_title(f'{band_lbl} — {sfc_name}', fontsize=9)

        fig.suptitle(
            f'{suptitle}\n'
            f'Lines = fp_area_km\u00b2 quintiles  (Q1=smallest … Q{actual_n}=largest)\n'
            f'y = 0 \u2192 obs matches clear-sky baseline',
            fontsize=11)
        fig.tight_layout()
        _save(fig, fp_outdir, fname)
