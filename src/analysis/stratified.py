"""
ca_stratified.py
================
Stratified analysis functions extracted from combined_analyze.py.

Contents
--------
- STRAT_CONFIG                       Dict of stratification variables and bin edges
- _safe_label                        Convert bin label to filesystem-safe string
- _build_strata                      Clip edges to data range; assign _strat column
- plot_k1_k2_overlay                 All strata k1/k2 profiles on one figure
- plot_intercept_overlay             All strata exp_intercept profiles on one figure
- plot_xco2_anomaly_binned_overlay   All strata XCO2 anomaly profiles on one figure
- run_stratified_analysis            Per-stratum core plots + overlay comparisons
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .utils import _save, bin_by_cld_dist
from .k_coeff import (plot_distributions_vs_cld_dist, plot_k1_k2_binned_profile,
                         plot_k1_k2_vs_cld_dist, plot_k2_over_k1_vs_cld_dist,
                         plot_k1_k2_joint, plot_cross_band_k_combinations)
from .albedo import plot_intercept_binned_profile, plot_alb_binned_profile
from .xco2 import (plot_xco2_anomaly_partial, plot_xco2_anomaly_vs_key_vars,
                     plot_xco2_anomaly_vs_cld_dist_binned, plot_xco2_anomaly_correlations)

logger = logging.getLogger(__name__)


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
    'mu_sza':      ([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.01], 'cosSZA'),
    # Raw SZA in degrees — used if the column exists instead of mu_sza
    'sza':         ([0, 20, 40, 55, 70, 90],         'deg'),
    'alb_o2a':     ([0.00, 0.05, 0.10, 0.25, 0.50, 1.00],  ''),
    'glint_angle': ([0.0,  5.0, 10.0, 20.0, 45.0],   'deg'),
    'aod_total':   ([0.00, 0.025, 0.05, 0.10, 0.25, 1.00],  ''),
    # dp = retrieved – prior surface pressure (hPa); range is roughly -8 to +10
    'dp':          ([-10, -5, -2, 0, 2, 5, 10],             'hPa'),
    # fp_area_km2 = footprint area in km²
    'fp_area_km2': ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0],   'km2'),
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
    """Mean XCO2 profiles for all strata overlaid on one figure (one file per target)."""
    targets = [
        ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)',  'C0'),
        ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)', 'C1'),
        ('xco2_bc',          'XCO\u2082 BC (ppm)',           'C2'),
        ('xco2_raw',         'XCO\u2082 raw (ppm)',          'C3'),
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
        _save(fig, outdir, f'{col}_binned_overlay.png')


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
        _XCO2_TARGETS = [
            ('xco2_bc_anomaly',  'XCO\u2082 BC anomaly (ppm)'),
            ('xco2_raw_anomaly', 'XCO\u2082 raw anomaly (ppm)'),
            ('xco2_bc',          'XCO\u2082 BC (ppm)'),
            ('xco2_raw',         'XCO\u2082 raw (ppm)'),
        ]
        plot_xco2_anomaly_vs_cld_dist_binned(
            sdf, cld_bins, cld_labels, sdir,
            targets=[(c, l, clr) for (c, l), clr in
                     zip(_XCO2_TARGETS, ('C0', 'C1', 'C2', 'C3'))
                     if c in sdf.columns])
        for _tcol, _tlbl in _XCO2_TARGETS:
            if _tcol in sdf.columns:
                plot_xco2_anomaly_vs_key_vars(sdf, sdir, target=_tcol,
                                              target_label=_tlbl)
        plot_k1_k2_binned_profile(sdf, cld_bins, cld_labels, sdir)
        plot_intercept_binned_profile(sdf, cld_bins, cld_labels, sdir)

    # Overlay comparison figures — all strata on one plot
    logger.info(f"    Generating overlay figures for '{strat_var}' …")
    plot_k1_k2_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
    plot_intercept_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
    plot_xco2_anomaly_binned_overlay(df, cld_bins, cld_labels, overlay_dir, strat_var, bin_labels)
