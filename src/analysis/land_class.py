"""
land_class.py
=============
Spec-feature response to cloud proximity, stratified by a categorical group
column — primarily the IGBP land-cover group (see land_cover.py), but the
engine is generic (``group_col``): run_all.py also uses it with the footprint
index to replace the legacy fp_0..7 per-directory loop.

Design principles (differ deliberately from the per-directory stratified suite):

1. Overlays on shared axes — one figure per variable, one line per group.
   Cross-group comparison IS the question, so groups share axes.
2. Compare deltas, not raw levels — raw k1/intercept levels differ across
   groups mainly through albedo/airmass. Each group is normalized to its own
   far-cloud (20–50 km) baseline and scaled by its far-cloud std, so profiles
   read in z-units ("how many far-field sigmas does this group move near
   cloud").  Raw-mean panels are kept alongside for sanity.  Where the parquet
   carries ref_* columns, ref-corrected per-sounding deltas (dk1/dk2/dexp) are
   analysed too.
3. One diagnosable artifact — {prefix}_effect_sizes.csv + heatmap: rows =
   variables, cols = groups, cell = near−far effect in z-units with an
   analytic 95% CI. Cells below n_min are blanked, never plotted noisy.

Outputs (under <outdir>):
    {prefix}_counts.csv / {prefix}_counts.png
    {prefix}_profile_{var}.png            (raw + z panels, one line per group)
    {prefix}_k_overview_z.png             (3 bands × k1/k2 z-profiles)
    {prefix}_effect_sizes.csv / {prefix}_effect_heatmap.png
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from plot_style import (apply_manuscript_style, panel_label, CMAPS,  # noqa: E402
                        XCO2_LABEL, MEAN_L_LABEL, VAR_L_LABEL)

from .utils import _save, bin_by_cld_dist
from .land_cover import GROUP_ORDER, GROUP_COLORS

logger = logging.getLogger(__name__)

# (column, short label) — profiles/effects computed for those present.
LAND_CLASS_VARS = [
    ('o2a_k1',              f'O$_2$A {MEAN_L_LABEL}'),
    ('o2a_k2',              f'O$_2$A {VAR_L_LABEL}'),
    ('wco2_k1',             f'WCO$_2$ {MEAN_L_LABEL}'),
    ('wco2_k2',             f'WCO$_2$ {VAR_L_LABEL}'),
    ('sco2_k1',             f'SCO$_2$ {MEAN_L_LABEL}'),
    ('sco2_k2',             f'SCO$_2$ {VAR_L_LABEL}'),
    ('exp_o2a_intercept',   'O$_2$A exp-intercept'),
    ('exp_wco2_intercept',  'WCO$_2$ exp-intercept'),
    ('exp_sco2_intercept',  'SCO$_2$ exp-intercept'),
    ('xco2_bc_anomaly',     f'{XCO2_LABEL} BC anomaly'),
    ('xco2_raw_minus_apriori', f'{XCO2_LABEL} raw - apriori'),
]

# Ref-corrected per-sounding deltas (obs − clear-sky-neighbor reference).
# These remove the scene baseline per SOUNDING (vs the per-group far-cloud
# normalization applied to the raw variables above), so cross-group curves
# compare pure cloud response. Computed by ref_corrected.add_ref_anomalies
# when the parquet carries ref_* columns.
LAND_CLASS_DELTA_VARS = [
    ('dk1_o2a',   f'Δ{MEAN_L_LABEL} O$_2$A (obs-ref)'),
    ('dk2_o2a',   f'Δ{VAR_L_LABEL} O$_2$A (obs-ref)'),
    ('dk1_wco2',  f'Δ{MEAN_L_LABEL} WCO$_2$ (obs-ref)'),
    ('dk2_wco2',  f'Δ{VAR_L_LABEL} WCO$_2$ (obs-ref)'),
    ('dk1_sco2',  f'Δ{MEAN_L_LABEL} SCO$_2$ (obs-ref)'),
    ('dk2_sco2',  f'Δ{VAR_L_LABEL} SCO$_2$ (obs-ref)'),
    ('dexp_o2a',  'Δexp-int O$_2$A (obs-ref)'),
    ('dexp_wco2', 'Δexp-int WCO$_2$ (obs-ref)'),
    ('dexp_sco2', 'Δexp-int SCO$_2$ (obs-ref)'),
]

_NEAR = (0.0, 5.0)     # near-cloud window (km)
_FAR = (20.0, 50.0)    # far-cloud baseline window (km)

# Groups analysed over land (water = coastal-cell leakage, fill = no data).
_EXCLUDED_GROUPS = ('water', 'fill')


def _select_groups(df: pd.DataFrame, group_col: str, group_order: list[str],
                   min_class_n: int) -> list[str]:
    counts = df[group_col].value_counts()
    groups = [g for g in group_order if counts.get(g, 0) >= min_class_n]
    dropped = [f"{g}({counts.get(g, 0):,})" for g in group_order
               if 0 < counts.get(g, 0) < min_class_n]
    if dropped:
        logger.info(f"  groups below min_class_n={min_class_n:,}: {', '.join(dropped)}")
    return groups


def _binned_stats(df: pd.DataFrame, var: str, groups: list[str],
                  bins, labels, group_col: str) -> pd.DataFrame:
    """Per (group, cld bin): mean, sem, count for *var*."""
    sub = df[[group_col, 'cld_dist_km', var]].dropna()
    sub = sub[sub[group_col].isin(groups)]
    b = bin_by_cld_dist(sub, bins, labels)
    g = sub.groupby([sub[group_col], b], observed=False)[var]
    out = g.agg(['mean', 'sem', 'count'])
    out.index.names = ['group', 'cld_bin']
    return out


def _far_baseline(df: pd.DataFrame, var: str, groups: list[str],
                  group_col: str) -> pd.DataFrame:
    """Per group: far-window mean/std/sem/count for *var*."""
    d = df['cld_dist_km']
    far = df[(d >= _FAR[0]) & (d < _FAR[1])]
    g = far.groupby(group_col, observed=True)[var].agg(
        ['mean', 'std', 'sem', 'count'])
    return g.reindex(groups)


def write_count_matrix(df: pd.DataFrame, bins, labels, outdir: str,
                       groups: list[str], group_col: str,
                       prefix: str, title_tag: str) -> pd.DataFrame:
    """Group × cloud-distance-bin count matrix (CSV + figure)."""
    b = bin_by_cld_dist(df, bins, labels)
    mat = (df.groupby([df[group_col], b], observed=False)
             .size().unstack(fill_value=0).reindex(groups))
    mat.to_csv(f"{outdir}/{prefix}_counts.csv")
    logger.info(f"  wrote {outdir}/{prefix}_counts.csv")

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 0.5 * len(groups) + 1.5))
    im = ax.imshow(np.log10(mat.values + 1), cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_yticks(range(len(groups)), groups)
    for i in range(len(groups)):
        for j in range(len(labels)):
            n = mat.values[i, j]
            txt = f"{n/1e6:.1f}M" if n >= 1e6 else (f"{n/1e3:.0f}k" if n >= 1e3 else str(n))
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if np.log10(n + 1) < 0.6 * np.log10(mat.values.max() + 1) else 'black')
    ax.set_xlabel('cloud distance bin (km)')
    ax.set_title(f'Sounding counts by {title_tag} × cloud distance')
    fig.colorbar(im, ax=ax, label='log10(count)')
    _save(fig, outdir, f'{prefix}_counts.png')
    return mat


def plot_class_profile(df: pd.DataFrame, var: str, label: str,
                       bins, labels, outdir: str, groups: list[str],
                       n_min: int, group_col: str, colors: dict,
                       prefix: str, title_tag: str) -> None:
    """Raw and z-normalized binned profiles vs cloud distance, one line/group."""
    stats = _binned_stats(df, var, groups, bins, labels, group_col)
    base = _far_baseline(df, var, groups, group_col)
    x = np.arange(len(labels))

    fig, (ax_raw, ax_z) = plt.subplots(1, 2, figsize=(13, 4.5))
    for grp in groups:
        try:
            s = stats.loc[grp].reindex(labels)
        except KeyError:
            continue
        ok = s['count'] >= n_min
        if not ok.any():
            continue
        m = np.where(ok, s['mean'], np.nan)
        e = np.where(ok, s['sem'], np.nan)
        c = colors.get(grp, 'k')
        ax_raw.errorbar(x, m, yerr=e, color=c, marker='o', ms=3.5,
                        capsize=2, lw=1.3, label=str(grp))

        mu0, sd0 = base.loc[grp, 'mean'], base.loc[grp, 'std']
        if not np.isfinite(sd0) or sd0 == 0:
            continue
        ax_z.errorbar(x, (m - mu0) / sd0, yerr=e / sd0, color=c, marker='o',
                      ms=3.5, capsize=2, lw=1.3, label=str(grp))

    for ax, ttl, yl in ((ax_raw, f'{label} — raw mean ± SEM', label),
                        (ax_z, f'{label} — Δ from far-cloud baseline (z-units)',
                         r'(mean $-$ far) / $\sigma_\mathrm{far}$')):
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_xlabel('cloud distance bin (km)')
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    panel_label(ax_raw, '(a)')
    panel_label(ax_z, '(b)')
    ax_z.axhline(0, color='gray', lw=0.8)
    ax_raw.legend(fontsize=8, ncol=2)
    fig.suptitle(f'{label} vs cloud distance by {title_tag}', y=1.02)
    fig.tight_layout()
    _save(fig, outdir, f'{prefix}_profile_{var}.png')


def plot_k_overview_z(df: pd.DataFrame, bins, labels, outdir: str,
                      groups: list[str], n_min: int, group_col: str,
                      colors: dict, prefix: str, title_tag: str) -> None:
    """3 bands × (k1, k2) z-profiles — the single cross-group mechanism figure."""
    panel_vars = [('o2a_k1', f'O$_2$A {MEAN_L_LABEL}'),
                  ('o2a_k2', f'O$_2$A {VAR_L_LABEL}'),
                  ('wco2_k1', f'WCO$_2$ {MEAN_L_LABEL}'),
                  ('wco2_k2', f'WCO$_2$ {VAR_L_LABEL}'),
                  ('sco2_k1', f'SCO$_2$ {MEAN_L_LABEL}'),
                  ('sco2_k2', f'SCO$_2$ {VAR_L_LABEL}')]
    panel_vars = [(v, l) for v, l in panel_vars if v in df.columns]
    if not panel_vars:
        return
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for k, (ax, (var, label)) in enumerate(zip(axes.ravel(), panel_vars)):
        panel_label(ax, f'({chr(ord("a") + k)})')
        stats = _binned_stats(df, var, groups, bins, labels, group_col)
        base = _far_baseline(df, var, groups, group_col)
        for grp in groups:
            try:
                s = stats.loc[grp].reindex(labels)
            except KeyError:
                continue
            ok = s['count'] >= n_min
            mu0, sd0 = base.loc[grp, 'mean'], base.loc[grp, 'std']
            if not ok.any() or not np.isfinite(sd0) or sd0 == 0:
                continue
            m = np.where(ok, s['mean'], np.nan)
            e = np.where(ok, s['sem'], np.nan)
            ax.errorbar(x, (m - mu0) / sd0, yerr=e / sd0,
                        color=colors.get(grp, 'k'), marker='o', ms=3,
                        capsize=2, lw=1.2, label=str(grp))
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_xlabel('cloud distance bin (km)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'(mean $-$ far) / $\sigma_\mathrm{far}$')
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f'Path-length cumulants vs cloud distance by {title_tag} '
                 '(z-normalized to each group’s far-cloud baseline)', y=0.995)
    fig.tight_layout()
    _save(fig, outdir, f'{prefix}_k_overview_z.png')


def build_effect_sizes(df: pd.DataFrame, variables, outdir: str,
                       groups: list[str], n_min: int, group_col: str,
                       prefix: str) -> pd.DataFrame:
    """Near−far effect per (variable, group), raw and z-units, analytic 95% CI."""
    d = df['cld_dist_km']
    near_m = (d >= _NEAR[0]) & (d < _NEAR[1])
    far_m = (d >= _FAR[0]) & (d < _FAR[1])
    slope_m = (d >= 0) & (d < 50)

    rows = []
    for var, label in variables:
        if var not in df.columns:
            continue
        for grp in groups:
            gm = df[group_col] == grp
            yn = df.loc[gm & near_m, var].dropna()
            yf = df.loc[gm & far_m, var].dropna()
            rec = {'variable': var, 'label': label, 'group': grp,
                   'n_near': len(yn), 'n_far': len(yf)}
            if len(yn) >= n_min and len(yf) >= n_min and yf.std() > 0:
                sd0 = float(yf.std())
                eff = float(yn.mean() - yf.mean())
                se = float(np.hypot(yn.sem(), yf.sem()))
                sub = df.loc[gm & slope_m, ['cld_dist_km', var]].dropna()
                r = (float(sub['cld_dist_km'].corr(sub[var]))
                     if len(sub) >= n_min else np.nan)
                rec.update({
                    'near_mean': float(yn.mean()), 'far_mean': float(yf.mean()),
                    'far_std': sd0,
                    'effect_raw': eff, 'effect_z': eff / sd0,
                    'ci95_z': 1.96 * se / sd0,
                    'pearson_r_cld_dist': r,
                })
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(f"{outdir}/{prefix}_effect_sizes.csv", index=False)
    logger.info(f"  wrote {outdir}/{prefix}_effect_sizes.csv")
    return out


def plot_effect_heatmap(eff: pd.DataFrame, outdir: str,
                        groups: list[str], prefix: str) -> None:
    """Variable × group heatmap of near−far effects in z-units."""
    if eff.empty or 'effect_z' not in eff.columns:
        return
    labels_by_var = dict(zip(eff['variable'], eff['label']))
    mat = eff.pivot(index='variable', columns='group', values='effect_z')
    ci = eff.pivot(index='variable', columns='group', values='ci95_z')
    var_order = [v for v, _ in LAND_CLASS_VARS + LAND_CLASS_DELTA_VARS
                 if v in mat.index]
    mat = mat.reindex(index=var_order, columns=groups)
    ci = ci.reindex(index=var_order, columns=groups)

    vmax = np.nanmax(np.abs(mat.values)) or 1.0
    fig, ax = plt.subplots(figsize=(1.1 * len(groups) + 3, 0.55 * len(var_order) + 2))
    im = ax.imshow(mat.values, cmap=CMAPS['mu'], vmin=-vmax, vmax=vmax,
                   aspect='auto')
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha='right')
    ax.set_yticks(range(len(var_order)), [labels_by_var[v] for v in var_order])
    for i in range(len(var_order)):
        for j in range(len(groups)):
            v, c = mat.values[i, j], ci.values[i, j]
            if np.isfinite(v):
                sig = abs(v) > c
                ax.text(j, i, f"{v:+.2f}{'*' if sig else ''}", ha='center',
                        va='center', fontsize=8,
                        fontweight='bold' if sig else 'normal',
                        color='white' if abs(v) > 0.6 * vmax else 'black')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=8, color='gray')
    ax.set_title('Near-cloud (0–5 km) $-$ far (20–50 km) effect, z-units\n'
                 '(* = |effect| exceeds its 95% CI; — = below n_min)')
    fig.colorbar(im, ax=ax, label='effect (z)')
    _save(fig, outdir, f'{prefix}_effect_heatmap.png')


def run_land_class_analysis(df: pd.DataFrame, bins, labels, outdir: str,
                            n_min: int = 500,
                            min_class_n: int = 10_000,
                            group_col: str = 'igbp_group',
                            group_order: list | None = None,
                            colors: dict | None = None,
                            prefix: str = 'landclass',
                            title_tag: str = 'land-cover group') -> None:
    """Group-stratified spec-feature suite.

    Defaults analyse the IGBP land-cover group (pass a land / sfc_type==1
    subset); with group_col/group_order/colors/prefix overridden the same
    engine serves any categorical stratifier (e.g. footprint index).

    n_min       minimum soundings per (group, cld-bin) cell to plot/score
    min_class_n minimum total soundings for a group to be analysed at all
    """
    if group_col not in df.columns:
        logger.warning(f"No {group_col} column — nothing to do "
                       "(for igbp_group run land_cover.assign_land_cover first)")
        return

    apply_manuscript_style()

    import os
    os.makedirs(outdir, exist_ok=True)

    if group_order is None:
        group_order = [g for g in GROUP_ORDER if g not in _EXCLUDED_GROUPS]
    if colors is None:
        colors = GROUP_COLORS

    groups = _select_groups(df, group_col, group_order, min_class_n)
    if not groups:
        logger.warning("No group meets min_class_n — nothing to do")
        return
    logger.info(f"{title_tag} analysis over groups: {groups}")

    # Ref-corrected deltas (per-sounding baseline removal) where ref_* exist.
    if 'dk1_o2a' not in df.columns and 'ref_o2a_k1_mean' in df.columns:
        from .ref_corrected import add_ref_anomalies
        logger.info("  computing ref-corrected delta columns …")
        df = add_ref_anomalies(df)

    write_count_matrix(df, bins, labels, outdir, groups, group_col,
                       prefix, title_tag)

    variables = [(v, l) for v, l in LAND_CLASS_VARS + LAND_CLASS_DELTA_VARS
                 if v in df.columns]
    for var, label in variables:
        logger.info(f"  profile: {var}")
        plot_class_profile(df, var, label, bins, labels, outdir, groups,
                           n_min, group_col, colors, prefix, title_tag)

    plot_k_overview_z(df, bins, labels, outdir, groups, n_min, group_col,
                      colors, prefix, title_tag)

    eff = build_effect_sizes(df, variables, outdir, groups, n_min,
                             group_col, prefix)
    plot_effect_heatmap(eff, outdir, groups, prefix)
    logger.info(f"{title_tag} analysis written to {outdir}")
