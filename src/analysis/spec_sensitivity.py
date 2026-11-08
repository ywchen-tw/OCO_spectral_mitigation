"""
spec_sensitivity.py
===================
Three analyses that pin down what the spectral (cumulant) features contribute
beyond predictive skill — the "spec features are the physics" evidence chain:

1. sub-pixel  — Far-cloud soundings (cld_dist ≥ 20 km, i.e. MODIS sees NO
   cloud within 20 km) binned by a spec elevation index (mean ref-corrected
   Δk₁ z-score across the three bands). If the XCO₂ anomaly mean/scatter grows
   with spec elevation among MODIS-far soundings, the spectra detect cloud
   contamination below the 1-km MODIS detection/quantization floor —
   information the cloud mask cannot provide even in principle.
   (Caveat for the text: elevated far-field Δk₁ can also be aerosol; the
   cross-band coherence requirement suppresses single-band noise.)

2. shadow     — Near-cloud soundings (< 10 km by default, --near-max 15 for
   the 15-km window used by the manuscript Fig. 5) split by the sign of the
   ref-corrected continuum shift (zexp_o2a > +0.5 brightened / < −0.5
   shadowed). Binned profiles of Δk₁ per band and the XCO₂ anomaly for the two
   branches: coherent, opposite-signed responses demonstrate the
   brightening-vs-shadowing mechanism that scalar XCO₂ features cannot
   disentangle.

3. classifier — Per-surface logistic regression (+ gradient-boosted ceiling)
   predicting near-cloud (< 10 km) from RAW per-footprint spec features only
   (k1/k2/k3, exp-intercepts, intercept−albedo mismatches). Deployment-
   compatible: no neighbors, no MODIS. Date-blocked holdout; reports ROC AUC.
   AUC ≫ 0.5 = "the spectra see the clouds", the MODIS-free selling point.

Outputs → <results-root>/figures/cld_dist_analysis/spec_sensitivity/

FIT SCHEMA (2026-11-02): ``--fit-schema v1`` (default, unchanged) reads the
v1 cumulant columns (o2a_k1, ..., exp_o2a_intercept, ``_nosg`` fit suffix via
analysis.ref_corrected).  ``--fit-schema v2`` reads the agreed-model ``_am``
coefficients of the Doppler refit (results_doppler/): ⟨l'⟩ = <band>_k1_am,
var(l') = sco2_k2_am / <band>_k2diag_am (O2A, WCO2), γ = <band>_exp_int_am;
k3 does not exist in v2.  The derived delta/z columns keep the v1 names
(dk1_<band>, zexp_o2a, ...), so the analyses themselves are schema-agnostic.

Usage
-----
    python src/analysis/spec_sensitivity.py \
        --parquet-fname combined_2016_2020_dates.parquet \
        [--analyses subpixel shadow classifier] [--max-rows 4000000]

    # manuscript Fig. 5 edition (2026-09-13): snow-free, BOTH quality flags,
    # 15-km window, same-surface reference windows only
    PYTHONPATH=src python3 -m analysis.spec_sensitivity --analyses shadow \
        --reference production --population snowfree --near-max 15 \
        --exclude-mixed-ref .../mixed_window_flags_snowfree_15km.parquet

    # 2026-11-02 edition on the fit-schema-v2 data (outputs mirror the old
    # layout under results_doppler/figures/...)
    PYTHONPATH=src python3 -m analysis.spec_sensitivity --analyses shadow \
        --fit-schema v2 --results-root <repo>/results_doppler \
        --reference production --population snowfree --near-max 15 \
        --exclude-mixed-ref <repo>/results_doppler/figures/cld_dist_analysis/\
spec_sensitivity/prodref/mixed_surface_windows/mixed_window_flags_snowfree_15km.parquet
"""

import sys
import logging
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (apply_manuscript_style, panel_label,  # noqa: E402
                        XCO2_LABEL, MEAN_L_LABEL)
from analysis.utils import get_storage_dir, apply_quality_filter, _save
from analysis.ref_corrected import add_ref_anomalies, _REF_PAIRS

logger = logging.getLogger(__name__)

_FAR = (20.0, 50.0)
_NEAR_MAX = 10.0

# Raw per-footprint spec features (deployment-compatible: no neighbor info).
SPEC_FEATURES_K = [
    'o2a_k1', 'o2a_k2', 'o2a_k3',
    'wco2_k1', 'wco2_k2', 'wco2_k3',
    'sco2_k1', 'sco2_k2', 'sco2_k3',
]
SPEC_FEATURES_INT = [
    'exp_o2a_intercept', 'exp_wco2_intercept', 'exp_sco2_intercept',
    'o2a_exp_intercept-alb', 'wco2_exp_intercept-alb', 'sco2_exp_intercept-alb',
]
SPEC_FEATURE_SETS = {
    'k_only':         SPEC_FEATURES_K,
    'intercept_only': SPEC_FEATURES_INT,
    'full_spec':      SPEC_FEATURES_K + SPEC_FEATURES_INT,
}

# ── fit schema v2 (agreed model, suffix _am) ──────────────────────────────────
# (delta term, band, coefficient stub): observation <band>_<stub>_am, reference
# <prefix>_<band>_<stub>_am_{mean,std}.  The term names follow the v1 delta
# columns (d<term>_<band>, z<term>_<band>) so downstream code is unchanged.
# 'k2' is var(l'): the SCO2 two-term k2_am, the diagonal-only k2diag_am for
# O2A/WCO2 (same mapping as the Fig. 4 heatmap rows, AM_ROWS in
# manuscript/JQSRT_draft/scripts/make_landclass_heatmap_figure.py).
V2_COEFS = [
    ('k1', 'o2a', 'k1'), ('k2', 'o2a', 'k2diag'), ('exp', 'o2a', 'exp_int'),
    ('a1', 'o2a', 'a1'), ('f', 'o2a', 'f'),
    ('k1', 'wco2', 'k1'), ('k1w', 'wco2', 'k1w'), ('k2', 'wco2', 'k2diag'),
    ('exp', 'wco2', 'exp_int'), ('a1', 'wco2', 'a1'),
    ('k1', 'sco2', 'k1'), ('k1w', 'sco2', 'k1w'), ('k2', 'sco2', 'k2'),
    ('exp', 'sco2', 'exp_int'), ('a1', 'sco2', 'a1'), ('a2', 'sco2', 'a2'),
]
_V2_BANDS = ('o2a', 'wco2', 'sco2')
_V2_VAR_L = {'o2a': 'k2diag', 'wco2': 'k2diag', 'sco2': 'k2'}
SPEC_FEATURES_K_V2 = ([f'{b}_k1_am' for b in _V2_BANDS]
                      + [f'{b}_{_V2_VAR_L[b]}_am' for b in _V2_BANDS])
SPEC_FEATURES_INT_V2 = ([f'{b}_exp_int_am' for b in _V2_BANDS]
                        + [f'{b}_exp_int_am_minus_alb' for b in _V2_BANDS])
SPEC_FEATURE_SETS_V2 = {
    'k_only':         SPEC_FEATURES_K_V2,
    'intercept_only': SPEC_FEATURES_INT_V2,
    'full_spec':      SPEC_FEATURES_K_V2 + SPEC_FEATURES_INT_V2,
}
FEATURE_SETS_BY_SCHEMA = {'v1': SPEC_FEATURE_SETS, 'v2': SPEC_FEATURE_SETS_V2}
# Extra delta columns written to the shadow statistics CSV (not plotted), so
# var(l'), γ and the new agreed-model terms can be read per branch and bin.
SHADOW_EXTRA_VARS = {
    'v1': [],
    'v2': [f'd{term}_{band}' for term, band, _ in V2_COEFS if term != 'k1'],
}

_SURFACES = [('ocean', 0), ('land', 1)]


# ── 1. Sub-pixel cloud check ──────────────────────────────────────────────────

def run_subpixel_check(df: pd.DataFrame, outdir: str,
                       n_bins: int = 10, n_min: int = 2000) -> None:
    """XCO₂ anomaly vs spec elevation index among MODIS-far soundings."""
    zcols = ['zk1_o2a', 'zk1_wco2', 'zk1_sco2']
    if not all(c in df.columns for c in zcols) or 'xco2_bc_anomaly' not in df.columns:
        logger.warning("subpixel: missing zk1_*/xco2_bc_anomaly columns — skipped")
        return

    d = df['cld_dist_km']
    far = df[(d >= _FAR[0]) & (d < _FAR[1])].copy()
    # cross-band coherent elevation: require all three bands present
    z = far[zcols]
    far = far[z.notna().all(axis=1)]
    far['spec_idx'] = far[zcols].mean(axis=1)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for sname, scode in _SURFACES:
        sub = far[far['sfc_type'] == scode].dropna(subset=['xco2_bc_anomaly'])
        if len(sub) < n_bins * n_min:
            logger.info(f"subpixel [{sname}]: only {len(sub):,} rows — skipped")
            continue
        try:
            q = pd.qcut(sub['spec_idx'], n_bins, duplicates='drop')
        except ValueError:
            continue
        g = sub.groupby(q, observed=True)
        stat = g.agg(spec_idx_med=('spec_idx', 'median'),
                     anom_mean=('xco2_bc_anomaly', 'mean'),
                     anom_sem=('xco2_bc_anomaly', 'sem'),
                     anom_std=('xco2_bc_anomaly', 'std'),
                     n=('xco2_bc_anomaly', 'size'))
        stat = stat[stat['n'] >= n_min]
        for _, r in stat.iterrows():
            rows.append({'surface': sname, **r.to_dict()})
        c = 'C0' if sname == 'ocean' else 'C1'
        axes[0].errorbar(stat['spec_idx_med'], stat['anom_mean'],
                         yerr=stat['anom_sem'], color=c, marker='o',
                         capsize=2, label=sname)
        axes[1].plot(stat['spec_idx_med'], stat['anom_std'], color=c,
                     marker='o', label=sname)

    axes[0].set_ylabel(f'mean {XCO2_LABEL} BC anomaly (ppm)')
    axes[0].set_title('Anomaly mean vs spec elevation (MODIS-far only)')
    axes[1].set_ylabel(rf'$\sigma$ of {XCO2_LABEL} BC anomaly (ppm)')
    axes[1].set_title('Anomaly scatter vs spec elevation (MODIS-far only)')
    panel_label(axes[0], '(a)')
    panel_label(axes[1], '(b)')
    for ax in axes:
        ax.set_xlabel(f'spec elevation index (mean z(Δ{MEAN_L_LABEL}) '
                      'over 3 bands)')
        ax.axvline(0, color='gray', lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'Sub-pixel check: soundings with no MODIS cloud within '
                 f'{_FAR[0]:.0f} km, binned by spectral cloud signature', y=1.02)
    fig.tight_layout()
    _save(fig, outdir, 'subpixel_anomaly_vs_specidx.png')
    pd.DataFrame(rows).to_csv(f"{outdir}/subpixel_anomaly_vs_specidx.csv",
                              index=False)
    logger.info(f"  wrote {outdir}/subpixel_anomaly_vs_specidx.csv")


# ── 2. Shadow vs brightening ──────────────────────────────────────────────────

def _median_ci(vals) -> tuple[float, float, float]:
    """Median and its 95 % confidence interval from order statistics.

    With ``n`` finite values sorted ascending, the interval endpoints are the
    values at ranks ``floor(n/2 - 1.96*sqrt(n)/2)`` and
    ``ceil(n/2 + 1.96*sqrt(n)/2)``, clipped to ``[0, n-1]`` (the usual
    distribution-free binomial/normal-approximation interval).
    """
    v = np.asarray(vals, dtype=np.float64)
    v = np.sort(v[np.isfinite(v)])
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(v))
    half = 1.96 * np.sqrt(n) / 2.0
    lo = int(np.clip(int(np.floor(n / 2.0 - half)), 0, n - 1))
    hi = int(np.clip(int(np.ceil(n / 2.0 + half)), 0, n - 1))
    return med, float(v[lo]), float(v[hi])


def run_shadow_brightening(df: pd.DataFrame, outdir: str,
                           z_thresh: float = 0.5, n_min: int = 500,
                           exclude_mixed_ref: str | Path | None = None,
                           near_max: float = _NEAR_MAX,
                           pop_tag: str = '',
                           extra_vars: list[str] | tuple = ()) -> None:
    """Near-cloud Δk₁/anomaly profiles split by continuum shift sign.

    ``exclude_mixed_ref`` (opt-in) points at the per-footprint flag file written
    by ``workspace/mixed_surface_reference_count.py``
    (``mixed_window_flags*.parquet``: fp_id, sfc_type, n_ref, n_ref_other,
    mixed).  Targets whose clear-sky reference window contains at least one
    footprint of the other surface type are dropped before the branch
    statistics, and the outputs get a ``_samesfc`` stem so the unfiltered
    versions survive for comparison.

    ``near_max`` is the outer edge of the near-cloud window (10 km, the
    original, or 15 km — the land clear-sky reference starts beyond 15 km).
    The bin edges follow: ``[0,1,2,3,5,7,10]`` at 10 km, extended by
    ``10,12,15`` at 15 km.  ``pop_tag`` labels the population in the output
    file names (``''`` = QF0 snow-free, ``'_snowfree'`` = snow-free with both
    quality flags); a ``_<n>km`` window tag is appended for near_max > 10.

    Every (surface, branch, variable) gets one row per distance bin plus a
    pooled ``cld_bin == 'all'`` row over the whole near-cloud window.  Each row
    carries mean/sem/n and the median with its 95 % order-statistic interval
    (median/med_lo/med_hi).  ``extra_vars`` (columns present in ``df``) get the
    same rows in the CSV but are not plotted.
    """
    need = ['zexp_o2a', 'dk1_o2a', 'dk1_wco2', 'dk1_sco2', 'xco2_bc_anomaly']
    if not all(c in df.columns for c in need):
        logger.warning("shadow: missing delta/z columns — skipped")
        return

    d = df['cld_dist_km']
    near = df[(d >= 0) & (d < near_max)].copy()
    suffix = ''
    if exclude_mixed_ref is not None:
        suffix = '_samesfc'
        if 'fp_id' not in near.columns:
            raise ValueError("--exclude-mixed-ref needs the fp_id column")
        flags = pd.read_parquet(exclude_mixed_ref, columns=['fp_id', 'mixed'])
        mixed_ids = flags.loc[flags['mixed'], 'fp_id'].to_numpy()
        drop = near['fp_id'].isin(mixed_ids)
        logger.info(f"  exclude_mixed_ref: {len(flags):,} flagged footprints, "
                    f"{len(mixed_ids):,} mixed-window")
        for _sname, _scode in _SURFACES:
            _s = near['sfc_type'] == _scode
            logger.info(f"  exclude_mixed_ref [{_sname}]: dropping "
                        f"{int((_s & drop).sum()):,} of {int(_s.sum()):,} "
                        f"near-cloud footprints")
        near = near[~drop]
    edges = ([0, 1, 2, 3, 5, 7, 10] if near_max <= 10
             else [0, 1, 2, 3, 5, 7, 10, 12, 15])
    labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges) - 1)]
    win_tag = '' if near_max <= 10 else f'_{int(near_max)}km'
    tag = f"{pop_tag}{win_tag}{suffix}"
    near['_bin'] = pd.cut(near['cld_dist_km'], bins=edges, labels=labels,
                          right=False)
    branches = [
        ('brightened', near['zexp_o2a'] > z_thresh, '#d95f02'),
        ('neutral', near['zexp_o2a'].abs() <= z_thresh, '#7570b3'),
        ('shadowed', near['zexp_o2a'] < -z_thresh, '#1b9e77'),
    ]
    panel_vars = [('dk1_o2a', f'Δ{MEAN_L_LABEL} O$_2$A'),
                  ('dk1_wco2', f'Δ{MEAN_L_LABEL} WCO$_2$'),
                  ('dk1_sco2', f'Δ{MEAN_L_LABEL} SCO$_2$'),
                  ('xco2_bc_anomaly', f'{XCO2_LABEL} BC anomaly (ppm)')]

    extra_vars = [v for v in extra_vars if v in near.columns]

    rows = []
    x = np.arange(len(labels))
    for sname, scode in _SURFACES:
        sdf = near[near['sfc_type'] == scode]
        if not len(sdf):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        # extra variables: statistics only (ax None)
        stat_vars = (list(zip(axes.ravel(), panel_vars))
                     + [(None, (v, v)) for v in extra_vars])
        for k, (ax, (var, vlabel)) in enumerate(stat_vars):
            if ax is not None:
                panel_label(ax, f'({chr(ord("a") + k)})')
            for bname, bmask, color in branches:
                b = sdf[bmask.reindex(sdf.index, fill_value=False)]
                g = b.groupby('_bin', observed=False)[var].agg(
                    ['mean', 'sem', 'count']).reindex(labels)
                per_bin = {str(_k): _v.to_numpy()
                           for _k, _v in b.groupby('_bin', observed=False)[var]}
                ok = g['count'] >= n_min
                if ax is not None and ok.any():
                    m = np.where(ok, g['mean'], np.nan)
                    e = np.where(ok, g['sem'], np.nan)
                    ax.errorbar(x, m, yerr=e, color=color, marker='o', ms=3.5,
                                capsize=2, lw=1.3, label=bname)
                for lbl, (_, r) in zip(labels, g.iterrows()):
                    med, mlo, mhi = _median_ci(per_bin.get(lbl, []))
                    rows.append({'surface': sname, 'branch': bname,
                                 'variable': var, 'cld_bin': lbl,
                                 'mean': r['mean'], 'sem': r['sem'],
                                 'n': int(r['count']), 'median': med,
                                 'med_lo': mlo, 'med_hi': mhi})
                # pooled row over the whole near-cloud window
                pooled = b[var]
                med, mlo, mhi = _median_ci(pooled.to_numpy())
                rows.append({'surface': sname, 'branch': bname,
                             'variable': var, 'cld_bin': 'all',
                             'mean': (pooled.mean() if len(pooled) else np.nan),
                             'sem': (pooled.sem() if len(pooled) else np.nan),
                             'n': int(pooled.count()), 'median': med,
                             'med_lo': mlo, 'med_hi': mhi})
            if ax is None:
                continue
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_title(vlabel, fontsize=10)
            ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xticks(x, labels, rotation=45, ha='right')
            ax.set_xlabel('cloud distance bin (km)')
        axes[0, 0].legend(fontsize=9, title=f'zexp_o2a vs ±{z_thresh}')
        fig.suptitle(f'{sname.capitalize()}: near-cloud (<{near_max:.0f} km) '
                     'response split by continuum shift '
                     '(brightened vs shadowed)', y=0.995)
        fig.tight_layout()
        _save(fig, outdir, f'shadow_brightening_{sname}{tag}.png')

    pd.DataFrame(rows).to_csv(
        f"{outdir}/shadow_brightening_stats{tag}.csv", index=False)
    logger.info(f"  wrote {outdir}/shadow_brightening_stats{tag}.csv")


# ── 3. Spec-only cloud-proximity classifier ───────────────────────────────────

def run_spec_classifier(df: pd.DataFrame, outdir: str,
                        max_train: int = 2_000_000,
                        holdout_frac: float = 0.2,
                        feature_sets: dict | None = None) -> None:
    """Predict near-cloud (<10 km) from raw spec features; date-blocked AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, roc_curve

    d = df['cld_dist_km']
    data = df[(d >= 0) & (d < _FAR[1])].copy()
    data['_near'] = (data['cld_dist_km'] < _NEAR_MAX).astype(np.int8)

    # date-blocked holdout: every 5th date (deterministic, spans seasons/years)
    dates = np.sort(data['date'].unique())
    test_dates = set(dates[::int(round(1 / holdout_frac))])
    is_test = data['date'].isin(test_dates)
    logger.info(f"classifier: {len(dates)} dates, {len(test_dates)} held out")

    rng = np.random.default_rng(0)
    rows, curves = [], {}
    for sname, scode in _SURFACES:
        sm = data['sfc_type'] == scode
        for set_name, feats in (feature_sets or SPEC_FEATURE_SETS).items():
            feats = [f for f in feats if f in data.columns]
            if not feats:
                continue
            cols = feats + ['_near']
            tr = data.loc[sm & ~is_test, cols].dropna()
            te = data.loc[sm & is_test, cols].dropna()
            if len(tr) < 10_000 or len(te) < 5_000 or te['_near'].nunique() < 2:
                continue
            if len(tr) > max_train:
                tr = tr.iloc[rng.choice(len(tr), max_train, replace=False)]

            models = {'logreg': make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))}
            if set_name == 'full_spec':
                models['hgb'] = HistGradientBoostingClassifier(random_state=0)

            for mname, model in models.items():
                model.fit(tr[feats], tr['_near'])
                p = model.predict_proba(te[feats])[:, 1]
                auc = roc_auc_score(te['_near'], p)
                rows.append({'surface': sname, 'feature_set': set_name,
                             'model': mname, 'n_train': len(tr),
                             'n_test': len(te), 'n_features': len(feats),
                             'near_rate_test': float(te['_near'].mean()),
                             'auc': auc})
                logger.info(f"  {sname:5s} {set_name:15s} {mname:6s} "
                            f"AUC={auc:.3f} (n_test={len(te):,})")
                if set_name == 'full_spec':
                    fpr, tpr, _ = roc_curve(te['_near'], p)
                    step = max(1, len(fpr) // 500)
                    curves[(sname, mname)] = (fpr[::step], tpr[::step], auc)

    if not rows:
        logger.warning("classifier: no (surface, feature set) had enough data")
        return
    pd.DataFrame(rows).to_csv(f"{outdir}/spec_cloud_classifier_metrics.csv",
                              index=False)
    logger.info(f"  wrote {outdir}/spec_cloud_classifier_metrics.csv")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    styles = {'logreg': '-', 'hgb': '--'}
    colors = {'ocean': 'C0', 'land': 'C1'}
    for (sname, mname), (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, styles[mname], color=colors[sname],
                label=f'{sname} {mname} (AUC {auc:.3f})')
    ax.plot([0, 1], [0, 1], ':', color='gray', lw=1)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('Near-cloud (<10 km) detection from per-footprint\n'
                 'spec features alone (date-blocked holdout)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, outdir, 'spec_cloud_classifier_roc.png')


# ── driver ────────────────────────────────────────────────────────────────────

def _v2_ref_prefixes(reference: str) -> tuple[str, str]:
    """(ocean, land) reference-statistics prefixes of a --reference choice."""
    return ('r05', 'r15') if reference == 'production' else ('ref', 'ref')


def _needed_columns(analyses: list[str], reference: str = 'common-r10',
                    need_fp_id: bool = False,
                    fit_schema: str = 'v1') -> list[str]:
    cols = {'date', 'lon', 'lat', 'sfc_type', 'cld_dist_km',
            'xco2_bc', 'xco2_qf', 'snow_flag', 'xco2_bc_anomaly'}
    if need_fp_id:
        cols.add('fp_id')
    if fit_schema == 'v2':
        if 'subpixel' in analyses or 'shadow' in analyses:
            for _, band, stub in V2_COEFS:
                cols.add(f'{band}_{stub}_am')
                for p in set(_v2_ref_prefixes(reference)):
                    cols.update((f'{p}_{band}_{stub}_am_mean',
                                 f'{p}_{band}_{stub}_am_std'))
        if reference == 'production':
            cols.update(('xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15'))
        if 'classifier' in analyses:
            cols.update(SPEC_FEATURE_SETS_V2['full_spec'])
        return sorted(cols)
    if 'subpixel' in analyses or 'shadow' in analyses:
        for obs, ref_m, ref_s, *_ in _REF_PAIRS:
            cols.update((obs, ref_m, ref_s))
    if reference == 'production':
        from analysis.ref_corrected import _R05_PAIRS, _R15_PAIRS
        for obs, ref_m, ref_s, *_ in _R05_PAIRS + _R15_PAIRS:
            cols.update((obs, ref_m, ref_s))
        cols.update(('xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15'))
    if 'classifier' in analyses:
        cols.update(SPEC_FEATURE_SETS['full_spec'])
    return sorted(cols)


# production-reference aliasing (2026-08-01, manuscript consistency pass):
# expected column <- (ocean r05 source, land r15 source).  The shadow analysis
# then runs unchanged on per-surface production-reference deltas/z-scores.
_MIXED_FLAGS_DEFAULT = str(
    ROOT / 'results' / 'figures' / 'cld_dist_analysis' / 'spec_sensitivity'
    / 'prodref' / 'mixed_surface_windows' / 'mixed_window_flags.parquet')

_PRODREF_ALIAS = {
    'zexp_o2a': ('zr05exp_o2a', 'zr15exp_o2a'),
    'dk1_o2a': ('dr05k1_o2a', 'dr15k1_o2a'),
    'dk1_wco2': ('dr05k1_wco2', 'dr15k1_wco2'),
    'dk1_sco2': ('dr05k1_sco2', 'dr15k1_sco2'),
    'xco2_bc_anomaly': ('xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15'),
}


def _add_v2_deltas(df: pd.DataFrame, ocean_prefix: str,
                   land_prefix: str) -> pd.DataFrame:
    """Fit-schema-v2 delta/z columns d<term>_<band> / z<term>_<band> for
    every V2_COEFS row, against the ocean / land reference prefix (r05/r15
    for the production reference, ref/ref for common-r10).  Rows without
    reference statistics stay NaN; a zero reference σ gives a NaN z-score."""
    ocean = (df['sfc_type'] == 0).to_numpy()
    new_cols = {}
    for term, band, stub in V2_COEFS:
        stem = f'{band}_{stub}_am'
        need = [stem] + [f'{p}_{stem}_{s}' for p in (ocean_prefix, land_prefix)
                         for s in ('mean', 'std')]
        if not all(c in df.columns for c in need):
            logger.warning(f"v2 deltas: {stem} or its reference statistics "
                           "missing — d/z columns skipped")
            continue
        ref_m = np.where(ocean, df[f'{ocean_prefix}_{stem}_mean'],
                         df[f'{land_prefix}_{stem}_mean'])
        ref_s = np.where(ocean, df[f'{ocean_prefix}_{stem}_std'],
                         df[f'{land_prefix}_{stem}_std'])
        d = df[stem].to_numpy() - ref_m
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(ref_s > 0, d / ref_s, np.nan)
        new_cols[f'd{term}_{band}'] = d.astype(np.float32)
        new_cols[f'z{term}_{band}'] = z.astype(np.float32)
    return df.assign(**new_cols)


def _apply_production_reference(df: pd.DataFrame,
                                fit_schema: str = 'v1') -> pd.DataFrame:
    """Alias the shadow-analysis columns to each surface's production
    reference (ocean r05 / land r15) and screen the anomaly at the
    training threshold (models.pipeline.filter_target_outliers)."""
    ocean = df['sfc_type'] == 0
    if fit_schema == 'v2':
        df = _add_v2_deltas(df, 'r05', 'r15')
        df['xco2_bc_anomaly'] = np.where(ocean, df['xco2_bc_anomaly_r05'],
                                         df['xco2_bc_anomaly_r15'])
    else:
        from analysis.ref_corrected import add_r05_anomalies, add_r15_anomalies
        df = add_r15_anomalies(add_r05_anomalies(df))
        for alias, (c05, c15) in _PRODREF_ALIAS.items():
            df[alias] = np.where(ocean, df[c05], df[c15])
    df.loc[df['xco2_bc_anomaly'].abs() > 100.0, 'xco2_bc_anomaly'] = np.nan
    return df


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    apply_manuscript_style()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--parquet-fname', default='combined_2016_2020_dates.parquet')
    ap.add_argument('--results-root', type=Path, default=None,
                    help="Results tree holding csv_collection/<parquet-fname> "
                         "and receiving figures/cld_dist_analysis/"
                         "spec_sensitivity/ (default <storage_dir>/results; "
                         "the fit-schema-v2 data live in results_doppler).")
    ap.add_argument('--parquet', type=Path, default=None,
                    help='Explicit parquet path (overrides --results-root/'
                         'csv_collection/--parquet-fname for the input only).')
    ap.add_argument('--fit-schema', choices=['v1', 'v2'], default='v1',
                    help="Spectral-fit column schema: 'v1' (default, the "
                         "o2a_k1/k2/k3 cumulant columns) or 'v2' (agreed-model "
                         "_am coefficients of the Doppler refit; var(l') = "
                         "sco2_k2_am / <band>_k2diag_am, no k3).")
    ap.add_argument('--analyses', nargs='+', default=['subpixel', 'shadow', 'classifier'],
                    choices=['subpixel', 'shadow', 'classifier'])
    ap.add_argument('--max-rows', type=int, default=None,
                    help='Optional row cap for quick runs (head of file).')
    ap.add_argument('--reference', choices=['common-r10', 'production'],
                    default='common-r10',
                    help="Clear-sky reference for the shadow analysis: "
                         "'common-r10' (original) or 'production' (ocean r05 "
                         "/ land r15 per-surface aliasing + |anomaly|<=100 ppm "
                         "screen; shadow-only, writes to <outdir>/prodref).")
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--population', choices=['qf0snowfree', 'snowfree'],
                    default='qf0snowfree',
                    help="Footprint population: 'qf0snowfree' (default, "
                         "unchanged: xco2_qf == 0 and snow_flag == 0) or "
                         "'snowfree' (snow-free, BOTH quality flags). "
                         "'snowfree' tags the shadow outputs with _snowfree.")
    ap.add_argument('--near-max', type=float, choices=[10.0, 15.0],
                    default=10.0,
                    help='Outer edge of the shadow-analysis near-cloud window '
                         'in km (10 = original, 15 = out to the land clear-sky '
                         'reference distance; 15 adds the 10–12 and 12–15 km '
                         'bins and a _15km output tag).')
    ap.add_argument('--exclude-mixed-ref', nargs='?', const=_MIXED_FLAGS_DEFAULT,
                    default=None, metavar='FLAGS_PARQUET',
                    help="Drop shadow-analysis targets whose clear-sky "
                         "reference window contains footprints of the other "
                         "surface type, using the flag file from "
                         "workspace/mixed_surface_reference_count.py (bare "
                         f"flag = {_MIXED_FLAGS_DEFAULT}). Writes "
                         "shadow_brightening_stats*_samesfc.csv; the "
                         "unfiltered outputs are left in place.")
    args = ap.parse_args()
    pop_tag = '' if args.population == 'qf0snowfree' else '_snowfree'
    if args.reference == 'production' and args.analyses != ['shadow']:
        ap.error("--reference production supports --analyses shadow only")
    if args.exclude_mixed_ref and not Path(args.exclude_mixed_ref).exists():
        ap.error(f"--exclude-mixed-ref file not found: {args.exclude_mixed_ref}")

    results_root = args.results_root or (get_storage_dir() / 'results')
    path = args.parquet or (results_root / 'csv_collection' / args.parquet_fname)
    outdir = args.outdir or str(results_root / 'figures'
                                / 'cld_dist_analysis' / 'spec_sensitivity')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.reference == 'production':
        outdir = str(Path(outdir) / 'prodref')
        Path(outdir).mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    from analysis.nosg_reference import read_combined, sidecar_path
    # The _nosg reference statistics live in the row-aligned sidecar; a needed
    # column in neither file is dropped (as before) but now logged.
    avail = set(pq.read_schema(path).names)
    if sidecar_path(path).exists():
        avail |= set(pq.read_schema(sidecar_path(path)).names)
    # a schema mismatch would otherwise only surface as a skipped analysis
    probe = 'o2a_k1_am' if args.fit_schema == 'v2' else 'o2a_k1'
    if probe not in avail:
        ap.error(f"{path.name} has no {probe} column; wrong --fit-schema "
                 f"({args.fit_schema})?")
    needed = _needed_columns(args.analyses, args.reference,
                             need_fp_id=bool(args.exclude_mixed_ref),
                             fit_schema=args.fit_schema)
    cols = [c for c in needed if c in avail]
    if len(cols) < len(needed):
        logger.warning("columns not found, skipped: %s", sorted(set(needed) - avail))
    logger.info(f"Loading {len(cols)} columns from {path.name} …")
    df = read_combined(path, cols)
    if args.max_rows:
        df = df.iloc[:args.max_rows]
    logger.info(f"Loaded {len(df):,} rows")

    df = apply_quality_filter(
        df, require_qf0=(args.population == 'qf0snowfree'))
    if args.reference == 'production':
        logger.info("Computing production-reference (r05/r15) deltas …")
        df = _apply_production_reference(df, args.fit_schema)
    elif any(a in args.analyses for a in ('subpixel', 'shadow')):
        logger.info("Computing ref-corrected deltas …")
        df = (_add_v2_deltas(df, 'ref', 'ref') if args.fit_schema == 'v2'
              else add_ref_anomalies(df))

    if 'subpixel' in args.analyses:
        logger.info("── sub-pixel cloud check ──")
        run_subpixel_check(df, outdir)
    if 'shadow' in args.analyses:
        logger.info("── shadow vs brightening ──")
        run_shadow_brightening(df, outdir,
                               exclude_mixed_ref=args.exclude_mixed_ref,
                               near_max=args.near_max, pop_tag=pop_tag,
                               extra_vars=SHADOW_EXTRA_VARS[args.fit_schema])
    if 'classifier' in args.analyses:
        logger.info("── spec-only cloud classifier ──")
        run_spec_classifier(
            df, outdir, feature_sets=FEATURE_SETS_BY_SCHEMA[args.fit_schema])
    logger.info(f"Done → {outdir}")


if __name__ == '__main__':
    main()
