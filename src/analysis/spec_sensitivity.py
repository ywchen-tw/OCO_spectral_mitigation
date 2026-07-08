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

2. shadow     — Near-cloud soundings (< 10 km) split by the sign of the
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

Outputs → results/figures/<analysis_dir>/spec_sensitivity/

Usage
-----
    python src/analysis/spec_sensitivity.py \
        --parquet-fname combined_2016_2020_dates.parquet \
        [--analyses subpixel shadow classifier] [--max-rows 4000000]
"""

import sys
import logging
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    axes[0].set_ylabel('mean XCO₂ BC anomaly (ppm)')
    axes[0].set_title('Anomaly mean vs spec elevation (MODIS-far only)')
    axes[1].set_ylabel('σ of XCO₂ BC anomaly (ppm)')
    axes[1].set_title('Anomaly scatter vs spec elevation (MODIS-far only)')
    for ax in axes:
        ax.set_xlabel('spec elevation index (mean z(Δk₁) over 3 bands)')
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

def run_shadow_brightening(df: pd.DataFrame, outdir: str,
                           z_thresh: float = 0.5, n_min: int = 500) -> None:
    """Near-cloud Δk₁/anomaly profiles split by continuum shift sign."""
    need = ['zexp_o2a', 'dk1_o2a', 'dk1_wco2', 'dk1_sco2', 'xco2_bc_anomaly']
    if not all(c in df.columns for c in need):
        logger.warning("shadow: missing delta/z columns — skipped")
        return

    d = df['cld_dist_km']
    near = df[(d >= 0) & (d < _NEAR_MAX)].copy()
    edges = [0, 1, 2, 3, 5, 7, 10]
    labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges) - 1)]
    near['_bin'] = pd.cut(near['cld_dist_km'], bins=edges, labels=labels,
                          right=False)
    branches = [
        ('brightened', near['zexp_o2a'] > z_thresh, '#d95f02'),
        ('neutral', near['zexp_o2a'].abs() <= z_thresh, '#7570b3'),
        ('shadowed', near['zexp_o2a'] < -z_thresh, '#1b9e77'),
    ]
    panel_vars = [('dk1_o2a', 'Δk₁ O₂A'), ('dk1_wco2', 'Δk₁ WCO₂'),
                  ('dk1_sco2', 'Δk₁ SCO₂'),
                  ('xco2_bc_anomaly', 'XCO₂ BC anomaly (ppm)')]

    rows = []
    x = np.arange(len(labels))
    for sname, scode in _SURFACES:
        sdf = near[near['sfc_type'] == scode]
        if not len(sdf):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (var, vlabel) in zip(axes.ravel(), panel_vars):
            for bname, bmask, color in branches:
                b = sdf[bmask.reindex(sdf.index, fill_value=False)]
                g = b.groupby('_bin', observed=False)[var].agg(
                    ['mean', 'sem', 'count']).reindex(labels)
                ok = g['count'] >= n_min
                if not ok.any():
                    continue
                m = np.where(ok, g['mean'], np.nan)
                e = np.where(ok, g['sem'], np.nan)
                ax.errorbar(x, m, yerr=e, color=color, marker='o', ms=3.5,
                            capsize=2, lw=1.3, label=bname)
                for lbl, (_, r) in zip(labels, g.iterrows()):
                    rows.append({'surface': sname, 'branch': bname,
                                 'variable': var, 'cld_bin': lbl,
                                 'mean': r['mean'], 'sem': r['sem'],
                                 'n': int(r['count'])})
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_title(vlabel, fontsize=10)
            ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xticks(x, labels, rotation=45, ha='right')
            ax.set_xlabel('cloud distance bin (km)')
        axes[0, 0].legend(fontsize=9, title=f'zexp_o2a vs ±{z_thresh}')
        fig.suptitle(f'{sname.capitalize()}: near-cloud response split by '
                     'continuum shift (brightened vs shadowed)', y=0.995)
        fig.tight_layout()
        _save(fig, outdir, f'shadow_brightening_{sname}.png')

    pd.DataFrame(rows).to_csv(f"{outdir}/shadow_brightening_stats.csv",
                              index=False)
    logger.info(f"  wrote {outdir}/shadow_brightening_stats.csv")


# ── 3. Spec-only cloud-proximity classifier ───────────────────────────────────

def run_spec_classifier(df: pd.DataFrame, outdir: str,
                        max_train: int = 2_000_000,
                        holdout_frac: float = 0.2) -> None:
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
        for set_name, feats in SPEC_FEATURE_SETS.items():
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

def _needed_columns(analyses: list[str]) -> list[str]:
    cols = {'date', 'lon', 'lat', 'sfc_type', 'cld_dist_km',
            'xco2_bc', 'xco2_qf', 'snow_flag', 'xco2_bc_anomaly'}
    if 'subpixel' in analyses or 'shadow' in analyses:
        for obs, ref_m, ref_s, *_ in _REF_PAIRS:
            cols.update((obs, ref_m, ref_s))
    if 'classifier' in analyses:
        cols.update(SPEC_FEATURE_SETS['full_spec'])
    return sorted(cols)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--parquet-fname', default='combined_2016_2020_dates.parquet')
    ap.add_argument('--analyses', nargs='+', default=['subpixel', 'shadow', 'classifier'],
                    choices=['subpixel', 'shadow', 'classifier'])
    ap.add_argument('--max-rows', type=int, default=None,
                    help='Optional row cap for quick runs (head of file).')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    storage_dir = get_storage_dir()
    path = storage_dir / 'results' / 'csv_collection' / args.parquet_fname
    outdir = args.outdir or str(storage_dir / 'results' / 'figures'
                                / 'cld_dist_analysis' / 'spec_sensitivity')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    avail = set(pq.read_schema(path).names)
    cols = [c for c in _needed_columns(args.analyses) if c in avail]
    logger.info(f"Loading {len(cols)} columns from {path.name} …")
    if args.max_rows:
        pf = pq.ParquetFile(path)
        batches = []
        n = 0
        for b in pf.iter_batches(columns=cols, batch_size=1 << 20):
            batches.append(b)
            n += b.num_rows
            if n >= args.max_rows:
                break
        import pyarrow as pa
        df = pa.Table.from_batches(batches).to_pandas()
    else:
        df = pd.read_parquet(path, columns=cols)
    logger.info(f"Loaded {len(df):,} rows")

    df = apply_quality_filter(df)
    if any(a in args.analyses for a in ('subpixel', 'shadow')):
        logger.info("Computing ref-corrected deltas …")
        df = add_ref_anomalies(df)

    if 'subpixel' in args.analyses:
        logger.info("── sub-pixel cloud check ──")
        run_subpixel_check(df, outdir)
    if 'shadow' in args.analyses:
        logger.info("── shadow vs brightening ──")
        run_shadow_brightening(df, outdir)
    if 'classifier' in args.analyses:
        logger.info("── spec-only cloud classifier ──")
        run_spec_classifier(df, outdir)
    logger.info(f"Done → {outdir}")


if __name__ == '__main__':
    main()
