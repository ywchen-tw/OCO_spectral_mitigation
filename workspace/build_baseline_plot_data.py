"""build_baseline_plot_data.py — linreg / XGBoost analog of build_tabm_plot_data.py.

Applies a per-surface baseline model (Ridge or plain XGBoost-mean), pooling all
date_kfold folds, to an unseen parquet and writes one plot_data.parquet with the
columns the TCCON collocator / tccon_comparison_report.py expect:

    time, lon, lat, cld_dist_km, sfc_type, xco2_bc, xco2_bc_anomaly, xco2_apriori,
    pred_anomaly, clim_guard, anomaly_guard, <kind>_corrected_xco2

Both baselines are POINT predictors (single mu = corrected anomaly), so — unlike
the DE / TabM builders — there are no sigma / quantile columns.  Folds are pooled
by averaging mu across every fold (each fold uses its OWN fitted pipeline / scaler),
mirroring the cross-fold pooling in build_tabm_plot_data.py / build_deepens_plot_data.py.
Guards (climatology on the input, |anomaly| on the output) zero the correction on
non-physical rows, identical policy to the DE / TabM builds.

Each fold dir is expected to contain a FeaturePipeline pkl (``*pipeline*.pkl``) and
one model file:
    --model-kind linreg : model_ridge.joblib   (sklearn Ridge)
    --model-kind xgb    : model_mean_*.joblib   (XGBRegressor, objective mean)

Example (both surfaces, all folds):
    PYTHONPATH=src python workspace/build_baseline_plot_data.py \
        --model-kind linreg \
        --ocean-model-dir results/model_linear_baseline/linreg_ocean_full_prof_foldpca_r05_f* \
        --land-model-dir  results/model_linear_baseline/linreg_land_full_prof_foldpca_r15_f* \
        --input  results/csv_collection/combined_2018-11-29_all_orbits.parquet \
        --output .../linreg/combined_2018-11-29_bu/plot_data.parquet
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.pipeline import FeaturePipeline, _ensure_derived_features
from models.leakage_guard import check_training_overlap
from apply.apply_deep_ensemble import _check_required_columns, _domain_report

KEEP_COLS = ('time', 'lon', 'lat', 'cld_dist_km', 'sfc_type',
             'xco2_bc', 'xco2_bc_anomaly', 'xco2_apriori',
             'xco2_raw', 'xco2_raw_anomaly')

# Per-kind model-file glob inside each fold dir (pipeline is always *pipeline*.pkl).
MODEL_GLOB = {'linreg': 'model_ridge.joblib', 'xgb': 'model_mean_*.joblib'}


def _load_fold(model_dir, kind: str):
    """Load one baseline fold: (pipeline, sklearn/xgb model)."""
    md = Path(model_dir)
    pipes = sorted(md.glob('*pipeline*.pkl'))
    if not pipes:
        raise FileNotFoundError(f"no *pipeline*.pkl in {md}")
    pipe = FeaturePipeline.load(pipes[0])
    mfiles = sorted(md.glob(MODEL_GLOB[kind]))
    if not mfiles:
        raise FileNotFoundError(f"no {MODEL_GLOB[kind]} in {md}")
    model = joblib.load(mfiles[0])
    return pipe, model


def _predict_pooled(df, fold_dirs, sfc_type, kind, *, ood_thresh=8.0, max_ood_frac=0.02,
                    strict=False, tag='input'):
    """Cross-fold ensemble: pool every fold's point prediction (mean over folds)."""
    folds = [_load_fold(d, kind) for d in fold_dirs]
    pipe0, _ = folds[0]
    print(f"  [loaded] {len(folds)} {kind} fold(s), {pipe0.n_features} features")

    _check_required_columns(df, pipe0)
    df = df[df['sfc_type'] == sfc_type].copy()
    df = _ensure_derived_features(df)
    X0 = pipe0.transform(df)
    valid = np.all(np.isfinite(X0), axis=1)
    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        return None, df

    overall, worst = _domain_report(X0[valid], pipe0, thresh=ood_thresh)
    if overall > max_ood_frac:
        flagged = [f"{n}({f:.0%})" for n, f in worst[:5] if f > max_ood_frac]
        msg = (f"  ⚠ DOMAIN WARNING [{tag}]: {overall:.1%} of feature values OOD "
               f"(|z|>{ood_thresh:g}); worst: {flagged}")
        if strict:
            raise SystemExit(msg + "\n  (refused: --strict)")
        print(msg)
    else:
        print(f"  domain check [{tag}]: OK ({overall:.2%} OOD)")

    mu_stack = []
    for pipe, model in folds:
        Xi = pipe.transform(df)              # this fold's own scaler
        mu_stack.append(np.asarray(model.predict(Xi), dtype=float))
    mu = np.stack(mu_stack).mean(0)
    return mu.astype(np.float32), df


def _build_surface(df, fold_dirs, sfc_type, kind, corr_col, *, dk, clim_max_ppm=50.0,
                   max_abs_anomaly=25.0, base_col='xco2_bc', truth_col='xco2_bc_anomaly'):
    mu, kept = _predict_pooled(df, fold_dirs, sfc_type, kind, tag=f'sfc{sfc_type}', **dk)
    if mu is None or len(kept) == 0:
        print(f"  sfc={sfc_type}: no rows after filter — skipped")
        return None
    out = kept[[c for c in KEEP_COLS if c in kept.columns]].reset_index(drop=True).copy()
    mu = np.asarray(mu, dtype=float)

    clim_guard = np.zeros(len(out), dtype=bool)
    if 'xco2_apriori' in out.columns:
        diff = out[base_col].to_numpy(float) - out['xco2_apriori'].to_numpy(float)
        clim_guard = np.isfinite(diff) & (diff > clim_max_ppm)

    anomaly_guard = np.zeros(len(out), dtype=bool)
    if max_abs_anomaly is not None:
        anomaly_guard = np.isfinite(mu) & (np.abs(mu) > max_abs_anomaly)

    guard = clim_guard | anomaly_guard
    if guard.any():
        mu = mu.copy(); mu[guard] = 0.0
        print(f"  sfc={sfc_type}: guards skipped correction on {int(guard.sum())} sounding(s)")

    out['clim_guard'] = clim_guard
    out['anomaly_guard'] = anomaly_guard
    out['pred_anomaly'] = mu
    xb = out[base_col].to_numpy(dtype=float)
    out[corr_col] = xb - mu

    if truth_col in out.columns:
        y = out[truth_col].to_numpy(float)
        keep = ~guard
        pre = np.sqrt(np.nanmean(y[keep] ** 2))
        post = np.sqrt(np.nanmean((y[keep] - mu[keep]) ** 2))
        print(f"  sfc={sfc_type}: {int(keep.sum()):,}/{len(out):,} corrected  {truth_col} RMS "
              f"{pre:.3f} → {post:.3f} ppm ({100*(1-post/pre):+.1f}%)")
    return out


def _expand(dirs):
    """Expand any glob patterns the shell didn't (e.g. quoted globs)."""
    if not dirs:
        return dirs
    out = []
    for d in dirs:
        hits = sorted(glob.glob(d))
        out.extend(hits if hits else [d])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-kind', choices=('linreg', 'xgb'), required=True,
                    help="Which baseline: 'linreg' (Ridge) or 'xgb' (plain XGBoost-mean). "
                         "Sets the output column <kind>_corrected_xco2 and the model-file glob.")
    ap.add_argument('--ocean-model-dir', nargs='+', default=None,
                    help="Baseline fold dir(s) trained on ocean (sfc_type==0).")
    ap.add_argument('--land-model-dir', nargs='+', default=None,
                    help="Baseline fold dir(s) trained on land (sfc_type==1).")
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--ood-thresh', type=float, default=8.0)
    ap.add_argument('--max-ood-frac', type=float, default=0.02)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--climatology-max-ppm', type=float, default=50.0)
    ap.add_argument('--max-abs-anomaly', type=float, default=25.0)
    ap.add_argument('--correction-base', choices=('bc', 'raw'), default='bc')
    ap.add_argument('--allow-train-overlap', action='store_true',
                    help="Downgrade the training-date leakage guard from refusal to "
                         "a loud warning (in-sample diagnostics only).")
    args = ap.parse_args()
    kind = args.model_kind
    corr_col = f'{kind}_corrected_xco2'
    dk = dict(ood_thresh=args.ood_thresh, max_ood_frac=args.max_ood_frac, strict=args.strict)
    base_col = 'xco2_raw' if args.correction_base == 'raw' else 'xco2_bc'
    truth_col = 'xco2_raw_anomaly' if args.correction_base == 'raw' else 'xco2_bc_anomaly'

    surfaces = []
    if args.ocean_model_dir:
        surfaces.append((_expand(args.ocean_model_dir), 0))
    if args.land_model_dir:
        surfaces.append((_expand(args.land_model_dir), 1))
    if not surfaces:
        raise SystemExit("pass --ocean-model-dir and/or --land-model-dir")

    df = pd.concat([pd.read_parquet(p) for p in args.input], ignore_index=True)
    print(f"  read {len(df):,} rows from {len(args.input)} file(s)  [kind={kind}]")

    for md, sfc in surfaces:
        check_training_overlap(
            md, input_paths=args.input,
            times=df['time'].to_numpy() if 'time' in df.columns else None,
            allow=args.allow_train_overlap, tag=f'sfc{sfc}')

    max_abs_anom = args.max_abs_anomaly if args.max_abs_anomaly > 0 else None
    parts = [p for p in (_build_surface(df, md, sfc, kind, corr_col, dk=dk,
                                        clim_max_ppm=args.climatology_max_ppm,
                                        max_abs_anomaly=max_abs_anom,
                                        base_col=base_col, truth_col=truth_col)
                         for md, sfc in surfaces)
             if p is not None]
    if not parts:
        raise SystemExit("no soundings predicted on any surface")
    out = pd.concat(parts, ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"  [saved] {len(out):,} rows ({len(parts)} surface(s)) → {args.output}")


if __name__ == '__main__':
    main()
