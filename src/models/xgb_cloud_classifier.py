"""XGBoost binary classifier for near-cloud proximity (cld_dist_km <= near_cloud_km).

A SINGLE XGBoost model (no ensemble → cheap, CPU-only), the dedicated counterpart
to bundling the cloud task into the deep-ensemble multi-task head.  Supports
date_kfold so the OUT-OF-DISTRIBUTION (unseen-date) AUC can be measured: the local
12-date probe showed the in-distribution AUC (~0.99 land / 0.95 ocean) is largely
spatial/temporal-autocorrelation leakage and collapses to ~0.84 land / ~0.69 ocean
on unseen dates.  This module runs the honest test on the full 66-date set.

Defaults use the regularized config that generalized best OOD in the local sweep.

Example (CURC, one fold):
    PYTHONPATH=src python -m models.xgb_cloud_classifier --sfc_type 1 \\
        --suffix xgbcloud_land_f0 --val_split date_kfold --n_folds 5 --fold 0
"""
import argparse
import json
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score)

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers
from .splits import split_dataframe
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir


def _xy(pipeline, frame, near_km):
    """Return (X, y_near, n_dropped). y_near = 1 where cld_dist_km <= near_km.
    Rows are filtered to finite features + finite xco2_bc_anomaly so the held set
    matches the regression models' held set (apples-to-apples)."""
    X = pipeline.transform(frame)
    anom = frame['xco2_bc_anomaly'].to_numpy(dtype=float)
    cd = frame['cld_dist_km'].to_numpy(dtype=float)
    valid = np.isfinite(anom) & np.all(np.isfinite(X), axis=1)
    y = (np.isfinite(cd) & (cd <= near_km)).astype(np.int64)
    return X[valid].astype(np.float32), y[valid], int((~valid).sum())


def main():
    p = argparse.ArgumentParser(description="XGBoost near-cloud (<=km) binary classifier.")
    p.add_argument('--sfc_type', type=int, required=True, help="0=ocean, 1=land.")
    p.add_argument('--suffix', type=str, default='')
    p.add_argument('--val_split', type=str, default='random',
                   choices=['random', 'date', 'date_kfold'])
    p.add_argument('--n_folds', type=int, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--near_cloud_km', type=float, default=10.0)
    p.add_argument('--feature_set', type=str, default='full',
                   choices=['full', 'no_xco2', 'no_spec', 'full_fitqual'])
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    # XGBoost hyperparameters — defaults = the OOD-better regularized config.
    p.add_argument('--n_estimators', type=int, default=400)
    p.add_argument('--max_depth', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=0.03)
    p.add_argument('--min_child_weight', type=float, default=50.0)
    p.add_argument('--subsample', type=float, default=0.7)
    p.add_argument('--colsample_bytree', type=float, default=0.6)
    p.add_argument('--reg_lambda', type=float, default=5.0)
    p.add_argument('--reg_alpha', type=float, default=1.0)
    args = p.parse_args()

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = ('combined_2016_2020_dates.parquet' if platform.system() == "Linux"
                 else 'combined_2020_dates.parquet')
    output_dir = storage_dir / 'results/model_xgb_cloud' / args.suffix if args.suffix \
        else storage_dir / 'results/model_xgb_cloud'
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    run_id = args.suffix or datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    _dp = args.data if args.data else os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == args.sfc_type]
    df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)

    train_df, held_df = split_dataframe(df, mode=args.val_split, test_size=args.test_size,
                                        random_state=args.seed,
                                        n_folds=args.n_folds, fold=args.fold)

    pipeline = FeaturePipeline.fit(train_df, sfc_type=args.sfc_type, feature_set=args.feature_set)
    pipeline.save(output_dir / 'xgb_cloud_pipeline.pkl')

    X_tr, y_tr, _ = _xy(pipeline, train_df, args.near_cloud_km)
    X_te, y_te, _ = _xy(pipeline, held_df, args.near_cloud_km)
    pos = int(y_tr.sum()); neg = len(y_tr) - pos
    spw = neg / max(pos, 1)
    print(f"[xgb_cloud] sfc={args.sfc_type} train {X_tr.shape} held {X_te.shape}  "
          f"train pos_rate={y_tr.mean():.3f} held pos_rate={y_te.mean():.3f} "
          f"scale_pos_weight={spw:.3f}")

    # Early-stopping validation carved from train (in-distribution to train dates —
    # realistic, since you cannot early-stop on the future under a date split).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=args.seed, stratify=y_tr)

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, min_child_weight=args.min_child_weight,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
        scale_pos_weight=spw, tree_method='hist', eval_metric='auc',
        early_stopping_rounds=50, random_state=args.seed, n_jobs=-1)
    clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        'auc': float(roc_auc_score(y_te, proba)),
        'ap': float(average_precision_score(y_te, proba)),
        'recall@0.5': float(recall_score(y_te, pred, zero_division=0)),
        'precision@0.5': float(precision_score(y_te, pred, zero_division=0)),
        'f1@0.5': float(f1_score(y_te, pred, zero_division=0)),
        'held_pos_rate': float(y_te.mean()),
        'n_held': int(len(y_te)),
        'best_iteration': int(clf.best_iteration),
    }
    print(f"[xgb_cloud] {args.val_split}: AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} "
          f"recall@0.5={metrics['recall@0.5']:.4f} prec@0.5={metrics['precision@0.5']:.4f} "
          f"(trees={metrics['best_iteration']})")

    with open(output_dir / f'xgb_cloud_{args.val_split}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    clf.save_model(str(output_dir / 'xgb_cloud_model.json'))
    held_out = pd.DataFrame({'y_true': y_te, 'proba': proba})
    held_out.to_parquet(output_dir / 'xgb_cloud_held_predictions.parquet', index=False)
    with open(output_dir / 'xgb_cloud_meta.pkl', 'wb') as f:
        pickle.dump({'near_cloud_km': args.near_cloud_km, 'sfc_type': args.sfc_type,
                     'val_split': args.val_split, 'feature_set': args.feature_set}, f)

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='xgb_cloud',
        commit=commit, status='success',
        primary_metric_name='cloud_auc', primary_metric_value=metrics['auc'],
        secondary_metrics={k: metrics[k] for k in ('ap', 'recall@0.5', 'precision@0.5', 'f1@0.5')},
        runtime_seconds=float(time.monotonic() - run_start),
        description=f'XGBoost cloud<={args.near_cloud_km}km classifier, {args.val_split}-split, '
                    f'sfc={args.sfc_type}',
        artifacts={'output_dir': str(output_dir),
                   'metrics_json': str(output_dir / f'xgb_cloud_{args.val_split}_metrics.json')},
        config={'sfc_type': args.sfc_type, 'val_split': args.val_split,
                'near_cloud_km': args.near_cloud_km, 'feature_set': args.feature_set,
                'seed': args.seed, 'n_estimators': args.n_estimators,
                'max_depth': args.max_depth, 'learning_rate': args.learning_rate,
                'min_child_weight': args.min_child_weight, 'subsample': args.subsample,
                'colsample_bytree': args.colsample_bytree, 'reg_lambda': args.reg_lambda,
                'reg_alpha': args.reg_alpha},
    )
    with open(output_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
