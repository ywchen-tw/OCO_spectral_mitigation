"""xgb_models.py — XGBoost XCO2 bias correction model.

Trains an XGBRegressor on the same FeaturePipeline features as Ridge/MLP,
producing:
  - Built-in importance  (gain / cover / weight)
  - SHAP importance      (bar, beeswarm, dependence plots, heatmap)
  - Learning curve       (train vs eval RMSE vs boosting round)
  - Saved XGBoostAdapter (xgb_model.ubj + xgb_meta.pkl)

Usage:
    python src/xgb_models.py [--sfc_type 0|1] [--suffix <str>] [--pipeline <path>]
"""

import argparse
import gc
import json
import logging
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from .pipeline import FeaturePipeline, _ensure_derived_features
from .adapters import XGBoostAdapter
from utils import get_storage_dir

logger = logging.getLogger(__name__)


def _default_run_config() -> dict:
    return {
        'data': {
            'sfc_type': 0,
            'snow_flag_value': 0,
            'linux_data_name': 'combined_2016_2020_dates.parquet',
            'darwin_data_name': 'combined_2020-02-01_all_orbits.parquet',
        },
        'split': {
            'test_size': 0.2,
            'random_state': 42,
            'stratify_by_pc1': True,
        },
        'pipeline': {
            'pca_augment': False,
        },
        'objective': {
            'huber_delta': 1.0,
            'pred_l1_weight': 0.05,
        },
        'xgb': {
            'n_estimators': 12000,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 400,
            'gamma': 1.0,
            'reg_lambda': 5.0,
            'reg_alpha': 0.5,
            'tree_method': 'hist',
            'early_stopping_rounds': 100,
            'eval_metric': 'mae',
            'n_jobs': -1,
            'verbosity': 1,
            'eval_verbose': 50,
        },
        'shap': {
            'n_shap_full': 10000,
            'n_shap_viz': 2000,
        },
    }


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_run_config(config_path: str | None) -> dict:
    cfg = _default_run_config()
    if config_path is None:
        return cfg
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Run config not found: {config_path}\n"
            "Create the file first or run without --config to use defaults."
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise TypeError(f"Run config must be a JSON object, got {type(loaded)}")
    return _deep_update(cfg, loaded)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    """Return True if nvidia-smi is present and reports a GPU."""
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


# ─── Custom objective ───────────────────────────────────────────────────────────

def _build_huber_l1_pred_obj(delta: float, lam: float):
    def _huber_l1_pred_obj(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Custom XGBoost objective: Huber loss + λ·|ŷ| prediction shrinkage.

        XGBRegressor sklearn API calls custom objectives as fn(labels, preds).

        Mirrors the `+ 0.05 * pred.abs().mean()` term added to the MLP/Transformer
        training losses — pushing leaf weights toward zero when signal is weak so
        that small anomalies near zero are not over-predicted.

        Gradient = Huber'(ŷ − y) + λ·sign(ŷ)
        Hessian  = Huber''(ŷ − y)   (L1 hessian is 0 a.e.; omitted)
        """
        residual = y_pred - y_true
        abs_r = np.abs(residual)
        grad = np.where(abs_r <= delta, residual, delta * np.sign(residual))
        hess = np.where(abs_r <= delta, np.ones_like(residual),
                        delta / np.clip(abs_r, 1e-9, None))
        grad += lam * np.sign(y_pred)
        return grad, hess

    return _huber_l1_pred_obj


# ─── Built-in importance ────────────────────────────────────────────────────────

def plot_xgb_importance_builtin(model, features: list, output_dir: Path) -> None:
    """3-panel bar chart: gain / cover / weight (built-in XGBoost importance types).

    Gain   = avg loss reduction per split using the feature (best for ranking).
    Cover  = avg # training samples affected by splits on the feature.
    Weight = # times the feature is used in a split across all trees.
    """
    booster = model.get_booster()
    n_feat  = len(features)

    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, n_feat * 0.28)))
    panels = [
        (axes[0], 'gain',   'Gain (avg loss reduction per split)'),
        (axes[1], 'cover',  'Cover (avg # samples per split)'),
        (axes[2], 'weight', 'Weight (# times feature used in splits)'),
    ]
    for ax, imp_type, title in panels:
        scores = booster.get_score(importance_type=imp_type)
        # When trained on a numpy array (no feature names in DMatrix), XGBoost
        # uses 'f0','f1',... keys.  Remap to actual feature names by index.
        if scores:
            first = next(iter(scores))
            if first.startswith('f') and first[1:].isdigit():
                scores = {features[int(k[1:])]: v
                          for k, v in scores.items()
                          if int(k[1:]) < len(features)}
        vals   = np.array([scores.get(f, 0.0) for f in features])
        order  = np.argsort(vals)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_feat))
        ax.barh(range(n_feat), vals[order], color=colors)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels([features[i] for i in order], fontsize=7)
        ax.set_xlabel(imp_type)
        ax.set_title(title, fontsize=10)

    fig.suptitle('XGBoost Built-in Feature Importance', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / 'xgb_importance_builtin.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved xgb_importance_builtin.png")


# ─── SHAP importance ────────────────────────────────────────────────────────────

def plot_xgb_shap(model, X_test: np.ndarray, features: list, output_dir: Path,
                  n_shap_full: int = 10_000, n_shap_viz: int = 2_000,
                  display_features: list | None = None) -> None:
    """SHAP-based feature importance via TreeExplainer.

    Produces four outputs:
      xgb_shap_bar.png            — mean(|SHAP|) bar chart for all features
      xgb_shap_importance.csv     — mean_abs_shap + std per feature
      xgb_shap_beeswarm.png       — dot summary plot (top 30), shows direction
      xgb_shap_dependence_*.png   — dependence + interaction for top-5 features
      xgb_shap_heatmap.png        — per-sample heatmap for top-20 features

    Parameters
    ----------
    n_shap_full      : rows used for bar/CSV (full importance ranking)
    n_shap_viz       : rows used for beeswarm/heatmap (memory-bounded subset)
    display_features : if given, only these features appear in the plots/CSV;
                       SHAP values are still computed on the full feature set
                       (the model requires all columns).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plots.  Run: pip install shap")
        return

    feature_names = list(features)

    rng      = np.random.default_rng(42)
    n_full   = min(n_shap_full, len(X_test))
    n_viz    = min(n_shap_viz,  n_full)
    idx_full = rng.choice(len(X_test), size=n_full, replace=False)
    X_full   = np.ascontiguousarray(X_test[idx_full], dtype=np.float32)

    print(f"  Computing SHAP values (TreeExplainer, {n_full} samples) …", flush=True)
    explainer = shap.TreeExplainer(model)
    sv_full   = explainer.shap_values(X_full)   # [n_full, n_features]
    print("  SHAP values done.", flush=True)

    # Subset columns for display (e.g. drop fp_{i} dummy features).
    # The full X / sv arrays were used for computation; now filter for plots.
    if display_features is not None:
        _keep     = [i for i, f in enumerate(feature_names) if f in set(display_features)]
        sv_full   = sv_full[:, _keep]
        X_full    = X_full[:, _keep]
        feature_names = [feature_names[i] for i in _keep]

    n_feat = len(feature_names)
    sv_viz = sv_full[:n_viz]
    X_viz  = X_full[:n_viz]

    # ── 1. Bar chart + CSV ──────────────────────────────────────────────────
    mean_abs = np.abs(sv_full).mean(axis=0)
    std_abs  = np.abs(sv_full).std(axis=0)
    order    = np.argsort(mean_abs)

    fig_h   = max(6, n_feat * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, n_feat))
    ax.barh(range(n_feat), mean_abs[order],
            xerr=std_abs[order],
            error_kw=dict(ecolor='gray', capsize=2),
            color=colors)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=7)
    ax.set_xlabel('mean(|SHAP value|)  —  average impact on model output (ppm)')
    ax.set_title(f'XGBoost SHAP Feature Importance  ({n_full:,} samples)')
    plt.tight_layout()
    fig.savefig(output_dir / 'xgb_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved xgb_shap_bar.png")

    imp_df = pd.DataFrame({
        'feature':       feature_names,
        'mean_abs_shap': mean_abs,
        'std_abs_shap':  std_abs,
    }).sort_values('mean_abs_shap', ascending=False)
    imp_df.to_csv(output_dir / 'xgb_shap_importance.csv', index=False)
    logger.info("Saved xgb_shap_importance.csv")

    # Top-5 names for dependence plots (computed before beeswarm resets plt state)
    top5_names = [feature_names[i] for i in np.argsort(mean_abs)[::-1][:5]]

    # ── 2. Beeswarm (dot summary plot) ──────────────────────────────────────
    plt.figure(figsize=(10, max(6, min(30, n_feat) * 0.35)))
    shap.summary_plot(sv_viz, X_viz, feature_names=feature_names,
                      plot_type='dot', show=False, max_display=min(30, n_feat))
    plt.title(f'XGBoost SHAP Beeswarm  ({n_viz:,} samples, top 30 features)')
    plt.tight_layout()
    plt.savefig(output_dir / 'xgb_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    logger.info("Saved xgb_shap_beeswarm.png")

    # ── 3. Dependence plots: top-5 features ─────────────────────────────────
    # interaction_index='auto' lets SHAP auto-select the feature with the
    # strongest interaction (often cld_dist_km for spectral k-coefficients).
    for feat_name in top5_names:
        feat_idx  = feature_names.index(feat_name)
        fig, ax   = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat_idx, sv_viz, X_viz,
                             feature_names=feature_names,
                             interaction_index='auto',
                             show=False, ax=ax)
        fig.tight_layout()
        safe = feat_name.replace('/', '_').replace(' ', '_')
        fig.savefig(output_dir / f'xgb_shap_dependence_{safe}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved xgb_shap_dependence_%s.png", safe)

    # ── 4. Heatmap (per-sample SHAP for top-20 features) ────────────────────
    top_n_heat   = min(20, n_feat)
    top_heat_idx = np.argsort(mean_abs)[::-1][:top_n_heat]
    explanation  = shap.Explanation(
        values       = sv_viz[:, top_heat_idx],
        data         = X_viz[:, top_heat_idx],
        feature_names= [feature_names[i] for i in top_heat_idx],
    )
    plt.figure(figsize=(max(8, top_n_heat * 0.55), 8))
    shap.plots.heatmap(explanation, show=False)
    plt.title(f'SHAP Heatmap — top {top_n_heat} features  ({n_viz:,} samples)')
    plt.tight_layout()
    plt.savefig(output_dir / 'xgb_shap_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    logger.info("Saved xgb_shap_heatmap.png")


# ─── Learning curve ──────────────────────────────────────────────────────────────

def plot_xgb_learning_curve(model, output_dir: Path) -> None:
    """Val MAE from evals_result_ (written by XGBRegressor.fit).

    eval_set contains only X_test (no X_train) to avoid scoring 3M+ train
    rows every round; XGBoost labels the single entry 'validation_0'.
    """
    results     = model.evals_result()
    eval_metric = results['validation_0']['mae']
    best_round  = int(np.argmin(eval_metric))
    epochs      = range(1, len(eval_metric) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, eval_metric, label='Val MAE', color='tomato')
    ax.axvline(best_round + 1, color='gray', linestyle='--', linewidth=0.8,
               label=f'Best round {best_round + 1}')
    ax.set_xlabel('Boosting round')
    ax.set_ylabel('MAE (ppm)')
    ax.set_title('XGBoost Learning Curve — validation MAE (eval metric: MAE)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'xgb_learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved xgb_learning_curve.png")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XGBoost XCO2 bias correction')
    parser.add_argument('--sfc_type', type=int, default=None,
                        help='Surface type filter (0=ocean, 1=land; default: 0)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder appended to output base dir '
                             '(e.g. --suffix land_2016_2020). '
                             'Creates results/model_xgb/<suffix>/.')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a saved FeaturePipeline (.pkl). '
                             'If omitted, a new pipeline is fitted on the training data '
                             'and saved to <output_dir>/pipeline.pkl.')
    parser.add_argument('--pca-augment', dest='pca_augment', action='store_true',
                        help='Append selected PC scores after scaled features '
                             '(land: PC1/PC4/PC8; ocean: PC3/PC6).  '
                             'Note: --scaler pca_whitening is not supported for XGBoost '
                             '(collinearity immunity; SHAP interpretability would be lost).')
    parser.add_argument('--no-pca-augment', dest='pca_augment', action='store_false',
                        help='Disable PCA-score augmentation.')
    parser.set_defaults(pca_augment=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON config file that overrides data/model/training '
                             'hyperparameters. Unspecified keys keep defaults.')
    args = parser.parse_args()

    run_cfg = _load_run_config(args.config)
    if args.sfc_type is not None:
        run_cfg['data']['sfc_type'] = int(args.sfc_type)
    if args.pca_augment is not None:
        run_cfg['pipeline']['pca_augment'] = bool(args.pca_augment)
    run_cfg['pipeline']['pipeline_arg'] = args.pipeline

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    def _checkpoint(label: str) -> None:
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb  = rss_raw / (1024 * 1024) if platform.system() == 'Darwin' else rss_raw / 1024
        print(f"[MEM] {label:55s}  RSS={rss_mb:.0f} MB", flush=True)

    storage_dir = get_storage_dir()
    fdir        = storage_dir / 'results/csv_collection'
    if platform.system() == 'Linux':
        data_name = run_cfg['data']['linux_data_name']
    else:
        data_name = run_cfg['data']['darwin_data_name']

    base_dir   = storage_dir / 'results/model_xgb'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = output_dir / 'xgb_run_config.json'
    with open(run_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)
    print(f"Saved resolved run config → {run_cfg_path}", flush=True)

    sfc_type = int(run_cfg['data']['sfc_type'])
    logger.info("Surface type: %d (0=ocean, 1=land)", sfc_type)

    # ── Load data ──────────────────────────────────────────────────────────
    data_path = str(fdir / data_name)
    print(f"Loading data: {data_path}", flush=True)
    _checkpoint("before data load")
    df = (pd.read_parquet(data_path) if data_path.endswith('.parquet')
          else pd.read_csv(data_path))
    df = df[df['sfc_type'] == sfc_type]
    df = df[df['snow_flag'] == run_cfg['data']['snow_flag_value']]
    print(f"  Rows after filter: {len(df):,}", flush=True)
    _checkpoint("after data load + filter")

    # ── Pipeline: load or fit ──────────────────────────────────────────────
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=sfc_type,
                                       pca_augment=bool(run_cfg['pipeline']['pca_augment']))
        pipeline_path = output_dir / 'pipeline.pkl'
        pipeline.save(pipeline_path)
        print(f"  Pipeline fitted and saved → {pipeline_path}", flush=True)

    features = pipeline.features

    # ── Build feature matrix ───────────────────────────────────────────────
    df = _ensure_derived_features(df)
    missing_fp = [i for i in range(8) if f'fp_{i}' not in df.columns]
    if missing_fp:
        df = pd.concat(
            [df, pd.DataFrame(
                {f'fp_{i}': (df['fp'] == i).astype(np.float32) for i in missing_fp},
                index=df.index,
            )], axis=1,
        )

    valid_rows = ~df['xco2_bc_anomaly'].isna()
    X_all = pipeline.transform(df)
    X     = X_all[valid_rows.values].astype(np.float32)
    y     = df['xco2_bc_anomaly'][valid_rows].values.astype(np.float32)
    del X_all, df   # df is no longer needed; free before split
    gc.collect()
    _checkpoint("after pipeline.transform  (df freed)")

    # Drop rows with NaN/inf features before the split
    feat_finite = np.all(np.isfinite(X), axis=1)
    if not feat_finite.all():
        n_dropped = int((~feat_finite).sum())
        print(f"[warn] Dropping {n_dropped} rows with non-finite features", flush=True)
        X = X[feat_finite]
        y = y[feat_finite]

    print(f"X shape: {X.shape}  |  y shape: {y.shape}", flush=True)
    _pc1_col = getattr(pipeline, 'pc1_col_idx', None)
    if bool(run_cfg['split']['stratify_by_pc1']) and _pc1_col is not None:
        _pc1_vals  = X[:, _pc1_col]
        _pc1_strat = pd.qcut(_pc1_vals, q=5, labels=False, duplicates='drop')
        _stratify  = _pc1_strat
    else:
        _stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(run_cfg['split']['test_size']),
        random_state=int(run_cfg['split']['random_state']),
        stratify=_stratify,
    )
    del X, y
    gc.collect()
    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}", flush=True)
    _checkpoint("after train_test_split")

    # ── Train ──────────────────────────────────────────────────────────────
    import xgboost as xgb

    device = 'cuda' if _cuda_available() else 'cpu'
    print(f"  Device: {device}", flush=True)

    objective_fn = _build_huber_l1_pred_obj(
        delta=float(run_cfg['objective']['huber_delta']),
        lam=float(run_cfg['objective']['pred_l1_weight']),
    )

    model = xgb.XGBRegressor(
        n_estimators          = int(run_cfg['xgb']['n_estimators']),
        max_depth             = int(run_cfg['xgb']['max_depth']),
        learning_rate         = float(run_cfg['xgb']['learning_rate']),
        subsample             = float(run_cfg['xgb']['subsample']),
        colsample_bytree      = float(run_cfg['xgb']['colsample_bytree']),
        min_child_weight      = float(run_cfg['xgb']['min_child_weight']),
        gamma                 = float(run_cfg['xgb']['gamma']),
        reg_lambda            = float(run_cfg['xgb']['reg_lambda']),
        reg_alpha             = float(run_cfg['xgb']['reg_alpha']),
        objective             = objective_fn,
        tree_method           = run_cfg['xgb']['tree_method'],
        device                = device,
        early_stopping_rounds = int(run_cfg['xgb']['early_stopping_rounds']),
        eval_metric           = run_cfg['xgb']['eval_metric'],
        n_jobs                = int(run_cfg['xgb']['n_jobs']),
        verbosity             = int(run_cfg['xgb']['verbosity']),
    )

    print("Training XGBoost …", flush=True)
    # Only pass X_test to eval_set — scoring X_train (3M+ rows) every round
    # doubles per-round overhead without adding signal (early stopping uses val only).
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=int(run_cfg['xgb']['eval_verbose']),
    )
    del X_train, y_train   # no longer needed after fit
    gc.collect()
    _checkpoint("after XGBoost fit  (X_train freed)")

    y_pred  = model.predict(X_test)
    ss_res  = float(((y_test - y_pred) ** 2).sum())
    ss_tot  = float(((y_test - y_test.mean()) ** 2).sum())
    r2      = 1.0 - ss_res / ss_tot
    mae     = float(np.abs(y_test - y_pred).mean())
    print(f"\nTest R²: {r2:.4f}  |  MAE: {mae:.4f} ppm  "
          f"|  best_ntree_limit: {model.best_iteration}", flush=True)
    logger.info("Test R²=%.4f  MAE=%.4f  best_iteration=%d", r2, mae, model.best_iteration)
    if _pc1_col is not None:
        _pc1_test = X_test[:, _pc1_col]
        _bins = pd.qcut(_pc1_test, q=5, labels=False, duplicates='drop')
        _strat_rows = []
        for _q in range(5):
            _mask = _bins == _q
            if _mask.sum() < 10:
                continue
            _rmse = float(np.sqrt(np.mean((y_test[_mask] - y_pred[_mask]) ** 2)))
            _mae  = float(np.mean(np.abs(y_test[_mask] - y_pred[_mask])))
            _strat_rows.append({'model': 'XGB', 'pc1_quintile': _q + 1,
                                'n': int(_mask.sum()), 'rmse': _rmse, 'mae': _mae})
            print(f"  XGB PC1-Q{_q+1}: RMSE={_rmse:.4f}  MAE={_mae:.4f}  n={_mask.sum():,}")
        pd.DataFrame(_strat_rows).to_csv(output_dir / 'stratified_pc1_rmse_XGB.csv', index=False)

    # ── Save adapter ───────────────────────────────────────────────────────
    XGBoostAdapter(model, feature_names=features).save(output_dir)
    print(f"  XGBoostAdapter saved → {output_dir}", flush=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    # Exclude fp_{i} one-hot dummies from importance displays (8 dummy columns
    # split their aggregate signal and clutter the ranking).
    import re as _re
    _fp_pat      = _re.compile(r'^fp_\d+$')
    plot_features = [f for f in features if not _fp_pat.match(f)]

    plot_xgb_learning_curve(model, output_dir)
    plot_xgb_importance_builtin(model, plot_features, output_dir)
    plot_xgb_shap(
        model,
        X_test,
        features,
        output_dir,
        n_shap_full=int(run_cfg['shap']['n_shap_full']),
        n_shap_viz=int(run_cfg['shap']['n_shap_viz']),
        display_features=plot_features,
    )

    print(f"\nAll outputs written to {output_dir}", flush=True)


if __name__ == '__main__':
    main()
