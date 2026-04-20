import argparse
import h5py
import numpy as np
import pandas as pd
import os
import sys
import json
import traceback
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
import copy
import gc
from sklearn.model_selection import train_test_split
from pipeline import FeaturePipeline
from model_adapters import _ResBlock, _MLP, RidgeAdapter, MLPAdapter
from experiment_tracking import RunSummary, get_git_commit_hash, write_run_summary
import platform
import logging
from datetime import datetime
from tqdm import tqdm
from utils import get_storage_dir
from joblib import Parallel, delayed



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
        'ridge': {
            'alpha': 1.0,
        },
        'mlp': {
            'batch_size': 8192,
            'lr': 6e-4,
            'weight_decay': 1e-3,
            'max_epochs': 500,
            'patience': 30,
            'grad_clip_norm': 1.0,
            'cosine_t_max': 150,
            'cosine_eta_min': 1e-6,
            'val_batch_size': 8192,
            'infer_batch_size': 4096,
            'target_scale_eps': 1e-8,
        },
        'loss': {
            'huber_delta': 1.0,
            'pred_l1_weight': 0.05,
        },
        'importance': {
            'perm_max_rows': 5000,
            'perm_repeats': 5,
        },
        'seeds': {
            'plot_rng_seed': 1,
            'perm_rng_seed': 42,
            'perm_inner_rng_seed': 0,
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


# ─── Main analysis entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLP + Ridge XCO2 bias correction")
    parser.add_argument('--sfc_type', type=int, default=0,
                        help="Surface type filter for training and evaluation (default: 0 = ocean only).")
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder name appended to the base output directory '
                             '(e.g. --suffix v2_reduced).  '
                             'Creates results/model_mlp_lr/<suffix>/.')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a saved FeaturePipeline (.pkl).  '
                             'If omitted, a new pipeline is fitted on the training data '
                             'and saved to <output_dir>/pipeline.pkl.')
    parser.add_argument('--scaler', default='robust_standard',
                        choices=['robust_standard', 'pca_whitening'],
                        help='Scaler type: robust_standard (default) or pca_whitening '
                             '(RobustScaler → PCA(whiten=True)).')
    parser.add_argument('--pca-augment', action='store_true',
                        help='Append selected PC scores after scaled features '
                             '(land: PC1/PC4/PC8; ocean: PC3/PC6).')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON config file that overrides training '
                             'and model hyperparameters. Unspecified keys keep defaults.')
    args = parser.parse_args()

    run_cfg = _load_run_config(args.config)
    run_cfg['data']['sfc_type'] = int(args.sfc_type)
    run_cfg['pipeline'] = {
        'scaler': args.scaler,
        'pca_augment': bool(args.pca_augment),
        'pipeline_arg': args.pipeline,
    }

    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results/csv_collection'
    if platform.system() == "Linux":
        data_name = run_cfg['data']['linux_data_name']
    elif platform.system() == "Darwin":
        data_name = run_cfg['data']['darwin_data_name']
    base_dir   = storage_dir / 'results/model_mlp_lr'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = storage_dir / 'results' / 'autoresearch_ledger.tsv'
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    run_cfg_path = output_dir / 'mlp_run_config.json'
    with open(run_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)
    print(f"  Saved resolved run config → {run_cfg_path}", flush=True)

    sfc_type = run_cfg['data']['sfc_type']
    logger.info(f"Surface type filter: {sfc_type} (0=ocean only, 1=land only, 2=sea-ice only)")

    data_path = os.path.join(fdir, data_name)
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    df = df[df['sfc_type'] == sfc_type]
    df = df[df['snow_flag'] == run_cfg['data']['snow_flag_value']]

    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=sfc_type,
                                       scaler=args.scaler,
                                       pca_augment=args.pca_augment)
        pipeline_path = output_dir / 'pipeline.pkl'
        pipeline.save(pipeline_path)
        print(f"  Fitted and saved pipeline → {pipeline_path}", flush=True)

    try:
        mitigation_test(
            df,
            output_dir=output_dir,
            pipeline=pipeline,
            test_csv=None,
            perm_max_rows=int(run_cfg['importance']['perm_max_rows']),
            run_cfg=run_cfg,
            ledger_path=ledger_path,
            run_id=run_id,
            commit=commit,
            script_name=os.path.basename(__file__),
        )
    except Exception:
        crash_summary = RunSummary(
            run_id=run_id,
            script_name=os.path.basename(__file__),
            model_family='mlp_lr',
            commit=commit,
            status='crash',
            primary_metric_name='mlp_test_rmse',
            primary_metric_value=0.0,
            secondary_metrics={'traceback': traceback.format_exc()},
            peak_memory_mb=0.0,
            runtime_seconds=0.0,
            description='MLP+Ridge run crashed',
            artifacts={
                'output_dir': str(output_dir),
                'run_config': str(run_cfg_path),
            },
            config=run_cfg,
        )
        write_run_summary(crash_summary, output_dir=output_dir, ledger_path=ledger_path)
        raise

def _compute_anomaly_group(idx_list, fp_lat, cld_dist_km, xco2,
                           lat_thres, std_thres, min_cld_dist, chunk_size):
    """Compute XCO2 anomaly for a single (date, orbit_id) group.

    Top-level function so joblib can pickle it for parallel dispatch.

    Returns
    -------
    (idx, anomaly_vals) : indices into the original array and their anomaly values
    """
    idx    = np.array(idx_list)
    g_lat  = fp_lat[idx]
    g_dist = cld_dist_km[idx]
    g_xco2 = xco2[idx]
    M      = len(idx)

    valid_lat  = ~np.isnan(g_lat)
    clear_mask = valid_lat & (g_dist > min_cld_dist)

    ref_lat  = np.where(clear_mask, g_lat,  np.nan)
    ref_xco2 = np.where(clear_mask, g_xco2, np.nan)

    g_ref_mean = np.full(M, np.nan)
    g_ref_std  = np.full(M, np.nan)

    for start in range(0, M, chunk_size):
        end   = min(start + chunk_size, M)
        q_lat = g_lat[start:end]

        lat_diff  = np.abs(q_lat[:, None] - ref_lat[None, :])
        in_window = lat_diff <= lat_thres
        xco2_win  = np.where(in_window, ref_xco2[None, :], np.nan)
        has_refs  = in_window.any(axis=1)

        c_mean = np.full(end - start, np.nan)
        c_std  = np.full(end - start, np.nan)
        if has_refs.any():
            c_mean[has_refs] = np.nanmean(xco2_win[has_refs], axis=1)
            c_std[has_refs]  = np.nanstd( xco2_win[has_refs], axis=1)

        g_ref_mean[start:end] = c_mean
        g_ref_std[start:end]  = c_std
        del lat_diff, in_window, xco2_win

    valid = valid_lat & ~np.isnan(g_ref_mean) & (g_ref_std <= std_thres)
    return idx[valid], g_xco2[valid] - g_ref_mean[valid]


def compute_xco2_anomaly_date_id(fp_date, fp_orbit_id, fp_lat, cld_dist_km, xco2,
                         lat_thres=0.5, std_thres=2.0, min_cld_dist=10.0,
                         chunk_size=512, n_jobs=-1):
    """XCO2 anomaly relative to nearby clear-sky soundings within the same orbit.

    Groups footprints by (date, orbit_id) and processes each group
    independently, so the pairwise broadcast is O(M²) per orbit rather than
    O(N²) over the full dataset.  Within each group every footprint already
    shares the same date and orbit_id, so no cross-group comparisons are made.

    Parameters
    ----------
    fp_date      : [N] footprint date (string or any equality-comparable type)
    fp_orbit_id  : [N] orbit ID (int, str, or float)
    fp_lat       : [N] footprint latitudes (may contain NaN)
    cld_dist_km  : [N] nearest-cloud distance in km (may contain NaN)
    xco2         : [N] XCO2 values (may contain NaN)
    lat_thres    : float, half-width of latitude search window (degrees)
    std_thres    : float, maximum allowed std of reference XCO2 (ppm)
    min_cld_dist : float, minimum cloud distance for clear-sky reference (km)
    chunk_size   : int, query rows per iteration within each group.
                   Each group typically has O(1 000) rows, so the default 512
                   usually means 1-2 iterations per group with small arrays.
    n_jobs       : int, number of parallel workers (default: -1 = all cores).

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy
    """
    fp_date     = np.asarray(fp_date)
    fp_orbit_id = np.asarray(fp_orbit_id)
    fp_lat      = np.asarray(fp_lat,      dtype=float)
    xco2        = np.asarray(xco2,        dtype=float)
    cld_dist_km = np.asarray(cld_dist_km, dtype=float)

    chunk_size = int(chunk_size)
    anomaly    = np.full(len(fp_lat), np.nan)

    # Build index groups: {(date, orbit_id): [row indices]}
    groups: dict = {}
    for i, key in enumerate(zip(fp_date, fp_orbit_id)):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_compute_anomaly_group)(
            idx_list, fp_lat, cld_dist_km, xco2,
            lat_thres, std_thres, min_cld_dist, chunk_size
        )
        for idx_list in groups.values()
    )

    for valid_idx, valid_vals in results:
        anomaly[valid_idx] = valid_vals

    return anomaly

# ─── LR mitigation test ────────────────────────────────────────────────────────

def mitigation_test(df, output_dir, pipeline: FeaturePipeline, test_csv=None,
                    perm_max_rows: int = 5000, run_cfg: dict | None = None,
                    ledger_path=None, run_id: str | None = None,
                    commit: str | None = None, script_name: str | None = None):
    """Train a per-footprint linear regression to predict XCO2 bias from kappas.

    Parameters
    ----------
    sat          : dict from preprocess()
    df           : DataFrame from k1k2_analysis
    output_dir   : str
    test_csv: str or None
    """
    
    import resource as _res
    run_start = datetime.now()
    _t0 = datetime.now()
    def _checkpoint(label):
        rss_raw = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
        # macOS: ru_maxrss is bytes; Linux: kilobytes
        rss_mb = rss_raw / (1024 * 1024) if platform.system() == "Darwin" else rss_raw / 1024
        elapsed = (datetime.now() - _t0).total_seconds()
        print(f"[MEM] {label:55s}  RSS={rss_mb:.0f} MB  t={elapsed:7.1f}s", flush=True)

    _checkpoint("mitigation_test: entry")

    # Max points per scatter plot — keeps matplotlib PathCollection memory bounded on large datasets
    _SCATTER_MAX = 150_000

    def _scatter_ss(rng, *arrs):
        """Subsample all arrays to ≤ _SCATTER_MAX rows using the same random index."""
        N = len(arrs[0])
        if N <= _SCATTER_MAX:
            return arrs
        idx = rng.choice(N, size=_SCATTER_MAX, replace=False)
        return tuple(a[idx] for a in arrs)

    cfg = _default_run_config() if run_cfg is None else run_cfg
    rng_plot = np.random.default_rng(int(cfg['seeds']['plot_rng_seed']))

    # Keep only the 3 columns needed downstream — avoids copying all df columns for the filtered subset
    df_xco2_anomaly = df.loc[df['xco2_bc_anomaly'].notna(), ['xco2_bc', 'xco2_bc_anomaly', 'cld_dist_km']]

    _checkpoint("before pipeline.transform + train_test_split")
    features    = pipeline.features
    # pipeline.transform() already returns float32; copy=False is a no-op if dtype already matches
    X_all       = pipeline.transform(df).astype(np.float32, copy=False)
    valid_rows  = ~df['xco2_bc_anomaly'].isna()
    # df_X is the single valid-row subset used for both training and later inference
    df_X        = X_all[valid_rows.values]
    df_all_X    = X_all
    del X_all   # df_all_X holds the only reference; free the alias
    # Drop feature columns from df — only plot/anomaly columns are needed hereafter.
    # This frees ~25-35 feature columns (~50% of df's memory) before train_test_split.
    _keep_cols = ['lon', 'lat', 'xco2_bc', 'xco2_raw', 'xco2_bc_anomaly',
                  'date', 'orbit_id', 'cld_dist_km', 'o2a_k1', 'o2a_k2']
    df = df[[c for c in _keep_cols if c in df.columns]]
    gc.collect()
    y           = df['xco2_bc_anomaly'][valid_rows].values

    # Drop rows with NaN/inf features for training only — df_X stays intact for inference
    # (inference code below already applies its own isfinite mask over df_X)
    feat_finite = np.all(np.isfinite(df_X), axis=1)
    if not feat_finite.all():
        n_dropped = int((~feat_finite).sum())
        print(f"[warn] Dropping {n_dropped} rows with non-finite features before training")
        X_for_train = df_X[feat_finite]
        y_for_train = y[feat_finite]
    else:
        X_for_train = df_X
        y_for_train = y

    print("X shape (train-eligible):", X_for_train.shape)
    print("y shape (train-eligible):", y_for_train.shape)
    _pc1_col = getattr(pipeline, 'pc1_col_idx', None)
    stratify_by_pc1 = bool(cfg['split'].get('stratify_by_pc1', True))
    if _pc1_col is not None and stratify_by_pc1:
        import pandas as _pd_strat
        _pc1_vals  = X_for_train[:, _pc1_col]
        _pc1_strat = _pd_strat.qcut(_pc1_vals, q=5, labels=False, duplicates='drop')
        _stratify  = _pc1_strat
    else:
        _stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X_for_train,
        y_for_train,
        test_size=float(cfg['split']['test_size']),
        random_state=int(cfg['split']['random_state']),
        stratify=_stratify,
    )
    X_test_eval = X_test.copy()
    y_test_eval = y_test.copy()
    del X_for_train, y_for_train
    del y  # y_train + y_test contain all labels; drop the combined copy
    gc.collect()
    print("X_train shape", X_train.shape)
    _checkpoint("after  pipeline.transform + train_test_split")

    # ── Feature correlation matrix ─────────────────────────────────────────
    corr = np.corrcoef(X_train, rowvar=False)
    fig_corr, ax_corr = plt.subplots(figsize=(max(10, len(features) * 0.32),
                                               max(10, len(features) * 0.32)))
    im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax_corr.set_xticks(np.arange(len(features)))
    ax_corr.set_yticks(np.arange(len(features)))
    ax_corr.set_xticklabels(features, rotation=90, fontsize=7)
    ax_corr.set_yticklabels(features, fontsize=7)
    fig_corr.colorbar(im, ax=ax_corr, label='Pearson r')
    ax_corr.set_title('Feature correlation matrix (train set, quantile-transformed)')
    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(output_dir, 'feature_correlation_matrix.png'),
                     dpi=150, bbox_inches='tight')
    plt.close(fig_corr)
    del corr  # free [F, F] float64 matrix
    # ── end correlation matrix ─────────────────────────────────────────────

    # df_X / df_all_X already computed above; checkpoint retained for timing
    _checkpoint("df_X / df_all_X already ready (no-op)")

    # Extract all needed numpy arrays; free df_xco2_anomaly immediately after
    df_xco2_bc             = df_xco2_anomaly['xco2_bc'].to_numpy()
    df_xco2_bc_anomaly     = df_xco2_anomaly['xco2_bc_anomaly'].to_numpy()
    _cd_orig_for_masks     = df_xco2_anomaly['cld_dist_km'].to_numpy()  # for row masks below
    del df_xco2_anomaly
    df_all_xco2_bc         = df['xco2_bc'].to_numpy()
    df_all_xco2_bc_anomaly = df['xco2_bc_anomaly'].to_numpy()

    # Determine training data and clear-sky reference level
    if test_csv is None:
        test_df = df  
    else:
        test_df = (pd.read_parquet(test_csv) if str(test_csv).endswith('.parquet')
                   else pd.read_csv(test_csv))
        test_df = test_df[test_df['sfc_type_lt'] == 0]  # Ocean only for now
        # onehot encode the fp_number in the training DataFrame as well
        for i in range(8):
            test_df[f'fp_{i}'] = (test_df['fp'] == i).astype(int)

    
    # LR correction arrays (including those not in the training set)
    xco2_bc_pred_anomaly = np.full(df_xco2_bc.shape[0], np.nan)
    xco2_bc_corrected       = np.full(df_xco2_bc.shape[0], np.nan)

    # MLP parallel correction arrays
    xco2_bc_pred_anomaly_mlp  = np.full(df_xco2_bc.shape[0], np.nan)
    xco2_bc_corrected_mlp     = np.full(df_xco2_bc.shape[0], np.nan)
    
    # LR correction arrays for all data (including those without valid XCO2 anomaly for training)
    xco2_bc_predict_all_anomaly = np.full(df_all_xco2_bc.shape[0], np.nan)
    xco2_bc_corrected_all      = np.full(df_all_xco2_bc.shape[0], np.nan)
    
    # MLP correction arrays for all data (including those without valid XCO2 anomaly for training)
    xco2_bc_predict_all_anomaly_mlp = np.full(df_all_xco2_bc.shape[0], np.nan)
    xco2_bc_corrected_all_mlp      = np.full(df_all_xco2_bc.shape[0], np.nan)

    print("y_train", y_train.shape, "X_train shape:", X_train.shape)
    print("y_test", y_test.shape, "X_test shape:", X_test.shape)

    # ── PC1-regime RMSE helper ─────────────────────────────────────
    def _pc1_stratified_rmse(y_true, y_pred, pc1_vals, n_bins=5, label='model', out_dir=None):
        """Print RMSE/MAE per PC1 quintile; optionally save CSV."""
        import pandas as _pd_rmse
        bins = _pd_rmse.qcut(pc1_vals, q=n_bins, labels=False, duplicates='drop')
        rows = []
        for q in range(n_bins):
            mask = bins == q
            if mask.sum() < 10:
                continue
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
            mae  = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            rows.append({'model': label, 'pc1_quintile': q + 1,
                         'n': int(mask.sum()), 'rmse': rmse, 'mae': mae})
            print(f"  {label} PC1-Q{q+1}: RMSE={rmse:.4f}  MAE={mae:.4f}  n={mask.sum():,}")
        if rows and out_dir is not None:
            _pd_rmse.DataFrame(rows).to_csv(
                out_dir / f'stratified_pc1_rmse_{label}.csv', index=False
            )

    # ── Linear baseline ────────────────────────────────────────────
    _checkpoint("before LinearRegression fit")
    model = Ridge(alpha=float(cfg['ridge']['alpha']))
    model.fit(X_train, y_train)
    print(f"Test FP R² (linear) for predicting xco2_bc anomaly: {model.score(X_test, y_test):.3f}")
    _pc1_col_eval = getattr(pipeline, 'pc1_col_idx', None)
    if _pc1_col_eval is not None:
        _pc1_stratified_rmse(y_test, model.predict(X_test),
                             X_test[:, _pc1_col_eval], label='Ridge', out_dir=output_dir)

    df_X_mask     = np.all(np.isfinite(df_X), axis=1)
    y_pred_df   = model.predict(df_X[df_X_mask])
    xco2_bc_pred_anomaly[df_X_mask] = y_pred_df
    xco2_bc_corrected[df_X_mask] = df_xco2_bc[df_X_mask] - y_pred_df
    
    df_all_X_mask = np.all(np.isfinite(df_all_X), axis=1)
    y_pred_all_df = model.predict(df_all_X[df_all_X_mask])
    xco2_bc_predict_all_anomaly[df_all_X_mask] = y_pred_all_df
    xco2_bc_corrected_all[df_all_X_mask]       = df_all_xco2_bc[df_all_X_mask] - y_pred_all_df

    # ── PyTorch MLP ────────────────────────────────────────────────
    _checkpoint("before MLP init + training")
    n_in = X_train.shape[1]

    # Normalise target: median-centre + IQR-scale (robust to heavy left tail
    # from cloud-affected anomalies; IQR/1.349 ≈ σ for a Gaussian distribution)
    # 1.3490 = IQR of N(0,1) = 2 × Φ⁻¹(0.75) ≈ 2 × 0.6745, so IQR/1.3490 ≈ σ
    y_mean = float(np.median(y_train))
    y_std  = float(np.percentile(y_train, 75) - np.percentile(y_train, 25)) / 1.3490 + float(cfg['mlp']['target_scale_eps'])
    y_train_n = (y_train - y_mean) / y_std
    y_test_n  = (y_test  - y_mean) / y_std

    # ── Device selection (shared by both checkpoint-load and training paths) ──
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"  Device: {device}", flush=True)

    train_ds = train_loader = best_state = None  # initialised here; set inside else-branch if training runs

    loss_huber_delta = float(cfg['loss']['huber_delta'])
    loss_pred_l1_weight = float(cfg['loss']['pred_l1_weight'])

    def _loss(pred, by):
        return nn.functional.huber_loss(pred, by, delta=loss_huber_delta) + loss_pred_l1_weight * pred.abs().mean()

    if RidgeAdapter.can_load(output_dir) and MLPAdapter.can_load(output_dir):
        # ── Load from checkpoint, skip training ───────────────────────────
        print(f"  [checkpoint] Found existing adapter checkpoints → {output_dir}", flush=True)
        ridge_adapter = RidgeAdapter.load(output_dir)
        model         = ridge_adapter.model
        mlp_adapter   = MLPAdapter.load(output_dir, device=device)
        mlp    = mlp_adapter.model
        y_mean = mlp_adapter.y_mean
        y_std  = mlp_adapter.y_std
        print("  [checkpoint] Loaded. Skipping training.", flush=True)
        _mlp_from_ckpt = True
    else:
        _mlp_from_ckpt = False
        train_ds   = torch.utils.data.TensorDataset(
            torch.tensor(X_train,   dtype=torch.float32),
            torch.tensor(y_train_n, dtype=torch.float32),
        )
        del y_train_n  # data copied into train_ds tensor; free the numpy array
        # Larger batch size for full dataset efficiency; 512 is fine for the local subset too
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=int(cfg['mlp']['batch_size']),
            shuffle=True,
        )

        # Val loss uses batched inference to keep peak memory bounded on the 76K-sample test set
        def _val_loss(model_):
            model_.eval()
            total, n = 0.0, 0
            val_batch_size = int(cfg['mlp']['val_batch_size'])
            with torch.no_grad():
                for start in range(0, len(X_test), val_batch_size):
                    Xb = torch.tensor(X_test[start:start + val_batch_size], dtype=torch.float32).to(device)
                    yb = torch.tensor(y_test_n[start:start + val_batch_size], dtype=torch.float32).to(device)
                    pred = model_(Xb)
                    total += _loss(pred, yb).item() * len(Xb)
                    n += len(Xb)
                    del Xb, yb
            return total / n

        mlp      = _MLP(n_in).to(device)
        n_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print(f"  MLP parameters: {n_params:,}  |  train samples: {len(X_train):,}  "
              f"|  ratio: {len(X_train)/n_params:.2f}", flush=True)

        optimizer = torch.optim.AdamW(
            mlp.parameters(),
            lr=float(cfg['mlp']['lr']),
            weight_decay=float(cfg['mlp']['weight_decay']),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg['mlp']['cosine_t_max']),
            eta_min=float(cfg['mlp']['cosine_eta_min']),
        )

        train_losses, val_losses = [], []
        best_val_loss, best_state, patience, no_improve = float('inf'), None, int(cfg['mlp']['patience']), 0
        epoch_bar = tqdm(range(int(cfg['mlp']['max_epochs'])), desc="MLP training", unit="epoch")
        for epoch in epoch_bar:
            mlp.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = mlp(bx)
                loss = _loss(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=float(cfg['mlp']['grad_clip_norm']))
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            val_loss = _val_loss(mlp)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(mlp.state_dict())
                no_improve    = 0
            else:
                no_improve += 1

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                patience=f"{no_improve}/{patience}",
                saved="✓" if improved else "",
            )

            if no_improve >= patience:
                tqdm.write(f"  Early stop at epoch {epoch} (best val={best_val_loss:.4f})")
                break

        # ── Learning curve: train vs val loss ──────────────────────────
        fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
        epochs_ran = range(1, len(train_losses) + 1)
        ax_lc.plot(epochs_ran, train_losses, label='Train (Huber + L1 reg)', color='steelblue')
        ax_lc.plot(epochs_ran, val_losses,   label='Val   (Huber + L1 reg)', color='tomato')
        best_ep = int(np.argmin(val_losses)) + 1
        ax_lc.axvline(best_ep, color='gray', linestyle='--', linewidth=0.8,
                      label=f'Best val epoch {best_ep}')
        ax_lc.set_xlabel('Epoch')
        ax_lc.set_ylabel('Huber + L1 reg loss (normalised target)')
        ax_lc.set_title('MLP Learning Curve — train vs validation')
        ax_lc.legend()
        fig_lc.tight_layout()
        fig_lc.savefig(os.path.join(output_dir, 'mlp_learning_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_lc)
        gap = val_losses[best_ep - 1] - train_losses[best_ep - 1]
        print(f"  Best epoch {best_ep}: train={train_losses[best_ep-1]:.4f}  "
              f"val={val_losses[best_ep-1]:.4f}  gap={gap:.4f}", flush=True)

        mlp.load_state_dict(best_state)
        mlp.eval()

        # ── Save model adapters ──────────────────────────────────────────────
        RidgeAdapter(model).save(output_dir)
        MLPAdapter(mlp, y_mean, y_std, device=device).save(output_dir)
        mlp.to(device)   # restore to device for inference below (MLPAdapter.save() moves to CPU)
        print(f"  Saved adapters → {output_dir}", flush=True)
        # ── end save ─────────────────────────────────────────────────────────

    # Free training allocations before any full-dataset inference
    if not _mlp_from_ckpt:
        del train_ds, train_loader, best_state
    gc.collect()
    _checkpoint("after MLP training  (post-gc)")

    def _mlp_infer(X_np, batch_size=None):
        """Batched inference; returns predictions in original (ppm) units."""
        if batch_size is None:
            batch_size = int(cfg['mlp']['infer_batch_size'])
        out = []
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                out.append(mlp(Xb).cpu().numpy())
            del Xb
        return np.concatenate(out) * y_std + y_mean   # denormalise

    _checkpoint("MLP infer: X_test")
    y_all_mlp = _mlp_infer(X_test)
    ss_res = ((y_test - y_all_mlp) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    print(f"Test FP R² (MLP)    for predicting xco2_bc anomaly: {1 - ss_res/ss_tot:.3f}", flush=True)
    if _pc1_col_eval is not None:
        _pc1_stratified_rmse(y_test, y_all_mlp,
                             X_test[:, _pc1_col_eval], label='MLP', out_dir=output_dir)

    _checkpoint("MLP infer: df_X (anomaly-valid subset)")
    y_pred_mlp = _mlp_infer(df_X[df_X_mask])
    xco2_bc_pred_anomaly_mlp[df_X_mask] = y_pred_mlp
    xco2_bc_corrected_mlp[df_X_mask]    = df_xco2_bc[df_X_mask] - y_pred_mlp

    _checkpoint("MLP infer: df_all_X (full dataset)")
    y_all_pred_mlp = _mlp_infer(df_all_X[df_all_X_mask])
    xco2_bc_predict_all_anomaly_mlp[df_all_X_mask] = y_all_pred_mlp
    xco2_bc_corrected_all_mlp[df_all_X_mask]       = df_all_xco2_bc[df_all_X_mask] - y_all_pred_mlp
    _checkpoint("MLP infer: complete")
    # df_X / df_all_X are no longer needed — free before permutation importance
    del df_X, df_all_X, df_X_mask, df_all_X_mask
    gc.collect()
    _checkpoint("after del df_X/df_all_X")
    # ── end MLP ────────────────────────────────────────────────────

    # ── Feature importance (LR + MLP permutation) ─────────────────
    y_label = 'xco2_bc_anomaly'

    # LR: standardised absolute coefficients  |coef_i * std(X_train_i)|
    lr_std_importance = np.abs(model.coef_) * X_train.std(axis=0)
    del X_train  # no longer needed after std computation
    gc.collect()

    # Permutation importance — subsample to cap memory/compute (perm_max_rows rows is enough for stable estimates)
    rng_pi   = np.random.default_rng(int(cfg['seeds']['perm_rng_seed']))
    pi_n     = min(perm_max_rows, X_test.shape[0])
    pi_idx   = rng_pi.choice(X_test.shape[0], size=pi_n, replace=False)
    X_pi     = X_test[pi_idx]
    y_pi     = y_test[pi_idx]

    def _permutation_importance(predict_fn, X_eval, y_eval, n_repeats=5, skip_cols=frozenset()):
        """Return mean R² drop per feature when that feature is shuffled.
        Columns in ``skip_cols`` (by index) are left at 0.0 importance.
        """
        ss_tot      = ((y_eval - y_eval.mean()) ** 2).sum()
        baseline_r2 = 1.0 - ((y_eval - predict_fn(X_eval)) ** 2).sum() / ss_tot
        importances = np.zeros(X_eval.shape[1])
        rng_inner   = np.random.default_rng(int(cfg['seeds']['perm_inner_rng_seed']))
        for col in range(X_eval.shape[1]):
            if col in skip_cols:
                continue  # skip one-hot footprint columns
            drops = np.zeros(n_repeats)
            for r in range(n_repeats):
                X_shuf = X_eval.copy()
                X_shuf[:, col] = rng_inner.permutation(X_shuf[:, col])
                r2_shuf = 1.0 - ((y_eval - predict_fn(X_shuf)) ** 2).sum() / ss_tot
                drops[r] = baseline_r2 - r2_shuf
                del X_shuf
            importances[col] = drops.mean()
        return importances

    fp_skip_cols = frozenset(
        i for i, f in enumerate(features)
        if f.startswith('fp_') and f[3:].isdigit()
    )
    _checkpoint("permutation importance: LR")
    perm_repeats = int(cfg['importance']['perm_repeats'])
    perm_imp_lr = _permutation_importance(model.predict, X_pi, y_pi, n_repeats=perm_repeats, skip_cols=fp_skip_cols)
    _checkpoint("permutation importance: MLP")
    perm_imp_mlp = _permutation_importance(_mlp_infer, X_pi, y_pi, n_repeats=perm_repeats, skip_cols=fp_skip_cols)
    _checkpoint("permutation importance: complete")

    # Save importance to CSV
    imp_df = pd.DataFrame({
        'feature': features,
        'lr_std_coef': lr_std_importance,
        'lr_perm_importance': perm_imp_lr,
        'mlp_perm_importance': perm_imp_mlp,
    })
    imp_df = imp_df.sort_values('mlp_perm_importance', ascending=False)
    imp_df.to_csv(os.path.join(output_dir, f'feature_importance_{y_label}.csv'), index=False)

    # Bar plot: top 25 features by permutation importance
    top_n = min(25, len(features))
    top_df = imp_df.head(top_n).iloc[::-1]  # reverse for horizontal bar (top at top)

    plt.close('all')
    fig, (ax_lr, ax_mlp) = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.32)))

    ax_lr.barh(top_df['feature'], top_df['lr_perm_importance'], color='steelblue', label='LR perm.')
    ax_lr.set_xlabel('Permutation importance (R² drop)')
    ax_lr.set_title(f'LR — {y_label}')

    ax_mlp.barh(top_df['feature'], top_df['mlp_perm_importance'], color='forestgreen', label='MLP perm.')
    ax_mlp.set_xlabel('Permutation importance (R² drop)')
    ax_mlp.set_title(f'MLP — {y_label}')

    fig.suptitle(f'Feature importance (top {top_n})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'feature_importance_{y_label}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    # ── end feature importance ────────────────────────────────────

    # _val_loss is only defined in the training branch; guard before deletion
    if not _mlp_from_ckpt:
        del _val_loss
    del _loss
    del _mlp_infer, mlp
    del X_pi, y_pi
    del y_train, y_test, X_test
    gc.collect()
    _checkpoint("after feature importance gc")

    # ── Recompute XCO2 anomaly on corrected fields ─────────────────────────
    # Lazy import avoids circular import (oco_fp_spec_anal ↔ result_ana)
    # Same parameters as used in oco_fp_spec_anal.py
    _anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
    _date     = df['date']
    _orbit_id = df['orbit_id']
    _lat      = df['lat'].to_numpy()
    _cld_dist = df['cld_dist_km'].to_numpy()
    
    print("_date shape", _date.shape)
    print("_orbit_id shape", _orbit_id.shape)

    anomaly_orig    = df['xco2_bc_anomaly'].to_numpy()
    # Use _all variants (shape = full df N=84780) — subset arrays (38987,) mismatched _lat
    _checkpoint("compute_xco2_anomaly: calling LR")
    anomaly_lr  = compute_xco2_anomaly_date_id(_date, _orbit_id, _lat, _cld_dist, xco2_bc_corrected_all,     **_anomaly_args)
    _checkpoint("compute_xco2_anomaly: LR done")
    anomaly_mlp = compute_xco2_anomaly_date_id(_date, _orbit_id, _lat, _cld_dist, xco2_bc_corrected_all_mlp, **_anomaly_args)
    _checkpoint("compute_xco2_anomaly: MLP done")

    # Scatter + histogram comparison of anomalies
    _checkpoint("plot: anomaly scatter+hist")
    plt.close('all')
    fig, (ax_sc1, ax_sc2, ax_hist) = plt.subplots(1, 3, figsize=(18, 6))

    valid = np.isfinite(anomaly_orig) & np.isfinite(anomaly_lr)
    _x, _y = _scatter_ss(rng_plot, anomaly_orig[valid], anomaly_lr[valid])
    ax_sc1.scatter(_x, _y, c='orange', edgecolor=None, s=5, alpha=0.6, rasterized=True)
    _lim = np.nanpercentile(np.abs(anomaly_orig[valid]), 99)
    ax_sc1.set_xlim(-_lim, _lim); ax_sc1.set_ylim(-_lim, _lim)
    ax_sc1.set_aspect('equal', adjustable='box')
    ax_sc1.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc1.set_ylabel('LR-corrected XCO2_bc anomaly (ppm)')
    ax_sc1.set_title('Original vs LR-corrected anomaly')
    r2_lr = 1 - np.nansum((anomaly_orig[valid] - anomaly_lr[valid])**2) / \
                np.nansum((anomaly_orig[valid] - np.nanmean(anomaly_orig[valid]))**2)
    ax_sc1.text(0.05, 0.95, f'R²={r2_lr:.3f}', transform=ax_sc1.transAxes, va='top')

    valid2 = np.isfinite(anomaly_orig) & np.isfinite(anomaly_mlp)
    _x, _y = _scatter_ss(rng_plot, anomaly_orig[valid2], anomaly_mlp[valid2])
    ax_sc2.scatter(_x, _y, c='green', edgecolor=None, s=5, alpha=0.6, rasterized=True)
    ax_sc2.set_xlim(-_lim, _lim); ax_sc2.set_ylim(-_lim, _lim)
    ax_sc2.set_aspect('equal', adjustable='box')
    ax_sc2.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc2.set_ylabel('MLP-corrected XCO2_bc anomaly (ppm)')
    ax_sc2.set_title('Original vs MLP-corrected anomaly')
    r2_mlp = 1 - np.nansum((anomaly_orig[valid2] - anomaly_mlp[valid2])**2) / \
                 np.nansum((anomaly_orig[valid2] - np.nanmean(anomaly_orig[valid2]))**2)
    ax_sc2.text(0.05, 0.95, f'R²={r2_mlp:.3f}', transform=ax_sc2.transAxes, va='top')

    _bins = np.linspace(-3, 3, 211)
    for _anom, _color, _label in [
            (anomaly_orig, 'blue',   'Original'),
            (anomaly_lr,   'orange', 'LR-corrected'),
            (anomaly_mlp,  'green',  'MLP-corrected'),
    ]:
        _v = _anom[np.isfinite(_anom)]
        _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
        ax_hist.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                     label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
        ax_hist.axvline(_mu, color=_color, linestyle='-',  linewidth=1.2)
        ax_hist.axvline(_mu - _sigma, color=_color, linestyle=':', linewidth=0.9)
        ax_hist.axvline(_mu + _sigma, color=_color, linestyle=':', linewidth=0.9)

    ax_hist.set_xlabel('XCO2_bc anomaly (ppm)')
    ax_hist.set_title('Anomaly distribution comparison')
    ax_hist.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax_hist.legend(fontsize=10)

    fig.tight_layout()
    fname = (f"anomaly_comparison_reference_{os.path.basename(test_csv).split('.')[0]}.png"
             if test_csv else "anomaly_comparison.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3×3 XCO2_bc comparison by cloud-distance regime ──────────────────
    _checkpoint("plot: 3×3 XCO2 comparison")
    # Row masks — use pre-extracted numpy arrays (df_xco2_anomaly already freed)
    _cd_orig       = _cd_orig_for_masks
    mask_r1        = _cd_orig > 10
    mask_r2        = _cd_orig < 10
    _cd_all_arr    = np.array(df['cld_dist_km'])
    _anom_all_orig = np.array(df['xco2_bc_anomaly'])
    mask_r3        = (_cd_all_arr < 10) & np.isnan(_anom_all_orig)
    del _cd_all_arr, _anom_all_orig

    # Data sources per row: use pre-extracted numpy arrays (no df_xco2_anomaly reference)
    row_configs = [
        (df_xco2_bc_anomaly, df_xco2_bc, xco2_bc_pred_anomaly, xco2_bc_pred_anomaly_mlp,
         mask_r1, 'Clear-sky FPs (cld_dist > 10 km)', True),
        (df_xco2_bc_anomaly, df_xco2_bc, xco2_bc_pred_anomaly, xco2_bc_pred_anomaly_mlp,
         mask_r2, 'Cloud-affected FPs from df_orig (cld_dist < 10 km)', True),
        (df_all_xco2_bc, df_all_xco2_bc, xco2_bc_predict_all_anomaly, xco2_bc_predict_all_anomaly_mlp,
         mask_r3, 'Cloud FPs with NaN anomaly from df (cld_dist < 10 km)', False),
    ]
    # _all_xco2 concatenation was dead code (downstream vars were commented out) — removed

    plt.close('all')
    fig, axes = plt.subplots(3, 5, figsize=(27, 17))

    for row_i, (xco2_orig, xco2_orig_bc, xco2_lr, xco2_mlp, mask, row_label, anomaly_label) in enumerate(row_configs):
        ax_sc1, ax_sc2, ax_h, ax_h2, ax_h3 = axes[row_i]

        x_orig = xco2_orig[mask]
        xco2_orig_bc = xco2_orig_bc[mask]
        x_lr   = xco2_lr[mask]
        x_mlp  = xco2_mlp[mask]

        v_lr  = np.isfinite(x_orig) & np.isfinite(x_lr)
        v_mlp = np.isfinite(x_orig) & np.isfinite(x_mlp)

        _fin = x_orig[np.isfinite(x_orig)]
        _lo = float(np.nanpercentile(_fin, 1))  if len(_fin) > 0 else -5.0
        _hi = float(np.nanpercentile(_fin, 99)) if len(_fin) > 0 else  5.0

        
        for ax, x_corr, _color, method in [
                (ax_sc1, x_lr,  'orange', 'LR'),
                (ax_sc2, x_mlp, 'green',  'MLP'),
        ]:
            if not (x_orig == xco2_orig_bc).all():
                v = np.isfinite(x_orig) & np.isfinite(x_corr)
                _sx, _sy = _scatter_ss(rng_plot, x_orig[v], x_corr[v])
                ax.scatter(_sx, _sy, c=_color, edgecolor=None, s=5, alpha=0.6, rasterized=True)
                ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
                ax.set_aspect('equal', adjustable='box')
                ax.axline((_lo, _lo), slope=1, color='r', linestyle='--')
                if anomaly_label:
                    ax.set_xlabel('Original XCO2_bc anomaly(ppm)')
                    ax.set_ylabel(f'{method}-corrected XCO2_bc anomaly (ppm)')
                else:
                    ax.set_xlabel('Original XCO2_bc (ppm)')
                    ax.set_ylabel(f'{method}-corrected XCO2_bc (ppm)')
                ax.set_title(f'{row_label}\n[{method} scatter]')
                if v.sum() > 1:
                    r2 = 1 - np.nansum((x_orig[v] - x_corr[v])**2) / \
                            np.nansum((x_orig[v] - np.nanmean(x_orig[v]))**2)
                    ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, va='top')
            else:
                ax.set_visible(False)  # hide scatter if original vs original

        for _xco2, _xco2_bc, _color, _label in [
                (x_orig, xco2_orig_bc, 'blue',   'Original'),
                (x_lr, xco2_orig_bc,   'orange', 'LR-corrected'),
                (x_mlp, xco2_orig_bc,  'green',  'MLP-corrected'),
        ]:
            _v = _xco2[np.isfinite(_xco2)]
            _v2 = _xco2_bc[np.isfinite(_xco2)]
            if len(_v) == 0:
                continue
            if len(_v2) > 0 and not (_v == _v2).all():
                _mu, _sigma = _v.mean(), _v.std()
                _xco2_lo = np.nanpercentile(_v, 1)
                _xco2_hi = np.nanpercentile(_v, 99)
                _bins_3x4 = np.linspace(_xco2_lo, _xco2_hi, 100)
                ax_h.hist(_v, bins=_bins_3x4, color=_color, alpha=0.6, density=True,
                        label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
                ax_h.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
                ax_h.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
                ax_h.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)
             
            if _label != 'Original':
                _v2 = _v2 - _v # bias-corrected distribution (centered on zero mean) for clearer comparison of spread
            _mu2, _sigma2 = _v2.mean(), _v2.std()
            _xco2_v2_lo = np.nanpercentile(_v2, 1)
            _xco2_v2_hi = np.nanpercentile(_v2, 99)
            _bins_v2_3x4 = np.linspace(_xco2_v2_lo, _xco2_v2_hi, 100)
            ax_h2.hist(_v2, bins=_bins_v2_3x4, color=_color, alpha=0.3, density=True,
                        label=f'{_label}  \nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            ax_h2.axvline(_mu2,          color=_color, linestyle='-',  linewidth=1.0)
            ax_h2.axvline(_mu2 - _sigma2, color=_color, linestyle=':',  linewidth=0.8)
            ax_h2.axvline(_mu2 + _sigma2, color=_color, linestyle=':',  linewidth=0.8)
            
            
            # Col 4: ideal distribution of xco2_bc values — only for Original, and only if _v != _v2 (e.g. row-3 Original)
            if _label == 'Original' and not (_v == _v2).all():
                _v2 = _v2 - _v   # corrected xco2_bc = raw − predicted_anomaly
                _mu2, _sigma2 = _v2.mean(), _v2.std()
                bins3 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
                ax_h3.hist(_v2, bins=bins3, color=_color, alpha=0.2, density=True,
                        label=f'Idael {_label}-corrected \nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            elif (_v == _v2).all():
                ax_h3.set_visible(False)
            else:
                _mu2, _sigma2 = _v2.mean(), _v2.std()
                bins3 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
                ax_h3.hist(_v2, bins=bins3, color=_color, alpha=0.2, density=True,
                        label=f' {_label}\nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            ax_h3.axvline(_mu2,           color=_color, linestyle='-',  linewidth=0.8)
            ax_h3.axvline(_mu2 - _sigma2, color=_color, linestyle=':',  linewidth=0.6)
            ax_h3.axvline(_mu2 + _sigma2, color=_color, linestyle=':',  linewidth=0.6)

            ax_h.set_title(f'{row_label}\n[Distribution]')
            ax_h.set_xlabel('XCO2_bc anomaly (ppm)')
            ax_h.legend(fontsize=10)
            ax_h2.set_title(f'{row_label}\n[Distribution]')
            ax_h2.legend(fontsize=10)
            if not (_v == _v2).all():
                ax_h2.set_xlabel('Corrected XCO2 (ppm)')
            else:
                ax_h2.set_xlabel('XCO2 (ppm)')
            ax_h3.set_title(f'{row_label}\n[Ideal corrected XCO2 distribution]')
            ax_h3.legend(fontsize=10)


                

    fig.suptitle('XCO2_bc by cloud-distance regime', fontsize=13, y=1.01)
    fig.tight_layout()
    fname = (f"xco2bc_comparison_3x3_reference_{os.path.basename(test_csv).split('.')[0]}.png"
             if test_csv else "xco2bc_comparison_3x3.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Scatter: original vs corrected (linear baseline + MLP) ───────────
    _checkpoint("plot: original vs corrected scatter")
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax, ax_mlp, ax2 = axes

    _vmin = min(np.nanmin(df_all_xco2_bc), np.nanmin(xco2_bc_corrected_all), np.nanmin(xco2_bc_corrected_all_mlp))
    _vmax = max(np.nanmax(df_all_xco2_bc), np.nanmax(xco2_bc_corrected_all), np.nanmax(xco2_bc_corrected_all_mlp))

    _sx, _sy = _scatter_ss(rng_plot, df_all_xco2_bc, xco2_bc_corrected_all)
    ax.scatter(_sx, _sy, c='blue', edgecolor=None, s=5, alpha=0.7, rasterized=True)
    ax.set_xlabel('Original XCO2 (ppm)')
    ax.set_ylabel('LR Corrected XCO2 (ppm)')
    ax.set_title('LR Correction of OCO-2 L2 XCO2')
    ax.plot([_vmin, _vmax], [_vmin, _vmax], 'r--')

    _sx, _sy = _scatter_ss(rng_plot, df_all_xco2_bc, xco2_bc_corrected_all_mlp)
    ax_mlp.scatter(_sx, _sy, c='green', edgecolor=None, s=5, alpha=0.7, rasterized=True)
    ax_mlp.set_xlabel('Original XCO2 (ppm)')
    ax_mlp.set_ylabel('MLP Corrected XCO2 (ppm)')
    ax_mlp.set_title('MLP Correction of OCO-2 L2 XCO2')
    ax_mlp.plot([_vmin, _vmax], [_vmin, _vmax], 'r--')

    bins = np.arange(390, 420.1, 0.5)
    ax2.hist(df_all_xco2_bc,       bins=bins, color='blue',   alpha=0.6, density=True, label='Original Lite')
    ax2.hist(xco2_bc_corrected_all,       bins=bins, color='orange', alpha=0.6, density=True, label='LR Corrected')
    ax2.hist(xco2_bc_corrected_all_mlp,   bins=bins, color='green',  alpha=0.6, density=True, label='MLP Corrected')
    ax2.set_xlabel('XCO2 (ppm)')
    ax2.set_title('Distribution: Original / LR / MLP Corrected XCO2')
    ax2.legend()

    fname = (f"LR_MLP_correction_lt_xco2_scatter_reference_{os.path.basename(test_csv).split('.')[0]}.png"
             if test_csv else "LR_MLP_bc_correction_lt_xco2_scatter.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Map: spatial distribution of k1, XCO2, and LR correction ─────────
    xco2_min = min(np.nanmin(df['xco2_bc']), np.nanmin(df['xco2_raw']))
    xco2_max = max(np.nanmax(df['xco2_bc']), np.nanmax(df['xco2_raw']))

    plt.close('all')
    fig, ((ax1, ax2, ax6, ax7), (ax4, ax3, ax5, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
    
    plot_lon = np.array(df.lon)
    if plot_lon.min() < -90 and plot_lon.max() > 90:
        # Assume longitude is in [0, 360] and convert to [-180, 180] for better map visualization
        plot_lon[plot_lon < 0] += 360

    # Subsample once for the whole map figure — same index for all 8 panels
    _map_arrs = _scatter_ss(rng_plot,
        plot_lon, np.array(df.lat),
        np.array(df.o2a_k1) if 'o2a_k1' in df.columns else np.zeros(len(plot_lon)),
        np.array(df.o2a_k2) if 'o2a_k2' in df.columns else np.zeros(len(plot_lon)),
        np.array(df['xco2_bc']), np.array(df['xco2_raw']),
        xco2_bc_corrected_all, xco2_bc_predict_all_anomaly,
        xco2_bc_predict_all_anomaly_mlp, xco2_bc_corrected_all_mlp,
    )
    _mlon, _mlat, _mk1, _mk2, _mxbc, _mxraw, _mlr, _mlr_anom, _mmlp_anom, _mmlp = _map_arrs

    sc1 = ax1.scatter(_mlon, _mlat, c=_mk1, cmap='jet', s=20, alpha=0.7, rasterized=True)
    ax1.set_title('Retrieved O2A k1');  fig.colorbar(sc1, ax=ax1, label='k1')

    sc2 = ax2.scatter(_mlon, _mlat, c=_mk2, cmap='jet', s=20, alpha=0.7, rasterized=True)
    ax2.set_title('Retrieved O2A k2');  fig.colorbar(sc2, ax=ax2, label='k2')

    sc3 = ax3.scatter(_mlon, _mlat, c=_mxbc, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max, rasterized=True)
    ax3.set_title('OCO-2 L2 XCO2 (bias-corrected)');  fig.colorbar(sc3, ax=ax3, label='XCO2 (ppm)')

    sc4 = ax4.scatter(_mlon, _mlat, c=_mxraw, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max, rasterized=True)
    ax4.set_title('OCO-2 L2 XCO2 raw');  fig.colorbar(sc4, ax=ax4, label='XCO2 raw (ppm)')

    sc5 = ax5.scatter(_mlon, _mlat, c=_mlr, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max, rasterized=True)
    ax5.set_title('LR Corrected XCO2');  fig.colorbar(sc5, ax=ax5, label='LR corrected XCO2 (ppm)')

    sc6 = ax6.scatter(_mlon, _mlat, c=_mlr_anom, cmap='jet', s=20, alpha=0.7, rasterized=True)
    ax6.set_title('LR predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc6, ax=ax6, label='LR predicted XCO2 anomaly (ppm)')

    sc7 = ax7.scatter(_mlon, _mlat, c=_mmlp_anom, cmap='jet', s=20, alpha=0.7, rasterized=True)
    ax7.set_title('MLP predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc7, ax=ax7, label='MLP predicted XCO2 anomaly (ppm)')

    sc8 = ax8.scatter(_mlon, _mlat, c=_mmlp, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max, rasterized=True)
    ax8.set_title('MLP Corrected XCO2');  fig.colorbar(sc8, ax=ax8, label='MLP corrected XCO2 (ppm)')
    del _map_arrs, _mlon, _mlat, _mk1, _mk2, _mxbc, _mxraw, _mlr, _mlr_anom, _mmlp_anom, _mmlp

    fig.suptitle(f"Cloud distance threshold for XCO2 mean: 10 km")
    fname = (f"LR_MLP_correction_lt_xco2_map_reference_{os.path.basename(test_csv).split('.')[0]}.png"
             if test_csv else "LR_MLP_correction_lt_xco2_map_all.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if ledger_path is not None and run_id is not None and commit is not None and script_name is not None:
        try:
            import resource as _res_summary
            rss_raw = _res_summary.getrusage(_res_summary.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = rss_raw / (1024 * 1024) if platform.system() == "Darwin" else rss_raw / 1024
        except Exception:
            peak_memory_mb = 0.0

        runtime_seconds = (datetime.now() - run_start).total_seconds()
        y_test_arr = np.asarray(y_test_eval, dtype=float)
        ridge_pred = model.predict(X_test_eval)
        mlp_pred = y_all_mlp

        ridge_rmse = float(np.sqrt(np.mean((y_test_arr - ridge_pred) ** 2)))
        ridge_mae = float(np.mean(np.abs(y_test_arr - ridge_pred)))
        ridge_r2 = float(model.score(X_test_eval, y_test_arr))
        mlp_rmse = float(np.sqrt(np.mean((y_test_arr - mlp_pred) ** 2)))
        mlp_mae = float(np.mean(np.abs(y_test_arr - mlp_pred)))
        denom = np.sum((y_test_arr - y_test_arr.mean()) ** 2)
        mlp_r2 = float(1.0 - np.sum((y_test_arr - mlp_pred) ** 2) / denom) if denom > 0 else float('nan')

        summary = RunSummary(
            run_id=run_id,
            script_name=script_name,
            model_family='mlp_lr',
            commit=commit,
            status='success',
            primary_metric_name='mlp_test_rmse',
            primary_metric_value=mlp_rmse,
            secondary_metrics={
                'mlp_test_mae': mlp_mae,
                'mlp_test_r2': mlp_r2,
                'ridge_test_rmse': ridge_rmse,
                'ridge_test_mae': ridge_mae,
                'ridge_test_r2': ridge_r2,
            },
            peak_memory_mb=float(peak_memory_mb),
            runtime_seconds=float(runtime_seconds),
            description='MLP+Ridge mitigation test with shared feature pipeline',
            artifacts={
                'output_dir': str(output_dir),
                'run_config': str(output_dir / 'mlp_run_config.json'),
                'feature_importance_csv': str(output_dir / 'feature_importance_xco2_bc_anomaly.csv'),
                'learning_curve_png': str(output_dir / 'mlp_learning_curve.png'),
                'summary_json': str(output_dir / 'run_summary.json'),
            },
            config=cfg,
        )
        write_run_summary(summary, output_dir=output_dir, ledger_path=ledger_path)


    

if __name__ == "__main__":
    main()
