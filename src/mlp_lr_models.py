import argparse
import h5py
import numpy as np
import pandas as pd
import os
import sys
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
import platform
import logging
from datetime import datetime
from tqdm import tqdm
from utils import get_storage_dir



logger = logging.getLogger(__name__)


# ─── Main analysis entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLP + Ridge XCO2 bias correction")
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder name appended to the base output directory '
                             '(e.g. --suffix v2_reduced).  '
                             'Creates results/model_mlp_lr/<suffix>/.')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a saved FeaturePipeline (.pkl).  '
                             'If omitted, a new pipeline is fitted on the training data '
                             'and saved to <output_dir>/pipeline.pkl.')
    args = parser.parse_args()

    storage_dir = get_storage_dir()
    fdir      = storage_dir / 'results/csv_collection'
    data_name = 'combined_2020_dates.csv'
    data_name = 'combined_2017_2021_dates.csv'  # for full 2-year dataset
    # data_name = 'combined_2020-01-01_all_orbits.csv'  # for quick testing with one date's data
    base_dir   = storage_dir / 'results/model_mlp_lr'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sfc_type = 0  # Ocean only for now
    # sfc_type = 1  # Land only for now

    data_path = os.path.join(fdir, data_name)
    df = pd.read_csv(data_path)
    df = df[df['sfc_type'] == sfc_type]
    df = df[df['snow_flag'] == 0]

    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=sfc_type)
        pipeline_path = output_dir / 'pipeline.pkl'
        pipeline.save(pipeline_path)
        print(f"  Fitted and saved pipeline → {pipeline_path}", flush=True)

    mitigation_test(df, output_dir=output_dir, pipeline=pipeline, test_csv=None)

def compute_xco2_anomaly_date_id(fp_date, fp_orbit_id, fp_lat, cld_dist_km, xco2,
                         lat_thres=0.5, std_thres=2.0, min_cld_dist=10.0,
                         chunk_size=512):
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

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy
    """
    fp_date     = np.asarray(fp_date)
    fp_orbit_id = np.asarray(fp_orbit_id)
    fp_lat      = np.asarray(fp_lat,      dtype=float)
    xco2        = np.asarray(xco2,        dtype=float)
    cld_dist_km = np.asarray(cld_dist_km, dtype=float)

    N          = len(fp_lat)
    chunk_size = int(chunk_size)
    anomaly    = np.full(N, np.nan)

    # Build index groups: {(date, orbit_id): [row indices]}
    groups: dict = {}
    for i, key in enumerate(zip(fp_date, fp_orbit_id)):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    for key, idx_list in groups.items():
        idx    = np.array(idx_list)
        g_lat  = fp_lat[idx]
        g_dist = cld_dist_km[idx]
        g_xco2 = xco2[idx]
        M      = len(idx)

        valid_lat  = ~np.isnan(g_lat)
        clear_mask = valid_lat & (g_dist > min_cld_dist)

        # Reference arrays for this group (non-clear-sky → NaN, excluded by lat test)
        ref_lat  = np.where(clear_mask, g_lat,  np.nan)   # [M]
        ref_xco2 = np.where(clear_mask, g_xco2, np.nan)   # [M]

        g_ref_mean = np.full(M, np.nan)
        g_ref_std  = np.full(M, np.nan)

        for start in range(0, M, chunk_size):
            end   = min(start + chunk_size, M)
            q_lat = g_lat[start:end]                              # [c]

            lat_diff  = np.abs(q_lat[:, None] - ref_lat[None, :])  # [c, M]
            in_window = lat_diff <= lat_thres                       # [c, M]
            xco2_win  = np.where(in_window, ref_xco2[None, :], np.nan)
            has_refs  = in_window.any(axis=1)                       # [c]

            c_mean = np.full(end - start, np.nan)
            c_std  = np.full(end - start, np.nan)
            if has_refs.any():
                c_mean[has_refs] = np.nanmean(xco2_win[has_refs], axis=1)
                c_std[has_refs]  = np.nanstd( xco2_win[has_refs], axis=1)

            g_ref_mean[start:end] = c_mean
            g_ref_std[start:end]  = c_std
            del lat_diff, in_window, xco2_win

        valid = valid_lat & ~np.isnan(g_ref_mean) & (g_ref_std <= std_thres)
        anomaly[idx[valid]] = g_xco2[valid] - g_ref_mean[valid]

    return anomaly

# ─── LR mitigation test ────────────────────────────────────────────────────────

def mitigation_test(df, output_dir, pipeline: FeaturePipeline, test_csv=None,
                    perm_max_rows: int = 5000):
    """Train a per-footprint linear regression to predict XCO2 bias from kappas.

    Parameters
    ----------
    sat          : dict from preprocess()
    df           : DataFrame from k1k2_analysis
    output_dir   : str
    test_csv: str or None
    """
    
    import resource as _res
    def _checkpoint(label):
        rss_raw = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
        # macOS: ru_maxrss is bytes; Linux: kilobytes
        rss_mb = rss_raw / (1024 * 1024) if platform.system() == "Darwin" else rss_raw / 1024
        print(f"[MEM] {label:55s}  RSS={rss_mb:.0f} MB", flush=True)

    _checkpoint("mitigation_test: entry")

    df_xco2_anomaly = df[df['xco2_bc_anomaly'].notna()]
    # df_orig was a full deepcopy used only to read 4 columns — replaced with df[col] directly.

    _checkpoint("before pipeline.transform + train_test_split")
    features = pipeline.features
    X_all    = pipeline.transform(df)
    valid_rows  = ~df['xco2_bc_anomaly'].isna()
    X           = X_all[valid_rows.values]
    y           = df['xco2_bc_anomaly'][valid_rows].values
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    # ── end correlation matrix ─────────────────────────────────────────────

    _checkpoint("before pipeline.transform (df_X / df_all_X)")
    df_X     = X_all[valid_rows.values]
    df_all_X = X_all
    _checkpoint("after  pipeline.transform")

    # Convert to numpy immediately so pandas doesn't keep the full DataFrame columns pinned
    df_xco2_bc             = df_xco2_anomaly['xco2_bc'].to_numpy()
    df_xco2_bc_anomaly     = df_xco2_anomaly['xco2_bc_anomaly'].to_numpy()
    df_all_xco2_bc         = df['xco2_bc'].to_numpy()
    df_all_xco2_bc_anomaly = df['xco2_bc_anomaly'].to_numpy()

    # Determine training data and clear-sky reference level
    if test_csv is None:
        test_df = df  
    else:
        test_df = pd.read_csv(test_csv)
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

    # ── Linear baseline ────────────────────────────────────────────
    _checkpoint("before LinearRegression fit")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"Test FP R² (linear) for predicting xco2_bc anomaly: {model.score(X_test, y_test):.3f}")

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

    # Normalise target to zero-mean unit-variance so Huber loss is scale-invariant
    y_mean = float(y_train.mean())
    y_std  = float(y_train.std()) + 1e-8
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
        # Larger batch size for full dataset efficiency; 512 is fine for the local subset too
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)

        # Val loss uses batched inference to keep peak memory bounded on the 76K-sample test set
        def _val_loss(model_):
            model_.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(X_test), 4096):
                    Xb = torch.tensor(X_test[start:start + 4096], dtype=torch.float32).to(device)
                    yb = torch.tensor(y_test_n[start:start + 4096], dtype=torch.float32).to(device)
                    total += nn.functional.huber_loss(model_(Xb), yb, delta=1.0).item() * len(Xb)
                    n += len(Xb)
                    del Xb, yb
            return total / n

        mlp      = _MLP(n_in).to(device)
        n_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print(f"  MLP parameters: {n_params:,}  |  train samples: {len(X_train):,}  "
              f"|  ratio: {len(X_train)/n_params:.2f}", flush=True)

        optimizer = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

        train_losses, val_losses = [], []
        best_val_loss, best_state, patience, no_improve = float('inf'), None, 30, 0
        epoch_bar = tqdm(range(500), desc="MLP training", unit="epoch")
        for epoch in epoch_bar:
            mlp.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = nn.functional.huber_loss(mlp(bx), by, delta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
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
        ax_lc.plot(epochs_ran, train_losses, label='Train (Huber)', color='steelblue')
        ax_lc.plot(epochs_ran, val_losses,   label='Val   (Huber)', color='tomato')
        best_ep = int(np.argmin(val_losses)) + 1
        ax_lc.axvline(best_ep, color='gray', linestyle='--', linewidth=0.8,
                      label=f'Best val epoch {best_ep}')
        ax_lc.set_xlabel('Epoch')
        ax_lc.set_ylabel('Huber loss (normalised target)')
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

    def _mlp_infer(X_np, batch_size=4096):
        """Batched inference; returns predictions in original (ppm) units."""
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

    # Permutation importance — subsample to cap memory/compute (perm_max_rows rows is enough for stable estimates)
    rng_pi   = np.random.default_rng(42)
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
        rng_inner   = np.random.default_rng(0)
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
    perm_imp_lr = _permutation_importance(model.predict, X_pi, y_pi, skip_cols=fp_skip_cols)
    _checkpoint("permutation importance: MLP")
    perm_imp_mlp = _permutation_importance(_mlp_infer, X_pi, y_pi, skip_cols=fp_skip_cols)
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
    del _mlp_infer, mlp
    del X_pi, y_pi
    del y_train, y_test, X_train, X_test
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
    ax_sc1.scatter(anomaly_orig[valid], anomaly_lr[valid],
                   c='orange', edgecolor=None, s=5, alpha=0.6)
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
    ax_sc2.scatter(anomaly_orig[valid2], anomaly_mlp[valid2],
                   c='green', edgecolor=None, s=5, alpha=0.6)
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
    # Row masks
    _cd_orig       = np.array(df_xco2_anomaly['cld_dist_km'])
    mask_r1        = _cd_orig > 10
    mask_r2        = _cd_orig < 10
    _cd_all_arr    = np.array(df['cld_dist_km'])
    _anom_all_orig = np.array(df['xco2_bc_anomaly'])
    mask_r3        = (_cd_all_arr < 10) & np.isnan(_anom_all_orig)

    # Data sources per row: (xco2_orig, xco2_lr, xco2_mlp, mask)
    row_configs = [
        (np.array(df_xco2_anomaly['xco2_bc_anomaly']), np.array(df_xco2_anomaly['xco2_bc']), xco2_bc_pred_anomaly, xco2_bc_pred_anomaly_mlp,
         mask_r1, 'Clear-sky FPs (cld_dist > 10 km)', True),
        (np.array(df_xco2_anomaly['xco2_bc_anomaly']), np.array(df_xco2_anomaly['xco2_bc']), xco2_bc_pred_anomaly, xco2_bc_pred_anomaly_mlp,
         mask_r2, 'Cloud-affected FPs from df_orig (cld_dist < 10 km)', True),
        (np.array(df['xco2_bc']), np.array(df['xco2_bc']), xco2_bc_predict_all_anomaly, xco2_bc_predict_all_anomaly_mlp,
         mask_r3, 'Cloud FPs with NaN anomaly from df (cld_dist < 10 km)', False),
    ]

    # Shared histogram bins across all rows
    _all_xco2 = np.concatenate([
        np.array(df_xco2_anomaly['xco2_bc']),
        np.array(df['xco2_bc']),
        xco2_bc_corrected, xco2_bc_corrected_mlp,
        xco2_bc_corrected_all, xco2_bc_corrected_all_mlp,
    ])
    # _xco2_lo = np.nanpercentile(_all_xco2, 1)
    # _xco2_hi = np.nanpercentile(_all_xco2, 99)
    # _bins_3x3 = np.linspace(_xco2_lo, _xco2_hi, 100)

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

        _lo = np.nanpercentile(x_orig[np.isfinite(x_orig)], 1)  if np.isfinite(x_orig).any() else _xco2_lo
        _hi = np.nanpercentile(x_orig[np.isfinite(x_orig)], 99) if np.isfinite(x_orig).any() else _xco2_hi

        
        for ax, x_corr, _color, method in [
                (ax_sc1, x_lr,  'orange', 'LR'),
                (ax_sc2, x_mlp, 'green',  'MLP'),
        ]:
            if not (x_orig == xco2_orig_bc).all():
                v = np.isfinite(x_orig) & np.isfinite(x_corr)
                ax.scatter(x_orig[v], x_corr[v], c=_color, edgecolor=None, s=5, alpha=0.6)
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

    ax.scatter(df_all_xco2_bc, xco2_bc_corrected_all, c='blue', edgecolor=None, s=5, alpha=0.7)
    ax.set_xlabel('Original XCO2 (ppm)')
    ax.set_ylabel('LR Corrected XCO2 (ppm)')
    ax.set_title('LR Correction of OCO-2 L2 XCO2')
    ax.plot([_vmin, _vmax], [_vmin, _vmax], 'r--')

    ax_mlp.scatter(df_all_xco2_bc, xco2_bc_corrected_all_mlp, c='green', edgecolor=None, s=5, alpha=0.7)
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
    
    sc1 = ax1.scatter(plot_lon, df.lat, c=df.o2a_k1, cmap='jet', s=20, alpha=0.7)
    ax1.set_title('Retrieved O2A k1');  fig.colorbar(sc1, ax=ax1, label='k1')

    sc2 = ax2.scatter(plot_lon, df.lat, c=df.o2a_k2, cmap='jet', s=20, alpha=0.7)
    ax2.set_title('Retrieved O2A k2');  fig.colorbar(sc2, ax=ax2, label='k2')

    sc3 = ax3.scatter(plot_lon, df.lat, c=df['xco2_bc'], cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax3.set_title('OCO-2 L2 XCO2 (bias-corrected)');  fig.colorbar(sc3, ax=ax3, label='XCO2 (ppm)')

    sc4 = ax4.scatter(plot_lon, df.lat, c=df['xco2_raw'], cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax4.set_title('OCO-2 L2 XCO2 raw');  fig.colorbar(sc4, ax=ax4, label='XCO2 raw (ppm)')

    sc5 = ax5.scatter(plot_lon, df.lat, c=xco2_bc_corrected_all, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax5.set_title('LR Corrected XCO2');  fig.colorbar(sc5, ax=ax5, label='LR corrected XCO2 (ppm)')

    sc6 = ax6.scatter(plot_lon, df.lat, c=xco2_bc_predict_all_anomaly, cmap='jet', s=20, alpha=0.7)
    ax6.set_title('LR predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc6, ax=ax6, label='LR predicted XCO2 anomaly (ppm)')

    sc7 = ax7.scatter(plot_lon, df.lat, c=xco2_bc_predict_all_anomaly_mlp, cmap='jet', s=20, alpha=0.7)
    ax7.set_title('MLP predicted XCO2 anomaly (ppm)')
    fig.colorbar(sc7, ax=ax7, label='MLP predicted XCO2 anomaly (ppm)')

    sc8 = ax8.scatter(plot_lon, df.lat, c=xco2_bc_corrected_all_mlp, cmap='jet', s=20, alpha=0.7,
                      vmin=xco2_min, vmax=xco2_max)
    ax8.set_title('MLP Corrected XCO2');  fig.colorbar(sc8, ax=ax8, label='MLP corrected XCO2 (ppm)')

    fig.suptitle(f"Cloud distance threshold for XCO2 mean: 10 km")
    fname = (f"LR_MLP_correction_lt_xco2_map_reference_{os.path.basename(test_csv).split('.')[0]}.png"
             if test_csv else "LR_MLP_correction_lt_xco2_map_all.png")
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


    

if __name__ == "__main__":
    main()
