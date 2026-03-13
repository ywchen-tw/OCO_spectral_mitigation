"""mlp_lr_cloud_classifier.py — Binary classifier for cloud proximity (cld_dist_km < 10).

Trains two models that predict whether a sounding's nearest cloud is within 10 km,
and outputs a calibrated confidence (probability):

    1. Logistic Regression (sklearn baseline)
    2. Residual MLP (PyTorch)  — same _MLP backbone as mlp_lr_models.py

Usage
-----
    python src/mlp_lr_cloud_classifier.py [--sfc_type 0] [--suffix my_run] [--pipeline path.pkl]

Outputs (in results/model_mlp_clf/<suffix>/):
    logistic_model.pkl         — LogisticAdapter checkpoint
    mlp_clf_weights.pt         — MLP weights (CPU)
    mlp_clf_meta.pkl           — MLP architecture + threshold meta
    pipeline.pkl               — fitted FeaturePipeline (if not supplied)
    clf_learning_curve.png
    clf_roc_curve.png
    clf_pr_curve.png
    clf_calibration.png
    clf_score_distribution.png
    clf_spatial_map.png
    clf_feature_importance.png
"""

import argparse
import copy
import gc
import logging
import os
import platform
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model_adapters import LogisticAdapter, MLPClassifierAdapter, _MLP
from pipeline import FeaturePipeline
from utils import get_storage_dir

logger = logging.getLogger(__name__)

_SCATTER_MAX = 150_000


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _scatter_ss(rng, *arrs):
    """Subsample all arrays to ≤ _SCATTER_MAX rows using the same random index."""
    N = len(arrs[0])
    if N <= _SCATTER_MAX:
        return arrs
    idx = rng.choice(N, size=_SCATTER_MAX, replace=False)
    return tuple(a[idx] for a in arrs)


def _checkpoint_fn(label, t0):
    import resource as _res
    rss_raw = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_raw / (1024 * 1024) if platform.system() == "Darwin" else rss_raw / 1024
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[MEM] {label:55s}  RSS={rss_mb:.0f} MB  t={elapsed:7.1f}s", flush=True)


# ─── Main classification function ──────────────────────────────────────────────

def cloud_proximity_classification(df: pd.DataFrame, output_dir, pipeline: FeaturePipeline):
    """Train LR + MLP classifiers for cld_dist_km < 10 and produce all evaluation plots.

    Parameters
    ----------
    df          : DataFrame filtered by sfc_type and snow_flag
    output_dir  : Path or str — directory to write checkpoints and plots
    pipeline    : fitted FeaturePipeline
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t0 = datetime.now()
    def _ckpt(label):
        _checkpoint_fn(label, t0)

    _ckpt("cloud_proximity_classification: entry")

    # ── Device selection ───────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"  Device: {device}", flush=True)

    # ── Target ────────────────────────────────────────────────────────────────
    cld_dist = df['cld_dist_km'].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(cld_dist)
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"[warn] Dropping {n_dropped} rows with non-finite cld_dist_km")
        df = df[valid_mask].reset_index(drop=True)
        cld_dist = cld_dist[valid_mask]

    y_all = (cld_dist < 10.0).astype(np.float32)
    n_pos = int(y_all.sum())
    n_neg = int((~y_all.astype(bool)).sum())
    pos_rate = n_pos / len(y_all)
    print(f"  Class distribution: cloud (cld<10 km) = {n_pos:,} ({pos_rate:.1%}), "
          f"clear = {n_neg:,} ({1-pos_rate:.1%})", flush=True)
    pos_weight = n_neg / max(n_pos, 1)

    # ── Feature transform ──────────────────────────────────────────────────────
    _ckpt("before pipeline.transform")
    X_all = pipeline.transform(df).astype(np.float32, copy=False)
    features = pipeline.features

    # Drop non-finite feature rows
    feat_finite = np.all(np.isfinite(X_all), axis=1)
    if not feat_finite.all():
        n_drop = int((~feat_finite).sum())
        print(f"[warn] Dropping {n_drop} rows with non-finite features")
        X_all = X_all[feat_finite]
        y_all = y_all[feat_finite]
        df    = df[feat_finite].reset_index(drop=True)

    _ckpt("after pipeline.transform")

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}", flush=True)
    _ckpt("after train_test_split")

    rng_plot = np.random.default_rng(1)

    # ── Logistic Regression ────────────────────────────────────────────────────
    _ckpt("before LogisticRegression fit")
    if LogisticAdapter.can_load(output_dir):
        print("  [checkpoint] LogisticAdapter found → loading", flush=True)
        lr_adapter = LogisticAdapter.load(output_dir)
    else:
        lr_model = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'
        )
        lr_model.fit(X_train, y_train)
        lr_adapter = LogisticAdapter(lr_model)
        lr_adapter.save(output_dir)
    _ckpt("after LogisticRegression fit")

    lr_proba_test = lr_adapter.predict_proba(X_test)
    lr_pred_test  = (lr_proba_test >= 0.5).astype(np.int8)
    lr_auroc = roc_auc_score(y_test, lr_proba_test)
    lr_auprc = average_precision_score(y_test, lr_proba_test)
    lr_brier = brier_score_loss(y_test, lr_proba_test)
    lr_f1    = f1_score(y_test, lr_pred_test)
    print(f"  LR  AUROC={lr_auroc:.4f}  AUPRC={lr_auprc:.4f}  "
          f"F1={lr_f1:.4f}  Brier={lr_brier:.4f}", flush=True)

    # ── MLP Classifier ─────────────────────────────────────────────────────────
    _ckpt("before MLP init / training")
    n_in = X_train.shape[1]

    if MLPClassifierAdapter.can_load(output_dir):
        print("  [checkpoint] MLPClassifierAdapter found → loading", flush=True)
        mlp_adapter = MLPClassifierAdapter.load(output_dir, device=device)
        mlp = mlp_adapter.model
        train_losses, val_losses = [], []   # empty — skip learning curve
        _mlp_from_ckpt = True
    else:
        _mlp_from_ckpt = False
        mlp = _MLP(n_in).to(device)
        n_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print(f"  MLP parameters: {n_params:,}  |  train: {len(X_train):,}  "
              f"|  ratio: {len(X_train)/n_params:.2f}", flush=True)

        pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=8192, shuffle=True
        )

        n_epochs = 150 if platform.system() == "Darwin" else 500
        t_max    = n_epochs

        optimizer = torch.optim.AdamW(mlp.parameters(), lr=6e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-6
        )

        def _val_metrics(model_):
            """Return val BCE loss and AUROC on the full test set (batched)."""
            model_.eval()
            all_logits = []
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(X_test), 8192):
                    Xb = torch.tensor(X_test[start:start+8192],
                                      dtype=torch.float32).to(device)
                    yb = torch.tensor(y_test[start:start+8192],
                                      dtype=torch.float32).to(device)
                    logits = model_(Xb)
                    total_loss += criterion(logits, yb).item() * len(Xb)
                    all_logits.append(torch.sigmoid(logits).cpu().numpy())
                    n += len(Xb)
                    del Xb, yb
            proba = np.concatenate(all_logits)
            bce = total_loss / n
            try:
                auroc = roc_auc_score(y_test, proba)
            except Exception:
                auroc = 0.5
            return bce, auroc

        train_losses, val_losses = [], []
        best_val_loss, best_state, patience, no_improve = float('inf'), None, 30, 0

        epoch_bar = tqdm(range(n_epochs), desc="MLP clf training", unit="epoch")
        for epoch in epoch_bar:
            mlp.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(mlp(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            val_bce, val_auroc = _val_metrics(mlp)

            train_losses.append(train_loss)
            val_losses.append(val_bce)

            improved = val_bce < best_val_loss
            if improved:
                best_val_loss = val_bce
                best_state    = copy.deepcopy(mlp.state_dict())
                no_improve    = 0
            else:
                no_improve += 1

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val_bce=f"{val_bce:.4f}",
                val_auroc=f"{val_auroc:.4f}",
                patience=f"{no_improve}/{patience}",
                saved="✓" if improved else "",
            )

            if no_improve >= patience:
                tqdm.write(f"  Early stop epoch {epoch} (best val BCE={best_val_loss:.4f})")
                break

        mlp.load_state_dict(best_state)
        mlp.eval()

        MLPClassifierAdapter(mlp, threshold=0.5, device=device).save(output_dir)
        mlp_adapter = MLPClassifierAdapter.load(output_dir, device=device)
        mlp.to(device)
        del train_ds, train_loader, best_state
        gc.collect()

    _ckpt("after MLP training")

    # ── MLP inference on test set ──────────────────────────────────────────────
    mlp_proba_test = mlp_adapter.predict_proba(X_test)
    mlp_pred_test  = (mlp_proba_test >= 0.5).astype(np.int8)
    mlp_auroc = roc_auc_score(y_test, mlp_proba_test)
    mlp_auprc = average_precision_score(y_test, mlp_proba_test)
    mlp_brier = brier_score_loss(y_test, mlp_proba_test)
    mlp_f1    = f1_score(y_test, mlp_pred_test)
    mlp_acc   = accuracy_score(y_test, mlp_pred_test)
    mlp_prec  = precision_score(y_test, mlp_pred_test, zero_division=0)
    mlp_rec   = recall_score(y_test, mlp_pred_test, zero_division=0)
    print(f"  MLP AUROC={mlp_auroc:.4f}  AUPRC={mlp_auprc:.4f}  "
          f"F1={mlp_f1:.4f}  Brier={mlp_brier:.4f}", flush=True)
    print(f"  MLP Acc={mlp_acc:.4f}  Prec={mlp_prec:.4f}  Rec={mlp_rec:.4f}", flush=True)
    cm = confusion_matrix(y_test, mlp_pred_test)
    print(f"  MLP Confusion matrix:\n{cm}", flush=True)

    # Full-dataset predictions for spatial map
    mlp_proba_all = mlp_adapter.predict_proba(X_all)
    lr_proba_all  = lr_adapter.predict_proba(X_all)

    # ── Plot: Learning curve ───────────────────────────────────────────────────
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs_ran = range(1, len(train_losses) + 1)
        ax.plot(epochs_ran, train_losses, label='Train BCE', color='steelblue')
        ax.plot(epochs_ran, val_losses,   label='Val BCE',   color='tomato')
        best_ep = int(np.argmin(val_losses)) + 1
        ax.axvline(best_ep, color='gray', linestyle='--', linewidth=0.8,
                   label=f'Best epoch {best_ep}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.set_title('MLP Classifier — Learning Curve')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'clf_learning_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Plot: ROC curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    for proba, label, color in [
        (lr_proba_test,  f'LR  (AUROC={lr_auroc:.4f})',  'steelblue'),
        (mlp_proba_test, f'MLP (AUROC={mlp_auroc:.4f})', 'forestgreen'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=label, color=color)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — cld_dist_km < 10')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Precision-Recall curve ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    for proba, label, color in [
        (lr_proba_test,  f'LR  (AUPRC={lr_auprc:.4f})',  'steelblue'),
        (mlp_proba_test, f'MLP (AUPRC={mlp_auprc:.4f})', 'forestgreen'),
    ]:
        prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec_c, prec_c, label=label, color=color)
    ax.axhline(pos_rate, color='gray', linestyle='--', linewidth=0.8,
               label=f'No-skill ({pos_rate:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — cld_dist_km < 10')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Calibration ──────────────────────────────────────────────────────
    n_bins = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, proba, label, color, brier in [
        (axes[0], lr_proba_test,  'LR',  'steelblue',   lr_brier),
        (axes[1], mlp_proba_test, 'MLP', 'forestgreen', mlp_brier),
    ]:
        bin_edges  = np.linspace(0, 1, n_bins + 1)
        bin_ids    = np.digitize(proba, bin_edges[1:-1])
        mean_pred  = np.array([proba[bin_ids == i].mean() if (bin_ids == i).any()
                               else (bin_edges[i] + bin_edges[i+1]) / 2
                               for i in range(n_bins)])
        frac_pos   = np.array([y_test[bin_ids == i].mean() if (bin_ids == i).any()
                               else np.nan
                               for i in range(n_bins)])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', color=color, label=f'{label} (Brier={brier:.4f})')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'{label} Calibration Curve')
        ax.legend(fontsize=9)
    fig.suptitle('Reliability Diagrams (cld_dist_km < 10)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Score distribution ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, proba, label, color in [
        (axes[0], lr_proba_test,  'LR',  'steelblue'),
        (axes[1], mlp_proba_test, 'MLP', 'forestgreen'),
    ]:
        bins = np.linspace(0, 1, 51)
        ax.hist(proba[y_test == 0], bins=bins, color='royalblue', alpha=0.6,
                density=True, label='Clear (y=0)')
        ax.hist(proba[y_test == 1], bins=bins, color='tomato',   alpha=0.6,
                density=True, label='Cloud (y=1)')
        ax.axvline(0.5, color='k', linestyle='--', linewidth=0.8, label='Threshold 0.5')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} — Score Distribution')
        ax.legend(fontsize=9)
    fig.suptitle('Probability Score Distributions (cld_dist_km < 10)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_score_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Spatial map ──────────────────────────────────────────────────────
    if 'lon' in df.columns and 'lat' in df.columns:
        plot_lon = np.array(df['lon'], dtype=float)
        if plot_lon.min() < -90 and plot_lon.max() > 90:
            plot_lon = np.where(plot_lon < 0, plot_lon + 360, plot_lon)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, proba, label in [
            (axes[0], lr_proba_all,  'LR'),
            (axes[1], mlp_proba_all, 'MLP'),
        ]:
            _lon, _lat, _p = _scatter_ss(rng_plot, plot_lon, df['lat'].to_numpy(), proba)
            sc = ax.scatter(_lon, _lat, c=_p, cmap='RdYlGn_r', vmin=0, vmax=1,
                            s=8, alpha=0.7, rasterized=True)
            fig.colorbar(sc, ax=ax, label='P(cld_dist < 10 km)')
            ax.set_title(f'{label} — Predicted Cloud Proximity Probability')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'clf_spatial_map.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Plot: Feature importance ───────────────────────────────────────────────
    _ckpt("feature importance: LR coef")
    lr_model_obj = lr_adapter.model
    lr_std_importance = np.abs(lr_model_obj.coef_[0]) * X_train.std(axis=0)

    # MLP permutation importance (subsampled)
    _ckpt("feature importance: MLP permutation")
    rng_pi  = np.random.default_rng(42)
    pi_n    = min(5000, X_test.shape[0])
    pi_idx  = rng_pi.choice(X_test.shape[0], size=pi_n, replace=False)
    X_pi    = X_test[pi_idx]
    y_pi    = y_test[pi_idx]

    def _mlp_auroc_fn(X_eval):
        proba = mlp_adapter.predict_proba(X_eval)
        try:
            return roc_auc_score(y_pi, proba)
        except Exception:
            return 0.5

    baseline_auroc = _mlp_auroc_fn(X_pi)
    fp_skip = frozenset(i for i, f in enumerate(features) if f.startswith('fp_') and f[3:].isdigit())
    perm_imp_mlp = np.zeros(len(features))
    rng_inner = np.random.default_rng(0)
    for col in range(len(features)):
        if col in fp_skip:
            continue
        drops = np.zeros(5)
        for r in range(5):
            X_shuf = X_pi.copy()
            X_shuf[:, col] = rng_inner.permutation(X_shuf[:, col])
            drops[r] = baseline_auroc - _mlp_auroc_fn(X_shuf)
            del X_shuf
        perm_imp_mlp[col] = drops.mean()

    _ckpt("feature importance: done")

    top_n  = min(25, len(features))
    imp_df = pd.DataFrame({
        'feature':          features,
        'lr_std_coef':      lr_std_importance,
        'mlp_perm_auroc':   perm_imp_mlp,
    }).sort_values('mlp_perm_auroc', ascending=False)
    imp_df.to_csv(os.path.join(output_dir, 'feature_importance_clf.csv'), index=False)

    top_df = imp_df.head(top_n).iloc[::-1]
    fig, (ax_lr, ax_mlp) = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.32)))
    ax_lr.barh(top_df['feature'], top_df['lr_std_coef'],    color='steelblue')
    ax_lr.set_xlabel('|coef| × std(X_train)')
    ax_lr.set_title(f'LR — Feature Importance (cld_dist < 10)')
    ax_mlp.barh(top_df['feature'], top_df['mlp_perm_auroc'], color='forestgreen')
    ax_mlp.set_xlabel('Permutation AUROC drop')
    ax_mlp.set_title(f'MLP — Permutation Importance (cld_dist < 10)')
    fig.suptitle(f'Feature Importance (top {top_n})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    _ckpt("cloud_proximity_classification: complete")
    print(f"\n  Results saved → {output_dir}", flush=True)


# ─── CLI entry point ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    parser = argparse.ArgumentParser(
        description="LR + MLP binary classifier for cld_dist_km < 10"
    )
    parser.add_argument('--sfc_type', type=int, default=0,
                        help="Surface type filter (default: 0 = ocean only)")
    parser.add_argument('--suffix', type=str, default='',
                        help="Subfolder under results/model_mlp_clf/")
    parser.add_argument('--pipeline', type=str, default=None,
                        help="Path to a saved FeaturePipeline (.pkl). "
                             "If omitted, a new pipeline is fitted and saved.")
    args = parser.parse_args()

    storage_dir = get_storage_dir()

    if platform.system() == 'Linux':
        data_name = 'combined_2016_2020_dates.parquet'
    else:
        data_name = 'combined_2020-02-01_all_orbits.parquet'

    data_path  = storage_dir / 'results/csv_collection' / data_name
    base_dir   = storage_dir / 'results/model_mlp_clf'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path}", flush=True)
    df = (pd.read_parquet(data_path) if str(data_path).endswith('.parquet')
          else pd.read_csv(data_path))
    print(f"  Total rows: {len(df):,}", flush=True)

    df = df[df['sfc_type'] == args.sfc_type]
    df = df[df['snow_flag'] == 0]
    df = df.dropna(subset=['cld_dist_km'])
    df = df.reset_index(drop=True)
    print(f"  Rows after filters (sfc_type={args.sfc_type}, snow_flag=0, cld_dist_km valid): "
          f"{len(df):,}", flush=True)

    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=args.sfc_type)
        pipeline_path = output_dir / 'pipeline.pkl'
        pipeline.save(pipeline_path)
        print(f"  Fitted and saved pipeline → {pipeline_path}", flush=True)

    cloud_proximity_classification(df, output_dir=output_dir, pipeline=pipeline)


if __name__ == '__main__':
    main()
