"""transformer_cloud_classifier.py — FT-Transformer binary classifier for cloud proximity.

Trains a FT-Transformer (UncertainFTTransformerRefined architecture with a
single-logit classification head) to predict whether a sounding's nearest cloud
is within 10 km, and outputs a calibrated confidence (probability).

Produces the same evaluation suite as mlp_lr_cloud_classifier.py, plus the
attention-map diagnostics available from the FT-Transformer architecture.

Usage
-----
    python src/transformer_cloud_classifier.py [--sfc_type 0] [--suffix my_run] [--pipeline path.pkl]

Outputs (in results/model_ft_clf/<suffix>/):
    ft_clf_weights.pt          — FTClassifier weights (CPU)
    ft_clf_meta.pkl            — FTClassifier architecture + threshold meta
    pipeline.pkl               — fitted FeaturePipeline (if not supplied)
    clf_learning_curve.png
    clf_roc_curve.png
    clf_pr_curve.png
    clf_calibration.png
    clf_score_distribution.png
    clf_spatial_map.png
    clf_feature_importance.png
    attention_top_features.png
    attention_group_heatmap.png
    attention_per_head.png
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
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model_adapters import FTClassifierAdapter
from models_transformer import (
    AdvancedTransformerBlock, GatedResidualNetwork,
    MLPTokenizer, FeatureTokenizer,
    _build_feature_to_group, _FEATURE_GROUPS,
    plot_attention_map,
)
from pipeline import FeaturePipeline
from utils import get_storage_dir

logger = logging.getLogger(__name__)

_SCATTER_MAX = 150_000


# ─── FT-Transformer classification model ───────────────────────────────────────

class FTClassifier(nn.Module):
    """FT-Transformer with a [CLS] token and a single-logit binary classification head.

    Architecture mirrors UncertainFTTransformerRefined exactly up to the output
    head, which is replaced with GRN → Linear(1) instead of GRN → Linear(3).

    forward() returns raw logits [batch] (no sigmoid).  Use BCEWithLogitsLoss
    during training and torch.sigmoid() at inference.
    """

    def __init__(self, n_features: int, d_token: int = 128, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 256,
                 tokenizer_type: str = 'mlp', drop_path_rate: float = 0.15,
                 feature_names: 'list | None' = None):
        super().__init__()
        self.n_features     = n_features
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == 'mlp':
            self.tokenizer = MLPTokenizer(n_features, d_token)
        else:
            self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Segment embeddings — same as UncertainFTTransformerRefined
        if feature_names is not None:
            n_groups = len(_FEATURE_GROUPS)
            self.group_emb = nn.Embedding(n_groups, d_token)
            nn.init.trunc_normal_(self.group_emb.weight, std=0.02)
            self.register_buffer(
                'feature_to_group',
                _build_feature_to_group(list(feature_names)),
            )
        else:
            self.group_emb = None

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Stochastic depth: scale drop rate linearly across layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(d_token, n_heads, d_ff, drop_path_rate=dpr[i])
            for i in range(n_layers)
        ])

        # Classification head: GRN on [CLS] representation → single logit
        self.head = nn.Sequential(
            GatedResidualNetwork(
                input_size=d_token, hidden_size=d_token,
                output_size=d_token // 2, dropout=0.1,
            ),
            nn.Linear(d_token // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [batch, n_features]
        return_attn: if True, return (logits, last_layer_attn_weights)

        Returns
        -------
        logits : [batch]  (raw, no sigmoid)
        """
        x = self.tokenizer(x)   # [batch, n_features, d_token]

        if self.group_emb is not None:
            x = x + self.group_emb(self.feature_to_group).unsqueeze(0)

        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # [batch, n_features+1, d_token]

        last_attn = None
        for layer in self.layers:
            if return_attn:
                x_norm = layer.norm1(x)
                attn_out, last_attn = layer.attn(x_norm, x_norm, x_norm)
                x = x + attn_out
                x = x + layer.ff(layer.norm2(x))
            else:
                x = layer(x, training=self.training)

        x_global = x[:, 0]                   # [batch, d_token]  — [CLS] token
        logits = self.head(x_global).squeeze(-1)   # [batch]

        if return_attn:
            return logits, last_attn
        return logits


# ─── Helpers ────────────────────────────────────────────────────────────────────

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


# ─── Main classification function ───────────────────────────────────────────────

def cloud_proximity_classification_ft(df: pd.DataFrame, output_dir,
                                       pipeline: FeaturePipeline):
    """Train FTClassifier for cld_dist_km < 10 and produce all evaluation plots.

    Parameters
    ----------
    df         : DataFrame filtered by sfc_type and snow_flag
    output_dir : Path or str — directory to write checkpoints and plots
    pipeline   : fitted FeaturePipeline
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    t0 = datetime.now()
    def _ckpt(label):
        _checkpoint_fn(label, t0)

    _ckpt("cloud_proximity_classification_ft: entry")

    # ── Device selection ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}", flush=True)

    # ── Target ────────────────────────────────────────────────────────────────
    cld_dist   = df['cld_dist_km'].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(cld_dist)
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"[warn] Dropping {n_dropped} rows with non-finite cld_dist_km")
        df       = df[valid_mask].reset_index(drop=True)
        cld_dist = cld_dist[valid_mask]

    y_all    = (cld_dist < 10.0).astype(np.float32)
    n_pos    = int(y_all.sum())
    n_neg    = int((~y_all.astype(bool)).sum())
    pos_rate = n_pos / len(y_all)
    print(f"  Class distribution: cloud (cld<10 km) = {n_pos:,} ({pos_rate:.1%}), "
          f"clear = {n_neg:,} ({1-pos_rate:.1%})", flush=True)
    pos_weight = n_neg / max(n_pos, 1)

    # ── Feature transform ──────────────────────────────────────────────────────
    _ckpt("before pipeline.transform")
    X_all    = pipeline.transform(df).astype(np.float32, copy=False)
    features = pipeline.features

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

    # ── FT Classifier ──────────────────────────────────────────────────────────
    _ckpt("before FTClassifier init / training")
    n_features = X_train.shape[1]

    d_token  = 128
    n_heads  = 8
    n_layers = 4
    d_ff     = 256

    if FTClassifierAdapter.can_load(output_dir):
        print("  [checkpoint] FTClassifierAdapter found → loading", flush=True)
        ft_adapter   = FTClassifierAdapter.load(output_dir, device=device)
        ft_clf       = ft_adapter.model.to(device)
        train_losses, val_losses = [], []
    else:
        ft_clf = FTClassifier(
            n_features=n_features, d_token=d_token, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, tokenizer_type='mlp',
            feature_names=features,
        ).to(device)

        n_params = sum(p.numel() for p in ft_clf.parameters() if p.requires_grad)
        print(f"  FTClassifier parameters: {n_params:,}  |  train: {len(X_train):,}  "
              f"|  ratio: {len(X_train)/n_params:.2f}", flush=True)

        pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4096, shuffle=True
        )

        n_epochs = 100 if platform.system() == "Darwin" else 300
        optimizer = torch.optim.AdamW(ft_clf.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )

        def _val_metrics(model_):
            model_.eval()
            all_logits = []
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(X_test), 4096):
                    Xb = torch.tensor(X_test[start:start + 4096],
                                      dtype=torch.float32).to(device)
                    yb = torch.tensor(y_test[start:start + 4096],
                                      dtype=torch.float32).to(device)
                    logits = model_(Xb)
                    total_loss += criterion(logits, yb).item() * len(Xb)
                    all_logits.append(torch.sigmoid(logits).cpu().numpy())
                    n += len(Xb)
                    del Xb, yb
            proba = np.concatenate(all_logits)
            bce   = total_loss / n
            try:
                auroc = roc_auc_score(y_test, proba)
            except Exception:
                auroc = 0.5
            return bce, auroc

        train_losses, val_losses = [], []
        best_val_loss, best_state, patience, no_improve = float('inf'), None, 30, 0

        epoch_bar = tqdm(range(n_epochs), desc="FT clf training", unit="epoch")
        for epoch in epoch_bar:
            ft_clf.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(ft_clf(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(ft_clf.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            val_bce, val_auroc = _val_metrics(ft_clf)
            train_losses.append(train_loss)
            val_losses.append(val_bce)

            improved = val_bce < best_val_loss
            if improved:
                best_val_loss = val_bce
                best_state    = copy.deepcopy(ft_clf.state_dict())
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

        ft_clf.load_state_dict(best_state)
        ft_clf.eval()

        FTClassifierAdapter(
            ft_clf.cpu(), n_features=n_features, d_token=d_token,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            tokenizer_type='mlp', feature_names=features,
            threshold=0.5, device=device,
        ).save(output_dir)
        ft_adapter = FTClassifierAdapter.load(output_dir, device=device)
        ft_clf = ft_adapter.model.to(device)

        del train_ds, train_loader, best_state
        gc.collect()

    _ckpt("after FT training")

    # ── Inference on test set ──────────────────────────────────────────────────
    ft_proba_test = ft_adapter.predict_proba(X_test)
    ft_pred_test  = (ft_proba_test >= 0.5).astype(np.int8)
    ft_auroc = roc_auc_score(y_test, ft_proba_test)
    ft_auprc = average_precision_score(y_test, ft_proba_test)
    ft_brier = brier_score_loss(y_test, ft_proba_test)
    ft_f1    = f1_score(y_test, ft_pred_test)
    ft_acc   = accuracy_score(y_test, ft_pred_test)
    ft_prec  = precision_score(y_test, ft_pred_test, zero_division=0)
    ft_rec   = recall_score(y_test, ft_pred_test, zero_division=0)
    print(f"  FT  AUROC={ft_auroc:.4f}  AUPRC={ft_auprc:.4f}  "
          f"F1={ft_f1:.4f}  Brier={ft_brier:.4f}", flush=True)
    print(f"  FT  Acc={ft_acc:.4f}  Prec={ft_prec:.4f}  Rec={ft_rec:.4f}", flush=True)
    cm = confusion_matrix(y_test, ft_pred_test)
    print(f"  FT  Confusion matrix:\n{cm}", flush=True)

    # Full-dataset predictions for spatial map
    ft_proba_all = ft_adapter.predict_proba(X_all)

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
        ax.set_title('FT Classifier — Learning Curve')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'clf_learning_curve.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Plot: ROC curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_test, ft_proba_test)
    ax.plot(fpr, tpr, label=f'FT  (AUROC={ft_auroc:.4f})', color='darkorange')
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
    prec_c, rec_c, _ = precision_recall_curve(y_test, ft_proba_test)
    ax.plot(rec_c, prec_c, label=f'FT  (AUPRC={ft_auprc:.4f})', color='darkorange')
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
    fig, ax = plt.subplots(figsize=(7, 5))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids   = np.digitize(ft_proba_test, bin_edges[1:-1])
    mean_pred = np.array([ft_proba_test[bin_ids == i].mean()
                          if (bin_ids == i).any()
                          else (bin_edges[i] + bin_edges[i + 1]) / 2
                          for i in range(n_bins)])
    frac_pos  = np.array([y_test[bin_ids == i].mean()
                          if (bin_ids == i).any() else np.nan
                          for i in range(n_bins)])
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', color='darkorange',
            label=f'FT (Brier={ft_brier:.4f})')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('FT Classifier — Reliability Diagram (cld_dist_km < 10)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Score distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(ft_proba_test[y_test == 0], bins=bins, color='royalblue', alpha=0.6,
            density=True, label='Clear (y=0)')
    ax.hist(ft_proba_test[y_test == 1], bins=bins, color='tomato', alpha=0.6,
            density=True, label='Cloud (y=1)')
    ax.axvline(0.5, color='k', linestyle='--', linewidth=0.8, label='Threshold 0.5')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('FT Classifier — Score Distribution (cld_dist_km < 10)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_score_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Spatial map ──────────────────────────────────────────────────────
    if 'lon' in df.columns and 'lat' in df.columns:
        plot_lon = np.array(df['lon'], dtype=float)
        if plot_lon.min() < -90 and plot_lon.max() > 90:
            plot_lon = np.where(plot_lon < 0, plot_lon + 360, plot_lon)

        fig, ax = plt.subplots(figsize=(10, 5))
        _lon, _lat, _p = _scatter_ss(rng_plot, plot_lon, df['lat'].to_numpy(), ft_proba_all)
        sc = ax.scatter(_lon, _lat, c=_p, cmap='RdYlGn_r', vmin=0, vmax=1,
                        s=8, alpha=0.7, rasterized=True)
        fig.colorbar(sc, ax=ax, label='P(cld_dist < 10 km)')
        ax.set_title('FT Classifier — Predicted Cloud Proximity Probability')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'clf_spatial_map.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Plot: Permutation feature importance ───────────────────────────────────
    _ckpt("feature importance: FT permutation")
    rng_pi = np.random.default_rng(42)
    pi_n   = min(5000, X_test.shape[0])
    pi_idx = rng_pi.choice(X_test.shape[0], size=pi_n, replace=False)
    X_pi   = X_test[pi_idx]
    y_pi   = y_test[pi_idx]

    def _ft_auroc_fn(X_eval):
        proba = ft_adapter.predict_proba(X_eval)
        try:
            return roc_auc_score(y_pi, proba)
        except Exception:
            return 0.5

    baseline_auroc = _ft_auroc_fn(X_pi)
    fp_skip        = frozenset(i for i, f in enumerate(features)
                               if f.startswith('fp_') and f[3:].isdigit())
    perm_imp       = np.zeros(len(features))
    rng_inner      = np.random.default_rng(0)
    for col in range(len(features)):
        if col in fp_skip:
            continue
        drops = np.zeros(5)
        for r in range(5):
            X_shuf          = X_pi.copy()
            X_shuf[:, col]  = rng_inner.permutation(X_shuf[:, col])
            drops[r]        = baseline_auroc - _ft_auroc_fn(X_shuf)
            del X_shuf
        perm_imp[col] = drops.mean()

    _ckpt("feature importance: done")

    top_n  = min(25, len(features))
    imp_df = pd.DataFrame({
        'feature':         features,
        'ft_perm_auroc':   perm_imp,
    }).sort_values('ft_perm_auroc', ascending=False)
    imp_df.to_csv(os.path.join(output_dir, 'feature_importance_clf.csv'), index=False)

    top_df = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.32)))
    ax.barh(top_df['feature'], top_df['ft_perm_auroc'], color='darkorange')
    ax.set_xlabel('Permutation AUROC drop')
    ax.set_title(f'FT Classifier — Permutation Feature Importance (top {top_n}, cld_dist < 10)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'clf_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: Attention maps ───────────────────────────────────────────────────
    _ckpt("attention maps")
    ft_clf.eval()
    n_viz        = min(200, len(X_test))
    sample_batch = torch.tensor(X_test[:n_viz], dtype=torch.float32)
    # plot_attention_map expects a model that supports return_attn=True via the
    # same interface as UncertainFTTransformerRefined.  FTClassifier provides
    # an identical forward signature, so we can pass it directly.
    plot_attention_map(ft_clf.cpu(), sample_batch, features, output_dir)

    _ckpt("cloud_proximity_classification_ft: complete")
    print(f"\n  Results saved → {output_dir}", flush=True)


# ─── CLI entry point ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    parser = argparse.ArgumentParser(
        description="FT-Transformer binary classifier for cld_dist_km < 10"
    )
    parser.add_argument('--sfc_type', type=int, default=0,
                        help="Surface type filter (default: 0 = ocean only)")
    parser.add_argument('--suffix', type=str, default='',
                        help="Subfolder under results/model_ft_clf/")
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
    base_dir   = storage_dir / 'results/model_ft_clf'
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

    cloud_proximity_classification_ft(df, output_dir=output_dir, pipeline=pipeline)


if __name__ == '__main__':
    main()
