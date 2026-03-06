"""models_hybrid.py — Dual-Tower Hybrid (MLP anchor + FT-Transformer) for XCO2 bias correction.

Architecture: HybridDualTower
  Input [N, n_features]
    ├─ MLP Branch  → repr [N, fusion_dim]
    │    Linear(n→256) → LN → GELU
    │    Linear(256→256) → LN → GELU → Dropout(0.15)
    │    Linear(256→fusion_dim) → LN → GELU
    └─ FT-Transformer Branch  → CLS [N, d_token] → project [N, fusion_dim]
         MLPTokenizer → GroupEmbeddings → [CLS] prepend
         → N × AdvancedTransformerBlock (PreNorm + MHA + GRN-FFN + DropPath)
         → extract CLS token
  Fusion [N, 2*fusion_dim]  ← Concat([mlp_repr, ft_repr])
    GRN(2*fusion_dim → fusion_dim)
    Linear(fusion_dim → 3)  ← [q05, q50, q95]

CLI usage:
    python src/models_hybrid.py \\
        --sfc_type 0 --suffix ocean_2016_2020 \\
        --pipeline results/train_data/pipeline_ocean_2016_2020.pkl

See log/hybrid_model_plan.md for the full architectural rationale.
"""

import argparse
import copy
import gc
import logging
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pipeline import FeaturePipeline, _ensure_derived_features
from model_adapters import HybridAdapter
from models_transformer import (
    UncertainFTTransformerRefined,
    GatedResidualNetwork,
    AdvancedTransformerBlock,
    huber_pinball_loss,
    quantile_loss,
    _batched_predict,
    plot_permutation_importance,
    _FEATURE_GROUPS,
)
from utils import get_storage_dir

logger = logging.getLogger(__name__)

_QUANTILES = [0.05, 0.5, 0.95]


# ─── MLP branch ────────────────────────────────────────────────────────────────

class _MLPBranch(nn.Module):
    """Deep MLP anchor: Linear projections with LayerNorm + GELU activations.

    Provides a stable, regularised first-order path that acts as an anchor for
    the fusion head.  Even if the Transformer branch collapses during training,
    this path ensures a reasonable baseline prediction is always available.
    """

    def __init__(self, n_in: int, hidden: int = 256, out_dim: int = 128,
                 dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Hybrid Dual-Tower model ───────────────────────────────────────────────────

class HybridDualTower(nn.Module):
    """Parallel MLP anchor + FT-Transformer with GRN fusion head.

    The FT-Transformer backbone (tokenizer, group embeddings, [CLS] token,
    Transformer layers) is taken from UncertainFTTransformerRefined but its
    regression head is not used.  Only the [CLS] token representation is
    extracted and projected to fusion_dim before being concatenated with the
    MLP branch output.

    Output: [batch, 3] — (q05, q50, q95) quantiles, compatible with
    quantile_loss and huber_pinball_loss from models_transformer.py.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        mlp_hidden: int = 256,
        fusion_dim: int = 128,
        tokenizer_type: str = 'mlp',
        drop_path_rate: float = 0.15,
        feature_names: list | None = None,
    ):
        super().__init__()
        self.n_features  = n_features
        self.d_token     = d_token
        self.fusion_dim  = fusion_dim

        # ── MLP branch ────────────────────────────────────────────────────────
        self.mlp_branch = _MLPBranch(n_features, hidden=mlp_hidden, out_dim=fusion_dim)

        # ── FT-Transformer backbone (head stripped below) ─────────────────────
        _ft = UncertainFTTransformerRefined(
            n_features=n_features,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            tokenizer_type=tokenizer_type,
            drop_path_rate=drop_path_rate,
            feature_names=feature_names,
        )
        # Extract the backbone components; discard the regression head so the
        # hybrid's fusion head owns the only output projection.
        self.ft_tokenizer    = _ft.tokenizer
        self.ft_group_emb    = _ft.group_emb      # may be None
        self.ft_cls_token    = _ft.cls_token
        self.ft_layers       = _ft.layers
        if hasattr(_ft, 'feature_to_group'):
            self.register_buffer('feature_to_group', _ft.feature_to_group)
        else:
            self.feature_to_group = None
        del _ft

        # Project CLS token [d_token] → [fusion_dim]
        self.ft_proj = nn.Linear(d_token, fusion_dim)

        # ── Fusion ────────────────────────────────────────────────────────────
        # GRN adaptively weights the two branches per sample.
        self.fusion = GatedResidualNetwork(
            input_size=2 * fusion_dim,
            hidden_size=2 * fusion_dim,
            output_size=fusion_dim,
            dropout=0.2,
        )

        # ── Regression head ───────────────────────────────────────────────────
        # Single shared head for [q05, q50, q95]; both branches train together.
        self.head = nn.Linear(fusion_dim, 3)

    # ── Internal FT backbone encode (no head) ─────────────────────────────────

    def _ft_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the FT backbone up to CLS extraction. Returns [batch, d_token]."""
        xf = self.ft_tokenizer(x)                               # [batch, n_feat, d_token]

        if self.ft_group_emb is not None and self.feature_to_group is not None:
            xf = xf + self.ft_group_emb(self.feature_to_group).unsqueeze(0)

        b = xf.shape[0]
        cls = self.ft_cls_token.expand(b, -1, -1)
        xf  = torch.cat((cls, xf), dim=1)                       # [batch, n_feat+1, d_token]

        for layer in self.ft_layers:
            xf = layer(xf, training=self.training)

        return xf[:, 0]                                          # [batch, d_token]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [batch, n_features]  QT-transformed float32

        Returns
        -------
        out : [batch, 3]  (q05, q50, q95)
        """
        mlp_repr = self.mlp_branch(x)                           # [batch, fusion_dim]
        ft_repr  = self.ft_proj(self._ft_encode(x))             # [batch, fusion_dim]
        fused    = self.fusion(torch.cat([mlp_repr, ft_repr], dim=-1))  # [batch, fusion_dim]
        return self.head(fused)                                  # [batch, 3]


# ─── Training ──────────────────────────────────────────────────────────────────

def train_hybrid_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: list,
    output_dir: str = ".",
    d_token: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 256,
    mlp_hidden: int = 256,
    fusion_dim: int = 128,
    batch_size: int = 1024,
    n_epochs: int = 500,
    log_every: int = 10,
    patience: int | None = 50,
    loss_fn: str = 'huber',
    huber_delta: float = 1.0,
) -> HybridDualTower:
    """Train a HybridDualTower model. Returns the best model on CPU.

    Checkpoint is saved to <output_dir>/model_hybrid_best.pt.
    """
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pin      = device.type in ("cuda", "mps")
    n_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=n_workers,
                              persistent_workers=n_workers > 0, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=n_workers,
                              persistent_workers=n_workers > 0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridDualTower(
        n_features=len(features),
        d_token=d_token, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        mlp_hidden=mlp_hidden, fusion_dim=fusion_dim,
        feature_names=features,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=n_epochs * len(train_loader),
        pct_start=0.05,
        div_factor=25,
        final_div_factor=1000,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if loss_fn == 'huber':
        def _loss(preds, targets):
            return huber_pinball_loss(preds, targets, _QUANTILES, delta=huber_delta) + 0.05 * preds[:, 1].abs().mean()
    else:
        def _loss(preds, targets):
            return quantile_loss(preds, targets, _QUANTILES) + 0.05 * preds[:, 1].abs().mean()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_path = os.path.join(output_dir, 'model_hybrid_best.pt')
    tqdm.write("=" * 60)
    tqdm.write("  HybridDualTower — training config")
    tqdm.write("=" * 60)
    tqdm.write(f"  Device      : {device}")
    tqdm.write(f"  Features    : {len(features)}")
    tqdm.write(f"  d_token     : {d_token}  n_heads: {n_heads}  n_layers: {n_layers}  d_ff: {d_ff}")
    tqdm.write(f"  mlp_hidden  : {mlp_hidden}  fusion_dim: {fusion_dim}")
    tqdm.write(f"  Params      : {n_params:,}")
    tqdm.write(f"  Train size  : {len(train_ds):,}  |  Val size: {len(val_ds):,}")
    tqdm.write(f"  Batch size  : {batch_size}  |  Epochs: {n_epochs}")
    tqdm.write(f"  Early stop  : {'disabled' if patience is None else f'patience={patience}'}")
    tqdm.write(f"  LR schedule : OneCycleLR(max_lr=5e-4, warmup=5%, total_steps={n_epochs * len(train_loader):,})")
    tqdm.write(f"  Loss        : {loss_fn}" + (f"  (delta={huber_delta})" if loss_fn == 'huber' else ""))
    tqdm.write(f"  Checkpoint  : {ckpt_path}")
    tqdm.write("=" * 60)

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    train_history     = []

    epoch_bar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss    = 0.0
        grad_norm_sum = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,
                                enabled=(device.type in ("cuda", "mps"))):
                preds = model(batch_x)
                loss  = _loss(preds, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss    += loss.item()
            grad_norm_sum += grad_norm.item()

        model.eval()
        val_loss       = 0.0
        val_preds_q50  = []
        val_targets    = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds     = model(batch_x)
                val_loss += _loss(preds, batch_y).item()
                val_preds_q50.append(preds[:, 1].cpu())
                val_targets.append(batch_y.cpu())

        avg_train  = train_loss    / len(train_loader)
        avg_val    = val_loss      / len(val_loader)
        avg_gnorm  = grad_norm_sum / len(train_loader)

        q50_np  = torch.cat(val_preds_q50).numpy()
        tgt_np  = torch.cat(val_targets).numpy()
        val_mae = float(np.abs(q50_np - tgt_np).mean())
        val_r2  = float(1.0 - np.sum((tgt_np - q50_np) ** 2) /
                        np.sum((tgt_np - tgt_np.mean()) ** 2))

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss     = avg_val
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_mae":  val_mae,
                "val_r2":   val_r2,
            }, ckpt_path)
            tqdm.write(f"  [epoch {epoch:4d}] ✓ saved  "
                       f"val={avg_val:.5f}  MAE={val_mae:.4f}  R²={val_r2:.4f}")
            logger.info("Epoch %d: new best val_loss=%.5f MAE=%.4f R²=%.4f",
                        epoch, avg_val, val_mae, val_r2)
        else:
            epochs_no_improve += 1

        train_history.append((epoch, avg_train, avg_val))

        epoch_bar.set_postfix(
            train=f"{avg_train:.5f}",
            val=f"{avg_val:.5f}",
            best=f"{best_val_loss:.5f}",
            MAE=f"{val_mae:.4f}",
            R2=f"{val_r2:.4f}",
            gnorm=f"{avg_gnorm:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            saved="✓" if improved else "",
        )

        if patience is not None and epochs_no_improve >= patience:
            tqdm.write(f"  [epoch {epoch:4d}] Early stopping — "
                       f"no improvement for {patience} epochs.")
            break

        if (epoch + 1) % log_every == 0 or epoch == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            tqdm.write(
                f"  [{ts}] epoch {epoch+1:4d}/{n_epochs}  "
                f"train={avg_train:.5f}  val={avg_val:.5f}  "
                f"best={best_val_loss:.5f}  MAE={val_mae:.4f}  R²={val_r2:.4f}  "
                f"gnorm={avg_gnorm:.3f}"
            )
            logger.info(
                "Epoch %d/%d  train=%.5f  val=%.5f  best=%.5f  MAE=%.4f  R2=%.4f",
                epoch + 1, n_epochs, avg_train, avg_val, best_val_loss, val_mae, val_r2,
            )

    # ── Restore best checkpoint ────────────────────────────────────────────────
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    tqdm.write("=" * 60)
    tqdm.write(f"  Training complete.  Best epoch={best_ckpt['epoch']}  "
               f"val_loss={best_ckpt['val_loss']:.5f}  "
               f"MAE={best_ckpt.get('val_mae', float('nan')):.4f}  "
               f"R²={best_ckpt.get('val_r2', float('nan')):.4f}")
    tqdm.write("=" * 60)

    # ── Learning curve ────────────────────────────────────────────────────────
    epochs_ran = [e for e, _, _ in train_history]
    train_vals = [t for _, t, _ in train_history]
    val_vals   = [v for _, _, v in train_history]
    best_ep    = int(np.argmin(val_vals)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_ran, train_vals, label='Train', color='steelblue')
    ax.plot(epochs_ran, val_vals,   label='Val',   color='tomato')
    ax.axvline(best_ep - 1, color='gray', linestyle='--', linewidth=0.8,
               label=f'Best epoch {best_ep}')
    ax.set_xlabel('Epoch');  ax.set_ylabel('Loss')
    ax.set_title('HybridDualTower — Learning Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hybrid_learning_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    return model.cpu()


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_hybrid(model: HybridDualTower,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    output_dir: str) -> None:
    """Scatter plot of true vs predicted (q50) with uncertainty intervals."""
    preds = _batched_predict(model, np.asarray(X_test, dtype=np.float32))
    q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]

    lower_err = np.clip(q50 - q05, 0, None)
    upper_err = np.clip(q95 - q50, 0, None)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae  = mean_absolute_error(y_test, q50)
    r2   = r2_score(y_test, q50)
    slope, intercept = np.polyfit(y_test, q50, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(y_test, q50, yerr=np.array([lower_err, upper_err]),
                fmt='o', alpha=0.5, markersize=3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("True XCO2_bc anomaly (ppm)")
    ax.set_ylabel("Predicted XCO2_bc anomaly (ppm)")
    ax.set_title("HybridDualTower — val set pred vs true")
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.05, 0.95, f"R²: {r2:.3f}\nMAE: {mae:.4f}\nSlope: {slope:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hybrid_pred_vs_true.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved hybrid_pred_vs_true.png  R²=%.4f  MAE=%.4f", r2, mae)


def plot_hybrid_evaluation_by_regime(model: HybridDualTower,
                                     df: pd.DataFrame,
                                     pipeline: FeaturePipeline,
                                     output_dir: str) -> None:
    """3×5 regime comparison plot (mirrors models_transformer.plot_evaluation_by_regime).

    Reuses the FT-Transformer version from models_transformer.py by passing the
    hybrid model directly — HybridDualTower.forward() returns [N, 3] quantiles,
    which is the same interface.
    """
    from models_transformer import plot_evaluation_by_regime
    plot_evaluation_by_regime(
        model=model,
        df=df,
        qt=pipeline.qt,
        features=pipeline.features,
        output_dir=output_dir,
    )
    # Rename output file so it doesn't collide with the FT output
    src = Path(output_dir) / 'ft_evaluation_by_regime.png'
    dst = Path(output_dir) / 'hybrid_evaluation_by_regime.png'
    if src.exists():
        src.rename(dst)
        logger.info("Renamed ft_evaluation_by_regime.png → hybrid_evaluation_by_regime.png")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HybridDualTower (MLP + FT-Transformer) XCO2 bias correction"
    )
    parser.add_argument('--sfc_type', type=int, default=0,
                        help="Surface type (0=ocean, 1=land, 2=sea-ice)")
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder under results/model_hybrid/ (e.g. ocean_2016_2020)')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a pre-fitted FeaturePipeline (.pkl).  '
                             'If not supplied, fitted from training data and saved.')
    parser.add_argument('--loss', type=str, default='huber',
                        choices=['quantile', 'huber'],
                        help='Loss for q50: huber (default) or quantile (pinball).')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Huber δ (ppm). Only used when --loss huber. Default: 1.0.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    storage_dir = get_storage_dir()
    fdir        = storage_dir / 'results/csv_collection'
    if platform.system() == "Linux":
        data_name = 'combined_2016_2020_dates.parquet'
    else:
        data_name = 'combined_2020-02-01_all_orbits.parquet'

    base_dir   = storage_dir / 'results/model_hybrid'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Surface type: %d  output_dir: %s", args.sfc_type, output_dir)

    # ── Load data ──────────────────────────────────────────────────────────────
    _dp = str(fdir / data_name)
    df  = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df  = df[df['sfc_type'] == args.sfc_type]
    df  = df[df['snow_flag'] == 0]

    # ── Pipeline: load or fit ──────────────────────────────────────────────────
    pipeline_path = output_dir / 'pipeline.pkl'
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    elif pipeline_path.exists():
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline.fit(df, sfc_type=args.sfc_type)
        pipeline.save(pipeline_path)
        print(f"  Fitted and saved pipeline → {pipeline_path}", flush=True)

    features = pipeline.features

    # ── Transform + split ──────────────────────────────────────────────────────
    df = _ensure_derived_features(df)
    missing_fp = [i for i in range(8) if f'fp_{i}' not in df.columns]
    if missing_fp:
        df = pd.concat(
            [df, pd.DataFrame(
                {f'fp_{i}': (df['fp'] == i).astype(np.float32) for i in missing_fp},
                index=df.index,
            )],
            axis=1,
        )

    valid_rows = ~df['xco2_bc_anomaly'].isna()
    y_all    = df['xco2_bc_anomaly'].values.astype(np.float32)
    X_qt_raw = df[pipeline.qt_features].to_numpy(dtype=np.float32)
    X_fp_raw = df[pipeline.fp_cols].to_numpy(dtype=np.float32)
    del df
    gc.collect()

    X_qt_out = pipeline.qt.transform(X_qt_raw).astype(np.float32)
    del X_qt_raw
    gc.collect()

    X_all = np.concatenate([X_qt_out, X_fp_raw], axis=1)
    del X_qt_out, X_fp_raw
    gc.collect()

    X = X_all[valid_rows]
    y = y_all[valid_rows]
    del X_all, y_all
    gc.collect()

    print(f"  X shape: {X.shape}  y shape: {y.shape}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    del X, y
    gc.collect()
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}", flush=True)

    # ── Load or train model ────────────────────────────────────────────────────
    if HybridAdapter.can_load(output_dir):
        adapter = HybridAdapter.load(output_dir)
        model   = adapter.model
        print(f"  [checkpoint] Loaded from {output_dir}", flush=True)
    else:
        epochs = 100 if platform.system() == "Darwin" else 500
        model = train_hybrid_model(
            X_train, y_train, X_test, y_test,
            features=features,
            output_dir=str(output_dir),
            d_token=128, n_heads=8, n_layers=3, d_ff=256,
            mlp_hidden=256, fusion_dim=128,
            batch_size=1024 if platform.system() == "Darwin" else 4096, n_epochs=epochs,
            patience=50,
            loss_fn=args.loss,
            huber_delta=args.huber_delta,
        )
        HybridAdapter(
            model,
            n_features=pipeline.n_features,
            d_token=128, n_heads=8, n_layers=3, d_ff=256,
            mlp_hidden=256, fusion_dim=128,
            feature_names=features,
        ).save(output_dir)
        print(f"  Saved adapter → {output_dir}", flush=True)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluate_hybrid(model, X_test, y_test, str(output_dir))

    # ── Permutation importance (reused from models_transformer) ───────────────
    plot_permutation_importance(model, X_test, y_test, features, str(output_dir))
    # Rename to avoid colliding with FT output
    for stem in ('ft_permutation_importance.csv', 'ft_permutation_importance.png'):
        src = Path(output_dir) / stem
        dst = Path(output_dir) / stem.replace('ft_', 'hybrid_')
        if src.exists():
            src.rename(dst)

    # ── Regime comparison plot ─────────────────────────────────────────────────
    _dp2   = str(fdir / data_name)
    df_eval = pd.read_parquet(_dp2) if _dp2.endswith('.parquet') else pd.read_csv(_dp2)
    df_eval = df_eval[df_eval['sfc_type'] == args.sfc_type]
    df_eval = df_eval[df_eval['snow_flag'] == 0]
    plot_hybrid_evaluation_by_regime(model, df_eval, pipeline, str(output_dir))
    del df_eval
    gc.collect()


if __name__ == "__main__":
    main()
