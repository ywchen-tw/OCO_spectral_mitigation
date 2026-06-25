"""TabM-style BatchEnsemble MLP with a monotonic quantile head for OCO-2
XCO2 anomaly prediction.

Architecture reference:
  Gorishniy, Kotelnikov, Babenko. "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling." 2024.  arXiv:2410.24210.

This implementation is a TabM-*inspired* variant and deviates from the
reference as follows (see TABM_PLAN.md for the rationale):
  - Shared input projection for all K members (vs. per-member projection);
    member diversity is deferred to the BatchEnsemble hidden r/s vectors.
  - Monotonic three-quantile output head (q05 ≤ q50 ≤ q95) via softplus deltas
    (vs. scalar point regression).
  - Per-member three-quantile output [batch, K, 3]; the default forward returns
    the *mean member quantile* (not the quantile of the ensemble mixture).
  - Optional auxiliary cloud-proximity head for the multi-task ablation.

Describe this model as a "TabM-style BatchEnsemble MLP with monotonic quantile
head" in any paper, not as "TabM", to avoid misrepresenting the reference.
"""

import argparse
import gc
import json
import logging
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers
from .splits import split_dataframe
from .adapters import TabMAdapter
from . import diagnostics as diag
# Reuse loss functions from the FT-Transformer module (imported, not duplicated).
from .transformer import (
    huber_pinball_loss, quantile_loss, variance_penalty, mmd_loss_1d,
    plot_permutation_importance, evaluate_model_X_text, plot_evaluation_by_regime,
)
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

QUANTILES = [0.05, 0.5, 0.95]


# ─── BatchEnsemble building blocks ─────────────────────────────────────────────

class TabMLayer(nn.Module):
    """BatchEnsemble linear layer: one shared weight + K per-member r/s vectors.

    Extra parameters vs. a naive K-fold ensemble are K·(d_in + d_out) rather
    than K·d_in·d_out — orders of magnitude cheaper.

    Forward, for input x of shape [batch, K, d_in]:
        x_scaled = x * r           # [K, d_in] broadcast
        h        = (x_scaled @ Wᵀ + b)   # shared linear over flattened members
        out      = h * s           # [K, d_out] broadcast
    """

    def __init__(self, d_in: int, d_out: int, K: int):
        super().__init__()
        self.d_in, self.d_out, self.K = d_in, d_out, K
        self.linear = nn.Linear(d_in, d_out)
        # init: ones + N(0, 0.01) so members start near-identical but diverge.
        self.r = nn.Parameter(torch.ones(K, d_in) + 0.01 * torch.randn(K, d_in))
        self.s = nn.Parameter(torch.ones(K, d_out) + 0.01 * torch.randn(K, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, K, d_in]
        x = x * self.r                       # broadcast [K, d_in]
        b, k, din = x.shape
        h = self.linear(x.reshape(b * k, din))   # [b*k, d_out]
        h = h.reshape(b, k, self.d_out)
        return h * self.s                    # broadcast [K, d_out]


class TabMBlock(nn.Module):
    """Pre-activation residual block (mirrors _ResBlock in adapters.py) with
    per-member TabMLayer inner projections and a shared skip path."""

    def __init__(self, d: int, K: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.lin1 = TabMLayer(d, d, K)
        self.lin2 = TabMLayer(d, d, K)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.lin1(F.gelu(self.ln1(x)))
        x = self.lin2(self.drop(F.gelu(self.ln2(x))))
        return skip + x


def _monotonic_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map raw head outputs (a, b, c) → ordered (q05, q50, q95).

        q50 = a;  q05 = q50 − softplus(b);  q95 = q50 + softplus(c)

    Guarantees q05 < q50 < q95 for every member and sample, eliminating
    quantile crossing without a penalty term.  raw: [..., 3] → [..., 3].
    """
    q50 = raw[..., 0]
    q05 = q50 - F.softplus(raw[..., 1])
    q95 = q50 + F.softplus(raw[..., 2])
    return torch.stack([q05, q50, q95], dim=-1)


class TabM(nn.Module):
    """Full TabM-style model.

    forward(x) → [batch, 3]                      (q05, q50, q95) — drop-in default
    forward(x, return_members=True) → [batch, K, 3]   per-member ordered quantiles

    When the auxiliary cloud head is active, forward() returns a dict:
        {"quantiles": [batch, 3],
         "members":   [batch, K, 3]   (only if return_members=True),
         "cloud_logit": [batch, n_cloud_classes]}
    so existing evaluation code (which expects a [batch, 3] tensor) must unwrap
    output["quantiles"] — see _batched_predict's dict guard in transformer.py.
    """

    def __init__(self, n_features: int, K: int = 16, d_model: int = 256,
                 n_layers: int = 4, dropout: float = 0.2,
                 aux_cloud: bool = False, n_cloud_classes: int = 1):
        super().__init__()
        self.n_features = n_features
        self.K = K
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.aux_cloud = aux_cloud
        self.n_cloud_classes = n_cloud_classes

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.blocks = nn.ModuleList(
            [TabMBlock(d_model, K, dropout) for _ in range(n_layers)]
        )
        self.head = TabMLayer(d_model, 3, K)     # per-member (q50, dlo, dhi) logits
        if aux_cloud:
            self.cloud_head = nn.Linear(d_model, n_cloud_classes)

    def init_from_targets(self, y_train: np.ndarray) -> None:
        """Initialise the head bias from the target distribution so the model
        starts with a plausible interval (see TABM_PLAN.md init note).

        head.linear.bias channels are (q50, dlo_raw, dhi_raw); softplus⁻¹(x) =
        log(exp(x) − 1) recovers the pre-softplus value for a target delta.
        """
        y = np.asarray(y_train, dtype=float)
        q50 = float(np.median(y))
        dlo = max(q50 - float(np.quantile(y, 0.05)), 1e-3)
        dhi = max(float(np.quantile(y, 0.95)) - q50, 1e-3)
        with torch.no_grad():
            self.head.linear.bias[0] = q50
            self.head.linear.bias[1] = math.log(math.expm1(dlo))
            self.head.linear.bias[2] = math.log(math.expm1(dhi))
        logger.info("TabM head init from targets: q50=%.4f dlo=%.4f dhi=%.4f",
                    q50, dlo, dhi)

    def forward(self, x: torch.Tensor, return_members: bool = False):
        h = self.input_proj(x)                       # [batch, d_model]
        h = h.unsqueeze(1).expand(-1, self.K, -1)    # [batch, K, d_model]
        for block in self.blocks:
            h = block(h)
        members = _monotonic_quantiles(self.head(h))  # [batch, K, 3]
        q = members.mean(dim=1)                       # [batch, 3]

        if self.aux_cloud:
            h_pooled = h.mean(dim=1)                  # [batch, d_model]
            out = {"quantiles": q, "cloud_logit": self.cloud_head(h_pooled)}
            if return_members:
                out["members"] = members
            return out

        return members if return_members else q


# ─── Inference helper ──────────────────────────────────────────────────────────

def _tabm_predict(model: TabM, X_np: np.ndarray, batch_size: int = 8192,
                  want_members: bool = False) -> tuple:
    """Batched inference.  Returns (preds [N,3], members [N,K,3] | None)."""
    model.eval()
    device = next(model.parameters()).device
    preds, members = [], []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32).to(device)
            if want_members:
                m = model(Xb, return_members=True)
                if isinstance(m, dict):
                    m = m["members"]
                members.append(m.cpu().numpy())
                preds.append(m.mean(dim=1).cpu().numpy())
            else:
                out = model(Xb)
                if isinstance(out, dict):
                    out = out["quantiles"]
                preds.append(out.cpu().numpy())
            del Xb
    preds = np.concatenate(preds, axis=0)
    members = np.concatenate(members, axis=0) if want_members else None
    return preds, members


# ─── Run config ────────────────────────────────────────────────────────────────

def _default_run_config() -> dict:
    return {
        'data': {
            'sfc_type': 0,
            'snow_flag_value': 0,
            'linux_data_name': 'combined_2016_2020_dates.parquet',
            'darwin_data_name': 'combined_2020-02-01_all_orbits.parquet',
        },
        'split': {
            'val_split': 'random',     # 'random' or 'date'
            'test_size': 0.2,          # date mode: fraction of unique dates (use 0.1)
            'random_state': 42,
        },
        'model': {
            'K': 16,
            'd_model': 256,
            'n_layers': 4,
            'dropout': 0.2,
        },
        'train': {
            'darwin_epochs': 100,
            'linux_epochs': 500,
            'darwin_batch_size': 2048,
            'linux_batch_size': 8192,
            'patience': 50,
            'log_every': 10,
            'seed': 42,
        },
        'loss': {
            'loss': 'huber',
            'huber_delta': 1.0,
            'range_loss_weight': 0.0,
            'range_loss_type': 'variance',
        },
        'pipeline': {
            'feature_set': 'full',
            'pca_augment': False,
        },
        'aux_cloud': {
            'enabled': False,
            'cloud_label': 'binary',   # 'binary' or 'bins'
            'near_cloud_km': 10.0,
            'lambda_cloud': 0.1,
        },
    }


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_run_config(config_path: 'str | None') -> dict:
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


# ─── Training loop ─────────────────────────────────────────────────────────────

# 4-class cloud-distance bins (km) for the 'bins' auxiliary label.
_CLOUD_BIN_EDGES = [0.0, 5.0, 10.0, 30.0, float('inf')]


def _cloud_bin_labels(cld_dist_km: np.ndarray) -> np.ndarray:
    """0: 0–5, 1: 5–10, 2: 10–30, 3: >30 km."""
    return np.clip(np.digitize(cld_dist_km, _CLOUD_BIN_EDGES[1:-1], right=False),
                   0, 3).astype(np.int64)


def train_tabm(X_train, y_train, X_test, y_test, features, *,
               output_dir: str = ".",
               K: int = 16, d_model: int = 256, n_layers: int = 4, dropout: float = 0.2,
               batch_size: int = 8192, n_epochs: int = 100,
               log_every: int = 10, patience: 'int | None' = 50,
               loss_fn: str = 'huber', huber_delta: float = 1.0,
               range_loss_weight: float = 0.0, range_loss_type: str = 'variance',
               y_init: 'np.ndarray | None' = None,
               seed: int = 42,
               aux_cloud: bool = False, cloud_label: str = 'binary',
               lambda_cloud: float = 0.1,
               cloud_train: 'np.ndarray | None' = None,
               cloud_test: 'np.ndarray | None' = None) -> TabM:
    """Train a TabM model; mirrors train_uncertainty_transformer() structure.

    The anomaly loss is computed per member (members reshaped to [B·K, 3]) so
    every ensemble member is trained, then averaged.  Validation MAE/R² use the
    default mean-member q50.  Writes model_tabm_best.pt (keys: epoch,
    model_state_dict, val_loss, val_mae, val_r2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    n_cloud_classes = 4 if (aux_cloud and cloud_label == 'bins') else 1

    # ── Datasets (optionally carry the cloud label as a third tensor) ──────────
    train_tensors = [torch.tensor(X_train, dtype=torch.float32),
                     torch.tensor(y_train, dtype=torch.float32)]
    val_tensors = [torch.tensor(X_test, dtype=torch.float32),
                   torch.tensor(y_test, dtype=torch.float32)]
    pos_weight = None
    if aux_cloud:
        if cloud_train is None or cloud_test is None:
            raise ValueError("aux_cloud=True requires cloud_train and cloud_test labels")
        cloud_dtype = torch.long if cloud_label == 'bins' else torch.float32
        train_tensors.append(torch.tensor(cloud_train, dtype=cloud_dtype))
        val_tensors.append(torch.tensor(cloud_test, dtype=cloud_dtype))
        if cloud_label == 'binary':
            n_neg = float((cloud_train == 0).sum())
            n_pos = float((cloud_train == 1).sum())
            pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
            logger.info("aux_cloud binary pos_weight=%.4f (n_neg=%d n_pos=%d)",
                        pos_weight.item(), int(n_neg), int(n_pos))

    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    pin = device.type in ("cuda", "mps")
    n_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=n_workers,
                              persistent_workers=n_workers > 0, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=pin, num_workers=n_workers,
                            persistent_workers=n_workers > 0)

    model = TabM(n_features=len(features), K=K, d_model=d_model,
                 n_layers=n_layers, dropout=dropout,
                 aux_cloud=aux_cloud, n_cloud_classes=n_cloud_classes).to(device)
    if y_init is not None:
        model.init_from_targets(y_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=n_epochs * len(train_loader),
        pct_start=0.05, div_factor=25, final_div_factor=1000,
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if (aux_cloud and cloud_label == 'binary') else None
    ce = nn.CrossEntropyLoss() if (aux_cloud and cloud_label == 'bins') else None

    def _anomaly_loss(members: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # members: [B, K, 3] → per-member loss via flatten to [B*K, 3]
        b, k, _ = members.shape
        flat = members.reshape(b * k, 3)
        tgt = targets.repeat_interleave(k)
        if loss_fn == 'huber':
            base = huber_pinball_loss(flat, tgt, QUANTILES, delta=huber_delta)
        else:
            base = quantile_loss(flat, tgt, QUANTILES)
        base = base + 0.05 * flat[:, 1].abs().mean()    # mild q50 shrinkage (matches FT scaffold)
        if range_loss_weight > 0.0:
            q50_mean = members.mean(dim=1)[:, 1]
            if range_loss_type == 'mmd':
                base = base + range_loss_weight * mmd_loss_1d(q50_mean, targets)
            else:
                base = base + range_loss_weight * variance_penalty(q50_mean, targets)
        return base

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write("=" * 60)
    tqdm.write("  TabM (BatchEnsemble MLP, monotonic quantile head) — training")
    tqdm.write("=" * 60)
    tqdm.write(f"  Device      : {device}")
    tqdm.write(f"  Features    : {len(features)}")
    tqdm.write(f"  K           : {K}  d_model: {d_model}  n_layers: {n_layers}  dropout: {dropout}")
    tqdm.write(f"  Params      : {n_params:,}")
    tqdm.write(f"  Train size  : {len(train_ds):,}  |  Val size: {len(val_ds):,}")
    tqdm.write(f"  Batch size  : {batch_size}  |  Epochs: {n_epochs}  |  Log every: {log_every}")
    tqdm.write(f"  Early stop  : {'disabled' if patience is None else f'patience={patience}'}")
    tqdm.write(f"  Loss        : {loss_fn}" + (f"  (huber_delta={huber_delta})" if loss_fn == 'huber' else ""))
    tqdm.write(f"  Aux cloud   : {'%s λ=%.3f' % (cloud_label, lambda_cloud) if aux_cloud else 'disabled'}")
    ckpt_path = os.path.join(output_dir, 'model_tabm_best.pt')
    tqdm.write(f"  Checkpoint  : {ckpt_path}")
    tqdm.write("=" * 60)
    logger.info("TabM training: n_params=%d device=%s aux_cloud=%s", n_params, device, aux_cloud)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    epoch_bar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch:3d} [train]",
                          leave=False, unit="batch"):
            batch = [t.to(device) for t in batch]
            batch_x, batch_y = batch[0], batch[1]
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type in ("cuda", "mps"))):
                if aux_cloud:
                    out = model(batch_x, return_members=True)
                    members, cloud_logit = out["members"], out["cloud_logit"]
                    loss = _anomaly_loss(members, batch_y)
                    cloud_y = batch[2]
                    if cloud_label == 'binary':
                        loss = loss + lambda_cloud * bce(cloud_logit.squeeze(-1), cloud_y.float())
                    else:
                        loss = loss + lambda_cloud * ce(cloud_logit, cloud_y)
                else:
                    members = model(batch_x, return_members=True)
                    loss = _anomaly_loss(members, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            train_loss += loss.item()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_q50, val_tgt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = [t.to(device) for t in batch]
                batch_x, batch_y = batch[0], batch[1]
                if aux_cloud:
                    out = model(batch_x, return_members=True)
                    members, cloud_logit = out["members"], out["cloud_logit"]
                    vl = _anomaly_loss(members, batch_y)
                    cloud_y = batch[2]
                    if cloud_label == 'binary':
                        vl = vl + lambda_cloud * bce(cloud_logit.squeeze(-1), cloud_y.float())
                    else:
                        vl = vl + lambda_cloud * ce(cloud_logit, cloud_y)
                else:
                    members = model(batch_x, return_members=True)
                    vl = _anomaly_loss(members, batch_y)
                val_loss += vl.item()
                val_q50.append(members.mean(dim=1)[:, 1].cpu())
                val_tgt.append(batch_y.cpu())

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        q50_np = torch.cat(val_q50).numpy()
        tgt_np = torch.cat(val_tgt).numpy()
        val_mae = float(np.abs(q50_np - tgt_np).mean())
        _denom = np.sum((tgt_np - tgt_np.mean()) ** 2)
        val_r2 = float(1.0 - np.sum((tgt_np - q50_np) ** 2) / _denom) if _denom > 0 else float('nan')

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss, "val_mae": val_mae, "val_r2": val_r2},
                       ckpt_path)
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(train=f"{avg_train:.5f}", val=f"{avg_val:.5f}",
                              best=f"{best_val_loss:.5f}", MAE=f"{val_mae:.4f}",
                              R2=f"{val_r2:.4f}", saved="✓" if improved else "")

        if patience is not None and epochs_no_improve >= patience:
            tqdm.write(f"  [epoch {epoch:4d}] Early stopping (no improvement for {patience} epochs).")
            break
        if (epoch + 1) % log_every == 0 or epoch == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            tqdm.write(f"  [{ts}] epoch {epoch+1:4d}/{n_epochs}  train={avg_train:.5f}  "
                       f"val={avg_val:.5f}  best={best_val_loss:.5f}  MAE={val_mae:.4f}  R²={val_r2:.4f}")
            logger.info("Epoch %d/%d train=%.5f val=%.5f best=%.5f MAE=%.4f R2=%.4f",
                        epoch + 1, n_epochs, avg_train, avg_val, best_val_loss, val_mae, val_r2)

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"No checkpoint saved to {ckpt_path}. Training produced only NaN losses — "
            "check for NaN/Inf in input features."
        )
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    tqdm.write("=" * 60)
    tqdm.write(f"  Training complete.  Best epoch={best['epoch']}  "
               f"val_loss={best['val_loss']:.5f}  MAE={best.get('val_mae', float('nan')):.4f}  "
               f"R²={best.get('val_r2', float('nan')):.4f}")
    tqdm.write("=" * 60)
    return model.cpu()


# ─── CLI / orchestration ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TabM-style BatchEnsemble MLP for XCO2 anomaly prediction")
    parser.add_argument('--sfc_type', type=int, default=None,
                        help="Surface type filter: 0=ocean, 1=land.")
    parser.add_argument('--suffix', type=str, default='',
                        help='Subfolder under results/model_tabm/.')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Path to a pre-fitted FeaturePipeline (.pkl).  If omitted, '
                             'a new pipeline is fitted ON THE TRAIN SPLIT ONLY and saved.')
    parser.add_argument('--val_split', type=str, default=None,
                        choices=['random', 'date', 'date_kfold'],
                        help="Validation split: 'random' (default), 'date' (trailing block), "
                             "or 'date_kfold' (block-rotation k-fold; needs --n_folds/--fold).")
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of date blocks for --val_split date_kfold.')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which date block (0-based) to hold out for date_kfold.')
    parser.add_argument('--feature_set', type=str, default=None,
                        choices=['full', 'no_xco2', 'no_spec', 'full_fitqual'],
                        help="Feature ablation set (see pipeline._FEATURE_SETS).")
    parser.add_argument('--K', type=int, default=None,
                        help="Ensemble size override (K=1 is the degenerate-MLP ablation). "
                             "Overrides model.K from --config.")
    parser.add_argument('--loss', type=str, default=None, choices=['quantile', 'huber'],
                        help='Loss: "huber" (default) or "quantile" (pinball on all three).')
    parser.add_argument('--huber-delta', type=float, default=None,
                        help='Huber transition δ (ppm).  Only used with --loss huber.')
    parser.add_argument('--aux_cloud', action='store_true',
                        help='Enable the auxiliary cloud-proximity head (multi-task ablation).')
    parser.add_argument('--cloud_label', type=str, default=None, choices=['binary', 'bins'],
                        help="Auxiliary cloud label: 'binary' (near_cloud ≤ km) or 'bins' (4-class).")
    parser.add_argument('--lambda_cloud', type=float, default=None,
                        help='Weight on the auxiliary cloud loss (default 0.1).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--data', type=str, default=None,
                        help='Override the input data file (default: platform data_name in '
                             'results/csv_collection/). Use for local multi-date testing.')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config overriding data/model/train/loss/aux_cloud keys.')
    args = parser.parse_args()

    run_cfg = _load_run_config(args.config)
    if args.sfc_type is not None:
        run_cfg['data']['sfc_type'] = int(args.sfc_type)
    if args.val_split is not None:
        run_cfg['split']['val_split'] = args.val_split
    if args.feature_set is not None:
        run_cfg['pipeline']['feature_set'] = args.feature_set
    if args.K is not None:
        run_cfg['model']['K'] = int(args.K)
    if args.loss is not None:
        run_cfg['loss']['loss'] = args.loss
    if args.huber_delta is not None:
        run_cfg['loss']['huber_delta'] = float(args.huber_delta)
    if args.aux_cloud:
        run_cfg['aux_cloud']['enabled'] = True
    if args.cloud_label is not None:
        run_cfg['aux_cloud']['cloud_label'] = args.cloud_label
    if args.lambda_cloud is not None:
        run_cfg['aux_cloud']['lambda_cloud'] = float(args.lambda_cloud)
    if args.seed is not None:
        run_cfg['train']['seed'] = int(args.seed)

    storage_dir = get_storage_dir()
    fdir = storage_dir / 'results/csv_collection'
    data_name = (run_cfg['data']['linux_data_name'] if platform.system() == "Linux"
                 else run_cfg['data']['darwin_data_name'])
    base_dir = storage_dir / 'results/model_tabm'
    output_dir = base_dir / args.suffix if args.suffix else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.monotonic()
    run_id = args.suffix if args.suffix else datetime.now().strftime('%Y%m%d-%H%M%S')
    commit = get_git_commit_hash(storage_dir)

    run_cfg_path = output_dir / 'tabm_run_config.json'
    with open(run_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)
    print(f"Saved resolved run config → {run_cfg_path}")

    surface_type = int(run_cfg['data']['sfc_type'])
    val_split = run_cfg['split']['val_split']
    feature_set = run_cfg['pipeline']['feature_set']
    aux_cfg = run_cfg['aux_cloud']
    aux_cloud = bool(aux_cfg['enabled'])
    cloud_label = aux_cfg['cloud_label']

    # ── Load + filter data ─────────────────────────────────────────────────────
    _dp = args.data if args.data else os.path.join(fdir, data_name)
    df = pd.read_parquet(_dp) if _dp.endswith('.parquet') else pd.read_csv(_dp)
    df = df[df['sfc_type'] == surface_type]
    df = df[df['snow_flag'] == run_cfg['data']['snow_flag_value']]
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)

    # ── Split the RAW dataframe FIRST (leakage discipline) ─────────────────────
    train_df, held_df = split_dataframe(
        df, mode=val_split,
        test_size=float(run_cfg['split']['test_size']),
        random_state=int(run_cfg['split']['random_state']),
        n_folds=args.n_folds, fold=args.fold,
    )
    del df
    gc.collect()
    print(f"Split [{val_split}]: train={len(train_df):,}  held={len(held_df):,}")

    # ── Fit pipeline on the TRAIN split only (or load a provided one) ──────────
    pipeline_path = output_dir / 'tabm_pipeline.pkl'
    if args.pipeline:
        pipeline = FeaturePipeline.load(args.pipeline)
    elif TabMAdapter.can_load(output_dir) and pipeline_path.exists():
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline.fit(train_df, sfc_type=surface_type,
                                       feature_set=feature_set,
                                       pca_augment=bool(run_cfg['pipeline']['pca_augment']))
        pipeline.save(pipeline_path)
    features = pipeline.features

    # ── Leakage check for the auxiliary cloud head ─────────────────────────────
    if aux_cloud:
        banned = {'cld_dist_km', 'near_cloud'}
        present = [f for f in features if f in banned or 'cld_dist' in f]
        if present:
            raise ValueError(
                "Auxiliary cloud head requires cloud-distance features to be ABSENT "
                f"from pipeline.features, but found: {present}.  These are target-derived; "
                "run the 'direct cloud-distance feature' case as a separate experiment."
            )
        if 'cld_dist_km' not in held_df.columns:
            raise ValueError("aux_cloud requires a 'cld_dist_km' column to build labels.")

    # ── Transform (train-fitted pipeline applied to both splits) ───────────────
    def _prep(frame):
        X = pipeline.transform(frame)
        y = frame['xco2_bc_anomaly'].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], frame.loc[valid]

    X_train, y_train, train_valid = _prep(train_df)
    X_held, y_held, held_valid = _prep(held_df)
    print(f"X_train {X_train.shape}  X_held {X_held.shape}")

    cloud_train = cloud_held = None
    if aux_cloud:
        near_km = float(aux_cfg['near_cloud_km'])
        cd_tr = train_valid['cld_dist_km'].to_numpy(dtype=float)
        cd_hd = held_valid['cld_dist_km'].to_numpy(dtype=float)
        if cloud_label == 'binary':
            cloud_train = (cd_tr <= near_km).astype(np.float32)
            cloud_held = (cd_hd <= near_km).astype(np.float32)
        else:
            cloud_train = _cloud_bin_labels(cd_tr)
            cloud_held = _cloud_bin_labels(cd_hd)

    # ── Load or train model ────────────────────────────────────────────────────
    if TabMAdapter.can_load(output_dir):
        adapter = TabMAdapter.load(output_dir)
        model = adapter.model
    else:
        if platform.system() == "Darwin":
            epochs = int(run_cfg['train']['darwin_epochs'])
            batch_size = int(run_cfg['train']['darwin_batch_size'])
        else:
            epochs = int(run_cfg['train']['linux_epochs'])
            batch_size = int(run_cfg['train']['linux_batch_size'])
        model = train_tabm(
            X_train, y_train, X_held, y_held, features=features,
            output_dir=str(output_dir),
            K=int(run_cfg['model']['K']), d_model=int(run_cfg['model']['d_model']),
            n_layers=int(run_cfg['model']['n_layers']), dropout=float(run_cfg['model']['dropout']),
            batch_size=batch_size, n_epochs=epochs,
            log_every=int(run_cfg['train']['log_every']), patience=int(run_cfg['train']['patience']),
            loss_fn=run_cfg['loss']['loss'], huber_delta=float(run_cfg['loss']['huber_delta']),
            range_loss_weight=float(run_cfg['loss']['range_loss_weight']),
            range_loss_type=run_cfg['loss']['range_loss_type'],
            y_init=y_train, seed=int(run_cfg['train']['seed']),
            aux_cloud=aux_cloud, cloud_label=cloud_label,
            lambda_cloud=float(aux_cfg['lambda_cloud']),
            cloud_train=cloud_train, cloud_test=cloud_held,
        )
        TabMAdapter(model, n_features=pipeline.n_features, K=int(run_cfg['model']['K']),
                    d_model=int(run_cfg['model']['d_model']), n_layers=int(run_cfg['model']['n_layers']),
                    dropout=float(run_cfg['model']['dropout']), feature_names=features,
                    feature_set=feature_set, val_split=val_split,
                    aux_cloud=aux_cloud, n_cloud_classes=(4 if (aux_cloud and cloud_label == 'bins') else 1)
                    ).save(output_dir)

    # ── Diagnostics on the held-out set ────────────────────────────────────────
    preds, members = _tabm_predict(model, X_held, want_members=True)
    global_metrics = diag.compute_metrics(y_held, preds, members=members)
    strat = diag.stratified_metrics(held_valid, y_held, preds)
    calib = diag.calibration_report(global_metrics, strat)
    prefix = f"tabm_{val_split}"
    diag.save_diagnostics(output_dir, prefix, global_metrics, strat, calib)
    diag.save_correction_and_preds(output_dir, prefix, held_valid, y_held, preds)
    print(f"[{prefix}] RMSE={global_metrics['rmse']:.4f}  MAE={global_metrics['mae']:.4f}  "
          f"R²={global_metrics['r2']:.4f}  cov90={global_metrics['coverage_90']:.3f}  "
          f"width={global_metrics['mean_interval_width']:.3f}  cross={global_metrics['crossing_rate']:.3g}")

    # ── Reused FT diagnostics ──────────────────────────────────────────────────
    plot_permutation_importance(model, X_held, y_held, features, str(output_dir),
                                output_prefix="tabm")
    evaluate_model_X_text(model, X_held, y_held, fig_dir=str(output_dir))
    try:
        plot_evaluation_by_regime(model, held_valid.copy(), pipeline.qt, pipeline.features, str(output_dir))
    except Exception as exc:                      # visual extra; never fail the run on it
        logger.warning("plot_evaluation_by_regime skipped: %s", exc)

    # ── Run summary ────────────────────────────────────────────────────────────
    try:
        import resource as _res
        rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
        peak_mb = rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024
    except Exception:
        peak_mb = 0.0

    summary = RunSummary(
        run_id=run_id, script_name=os.path.basename(__file__), model_family='tabm',
        commit=commit, status='success',
        primary_metric_name='tabm_held_rmse', primary_metric_value=global_metrics['rmse'],
        secondary_metrics={
            'tabm_held_mae': global_metrics['mae'],
            'tabm_held_r2': global_metrics['r2'],
            'coverage_90': global_metrics['coverage_90'],
            'mean_interval_width': global_metrics['mean_interval_width'],
            'crossing_rate': global_metrics['crossing_rate'],
            'pinball_q05': global_metrics['pinball_q05'],
            'pinball_q50': global_metrics['pinball_q50'],
            'pinball_q95': global_metrics['pinball_q95'],
            **({'member_spread_mean': global_metrics['member_spread_mean']}
               if 'member_spread_mean' in global_metrics else {}),
        },
        peak_memory_mb=float(peak_mb), runtime_seconds=float(time.monotonic() - run_start),
        description=f'TabM K={run_cfg["model"]["K"]} {val_split}-split, feature_set={feature_set}',
        artifacts={
            'output_dir': str(output_dir),
            'run_config': str(run_cfg_path),
            'pipeline': str(pipeline_path),
            'model_best': str(output_dir / 'model_tabm_best.pt'),
            'metrics_json': str(output_dir / f'{prefix}_metrics.json'),
        },
        config=run_cfg,
    )
    summary_path = output_dir / 'run_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
    print(f"Saved run summary → {summary_path}")


if __name__ == "__main__":
    main()
