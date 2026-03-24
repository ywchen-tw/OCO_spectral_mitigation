"""apply_model_with_cld.py — apply regression + cloud-proximity classifier together.

Extends apply_models.py with three additional steps:

  1. Load a cloud-proximity classifier (MLP/LR from mlp_lr_cloud_classifier.py or
     FT-Transformer from transformer_cloud_classifier.py) and predict P(cld_dist < 10 km)
     for every sounding.

  2. Predict the xco2_bc anomaly as normal (Ridge / MLP / FT / XGBoost / Hybrid).

  3. Apply the regression correction to xco2_bc **only** when the classifier predicts
     cloud-affected (P > 0.5); leave xco2_bc unchanged for clear-sky soundings.

  4. All standard comparison plots from apply_models.py are produced, plus a new
     spatial-map panel showing the cloud-proximity probability coloured Red (> 0.5)
     / Blue (< 0.5).

Usage:
    python src/apply_model_with_cld.py \\
        --pipeline   results/exp_v1/pipeline.pkl \\
        --ridge-dir  results/model_mlp_lr/exp_v1/ \\
        --clf-dir    results/model_mlp_clf/        \\
        --input      new_data.csv \\
        --output     corrected_cld.csv

    # FT-Transformer classifier instead of MLP/LR
    python src/apply_model_with_cld.py \\
        --pipeline   results/exp_v1/pipeline.pkl \\
        --ridge-dir  results/model_mlp_lr/exp_v1/ \\
        --ft-clf-dir results/model_ft_clf/        \\
        --input      new_data.csv

Output columns added (beyond standard apply_models.py columns):
    cld_prob_lr      — LR cloud-proximity probability (if LR clf available)
    cld_prob_mlp_clf — MLP cloud-proximity probability (if MLP clf available)
    cld_prob_ft_clf  — FT classifier cloud-proximity probability (if FT clf available)
    cld_prob         — ensemble mean (or single model) used for gating
    cld_pred         — binary prediction (1 = cloud-affected) from best available clf
    <model>_cond_corrected_xco2  — xco2_bc corrected only where cld_pred == 1
"""

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import FeaturePipeline
from model_adapters import (
    RidgeAdapter, MLPAdapter, FTAdapter, XGBoostAdapter, HybridAdapter,
    LogisticAdapter, MLPClassifierAdapter, FTClassifierAdapter,
)
from utils import get_storage_dir

# Re-use helpers from apply_models
from apply_models import (
    _try_load, _load_ft_with_bootstrap, _recompute_anomaly,
    _comparison_plots, _ANOMALY_ARGS, _ORBIT_COLS,
)

logger = logging.getLogger(__name__)

_SCATTER_MAX = 200_000


# ─── Cloud classifier loading ──────────────────────────────────────────────────

def _load_clf(clf_dir: 'str | None'):
    """Load LogisticAdapter + MLPClassifierAdapter from clf_dir.

    Returns (lr_adapter_or_None, mlp_clf_adapter_or_None).
    """
    if clf_dir is None:
        return None, None
    lr  = None
    mlp = None
    if LogisticAdapter.can_load(clf_dir):
        try:
            lr = LogisticAdapter.load(clf_dir)
            print(f"  [loaded] LR classifier ← {clf_dir}", flush=True)
        except Exception as exc:
            logger.warning("LR classifier load failed (%s) — skipping.", exc)
    else:
        logger.warning("LogisticAdapter not found in %s — skipping LR clf.", clf_dir)

    if MLPClassifierAdapter.can_load(clf_dir):
        try:
            mlp = MLPClassifierAdapter.load(clf_dir)
            print(f"  [loaded] MLP classifier ← {clf_dir}", flush=True)
        except Exception as exc:
            logger.warning("MLP classifier load failed (%s) — skipping.", exc)
    else:
        logger.warning("MLPClassifierAdapter not found in %s — skipping MLP clf.", clf_dir)

    return lr, mlp


def _load_ft_clf(ft_clf_dir: 'str | None'):
    """Load FTClassifierAdapter from ft_clf_dir."""
    if ft_clf_dir is None:
        return None
    if FTClassifierAdapter.can_load(ft_clf_dir):
        try:
            adapter = FTClassifierAdapter.load(ft_clf_dir)
            print(f"  [loaded] FT classifier ← {ft_clf_dir}", flush=True)
            return adapter
        except Exception as exc:
            logger.warning("FT classifier load failed (%s) — skipping.", exc)
    else:
        logger.warning("FTClassifierAdapter not found in %s — skipping FT clf.", ft_clf_dir)
    return None


# ─── Cloud probability spatial-map plot ───────────────────────────────────────

def _plot_cloud_prob_panel(df_out: pd.DataFrame, prob_col: str, plot_dir: Path) -> None:
    """Scatter map coloured Red (P > 0.5) / Blue (P ≤ 0.5).

    Saved as clf_cloud_prob_map.png inside plot_dir.
    """
    if prob_col not in df_out.columns:
        return
    if 'lon' not in df_out.columns or 'lat' not in df_out.columns:
        logger.info("No lon/lat columns — skipping cloud prob map.")
        return

    proba = df_out[prob_col].to_numpy(dtype=float)
    lon   = df_out['lon'].to_numpy(dtype=float)
    lat   = df_out['lat'].to_numpy(dtype=float)

    # Wrap longitude to [0, 360] if needed
    if lon.min() < -90 and lon.max() > 90:
        lon = np.where(lon < 0, lon + 360, lon)

    # Sub-sample for scatter performance
    N = len(proba)
    if N > _SCATTER_MAX:
        rng = np.random.default_rng(7)
        idx = rng.choice(N, size=_SCATTER_MAX, replace=False)
        lon, lat, proba = lon[idx], lat[idx], proba[idx]

    # Binary Red/Blue colouring: Red = cloud-affected (> 0.5), Blue = clear
    colors = np.where(proba > 0.5, 'red', 'blue')

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # ── Left panel: binary Red/Blue ────────────────────────────────────────
    ax = axes[0]
    mask_cld   = proba > 0.5
    mask_clear = ~mask_cld
    if mask_clear.any():
        ax.scatter(lon[mask_clear], lat[mask_clear],
                   c='blue', s=5, alpha=0.5, rasterized=True, label='Clear (P ≤ 0.5)')
    if mask_cld.any():
        ax.scatter(lon[mask_cld], lat[mask_cld],
                   c='red', s=5, alpha=0.5, rasterized=True, label='Cloud-affected (P > 0.5)')
    ax.set_title(f'Cloud Proximity Prediction\n({prob_col}  |  Red=cloud-affected, Blue=clear)')
    ax.set_xlabel('Longitude');  ax.set_ylabel('Latitude')
    n_cld   = int(mask_cld.sum())
    n_clear = int(mask_clear.sum())
    ax.legend(title=f'cloud={n_cld:,}  clear={n_clear:,}', fontsize=9)

    # ── Right panel: continuous probability with RdBu_r ────────────────────
    ax2 = axes[1]
    sc  = ax2.scatter(lon, lat, c=proba, cmap='RdBu_r', vmin=0, vmax=1,
                      s=5, alpha=0.6, rasterized=True)
    fig.colorbar(sc, ax=ax2, label='P(cld_dist < 10 km)')
    ax2.set_title(f'Cloud Proximity Probability\n({prob_col})')
    ax2.set_xlabel('Longitude');  ax2.set_ylabel('Latitude')

    fig.tight_layout()
    out_path = plot_dir / 'clf_cloud_prob_map.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved clf_cloud_prob_map.png")
    print(f"  Cloud prob map → {out_path}", flush=True)


# ─── Conditional correction plot ──────────────────────────────────────────────

def _plot_conditional_correction(df_out: pd.DataFrame,
                                  prob_col: str,
                                  model_name: str,
                                  cond_col: str,
                                  plot_dir: Path) -> None:
    """Compare xco2_bc vs conditionally-corrected xco2_bc on a map + histogram."""
    needed = {'lon', 'lat', 'xco2_bc', cond_col}
    if not needed.issubset(df_out.columns):
        return

    lon      = df_out['lon'].to_numpy(dtype=float)
    lat      = df_out['lat'].to_numpy(dtype=float)
    xco2_bc  = df_out['xco2_bc'].to_numpy(dtype=float)
    cond_xco2 = df_out[cond_col].to_numpy(dtype=float)
    proba     = df_out[prob_col].to_numpy(dtype=float) if prob_col in df_out.columns \
                else np.full(len(df_out), np.nan)

    if lon.min() < -90 and lon.max() > 90:
        lon = np.where(lon < 0, lon + 360, lon)

    diff = cond_xco2 - xco2_bc  # non-zero only where correction was applied

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ── Map: original XCO2 ─────────────────────────────────────────────────
    vlo = np.nanpercentile(xco2_bc, 2)
    vhi = np.nanpercentile(xco2_bc, 98)
    sc0 = axes[0].scatter(lon, lat, c=xco2_bc, cmap='viridis',
                          vmin=vlo, vmax=vhi, s=4, alpha=0.6, rasterized=True)
    fig.colorbar(sc0, ax=axes[0], label='XCO2_bc (ppm)')
    axes[0].set_title('Original XCO2_bc')
    axes[0].set_xlabel('Longitude');  axes[0].set_ylabel('Latitude')

    # ── Map: conditionally-corrected XCO2 ─────────────────────────────────
    sc1 = axes[1].scatter(lon, lat, c=cond_xco2, cmap='viridis',
                          vmin=vlo, vmax=vhi, s=4, alpha=0.6, rasterized=True)
    fig.colorbar(sc1, ax=axes[1], label='Cond-corrected XCO2_bc (ppm)')
    axes[1].set_title(f'{model_name} Conditional Correction\n(applied only where P > 0.5)')
    axes[1].set_xlabel('Longitude');  axes[1].set_ylabel('Latitude')

    # ── Histogram: diff ────────────────────────────────────────────────────
    corrected_mask = np.abs(diff) > 0
    if corrected_mask.any():
        d = diff[corrected_mask]
        bins = np.linspace(np.nanpercentile(d, 1), np.nanpercentile(d, 99), 80)
        axes[2].hist(d, bins=bins, color='darkorange', alpha=0.7,
                     label=f'n={corrected_mask.sum():,}')
        axes[2].axvline(0, color='k', linestyle='--', linewidth=0.8)
        mu, sigma = float(np.nanmean(d)), float(np.nanstd(d))
        axes[2].set_title(f'Correction applied  μ={mu:.3f}  σ={sigma:.3f}')
    else:
        axes[2].set_title('No corrections applied (P ≤ 0.5 everywhere)')
    axes[2].set_xlabel('Δ XCO2_bc (ppm)')
    axes[2].set_ylabel('Count')
    axes[2].legend(fontsize=9)

    fig.suptitle(f'Conditional Correction — {model_name}', fontsize=12)
    fig.tight_layout()
    out_path = plot_dir / f'cond_correction_{model_name.lower()}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved %s", out_path.name)
    print(f"  Conditional correction map → {out_path}", flush=True)


# ─── Per-file processing ───────────────────────────────────────────────────────

_GATE_CLF_PROB_COL = {
    'lr':      'cld_prob_lr',
    'mlp_clf': 'cld_prob_mlp_clf',
    'ft_clf':  'cld_prob_ft_clf',
}


def _process_one_file(
    input_path: str,
    output_path: str,
    plot_dir: str,
    suffix_tag: str,
    pipeline,
    # regression models
    ridge, mlp, ft, xgb, hybrid,
    available: set,
    # cloud classifiers
    lr_clf, mlp_clf, ft_clf,
    gate_clf: str,
    sfc_type: int,
) -> None:
    file_dir = Path(plot_dir) / Path(input_path).stem
    file_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + filter ──────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {input_path}", flush=True)
    try:
        df = (pd.read_parquet(input_path) if input_path.endswith('.parquet')
              else pd.read_csv(input_path))
    except FileNotFoundError:
        logger.warning("File not found, skipping: %s", input_path)
        return
    print(f"  Rows loaded: {len(df):,}", flush=True)

    df = df[df['sfc_type'] == sfc_type]
    if 'snow_flag' in df.columns:
        df = df[df['snow_flag'] == 0]
    df = df.reset_index(drop=True)
    print(f"  Rows after sfc_type={sfc_type} + snow_flag==0: {len(df):,}", flush=True)

    # ── Transform features ─────────────────────────────────────────────────
    print("Transforming features …", flush=True)
    X = pipeline.transform(df)
    print(f"  X shape: {X.shape}", flush=True)

    df_out = df.copy()

    # ── Cloud-proximity classification ─────────────────────────────────────
    print("\nRunning cloud-proximity classifiers …", flush=True)
    clf_proba_cols = []

    if lr_clf is not None:
        print("  Predicting cloud proximity with LR classifier …", flush=True)
        df_out['cld_prob_lr'] = lr_clf.predict_proba(X)
        clf_proba_cols.append('cld_prob_lr')

    if mlp_clf is not None:
        print("  Predicting cloud proximity with MLP classifier …", flush=True)
        df_out['cld_prob_mlp_clf'] = mlp_clf.predict_proba(X)
        clf_proba_cols.append('cld_prob_mlp_clf')

    if ft_clf is not None:
        print("  Predicting cloud proximity with FT classifier …", flush=True)
        df_out['cld_prob_ft_clf'] = ft_clf.predict_proba(X)
        clf_proba_cols.append('cld_prob_ft_clf')

    # Select gating probability from the chosen classifier only
    gate_prob_col = _GATE_CLF_PROB_COL.get(gate_clf)
    if gate_prob_col and gate_prob_col in df_out.columns:
        df_out['cld_prob'] = df_out[gate_prob_col].to_numpy(dtype=float)
        df_out['cld_pred'] = (df_out['cld_prob'] > 0.5).astype(np.int8)
        cloud_mask = df_out['cld_pred'].to_numpy(dtype=bool)
        n_cld   = int(cloud_mask.sum())
        n_total = len(cloud_mask)
        print(f"  Gate classifier: {gate_clf} ({gate_prob_col})", flush=True)
        print(f"  Cloud-affected (P > 0.5): {n_cld:,} / {n_total:,} "
              f"({100*n_cld/max(n_total,1):.1f}%)", flush=True)
    elif clf_proba_cols:
        # Requested gate clf was not loaded — fall back to first available
        fallback = clf_proba_cols[0]
        logger.warning("gate_clf='%s' not available; falling back to '%s'.", gate_clf, fallback)
        df_out['cld_prob'] = df_out[fallback].to_numpy(dtype=float)
        df_out['cld_pred'] = (df_out['cld_prob'] > 0.5).astype(np.int8)
        cloud_mask = df_out['cld_pred'].to_numpy(dtype=bool)
        n_cld   = int(cloud_mask.sum())
        n_total = len(cloud_mask)
        print(f"  Gate classifier (fallback): {fallback}")
        print(f"  Cloud-affected (P > 0.5): {n_cld:,} / {n_total:,} "
              f"({100*n_cld/max(n_total,1):.1f}%)", flush=True)
    else:
        cloud_mask = None
        print("  No classifiers loaded — correction will be applied to all soundings.",
              flush=True)

    # ── Regression predictions ─────────────────────────────────────────────
    print("\nPredicting xco2_bc anomaly …", flush=True)

    if ridge is not None:
        print("  Predicting with Ridge …", flush=True)
        df_out['ridge_pred'] = ridge.predict(X)

    if mlp is not None:
        print("  Predicting with MLP …", flush=True)
        df_out['mlp_pred'] = mlp.predict(X)

    if ft is not None:
        print("  Predicting with FT-Transformer …", flush=True)
        q05, q50, q95            = ft.predict_quantiles(X)
        df_out['ft_q05']         = q05
        df_out['ft_q50']         = q50
        df_out['ft_q95']         = q95
        df_out['ft_uncertainty'] = np.clip(q95 - q05, 0, None)

    if xgb is not None:
        print("  Predicting with XGBoost …", flush=True)
        df_out['xgb_pred'] = xgb.predict(X)

    if hybrid is not None:
        print("  Predicting with Hybrid …", flush=True)
        h_q05, h_q50, h_q95          = hybrid.predict_quantiles(X)
        df_out['hybrid_q05']          = h_q05
        df_out['hybrid_q50']          = h_q50
        df_out['hybrid_q95']          = h_q95
        df_out['hybrid_uncertainty']  = np.clip(h_q95 - h_q05, 0, None)

    # ── Conditional correction (apply only where cloud_mask == True) ───────
    has_xco2_bc = 'xco2_bc' in df_out.columns
    if has_xco2_bc:
        xco2_bc = df_out['xco2_bc'].to_numpy(dtype=float)
        # gate: True for every row if no classifier; cloud-affected rows otherwise
        gate = cloud_mask if cloud_mask is not None else np.ones(len(df_out), dtype=bool)

        print("\nApplying conditional correction (cloud-affected soundings only) …",
              flush=True)

        for pred_col, cond_name in [
            ('ridge_pred',  'ridge'),
            ('mlp_pred',    'mlp'),
            ('ft_q50',      'ft'),
            ('xgb_pred',    'xgb'),
            ('hybrid_q50',  'hybrid'),
        ]:
            if pred_col not in df_out.columns:
                continue
            pred     = df_out[pred_col].to_numpy(dtype=float)
            cond_xco2 = xco2_bc.copy()
            cond_xco2[gate] -= pred[gate]
            df_out[f'{cond_name}_cond_corrected_xco2'] = cond_xco2
            n_applied = int(gate.sum())
            print(f"  {cond_name:8s}: correction applied to {n_applied:,} rows", flush=True)

    # ── Anomaly re-computation on corrected fields ─────────────────────────
    has_orbit_cols = _ORBIT_COLS.issubset(df_out.columns)

    if has_orbit_cols and has_xco2_bc:
        print("\nRecomputing XCO2 anomaly on conditionally-corrected fields …", flush=True)
        xco2_bc = df_out['xco2_bc'].to_numpy(dtype=float)

        for pred_col, anom_name, cond_name in [
            ('ridge_pred',  'ridge_anomaly',  'ridge'),
            ('mlp_pred',    'mlp_anomaly',    'mlp'),
            ('ft_q50',      'ft_anomaly',     'ft'),
            ('xgb_pred',    'xgb_anomaly',    'xgb'),
            ('hybrid_q50',  'hybrid_anomaly', 'hybrid'),
        ]:
            # Standard (unconditional) anomaly — needed for comparison plots
            if pred_col in df_out.columns:
                corrected = xco2_bc - df_out[pred_col].to_numpy()
                df_out[anom_name] = _recompute_anomaly(df_out, corrected, cond_name.capitalize())

            # Conditional anomaly
            cond_col = f'{cond_name}_cond_corrected_xco2'
            if cond_col in df_out.columns:
                df_out[f'{cond_name}_cond_anomaly'] = _recompute_anomaly(
                    df_out, df_out[cond_col].to_numpy(), f'{cond_name.capitalize()} (cond)'
                )
    else:
        if not has_orbit_cols:
            logger.info("Orbit metadata columns not present — skipping anomaly re-computation.")
        if not has_xco2_bc:
            logger.info("'xco2_bc' column not present — skipping anomaly re-computation.")

    # ── Save output ────────────────────────────────────────────────────────
    out_path = file_dir / Path(output_path).name
    if str(out_path).endswith('.parquet'):
        df_out.to_parquet(out_path, index=False, compression='zstd')
    else:
        df_out.to_csv(out_path, index=False)
    print(f"\nOutput saved → {out_path}  ({len(df_out):,} rows)", flush=True)
    added = [c for c in df_out.columns if c not in df.columns]
    print(f"  New columns: {added}", flush=True)

    # ── Save slim plot-data Parquet ────────────────────────────────────────
    _plot_data: dict = {}
    for _c in ('sounding_id', 'time', 'footprint_id', 'lon', 'lat',
               'cld_dist_km', 'xco2_bc', 'xco2_bc_anomaly',
               'cld_prob', 'cld_pred', 'cld_prob_lr', 'cld_prob_mlp_clf', 'cld_prob_ft_clf'):
        if _c in df_out.columns:
            _plot_data[_c] = df_out[_c].values

    for _model, _pred_col in [('ridge', 'ridge_pred'), ('mlp', 'mlp_pred'),
                               ('ft', 'ft_q50'), ('xgb', 'xgb_pred'), ('hybrid', 'hybrid_q50')]:
        if _pred_col not in df_out.columns:
            continue
        _plot_data[_pred_col] = df_out[_pred_col].to_numpy(dtype=np.float32)
        if has_xco2_bc:
            _plot_data[f'{_model}_corrected_xco2'] = (
                df_out['xco2_bc'].to_numpy(dtype=np.float32)
                - df_out[_pred_col].to_numpy(dtype=np.float32)
            )
        cond_col = f'{_model}_cond_corrected_xco2'
        if cond_col in df_out.columns:
            _plot_data[cond_col] = df_out[cond_col].to_numpy(dtype=np.float32)

    for _c in ('ft_uncertainty', 'hybrid_uncertainty',
               'ridge_anomaly', 'mlp_anomaly', 'ft_anomaly', 'xgb_anomaly', 'hybrid_anomaly',
               'ridge_cond_anomaly', 'mlp_cond_anomaly', 'ft_cond_anomaly',
               'xgb_cond_anomaly', 'hybrid_cond_anomaly'):
        if _c in df_out.columns:
            _plot_data[_c] = df_out[_c].values

    _plot_path = file_dir / f'plot_data{suffix_tag}.parquet'
    pd.DataFrame(_plot_data, index=df_out.index).to_parquet(
        _plot_path, index=False, compression='zstd'
    )
    print(f"Plot data saved → {_plot_path}  ({len(_plot_data)} cols)", flush=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    # Standard comparison plots
    if 'xco2_bc_anomaly' in df_out.columns:
        print(f"\nGenerating standard comparison plots → {file_dir}", flush=True)
        _comparison_plots(df_out, available, file_dir)

    # Cloud-probability map (new panel)
    prob_col = ('cld_prob' if 'cld_prob' in df_out.columns
                else clf_proba_cols[0] if clf_proba_cols else None)
    if prob_col is not None:
        _plot_cloud_prob_panel(df_out, prob_col, file_dir)

    # Conditional correction maps
    if has_xco2_bc and prob_col is not None:
        for _model, _pred_col in [('ridge', 'ridge_pred'), ('mlp', 'mlp_pred'),
                                   ('ft', 'ft_q50'), ('xgb', 'xgb_pred'),
                                   ('hybrid', 'hybrid_q50')]:
            cond_col = f'{_model}_cond_corrected_xco2'
            if cond_col in df_out.columns:
                _plot_conditional_correction(
                    df_out, prob_col, _model.upper(), cond_col, file_dir
                )


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply regression + cloud-proximity classifier; "
            "conditional xco2_bc correction for cloud-affected soundings."
        )
    )
    parser.add_argument('--suffix',      type=str, default='')
    parser.add_argument('--pipeline',    default=None)
    parser.add_argument('--ridge-dir',   default=None)
    parser.add_argument('--mlp-dir',     default=None)
    parser.add_argument('--ft-dir',      default=None)
    parser.add_argument('--xgb-dir',     default=None)
    parser.add_argument('--hybrid-dir',  default=None)
    parser.add_argument('--clf-dir',     default=None,
                        help='Dir containing logistic_model.pkl / mlp_clf_weights.pt '
                             '(from mlp_lr_cloud_classifier.py). '
                             'Defaults to <storage_dir>/results/model_mlp_clf/<suffix>/')
    parser.add_argument('--ft-clf-dir',  default=None,
                        help='Dir containing ft_clf_weights.pt / ft_clf_meta.pkl '
                             '(from transformer_cloud_classifier.py). '
                             'Defaults to <storage_dir>/results/model_ft_clf/<suffix>/')
    parser.add_argument('--gate-clf',    default='ft_clf',
                        choices=['lr', 'mlp_clf', 'ft_clf'],
                        help='Which cloud-proximity classifier to use for gating the correction. '
                             'lr      → LogisticRegression from --clf-dir. '
                             'mlp_clf → MLP classifier from --clf-dir. '
                             'ft_clf  → FT-Transformer classifier from --ft-clf-dir (default).')
    parser.add_argument('--input-dir',   default=None)
    parser.add_argument('--input',       nargs='+', default=None)
    parser.add_argument('--output',      default=None)
    parser.add_argument('--plot-dir',    default=None)
    parser.add_argument('--sfc-type',    type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    storage_dir    = get_storage_dir()
    suffix         = args.suffix
    suffix_tag     = f'_{suffix}' if suffix else ''
    lr_dir         = storage_dir / 'results/model_mlp_lr'        / suffix if suffix else storage_dir / 'results/model_mlp_lr'
    ft_dir_def     = storage_dir / 'results/model_ft_transformer' / suffix if suffix else storage_dir / 'results/model_ft_transformer'
    xgb_dir_def    = storage_dir / 'results/model_xgb'           / suffix if suffix else storage_dir / 'results/model_xgb'
    hybrid_dir_def = storage_dir / 'results/model_hybrid'        / suffix if suffix else storage_dir / 'results/model_hybrid'
    clf_dir_def    = storage_dir / 'results/model_mlp_clf'        / suffix if suffix else storage_dir / 'results/model_mlp_clf'
    ft_clf_dir_def = storage_dir / 'results/model_ft_clf'         / suffix if suffix else storage_dir / 'results/model_ft_clf'

    def _abs(p):
        if p is None:
            return None
        pp = Path(p)
        return str(storage_dir / pp) if not pp.is_absolute() else p

    pipeline_path = _abs(args.pipeline)  or str(lr_dir / 'pipeline.pkl')
    ridge_dir     = _abs(args.ridge_dir) or str(lr_dir)
    mlp_dir       = _abs(args.mlp_dir)   or str(lr_dir)
    ft_dir        = _abs(args.ft_dir)    or str(ft_dir_def)
    xgb_dir       = _abs(args.xgb_dir)  or str(xgb_dir_def)
    hybrid_dir    = _abs(args.hybrid_dir) or str(hybrid_dir_def)
    clf_dir       = _abs(args.clf_dir)   or str(clf_dir_def)
    ft_clf_dir    = _abs(args.ft_clf_dir) or str(ft_clf_dir_def)

    # ── Resolve input paths ────────────────────────────────────────────────
    if args.input_dir:
        _idir = Path(str(_abs(args.input_dir) or args.input_dir))
        if args.input:
            input_paths: list[str] = [str(_idir / f) for f in args.input]
        else:
            input_paths = sorted(
                [str(p) for p in _idir.glob('*.parquet')] +
                [str(p) for p in _idir.glob('*.csv')]
            )
    elif args.input:
        input_paths = [str(_abs(p) or p) for p in args.input]
    else:
        input_paths = [str(storage_dir / 'results/csv_collection/combined_2020_dates.parquet')]

    multi = len(input_paths) > 1

    # ── Load pipeline ──────────────────────────────────────────────────────
    print(f"Loading pipeline: {pipeline_path}", flush=True)
    pipeline = FeaturePipeline.load(pipeline_path)
    print(f"  {pipeline}", flush=True)

    sfc_type = args.sfc_type if args.sfc_type is not None else pipeline.sfc_type

    # ── Load regression adapters ───────────────────────────────────────────
    print("\nLoading regression model adapters …", flush=True)
    ridge  = _try_load(RidgeAdapter,   ridge_dir,  'Ridge')
    mlp    = _try_load(MLPAdapter,     mlp_dir,    'MLP')
    ft     = _load_ft_with_bootstrap(ft_dir)
    xgb    = _try_load(XGBoostAdapter, xgb_dir,    'XGBoost')
    hybrid = _try_load(HybridAdapter,  hybrid_dir, 'Hybrid')

    available = {name for name, adapter in [
        ('ridge', ridge), ('mlp', mlp), ('ft', ft), ('xgboost', xgb), ('hybrid', hybrid)
    ] if adapter is not None}

    # ── Load cloud classifiers ─────────────────────────────────────────────
    print("\nLoading cloud-proximity classifiers …", flush=True)
    lr_clf, mlp_clf = _load_clf(clf_dir)
    ft_clf          = _load_ft_clf(ft_clf_dir)

    gate_clf = args.gate_clf

    any_clf = any(c is not None for c in (lr_clf, mlp_clf, ft_clf))
    if not any_clf:
        print("WARNING: No cloud classifiers found.  Correction will be applied to ALL "
              "soundings (equivalent to apply_models.py).", flush=True)
    else:
        print(f"  Gate classifier selected: --gate-clf {gate_clf}", flush=True)

    if not available and not any_clf:
        print("ERROR: No adapters found at all.", file=sys.stderr)
        sys.exit(1)

    # ── Output paths ───────────────────────────────────────────────────────
    _results_dir = storage_dir / 'results'
    if suffix:
        _results_dir = _results_dir / suffix
    base_dir = _abs(args.plot_dir) or str(_results_dir)
    out_name = (Path(args.output).name if (not multi and args.output)
                else f'corrected_cld{suffix_tag}.parquet')

    # ── Process ────────────────────────────────────────────────────────────
    for input_path in input_paths:
        _process_one_file(
            input_path=input_path,
            output_path=out_name,
            plot_dir=base_dir,
            suffix_tag=suffix_tag,
            pipeline=pipeline,
            ridge=ridge, mlp=mlp, ft=ft, xgb=xgb, hybrid=hybrid,
            available=available,
            lr_clf=lr_clf, mlp_clf=mlp_clf, ft_clf=ft_clf,
            gate_clf=gate_clf,
            sfc_type=sfc_type,
        )

    if multi:
        print(f"\nDone — processed {len(input_paths)} files.", flush=True)


if __name__ == '__main__':
    main()
