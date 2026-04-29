"""apply_models.py — apply trained Ridge/MLP/FT/XGBoost/Hybrid models to new data.

Loads a shared FeaturePipeline and whichever model adapters are available,
applies all of them to an input CSV, and writes an output CSV with the
predicted correction columns appended.

Optionally:
  - recomputes XCO2 anomaly on corrected fields (when orbit metadata present)
  - generates cross-model comparison plots (when ground-truth anomaly present)

Usage:
    # Single explicit file
    python src/apply_models.py \\
        --pipeline   results/exp_v1/pipeline.pkl \\
        --ridge-dir  results/model_mlp_lr/exp_v1/ \\
        --input      new_data.csv \\
        --output     corrected.csv

    # All files in a directory (models loaded once, reused for each file)
    python src/apply_models.py \\
        --pipeline   results/exp_v1/pipeline.pkl \\
        --input-dir  results/csv_collection/

    # Mix: directory glob + extra explicit files
    python src/apply_models.py \\
        --input-dir  results/csv_collection/ \\
        --input      extra_a.parquet extra_b.parquet

Output layout (per input file, under <plot_dir>/<input_stem>/):
    corrected<suffix>.parquet        ← full prediction output
    plot_data<suffix>.parquet        ← slim spatial + prediction file for plotting
    comparison_scatter.png           ┐
    comparison_hist.png              ├ comparison plots (when xco2_bc_anomaly present)
    anomaly_comparison.png           │
    xco2bc_comparison_regime.png     ┘

Prediction columns added to output:
    ridge_pred, mlp_pred, ft_q05, ft_q50, ft_q95, ft_uncertainty, xgb_pred,
    hybrid_q05, hybrid_q50, hybrid_q95, hybrid_uncertainty
    (+ ridge_anomaly, mlp_anomaly, ft_anomaly, xgb_anomaly, hybrid_anomaly
       when orbit metadata present)
"""

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from models.pipeline import FeaturePipeline
from models.adapters import RidgeAdapter, MLPAdapter, FTAdapter, XGBoostAdapter, HybridAdapter
from utils import get_storage_dir

logger = logging.getLogger(__name__)

_ANOMALY_ARGS = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
_ORBIT_COLS   = {'date', 'orbit_id', 'lat', 'cld_dist_km'}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _try_load(adapter_cls, adapter_dir, label: str):
    """Load adapter if files exist, else return None with a warning."""
    if adapter_dir is None:
        return None
    if not adapter_cls.can_load(adapter_dir):
        logger.warning("%s: adapter files not found in %s — skipping.", label, adapter_dir)
        return None
    try:
        adapter = adapter_cls.load(adapter_dir)
        print(f"  [loaded] {label} ← {adapter_dir}", flush=True)
        return adapter
    except Exception as exc:
        logger.warning("%s: load failed (%s) — skipping.", label, exc)
        return None


def _infer_ft_meta_from_checkpoint(ckpt_path: Path) -> dict:
    """Infer FTAdapter meta dict from a model_best.pt state dict.

    Reads key shapes to recover n_features, d_token, n_layers, d_ff, and
    tokenizer_type.  n_heads cannot be read from weights; defaults to 8 (the
    training-script default), falling back to 4 if d_token is not divisible.
    feature_names is set to a list of dummy strings when group_emb is present
    so that UncertainFTTransformerRefined creates group_emb; the actual
    feature_to_group buffer is then overwritten by load_state_dict.
    """
    import torch
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd   = ckpt['model_state_dict']

    # ── tokenizer type, n_features, d_token ───────────────────────────────
    if 'tokenizer.weight' in sd:
        tokenizer_type = 'linear'
        n_features     = sd['tokenizer.weight'].shape[0]
        d_token        = sd['tokenizer.weight'].shape[1]
    else:
        tokenizer_type = 'mlp'
        tok_idxs: set[int] = set()
        for k in sd:
            m = re.match(r'tokenizer\.tokenizers\.(\d+)\.', k)
            if m:
                tok_idxs.add(int(m.group(1)))
        n_features = len(tok_idxs)
        # MLPTokenizer: tokenizers[i] = Linear(1→d//2) + GELU + Linear(d//2→d)
        d_token = int(sd['tokenizer.tokenizers.0.2.weight'].shape[0])

    # ── n_layers ──────────────────────────────────────────────────────────
    layer_idxs: set[int] = set()
    for k in sd:
        m = re.match(r'layers\.(\d+)\.', k)
        if m:
            layer_idxs.add(int(m.group(1)))
    n_layers = len(layer_idxs) if layer_idxs else 4

    # ── d_ff (GRN hidden size = first Linear in dense seq) ───────────────
    d_ff = int(sd['layers.0.ff.dense.0.weight'].shape[0])

    # ── n_heads: not encoded in weights; use training default ─────────────
    n_heads = 8
    if d_token % n_heads != 0:
        n_heads = 4   # fall back to next common divisor

    # ── group_emb present → need dummy feature_names to trigger creation ──
    has_group_emb = 'group_emb.weight' in sd
    feature_names = [f'_f{i}' for i in range(n_features)] if has_group_emb else None

    return {
        'n_features':     n_features,
        'd_token':        d_token,
        'n_heads':        n_heads,
        'n_layers':       n_layers,
        'd_ff':           d_ff,
        'tokenizer_type': tokenizer_type,
        'feature_names':  feature_names,
        'arch_version':   FTAdapter._ARCH_VERSION,
    }


def _load_ft_with_bootstrap(ft_dir: str | None) -> 'FTAdapter | None':
    """Load FTAdapter, creating ft_meta.pkl from model_best.pt if absent.

    Normal path  : both ft_meta.pkl and model_best.pt present → _try_load.
    Bootstrap    : model_best.pt exists but ft_meta.pkl missing → infer arch
                   params from state dict, write ft_meta.pkl, then load.
    Missing ckpt : return None with a warning (nothing to bootstrap from).
    """
    if ft_dir is None:
        return None

    out       = Path(ft_dir)
    ckpt_path = out / FTAdapter.CHECKPOINT_FILE
    meta_path = out / FTAdapter.META_FILE

    if FTAdapter.can_load(ft_dir):
        return _try_load(FTAdapter, ft_dir, 'FT')

    if ckpt_path.exists() and not meta_path.exists():
        print(f"  [FT] ft_meta.pkl missing — inferring from {ckpt_path.name} …", flush=True)
        try:
            meta = _infer_ft_meta_from_checkpoint(ckpt_path)
        except Exception as exc:
            logger.warning("FT: could not infer meta from checkpoint (%s) — skipping.", exc)
            return None

        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        print(
            f"  [FT] ft_meta.pkl created  "
            f"n_features={meta['n_features']}  d_token={meta['d_token']}  "
            f"n_layers={meta['n_layers']}  d_ff={meta['d_ff']}  "
            f"tokenizer={meta['tokenizer_type']}  "
            f"group_emb={'yes' if meta['feature_names'] else 'no'}",
            flush=True,
        )
        return _try_load(FTAdapter, ft_dir, 'FT')

    logger.warning("FT: adapter files not found in %s — skipping.", ft_dir)
    return None


def _recompute_anomaly(df: pd.DataFrame,
                       xco2_corrected: np.ndarray,
                       label: str) -> np.ndarray:
    """Call compute_xco2_anomaly_date_id on a corrected XCO2 field."""
    from models.mlp_lr import compute_xco2_anomaly_date_id
    print(f"  Computing anomaly for {label} corrected XCO2 …", flush=True)
    return compute_xco2_anomaly_date_id(
        df['date'], df['orbit_id'],
        df['lat'].to_numpy(), df['cld_dist_km'].to_numpy(),
        xco2_corrected,
        lat_thres=_ANOMALY_ARGS['lat_thres'],
        std_thres=_ANOMALY_ARGS['std_thres'],
        min_cld_dist=_ANOMALY_ARGS['min_cld_dist'],
    )


def _plot_regime_comparison(df_out: pd.DataFrame, active: list,
                            plot_dir: Path) -> None:
    """3-row × (n_models+3)-col regime comparison plot.

    Replicates mlp_lr_models.py lines 599-743, adapted for apply_models.py.

    Rows
    ----
    1. Clear-sky FPs (cld_dist > 10 km)  — from rows with finite xco2_bc_anomaly
    2. Cloud-affected FPs (cld_dist ≤ 10 km) — same subset
    3. Cloud FPs with NaN anomaly (cld_dist ≤ 10 km) — full df, anomaly absent

    Columns (per row)
    -----------------
    0..n_models-1 : scatter  original vs model prediction
    n_models      : distribution histogram  (original + each model)
    n_models+1    : corrected-XCO2 histogram
    n_models+2    : ideal corrected-XCO2 histogram
    """
    required = {'xco2_bc', 'xco2_bc_anomaly', 'cld_dist_km'}
    if not required.issubset(df_out.columns):
        logger.info("_plot_regime_comparison: missing %s — skipping.",
                    required - set(df_out.columns))
        return

    xco2_bc_all  = df_out['xco2_bc'].to_numpy(dtype=float)
    anom_all     = df_out['xco2_bc_anomaly'].to_numpy(dtype=float)
    cld_dist_all = df_out['cld_dist_km'].to_numpy(dtype=float)

    # Rows with valid anomaly (mirrors df_xco2_anomaly in mlp_lr_models.py)
    sub_mask = np.isfinite(anom_all)
    anom_sub = anom_all[sub_mask]
    xco2_sub = xco2_bc_all[sub_mask]
    cld_sub  = cld_dist_all[sub_mask]

    mask_r1 = cld_sub > 10
    mask_r2 = cld_sub <= 10
    mask_r3 = (cld_dist_all <= 10) & ~np.isfinite(anom_all)

    # Per-model arrays (sub = anomaly-valid rows, all = full df)
    model_cfgs = []   # (name, color, pred_sub, pred_all)
    for name, pred_col, color in active:
        if pred_col not in df_out.columns:
            continue
        pred_all = df_out[pred_col].to_numpy(dtype=float)
        model_cfgs.append((name, color, pred_all[sub_mask], pred_all))

    if not model_cfgs:
        return

    n_models = len(model_cfgs)
    n_cols   = n_models + 3

    # row_configs: (xco2_orig, xco2_orig_bc, mask, label, is_anomaly)
    row_configs = [
        (anom_sub,    xco2_sub,    mask_r1,
         'Clear-sky FPs (cld_dist > 10 km)',               True),
        (anom_sub,    xco2_sub,    mask_r2,
         'Cloud-affected FPs (cld_dist \u2264 10 km)',     True),
        (xco2_bc_all, xco2_bc_all, mask_r3,
         'Cloud FPs with NaN anomaly (cld_dist \u2264 10 km)', False),
    ]

    plt.close('all')
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols + 2, 17))

    for row_i, (xco2_orig, xco2_orig_bc, mask, row_label, is_anomaly) in enumerate(row_configs):
        scatter_axes = [axes[row_i, j] for j in range(n_models)]
        ax_h  = axes[row_i, n_models]
        ax_h2 = axes[row_i, n_models + 1]
        ax_h3 = axes[row_i, n_models + 2]

        x_orig   = xco2_orig[mask]
        x_orig_bc = xco2_orig_bc[mask]

        if np.isfinite(x_orig).sum() < 2:
            for ax in scatter_axes + [ax_h, ax_h2, ax_h3]:
                ax.set_visible(False)
            continue

        _lo = np.nanpercentile(x_orig[np.isfinite(x_orig)], 1)
        _hi = np.nanpercentile(x_orig[np.isfinite(x_orig)], 99)

        preds_row = []
        for (name, color, pred_sub, pred_all), ax in zip(model_cfgs, scatter_axes):
            x_pred = (pred_sub if row_i < 2 else pred_all)[mask]
            preds_row.append((name, color, x_pred))

            if not (x_orig == x_orig_bc).all():
                v = np.isfinite(x_orig) & np.isfinite(x_pred)
                ax.scatter(x_orig[v], x_pred[v], c=color, edgecolor=None, s=5, alpha=0.6)
                ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
                ax.set_aspect('equal', adjustable='box')
                ax.axline((_lo, _lo), slope=1, color='r', linestyle='--')
                ax.set_xlabel('Original XCO2_bc anomaly (ppm)' if is_anomaly
                              else 'Original XCO2_bc (ppm)')
                ax.set_ylabel(f'{name}-corrected XCO2_bc anomaly (ppm)' if is_anomaly
                              else f'{name}-corrected XCO2_bc (ppm)')
                ax.set_title(f'{row_label}\n[{name} scatter]')
                if v.sum() > 1:
                    r2 = 1 - np.nansum((x_orig[v] - x_pred[v])**2) / \
                             np.nansum((x_orig[v] - np.nanmean(x_orig[v]))**2)
                    ax.text(0.05, 0.95, f'R\u00b2={r2:.3f}', transform=ax.transAxes, va='top')
            else:
                ax.set_visible(False)

        # Histogram columns — mirrors mlp_lr_models.py lines 672-733
        hist_series = [('Original', 'blue', x_orig, x_orig_bc)] + \
                      [(nm, clr, x_pred, x_orig_bc) for nm, clr, x_pred in preds_row]

        for _label, _color, _xco2, _xco2_bc in hist_series:
            _v  = _xco2[np.isfinite(_xco2)]
            _v2 = _xco2_bc[np.isfinite(_xco2)]
            if len(_v) == 0:
                continue
            # Col: distribution of xco2 values
            if len(_v2) > 0 and not (_v == _v2).all():
                _mu, _sigma = _v.mean(), _v.std()
                _bins = np.linspace(np.nanpercentile(_v, 1), np.nanpercentile(_v, 99), 100)
                ax_h.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                          label=f'{_label}\n\u03bc={_mu:.3f}, \u03c3={_sigma:.3f}')
                ax_h.axvline(_mu,           color=_color, linestyle='-',  linewidth=1.2)
                ax_h.axvline(_mu - _sigma,  color=_color, linestyle=':',  linewidth=0.9)
                ax_h.axvline(_mu + _sigma,  color=_color, linestyle=':',  linewidth=0.9)

            # Col: corrected XCO2_bc distribution
            if _label != 'Original':
                _v2 = _v2 - _v
            _mu2, _sigma2 = _v2.mean(), _v2.std()
            _bins2 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
            ax_h2.hist(_v2, bins=_bins2, color=_color, alpha=0.3, density=True,
                       label=f'{_label}\n\u03bc={_mu2:.3f}, \u03c3={_sigma2:.3f}')
            ax_h2.axvline(_mu2,           color=_color, linestyle='-',  linewidth=1.0)
            ax_h2.axvline(_mu2 - _sigma2, color=_color, linestyle=':',  linewidth=0.8)
            ax_h2.axvline(_mu2 + _sigma2, color=_color, linestyle=':',  linewidth=0.8)

            # Col: ideal corrected XCO2 distribution
            if _label == 'Original' and not (_v == _v2).all():
                _v2_h3 = _v2 - _v
                _mu3, _sigma3 = _v2_h3.mean(), _v2_h3.std()
                bins3 = np.linspace(np.nanpercentile(_v2_h3, 1), np.nanpercentile(_v2_h3, 99), 100)
                ax_h3.hist(_v2_h3, bins=bins3, color=_color, alpha=0.2, density=True,
                           label=f'Ideal {_label}-corrected\n\u03bc={_mu3:.3f}, \u03c3={_sigma3:.3f}')
                ax_h3.axvline(_mu3,           color=_color, linestyle='-',  linewidth=0.8)
                ax_h3.axvline(_mu3 - _sigma3, color=_color, linestyle=':',  linewidth=0.6)
                ax_h3.axvline(_mu3 + _sigma3, color=_color, linestyle=':',  linewidth=0.6)
            elif (_v == _v2).all():
                ax_h3.set_visible(False)
            else:
                _mu3, _sigma3 = _v2.mean(), _v2.std()
                bins3 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
                ax_h3.hist(_v2, bins=bins3, color=_color, alpha=0.2, density=True,
                           label=f'{_label}\n\u03bc={_mu3:.3f}, \u03c3={_sigma3:.3f}')
                ax_h3.axvline(_mu3,           color=_color, linestyle='-',  linewidth=0.8)
                ax_h3.axvline(_mu3 - _sigma3, color=_color, linestyle=':',  linewidth=0.6)
                ax_h3.axvline(_mu3 + _sigma3, color=_color, linestyle=':',  linewidth=0.6)

        ax_h.set_title(f'{row_label}\n[Distribution]')
        ax_h.set_xlabel('XCO2_bc anomaly (ppm)' if is_anomaly else 'XCO2_bc (ppm)')
        ax_h.legend(fontsize=9)
        ax_h2.set_title(f'{row_label}\n[Corrected distribution]')
        ax_h2.set_xlabel('Corrected XCO2_bc (ppm)')
        ax_h2.legend(fontsize=9)
        ax_h3.set_title(f'{row_label}\n[Ideal corrected XCO2 distribution]')
        ax_h3.legend(fontsize=9)

    fig.suptitle('XCO2_bc by cloud-distance regime', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(plot_dir / 'xco2bc_comparison_regime.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved xco2bc_comparison_regime.png")


def _plot_anomaly_scatter_hist(df_out: pd.DataFrame, active: list,
                               true_anom: np.ndarray, plot_dir: Path) -> None:
    """Combined scatter + histogram figure — mirrors mlp_lr_models.py lines 545-600.

    Layout: 1 row × (n_models + 1) columns
      cols 0..n_models-1 : scatter  original anomaly vs recomputed anomaly after
                           correction (uses {name.lower()}_anomaly col; falls back
                           to raw prediction col if the anomaly col is absent)
      col  n_models       : histogram of original + each model's corrected anomaly

    The recomputed anomaly columns (ridge_anomaly, mlp_anomaly, ft_anomaly) are
    written by main() when orbit metadata is present, matching the mlp_lr_models.py
    compute_xco2_anomaly_date_id step.
    """
    # Map model name → recomputed anomaly column
    _ANOM_COL = {'Ridge': 'ridge_anomaly', 'MLP': 'mlp_anomaly', 'FT': 'ft_anomaly',
                 'XGBoost': 'xgb_anomaly', 'Hybrid': 'hybrid_anomaly'}

    # Only include models that have a proper recomputed anomaly column.
    # The raw prediction col (ridge_pred / mlp_pred / ft_q50) is the model's
    # estimate of the original anomaly — correlated with true_anom by design —
    # so it must NOT be used in the histogram or scatter as a recomputed anomaly.
    active_anom = []
    for name, _, color in active:
        anom_col = _ANOM_COL.get(name)
        if anom_col and anom_col in df_out.columns:
            active_anom.append((name, anom_col, color))
        else:
            logger.debug(
                "%s recomputed anomaly column '%s' not found — skipping in anomaly histogram",
                name, anom_col,
            )

    valid = np.isfinite(true_anom)
    if valid.sum() < 2:
        return

    n_models = len(active_anom)
    n_cols   = n_models + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    _lim = np.nanpercentile(np.abs(true_anom[valid]), 99)

    for ax, (name, col, color) in zip(axes[:n_models], active_anom):
        corr_anom = df_out[col].to_numpy(dtype=float)
        v         = valid & np.isfinite(corr_anom)
        ax.scatter(true_anom[v], corr_anom[v], c=color, edgecolor=None, s=5, alpha=0.6)
        ax.set_xlim(-_lim, _lim);  ax.set_ylim(-_lim, _lim)
        ax.set_aspect('equal', adjustable='box')
        ax.axline((0, 0), slope=1, color='r', linestyle='--')
        ax.set_xlabel('Original XCO2_bc anomaly (ppm)')
        ax.set_ylabel(f'{name}-corrected XCO2_bc anomaly (ppm)')
        ax.set_title(f'Original vs {name}-corrected anomaly')
        if v.sum() > 1:
            r2  = 1 - np.nansum((true_anom[v] - corr_anom[v])**2) / \
                      np.nansum((true_anom[v] - np.nanmean(true_anom[v]))**2)
            mae = float(np.abs(true_anom[v] - corr_anom[v]).mean())
            ax.text(0.05, 0.95, f'R²={r2:.3f}\nMAE={mae:.4f}',
                    transform=ax.transAxes, va='top')

    # Histogram panel
    ax_hist = axes[n_models]
    _bins = np.linspace(-3, 3, 211)
    for anom, color, label in [
            (true_anom, 'blue', 'Original'),
            *[(df_out[col].to_numpy(dtype=float), clr, nm) for nm, col, clr in active_anom],
    ]:
        _v = anom[np.isfinite(anom)]
        _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
        ax_hist.hist(_v, bins=_bins, color=color, alpha=0.6, density=True,
                     label=f'{label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
        ax_hist.axvline(_mu,          color=color, linestyle='-',  linewidth=1.2)
        ax_hist.axvline(_mu - _sigma, color=color, linestyle=':', linewidth=0.9)
        ax_hist.axvline(_mu + _sigma, color=color, linestyle=':', linewidth=0.9)
    ax_hist.set_xlabel('XCO2_bc anomaly (ppm)')
    ax_hist.set_title('Anomaly distribution comparison')
    ax_hist.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax_hist.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(plot_dir / 'anomaly_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved anomaly_comparison.png")


def _comparison_plots(df_out: pd.DataFrame, available: set,
                      plot_dir: Path) -> None:
    """Scatter + histogram comparison plots when ground-truth anomaly exists."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    true_anom = df_out['xco2_bc_anomaly'].to_numpy(dtype=float)
    valid     = np.isfinite(true_anom)
    if valid.sum() < 10:
        logger.warning("Too few labelled samples for comparison plots — skipping.")
        return

    # ── Scatter panels ─────────────────────────────────────────────────────
    pred_cols = {
        'Ridge':   ('ridge_pred',  'orange'),
        'MLP':     ('mlp_pred',    'limegreen'),
        'FT':      ('ft_q50',      'purple'),
        'XGBoost': ('xgb_pred',    'crimson'),
        'Hybrid':  ('hybrid_q50',  'deepskyblue'),
    }
    active = [(name, col, color)
              for name, (col, color) in pred_cols.items()
              if name.lower() in available and col in df_out.columns]

    if active:
        n  = len(active)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        _lim = np.nanpercentile(np.abs(true_anom[valid]), 99)
        metrics_rows = []

        for ax, (name, col, color) in zip(axes, active):
            pred = df_out[col].to_numpy(dtype=float)
            v    = valid & np.isfinite(pred)
            ax.scatter(true_anom[v], pred[v], c=color, edgecolor=None, s=5, alpha=0.5)
            ax.set_xlim(-_lim, _lim);  ax.set_ylim(-_lim, _lim)
            ax.set_aspect('equal', adjustable='box')
            ax.axline((0, 0), slope=1, color='r', linestyle='--', linewidth=0.8)
            ax.set_xlabel('True XCO2_bc anomaly (ppm)')
            ax.set_ylabel(f'{name} predicted anomaly (ppm)')
            ax.set_title(name)
            if v.sum() > 1:
                r2  = 1 - np.nansum((true_anom[v] - pred[v])**2) / \
                          np.nansum((true_anom[v] - np.nanmean(true_anom[v]))**2)
                mae = float(np.abs(true_anom[v] - pred[v]).mean())
                ax.text(0.05, 0.95, f'R²={r2:.3f}\nMAE={mae:.4f}',
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                metrics_rows.append({'model': name, 'R2': r2, 'MAE': mae,
                                     'sigma': float(pred[v].std()), 'n': int(v.sum())})

        fig.suptitle('Cross-model comparison — anomaly scatter', fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_dir / 'comparison_scatter.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved comparison_scatter.png")

        # ── Histogram ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        _bins = np.linspace(-3, 3, 211)
        for anom, color, label in [
                (true_anom, 'blue', 'Original'),
                *[(df_out[col].to_numpy(dtype=float), clr, nm)
                  for nm, col, clr in active],
        ]:
            _v = anom[np.isfinite(anom)]
            _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
            ax.hist(_v, bins=_bins.tolist(), color=color, alpha=0.5, density=True,
                    label=f'{label}  μ={_mu:.3f} σ={_sigma:.3f}')
            ax.axvline(float(_mu), color=color, linewidth=1.0)
        ax.set_xlabel('XCO2_bc anomaly (ppm)')
        ax.set_title('Anomaly distribution comparison')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.7)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / 'comparison_hist.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved comparison_hist.png")

        # ── Metrics CSV ────────────────────────────────────────────────────
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(plot_dir / 'metrics.csv', index=False)
            logger.info("Saved metrics.csv")
            print("\n  ── Metrics ──────────────────────────", flush=True)
            for row in metrics_rows:
                print(f"  {row['model']:8s}  R²={row['R2']:.4f}  MAE={row['MAE']:.4f}  "
                      f"σ={row['sigma']:.4f}  n={row['n']:,}", flush=True)

        # ── Combined scatter+hist (mirrors mlp_lr_models.py lines 545-600) ──
        _plot_anomaly_scatter_hist(df_out, active, true_anom, plot_dir)

        # ── Regime comparison (3-row × n_models+3 grid) ────────────────────
        _plot_regime_comparison(df_out, active, plot_dir)


# ─── Per-file processing ───────────────────────────────────────────────────────

def _process_one_file(
    input_path: str,
    output_path: str,
    plot_dir: str,
    suffix_tag: str,
    pipeline,
    ridge, mlp, ft, xgb, hybrid,
    available: set,
    sfc_type: int,
) -> None:
    """Load one input file, predict with all loaded adapters, and save outputs.

    All outputs (parquet + figures) land flat under <plot_dir>/<input_stem>/.
    """
    file_dir = Path(plot_dir) / Path(input_path).stem
    file_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + filter input ────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {input_path}", flush=True)
    try:
        df = pd.read_parquet(input_path) if input_path.endswith('.parquet') else pd.read_csv(input_path)
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

    # ── Predict ────────────────────────────────────────────────────────────
    df_out = df.copy()

    if ridge is not None:
        print("Predicting with Ridge …", flush=True)
        df_out['ridge_pred'] = ridge.predict(X)

    if mlp is not None:
        print("Predicting with MLP …", flush=True)
        df_out['mlp_pred'] = mlp.predict(X)

    if ft is not None:
        print("Predicting with FT-Transformer …", flush=True)
        q05, q50, q95            = ft.predict_quantiles(X)
        df_out['ft_q05']         = q05
        df_out['ft_q50']         = q50
        df_out['ft_q95']         = q95
        df_out['ft_uncertainty'] = np.clip(q95 - q05, 0, None)

    if xgb is not None:
        print("Predicting with XGBoost …", flush=True)
        df_out['xgb_pred'] = xgb.predict(X)

    if hybrid is not None:
        print("Predicting with Hybrid …", flush=True)
        h_q05, h_q50, h_q95          = hybrid.predict_quantiles(X)
        df_out['hybrid_q05']          = h_q05
        df_out['hybrid_q50']          = h_q50
        df_out['hybrid_q95']          = h_q95
        df_out['hybrid_uncertainty']  = np.clip(h_q95 - h_q05, 0, None)

    # ── Anomaly re-computation ─────────────────────────────────────────────
    has_orbit_cols = _ORBIT_COLS.issubset(df_out.columns)
    has_xco2_bc    = 'xco2_bc' in df_out.columns

    if has_orbit_cols and has_xco2_bc:
        print("\nRecomputing XCO2 anomaly on corrected fields …", flush=True)
        xco2_bc = df_out['xco2_bc'].to_numpy(dtype=float)

        if ridge is not None:
            corrected = xco2_bc - df_out['ridge_pred'].to_numpy()
            df_out['ridge_anomaly'] = _recompute_anomaly(df_out, corrected, 'Ridge')

        if mlp is not None:
            corrected = xco2_bc - df_out['mlp_pred'].to_numpy()
            df_out['mlp_anomaly'] = _recompute_anomaly(df_out, corrected, 'MLP')

        if ft is not None:
            corrected = xco2_bc - df_out['ft_q50'].to_numpy()
            df_out['ft_anomaly'] = _recompute_anomaly(df_out, corrected, 'FT')

        if xgb is not None:
            corrected = xco2_bc - df_out['xgb_pred'].to_numpy()
            df_out['xgb_anomaly'] = _recompute_anomaly(df_out, corrected, 'XGBoost')

        if hybrid is not None:
            corrected = xco2_bc - df_out['hybrid_q50'].to_numpy()
            df_out['hybrid_anomaly'] = _recompute_anomaly(df_out, corrected, 'Hybrid')
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

    # ── Save plot-data (Parquet) ───────────────────────────────────────────
    _has_xco2_bc = 'xco2_bc' in df_out.columns
    _plot_data: dict = {}
    for _c in ('sounding_id', 'time', 'footprint_id', 'lon', 'lat', 'cld_dist_km', 'xco2_bc', 'xco2_bc_anomaly'):
        if _c in df_out.columns:
            _plot_data[_c] = df_out[_c].values
    for _model, _pred_col in [('ridge', 'ridge_pred'), ('mlp', 'mlp_pred'),
                               ('ft', 'ft_q50'), ('xgb', 'xgb_pred'),
                               ('hybrid', 'hybrid_q50')]:
        if _pred_col not in df_out.columns:
            continue
        _plot_data[_pred_col] = df_out[_pred_col].to_numpy(dtype=np.float32)
        if _has_xco2_bc:
            _plot_data[f'{_model}_corrected_xco2'] = (
                df_out['xco2_bc'].to_numpy(dtype=np.float32)
                - df_out[_pred_col].to_numpy(dtype=np.float32)
            )
    for _c in ('ft_uncertainty', 'hybrid_uncertainty',
               'ridge_anomaly', 'mlp_anomaly', 'ft_anomaly', 'xgb_anomaly', 'hybrid_anomaly'):
        if _c in df_out.columns:
            _plot_data[_c] = df_out[_c].values
    _plot_path = file_dir / f'plot_data{suffix_tag}.parquet'
    pd.DataFrame(_plot_data, index=df_out.index).to_parquet(_plot_path, index=False, compression='zstd')
    print(f"Plot data saved → {_plot_path}  ({len(_plot_data)} cols: {list(_plot_data)})",
          flush=True)

    # ── Comparison plots ───────────────────────────────────────────────────
    if plot_dir and 'xco2_bc_anomaly' in df_out.columns:
        print(f"\nGenerating comparison plots → {file_dir}", flush=True)
        _comparison_plots(df_out, available, file_dir)
    elif plot_dir:
        logger.info("'xco2_bc_anomaly' column not present — comparison plots skipped.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained Ridge/MLP/FT models to one or more input files."
    )
    parser.add_argument('--suffix',    type=str, default='',
                        help='Experiment subfolder used to derive default model dirs and output paths '
                             '(e.g. --suffix exp_v1).  Ignored when the corresponding explicit '
                             'path arg is supplied.')
    parser.add_argument('--pipeline',  default=None,
                        help='Path to pipeline.pkl.  '
                             'Defaults to <storage_dir>/results/model_mlp_lr/<suffix>/pipeline.pkl')
    parser.add_argument('--ridge-dir', default=None,
                        help='Dir containing ridge_model.pkl.  '
                             'Defaults to <storage_dir>/results/model_mlp_lr/<suffix>/')
    parser.add_argument('--mlp-dir',   default=None,
                        help='Dir containing mlp_meta.pkl + mlp_weights.pt.  '
                             'Defaults to <storage_dir>/results/model_mlp_lr/<suffix>/')
    parser.add_argument('--ft-dir',    default=None,
                        help='Dir containing ft_meta.pkl + model_best.pt.  '
                             'Defaults to <storage_dir>/results/model_ft_transformer/<suffix>/')
    parser.add_argument('--xgb-dir',   default=None,
                        help='Dir containing xgb_model.ubj + xgb_meta.pkl.  '
                             'Defaults to <storage_dir>/results/model_xgb/<suffix>/')
    parser.add_argument('--hybrid-dir', default=None,
                        help='Dir containing model_hybrid_best.pt + hybrid_meta.pkl.  '
                             'Defaults to <storage_dir>/results/model_hybrid/<suffix>/')
    parser.add_argument('--input-dir',  default=None,
                        help='Directory of input files; all *.parquet and *.csv files inside are '
                             'processed in sorted order.  Combined with --input when both are given.')
    parser.add_argument('--input',     nargs='+', default=None,
                        help='One or more filenames to process.  When --input-dir is also given, '
                             'these are treated as basenames inside that directory (no glob).  '
                             'When --input-dir is absent, treated as full/relative paths.  '
                             'Defaults to combined_2020_dates.parquet when neither arg is supplied.')
    parser.add_argument('--output',    default=None,
                        help='Output filename (basename only, directory is auto-derived per input).  '
                             'Ignored when more than one input file is resolved.  '
                             'Defaults to corrected<suffix>.parquet')
    parser.add_argument('--plot-dir',  default=None,
                        help='Base output directory.  Each input file writes all outputs '
                             '(parquet + figures) flat into <plot_dir>/<input_stem>/.  '
                             'Ignored when more than one input file is resolved (auto-derived).  '
                             'Defaults to <storage_dir>/results/')
    parser.add_argument('--sfc-type',  type=int, default=None,
                        help='Filter input by sfc_type (0=ocean, 1=land). '
                             'Defaults to the pipeline\'s sfc_type.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    storage_dir  = get_storage_dir()
    suffix       = args.suffix
    lr_dir       = storage_dir / 'results/model_mlp_lr'    / suffix if suffix else storage_dir / 'results/model_mlp_lr'
    ft_dir_def   = storage_dir / 'results/model_ft_transformer' / suffix if suffix else storage_dir / 'results/model_ft_transformer'
    xgb_dir_def    = storage_dir / 'results/model_xgb'    / suffix if suffix else storage_dir / 'results/model_xgb'
    hybrid_dir_def = storage_dir / 'results/model_hybrid' / suffix if suffix else storage_dir / 'results/model_hybrid'
    suffix_tag   = f'_{suffix}' if suffix else ''

    def _abs(p):
        """Resolve a relative path against storage_dir; leave absolute paths unchanged."""
        if p is None:
            return None
        pp = Path(p)
        return str(storage_dir / pp) if not pp.is_absolute() else p

    pipeline_path = _abs(args.pipeline) or str(lr_dir / 'pipeline.pkl')
    ridge_dir     = _abs(args.ridge_dir) or str(lr_dir)
    mlp_dir       = _abs(args.mlp_dir)   or str(lr_dir)
    ft_dir        = _abs(args.ft_dir)     or str(ft_dir_def)
    xgb_dir       = _abs(args.xgb_dir)   or str(xgb_dir_def)
    hybrid_dir    = _abs(args.hybrid_dir) or str(hybrid_dir_def)

    if args.input_dir:
        _idir = Path(str(_abs(args.input_dir) or args.input_dir))
        if args.input:
            # Only the named files within input_dir — no directory glob.
            input_paths: list[str] = [str(_idir / f) for f in args.input]
        else:
            # No explicit files given — glob everything in the directory.
            input_paths = sorted(
                [str(p) for p in _idir.glob('*.parquet')] +
                [str(p) for p in _idir.glob('*.csv')]
            )
            if not input_paths:
                print(f"WARNING: --input-dir {_idir} contains no .parquet or .csv files.",
                      file=sys.stderr)
    elif args.input:
        input_paths = [str(_abs(p) or p) for p in args.input]
    else:
        input_paths = [str(storage_dir / 'results/csv_collection/combined_2020_dates.parquet')]

    multi = len(input_paths) > 1
    if multi and args.output:
        logger.warning("--output is ignored when more than one input file is resolved; "
                       "output filename defaults to corrected<suffix>.parquet per file.")

    # ── Load pipeline ──────────────────────────────────────────────────────
    print(f"Loading pipeline: {pipeline_path}", flush=True)
    pipeline = FeaturePipeline.load(pipeline_path)
    print(f"  {pipeline}", flush=True)

    sfc_type = args.sfc_type if args.sfc_type is not None else pipeline.sfc_type

    # ── Load adapters (once) ───────────────────────────────────────────────
    ridge  = _try_load(RidgeAdapter,   ridge_dir,  'Ridge')
    mlp    = _try_load(MLPAdapter,     mlp_dir,    'MLP')
    ft     = _load_ft_with_bootstrap(ft_dir)
    xgb    = _try_load(XGBoostAdapter, xgb_dir,    'XGBoost')
    hybrid = _try_load(HybridAdapter,  hybrid_dir, 'Hybrid')

    available = {name for name, adapter in [('ridge', ridge), ('mlp', mlp),
                                             ('ft', ft), ('xgboost', xgb),
                                             ('hybrid', hybrid)]
                 if adapter is not None}
    if not available:
        print("ERROR: No adapter files found. Supply at least one of "
              "--ridge-dir, --mlp-dir, --ft-dir, --xgb-dir, --hybrid-dir.", file=sys.stderr)
        sys.exit(1)

    # base_dir: shared root under which each file gets its own <input_stem>/ subfolder.
    # --plot-dir is always honoured; only --output (filename) is ignored in multi mode.
    _results_dir = storage_dir / 'results'
    if suffix:
        _results_dir = _results_dir / suffix
    base_dir = _abs(args.plot_dir) or str(_results_dir)
    out_name = Path(args.output).name if (not multi and args.output) \
               else f'corrected{suffix_tag}.parquet'

    # ── Process each input file ────────────────────────────────────────────
    for input_path in input_paths:
        _process_one_file(
            input_path=input_path,
            output_path=out_name,
            plot_dir=base_dir,
            suffix_tag=suffix_tag,
            pipeline=pipeline,
            ridge=ridge, mlp=mlp, ft=ft, xgb=xgb, hybrid=hybrid,
            available=available,
            sfc_type=sfc_type,
        )

    if multi:
        print(f"\nDone — processed {len(input_paths)} files.", flush=True)


if __name__ == '__main__':
    main()
