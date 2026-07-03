"""tabm_eval.py — shared loss functions and evaluation/plot utilities for TabM.

Extracted verbatim from the retired transformer.py (FT-Transformer regressor).
TabM is the sole consumer; these are model-agnostic (operate on a `model` with a
batched predict path) so they were split out when the FT-Transformer, Hybrid,
MLP/LR trainers and the classifier suite were removed.

Contents
--------
Losses (torch): quantile_loss, huber_pinball_loss, variance_penalty, mmd_loss_1d
Helper:         _batched_predict
Eval/plots:     evaluate_model_X_text, plot_evaluation_by_regime,
                plot_permutation_importance
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .pipeline import compute_xco2_anomaly_date_id

logger = logging.getLogger(__name__)

def quantile_loss(preds, targets, quantiles):
    """Pinball / quantile loss for all three outputs.

    preds: [batch, 3]
    targets: [batch]
    quantiles: [0.05, 0.5, 0.95]
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        # Pinball loss formula
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(-1))

    return torch.mean(torch.cat(losses, dim=-1))


def huber_pinball_loss(preds, targets, quantiles, delta: float = 1.0):
    """Combined loss: Huber for q50 (index 1), pinball for q05/q95 (indices 0, 2).

    Using Huber for the median makes the point-prediction robust to the heavy
    left tail of cloud-affected XCO2 anomalies (which can reach −3 ppm) while
    still behaving like MSE for typical clear-sky anomalies near 0.  The q05/q95
    outputs retain their pinball loss so the uncertainty intervals stay calibrated.

    preds      : [batch, 3]
    targets    : [batch]
    quantiles  : [0.05, 0.5, 0.95]
    delta      : Huber transition point (ppm).  Errors with |e| ≤ delta are
                 penalised quadratically; larger errors linearly.
                 Recommended range for xco2_bc_anomaly: 0.5–1.0 ppm.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        if q == 0.5:
            # Huber loss: quadratic for |e| ≤ delta, linear beyond
            loss = torch.where(
                errors.abs() <= delta,
                0.5 * errors ** 2,
                delta * (errors.abs() - 0.5 * delta),
            )
        else:
            # Pinball loss for q05 / q95 — unchanged
            loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.unsqueeze(-1))

    return torch.mean(torch.cat(losses, dim=-1))


def variance_penalty(preds_q50: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """One-sided variance penalty: penalise std(preds_q50) < std(targets).

    Fires only when predictions are under-dispersed relative to the target
    distribution.  The one-sided clamp prevents fighting the pinball loss on
    q05/q95, which already exerts an outward force on the prediction spread.

    preds_q50 : [batch]  predicted median values
    targets   : [batch]  ground-truth values (raw ppm, not normalised)
    """
    pred_std = preds_q50.std(unbiased=False)
    tgt_std  = targets.std(unbiased=False).detach()   # fixed target; no gradient
    return torch.clamp(tgt_std - pred_std, min=0.0) ** 2


def mmd_loss_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximum Mean Discrepancy with Gaussian kernel for 1-D inputs.

    Uses the median heuristic for bandwidth: σ = median(|xi − yj|) over all
    N² cross-pairs (detached — σ is not part of the gradient graph).

    Requires batch size ≥ 64 for a stable σ estimate.  The recommended
    batch size at the call site is ≥ 1024.  For very small batches prefer
    variance_penalty instead.

    x : [N]  predicted q50 values within the mini-batch
    y : [N]  corresponding target values
    """
    with torch.no_grad():
        dists = (x.unsqueeze(1) - y.unsqueeze(0)).abs()   # [N, N]
        sigma = dists.median().clamp(min=1e-4)

    def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((a.unsqueeze(1) - b.unsqueeze(0)) ** 2) / (2.0 * sigma ** 2))

    kxx = _k(x, x).mean()
    kyy = _k(y, y).mean()
    kxy = _k(x, y).mean()
    return kxx - 2.0 * kxy + kyy


# Training setup
quantiles = [0.05, 0.5, 0.95]


def _batched_predict(model, X_np, batch_size=2048):
    """Run model inference in batches to avoid OOM from the [N, n_feat*d_token] flatten.

    Parameters
    ----------
    model : nn.Module  (on CPU after training)
    X_np  : np.ndarray  [N, n_features]  float32
    batch_size : int  rows per forward pass

    Returns
    -------
    preds : np.ndarray  [N, 3]  (q05, q50, q95) as float32
    """
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
            out = model(Xb)
            # Multi-head models (e.g. TabM with the auxiliary cloud head active)
            # return a dict; unwrap the always-present "quantiles" tensor so
            # evaluation code does not break when switching head configurations.
            preds = out["quantiles"] if isinstance(out, dict) else out
            parts.append(preds.numpy())
            del Xb, out
    return np.concatenate(parts, axis=0)



def evaluate_model_X_text(model, X_test, y_test, fig_dir):
    preds = _batched_predict(model, np.asarray(X_test, dtype=np.float32))
    q05, q50, q95 = preds[:, 0], preds[:, 1], preds[:, 2]
    uncertainty = q95 - q05
    residuals = y_test - q50

    q50_np = q50
    lower_err = np.clip(q50_np - q05, 0, None)
    upper_err = np.clip(q95 - q50_np, 0, None)
    
    # import metrics here to avoid circular import issues
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, q50_np)
    r2 = r2_score(y_test, q50_np)
    slope, intercept = np.polyfit(y_test, q50_np, 1)
    

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(y_test, q50_np,
                yerr=np.array([lower_err, upper_err]),
                fmt='o', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("True XCO2 bc anomaly")
    ax.set_ylabel("Predicted XCO2 bc anomaly")
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.05, 0.95, f"R²: {r2:.3f}\nSlope: {slope:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.savefig(os.path.join(fig_dir, "pred_vs_true.png"), dpi=150, bbox_inches='tight')
    


def plot_evaluation_by_regime(model, df, qt, features, output_dir,
                              target_col: str = 'xco2_bc_anomaly'):
    """3×4 evaluation plot matching mlp_lr_models.py lines 734-822.

    Row structure (mirrors mlp_lr_models.py lines 706-721):
        0 – Clear-sky FPs (cld_dist > 10 km), valid anomaly   [anomaly_label=True]
        1 – Cloud-affected FPs (cld_dist ≤ 10 km), valid anomaly [anomaly_label=True]
        2 – Cloud FPs with NaN anomaly (cld_dist ≤ 10 km)       [anomaly_label=False]

    Column structure (FT-Transformer adaptation):
        0 – Scatter: xco2_orig vs FT q50 direct prediction  (like LR scatter)
        1 – Scatter: same, colour-coded by uncertainty q95−q05  (FT adaptation of MLP scatter)
        2 – Distribution of anomaly values: [Original, FT q50]
        3 – Distribution of XCO2_bc values: [raw xco2_bc, xco2_bc−q50 corrected]
            (mirrors mlp col-3: raw vs bias-corrected xco2_bc)
    """
    # ── Feature matrix — avoid df.copy() + hstack loop to keep peak RAM low ─────
    qt_features = list(features[:-8])    # non-fp continuous features
    fp_cols     = list(features[-8:])    # fp_0 … fp_7

    # Ensure fp one-hot columns exist in df — build all missing cols at once to
    # avoid DataFrame fragmentation from repeated single-column assignments.
    missing_fp = [i for i in range(8) if f'fp_{i}' not in df.columns]
    if missing_fp:
        df = pd.concat(
            [df, pd.DataFrame(
                {f'fp_{i}': (df['fp'] == i).astype(np.float32) for i in missing_fp},
                index=df.index,
            )],
            axis=1,
        )

    X_qt  = qt.transform(df[qt_features].to_numpy(dtype=float)).astype(np.float32)
    X_fp  = df[fp_cols].to_numpy(dtype=np.float32)
    X_all = np.concatenate([X_qt, X_fp], axis=1)
    del X_qt, X_fp
    gc.collect()

    # ── Batched inference — avoids [N, n_feat*d_token] OOM tensor ─────────────
    preds = _batched_predict(model, X_all)
    del X_all
    gc.collect()
    q05 = preds[:, 0]
    q50 = preds[:, 1]
    q95 = preds[:, 2]
    uncertainty  = np.clip(q95 - q05, 0, None)
    true_anomaly = df[target_col].values.astype(float)
    xco2_bc      = df['xco2_bc'].values.astype(float)

    # ── Recompute anomaly from FT-corrected XCO2 (mirrors mlp_lr_models.py L641-643) ──
    from .pipeline import compute_xco2_anomaly_date_id
    _anomaly_args = {'lat_thres': 0.25, 'std_thres': 1.0, 'min_cld_dist': 10.0}
    xco2_bc_corrected_ft = xco2_bc - q50
    _req_cols = {'date', 'orbit_id', 'lat', 'cld_dist_km'}
    if _req_cols.issubset(df.columns):
        anomaly_ft = compute_xco2_anomaly_date_id(
            df['date'], df['orbit_id'], df['lat'].values,
            df['cld_dist_km'].values, xco2_bc_corrected_ft,
            **_anomaly_args,
        )
    else:
        anomaly_ft = None

    # ── Row masks (mirrors mlp_lr_models.py lines 706-711) ─────────────────────
    valid_anom = np.isfinite(true_anomaly)
    if 'cld_dist_km' in df.columns:
        cd = df['cld_dist_km'].values.astype(float)
        mask_r0 = valid_anom & (cd > 10)
        mask_r1 = valid_anom & (cd <= 10)
        mask_r2 = ~valid_anom & (cd <= 10)
    else:
        mask_r0 = valid_anom
        mask_r1 = np.zeros(len(df), dtype=bool)
        mask_r2 = ~valid_anom

    # row_configs tuple: (xco2_orig, xco2_orig_bc, xco2_ft, mask, row_label, anomaly_label)
    #   xco2_orig    – x-axis of scatter / anomaly distribution (anomaly or xco2_bc)
    #   xco2_orig_bc – raw xco2_bc reference used to build the col-3 corrected distribution
    #   xco2_ft      – FT direct prediction (q50)
    #   anomaly_label – True: use anomaly axis labels; False: use xco2_bc labels
    row_configs = [
        (true_anomaly, xco2_bc, q50, mask_r0,
         'Clear-sky FPs (cld_dist > 10 km)', True),
        (true_anomaly, xco2_bc, q50, mask_r1,
         'Cloud-affected FPs (cld_dist ≤ 10 km)', True),
        (xco2_bc,      xco2_bc, q50, mask_r2,
         'Cloud FPs with NaN anomaly (cld_dist ≤ 10 km)', False),
    ]

    # ── Figure ─────────────────────────────────────────────────────────────────
    plt.close('all')
    fig, axes = plt.subplots(3, 5, figsize=(27, 17))

    for row_i, (xco2_orig, xco2_orig_bc, xco2_ft, mask, row_label, anomaly_label) in enumerate(row_configs):
        ax_sc1, ax_sc2, ax_h, ax_h2, ax_h3 = axes[row_i]

        x_orig = xco2_orig[mask]
        x_bc   = xco2_orig_bc[mask]
        x_ft   = xco2_ft[mask]
        unc    = uncertainty[mask]

        _lo = np.nanpercentile(x_orig[np.isfinite(x_orig)], 1)  if np.isfinite(x_orig).any() else -3.0
        _hi = np.nanpercentile(x_orig[np.isfinite(x_orig)], 99) if np.isfinite(x_orig).any() else  3.0

        # ── Cols 0 & 1: Scatter (hidden for row 2 where x_orig == x_bc) ─────────
        if not (x_orig == x_bc).all():
            v = np.isfinite(x_orig) & np.isfinite(x_ft)

            # Col 0: plain scatter — like LR scatter in mlp
            ax_sc1.scatter(x_orig[v], x_ft[v], c='orange', edgecolor=None, s=5, alpha=0.6)
            ax_sc1.set_xlim(_lo, _hi); ax_sc1.set_ylim(_lo, _hi)
            ax_sc1.set_aspect('equal', adjustable='box')
            ax_sc1.axline((_lo, _lo), slope=1, color='r', linestyle='--')
            if anomaly_label:
                ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
                ax_sc1.set_ylabel('FT-corrected XCO2_bc anomaly (ppm)')
            else:
                ax_sc1.set_xlabel('Original XCO2_bc (ppm)')
                ax_sc1.set_ylabel('FT-corrected XCO2_bc (ppm)')
            ax_sc1.set_title(f'{row_label}\n[FT scatter]')
            if v.sum() > 1:
                r2 = 1 - np.nansum((x_orig[v] - x_ft[v])**2) / \
                         np.nansum((x_orig[v] - np.nanmean(x_orig[v]))**2)
                ax_sc1.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax_sc1.transAxes, va='top')

            # Col 1: scatter coloured by uncertainty — FT adaptation of MLP scatter
            unc_v    = unc[v]
            vmin_unc = max(np.nanpercentile(unc_v, 5), 1e-4)
            vmax_unc = np.nanpercentile(unc_v, 95)
            sc = ax_sc2.scatter(x_orig[v], x_ft[v], c=unc_v, cmap='plasma',
                                vmin=vmin_unc, vmax=vmax_unc,
                                edgecolor=None, s=5, alpha=0.6)
            ax_sc2.set_xlim(_lo, _hi); ax_sc2.set_ylim(_lo, _hi)
            ax_sc2.set_aspect('equal', adjustable='box')
            ax_sc2.axline((_lo, _lo), slope=1, color='r', linestyle='--')
            if anomaly_label:
                ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
                ax_sc2.set_ylabel('FT-corrected XCO2_bc anomaly (ppm)')
            else:
                ax_sc2.set_xlabel('Original XCO2_bc (ppm)')
                ax_sc2.set_ylabel('FT-corrected XCO2_bc (ppm)')
            ax_sc2.set_title(f'{row_label}\n[FT scatter, colour = q95−q05]')
            fig.colorbar(sc, ax=ax_sc2, label='q95−q05 (ppm)', pad=0.02)
            if v.sum() > 1:
                ax_sc2.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax_sc2.transAxes, va='top')
        else:
            ax_sc1.set_visible(False)
            ax_sc2.set_visible(False)

        # ── Cols 2 & 3: Distribution histograms (mirrors mlp_lr_models.py L776-813) ──
        # Items: (prediction_array, xco2_bc_ref, colour, label)
        #   xco2_bc_ref is used to: (a) detect if col-2 hist should show, and
        #                           (b) build col-3 corrected xco2_bc = xco2_bc_ref − pred
        items = [
            (x_orig, x_bc, 'blue',   'Original'),
            (x_ft,   x_bc, 'orange', 'FT-corrected'),
        ]

        for _xco2, _xco2_bc, _color, _label in items:
            _v  = _xco2[np.isfinite(_xco2)]
            _v2 = _xco2_bc[np.isfinite(_xco2)]
            if len(_v) == 0:
                continue

            # Col 2: anomaly distribution — skip if _v == _v2 (e.g. row-3 Original)
            if len(_v2) > 0 and not (_v == _v2).all():
                _mu, _sigma = _v.mean(), _v.std()
                _bins = np.linspace(np.nanpercentile(_v, 1), np.nanpercentile(_v, 99), 100)
                ax_h.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                          label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
                ax_h.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
                ax_h.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
                ax_h.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)

            # Col 3: xco2_bc distribution — raw for Original, corrected (xco2_bc − pred) for others
            if _label != 'Original':
                _v2 = _v2 - _v   # corrected xco2_bc = raw − predicted_anomaly
            _mu2, _sigma2 = _v2.mean(), _v2.std()
            _bins2 = np.linspace(np.nanpercentile(_v2, 1), np.nanpercentile(_v2, 99), 100)
            ax_h2.hist(_v2, bins=_bins2, color=_color, alpha=0.3, density=True,
                       label=f'{_label}\nμ={_mu2:.3f}, σ={_sigma2:.3f}')
            ax_h2.axvline(_mu2,           color=_color, linestyle='-',  linewidth=1.0)
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
        
        

    fig.suptitle('FT-Transformer XCO2_bc by cloud-distance regime', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ft_evaluation_by_regime.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved ft_evaluation_by_regime.png")

    # ── 1×3 recomputed-anomaly comparison (mirrors mlp_lr_models.py L646-701) ──
    if anomaly_ft is None:
        return

    anomaly_orig = true_anomaly   # full-dataset original anomaly

    plt.close('all')
    fig2, (ax_sc1, ax_sc2, ax_hist) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 0: scatter original vs FT-recomputed anomaly (like LR scatter)
    valid = np.isfinite(anomaly_orig) & np.isfinite(anomaly_ft)
    ax_sc1.scatter(anomaly_orig[valid], anomaly_ft[valid],
                   c='orange', edgecolor=None, s=5, alpha=0.6)
    _lim = np.nanpercentile(np.abs(anomaly_orig[valid]), 99)
    ax_sc1.set_xlim(-_lim, _lim); ax_sc1.set_ylim(-_lim, _lim)
    ax_sc1.set_aspect('equal', adjustable='box')
    ax_sc1.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc1.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc1.set_ylabel('FT-recomputed XCO2_bc anomaly (ppm)')
    ax_sc1.set_title('Original vs FT-recomputed anomaly')
    r2_ft = 1 - np.nansum((anomaly_orig[valid] - anomaly_ft[valid])**2) / \
                np.nansum((anomaly_orig[valid] - np.nanmean(anomaly_orig[valid]))**2)
    ax_sc1.text(0.05, 0.95, f'R²={r2_ft:.3f}', transform=ax_sc1.transAxes, va='top')

    # Panel 1: scatter coloured by uncertainty (FT adaptation of MLP scatter)
    unc_valid = uncertainty[valid]
    vmin_u = max(np.nanpercentile(unc_valid, 5), 1e-4)
    vmax_u = np.nanpercentile(unc_valid, 95)
    sc = ax_sc2.scatter(anomaly_orig[valid], anomaly_ft[valid],
                        c=unc_valid, cmap='plasma', vmin=vmin_u, vmax=vmax_u,
                        edgecolor=None, s=5, alpha=0.6)
    ax_sc2.set_xlim(-_lim, _lim); ax_sc2.set_ylim(-_lim, _lim)
    ax_sc2.set_aspect('equal', adjustable='box')
    ax_sc2.axline((0, 0), slope=1, color='r', linestyle='--')
    ax_sc2.set_xlabel('Original XCO2_bc anomaly (ppm)')
    ax_sc2.set_ylabel('FT-recomputed XCO2_bc anomaly (ppm)')
    ax_sc2.set_title('Original vs FT-recomputed anomaly\n(colour = q95−q05 uncertainty)')
    fig2.colorbar(sc, ax=ax_sc2, label='q95−q05 (ppm)', pad=0.02)
    ax_sc2.text(0.05, 0.95, f'R²={r2_ft:.3f}', transform=ax_sc2.transAxes, va='top')

    # Panel 2: distribution histogram (mirrors mlp L678-695)
    _bins = np.linspace(-3, 3, 211)
    for _anom, _color, _label in [
            (anomaly_orig, 'blue',   'Original'),
            (anomaly_ft,   'orange', 'FT-recomputed'),
    ]:
        _v = _anom[np.isfinite(_anom)]
        _mu, _sigma = np.nanmean(_v), np.nanstd(_v)
        ax_hist.hist(_v, bins=_bins, color=_color, alpha=0.6, density=True,
                     label=f'{_label}\nμ={_mu:.3f}, σ={_sigma:.3f}')
        ax_hist.axvline(_mu,          color=_color, linestyle='-',  linewidth=1.2)
        ax_hist.axvline(_mu - _sigma, color=_color, linestyle=':',  linewidth=0.9)
        ax_hist.axvline(_mu + _sigma, color=_color, linestyle=':',  linewidth=0.9)

    ax_hist.set_xlabel('XCO2_bc anomaly (ppm)')
    ax_hist.set_title('Anomaly distribution comparison')
    ax_hist.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax_hist.legend(fontsize=10)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'ft_recomputed_anomaly_comparison.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info("Saved ft_recomputed_anomaly_comparison.png")


# ─── Permutation importance ────────────────────────────────────────────────────
def plot_permutation_importance(model, X_test, y_test, features, output_dir,
                                n_repeats=5, subsample=3000, batch_size=1024,
                                output_prefix: str = "ft"):
    """Permutation importance for a quantile model (q50 head).

    For each feature in turn, shuffles that column of X_sub **in-place**
    (restoring it afterward) so no full-array copy is needed per iteration —
    only a single column (shape [n]) is duplicated.  `batch_size` is kept
    small to bound the peak attention-tensor memory [batch, n_feat, n_feat].

    Parameters
    ----------
    output_prefix : str
        Filename / title prefix.  Use "ft" for the FT-Transformer (default)
        and "tabm" for TabM so each model writes its own files instead of
        overwriting a shared ``ft_`` artifact.

    Outputs
    -------
    {output_prefix}_permutation_importance.csv – feature, mean_importance, std_importance
    {output_prefix}_permutation_importance.png – horizontal bar chart (all features)
    """
    model.eval()
    features = list(features)
    rng = np.random.default_rng(42)

    # ── Sub-sample ──────────────────────────────────────────────────────────────
    n = min(subsample, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    # C-contiguous float32 so torch.tensor() gets a zero-copy view later
    X_sub = np.ascontiguousarray(X_test[idx], dtype=np.float32)
    y_sub = y_test[idx].astype(np.float32)

    def _predict_q50(X_np):
        """Batched inference → q50 predictions (numpy).  Small batch_size
        keeps intermediate attention tensors [batch, n_feat, n_feat] small."""
        out = []
        for start in range(0, len(X_np), batch_size):
            Xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
            with torch.no_grad():
                preds = model(Xb)          # [batch, 3]  (or dict for multi-head models)
            if isinstance(preds, dict):
                preds = preds["quantiles"]
            out.append(preds[:, 1].numpy())  # q50
            del Xb, preds
        return np.concatenate(out)

    # ── Baseline R² ─────────────────────────────────────────────────────────────
    y_base      = _predict_q50(X_sub)
    ss_tot      = float(((y_sub - y_sub.mean()) ** 2).sum())
    baseline_r2 = 1.0 - float(((y_sub - y_base) ** 2).sum()) / ss_tot
    logger.info("Permutation importance baseline R²: %.4f", baseline_r2)

    # ── Per-feature importance — in-place column swap ───────────────────────────
    # Exclude footprint one-hot columns (fp_0 … fp_7) from both calculation and plot.
    non_fp = [(col, fname) for col, fname in enumerate(features)
              if not (fname.startswith('fp_') and fname[3:].isdigit())]

    importances = np.zeros((len(non_fp), n_repeats))
    rng_inner   = np.random.default_rng(0)

    for i, (col, fname) in enumerate(tqdm(non_fp, desc="Permutation importance", unit="feat")):
        orig_col = X_sub[:, col].copy()      # save one column only (n floats)
        for r in range(n_repeats):
            X_sub[:, col] = rng_inner.permutation(orig_col)   # shuffle in-place
            y_shuf        = _predict_q50(X_sub)
            r2_shuf       = 1.0 - float(((y_sub - y_shuf) ** 2).sum()) / ss_tot
            importances[i, r] = baseline_r2 - r2_shuf
            del y_shuf
        X_sub[:, col] = orig_col             # restore column
        del orig_col
        gc.collect()

    non_fp_names = [fname for _, fname in non_fp]
    mean_imp = importances.mean(axis=1)
    std_imp  = importances.std(axis=1)

    # ── Save CSV ─────────────────────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        'feature':          non_fp_names,
        'mean_importance':  mean_imp,
        'std_importance':   std_imp,
    }).sort_values('mean_importance', ascending=False)
    csv_path = os.path.join(output_dir, f'{output_prefix}_permutation_importance.csv')
    imp_df.to_csv(csv_path, index=False)
    logger.info("Saved %s_permutation_importance.csv", output_prefix)

    # ── Bar chart ────────────────────────────────────────────────────────────────
    n_feat   = len(non_fp_names)
    fig_h    = max(6, n_feat * 0.28)
    fig, ax  = plt.subplots(figsize=(8, fig_h))

    sorted_names = imp_df['feature'].tolist()
    sorted_mean  = imp_df['mean_importance'].tolist()
    sorted_std   = imp_df['std_importance'].tolist()

    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_feat))
    ax.barh(range(n_feat), sorted_mean[::-1],
            xerr=sorted_std[::-1], error_kw=dict(ecolor='gray', capsize=3),
            color=bar_colors)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(sorted_names[::-1], fontsize=7)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean R² drop when feature is permuted')
    ax.set_title(
        f'{output_prefix.upper()} Permutation Importance ({n_feat} features)\n'
        f'(baseline R²={baseline_r2:.3f}, {n_repeats} repeats, n={n} samples)'
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{output_prefix}_permutation_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved %s_permutation_importance.png", output_prefix)
