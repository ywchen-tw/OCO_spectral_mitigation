"""Uncertainty / calibration diagnostics shared by all quantile models.

Reference: no external reference (standard quantile-calibration diagnostics).

Implements the diagnostic suite required by TABM_PLAN.md ("Uncertainty
diagnostics" + "Calibration success criteria") so TabM, the GBDT baselines,
and the MLP baseline are all scored identically:

  - point metrics            : RMSE / MAE / R²
  - pinball loss             : per quantile (q05, q50, q95), reported separately
  - 90% empirical coverage   : fraction of y in [q05, q95] (target ≈ 0.90)
  - mean interval width      : E[q95 − q05]
  - quantile crossing rate   : fraction with q05 ≥ q95 (0.0 with monotonic head)
  - member spread            : std of K member q50s (TabM only; epistemic proxy)
  - stratified metrics       : by cloud proximity / AOD / glint / footprint /
                               surface type, plus left-tail (bottom 5% / 10%)

All predictions are passed as ``preds`` of shape [N, 3] = (q05, q50, q95).

GBDT models train one model per quantile independently, so their outputs may
cross (q05 ≥ q95).  ``monotone_rearrange`` sorts the three columns per sample;
callers should report metrics both before and after rearrangement so the GBDT
is not penalised for crossing where the monotonic-head neural model has a
structural guarantee.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUANTILES = (0.05, 0.5, 0.95)

# Calibration pass/fail thresholds (chosen before experiments — see
# TABM_PLAN.md "Calibration success criteria").
COVERAGE_GLOBAL_RANGE = (0.87, 0.93)
COVERAGE_REGIME_MIN = 0.85          # near-cloud / high-AOD
COVERAGE_LEFT_TAIL_MIN = 0.80       # bottom 10%
DATEBLOCK_RMSE_RATIO_FLAG = 1.20


# ── primitives ──────────────────────────────────────────────────────────────

def monotone_rearrange(preds: np.ndarray) -> np.ndarray:
    """Sort the three quantile columns per sample so q05 ≤ q50 ≤ q95.

    preds : [N, 3] for (q05, q50, q95).  Returns a new array; input untouched.
    """
    preds = np.asarray(preds, dtype=float)
    if preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(f"preds must be [N, 3], got {preds.shape}")
    return np.sort(preds, axis=1)


def pinball_loss(y: np.ndarray, pred_q: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss for a single quantile level q."""
    err = y - pred_q
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def crossing_rate(preds: np.ndarray) -> float:
    """Fraction of samples with q05 ≥ q95 (should be 0.0 for a monotonic head)."""
    preds = np.asarray(preds, dtype=float)
    return float(np.mean(preds[:, 0] >= preds[:, 2]))


def coverage_90(y: np.ndarray, preds: np.ndarray) -> float:
    """Empirical fraction of y inside [q05, q95]."""
    preds = np.asarray(preds, dtype=float)
    return float(np.mean((y >= preds[:, 0]) & (y <= preds[:, 2])))


def mean_interval_width(preds: np.ndarray) -> float:
    preds = np.asarray(preds, dtype=float)
    return float(np.mean(preds[:, 2] - preds[:, 0]))


# ── aggregate metric dict ────────────────────────────────────────────────────

def compute_metrics(y: np.ndarray,
                    preds: np.ndarray,
                    quantiles=QUANTILES,
                    members: 'np.ndarray | None' = None) -> dict:
    """Full metric dict for one (y, preds) pair.

    Parameters
    ----------
    y : [N] ground-truth anomaly.
    preds : [N, 3] (q05, q50, q95).
    members : optional [N, K, 3] member-level outputs (TabM `return_members`).
        When given, adds 'member_spread_mean' = mean over samples of std(K q50s),
        a proxy (not a calibrated estimate) for epistemic uncertainty.
    """
    y = np.asarray(y, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if preds.shape[0] != y.shape[0]:
        raise ValueError(f"y ({y.shape[0]}) and preds ({preds.shape[0]}) length mismatch")

    q50 = preds[:, 1]
    resid = y - q50
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float('nan')

    metrics = {
        'n': int(y.shape[0]),
        'rmse': float(np.sqrt(np.mean(resid ** 2))),
        'mae': float(np.mean(np.abs(resid))),
        'r2': r2,
        'coverage_90': coverage_90(y, preds),
        'mean_interval_width': mean_interval_width(preds),
        'crossing_rate': crossing_rate(preds),
    }
    for i, q in enumerate(quantiles):
        metrics[f'pinball_q{int(round(q * 100)):02d}'] = pinball_loss(y, preds[:, i], q)

    if members is not None:
        members = np.asarray(members, dtype=float)
        # std across the K members of each sample's q50, then averaged over samples
        spread = members[:, :, 1].std(axis=1)
        metrics['member_spread_mean'] = float(spread.mean())
        metrics['member_crossing_rate'] = float(
            np.mean(members[:, :, 0] >= members[:, :, 2])
        )
    return metrics


# ── stratified metrics ───────────────────────────────────────────────────────

def _aod_bins(aod: np.ndarray) -> pd.Series:
    # Quartile bins; robust to the heavy right tail (log1p not needed for binning).
    try:
        return pd.qcut(aod, q=4, labels=[f'aod_q{i+1}' for i in range(4)],
                       duplicates='drop')
    except (ValueError, IndexError):
        return pd.Series(['aod_all'] * len(aod))


def _build_regimes(meta: pd.DataFrame, y: np.ndarray) -> dict:
    """Return {regime_name: {group_label: boolean mask}} from available columns.

    Silently skips any regime whose source column is absent so the same code
    works across surface types and CSV versions.
    """
    n = len(y)
    regimes: dict = {}

    if 'cld_dist_km' in meta.columns:
        cd = meta['cld_dist_km'].to_numpy(dtype=float)
        regimes['cloud_proximity'] = {
            'near_cloud(<=10km)': cd <= 10.0,
            'far_cloud(>10km)': cd > 10.0,
        }

    aod_col = 'aod_total' if 'aod_total' in meta.columns else None
    if aod_col is None:
        # Sum available components as a proxy when aod_total is absent.
        comp = [c for c in ('aod_dust', 'aod_oc', 'aod_seasalt',
                            'aod_strataer', 'aod_sulfate') if c in meta.columns]
        aod = meta[comp].to_numpy(dtype=float).sum(axis=1) if comp else None
    else:
        aod = meta[aod_col].to_numpy(dtype=float)
    if aod is not None:
        bins = _build_categorical(_aod_bins(aod))
        regimes['aod_load'] = bins

    if 'cos_glint_angle' in meta.columns:
        ga = meta['cos_glint_angle'].to_numpy(dtype=float)
        med = np.nanmedian(ga)
        regimes['glint_angle'] = {
            f'cos_glint<=med({med:.3f})': ga <= med,
            'cos_glint>med': ga > med,
        }

    if 'fp' in meta.columns:
        fp = meta['fp'].to_numpy()
        regimes['footprint'] = {f'fp_{i}': fp == i for i in range(8)}

    if 'sfc_type' in meta.columns:
        st = meta['sfc_type'].to_numpy()
        regimes['surface_type'] = {
            'ocean(sfc0)': st == 0,
            'land(sfc1)': st == 1,
        }

    # Left tail of the target — where cloud-contaminated failures concentrate.
    order = np.argsort(y)
    tail = {}
    for frac in (0.05, 0.10):
        k = max(1, int(np.floor(frac * n)))
        m = np.zeros(n, dtype=bool)
        m[order[:k]] = True
        tail[f'bottom_{int(frac*100)}pct'] = m
    regimes['left_tail'] = tail

    # Crossed regime: the contaminated tail *within* near-cloud.  This is the
    # deployment-relevant metric when corrections are applied near-cloud only and
    # the large anomalies (which concentrate near-cloud) are what we care about.
    if 'cld_dist_km' in meta.columns:
        near = meta['cld_dist_km'].to_numpy(dtype=float) <= 10.0
        crossed = {}
        for frac in (0.05, 0.10):
            k = max(1, int(np.floor(frac * n)))
            tail_mask = np.zeros(n, dtype=bool)
            tail_mask[order[:k]] = True
            crossed[f'near&bottom_{int(frac*100)}pct'] = near & tail_mask
        crossed['near&bulk(rest)'] = near & ~tail['bottom_10pct']
        regimes['near_cloud_tail'] = crossed

    return regimes


def _build_categorical(cat: pd.Series) -> dict:
    """Convert a categorical/qcut Series into {label: mask}."""
    arr = np.asarray(cat)
    out = {}
    for label in pd.unique(arr):
        if pd.isna(label):
            continue
        out[str(label)] = (arr == label)
    return out


def stratified_metrics(meta: pd.DataFrame,
                       y: np.ndarray,
                       preds: np.ndarray,
                       quantiles=QUANTILES) -> pd.DataFrame:
    """Per-regime metric table.

    meta : DataFrame of metadata aligned row-for-row with y / preds (the valid,
           finite held-out rows).  Regimes are built from whichever of
           cld_dist_km / aod_* / cos_glint_angle / fp / sfc_type are present,
           plus the target's left tail (bottom 5% / 10%).
    Returns a long-format DataFrame: regime, group, n, rmse, mae, r2,
    coverage_90, mean_interval_width, crossing_rate, pinball_q05/50/95.
    """
    y = np.asarray(y, dtype=float)
    preds = np.asarray(preds, dtype=float)
    regimes = _build_regimes(meta, y)
    rows = []
    for regime, groups in regimes.items():
        for label, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            m = compute_metrics(y[mask], preds[mask], quantiles=quantiles)
            m.update({'regime': regime, 'group': label})
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    cols = ['regime', 'group', 'n', 'rmse', 'mae', 'r2', 'coverage_90',
            'mean_interval_width', 'crossing_rate',
            'pinball_q05', 'pinball_q50', 'pinball_q95']
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def correction_by_cloud_distance(meta: pd.DataFrame,
                                 y: np.ndarray,
                                 point_pred: np.ndarray,
                                 edges=(0.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf),
                                 col: str = 'cld_dist_km') -> pd.DataFrame:
    """Correction effectiveness as a function of cloud distance.

    For each cloud-distance bin compares the anomaly magnitude *before* the
    correction (|y|, since the uncorrected residual from truth IS the anomaly)
    to the magnitude *after* subtracting the model's point prediction
    (|y - yhat|).  Answers: 'how much does applying the predicted correction
    shrink the anomaly in the near-cloud footprints, and where does it stop
    helping?'

    point_pred may be a 1-D point array or the (lo, mid, hi) interval matrix
    (the middle column is taken as the point estimate).

    Returns long-format: bin, lo_km, hi_km, n, pre_rms, post_rms,
    rms_reduction_pct, pre_bias, post_bias, pre_std, post_std, r2.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(point_pred, dtype=float)
    if yhat.ndim > 1:
        yhat = yhat[:, yhat.shape[1] // 2]
    if col not in meta.columns:
        return pd.DataFrame()
    cd = meta[col].to_numpy(dtype=float)
    resid = y - yhat
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cd >= lo) & (cd < hi) & np.isfinite(cd)
        if m.sum() == 0:
            continue
        yy, rr = y[m], resid[m]
        pre_rms = float(np.sqrt(np.mean(yy ** 2)))
        post_rms = float(np.sqrt(np.mean(rr ** 2)))
        ss_res = float(np.sum(rr ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        rows.append({
            'bin': f'[{lo:g},{hi:g})km', 'lo_km': lo, 'hi_km': hi, 'n': int(m.sum()),
            'pre_rms': pre_rms, 'post_rms': post_rms,
            'rms_reduction_pct': 100.0 * (1.0 - post_rms / pre_rms) if pre_rms > 0 else float('nan'),
            'pre_bias': float(yy.mean()), 'post_bias': float(rr.mean()),
            'pre_std': float(yy.std()), 'post_std': float(rr.std()),
            'r2': (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        })
    return pd.DataFrame(rows)


# ── calibration pass/fail ─────────────────────────────────────────────────────

def calibration_report(global_metrics: dict,
                       strat: pd.DataFrame,
                       dateblock_rmse_ratio: 'float | None' = None) -> dict:
    """Evaluate the plan's pass/fail criteria.  Returns a dict of bool/None checks.

    A None value means the criterion could not be evaluated (regime absent).
    """
    report: dict = {}

    cov = global_metrics.get('coverage_90')
    report['coverage_global_in_range'] = (
        COVERAGE_GLOBAL_RANGE[0] <= cov <= COVERAGE_GLOBAL_RANGE[1]
        if cov is not None else None
    )
    report['crossing_rate_zero'] = (
        global_metrics.get('crossing_rate', 1.0) == 0.0
    )

    def _regime_cov(regime, group):
        if strat is None or strat.empty:
            return None
        sel = strat[(strat['regime'] == regime) & (strat['group'] == group)]
        return float(sel['coverage_90'].iloc[0]) if len(sel) else None

    near = _regime_cov('cloud_proximity', 'near_cloud(<=10km)')
    report['coverage_near_cloud_ok'] = (near >= COVERAGE_REGIME_MIN) if near is not None else None

    # high-AOD = top AOD quartile bin (aod_q4 if present)
    aod_hi = _regime_cov('aod_load', 'aod_q4')
    report['coverage_high_aod_ok'] = (aod_hi >= COVERAGE_REGIME_MIN) if aod_hi is not None else None

    tail = _regime_cov('left_tail', 'bottom_10pct')
    report['coverage_left_tail_ok'] = (tail >= COVERAGE_LEFT_TAIL_MIN) if tail is not None else None

    if dateblock_rmse_ratio is not None:
        report['dateblock_rmse_ratio'] = float(dateblock_rmse_ratio)
        report['dateblock_rmse_ratio_flag'] = dateblock_rmse_ratio > DATEBLOCK_RMSE_RATIO_FLAG

    return report


# ── persistence ───────────────────────────────────────────────────────────────

def save_diagnostics(output_dir,
                     prefix: str,
                     global_metrics: dict,
                     strat: 'pd.DataFrame | None' = None,
                     calibration: 'dict | None' = None,
                     extra: 'dict | None' = None) -> None:
    """Write {prefix}_metrics.json and {prefix}_stratified_metrics.csv.

    prefix typically encodes model + split + (rearranged) variant, e.g.
    'tabm_date' or 'xgboost_random_rearranged'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {'global': global_metrics}
    if calibration is not None:
        payload['calibration'] = calibration
    if extra is not None:
        payload['extra'] = extra
    json_path = out / f'{prefix}_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Saved %s", json_path)
    if strat is not None and not strat.empty:
        csv_path = out / f'{prefix}_stratified_metrics.csv'
        strat.to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path)
