"""Apply a trained deep-ensemble model to UNSEEN data, with calibrated intervals.

Loads a deep-ensemble fold directory (member_*.pt + deep_ensemble_pipeline.pkl +
deep_ensemble_meta.pkl), predicts the XCO2 anomaly mu and uncertainty sigma on new
parquet(s), forms conformal prediction intervals, and writes the cloud-aware
correction.

Two calibration modes:
  • frozen (default): reuse the conformal quantiles stored at train time
    (q_split / q_by_bin + mondrian_edges in meta) — no calibration data needed.
  • recalibrate (--calib PARQUET, must contain xco2_bc_anomaly): recompute the
    conformal quantiles on a held-out calibration set; required for
    --near_cloud_target (regime-elevated near-cloud coverage).

If the unseen parquet itself carries xco2_bc_anomaly (truth), metrics + correction
effectiveness by cloud distance are reported; otherwise only predictions are written.

Example:
    PYTHONPATH=src python -m apply.apply_deep_ensemble \\
        --model-dir results/model_deep_ensemble/de_ocean_beta_nll_f0 --sfc_type 0 \\
        --calib     results/csv_collection/combined_2020-02-11_all_orbits.parquet \\
        --near_cloud_target 0.98 \\
        --input-dir results/csv_collection \\
        --input     combined_2020-09-06_all_orbits.parquet \\
        --output    results/model_comparison/de_apply_corrected.csv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.pipeline import FeaturePipeline, _ensure_derived_features
from models.deep_ensemble import GaussianMLP, ensemble_predict, Z90
from models import conformal as cf
from models import diagnostics as diag


def _load_model(model_dir: Path):
    meta = pickle.load(open(model_dir / 'deep_ensemble_meta.pkl', 'rb'))
    pipeline = FeaturePipeline.load(model_dir / 'deep_ensemble_pipeline.pkl')
    members = []
    for p in sorted(model_dir.glob('member_*.pt')):
        m = GaussianMLP(pipeline.n_features)
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.eval()
        members.append(m)
    if not members:
        raise SystemExit(f"no member_*.pt found in {model_dir}")
    print(f"  [loaded] {len(members)} members, {pipeline.n_features} features, "
          f"loss={meta.get('loss','gaussian_nll')} ← {model_dir}")
    return pipeline, members, meta


def _read(paths) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _predict(df, pipeline, members, meta, sfc_type):
    """Return (mu, sigma, kept_df) for the rows that pass surface filter + finite features."""
    df = df[df['sfc_type'] == sfc_type].copy()
    df = _ensure_derived_features(df)
    X = pipeline.transform(df)
    valid = np.all(np.isfinite(X), axis=1)
    df = df.loc[valid].reset_index(drop=True)
    mu, sigma = ensemble_predict(members, X[valid], torch.device('cpu'),
                                 loss=meta.get('loss', 'gaussian_nll'), nu=meta.get('nu', 4.0))
    return mu, sigma, df


def main():
    ap = argparse.ArgumentParser(description="Apply a trained deep ensemble to unseen data.")
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--sfc_type', type=int, required=True,
                    help="Surface the model was trained on (0=ocean, 1=land).")
    ap.add_argument('--input-dir', default='')
    ap.add_argument('--input', nargs='+', required=True, help="Unseen parquet file(s).")
    ap.add_argument('--calib', default=None,
                    help="Calibration parquet (needs xco2_bc_anomaly) to RECOMPUTE "
                         "conformal; required for --near_cloud_target.")
    ap.add_argument('--near_cloud_target', type=float, default=None)
    ap.add_argument('--near_cloud_km', type=float, default=10.0)
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    pipeline, members, meta = _load_model(model_dir)
    mcol = meta.get('mondrian_col', 'cld_dist_km')
    idir = Path(args.input_dir) if args.input_dir else Path('.')

    # ── predictions on unseen data ─────────────────────────────────────────────
    inp = _read([idir / f for f in args.input])
    mu, sigma, df = _predict(inp, pipeline, members, meta, args.sfc_type)
    print(f"  unseen: {len(df):,} soundings (sfc={args.sfc_type})")

    # ── conformal intervals ────────────────────────────────────────────────────
    if mcol not in df.columns:
        raise SystemExit(f"unseen data lacks Mondrian column {mcol!r}")
    te_val = df[mcol].to_numpy(dtype=float)
    te_val = np.where(np.isfinite(te_val), te_val, np.nanmedian(te_val[np.isfinite(te_val)]))

    if args.calib:
        cal_df = pd.read_parquet(Path(args.calib) if Path(args.calib).is_absolute()
                                 else idir / args.calib)
        cmu, csig, cal = _predict(cal_df, pipeline, members, meta, args.sfc_type)
        cy = cal['xco2_bc_anomaly'].to_numpy(dtype=float)
        cval = cal[mcol].to_numpy(dtype=float)
        cval = np.where(np.isfinite(cval), cval, np.nanmedian(cval[np.isfinite(cval)]))
        cal_bin, edges = cf.make_quantile_bins(cval, meta.get('mondrian_bins', 10))
        te_bin, _ = cf.make_quantile_bins(te_val, meta.get('mondrian_bins', 10), edges=edges)
        bin_alpha = None
        if args.near_cloud_target is not None:
            is_near = cal['cld_dist_km'].to_numpy(dtype=float) <= args.near_cloud_km
            bin_alpha = cf.regime_alphas(cal_bin, is_near,
                                         near_alpha=1.0 - args.near_cloud_target,
                                         far_alpha=args.alpha)
            print(f"  recalibrated; near-cloud target {args.near_cloud_target}: "
                  f"{sum(a < args.alpha for a in bin_alpha.values())} bins elevated")
        preds, _ = cf.mondrian_conformal(cy, cmu, csig, cal_bin, mu, sigma, te_bin,
                                         alpha=args.alpha, bin_alpha=bin_alpha)
    else:
        if args.near_cloud_target is not None:
            raise SystemExit("--near_cloud_target requires --calib (frozen quantiles are flat).")
        edges = np.asarray(meta['mondrian_edges'], dtype=float)
        q_by_bin = {int(k): float(v) for k, v in meta['q_by_bin'].items()}
        te_bin = np.clip(np.digitize(te_val, edges[1:-1]), 0, len(edges) - 2)
        qg = float(np.median(list(q_by_bin.values())))
        q = np.array([q_by_bin.get(int(b), qg) for b in te_bin])
        preds = np.column_stack([mu - q * sigma, mu, mu + q * sigma])
        print("  frozen calibration (stored q_by_bin, flat target)")

    lo, hi = preds[:, 0], preds[:, 2]

    # ── corrected XCO2 (subtract the predicted anomaly) ────────────────────────
    out = pd.DataFrame({'pred_anomaly': mu, 'sigma': sigma,
                        'anomaly_lo': lo, 'anomaly_hi': hi})
    for c in ('sounding_id', 'lon', 'lat', 'cld_dist_km', 'sfc_type', 'xco2_bc',
              'xco2_bc_anomaly'):
        if c in df.columns:
            out[c] = df[c].to_numpy()
    if 'xco2_bc' in df.columns:
        xb = df['xco2_bc'].to_numpy(dtype=float)
        out['xco2_corrected'] = xb - mu
        out['xco2_corrected_lo'] = xb - hi
        out['xco2_corrected_hi'] = xb - lo

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"  [saved] {len(out):,} rows → {args.output}")

    # ── evaluation if truth is present in the unseen data ──────────────────────
    if 'xco2_bc_anomaly' in df.columns:
        y = df['xco2_bc_anomaly'].to_numpy(dtype=float)
        m = np.isfinite(y)
        g = diag.compute_metrics(y[m], preds[m])
        cov_near = np.nan
        if 'cld_dist_km' in df.columns:
            near = (df['cld_dist_km'].to_numpy(dtype=float) <= 10.0) & m
            cov_near = ((y[near] >= lo[near]) & (y[near] <= hi[near])).mean()
        print(f"\n  UNSEEN metrics: R²={g['r2']:.3f}  RMSE={g['rmse']:.3f}  "
              f"cov90={g['coverage_90']:.3f}  near-cloud cov={cov_near:.3f}")
        corr = diag.correction_by_cloud_distance(df.loc[m], y[m], mu[m])
        if not corr.empty:
            print("  correction RMS reduction by cloud distance:")
            print(corr[['bin', 'n', 'pre_rms', 'post_rms', 'rms_reduction_pct', 'r2']]
                  .to_string(index=False))


if __name__ == '__main__':
    main()
