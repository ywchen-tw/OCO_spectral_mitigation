"""build_structured_residual_plot_data.py — structured-residual plot-data bridge.

Applies the per-surface structured-residual ensemble produced by
``models.structured_dcn_ensemble`` to unseen per-date parquet files, pooling all
date-kfold folds in the same style as ``build_deepens_plot_data.py``.  The output
is a ``plot_data.parquet`` carrying the columns expected by
``plot_corrected_xco2.py`` and the TCCON aggregate reports:

    time, lon, lat, cld_dist_km, sfc_type, xco2_bc, xco2_bc_anomaly,
    pred_anomaly, sigma, structured_residual_corrected_xco2

The correction is per-surface: ocean footprints are scored only by ocean folds and
land footprints only by land folds.  Each fold uses its own saved
``FeaturePipeline``/profile-PCA/scaler, preserving the leakage-safe fold-PCA
discipline used during training.

Example:
    PYTHONPATH=src python workspace/build_structured_residual_plot_data.py \
        --ocean-model-dir results/model_structured_dcn_ensemble/de2016_2020_structured_shared_h64x32_b8_foldpca_ocean_f* \
        --land-model-dir  results/model_structured_dcn_ensemble/de2016_2020_structured_shared_h64x32_b8_foldpca_land_f* \
        --input results/csv_collection/combined_2018-10-24_all_orbits.parquet \
        --output .../structured_residual/plot_data.parquet
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from apply.apply_deep_ensemble import _check_required_columns, _domain_report
from models.deep_ensemble import _member_predict
from models.pipeline import FeaturePipeline, _ensure_derived_features
from models.leakage_guard import check_training_overlap
from models.structured_dcn_ensemble import build_experimental_member

CORR_COL = "structured_residual_corrected_xco2"
KEEP_COLS = (
    "time",
    "lon",
    "lat",
    "cld_dist_km",
    "sfc_type",
    "fp",
    "xco2_bc",
    "xco2_bc_anomaly",
    "xco2_apriori",
    "xco2_raw",
    "xco2_raw_anomaly",
    "xco2_uncertainty",
    "aod_total",
    "sza",
    "snr_o2a",
    "snr_wco2",
    "snr_sco2",
)


def _load_fold(model_dir):
    """Load one structured-residual fold: (pipeline, members, metadata)."""
    md = Path(model_dir)
    with open(md / "experimental_meta.pkl", "rb") as stream:
        meta = pickle.load(stream)
    pipe = FeaturePipeline.load(md / "experimental_pipeline.pkl")

    backbone = meta.get("backbone", "structured_residual")
    feature_names = list(meta.get("feature_names") or pipe.feature_names)
    hidden_dims = tuple(meta.get("hidden_dims", (64, 32)))
    dropout = float(meta.get("dropout", 0.0))
    norm = meta.get("norm", "none")
    block_dim = int(meta.get("block_dim", 16))
    n_experts = int(meta.get("n_experts", 4))
    cross_layers = int(meta.get("cross_layers", 2))
    cross_rank = int(meta.get("cross_rank", 16))

    members = []
    for checkpoint in sorted(md.glob("member_*.pt")):
        model = build_experimental_member(
            backbone,
            pipe.n_features,
            feature_names=feature_names,
            hidden_dims=hidden_dims,
            dropout=dropout,
            norm=norm,
            block_dim=block_dim,
            n_experts=n_experts,
            cross_layers=cross_layers,
            cross_rank=cross_rank,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        members.append(model)
    if not members:
        raise SystemExit(f"no member_*.pt in {md}")
    return pipe, members, meta


def _predict_pooled(
    df,
    fold_dirs,
    sfc_type,
    *,
    ood_thresh=8.0,
    max_ood_frac=0.02,
    strict=False,
    tag="input",
):
    """Pool every member from every supplied fold into one mixture ensemble."""
    folds = [_load_fold(d) for d in fold_dirs]
    pipe0, _, meta0 = folds[0]
    loss = meta0.get("loss", "beta_nll")
    nu = float(meta0.get("nu", 4.0))
    n_mem = sum(len(members) for _, members, _ in folds)
    print(
        f"  [loaded] {len(folds)} fold(s) × members = {n_mem} total, "
        f"{pipe0.n_features} features, loss={loss}"
    )

    _check_required_columns(df, pipe0)
    df = df[df["sfc_type"] == sfc_type].copy()
    df = _ensure_derived_features(df)
    X0 = pipe0.transform(df)
    valid = np.all(np.isfinite(X0), axis=1)
    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        return None, None, df

    overall, worst = _domain_report(X0[valid], pipe0, thresh=ood_thresh)
    if overall > max_ood_frac:
        flagged = [f"{name}({frac:.0%})" for name, frac in worst[:5] if frac > max_ood_frac]
        msg = (
            f"  ⚠ DOMAIN WARNING [{tag}]: {overall:.1%} of feature values OOD "
            f"(|z|>{ood_thresh:g}); worst: {flagged} → predictions UNRELIABLE"
        )
        if strict:
            raise SystemExit(msg + "\n  (refused: --strict)")
        print(msg)
    else:
        print(f"  domain check [{tag}]: OK ({overall:.2%} OOD)")

    device = torch.device("cpu")
    sum_mu = np.zeros(len(df), dtype=np.float64)
    sum_m2 = np.zeros(len(df), dtype=np.float64)
    count = 0
    for pipe, members, meta in folds:
        Xi = pipe.transform(df)
        fold_loss = meta.get("loss", loss)
        fold_nu = float(meta.get("nu", nu))
        for member in members:
            mu_i, var_i = _member_predict(member, Xi, device, loss=fold_loss, nu=fold_nu)
            mu64 = mu_i.astype(np.float64)
            sum_mu += mu64
            sum_m2 += var_i.astype(np.float64) + mu64**2
            count += 1

    mu = sum_mu / count
    var = np.maximum(sum_m2 / count - mu**2, 1e-12)
    sigma = np.sqrt(var)
    return mu.astype(np.float32), sigma.astype(np.float32), df


def _build_surface(
    df,
    fold_dirs,
    sfc_type,
    *,
    dk,
    clim_max_ppm=50.0,
    max_abs_anomaly=25.0,
    base_col="xco2_bc",
    truth_col="xco2_bc_anomaly",
):
    mu, sigma, kept = _predict_pooled(df, fold_dirs, sfc_type, tag=f"sfc{sfc_type}", **dk)
    if mu is None or len(kept) == 0:
        print(f"  sfc={sfc_type}: no rows after filter — skipped")
        return None

    out = kept[[c for c in KEEP_COLS if c in kept.columns]].reset_index(drop=True).copy()
    mu = np.asarray(mu, dtype=float)

    clim_guard = np.zeros(len(out), dtype=bool)
    if "xco2_apriori" in out.columns:
        diff = out[base_col].to_numpy(float) - out["xco2_apriori"].to_numpy(float)
        clim_guard = np.isfinite(diff) & (diff > clim_max_ppm)
    else:
        print(f"  sfc={sfc_type}: no xco2_apriori column — climatology guard disabled")

    anomaly_guard = np.zeros(len(out), dtype=bool)
    if max_abs_anomaly is not None:
        anomaly_guard = np.isfinite(mu) & (np.abs(mu) > max_abs_anomaly)

    guard = clim_guard | anomaly_guard
    if guard.any():
        mu = mu.copy()
        mu[guard] = 0.0
        print(
            f"  sfc={sfc_type}: guards skipped correction on {int(guard.sum())} "
            f"sounding(s)  [clim>{clim_max_ppm:g}ppm: {int(clim_guard.sum())}, "
            f"|anomaly|>{max_abs_anomaly}: {int(anomaly_guard.sum())}]"
        )

    out["clim_guard"] = clim_guard
    out["anomaly_guard"] = anomaly_guard
    out["pred_anomaly"] = mu
    out["sigma"] = sigma
    out["structured_residual_sigma"] = sigma
    out[CORR_COL] = out[base_col].to_numpy(dtype=float) - mu

    if truth_col in out.columns:
        y = out[truth_col].to_numpy(float)
        keep = ~guard
        pre = np.sqrt(np.nanmean(y[keep] ** 2))
        post = np.sqrt(np.nanmean((y[keep] - mu[keep]) ** 2))
        print(
            f"  sfc={sfc_type}: {int(keep.sum()):,}/{len(out):,} corrected  "
            f"{truth_col} RMS {pre:.3f} → {post:.3f} ppm ({100 * (1 - post / pre):+.1f}%)"
        )
    else:
        print(f"  sfc={sfc_type}: {len(out):,} soundings (no truth column)")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ocean-model-dir", nargs="+", default=None)
    parser.add_argument("--land-model-dir", nargs="+", default=None)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ood-thresh", type=float, default=8.0)
    parser.add_argument("--max-ood-frac", type=float, default=0.02)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--climatology-max-ppm", type=float, default=50.0)
    parser.add_argument("--max-abs-anomaly", type=float, default=25.0)
    parser.add_argument("--correction-base", choices=("bc", "raw"), default="bc")
    parser.add_argument("--allow-train-overlap", action="store_true",
                        help="Downgrade the training-date leakage guard from refusal "
                             "to a loud warning (in-sample diagnostics only).")
    args = parser.parse_args()

    dk = dict(ood_thresh=args.ood_thresh, max_ood_frac=args.max_ood_frac, strict=args.strict)
    base_col = "xco2_raw" if args.correction_base == "raw" else "xco2_bc"
    truth_col = "xco2_raw_anomaly" if args.correction_base == "raw" else "xco2_bc_anomaly"

    surfaces = []
    if args.ocean_model_dir:
        surfaces.append((args.ocean_model_dir, 0))
    if args.land_model_dir:
        surfaces.append((args.land_model_dir, 1))
    if not surfaces:
        raise SystemExit("pass --ocean-model-dir and/or --land-model-dir")

    df = pd.concat([pd.read_parquet(p) for p in args.input], ignore_index=True)
    print(f"  read {len(df):,} rows from {len(args.input)} file(s)")

    for md, sfc in surfaces:
        check_training_overlap(
            md, input_paths=args.input,
            times=df["time"].to_numpy() if "time" in df.columns else None,
            allow=args.allow_train_overlap, tag=f"sfc{sfc}")

    max_abs_anom = args.max_abs_anomaly if args.max_abs_anomaly > 0 else None
    parts = [
        part
        for part in (
            _build_surface(
                df,
                fold_dirs,
                sfc_type,
                dk=dk,
                clim_max_ppm=args.climatology_max_ppm,
                max_abs_anomaly=max_abs_anom,
                base_col=base_col,
                truth_col=truth_col,
            )
            for fold_dirs, sfc_type in surfaces
        )
        if part is not None
    ]
    if not parts:
        raise SystemExit("no soundings predicted on any surface")

    out = pd.concat(parts, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"  [saved] {len(out):,} rows ({len(parts)} surface(s)) → {args.output}")
    print(f"  correction column: {CORR_COL!r}  (use --poster-model {CORR_COL})")


if __name__ == "__main__":
    main()
