"""Standalone structured-residual / DCN-V2 ensemble experiment.

This module intentionally does not modify the production implementation in
``models.deep_ensemble``.  It reuses stable, model-agnostic utilities from that
module (loss construction, ensemble moment aggregation, and interval constant)
while owning its model factory, training loop, CLI, checkpoints, and metadata.

Every prediction consumes one sounding's compact ``FeaturePipeline`` vector.
Raw radiance spectra, neighboring soundings, orbit context, and retrieval from
other examples are never used.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import conformal as cf
from . import diagnostics as diag
from . import train_common as tc
from .dcn_v2 import DCNV2Regressor
from .deep_ensemble import Z90, _make_criterion, ensemble_predict
from .pipeline import (
    FeaturePipeline,
    _ensure_derived_features,
    filter_target_outliers,
    resolve_target_col,
)
from .splits import split_dataframe
from .structured_residual import StructuredResidualNet
from search.tracking import RunSummary, get_git_commit_hash
from utils import get_storage_dir

logger = logging.getLogger(__name__)

BACKBONES = ("structured_residual", "dcn_v2")


def build_experimental_member(
    backbone: str,
    n_features: int,
    *,
    feature_names: list[str],
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    norm: str = "none",
    block_dim: int = 16,
    cross_layers: int = 2,
    cross_rank: int = 16,
) -> nn.Module:
    """Construct one experimental ensemble member."""
    if len(feature_names) != n_features:
        raise ValueError(
            "feature_names must match the transformed input width: "
            f"{len(feature_names)} != {n_features}"
        )
    if backbone == "structured_residual":
        return StructuredResidualNet(
            feature_names,
            hidden_dims=hidden_dims,
            block_dim=block_dim,
            dropout=dropout,
            norm=norm,
        )
    if backbone == "dcn_v2":
        return DCNV2Regressor(
            n_features,
            hidden_dims=hidden_dims,
            cross_layers=cross_layers,
            cross_rank=cross_rank,
            dropout=dropout,
            norm=norm,
        )
    raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")


def train_member(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    feature_names: list[str],
    backbone: str,
    seed: int,
    device: torch.device,
    config: tc.TrainConfig,
    checkpoint: str,
    loss: str,
    nu: float,
    beta: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    norm: str,
    block_dim: int,
    cross_layers: int,
    cross_rank: int,
    gpu_resident: bool | None,
) -> nn.Module:
    """Train one member with the production optimizer and uncertainty loss."""
    criterion_fn = _make_criterion(loss, nu, beta)
    generator = tc.set_seeds(seed, config.deterministic)
    train_batches = tc.make_batches(
        (X_train, y_train),
        config.batch_size,
        shuffle=True,
        device=device,
        generator=generator,
        gpu_resident=gpu_resident,
    )
    val_batches = tc.make_batches(
        (X_val, y_val),
        config.batch_size,
        shuffle=False,
        device=device,
        generator=generator,
        gpu_resident=gpu_resident,
    )
    model = build_experimental_member(
        backbone,
        X_train.shape[1],
        feature_names=feature_names,
        hidden_dims=hidden_dims,
        dropout=dropout,
        norm=norm,
        block_dim=block_dim,
        cross_layers=cross_layers,
        cross_rank=cross_rank,
    ).to(device)

    def criterion(module: nn.Module, batch) -> torch.Tensor:
        xb, yb = batch
        mu, raw2 = module(xb)
        return criterion_fn(mu, raw2, yb)

    member_config = tc.TrainConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        pct_start=config.pct_start,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        grad_clip=config.grad_clip,
        patience=config.patience,
        seed=seed,
        deterministic=config.deterministic,
        gpu_resident=gpu_resident,
        num_workers=config.num_workers,
        log_every=config.log_every,
    )
    model, _ = tc.train_model(
        model,
        criterion,
        train_batches,
        val_batches,
        member_config,
        checkpoint,
        device,
        log_prefix=f"[{backbone} seed={seed}]",
    )
    return model.cpu()


def _date_list(frame: pd.DataFrame) -> list[str]:
    if "date" not in frame.columns:
        return []
    dates = frame["date"].astype(str).str.replace("b'", "", regex=False)
    return sorted(dates.str.replace("'", "", regex=False).unique().tolist())


def _carve_calibration(
    train_df: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use a trailing date block when possible, otherwise a random block."""
    try:
        if "date" in train_df.columns:
            dates = train_df["date"]
            if dates.dtype == object:
                dates = dates.astype(str).str.replace("b'", "").str.replace("'", "")
            if pd.to_datetime(dates).nunique() >= 2:
                return split_dataframe(train_df, mode="date", test_size=fraction)
    except (TypeError, ValueError):
        logger.warning(
            "Could not construct a date calibration block; falling back to random.",
            exc_info=True,
        )
    return split_dataframe(
        train_df,
        mode="random",
        test_size=fraction,
        random_state=seed,
    )


def _bin_values(which: str, frame: pd.DataFrame, mu: np.ndarray) -> np.ndarray:
    if which == "mu":
        return np.asarray(mu, dtype=float)
    if which not in frame.columns:
        raise ValueError(f"--mondrian_col {which!r} not in dataframe columns")
    values = frame[which].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"--mondrian_col {which!r} has no finite values")
    return np.where(finite, values, np.nanmedian(values[finite]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone structured-residual/DCN-V2 deep-ensemble experiment."
    )
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--sfc_type", type=int, choices=[0, 1], required=True)
    parser.add_argument(
        "--feature_set",
        choices=["full", "no_spec"],
        default="full",
    )
    parser.add_argument(
        "--profile-pca",
        dest="profile_pca",
        nargs="?",
        const="auto",
        default=None,
    )
    parser.add_argument("--target", default=None)
    parser.add_argument(
        "--val_split",
        choices=["random", "date", "date_kfold"],
        default="date_kfold",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--calib_frac", type=float, default=0.15)
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument("--hidden_dims", default="64,32")
    parser.add_argument("--block_dim", type=int, default=16)
    parser.add_argument("--cross_layers", type=int, default=2)
    parser.add_argument("--cross_rank", type=int, default=16)
    parser.add_argument(
        "--loss",
        choices=["gaussian_nll", "beta_nll", "student_t"],
        default="beta_nll",
    )
    parser.add_argument("--nu", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--norm",
        choices=["none", "layer", "batch"],
        default="layer",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--gpu_resident",
        choices=["auto", "on", "off"],
        default="auto",
    )
    parser.add_argument("--near_cloud_target", type=float, default=None)
    parser.add_argument("--near_cloud_km", type=float, default=10.0)
    parser.add_argument("--mondrian_bins", type=int, default=10)
    parser.add_argument("--mondrian_col", default="mu")
    parser.add_argument("--exclude_snow", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--data", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_start = time.monotonic()
    storage_dir = get_storage_dir()
    output_root = storage_dir / "results/model_structured_dcn_ensemble"
    output_dir = output_root / args.suffix if args.suffix else output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.suffix or datetime.now().strftime("%Y%m%d-%H%M%S")

    data_path = os.fspath(args.data)
    frame = (
        pd.read_parquet(data_path)
        if data_path.endswith(".parquet")
        else pd.read_csv(data_path)
    )
    frame = frame[frame["sfc_type"] == args.sfc_type]
    if args.exclude_snow:
        frame = frame[frame["snow_flag"] == 0]
    frame = _ensure_derived_features(frame)
    target_col = resolve_target_col(args.target)
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} is absent from {data_path}")
    frame = filter_target_outliers(frame, target_col=target_col)

    train_df, held_df = split_dataframe(
        frame,
        mode=args.val_split,
        test_size=args.test_size,
        random_state=args.seed,
        n_folds=args.n_folds,
        fold=args.fold,
    )
    del frame
    gc.collect()
    proper_df, calib_df = _carve_calibration(
        train_df,
        fraction=args.calib_frac,
        seed=args.seed,
    )
    del train_df
    gc.collect()

    with open(output_dir / "training_dates.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "train_dates": _date_list(proper_df),
                "calib_dates": _date_list(calib_df),
                "held_dates": _date_list(held_df),
            },
            stream,
            indent=2,
        )

    profile_pca = True if args.profile_pca in (True, "auto") else args.profile_pca
    pipeline = FeaturePipeline.fit(
        proper_df,
        sfc_type=args.sfc_type,
        feature_set=args.feature_set,
        profile_pca=profile_pca,
    )
    pipeline.save(output_dir / "experimental_pipeline.pkl")
    feature_names = list(pipeline.feature_names)

    def prepare(dataframe: pd.DataFrame):
        X = pipeline.transform(dataframe)
        y = dataframe[target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        return X[valid], y[valid], dataframe.loc[valid]

    X_train, y_train, train_valid = prepare(proper_df)
    X_calib, y_calib, calib_valid = prepare(calib_df)
    X_held, y_held, held_valid = prepare(held_df)
    del proper_df, calib_df, held_df, train_valid
    gc.collect()

    hidden_dims = tuple(
        int(width) for width in args.hidden_dims.split(",") if width.strip()
    )
    device = tc.select_device()
    train_config = tc.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
    ).resolved()
    gpu_resident = {"auto": None, "on": True, "off": False}[args.gpu_resident]

    print(
        f"[{args.backbone}] train={X_train.shape} calib={X_calib.shape} "
        f"held={X_held.shape} features={args.feature_set}+profile"
    )
    members = []
    for member_idx in range(args.n_members):
        seed = args.seed + member_idx
        checkpoint = output_dir / f"member_{member_idx}.pt"
        print(
            f"  training member {member_idx + 1}/{args.n_members} "
            f"(seed={seed}, backbone={args.backbone})"
        )
        members.append(
            train_member(
                X_train,
                y_train,
                X_calib,
                y_calib,
                feature_names=feature_names,
                backbone=args.backbone,
                seed=seed,
                device=device,
                config=train_config,
                checkpoint=os.fspath(checkpoint),
                loss=args.loss,
                nu=args.nu,
                beta=args.beta,
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                norm=args.norm,
                block_dim=args.block_dim,
                cross_layers=args.cross_layers,
                cross_rank=args.cross_rank,
                gpu_resident=gpu_resident,
            )
        )

    del X_train, y_train
    gc.collect()

    mu_calib, sigma_calib = ensemble_predict(
        members,
        X_calib,
        device,
        loss=args.loss,
        nu=args.nu,
    )
    mu_held, sigma_held = ensemble_predict(
        members,
        X_held,
        device,
        loss=args.loss,
        nu=args.nu,
    )

    raw = np.column_stack(
        [
            mu_held - Z90 * sigma_held,
            mu_held,
            mu_held + Z90 * sigma_held,
        ]
    )
    split, split_q = cf.split_conformal(
        y_calib,
        mu_calib,
        sigma_calib,
        mu_held,
        sigma_held,
        alpha=0.10,
    )
    calib_values = _bin_values(args.mondrian_col, calib_valid, mu_calib)
    held_values = _bin_values(args.mondrian_col, held_valid, mu_held)
    calib_bins, edges = cf.make_quantile_bins(
        calib_values,
        args.mondrian_bins,
    )
    held_bins, _ = cf.make_quantile_bins(
        held_values,
        args.mondrian_bins,
        edges=edges,
    )

    bin_alpha = None
    if args.near_cloud_target is not None:
        if args.mondrian_col != "cld_dist_km":
            raise ValueError(
                "--near_cloud_target requires --mondrian_col cld_dist_km"
            )
        near_calib = (
            calib_valid["cld_dist_km"].to_numpy(dtype=float)
            <= args.near_cloud_km
        )
        bin_alpha = cf.regime_alphas(
            calib_bins,
            near_calib,
            near_alpha=1.0 - args.near_cloud_target,
            far_alpha=0.10,
        )
    mondrian, q_by_bin = cf.mondrian_conformal(
        y_calib,
        mu_calib,
        sigma_calib,
        calib_bins,
        mu_held,
        sigma_held,
        held_bins,
        alpha=0.10,
        bin_alpha=bin_alpha,
    )

    results = {}
    for tag, predictions in (
        ("raw", raw),
        ("split", split),
        ("mondrian", mondrian),
    ):
        metrics = diag.compute_metrics(y_held, predictions)
        stratified = diag.stratified_metrics(
            held_valid,
            y_held,
            predictions,
        )
        prefix = f"{args.backbone}_{tag}_{args.val_split}"
        diag.save_diagnostics(output_dir, prefix, metrics, stratified)
        results[tag] = metrics
        print(
            f"[{prefix}] RMSE={metrics['rmse']:.4f} "
            f"R²={metrics['r2']:.4f} "
            f"cov90={metrics['coverage_90']:.4f}"
        )

    correction = diag.correction_by_cloud_distance(
        held_valid,
        y_held,
        mu_held,
    )
    if not correction.empty:
        correction.to_csv(
            output_dir / f"{args.backbone}_correction_clddist.csv",
            index=False,
        )

    held_predictions = pd.DataFrame(
        {
            "y_true": y_held,
            "mu": mu_held,
            "sigma": sigma_held,
            "lo_mondrian": mondrian[:, 0],
            "hi_mondrian": mondrian[:, 2],
        }
    )
    for column in (
        "date",
        "cld_dist_km",
        "sfc_type",
        "aod_total",
        "fp",
        "lat",
        "snow_flag",
    ):
        if column in held_valid.columns:
            held_predictions[column] = held_valid[column].to_numpy()
    held_predictions.to_parquet(
        output_dir / "held_out_predictions.parquet",
        index=False,
    )

    metadata = {
        "backbone": args.backbone,
        "n_features": pipeline.n_features,
        "feature_names": feature_names,
        "n_members": args.n_members,
        "feature_set": args.feature_set,
        "profile_pca": bool(profile_pca),
        "target_col": target_col,
        "loss": args.loss,
        "nu": args.nu,
        "beta": args.beta,
        "hidden_dims": list(hidden_dims),
        "block_dim": args.block_dim,
        "cross_layers": args.cross_layers,
        "cross_rank": args.cross_rank,
        "dropout": args.dropout,
        "norm": args.norm,
        "q_split": split_q,
        "q_by_bin": q_by_bin,
        "mondrian_edges": edges.tolist(),
        "mondrian_col": args.mondrian_col,
        "near_cloud_target": args.near_cloud_target,
        "near_cloud_km": args.near_cloud_km,
        "train_config": train_config.to_dict(),
    }
    if args.backbone == "structured_residual":
        metadata["feature_groups"] = members[0].feature_groups
    with open(output_dir / "experimental_meta.pkl", "wb") as stream:
        pickle.dump(metadata, stream)

    headline = results["mondrian"]
    summary = RunSummary(
        run_id=run_id,
        script_name=os.path.basename(__file__),
        model_family=f"experimental_{args.backbone}",
        commit=get_git_commit_hash(storage_dir),
        status="success",
        primary_metric_name="held_rmse",
        primary_metric_value=headline["rmse"],
        secondary_metrics={
            "held_r2": headline["r2"],
            "mondrian_cov90": headline["coverage_90"],
            "split_cov90": results["split"]["coverage_90"],
            "raw_cov90": results["raw"]["coverage_90"],
        },
        runtime_seconds=float(time.monotonic() - run_start),
        description=(
            f"{args.backbone} ensemble M={args.n_members}, "
            f"{args.feature_set}+profile, {args.val_split}"
        ),
        artifacts={
            "output_dir": os.fspath(output_dir),
            "metrics_json": os.fspath(
                output_dir
                / f"{args.backbone}_mondrian_{args.val_split}_metrics.json"
            ),
        },
        config={
            "sfc_type": args.sfc_type,
            "fold": args.fold,
            "n_folds": args.n_folds,
            "feature_set": args.feature_set,
            "backbone": args.backbone,
            "seed": args.seed,
            "loss": args.loss,
            "beta": args.beta,
        },
    )
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary.to_dict(), stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
