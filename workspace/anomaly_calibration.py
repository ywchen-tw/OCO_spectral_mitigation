"""Fit and apply post-training anomaly calibration.

The calibration is intentionally separate from model training.  It uses
date-kfold held-out predictions to fit a small analytical map

    y_true ≈ alpha_g * mu + beta_g

where ``g`` is an interpretable regime such as global, surface, or
surface-footprint.  At plot time the calibrated anomaly is subtracted from
``xco2_bc`` to produce a new corrected-XCO2 column for the existing TCCON
comparison scripts.

No TCCON data are used here; TCCON remains the external validation target.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REGIME_CHOICES = (
    "global",
    "surface",
    "surface_fp",
    "surface_abs_mu_bin",
    "surface_sigma_bin",
    "surface_lat_band",
)


@dataclass(frozen=True)
class FitResult:
    alpha: float
    beta: float
    n: int
    rmse_before: float
    rmse_after: float


def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in patterns:
        matches = sorted(glob.glob(spec))
        paths.extend(Path(m) for m in matches)
    unique = []
    seen = set()
    for path in paths:
        key = path.resolve()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise SystemExit("no input files matched")
    return unique


def _read_prediction_file(
    path: Path,
    *,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"y_true", "mu"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    keep_cols = [
        col
        for col in ("y_true", "mu", "sigma", "sfc_type", "fp", "lat", "aod_total")
        if col in frame.columns
    ]
    return frame[keep_cols].copy()


def _fit_line(mu: np.ndarray, y: np.ndarray) -> FitResult:
    finite = np.isfinite(mu) & np.isfinite(y)
    mu = mu[finite].astype(float)
    y = y[finite].astype(float)
    n = int(mu.size)
    if n == 0:
        return FitResult(alpha=1.0, beta=0.0, n=0, rmse_before=np.nan, rmse_after=np.nan)
    if n == 1 or float(np.nanstd(mu)) == 0.0:
        alpha = 1.0
        beta = float(np.nanmean(y - mu))
    else:
        X = np.column_stack([mu, np.ones_like(mu)])
        alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha = float(alpha)
        beta = float(beta)
    before = y - mu
    after = y - (alpha * mu + beta)
    return FitResult(
        alpha=alpha,
        beta=beta,
        n=n,
        rmse_before=float(np.sqrt(np.mean(before**2))),
        rmse_after=float(np.sqrt(np.mean(after**2))),
    )


def _lat_band(lat: pd.Series | np.ndarray) -> pd.Series:
    values = pd.Series(lat, copy=False).astype(float)
    out = pd.Series("missing", index=values.index, dtype=object)
    out[values < -30.0] = "south"
    out[(values >= -30.0) & (values <= 30.0)] = "tropics"
    out[values > 30.0] = "north"
    return out


def _add_regime_columns(
    frame: pd.DataFrame,
    *,
    strategy: str,
    bin_edges: list[float] | None = None,
    n_bins: int = 4,
) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    meta: dict = {}

    if "sfc_type" in frame.columns:
        sfc = frame["sfc_type"].round().astype("Int64").astype(str)
    else:
        sfc = pd.Series("all", index=frame.index)

    if strategy == "global":
        frame["_regime"] = "global"
    elif strategy == "surface":
        frame["_regime"] = "sfc=" + sfc
    elif strategy == "surface_fp":
        if "fp" not in frame.columns:
            raise ValueError("surface_fp strategy requires an fp column")
        fp = frame["fp"].round().astype("Int64").astype(str)
        frame["_regime"] = "sfc=" + sfc + "|fp=" + fp
    elif strategy in {"surface_abs_mu_bin", "surface_sigma_bin"}:
        source = "mu" if strategy == "surface_abs_mu_bin" else "sigma"
        if source not in frame.columns:
            raise ValueError(f"{strategy} strategy requires a {source!r} column")
        values = np.abs(frame[source].to_numpy(float)) if source == "mu" else frame[source].to_numpy(float)
        finite = values[np.isfinite(values)]
        if bin_edges is None:
            if finite.size == 0:
                edges = np.array([-np.inf, np.inf], dtype=float)
            else:
                quantiles = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.unique(np.quantile(finite, quantiles))
                if edges.size < 2:
                    edges = np.array([finite.min(), finite.max()], dtype=float)
                edges[0] = -np.inf
                edges[-1] = np.inf
        else:
            edges = np.asarray(bin_edges, dtype=float)
        labels = pd.cut(values, bins=edges, labels=False, include_lowest=True)
        labels = pd.Series(labels, index=frame.index).astype("Int64").astype(str)
        frame["_regime"] = "sfc=" + sfc + f"|{source}_bin=" + labels
        meta["bin_edges"] = [float(x) for x in edges]
    elif strategy == "surface_lat_band":
        if "lat" not in frame.columns:
            raise ValueError("surface_lat_band strategy requires a lat column")
        frame["_regime"] = "sfc=" + sfc + "|lat=" + _lat_band(frame["lat"])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return frame, meta


def _shrink_toward_global(
    local: FitResult,
    global_fit: FitResult,
    *,
    prior_n: float,
) -> FitResult:
    if local.n <= 0:
        return global_fit
    weight = local.n / (local.n + prior_n)
    return FitResult(
        alpha=float(weight * local.alpha + (1.0 - weight) * global_fit.alpha),
        beta=float(weight * local.beta + (1.0 - weight) * global_fit.beta),
        n=local.n,
        rmse_before=local.rmse_before,
        rmse_after=local.rmse_after,
    )


def _fit_command(args: argparse.Namespace) -> None:
    paths = _expand_paths(args.predictions)
    frames = [
        _read_prediction_file(path, max_rows=args.max_rows_per_file, seed=args.seed + idx)
        for idx, path in enumerate(paths)
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data[np.isfinite(data["y_true"].to_numpy(float)) & np.isfinite(data["mu"].to_numpy(float))]
    if data.empty:
        raise SystemExit("no finite y_true/mu pairs available for calibration")

    data, regime_meta = _add_regime_columns(
        data,
        strategy=args.strategy,
        n_bins=args.n_bins,
    )
    global_fit = _fit_line(data["mu"].to_numpy(float), data["y_true"].to_numpy(float))

    regimes = {}
    for key, group in data.groupby("_regime", sort=True):
        local = _fit_line(group["mu"].to_numpy(float), group["y_true"].to_numpy(float))
        fit = _shrink_toward_global(local, global_fit, prior_n=args.prior_n)
        regimes[str(key)] = {
            "alpha": fit.alpha,
            "beta": fit.beta,
            "n": fit.n,
            "local_alpha": local.alpha,
            "local_beta": local.beta,
            "rmse_before": local.rmse_before,
            "rmse_after_local": local.rmse_after,
        }

    payload = {
        "version": 1,
        "kind": "post_training_anomaly_calibration",
        "strategy": args.strategy,
        "n_bins": args.n_bins,
        "prior_n": args.prior_n,
        "inputs": [str(path) for path in paths],
        "n_rows": int(len(data)),
        "global": {
            "alpha": global_fit.alpha,
            "beta": global_fit.beta,
            "n": global_fit.n,
            "rmse_before": global_fit.rmse_before,
            "rmse_after": global_fit.rmse_after,
        },
        "regime_meta": regime_meta,
        "regimes": regimes,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote calibration → {output}")
    print(
        "global: "
        f"alpha={global_fit.alpha:.4f}, beta={global_fit.beta:+.4f}, "
        f"RMSE {global_fit.rmse_before:.4f} → {global_fit.rmse_after:.4f}, "
        f"n={global_fit.n:,}"
    )
    print(f"regimes: {len(regimes)} ({args.strategy})")


def _load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "post_training_anomaly_calibration":
        raise ValueError(f"{path} is not an anomaly calibration file")
    return payload


def _calibrated_mu(frame: pd.DataFrame, payload: dict, *, mu_col: str) -> np.ndarray:
    work = frame.rename(columns={mu_col: "mu"})
    meta = payload.get("regime_meta") or {}
    work, _ = _add_regime_columns(
        work,
        strategy=payload["strategy"],
        bin_edges=meta.get("bin_edges"),
        n_bins=int(payload.get("n_bins", 4)),
    )
    global_coef = payload["global"]
    regimes = payload.get("regimes") or {}
    mu = work["mu"].to_numpy(float)
    out = np.empty(len(work), dtype=float)
    for regime, idx in work.groupby("_regime", sort=False).groups.items():
        coef = regimes.get(str(regime), global_coef)
        rows = np.asarray(list(idx), dtype=int)
        out[rows] = float(coef["alpha"]) * mu[rows] + float(coef["beta"])
    finite = np.isfinite(out) & np.isfinite(mu)
    out[~finite] = mu[~finite]
    return out


def _plot_data_paths(input_base: Path) -> list[Path]:
    if input_base.is_file():
        return [input_base]
    return sorted(input_base.glob("combined_*/plot_data.parquet"))


def _apply_one(
    path: Path,
    *,
    output_path: Path,
    payload: dict,
    mu_col: str,
    base_col: str,
    output_mu_col: str,
    output_corr_col: str,
    respect_guards: bool,
) -> None:
    frame = pd.read_parquet(path)
    for col in (mu_col, base_col):
        if col not in frame.columns:
            raise ValueError(f"{path} is missing required column {col!r}")
    mu_cal = _calibrated_mu(frame, payload, mu_col=mu_col)
    if respect_guards:
        guard = np.zeros(len(frame), dtype=bool)
        for col in ("clim_guard", "anomaly_guard"):
            if col in frame.columns:
                guard |= frame[col].fillna(False).to_numpy(bool)
        mu_cal[guard] = 0.0
    out = frame.copy()
    out[output_mu_col] = mu_cal.astype(np.float32)
    out[output_corr_col] = out[base_col].to_numpy(float) - mu_cal
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)


def _apply_command(args: argparse.Namespace) -> None:
    payload = _load_calibration(Path(args.calibration))
    input_base = Path(args.input_base)
    paths = _plot_data_paths(input_base)
    if not paths:
        raise SystemExit(f"no plot_data.parquet files found under {input_base}")

    output_base = Path(args.output_base) if args.output_base else input_base
    for path in paths:
        if input_base.is_file():
            output_path = output_base
        else:
            output_path = output_base / path.relative_to(input_base)
        _apply_one(
            path,
            output_path=output_path,
            payload=payload,
            mu_col=args.mu_col,
            base_col=args.base_col,
            output_mu_col=args.output_mu_col,
            output_corr_col=args.output_corr_col,
            respect_guards=not args.ignore_guards,
        )
    print(f"wrote calibrated plot data for {len(paths)} case(s) → {output_base}")
    print(f"correction column: {args.output_corr_col}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit calibration from held-out predictions")
    fit.add_argument("--predictions", nargs="+", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--strategy", choices=REGIME_CHOICES, default="surface")
    fit.add_argument("--n-bins", type=int, default=4)
    fit.add_argument("--prior-n", type=float, default=100_000.0)
    fit.add_argument("--max-rows-per-file", type=int, default=None)
    fit.add_argument("--seed", type=int, default=20260706)
    fit.set_defaults(func=_fit_command)

    apply = sub.add_parser("apply", help="apply calibration to plot_data parquet files")
    apply.add_argument("--calibration", required=True)
    apply.add_argument("--input-base", required=True)
    apply.add_argument("--output-base", default=None)
    apply.add_argument("--mu-col", default="pred_anomaly")
    apply.add_argument("--base-col", default="xco2_bc")
    apply.add_argument("--output-mu-col", default="calibrated_pred_anomaly")
    apply.add_argument("--output-corr-col", required=True)
    apply.add_argument("--ignore-guards", action="store_true")
    apply.set_defaults(func=_apply_command)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
