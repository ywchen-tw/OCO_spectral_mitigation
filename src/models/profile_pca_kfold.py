"""Fit leakage-safe fold-specific ProfilePCA artifacts for date-kfold runs.

This module mirrors the model validation split used by
``models.structured_dcn_ensemble``:

1. filter target outliers;
2. hold out one contiguous date-kfold block;
3. carve a fold-local adjacent calibration date block;
4. fit ProfilePCA on the proper-training block only.

The resulting artifacts can be passed to ``FeaturePipeline.fit`` via
``--profile-pca /path/to/profile_pca_<surface>.pkl``.  They avoid the mild
unsupervised leakage caused by fitting one profile EOF basis on all 2016-2020
dates before blocked validation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .pipeline import filter_target_outliers, resolve_target_col
from .profile_pca import (
    ProfilePCA,
    _PROFILE_GROUPS,
    _SCALAR_PASSTHROUGH,
)
from .splits import split_date_kfold_train_calib_test
from utils import get_storage_dir

logger = logging.getLogger(__name__)

SURFACE_NAMES = {0: "ocean", 1: "land"}


def _resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else get_storage_dir() / candidate


def _date_list(frame: pd.DataFrame) -> list[str]:
    if "date" not in frame.columns:
        return []
    dates = frame["date"]
    if dates.dtype == object and len(dates) > 0 and isinstance(dates.iloc[0], bytes):
        dates = dates.str.decode("utf-8")
    return sorted(pd.to_datetime(dates).dt.strftime("%Y-%m-%d").unique().tolist())


def _read_profile_frame(data_path: Path, target_col: str) -> pd.DataFrame:
    """Read only columns needed for fold splitting, target filtering, and PCA."""
    if str(data_path).endswith(".parquet"):
        import pyarrow.parquet as pq

        available = set(pq.read_schema(data_path).names)
        columns = [
            col
            for col in available
            if any(col.startswith(prefix) for prefix in _PROFILE_GROUPS)
        ]
        columns += [col for col in _SCALAR_PASSTHROUGH if col in available]
        columns += [
            col
            for col in ("sfc_type", "date", "sounding_id", "orbit_id", target_col)
            if col in available
        ]
        return pd.read_parquet(data_path, columns=sorted(set(columns)))
    return pd.read_csv(data_path)


def _parse_components(raw: str) -> Union[int, float]:
    return float(raw) if "." in raw else int(raw)


def _fit_surface(
    proper_df: pd.DataFrame,
    *,
    sfc_type: int,
    n_components: Union[int, float],
    groups: dict,
    out_dir: Path,
) -> None:
    surface = SURFACE_NAMES[sfc_type]
    surface_df = proper_df[proper_df["sfc_type"] == sfc_type]
    if len(surface_df) < 2:
        raise ValueError(
            f"Fold-specific ProfilePCA for {surface} has too few rows: "
            f"{len(surface_df)}"
        )

    logger.info(
        "Fitting fold ProfilePCA: surface=%s rows=%d n_components=%s",
        surface,
        len(surface_df),
        n_components,
    )
    ppca = ProfilePCA.fit(surface_df, n_components=n_components, groups=groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppca.save(out_dir / f"profile_pca_{surface}.pkl")
    ppca.explained_variance_report().to_csv(
        out_dir / f"profile_pca_{surface}_explained_variance.csv",
        index=False,
        float_format="%.6f",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-fold, per-surface ProfilePCA artifacts for leakage-safe "
            "date_kfold model validation."
        )
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Input parquet/csv. Defaults to "
            "<storage>/results/csv_collection/combined_2016_2020_dates.parquet."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output root. Defaults to "
            "<storage>/results/profile_pca_date_kfold_2016_2020."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--calib-frac", type=float, default=0.15)
    parser.add_argument("--target", default="10km")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", default="4")
    parser.add_argument(
        "--co2-norm",
        default="mean",
        choices=["none", "mean", "max"],
        help="Per-sounding CO2-prior profile normalization before PCA.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    storage = get_storage_dir()
    data_path = _resolve_path(
        args.data,
        storage / "results/csv_collection/combined_2016_2020_dates.parquet",
    )
    out_root = _resolve_path(
        args.out_dir,
        storage / "results/profile_pca_date_kfold_2016_2020",
    )
    out_dir = out_root / f"fold{args.fold}"
    target_col = resolve_target_col(args.target)
    n_components = _parse_components(args.n_components)

    frame = _read_profile_frame(data_path, target_col)
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} is absent from {data_path}")
    frame = filter_target_outliers(frame, target_col=target_col)

    proper_df, calib_df, held_df = split_date_kfold_train_calib_test(
        frame,
        n_folds=args.n_folds,
        fold=args.fold,
        calib_frac=args.calib_frac,
    )

    groups = {prefix: dict(spec) for prefix, spec in _PROFILE_GROUPS.items()}
    groups["co2prior_sigma_"]["norm"] = (
        None if args.co2_norm == "none" else args.co2_norm
    )

    for sfc_type in (0, 1):
        _fit_surface(
            proper_df,
            sfc_type=sfc_type,
            n_components=n_components,
            groups=groups,
            out_dir=out_dir,
        )

    with (out_dir / "fold_dates.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "fold": args.fold,
                "n_folds": args.n_folds,
                "calib_frac": args.calib_frac,
                "target_col": target_col,
                "proper_dates": _date_list(proper_df),
                "calib_dates": _date_list(calib_df),
                "held_dates": _date_list(held_df),
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    logger.info("Saved fold-specific ProfilePCA artifacts under %s", out_dir)


if __name__ == "__main__":
    main()
