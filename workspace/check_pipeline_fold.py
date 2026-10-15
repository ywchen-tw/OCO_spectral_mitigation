"""check_pipeline_fold.py — fit one fold's FeaturePipeline on real data and audit the scaling.

The pre-submission check of log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md.  Repeats
the deep-ensemble filter chain (surface → derived features → target outliers →
date k-fold split) for one fold, fits FeaturePipeline on the proper-train block
with the new defaults, and reports per predictor:

  n_imputed  fills/spikes replaced by the training median (train block)
  sd_train   SD of the scaled predictor on the train block   (pass: 0.3-3)
  sd_held    SD of the scaled predictor on the held-out fold
  clipped    training fraction beyond the RobustClipScaler clip  (pass: < 0.5 %)
  sd_legacy  the same SD under the pre-fix RobustStandardScaler, for contrast

and the alt_std 5 m vs 50 m mapping (land).  The profile block is left out: its
inputs are not masked and its scaler is unchanged.

Run:  PYTHONPATH=src python workspace/check_pipeline_fold.py --sfc-type 1 --fold 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from models.pipeline import (_DERIVED_FEATURES, _FEATURE_MAP, FeaturePipeline,
                             _ensure_derived_features, filter_target_outliers)
from models.splits import split_date_kfold_train_calib_test

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "results" / "csv_collection" / "combined_2016_2020_dates.parquet"
TARGET = {0: "xco2_bc_anomaly_r05", 1: "xco2_bc_anomaly_r15"}
SD_RANGE = (0.3, 3.0)
CLIP_MAX = 0.005


def _load(sfc_type: int, path: Path) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    need = set(_FEATURE_MAP[sfc_type]) | {"fp", "date", "sfc_type", "snow_flag",
                                          TARGET[sfc_type]}
    for spec in _DERIVED_FEATURES.values():
        need |= set(spec.bases)
    cols = sorted(c for c in need if c in names)
    df = pd.read_parquet(path, columns=cols, filters=[("sfc_type", "==", float(sfc_type))])
    df = _ensure_derived_features(df)
    return filter_target_outliers(df, target_col=TARGET[sfc_type])


def _qt(pipe: FeaturePipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipe.transform(frame)[:, :len(pipe.qt_features)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sfc-type", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--data", type=Path, default=PARQUET)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    df = _load(args.sfc_type, args.data)
    proper, _calib, held = split_date_kfold_train_calib_test(
        df, n_folds=args.n_folds, fold=args.fold, calib_frac=0.15)
    del df
    print(f"sfc_type={args.sfc_type} fold={args.fold}: train {len(proper):,} rows, "
          f"held {len(held):,} rows")

    pipe = FeaturePipeline.fit(proper, sfc_type=args.sfc_type)
    legacy = FeaturePipeline.fit(proper, sfc_type=args.sfc_type, scaler="robust_standard")
    del legacy.impute_values          # pre-fix behaviour: no masking

    from models.pipeline import _invalid_mask
    raw = proper[pipe.qt_features].to_numpy(dtype=float)
    n_imputed = _invalid_mask(raw, pipe.qt_features, pipe.valid_range).sum(axis=0)
    del raw

    Z_tr, Z_held, Z_old = _qt(pipe, proper), _qt(pipe, held), _qt(legacy, proper)
    report = pd.DataFrame({
        "n_imputed": n_imputed,
        "sd_train": np.nanstd(Z_tr, axis=0),
        "sd_held": np.nanstd(Z_held, axis=0),
        "clipped": pipe.qt.clipped_frac_,
        "sd_legacy": np.nanstd(Z_old, axis=0),
    }, index=pipe.qt_features)
    with pd.option_context("display.max_rows", None, "display.width", 120,
                           "display.float_format", "{:.4g}".format):
        print(report)

    failures = report[(report.sd_train < SD_RANGE[0]) | (report.sd_train > SD_RANGE[1])
                      | (report.clipped > CLIP_MAX)]
    if "alt_std" in pipe.qt_features:
        probe = proper.iloc[:2].copy()
        probe["alt_std"] = [5.0, 50.0]
        z = _qt(pipe, probe)[:, pipe.qt_features.index("alt_std")]
        print(f"alt_std 5 m -> {z[0]:.3f}, 50 m -> {z[1]:.3f}")
    print("PASS" if failures.empty else f"FAIL:\n{failures}")
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
