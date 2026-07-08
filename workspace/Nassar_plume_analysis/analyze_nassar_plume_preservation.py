#!/usr/bin/env python
"""Analyze whether the Nassar plume-control cases preserve real XCO2 structure.

This is intentionally not a flux inversion.  It summarizes the local XCO2
enhancement, model correction, and spectral-coefficient behavior near selected
power plants.  A downwind-side check is included when wind direction is supplied;
the local feature files currently carry wind speed but not wind direction.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PLANTS = Path(__file__).with_name("nassar_power_plants.csv")
DEFAULT_PLOT_BASE = Path(
    "results/model_comparison/deep_ensemble/"
    "de_beta_nll_prof_reg_o05l15_m5/nassar_plumes"
)
DEFAULT_CSV_DIR = Path("results/csv_collection")
DEFAULT_OUTPUT_DIR = DEFAULT_PLOT_BASE / "plume_preservation"
DEFAULT_PAIRS = (
    ("ghent", "2024-04-15"),
    ("comanche", "2020-09-06"),
    ("colstrip", "2015-08-01"),
)

VALUE_COLUMNS = (
    "xco2_bc",
    "deep_ensemble_corrected_xco2",
    "pred_anomaly",
    "xco2_raw",
    "xco2_bc_anomaly",
    "xco2_raw_anomaly",
    "o2a_k1",
    "o2a_k2",
    "wco2_k1",
    "wco2_k2",
    "sco2_k1",
    "sco2_k2",
    "cld_dist_km",
    "ws",
)
FOOTPRINT_GROUP_COLUMNS = (
    "xco2_bc",
    "deep_ensemble_corrected_xco2",
    "pred_anomaly",
    "cld_dist_km",
    "o2a_k1",
    "o2a_k2",
    "wco2_k1",
    "wco2_k2",
)
SPECTRAL_CLOUD_COLUMNS = ("o2a_k1", "o2a_k2", "wco2_k1", "wco2_k2")


def read_plants(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        return {row["plant_id"]: row for row in csv.DictReader(stream)}


def normalize_date(value: str) -> str:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def parse_pair(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Pair must look like plant_id:YYYY-MM-DD")
    plant_id, date = value.split(":", 1)
    return plant_id.strip(), normalize_date(date)


def parse_wind_spec(value: str) -> tuple[tuple[str, str], float]:
    """Parse plant:date:deg wind direction inputs."""
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Wind spec must look like plant_id:YYYY-MM-DD:deg")
    plant_id, date, deg = parts
    return (plant_id.strip(), normalize_date(date)), float(deg) % 360.0


def angular_difference_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def bearing_deg(src_lat: float, src_lon: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Initial bearing from source to footprints, degrees clockwise from north."""
    lat1 = math.radians(src_lat)
    lat2 = np.radians(lat.astype(float))
    dlon = np.radians(lon.astype(float) - src_lon)
    y = np.sin(dlon) * np.cos(lat2)
    x = math.cos(lat1) * np.sin(lat2) - math.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def haversine_km(lat: np.ndarray, lon: np.ndarray, src_lat: float, src_lon: float) -> np.ndarray:
    radius_km = 6371.0088
    lat1 = np.radians(lat.astype(float))
    lat2 = math.radians(src_lat)
    dlat = lat1 - lat2
    dlon = np.radians(lon.astype(float) - src_lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def finite_array(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def stats_for_column(values: pd.Series, prefix: str) -> dict[str, float | int]:
    arr = finite_array(values)
    if len(arr) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_max_minus_min": np.nan,
            f"{prefix}_p95_minus_p05": np.nan,
            f"{prefix}_p90_minus_p10": np.nan,
        }
    return {
        f"{prefix}_n": int(len(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_max_minus_min": float(np.max(arr) - np.min(arr)),
        f"{prefix}_p95_minus_p05": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
        f"{prefix}_p90_minus_p10": float(np.percentile(arr, 90) - np.percentile(arr, 10)),
    }


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def corr(values_a: pd.Series, values_b: pd.Series) -> float:
    frame = pd.concat([values_a, values_b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return np.nan
    arr_a = frame.iloc[:, 0].to_numpy(dtype=float)
    arr_b = frame.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(arr_a) < 1e-12 or np.nanstd(arr_b) < 1e-12:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def add_cloud_context(out: dict[str, object], subset: pd.DataFrame) -> None:
    """Add diagnostics that put spectral spread in cloud-distance context."""
    cld_spread = float(out.get("cld_dist_km_p95_minus_p05", np.nan))
    cld_std = float(out.get("cld_dist_km_std", np.nan))
    cld_median = float(out.get("cld_dist_km_median", np.nan))
    if not np.isfinite(cld_spread):
        regime = "missing_cloud_distance"
    elif cld_spread < 5.0:
        if np.isfinite(cld_median) and cld_median < 20.0:
            regime = "homogeneous_close_to_cloud"
        elif np.isfinite(cld_median):
            regime = "homogeneous_far_from_cloud"
        else:
            regime = "homogeneous_cloud_distance"
    else:
        regime = "mixed_cloud_distance"
    out["cloud_distance_regime"] = regime

    for col in SPECTRAL_CLOUD_COLUMNS:
        if col not in subset.columns:
            continue
        std = float(out.get(f"{col}_std", np.nan))
        spread = float(out.get(f"{col}_p95_minus_p05", np.nan))
        out[f"{col}_std_per_cld_dist_std"] = safe_ratio(std, cld_std)
        out[f"{col}_p95_p05_per_cld_dist_p95_p05"] = safe_ratio(spread, cld_spread)
        if "cld_dist_km" in subset.columns:
            out[f"corr_cld_dist_km_vs_{col}"] = corr(subset["cld_dist_km"], subset[col])


def read_feature_columns(path: Path, key_cols: Iterable[str], value_cols: Iterable[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None

    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(path).names)
    columns = [c for c in [*key_cols, *value_cols] if c in schema_cols]
    if not set(key_cols).issubset(columns):
        return None
    return pd.read_parquet(path, columns=columns)


def add_feature_columns(df: pd.DataFrame, csv_dir: Path, date: str) -> pd.DataFrame:
    key_cols = [c for c in ("time", "lon", "lat") if c in df.columns]
    if len(key_cols) < 3:
        return df

    wanted = (
        "o2a_k1",
        "o2a_k2",
        "wco2_k1",
        "wco2_k2",
        "sco2_k1",
        "sco2_k2",
        "ws",
        "ws_apriori",
        "xco2_qf",
        "psfc",
    )
    missing = [col for col in wanted if col not in df.columns]
    if not missing:
        return df

    feature_path = csv_dir / f"combined_{date}_all_orbits.parquet"
    features = read_feature_columns(feature_path, key_cols, wanted)
    if features is None:
        print(f"  Warning: feature columns unavailable for {date}: {feature_path}", flush=True)
        return df

    return df.merge(features.drop_duplicates(key_cols), on=key_cols, how="left", sort=False)


def wind_maps(args: argparse.Namespace) -> dict[tuple[str, str], float]:
    winds: dict[tuple[str, str], float] = {}
    for key, direction_from in args.wind_from:
        winds[key] = (direction_from + 180.0) % 360.0
    for key, direction_to in args.wind_to:
        winds[key] = direction_to % 360.0

    if args.wind_file:
        table = pd.read_csv(args.wind_file)
        required = {"plant_id", "date"}
        missing = required - set(table.columns)
        if missing:
            raise SystemExit(f"{args.wind_file} missing required columns: {sorted(missing)}")
        for row in table.itertuples(index=False):
            key = (str(row.plant_id), normalize_date(str(row.date)))
            if "wind_to_deg" in table.columns and pd.notna(row.wind_to_deg):
                winds[key] = float(row.wind_to_deg) % 360.0
            elif "wind_from_deg" in table.columns and pd.notna(row.wind_from_deg):
                winds[key] = (float(row.wind_from_deg) + 180.0) % 360.0
    return winds


def summarize_subset(
    pair: tuple[str, str],
    source: dict[str, str],
    df: pd.DataFrame,
    radius_km: float,
    subset_name: str,
    subset: pd.DataFrame,
    wind_to_deg: float | None,
) -> dict[str, object]:
    plant_id, date = pair
    out: dict[str, object] = {
        "plant_id": plant_id,
        "source_name": source["source_name"],
        "date": date,
        "radius_km": radius_km,
        "subset": subset_name,
        "n": int(len(subset)),
        "n_total_case": int(len(df)),
        "min_source_dist_km": float(np.nanmin(df["source_dist_km"])) if len(df) else np.nan,
        "median_source_dist_km": float(np.nanmedian(subset["source_dist_km"])) if len(subset) else np.nan,
        "median_cld_dist_km": float(np.nanmedian(subset["cld_dist_km"])) if "cld_dist_km" in subset and len(subset) else np.nan,
        "wind_to_deg": wind_to_deg if wind_to_deg is not None else np.nan,
    }

    if wind_to_deg is None:
        out.update(
            {
                "downwind_status": "missing_wind_direction",
                "n_downwind_half_plane": np.nan,
                "frac_downwind_half_plane": np.nan,
                "n_downwind_core_45deg": np.nan,
                "frac_downwind_core_45deg": np.nan,
                "median_abs_angle_from_downwind_deg": np.nan,
            }
        )
    elif len(subset):
        angle = angular_difference_deg(subset["source_bearing_deg"].to_numpy(dtype=float), wind_to_deg)
        out.update(
            {
                "downwind_status": "ok",
                "n_downwind_half_plane": int(np.count_nonzero(angle <= 90.0)),
                "frac_downwind_half_plane": float(np.mean(angle <= 90.0)),
                "n_downwind_core_45deg": int(np.count_nonzero(angle <= 45.0)),
                "frac_downwind_core_45deg": float(np.mean(angle <= 45.0)),
                "median_abs_angle_from_downwind_deg": float(np.nanmedian(angle)),
            }
        )
    else:
        out.update(
            {
                "downwind_status": "no_footprints",
                "n_downwind_half_plane": 0,
                "frac_downwind_half_plane": np.nan,
                "n_downwind_core_45deg": 0,
                "frac_downwind_core_45deg": np.nan,
                "median_abs_angle_from_downwind_deg": np.nan,
            }
        )

    for col in VALUE_COLUMNS:
        if col in subset.columns:
            out.update(stats_for_column(subset[col], col))

    orig = out.get("xco2_bc_p95_minus_p05", np.nan)
    corr_xco2 = out.get("deep_ensemble_corrected_xco2_p95_minus_p05", np.nan)
    mu = out.get("pred_anomaly_p95_minus_p05", np.nan)
    out["corrected_over_original_p95_p05"] = safe_ratio(float(corr_xco2), float(orig))
    out["mu_over_original_p95_p05"] = safe_ratio(float(mu), float(orig))

    if {"xco2_bc", "pred_anomaly"}.issubset(subset.columns):
        out["corr_xco2_bc_vs_mu"] = corr(subset["xco2_bc"], subset["pred_anomaly"])
    if {"xco2_bc", "deep_ensemble_corrected_xco2"}.issubset(subset.columns):
        out["corr_original_vs_corrected"] = corr(
            subset["xco2_bc"], subset["deep_ensemble_corrected_xco2"]
        )
    if {"o2a_k1", "wco2_k1"}.issubset(subset.columns):
        out["wco2_k1_over_o2a_k1_p95_p05"] = safe_ratio(
            float(out.get("wco2_k1_p95_minus_p05", np.nan)),
            float(out.get("o2a_k1_p95_minus_p05", np.nan)),
        )
    out["delta_wco2_k1_max_minus_min"] = out.get("wco2_k1_max_minus_min", np.nan)
    out["delta_wco2_k1_p95_minus_p05"] = out.get("wco2_k1_p95_minus_p05", np.nan)
    add_cloud_context(out, subset)

    return out


def summarize_footprint_groups(
    pair: tuple[str, str],
    source: dict[str, str],
    radius_km: float,
    subset_name: str,
    subset: pd.DataFrame,
    wind_to_deg: float | None,
) -> list[dict[str, object]]:
    plant_id, date = pair
    rows = []
    if "fp" not in subset.columns:
        return rows

    for fp in range(8):
        group = subset[subset["fp"] == fp]
        row: dict[str, object] = {
            "plant_id": plant_id,
            "source_name": source["source_name"],
            "date": date,
            "radius_km": radius_km,
            "subset": subset_name,
            "fp": fp,
            "n": int(len(group)),
            "wind_to_deg": wind_to_deg if wind_to_deg is not None else np.nan,
        }
        if wind_to_deg is None:
            row.update(
                {
                    "downwind_status": "missing_wind_direction",
                    "frac_downwind_half_plane": np.nan,
                    "frac_downwind_core_45deg": np.nan,
                    "median_abs_angle_from_downwind_deg": np.nan,
                }
            )
        elif len(group):
            angle = angular_difference_deg(group["source_bearing_deg"].to_numpy(dtype=float), wind_to_deg)
            row.update(
                {
                    "downwind_status": "ok",
                    "frac_downwind_half_plane": float(np.mean(angle <= 90.0)),
                    "frac_downwind_core_45deg": float(np.mean(angle <= 45.0)),
                    "median_abs_angle_from_downwind_deg": float(np.nanmedian(angle)),
                }
            )
        else:
            row.update(
                {
                    "downwind_status": "no_footprints",
                    "frac_downwind_half_plane": np.nan,
                    "frac_downwind_core_45deg": np.nan,
                    "median_abs_angle_from_downwind_deg": np.nan,
                }
            )

        row["median_source_dist_km"] = (
            float(np.nanmedian(group["source_dist_km"])) if len(group) else np.nan
        )
        row["median_cld_dist_km"] = (
            float(np.nanmedian(group["cld_dist_km"]))
            if "cld_dist_km" in group.columns and len(group)
            else np.nan
        )

        for col in FOOTPRINT_GROUP_COLUMNS:
            if col in group.columns:
                row.update(stats_for_column(group[col], col))
        row["delta_wco2_k1_max_minus_min"] = row.get("wco2_k1_max_minus_min", np.nan)
        row["delta_wco2_k1_p95_minus_p05"] = row.get("wco2_k1_p95_minus_p05", np.nan)
        add_cloud_context(row, group)
        rows.append(row)
    return rows


def analyze_pair(
    pair: tuple[str, str],
    source: dict[str, str],
    plot_base: Path,
    csv_dir: Path,
    radii_km: Iterable[float],
    clear_cloud_km: float,
    wind_to_deg: float | None,
    footprint_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    plant_id, date = pair
    plot_data = plot_base / f"combined_{date}" / "plot_data.parquet"
    if not plot_data.exists():
        raise FileNotFoundError(plot_data)

    src_lat = float(source["lat"])
    src_lon = float(source["lon"])
    df = pd.read_parquet(plot_data)
    df = add_feature_columns(df, csv_dir, date)
    df = df.assign(
        source_dist_km=haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(), src_lat, src_lon),
        source_bearing_deg=bearing_deg(src_lat, src_lon, df["lat"].to_numpy(), df["lon"].to_numpy()),
    )

    if wind_to_deg is not None:
        df["abs_angle_from_downwind_deg"] = angular_difference_deg(
            df["source_bearing_deg"].to_numpy(dtype=float), wind_to_deg
        )
        df["is_downwind_half_plane"] = df["abs_angle_from_downwind_deg"] <= 90.0
        df["is_downwind_core_45deg"] = df["abs_angle_from_downwind_deg"] <= 45.0
    else:
        df["abs_angle_from_downwind_deg"] = np.nan
        df["is_downwind_half_plane"] = pd.NA
        df["is_downwind_core_45deg"] = pd.NA

    footprint_cols = [
        c
        for c in (
            "time",
            "lat",
            "lon",
            "fp",
            "source_dist_km",
            "source_bearing_deg",
            "abs_angle_from_downwind_deg",
            "is_downwind_half_plane",
            "is_downwind_core_45deg",
            *VALUE_COLUMNS,
        )
        if c in df.columns
    ]
    footprint_dir.mkdir(parents=True, exist_ok=True)
    df.loc[df["source_dist_km"] <= max(radii_km), footprint_cols].to_csv(
        footprint_dir / f"nassar_plume_footprints_{plant_id}_{date}.csv", index=False
    )

    rows = []
    fp_rows = []
    for radius_km in radii_km:
        near = df[df["source_dist_km"] <= radius_km].copy()
        rows.append(summarize_subset(pair, source, df, radius_km, "all", near, wind_to_deg))
        fp_rows.extend(
            summarize_footprint_groups(pair, source, radius_km, "all", near, wind_to_deg)
        )
        if "cld_dist_km" in near.columns:
            clear = near[near["cld_dist_km"] >= clear_cloud_km].copy()
            rows.append(
                summarize_subset(
                    pair,
                    source,
                    df,
                    radius_km,
                    f"clear_cld_ge_{clear_cloud_km:g}km",
                    clear,
                    wind_to_deg,
                )
            )
            fp_rows.extend(
                summarize_footprint_groups(
                    pair,
                    source,
                    radius_km,
                    f"clear_cld_ge_{clear_cloud_km:g}km",
                    clear,
                    wind_to_deg,
                )
            )
    return rows, fp_rows


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    keep = [
        "plant_id",
        "date",
        "radius_km",
        "subset",
        "n",
        "downwind_status",
        "frac_downwind_half_plane",
        "frac_downwind_core_45deg",
        "xco2_bc_max_minus_min",
        "xco2_bc_p95_minus_p05",
        "deep_ensemble_corrected_xco2_max_minus_min",
        "deep_ensemble_corrected_xco2_p95_minus_p05",
        "pred_anomaly_p95_minus_p05",
        "corrected_over_original_p95_p05",
        "mu_over_original_p95_p05",
        "corr_xco2_bc_vs_mu",
        "cld_dist_km_median",
        "cld_dist_km_std",
        "cld_dist_km_p95_minus_p05",
        "cloud_distance_regime",
        "o2a_k1_p95_minus_p05",
        "o2a_k2_p95_minus_p05",
        "delta_wco2_k1_max_minus_min",
        "wco2_k1_p95_minus_p05",
        "wco2_k2_mean",
        "wco2_k2_std",
        "wco2_k2_p95_minus_p05",
        "corr_cld_dist_km_vs_o2a_k1",
        "corr_cld_dist_km_vs_o2a_k2",
        "corr_cld_dist_km_vs_wco2_k1",
        "corr_cld_dist_km_vs_wco2_k2",
    ]
    cols = [col for col in keep if col in summary.columns]
    lines = [
        "# Nassar Plume-Preservation Diagnostic",
        "",
        "This diagnostic is not a power-plant flux estimate.  It checks whether the",
        "cloud-proximity correction preserves real local XCO2 structure near selected",
        "plants.  Downwind fractions are only populated when a wind direction is",
        "provided with `--wind-from`, `--wind-to`, or `--wind-file`.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in summary[cols].itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def write_footprint_markdown(summary: pd.DataFrame, path: Path) -> None:
    keep = [
        "plant_id",
        "date",
        "radius_km",
        "subset",
        "fp",
        "n",
        "downwind_status",
        "frac_downwind_half_plane",
        "xco2_bc_max_minus_min",
        "deep_ensemble_corrected_xco2_max_minus_min",
        "pred_anomaly_p95_minus_p05",
        "cld_dist_km_median",
        "cld_dist_km_std",
        "cld_dist_km_p95_minus_p05",
        "cloud_distance_regime",
        "delta_wco2_k1_max_minus_min",
        "delta_wco2_k1_p95_minus_p05",
        "wco2_k2_mean",
        "wco2_k2_std",
        "o2a_k1_mean",
        "o2a_k1_std",
        "o2a_k2_mean",
        "o2a_k2_std",
    ]
    cols = [col for col in keep if col in summary.columns]
    lines = [
        "# Nassar Plume-Preservation by Footprint Number",
        "",
        "Rows split each case/radius/subset by OCO-2 footprint number `fp=0..7`.",
        "The spectral columns highlight footprint-dependent WCO2/O2A behavior.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in summary[cols].itertuples(index=False):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    parser.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair", action="append", type=parse_pair, default=[])
    parser.add_argument("--radius-km", nargs="+", type=float, default=[25.0, 50.0])
    parser.add_argument("--clear-cloud-km", type=float, default=20.0)
    parser.add_argument("--wind-from", action="append", type=parse_wind_spec, default=[])
    parser.add_argument("--wind-to", action="append", type=parse_wind_spec, default=[])
    parser.add_argument(
        "--wind-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV with plant_id,date and either wind_from_deg or wind_to_deg. "
            "Directions are degrees clockwise from north."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plants = read_plants(args.plants)
    pairs = args.pair or list(DEFAULT_PAIRS)
    winds = wind_maps(args)

    rows: list[dict[str, object]] = []
    fp_rows: list[dict[str, object]] = []
    footprint_dir = args.output_dir / "footprints"
    for pair in pairs:
        plant_id, date = pair
        if plant_id not in plants:
            raise SystemExit(f"Unknown plant_id {plant_id!r}; check {args.plants}")
        try:
            case_rows, case_fp_rows = analyze_pair(
                pair=pair,
                source=plants[plant_id],
                plot_base=args.plot_base,
                csv_dir=args.csv_dir,
                radii_km=args.radius_km,
                clear_cloud_km=args.clear_cloud_km,
                wind_to_deg=winds.get(pair),
                footprint_dir=footprint_dir,
            )
            rows.extend(case_rows)
            fp_rows.extend(case_fp_rows)
        except FileNotFoundError as exc:
            print(f"  SKIP missing plot data: {exc}", flush=True)

    if not rows:
        raise SystemExit("No plume-preservation rows were generated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    csv_path = args.output_dir / "nassar_plume_preservation_summary.csv"
    md_path = args.output_dir / "nassar_plume_preservation_summary.md"
    summary.to_csv(csv_path, index=False)
    write_markdown(summary, md_path)
    fp_csv_path = args.output_dir / "nassar_plume_preservation_by_footprint.csv"
    fp_md_path = args.output_dir / "nassar_plume_preservation_by_footprint.md"
    fp_summary = pd.DataFrame(fp_rows)
    fp_summary.to_csv(fp_csv_path, index=False)
    write_footprint_markdown(fp_summary, fp_md_path)
    print(f"Wrote {len(summary)} summary rows -> {csv_path}")
    print(f"Wrote markdown summary -> {md_path}")
    print(f"Wrote {len(fp_summary)} footprint-group rows -> {fp_csv_path}")
    print(f"Wrote footprint-group markdown -> {fp_md_path}")
    print(f"Wrote per-footprint tables -> {footprint_dir}")


if __name__ == "__main__":
    main()
