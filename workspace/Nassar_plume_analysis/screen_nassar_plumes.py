#!/usr/bin/env python
"""Screen processed OCO-2 dates against Nassar et al. power-plant locations.

The script can run before or after deep-ensemble plot data exists:

* If ``plot_data.parquet`` exists, it reports source proximity, cloud-distance
  coverage, XCO2, predicted mu, and corrected-XCO2 statistics.
* Otherwise it falls back to ``results/csv_collection/combined_<date>_all_orbits``
  and reports the geometry/cloud/source information for the selected dates.
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
DEFAULT_CSV_DIR = Path("results/csv_collection")
DEFAULT_PLOT_BASE = Path(
    "results/model_comparison/deep_ensemble/"
    "de_beta_nll_prof_reg_o05l15_m5/nassar_plumes"
)


def haversine_km(lat: np.ndarray, lon: np.ndarray, src_lat: float, src_lon: float) -> np.ndarray:
    radius_km = 6371.0
    lat1 = np.radians(lat.astype(float))
    lat2 = math.radians(src_lat)
    dlat = lat1 - lat2
    dlon = np.radians(lon.astype(float) - src_lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return None


def read_plant_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_subset(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    columns = pq.read_schema(path).names
    wanted = [
        "lat",
        "latitude",
        "lon",
        "longitude",
        "time",
        "sfc_type",
        "xco2_bc",
        "deep_ensemble_corrected_xco2",
        "pred_anomaly",
        "deep_ensemble_mu",
        "cld_dist_km",
        "o2a_k1",
        "wco2_k1",
        "sco2_k1",
    ]
    usecols = [c for c in wanted if c in columns]
    return pd.read_parquet(path, columns=usecols)


def normalize_date(raw: str) -> str:
    value = raw.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def source_plot_path(plot_base: Path, date: str) -> Path:
    return plot_base / f"combined_{date}" / "plot_data.parquet"


def source_feature_path(csv_dir: Path, date: str) -> Path:
    return csv_dir / f"combined_{date}_all_orbits.parquet"


def discover_plot_dates(plot_base: Path) -> list[str]:
    dates = []
    for path in sorted(plot_base.glob("combined_*/plot_data.parquet")):
        name = path.parent.name
        date = name.removeprefix("combined_")
        if len(date) >= 10:
            dates.append(date[:10])
    return sorted(set(dates))


def summarize_pair(
    plant: dict[str, str],
    date: str,
    csv_dir: Path,
    plot_base: Path,
    radius_km: float,
    clear_cloud_km: float,
) -> dict[str, object]:
    plant_id = plant["plant_id"]
    src_lat = float(plant["lat"])
    src_lon = float(plant["lon"])

    plot_path = source_plot_path(plot_base, date)
    feature_path = source_feature_path(csv_dir, date)
    data_path = plot_path if plot_path.exists() else feature_path

    out: dict[str, object] = {
        "plant_id": plant_id,
        "source_name": plant["source_name"],
        "date": date,
        "country": plant["country"],
        "src_lat": src_lat,
        "src_lon": src_lon,
        "data_stage": "plot_data" if plot_path.exists() else "feature",
        "data_path": str(data_path),
    }

    if not data_path.exists():
        out.update({"status": "missing", "n_total": 0})
        return out

    df = read_subset(data_path)
    lat_col = first_existing_column(df.columns, ["lat", "latitude"])
    lon_col = first_existing_column(df.columns, ["lon", "longitude"])
    if lat_col is None or lon_col is None:
        out.update({"status": "missing_lat_lon", "n_total": len(df)})
        return out

    distance = haversine_km(df[lat_col].to_numpy(), df[lon_col].to_numpy(), src_lat, src_lon)
    df = df.assign(source_dist_km=distance)
    near = df[df["source_dist_km"] <= radius_km].copy()
    clear = near
    if "cld_dist_km" in near.columns:
        clear = near[near["cld_dist_km"] >= clear_cloud_km].copy()

    out.update(
        {
            "status": "ok",
            "n_total": int(len(df)),
            "n_within_radius": int(len(near)),
            "n_clear_within_radius": int(len(clear)),
            "min_source_dist_km": float(np.nanmin(distance)) if len(distance) else np.nan,
            "p10_source_dist_km": float(np.nanpercentile(distance, 10)) if len(distance) else np.nan,
            "median_source_dist_km": float(np.nanmedian(distance)) if len(distance) else np.nan,
        }
    )

    for threshold in (5.0, 10.0, 25.0, 50.0):
        out[f"n_within_{int(threshold)}km"] = int(np.count_nonzero(distance <= threshold))

    if "cld_dist_km" in near.columns and len(near):
        out["median_cld_dist_near_km"] = float(np.nanmedian(near["cld_dist_km"]))
        out["min_cld_dist_near_km"] = float(np.nanmin(near["cld_dist_km"]))
    if "sfc_type" in near.columns and len(near):
        out["land_fraction_near"] = float(np.nanmean(near["sfc_type"].to_numpy() == 1))

    stat_df = clear if len(clear) else near
    stat_suffix = "clear" if len(clear) else "near"
    for col in ("xco2_bc", "pred_anomaly", "deep_ensemble_mu", "deep_ensemble_corrected_xco2"):
        if col in stat_df.columns and len(stat_df):
            values = stat_df[col].to_numpy(dtype=float)
            out[f"{col}_{stat_suffix}_mean"] = float(np.nanmean(values))
            out[f"{col}_{stat_suffix}_sd"] = float(np.nanstd(values))
            out[f"{col}_{stat_suffix}_p95_minus_p05"] = float(
                np.nanpercentile(values, 95) - np.nanpercentile(values, 5)
            )

    for col in ("o2a_k1", "wco2_k1", "sco2_k1"):
        if col in stat_df.columns and len(stat_df):
            values = stat_df[col].to_numpy(dtype=float)
            out[f"{col}_{stat_suffix}_p95_minus_p05"] = float(
                np.nanpercentile(values, 95) - np.nanpercentile(values, 5)
            )

    if len(clear) >= 20:
        out["screen_rank_note"] = "good_clear_source_coverage"
    elif len(near) >= 20:
        out["screen_rank_note"] = "near_source_but_cloudy_or_no_cloud_column"
    elif out.get("min_source_dist_km", np.inf) <= radius_km:
        out["screen_rank_note"] = "sparse_source_coverage"
    else:
        out["screen_rank_note"] = "no_source_coverage"
    return out


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    keep = [
        "plant_id",
        "date",
        "source_name",
        "data_stage",
        "status",
        "min_source_dist_km",
        "n_within_10km",
        "n_within_25km",
        "n_clear_within_radius",
        "median_cld_dist_near_km",
        "pred_anomaly_clear_mean",
        "xco2_bc_clear_p95_minus_p05",
        "screen_rank_note",
    ]
    cols = [c for c in keep if c in df.columns]
    view = df[cols].copy()
    for col in view.select_dtypes(include=[float]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")

    table_rows = []
    table_rows.append("| " + " | ".join(cols) + " |")
    table_rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in cols]
        table_rows.append("| " + " | ".join(values) + " |")

    lines = [
        "# Nassar Power-Plant Plume Screen",
        "",
        "Cases are ranked after processing by footprint/source proximity, clear-sky "
        "coverage, and correction inertness. `pred_anomaly` is the deep-ensemble mu.",
        "",
        "\n".join(table_rows),
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", "--cases", dest="plants", type=Path, default=DEFAULT_PLANTS)
    parser.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="Dates to screen, as YYYY-MM-DD or YYYYMMDD. If omitted, existing plot-base outputs are used.",
    )
    parser.add_argument(
        "--date-file",
        type=Path,
        default=None,
        help="Optional text file containing one YYYY-MM-DD or YYYYMMDD date per line.",
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT_BASE / "nassar_plume_screen.csv")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_PLOT_BASE / "nassar_plume_screen.md")
    parser.add_argument("--radius-km", type=float, default=150.0)
    parser.add_argument("--clear-cloud-km", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plants = read_plant_rows(args.plants)
    raw_dates = list(args.dates or [])
    if args.date_file is not None:
        raw_dates.extend(
            line.strip()
            for line in args.date_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if raw_dates:
        dates = sorted({normalize_date(date) for date in raw_dates})
    else:
        dates = discover_plot_dates(args.plot_base)

    if not dates:
        raise SystemExit(
            "No dates supplied and no existing plot_data dates found under "
            f"{args.plot_base}. Use --dates or --date-file."
        )

    summaries = [
        summarize_pair(plant, date, args.csv_dir, args.plot_base, args.radius_km, args.clear_cloud_km)
        for date in dates
        for plant in plants
    ]
    df = pd.DataFrame(summaries)
    sort_cols = [c for c in ["status", "date", "min_source_dist_km"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    write_markdown(df, args.markdown)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
