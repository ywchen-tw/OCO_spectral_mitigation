#!/usr/bin/env python
"""Plot Nassar power-plant plume cases over MODIS true-color imagery.

This is the source-only analog of ``plot_corrected_xco2.py`` for the Nassar
plume-control workflow.  It reads the downloaded ``plot_data.parquet`` files,
marks the power-plant location, overlays OCO-2 footprints on a MODIS Aqua
true-color background from NASA GIBS, and writes one PNG per plant/date pair.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import (apply_manuscript_style, panel_label, CMAPS,  # noqa: E402
                        XCO2_LABEL, MEAN_L_LABEL)


DEFAULT_PLANTS = Path(__file__).with_name("nassar_power_plants.csv")
DEFAULT_PLOT_BASE = Path(
    "results/model_comparison/deep_ensemble/"
    "de_beta_nll_prof_reg_foldpca_o05l15_m5/nassar_plumes"
)
DEFAULT_CSV_DIR = Path("results/csv_collection")
DEFAULT_SCREEN = DEFAULT_PLOT_BASE / "nassar_plume_screen.csv"
DEFAULT_PRESERVATION_SUMMARY = (
    DEFAULT_PLOT_BASE / "plume_preservation" / "nassar_plume_preservation_summary.csv"
)
DEFAULT_PRESERVATION_BY_FOOTPRINT = (
    DEFAULT_PLOT_BASE / "plume_preservation" / "nassar_plume_preservation_by_footprint.csv"
)


def read_plants(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["plant_id"]: row for row in csv.DictReader(f)}


def normalize_date(value: str) -> str:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def parse_pair(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"([^:@,]+)[:@,]([0-9]{4}-?[0-9]{2}-?[0-9]{2})", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Pair {value!r} must look like plant_id:YYYY-MM-DD"
        )
    return match.group(1), normalize_date(match.group(2))


def load_pairs_from_screen(path: Path, min_within_25: int, min_within_50: int) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    mask = (df["n_within_25km"].fillna(0) >= min_within_25) | (
        df["n_within_50km"].fillna(0) >= min_within_50
    )
    view = df.loc[mask & (df["status"] == "ok"), ["plant_id", "date", "min_source_dist_km"]]
    view = view.sort_values(["date", "min_source_dist_km"])
    return [(row.plant_id, row.date) for row in view.itertuples(index=False)]


def haversine_km(lat: np.ndarray, lon: np.ndarray, src_lat: float, src_lon: float) -> np.ndarray:
    radius_km = 6371.0088
    lat1 = np.radians(lat.astype(float))
    lat2 = math.radians(src_lat)
    dlat = lat1 - lat2
    dlon = np.radians(lon.astype(float) - src_lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def plant_extent(lat: float, lon: float, radius_km: float, pad_deg: float) -> list[float]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.15))
    return [lon - dlon - pad_deg, lon + dlon + pad_deg, lat - dlat - pad_deg, lat + dlat + pad_deg]


def circle_lon_lat(lat: float, lon: float, radius_km: float, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    radius_earth_km = 6371.0088
    lat0 = math.radians(lat)
    lon0 = math.radians(lon)
    angular = radius_km / radius_earth_km
    bearings = np.linspace(0, 2 * np.pi, n)
    out_lat = np.arcsin(
        math.sin(lat0) * math.cos(angular)
        + math.cos(lat0) * math.sin(angular) * np.cos(bearings)
    )
    out_lon = lon0 + np.arctan2(
        np.sin(bearings) * math.sin(angular) * math.cos(lat0),
        math.cos(angular) - math.sin(lat0) * np.sin(out_lat),
    )
    return np.degrees(out_lon), np.degrees(out_lat)


def download_gibs_wms_rgb(date: str, extent: Iterable[float], output: Path, width: int = 1400) -> Path:
    """Fetch MODIS Aqua true-color from NASA GIBS WMS in EPSG:4326."""
    lon_min, lon_max, lat_min, lat_max = [float(v) for v in extent]
    aspect = max((lat_max - lat_min) / max(lon_max - lon_min, 1e-6), 0.35)
    height = int(np.clip(width * aspect, 700, 1800))
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": "MODIS_Aqua_CorrectedReflectance_TrueColor",
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
        "CRS": "EPSG:4326",
        "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "TIME": date,
    }
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?" + urllib.parse.urlencode(params)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return output
    with urllib.request.urlopen(url, timeout=60) as response:
        output.write_bytes(response.read())
    return output


def finite_percentile(values: pd.Series, q: float) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanpercentile(arr, q)) if len(arr) else np.nan


def centered_norm(values: pd.Series, pct: float = 98.0, floor: float = 0.25) -> mcolors.TwoSlopeNorm:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    limit = float(np.nanpercentile(np.abs(arr), pct)) if len(arr) else floor
    limit = max(limit, floor)
    return mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def percentile_limits(values: pd.Series, lo: float = 2.0, hi: float = 98.0) -> tuple[float | None, float | None]:
    vmin = finite_percentile(values, lo)
    vmax = finite_percentile(values, hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None
    return vmin, vmax


def add_spectral_params(df: pd.DataFrame, csv_dir: Path, date: str) -> pd.DataFrame:
    """Attach spectral fit parameters dropped by the DE plot-data bridge."""
    if {"o2a_k1", "wco2_k1"}.issubset(df.columns):
        return df

    feature_path = csv_dir / f"combined_{date}_all_orbits.parquet"
    if not feature_path.exists():
        print(f"  Warning: spectral feature parquet not found: {feature_path}", flush=True)
        return df

    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(feature_path).names)
    key_cols = [c for c in ("time", "lon", "lat") if c in df.columns and c in schema_cols]
    value_cols = [c for c in ("o2a_k1", "wco2_k1") if c in schema_cols]
    if len(key_cols) < 3 or len(value_cols) < 2:
        print(f"  Warning: cannot attach spectral columns from {feature_path}", flush=True)
        return df

    features = pd.read_parquet(feature_path, columns=key_cols + value_cols)
    merged = df.merge(features.drop_duplicates(key_cols), on=key_cols, how="left", sort=False)
    missing = {col: int(merged[col].isna().sum()) for col in value_cols if col in merged.columns}
    if any(count == len(merged) for count in missing.values()):
        print(f"  Warning: spectral merge produced all-NaN columns for {date}; check row keys.", flush=True)
    return merged


def fmt_stat(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  Warning: optional summary not found: {path}", flush=True)
        return None
    return pd.read_csv(path)


def select_summary_row(
    summary: pd.DataFrame | None,
    plant_id: str,
    date: str,
    radius_km: float,
    preferred_subset: str = "clear_cld_ge_20km",
) -> pd.Series | None:
    if summary is None:
        return None
    rows = summary[
        (summary["plant_id"] == plant_id)
        & (summary["date"].astype(str) == date)
        & np.isclose(summary["radius_km"].astype(float), radius_km)
    ]
    if rows.empty:
        return None
    preferred = rows[(rows["subset"] == preferred_subset) & (rows["n"].fillna(0) > 0)]
    if not preferred.empty:
        return preferred.iloc[0]
    all_rows = rows[rows["subset"] == "all"]
    if not all_rows.empty:
        return all_rows.iloc[0]
    return rows.iloc[0]


def select_footprint_subset(
    fp_summary: pd.DataFrame | None,
    plant_id: str,
    date: str,
) -> tuple[pd.DataFrame | None, str]:
    if fp_summary is None:
        return None, "footprint table unavailable"

    rows = fp_summary[(fp_summary["plant_id"] == plant_id) & (fp_summary["date"].astype(str) == date)]
    if rows.empty:
        return None, "footprint table unavailable"

    candidates = [
        (25.0, "clear_cld_ge_20km"),
        (50.0, "clear_cld_ge_20km"),
        (25.0, "all"),
        (50.0, "all"),
    ]
    for radius_km, subset in candidates:
        cand = rows[
            np.isclose(rows["radius_km"].astype(float), radius_km)
            & (rows["subset"] == subset)
        ].sort_values("fp")
        if int(cand["n"].fillna(0).sum()) > 0:
            return cand, f"fp means ({radius_km:g} km, {subset})"
    return None, "footprint table has no local footprints"


def build_info_box_text(
    plant_id: str,
    date: str,
    preservation_summary: pd.DataFrame | None,
    preservation_by_footprint: pd.DataFrame | None,
) -> str:
    lines = ["Plume diagnostic", ""]
    if preservation_summary is None:
        lines.extend(["Run plume-preservation", "analysis for stats."])
        return "\n".join(lines)

    for radius_km in (25.0, 50.0):
        row = select_summary_row(preservation_summary, plant_id, date, radius_km)
        if row is None:
            lines.extend([f"r <= {radius_km:g} km: unavailable", ""])
            continue
        subset = str(row["subset"])
        lines.append(f"r <= {radius_km:g} km  {subset}")
        lines.append(
            f"n={int(row['n'])}  downwind={row.get('downwind_status', 'NA')}"
        )
        lines.append(
            "XCO2 dmax="
            f"{fmt_stat(row.get('xco2_bc_max_minus_min'))}, "
            "p95-p5="
            f"{fmt_stat(row.get('xco2_bc_p95_minus_p05'))} ppm"
        )
        lines.append(
            "Corr dmax="
            f"{fmt_stat(row.get('deep_ensemble_corrected_xco2_max_minus_min'))}, "
            "p95-p5="
            f"{fmt_stat(row.get('deep_ensemble_corrected_xco2_p95_minus_p05'))}"
        )
        lines.append(
            "mu p95-p5="
            f"{fmt_stat(row.get('pred_anomaly_p95_minus_p05'))}, "
            "corr(x,mu)="
            f"{fmt_stat(row.get('corr_xco2_bc_vs_mu'))}"
        )
        lines.append(
            "cld med="
            f"{fmt_stat(row.get('cld_dist_km_median'))}, "
            "p95-p5="
            f"{fmt_stat(row.get('cld_dist_km_p95_minus_p05'))} km"
        )
        lines.append(f"cld regime={row.get('cloud_distance_regime', 'NA')}")
        lines.append(
            "corr(cld,k): "
            "O1="
            f"{fmt_stat(row.get('corr_cld_dist_km_vs_o2a_k1'), 2)} "
            "O2="
            f"{fmt_stat(row.get('corr_cld_dist_km_vs_o2a_k2'), 2)} "
            "W1="
            f"{fmt_stat(row.get('corr_cld_dist_km_vs_wco2_k1'), 2)} "
            "W2="
            f"{fmt_stat(row.get('corr_cld_dist_km_vs_wco2_k2'), 2)}"
        )
        lines.append(
            "dWCO2 k1 max="
            f"{fmt_stat(row.get('delta_wco2_k1_max_minus_min'))}, "
            "p95-p5="
            f"{fmt_stat(row.get('delta_wco2_k1_p95_minus_p05'))}"
        )
        lines.append(
            "WCO2 k2="
            f"{fmt_stat(row.get('wco2_k2_mean'))}+/-{fmt_stat(row.get('wco2_k2_std'))}"
        )
        lines.append(
            "O2A k1="
            f"{fmt_stat(row.get('o2a_k1_mean'))}+/-{fmt_stat(row.get('o2a_k1_std'))}"
        )
        lines.append(
            "O2A k2="
            f"{fmt_stat(row.get('o2a_k2_mean'))}+/-{fmt_stat(row.get('o2a_k2_std'))}"
        )
        lines.append("")

    fp_rows, fp_title = select_footprint_subset(preservation_by_footprint, plant_id, date)
    lines.append(fp_title)
    if fp_rows is not None:
        lines.append("fp n cldR  WCO2k2       O2Ak1        O2Ak2")
        for row in fp_rows.sort_values("fp").itertuples(index=False):
            lines.append(
                f"{int(row.fp):1d} {int(row.n):2d} "
                f"{fmt_stat(getattr(row, 'cld_dist_km_p95_minus_p05', np.nan), 2):>4} "
                f"{fmt_stat(row.wco2_k2_mean, 2):>4}+/-{fmt_stat(row.wco2_k2_std, 2):<4} "
                f"{fmt_stat(row.o2a_k1_mean, 3):>5}+/-{fmt_stat(row.o2a_k1_std, 2):<4} "
                f"{fmt_stat(row.o2a_k2_mean, 2):>4}+/-{fmt_stat(row.o2a_k2_std, 2):<4}"
            )
    return "\n".join(lines).rstrip()


def plot_pair(
    plant_id: str,
    date: str,
    plant: dict[str, str],
    plot_base: Path,
    csv_dir: Path,
    output_dir: Path,
    radius_km: float,
    extent_radius_km: float,
    modis_dir: Path,
    no_modis: bool,
    preservation_summary: pd.DataFrame | None,
    preservation_by_footprint: pd.DataFrame | None,
) -> Path:
    source_lat = float(plant["lat"])
    source_lon = float(plant["lon"])
    plot_data = plot_base / f"combined_{date}" / "plot_data.parquet"
    if not plot_data.exists():
        raise FileNotFoundError(plot_data)

    df = pd.read_parquet(plot_data)
    needed = {"lat", "lon", "xco2_bc", "deep_ensemble_corrected_xco2", "pred_anomaly"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{plot_data} missing columns: {sorted(missing)}")
    df = add_spectral_params(df, csv_dir, date)

    df = df.assign(
        source_dist_km=haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(), source_lat, source_lon)
    )
    extent = plant_extent(source_lat, source_lon, extent_radius_km, pad_deg=0.08)
    view = df[
        df["lon"].between(extent[0], extent[1])
        & df["lat"].between(extent[2], extent[3])
    ].copy()
    if view.empty:
        view = df.nsmallest(500, "source_dist_km").copy()

    xco2_cols = ["xco2_bc", "deep_ensemble_corrected_xco2"]
    vmin = min(finite_percentile(view[col], 2) for col in xco2_cols)
    vmax = max(finite_percentile(view[col], 98) for col in xco2_cols)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None
    mu_norm = centered_norm(view["pred_anomaly"])

    bg_img = None
    if not no_modis:
        bg_path = modis_dir / f"aqua_rgb_{date}_{plant_id}.png"
        try:
            download_gibs_wms_rgb(date, extent, bg_path)
            bg_img = plt.imread(bg_path)
        except Exception as exc:
            print(f"  Warning: MODIS background unavailable for {plant_id} {date}: {exc}", flush=True)

    o2a_k1_min, o2a_k1_max = percentile_limits(view["o2a_k1"]) if "o2a_k1" in view.columns else (None, None)
    wco2_k1_min, wco2_k1_max = percentile_limits(view["wco2_k1"]) if "wco2_k1" in view.columns else (None, None)

    fig = plt.figure(figsize=(27.0, 12.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.15])
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])],
        ]
    )
    info_ax = fig.add_subplot(grid[:, 3])
    xco2_ppm = f"{XCO2_LABEL} (ppm)"
    panels = [
        (f"Original {XCO2_LABEL} (bias-corrected)", "xco2_bc", CMAPS["xco2"], vmin, vmax, None, xco2_ppm),
        (f"DeepEns corrected {XCO2_LABEL}", "deep_ensemble_corrected_xco2", CMAPS["xco2"], vmin, vmax, None, xco2_ppm),
        ("Predicted correction μ", "pred_anomaly", CMAPS["mu"], None, None, mu_norm, "μ (ppm)"),
        ("Nearest-cloud distance", "cld_dist_km", CMAPS["cld_dist"], 0.0, 50.0, None, "km"),
        (f"{MEAN_L_LABEL} (O2A)", "o2a_k1", CMAPS["spec"], o2a_k1_min, o2a_k1_max, None, f"{MEAN_L_LABEL} (O2A)"),
        (f"{MEAN_L_LABEL} (WCO2)", "wco2_k1", CMAPS["spec"], wco2_k1_min, wco2_k1_max, None, f"{MEAN_L_LABEL} (WCO2)"),
    ]

    for i, (ax, (title, col, cmap, pmin, pmax, norm, cbar_label)) in enumerate(
            zip(axes.ravel(), panels)):
        if bg_img is not None:
            ax.imshow(bg_img, extent=[extent[0], extent[1], extent[2], extent[3]], origin="upper", aspect="auto")
        if col in view.columns:
            scatter_kwargs = {
                "c": view[col],
                "s": 11,
                "cmap": cmap,
                "alpha": 0.78,
                "linewidth": 0,
                "rasterized": True,
                "zorder": 3,
            }
            if norm is not None:
                scatter_kwargs["norm"] = norm
            else:
                scatter_kwargs["vmin"] = pmin
                scatter_kwargs["vmax"] = pmax
            sc = ax.scatter(view["lon"], view["lat"], **scatter_kwargs)
            fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.88)
        else:
            ax.text(
                0.5,
                0.5,
                f"{col} not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
            )
        ax.scatter(
            [source_lon],
            [source_lat],
            marker="*",
            s=220,
            c="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=7,
            label=plant["source_name"],
        )
        for rr, ls in [(25.0, "-"), (50.0, "--")]:
            lon_c, lat_c = circle_lon_lat(source_lat, source_lon, rr)
            ax.plot(
                lon_c,
                lat_c,
                color="black",
                lw=1.8,
                ls=ls,
                alpha=0.88,
                zorder=5,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white", alpha=0.62), pe.Normal()],
            )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        ax.set_xlabel("Lon (deg E)")
        ax.set_ylabel("Lat (deg N)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        panel_label(ax, f"({chr(ord('a') + i)})")

    info_ax.axis("off")
    info_ax.text(
        0.01,
        0.98,
        build_info_box_text(plant_id, date, preservation_summary, preservation_by_footprint),
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.4,
        family="monospace",
        linespacing=1.22,
        bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.94, "boxstyle": "round,pad=0.65"},
    )

    n25 = int((view["source_dist_km"] <= 25.0).sum())
    n50 = int((view["source_dist_km"] <= 50.0).sum())
    nclear25 = int(((view["source_dist_km"] <= 25.0) & (view.get("cld_dist_km", np.nan) >= 20.0)).sum())
    fig.suptitle(
        f"{plant['source_name']}  {date}  |  n<=25 km: {n25} ({nclear25} clear), n<=50 km: {n50}",
        fontsize=14,
        weight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"nassar_plume_{plant_id}_{date}.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    parser.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--pair", action="append", type=parse_pair, default=[])
    parser.add_argument("--min-within-25", type=int, default=1)
    parser.add_argument("--min-within-50", type=int, default=10)
    parser.add_argument("--source-radius-km", type=float, default=50.0)
    parser.add_argument("--extent-radius-km", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_BASE / "plots")
    parser.add_argument("--modis-dir", type=Path, default=DEFAULT_PLOT_BASE / "_modis_tiles")
    parser.add_argument("--preservation-summary", type=Path, default=DEFAULT_PRESERVATION_SUMMARY)
    parser.add_argument("--preservation-by-footprint", type=Path, default=DEFAULT_PRESERVATION_BY_FOOTPRINT)
    parser.add_argument("--no-modis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    plants = read_plants(args.plants)
    preservation_summary = load_optional_csv(args.preservation_summary)
    preservation_by_footprint = load_optional_csv(args.preservation_by_footprint)
    pairs = args.pair or load_pairs_from_screen(args.screen, args.min_within_25, args.min_within_50)
    if not pairs:
        raise SystemExit("No plant/date pairs selected.")

    seen = set()
    outputs = []
    for plant_id, date in pairs:
        key = (plant_id, date)
        if key in seen:
            continue
        seen.add(key)
        if plant_id not in plants:
            raise SystemExit(f"Unknown plant_id {plant_id!r}; check {args.plants}")
        try:
            out = plot_pair(
                plant_id,
                date,
                plants[plant_id],
                args.plot_base,
                args.csv_dir,
                args.output_dir,
                args.source_radius_km,
                args.extent_radius_km,
                args.modis_dir,
                args.no_modis,
                preservation_summary,
                preservation_by_footprint,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP missing plot_data: {exc}", flush=True)
            continue
        print(f"  saved -> {out}", flush=True)
        outputs.append(out)

    print(f"Wrote {len(outputs)} plot(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
