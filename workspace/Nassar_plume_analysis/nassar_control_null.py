#!/usr/bin/env python
"""Matched control-region null for the Nassar plume-preservation diagnostic.

The preservation summary shows the correction collapsing local XCO2 spread
near plants (corrected/original p95-p05 down to ~0.2 in near-cloud cases,
corr(xco2_bc, mu) ~ 0.99). By itself that number is ambiguous: it is either
the correction doing its job on cloud-driven scatter, or it is removing real
plume structure. This script builds the null: for each (plant, date) case it
samples control windows from the SAME date's swath — far from every catalog
plant, matched on cloud-distance regime — and asks whether the plant-disk
spread collapse is an outlier against those controls.

Reading the result:
  plant ratio ~ control distribution  -> collapse is generic smoothing (PASS:
                                         no evidence of plume removal)
  plant ratio << controls (low pctile)-> correction removes MORE structure at
                                         the plant than elsewhere (flag)

Outputs (under --output-dir, default <plot-base>/plume_preservation/):
  nassar_control_null.csv        one row per case
  nassar_control_null_windows.csv  every control window's metrics
  nassar_control_null.md         summary table
  nassar_control_null.png        control distributions + plant markers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import apply_manuscript_style, XCO2_LABEL  # noqa: E402
from analyze_nassar_plume_preservation import (  # noqa: E402
    DEFAULT_PLANTS,
    DEFAULT_PLOT_BASE,
    corr,
    haversine_km,
    parse_pair,
    read_plants,
    safe_ratio,
)

DEFAULT_OUTPUT_DIR = DEFAULT_PLOT_BASE / "plume_preservation"

# All screen cases with source coverage (near-source and/or clear).
DEFAULT_PAIRS = (
    ("lipetsk", "2015-08-01"),
    ("colstrip", "2015-08-01"),
    ("comanche", "2020-09-06"),
    ("kozienice", "2021-09-06"),
    ("taean", "2021-09-08"),
    ("westar", "2023-03-13"),
    ("westar", "2023-06-26"),
    ("vindhyachal", "2023-06-26"),
    ("ghent", "2024-04-15"),
    ("matimba", "2024-04-15"),
    ("kozienice", "2024-06-26"),
)

MIN_EXCLUSION_KM = 100.0     # control windows keep this distance from EVERY plant
SEGMENT_GAP_S = 30.0         # time gap that starts a new along-track segment


def region_metrics(sub: pd.DataFrame) -> dict[str, float]:
    """Spread/correlation metrics for one region (plant disk or control window)."""
    out: dict[str, float] = {"n": int(len(sub))}
    for col, tag in (("xco2_bc", "bc"), ("deep_ensemble_corrected_xco2", "corr"),
                     ("pred_anomaly", "mu")):
        arr = sub[col].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        out[f"{tag}_p95_p05"] = (float(np.percentile(arr, 95) - np.percentile(arr, 5))
                                 if len(arr) >= 3 else np.nan)
    out["ratio_corr_over_bc"] = safe_ratio(out["corr_p95_p05"], out["bc_p95_p05"])
    out["ratio_mu_over_bc"] = safe_ratio(out["mu_p95_p05"], out["bc_p95_p05"])
    out["corr_bc_vs_mu"] = corr(sub["xco2_bc"], sub["pred_anomaly"])
    out["median_cld_km"] = (float(np.nanmedian(sub["cld_dist_km"]))
                            if "cld_dist_km" in sub else np.nan)
    out["frac_near10"] = (float((sub["cld_dist_km"] < 10).mean())
                          if "cld_dist_km" in sub else np.nan)
    return out


def control_windows(df: pd.DataFrame, all_plants: pd.DataFrame,
                    window_km: float) -> list[pd.DataFrame]:
    """Slice the date's swath into contiguous along-track windows far from plants."""
    df = df.sort_values("time").reset_index(drop=True)

    # distance to nearest catalog plant, vectorized over the 14 plants
    min_plant = np.full(len(df), np.inf)
    for _, p in all_plants.iterrows():
        d = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                         float(p["lat"]), float(p["lon"]))
        min_plant = np.minimum(min_plant, d)
    df["_min_plant_km"] = min_plant

    seg = (df["time"].diff().abs() > SEGMENT_GAP_S).cumsum()
    windows = []
    lat_span = window_km / 111.0
    for _, sdf in df.groupby(seg):
        if len(sdf) < 10:
            continue
        lat0 = sdf["lat"].iloc[0]
        bin_idx = np.floor((sdf["lat"] - lat0).abs() / lat_span)
        for _, w in sdf.groupby(bin_idx):
            if w["_min_plant_km"].min() >= MIN_EXCLUSION_KM:
                windows.append(w)
    return windows


def match_controls(windows: list[pd.DataFrame], plant: dict[str, float],
                   min_n: int) -> list[dict[str, float]]:
    """Keep windows matched to the plant disk's cloud-distance regime."""
    med = plant["median_cld_km"]
    tol = max(3.0, 0.5 * med) if np.isfinite(med) else np.inf
    out = []
    for w in windows:
        if len(w) < min_n:
            continue
        m = region_metrics(w)
        if not np.isfinite(m["ratio_corr_over_bc"]):
            continue
        if np.isfinite(med) and abs(m["median_cld_km"] - med) > tol:
            continue
        m["lat_center"] = float(w["lat"].median())
        m["lon_center"] = float(w["lon"].median())
        out.append(m)
    return out


def analyze_case(pair, plants, plants_df, plot_base, radius_km):
    plant_id, date = pair
    path = plot_base / f"combined_{date}" / "plot_data.parquet"
    if not path.exists():
        print(f"  SKIP {plant_id}:{date} — missing {path}")
        return None, []
    src = plants[plant_id]
    df = pd.read_parquet(path, columns=[
        "time", "lat", "lon", "cld_dist_km", "xco2_bc",
        "deep_ensemble_corrected_xco2", "pred_anomaly"])
    df = df[np.isfinite(df["xco2_bc"])]

    dist = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                        float(src["lat"]), float(src["lon"]))
    disk = df[dist <= radius_km]
    if len(disk) < 20:
        print(f"  SKIP {plant_id}:{date} — only {len(disk)} footprints "
              f"within {radius_km} km")
        return None, []

    plant_m = region_metrics(disk)
    min_n = max(30, int(0.3 * plant_m["n"]))
    ctrls = match_controls(
        control_windows(df, plants_df, window_km=2 * radius_km), plant_m, min_n)

    row = {"plant_id": plant_id, "date": date, "radius_km": radius_km,
           **{f"plant_{k}": v for k, v in plant_m.items()},
           "n_controls": len(ctrls)}
    if ctrls:
        cr = np.array([c["ratio_corr_over_bc"] for c in ctrls])
        pr = plant_m["ratio_corr_over_bc"]
        row.update({
            "control_ratio_median": float(np.median(cr)),
            "control_ratio_p05": float(np.percentile(cr, 5)),
            "control_ratio_p95": float(np.percentile(cr, 95)),
            "plant_ratio_pctile_in_controls": float(np.mean(cr < pr) * 100.0),
            "verdict": ("plant_collapses_more" if np.mean(cr < pr) < 0.05
                        else "consistent_with_controls"),
        })
    else:
        row.update({"control_ratio_median": np.nan, "control_ratio_p05": np.nan,
                    "control_ratio_p95": np.nan,
                    "plant_ratio_pctile_in_controls": np.nan,
                    "verdict": "no_matched_controls"})
    ctrl_rows = [{"plant_id": plant_id, "date": date, "radius_km": radius_km, **c}
                 for c in ctrls]
    return row, ctrl_rows


def plot_null(summary: pd.DataFrame, ctrl: pd.DataFrame, path: Path) -> None:
    cases = summary.dropna(subset=["plant_ratio_pctile_in_controls"])
    if cases.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(cases)), 5))
    for i, row in enumerate(cases.itertuples(index=False)):
        cr = ctrl[(ctrl["plant_id"] == row.plant_id) & (ctrl["date"] == row.date)
                  & (ctrl["radius_km"] == row.radius_km)]["ratio_corr_over_bc"]
        x = np.full(len(cr), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(cr))
        ax.plot(x, cr, ".", color="gray", ms=3, alpha=0.5)
        ax.plot(i, row.plant_ratio_corr_over_bc, "r*", ms=15, zorder=5)
    labels = [f"{r.plant_id}\n{r.date}\n(p{r.plant_ratio_pctile_in_controls:.0f})"
              for r in cases.itertuples(index=False)]
    ax.set_xticks(range(len(cases)), labels, fontsize=7)
    ax.set_ylabel(f"corrected / original {XCO2_LABEL} spread (p95-p05 ratio)")
    ax.set_title("Plume-preservation null: plant disk (red star) vs cloud-matched\n"
                 "control windows (grey dots) from the same date, "
                 "$\\geq$100 km from any plant")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = ["plant_id", "date", "radius_km", "plant_n", "plant_median_cld_km",
            "plant_ratio_corr_over_bc", "n_controls", "control_ratio_median",
            "control_ratio_p05", "control_ratio_p95",
            "plant_ratio_pctile_in_controls", "verdict"]
    cols = [c for c in cols if c in summary.columns]
    lines = [
        "# Nassar Control-Region Null",
        "",
        "Is the spread collapse at each plant an outlier against cloud-matched",
        "control windows from the same date (≥100 km from every catalog plant)?",
        "`plant_ratio_pctile_in_controls` is the percentile of the plant-disk",
        "corrected/original spread ratio within the control distribution:",
        "LOW percentile = correction removes MORE structure at the plant than in",
        "plume-free scenes (possible plume erasure); mid-range = the collapse is",
        "generic near-cloud smoothing and the preservation concern is defused.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in summary[cols].itertuples(index=False):
        vals = [f"{v:.3g}" if isinstance(v, float) and np.isfinite(v)
                else ("" if isinstance(v, float) else str(v)) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--radius-km", nargs="+", type=float, default=[25.0])
    args = ap.parse_args()

    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    plants = read_plants(args.plants)
    plants_df = pd.read_csv(args.plants)
    pairs = args.pair or list(DEFAULT_PAIRS)

    rows, ctrl_rows = [], []
    for pair in pairs:
        for radius in args.radius_km:
            print(f"{pair[0]}:{pair[1]} r={radius:g} km", flush=True)
            row, ctrls = analyze_case(pair, plants, plants_df,
                                      args.plot_base, radius)
            if row is not None:
                rows.append(row)
                ctrl_rows.extend(ctrls)

    if not rows:
        raise SystemExit("No cases produced results.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    ctrl = pd.DataFrame(ctrl_rows)
    summary.to_csv(args.output_dir / "nassar_control_null.csv", index=False)
    ctrl.to_csv(args.output_dir / "nassar_control_null_windows.csv", index=False)
    write_markdown(summary, args.output_dir / "nassar_control_null.md")
    plot_null(summary, ctrl, args.output_dir / "nassar_control_null.png")
    print(f"Wrote {len(summary)} case rows, {len(ctrl)} control windows "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
