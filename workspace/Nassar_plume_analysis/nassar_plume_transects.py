#!/usr/bin/env python
"""Along-track transect figures for the Nassar plume-control cases.

One figure per (plant, date): xco2_bc and the corrected product, the model
correction mu, and cloud distance, all against along-track distance from the
plant's closest approach. This is the visual companion to the preservation
tables — real plume structure should be visible in xco2_bc AND survive in the
corrected trace, while mu should track cloud distance, not the plant.

Along-track coordinate: OCO-2 flies a near-polar orbit, so latitude is
monotone within a segment; x = (lat − lat_closest) × 111 km on the segment
containing the closest approach.

Outputs: <output-dir>/transects/nassar_transect_{plant}_{date}.png
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
from plot_style import (apply_manuscript_style, panel_label, XCO2_LABEL,
                        XCO2_BC_LABEL, XCO2_DE_LABEL)  # noqa: E402
from analyze_nassar_plume_preservation import (  # noqa: E402
    DEFAULT_PLANTS,
    DEFAULT_PLOT_BASE,
    haversine_km,
    parse_pair,
    read_plants,
)
from nassar_control_null import DEFAULT_PAIRS, SEGMENT_GAP_S  # noqa: E402

DEFAULT_OUTPUT_DIR = DEFAULT_PLOT_BASE / "plume_preservation"


def rolling(x: np.ndarray, y: np.ndarray, width_km: float = 5.0):
    """Centered rolling median of y vs x (km)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    med = np.array([np.nanmedian(ys[(xs >= xi - width_km / 2)
                                    & (xs <= xi + width_km / 2)]) for xi in xs])
    return xs, med


def transect_case(pair, plants, plot_base, outdir, max_km=100.0):
    plant_id, date = pair
    path = plot_base / f"combined_{date}" / "plot_data.parquet"
    if not path.exists():
        print(f"  SKIP {plant_id}:{date} — missing {path}")
        return
    src = plants[plant_id]
    df = pd.read_parquet(path, columns=[
        "time", "lat", "lon", "cld_dist_km", "xco2_bc",
        "deep_ensemble_corrected_xco2", "pred_anomaly", "fp"])
    df = df[np.isfinite(df["xco2_bc"])].sort_values("time").reset_index(drop=True)
    df["source_dist_km"] = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                                        float(src["lat"]), float(src["lon"]))

    # segment containing the closest approach
    seg_id = (df["time"].diff().abs() > SEGMENT_GAP_S).cumsum()
    best_seg = seg_id.iloc[df["source_dist_km"].idxmin()]
    seg = df[seg_id == best_seg].copy()
    seg = seg[seg["source_dist_km"] <= max_km]
    if len(seg) < 20:
        print(f"  SKIP {plant_id}:{date} — only {len(seg)} footprints "
              f"within {max_km} km on the overpass segment")
        return
    lat0 = seg.loc[seg["source_dist_km"].idxmin(), "lat"]
    seg["x_km"] = (seg["lat"] - lat0) * 111.0
    min_dist = float(seg["source_dist_km"].min())

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1]})
    ax = axes[0]
    x = seg["x_km"].to_numpy()
    ax.plot(x, seg["xco2_bc"], ".", color="C0", ms=3, alpha=0.4)
    ax.plot(x, seg["deep_ensemble_corrected_xco2"], ".", color="C3", ms=3, alpha=0.4)
    for col, c, lbl in (("xco2_bc", "C0", XCO2_BC_LABEL),
                        ("deep_ensemble_corrected_xco2", "C3", XCO2_DE_LABEL)):
        xs, med = rolling(x, seg[col].to_numpy())
        ax.plot(xs, med, "-", color=c, lw=1.8, label=f"{lbl} (5 km median)")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvspan(-10, 10, color="gold", alpha=0.15)
    ax.set_ylabel(f"{XCO2_LABEL} (ppm)")
    ax.legend()
    ax.set_title(f"{src['source_name']} — {date}   "
                 f"(closest approach {min_dist:.1f} km)")
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(x, seg["pred_anomaly"], ".", color="C2", ms=3, alpha=0.4)
    xs, med = rolling(x, seg["pred_anomaly"].to_numpy())
    ax.plot(xs, med, "-", color="C2", lw=1.8)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvspan(-10, 10, color="gold", alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("μ (ppm)")
    panel_label(ax, "(b)")

    ax = axes[2]
    ax.plot(x, seg["cld_dist_km"], ".", color="C7", ms=3, alpha=0.5)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvspan(-10, 10, color="gold", alpha=0.15)
    ax.set_ylabel("nearest-cloud distance (km)")
    ax.set_xlabel("along-track distance from closest approach (km)")
    panel_label(ax, "(c)")

    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = outdir / f"nassar_transect_{plant_id}_{date}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--max-km", type=float, default=100.0)
    args = ap.parse_args()

    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    plants = read_plants(args.plants)
    pairs = args.pair or list(DEFAULT_PAIRS)
    outdir = args.output_dir / "transects"
    outdir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        print(f"{pair[0]}:{pair[1]}", flush=True)
        transect_case(pair, plants, args.plot_base, outdir, max_km=args.max_km)


if __name__ == "__main__":
    main()
