#!/usr/bin/env python
"""Category atlas figures for the manuscript appendix.

One figure per shortlist category (good_land / good_ocean / false_positive /
cloud_no_bias), one column per case, four compact rows:
  1. zoomed Aqua true-color RGB (GIBS) with footprints colored by MODIS
     cloud distance + 5 km scale bar,
  2. delta k1 for the three bands (chosen footprint),
  3. continuum / clear ratio,
  4. absolute XCO2 BC vs the clear-scene median.
Yellow shading marks where the chosen footprint is within --shade-km of a
MODIS cloud. Each column is annotated with the peak |dXCO2| inside the
shaded window. Categories with more than --max-cols cases are split into
numbered pages.

Usage:
  python workspace/spec_case_study/spec_case_atlas.py \
      [--shortlist workspace/spec_case_study/shortlist_2026-07-08.csv] \
      [--category good_ocean ...] [--fmt pdf]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_case_figure import (  # noqa: E402
    BAND_COLORS,
    BANDS,
    DEFAULT_OUTDIR,
    DEFAULT_PARQUET,
    XCO2_LABEL,
    add_scalebar,
    contiguous_spans,
    fetch_rgb,
    load_case,
    xco2_bc_ylim,
)

DEFAULT_SHORTLIST = Path(__file__).resolve().parent / "shortlist_2026-07-08.csv"
CATEGORY_TITLES = {
    "good_land": "Small clouds over land",
    "good_ocean": "Small clouds over ocean (glint)",
    "false_positive": "MODIS cloud-mask false positives (surface features)",
    "cloud_no_bias": "Clouds without an apparent XCO2 bias",
}


def case_column(fig, gs_col, case_row, args, show_ylabels: bool) -> None:
    case = load_case(args.parquet_fname, case_row["date"],
                     frame=int(case_row["frame"]), fp=int(case_row["fp"]),
                     half_span_km=args.half_span_km, clear_km=args.clear_km)
    seg, center, sel = case["seg"], case["center"], case["sel"]
    fp_sel, t0 = case["fp_sel"], case["t0"]
    mine = seg[sel].sort_values("x_km")

    ax_rgb = fig.add_subplot(gs_col[0])
    ax_k1 = fig.add_subplot(gs_col[1])
    ax_cont = fig.add_subplot(gs_col[2], sharex=ax_k1)
    ax_xco2 = fig.add_subplot(gs_col[3], sharex=ax_k1)

    # row 1: zoom RGB
    lat0 = float(center["lat"].mean())
    coslat = max(np.cos(np.deg2rad(lat0)), 0.1)
    clat, clon = lat0, float(center["lon"].mean())
    zlat, zlon = args.zoom_km / 111.0, args.zoom_km / (111.0 * coslat)
    zbox = (clon - zlon, clat - zlat, clon + zlon, clat + zlat)
    try:
        img, extent = fetch_rgb(case_row["date"], zbox,
                                args.output_dir / "rgb_cache", args.img_width)
        ax_rgb.imshow(img, extent=extent, origin="upper", zorder=0)
    except Exception as exc:
        print(f"GIBS fetch failed ({exc}); plain panel.")
    ax_rgb.scatter(seg["lon"], seg["lat"], c=seg["cld_dist_km"],
                   cmap="viridis", vmin=0, vmax=args.clear_km, s=9,
                   edgecolors="k", linewidths=0.15, zorder=2)
    ax_rgb.scatter(seg.loc[sel, "lon"], seg.loc[sel, "lat"], facecolors="none",
                   edgecolors="r", s=24, linewidths=0.5, zorder=3)
    ax_rgb.set_xlim(zbox[0], zbox[2]); ax_rgb.set_ylim(zbox[1], zbox[3])
    ax_rgb.set_aspect(1.0 / coslat)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    add_scalebar(ax_rgb, zbox[0] + 0.10 * (zbox[2] - zbox[0]),
                 zbox[1] + 0.07 * (zbox[3] - zbox[1]), 5, coslat)
    tstr = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%H:%M")
    ax_rgb.set_title(f"{case_row['date']} {tstr}Z\n"
                     f"({clat:.1f}°, {clon:.1f}°)  fp {fp_sel}", fontsize=7)

    # row 2: delta k1
    for b in BANDS:
        ax_k1.plot(mine["x_km"], mine[f"d_{b}_k1"], "-", color=BAND_COLORS[b],
                   lw=0.9, label=b.upper())
    ax_k1.axhline(0, color="k", lw=0.4)

    # row 3: continuum ratio
    for b in BANDS:
        ax_cont.plot(mine["x_km"], mine[f"r_h_cont_{b}"], "-",
                     color=BAND_COLORS[b], lw=0.9)
    ax_cont.axhline(1, color="k", lw=0.4)

    # row 4: absolute XCO2 BC
    base_bc = mine.loc[mine["cld_dist_km"] >= args.clear_km,
                       "xco2_bc"].median()
    ax_xco2.plot(mine["x_km"], mine["xco2_bc"], "-o", ms=1.6, lw=0.9,
                 color="k")
    if np.isfinite(base_bc):
        ax_xco2.axhline(base_bc, color="0.45", lw=0.6, ls="--")
    ax_xco2.set_ylim(*xco2_bc_ylim(mine["xco2_bc"], base_bc))
    ax_xco2.set_xlabel("along-track (km)", fontsize=7)

    # near-cloud shading + peak |dXCO2| annotation inside it
    near = mine["cld_dist_km"] < args.shade_km
    spans = contiguous_spans(mine["x_km"].to_numpy(), near.to_numpy())
    for a in (ax_k1, ax_cont, ax_xco2):
        for x0, x1 in spans:
            a.axvspan(x0, x1, color="gold", alpha=0.18, zorder=0)
    if near.any() and np.isfinite(base_bc):
        dev = (mine.loc[near, "xco2_bc"] - base_bc)
        dev = dev[np.isfinite(dev)]
        if len(dev):
            peak = dev.iloc[np.argmax(np.abs(dev.to_numpy()))]
            ax_xco2.text(0.03, 0.05,
                         rf"peak $\Delta$: {peak:+.1f} ppm",
                         transform=ax_xco2.transAxes, fontsize=6.5, va="bottom",
                         bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))

    for a in (ax_k1, ax_cont):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (ax_k1, ax_cont, ax_xco2):
        a.tick_params(labelsize=6.5)
    if show_ylabels:
        ax_k1.set_ylabel(r"$\Delta k_1$", fontsize=8)
        ax_cont.set_ylabel("cont./clear", fontsize=8)
        ax_xco2.set_ylabel(f"{XCO2_LABEL} BC (ppm)", fontsize=8)
    return ax_k1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    ap.add_argument("--category", action="append", default=None,
                    help="Restrict to these categories (default: all).")
    ap.add_argument("--max-cols", type=int, default=4)
    ap.add_argument("--half-span-km", type=float, default=50.0)
    ap.add_argument("--clear-km", type=float, default=20.0)
    ap.add_argument("--shade-km", type=float, default=10.0)
    ap.add_argument("--zoom-km", type=float, default=15.0)
    ap.add_argument("--img-width", type=int, default=800)
    ap.add_argument("--fmt", default="png", choices=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--parquet-fname", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    shortlist = pd.read_csv(args.shortlist, dtype={"date": str})
    cats = args.category or list(dict.fromkeys(shortlist["category"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cat in cats:
        rows = shortlist[shortlist["category"] == cat].reset_index(drop=True)
        if not len(rows):
            print(f"No cases in category {cat}"); continue
        pages = [rows.iloc[i:i + args.max_cols]
                 for i in range(0, len(rows), args.max_cols)]
        for ipage, page in enumerate(pages, start=1):
            ncol = len(page)
            fig = plt.figure(figsize=(2.4 * ncol + 0.9, 6.4))
            outer = fig.add_gridspec(1, ncol, wspace=0.30)
            legend_ax = None
            for i, (_, r) in enumerate(page.iterrows()):
                gs_col = outer[i].subgridspec(
                    4, 1, height_ratios=[1.7, 1, 1, 1], hspace=0.24)
                ax_k1 = case_column(fig, gs_col, r, args, show_ylabels=(i == 0))
                legend_ax = legend_ax or ax_k1
            if legend_ax is not None:
                legend_ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                                 fontsize=6.5, ncol=3, frameon=False,
                                 borderaxespad=0.0)
            suffix = f"_{ipage}" if len(pages) > 1 else ""
            title = CATEGORY_TITLES.get(cat, cat)
            fig.suptitle(title + (f" ({ipage}/{len(pages)})"
                                  if len(pages) > 1 else ""),
                         fontsize=10, y=1.005)
            out = args.output_dir / f"atlas_{cat}{suffix}.{args.fmt}"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
