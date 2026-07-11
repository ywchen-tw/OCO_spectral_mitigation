#!/usr/bin/env python3
"""Manuscript Fig. 1a — OCO-2 / MODIS collocation + nearest-cloud-distance
geometry schematic, built from real pipeline output (test date 2018-10-18).

Main panel: Aqua-MODIS MYD35 Cloudy/Uncertain pixels (grey) with the OCO-2
glint track coloured by the Phase-4 nearest-cloud distance; one representative
sounding is annotated with an arrow to its nearest cloud pixel and the
distance labelled.  A small inset shows the whole granule track for context.

Usage (from repo root or anywhere):
    python3 workspace/manuscript_figures/make_collocation_schematic.py
    ... [--granule 22845a_GL] [--date 2018-10-18] [--annotate-sid SID]
    ... [--extent LON0 LON1 LAT0 LAT1] [--out results/figures/manuscript/...]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# workspace/ (plot_style) and src/ (pickle dataclasses) on the path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # workspace/
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import CMAPS, apply_manuscript_style, panel_label

# ----------------------------------------------------------------------------
# Legacy-pickle shim: granule pickles were written when the pipeline modules
# were named pipeline.phase_0X_*; alias them to the current step_0X_* modules.
import pipeline.step_03_processing as _s3  # noqa: E402
import pipeline.step_04_geometry as _s4    # noqa: E402

sys.modules.setdefault("pipeline.phase_03_processing", _s3)
sys.modules.setdefault("pipeline.phase_04_geometry", _s4)

KM_PER_DEG = 111.195


def load_granule(proc_dir: Path, granule: str):
    """Return (cloud dict with lon/lat/cloud_flag, list[CollocationResult])."""
    gdir = proc_dir / granule
    with open(gdir / f"granule_combined_{granule}.pkl", "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined = pickle.load(f)
    with open(gdir / "phase4_results.pkl", "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phase4 = pickle.load(f)
    return combined, phase4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="2018-10-18")
    p.add_argument("--granule", default="22845a_GL")
    p.add_argument("--extent", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                   default=[-170.9, -168.3, -29.2, -26.05],
                   help="zoom box [lon_min lon_max lat_min lat_max]")
    p.add_argument("--annotate-sid", type=int, default=2018101801004732,
                   help="sounding_id of the representative footprint")
    p.add_argument("--vmax", type=float, default=40.0, help="colorbar upper bound (km)")
    p.add_argument("--out", default=None,
                   help="output PNG path (default results/figures/manuscript/"
                        "fig01a_collocation_schematic.png); a .pdf twin is saved too")
    args = p.parse_args()

    dt = datetime.strptime(args.date, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    proc_dir = _REPO / "data" / "processing" / f"{dt.year}" / f"{doy:03d}"
    combined, phase4 = load_granule(proc_dir, args.granule)

    lon0, lon1, lat0, lat1 = args.extent
    clon, clat = combined["lon"], combined["lat"]
    cflag = combined["cloud_flag"]  # 0=Uncertain, 1=Cloudy

    fp_lon = np.array([r.footprint_lon for r in phase4])
    fp_lat = np.array([r.footprint_lat for r in phase4])
    fp_d = np.array([r.nearest_cloud_dist_km for r in phase4])

    apply_manuscript_style()

    fig, ax = plt.subplots(figsize=(5.6, 6.3))

    # --- MODIS cloud pixels (Cloudy darker, Uncertain lighter) ---------------
    box = (clon >= lon0) & (clon <= lon1) & (clat >= lat0) & (clat <= lat1)
    unc = box & (cflag == 0)
    cld = box & (cflag == 1)
    ax.scatter(clon[unc], clat[unc], s=0.6, c="0.85", marker="s",
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(clon[cld], clat[cld], s=0.6, c="#7e94ae", marker="s",
               linewidths=0, rasterized=True, zorder=1)

    # --- OCO-2 track coloured by nearest-cloud distance ----------------------
    fbox = (fp_lon >= lon0) & (fp_lon <= lon1) & (fp_lat >= lat0) & (fp_lat <= lat1)
    sc = ax.scatter(fp_lon[fbox], fp_lat[fbox], c=fp_d[fbox], s=7,
                    cmap=CMAPS["cld_dist"], vmin=0.0, vmax=args.vmax,
                    linewidths=0, rasterized=True, zorder=3)

    # --- representative footprint: arrow to its nearest cloud pixel ----------
    rep = next(r for r in phase4 if r.sounding_id == args.annotate_sid)
    ax.scatter([rep.footprint_lon], [rep.footprint_lat], s=55,
               facecolors="none", edgecolors="crimson", linewidths=1.3, zorder=5)
    ax.annotate("", xy=(rep.nearest_cloud_lon, rep.nearest_cloud_lat),
                xytext=(rep.footprint_lon, rep.footprint_lat),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5,
                                shrinkA=6, shrinkB=1), zorder=5)
    mx = 0.5 * (rep.footprint_lon + rep.nearest_cloud_lon)
    my = 0.5 * (rep.footprint_lat + rep.nearest_cloud_lat)
    ax.text(mx + 0.10, my + 0.07,
            f"$d$ = {rep.nearest_cloud_dist_km:.1f} km",
            color="crimson", fontsize=9, ha="left", va="center", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

    # --- protocol facts -------------------------------------------------------
    buffer_txt = (
        "Aqua-MODIS MYD35, $\\Delta t \\leq$ 10 min\n"
        "($\\leq$ 20 min from 2022, Aqua drift)\n"
        "Cloudy + Uncertain pixels retained\n"
        "$d$ capped at 50 km"
    )
    ax.text(0.03, 0.03, buffer_txt, transform=ax.transAxes, fontsize=7.5,
            ha="left", va="bottom", linespacing=1.45, zorder=6,
            bbox=dict(fc="white", ec="0.6", lw=0.6, alpha=0.9,
                      boxstyle="round,pad=0.45"))

    # --- 50-km scale bar (lower right) ---------------------------------------
    mid_lat = 0.5 * (lat0 + lat1)
    bar_deg = 50.0 / (KM_PER_DEG * np.cos(np.radians(mid_lat)))
    bx1 = lon1 - 0.12 * (lon1 - lon0)
    bx0 = bx1 - bar_deg
    by = lat0 + 0.035 * (lat1 - lat0)
    ax.plot([bx0, bx1], [by, by], color="k", lw=2.0, solid_capstyle="butt", zorder=6)
    ax.text(0.5 * (bx0 + bx1), by + 0.018 * (lat1 - lat0), "50 km",
            ha="center", va="bottom", fontsize=8, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))

    # --- axes cosmetics (aspect-locked lat/lon) -------------------------------
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / max(np.cos(np.radians(mid_lat)), 0.05))
    ax.set_xlabel("Longitude ($\\degree$E)")
    ax.set_ylabel("Latitude ($\\degree$N)")
    panel_label(ax, "(a)")
    ax.text(1.0, 1.02, f"OCO-2 orbit {args.granule.split('a')[0]} (glint), {args.date}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom")

    # legend for the grey cloud classes + track
    handles = [
        Line2D([], [], marker="s", ls="none", ms=5, mfc="#7e94ae", mec="none",
               label="MODIS Cloudy"),
        Line2D([], [], marker="s", ls="none", ms=5, mfc="0.85", mec="none",
               label="MODIS Uncertain"),
        Line2D([], [], marker="o", ls="none", ms=5,
               mfc=plt.get_cmap(CMAPS["cld_dist"])(0.55), mec="none",
               label="OCO-2 sounding"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5,
              framealpha=0.9, borderpad=0.6, handletextpad=0.5)

    # colorbar in an axes-divider cax so it tracks the aspect-locked map height
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.10)
    cb = fig.colorbar(sc, cax=cax, extend="max")
    cb.set_label("Nearest-cloud distance $d$ (km)")

    # --- inset: whole granule track for context ------------------------------
    axin = ax.inset_axes([0.015, 0.40, 0.30, 0.24])
    glon = np.mod(fp_lon, 360.0)  # granule crosses the antimeridian
    order = np.argsort(fp_lat)
    axin.plot(glon[order], fp_lat[order], color="0.45", lw=0.9)
    zlon = np.mod(0.5 * (lon0 + lon1), 360.0)
    from matplotlib.patches import Rectangle
    axin.add_patch(Rectangle((np.mod(lon0, 360.0), lat0), lon1 - lon0, lat1 - lat0,
                             fill=False, ec="crimson", lw=1.0))
    axin.set_xlim(zlon - 35, zlon + 35)
    axin.set_ylim(-70, 70)
    axin.set_xticks([])
    axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(0.6)
    axin.set_facecolor("white")
    axin.text(0.5, 0.04, "full granule track", transform=axin.transAxes,
              ha="center", va="bottom", fontsize=6.5, color="0.35")

    # --- save -----------------------------------------------------------------
    out_png = Path(args.out) if args.out else (
        _REPO / "results" / "figures" / "manuscript" / "fig01a_collocation_schematic.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_png.with_suffix('.pdf')}")
    print(f"annotated sounding {rep.sounding_id}: d = {rep.nearest_cloud_dist_km:.2f} km "
          f"({rep.cloud_classification} pixel)")


if __name__ == "__main__":
    main()
