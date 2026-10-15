#!/usr/bin/env python
"""Per-case MODIS maps plus a Mauceri-format panel for the Mauceri et al. plants.

Mauceri et al. (2023, AMT 16:1461, Fig. 11) and Mauceri et al. (2025, ESS,
Part 1, Fig. 10) show that their ML bias corrections keep the plumes of
Westar, Ghent, Sasan and Matimba.  For each overpass this script draws

  (a)-(d)  MODIS Aqua true colour (NASA GIBS) with the OCO-2 footprints
           coloured by original XCO2, DE-corrected XCO2, the correction mu and
           the nearest-cloud distance, as in plot_nassar_plumes.py
  (e)      our footprints in the format of the Mauceri panels: XCO2 anomaly
           (each series minus its own mean) against latitude (longitude for
           Westar) on the same axis range, original as black '+', corrected as
           red dots
  (f)      the plume-preservation numbers of this case and what Mauceri et al.
           reported for it

Run the standard plume scripts with --output-dir
<plot-base>/plume_preservation_mauceri first (nassar_k1_contrast.py,
nassar_control_null.py); missing results are shown as 'not tested'.

Usage:  PYTHONPATH=src:workspace python workspace/Nassar_plume_analysis/plot_mauceri_comparison.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import (apply_manuscript_style, panel_label, CMAPS,  # noqa: E402
                        XCO2_LABEL)
from plot_nassar_plumes import (DEFAULT_PLANTS, DEFAULT_PLOT_BASE,  # noqa: E402
                                centered_norm, circle_lon_lat,
                                aqua_time_lag_min, download_gibs_wms_rgb, finite_percentile,
                                modis_lag_note,
                                haversine_km, plant_extent, read_plants)

# (plant, date, x column, x range of the Mauceri panel, plume peak in that panel,
#  Mauceri reference text)
CASES = (
    ("westar", "2015-12-04", "lon", (-96.32, -95.90), -96.13,
     "Mauceri et al. (2023), Fig. 11a: enhancement of about 1.5 ppm near\n"
     "-96.13 deg E, retrieved and corrected agree closely."),
    ("ghent", "2015-08-13", "lat", (38.15, 39.35), 38.73,
     "Mauceri et al. (2023), Fig. 11b, and (2025, Part 1), Fig. 10b:\n"
     "spike of 4 to 6 ppm near 38.73 deg N, kept by both corrections."),
    ("sasan", "2014-10-23", "lat", (23.38, 24.58), 23.95,
     "Mauceri et al. (2023), Fig. 11c: broad plume peaking near 10 ppm\n"
     "at 23.98 deg N, retrieved and corrected agree closely."),
    ("matimba", "2018-11-18", "lat", (-24.27, -23.07), -23.85,
     "Mauceri et al. (2025, Part 1), Fig. 10a: enhancement of about\n"
     "2 ppm near -23.85 deg N (downwind of the plant), kept by the ML\n"
     "correction."),
)
SEGMENT_GAP_S = 60.0
# Published panels for the side-by-side version (Mauceri et al. 2023 is CC BY 4.0).
# plant -> (image in <results-dir>/mauceri_figs, pixel rows of the panel, citation,
#           y range of the published panel)
MAUCERI_PANELS = {
    "westar": ("mauceri2023_fig11.png", (0, 828),
               "Mauceri et al. (2023), AMT 16:1461, Fig. 11a (CC BY 4.0)", (-2.2, 2.7)),
    "ghent": ("mauceri2023_fig11.png", (900, 1725),
              "Mauceri et al. (2023), AMT 16:1461, Fig. 11b (CC BY 4.0)", (-13.0, 7.5)),
    "sasan": ("mauceri2023_fig11.png", (1800, 2625),
              "Mauceri et al. (2023), AMT 16:1461, Fig. 11c (CC BY 4.0)", (-5.0, 13.0)),
}
PEAK_HALF_KM, BG_MIN_KM = 8.0, 33.0     # peak window and background in panel (e)


def peak_distance_km(seg, xcol, xpeak):
    """Distance of every footprint from the plume peak of the Mauceri panel."""
    near = seg.iloc[(seg[xcol] - xpeak).abs().argsort()[:5]]
    return haversine_km(seg["lat"].to_numpy(), seg["lon"].to_numpy(),
                        float(near["lat"].median()), float(near["lon"].median()))


def peak_rise(seg, xcol, xpeak):
    """Median within PEAK_HALF_KM of the panel's plume peak minus the median
    farther than BG_MIN_KM, for the original and the corrected series."""
    d = peak_distance_km(seg, xcol, xpeak)
    pk, bg = seg[d <= PEAK_HALF_KM], seg[d >= BG_MIN_KM]
    out = {}
    for col in ("xco2_bc", "deep_ensemble_corrected_xco2"):
        out[col] = float(pk[col].median() - bg[col].median())
    return out, len(pk), len(bg)


def load_overpass(plot_base, date, plant, xcol, xrange):
    df = pd.read_parquet(plot_base / f"combined_{date}" / "plot_data.parquet",
                         columns=["time", "lat", "lon", "cld_dist_km", "xco2_bc",
                                  "deep_ensemble_corrected_xco2", "pred_anomaly"])
    df = df[np.isfinite(df["xco2_bc"])].sort_values("time").reset_index(drop=True)
    df["source_dist_km"] = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                                        float(plant["lat"]), float(plant["lon"]))
    seg_id = (df["time"].diff().abs() > SEGMENT_GAP_S).cumsum()
    seg = df[seg_id == seg_id.iloc[df["source_dist_km"].idxmin()]]
    seg = seg[seg[xcol].between(*xrange)].copy()
    return df, seg


def draw_map(ax, fig, bg_img, extent, view, col, cmap, plant, title, cbar_label,
             vmin=None, vmax=None, norm=None):
    if bg_img is not None:
        ax.imshow(bg_img, extent=extent, origin="upper", aspect="auto")
    kw = dict(c=view[col], s=10, cmap=cmap, alpha=0.85, linewidth=0,
              rasterized=True, zorder=3)
    kw.update(dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax))
    sc = ax.scatter(view["lon"], view["lat"], **kw)
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)
    ax.scatter([float(plant["lon"])], [float(plant["lat"])], marker="*", s=220,
               c="white", edgecolor="black", linewidth=1.2, zorder=7,
               label=plant["source_name"])
    for rr, ls in ((25.0, "-"), (50.0, "--")):
        lon_c, lat_c = circle_lon_lat(float(plant["lat"]), float(plant["lon"]), rr)
        ax.plot(lon_c, lat_c, color="black", lw=1.5, ls=ls, zorder=5,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white", alpha=0.6),
                              pe.Normal()])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lon (deg E)")
    ax.set_ylabel("Lat (deg N)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)


def case_text(plant_id, date, k1, ctrl, mauceri_text, peak):
    lines = ["This study (production deep ensemble, B11)", ""]
    rise, n_pk, n_bg = peak
    kept = rise["deep_ensemble_corrected_xco2"] / rise["xco2_bc"]
    lines += [f"plume peak in panel (e) (<={PEAK_HALF_KM:.0f} km of the peak, n={n_pk},",
              f"  minus beyond {BG_MIN_KM:.0f} km, n={n_bg})",
              f"  original   {rise['xco2_bc']:+.2f} ppm",
              f"  corrected  {rise['deep_ensemble_corrected_xco2']:+.2f} ppm   ({100 * kept:.0f} % kept)",
              ""]
    r = k1[(k1.plant_id == plant_id) & (k1.date == date)] if k1 is not None else []
    if len(r):
        r = r.iloc[0]
        after = r.d_xco2_ppm - r.d_mu_ppm
        lines += [f"plume rise (<=15 km minus 30-100 km)",
                  f"  original   {r.d_xco2_ppm:+.2f} +/- {r.d_xco2_se:.2f} ppm",
                  f"  corrected  {after:+.2f} ppm   ({100 * after / r.d_xco2_ppm:.0f} % kept)",
                  "",
                  "change in <l'> in the plume window",
                  f"  O2 A-band   {r.d_o2a_k1:+.4f} (expected  0)",
                  f"  weak CO2    {r.d_wco2_k1:+.4f} (expected {r.d_wco2_k1_expected:+.4f})",
                  f"  strong CO2  {r.d_sco2_k1:+.4f} (expected {r.d_sco2_k1_expected:+.4f})"]
    else:
        lines += ["plume-window test not applicable", "(no footprint within 15 km of the plant)"]
    c = ctrl[(ctrl.plant_id == plant_id) & (ctrl.date == date)] if ctrl is not None else []
    if len(c):
        c = c.iloc[0]
        lines += ["", "matched controls (same orbit, 25 km disks)",
                  f"  spread kept, plant      {c.plant_ratio_corr_over_bc:.2f}",
                  f"  spread kept, controls   {c.control_ratio_median:.2f} "
                  f"(5-95 %: {c.control_ratio_p05:.2f}-{c.control_ratio_p95:.2f})",
                  f"  plant percentile        p{c.plant_ratio_pctile_in_controls:.0f}"
                  f"  -> {c.verdict.replace('_', ' ')}"]
    lines += ["", "-" * 46, mauceri_text]
    return "\n".join(lines)


def plot_case(case, plants, plot_base, modis_dir, k1, ctrl, outdir, extent_km,
              published=None):
    """published = (image array, citation, y range) adds the Mauceri panel next to (e)."""
    plant_id, date, xcol, xrange, xpeak, mauceri_text = case
    plant = plants[plant_id]
    df, seg = load_overpass(plot_base, date, plant, xcol, xrange)
    extent = plant_extent(float(plant["lat"]), float(plant["lon"]), extent_km, pad_deg=0.05)
    view = df[df["lon"].between(extent[0], extent[1])
              & df["lat"].between(extent[2], extent[3])]
    bg_img = None
    try:
        bg_img = plt.imread(download_gibs_wms_rgb(
            date, extent, modis_dir / f"aqua_rgb_{date}_{plant_id}_{int(extent_km)}km.png"))
    except Exception as exc:
        print(f"  MODIS unavailable for {plant_id} {date}: {exc}")

    fig = plt.figure(figsize=(24, 11.5), constrained_layout=True)
    g = fig.add_gridspec(2, 4)
    vmin = min(finite_percentile(view[c], 2) for c in ("xco2_bc", "deep_ensemble_corrected_xco2"))
    vmax = max(finite_percentile(view[c], 98) for c in ("xco2_bc", "deep_ensemble_corrected_xco2"))
    maps = (("xco2_bc", f"Original {XCO2_LABEL}", CMAPS["xco2"], dict(vmin=vmin, vmax=vmax), f"{XCO2_LABEL} (ppm)"),
            ("deep_ensemble_corrected_xco2", f"DE-corrected {XCO2_LABEL}", CMAPS["xco2"], dict(vmin=vmin, vmax=vmax), f"{XCO2_LABEL} (ppm)"),
            ("pred_anomaly", "Correction μ", CMAPS["mu"], dict(norm=centered_norm(view["pred_anomaly"])), "μ (ppm)"),
            ("cld_dist_km", "Nearest-cloud distance (MODIS)", CMAPS["cld_dist"], dict(vmin=0, vmax=50), "km"))
    for i, (col, title, cmap, kw, lab) in enumerate(maps):
        ax = fig.add_subplot(g[0, i])
        draw_map(ax, fig, bg_img, extent, view, col, cmap, plant, title, lab, **kw)
        panel_label(ax, f"({'abcd'[i]})")

    # (e) Mauceri format
    if published is None:
        ax = fig.add_subplot(g[1, :2])
    else:
        gb = g[1, :].subgridspec(1, 3, width_ratios=[1.35, 1.35, 1.1])
        ax = fig.add_subplot(gb[0, 0])
    x = seg[xcol].to_numpy()
    ax.plot(x, seg["xco2_bc"] - seg["xco2_bc"].mean(), "+", color="k", ms=7, mew=1.1,
            label=f"OCO-2 B11 ({XCO2_LABEL} bias-corrected)")
    ax.plot(x, seg["deep_ensemble_corrected_xco2"] - seg["deep_ensemble_corrected_xco2"].mean(),
            ".", color="red", ms=6, label="OCO-2 corr. (this study, DE)")
    xp = float(plant["lon"] if xcol == "lon" else plant["lat"])
    ax.axvline(xp, color="0.4", lw=1, ls="--", label="plant " + ("longitude" if xcol == "lon" else "latitude"))
    in_pk = seg[xcol][peak_distance_km(seg, xcol, xpeak) <= PEAK_HALF_KM]
    ax.axvspan(in_pk.min(), in_pk.max(), color="gold", alpha=0.2,
               label=f"plume-peak window, {PEAK_HALF_KM:.0f} km "
                     f"({'f' if published is None else 'g'})")
    ax.set_xlim(*xrange)
    ax.set_xlabel("Longitude [°]" if xcol == "lon" else "Latitude [°]")
    ax.set_ylabel(f"{XCO2_LABEL} anomaly [ppm]")
    if published is None:
        ax.set_title(f"Lat: {float(plant['lat']):.2f}° | Long: {float(plant['lon']):.2f}°   "
                     f"(same format and axis range as Mauceri et al.; n = {len(seg)})", fontsize=11)
    else:
        ax.set_ylim(*published[2])
        ax.set_title(f"This study (B11, deep ensemble), n = {len(seg)}\n"
                     f"Lat: {float(plant['lat']):.2f}° | Long: {float(plant['lon']):.2f}°",
                     fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    panel_label(ax, "(e)")

    if published is not None:
        axp = fig.add_subplot(gb[0, 1])
        axp.imshow(published[0])
        axp.axis("off")
        axp.set_title(f"{published[1]}\n(OCO-2 = retrieved, OCO-2 corr. = their correction)",
                      fontsize=11)
        panel_label(axp, "(f)")

    # numbers
    ax = fig.add_subplot(g[1, 2:] if published is None else gb[0, 2])
    ax.axis("off")
    ax.text(0.01, 0.98, case_text(plant_id, date, k1, ctrl, mauceri_text,
                                  peak_rise(seg, xcol, xpeak)),
            transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=10.5 if published is None else 9.5, linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.6"))
    panel_label(ax, "(f)" if published is None else "(g)")

    dmin = float(df["source_dist_km"].min())
    t_close = datetime.fromtimestamp(float(df.loc[df["source_dist_km"].idxmin(), "time"]), tz=timezone.utc)
    note = "" if bg_img is None else modis_lag_note(
        aqua_time_lag_min(date, float(plant["lat"]), float(plant["lon"]), t_close, modis_dir))
    fig.suptitle(f"{plant['source_name']}  {date}  |  closest footprint {dmin:.1f} km  |  "
                 f"circles 25 / 50 km" + (f"\nNote: {note}" if note else ""), fontsize=14,
                 weight="bold")
    out = outdir / (f"mauceri_comparison_{plant_id}_{date}"
                    f"{'' if published is None else '_side_by_side'}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--results-dir", type=Path,
                    default=DEFAULT_PLOT_BASE / "plume_preservation_mauceri")
    ap.add_argument("--modis-dir", type=Path, default=DEFAULT_PLOT_BASE / "_modis_tiles")
    ap.add_argument("--extent-km", type=float, default=70.0)
    ap.add_argument("--side-by-side", nargs="*", default=["westar", "ghent", "sasan"],
                    choices=sorted(MAUCERI_PANELS),
                    help="plants that also get a version with the published Mauceri panel")
    args = ap.parse_args()

    apply_manuscript_style()
    plants = read_plants(args.plants)
    read = lambda name: (pd.read_csv(args.results_dir / name)
                         if (args.results_dir / name).exists() else None)
    k1, ctrl = read("nassar_k1_contrast.csv"), read("nassar_control_null.csv")
    outdir = args.results_dir / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        plot_case(case, plants, args.plot_base, args.modis_dir, k1, ctrl, outdir,
                  args.extent_km)
        if case[0] in args.side_by_side:
            fname, (r0, r1), cite, ylim = MAUCERI_PANELS[case[0]]
            img = plt.imread(args.results_dir / "mauceri_figs" / fname)[r0:r1]
            plot_case(case, plants, args.plot_base, args.modis_dir, k1, ctrl, outdir,
                      args.extent_km, published=(img, cite, ylim))


if __name__ == "__main__":
    main()
