#!/usr/bin/env python
"""Single-case spec-feature figure: MODIS RGB context + along-track traces.

For one frame picked by screen_spec_cases.py, renders:
  - Aqua-MODIS true-color RGB (NASA GIBS WMS, no login; the daily composite
    at the overpass location IS the A-Train granule minutes from OCO-2) with
    the 8-footprint track overlaid, colored by MODIS cloud distance
    (overview + zoom);
  - along-track traces for the chosen footprint (bold; other footprints as
    faint dots): delta k1, delta k2, delta (exp_intercept - albedo),
    continuum-radiance ratio, delta band albedo (surface-uniformity
    control), absolute XCO2 BC, and per-footprint cloud distance;
  - optionally (--with-heatmap) a footprint x along-track heatmap of the
    cross-band delta-k1 z-index: per band, delta k1 divided by the
    clear-scene std of that band, averaged over the three bands.

All deltas are vs the footprint's own clear-scene baseline (median over
cld_dist >= --clear-km inside the plotted segment), so the three bands
share axes and all-band coherence is visible at a glance.

Usage (after screening):
  python workspace/spec_case_study/spec_case_figure.py \
      --date 2018-07-21 --frame 201807211234567 [--fp 2]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import (apply_manuscript_style, panel_label, CMAPS,  # noqa: E402
                        XCO2_LABEL, MEAN_L_LABEL, VAR_L_LABEL)

DEFAULT_PARQUET = Path("results/csv_collection/combined_2016_2020_dates.parquet")
DEFAULT_OUTDIR = Path("results/figures/cld_dist_analysis/spec_case_study")

GROUND_SPEED_KMPS = 6.74          # OCO-2 along-track ground speed
BANDS = ("o2a", "wco2", "sco2")
# Okabe-Ito: CVD-safe (all pairs dE>37 under deutan/protan simulation)
BAND_COLORS = {"o2a": "#0072B2", "wco2": "#009E73", "sco2": "#D55E00"}
GIBS_WMS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
            "?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
            "&SRS=EPSG:4326&BBOX={w},{s},{e},{n}&WIDTH={px_w}&HEIGHT={px_h}"
            "&FORMAT=image/png&TIME={date}")
GIBS_LAYER = "MODIS_Aqua_CorrectedReflectance_TrueColor"

LOAD_COLS = (["date", "time", "fp", "fp_id", "lat", "lon", "cld_dist_km",
              "sfc_type", "xco2_qf", "xco2_bc", "xco2_raw_minus_apriori",
              "xco2_bc_anomaly"]
             + [f"{b}_k1" for b in BANDS] + [f"{b}_k2" for b in BANDS]
             + [f"{b}_exp_intercept-alb" for b in BANDS]
             + [f"h_cont_{b}" for b in BANDS] + [f"alb_{b}" for b in BANDS])


def fetch_rgb(date: str, bbox: tuple[float, float, float, float],
              cache_dir: Path, img_width: int) -> tuple[np.ndarray, tuple]:
    """Fetch (and cache) the GIBS true-color PNG for bbox=(w, s, e, n)."""
    w, s, e, n = bbox
    px_w = img_width
    px_h = max(64, min(2048, int(round(px_w * (n - s) / max(e - w, 1e-6)))))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / (f"gibs_{date}_{w:.3f}_{s:.3f}_{e:.3f}_{n:.3f}"
                         f"_{px_w}.png")
    if not fname.exists():
        url = GIBS_WMS.format(layer=GIBS_LAYER, w=w, s=s, e=e, n=n,
                              px_w=px_w, px_h=px_h, date=date)
        print(f"Fetching GIBS RGB -> {fname.name}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            fname.write_bytes(resp.read())
    return mpimg.imread(fname), (w, e, s, n)


def clear_baseline(seg: pd.DataFrame, var: str, clear_km: float) -> pd.Series:
    """Per-footprint clear-scene median of `var`; fallback all-fp clear."""
    clear = seg[seg["cld_dist_km"] >= clear_km]
    base = clear.groupby("fp")[var].median()
    fallback = clear[var].median()
    if not np.isfinite(fallback):
        fallback = seg[var].median()
    return seg["fp"].map(base).fillna(fallback)


def contiguous_spans(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    spans, start = [], None
    for xi, mi in zip(x, mask):
        if mi and start is None:
            start = xi
        elif not mi and start is not None:
            spans.append((start, xi))
            start = None
    if start is not None:
        spans.append((start, x[-1]))
    return spans


def load_case(parquet: Path, date: str, frame: int | None = None,
              frame_time: float | None = None, fp: int | None = None,
              half_span_km: float = 60.0, clear_km: float = 20.0) -> dict:
    """Load one case segment and attach deltas/ratios/z-index.

    Returns dict with seg (with x_km, d_*, r_h_cont_*, zk1_index columns),
    center frame rows, t0, fp_sel, sel (bool mask for fp_sel).
    """
    tbl = pq.read_table(parquet, columns=LOAD_COLS,
                        filters=[("date", "==", date.encode())])
    df = tbl.to_pandas().sort_values("time").reset_index(drop=True)
    if not len(df):
        raise SystemExit(f"No rows for date {date}")

    if frame is not None:
        center = df[df["fp_id"] // 10 == frame]
        if not len(center):
            raise SystemExit(f"Frame {frame} not found on {date}")
        t0 = float(center["time"].mean())
    else:
        t0 = float(frame_time)
        center = df[(df["time"] - t0).abs() < 0.4]

    half_s = half_span_km / GROUND_SPEED_KMPS
    seg = df[(df["time"] - t0).abs() <= half_s].copy()
    seg["x_km"] = (seg["time"] - t0) * GROUND_SPEED_KMPS
    fp_sel = (fp if fp is not None
              else int(center.loc[center["cld_dist_km"].idxmin(), "fp"]))

    trace_vars = ([f"{b}_k1" for b in BANDS] + [f"{b}_k2" for b in BANDS]
                  + [f"{b}_exp_intercept-alb" for b in BANDS]
                  + [f"alb_{b}" for b in BANDS])
    for var in trace_vars:
        seg[f"d_{var}"] = seg[var] - clear_baseline(seg, var, clear_km)
    for b in BANDS:  # continuum radiance as ratio to clear level
        seg[f"r_h_cont_{b}"] = seg[f"h_cont_{b}"] / clear_baseline(
            seg, f"h_cont_{b}", clear_km)

    # cross-band delta-k1 z-index (per band: d_k1 / clear-scene std; mean of 3)
    clear_rows = seg["cld_dist_km"] >= clear_km
    zcols = []
    for b in BANDS:
        sd = seg.loc[clear_rows, f"d_{b}_k1"].std()
        seg[f"z_{b}_k1"] = seg[f"d_{b}_k1"] / sd if sd > 0 else np.nan
        zcols.append(f"z_{b}_k1")
    seg["zk1_index"] = seg[zcols].mean(axis=1)

    return dict(seg=seg, center=center, t0=t0, fp_sel=fp_sel,
                sel=seg["fp"] == fp_sel)


def xco2_bc_ylim(mine_bc: pd.Series, base_bc: float,
                 max_dev_ppm: float = 8.0) -> tuple[float, float]:
    """Robust y-limits: follow the chosen footprint but clamp blown-up
    retrievals to +/- max_dev_ppm around the clear median."""
    vals = mine_bc[np.isfinite(mine_bc)]
    med = base_bc if np.isfinite(base_bc) else float(vals.median())
    lo = max(float(vals.min()), med - max_dev_ppm)
    hi = min(float(vals.max()), med + max_dev_ppm)
    pad = 0.12 * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def add_scalebar(ax, lon0, lat0, km, coslat, color="w", fontsize=7):
    dlon = km / (111.0 * coslat)
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=color, lw=2,
            solid_capstyle="butt")
    ax.text(lon0 + dlon / 2, lat0, f" {km:.0f} km ", color=color, fontsize=fontsize,
            ha="center", va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--frame", type=int, default=None,
                    help="Frame ID (fp_id // 10) from the screener CSV.")
    ap.add_argument("--frame-time", type=float, default=None,
                    help="Alternative to --frame: epoch time of the frame.")
    ap.add_argument("--fp", type=int, default=None,
                    help="Footprint (0-7) for the bold traces "
                         "(default: nearest-to-cloud fp of the center frame).")
    ap.add_argument("--half-span-km", type=float, default=60.0)
    ap.add_argument("--clear-km", type=float, default=20.0)
    ap.add_argument("--shade-km", type=float, default=10.0)
    ap.add_argument("--pad-km", type=float, default=10.0,
                    help="Map padding around the footprints.")
    ap.add_argument("--zoom-km", type=float, default=15.0,
                    help="Half-span of the zoomed RGB panel around the case "
                         "frame (small clouds are invisible at segment scale).")
    ap.add_argument("--img-width", type=int, default=1200)
    ap.add_argument("--no-rgb", action="store_true",
                    help="Skip the GIBS fetch (plain map panel).")
    ap.add_argument("--with-heatmap", action="store_true",
                    help="Include the fp x track cross-band dk1 z-index "
                         "heatmap (exploration aid, off for journal figures).")
    ap.add_argument("--journal", action="store_true",
                    help="Print-size layout (2026-10-11): 6.3 in wide x 7.9 in tall, drawn "
                         "at the printed width so fonts print at native size (axis labels "
                         "10 pt, ticks/legends 9 pt, panel letters 10 pt). Writes PNG + PDF. "
                         "Not combinable with --with-heatmap.")
    ap.add_argument("--slides", action="store_true",
                    help="16:9 slide layout (2026-10-12): maps on the left, five trace panels "
                         "on the right (delta <l'>, delta var(l'), continuum / clear, XCO2 B11, "
                         "cloud distance), 13-14 pt text. Writes PNG + PDF. Not combinable "
                         "with --journal or --with-heatmap.")
    ap.add_argument("--out-stem", default=None,
                    help="Output file stem (default case_<date>_<frame>_fp<fp>).")
    ap.add_argument("--rgb-cache", type=Path, default=None,
                    help="GIBS tile cache (default <output-dir>/rgb_cache).")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--parquet-fname", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    if args.frame is None and args.frame_time is None:
        ap.error("give --frame or --frame-time")
    if (args.journal or args.slides) and args.with_heatmap:
        ap.error("--journal and --slides do not support --with-heatmap")
    if args.journal and args.slides:
        ap.error("give only one of --journal and --slides")
    if args.rgb_cache is None:
        args.rgb_cache = args.output_dir / "rgb_cache"
    J, S = args.journal, args.slides
    # J and S share the compact styling (horizontal y labels, B11 label, PNG + PDF)
    C = J or S
    # font sizes: journal layout, slide layout, or the original 9.5 in exploration layout
    if J:
        FS = dict(tick=9, label=10, ylab=10, leg=9, letter=10, title=9.5, scale=8.5)
    elif S:
        FS = dict(tick=13, label=14, ylab=14, leg=13, letter=14, title=14, scale=12)
    else:
        FS = dict(tick=None, label=None, ylab=9, leg=7, letter=9, title=10, scale=7)
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi

    print(f"Loading {args.date} from {args.parquet_fname} ...")
    case = load_case(args.parquet_fname, args.date, frame=args.frame,
                     frame_time=args.frame_time, fp=args.fp,
                     half_span_km=args.half_span_km, clear_km=args.clear_km)
    seg, center, t0 = case["seg"], case["center"], case["t0"]
    fp_sel, sel = case["fp_sel"], case["sel"]
    print(f"Segment: {len(seg)} soundings, fp={fp_sel}, "
          f"center ({center['lat'].mean():.2f}, {center['lon'].mean():.2f})")

    # ── figure ────────────────────────────────────────────────────────────
    # trace panels, top to bottom; the slide layout drops the two control panels
    keys = (["k1", "k2", "cont", "xco2", "dist"] if S
            else ["k1", "k2", "gam", "cont", "alb", "xco2", "dist"])
    n_tr = len(keys)
    n_heat = 1 if args.with_heatmap else 0
    if S:
        # maps on the left with a horizontal colorbar below them; traces on the right
        # with a thin spacer row before the XCO2 panel for its above-the-box legend
        fig = plt.figure(figsize=(13.33, 7.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.32,
                              left=0.045, right=0.985, top=0.93, bottom=0.09)
        gs_maps = gs[0].subgridspec(1, 2, wspace=0.75, width_ratios=[0.5, 1.0])
        ax_map = fig.add_subplot(gs_maps[0])
        ax_zoom = fig.add_subplot(gs_maps[1])
        ax_heat = None
        i_x = keys.index("xco2")
        gs_tr = gs[1].subgridspec(n_tr + 1, 1, hspace=0.14,
                                  height_ratios=[1.0] * i_x + [0.25]
                                  + [1.0] * (n_tr - i_x))
        ax0 = fig.add_subplot(gs_tr[0])
        axes = [ax0] + [fig.add_subplot(gs_tr[i if i < i_x else i + 1], sharex=ax0)
                        for i in range(1, n_tr)]
    elif J:
        # maps on top; seven shared-x trace panels, packed, with one thin spacer row
        # before the XCO2 panel for its above-the-box legend
        # trace y labels are horizontal in a left column (vertical ones collide at
        # this panel height), hence the wide left margin
        fig = plt.figure(figsize=(6.3, 7.9))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 5.25], hspace=0.2,
                              left=0.2, right=0.985, top=0.95, bottom=0.07)
        gs_maps = gs[0].subgridspec(1, 2, wspace=0.45, width_ratios=[0.5, 1.0])
        ax_map = fig.add_subplot(gs_maps[0])
        ax_zoom = fig.add_subplot(gs_maps[1])
        ax_heat = None
        gs_tr = gs[1].subgridspec(n_tr + 1, 1, hspace=0.13,
                                  height_ratios=[1.0] * 5 + [0.22] + [1.0] * 2)
        ax0 = fig.add_subplot(gs_tr[0])
        axes = [ax0] + [fig.add_subplot(gs_tr[i if i < 5 else i + 1], sharex=ax0)
                        for i in range(1, n_tr)]
    else:
        fig = plt.figure(figsize=(9.5, 19.5 + 1.5 * n_heat))
        gs = fig.add_gridspec(1 + n_heat + n_tr, 1,
                              height_ratios=[4.2] + [1.1] * n_heat + [1.0] * n_tr,
                              hspace=0.34)
        gs_maps = gs[0].subgridspec(1, 2, wspace=0.18)
        ax_map = fig.add_subplot(gs_maps[0])
        ax_zoom = fig.add_subplot(gs_maps[1])
        ax_heat = fig.add_subplot(gs[1]) if args.with_heatmap else None
        ax0 = ax_heat if ax_heat is not None else fig.add_subplot(gs[1 + n_heat])
        axes = ([fig.add_subplot(gs[1 + n_heat + i], sharex=ax0)
                 for i in range(1, n_tr)] if ax_heat is None
                else [fig.add_subplot(gs[2 + i], sharex=ax0) for i in range(n_tr)])
        if ax_heat is None:
            axes = [ax0] + axes
    ax_by = dict(zip(keys, axes))

    # map + RGB (overview of the segment; zoom around the case frame)
    lat0 = float(seg["lat"].mean())
    coslat = max(np.cos(np.deg2rad(lat0)), 0.1)
    pad_lat, pad_lon = args.pad_km / 111.0, args.pad_km / (111.0 * coslat)
    bbox = (float(seg["lon"].min() - pad_lon), float(seg["lat"].min() - pad_lat),
            float(seg["lon"].max() + pad_lon), float(seg["lat"].max() + pad_lat))
    clat, clon = float(center["lat"].mean()), float(center["lon"].mean())
    zlat, zlon = args.zoom_km / 111.0, args.zoom_km / (111.0 * coslat)
    zbox = (clon - zlon, clat - zlat, clon + zlon, clat + zlat)

    def map_panel(ax, box, s_pts, title, scalebar=None, fontsize=10):
        if not args.no_rgb:
            try:
                img, extent = fetch_rgb(args.date, box, args.rgb_cache,
                                        args.img_width)
                ax.imshow(img, extent=extent, origin="upper", zorder=0)
            except Exception as exc:  # keep the figure usable offline
                print(f"GIBS fetch failed ({exc}); plain map panel.")
        # plasma over GIBS RGB: bright low end separates from dark ocean/land
        sc = ax.scatter(seg["lon"], seg["lat"], c=seg["cld_dist_km"],
                        cmap=CMAPS["spec"], vmin=0, vmax=args.clear_km, s=s_pts,
                        edgecolors="k", linewidths=0.2, zorder=2)
        ax.scatter(seg.loc[sel, "lon"], seg.loc[sel, "lat"], facecolors="none",
                   edgecolors="r", s=s_pts * 2.8, linewidths=0.7, zorder=3,
                   label=f"fp {fp_sel}")
        ax.scatter(clon, clat, marker="*", s=90 if J else 180, color="r",
                   edgecolors="w", zorder=4, label="case frame")
        ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
        ax.set_aspect(1.0 / coslat)
        ax.set_xlabel("lon", fontsize=FS["label"]); ax.set_ylabel("lat", fontsize=FS["label"])
        if C:
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=FS["tick"])
        ax.set_title(title, fontsize=fontsize)
        if scalebar:
            add_scalebar(ax, box[0] + 0.12 * (box[2] - box[0]),
                         box[1] + 0.06 * (box[3] - box[1]), scalebar, coslat,
                         fontsize=FS["scale"])
        return sc

    tstr = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%H:%M:%S")
    head = f"{args.date} {tstr} UTC — Aqua true color (GIBS)"
    sc = map_panel(ax_map, bbox, 6 if J else 14,
                   head.replace(" — ", " —\n") if C else head, fontsize=FS["title"])
    map_panel(ax_zoom, zbox, 18 if J else 42, f"zoom ±{args.zoom_km:.0f} km", scalebar=5,
              fontsize=FS["title"])
    ax_zoom.legend(loc="upper right", fontsize=FS["leg"] if C else 8)
    if S:
        cb = fig.colorbar(sc, ax=[ax_map, ax_zoom], orientation="horizontal",
                          shrink=0.9, pad=0.1, aspect=30)
    else:
        cb = fig.colorbar(sc, ax=[ax_map, ax_zoom], shrink=0.8 if not J else 0.9,
                          pad=0.02 if not J else 0.03, label="MODIS cloud distance (km)")
    if C:
        cb.ax.tick_params(labelsize=FS["tick"])
        cb.set_label("MODIS cloud distance (km)", fontsize=FS["label"])
        ax_map.xaxis.set_major_locator(MaxNLocator(2))   # narrow overview panel

    if ax_heat is not None:
        vmax = max(2.0, float(np.nanpercentile(np.abs(seg["zk1_index"]), 98)))
        hm = ax_heat.scatter(seg["x_km"], seg["fp"], c=seg["zk1_index"],
                             cmap=CMAPS["mu"], vmin=-vmax, vmax=vmax,
                             marker="s", s=16)
        ax_heat.set_ylim(-0.7, 7.7); ax_heat.set_ylabel("footprint")
        ax_heat.axhline(fp_sel, color="k", lw=0.5, ls=":")
        fig.colorbar(hm, ax=ax_heat, pad=0.01,
                     label=rf"cross-band $\Delta${MEAN_L_LABEL} (z-score)")

    def band_panel(ax, fmt, ylabel, ratio=False, legend=False):
        for b in BANDS:
            col = fmt.format(b=b)
            other = seg[~sel]
            ax.plot(other["x_km"], other[col], ".", color=BAND_COLORS[b],
                    ms=2.5, alpha=0.22)
            mine = seg[sel].sort_values("x_km")
            ax.plot(mine["x_km"], mine[col], "-o", color=BAND_COLORS[b],
                    ms=3, lw=1.0, label=b.upper())
        ax.axhline(1.0 if ratio else 0.0, color="k", lw=0.5)
        ax.set_ylabel(ylabel, fontsize=FS["ylab"])
        if legend:  # above the panel, right-aligned, outside the box
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0),
                      fontsize=FS["leg"], ncol=3, frameon=False, borderaxespad=0.1)

    band_panel(ax_by["k1"], "d_{b}_k1", rf"$\Delta${MEAN_L_LABEL}", legend=True)
    band_panel(ax_by["k2"], "d_{b}_k2", rf"$\Delta${VAR_L_LABEL}")
    if "gam" in ax_by:
        band_panel(ax_by["gam"], "d_{b}_exp_intercept-alb", r"$\Delta$($\gamma$ $-$ alb)")
    band_panel(ax_by["cont"], "r_h_cont_{b}",
               "continuum\n/ clear" if C else "continuum / clear", ratio=True)
    if "alb" in ax_by:
        band_panel(ax_by["alb"], "d_alb_{b}", r"$\Delta$ albedo")

    ax = ax_by["xco2"]
    other = seg[~sel]
    ax.plot(other["x_km"], other["xco2_bc"], ".", color="0.55", ms=2.5,
            alpha=0.35, label="other fps")
    mine = seg[sel].sort_values("x_km")
    ax.plot(mine["x_km"], mine["xco2_bc"], "-o", ms=3, lw=1.2, color="k",
            label=f"fp {fp_sel}")
    base_bc = mine.loc[mine["cld_dist_km"] >= args.clear_km, "xco2_bc"].median()
    if np.isfinite(base_bc):
        ax.axhline(base_bc, color="0.4", lw=0.7, ls="--", label="clear median")
    ax.set_ylim(*xco2_bc_ylim(mine["xco2_bc"], base_bc))
    # journal edition: the manuscript names this product B11, not "BC"
    ax.set_ylabel((r"$X_{\mathrm{CO2}}^{\mathrm{B11}}$" if C else f"{XCO2_LABEL} BC")
                  + ("\n" if C else " ") + "(ppm)", fontsize=FS["ylab"])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=FS["leg"],
              ncol=3, frameon=False, borderaxespad=0.1)

    ax = ax_by["dist"]
    for f, g in seg.groupby("fp"):
        g = g.sort_values("x_km")
        ax.plot(g["x_km"], g["cld_dist_km"].clip(lower=0.05),
                color="k" if f == fp_sel else "0.75",
                lw=1.4 if f == fp_sel else 0.6)
    ax.set_yscale("log")
    ax.axhline(args.clear_km, color="g", lw=0.6, ls="--")
    ax.set_ylabel("cld dist" + ("\n" if C else " ") + "(km)", fontsize=FS["ylab"])
    ax.set_xlabel("along-track distance from case frame (km)", fontsize=FS["label"])
    if C:
        for a in axes:
            a.tick_params(labelsize=FS["tick"])
        for a in axes[:-1]:                      # linear panels: about three y ticks
            a.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 4, 5, 10]))
        for a in axes:                           # horizontal labels, one aligned column
            a.set_ylabel(a.get_ylabel(), rotation=0, ha="right", va="center",
                         labelpad=6, linespacing=1.05, fontsize=FS["ylab"])
        fig.align_ylabels(axes)

    mine = seg[sel].sort_values("x_km")
    spans = contiguous_spans(mine["x_km"].to_numpy(),
                             (mine["cld_dist_km"] < args.shade_km).to_numpy())
    shade_axes = ([ax_heat] if ax_heat is not None else []) + axes
    for a in shade_axes:
        for x0, x1 in spans:
            a.axvspan(x0, x1, color="gold", alpha=0.15, zorder=0)
    for a in shade_axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    lettered = ([ax_map, ax_zoom] + ([ax_heat] if ax_heat is not None else [])
                + axes)
    for letter, a in zip("abcdefghij", lettered):
        panel_label(a, f"({letter})", size=FS["letter"], inside=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_id = int(center["fp_id"].iloc[0] // 10)
    stem = args.out_stem or f"case_{args.date}_{frame_id}_fp{fp_sel}"
    for fmt in (("png", "pdf") if C else (args.fmt,)):
        out = args.output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
