#!/usr/bin/env python
"""Single-case spec-feature figure: MODIS RGB context + along-track traces.

For one frame picked by screen_spec_cases.py, renders:
  - Aqua-MODIS true-color RGB (NASA GIBS WMS, no login; the daily composite
    at the overpass location IS the A-Train granule minutes from OCO-2) with
    the 8-footprint track overlaid, colored by MODIS cloud distance;
  - a footprint x along-track heatmap of the cross-band delta-k1 z-index
    (the spectra localizing the cloud without the imager);
  - along-track traces for the chosen footprint (bold; other footprints as
    faint dots): delta k1, delta k2, delta (exp_intercept - albedo),
    continuum-radiance ratio, delta band albedo (surface-uniformity
    control), delta XCO2, and per-footprint cloud distance.

All deltas are vs the footprint's own clear-scene baseline (median over
cld_dist >= --clear-km inside the plotted segment), so the three bands
share axes and all-band coherence is visible at a glance.

Usage (after screening):
  python workspace/spec_case_study/spec_case_figure.py \
      --date 2018-07-21 --frame 201807211234567 [--fp 2]
"""

from __future__ import annotations

import argparse
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

DEFAULT_PARQUET = Path("results/csv_collection/combined_2016_2020_dates.parquet")
DEFAULT_OUTDIR = Path("results/figures/cld_dist_analysis/spec_case_study")

GROUND_SPEED_KMPS = 6.74          # OCO-2 along-track ground speed
BANDS = ("o2a", "wco2", "sco2")
BAND_COLORS = {"o2a": "#1f77b4", "wco2": "#2ca02c", "sco2": "#d62728"}
GIBS_WMS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
            "?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
            "&SRS=EPSG:4326&BBOX={w},{s},{e},{n}&WIDTH={px_w}&HEIGHT={px_h}"
            "&FORMAT=image/png&TIME={date}")
GIBS_LAYER = "MODIS_Aqua_CorrectedReflectance_TrueColor"

LOAD_COLS = (["date", "time", "fp", "fp_id", "lat", "lon", "cld_dist_km",
              "sfc_type", "xco2_qf", "xco2_raw_minus_apriori",
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
    ap.add_argument("--parquet-fname", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    if args.frame is None and args.frame_time is None:
        ap.error("give --frame or --frame-time")

    print(f"Loading {args.date} from {args.parquet_fname} ...")
    tbl = pq.read_table(args.parquet_fname, columns=LOAD_COLS,
                        filters=[("date", "==", args.date.encode())])
    df = tbl.to_pandas().sort_values("time").reset_index(drop=True)
    if not len(df):
        raise SystemExit(f"No rows for date {args.date}")

    if args.frame is not None:
        center = df[df["fp_id"] // 10 == args.frame]
        if not len(center):
            raise SystemExit(f"Frame {args.frame} not found on {args.date}")
        t0 = float(center["time"].mean())
    else:
        t0 = args.frame_time
        center = df[(df["time"] - t0).abs() < 0.4]

    half_s = args.half_span_km / GROUND_SPEED_KMPS
    seg = df[(df["time"] - t0).abs() <= half_s].copy()
    seg["x_km"] = (seg["time"] - t0) * GROUND_SPEED_KMPS
    fp_sel = (args.fp if args.fp is not None
              else int(center.loc[center["cld_dist_km"].idxmin(), "fp"]))
    print(f"Segment: {len(seg)} soundings, fp={fp_sel}, "
          f"center ({center['lat'].mean():.2f}, {center['lon'].mean():.2f})")

    # deltas vs per-footprint clear baseline
    trace_vars = ([f"{b}_k1" for b in BANDS] + [f"{b}_k2" for b in BANDS]
                  + [f"{b}_exp_intercept-alb" for b in BANDS]
                  + [f"alb_{b}" for b in BANDS]
                  + ["xco2_raw_minus_apriori", "xco2_bc_anomaly"])
    for var in trace_vars:
        seg[f"d_{var}"] = seg[var] - clear_baseline(seg, var, args.clear_km)
    for b in BANDS:  # continuum radiance as ratio to clear level
        seg[f"r_h_cont_{b}"] = seg[f"h_cont_{b}"] / clear_baseline(
            seg, f"h_cont_{b}", args.clear_km)

    # cross-band delta-k1 z-index (needs all three bands)
    clear_rows = seg["cld_dist_km"] >= args.clear_km
    zcols = []
    for b in BANDS:
        sd = seg.loc[clear_rows, f"d_{b}_k1"].std()
        seg[f"z_{b}_k1"] = seg[f"d_{b}_k1"] / sd if sd > 0 else np.nan
        zcols.append(f"z_{b}_k1")
    seg["zk1_index"] = seg[zcols].mean(axis=1)

    # ── figure ────────────────────────────────────────────────────────────
    n_tr = 7
    fig = plt.figure(figsize=(9.5, 21))
    gs = fig.add_gridspec(2 + n_tr, 1,
                          height_ratios=[4.2, 1.1] + [1.0] * n_tr,
                          hspace=0.34)
    gs_maps = gs[0].subgridspec(1, 2, wspace=0.18)
    ax_map = fig.add_subplot(gs_maps[0])
    ax_zoom = fig.add_subplot(gs_maps[1])
    ax_heat = fig.add_subplot(gs[1])
    axes = [fig.add_subplot(gs[2 + i], sharex=ax_heat) for i in range(n_tr)]

    # map + RGB (overview of the segment; zoom around the case frame)
    lat0 = float(seg["lat"].mean())
    coslat = max(np.cos(np.deg2rad(lat0)), 0.1)
    pad_lat, pad_lon = args.pad_km / 111.0, args.pad_km / (111.0 * coslat)
    bbox = (float(seg["lon"].min() - pad_lon), float(seg["lat"].min() - pad_lat),
            float(seg["lon"].max() + pad_lon), float(seg["lat"].max() + pad_lat))
    clat, clon = float(center["lat"].mean()), float(center["lon"].mean())
    zlat, zlon = args.zoom_km / 111.0, args.zoom_km / (111.0 * coslat)
    zbox = (clon - zlon, clat - zlat, clon + zlon, clat + zlat)
    sel = seg["fp"] == fp_sel

    def map_panel(ax, box, s_pts, title, fontsize=10):
        if not args.no_rgb:
            try:
                img, extent = fetch_rgb(args.date, box,
                                        args.output_dir / "rgb_cache",
                                        args.img_width)
                ax.imshow(img, extent=extent, origin="upper", zorder=0)
            except Exception as exc:  # keep the figure usable offline
                print(f"GIBS fetch failed ({exc}); plain map panel.")
        sc = ax.scatter(seg["lon"], seg["lat"], c=seg["cld_dist_km"],
                        cmap="viridis", vmin=0, vmax=args.clear_km, s=s_pts,
                        edgecolors="k", linewidths=0.2, zorder=2)
        ax.scatter(seg.loc[sel, "lon"], seg.loc[sel, "lat"], facecolors="none",
                   edgecolors="r", s=s_pts * 2.8, linewidths=0.7, zorder=3,
                   label=f"fp {fp_sel}")
        ax.scatter(clon, clat, marker="*", s=180, color="r", edgecolors="w",
                   zorder=4, label="case frame")
        ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
        ax.set_aspect(1.0 / coslat)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(title, fontsize=fontsize)
        return sc

    tstr = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%H:%M:%S")
    sc = map_panel(ax_map, bbox, 14,
                   f"{args.date} {tstr} UTC — Aqua true color (GIBS)")
    map_panel(ax_zoom, zbox, 42, f"zoom ±{args.zoom_km:.0f} km")
    ax_zoom.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=[ax_map, ax_zoom], shrink=0.8, pad=0.02,
                 label="MODIS cloud distance (km)")

    # heatmap: fp x along-track, cross-band delta-k1 z-index
    vmax = max(2.0, float(np.nanpercentile(np.abs(seg["zk1_index"]), 98)))
    hm = ax_heat.scatter(seg["x_km"], seg["fp"], c=seg["zk1_index"],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, marker="s", s=16)
    ax_heat.set_ylim(-0.7, 7.7); ax_heat.set_ylabel("footprint")
    ax_heat.axhline(fp_sel, color="k", lw=0.5, ls=":")
    fig.colorbar(hm, ax=ax_heat, pad=0.01, label="cross-band Δk1 z")
    ax_heat.set_title("spectra-only cloud localization (no imager input)",
                      fontsize=9)

    def band_panel(ax, fmt, ylabel, ratio=False):
        for b in BANDS:
            col = fmt.format(b=b)
            other = seg[~sel]
            ax.plot(other["x_km"], other[col], ".", color=BAND_COLORS[b],
                    ms=2.5, alpha=0.22)
            mine = seg[sel].sort_values("x_km")
            ax.plot(mine["x_km"], mine[col], "-o", color=BAND_COLORS[b],
                    ms=3, lw=1.0, label=b)
        ax.axhline(1.0 if ratio else 0.0, color="k", lw=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    band_panel(axes[0], "d_{b}_k1", "Δk1")
    band_panel(axes[1], "d_{b}_k2", "Δk2")
    band_panel(axes[2], "d_{b}_exp_intercept-alb", "Δ(exp int − alb)")
    band_panel(axes[3], "r_h_cont_{b}", "continuum / clear", ratio=True)
    band_panel(axes[4], "d_alb_{b}", "Δ albedo")

    ax = axes[5]
    mine = seg[sel].sort_values("x_km")
    ax.plot(mine["x_km"], mine["d_xco2_raw_minus_apriori"], "-o", ms=3, lw=1.0,
            color="#9467bd", label="Δ(raw − apriori)")
    ax.plot(mine["x_km"], mine["d_xco2_bc_anomaly"], "-o", ms=3, lw=1.0,
            color="#8c564b", label="Δ XCO2 BC anomaly")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("ΔXCO2 (ppm)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[6]
    for f, g in seg.groupby("fp"):
        g = g.sort_values("x_km")
        ax.plot(g["x_km"], g["cld_dist_km"].clip(lower=0.05),
                "-" if f == fp_sel else "-",
                color="k" if f == fp_sel else "0.75",
                lw=1.4 if f == fp_sel else 0.6)
    ax.set_yscale("log")
    ax.axhline(args.clear_km, color="g", lw=0.6, ls="--")
    ax.set_ylabel("cld dist (km)", fontsize=9)
    ax.set_xlabel("along-track distance from case frame (km)")

    mine = seg[sel].sort_values("x_km")
    spans = contiguous_spans(mine["x_km"].to_numpy(),
                             (mine["cld_dist_km"] < args.shade_km).to_numpy())
    for a in [ax_heat] + axes:
        for x0, x1 in spans:
            a.axvspan(x0, x1, color="gold", alpha=0.15, zorder=0)
    for a in [ax_heat] + axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_id = int(center["fp_id"].iloc[0] // 10)
    out = args.output_dir / f"case_{args.date}_{frame_id}_fp{fp_sel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
