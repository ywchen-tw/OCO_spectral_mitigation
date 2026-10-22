#!/usr/bin/env python3
"""plot_atom_poster.py — poster-styled ATom × OCO-2 figure, three candidate layouts.

Sibling of the TCCON poster (plot_corrected_xco2._save_poster_comparison_figure):
same fonts, colour-scale conventions and "B11 | ML-corrected | colourbar | ..."
rhythm, adapted to a moving aircraft reference (one AK pseudo-column per profile
leg instead of one station time series).

Layouts (one PNG each so they can be compared side by side):
  A  sibling      : B11 map | ML map | cbar | pooled histogram, one reference line
                    per leg
  B  per-leg bias : B11 map | ML map | cbar | per-leg (OCO-2 − ATom) dot plot
  C  both         : B11 map | ML map | cbar | histogram | dot plot

Map panels: MODIS Aqua background; collocated footprints at full opacity, other
ocean-glint footprints faint; ATom track trimmed to the collocation window; each
collocated leg marked by a numbered diamond; the collocation corridor (leg track
buffered by the radius) dashed.  Extent = corridor envelope, widened so the strip
is at most MAX_ASPECT tall (the extra MODIS area shows the cloud field).

Default mode is per-leg: each profile leg is a manuscript case (the ATom analogue of
a TCCON overpass, Table D3), so each gets its own layout-A figure with its corridor,
its collocated footprints and its single reference line.  --mode date draws all
legs of the date in one figure (layouts A/B/C) for comparison.

Usage: python plot_atom_poster.py [--date 2017-10-20] [--mode per-leg|date]
                                  [--layout A|B|C|all] [--radius-km 100]
                                  [--window-min 120] [--no-modis]
Output: combined_<date>_atom/poster_atom_<date>_leg<NN>_r<R>km.png     (per-leg)
        combined_<date>_atom/poster_atom_layout<X>_<date>_r<R>km.png  (date)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from atom_pseudo_column import CORR_COL, OUT_BASE, PLOT_BASE  # noqa: E402
from plot_atom_comparison import TILES_DIR, collocate_date  # noqa: E402
from plot_corrected_xco2 import download_modis_rgb  # noqa: E402
from plot_style import CMAPS, XCO2_LABEL  # noqa: E402

KM_PER_DEG = 111.195
MAX_ASPECT = 1.8          # date-wide strips: cap on height : width (km)
LEG_ASPECT = 1.4          # per-leg maps: every box padded to this height : width,
                          # so all per-leg figures share one width and panel grid
SMALL_N = 10              # legs with fewer footprints are drawn hollow / grey
FS = dict(title=23, map_title=20, label=18, tick=16, legend=17, map_legend=15, leg_num=13)
COL_BC, COL_ML, COL_ATOM, COL_TRACK, COL_LEG = "black", "green", "coral", "magenta", "cyan"


# ── geometry ─────────────────────────────────────────────────────────────────
def _local_xy(lon, lat, lon_c, lat_c):
    x = (np.asarray(lon) - lon_c) * KM_PER_DEG * np.cos(np.radians(lat_c))
    y = (np.asarray(lat) - lat_c) * KM_PER_DEG
    return x, y


def _lonlat(x, y, lon_c, lat_c):
    lon = lon_c + np.asarray(x) / (KM_PER_DEG * np.cos(np.radians(lat_c)))
    lat = lat_c + np.asarray(y) / KM_PER_DEG
    return lon, lat


def leg_corridor(leg: pd.DataFrame, radius_km: float, lon_c: float, lat_c: float):
    """Shapely polygon (local km) of the leg track buffered by the collocation radius."""
    pts = leg[["lon", "lat"]].dropna().drop_duplicates().to_numpy()
    x, y = _local_xy(pts[:, 0], pts[:, 1], lon_c, lat_c)
    if len(pts) < 2:
        return Point(x[0], y[0]).buffer(radius_km)
    return LineString(np.c_[x, y]).simplify(1.0).buffer(radius_km)


def strip_extent(corridor, lon_c, lat_c, pad_km=15.0, fixed_aspect=None):
    """[lon0, lon1, lat0, lat1] around the corridor union, padded.  With
    fixed_aspect the box is grown in the shorter direction to exactly that
    height:width; otherwise it is only clamped to 1/MAX_ASPECT .. MAX_ASPECT."""
    x0, y0, x1, y1 = corridor.bounds
    x0, y0, x1, y1 = x0 - pad_km, y0 - pad_km, x1 + pad_km, y1 + pad_km
    w, h = x1 - x0, y1 - y0
    if fixed_aspect is not None:
        if h / w > fixed_aspect:
            grow = (h / fixed_aspect - w) / 2
            x0, x1 = x0 - grow, x1 + grow
        else:
            grow = (w * fixed_aspect - h) / 2
            y0, y1 = y0 - grow, y1 + grow
    elif h / w > MAX_ASPECT:
        grow = (h / MAX_ASPECT - w) / 2
        x0, x1 = x0 - grow, x1 + grow
    elif w / h > MAX_ASPECT:
        grow = (w / MAX_ASPECT - h) / 2
        y0, y1 = y0 - grow, y1 + grow
    lon0, lat0 = _lonlat(x0, y0, lon_c, lat_c)
    lon1, lat1 = _lonlat(x1, y1, lon_c, lat_c)
    aspect = (y1 - y0) / (x1 - x0)
    return [float(lon0), float(lon1), float(lat0), float(lat1)], aspect


def _polygons(geom):
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


# ── data ─────────────────────────────────────────────────────────────────────
def prepare(date: str, radius_km: float, twin_s: float) -> dict:
    """Collocate the date once; per-leg statistics and the date-wide colour range."""
    oco, atom, legs, pooled = collocate_date(date, radius_km, twin_s)
    if not legs:
        raise SystemExit(f"{date}: no collocated legs")
    legs = sorted(legs, key=lambda L: -L["lat"])            # north → south

    # per-leg reference uncertainty from the Stage-2/3 table (NaN if absent)
    res_path = os.path.join(OUT_BASE, "atom_pseudo_column_results.csv")
    ak_sd = {}
    if os.path.exists(res_path):
        r = pd.read_csv(res_path)
        r = r[r.date == date]
        ak_sd = dict(zip(r.profile_id.astype(int), r.atom_ak_sd))

    for L in legs:
        bc = L["sub"].xco2_bc.to_numpy(float)
        ml = L["sub"][CORR_COL].to_numpy(float)
        L["row"] = dict(
            pid=L["pid"], lat=L["lat"], lon=L["lon"], n=len(L["sub"]),
            cld_med=float(np.nanmedian(L["sub"].cld_dist_km)),
            c_ak=L["c_ak"], ak_sd=float(ak_sd.get(L["pid"], np.nan)),
            bc_mean=float(np.nanmean(bc)), bc_sd=float(np.nanstd(bc)),
            ml_mean=float(np.nanmean(ml)), ml_sd=float(np.nanstd(ml)),
        )
        L["row"]["resid_bc"] = L["row"]["bc_mean"] - L["c_ak"]
        L["row"]["resid_ml"] = L["row"]["ml_mean"] - L["c_ak"]

    # one colour range and one histogram x-range per date so the per-leg figures
    # of a date are comparable
    o_ml = pooled[CORR_COL].to_numpy(float)
    o_bc = pooled.xco2_bc.to_numpy(float)
    vmin = float(np.floor(np.nanpercentile(o_ml, 2) * 2) / 2)
    vmax = float(np.ceil(np.nanpercentile(o_ml, 98) * 2) / 2)
    c_ak = np.array([L["c_ak"] for L in legs])
    xlo = min(np.nanpercentile(o_bc, 1), np.nanpercentile(o_ml, 1), c_ak.min()) - 0.2
    xhi = max(np.nanpercentile(o_bc, 99), np.nanpercentile(o_ml, 99), c_ak.max()) + 0.2
    xlo, xhi = float(np.floor(xlo * 2) / 2), float(np.ceil(xhi * 2) / 2)
    return dict(date=date, radius_km=radius_km, twin_s=twin_s, oco=oco, atom=atom,
                legs=legs, vmin=vmin, vmax=vmax, xlo=xlo, xhi=xhi)


def scene(P: dict, legs: list, do_modis: bool) -> dict:
    """Map geometry and panel data for a subset of the date's legs (one leg for the
    per-leg figures, all legs for the date-wide layouts)."""
    radius_km, twin_s, oco, atom = P["radius_km"], P["twin_s"], P["oco"], P["atom"]
    lon_c = float(np.mean([L["lon"] for L in legs]))
    lat_c = float(np.mean([L["lat"] for L in legs]))
    pooled = oco.loc[np.unique(np.concatenate([L["sub"].index.to_numpy() for L in legs]))]
    rows = [L["row"] for L in legs]

    corridor = unary_union([leg_corridor(L["leg"], radius_km, lon_c, lat_c) for L in legs])
    ext, aspect = strip_extent(corridor, lon_c, lat_c,
                               fixed_aspect=LEG_ASPECT if len(legs) == 1 else None)
    corridor_lonlat = []
    for poly in _polygons(corridor):
        bx, by = np.asarray(poly.exterior.coords).T
        corridor_lonlat.append(_lonlat(bx, by, lon_c, lat_c))

    # ATom track restricted to the collocation window and the map box
    t0, t1 = float(pooled.time.min()) - twin_s, float(pooled.time.max()) + twin_s
    track = atom[(atom.epoch >= t0) & (atom.epoch <= t1)
                 & atom.lon.between(ext[0], ext[1]) & atom.lat.between(ext[2], ext[3])]

    inx = oco[oco.lon.between(ext[0], ext[1]) & oco.lat.between(ext[2], ext[3])]
    inx = inx[np.isfinite(inx[CORR_COL]) & np.isfinite(inx.xco2_bc)]
    ctx = inx[~inx.index.isin(pooled.index)]

    bg = None
    if do_modis:
        try:
            os.makedirs(TILES_DIR, exist_ok=True)
            # GIBS daily mosaics are keyed by the LOCAL day of the overpass
            # (see plot_atom_comparison.make_fig for the antimeridian case).
            gibs_dt = (pd.Timestamp(float(pooled.time.median()), unit="s")
                       + pd.Timedelta(hours=float(pooled.lon.median()) / 15.0))
            gibs_date = gibs_dt.normalize()
            cached = download_modis_rgb(gibs_date, ext, which="aqua", fdir=TILES_DIR, run=False)
            png = cached if os.path.exists(cached) else \
                download_modis_rgb(gibs_date, ext, which="aqua", fdir=TILES_DIR)
            bg = plt.imread(png)
        except Exception as e:                       # noqa: BLE001 — points-only fallback
            print(f"{P['date']}: MODIS unavailable ({e})")

    leg_tracks = [L["leg"][["lon", "lat"]] for L in legs]
    return dict(date=P["date"], radius_km=radius_km, rows=rows, pooled=pooled, ctx=ctx,
                track=track, leg_tracks=leg_tracks, corridor=corridor_lonlat, ext=ext,
                aspect=aspect, vmin=P["vmin"], vmax=P["vmax"], xlo=P["xlo"], xhi=P["xhi"],
                bg=bg)


# ── panels ───────────────────────────────────────────────────────────────────
def draw_map(ax, D: dict, col: str, title: str, *, show_ylabel: bool):
    ext, norm = D["ext"], mcolors.Normalize(D["vmin"], D["vmax"])
    if D["bg"] is not None:
        ax.imshow(D["bg"], extent=ext, aspect="auto", origin="upper", zorder=0)
    ax.scatter(D["pooled"].lon, D["pooled"].lat, c=D["pooled"][col], cmap=CMAPS["xco2"],
               norm=norm, s=45, alpha=0.9, linewidths=0, rasterized=True, zorder=3,
               label="OCO-2 footprint")
    ax.plot(D["track"].lon, D["track"].lat, "-", color=COL_TRACK, lw=2.0, zorder=4,
            label="ATom track")
    for t in D["leg_tracks"]:                     # the profile leg itself, on top
        ax.plot(t.lon, t.lat, "-", color=COL_LEG, lw=3.0, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
    for i, (cx, cy) in enumerate(D["corridor"]):
        ax.plot(cx, cy, ls="--", lw=1.8, color="white", zorder=4,
                path_effects=[pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()],
                label=f"{D['radius_km']:g} km" if i == 0 else None)
    for i, r in enumerate(D["rows"]):
        ax.scatter([r["lon"]], [r["lat"]], marker="D", s=320, c="white", edgecolors="black",
                   linewidths=1.6, zorder=6, label="ATom profile leg" if i == 0 else None)
        ax.text(r["lon"], r["lat"], str(r["pid"]), ha="center", va="center",
                fontsize=FS["leg_num"] - 1, weight="bold", zorder=7)
    ax.set_title(title, fontsize=FS["map_title"], weight="bold", pad=14)
    ax.set_xlabel("Longitude (°E)", fontsize=FS["label"])
    ax.set_ylabel("Latitude (°N)" if show_ylabel else "", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1.0 / max(np.cos(np.radians(0.5 * (ext[2] + ext[3]))), 0.05))
    ax.grid(alpha=0.18, linewidth=0.6)


def _two_col_legend(ax, handles, names, stats, *, outside: bool):
    """TCCON-poster legend: (handle, name) column | (blank, statistic) column."""
    blank = Line2D([], [], linestyle="none")
    kw = dict(ncol=2, columnspacing=-1.0, handlelength=1.6, fontsize=FS["legend"],
              frameon=False)
    if outside:
        kw.update(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        kw.update(loc="upper center", bbox_to_anchor=(0.5, -0.17), borderaxespad=0.0)
    ax.legend(handles + [blank] * len(stats), names + stats, **kw)


def draw_hist(ax, D: dict, *, legend_outside: bool):
    rows, pooled = D["rows"], D["pooled"]
    bc = pooled.xco2_bc.to_numpy(float)
    ml = pooled[CORR_COL].to_numpy(float)
    c_ak = np.array([r["c_ak"] for r in rows])
    lo, hi = D["xlo"], D["xhi"]                  # one x-range per date
    n_ml = int(np.isfinite(ml).sum())
    bins = np.arange(lo, hi + 1e-9, 0.05 if n_ml >= 100 else 0.1)

    # Too few footprints for a density: show every footprint as a tick instead
    small = n_ml < SMALL_N
    handles, names, stats = [], [], []
    for k, (vals, label, color) in enumerate(((bc, f"OCO-2 B11 {XCO2_LABEL}", COL_BC),
                                              (ml, f"ML-corrected {XCO2_LABEL}", COL_ML))):
        vals = vals[np.isfinite(vals)]
        if small:
            y0 = 0.62 - 0.4 * k
            ax.vlines(vals, y0, y0 + 0.22, color=color, linewidth=4.0, alpha=0.7)
            handles.append(Line2D([], [], color=color, lw=4.0, alpha=0.7))
        else:
            _, _, patches = ax.hist(vals, bins=bins, density=True, histtype="step",
                                    linewidth=2.6, color=color, alpha=0.6, label=label)
            handles.append(patches[0])
        ax.axvline(vals.mean(), color=color, linewidth=3.0, alpha=0.95)
        names.append(label)
        stats.append(f"{vals.mean():.1f} $\\pm$ {vals.std():.1f} ppm")

    # one reference line per leg (± its AK pseudo-column uncertainty), numbered on top
    rank = {r["pid"]: k for k, r in enumerate(sorted(rows, key=lambda r: r["c_ak"]))}
    for r in rows:
        i = rank[r["pid"]]
        if np.isfinite(r["ak_sd"]):
            ax.axvspan(r["c_ak"] - r["ak_sd"], r["c_ak"] + r["ak_sd"], color=COL_ATOM, alpha=0.12,
                       linewidth=0)
        ax.axvline(r["c_ak"], color=COL_ATOM, linewidth=2.2, linestyle="--", alpha=0.95)
        if len(rows) > 1:
            ax.text(r["c_ak"], 0.975 - 0.06 * (i % 2), str(r["pid"]),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=COL_ATOM, fontsize=14, weight="bold")
    handles.append(Line2D([], [], color=COL_ATOM, lw=2.2, ls="--"))
    n_leg = len(rows)
    if n_leg == 1:
        r = rows[0]
        names.append(f"ATom {XCO2_LABEL} (leg {r['pid']})")
        stats.append(f"{r['c_ak']:.1f} $\\pm$ {r['ak_sd']:.1f} ppm" if np.isfinite(r["ak_sd"])
                     else f"{r['c_ak']:.1f} ppm")
    else:
        names.append(f"ATom {XCO2_LABEL} ({n_leg} legs)")
        stats.append(f"{c_ak.min():.1f} ppm" if np.ptp(c_ak) < 0.05
                     else f"{c_ak.min():.1f}–{c_ak.max():.1f} ppm")
    _two_col_legend(ax, handles, names, stats, outside=legend_outside)
    if n_leg == 1:
        # the poster claim is about near-cloud footprints: say how near these are
        ax.text(1.04 if legend_outside else 0.5, 0.22 if legend_outside else -0.36,
                f"median cloud distance of the footprints: {rows[0]['cld_med']:.1f} km",
                transform=ax.transAxes, ha="left" if legend_outside else "center",
                va="top", fontsize=FS["tick"], color="dimgray")

    ax.set_title(f"{XCO2_LABEL} distributions (n = {n_ml})", fontsize=FS["title"],
                 weight="bold", pad=14)
    ax.set_xlabel(f"{XCO2_LABEL} (ppm)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.set_xlim(lo, hi)
    if small:
        ax.set_ylabel("footprints", fontsize=FS["label"])
        ax.set_yticks([])
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel("Density", fontsize=FS["label"])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)   # head-room for the leg numbers


def draw_dots(ax, D: dict):
    """Per-leg OCO-2 − ATom bias, B11 (open) → ML-corrected (filled), top row = north."""
    rows = D["rows"]
    n = len(rows)
    ys = np.arange(n)[::-1]
    dy = 0.16
    for y, r in zip(ys, rows):
        small = r["n"] < SMALL_N
        bc_face = "lightgray" if small else "white"
        ml_col = "darkseagreen" if small else COL_ML
        ax.errorbar(r["resid_bc"], y + dy, xerr=r["bc_sd"], fmt="o", ms=11, mfc=bc_face,
                    mec="black", ecolor="black", elinewidth=1.8, capsize=4, zorder=3)
        ax.errorbar(r["resid_ml"], y - dy, xerr=r["ml_sd"], fmt="o", ms=11, mfc=ml_col,
                    mec=ml_col, ecolor=ml_col, elinewidth=1.8, capsize=4, zorder=3)
        ax.annotate("", xy=(r["resid_ml"], y), xytext=(r["resid_bc"], y),
                    arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.6,
                                    shrinkA=0, shrinkB=2), zorder=2)
        ax.text(1.02, y, f"{r['cld_med']:.1f} km", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=FS["tick"], color="dimgray")
    ax.text(1.02, 1.0, "cloud\ndist.", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS["tick"], color="dimgray")
    ax.axvline(0, color="black", lw=1.2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"leg {r['pid']}  (n={r['n']})" for r in rows], fontsize=FS["tick"])
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="x", labelsize=FS["tick"])
    ax.set_xlabel(f"OCO-2 $-$ ATom {XCO2_LABEL} (ppm)", fontsize=FS["label"])
    ax.set_title("Per-leg bias", fontsize=FS["title"], weight="bold", pad=14)
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax.legend([Line2D([], [], marker="o", ms=11, mfc="white", mec="black", ls="none"),
               Line2D([], [], marker="o", ms=11, color=COL_ML, ls="none")],
              ["OCO-2 B11", "ML-corrected"], loc="lower right", fontsize=FS["map_legend"],
              frameon=True)
    if any(r["n"] < SMALL_N for r in rows):
        ax.text(0.0, -0.17, f"grey: n < {SMALL_N} footprints", transform=ax.transAxes,
                fontsize=13, color="dimgray", va="top")


# ── figure assembly ──────────────────────────────────────────────────────────
def build(layout: str, D: dict, out_path: str, dpi: int):
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    # All cells share the axes height; a map cell is made exactly as wide as its
    # geographic aspect needs at that height, so maps, colourbar and histogram
    # come out the same height (set_aspect would otherwise shrink the maps).
    fig_h, top, bottom = 6.2, 0.88, 0.12
    ax_h = fig_h * (top - bottom)
    map_w = ax_h / D["aspect"] * 1.01
    cols = [("map_bc", map_w), ("map_ml", map_w), ("cbar", 0.22), ("gap", 1.4)]
    if layout in ("A", "C"):
        cols.append(("hist", 5.6))
    if layout == "C":
        cols.append(("gap2", 1.6))
    elif layout == "B":
        cols.append(("gap2", 1.3))
    if layout in ("B", "C"):
        cols.append(("dots", 4.6))
    widths = [w for _, w in cols]
    wspace, left_in, right_in = 0.32, 0.9, 0.3
    fig_w = left_in + sum(widths) + wspace * np.mean(widths) * (len(cols) - 1) + right_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w, top=top, bottom=bottom)
    gs = fig.add_gridspec(1, len(cols), width_ratios=widths, wspace=wspace)
    axes = {name: fig.add_subplot(gs[0, i]) for i, (name, _) in enumerate(cols)
            if not name.startswith("gap")}

    draw_map(axes["map_bc"], D, "xco2_bc", f"OCO-2 B11 {XCO2_LABEL}", show_ylabel=True)
    draw_map(axes["map_ml"], D, CORR_COL, f"ML-corrected {XCO2_LABEL}", show_ylabel=False)
    sm = mcm.ScalarMappable(norm=mcolors.Normalize(D["vmin"], D["vmax"]), cmap=CMAPS["xco2"])
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axes["cbar"])
    cbar.set_label(f"{XCO2_LABEL} (ppm)", fontsize=20, labelpad=16)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    cbar.ax.tick_params(labelsize=FS["tick"])
    if "hist" in axes:
        draw_hist(axes["hist"], D, legend_outside=(layout == "A"))
    if "dots" in axes:
        draw_dots(axes["dots"], D)

    # one map legend under both strips (inside the strip it hides the north end)
    fig.canvas.draw()                          # geographic aspect fixes axes positions
    bb0 = axes["map_bc"].get_position()
    mid = matplotlib.colormaps[CMAPS["xco2"]](0.6)
    handles = [
        Line2D([], [], marker="o", ms=7, color=mid, ls="none"),
        Line2D([], [], color=COL_TRACK, lw=2.0),
        Line2D([], [], color="white", lw=1.8, ls="--",
               path_effects=[pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()]),
        Line2D([], [], marker="D", ms=12, mfc="white", mec="black", mew=1.6, ls="none"),
    ]
    handles.insert(2, Line2D([], [], color=COL_LEG, lw=3.0,
                             path_effects=[pe.Stroke(linewidth=4.5, foreground="black"),
                                           pe.Normal()]))
    labels = ["OCO-2 footprint", "ATom track", "profile leg", f"{D['radius_km']:g} km corridor",
              "leg mean position"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(bb0.x0, bb0.y0 - 0.11),
               bbox_transform=fig.transFigure, ncol=3, fontsize=FS["map_legend"],
               frameon=False, columnspacing=1.2, handlelength=2.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"layout {layout} → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2017-10-20")
    ap.add_argument("--mode", default="per-leg", choices=["per-leg", "date"],
                    help="per-leg: one TCCON-style figure per profile leg (the manuscript's "
                         "unit of analysis); date: all legs of the date in one figure")
    ap.add_argument("--layout", default="all", choices=["A", "B", "C", "all"],
                    help="date mode only; per-leg figures always use layout A")
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--no-modis", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    P = prepare(args.date, args.radius_km, args.window_min * 60)
    print(f"{args.date}: {len(P['legs'])} collocated legs, colour range "
          f"[{P['vmin']:.1f}, {P['vmax']:.1f}] ppm")
    for L in P["legs"]:
        r = L["row"]
        print(f"  leg {r['pid']:>2}  n={r['n']:>4}  cld_med={r['cld_med']:5.1f} km  "
              f"ATom {r['c_ak']:.2f}  bias B11 {r['resid_bc']:+.2f}±{r['bc_sd']:.2f} → "
              f"ML {r['resid_ml']:+.2f}±{r['ml_sd']:.2f} ppm")
    out_dir = os.path.join(PLOT_BASE, f"combined_{args.date}_atom")
    if args.mode == "per-leg":
        for L in P["legs"]:
            D = scene(P, [L], not args.no_modis)
            build("A", D, os.path.join(out_dir, f"poster_atom_{args.date}_leg{L['pid']:02d}"
                                               f"_r{args.radius_km:g}km.png"), args.dpi)
    else:
        D = scene(P, P["legs"], not args.no_modis)
        print(f"  extent {[round(e, 2) for e in D['ext']]}, strip aspect {D['aspect']:.2f}")
        for layout in (["A", "B", "C"] if args.layout == "all" else [args.layout]):
            build(layout, D, os.path.join(out_dir, f"poster_atom_layout{layout}_{args.date}"
                                                  f"_r{args.radius_km:g}km.png"), args.dpi)


if __name__ == "__main__":
    main()
