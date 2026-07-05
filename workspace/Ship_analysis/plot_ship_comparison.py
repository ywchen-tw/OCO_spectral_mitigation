#!/usr/bin/env python3
"""Stage 2 — ship EM27/SUN vs OCO-2 (DeepEns-corrected) comparison figure.

Ship-native analog of plot_corrected_xco2.py, WITHOUT the TCCON machinery: the
reference is a moving vessel, labelled as such (no "TCCON station").  Reads the
DeepEns-corrected plot_data.parquet (from build_deepens_plot_data.py) + the raw
ship .tab, collocates OCO-2 ocean-glint footprints to the ship track in space+time,
and draws a 4-panel figure + a stats line.

Panels:
  (1) map: DeepEns-corrected XCO2 footprints + the ship track (coloured by ship XCO2)
  (2) histogram: OCO-2 original vs DeepEns-corrected vs ship
  (3) collocated footprints' nearest-cloud distance (the near-cloud context)
  (4) ship XCO2 time series with the OCO-2 overpass window shaded

Usage:
  python plot_ship_comparison.py --plot-data <plot_data.parquet> --ship-tag so268 \
      --date 2019-06-22 --lon-range 152.36 152.80 --lat-range 27.43 28.96 \
      --vmin 411.5 --vmax 412.5 --output-dir <dir>
"""
from __future__ import annotations
import argparse, datetime as dt, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "workspace"))   # reuse the TCCON MODIS-RGB fetcher
EPOCH = dt.datetime(1970, 1, 1)
TABS = {
    "so268":  (os.path.join(REPO, "data/Other/SO268-3_track_XCO2_XCH4_XCO.tab"), "MORE-2 (RV Sonne)"),
    "mr2101": (os.path.join(REPO, "data/Other/Hanft_2021_XCO2_XCH4_XCO.tab"),   "MR21-01 (RV Mirai)"),
}


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def load_ship(tag, date):
    path = TABS[tag][0]
    t0 = (dt.datetime.strptime(date, "%Y-%m-%d") - EPOCH).total_seconds()
    rows, started = [], False
    for line in open(path):
        if not started:
            if line.startswith("Date/Time\t"):
                started = True
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4 or not p[0].startswith("20"):
            continue
        try:
            lon, lat, x = float(p[1]), float(p[2]), float(p[3])
        except ValueError:
            continue
        fmt = "%Y-%m-%dT%H:%M:%S" if p[0].count(":") == 2 else "%Y-%m-%dT%H:%M"
        e = (dt.datetime.strptime(p[0], fmt) - EPOCH).total_seconds()
        if t0 <= e < t0 + 86400:
            rows.append((e, lon, lat, x))
    return pd.DataFrame(rows, columns=["epoch", "lon", "lat", "xco2"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-data", required=True)
    ap.add_argument("--ship-tag", required=True, choices=list(TABS))
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (OCO overpass day)")
    ap.add_argument("--lon-range", type=float, nargs=2, required=True)
    ap.add_argument("--lat-range", type=float, nargs=2, required=True)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--modis-auto", action="store_true",
                    help="overlay a MODIS Aqua true-colour RGB (NASA GIBS) on the map panels")
    ap.add_argument("--modis-which", default="aqua", choices=["aqua", "terra"])
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    corr_col = "deep_ensemble_corrected_xco2"

    oco = pd.read_parquet(args.plot_data)
    oco = oco[(oco.sfc_type == 0)]                                   # ocean glint
    oco = oco[oco.lon.between(*args.lon_range) & oco.lat.between(*args.lat_range)]
    ship = load_ship(args.ship_tag, args.date)
    if ship.empty:
        raise SystemExit(f"No ship points on {args.date} for {args.ship_tag}")

    # collocate OCO footprints to the moving ship track (space AND time)
    olon = oco.lon.to_numpy(); olat = oco.lat.to_numpy(); ot = oco.time.to_numpy()
    slon = ship.lon.to_numpy(); slat = ship.lat.to_numpy(); st = ship.epoch.to_numpy()
    twin = args.window_min * 60
    dmin = np.full(len(oco), np.inf)
    for i0 in range(0, len(oco), 200):
        sl = slice(i0, i0+200)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None], slon[None, :], slat[None, :])
        tg = np.abs(ot[sl][:, None] - st[None, :])
        dmin[sl] = np.where(tg <= twin, d, np.inf).min(axis=1)
    near = oco[dmin <= args.radius_km].copy()
    if near.empty:
        raise SystemExit(f"0 OCO footprints within {args.radius_km} km / +-{args.window_min} min")

    # ship subset in the OCO overpass window (for the reference distribution)
    ow = (st >= ot.min() - twin) & (st <= ot.max() + twin)
    ship_w = ship[ow]

    o_orig, o_corr = near.xco2_bc.to_numpy(), near[corr_col].to_numpy()
    s_ref = ship_w.xco2.to_numpy()
    d_orig = np.nanmedian(o_orig) - np.nanmedian(s_ref)
    d_corr = np.nanmedian(o_corr) - np.nanmedian(s_ref)
    site = TABS[args.ship_tag][1]

    # ── optional MODIS Aqua true-colour background (NASA GIBS, same fetcher as
    #    plot_corrected_xco2.py --modis-auto) ─────────────────────────────────────
    bg_img = bg_extent = None
    if args.modis_auto:
        try:
            from plot_corrected_xco2 import download_modis_rgb
            ext = [args.lon_range[0], args.lon_range[1], args.lat_range[0], args.lat_range[1]]
            rgb_path = download_modis_rgb(pd.Timestamp(args.date), ext,
                                          which=args.modis_which, fdir=args.output_dir,
                                          coastline=True)
            bg_img, bg_extent = plt.imread(rgb_path), ext
        except Exception as exc:
            print(f"  Warning: MODIS RGB unavailable ({exc}) — maps show soundings only.")

    def _bg(a):
        if bg_img is not None:
            a.imshow(bg_img, extent=bg_extent, aspect="auto", origin="upper", zorder=0)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"OCO-2 (DeepEns-corrected) vs shipborne EM27/SUN — {site}  {args.date}\n"
                 f"{len(near)} ocean-glint footprints ≤{args.radius_km:.0f} km / ±{args.window_min:.0f} min",
                 fontsize=15, weight="bold")

    # (1) map
    a = ax[0, 0]
    _bg(a)
    sc = a.scatter(near.lon, near.lat, c=near[corr_col], s=14, cmap="turbo",
                   vmin=args.vmin, vmax=args.vmax, label="OCO-2 corrected", zorder=3)
    a.plot(ship.lon, ship.lat, "-", color="0.5", lw=0.8, alpha=0.7, zorder=4)
    a.scatter(ship.lon, ship.lat, c=ship.xco2, s=40, cmap="turbo", vmin=args.vmin, vmax=args.vmax,
              edgecolor="k", linewidth=0.6, marker="D", zorder=6, label="ship EM27/SUN")
    a.set(xlim=args.lon_range, ylim=args.lat_range, xlabel="Lon (°E)", ylabel="Lat (°N)",
          title="DeepEns-corrected XCO₂ + ship track"
                + (" (MODIS Aqua)" if bg_img is not None else ""))
    fig.colorbar(sc, ax=a, label="XCO₂ (ppm)"); a.legend(loc="upper left", fontsize=8)

    # (2) histogram
    a = ax[0, 1]
    bins = np.linspace(min(np.nanmin(o_orig), np.nanmin(s_ref)),
                       max(np.nanmax(o_orig), np.nanmax(s_ref)), 60)
    a.hist(o_orig, bins, density=True, alpha=0.45, color="tab:blue",
           label=f"OCO-2 original  μ={np.nanmean(o_orig):.2f}")
    a.hist(o_corr, bins, density=True, alpha=0.45, color="tab:green",
           label=f"OCO-2 DeepEns   μ={np.nanmean(o_corr):.2f}")
    a.hist(s_ref, bins, density=True, alpha=0.45, color="tab:red",
           label=f"ship EM27/SUN   μ={np.nanmean(s_ref):.2f}")
    a.axvline(np.nanmedian(s_ref), color="tab:red", lw=2)
    a.set(xlabel="XCO₂ (ppm)", ylabel="Density",
          title=f"Δmedian(OCO−ship):  original {d_orig:+.2f}  →  corrected {d_corr:+.2f} ppm")
    a.legend(fontsize=9)

    # (3) nearest-cloud distance of collocated footprints (over the MODIS clouds)
    a = ax[1, 0]
    _bg(a)
    sc = a.scatter(near.lon, near.lat, c=near.cld_dist_km, s=14, cmap="viridis_r",
                   vmin=0, vmax=50, zorder=3)
    a.scatter(ship.lon, ship.lat, c="red", s=25, marker="D", zorder=6)
    a.set(xlim=args.lon_range, ylim=args.lat_range, xlabel="Lon (°E)", ylabel="Lat (°N)",
          title=f"Nearest-cloud distance  (≤10 km: {(near.cld_dist_km<=10).sum()}/{len(near)})")
    fig.colorbar(sc, ax=a, label="km")

    # (4) ship time series
    a = ax[1, 1]
    st_t = pd.to_datetime(ship.epoch, unit="s", utc=True)
    a.scatter(st_t, ship.xco2, s=12, color="tab:red", alpha=0.6, label="ship XCO₂")
    ot_t = pd.to_datetime(near.time, unit="s", utc=True)
    a.axvspan(ot_t.min(), ot_t.max(), color="tab:blue", alpha=0.25, label="OCO-2 overpass")
    a.axhline(np.nanmedian(o_corr), color="tab:green", lw=1.5, ls="--", label="OCO-2 corrected median")
    a.set(xlabel="UTC", ylabel="XCO₂ (ppm)", title=f"ship XCO₂ time series  (n={len(ship)})")
    a.legend(fontsize=8); a.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.output_dir, f"ship_comparison_{args.ship_tag}_{args.date}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  n_fp={len(near)}  ship_pts_in_window={len(ship_w)}")
    print(f"  OCO original median {np.nanmedian(o_orig):.2f}  corrected {np.nanmedian(o_corr):.2f}  "
          f"ship {np.nanmedian(s_ref):.2f} ppm")
    print(f"  Δmedian(OCO−ship): original {d_orig:+.2f} → corrected {d_corr:+.2f} ppm")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
