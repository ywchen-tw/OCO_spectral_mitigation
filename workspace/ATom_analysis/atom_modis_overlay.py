#!/usr/bin/env python3
"""Per-date MODIS Aqua overlay for the ATom↔OCO-2 collocation.

Mirrors plot_corrected_xco2.py's --modis-auto map: a MODIS Aqua true-colour
composite (NASA GIBS) as background, the OCO-2 ocean-glint soundings scattered on
top (coloured by DE-corrected XCO2), and the ATom flight track overlaid — so the
aircraft profile, the satellite soundings, and the actual cloud field are all in
one frame.

Reuses download_modis_rgb() from workspace/plot_corrected_xco2.py (GIBS, Aqua).

Output: $OUT/combined_<date>_atom/atom_modis_<date>.png  (per usable date)

Usage: python atom_modis_overlay.py [--date 2017-10-20] [--radius-km 100]
                                    [--window-min 120] [--pad 0.5] [--no-modis]
"""
from __future__ import annotations
import argparse, datetime as dt, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # workspace/
from plot_corrected_xco2 import download_modis_rgb       # noqa: E402
from ak_harmonize import _haversine_km                   # noqa: E402
from plot_style import apply_manuscript_style, CMAPS, XCO2_LABEL, XCO2_DE_LABEL  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_DIR = os.path.join(REPO, "results", "csv_collection")
TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
PLOT_BASE = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
MERGED_DIR = os.path.join(PLOT_BASE, "atom_merged")     # merged profiles (input)
TILES_DIR = os.path.join(PLOT_BASE, "_modis_tiles")     # cached GIBS tiles
EPOCH = dt.datetime(1970, 1, 1)
DATES = ["2017-01-26", "2017-02-04", "2017-02-06", "2017-02-10", "2017-10-09",
         "2017-10-20", "2017-10-27", "2018-05-12"]
CORR = "deep_ensemble_corrected_xco2"
# Flights whose OCO coincidence day (key) differs from the ATom flight (merged) date.
OCO_TO_FLIGHT = {"2017-02-04": "2017-02-03", "2017-02-06": "2017-02-05",
                 "2017-10-09": "2017-10-08"}


def load_atom(date):
    ymd = OCO_TO_FLIGHT.get(date, date).replace("-", "")
    df = pd.read_parquet(os.path.join(MERGED_DIR, f"atom_merged_{ymd}.parquet"),
                         columns=["time_utc_s", "lat", "lon", "alt_m", "profile_id"])
    base = (dt.datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])) - EPOCH).total_seconds()
    df["epoch"] = base + df["time_utc_s"].to_numpy()
    return df


def load_oco(date):
    pd_path = os.path.join(PLOT_BASE, f"combined_{date}_atom", "plot_data.parquet")
    oco = pd.read_parquet(pd_path, columns=["time", "lon", "lat", "sfc_type",
                                            "cld_dist_km", "xco2_bc", CORR])
    return oco[oco.sfc_type == 0].copy()


def collocate(atom, oco, radius_km, twin_s):
    """Boolean masks: OCO soundings and ATom points that collocate (100km/±2h)."""
    olon, olat, ot = oco.lon.values, oco.lat.values, oco.time.values
    alon, alat, at = atom.lon.values, atom.lat.values, atom.epoch.values
    oco_hit = np.zeros(len(oco), bool)
    atom_near = np.zeros(len(atom), bool)
    for i0 in range(0, len(oco), 400):
        sl = slice(i0, i0 + 400)
        d = _haversine_km(olon[sl][:, None], olat[sl][:, None], alon[None, :], alat[None, :])
        tg = np.abs(ot[sl][:, None] - at[None, :])
        dm = np.where(tg <= twin_s, d, np.inf)
        oco_hit[sl] = dm.min(axis=1) <= radius_km
        atom_near |= (dm <= radius_km).any(axis=0)
    return oco_hit, atom_near


def make_map(date, radius_km, twin_s, pad, do_modis):
    atom, oco = load_atom(date), load_oco(date)
    oco_hit, atom_near = collocate(atom, oco, radius_km, twin_s)
    sel = oco[oco_hit]
    if sel.empty:
        print(f"{date}: no collocated OCO soundings"); return
    # extent covers the collocated OCO box + the ATom track near it, padded
    an = atom[atom_near]
    lon0 = min(sel.lon.min(), an.lon.min()) - pad
    lon1 = max(sel.lon.max(), an.lon.max()) + pad
    lat0 = min(sel.lat.min(), an.lat.min()) - pad
    lat1 = max(sel.lat.max(), an.lat.max()) + pad
    extent = [lon0, lon1, lat0, lat1]

    # OCO soundings shown in-frame (swath context), coloured by corrected XCO2
    inx = oco[(oco.lon.between(lon0, lon1)) & (oco.lat.between(lat0, lat1))]
    vmin = np.floor(np.nanpercentile(sel[CORR], 5) * 2) / 2
    vmax = np.ceil(np.nanpercentile(sel[CORR], 95) * 2) / 2
    at_in = atom[(atom.lon.between(lon0, lon1)) & (atom.lat.between(lat0, lat1))]

    bg = None
    if do_modis:
        try:
            # reuse a cached tile if the exact date+extent PNG already exists
            os.makedirs(TILES_DIR, exist_ok=True)
            cached = download_modis_rgb(pd.Timestamp(date), extent, which="aqua",
                                        fdir=TILES_DIR, run=False)
            png = cached if os.path.exists(cached) else \
                download_modis_rgb(pd.Timestamp(date), extent, which="aqua", fdir=TILES_DIR)
            bg = plt.imread(png)
        except Exception as e:
            print(f"{date}: MODIS download failed ({e}); plotting without background")

    fig, ax = plt.subplots(figsize=(9, 8))
    if bg is not None:
        ax.imshow(bg, extent=extent, origin="upper", aspect="auto", zorder=0)
    # OCO soundings coloured by DE-corrected XCO2. Non-collocated in-frame soundings
    # are faint (context); the collocated subset is full-opacity + slightly larger.
    non = inx[~inx.index.isin(sel.index)]
    ax.scatter(non.lon, non.lat, c=non[CORR], s=9, cmap=CMAPS["xco2"],
               vmin=vmin, vmax=vmax, zorder=2, edgecolors="none", alpha=0.35)
    sc = ax.scatter(sel.lon, sel.lat, c=sel[CORR], s=16, cmap=CMAPS["xco2"],
                    vmin=vmin, vmax=vmax, zorder=3, edgecolors="none",
                    label=f"collocated ({len(sel)})")
    # ATom flight track (in-frame): single bright colour so it doesn't compete
    # with the XCO2 colourmap
    ax.plot(at_in.lon, at_in.lat, "-", color="magenta", lw=1.3, alpha=0.9, zorder=4,
            label="ATom track")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(f"OCO-2 {XCO2_DE_LABEL} (ppm)")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / max(np.cos(np.radians(0.5 * (lat0 + lat1))), 0.05))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"ATom × OCO-2 ocean-glint over MODIS Aqua — {date}\n"
                 f"{len(sel)} collocated soundings (≤{radius_km:.0f} km, ±{twin_s/60:.0f} min)")
    ax.legend(loc="upper right", fontsize=8)
    out = os.path.join(PLOT_BASE, f"combined_{date}_atom", f"atom_modis_{date}.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"{date}: wrote {out}  (extent {['%.1f'%e for e in extent]}, "
          f"{len(sel)} collocated)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="single date YYYY-MM-DD (default: all 5)")
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    ap.add_argument("--pad", type=float, default=0.5, help="extent padding (deg)")
    ap.add_argument("--no-modis", action="store_true", help="skip GIBS download")
    args = ap.parse_args()
    apply_manuscript_style()   # Arial (AMT), Arial mathtext, thin axes, 300 dpi
    dates = [args.date] if args.date else DATES
    for d in dates:
        make_map(d, args.radius_km, args.window_min * 60, args.pad, not args.no_modis)


if __name__ == "__main__":
    main()
