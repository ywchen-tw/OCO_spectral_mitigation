#!/usr/bin/env python3
"""Stage 1.5 — footprint-level ship<->OCO-2 collocation on the PROCESSED parquets.

The Lite screen (ship_lite_collocate.py) told us which ship days coincide with an
OCO-2 ocean-glint overpass.  This finds the actual processed OCO-2 footprints within
RADIUS_KM & TIME_WIN of the ship track for each date, and reports per date:
  * ocean-glint good-QF footprint bounding box  (-> ship_case LON/LAT range),
  * VMIN/VMAX suggestion (2nd/98th pct of xco2_bc, rounded to 0.5),
  * cloud-distance coverage (the near-cloud second filter): how many <=10 km, and
    min/median/max cld_dist_km,
  * OCO-2 vs ship XCO2 medians of the collocated set.

Prints a ready-to-paste ship_case-style block per date (for curc_shell_blanca_ship_deepens.sh).

Usage: python ship_footprint_collocate.py [--radius-km 100] [--window-min 120]
"""
from __future__ import annotations
import argparse, datetime as dt, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_DIR = os.path.join(REPO, "results", "csv_collection")
EPOCH = dt.datetime(1970, 1, 1)

TABS = {
    "so268":  (os.path.join(REPO, "data/Other/SO268-3_track_XCO2_XCH4_XCO.tab"), "MORE-2"),
    "mr2101": (os.path.join(REPO, "data/Other/Hanft_2021_XCO2_XCH4_XCO.tab"),   "MR21-01"),
}
# ship, YYYY-MM-DD — the dates that clear the Lite screen (output/process_dates.txt).
# 2021-03-18 excluded: nearest OCO pass ~212 km, 0 footprints <=150 km.
DATES = [
    ("so268",  "2019-06-09"),
    ("so268",  "2019-06-14"),
    ("so268",  "2019-06-22"),
    ("mr2101", "2021-03-15"),
]


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def ship_track(tag, date):
    """Ship (epoch, lon, lat, xco2) rows for a given YYYY-MM-DD."""
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
    return np.array(rows, float)


def collocate(tag, date, radius_km, twin_s):
    sh = ship_track(tag, date)
    if sh.size == 0:
        return dict(date=date, tag=tag, err="no ship points that day")
    pq = os.path.join(CSV_DIR, f"combined_{date}_all_orbits.parquet")
    if not os.path.exists(pq):
        return dict(date=date, tag=tag, err=f"no processed parquet: combined_{date}_all_orbits.parquet")
    oco = pd.read_parquet(pq, columns=["time", "lon", "lat", "sfc_type",
                                       "xco2_qf", "xco2_bc", "cld_dist_km"])
    oco = oco[(oco.sfc_type == 0) & (oco.xco2_qf == 0)]           # ocean glint, good QF
    st, slon, slat, sx = sh[:, 0], sh[:, 1], sh[:, 2], sh[:, 3]
    pad = radius_km / 111.0 + 0.5
    oco = oco[oco.lat.between(slat.min()-pad, slat.max()+pad)]
    if (slon.max() - slon.min()) < 180:      # skip lon crop for dateline-crossing tracks
        oco = oco[oco.lon.between(slon.min()-pad, slon.max()+pad)]
    if oco.empty:
        return dict(date=date, tag=tag, err="no ocean-glint good-QF footprints near track")

    olon = oco.lon.to_numpy(); olat = oco.lat.to_numpy(); ot = oco.time.to_numpy()
    hit = np.zeros(len(oco), bool); dmin = np.full(len(oco), np.inf)
    for i0 in range(0, len(oco), 200):
        sl = slice(i0, i0+200)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None], slon[None, :], slat[None, :])
        tg = np.abs(ot[sl][:, None] - st[None, :])
        dm = np.where(tg <= twin_s, d, np.inf).min(axis=1)
        dmin[sl] = dm; hit[sl] = dm <= radius_km
    sel = oco[hit].copy()
    if sel.empty:
        return dict(date=date, tag=tag, err=f"0 footprints within {radius_km:.0f}km/{twin_s/60:.0f}min",
                    closest_km=float(dmin.min()))
    cld = sel.cld_dist_km.to_numpy(); xb = sel.xco2_bc.to_numpy()
    return dict(
        date=date, tag=tag, n=len(sel),
        lon=(float(sel.lon.min()), float(sel.lon.max())),
        lat=(float(sel.lat.min()), float(sel.lat.max())),
        vmin=np.floor(np.nanpercentile(xb, 2)*2)/2, vmax=np.ceil(np.nanpercentile(xb, 98)*2)/2,
        cld_min=float(np.nanmin(cld)), cld_med=float(np.nanmedian(cld)),
        cld_max=float(np.nanmax(cld)), n_near=int((cld <= 10).sum()),
        closest_km=float(dmin.min()),
        oco_xco2=float(np.nanmedian(xb)), ship_xco2=float(np.nanmedian(sx)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    args = ap.parse_args()
    twin = args.window_min * 60

    print(f"# ship<->OCO-2 footprint collocation  ({args.radius_km:.0f} km / "
          f"+/-{args.window_min:.0f} min, ocean-glint good-QF)\n")
    hdr = (f"{'date':>12} {'ship':>7} {'n_fp':>5} {'near<=10km':>10} {'cld_min':>8} {'cld_med':>8} "
           f"{'lon_min':>8} {'lon_max':>8} {'lat_min':>8} {'lat_max':>8} {'vmin':>6} {'vmax':>6} "
           f"{'OCO':>7} {'ship':>7}")
    print(hdr); print("-"*len(hdr))
    case_lines = []
    for tag, d in DATES:
        r = collocate(tag, d, args.radius_km, twin)
        if "err" in r:
            note = r["err"] + (f" (closest {r['closest_km']:.0f} km)" if "closest_km" in r else "")
            print(f"{d:>12} {tag:>7}  --  {note}")
            continue
        print(f"{r['date']:>12} {tag:>7} {r['n']:>5} {r['n_near']:>10} {r['cld_min']:>8.1f} "
              f"{r['cld_med']:>8.1f} {r['lon'][0]:>8.2f} {r['lon'][1]:>8.2f} {r['lat'][0]:>8.2f} "
              f"{r['lat'][1]:>8.2f} {r['vmin']:>6.1f} {r['vmax']:>6.1f} "
              f"{r['oco_xco2']:>7.2f} {r['ship_xco2']:>7.2f}")
        case_lines.append(
            f"ship_case  {r['date']}  {tag:7} {r['lon'][0]:8.2f} {r['lon'][1]:8.2f} "
            f"{r['lat'][0]:8.2f} {r['lat'][1]:8.2f}  {r['vmin']:6.1f} {r['vmax']:6.1f}  "
            f'"{r["n"]} fp, {r["n_near"]} near<=10km"')
    print("\n# ─ ready-to-paste ship_case lines (curc_shell_blanca_ship_deepens.sh) ─")
    print("#          DATE(OCO)   SHIP   LON_MIN  LON_MAX  LAT_MIN  LAT_MAX   VMIN   VMAX  NOTE")
    for ln in case_lines:
        print(ln)


if __name__ == "__main__":
    main()
