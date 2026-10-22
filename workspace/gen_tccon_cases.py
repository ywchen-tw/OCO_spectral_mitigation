#!/usr/bin/env python
"""Generate run_case lines (TCCON collocations) for the deepens plot scripts.

For each date's combined_<date>_all_orbits.parquet, every TCCON station with at
least --min-footprints OCO-2 footprints within --radius-km gets one run_case
line: lon/lat box of those footprints, vmin/vmax from xco2_bc percentiles, the
dominant surface, and TCCON_AVAIL = any station observation within
+/- --window-min of the overpass.  Lines are grouped by station and tagged
with the script era (A-Train: date < 2023 -> deepens.sh; drift: >= 2023 ->
deepens_drift.sh).  Generalises workspace/gen_drift_cases.py (one best station
per date, hard-coded local paths).

    CURC_DATA_ROOT=/scratch/alpine/<user>/oco/spec_anal \\
        python workspace/gen_tccon_cases.py 2020-07-03 2023-09-04 ...
"""
import argparse
import glob
import os
import re

import netCDF4 as nc
import numpy as np
import pandas as pd


def haversine_km(lon1, lat1, lon2, lat2):
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    dlat = (lat2 - lat1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_stations(tccon_dir):
    stations = []
    for f in sorted(glob.glob(os.path.join(tccon_dir, "*.nc"))):
        base = os.path.basename(f)
        m = re.match(r"([a-z]{2})(\d{8})_(\d{8})", base)
        if not m:
            continue
        try:
            d = nc.Dataset(f)
            lat = float(np.nanmedian(np.asarray(d.variables["lat"][:], dtype="float64")))
            lon = float(np.nanmedian(np.asarray(d.variables["long"][:], dtype="float64")))
            t = np.asarray(d.variables["time"][:], dtype="float64")   # s since 1970
            name = getattr(d, "long_name", m.group(1))
            d.close()
        except Exception as e:  # noqa: BLE001
            print(f"# WARN could not read {base}: {e}")
            continue
        stations.append(dict(file=base, code=m.group(1), name=name, lon=lon, lat=lat,
                             t=np.sort(t), start=m.group(2), end=m.group(3)))
    return stations


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dates", nargs="+", help="YYYY-MM-DD (or YYYYMMDD)")
    root = os.environ.get("CURC_DATA_ROOT") or os.environ.get("OCO2_DATAROOT") or "."
    ap.add_argument("--csv-dir", default=os.path.join(root, "results", "csv_collection"))
    ap.add_argument("--tccon-dir", default=os.path.join(root, "data", "TCCON"))
    ap.add_argument("--radius-km", type=float, default=100.0)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--min-footprints", type=int, default=10,
                    help="guard used by the TCCON comparison: footprints within radius")
    args = ap.parse_args()

    stations = load_stations(args.tccon_dir)
    print(f"# {len(stations)} TCCON stations from {args.tccon_dir}")
    print(f"# RADIUS_KM={args.radius_km:g}  WINDOW_MIN={args.window_min:g}  "
          f"MIN_FOOTPRINTS={args.min_footprints}")

    rows, unmatched = [], []
    for raw in args.dates:
        raw = raw.replace("-", "")
        date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        pq = os.path.join(args.csv_dir, f"combined_{date}_all_orbits.parquet")
        if not os.path.isfile(pq):
            print(f"# {date}: NO PARQUET {pq}")
            continue
        df = pd.read_parquet(pq, columns=["lon", "lat", "xco2_bc", "time", "sfc_type"])
        lon = df["lon"].to_numpy("float64")
        lat = df["lat"].to_numpy("float64")
        hit = False
        for s in stations:
            if not (s["start"] <= raw <= s["end"]):
                continue                      # date outside the station's record
            dist = haversine_km(lon, lat, s["lon"], s["lat"])
            m = dist <= args.radius_km
            n = int(m.sum())
            if n < args.min_footprints:
                continue
            hit = True
            sub = df[m]
            lonmin, lonmax = np.floor(lon[m].min() * 100) / 100, np.ceil(lon[m].max() * 100) / 100
            latmin, latmax = np.floor(lat[m].min() * 100) / 100, np.ceil(lat[m].max() * 100) / 100
            x = sub["xco2_bc"].to_numpy("float64")
            x = x[np.isfinite(x) & (x > 300.0) & (x < 1000.0)]   # drop fill/outlier values
            if x.size:
                vmin = np.floor(np.nanpercentile(x, 2) * 2) / 2
                vmax = np.ceil(np.nanpercentile(x, 98) * 2) / 2
                if vmax - vmin < 2.0:
                    vmax = vmin + 2.0
            else:
                vmin, vmax = 400.0, 420.0
            sfc = sub["sfc_type"].to_numpy("float64")
            frac_land = float(np.nanmean(sfc == 1))
            surf = "land" if frac_land > 0.9 else "ocean" if frac_land < 0.1 else "both"
            ot = sub["time"].to_numpy("float64")
            ot = ot[np.isfinite(ot)]
            w = args.window_min * 60.0
            lo, hi = ot.min() - w, ot.max() + w
            i0, i1 = np.searchsorted(s["t"], [lo, hi])
            n_tccon = int(i1 - i0)
            rows.append(dict(date=date, file=s["file"], code=s["code"], name=s["name"],
                             slat=s["lat"], slon=s["lon"],
                             lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax,
                             vmin=vmin, vmax=vmax, surf=surf,
                             avail="yes" if n_tccon > 0 else "no", n_tccon=n_tccon,
                             n=n, dmin=float(dist[m].min()),
                             era="drift" if int(raw[:4]) >= 2023 else "atrain"))
        if not hit:
            unmatched.append(date)

    for era, label in (("atrain", "A-Train era -> curc_shell_blanca_plot_corr_xco2_deepens.sh"),
                       ("drift", "post-drift -> curc_shell_blanca_plot_corr_xco2_deepens_drift.sh")):
        sel = sorted((r for r in rows if r["era"] == era), key=lambda r: (r["code"], r["date"]))
        if not sel:
            continue
        print(f"\n# ======== {label} ========")
        last = None
        for r in sel:
            if r["code"] != last:
                print(f"\n# ── {r['name']} ({r['code']}; lat {r['slat']:.2f}, lon {r['slon']:.2f}) ──")
                last = r["code"]
            print("run_case  {date}   {file:<36} {lonmin:8.2f} {lonmax:8.2f} "
                  "{latmin:8.2f} {latmax:8.2f}  {vmin:6.1f} {vmax:6.1f}  {surf:<5} poster {code}  "
                  "{avail:<3}   # n={n} dmin={dmin:.0f}km tccon_obs={n_tccon}".format(**r))

    print(f"\n# {len(rows)} (date, station) collocations; "
          f"{sum(r['avail'] == 'yes' for r in rows)} with TCCON within +/-{args.window_min:g} min")
    if unmatched:
        print(f"# no station within {args.radius_km:g} km: {' '.join(unmatched)}")


if __name__ == "__main__":
    main()
