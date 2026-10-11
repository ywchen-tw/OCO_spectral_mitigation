#!/usr/bin/env python
"""Emit run_case lines for the 27 new OCO-2/TCCON station-days.

Same derivation as gen_drift_cases.py, but the (station, date) pairing is given
explicitly instead of being searched for: the box is the floor/ceil-to-0.01-deg
hull of the footprints within RADIUS_KM of the station, vmin/vmax are the 2nd/98th
xco2_bc percentiles of those footprints (rounded outwards to 0.5 ppm, floored at a
2 ppm span), and SURF is the dominant sfc_type.  Station lon/lat are the medians of
the TCCON file's long/lat variables.

Writes two launcher-format case lists, split on src.constants.AQUA_FREE_DRIFT_YEAR:
    workspace/newsites_cases_atrain.sh   (dates  < 2022)
    workspace/newsites_cases_drift.sh    (dates >= 2022)
"""
import glob
import os
import sys

import numpy as np
import netCDF4 as nc
import pandas as pd

ROOT = "/Users/yuch8913/programming/oco_fp_analysis"
sys.path.insert(0, os.path.join(ROOT, "src"))
from constants import AQUA_FREE_DRIFT_YEAR  # noqa: E402

TCCON_DIR = os.path.join(ROOT, "data/TCCON")
CSV_DIR = os.path.join(ROOT, "results/csv_collection")
RADIUS_KM = 100.0
WINDOW_MIN = 60.0

# (2-letter TCCON code, date) pairs verified to have >=20 footprints within
# RADIUS_KM and >=1 TCCON observation within +/-WINDOW_MIN of the pass.
PAIRS = [
    ("gm", "2015-02-13"), ("gm", "2015-02-18"), ("gm", "2015-07-14"),
    ("gm", "2017-01-26"), ("gm", "2019-06-14"), ("gm", "2023-07-04"),
    ("so", "2016-05-29"), ("so", "2018-09-08"), ("so", "2018-10-10"),
    ("so", "2019-06-09"), ("so", "2020-03-30"),
    ("tk", "2015-03-17"), ("tk", "2016-03-03"), ("tk", "2017-01-17"),
    ("tk", "2018-04-10"),
    ("lr", "2019-06-22"), ("lr", "2021-05-26"), ("lr", "2021-09-08"),
    ("lr", "2023-03-13"),
    ("bi", "2015-03-23"), ("bi", "2015-07-15"), ("bi", "2018-06-03"),
    ("br", "2018-08-07"),
    ("hw", "2021-09-08"),
    ("ll", "2016-09-10"),
    ("ni", "2020-12-23"),
    ("hf", "2024-05-10"),
]


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    dlat = (lat2 - lat1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def load_station(code):
    hits = sorted(glob.glob(os.path.join(TCCON_DIR, f"{code}*.public.qc.nc")))
    if len(hits) != 1:
        raise SystemExit(f"{code}: expected 1 TCCON file, found {len(hits)}: {hits}")
    d = nc.Dataset(hits[0])
    lat = float(np.nanmedian(d.variables["lat"][:]))
    lon = float(np.nanmedian(d.variables["long"][:]))
    t = np.asarray(d.variables["time"][:], dtype="float64")
    name = getattr(d, "long_name", code)
    d.close()
    return dict(file=os.path.basename(hits[0]), code=code, name=name,
                lon=lon, lat=lat, t=t)


def main():
    stations = {}
    for code, _ in PAIRS:
        if code not in stations:
            stations[code] = load_station(code)

    dfs = {}
    rows = []
    for code, date in PAIRS:
        s = stations[code]
        if date not in dfs:
            pq = os.path.join(CSV_DIR, f"combined_{date}_all_orbits.parquet")
            if not os.path.isfile(pq):
                raise SystemExit(f"{date}: parquet missing: {pq}")
            dfs[date] = pd.read_parquet(
                pq, columns=["lon", "lat", "xco2_bc", "time", "sfc_type"])
        df = dfs[date]
        lon = df["lon"].to_numpy("float64")
        lat = df["lat"].to_numpy("float64")
        dist = haversine_km(lon, lat, s["lon"], s["lat"])
        m = dist <= RADIUS_KM
        n = int(m.sum())
        if n < 20:
            raise SystemExit(f"{code} {date}: only {n} footprints within {RADIUS_KM:g} km")
        sub = df[m]
        lon_s, lat_s = lon[m], lat[m]
        lonmin = np.floor(lon_s.min() * 100) / 100
        lonmax = np.ceil(lon_s.max() * 100) / 100
        latmin = np.floor(lat_s.min() * 100) / 100
        latmax = np.ceil(lat_s.max() * 100) / 100

        x = sub["xco2_bc"].to_numpy("float64")
        x = x[np.isfinite(x)]
        vmin = np.floor(np.nanpercentile(x, 2) * 2) / 2
        vmax = np.ceil(np.nanpercentile(x, 98) * 2) / 2
        if vmax - vmin < 2.0:
            vmax = vmin + 2.0

        sfc = sub["sfc_type"].to_numpy("float64")
        frac_land = np.nanmean(sfc == 1)
        surf = "land" if frac_land > 0.9 else "ocean" if frac_land < 0.1 else "both"

        ot = sub["time"].to_numpy("float64")
        ot = ot[np.isfinite(ot)]
        w = WINDOW_MIN * 60.0
        lo, hi = ot.min() - w, ot.max() + w
        avail = "yes" if np.any((s["t"] >= lo) & (s["t"] <= hi)) else "no"
        if avail != "yes":
            print(f"# WARN {code} {date}: no TCCON obs within +/-{WINDOW_MIN:g} min",
                  file=sys.stderr)

        rows.append(dict(date=date, file=s["file"], code=code, name=s["name"],
                         lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax,
                         vmin=vmin, vmax=vmax, surf=surf, avail="yes", n=n,
                         dmin=float(dist.min()),
                         era="drift" if int(date[:4]) >= AQUA_FREE_DRIFT_YEAR else "atrain"))

    fmt = ("run_case  {date}   {file:<34} {lonmin:8.2f} {lonmax:8.2f} "
           "{latmin:8.2f} {latmax:8.2f}  {vmin:6.1f} {vmax:6.1f}  {surf:<5} poster "
           "{code}  {avail}   # n={n} dmin={dmin:.0f}km")

    for era, out in (("atrain", "workspace/newsites_cases_atrain.sh"),
                     ("drift", "workspace/newsites_cases_drift.sh")):
        sel = sorted((r for r in rows if r["era"] == era),
                     key=lambda r: (r["code"], r["date"]))
        lines = [
            "#!/bin/env bash",
            f"# run_case list for the {era}-era new TCCON station-days "
            f"({len(sel)} cases).",
            "# Generated by workspace/newsites_cases.py; consumed by",
            "# tccon_comparison_report.py --script (only the run_case lines are read).",
            f"# RADIUS_KM={RADIUS_KM:g}  WINDOW_MIN={WINDOW_MIN:g}  "
            f"AQUA_FREE_DRIFT_YEAR={AQUA_FREE_DRIFT_YEAR}",
            "#" + "-" * 78,
        ]
        last = None
        for r in sel:
            if r["code"] != last:
                lines.append(f"\n# -- {r['name']} ({r['code']}) --")
                last = r["code"]
            lines.append(fmt.format(**r))
        path = os.path.join(ROOT, out)
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {out}: {len(sel)} run_case lines")


if __name__ == "__main__":
    main()
