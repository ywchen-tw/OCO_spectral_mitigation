#!/usr/bin/env python
"""Generate run_case lines for the post-Aqua-drift deepens script.

For each local combined_<date>_all_orbits.parquet it finds the TCCON station the
orbit overpasses (most footprints within RADIUS_KM), derives the lon/lat box,
vmin/vmax (xco2_bc percentiles), the dominant surface, and the TCCON_AVAIL flag
(any TCCON obs within +/-WINDOW_MIN of the overpass), then prints a run_case line.
"""
import glob, os, re
import numpy as np, pandas as pd
import netCDF4 as nc

ROOT = "/Users/yuch8913/programming/oco_fp_analysis"
TCCON_DIR = os.path.join(ROOT, "data/TCCON")
CSV_DIR = os.path.join(ROOT, "results/csv_collection")
RADIUS_KM = 100.0
WINDOW_MIN = 60.0

DATES = """
20230123 20230128 20230210 20230313 20230319 20230321 20230323
20230404 20230425 20230506 20230520 20230526
20230611 20230626 20230704 20230720 20230804 20230810 20230814
20230902 20230911 20231011 20231013
20240206 20240218 20240310 20240415 20240422 20240510 20240626
20240727 20240731 20240803 20240826 20240920 20241003 20241029
20241123 20241202 20241216
""".split()


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    dlat = (lat2 - lat1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ── station catalog ───────────────────────────────────────────────────────────
stations = []  # dict: file, code, name, lon, lat, t (np array of obs epoch sec)
for f in sorted(glob.glob(os.path.join(TCCON_DIR, "*.nc"))):
    base = os.path.basename(f)
    code = base[:2]
    try:
        d = nc.Dataset(f)
        lat = float(np.nanmedian(d.variables["lat"][:]))
        lon = float(np.nanmedian(d.variables["long"][:]))
        t = np.asarray(d.variables["time"][:], dtype="float64")
        name = getattr(d, "long_name", code)
        d.close()
    except Exception as e:
        print(f"# WARN could not read {base}: {e}")
        continue
    stations.append(dict(file=base, code=code, name=name, lon=lon, lat=lat, t=t))

print(f"# {len(stations)} TCCON stations loaded")
print(f"# RADIUS_KM={RADIUS_KM:g}  WINDOW_MIN={WINDOW_MIN:g}")
print("#" + "-" * 78)

rows = []
for raw in DATES:
    date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    pq = os.path.join(CSV_DIR, f"combined_{date}_all_orbits.parquet")
    if not os.path.isfile(pq):
        print(f"# {date}: NO PARQUET")
        continue
    df = pd.read_parquet(pq, columns=["lon", "lat", "xco2_bc", "time", "sfc_type"])
    lon = df["lon"].to_numpy("float64")
    lat = df["lat"].to_numpy("float64")

    best = None
    for s in stations:
        # date must be within the station's record
        if s["t"].size == 0:
            continue
        dist = haversine_km(lon, lat, s["lon"], s["lat"])
        m = dist <= RADIUS_KM
        n = int(m.sum())
        if n < 20:
            continue
        if best is None or n > best["n"]:
            best = dict(s=s, n=n, mask=m, dist=dist)

    if best is None:
        print(f"# {date}: no TCCON station with >=20 footprints within {RADIUS_KM:g} km")
        continue

    s = best["s"]
    m = best["mask"]
    sub = df[m]
    lon_s, lat_s = lon[m], lat[m]
    lonmin, lonmax = np.floor(lon_s.min() * 100) / 100, np.ceil(lon_s.max() * 100) / 100
    latmin, latmax = np.floor(lat_s.min() * 100) / 100, np.ceil(lat_s.max() * 100) / 100

    x = sub["xco2_bc"].to_numpy("float64")
    x = x[np.isfinite(x)]
    vmin = np.floor(np.nanpercentile(x, 2) * 2) / 2
    vmax = np.ceil(np.nanpercentile(x, 98) * 2) / 2
    if vmax - vmin < 2.0:
        vmax = vmin + 2.0

    sfc = sub["sfc_type"].to_numpy("float64")
    frac_land = np.nanmean(sfc == 1)
    surf = "land" if frac_land > 0.9 else "ocean" if frac_land < 0.1 else "both"

    # TCCON_AVAIL: any station obs within +/- WINDOW_MIN of the overpass window
    ot = sub["time"].to_numpy("float64")
    ot = ot[np.isfinite(ot)]
    w = WINDOW_MIN * 60.0
    lo, hi = ot.min() - w, ot.max() + w
    st = s["t"]
    avail = "yes" if np.any((st >= lo) & (st <= hi)) else "no"

    rows.append(dict(date=date, file=s["file"], code=s["code"], name=s["name"],
                     lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax,
                     vmin=vmin, vmax=vmax, surf=surf, avail=avail, n=best["n"],
                     dmin=float(best["dist"].min())))

# ── emit run_case lines, grouped by station ─────────────────────────────────────
rows.sort(key=lambda r: (r["code"], r["date"]))
print()
last = None
for r in rows:
    if r["code"] != last:
        print(f"\n# -- {r['name']} ({r['code']}; lat {r['latmin']:.2f}..{r['latmax']:.2f}) --")
        last = r["code"]
    print("run_case  {date}   {file:<34} {lonmin:8.2f} {lonmax:8.2f} "
          "{latmin:8.2f} {latmax:8.2f}  {vmin:6.1f} {vmax:6.1f}  {surf:<5} poster {code}  {avail}"
          "   # n={n} dmin={dmin:.0f}km".format(**r))

print(f"\n# {len(rows)}/{len(DATES)} dates matched a station")
navail = sum(r["avail"] == "yes" for r in rows)
print(f"# {navail} have TCCON within +/-{WINDOW_MIN:g} min (AVAIL=yes)")
