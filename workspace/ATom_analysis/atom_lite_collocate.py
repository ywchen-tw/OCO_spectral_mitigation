#!/usr/bin/env python3
"""Screen ATom flight dates for OCO-2 ocean-glint coincidences, using Lite files only.

Pre-flight gate for the ATom pseudo-column comparison: before processing any date
through the full pipeline (expensive), find which ATom flights actually have OCO-2
*ocean-glint, good-quality* soundings near the flight track and close in time.

For each ATom merged flight ($OUT/atom_merged/atom_merged_<date>.parquet from
merge_atom_profiles.py) this:
  1. converts ATom time (sec since midnight UTC) to epoch and groups points by the
     true UTC day (handles flights that cross midnight);
  2. pulls that day's global OCO-2 L2 Lite granule (cached, shared with the ship scan);
  3. keeps ocean-glint good-quality soundings
        operation_mode==1 (Glint) AND land_water_indicator==1 (ocean) AND xco2_quality_flag==0;
  4. counts soundings within {50,100,250} km and {2,6,24} h of any track point;
  5. prints a ranked table and writes $OUT/atom_oco2_collocation.csv.

Reuses the download/CMR/haversine approach of workspace/ship_lite_collocate.py.

Usage:
    python atom_lite_collocate.py                # all merged flights
    python atom_lite_collocate.py --date 20171001
    python atom_lite_collocate.py --no-download  # only use already-cached Lite files
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
OUT_BASE = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
MERGED_DIR = os.path.join(OUT_BASE, "atom_merged")   # merged profiles (input)
# Share the Lite cache with ship_lite_collocate.py.
LITE = os.path.join(REPO, "data", "Other", "lite_cache")
os.makedirs(LITE, exist_ok=True)

EPOCH = dt.datetime(1970, 1, 1)
RADII = [50, 100, 250]          # km
TWINS = [2 * 3600, 6 * 3600, 24 * 3600]  # s  (24h == same-day, effectively no time gate)
TWIN_LABEL = {2 * 3600: "2h", 6 * 3600: "6h", 24 * 3600: "24h"}
BBOX_PAD_DEG = 4.0              # spatial pre-filter pad around the flight track


# ---- OCO-2 Lite fetch (CMR resolve + streamed download, cached) --------------
def resolve_url(ymd: str) -> str | None:
    d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    r = requests.get(
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        params={"short_name": "OCO2_L2_Lite_FP",
                "temporal": f"{d}T00:00:00Z,{d}T23:59:59Z", "page_size": 20},
        timeout=60,
    )
    r.raise_for_status()
    yy = ymd[2:]
    for e in r.json()["feed"]["entry"]:
        if f"LtCO2_{yy}_" in e["title"]:
            for l in e.get("links", []):
                h = l.get("href", "")
                if h.endswith(".nc4") and "gesdisc" in h and "/data/" in h:
                    return h
    return None


def fetch(ymd: str, allow_download: bool = True) -> str | None:
    dest = os.path.join(LITE, f"lite_{ymd}.nc4")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    if not allow_download:
        return None
    url = resolve_url(ymd)
    if not url:
        return None
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    with s.get(url, stream=True, timeout=600, allow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for c in resp.iter_content(1 << 20):
                f.write(c)
    return dest


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lo2 - lo1
    dlat = la2 - la1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---- read ocean-glint good-quality soundings from a Lite granule ------------
def read_ocean_glint(path: str):
    ds = nc.Dataset(path)
    lon = ds.variables["longitude"][:].astype(float)
    lat = ds.variables["latitude"][:].astype(float)
    t = ds.variables["time"][:].astype(float)          # epoch seconds
    qf = ds.variables["xco2_quality_flag"][:].astype(int)
    xco2 = ds.variables["xco2"][:].astype(float)
    mode = ds.groups["Sounding"].variables["operation_mode"][:].astype(int)   # 0 nadir,1 glint
    lwi = ds.groups["Sounding"].variables["land_water_indicator"][:].astype(int)  # 0 land,1 water
    ds.close()
    keep = (mode == 1) & (lwi == 1) & (qf == 0)
    return lon[keep], lat[keep], t[keep], xco2[keep]


# ---- load ATom track points (epoch, lon, lat) grouped by true UTC day -------
def load_atom_points(date_filter: str | None) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(MERGED_DIR, "atom_merged_*.parquet"))):
        fdate = re.search(r"atom_merged_(\d{8})\.parquet", os.path.basename(p)).group(1)
        if date_filter and fdate != date_filter:
            continue
        df = pd.read_parquet(p, columns=["time_utc_s", "lat", "lon"])
        base = (dt.datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:])) - EPOCH).total_seconds()
        df["epoch"] = base + df["time_utc_s"].to_numpy()
        df["flight_date"] = fdate
        rows.append(df[["flight_date", "epoch", "lon", "lat"]])
    if not rows:
        raise SystemExit("No merged ATom parquets found -- run merge_atom_profiles.py first.")
    allp = pd.concat(rows, ignore_index=True)
    allp["utc_day"] = ((allp["epoch"] // 86400) * 86400).map(
        lambda s: dt.datetime.utcfromtimestamp(s).strftime("%Y%m%d"))
    return allp


# ---- collocate one UTC day's OCO soundings against that day's track points ---
def collocate_day(utc_day: str, pts: pd.DataFrame, allow_download: bool):
    path = fetch(utc_day, allow_download)
    if not path:
        return None, "no granule"
    olon, olat, ot, oxco2 = read_ocean_glint(path)
    if olon.size == 0:
        return dict(counts={(r, w): 0 for r in RADII for w in TWINS}), "no ocean-glint"

    alon = pts["lon"].to_numpy(); alat = pts["lat"].to_numpy(); at = pts["epoch"].to_numpy()
    # spatial pre-filter: OCO soundings within padded bbox of the track
    box = (olat >= alat.min() - BBOX_PAD_DEG) & (olat <= alat.max() + BBOX_PAD_DEG) & \
          (olon >= alon.min() - BBOX_PAD_DEG) & (olon <= alon.max() + BBOX_PAD_DEG)
    olon, olat, ot, oxco2 = olon[box], olat[box], ot[box], oxco2[box]
    counts = {(r, w): 0 for r in RADII for w in TWINS}
    best = dict(km=np.inf)
    if olon.size == 0:
        return dict(counts=counts, best=best, n_glint=0), None

    for i0 in range(0, olon.size, 300):
        sl = slice(i0, i0 + 300)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None],
                         alon[None, :], alat[None, :])            # (chunk, n_atom)
        tg = np.abs(ot[sl][:, None] - at[None, :])                # s
        for w in TWINS:
            dm = np.where(tg <= w, d, np.inf)
            nn = dm.min(axis=1)                                   # nearest in-window track pt
            for r in RADII:
                counts[(r, w)] += int((nn <= r).sum())
        # closest sounding within the widest window, for eyeballing
        dm = np.where(tg <= TWINS[-1], d, np.inf)
        nn = dm.min(axis=1)
        if np.isfinite(nn.min()):
            bi = int(nn.argmin())
            if nn[bi] < best["km"]:
                jj = int(dm[bi].argmin())
                best = dict(km=float(nn[bi]), tgap_min=abs(ot[sl][bi] - at[jj]) / 60.0,
                            xco2=float(oxco2[sl][bi]))
    return dict(counts=counts, best=best, n_glint=int(olon.size)), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="single ATom flight YYYYMMDD")
    ap.add_argument("--no-download", action="store_true", help="use only cached Lite files")
    args = ap.parse_args()
    allow_dl = not args.no_download

    pts = load_atom_points(args.date)
    utc_days = sorted(pts["utc_day"].unique())
    print(f"{pts['flight_date'].nunique()} ATom flights span {len(utc_days)} UTC days; "
          f"resolving OCO-2 Lite granules...")
    if allow_dl:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda d: fetch(d, True), utc_days))

    # collocate per UTC day, then aggregate to the flight start date
    per_flight: dict[str, dict] = {}
    for utc_day, g in pts.groupby("utc_day"):
        res, err = collocate_day(utc_day, g, allow_dl)
        if res is None:
            continue
        # a UTC day maps to exactly one flight in practice (flights don't overlap days)
        for fdate in g["flight_date"].unique():
            acc = per_flight.setdefault(fdate, dict(counts={(r, w): 0 for r in RADII for w in TWINS},
                                                    best=dict(km=np.inf)))
            for k, v in res["counts"].items():
                acc["counts"][k] += v
            b = res.get("best", {})
            if b.get("km", np.inf) < acc["best"]["km"]:
                acc["best"] = b

    # ---- table ----
    hdr = (f"{'flight':>10} {'min_km':>7} {'tgap_m':>7} {'xco2':>7} "
           f"{'100/2h':>7} {'250/2h':>7} {'100/6h':>7} {'100/24h':>8} {'250/24h':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    rows_csv = []
    for fdate in sorted(per_flight):
        c = per_flight[fdate]["counts"]; b = per_flight[fdate]["best"]
        mk = b.get("km", np.inf)
        print(f"{fdate:>10} {mk:7.1f} {b.get('tgap_min', float('nan')):7.0f} "
              f"{b.get('xco2', float('nan')):7.2f} "
              f"{c[(100, 7200)]:7d} {c[(250, 7200)]:7d} {c[(100, 21600)]:7d} "
              f"{c[(100, 86400)]:8d} {c[(250, 86400)]:8d}")
        row = dict(flight_date=fdate, min_km=round(mk, 1),
                   tgap_min=round(b.get("tgap_min", float("nan")), 1),
                   best_xco2=round(b.get("xco2", float("nan")), 2))
        for r in RADII:
            for w in TWINS:
                row[f"n_{r}km_{TWIN_LABEL[w]}"] = c[(r, w)]
        rows_csv.append(row)

    df = pd.DataFrame(rows_csv)
    os.makedirs(OUT_BASE, exist_ok=True)
    csv_path = os.path.join(OUT_BASE, "atom_oco2_collocation.csv")
    df.to_csv(csv_path, index=False)

    # ---- ranked recommendation ----
    good = df[df["n_100km_2h"] >= 1].sort_values("n_100km_2h", ascending=False)
    print(f"\n=== dates with >=1 ocean-glint good-QF sounding within 100 km & +/-2h "
          f"({len(good)} of {len(df)}) ===")
    for _, r in good.iterrows():
        print(f"  {r['flight_date']}  n(100km,2h)={int(r['n_100km_2h']):4d}  "
              f"closest={r['min_km']:.1f} km @ {r['tgap_min']:.0f} min")
    loose = df[(df["n_100km_2h"] == 0) & (df["n_250km_24h"] >= 1)]
    print(f"\n{len(loose)} more dates have same-day soundings within 250 km but none within "
          f"100 km/+/-2h (loose spot-check only).")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
