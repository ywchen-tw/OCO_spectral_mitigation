#!/usr/bin/env python3
"""Footprint-level ATom<->OCO-2 collocation on the PROCESSED parquets.

Stage 1.5b: the Lite screen (atom_lite_collocate.py) told us which dates coincide.
This finds the actual processed OCO-2 footprints within RADIUS_KM & TIME_WIN of the
ATom track for each date, and reports, per date:
  * the ocean-glint good-QF footprint bounding box (-> run_case LON/LAT range),
  * a VMIN/VMAX suggestion (2nd/98th pct of xco2_bc, rounded to 0.5),
  * cloud-distance coverage of the collocated footprints (the near-cloud second
    filter): how many are <=10 km, and the min/median/max cld_dist_km.

Prints a ready-to-paste run_case-style comment block per date.

Usage: python atom_footprint_collocate.py [--radius-km 100] [--window-min 120]
"""
from __future__ import annotations
import argparse, datetime as dt, glob, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_DIR = os.path.join(REPO, "results", "csv_collection")
TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
OUT_BASE = os.path.join(REPO, "results", "model_comparison", "deep_ensemble", TAG, "atom")
MERGED_DIR = os.path.join(OUT_BASE, "atom_merged")   # merged profiles (input)
EPOCH = dt.datetime(1970, 1, 1)

# dates chosen from the Lite screen, minus 20180501 (training data)
DATES = ["20170126", "20170203", "20170205", "20170210",
         "20171008", "20171020", "20171027", "20180512"]


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def atom_track(date: str):
    p = os.path.join(MERGED_DIR, f"atom_merged_{date}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=["time_utc_s", "lat", "lon"])
    base = (dt.datetime(int(date[:4]), int(date[4:6]), int(date[6:])) - EPOCH).total_seconds()
    df["epoch"] = base + df["time_utc_s"].to_numpy()
    return df


def utc_days(at: pd.DataFrame) -> list[str]:
    """UTC calendar days the flight's points fall on (handles dateline/midnight)."""
    ep = at["epoch"].to_numpy()
    return sorted({dt.datetime.utcfromtimestamp(e).strftime("%Y-%m-%d") for e in ep})


def collocate(date: str, radius_km: float, twin_s: float):
    dd = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    at = atom_track(date)
    if at is None:
        return dict(date=dd, err="no ATom merged parquet")

    # A flight can span two UTC days; the coincident OCO orbit may be on either.
    # Load every day-parquet that exists and note which are missing.
    days = utc_days(at)
    frames, missing = [], []
    for d in days:
        pq = os.path.join(CSV_DIR, f"combined_{d}_all_orbits.parquet")
        if os.path.exists(pq):
            f = pd.read_parquet(pq, columns=["time", "lon", "lat", "sfc_type",
                                             "xco2_qf", "xco2_bc", "cld_dist_km"])
            f["oco_day"] = d
            frames.append(f)
        else:
            missing.append(d)
    if not frames:
        return dict(date=dd, err=f"no processed parquet for {days}")
    oco = pd.concat(frames, ignore_index=True)
    oco = oco[(oco.sfc_type == 0) & (oco.xco2_qf == 0)]            # ocean, good quality
    # spatial pre-filter to padded track bbox. Latitude always; longitude only when
    # the track doesn't wrap the antimeridian (dateline-crossing flights span the
    # full lon range, so a lon filter would be a no-op / wrong).
    pad = radius_km / 111.0 + 0.5
    oco = oco[oco.lat.between(at.lat.min()-pad, at.lat.max()+pad)]
    if (at.lon.max() - at.lon.min()) < 180:
        oco = oco[oco.lon.between(at.lon.min()-pad, at.lon.max()+pad)]
    if oco.empty:
        return dict(date=dd, err="no ocean-glint good-QF footprints near track", missing=missing)

    olon = oco.lon.to_numpy(); olat = oco.lat.to_numpy(); ot = oco.time.to_numpy()
    alon = at.lon.to_numpy(); alat = at.lat.to_numpy(); att = at.epoch.to_numpy()
    hit = np.zeros(len(oco), bool)
    dmin = np.full(len(oco), np.inf)
    for i0 in range(0, len(oco), 200):
        sl = slice(i0, i0+200)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None], alon[None, :], alat[None, :])
        tg = np.abs(ot[sl][:, None] - att[None, :])
        dm = np.where(tg <= twin_s, d, np.inf).min(axis=1)
        dmin[sl] = dm
        hit[sl] = dm <= radius_km
    sel = oco[hit].copy()
    if sel.empty:
        return dict(date=dd, err=f"0 footprints within {radius_km}km/{twin_s/60:.0f}min",
                    closest_km=float(dmin.min()), missing=missing)

    cld = sel.cld_dist_km.to_numpy()
    xb = sel.xco2_bc.to_numpy()
    return dict(
        date=dd, n=len(sel),
        oco_day=sel.oco_day.value_counts().to_dict(),   # which day's parquet the hits came from
        missing=missing,
        lon=(float(sel.lon.min()), float(sel.lon.max())),
        lat=(float(sel.lat.min()), float(sel.lat.max())),
        vmin=np.floor(np.nanpercentile(xb, 2)*2)/2, vmax=np.ceil(np.nanpercentile(xb, 98)*2)/2,
        cld_min=float(np.nanmin(cld)), cld_med=float(np.nanmedian(cld)),
        cld_max=float(np.nanmax(cld)), n_near=int((cld <= 10).sum()),
        closest_km=float(dmin.min()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--window-min", type=float, default=120)
    args = ap.parse_args()
    twin = args.window_min * 60

    print(f"# ATom<->OCO-2 footprint collocation  ({args.radius_km:.0f} km / "
          f"+/-{args.window_min:.0f} min, ocean-glint good-QF)\n")
    hdr = f"{'date':>12} {'n_fp':>5} {'near<=10km':>10} {'cld_min':>8} {'cld_med':>8} " \
          f"{'lon_min':>8} {'lon_max':>8} {'lat_min':>8} {'lat_max':>8} {'vmin':>6} {'vmax':>6}"
    print(hdr); print("-"*len(hdr))
    for d in DATES:
        r = collocate(d, args.radius_km, twin)
        miss = f"  [missing day parquet: {','.join(r['missing'])}]" if r.get("missing") else ""
        if "err" in r:
            note = r["err"] + (f" (closest {r['closest_km']:.0f} km)" if "closest_km" in r else "")
            print(f"{r['date']:>12}  --  {note}{miss}")
            continue
        day = "+".join(r["oco_day"]) if r.get("oco_day") else r["date"]
        print(f"{r['date']:>12} {r['n']:>5} {r['n_near']:>10} {r['cld_min']:>8.1f} "
              f"{r['cld_med']:>8.1f} {r['lon'][0]:>8.2f} {r['lon'][1]:>8.2f} "
              f"{r['lat'][0]:>8.2f} {r['lat'][1]:>8.2f} {r['vmin']:>6.1f} {r['vmax']:>6.1f}"
              f"   oco_day={day}{miss}")


if __name__ == "__main__":
    main()
