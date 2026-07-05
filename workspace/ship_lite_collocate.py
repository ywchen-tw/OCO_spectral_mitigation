#!/usr/bin/env python3
"""Identify OCO-2 <-> ship-track overlap days using OCO-2 L2 Lite files only.

For each ship measurement day we pull that day's global Lite granule (cached),
then find OCO-2 soundings that fall within a distance radius AND time window of
the moving ship track.  No MODIS / full pipeline needed.
"""
import os, sys, re, requests, datetime as dt, numpy as np, netCDF4 as nc
from concurrent.futures import ThreadPoolExecutor

SC = os.environ.get("OCO_SHIP_SCRATCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Other"))
LITE = os.environ.get("OCO_SHIP_LITE_DIR", os.path.join(SC, "lite_cache")); os.makedirs(LITE, exist_ok=True)
REPO = "/Users/yuch8913/programming/oco_fp_analysis"
TABS = {
    "MORE-2":  os.path.join(REPO, "data/Other/SO268-3_track_XCO2_XCH4_XCO.tab"),
    "MR21-01": os.path.join(REPO, "data/Other/Hanft_2021_XCO2_XCH4_XCO.tab"),
}
EPOCH = dt.datetime(1970, 1, 1)

# ---- parse a PANGAEA .tab: rows after the header line "Date/Time\tLongitude..." ----
def parse_tab(path):
    rows = []
    started = False
    with open(path) as f:
        for line in f:
            if not started:
                if line.startswith("Date/Time\t"):
                    started = True
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or not p[0].startswith("20"):
                continue
            t = p[0]
            # formats: 2019-06-04T17:33:00  or  2021-02-13T00:00
            fmt = "%Y-%m-%dT%H:%M:%S" if t.count(":") == 2 else "%Y-%m-%dT%H:%M"
            d = dt.datetime.strptime(t, fmt)
            rows.append(((d - EPOCH).total_seconds(), float(p[1]), float(p[2]), d))
    a = np.array([(r[0], r[1], r[2]) for r in rows], float)
    days = np.array([r[3].strftime("%Y%m%d") for r in rows])
    return a, days  # a[:,0]=epoch s, a[:,1]=lon, a[:,2]=lat

# ---- CMR: resolve the Lite granule URL for a YYYYMMDD ----
def resolve_url(ymd):
    d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
                     params={"short_name": "OCO2_L2_Lite_FP",
                             "temporal": f"{d}T00:00:00Z,{d}T23:59:59Z",
                             "page_size": 20}, timeout=60)
    r.raise_for_status()
    yy = ymd[2:]
    for e in r.json()["feed"]["entry"]:
        if f"LtCO2_{yy}_" in e["title"]:
            for l in e.get("links", []):
                h = l.get("href", "")
                if h.endswith(".nc4") and "gesdisc" in h and "/data/" in h:
                    return h
    return None

def fetch(ymd):
    dest = os.path.join(LITE, f"lite_{ymd}.nc4")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    url = resolve_url(ymd)
    if not url:
        return None
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0"})
    with s.get(url, stream=True, timeout=300, allow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for c in resp.iter_content(1 << 20):
                f.write(c)
    return dest

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lo2 - lo1; dlat = la2 - la1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def collocate_day(ymd, ship_a, ship_days):
    m = ship_days == ymd
    st, slon, slat = ship_a[m, 0], ship_a[m, 1], ship_a[m, 2]
    path = fetch(ymd)
    if not path:
        return dict(ymd=ymd, err="no granule")
    ds = nc.Dataset(path)
    olon = ds.variables["longitude"][:].astype(float)
    olat = ds.variables["latitude"][:].astype(float)
    ot   = ds.variables["time"][:].astype(float)
    qf   = ds.variables["xco2_quality_flag"][:].astype(int)
    xco2 = ds.variables["xco2"][:].astype(float)
    ds.close()
    # spatial pre-filter to ship daily bbox padded 6 deg (handle dateline by pad on both)
    lo0, lo1 = slon.min()-6, slon.max()+6
    la0, la1 = slat.min()-6, slat.max()+6
    box = (olat >= la0) & (olat <= la1)
    if lo0 < -180 or lo1 > 180:  # near dateline: don't lon-filter
        pass
    else:
        box &= (olon >= lo0) & (olon <= lo1)
    olon, olat, ot, qf, xco2 = olon[box], olat[box], ot[box], qf[box], xco2[box]
    RADII = [50, 100, 250]
    if olon.size == 0:
        return dict(ymd=ymd, n_ship=int(m.sum()),
                    counts={r: 0 for r in RADII}, counts_q={r: 0 for r in RADII},
                    best=dict(km=np.inf))
    # pairwise nearest ship point (space) + time gap, chunked over oco soundings
    TWIN  = 120*60  # s, headline time window (+/-2h)
    counts = {r: 0 for r in RADII}
    counts_q = {r: 0 for r in RADII}   # good quality (qf==0)
    best = dict(km=np.inf)
    for i0 in range(0, olon.size, 400):
        sl = slice(i0, i0+400)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None],
                         slon[None, :], slat[None, :])           # (chunk, nship)
        tg = np.abs(ot[sl][:, None] - st[None, :])               # s
        valid = tg <= TWIN
        dm = np.where(valid, d, np.inf)
        nn = dm.min(axis=1)                                       # nearest in-window ship pt
        jj = np.where(np.isfinite(nn), dm.argmin(axis=1), 0)
        for k, r in enumerate(RADII):
            hit = nn <= r
            counts[r] += int(hit.sum())
            counts_q[r] += int((hit & (qf[sl] == 0)).sum())
        # track single best (closest space within window)
        if np.isfinite(nn.min()):
            bi = int(nn.argmin())
            if nn[bi] < best["km"]:
                gap = abs(ot[sl][bi] - st[jj[bi]])
                best = dict(km=float(nn[bi]), tgap_min=gap/60.0,
                            qf=int(qf[sl][bi]), xco2=float(xco2[sl][bi]),
                            olon=float(olon[sl][bi]), olat=float(olat[sl][bi]))
    return dict(ymd=ymd, n_ship=int(m.sum()),
                counts=counts, counts_q=counts_q, best=best)

def main():
    all_days = []
    campaign_of = {}
    ship = {}
    for camp, path in TABS.items():
        a, days = parse_tab(path)
        ship[camp] = (a, days)
        for d in sorted(set(days.tolist())):
            all_days.append((camp, d))
            campaign_of[d] = camp
    print(f"{len(all_days)} ship days across {len(TABS)} campaigns")
    # download all first (parallel)
    ymds = [d for _, d in all_days]
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(fetch, ymds))
    print("downloads done\n")
    hdr = f"{'date':10} {'camp':8} {'nship':>5} {'min_km':>7} {'tgap_m':>7} " \
          f"{'<=50':>5} {'<=100':>6} {'<=250':>6} {'q<=100':>7}"
    print(hdr); print("-"*len(hdr))
    results = []
    for camp, d in all_days:
        r = collocate_day(d, *ship[camp])
        results.append((camp, d, r))
        if "err" in r:
            print(f"{d:10} {camp:8}  ERR {r['err']}"); continue
        b = r["best"]; c = r["counts"]; cq = r["counts_q"]
        mk = b["km"] if b and np.isfinite(b["km"]) else float("inf")
        tg = b.get("tgap_min", float("nan")) if b else float("nan")
        print(f"{d:10} {camp:8} {r['n_ship']:5d} {mk:7.1f} {tg:7.1f} "
              f"{c[50]:5d} {c[100]:6d} {c[250]:6d} {cq[100]:7d}")
    # summary of GOOD overlap days (>=1 good-QF sounding within 100 km, +/-2h)
    print("\n=== GOOD overlap (>=1 good-quality sounding <=100 km, +/-2h) ===")
    good = []
    for camp, d, r in results:
        if "err" in r: continue
        if r["counts_q"][100] >= 1:
            good.append((camp, d, r["counts_q"][100], r["best"]["km"]))
    for camp, d, nq, mk in good:
        print(f"  {d}  {camp:8} good_q<=100km={nq:3d}  closest={mk:.1f} km")
    print(f"\nGOOD days: {len(good)}")
    print("2019:", " ".join(d for c,d,_,_ in good if d.startswith('2019')))
    print("2021:", " ".join(d for c,d,_,_ in good if d.startswith('2021')))

if __name__ == "__main__":
    main()
