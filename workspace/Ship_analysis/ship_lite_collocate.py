#!/usr/bin/env python3
"""Stage 1 — ship<->OCO-2 overlap screen from OCO-2 L2 Lite files only.

Two shipborne EM27/SUN cruises give open-ocean XCO2 truth (the OCEAN anchor the
land TCCON stations can't provide):
  * MORE-2 (RV Sonne, 2019-06)  — data/Other/SO268-3_track_XCO2_XCH4_XCO.tab
  * MR21-01 (RV Mirai, 2021-02/03) — data/Other/Hanft_2021_XCO2_XCH4_XCO.tab

For each ship measurement day this pulls that day's global OCO-2 L2_Lite_FP granule
(via CMR + Earthdata auth; cached in data/Other/lite_cache/, shared with the ATom
scan) and matches soundings to the *moving* ship track in space AND time.  No MODIS /
full pipeline needed — this is the cheap pre-screen that says which dates are worth
running through the cloud-distance pipeline.

Outputs (output/):
  * ship_oco2_collocation.csv — per ship day: n soundings within {50,100,250} km,
    good-QF counts, closest sounding, OCO-2 vs ship XCO2.
  * process_dates.txt — the days that clear the strict 100 km / +-2 h / good-QF gate.

Usage:
  python ship_lite_collocate.py                # download (cached) + screen
  python ship_lite_collocate.py --no-download  # cached Lite only
"""
from __future__ import annotations
import argparse, csv, datetime as dt, os, requests
import numpy as np, netCDF4 as nc
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output"); os.makedirs(OUT_DIR, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LITE = os.environ.get("OCO_SHIP_LITE_DIR", os.path.join(REPO, "data", "Other", "lite_cache"))
os.makedirs(LITE, exist_ok=True)
EPOCH = dt.datetime(1970, 1, 1)

TABS = {
    "so268":  (os.path.join(REPO, "data/Other/SO268-3_track_XCO2_XCH4_XCO.tab"), "MORE-2"),
    "mr2101": (os.path.join(REPO, "data/Other/Hanft_2021_XCO2_XCH4_XCO.tab"),   "MR21-01"),
}
RADII = [50, 100, 250]
TWIN_S = 120 * 60          # +-2 h headline window
GATE_KM, GATE_MIN = 100, 120   # strict gate for process_dates.txt


def parse_tab(path):
    """Return (epoch_s, lon, lat, xco2) rows and a YYYYMMDD day label per row."""
    rows, days = [], []
    started = False
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
        d = dt.datetime.strptime(p[0], fmt)
        rows.append(((d - EPOCH).total_seconds(), lon, lat, x))
        days.append(d.strftime("%Y%m%d"))
    return np.array(rows, float), np.array(days)


def resolve_url(ymd):
    d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json",
                     params={"short_name": "OCO2_L2_Lite_FP",
                             "temporal": f"{d}T00:00:00Z,{d}T23:59:59Z", "page_size": 20},
                     timeout=60)
    r.raise_for_status()
    for e in r.json()["feed"]["entry"]:
        if f"LtCO2_{ymd[2:]}_" in e["title"]:
            for l in e.get("links", []):
                h = l.get("href", "")
                if h.endswith(".nc4") and "gesdisc" in h and "/data/" in h:
                    return h
    return None


def fetch(ymd, download=True):
    dest = os.path.join(LITE, f"lite_{ymd}.nc4")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    if not download:
        return None
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
    R = 6371.0088
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def collocate_day(ymd, ship_a, ship_days, download=True):
    m = ship_days == ymd
    st, slon, slat, sx = ship_a[m, 0], ship_a[m, 1], ship_a[m, 2], ship_a[m, 3]
    res = dict(ymd=ymd, n_ship=int(m.sum()), oco="ok",
               counts={r: 0 for r in RADII}, counts_q={r: 0 for r in RADII},
               best_km=np.inf, best_tgap=np.nan, oco_xco2=np.nan,
               ship_xco2=float(np.nanmedian(sx)) if sx.size else np.nan)
    path = fetch(ymd, download)
    if not path:
        res["oco"] = "NO_GRANULE"; return res
    ds = nc.Dataset(path)
    olon = ds.variables["longitude"][:].astype(float)
    olat = ds.variables["latitude"][:].astype(float)
    ot   = ds.variables["time"][:].astype(float)
    qf   = ds.variables["xco2_quality_flag"][:].astype(int)
    ox   = ds.variables["xco2"][:].astype(float)
    ds.close()
    lo0, lo1 = slon.min()-6, slon.max()+6
    box = (olat >= slat.min()-6) & (olat <= slat.max()+6)
    if not (lo0 < -180 or lo1 > 180):        # skip lon filter near the dateline
        box &= (olon >= lo0) & (olon <= lo1)
    olon, olat, ot, qf, ox = olon[box], olat[box], ot[box], qf[box], ox[box]
    if olon.size == 0:
        return res
    good_x = []
    for i0 in range(0, olon.size, 400):
        sl = slice(i0, i0+400)
        d = haversine_km(olon[sl][:, None], olat[sl][:, None], slon[None, :], slat[None, :])
        tg = np.abs(ot[sl][:, None] - st[None, :])
        dm = np.where(tg <= TWIN_S, d, np.inf)
        nn = dm.min(axis=1)
        for r in RADII:
            hit = nn <= r
            res["counts"][r] += int(hit.sum())
            good = hit & (qf[sl] == 0)
            res["counts_q"][r] += int(good.sum())
            if r == GATE_KM:
                good_x.extend(ox[sl][good].tolist())
        fin = np.isfinite(nn)
        if fin.any() and nn[fin].min() < res["best_km"]:
            bi = int(np.where(fin, nn, np.inf).argmin())
            res["best_km"] = float(nn[bi])
            res["best_tgap"] = float(np.abs(ot[sl][bi] - st).min() / 60.0)
    if good_x:
        res["oco_xco2"] = float(np.nanmedian(good_x))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-download", action="store_true", help="use cached Lite files only")
    args = ap.parse_args()
    download = not args.no_download

    ship, all_days = {}, []
    for tag, (path, _desc) in TABS.items():
        a, days = parse_tab(path)
        ship[tag] = (a, days)
        for d in sorted(set(days.tolist())):
            all_days.append((tag, d))
    print(f"{len(all_days)} ship days across {len(TABS)} campaigns")
    if download:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda td: fetch(td[1], True), all_days))
        print("downloads done")

    rows, good = [], []
    hdr = (f"{'date':10} {'camp':7} {'nship':>5} {'oco':>10} {'min_km':>7} {'tgap_m':>7} "
           f"{'<=50':>5} {'<=100':>6} {'<=250':>6} {'q<=100':>7} {'dOCO-ship':>9}")
    print("\n" + hdr); print("-"*len(hdr))
    for tag, d in all_days:
        r = collocate_day(d, *ship[tag], download=download)
        dd = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        has_oco = np.isfinite(r.get("oco_xco2", np.nan))
        delta = (r["oco_xco2"] - r["ship_xco2"]) if (r["oco"] == "ok" and has_oco) else np.nan
        rows.append(dict(
            date=dd, campaign=TABS[tag][1], oco_data=r["oco"], n_ship=r["n_ship"],
            min_km=(f"{r['best_km']:.1f}" if np.isfinite(r["best_km"]) else ""),
            tgap_min=(f"{r['best_tgap']:.1f}" if np.isfinite(r["best_tgap"]) else ""),
            n50=r["counts"][50] if r["oco"] == "ok" else "",
            n100=r["counts"][100] if r["oco"] == "ok" else "",
            n250=r["counts"][250] if r["oco"] == "ok" else "",
            n100_goodqf=r["counts_q"][100] if r["oco"] == "ok" else "",
            oco_xco2_med=(f"{r['oco_xco2']:.2f}" if has_oco else ""),
            ship_xco2_med=(f"{r['ship_xco2']:.2f}" if np.isfinite(r["ship_xco2"]) else ""),
            delta_ppm=(f"{delta:.2f}" if np.isfinite(delta) else "")))
        if r["oco"] != "ok":
            print(f"{dd:10} {tag:7} {r['n_ship']:5d} {r['oco']:>10}")
            continue
        nq = r["counts_q"][GATE_KM]
        dstr = f"{delta:9.2f}" if np.isfinite(delta) else f"{'':>9}"
        print(f"{dd:10} {tag:7} {r['n_ship']:5d} {'ok':>10} {r['best_km']:7.1f} {r['best_tgap']:7.1f} "
              f"{r['counts'][50]:5d} {r['counts'][100]:6d} {r['counts'][250]:6d} {nq:7d} {dstr}")
        if nq >= 1 and r["best_km"] <= GATE_KM:
            good.append(dd)

    csv_path = os.path.join(OUT_DIR, "ship_oco2_collocation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "process_dates.txt"), "w") as f:
        f.write(" ".join(d.replace("-", "") for d in good) + "\n")
    print(f"\nwrote {csv_path}")
    print(f"GOOD (>=1 good-QF <= {GATE_KM} km, +-{GATE_MIN} min): {len(good)} -> {good}")


if __name__ == "__main__":
    main()
