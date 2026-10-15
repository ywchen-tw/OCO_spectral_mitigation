#!/usr/bin/env python
"""Simple Gaussian-plume emission estimates for the US power-plant overpasses.

Internal check (not for the manuscript): does the deep-ensemble correction
change the CO2 emission rate that a single OCO-2 overpass implies, relative
to the emissions the plant reported for that hour?

Per overpass
  * footprints: the closest-approach segment of plot_data.parquet within
    --radius-km of the plant (original xco2_bc and DE-corrected XCO2)
  * wind: ERA5 hourly 100 m wind (and 10 m for reference) at the plant,
    linearly interpolated in u/v to the overpass time, from the public
    Open-Meteo archive API; surface pressure from the same source
  * reported: EPA CAMPD hourly CO2 mass summed over all units, for the
    local-standard-time hour that contains the overpass
  * model (Nassar et al. 2017 style): column enhancement of a Gaussian plume,
        dOmega(x, y) = Q / (sqrt(2 pi) sigma_y(x) u) exp(-y^2 / (2 sigma_y^2)),
    sigma_y = a (x / 1 km)^0.894 m with a from the Pasquill-Gifford daytime
    class for the wind speed, averaged over a 1.3 km x 2.25 km footprint,
    converted to ppm with the dry-air column p_s / (g M_air).  XCO2 is fitted
    by ordinary least squares as b0 + b1 s + Q G, with s the along-track
    distance, once for the ERA5 wind direction and once for the direction
    within +/-30 deg that minimizes the residual.

Usage:  PYTHONPATH=src:workspace python workspace/Nassar_plume_analysis/us_plant_emissions.py
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import CMAPS, apply_manuscript_style, panel_label, XCO2_BC_LABEL  # noqa: E402
# The corrected product is labelled "model" in the manuscript ($X_{CO2}^{model}$, 2026-10-11);
# plot_style still exports the older DE superscript, so the label is set here.
XCO2_MODEL_LABEL = r"$X_{\mathrm{CO2}}^{\mathrm{ML}}$"
from plot_nassar_plumes import (DEFAULT_PLANTS, DEFAULT_PLOT_BASE, aqua_time_lag_min,  # noqa: E402
                                circle_lon_lat, download_gibs_wms_rgb, haversine_km,
                                modis_lag_note, plant_extent, read_plants)

# (plant_id, date, EPA ORIS facility id, UTC offset of local standard time)
CASES = (
    ("westar", "2015-12-04", 6068, -6),   # Jeffrey Energy Center, KS
    ("ghent", "2015-08-13", 1356, -5),    # Ghent, KY
    ("westar", "2023-03-13", 6068, -6),
    ("westar", "2023-06-26", 6068, -6),
    ("ghent", "2024-04-15", 1356, -5),
)
# Stack heights for the "stack250" wind (plume midline = stack + 250 m plume rise,
# as in Nassar et al. 2021, 2022).  Jeffrey Energy Center: "The flue-gas stacks are
# 600 feet tall" (HPPR, 2017).  Ghent: "New 662-foot chimneys were constructed for
# Units 1 and 4" with the FGD projects (LG&E/KU station page).
STACK_HEIGHT_M = {"westar": 600 * 0.3048, "ghent": 662 * 0.3048}
PLUME_RISE_M = 250.0

# Cases known not to be plumes from the earlier plume tests (SI S4).
CASE_NOTES = {
    ("westar", "2023-03-13"): "S4: near-cloud window, all-band <l'> shift (cloud artifact)",
}
SEGMENT_GAP_S = 60.0
# Cases of the SI figure (top to bottom) and the plot inputs kept by run_case for it.
SI_CASES = (("ghent", "2015-08-13"), ("westar", "2015-12-04"))
PLOT_INPUTS = {}
G, M_AIR, M_CO2 = 9.80665, 0.0289644, 0.04401
SHORT_TON_KG = 907.18474
FP_ALONG_KM, FP_CROSS_KM = 2.25, 1.3


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read())


def era5_at(lat, lon, t_utc, cache):
    key = f"era5_{lat:.3f}_{lon:.3f}_{t_utc:%Y-%m-%d}.json"
    if (cache / key).exists():
        j = json.loads((cache / key).read_text())
    else:
        q = dict(latitude=lat, longitude=lon, start_date=f"{t_utc:%Y-%m-%d}",
                 end_date=f"{t_utc:%Y-%m-%d}", timezone="GMT", wind_speed_unit="ms",
                 hourly="wind_speed_10m,wind_direction_10m,wind_speed_100m,"
                        "wind_direction_100m,surface_pressure")
        j = get_json("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
        (cache / key).write_text(json.dumps(j))
    h = pd.DataFrame(j["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    tt = (t_utc - h["time"].iloc[0].to_pydatetime()).total_seconds() / 3600.0
    out = {}
    for lev in ("10m", "100m"):
        sp, dr = h[f"wind_speed_{lev}"].to_numpy(), np.radians(h[f"wind_direction_{lev}"].to_numpy())
        u, v = -sp * np.sin(dr), -sp * np.cos(dr)           # direction is "from"
        ui, vi = np.interp(tt, np.arange(len(h)), u), np.interp(tt, np.arange(len(h)), v)
        out[f"ws_{lev}"] = float(np.hypot(ui, vi))
        out[f"wd_{lev}"] = float(np.degrees(np.arctan2(-ui, -vi)) % 360)
    out["ps_pa"] = float(np.interp(tt, np.arange(len(h)), h["surface_pressure"].to_numpy())) * 100.0
    return out


def campd_hour(oris, t_utc, utc_offset_h, cache, api_key):
    t_local = t_utc + timedelta(hours=utc_offset_h)
    key = f"campd_{oris}_{t_local:%Y-%m-%d}.json"
    if (cache / key).exists():
        items = json.loads((cache / key).read_text())
    else:
        q = dict(api_key=api_key, facilityId=oris, beginDate=f"{t_local:%Y-%m-%d}",
                 endDate=f"{t_local:%Y-%m-%d}", perPage=500, page=1)
        items = get_json("https://api.epa.gov/easey/emissions-mgmt/emissions/apportioned/hourly?"
                         + urllib.parse.urlencode(q))
        (cache / key).write_text(json.dumps(items))
    d = pd.DataFrame(items["items"] if isinstance(items, dict) else items)
    d["co2Mass"] = pd.to_numeric(d["co2Mass"], errors="coerce").fillna(0.0)
    per_hour = d.groupby("hour")["co2Mass"].sum() * SHORT_TON_KG / 3600.0   # kg/s
    return {"facility": d["facilityName"].iloc[0], "local_hour": t_local.hour,
            "q_rep_kgs": float(per_hour.get(t_local.hour, np.nan)),
            "q_rep_day_mean_kgs": float(per_hour.mean()),
            "n_units": int(d["unitId"].nunique())}


def pg_a(ws):
    """Pasquill-Gifford daytime (strong insolation) sigma_y coefficient (m at 1 km)."""
    return 213.0 if ws < 2 else 156.0 if ws < 3 else 104.0 if ws < 5 else 68.0


def local_km(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 111.32)


def plume_kernel(ex, ny, wd_from, ws, a, track_dir):
    """Footprint-mean Gaussian column per unit emission (kg m-2 per kg s-1)."""
    to = np.radians((wd_from + 180.0) % 360)               # plume travel direction
    dx, dy = np.sin(to), np.cos(to)
    ta, tc = track_dir, np.array([track_dir[1], -track_dir[0]])
    offs = [(sa, sc) for sa in np.linspace(-0.5, 0.5, 5) * FP_ALONG_KM
            for sc in np.linspace(-0.5, 0.5, 5) * FP_CROSS_KM]
    k = np.zeros_like(ex)
    for sa, sc in offs:
        px, py = ex + sa * ta[0] + sc * tc[0], ny + sa * ta[1] + sc * tc[1]
        xd = px * dx + py * dy                               # downwind km
        yc = -px * dy + py * dx                              # crosswind km
        sig = a * np.clip(xd, 1e-3, None) ** 0.894           # m
        val = np.exp(-(yc * 1e3) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig * ws)
        k += np.where(xd > 0, val, 0.0)
    return k / len(offs)


def fit_q(seg, col, kernel_ppm, mask=None):
    """OLS fit; with mask=None one 3-sigma screening pass, else use mask as given."""
    y = seg[col].to_numpy()
    X = np.column_stack([np.ones_like(y), seg["s_km"].to_numpy(), kernel_ppm])
    ok = np.isfinite(y)
    if np.nanmax(kernel_ppm) < 1e-3:                         # plume misses the track
        return np.nan, np.nan, np.inf, np.array([np.nanmean(y), 0.0, 0.0]), ok
    if mask is not None:
        ok = ok & mask
    else:
        for _ in range(2):                                   # one 3-sigma screening pass
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            res = y - X @ beta
            sd = np.nanstd(res[ok])
            ok = ok & (np.abs(res) < 3 * sd)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res = y[ok] - X[ok] @ beta
    dof = max(ok.sum() - 3, 1)
    cov = np.linalg.inv(X[ok].T @ X[ok]) * (res @ res / dof)
    return float(beta[2]), float(np.sqrt(cov[2, 2])), float(np.sqrt(res @ res / dof)), beta, ok


def era5_profile(tag, date, lat0, lon0, t_utc, cds_dir):
    """u, v profile above ground at the plant from the CDS downloads, or None.

    Returns heights (m AGL, ascending), u, v (m/s) at the overpass time,
    interpolated linearly in time, plus the boundary-layer height (m)."""
    import netCDF4 as nc4
    fp, fs = cds_dir / f"era5_plev_{tag}_{date}.nc", cds_dir / f"era5_sfc_{tag}_{date}.nc"
    if not (fp.exists() and fs.exists()):
        return None
    with nc4.Dataset(fp) as dp_, nc4.Dataset(fs) as ds_:
        def near(ds):
            la, lo = ds["latitude"][:], ds["longitude"][:]
            return int(np.argmin(abs(la - lat0))), int(np.argmin(abs(lo - lon0)))
        tp = nc4.num2date(dp_["valid_time"][:], dp_["valid_time"].units,
                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        tsec = np.array([(t.replace(tzinfo=timezone.utc) - t_utc).total_seconds() for t in tp])
        w = np.interp(0.0, tsec, np.arange(len(tsec)))     # fractional time index
        i0, i1 = int(np.floor(w)), min(int(np.floor(w)) + 1, len(tsec) - 1)
        f = w - i0
        tint = lambda a: (1 - f) * a[i0] + f * a[i1]
        jp, kp = near(dp_)
        js, ks = near(ds_)
        zsfc = tint(ds_["z"][:, js, ks])
        u100, v100 = tint(ds_["u100"][:, js, ks]), tint(ds_["v100"][:, js, ks])
        blh = float(tint(ds_["blh"][:, js, ks]))
        h = (tint(dp_["z"][:, :, jp, kp]) - zsfc) / G
        u, v = tint(dp_["u"][:, :, jp, kp]), tint(dp_["v"][:, :, jp, kp])
    above = h > 120.0
    hh = np.concatenate([[100.0], np.asarray(h)[above]])
    uu = np.concatenate([[float(u100)], np.asarray(u)[above]])
    vv = np.concatenate([[float(v100)], np.asarray(v)[above]])
    o = np.argsort(hh)
    return hh[o], uu[o], vv[o], blh, float(zsfc) / G


def wind_at_height(prof, height):
    hh, uu, vv, blh = prof[:4]
    if height == "blh":                                     # mean wind 100 m .. BLH
        top = float(np.clip(blh, 150.0, hh.max()))
        zz = np.linspace(100.0, top, 50)
        ui, vi = np.interp(zz, hh, uu).mean(), np.interp(zz, hh, vv).mean()
    else:
        ui, vi = np.interp(float(height), hh, uu), np.interp(float(height), hh, vv)
    return float(np.hypot(ui, vi)), float(np.degrees(np.arctan2(-ui, -vi)) % 360)


def background_rss(seg, col, mask):
    y = seg[col].to_numpy()
    ok = np.isfinite(y) & mask
    X = np.column_stack([np.ones(ok.sum()), seg["s_km"].to_numpy()[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    r = y[ok] - X @ beta
    return float(r @ r), int(ok.sum())


def scan_direction(seg, col, kernels, mask, min_fp=10):
    """Best wind-direction offset among those whose plume covers >= min_fp used
    footprints; F statistic against the background-only fit."""
    cands = []
    for d, kk in kernels.items():
        if not np.isfinite(kk).all() or kk.max() <= 0:
            continue
        if ((kk > 0.05 * kk.max()) & mask).sum() < min_fp:
            continue
        q, qse, rsd, beta, ok = fit_q(seg, col, kk, mask)
        if np.isfinite(rsd):
            cands.append((rsd, d, q, qse, beta, ok))
    if not cands:
        return None
    rsd, d, q, qse, beta, ok = min(cands, key=lambda c: c[0])
    rss0, n = background_rss(seg, col, mask)
    rss1 = rsd ** 2 * max(n - 3, 1)
    F = (rss0 - rss1) / (rss1 / max(n - 3, 1))
    return dict(dwd=d, q=q, qse=qse, rsd=rsd, beta=beta, ok=ok, F=F, kernel=kernels[d])


def merra2_profile(tag, date, t_utc, zsfc_m, m2_dir):
    """MERRA-2 (M2I3NPASM) u, v profile above ground at the plant, linearly
    interpolated in time to the overpass; heights from H minus the ERA5 surface
    height.  None when the download is missing."""
    import netCDF4 as nc4
    f = m2_dir / f"merra2_{tag}_{date}.nc"
    if not f.exists():
        return None
    with nc4.Dataset(f) as d:
        t = nc4.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)
        tsec = np.array([(x.replace(tzinfo=timezone.utc) - t_utc).total_seconds() for x in t])
        w = np.interp(0.0, tsec, np.arange(len(tsec)))
        i0, i1 = int(np.floor(w)), min(int(np.floor(w)) + 1, len(tsec) - 1)
        fr = w - i0
        get = lambda v: np.ma.filled((1 - fr) * d[v][i0] + fr * d[v][i1], np.nan).ravel()
        h, u, v = get("H") - zsfc_m, get("U"), get("V")
    ok = np.isfinite(h) & np.isfinite(u) & np.isfinite(v) & (h > 10.0)
    o = np.argsort(h[ok])
    return h[ok][o], u[ok][o], v[ok][o], np.nan


def vector_mean(*winds):
    """Vector average of (speed, from-direction) pairs."""
    u = np.mean([-ws * np.sin(np.radians(wd)) for ws, wd in winds])
    v = np.mean([-ws * np.cos(np.radians(wd)) for ws, wd in winds])
    return float(np.hypot(u, v)), float(np.degrees(np.arctan2(-u, -v)) % 360)


def add_geometry(seg, date, csv_dir):
    """Attach sza, vza, saa, vaa from the per-date feature parquet (joined on
    time and position, since the two files do not share a row order)."""
    c = pd.read_parquet(csv_dir / f"combined_{date}_all_orbits.parquet",
                        columns=["time", "lat", "lon", "sza", "vza", "saa", "vaa"])
    key = lambda d: (d["time"].round(3).astype(str) + "_" + d["lat"].round(4).astype(str)
                     + "_" + d["lon"].round(4).astype(str))
    c = c.assign(_k=key(c)).drop_duplicates("_k").set_index("_k")
    g = c.reindex(key(seg).to_numpy())[["sza", "vza", "saa", "vaa"]]
    out = seg.copy()
    for col in g.columns:
        out[col] = g[col].to_numpy()
    return out


def plume_kernel_slant(ex, ny, wd_from, ws, a, track_dir, sza, vza, saa, vaa, h_km):
    """Footprint-mean Gaussian column seen along the slant light path.

    The sun ray reaches the plume layer h_km above the footprint at a horizontal
    offset h tan(sza) toward the sun, the view ray at h tan(vza) toward the
    satellite; the two crossings are weighted by their air-mass factors
    1/cos(sza) and 1/cos(vza), as XCO2 is retrieved relative to the two-way
    air mass.  Per unit emission, in kg m-2 per kg s-1."""
    mu0, mu = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    ds, dv = h_km * np.tan(np.radians(sza)), h_km * np.tan(np.radians(vza))
    sx, sy = ds * np.sin(np.radians(saa)), ds * np.cos(np.radians(saa))
    vx, vy = dv * np.sin(np.radians(vaa)), dv * np.cos(np.radians(vaa))
    ks = plume_kernel(ex + sx, ny + sy, wd_from, ws, a, track_dir)
    kv = plume_kernel(ex + vx, ny + vy, wd_from, ws, a, track_dir)
    return (ks / mu0 + kv / mu) / (1 / mu0 + 1 / mu)


def fit_nassar(seg, col, kernel_ppm, buffer_km=3.0, bg_km=30.0, edge_frac=0.01):
    """Nassar et al. (2021, 2022) style fit on unfiltered footprints.

    Plume = footprints where the model exceeds edge_frac of its peak (the 1 %
    plume boundary of Nassar et al. 2017); background = mean of the footprints
    within bg_km along track beyond a buffer_km buffer on either side of the
    plume; Q = least squares of (XCO2 - background) on the model over the plume
    footprints.  Returns None when fewer than 10 plume footprints."""
    y = seg[col].to_numpy()
    s = seg["s_km"].to_numpy()
    k = kernel_ppm
    if not np.isfinite(k).all() or k.max() <= 0:
        return None
    plume = k > edge_frac * k.max()
    if plume.sum() < 10:
        return None
    lo, hi = s[plume].min(), s[plume].max()
    bgm = (~plume) & (((s < lo - buffer_km) & (s >= lo - buffer_km - bg_km))
                      | ((s > hi + buffer_km) & (s <= hi + buffer_km + bg_km)))
    bgm &= np.isfinite(y)
    if bgm.sum() < 10:
        return None
    bg = float(np.mean(y[bgm]))
    pm = plume & np.isfinite(y)
    d = y[pm] - bg
    q = float(k[pm] @ d / (k[pm] @ k[pm]))
    res = d - q * k[pm]
    rss1 = float(res @ res)
    rss0 = float(d @ d)
    n = int(pm.sum())
    qse = float(np.sqrt(rss1 / max(n - 1, 1) / (k[pm] @ k[pm])))
    F = (rss0 - rss1) / (rss1 / max(n - 1, 1))
    return dict(q=q, qse=qse, bg=bg, plume=plume, bgmask=bgm, F=F, n_plume=n,
                n_bg=int(bgm.sum()), rsd=float(np.sqrt(rss1 / max(n - 1, 1))), kernel=k)


def scan_nassar(seg, col, kernels):
    """Wind-direction offset whose Nassar-style fit leaves the smallest residual
    over all footprints of the segment (same footprint set for every offset)."""
    y = seg[col].to_numpy()
    fin = np.isfinite(y)
    best = None
    for d, kk in kernels.items():
        f = fit_nassar(seg, col, kk)
        if f is None:
            continue
        r = (y - f["bg"] - f["q"] * kk)[fin]
        rss = float(r @ r)
        if best is None or rss < best[0]:
            best = (rss, d, f)
    if best is None:
        return None
    rss, d, f = best
    n = int(fin.sum())
    f.update(dwd=d, beta=np.array([f["bg"], 0.0, f["q"]]), ok=fin)
    return f


def run_case(case, plants, plot_base, cache, api_key, radius_km, outdir, heights, f_min,
             plot_height, method="nassar", csv_dir=Path("results/csv_collection"),
             modis_dir=None):
    pid, date, oris, utc_off = case
    plant = plants[pid]
    lat0, lon0 = float(plant["lat"]), float(plant["lon"])
    df = pd.read_parquet(plot_base / f"combined_{date}" / "plot_data.parquet",
                         columns=["time", "lat", "lon", "xco2_bc", "deep_ensemble_corrected_xco2",
                                  "cld_dist_km"])
    df = df[np.isfinite(df["xco2_bc"])].sort_values("time").reset_index(drop=True)
    df["dist"] = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(), lat0, lon0)
    seg_id = (df["time"].diff().abs() > SEGMENT_GAP_S).cumsum()
    seg = df[(seg_id == seg_id.iloc[df["dist"].idxmin()]) & (df["dist"] <= radius_km)].copy()
    t_utc = datetime.fromtimestamp(float(seg.loc[seg["dist"].idxmin(), "time"]), tz=timezone.utc)

    met = era5_at(lat0, lon0, t_utc, cache)                # 10 m wind and surface pressure
    modis_lag = (aqua_time_lag_min(date, lat0, lon0, t_utc, modis_dir)
                 if modis_dir is not None else None)
    rep = campd_hour(oris, t_utc, utc_off, cache, api_key)
    prof = era5_profile(pid, date, lat0, lon0, t_utc, cache / "era5_cds")
    a = pg_a(met["ws_10m"])                                # stability class from the 10 m wind

    ex, ny = local_km(seg["lat"].to_numpy(), seg["lon"].to_numpy(), lat0, lon0)
    seg["ex"], seg["ny"] = ex, ny
    i0, i1 = np.argmin(ny), np.argmax(ny)
    track = np.array([ex[i1] - ex[i0], ny[i1] - ny[i0]]); track /= np.linalg.norm(track)
    seg["s_km"] = ex * track[0] + ny * track[1]
    ppm_per_kgm2 = 1.0 / (1e-6 * M_CO2 / M_AIR * met["ps_pa"] / G)

    plume_h_m = STACK_HEIGHT_M[pid] + PLUME_RISE_M
    m2 = (merra2_profile(pid, date, t_utc, prof[4], cache / "merra2")
          if prof is not None else None)
    winds = {}
    for hgt in heights:
        if hgt == "stack250_mix":
            if prof is not None and m2 is not None:
                winds[hgt] = vector_mean(wind_at_height(prof, plume_h_m),
                                         wind_at_height(m2, plume_h_m))
            continue
        if hgt == "100" and prof is None:
            winds[hgt] = (met["ws_100m"], met["wd_100m"])
        elif prof is not None and hgt == "stack250":
            winds[hgt] = wind_at_height(prof, STACK_HEIGHT_M[pid] + PLUME_RISE_M)
        elif prof is not None:
            winds[hgt] = wind_at_height(prof, hgt if hgt == "blh" else float(hgt))
    if method == "nassar":
        seg = add_geometry(seg, date, csv_dir)
        geo = [seg[c].to_numpy() for c in ("sza", "vza", "saa", "vaa")]

    def kernels_for(ws, wd):
        if method == "nassar":
            return {dwd: plume_kernel_slant(ex, ny, (wd + dwd) % 360, ws, a, track, *geo,
                                            plume_h_m / 1000.0) * ppm_per_kgm2
                    for dwd in np.arange(-30, 31, 2)}
        return {dwd: plume_kernel(ex, ny, (wd + dwd) % 360, ws, a, track) * ppm_per_kgm2
                for dwd in np.arange(-30, 31, 2)}

    # One footprint mask for every height and both series: 3-sigma screening of
    # the original series around its best 100 m plume fit (or the background).
    if method == "nassar":                                 # unfiltered, as Nassar et al.
        mask = np.isfinite(seg["xco2_bc"].to_numpy())
    else:
        ws100, wd100 = winds.get("100", (met["ws_100m"], met["wd_100m"]))
        k100 = kernels_for(ws100, wd100)
        allm = np.ones(len(seg), bool)
        sc0 = scan_direction(seg, "xco2_bc", k100, allm)
        mask = fit_q(seg, "xco2_bc", sc0["kernel"] if sc0 else np.zeros(len(seg)) + 1e-9)[4]

    rows, plot_fits = [], {}
    for hgt, (ws, wd) in winds.items():
        to = np.radians((wd + 180) % 360)
        wt = float(np.degrees(np.arccos(abs(np.sin(to) * track[0] + np.cos(to) * track[1]))))
        kern = kernels_for(ws, wd)
        if method == "nassar":
            det = scan_nassar(seg, "xco2_bc", kern)
        else:
            det = scan_direction(seg, "xco2_bc", kern, mask)
        for col in ("xco2_bc", "deep_ensemble_corrected_xco2"):
            if method == "nassar":
                f0 = fit_nassar(seg, col, kern[0])
                q0, q0se = (f0["q"], f0["qse"]) if f0 else (np.nan, np.nan)
                sc = scan_nassar(seg, col, kern)
            else:
                q0, q0se, *_ = fit_q(seg, col, kern[0], mask) if ((kern[0] > 0.05 * max(kern[0].max(), 1e-30)) & mask).sum() >= 10 else (np.nan, np.nan)
                sc = scan_direction(seg, col, kern, mask)
            detected = det is not None and det["F"] >= f_min
            flags = [f for f, c in (
                (CASE_NOTES.get((pid, date), ""), (pid, date) in CASE_NOTES),
                ("plume does not cross the track", det is None),
                (f"no plume detected (F={det['F']:.1f})" if det else "", det is not None and not detected),
                ("best direction at scan edge", sc is not None and abs(sc["dwd"]) >= 30),
                ("10 m wind < 2 m/s", met["ws_10m"] < 2.0),
                ("wind within 30 deg of track", wt < 30.0)) if c]
            rows.append(dict(plant=plant["source_name"], date=date, series=col, wind_height=hgt,
                             overpass_utc=f"{t_utc:%Y-%m-%d %H:%M}",
                             min_dist_km=float(seg["dist"].min()), n=int(len(seg)),
                             n_used=int(mask.sum()), ws=ws, wd=wd, ws_10m=met["ws_10m"], pg_a=a,
                             blh_m=prof[3] if prof is not None else np.nan,
                             method=method, plume_height_m=plume_h_m,
                             wind_track_angle_deg=wt, q_era5_kgs=q0, q_era5_se=q0se,
                             q_best_kgs=sc["q"] if (sc and detected) else np.nan,
                             q_best_se=sc["qse"] if (sc and detected) else np.nan,
                             best_dwd=sc["dwd"] if sc else np.nan,
                             F_detect=det["F"] if det else np.nan,
                             q_reported_kgs=rep["q_rep_kgs"],
                             q_reported_daymean_kgs=rep["q_rep_day_mean_kgs"],
                             facility=rep["facility"], resid_sd_ppm=sc["rsd"] if sc else np.nan,
                             modis_lag_min=modis_lag if modis_lag is not None else np.nan,
                             flag="; ".join(flags) or "ok"))
            if hgt == plot_height:
                plot_fits[col] = sc
        if hgt == plot_height:
            plot_rows = rows[-2:]
            plot_wind = (ws, wd)
    if plot_fits:
        plot_case(seg, plot_fits, plant, date, plot_wind, plot_height, rep, plot_rows, outdir,
                  method=method, modis_dir=modis_dir, radius_km=radius_km, modis_lag=modis_lag)
        PLOT_INPUTS[(pid, date)] = ((seg, plot_fits, plant, date, plot_wind, plot_height, rep,
                                     plot_rows, outdir),
                                    dict(method=method, modis_dir=modis_dir, radius_km=radius_km,
                                         modis_lag=modis_lag))
    return rows


def hlabel(h):
    return {"blh": "BL mean", "stack250": "stack + 250 m (ERA5)",
            "stack250_mix": "stack + 250 m (ERA5 + MERRA-2)"}.get(h, f"{h} m")


# Page layout of plot_case in inches (7.2 in = full page width): left margin, gap between
# the maps, colour bar gap / width / label room, bottom margin, transect height, gap between
# the maps and the transect, top margin; MAP_HALF_Y_KM crops the maps north-south so that
# two cases fit one page (the transect still shows the full track).
# 2026-10-11 print-size redraw: drawn at 6.3 in (the printed width, scale ~1) instead of
# 7.2 in shrunk to ~4.9 in; transect 1.5 -> 1.05 in and the maps cropped to +-32 km
# north-south (was 40; the 25 km ring stays whole) so two cases fit <= 8 in.
LAYOUT = dict(fig_w=6.2, left=0.62, gap=0.12, cb_gap=0.10, cb_w=0.12, cb_lab=0.62,
              bot=0.42, c_h=1.05, row_gap=0.74, top=0.40)
MAP_HALF_Y_KM = 32.0


def map_view(plant, radius_km):
    """MODIS/map extent and the displayed (north-south cropped) extent, lon/lat."""
    lat0, lon0 = float(plant["lat"]), float(plant["lon"])
    ext = plant_extent(lat0, lon0, radius_km, 0.05)
    dy = min(MAP_HALF_Y_KM, radius_km) / 111.32
    return ext, (ext[0], ext[1], lat0 - dy, lat0 + dy)


WARN_H = 0.2                                               # extra top room for a MODIS-lag note


def case_block_h(plant, radius_km=50.0, modis_lag=None):
    """Height (in) of one case block of plot_case."""
    L = LAYOUT
    _, v = map_view(plant, radius_km)
    map_w = (L["fig_w"] - L["left"] - L["gap"] - L["cb_gap"] - L["cb_w"] - L["cb_lab"]) / 2
    map_h = map_w * (v[3] - v[2]) / ((v[1] - v[0]) * np.cos(np.radians(float(plant["lat"]))))
    return (L["bot"] + L["c_h"] + L["row_gap"] + map_h + L["top"]
            + (WARN_H if modis_lag_note(modis_lag) else 0.0))


def plot_case(seg, fits, plant, date, wind, height, rep, rows, outdir, method="nassar",
              modis_dir=None, radius_km=50.0, fig=None, fig_h=None, y0=0.0, tags="abc",
              modis_lag=None):
    """One case: (a, b) maps over MODIS, (c) along-track transect with the fits.  With
    fig given, draws the block with its bottom y0 inches up a figure fig_h inches tall
    (for the combined SI figure) instead of writing its own PNG."""
    ws, wd = wind
    lat0, lon0 = float(plant["lat"]), float(plant["lon"])
    # manuscript colours: B11 blue, corrected (model) vermillion
    cols = (("xco2_bc", XCO2_BC_LABEL, "#0072B2"), ("deep_ensemble_corrected_xco2", XCO2_MODEL_LABEL, "#D55E00"))
    halo = [pe.withStroke(linewidth=2.6, foreground="k", alpha=0.6)]

    # Map window in lon/lat as in plot_mauceri_comparison.py: radius + 0.05 deg around the
    # plant, 25 and 50 km distance circles, white star (same for a and b).
    arrows = [(wd, "-")]
    same_dir = fits.get("xco2_bc") is not None and abs(fits["xco2_bc"]["dwd"]) < 1
    if fits.get("xco2_bc") is not None and not same_dir:     # coincident arrows would hide the wind
        arrows.append(((wd + fits["xco2_bc"]["dwd"]) % 360, "--"))
    km_lon = 111.32 * np.cos(np.radians(lat0))
    ext, vext = map_view(plant, radius_km)
    # Wind arrows next to the plant: the arrow pair (common tail) sits at the position closest
    # to the plant that keeps it >= 4 km from every footprint and >= 4 km from the star, so it
    # stays near the source without hiding footprints.
    arrow_km = 16.0
    ux = np.array([np.sin(np.radians((d + 180) % 360)) for d, _ in arrows])
    uy = np.array([np.cos(np.radians((d + 180) % 360)) for d, _ in arrows])
    fx, fy = seg["ex"].to_numpy(), seg["ny"].to_numpy()
    frac = np.linspace(0, 1, 14)

    def arrow_pts(ox, oy):
        t0x, t0y = ox - 0.5 * arrow_km * ux.mean(), oy - 0.5 * arrow_km * uy.mean()
        px = np.concatenate([t0x + arrow_km * frac * a for a in ux])
        py = np.concatenate([t0y + arrow_km * frac * b for b in uy])
        return t0x, t0y, px, py

    def ok(o):
        _, _, px, py = arrow_pts(*o)
        return (np.min(np.hypot(fx[:, None] - px, fy[:, None] - py)) >= 4.0
                and np.min(np.hypot(px, py)) >= 4.0)

    grid = [(gx, gy) for gx in np.arange(-30, 31, 1.0) for gy in np.arange(-30, 31, 1.0)]
    centre = next((o for o in sorted(grid, key=lambda o: np.hypot(*o)) if ok(o)), (0.0, 0.0))
    t0x, t0y, px, py = arrow_pts(*centre)
    tail = (lon0 + t0x / km_lon, lat0 + t0y / 111.32)
    tips = [(lon0 + (t0x + arrow_km * a) / km_lon, lat0 + (t0y + arrow_km * b) / 111.32)
            for a, b in zip(ux, uy)]
    # Ring labels at the NE, SE or NW of the rings, whichever is clearest of track and arrows.
    def _clear(b):
        bx, by = 50 * np.sin(np.radians(b)), 50 * np.cos(np.radians(b))
        return min(np.min(np.hypot(fx - bx, fy - by)), np.min(np.hypot(px - bx, py - by)))
    ring_bearing = max((45, 135, 315), key=_clear)
    # Arrow key in the corner of (b) farthest from the track.
    hx, hy = 0.5 * (vext[1] - vext[0]) * km_lon, 0.5 * (vext[3] - vext[2]) * 111.32
    corners = {"upper right": (1, 1), "lower right": (1, -1), "upper left": (-1, 1)}
    key_loc = max(corners, key=lambda k: np.min(np.hypot(fx - corners[k][0] * (hx - 12),
                                                         fy - corners[k][1] * (hy - 8))))
    lon, lat = lon0 + seg["ex"].to_numpy() / km_lon, lat0 + seg["ny"].to_numpy() / 111.32

    # Layout in inches: two maps + colour bar on top, the transect below.
    fig_w, left, gap, cb_gap, cb_w, cb_lab = (LAYOUT[k] for k in
                                              ("fig_w", "left", "gap", "cb_gap", "cb_w", "cb_lab"))
    map_w = (fig_w - left - gap - cb_gap - cb_w - cb_lab) / 2
    map_h = map_w * (vext[3] - vext[2]) * 111.32 / ((vext[1] - vext[0]) * km_lon)
    bot, c_h, row_gap, top = (LAYOUT[k] for k in ("bot", "c_h", "row_gap", "top"))
    note = modis_lag_note(modis_lag) if modis_dir is not None else ""
    top += WARN_H if note else 0.0
    block_h = bot + c_h + row_gap + map_h + top
    own = fig is None
    if own:
        fig_h = block_h
        fig = plt.figure(figsize=(fig_w, fig_h))
    box = lambda x, y, w, h: fig.add_axes([x / fig_w, (y0 + y) / fig_h, w / fig_w, h / fig_h])
    y_map = bot + c_h + row_gap
    axes = [box(left, y_map, map_w, map_h), box(left + map_w + gap, y_map, map_w, map_h)]
    cax = box(left + 2 * map_w + gap + cb_gap, y_map, cb_w, map_h)

    vmin = min(np.nanpercentile(seg[c], 2) for c, *_ in cols)
    vmax = max(np.nanpercentile(seg[c], 98) for c, *_ in cols)
    bg_img = None
    if modis_dir is not None:                              # MODIS Aqua true colour (GIBS, EPSG:4326)
        try:
            bg_img = plt.imread(download_gibs_wms_rgb(
                date, ext, modis_dir / f"aqua_rgb_{date}_{plant['plant_id']}_{int(radius_km)}km.png"))
        except Exception as exc:                           # map still useful without imagery
            print(f"  MODIS imagery unavailable ({exc})")
    ring_fx = [pe.Stroke(linewidth=2.2, foreground="white", alpha=0.6), pe.Normal()]
    for i, (col, lab, _) in enumerate(cols):
        ax = axes[i]
        if bg_img is not None:
            ax.imshow(bg_img, extent=ext, origin="upper", interpolation="bilinear", zorder=0)
        sc = ax.scatter(lon, lat, c=seg[col], s=7, cmap=CMAPS["xco2"], vmin=vmin, vmax=vmax,
                        alpha=0.9, lw=0, rasterized=True, zorder=3)
        for rr, ls in ((25.0, "-"), (50.0, "--")):
            lon_c, lat_c = circle_lon_lat(lat0, lon0, rr)
            ax.plot(lon_c, lat_c, color="k", lw=1.0, ls=ls, zorder=5, path_effects=ring_fx)
            if i == 0:                                      # label the rings once
                # A label whose anchor would fall near or beyond the north-south crop
                # (the 50 km ring) slides along its ring to 80 % of the half-height, in
                # the same quadrant, and is written on the inner side of the ring.
                b, inner = ring_bearing, False
                half_y = min(MAP_HALF_Y_KM, radius_km)
                if abs(rr * np.cos(np.radians(b))) > 0.8 * half_y:
                    db = np.degrees(np.arccos(0.8 * half_y / rr))
                    b = {45: db, 135: 180 - db, 315: 360 - db}[ring_bearing]
                    inner = True
                k = int(round(b / 360 * (len(lon_c) - 1)))
                ax.text(lon_c[k], lat_c[k], f"{rr:.0f} km", fontsize=8.5,
                        ha=("left" if b < 180 else "right") if not inner
                        else ("right" if b < 180 else "left"),
                        va="bottom" if ring_bearing in (45, 315) else "top",
                        zorder=6, path_effects=[pe.withStroke(linewidth=2, foreground="w")])
        for (d, ls), (tx, ty) in zip(arrows, tips):
            an = ax.annotate("", xy=(tx, ty), xytext=tail, zorder=6,
                             arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22",
                                             color="w", lw=1.4, ls=ls, mutation_scale=9,
                                             shrinkA=0, shrinkB=0))
            an.arrow_patch.set_path_effects(halo)
        ax.scatter([lon0], [lat0], marker="*", s=130, c="w", edgecolor="k", lw=0.9, zorder=7,
                   label=plant["source_name"])
        ax.set_xlim(vext[0], vext[1]); ax.set_ylim(vext[2], vext[3])
        ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Lon (deg E)")
        if i == 0:
            ax.set_ylabel("Lat (deg N)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(lab, fontsize=10, pad=4)
        panel_label(ax, f"({tags[i]})")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85, handletextpad=0.3,
                  borderaxespad=0.4, markerscale=0.8)
    # Arrow key without a box: white text and lines with a dark halo.
    key = [Line2D([], [], color="w", lw=1.6, ls=ls, path_effects=halo) for _, ls in arrows]
    leg = axes[1].legend(key, [f"Wind {ws:.1f} m s$^{{-1}}$" + (" (= best fit)" if same_dir else ""),
                               "Best fit"][:len(key)], loc=key_loc,
                         fontsize=9, frameon=False, labelcolor="w", handlelength=1.8,
                         borderaxespad=0.3, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="k", alpha=0.7)])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label(r"$X_{\mathrm{CO2}}$ (ppm)")
    cb.ax.yaxis.set_major_locator(MaxNLocator(6))
    cb.solids.set_alpha(1)

    # (c) both series, each relative to its own fitted background
    ax = box(left, bot, 2 * map_w + gap, c_h)
    s = seg["s_km"].to_numpy()
    edges = np.arange(np.floor(s.min()), s.max() + 2, 2.0)
    mid = 0.5 * (edges[1:] + edges[:-1])
    binmean = lambda v, m: np.array([np.nanmean(v[m & (s >= lo) & (s < hi)])
                                     if (m & (s >= lo) & (s < hi)).any() else np.nan
                                     for lo, hi in zip(edges[:-1], edges[1:])])
    handles, labels, shown = [], [], []
    for j, (col, lab, c) in enumerate(cols):
        f, r = fits.get(col), rows[j]
        y = seg[col].to_numpy()
        if f is None or not np.isfinite(r["q_best_kgs"]):
            d = y - np.nanmedian(y)
            ax.plot(s, d, "o", color=c, ms=2.6, alpha=0.55, mew=0)
            shown.append(d[np.isfinite(d)])
            handles.append(Line2D([], [], color=c, marker="o", ms=3, mew=0, ls="none"))
            labels.append(f"{lab}  no plume")
            continue
        beta, ok, model = f["beta"], f["ok"], f["beta"][2] * f["kernel"]
        d = y - (beta[0] + beta[1] * s)
        if j == 0 and method == "nassar" and "plume" in f:
            sp, sb = s[f["plume"]], s[f["bgmask"]]
            ax.axvspan(sp.min(), sp.max(), color="0.35", alpha=0.18, lw=0, zorder=0)
            ax.text(0.5 * (sp.min() + sp.max()), 0.97, "plume", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color="0.35")
            for side in (sb[sb < sp.min()], sb[sb > sp.max()]):
                if len(side):
                    ax.axvspan(side.min(), side.max(), color="0.6", alpha=0.09, lw=0, zorder=0)
                    ax.text(0.5 * (side.min() + side.max()), 0.97, "background",
                            transform=ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=8.5, color="0.45")
        mfit = binmean(model, np.ones_like(ok))
        ax.plot(s[ok], d[ok], "o", color=c, ms=2.6, alpha=0.55, mew=0)
        ax.plot(mid, mfit, "-", color=c, lw=2.4, solid_capstyle="round", zorder=4,
                path_effects=[pe.withStroke(linewidth=3.6, foreground="w")])
        shown += [d[ok & np.isfinite(d)], mfit[np.isfinite(mfit)]]
        handles.append(Line2D([], [], color=c, lw=2.4))
        labels.append(f"{lab}  {r['q_best_kgs']:.0f} ± {r['q_best_se']:.0f}")
    handles = [Line2D([], [], color="none")] + handles + [Line2D([], [], color="none")]
    labels = ["Q (kg s$^{-1}$)"] + labels + [f"Reported  {rows[0]['q_reported_kgs']:.0f}"]
    # Robust y-range: keep the fitted model, let isolated outliers fall off the axes.
    v = np.concatenate(shown)
    lo, hi = np.percentile(v, [1, 99])
    for f in fits.values():
        if f is not None:
            hi = max(hi, float(np.nanmax(f["beta"][2] * f["kernel"])))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)                 # head-room for the shading labels
    ax.set_xlim(min(s.min(), -radius_km) - 1.5, max(s.max(), radius_km) + 1.5)
    ax.set_xlabel("Along-track distance (km)"); ax.set_ylabel(r"$\Delta X_{\mathrm{CO2}}$ (ppm)")
    ax.axhline(0, color="k", lw=0.6, zorder=1); ax.grid(alpha=0.25, lw=0.5)
    ax.legend(handles, labels, fontsize=9, ncol=len(handles), loc="lower right",
              bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=1.4, handletextpad=0.4,
              columnspacing=0.8, borderaxespad=0.1)
    panel_label(ax, f"({tags[2]})")
    fig.text(0.5, (y0 + block_h - 0.06) / fig_h, f"{plant['source_name']}, {date}",
             fontsize=10.5, ha="center", va="top")
    if note and bg_img is not None:                        # image not time-matched to OCO-2
        fig.text(0.5, (y0 + block_h - 0.06 - WARN_H) / fig_h, f"Note: {note}",
                 fontsize=8, ha="center", va="top", color="0.3")
    if not own:
        return
    out = outdir / f"us_emission_{plant['plant_id']}_{date}_{height}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    print(f"  wrote {out}")


def plot_si_figure(cases, outdir, name="us_emission_si"):
    """The SI figure: the case blocks of plot_case stacked top to bottom, panels (a)-(c),
    (d)-(f), ..."""
    cases = [c for c in cases if c in PLOT_INPUTS]
    if not cases:
        return
    hs = [case_block_h(PLOT_INPUTS[c][0][2], PLOT_INPUTS[c][1]["radius_km"],
                       PLOT_INPUTS[c][1]["modis_lag"]) for c in cases]
    space = 0.12                                           # extra gap between case blocks (in)
    fig_h = sum(hs) + space * (len(cases) - 1)
    fig = plt.figure(figsize=(LAYOUT["fig_w"], fig_h))
    letters = "abcdefghijkl"
    for k, c in enumerate(cases):
        a, kw = PLOT_INPUTS[c]
        plot_case(*a, **kw, fig=fig, fig_h=fig_h, y0=sum(hs[k + 1:]) + space * (len(cases) - 1 - k),
                  tags=letters[3 * k:3 * k + 3])
    out = outdir / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    print(f"  wrote {out}")


def plot_summary(tab, heights, outdir):
    cases = tab.drop_duplicates(["plant", "date"])[["plant", "date"]].values.tolist()
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(cases))
    nh = len(heights)
    width = 0.8 / nh
    markers = dict(zip(heights, "osPD^v"))
    for ih, h in enumerate(heights):
        for k, (col, c) in enumerate((("xco2_bc", "C0"), ("deep_ensemble_corrected_xco2", "C3"))):
            t = tab[(tab.series == col) & (tab.wind_height == h)].set_index(["plant", "date"])
            t = t.reindex([tuple(cc) for cc in cases])
            xx = x - 0.4 + width * (ih + 0.5) + (k - 0.5) * width * 0.35
            ax.errorbar(xx, t["q_best_kgs"], yerr=t["q_best_se"], fmt=markers[h], color=c,
                        ms=6, capsize=2, mfc=c if k == 0 else "w",
                        label=f"{'original' if k == 0 else 'corrected'}, wind {hlabel(h)}")
    t = tab[(tab.series == "xco2_bc") & (tab.wind_height == heights[0])].set_index(["plant", "date"])
    t = t.reindex([tuple(cc) for cc in cases])
    ax.plot(x, t["q_reported_kgs"], "k*", ms=16, label="reported (EPA CAMPD, overpass hour)")
    for xi, (_, r) in zip(x, t.iterrows()):
        if "does not cross" in r["flag"] or "no plume detected" in r["flag"]:
            ax.text(xi, -60, "no usable\nplume crossing", ha="center", va="top", fontsize=8, color="0.3")
        elif "cloud artifact" in r["flag"]:
            ax.text(xi, 950, "cloud artifact\n(S4), not a plume", ha="center", va="top", fontsize=8,
                    color="0.3")
    ax.set_xticks(x, [f"{p}\n{d}\n10 m: {w:.1f} m/s" for (p, d), w in zip(cases, t["ws_10m"])])
    ax.set_ylabel("CO$_2$ emission (kg s$^{-1}$)")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.set_title("Single-overpass Gaussian-plume estimates vs reported emissions, by wind height "
                 "(internal check; filled = original, open = corrected)", fontsize=11)
    fig.tight_layout()
    out = outdir / "us_emission_summary.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_BASE / "us_emissions")
    ap.add_argument("--radius-km", type=float, default=50.0)
    ap.add_argument("--method", choices=["nassar", "simple"], default="nassar",
                    help="nassar: Nassar et al. (2021, 2022) setup (unfiltered footprints, "
                         "background = mean outside plume + 3 km buffer, slant light path, "
                         "ERA5 + MERRA-2 wind at stack + 250 m); simple: earlier fit with "
                         "linear background and 3-sigma screening")
    ap.add_argument("--heights", nargs="+", default=None,
                    help="wind heights above ground in m, 'stack250' for the plume midline "
                         "(stack height + 250 m), or 'blh' for the 100 m-BLH mean "
                         "(heights other than 100 need era5_plant_winds_download.py)")
    ap.add_argument("--plot-height", default=None)
    ap.add_argument("--f-min", type=float, default=10.0,
                    help="F statistic (plume vs background-only fit) needed to call a plume detected")
    ap.add_argument("--modis-dir", type=Path, default=DEFAULT_PLOT_BASE / "_modis_tiles")
    ap.add_argument("--no-modis", action="store_true")
    ap.add_argument("--api-key", default="DEMO_KEY", help="api.data.gov key for EPA CAMPD")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="wind/emission cache (default <output-dir>/_cache); lets a scratch "
                         "--output-dir reuse the cached ERA5/MERRA-2/CAMPD inputs")
    args = ap.parse_args()

    if args.heights is None:
        args.heights = (["stack250_mix", "stack250", "300", "blh"] if args.method == "nassar"
                        else ["100", "300", "stack250", "500", "700", "blh"])
    if args.plot_height is None:
        args.plot_height = "stack250_mix" if args.method == "nassar" else "stack250"
    cache = args.cache_dir or args.output_dir / "_cache"  # shared by both methods
    if args.method == "simple":
        args.output_dir = args.output_dir / "simple"
        args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_manuscript_style()
    plants = read_plants(args.plants)
    for p_id, p in plants.items():
        p["plant_id"] = p_id
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"{case[0]} {case[1]}")
        rows += run_case(case, plants, args.plot_base, cache, args.api_key, args.radius_km,
                         args.output_dir, args.heights, args.f_min, args.plot_height,
                         method=args.method,
                         modis_dir=None if args.no_modis else args.modis_dir)
    tab = pd.DataFrame(rows)
    tab.to_csv(args.output_dir / "us_emission_estimates.csv", index=False)
    show = tab[["plant", "date", "series", "wind_height", "ws", "wd", "wind_track_angle_deg",
                "q_era5_kgs", "q_best_kgs", "q_best_se", "best_dwd", "F_detect",
                "q_reported_kgs", "flag"]].round(2)
    (args.output_dir / "us_emission_estimates.md").write_text(show.to_markdown(index=False))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(show.to_string(index=False))
    plot_summary(tab, [h for h in args.heights if h in set(tab.wind_height)], args.output_dir)
    plot_si_figure(SI_CASES, args.output_dir)


if __name__ == "__main__":
    main()
