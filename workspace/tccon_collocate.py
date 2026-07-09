"""tccon_collocate.py — shared OCO-2 ↔ TCCON collocation for the DE reports.

Both tccon_comparison_report.py and tccon_correction_policy_stats.py call
``collocate()`` so they select the IDENTICAL footprint set from the SAME
build_deepens_plot_data.py output (plot_data.parquet).  Previously each script
re-derived its own collocation (and policy_stats even re-ran the model into a
private cache), which made their figures disagree.  With one generator
(build_deepens_plot_data.py) and one collocator (this module) the two reports
compute from exactly the same footprints and corrected XCO2.

Guarded footprints (clim_guard | anomaly_guard — correction skipped, so the
corrected column == raw xco2_bc there) are KEPT and flagged with ``is_guarded``;
callers report the end-to-end number (guarded kept) and the drop-guards number.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the TCCON reader + haversine already used across the plotting scripts.
from plot_corrected_xco2 import load_tccon, _haversine_km, get_storage_dir  # noqa: F401

GUARD_COLS = ('clim_guard', 'anomaly_guard')

# Published TCCON station coordinates (lon, lat), keyed by the 2-letter site
# code (= first two characters of the GGG2020 file name).  Values are the
# constant per-spectrum lat/long metadata of the GGG2020 .public.qc.nc files
# (site coordinates as published by the PIs; file precision 0.01° ≈ 1 km,
# negligible against the 50–100 km collocation radii).  collocate() prefers
# these over the median-observation fallback (M2 polish, 2026-07-08): a fixed
# table is deterministic across file updates and immune to files whose
# coordinate variables are unreadable (e.g. pre-qc 'ae').
SITE_COORDS = {
    'an': (126.3300, 36.5400),   # Anmyeondo
    'bu': (120.6500, 18.5300),   # Burgos
    'ci': (-118.1300, 34.1400),  # Caltech (Pasadena)
    'db': (130.9300, -12.4600),  # Darwin
    'df': (-117.8800, 34.9600),  # Edwards (Dryden/Armstrong)
    'et': (-104.9900, 54.3500),  # East Trout Lake
    'hf': (117.1700, 31.9100),   # Hefei
    'iz': (-16.5000, 28.3100),   # Izaña
    'js': (130.2900, 33.2400),   # Saga
    'ka': (8.4400, 49.1000),     # Karlsruhe
    'ma': (-60.6000, -3.2100),   # Manaus
    'ny': (11.9200, 78.9200),    # Ny-Ålesund
    'oc': (-97.4900, 36.6000),   # Lamont (Oklahoma)
    'or': (2.1100, 47.9600),     # Orléans
    'pa': (-90.2700, 45.9400),   # Park Falls
    'pr': (2.3600, 48.8500),     # Paris
    'ra': (55.4800, -20.9000),   # Réunion (St-Denis)
    'rj': (143.7700, 43.4600),   # Rikubetsu
    'wg': (150.8800, -34.4100),  # Wollongong
    'xh': (116.9600, 39.8000),   # Xianghe
}


def find_plotdata(base, date, site):
    """Locate the build_deepens_plot_data.py output for one case under *base*
    (= the plot script's OUT_BASE).  Mirrors its outdir naming:
    combined_<date>_<site>/ (site set) or combined_<date>_all_orbits/."""
    base = Path(base)
    for name in (f"combined_{date}_{site}", f"combined_{date}_all_orbits"):
        p = base / name / 'plot_data.parquet'
        if p.exists():
            return p
    return None


def collocate(oco, tccon, *, box, radius_km, window_min, sanity_ppm=50.0,
              site=None):
    """Collocate one case's OCO-2 footprints to its TCCON station.

    Parameters
    ----------
    oco : DataFrame  plot_data.parquet (time/lon/lat/xco2_bc/…/guard columns).
    tccon : DataFrame  loaded TCCON record (time, xco2, lon, lat).
    box : (lonmin, lonmax, latmin, latmax)  lon/lat window from the run_case line.
    radius_km, window_min : collocation thresholds (spatial, temporal).
    sanity_ppm : drop footprints whose xco2_bc is >this from the TCCON mean
                 (gross retrieval failures); None to disable.
    site : 2-letter TCCON site code; when it resolves in SITE_COORDS the
           published station coordinate is used (median-observation lat/lon
           remains the fallback, with a warning if the two disagree > 5 km).

    Returns dict:
      near       : footprints in box AND ≤radius of station (guarded KEPT),
                   with an added bool ``is_guarded`` column.
      tccon_ref, tccon_sd, n_tccon : TCCON window-mean / std / count.
      st_lon, st_lat : station location used.
    """
    lonmin, lonmax, latmin, latmax = box
    sel = ((oco['lon'] >= lonmin) & (oco['lon'] <= lonmax) &
           (oco['lat'] >= latmin) & (oco['lat'] <= latmax))
    oco = oco[sel]
    med_lon = float(tccon['lon'].median()) if len(tccon) else np.nan
    med_lat = float(tccon['lat'].median()) if len(tccon) else np.nan
    pub = SITE_COORDS.get(str(site).lower()[:2]) if site else None
    if pub is not None:
        st_lon, st_lat = pub
        if np.isfinite(med_lon):
            d_km = float(_haversine_km(np.array([med_lon]), np.array([med_lat]),
                                       st_lon, st_lat)[0])
            if d_km > 5.0:
                print(f"  ⚠ collocate[{site}]: published station coordinate is "
                      f"{d_km:.1f} km from the median-observation position — "
                      f"check the site table / file pairing")
    else:
        if site:
            print(f"  ⚠ collocate[{site}]: site not in SITE_COORDS — falling "
                  f"back to median-observation lat/lon")
        st_lon, st_lat = med_lon, med_lat
    out = dict(near=oco.iloc[0:0].copy(), tccon_ref=np.nan, tccon_sd=np.nan,
               n_tccon=0, tccon_err_mean=np.nan, st_lon=st_lon, st_lat=st_lat)
    if not len(oco) or not np.isfinite(st_lon):
        return out
    d = _haversine_km(oco['lon'].values, oco['lat'].values, st_lon, st_lat)
    near = oco[d <= radius_km].copy()
    if not len(near):
        return out

    # TCCON window-mean around the footprint time span.
    tref = tsd = terr = np.nan
    n_tc = 0
    if len(tccon) and 'time' in near.columns:
        ot = pd.to_datetime(near['time'], unit='s', utc=True, errors='coerce').dropna()
        if len(ot):
            w = pd.Timedelta(minutes=window_min)
            sub = tccon[(tccon['time'] >= ot.min() - w) & (tccon['time'] <= ot.max() + w)]
            if len(sub):
                tref = float(sub['xco2'].mean()); tsd = float(sub['xco2'].std()); n_tc = len(sub)
                if 'xco2_error' in sub.columns:
                    terr = float(np.nanmean(sub['xco2_error'].to_numpy(float)))

    # Gross-outlier sanity band vs TCCON (retrieval failures the guards missed).
    if sanity_ppm and np.isfinite(tref):
        near = near[np.abs(near['xco2_bc'].to_numpy(float) - tref) < sanity_ppm]
        if not len(near):
            return dict(out, tccon_ref=tref, tccon_sd=tsd, n_tccon=n_tc,
                        tccon_err_mean=terr)

    g = np.zeros(len(near), bool)
    for gc in GUARD_COLS:
        if gc in near.columns:
            g |= near[gc].to_numpy(bool)
    near['is_guarded'] = g
    return dict(near=near, tccon_ref=tref, tccon_sd=tsd, n_tccon=n_tc,
                tccon_err_mean=terr, st_lon=st_lon, st_lat=st_lat)


def series_stats(oco_vals, tccon_ref):
    """Bias / RMSE / std of (oco_vals − tccon_ref) over finite footprints."""
    d = np.asarray(oco_vals, float) - tccon_ref
    d = d[np.isfinite(d)]
    if not d.size:
        return dict(n=0, bias=np.nan, rmse=np.nan, std=np.nan)
    return dict(n=int(d.size), bias=float(d.mean()),
                rmse=float(np.sqrt(np.mean(d ** 2))), std=float(d.std()))
