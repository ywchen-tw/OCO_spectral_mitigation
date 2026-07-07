"""ak_harmonize.py — averaging-kernel / prior harmonization of TCCON vs OCO-2 (M2).

Implements the Rodgers & Connor (2003) comparison as applied to OCO-2 vs TCCON
by Wunch et al. (2017, AMT, appendix): predict what OCO-2 *should* retrieve if
TCCON is the truth proxy,

    c_TC->OCO2 = c_a,OCO2 + sum_j h_j a_j (x_TC,j - x_a,OCO2,j)

with, per level j on the OCO-2 20-level grid,
    h_j        pressure weighting function (Lite ``pressure_weight``),
    a_j        normalized XCO2 column averaging kernel (Lite ``xco2_averaging_kernel``),
    x_a,OCO2   OCO-2 prior CO2 profile (Lite ``co2_profile_apriori``),
    c_a,OCO2   OCO-2 prior XCO2 (Lite ``xco2_apriori``),
and x_TC the TCCON truth-proxy profile: the TCCON prior scaled by the retrieved
column scale factor gamma = xco2_TCCON / prior_xco2_TCCON (GGG is a profile-
scaling retrieval), converted from wet to dry mole fraction, and interpolated
onto the OCO-2 pressure grid in log-p.

WET->DRY (fix 2026-07-07): the GGG2020 public-file ``prior_co2`` profile is a
WET mole fraction, while the OCO-2 operator quantities (h, a, x_a, c_a) are all
dry-air.  The proxy therefore uses prior_co2 / (1 - prior_h2o).  Verified two
ways: dividing closes the OCO-grid proxy column against ``prior_xco2`` to
~0.2 ppm per case, and a native-grid closure test across all 20 TCCON sites
accepts the wet interpretation (+0.40 +/- 0.42 ppm) and rejects dry
(-1.11 +/- 0.57, worst at humid sites).  Without the conversion ak_delta was
biased by ~ -1.3 ppm x column-H2O-fraction (mean -0.93 ppm over the 75
production cases; fixed mean +0.34 +/- 0.55).

Case-level design: the reports compare footprints against a single per-case
TCCON window mean, so per-footprint identity is unnecessary — h, a, x_a, c_a
are averaged over Lite soundings (QF 0 and 1) within the collocation radius of the
station, and the adjustment is applied per TCCON observation before averaging.

The adjustment shifts ABSOLUTE biases only: before/after-correction improvement
metrics are invariant to it because the same soundings and the same AK sit on
both sides of the difference.

Requires the day's OCO-2 L2 Lite file (searched under <root>/data/OCO2/<year>/
<doy>/oco2_LtCO2_<yymmdd>_*.nc4, roots = CWD + storage dir + $OCO2_LITE_DIR).
Returns None when it cannot harmonize (missing Lite file / no soundings in
radius); callers then fall back to the un-harmonized TCCON mean and flag it.
"""
import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4 as nc4

# Pressure-unit conversion factors to hPa, keyed by the netCDF units attribute.
_P_TO_HPA = {'hpa': 1.0, 'mb': 1.0, 'mbar': 1.0, 'millibar': 1.0,
             'atm': 1013.25, 'pa': 0.01}


def _pressure_to_hpa(arr, units):
    fac = _P_TO_HPA.get(str(units).strip().lower())
    if fac is None:
        raise ValueError(f"Unrecognized pressure units {units!r}")
    return arr * fac


def find_lite_file(date_str, roots=None):
    """Locate the day-level OCO-2 Lite file for date_str ('YYYY-MM-DD').

    Searches <root>/data/OCO2/<year>/<doy:03d>/oco2_LtCO2_<yymmdd>_*.nc4* for
    each root (CWD by default, plus $OCO2_LITE_DIR both as a flat directory
    and as a data root).  Returns a Path or None.
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    yymmdd = dt.strftime('%y%m%d')
    doy = dt.timetuple().tm_yday
    roots = [Path(r) for r in (roots or ['.'])]
    env = os.environ.get('OCO2_LITE_DIR')
    patterns = []
    for root in roots:
        patterns.append(str(root / 'data' / 'OCO2' / f'{dt.year}' / f'{doy:03d}'
                            / f'oco2_LtCO2_{yymmdd}_*.nc4*'))
    if env:
        patterns.append(str(Path(env) / f'oco2_LtCO2_{yymmdd}_*.nc4*'))
        patterns.append(str(Path(env) / f'{dt.year}' / f'{doy:03d}'
                            / f'oco2_LtCO2_{yymmdd}_*.nc4*'))
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    return None


def _haversine_km(lon, lat, lon0, lat0):
    R = 6371.0088
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0, lat0 = np.radians(lon0), np.radians(lat0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def mean_oco2_operator(lite_path, st_lon, st_lat, radius_km, min_n=3):
    """Mean OCO-2 column operator near the station.

    Averages pressure_weight h, averaging kernel a, prior profile x_a (ppm),
    prior XCO2 c_a and pressure levels (hPa) over Lite soundings (QF 0 and 1)
    within radius_km of (st_lon, st_lat).  Returns dict or None if < min_n
    soundings.
    """
    with nc4.Dataset(lite_path, 'r') as ds:
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        qf = ds.variables['xco2_quality_flag'][:]
        d = _haversine_km(lon, lat, st_lon, st_lat)
        # Include BOTH quality flags (0=good, 1=bad): the ML correction operates
        # on near-cloud footprints that are largely QF==1, so the AK operator must
        # represent that same population (matches the QF-agnostic parquet path in
        # operator_from_dataframe).  np.isin guards against fill values.
        sel = np.where(np.isin(qf, (0, 1)) & (d <= radius_km))[0]
        if sel.size < min_n:
            return None
        h = np.ma.filled(ds.variables['pressure_weight'][sel], np.nan)
        a = np.ma.filled(ds.variables['xco2_averaging_kernel'][sel], np.nan)
        xa = np.ma.filled(ds.variables['co2_profile_apriori'][sel], np.nan)
        pl = _pressure_to_hpa(np.ma.filled(ds.variables['pressure_levels'][sel], np.nan),
                              getattr(ds.variables['pressure_levels'], 'units', 'hPa'))
        ca = np.ma.filled(ds.variables['xco2_apriori'][sel], np.nan)
    return dict(h=np.nanmean(h, axis=0), a=np.nanmean(a, axis=0),
                xa=np.nanmean(xa, axis=0), pl=np.nanmean(pl, axis=0),
                ca=float(np.nanmean(ca)), n_lite=int(sel.size))


# Parquet column prefixes for the flattened Lite column operator, as written by
# build_feature_dataset.compute_ak_columns (keep in sync with AK_COLUMN_MAP there).
_FRAME_PREFIXES = (('a', 'ak'), ('h', 'pwf'), ('xa', 'co2_ap'), ('pl', 'plev'))


def operator_from_dataframe(df, min_n=3):
    """Mean OCO-2 column operator from flattened parquet columns.

    Uses the ak_NN / pwf_NN / co2_ap_NN / plev_NN columns (written by
    build_feature_dataset from dual-fit-era fitting_details) plus the
    xco2_apriori column, averaged over the rows of *df* — normally the
    already-collocated footprints, which is strictly better than the
    Lite-file fallback (radius-averaged over all QF0 soundings).
    Returns the same dict shape as mean_oco2_operator(), or None when the
    columns are absent/empty (caller then falls back to the Lite file).
    """
    if len(df) < min_n or 'xco2_apriori' not in df.columns:
        return None
    out = {}
    for key, prefix in _FRAME_PREFIXES:
        cols = sorted(c for c in df.columns
                      if c.startswith(prefix + '_') and c[len(prefix) + 1:].isdigit())
        if not cols:
            return None
        arr = df[cols].to_numpy(float)
        row_ok = np.isfinite(arr).all(axis=1)
        if row_ok.sum() < min_n:
            return None
        out[key] = np.nanmean(arr[row_ok], axis=0)
    ca = df['xco2_apriori'].to_numpy(float)
    ca = ca[np.isfinite(ca) & (ca > 0)]
    if ca.size < min_n:
        return None
    return dict(h=out['h'], a=out['a'], xa=out['xa'], pl=out['pl'],
                ca=float(ca.mean()), n_lite=int(len(df)))


def ak_adjusted_ref_from_operator(op, tccon_nc_path, tmin, tmax, window_min=60.0):
    """AK/prior-harmonized TCCON reference given a prepared OCO-2 operator.

    op : dict from mean_oco2_operator() or operator_from_dataframe().
    Returns dict(tccon_ref_ak, tccon_sd_ak, n_tccon, ak_delta, n_lite) or None.
    ak_delta = harmonized mean − raw window mean (ppm): the smoothing/prior term.
    """
    if op is None:
        return None

    t0 = (np.datetime64(tmin) - np.timedelta64(int(window_min * 60), 's'))
    t1 = (np.datetime64(tmax) + np.timedelta64(int(window_min * 60), 's'))
    epoch = np.datetime64('1970-01-01T00:00:00')
    s0, s1 = float((t0 - epoch) / np.timedelta64(1, 's')), \
             float((t1 - epoch) / np.timedelta64(1, 's'))

    with nc4.Dataset(tccon_nc_path, 'r') as ds:
        t = np.ma.filled(ds.variables['time'][:], np.nan).astype(float)
        xco2 = np.ma.filled(ds.variables['xco2'][:], np.nan).astype(float)
        sel = np.where((t >= s0) & (t <= s1)
                       & np.isfinite(xco2) & (xco2 > 300) & (xco2 < 550))[0]
        if not sel.size:
            return None
        prior_xco2 = np.ma.filled(ds.variables['prior_xco2'][sel], np.nan).astype(float)
        prior_co2 = np.ma.filled(ds.variables['prior_co2'][sel], np.nan).astype(float)
        # prior_co2 is a WET mole fraction (see module docstring); prior_h2o is
        # needed to convert it to the dry-air basis of the OCO-2 operator.
        prior_h2o = np.ma.filled(ds.variables['prior_h2o'][sel], np.nan).astype(float)
        p_var = ds.variables['prior_pressure']
        prior_p = _pressure_to_hpa(np.ma.filled(p_var[sel], np.nan).astype(float),
                                   getattr(p_var, 'units', 'atm'))
        xco2 = xco2[sel]

    # log-p interpolation grid on the OCO-2 side (np.interp needs ascending x)
    logp_oco = np.log(op['pl'])
    order_oco = np.argsort(logp_oco)

    ests = []
    for i in range(xco2.size):
        if not (np.isfinite(prior_xco2[i]) and prior_xco2[i] > 0):
            continue
        gamma = xco2[i] / prior_xco2[i]
        # truth-proxy profile (ppm), wet -> dry mole fraction (clip guards
        # against pathological prior_h2o; q < 0.5 everywhere in practice)
        x_tc = gamma * prior_co2[i] / np.clip(1.0 - prior_h2o[i], 0.5, 1.0)
        m = np.isfinite(x_tc) & np.isfinite(prior_p[i]) & (prior_p[i] > 0)
        if m.sum() < 5:
            continue
        logp_tc = np.log(prior_p[i][m])
        order_tc = np.argsort(logp_tc)
        x_on_oco = np.empty_like(logp_oco)
        x_on_oco[order_oco] = np.interp(logp_oco[order_oco],
                                        logp_tc[order_tc], x_tc[m][order_tc])
        c_est = op['ca'] + np.nansum(op['h'] * op['a'] * (x_on_oco - op['xa']))
        if np.isfinite(c_est):
            ests.append(c_est)
    if not ests:
        return None
    ests = np.asarray(ests)
    raw_mean = float(np.mean(xco2))
    return dict(tccon_ref_ak=float(ests.mean()),
                tccon_sd_ak=float(ests.std()),
                n_tccon=int(ests.size),
                ak_delta=float(ests.mean() - raw_mean),
                n_lite=op['n_lite'])


def ak_adjusted_ref(lite_path, tccon_nc_path, st_lon, st_lat, radius_km,
                    tmin, tmax, window_min=60.0):
    """AK/prior-harmonized TCCON reference for one case, from the Lite file.

    Builds the mean OCO-2 operator over Lite soundings (QF 0 and 1) within radius_km
    of (st_lon, st_lat), then applies ak_adjusted_ref_from_operator().  Use
    operator_from_dataframe() instead when the collocated parquet already
    carries the flattened ak_NN/pwf_NN/co2_ap_NN/plev_NN columns.
    """
    op = mean_oco2_operator(lite_path, st_lon, st_lat, radius_km)
    return ak_adjusted_ref_from_operator(op, tccon_nc_path, tmin, tmax,
                                         window_min=window_min)
