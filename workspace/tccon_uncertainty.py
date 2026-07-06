"""tccon_uncertainty.py — Side-B (AK-harmonized TCCON reference) uncertainty budget
for the uncertainty-aware TCCON comparison (Phase 3;
src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md §3).

Per case, the harmonized reference c̄_TC (ak_harmonize.ak_adjusted_ref_from_operator)
carries:

    u_TC² =  u_meas²/N_tccon          TCCON retrieval precision (xco2_error)
          +  u_temporal²/N_eff        window scatter of the harmonized obs (tccon_sd_ak)
          +  u_harm²                  AK/prior-leakage: prior error through h·(1−a)
          +  u_colloc²                representativeness: point station vs ≤R-km footprints
          (+ u_scale²)                TCCON WMO-scale systematic ~0.4 ppm — absolute only

Design notes:
  * u_harm uses the closure identity c_est = Σ h_j[a_j x_TC,j + (1−a_j) x_a,j]; the
    prior-dependent part is Σ h_j(1−a_j)x_a,j, so a per-level prior error σ_prior,j
    propagates as u_harm² = Σ_j [h_j(1−a_j)]² σ_prior,j²  (independent-level lower
    bound; a coherent prior tilt gives a similar few-tenths ppm — see §1 trace).
  * u_colloc fits a plane to the corrected XCO2 over (lon,lat) and multiplies the
    horizontal gradient by the mean station→footprint distance.
"""
from __future__ import annotations

import numpy as np

# Nominal per-sigma-level OCO-2 prior CO2 uncertainty (ppm): well-mixed
# troposphere ~1 ppm, stratosphere ~3 ppm (seasonal/transport, weakly constrained).
# sigma = p/psurf on the OCO-2 20-level grid (op['pl']/psurf).
_PRIOR_TROP_PPM = 1.0
_PRIOR_STRAT_PPM = 3.0


def _sigma_prior_profile(op):
    """Per-level prior-CO2 uncertainty (ppm) on the OCO-2 grid, from a simple
    tropopause-split model keyed on sigma = p/psurf (strat = sigma < 0.3)."""
    pl = np.asarray(op['pl'], float)
    sigma = pl / np.nanmax(pl)
    return np.where(sigma < 0.3, _PRIOR_STRAT_PPM, _PRIOR_TROP_PPM)


def u_harm(op, sigma_prior=None):
    """AK/prior-leakage uncertainty of the harmonized reference (ppm).

    u_harm² = Σ_j [h_j (1−a_j)]² σ_prior,j²  — prior errors matter only where the
    kernel a_j departs from 1 (weighted by pressure weight h_j).
    """
    h = np.asarray(op['h'], float)
    a = np.asarray(op['a'], float)
    sp = _sigma_prior_profile(op) if sigma_prior is None else np.asarray(sigma_prior, float)
    w = h * (1.0 - a)
    return float(np.sqrt(np.nansum((w * sp) ** 2)))


def _haversine_km(lon, lat, lon0, lat0):
    R = 6371.0088
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0, lat0 = np.radians(lon0), np.radians(lat0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def u_colloc(lon, lat, xco2, st_lon, st_lat):
    """Representativeness uncertainty (ppm): horizontal XCO2 gradient × mean
    station→footprint distance.

    Fits xco2 ≈ b0 + bx·(lon−st_lon)·kx + by·(lat−st_lat)·ky (km-scaled), then
    returns |∇xco2| · mean(dist_to_station).  Returns 0 when < 5 footprints.

    The gradient×distance extrapolation is CAPPED at the robust spatial spread
    of the field (1.4826·MAD): an ill-conditioned plane fit (near-degenerate
    footprint geometry, or a residual contamination outlier that survived the
    sanity band) can blow the OLS gradient up to tens of ppm, but the point
    station cannot credibly differ from the footprint-cloud mean by more than the
    field's own variability.  Fit uses a Huber-style outlier trim first.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    xco2 = np.asarray(xco2, float)
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(xco2)
    if m.sum() < 5:
        return 0.0
    lo, la, xc = lon[m], lat[m], xco2[m]
    # robust field spread (cap) and outlier trim (>4·robust-σ from the median)
    med = np.median(xc)
    mad = np.median(np.abs(xc - med))
    robust_sd = 1.4826 * mad if mad > 0 else float(np.std(xc))
    keep = np.abs(xc - med) <= 4.0 * robust_sd if robust_sd > 0 else np.ones(xc.size, bool)
    if keep.sum() >= 5:
        lo, la, xc = lo[keep], la[keep], xc[keep]
    lat0 = np.mean(la)
    kx = 111.32 * np.cos(np.radians(lat0)); ky = 111.32
    ex = (lo - st_lon) * kx           # km east of station
    ny = (la - st_lat) * ky           # km north of station
    A = np.column_stack([np.ones(ex.size), ex, ny])
    coef, *_ = np.linalg.lstsq(A, xc, rcond=None)
    grad = float(np.hypot(coef[1], coef[2]))          # ppm per km
    mean_d = float(np.mean(_haversine_km(lo, la, st_lon, st_lat)))
    return float(min(grad * mean_d, robust_sd)) if robust_sd > 0 else grad * mean_d


def side_b_uncertainty(op, ak_res, footprints=None, station=None,
                       tccon_errors=None, n_eff_tccon=None, u_scale=None,
                       corr_col=None):
    """Assemble u_TC (ppm) and its components for one case.

    op        : OCO-2 operator dict (ak_harmonize.operator_from_dataframe).
    ak_res    : dict from ak_harmonize.ak_adjusted_ref_from_operator
                (needs tccon_sd_ak, n_tccon).
    footprints: DataFrame with lon/lat + a corrected-XCO2 column, for u_colloc.
    station   : (lon, lat).
    tccon_errors : per-obs TCCON xco2_error array (ppm) → u_meas; optional.
    n_eff_tccon  : effective independent count for the window scatter; default
                   n_tccon (TCCON obs are ~minutes apart, largely independent).
    u_scale   : TCCON WMO-scale systematic (ppm); include only for absolute claims.
    corr_col  : name of the corrected-XCO2 column to use for u_colloc (so the
                representativeness gradient is measured on the CLEAN corrected
                field, not xco2_bc).  Falls back to the first of
                deep_ensemble_corrected_xco2 / xco2_bc / xco2_corrected present.
    """
    n_tc = int(ak_res.get('n_tccon', 0)) or 1
    sd = float(ak_res.get('tccon_sd_ak', np.nan))
    neff = n_tc if n_eff_tccon is None else max(1, int(n_eff_tccon))

    u_temporal = (sd / np.sqrt(neff)) if np.isfinite(sd) else 0.0
    u_meas = (float(np.nanmean(np.asarray(tccon_errors, float))) / np.sqrt(n_tc)
              if tccon_errors is not None and len(tccon_errors) else 0.0)
    uh = u_harm(op)
    uc = 0.0
    if footprints is not None and station is not None:
        cands = ([corr_col] if corr_col else []) + \
            ['deep_ensemble_corrected_xco2', 'xco2_bc', 'xco2_corrected']
        col = next((c for c in cands if c and c in footprints.columns), None)
        if col is not None:
            uc = u_colloc(footprints['lon'], footprints['lat'], footprints[col],
                          station[0], station[1])
    parts = dict(u_meas=u_meas, u_temporal=u_temporal, u_harm=uh, u_colloc=uc)
    var = u_meas**2 + u_temporal**2 + uh**2 + uc**2
    if u_scale:
        parts['u_scale'] = float(u_scale); var += float(u_scale) ** 2
    return dict(u_TC=float(np.sqrt(var)), **parts)
