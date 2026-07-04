"""Clear-sky-reference XCO2 anomaly — the training-target definition.

For each sounding, the anomaly is XCO2 minus the mean XCO2 of clear-sky
(cld_dist > min_cld_dist) soundings within ±lat_thres° latitude, kept only
where the reference std ≤ std_thres.  Production parameters live in
src/constants.py (0.25° / 1.0 ppm / 10 km; r05/r15 alternates override
min_cld_dist).

Split out of fitting.py (2026-07, review §7.4); fitting.py re-exports
compute_xco2_anomaly.
"""

# When run directly / imported with only src/spectral on sys.path, add src/
# so sibling top-level modules (utils, abs_util, constants, config) resolve.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import numpy as np

from constants import (ANOMALY_LAT_THRES_DEG, ANOMALY_STD_THRES_PPM,
                       ANOMALY_MIN_CLD_DIST_KM)


def compute_xco2_anomaly(fp_lat, cld_dist_km, xco2,
                         lat_thres=ANOMALY_LAT_THRES_DEG,
                         std_thres=ANOMALY_STD_THRES_PPM,
                         min_cld_dist=ANOMALY_MIN_CLD_DIST_KM,
                         chunk_size=128, extra_vars=None):
    """XCO2 anomaly relative to nearby clear-sky soundings.

    For each footprint i, the reference set is all footprints within ±lat_thres°
    latitude that are more than min_cld_dist km from a cloud.  The anomaly is
    defined only when the reference std is below std_thres (stable background).

    Parameters
    ----------
    fp_lat      : [N] footprint latitudes (may contain NaN)
    cld_dist_km : [N] nearest-cloud distance in km (may contain NaN)
    xco2        : [N] XCO2 values (may contain NaN)
    lat_thres   : float, half-width of latitude search window (degrees)
    std_thres   : float, maximum allowed std of reference XCO2 (ppm)
    min_cld_dist: float, minimum cloud distance to be considered clear-sky (km)
    chunk_size  : int, rows processed per iteration (controls peak memory).
                  Peak memory per chunk ≈ chunk_size × N × 24 bytes (three
                  [chunk, N] arrays).  Default 32 → ~65 MB for N=84 000.
    extra_vars  : dict[str, np.ndarray] | None
                  Optional additional [N] arrays whose mean and std over the
                  same clear-sky reference window are returned alongside the
                  anomaly.  When provided, returns (anomaly, means, stds).

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy.
    If extra_vars is not None, also returns:
    extra_means : dict[str, [N] float array]  per-sounding reference mean
    extra_stds  : dict[str, [N] float array]  per-sounding reference std
    """
    N          = len(fp_lat)
    chunk_size = int(chunk_size)   # guard against float passed via **kwargs
    anomaly  = np.full(N, np.nan)
    ref_mean = np.full(N, np.nan)
    ref_std  = np.full(N, np.nan)

    valid_lat  = ~np.isnan(fp_lat)
    clear_mask = valid_lat & (cld_dist_km > min_cld_dist) & ~np.isnan(xco2)  # [N] bool

    # Extend clear_mask to require valid values for ALL extra_vars so that
    # every reference sounding contributes to every variable (shared reference pool).
    if extra_vars is not None:
        for v in extra_vars.values():
            clear_mask = clear_mask & ~np.isnan(np.asarray(v))

    # Pre-extract clear-sky reference latitudes and xco2 values to avoid
    # broadcasting the full array on every chunk.
    ref_lat  = np.where(clear_mask, fp_lat, np.nan)          # [N]
    ref_xco2 = np.where(clear_mask, xco2,   np.nan)          # [N]

    # Pre-extract extra reference arrays (if any)
    extra_ref      = {}
    extra_mean_out = {}
    extra_std_out  = {}
    if extra_vars is not None:
        for k, v in extra_vars.items():
            extra_ref[k]      = np.where(clear_mask, np.asarray(v), np.nan)
            extra_mean_out[k] = np.full(N, np.nan)
            extra_std_out[k]  = np.full(N, np.nan)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        q_lat = fp_lat[start:end]                             # [chunk]

        # [chunk, N] pairwise latitude difference for this row block only
        lat_diff  = np.abs(q_lat[:, None] - ref_lat[None, :])   # [chunk, N]
        in_window = lat_diff <= lat_thres                        # [chunk, N] bool

        # xco2 of clear-sky refs within window; NaN elsewhere
        xco2_win  = np.where(in_window, ref_xco2[None, :], np.nan)  # [chunk, N]

        n_refs   = np.sum(~np.isnan(xco2_win), axis=1)          # [chunk] clear-sky refs in window
        has_refs = n_refs >= 5                                    # [chunk] require ≥5 for meaningful std

        chunk_mean = np.full(end - start, np.nan)
        chunk_std  = np.full(end - start, np.nan)
        if has_refs.any():
            chunk_mean[has_refs] = np.nanmean(xco2_win[has_refs], axis=1)
            chunk_std[has_refs]  = np.nanstd( xco2_win[has_refs], axis=1)

        ref_mean[start:end] = chunk_mean
        ref_std[start:end]  = chunk_std

        # Only populate extra vars for soundings whose XCO2 background is stable
        # (chunk_std <= std_thres).  Noisy-ref soundings keep NaN in-loop rather
        # than relying solely on the post-hoc not_valid mask below.
        has_stable = has_refs & (chunk_std <= std_thres)

        if extra_vars is not None and has_stable.any():
            for k, ref_v in extra_ref.items():
                ev_win = np.where(in_window, ref_v[None, :], np.nan)
                chunk_emean = np.full(end - start, np.nan)
                chunk_estd  = np.full(end - start, np.nan)
                chunk_emean[has_stable] = np.nanmean(ev_win[has_stable], axis=1)
                chunk_estd[has_stable]  = np.nanstd( ev_win[has_stable], axis=1)
                extra_mean_out[k][start:end] = chunk_emean
                extra_std_out[k][start:end]  = chunk_estd

        del lat_diff, in_window, xco2_win

    valid = valid_lat & ~np.isnan(ref_mean) & (ref_std <= std_thres)
    anomaly[valid] = xco2[valid] - ref_mean[valid]
    if extra_vars is not None:
        not_valid = ~valid
        for k in extra_mean_out:
            extra_mean_out[k][not_valid] = np.nan
            extra_std_out[k][not_valid]  = np.nan
        return anomaly, extra_mean_out, extra_std_out
    return anomaly

