"""Shared helpers for the WCO2 low-SOD spread study (2026-10-30).

Loads orbit 29252a_GL as ``final_model_test.load("GL")`` plus the L1B items the study needs
(per-sounding velocities, solar distance, the static bad-sample list and the per-sounding
spike-EOF residual flags), computes the per-footprint ILS-convolved solar spectrum on a grid of
Doppler factors, fits the revised WCO2 model with optional extra columns or channel masks, and
evaluates the metrics asked for (rms at SOD < 0.1, k1 median and far-clear spread, near-cloud
k1 contrast, corr with SCO2 k1).

Nothing here uses an L2-retrieved quantity as a fit input; cld_dist_km and xco2_qf are labels.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import final_model_test as fm  # noqa: E402
from h2o_analysis import bin_stats, lat_local_contrast  # noqa: E402
from orbit_io import BANDS  # noqa: E402
from run_study import robust_sd, spread_pct  # noqa: E402

OUT = HERE / "results"
C_LIGHT = 299792458.0
SOLAR_H5 = ROOT / "src/abs_util/abs/l2_solar_model.h5"
X_ALL = fm.X_ALL
LOW_SOD = 0.1
# Doppler-factor grid for the convolved solar spectrum (β = v/c), ±9 km/s in 0.25 km/s steps.
V_GRID = np.arange(-9000.0, 9000.0 + 1, 250.0)


# --------------------------------------------------------------------------- L1B extras


def l1b_track_index(l1b_sid: np.ndarray, sid: np.ndarray, fp: np.ndarray) -> np.ndarray:
    track = np.full(len(sid), -1)
    for ifp in range(8):
        col = l1b_sid[:, ifp]
        order = np.argsort(col, kind="stable")
        sel = fp == ifp
        pos = np.clip(np.searchsorted(col[order], sid[sel]), 0, len(col) - 1)
        hit = col[order][pos] == sid[sel]
        track[sel] = np.where(hit, order[pos], -1)
    return track


def l1b_extras(l1b: Path, sid: np.ndarray, fp: np.ndarray, spikes: bool = True) -> dict:
    """Velocities, solar distance, bad-sample list [3, 8, 1016], spike-EOF residual flags [3, N, 1016]."""
    with h5py.File(l1b, "r") as f:
        g = f["SoundingGeometry"]
        track = l1b_track_index(g["sounding_id"][...].astype(np.int64), sid, fp)
        if (track < 0).any():
            raise ValueError("sounding ids not found in L1B")
        out = dict(v_inst=g["sounding_relative_velocity"][...][track, fp].astype(float),
                   v_sol=g["sounding_solar_relative_velocity"][...][track, fp].astype(float),
                   dist_au=g["sounding_solar_distance"][...][track, fp] / 1.495978707e11,
                   bad=f["InstrumentHeader/bad_sample_list"][...], track=track)
        if spikes:
            out["spike"] = np.stack([f[f"SpikeEOF/spike_eof_weighted_residual_{n}"][...][track, fp]
                                     for n in ("o2", "weak_co2", "strong_co2")])
    return out


# --------------------------------------------------------------------------- solar spectrum


def ils_tables(l1b: Path) -> list[dict]:
    """Per band: channel wavelengths wl [8, 1016] (µm), ILS abscissa xx and response yy [8, 1016, 200]."""
    with h5py.File(l1b, "r") as f:
        coef = f["InstrumentHeader/dispersion_coef_samp"][...]
        dlam = f["InstrumentHeader/ils_delta_lambda"][...].astype(np.float64)
        resp = f["InstrumentHeader/ils_relative_response"][...].astype(np.float64)
    pix = np.arange(1, 1017, dtype=float)
    out = []
    for ib in range(3):
        wl = np.stack([sum(coef[ib, fp, k] * pix**k for k in range(coef.shape[2])) for fp in range(8)])
        out.append(dict(wl=wl, xx=dlam[ib], yy=resp[ib] / resp[ib].max(axis=2, keepdims=True)))
    return out


def solar_tables() -> list[dict]:
    """Per band: solar wavelength grid (µm) and absorption × continuum (production formula)."""
    sol = []
    with h5py.File(SOLAR_H5, "r") as f:
        for ib in range(3):
            g = f[f"Solar/Absorption/Absorption_{ib + 1}"]
            nu, val = g["wavenumber"][...], g["spectrum"][...]
            c = f[f"Solar/Continuum/Continuum_{ib + 1}"]
            cont = np.interp(nu, c["wavenumber"][...], c["spectrum"][...])
            sol.append(dict(wvl=1e4 / nu, fsol=val * cont, abs_only=val))
    return sol


def _conv_solar(wv, fs, wl, xx, yy):
    """ILS-weighted mean of fs on grid wv (µm, descending) at each channel (fp_abs_coeff formula)."""
    import resid_attrs as ra
    start, end = ra.conv_indices(wl, xx, wv)
    xw = xx + wl[:, None]
    out = np.empty(len(wl))
    for c in range(len(wl)):
        w = np.interp(wv[start[c]:end[c]], xw[c], yy[c])
        out[c] = np.dot(fs[start[c]:end[c]], w) / w.sum()
    return out


def solar_grid(l1b: Path, bands=(0, 1, 2), field="fsol") -> np.ndarray:
    """ln S_conv [3, 8, len(V_GRID), 1016] with the solar grid at λ_rest·(1 + v/c).

    Distance, cos(sza) and the 1/(1+β) bin-stretch factor are channel-independent and are left out:
    they only move the fitted intercept.
    """
    ils = ils_tables(l1b)
    sol = solar_tables()
    out = np.full((3, 8, len(V_GRID), 1016), np.nan)
    for ib in bands:
        for fp in range(8):
            for iv, v in enumerate(V_GRID):
                out[ib, fp, iv] = np.log(_conv_solar(sol[ib]["wvl"] * (1 + v / C_LIGHT), sol[ib][field],
                                                     ils[ib]["wl"][fp], ils[ib]["xx"][fp], ils[ib]["yy"][fp]))
    return out


def interp_v(grid_fp: np.ndarray, v: float) -> np.ndarray:
    """Linear interpolation of a [len(V_GRID), 1016] table at velocity v."""
    k = np.clip(np.searchsorted(V_GRID, v) - 1, 0, len(V_GRID) - 2)
    w = (v - V_GRID[k]) / (V_GRID[k + 1] - V_GRID[k])
    return (1 - w) * grid_fp[k] + w * grid_fp[k + 1]


# --------------------------------------------------------------------------- fitting


def base_mask(y, t, tw, s):
    return np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(tw)


def fit_wco2_ext(td, tw, x, y, extra=()):
    """Revised WCO2 model + extra columns (each enters with a free coefficient). Returns coef, resid."""
    a = np.column_stack([np.ones_like(td), x, -td, -tw, *extra])
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    return c, y - a @ c


def legendre_cols(x, order):
    """Legendre P2..P_order on x ∈ [−1, 1] (P0, P1 are already in the model)."""
    return [np.polynomial.legendre.Legendre.basis(k)(x) for k in range(2, order + 1)]


# --------------------------------------------------------------------------- metrics


def k1_metrics(k1, k1_sco2, meta, rms_low=None, rms_all=None, groups=None) -> dict:
    """k1 median, far-clear spread (%), near-cloud contrasts (0–1, 1–3 km), corr with SCO2 k1 (far-clear)."""
    cld = meta["cld_dist_km"].astype(float)
    qf, lat, fp = meta["xco2_qf"], meta["lat"].astype(float), meta["fp_number"].astype(int)
    far = (cld > 20) & (qf == 0)
    m = far & np.isfinite(k1)
    rec = dict(k1_med=np.median(k1[m]), k1_spread=spread_pct(k1[m], fp[m]))
    con = lat_local_contrast(k1, cld, qf, lat, fp, True)
    qf0 = qf == 0
    st = bin_stats(np.where(qf0, con, np.nan), cld, far)
    for lab in ("0-1", "1-3"):
        rec[f"con_{lab}"] = 100 * st[f"con_{lab}"]
        rec[f"snr_{lab}"] = st[f"snr_{lab}"]
        rec[f"n_{lab}"] = st[f"n_{lab}"]
    # correlation of per-footprint-demeaned far-clear k1 with SCO2 k1
    mm = m & np.isfinite(k1_sco2)
    a = pd.Series(k1[mm]).groupby(fp[mm]).transform(lambda s: s - s.median()).values
    b = pd.Series(k1_sco2[mm]).groupby(fp[mm]).transform(lambda s: s - s.median()).values
    rec["corr_sco2"] = np.corrcoef(a, b)[0, 1]
    rec["corr_sco2_raw"] = np.corrcoef(k1[mm], k1_sco2[mm])[0, 1]
    if groups is not None:
        # within-group version (groups = footprint × v_inst cluster): sounding-to-sounding agreement only
        g = groups[mm]
        a = pd.Series(k1[mm]).groupby(g).transform(lambda s: s - s.median()).values
        b = pd.Series(k1_sco2[mm]).groupby(g).transform(lambda s: s - s.median()).values
        rec["corr_sco2_within"] = np.corrcoef(a, b)[0, 1]
        rec["k1_spread_within"] = 100 * robust_sd(a / np.median(k1[mm]))
        cl = groups[mm] % 2
        rec["cluster_step"] = 100 * (np.median(k1[mm][cl == 1]) / np.median(k1[mm][cl == 0]) - 1)
    if rms_low is not None:
        rec["rms_low_far"] = np.nanmedian(rms_low[far])
        rec["rms_all_far"] = np.nanmedian(rms_all[far])
        rec["rms_low_allsnd"] = np.nanmedian(rms_low)
    return rec


__all__ = ["OUT", "BANDS", "X_ALL", "LOW_SOD", "V_GRID", "fm", "robust_sd", "spread_pct", "l1b_extras",
           "solar_grid", "interp_v", "base_mask", "fit_wco2_ext", "legendre_cols", "k1_metrics", "ils_tables"]
