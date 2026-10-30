"""Per-channel attributes for the residual decomposition (orbit 29252a_GL).

Re-runs the production optical-depth step (``src/abs_util/fp_abs_coeff.py``
``_process_track_all_bands`` with the atmosphere of ``src/abs_util/fp_atm.py``)
for each sounding and, inside the same transmission-space ILS convolution,
splits every channel's τ into layers and species:

    τ_c = −ln Σ_ν w_ν e^{−τ_hr(ν)} / Σ_ν w_ν,    τ_hr = Σ_z A_z(ν) + A_ray(ν).

τ_c(s) = −ln⟨e^{−s τ_hr}⟩_w is concave in s with dτ_c/ds = ⟨τ_hr⟩_{p_s},
p_s ∝ w e^{−s τ_hr}, so the integrated-gradient attribution

    C_z = ∫_0^1 ⟨A_z⟩_{p_s} ds

sums exactly to τ_c (Rayleigh is one more component). The s-integral uses
Gauss–Legendre nodes in u with s = u² (nodes crowd near s = 0, where the deep
cores change ⟨τ_hr⟩_{p_s} fastest). Attributes per channel:

- ``eta``      absorber-weighted mean altitude Σ C_z h_z / Σ C_z (km, O2/CO2 + H2O)
- ``eta_jac``  same with the marginal (s = 1) weights p_1: where a perturbation
               of the absorber changes ln T
- ``f_h2o``    τ_H2O / τ_c;  ``f_ray``: τ_Rayleigh / τ_c
- ``f_bl``     share of the absorber contribution below 1.5 km
- ``dtau_dpix``  ∂τ_c/∂(wavelength shift in pixels), central difference ±0.1 px
- ``dtau_dlnw``  ∂τ_c/∂ln(ILS width), stretch of the ILS abscissa by ±2 %
- ``dtau_dT``    ∂τ_c/∂T (per K) for a uniform ±2 K shift of the temperature used in
               the ABSCO lookup only (layer columns n·dz are T-invariant at fixed
               pressure, so this is the line-strength/width temperature sensitivity)
- ``tau_rep``  reproduced τ_c (validation against ``{band}_tau_output``)

Small pure helpers are copied from ``src/abs_util`` (the originals import
numba, which is not installed in the default environment); see attributions.

Usage:
    python idea_test/fit_order_study/resid_attrs.py [--subset all|N] [--workers 8]
"""

from __future__ import annotations

import argparse
import bisect
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from orbit_io import BANDS  # noqa: E402

ORBIT = "29252a_GL"
DATA = ROOT / "data/OCO2/2020/001/29252a_GL"
L1B = DATA / "oco2_L1bScGL_29252a_191231_B11006r_220715045716.h5"
MET = DATA / "oco2_L2MetGL_29252a_191231_B11006r_220624111442.h5"
CPR = DATA / "oco2_L2CPrGL_29252a_191231_B11006r_220624112210.h5"
FP_TAU = ROOT / "results/2020-01-01/29252a_GL/fp_tau_combined.h5"
FIT_DETAILS = ROOT / "results/fitting_details/fitting_details_2020-01-01_29252a_GL.h5"
ABSCO_DIR = ROOT / "src/abs_util/abs"

XR = {0: (0.755, 0.784), 1: (1.587, 1.629), 2: (2.020, 2.101)}   # µm, as production
GAS = {0: ("o2_v52.hdf", "Gas_07_Absorption"), 1: ("co2_v52.hdf", "Gas_02_Absorption"),
       2: ("co2_v52.hdf", "Gas_02_Absorption")}
O2MIX = 0.20935
N_QUAD = 32            # Gauss–Legendre nodes for the s-integral
SHIFT_PIX = 0.1        # dispersion-signature step (pixels)
WIDTH_EPS = 0.02       # ILS-width step (relative)
DT_K = 2.0             # temperature step for dtau_dT (K)
BL_KM = 1.5
TRILINEAR = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 1, 0, 0, 0],
                      [-1, 0, 1, 0, 0, 0, 0, 0],
                      [-1, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, -1, 0, -1, 0, 1, 0],
                      [1, -1, -1, 1, 0, 0, 0, 0],
                      [1, -1, 0, 0, -1, 1, 0, 0],
                      [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=np.float64)
ATTRS = ("tau_rep", "eta", "eta_jac", "f_h2o", "f_ray", "f_bl", "dtau_dpix", "dtau_dlnw",
         "dtau_dT", "closure")

# GEOS-5 72-level hybrid coefficients (copied from src/abs_util/fp_atm.py).
AP = np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
               3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
               7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
               1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
               2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
               2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
               7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
               2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
               9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
               2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
               6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
               1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
               1.000000e-02])
BP = np.zeros(73)
BP[:32] = [1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
           8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
           7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
           5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
           2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
           6.960025e-03, 8.175413e-09]


# --------------------------------------------------------------------------- copied helpers


def find_boundary(nu1, nu2, nu):
    """Copy of src/abs_util/abs/find_bound.py:find_boundary (no warning print)."""
    inu1, inu2 = np.searchsorted(nu, [nu1, nu2], side="left")
    if inu2 == len(nu):
        inu2 = inu2 - 1
    if nu[inu2] - nu[inu2 - 1] > 10:
        inu2 = inu2 - 1
    return inu1, inu2


def p_index(pobs, hpa):
    """Copy of src/abs_util/abs/get_index.py:get_P_index (trilinear)."""
    return np.clip(np.searchsorted(hpa, pobs, side="left") - 1, 0, len(hpa) - 2)


def t_index(tkobs, tk, p_ind):
    """Copy of src/abs_util/abs/get_index.py:get_T_index (trilinear, no warning)."""
    t_ind = np.searchsorted(tk[p_ind, :], tkobs, side="left") - 1
    return max(0, min(t_ind, tk.shape[1] - 2))


def absco_trilinear(tab, hpa, tk, broad, tkobs, pobs, t_ind, p_ind, h2o_vmr):
    """Copy of src/abs_util/abs/rdabsco_gas_absco.py:rdabsco_species_absco, mode 'trilinear'.

    ``tab`` is the band slice of Gas_XX_Absorption [P, T, broad, nu].
    """
    hh = bisect.bisect_left(broad, h2o_vmr) - 1
    hh = max(0, min(hh, len(broad) - 2))
    dp = (pobs - hpa[p_ind]) / (hpa[p_ind + 1] - hpa[p_ind])
    dt = (tkobs - tk[p_ind, t_ind]) / (tk[p_ind, t_ind + 1] - tk[p_ind, t_ind])
    dh = (h2o_vmr - broad[hh]) / (broad[hh + 1] - broad[hh])
    block = tab[p_ind:p_ind + 2, t_ind:t_ind + 2, hh:hh + 2, :].reshape(8, -1).astype(np.float64)
    q = np.array([1.0, dp, dt, dh, dp * dt, dt * dh, dh * dp, dp * dt * dh])
    return (q @ TRILINEAR) @ block


def conv_indices(wvloco, xx, wvldat):
    """Copy of src/abs_util/abs/oco_convolve_absco.py:oco_conv_absco → (start, end) per channel."""
    nwav = len(wvldat)
    lo = wvloco + np.min(xx, axis=1)
    hi = wvloco + np.max(xx, axis=1)
    rev = wvldat[::-1]
    il = np.clip(np.searchsorted(rev, lo, side="left") - 1, 0, nwav - 1)
    ir = np.clip(np.searchsorted(rev, hi, side="right"), 0, nwav - 1)
    il, ir = nwav - 1 - il, nwav - 1 - ir
    return ir, il + 1


def rayleigh_xsec(wv0):
    """Copy of src/abs_util/fp_abs_coeff.py:mol_ext_wvl (Bodhaine 1999, 1e-28 cm²)."""
    num = 1.0455996 - 341.29061 * wv0**(-2.0) - 0.90230850 * wv0**2.0
    den = 1.0 + 0.0027059889 * wv0**(-2.0) - 85.968563 * wv0**2.0
    return num / den


# --------------------------------------------------------------------------- inputs


def load_tables():
    """ABSCO band slices, ILS and wavelength grids (as oco_fp_abs_all_bands)."""
    with h5py.File(L1B, "r") as f:
        wvl_coef = f["InstrumentHeader/dispersion_coef_samp"][...]
        dlam = f["InstrumentHeader/ils_delta_lambda"][...]
        resp = f["InstrumentHeader/ils_relative_response"][...]
    pix = np.arange(1, 1017, dtype=float)
    with h5py.File(ABSCO_DIR / "h2o_v52.hdf", "r") as fh:
        nuh2o = fh["Wavenumber"][...]
        h2o_meta = dict(hpa=fh["Pressure"][...] / 100.0, tk=fh["Temperature"][...],
                        broad=fh["Broadener_01_VMR"][...])
        bands = []
        for ib in range(3):
            fn, key = GAS[ib]
            nu1, nu2 = 1e4 / XR[ib][1], 1e4 / XR[ib][0]
            with h5py.File(ABSCO_DIR / fn, "r") as fg:
                nu = fg["Wavenumber"][...]
                i1, i2 = find_boundary(nu1, nu2, nu)
                tab = fg[key][:, :, :, i1:i2 + 1]
                meta = dict(hpa=fg["Pressure"][...] / 100.0, tk=fg["Temperature"][...],
                            broad=fg["Broadener_01_VMR"][...])
            nudat = nu[i1:i2 + 1]
            j1, j2 = find_boundary(nu1, nu2, nuh2o)
            tab_h2o = fh["Gas_01_Absorption"][:, :, :, j1:j2 + 1]
            w0, w1 = np.searchsorted(nudat, nuh2o[[j1, j2]], side="left")
            wl = np.stack([sum(wvl_coef[ib, fp, k] * pix**k for k in range(wvl_coef.shape[2]))
                           for fp in range(8)])
            yy = resp[ib] / resp[ib].max(axis=2, keepdims=True)
            bands.append(dict(tab=tab, meta=meta, tab_h2o=tab_h2o, h2o_slot=(w0, w1),
                              nudat=nudat, wvldat=1e4 / nudat, wl=wl, xx=dlam[ib].astype(np.float64),
                              yy=yy.astype(np.float64),
                              ray_xsec=rayleigh_xsec(1e4 / nudat)))
            print(f"band {BANDS[ib]}: {len(nudat)} ABSCO points, H2O slot {w0}-{w1}", flush=True)
    return bands, h2o_meta


def build_atmosphere(sid: np.ndarray) -> dict:
    """Per-sounding layer arrays, same formulas as src/abs_util/fp_atm.py:oco_fp_atm_abs."""
    rd, eps, kb, g = 287.052874, 0.622, 1.380649e-23, 9.81
    with h5py.File(MET, "r") as f:
        msid = f["SoundingGeometry/sounding_id"][...].ravel()
        pos = {s: i for i, s in enumerate(msid)}
        idx = np.array([pos[s] for s in sid])

        def rd_(k):
            a = f[k][...]
            return a.reshape(len(msid), *a.shape[2:])[idx].astype(np.float64)
        q = rd_("Meteorology/specific_humidity_profile_met")[:, ::-1]
        t = rd_("Meteorology/temperature_profile_met")[:, ::-1]
        p = rd_("Meteorology/vector_pressure_levels_met")[:, ::-1]
        psfc = rd_("Meteorology/surface_pressure_met")
        gph = rd_("Meteorology/gph_met")
    with h5py.File(CPR, "r") as f:
        csid = f["SoundingGeometry/sounding_id"][...].ravel()
        pos = {s: i for i, s in enumerate(csid)}
        idx = np.array([pos[s] for s in sid])
        co2 = f["CO2Prior/co2_prior_profile_cpr"][...].reshape(len(csid), -1)[idx][:, ::-1].astype(np.float64)
    with h5py.File(L1B, "r") as f:
        lsid = f["SoundingGeometry/sounding_id"][...].ravel()
        pos = {s: i for i, s in enumerate(lsid)}
        idx = np.array([pos[s] for s in sid])
        v_inst = f["SoundingGeometry/sounding_relative_velocity"][...].ravel()[idx]
        v_sol = f["SoundingGeometry/sounding_solar_relative_velocity"][...].ravel()[idx]
    for a in (q, t, p, psfc, gph, co2):
        a[a == -999999] = np.nan
    p_edge = AP * 100 + BP * psfc[:, None]
    log_ratio = np.log(p_edge[:, :-1] / p_edge[:, 1:])
    r = q / (1 - q)
    e = p * r / (eps + r)
    tv = t / (1 - (r / (r + eps)) * (1 - eps))
    dz = rd * tv / g * log_ratio / 1e3                                     # km
    h_edge = np.concatenate([gph[:, None], gph[:, None] + np.cumsum(dz * 1e3, axis=1)], axis=1) / 1e3
    air = p / (kb * t) / 1e6
    dry = (p - e) / (kb * t) / 1e6
    h2o = e / (kb * t) / 1e6
    sl = slice(0, 71)                                                      # top layer dropped
    return dict(p=p[:, sl] / 100, t=t[:, sl], air=air[:, sl], o2=dry[:, sl] * O2MIX,
                co2=dry[:, sl] * co2[:, sl], h2o=h2o[:, sl], h2o_vmr=(h2o / dry)[:, sl],
                dz=dz[:, sl], h_mid=((h_edge[:, :-1] + h_edge[:, 1:]) / 2)[:, sl],
                v_inst=v_inst, v_sol=v_sol)


# --------------------------------------------------------------------------- core

_G: dict = {}


def _ext_layers(b, atm, i, gas_den, dt=0.0):
    """High-resolution gas and H2O extinction [nu, layer] (1/km)."""
    meta, hm = b["meta"], _G["h2o_meta"]
    nlay = atm["p"].shape[1]
    n = len(b["nudat"])
    e_gas = np.empty((n, nlay))
    e_h2o = np.zeros((n, nlay))
    w0, w1 = b["h2o_slot"]
    pp, tt, hv = atm["p"][i], atm["t"][i] + dt, atm["h2o_vmr"][i]
    pi_g = p_index(pp, meta["hpa"])
    pi_h = p_index(pp, hm["hpa"])
    for iz in range(nlay):
        tg = t_index(tt[iz], meta["tk"], pi_g[iz])
        th = t_index(tt[iz], hm["tk"], pi_h[iz])
        ag = absco_trilinear(b["tab"], meta["hpa"], meta["tk"], meta["broad"], tt[iz], pp[iz],
                             tg, pi_g[iz], hv[iz])
        ah = absco_trilinear(b["tab_h2o"], hm["hpa"], hm["tk"], hm["broad"], tt[iz], pp[iz],
                             th, pi_h[iz], hv[iz])
        e_gas[:, iz] = 1.0e5 * gas_den[iz] * ag
        e_h2o[w0:w1 + 1, iz] = 1.0e5 * atm["h2o"][i, iz] * ah
    return e_gas, e_h2o


def _tau_conv(tau_hr, wvldat, wl, xx, yy):
    """Transmission-space ILS convolution for all channels (production formula)."""
    start, end = conv_indices(wl, xx, wvldat)
    out = np.empty(len(wl))
    xw = xx + wl[:, None]
    for c in range(len(wl)):
        s, e = start[c], end[c]
        w = np.interp(wvldat[s:e], xw[c], yy[c])
        th = tau_hr[s:e]
        m = th.min()
        tr = np.dot(np.exp(-(th - m)), w) / w.sum()
        out[c] = m - np.log(max(tr, np.finfo(float).tiny))
    return out


def sounding_attrs(i: int) -> dict:
    atm, bands = _G["atm"], _G["bands"]
    fp = int(_G["fp"][i])
    airmass = 1 / np.cos(np.deg2rad(_G["sza"][i])) + 1 / np.cos(np.deg2rad(_G["vza"][i]))
    u, wq = np.polynomial.legendre.leggauss(N_QUAD)
    u = 0.5 * (u + 1)
    s_nodes, s_w = u**2, wq * 0.5 * 2 * u                     # ∫ ds = ∫ 2u du on [0, 1]
    dz, h = atm["dz"][i], atm["h_mid"][i]
    bl = h - h[0] < BL_KM
    out = {}
    for ib, b in enumerate(bands):
        gas_den = atm["o2"][i] if ib == 0 else atm["co2"][i]
        e_gas, e_h2o = _ext_layers(b, atm, i, gas_den)
        ray = (b["ray_xsec"][:, None] * 1e-28 * atm["air"][i] * dz * 1e5).sum(axis=1)
        a_gas = e_gas * dz * airmass                          # [nu, z] slant layer OD
        a_h2o = e_h2o * dz * airmass
        a_ray = ray * airmass
        tau_hr = a_gas.sum(axis=1) + a_h2o.sum(axis=1) + a_ray
        wl, xx, yy, wv = b["wl"][fp], b["xx"][fp], b["yy"][fp], b["wvldat"]
        start, end = conv_indices(wl, xx, wv)
        xw = xx + wl[:, None]
        nch = len(wl)
        res = {k: np.full(nch, np.nan) for k in ATTRS}
        for c in range(nch):
            s, e = start[c], end[c]
            w = np.interp(wv[s:e], xw[c], yy[c])
            th = tau_hr[s:e]
            m = th.min()
            tr = np.dot(np.exp(-(th - m)), w) / w.sum()
            tau_c = m - np.log(max(tr, np.finfo(float).tiny))
            ex = np.exp(-np.outer(s_nodes, th - m)) * w       # [q, nu]
            p_s = ex / ex.sum(axis=1, keepdims=True)
            p_bar = s_w @ p_s                                 # integrated weights
            c_gas = p_bar @ a_gas[s:e]
            c_h2o = p_bar @ a_h2o[s:e]
            c_ray = p_bar @ a_ray[s:e]
            c_abs = c_gas + c_h2o
            tot_abs = c_abs.sum()
            j1 = p_s[-1] @ (a_gas[s:e] + a_h2o[s:e])          # s = 1 marginal weights
            res["tau_rep"][c] = tau_c
            res["closure"][c] = (tot_abs + c_ray) / tau_c - 1
            res["eta"][c] = c_abs @ h / tot_abs
            res["eta_jac"][c] = j1 @ h / j1.sum()
            res["f_h2o"][c] = c_h2o.sum() / tau_c
            res["f_ray"][c] = c_ray / tau_c
            res["f_bl"][c] = c_abs[bl].sum() / tot_abs
        dwl = np.gradient(wl)
        tp = _tau_conv(tau_hr, wv, wl + SHIFT_PIX * dwl, xx, yy)
        tm = _tau_conv(tau_hr, wv, wl - SHIFT_PIX * dwl, xx, yy)
        res["dtau_dpix"] = (tp - tm) / (2 * SHIFT_PIX)
        tp = _tau_conv(tau_hr, wv, wl, xx * (1 + WIDTH_EPS), yy)
        tm = _tau_conv(tau_hr, wv, wl, xx * (1 - WIDTH_EPS), yy)
        res["dtau_dlnw"] = (tp - tm) / (2 * WIDTH_EPS)
        tt = []
        for sign in (1.0, -1.0):
            eg, eh = _ext_layers(b, atm, i, gas_den, dt=sign * DT_K)
            tt.append(_tau_conv(((eg + eh) * dz).sum(axis=1) * airmass + a_ray, wv, wl, xx, yy))
        res["dtau_dT"] = (tt[0] - tt[1]) / (2 * DT_K)
        out[BANDS[ib]] = res
    return out


def _work(idx_list):
    return [(i, sounding_attrs(i)) for i in idx_list]


def select_soundings(subset: str, n_total: int, fp: np.ndarray) -> np.ndarray:
    if subset == "all":
        return np.arange(n_total)
    n = int(subset)
    return np.unique(np.linspace(0, n_total - 1, n).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all", help="'all' or a number of evenly spaced soundings")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / f"resid_attrs_{ORBIT}.npz"))
    args = ap.parse_args()

    with h5py.File(FP_TAU, "r") as f:
        sid = np.round(f["sounding_id"][...]).astype(np.int64)
        fp = f["fp_number"][...].astype(int)
        sza, vza = f["sza"][...], f["vza"][...]
        tau_ref = np.stack([f[f"{b}_tau_output"][...] for b in BANDS])
    sel = select_soundings(args.subset, len(sid), fp)
    t0 = time.time()
    bands, h2o_meta = load_tables()
    print(f"tables loaded in {time.time() - t0:.0f} s", flush=True)
    atm = build_atmosphere(sid[sel])
    # worker indices refer to positions in `sel`
    _G.update(bands=bands, h2o_meta=h2o_meta, atm=atm, fp=fp[sel], sza=sza[sel], vza=vza[sel])

    t0 = time.time()
    chunks = np.array_split(np.arange(len(sel)), max(1, len(sel) // 4))
    res = []
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for k, part in enumerate(pool.imap(_work, chunks)):
                res.extend(part)
                if k % 100 == 0:
                    print(f"  {len(res)}/{len(sel)} soundings, {time.time() - t0:.0f} s", flush=True)
    else:
        for c in chunks:
            res.extend(_work(c))
    print(f"attributes for {len(sel)} soundings in {time.time() - t0:.0f} s", flush=True)
    res.sort(key=lambda x: x[0])
    arrays = {f"{b}_{k}": np.stack([r[1][b][k] for r in res]).astype(np.float32)
              for b in BANDS for k in ATTRS}

    # validation against the stored production τ
    lines = []
    for ib, b in enumerate(BANDS):
        ref = tau_ref[ib, sel]
        rel = np.abs(arrays[f"{b}_tau_rep"] - ref) / ref
        clo = np.abs(arrays[f"{b}_closure"])
        lines.append(f"{b}: |Δτ|/τ median {np.nanmedian(rel):.2e}, p99 {np.nanpercentile(rel, 99):.2e}, "
                     f"max {np.nanmax(rel):.2e}; layer-sum closure median {np.nanmedian(clo):.1e}, "
                     f"max {np.nanmax(clo):.1e}")
    print("\n".join(lines))
    np.savez_compressed(args.out, j=sel, sounding_id=sid[sel], fp=fp[sel],
                        v_inst=atm["v_inst"], v_sol=atm["v_sol"],
                        h2o_col=(atm["h2o"] * atm["dz"]).sum(axis=1), validation="\n".join(lines),
                        **arrays)
    print("saved", args.out)


if __name__ == "__main__":
    main()
