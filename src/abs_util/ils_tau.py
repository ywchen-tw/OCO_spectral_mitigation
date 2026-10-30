import numpy as np


TAU_CONVOLUTION_VERSION = "transmission_ils_rayleigh_hr_v2"
# Test variant with the telluric Doppler shift of the ABSCO grid applied
# (fp_abs_coeff.py, fitting.py --telluric-doppler).  A separate version string
# keeps the two tau caches from ever being mixed.  v4 (2026-10-30): the TOA
# solar grid is shifted by the spacecraft velocity too, beta = (v_solar -
# v_inst)/c (fp_abs_coeff.solar_doppler_beta); tau and the attribution
# datasets are unchanged from v3.
TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER = "transmission_ils_rayleigh_hr_v4_telluric_solar_doppler"
# Previous variant stamp (solar grid shifted by v_solar only).  A v3 file can be
# upgraded in place by recomputing only the <band>_toa_sol_output datasets
# (abs_util.toa_solar.recompute_toa_sol, compute_tau_only.py --toa-only).
TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER_V3 = "transmission_ils_rayleigh_hr_v3_telluric_doppler"


def tau_convolution_version(telluric_doppler=False):
    """Version stamp written to / expected in a tau file for the given variant."""
    if telluric_doppler:
        return TAU_CONVOLUTION_VERSION_TELLURIC_DOPPLER
    return TAU_CONVOLUTION_VERSION


def convolve_optical_depth_in_transmission(
    ext_window,
    dz,
    ils_weights,
    airmass,
    rayleigh_tau_window=None,
):
    """Return ILS-effective optical depth from high-resolution extinction.

    The instrument line shape acts on radiance/transmission, not directly on
    optical depth. This distinction matters for saturated absorption lines:
    mean(tau) is not equivalent to -log(mean(exp(-tau))).
    """
    ils_sum = np.sum(ils_weights)
    if not np.isfinite(ils_sum) or ils_sum <= 0:
        raise ValueError("ILS weights must have a positive finite sum.")

    tau_vertical = ext_window @ dz
    if rayleigh_tau_window is not None:
        tau_vertical = tau_vertical + rayleigh_tau_window

    tau_hr = tau_vertical * airmass
    trans_eff = np.dot(np.exp(-tau_hr), ils_weights) / ils_sum
    trans_eff = np.clip(trans_eff, np.finfo(float).tiny, 1.0)
    return -np.log(trans_eff)


def attribute_optical_depth(
    ext_window,
    ext_h2o_window,
    dz,
    ils_weights,
    airmass,
    rayleigh_tau_window,
    p_lay,
):
    """First-order attribution of one channel's ILS-effective tau to its sources.

    The transmission-space convolution is nonlinear, so component optical
    depths convolved separately do not add up to the total.  Instead the
    high-resolution contributions are weighted by the same factor that
    determines the effective tau, w = ILS * exp(-tau_hr), and the channel tau is
    attributed proportionally:

        frac_h2o  = sum(w * tau_h2o_hr) / sum(w * tau_hr)
        frac_ray  = sum(w * tau_ray_hr) / sum(w * tau_hr)
        p_abs     = pressure centroid of the gas absorption (hPa), weighted by
                    w and by each layer's share of the gas optical depth

    frac_h2o + frac_ray + frac_dry = 1 by construction.  ``ext_window`` is the
    total gas extinction (dry species + H2O); ``ext_h2o_window`` the H2O part.

    Returns
    -------
    (frac_h2o, frac_ray, p_abs_hpa), NaN when the channel has no absorption.
    """
    tau_gas_layers = ext_window * dz                   # [n_hr, n_lay], vertical
    tau_gas = tau_gas_layers.sum(axis=1)
    tau_h2o = ext_h2o_window @ dz
    tau_hr = (tau_gas + rayleigh_tau_window) * airmass
    w = ils_weights * np.exp(-tau_hr)
    denom = float(np.dot(w, tau_hr))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan, np.nan, np.nan
    frac_h2o = airmass * float(np.dot(w, tau_h2o)) / denom
    frac_ray = airmass * float(np.dot(w, rayleigh_tau_window)) / denom
    gas_weight = float(np.dot(w, tau_gas))
    if gas_weight > 0.0:
        p_abs = float(np.dot(w, tau_gas_layers @ p_lay)) / gas_weight
    else:
        p_abs = np.nan
    return frac_h2o, frac_ray, p_abs
