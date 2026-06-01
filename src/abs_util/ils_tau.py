import numpy as np


TAU_CONVOLUTION_VERSION = "transmission_ils_rayleigh_hr_v1"


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
