"""Cumulant-expansion fit core (ln T = −k1·τ + ½k2·τ² − … + intercept).

Model functions per truncation order, the transmittance helper, the exact
linear least-squares solver (SVD lstsq with a BVLS fallback for the
k1/k2 ≥ 0 bounds), and the chunked per-sounding fit worker used by
process_orbit (optionally across processes).

Split out of fitting.py (2026-07, review §7.4).  fitting.py re-exports all
public names, so `from spectral.fitting import fit_spectral_model` keeps
working.
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter


# ─── Cumulant expansion models ────────────────────────────────────────────────
# ln(T) = -k1*τ + ½k2*τ² - ⅓k3*τ³ + ... + intercept
# Each function corresponds to a truncation order.

def log_transmittance_model_1(tau, k1, intercept):
    return -k1 * tau + intercept

def log_transmittance_model_2(tau, k1, k2, intercept):
    return -k1 * tau + 0.5 * k2 * tau**2 + intercept

def log_transmittance_model_3(tau, k1, k2, k3, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + intercept)

def log_transmittance_model_4(tau, k1, k2, k3, k4, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            + intercept)

def log_transmittance_model_5(tau, k1, k2, k3, k4, k5, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + intercept)

def log_transmittance_model_7(tau, k1, k2, k3, k4, k5, k6, k7, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + intercept)
    
def log_transmittance_model_9(tau, k1, k2, k3, k4, k5, k6, k7, k8, k9, intercept):
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + (1/8) * k8 * tau**8
            - (1/9) * k9 * tau**9
            + intercept)

def transmittance_model(tau, l_mean, kappa, intercept):
    """Gamma-distribution transmittance model: γ·(1 + SOD·⟨l'⟩/κ)^(−κ).

    l_mean   = ⟨l'⟩  (mean path-length enhancement, first cumulant k1)
    kappa    = κ      (gamma shape parameter = ⟨l'⟩²/var(l') = k1²/k2)
    intercept = γ     (surface reflectance)
    """
    return (1 + tau * l_mean / kappa) ** (-kappa) * intercept


# Maps fit_order integer → model function.  Add new orders here as needed.
LOG_TRANSMITTANCE_MODELS = {
    1: log_transmittance_model_1,
    2: log_transmittance_model_2,
    3: log_transmittance_model_3,
    4: log_transmittance_model_4,
    5: log_transmittance_model_5,
    7: log_transmittance_model_7,
    9: log_transmittance_model_9,
}


# ─── Pure computation helpers ─────────────────────────────────────────────────

def compute_transmittance(radiances, toa_sol):
    """T = radiance / TOA_solar; mask unphysical values T > 1.

    Parameters
    ----------
    radiances : array [3, N, 1016]
    toa_sol   : array [3, N, 1016]

    Returns
    -------
    T : array [3, N, 1016], NaN where T > 1 or division is undefined
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        T = radiances / toa_sol * np.pi  # Scale by π to convert from radiance to irradiance ratio
    T[T > 1] = np.nan
    return T


def get_design_matrix(tau, order):
    """
    Creates a design matrix for the equation:
    Intercept + (-1/1)*k1*tau + (1/2)*k2*tau^2 - (1/3)*k3*tau^3 ...
    """
    # 1. Create a column of ones for the intercept
    X = [np.ones(tau.shape)]
    
    # 2. Generate each k_i column dynamically
    for i in range(1, order + 1):
        sign = (-1)**i
        multiplier = 1.0 / i
        column = sign * multiplier * (tau**i)
        X.append(column)
        
    # Stack them horizontally into a single matrix
    return np.column_stack(X)

MAX_KAPPAS = 5     # store k1..k5 in fitting_details; higher kappas are not saved
SAVGOL_WINDOW = 51
SAVGOL_ORDER = 3


def _solve_cumulant(A, y, fit_order):
    """Solve the cumulant expansion exactly (linear least squares).

    The model ln T = intercept − k1·τ + ½k2·τ² − … is linear in its
    parameters, so the historical per-sounding ``curve_fit`` call was an
    iterative solver for a (bounded) linear problem.  Solve unconstrained via
    SVD ``lstsq`` first; only when the k1/k2 ≥ 0 bounds are violated fall back
    to the exact bounded solver (BVLS).  Where the bounds are inactive both
    paths coincide with the ``curve_fit`` optimum (to solver tolerance).

    Parameters
    ----------
    A : [n_chan, fit_order+1] design matrix from get_design_matrix(tau, order)
        (coefficient vector ordered [intercept, k1, ..., k_order])
    y : [n_chan] log-transmittance values (smoothed or raw)

    Returns
    -------
    popt : 1-D array in the historical curve_fit order [k1..k_order, intercept]
    """
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    n_pos = min(2, fit_order)  # k1 (and k2 if order>=2) must be >= 0
    if np.any(coef[1:1 + n_pos] < 0.0):
        n_params = fit_order + 1
        lb = np.full(n_params, -np.inf)
        lb[1:1 + n_pos] = 0.0
        coef = lsq_linear(A, y, bounds=(lb, np.full(n_params, np.inf)),
                          method='bvls').x
    return np.concatenate([coef[1:], coef[:1]])


def fit_spectral_model(tau, ln_T, fit_order, smooth=True):
    """Fit a cumulant expansion to log-transmittance vs optical depth.

    With smooth=True (production default) a Savitzky-Golay smooth is applied
    to ln_T (sorted by tau) before fitting, which suppresses high-frequency
    noise.  With smooth=False the raw ln_T is fitted directly — used to
    produce the parallel "_nosg" parameter set that quantifies how much the
    pre-smoothing biases k1/k2 (review item M9a) without a separate rerun.

    The model is linear in [k1..k_order, intercept], so the fit is an exact
    linear least-squares solve (see _solve_cumulant) rather than iterative
    curve_fit — same optimum, ~10× faster.

    Parameters
    ----------
    tau      : 1-D array, optical depth per spectral channel
    ln_T     : 1-D array, log(transmittance) corresponding to tau
    fit_order: int  cumulant truncation order; must be a key of LOG_TRANSMITTANCE_MODELS
    smooth   : bool  apply Savitzky-Golay (51, 3) smoothing before the fit

    Returns
    -------
    popt : 1-D array [k1, k2, ..., k_order, intercept]
    """
    tau  = np.asarray(tau, dtype=np.float64)
    ln_T = np.asarray(ln_T, dtype=np.float64)
    sort_idx   = np.argsort(tau)
    tau_sorted = tau[sort_idx]
    if smooth:
        y = savgol_filter(ln_T[sort_idx],
                          window_length=SAVGOL_WINDOW, polyorder=SAVGOL_ORDER)
    else:
        y = ln_T[sort_idx]
    A = get_design_matrix(tau_sorted, fit_order)
    return _solve_cumulant(A, y, fit_order)


def _fit_chunk(j_indices, tau_slab, lnT_slab, fit_orders, dual_fit):
    """Fit all bands for a contiguous chunk of soundings (worker-safe).

    Parameters
    ----------
    j_indices : [m] global sounding indices (ascending), for plot bookkeeping
    tau_slab  : [3, m, n_chan] optical depths, edge channels already excluded
    lnT_slab  : [3, m, n_chan] log-transmittance, edge channels excluded
    fit_orders: (o2a, wco2, sco2) truncation orders
    dual_fit  : also fit the raw (no-Savitzky-Golay) ln_T

    Returns
    -------
    (j_indices, kappa [3,m,MAX_KAPPAS], intercept [3,m],
     kappa_nosg, intercept_nosg,
     plot_cands {(i_band, j_global) -> full popt})   — plot candidates are the
    first success per band in this chunk plus every j_global % 1000 == 0
    success, a superset of what the sequential plotting rule selects.
    """
    m = len(j_indices)
    kappa          = np.full((3, m, MAX_KAPPAS), np.nan)
    intercept      = np.full((3, m), np.nan)
    kappa_nosg     = np.full((3, m, MAX_KAPPAS), np.nan)
    intercept_nosg = np.full((3, m), np.nan)
    plot_cands = {}
    band_seen  = [False, False, False]

    for jl in range(m):
        for i_band, band_order in enumerate(fit_orders):
            tau_j = np.asarray(tau_slab[i_band, jl], dtype=np.float64)
            ln_j  = np.asarray(lnT_slab[i_band, jl], dtype=np.float64)
            mask = ~np.isnan(ln_j) & ~np.isnan(tau_j)
            if mask.sum() < band_order + 2:   # need more points than free params
                continue
            tau_m = tau_j[mask]
            sort_idx = np.argsort(tau_m)
            tau_s    = tau_m[sort_idx]
            y_raw    = ln_j[mask][sort_idx]
            try:
                y_sg = savgol_filter(y_raw, window_length=SAVGOL_WINDOW,
                                     polyorder=SAVGOL_ORDER)
            except ValueError:
                # fewer valid channels than the SG window — historical skip
                continue
            A = get_design_matrix(tau_s, band_order)
            try:
                popt = _solve_cumulant(A, y_sg, band_order)
            except (np.linalg.LinAlgError, ValueError):
                continue
            n_k = min(band_order, MAX_KAPPAS)
            intercept[i_band, jl]       = popt[-1]
            kappa[i_band, jl, :n_k]     = popt[:n_k]

            j_global = int(j_indices[jl])
            if not band_seen[i_band] or j_global % 1000 == 0:
                plot_cands[(i_band, j_global)] = popt
                band_seen[i_band] = True

            if dual_fit:
                # Parallel fit on the RAW (unsmoothed) ln_T; failures leave NaN.
                try:
                    popt_ns = _solve_cumulant(A, y_raw, band_order)
                    intercept_nosg[i_band, jl]       = popt_ns[-1]
                    kappa_nosg[i_band, jl, :n_k]     = popt_ns[:n_k]
                except (np.linalg.LinAlgError, ValueError):
                    pass

    return j_indices, kappa, intercept, kappa_nosg, intercept_nosg, plot_cands
