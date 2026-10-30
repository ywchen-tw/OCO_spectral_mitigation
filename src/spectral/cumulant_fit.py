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
from scipy.optimize import least_squares, lsq_linear
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


# Extra per-sounding fit statistics (fitting.py --extra-fits): unconstrained
# order-2 and order-3 coefficients on the raw ln_T, their residual-based
# standard errors, residual rms per order, and the channel count.  Produced by
# _extra_fit_chunk and stored as <band>_<field>_nosg in fitting_details.
#
# Appended (2026-10-30), only computed when the tau file carries the per-channel
# H2O attribution <band>_frac_h2o (telluric-Doppler variant); NaN otherwise:
#   _t2 : two-tau model, no continuum, unconstrained lstsq
#         ln T = c − k1_dry·τ_dry − k1_w·τ_h2o + ½·k2·τ²
#   _am : the agreed per-band model (see _fit_am_o2a / _fit_am_wco2 / _fit_am_sco2)
# with τ_h2o = frac_h2o·τ (frac NaN → 0), τ_dry = τ − τ_h2o.
EXTRA_FIT_FIELDS = (
    "k1_o2", "k2_o2", "sig_k1_o2", "sig_k2_o2", "rms_o2",
    "k1_o3", "k2_o3", "k3_o3", "sig_k1_o3", "sig_k2_o3", "rms_o3",
    "rms_prod", "nchan",
    # two-tau order-2 (T2)
    "k1_dry_t2", "k1w_t2", "k2_t2", "rms_t2", "sig_k1_dry_t2", "sig_k1w_t2", "sig_k2_t2",
    # agreed model (AM)
    "k1_am", "k1w_am", "k2_am", "a1_am", "a2_am", "f_am", "c_am", "rms_am", "k2diag_am", "ok_am",
    "sig_k1_am", "sig_k1w_am", "sig_k2_am", "sig_a1_am", "sig_a2_am", "sig_f_am", "sig_c_am",
    "sig_k2diag_am",
)
_EXTRA_INDEX = {name: i for i, name in enumerate(EXTRA_FIT_FIELDS)}
_T2_FIELDS = tuple(n for n in EXTRA_FIT_FIELDS if n.endswith("_t2"))
_AM_FIELDS = tuple(n for n in EXTRA_FIT_FIELDS if n.endswith("_am"))

# Agreed-model spectral coordinate x = (j − 507.5)/507.5 on the ORIGINAL
# 0..1015 channel index j.  The chunk slabs passed to _extra_fit_chunk are
# already sliced [1:-1], so slab position p is channel j = p + 1.
_N_CHAN_FULL = 1016
_SLAB_CHANNEL_OFFSET = 1
_X_HALF = (_N_CHAN_FULL - 1) / 2.0     # 507.5
_AM_MIN_CHAN = 8
_AM_F_MAX = 0.2


def _unconstrained_fit_stats(A, y):
    """lstsq coefficients [intercept, k1, ...], their residual-based standard
    errors (iid-residual assumption; photon noise is not modelled here) and the
    residual rms."""
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    dof = max(len(y) - A.shape[1], 1)
    cov = np.linalg.pinv(A.T @ A) * float(resid @ resid) / dof
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    rms = float(np.sqrt(np.mean(resid**2)))
    return coef, sig, rms


# ─── Two-tau and agreed-model fits (extra statistics) ────────────────────────
# Formal 1-sigma errors are s²·(JᵀJ)⁻¹ with s² = RSS/(n − p), J the design
# matrix (linear fits) or least_squares(...).jac (O2A).  The scale is the
# fitted residual, not the L1B noise model: the residual is misfit-dominated
# (reduced χ² ≫ 1 against the photon noise), so these are the formal errors for
# independent residuals and UNDERSTATE scene-correlated error.

def _cov_se(jac, resid):
    """1-sigma parameter errors s·sqrt(diag((JᵀJ)⁻¹)), s² = RSS/(n − p)."""
    n, p = jac.shape
    s2 = float(resid @ resid) / max(n - p, 1)
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)


def _lin_se(cols, y):
    """Unconstrained lstsq on the column design; (coef, rms, formal 1-sigma errors)."""
    a = np.column_stack(cols)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ c
    return c, float(np.sqrt(np.mean(r**2))), _cov_se(a, r)


def _fit_t2(td, tw, y):
    """Two-tau order-2 model, no continuum: ln T = c − k1_dry·τ_dry − k1_w·τ_h2o + ½k2·τ²."""
    t = td + tw
    c, rms, se = _lin_se([np.ones_like(td), -td, -tw, 0.5 * t**2], y)
    return {"k1_dry_t2": c[1], "k1w_t2": c[2], "k2_t2": c[3], "rms_t2": rms,
            "sig_k1_dry_t2": se[1], "sig_k1w_t2": se[2], "sig_k2_t2": se[3]}


def _am_nan():
    return {name: np.nan for name in _AM_FIELDS}


def _fit_am_o2a(t, x, y):
    """O2A agreed model: ln T = c + a1·x + ln[(1 − f)·exp(−k1·τ) + f].

    Bounds k1 ≥ 0, 0 ≤ f ≤ 0.2; k2 fixed 0.  Started from the linear fit on
    [1, x, −τ] with f = 0.01 and k1 = max(k1_lin, 0.1).  On solver failure all
    values are NaN and ok_am = 0 (no fallback to the linear fit).  k2diag_am is
    the free k2 of the refit with ½k2τ² inside the exponential, started from
    the converged solution (NaN if that refit fails).
    """
    c0 = np.linalg.lstsq(np.column_stack([np.ones_like(t), x, -t]), y, rcond=None)[0]

    def res(p, free_k2):
        c, a1, k1, f = p[:4]
        k2 = p[4] if free_k2 else 0.0
        return c + a1 * x + np.log((1 - f) * np.exp(-k1 * t + 0.5 * k2 * t**2) + f) - y

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = least_squares(res, [c0[0], c0[1], max(c0[2], 0.1), 0.01], args=(False,),
                          bounds=([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, _AM_F_MAX]))
        if not r.success or not np.all(np.isfinite(r.x)):
            out = _am_nan()
            out["ok_am"] = 0.0
            return out
        se = _cov_se(r.jac, r.fun)
        out = _am_nan()
        out.update({"k1_am": r.x[2], "k2_am": 0.0, "a1_am": r.x[1], "f_am": r.x[3], "c_am": r.x[0],
                    "rms_am": float(np.sqrt(np.mean(r.fun**2))), "ok_am": 1.0,
                    "sig_k1_am": se[2], "sig_a1_am": se[1], "sig_f_am": se[3], "sig_c_am": se[0]})
        try:
            rd = least_squares(res, list(r.x) + [0.0], args=(True,),
                               bounds=([-np.inf, -np.inf, 0.0, 0.0, -np.inf],
                                       [np.inf, np.inf, np.inf, _AM_F_MAX, np.inf]))
            if rd.success:
                out["k2diag_am"] = rd.x[4]
                out["sig_k2diag_am"] = _cov_se(rd.jac, rd.fun)[4]
        except ValueError:
            pass
    return out


def _fit_am_wco2(td, tw, x, y):
    """WCO2 agreed model: lstsq on [1, x, −τ_dry, −τ_h2o]; k2 fixed 0.
    k2diag_am from the same design plus ½τ² (τ = τ_dry + τ_h2o)."""
    one = np.ones_like(td)
    c, rms, se = _lin_se([one, x, -td, -tw], y)
    cd, _, sed = _lin_se([one, x, -td, -tw, 0.5 * (td + tw) ** 2], y)
    out = _am_nan()
    out.update({"k1_am": c[2], "k1w_am": c[3], "k2_am": 0.0, "a1_am": c[1], "c_am": c[0],
                "rms_am": rms, "k2diag_am": cd[4], "ok_am": 1.0,
                "sig_k1_am": se[2], "sig_k1w_am": se[3], "sig_a1_am": se[1], "sig_c_am": se[0],
                "sig_k2diag_am": sed[4]})
    return out


def _fit_am_sco2(td, tw, x, y, land):
    """SCO2 agreed model: lstsq on [1, x, (x² if land), −τ_dry, −τ_h2o, ½τ²];
    a2_am NaN over water; k2diag_am = k2_am."""
    one = np.ones_like(td)
    t = td + tw
    cols = [one, x] + ([x**2] if land else []) + [-td, -tw, 0.5 * t**2]
    c, rms, se = _lin_se(cols, y)
    k = 3 if land else 2
    out = _am_nan()
    out.update({"k1_am": c[k], "k1w_am": c[k + 1], "k2_am": c[k + 2], "a1_am": c[1],
                "a2_am": c[2] if land else np.nan, "c_am": c[0], "rms_am": rms,
                "k2diag_am": c[k + 2], "ok_am": 1.0,
                "sig_k1_am": se[k], "sig_k1w_am": se[k + 1], "sig_k2_am": se[k + 2],
                "sig_a1_am": se[1], "sig_a2_am": se[2] if land else np.nan, "sig_c_am": se[0],
                "sig_k2diag_am": se[k + 2]})
    return out


def _extra_fit_chunk(j_indices, tau_slab, lnT_slab, fit_orders, frac_slab=None, land_chunk=None):
    """Extra statistics on the raw ln_T for a chunk of soundings (worker-safe).

    Independent of _fit_chunk so that the production path is untouched; the
    production-order fit is repeated here only to record its residual rms.

    Parameters
    ----------
    j_indices  : [m] global sounding indices
    tau_slab   : [3, m, n_chan] band tau, edge channels already excluded ([1:-1])
    lnT_slab   : [3, m, n_chan] raw ln T, same channels
    fit_orders : (o2a, wco2, sco2) production truncation orders
    frac_slab  : [3, m, n_chan] H2O fraction of τ (<band>_frac_h2o, NaN → 0) or
                 None (production tau files) → all _t2 / _am fields NaN
    land_chunk : [m] bool land mask (SCO2 x² term), None → all water

    The _t2/_am fits use the same channel mask and τ-sort as the order-2/3
    fits (finite ln T and τ) and need ≥ 8 channels.  The agreed model's x is
    (j − 507.5)/507.5 with j = slab position + 1 (original channel index),
    permuted with the same sort.  Bands: 0 = O2A, 1 = WCO2, 2 = SCO2.  The
    sig_* values are formal errors (see _cov_se: residual-scaled, understate
    scene-correlated error).

    Returns (j_indices, extras [3, m, len(EXTRA_FIT_FIELDS)]), NaN on failure.
    """
    m = len(j_indices)
    extras = np.full((3, m, len(EXTRA_FIT_FIELDS)), np.nan)
    for jl in range(m):
        for i_band, band_order in enumerate(fit_orders):
            tau_j = np.asarray(tau_slab[i_band, jl], dtype=np.float64)
            ln_j  = np.asarray(lnT_slab[i_band, jl], dtype=np.float64)
            mask = ~np.isnan(ln_j) & ~np.isnan(tau_j)
            n = int(mask.sum())
            row = extras[i_band, jl]
            row[_EXTRA_INDEX["nchan"]] = n
            if n < 5:
                continue
            tau_m = tau_j[mask]
            sort_idx = np.argsort(tau_m)
            tau_s = tau_m[sort_idx]
            y_raw = ln_j[mask][sort_idx]
            try:
                if n >= band_order + 2:
                    A_prod = get_design_matrix(tau_s, band_order)
                    popt = _solve_cumulant(A_prod, y_raw, band_order)
                    coef = np.concatenate([popt[-1:], popt[:-1]])
                    row[_EXTRA_INDEX["rms_prod"]] = float(np.sqrt(np.mean((y_raw - A_prod @ coef)**2)))
                for order, tag in ((2, "o2"), (3, "o3")):
                    coef, sig, rms = _unconstrained_fit_stats(get_design_matrix(tau_s, order), y_raw)
                    row[_EXTRA_INDEX[f"k1_{tag}"]] = coef[1]
                    row[_EXTRA_INDEX[f"k2_{tag}"]] = coef[2]
                    if order == 3:
                        row[_EXTRA_INDEX["k3_o3"]] = coef[3]
                    row[_EXTRA_INDEX[f"sig_k1_{tag}"]] = sig[1]
                    row[_EXTRA_INDEX[f"sig_k2_{tag}"]] = sig[2]
                    row[_EXTRA_INDEX[f"rms_{tag}"]] = rms
            except (np.linalg.LinAlgError, ValueError):
                pass

            # ── two-tau (T2) and agreed-model (AM) fits ──
            if frac_slab is None or n < _AM_MIN_CHAN:
                continue
            frac = np.asarray(frac_slab[i_band, jl], dtype=np.float64)[mask][sort_idx]
            frac = np.where(np.isfinite(frac), frac, 0.0)
            tw = frac * tau_s
            td = tau_s - tw
            pos = np.flatnonzero(mask)[sort_idx] + _SLAB_CHANNEL_OFFSET
            x = (pos - _X_HALF) / _X_HALF
            try:
                for name, val in _fit_t2(td, tw, y_raw).items():
                    row[_EXTRA_INDEX[name]] = val
            except (np.linalg.LinAlgError, ValueError):
                pass
            land = bool(land_chunk[jl]) if land_chunk is not None else False
            try:
                if i_band == 0:
                    am = _fit_am_o2a(tau_s, x, y_raw)
                elif i_band == 1:
                    am = _fit_am_wco2(td, tw, x, y_raw)
                else:
                    am = _fit_am_sco2(td, tw, x, y_raw, land)
            except (np.linalg.LinAlgError, ValueError):
                am = _am_nan()
                am["ok_am"] = 0.0
            for name, val in am.items():
                row[_EXTRA_INDEX[name]] = val
    return j_indices, extras


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
