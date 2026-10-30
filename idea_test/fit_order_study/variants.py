"""Fit variants for the truncation-order study (pure numpy/scipy, per sounding).

The cumulant model  ln T = c − k1·τ + ½k2·τ² − ⅓k3·τ³ + …  is linear in
[c, k1, …, kN]; the design matrix and the k1, k2 ≥ 0 bound follow the
production solver in ``src/spectral/cumulant_fit.py`` so that ``ols_7`` here
reproduces the production ``_nosg`` coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, lsq_linear

MAX_ORDER = 7
K2_BOUND_EPS = 1e-6
TAU_CUT = 3.0
FIELDS = ("k1", "k2", "intercept", "sig_k1", "sig_k2", "sig_k1_rss", "chi2_red",
          "resid_rms", "lag1", "bound", "nchan")


def design_matrix(tau: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(tau)]
    for i in range(1, order + 1):
        cols.append(((-1.0) ** i / i) * tau**i)
    return np.column_stack(cols)


def _solve_bounded(a: np.ndarray, y: np.ndarray, order: int, w: np.ndarray | None):
    """Return coef [c, k1..kN] with k1, k2 ≥ 0, plus a bound-active flag."""
    aw = a if w is None else a * w[:, None]
    yw = y if w is None else y * w
    coef = np.linalg.lstsq(aw, yw, rcond=None)[0]
    n_pos = min(2, order)
    bound = bool(np.any(coef[1:1 + n_pos] < 0.0))
    if bound:
        lb = np.full(order + 1, -np.inf)
        lb[1:1 + n_pos] = 0.0
        coef = lsq_linear(aw, yw, bounds=(lb, np.full(order + 1, np.inf)),
                          method="bvls").x
    return coef, bound


def _stats(a, y, sig, coef, chan_order, weighted: bool) -> dict:
    """Noise covariance (sandwich for OLS, inverse Fisher for WLS) and residual diagnostics."""
    n, p = a.shape
    resid = y - a @ coef
    var = sig**2
    if weighted:
        ainv = np.linalg.pinv(a.T @ (a / var[:, None]))
        cov = ainv
    else:
        ata_inv = np.linalg.pinv(a.T @ a)
        cov = ata_inv @ (a.T @ (a * var[:, None])) @ ata_inv
    rss = float(resid @ resid)
    sig_rss = np.sqrt(np.diag(np.linalg.pinv(a.T @ a)) * rss / max(n - p, 1))
    r_chan = resid[np.argsort(chan_order)]           # back to wavelength order
    r0, r1 = r_chan[:-1], r_chan[1:]
    denom = np.sqrt((r0 @ r0) * (r1 @ r1))
    lag1 = float(r0 @ r1 / denom) if denom > 0 else np.nan
    out = {
        "sig_k1": float(np.sqrt(cov[1, 1])),
        "sig_k2": float(np.sqrt(cov[2, 2])) if p > 2 else np.nan,
        "sig_k1_rss": float(sig_rss[1]),
        "chi2_red": float(np.sum(resid**2 / var) / max(n - p, 1)),
        "resid_rms": float(np.sqrt(rss / n)),
        "lag1": lag1,
        "nchan": n,
    }
    return out


def fit_cumulant(tau, ln_t, sig, order: int, weighted: bool, tau_cut: float | None = None) -> dict:
    """One band, one sounding. Returns a dict with FIELDS, NaN-filled on failure."""
    mask = np.isfinite(ln_t) & np.isfinite(tau) & np.isfinite(sig) & (sig > 0)
    if tau_cut is not None:
        mask &= tau <= tau_cut
    chan = np.flatnonzero(mask)
    if len(chan) < order + 2:
        return _nan_result()
    t, y, s = tau[chan].astype(np.float64), ln_t[chan].astype(np.float64), sig[chan].astype(np.float64)
    a = design_matrix(t, order)
    try:
        coef, bound = _solve_bounded(a, y, order, 1.0 / s if weighted else None)
    except (np.linalg.LinAlgError, ValueError):
        return _nan_result()
    out = _stats(a, y, s, coef, chan, weighted)
    out.update(k1=float(coef[1]), k2=float(coef[2]) if order >= 2 else np.nan,
               intercept=float(coef[0]), bound=bound)
    return out


def fit_gamma(tau, ln_t, sig, init: dict | None = None, weighted: bool = True) -> dict:
    """Two-parameter gamma path distribution: ln T = c − κ·ln(1 + k1·τ/κ).

    ``weighted=False`` fits with unit weights (as the production cumulant fit
    does) so that its k1 is directly comparable to ``ols_N``; σ(k) is then the
    noise sandwich on the unit-weight estimator.
    """
    mask = np.isfinite(ln_t) & np.isfinite(tau) & np.isfinite(sig) & (sig > 0)
    chan = np.flatnonzero(mask)
    if len(chan) < 5:
        return _nan_result()
    t, y, s = tau[chan].astype(np.float64), ln_t[chan].astype(np.float64), sig[chan].astype(np.float64)
    k1_0 = max(init.get("k1", 1.0), 0.05) if init else 1.0
    k2_0 = init.get("k2", np.nan) if init else np.nan
    kap_0 = float(np.clip(k1_0**2 / k2_0, 0.5, 1e3)) if np.isfinite(k2_0) and k2_0 > 0 else 50.0
    c_0 = init.get("intercept", 0.0) if init else 0.0

    def model(p):
        k1, kap, c = p
        return c - kap * np.log1p(k1 * t / kap)

    w = 1.0 / s if weighted else np.ones_like(s)

    def resid(p):
        return (y - model(p)) * w

    try:
        sol = least_squares(resid, x0=[k1_0, kap_0, c_0],
                            bounds=([0.0, 0.1, -np.inf], [np.inf, 1e4, np.inf]),
                            method="trf", x_scale="jac", max_nfev=200)
    except ValueError:
        return _nan_result()
    k1, kap, c = sol.x
    k2 = k1**2 / kap
    n, p = len(t), 3
    jac = sol.jac                              # rows already multiplied by w
    if weighted:
        cov = np.linalg.pinv(jac.T @ jac)
    else:                                       # sandwich: unit-weight estimator, noise s
        jtj_inv = np.linalg.pinv(jac.T @ jac)
        cov = jtj_inv @ (jac.T @ (jac * (s**2)[:, None])) @ jtj_inv
    # k2 = k1²/κ: first-order propagation
    g = np.array([2 * k1 / kap, -(k1**2) / kap**2])
    sig_k2 = float(np.sqrt(g @ cov[:2, :2] @ g))
    r = y - model(sol.x)
    r_chan = r[np.argsort(chan)]
    denom = np.sqrt((r_chan[:-1] @ r_chan[:-1]) * (r_chan[1:] @ r_chan[1:]))
    return {
        "k1": float(k1), "k2": float(k2), "intercept": float(c),
        "sig_k1": float(np.sqrt(cov[0, 0])), "sig_k2": sig_k2,
        "sig_k1_rss": np.nan,
        "chi2_red": float(np.sum((r / s)**2) / max(n - p, 1)),
        "resid_rms": float(np.sqrt(r @ r / n)),
        "lag1": float(r_chan[:-1] @ r_chan[1:] / denom) if denom > 0 else np.nan,
        "bound": bool(kap >= 1e4 - 1e-6 or k1 <= 0),
        "nchan": n,
    }


def _nan_result() -> dict:
    out = {k: np.nan for k in FIELDS}
    out["bound"] = False
    out["nchan"] = 0
    return out


def variant_list(orders=range(1, MAX_ORDER + 1), with_gamma=True):
    """(tag, kind, order) triples; kind in {ols, wls, cut, gamma}."""
    out = []
    for n in orders:
        out.append((f"ols_{n}", "ols", n))
    for n in orders:
        out.append((f"wls_{n}", "wls", n))
    for n in orders:
        if n >= 2:
            out.append((f"cut_{n}", "cut", n))
    if with_gamma:
        out.append(("gamma", "gamma", 0))
        out.append(("gamma_u", "gamma_u", 0))
    return out


def fit_variant(kind: str, order: int, tau, ln_t, sig, init=None) -> dict:
    if kind == "ols":
        return fit_cumulant(tau, ln_t, sig, order, weighted=False)
    if kind == "wls":
        return fit_cumulant(tau, ln_t, sig, order, weighted=True)
    if kind == "cut":
        return fit_cumulant(tau, ln_t, sig, order, weighted=False, tau_cut=TAU_CUT)
    if kind == "gamma":
        return fit_gamma(tau, ln_t, sig, init=init)
    if kind == "gamma_u":
        return fit_gamma(tau, ln_t, sig, init=init, weighted=False)
    raise ValueError(kind)
