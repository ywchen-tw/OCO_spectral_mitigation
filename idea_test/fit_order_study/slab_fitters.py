"""Fitters compared in the slab closure test (thin wrappers over ``variants.py``).

Every fitter returns (k1, k2, intercept).  The cumulant fits reuse
``variants.fit_cumulant`` (k1, k2 ≥ 0 via BVLS, identical to production);
``ols_2u`` drops the bound; the gamma fits reuse ``variants.fit_gamma``.
``*_eta`` add a channel vertical-shape regressor −β·τ·η (test T5).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from variants import design_matrix, fit_cumulant, fit_gamma

FITTERS = ("ols_7", "ols_3", "ols_2", "ols_2u", "gamma_u", "gamma_w")
SHAPE_FITTERS = ("ols_2_eta", "gamma_u_eta")


def _ols_unbounded(tau, ln_t, order):
    coef = np.linalg.lstsq(design_matrix(tau, order), ln_t, rcond=None)[0]
    return coef[1], coef[2], coef[0]


def _ols_2_eta(tau, ln_t, eta):
    a = np.column_stack([design_matrix(tau, 2), -tau * eta])
    coef = np.linalg.lstsq(a, ln_t, rcond=None)[0]
    return coef[1], coef[2], coef[0]


def _gamma_eta(tau, ln_t, eta, init):
    k1_0 = max(init[0], 0.05)
    kap_0 = float(np.clip(init[0] ** 2 / init[1], 0.5, 1e3)) if init[1] > 0 else 50.0

    def model(p):
        k1, kap, c, beta = p
        x = np.maximum(1.0 + (k1 + beta * eta) * tau / kap, 1e-12)
        return c - kap * np.log(x)

    sol = least_squares(lambda p: model(p) - ln_t, x0=[k1_0, kap_0, init[2], 0.0],
                        bounds=([0.0, 0.1, -np.inf, -np.inf], [np.inf, 1e4, np.inf, np.inf]),
                        method="trf", x_scale="jac", max_nfev=300)
    k1, kap, c, _ = sol.x
    return k1, k1**2 / kap, c


def fit_one(tag: str, tau, ln_t, sig=None, eta=None, init=None):
    """(k1, k2, intercept) of one spectrum; ``init`` = ols_2 result for the gamma fits."""
    tau = np.asarray(tau, float)
    ln_t = np.asarray(ln_t, float)
    unit = np.ones_like(tau)
    if tag.startswith("ols_") and tag[4:].isdigit():
        r = fit_cumulant(tau, ln_t, unit, int(tag[4:]), weighted=False)
        return r["k1"], r["k2"], r["intercept"]
    if tag == "ols_2u":
        return _ols_unbounded(tau, ln_t, 2)
    if tag in ("gamma_u", "gamma_w"):
        init_d = None if init is None else {"k1": init[0], "k2": init[1], "intercept": init[2]}
        weighted = tag == "gamma_w"
        s = sig if weighted else unit
        r = fit_gamma(tau, ln_t, s, init=init_d, weighted=weighted)
        return r["k1"], r["k2"], r["intercept"]
    if tag == "ols_2_eta":
        return _ols_2_eta(tau, ln_t, eta)
    if tag == "gamma_u_eta":
        return _gamma_eta(tau, ln_t, eta, init)
    raise ValueError(tag)


def fit_all(tau, ln_t, sig=None, eta=None, tags=FITTERS) -> dict[str, tuple[float, float, float]]:
    out = {}
    init = fit_one("ols_2u", tau, ln_t)
    for tag in tags:
        out[tag] = fit_one(tag, tau, ln_t, sig=sig, eta=eta, init=init)
    return out
