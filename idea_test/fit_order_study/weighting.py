"""Variance-function weighting for the order-2 cumulant and the gamma model.

The weighting under test (PLAN.md, "Weighting test"):

* a first-pass unit-weight fit gives residuals r(τ);
* the far-clear residuals, pooled per band (optionally per footprint index),
  define a misfit variance function  σ²(τ) = σ²_noise + a + b·τ^p;
* the second pass weights each channel by 1/σ²(τ), optionally followed by
  Huber IRLS on the standardised residual r/σ(τ);
* σ(k) comes from the covariance under the calibrated σ(τ), inflated for the
  lag-1 residual autocorrelation ρ by n/n_eff = (1 + ρ)/(1 − ρ).

Weighted least squares is invariant to a common scale of the weights, so a
variance function only matters through its *shape* (a/b, p and the noise
term); calibration of the scale only enters σ(k) and χ².

All fits here share one routine, ``fit_weighted``, so that baselines (unit,
noise) and the new variants report the same diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, nnls

from variants import _solve_bounded, design_matrix

P_GRID = np.round(np.arange(0.5, 4.01, 0.05), 3)
POOL_BINS = 40
SOUNDING_BINS = 12
N_IRLS = 2
KAPPA_MAX = 1e4

EXTRA_FIELDS = ("sig_k1_cal", "sig_k2_cal", "sig_k1_infl", "sig_k2_infl", "chi2_cal",
                "chi2_own", "lag1_std", "frac_down")


# --------------------------------------------------------------------------- variance function


@dataclass(frozen=True)
class VarFunc:
    """Parametric misfit variance a + b·τ^p; the channel noise σ² is added in ``total``."""

    a: float
    b: float
    p: float

    def misfit_var(self, tau: np.ndarray) -> np.ndarray:
        return self.a + self.b * np.power(np.clip(tau, 0.0, None), self.p)

    def total(self, tau: np.ndarray, sig: np.ndarray) -> np.ndarray:
        return sig**2 + self.misfit_var(tau)


@dataclass(frozen=True)
class TabVarFunc:
    """Non-parametric total variance: log-log interpolation of binned variances (noise included)."""

    tau_b: np.ndarray
    var_b: np.ndarray

    def total(self, tau: np.ndarray, sig: np.ndarray) -> np.ndarray:
        lt = np.log(np.clip(tau, self.tau_b[0], self.tau_b[-1]))
        return np.exp(np.interp(lt, np.log(self.tau_b), np.log(self.var_b)))


def tab_varfunc(bins: dict, key: str = "var") -> TabVarFunc:
    good = (bins[key] > 0) & (bins["tau"] > 0)
    return TabVarFunc(bins["tau"][good].copy(), bins[key][good].copy())


def binned_variance(tau, resid, noise_var, n_bins: int) -> dict:
    """Robust residual variance in equal-count τ bins.

    The variance is (1.4826·MAD)² about the bin median, so a systematic shape
    misfit common to the bin (the bin median, returned as ``med``) does not
    count as scatter.
    """
    ok = np.isfinite(tau) & np.isfinite(resid) & np.isfinite(noise_var)
    tau, resid, noise_var = tau[ok], resid[ok], noise_var[ok]
    edges = np.unique(np.quantile(tau, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, tau, side="right") - 1, 0, len(edges) - 2)
    out = {k: [] for k in ("tau", "var", "noise", "med", "msq", "count")}
    for i in range(len(edges) - 1):
        sel = idx == i
        if sel.sum() < 8:
            continue
        r = resid[sel]
        med = np.median(r)
        out["tau"].append(np.median(tau[sel]))
        out["var"].append((1.4826 * np.median(np.abs(r - med))) ** 2)
        out["noise"].append(np.mean(noise_var[sel]))
        out["med"].append(med)
        out["msq"].append(np.mean(r**2))
        out["count"].append(int(sel.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def _fit_ab(tb, vb, nb, p):
    """a, b ≥ 0 minimising Σ log²((nb + a + b·tb^p)/vb); returns (a, b, cost)."""
    scale = np.median(vb)
    basis = np.column_stack([np.ones_like(tb), np.power(tb, p)]) * scale   # model = nb + basis @ x
    # relative-error NNLS as a start, then the symmetric log-space cost
    x0, _ = nnls(basis / vb[:, None], (vb - nb) / vb)
    x0 = np.maximum(x0, 1e-8)

    def res(x):
        return np.log(np.maximum(nb + basis @ x, 1e-30) / vb)

    sol = least_squares(res, x0, bounds=(0.0, np.inf), method="trf", x_scale="jac")
    a, b = sol.x * scale
    return float(a), float(b), float(np.sum(sol.fun**2))


def fit_varfunc(bins: dict, p: float | None = None) -> tuple[VarFunc, float, np.ndarray]:
    """Fit a + b·τ^p to binned robust variances in log space; p free on P_GRID if None.

    Returns the function, its log-space rms misfit, and the cost curve over P_GRID
    (to see how well p is determined).
    """
    tb, vb, nb = bins["tau"], bins["var"], bins["noise"]
    good = (vb > 0) & (tb > 0)
    tb, vb, nb = tb[good], vb[good], nb[good]
    costs = np.array([_fit_ab(tb, vb, nb, pp)[2] for pp in P_GRID])
    p_use = float(P_GRID[np.argmin(costs)]) if p is None else float(p)
    a, b, cost = _fit_ab(tb, vb, nb, p_use)
    return VarFunc(a, b, p_use), float(np.sqrt(cost / len(tb))), costs


def sounding_varfunc(tau, resid, noise_var, p: float) -> VarFunc | None:
    """Per-sounding (adaptive) variance function from that sounding's own residuals."""
    bins = binned_variance(tau, resid, noise_var, SOUNDING_BINS)
    if len(bins["tau"]) < 4:
        return None
    good = (bins["var"] > 0) & (bins["tau"] > 0)
    if good.sum() < 4:
        return None
    a, b, _ = _fit_ab(bins["tau"][good], bins["var"][good], bins["noise"][good], p)
    if a <= 0 and b <= 0:
        return None
    return VarFunc(float(a), float(b), float(p))


# --------------------------------------------------------------------------- models


def c2_model(params, t):
    c, k1, k2 = params
    return c - k1 * t + 0.5 * k2 * t**2


def gamma_model(params, t):
    c, k1, kap = params
    return c - kap * np.log1p(k1 * t / kap)


def gamma_jac(params, t):
    """d model / d (c, k1, κ)."""
    _, k1, kap = params
    x = k1 * t / kap
    return np.column_stack([np.ones_like(t), -t / (1.0 + x), -np.log1p(x) + x / (1.0 + x)])


def _solve_c2(t, y, w_sqrt, x0=None):
    coef, bound = _solve_bounded(design_matrix(t, 2), y, 2, w_sqrt)
    return coef, bound


def _solve_gamma(t, y, w_sqrt, x0):
    def resid(p):
        return (y - gamma_model(p, t)) * w_sqrt

    def jac(p):
        return -gamma_jac(p, t) * w_sqrt[:, None]

    sol = least_squares(resid, x0=x0, jac=jac,
                        bounds=([-np.inf, 0.0, 0.1], [np.inf, np.inf, KAPPA_MAX]),
                        method="trf", x_scale="jac", max_nfev=200)
    c, k1, kap = sol.x
    return sol.x, bool(kap >= KAPPA_MAX - 1e-6 or k1 <= 0)


def gamma_x0(init: dict | None):
    """(c, k1, κ) starting point, following ``variants.fit_gamma``."""
    if init is None:
        return np.array([0.0, 1.0, 50.0])
    if "kappa" in init and np.isfinite(init.get("kappa", np.nan)):
        return np.array([init["intercept"], max(init["k1"], 0.05), init["kappa"]])
    k1_0 = max(init.get("k1", 1.0), 0.05) if np.isfinite(init.get("k1", np.nan)) else 1.0
    k2_0 = init.get("k2", np.nan)
    kap_0 = float(np.clip(k1_0**2 / k2_0, 0.5, 1e3)) if np.isfinite(k2_0) and k2_0 > 0 else 50.0
    c_0 = init.get("intercept", 0.0)
    return np.array([c_0 if np.isfinite(c_0) else 0.0, k1_0, kap_0])


# --------------------------------------------------------------------------- one fit


VarFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _lag1(x_chan_order):
    x0, x1 = x_chan_order[:-1], x_chan_order[1:]
    den = np.sqrt((x0 @ x0) * (x1 @ x1))
    return float(x0 @ x1 / den) if den > 0 else np.nan


def fit_weighted(model: str, tau, ln_t, sig, weight_var: VarFn, cal_var: VarFn,
                 huber_c: float | None = None, x0=None, return_down: bool = False) -> dict:
    """One band, one sounding.

    ``weight_var(t, chan, s)`` gives the per-channel variance used for weights
    (``chi2_own`` is χ² against it); ``cal_var`` the calibrated total error
    variance used for the σ(k) sandwich, ``chi2_cal`` and the standardised-
    residual autocorrelation. ``huber_c`` adds N_IRLS Huber
    reweightings on r/sqrt(weight_var).
    """
    mask = np.isfinite(ln_t) & np.isfinite(tau) & np.isfinite(sig) & (sig > 0)
    chan = np.flatnonzero(mask)
    if len(chan) < 8:
        return nan_result(return_down)
    t = tau[chan].astype(np.float64)
    y = ln_t[chan].astype(np.float64)
    s = sig[chan].astype(np.float64)
    v_w = weight_var(t, chan, s)
    v_c = cal_var(t, chan, s)
    if not (np.all(np.isfinite(v_w)) and np.all(v_w > 0)):
        return nan_result(return_down)
    solve = _solve_c2 if model == "c2" else _solve_gamma
    fmodel = c2_model if model == "c2" else gamma_model
    w_h = np.ones_like(t)
    n_pass = 1 + (N_IRLS if huber_c else 0)
    params = x0
    try:
        for it in range(n_pass):
            params, bound = solve(t, y, np.sqrt(w_h / v_w), params)
            r = y - fmodel(params, t)
            if huber_c and it < n_pass - 1:
                u = np.abs(r) / np.sqrt(v_w)
                w_h = np.where(u <= huber_c, 1.0, huber_c / np.maximum(u, 1e-12))
    except (np.linalg.LinAlgError, ValueError):
        return nan_result(return_down)

    jac = design_matrix(t, 2) if model == "c2" else gamma_jac(params, t)
    w = w_h / v_w
    n, npar = jac.shape

    def sandwich(var):
        bread = np.linalg.pinv(jac.T @ (jac * w[:, None]))
        meat = jac.T @ (jac * (w**2 * var)[:, None])
        return bread @ meat @ bread

    cov_noise = sandwich(s**2)
    cov_cal = sandwich(v_c)
    if model == "c2":
        c, k1, k2 = params
        g2 = np.array([0.0, 0.0, 1.0])
    else:
        c, k1, kap = params
        k2 = k1**2 / kap
        g2 = np.array([0.0, 2 * k1 / kap, -(k1**2) / kap**2])   # k2 = k1²/κ
    order_back = np.argsort(chan)
    rho = _lag1((r / np.sqrt(v_c))[order_back])
    rho_c = float(np.clip(rho, 0.0, 0.99)) if np.isfinite(rho) else 0.0
    infl = np.sqrt((1.0 + rho_c) / (1.0 - rho_c))
    sig_k1_cal = float(np.sqrt(cov_cal[1, 1]))
    sig_k2_cal = float(np.sqrt(g2 @ cov_cal @ g2))
    out = {
        "k1": float(k1), "k2": float(k2), "intercept": float(c),
        "sig_k1": float(np.sqrt(cov_noise[1, 1])),
        "sig_k2": float(np.sqrt(g2 @ cov_noise @ g2)),
        "sig_k1_rss": np.nan,
        "chi2_red": float(np.sum(r**2 / s**2) / max(n - npar, 1)),
        "resid_rms": float(np.sqrt(r @ r / n)),
        "lag1": _lag1(r[order_back]),
        "bound": bool(bound),
        "nchan": n,
        "sig_k1_cal": sig_k1_cal, "sig_k2_cal": sig_k2_cal,
        "sig_k1_infl": sig_k1_cal * infl, "sig_k2_infl": sig_k2_cal * infl,
        "chi2_cal": float(np.sum(r**2 / v_c) / max(n - npar, 1)),
        "chi2_own": float(np.sum(r**2 / v_w) / max(n - npar, 1)),
        "lag1_std": rho,
        "frac_down": float(np.mean(w_h < 1.0)) if huber_c else 0.0,
    }
    if model == "gamma":
        out["kappa"] = float(kap)
    if return_down:
        down = np.zeros(len(tau), bool)
        down[chan[w_h < 1.0]] = True
        out["_down"] = down
    return out


def nan_result(return_down: bool = False) -> dict:
    from variants import FIELDS
    out = {k: np.nan for k in (*FIELDS, *EXTRA_FIELDS)}
    out["bound"] = False
    out["nchan"] = 0
    if return_down:
        out["_down"] = None
    return out


def k1_kernel(model: str, params, tau, w_sqrt):
    """Equivalent kernel h of k1 (k1 ≈ Σ h_i y_i at the linearisation point)."""
    jac = design_matrix(tau, 2) if model == "c2" else gamma_jac(params, tau)
    jw = jac * w_sqrt[:, None]
    h = np.linalg.pinv(jw.T @ jw) @ (jw.T * w_sqrt[None, :])
    return h[1]
