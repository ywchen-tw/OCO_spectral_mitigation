"""Physical PPDF retrieval in cumulant space (round 3, 2026-11-01).

Non-ML, few-parameter photon-path model (in the spirit of GOSAT PPDF-S, Bril / Oshchepkov /
Yokota 2007-2012) inverted by Levenberg-Marquardt optimal estimation against the observed
agreed-model (_am) fit coefficients.  The synthetic spectrum of each band is fitted with the
SAME _am fit functions as the observations (imported from src/spectral/cumulant_fit.py), so
predicted and observed coefficients have identical definitions.

Forward model (per band b, per channel j; reflectance-factor units RF = pi I / (mu0 F0)):
  gas  : slant tau_j from the tau file split into Rayleigh (frac_ray), H2O (frac_h2o) and dry gas;
         dry gas scaled by s_p (O2A) or g*s_p (WCO2/SCO2), H2O by h.  The fraction of a channel's
         gas absorption above sigma = p/ps is F_j(sigma) = sigma^(m_j+1), m_j from the channel's
         absorption-weighted pressure p_abs (slab_mc's profile construction).
  layer: one scattering layer at sigma_H, OD tau_L(765) * (lambda/0.765)^-alpha, omega = 0.95,
         HG asymmetry 0.7 (delta-M scaled for attenuation / diffuse terms, exact phase for
         single scattering).
  components
    (a) direct surface reflection:          rho * E0 Ev * exp(-tau_gas - tau_R)
    (b) layer single scattering (shortening): omega P(Theta) [1-exp(-tau_L M)] / (4(mu0+muv))
                                              * exp(-F(sigma_H) tau_gas - sigma_H tau_R)
    (c) surface-layer multiple reflections (lengthening): (a) * sum_k (rho_f r_L t_b^2)^k,
        r_L = omega* tau* (1-g*) (thin-layer spherical albedo), t_b^2 = exp(-2 D (1-F) tau_gas_vert),
        D = 5/3 diffusivity
    (e) diffuse transmission through the layer (one diffuse leg below the layer, both orders)
    (d) Rayleigh single scattering integrated over sigma (8-point Gauss-Legendre) [real data only]
  rho (direct surface reflectance) is set per band from the OBSERVED continuum exp(c_am) so that
  the model continuum equals it; rho_f (reflectance for diffuse light) = rho over land and in the
  slab MC (Lambertian), 0.06 over ocean (glint mode).  T = K * RF with K = mu0^2 for the OCO-2
  T definition of fitting.py (toa_sol = 0.5 F0 / mu0, one polarization) and K = N_PHOT/pi in
  the slab MC.

State x = [g, s_p, h, ln tau_L(765), alpha, sigma_H] with weak priors (XA, SA_SD).
Observables: O2A k1_am, f_am, k2diag_am; WCO2 k1_am, k1w_am; SCO2 k1_am, k1w_am, k2diag_am.
(WCO2 k2diag_am is excluded: it is negative in the real data, -0.08 to -0.19, which no
positive photon-path distribution can produce.)
S_y = ALPHA^2 * S_formal + diag(FLOOR^2): S_formal is the fit's formal covariance reconstructed
by linear propagation of iid residuals (s^2 = rms^2 n/(n-p)) through the fit (full cross-band-
coefficient covariance, e.g. the strong O2A k1/f anticorrelation); FLOOR is a forward-model
error floor.
"""
from __future__ import annotations

import time

import numpy as np

from spectral.cumulant_fit import _fit_am_o2a, _fit_am_sco2, _fit_am_wco2

BANDS = ("o2a", "wco2", "sco2")
WL = {"o2a": 0.765, "wco2": 1.61, "sco2": 2.06}
OMEGA, G_HG, D_DIFF, RHO_OCEAN_DIFF = 0.95, 0.7, 5.0 / 3.0, 0.06
STATE = ("g", "s_p", "h", "ln_tauL", "alpha", "sigma_H")
XA = np.array([1.0, 1.0, 1.0, np.log(0.08), 1.0, 0.85])
SA_SD = np.array([0.01, 0.002, 0.2, 1.0, 0.7, 0.15])
LB = np.array([0.9, 0.97, 0.2, np.log(1e-3), -1.0, 0.3])
UB = np.array([1.1, 1.03, 3.0, np.log(3.0), 4.0, 1.0])
STEP = np.array([1e-3, 1e-3, 0.02, 0.05, 0.05, 0.01])
OBS = (("o2a", "k1_am"), ("o2a", "f_am"), ("o2a", "k2diag_am"),
       ("wco2", "k1_am"), ("wco2", "k1w_am"),
       ("sco2", "k1_am"), ("sco2", "k1w_am"), ("sco2", "k2diag_am"))
OBS_NAMES = tuple(f"{b}_{k}" for b, k in OBS)
FLOOR = np.array([0.004, 0.002, 0.005, 0.002, 0.03, 0.002, 0.03, 0.003])
ALPHA = 1.0
OBS_ALL, OBS_NAMES_ALL, FLOOR_ALL = OBS, OBS_NAMES, FLOOR.copy()


def set_obs(exclude=()):
    """Select the matched observables (module-level; call before forking workers)."""
    global OBS, OBS_NAMES, FLOOR
    keep = [i for i, n in enumerate(OBS_NAMES_ALL) if n not in set(exclude)]
    OBS = tuple(OBS_ALL[i] for i in keep)
    OBS_NAMES = tuple(OBS_NAMES_ALL[i] for i in keep)
    FLOOR = FLOOR_ALL[keep]
GL_X, GL_W = np.polynomial.legendre.leggauss(8)
GL_S, GL_W = 0.5 * (GL_X + 1.0), 0.5 * GL_W


def m_profile(pabs, ps):
    """slab_mc's vertical profile: d tau/d sigma ~ (m+1) sigma^m with mean sigma = p_abs/ps."""
    sr = np.clip(np.nan_to_num(pabs / ps, nan=0.5), 0.4, 0.93)
    return np.clip((2 * sr - 1) / (1 - sr), -0.5, 12.0)


def hg(g, c):
    return (1 - g * g) / (1 + g * g - 2 * g * c) ** 1.5


# ── scene container ─────────────────────────────────────────────────────────────

def make_scene(mu0, muv, cos_sca, K, land_fit, lambert, bands):
    """bands[b] = dict(tau, fh, fr, m, x, c_obs) on the fit channel set (any order)."""
    sc = dict(mu0=mu0, muv=muv, amf=1 / mu0 + 1 / muv, cos_sca=cos_sca, K=K,
              land_fit=bool(land_fit), lambert=bool(lambert), b={})
    for b, d in bands.items():
        tau = np.asarray(d["tau"], float)
        ok = np.isfinite(tau)
        s = np.argsort(tau[ok])
        sel = lambda a: np.asarray(a, float)[ok][s]
        tau_s = sel(tau)
        fh = np.nan_to_num(sel(d["fh"]))
        fr = np.nan_to_num(sel(d["fr"]))
        e = dict(tau=tau_s, x=sel(d["x"]), m=sel(d["m"]), c_obs=float(d["c_obs"]),
                 order=np.flatnonzero(ok)[s])
        e["tw_fit"] = fh * tau_s                    # what the _am fit uses
        e["td_fit"] = tau_s - e["tw_fit"]
        e["t_ray"] = fr * tau_s
        e["t_h2o"] = fh * tau_s
        e["t_dry"] = np.clip(tau_s - e["t_ray"] - e["t_h2o"], 0, None)
        sc["b"][b] = e
    return sc


# ── forward model ───────────────────────────────────────────────────────────────

def band_spectrum(sc, b, th):
    """ln T for band b at state th (and the surface reflectance used)."""
    g, sp, h, ltau, ang, sigH = th
    e = sc["b"][b]
    mu0, muv, M = sc["mu0"], sc["muv"], sc["amf"]
    scale = sp if b == "o2a" else g * sp
    tg = scale * e["t_dry"] + h * e["t_h2o"]          # slant gas tau
    tgv = tg / M
    tR = e["t_ray"]
    tRv = tR / M
    F = sigH ** (e["m"] + 1.0)
    tauL = np.exp(ltau) * (WL[b] / 0.765) ** (-ang)
    f = G_HG ** 2
    ts = tauL * (1 - OMEGA * f)
    ws = OMEGA * (1 - f) / (1 - OMEGA * f)
    gs = (G_HG - f) / (1 - f)
    E0, Ev = np.exp(-ts / mu0), np.exp(-ts / muv)
    rL = min(ws * ts * (1 - gs), 0.5)
    td0 = ws * (1 - E0) * 0.5 * (1 + gs)
    tdv = ws * (1 - Ev) * 0.5 * (1 + gs)
    # gas-free continuum pieces (Rayleigh extinction kept: it is smooth and absorbed by c, a1)
    rb_amp = OMEGA * hg(G_HG, sc["cos_sca"]) * (1 - np.exp(-tauL * M)) / (4 * (mu0 + muv))
    pr = 0.75 * (1 + sc["cos_sca"] ** 2)
    # Rayleigh single scattering, integrated over sigma
    rd = np.zeros_like(tg)
    if np.any(tRv > 0):
        for s_, w_ in zip(GL_S, GL_W):
            att = np.exp(-M * (s_ ** (e["m"] + 1.0) * tgv + s_ * tRv)) * (np.exp(-tauL * M) if s_ > sigH else 1.0)
            rd += w_ * tRv * att
        rd *= pr / (4 * mu0 * muv)
    gas_full = np.exp(-tg - tR)
    gas_above = np.exp(-F * tg - sigH * tR)
    tb2 = np.exp(-2 * D_DIFF * (1 - F) * tgv)
    gas_e1 = np.exp(-(F * tgv / mu0 + (1 - F) * tgv * D_DIFF + tgv / muv) - tR)
    gas_e2 = np.exp(-(tgv / mu0 + (1 - F) * tgv * D_DIFF + F * tgv / muv) - tR)
    # surface reflectance from the observed continuum (gas = 0): fixed point
    target = np.exp(e["c_obs"]) / sc["K"]
    rR0 = np.exp(-np.median(tR))
    rd0 = float(np.median(rd[:5])) if np.any(tRv > 0) else 0.0   # least-absorbing channels
    rho = max(target / (E0 * Ev * rR0), 1e-3)
    for _ in range(4):
        rf = rho if (sc["land_fit"] or sc["lambert"]) else RHO_OCEAN_DIFF
        q = rf * rL
        per = E0 * Ev * rR0 * (1 + q / (1 - q)) + (rf / rho) * (td0 * Ev + E0 * tdv) * rR0
        rho = max((target - rb_amp * np.exp(-sigH * np.median(tR)) - rd0) / per, 1e-3)
    rf = rho if (sc["land_fit"] or sc["lambert"]) else RHO_OCEAN_DIFF
    q = rf * rL * tb2
    RF = (rho * E0 * Ev * gas_full * (1 + q / (1 - q))
          + rb_amp * gas_above
          + rf * (td0 * Ev * gas_e1 + E0 * tdv * gas_e2)
          + rd)
    return np.log(np.maximum(sc["K"] * RF, 1e-300)), rho


def fit_band(sc, b, y):
    e = sc["b"][b]
    if b == "o2a":
        return _fit_am_o2a(e["tau"], e["x"], y)
    if b == "wco2":
        return _fit_am_wco2(e["td_fit"], e["tw_fit"], e["x"], y)
    return _fit_am_sco2(e["td_fit"], e["tw_fit"], e["x"], y, sc["land_fit"])


def forward(sc, th, base=None, skip_o2a=False):
    """Predicted observable vector; with skip_o2a the O2A entries are copied from ``base``."""
    out = np.full(len(OBS), np.nan)
    fits = {}
    for b in BANDS:
        if b == "o2a" and skip_o2a and base is not None:
            continue
        y, _ = band_spectrum(sc, b, th)
        fits[b] = fit_band(sc, b, y)
    for i, (b, k) in enumerate(OBS):
        if b in fits:
            out[i] = fits[b].get(k, np.nan)
        else:
            out[i] = base[i]
    return out


# ── formal covariance of the observed coefficients ──────────────────────────────

def _o2a_jac(tau, x, c, a1, k1, f, k2):
    E = np.exp(-k1 * tau + 0.5 * k2 * tau ** 2)
    Dn = (1 - f) * E + f
    return np.column_stack([np.ones_like(tau), x, -(1 - f) * E * tau / Dn, (1 - E) / Dn,
                            (1 - f) * E * 0.5 * tau ** 2 / Dn])


def formal_cov(sc, obsfit, rms):
    """S_formal (len(OBS) square) by linear propagation of iid residuals through each band's
    fit(s).  obsfit[b] = dict of observed coefficients (c_am, a1_am, k1_am, f_am, k2diag_am);
    rms[b] = residual rms of the observed _am fit."""
    rows = {}
    for b in BANDS:
        e = sc["b"][b]
        n = len(e["tau"])
        o = obsfit[b]
        if b == "o2a":
            J4 = _o2a_jac(e["tau"], e["x"], o["c_am"], o["a1_am"], o["k1_am"], o["f_am"], 0.0)[:, :4]
            J5 = _o2a_jac(e["tau"], e["x"], o["c_am"], o["a1_am"], o["k1_am"], o["f_am"], o["k2diag_am"])
            G4, G5 = np.linalg.pinv(J4), np.linalg.pinv(J5)
            rows[b] = {"k1_am": G4[2], "f_am": G4[3], "k2diag_am": G5[4]}
            p = 4
        elif b == "wco2":
            one = np.ones(n)
            A4 = np.column_stack([one, e["x"], -e["td_fit"], -e["tw_fit"]])
            A5 = np.column_stack([A4, 0.5 * e["tau"] ** 2])
            G4, G5 = np.linalg.pinv(A4), np.linalg.pinv(A5)
            rows[b] = {"k1_am": G4[2], "k1w_am": G4[3], "k2diag_am": G5[4]}
            p = 4
        else:
            one = np.ones(n)
            cols = [one, e["x"]] + ([e["x"] ** 2] if sc["land_fit"] else []) + \
                [-e["td_fit"], -e["tw_fit"], 0.5 * e["tau"] ** 2]
            G = np.linalg.pinv(np.column_stack(cols))
            k = 3 if sc["land_fit"] else 2
            rows[b] = {"k1_am": G[k], "k1w_am": G[k + 1], "k2diag_am": G[k + 2]}
            p = len(cols)
        rows[b]["s2"] = float(rms[b]) ** 2 * n / max(n - p, 1)
    S = np.zeros((len(OBS), len(OBS)))
    for i, (bi, ki) in enumerate(OBS):
        for j, (bj, kj) in enumerate(OBS):
            if bi == bj:
                S[i, j] = rows[bi]["s2"] * float(rows[bi][ki] @ rows[bj][kj])
    return S


# ── optimal estimation (Levenberg-Marquardt, Rodgers 2000 sec. 5.7) ─────────────

def jacobian(sc, th, F0):
    Kj = np.zeros((len(OBS), len(STATE)))
    nf = 0
    for p in range(len(STATE)):
        t2 = th.copy()
        step = STEP[p] if th[p] + STEP[p] <= UB[p] else -STEP[p]
        t2[p] += step
        Fp = forward(sc, t2, base=F0, skip_o2a=(p == 0))
        nf += 1
        Kj[:, p] = (Fp - F0) / step
    return Kj, nf


def retrieve(sc, y, Sy, xa=XA, sa_sd=SA_SD, max_iter=12, gamma0=1.0):
    t0 = time.perf_counter()
    Syi = np.linalg.inv(Sy)
    Sai = np.diag(1.0 / sa_sd ** 2)

    def cost(x, F):
        if not np.all(np.isfinite(F)):
            return np.inf
        r = y - F
        dx = x - xa
        return float(r @ Syi @ r + dx @ Sai @ dx)

    x = xa.copy()
    F = forward(sc, x)
    nfwd = 1
    c = cost(x, F)
    gamma = gamma0
    converged = False
    it = 0
    Kj = None
    for it in range(1, max_iter + 1):
        Kj, nf = jacobian(sc, x, F)
        nfwd += nf
        if not np.all(np.isfinite(Kj)):
            break
        accepted = False
        while gamma <= 1e5:
            H = (1 + gamma) * Sai + Kj.T @ Syi @ Kj
            rhs = Kj.T @ Syi @ (y - F) - Sai @ (x - xa)
            xn = np.clip(x + np.linalg.solve(H, rhs), LB, UB)
            Fn = forward(sc, xn)
            nfwd += 1
            cn = cost(xn, Fn)
            if cn < c:
                accepted = True
                break
            gamma *= 10
        if not accepted:
            converged = np.isfinite(c)      # no downhill step left: at a (local) minimum
            break
        dc = c - cn
        x, F, c = xn, Fn, cn
        gamma = max(gamma / 10, 1e-3)
        if dc < 0.01 * len(OBS) or dc / max(c, 1e-9) < 1e-3:
            converged = True
            break
    out = dict(converged=bool(converged), n_iter=it, n_fwd=nfwd, cost=c,
               ms=(time.perf_counter() - t0) * 1e3)
    if Kj is not None and np.all(np.isfinite(Kj)) and np.isfinite(c):
        Kj, nf = jacobian(sc, x, F)
        out["n_fwd"] += nf
        Sh = np.linalg.inv(Kj.T @ Syi @ Kj + Sai)
        A = Sh @ Kj.T @ Syi @ Kj
        r = y - F
        out.update(chi2_obs=float(r @ Syi @ r), dof_total=float(np.trace(A)), dof_g=float(A[0, 0]),
                   sig_g=float(np.sqrt(Sh[0, 0])), A_row_g=A[0].copy(),
                   pull=r / np.sqrt(np.diag(Sy)), resid=r, F=F)
    for i, s in enumerate(STATE):
        out[s] = float(x[i])
    return out
