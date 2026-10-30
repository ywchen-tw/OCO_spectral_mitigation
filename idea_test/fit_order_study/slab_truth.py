"""Slab-simulation inputs and the 'true' photon path-length distributions (PPDFs).

Reads the read-only Appendix-G x-z slab outputs (``results/rt_slab_sim``) and
builds, per column, the reference PPDFs against which the spectral fitters are
judged.  Conventions follow ``workspace/rt_slab_sim``:

* T = π·rad / (μ0·toa); the constant normalization goes into the intercept.
* Relative path l' = 1 + (L − L_peak)/l_direct, with L the tallied total
  geometric path, L_peak the far-field (x < 5 km) dark/3-D histogram peak (the
  direct surface bounce, l' = 1), and l_direct = 60 km·(1/μ0 + 1/μv)
  (``make_fig_g1.py``).

Three reference PPDFs per column:

``geo784``  Geometric l' from the three τ ≈ 1e-4 anchor channels at 784 nm
            (the reference of ``closure_stats.json``).
``geo_rw``  Geometric l', radiance-weighted average of all 33 channel
            histograms.
``abs``     Absorption-weighted relative path A for the band-mean vertical
            absorber profile.  The MCARaTS tally is geometric, but each
            channel's histogram is radiance weighted *with* that channel's
            absorption, so for every geometric-path bin b the channel sequence
            ln[T_λ·h_λ(b)] is the log Laplace transform of A within the bin.
            It is fitted over the 30 absorbing channels as
                ln[T_λ h_λ(b)] = α_b − τ_λ (a_b + β_b η_λ) + ½ v_b τ_λ²,
            where η_λ is the channel's vertical-shape index (leading SVD mode
            of its normalized layer-OD profile, zero for the band mean).  Then
            exp(α_b) is the bin's non-absorbing radiance at band wavelengths,
            a_b its conditional mean A for the mean profile and v_b its
            conditional variance.  The PPDF is the mixture of per-bin gammas
            with these moments.  ``geo_band`` is the geometric l' under the
            same exp(α_b) weights (band-wavelength Rayleigh, unlike geo784).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
SIM_DIR = REPO / "results" / "rt_slab_sim"
TOA_KM = 60.0
SENSOR_ZENITH_DEG = 0.0
NX, DX_KM = 64, 0.5
CLOUD_X_KM = (9.5, 14.5)
ANCHOR_TAU = 1e-3          # channels below this are the 784-nm continuum anchors
MERGE_MASS = 0.005         # adaptive bins: merge native bins up to this mass ...
MERGE_WIDTH = 0.03         # ... or this width in l'
MIN_BIN_MASS = 1e-5


@dataclass
class SlabCase:
    surface: str
    solver: str
    tau: np.ndarray            # (nw,) slant τ
    wvl_nm: np.ndarray         # (nw,)
    eta: np.ndarray            # (nw,) vertical-shape index (0 for anchors)
    ln_t: np.ndarray           # (nw, nx) MC-mean ln T
    ln_t_runs: np.ndarray      # (nw, nx, nrun)
    sig_ln_t: np.ndarray       # (nw, nx) σ of the run mean, from the run spread
    hist: np.ndarray           # (nw, nrun, nx, nbin) native radiance-fraction histograms
    l_mid: np.ndarray          # (nbin,) l' at native bin centres
    x_km: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.surface}/{self.solver}"

    @property
    def absorbing(self) -> np.ndarray:
        return self.tau > ANCHOR_TAU


def column_masks(x_km: np.ndarray) -> dict[str, np.ndarray]:
    """'all', 'clear' (outside the cloud, as closure_stats.json) and 'far'
    (outside the cloud and its 3-D shadow band, x < 5 or x > 25 km)."""
    in_cloud = (x_km >= CLOUD_X_KM[0]) & (x_km < CLOUD_X_KM[1])
    far = (x_km < 5.0) | (x_km > 25.0)
    return {"all": np.ones_like(x_km, bool), "clear": ~in_cloud, "far": far}


def shape_index(od_layer: np.ndarray, absorbing: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Leading SVD score of the normalized layer-OD profiles (absorbing channels).

    Returns (eta, mode, frac): eta is zero for the band-mean profile and
    positive when the channel's absorption sits higher than the mean; frac is
    the share of the profile variance in the mode."""
    w = od_layer / od_layer.sum(axis=1, keepdims=True)
    mean = w[absorbing].mean(axis=0)
    d = w[absorbing] - mean
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    mode = vt[0]
    nlay = w.shape[1]
    if mode[nlay // 2:].sum() < 0:      # sign: positive = weight aloft
        mode = -mode
    eta = np.zeros(len(w))
    eta[absorbing] = d @ mode
    return eta, mode, float(s[0] ** 2 / np.sum(s**2))


def load_cases(sim_dir: Path = SIM_DIR) -> tuple[dict[str, SlabCase], dict]:
    with h5py.File(sim_dir / "slab_od.h5", "r") as f:
        sza = float(f.attrs["sza"])
        od_layer = f["od_layer"][...]
        tau_od = f["slant_tau"][...]
    mu0 = np.cos(np.deg2rad(sza))
    l_direct = TOA_KM * (1.0 / mu0 + 1.0 / np.cos(np.deg2rad(SENSOR_ZENITH_DEG)))
    x_km = (np.arange(NX) + 0.5) * DX_KM
    cases: dict[str, SlabCase] = {}
    with h5py.File(sim_dir / "slab_rad.h5", "r") as f:
        tau_all = f["slant_tau"][...]
        wvl_all = f["wvl_nm"][...]
        if not np.allclose(tau_all, tau_od):
            raise ValueError("slab_rad and slab_od slant_tau differ")
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"], f.attrs["plen_nbin"] + 1) / 1e3
        mid_km = 0.5 * (edges[:-1] + edges[1:])
        raw = {}
        for surface in ("dark", "bright"):
            for solver in ("3d", "ipa"):
                g = f[f"{surface}/{solver}"]
                iw = g["iw"][...]
                rad = g["rad"][...].reshape(iw.size, NX).astype(np.float64)
                runs = g["rad_runs"][...].reshape(iw.size, NX, -1).astype(np.float64)
                toa = g["toa"][...]
                hist = g["plen_hist"][...].astype(np.float64)
                raw[(surface, solver)] = (iw, rad, runs, toa, hist)
    # direct-bounce anchor: far-field dark/3-D histogram peak at the anchor channels
    iw, _, _, _, hist = raw[("dark", "3d")]
    anchor = tau_all[iw] < ANCHOR_TAU
    h0 = hist[anchor].mean(axis=(0, 1))
    l_peak = float(mid_km[np.argmax(h0[x_km < 5.0].mean(axis=0))])
    l_mid = 1.0 + (mid_km - l_peak) / l_direct

    for (surface, solver), (iw, rad, runs, toa, hist) in raw.items():
        tau = tau_all[iw]
        absorbing = tau > ANCHOR_TAU
        eta, mode, frac = shape_index(od_layer[iw], absorbing)
        t_mean = np.pi * rad / (mu0 * toa[:, None])
        t_runs = np.pi * runs / (mu0 * toa[:, None, None])
        nrun = runs.shape[-1]
        sig = runs.std(axis=-1, ddof=1) / np.sqrt(nrun) / rad
        cases[f"{surface}/{solver}"] = SlabCase(
            surface, solver, tau, wvl_all[iw], eta, np.log(t_mean), np.log(t_runs), sig,
            hist, l_mid, x_km, meta={"shape_mode": mode, "shape_frac": frac})
    meta = {"sza": sza, "mu0": mu0, "l_direct_km": l_direct, "l_peak_km": l_peak,
            "bin_dl": float(l_mid[1] - l_mid[0])}
    return cases, meta


# --------------------------------------------------------------------------
# Moments of discrete distributions
# --------------------------------------------------------------------------
def moments(w: np.ndarray, x: np.ndarray, var_within: np.ndarray | None = None) -> dict:
    """Mean, variance, skewness, excess kurtosis of a mixture with component
    weights w at means x (optional component variances, assumed gamma-shaped
    for the 3rd/4th central moments)."""
    w = np.asarray(w, float)
    w = w / w.sum()
    x = np.asarray(x, float)
    v = np.zeros_like(x) if var_within is None else np.clip(var_within, 0.0, None)
    m = np.sum(w * x)
    d = x - m
    # component central moments of a gamma with mean x, var v: μ3 = 2v²/x, μ4 = 3v² + 6v³/x²
    with np.errstate(divide="ignore", invalid="ignore"):
        c3 = np.where(x > 0, 2 * v**2 / x, 0.0)
        c4 = 3 * v**2 + np.where(x > 0, 6 * v**3 / x**2, 0.0)
    var = np.sum(w * (d**2 + v))
    mu3 = np.sum(w * (d**3 + 3 * d * v + c3))
    mu4 = np.sum(w * (d**4 + 6 * d**2 * v + 4 * d * c3 + c4))
    return {"mean": m, "var": var, "skew": mu3 / var**1.5, "exkurt": mu4 / var**2 - 3.0,
            "var_between": float(np.sum(w * d**2))}


def laplace_log(w: np.ndarray, x: np.ndarray, tau: np.ndarray, var_within: np.ndarray | None = None) -> np.ndarray:
    """ln Σ_b w_b·E[exp(−τ A_b)] for gamma components (mean x, var v) or deltas."""
    w = np.asarray(w, float) / np.sum(w)
    tau = np.asarray(tau, float)[:, None]
    if var_within is None:
        lt = -tau * x[None, :]
    else:
        v = np.clip(var_within, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = np.where(v > 0, x**2 / np.where(v > 0, v, 1.0), np.inf)
            lt_g = -kap[None, :] * np.log1p(tau * x[None, :] / kap[None, :])
        lt = np.where(np.isfinite(kap)[None, :], lt_g, -tau * x[None, :])
    lmax = lt.max(axis=1, keepdims=True)
    return (lmax + np.log(np.sum(w[None, :] * np.exp(lt - lmax), axis=1, keepdims=True)))[:, 0]


# --------------------------------------------------------------------------
# Reference PPDFs per column
# --------------------------------------------------------------------------
def adaptive_bins(h: np.ndarray, l_mid: np.ndarray) -> list[np.ndarray]:
    """Merge consecutive native bins until MERGE_MASS or MERGE_WIDTH is reached."""
    groups, cur, mass = [], [], 0.0
    for i in np.flatnonzero(h > 0):
        if cur and (mass >= MERGE_MASS or l_mid[i] - l_mid[cur[0]] >= MERGE_WIDTH):
            groups.append(np.array(cur))
            cur, mass = [], 0.0
        cur.append(i)
        mass += h[i]
    if cur:
        groups.append(np.array(cur))
    # fold too-light groups into their left neighbour
    out = []
    for gidx in groups:
        if out and h[gidx].sum() < MIN_BIN_MASS:
            out[-1] = np.concatenate([out[-1], gidx])
        else:
            out.append(gidx)
    return out


def _bin_model(p, tau, eta):
    alpha, a, beta, log_kap = p
    kap = np.exp(log_kap)
    x = np.maximum(1.0 + tau * (a + beta * eta) / kap, 1e-12)
    return alpha - kap * np.log(x)


def _fit_bin(y, tau, eta, wgt):
    """Weighted fit of one bin's ln[T h] sequence; linear start, then gamma form."""
    design = np.column_stack([np.ones_like(tau), -tau, -tau * eta])
    lin = np.linalg.lstsq(design * wgt[:, None], y * wgt, rcond=None)[0]
    p0 = [lin[0], max(lin[1], 1e-3), lin[2], np.log(1e3)]
    sol = least_squares(lambda p: (_bin_model(p, tau, eta) - y) * wgt, p0,
                        bounds=([-np.inf, 0.0, -np.inf, np.log(0.2)], [np.inf, np.inf, np.inf, np.log(1e6)]),
                        method="trf", x_scale="jac", max_nfev=200)
    return sol.x


def reconstruct_abs(case: SlabCase, ix: int, run: int | None = None) -> dict:
    """Per-bin absorption-weighted reconstruction for one column (see module doc).

    Per geometric-path bin b, the 30 absorbing channels give
        ln[T_λ h_λ(b)] = α_b − κ_b ln(1 + τ_λ (a_b + β_b η_λ)/κ_b),
    the log Laplace transform of a gamma-shaped conditional distribution of the
    absorption-weighted path A with mean a_b + β_b η_λ and variance a_b²/κ_b
    (for the band-mean profile, η = 0).  Weights ∝ sqrt(T h), i.e. Poisson-like
    Monte Carlo noise of the bin's radiance contribution.
    """
    ab = case.absorbing
    tau, eta = case.tau[ab], case.eta[ab]
    if run is None:
        hist = case.hist[:, :, ix].mean(axis=1)                 # (nw, nbin)
        ln_t = case.ln_t[:, ix]
    else:
        hist = case.hist[:, run, ix]
        ln_t = case.ln_t_runs[:, ix, run]
    hist = hist / hist.sum(axis=1, keepdims=True)
    base = np.argmin(np.where(ab, case.tau, np.inf))            # weakest absorbing channel
    groups = adaptive_bins(hist[base], case.l_mid)
    t_ab = np.exp(ln_t[ab])
    alpha, a, beta, v, lg, lg2, rms = (np.full(len(groups), np.nan) for _ in range(7))
    for k, gidx in enumerate(groups):
        hb = hist[ab][:, gidx].sum(axis=1)
        ok = hb > 0
        if ok.sum() < 8:
            continue
        contrib = t_ab[ok] * hb[ok]
        y = np.log(contrib)
        wgt = np.sqrt(contrib / contrib.max())
        p = _fit_bin(y, tau[ok], eta[ok], wgt)
        alpha[k], a[k], beta[k] = p[:3]
        v[k] = p[1] ** 2 / np.exp(p[3])
        rms[k] = np.sqrt(np.sum(wgt**2 * (y - _bin_model(p, tau[ok], eta[ok])) ** 2) / np.sum(wgt**2))
        hw = hist[base, gidx]
        lg[k] = np.sum(hw * case.l_mid[gidx]) / hw.sum()
        lg2[k] = np.sum(hw * case.l_mid[gidx] ** 2) / hw.sum() - lg[k] ** 2
    good = np.isfinite(alpha)
    w = np.exp(alpha[good] - np.nanmax(alpha))
    lost = 1.0 - hist[base][np.concatenate([g for g, ok_ in zip(groups, good) if ok_])].sum()
    # self-consistency: rebuild the column spectrum (with each channel's own shape) from the bins
    kap = np.where(v[good] > 0, a[good] ** 2 / np.where(v[good] > 0, v[good], 1.0), 1e6)
    xx = np.maximum(1.0 + tau[:, None] * (a[good] + beta[good] * eta[:, None]) / kap, 1e-12)
    pred = np.log(np.sum(np.exp(alpha[good][None, :] - kap * np.log(xx)), axis=1))
    return {"w": w / w.sum(), "a": a[good], "beta": beta[good], "v": v[good], "lg": lg[good],
            "lg_var": lg2[good], "rms": rms[good], "ln_t0": float(np.log(np.sum(np.exp(alpha[good])))),
            "rebuild_resid": pred - ln_t[ab],
            "rebuild_rms": float(np.sqrt(np.mean((pred - ln_t[ab]) ** 2))), "lost_mass": float(lost),
            "nbin": int(good.sum())}


def geometric_hist(case: SlabCase, ix: int, which: str, run: int | None = None) -> np.ndarray:
    h = case.hist[:, :, ix].mean(axis=1) if run is None else case.hist[:, run, ix]
    h = h / h.sum(axis=1, keepdims=True)
    if which == "geo784":
        out = h[~case.absorbing].mean(axis=0)
    elif which == "geo_rw":
        t = np.exp(case.ln_t[:, ix] if run is None else case.ln_t_runs[:, ix, run])
        out = (t[:, None] * h).sum(axis=0)
    else:
        raise ValueError(which)
    return out / out.sum()


def column_truths(case: SlabCase, run: int | None = None) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    """Moments of every reference PPDF for every column, plus the per-column
    absorption-weighted reconstructions (for the theorem spectra)."""
    names = ("geo784", "geo_rw", "geo_band", "abs")
    stats = ("mean", "var", "skew", "exkurt", "var_between")
    out = {n: {s: np.full(NX, np.nan) for s in stats} for n in names}
    recs = []
    for ix in range(NX):
        for n in ("geo784", "geo_rw"):
            m = moments(geometric_hist(case, ix, n, run), case.l_mid)
            for s in stats:
                out[n][s][ix] = m[s]
        rec = reconstruct_abs(case, ix, run)
        recs.append(rec)
        for n, (x, v) in {"geo_band": (rec["lg"], rec["lg_var"]), "abs": (rec["a"], rec["v"])}.items():
            m = moments(rec["w"], x, v)
            for s in stats:
                out[n][s][ix] = m[s]
    return out, recs


def tilted_cumulants(w: np.ndarray, x: np.ndarray, tau: np.ndarray, var_within: np.ndarray | None = None):
    """Local cumulants of ln T(τ) = ln Σ w_b E[exp(−τ A_b)]:
    K1(τ) = −d lnT/dτ and K2(τ) = d² lnT/dτ², i.e. mean and variance of the
    PPDF tilted by exp(−τ A).  K1(0), K2(0) are the PPDF mean and variance."""
    w = np.asarray(w, float) / np.sum(w)
    tau = np.atleast_1d(np.asarray(tau, float))[:, None]
    x = np.asarray(x, float)[None, :]
    v = np.zeros_like(x) if var_within is None else np.clip(var_within, 0.0, None)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = np.where(v > 0, x**2 / np.where(v > 0, v, 1.0), np.inf)
        g = np.where(np.isfinite(kap), 1.0 + tau * x / kap, 1.0)
        log_phi = np.where(np.isfinite(kap), -kap * np.log(g), -tau * x)
    m_b = x / g                                   # tilted component mean
    v_b = v / g**2                                # tilted component variance
    lp = log_phi + np.log(w)[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    k1 = np.sum(p * m_b, axis=1)
    k2 = np.sum(p * (v_b + m_b**2), axis=1) - k1**2
    return k1, k2
