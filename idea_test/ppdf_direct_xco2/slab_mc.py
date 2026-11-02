"""Slab Monte Carlo identifiability test for the PPDF direct-XCO2 path model (_am fits).

Question: even without instrument noise, can the gas-free observables of the agreed-model
fits (O2A k1, k2diag/k1^2 shapes, continuum exp(c) and slopes a1/a2, O2A fill fraction f,
geometry) predict the CO2-band path factor <l'> well enough to recover the CO2 scale g
to ~0.25 % (1 ppm)?

Model (deliberately simple, 1D plane-parallel, gray scattering within a band):
* real per-channel slant optical depths tau (dry + H2O split, frac_h2o) and absorption-
  weighted pressures p_abs from telluric-Doppler tau files (read only);
* vertical distribution of each channel's absorption d tau/d sigma ~ (m+1) sigma^m,
  sigma = p/ps, with m set so that the tau-weighted mean sigma equals p_abs/ps
  (ps ~ 99th percentile of p_abs over the three bands / 0.9); dry and water share m;
* an aerosol layer (always) and, in 30 % of scenes, a cloud layer: Henyey-Greenstein
  scattering, conservative cloud, aerosol single-scattering albedo omega_b; Lambertian
  surface with band albedo and an in-band linear slope;
* backward-free forward MC with local estimates toward the nadir/glint sensor direction
  at every scattering / surface event; each contribution carries its absorbing path for
  a grid of m, the number of surface reflections k and of aerosol scatterings n, so
  R_ch = sum_c b_c A_ch^k omega^n exp(-tau_vert,ch * L_ch(m_ch)).
* the WCO2/SCO2 dry tau is scaled by the true CO2 factor g; fits use the unscaled tau,
  exactly as the retrieval-free fits do.
Not modelled: 3D cloud effects (no near-cloud photon leakage), polarization, ocean glint
BRDF, spectrally varying scattering inside a band, separate dry/H2O vertical profiles,
instrument line shape errors.  So the floor is optimistic (a lower bound).

    python idea_test/ppdf_direct_xco2/slab_mc.py --n-train 2000 --n-test 1000 --workers 12
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_2016_2020_v2"
TAU_FILES = [
    "/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/2018-07-01/21257a_GL/fp_tau_combined.h5",
    "/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/2016-01-01/{}/fp_tau_combined.h5",
    "/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/2019-10-01/{}/fp_tau_combined.h5",
]
BANDS = ("o2a", "wco2", "sco2")
CH_STEP = 4                       # every 4th channel of slab [1:-1]
CHAN = np.arange(1, 1015, CH_STEP)
X = (CHAN - 507.5) / 507.5
M_GRID = np.array([-0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12], float)
N_PHOT = 2000
MAX_EV = 30
SNR = {"o2a": 400.0, "wco2": 500.0, "sco2": 350.0}


def load_tau_bank(n_per_file=150, seed=0):
    import glob
    import h5py
    rng = np.random.default_rng(seed)
    bank = []
    for pat in TAU_FILES:
        files = sorted(glob.glob(pat.format("*")))
        if not files:
            continue
        with h5py.File(files[len(files) // 2], "r") as f:
            n = len(f["sza"])
            idx = np.sort(rng.choice(n, size=min(n_per_file, n), replace=False))
            d = {"sza": f["sza"][idx], "vza": f["vza"][idx]}
            for b in BANDS:
                d[f"{b}_tau"] = f[f"{b}_tau_output"][idx][:, CHAN]
                d[f"{b}_fh"] = np.nan_to_num(f[f"{b}_frac_h2o"][idx][:, CHAN])
                d[f"{b}_pabs"] = f[f"{b}_p_abs_hpa"][idx][:, CHAN]
            # p_abs is the gas-absorption pressure centroid (Rayleigh excluded); the
            # near-surface H2O / far-wing channels sit at ~0.9 ps, so ps ~ q99(p_abs)/0.9
            pa = np.concatenate([d[f"{b}_pabs"] for b in BANDS], axis=1)
            d["ps"] = np.nanpercentile(pa, 99, axis=1) / 0.9
        for i in range(len(idx)):
            bank.append({k: v[i] for k, v in d.items()})
    return bank


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

def hg_sample(g, u):
    g = np.asarray(g, float)
    gs = np.where(np.abs(g) < 1e-3, 1e-3, g)
    s = (1 - gs * gs) / (1 - gs + 2 * gs * u)
    return np.where(np.abs(g) < 1e-3, 2 * u - 1, np.clip((1 + gs * gs - s * s) / (2 * gs), -1, 1))


def hg_phase(g, c):
    return (1 - g * g) / (1 + g * g - 2 * g * c) ** 1.5


def run_mc(layers, mu0, muv, rng):
    """layers: list of (sig_top, sig_bot, tau_vert, g_hg, is_aerosol).
    Returns contributions: b, k, n_aer, L (n x len(M_GRID)) in units of the vertical column."""
    sg = np.linspace(0, 1, 2001)
    beta = np.zeros_like(sg)
    gl = np.zeros_like(sg)
    aer = np.zeros_like(sg, bool)
    for st, sb, tv, g, isa in layers:
        m = (sg >= st) & (sg <= sb)
        beta[m] += tv / max(sb - st, 1e-6)
        gl[m] = g
        aer[m] |= isa
    tcum = np.concatenate([[0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(sg))])
    tcum = tcum + 1e-9 * np.arange(len(sg))          # strictly increasing
    t_surf = tcum[-1]
    g_at = lambda s: np.interp(s, sg, gl)
    aer_at = lambda s: np.interp(s, sg, aer.astype(float)) > 0.5
    n = N_PHOT
    sig = np.zeros(n)
    mu = np.full(n, mu0)                              # >0 downward
    phi = np.zeros(n)
    k = np.zeros(n, int)
    na = np.zeros(n, int)
    L = np.zeros((n, len(M_GRID)))
    alive = np.ones(n, bool)
    out_b, out_k, out_n, out_L = [], [], [], []

    def seg(s1, s2, m_):
        return np.abs(s2[:, None] ** (M_GRID + 1) - s1[:, None] ** (M_GRID + 1)) / np.abs(m_)[:, None]

    def emit(idx, weight):
        # path from the event at sig[idx] to TOA along the sensor direction
        s = sig[idx]
        tr = np.exp(-np.interp(s, sg, tcum) / muv)
        out_b.append(weight * tr)
        out_k.append(k[idx].copy())
        out_n.append(na[idx].copy())
        out_L.append(L[idx] + s[:, None] ** (M_GRID + 1) / muv)

    for _ in range(MAX_EV):
        idx = np.where(alive)[0]
        if not idx.size:
            break
        t0 = np.interp(sig[idx], sg, tcum)
        dt = -np.log(rng.random(idx.size)) * mu[idx]
        t1 = t0 + dt
        s_new = np.interp(np.clip(t1, 0, t_surf), tcum, sg)
        esc = t1 <= 0
        hit = t1 >= t_surf
        sca = ~esc & ~hit
        # escape upward: path to TOA
        if esc.any():
            ii = idx[esc]
            L[ii] += seg(sig[ii], np.zeros(ii.size), mu[ii])
            alive[ii] = False
        # surface hit: Lambertian, local estimate, new cosine-weighted upward direction
        if hit.any():
            ii = idx[hit]
            L[ii] += seg(sig[ii], np.ones(ii.size), mu[ii])
            sig[ii] = 1.0
            k[ii] += 1
            emit(ii, np.full(ii.size, 1.0 / np.pi))   # Lambertian radiance A/pi; A^k applied later
            mu[ii] = -np.sqrt(rng.random(ii.size))
            phi[ii] = 2 * np.pi * rng.random(ii.size)
        if sca.any():
            ii = idx[sca]
            L[ii] += seg(sig[ii], s_new[sca], mu[ii])
            sig[ii] = s_new[sca]
            isa = aer_at(sig[ii])
            na[ii] += isa
            g_here = g_at(sig[ii])
            # local estimate toward the sensor (direction -muv, azimuth 0)
            cth = (mu[ii] * -muv + np.sqrt(np.clip(1 - mu[ii] ** 2, 0, 1)) * np.sqrt(1 - muv ** 2) * np.cos(phi[ii]))
            emit(ii, hg_phase(g_here, cth) / (4 * np.pi * muv))
            # new direction
            ct = hg_sample(g_here, rng.random(ii.size))
            st_ = np.sqrt(np.clip(1 - ct ** 2, 0, 1))
            ps = 2 * np.pi * rng.random(ii.size)
            mu_old = mu[ii]
            so = np.sqrt(np.clip(1 - mu_old ** 2, 0, 1))
            mu_new = mu_old * ct + so * st_ * np.cos(ps)
            mu_new = np.clip(mu_new, -1, 1)
            mu_new = np.where(np.abs(mu_new) < 1e-3, np.sign(mu_new + 1e-12) * 1e-3, mu_new)
            mu[ii] = mu_new
            phi[ii] = 2 * np.pi * rng.random(ii.size)
    b = np.concatenate(out_b)
    return b, np.concatenate(out_k), np.concatenate(out_n), np.vstack(out_L)


def reflectance(contrib, tau_vert, m_ch, alb, omega):
    b, k, na, L = contrib
    j = np.clip(np.searchsorted(M_GRID, m_ch) - 1, 0, len(M_GRID) - 2)
    a = (m_ch - M_GRID[j]) / (M_GRID[j + 1] - M_GRID[j])
    Lch = L[:, j] * (1 - a) + L[:, j + 1] * a                 # contributions x channels
    w = b[:, None] * alb[None, :] ** k[:, None] * (omega ** na)[:, None]
    return np.sum(w * np.exp(-tau_vert[None, :] * Lch), axis=0)


# ── fits (agreed model, as src/spectral/cumulant_fit.py; reimplemented) ─────────

def _lin(cols, y):
    a = np.column_stack(cols)
    return np.linalg.lstsq(a, y, rcond=None)[0]


def fit_o2a(t, x, y):
    from scipy.optimize import least_squares
    c0 = _lin([np.ones_like(t), x, -t], y)
    p0 = [c0[0], c0[1], max(c0[2], 0.1), 0.01]
    fun = lambda p: p[0] + p[1] * x + np.log((1 - p[3]) * np.exp(-p[2] * t) + p[3]) - y
    r = least_squares(fun, p0, bounds=([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, 0.2]))
    fun2 = lambda p: p[0] + p[1] * x + np.log((1 - p[3]) * np.exp(-p[2] * t + 0.5 * p[4] * t ** 2) + p[3]) - y
    r2 = least_squares(fun2, list(r.x) + [0.0],
                       bounds=([-np.inf, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 0.2, np.inf]))
    return dict(o2a_c_am=r.x[0], o2a_a1_am=r.x[1], o2a_k1_am=r.x[2], o2a_f_am=r.x[3],
                o2a_k2diag_am=r2.x[4])


def fit_wco2(td, tw, x, y):
    one = np.ones_like(td)
    c = _lin([one, x, -td, -tw], y)
    cd = _lin([one, x, -td, -tw, 0.5 * (td + tw) ** 2], y)
    return dict(wco2_c_am=c[0], wco2_a1_am=c[1], wco2_k1_am=c[2], wco2_k1w_am=c[3], wco2_k2diag_am=cd[4])


def fit_sco2(td, tw, x, y, land):
    one = np.ones_like(td)
    t = td + tw
    cols = [one, x] + ([x ** 2] if land else []) + [-td, -tw, 0.5 * t ** 2]
    c = _lin(cols, y)
    k = 3 if land else 2
    return dict(sco2_c_am=c[0], sco2_a1_am=c[1], sco2_a2_am=c[2] if land else 0.0,
                sco2_k1_am=c[k], sco2_k1w_am=c[k + 1], sco2_k2diag_am=c[k + 2])


# ── one scene ───────────────────────────────────────────────────────────────────

def scene(args):
    i, g_co2, bank_row, seed = args
    rng = np.random.default_rng(seed)
    s = bank_row
    mu0 = np.cos(np.radians(s["sza"]))
    muv = np.cos(np.radians(s["vza"]))
    amf = 1 / mu0 + 1 / muv
    land = rng.random() < 0.6
    cloudy = rng.random() < 0.3
    tau_a = float(np.exp(rng.normal(np.log(0.08), 0.8)))
    ang = rng.uniform(0.2, 1.5)
    if rng.random() < 0.3:
        sb = rng.uniform(0.5, 0.85)
        st = sb - rng.uniform(0.1, 0.2)
    else:
        sb, st = 1.0, rng.uniform(0.75, 0.92)
    tau_c = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0)))) if cloudy else 0.0
    cb = rng.uniform(0.3, 0.9)
    ct = cb - rng.uniform(0.05, 0.15)
    if land:
        a_o = rng.uniform(0.05, 0.35)
        alb = {"o2a": a_o, "wco2": a_o * rng.uniform(0.8, 1.8)}
        alb["sco2"] = alb["wco2"] * rng.uniform(0.7, 1.2)
    else:
        a = rng.uniform(0.02, 0.06)
        alb = {"o2a": a, "wco2": a * rng.uniform(0.9, 1.1), "sco2": a * rng.uniform(0.9, 1.1)}
    omega_a = rng.uniform(0.88, 0.99)
    wl = {"o2a": 0.765, "wco2": 1.61, "sco2": 2.06}
    feats = dict(i=i, g_co2=g_co2, land=int(land), cloudy=int(cloudy), tau_a=tau_a, tau_c=tau_c,
                 inv_mu0=1 / mu0, inv_muv=1 / muv)
    ps = s["ps"]
    for b in BANDS:
        ta = tau_a * (wl[b] / 0.765) ** (-ang)
        layers = [(st, sb, ta, 0.7, True)] + ([(ct, cb, tau_c, 0.85, False)] if cloudy else [])
        contrib = run_mc(layers, mu0, muv, rng)
        tau = s[f"{b}_tau"].astype(float)
        fh = s[f"{b}_fh"].astype(float)
        tw = tau * fh
        td = tau - tw
        sr = np.clip(np.nan_to_num(s[f"{b}_pabs"] / ps, nan=0.5), 0.4, 0.93)
        m_ch = np.clip((2 * sr - 1) / (1 - sr), M_GRID[0], M_GRID[-1])
        gscale = g_co2 if b != "o2a" else 1.0
        tau_true_vert = (td * gscale + tw) / amf
        slope = rng.uniform(-0.05, 0.05)
        alb_ch = np.clip(alb[b] * (1 + slope * X), 1e-3, 1)
        R = reflectance(contrib, tau_true_vert, m_ch, alb_ch, omega_a)
        y_clean = np.log(np.maximum(R, 1e-30))
        noise = rng.normal(size=R.size) / (SNR[b] * rng.uniform(0.6, 1.4)) \
            / np.sqrt(np.maximum(R / R.max(), 1e-4)) / np.sqrt(CH_STEP)
        for tag, y in (("", y_clean), ("n_", y_clean + noise)):
            if b == "o2a":
                r = fit_o2a(tau, X, y)
            elif b == "wco2":
                r = fit_wco2(td, tw, X, y)
            else:
                r = fit_sco2(td, tw, X, y, land)
            feats.update({tag + k: v for k, v in r.items()})
        # true path factor of the band (mean over the dry-tau-weighted spectrum, diagnostic)
        feats[f"{b}_true_L"] = float(np.sum(contrib[0] * contrib[3][:, 3]) / np.sum(contrib[0]) / amf)
    return feats


def path_feature_names(tag=""):
    f = [f"{tag}o2a_k1_am"] + [f"{tag}shape_{b}" for b in BANDS] + [f"{tag}exp_int_{b}" for b in BANDS] \
        + ["inv_mu0", "inv_muv", f"{tag}o2a_f_am", f"{tag}o2a_a1_am", f"{tag}wco2_a1_am",
           f"{tag}sco2_a1_am", f"{tag}sco2_a2_am"]
    return f


def add_derived(df, tag=""):
    for b in BANDS:
        df[f"{tag}shape_{b}"] = df[f"{tag}{b}_k2diag_am"] / df[f"{tag}{b}_k1_am"] ** 2
        df[f"{tag}exp_int_{b}"] = np.exp(df[f"{tag}{b}_c_am"])
    return df


def evaluate(train, test):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for tag, noise in (("", "noise-free"), ("n_", "noisy")):
        feats = path_feature_names(tag)
        for band in ("sco2", "wco2"):
            tgt = f"{tag}{band}_k1_am"
            for mname, model in (("E0 (k1/mean)", None),
                                 ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                                 ("hgb", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                                       min_samples_leaf=20, random_state=0))):
                if model is None:
                    pred = np.full(len(test), train[tgt].mean())
                else:
                    model.fit(train[feats], train[tgt])
                    pred = model.predict(test[feats])
                ghat = test[tgt] / pred
                err = (ghat - test["g_co2"]).to_numpy() * 400.0
                for sub, m in (("all", np.ones(len(test), bool)),
                               ("clear (no cloud, tau_a<0.1)", ((test.cloudy == 0) & (test.tau_a < 0.1)).to_numpy()),
                               ("cloudy", (test.cloudy == 1).to_numpy()),
                               ("land", (test.land == 1).to_numpy()), ("ocean", (test.land == 0).to_numpy())):
                    e = err[m & np.isfinite(err)]
                    rows.append(dict(noise=noise, band=band, model=mname, subset=sub, n=len(e),
                                     rsd_ppm=1.4826 * np.median(np.abs(e - np.median(e))),
                                     sd_ppm=float(np.std(e)), bias_ppm=float(np.mean(e)),
                                     corr_ghat_g=float(np.corrcoef(ghat[m], test["g_co2"][m])[0, 1])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bank = load_tau_bank()
    print(f"tau bank: {len(bank)} soundings", flush=True)
    rng = np.random.default_rng(args.seed)
    n = args.n_train + args.n_test
    g = np.where(np.arange(n) < args.n_train, 1.0, rng.uniform(0.99, 1.01, n))
    jobs = [(i, g[i], bank[rng.integers(len(bank))], int(rng.integers(2 ** 31))) for i in range(n)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(scene, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    df = add_derived(add_derived(df, ""), "n_")
    df.to_csv(OUT / "slab_mc_scenes.csv", index=False)
    train, test = df[df.i < args.n_train], df[df.i >= args.n_train]
    res = evaluate(train, test)
    res.to_csv(OUT / "slab_mc_floor_full.csv", index=False)
    piv = res[res.subset.isin(["all", "clear (no cloud, tau_a<0.1)", "cloudy", "land", "ocean"])]
    piv = piv.pivot_table(index=["noise", "band", "model"], columns="subset", values="rsd_ppm").round(2).reset_index()
    piv.to_csv(OUT / "slab_mc_floor.csv", index=False)
    print(piv.to_string(index=False))
    print("k1 sanity (train, clear):", train[(train.cloudy == 0) & (train.tau_a < 0.1)][["o2a_k1_am", "wco2_k1_am", "sco2_k1_am", "o2a_f_am"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
