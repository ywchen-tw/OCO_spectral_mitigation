"""Decompose the far-clear structured residual of the unit-weight cumulant fit by channel attribute.

Structured residual r: residual of the *unconstrained* unit-weight order-N fit
(orthogonal per sounding to [1, τ, …, τ^N]). Every regressor column is
residualized per sounding against the same polynomial (Frisch–Waugh–Lovell), so
an attribute is credited only with what the τ polynomial cannot represent.
All regressions are evaluated from per-sounding Gram matrices:

- pooled model: one coefficient per column for all far-clear soundings
  (linear + piecewise-linear "hat" bases → fixed nonlinear relationships);
- per-sounding model: one coefficient per sounding per linear column
  (per-sounding amplitude, e.g. a Doppler shift or an H2O scaling error).

R²_alone(g) and partial R²(g) = (RSS_without g − RSS_all)/RSS_without g are
reported per band, order and model; see PLAN.md "Residual decomposition".

Usage:
    python idea_test/fit_order_study/resid_decompose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS, load_orbit  # noqa: E402
from resid_attrs import FIT_DETAILS, FP_TAU, L1B, ORBIT  # noqa: E402
from run_study import FAR_KM, md_table  # noqa: E402
from variants import _solve_bounded, design_matrix  # noqa: E402

OUT = HERE / "results"
ATTR_FILE = OUT / f"resid_attrs_{ORBIT}.npz"
ORDERS = (3, 2)
GROUPS = ("VERT", "H2O", "SOLAR", "CONT", "CHAN", "DISP", "ILSW", "TEMP", "TAU")
CATEGORY = {"VERT": "path", "H2O": "path", "SOLAR": "solar", "CONT": "continuum",
            "CHAN": "instrument", "DISP": "instrument", "ILSW": "instrument",
            "TEMP": "spectroscopy/met", "TAU": "truncation"}
CATEGORIES = ("path", "solar", "continuum", "instrument", "spectroscopy/met", "truncation")
SOLAR_HALF = 8               # running max over ±8 channels
CONT_Q = 5.0                 # continuum: τ below the band's 5th percentile (per sounding)
N_HAT = 8                    # knots of attribute hat bases (pooled quantiles)
N_HAT_CHAN = 13
N_TAU_BINS = 6
BIN_ATTRS = ("eta", "f_h2o", "sol", "dopp", "dT")   # attributes with τ-binned slopes
SVD_MODES = 4


# --------------------------------------------------------------------------- inputs


def load_inputs(attr_file=ATTR_FILE):
    """Orbit spectra, attributes [N, C-2] (NaN rows for soundings not in the file), v_inst."""
    orb = load_orbit(str(L1B), str(FP_TAU), str(FIT_DETAILS))
    n = len(orb.sounding_id)
    z = np.load(attr_file)
    jj = z["j"]
    if not np.array_equal(z["sounding_id"], orb.sounding_id[jj]):
        raise ValueError("attribute file and fp_tau sounding order differ")
    with h5py.File(FP_TAU, "r") as f:
        toa = np.stack([f[f"{b}_toa_sol_output"][...] for b in BANDS])
    env = maximum_filter1d(toa, size=2 * SOLAR_HALF + 1, axis=2)
    sol = 1.0 - toa / env
    with h5py.File(FIT_DETAILS, "r") as f:
        extra = {k: f[k][...] for k in ("h2o_scale", "delT", "dp", "aod_total", "airmass", "tcwv")}
    attrs = {}
    for ib, b in enumerate(BANDS):
        a = {}
        for k in ("eta", "eta_jac", "f_h2o", "f_ray", "f_bl", "dtau_dpix", "dtau_dlnw", "dtau_dT",
                  "tau_rep"):
            full = np.full((n, 1014), np.nan, dtype=np.float32)
            full[jj] = z[f"{b}_{k}"][:, 1:-1]
            a[k] = full
        a["sol"] = sol[ib][:, 1:-1].astype(np.float32)
        attrs[b] = a
    v_inst = np.full(n, np.nan)
    v_inst[jj] = z["v_inst"].astype(float) / 1e3            # km/s
    return orb, attrs, v_inst, extra, str(z["validation"]), jj


def hat_basis(x, knots):
    """Piecewise-linear interpolation basis at the knots, first function dropped (sum = 1)."""
    eye = np.eye(len(knots))
    return np.stack([np.interp(x, knots, eye[k]) for k in range(1, len(knots))], axis=1)


class Columns:
    """Regressor columns for one band; knots fixed from the pooled far-clear distribution."""

    def __init__(self, attrs_b, tau_b, far_idx):
        def q(x, n):
            v = np.asarray(x[far_idx], float).ravel()
            v = v[np.isfinite(v)]
            return np.unique(np.quantile(v, np.linspace(0.005, 0.995, n)))
        self.k_eta = q(attrs_b["eta"], N_HAT)
        self.k_f = q(attrs_b["f_h2o"], N_HAT)
        self.k_sol = q(attrs_b["sol"], N_HAT)
        self.k_lntau = q(np.log(np.clip(tau_b, 1e-6, None)), 10)
        self.k_tau_bins = q(tau_b, N_TAU_BINS + 1)
        nch = tau_b.shape[1]
        self.k_chan = np.linspace(0, nch - 1, N_HAT_CHAN)
        self.names, self.group, self.pooled, self.ps = None, None, None, None

    def build(self, a, t, cidx, v):
        """a: attribute dict for one sounding (valid channels), t: τ, cidx: channel index, v: km/s."""
        cols = []                                         # (name, group, pooled, per-sounding, x)

        def add(name, grp, pooled, ps, x):
            cols.append((name, grp, pooled, ps, np.asarray(x, float)))
        eta, f, sol = a["eta"], a["f_h2o"], a["sol"]
        g, w, dT = a["dtau_dpix"], a["dtau_dlnw"], a["dtau_dT"]
        add("eta", "VERT", True, True, eta)
        add("tau_eta", "VERT", True, True, t * eta)
        add("tau2_eta", "VERT", True, False, t**2 * eta)
        for k, col in enumerate(hat_basis(eta, self.k_eta).T):
            add(f"hat_eta{k}", "VERT", True, False, col)
        add("f_bl", "VERT", True, False, a["f_bl"])
        add("tau_f_bl", "VERT", True, False, t * a["f_bl"])
        add("f_h2o", "H2O", True, True, f)
        add("tau_f_h2o", "H2O", True, True, t * f)
        add("tau2_f_h2o", "H2O", True, False, t**2 * f)
        for k, col in enumerate(hat_basis(f, self.k_f).T):
            add(f"hat_f{k}", "H2O", True, False, col)
        add("sol", "SOLAR", True, True, sol)
        add("tau_sol", "SOLAR", True, True, t * sol)
        for k, col in enumerate(hat_basis(sol, self.k_sol).T):
            add(f"hat_sol{k}", "SOLAR", True, False, col)
        cont = (t < np.percentile(t, CONT_Q)).astype(float)
        add("cont", "CONT", True, True, cont)
        add("tau_cont", "CONT", True, False, t * cont)
        add("f_ray", "CONT", True, True, a["f_ray"])
        add("tau_f_ray", "CONT", True, False, t * a["f_ray"])
        for k, col in enumerate(hat_basis(cidx, self.k_chan).T):
            add(f"hat_chan{k}", "CHAN", True, False, col)
        xc = 2 * cidx / 1013.0 - 1
        for k, col in enumerate((xc, 1.5 * xc**2 - 0.5, 2.5 * xc**3 - 1.5 * xc), start=1):
            add(f"chan_P{k}", "CHAN", False, True, col)
        add("dtau_dpix", "DISP", True, True, g)
        add("tau_dtau_dpix", "DISP", True, True, t * g)
        add("dopp", "DISP", True, False, v * g)
        add("tau_dopp", "DISP", True, False, v * t * g)
        add("dtau_dlnw", "ILSW", True, True, w)
        add("tau_dtau_dlnw", "ILSW", True, True, t * w)
        add("dtau_dT", "TEMP", True, True, dT)
        add("tau_dtau_dT", "TEMP", True, True, t * dT)
        lt = np.log(np.clip(t, 1e-6, None))
        for k, col in enumerate(hat_basis(lt, self.k_lntau).T):
            add(f"hat_lntau{k}", "TAU", True, False, col)
        for k in (1, 2, 3):
            add(f"lntau{k}", "TAU", False, True, lt**k)
        # diagnostic columns: attribute × τ-bin indicator (alone regressions only)
        binv = {"eta": eta, "f_h2o": f, "sol": sol, "dopp": v * g, "dT": dT}
        kb = self.k_tau_bins
        ib = np.clip(np.searchsorted(kb, t, side="right") - 1, 0, len(kb) - 2)
        for name in BIN_ATTRS:
            for k in range(len(kb) - 1):
                add(f"bin_{name}{k}", "BIN", False, False, binv[name] * (ib == k))
        if self.names is None:
            self.names = [c[0] for c in cols]
            self.group = np.array([c[1] for c in cols])
            self.pooled = np.array([c[2] for c in cols])
            self.ps = np.array([c[3] for c in cols])
        return np.column_stack([c[4] for c in cols])


# --------------------------------------------------------------------------- regression helpers


def explained(gram, b, sel):
    """Explained sum of squares b·β for the column subset ``sel`` (scaled pseudo-inverse)."""
    if not np.any(sel):
        return 0.0, np.zeros(0)
    g = gram[np.ix_(sel, sel)]
    bb = b[sel]
    s = np.sqrt(np.clip(np.diag(g), 0, None))
    s[s == 0] = 1.0
    beta_s = np.linalg.lstsq(g / s[:, None] / s[None, :], bb / s, rcond=1e-10)[0]
    return float(bb / s @ beta_s), beta_s / s


def decompose_band(orb, attrs_b, ib, order, far_idx, v_inst):
    tau_b = orb.tau[ib][:, 1:-1].astype(float)
    lnt_b = orb.ln_t[ib][:, 1:-1].astype(float)
    cols = Columns(attrs_b, tau_b, far_idx)
    nch = tau_b.shape[1]
    cidx_all = np.arange(nch, dtype=float)
    grams, bs, rrs, bs_b, rrs_b, ns = [], [], [], [], [], []
    r_u = np.full((len(far_idx), nch), np.nan)
    r_b = np.full((len(far_idx), nch), np.nan)
    k2_free = np.full(len(far_idx), np.nan)
    keys = ("eta", "f_h2o", "sol", "f_ray", "f_bl", "dtau_dpix", "dtau_dlnw", "dtau_dT")
    for n, j in enumerate(far_idx):
        t, y = tau_b[j], lnt_b[j]
        m = np.isfinite(t) & np.isfinite(y)
        for k in keys:
            m &= np.isfinite(attrs_b[k][j])
        t, y = t[m], y[m]
        a = {k: attrs_b[k][j][m].astype(float) for k in keys}
        A = design_matrix(t, order)
        qm, _ = np.linalg.qr(A)
        ru = y - qm @ (qm.T @ y)
        coef_u = np.linalg.lstsq(A, y, rcond=None)[0]
        k2_free[n] = coef_u[2] if order >= 2 else np.nan
        coef_b, _ = _solve_bounded(A, y, order, None)
        rb = y - A @ coef_b
        x = cols.build(a, t, cidx_all[m], v_inst[j])
        xp = x - qm @ (qm.T @ x)
        grams.append(xp.T @ xp)
        bs.append(xp.T @ ru)
        rrs.append(ru @ ru)
        bs_b.append(xp.T @ rb)
        rrs_b.append(rb @ rb)
        ns.append(len(t))
        r_u[n, m] = ru
        r_b[n, m] = rb
    return dict(cols=cols, gram=np.array(grams), b=np.array(bs), rr=np.array(rrs),
                b_b=np.array(bs_b), rr_b=np.array(rrs_b), n=np.array(ns), r_u=r_u, r_b=r_b,
                k2_free=k2_free)


def r2_tables(d, band, order):
    cols = d["cols"]
    grp, pooled, ps = cols.group, cols.pooled, cols.ps
    rows = []
    for resid_kind, bkey, rkey in (("unconstrained", "b", "rr"), ("bounded", "b_b", "rr_b")):
        b_all, rr_all = d[bkey], d[rkey]
        tss = rr_all.sum()
        g_pool, b_pool = d["gram"].sum(axis=0), b_all.sum(axis=0)

        def ess_pooled(sel):
            return explained(g_pool, b_pool, sel)[0]

        def ess_ps(sel):
            return sum(explained(g, b, sel)[0] for g, b in zip(d["gram"], b_all))
        for model, base, ess_fn in (("pooled", pooled, ess_pooled), ("per_sounding", ps, ess_ps)):
            full = ess_fn(base)
            chance = float(np.mean(base.sum() / d["n"])) if model == "per_sounding" else 0.0
            rows.append(dict(band=band, order=order, resid=resid_kind, model=model, unit="ALL",
                             kind="group", r2_alone=full / tss, partial_r2=np.nan, unique=np.nan,
                             n_cols=int(base.sum()), chance_r2=chance))
            units = [(g, "group", base & (grp == g)) for g in GROUPS]
            units += [(c, "category", base & np.isin(grp, [g for g in GROUPS if CATEGORY[g] == c]))
                      for c in CATEGORIES]
            for name, kind, sel in units:
                if not sel.any():
                    continue
                alone = ess_fn(sel)
                without = ess_fn(base & ~sel)
                rss_wo, rss_all = tss - without, tss - full
                rows.append(dict(band=band, order=order, resid=resid_kind, model=model, unit=name,
                                 kind=kind, r2_alone=alone / tss,
                                 partial_r2=(rss_wo - rss_all) / rss_wo, unique=(full - without) / tss,
                                 n_cols=int(sel.sum()),
                                 chance_r2=float(np.mean(sel.sum() / d["n"])) if model == "per_sounding" else 0.0))
    return rows


def tau_binned_slopes(d, band, order):
    cols = d["cols"]
    g_pool, b_pool = d["gram"].sum(axis=0), d["b"].sum(axis=0)
    kb = cols.k_tau_bins
    rows = []
    for name in BIN_ATTRS:
        sel = np.array([n.startswith(f"bin_{name}") for n in cols.names])
        _, beta = explained(g_pool, b_pool, sel)
        for k, bk in enumerate(beta):
            rows.append(dict(band=band, order=order, attr=name, tau_lo=kb[k], tau_hi=kb[k + 1],
                             tau_mid=np.sqrt(max(kb[k], 1e-4) * kb[k + 1]), slope=bk))
    return rows


def per_sounding_coefs(d, far_idx, orb, extra, v_inst, ib, order):
    """Alone per-sounding coefficients for key regressors, against L2 / geometry variables."""
    cols = d["cols"]
    names = cols.names
    k1 = orb.meta[f"{BANDS[ib]}_k1"][far_idx]          # production k1 (scale only)
    out = {}
    for key in ("dtau_dpix", "tau_f_h2o", "dtau_dT", "tau_eta", "eta"):
        sel = np.array([n == key for n in names])
        out[key] = np.array([explained(g, b, sel)[1][0] if np.any(g[np.ix_(sel, sel)]) else np.nan
                             for g, b in zip(d["gram"], d["b"])])
    df = pd.DataFrame(dict(j=far_idx, fp=orb.fp_number[far_idx], lat=orb.meta["lat"][far_idx],
                           v_inst=v_inst[far_idx], k1_prod=k1,
                           h2o_scale=extra["h2o_scale"][far_idx], delT=extra["delT"][far_idx],
                           airmass=extra["airmass"][far_idx], aod=extra["aod_total"][far_idx],
                           tcwv=extra["tcwv"][far_idx], k2_free=d["k2_free"],
                           **{f"beta_{k}": v for k, v in out.items()}))
    df.insert(0, "order", order)
    df.insert(0, "band", BANDS[ib])
    return df


def fixed_pattern(d, fp_far):
    """Share of Σr² carried by the per-channel mean (pooled over fp, and per fp)."""
    r = d["r_u"]
    tss = np.nansum(r**2)
    mu = np.nanmean(r, axis=0)
    res_pool = np.nansum((r - mu) ** 2)
    res_fp = 0.0
    mu_fp = np.full((8, r.shape[1]), np.nan)
    for ifp in range(8):
        rr = r[fp_far == ifp]
        mu_fp[ifp] = np.nanmean(rr, axis=0)
        res_fp += np.nansum((rr - mu_fp[ifp]) ** 2)
    return dict(fixed_share_pooled=1 - res_pool / tss, fixed_share_fp=1 - res_fp / tss), mu, mu_fp


def svd_modes(d, attrs_b, far_idx, extra, v_inst, orb, band, order):
    r = np.nan_to_num(d["r_u"])
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    var = s**2 / np.sum(s**2)
    chan_attr = {k: np.nanmean(attrs_b[k][far_idx], axis=0) for k in
                 ("eta", "f_h2o", "sol", "dtau_dpix", "dtau_dlnw", "dtau_dT", "f_ray", "tau_rep")}
    snd = dict(v_inst=v_inst[far_idx], lat=orb.meta["lat"][far_idx], fp=orb.fp_number[far_idx].astype(float),
               h2o_scale=extra["h2o_scale"][far_idx], delT=extra["delT"][far_idx],
               airmass=extra["airmass"][far_idx], aod=extra["aod_total"][far_idx])
    rows = []
    for k in range(SVD_MODES):
        rec = dict(band=band, order=order, mode=k + 1, var_frac=var[k])
        for name, x in chan_attr.items():
            ok = np.isfinite(x)
            rec[f"chan_corr_{name}"] = np.corrcoef(vt[k][ok], x[ok])[0, 1]
        # sign-insensitive: |corr| of the channel pattern with |attribute| as well
        for name, x in snd.items():
            ok = np.isfinite(x)
            rec[f"amp_corr_{name}"] = np.corrcoef(u[ok, k], x[ok])[0, 1]
        rows.append(rec)
    return rows, u[:, :SVD_MODES] * s[:SVD_MODES], vt[:SVD_MODES]


def streak_auc(d, attrs_b, orb, ib, far_idx, v_inst, max_snd=400):
    """Per sounding: upper vs lower branch (|r| > 1.5 robust SD, τ above band median); AUC per attribute."""
    tau_b = orb.tau[ib][:, 1:-1].astype(float)
    keys = ("eta", "f_h2o", "sol", "dtau_dpix", "dtau_dlnw", "dtau_dT", "f_bl")
    recs = []
    for n, j in enumerate(far_idx[:max_snd]):
        r = d["r_u"][n]
        t = tau_b[j]
        ok = np.isfinite(r) & (t > np.nanmedian(t))
        if ok.sum() < 50:
            continue
        rr = r[ok]
        sd = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        up, lo = rr > 1.5 * sd, rr < -1.5 * sd
        if up.sum() < 10 or lo.sum() < 10:
            continue
        rec = dict(j=j, v_inst=v_inst[j], n_up=int(up.sum()), n_lo=int(lo.sum()))
        for k in keys:
            x = attrs_b[k][j][ok].astype(float)
            xu, xl = x[up], x[lo]
            ranks = pd.Series(np.r_[xu, xl]).rank().values
            auc = (ranks[:len(xu)].sum() - len(xu) * (len(xu) + 1) / 2) / (len(xu) * len(xl))
            rec[f"auc_{k}"] = auc
        recs.append(rec)
    df = pd.DataFrame(recs)
    out = dict(band=BANDS[ib], order=None, n_soundings=len(df))
    for k in keys:
        a = df[f"auc_{k}"]
        out[f"median_abs_auc_{k}"] = float(np.median(np.abs(a - 0.5)) + 0.5)
        ok = np.isfinite(df.v_inst) & np.isfinite(a)
        out[f"corr_auc_v_{k}"] = float(np.corrcoef(a[ok], df.v_inst[ok])[0, 1])
    return out, df


# --------------------------------------------------------------------------- figures


def figures(store, orb, attrs, v_inst, far_idx, r2, slopes, wl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. partial R² bars
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for io, order in enumerate(ORDERS):
        for ib, band in enumerate(BANDS):
            ax = axes[io, ib]
            sub = r2[(r2.band == band) & (r2.order == order) & (r2.resid == "unconstrained")
                     & (r2.kind == "group")]
            x = np.arange(len(GROUPS))
            for k, (model, c) in enumerate((("pooled", "C0"), ("per_sounding", "C1"))):
                s = sub[sub.model == model].set_index("unit").reindex(GROUPS)
                ax.bar(x + (k - 0.5) * 0.38, s.partial_r2, 0.38, color=c, label=f"{model} partial")
                ax.plot(x + (k - 0.5) * 0.38, s.r2_alone, "k_", ms=10)
            ax.axhline(0.5, color="r", lw=0.8, ls="--")
            ax.axhline(0.2, color="0.5", lw=0.8, ls=":")
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=60, fontsize=8)
            ax.set_title(f"{band}, order {order}")
            if ib == 0:
                ax.set_ylabel("R² (bars: partial; ticks: alone)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "resid_partial_r2.png", dpi=130)
    plt.close(fig)

    # 2. residual vs τ coloured by η⊥-proxy and by Doppler-scaled dispersion signature
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(far_idx), size=min(60, len(far_idx)), replace=False)
    for ib, band in enumerate(BANDS):
        d = store[(band, 3)]
        tau = orb.tau[ib][far_idx][:, 1:-1][pick].ravel()
        r = d["r_u"][pick].ravel()
        eta = attrs[band]["eta"][far_idx][pick].ravel()
        dop = (attrs[band]["dtau_dpix"][far_idx][pick] * v_inst[far_idx][pick][:, None]).ravel()
        ok = np.isfinite(r) & np.isfinite(tau) & np.isfinite(eta)
        o = rng.permutation(np.flatnonzero(ok))
        for row, (c, lab, cmap, lim) in enumerate(((eta, "η (km)", "viridis", np.nanpercentile(eta[ok], [2, 98])),
                                                   (dop, "v_inst·∂τ/∂pix (km/s)", "coolwarm",
                                                    np.array([-1, 1]) * np.nanpercentile(np.abs(dop[ok]), 98)))):
            ax = axes[row, ib]
            sc = ax.scatter(tau[o], r[o], c=c[o], s=1, cmap=cmap, vmin=lim[0], vmax=lim[1], rasterized=True)
            fig.colorbar(sc, ax=ax, label=lab)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("τ")
            ax.set_ylabel("order-3 residual (unconstrained)")
            ax.set_title(f"{band} (60 far-clear soundings)")
    fig.tight_layout()
    fig.savefig(OUT / "resid_vs_tau_attrs.png", dpi=120)
    plt.close(fig)

    # 3. mean residual per channel vs wavelength with attribute overlays (fp 0 grid)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ib, band in enumerate(BANDS):
        ax = axes[ib]
        d = store[(band, 3)]
        mu = np.nanmean(d["r_u"], axis=0)
        sd = np.nanstd(d["r_u"], axis=0)
        x = wl[ib, 0, 1:-1] * 1e3
        ax.plot(x, mu, "k-", lw=0.7, label="mean residual (fixed pattern)")
        ax.fill_between(x, mu - sd, mu + sd, color="0.8", label="± SD across soundings")
        a2 = ax.twinx()
        for k, c, lab in (("eta", "C0", "η (km)"), ("f_h2o", "C2", "f_H2O×10"), ("sol", "C3", "solar depth×10")):
            val = np.nanmean(attrs[band][k][far_idx], axis=0) * (10 if k != "eta" else 1)
            a2.plot(x, val, color=c, lw=0.6, alpha=0.7, label=lab)
        a2.legend(fontsize=7, loc="upper right")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"{band}: order-3 residual per channel, far-clear")
        ax.set_xlabel("wavelength (nm)")
    fig.tight_layout()
    fig.savefig(OUT / "resid_mean_channel.png", dpi=120)
    plt.close(fig)

    # 4. streaks: one sounding per band at large |v_inst|, coloured by ∂τ/∂pix; mode-1 amplitude vs v
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ib, band in enumerate(BANDS):
        d = store[(band, 3)]
        n = int(np.nanargmax(np.abs(v_inst[far_idx])))
        j = far_idx[n]
        t = orb.tau[ib][j, 1:-1]
        r = d["r_u"][n]
        g = attrs[band]["dtau_dpix"][j]
        lim = np.nanpercentile(np.abs(g), 98)
        sc = axes[0, ib].scatter(t, r, c=g, s=3, cmap="coolwarm", vmin=-lim, vmax=lim)
        fig.colorbar(sc, ax=axes[0, ib], label="∂τ/∂pix")
        axes[0, ib].set_title(f"{band}: sounding {orb.sounding_id[j]}, v_inst {v_inst[j]:+.2f} km/s")
        axes[0, ib].set_xlabel("τ")
        axes[0, ib].set_ylabel("residual")
        amp = d["svd_amp"][:, 0]
        axes[1, ib].scatter(v_inst[far_idx], amp, s=3, c=orb.fp_number[far_idx], cmap="tab10")
        axes[1, ib].set_xlabel("v_inst (km/s)")
        axes[1, ib].set_ylabel("SVD mode-1 amplitude")
        axes[1, ib].set_title(f"{band}: mode 1 ({100 * d['svd_var'][0]:.0f} % of Σr²)")
    fig.tight_layout()
    fig.savefig(OUT / "resid_streaks.png", dpi=120)
    plt.close(fig)

    # 5. τ-binned pooled slopes
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ib, band in enumerate(BANDS):
        ax = axes[ib]
        s = slopes[(slopes.band == band) & (slopes.order == 3)]
        for name in BIN_ATTRS:
            ss = s[s.attr == name]
            norm = np.nanmax(np.abs(ss.slope)) or 1.0
            ax.plot(ss.tau_mid, ss.slope / norm, "o-", label=name)
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{band}: pooled slope per τ bin (normalized)")
        ax.set_xlabel("τ")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "resid_tau_binned_slopes.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default=str(ATTR_FILE))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--tau", default=None, help="npz (j, tau [n, 3, 1016]) overriding τ, e.g. Doppler-shifted")
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    orb, attrs, v_inst, extra, validation, jj = load_inputs(args.attrs)
    if args.tau:
        zt = np.load(args.tau)
        orb.tau[:, zt["j"], :] = np.moveaxis(zt["tau"], 1, 0)
        has_tau = np.zeros(len(orb.sounding_id), bool)
        has_tau[zt["j"]] = True
        validation += f"\nτ overridden from {args.tau} for {len(zt['j'])} soundings"
    else:
        has_tau = np.ones(len(orb.sounding_id), bool)
    cld, qf = orb.meta["cld_dist_km"], orb.meta["xco2_qf"]
    has = np.zeros(len(cld), bool)
    has[jj] = True
    far_idx = np.flatnonzero((cld > FAR_KM) & (qf == 0) & has & has_tau)
    fp_far = orb.fp_number[far_idx]
    print(f"far-clear soundings: {len(far_idx)}")
    from run_weighting import wavelengths
    wl = wavelengths(str(L1B))

    r2_rows, slope_rows, svd_rows, fixed_rows, auc_rows, coef_dfs = [], [], [], [], [], []
    store = {}
    for ib, band in enumerate(BANDS):
        for order in ORDERS:
            d = decompose_band(orb, attrs[band], ib, order, far_idx, v_inst)
            r2_rows += r2_tables(d, band, order)
            slope_rows += tau_binned_slopes(d, band, order)
            fx, mu, mu_fp = fixed_pattern(d, fp_far)
            # how much of the fixed pattern each pooled group explains (channel means of predictions)
            cols = d["cols"]
            g_pool, b_pool = d["gram"].sum(axis=0), d["b"].sum(axis=0)
            fx.update(band=band, order=order,
                      k2_free_neg_frac=float(np.mean(d["k2_free"] < 0)),
                      rms_unconstrained=float(np.sqrt(np.nanmean(d["r_u"] ** 2))),
                      rms_bounded=float(np.sqrt(np.nanmean(d["r_b"] ** 2))))
            fixed_rows.append(fx)
            rows, amp, vt = svd_modes(d, attrs[band], far_idx, extra, v_inst, orb, band, order)
            svd_rows += rows
            d["svd_amp"], d["svd_var"] = amp, [r["var_frac"] for r in rows]
            coef_dfs.append(per_sounding_coefs(d, far_idx, orb, extra, v_inst, ib, order))
            if order == 3:
                a, _ = streak_auc(d, attrs[band], orb, ib, far_idx, v_inst)
                a["order"] = order
                auc_rows.append(a)
            store[(band, order)] = d
            sub = pd.DataFrame(r2_rows)
            sub = sub[(sub.band == band) & (sub.order == order) & (sub.resid == "unconstrained")
                      & (sub.kind == "group")]
            print(band, order, "\n", sub[["model", "unit", "r2_alone", "partial_r2"]].round(3).to_string())
            del cols, g_pool, b_pool, mu, mu_fp

    r2 = pd.DataFrame(r2_rows)
    slopes = pd.DataFrame(slope_rows)
    svd = pd.DataFrame(svd_rows)
    fixed = pd.DataFrame(fixed_rows)
    auc = pd.DataFrame(auc_rows)
    coefs = pd.concat(coef_dfs, ignore_index=True)
    r2.to_csv(OUT / f"resid_decomp_{ORBIT}.csv", index=False)
    slopes.to_csv(OUT / f"resid_tau_slopes_{ORBIT}.csv", index=False)
    svd.to_csv(OUT / f"resid_svd_{ORBIT}.csv", index=False)
    fixed.to_csv(OUT / f"resid_fixed_pattern_{ORBIT}.csv", index=False)
    auc.to_csv(OUT / f"resid_streak_auc_{ORBIT}.csv", index=False)
    coefs.to_parquet(OUT / f"resid_ps_coefs_{ORBIT}.parquet", index=False)
    figures(store, orb, attrs, v_inst, far_idx, r2, slopes, wl)
    write_summary(r2, slopes, svd, fixed, auc, coefs, validation)


def _coef_stats(coefs):
    """Per-sounding alone coefficients vs physical drivers (expected slopes in the docstring)."""
    rows = []
    for (band, order), d in coefs.groupby(["band", "order"]):
        k1 = d.k1_prod.values
        delta = -d.beta_dtau_dpix.values / k1                  # implied shift (pixels)
        ok = np.isfinite(delta) & np.isfinite(d.v_inst)
        sl, ic = np.polyfit(d.v_inst[ok], delta[ok], 1)
        rec = dict(band=band, order=order,
                   corr_shift_v=np.corrcoef(d.v_inst[ok], delta[ok])[0, 1],
                   shift_per_kms=sl, shift_at_v0=ic, shift_rms=np.std(delta[ok]))
        s = -d.beta_tau_f_h2o.values / k1                       # implied H2O scale − 1
        ok = np.isfinite(s) & np.isfinite(d.h2o_scale)
        rec["corr_h2o_vs_l2scale"] = np.corrcoef(s[ok], d.h2o_scale[ok] - 1)[0, 1]
        rec["h2o_scale_implied_median"] = float(np.median(1 + s[ok]))
        dt = -d.beta_dtau_dT.values / k1                         # implied ΔT (K)
        ok = np.isfinite(dt) & np.isfinite(d.delT)
        rec["corr_dT_vs_l2delT"] = np.corrcoef(dt[ok], d.delT[ok])[0, 1]
        rec["dT_implied_median"] = float(np.median(dt[ok]))
        b = d.beta_tau_eta.values
        for var in ("airmass", "aod", "lat", "v_inst"):
            ok = np.isfinite(b) & np.isfinite(d[var])
            rec[f"corr_beta_tau_eta_{var}"] = np.corrcoef(b[ok], d[var][ok])[0, 1]
        rows.append(rec)
    return pd.DataFrame(rows)


def write_summary(r2, slopes, svd, fixed, auc, coefs, validation):
    parts = [f"# Residual decomposition — orbit {ORBIT}\n",
             "Far-clear (cld_dist > 20 km, QF0). Structured residual = unconstrained unit-weight "
             "order-N residual; regressors residualized per sounding against the τ polynomial. "
             "Partial R² = (RSS_without − RSS_all)/RSS_without. See PLAN.md.\n",
             "## Attribute validation (reproduction of stored τ)\n", "```", validation, "```\n"]
    u = r2[(r2.resid == "unconstrained")]
    for kind in ("group", "category"):
        for model in ("pooled", "per_sounding"):
            s = u[(u.kind == kind) & (u.model == model)]
            piv_a = s.pivot_table(index="unit", columns=["band", "order"], values="r2_alone")
            piv_p = s.pivot_table(index="unit", columns=["band", "order"], values="partial_r2")
            parts.append(f"\n## {kind} R², {model} model (cells: alone / partial)\n")
            colsn = [f"{b} o{o}" for b in BANDS for o in ORDERS]
            lines = ["| unit | " + " | ".join(colsn) + " |", "|" + "---|" * (len(colsn) + 1)]
            order_units = ["ALL", *GROUPS] if kind == "group" else list(CATEGORIES)
            for unit in order_units:
                if unit not in piv_a.index:
                    continue
                cells = []
                for b in BANDS:
                    for o in ORDERS:
                        a = piv_a.loc[unit].get((b, o), np.nan)
                        p = piv_p.loc[unit].get((b, o), np.nan) if unit in piv_p.index else np.nan
                        cells.append(f"{a:.3f} / {p:.3f}" if np.isfinite(p) else f"{a:.3f}")
                lines.append(f"| {unit} | " + " | ".join(cells) + " |")
            parts.append("\n".join(lines))
            if model == "per_sounding":
                ch = s[s.unit == "ALL"][["band", "order", "chance_r2"]]
                parts.append("\nchance level (p/n) of the full per-sounding model: "
                             + ", ".join(f"{r.band} o{r.order} {r.chance_r2:.3f}" for r in ch.itertuples()))
    b = r2[(r2.resid == "bounded") & (r2.kind == "group") & (r2.unit == "ALL")]
    parts.append("\n## Bounded (k1, k2 ≥ 0) residual, R² of all groups\n")
    parts.append(md_table(b[["band", "order", "model", "r2_alone"]], ["band", "order", "model", "r2_alone"]))
    parts.append("\n## Fixed channel pattern and rms\n")
    parts.append(md_table(fixed, list(fixed.columns)))
    parts.append("\n## τ-binned pooled slopes (alone, order 3)\n")
    s3 = slopes[slopes.order == 3]
    parts.append(md_table(s3, ["band", "attr", "tau_lo", "tau_hi", "slope"]))
    parts.append("\n## SVD modes of the residual matrix (soundings × channels)\n")
    keep = ["band", "order", "mode", "var_frac"] + [c for c in svd.columns if c.startswith(("chan_corr", "amp_corr"))]
    parts.append(md_table(svd[keep], keep, fmt="{:.2f}"))
    parts.append("\n## Streak branches (τ > median, |r| > 1.5 robust SD): AUC upper vs lower\n")
    parts.append(md_table(auc, list(auc.columns), fmt="{:.2f}"))
    cs = _coef_stats(coefs)
    cs.to_csv(OUT / f"resid_ps_coef_stats_{ORBIT}.csv", index=False)
    parts.append("\n## Per-sounding coefficients vs physical drivers\n")
    parts.append("shift = −β(∂τ/∂pix)/k1 (pixels; a Doppler-uncorrected grid predicts ±0.17 px per km/s); "
                 "H2O: −β(τ·f)/k1 vs L2 h2o_scale − 1; T: −β(∂τ/∂T)/k1 vs L2 delT.\n")
    parts.append(md_table(cs, list(cs.columns)))
    (OUT / "resid_summary.md").write_text("\n".join(parts) + "\n")
    print((OUT / "resid_summary.md").read_text())


if __name__ == "__main__":
    main()
