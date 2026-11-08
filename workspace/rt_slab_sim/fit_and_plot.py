"""
Stage 4: refit the slab radiances with an OCO-2 spectral estimator and draw
the quick-look (pre-Fig.-G1) figure.

Per column x, per surface, per solver, T = pi * radiance / (mu0 * toa) is
fitted with one of three estimators (--estimator):

  am     (default) the production O2A agreed model, fit schema
         doppler_am_t2_v2: src/spectral/cumulant_fit.fit_am_o2a,
         ln T = c + a1*x + ln[(1 - f) exp(-k1 tau) + f], k1 >= 0,
         0 <= f <= 0.2, k2 fixed 0, plus the free-k2 diagnostic k2diag.
         Inputs as in cumulant_fit._doppler_fit_chunk: raw ln T (no
         smoothing), total slant tau (O2 + H2O; O2A is not split into dry
         and H2O parts), channels with T > 1 or non-finite ln T / tau
         dropped, sorted by tau, >= _AM_MIN_CHAN channels.  The production
         first/last-channel cut is a detector-edge rule with no slab
         counterpart, so all 33 monochromatic wavelengths enter.  The
         spectral coordinate x (production: (j - 507.5)/507.5 on the 1016
         detector channels, wavelength increasing with j) becomes the
         normalised wavelength position over the simulated window,
         x = 2 (lambda - lambda_min)/(lambda_max - lambda_min) - 1 with
         lambda_min/max the extremes of the 33 wavelengths (759.45 and
         784.00 nm), fixed for all columns (spectral_x).  -> slab_fit_am.h5
  t2s    test model (2026-11-02), spectral.cumulant_fit.fit_t2s, the
         two-tau order-2 fit with slope, OLS without bounds or floor:
         ln T = c + a1*x - k1*tau_dry - k1w*tau_h2o + 0.5*k2*tau^2,
         tau = tau_dry + tau_h2o.  Same T, ln T, channel mask and x as am;
         tau_h2o = frac_h2o * tau from slab_od_h2o.h5 (gas_od.py
         --h2o-split; the H2O share of the slab OD is <= 5.4 % and only
         at near-continuum wavelengths, so k1w is barely constrained).
         -> slab_fit_t2s.h5
  o2x    closure (i) of the paper (2026-11-03), the second-order fit with a
         linear continuum term used for the CO2 bands and TEMPO, applied to
         the slab O2A spectra: unconstrained least squares on [1, x, -tau,
         0.5 tau^2], ln T = c + a1*x - k1*tau + 0.5*k2*tau^2, total slant tau
         (no dry/H2O split), no bounds, no floor.  Same T, ln T, channel mask
         and x as am.  --tau-max T additionally drops channels with
         tau > T (the continuum anchors stay) and writes the file with the
         tag o2x_tmax<T>.  -> slab_fit_o2x.h5 / slab_fit_o2x_tmax<T>.h5
  order7 the retired order-7 cumulant fit, fit_spectral_model(order 7,
         exact lstsq/BVLS, no Savitzky-Golay, k1/k2 >= 0); reproduces the
         pre-2026-11-02 slab_fit.h5.  The constant normalization is absorbed
         by the intercept.

The *_std datasets are the run-to-run spread (ddof 1) of the same fit applied
to each of the NRUN independent Monte Carlo runs; the am file also stores the
formal 1-sigma errors of the pooled-run fit as *_sig (cumulant_fit._cov_se).

Also reduces the MCARaTS Rad_mplen=3 path-length histograms to per-column
mean/std of total geometric path (the PPDF moments the fit is supposed to
encode -- closure panel).

Outputs:
  results/rt_slab_sim/slab_fit_am.h5          (am)
  results/rt_slab_sim/slab_fit_t2s.h5         (t2s)
  results/rt_slab_sim/slab_fit_o2x[_tmax<T>].h5  (o2x)
  results/rt_slab_sim/slab_fit.h5             (order7)
  results/rt_slab_sim/figs/slab_quicklook{_am,_t2s,_o2x[_tmax<T>],}.png
  --variant V reads slab_rad_V.h5 (run_slab.py --variant V) and appends _V
  to every output name, e.g. slab_fit_am_norayleigh.h5.

Run:  python workspace/rt_slab_sim/fit_and_plot.py [--estimator {am,t2s,o2x,order7}]
                                                   [--tau-max T  (o2x only)]
                                                   [--variant norayleigh]
"""
import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from spectral.cumulant_fit import (_AM_MIN_CHAN, T2S_MODEL, _lin_se,
                                   fit_am_o2a, fit_spectral_model, fit_t2s)

FIG_DIR = os.path.join(cfg.OUT_DIR, "figs")

ESTIMATORS = ("am", "t2s", "o2x", "order7")
FIT_FILES = {"am": "slab_fit_am.h5", "t2s": "slab_fit_t2s.h5",
             "o2x": "slab_fit_o2x.h5", "order7": "slab_fit.h5"}
# fitted fields that get a run-to-run *_std
FIT_NAMES = {"am": ("k1", "k2diag", "f", "a1", "intercept"),
             "t2s": ("k1", "k2", "k1w", "a1", "intercept"),
             "o2x": ("k1", "k2", "a1", "intercept"),
             "order7": ("k1", "k2", "intercept")}
QUICKLOOK_SUFFIX = {"am": "_am", "t2s": "_t2s", "o2x": "_o2x", "order7": ""}
# slab name <- fit_am_o2a key; c_am is stored as "intercept" (gamma = exp(c))
AM_KEYS = {"k1": "k1_am", "k2diag": "k2diag_am", "f": "f_am", "a1": "a1_am",
           "intercept": "c_am", "rms": "rms_am", "ok": "ok_am"}
# slab name <- fit_t2s key
T2S_KEYS = {"k1": "k1_t2s", "k2": "k2_t2s", "k1w": "k1w_t2s", "a1": "a1_t2s",
            "intercept": "c_t2s", "rms": "rms_t2s"}
# slab name <- fit_columns_o2x key (closure (i), unconstrained lstsq)
O2X_FIELDS = ("k1", "k2", "a1", "intercept", "rms", "maxres", "nchan")
O2X_MODEL = ("closure (i): ln T = c + a1*x - k1*tau + 0.5*k2*tau^2, unconstrained "
             "lstsq on [1, x, -tau, 0.5*tau^2] (spectral.cumulant_fit._lin_se), "
             "total slant tau, no bounds, no floor; intercept = c")
X_DEFINITION = ("x = 2*(wvl - wvl_min)/(wvl_max - wvl_min) - 1 over the "
                "simulated wavelengths (wvl_min/max = extremes of the fitted "
                "window); production uses (j - 507.5)/507.5 on the channel index")


def spectral_x(wvl_nm):
    """Agreed-model spectral coordinate of the slab wavelengths in [-1, 1].

    Production x is linear in detector channel index and therefore (nearly)
    linear in wavelength, increasing with wavelength in O2A.  The slab has no
    detector, so x is the normalised wavelength position over the window
    spanned by the simulated wavelengths, the same for every column (masked
    wavelengths keep their x, as masked channels do in production)."""
    lo, hi = float(np.min(wvl_nm)), float(np.max(wvl_nm))
    return 2.0 * (wvl_nm - lo) / (hi - lo) - 1.0


def fit_columns_am(rad, toa, slant_tau, mu0, x):
    """Agreed-model O2A fit of every column: rad (nw, Nx) -> dict of (Nx,)."""
    nw, nx = rad.shape
    out = {name: np.full(nx, np.nan) for name in AM_KEYS}
    for name in FIT_NAMES["am"]:
        out[f"{name}_sig"] = np.full(nx, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.pi * rad / (mu0 * toa[:, None])
    T = np.where(T > 1, np.nan, T)                     # compute_transmittance
    lnT = np.where(T > 0, np.log(T), np.nan)
    for ix in range(nx):
        y = lnT[:, ix]
        mask = np.isfinite(y) & np.isfinite(slant_tau)
        if mask.sum() < _AM_MIN_CHAN:
            continue
        order = np.argsort(slant_tau[mask])
        am = fit_am_o2a(slant_tau[mask][order], x[mask][order], y[mask][order])
        for name, key in AM_KEYS.items():
            out[name][ix] = am[key]
        for name in FIT_NAMES["am"]:
            out[f"{name}_sig"][ix] = am["sig_" + AM_KEYS[name]]
    return out


def fit_columns_t2s(rad, toa, slant_tau, tau_h2o, mu0, x):
    """t2s fit of every column (fit_t2s, no x^2 term: the O2A design)."""
    nw, nx = rad.shape
    out = {name: np.full(nx, np.nan) for name in T2S_KEYS}
    for name in FIT_NAMES["t2s"]:
        out[f"{name}_sig"] = np.full(nx, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.pi * rad / (mu0 * toa[:, None])
    T = np.where(T > 1, np.nan, T)                     # compute_transmittance
    lnT = np.where(T > 0, np.log(T), np.nan)
    tau_dry = slant_tau - tau_h2o
    for ix in range(nx):
        res = fit_t2s(tau_dry, tau_h2o, x, lnT[:, ix], land=False)
        for name, key in T2S_KEYS.items():
            out[name][ix] = res[key]
        for name in FIT_NAMES["t2s"]:
            out[f"{name}_sig"][ix] = res["sig_" + T2S_KEYS[name]]
    return out


def o2x_tag(tau_max=None):
    """Estimator tag of an o2x run: 'o2x' (all tau) or 'o2x_tmax<T>'."""
    return "o2x" if tau_max is None else f"o2x_tmax{tau_max:g}"


def fit_columns_o2x(rad, toa, slant_tau, mu0, x, tau_max=None):
    """Closure (i) fit of every column: rad (nw, Nx) -> dict of (Nx,).

    T, ln T and the channel mask are those of fit_columns_am (T > 1 and
    non-finite ln T / tau dropped, >= _AM_MIN_CHAN channels); tau_max, if
    given, also drops tau > tau_max.  maxres = max |ln T residual|."""
    nw, nx = rad.shape
    out = {name: np.full(nx, np.nan) for name in O2X_FIELDS}
    for name in FIT_NAMES["o2x"]:
        out[f"{name}_sig"] = np.full(nx, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.pi * rad / (mu0 * toa[:, None])
    T = np.where(T > 1, np.nan, T)                     # compute_transmittance
    lnT = np.where(T > 0, np.log(T), np.nan)
    in_range = np.isfinite(slant_tau)
    if tau_max is not None:
        in_range &= slant_tau <= tau_max
    for ix in range(nx):
        y = lnT[:, ix]
        mask = np.isfinite(y) & in_range
        out["nchan"][ix] = mask.sum()
        if mask.sum() < _AM_MIN_CHAN:
            continue
        t, xm, ym = slant_tau[mask], x[mask], y[mask]
        cols = [np.ones_like(t), xm, -t, 0.5 * t**2]
        coef, rms, se = _lin_se(cols, ym)
        resid = ym - np.column_stack(cols) @ coef
        for i, name in enumerate(("intercept", "a1", "k1", "k2")):
            out[name][ix] = coef[i]
            out[f"{name}_sig"][ix] = se[i]
        out["rms"][ix] = rms
        out["maxres"][ix] = float(np.max(np.abs(resid)))
    return out


def fit_columns(rad, toa, slant_tau, mu0):
    """Order-7 fit of every column: rad (nw, Nx) -> dict of (Nx,) arrays."""
    nw, nx = rad.shape
    k1 = np.full(nx, np.nan)
    k2 = np.full(nx, np.nan)
    intercept = np.full(nx, np.nan)
    T = np.pi * rad / (mu0 * toa[:, None])
    for ix in range(nx):
        y = T[:, ix]
        if np.any(y <= 0):
            continue
        popt = fit_spectral_model(slant_tau, np.log(y), cfg.FIT_ORDER,
                                  smooth=cfg.FIT_SMOOTH)
        k1[ix], k2[ix] = popt[0], popt[1]
        intercept[ix] = popt[-1]
    return {"k1": k1, "k2": k2, "intercept": intercept}


def plen_moments(plen_hist, edges_mid):
    """(nw, nrun, Nx, Ntp) radiance-fraction histograms -> per-column mean and
    std of total geometric path, per wavelength (runs pooled).

    The histograms are MCARaTS Rad_mplen=3 tallies: fraction of each pixel's
    radiance per total-geometric-path bin (see run_slab.SlabMcarats).  The
    totals include the constant vacuum leg TOA -> 705-km sensor, which is
    identical for all columns/solvers and therefore cancels when moments are
    plotted as anomalies against the clear-sky far field.

    Caveat for closure against the fitted cumulants: these are GEOMETRIC
    path moments, while the spectral fit senses ABSORPTION-WEIGHTED paths
    (layers contribute in proportion to absorber density).  The two agree
    where surface-reflected photons dominate the radiance (clear columns,
    bright surfaces) and diverge where high-altitude scattering carries the
    radiance (dark-surface shadow band, in-cloud) -- an exact second-moment
    closure would need an absorption-weighted accumulator in mcarRad.F90."""
    h = plen_hist.mean(axis=1)                     # pool runs: (nw, Nx, Ntp)
    wsum = h.sum(axis=-1)
    wsum = np.where(wsum > 0, wsum, np.nan)
    mean = (h * edges_mid).sum(axis=-1) / wsum
    second = (h * edges_mid**2).sum(axis=-1) / wsum
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def main(estimator="am", tau_max=None, variant=None):
    if tau_max is not None and estimator != "o2x":
        raise SystemExit("--tau-max applies to --estimator o2x only")
    tag = o2x_tag(tau_max) if estimator == "o2x" else estimator
    rad_file = cfg.variant_path(cfg.RAD_FILE, variant)
    with h5py.File(cfg.ATM_FILE, "r") as f:
        sza = f["meta"].attrs["sza"]
    mu0 = np.cos(np.deg2rad(sza))

    x_km = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM
    results = {}
    if estimator == "t2s":
        with h5py.File(cfg.OD_H2O_FILE, "r") as f:
            nu_h2o, frac_h2o_all = f["nu_cm1"][...], f["frac_h2o"][...]
        with h5py.File(cfg.OD_FILE, "r") as f:
            assert np.array_equal(f["nu_cm1"][...], nu_h2o), cfg.OD_H2O_FILE
    with h5py.File(rad_file, "r") as f:
        slant_tau_all = f["slant_tau"][...]
        wvl_nm_all = f["wvl_nm"][...]
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"],
                            f.attrs["plen_nbin"] + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        x_spec_all = spectral_x(wvl_nm_all)
        for surface in cfg.SURFACE_ALBEDOS:
            for solver in cfg.SOLVERS:
                key = f"{surface}/{solver}"
                if key not in f:
                    continue
                g = f[key]
                iw = g["iw"][...]
                rad = np.squeeze(g["rad"][...]).reshape(iw.size, cfg.NX)
                toa = g["toa"][...]
                tau = slant_tau_all[iw]
                if estimator == "am":
                    x_spec = x_spec_all[iw]

                    def fit(r):
                        return fit_columns_am(r, toa, tau, mu0, x_spec)
                elif estimator == "t2s":
                    x_spec = x_spec_all[iw]
                    tau_w = frac_h2o_all[iw] * tau

                    def fit(r):
                        return fit_columns_t2s(r, toa, tau, tau_w, mu0, x_spec)
                elif estimator == "o2x":
                    x_spec = x_spec_all[iw]

                    def fit(r):
                        return fit_columns_o2x(r, toa, tau, mu0, x_spec, tau_max)
                else:
                    def fit(r):
                        return fit_columns(r, toa, tau, mu0)
                res = fit(rad)
                res["rad"] = rad
                if "rad_runs" in g:
                    runs = g["rad_runs"][...]        # (nw, Nx, Nrun) squeezed?
                    runs = runs.reshape(iw.size, cfg.NX, -1)
                    nrun = runs.shape[-1]
                    per_run = [fit(runs[..., ir]) for ir in range(nrun)]
                    for name in FIT_NAMES[estimator]:
                        res[f"{name}_std"] = np.std(
                            [p[name] for p in per_run], axis=0, ddof=1)
                if "plen_hist" in g:
                    pl_mean, pl_std = plen_moments(g["plen_hist"][...], mid)
                    res["plen_mean"] = pl_mean       # (nw, Nx), meters
                    res["plen_std"] = pl_std
                results[key] = res

    fit_file = cfg.variant_path(
        os.path.join(cfg.OUT_DIR, f"slab_fit_{tag}.h5"
                     if estimator == "o2x" else FIT_FILES[estimator]), variant)
    with h5py.File(fit_file, "w") as f:
        f["x_km"] = x_km
        if variant:
            f.attrs["variant"] = variant
            f.attrs["rad_file"] = os.path.basename(rad_file)
        if estimator == "am":
            f["x_spec"] = x_spec_all
            f.attrs["estimator"] = "am"
            f.attrs["fit_model"] = ("O2A agreed model, fit schema doppler_am_t2_v2 "
                                    "(spectral.cumulant_fit.fit_am_o2a): ln T = c + "
                                    "a1*x + ln[(1-f) exp(-k1 tau) + f], k2 = 0; "
                                    "k2diag = free k2 of the refit with "
                                    "0.5*k2*tau^2 in the exponent; intercept = c")
            f.attrs["x_definition"] = X_DEFINITION
            f.attrs["wvl_window_nm"] = (float(wvl_nm_all.min()),
                                        float(wvl_nm_all.max()))
            f.attrs["std_definition"] = ("*_std: run-to-run std (ddof 1) over the "
                                         "NRUN Monte Carlo runs; *_sig: formal "
                                         "1-sigma of the pooled-run fit")
        elif estimator == "t2s":
            f["x_spec"] = x_spec_all
            f["frac_h2o"] = frac_h2o_all
            f.attrs["estimator"] = "t2s"
            f.attrs["fit_model"] = (f"{T2S_MODEL} (spectral.cumulant_fit.fit_t2s): "
                                    "ln T = c + a1*x - k1*tau_dry - k1w*tau_h2o + "
                                    "0.5*k2*tau^2, OLS, no bounds, no floor; "
                                    "tau_h2o = frac_h2o*tau from slab_od_h2o.h5; "
                                    "intercept = c")
            f.attrs["x_definition"] = X_DEFINITION
            f.attrs["wvl_window_nm"] = (float(wvl_nm_all.min()),
                                        float(wvl_nm_all.max()))
            f.attrs["std_definition"] = ("*_std: run-to-run std (ddof 1) over the "
                                         "NRUN Monte Carlo runs; *_sig: formal "
                                         "1-sigma of the pooled-run fit")
        elif estimator == "o2x":
            f["x_spec"] = x_spec_all
            f.attrs["estimator"] = tag
            f.attrs["fit_model"] = O2X_MODEL
            f.attrs["tau_max"] = np.nan if tau_max is None else float(tau_max)
            f.attrs["x_definition"] = X_DEFINITION
            f.attrs["wvl_window_nm"] = (float(wvl_nm_all.min()),
                                        float(wvl_nm_all.max()))
            f.attrs["std_definition"] = ("*_std: run-to-run std (ddof 1) over the "
                                         "NRUN Monte Carlo runs; *_sig: formal "
                                         "1-sigma of the pooled-run fit")
        for key, res in results.items():
            g = f.create_group(key)
            for name, val in res.items():
                g[name] = val
    print(f"Wrote {fit_file}")

    # ---------------- quick-look figure ----------------
    os.makedirs(FIG_DIR, exist_ok=True)
    surfaces = [s for s in cfg.SURFACE_ALBEDOS
                if any(k.startswith(s) for k in results)]
    nrows, ncols = 4, len(surfaces)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 11),
                             sharex=True, squeeze=False)
    colors = {"3d": "tab:red", "ipa": "tab:blue"}
    # mid-tau wavelength for the path-length panel
    iw_mid = int(np.argmin(np.abs(slant_tau_all - 1.0)))

    for jc, surface in enumerate(surfaces):
        for solver in cfg.SOLVERS:
            key = f"{surface}/{solver}"
            if key not in results:
                continue
            res = results[key]
            lbl = "3-D" if solver == "3d" else "ICA"
            c = colors[solver]
            for jr, name in enumerate(("k1", FIT_NAMES[estimator][1],
                                       "intercept")):
                ax = axes[jr, jc]
                # show the intercept as exp() = effective scene reflectance
                tf = np.exp if name == "intercept" else (lambda v: v)
                ax.plot(x_km, tf(res[name]), color=c, lw=1.5, label=lbl)
                if f"{name}_std" in res:
                    ax.fill_between(x_km, tf(res[name] - res[f"{name}_std"]),
                                    tf(res[name] + res[f"{name}_std"]),
                                    color=c, alpha=0.25, lw=0)
            if "plen_mean" in res:
                ax = axes[3, jc]
                iw_arr = np.arange(res["plen_mean"].shape[0])
                pl = res["plen_mean"][iw_mid] / 1e3
                ax.plot(x_km, pl - np.nanmedian(pl[x_km < 5.0]),
                        color=c, lw=1.5, label=f"{lbl} mean")
                pls = res["plen_std"][iw_mid] / 1e3
                ax.plot(x_km, pls - np.nanmedian(pls[x_km < 5.0]),
                        color=c, lw=1.2, ls="--", label=f"{lbl} std")

        alb = cfg.SURFACE_ALBEDOS[surface]
        axes[0, jc].set_title(f"{surface} surface (albedo {alb:.2f})")
        for jr, lab in enumerate((r"$k_1$  ($\langle l'\rangle$)",
                                  (r"$k_2^{\mathrm{diag}}$" if estimator == "am"
                                   else r"$k_2$  (var $l'$)"),
                                  "$\\gamma$  (effective reflectance)",
                                  "path enh. @ %.2f nm, slant $\\tau\\approx$"
                                  "%.2f (km)"
                                  % (wvl_nm_all[iw_mid], slant_tau_all[iw_mid]))):
            ax = axes[jr, jc]
            ax.axvspan(*cfg.CLOUD_X_KM, color="0.85", zorder=0)
            ax.set_ylabel(lab)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1, jc].set_xlabel("x (km)")

    est_txt = {"am": "agreed model (doppler_am_t2_v2)",
               "t2s": "t2s test model (two-tau order 2 + slope, no floor)",
               "o2x": ("closure (i), order 2 + slope, unconstrained"
                       + ("" if tau_max is None else f", tau <= {tau_max:g}")),
               "order7": "order-7 cumulant fit (no-SG)"}[estimator]
    fig.suptitle(f"x-z slab, O2A, {est_txt}  --  "
                 "3-D vs ICA, identical scene; grey band = cloud", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    out = cfg.variant_path(
        os.path.join(FIG_DIR, "slab_quicklook"
                     + ("_" + tag if estimator == "o2x"
                        else QUICKLOOK_SUFFIX[estimator]) + ".png"), variant)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", choices=ESTIMATORS, default="am",
                    help="am: production O2A agreed model -> slab_fit_am.h5 "
                         "(default); t2s: two-tau order-2 test model -> "
                         "slab_fit_t2s.h5; o2x: closure (i), second order "
                         "with a linear continuum term -> slab_fit_o2x.h5; "
                         "order7: retired order-7 fit -> slab_fit.h5")
    ap.add_argument("--tau-max", type=float, default=None, metavar="T",
                    help="o2x only: fit channels with slant tau <= T "
                         "-> slab_fit_o2x_tmax<T>.h5")
    ap.add_argument("--variant", choices=cfg.VARIANTS, default=None,
                    help="fit slab_rad_<variant>.h5 (run_slab.py --variant) "
                         "-> slab_fit_<estimator>_<variant>.h5")
    args = ap.parse_args()
    main(args.estimator, args.tau_max, args.variant)
