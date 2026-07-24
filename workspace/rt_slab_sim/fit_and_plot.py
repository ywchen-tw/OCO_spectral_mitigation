"""
Stage 4: refit the slab radiances with the PRODUCTION cumulant estimator
and draw the quick-look (pre-Fig.-G1) figure.

Per column x, per surface, per solver: fit ln T(lambda) vs slant gas tau
with src/spectral/cumulant_fit.fit_spectral_model (order 7, exact
lstsq/BVLS, no Savitzky-Golay, k1/k2 >= 0) -- the identical code path used
for the OCO-2 observations.  T = pi * radiance / (mu0 * toa); the constant
normalization is absorbed by the intercept.

Also reduces the MCARaTS Rad_mplen=3 path-length histograms to per-column
mean/std of total geometric path (the PPDF moments the fit is supposed to
encode -- closure panel).

Outputs:
  results/rt_slab_sim/slab_fit.h5
  results/rt_slab_sim/figs/slab_quicklook.png

Run:  python workspace/rt_slab_sim/fit_and_plot.py
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from spectral.cumulant_fit import fit_spectral_model

FIG_DIR = os.path.join(cfg.OUT_DIR, "figs")


def fit_columns(rad, toa, slant_tau, mu0):
    """Fit every column: rad (nw, Nx) -> dict of (Nx,) arrays."""
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
    std of total geometric path, per wavelength (runs pooled)."""
    h = plen_hist.mean(axis=1)                     # pool runs: (nw, Nx, Ntp)
    wsum = h.sum(axis=-1)
    wsum = np.where(wsum > 0, wsum, np.nan)
    mean = (h * edges_mid).sum(axis=-1) / wsum
    second = (h * edges_mid**2).sum(axis=-1) / wsum
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def main():
    with h5py.File(cfg.ATM_FILE, "r") as f:
        sza = f["meta"].attrs["sza"]
    mu0 = np.cos(np.deg2rad(sza))

    x_km = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM
    results = {}
    with h5py.File(cfg.RAD_FILE, "r") as f:
        slant_tau_all = f["slant_tau"][...]
        wvl_nm_all = f["wvl_nm"][...]
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"],
                            f.attrs["plen_nbin"] + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
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
                res = fit_columns(rad, toa, tau, mu0)
                res["rad"] = rad
                if "rad_runs" in g:
                    runs = g["rad_runs"][...]        # (nw, Nx, Nrun) squeezed?
                    runs = runs.reshape(iw.size, cfg.NX, -1)
                    nrun = runs.shape[-1]
                    per_run = [fit_columns(runs[..., ir], toa, tau, mu0)
                               for ir in range(nrun)]
                    for name in ("k1", "k2", "intercept"):
                        res[f"{name}_std"] = np.std(
                            [p[name] for p in per_run], axis=0, ddof=1)
                if "plen_hist" in g:
                    pl_mean, pl_std = plen_moments(g["plen_hist"][...], mid)
                    res["plen_mean"] = pl_mean       # (nw, Nx), meters
                    res["plen_std"] = pl_std
                results[key] = res

    with h5py.File(os.path.join(cfg.OUT_DIR, "slab_fit.h5"), "w") as f:
        f["x_km"] = x_km
        for key, res in results.items():
            g = f.create_group(key)
            for name, val in res.items():
                g[name] = val
    print("Wrote slab_fit.h5")

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
            for jr, name in enumerate(("k1", "k2", "intercept")):
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
                                  r"$k_2$  (var $l'$)",
                                  "exp(intercept)  (effective reflectance)",
                                  "path enh. @ %.2f nm, slant $\\tau\\approx$"
                                  "%.2f (km)"
                                  % (wvl_nm_all[iw_mid], slant_tau_all[iw_mid]))):
            ax = axes[jr, jc]
            ax.axvspan(*cfg.CLOUD_X_KM, color="0.85", zorder=0)
            ax.set_ylabel(lab)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1, jc].set_xlabel("x (km)")

    fig.suptitle("x-z slab, O2A, production estimator (order 7, no-SG)  --  "
                 "3-D vs ICA, identical scene; grey band = cloud", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    out = os.path.join(FIG_DIR, "slab_quicklook.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
