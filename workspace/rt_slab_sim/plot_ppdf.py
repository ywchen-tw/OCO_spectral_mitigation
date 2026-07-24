"""
Plot the tallied photon path-length distribution function (PPDF) across the
slab -- the direct MC view of the object the cumulant fit reads spectrally.

Uses the MCARaTS Rad_mplen=3 per-pixel histograms stored in slab_rad.h5
(radiance-fraction per total-geometric-path bin, per column / wavelength /
run).  At the continuum wavelength (slant tau ~ 0) there is no absorption
weighting, so the histogram IS the pure PPDF.

Figure per surface:
  (a,b) PPDF vs x heat maps for 3-D and ICA (path anomaly on y),
  (c)   line cuts at selected columns (far-clear, illuminated edge,
        cloud center, shadow) for 3-D (solid) and ICA (dotted),
  (d)   the same cuts on the RELATIVE path axis l = L_atm / L_direct with
        L_direct = TOA slant down + nadir up (the l' = 1 reference of the
        cumulant framework).  The constant instrumental offset (vacuum leg
        to the sensor + injection-plane accounting) is removed by anchoring
        the clear-sky far-field peak -- which is the direct-bounce
        population -- at exactly l = 1.

Run:  python workspace/rt_slab_sim/plot_ppdf.py [--tau 0.0] [--rebin 10]
"""
import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

FIG_DIR = os.path.join(cfg.OUT_DIR, "figs")

# columns to cut: (x_km, label)
CUTS = [(2.5, "clear far-field"), (9.25, "illuminated edge"),
        (12.0, "cloud center"), (16.0, "shadow (near)"),
        (19.0, "shadow (far)"), (27.0, "clear east")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.0,
                    help="target slant tau of the wavelength to show "
                         "(0 -> continuum = pure PPDF)")
    ap.add_argument("--rebin", type=int, default=10,
                    help="path-bin aggregation factor (10 -> 2 km bins)")
    args = ap.parse_args()

    x_km = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM

    with h5py.File(cfg.ATM_FILE, "r") as f:
        sza = float(f["meta"].attrs["sza"])
    mu0 = np.cos(np.deg2rad(sza))
    muv = np.cos(np.deg2rad(cfg.SENSOR_ZENITH))
    l_direct = cfg.LEVELS_KM[-1] * (1.0 / mu0 + 1.0 / muv)   # km, in-atmosphere

    with h5py.File(cfg.RAD_FILE, "r") as f:
        slant_tau = f["slant_tau"][...]
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"],
                            f.attrs["plen_nbin"] + 1)
        data = {}
        for surface in cfg.SURFACE_ALBEDOS:
            for solver in cfg.SOLVERS:
                g = f[f"{surface}/{solver}"]
                iw_stored = g["iw"][...]
                iw = int(np.argmin(np.abs(slant_tau[iw_stored] - args.tau)))
                h = g["plen_hist"][iw].mean(axis=0)      # pool runs: (Nx, Ntp)
                data[(surface, solver)] = h
        tau_shown = slant_tau[iw_stored][iw]

    # rebin the path axis
    nb = (edges.size - 1) // args.rebin
    mid = 0.5 * (edges[:-1] + edges[1:])
    mid_rb = mid[:nb * args.rebin].reshape(nb, args.rebin).mean(axis=1) / 1e3  # km

    def rebin(h):
        h = h[:, :nb * args.rebin].reshape(h.shape[0], nb, args.rebin).sum(axis=2)
        s = h.sum(axis=1, keepdims=True)
        return h / np.where(s > 0, s, 1)                 # renormalize per column

    # common path-anomaly axis: subtract the clear-sky far-field mean (dark/3d)
    h_ref = rebin(data[("dark", "3d")])
    ref_cols = x_km < 5.0
    L0 = np.nansum(h_ref[ref_cols] * mid_rb, axis=1).mean()
    dL = mid_rb - L0

    for surface in cfg.SURFACE_ALBEDOS:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.2),
                                 gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
        vmax = 0.0
        H = {}
        for solver in cfg.SOLVERS:
            H[solver] = rebin(data[(surface, solver)])
            vmax = max(vmax, np.nanmax(H[solver]))
        for j, solver in enumerate(cfg.SOLVERS):
            ax = axes[j]
            pm = ax.pcolormesh(x_km, dL, H[solver].T,
                               norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
                               cmap="viridis", shading="nearest")
            ax.axvspan(*cfg.CLOUD_X_KM, color="w", alpha=0.15)
            ax.axvline(cfg.CLOUD_X_KM[0], color="w", lw=0.8, ls=":")
            ax.axvline(cfg.CLOUD_X_KM[1], color="w", lw=0.8, ls=":")
            ax.set_ylim(-25, 60)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("path anomaly vs clear-sky mean (km)")
            ax.set_title(f"({'ab'[j]}) PPDF, "
                         f"{'3-D' if solver == '3d' else 'ICA'}")
            fig.colorbar(pm, ax=ax, label="radiance fraction / 2 km bin")

        ax = axes[2]
        cmap = plt.get_cmap("plasma")
        for i, (xc, lab) in enumerate(CUTS):
            ix = int(np.argmin(np.abs(x_km - xc)))
            c = cmap(i / (len(CUTS) - 1) * 0.9)
            ax.plot(dL, H["3d"][ix], color=c, lw=1.6,
                    label=f"{lab} (x={x_km[ix]:.2f})")
            ax.plot(dL, H["ipa"][ix], color=c, lw=1.1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(vmax * 1e-4, vmax * 2)
        ax.set_xlim(-25, 60)
        ax.set_xlabel("path anomaly vs clear-sky mean (km)")
        ax.set_ylabel("radiance fraction / 2 km bin")
        ax.set_title("(c) column cuts  (solid 3-D, dotted ICA)")
        ax.legend(fontsize=8, loc="upper right")

        # (d) same cuts on the relative-path axis: anchor the clear-sky
        # far-field direct-bounce peak at l = 1, scale by the direct slant
        # path (TOA down at SZA + nadir up)
        ax = axes[3]
        clear_mean_hist = H["3d"][ref_cols].mean(axis=0)
        L_peak = mid_rb[int(np.argmax(clear_mean_hist))]
        l_rel = 1.0 + (mid_rb - L_peak) / l_direct
        for i, (xc, lab) in enumerate(CUTS):
            ix = int(np.argmin(np.abs(x_km - xc)))
            c = cmap(i / (len(CUTS) - 1) * 0.9)
            ax.plot(l_rel, H["3d"][ix], color=c, lw=1.6,
                    label=f"{lab} (x={x_km[ix]:.2f})")
            ax.plot(l_rel, H["ipa"][ix], color=c, lw=1.1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(vmax * 1e-4, vmax * 2)
        ax.set_xlim(1.0 - 25.0 / l_direct, 1.0 + 60.0 / l_direct)
        ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("relative photon path  $l = L_{atm}/L_{direct}$")
        ax.set_ylabel("radiance fraction / 2 km bin")
        ax.set_title("(d) same cuts, relative path "
                     "(direct slant = 1, $L_{direct}$ = %.0f km)" % l_direct)

        alb = cfg.SURFACE_ALBEDOS[surface]
        fig.suptitle(
            f"Photon path-length distributions, {surface} surface "
            f"(albedo {alb:.2f}), slant tau = {tau_shown:.3f}; "
            f"cloud {cfg.CLOUD_BASE_KM:.0f}-{cfg.CLOUD_TOP_KM:.0f} km at "
            f"x = {cfg.CLOUD_X_KM[0]}-{cfg.CLOUD_X_KM[1]} km, SZA 55",
            y=1.0)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"slab_ppdf_{surface}.png")
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
