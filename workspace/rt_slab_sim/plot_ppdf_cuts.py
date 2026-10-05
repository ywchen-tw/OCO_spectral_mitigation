"""
Diagnostic (author-only, not a manuscript figure): per-column cuts of the
Monte Carlo photon path-length distribution, grouped by scene region.

Same data path as plot_ppdf.py -- the MCARaTS Rad_mplen=3 per-pixel
histograms in slab_rad.h5 at the continuum wavelength (slant tau nearest 0,
so the histogram is the pure PPDF), pooled over runs, rebinned on the 200 m
path axis and renormalized per column.  The x axis is the relative path
l = 1 + (L - L_peak) / l_direct with l_direct = TOA slant down + nadir up and
L_peak the far-field (x < 5 km) direct-bounce peak of the dark/3-D case,
exactly as in plot_ppdf.py panel (d).

The sun comes from the west, so the illuminated side of the cloud is
x < 9.5 km and the shadow band is x ~ 15-22 km.

Run:  python workspace/rt_slab_sim/plot_ppdf_cuts.py [--tau 0.0] [--rebin 5]
"""
import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

FIG_DIR = os.path.join(cfg.OUT_DIR, "figs")

# regions of the slab -> the columns cut in each
REGIONS = [
    ("illuminated side", [6.25, 7.75, 8.75, 9.25]),
    ("shadow side", [15.25, 16.75, 18.25, 20.25, 21.75]),
    ("far from cloud", [1.25, 3.25, 26.25, 30.25]),
]

DETOUR_L = 1.05          # l above which a photon counts as "detoured"
XLIM_LOG = (0.85, 1.70)
XLIM_LIN = (0.90, 1.50)


def cloud_edge_distance(x):
    """Distance (km) from column center x to the nearest cloud edge."""
    x0, x1 = cfg.CLOUD_X_KM
    if x < x0:
        return x0 - x
    if x > x1:
        return x - x1
    return 0.0


def load_ppdf(tau, rebin):
    """Return (x_km, l_rel, hist dict keyed (surface, solver), tau_shown,
    l_direct, bin_km)."""
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
        raw = {}
        for surface in cfg.SURFACE_ALBEDOS:
            for solver in cfg.SOLVERS:
                g = f[f"{surface}/{solver}"]
                iw_stored = g["iw"][...]
                iw = int(np.argmin(np.abs(slant_tau[iw_stored] - tau)))
                raw[(surface, solver)] = g["plen_hist"][iw].mean(axis=0)
        tau_shown = slant_tau[iw_stored][iw]

    nb = (edges.size - 1) // rebin
    mid = 0.5 * (edges[:-1] + edges[1:])
    mid_rb = mid[:nb * rebin].reshape(nb, rebin).mean(axis=1) / 1e3      # km
    bin_km = (edges[1] - edges[0]) * rebin / 1e3

    def do_rebin(h):
        h = h[:, :nb * rebin].reshape(h.shape[0], nb, rebin).sum(axis=2)
        s = h.sum(axis=1, keepdims=True)
        return h / np.where(s > 0, s, 1)             # renormalize per column

    hist = {k: do_rebin(v) for k, v in raw.items()}

    # anchor the far-field direct-bounce peak (dark / 3-D) at l = 1
    ref_cols = x_km < 5.0
    clear_mean_hist = hist[("dark", "3d")][ref_cols].mean(axis=0)
    L_peak = mid_rb[int(np.argmax(clear_mean_hist))]
    l_rel = 1.0 + (mid_rb - L_peak) / l_direct

    return x_km, l_rel, hist, tau_shown, l_direct, bin_km, sza


def cut_stats(h, l_rel):
    """Mean l, var l and detour fraction of one column histogram."""
    w = h / h.sum() if h.sum() > 0 else h
    m = float(np.sum(w * l_rel))
    v = float(np.sum(w * (l_rel - m) ** 2))
    frac = float(np.sum(w[l_rel > DETOUR_L]))
    return m, v, frac


def make_figure(x_km, l_rel, hist, tau_shown, l_direct, bin_km, sza,
                logy, xlim, ylim, out):
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(len(cfg.SURFACE_ALBEDOS), len(REGIONS),
                             figsize=(15, 8), sharex=True, sharey=True)
    win = (l_rel >= xlim[0]) & (l_rel <= xlim[1])
    for i, surface in enumerate(cfg.SURFACE_ALBEDOS):
        for j, (region, xs) in enumerate(REGIONS):
            ax = axes[i, j]
            xs_sorted = sorted(xs, key=cloud_edge_distance)
            peak = 0.0
            for k, xc in enumerate(xs_sorted):
                ix = int(np.argmin(np.abs(x_km - xc)))
                c = cmap(0.05 + 0.80 * k / max(len(xs_sorted) - 1, 1))
                d = cloud_edge_distance(x_km[ix])
                ax.plot(l_rel, hist[(surface, "3d")][ix], color=c, lw=1.6,
                        label=f"x = {x_km[ix]:.2f} km  (d = {d:.2f})")
                ax.plot(l_rel, hist[(surface, "ipa")][ix], color=c, lw=1.1,
                        ls=":")
                peak = max(peak, float(np.nanmax(
                    hist[(surface, "3d")][ix][win])))
            ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
            if logy:
                ax.set_yscale("log")
            else:
                ax.text(0.98, 0.30,
                        f"$l\\approx1$ peak {peak:.2f}, clipped",
                        transform=ax.transAxes, fontsize=8, va="center",
                        ha="right", color="0.35")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            alb = cfg.SURFACE_ALBEDOS[surface]
            ax.set_title(f"{region}, {surface} surface (albedo {alb:.2f})",
                         fontsize=10)
            ax.legend(fontsize=7.5, loc="upper right", ncol=1,
                      framealpha=0.85, handlelength=1.6,
                      title="solid 3-D, dotted ICA", title_fontsize=7.5)
            if i == len(cfg.SURFACE_ALBEDOS) - 1:
                ax.set_xlabel("relative photon path  "
                              "$l = 1 + (L-L_{peak})/L_{direct}$")
            if j == 0:
                ax.set_ylabel(f"radiance fraction / {bin_km:.1f} km bin")

    fig.suptitle(
        f"PPDF column cuts by region, slant tau = {tau_shown:.3f}, "
        f"SZA {sza:.1f} deg (sun from the west); cloud "
        f"{cfg.CLOUD_BASE_KM:.0f}-{cfg.CLOUD_TOP_KM:.0f} km at x = "
        f"{cfg.CLOUD_X_KM[0]}-{cfg.CLOUD_X_KM[1]} km, "
        f"$L_{{direct}}$ = {l_direct:.0f} km",
        y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Wrote {out}")


def print_table(x_km, l_rel, hist):
    print()
    print(f"PPDF cut statistics (full path axis; detour = fraction with "
          f"l > {DETOUR_L})")
    print(f"{'region':<18}{'surf':<7}{'x_km':>6}{'d_edge':>8}"
          f"{'mean_3d':>9}{'var_3d':>9}{'det_3d':>8}"
          f"{'mean_ica':>10}{'var_ica':>9}{'det_ica':>9}")
    print("-" * 93)
    for surface in cfg.SURFACE_ALBEDOS:
        for region, xs in REGIONS:
            for xc in xs:
                ix = int(np.argmin(np.abs(x_km - xc)))
                m3, v3, f3 = cut_stats(hist[(surface, "3d")][ix], l_rel)
                mi, vi, fi = cut_stats(hist[(surface, "ipa")][ix], l_rel)
                print(f"{region:<18}{surface:<7}{x_km[ix]:>6.2f}"
                      f"{cloud_edge_distance(x_km[ix]):>8.2f}"
                      f"{m3:>9.4f}{v3:>9.5f}{f3:>8.4f}"
                      f"{mi:>10.4f}{vi:>9.5f}{fi:>9.4f}")
        print("-" * 93)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.0,
                    help="target slant tau of the wavelength to show "
                         "(0 -> continuum = pure PPDF)")
    ap.add_argument("--rebin", type=int, default=5,
                    help="path-bin aggregation factor (5 -> 1 km bins)")
    args = ap.parse_args()

    x_km, l_rel, hist, tau_shown, l_direct, bin_km, sza = load_ppdf(
        args.tau, args.rebin)

    # common y limits: over everything actually drawn, on the log window
    shown = []
    for surface in cfg.SURFACE_ALBEDOS:
        for _, xs in REGIONS:
            for xc in xs:
                ix = int(np.argmin(np.abs(x_km - xc)))
                for solver in cfg.SOLVERS:
                    shown.append(hist[(surface, solver)][ix])
    shown = np.array(shown)
    win = (l_rel >= XLIM_LOG[0]) & (l_rel <= XLIM_LOG[1])
    vmax = float(np.nanmax(shown[:, win]))
    pos = shown[:, win]
    pos = pos[pos > 0]
    vmin = max(float(np.percentile(pos, 2.0)), vmax * 1e-5)

    # linear panels: scale to the detour shoulder (l > 1.02), which is ~10x
    # below the direct-bounce spike; the spike is clipped and annotated
    shoulder = float(np.nanmax(shown[:, l_rel > 1.02]))

    make_figure(x_km, l_rel, hist, tau_shown, l_direct, bin_km, sza,
                logy=True, xlim=XLIM_LOG, ylim=(vmin, vmax * 3.0),
                out=os.path.join(FIG_DIR, "slab_ppdf_cuts.png"))
    make_figure(x_km, l_rel, hist, tau_shown, l_direct, bin_km, sza,
                logy=False, xlim=XLIM_LIN, ylim=(0.0, shoulder * 2.8),
                out=os.path.join(FIG_DIR, "slab_ppdf_cuts_linear.png"))

    print_table(x_km, l_rel, hist)


if __name__ == "__main__":
    main()
