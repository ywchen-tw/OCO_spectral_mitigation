"""
Fig. G1 (manuscript, AMT style): controlled 3-D-vs-IPA slab demonstration.

Layout: 2 columns (dark / bright surface) x 5 rows --
  (a,b) fitted <l'> (production estimator, order 7, no-SG)
  (c,d) fitted var(l')
  (e,f) MC-tallied PPDF mean (relative-path units, continuum wavelength)
  (g,h) MC-tallied PPDF variance -- rows (e-h) are the closure rows: the
        directly tallied path moments the fitted cumulants are supposed
        to encode.
  (i,j) continuum reflectance gamma = exp(intercept)
Each row shares one fixed y range across the two surfaces (ROW_YLIM;
reviewer request, 2026-10) so the dark and bright columns compare by eye.

Also computes closure statistics (Pearson r between fitted k1/k2 and the
tallied geometric moments across 3-D columns; IPA-null residuals) ->
results/rt_slab_sim/closure_stats.json, printed for the appendix text.
NOTE the closure is a SHAPE comparison: fitted cumulants are
absorption-weighted (pressure-weighted altitude sampling) while the mode-3
tally is geometric path -- amplitudes differ where photons turn around
aloft; state this in the caption.

Outputs: figG1_mc_3d_vs_ica.{png,pdf} into BOTH manuscript figure trees,
         manuscript/AMT_draft/figures/ and manuscript/JQSRT_draft/figures/
         (+ preview copy in results/rt_slab_sim/figs/).
         Set SLAB_FIG_OUT_DIR to send the pair to a single scratch directory
         instead (verification runs); the default is the two manuscript trees.

Run:  python workspace/rt_slab_sim/make_fig_g1.py
"""
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "workspace"))
from plot_style import (MEAN_L_LABEL, VAR_L_LABEL, apply_manuscript_style,
                        panel_label)

FIG_DIRS = [
    os.path.join(cfg.REPO_ROOT, "manuscript", "AMT_draft", "figures"),
    os.path.join(cfg.REPO_ROOT, "manuscript", "JQSRT_draft", "figures"),
]
# verification escape hatch: send the pair to one scratch directory instead
if os.environ.get("SLAB_FIG_OUT_DIR"):
    FIG_DIRS = [os.environ["SLAB_FIG_OUT_DIR"]]
OUT_BASE = "figG1_mc_3d_vs_ica"
# opt-in single-surface edition (2026-10-10, main-text figure with the dark surface only;
# the default two-column figure moves to the SI): SLAB_FIG_SURFACES=dark
SURFACES = [s for s in os.environ.get("SLAB_FIG_SURFACES", ",".join(cfg.SURFACE_ALBEDOS)).split(",") if s]
if len(SURFACES) < len(cfg.SURFACE_ALBEDOS):
    OUT_BASE += "_" + "_".join(SURFACES)
# opt-in row subset (2026-10-11, KSS review: the main text uses only <l'>, so the main
# figure drops the two variance rows): SLAB_FIG_ROWS=0,2,4 (0 fitted mean, 1 fitted var,
# 2 MC mean, 3 MC var, 4 reflectance). Default: all five rows.
ROWS = [int(r) for r in os.environ.get("SLAB_FIG_ROWS", "0,1,2,3,4").split(",")]
if ROWS != [0, 1, 2, 3, 4]:
    OUT_BASE += "_rows" + "".join(str(r) for r in ROWS)


class _NullAx:
    """Stand-in for a dropped row: every plotting call is a no-op."""
    def __getattr__(self, name):
        return lambda *a, **k: None

COL_3D = "#c1272d"     # 3-D
COL_ICA = "#0000a7"    # IPA

# one y range per row, shared by the dark and bright columns
ROW_YLIM = ((0.4, 1.3),      # fitted <l'>
            (0.0, 0.8),      # fitted var(l')
            (0.9, 1.04),     # MC mean
            (0.0, 0.035),    # MC variance
            (0.0, 0.8))      # effective reflectance


def continuum_moments(f, surface, solver, mid):
    """Relative-path mean/std per column from the continuum plen histogram.

    Uses the wavelength with the smallest slant tau (~1e-4): with no
    absorption weighting the Rad_mplen=3 histogram is the PURE geometric
    photon path-length distribution, i.e. the PPDF whose cumulants the
    spectral fit is supposed to encode (tally mechanics documented in
    run_slab.SlabMcarats; geometric-vs-absorption-weighted caveat in
    fit_and_plot.plen_moments)."""
    g = f[f"{surface}/{solver}"]
    slant = f["slant_tau"][...][g["iw"][...]]
    iw = int(np.argmin(slant))
    h = g["plen_hist"][iw].mean(axis=0)          # (Nx, Ntp)
    s = h.sum(axis=1)
    s = np.where(s > 0, s, np.nan)
    mean = (h * mid).sum(axis=1) / s
    var = (h * mid**2).sum(axis=1) / s - mean**2
    return mean, np.sqrt(np.maximum(var, 0.0)), h


def main():
    apply_manuscript_style()

    with h5py.File(cfg.ATM_FILE, "r") as f:
        sza = float(f["meta"].attrs["sza"])
    mu0 = np.cos(np.deg2rad(sza))
    l_direct = cfg.LEVELS_KM[-1] * (1.0 / mu0
                                    + 1.0 / np.cos(np.deg2rad(cfg.SENSOR_ZENITH)))

    with h5py.File(os.path.join(cfg.OUT_DIR, "slab_fit.h5"), "r") as f:
        x_km = f["x_km"][...]
        fit = {k: {n: f[k][n][...] for n in f[k]}
               for k in (f"{s}/{v}" for s in cfg.SURFACE_ALBEDOS
                         for v in cfg.SOLVERS)}

    mom = {}
    hist = {}
    with h5py.File(cfg.RAD_FILE, "r") as f:
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"],
                            f.attrs["plen_nbin"] + 1)
        mid = 0.5 * (edges[:-1] + edges[1:]) / 1e3       # km
        for s in cfg.SURFACE_ALBEDOS:
            for v in cfg.SOLVERS:
                mean, std, h = continuum_moments(f, s, v, mid)
                mom[f"{s}/{v}"] = (mean, std)
                hist[f"{s}/{v}"] = h

    far = x_km < 5.0

    # Absolute-path anchor for the MC moment row: the far-field clear-sky
    # histogram peak IS the direct-bounce population, so its bin midpoint is
    # the tallied path of an l = 1 photon.  Subtracting it removes the
    # constant instrumental offset (645 km vacuum TOA->sensor leg plus the
    # injection-plane accounting, slab_config.PLEN_MODE) exactly as
    # plot_ppdf.py panel (d) does.  One anchor for both surfaces and both
    # solvers; native 200 m bins, no rebin.
    L_peak = float(mid[int(np.argmax(hist["dark/3d"][far].mean(axis=0)))])
    in_cloud = (x_km >= cfg.CLOUD_X_KM[0]) & (x_km < cfg.CLOUD_X_KM[1])

    # ---------------- closure statistics ----------------
    stats = {"l_direct_km": l_direct, "sza": sza}
    for s in cfg.SURFACE_ALBEDOS:
        k1 = fit[f"{s}/3d"]["k1"]
        k2 = fit[f"{s}/3d"]["k2"]
        m3, sd3 = mom[f"{s}/3d"]
        ok = np.isfinite(k1) & np.isfinite(m3)
        stats[f"r_k1_meanL_{s}"] = float(np.corrcoef(k1[ok], m3[ok])[0, 1])
        okv = np.isfinite(k2) & np.isfinite(sd3)
        stats[f"r_k2_varL_{s}"] = float(np.corrcoef(k2[okv], sd3[okv]**2)[0, 1])
        # clear-column (adjacency) closure: in-cloud columns excluded, where
        # geometric spread narrows (cloud-top bounce) while the
        # absorption-weighted spread the fit senses broadens (in-cloud
        # multiple scattering) -- the geometric tally is only a shape proxy
        # there
        clr = ok & ~in_cloud
        stats[f"r_k1_meanL_clear_{s}"] = float(
            np.corrcoef(k1[clr], m3[clr])[0, 1])
        clrv = okv & ~in_cloud
        stats[f"r_k2_varL_clear_{s}"] = float(
            np.corrcoef(k2[clrv], sd3[clrv]**2)[0, 1])
        # ICA null residuals outside the cloud (fraction of the 3-D range)
        for name in ("k1", "k2"):
            v_ica = fit[f"{s}/ipa"][name]
            resid = np.nanmax(np.abs(v_ica[~in_cloud]
                                     - np.nanmedian(v_ica[far])))
            rng = np.nanmax(fit[f"{s}/3d"][name]) - np.nanmin(fit[f"{s}/3d"][name])
            stats[f"ica_null_{name}_{s}"] = float(resid)
            stats[f"ica_null_{name}_frac_{s}"] = float(resid / rng)

    with open(os.path.join(cfg.OUT_DIR, "closure_stats.json"), "w") as fj:
        json.dump(stats, fj, indent=2)
    print(json.dumps(stats, indent=2))

    # ---------------- figure ----------------
    ncol = len(SURFACES)
    nrow = len(ROWS)
    fig, axes_real = plt.subplots(nrow, ncol, figsize=(3.9 * ncol if ncol > 1 else 4.2, 10.8 * nrow / 5 + 0.4),
                                  sharex=True, squeeze=False)
    axes = np.empty((5, ncol), dtype=object)
    for jr in range(5):
        for jc in range(ncol):
            axes[jr, jc] = axes_real[ROWS.index(jr), jc] if jr in ROWS else _NullAx()
    tags = "abcdefghij"

    for jc, s in enumerate(SURFACES):
        for v, col, lbl in (("3d", COL_3D, "3-D"), ("ipa", COL_ICA, "IPA")):
            res = fit[f"{s}/{v}"]
            for jr, name in enumerate(("k1", "k2")):
                ax = axes[jr, jc]
                # k2 dashed in both columns, matching the variance styling of
                # the MC row below it
                ax.plot(x_km, res[name], color=col, lw=1.3,
                        ls="--" if name == "k2" else "-", label=lbl)
                if f"{name}_std" in res:
                    ax.fill_between(x_km, res[name] - res[f"{name}_std"],
                                    res[name] + res[f"{name}_std"],
                                    color=col, alpha=0.25, lw=0)
            mean, std = mom[f"{s}/{v}"]
            # direct moments (not anomalies): the far-field direct-bounce
            # population sits at l = 1 by construction of L_peak
            l_mean = 1.0 + (mean - L_peak) / l_direct
            var_l = std**2 / l_direct**2
            axes[2, jc].plot(x_km, l_mean, color=col, lw=1.3, label=lbl)
            axes[3, jc].plot(x_km, var_l, color=col, lw=1.3, ls="--",
                             label=lbl)
            axes[4, jc].plot(x_km, np.exp(res["intercept"]), color=col,
                             lw=1.3, label=lbl)

        alb = cfg.SURFACE_ALBEDOS[s]
        axes[0, jc].set_title(
            ("Dark" if s == "dark" else "Bright")
            + f" surface (albedo {alb:.2f})", fontsize=10)
        axes_real[-1, jc].set_xlabel("Along-slab distance $x$ (km)")

    # MC rows reuse the plot_style l' labels with an "MC" prefix so the
    # tallied moments read as the same quantities the fit estimates
    row_labels = (MEAN_L_LABEL, VAR_L_LABEL,
                  "MC " + MEAN_L_LABEL,
                  "MC " + VAR_L_LABEL,
                  "Continuum reflectance $\\gamma$")
    for kr, jr in enumerate(ROWS):
        for jc in range(ncol):
            ax = axes[jr, jc]
            ax.axvspan(*cfg.CLOUD_X_KM, color="0.88", zorder=0)
            ax.set_ylim(*ROW_YLIM[jr])
            panel_label(ax, f"({tags[kr * ncol + jc]})")
            if jc == 0:
                ax.set_ylabel(row_labels[jr])
    axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)

    # sun direction (tilted by the SZA, pointing down-sun toward +x) +
    # shadow annotation on the top-left panel
    ax = axes[0, 0]
    dx, dy = 0.10 * np.sin(np.deg2rad(sza)), 0.10 * np.cos(np.deg2rad(sza))
    ax.annotate("", xy=(0.05 + dx, 0.97 - dy), xytext=(0.05, 0.97),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.05 + dx + 0.02, 0.99, f"sun\nSZA {sza:.0f}$^\\circ$",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.text(18.5, 0.55, "shadow", fontsize=8, ha="center", color="0.3")
    ax.text(12.0, 0.88, "cloud", fontsize=8, ha="center", color="0.3")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.14, wspace=0.25)

    for fig_dir in FIG_DIRS:
        os.makedirs(fig_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"{OUT_BASE}.{ext}"))
    prev = os.path.join(cfg.OUT_DIR, "figs", f"{OUT_BASE}.png")
    fig.savefig(prev)
    print(f"Wrote {OUT_BASE}.png/.pdf into "
          + ", ".join(FIG_DIRS) + f" (+ preview {prev})")


if __name__ == "__main__":
    main()
