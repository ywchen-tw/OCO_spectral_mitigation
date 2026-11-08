"""
Fig. G1 (manuscript, AMT style): controlled 3-D-vs-IPA slab demonstration.

Estimator (SLAB_FIG_ESTIMATOR, 2026-11-02): "am" (default) reads
results/rt_slab_sim/slab_fit_am.h5, the production O2A agreed model (fit
schema doppler_am_t2_v2, fit_and_plot.py --estimator am); "order7" reads the
retired order-7 slab_fit.h5 and reproduces the pre-2026-11-02 figure under
the stem suffix "_order7" (so it never overwrites the am editions).
"o2x" (2026-11-03 test) reads slab_fit_o2x.h5, closure (i) of the paper
(second order with a linear continuum term, unconstrained lstsq;
fit_and_plot.py --estimator o2x); "o2x_tmax<T>" reads the tau <= T refit
slab_fit_o2x_tmax<T>.h5.  Both draw k2 = var(l') in rows (c,d), get the stem
suffix "_<tag>" and write closure_stats_<tag>.json.

Rows (2026-11-03, author decision: the paper shows no variance, so the O2 A-band
has one fit, the production floor form): the am default is rows 0,5,2,4 =
fitted <l'> = k1, floor f (its upper bound _AM_F_MAX dashed), MC <l'>, gamma;
panel letters follow the rows drawn.  The other estimators keep the five-row
default below; SLAB_FIG_ROWS=0,1,2,3,4 redraws the former am edition (stem
suffix "_rows01234").

Row indices (historical layout: 2 columns (dark / bright surface) x 5 rows) --
  (a,b) fitted <l'> = k1 (am: agreed model; order7: order-7 cumulant fit)
  (c,d) am: free-k2 diagnostic k2^diag (k2 is fixed at 0 in the production
        O2A fit; k2^diag is the k2 of the refit with 0.5*k2*tau^2 added,
        the Fig. 5 / Table A1 variance diagnostic); order7: fitted var(l')
  (e,f) MC-tallied PPDF mean (relative-path units, continuum wavelength)
  (g,h) MC-tallied PPDF variance -- rows (e-h) are the closure rows: the
        directly tallied path moments the fitted cumulants are supposed
        to encode.
  (i,j) continuum reflectance gamma = exp(intercept) (am: exp(c), c at x = 0)
  row 5 (am only): unabsorbed floor f, with its upper bound as a thin dashed line
SLAB_FIG_LAYOUT=stack (2026-11-03, main-text Fig. 3): one surface, the chosen
rows stacked in one column at the printed width (0.8 x textwidth = 5.2 in).
SLAB_FIG_STEM overrides the output stem (Fig. 3 keeps the tex file name
figG1_mc_3d_vs_ica_dark_grid although it is no longer a grid).
Each row shares one fixed y range across the two surfaces (ROW_YLIM;
reviewer request, 2026-10) so the dark and bright columns compare by eye.

Also computes closure statistics (Pearson r between fitted k1 / k2 (am:
k2diag) and the tallied geometric moments across 3-D columns; IPA-null
residuals) -> results/rt_slab_sim/closure_stats_am.json (am) or
closure_stats.json (order7), printed for the appendix text.
NOTE the closure is a SHAPE comparison: fitted cumulants are
absorption-weighted (pressure-weighted altitude sampling) while the mode-3
tally is geometric path -- amplitudes differ where photons turn around
aloft; state this in the caption.

Outputs: figG1_mc_3d_vs_ica.{png,pdf} into BOTH manuscript figure trees,
         manuscript/AMT_draft/figures/ and manuscript/JQSRT_draft/figures/
         (+ preview copy in results/rt_slab_sim/figs/).
         Set SLAB_FIG_OUT_DIR to send the pair to a single scratch directory
         instead (verification runs); the default is the two manuscript trees.
         SLAB_FIG_WRITE_STATS=1 writes the closure JSON even then.

Run:  python workspace/rt_slab_sim/make_fig_g1.py
"""
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import slab_config as cfg

sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "workspace"))
sys.path.insert(0, os.path.join(cfg.REPO_ROOT, "src"))
from spectral.cumulant_fit import _AM_F_MAX
from plot_style import (MEAN_L_LABEL, VAR_L_LABEL, apply_manuscript_style,
                        panel_label)

FIG_DIRS = [
    os.path.join(cfg.REPO_ROOT, "manuscript", "AMT_draft", "figures"),
    os.path.join(cfg.REPO_ROOT, "manuscript", "JQSRT_draft", "figures"),
]
# verification escape hatch: send the pair to one scratch directory instead.
# With it set, the results/rt_slab_sim side outputs (preview PNG and
# closure_stats.json) are left untouched, so a scratch run changes nothing else.
SCRATCH_RUN = bool(os.environ.get("SLAB_FIG_OUT_DIR"))
if SCRATCH_RUN:
    FIG_DIRS = [os.environ["SLAB_FIG_OUT_DIR"]]
# SLAB_FIG_WRITE_STATS=1 still writes the closure-stats JSON in a redirected run
# (2026-11-02: JQSRT-only regeneration, AMT tree left alone)
WRITE_STATS = (not SCRATCH_RUN) or os.environ.get("SLAB_FIG_WRITE_STATS") == "1"
OUT_BASE = "figG1_mc_3d_vs_ica"
# spectral estimator whose slab refit is drawn (see the module docstring)
ESTIMATOR = os.environ.get("SLAB_FIG_ESTIMATOR", "am")
IS_O2X = ESTIMATOR.startswith("o2x")          # "o2x" or "o2x_tmax<T>"
assert ESTIMATOR in ("am", "order7") or IS_O2X, f"SLAB_FIG_ESTIMATOR={ESTIMATOR!r}"
FIT_FILE = ({"am": "slab_fit_am.h5", "order7": "slab_fit.h5"}.get(ESTIMATOR)
            or f"slab_fit_{ESTIMATOR}.h5")
CLOSURE_FILE = ({"am": "closure_stats_am.json",
                 "order7": "closure_stats.json"}.get(ESTIMATOR)
                or f"closure_stats_{ESTIMATOR}.json")
VAR_FIELD = {"am": "k2diag"}.get(ESTIMATOR, "k2")   # fitted row (c,d)
# opt-in single-surface edition (2026-10-10, main-text figure with the dark surface only;
# the default two-column figure moves to the SI): SLAB_FIG_SURFACES=dark
SURFACES = [s for s in os.environ.get("SLAB_FIG_SURFACES", ",".join(cfg.SURFACE_ALBEDOS)).split(",") if s]
if ESTIMATOR != "am":
    OUT_BASE += "_" + ESTIMATOR
if len(SURFACES) < len(cfg.SURFACE_ALBEDOS):
    OUT_BASE += "_" + "_".join(SURFACES)
# opt-in row subset (2026-10-11, KSS review: the main text uses only <l'>, so the main
# figure drops the two variance rows): SLAB_FIG_ROWS=0,2,4 (0 fitted mean, 1 fitted var,
# 2 MC mean, 3 MC var, 4 reflectance; 2026-11-02, am only: 5 floor f, appended last
# so the panel letters of the default rows do not move). Default: rows 0-4.
# 2026-11-03: the am default drops both variance rows and adds the floor f
# below the fitted mean (author decision: no variance in the paper); the
# list order is the drawn order and the panel letters follow it.
DEFAULT_ROWS = [0, 5, 2, 4] if ESTIMATOR == "am" else [0, 1, 2, 3, 4]
_rows_env = os.environ.get("SLAB_FIG_ROWS")
ROWS = ([int(r) for r in _rows_env.split(",")] if _rows_env else DEFAULT_ROWS)
assert 5 not in ROWS or ESTIMATOR == "am", "row 5 (floor f) needs the am estimator"
NROW_ALL = 6
if ROWS != DEFAULT_ROWS:
    OUT_BASE += "_rows" + "".join(str(r) for r in ROWS)
# opt-in grid layout for the one-surface edition (2026-10-11, author: keep var(l') in the
# main-text figure): SLAB_FIG_LAYOUT=grid puts the means in the left column and the
# variances in the right (fitted on top, MC below) and gamma bottom left, with the
# legend in the free bottom-right cell. Needs one surface and all five rows.
LAYOUT = os.environ.get("SLAB_FIG_LAYOUT", "")
assert LAYOUT in ("", "grid", "stack"), f"SLAB_FIG_LAYOUT={LAYOUT!r}"
GRID = LAYOUT == "grid"
if GRID:
    assert len(SURFACES) == 1 and ROWS == [0, 1, 2, 3, 4], "grid layout: one surface, all rows"
    OUT_BASE += "_grid"
GRID_POS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0)}
# opt-in stacked layout (2026-11-03, main-text Fig. 3 without variance): one
# surface, the chosen rows in one column drawn at the printed width, so the
# 0.8 x textwidth include of the tex prints the fonts at native size
STACK = LAYOUT == "stack"
STACK_SIZE_IN = (5.2, 1.6)          # width, height per row
if STACK:
    assert len(SURFACES) == 1, "stack layout: one surface"
    OUT_BASE += "_stack"
OUT_BASE = os.environ.get("SLAB_FIG_STEM") or OUT_BASE


class _NullAx:
    """Stand-in for a dropped row: every plotting call is a no-op."""
    def __getattr__(self, name):
        return lambda *a, **k: None

COL_3D = "#c1272d"     # 3-D
COL_ICA = "#0000a7"    # IPA

# one y range per row, shared by the dark and bright columns
# o2x family: k2 spans -0.005..0.14 on all tau, 0.05..0.40 with tau <= 2
_O2X_K2_YLIM = (-0.02, 0.16) if ESTIMATOR == "o2x" else (0.0, 0.45)
ROW_YLIM = {
    "o2x": ((0.3, 1.3), _O2X_K2_YLIM, (0.9, 1.04), (0.0, 0.035), (0.0, 0.8)),
    "order7": ((0.4, 1.3),      # fitted <l'>
               (0.0, 0.8),      # fitted var(l')
               (0.9, 1.04),     # MC mean
               (0.0, 0.035),    # MC variance
               (0.0, 0.8)),     # effective reflectance
    # am: k1 reaches 0.37 in the cloud (order 7: 0.41) and tops out at 0.97;
    # k2diag spans -0.005..0.12; f spans 0..0.2 (upper bound _AM_F_MAX)
    "am": ((0.3, 1.3),          # fitted <l'> = k1 (top as order7: keeps labels clear)
           (-0.02, 0.14),       # k2diag
           (0.9, 1.04),         # MC mean
           (0.0, 0.035),        # MC variance
           (0.0, 0.8),          # effective reflectance
           (0.0, 0.22)),        # floor f
}["o2x" if IS_O2X else ESTIMATOR]
ROW_YTICK = {"order7": (0.2, 0.2, 0.025, 0.01, 0.2),
             "am": (0.2, 0.04, 0.025, 0.01, 0.2, 0.05),
             "o2x": (0.2, 0.04 if ESTIMATOR == "o2x" else 0.1, 0.025, 0.01, 0.2),
             }["o2x" if IS_O2X else ESTIMATOR]


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

    with h5py.File(os.path.join(cfg.OUT_DIR, FIT_FILE), "r") as f:
        assert f.attrs.get("estimator", "order7") == ESTIMATOR, FIT_FILE
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
    if ESTIMATOR != "order7":
        # order7 keeps the historical key set of closure_stats.json
        stats["estimator"] = ESTIMATOR
        stats["fit_file"] = FIT_FILE
    for s in cfg.SURFACE_ALBEDOS:
        k1 = fit[f"{s}/3d"]["k1"]
        k2 = fit[f"{s}/3d"][VAR_FIELD]
        m3, sd3 = mom[f"{s}/3d"]
        ok = np.isfinite(k1) & np.isfinite(m3)
        stats[f"r_k1_meanL_{s}"] = float(np.corrcoef(k1[ok], m3[ok])[0, 1])
        okv = np.isfinite(k2) & np.isfinite(sd3)
        stats[f"r_{VAR_FIELD}_varL_{s}"] = float(
            np.corrcoef(k2[okv], sd3[okv]**2)[0, 1])
        # clear-column (adjacency) closure: in-cloud columns excluded, where
        # geometric spread narrows (cloud-top bounce) while the
        # absorption-weighted spread the fit senses broadens (in-cloud
        # multiple scattering) -- the geometric tally is only a shape proxy
        # there
        clr = ok & ~in_cloud
        stats[f"r_k1_meanL_clear_{s}"] = float(
            np.corrcoef(k1[clr], m3[clr])[0, 1])
        clrv = okv & ~in_cloud
        stats[f"r_{VAR_FIELD}_varL_clear_{s}"] = float(
            np.corrcoef(k2[clrv], sd3[clrv]**2)[0, 1])
        if ESTIMATOR == "am":
            # (1 - f) k1 is the initial slope -dlnT/dtau at tau -> 0 of the
            # floor model, i.e. the mean path of the two-population PPDF
            # {(1 - f) at k1, f at 0}; reported beside k1 for comparison
            k1e = (1.0 - fit[f"{s}/3d"]["f"]) * k1
            stats[f"r_k1eff_meanL_{s}"] = float(np.corrcoef(k1e[ok], m3[ok])[0, 1])
            stats[f"r_k1eff_meanL_clear_{s}"] = float(
                np.corrcoef(k1e[clr], m3[clr])[0, 1])
        # ICA null residuals outside the cloud (fraction of the 3-D range)
        for name in ("k1", VAR_FIELD) + (("f",) if ESTIMATOR == "am" else ()):
            v_ica = fit[f"{s}/ipa"][name]
            resid = np.nanmax(np.abs(v_ica[~in_cloud]
                                     - np.nanmedian(v_ica[far])))
            rng = np.nanmax(fit[f"{s}/3d"][name]) - np.nanmin(fit[f"{s}/3d"][name])
            stats[f"ica_null_{name}_{s}"] = float(resid)
            stats[f"ica_null_{name}_frac_{s}"] = float(resid / rng)

    if ESTIMATOR == "am":
        # 2026-11-03: the profile values the Sect. 3 / S6 text quotes (3-D run).
        # far = median over x < 5 km; shadow = the column of the shadow band
        # x = 15-22 km farthest from the far value (the extremum, whichever its
        # sign: the bright-surface MC mean rises while k1 falls); sunlit edge =
        # the last clear column before the cloud (x just below CLOUD_X_KM[0]).
        shadow = (x_km >= 15.0) & (x_km <= 22.0)
        i_edge = int(np.flatnonzero(x_km < cfg.CLOUD_X_KM[0])[-1])
        stats["profile_definitions"] = (
            "far: median over x < 5 km; shadow: value farthest from far over "
            "15 <= x <= 22 km; edge: column at x = %.2f km (last clear column "
            "on the sunlit side); MC mean in relative-path units "
            "1 + (L - L_peak)/l_direct as drawn" % x_km[i_edge])
        for s in cfg.SURFACE_ALBEDOS:
            res = fit[f"{s}/3d"]
            m3 = 1.0 + (mom[f"{s}/3d"][0] - L_peak) / l_direct
            prof = {"k1": res["k1"], "f": res["f"],
                    "gamma": np.exp(res["intercept"]), "mcmean": m3}
            for name, v in prof.items():
                v_far = float(np.nanmedian(v[far]))
                i_sh = int(np.nanargmax(np.abs(v[shadow] - v_far)))
                stats[f"{name}_far_{s}"] = v_far
                stats[f"{name}_shadow_{s}"] = float(v[shadow][i_sh])
                stats[f"{name}_shadow_x_{s}"] = float(x_km[shadow][i_sh])
                stats[f"{name}_edge_{s}"] = float(v[i_edge])
        stats["f_max_bound"] = float(_AM_F_MAX)

    if WRITE_STATS:
        with open(os.path.join(cfg.OUT_DIR, CLOSURE_FILE), "w") as fj:
            json.dump(stats, fj, indent=2)
    print(json.dumps(stats, indent=2))

    # ---------------- figure ----------------
    ncol = len(SURFACES)
    nrow = len(ROWS)
    # 2026-10-11 print-size redraw: drawn at (about) the printed width so fonts print at
    # native size (axis labels 10 pt, ticks 9.5 pt, legend 9 pt, annotations 9 pt).
    # Two columns (SI): 6.2 in wide, 1.5 in per row; one column (main text, rows 0,2,4):
    # 4.0 in wide, 1.95 in per row.  Was 3.9 in per column x 2.16 in per row.
    figsize = ((6.2, 1.5 * nrow + 0.1) if ncol > 1 else (4.0, 2.0 * nrow + 0.2))
    if STACK:
        figsize = (STACK_SIZE_IN[0], STACK_SIZE_IN[1] * nrow + 0.25)
    if GRID:
        # 3 x 2 grid at print width (6.5 in), 1.75 in per row
        fig, grid = plt.subplots(3, 2, figsize=(6.5, 5.6), sharex=True, squeeze=False)
        axes = np.empty((NROW_ALL, 1), dtype=object)
        axes[5, 0] = _NullAx()
        for jr, (r, c) in GRID_POS.items():
            axes[jr, 0] = grid[r, c]
        grid[2, 1].axis("off")
        axes_real = np.array([[grid[2, 0]]])          # x label goes under gamma only
        grid[1, 1].tick_params(labelbottom=True)
        grid[1, 1].set_xlabel("Along-slab distance $x$ (km)")
    else:
        fig, axes_real = plt.subplots(nrow, ncol, figsize=figsize,
                                      sharex=True, squeeze=False)
        axes = np.empty((NROW_ALL, ncol), dtype=object)
        for jr in range(NROW_ALL):
            for jc in range(ncol):
                axes[jr, jc] = axes_real[ROWS.index(jr), jc] if jr in ROWS else _NullAx()
    tags = "abcdefghijkl"

    for jc, s in enumerate(SURFACES):
        for v, col, lbl in (("3d", COL_3D, "3-D"), ("ipa", COL_ICA, "IPA")):
            res = fit[f"{s}/{v}"]
            for jr, name in enumerate(("k1", VAR_FIELD)):
                ax = axes[jr, jc]
                # k2 / k2diag dashed in both columns, matching the variance
                # styling of the MC row below it
                ax.plot(x_km, res[name], color=col, lw=1.3,
                        ls="--" if name == VAR_FIELD else "-", label=lbl)
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
            if ESTIMATOR == "am":
                axes[5, jc].plot(x_km, res["f"], color=col, lw=1.3, label=lbl)
        if ESTIMATOR == "am":
            # upper bound of the floor in the fit (f at it = not constrained)
            axes[5, jc].axhline(_AM_F_MAX, color="0.35", lw=0.7, ls="--",
                                zorder=1)

        alb = cfg.SURFACE_ALBEDOS[s]
        if not (GRID or STACK):   # grid / stack editions: surface named in the caption
            axes[0, jc].set_title(
                ("Dark" if s == "dark" else "Bright")
                + f" surface (albedo {alb:.2f})", fontsize=10)
        axes_real[-1, jc].set_xlabel("Along-slab distance $x$ (km)")

    # MC rows reuse the plot_style l' labels with an "MC" prefix so the
    # tallied moments read as the same quantities the fit estimates
    # am row (c,d): Fig. 5 AM_ROWS label of the free-k2 diagnostic
    var_label = ("$k_2^{\\mathrm{diag}}$" if ESTIMATOR == "am" else VAR_L_LABEL)
    row_labels = (MEAN_L_LABEL, var_label,
                  "MC " + MEAN_L_LABEL,
                  "MC " + VAR_L_LABEL,
                  # two lines in the 1.5 in rows of the two-column edition
                  ("Continuum\nreflectance $\\gamma$" if ncol > 1 or STACK
                   else "Continuum reflectance $\\gamma$"),
                  "Floor $f$")
    for kr, jr in enumerate(ROWS):
        for jc in range(ncol):
            ax = axes[jr, jc]
            ax.axvspan(*cfg.CLOUD_X_KM, color="0.88", zorder=0)
            ax.set_ylim(*ROW_YLIM[jr])
            # same tick steps in both editions (fixed, so the SI and main-text
            # figures read alike at their different panel heights)
            ax.yaxis.set_major_locator(MultipleLocator(ROW_YTICK[jr]))
            panel_label(ax, f"({tags[jr if GRID else kr * ncol + jc]})")
            if jc == 0:
                ax.set_ylabel(row_labels[jr])
    # two-column edition: the shorter panel (a) has no room for the legend next to the
    # "shadow" label, so the legend moves to the empty upper right
    if GRID:
        # legend in the free bottom-right cell, solid = mean rows, dashed = variance rows
        h, l = axes[0, 0].get_legend_handles_labels()
        grid[2, 1].legend(h, ["3D", "IPA"], loc="center", fontsize=10, frameon=False,
                          handlelength=2.2)
    else:
        # "3D" as in the text: stack edition (main-text Fig. 3, as the grid
        # edition it replaces) and, since 2026-11-03, the am two-surface SI
        # edition; the other editions keep "3-D" so they re-render unchanged
        use_3d = STACK or (ncol > 1 and ESTIMATOR == "am")
        h, l = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(h, ["3D", "IPA"] if use_3d else l,
                          fontsize=9, loc="upper right" if ncol > 1 else "lower right",
                          frameon=False,
                          handlelength=1.6, borderaxespad=0.3)

    # sun direction (tilted by the SZA, pointing down-sun toward +x) +
    # shadow annotation on the top-left panel
    ax = axes[0, 0]
    dx, dy = 0.10 * np.sin(np.deg2rad(sza)), 0.10 * np.cos(np.deg2rad(sza))
    ax.annotate("", xy=(0.05 + dx, 0.97 - dy), xytext=(0.05, 0.97),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.05 + dx + 0.02, 0.99, f"sun\nSZA {sza:.0f}$^\\circ$",
            transform=ax.transAxes, fontsize=9, va="top")
    # two-column edition: panel (a) is too short for "shadow" inside the dip, so it sits
    # above the shadow at the height of the "cloud" label
    # grid edition: both labels sit above the far-field line, clear of the steep dip edge
    ax.text(18.6 if ncol > 1 else 18.5,
            0.91 if ncol > 1 else (0.88 if GRID or STACK else 0.55), "shadow", fontsize=9,
            ha="center", color="0.3")
    ax.text(12.0, 0.91 if ncol > 1 else 0.88, "cloud", fontsize=9, ha="center", color="0.3")

    fig.tight_layout()
    if GRID:
        fig.subplots_adjust(hspace=0.25, wspace=0.32)
    else:
        fig.subplots_adjust(hspace=0.30 if ncol > 1 else (0.25 if STACK else 0.22),
                            wspace=0.30)

    for fig_dir in FIG_DIRS:
        os.makedirs(fig_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"{OUT_BASE}.{ext}"))
    prev = os.path.join(cfg.OUT_DIR, "figs", f"{OUT_BASE}.png")
    if not SCRATCH_RUN:
        fig.savefig(prev)
    print(f"Wrote {OUT_BASE}.png/.pdf into "
          + ", ".join(FIG_DIRS) + ("" if SCRATCH_RUN else f" (+ preview {prev})"))


if __name__ == "__main__":
    main()
