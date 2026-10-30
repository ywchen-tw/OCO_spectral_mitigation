"""One-sounding comparison: production fit (production τ, polynomial order 7/3/7) vs the agreed model
(Doppler-corrected two-τ, floor in O2A, continuum terms), as ln T against the channel slant optical depth.

Top row: production. Points = measured ln T against the production SOD; curve = order-N polynomial fitted
here with the production convention (intercept, (−1)^n k_n/n τ^n), channels 1:-1, finite ln T.
Bottom row: agreed model. Points = measured ln T with the fitted continuum slope and H2O-scaling terms
removed, against the corrected SOD, so that the remaining model is a function of SOD alone; curve = that
function. Dashed = the same without the floor (O2A) or without the quadratic term (SCO2).
Default sounding = the manuscript Fig. example 2020010100214501 (29252a, 0.9 km from cloud).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_model_test as fm  # noqa: E402
from final_model_test import X_ALL  # noqa: E402
from orbit_io import BANDS  # noqa: E402

OUT = HERE / "results"
SID = int(sys.argv[1]) if len(sys.argv) > 1 else 2020010100214501
ORDER = {"o2a": 7, "wco2": 3, "sco2": 7}
LABEL = {"o2a": "O$_2$ A-band", "wco2": "WCO$_2$", "sco2": "SCO$_2$"}
INK, MUTED, C_PROD, C_NEW = "#222222", "#8a8a8a", "#d95f02", "#1f5fa8"


def production_fit(t, y, n):
    ok = np.isfinite(y) & np.isfinite(t)
    ok[[0, -1]] = False
    A = np.column_stack([np.ones(ok.sum())] + [((-1.0) ** k) * t[ok] ** k / k for k in range(1, n + 1)])
    c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    pred = A @ c
    return ok, c, float(np.sqrt(np.mean((y[ok] - pred) ** 2)))


def poly_curve(c, t):
    return c[0] + sum(((-1.0) ** k) * c[k] * t**k / k for k in range(1, len(c)))


def main():
    orb, tau, tau_h2o, meta, tag = fm.load("GL")
    j = int(np.flatnonzero(orb.sounding_id == SID)[0])
    land = bool(meta["sfc_type"][j] == 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for ib, b in enumerate(BANDS):
        y = orb.ln_t[ib, j].astype(float)
        s = orb.sigma_ln_t[ib, j].astype(float)
        # --- production
        tp = orb.tau[ib, j].astype(float)
        ok, c, rms = production_fit(tp, y, ORDER[b])
        ax = axes[0, ib]
        ax.scatter(tp[ok], y[ok], s=5, c=MUTED, alpha=0.6, label="measured ln T")
        tt = np.linspace(0, np.nanmax(tp[ok]), 300)
        ax.plot(tt, poly_curve(c, tt), color=C_PROD, lw=2, label=f"order {ORDER[b]} fit")
        ax.set_title(f"{LABEL[b]}: production fit, order {ORDER[b]}", color=INK, fontsize=11)
        ax.text(0.03, 0.05, f"k1 = {c[1]:.3f}\nk2 = {c[2]:+.3f}\nrms = {rms:.4f}", transform=ax.transAxes, va="bottom",
                fontsize=9, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))
        # --- agreed
        t = tau[ib, j].astype(float)
        tw = tau_h2o[ib, j].astype(float)
        ok2 = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0) & np.isfinite(tw)
        tm, twm, ym, xm = t[ok2], tw[ok2], y[ok2], X_ALL[ok2]
        ax = axes[1, ib]
        tt = np.linspace(0, tm.max(), 300)
        if b == "o2a":
            r = fm.fit_o2a(tm, xm, ym)
            y_adj = ym - r["a1"] * xm
            curve = r["c"] + np.log((1 - r["f"]) * np.exp(-r["k1"] * tt) + r["f"])
            dashed = r["c"] - r["k1"] * tt
            note = f"k1 = {r['k1']:.3f}\nf = {r['f']:.4f} (k2 = 0)\na1 = {r['a1']:+.3f}\nrms = {r['rms']:.4f}"
            dlab = "same without floor"
        elif b == "wco2":
            r = fm.fit_wco2(tm - twm, twm, xm, ym)
            y_adj = ym - r["a1"] * xm - (r["k1"] - r["k1_w"]) * twm
            curve = r["c"] - r["k1"] * tt
            dashed = None
            note = f"k1 = {r['k1']:.3f}, k1_w = {r['k1_w']:.3f}\nk2 = 0\na1 = {r['a1']:+.3f}\nrms = {r['rms']:.4f}"
        else:
            r = fm.fit_sco2(tm - twm, twm, xm, ym, land)
            y_adj = ym - r["a1"] * xm - (r["a2"] * xm**2 if land else 0) - (r["k1"] - r["k1_w"]) * twm
            curve = r["c"] - r["k1"] * tt + 0.5 * r["k2"] * tt**2
            dashed = r["c"] - r["k1"] * tt
            note = f"k1 = {r['k1']:.3f}, k1_w = {r['k1_w']:.3f}\nk2 = {r['k2']:+.3f}\na1 = {r['a1']:+.3f}\nrms = {r['rms']:.4f}"
            dlab = "same without ½k2τ²"
        ax.scatter(tm, y_adj, s=5, c=MUTED, alpha=0.6, label="ln T − continuum − H$_2$O terms")
        ax.plot(tt, curve, color=C_NEW, lw=2, label="revised fit")
        if dashed is not None:
            ax.plot(tt, dashed, color=C_NEW, lw=1.2, ls="--", label=dlab)
        ax.set_title(f"{LABEL[b]}: revised fit, " + ("floor term, k2 = 0" if b == "o2a" else "dry/H$_2$O split" + (", k2 = 0" if b == "wco2" else ", free k2")), color=INK, fontsize=11)
        ax.text(0.03, 0.05, note, transform=ax.transAxes, va="bottom", fontsize=9, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    for ax in axes.ravel():
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK, labelsize=9)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("channel slant optical depth SOD", color=INK)
    axes[0, 0].set_ylabel("production: τ uncorrected\nln T", color=INK)
    axes[1, 0].set_ylabel("revised: τ Doppler-corrected\nln T − continuum − H$_2$O terms", color=INK)
    fig.suptitle(f"Sounding {SID} (granule 29252a, {float(meta['cld_dist_km'][j]):.1f} km from cloud, fp {int(meta['fp_number'][j]) + 1}, "
                 f"QF {int(meta['xco2_qf'][j])}): production fit vs revised fit", color=INK)
    out = OUT / f"fit_comparison_{SID}.png"
    fig.savefig(out, dpi=160)
    print(out)
    # reproduction check of the stored production k1
    import h5py
    with h5py.File(fm.DEFAULTS["fit_details"]) as f:
        print("stored production k1 nosg:", {b: float(f[f"{b}_k1_fitting_nosg"][j]) for b in BANDS})


if __name__ == "__main__":
    main()
