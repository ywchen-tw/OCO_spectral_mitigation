"""All-band example figures for the solar-Doppler fix (29252a_GL, one sounding), 2026-10-30.

(1) results/wco2_lowsod_example_allbands_<sid>.png: 3 rows (O2A, WCO2, SCO2) × 2 columns (revised fit as is;
    solar-Doppler corrected, α = 1). Same channel mask in both (finite ln T, τ, σ > 0, τ_h2o), no channel dropping.
    Points: ln T with the fitted continuum slope and H2O-scaling terms removed (O2A y − a1·x; CO2 bands
    y − a1·x [− a2·x² over land] − (k1 − k1_w)·τ_h2o) against the Doppler-corrected SOD; line: the fitted model.
(2) results/wco2_lowsod_residual_allbands_<sid>.png: residual (data − fit) against channel index, before (grey) and
    after (blue) the solar fix, with the convolved solar spectrum ln S (scaled and offset, light orange) overlaid.
Usage: python wco2_lowsod_fig_allbands.py [sounding_id]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wco2_lowsod_common as wc  # noqa: E402
from wco2_lowsod_fig import C1, GRID, INK, MUTED, style  # noqa: E402

OUT = wc.OUT
SID = int(sys.argv[1]) if len(sys.argv) > 1 else 2020010100214501
C_SOL = "#f5b58f"                      # light orange (slot-2 hue, lightened) for the solar spectrum
LABEL = {0: "O$_2$ A-band", 1: "WCO$_2$", 2: "SCO$_2$"}
fm = wc.fm


def fit_band(ib, y, t, tw, x, land):
    """Revised-fit model for one band → (coefs dict, model at channels, y_adj, curve function of SOD)."""
    if ib == 0:
        r = fm.fit_o2a(t, x, y)
        model = r["c"] + r["a1"] * x + np.log((1 - r["f"]) * np.exp(-r["k1"] * t) + r["f"])
        yadj = y - r["a1"] * x
        curve = lambda tt: r["c"] + np.log((1 - r["f"]) * np.exp(-r["k1"] * tt) + r["f"])  # noqa: E731
    elif ib == 1:
        r = fm.fit_wco2(t - tw, tw, x, y)
        model = r["c"] + r["a1"] * x - r["k1"] * (t - tw) - r["k1_w"] * tw
        yadj = y - r["a1"] * x - (r["k1"] - r["k1_w"]) * tw
        curve = lambda tt: r["c"] - r["k1"] * tt  # noqa: E731
    else:
        r = fm.fit_sco2(t - tw, tw, x, y, land)
        cont = r["a1"] * x + (r["a2"] * x**2 if land else 0)
        model = r["c"] + cont - r["k1"] * (t - tw) - r["k1_w"] * tw + 0.5 * r["k2"] * t**2
        yadj = y - cont - (r["k1"] - r["k1_w"]) * tw
        curve = lambda tt: r["c"] - r["k1"] * tt + 0.5 * r["k2"] * tt**2  # noqa: E731
    return r, model, yadj, curve


def main():
    orb, tau, tau_h2o, meta, _ = fm.load("GL")
    j = int(np.flatnonzero(orb.sounding_id == SID)[0])
    fp = int(orb.fp_number[j])
    land = bool(meta["sfc_type"][j] == 1)
    ex = wc.l1b_extras(fm.DEFAULTS["l1b"], orb.sounding_id[[j]], orb.fp_number[[j]], spikes=False)
    vs, vi = ex["v_sol"][0], ex["v_inst"][0]
    grid = np.load(OUT / "wco2_lowsod_solar_grid_GL.npz")["lnS"].astype(float)
    fig1, ax1 = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    fig2, ax2 = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    rows = []
    for ib in range(3):
        lnS_prod = wc.interp_v(grid[ib, fp], vs)
        dS = lnS_prod - wc.interp_v(grid[ib, fp], vs - vi)
        y0 = orb.ln_t[ib, j].astype(float)
        t, tw, s = tau[ib, j].astype(float), tau_h2o[ib, j].astype(float), orb.sigma_ln_t[ib, j].astype(float)
        ok = wc.base_mask(y0, t, tw, s)
        x = wc.X_ALL[ok]
        lo = t[ok] < wc.LOW_SOD
        res = {}
        ylims = []
        for col, (lab, y) in enumerate((("revised fit as is", y0), ("solar-Doppler corrected (α = 1)", y0 + dS))):
            r, model, yadj, curve = fit_band(ib, y[ok], t[ok], tw[ok], x, land)
            rr = y[ok] - model
            res[col] = rr
            rms_lo, rms_all = np.sqrt(np.mean(rr[lo] ** 2)), np.sqrt(np.mean(rr**2))
            rows.append(dict(band=("o2a", "wco2", "sco2")[ib], panel=lab, k1=r["k1"], rms_low=rms_lo, rms_all=rms_all,
                             n_low=int(lo.sum()), sigma=float(np.median(s[ok][lo])) if lo.any() else np.nan))
            ax = ax1[ib, col]
            ax.scatter(t[ok], yadj, s=4, c=MUTED, alpha=0.6, lw=0, label="channels")
            tt = np.linspace(0, t[ok].max(), 300)
            ax.plot(tt, curve(tt), color=C1, lw=2, label="fit")
            note = f"k1 = {r['k1']:.4f}"
            if ib == 0:
                note += f", f = {r['f']:.4f}"
            else:
                note += f", k1_w = {r['k1_w']:.3f}"
                if ib == 2:
                    note += f"\nk2 = {r['k2']:+.4f}"
            note += (f"\nrms SOD<0.1 = {rms_lo:.4f} ({lo.sum()} ch.)\nrms all = {rms_all:.4f}"
                     f"\nnoise σ (SOD<0.1) ≈ {rows[-1]['sigma']:.4f}")
            ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.85))
            ax.set_title(f"{LABEL[ib]}: {lab}", fontsize=10, color=INK)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ylims.append(ax.get_ylim())
            style(ax)
        lo_y, hi_y = min(a for a, _ in ylims), max(b for _, b in ylims)
        for ax in ax1[ib]:
            ax.set_ylim(lo_y, hi_y)
        ax1[ib, 0].set_ylabel("ln T − continuum" + ("" if ib == 0 else " − H$_2$O terms"), color=INK)
        # residual vs channel index
        ax = ax2[ib]
        ch = np.flatnonzero(ok)
        amp = np.nanpercentile(np.abs(res[0]), 99)
        sol = lnS_prod[ok] - np.nanmax(lnS_prod[ok])          # ≤ 0, 0 at the continuum
        sol_scaled = sol / max(np.nanmax(-sol), 1e-9) * amp - 0.0   # deepest solar line → −amp
        ax.plot(ch, sol_scaled - 1.15 * amp, color=C_SOL, lw=1.0, label="ln S convolved (scaled, offset)")
        ax.plot(ch, res[0], color=MUTED, lw=0.7, label=f"revised fit as is (rms {np.sqrt(np.mean(res[0] ** 2)):.4f})")
        ax.plot(ch, res[1], color=C1, lw=0.7, label=f"solar-Doppler corrected (rms {np.sqrt(np.mean(res[1] ** 2)):.4f})")
        ax.set_title(f"{LABEL[ib]}: residual (data − fit)", fontsize=10, color=INK)
        ax.set_ylabel("residual ln T", color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
        style(ax)
    ax2[-1].set_xlabel("channel index", color=INK)
    for ax in ax1[-1]:
        ax.set_xlabel("channel slant optical depth SOD (Doppler-corrected τ)", color=INK)
    ttl = (f"Sounding {SID} (29252a_GL, fp {fp + 1}, v_inst {vi / 1000:+.2f} km/s, "
           f"{float(meta['cld_dist_km'][j]):.1f} km from cloud)")
    fig1.suptitle(ttl + ": revised fit as is vs solar-Doppler corrected", fontsize=11, color=INK)
    fig2.suptitle(ttl + ": residual before and after the solar-Doppler fix", fontsize=11, color=INK)
    p1 = OUT / f"wco2_lowsod_example_allbands_{SID}.png"
    p2 = OUT / f"wco2_lowsod_residual_allbands_{SID}.png"
    fig1.savefig(p1, dpi=150)
    fig2.savefig(p2, dpi=150)
    print(p1, p2, sep="\n")
    for r in rows:
        print(f"{r['band']:5s} {r['panel']:34s} k1 {r['k1']:.4f} rms<0.1 {r['rms_low']:.4f} (n {r['n_low']}) all {r['rms_all']:.4f} σ {r['sigma']:.4f}")


if __name__ == "__main__":
    main()
