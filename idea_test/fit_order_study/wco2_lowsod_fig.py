"""Figures for the WCO2 low-SOD study (29252a_GL, 2026-10-30).

(1) results/wco2_lowsod_patterns_29252a_GL.png: far-clear mean residual of the revised WCO2 fit per channel for
    footprints 1 and 4 in a 150-channel window: baseline residual for the two v_inst clusters (opposite signs),
    the same after the solar-Doppler correction, the predicted solar error −v̄·∂lnS/∂v, the solar-line depth and
    the median SOD (each in its own panel, shared wavelength axis).
(2) results/wco2_lowsod_example_<sid>.png: ln T − continuum − H2O terms against SOD for one sounding, revised
    fit as is (L1B-flagged channels marked) and after the solar-Doppler correction with flagged channels dropped.

Needs results/wco2_lowsod_patterns_29252a_GL.npz and wco2_lowsod_solar_grid_GL.npz.
Usage: python wco2_lowsod_fig.py [sounding_id]
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
from wco2_lowsod_analysis import solar_delta  # noqa: E402

OUT = wc.OUT
SID = int(sys.argv[1]) if len(sys.argv) > 1 else 2020010100214501
INK, MUTED, GRID = "#222222", "#8a8a8a", "#e6e6e6"
C1, C2 = "#2a78d6", "#eb6834"          # categorical slots 1, 2 (v < 0, v > 0 cluster)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def fig_patterns():
    z = np.load(OUT / "wco2_lowsod_patterns_29252a_GL.npz")
    P0v, P1, depth, dlnS, sod, wl, vbar = (z[k] for k in ("P0v", "P1", "depth", "dlnS_dv", "sod_med", "wl", "vbar"))
    fps = (0, 3)
    # window: 150 channels with the largest baseline pattern variance at low SOD (footprint 1)
    # (unflagged interior channels 90–1005; the detector edge and the last channels carry isolated outliers)
    w = np.nan_to_num(P0v[0, 0] ** 2) * (sod[0] < 0.1)
    w[:90] = 0
    w[1005:] = 0
    score = np.convolve(w, np.ones(150), "valid")
    c0 = int(np.clip(np.argmax(score), 90, 1005 - 150))
    sl = slice(c0, c0 + 150)
    fig, axes = plt.subplots(4, 2, figsize=(12, 8.2), sharex="col", constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[2.2, 2.2, 1, 1]))
    for col, f in enumerate(fps):
        x = wl[f, sl] * 1000                                    # nm
        ax = axes[0, col]
        ax.plot(x, P0v[f, 0, sl], color=C1, lw=1.2, label=f"v_inst ≈ {vbar[0] / 1000:+.1f} km/s")
        ax.plot(x, P0v[f, 1, sl], color=C2, lw=1.2, label=f"v_inst ≈ {vbar[1] / 1000:+.1f} km/s")
        ax.set_title(f"Footprint {f + 1}: revised fit as is (production TOA solar)", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax = axes[1, col]
        ax.plot(x, P1[f, sl], color=INK, lw=1.2, label="after solar-Doppler correction (both clusters)")
        ax.set_title("same, solar lines shifted by −v_inst (no fitted parameter)", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        lim = np.nanmax(np.abs(P0v[f, :, sl])) * 1.05
        for a in axes[:2, col]:
            a.set_ylim(-lim, lim)
        axes[2, col].fill_between(x, 0, 100 * depth[f, sl], color=MUTED, lw=0)
        axes[3, col].plot(x, sod[f, sl], color=INK, lw=1)
        axes[3, col].axhline(0.1, color=C2, lw=0.8, ls="--")
        axes[3, col].set_xlabel("wavelength (nm)", color=INK)
    axes[0, 0].set_ylabel("mean residual\nln T (far-clear)", color=INK)
    axes[1, 0].set_ylabel("mean residual\nln T (far-clear)", color=INK)
    axes[2, 0].set_ylabel("solar-line\ndepth (%)", color=INK)
    axes[3, 0].set_ylabel("median SOD", color=INK)
    for ax in axes.ravel():
        style(ax)
    fig.suptitle("WCO$_2$ far-clear residual pattern, 29252a_GL: the low-SOD structure flips sign with the "
                 "spacecraft velocity and follows the solar lines", fontsize=11, color=INK)
    out = OUT / "wco2_lowsod_patterns_29252a_GL.png"
    fig.savefig(out, dpi=150)
    print(out)


def fig_example():
    orb, tau, tau_h2o, meta, _ = wc.fm.load("GL")
    j = int(np.flatnonzero(orb.sounding_id == SID)[0])
    fp = orb.fp_number.astype(int)
    ex = wc.l1b_extras(wc.fm.DEFAULTS["l1b"], orb.sounding_id[[j]], fp[[j]], spikes=False)
    grid = np.load(OUT / "wco2_lowsod_solar_grid_GL.npz")["lnS"].astype(float)
    d = dict(orb=type("o", (), dict(fp_number=fp[[j]]))(), ex=ex, grid=grid)
    dS = solar_delta(d, 1, 1.0)[0]
    bad = ex["bad"][1][fp[j]]
    y0 = orb.ln_t[1, j].astype(float)
    t, tw, s = tau[1, j].astype(float), tau_h2o[1, j].astype(float), orb.sigma_ln_t[1, j].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, (lab, y, drop) in zip(axes, (("revised fit as is", y0, False),
                                         ("solar-Doppler corrected, L1B-flagged channels dropped", y0 + dS, True))):
        ok = wc.base_mask(y, t, tw, s) & ((bad == 0) if drop else True)
        c, r = wc.fit_wco2_ext(t[ok] - tw[ok], tw[ok], wc.X_ALL[ok], y[ok])
        k1, k1w, a1 = c[2], c[3], c[1]
        yadj = y[ok] - a1 * wc.X_ALL[ok] - (k1 - k1w) * tw[ok]
        tm = t[ok]
        flag = bad[ok] != 0
        ax.scatter(tm[~flag], yadj[~flag], s=5, c=MUTED, alpha=0.7, label="channels")
        if flag.any():
            ax.scatter(tm[flag], yadj[flag], s=7, c=C2, alpha=0.9, label="L1B bad_sample_list ≠ 0")
        tt = np.linspace(0, tm.max(), 200)
        ax.plot(tt, c[0] - k1 * tt, color=C1, lw=2, label="fit")
        lo = tm < 0.1
        ax.text(0.03, 0.05, f"k1 = {k1:.4f}, k1_w = {k1w:.3f}, a1 = {a1:+.4f}\nrms SOD<0.1 = {np.sqrt(np.mean(r[lo] ** 2)):.4f}, "
                f"all = {np.sqrt(np.mean(r ** 2)):.4f}\nnoise σ ≈ {np.median(s[ok][lo]):.4f}", transform=ax.transAxes, fontsize=8.5,
                color=INK, bbox=dict(fc="white", ec="none", alpha=0.85))
        ax.set_title(lab, fontsize=10, color=INK)
        ax.set_xlabel("channel slant optical depth SOD", color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        style(ax)
    axes[0].set_ylabel("ln T − continuum − H$_2$O terms", color=INK)
    fig.suptitle(f"WCO$_2$, sounding {SID} (fp {fp[j] + 1}, v_inst {ex['v_inst'][0] / 1000:+.2f} km/s, "
                 f"{float(meta['cld_dist_km'][j]):.1f} km from cloud)", fontsize=11, color=INK)
    out = OUT / f"wco2_lowsod_example_{SID}.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    fig_patterns()
    fig_example()
