"""One far-clear sounding, three bands: ln T vs production τ and vs Doppler-corrected τ,
with order-2 / order-3 / gamma fits on the corrected τ and their residuals.

Uses the corrected τ stored by resid_doppler.py (results/resid_doppler_tau_<orbit>.npz).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from variants import design_matrix, fit_cumulant, fit_gamma  # noqa: E402

OUT = HERE / "results"
TAG = "29252a_GL"
TARGET_V_KMS = 4.0          # pick the far-clear sounding closest to this |v|
BAND_LABEL = {"o2a": "O2A", "wco2": "WCO2", "sco2": "SCO2"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cld-km", type=float, default=None, help="pick a near-cloud sounding (cld_dist < this) instead of far-clear")
    ap.add_argument("--target-v", type=float, default=TARGET_V_KMS)
    args = ap.parse_args()
    orb = load_orbit(str(DEFAULTS["l1b"]), str(DEFAULTS["fp_tau"]), str(DEFAULTS["fit_details"]))
    z = np.load(OUT / f"resid_doppler_tau_{TAG}.npz")
    assert np.array_equal(z["j"], np.arange(orb.tau.shape[1]))
    tau_corr = z["tau"]                                   # [N, 3, 1016]
    with h5py.File(DEFAULTS["l1b"]) as f:
        sid = f["SoundingGeometry/sounding_id"][...].astype(np.int64).ravel()
        v = f["SoundingGeometry/sounding_relative_velocity"][...].ravel() / 1e3
    vmap = dict(zip(sid.tolist(), v.tolist()))
    v_orb = np.array([vmap.get(int(s), np.nan) for s in orb.sounding_id])
    if args.max_cld_km is None:
        sel = (orb.meta["cld_dist_km"] > 20) & (orb.meta["xco2_qf"] == 0) & np.isfinite(v_orb)
        label, suffix = "far-clear", ""
    else:
        sel = (orb.meta["cld_dist_km"] < args.max_cld_km) & np.isfinite(v_orb)
        label, suffix = f"near cloud ({orb.meta['cld_dist_km'].min():.1f} km)", "_near"
    cand = np.flatnonzero(sel)
    j = cand[np.argmin(np.abs(np.abs(v_orb[cand]) - args.target_v))]
    label = label if args.max_cld_km is None else f"cloud distance {orb.meta['cld_dist_km'][j]:.1f} km, QF {int(orb.meta['xco2_qf'][j])}"
    print(f"sounding {orb.sounding_id[j]} fp {orb.fp_number[j]} lat {orb.meta['lat'][j]:.1f} "
          f"v_rel {v_orb[j]:+.2f} km/s, shift sign in npz: {z['sign']}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1, 0.6]})
    for ib, band in enumerate(BANDS):
        y = orb.ln_t[ib, j].astype(float)
        s = orb.sigma_ln_t[ib, j].astype(float)
        tp = orb.tau[ib, j].astype(float)
        tc = tau_corr[j, ib].astype(float)
        ok = np.isfinite(y) & np.isfinite(tp) & np.isfinite(tc) & np.isfinite(s) & (s > 0)
        fits = {}
        for name, t in (("prod", tp), ("corr", tc)):
            fits[(name, 3)] = fit_cumulant(t, y, s, 3, weighted=False)
        fits[("corr", 2)] = fit_cumulant(tc, y, s, 2, weighted=False)
        init = {"k1": fits[("corr", 2)]["k1"], "k2": fits[("corr", 2)]["k2"],
                "intercept": fits[("corr", 2)]["intercept"]}
        fits[("corr", "gamma")] = fit_gamma(tc, y, s, init=init, weighted=False)

        def poly(t, r, order):
            coef = np.linalg.lstsq(design_matrix(t[ok], order), y[ok], rcond=None)[0]  # unbounded, for the curve
            return design_matrix(t, order) @ coef, coef

        tt = np.linspace(0, np.nanmax(tc[ok]) * 1.02, 300)

        # row 1: production τ
        ax = axes[0, ib]
        ax.plot(tp[ok], y[ok], ".", ms=2.5, color="0.45", label="production τ")
        c3p, _ = poly(tp, fits[("prod", 3)], 3)
        r = fits[("prod", 3)]
        ax.plot(tt, design_matrix(tt, 3) @ np.linalg.lstsq(design_matrix(tp[ok], 3), y[ok], rcond=None)[0],
                "-", color="C3", lw=1.3, label=f"order 3: k1 {r['k1']:.3f}, k2 {r['k2']:.3f}, rms {r['resid_rms']:.3f}")
        ax.set_title(f"{BAND_LABEL[band]} — production τ (no telluric Doppler shift)")
        ax.set_ylabel("ln T")
        ax.legend(fontsize=8, loc="lower left")

        # row 2: corrected τ with fits
        ax = axes[1, ib]
        ax.plot(tc[ok], y[ok], ".", ms=2.5, color="0.25", label="Doppler-corrected τ")
        for order, col in ((2, "C0"), (3, "C3")):
            r = fits[("corr", order)]
            coef = np.linalg.lstsq(design_matrix(tc[ok], order), y[ok], rcond=None)[0]
            ax.plot(tt, design_matrix(tt, order) @ coef, "-", color=col, lw=1.3,
                    label=f"order {order}: k1 {r['k1']:.3f}, k2 {r['k2']:.3f}, rms {r['resid_rms']:.3f}")
        g = fits[("corr", "gamma")]
        kap = g["k1"] ** 2 / g["k2"] if g["k2"] > 0 else np.inf
        ax.plot(tt, g["intercept"] - kap * np.log1p(g["k1"] * tt / kap), "--", color="C2", lw=1.3,
                label=f"gamma: k1 {g['k1']:.3f}, k2 {g['k2']:.3f} (κ {kap:.1f}), rms {g['resid_rms']:.3f}")
        ax.set_title(f"{BAND_LABEL[band]} — corrected τ, unit-weight fits")
        ax.set_ylabel("ln T")
        ax.legend(fontsize=8, loc="lower left")

        # row 3: residuals on corrected τ
        ax = axes[2, ib]
        for order, col in ((2, "C0"), (3, "C3")):
            coef = np.linalg.lstsq(design_matrix(tc[ok], order), y[ok], rcond=None)[0]
            ax.plot(tc[ok], y[ok] - design_matrix(tc[ok], order) @ coef, ".", ms=2, color=col, alpha=0.6,
                    label=f"order {order}")
        ax.plot(tc[ok], y[ok] - (g["intercept"] - kap * np.log1p(g["k1"] * tc[ok] / kap)), ".", ms=2,
                color="C2", alpha=0.6, label="gamma")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel("slant optical depth τ")
        ax.set_ylabel("ln T − fit")
        ax.legend(fontsize=8, loc="upper right", ncol=3)

    fig.suptitle(f"Orbit {TAG}, sounding {orb.sounding_id[j]} (fp {orb.fp_number[j] + 1}, lat {orb.meta['lat'][j]:.1f}°, "
                 f"{label}, relative velocity {v_orb[j]:+.2f} km/s → shift ≈ {0.174 * abs(v_orb[j]):.2f} px)", fontsize=12)
    fig.tight_layout()
    path = OUT / f"fig_doppler_example{suffix}_{TAG}.png"
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    main()
