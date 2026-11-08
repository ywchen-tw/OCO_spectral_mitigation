"""Closure (i) (`o2x`) on the Appendix G x-z slab: column table and tau-cut sweep (2026-11-03).

Closure (i) of the paper: ln T = c + a1 x - k1 tau + 0.5 k2 tau^2, unconstrained lstsq,
fitted by workspace/rt_slab_sim/fit_and_plot.py --estimator o2x [--tau-max T].
Compares, at the example columns of t2s_slab_test.py (far 1.25 km, sunlit edge 9.25 km,
shadow 18.25 km; cloud 12.25 km for reference), the stored fits
    order7      slab_fit.h5            (retired order-7 cumulant fit; rms refitted here)
    am          slab_fit_am.h5         (floor form, k2 = 0; k2diag shown in the k2 slot)
    o2x         slab_fit_o2x.h5        (closure (i), all 33 wavelengths)
    o2x_tmax2   slab_fit_o2x_tmax2.h5  (closure (i), slant tau <= 2, 25 wavelengths)
with the traced MC continuum moments (make_fig_g1 definitions).  Closure r values come from
the closure_stats_*.json files written by make_fig_g1.py and are recomputed here as a check.
A tau-cut sweep (tau_max = 8, 4, 2, 1; refitted here, not stored) shows how the closure
depends on the cut.

Usage:
    PYTHONPATH=src:workspace python idea_test/fit_order_study/o2x_slab_test.py
Output: idea_test/fit_order_study/results/o2x_slab_test.json (+ printed tables)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "workspace", REPO / "workspace" / "rt_slab_sim",
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import fit_and_plot as fap  # noqa: E402
import slab_config as cfg  # noqa: E402
from t2s_slab_test import closure, mc_moments, refit_order7_rms, slab_inputs  # noqa: E402

COLUMNS = {"far": 2, "sunlit_edge": 18, "cloud": 24, "shadow": 36}
FILES = {"order7": ("slab_fit.h5", "closure_stats.json", "k2"),
         "am": ("slab_fit_am.h5", "closure_stats_am.json", "k2diag"),
         "o2x": ("slab_fit_o2x.h5", "closure_stats_o2x.json", "k2"),
         "o2x_tmax2": ("slab_fit_o2x_tmax2.h5", "closure_stats_o2x_tmax2.json", "k2")}
SWEEP = (8.0, 4.0, 2.0, 1.0)
OUT_JSON = Path(__file__).resolve().parent / "results" / "o2x_slab_test.json"


def load_fit(fname, tag):
    with h5py.File(os.path.join(cfg.OUT_DIR, fname), "r") as f:
        assert f.attrs.get("estimator", "order7") == tag, (fname, tag)
        return {k: {n: f[k][n][...] for n in f[k] if f[k][n].ndim == 1}
                for k in (f"{s}/{v}" for s in cfg.SURFACE_ALBEDOS for v in cfg.SOLVERS)}


def column_rows(fit, var_field):
    return {key: {reg: {"k1": float(res["k1"][ix]), "k2": float(res[var_field][ix]),
                        "gamma": float(np.exp(res["intercept"][ix])),
                        "rms": float(res["rms"][ix]),
                        **({"f": float(res["f"][ix])} if "f" in res else {})}
                  for reg, ix in COLUMNS.items()}
            for key, res in fit.items()}


def main():
    x_km = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM
    mu0, inputs = slab_inputs()
    fits = {e: load_fit(fn, e) for e, (fn, _, _) in FILES.items()}
    refit_order7_rms(mu0, inputs, fits["order7"])

    # stored o2x files reproduce from fit_columns_o2x
    for e, tmax in (("o2x", None), ("o2x_tmax2", 2.0)):
        for key, (rad, toa, tau, x) in inputs.items():
            r = fap.fit_columns_o2x(rad, toa, tau, mu0, x, tmax)
            for n in ("k1", "k2", "a1", "intercept", "rms"):
                assert np.array_equal(r[n], fits[e][key][n], equal_nan=True), (e, key, n)

    sza, l_direct, mom, rel = mc_moments(x_km)
    stats = {}
    for e, (_, js, vf) in FILES.items():
        stats[e] = closure(fits[e], vf, mom, x_km)
        stored = json.loads((Path(cfg.OUT_DIR) / js).read_text())
        for k, v in stats[e].items():
            assert np.isclose(v, stored[k], rtol=1e-12, atol=1e-15), (e, k)
    cols = {e: column_rows(fits[e], FILES[e][2]) for e in FILES}
    mc = {key: {reg: {"mean_l": float(rel[key][0][ix]), "var_l": float(rel[key][1][ix])}
                for reg, ix in COLUMNS.items()} for key in rel}

    print("MC traced (continuum) mean / var at far, edge, shadow:")
    for key in rel:
        print(f"  {key:11s}" + "".join(
            f"  {r}: {mc[key][r]['mean_l']:.3f} / {mc[key][r]['var_l']:.4f}"
            for r in ("far", "sunlit_edge", "shadow")))
    for e in FILES:
        print(f"\n{e}")
        for key in inputs:
            c = cols[e][key]
            print(f"  {key:11s} k1 far/edge/shadow {c['far']['k1']:.3f} {c['sunlit_edge']['k1']:.3f}"
                  f" {c['shadow']['k1']:.3f} | k2 far/shadow {c['far']['k2']:.3f} {c['shadow']['k2']:.3f}"
                  f" | gamma far/edge/shadow {c['far']['gamma']:.3f} {c['sunlit_edge']['gamma']:.3f}"
                  f" {c['shadow']['gamma']:.3f} | rms far/shadow {c['far']['rms']:.4f} {c['shadow']['rms']:.4f}")
        s = stats[e]
        vf = FILES[e][2]
        print("  clear r(k1,MCmean) dark/bright %.3f / %.3f; clear r(k2,MCvar) %.3f / %.3f; "
              "ICA null frac k1 %.3f / %.3f, k2 %.3f / %.3f" % (
                  s["r_k1_meanL_clear_dark"], s["r_k1_meanL_clear_bright"],
                  s[f"r_{vf}_varL_clear_dark"], s[f"r_{vf}_varL_clear_bright"],
                  s["ica_null_k1_frac_dark"], s["ica_null_k1_frac_bright"],
                  s[f"ica_null_{vf}_frac_dark"], s[f"ica_null_{vf}_frac_bright"]))

    # tau-cut sweep (refit, 3-D and IPA)
    sweep = {}
    print("\ntau-cut sweep (o2x refit): nchan | dark k1 far/shadow, k2 far/shadow | bright same | "
          "clear r(k1), r(k2) dark, bright | rms range 3-D")
    for tmax in SWEEP:
        fit = {key: fap.fit_columns_o2x(rad, toa, tau, mu0, x, tmax)
               for key, (rad, toa, tau, x) in inputs.items()}
        st = closure(fit, "k2", mom, x_km)
        row = {"nchan": int(fit["dark/3d"]["nchan"][0]), "closure": st,
               "columns": column_rows(fit, "k2")}
        sweep[f"{tmax:g}"] = row
        c = row["columns"]
        rms3 = np.concatenate([fit[f"{s}/3d"]["rms"] for s in cfg.SURFACE_ALBEDOS])
        print(f"  tau<={tmax:<4g} {row['nchan']:2d} | "
              + " | ".join(f"{c[f'{s}/3d']['far']['k1']:.3f} {c[f'{s}/3d']['shadow']['k1']:.3f}, "
                           f"{c[f'{s}/3d']['far']['k2']:.3f} {c[f'{s}/3d']['shadow']['k2']:.3f}"
                           for s in cfg.SURFACE_ALBEDOS)
              + " | " + ", ".join(f"{st[f'r_k1_meanL_clear_{s}']:.3f} {st[f'r_k2_varL_clear_{s}']:.3f}"
                                  for s in cfg.SURFACE_ALBEDOS)
              + f" | {np.nanmin(rms3):.4f}-{np.nanmax(rms3):.4f}")

    OUT_JSON.write_text(json.dumps({"closure": stats, "columns": cols, "mc_columns": mc,
                                    "column_index": COLUMNS, "tau_sweep": sweep}, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
