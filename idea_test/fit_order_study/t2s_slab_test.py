"""t2s on the Appendix G x-z slab: closure against the MC path moments (2026-11-02).

Compares three O2A estimators on the same slab radiances:
    order7  results/rt_slab_sim/slab_fit.h5      (retired order-7 cumulant fit)
    am      results/rt_slab_sim/slab_fit_am.h5   (agreed model, floor; k2 -> k2diag)
    t2s     results/rt_slab_sim/slab_fit_t2s.h5  (fit_and_plot.py --estimator t2s)
plus t2s with tau_h2o = 0 (refitted here, not stored).

Closure statistics follow make_fig_g1.main exactly (Pearson r over the 3-D
columns of k1 vs the continuum MC mean path and of the variance term vs the MC
path variance, all and clear = outside-cloud columns; ICA null residuals); the
recomputed order7 and am values are checked against closure_stats.json and
closure_stats_am.json.  The MC moments come from make_fig_g1.continuum_moments
(imported).  Example columns as slab_closure.py: far x = 1.25 km (ix 2), sunlit
edge 9.25 km (ix 18, 0.25 km up-sun of the cloud), cloud 12.25 km (ix 24),
shadow 18.25 km (ix 36).

Usage:
    PYTHONPATH=src:workspace python idea_test/fit_order_study/t2s_slab_test.py
Output: results/rt_slab_sim/closure_stats_t2s.json,
        idea_test/fit_order_study/results/t2s_slab_test.json,
        results/rt_slab_sim/figs/slab_closure_compare_t2s.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "workspace", REPO / "workspace" / "rt_slab_sim"):
    sys.path.insert(0, str(p))

import fit_and_plot as fap  # noqa: E402
import make_fig_g1 as g1  # noqa: E402
import slab_config as cfg  # noqa: E402
from spectral.cumulant_fit import LOG_TRANSMITTANCE_MODELS, fit_spectral_model  # noqa: E402

COLUMNS = {"far": 2, "sunlit_edge": 18, "cloud": 24, "shadow": 36}
VAR_FIELD = {"order7": "k2", "am": "k2diag", "t2s": "k2", "t2s_nosplit": "k2"}
OUT_JSON = Path(__file__).resolve().parent / "results" / "t2s_slab_test.json"
CLOSURE_JSON = Path(cfg.OUT_DIR) / "closure_stats_t2s.json"
FIG = Path(cfg.OUT_DIR) / "figs" / "slab_closure_compare_t2s.png"


def load_fit(estimator):
    with h5py.File(os.path.join(cfg.OUT_DIR, fap.FIT_FILES[estimator]), "r") as f:
        assert f.attrs.get("estimator", "order7") == estimator
        return {k: {n: f[k][n][...] for n in f[k] if f[k][n].ndim == 1}
                for k in (f"{s}/{v}" for s in cfg.SURFACE_ALBEDOS for v in cfg.SOLVERS)}


def slab_inputs():
    with h5py.File(cfg.ATM_FILE, "r") as f:
        mu0 = float(np.cos(np.deg2rad(f["meta"].attrs["sza"])))
    out = {}
    with h5py.File(cfg.RAD_FILE, "r") as f:
        tau_all, wvl = f["slant_tau"][...], f["wvl_nm"][...]
        for s in cfg.SURFACE_ALBEDOS:
            for v in cfg.SOLVERS:
                g = f[f"{s}/{v}"]
                iw = g["iw"][...]
                out[f"{s}/{v}"] = (np.squeeze(g["rad"][...]).reshape(iw.size, cfg.NX),
                                   g["toa"][...], tau_all[iw], fap.spectral_x(wvl)[iw])
    return mu0, out


def refit_order7_rms(mu0, inputs, fits7):
    """ln T rms of the order-7 fit (not stored in slab_fit.h5); k1 re-checked."""
    for key, (rad, toa, tau, _) in inputs.items():
        T = np.pi * rad / (mu0 * toa[:, None])
        rms = np.full(cfg.NX, np.nan)
        for ix in range(cfg.NX):
            if np.any(T[:, ix] <= 0):
                continue
            y = np.log(T[:, ix])
            popt = fit_spectral_model(tau, y, cfg.FIT_ORDER, smooth=cfg.FIT_SMOOTH)
            assert popt[0] == fits7[key]["k1"][ix], key
            rms[ix] = np.sqrt(np.mean((y - LOG_TRANSMITTANCE_MODELS[cfg.FIT_ORDER](tau, *popt))**2))
        fits7[key]["rms"] = rms


def mc_moments(x_km):
    """make_fig_g1.main: continuum moments, L_peak anchor, l_direct."""
    with h5py.File(cfg.ATM_FILE, "r") as f:
        sza = float(f["meta"].attrs["sza"])
    l_direct = cfg.LEVELS_KM[-1] * (1.0 / np.cos(np.deg2rad(sza))
                                    + 1.0 / np.cos(np.deg2rad(cfg.SENSOR_ZENITH)))
    mom, hist = {}, {}
    with h5py.File(cfg.RAD_FILE, "r") as f:
        edges = np.linspace(f.attrs["plen_min_m"], f.attrs["plen_max_m"], f.attrs["plen_nbin"] + 1)
        mid = 0.5 * (edges[:-1] + edges[1:]) / 1e3
        for s in cfg.SURFACE_ALBEDOS:
            for v in cfg.SOLVERS:
                mean, std, h = g1.continuum_moments(f, s, v, mid)
                mom[f"{s}/{v}"], hist[f"{s}/{v}"] = (mean, std), h
    L_peak = float(mid[int(np.argmax(hist["dark/3d"][x_km < 5.0].mean(axis=0)))])
    rel = {k: (1.0 + (m - L_peak) / l_direct, sd**2 / l_direct**2) for k, (m, sd) in mom.items()}
    return sza, l_direct, mom, rel


def closure(fit, var_field, mom, x_km, extra_null=()):
    """make_fig_g1.main closure block (same masks, same r, same ICA null)."""
    far = x_km < 5.0
    in_cloud = (x_km >= cfg.CLOUD_X_KM[0]) & (x_km < cfg.CLOUD_X_KM[1])
    st = {}
    for s in cfg.SURFACE_ALBEDOS:
        k1, k2 = fit[f"{s}/3d"]["k1"], fit[f"{s}/3d"][var_field]
        m3, sd3 = mom[f"{s}/3d"]
        ok, okv = np.isfinite(k1) & np.isfinite(m3), np.isfinite(k2) & np.isfinite(sd3)
        st[f"r_k1_meanL_{s}"] = float(np.corrcoef(k1[ok], m3[ok])[0, 1])
        st[f"r_{var_field}_varL_{s}"] = float(np.corrcoef(k2[okv], sd3[okv]**2)[0, 1])
        clr, clrv = ok & ~in_cloud, okv & ~in_cloud
        st[f"r_k1_meanL_clear_{s}"] = float(np.corrcoef(k1[clr], m3[clr])[0, 1])
        st[f"r_{var_field}_varL_clear_{s}"] = float(np.corrcoef(k2[clrv], sd3[clrv]**2)[0, 1])
        for name in ("k1", var_field) + tuple(extra_null):
            v_ica = fit[f"{s}/ipa"][name]
            resid = np.nanmax(np.abs(v_ica[~in_cloud] - np.nanmedian(v_ica[far])))
            rng = np.nanmax(fit[f"{s}/3d"][name]) - np.nanmin(fit[f"{s}/3d"][name])
            st[f"ica_null_{name}_{s}"] = float(resid)
            st[f"ica_null_{name}_frac_{s}"] = float(resid / rng)
    return st


def main():
    x_km = (np.arange(cfg.NX) + 0.5) * cfg.DX_KM
    mu0, inputs = slab_inputs()
    fits = {e: load_fit(e) for e in ("order7", "am", "t2s")}

    # am path of fit_and_plot reproduces the stored slab_fit_am.h5
    for key, (rad, toa, tau, x) in inputs.items():
        am = fap.fit_columns_am(rad, toa, tau, mu0, x)
        for n in ("k1", "k2diag", "f", "a1", "intercept", "rms"):
            assert np.array_equal(am[n], fits["am"][key][n], equal_nan=True), (key, n)
    print("fit_columns_am reproduces slab_fit_am.h5 bitwise (k1, k2diag, f, a1, c, rms)")
    refit_order7_rms(mu0, inputs, fits["order7"])
    print("order-7 refit reproduces slab_fit.h5 k1 bitwise; rms computed")
    fits["t2s_nosplit"] = {key: fap.fit_columns_t2s(rad, toa, tau, np.zeros_like(tau), mu0, x)
                           for key, (rad, toa, tau, x) in inputs.items()}

    sza, l_direct, mom, rel = mc_moments(x_km)
    stats = {e: closure(fits[e], VAR_FIELD[e], mom, x_km, ("f",) if e == "am" else ())
             for e in fits}
    for e, ref in (("order7", "closure_stats.json"), ("am", "closure_stats_am.json")):
        stored = json.loads((Path(cfg.OUT_DIR) / ref).read_text())
        for k, v in stats[e].items():
            assert np.isclose(v, stored[k], rtol=1e-12, atol=1e-15), (e, k, v, stored[k])
    print("closure recomputation reproduces closure_stats.json and closure_stats_am.json")

    out_closure = {"l_direct_km": l_direct, "sza": sza, "estimator": "t2s",
                   "fit_file": fap.FIT_FILES["t2s"], **stats["t2s"]}
    CLOSURE_JSON.write_text(json.dumps(out_closure, indent=2))

    # example columns
    cols = {}
    for e, fit in fits.items():
        for key, res in fit.items():
            for reg, ix in COLUMNS.items():
                row = {n: float(res[n][ix]) for n in ("k1", "k2", "k2diag", "k1w", "a1", "f", "rms")
                       if n in res}
                row["gamma"] = float(np.exp(res["intercept"][ix]))
                cols.setdefault(e, {}).setdefault(key, {})[reg] = row
    mc = {key: {reg: {"mean_l": float(rel[key][0][ix]), "var_l": float(rel[key][1][ix])}
                for reg, ix in COLUMNS.items()} for key in rel}

    # print tables
    print("\nclosure r (3-D columns)       " + "".join(f"{e:>13s}" for e in fits))
    for s in cfg.SURFACE_ALBEDOS:
        for lab, fmt in (("r(k1, MC mean)  all", "r_k1_meanL_{s}"),
                         ("r(k1, MC mean)  clear", "r_k1_meanL_clear_{s}"),
                         ("r(k2, MC var)   all", "r_{v}_varL_{s}"),
                         ("r(k2, MC var)   clear", "r_{v}_varL_clear_{s}"),
                         ("ICA null k1 frac", "ica_null_k1_frac_{s}"),
                         ("ICA null k2 frac", "ica_null_{v}_frac_{s}")):
            print(f"  {s:6s} {lab:22s}" + "".join(
                f"{stats[e][fmt.format(s=s, v=VAR_FIELD[e])]:13.3f}" for e in fits))
    for key in inputs:
        print(f"\n{key}: column values (MC: mean l', var l')")
        print(f"  {'':20s}" + "".join(f"{r:>12s}" for r in COLUMNS))
        print(f"  {'MC mean l':20s}" + "".join(f"{mc[key][r]['mean_l']:+12.4f}" for r in COLUMNS))
        print(f"  {'MC var l':20s}" + "".join(f"{mc[key][r]['var_l']:+12.5f}" for r in COLUMNS))
        for e in fits:
            for n in ("k1", VAR_FIELD[e], "a1", "gamma", "rms") + (("f",) if e == "am" else ()) \
                    + (("k1w",) if e == "t2s" else ()):
                print(f"  {e + ' ' + n:20s}" + "".join(
                    f"{cols[e][key][r].get(n, np.nan):+12.4f}" for r in COLUMNS))

    OUT_JSON.write_text(json.dumps({"closure": stats, "columns": cols, "mc_columns": mc,
                                    "column_index": COLUMNS}, indent=2))

    # quick-look: 3-D columns, three estimators vs MC moments
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    colors = {"order7": "0.5", "am": "#0072B2", "t2s": "#D55E00", "t2s_nosplit": "#D55E00"}
    for jc, s in enumerate(cfg.SURFACE_ALBEDOS):
        for e in ("order7", "am", "t2s", "t2s_nosplit"):
            res = fits[e][f"{s}/3d"]
            ls = ":" if e == "t2s_nosplit" else "-"
            axes[0, jc].plot(x_km, res["k1"], color=colors[e], ls=ls, label=e)
            axes[1, jc].plot(x_km, res[VAR_FIELD[e]], color=colors[e], ls=ls,
                             label=f"{e} {VAR_FIELD[e]}")
        for v, ls in (("3d", "-"), ("ipa", "--")):
            axes[2, jc].plot(x_km, rel[f"{s}/{v}"][0], "k", ls=ls, label=f"MC {v}")
            axes[3, jc].plot(x_km, rel[f"{s}/{v}"][1], "k", ls=ls, label=f"MC {v}")
        for ax, lab in zip(axes[:, jc], ("k1", "k2 / k2diag", "MC mean l'", "MC var l'")):
            ax.axvspan(*cfg.CLOUD_X_KM, color="0.9", zorder=0)
            ax.set_ylabel(lab)
            ax.legend(fontsize=7)
        axes[0, jc].set_title(f"{s} surface, 3-D columns")
        axes[-1, jc].set_xlabel("x (km)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"\nwrote {CLOSURE_JSON}\nwrote {OUT_JSON}\nwrote {FIG}")


if __name__ == "__main__":
    main()
