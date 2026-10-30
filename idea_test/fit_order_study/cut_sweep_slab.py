"""τ-cut sweep on the slab Monte Carlo spectra against the absorption-weighted truth.

Question: does fitting only the low-τ channels bring k1 closer to ⟨A⟩ = K1(0) and
k2 closer to var A = K2(0), or does the local-curvature behaviour persist?  Also
tabulates the true local cumulants K1(τ), K2(τ) on a τ grid, which bound what any
τ-restricted fit can recover.

Reads ``results/rt_slab_sim`` (read-only) and ``results/slab_closure_truth.csv``;
writes ``results/cut_slab_*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slab_truth as st  # noqa: E402
from cut_sweep_real import md_table  # noqa: E402
from variants import fit_cumulant  # noqa: E402

OUT = HERE / "results"
CASES = ("dark/3d", "bright/3d")
CUTS = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf)
ORDERS = (2, 3, 7)
TAU_GRID = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
TAU_FINE = np.concatenate([[0.0], np.logspace(-2, 1, 300)])
K2_TARGETS = ("v_abs", "K2_abs@0.1", "K2_abs@0.3", "K2_abs@1.0")


def regions(x_km):
    in_cloud = (x_km >= st.CLOUD_X_KM[0]) & (x_km < st.CLOUD_X_KM[1])
    shadow = (x_km >= st.CLOUD_X_KM[1]) & (x_km <= 25.0)
    return np.where(in_cloud, "cloud", np.where(shadow, "shadow", "clear"))


def tau_star(rec, k, which):
    """Smallest τ with K_which(τ) = k on TAU_FINE (linear interpolation), NaN if none."""
    if not np.isfinite(k):
        return np.nan
    k1, k2 = st.tilted_cumulants(rec["w"], rec["a"], TAU_FINE, rec["v"])
    d = (k1 if which == 1 else k2) - k
    s = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if len(s) == 0:
        return np.nan
    i = s[0]
    return float(TAU_FINE[i] + (TAU_FINE[i + 1] - TAU_FINE[i]) * d[i] / (d[i] - d[i + 1]))


def _rel(est, tru):
    m = np.isfinite(est) & np.isfinite(tru) & (tru != 0)
    return float(np.median(est[m] / tru[m] - 1)) if m.sum() else np.nan


def _corr(est, tru):
    m = np.isfinite(est) & np.isfinite(tru)
    return float(np.corrcoef(est[m], tru[m])[0, 1]) if m.sum() > 3 else np.nan


def main():
    cases, meta = st.load_cases()
    truth = pd.read_csv(OUT / "slab_closure_truth.csv").set_index(["case", "ix"])
    fit_rows, curve_rows = [], []
    for key in CASES:
        c = cases[key]
        reg = regions(c.x_km)
        _, recs = st.column_truths(c)
        print(f"{key}: truths rebuilt", flush=True)
        unit = np.ones_like(c.tau)
        for ix in range(st.NX):
            k1c, k2c = st.tilted_cumulants(recs[ix]["w"], recs[ix]["a"], TAU_GRID, recs[ix]["v"])
            curve_rows += [(key, ix, reg[ix], t, a, b) for t, a, b in zip(TAU_GRID, k1c, k2c)]
            for order in ORDERS:
                for cut in CUTS:
                    r = fit_cumulant(c.tau, c.ln_t[:, ix], unit, order, weighted=False,
                                     tau_cut=None if np.isinf(cut) else cut)
                    fit_rows.append((key, ix, reg[ix], order, cut, r["k1"], r["k2"], r["nchan"], r["bound"],
                                     tau_star(recs[ix], r["k1"], 1), tau_star(recs[ix], r["k2"], 2)))
    fits = pd.DataFrame(fit_rows, columns=["case", "ix", "region", "order", "cut", "k1", "k2", "nchan",
                                           "bound", "tau_star_k1", "tau_star_k2"])
    curves = pd.DataFrame(curve_rows, columns=["case", "ix", "region", "tau", "K1", "K2"])
    fits.to_csv(OUT / "cut_slab_fits.csv", index=False)
    curves.to_csv(OUT / "cut_slab_local_cumulants.csv", index=False)

    # local cumulants relative to the τ = 0 values, clear-column medians
    cm = []
    for key in CASES:
        t0 = curves[(curves.case == key) & (curves.tau == 0)].set_index("ix")
        for tau, g in curves[curves.case == key].groupby("tau"):
            g = g.set_index("ix")
            clr = g.region == "clear"
            cm.append({"case": key, "tau": tau,
                       "K1_over_meanA_clear": float(np.median((g.K1 / t0.K1.reindex(g.index))[clr])),
                       "K2_over_varA_clear": float(np.median((g.K2 / t0.K2.reindex(g.index))[clr])),
                       "K2_clear": float(np.median(g.K2[clr])),
                       "K2_over_varA_cloud": float(np.median((g.K2 / t0.K2.reindex(g.index))[g.region == "cloud"]))})
    cm = pd.DataFrame(cm)
    cm.to_csv(OUT / "cut_slab_local_cumulants_summary.csv", index=False)

    # closure per (case, order, cut, region)
    recs_out = []
    for (key, order, cut), g in fits.groupby(["case", "order", "cut"]):
        g = g.set_index("ix")
        tr = truth.xs(key).reindex(g.index)
        for region in ("clear", "all"):
            m = np.ones(len(g), bool) if region == "all" else (g.region == "clear").values
            k1, k2 = g.k1.values[m], g.k2.values[m]
            rec = {"case": key, "order": order, "cut": cut, "region": region,
                   "nchan": int(np.nanmedian(g.nchan.values[m])),
                   "k1_bias_meanA_pct": 100 * _rel(k1, tr.m_abs.values[m]),
                   "k1_r_meanA": _corr(k1, tr.m_abs.values[m]),
                   "k1_bias_K1at0.3_pct": 100 * _rel(k1, tr["K1_abs@0.3"].values[m]),
                   "k1_tau_star_med": float(np.nanmedian(g.tau_star_k1.values[m])),
                   "k2_zero_frac": float(np.mean(k2 <= 1e-6)),
                   "k2_tau_star_med": float(np.nanmedian(g.tau_star_k2.values[m])),
                   "bound_frac": float(np.mean(g.bound.values[m].astype(float)))}
            for t in K2_TARGETS:
                rec[f"k2_over_{t}"] = float(np.nanmedian(k2 / tr[t].values[m]))
                rec[f"k2_r_{t}"] = _corr(k2, tr[t].values[m])
            recs_out.append(rec)
    summ = pd.DataFrame(recs_out)
    summ.to_csv(OUT / "cut_slab_sweep.csv", index=False)

    cols = ["order", "cut", "nchan", "k1_bias_meanA_pct", "k1_r_meanA", "k1_bias_K1at0.3_pct", "k1_tau_star_med",
            "k2_over_v_abs", "k2_over_K2_abs@0.1", "k2_over_K2_abs@0.3", "k2_over_K2_abs@1.0", "k2_r_v_abs",
            "k2_r_K2_abs@0.3", "k2_zero_frac", "k2_tau_star_med"]
    lines = ["# τ-cut sweep on the slab MC spectra (unit weights, MC-mean ln T, 33-channel slab grid)", "",
             "Truth: absorption-weighted PPDF per column (⟨A⟩ = K1(0), var A = K2(0)) and its local "
             "cumulants K1(τ), K2(τ). `k2_over_X` = median ratio of fitted k2 to X; τ* = τ at which the "
             "true local cumulant equals the fitted value. Region `clear` = outside the cloud.", ""]
    for key in CASES:
        for region in ("clear", "all"):
            d = summ[(summ.case == key) & (summ.region == region)].sort_values(["order", "cut"])
            lines += [f"## {key} — {region}", "", md_table(d, cols), ""]
    lines += ["## True local cumulants (clear-column median, relative to τ = 0)", "",
              md_table(cm, ["case", "tau", "K1_over_meanA_clear", "K2_over_varA_clear", "K2_clear",
                            "K2_over_varA_cloud"]), ""]
    (OUT / "cut_sweep_slab.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ik, key in enumerate(CASES):
        d = cm[cm.case == key]
        axes[ik, 0].plot(d.tau.clip(0.03), d.K1_over_meanA_clear, "k-o", label="K1(τ)/⟨A⟩")
        axes[ik, 0].plot(d.tau.clip(0.03), d.K2_over_varA_clear, "r-o", label="K2(τ)/var A")
        axes[ik, 0].set_xscale("log")
        axes[ik, 0].set_title(f"{key}: true local cumulants (clear)")
        axes[ik, 0].set_xlabel("τ")
        axes[ik, 0].legend()
        for order, ls in zip(ORDERS, ("-", "--", ":")):
            s = summ[(summ.case == key) & (summ.region == "clear") & (summ.order == order)].sort_values("cut")
            x = np.where(np.isinf(s.cut), 20.0, s.cut)
            axes[ik, 1].plot(x, s.k1_bias_meanA_pct, ls, marker="o", label=f"order {order}")
            axes[ik, 2].plot(x, s.k2_over_v_abs, ls, marker="o", label=f"order {order}")
        axes[ik, 1].axhline(0, color="k", lw=0.6)
        axes[ik, 1].set_xscale("log")
        axes[ik, 1].set_title("k1 bias vs ⟨A⟩ (%)")
        axes[ik, 1].set_xlabel("τ_cut (20 = none)")
        axes[ik, 2].axhline(1, color="k", lw=0.6)
        axes[ik, 2].set_xscale("log")
        axes[ik, 2].set_yscale("log")
        axes[ik, 2].set_title("fitted k2 / var A (median, clear)")
        axes[ik, 2].set_xlabel("τ_cut (20 = none)")
        axes[ik, 1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cut_slab_sweep.png", dpi=130)
    print("done", flush=True)


if __name__ == "__main__":
    main()
