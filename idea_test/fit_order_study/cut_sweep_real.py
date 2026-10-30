"""τ-cut sweep on one real orbit: does restricting the fit to τ ≤ τ_cut help k1 / k2?

Fits orders 2 and 3 with unit weights on channels τ ≤ τ_cut for a ladder of cuts,
and records the binned residual of the full-range fit against τ (the direct test
of whether the high-τ channels break the truncated expansion).

Outputs (results/, prefix ``cut_``): per-sounding parquet, summary csv/md, two figures.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import FAR_KM, NEAR_KM, spread_pct  # noqa: E402
from variants import _solve_bounded, design_matrix, fit_cumulant  # noqa: E402

ROOT = HERE.parents[1]
DEFAULTS = {
    "l1b": ROOT / "data/OCO2/2020/001/29252a_GL/oco2_L1bScGL_29252a_191231_B11006r_220715045716.h5",
    "fp_tau": ROOT / "results/2020-01-01/29252a_GL/fp_tau_combined.h5",
    "fit_details": ROOT / "results/fitting_details/fitting_details_2020-01-01_29252a_GL.h5",
}
CUTS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, np.inf)
ORDERS = (2, 3)
TAU_EDGES = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, np.inf])
NBIN = len(TAU_EDGES) - 1
_G: dict = {}


def _one(j):
    tau, ln_t, sig = _G["tau"], _G["ln_t"], _G["sig"]
    rows, resid_rows = [], []
    for ib, band in enumerate(BANDS):
        t, y, s = tau[ib, j], ln_t[ib, j], sig[ib, j]
        for order in ORDERS:
            for cut in CUTS:
                r = fit_cumulant(t, y, s, order, weighted=False,
                                 tau_cut=None if np.isinf(cut) else cut)
                rows.append((j, band, order, cut, r["k1"], r["k2"], r["resid_rms"],
                             r["nchan"], r["bound"]))
            m = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0)
            if m.sum() > order + 2:
                tt, yy = t[m].astype(float), y[m].astype(float)
                a = design_matrix(tt, order)
                coef, _ = _solve_bounded(a, yy, order, None)
                res = yy - a @ coef
                idx = np.digitize(tt, TAU_EDGES) - 1
                cnt = np.bincount(idx, minlength=NBIN)
                sm = np.bincount(idx, weights=res, minlength=NBIN)
                bm = np.full(NBIN, np.nan)
                bm[cnt > 0] = sm[cnt > 0] / cnt[cnt > 0]
                resid_rows.append((j, band, order, *bm))
    return rows, resid_rows


def _chunk(js):
    rows, rr = [], []
    for j in js:
        a, b = _one(j)
        rows += a
        rr += b
    return rows, rr


def aggregate(df: pd.DataFrame, meta: dict, fp: np.ndarray) -> pd.DataFrame:
    far = (meta["cld_dist_km"] > FAR_KM) & (meta["xco2_qf"] == 0)
    near = meta["cld_dist_km"] < NEAR_KM
    anom = meta["xco2_bc_anomaly"]
    ref = df[(df.order == 3) & np.isinf(df.cut)].set_index(["band", "j"])["k1"]
    o2a = df[df.band == "o2a"].set_index(["order", "cut", "j"])["k1"].sort_index()
    recs = []
    for band in BANDS:
        ref_b = ref.xs(band)
        for (order, cut), g in df[df.band == band].groupby(["order", "cut"]):
            g = g.set_index("j")
            j = g.index.values
            k1, k2 = g.k1.values, g.k2.values
            m = np.isfinite(k1) & far[j]
            full_n = np.nanmedian(df[(df.band == band) & np.isinf(df.cut)].nchan.values)
            rec = {
                "band": band, "order": order, "cut": cut, "n_far": int(m.sum()),
                "nchan_median": float(np.nanmedian(g.nchan.values[m])),
                "chan_frac": float(np.nanmedian(g.nchan.values[m]) / full_n),
                "k1_median": float(np.nanmedian(k1[m])),
                "k1_bias_vs_ols3_full_pct": 100 * (np.nanmedian(k1[m] / ref_b.reindex(j).values[m]) - 1),
                "k1_spread_pct": spread_pct(k1[m], fp[j][m]),
                "k2_median": float(np.nanmedian(k2[m])),
                "k2_zero_frac": float(np.nanmean(k2[m] <= 1e-6)),
                "k2_spread_pct": spread_pct(k2[m], fp[j][m]) if np.nanmedian(k2[m]) > 1e-6 else np.nan,
                "bound_frac": float(np.nanmean(g.bound.values[m].astype(float))),
                "resid_rms_median": float(np.nanmedian(g.resid_rms.values[m])),
                "k1_near_corr_anom": np.nan, "ratio_to_o2a_spread_pct": np.nan,
            }
            mn = np.isfinite(k1) & near[j] & np.isfinite(anom[j])
            if mn.sum() > 30:
                rec["k1_near_corr_anom"] = float(np.corrcoef(k1[mn], anom[j][mn])[0, 1])
            if band != "o2a":
                ko = o2a.loc[(order, cut)].reindex(j).values
                mr = m & np.isfinite(ko)
                rec["ratio_to_o2a_spread_pct"] = spread_pct(k1[mr] / ko[mr], fp[j][mr])
            recs.append(rec)
    return pd.DataFrame(recs)


def resid_table(rr: pd.DataFrame, meta: dict) -> pd.DataFrame:
    far = (meta["cld_dist_km"] > FAR_KM) & (meta["xco2_qf"] == 0)
    bins = [f"b{i}" for i in range(NBIN)]
    recs = []
    for (band, order), g in rr[far[rr.j.values]].groupby(["band", "order"]):
        for i, c in enumerate(bins):
            v = g[c].values
            v = v[np.isfinite(v)]
            recs.append({"band": band, "order": order, "tau_lo": TAU_EDGES[i], "tau_hi": TAU_EDGES[i + 1],
                         "n_soundings": len(v), "resid_median": np.median(v) if len(v) else np.nan,
                         "resid_q1": np.quantile(v, 0.25) if len(v) else np.nan,
                         "resid_q3": np.quantile(v, 0.75) if len(v) else np.nan})
    return pd.DataFrame(recs)


def md_table(df: pd.DataFrame, cols, fmt="{:.3g}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figures(summ: pd.DataFrame, rt: pd.DataFrame, out: Path, tag: str):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ib, band in enumerate(BANDS):
        for order, ls in zip(ORDERS, ("-", "--")):
            d = summ[(summ.band == band) & (summ.order == order)].sort_values("cut")
            x = np.where(np.isinf(d.cut), 30.0, d.cut)
            axes[0, ib].plot(x, d.k1_spread_pct, ls, marker="o", label=f"order {order}")
            axes[1, ib].plot(x, d.k1_bias_vs_ols3_full_pct, ls, marker="o")
            axes[2, ib].plot(x, d.k2_median, ls, marker="o")
        axes[0, ib].set_title(band)
        axes[0, ib].set_ylabel("far-clear k1 spread (%)")
        axes[1, ib].set_ylabel("k1 bias vs ols_3 full (%)")
        axes[2, ib].set_ylabel("median k2")
        axes[2, ib].set_xlabel("τ_cut (30 = no cut)")
        axes[2, ib].set_xscale("log")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / f"cut_sweep_{tag}.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ib, band in enumerate(BANDS):
        for order, ls in zip(ORDERS, ("-", "--")):
            d = rt[(rt.band == band) & (rt.order == order)]
            xm = np.sqrt(d.tau_lo.clip(0.05) * np.where(np.isinf(d.tau_hi), 30.0, d.tau_hi))
            axes[ib].plot(xm, d.resid_median, ls, marker="o", label=f"order {order}")
            axes[ib].fill_between(xm, d.resid_q1, d.resid_q3, alpha=0.15)
        axes[ib].axhline(0, color="k", lw=0.6)
        axes[ib].set_xscale("log")
        axes[ib].set_title(f"{band}: full-range fit residual vs τ (far-clear median, IQR)")
        axes[ib].set_xlabel("τ (bin centre)")
        axes[ib].set_ylabel("ln T − model")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / f"cut_resid_vs_tau_{tag}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b", default=str(DEFAULTS["l1b"]))
    ap.add_argument("--fp-tau", default=str(DEFAULTS["fp_tau"]))
    ap.add_argument("--fit-details", default=str(DEFAULTS["fit_details"]))
    ap.add_argument("--max-soundings", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    tag = Path(args.fp_tau).parent.name

    orb = load_orbit(args.l1b, args.fp_tau, args.fit_details, max_soundings=args.max_soundings)
    n = orb.tau.shape[1]
    print(f"loaded {n} soundings", flush=True)
    _G.update(tau=orb.tau, ln_t=orb.ln_t, sig=orb.sigma_ln_t)
    chunks = np.array_split(np.arange(n), args.workers * 8)
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_chunk, chunks)
    rows = [r for p in parts for r in p[0]]
    rr = [r for p in parts for r in p[1]]
    df = pd.DataFrame(rows, columns=["j", "band", "order", "cut", "k1", "k2", "resid_rms", "nchan", "bound"])
    rr = pd.DataFrame(rr, columns=["j", "band", "order", *[f"b{i}" for i in range(NBIN)]])
    df.to_parquet(out / f"cut_per_sounding_{tag}.parquet")
    print(f"fitted {len(df)} rows", flush=True)

    summ = aggregate(df, orb.meta, orb.fp_number)
    rt = resid_table(rr, orb.meta)
    summ.to_csv(out / f"cut_sweep_{tag}.csv", index=False)
    rt.to_csv(out / f"cut_resid_vs_tau_{tag}.csv", index=False)

    cols = ["order", "cut", "chan_frac", "k1_median", "k1_bias_vs_ols3_full_pct", "k1_spread_pct",
            "ratio_to_o2a_spread_pct", "k2_median", "k2_zero_frac", "k2_spread_pct", "resid_rms_median",
            "k1_near_corr_anom"]
    lines = [f"# τ-cut sweep — orbit {tag}", "",
             f"Unit-weight cumulant fit on channels with τ ≤ τ_cut (cut = inf → all channels). "
             f"Far-clear: cld_dist > {FAR_KM:g} km, QF0. Spread = robust SD (%) after removing "
             "per-footprint-index means. Bias is relative to the full-range order-3 fit.", ""]
    for band in BANDS:
        lines += [f"## {band}", "", md_table(summ[summ.band == band].sort_values(["order", "cut"]), cols), ""]
    lines += ["## Full-range fit residual vs τ (far-clear, median over soundings of per-sounding bin means)", "",
              md_table(rt, ["band", "order", "tau_lo", "tau_hi", "n_soundings", "resid_median", "resid_q1", "resid_q3"]), ""]
    (out / "cut_sweep.md").write_text("\n".join(lines))
    figures(summ, rt, out, tag)
    print("done", flush=True)


if __name__ == "__main__":
    main()
