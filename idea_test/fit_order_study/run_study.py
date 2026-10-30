"""Truncation-order / weighting study for the three-band cumulant fit.

Usage (one orbit with local L1B, fp_tau and fitting-details files):
    python idea_test/fit_order_study/run_study.py --l1b ... --fp-tau ... --fit-details ...
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orbit_io import BANDS, load_orbit  # noqa: E402
from variants import FIELDS, fit_variant, variant_list  # noqa: E402

FAR_KM = 20.0
NEAR_KM = 3.0
PPM_PER_UNIT = 400.0  # 1 % of k1 ≈ 4 ppm XCO2 at 400 ppm
# Production truncation orders (src/constants.py FIT_ORDER = (7, 3, 7)).
PROD_TAG = {"o2a": "ols_7", "wco2": "ols_3", "sco2": "ols_7"}

# Worker globals (fork context: arrays are inherited, not pickled).
_G: dict = {}


def _fit_rows(j_list):
    tau, ln_t, sig, variants, gamma_idx = (_G[k] for k in ("tau", "ln_t", "sig", "variants", "gamma_idx"))
    rows = []
    for j in j_list:
        for ib, band in enumerate(BANDS):
            init = None
            for tag, kind, order in variants:
                if kind in ("gamma", "gamma_u") and j not in gamma_idx:
                    continue
                res = fit_variant(kind, order, tau[ib, j], ln_t[ib, j], sig[ib, j], init=init)
                if tag == "ols_2":
                    init = res
                rows.append((j, band, tag, *[res[f] for f in FIELDS]))
    return rows


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def spread_pct(x, fp):
    """Robust SD (%) of x after removing the per-footprint-index mean."""
    s = pd.Series(x, dtype=float)
    rel = s / s.groupby(fp).transform("mean") - 1.0
    return 100.0 * robust_sd(rel)


def fp_mean_spread_pct(x, fp):
    s = pd.Series(x, dtype=float)
    m = s.groupby(fp).mean()
    return 100.0 * (m.max() - m.min()) / s.mean()


def aggregate(long: pd.DataFrame, meta: dict, fp: np.ndarray) -> pd.DataFrame:
    cld = meta["cld_dist_km"]
    qf = meta["xco2_qf"]
    far = (cld > FAR_KM) & (qf == 0)
    near = cld < NEAR_KM
    wide = long.set_index(["j", "band", "variant"])[list(FIELDS)].unstack("variant")
    def _key(t):
        kind, _, rest = t.partition("_")
        return (kind, int(rest) if rest.isdigit() else 0, rest)
    tags = sorted(long["variant"].unique(), key=_key)
    recs = []
    for band in BANDS:
        w = wide.xs(band, level="band")
        j = w.index.values
        f_far, f_near = far[j], near[j]
        ref_k1 = w[("k1", PROD_TAG[band])].values
        o2a = wide.xs("o2a", level="band")
        for tag in tags:
            k1 = w[("k1", tag)].values
            k2 = w[("k2", tag)].values
            ok = np.isfinite(k1)
            m = ok & f_far
            rec = {"band": band, "variant": tag, "n_far": int(m.sum()),
                   "frac_fit_ok": float(ok.mean()),
                   "sig_k1_pct": 100 * np.nanmedian(w[("sig_k1", tag)].values[m] / k1[m]),
                   "sig_k1_rss_pct": 100 * np.nanmedian(w[("sig_k1_rss", tag)].values[m] / k1[m]),
                   "sig_k2": np.nanmedian(w[("sig_k2", tag)].values[m]),
                   "k1_median": np.nanmedian(k1[m]),
                   "k2_median": np.nanmedian(k2[m]),
                   "k2_zero_frac": float(np.nanmean(k2[m] <= 1e-6)) if np.isfinite(k2[m]).any() else np.nan,
                   "bound_frac": float(np.nanmean(w[("bound", tag)].values[m].astype(float))),
                   "chi2_red_median": np.nanmedian(w[("chi2_red", tag)].values[m]),
                   "resid_rms_median": np.nanmedian(w[("resid_rms", tag)].values[m]),
                   "lag1_median": np.nanmedian(w[("lag1", tag)].values[m]),
                   "k1_bias_vs_prod_pct": 100 * (np.nanmedian(k1[m] / ref_k1[m]) - 1),
                   "k1_spread_pct": spread_pct(k1[m], fp[j][m]),
                   "k1_fp_mean_spread_pct": fp_mean_spread_pct(k1[m], fp[j][m]),
                   "k1_near_corr_anom": np.nan, "ratio_to_o2a_spread_pct": np.nan}
            mn = ok & f_near & np.isfinite(meta["xco2_bc_anomaly"][j])
            if mn.sum() > 30:
                rec["k1_near_corr_anom"] = float(np.corrcoef(k1[mn], meta["xco2_bc_anomaly"][j][mn])[0, 1])
            if band != "o2a":
                k1_o2a = o2a[("k1", tag)].reindex(j).values
                mr = m & np.isfinite(k1_o2a)
                rec["ratio_to_o2a_spread_pct"] = spread_pct(k1[mr] / k1_o2a[mr], fp[j][mr])
            recs.append(rec)
    return pd.DataFrame(recs)


def md_table(df: pd.DataFrame, cols, fmt="{:.3g}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(summary: pd.DataFrame, out_dir: Path, orbit: str, repro: dict):
    cols = ["variant", "sig_k1_pct", "k1_spread_pct", "k1_fp_mean_spread_pct", "ratio_to_o2a_spread_pct",
            "k1_bias_vs_prod_pct", "k2_zero_frac", "chi2_red_median", "resid_rms_median", "lag1_median",
            "k1_near_corr_anom", "n_far"]
    parts = [f"# Fit-order study — orbit {orbit}\n",
             f"Far-clear: cld_dist > {FAR_KM:g} km and QF0. Spreads are robust SD (%) after removing "
             "per-footprint-index means; 1 % of k1 ≈ 4 ppm XCO2. `sig_k1_pct` is the formal "
             "instrument-noise σ(k1)/k1; `chi2_red` = residual/noise variance ratio.\n",
             "Reproduction of production `_nosg` by the production-order variant (median |Δk1|/k1): "
             + ", ".join(f"{b} {v:.2e}" for b, v in repro.items()) + "\n"]
    for band in BANDS:
        parts.append(f"\n## {band}\n")
        parts.append(md_table(summary[summary.band == band], cols))
    bars = summary.set_index(["band", "variant"])
    r = lambda b, t: bars.loc[(b, t), "ratio_to_o2a_spread_pct"]  # noqa: E731
    parts.append("\n## Bars (PLAN.md)\n")
    parts.append(f"- H1 (order): SCO2/O2A spread ols_7 = {r('sco2','ols_7'):.2f} %, ols_3 = {r('sco2','ols_3'):.2f} %, "
                 f"ols_2 = {r('sco2','ols_2'):.2f} % → {'PASS' if min(r('sco2','ols_2'), r('sco2','ols_3')) <= 0.5*r('sco2','ols_7') else 'FAIL'}")
    chi = bars.loc[("sco2", "ols_7"), "chi2_red_median"]
    parts.append(f"- H2 (weighting): SCO2/O2A spread wls_2 = {r('sco2','wls_2'):.2f} % vs ols_2 = {r('sco2','ols_2'):.2f} %; "
                 f"median χ²(ols_7, SCO2) = {chi:.1f}")
    o2 = bars.loc["o2a"]
    parts.append(f"- H3 (τ-cut, O2A): cut_2 bias {o2.loc['cut_2','k1_bias_vs_prod_pct']:+.2f} %, spread "
                 f"{o2.loc['cut_2','k1_spread_pct']:.2f} % vs ols_2 {o2.loc['ols_2','k1_spread_pct']:.2f} %")
    if ("sco2", "gamma") in bars.index:
        g = bars.loc[("sco2", "gamma")]
        parts.append(f"- H4 (gamma): SCO2/O2A spread {g['ratio_to_o2a_spread_pct']:.2f} %, k2>0 fraction "
                     f"{1 - g['k2_zero_frac']:.2f}, bias {g['k1_bias_vs_prod_pct']:+.2f} %")
    (out_dir / "summary.md").write_text("\n".join(parts) + "\n")


def figures(summary: pd.DataFrame, long: pd.DataFrame, orb, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from variants import design_matrix

    kinds = {"ols": "C0", "wls": "C1", "cut": "C2"}
    s = summary.copy()
    s["kind"] = s.variant.str.split("_").str[0]
    s["order"] = pd.to_numeric(s.variant.str.split("_").str[1], errors="coerce")

    def _panel(ax, band, col, ylabel, log=True):
        for kind, c in kinds.items():
            d = s[(s.band == band) & (s.kind == kind)].sort_values("order")
            ax.plot(d.order, d[col], "o-", color=c, label=kind)
        g = s[(s.band == band) & (s.variant == "gamma")]
        if len(g):
            ax.axhline(g[col].values[0], color="k", ls="--", label="gamma")
        ax.set_title(band); ax.set_xlabel("truncation order"); ax.set_ylabel(ylabel)
        vals = s[s.band == band][col].values
        if log and np.any(np.isfinite(vals) & (vals > 0)):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for col, ylabel, fname in (("sig_k1_pct", "formal σ(k1)/k1 [%]", "fig_sigma_k1_vs_order.png"),
                               ("k1_spread_pct", "far-clear k1 spread, per-fp removed [%]", "fig_k1_spread_vs_order.png"),
                               ("ratio_to_o2a_spread_pct", "spread of k1_band/k1_O2A [%]", "fig_crossband_spread_vs_order.png"),
                               ("chi2_red_median", "median reduced χ²", "fig_chi2_vs_order.png"),
                               ("k1_bias_vs_prod_pct", "k1 median bias vs production order [%]", "fig_k1_bias_vs_order.png")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, band in zip(axes, BANDS):
            _panel(ax, band, col, ylabel, log=col not in ("k1_bias_vs_prod_pct",))
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=130); plt.close(fig)

    # per-fp relative k1 means
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    far = (orb.meta["cld_dist_km"] > FAR_KM) & (orb.meta["xco2_qf"] == 0)
    for ax, band in zip(axes, BANDS):
        for tag, c in (("ols_7", "C3"), ("ols_3", "C0"), ("ols_2", "C2"), ("wls_2", "C1"), ("gamma", "k")):
            d = long[(long.band == band) & (long.variant == tag)].set_index("j")
            if d.empty:
                continue
            k1 = d["k1"].reindex(np.flatnonzero(far)).values
            fp = orb.fp_number[far]
            m = pd.Series(k1).groupby(fp).mean()
            ax.plot(m.index + 1, m / m.mean() - 1, "o-", color=c, label=tag)
        ax.set_title(band); ax.set_xlabel("footprint index"); ax.set_ylabel("per-fp mean k1 / overall − 1")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_fp_means.png", dpi=130); plt.close(fig)

    # example fits: first far-clear sounding with all variants finite
    cand = np.flatnonzero(far)
    jx = None
    for j in cand:
        d = long[(long.j == j) & (long.variant.isin(["ols_2", "ols_3", "ols_7", "gamma"]))]
        if len(d) == 12 and d.k1.notna().all():
            jx = j; break
    if jx is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for ib, band in enumerate(BANDS):
        t, y = orb.tau[ib, jx].astype(float), orb.ln_t[ib, jx].astype(float)
        ok = np.isfinite(t) & np.isfinite(y)
        t, y = t[ok], y[ok]
        o = np.argsort(t)
        ax, axr = axes[0, ib], axes[1, ib]
        ax.plot(t, y, ".", ms=2, color="0.6", label="ln T")
        d = long[(long.j == jx) & (long.band == band)].set_index("variant")
        for tag, c in (("ols_7", "C3"), ("ols_3", "C0"), ("ols_2", "C2"), ("gamma", "k")):
            r = d.loc[tag]
            if tag == "gamma":
                kap = r.k1**2 / r.k2
                yy = r.intercept - kap * np.log1p(r.k1 * t[o] / kap)
            else:
                order = int(tag.split("_")[1])
                coef = np.r_[r.intercept, r.k1, r.k2, [0.0] * (order - 2)] if order >= 2 else np.r_[r.intercept, r.k1]
                # higher-order coefficients are not stored; re-solve for the curve
                a = design_matrix(t, order)
                coef = np.linalg.lstsq(a, y, rcond=None)[0]
                yy = a[o] @ coef
            ax.plot(t[o], yy, "-", color=c, lw=1, label=f"{tag} (k1={r.k1:.3f})")
            axr.plot(t[o], y[o] - yy, ".", ms=2, color=c)
        ax.set_title(f"{band}, sounding {orb.sounding_id[jx]}"); ax.legend(fontsize=7)
        axr.axhline(0, color="k", lw=0.5); axr.set_xlabel("τ"); axr.set_ylabel("residual")
        ax.set_ylabel("ln T")
    fig.tight_layout(); fig.savefig(out_dir / "fig_example_fits.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b", required=True)
    ap.add_argument("--fp-tau", required=True)
    ap.add_argument("--fit-details", required=True)
    ap.add_argument("--orbit", default=None)
    ap.add_argument("--orders", default="1,2,3,4,5,6,7")
    ap.add_argument("--max-soundings", type=int, default=None)
    ap.add_argument("--gamma-subsample", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    orbit = args.orbit or Path(args.fp_tau).parent.name
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    orb = load_orbit(args.l1b, args.fp_tau, args.fit_details, args.max_soundings)
    n = len(orb.sounding_id)
    print(f"loaded {n} soundings in {time.time() - t0:.0f} s")

    orders = [int(x) for x in args.orders.split(",")]
    variants = variant_list(orders, with_gamma=args.gamma_subsample > 0)
    rng = np.random.default_rng(0)
    gamma_idx = set(rng.choice(n, size=min(args.gamma_subsample, n), replace=False).tolist())
    # Edge channels are excluded in production ([1:-1]); mirror that here.
    _G.update(tau=orb.tau[:, :, 1:-1], ln_t=orb.ln_t[:, :, 1:-1], sig=orb.sigma_ln_t[:, :, 1:-1],
              variants=variants, gamma_idx=gamma_idx)

    chunks = np.array_split(np.arange(n), max(1, n // 200))
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            parts = pool.map(_fit_rows, chunks)
    else:
        parts = [_fit_rows(c) for c in chunks]
    rows = [r for p in parts for r in p]
    long = pd.DataFrame(rows, columns=["j", "band", "variant", *FIELDS])
    print(f"fitted {len(long)} rows in {time.time() - t0:.0f} s")
    long.to_parquet(out_dir / f"per_sounding_{orbit}.parquet", index=False)

    repro = {}
    for band in BANDS:
        d = long[(long.band == band) & (long.variant == PROD_TAG[band])].set_index("j")["k1"]
        ref = orb.meta[f"{band}_k1"][d.index.values]
        ok = np.isfinite(ref) & np.isfinite(d.values)
        repro[band] = float(np.median(np.abs(d.values[ok] - ref[ok]) / ref[ok]))
    print("reproduction of production nosg k1 by production-order variant:", repro)

    summary = aggregate(long, orb.meta, orb.fp_number)
    summary.insert(0, "orbit", orbit)
    summary.to_csv(out_dir / f"summary_{orbit}.csv", index=False)
    write_summary(summary, out_dir, orbit, repro)
    if not args.no_fig:
        figures(summary, long, orb, out_dir)
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
