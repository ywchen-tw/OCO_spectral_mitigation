"""Replication of the truncation-order study on orbits without a fitting-details file.

Runs the ``run_study`` variants on one orbit loaded with proxy cloud labels
(``rep_orbit_io.load_orbit_proxy``), aggregates with ``run_study.aggregate``
split into all / ocean / land, and adds the latitude-local near-cloud SNR of
the weighting test for ols_2, ols_3, ols_7 and gamma.

Validation mode (``--fit-details``, orbit 29252a_GL): also applies the
pre-registered proxy choice (``rep_proxy.choose_proxy``), and aggregates the
same fits under true and proxy labels so the effect of the proxy on every
metric is explicit.

Usage (from the repo root):
    python idea_test/fit_order_study/rep_run.py --l1b ... --fp-tau ... --fit-details ...   # first
    python idea_test/fit_order_study/rep_run.py --l1b ... --fp-tau ... --lite ...
    python idea_test/fit_order_study/rep_run.py --figures
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_study  # noqa: E402
from orbit_io import BANDS  # noqa: E402
from rep_orbit_io import load_orbit_proxy  # noqa: E402
from rep_proxy import (anomaly_validation, choose_proxy, label_validation,  # noqa: E402
                       proxy_distances, scatter_figure)
from run_study import NEAR_KM, PROD_TAG, md_table, robust_sd  # noqa: E402
from run_weighting import LOCAL_REF_KM, lat_local_ref  # noqa: E402
from variants import FIELDS, variant_list  # noqa: E402

OUT = HERE / "results"
SNR_TAGS = ("ols_2", "ols_3", "ols_7", "gamma")
N_BOOT = 300
MIN_SUBSET_FAR = 100
TRUE_FAR_KM = 20.0


@contextlib.contextmanager
def far_threshold(km: float):
    """run_study.aggregate reads the module constant FAR_KM; override it locally."""
    old = run_study.FAR_KM
    run_study.FAR_KM = km
    try:
        yield
    finally:
        run_study.FAR_KM = old


def fit_all(orb, variants, gamma_n: int, workers: int) -> pd.DataFrame:
    n = len(orb.sounding_id)
    rng = np.random.default_rng(0)
    gamma_idx = set(rng.choice(n, size=min(gamma_n, n), replace=False).tolist())
    # run_study._fit_rows reads its module globals (fork inherits them).
    run_study._G.update(tau=orb.tau[:, :, 1:-1], ln_t=orb.ln_t[:, :, 1:-1],
                        sig=orb.sigma_ln_t[:, :, 1:-1], variants=variants, gamma_idx=gamma_idx)
    chunks = np.array_split(np.arange(n), max(1, n // 200))
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(run_study._fit_rows, chunks)
    else:
        parts = [run_study._fit_rows(c) for c in chunks]
    rows = [r for p in parts for r in p]
    return pd.DataFrame(rows, columns=["j", "band", "variant", *FIELDS])


def subsets(meta) -> dict:
    land = meta["is_land"]
    return {"all": np.ones(len(land), bool), "ocean": ~land, "land": land}


def aggregate_split(long, meta, fp, far_km) -> pd.DataFrame:
    out = []
    for name, m in subsets(meta).items():
        jj = np.flatnonzero(m)
        sub = long[long.j.isin(jj)]
        far = (meta["cld_dist_km"][jj] > far_km) & (meta["xco2_qf"][jj] == 0)
        if far.sum() < 10:
            continue
        with far_threshold(far_km):
            s = run_study.aggregate(sub, meta, fp)
        s.insert(0, "subset", name)
        s.insert(1, "n_subset", len(jj))
        out.append(s)
    return pd.concat(out, ignore_index=True)


def local_snr(long, meta, fp, far_km, subset_mask, rng) -> pd.DataFrame:
    """Latitude-local near-cloud SNR of k1 (run_weighting.signal_metrics, snr_near_loc).

    rel = k1 / per-fp far-clear mean − 1; local = rel − median rel of reference
    footprints (cld > 10 km, QF0) within ±1° lat; SNR = median local value of
    cld < 3 km footprints / robust SD of far-clear local values.  The paired
    bootstrap resamples the near and far sets once for all variants.
    """
    cld, qf, lat = meta["cld_dist_km"], meta["xco2_qf"], meta["lat"]
    wide = long.pivot_table(index="j", columns=["band", "variant"], values="k1")
    j_all = wide.index.values
    keep = subset_mask[j_all]
    j = j_all[keep]
    far = (cld[j] > far_km) & (qf[j] == 0)
    near = cld[j] < NEAR_KM
    ref = (cld[j] > LOCAL_REF_KM) & (qf[j] == 0)
    rel_loc = {}
    for band in BANDS:
        for tag in SNR_TAGS:
            if (band, tag) not in wide.columns:
                continue
            k1 = wide[(band, tag)].values[keep]
            fpm = pd.Series(k1[far]).groupby(fp[j][far]).mean()
            rel = k1 / fpm.reindex(fp[j]).values - 1.0
            rel_loc[(band, tag)] = rel - lat_local_ref(rel, lat[j], ref)
    i_far, i_near = np.flatnonzero(far), np.flatnonzero(near)
    draws = [(rng.choice(i_far, len(i_far)), rng.choice(i_near, len(i_near))) for _ in range(N_BOOT)]

    def snr(x, a, b):
        xa, xb = x[a], x[b]
        return np.nanmedian(xb[np.isfinite(xb)]) / robust_sd(xa) if np.isfinite(xb).sum() > 10 else np.nan

    rows, boot = [], {}
    for (band, tag), x in rel_loc.items():
        ok_far = far & np.isfinite(x)
        ok_near = near & np.isfinite(x)
        boot[(band, tag)] = np.array([snr(x, a, b) for a, b in draws])
        rows.append(dict(band=band, variant=tag, n_far_loc=int(ok_far.sum()), n_near_loc=int(ok_near.sum()),
                         sd_far_loc_pct=100 * robust_sd(x[ok_far]),
                         shift_near_loc_pct=100 * np.nanmedian(x[ok_near]) if ok_near.any() else np.nan,
                         snr_near_loc=snr(x, np.flatnonzero(far), np.flatnonzero(near)),
                         snr_lo=np.nanpercentile(boot[(band, tag)], 2.5),
                         snr_hi=np.nanpercentile(boot[(band, tag)], 97.5)))
    out = pd.DataFrame(rows)
    # |SNR(ols_3)| − |SNR(ols_7)|, paired bootstrap
    out["d_abs_vs_ols7_lo"] = np.nan
    out["d_abs_vs_ols7_hi"] = np.nan
    for i, r in out.iterrows():
        if (r.band, "ols_7") in boot and r.variant != "gamma":
            d = np.abs(boot[(r.band, r.variant)]) - np.abs(boot[(r.band, "ols_7")])
            out.loc[i, ["d_abs_vs_ols7_lo", "d_abs_vs_ols7_hi"]] = np.nanpercentile(d, [2.5, 97.5])
    return out


def snr_split(long, meta, fp, far_km, seed=3) -> pd.DataFrame:
    out = []
    for name, m in subsets(meta).items():
        far = (meta["cld_dist_km"] > far_km) & (meta["xco2_qf"] == 0) & m
        if far.sum() < MIN_SUBSET_FAR:
            continue
        s = local_snr(long, meta, fp, far_km, m, np.random.default_rng(seed))
        s.insert(0, "subset", name)
        out.append(s)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- bars


def bar_verdicts(summ: pd.DataFrame, snr: pd.DataFrame, subset="all", snr_valid=True) -> dict:
    s = summ[summ.subset == subset].set_index(["band", "variant"])
    v = {}
    if s.empty or s.n_far.max() < MIN_SUBSET_FAR:
        return {"insufficient": True, "n_far": int(s.n_far.max()) if not s.empty else 0}
    sp = lambda b, t: s.loc[(b, t), "k1_spread_pct"]  # noqa: E731
    rr = lambda t: s.loc[("sco2", t), "ratio_to_o2a_spread_pct"]  # noqa: E731
    v["n_far"] = int(s.loc[("o2a", "ols_3"), "n_far"])
    v["ols3_spread"] = {b: float(sp(b, "ols_3")) for b in BANDS}
    v["ols3_ratio"] = float(rr("ols_3"))
    v["R1"] = bool(sp("o2a", "ols_3") <= 1.0 and sp("sco2", "ols_3") <= 1.0 and rr("ols_3") <= 1.5)
    v["ols7_ratio"] = float(rr("ols_7"))
    v["R2_factor"] = float(rr("ols_7") / rr("ols_3"))
    v["R2"] = bool(v["R2_factor"] >= 4.0)
    v["k2_zero_ols3"] = {b: float(s.loc[(b, "ols_3"), "k2_zero_frac"]) for b in ("o2a", "sco2")}
    v["R4_reproduces"] = bool(abs(v["k2_zero_ols3"]["o2a"] - 0.63) <= 0.15
                              and abs(v["k2_zero_ols3"]["sco2"] - 0.97) <= 0.15)
    order = {t: float(rr(t)) for t in ("ols_3", "ols_2", "gamma", "ols_7") if ("sco2", t) in s.index}
    v["ratio_by_tag"] = order
    v["R5"] = bool(all(np.diff([order[t] for t in ("ols_3", "ols_2", "gamma", "ols_7")]) > 0)) \
        if len(order) == 4 else None
    if snr_valid and snr is not None and not snr[snr.subset == subset].empty:
        q = snr[snr.subset == subset].set_index(["band", "variant"])
        v["snr"] = {b: {t: float(q.loc[(b, t), "snr_near_loc"]) for t in SNR_TAGS if (b, t) in q.index}
                    for b in BANDS}
        v["R3"] = bool(all(abs(v["snr"][b]["ols_3"]) >= abs(v["snr"][b]["ols_7"]) for b in ("o2a", "sco2")))
        v["R3_ci"] = {b: [float(q.loc[(b, "ols_3"), "d_abs_vs_ols7_lo"]),
                          float(q.loc[(b, "ols_3"), "d_abs_vs_ols7_hi"])] for b in ("o2a", "sco2")}
    return v


# --------------------------------------------------------------------------- reporting


KEY_TAGS = ("ols_1", "ols_2", "ols_3", "ols_4", "ols_5", "ols_6", "ols_7",
            "wls_2", "wls_3", "wls_7", "gamma", "gamma_u")


def compact(summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, sub), g in summ.groupby(["labels", "subset"], sort=False):
        g = g.set_index(["band", "variant"])
        for tag in KEY_TAGS:
            if ("o2a", tag) not in g.index:
                continue
            rows.append({"labels": lab, "subset": sub, "variant": tag,
                         "n_far": int(g.loc[("o2a", tag), "n_far"]),
                         "O2A spread": g.loc[("o2a", tag), "k1_spread_pct"],
                         "WCO2 spread": g.loc[("wco2", tag), "k1_spread_pct"],
                         "SCO2 spread": g.loc[("sco2", tag), "k1_spread_pct"],
                         "SCO2/O2A": g.loc[("sco2", tag), "ratio_to_o2a_spread_pct"],
                         "WCO2/O2A": g.loc[("wco2", tag), "ratio_to_o2a_spread_pct"],
                         "k2=0 O2A": g.loc[("o2a", tag), "k2_zero_frac"],
                         "k2=0 SCO2": g.loc[("sco2", tag), "k2_zero_frac"],
                         "O2A fp-mean": g.loc[("o2a", tag), "k1_fp_mean_spread_pct"],
                         "SCO2 fp-mean": g.loc[("sco2", tag), "k1_fp_mean_spread_pct"],
                         "corr(k1,anom) SCO2": g.loc[("sco2", tag), "k1_near_corr_anom"]})
    return pd.DataFrame(rows)


def write_md(orbit, choice, comp, snr, verdicts, extra_md, path):
    parts = [f"# Replication — orbit {orbit}\n",
             f"Cloud proxy {choice['proxy']}; far set = proxy > {choice['far_km']:g} km and QF0 "
             f"({choice['far_label']}); near = proxy < {NEAR_KM:g} km. Spreads: robust SD (%) after "
             "per-footprint-index mean removal, within each subset. gamma: noise-weighted, subsample.\n"]
    for (lab, sub), g in comp.groupby(["labels", "subset"], sort=False):
        parts.append(f"\n## {lab} labels — {sub}\n")
        parts.append(md_table(g.drop(columns=["labels", "subset"]), [c for c in g.columns
                                                                     if c not in ("labels", "subset")]))
    if snr is not None and len(snr):
        parts.append("\n## Local near-cloud SNR of k1 (±1° lat reference, cld > 10 km QF0)\n")
        cols = ["labels", "subset", "band", "variant", "n_far_loc", "n_near_loc", "sd_far_loc_pct",
                "shift_near_loc_pct", "snr_near_loc", "snr_lo", "snr_hi", "d_abs_vs_ols7_lo", "d_abs_vs_ols7_hi"]
        parts.append(md_table(snr, cols))
    parts.append("\n## Bar verdicts (machine-computed)\n")
    parts.append("```\n" + json.dumps(verdicts, indent=1, default=float) + "\n```")
    parts.extend(extra_md)
    path.write_text("\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b")
    ap.add_argument("--fp-tau")
    ap.add_argument("--lite", default=None)
    ap.add_argument("--fit-details", default=None)
    ap.add_argument("--orbit", default=None)
    ap.add_argument("--orders", default="1,2,3,4,5,6,7")
    ap.add_argument("--gamma-subsample", type=int, default=4000)
    ap.add_argument("--with-cut", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-soundings", type=int, default=None)
    ap.add_argument("--figures", action="store_true", help="only make the cross-orbit figures")
    args = ap.parse_args()
    if args.figures:
        figures()
        return

    orbit = args.orbit or Path(args.fp_tau).parent.name
    validation = args.fit_details is not None
    extra_md = []
    t0 = time.time()
    if validation:
        meta_v, prox = proxy_distances(args.l1b, args.fp_tau, args.fit_details)
        lab = label_validation(meta_v, prox)
        choice = choose_proxy(lab)
        anom_tab, anom_prox = anomaly_validation(meta_v, prox[choice["proxy"]])
        (OUT / "rep_proxy_choice.json").write_text(json.dumps(choice, indent=1))
        scatter_figure(meta_v, prox[choice["proxy"]], anom_prox,
                       OUT / f"rep_proxy_scatter_{orbit}.png", choice["far_km"])
        print(md_table(lab, list(lab.columns))); print(choice); print(md_table(anom_tab, list(anom_tab.columns)))
    else:
        choice = json.loads((OUT / "rep_proxy_choice.json").read_text())
    orb = load_orbit_proxy(args.l1b, args.fp_tau, lite_file=args.lite, fit_details_file=args.fit_details,
                           cloud_proxy=choice["proxy"], max_soundings=args.max_soundings)
    meta, fp = orb.meta, orb.fp_number
    n = len(orb.sounding_id)
    print(f"loaded {n} soundings in {time.time() - t0:.0f} s; land {meta['is_land'].mean():.3f}")

    orders = [int(x) for x in args.orders.split(",")]
    variants = [v for v in variant_list(orders, with_gamma=args.gamma_subsample > 0)
                if args.with_cut or v[1] != "cut"]
    t0 = time.time()
    long = fit_all(orb, variants, args.gamma_subsample, args.workers)
    print(f"fitted {len(long)} rows in {time.time() - t0:.0f} s")

    per = long.copy()
    for k in ("cld_dist_km", "xco2_qf", "xco2_bc_anomaly", "lat", "land_fraction", "cld_dist_true_km",
              "xco2_bc_anomaly_true"):
        if k in meta:
            per[k] = meta[k][per.j.values]
    per["sounding_id"] = orb.sounding_id[per.j.values]
    per["fp_number"] = fp[per.j.values]
    per.to_parquet(OUT / f"rep_per_sounding_{orbit}.parquet", index=False)

    label_sets = {"proxy": (meta, choice["far_km"])}
    if validation:
        meta_true = dict(meta, cld_dist_km=meta["cld_dist_true_km"],
                         xco2_bc_anomaly=meta["xco2_bc_anomaly_true"])
        label_sets["true"] = (meta_true, TRUE_FAR_KM)
        repro = {}
        for band in BANDS:
            d = long[(long.band == band) & (long.variant == PROD_TAG[band])].set_index("j")["k1"]
            ref = meta[f"{band}_k1"][d.index.values]
            ok = np.isfinite(ref) & np.isfinite(d.values)
            repro[band] = float(np.median(np.abs(d.values[ok] - ref[ok]) / ref[ok]))
        extra_md.append(f"\nReproduction of stored production `_nosg` k1 by PROD_TAG fits: {repro}\n")
        print("repro", repro)

    summ_parts, snr_parts, verdicts = [], [], {}
    snr_valid = True
    if not validation:
        pv = OUT / "rep_proxy_validation_29252a_GL.json"
        if pv.exists():
            snr_valid = bool(json.loads(pv.read_text()).get("P0b_pass", False))
    for lab_name, (m, far_km) in label_sets.items():
        s = aggregate_split(long, m, fp, far_km)
        s.insert(0, "labels", lab_name)
        s.insert(0, "orbit", orbit)
        summ_parts.append(s)
        q = snr_split(long, m, fp, far_km)
        q.insert(0, "labels", lab_name)
        snr_parts.append(q)
        for sub in ("all", "ocean", "land"):
            verdicts[f"{lab_name}/{sub}"] = bar_verdicts(s, q, sub, snr_valid=snr_valid or lab_name == "true")
    summ = pd.concat(summ_parts, ignore_index=True)
    snr = pd.concat(snr_parts, ignore_index=True)
    snr.insert(0, "orbit", orbit)
    summ.to_csv(OUT / f"rep_summary_{orbit}.csv", index=False)
    snr.to_csv(OUT / f"rep_snr_{orbit}.csv", index=False)
    comp = compact(summ)

    if validation:
        val = transfer_table(summ, snr)
        p0 = p0_verdicts(val, choice)
        val_md = ["# Proxy validation — 29252a_GL\n",
                  "## Label level (proxy vs MODIS cld_dist)\n", md_table(lab, list(lab.columns)),
                  f"\nChoice (pre-registered rule): {json.dumps(choice)}\n",
                  "\n## Anomaly proxy vs stored production anomaly\n", md_table(anom_tab, list(anom_tab.columns)),
                  "\n## Metric transfer (same fits, true vs proxy labels, subset all)\n",
                  md_table(val, list(val.columns)),
                  f"\n## P0 verdicts\n\n```\n{json.dumps(p0, indent=1)}\n```\n"]
        (OUT / f"rep_proxy_validation_{orbit}.md").write_text("\n".join(val_md))
        out_csv = pd.concat([lab.assign(table="labels"), anom_tab.assign(table="anomaly"),
                             val.assign(table="transfer")], ignore_index=True)
        out_csv.to_csv(OUT / f"rep_proxy_validation_{orbit}.csv", index=False)
        (OUT / f"rep_proxy_validation_{orbit}.json").write_text(json.dumps(p0, indent=1))
        print(json.dumps(p0, indent=1))
    write_md(orbit, choice, comp, snr, verdicts, extra_md, OUT / f"rep_summary_{orbit}.md")
    print(md_table(comp[comp.subset == "all"], list(comp.columns)))
    print(json.dumps(verdicts, indent=1, default=float))


def transfer_table(summ, snr) -> pd.DataFrame:
    rows = []
    a = summ[summ.subset == "all"].set_index(["labels", "band", "variant"])
    q = snr[snr.subset == "all"].set_index(["labels", "band", "variant"])
    for tag in ("ols_2", "ols_3", "ols_7", "gamma"):
        rec = {"variant": tag}
        for lab in ("true", "proxy"):
            rec[f"n_far_{lab}"] = int(a.loc[(lab, "sco2", tag), "n_far"])
            rec[f"ratio_{lab}"] = a.loc[(lab, "sco2", tag), "ratio_to_o2a_spread_pct"]
            for b in ("o2a", "sco2"):
                rec[f"spread_{b}_{lab}"] = a.loc[(lab, b, tag), "k1_spread_pct"]
                rec[f"snr_{b}_{lab}"] = q.loc[(lab, b, tag), "snr_near_loc"]
        rec["ratio_rel_diff"] = rec["ratio_proxy"] / rec["ratio_true"] - 1
        rows.append(rec)
    return pd.DataFrame(rows)


def p0_verdicts(val, choice) -> dict:
    v = val.set_index("variant")
    p0c = {t: float(v.loc[t, "ratio_rel_diff"]) for t in ("ols_2", "ols_3", "ols_7")}
    snr_ok = {}
    for t in ("ols_3", "ols_7"):
        for b in ("o2a", "sco2"):
            st, sp = v.loc[t, f"snr_{b}_true"], v.loc[t, f"snr_{b}_proxy"]
            snr_ok[f"{t}/{b}"] = dict(true=float(st), proxy=float(sp),
                                      ok=bool(np.sign(st) == np.sign(sp) and 1 / 1.5 <= sp / st <= 1.5))
    return {"P0a_pass": bool(choice["p0a_pass_at_20"]), "far_km_used": choice["far_km"],
            "far_label": choice["far_label"], "near_precision": choice["near_precision"],
            "P0b_snr": snr_ok,
            "P0b_pass": bool(choice["p0b_label_pass"] and all(x["ok"] for x in snr_ok.values())),
            "P0c_ratio_rel_diff": p0c, "P0c_pass": bool(all(abs(x) <= 0.25 for x in p0c.values()))}


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(OUT.glob("rep_summary_*.csv"))
    allsum = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allsum["kind"] = allsum.variant.str.split("_").str[0]
    allsum["order"] = pd.to_numeric(allsum.variant.str.split("_").str[1], errors="coerce")
    series = [(o, lab) for o, lab in allsum[["orbit", "labels"]].drop_duplicates().itertuples(index=False)]
    styles = {}
    for i, (o, lab) in enumerate(series):
        styles[(o, lab)] = dict(color=f"C{i}", ls="-" if lab == "proxy" else "--")

    # spread vs order per band (subset all), plus SCO2/O2A ratio
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    panels = [("o2a", "k1_spread_pct", "O2A k1 spread [%]"), ("wco2", "k1_spread_pct", "WCO2 k1 spread [%]"),
              ("sco2", "k1_spread_pct", "SCO2 k1 spread [%]"), ("sco2", "ratio_to_o2a_spread_pct", "SCO2/O2A spread [%]")]
    for ax, (band, col, ylab) in zip(axes, panels):
        for (o, lab), st in styles.items():
            d = allsum[(allsum.orbit == o) & (allsum.labels == lab) & (allsum.subset == "all")
                       & (allsum.band == band)]
            k = d[d.kind == "ols"].sort_values("order")
            ax.plot(k.order, k[col], "o", label=f"{o} ({lab})", **st)
            g = d[d.variant == "gamma"]
            if len(g):
                ax.plot([7.4], g[col].values, marker="*", ms=10, color=st["color"], ls="none")
        ax.set_yscale("log"); ax.set_xlabel("truncation order (ols); ★ = gamma"); ax.set_ylabel(ylab)
        ax.axhline(1.0, color="0.6", lw=0.6); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Far-clear k1 scatter vs truncation order — replication (solid: proxy labels, dashed: MODIS labels)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "rep_spread_vs_order.png", dpi=130); plt.close(fig)

    # land vs ocean (proxy labels)
    d0 = allsum[(allsum.labels == "proxy") & allsum.subset.isin(["ocean", "land"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (band, col, ylab) in zip(axes, [panels[0], panels[2], panels[3]]):
        for i, ((o, sub), g) in enumerate(d0.groupby(["orbit", "subset"])):
            k = g[(g.band == band) & (g.kind == "ols")].sort_values("order")
            n_far = int(k.n_far.max()) if len(k) else 0
            ax.plot(k.order, k[col], "o-" if sub == "ocean" else "s--",
                    color=f"C{sorted(d0.orbit.unique()).index(o)}", label=f"{o} {sub} (n_far={n_far})")
        ax.set_yscale("log"); ax.set_xlabel("truncation order (ols)"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Land vs ocean (proxy labels)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "rep_land_ocean.png", dpi=130); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
