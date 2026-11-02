"""PPDF direct XCO2, round 2: surface-binned and pooled path models, SCO2+WCO2
combination, prior blends, scores against xco2_bc AND xco2_apriori, TCCON check and a
near-cloud diagnostic.  See estimators_v2.py for the estimator definitions.

Stages (run in order; each reads the previous stage's files):
  perdate  per date (parallel): per-date estimators, combos, blends -> compact parquet
  pooled   pooled path models (Ridge, HistGradientBoosting) per surface: 5 date folds of
           fm1620k5 (held-out date blocks) for the 116 dates; a model on all 116 dates
           for every other date (first-pass + TCCON dates).  Adds pooled estimators,
           their per-date recalibration / combos / blends, then all metrics.
  tccon    collocate every AVAIL=yes run_case of the DE plotting script and score
           xco2_bc, xco2_apriori and the PPDF estimates against TCCON (direct window
           mean and AK-harmonized reference; shared helpers of tccon_comparison_report).
  neardiag ocean: E0_am / k1_am vs xco2_bc_anomaly_r05 by cloud-distance bin
  summary  summary.md

    python idea_test/ppdf_direct_xco2/run_v2.py perdate --workers 12
    python idea_test/ppdf_direct_xco2/run_v2.py pooled
    python idea_test/ppdf_direct_xco2/run_v2.py tccon
    python idea_test/ppdf_direct_xco2/run_v2.py neardiag
    python idea_test/ppdf_direct_xco2/run_v2.py summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from estimators import far_clear_mask, load_columns, path_features  # noqa: E402
from estimators_v2 import (PERDATE_SPECS, add_blend, add_combo, blend_slopes,  # noqa: E402
                           prepare)
import estimators_v2  # noqa: E402

PARQUET_DIR = Path("/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/csv_collection")
COMBINED = PARQUET_DIR / "combined_2016_2020_dates.parquet"
FOLD_DIR = Path("/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/model_deep_ensemble/am")
TCCON_DIR = Path("/scratch/alpine/yuch8913/oco/spec_anal/data/TCCON")
CASE_SCRIPT = ROOT / "curc_shell_blanca_plot_corr_xco2_deepens.sh"
SCRATCH = Path("/projects/yuch8913/code-server/tmp/claude-777013/-projects-yuch8913-OCO-spectral-mitigation/"
               "dbbff218-f8c4-4618-9e1f-97576c0dfb85/scratchpad/ppdf_test/v2")
COMPACT = SCRATCH / "compact"
OUT = HERE / "results_2016_2020_v2"
FIRST_PASS_DATES = ["2018-07-10", "2021-09-06", "2023-06-26", "2019-06-22", "2017-10-09"]
SURFACE = {0.0: "ocean", 1.0: "land"}
SUBSETS = {
    "far_heldout": lambda d: far_clear_mask(d),
    "near_0_3": lambda d: (d.cld_dist_km > 0) & (d.cld_dist_km < 3),
    "near_3_10": lambda d: (d.cld_dist_km >= 3) & (d.cld_dist_km < 10),
}
BAR_FAR_SD, BAR_NEAR_CORR, BAR_NEAR_SD = 2.0, 0.3, 5.0
POOL_FEATS = path_features("am")
POOL_TARGETS = {"sco2": "k1pos_sco2_am", "wco2": "k1pos_wco2_am"}
META = ["date", "orbit_id", "time", "fp", "lat", "lon", "sfc_type", "xco2_qf", "cld_dist_km",
        "xco2_apriori", "xco2_bc", "xco2_bc_anomaly", "xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15",
        "fold"]
POOL_SPECS = {   # id -> (band, model)
    "E2P_am": ("sco2", "ridge"), "E2wP_am": ("wco2", "ridge"),
    "E2P_am_hgb": ("sco2", "hgb"), "E2wP_am_hgb": ("wco2", "hgb"),
}
POOL_RECAL = [("E2P_am_hgb_rc", "E2P_am_hgb"), ("E2wP_am_hgb_rc", "E2wP_am_hgb"),
              ("E2P_am_rc", "E2P_am"), ("E2wP_am_rc", "E2wP_am")]
POOL_COMBOS = [("E2cP_am_hgb", "E2P_am_hgb", "E2wP_am_hgb"),
               ("E2cP_am_hgb_rc", "E2P_am_hgb_rc", "E2wP_am_hgb_rc")]
POOL_BLENDS = [("BL_E2cP_am_hgb_rc", ["E2cP_am_hgb_rc"]),
               ("BL2P_am_hgb_rc", ["E2P_am_hgb_rc", "E2wP_am_hgb_rc"])]
# blends whose slope is pooled over the training dates of each date fold (all 116 for others)
POOLSLOPE_BLENDS = [("BLP_E0_am", ["E0_am"]), ("BLP_E2w_am", ["E2w_am"]),
                    ("BLP_E2c_am", ["E2c_am"]), ("BLP_E2c_am_ab", ["E2c_am_ab"]),
                    ("BLP2_am", ["E2_am", "E2w_am"]),
                    ("BLP_E2cP_am_hgb_rc", ["E2cP_am_hgb_rc"]),
                    ("BLP2P_am_hgb_rc", ["E2P_am_hgb_rc", "E2wP_am_hgb_rc"])]


def est_ids(df):
    return [c[4:] for c in df.columns if c.startswith("est_")]


# ── stage: perdate ──────────────────────────────────────────────────────────────

def combined_index():
    f = pq.ParquetFile(COMBINED)
    i = f.schema_arrow.names.index("date")
    return {f.metadata.row_group(g).column(i).statistics.min.decode(): g
            for g in range(f.metadata.num_row_groups)}


def load_date(date, rg_index):
    extra = ["xco2_bc_anomaly_r05"]
    if date in rg_index:
        f = pq.ParquetFile(COMBINED)
        cols = load_columns(set(f.schema_arrow.names)) + extra
        df = f.read_row_group(rg_index[date], columns=cols).to_pandas()
    else:
        f = pq.ParquetFile(PARQUET_DIR / f"combined_{date}_all_orbits.parquet")
        df = f.read(columns=load_columns(set(f.schema_arrow.names)) + extra).to_pandas()
    df["date"] = date
    return prepare(df)


def tccon_cases():
    cases = []
    for ln in CASE_SCRIPT.read_text().splitlines():
        if not re.match(r"^run_case\s", ln):
            continue
        t = ln.split("#")[0].split()
        if len(t) < 13 or t[12] != "yes":
            continue
        cases.append(dict(date=t[1], tccon=t[2], box=tuple(map(float, t[3:7])), surf=t[9],
                          site=t[11]))
    return cases


def compact_frame(df):
    keep = META + [f for f in POOL_FEATS if f in df] + list(POOL_TARGETS.values()) \
        + [c for c in df.columns if c.startswith("est_")]
    out = df[keep].copy()
    out.attrs = {}
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in ("time", "lat", "lon"):
            out[c] = out[c].astype(np.float32)
    out["orbit_id"] = out["orbit_id"].astype(str)
    return out


def perdate_one(date, rg_index, skip):
    path = COMPACT / f"{date}.parquet"
    if skip and path.exists():
        return f"[{date}] exists"
    df = estimators_v2.run_perdate(load_date(date, rg_index))
    pd.DataFrame(df.attrs.get("weights", [])).assign(date=date).to_csv(
        SCRATCH / "weights" / f"{date}.csv", index=False)
    compact_frame(df).to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").rename(path)
    return f"[{date}] n={len(df)}"


def stage_perdate(args):
    for d in (COMPACT, SCRATCH / "weights"):
        d.mkdir(parents=True, exist_ok=True)
    rg_index = combined_index()
    dates = sorted(rg_index) + FIRST_PASS_DATES + sorted({c["date"] for c in tccon_cases()})
    dates = list(dict.fromkeys(dates))
    dates = [d for d in dates if d in rg_index or (PARQUET_DIR / f"combined_{d}_all_orbits.parquet").exists()]
    print(f"{len(dates)} dates", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(perdate_one, d, rg_index, args.skip_existing): d for d in dates}
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as exc:
                print(f"[{futs[fu]}] FAILED {exc!r}", flush=True)


# ── stage: pooled ───────────────────────────────────────────────────────────────

def fit_pooled(train, target, model, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    feats = [f for f in POOL_FEATS if f in train and train[f].notna().mean() > 0.5]
    t = train.dropna(subset=[target])
    if model == "ridge":
        t = t.dropna(subset=feats)
        if len(t) > 3_000_000:
            t = t.sample(3_000_000, random_state=seed)
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(t[feats], t[target])
    else:
        if len(t) > 800_000:
            t = t.sample(800_000, random_state=seed)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          min_samples_leaf=100, early_stopping=True,
                                          validation_fraction=0.1, random_state=seed)
        m.fit(t[feats].to_numpy(np.float32), t[target].to_numpy(np.float32))
    return dict(model=m, feats=feats, med=float(t[target].median()), kind=model, n=len(t))


def predict_pooled(fm, df, target):
    x = df[fm["feats"]]
    pred = np.full(len(df), np.nan)
    if fm["kind"] == "ridge":
        ok = x.notna().all(axis=1).to_numpy()
        if ok.any():
            pred[ok] = fm["model"].predict(x[ok])
    else:
        pred = fm["model"].predict(x.to_numpy(np.float32))
    pred = np.where((pred > 0.5 * fm["med"]) & (pred < 1.5 * fm["med"]), pred, np.nan)
    f = df[target].to_numpy(float) / pred
    f = np.where(np.abs(f - 1) <= 0.2, f, np.nan)
    return df["xco2_apriori"].to_numpy(float) * f


def add_recal(df, dst, src):
    """Per date x surface x orbit fold: divide f by its mean over the OTHER fold's far rows."""
    res = np.full(len(df), np.nan)
    pri = df["xco2_apriori"].to_numpy(float)
    f = df[f"est_{src}"].to_numpy(float) / pri
    far = far_clear_mask(df).to_numpy()
    sfc, fold = df["sfc_type"].to_numpy(), df["fold"].to_numpy()
    for s in np.unique(sfc[np.isfinite(sfc)]):
        for k in np.unique(fold[(sfc == s) & (fold >= 0)]):
            cal = (sfc == s) & (fold != k) & (fold >= 0) & far & np.isfinite(f)
            app = (sfc == s) & (fold == k)
            if cal.sum() >= 50:
                g = f[app] / np.mean(f[cal])
                res[app] = pri[app] * np.where(np.abs(g - 1) <= 0.2, g, np.nan)
    df[f"est_{dst}"] = res


def pooled_derived(df):
    """Recalibrations, combos and blends of the pooled estimates, per date."""
    weights = []
    for cid, src in POOL_RECAL:
        add_recal(df, cid, src)
    for cid, s, w in POOL_COMBOS:
        weights += add_combo(df, cid, s, w)
    for bid, ins in POOL_BLENDS:
        weights += add_blend(df, bid, ins)
    return weights


def rsd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() <= 20 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def metrics_for(df, date):
    rows = []
    for sfc, g in df.groupby("sfc_type"):
        for sub, fn in SUBSETS.items():
            s = g[fn(g).to_numpy()]
            if len(s) < 20:
                continue
            bc, pri = s["xco2_bc"].to_numpy(float), s["xco2_apriori"].to_numpy(float)
            an, an5 = s["xco2_bc_anomaly"].to_numpy(float), s["xco2_bc_anomaly_r05"].to_numpy(float)
            rows.append(dict(date=date, surface=SURFACE.get(sfc, str(sfc)), subset=sub,
                             estimator="xco2_bc_vs_prior", n=len(s), frac_valid=1.0,
                             rsd_vs_bc=np.nan, rsd_vs_prior=rsd(bc - pri),
                             bias_vs_prior=float(np.mean(bc - pri)), rsd_bc_prior_same=rsd(bc - pri)))
            for e in est_ids(s):
                v = s[f"est_{e}"].to_numpy(float)
                ok = np.isfinite(v)
                if ok.sum() < 20:
                    continue
                rb, rp, dbp = v[ok] - bc[ok], v[ok] - pri[ok], bc[ok] - pri[ok]
                rows.append(dict(
                    date=date, surface=SURFACE.get(sfc, str(sfc)), subset=sub, estimator=e,
                    n=int(ok.sum()), frac_valid=ok.mean(),
                    bias_vs_bc=float(np.mean(rb)), rsd_vs_bc=rsd(rb),
                    bias_vs_prior=float(np.mean(rp)), rsd_vs_prior=rsd(rp),
                    rsd_bc_prior_same=rsd(dbp),
                    corr_anomaly=_corr(rb, an[ok]), corr_anomaly_r05=_corr(rb, an5[ok]),
                    corr_bc_dev=_corr(rp, dbp)))
    return pd.DataFrame(rows)


def fold_dates():
    out = []
    for k in range(5):
        d = json.loads((FOLD_DIR / f"fm1620k5_ocean_full_sig_f{k}" / "training_dates.json").read_text())
        out.append(set(d["held_dates"]))
    return out


def stage_pooled(args):
    import joblib
    OUT.mkdir(parents=True, exist_ok=True)
    rg_dates = sorted(combined_index())
    held = fold_dates()
    assert set().union(*held) == set(rg_dates), "fold held_dates do not cover the 116 dates"
    print("loading compact frames", flush=True)
    big = pd.concat([pd.read_parquet(COMPACT / f"{d}.parquet") for d in rg_dates], ignore_index=True)
    print(f"{len(big)} rows, {big.memory_usage().sum() / 1e9:.1f} GB", flush=True)
    far = far_clear_mask(big).to_numpy()
    for e in POOL_SPECS:
        big[f"est_{e}"] = np.nan
    big["pool_fold"] = -1
    info = []
    for k, H in enumerate(held):
        is_held = big["date"].isin(H).to_numpy()
        big.loc[is_held, "pool_fold"] = k
        for s in (0.0, 1.0):
            sm = (big["sfc_type"] == s).to_numpy()
            train = big[far & sm & ~is_held]
            app = np.where(sm & is_held)[0]
            for e, (band, model) in POOL_SPECS.items():
                fm = fit_pooled(train, POOL_TARGETS[band], model, seed=k)
                big.loc[app, f"est_{e}"] = predict_pooled(fm, big.iloc[app], POOL_TARGETS[band])
                info.append(dict(fold=k, sfc_type=s, est=e, n_train=fm["n"]))
                print(f"fold {k} sfc {s} {e}: n_train {fm['n']}", flush=True)
    pd.DataFrame(info).to_csv(OUT / "pooled_training.csv", index=False)
    # all-116 models for the other dates
    models = {}
    for s in (0.0, 1.0):
        train = big[far & (big["sfc_type"] == s).to_numpy()]
        for e, (band, model) in POOL_SPECS.items():
            models[(s, e)] = fit_pooled(train, POOL_TARGETS[band], model, seed=99)
            print(f"all-116 sfc {s} {e}", flush=True)
    joblib.dump(models, SCRATCH / "pooled_all116.joblib")

    weights, metrics = [], []
    frames = {}
    for d, g in big.groupby("date", sort=True):
        g = g.copy()
        weights += [dict(x, date=d) for x in pooled_derived(g)]
        frames[d] = g
    del big
    # pooled blend slopes: per date fold from its training dates, plus all 116 dates
    slopes = {}
    for bid, ins in POOLSLOPE_BLENDS:
        for k, H in enumerate(held):
            slopes[(bid, k)] = blend_slopes([frames[d] for d in rg_dates if d not in H], ins)
        slopes[(bid, "all")] = blend_slopes([frames[d] for d in rg_dates], ins)
    pd.DataFrame([dict(est=b, fold=k, sfc_type=s_, **{f"w{i}": v for i, v in enumerate(w)})
                  for (b, k), sl in slopes.items() for s_, w in sl.items()]).to_csv(
        OUT / "pooled_blend_slopes.csv", index=False)
    fold_of = {d: k for k, H in enumerate(held) for d in H}
    for d, g in frames.items():
        for bid, ins in POOLSLOPE_BLENDS:
            weights += [dict(x, date=d) for x in add_blend(g, bid, ins, slopes=slopes[(bid, fold_of[d])])]
        metrics.append(metrics_for(g, d))
        g.drop(columns=["pool_fold"]).to_parquet(COMPACT / f"{d}.parquet", index=False)
    del frames
    other = sorted(p.stem for p in COMPACT.glob("*.parquet") if p.stem not in set(rg_dates))
    for d in other:
        g = pd.read_parquet(COMPACT / f"{d}.parquet")
        for s in (0.0, 1.0):
            sm = np.where((g["sfc_type"] == s).to_numpy())[0]
            for e, (band, _) in POOL_SPECS.items():
                if f"est_{e}" not in g:
                    g[f"est_{e}"] = np.nan
                if len(sm):
                    g.loc[g.index[sm], f"est_{e}"] = predict_pooled(models[(s, e)], g.iloc[sm], POOL_TARGETS[band])
        weights += [dict(x, date=d) for x in pooled_derived(g)]
        for bid, ins in POOLSLOPE_BLENDS:
            weights += [dict(x, date=d) for x in add_blend(g, bid, ins, slopes=slopes[(bid, "all")])]
        metrics.append(metrics_for(g, d))
        g.to_parquet(COMPACT / f"{d}.parquet", index=False)
    pd.DataFrame(weights).to_csv(OUT / "pooled_weights.csv", index=False)
    pw = pd.concat([pd.read_csv(p) for p in sorted((SCRATCH / "weights").glob("*.csv"))
                    if p.stat().st_size > 1], ignore_index=True)
    pw.to_csv(OUT / "perdate_weights.csv", index=False)
    m = pd.concat(metrics, ignore_index=True)
    m["main"] = m["date"].isin(rg_dates)
    m.to_csv(OUT / "metrics_all.csv", index=False)
    (SCRATCH / "pooled.done").write_text("ok")
    print(f"metrics: {len(m)} rows, {m.date.nunique()} dates", flush=True)


# ── stage: opblend (fully operational prior blends; no same-date xco2_bc) ───────
# OB_<id> = prior + a + w (est_id - prior), (a, w) by trimmed OLS on the far-cloud QF0
# rows of the fold's TRAINING dates only; scored on the held-out date block.  Unlike N0 /
# BL* / BLP* (whose offset is the same date's mean(bc - prior)), nothing from the scored
# date's xco2_bc enters, so this is what a fast retrieval could actually run.
OP_BLENDS = [("OB_E0_am", ["E0_am"]), ("OB_E2c_am", ["E2c_am"]), ("OB_E2w_am", ["E2w_am"]),
             ("OB_E2wP_am", ["E2wP_am"]), ("OB_E2cP_am_hgb", ["E2cP_am_hgb"]),
             ("OB2P_am_hgb", ["E2P_am_hgb", "E2wP_am_hgb"]),
             ("OB3_am", ["E0_am", "E2c_am", "E2cP_am_hgb"])]


def stage_opblend(args):
    from estimators_v2 import _trim
    rg_dates = sorted(combined_index())
    held = fold_dates()
    need = sorted({i for _, ins in OP_BLENDS for i in ins})
    cols = META + [f"est_{i}" for i in need]
    frames = {d: pd.read_parquet(COMPACT / f"{d}.parquet", columns=cols) for d in rg_dates}
    coefs, metrics = [], []
    for k, H in enumerate(held):
        train = pd.concat([frames[d] for d in rg_dates if d not in H], ignore_index=True)
        train = train[far_clear_mask(train).to_numpy()]
        fits = {}
        for s in (0.0, 1.0):
            t = train[train.sfc_type == s]
            pri = t.xco2_apriori.to_numpy(float)
            y = t.xco2_bc.to_numpy(float) - pri
            for bid, ins in OP_BLENDS:
                X = np.column_stack([t[f"est_{i}"].to_numpy(float) - pri for i in ins])
                m = _trim(y, *X.T)
                A = np.column_stack([np.ones(m.sum()), X[m]])
                c = np.linalg.lstsq(A, y[m], rcond=None)[0]
                fits[(s, bid)] = c
                coefs.append(dict(fold=k, sfc_type=s, est=bid, a=c[0],
                                  **{f"w_{i}": v for i, v in zip(ins, c[1:])}))
        for d in sorted(H):
            g = frames[d].copy()
            pri = g.xco2_apriori.to_numpy(float)
            for bid, ins in OP_BLENDS:
                res = np.full(len(g), np.nan)
                X = np.column_stack([g[f"est_{i}"].to_numpy(float) - pri for i in ins])
                for s in (0.0, 1.0):
                    sm = (g.sfc_type == s).to_numpy()
                    c = fits[(s, bid)]
                    res[sm] = pri[sm] + c[0] + X[sm] @ c[1:]
                g[f"est_{bid}"] = res
            g = g[["date", "sfc_type", "xco2_qf", "cld_dist_km", "xco2_apriori", "xco2_bc",
                   "xco2_bc_anomaly", "xco2_bc_anomaly_r05"] + [f"est_{b}" for b, _ in OP_BLENDS]]
            metrics.append(metrics_for(g, d))
    pd.DataFrame(coefs).to_csv(OUT / "opblend_coefs.csv", index=False)
    mo = pd.concat(metrics, ignore_index=True)
    mo = mo[mo.estimator != "xco2_bc_vs_prior"]
    mo["main"] = True
    m = pd.read_csv(OUT / "metrics_all.csv")
    m = m[~m.estimator.str.startswith("OB")]
    pd.concat([m, mo], ignore_index=True).to_csv(OUT / "metrics_all.csv", index=False)
    print(pd.DataFrame(coefs).groupby(["est", "sfc_type"]).median().round(3).to_string())


# ── stage: tccon ────────────────────────────────────────────────────────────────

def stage_tccon(args):
    # Dropped by the user on 2026-11-01 (scope: far-cloud fast-retrieval question only).
    # The collocation code below is kept but not run.
    print("tccon stage disabled (out of scope)", flush=True)
    return
    sys.path.insert(0, str(ROOT / "workspace"))
    sys.path.insert(0, str(ROOT / "src"))
    from tccon_collocate import PRODUCTION_SITE_RADII, collocate, site_radius_km
    from plot_corrected_xco2 import load_tccon
    from ak_harmonize import ak_adjusted_ref_from_operator, operator_from_dataframe
    tc_cache, rows = {}, []
    for c in tccon_cases():
        cp = COMPACT / f"{c['date']}.parquet"
        if not cp.exists():
            print(f"[{c['date']} {c['site']}] no compact file", flush=True)
            continue
        oco = pd.read_parquet(cp)
        src = PARQUET_DIR / f"combined_{c['date']}_all_orbits.parquet"
        names = pq.ParquetFile(src).schema_arrow.names
        akc = [n for n in names if re.match(r"^(ak|pwf|co2_ap|plev)_\d+$", n)]
        if akc:
            a = pd.read_parquet(src, columns=["time", "fp"] + akc).drop_duplicates(["time", "fp"])
            oco = oco.merge(a, on=["time", "fp"], how="left")
        tpath = TCCON_DIR / c["tccon"]
        if not tpath.exists():
            hits = sorted(TCCON_DIR.glob(f"{c['tccon'][:2]}*.public.qc.nc"))
            if len(hits) != 1:
                print(f"[{c['date']} {c['site']}] TCCON file not found", flush=True)
                continue
            tpath = hits[0]
        if tpath not in tc_cache:
            tc_cache[tpath] = load_tccon(str(tpath))
        code = c["tccon"][:2]
        radius = site_radius_km(code, 100.0, PRODUCTION_SITE_RADII)
        col = collocate(oco, tc_cache[tpath], box=c["box"], radius_km=radius, window_min=60.0,
                        site=code, site_radii=PRODUCTION_SITE_RADII)
        near, tmu = col["near"], col["tccon_ref"]
        if not len(near) or not np.isfinite(tmu):
            continue
        tak = np.nan
        try:
            ot = pd.to_datetime(near["time"], unit="s", utc=True)
            op = operator_from_dataframe(near)
            adj = ak_adjusted_ref_from_operator(op, str(tpath), ot.min().tz_localize(None),
                                                ot.max().tz_localize(None), window_min=60.0)
            if adj is not None:
                tak = adj["tccon_ref_ak"]
        except Exception as exc:  # noqa: BLE001
            print(f"[{c['date']} {c['site']}] AK failed: {exc}", flush=True)
        series = ["xco2_bc", "xco2_apriori"] + [f"est_{e}" for e in est_ids(near)]
        for sname, sm in (("all", np.ones(len(near), bool)),
                          ("ocean", (near["sfc_type"] == 0).to_numpy()),
                          ("land", (near["sfc_type"] == 1).to_numpy())):
            for qname, qm in (("all", np.ones(len(near), bool)), ("qf0", (near["xco2_qf"] == 0).to_numpy())):
                g = near[sm & qm]
                if len(g) < 10:
                    continue
                for ser in series:
                    v = g[ser].to_numpy(float)
                    v = v[np.isfinite(v)]
                    if v.size < 10:
                        continue
                    rows.append(dict(date=c["date"], site=c["site"], surface=sname, qf=qname,
                                     series=ser.replace("est_", ""), n_fp=v.size,
                                     n_tccon=col["n_tccon"], tccon_direct=tmu, tccon_ak=tak,
                                     mean=float(v.mean()), sd=float(v.std()),
                                     bias_direct=float(v.mean() - tmu), bias_ak=float(v.mean() - tak),
                                     fp_rmse_direct=float(np.sqrt(np.mean((v - tmu) ** 2)))))
        print(f"[{c['date']} {c['site']}] n_near={len(near)} tccon={tmu:.2f} ak={tak:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "tccon_cases.csv", index=False)


# ── stage: neardiag ─────────────────────────────────────────────────────────────

CLD_BINS = [0, 1, 2, 4, 6, 10, 20, np.inf]


def stage_neardiag(args):
    rg_dates = sorted(combined_index())
    rows = []
    for d in rg_dates:
        g = pd.read_parquet(COMPACT / f"{d}.parquet")
        g = g[g["sfc_type"] == 0]
        pri = g["xco2_apriori"].to_numpy(float)
        far = far_clear_mask(g).to_numpy()
        v = {"E0_am-prior": g["est_E0_am"].to_numpy(float) - pri,
             "E0_am-bc": g["est_E0_am"].to_numpy(float) - g["xco2_bc"].to_numpy(float),
             "sco2_k1_am": g["k1pos_sco2_am"].to_numpy(float),
             "wco2_k1_am": g["k1pos_wco2_am"].to_numpy(float),
             "E2c_am-prior": g["est_E2c_am"].to_numpy(float) - pri,
             "bc-prior": g["xco2_bc"].to_numpy(float) - pri}
        an = g["xco2_bc_anomaly_r05"].to_numpy(float)
        farmean = {k: np.nanmean(x[far]) for k, x in v.items()}
        cd = g["cld_dist_km"].to_numpy(float)
        qf0 = (g["xco2_qf"] == 0).to_numpy()
        for lo, hi in zip(CLD_BINS[:-1], CLD_BINS[1:]):
            m = (cd >= lo) & (cd < hi)
            if m.sum() < 30:
                continue
            r = dict(date=d, bin=f"{lo:g}-{hi:g}", n=int(m.sum()), mean_anom_r05=float(np.nanmean(an[m])))
            for k, x in v.items():
                r[f"corr_{k}"] = _corr(x[m], an[m])
                r[f"corr_qf0_{k}"] = _corr(x[m & qf0], an[m & qf0])
                r[f"shift_{k}"] = float(np.nanmean(x[m]) - farmean[k])
                r[f"dm_{k}"] = x[m] - np.nanmean(x[m])   # for the pooled correlation
            r["dm_anom"] = an[m] - np.nanmean(an[m])
            rows.append(r)
    keys = ["E0_am-prior", "E0_am-bc", "sco2_k1_am", "wco2_k1_am", "E2c_am-prior", "bc-prior"]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("dm_")} for r in rows])
    df.to_csv(OUT / "neardiag_ocean_perdate.csv", index=False)
    agg = []
    for b in df["bin"].unique():
        rr = [r for r in rows if r["bin"] == b]
        a = dict(bin=b, n_dates=len(rr), n_fp=sum(r["n"] for r in rr),
                 mean_anom_r05=np.nanmedian([r["mean_anom_r05"] for r in rr]))
        anom = np.concatenate([r["dm_anom"] for r in rr])
        for k in keys:
            a[f"medcorr_{k}"] = np.nanmedian([r[f"corr_{k}"] for r in rr])
            a[f"medcorr_qf0_{k}"] = np.nanmedian([r[f"corr_qf0_{k}"] for r in rr])
            a[f"poolcorr_{k}"] = _corr(np.concatenate([r[f"dm_{k}"] for r in rr]), anom)
            a[f"shift_{k}"] = np.nanmedian([r[f"shift_{k}"] for r in rr])
        agg.append(a)
    pd.DataFrame(agg).to_csv(OUT / "neardiag_ocean.csv", index=False)
    print(pd.DataFrame(agg).T.to_string(), flush=True)


# ── stage: summary ──────────────────────────────────────────────────────────────

def md_table(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def headline(m):
    far = m[m.subset == "far_heldout"].copy()
    near = m[m.subset == "near_0_3"].copy()
    for t in (far, near):
        null = t[t.estimator == "N0_prior"].set_index(["date", "surface"]).rsd_vs_bc
        key = pd.MultiIndex.from_frame(t[["date", "surface"]])
        t["over_null"] = t.rsd_vs_bc.to_numpy() / null.reindex(key).to_numpy()
        t["over_same"] = t.rsd_vs_bc / t.rsd_bc_prior_same
    g, gn = far.groupby("estimator"), near.groupby("estimator")
    h = pd.DataFrame({
        "far_vs_bc": g.rsd_vs_bc.median(), "far_vs_prior": g.rsd_vs_prior.median(),
        "far_bc_vs_prior": g.rsd_bc_prior_same.median(),
        "far/N0": g.over_null.median(), "far/N0_same": g.over_same.median(),
        "far_R2": g.over_same.apply(lambda x: 1 - np.nanmedian(x) ** 2),
        "far_corr_bcdev": g.corr_bc_dev.median(),
        "near_vs_bc": gn.rsd_vs_bc.median(), "near_vs_prior": gn.rsd_vs_prior.median(),
        "near/N0": gn.over_null.median(), "near_corr": gn.corr_anomaly.median(),
        "near_corr_r05": gn.corr_anomaly_r05.median(),
        "far_fvalid": g.frac_valid.median(), "near_fvalid": gn.frac_valid.median(),
        "share_far<N0": g.over_null.apply(lambda x: float(np.mean(x < 1))),
        "n_cells": g.size(),
    })
    tri = far[far.estimator == "xco2_bc_vs_prior"]
    h.loc["xco2_bc_vs_prior", "far_vs_prior"] = tri.rsd_vs_prior.median()
    h.loc["xco2_bc_vs_prior", "near_vs_prior"] = near[near.estimator == "xco2_bc_vs_prior"].rsd_vs_prior.median()
    return h


ORDER = ["xco2_bc_vs_prior", "N0_prior", "B_IDP", "E2_o2", "E2w_o2", "E2_t2", "E2w_t2", "E0_am",
         "E2_am", "E2w_am", "E2_am_scr", "E2w_am_scr", "E2_am_ab", "E2w_am_ab", "E2_am_sc",
         "E2w_am_sc", "E2c_am", "E2c_am_ab", "E2P_am", "E2wP_am", "E2P_am_rc", "E2wP_am_rc",
         "E2P_am_hgb", "E2wP_am_hgb", "E2P_am_hgb_rc", "E2wP_am_hgb_rc", "E2cP_am_hgb",
         "E2cP_am_hgb_rc", "BL_E0_am", "BL_E2_am", "BL_E2w_am", "BL_E2c_am", "BL_E2c_am_ab",
         "BL2_am", "BL_E2cP_am_hgb_rc", "BL2P_am_hgb_rc", "BLP_E0_am", "BLP_E2w_am", "BLP_E2c_am",
         "BLP_E2c_am_ab", "BLP2_am", "BLP_E2cP_am_hgb_rc", "BLP2P_am_hgb_rc",
         "OB_E0_am", "OB_E2c_am", "OB_E2w_am", "OB_E2wP_am", "OB_E2cP_am_hgb", "OB2P_am_hgb", "OB3_am"]


def _ord(df):
    idx = [e for e in ORDER if e in df.index] + [e for e in df.index if e not in ORDER]
    return df.loc[idx]


def tccon_tables(t, primary):
    out = []
    for ref in ("direct", "ak"):
        for qf in ("all", "qf0"):
            for surf in ("all", "ocean", "land"):
                s = t[(t.qf == qf) & (t.surface == surf)]
                if s.empty:
                    continue
                b = f"bias_{ref}"
                # common case set: cases where bc, prior and the primary PPDF are all present
                need = {"xco2_bc", "xco2_apriori", primary}
                ok = s.groupby(["date", "site"]).series.apply(lambda x: need <= set(x))
                ok = ok[ok].index
                sc = s.set_index(["date", "site"]).loc[ok].reset_index()
                g = sc.groupby("series")[b]
                tab = pd.DataFrame({"n_cases": g.count(), "mean_bias": g.mean(), "sd_bias": g.std(),
                                    "rmse": g.apply(lambda x: float(np.sqrt(np.nanmean(x ** 2)))),
                                    "mae": g.apply(lambda x: float(np.nanmean(np.abs(x))))})
                tab.index = ["xco2_prior" if i == "xco2_apriori" else i for i in tab.index]
                out.append((ref, qf, surf, tab))
    return out


def stage_summary(args):
    m = pd.read_csv(OUT / "metrics_all.csv")
    main = m[m.main]
    h = _ord(headline(main))
    candidates = h.drop(index=["xco2_bc_vs_prior", "N0_prior", "B_IDP"], errors="ignore")
    primary = candidates["far/N0"].idxmin()
    L = ["# Summary — PPDF direct XCO2, round 2 (2026-11-01)", "",
         "rsd = robust SD (1.4826·MAD, ppm); medians over date × surface cells of the 116 "
         "2016–2020 dates. `far_vs_bc` / `far_vs_prior` = rsd(est − xco2_bc) / rsd(est − "
         "xco2_apriori) on far-cloud QF0 held-out footprints; `far_bc_vs_prior` = rsd(xco2_bc − "
         "prior) on the same footprints (the third side of the triangle). `far/N0` = cell rsd "
         "vs bc ÷ rsd of the prior-only null N0_prior in the cell (criterion: < 1); `far/N0_same` "
         "uses the estimator's own valid footprints; `far_R2` = 1 − (far/N0_same)², the robust "
         "out-of-sample share of var(bc − prior) explained. `near_*` = cld_dist < 3 km, all QF. "
         "`near_corr(_r05)` = corr(est − bc, xco2_bc_anomaly(_r05)) (pre-registered diagnostic "
         "bar). Blends (BL*) use xco2_bc on the other orbit fold's far-cloud rows for (a, w); "
         "no estimator uses ref_*/r15_*/r05_* statistics.", "",
         f"## Main table ({main.date.nunique()} dates)", "", md_table(h.round(3)), ""]
    for surf in ("ocean", "land"):
        hs = _ord(headline(main[main.surface == surf]))
        L += [f"### {surf}", "", md_table(hs[["far_vs_bc", "far_vs_prior", "far_bc_vs_prior", "far/N0",
                                              "far_R2", "far_corr_bcdev", "near_vs_bc", "near/N0",
                                              "near_corr", "far_fvalid", "share_far<N0"]].round(3)), ""]
    far = main[main.subset == "far_heldout"].copy()
    far["year"] = far.date.str[:4]
    null = far[far.estimator == "N0_prior"].set_index(["date", "surface"]).rsd_vs_bc
    far["over_null"] = far.rsd_vs_bc.to_numpy() / null.reindex(pd.MultiIndex.from_frame(far[["date", "surface"]])).to_numpy()
    py = far.groupby(["estimator", "year"]).over_null.median().unstack()
    L += ["## Per-year far rsd ÷ N0 (median over cells)", "", md_table(_ord(py).round(2)), ""]
    pw = pd.read_csv(OUT / "perdate_weights.csv")
    bl = pw[pw.est.str.startswith("BL")]
    wcols = [c for c in bl.columns if c.startswith("w_")]
    L += ["## Blend weights (median over date × surface × fold)", "",
          md_table(bl.groupby(["est", "sfc_type"])[wcols].median().round(3)), ""]
    cw = pw[pw.est.str.startswith("E2c")]
    if not cw.empty:
        L += ["Combination weights (SCO2, WCO2):", "",
              md_table(cw.groupby(["est", "sfc_type"])[["w_sco2", "w_wco2"]].median().round(3)), ""]
    fp_ = m[m.date.isin(FIRST_PASS_DATES) & (m.subset == "far_heldout")]
    if not fp_.empty:
        piv = fp_.pivot_table(index="estimator", columns=["date", "surface"], values="rsd_vs_bc")
        L += ["## First-pass dates: far rsd vs bc", "", md_table(_ord(piv).round(2)), ""]
    near = OUT / "neardiag_ocean.csv"
    if near.exists():
        nd = pd.read_csv(near).set_index("bin")
        cols = ["n_dates", "n_fp", "mean_anom_r05"] + [c for c in nd.columns if c.startswith(("medcorr_E0", "poolcorr_", "shift_E0", "shift_bc", "medcorr_qf0_E0_am-prior", "medcorr_sco2", "medcorr_bc"))]
        L += ["## Near-cloud diagnostic over ocean vs xco2_bc_anomaly_r05 (by cld_dist_km bin)", "",
              md_table(nd[cols].T.round(3)), ""]
    tc = OUT / "tccon_cases.csv"
    if False and tc.exists():   # TCCON comparison dropped from scope (2026-11-01)
        t = pd.read_csv(tc)
        keep = ["xco2_bc", "xco2_apriori", "N0_prior", primary, "E0_am", "E2w_am", "E2c_am",
                "E2wP_am", "E2cP_am_hgb", "E2cP_am_hgb_rc", "BL_E2c_am", "BLP_E0_am", "BLP_E2c_am",
                "BLP2P_am_hgb_rc", "B_IDP"]
        t = t[t.series.isin(keep)]
        L += [f"## TCCON ({t[['date','site']].drop_duplicates().shape[0]} station-days with "
              f"≥ 10 collocated footprints; ±60 min, 100 km / das2025_tk25 radii)", "",
              f"Primary PPDF estimator (lowest far/N0 on the 116 dates, chosen before TCCON): "
              f"**{primary}**. Case bias = mean over collocated footprints − TCCON; each table "
              "uses the cases where xco2_bc, the prior and the primary estimator all have ≥ 10 "
              "footprints. `direct` = TCCON window mean; `ak` = AK/prior-harmonized reference "
              "(the expected OCO-2 retrieval; meaningful for xco2_bc, less so for the prior).", ""]
        for ref, qf, surf, tab in tccon_tables(t, primary):
            L += [f"### ref={ref}, QF={qf}, surface={surf}", "", md_table(tab.round(3)), ""]
    slab = OUT / "slab_mc_floor.csv"
    if slab.exists():
        L += ["## Slab Monte Carlo identifiability floor", "",
              md_table(pd.read_csv(slab), index=False), ""]
    hh = headline(main)
    L += ["## Bars (main set, medians)", "",
          f"- Retrieval (far rsd vs bc ≤ {BAR_FAR_SD}): {[e for e in _ord(hh).index if hh.loc[e, 'far_vs_bc'] <= BAR_FAR_SD]}",
          f"- Beats the prior-only null (far/N0 < 1): {[e for e in _ord(hh).index if hh.loc[e, 'far/N0'] < 1 and e != 'N0_prior'] or 'none'}",
          f"- Diagnostic (near corr ≥ {BAR_NEAR_CORR}, near rsd ≤ {BAR_NEAR_SD}): "
          f"{[e for e in _ord(hh).index if hh.loc[e, 'near_corr'] >= BAR_NEAR_CORR and hh.loc[e, 'near_vs_bc'] <= BAR_NEAR_SD] or 'none'}",
          f"- Primary PPDF estimator: {primary} (far/N0 {hh.loc[primary, 'far/N0']:.3f})", ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print(f"-> {OUT / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["perdate", "pooled", "opblend", "tccon", "neardiag", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"perdate": stage_perdate, "pooled": stage_pooled, "opblend": stage_opblend, "tccon": stage_tccon,
     "neardiag": stage_neardiag, "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
