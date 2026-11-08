"""T3: global clear-sky baseline for absolute quantities, then E_z of the residual.

Baseline v_hat(regressors) is fitted with XGBoost on far footprints
(cloud distance > 20 km, including the 50-km cap) of half the evaluation
dates and applied to the other half (date-level 2-fold: each half is held
out once, residuals pooled).  Regressors are not derived from the footprint's
own radiance: geometry (SZA, VZA, scattering angle, glint angle, airmass,
observation mode, footprint index), latitude, longitude, month, surface
elevation and its sub-footprint SD, prior surface pressure, T700, prior wind
speed (ocean), MCD12C1 IGBP class (land).  The contrast model adds the
retrieved Lite O2A albedo (which sees the 3D-perturbed radiance).

E_z of the residual (near 0-5 km vs far 20-50 km, far-SD units) on held-out
footprints: all footprints with a cloud distance, the Fig. 5 reference
population, QF split, and (land, reference population) by illumination class.

Outputs: out/t3_residual_effect.csv, out/t3_illum.csv, out/t3_fit_quality.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import common
from common import effect_z, illum_class

TARGETS = [("k1_o2a", "o2a_k1_am", "dk1_o2a"),
           ("k2k1sq_sco2", "sco2_k2k1sq", "dk2k1sq_sco2"),
           ("k1ratio_sw", "sco2_wco2_k1", "dk1ratio_sw")]
BASE_FEATS = ["sza", "vza", "cos_scat", "glint_angle", "airmass", "operation_mode", "fp",
              "lat", "lon", "month", "alt", "alt_std", "psfc_prior", "t700"]
SURF_FEATS = {"ocean": ["ws_apriori"], "land": ["igbp_class"]}
VARIANTS = {"geom": [], "geom+alb": ["alb_o2a"]}
N_TRAIN_MAX = 3_000_000
# fixed tree count: early stopping on ~7 held-out training dates was too noisy
# (first run: land O2A fold 1 stopped at 6 trees, R2 0.46; log in out/t3_earlystop_run.log)
PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.9, min_child_weight=200, tree_method="hist",
              n_jobs=10)
SEED = 20261105


def load() -> pd.DataFrame:
    cols = (BASE_FEATS + ["ws_apriori", "igbp_class", "alb_o2a", "date", "cld_dist_km",
                          "sfc_type", "xco2_qf", "zexp_o2a", "xco2_bc_anomaly", "mixed"]
            + [c for _, a, d in TARGETS for c in (a, d)])
    df = common.load_table(sorted(set(cols)))
    df["igbp_class"] = df["igbp_class"].astype("float32")
    df["ref_ok"] = (np.isfinite(df.zexp_o2a) & np.isfinite(df.xco2_bc_anomaly) & ~df.mixed)
    df["illum"] = illum_class(df.zexp_o2a.to_numpy())
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                target: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    tr = train[np.isfinite(train[target])]
    if len(tr) > N_TRAIN_MAX:
        tr = tr.iloc[rng.choice(len(tr), N_TRAIN_MAX, replace=False)]
    model = xgb.XGBRegressor(**PARAMS, random_state=SEED)
    model.fit(tr[feats], tr[target], verbose=False)
    pred = model.predict(test[feats])
    imp = model.get_booster().get_score(importance_type="gain")
    tot = sum(imp.values())
    return pred, {"n_trees": PARAMS["n_estimators"], "n_train": len(tr),
                  "gain_share": {k: round(val / tot, 3) for k, val in
                                 sorted(imp.items(), key=lambda kv: -kv[1])[:5]}}


def main() -> None:
    df = load()
    rng = np.random.default_rng(SEED)
    dates = np.array(sorted(df["date"].unique()))
    half = rng.permutation(len(dates)) < len(dates) // 2
    fold_of_date = dict(zip(dates, half.astype(int)))
    df["fold"] = df["date"].map(fold_of_date).astype(int)
    print(f"{len(dates)} dates, fold sizes {np.bincount(half.astype(int))}")

    rows, illum_rows, fitq = [], [], []
    for sname, code in (("ocean", 0), ("land", 1)):
        s = df[df.sfc_type == code].copy()
        far_train = s.cld_dist_km > 20
        for key, target, dcol in TARGETS:
            for vname, extra in VARIANTS.items():
                feats = BASE_FEATS + SURF_FEATS[sname] + extra
                resid = np.full(len(s), np.nan, dtype=np.float64)
                for k in (0, 1):
                    tr = s[far_train & (s.fold != k)]
                    te_mask = (s.fold == k).to_numpy()
                    pred, info = fit_predict(tr, s[te_mask], feats, target, rng)
                    resid[te_mask] = s.loc[te_mask, target].to_numpy() - pred
                    te_far = te_mask & (s.cld_dist_km > 20).to_numpy() & np.isfinite(resid)
                    y = s[target].to_numpy()[te_far]
                    r2 = 1 - np.var(resid[te_far]) / np.var(y)
                    fitq.append(dict(surface=sname, quantity=key, variant=vname, test_fold=k,
                                     r2_far_heldout=r2, sd_far_raw=np.std(y),
                                     sd_far_resid=np.std(resid[te_far]), **info))
                    print(f"  {sname} {key} {vname} fold{k}: R2(far, held-out) {r2:.3f} "
                          f"n_train {info['n_train']} top {info['gain_share']}", flush=True)
                d = s.cld_dist_km.to_numpy()
                ref = s.ref_ok.to_numpy()
                qf = s.xco2_qf.to_numpy()
                for qname, qm in (("all", np.ones(len(s), bool)), ("0", qf == 0), ("1", qf == 1)):
                    for pname, pm in (("all", np.ones(len(s), bool)), ("ref", ref)):
                        m = qm & pm
                        rec = dict(surface=sname, quantity=key, variant=vname, qf=qname, pop=pname)
                        for lbl, val in (("resid", resid),
                                         ("abs", s[target].to_numpy(dtype=float))):
                            r = effect_z(d[m], val[m])
                            rec.update({f"{lbl}_E_z": r["E_z"], f"{lbl}_ci": r["ci95"],
                                        f"{lbl}_raw": r["near_mean"] - r["far_mean"],
                                        f"{lbl}_far_sd": r["far_sd"], "n_near": r["n_near"]})
                        if pname == "ref":
                            r = effect_z(d[m], s[dcol].to_numpy(dtype=float)[m])
                            rec.update(delta_E_z=r["E_z"], delta_raw=r["near_mean"] - r["far_mean"],
                                       delta_far_sd=r["far_sd"])
                        rows.append(rec)
                if sname == "land":
                    il = s.illum.to_numpy()
                    for c in ("shadowed", "neutral", "brightened"):
                        m = ref & (il == c)
                        far_all = ref & (d >= 20) & (d < 50)
                        for lbl, val in (("resid", resid), ("abs", s[target].to_numpy(dtype=float)),
                                         ("delta", s[dcol].to_numpy(dtype=float))):
                            same = effect_z(d[m], val[m])
                            mm = (m & (d < 5)) | far_all
                            allfar = effect_z(d[mm], val[mm])
                            illum_rows.append(dict(quantity=key, variant=vname, illum=c, kind=lbl,
                                                   E_z_samefar=same["E_z"], E_z_allfar=allfar["E_z"],
                                                   n_near=same["n_near"]))
    res = pd.DataFrame(rows)
    res.to_csv(common.OUT / "t3_residual_effect.csv", index=False)
    il = pd.DataFrame(illum_rows).drop_duplicates(["quantity", "variant", "illum", "kind"])
    il.to_csv(common.OUT / "t3_illum.csv", index=False)
    pd.DataFrame(fitq).to_csv(common.OUT / "t3_fit_quality.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(res[["surface", "quantity", "variant", "qf", "pop", "abs_E_z", "resid_E_z",
                   "delta_E_z", "abs_far_sd", "resid_far_sd", "delta_far_sd", "n_near"]]
              .round(3).to_string(index=False))
        print(il.pivot_table(index=["quantity", "illum"], columns=["kind", "variant"],
                             values="E_z_samefar").round(2).to_string())
        print(il.pivot_table(index=["quantity", "illum"], columns=["kind", "variant"],
                             values="E_z_allfar").round(2).to_string())


if __name__ == "__main__":
    main()
