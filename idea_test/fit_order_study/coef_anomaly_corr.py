"""Do the agreed-model coefficients correlate with the L2 XCO2 anomaly better than the production coefficients?

Orbit 29252a_GL (ocean; production single-τ fit and agreed fit both available).  Target: the production
`xco2_bc_anomaly` (L2 bc minus nearby far-clear footprints) and `xco2_raw_anomaly`.  Each coefficient is
used (a) as stored and (b) as a local anomaly against the running median of the same footprint among
far-clear QF0 neighbours (±W, self excluded), identically for both coefficient sets.  Metrics: Pearson and
Spearman correlation per coefficient in near-cloud subsets, and held-out R² of a ridge model on each
coefficient set (10 along-track blocks, leave-one-block-out).  Validation only: nothing here enters a fit.
Outputs results/coef_anomaly_corr.{md,csv}.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_model_test import DEFAULTS  # noqa: E402
from o2a_anomaly_estimator import running_anomaly  # noqa: E402

OUT = HERE / "results"
W = 25
PROD = {f"{b}_{k}_fitting_nosg": f"prod {b} {k}" for b in ("o2a", "wco2", "sco2") for k in ("k1", "k2", "intercept")}


def load():
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        fpk = "fp_number" if "fp_number" in f else "fp_id"
        cols = ["xco2_bc_anomaly", "xco2_raw_anomaly", "cld_dist_km", "xco2_qf", "lat", fpk] + list(PROD)
        df = pd.DataFrame({k: f[k][...].astype(float) for k in cols}).rename(columns={fpk: "fp", **PROD})
    ag = pd.read_parquet(OUT / "final_model_per_sounding_29252a_GL.parquet")
    for b in ("o2a", "wco2", "sco2"):
        s = ag[ag.band == b].set_index("j")
        for k in ("k1", "k2", "k1_w", "a1", "a2", "f", "c", "rms", "k2_diag"):
            v = s[k].reindex(df.index).values
            if np.isfinite(v).sum() > 100 and np.nanstd(v) > 0:
                df[f"agreed {b} {k}"] = v
    df["agreed sco2/o2a k1"] = df["agreed sco2 k1"] / df["agreed o2a k1"]
    df["prod sco2/o2a k1"] = df["prod sco2 k1"] / df["prod o2a k1"]
    return df


def ridge_heldout(X, y, nblock=10, lam=1e-2):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    Xs = (X - mu) / sd
    blocks = np.arange(len(y)) * nblock // len(y)
    pred = np.full(len(y), np.nan)
    for b in range(nblock):
        tr, te = blocks != b, blocks == b
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), Xs[te]]) @ beta
    return 1 - np.var(y - pred) / np.var(y)


def main():
    df = load()
    far = ((df.xco2_qf == 0) & (df.cld_dist_km > 20)).values
    order = np.arange(len(df))
    coefs = [c for c in df.columns if c.startswith(("prod ", "agreed "))]
    anom = {c: running_anomaly(df, c, far & np.isfinite(df[c].values), order) for c in coefs}
    subsets = {"QF0 cld<1": (df.xco2_qf == 0) & (df.cld_dist_km < 1), "QF0 cld<3": (df.xco2_qf == 0) & (df.cld_dist_km < 3),
               "QF0 cld<10": (df.xco2_qf == 0) & (df.cld_dist_km < 10), "QF0+1 cld<3": np.isfinite(df.xco2_qf) & (df.cld_dist_km < 3),
               "QF0 all": df.xco2_qf == 0}
    rows = []
    for tgt in ("xco2_bc_anomaly", "xco2_raw_anomaly"):
        y_all = df[tgt].values
        for sname, m in subsets.items():
            m = m.values & np.isfinite(y_all)
            for c in coefs:
                for form, v in (("stored", df[c].values), ("local anomaly", anom[c])):
                    mm = m & np.isfinite(v)
                    if mm.sum() < 30:
                        continue
                    rows.append(dict(target=tgt, subset=sname, coef=c, form=form, n=int(mm.sum()),
                                     pearson=np.corrcoef(v[mm], y_all[mm])[0, 1], spearman=spearmanr(v[mm], y_all[mm])[0]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "coef_anomaly_corr.csv", index=False)

    lines = ["# Coefficients vs L2 XCO2 anomaly: production single-τ fit vs agreed model (29252a_GL)", "",
             f"Local anomaly = coefficient − running median of the same footprint among far-clear QF0 neighbours (±{W}).", ""]
    for tgt in ("xco2_bc_anomaly", "xco2_raw_anomaly"):
        for sname in ("QF0 cld<1", "QF0 cld<3", "QF0+1 cld<3", "QF0 all"):
            g = res[(res.target == tgt) & (res.subset == sname)]
            lines += [f"## {tgt}, {sname} (n {g.n.max()})", "", "| coefficient | stored r | stored ρ | anomaly r | anomaly ρ |", "|---|---|---|---|---|"]
            for c in coefs:
                s = g[(g.coef == c) & (g.form == "stored")]
                a = g[(g.coef == c) & (g.form == "local anomaly")]
                if len(s) == 0:
                    continue
                av = (f"{a.pearson.iloc[0]:+.2f} | {a.spearman.iloc[0]:+.2f}" if len(a) else "— | —")
                lines.append(f"| {c} | {s.pearson.iloc[0]:+.2f} | {s.spearman.iloc[0]:+.2f} | {av} |")
            lines.append("")
    # held-out multi-coefficient models
    lines += ["## Held-out ridge R² on the L2 bc anomaly (10 along-track blocks)", "",
              "| subset | n | production set (k1,k2,intercept ×3 bands) | agreed set (k1,k1_w,k2,a1,a2,f,c,rms per band) | agreed k1 ×3 only | production k1 ×3 only | both sets |", "|---|---|---|---|---|---|---|"]
    sets = {"production set": [c for c in coefs if c.startswith("prod ") and "/" not in c],
            "agreed set": [c for c in coefs if c.startswith("agreed ") and "/" not in c],
            "agreed k1 ×3 only": [f"agreed {b} k1" for b in ("o2a", "wco2", "sco2")],
            "production k1 ×3 only": [f"prod {b} k1" for b in ("o2a", "wco2", "sco2")]}
    sets["both sets"] = sets["production set"] + sets["agreed set"]
    for sname in ("QF0 cld<3", "QF0 cld<10", "QF0+1 cld<3", "QF0 all"):
        m = subsets[sname].values & np.isfinite(df.xco2_bc_anomaly.values)
        row = []
        for form in ("stored", "local anomaly"):
            vals = []
            for k, cs in sets.items():
                X = np.column_stack([df[c].values if form == "stored" else anom[c] for c in cs])
                mm = m & np.all(np.isfinite(X), axis=1)
                vals.append(ridge_heldout(X[mm], df.xco2_bc_anomaly.values[mm]))
            row.append((form, mm.sum(), vals))
        for form, n, vals in row:
            lines.append(f"| {sname}, {form} | {n} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    (OUT / "coef_anomaly_corr.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
