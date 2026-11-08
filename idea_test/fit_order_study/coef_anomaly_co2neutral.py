"""Can the revised-fit coefficients replace the retrieval-state departure dXCO2 = xco2_raw − xco2_apriori as the
anomaly predictor, and is their skill CO2-neutral? (2026-10-30, after a meeting comment that a dXCO2 predictor
pulls the corrected XCO2 toward the prior.)

Same orbit, targets and held-out ridge as coef_anomaly_corr_pipeline.py (29252a_GL, pipeline anomaly reference
windows, 10-block held-out R²). Coefficient anomalies use the pipeline definition. Predictor sets:
  dXCO2            : xco2_raw − xco2_apriori, stored and as pipeline anomaly
  agreed full      : all revised-fit coefficients
  CO2-neutral      : agreed full without the CO2-band k1 (sco2 k1, wco2 k1, sco2/o2a ratio): O2A k1, floor f,
                     continuum a1 (a2), H2O k1_w, SCO2 k2, k2_diag, rms per band
  CO2-band k1 only : sco2 k1, wco2 k1
  dXCO2 + CO2-neutral, dXCO2 + agreed full
A real CO2 enhancement moves the CO2-band k1 (SCO2 ≈ 1.6× the column change), so skill that survives in the
CO2-neutral set is path/scene skill; skill that needs the CO2-band k1 is partly the CO2 signal itself.
Usage: python coef_anomaly_co2neutral.py   → results/coef_anomaly_co2neutral.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from coef_anomaly_corr import load, ridge_heldout  # noqa: E402
from cut_sweep_real import DEFAULTS  # noqa: E402
from spectral.anomaly import compute_xco2_anomaly  # noqa: E402
from constants import anomaly_args  # noqa: E402

OUT = HERE / "results"
RADII = {"r05 (ML ocean target)": 5.0, "r10 (default)": 10.0}
TCOL = {5.0: "xco2_bc_anomaly_r05", 10.0: "xco2_bc_anomaly"}
CO2_K1 = ("agreed sco2 k1", "agreed wco2 k1", "agreed sco2/o2a k1")


def main():
    df = load()
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        for c in ("xco2_bc_anomaly_r05", "xco2_bc", "xco2_raw", "xco2_apriori"):
            df[c] = f[c][...].astype(float)
    df["dxco2"] = df.xco2_raw - df.xco2_apriori
    agreed = [c for c in df.columns if c.startswith("agreed ")]
    neutral = [c for c in agreed if c not in CO2_K1]
    sets = {"dXCO2 stored": (["dxco2"], "stored"), "dXCO2 anomaly": (["dxco2"], "anom"),
            "agreed full": (agreed, "anom"), "CO2-neutral": (neutral, "anom"),
            "CO2-band k1 only": (list(CO2_K1[:2]), "anom"),
            "dXCO2 + CO2-neutral": (["dxco2"] + neutral, "anom"), "dXCO2 + agreed full": (["dxco2"] + agreed, "anom")}
    subsets = {"QF0 cld<3": (df.xco2_qf == 0) & (df.cld_dist_km < 3), "QF0+1 cld<3": np.isfinite(df.xco2_qf) & (df.cld_dist_km < 3),
               "QF0 cld<10": (df.xco2_qf == 0) & (df.cld_dist_km < 10), "QF0 all": df.xco2_qf == 0}
    lines = ["# dXCO2 vs revised-fit coefficients as anomaly predictors (29252a_GL, held-out ridge R²)", "",
             "Coefficient and dXCO2 anomalies use the target's own pipeline reference window. CO2-neutral = agreed set without "
             f"{', '.join(CO2_K1)} ({len(neutral)} predictors: {', '.join(c.replace('agreed ', '') for c in neutral)}).", ""]
    for rname, r in RADII.items():
        extra = {c: df[c].values for c in agreed + ["dxco2"]}
        _, means, _ = compute_xco2_anomaly(df.lat.values, df.cld_dist_km.values, df.xco2_bc.values, extra_vars=extra,
                                           **anomaly_args(r))
        y = df[TCOL[r]].values
        src = {"stored": {c: df[c].values for c in extra}, "anom": {c: df[c].values - means[c] for c in extra}}
        lines += [f"## target {TCOL[r]} — {rname}", "", "| subset | n | " + " | ".join(sets) + " |",
                  "|---|---|" + "---|" * len(sets)]
        for sname, m in subsets.items():
            m = m.values & np.isfinite(y)
            r2 = []
            for cs, form in sets.values():
                X = np.column_stack([src[form][c] for c in cs])
                mm = m & np.all(np.isfinite(X), axis=1)
                r2.append(ridge_heldout(X[mm], y[mm]) if mm.sum() > 40 else np.nan)
            lines.append(f"| {sname} | {m.sum()} | " + " | ".join(f"{v:.2f}" for v in r2) + " |")
        # single-predictor correlations of the CO2-neutral members, QF0 cld<10
        m = subsets["QF0 cld<10"].values & np.isfinite(y)
        cors = sorted(((np.corrcoef(src["anom"][c][m & np.isfinite(src["anom"][c])],
                                    y[m & np.isfinite(src["anom"][c])])[0, 1], c) for c in neutral + ["dxco2"]),
                      key=lambda t: -abs(t[0]))
        lines += ["", "QF0 cld<10, single-predictor r (anomaly form): " + ", ".join(f"{c.replace('agreed ', '')} {v:+.2f}" for v, c in cors), ""]
    (OUT / "coef_anomaly_co2neutral.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
