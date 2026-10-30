"""coef_anomaly_corr with the pipeline's own anomaly definitions (29252a_GL).

Target: pipeline `xco2_bc_anomaly` (10 km reference), `_r05` (the ML target over ocean) and `_r15` (the ML
target over land).  Coefficient anomalies: coefficient minus the mean over the SAME clear-sky reference
window as the target (±0.25° latitude, cld_dist > r km, all footprints, std gate on XCO2), computed with
src/spectral/anomaly.compute_xco2_anomaly(extra_vars=...), i.e. exactly the pipeline's dk1_* construction.
Outputs results/coef_anomaly_corr_pipeline.md.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from final_model_test import DEFAULTS  # noqa: E402
from coef_anomaly_corr import load, ridge_heldout, PROD  # noqa: E402
from spectral.anomaly import compute_xco2_anomaly  # noqa: E402
from constants import anomaly_args  # noqa: E402

OUT = HERE / "results"
RADII = {"r05 (ML ocean target)": 5.0, "r10 (default)": 10.0, "r15 (ML land target)": 15.0}
TCOL = {5.0: "xco2_bc_anomaly_r05", 10.0: "xco2_bc_anomaly", 15.0: "xco2_bc_anomaly_r15"}


def main():
    df = load()
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        for c in ("xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15", "xco2_bc"):
            df[c] = f[c][...].astype(float)
    coefs = [c for c in df.columns if c.startswith(("prod ", "agreed "))]
    sets = {"production set": [c for c in coefs if c.startswith("prod ") and "/" not in c],
            "agreed set": [c for c in coefs if c.startswith("agreed ") and "/" not in c],
            "production k1 ×3": [f"prod {b} k1" for b in ("o2a", "wco2", "sco2")],
            "agreed k1 ×3": [f"agreed {b} k1" for b in ("o2a", "wco2", "sco2")]}
    lines = ["# Coefficients vs L2 XCO2 anomaly with the pipeline reference definition (29252a_GL)", "",
             "Coefficient anomaly = coefficient − mean over the target's own clear-sky reference window "
             "(±0.25° lat, cld_dist > r, all footprints; pipeline `compute_xco2_anomaly(extra_vars)`).", ""]
    subsets = {"QF0 cld<1": (df.xco2_qf == 0) & (df.cld_dist_km < 1), "QF0 cld<3": (df.xco2_qf == 0) & (df.cld_dist_km < 3),
               "QF0+1 cld<3": np.isfinite(df.xco2_qf) & (df.cld_dist_km < 3), "QF0 cld<10": (df.xco2_qf == 0) & (df.cld_dist_km < 10),
               "QF0 all": df.xco2_qf == 0}
    for rname, r in RADII.items():
        extra = {c: df[c].values for c in coefs}
        anom_chk, means, _ = compute_xco2_anomaly(df.lat.values, df.cld_dist_km.values, df.xco2_bc.values,
                                                  extra_vars=extra, **anomaly_args(r))
        y = df[TCOL[r]].values
        agree = np.nanmax(np.abs(anom_chk - y))
        danom = {c: df[c].values - means[c] for c in coefs}
        lines += [f"## target {TCOL[r]} — {rname}; recomputed target matches stored to {agree:.2e} ppm; defined for {np.isfinite(y).sum()} soundings", "",
                  "| subset | n | form | production set R² | agreed set R² | production k1 ×3 | agreed k1 ×3 | best single coefficients (r) |", "|---|---|---|---|---|---|---|---|"]
        for sname, m in subsets.items():
            m = m.values & np.isfinite(y)
            for form, src in (("stored", {c: df[c].values for c in coefs}), ("pipeline anomaly", danom)):
                r2 = []
                for k, cs in sets.items():
                    X = np.column_stack([src[c] for c in cs])
                    mm = m & np.all(np.isfinite(X), axis=1)
                    r2.append(ridge_heldout(X[mm], y[mm]) if mm.sum() > 40 else np.nan)
                cors = []
                for c in coefs:
                    v = src[c]
                    mm = m & np.isfinite(v)
                    if mm.sum() > 30:
                        cors.append((np.corrcoef(v[mm], y[mm])[0, 1], c))
                cors.sort(key=lambda t: -abs(t[0]))
                best = ", ".join(f"{c} {cc:+.2f}" for cc, c in cors[:4])
                bestp = next((f"{c} {cc:+.2f}" for cc, c in cors if c.startswith("prod ")), "—")
                lines.append(f"| {sname} | {m.sum()} | {form} | {r2[0]:.2f} | {r2[1]:.2f} | {r2[2]:.2f} | {r2[3]:.2f} | {best}; best production: {bestp} |")
        lines.append("")
    (OUT / "coef_anomaly_corr_pipeline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
