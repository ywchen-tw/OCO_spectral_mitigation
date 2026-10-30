"""Quick test: pseudo-XCO2 with the O2A coefficient used as a local path anomaly instead of an absolute divisor.

E0     = prior · k1_S
E1     = prior · k1_S / k1_O2A                      (absolute divisor; carries SIF and τ-scale biases)
EA(g)  = prior · k1_S / (1 + g·Δk1_O2A)             Δk1_O2A = k1_O2A − running median of k1_O2A along track
                                                    (same footprint, ±W soundings, self excluded)
EB(g)  = prior · k1_S · (1 − g·Δk1_O2A) / ... same to first order; kept EA only.
The running median plays the role of the "nearby footprints" baseline (anomaly, not bias); it removes SIF,
ABSCO-scale and solar-model offsets that are smooth along track.  g = 1 means the path change seen by O2A is
removed one-to-one from SCO2; g = 1.6/1.8 is the ratio of the SCO2 boundary-layer kernel to the O2A
surface-layer kernel (near-surface path change).  Metrics: far-clear robust SD vs L2 bc, near-cloud bins
(ocean), plume contrast (Kozienice).  Uses the per-sounding tables of the earlier tests only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import o2a_dp_test as t  # noqa: E402

W = 25


def rsd(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def running_anomaly(df, col, ok, order):
    """k − running median over ±W soundings of the same footprint among `ok` soundings (self excluded)."""
    out = np.full(len(df), np.nan)
    for fp in np.unique(df.fp):
        idx = order[(df.fp.values[order] == fp)]
        good = idx[ok[idx]]
        vals = df[col].values
        for k in idx:
            pos = np.searchsorted(good, k)
            nb = np.r_[good[max(0, pos - W):pos], good[pos + 1:pos + 1 + W]]
            nb = nb[nb != k]
            if len(nb) >= 8:
                out[k] = vals[k] - np.median(vals[nb])
    return out


def estimators(df, pri, ok_base, order):
    d = running_anomaly(df, "k1_o2a", ok_base, order)
    est = {"E0": pri * df.k1_sco2.values, "E1": pri * df.k1_sco2.values / df.k1_o2a.values}
    for g in (1.0, 1.6 / 1.8, 0.5):
        est[f"EA g={g:.2f}"] = pri * df.k1_sco2.values / (1 + g * d)
    est["E0 WCO2"] = pri * df.k1_wco2.values
    est["EA WCO2 g=1/1.8"] = pri * df.k1_wco2.values / (1 + d / 1.8)
    return est, d


def main():
    oc, ld = t.ocean_table(), t.land_table()
    lines = ["# O2A as local path anomaly: estimator comparison", ""]
    # ---- ocean: far-clear scatter and near-cloud contrast (vs L2 bc anomaly in the same bins)
    import h5py
    from final_model_test import DEFAULTS
    with h5py.File(DEFAULTS["fit_details"]) as f:
        oc["xco2_bc"] = f["xco2_bc"][...] if "xco2_bc" in f else f["xco2"][...]
    order = np.arange(len(oc))
    okb = (oc.qf == 0).values & np.isfinite(oc.k1_o2a.values)      # baseline from QF0 neighbours (screen proxy)
    est, d = estimators(oc, oc.xco2_apriori.values, okb, order)
    far = ((oc.qf == 0) & oc.far).values
    lines += ["## 29252a_GL ocean", "", f"Δk1_O2A (running median, ±{W}, QF0 neighbours): far-clear rSD {100*rsd(d[far]):.2f} %",
              "", "| estimator | far-clear bias vs bc | far-clear rSD | 0–1 km − far | 1–3 km − far | 3–10 km − far |", "|---|---|---|---|---|---|"]
    bins = [(0, 1), (1, 3), (3, 10)]
    q0 = (oc.qf == 0).values
    ref = {}
    for name, e in list(est.items()) + [("L2 bc", oc.xco2_bc.values), ("L2 raw", oc.xco2_raw.values)]:
        dv = e - oc.xco2_apriori.values
        row = [f"{np.nanmedian((e - oc.xco2_bc.values)[far]):+.1f}", f"{rsd((e - oc.xco2_bc.values)[far]):.2f}"]
        for lo, hi in bins:
            m = q0 & (oc.cld_dist_km.values >= lo) & (oc.cld_dist_km.values < hi)
            row.append(f"{np.nanmedian(dv[m]) - np.nanmedian(dv[far]):+.2f} (n {m.sum()})")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    # ---- land: background scatter and plume contrast
    order = np.argsort(ld.along_km.values, kind="stable")
    okb = np.isfinite(ld.k1_o2a.values) & (ld.qf == 0).values
    pri = ld.xco2_prior_cpr.values
    est, d = estimators(ld, pri, okb, order)
    bg = ((ld.qf == 0) & ld.far).values
    bgm = np.nanmedian(ld.xco2_bc.values[bg])
    plume = ((ld.xco2_bc - bgm > 2) & (ld.dist_km < 30)).values
    pl0 = plume & (ld.qf == 0).values
    lines += ["", "## 38200a_ND Kozienice land", "", f"Δk1_O2A background rSD {100*rsd(d[bg]):.2f} %; plume median Δk1_O2A {100*np.nanmedian(d[plume]):+.2f} % (QF0 {100*np.nanmedian(d[pl0]):+.2f} %)",
              "", "| estimator | bg bias vs bc | bg rSD | plume − bg (n 14) | QF0 plume − bg (n 4) |", "|---|---|---|---|---|"]
    for name, e in list(est.items()) + [("L2 bc", ld.xco2_bc.values), ("L2 raw", ld.xco2_raw.values)]:
        b = np.nanmedian(e[bg])
        lines.append(f"| {name} | {np.nanmedian((e - ld.xco2_bc.values)[bg]):+.1f} | {rsd((e - ld.xco2_bc.values)[bg]):.2f} | "
                     f"{np.nanmedian(e[plume]) - b:+.1f} | {np.nanmedian(e[pl0]) - b:+.1f} |")
    (HERE / "results" / "o2a_anomaly_estimator.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
