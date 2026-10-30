"""Kozienice plume re-test with the revised pseudo-XCO2 design (after the dp finding).

Design: prior air column (no O2A divisor); XCO2 anomaly from the CO2-band coefficient anomaly divided by its
boundary-layer kernel (SCO2 1.60, WCO2 1.01); O2A coefficient anomaly only as a path flag.  Anomalies are
against the running median of the same footprint over ±W soundings (self excluded) among clear neighbours
(MODIS cloud distance > CLR_KM; non-retrieved soundings have no MODIS distance here and never serve as
baseline).  Known answer: the L2 bias-corrected XCO2 anomaly computed the same way among QF0 neighbours.
Inputs: results/kozienice_38200a_ND_per_sounding.parquet.  Outputs results/kozienice_plume_retest.{md,png}.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from o2a_anomaly_estimator import running_anomaly, rsd  # noqa: E402

OUT = HERE / "results"
W = 25
CLR_KM = 5.0
K_BL = {"sco2": 1.60, "wco2": 1.01}
FLAG_PCT = 1.0


def main():
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    order = np.argsort(df.along_km.values, kind="stable")
    pri = df.xco2_prior_cpr.values
    clear = (df.cld_dist_km.values > CLR_KM) & np.isfinite(df.k1_o2a.values)
    qf0 = (df.qf == 0).values
    an = {b: running_anomaly(df, f"k1_{b}", clear, order) for b in ("o2a", "wco2", "sco2")}
    an["f"] = running_anomaly(df, "f_o2a", clear, order)
    an["k1w_sco2"] = running_anomaly(df, "k1w_sco2", clear, order)
    est = {"ΔX SCO2 / 1.60": pri * an["sco2"] / K_BL["sco2"], "ΔX WCO2 / 1.01": pri * an["wco2"] / K_BL["wco2"],
           "ΔX SCO2 raw kernel": pri * an["sco2"], "ΔX E1-style (SCO2 − O2A)": pri * (an["sco2"] - an["o2a"])}
    ref = {"L2 bc anomaly": running_anomaly(df, "xco2_bc", qf0, order),
           "L2 raw anomaly": running_anomaly(df, "xco2_raw", qf0, order)}
    flag = np.abs(an["o2a"]) > FLAG_PCT / 100
    bg = qf0 & (df.dist_km.values >= 40)
    bgm = np.nanmedian(df.xco2_bc.values[bg])
    plume = ((df.xco2_bc.values - bgm > 2) & (df.dist_km.values < 30))
    pl0 = plume & qf0

    lines = ["# Kozienice plume re-test with the revised design", "",
             f"Anomalies vs running median of the same footprint, ±{W} soundings, clear neighbours = MODIS cloud distance > {CLR_KM:.0f} km "
             f"(n clear baseline soundings {clear.sum()}); L2 anomalies vs QF0 neighbours. Flag: |Δk1_O2A| > {FLAG_PCT:.0f} %.", "",
             "## Background (QF0, ≥ 40 km): scatter of each anomaly against the L2 bc anomaly", "",
             "| quantity | n | rSD (ppm) | rSD of (quantity − L2 bc anomaly) | corr with L2 bc anomaly |", "|---|---|---|---|---|"]
    rb = ref["L2 bc anomaly"]
    for name, v in list(est.items()) + list(ref.items()):
        m = bg & np.isfinite(v) & np.isfinite(rb)
        c = np.corrcoef(v[m], rb[m])[0, 1]
        lines.append(f"| {name} | {m.sum()} | {rsd(v[m]):.2f} | {rsd((v - rb)[m]):.2f} | {c:+.2f} |")
    m = bg & np.isfinite(an["o2a"])
    lines += ["", f"Background Δk1_O2A rSD {100*rsd(an['o2a'][m]):.2f} %; flagged: background QF0 {100*flag[bg].mean():.0f} %, "
              f"all QF0 {100*flag[qf0].mean():.0f} %, QF1 {100*flag[(df.qf==1).values].mean():.0f} %, "
              f"non-retrieved {100*flag[~np.isfinite(df.qf.values)].mean():.0f} %.", ""]

    lines += ["## Plume soundings (L2 bc > background + 2 ppm within 30 km)", "",
              "| quantity | all plume (n 14) | QF0 plume (n 4) | unflagged plume (n) |", "|---|---|---|---|"]
    unf = plume & ~flag
    for name, v in list(est.items()) + list(ref.items()):
        lines.append(f"| {name} | {np.nanmedian(v[plume]):+.1f} | {np.nanmedian(v[pl0]):+.1f} | {np.nanmedian(v[unf]):+.1f} ({np.isfinite(v[unf]).sum()}) |")
    for name, v, sc in (("Δk1_SCO2 (%)", an["sco2"], 100), ("Δk1_WCO2 (%)", an["wco2"], 100), ("Δk1_O2A (%)", an["o2a"], 100),
                        ("Δf_O2A (×100)", an["f"], 100), ("Δk1_w SCO2 (%)", an["k1w_sco2"], 100)):
        lines.append(f"| {name} | {sc*np.nanmedian(v[plume]):+.2f} | {sc*np.nanmedian(v[pl0]):+.2f} | {sc*np.nanmedian(v[unf]):+.2f} ({np.isfinite(v[unf]).sum()}) |")
    lines.append(f"| flagged | {flag[plume].sum()} of {plume.sum()} | {flag[pl0].sum()} of {pl0.sum()} | — |")

    # along-track 5-km bins near the plant
    edges = np.arange(-20, 21, 5)
    lines += ["", "## Along-track 5-km bins (medians; unflagged soundings only for the estimators)", "",
              "| bin (km) | n | n unflagged | L2 bc anom | L2 raw anom | ΔX SCO2/1.60 | ΔX WCO2/1.01 | Δk1_O2A % | flagged |", "|---|---|---|---|---|---|---|---|---|"]
    ak = df.along_km.values
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ak >= lo) & (ak < hi) & (df.dist_km.values < 40)
        mu = m & ~flag
        lines.append(f"| {lo:+d}..{hi:+d} | {m.sum()} | {mu.sum()} | {np.nanmedian(rb[m]):+.1f} | {np.nanmedian(ref['L2 raw anomaly'][m]):+.1f} | "
                     f"{np.nanmedian(est['ΔX SCO2 / 1.60'][mu]):+.1f} | {np.nanmedian(est['ΔX WCO2 / 1.01'][mu]):+.1f} | "
                     f"{100*np.nanmedian(an['o2a'][m]):+.2f} | {flag[m].sum()} |")
    (OUT / "kozienice_plume_retest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    near = df.dist_km.values < 60
    centers = np.arange(-57.5, 60, 5)
    def binmed(v, m):
        return np.array([np.nanmedian(v[m & (ak >= c - 2.5) & (ak < c + 2.5)]) if (m & (ak >= c - 2.5) & (ak < c + 2.5)).sum() >= 3 else np.nan for c in centers])
    ax = axes[0]
    ax.plot(centers, binmed(rb, near), "k-o", ms=3, label="L2 bc anomaly (QF0 baseline)")
    ax.plot(centers, binmed(ref["L2 raw anomaly"], near), "k--", lw=0.8, label="L2 raw anomaly")
    ax.plot(centers, binmed(est["ΔX SCO2 / 1.60"], near & ~flag), "C3-s", ms=3, label="ΔX SCO2 / 1.60, unflagged")
    ax.plot(centers, binmed(est["ΔX WCO2 / 1.01"], near & ~flag), "C0-^", ms=3, label="ΔX WCO2 / 1.01, unflagged")
    ax.plot(centers, binmed(est["ΔX E1-style (SCO2 − O2A)"], near & ~flag), "C2:", lw=1, label="E1-style (SCO2 − O2A), unflagged")
    ax.axhline(0, color="0.7", lw=0.5); ax.set_ylabel("XCO2 anomaly (ppm)"); ax.legend(fontsize=8); ax.set_title("Kozienice 2021-09-06, orbit 38200a_ND, 5-km along-track medians")
    ax = axes[1]
    ax.scatter(ak[near], 100 * an["o2a"][near], s=6, c=np.where(flag[near], "C3", "0.5"), label="Δk1_O2A (red = flagged)")
    ax.scatter(ak[near & plume], 100 * an["o2a"][near & plume], s=30, facecolors="none", edgecolors="k", label="L2 plume soundings")
    ax.axhline(FLAG_PCT, color="C3", lw=0.5); ax.axhline(-FLAG_PCT, color="C3", lw=0.5)
    ax.set(xlabel="along-track distance from closest approach (km)", ylabel="Δk1_O2A (%)"); ax.legend(fontsize=8)
    fig.savefig(OUT / "kozienice_plume_retest.png", dpi=150)


if __name__ == "__main__":
    main()
