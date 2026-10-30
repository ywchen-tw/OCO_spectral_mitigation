"""Absolute pseudo-XCO2 on the Kozienice orbit with the revised design (no scaling to any L2 quantity).

E_S      = prior · k1_S                                   (column-type reading of the whole coefficient)
E_S,kern = prior · (k1_S,base + Δk1_S / 1.60)             (slow baseline read as column, local anomaly as BL)
E_W      = prior · k1_W
Baseline k1_S,base = running median of the same footprint among clear neighbours (as in the re-test).
Compared with L2 xco2_raw and xco2 (bc) as stored, by footprint, by along-track position and in the plume.
Outputs results/kozienice_absolute.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from o2a_anomaly_estimator import running_anomaly, rsd  # noqa: E402
from kozienice_plume_retest import W, CLR_KM, K_BL, FLAG_PCT  # noqa: E402

OUT = HERE / "results"


def main():
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    order = np.argsort(df.along_km.values, kind="stable")
    pri = df.xco2_prior_cpr.values
    clear = (df.cld_dist_km.values > CLR_KM) & np.isfinite(df.k1_o2a.values)
    qf0 = (df.qf == 0).values
    d_s = running_anomaly(df, "k1_sco2", clear, order)
    d_o = running_anomaly(df, "k1_o2a", clear, order)
    base_s = df.k1_sco2.values - d_s
    flag = np.abs(d_o) > FLAG_PCT / 100
    est = {"E_S = prior·k1_S": pri * df.k1_sco2.values,
           "E_S,kern = prior·(base + Δ/1.60)": pri * (base_s + d_s / K_BL["sco2"]),
           "E_W = prior·k1_W": pri * df.k1_wco2.values,
           "prior": pri}
    bc, raw = df.xco2_bc.values, df.xco2_raw.values
    bg = qf0 & (df.dist_km.values >= 40)
    bgm = np.nanmedian(bc[bg])
    plume = (bc - bgm > 2) & (df.dist_km.values < 30)
    pl0 = plume & qf0
    lines = ["# Absolute pseudo-XCO2, Kozienice orbit 38200a_ND (revised design, unscaled)", "",
             f"Background = QF0, ≥ 40 km (n {bg.sum()}). L2 values as stored. Medians; rSD = 1.4826·MAD.", "",
             "## Background level and scatter", "",
             "| quantity | median (ppm) | − L2 bc | − L2 raw | rSD | rSD vs bc | rSD vs raw | rSD after per-footprint medians |", "|---|---|---|---|---|---|---|---|"]
    fp = df.fp.values
    for name, e in list(est.items()) + [("L2 raw", raw), ("L2 bc", bc)]:
        v = e[bg]
        fpm = pd.Series(v).groupby(fp[bg]).transform("median").values
        lines.append(f"| {name} | {np.nanmedian(v):.2f} | {np.nanmedian((e-bc)[bg]):+.2f} | {np.nanmedian((e-raw)[bg]):+.2f} | {rsd(v):.2f} | "
                     f"{rsd((e-bc)[bg]):.2f} | {rsd((e-raw)[bg]):.2f} | {rsd(v - fpm):.2f} |")
    lines += ["", "## Per-footprint background medians (ppm)", "", "| fp | n | prior | L2 raw | L2 bc | E_S | E_W | k1_S | k1_W | k1_O2A |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f_ in range(8):
        m = bg & (fp == f_)
        lines.append(f"| {f_} | {m.sum()} | {np.nanmedian(pri[m]):.2f} | {np.nanmedian(raw[m]):.2f} | {np.nanmedian(bc[m]):.2f} | "
                     f"{np.nanmedian(est['E_S = prior·k1_S'][m]):.2f} | {np.nanmedian(est['E_W = prior·k1_W'][m]):.2f} | "
                     f"{np.nanmedian(df.k1_sco2.values[m]):.4f} | {np.nanmedian(df.k1_wco2.values[m]):.4f} | {np.nanmedian(df.k1_o2a.values[m]):.4f} |")
    lines += ["", "## Along-track (QF0, 10-km bins of along-track distance, all footprints)", "",
              "| bin (km) | n | prior | L2 raw | L2 bc | E_S | E_S,kern | E_W | E_S − bc |", "|---|---|---|---|---|---|---|---|---|"]
    ak = df.along_km.values
    for lo in range(-60, 70, 10):
        m = qf0 & (ak >= lo) & (ak < lo + 10)
        if m.sum() < 3:
            continue
        es, ek, ew = (est[k] for k in ("E_S = prior·k1_S", "E_S,kern = prior·(base + Δ/1.60)", "E_W = prior·k1_W"))
        lines.append(f"| {lo:+d}..{lo+10:+d} | {m.sum()} | {np.nanmedian(pri[m]):.1f} | {np.nanmedian(raw[m]):.1f} | {np.nanmedian(bc[m]):.1f} | "
                     f"{np.nanmedian(es[m]):.1f} | {np.nanmedian(ek[m]):.1f} | {np.nanmedian(ew[m]):.1f} | {np.nanmedian((es-bc)[m]):+.1f} |")
    lines += ["", "## Plume soundings, absolute (ppm)", "", "| quantity | all plume (n 14) | QF0 plume (n 4) | unflagged (n) |", "|---|---|---|---|"]
    unf = plume & ~flag
    for name, e in list(est.items()) + [("L2 raw", raw), ("L2 bc", bc)]:
        lines.append(f"| {name} | {np.nanmedian(e[plume]):.1f} | {np.nanmedian(e[pl0]):.1f} | {np.nanmedian(e[unf]):.1f} ({np.isfinite(e[unf]).sum()}) |")
    lines += ["", "## Scene dependence of E_S − L2 bc in the background (QF0, ≥ 40 km)", "", "| variable | corr | slope per unit |", "|---|---|---|"]
    err = est["E_S = prior·k1_S"] - bc
    am = 1 / np.cos(np.deg2rad(df.sza.values)) + 1 / np.cos(np.deg2rad(df.vza.values))
    for name, x in (("airmass", am), ("lat", df.lat.values), ("fs_rel (Lite, validation)", df.fs_rel.values),
                    ("k1_O2A", df.k1_o2a.values), ("f_O2A", df.f_o2a.values), ("k1w_SCO2", df.k1w_sco2.values),
                    ("a1_SCO2", df.a1_sco2.values), ("k2_SCO2", df.k2_sco2.values), ("h2o_col", df.h2o_col.values)):
        m = bg & np.isfinite(x) & np.isfinite(err)
        c = np.corrcoef(x[m], err[m])[0, 1]
        s = np.polyfit(x[m], err[m], 1)[0]
        lines.append(f"| {name} | {c:+.2f} | {s:+.3g} |")
    (OUT / "kozienice_absolute.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
