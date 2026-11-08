"""Validation of sco2_floor_test.py: which SCO2 shape parameter carries the cloud and plume signals, and does the
SCO2 flattening behave like an additive radiance offset (f ∝ 1/albedo, f·albedo constant, as the O2A floor does)?

Inputs: results/sco2_floor_test_{29252a_GL,38200a_ND}.parquet (variants a–d per sounding), the per-sounding
Kozienice table, the GL fit-details file and the Kozienice Lite file (albedo_*, aod_total: validation only,
never fit inputs).
  1. 29252a_GL far-clear: latitude-local near-cloud contrast (0–1, 1–3 km) of k2(a), f(b), k2(c), f(c), k1(a), k1(b);
     r of k2(a), f(b) with albedo_sco2, 1/albedo_sco2, aod_total; relative robust SD of f and f·albedo.
  2. Kozienice land background: the same correlations, with the stored O2A floor f_o2a as the reference of a true
     additive offset; plume running-median anomalies of k1 per variant and of k2(a), f(b); per-plume-sounding table.
Usage: python sco2_floor_albedo.py   → results/sco2_floor_albedo.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS  # noqa: E402
from h2o_analysis import bin_stats, lat_local_contrast  # noqa: E402
from o2a_anomaly_estimator import rsd, running_anomaly  # noqa: E402

OUT = HERE / "results"
ROOT = HERE.parents[1]
LITE = {"38200a_ND": ROOT / "data/OCO2/2021/249/oco2_LtCO2_210906_B11210Ar_240916224317s.nc4"}
PARAMS = (("a", "k2"), ("b", "f"), ("c", "k2"), ("c", "f"), ("a", "k1"), ("b", "k1"), ("d", "k1"))


def columns(fits: pd.DataFrame, n: int):
    def col(v, c):
        d = fits[fits.variant == v].set_index("j")
        out = np.full(n, np.nan)
        out[d.index.values] = d[c].astype(float).values
        return out
    return col


def lite_fields(tag: str, sid: np.ndarray) -> dict:
    with h5py.File(LITE[tag]) as h:
        lsid = h["sounding_id"][...]
        fields = {k: h[f"Retrieval/{k}"][...] for k in ("albedo_sco2", "albedo_o2a", "aod_total")}
    pos = pd.Series(np.arange(len(lsid)), index=lsid).reindex(sid).values
    ok = np.isfinite(pos)
    out = {k: np.full(len(sid), np.nan) for k in fields}
    for k, v in fields.items():
        out[k][ok] = v[pos[ok].astype(int)]
    return out


def rel_rsd(v):
    q = v / np.median(v)
    return 1.4826 * np.median(np.abs(q - np.median(q)))


def albedo_table(col, sel, lite, extra=()):
    lines = ["| parameter | n | r(albedo_sco2) | r(1/albedo_sco2) | r(aod_total) | r(albedo_o2a) | rel. rSD | rel. rSD × albedo |",
             "|---|---|---|---|---|---|---|---|"]
    a_s, a_o, aod = lite["albedo_sco2"], lite["albedo_o2a"], lite["aod_total"]
    items = [(f"{c} ({v})", col(v, c), a_s) for v, c in (("a", "k2"), ("b", "f"), ("c", "k2"), ("c", "f"), ("a", "k1"), ("b", "k1"))]
    items += [(name, val, a_o) for name, val in extra]
    for name, v, a_own in items:
        m = sel & np.isfinite(v) & np.isfinite(a_s) & np.isfinite(a_o) & np.isfinite(aod)
        r = [np.corrcoef(v[m], q[m])[0, 1] for q in (a_s, 1 / a_s, aod, a_o)]
        pos = v[m] > 0
        rr = (rel_rsd(v[m][pos]), rel_rsd(v[m][pos] * a_own[m][pos])) if pos.sum() > 20 else (np.nan, np.nan)
        lines.append(f"| {name} | {m.sum()} | {r[0]:+.2f} | {r[1]:+.2f} | {r[2]:+.2f} | {r[3]:+.2f} | {rr[0]:.3f} | {rr[1]:.3f} |")
    return lines


def gl_part():
    fits = pd.read_parquet(OUT / "sco2_floor_test_29252a_GL.parquet")
    with h5py.File(DEFAULTS["fit_details"]) as f:
        meta = {k: f[k][...] for k in ("cld_dist_km", "xco2_qf", "lat", "fp_number")}
        lite = {"albedo_sco2": f["alb_sco2"][...].astype(float), "albedo_o2a": f["alb_o2a"][...].astype(float),
                "aod_total": f["aod_total"][...].astype(float)}
    n = len(meta["lat"])
    cld, qf = meta["cld_dist_km"].astype(float), meta["xco2_qf"]
    lat, fp = meta["lat"].astype(float), meta["fp_number"].astype(int)
    far = (cld > 20) & (qf == 0)
    col = columns(fits, n)
    lines = ["## 29252a_GL (ocean glint), far-clear n = %d" % far.sum(), "",
             "Near-cloud contrast in absolute units (value − latitude-local far reference, same footprint), QF0 "
             "soundings; SNR = contrast / far-clear robust SD of the contrast.", "",
             "| parameter | far rSD | 0–1 km (SNR) | 1–3 km (SNR) |", "|---|---|---|---|"]
    for v, c in PARAMS:
        con = lat_local_contrast(col(v, c), cld, qf, lat, fp, False)
        st = bin_stats(np.where(qf == 0, con, np.nan), cld, far)
        lines.append(f"| {c} ({v}) | {st['far_spread']:.5f} | {st['con_0-1']:+.5f} ({st['snr_0-1']:+.2f}) | "
                     f"{st['con_1-3']:+.5f} ({st['snr_1-3']:+.2f}) |")
    lines += ["", "Correlation with L2 fields (fit-details file) over far-clear soundings (validation only). rel. rSD = robust SD / median "
              "of the parameter and of parameter × own-band albedo (an additive radiance offset gives a constant product).", ""]
    lines += albedo_table(col, far, lite)
    return lines


def koz_part():
    fits = pd.read_parquet(OUT / "sco2_floor_test_38200a_ND.parquet")
    df = pd.read_parquet(OUT / "kozienice_38200a_ND_per_sounding.parquet")
    n = len(df)
    col = columns(fits, n)
    bg = ((df.dist_km >= 40) & (df.qf == 0)).values
    land = df.land.values.astype(bool)
    order = np.argsort(df.along_km.values, kind="stable")
    clear = (df.cld_dist_km.values > 5.0) & np.isfinite(df.k1_o2a.values)
    bgm = np.nanmedian(df.xco2_bc.values[bg])
    plume = (df.xco2_bc.values - bgm > 2) & (df.dist_km.values < 30)
    lite = lite_fields("38200a_ND", df.sounding_id.values)
    lines = ["## Kozienice 38200a_ND (land nadir), background = ≥ 40 km, QF0, land: n = %d" % (bg & land).sum(), "",
             "Same correlations; the stored O2A floor f_o2a is the reference behaviour of a true additive radiance offset "
             "(its product with albedo_o2a should be constant).", ""]
    lines += albedo_table(col, bg & land, lite, extra=[("f_o2a (stored O2A floor)", df.f_o2a.values.astype(float))])
    lines += ["", f"Plume (L2 bc > background + 2 ppm within 30 km; n = {plume.sum()}, QF0 {(plume & (df.qf == 0)).sum()}): "
              "running-median anomaly (same footprint, ±25 soundings, clear neighbours cld_dist_km > 5), median over plume "
              "soundings; bg rSD = robust SD of the anomaly over background soundings.", "",
              "| parameter | plume median anomaly | bg rSD | plume / bg rSD |", "|---|---|---|---|"]
    an = {}
    for v, c in PARAMS:
        tmp = df[["fp"]].copy()
        tmp["v"] = col(v, c)
        an[(v, c)] = running_anomaly(tmp, "v", clear & np.isfinite(tmp.v.values), order)
        a = an[(v, c)]
        lines.append(f"| {c} ({v}) | {np.nanmedian(a[plume]):+.5f} | {rsd(a[bg]):.5f} | {np.nanmedian(a[plume]) / rsd(a[bg]):+.2f} |")
    lines += ["", "Per plume sounding (along-track order):", "",
              "| QF | L2 bc − bg (ppm) | Δk1 (a) % | Δk1 (b) % | Δk1 (d) % | Δk2 (a) | Δf (b) |", "|---|---|---|---|---|---|---|"]
    for j in np.flatnonzero(plume)[np.argsort(df.along_km.values[plume])]:
        lines.append(f"| {int(df.qf[j])} | {df.xco2_bc[j] - bgm:+.1f} | {100 * an[('a', 'k1')][j]:+.2f} | "
                     f"{100 * an[('b', 'k1')][j]:+.2f} | {100 * an[('d', 'k1')][j]:+.2f} | {an[('a', 'k2')][j]:+.4f} | "
                     f"{an[('b', 'f')][j]:+.5f} |")
    return lines


def main():
    lines = ["# SCO2 floor vs k2: cloud / plume response and albedo dependence (2026-10-30)", ""]
    lines += gl_part() + [""] + koz_part()
    (OUT / "sco2_floor_albedo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
