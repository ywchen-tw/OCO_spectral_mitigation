"""Near-cloud vs far-clear fit coefficients on production τ and Doppler-corrected τ (orbit 29252a_GL).

For every sounding: unconstrained order-2 (k1, k2) and bounded order-3 k1 per band, on both τ sets.
Contrast = value − latitude-local far reference (median of cld_dist > 10 km, QF0 soundings within
±1° latitude, same footprint); SNR = median contrast of a cloud-distance bin / robust SD of the
far-clear contrasts. Outputs results/near_doppler_29252a_GL.csv and near_doppler_summary.md.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS, md_table  # noqa: E402
from orbit_io import BANDS, load_orbit  # noqa: E402
from run_study import robust_sd  # noqa: E402
from variants import design_matrix, fit_cumulant  # noqa: E402

OUT = HERE / "results"
TAG = "29252a_GL"
BINS = [(0, 1), (1, 3), (3, 10), (10, 20), (20, 1e9)]
_G: dict = {}


def _fit(j):
    rows = []
    for src in ("prod", "corr"):
        tau = _G["tau_" + src]
        for ib, band in enumerate(BANDS):
            t = tau[ib, j] if src == "prod" else tau[j, ib]
            y, s = _G["ln_t"][ib, j], _G["sig"][ib, j]
            ok = np.isfinite(y) & np.isfinite(t) & np.isfinite(s) & (s > 0)
            if ok.sum() < 10:
                rows.append((j, src, band, np.nan, np.nan, np.nan, np.nan))
                continue
            c2 = np.linalg.lstsq(design_matrix(t[ok].astype(float), 2), y[ok].astype(float), rcond=None)[0]
            r3 = fit_cumulant(t, y, s, 3, weighted=False)
            rows.append((j, src, band, c2[1], c2[2], r3["k1"], r3["resid_rms"]))
    return rows


def _chunk(js):
    out = []
    for j in js:
        out += _fit(j)
    return out


def main():
    orb = load_orbit(str(DEFAULTS["l1b"]), str(DEFAULTS["fp_tau"]), str(DEFAULTS["fit_details"]))
    tau_corr = np.load(OUT / f"resid_doppler_tau_{TAG}.npz")["tau"]
    _G.update(tau_prod=orb.tau, tau_corr=tau_corr, ln_t=orb.ln_t, sig=orb.sigma_ln_t)
    n = orb.tau.shape[1]
    with mp.get_context("fork").Pool(6) as pool:
        parts = pool.map(_chunk, np.array_split(np.arange(n), 48))
    df = pd.DataFrame([r for p in parts for r in p], columns=["j", "src", "band", "k1_o2", "k2_o2", "k1_o3", "rms_o3"])
    df.to_parquet(OUT / f"near_doppler_per_sounding_{TAG}.parquet")

    cld = orb.meta["cld_dist_km"]
    qf = orb.meta["xco2_qf"]
    lat = orb.meta["lat"]
    fp = orb.fp_number
    anom = orb.meta["xco2_bc_anomaly"]
    ref_ok = (cld > 10) & (qf == 0)
    far = (cld > 20) & (qf == 0)
    recs = []
    for src in ("prod", "corr"):
        w = df[df.src == src].pivot(index="j", columns="band", values=["k1_o2", "k2_o2", "k1_o3", "rms_o3"])
        vals = {("ratio", "k1_o2"): w[("k1_o2", "sco2")] / w[("k1_o2", "o2a")],
                ("ratio", "k1_o3"): w[("k1_o3", "sco2")] / w[("k1_o3", "o2a")]}
        for band in BANDS:
            for q in ("k1_o2", "k2_o2", "k1_o3", "rms_o3"):
                vals[(band, q)] = w[(q, band)]
        for (band, q), v in vals.items():
            v = v.reindex(range(n)).values
            # latitude-local far reference per sounding (±1°, same footprint)
            ref = np.full(n, np.nan)
            for f in range(8):
                m_f = fp == f
                rl = lat[ref_ok & m_f]
                rv = v[ref_ok & m_f]
                for j in np.flatnonzero(m_f):
                    sel = np.abs(rl - lat[j]) <= 1.0
                    if sel.sum() >= 5:
                        ref[j] = np.nanmedian(rv[sel])
            con = v - ref
            if q.startswith("k1") or q == "rms_o3" or band == "ratio":
                con = con / ref                      # relative contrast
            far_sd = robust_sd(con[far])
            rec = {"src": src, "band": band, "quantity": q, "far_median": np.nanmedian(v[far]),
                   "far_spread": far_sd}
            for lo, hi in BINS:
                m = (cld >= lo) & (cld < hi) & np.isfinite(con)
                lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
                rec[f"n_{lab}"] = int(m.sum())
                rec[f"con_{lab}"] = np.nanmedian(con[m]) if m.sum() > 20 else np.nan
                rec[f"snr_{lab}"] = rec[f"con_{lab}"] / far_sd if m.sum() > 20 else np.nan
            mn = (cld < 3) & np.isfinite(con) & np.isfinite(anom)
            rec["corr_anom_<3km"] = float(np.corrcoef(con[mn], anom[mn])[0, 1]) if mn.sum() > 30 else np.nan
            recs.append(rec)
    summ = pd.DataFrame(recs)
    summ.to_csv(OUT / f"near_doppler_{TAG}.csv", index=False)
    cols = ["src", "band", "quantity", "far_median", "far_spread", "n_0-1", "con_0-1", "snr_0-1", "con_1-3", "snr_1-3",
            "con_3-10", "snr_3-10", "con_10-20", "snr_10-20", "corr_anom_<3km"]
    lines = [f"# Near-cloud vs far-clear, production vs Doppler-corrected τ — {TAG}", "",
             "Contrast = value − latitude-local far reference (cld > 10 km, QF0, ±1° lat, same footprint); "
             "relative for k1, ratio and rms, absolute for k2. SNR = median contrast / robust SD of far-clear "
             "contrasts. Unconstrained order 2 (k1_o2, k2_o2), bounded order 3 (k1_o3). All QF.", ""]
    for q in ("k1_o2", "k2_o2", "k1_o3", "rms_o3"):
        lines += [f"## {q}", "", md_table(summ[summ.quantity == q].sort_values(["band", "src"]), cols), ""]
    (OUT / "near_doppler_summary.md").write_text("\n".join(lines))
    print(summ[summ.quantity.isin(["k1_o2", "k2_o2", "k1_o3"])][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
