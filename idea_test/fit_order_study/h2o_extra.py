"""Extra diagnostics after the pre-registered bars (not part of the decision table).

1. H2O-removed near-cloud residual: per target, regress the latitude-local contrast on the
   h2o_scale contrast (and h2o_scale + tcwv) over far-clear soundings, apply to all soundings,
   and report the median residual per cloud-distance bin with its SNR (far residual robust SD).
2. Bars A/B restricted to QF0 soundings.
3. (--lat-breakdown) near-cloud contrasts by 5° latitude band, to see which scenes carry them.
Outputs: results/h2o_extra_29252a_GL.csv, results/h2o_lat_breakdown_29252a_GL.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_sweep_real import DEFAULTS  # noqa: E402
from h2o_analysis import BIN_LABELS, BINS, TAG, lat_local_contrast, load_extra  # noqa: E402
from run_study import robust_sd  # noqa: E402

OUT = HERE / "results"


def lat_breakdown():
    out_file = OUT / f"h2o_lat_breakdown_{TAG}.csv"
    if out_file.exists():
        raise FileExistsError(out_file)
    long = pd.read_parquet(OUT / f"h2o_per_sounding_{TAG}.parquet")
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        cld = f["cld_dist_km"][...].astype(float)
        qf = f["xco2_qf"][...]
        lat = f["lat"][...].astype(float)
        fp = f["fp_number"][...].astype(int)
        hs_v = f["h2o_scale"][...].astype(float)
    n = len(cld)
    cols = {"h2o_scale_con": lat_local_contrast(hs_v, cld, qf, lat, fp, False)}
    w = long[long.band == "o2a"].set_index("j").reindex(np.arange(n))
    cols["o2a_S2_k1_con"] = lat_local_contrast(w["S2_k1"].values.astype(float), cld, qf, lat, fp, True)
    for band in ("sco2", "wco2"):
        w = long[long.band == band].set_index("j").reindex(np.arange(n))
        for q in ("S2_k1", "T2_k1_dry", "T2_ratio_w_dry"):
            cols[f"{band}_{q}_con"] = lat_local_contrast(w[q].values.astype(float), cld, qf, lat, fp, True)
        cols[f"{band}_S2_k2_con"] = lat_local_contrast(w["S2_k2"].values.astype(float), cld, qf, lat, fp, False)
    df = pd.DataFrame(cols)
    df["lat_band"] = np.floor(lat / 5) * 5
    df["qf1"] = qf != 0
    df["bin"] = pd.cut(cld, [0, 1, 3], right=False, labels=["0-1", "1-3"])
    df = df[df.bin.notna() & df.sco2_S2_k1_con.notna()]
    agg = df.groupby(["bin", "lat_band"], observed=True).agg(
        n=("h2o_scale_con", "size"), qf1_frac=("qf1", "mean"),
        **{c: (c, "median") for c in cols}).reset_index()
    agg.to_csv(out_file, index=False)
    pd.set_option("display.width", 250)
    print(agg.to_string(float_format=lambda x: f"{x:.3g}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-breakdown", action="store_true")
    if ap.parse_args().lat_breakdown:
        lat_breakdown()
        return
    out_file = OUT / f"h2o_extra_{TAG}.csv"
    if out_file.exists():
        raise FileExistsError(out_file)
    long = pd.read_parquet(OUT / f"h2o_per_sounding_{TAG}.parquet")
    with h5py.File(DEFAULTS["fit_details"], "r") as f:
        cld = f["cld_dist_km"][...].astype(float)
        qf = f["xco2_qf"][...]
        lat = f["lat"][...].astype(float)
        fp = f["fp_number"][...].astype(int)
    n = len(cld)
    ex = load_extra(n)
    far = (cld > 20) & (qf == 0)
    hs = lat_local_contrast(ex["h2o_scale"], cld, qf, lat, fp, False)
    tc = lat_local_contrast(ex["tcwv"], cld, qf, lat, fp, True)
    recs = []
    targets = [("sco2", "S2_k1", True), ("sco2", "S2_k2", False), ("sco2", "T2_ratio_w_dry", True),
               ("sco2", "T2_k1_w", True), ("wco2", "S2_k1", True), ("wco2", "S2_k2", False),
               ("wco2", "T2_ratio_w_dry", True), ("wco2", "T2_k1_w", True)]
    for band, q, rel in targets:
        v = long[long.band == band].set_index("j").reindex(np.arange(n))[q].values.astype(float)
        con = lat_local_contrast(v, cld, qf, lat, fp, rel)
        for model, xs in (("h2o_scale", [hs]), ("h2o_scale+tcwv", [hs, tc])):
            ok = np.isfinite(con) & np.all([np.isfinite(x) for x in xs], axis=0)
            mf = far & ok
            a = np.column_stack([np.ones(n), *xs])
            coef = np.linalg.lstsq(a[mf], con[mf], rcond=None)[0]
            pred = a @ coef
            res = con - pred
            sd = robust_sd(res[mf])
            for (lo, hi), lab in zip(BINS[:2], BIN_LABELS[:2]):
                mb = ok & (cld >= lo) & (cld < hi)
                recs.append(dict(target=f"{band}_{q}", model=model, bin=lab, n=int(mb.sum()),
                                 slope_h2o_scale=coef[1], obs_median=np.median(con[mb]),
                                 pred_median=np.median(pred[mb]), resid_median=np.median(res[mb]),
                                 resid_snr=np.median(res[mb]) / sd, h2o_scale_con_median=np.median(hs[mb])))
        # QF0-only near-cloud contrast (bars A/B check)
        far_sd = robust_sd(con[far])
        for (lo, hi), lab in zip(BINS[:2], BIN_LABELS[:2]):
            mb = np.isfinite(con) & (cld >= lo) & (cld < hi) & (qf == 0)
            recs.append(dict(target=f"{band}_{q}", model="QF0 only (no H2O model)", bin=lab, n=int(mb.sum()),
                             obs_median=np.median(con[mb]), resid_snr=np.median(con[mb]) / far_sd))
    for q in ("T2_k1_dry", "T2_k2"):
        v = long[long.band == "sco2"].set_index("j").reindex(np.arange(n))[q].values.astype(float)
        con = lat_local_contrast(v, cld, qf, lat, fp, q.endswith("dry"))
        far_sd = robust_sd(con[far])
        for (lo, hi), lab in zip(BINS[:2], BIN_LABELS[:2]):
            mb = np.isfinite(con) & (cld >= lo) & (cld < hi) & (qf == 0)
            recs.append(dict(target=f"sco2_{q}", model="QF0 only (no H2O model)", bin=lab, n=int(mb.sum()),
                             obs_median=np.median(con[mb]), resid_snr=np.median(con[mb]) / far_sd))
    df = pd.DataFrame(recs)
    df.to_csv(out_file, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
