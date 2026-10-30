"""Sensitivity of the replication metrics to the far-clear set definition (not a bar).

Re-aggregates the stored ``rep_per_sounding_*`` fits under several far-set
definitions (proxy B or A, threshold 10/20/30 km; MODIS truth on 29252a) and
adds a latitude-local version of the SCO2/O2A ratio spread (each far-clear
ratio minus the median of far-clear ratios within ±1° latitude), which removes
along-track geophysical gradients so that orbits with different far-set
latitude spans can be compared.

Usage:  python idea_test/fit_order_study/rep_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from rep_orbit_io import cloud_proxy_km, l1b_geolocation  # noqa: E402
from rep_run import aggregate_split  # noqa: E402
from run_study import md_table, robust_sd  # noqa: E402
from run_weighting import lat_local_ref  # noqa: E402

OUT = HERE / "results"
SCRATCH_L1B_22846 = Path("/private/tmp/claude-370367/-Volumes-YWC-SSD-programming-oco-fp-analysis/"
                         "106fa6df-e817-4e62-8f1a-c441f33123ad/scratchpad/dl_22846a/"
                         "oco2_L1bScGL_22846a_181018_B11006r_220921190030.h5")
ORBITS = {
    "29252a_GL": (REPO / "data/OCO2/2020/001/29252a_GL/oco2_L1bScGL_29252a_191231_B11006r_220715045716.h5",
                  REPO / "results/2020-01-01/29252a_GL/fp_tau_combined.h5"),
    "22845a_GL": (REPO / "data/OCO2/2018/291/22845a_GL/oco2_L1bScGL_22845a_181018_B11006r_220921185957.h5",
                  REPO / "results/2018-10-18/22845a_GL/fp_tau_combined.h5"),
    "22846a_GL": (SCRATCH_L1B_22846, REPO / "results/2018-10-18/22846a_GL/fp_tau_combined.h5"),
}
TAGS = ("ols_2", "ols_3", "ols_7", "gamma")


def local_ratio_spread(long, meta, fp, far, tag):
    w = long[long.variant == tag].pivot_table(index="j", columns="band", values="k1")
    j = w.index.values
    r = (w["sco2"] / w["o2a"]).values
    f = far[j] & np.isfinite(r)
    fpm = pd.Series(r[f]).groupby(fp[j][f]).mean()
    rel = r / fpm.reindex(fp[j]).values - 1.0
    loc = rel - lat_local_ref(rel, meta["lat"][j], f)
    ok = f & np.isfinite(loc)
    return 100 * robust_sd(loc[ok]), int(ok.sum())


def main():
    rows = []
    for orbit, (l1b, fp_tau) in ORBITS.items():
        per = pd.read_parquet(OUT / f"rep_per_sounding_{orbit}.parquet")
        long = per[["j", "band", "variant", "k1", "k2", "intercept", "sig_k1", "sig_k2", "sig_k1_rss",
                    "chi2_red", "resid_rms", "lag1", "bound", "nchan"]]
        first = per.drop_duplicates("j").set_index("j").sort_index()
        n = int(first.index.max()) + 1
        assert np.array_equal(first.index.values, np.arange(n))
        fp = first.fp_number.values
        base = {"xco2_qf": first.xco2_qf.values, "xco2_bc_anomaly": first.xco2_bc_anomaly.values,
                "lat": first.lat.values, "is_land": first.land_fraction.values >= 50.0}
        with h5py.File(fp_tau, "r") as f:
            sid = f["sounding_id"][...].astype(np.int64)
        assert np.array_equal(sid, first.sounding_id.values)
        dist = {"B": first.cld_dist_km.values,
                "A": cloud_proxy_km(l1b_geolocation(str(l1b)), sid, sid)}
        if "cld_dist_true_km" in first:
            dist["true"] = first.cld_dist_true_km.values
        defs = [("B", 20.0), ("B", 10.0), ("A", 20.0), ("A", 30.0)] + ([("true", 20.0)] if "true" in dist else [])
        for lab, km in defs:
            meta = dict(base, cld_dist_km=dist[lab])
            s = aggregate_split(long, meta, fp, km)
            s = s[s.subset == "all"].set_index(["band", "variant"])
            far = (meta["cld_dist_km"] > km) & (meta["xco2_qf"] == 0)
            rec = {"orbit": orbit, "labels": lab, "far_km": km, "n_far": int(far.sum()),
                   "far_lat_range": f"{np.nanmin(base['lat'][far]):.0f}..{np.nanmax(base['lat'][far]):.0f}"}
            for t in TAGS:
                rec[f"ratio_{t}"] = s.loc[("sco2", t), "ratio_to_o2a_spread_pct"]
            for t in ("ols_3", "ols_7"):
                rec[f"o2a_{t}"] = s.loc[("o2a", t), "k1_spread_pct"]
                rec[f"sco2_{t}"] = s.loc[("sco2", t), "k1_spread_pct"]
            rec["ratio_factor_7_3"] = rec["ratio_ols_7"] / rec["ratio_ols_3"]
            for t in ("ols_2", "ols_3", "ols_7"):
                rec[f"ratio_loc_{t}"], rec["n_loc"] = local_ratio_spread(long, meta, fp, far, t)
            rec["ratio_loc_factor_7_3"] = rec["ratio_loc_ols_7"] / rec["ratio_loc_ols_3"]
            rows.append(rec)
            print(orbit, lab, km, rec["n_far"], f"{rec['ratio_ols_3']:.2f} {rec['ratio_ols_2']:.2f} "
                  f"{rec['ratio_ols_7']:.2f} loc {rec['ratio_loc_ols_3']:.2f} {rec['ratio_loc_ols_2']:.2f} "
                  f"{rec['ratio_loc_ols_7']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rep_sensitivity.csv", index=False)
    (OUT / "rep_sensitivity.md").write_text(
        "# Far-set sensitivity (not a bar)\n\nratio_* = SCO2/O2A k1 ratio spread (%), global over the far set; "
        "ratio_loc_* = same after subtracting the ±1° latitude-local far-set median.\n\n"
        + md_table(df, list(df.columns)) + "\n")
    print(md_table(df, list(df.columns)))


if __name__ == "__main__":
    main()
