"""O2A truth check of the round-3 physical retrieval (2026-11-01).

The round-3 OE (phys_ppdf.py, run_phys.py) is rerun with the O2 scale freed: s_p prior sigma
0.2 % -> 5 % (bounds 0.85-1.15), everything else unchanged, with and without O2A f_am:
    run_phys.py slab|real --sp-sd 0.05 [--exclude-obs o2a_f_am] --out-name round3truth_<slab|real>_<f|nof>
O2 is known, so s_p-hat - 1 is the method's own path error at O2A (no xco2_bc needed).
This script only tabulates -> results_phys/round3truth_*.csv.  xco2_bc only for scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from path_transfer import SLAB_SUBSETS, corr, rsd  # noqa: E402

OUT = HERE / "results_phys"
PPM = 0.0025
SCREENS = (0.003, 0.005, 0.01)
DATE = "2018-07-01"


def real_tables():
    runs = {"round3 (s_p prior 0.2 %), with f": OUT / f"real_{DATE}.parquet",
            "round3 (s_p prior 0.2 %), no f": OUT / f"real_{DATE}_nof.parquet",
            "freed s_p (5 %), with f": OUT / "round3truth_real_f.parquet",
            "freed s_p (5 %), no f": OUT / "round3truth_real_nof.parquet"}
    o2, co2, scr = [], [], []
    for name, p in runs.items():
        d = pd.read_parquet(p)
        d = d[d.r_converged.fillna(False).astype(bool)]
        for s_, g in d.groupby("sfc_type"):
            surf = {0.0: "ocean", 1.0: "land"}[s_]
            sp, gh = g.r_s_p.to_numpy(float), g.r_g.to_numpy(float)
            pri, bc = g.xco2_apriori.to_numpy(float), g.xco2_bc.to_numpy(float)
            dev = bc / pri - 1
            est = pri * gh
            pr = (g.psfc / g.psfc_prior).to_numpy(float)
            o2.append(dict(run=name, surface=surf, n=len(g), conv=1.0,
                           med_sp_minus_1_pct=float(np.median(sp - 1) * 100), rsd_sp_pct=rsd(sp) * 100,
                           med_sp_ppm_eq=float(np.median(sp - 1) / PPM), rsd_sp_ppm_eq=rsd(sp) / PPM,
                           q10_q90_pct=f"{(np.quantile(sp, .1) - 1) * 100:.2f} / {(np.quantile(sp, .9) - 1) * 100:.2f}",
                           corr_sp_psfc_ratio=corr(sp, pr), rsd_psfc_ratio_pct=rsd(pr) * 100,
                           corr_sp_minus_psfcratio_vs_gerr=corr(sp - pr, gh - 1 - dev),
                           chi2_per_obs_med=float(np.median(g.r_chi2_obs) / sum(c.startswith("r_pull_") for c in g.columns)),
                           dof_g_med=float(g.r_dof_g.median()),
                           spearman_sp_aod=float(pd.Series(sp).corr(pd.Series(g.aod_total.to_numpy()), method="spearman"))))
            co2.append(dict(run=name, surface=surf, n=len(g), rsd_vs_bc=rsd(est - bc), bias_vs_bc=float(np.mean(est - bc)),
                            rsd_vs_prior=rsd(est - pri), rsd_bc_vs_prior=rsd(bc - pri),
                            R2=1 - (rsd(est - bc) / rsd(bc - pri)) ** 2, corr_bcdev=corr(est - pri, bc - pri),
                            corr_sp1_g1=corr(sp - 1, gh - 1), corr_sp1_bcdev=corr(sp - 1, dev),
                            corr_sp1_co2err=corr(sp - 1, gh - 1 - dev)))
            if name.startswith("freed"):
                for th in SCREENS:
                    k = np.abs(sp - 1) < th
                    scr.append(dict(run=name, surface=surf, screen=f"|s_p-1| < {th * 100:.1f} %",
                                    kept=float(k.mean()), n_kept=int(k.sum()),
                                    rsd_vs_bc_kept=rsd((est - bc)[k]), bias_vs_bc_kept=float(np.mean((est - bc)[k])) if k.any() else np.nan,
                                    rsd_prior_vs_bc_kept=rsd((pri - bc)[k]), rsd_vs_prior_kept=rsd((est - pri)[k]),
                                    corr_bcdev_kept=corr((est - pri)[k], (bc - pri)[k]),
                                    rsd_vs_bc_rejected=rsd((est - bc)[~k]), rsd_prior_vs_bc_rejected=rsd((pri - bc)[~k])))
        if name.startswith("round3"):
            continue
    for s_ in (0.0, 1.0):
        g = pd.read_parquet(runs["freed s_p (5 %), with f"])
        g = g[g.sfc_type == s_]
        pri, bc = g.xco2_apriori.to_numpy(float), g.xco2_bc.to_numpy(float)
        co2.insert(0, dict(run="prior only", surface={0.0: "ocean", 1.0: "land"}[s_], n=len(g),
                           rsd_vs_bc=rsd(pri - bc), bias_vs_bc=float(np.mean(pri - bc)), rsd_vs_prior=0.0,
                           rsd_bc_vs_prior=rsd(bc - pri), R2=0.0))
    return pd.DataFrame(o2), pd.DataFrame(co2), pd.DataFrame(scr)


def slab_table():
    runs = {"round3 (s_p prior 0.2 %), with f": OUT / "slab_phys_scenes.parquet",
            "round3 (s_p prior 0.2 %), no f": OUT / "slab_phys_scenes_nof.parquet",
            "freed s_p (5 %), with f": OUT / "round3truth_slab_f.parquet",
            "freed s_p (5 %), no f": OUT / "round3truth_slab_nof.parquet"}
    rows = []
    for name, p in runs.items():
        d = pd.read_parquet(p)
        for tag, noise in (("", "noise-free"), ("n_", "noisy")):
            esp = ((d[f"r_{tag}s_p"] - 1) / PPM).to_numpy()
            eg = ((d[f"r_{tag}g"] - d.g_co2) / PPM).to_numpy()
            for sub in ("all", "clear (no cloud, tau_a<0.1)", "cloudy", "land", "ocean",
                        "no cloud, tau_a<0.05", "no cloud, 0.1-0.2", "no cloud, 0.2-0.5"):
                m = SLAB_SUBSETS[sub](d) & np.isfinite(esp) & np.isfinite(eg)
                rows.append(dict(run=name, noise=noise, subset=sub, n=int(m.sum()),
                                 rsd_sp_ppm_eq=rsd(esp[m]), med_sp_ppm_eq=float(np.median(esp[m])),
                                 rsd_sp_pct=rsd(esp[m]) * PPM * 100,
                                 rsd_g_ppm=rsd(eg[m]), med_g_ppm=float(np.median(eg[m])),
                                 corr_sp_err_g_err=corr(esp[m], eg[m])))
    return pd.DataFrame(rows)


def main():
    o2, co2, scr = real_tables()
    sl = slab_table()
    for n, t in (("real_o2a_truth", o2), ("real_co2", co2), ("real_screen", scr), ("slab", sl)):
        t.round(4).to_csv(OUT / f"round3truth_{n}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(o2.round(3).to_string(index=False))
    print(co2.round(3).to_string(index=False))
    print(scr.round(3).to_string(index=False))
    print(sl[sl.subset.isin(["all", "clear (no cloud, tau_a<0.1)", "cloudy", "land", "ocean"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
