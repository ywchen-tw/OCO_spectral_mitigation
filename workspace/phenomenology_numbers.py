#!/usr/bin/env python3
"""Inline numbers of Sect. 5.3 and SI S1 that no figure or table carries.

  illum_shares   share of shadowed / neutral / brightened footprints within
                 5 km of cloud, per surface (Sect. 5.3: "48 % against 31 %")
  strong_shadow  O2 A-band Delta<l'> of strongly shadowed (z_gamma < -2) and of
                 brightened (z_gamma > +0.5) footprints within 5 km
                 (Sect. 5.3: ocean -0.010 / +0.003, land -0.007 / +0.007)
  signal_size    ocean median <l'> (O2 A-band, WCO2) at 0.5-1.5 km and 9-11 km
                 from cloud and the interquartile range there (SI S1)

Population (as Fig. 6): training dates, snow-free, both quality flags,
per-surface production reference (ocean r05, land r15), anomaly screen
|y| <= 100 ppm, footprints whose reference window mixes land and ocean
excluded (--keep-mixed-ref keeps them).  Illumination classes use the standardized O2 A-band gamma
departure z_gamma (shadowed < -0.5, brightened > +0.5).

--fit selects the path-length fit: nosg (default, PATH_LENGTH_FIT_SUFFIX) or
smoothed (reproduces the values computed before 2026-10-14).

Run:  python workspace/phenomenology_numbers.py [--fit nosg|smoothed] [--keep-mixed-ref]
Out:  results/figures/cld_dist_analysis/phenomenology_numbers_<fit>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from analysis.nosg_reference import DEFAULT_PARQUET, read_combined  # noqa: E402
from models.pipeline import MAX_ABS_ANOMALY_PPM  # noqa: E402

OUT_DIR = REPO / "results" / "figures" / "cld_dist_analysis"
SURFACES = {"ocean": (0, "r05"), "land": (1, "r15")}
NEAR_KM = 5.0
MIXED_FLAGS = (OUT_DIR / "spec_sensitivity" / "prodref" / "mixed_surface_windows"
               / "mixed_window_flags_snowfree_15km.parquet")
Z_SHADOW, Z_STRONG = 0.5, 2.0


def load(fit: str, exclude_mixed: bool) -> pd.DataFrame:
    gamma_ref = [f"{r}_exp_int_o2a{fit}_{s}" for r in ("r05", "r15") for s in ("mean", "std")]
    k1_ref = [f"{r}_{b}_k1{fit}_mean" for r in ("r05", "r15") for b in ("o2a", "wco2")]
    cols = (["fp_id", "sfc_type", "cld_dist_km", "snow_flag", "xco2_bc",
             "xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15",
             f"exp_o2a_intercept{fit}", f"o2a_k1{fit}", f"wco2_k1{fit}"]
            + gamma_ref + k1_ref)
    df = read_combined(DEFAULT_PARQUET, cols)
    df = df[(df.snow_flag == 0) & (df.xco2_bc > 0)]
    if exclude_mixed:
        flags = pd.read_parquet(MIXED_FLAGS, columns=["fp_id", "mixed"])
        df = df[~df.fp_id.isin(flags.loc[flags.mixed, "fp_id"].to_numpy())]
    return df


def per_surface(df: pd.DataFrame, fit: str, sfc: int, ref: str) -> pd.DataFrame:
    d = df[df.sfc_type == sfc]
    y = d[f"xco2_bc_anomaly_{ref}"]
    keep = np.isfinite(y) & (y.abs() <= MAX_ABS_ANOMALY_PPM)
    d = d[keep]
    z = ((d[f"exp_o2a_intercept{fit}"] - d[f"{ref}_exp_int_o2a{fit}_mean"])
         / d[f"{ref}_exp_int_o2a{fit}_std"].replace(0, np.nan))
    return pd.DataFrame({
        "cld": d.cld_dist_km.to_numpy(),
        "z": z.to_numpy(),
        "dk1_o2a": (d[f"o2a_k1{fit}"] - d[f"{ref}_o2a_k1{fit}_mean"]).to_numpy(),
        "k1_o2a": d[f"o2a_k1{fit}"].to_numpy(),
        "k1_wco2": d[f"wco2_k1{fit}"].to_numpy(),
    })


def numbers(fit: str, exclude_mixed: bool = True) -> pd.DataFrame:
    df = load(fit, exclude_mixed)
    rows = []
    for name, (sfc, ref) in SURFACES.items():
        d = per_surface(df, fit, sfc, ref)
        near = d[(d.cld >= 0) & (d.cld <= NEAR_KM) & np.isfinite(d.z)]
        n = len(near)
        for cls, m in (("shadowed", near.z < -Z_SHADOW),
                       ("neutral", near.z.abs() <= Z_SHADOW),
                       ("brightened", near.z > Z_SHADOW)):
            rows.append(dict(item="illum_shares", surface=name, what=cls,
                             value=float(m.sum() / n), n=int(m.sum())))
        for cls, m in (("strong_shadow_z<-2", near.z < -Z_STRONG),
                       ("brightened_z>0.5", near.z > Z_SHADOW)):
            v = near.dk1_o2a[m]
            rows.append(dict(item="strong_shadow", surface=name, what=f"dk1_o2a {cls}",
                             value=float(v.mean()), n=int(np.isfinite(v).sum())))
        if name == "ocean":
            for lo, hi in ((0.5, 1.5), (9.0, 11.0)):
                w = d[(d.cld >= lo) & (d.cld <= hi)]
                for band in ("o2a", "wco2"):
                    v = w[f"k1_{band}"].dropna()
                    q25, q50, q75 = np.percentile(v, [25, 50, 75])
                    rows.append(dict(item="signal_size", surface=name,
                                     what=f"k1_{band} median {lo}-{hi} km", value=q50, n=len(v)))
                    rows.append(dict(item="signal_size", surface=name,
                                     what=f"k1_{band} IQR {lo}-{hi} km", value=q75 - q25, n=len(v)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", choices=["nosg", "smoothed"], default="nosg")
    ap.add_argument("--keep-mixed-ref", action="store_true")
    args = ap.parse_args()
    fit = "_nosg" if args.fit == "nosg" else ""
    out = numbers(fit, exclude_mixed=not args.keep_mixed_ref)
    tag = "_allref" if args.keep_mixed_ref else ""
    path = OUT_DIR / f"phenomenology_numbers_{args.fit}{tag}.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))
    print("written", path)


if __name__ == "__main__":
    main()
