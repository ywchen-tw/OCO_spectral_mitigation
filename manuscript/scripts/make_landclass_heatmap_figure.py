#!/usr/bin/env python3
"""Manuscript Fig. 4 — surface-stratified near-cloud spectral response heatmap.

Population: QF = 0, snow-free, valid xco2_bc — identical to the scored
run_all land-class suite (utils.apply_quality_filter); with QF1/snow rows
included the barren column flips sign, so the filter is load-bearing.
`--qf {0,1,all}` (2026-07-22) selects the quality-flag population for the
sensitivity comparison (snow-free always): QF1-only (`_qf1` suffix) and
all-QF (`_allqf`) variants isolate how much of the signal lives in flagged
scenes — QF is mechanically unrelated to the L1B-derived spectral features,
but flagged soundings carry competing scattering perturbations (in-FOV
contamination, aerosol) at class- and distance-dependent rates.

Reference (PRIMARY since 2026-07-22e): per-surface, ocean r05 / land r15 —
consistent with the production target radii; Fig. 3 shows the land response
extends past 10 km, so the common 10-km reference is contaminated over land
and attenuates the effects (barren −0.40σ vs −1.29σ). The common-r10
variant remains available as robustness (`--reference common-r10`, _r10
suffix). Caveat: the stricter r15 reference roughly halves the valid
near-cloud land sample and shifts scene selection toward less-cloudy
orbits; the small urban class is reference-sensitive (−0.89σ → −0.02σ).

Rows: ref-corrected per-sounding deltas (obs − clear-sky-neighbor reference)
of the path-length statistics and continuum reflectance, per band. Columns:
OCEAN (added 2026-07-22 as the dark endpoint of the cloud−surface
albedo-contrast axis) followed by the MCD12C1 IGBP land-cover groups. Cell:
near-cloud (0–5 km) minus far-cloud (20–50 km) effect in far-cloud z-units
with an analytic 95% CI (asterisk = |effect| exceeds its CI).

Reuses the generic engine in src/analysis/land_class.py (build_effect_sizes)
and land_cover.py (MCD12C1 lookup); the heatmap itself is drawn here in the
locked AMT style with the L′ typesetting.

Data:  results/csv_collection/combined_2016_2020_dates.parquet
       data/MODIS/MCD12C1/MCD12C1.A{2016..2020}001.061.*.hdf
Output: manuscript/figures/fig04_landclass_effect_heatmap.{png,pdf}
        manuscript/figures/fig04_landclass_effect_sizes.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style
sys.path.insert(0, str(REPO / "src"))        # analysis package (bypasses
                                             # src/__init__, which pulls the
                                             # whole pipeline + config)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import CMAPS, apply_manuscript_style

from analysis.land_class import LAND_CLASS_DELTA_VARS, build_effect_sizes
from analysis.land_cover import GROUP_ORDER, assign_land_cover
from analysis.ref_corrected import add_ref_anomalies

PARQUET = REPO / "results" / "csv_collection" / "combined_2016_2020_dates.parquet"
MCD12C1_DIR = REPO / "data" / "MODIS" / "MCD12C1"
OUT_DIR = REPO / "manuscript" / "figures"

OBS_COLS = ["o2a_k1", "o2a_k2", "wco2_k1", "wco2_k2", "sco2_k1", "sco2_k2",
            "exp_o2a_intercept", "exp_wco2_intercept", "exp_sco2_intercept"]
REF_COLS = [f"ref_{b}_k{k}_{s}" for b in ("o2a", "wco2", "sco2")
            for k in (1, 2) for s in ("mean", "std")] + \
           [f"ref_exp_int_{b}_{s}" for b in ("o2a", "wco2", "sco2")
            for s in ("mean", "std")] + \
           [f"{p}_{b}_k{k}_mean" for p in ("r05", "r15")
            for b in ("o2a", "wco2", "sco2") for k in (1, 2)] + \
           [f"{p}_exp_int_{b}_mean" for p in ("r05", "r15")
            for b in ("o2a", "wco2", "sco2")]

MIN_CLASS_N = 10_000   # drop land groups thinner than this (matches land_class)
N_MIN = 500            # per-cell minimum (matches land_class default)
EXCLUDED = ("water", "fill")   # coastal-cell leakage / no data


def load(qf: str = "0") -> pd.DataFrame:
    import pyarrow.parquet as pq
    cols = (["date", "lon", "lat", "cld_dist_km", "sfc_type",
             "xco2_bc", "xco2_qf", "snow_flag"] + OBS_COLS + REF_COLS)
    df = pq.read_table(PARQUET, columns=cols).to_pandas()
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    # qf="0" reproduces the scored run_all land-class population
    # (utils.apply_quality_filter): valid xco2_bc, QF == 0, snow-free
    df = df[(df["xco2_bc"] > 0) & (df["snow_flag"] == 0)]
    if qf in ("0", "1"):
        df = df[df["xco2_qf"] == int(qf)]
    df = df[np.isfinite(df["cld_dist_km"]) & (df["cld_dist_km"] >= 0)]
    return df


def add_persurface_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Delta columns against the PER-SURFACE references (ocean r05, land r15),
    consistent with the production target radii. Rows whose scene lacks the
    stricter reference population become NaN (land near-cloud n roughly
    halves; scene selection shifts toward less-cloudy orbits)."""
    ocean = df["sfc_type"] == 0
    for b in ("o2a", "wco2", "sco2"):
        for k in (1, 2):
            ref = np.where(ocean, df[f"r05_{b}_k{k}_mean"],
                           df[f"r15_{b}_k{k}_mean"])
            df[f"dk{k}_{b}"] = df[f"{b}_k{k}"] - ref
        refe = np.where(ocean, df[f"r05_exp_int_{b}_mean"],
                        df[f"r15_exp_int_{b}_mean"])
        df[f"dexp_{b}"] = df[f"exp_{b}_intercept"] - refe
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", choices=["per-surface", "common-r10"],
                    default="per-surface",
                    help="clear-sky reference for the spectral deltas: the "
                         "production per-surface radii (ocean r05, land r15 "
                         "— PRIMARY since 2026-07-22e) or the common 10-km "
                         "reference (robustness variant; outputs get a _r10 "
                         "suffix)")
    ap.add_argument("--qf", choices=["0", "1", "all"], default="0",
                    help="quality-flag population (snow-free always): 0 = "
                         "primary scored population (no suffix), 1 = "
                         "flagged-only (_qf1 suffix), all = no QF filter "
                         "(_allqf suffix)")
    args = ap.parse_args()
    suffix = ("" if args.reference == "per-surface" else "_r10") + \
             {"0": "", "1": "_qf1", "all": "_allqf"}[args.qf]

    df = load(args.qf)
    if args.reference == "common-r10":
        df = add_ref_anomalies(df)
    else:
        df = add_persurface_anomalies(df)

    # group column: ocean + IGBP groups over land
    df = assign_land_cover(df, MCD12C1_DIR)
    grp = df["igbp_group"].astype(str)
    grp[df["sfc_type"] == 0] = "ocean"
    df["grp"] = grp

    counts = df["grp"].value_counts()
    land_groups = [g for g in GROUP_ORDER
                   if g not in EXCLUDED and counts.get(g, 0) >= MIN_CLASS_N]
    groups = ["ocean"] + land_groups
    print("groups:", {g: int(counts.get(g, 0)) for g in groups})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eff = build_effect_sizes(df, LAND_CLASS_DELTA_VARS, str(OUT_DIR), groups,
                             n_min=N_MIN, group_col="grp",
                             prefix=f"fig04_landclass{suffix}")

    # ---- manuscript heatmap -------------------------------------------------
    apply_manuscript_style()
    labels_by_var = dict(zip(eff["variable"], eff["label"]))
    mat = eff.pivot(index="variable", columns="group", values="effect_z")
    ci = eff.pivot(index="variable", columns="group", values="ci95_z")
    var_order = [v for v, _ in LAND_CLASS_DELTA_VARS if v in mat.index]
    mat = mat.reindex(index=var_order, columns=groups)
    ci = ci.reindex(index=var_order, columns=groups)
    # drop groups with no valid cell at all (class-size threshold passed but
    # no cell reaches n_min in both windows, e.g. snow_ice in the QF1 runs)
    dead = [g for g in groups if not np.isfinite(mat[g]).any()]
    if dead:
        print("dropping all-NaN groups:", dead)
        groups = [g for g in groups if g not in dead]
        mat, ci = mat[groups], ci[groups]

    vmax = np.nanmax(np.abs(mat.values))
    fig, ax = plt.subplots(
        figsize=(0.95 * len(groups) + 2.4, 0.42 * len(var_order) + 1.6))
    im = ax.imshow(mat.values, cmap=CMAPS["mu"], vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(groups)),
                  [g.replace("_", " ") for g in groups], rotation=40,
                  ha="right")
    ax.set_yticks(range(len(var_order)),
                  [labels_by_var[v] for v in var_order], fontsize=8)
    for i in range(len(var_order)):
        for j in range(len(groups)):
            v, c = mat.values[i, j], ci.values[i, j]
            if np.isfinite(v):
                sig = abs(v) > c
                ax.text(j, i, f"{v:+.2f}{'*' if sig else ''}", ha="center",
                        va="center", fontsize=6.5,
                        fontweight="bold" if sig else "normal",
                        color="white" if abs(v) > 0.6 * vmax else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.5,
                        color="gray")
    # separate the ocean column from the land classes
    ax.axvline(0.5, color="0.2", lw=1.0)
    ax.tick_params(length=0)
    if args.qf != "0":
        ax.set_title({"1": "QF = 1 soundings only",
                      "all": "all quality flags"}[args.qf],
                     fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, label="near $-$ far effect (z)", pad=0.015)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig04_landclass_effect_heatmap{suffix}.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
