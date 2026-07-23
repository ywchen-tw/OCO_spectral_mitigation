#!/usr/bin/env python3
"""Manuscript Fig. 5 — shadow vs brightening branches (Results 4.2).

Redraws the land shadow/brightening panel figure in the locked AMT style
from the saved binned statistics (no parquet needed — replaces the
pre-restyle copy from the full-parquet run, 2026-07-22r). The legend sits
BELOW the panels (the in-panel legend of the analysis version covered the
O2A curves).

Branches split near-cloud (< 10 km) land footprints by the sign of the
O2A continuum-reflectance departure from the clear-sky reference
(z(exp-int O2A) vs ±0.5): brightened / neutral / shadowed. Bins with
n < 500 are suppressed (same rule as the analysis run).

Data:   results/figures/cld_dist_analysis/spec_sensitivity/
        shadow_brightening_stats.csv
Output: manuscript/figures/fig05_shadow_brightening_land.{png,pdf}
        (--surface ocean writes the appendix-companion ocean variant to
        internal_shadow_brightening_ocean.* until it gets a slot)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import MEAN_L_LABEL, apply_manuscript_style, panel_label

CSV = (REPO / "results" / "figures" / "cld_dist_analysis"
       / "spec_sensitivity" / "shadow_brightening_stats.csv")
OUT = REPO / "manuscript" / "figures"

N_MIN = 500
BIN_ORDER = ["0–1", "1–2", "2–3", "3–5", "5–7", "7–10"]
BRANCHES = [("brightened", "#d95f02"),
            ("neutral", "#7570b3"),
            ("shadowed", "#1b9e77")]
PANELS = [("dk1_o2a", f"$\\Delta${MEAN_L_LABEL} O$_2$A"),
          ("dk1_wco2", f"$\\Delta${MEAN_L_LABEL} WCO$_2$"),
          ("dk1_sco2", f"$\\Delta${MEAN_L_LABEL} SCO$_2$"),
          ("xco2_bc_anomaly",
           "$\\Delta X_{\\mathrm{CO2}}^{\\mathrm{B11}}$ (ppm)")]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", choices=["land", "ocean"], default="land")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    df = df[(df["surface"] == args.surface) & (df["n"] >= N_MIN)]
    x = np.arange(len(BIN_ORDER))

    apply_manuscript_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    for k, (ax, (var, vlabel)) in enumerate(zip(axes.ravel(), PANELS)):
        panel_label(ax, f"({chr(ord('a') + k)})")
        for bname, color in BRANCHES:
            g = (df[(df["variable"] == var) & (df["branch"] == bname)]
                 .set_index("cld_bin").reindex(BIN_ORDER))
            ax.errorbar(x, g["mean"], yerr=g["sem"], color=color, marker="o",
                        ms=3.5, capsize=2, lw=1.3, label=bname)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(vlabel, fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xticks(x, BIN_ORDER, rotation=45, ha="right")
        ax.set_xlabel("Nearest-cloud distance bin (km)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9,
               title="O$_2$A continuum-reflectance branch "
                     "($z$(exp-int) vs $\\pm$0.5)", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    stem = ("fig05_shadow_brightening_land" if args.surface == "land"
            else "internal_shadow_brightening_ocean")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
