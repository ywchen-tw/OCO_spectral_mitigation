#!/usr/bin/env python3
"""Manuscript Fig. 7 — per-feature permutation importance of the production
deep ensemble (Sect. 4.3; added 2026-07-23j, renumbering downstream figures
8-12).

Per surface, the top-N individual features by permutation importance:
increase in held-fold RMSE when the feature is permuted (median over the
five date-blocked folds +- fold sd), from the feature-importance run

  results/model_comparison/feature_importance/{ocean,land}/
      importance_de_{ocean,land}_agg.csv   (stratum=global, scope=feature)

Bars are colored by predictor group. NOTE the standing caveat (carried in
the Sect. 4.3 draft text, not the caption): held-out CV importance
over-credits TCCON-neutral blocks (contamination), so the retrained
group-ablation (Fig. 6b, Table 2) remains the primary attribution
evidence; this figure shows relative individual contributions under
collinearity.

Output: manuscript/figures/fig07_feature_importance.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style
from plot_style import apply_manuscript_style, panel_label

apply_manuscript_style()

FI_DIR = REPO / "results" / "model_comparison" / "feature_importance"
OUT_DIR = REPO / "manuscript" / "figures"
TOP_N = 12

GROUP_COLOR = {          # Okabe-Ito, one color per predictor group
    "xco2": "#0072B2",
    "met_other": "#E69F00",
    "profile": "#009E73",
    "contam": "#D55E00",
    "spec": "#CC79A7",
    "geometry": "#56B4E9",
    "fp_onehot": "0.6",
}
GROUP_LABEL = {
    "xco2": "retrieval state",
    "met_other": "meteorology/surface",
    "profile": "profile EOFs",
    "contam": "contamination",
    "spec": "spectral path-length",
    "geometry": "geometry/L1B",
    "fp_onehot": "footprint",
}


def _load(surface: str) -> pd.DataFrame:
    d = pd.read_csv(FI_DIR / surface / f"importance_de_{surface}_agg.csv")
    d = d[(d.stratum == "global") & (d.scope == "feature")]
    return d.sort_values("delta_rmse", ascending=False).head(TOP_N)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    used_groups: list = []
    for ax, surface, letter in zip(axes, ("ocean", "land"), "ab"):
        d = _load(surface)
        ys = np.arange(len(d))[::-1]
        colors = [GROUP_COLOR[g] for g in d.group]
        ax.barh(ys, d.delta_rmse, xerr=d.fold_sd, height=0.62,
                color=colors, error_kw=dict(lw=0.9, capsize=2), zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels(d.name, fontsize=7, family="monospace")
        ax.set_xlabel("permutation $\\Delta$RMSE (ppm)")
        ax.xaxis.grid(True, color="0.9", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, f"({letter}) {surface}")
        for g in d.group:
            if g not in used_groups:
                used_groups.append(g)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g])
               for g in used_groups]
    fig.legend(handles, [GROUP_LABEL[g] for g in used_groups],
               frameon=False, fontsize=7, ncol=min(4, len(used_groups)),
               loc="lower center", bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig07_feature_importance.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
