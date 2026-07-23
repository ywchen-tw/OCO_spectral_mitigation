#!/usr/bin/env python3
"""Manuscript Fig. 6 — baseline comparison + feature-set ablation (table-figure).

(a) TCCON per-footprint RMSE after correction, per model, split into the pooled
    QF 0+1 set and the decisive near-cloud land tail (<=10 km).  Replaces the
    same-protocol baseline table.
(b) Feature-set ablation of the production deep ensemble: Delta RMSE relative
    to the `full` feature set on the same two slices.

Every number is copied verbatim from the generated comparison reports:

  [A] results/model_comparison/MODEL_COMPARISON_manuscript_DE_XGB_LinReg.md
      (3-model same-protocol table; DE/XGB/LinReg + `before` column)
  [B] results/model_comparison/MODEL_COMPARISON_land_ocean_2026-07-08.md
      (5-model writeup; adds Struct and TabM — DE/XGB/LinReg identical to [A])
  [C] results/model_comparison/deep_ensemble/FEATURESET_ABLATION_QF_2026-07-08.md
      (6-feature-set TCCON ablation, Delta-RMSE-vs-full tables)

Output: manuscript/figures/fig06_baseline_ablation.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workspace"))  # plot_style
from plot_style import (XCO2_BC_LABEL, XCO2_LABEL,
                        apply_manuscript_style, panel_label)

apply_manuscript_style()

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data (ppm).  Slices: "pooled"     = pooled TCCON, QF 0+1 (n_fp = 105,683)
#                      "near-cloud" = land <=10 km, QF 0+1 (n_fp = 75,157)
# ----------------------------------------------------------------------------
# Panel (a): sources [A] section 1 tables (DE, XGB, LinReg, before).
# TabM and Structured DCN EXCLUDED from the main text (user decision
# 2026-07-23) — main-text baselines are DE / XGBoost / Ridge, matching
# Table 1; the 5-model comparison stays in Appendix C.
MODELS = [
    # label,                      pooled, near-cloud land, emphasized
    ("No correction (%s)" % XCO2_BC_LABEL, 3.291, 3.843, False),
    ("Deep ensemble",              1.196, 1.312, True),
    ("XGBoost",                    1.442, 1.607, False),
    ("Ridge linear",               2.242, 2.581, False),
]

# Panel (b): source [C], "Delta RMSE vs full" tables
#   pooled QF 0+1 row and land <=10 km QF 0+1 row.  full reference RMSE:
#   pooled 1.289 ppm / near-cloud land 1.424 ppm ([C] section tables).
ABLATION = [
    # label,                                 pooled Delta, near-cloud Delta
    ("$-$ spectral",                          -0.006, -0.002),
    ("$-$ contamination",                     +0.028, +0.044),
    ("$-$ %s + spectral" % XCO2_LABEL,        +0.686, +0.826),
    ("$-$ %s retrieval" % XCO2_LABEL,         +0.738, +0.887),
    ("$-$ %s + contamination" % XCO2_LABEL,   +0.938, +1.127),
]
FULL_POOLED, FULL_NEAR = 1.289, 1.424  # [C] full-reference RMSE (ppm)

C_POOLED = "#9DC3E0"   # light blue  — pooled TCCON, QF 0+1
C_NEAR = "#0B5D8E"     # dark blue   — near-cloud land tail (decisive slice)
C_BEFORE_P = "#D9D9D9"  # gray pair for the no-correction reference row
C_BEFORE_N = "#8C8C8C"
INK = "#222222"

BAR_H = 0.36


def _pair_bars(ax, ys, pooled, near, colors_p, colors_n, fmt="%.2f",
               label_off=0.03):
    """Two thin horizontal bars per row + value printed at the bar end."""
    b1 = ax.barh(ys + BAR_H / 2, pooled, height=BAR_H, color=colors_p,
                 edgecolor="none", zorder=3)
    b2 = ax.barh(ys - BAR_H / 2, near, height=BAR_H, color=colors_n,
                 edgecolor="none", zorder=3)
    for bars, vals in ((b1, pooled), (b2, near)):
        for rect, v in zip(bars, vals):
            x = rect.get_width()
            off = label_off * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ha, xt = ("left", x + off) if x >= 0 else ("right", x - off)
            ax.text(xt, rect.get_y() + rect.get_height() / 2, fmt % v,
                    va="center", ha=ha, fontsize=7.5, color=INK, zorder=4)
    return b1, b2


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ------------------------------------------------------------------ (a)
    labels = [m[0] for m in MODELS]
    pooled = np.array([m[1] for m in MODELS])
    near = np.array([m[2] for m in MODELS])
    ys = np.arange(len(MODELS))[::-1].astype(float)  # first row on top

    ax_a.set_xlim(0, 4.55)
    colors_p = [C_BEFORE_P if i == 0 else C_POOLED for i in range(len(MODELS))]
    colors_n = [C_BEFORE_N if i == 0 else C_NEAR for i in range(len(MODELS))]
    _pair_bars(ax_a, ys, pooled, near, colors_p, colors_n)

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(labels)
    for tick, m in zip(ax_a.get_yticklabels(), MODELS):
        if m[3]:
            tick.set_fontweight("bold")
    ax_a.set_xlabel("TCCON per-footprint RMSE after correction (ppm)")
    ax_a.xaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_a.spines[spine].set_visible(False)
    ax_a.tick_params(axis="y", length=0)

    ax_a.set_ylim(-0.6, len(MODELS) - 0.4)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_POOLED),
        plt.Rectangle((0, 0), 1, 1, color=C_NEAR),
    ]
    ax_a.legend(handles,
                ["pooled, QF 0+1",
                 "near-cloud land $\\leq$ 10 km"],
                loc="upper center", bbox_to_anchor=(0.5, -0.24),
                ncols=2, columnspacing=1.4, frameon=False,
                handlelength=1.2, handletextpad=0.5, handleheight=1.0,
                borderaxespad=0.0, fontsize=7.5)
    panel_label(ax_a, "(a) baseline comparison")

    # ------------------------------------------------------------------ (b)
    ab_labels = [a[0] for a in ABLATION]
    d_pooled = np.array([a[1] for a in ABLATION])
    d_near = np.array([a[2] for a in ABLATION])
    ys_b = np.arange(len(ABLATION))[::-1].astype(float)

    ax_b.set_xlim(-0.42, 1.45)
    _pair_bars(ax_b, ys_b, d_pooled, d_near,
               [C_POOLED] * len(ABLATION), [C_NEAR] * len(ABLATION),
               fmt="%+.3f", label_off=0.018)
    ax_b.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax_b.set_ylim(-0.6, len(ABLATION) - 0.4 + 0.75)  # headroom for annotation

    ax_b.set_yticks(ys_b)
    ax_b.set_yticklabels(ab_labels)
    ax_b.set_xlabel("$\\Delta$RMSE vs full feature set (ppm)")
    ax_b.xaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    # zero line = full feature set (reference RMSE), leader from the line top
    ax_b.annotate("full feature set\n(RMSE 1.29 / 1.42 ppm)",
                  xy=(0.0, ys_b[0] + 0.42), xycoords="data",
                  xytext=(0.28, ys_b[0] + 0.72), textcoords="data",
                  fontsize=7.5, color=INK, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=INK,
                                  shrinkA=2, shrinkB=1))
    # takeaway, in the free upper-right region
    ax_b.text(0.985, 0.80,
              "spectral & contam.:\nfree to drop\n"
              "%s group:\nload-bearing" % XCO2_LABEL,
              transform=ax_b.transAxes, fontsize=7.5, ha="right",
              va="top", color=INK, style="italic", linespacing=1.3)
    panel_label(ax_b, "(b) deep-ensemble feature-set ablation")

    fig.tight_layout(w_pad=2.2)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig06_baseline_ablation.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
