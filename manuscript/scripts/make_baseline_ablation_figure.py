#!/usr/bin/env python3
"""Manuscript Fig. 6 — baseline comparison + feature-set ablation (table-figure).

(a) TCCON per-footprint RMSE after correction, per model, on three slices:
    the pooled QF 0+1 set and the near-cloud tails of BOTH surfaces at their
    production target radii (ocean <=5 km, land <=15 km; 2026-07-23 —
    replaces the single land <=10 km tail).  Replaces the same-protocol
    baseline table.
(b) Feature-set ablation of the production deep ensemble: Delta RMSE relative
    to the `full` feature set on the same three slices.

Every number is read from the per-surface-edge report CSVs
(tccon_metrics_ak_cldo5_r100km.csv / _cldl15_, same trees as Tables 1-2 via
make_manuscript_tables) so the figure cannot drift from the tables.

TabM and Structured DCN EXCLUDED from the main text (user decision
2026-07-23) — main-text baselines are DE / XGBoost / Ridge, matching
Table 1; the 5-model comparison stays in Appendix C.

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

sys.path.insert(0, str(REPO / "manuscript" / "scripts"))
from make_manuscript_tables import (ABL_VARIANTS, BASE, RAW_TREE,  # noqa: F401
                                    TREES, load_metrics, srow)

# The three plotted slices: (qf, surface, cld_group, edition)
SLICE_POOLED = ("all", "all", "all", "o5")
SLICE_OCEAN = ("all", "ocean", "0–5 km", "o5")
SLICE_LAND = ("all", "land", "0–15 km", "l15")


def _three(mets):
    """(pooled, ocean<=5, land<=15) rmse_after for one tree's metrics."""
    return tuple(srow(mets[ed], qf, sfc, cld).rmse_after
                 for qf, sfc, cld, ed in (SLICE_POOLED, SLICE_OCEAN,
                                          SLICE_LAND))


def _load_data():
    mets = {m: load_metrics(p) for m, p in TREES.items()}
    de = mets["DE"]
    before = tuple(srow(de[ed], qf, sfc, cld).rmse_before
                   for qf, sfc, cld, ed in (SLICE_POOLED, SLICE_OCEAN,
                                            SLICE_LAND))
    models = [("No correction (%s)" % XCO2_BC_LABEL, *before, False),
              ("Deep ensemble", *_three(mets["DE"]), True),
              ("XGBoost", *_three(mets["XGB"]), False),
              ("Ridge linear", *_three(mets["Ridge"]), False)]
    full = _three(mets["DE"])
    # labels + order match Table 2's column heads: single drops first,
    # then the double drops (2026-07-23: the old "- XCO2 + spectral"
    # phrasing read as add-spectral and hid the single -xco2 row)
    abl_labels = {
        "no_spec": "$-$ spectral",
        "no_contam": "$-$ contamination",
        "no_xco2": "$-$ %s" % XCO2_LABEL,
        "no_xco2_and_spec": "$-$ %s $-$ spectral" % XCO2_LABEL,
        "no_contam_and_xco2": "$-$ %s $-$ contamination" % XCO2_LABEL,
    }
    ablation = []
    for v in ("no_spec", "no_contam", "no_xco2", "no_xco2_and_spec",
              "no_contam_and_xco2"):
        vals = _three(load_metrics(BASE / "deep_ensemble"
                                   / f"de_prof_mix_{v}"))
        ablation.append((abl_labels[v],
                         *(a - f for a, f in zip(vals, full))))
    return models, ablation, full


C_POOLED = "#9DC3E0"   # light blue  — pooled TCCON, QF 0+1
C_OCEAN = "#009E73"    # green       — near-cloud ocean (<=5 km)
C_NEAR = "#0B5D8E"     # dark blue   — near-cloud land tail (decisive slice)
C_BEFORE = ("#E3E3E3", "#B5B5B5", "#8C8C8C")   # gray triple, no-correction row
INK = "#222222"

BAR_H = 0.26


def _triple_bars(ax, ys, triples, row_colors, fmt="%.2f", label_off=0.03):
    """Three thin horizontal bars per row + value printed at the bar end."""
    out = []
    for k in range(3):
        vals = [t[k] for t in triples]
        colors = [c[k] for c in row_colors]
        bars = ax.barh(ys + (1 - k) * BAR_H, vals, height=BAR_H,
                       color=colors, edgecolor="none", zorder=3)
        out.append(bars)
        for rect, v in zip(bars, vals):
            x = rect.get_width()
            off = label_off * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ha, xt = ("left", x + off) if x >= 0 else ("right", x - off)
            ax.text(xt, rect.get_y() + rect.get_height() / 2, fmt % v,
                    va="center", ha=ha, fontsize=6.6, color=INK, zorder=4)
    return out


def main() -> None:
    models, ablation, full = _load_data()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ------------------------------------------------------------------ (a)
    labels = [m[0] for m in models]
    triples = [m[1:4] for m in models]
    ys = np.arange(len(models))[::-1].astype(float)  # first row on top

    ax_a.set_xlim(0, 4.75)
    row_colors = [(C_BEFORE if i == 0 else (C_POOLED, C_OCEAN, C_NEAR))
                  for i in range(len(models))]
    _triple_bars(ax_a, ys, triples, row_colors)

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(labels)
    for tick, m in zip(ax_a.get_yticklabels(), models):
        if m[4]:
            tick.set_fontweight("bold")
    ax_a.set_xlabel("TCCON per-footprint RMSE after correction (ppm)")
    ax_a.xaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_a.spines[spine].set_visible(False)
    ax_a.tick_params(axis="y", length=0)

    ax_a.set_ylim(-0.7, len(models) - 0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (C_POOLED, C_OCEAN, C_NEAR)]
    leg_labels = ["pooled, QF 0+1",
                  "near-cloud ocean $\\leq$ 5 km",
                  "near-cloud land $\\leq$ 15 km"]
    ax_a.legend(handles, leg_labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.24),
                ncols=2, columnspacing=1.2, frameon=False,
                handlelength=1.2, handletextpad=0.5, handleheight=1.0,
                borderaxespad=0.0, fontsize=7)
    panel_label(ax_a, "(a) baseline comparison")

    # ------------------------------------------------------------------ (b)
    ab_labels = [a[0] for a in ablation]
    d_triples = [a[1:4] for a in ablation]
    ys_b = np.arange(len(ablation))[::-1].astype(float)

    ax_b.set_xlim(-0.42, 1.55)
    _triple_bars(ax_b, ys_b, d_triples,
                 [(C_POOLED, C_OCEAN, C_NEAR)] * len(ablation),
                 fmt="%+.3f", label_off=0.018)
    ax_b.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax_b.set_ylim(-0.7, len(ablation) - 0.3 + 0.85)  # headroom for annotation

    ax_b.set_yticks(ys_b)
    ax_b.set_yticklabels(ab_labels)
    ax_b.set_xlabel("$\\Delta$RMSE vs full feature set (ppm)")
    ax_b.xaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    # zero line = full feature set (reference RMSE), leader from the line top
    ax_b.annotate("full feature set (RMSE\n%.2f / %.2f / %.2f ppm)" % full,
                  xy=(0.0, ys_b[0] + 0.55), xycoords="data",
                  xytext=(0.30, ys_b[0] + 0.82), textcoords="data",
                  fontsize=7, color=INK, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=INK,
                                  shrinkA=2, shrinkB=1))
    panel_label(ax_b, "(b) deep-ensemble feature-set ablation")

    fig.tight_layout(w_pad=2.2)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig06_baseline_ablation.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
