#!/usr/bin/env python3
"""Manuscript Fig. 9b — plume-vs-cloud k1 fingerprint contrast.

Companion panel to the plume-preservation transect (Fig. 9a, Westar
2023-06-26).  For the two cases flagged by the matched control-region null
(Lipetsk 2015-08-01, Westar 2023-03-13) and the two clean clear-sky controls
(Westar 2023-06-26, Ghent 2024-04-15), shows the measured per-band k1 shift
inside the plume/removed-structure window relative to the along-track
background (bars, +/-1 SE), against the signature a REAL CO2 enhancement
would produce (black tick markers): the CO2 bands move by prior-tau scaling
k1 * dXCO2 / XCO2 (~ +0.0026 per ppm; per-case `*_expected` columns of the
source CSV) while the O2 A-band is exactly flat (O2 is insensitive to CO2).
A cloud artifact instead moves k1 in all bands together — in both flagged
cases o2a_k1 shifts as much as or more than wco2_k1, so the structure the
correction removes there is cloud-induced, not a plume.

Data (read directly, one row per case):
  results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/
    nassar_plumes/plume_preservation/nassar_k1_contrast.csv
Context: log/SPEC_EMPHASIS_STATUS_2026-07-08.md section 2 (item 3).

Output: results/figures/manuscript/fig09b_k1_contrast.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # workspace/
from plot_style import MEAN_L_LABEL, apply_manuscript_style, panel_label

apply_manuscript_style()

REPO = Path(__file__).resolve().parents[2]
CSV = (REPO / "results" / "model_comparison" / "deep_ensemble"
       / "de_beta_nll_prof_reg_o05l15_m5" / "nassar_plumes"
       / "plume_preservation" / "nassar_k1_contrast.csv")
OUT_DIR = REPO / "results" / "figures" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cases: the two control-null flagged windows (verdict: cloud) and the two
# clean clear-sky controls.  (plant_id, date, display name)
CASES = [
    ("lipetsk", "2015-08-01", "Lipetsk"),
    ("westar", "2023-03-13", "Westar"),
    ("westar", "2023-06-26", "Westar"),
    ("ghent", "2024-04-15", "Ghent"),
]
N_FLAGGED = 2  # first N_FLAGGED cases are the flagged (cloud) windows

# Bands: (measured col, SE col, expected col or None -> exactly 0, label, color)
# Okabe-Ito blue / orange / vermillion, ordered by band wavelength.
BANDS = [
    ("d_o2a_k1", "d_o2a_k1_se", None, "O$_2$A", "#0072B2"),
    ("d_wco2_k1", "d_wco2_k1_se", "d_wco2_k1_expected", "WCO$_2$", "#E69F00"),
    ("d_sco2_k1", "d_sco2_k1_se", "d_sco2_k1_expected", "SCO$_2$", "#D55E00"),
]

INK = "#222222"
BAR_W = 0.24
OFFSETS = (-0.26, 0.0, 0.26)


def main() -> None:
    df = pd.read_csv(CSV)
    rows = []
    for plant, date, _name in CASES:
        sel = df[(df.plant_id == plant) & (df.date == date)]
        if len(sel) != 1:
            raise SystemExit(f"expected 1 row for {plant} {date}, got {len(sel)}")
        rows.append(sel.iloc[0])

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    xs = np.arange(len(CASES), dtype=float)

    # flagged-group backdrop
    ax.axvspan(-0.5, N_FLAGGED - 0.5, color="0.94", zorder=0)
    ax.axvline(N_FLAGGED - 0.5, color="0.75", lw=0.8, zorder=1)

    print("plotted values (cross-check against nassar_k1_contrast.csv):")
    for off, (mcol, scol, ecol, blabel, color) in zip(OFFSETS, BANDS):
        vals = np.array([r[mcol] for r in rows])
        errs = np.array([r[scol] for r in rows])
        exps = np.array([0.0 if ecol is None else r[ecol] for r in rows])
        ax.bar(xs + off, vals, width=BAR_W, color=color, zorder=3,
               yerr=errs, error_kw=dict(ecolor="0.25", lw=0.8, capsize=2,
                                        capthick=0.8, zorder=4))
        # expected real-plume signature: short black tick per bar
        ax.plot(xs + off, exps, ls="none", marker="_", ms=11, mew=1.8,
                color=INK, zorder=5)
        for (plant, date, _n), v, e, x in zip(CASES, vals, errs, exps):
            print(f"  {plant:9s} {date}  {mcol}={v:+.4f} +/-{e:.4f}"
                  f"  expected={x:+.4f}")

    ax.axhline(0.0, color=INK, lw=0.8, zorder=2)

    # x tick labels: plant, date, median nearest-cloud distance in the window
    labels = []
    for (plant, date, name), r in zip(CASES, rows):
        cld = r["median_cld_plume"]
        cld_txt = f"{cld:.1f}" if cld < 10 else f"{cld:.0f}"
        labels.append(f"{name}\n{date}\ncloud {cld_txt} km")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(CASES) - 0.5)

    ax.set_ylabel(f"$\\Delta${MEAN_L_LABEL} (plume window $-$ background)")
    ax.set_ylim(-0.038, 0.064)
    ax.yaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # group headers just inside the top of the axes
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text((N_FLAGGED - 1) / 2.0, 0.985, "flagged windows",
            transform=blend, ha="center", va="top", fontsize=9,
            fontweight="bold", color=INK)
    ax.text((N_FLAGGED + len(CASES) - 1) / 2.0, 0.985, "clear-sky controls",
            transform=blend, ha="center", va="top", fontsize=9,
            fontweight="bold", color=INK)
    ax.text(1.0, 0.885,
            "O$_2$A shifts as much as the CO$_2$ bands\n"
            "$\\rightarrow$ cloud artifact, not a plume",
            transform=blend, ha="center", va="top", fontsize=8,
            style="italic", color=INK, multialignment="center")

    handles = [Rectangle((0, 0), 1, 1, color=c) for *_, c in BANDS]
    handles.append(Line2D([], [], ls="none", marker="_", ms=11, mew=1.8,
                          color=INK))
    leg_labels = [b[3] for b in BANDS]
    leg_labels.append("expected for a real CO$_2$ plume\n"
                      "($k_1\\Delta X_{\\mathrm{CO2}}/X_{\\mathrm{CO2}}$;"
                      " O$_2$A $= 0$)")
    ax.legend(handles, leg_labels, loc="lower right", frameon=False,
              handlelength=1.2, handleheight=1.0, borderaxespad=0.2,
              ncols=2, columnspacing=1.2)

    panel_label(ax, "(b) band-resolved $k_1$ fingerprint")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig09b_k1_contrast.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
