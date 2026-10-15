#!/usr/bin/env python
"""Overview figures for the Mauceri et al. power-plant overpasses.

Mauceri et al. (2023, AMT 16:1461, Fig. 11) and Mauceri et al. (2025, ESS,
Part 1, Fig. 10) check that their ML bias corrections keep the plumes of
Westar (2015-12-04), Ghent (2015-08-13), Sasan (2014-10-23) and Matimba
(2018-11-18).  This script shows the same overpasses for the production deep
ensemble, from the outputs of the standard plume scripts run with
``--output-dir <plot-base>/plume_preservation_mauceri``:

  mauceri_transects.png  along-track XCO2 (original vs corrected), the model
                         correction mu, and MODIS nearest-cloud distance
  mauceri_summary.png    (a) plume enhancement kept by the correction,
                         (b) matched-control test, (c) band signature

Usage:  PYTHONPATH=src:workspace python workspace/Nassar_plume_analysis/plot_mauceri_cases.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import (apply_manuscript_style, panel_label, XCO2_LABEL,  # noqa: E402
                        XCO2_BC_LABEL, XCO2_DE_LABEL)
from analyze_nassar_plume_preservation import (DEFAULT_PLANTS,  # noqa: E402
                                               DEFAULT_PLOT_BASE, read_plants)
from nassar_plume_transects import load_transect_segment, rolling  # noqa: E402

TRANSECT_CASES = (("westar", "2015-12-04"), ("ghent", "2015-08-13"),
                  ("sasan", "2014-10-23"))
SUMMARY_CASES = TRANSECT_CASES + (("vindhyachal", "2014-10-23"),)
PLUME_KM, BG_LO, BG_HI = 15.0, 30.0, 100.0     # nassar_k1_contrast defaults
K1_XLIM = 0.04                                  # panel (c) range; larger bars are clipped


def case_label(plants, plant_id, date):
    return f"{plants[plant_id]['source_name']}\n{date}"


def plot_transects(plants, plot_base, k1, outdir, max_km=100.0):
    fig, axes = plt.subplots(3, len(TRANSECT_CASES), figsize=(15, 9),
                             sharex="col",
                             gridspec_kw={"height_ratios": [2, 1.2, 1]})
    for j, pair in enumerate(TRANSECT_CASES):
        loaded = load_transect_segment(pair, plants, plot_base, max_km=max_km)
        if loaded is None:
            continue
        seg, src, min_dist = loaded
        x = seg["x_km"].to_numpy()

        ax = axes[0, j]
        for col, c, lbl in (("xco2_bc", "C0", XCO2_BC_LABEL),
                            ("deep_ensemble_corrected_xco2", "C3", XCO2_DE_LABEL)):
            ax.plot(x, seg[col], ".", color=c, ms=2.5, alpha=0.3)
            xs, med = rolling(x, seg[col].to_numpy())
            ax.plot(xs, med, "-", color=c, lw=1.8, label=f"{lbl} (5 km median)")
        lo, hi = np.nanpercentile(seg["xco2_bc"], [1, 99.5])
        ax.set_ylim(lo - 1.0, hi + 1.0)
        ax.set_title(f"{src['source_name']}, {pair[1]}\n"
                     f"closest footprint {min_dist:.1f} km", fontsize=11)
        r = k1[(k1.plant_id == pair[0]) & (k1.date == pair[1])]
        if len(r):
            r = r.iloc[0]
            kept = 1.0 - r.d_mu_ppm / r.d_xco2_ppm
            ax.text(0.02, 0.97,
                    f"plume rise, original  {r.d_xco2_ppm:+.2f} ppm\n"
                    f"plume rise, corrected {r.d_xco2_ppm - r.d_mu_ppm:+.2f} ppm\n"
                    f"kept {100 * kept:.0f} %",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        if j == 0:
            ax.set_ylabel(f"{XCO2_LABEL} (ppm)")
            ax.legend(loc="lower right", fontsize=8)

        ax = axes[1, j]
        ax.plot(x, seg["pred_anomaly"], ".", color="C2", ms=2.5, alpha=0.3)
        xs, med = rolling(x, seg["pred_anomaly"].to_numpy())
        ax.plot(xs, med, "-", color="C2", lw=1.8)
        ax.axhline(0, color="gray", lw=0.8)
        lo, hi = np.nanpercentile(seg["pred_anomaly"], [1, 99.5])
        ax.set_ylim(min(lo, -0.5) - 0.3, max(hi, 0.5) + 0.3)
        if j == 0:
            ax.set_ylabel("correction μ (ppm)")

        ax = axes[2, j]
        ax.plot(x, seg["cld_dist_km"], ".", color="C7", ms=2.5, alpha=0.5)
        ax.set_xlabel("along-track distance from closest approach (km)")
        if j == 0:
            ax.set_ylabel("nearest cloud (km)")

        for i in range(3):
            a = axes[i, j]
            a.axvspan(-PLUME_KM, PLUME_KM, color="gold", alpha=0.18, lw=0)
            for s in (-1, 1):
                a.axvspan(s * BG_LO, s * BG_HI, color="0.85", alpha=0.25, lw=0)
            a.axvline(0, color="k", lw=0.9, ls="--")
            a.grid(alpha=0.3)
            a.set_xlim(-max_km, max_km)
            panel_label(a, f"({'abc'[i]}{j + 1})")

    fig.suptitle("Mauceri et al. power-plant overpasses, production deep ensemble   "
                 f"(yellow: plume window ±{PLUME_KM:.0f} km, grey: background "
                 f"{BG_LO:.0f}–{BG_HI:.0f} km)", fontsize=11)
    fig.tight_layout()
    out = outdir / "mauceri_transects.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_summary(plants, k1, ctrl, ctrl_win, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    cases = [p for p in SUMMARY_CASES
             if ((k1.plant_id == p[0]) & (k1.date == p[1])).any()]
    labels = [case_label(plants, *p) for p in cases]
    y = np.arange(len(cases))[::-1]

    # (a) plume enhancement before vs after
    ax = axes[0]
    for yi, (pid, date) in zip(y, cases):
        r = k1[(k1.plant_id == pid) & (k1.date == date)].iloc[0]
        before, after = r.d_xco2_ppm, r.d_xco2_ppm - r.d_mu_ppm
        ax.plot([before, after], [yi, yi], "-", color="0.6", lw=2, zorder=1)
        ax.errorbar(before, yi, xerr=r.d_xco2_se, fmt="o", color="C0", ms=8,
                    capsize=3, zorder=2, label=XCO2_BC_LABEL if yi == y[0] else None)
        ax.errorbar(after, yi, xerr=np.hypot(r.d_xco2_se, r.d_mu_se), fmt="o",
                    color="C3", ms=8, capsize=3, zorder=2,
                    label=XCO2_DE_LABEL if yi == y[0] else None)
        ax.text(1.02, yi, f"{100 * after / before:.0f} %\nkept", va="center",
                fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"plume rise (ppm), ≤{PLUME_KM:.0f} km minus {BG_LO:.0f}–{BG_HI:.0f} km")
    ax.set_xlim(left=-0.3, right=ax.get_xlim()[1] + 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    ax.set_title("Is the plume kept?", fontsize=11)
    panel_label(ax, "(a)")

    # (b) matched-control test
    ax = axes[1]
    for yi, (pid, date) in zip(y, cases):
        w = ctrl_win[(ctrl_win.plant_id == pid) & (ctrl_win.date == date)]
        r = ctrl[(ctrl.plant_id == pid) & (ctrl.date == date)]
        if not len(r):
            continue
        r = r.iloc[0]
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(w))
        ax.plot(w.ratio_corr_over_bc, yi + jit, ".", color="0.6", ms=3, alpha=0.6,
                label="control windows, same orbit" if yi == y[0] else None)
        ax.plot([r.control_ratio_p05, r.control_ratio_p95], [yi, yi], "-",
                color="0.3", lw=3, alpha=0.6,
                label="controls, 5–95 %" if yi == y[0] else None)
        ax.plot(r.plant_ratio_corr_over_bc, yi, "*", color="C1", ms=16, mec="k",
                zorder=3, label="plant window" if yi == y[0] else None)
        ax.text(1.02, yi, f"p{r.plant_ratio_pctile_in_controls:.0f}",
                va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("spread kept by the correction (corrected / original p95–p05)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    ax.set_title("Does the plant window lose more than controls?\n"
                 "(star inside the bar = no plume-specific removal)", fontsize=11)
    panel_label(ax, "(b)")

    # (c) band signature
    ax = axes[2]
    bands = (("o2a", "O$_2$ A-band", "C0"), ("wco2", "weak CO$_2$", "C2"),
             ("sco2", "strong CO$_2$", "C3"))
    h = 0.25
    for yi, (pid, date) in zip(y, cases):
        r = k1[(k1.plant_id == pid) & (k1.date == date)].iloc[0]
        for b, (key, name, c) in enumerate(bands):
            yy = yi + (1 - b) * h
            val = r[f"d_{key}_k1"]
            shown = float(np.clip(val, -K1_XLIM, K1_XLIM))
            ax.barh(yy, shown, height=h * 0.9, color=c, alpha=0.75,
                    xerr=None if shown != val else r[f"d_{key}_k1_se"], capsize=2,
                    label=name if yi == y[0] else None)
            if shown != val:   # clipped bar: print the value at the axis edge
                ax.text(shown + (0.001 if val < 0 else -0.001), yy,
                        (f"\u2190 {val:+.3f} ± {r[f'd_{key}_k1_se']:.3f}" if val < 0
                         else f"{val:+.3f} ± {r[f'd_{key}_k1_se']:.3f} \u2192"),
                        va="center", ha="left" if val < 0 else "right",
                        fontsize=8, color="w", fontweight="bold")
            exp = 0.0 if key == "o2a" else r[f"d_{key}_k1_expected"]
            ax.plot([exp, exp], [yy - h * 0.45, yy + h * 0.45], "k-", lw=2,
                    label="expected for the CO$_2$ rise" if (yi == y[0] and b == 1) else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-K1_XLIM, K1_XLIM)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"plume-window change in $\langle l'\rangle$ (relative path length)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    ax.set_title("Band signature: real CO$_2$ moves the CO$_2$ bands only,\n"
                 "clouds move all three bands", fontsize=11)
    panel_label(ax, "(c)")

    for a in axes:
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = outdir / "mauceri_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--results-dir", type=Path,
                    default=DEFAULT_PLOT_BASE / "plume_preservation_mauceri")
    args = ap.parse_args()

    apply_manuscript_style()
    plants = read_plants(args.plants)
    k1 = pd.read_csv(args.results_dir / "nassar_k1_contrast.csv")
    ctrl = pd.read_csv(args.results_dir / "nassar_control_null.csv")
    ctrl_win = pd.read_csv(args.results_dir / "nassar_control_null_windows.csv")
    plot_transects(plants, args.plot_base, k1, args.results_dir)
    plot_summary(plants, k1, ctrl, ctrl_win, args.results_dir)


if __name__ == "__main__":
    main()
