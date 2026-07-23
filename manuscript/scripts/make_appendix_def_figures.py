#!/usr/bin/env python3
"""Generate the Appendix D/E/F display items that have local data
(2026-07-23p; the Appendix C set lives in make_appendix_c_figures.py).

Items (all read the PRODUCTION fold-PCA tag):

  figD1  coincidence-criteria sensitivity — station-day mean |bias| and
         footprint RMSE before/after for the 3x3 radius x window sweep
         (atrain/coincidence_sensitivity/coincidence_sensitivity.csv,
         regenerated on the foldpca tag 2026-07-23).
  figD5  random-effects residual forest — per-station-day corrected-minus-
         TCCON D with 95% CI (atrain/tccon_uncertainty_r100km.csv) and the
         DerSimonian-Laird pooled mean +- CI with between-case tau
         (recomputed here; cross-checked against the md report).
  figE1  smoother-null windows not shown in main-text Fig. 10: the +-10 s
         and +-100 s running-mean arms beside the deep ensemble, same
         2-panel construction (scatter collapse / |case bias|) as Fig. 10,
         from atrain/smoother_null/tccon_comparison_*.csv.
  figF2  all Nassar power-plant overpass transects (10 renderable cases;
         Sasan 2023-06-26 has no in-window footprints — missing wind
         direction/cloud distance — and Matimba 2024-04-15 no overpass
         segment) as a 3-column grid of the per-case transect figures
         under nassar_plumes/plume_preservation/transects/.
  figF3  band-resolved k1 contrast for ALL six audited windows (the two
         real-plume windows Kozienice 2021-09-06 / Taean 2021-09-08 join
         the two flagged and two clear-sky windows of main-text Fig. 12b),
         from nassar_plumes/plume_preservation/nassar_k1_contrast.csv.

Usage: python make_appendix_def_figures.py [--only figD5 figF3 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style + smoother_null_figure
from plot_style import (MEAN_L_LABEL, XCO2_LABEL, apply_manuscript_style,
                        panel_label)
from smoother_null_figure import load_cases

apply_manuscript_style()

TAG = (REPO / "results" / "model_comparison" / "deep_ensemble"
       / "de_beta_nll_prof_reg_foldpca_o05l15_m5")
ATRAIN = TAG / "atrain"
OUT_DIR = REPO / "manuscript" / "figures"

INK = "#222222"
C_BEFORE = "#B5B5B5"
C_AFTER = "#0072B2"


def _save(fig, stem: str, png_only: bool = False) -> None:
    for ext in (("png",) if png_only else ("png", "pdf")):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────── figD1 ──
def make_d1() -> None:
    d = pd.read_csv(ATRAIN / "coincidence_sensitivity"
                    / "coincidence_sensitivity.csv")
    d = d.sort_values(["radius_km", "window_min"]).reset_index(drop=True)
    # rows grouped by radius (extra gap between groups), top-down
    ys, y, labels = [], 0.0, []
    for i, r in d.iterrows():
        if i and r.radius_km != d.radius_km.iloc[i - 1]:
            y += 0.7
        ys.append(y)
        labels.append(f"{r.radius_km:.0f} km, ±{r.window_min:.0f} min"
                      f"  (n={r.n_cases:.0f})")
        y += 1.0
    ys = np.max(ys) - np.asarray(ys)   # first row on top

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    panels = [("absbias_before", "absbias_after",
               "station-day mean |bias| (ppm)", "(a)"),
              ("rmse_before", "rmse_after",
               "footprint RMSE (ppm)", "(b)")]
    for ax, (cb, ca, xlabel, letter) in zip(axes, panels):
        for yi, b, a in zip(ys, d[cb], d[ca]):
            ax.plot([a, b], [yi, yi], "-", color="lightgray", lw=1.4,
                    zorder=1)
        ax.plot(d[cb], ys, "o", ms=5.5, mfc="white", mec=C_BEFORE, mew=1.4,
                label="before", zorder=2)
        ax.plot(d[ca], ys, "o", ms=5.5, color=C_AFTER, mec="black",
                mew=0.4, label="after", zorder=3)
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, None)
        panel_label(ax, letter)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=7)
    # (b)'s lower-left quadrant is empty (all RMSE >= 1 ppm); (a)'s is not
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "figD1_coincidence_sensitivity")


# ────────────────────────────────────────────────────────────── figD5 ──
def _dl_pool(D: np.ndarray, u: np.ndarray):
    """DerSimonian-Laird random-effects pool of per-case D +- u."""
    w = 1.0 / u**2
    mu_fe = np.sum(w * D) / np.sum(w)
    q = float(np.sum(w * (D - mu_fe) ** 2))
    k = len(D)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1.0 / (u**2 + tau2)
    mu = float(np.sum(ws * D) / np.sum(ws))
    se = float(1.0 / np.sqrt(np.sum(ws)))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return mu, se, np.sqrt(tau2), i2


def make_d5() -> None:
    d = pd.read_csv(ATRAIN / "tccon_uncertainty_r100km.csv")
    d = d[(d.surface == "all") & (d.qf_group == "all")].copy()
    d = d.sort_values("D").reset_index(drop=True)
    mu, se, tau, i2 = _dl_pool(d.D.to_numpy(float), d.u_D.to_numpy(float))
    print(f"DL pool: mu = {mu:+.2f} ± {se:.2f} ppm "
          f"(95% CI [{mu - 1.96 * se:.2f}, {mu + 1.96 * se:.2f}]), "
          f"tau = {tau:.2f} ppm, I2 = {100 * i2:.0f}%  "
          f"(cross-check tccon_uncertainty_r100km.md)")

    y = np.arange(len(d))
    sig = d.significant.astype(bool).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    # pooled mean + 95% CI band, and the +-tau between-case spread band
    ax.axvspan(mu - tau, mu + tau, color="#0072B2", alpha=0.08, zorder=0,
               label=f"pooled μ ± τ ({tau:.2f} ppm)")
    ax.axvspan(mu - 1.96 * se, mu + 1.96 * se, color="#0072B2", alpha=0.25,
               zorder=1, label="pooled μ, 95% CI")
    ax.axvline(mu, color="#0072B2", lw=1.2, zorder=2)
    ax.axvline(0, color="k", lw=1.0, zorder=2)
    for m, color, lbl in ((~sig, "0.45", "case D ± 95% CI"),
                          (sig, "#D55E00",
                           "95% CI excludes 0")):
        ax.errorbar(d.D[m], y[m], xerr=1.96 * d.u_D[m], fmt="o", ms=4.2,
                    color=color, ecolor=color, elinewidth=0.8, capsize=1.8,
                    capthick=0.8, ls="none", label=lbl, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.site} {r.date}" for r in d.itertuples()],
                       fontsize=6)
    ax.set_ylim(-1, len(d))
    ax.set_ylabel("station-day (sorted by D)")
    ax.set_xlabel(f"D = corrected {XCO2_LABEL} − AK-harmonized TCCON (ppm)")
    ax.text(0.015, 0.99,
            f"DerSimonian–Laird: μ = {mu:+.2f} ± {se:.2f} ppm\n"
            f"between-case τ = {tau:.2f} ppm,  I$^2$ = {100 * i2:.0f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    _save(fig, "figD5_uncertainty_forest")


# ────────────────────────────────────────────────────────────── figE1 ──
def make_e1() -> None:
    rd = ATRAIN / "smoother_null"
    de = load_cases(rd / "tccon_comparison_de_r100km.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.4))
    letters = iter("abcd")
    for row, w in zip(axes, (10, 100)):
        sm = load_cases(rd / f"tccon_comparison_smoother_w{w}_r100km.csv")
        common = de.index.intersection(sm.index)
        dd, ss = de.loc[common], sm.loc[common]

        ax = row[0]   # footprint scatter collapse
        lim = max(dd.orig_sd.max(), dd.corr_sd.max(), ss.corr_sd.max()) * 1.06
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.7, zorder=1)
        ax.scatter(dd.orig_sd, dd.corr_sd, s=14, facecolors="none",
                   edgecolors="#0072B2", linewidths=0.8,
                   label="deep ensemble", zorder=3)
        ax.scatter(ss.orig_sd, ss.corr_sd, s=12, marker="^",
                   facecolors="none", edgecolors="#D55E00", linewidths=0.8,
                   label=f"running mean (±{w} s)", zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"footprint {XCO2_LABEL} scatter before (ppm)")
        ax.set_ylabel(f"footprint {XCO2_LABEL} scatter after (ppm)")
        ax.text(0.03, 0.97,
                f"mean {dd.orig_sd.mean():.2f} → {dd.corr_sd.mean():.2f} (DE)\n"
                f"mean {ss.orig_sd.mean():.2f} → {ss.corr_sd.mean():.2f}"
                " (smoother)",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        panel_label(ax, f"({next(letters)})")

        ax = row[1]   # |case bias to TCCON|
        lim = max(dd.bias_before.abs().max(), dd.bias_after.abs().max(),
                  ss.bias_after.abs().max()) * 1.06
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.7, zorder=1)
        ax.scatter(dd.bias_before.abs(), dd.bias_after.abs(), s=14,
                   facecolors="none", edgecolors="#0072B2", linewidths=0.8,
                   zorder=3)
        ax.scatter(ss.bias_before.abs(), ss.bias_after.abs(), s=12,
                   marker="^", facecolors="none", edgecolors="#D55E00",
                   linewidths=0.8, zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("|case bias to TCCON| before (ppm)")
        ax.set_ylabel("|case bias to TCCON| after (ppm)")
        ax.text(0.03, 0.97,
                f"mean {dd.bias_before.abs().mean():.2f} → "
                f"{dd.bias_after.abs().mean():.2f} (DE)\n"
                f"mean {ss.bias_before.abs().mean():.2f} → "
                f"{ss.bias_after.abs().mean():.2f} (smoother)",
                transform=ax.transAxes, fontsize=7, va="top")
        panel_label(ax, f"({next(letters)})")
    fig.tight_layout()
    _save(fig, "figE1_smoother_windows")


# ────────────────────────────────────────────────────────────── figF2 ──
F2_CASES = [   # chronological; the 10 renderable audited overpasses
    ("colstrip", "2015-08-01"), ("lipetsk", "2015-08-01"),
    ("comanche", "2020-09-06"), ("kozienice", "2021-09-06"),
    ("taean", "2021-09-08"), ("westar", "2023-03-13"),
    ("westar", "2023-06-26"), ("vindhyachal", "2023-06-26"),
    ("ghent", "2024-04-15"), ("kozienice", "2024-06-26"),
]


def make_f2() -> None:
    tdir = TAG / "nassar_plumes" / "plume_preservation" / "transects"
    ncol = 3
    nrow = int(np.ceil(len(F2_CASES) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(7.2, 7.2 / ncol * nrow * 1.0))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (plant, date) in zip(axes.flat, F2_CASES):
        img = mpimg.imread(tdir / f"nassar_transect_{plant}_{date}.png")
        ax.imshow(img, interpolation="antialiased")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.02, hspace=0.02)
    _save(fig, "figF2_nassar_transects", png_only=True)


# ────────────────────────────────────────────────────────────── figF3 ──
F3_GROUPS = [   # (group header, [(plant_id, date, display name), ...])
    ("plume windows", [("kozienice", "2021-09-06", "Kozienice"),
                       ("taean", "2021-09-08", "Taean")]),
    ("flagged windows", [("lipetsk", "2015-08-01", "Lipetsk"),
                         ("westar", "2023-03-13", "Westar")]),
    ("clear-sky controls", [("westar", "2023-06-26", "Westar"),
                            ("ghent", "2024-04-15", "Ghent")]),
]
BANDS = [
    ("d_o2a_k1", "d_o2a_k1_se", None, "O$_2$A", "#0072B2"),
    ("d_wco2_k1", "d_wco2_k1_se", "d_wco2_k1_expected", "WCO$_2$", "#E69F00"),
    ("d_sco2_k1", "d_sco2_k1_se", "d_sco2_k1_expected", "SCO$_2$", "#D55E00"),
]


def make_f3() -> None:
    df = pd.read_csv(TAG / "nassar_plumes" / "plume_preservation"
                     / "nassar_k1_contrast.csv")
    cases, edges = [], []
    for header, members in F3_GROUPS:
        edges.append((header, len(cases), len(cases) + len(members)))
        cases.extend(members)
    rows = []
    for plant, date, _n in cases:
        sel = df[(df.plant_id == plant) & (df.date == date)]
        if len(sel) != 1:
            raise SystemExit(f"expected 1 row for {plant} {date}, got {len(sel)}")
        rows.append(sel.iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(cases), dtype=float)
    # alternate group backdrops + separators
    for gi, (_h, i0, i1) in enumerate(edges):
        if gi % 2 == 0:
            ax.axvspan(i0 - 0.5, i1 - 0.5, color="0.94", zorder=0)
        if gi:
            ax.axvline(i0 - 0.5, color="0.75", lw=0.8, zorder=1)

    for off, (mcol, scol, ecol, _blabel, color) in zip((-0.26, 0.0, 0.26),
                                                       BANDS):
        vals = np.array([r[mcol] for r in rows])
        errs = np.array([r[scol] for r in rows])
        exps = np.array([0.0 if ecol is None else r[ecol] for r in rows])
        ax.bar(xs + off, vals, width=0.24, color=color, zorder=3,
               yerr=errs, error_kw=dict(ecolor="0.25", lw=0.8, capsize=2,
                                        capthick=0.8, zorder=4))
        ax.plot(xs + off, exps, ls="none", marker="_", ms=11, mew=1.8,
                color=INK, zorder=5)
    ax.axhline(0.0, color=INK, lw=0.8, zorder=2)

    labels = []
    for (_p, date, name), r in zip(cases, rows):
        cld = r["median_cld_plume"]
        cld_txt = f"{cld:.1f}" if cld < 10 else f"{cld:.0f}"
        labels.append(f"{name}\n{date}\ncloud {cld_txt} km")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(cases) - 0.5)
    ax.set_ylabel(f"$\\Delta${MEAN_L_LABEL} (plume window $-$ background)")
    ax.yaxis.grid(True, color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for _h, i0, i1 in edges:
        ax.text((i0 + i1 - 1) / 2.0, 0.985, _h, transform=blend,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=INK)

    handles = [Rectangle((0, 0), 1, 1, color=c) for *_, c in BANDS]
    handles.append(Line2D([], [], ls="none", marker="_", ms=11, mew=1.8,
                          color=INK))
    leg_labels = [b[3] for b in BANDS]
    leg_labels.append("expected for a real CO$_2$ plume\n"
                      f"({MEAN_L_LABEL}$\\,\\Delta X_{{\\mathrm{{CO2}}}}"
                      f"/X_{{\\mathrm{{CO2}}}}$; O$_2$A $= 0$)")
    # legend ABOVE the axes — every in-frame corner holds bars or headers
    fig.legend(handles, leg_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=7,
               handlelength=1.2, handleheight=1.0, ncols=4,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "figF3_k1_contrast_full")


ALL_ITEMS = {
    "figD1": make_d1,
    "figD5": make_d5,
    "figE1": make_e1,
    "figF2": make_f2,
    "figF3": make_f3,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", choices=sorted(ALL_ITEMS))
    args = ap.parse_args()
    for name in (args.only or ALL_ITEMS):
        print(f"── {name} ──")
        ALL_ITEMS[name]()


if __name__ == "__main__":
    main()
