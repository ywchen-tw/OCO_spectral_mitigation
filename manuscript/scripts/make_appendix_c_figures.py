#!/usr/bin/env python3
"""Appendix C figures — correction model: architecture, training, CV evaluation.

Generates (2026-07-22q):
  figC1_dataflow                training-vs-inference data-flow schematic;
                                carries the training-only cloud-information
                                boundary demanded by Appendix B's include list
  figC2_fold_timeline           REAL date-blocked fold membership from the
                                training_dates.json manifests (fold-PCA fold
                                structure, shared by all foldpca models)
  figC3a_random_split_inflation the ~0.29 R2 random-split inflation (ocean
                                testbed, ocean_robustness_comparison.md);
                                panel b of item C3 is figC3b_cv_design
  figC5_cv_model_comparison     date-blocked fold mean +- std RMSE / R2 for
                                DE / XGBoost / Ridge, both surfaces, parsed
                                from the *_de_vs_baselines_kfold_agg.md files

NOT generated here (status in the flow plan): C4 skill-vs-ceiling (the
label_noise_ceiling CSV's r2max columns need their defining analysis before
plotting — naively the achieved ocean R2 exceeds r2max_ref_ret), C6 (CV
ablation covered by fig06b + Table C7), C7 (needs the ML-on-raw artifacts,
not local).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import apply_manuscript_style

OUT = REPO / "manuscript" / "figures"
C_DE = "#0072B2"          # DE fixed paper-wide
C_XGB = "#E69F00"
C_RIDGE = "#009E73"
C_TRAINONLY = "#D55E00"


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = OUT / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


# ── C1: data-flow schematic ────────────────────────────────────────────────

def _box(ax, xy, w, h, text, fc="#F0F0F0", ec="0.3", fontsize=8, lw=1.0):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.35)


def _arrow(ax, p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=kw.pop("color", "0.25"),
                                 lw=kw.pop("lw", 1.1),
                                 linestyle=kw.pop("ls", "-"),
                                 shrinkA=0, shrinkB=0))


def make_c1() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # column headers
    ax.text(0.26, 0.97, "Training (per surface)", ha="center", fontsize=9.5,
            fontweight="bold")
    ax.text(0.815, 0.97, "Inference (deployed)", ha="center", fontsize=9.5,
            fontweight="bold")
    ax.axvline(0.585, color="0.75", lw=0.8, ls=(0, (4, 3)))

    # training column ------------------------------------------------------
    _box(ax, (0.02, 0.80), 0.24, 0.115,
         "OCO-2 L1B spectra\n+ L2 Lite retrieval")
    _box(ax, (0.02, 0.615), 0.24, 0.115,
         "Cumulant fit +\nfeature pipeline")
    _box(ax, (0.02, 0.43), 0.24, 0.115,
         "Per-sounding\nfeature vector")
    _arrow(ax, (0.14, 0.80), (0.14, 0.732))
    _arrow(ax, (0.14, 0.615), (0.14, 0.547))

    # training-only cloud branch (dashed boundary)
    ax.add_patch(FancyBboxPatch((0.295, 0.40), 0.265, 0.515,
                                boxstyle="round,pad=0.014", fc="none",
                                ec=C_TRAINONLY, lw=1.3, ls=(0, (4, 2))))
    ax.text(0.4275, 0.945, "label construction + evaluation ONLY",
            ha="center", fontsize=7.2, color=C_TRAINONLY, style="italic")
    _box(ax, (0.315, 0.80), 0.225, 0.10, "Aqua-MODIS MYD35\ncloud mask",
         fc="#FDEEE4")
    _box(ax, (0.315, 0.615), 0.225, 0.10,
         "Nearest-cloud distance\n(ECEF KD-tree)", fc="#FDEEE4")
    _box(ax, (0.315, 0.43), 0.225, 0.10,
         "Clear-sky-reference\ntarget (ocean $>$5 km /\nland $>$15 km)",
         fc="#FDEEE4", fontsize=7.3)
    _arrow(ax, (0.4275, 0.80), (0.4275, 0.717), color=C_TRAINONLY)
    _arrow(ax, (0.4275, 0.615), (0.4275, 0.532), color=C_TRAINONLY)

    # folds + ensemble
    _box(ax, (0.02, 0.215), 0.24, 0.115,
         "Date-blocked folds (5)\ntrain / calibration split")
    _box(ax, (0.02, 0.03), 0.515, 0.10,
         "Per-surface deep ensemble (M = 5 Gaussian-head MLPs)\n"
         "+ split & Mondrian conformal calibration", fc="#E4EEF7")
    _arrow(ax, (0.14, 0.43), (0.14, 0.332))
    _arrow(ax, (0.14, 0.215), (0.14, 0.132))
    _arrow(ax, (0.4275, 0.43), (0.4275, 0.132), color=C_TRAINONLY)
    ax.text(0.443, 0.28, "targets", fontsize=7, color=C_TRAINONLY,
            rotation=90, va="center")

    # inference column -----------------------------------------------------
    _box(ax, (0.65, 0.80), 0.33, 0.115,
         "ONE sounding:\nL1B spectrum + L2 Lite")
    _box(ax, (0.65, 0.615), 0.33, 0.115,
         "Same feature pipeline\n(no imager, no neighbors)")
    _box(ax, (0.65, 0.43), 0.33, 0.115, "Frozen ensemble", fc="#E4EEF7")
    _box(ax, (0.65, 0.215), 0.33, 0.145,
         "Correction $\\mu$ + calibrated\n90 % interval\n"
         "(guards may withhold)")
    _arrow(ax, (0.815, 0.80), (0.815, 0.732))
    _arrow(ax, (0.815, 0.615), (0.815, 0.547))
    _arrow(ax, (0.815, 0.43), (0.815, 0.362))
    _arrow(ax, (0.535, 0.08), (0.815, 0.43), color=C_DE, lw=1.4)
    ax.text(0.675, 0.13, "deploy", fontsize=7.5, color=C_DE,
            rotation=38, ha="center")

    fig.tight_layout()
    _save(fig, "figC1_dataflow")


# ── C2: fold timeline from the training manifests ─────────────────────────

MANIFEST_TMPL = {
    "Ocean (r05)": "linreg_ocean_full_prof_foldpca_r05_f{f}",
    "Land (r15)": "linreg_land_full_prof_foldpca_r15_f{f}",
}
ROLE_STYLE = {  # role -> (color, size, label)
    "train_dates": ("0.55", 12, "train"),
    "calib_dates": ("#E69F00", 22, "calibration"),
    "held_dates": (C_DE, 30, "held-out (evaluation)"),
}


def make_c2() -> None:
    import pandas as pd
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.2), sharex=True)
    for ax, (title, tmpl) in zip(axes, MANIFEST_TMPL.items()):
        for f in range(5):
            path = (REPO / "results" / "model_linear_baseline"
                    / tmpl.format(f=f) / "training_dates.json")
            man = json.loads(path.read_text())
            for role, (color, size, _) in ROLE_STYLE.items():
                dates = pd.to_datetime(sorted(man[role]))
                ax.scatter(dates, np.full(len(dates), f), s=size, c=color,
                           marker="|", linewidths=1.6)
        ax.set_yticks(range(5), [f"fold {f}" for f in range(5)], fontsize=8)
        ax.set_ylim(-0.6, 4.6)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", color="0.9", lw=0.6)
    handles = [plt.Line2D([], [], color=c, marker="|", ls="", ms=9,
                          markeredgewidth=2, label=lab)
               for c, _, lab in ROLE_STYLE.values()]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, ncol=3,
                   loc="lower right", bbox_to_anchor=(1.0, 1.12))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "figC2_fold_timeline")


# ── C3a: random-split inflation ────────────────────────────────────────────
# Numbers from results/model_comparison/ocean_robustness_comparison.md §1
# (ocean testbed, 2026-06; the models are the comparison-era TabM/XGB/MLP —
# the SPLIT effect, not the model ranking, is the point of this panel).

SPLIT_R2 = {  # model -> (random, date-single, date-kfold mean, kfold std)
    "TabM": (0.821, 0.541, 0.530, 0.029),
    "MLP": (0.672, 0.519, 0.523, 0.059),
    "XGBoost": (0.577, 0.512, 0.491, 0.066),
}


def make_c3a() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(SPLIT_R2))
    w = 0.26
    rand = [v[0] for v in SPLIT_R2.values()]
    single = [v[1] for v in SPLIT_R2.values()]
    kf = [v[2] for v in SPLIT_R2.values()]
    kfe = [v[3] for v in SPLIT_R2.values()]
    ax.bar(x - w, rand, w, color="0.75", label="random 80/20 (leaky)")
    ax.bar(x, single, w, color="#8AB8D8", label="date (single block)")
    ax.bar(x + w, kf, w, yerr=kfe, capsize=3, color=C_DE,
           label="date k-fold (5, mean $\\pm$ std)")
    for xi, (r, k) in enumerate(zip(rand, kf)):
        ax.annotate("", (xi - w, k), (xi - w, r),
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
        ax.text(xi - w - 0.05, (r + k) / 2, f"$-${r - k:.2f}", fontsize=7.5,
                ha="right", va="center", color="0.25")
    ax.set_xticks(x, list(SPLIT_R2))
    ax.set_ylabel("held-out $R^2$ (ocean testbed)")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "figC3a_random_split_inflation")


# ── C5: date-blocked CV model comparison (parsed from the agg reports) ────

AGG_FILES = {
    "Land": REPO / "results/model_comparison/land_r15_de_vs_baselines_kfold_agg.md",
    "Ocean": REPO / "results/model_comparison/ocean_r05_de_vs_baselines_kfold_agg.md",
}
MODEL_LABEL = {"DE": "Deep ensemble", "XGB": "XGBoost", "LinReg": "Ridge"}
MODEL_COLOR = {"DE": C_DE, "XGB": C_XGB, "LinReg": C_RIDGE}


def parse_agg(path: Path) -> dict:
    """{model: {metric: (mean, std, median)}} from a kfold_agg markdown."""
    out, model = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"## (\w+)\+foldpca", line)
        if m:
            model = m.group(1)
            out[model] = {}
            continue
        m = re.match(r"\| (rmse|r2) \| ([\d.eE+-]+) ± ([\d.eE+-]+) "
                     r"\| ([\d.eE+-]+) \|", line)
        if m and model and m.group(1) not in out[model]:
            out[model][m.group(1)] = tuple(float(m.group(i))
                                           for i in (2, 3, 4))
    return out


def make_table_c5() -> None:
    """Table C5 — date-blocked fold metrics for DE/XGBoost/Ridge, both
    surfaces (the frozen values quoted in the §4.3 headline). Same
    kfold_agg parse as Fig. C5; the land Ridge fold MEDIAN is flagged."""
    data = {surf: parse_agg(p) for surf, p in AGG_FILES.items()}
    out = REPO / "manuscript" / "tables" / "tabC5_cv_model_comparison.tex"
    lines = [
        "% Auto-generated by make_appendix_c_figures.py — do not hand-edit.",
        "% Source: results/model_comparison/*_de_vs_baselines_kfold_agg.md",
        "\\begin{table}[t]",
        "\\caption{Date-blocked five-fold cross-validation of the anomaly",
        "target: held-out fold RMSE and $R^2$ (mean $\\pm$ std over folds)",
        "for the deep ensemble, XGBoost, and ridge baselines, per surface.",
        "$^{\\dagger}$fold median (one out-of-domain footprint makes the",
        "land ridge fold mean non-robust; see the aggregation report).}",
        "\\label{tab:cv-model-comparison}",
        "\\begin{tabular}{llrr}",
        "\\tophline",
        "Surface & Model & RMSE (ppm) & $R^2$ \\\\",
        "\\middlehline",
    ]
    for surf in AGG_FILES:
        for mod, label in MODEL_LABEL.items():
            cells = []
            for metric in ("rmse", "r2"):
                mean, std, med = data[surf][mod][metric]
                if mod == "LinReg" and surf == "Land":
                    cells.append(f"${med:.3f}^{{\\dagger}}$")
                else:
                    cells.append(f"${mean:.3f} \\pm {std:.3f}$")
            lines.append(f"{surf} & {label} & {cells[0]} & {cells[1]} \\\\")
        if surf != list(AGG_FILES)[-1]:
            lines.append("\\middlehline")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def make_c5() -> None:
    data = {surf: parse_agg(p) for surf, p in AGG_FILES.items()}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, metric, label in [(axes[0], "rmse", "fold RMSE (ppm)"),
                              (axes[1], "r2", "fold $R^2$")]:
        x = np.arange(len(AGG_FILES))
        w = 0.24
        for i, mod in enumerate(MODEL_LABEL):
            vals, errs, hatches = [], [], []
            for surf in AGG_FILES:
                mean, std, med = data[surf][mod][metric]
                # LinReg land: one out-of-domain footprint blows up the
                # mean — the agg report itself says use the median
                outlier = mod == "LinReg" and surf == "Land"
                vals.append(med if outlier else mean)
                errs.append(np.nan if outlier else std)
                hatches.append("//" if outlier else None)
            bars = ax.bar(x + (i - 1) * w, vals, w,
                          yerr=[0 if np.isnan(e) else e for e in errs],
                          capsize=3, color=MODEL_COLOR[mod],
                          label=MODEL_LABEL[mod])
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
                    b.set_edgecolor("white")
        ax.set_xticks(x, list(AGG_FILES))
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].text(0.02, 0.97,
                 "hatched: fold median\n(outlier fold, see agg report)",
                 transform=axes[1].transAxes, fontsize=6.6, color="0.35",
                 va="top")
    fig.tight_layout()
    _save(fig, "figC5_cv_model_comparison")


if __name__ == "__main__":
    apply_manuscript_style()
    make_c1()
    make_c2()
    make_c3a()
    make_c5()
    make_table_c5()
