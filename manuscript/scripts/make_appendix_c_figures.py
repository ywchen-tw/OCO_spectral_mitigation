#!/usr/bin/env python3
"""Appendix C figures + tables — correction model: architecture, training,
CV evaluation.

Figures (2026-07-22q; C4/C7 + single-panel C2 added 2026-07-23):
  figC1_dataflow                training-vs-inference data-flow schematic;
                                carries the training-only cloud-information
                                boundary demanded by Appendix B's include list
  figC2_fold_timeline           REAL date-blocked fold membership from the
                                training_dates.json manifests. ONE panel: the
                                fold structure is shared by both surfaces and
                                all foldpca models (asserted here)
  internal_random_split_inflation
                                the ~0.29 R2 random-split inflation (ocean
                                testbed, ocean_robustness_comparison.md).
                                INTERNAL since 2026-07-23 (user decision: no
                                random-vs-date-split discussion in the
                                manuscript — reviewer-response material only;
                                its companion cv-design schematic is likewise
                                internal_cv_design, make_cv_design_figure.py)
  figC4_skill_vs_ceiling        achieved DE held-out fold R2 vs the label-
                                noise ceiling (label_noise_ceiling_140dates).
                                Semantics resolved 2026-07-23 from
                                analysis/label_noise_ceiling.py: r2max_ref
                                (reference-sampling noise only) is the only
                                HARD ceiling; r2max_ref_ret additionally
                                treats the retrieval posterior sigma^2 as
                                irreducible noise — an assumption the anomaly
                                target partly cancels (locally common-mode
                                retrieval error subtracts out) and the model
                                partly predicts (that is the correction's
                                job), so achieved skill may legitimately
                                exceed it (ocean all: 0.708 > 0.603); same
                                for the empirical far-field line (far-field
                                variance contains predictable retrieval-error
                                structure — the model's far-cloud R2 > 0).
  figC5_cv_model_comparison     date-blocked fold mean +- std RMSE / R2 for
                                DE / XGBoost / Ridge, both surfaces, computed
                                from the per-fold artifacts (Ridge under the
                                production |mu| output guard — see Table C5)
  (no figC6)                    the CV-side ablation is discussed with
                                main-text Fig. 6b and tabulated in Table C7;
                                a figC6 was briefly generated 2026-07-23c and
                                REMOVED same day (user decision 2026-07-23d)
  figC7_increment_attribution   raw/BC/ML increment relationship (pairs with
                                Table C8): per-footprint dmu = mu_raw - mu_bc
                                vs the operational increment inc = raw - bc,
                                from the SAME plot_data trees as
                                RAW_BC_ML_TCCON_2026-07-16.md section 4

Tables:
  tabC1_predictor_inventory     full predictor inventory from models.pipeline
                                (single source: _FEATURE_MAP + ablation sets)
  tabC2_training_config         production hyperparameters, read from the ten
                                fold run_summary.json files (asserted equal)
  tabC3_fold_sizes_metrics      per-fold date counts + held-out n / RMSE /
                                R2 / coverage for the production DE
  tabC4_manifest_verification   training-date manifests vs every independent
                                evaluation set; zero-overlap computed live
  tabC5_cv_model_comparison     frozen date-blocked fold metrics (section 4.3)
  tabC6_fold_resolved_baselines fold-by-fold DE / XGBoost / Ridge metrics
  tabC7_cv_ablation             held-out CV feature-set ablation (2026-07-17
                                retrained lndo01 fold-PCA variants)

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


def _load_manifests() -> dict:
    """{fold: manifest} — asserts ocean and land manifests are identical
    (one date-blocked fold structure shared by every foldpca model)."""
    out = {}
    for f in range(5):
        mans = []
        for tmpl in MANIFEST_TMPL.values():
            path = (REPO / "results" / "model_linear_baseline"
                    / tmpl.format(f=f) / "training_dates.json")
            mans.append(json.loads(path.read_text()))
        for role in ROLE_STYLE:
            assert sorted(mans[0][role]) == sorted(mans[1][role]), \
                f"fold {f} {role}: ocean and land manifests differ"
        out[f] = mans[0]
    return out


def make_c2() -> None:
    import pandas as pd
    manifests = _load_manifests()      # asserts ocean == land
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    for f, man in manifests.items():
        for role, (color, size, _) in ROLE_STYLE.items():
            dates = pd.to_datetime(sorted(man[role]))
            ax.scatter(dates, np.full(len(dates), f), s=size, c=color,
                       marker="|", linewidths=1.6)
    ax.set_yticks(range(5), [f"fold {f}" for f in range(5)], fontsize=8)
    ax.set_ylim(-0.6, 4.6)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9", lw=0.6)
    handles = [plt.Line2D([], [], color=c, marker="|", ls="", ms=9,
                          markeredgewidth=2, label=lab)
               for c, _, lab in ROLE_STYLE.values()]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3,
              loc="lower right", bbox_to_anchor=(1.0, 1.02))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, "figC2_fold_timeline")


# ── random-split inflation (INTERNAL — reviewer-response only) ────────────
# Numbers from results/model_comparison/ocean_robustness_comparison.md §1
# (ocean testbed, 2026-06; the models are the comparison-era TabM/XGB/MLP —
# the SPLIT effect, not the model ranking, is the point of this panel).
# Dropped from the manuscript 2026-07-23 (user decision): the random-vs-
# date-split comparison is not discussed unless reviewers ask.

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
    _save(fig, "internal_random_split_inflation")


# ── C5: date-blocked CV model comparison (computed from fold artifacts) ───
# 2026-07-23: no longer parsed from the kfold_agg markdowns.  DE/XGB come
# from their per-fold metrics jsons; Ridge is recomputed from its per-fold
# held_out_predictions.parquet with the PRODUCTION output guard applied
# (corrections withheld where |mu| > ANOMALY_GUARD_PPM — exactly what the
# deployed chain does for every model), which removes the single
# out-of-domain-footprint blowup (land fold 2, raw |mu| ~ 1e6 ppm) that
# used to force a fold-median + dagger.  The guard is NOT applied to the
# bounded DE/XGBoost outputs: their max |mu| is ~27/32 ppm, and a uniform
# guard would change the frozen land fold means by +0.03 (their
# large-|mu| predictions are mostly correct) — see the table note.

MODEL_LABEL = {"DE": "DE", "XGB": "XGBoost", "LinReg": "Ridge"}
MODEL_COLOR = {"DE": C_DE, "XGB": C_XGB, "LinReg": C_RIDGE}
ANOMALY_GUARD_PPM = 25.0      # build_deepens_plot_data output-guard default


def _point_metrics_files() -> dict:
    return {
        "DE": (DE_FOLD_TMPL, "de_mondrian_date_kfold_metrics.json"),
        "XGB": (XGB_FOLD_TMPL, "xgboost_mean_date_kfold_metrics.json"),
    }


def _ridge_guarded_fold(surface: str, f: int) -> dict:
    """Ridge held-out fold metrics with the production output guard."""
    import pandas as pd
    d = pd.read_parquet(
        REPO / LINREG_FOLD_TMPL[surface].format(f=f)
        / "held_out_predictions.parquet", columns=["y_true", "mu"])
    y = d.y_true.to_numpy(float)
    mu = d.mu.to_numpy(float)
    g = np.abs(mu) > ANOMALY_GUARD_PPM
    mu = np.where(g, 0.0, mu)                 # correction withheld
    res = y - mu
    # standard R2 (uncentered residual SS over centered total SS), matching
    # models/diagnostics.compute_metrics — NOT 1 - var(res)/var(y)
    return {"rmse": float(np.sqrt(np.mean(res ** 2))),
            "r2": float(1.0 - np.mean(res ** 2) / np.var(y)),
            "n": int(len(y)), "n_guarded": int(g.sum())}


def _cv_fold_data() -> dict:
    """{surface: {model: {"rmse": [...], "r2": [...], "n_guarded": int}}}"""
    out = {}
    for surface in ("Ocean", "Land"):
        out[surface] = {}
        for mod, (tmpls, fname) in _point_metrics_files().items():
            tmpl = tmpls[surface] if isinstance(tmpls[surface], str) \
                else tmpls[surface][0]
            gs = [_fold_global(REPO / tmpl.format(f=f) / fname)
                  for f in range(5)]
            out[surface][mod] = {"rmse": [g["rmse"] for g in gs],
                                 "r2": [g["r2"] for g in gs],
                                 "n_guarded": 0}
        gs = [_ridge_guarded_fold(surface, f) for f in range(5)]
        out[surface]["LinReg"] = {"rmse": [g["rmse"] for g in gs],
                                  "r2": [g["r2"] for g in gs],
                                  "n_guarded": sum(g["n_guarded"]
                                                   for g in gs)}
    return out


def make_table_c5() -> None:
    """Table C5 — date-blocked fold metrics for DE/XGBoost/Ridge, both
    surfaces (the frozen values quoted in the §4.3 headline)."""
    data = _cv_fold_data()
    ng = sum(data[s]["LinReg"]["n_guarded"] for s in data)
    body = []
    for surf in ("Ocean", "Land"):
        for mod, label in MODEL_LABEL.items():
            cells = []
            for metric in ("rmse", "r2"):
                v = np.array(data[surf][mod][metric])
                cells.append(f"${v.mean():.3f} \\pm {v.std():.3f}$")
            star = "$^{*}$" if mod == "LinReg" else ""
            body.append(f"{surf} & {label}{star} & {cells[0]} & "
                        f"{cells[1]} \\\\")
        if surf != "Land":
            body.append("\\middlehline")
    _tex_table(
        "tabC5_cv_model_comparison",
        "Date-blocked five-fold cross-validation of the anomaly target: "
        "held-out fold RMSE (ppm) and $R^2$ (mean $\\pm$ std over folds) "
        "for the deep ensemble (DE), XGBoost, and ridge baselines, per "
        "surface.",
        "tab:cv-model-comparison", "llrr",
        ["Surface & Model & RMSE (ppm) & $R^2$ \\\\"],
        body,
        notes=f"$^{{*}}$Ridge predictions pass the production output guard "
              f"(correction withheld where $|\\mu| > "
              f"{ANOMALY_GUARD_PPM:g}$\\,ppm), which the deployed chain "
              f"applies to every model; it triggers on {ng} of the "
              f"$\\sim$11.6\\,M ridge held-out footprints — dominated by a "
              f"single out-of-domain footprint whose unbounded linear "
              f"extrapolation reaches $|\\mu| \\sim 10^{{6}}$\\,ppm — and "
              f"is not applied to the bounded DE/XGBoost outputs "
              f"(max $|\\mu| < 32$\\,ppm).")


def make_c5() -> None:
    data = _cv_fold_data()
    surfaces = ["Ocean", "Land"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, metric, label in [(axes[0], "rmse", "fold RMSE (ppm)"),
                              (axes[1], "r2", "fold $R^2$")]:
        x = np.arange(len(surfaces))
        w = 0.24
        for i, mod in enumerate(MODEL_LABEL):
            vals = [float(np.mean(data[s][mod][metric])) for s in surfaces]
            errs = [float(np.std(data[s][mod][metric])) for s in surfaces]
            ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
                   color=MODEL_COLOR[mod], label=MODEL_LABEL[mod])
        ax.set_xticks(x, surfaces)
        ax.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "figC5_cv_model_comparison")


# ── C4: date-blocked skill vs label-noise ceiling ─────────────────────────
# Semantics (resolved 2026-07-23, analysis/label_noise_ceiling.py docstring):
# r2max_ref uses ONLY the clear-sky-reference-mean sampling error — the one
# noise term that is irreducible by construction → the hard ceiling.
# r2max_ref_ret adds the per-sounding retrieval posterior sigma^2 as if it
# were irreducible; the within-orbit anomaly cancels locally common-mode
# retrieval error, and feature-predictable retrieval error is exactly what
# the model corrects, so achieved skill may exceed this line (and does:
# ocean all 0.708 > 0.603).  r2max_emp treats far-field variance as pure
# noise; the model's far-cloud R2 > 0 shows that variance contains
# predictable structure, so this line is also exceedable.  The figure shows
# all three as labelled reference lines, not as a single "ceiling".

CEILING_CSV = REPO / "results" / "label_noise_ceiling_140dates.csv"
DE_FOLD_TMPL = {
    "Ocean": ("results/model_deep_ensemble/"
              "de_ocean_beta_nll_prof_reg_foldpca_r05_f{f}", "r05"),
    "Land": ("results/model_deep_ensemble/"
             "de_land_beta_nll_prof_reg_foldpca_r15_f{f}", "r15"),
}
XGB_FOLD_TMPL = {
    "Ocean": "results/model_gbdt/xgb_ocean_full_prof_foldpca_r05_f{f}",
    "Land": "results/model_gbdt/xgb_land_full_prof_foldpca_r15_f{f}",
}
LINREG_FOLD_TMPL = {
    "Ocean": "results/model_linear_baseline/"
             "linreg_ocean_full_prof_foldpca_r05_f{f}",
    "Land": "results/model_linear_baseline/"
            "linreg_land_full_prof_foldpca_r15_f{f}",
}


def _fold_global(path: Path) -> dict:
    return json.loads(path.read_text())["global"]


def _de_fold_r2(surface: str) -> dict:
    """{'all'|'near'|'far': [r2 per fold]} for the production DE."""
    import pandas as pd
    tmpl, _ = DE_FOLD_TMPL[surface]
    res = {"all": [], "near": [], "far": []}
    for f in range(5):
        d = REPO / tmpl.format(f=f)
        res["all"].append(
            _fold_global(d / "de_mondrian_date_kfold_metrics.json")["r2"])
        s = pd.read_csv(d / "de_mondrian_date_kfold_stratified_metrics.csv")
        cp = s[s.regime == "cloud_proximity"].set_index("group")["r2"]
        res["near"].append(float(cp["near_cloud(<=10km)"]))
        res["far"].append(float(cp["far_cloud(>10km)"]))
    return res


def make_c4() -> None:
    import pandas as pd
    ceil = pd.read_csv(CEILING_CSV)
    cats = [("all", "all"), ("near", "near\n$<$10 km"), ("far", "far\n$>$10 km")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, surface, letter in zip(axes, DE_FOLD_TMPL, "ab"):
        _, tgt = DE_FOLD_TMPL[surface]
        ach = _de_fold_r2(surface)
        x = np.arange(len(cats))
        means = [float(np.mean(ach[k])) for k, _ in cats]
        stds = [float(np.std(ach[k])) for k, _ in cats]
        ax.bar(x, means, 0.5, yerr=stds, capsize=3, color=C_DE,
               label="achieved (DE, held-out folds)")
        c = (ceil[(ceil.surface == surface.lower()) & (ceil.target == tgt)]
             .set_index("subset"))
        sub = {"all": "all", "near": "near<10km"}     # no far ceiling rows

        def seg(xi, y, color, ls, label=None):
            ax.plot([xi - 0.33, xi + 0.33], [y, y], color=color, ls=ls,
                    lw=1.7, label=label, solid_capstyle="butt")

        first = surface == "Ocean"
        for xi, (k, _) in enumerate(cats):
            if k not in sub:
                continue
            row = c.loc[sub[k]]
            seg(xi, row.r2max_ref, "0.15", "-",
                "hard ceiling (reference-sampling noise)" if first and xi == 0
                else None)
            seg(xi, row.r2max_ref_ret, C_XGB, (0, (4, 2)),
                "$+$ posterior $\\sigma^2$ treated as noise"
                if first and xi == 0 else None)
            if np.isfinite(row.get("r2max_emp", np.nan)):
                seg(xi, row.r2max_emp, C_RIDGE, (0, (1, 1.6)),
                    "far-field variance as noise (empirical)"
                    if first and xi == 0 else None)
        ax.set_xticks(x, [lab for _, lab in cats], fontsize=8)
        ax.set_title(f"({letter}) {surface} ({tgt})", fontsize=9, loc="left")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("held-out fold $R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.2, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "figC4_skill_vs_ceiling")


# ── C7: raw/BC/ML increment relationship (pairs with Table C8) ────────────

def make_c7() -> None:
    from matplotlib.colors import LogNorm
    from make_raw_bc_ml_report import _load_pairs   # workspace on sys.path
    d = _load_pairs()
    inc = d.inc.to_numpy(float)
    dmu = (d.mu_raw - d.mu_bc).to_numpy(float)
    slope = float(np.cov(inc, dmu, bias=True)[0, 1] / inc.var())
    r = float(np.corrcoef(inc, dmu)[0, 1])
    frac = float(1.0 - np.var(inc - dmu) / np.var(inc))

    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.05, 1.3]})

    ax = axes[0]
    lim = 6.0
    h = ax.hist2d(inc, dmu, bins=170, range=[[-lim, lim], [-lim, lim]],
                  norm=LogNorm(), cmap="plasma", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color="0.4", ls=(0, (4, 3)), lw=1.0,
            label="$\\Delta\\mu = \\mathrm{inc}$ (full internalization)")
    xs = np.array([-lim, lim])
    ax.plot(xs, slope * (xs - inc.mean()) + dmu.mean(), color="#0072B2",
            lw=1.4, label=f"OLS: slope {slope:+.2f}, $r$ {r:+.2f}")
    ax.set_xlabel("operational increment "
                  "inc $\\equiv X_{\\mathrm{CO2}}^{\\mathrm{raw}} - "
                  "X_{\\mathrm{CO2}}^{\\mathrm{B11}}$ (ppm)")
    ax.set_ylabel("$\\Delta\\mu \\equiv \\mu_{\\mathrm{raw}} - "
                  "\\mu_{\\mathrm{B11}}$ (ppm)")
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="lower left",
              bbox_to_anchor=(0.0, 1.01))
    ax.text(0.97, 0.03, f"var(inc) explained: {100 * frac:.0f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.colorbar(h[3], ax=ax, pad=0.015).set_label("soundings", fontsize=7)

    ax = axes[1]
    strata = [
        ("all", np.ones(len(d), bool)),
        ("ocean", (d.sfc_type == 0).to_numpy()),
        ("land", (d.sfc_type == 1).to_numpy()),
        ("ocean\n$\\leq$10 km",
         ((d.sfc_type == 0) & (d.cld_dist_km <= 10)).to_numpy()),
        ("land\n$\\leq$10 km",
         ((d.sfc_type == 1) & (d.cld_dist_km <= 10)).to_numpy()),
        ("land\n$>$10 km",
         ((d.sfc_type == 1) & (d.cld_dist_km > 10)).to_numpy()),
    ]
    mu_bc = d.mu_bc.to_numpy(float)
    r_d, r_bc = [], []
    for _, m in strata:
        r_d.append(float(np.corrcoef(inc[m], dmu[m])[0, 1]))
        r_bc.append(float(np.corrcoef(inc[m], mu_bc[m])[0, 1]))
    x = np.arange(len(strata))
    w = 0.36
    ax.bar(x - w / 2, r_d, w, color=C_DE,
           label="$r(\\Delta\\mu, \\mathrm{inc})$ — rediscovery by ML(raw)")
    ax.bar(x + w / 2, r_bc, w, color="0.6",
           label="$r(\\mu_{\\mathrm{B11}}, \\mathrm{inc})$ — overlap of "
                 "production ML")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x, [lab for lab, _ in strata], fontsize=7.5)
    ax.set_ylabel("correlation with the operational increment")
    ax.set_ylim(-0.55, 1.0)
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="lower left",
              bbox_to_anchor=(0.0, 1.01))
    fig.tight_layout()
    _save(fig, "figC7_increment_attribution")


# ── LaTeX table helper (Copernicus \tophline style, matching tabC5) ───────

def _tex_table(stem: str, caption: str, label: str, colspec: str,
               header_rows: list, body_rows: list, star: bool = False,
               size: 'str | None' = None, notes: 'str | None' = None,
               longtable: bool = False) -> None:
    # column count: top-level l/c/r/p specifiers, ignoring {...} arguments
    ncol, depth = 0, 0
    for ch in colspec:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and ch in "lcrp":
            ncol += 1
    lines = [
        "% Auto-generated by make_appendix_c_figures.py — do not hand-edit.",
    ]
    if longtable:
        # breaks across pages; header repeated on continuation pages
        if size:
            lines.append(f"\\begingroup\\{size}")
        lines += [
            f"\\begin{{longtable}}{{{colspec}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}\\\\",
            "\\tophline", *header_rows, "\\middlehline", "\\endfirsthead",
            f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{(continued)}}}}\\\\",
            "\\tophline", *header_rows, "\\middlehline", "\\endhead",
            "\\bottomhline", "\\endfoot",
        ]
        if notes:
            lines += ["\\bottomhline",
                      f"\\multicolumn{{{ncol}}}{{p{{0.9\\linewidth}}}}"
                      f"{{\\footnotesize {notes}}}\\\\",
                      "\\endlastfoot"]
        else:
            lines += ["\\bottomhline", "\\endlastfoot"]
        lines += [*body_rows, "\\end{longtable}"]
        if size:
            lines.append("\\endgroup")
        lines.append("")
    else:
        env = "table*" if star else "table"
        lines += [
            f"\\begin{{{env}}}[t]",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
        ]
        if size:
            lines.append(f"\\{size}")
        lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\tophline",
                  *header_rows, "\\middlehline", *body_rows, "\\bottomhline",
                  "\\end{tabular}"]
        if notes:
            lines.append(f"\\belowtable{{{notes}}}")
        lines += [f"\\end{{{env}}}", ""]
    out = REPO / "manuscript" / "tables" / f"{stem}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def _tt(name: str) -> str:
    """Feature name as escaped \\texttt{}."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


# ── Table C1: full predictor inventory (from models.pipeline) ─────────────
# Group = the ablation group where one exists (matches _FEATURE_SETS exactly),
# else a descriptive base group.  DRAFT descriptions — verify the marked ones
# against the B11 Data User's Guide at writing time.

_C1_DESC = {
    'xco2_raw_minus_apriori': "Retrieved (pre-bias-correction) XCO$_2$ minus prior XCO$_2$ (ppm)",
    'exp_o2a_intercept': "O$_2$A continuum reflectance (exponential of the cumulant-fit intercept)",
    'o2a_exp_intercept-alb': "O$_2$A continuum reflectance minus retrieved O$_2$A albedo",
    'wco2_exp_intercept-alb': "WCO$_2$ continuum reflectance minus retrieved WCO$_2$ albedo",
    'o2a_k1': "O$_2$A mean relative photon path $\\langle l'\\rangle$ (this work, Sect.~3.2)",
    'o2a_k2': "O$_2$A photon-path variance $\\mathrm{var}(l')$",
    'wco2_k1': "WCO$_2$ $\\langle l'\\rangle$",
    'wco2_k2': "WCO$_2$ $\\mathrm{var}(l')$",
    'wco2_k3': "WCO$_2$ third-order path-length cumulant",
    'sco2_k1': "SCO$_2$ $\\langle l'\\rangle$",
    'sco2_k2': "SCO$_2$ $\\mathrm{var}(l')$",
    'cos_glint_angle': "Cosine of the glint angle",
    '1_over_cos_sza': "Solar air-mass factor $1/\\cos(\\mathrm{SZA})$",
    '1_over_cos_vza': "Viewing air-mass factor $1/\\cos(\\mathrm{VZA})$",
    'sin_raa': "Sine of the relative azimuth angle",
    'pol_ang_rad': "Polarization angle (rad)",
    'fp_area_km2': "Footprint area (km$^2$)",
    'csnr_o2a': "Continuum signal-to-noise ratio, O$_2$A band",
    'csnr_sco2': "Continuum signal-to-noise ratio, SCO$_2$ band",
    'h_cont_o2a': "O$_2$A continuum-radiance variability across adjacent soundings (per-sounding preprocessor value)",
    'h_cont_wco2': "WCO$_2$ continuum-radiance variability across adjacent soundings",
    'h_cont_sco2': "SCO$_2$ continuum-radiance variability across adjacent soundings",
    's31': "Band signal ratio SCO$_2$/O$_2$A",
    'co2_ratio_bc': "IDP CO$_2$ band-ratio cloud-screen diagnostic (bias-corrected)",
    'h2o_ratio_bc': "IDP H$_2$O band-ratio cloud-screen diagnostic (bias-corrected)",
    'log_P': "Log of retrieved surface pressure",
    'dp_psfc_prior_ratio': "Retrieved-minus-prior surface pressure, relative to the prior",
    'h2o_scale': "Retrieved H$_2$O-profile scale factor",
    'delT': "Retrieved temperature-profile offset (K)",
    'co2_grad_del': "Change in the retrieved CO$_2$ vertical gradient relative to the prior",
    'tcwv': "Total column water vapour",
    'aod_dust': "Retrieved dust aerosol optical depth",
    'aod_oc': "Retrieved organic-carbon aerosol optical depth",
    'aod_seasalt': "Retrieved sea-salt aerosol optical depth",
    'aod_strataer': "Retrieved stratospheric-aerosol optical depth",
    'aod_sulfate': "Retrieved sulfate aerosol optical depth",
    'aod_water': "Retrieved water-cloud optical depth",
    'aod_ice': "Retrieved ice-cloud optical depth",
    'dp_abp': "A-band-preprocessor surface-pressure difference (cloud screen)",
    't700': "Temperature at 700\\,hPa (K)",
    'water_height': "Retrieved water-cloud layer height",
    'dust_height': "Retrieved dust layer height",
    'ice_height': "Retrieved ice-cloud layer height",
    'alb_sco2_over_wco2': "Retrieved albedo ratio SCO$_2$/WCO$_2$",
    'max_declock_wco2': "Maximum WCO$_2$ declocking correction (detector diagnostic)",
    'dpfrac': "Fractional retrieved-minus-prior surface-pressure difference",
    'fs_rel_0': "Relative fluorescence signal (clamped $\\geq 0$)",
    'alt_std': "Std.\\ dev.\\ of surface altitude within the footprint (roughness)",
}

_C1_BASE_GEOM = {
    'cos_glint_angle', '1_over_cos_sza', '1_over_cos_vza', 'sin_raa',
    'pol_ang_rad', 'fp_area_km2', 'csnr_o2a', 'csnr_sco2', 'h_cont_o2a',
    'h_cont_sco2', 's31', 'co2_ratio_bc', 'h2o_ratio_bc',
}
_C1_GROUP_ORDER = [
    ("Retrieval state", "xco2"),
    ("Spectral path-length", "spec"),
    ("Geometry \\& L1B diagnostics", "geom"),
    ("Meteorology, surface \\& aerosol", "met"),
    ("Contamination indicators", "contam"),
    ("Footprint", "fp"),
    ("Profile EOFs (appended block)", "prof"),
]


def make_table_c1() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from models import pipeline as fpipe

    def base(f):
        return f[:-5] if f.endswith("_nosg") else f

    def group_of(f):
        if f in fpipe.XCO2_FEATURES:
            return "xco2"
        if f in fpipe.SPEC_FEATURES:
            return "spec"
        if f in fpipe.CONTAM_FEATURES:
            return "contam"
        return "geom" if base(f) in _C1_BASE_GEOM else "met"

    ocean, land = fpipe._FEATURES_SFC0, fpipe._FEATURES_SFC1
    order = ocean + [f for f in land if f not in ocean]
    rows_by_group = {k: [] for _, k in _C1_GROUP_ORDER}
    for f in order:
        sfc = ("O$+$L" if f in ocean and f in land
               else ("O" if f in ocean else "L"))
        tr = ("log1p$+$RS" if f in fpipe._LOG1P_FEATURES else "RS")
        desc = _C1_DESC.get(base(f))
        assert desc is not None, f"no description for feature {f}"
        rows_by_group[group_of(f)].append(
            f"{_tt(f)} & {sfc} & {tr} & {desc}")
    rows_by_group["fp"].append(
        _tt("fp_0") + "\\,\\ldots\\," + _tt("fp_7")
        + " & O$+$L & none & Footprint-index one-hot (8 across-track positions)")
    rows_by_group["prof"] += [
        _tt("t_pc01") + "--" + _tt("t_pc04")
        + " & O$+$L & SS & Temperature-profile EOF scores ($\\sigma$-grid, per surface)",
        _tt("q_pc01") + "--" + _tt("q_pc04")
        + " & O$+$L & SS & Specific-humidity EOF scores (log-compressed $\\sigma$-grid)",
        _tt("co2prior_pc01") + "--" + _tt("co2prior_pc04")
        + " & O$+$L & SS & Shape-normalized prior-CO$_2$-profile EOF scores",
        _tt("tropopause_sigma")
        + " & O$+$L & SS & Tropopause pressure in $\\sigma$ coordinate",
        _tt("tropopause_temp") + " & O$+$L & SS & Tropopause temperature (K)",
    ]
    body = []
    for title, key in _C1_GROUP_ORDER:
        if not rows_by_group[key]:
            continue
        body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\")
        body += [r + " \\\\" for r in rows_by_group[key]]
        if key != _C1_GROUP_ORDER[-1][1]:
            body.append("\\middlehline")
    n_o = len(ocean) + 8 + 14
    n_l = len(land) + 8 + 14
    _tex_table(
        "tabC1_predictor_inventory",
        "Full predictor inventory of the per-surface correction models "
        f"(ocean: {n_o} inputs; land: {n_l}), generated from the feature "
        "pipeline source. All quantities are per-sounding OCO-2 L2 Lite "
        "columns except the spectral path-length block (fitted from the L1B "
        "spectrum, Sect.~3.2) and the profile-EOF block (fold-fitted PCA of "
        "the Lite meteorology/prior profiles). Sfc: which surface model uses "
        "the feature. Transform: RS $=$ robust--standard scaling "
        "(median/IQR, then standardization), log1p applied first where "
        "marked; SS $=$ block-own standard scaler; both fitted on the "
        "training split only. Cloud distance is not an input.",
        "tab:predictor-inventory", "lccp{0.50\\linewidth}",
        ["Feature & Sfc & Transform & Description \\\\"],
        body, size="footnotesize", longtable=True,
        notes="The three named ablation groups of Table~2 map exactly onto "
              "the groups above: $-$xco2 drops the retrieval-state row, "
              "$-$spec the spectral path-length block, $-$contam the "
              "contamination-indicator block.")


# ── Table C2: production hyperparameters (from the fold run summaries) ────

def make_table_c2() -> None:
    cfgs = []
    for surface, (tmpl, _) in DE_FOLD_TMPL.items():
        for f in range(5):
            cfgs.append(json.loads(
                (REPO / tmpl.format(f=f) / "run_summary.json").read_text()
            )["config"])
    keys = ["hidden_dims", "norm", "dropout", "loss", "beta", "n_members",
            "calib_frac", "train_lr", "train_weight_decay", "train_pct_start",
            "train_div_factor", "train_final_div_factor", "train_grad_clip",
            "train_batch_size", "train_epochs", "train_patience", "seed",
            "feature_set"]
    for k in keys:
        vals = {json.dumps(c[k]) for c in cfgs}
        assert len(vals) == 1, f"config key {k} differs across folds: {vals}"
    c = cfgs[0]
    hid = "$\\rightarrow$".join(str(h) for h in c["hidden_dims"])
    rows = [
        ("Member architecture", f"MLP $n\\rightarrow${hid}$\\rightarrow"
         "(\\mu, \\log\\sigma^2)$ Gaussian heads"),
        ("Normalization / dropout", f"layer normalization / {c['dropout']}"),
        ("Loss", f"$\\beta$-NLL, $\\beta = {c['beta']:g}$"),
        ("Members per fold $M$", f"{c['n_members']}"),
        ("Folds", "5, contiguous date blocks (Fig.~C2); "
                  "pooled inference ensemble $5\\times M = 25$"),
        ("Calibration split", f"{c['calib_frac']:g} of training dates "
         "(date-blocked; early stopping and conformal calibration)"),
        ("Optimizer", f"AdamW, lr {c['train_lr']:g}, "
         f"weight decay {c['train_weight_decay']:g}"),
        ("Schedule", "OneCycle (pct\\_start "
         f"{c['train_pct_start']:g}, div {c['train_div_factor']:g}, "
         f"final div {c['train_final_div_factor']:g})"),
        ("Batch size / max epochs", f"{c['train_batch_size']} / "
         f"{c['train_epochs']} (early stop, patience "
         f"{c['train_patience']}, best-validation checkpoint)"),
        ("Gradient clip / seed", f"{c['train_grad_clip']:g} / {c['seed']}"),
        ("Feature set", f"\\texttt{{{c['feature_set']}}} $+$ profile-EOF "
         "block (fold-fitted PCA)"),
        ("Feature scaling", "robust--standard scaler; log1p on AODs and "
         "footprint area; fitted on the training split only"),
        ("Uncertainty calibration", "split $+$ Mondrian conformal "
         "(cloud-distance bins), 90\\,\\% intervals"),
    ]
    _tex_table(
        "tabC2_training_config",
        "Production training configuration of the per-surface deep "
        "ensembles, read from the ten fold run manifests (identical across "
        "surfaces and folds by assertion).",
        "tab:training-config", "lp{0.60\\linewidth}",
        ["Setting & Value \\\\"],
        [f"{k} & {v} \\\\" for k, v in rows], size="small",
        longtable=True)


# ── Table C3: fold-level sample sizes and DE performance ──────────────────

def make_table_c3() -> None:
    manifests = _load_manifests()
    body = []
    for surface, (tmpl, tgt) in DE_FOLD_TMPL.items():
        body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{surface} "
                    f"({tgt})}}}} \\\\")
        for f in range(5):
            man = manifests[f]
            g = _fold_global(REPO / tmpl.format(f=f)
                             / "de_mondrian_date_kfold_metrics.json")
            body.append(
                f"fold {f} & {len(man['train_dates'])}/"
                f"{len(man['calib_dates'])}/{len(man['held_dates'])} & "
                f"{g['n']:,} & {g['rmse']:.3f} & {g['r2']:.3f} & "
                f"{g['coverage_90']:.3f} & {g['mean_interval_width']:.2f}"
                " \\\\".replace(",", "\\,"))
        if surface != list(DE_FOLD_TMPL)[-1]:
            body.append("\\middlehline")
    _tex_table(
        "tabC3_fold_sizes_metrics",
        "Fold-level sample sizes and held-out performance of the production "
        "deep ensemble. Dates: training/calibration/held-out date counts of "
        "the shared date-blocked fold structure (Fig.~C2); $n$: held-out "
        "soundings; RMSE (ppm) and $R^2$ against the anomaly target; "
        "cov$_{90}$ and width (ppm): empirical coverage and mean width of "
        "the Mondrian-conformal 90\\,\\% intervals.",
        "tab:fold-sizes-metrics", "lcrrrrr",
        ["Fold & Dates (tr/cal/held) & $n$ & RMSE & $R^2$ & cov$_{90}$ & "
         "width \\\\"],
        body, size="small")


# ── Table C4: training manifests + zero-overlap verification ──────────────

TAG_DIR = (REPO / "results" / "model_comparison" / "deep_ensemble"
           / "de_beta_nll_prof_reg_foldpca_o05l15_m5")
EVAL_TREES = {
    "TCCON (A-train era)": TAG_DIR / "atrain",
    "TCCON (drift era)": TAG_DIR / "drift",
    "ATom pseudo-columns": TAG_DIR / "atom",
    "Shipborne EM27/SUN": TAG_DIR / "ship",
}


def make_table_c4() -> None:
    manifests = _load_manifests()
    train_dates = set()
    for man in manifests.values():
        for role in ("train_dates", "calib_dates", "held_dates"):
            train_dates |= set(man[role])
    date_re = re.compile(r"combined_(\d{4}-\d{2}-\d{2})_")
    body = [f"Training $+$ calibration $+$ held (all folds) & — & "
            f"{len(train_dates)} & "
            f"{min(train_dates)} -- {max(train_dates)} & — \\\\",
            "\\middlehline"]
    for label, tree in EVAL_TREES.items():
        dates = sorted({m.group(1) for p in tree.glob("combined_*")
                        if (m := date_re.match(p.name))})
        cases = sum(1 for p in tree.glob("combined_*") if p.is_dir())
        overlap = sorted(set(dates) & train_dates)
        assert not overlap, f"{label}: eval dates overlap training: {overlap}"
        body.append(f"{label} & {cases} & {len(dates)} & "
                    f"{dates[0]} -- {dates[-1]} & 0 \\\\")
    _tex_table(
        "tabC4_manifest_verification",
        "Training-date manifests and evaluation-date disjointness. The 116 "
        "model dates (training, calibration, and held-out roles pooled over "
        "all five folds; identical for both surfaces) are compared against "
        "every independent evaluation set; the intersection column is "
        "computed from the per-run \\texttt{training\\_dates.json} manifests "
        "at table-generation time and enforced programmatically by the "
        "leakage guard in every evaluation builder.",
        "tab:manifest-verification", "lrrcr",
        ["Set & Cases & Unique dates & Date range & "
         "$\\cap$ model dates \\\\"],
        body, star=True, size="small")


# ── Table C6: fold-resolved baseline results ──────────────────────────────

def make_table_c6() -> None:
    guard_notes = []
    body = []
    for surface in ("Ocean", "Land"):
        tgt = DE_FOLD_TMPL[surface][1]
        body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{surface} "
                    f"({tgt})}}}} \\\\")
        for f in range(5):
            cells = [f"fold {f}"]
            for mod, (tmpls, fname) in _point_metrics_files().items():
                tmpl = tmpls[surface] if isinstance(tmpls[surface], str) \
                    else tmpls[surface][0]
                g = _fold_global(REPO / tmpl.format(f=f) / fname)
                cells += [f"{g['rmse']:.3f}", f"{g['r2']:.3f}"]
            g = _ridge_guarded_fold(surface, f)
            star = "$^{*}$" if g["n_guarded"] else ""
            cells += [f"{g['rmse']:.3f}{star}", f"{g['r2']:.3f}"]
            if g["n_guarded"]:
                guard_notes.append(f"{surface.lower()} fold {f}: "
                                   f"{g['n_guarded']}")
            body.append(" & ".join(cells) + " \\\\")
        if surface != "Land":
            body.append("\\middlehline")
    _tex_table(
        "tabC6_fold_resolved_baselines",
        "Fold-resolved held-out metrics (RMSE in ppm, $R^2$) for the deep "
        "ensemble, XGBoost, and ridge baselines under the shared "
        "date-blocked folds. Ridge values apply the production output "
        "guard of Table~C5.",
        "tab:fold-resolved-baselines", "lrrrrrr",
        ["& \\multicolumn{2}{c}{DE} & \\multicolumn{2}{c}{XGBoost} & "
         "\\multicolumn{2}{c}{Ridge} \\\\",
         "Fold & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ \\\\"],
        body, size="small",
        notes="$^{*}$Folds where the guard withheld ridge corrections "
              f"($|\\mu| > {ANOMALY_GUARD_PPM:g}$\\,ppm): "
              + "; ".join(guard_notes) + " footprint(s). Unguarded, land "
              "fold 2 is dominated by a single out-of-domain footprint "
              "(raw $|\\mu| \\sim 10^{6}$\\,ppm).")


# ── C6 + Table C7: held-out CV feature-set ablation (2026-07-17 variants) ─

ABLATION_VARIANTS = ["no_spec", "no_contam", "no_xco2", "no_xco2_and_spec",
                     "no_contam_and_xco2"]
ABLATION_LABELS = {
    "full": "full", "no_spec": "$-$spec", "no_contam": "$-$contam",
    "no_xco2": "$-$xco2", "no_xco2_and_spec": "$-$xco2$-$spec",
    "no_contam_and_xco2": "$-$xco2$-$contam",
}


def _variant_fold_metrics(surface: str, variant: str) -> list:
    s = surface.lower()
    rr = DE_FOLD_TMPL[surface][1]
    if variant == "full":
        tmpl = DE_FOLD_TMPL[surface][0]
    else:
        tmpl = (f"results/model_deep_ensemble/de_{s}_{variant}"
                f"_prof_foldpca_{rr}_f{{f}}")
    return [_fold_global(REPO / tmpl.format(f=f)
                         / "de_mondrian_date_kfold_metrics.json")
            for f in range(5)]


def make_table_c7() -> None:
    body = []
    for surface in ("Ocean", "Land"):
        body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{surface} "
                    f"({DE_FOLD_TMPL[surface][1]})}}}} \\\\")
        full_rmse = None
        for variant in ["full"] + ABLATION_VARIANTS:
            gs = _variant_fold_metrics(surface, variant)
            rmse = np.array([g["rmse"] for g in gs])
            r2 = np.array([g["r2"] for g in gs])
            if variant == "full":
                full_rmse = rmse.mean()
                delta = "—"
            else:
                delta = f"{rmse.mean() - full_rmse:+.3f}"
            body.append(
                f"{ABLATION_LABELS[variant]} & "
                f"${rmse.mean():.3f} \\pm {rmse.std():.3f}$ & "
                f"${r2.mean():.3f} \\pm {r2.std():.3f}$ & {delta} \\\\")
        if surface != "Land":
            body.append("\\middlehline")
    _tex_table(
        "tabC7_cv_ablation",
        "Held-out date-blocked cross-validation of the feature-set "
        "ablation: fold mean $\\pm$ std RMSE (ppm) and $R^2$ against the "
        "anomaly target for the production deep ensemble (full) and the "
        "variants retrained with one feature group removed (identical "
        "architecture, regularization, and folds). $\\Delta$RMSE is the "
        "fold-mean change versus full. The TCCON-side ablation of the same "
        "variants is Table~2.",
        "tab:cv-ablation", "lccr",
        ["Variant & RMSE & $R^2$ & $\\Delta$RMSE \\\\"],
        body, size="small",
        notes="Land fold 4 of the $-$contam and $-$xco2$-$contam variants "
              "carries the pre-regularization checkpoint (a healthy fold in "
              "both trainings; see the 2026-07-17 ablation report).")


ALL_ITEMS = {
    "figC1": make_c1, "figC2": make_c2, "internal_split": make_c3a,
    "figC4": make_c4, "figC5": make_c5, "figC7": make_c7,
    "tabC1": make_table_c1, "tabC2": make_table_c2, "tabC3": make_table_c3,
    "tabC4": make_table_c4, "tabC5": make_table_c5, "tabC6": make_table_c6,
    "tabC7": make_table_c7,
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(ALL_ITEMS),
                    help="generate only these items (default: all)")
    args = ap.parse_args()
    apply_manuscript_style()
    for name in (args.only or ALL_ITEMS):
        ALL_ITEMS[name]()
