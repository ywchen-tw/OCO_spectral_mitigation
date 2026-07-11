"""feature_importance_figure.py — manuscript figures for the 3-model
permutation feature importance (models/feature_importance.py output).

Reads the fold-aggregated CSVs (median ΔRMSE ± fold sd) from
results/model_comparison/feature_importance/{ocean,land}/ and renders:

  fi_groups.{png,pdf}          2×2 grouped horizontal bars — rows ocean/land,
                               cols global/near-cloud, block-level ΔRMSE
  fi_features_{surface}.{png,pdf}  top-N per-feature panels (global | near-cloud)

Colors: Okabe–Ito (CVD-safe), DE kept at the same blue as
smoother_null_figure.py so the deep ensemble is one hue across the paper.

Usage:  python workspace/feature_importance_figure.py [--top 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_manuscript_style, panel_label

FI_DIR = Path(__file__).parents[1] / "results/model_comparison/feature_importance"

# Okabe–Ito; C_DE matches smoother_null_figure.py so DE is one hue paper-wide.
MODELS = [("de", "Deep ensemble", "#0072B2"),
          ("xgb", "XGBoost (mean)", "#D55E00"),
          ("linreg", "Ridge", "#009E73")]
STRATA = [("global", "all held soundings"),
          ("nearcloud", "near-cloud (≤ 10 km)")]
# Fixed display names for the blocks (order set per surface by DE global ΔRMSE).
BLOCK_LABELS = {
    "xco2": "XCO2 departure", "spec": "spectral cumulants",
    "contam": "contamination", "geometry": "geometry",
    "profile": "profile EOFs", "met_other": "met / retrieval state",
    "fp_onehot": "footprint one-hot",
}


def _load(surface: str, model: str) -> pd.DataFrame:
    p = FI_DIR / surface / f"importance_{model}_{surface}_agg.csv"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run models.feature_importance --aggregate first")
    return pd.read_csv(p)


def _grouped_barh(ax, names: list, per_model: dict, ylabels: list) -> None:
    """Three thin horizontal bars per category, 2px-equivalent gaps, sd whiskers."""
    n, m = len(names), len(MODELS)
    bar_h, gap = 0.24, 0.04
    y0 = np.arange(n)[::-1]                       # top category first
    for j, (mk, label, color) in enumerate(MODELS):
        vals = np.array([per_model[mk].get(nm, (np.nan, np.nan))[0] for nm in names])
        sds = np.array([per_model[mk].get(nm, (np.nan, np.nan))[1] for nm in names])
        y = y0 + (m - 1 - j - (m - 1) / 2) * (bar_h + gap)
        ax.barh(y, vals, height=bar_h, color=color, label=label,
                xerr=np.where(np.isfinite(sds), sds, 0.0),
                error_kw=dict(elinewidth=0.7, capsize=1.5, capthick=0.7,
                              ecolor="#333333"))
    ax.set_yticks(y0)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.4, color="#cccccc", alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure_groups(out_dir: Path) -> None:
    """2×2 block-level panel: rows = surface, cols = stratum."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharey="row",
                             constrained_layout=True)
    tags = iter(["(a)", "(b)", "(c)", "(d)"])
    for i, surface in enumerate(("ocean", "land")):
        data = {mk: _load(surface, mk) for mk, _, _ in MODELS}
        g0 = data["de"]
        order = (g0[(g0.stratum == "global") & (g0.scope == "group")]
                 .sort_values("delta_rmse", ascending=False)["name"].tolist())
        for k, (stratum, sub) in enumerate(STRATA):
            per_model = {}
            for mk, _, _ in MODELS:
                d = data[mk]
                d = d[(d.stratum == stratum) & (d.scope == "group")]
                per_model[mk] = {r["name"]: (r["delta_rmse"], r["fold_sd"])
                                 for _, r in d.iterrows()}
            ax = axes[i, k]
            _grouped_barh(ax, order, per_model,
                          [BLOCK_LABELS.get(nm, nm) for nm in order])
            panel_label(ax, f"{next(tags)} {surface}, {sub}", size=9.0)
            if i == 1:
                ax.set_xlabel(r"permutation $\Delta$RMSE (ppm)")
    axes[0, 0].legend(loc="lower right", frameon=False, handlelength=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fi_groups.{ext}")
    plt.close(fig)
    print(f"[fi] {out_dir}/fi_groups.png|pdf")


def figure_features(surface: str, out_dir: Path, top: int) -> None:
    """Top-N per-feature panels (ranked by DE global ΔRMSE), global | near-cloud."""
    data = {mk: _load(surface, mk) for mk, _, _ in MODELS}
    g0 = data["de"]
    order = (g0[(g0.stratum == "global") & (g0.scope == "feature")]
             .sort_values("delta_rmse", ascending=False)["name"].tolist()[:top])
    grp = (g0[g0.scope == "feature"].drop_duplicates("name")
           .set_index("name")["group"].to_dict())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * top + 1.3), sharey=True,
                             constrained_layout=True)
    for k, (stratum, sub) in enumerate(STRATA):
        per_model = {}
        for mk, _, _ in MODELS:
            d = data[mk]
            d = d[(d.stratum == stratum) & (d.scope == "feature")]
            per_model[mk] = {r["name"]: (r["delta_rmse"], r["fold_sd"])
                             for _, r in d.iterrows()}
        ax = axes[k]
        # sharey: the last panel's yticklabels win, so pass the same tagged
        # labels for both panels.
        labels = [f"{nm}  [{BLOCK_LABELS.get(grp.get(nm, ''), grp.get(nm, ''))}]"
                  for nm in order]
        _grouped_barh(ax, order, per_model, labels)
        panel_label(ax, f"({'ab'[k]}) {surface}, {sub}", size=9.0)
        ax.set_xlabel(r"permutation $\Delta$RMSE (ppm)")
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fi_features_{surface}.{ext}")
    plt.close(fig)
    print(f"[fi] {out_dir}/fi_features_{surface}.png|pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-importance manuscript figures")
    parser.add_argument("--top", type=int, default=12,
                        help="features in the per-feature panels (ranked by DE global)")
    args = parser.parse_args()
    apply_manuscript_style(base_size=8.5)
    out_dir = FI_DIR
    figure_groups(out_dir)
    for surface in ("ocean", "land"):
        figure_features(surface, out_dir, args.top)


if __name__ == "__main__":
    main()
