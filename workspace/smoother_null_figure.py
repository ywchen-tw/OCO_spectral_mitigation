"""smoother_null_figure.py — the M4 figure: smoothing is not correcting.

Reads the per-case tccon_comparison CSVs produced by tccon_comparison_report.py
for (1) the production deep ensemble and (2) the pure-smoother null columns
(smoother_null.py), same flags, and renders one two-panel figure:

  (a) per-case footprint scatter sd, after vs before  — BOTH arms collapse it
      (a smoother trivially wins this metric);
  (b) per-case |bias to TCCON|, after vs before       — the smoother sits on
      the 1:1 line (it preserves the local mean by construction) while the
      deep ensemble falls below it.

So the TCCON bias improvement cannot be an artifact of variance removal.

Usage:
  python workspace/smoother_null_figure.py \
      [--report-dir results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/smoother_null] \
      [--primary-window 30] [--fmt png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})

XCO2 = r"$X_{\mathrm{CO}_2}$"
C_DE = "#0072B2"        # Okabe-Ito blue
C_SM = "#D55E00"        # Okabe-Ito vermillion
DEFAULT_DIR = Path("results/model_comparison/deep_ensemble/"
                   "de_beta_nll_prof_reg_o05l15_m5/smoother_null")


def load_cases(csv_path: Path) -> pd.DataFrame:
    """Headline rows: qf 'all', surface 'all', finite before/after bias."""
    df = pd.read_csv(csv_path)
    if "qf_group" in df.columns:
        df = df[df["qf_group"] == "all"]
    df = df[df["surface"] == "all"]
    df = df[np.isfinite(df["bias_before"]) & np.isfinite(df["bias_after"])]
    return df.set_index(["site", "date"]).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_DIR,
                    help="Dir holding tccon_comparison_{de,smoother_w*}_r100km.csv")
    ap.add_argument("--radius-tag", default="r100km")
    ap.add_argument("--primary-window", type=float, default=30.0,
                    help="Smoother half-width (s) shown in the scatter panels; the "
                         "other windows appear in the printed/annotated summary.")
    ap.add_argument("--windows", default="10,30,100")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rd, tag = args.report_dir, args.radius_tag
    de = load_cases(rd / f"tccon_comparison_de_{tag}.csv")
    windows = [float(w) for w in args.windows.split(",") if w.strip()]
    sm = {w: load_cases(rd / f"tccon_comparison_smoother_w{w:g}_{tag}.csv")
          for w in windows}
    wP = args.primary_window
    smP = sm[wP]

    # same-case alignment (the 'before' columns must agree between runs)
    common = de.index.intersection(smP.index)
    de, smP = de.loc[common], smP.loc[common]
    db = float(np.nanmax(np.abs(de["bias_before"] - smP["bias_before"])))
    if db > 1e-6:
        print(f"WARNING: before-bias mismatch across runs (max {db:.2e} ppm)")
    print(f"{len(common)} common cases; max |Δbias_before| across runs {db:.2e} ppm")

    def agg(frame):
        return dict(
            sd_b=float(frame["orig_sd"].mean()), sd_a=float(frame["corr_sd"].mean()),
            ab_b=float(frame["bias_before"].abs().mean()),
            ab_a=float(frame["bias_after"].abs().mean()),
            rm_b=float(frame["rmse_before"].mean()), rm_a=float(frame["rmse_after"].mean()))

    A_de, A_sm = agg(de), agg(smP)
    print(f"DE        : sd {A_de['sd_b']:.2f}→{A_de['sd_a']:.2f}  |bias| "
          f"{A_de['ab_b']:.2f}→{A_de['ab_a']:.2f}  fp-RMSE {A_de['rm_b']:.2f}→{A_de['rm_a']:.2f} ppm")
    for w in windows:
        s = sm[w].loc[sm[w].index.intersection(common)]
        A = agg(s)
        print(f"smoother w{w:<3g}: sd {A['sd_b']:.2f}→{A['sd_a']:.2f}  |bias| "
              f"{A['ab_b']:.2f}→{A['ab_a']:.2f}  fp-RMSE {A['rm_b']:.2f}→{A['rm_a']:.2f} ppm")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))

    # (a) footprint scatter sd, after vs before
    ax = axes[0]
    lim = max(de["orig_sd"].max(), de["corr_sd"].max(),
              smP["corr_sd"].max()) * 1.06
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.7, zorder=1)
    ax.scatter(de["orig_sd"], de["corr_sd"], s=14, facecolors="none",
               edgecolors=C_DE, linewidths=0.8, label="deep ensemble", zorder=3)
    ax.scatter(smP["orig_sd"], smP["corr_sd"], s=12, marker="^",
               facecolors="none", edgecolors=C_SM, linewidths=0.8,
               label=f"running mean (±{wP:g} s)", zorder=2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"footprint {XCO2} scatter before (ppm)")
    ax.set_ylabel(f"footprint {XCO2} scatter after (ppm)")
    ax.text(0.03, 0.92,
            f"mean {A_de['sd_b']:.2f} → {A_de['sd_a']:.2f} (DE)\n"
            f"mean {A_sm['sd_b']:.2f} → {A_sm['sd_a']:.2f} (smoother)",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")

    # (b) |case bias to TCCON|, after vs before
    ax = axes[1]
    lim = max(de["bias_before"].abs().max(), de["bias_after"].abs().max(),
              smP["bias_after"].abs().max()) * 1.06
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.7, zorder=1)
    ax.scatter(de["bias_before"].abs(), de["bias_after"].abs(), s=14,
               facecolors="none", edgecolors=C_DE, linewidths=0.8, zorder=3)
    ax.scatter(smP["bias_before"].abs(), smP["bias_after"].abs(), s=12,
               marker="^", facecolors="none", edgecolors=C_SM,
               linewidths=0.8, zorder=2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("|case bias to TCCON| before (ppm)")
    ax.set_ylabel("|case bias to TCCON| after (ppm)")
    ax.text(0.03, 0.92,
            f"mean {A_de['ab_b']:.2f} → {A_de['ab_a']:.2f} (DE)\n"
            f"mean {A_sm['ab_b']:.2f} → {A_sm['ab_a']:.2f} (smoother)",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.0, 1.02, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")

    fig.tight_layout()
    out = rd / f"smoother_null_{tag}.{args.fmt}"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
