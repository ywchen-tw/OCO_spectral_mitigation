"""Publication figure for the deep-ensemble MLP + conformal model (deep_ensemble.py).

Draws a two-panel schematic:
  (a) a single Gaussian-head MLP member
  (b) the M-member ensemble mixture + conformal calibration pipeline

Two variants are written to results/figures/ (vector PDF + 300-DPI PNG):
  deep_ensemble_architecture            — with the optional multi-task cloud head
  deep_ensemble_architecture_no_cloud   — single-task head only (the model actually used)

Run:  python -m src.models.make_deep_ensemble_figure
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── style: manuscript-friendly, serif, muted palette ────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})

C_IN = "#dfe7f0"      # inputs
C_HID = "#c7dbe6"     # hidden layers
C_HEAD = "#f6d9b0"    # heads
C_ENS = "#cfe6d4"     # ensemble / mixture
C_CONF = "#f3ccd0"    # conformal
C_OUT = "#e8e2f0"     # outputs
C_STAR = "#fbe9b0"    # headline output
EDGE = "#333333"
BSTYLE = "round,pad=0.02,rounding_size=0.05"


def box(ax, x, y, w, h, text, fc, fontsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=BSTYLE,
                                linewidth=0.9, edgecolor=EDGE, facecolor=fc,
                                mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, lw=1.0, ls="-", color=EDGE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, lw=lw, linestyle=ls, color=color))


def panel_member(axA, show_cloud_head=True):
    """Panel (a): a single Gaussian-head MLP member."""
    axA.set_title("(a) One ensemble member", fontsize=9, loc="left", pad=4)

    box(axA, 0.05, 4.15, 1.8, 1.7, "input\nfeatures\n$x$", C_IN, fontsize=7.6)
    box(axA, 2.55, 4.15, 1.8, 1.7, "Linear 64\nReLU", C_HID, fontsize=7.4)
    box(axA, 5.05, 4.15, 1.8, 1.7, "Linear 32\nReLU", C_HID, fontsize=7.4)
    arrow(axA, 1.85, 5.0, 2.55, 5.0)                  # input -> Linear 64
    arrow(axA, 4.35, 5.0, 5.05, 5.0)                  # Linear 64 -> Linear 32

    if show_cloud_head:
        box(axA, 7.65, 6.35, 1.9, 1.25, "head\n(Linear$\\to$2)", C_HEAD, fontsize=7.2)
        box(axA, 7.65, 2.55, 1.9, 1.25, "cloud head\n(Linear$\\to$1)", C_HEAD, fontsize=6.9)
        arrow(axA, 6.85, 5.0, 7.65, 6.95, lw=1.2)     # body -> main head
        arrow(axA, 6.85, 5.0, 7.65, 3.1, lw=1.2, ls=(0, (4, 3)))  # -> cloud head
        axA.text(7.0, 6.2, "$h$", fontsize=9.5, style="italic", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        # main-head outputs
        arrow(axA, 9.55, 7.15, 10.05, 7.15)
        arrow(axA, 9.55, 6.75, 10.05, 6.75)
        axA.text(10.2, 7.15, r"$\mu$", ha="left", va="center", fontsize=9)
        axA.text(10.2, 6.72, r"$\log\sigma^2$", ha="left", va="center", fontsize=7.0)
        # cloud-head output
        arrow(axA, 9.55, 3.15, 10.05, 3.15, ls=(0, (4, 3)))
        axA.text(10.2, 3.15, "cloud\nlogit", ha="left", va="center", fontsize=6.4)
        axA.text(0.3, 1.55, "— solid: single-task path (default)\n"
                            "-- dashed: optional multi-task cloud head\n"
                            r"$\log\sigma^2\!\to\!\log$-scale for Student-$t$ loss",
                 fontsize=6.4, va="top", color="#444444")
    else:
        # single head, centred on the body mid-line
        box(axA, 7.65, 4.375, 1.9, 1.25, "head\n(Linear$\\to$2)", C_HEAD, fontsize=7.2)
        arrow(axA, 6.85, 5.0, 7.65, 5.0, lw=1.2)      # body -> head
        axA.text(7.25, 5.55, "$h$", fontsize=9.5, style="italic", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        arrow(axA, 9.55, 5.25, 10.05, 5.25)
        arrow(axA, 9.55, 4.75, 10.05, 4.75)
        axA.text(10.2, 5.25, r"$\mu$", ha="left", va="center", fontsize=9)
        axA.text(10.2, 4.72, r"$\log\sigma^2$", ha="left", va="center", fontsize=7.0)
        axA.text(0.3, 2.2, r"$\log\sigma^2\!\to\!\log$-scale for Student-$t$ loss",
                 fontsize=6.4, va="top", color="#444444")


def panel_ensemble(axB):
    """Panel (b): ensemble mixture + conformal calibration (identical in both variants)."""
    axB.set_title("(b) Ensemble mixture + conformal calibration",
                  fontsize=9, loc="left", pad=4)

    # stacked members
    for i, dy in enumerate([0.0, 0.26, 0.52]):
        box(axB, 0.4 + dy, 6.9 - dy, 1.85, 1.05,
            "MLP\nmember" if i == 2 else "", C_HID, fontsize=7.2)
    axB.text(1.3, 6.05, r"$m=1\ldots M$", ha="center", fontsize=6.8, color="#444")
    axB.text(0.35, 8.75, "independent seeds\n(optional diverse archs)",
             fontsize=6.5, va="bottom", color="#444444")

    # mixture
    box(axB, 3.15, 7.05, 1.95, 1.25, "Gaussian\nmixture", C_ENS, fontsize=8)
    arrow(axB, 2.55, 7.55, 3.15, 7.55)

    # mixture output (point pred + variance)
    box(axB, 5.85, 7.0, 3.85, 1.35,
        r"$\mu^{\ast}=\overline{\mu_m}$" "\n"
        r"$\sigma^{\ast 2}=\overline{\sigma_m^2+\mu_m^2}-\mu^{\ast 2}$",
        C_OUT, fontsize=7.6)
    arrow(axB, 5.1, 7.65, 5.85, 7.65)
    axB.text(7.77, 6.72, r"point prediction $\hat y=\mu^{\ast}$",
             ha="center", va="top", fontsize=6.6, color="#444")

    # calibration block feeding conformal
    box(axB, 3.05, 4.15, 2.15, 1.15, "calibration\nblock (held-out)", C_IN, fontsize=7.2)

    # conformal
    box(axB, 5.85, 3.9, 2.4, 1.55, "conformal\nrecalibration", C_CONF, fontsize=8)
    arrow(axB, 7.05, 7.0, 7.05, 5.45)                 # mu*,sigma* -> conformal
    arrow(axB, 5.2, 4.7, 5.85, 4.7)                   # calib -> conformal

    # three interval outputs
    labels = [("raw Gaussian", C_OUT),
              ("split conformal", C_OUT),
              (r"Mondrian (per-bin) $\star$", C_STAR)]
    ys = [2.55, 1.6, 0.65]
    arrow(axB, 7.05, 3.9, 7.05, 3.35)
    for (lab, fc), yy in zip(labels, ys):
        box(axB, 5.55, yy, 3.9, 0.72, lab, fc, fontsize=7.2)
    axB.text(5.5, 0.25, r"90% intervals (share $\mu^{\ast}$);  $\star$ = headline",
             fontsize=6.5, va="top", color="#444444")


def build(show_cloud_head, basename):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.4),
                                   gridspec_kw={"width_ratios": [1.2, 1.2]})
    for ax in (axA, axB):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
    axA.set_xlim(0, 11)   # extra room so the body→head arrows have length
    panel_member(axA, show_cloud_head=show_cloud_head)
    panel_ensemble(axB)
    fig.tight_layout(pad=0.5)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    build(show_cloud_head=True, basename="deep_ensemble_architecture")
    build(show_cloud_head=False, basename="deep_ensemble_architecture_no_cloud")
