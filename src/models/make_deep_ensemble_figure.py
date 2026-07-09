"""Publication schematic for the deep-ensemble MLP + conformal model.

The figure mirrors ``deep_ensemble.py`` and writes manuscript-ready vector and
raster outputs using the shared AMT style (Arial text and Arial mathtext).

Default outputs in ``results/figures``:
  deep_ensemble_architecture.pdf/.png
  deep_ensemble_architecture_no_cloud.pdf/.png

Run:
  python -m src.models.make_deep_ensemble_figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from plot_style import apply_manuscript_style, panel_label  # noqa: E402


# Muted, print-friendly colors.  The palette uses hue changes as well as lightness
# changes so the schematic remains legible when printed in grayscale.
C_INPUT = "#e7edf3"
C_BODY = "#d9e8d4"
C_HEAD = "#f6e0b8"
C_ENSEMBLE = "#cfe5ee"
C_CALIB = "#ead5e7"
C_OUTPUT = "#f4f4f4"
C_HEADLINE = "#fff0b8"
C_NOTE = "#555555"
EDGE = "#2f2f2f"
ARROW = "#333333"
BOXSTYLE = "round,pad=0.025,rounding_size=0.055"


def _box(ax, x, y, w, h, text, facecolor, *, fontsize=6.4, lw=0.8,
         linestyle="-", fontweight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=BOXSTYLE,
        linewidth=lw,
        edgecolor=EDGE,
        facecolor=facecolor,
        linestyle=linestyle,
        mutation_aspect=1,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def _arrow(ax, x0, y0, x1, y1, *, lw=0.8, linestyle="-", color=ARROW,
           mutation_scale=7.5, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        lw=lw,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=1,
    ))


def _small_note(ax, x, y, text, *, ha="left", va="top", fontsize=6.2):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=C_NOTE, linespacing=1.18)


def _member_panel(ax, show_cloud_head: bool) -> None:
    panel_label(ax, "(a)", size=9.0, dx=0.0, dy=1.01)
    ax.text(0.06, 9.55, "One Gaussian-head MLP member",
            fontsize=8.6, fontweight="bold", ha="left", va="top")

    _box(ax, 0.15, 5.15, 1.62, 1.25, "input\nfeatures", C_INPUT,
         fontsize=6.0)
    _box(ax, 2.25, 5.15, 1.62, 1.25, "Linear 64\nReLU", C_BODY,
         fontsize=6.1)
    _box(ax, 4.35, 5.15, 1.62, 1.25, "Linear 32\nReLU", C_BODY,
         fontsize=6.1)
    _arrow(ax, 1.77, 5.78, 2.25, 5.78)
    _arrow(ax, 3.87, 5.78, 4.35, 5.78)

    ax.text(6.26, 6.08, r"$h$", fontsize=8.0, fontstyle="italic",
            ha="center", va="center")

    if show_cloud_head:
        _box(ax, 6.90, 6.55, 1.78, 0.95, "main head\nLinear -> 2",
             C_HEAD, fontsize=5.8)
        _box(ax, 6.90, 3.95, 1.78, 0.95, "cloud head\nLinear -> 1",
             C_HEAD, fontsize=5.8, linestyle=(0, (3, 2)))
        _arrow(ax, 5.97, 5.78, 6.90, 7.02, lw=0.95)
        _arrow(ax, 5.97, 5.78, 6.90, 4.43, lw=0.9, linestyle=(0, (3, 2)))
        _box(ax, 9.55, 6.88, 1.36, 0.47, r"$\mu_m$", C_OUTPUT,
             fontsize=6.2)
        _box(ax, 9.55, 6.24, 1.36, 0.47, r"$\log\sigma_m^2$", C_OUTPUT,
             fontsize=5.5)
        _box(ax, 9.55, 4.18, 1.36, 0.47, "near-cloud\nlogit", C_OUTPUT,
             fontsize=4.7, linestyle=(0, (3, 2)))
        _arrow(ax, 8.68, 7.03, 9.55, 7.12)
        _arrow(ax, 8.68, 6.75, 9.55, 6.47)
        _arrow(ax, 8.68, 4.43, 9.55, 4.43, linestyle=(0, (3, 2)))
        _small_note(
            ax,
            0.15,
            2.55,
            "Dashed path: optional auxiliary near-cloud classifier\n"
            "enabled only when --cloud_aux_weight > 0.",
        )
    else:
        _box(ax, 6.90, 5.30, 1.78, 0.95, "head\nLinear -> 2",
             C_HEAD, fontsize=5.8)
        _arrow(ax, 5.97, 5.78, 6.90, 5.78, lw=0.95)
        _box(ax, 9.55, 6.03, 1.36, 0.47, r"$\mu_m$", C_OUTPUT,
             fontsize=6.2)
        _box(ax, 9.55, 5.38, 1.36, 0.47, r"$\log\sigma_m^2$", C_OUTPUT,
             fontsize=5.5)
        _arrow(ax, 8.68, 5.88, 9.55, 6.17)
        _arrow(ax, 8.68, 5.66, 9.55, 5.61)

    _small_note(
        ax,
        0.15,
        1.45,
        "The second head output is clamped to [-10, 10].\n"
        "For Student-t loss it is interpreted as log scale;\n"
        "otherwise it is log variance.",
    )


def _ensemble_panel(ax) -> None:
    panel_label(ax, "(b)", size=9.0, dx=0.0, dy=1.01)
    ax.text(0.06, 9.55, "Ensemble mixture and conformal calibration",
            fontsize=8.6, fontweight="bold", ha="left", va="top")

    # Member stack.
    for i, offset in enumerate((0.36, 0.18, 0.0)):
        _box(ax, 0.35 + offset, 6.85 + offset, 1.58, 0.82,
             "MLP\nmember" if i == 2 else "", C_BODY, fontsize=5.8)
    ax.text(1.18, 6.50, r"$m = 1,\ldots,M$", fontsize=6.5,
            color=C_NOTE, ha="center")
    _small_note(ax, 0.22, 8.77, "independent seeds\noptional DE++ widths",
                fontsize=5.9)

    _box(ax, 2.72, 7.00, 1.58, 1.05, "Gaussian\nmixture", C_ENSEMBLE,
         fontsize=6.0)
    _arrow(ax, 2.06, 7.48, 2.72, 7.52)

    _box(
        ax,
        4.72,
        6.83,
        4.18,
        1.38,
        r"$\mu^* = M^{-1}\sum_m \mu_m$" "\n"
        r"$\sigma^{*2}=M^{-1}\sum_m(\sigma_m^2+\mu_m^2)-\mu^{*2}$",
        C_OUTPUT,
        fontsize=5.0,
    )
    _arrow(ax, 4.30, 7.53, 4.72, 7.53)
    _small_note(ax, 6.80, 6.58, r"point correction $\hat{y}=\mu^*$",
                ha="center", fontsize=5.6)

    _box(ax, 1.62, 4.23, 1.95, 0.95, "calibration\nblock", C_INPUT,
         fontsize=6.1)
    _box(ax, 4.72, 3.95, 2.20, 1.30, "conformal\nrecalibration", C_CALIB,
         fontsize=6.3)
    _arrow(ax, 6.65, 6.83, 6.65, 5.25)
    _arrow(ax, 3.52, 4.70, 4.72, 4.70)

    _box(ax, 7.78, 4.82, 2.35, 0.58, "raw Gaussian", C_OUTPUT,
         fontsize=5.6)
    _box(ax, 7.78, 3.96, 2.35, 0.58, "split conformal", C_OUTPUT,
         fontsize=5.6)
    _box(ax, 7.78, 3.10, 2.35, 0.58, "Mondrian bins", C_HEADLINE,
         fontsize=5.6, fontweight="bold")
    _arrow(ax, 6.92, 4.78, 7.78, 5.11)
    _arrow(ax, 6.92, 4.60, 7.78, 4.25)
    _arrow(ax, 6.92, 4.42, 7.78, 3.39)

    _small_note(
        ax,
        0.22,
        1.72,
        "All interval variants share the same ensemble mean.\n"
        "Mondrian conformal uses per-bin residual quantiles\n"
        "from predicted mean deciles or a physical column such as cloud distance.",
        fontsize=6.2,
    )


def build(show_cloud_head: bool, basename: str, out_dir: Path, *, dpi: int,
          formats: tuple[str, ...]) -> list[Path]:
    apply_manuscript_style(base_size=8.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.05, 3.55),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.10},
    )
    axes[0].set_xlim(0, 11.15)
    axes[1].set_xlim(0, 10.35)
    for ax in axes:
        ax.set_ylim(0.8, 9.85)
        ax.axis("off")

    _member_panel(axes[0], show_cloud_head=show_cloud_head)
    _ensemble_panel(axes[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        written.append(path)
        print(f"wrote {path}")
    plt.close(fig)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the deep-ensemble architecture schematic."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"),
                        help="Output directory.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Raster DPI for PNG/TIFF outputs.")
    parser.add_argument("--formats", default="pdf,png",
                        help="Comma-separated output formats, e.g. pdf,png,tiff.")
    parser.add_argument("--only", choices=["both", "cloud", "no-cloud"],
                        default="both",
                        help="Which schematic variant to write.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formats = tuple(f.strip().lower().lstrip(".") for f in args.formats.split(",")
                    if f.strip())
    if not formats:
        raise ValueError("--formats must include at least one file extension")

    if args.only in ("both", "cloud"):
        build(
            show_cloud_head=True,
            basename="deep_ensemble_architecture",
            out_dir=args.out_dir,
            dpi=args.dpi,
            formats=formats,
        )
    if args.only in ("both", "no-cloud"):
        build(
            show_cloud_head=False,
            basename="deep_ensemble_architecture_no_cloud",
            out_dir=args.out_dir,
            dpi=args.dpi,
            formats=formats,
        )


if __name__ == "__main__":
    main()
