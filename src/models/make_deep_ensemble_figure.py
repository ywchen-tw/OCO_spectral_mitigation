"""Publication schematic for the deep-ensemble MLP and its prediction intervals.

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

# Manuscript notation: the predicted anomaly is written with a hat over the
# target symbol (Sect. 6, Eq. S17), so mu is not used for it.
DX = r"\widehat{\Delta\mathrm{X}}"


# 2026-10-11 print-size redraw.  The figure is now drawn at its print width
# (6.50 in, the JQSRT text width) instead of 8.45 x 3.55 in shrunk to 77 %.
# Panels (a) and (b) are stacked so every box can carry 9 pt text; all
# coordinates below are in INCHES on the page (each panel axes spans the
# full figure width with 1 data unit = 1 in).  Boxes, arrows and wording
# are unchanged from the side-by-side version.
FIG_W = 6.44        # + 2 x 0.03 in save padding = 6.50 in
FS_BOX = 9.0        # text inside boxes
FS_NOTE = 8.5       # grey explanatory notes
FS_TITLE = 9.5      # panel titles (bold)
FS_PANEL = 10.5     # panel letters (bold)
ARROW_MS = 9.0      # arrow-head size in points


def _box(ax, x, y, w, h, text, facecolor, *, fontsize=FS_BOX, lw=0.8,
         linestyle="-", fontweight="normal", linespacing=1.15):
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
    txt = ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=linespacing,
        zorder=3,
    )
    _BOXES.append((patch, txt))
    return patch


# (patch, text) pairs drawn so far; build() checks every label fits its box.
_BOXES: list = []


def _arrow(ax, x0, y0, x1, y1, *, lw=0.8, linestyle="-", color=ARROW,
           mutation_scale=ARROW_MS, connectionstyle="arc3"):
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


def _small_note(ax, x, y, text, *, ha="left", va="top", fontsize=FS_NOTE):
    lines = text.split("\n")
    if len(lines) == 1:
        ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=C_NOTE)
        return
    # Multi-line notes: one text per line on a fixed baseline grid (data
    # units are inches), so the spacing does not vary with descenders.
    step = fontsize * 1.22 / 72
    if va == "top":
        base0 = y - fontsize * 0.78 / 72
    else:                                   # "center"
        base0 = y + 0.5 * (len(lines) - 1) * step - fontsize * 0.36 / 72
    for i, line in enumerate(lines):
        ax.text(x, base0 - i * step, line, ha=ha, va="baseline",
                fontsize=fontsize, color=C_NOTE)


def _heading(ax, top, tag, title):
    ax.text(0.0, top, tag, fontsize=FS_PANEL, fontweight="bold",
            ha="left", va="top")
    ax.text(0.36, top - 0.01, title, fontsize=FS_TITLE, fontweight="bold",
            ha="left", va="top")


# Panel heights (in) for the two variants of panel (a), and for panel (b).
H_A = {False: 1.36, True: 2.00}
# 2026-10-14: the JQSRT SI no longer reports the 90 % prediction intervals,
# so panel (b) shows only the ensemble mixture.  SHOW_INTERVALS = True
# restores the tuning-block / interval-width row.
SHOW_INTERVALS = False
H_B = 2.16 if SHOW_INTERVALS else 1.30


def _member_panel(ax, show_cloud_head: bool) -> None:
    top = H_A[show_cloud_head]
    _heading(ax, top, "(a)", "One network")

    # main chain, centred on yc; hidden blocks carry four 9 pt lines
    yc = top - 0.63 if not show_cloud_head else top - 0.91
    bh = 0.70
    _box(ax, 0.05, yc - bh / 2, 0.86, bh, "input\npredictors", C_INPUT)
    _box(ax, 1.28, yc - bh / 2, 0.92, bh,
         "Linear 64\nLayerNorm\nReLU\nDropout 0.1", C_BODY)
    _box(ax, 2.57, yc - bh / 2, 0.92, bh,
         "Linear 32\nLayerNorm\nReLU\nDropout 0.1", C_BODY)
    _arrow(ax, 0.91, yc, 1.28, yc)
    _arrow(ax, 2.20, yc, 2.57, yc)

    # h sits above the single arrow, or between the two forked arrows
    ax.text(3.80, yc + (0.0 if show_cloud_head else 0.13), r"$h$",
            fontsize=10, fontstyle="italic", ha="center", va="center")

    hw, hh = 0.98, 0.48            # head box
    hx = 4.12
    ow, oh = 0.92, 0.27            # output boxes
    ox = FIG_W - 0.04 - ow
    if show_cloud_head:
        my, cy = yc + 0.42, yc - 0.42          # main / cloud head centres
        _box(ax, hx, my - hh / 2, hw, hh, "main head\nLinear -> 2", C_HEAD)
        _box(ax, hx, cy - hh / 2, hw, hh, "cloud head\nLinear -> 1",
             C_HEAD, linestyle=(0, (3, 2)))
        _arrow(ax, 3.49, yc, hx, my, lw=0.95)
        _arrow(ax, 3.49, yc, hx, cy, lw=0.9, linestyle=(0, (3, 2)))
        _box(ax, ox, my + 0.04, ow, oh, rf"${DX}_m$", C_OUTPUT)
        _box(ax, ox, my - 0.04 - oh, ow, oh, r"$\log\hat\sigma_m^2$",
             C_OUTPUT)
        _box(ax, ox, cy - 0.22, ow, 0.44, "near-cloud\nlogit", C_OUTPUT,
             linestyle=(0, (3, 2)))
        _arrow(ax, hx + hw, my + 0.06, ox, my + 0.04 + oh / 2)
        _arrow(ax, hx + hw, my - 0.06, ox, my - 0.04 - oh / 2)
        _arrow(ax, hx + hw, cy, ox, cy, linestyle=(0, (3, 2)))
        note = (
            "Each hidden block is Linear -> LayerNorm -> ReLU -> Dropout "
            "(p = 0.1). Dropout is used in training only.\n"
            "The second output is limited to [-10, 10] and read "
            "as the log variance. "
            "Dashed path: optional auxiliary near-cloud classifier "
            "enabled only when --cloud_aux_weight > 0."
        )
        note_y = cy - hh / 2 - 0.10
    else:
        _box(ax, hx, yc - hh / 2, hw, hh, "head\nLinear -> 2", C_HEAD)
        _arrow(ax, 3.49, yc, hx, yc, lw=0.95)
        _box(ax, ox, yc + 0.04, ow, oh, rf"${DX}_m$", C_OUTPUT)
        _box(ax, ox, yc - 0.04 - oh, ow, oh, r"$\log\hat\sigma_m^2$",
             C_OUTPUT)
        _arrow(ax, hx + hw, yc + 0.06, ox, yc + 0.04 + oh / 2)
        _arrow(ax, hx + hw, yc - 0.06, ox, yc - 0.04 - oh / 2)
        note = (
            "Each hidden block is Linear -> LayerNorm -> ReLU -> Dropout "
            "(p = 0.1). Dropout is used in training only.\n"
            "The second output is limited to [-10, 10] and read "
            "as the log variance."
        )
        note_y = yc - bh / 2 - 0.10

    _small_note(ax, 0.05, note_y, note)


def _ensemble_panel(ax) -> None:
    top = H_B
    _heading(ax, top, "(b)", "Ensemble mixture and interval widths"
             if SHOW_INTERVALS else "Ensemble mixture")

    # ---- top row: member stack -> mixture -> mixture moments ------------
    yt = top - 0.72                       # centre of the top row
    _small_note(ax, 0.05, top - 0.25, "independent seeds")
    sw, sh, so = 0.80, 0.46, 0.07         # stacked member boxes
    for i, offset in enumerate((2 * so, so, 0.0)):
        _box(ax, 0.05 + offset, yt - sh / 2 - so + offset, sw, sh,
             "network" if i == 2 else "", C_BODY)
    ax.text(0.05 + sw / 2, yt - sh / 2 - so - 0.06, r"$m = 1,\ldots,M$",
            fontsize=9.0, color=C_NOTE, ha="center", va="top")
    stack_r = 0.05 + sw + 2 * so

    mx, mw, mh = stack_r + 0.30, 0.90, 0.50
    _box(ax, mx, yt - mh / 2, mw, mh, "Gaussian\nmixture", C_ENSEMBLE)
    _arrow(ax, stack_r, yt, mx, yt)

    fx, fw, fh = mx + mw + 0.30, 2.62, 0.78   # 0.74 overflowed the sum subscripts
    _box(
        ax, fx, yt - fh / 2, fw, fh,
        rf"${DX} = M^{{-1}}\sum_m {DX}_m$" "\n"
        rf"$\hat\sigma^{{2}}=M^{{-1}}\sum_m(\hat\sigma_m^2+{DX}_m^2)-{DX}^{{2}}$",
        C_OUTPUT,
        linespacing=2.1,      # room for the sum subscript and superscripts
    )
    _arrow(ax, mx + mw, yt, fx, yt)
    if not SHOW_INTERVALS:
        return

    # ---- bottom row: tuning block -> interval-width setting -> variants -
    ow, oh, og = 1.18, 0.25, 0.07         # three output boxes, right edge
    ox = FIG_W - 0.04 - ow
    oy = [0.05 + 2 * (oh + og), 0.05 + (oh + og), 0.05]   # lower-left y
    yb = oy[1] + oh / 2                   # centre of the bottom row
    iw, ih = 1.22, 0.52
    ix = ox - 0.40 - iw
    _box(ax, ix, yb - ih / 2, iw, ih, "interval-width\nsetting", C_CALIB)
    tw, th = 0.86, 0.48
    tx = ix - 0.42 - tw
    _box(ax, tx, yb - th / 2, tw, th, "tuning\nblock", C_INPUT)
    _arrow(ax, tx + tw, yb, ix, yb)

    # point correction flows down from the mixture moments
    ax_x = ix + iw - 0.30
    _arrow(ax, ax_x, yt - fh / 2, ax_x, yb + ih / 2)
    _small_note(ax, ax_x - 0.08, 0.5 * (yt - fh / 2 + yb + ih / 2),
                rf"point correction ${DX}$", ha="right", va="center")

    _box(ax, ox, oy[0], ow, oh, "raw Gaussian", C_OUTPUT)
    _box(ax, ox, oy[1], ow, oh, "90 % interval", C_OUTPUT)
    _box(ax, ox, oy[2], ow, oh, "Mondrian bins", C_HEADLINE,
         fontweight="bold")
    for y in oy:
        _arrow(ax, ix + iw, yb + 0.6 * (y + oh / 2 - yb), ox, y + oh / 2)

    _small_note(
        ax,
        0.05,
        0.05 + 0.47 * (3 * oh + 2 * og) + 0.01,
        "All interval variants share the same\n"
        "ensemble mean. Widths are set per\n"
        "decile of the predicted anomaly, so no\n"
        "cloud information is needed at inference.",
        va="center",
    )


def _check_boxes(fig) -> None:
    """Fail loudly if any box label overflows its box (render-time check)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for patch, txt in _BOXES:
        if not txt.get_text():
            continue
        pb = patch.get_window_extent(r)
        tb = txt.get_window_extent(r)
        if (tb.x0 < pb.x0 + 2 or tb.x1 > pb.x1 - 2
                or tb.y0 < pb.y0 + 1 or tb.y1 > pb.y1 - 1):
            bad.append(txt.get_text().replace("\n", " | "))
    if bad:
        raise RuntimeError("box labels overflow: " + "; ".join(bad))


def build(show_cloud_head: bool, basename: str, out_dir: Path, *, dpi: int,
          formats: tuple[str, ...]) -> list[Path]:
    apply_manuscript_style(base_size=9.0)
    _BOXES.clear()
    gap = 0.22
    h_a, h_b = H_A[show_cloud_head], H_B
    fig_h = h_a + gap + h_b
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax_a = fig.add_axes([0, (h_b + gap) / fig_h, 1, h_a / fig_h])
    ax_b = fig.add_axes([0, 0, 1, h_b / fig_h])
    for ax, h in ((ax_a, h_a), (ax_b, h_b)):
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, h)
        ax.axis("off")

    _member_panel(ax_a, show_cloud_head=show_cloud_head)
    _ensemble_panel(ax_b)
    _check_boxes(fig)

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
    parser.add_argument("--basename", default=None,
                        help="Override the output basename (single-variant "
                             "runs only, i.e. requires --only cloud|no-cloud); "
                             "used for the manuscript copy "
                             "(fig02_deep_ensemble_architecture).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formats = tuple(f.strip().lower().lstrip(".") for f in args.formats.split(",")
                    if f.strip())
    if not formats:
        raise ValueError("--formats must include at least one file extension")
    if args.basename and args.only == "both":
        raise ValueError("--basename requires --only cloud|no-cloud")

    if args.only in ("both", "cloud"):
        build(
            show_cloud_head=True,
            basename=args.basename or "deep_ensemble_architecture",
            out_dir=args.out_dir,
            dpi=args.dpi,
            formats=formats,
        )
    if args.only in ("both", "no-cloud"):
        build(
            show_cloud_head=False,
            basename=args.basename or "deep_ensemble_architecture_no_cloud",
            out_dir=args.out_dir,
            dpi=args.dpi,
            formats=formats,
        )


if __name__ == "__main__":
    main()
