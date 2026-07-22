"""Manuscript Fig. D1b (Appendix D): date-blocked cross-validation design schematic
(panel b of Fig. D1; panel a is the random-split inflation result).

Companion to the deep-ensemble architecture schematic
(``src/models/make_deep_ensemble_figure.py``) and drawn in the same muted,
print-friendly style.  The figure contrasts a random sounding-level split
(interleaved train/test dates -> leakage, inflated R^2) with the
date-blocked 5-fold protocol actually used for evaluation (contiguous
held-out date blocks plus a fold-local calibration block for early stopping
and conformal calibration).

Fold layout is read from the real ``training_dates.json`` manifests of the
production ocean deep ensemble (``de_ocean_beta_nll_prof_reg_r05_f0..f4``)
when present and already using fold-local calibration; otherwise a
stylized 116-date semi-monthly timeline (2016-2020) with the same block
structure is drawn.

Run:
  python3 manuscript/scripts/make_cv_design_figure.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from plot_style import apply_manuscript_style, panel_label  # noqa: E402


# Muted, print-friendly palette shared with make_deep_ensemble_figure.py.
C_TRAIN = "#d9e8d4"    # pale green   (MLP-body green in the architecture fig)
C_CALIB = "#ead5e7"    # pale mauve   (conformal-calibration mauve)
C_TEST = "#e3a94f"     # deeper amber (head color, darkened for grayscale sep)
C_NOTE = "#555555"
EDGE = "#2f2f2f"
ARROW = "#333333"
GRID = "#d9d9d9"

MANIFEST_TAG = "de_ocean_beta_nll_prof_reg_r05_f{fold}"
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------

def _frac_year(d: date) -> float:
    start = date(d.year, 1, 1).toordinal()
    end = date(d.year + 1, 1, 1).toordinal()
    return d.year + (d.toordinal() - start) / (end - start)


def _load_manifests(model_dir: Path):
    """Return (dates, roles) from real fold manifests, or None if absent.

    dates: sorted list of datetime.date (union over folds)
    roles: (N_FOLDS, n_dates) array of {"train", "calib", "test"}
    """
    folds = []
    for k in range(N_FOLDS):
        path = model_dir / MANIFEST_TAG.format(fold=k) / "training_dates.json"
        if not path.exists():
            return None
        folds.append(json.loads(path.read_text()))

    all_dates = sorted({d for f in folds
                        for key in ("train_dates", "calib_dates", "held_dates")
                        for d in f.get(key, [])})
    dates = [date.fromisoformat(d) for d in all_dates]
    index = {d: i for i, d in enumerate(all_dates)}

    roles = np.full((N_FOLDS, len(dates)), "train", dtype=object)
    for k, f in enumerate(folds):
        for d in f.get("calib_dates", []):
            roles[k, index[d]] = "calib"
        for d in f.get("held_dates", []):
            roles[k, index[d]] = "test"
    if not _roles_have_fold_local_calibration(roles):
        return None
    return dates, roles


def _roles_have_fold_local_calibration(roles: np.ndarray) -> bool:
    """Reject stale manifests from the old shared trailing-calibration design."""
    calib_sets = []
    for row in roles:
        calib_idx = np.flatnonzero(row == "calib")
        test_idx = np.flatnonzero(row == "test")
        if len(calib_idx) == 0 or len(test_idx) == 0:
            return False
        if not np.all(np.diff(calib_idx) == 1):
            return False
        adjacent_after = calib_idx[0] == test_idx[-1] + 1
        adjacent_before = calib_idx[-1] == test_idx[0] - 1
        wrapped_to_start = calib_idx[0] == 0
        if not (adjacent_after or adjacent_before or wrapped_to_start):
            return False
        calib_sets.append(tuple(calib_idx))
    return len(set(calib_sets)) == roles.shape[0]


def _stylized_timeline():
    """Fallback: 116 semi-monthly dates 2016-2020, same block structure."""
    dates = []
    for year in range(2016, 2021):
        for month in range(1, 13):
            dates.append(date(year, month, 1))
            dates.append(date(year, month, 15))
    dates = dates[:116]
    n = len(dates)
    blocks = np.array_split(np.arange(n), N_FOLDS)
    roles = np.full((N_FOLDS, n), "train", dtype=object)
    calib_frac = 0.15
    for k, block in enumerate(blocks):
        lo = int(block[0])
        hi = int(block[-1]) + 1
        roles[k, lo:hi] = "test"
        n_calib = max(1, math.ceil(calib_frac * (n - len(block))))
        n_calib = min(n_calib, n - len(block) - 1)
        if hi + n_calib <= n:
            calib = np.arange(hi, hi + n_calib)
        elif n_calib <= lo:
            calib = np.arange(n_calib)
        else:
            calib = np.arange(lo - n_calib, lo)
        roles[k, calib] = "calib"
    return dates, roles


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _sliver_edges(x: np.ndarray) -> np.ndarray:
    """Cell boundaries between consecutive date positions."""
    mids = 0.5 * (x[:-1] + x[1:])
    half = np.median(np.diff(x)) / 2.0
    return np.concatenate([[x[0] - half], mids, [x[-1] + half]])


def _row(ax, x, y0, h, colors):
    """One timeline row: per-date slivers plus a thin outline."""
    edges = _sliver_edges(x)
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((edges[i], y0), edges[i + 1] - edges[i], h,
                               facecolor=c, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((edges[0], y0), edges[-1] - edges[0], h,
                           facecolor="none", edgecolor=EDGE, lw=0.7, zorder=3))
    return edges


ROLE_COLOR = {"train": C_TRAIN, "calib": C_CALIB, "test": C_TEST}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build(out_dir: Path, *, dpi: int, formats: tuple[str, ...],
          seed: int = 42) -> list[Path]:
    loaded = _load_manifests(ROOT / "results" / "model_deep_ensemble")
    used_real = loaded is not None
    dates, roles = loaded if used_real else _stylized_timeline()
    n = len(dates)
    x = np.array([_frac_year(d) for d in dates])

    # Random sounding-level split: test dates interleaved through the record.
    rng = np.random.default_rng(seed)
    rand_test = np.zeros(n, dtype=bool)
    rand_test[rng.choice(n, size=n // 5, replace=False)] = True

    apply_manuscript_style(base_size=8.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x_lo, x_hi = 2015.92, 2021.08
    ax.set_xlim(2014.55, 2023.05)
    ax.set_ylim(0.3, 12.35)
    ax.axis("off")
    panel_label(ax, "(b)", size=9.0, dx=0.0, dy=1.005)

    h = 0.62
    y_rand = 10.30
    y_fold0 = 7.42
    dy_fold = 1.00
    y_axis = y_fold0 - (N_FOLDS - 1) * dy_fold - 0.62

    # Year gridlines behind everything.
    for yr in range(2016, 2022):
        ax.plot([yr, yr], [y_axis, y_rand + h + 0.72], color=GRID, lw=0.55,
                zorder=0)

    # -- Random split row --------------------------------------------------
    ax.text(x_lo, y_rand + h + 0.78, "Random split (sounding-level)",
            fontsize=7.8, fontweight="bold", ha="left", va="bottom")
    rand_colors = [C_TEST if t else C_TRAIN for t in rand_test]
    _row(ax, x, y_rand, h, rand_colors)
    ax.text(x_lo - 0.14, y_rand + h / 2, "random\nsplit", fontsize=6.4,
            ha="right", va="center", linespacing=1.15)
    ax.text(x_hi + 0.16, y_rand + h / 2,
            "$R^2 = 0.82$\n(inflated)", fontsize=6.8, ha="left", va="center",
            linespacing=1.25)

    # Leakage annotation: point at an interior test date.
    test_idx = np.flatnonzero(rand_test)
    target = test_idx[np.argmin(np.abs(x[test_idx] - 2019.15))]
    ax.text(2016.02, y_rand - 0.42,
            "test days sit between training days: same-day soundings share\n"
            "weather and orbit state, so the random split leaks",
            fontsize=6.4, color=C_NOTE, ha="left", va="top", linespacing=1.25)
    ax.add_patch(FancyArrowPatch(
        (x[target] + 0.55, y_rand - 0.68), (x[target], y_rand - 0.04),
        arrowstyle="-|>", mutation_scale=7.5, lw=0.8, color=ARROW,
        connectionstyle="arc3,rad=-0.25", shrinkA=1.0, shrinkB=0.5, zorder=4))

    # -- Date-blocked folds -------------------------------------------------
    ax.text(x_lo, y_fold0 + h + 0.26, "Date-blocked 5-fold CV (date_kfold)",
            fontsize=7.8, fontweight="bold", ha="left", va="bottom")
    for k in range(N_FOLDS):
        y0 = y_fold0 - k * dy_fold
        _row(ax, x, y0, h, [ROLE_COLOR[r] for r in roles[k]])
        ax.text(x_lo - 0.14, y0 + h / 2, f"fold {k + 1}", fontsize=6.4,
                ha="right", va="center")

    y_mid = y_fold0 - (N_FOLDS - 1) * dy_fold / 2 + h / 2
    ax.text(x_hi + 0.16, y_mid,
            "held-out $R^2$\n0.53 ocean\n0.39 land", fontsize=6.8,
            ha="left", va="center", linespacing=1.3)

    # Calibration-block callout on a later fold, where the calibration block is
    # visually separated from both the random-split note and the section title.
    callout_fold = 3
    callout_y = y_fold0 - callout_fold * dy_fold
    calib_x = x[np.array([r == "calib" for r in roles[callout_fold]])]
    cx = float(calib_x.mean())
    tx, ty = 2020.30, y_fold0 + h + 0.78
    ax.text(tx, ty,
            "fold-local calibration block\n(early stop + conformal)",
            fontsize=6.2, color=C_NOTE, ha="center", va="bottom",
            linespacing=1.2)
    ax.add_patch(FancyArrowPatch(
        (tx, ty - 0.04), (cx, callout_y + h + 0.03),
        arrowstyle="-|>", mutation_scale=7.5, lw=0.8, color=ARROW,
        connectionstyle="arc3,rad=0.15", shrinkA=1.0, shrinkB=0.5, zorder=4))

    # -- Year axis -----------------------------------------------------------
    ax.plot([x_lo, x_hi], [y_axis, y_axis], color=EDGE, lw=0.8, zorder=3)
    for yr in range(2016, 2022):
        ax.plot([yr, yr], [y_axis, y_axis - 0.14], color=EDGE, lw=0.8,
                zorder=3)
        ax.text(yr, y_axis - 0.30, str(yr), fontsize=6.8, ha="center",
                va="top")
    ax.text((x_lo + x_hi) / 2, y_axis - 0.98,
            f"{n} training dates (semi-monthly, 2016$-$2020)",
            fontsize=6.4, color=C_NOTE, ha="center", va="top")

    # -- Legend ---------------------------------------------------------------
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=C_TRAIN, edgecolor=EDGE, lw=0.6),
        Rectangle((0, 0), 1, 1, facecolor=C_CALIB, edgecolor=EDGE, lw=0.6),
        Rectangle((0, 0), 1, 1, facecolor=C_TEST, edgecolor=EDGE, lw=0.6),
    ]
    ax.legend(handles, ["train", "calibration", "held-out test"],
              loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3,
              frameon=False, fontsize=6.8, handlelength=1.5,
              handleheight=0.9, columnspacing=1.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"figD1b_cv_design.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        written.append(path)
        print(f"wrote {path}  (manifests: {'real' if used_real else 'stylized'})")
    plt.close(fig)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the date-blocked CV design schematic (Fig. D1b).")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "manuscript" / "figures",
                        help="Output directory.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", default="png,pdf",
                        help="Comma-separated output formats.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formats = tuple(f.strip().lower().lstrip(".")
                    for f in args.formats.split(",") if f.strip())
    build(args.out_dir, dpi=args.dpi, formats=formats)


if __name__ == "__main__":
    main()
