"""plot_style.py — shared manuscript figure style (AMT requires Arial).

One import at the top of every figure-producing script:

    from plot_style import apply_manuscript_style, panel_label
    apply_manuscript_style()

sets Arial for text AND mathtext (custom mathtext maps rm/it/bf to Arial so
$X_{\\mathrm{CO_2}}$, $k_1$ … render in the same face; glyphs Arial lacks
fall back to DejaVu Sans), journal-scale font sizes, thin axes, and 300-dpi
saves.  ``panel_label(ax, '(a)')`` puts the bold panel letter at the
conventional top-left-outside position.

Colormap policy (CVD-safe, AMT-friendly): sequential fields use
perceptually-uniform maps — XCO2 'viridis', predictive sigma 'magma',
nearest-cloud distance 'cividis' (CMAPS below).  No rainbow/jet.

Photon path-length symbol l' — lowercase Times ITALIC (final form
2026-07-22, after the capital-L experiment of 07-15 and a brief upright
trial): always use MEAN_L_LABEL / VAR_L_LABEL, never retype the mathtext.
Arial's italic lowercase l is a bare slash, so l is typeset through the
``\\mathcal`` slot (mapped to Times New Roman italic below) + a unicode
prime (U+2032).  If Times is missing (bare CURC node), matplotlib warns
and falls back — rebuild manuscript figures locally.  The paper's LaTeX
should use plain math-italic $l'$.
"""
from __future__ import annotations

import matplotlib as mpl

# Per-quantity colormaps shared across map figures (import instead of retyping
# strings so the whole suite re-themes from one place).
CMAPS = {
    "xco2": "plasma",      # bright low end — separates from dark ocean/land RGB
    "sigma": "magma",
    "cld_dist": "cividis",
    "spec": "plasma",
    "mu": "RdBu_r",        # signed predicted correction — diverging, 0-centred
}

XCO2_LABEL = r"$X_{\mathrm{CO2}}$"
# Spectral-fit cumulants shown with their physical meaning: k1 = mean and
# k2 = variance of the relative photon path l' (k_n defined in the text).
# Rendering: Times-italic l via the \mathcal slot (Arial's italic l is a
# bare slash) + unicode prime U+2032 (tight).
MEAN_L_LABEL = "$\\langle \\mathcal{l}′ \\rangle$"
VAR_L_LABEL = "var($\\mathcal{l}′$)"


def station_extent(lon0: float, lat0: float, radius_km: float = 100.0) -> list:
    """[lon_min, lon_max, lat_min, lat_max] box of ±radius_km around a station
    (longitude scaled by cos(lat)); matches the TCCON collocation radius."""
    import numpy as np
    dlat = radius_km / 111.195
    dlon = radius_km / (111.195 * max(np.cos(np.radians(lat0)), 0.05))
    return [lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat]


def _register_local_fonts() -> None:
    """Register .ttf files from <repo>/fonts/ and ~/.fonts/ with matplotlib.

    Arial and Times New Roman are macOS/Windows fonts; on a bare Linux node
    (CURC) matplotlib silently substitutes DejaVu.  Copy the six .ttf files
    (Arial, Arial Bold, Arial Italic, Arial Bold Italic, Times New Roman,
    Times New Roman Italic — e.g. from macOS /System/Library/Fonts/
    Supplemental/) into <repo>/fonts/ (gitignored: proprietary) and this
    picks them up.  Idempotent; missing dirs are fine.
    """
    from pathlib import Path
    from matplotlib import font_manager
    for d in (Path(__file__).resolve().parents[1] / "fonts",
              Path.home() / ".fonts"):
        if d.is_dir():
            for ttf in sorted(d.glob("*.[tT][tT][fF]")):
                try:
                    font_manager.fontManager.addfont(str(ttf))
                except Exception:   # unreadable/duplicate font — not fatal
                    pass


def fonts_available() -> bool:
    """True if BOTH manuscript fonts (Arial, Times New Roman) resolve to real
    files — the launcher preflight, so a 24 h CURC run cannot silently render
    the whole suite in DejaVu."""
    from pathlib import Path
    from matplotlib import font_manager
    _register_local_fonts()
    ok = True
    for fam in ("Arial", "Times New Roman"):
        path = font_manager.findfont(
            font_manager.FontProperties(family=fam), fallback_to_default=True)
        found = fam.split()[0].lower() in Path(path).name.lower()
        print(f"[plot_style] {fam}: {'OK  ' if found else 'MISSING'} ({path})")
        ok &= found
    return ok


def apply_manuscript_style(base_size: float = 10.0) -> None:
    """Set journal rcParams: Arial text + Arial mathtext, thin axes, 300 dpi."""
    _register_local_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "figure.titlesize": base_size + 1.5,
        # mathtext in Arial too (fallback glyphs come from dejavusans)
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        # \mathcal slot maps to Times ITALIC: it typesets the lowercase l
        # of MEAN_L_LABEL/VAR_L_LABEL (Arial's italic l is a bare slash;
        # final form 2026-07-22).
        "mathtext.cal": "Times New Roman:italic",
        "mathtext.fallback": "stixsans",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,   # Arial-safe minus sign in tick labels
    })


def panel_label(ax, tag: str, *, size: float = 10.0, dx: float = 0.0,
                dy: float = 1.02, inside: bool = False) -> None:
    """Bold panel tag ('(a)', '(b) land', …) top-left, outside the axes by
    default (``inside=True`` pins it just inside for tight map grids)."""
    if not tag:
        return
    if inside:
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, fontsize=size,
                fontweight="bold", va="top", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    else:
        ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=size,
                fontweight="bold", va="bottom", ha="left")
