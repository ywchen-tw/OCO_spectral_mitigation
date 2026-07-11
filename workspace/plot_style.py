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

Photon path-length symbol l' (LOCKED 2026-07-11): always use
MEAN_L_LABEL / VAR_L_LABEL, never retype the mathtext.  The symbol is
typeset in Times New Roman italic via the repurposed ``\\mathcal`` slot
plus a unicode prime (U+2032) — Arial's mathtext italic l renders as a
bare slash and the ascii prime floats detached, and the STIX script ell
was rejected on looks.  Times is a macOS/Windows font: rebuild
manuscript figures locally (a bare Linux node warns and substitutes).
If the paper's LaTeX uses $l'$, it matches this serif-italic face.
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
# Rendering: Arial's mathtext italic l is a bare slash and the ascii prime
# floats detached, so l' is typeset via the otherwise-unused \mathcal slot,
# mapped to Times New Roman italic in apply_manuscript_style, + unicode prime
# U+2032 (tight).  If Times is missing (bare CURC node), matplotlib warns and
# falls back — rebuild manuscript figures locally.
MEAN_L_LABEL = "$\\langle \\mathcal{l}′ \\rangle$"
VAR_L_LABEL = "var($\\mathcal{l}′$)"


def station_extent(lon0: float, lat0: float, radius_km: float = 100.0) -> list:
    """[lon_min, lon_max, lat_min, lat_max] box of ±radius_km around a station
    (longitude scaled by cos(lat)); matches the TCCON collocation radius."""
    import numpy as np
    dlat = radius_km / 111.195
    dlon = radius_km / (111.195 * max(np.cos(np.radians(lat0)), 0.05))
    return [lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat]


def apply_manuscript_style(base_size: float = 10.0) -> None:
    """Set journal rcParams: Arial text + Arial mathtext, thin axes, 300 dpi."""
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
        # \mathcal slot repurposed as the serif-italic face for the photon
        # path-length symbol l' (MEAN_L_LABEL/VAR_L_LABEL above).
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
