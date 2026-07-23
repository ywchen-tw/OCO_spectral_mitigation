#!/usr/bin/env python3
"""Manuscript Fig. 3 — land/ocean XCO2 anomaly vs cloud distance (Results 4.1).

Two panels (2026-07-21g layout):
  (a) both surfaces under the COMMON default target (`xco2_bc_anomaly`,
      clear-sky reference beyond 10 km) at 1-km bin resolution — the raw
      phenomenology that motivates the surface-specific target radii: the
      ocean response is confined within ~5 km whereas the land response
      persists out to ~15 km, i.e. beyond the 10-km reference threshold.
  (b) the adopted production targets (`xco2_bc_anomaly_r05` ocean,
      `xco2_bc_anomaly_r15` land) — the quantity the correction model is
      trained on, with each surface's reference threshold marked.

Anomalies return to zero beyond the reference threshold of the respective
target by construction; dotted lines mark the thresholds.

Data:  results/csv_collection/combined_2016_2020_dates.parquet
Output (two candidate renderings, choice pending 2026-07-21h):
  manuscript/figures/fig03_anomaly_decay.{png,pdf}          IQR band + median (dashed) + mean (solid)
  manuscript/figures/fig03alt_anomaly_decay_boxplot.{png,pdf}  per-bin boxes
  manuscript/figures/figB2_target_sensitivity.{png,pdf}     Appendix B2: each
      surface under all three reference radii (r05/r10/r15 bin means) —
      label robustness to the target choice, complementary to Fig. 3's
      arrangement (which compares surfaces under a fixed target).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))  # plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import DXCO2_BC_LABEL, apply_manuscript_style, panel_label

PARQUET = REPO / "results" / "csv_collection" / "combined_2016_2020_dates.parquet"
OUT_DIR = REPO / "manuscript" / "figures"

# Okabe-Ito, matching the significance panel's palette discipline.
C_OCEAN = "#0072B2"
C_LAND = "#D55E00"

BIN_EDGES = np.arange(0.0, 30.5, 1.0)   # 1-km bins to the 30 km display range
CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def binned_stats(d: np.ndarray, y: np.ndarray):
    """Per-bin mean, median, q25, q75 over BIN_EDGES (bins with n<50 -> NaN)."""
    idx = np.digitize(d, BIN_EDGES) - 1
    nb = len(BIN_EDGES) - 1
    out = {k: np.full(nb, np.nan) for k in ("mean", "med", "q25", "q75")}
    for b in range(nb):
        v = y[idx == b]
        if len(v) < 50:
            continue
        out["mean"][b] = v.mean()
        out["q25"][b], out["med"][b], out["q75"][b] = \
            np.percentile(v, [25, 50, 75])
    return out


def binned_mean(d: np.ndarray, y: np.ndarray):
    """Back-compat helper for the box-plot overlay: (mean, None)."""
    st = binned_stats(d, y)
    return st["mean"], None


def draw_series(ax, d, y, color, label):
    """IQR band + dashed median + solid mean, one surface."""
    m = np.isfinite(d) & (d >= 0) & np.isfinite(y)
    st = binned_stats(d[m], y[m])
    ax.fill_between(CENTERS, st["q25"], st["q75"], color=color, alpha=0.22,
                    linewidth=0)
    ax.plot(CENTERS, st["med"], color=color, lw=1.2, ls="--")
    ax.plot(CENTERS, st["mean"], color=color, lw=1.7,
            label=f"{label}, n = {m.sum() / 1e6:.1f} M")


def binned_boxstats(d: np.ndarray, y: np.ndarray):
    """Per-bin box stats (median/IQR, 1.5*IQR whiskers clipped to data)."""
    idx = np.digitize(d, BIN_EDGES) - 1
    stats = []
    for b in range(len(BIN_EDGES) - 1):
        v = y[idx == b]
        if len(v) < 50:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        stats.append(dict(
            med=med, q1=q1, q3=q3,
            whislo=max(v.min(), q1 - 1.5 * iqr),
            whishi=min(v.max(), q3 + 1.5 * iqr),
            label="", pos=CENTERS[b]))
    return stats


def draw_box_series(ax, d, y, color):
    m = np.isfinite(d) & (d >= 0) & np.isfinite(y)
    stats = binned_boxstats(d[m], y[m])
    ax.bxp(stats, positions=[s.pop("pos") for s in stats], widths=0.72,
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor=color, alpha=0.35, lw=0.5,
                         edgecolor="0.35"),
           medianprops=dict(color=color, lw=1.5),
           whiskerprops=dict(lw=0.5, color="0.55"),
           capprops=dict(lw=0.5, color="0.55"))
    return m.sum()


def make_boxplot_version(t, d, ocean, land) -> None:
    """2x2 per-bin box-plot rendering: rows = target version, cols = surface."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True)
    panels = [
        (axes[0, 0], ocean, "xco2_bc_anomaly", C_OCEAN, 10.0,
         "(a) Ocean — common target (reference $>$ 10 km)"),
        (axes[0, 1], land, "xco2_bc_anomaly", C_LAND, 10.0,
         "(b) Land — common target (reference $>$ 10 km)"),
        (axes[1, 0], ocean, "xco2_bc_anomaly_r05", C_OCEAN, 5.0,
         "(c) Ocean — adopted target (r05)"),
        (axes[1, 1], land, "xco2_bc_anomaly_r15", C_LAND, 15.0,
         "(d) Land — adopted target (r15)"),
    ]
    for ax, mask, col, color, thr, title in panels:
        y = t[col].to_numpy()
        n = draw_box_series(ax, d[mask], y[mask], color)
        # binned-mean overlay: the skewed near-cloud distribution puts the
        # signal in the mean, well above the median
        mm = np.isfinite(d[mask]) & (d[mask] >= 0) & np.isfinite(y[mask])
        mean, _ = binned_mean(d[mask][mm], y[mask][mm])
        ax.plot(CENTERS, mean, color="0.15", lw=1.2, label="bin mean")
        ax.axvline(thr, color="0.45", lw=0.8, ls=":")
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_xticklabels([str(v) for v in range(0, 31, 5)])
        ax.set_title(f"{title}, n = {n / 1e6:.1f} M", fontsize=8.5, loc="left")
    axes[0, 1].legend(frameon=False, loc="upper right", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Nearest-cloud distance (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{DXCO2_BC_LABEL} (ppm)")
    axes[0, 0].set_ylim(-2.0, 1.5)
    axes[1, 0].set_ylim(-2.0, 1.5)
    axes[0, 1].set_ylim(-2.5, 3.2)
    axes[1, 1].set_ylim(-2.5, 3.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig03alt_anomaly_decay_boxplot.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")


def make_target_sensitivity(t, d, ocean, land) -> None:
    """Appendix B2: per-surface anomaly under all three reference radii."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
    targets = [("xco2_bc_anomaly_r05", 5.0, "-"),
               ("xco2_bc_anomaly", 10.0, "--"),
               ("xco2_bc_anomaly_r15", 15.0, ":")]
    for ax, mask, color, name in [(axes[0], ocean, C_OCEAN, "(a) Ocean"),
                                  (axes[1], land, C_LAND, "(b) Land")]:
        dm = d[mask]
        for col, thr, ls in targets:
            y = t[col].to_numpy()[mask]
            m = np.isfinite(dm) & (dm >= 0) & np.isfinite(y)
            st = binned_stats(dm[m], y[m])
            ax.plot(CENTERS, st["mean"], color=color, ls=ls, lw=1.5,
                    label=f"reference $>$ {thr:.0f} km")
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlim(0, 30)
        ax.set_xlabel("Nearest-cloud distance (km)")
        ax.set_title(name, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8,
                  title="clear-sky reference", title_fontsize=8)
    axes[0].set_ylabel(f"{DXCO2_BC_LABEL} (ppm)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"figB2_target_sensitivity.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    import pyarrow.parquet as pq
    cols = ["cld_dist_km", "sfc_type", "xco2_bc_anomaly",
            "xco2_bc_anomaly_r05", "xco2_bc_anomaly_r15"]
    t = pq.read_table(PARQUET, columns=cols)
    d = t["cld_dist_km"].to_numpy()
    s = t["sfc_type"].to_numpy()
    ocean, land = (s == 0), (s == 1)

    apply_manuscript_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True)

    # -- (a) common r10 target: motivates the surface-specific radii ---------
    y10 = t["xco2_bc_anomaly"].to_numpy()
    draw_series(axa, d[ocean], y10[ocean], C_OCEAN, "Ocean")
    draw_series(axa, d[land], y10[land], C_LAND, "Land")
    axa.axvline(10.0, color="0.45", lw=0.8, ls=":")
    axa.text(10.4, 0.965, "common 10 km\nreference threshold", fontsize=7,
             color="0.35", ha="left", va="top", linespacing=1.3,
             transform=axa.get_xaxis_transform())
    axa.set_title("(a) common target (reference $>$ 10 km)", fontsize=9,
                  loc="left")

    # -- (b) adopted production targets (ocean r05, land r15) ----------------
    draw_series(axb, d[ocean], t["xco2_bc_anomaly_r05"].to_numpy()[ocean],
                C_OCEAN, "Ocean (r05)")
    draw_series(axb, d[land], t["xco2_bc_anomaly_r15"].to_numpy()[land],
                C_LAND, "Land (r15)")
    axb.axvline(5.0, color=C_OCEAN, lw=0.8, ls=":", alpha=0.7)
    axb.axvline(15.0, color=C_LAND, lw=0.8, ls=":", alpha=0.7)
    axb.text(0.985, 0.55,
             "solid: bin mean   dashed: median\nshading: interquartile range\n"
             "dotted verticals: reference thresholds",
             transform=axb.transAxes, fontsize=7, ha="right", va="top",
             color="0.35", linespacing=1.4)
    axb.set_title("(b) adopted targets (ocean r05, land r15)", fontsize=9,
                  loc="left")

    for ax in (axa, axb):
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_xlim(0, 30)
        ax.set_xlabel("Nearest-cloud distance (km)")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    axa.set_ylabel(f"{DXCO2_BC_LABEL} (ppm)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig03_anomaly_decay.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)

    make_boxplot_version(t, d, ocean, land)
    make_target_sensitivity(t, d, ocean, land)


if __name__ == "__main__":
    main()
