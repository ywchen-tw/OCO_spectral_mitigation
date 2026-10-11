#!/usr/bin/env python
"""Full vs no_spec plume-signal preservation on the Nassar power-plant windows.

The control-region null already attributes the *clear-sky contrast smoothing*
to the xco2 channel (nassar_channel_attribution.py: dropping the cumulants
moves the removal by ~1 pp).  This script closes the loop on the other side of
the same question: run the plume windows themselves through both models and
ask whether the along-track CO2 enhancement that survives the correction
depends on the spectral features.

Metric (per case, on the closest-approach overpass segment):

    enhancement = median(|x| <= plume_km) - median(bg_lo <= |x| <= bg_hi)

evaluated on xco2_bc (the input signal), on the full-model corrected product,
and on the no_spec corrected product; `retained` is the corrected enhancement
divided by the xco2_bc enhancement.  Also reports the footprint-level spread
between the two corrected products.

Outputs (under --output-dir):
    nassar_spec_variant_transect_compare.csv / .md
    nassar_spec_variant_transects.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT_WORKSPACE = Path(__file__).resolve().parents[1]
if str(ROOT_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(ROOT_WORKSPACE))
from plot_style import apply_manuscript_style, panel_label, XCO2_LABEL  # noqa: E402
from analyze_nassar_plume_preservation import DEFAULT_PLANTS, read_plants, parse_pair  # noqa: E402
from nassar_plume_transects import load_transect_segment, rolling  # noqa: E402

DE_TAG = "de_beta_nll_prof_reg_foldpca_o05l15_m5"
DEFAULT_FULL_BASE = Path(f"results/model_comparison/deep_ensemble/{DE_TAG}/nassar_plumes")
DEFAULT_VARIANT_BASE = Path(
    f"results/model_comparison/deep_ensemble/{DE_TAG}/nassar_plumes_variants/no_spec")
DEFAULT_OUTPUT_DIR = DEFAULT_VARIANT_BASE / "plume_preservation_11cases"

# Every catalog plant/date whose overpass comes within 12 km of the source
# (the "testable" set used in the manuscript plume-control table).
DEFAULT_PAIRS = (
    ("lipetsk", "2015-08-01"),
    ("kozienice", "2021-09-06"),
    ("taean", "2021-09-08"),
    ("westar", "2023-03-13"),
    ("westar", "2023-06-26"),
    ("ghent", "2024-04-15"),
)


def enhancement(x: np.ndarray, y: np.ndarray, plume_km: float,
                bg_lo: float, bg_hi: float) -> float:
    core = np.abs(x) <= plume_km
    bg = (np.abs(x) >= bg_lo) & (np.abs(x) <= bg_hi)
    if core.sum() < 3 or bg.sum() < 5:
        return np.nan
    return float(np.nanmedian(y[core]) - np.nanmedian(y[bg]))


def compare_case(pair, plants, full_base, var_base, args):
    full = load_transect_segment(pair, plants, full_base, max_km=args.max_km)
    var = load_transect_segment(pair, plants, var_base, max_km=args.max_km)
    if full is None or var is None:
        return None, None
    seg_f, src, min_dist = full
    seg_v = var[0]

    merged = seg_f.merge(
        seg_v[["time", "deep_ensemble_corrected_xco2", "pred_anomaly"]],
        on="time", how="inner", suffixes=("_full", "_ns"))
    if len(merged) < 20:
        print(f"  SKIP {pair[0]}:{pair[1]} — only {len(merged)} matched footprints")
        return None, None

    x = merged["x_km"].to_numpy()
    kw = dict(plume_km=args.plume_km, bg_lo=args.bg_lo_km, bg_hi=args.bg_hi_km)
    e_bc = enhancement(x, merged["xco2_bc"].to_numpy(), **kw)
    e_full = enhancement(x, merged["deep_ensemble_corrected_xco2_full"].to_numpy(), **kw)
    e_ns = enhancement(x, merged["deep_ensemble_corrected_xco2_ns"].to_numpy(), **kw)
    d_corr = (merged["deep_ensemble_corrected_xco2_ns"]
              - merged["deep_ensemble_corrected_xco2_full"]).to_numpy()
    core = np.abs(x) <= args.plume_km

    row = {
        "plant_id": pair[0], "source_name": src["source_name"], "date": pair[1],
        "n_segment": len(merged), "n_core": int(core.sum()),
        "closest_approach_km": min_dist,
        "median_cld_dist_km_core": float(np.nanmedian(merged["cld_dist_km"].to_numpy()[core])),
        "enh_xco2_bc_ppm": e_bc,
        "enh_corrected_full_ppm": e_full,
        "enh_corrected_no_spec_ppm": e_ns,
        "retained_full": e_full / e_bc if np.isfinite(e_bc) and abs(e_bc) > 1e-9 else np.nan,
        "retained_no_spec": e_ns / e_bc if np.isfinite(e_bc) and abs(e_bc) > 1e-9 else np.nan,
        "d_enh_no_spec_minus_full_ppm": e_ns - e_full,
        "mean_abs_d_corrected_ppm": float(np.nanmean(np.abs(d_corr))),
        "p95_abs_d_corrected_ppm": float(np.nanpercentile(np.abs(d_corr), 95)),
        "max_abs_d_corrected_core_ppm": float(np.nanmax(np.abs(d_corr[core]))) if core.any() else np.nan,
        "corr_mu_full_vs_no_spec": float(pd.Series(merged["pred_anomaly_full"]).corr(
            pd.Series(merged["pred_anomaly_ns"]))),
    }
    return row, merged


def plot_panel(ax, merged, row, args):
    x = merged["x_km"].to_numpy()
    for col, c, lbl in (("xco2_bc", "C0", "bias-corrected (input)"),
                        ("deep_ensemble_corrected_xco2_full", "C3", "DE full"),
                        ("deep_ensemble_corrected_xco2_ns", "C2", "DE no_spec")):
        ax.plot(x, merged[col], ".", color=c, ms=2, alpha=0.25)
        xs, med = rolling(x, merged[col].to_numpy())
        ax.plot(xs, med, "-", color=c, lw=1.6, label=lbl)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.axvspan(-args.plume_km, args.plume_km, color="gold", alpha=0.15)
    ax.grid(alpha=0.3)
    ax.set_title(f"{row['source_name']} {row['date']}  "
                 f"(closest {row['closest_approach_km']:.1f} km)", fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--full-base", type=Path, default=DEFAULT_FULL_BASE)
    ap.add_argument("--variant-base", type=Path, default=DEFAULT_VARIANT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--max-km", type=float, default=100.0)
    ap.add_argument("--plume-km", type=float, default=10.0)
    ap.add_argument("--bg-lo-km", type=float, default=20.0)
    ap.add_argument("--bg-hi-km", type=float, default=60.0)
    args = ap.parse_args()

    apply_manuscript_style()
    plants = read_plants(args.plants)
    pairs = args.pair or list(DEFAULT_PAIRS)

    rows, panels = [], []
    for pair in pairs:
        print(f"{pair[0]}:{pair[1]}", flush=True)
        row, merged = compare_case(pair, plants, args.full_base, args.variant_base, args)
        if row is not None:
            rows.append(row)
            panels.append((row, merged))
    if not rows:
        raise SystemExit("no cases compared")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    csv_path = args.output_dir / "nassar_spec_variant_transect_compare.csv"
    table.to_csv(csv_path, index=False)

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for i, (row, merged) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        plot_panel(ax, merged, row, args)
        panel_label(ax, f"({chr(97 + i)})")
        if i % ncol == 0:
            ax.set_ylabel(f"{XCO2_LABEL} (ppm)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("along-track distance from closest approach (km)")
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    png = args.output_dir / "nassar_spec_variant_transects.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    def f(v, n=2):
        return "" if not np.isfinite(v) else f"{v:.{n}f}"
    lines = [
        "# Plume-signal preservation — full vs no_spec",
        "",
        f"Enhancement = median(|x| ≤ {args.plume_km:g} km) − median("
        f"{args.bg_lo_km:g}–{args.bg_hi_km:g} km) along the closest-approach "
        "overpass segment. `retained` = corrected enhancement / bias-corrected "
        "enhancement.",
        "",
        "| case | date | closest (km) | n core | median cld (km) | enh bc | enh full | "
        "enh no_spec | retained full | retained no_spec | Δenh (ns−full) | "
        "mean &#124;Δcorrected&#124; | max &#124;Δ&#124; in core |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.plant_id} | {r.date} | {f(r.closest_approach_km,1)} | {int(r.n_core)} | "
            f"{f(r.median_cld_dist_km_core,1)} | {f(r.enh_xco2_bc_ppm)} | "
            f"{f(r.enh_corrected_full_ppm)} | {f(r.enh_corrected_no_spec_ppm)} | "
            f"{f(r.retained_full)} | {f(r.retained_no_spec)} | "
            f"{f(r.d_enh_no_spec_minus_full_ppm)} | {f(r.mean_abs_d_corrected_ppm)} | "
            f"{f(r.max_abs_d_corrected_core_ppm)} |")
    lines += [
        "",
        f"Median |Δ enhancement| (no_spec − full): "
        f"{np.nanmedian(np.abs(table.d_enh_no_spec_minus_full_ppm)):.3f} ppm; "
        f"max {np.nanmax(np.abs(table.d_enh_no_spec_minus_full_ppm)):.3f} ppm.",
        f"Median footprint-level mean |Δ corrected|: "
        f"{np.nanmedian(table.mean_abs_d_corrected_ppm):.3f} ppm.",
        "",
    ]
    md_path = args.output_dir / "nassar_spec_variant_transect_compare.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {csv_path}\nwrote {md_path}\nwrote {png}")


if __name__ == "__main__":
    main()
