#!/usr/bin/env python
"""O2A vs CO2-band k1 contrast at Nassar plume crossings (+ dmu/dk1 bound).

Physics being tested (PROJECT_REVIEW M1): the slant optical depth uses the
PRIOR CO2 profile, so a real plume raises the fitted CO2-band path-length
cumulant k1 by ~ dXCO2/XCO2 (about +0.0026 per ppm for k1~1.08), while
o2a_k1 stays flat (O2 is insensitive to CO2). That contrast is the built-in
discriminator between real CO2 structure (CO2-band-only k1 shift, small) and
cloud-induced path-length changes (all-band k1 shifts, large).

Per case this script reports, on the overpass segment:
  - plume window (source_dist <= plume-km) vs background (bg-lo..bg-hi km),
    optionally clear-sky-gated: dXCO2, per-band dk1 with SEs
  - expected CO2-band dk1 from the prior-tau scaling, vs observed
  - dmu (does the correction remove the local enhancement?)
  - empirical dmu/dk1 slope in the background, and the bound
    |dmu_from_k1| = |slope| x expected plume dk1  (how much plume the model
    COULD remove through the spec channel; empirical association, not a
    model derivative)

Caveat (state in any writeup): without wind direction the plume window is a
geometric disk, so dXCO2 underestimates the true enhancement and cases where
the plume missed the track dilute to zero — this is a consistency check, not
a flux estimate.

Outputs: nassar_k1_contrast.csv / .md under --output-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nassar_plume_preservation import (  # noqa: E402
    DEFAULT_CSV_DIR,
    DEFAULT_PLANTS,
    DEFAULT_PLOT_BASE,
    add_feature_columns,
    haversine_km,
    parse_pair,
    read_plants,
)
from nassar_control_null import DEFAULT_PAIRS, SEGMENT_GAP_S  # noqa: E402

DEFAULT_OUTPUT_DIR = DEFAULT_PLOT_BASE / "plume_preservation"
K_BANDS = ("o2a_k1", "wco2_k1", "sco2_k1")


def mean_se(s: pd.Series) -> tuple[float, float, int]:
    arr = s.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) \
        if len(arr) > 1 else np.nan, int(len(arr))


def ols_slope(x: pd.Series, y: pd.Series) -> float:
    m = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if m.sum() < 10 or x[m].std() < 1e-12:
        return np.nan
    return float(np.polyfit(x[m].astype(float), y[m].astype(float), 1)[0])


def contrast_case(pair, plants, plot_base, csv_dir,
                  plume_km, bg_lo, bg_hi, clear_km):
    plant_id, date = pair
    path = plot_base / f"combined_{date}" / "plot_data.parquet"
    if not path.exists():
        print(f"  SKIP {plant_id}:{date} — missing {path}")
        return None
    src = plants[plant_id]
    df = pd.read_parquet(path)
    df = add_feature_columns(df, csv_dir, date)
    if not all(c in df.columns for c in K_BANDS):
        print(f"  SKIP {plant_id}:{date} — spectral columns unavailable")
        return None
    df = df[np.isfinite(df["xco2_bc"])].sort_values("time").reset_index(drop=True)
    df["source_dist_km"] = haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(),
                                        float(src["lat"]), float(src["lon"]))

    seg_id = (df["time"].diff().abs() > SEGMENT_GAP_S).cumsum()
    seg = df[seg_id == seg_id.iloc[df["source_dist_km"].idxmin()]].copy()

    clear_ok = "cld_dist_km" in seg.columns
    gated = seg[seg["cld_dist_km"] >= clear_km] if clear_ok else seg
    gate_used = "clear" if (clear_ok and len(gated[gated["source_dist_km"] <= plume_km]) >= 10) else "all"
    use = gated if gate_used == "clear" else seg

    plume = use[use["source_dist_km"] <= plume_km]
    bg = use[(use["source_dist_km"] >= bg_lo) & (use["source_dist_km"] <= bg_hi)]
    if len(plume) < 10 or len(bg) < 30:
        print(f"  SKIP {plant_id}:{date} — n_plume={len(plume)}, n_bg={len(bg)} "
              f"(gate={gate_used})")
        return None

    row = {"plant_id": plant_id, "date": date, "gate": gate_used,
           "plume_km": plume_km, "n_plume": len(plume), "n_bg": len(bg),
           "median_cld_plume": float(np.nanmedian(plume["cld_dist_km"]))
           if clear_ok else np.nan}

    bc_p, bc_pse, _ = mean_se(plume["xco2_bc"])
    bc_b, bc_bse, _ = mean_se(bg["xco2_bc"])
    d_xco2 = bc_p - bc_b
    row["d_xco2_ppm"] = d_xco2
    row["d_xco2_se"] = float(np.hypot(bc_pse, bc_bse))

    mu_p, mu_pse, _ = mean_se(plume["pred_anomaly"])
    mu_b, mu_bse, _ = mean_se(bg["pred_anomaly"])
    row["d_mu_ppm"] = mu_p - mu_b
    row["d_mu_se"] = float(np.hypot(mu_pse, mu_bse))
    row["d_mu_over_d_xco2"] = (row["d_mu_ppm"] / d_xco2
                               if np.isfinite(d_xco2) and abs(d_xco2) > 0.05
                               else np.nan)

    xco2_bg = bc_b if np.isfinite(bc_b) else 415.0
    for band in K_BANDS:
        k_p, k_pse, _ = mean_se(plume[band])
        k_b, k_bse, _ = mean_se(bg[band])
        dk = k_p - k_b
        row[f"d_{band}"] = dk
        row[f"d_{band}_se"] = float(np.hypot(k_pse, k_bse))
        if band != "o2a_k1":
            row[f"d_{band}_expected"] = float(k_b * d_xco2 / xco2_bg)

    # empirical mu-vs-k1 association in the background (clear scene)
    slope = ols_slope(bg["wco2_k1"], bg["pred_anomaly"])
    row["dmu_dk1_wco2_bg_slope"] = slope
    exp_dk = row.get("d_wco2_k1_expected", np.nan)
    row["mu_from_plume_k1_bound_ppm"] = (abs(slope * exp_dk)
                                         if np.isfinite(slope) and np.isfinite(exp_dk)
                                         else np.nan)
    return row


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = ["plant_id", "date", "gate", "n_plume", "n_bg", "median_cld_plume",
            "d_xco2_ppm", "d_xco2_se", "d_mu_ppm", "d_mu_se", "d_mu_over_d_xco2",
            "d_o2a_k1", "d_o2a_k1_se",
            "d_wco2_k1", "d_wco2_k1_se", "d_wco2_k1_expected",
            "d_sco2_k1", "d_sco2_k1_se", "d_sco2_k1_expected",
            "dmu_dk1_wco2_bg_slope", "mu_from_plume_k1_bound_ppm"]
    cols = [c for c in cols if c in summary.columns]
    lines = [
        "# Nassar k1 Contrast (O2A vs CO2 bands) at Plume Crossings",
        "",
        "Plume window vs along-track background on the overpass segment.",
        "Expectation for a REAL CO2 enhancement: `d_o2a_k1` ~ 0 while",
        "`d_wco2_k1`/`d_sco2_k1` ~ their `_expected` columns (prior-tau scaling",
        "~ k1 x dXCO2/XCO2). Cloud contamination instead moves ALL bands, and by",
        "much more. `d_mu_ppm` is the correction's response to the plume window",
        "(0 = plume preserved); `mu_from_plume_k1_bound_ppm` bounds how much",
        "enhancement the spec channel could remove. No wind direction is used:",
        "d_xco2 underestimates true enhancements (consistency check, not flux).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in summary[cols].itertuples(index=False):
        vals = [f"{v:.3g}" if isinstance(v, float) and np.isfinite(v)
                else ("" if isinstance(v, float) else str(v)) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plants", type=Path, default=DEFAULT_PLANTS)
    ap.add_argument("--plot-base", type=Path, default=DEFAULT_PLOT_BASE)
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--plume-km", type=float, default=15.0)
    ap.add_argument("--bg-lo", type=float, default=30.0)
    ap.add_argument("--bg-hi", type=float, default=100.0)
    ap.add_argument("--clear-km", type=float, default=20.0)
    args = ap.parse_args()

    plants = read_plants(args.plants)
    pairs = args.pair or list(DEFAULT_PAIRS)
    rows = []
    for pair in pairs:
        print(f"{pair[0]}:{pair[1]}", flush=True)
        row = contrast_case(pair, plants, args.plot_base, args.csv_dir,
                            args.plume_km, args.bg_lo, args.bg_hi, args.clear_km)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No cases produced results.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "nassar_k1_contrast.csv", index=False)
    write_markdown(summary, args.output_dir / "nassar_k1_contrast.md")
    print(f"Wrote {len(summary)} rows -> {args.output_dir}/nassar_k1_contrast.csv")


if __name__ == "__main__":
    main()
