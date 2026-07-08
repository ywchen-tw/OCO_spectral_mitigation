#!/usr/bin/env python
"""Screen the combined parquet for small-cloud spec-feature case studies.

Finds frames where a small cloud clips one edge of the 8-footprint swath:
some footprints of the SAME frame are near cloud (< --near-km) while others
are far (> --far-km). Those frames share time, airmass, solar geometry and
(nearly) surface, so the cross-track contrast isolates the cloud effect —
the along-track "the surface changed" confound is gone.

A good showcase additionally needs clear context (so per-footprint clear
baselines exist) and a SHORT near-cloud run (a genuinely small cloud, not
the edge of a deck):
  - clear_frac : fraction of frames within +/- --window-s whose min
                 footprint cld_dist >= --clear-km
  - n_near     : number of frames in the window with min cld < --near-km

Output: spec_case_candidates_{land,ocean}.csv under --output-dir, ranked by
clear_frac then cross-track contrast. Feed a row's date + frame_time to
spec_case_figure.py.

Frame decode: parquet fp_id is the 16-digit sounding ID; frame = fp_id // 10,
footprint index = fp_id % 10 - 1 (== the `fp` column).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_PARQUET = Path("results/csv_collection/combined_2016_2020_dates.parquet")
DEFAULT_OUTDIR = Path("results/figures/cld_dist_analysis/spec_case_study")

LOAD_COLS = ["date", "time", "fp", "fp_id", "lat", "lon", "cld_dist_km",
             "sfc_type", "csnr_o2a", "xco2_qf"]


def frame_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate soundings to frames (fp_id // 10)."""
    df = df.copy()
    df["frame"] = df["fp_id"] // 10
    df["is_qf0"] = (df["xco2_qf"] == 0).astype(np.float64)
    g = df.groupby("frame", sort=False)
    fr = pd.DataFrame({
        "n_fp": g["fp"].size(),
        "cld_min": g["cld_dist_km"].min(),
        "cld_max": g["cld_dist_km"].max(),
        "time": g["time"].mean(),
        "lat": g["lat"].mean(),
        "lon": g["lon"].mean(),
        "sfc_type": g["sfc_type"].first(),
        "csnr_o2a": g["csnr_o2a"].median(),
        "qf0_frac": g["is_qf0"].mean(),
    })
    idx = g["cld_dist_km"].idxmin()
    fr["fp_nearest"] = df.loc[idx, "fp"].to_numpy()
    return fr.reset_index()


def add_context(fr: pd.DataFrame, window_s: float, near_km: float,
                clear_km: float) -> pd.DataFrame:
    """clear_frac / n_near within +/- window_s, vectorized per date."""
    fr = fr.sort_values("time").reset_index(drop=True)
    t = fr["time"].to_numpy()
    is_clear = (fr["cld_min"].to_numpy() >= clear_km).astype(np.int64)
    is_near = (fr["cld_min"].to_numpy() < near_km).astype(np.int64)
    cum_clear = np.concatenate([[0], np.cumsum(is_clear)])
    cum_near = np.concatenate([[0], np.cumsum(is_near)])
    lo = np.searchsorted(t, t - window_s, side="left")
    hi = np.searchsorted(t, t + window_s, side="right")
    n_win = hi - lo
    fr["clear_frac"] = (cum_clear[hi] - cum_clear[lo]) / np.maximum(n_win, 1)
    fr["n_near"] = cum_near[hi] - cum_near[lo]
    fr["n_win"] = n_win
    return fr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-fname", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--near-km", type=float, default=2.0)
    ap.add_argument("--far-km", type=float, default=15.0)
    ap.add_argument("--clear-km", type=float, default=20.0)
    ap.add_argument("--min-fp", type=int, default=5,
                    help="Minimum surviving footprints in the frame.")
    ap.add_argument("--window-s", type=float, default=45.0,
                    help="Context window (+/- s ~ 6.74 km/s ground speed).")
    ap.add_argument("--min-clear-frac", type=float, default=0.25)
    ap.add_argument("--max-near", type=int, default=15,
                    help="Max near-cloud frames in the window (small cloud).")
    ap.add_argument("--top", type=int, default=200, help="Rows kept per surface.")
    args = ap.parse_args()

    print(f"Loading {args.parquet_fname} ({LOAD_COLS}) ...", flush=True)
    df = pq.read_table(args.parquet_fname, columns=LOAD_COLS).to_pandas()
    df["date"] = df["date"].astype(str)
    df = df[np.isfinite(df["cld_dist_km"])]
    print(f"  {len(df):,} soundings with cld_dist, {df['date'].nunique()} dates",
          flush=True)

    frames = []
    for date, ddf in df.groupby("date", sort=True):
        fr = frame_table(ddf)
        fr = add_context(fr, args.window_s, args.near_km, args.clear_km)
        fr.insert(0, "date", date)
        cand = fr[(fr["n_fp"] >= args.min_fp)
                  & (fr["cld_min"] < args.near_km)
                  & (fr["cld_max"] > args.far_km)
                  & (fr["clear_frac"] >= args.min_clear_frac)
                  & (fr["n_near"] <= args.max_near)]
        if len(cand):
            frames.append(cand)
    if not frames:
        raise SystemExit("No candidate frames found — loosen the thresholds.")
    cand = pd.concat(frames, ignore_index=True)
    cand["contrast_km"] = cand["cld_max"] - cand["cld_min"]
    cand = cand.sort_values(["clear_frac", "contrast_km"],
                            ascending=[False, False])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, sel in (("land", cand["sfc_type"] == 1),
                      ("ocean", cand["sfc_type"] == 0)):
        sub = cand[sel].head(args.top)
        out = args.output_dir / f"spec_case_candidates_{name}.csv"
        sub.to_csv(out, index=False)
        print(f"{name}: {sel.sum()} candidates, wrote top {len(sub)} -> {out}")
        if len(sub):
            show = sub.head(10)[["date", "frame", "time", "lat", "lon", "n_fp",
                                 "cld_min", "cld_max", "fp_nearest",
                                 "clear_frac", "n_near", "csnr_o2a"]]
            print(show.to_string(index=False))


if __name__ == "__main__":
    main()
