"""smoother_null.py — pure-smoother null correction (reviewer objection M4).

The objection: "your correction just smooths XCO2 locally — of course the
local anomaly collapses; TCCON cannot tell the difference."  The null: replace
the deep-ensemble mu with an orbit-local running mean that uses NO features,

    fake_mu(t)                 = xco2_bc(t) − ⟨xco2_bc⟩_{|t'−t| ≤ W, same segment/surface}
    smoother_corrected_xco2(t) = xco2_bc(t) − fake_mu(t) = the running mean,

then push it through the SAME TCCON comparison (tccon_comparison_report.py
--corr-col smoother_w{W:g}_corrected_xco2).  A pure smoother collapses the
footprint-level scatter (corr_sd, hence fp-RMSE) at least as hard as the model
does, but it preserves the local mean, so the case-level TCCON bias must stay
put — whereas the deep ensemble moves the bias itself.

Segments are contiguous-in-time runs (gap > --gap-s starts a new one), split
per surface to mirror the per-surface model (the strongest form of the null).
The mean at each footprint includes the footprint itself, as a smoother would.

Two entry points:
  • add_smoother_columns(df, ...)  — used by build_deepens_plot_data.py so new
    plot_data.parquet files carry the columns from birth;
  • CLI — retrofits existing per-case plot_data.parquet files in place
    (atomic rewrite; idempotent unless --force):

    python workspace/smoother_null.py \
        --base-dir results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5 \
        --windows-s 10,30,100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WINDOWS_S = (10.0, 30.0, 100.0)   # half-widths: mean over ±W s (~±67/200/675 km)


def corr_col(window_s: float) -> str:
    return f"smoother_w{window_s:g}_corrected_xco2"


def anom_col(window_s: float) -> str:
    return f"smoother_w{window_s:g}_pred_anomaly"


def _running_mean(t: np.ndarray, y: np.ndarray, half_window_s: float) -> np.ndarray:
    """Centered running mean of y over |t'−t| ≤ half_window_s.  t must be sorted
    ascending.  NaN-aware (NaN y contribute nothing); windows with no finite
    values return NaN."""
    lo = np.searchsorted(t, t - half_window_s, side="left")
    hi = np.searchsorted(t, t + half_window_s, side="right")
    ok = np.isfinite(y)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    n = cnt[hi] - cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)


def add_smoother_columns(df: pd.DataFrame, *, windows_s=DEFAULT_WINDOWS_S,
                         gap_s: float = 60.0, base_col: str = "xco2_bc",
                         time_col: str = "time",
                         clim_max_ppm: float = 50.0) -> pd.DataFrame:
    """Return df with smoother_w{W}_pred_anomaly / _corrected_xco2 per window.

    Segments = runs of soundings with consecutive time gaps ≤ gap_s, split per
    sfc_type (mirrors the per-surface production model).  Row order and index
    of df are preserved.

    Non-physical retrievals (|base_col − xco2_apriori| > clim_max_ppm, the same
    feature-free criterion as the production climatology input guard) are
    EXCLUDED from the running-mean input so a single fill-value footprint does
    not contaminate its neighbors' smoothed values — without this the null
    would be a straw man.  Screened rows still receive a corrected value (the
    mean of their surviving neighbors)."""
    windows_s = tuple(float(w) for w in windows_s)
    out = df.copy()
    for w in windows_s:
        out[anom_col(w)] = np.nan
        out[corr_col(w)] = np.nan

    order = np.argsort(out[time_col].to_numpy(float), kind="stable")
    t_sorted = out[time_col].to_numpy(float)[order]
    seg_sorted = np.concatenate([[0], np.cumsum(np.diff(t_sorted) > gap_s)])
    seg = np.empty(len(out), dtype=np.int64)
    seg[order] = seg_sorted

    sfc = out["sfc_type"].to_numpy(float) if "sfc_type" in out.columns \
        else np.zeros(len(out))
    y_raw = out[base_col].to_numpy(float)
    t_all = out[time_col].to_numpy(float)
    y_in = y_raw
    if clim_max_ppm is not None and "xco2_apriori" in out.columns:
        dev = np.abs(y_raw - out["xco2_apriori"].to_numpy(float))
        y_in = np.where(np.isfinite(dev) & (dev > clim_max_ppm), np.nan, y_raw)

    means = {w: np.full(len(out), np.nan) for w in windows_s}
    key_all = seg * 2 + (sfc == 1)
    for key in np.unique(key_all):
        m = np.flatnonzero(key_all == key)
        m = m[np.argsort(t_all[m], kind="stable")]
        for w in windows_s:
            means[w][m] = _running_mean(t_all[m], y_in[m], w)

    for w in windows_s:
        out[anom_col(w)] = (y_raw - means[w]).astype(np.float32)
        out[corr_col(w)] = means[w].astype(np.float32)
    return out


def _augment_file(path: Path, *, windows_s, gap_s: float, force: bool) -> str:
    df = pd.read_parquet(path)
    want = [c for w in windows_s for c in (anom_col(w), corr_col(w))]
    if not force and all(c in df.columns for c in want):
        return "skip (columns present)"
    if "xco2_bc" not in df.columns or "time" not in df.columns:
        return "skip (no xco2_bc/time)"
    df = df.drop(columns=[c for c in want if c in df.columns])
    df = add_smoother_columns(df, windows_s=windows_s, gap_s=gap_s)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    rms = {f"w{w:g}": float(np.sqrt(np.nanmean(df[anom_col(w)] ** 2)))
           for w in windows_s}
    return "wrote " + " ".join(f"{k}:fake-mu RMS {v:.3f}ppm" for k, v in rms.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, required=True,
                    help="Model dir holding combined_*/plot_data.parquet case dirs "
                         "(or a dir that IS a case dir).")
    ap.add_argument("--glob", default="combined_*/plot_data.parquet")
    ap.add_argument("--windows-s", default=",".join(f"{w:g}" for w in DEFAULT_WINDOWS_S),
                    help="Comma-separated half-widths W in seconds: mean over ±W s "
                         f"(default {','.join(f'{w:g}' for w in DEFAULT_WINDOWS_S)}; "
                         "±30 s ≈ ±200 km along-track).")
    ap.add_argument("--gap-s", type=float, default=60.0,
                    help="Time gap (s) that starts a new orbit segment (default 60).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even when the columns already exist.")
    args = ap.parse_args()

    windows_s = tuple(float(x) for x in args.windows_s.split(",") if x.strip())
    files = sorted(args.base_dir.glob(args.glob)) or \
        ([args.base_dir / "plot_data.parquet"]
         if (args.base_dir / "plot_data.parquet").exists() else [])
    if not files:
        raise SystemExit(f"no plot_data.parquet under {args.base_dir}/{args.glob}")
    print(f"augmenting {len(files)} file(s); windows ±{windows_s} s, gap {args.gap_s} s")
    for p in files:
        print(f"  {p.parent.name}: {_augment_file(p, windows_s=windows_s, gap_s=args.gap_s, force=args.force)}")
    print(f"report with: --corr-col {corr_col(windows_s[len(windows_s)//2])}"
          f"  (columns: {[corr_col(w) for w in windows_s]})")


if __name__ == "__main__":
    main()
