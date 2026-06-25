"""build_deepens_plot_data.py — bridge apply_deep_ensemble → plot_corrected_xco2.

apply_deep_ensemble.py writes a CSV keyed for analysis (pred_anomaly, intervals,
xco2_corrected) but DROPS the `time` column that plot_corrected_xco2.py needs for
TCCON/MODIS alignment, and names its correction `xco2_corrected` rather than the
`*_corrected_xco2` columns the plotter recognizes.

This script runs the deep-ensemble prediction inline (reusing the apply helpers so
the feature pipeline / domain guard are identical) for OCEAN and/or LAND, concats
the two surfaces, and writes one plot_data.parquet with the columns
plot_corrected_xco2.py expects:

    time, lon, lat, cld_dist_km, sfc_type, xco2_bc, xco2_bc_anomaly,
    deep_ensemble_corrected_xco2   (= xco2_bc - predicted_anomaly)

`ideal_corrected_xco2` is derived by the plotter from xco2_bc - xco2_bc_anomaly.

The deep ensemble is per-surface: ocean footprints (sfc_type==0) are corrected by
the ocean model and land footprints (sfc_type==1) by the land model.  Pass whichever
surfaces you want; only correct footprints from a passed surface appear in the output.
Only the point correction (ensemble mean mu) is needed for the poster figure, so no
conformal calibration is required here.

Example (both surfaces):
    PYTHONPATH=src python workspace/build_deepens_plot_data.py \
        --ocean-model-dir results/model_deep_ensemble/de_ocean_beta_nll_f0 \
        --land-model-dir  results/model_deep_ensemble/de_land_beta_nll_f0 \
        --input results/csv_collection/combined_2018-09-02_all_orbits.parquet \
        --output .../deep_ensemble/plot_data.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from apply.apply_deep_ensemble import _load_model, _predict

CORR_COL = 'deep_ensemble_corrected_xco2'
KEEP_COLS = ('time', 'lon', 'lat', 'cld_dist_km', 'sfc_type',
             'xco2_bc', 'xco2_bc_anomaly', 'xco2_apriori')


def _build_surface(df, model_dir, sfc_type, *, dk, clim_max_ppm=50.0,
                   max_abs_anomaly=15.0):
    """Predict one surface; return a plot_data frame (or None if no rows).

    Two guards skip the correction (set mu=0 → corrected XCO2 == raw xco2_bc) and
    flag the row, so non-physical points don't pollute the histogram:
      • clim_guard    — INPUT guard: xco2_bc > xco2_apriori + `clim_max_ppm`
                        (fill values / failed retrievals).
      • anomaly_guard — OUTPUT guard: |predicted anomaly| > `max_abs_anomaly`
                        (model blow-ups; a bias correction should be a few ppm,
                        not hundreds).  No effect when `max_abs_anomaly` is None.
    """
    pipeline, members, meta = _load_model(Path(model_dir))
    mu, sigma, kept = _predict(df, pipeline, members, meta, sfc_type,
                               tag=f'sfc{sfc_type}', **dk)
    if len(kept) == 0:
        print(f"  sfc={sfc_type}: no rows after filter — skipped")
        return None
    out = kept[[c for c in KEEP_COLS if c in kept.columns]].reset_index(drop=True).copy()
    mu = np.asarray(mu, dtype=float)

    # INPUT guard (climatology)
    clim_guard = np.zeros(len(out), dtype=bool)
    if 'xco2_apriori' in out.columns:
        diff = out['xco2_bc'].to_numpy(float) - out['xco2_apriori'].to_numpy(float)
        clim_guard = np.isfinite(diff) & (diff > clim_max_ppm)
    else:
        print(f"  sfc={sfc_type}: no xco2_apriori column — climatology guard disabled")

    # OUTPUT guard (anomaly magnitude)
    anomaly_guard = np.zeros(len(out), dtype=bool)
    if max_abs_anomaly is not None:
        anomaly_guard = np.isfinite(mu) & (np.abs(mu) > max_abs_anomaly)

    guard = clim_guard | anomaly_guard
    if guard.any():
        mu = mu.copy(); mu[guard] = 0.0
        print(f"  sfc={sfc_type}: guards skipped correction on {int(guard.sum())} "
              f"sounding(s)  [clim>{clim_max_ppm:g}ppm: {int(clim_guard.sum())}, "
              f"|anomaly|>{max_abs_anomaly}: {int(anomaly_guard.sum())}]")

    out['clim_guard'] = clim_guard
    out['anomaly_guard'] = anomaly_guard
    out['pred_anomaly'] = mu
    out['sigma'] = sigma
    out[CORR_COL] = out['xco2_bc'].to_numpy(dtype=float) - mu
    if 'xco2_bc_anomaly' in out.columns:
        y = out['xco2_bc_anomaly'].to_numpy(float)
        keep = ~guard                      # report only where correction was applied
        pre = np.sqrt(np.nanmean(y[keep] ** 2))
        post = np.sqrt(np.nanmean((y[keep] - mu[keep]) ** 2))
        print(f"  sfc={sfc_type}: {int(keep.sum()):,}/{len(out):,} corrected  anomaly RMS "
              f"{pre:.3f} → {post:.3f} ppm ({100*(1-post/pre):+.1f}%)")
    else:
        print(f"  sfc={sfc_type}: {len(out):,} soundings (no truth column)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ocean-model-dir', default=None,
                    help="Deep-ensemble dir trained on ocean (sfc_type==0).")
    ap.add_argument('--land-model-dir', default=None,
                    help="Deep-ensemble dir trained on land (sfc_type==1).")
    # backward-compatible single-surface form
    ap.add_argument('--model-dir', default=None, help="(single-surface) model dir.")
    ap.add_argument('--sfc_type', type=int, default=None,
                    help="(single-surface) 0=ocean, 1=land; pairs with --model-dir.")
    ap.add_argument('--input', nargs='+', required=True,
                    help="Unseen parquet(s) carrying time/lon/lat/xco2_bc/fp/cld_dist_km.")
    ap.add_argument('--output', required=True, help="plot_data.parquet to write.")
    ap.add_argument('--ood-thresh', type=float, default=8.0)
    ap.add_argument('--max-ood-frac', type=float, default=0.02)
    ap.add_argument('--strict', action='store_true',
                    help="Refuse if the domain check fails (incompatible parquet).")
    ap.add_argument('--climatology-max-ppm', type=float, default=50.0,
                    help="Skip correction where xco2_bc > xco2_apriori + this many ppm "
                         "(non-physical retrievals); default 50.")
    ap.add_argument('--max-abs-anomaly', type=float, default=15.0,
                    help="Skip correction where |predicted anomaly| exceeds this many ppm "
                         "(model blow-ups); default 15. Set <=0 to disable.")
    args = ap.parse_args()
    dk = dict(ood_thresh=args.ood_thresh, max_ood_frac=args.max_ood_frac, strict=args.strict)

    surfaces = []  # (model_dir, sfc_type)
    if args.ocean_model_dir:
        surfaces.append((args.ocean_model_dir, 0))
    if args.land_model_dir:
        surfaces.append((args.land_model_dir, 1))
    if args.model_dir is not None:
        if args.sfc_type is None:
            raise SystemExit("--model-dir requires --sfc_type")
        surfaces.append((args.model_dir, args.sfc_type))
    if not surfaces:
        raise SystemExit("pass --ocean-model-dir and/or --land-model-dir "
                         "(or --model-dir with --sfc_type)")

    df = pd.concat([pd.read_parquet(p) for p in args.input], ignore_index=True)
    print(f"  read {len(df):,} rows from {len(args.input)} file(s)")

    max_abs_anom = args.max_abs_anomaly if args.max_abs_anomaly > 0 else None
    parts = [p for p in (_build_surface(df, md, sfc, dk=dk,
                                        clim_max_ppm=args.climatology_max_ppm,
                                        max_abs_anomaly=max_abs_anom)
                         for md, sfc in surfaces)
             if p is not None]
    if not parts:
        raise SystemExit("no soundings predicted on any surface")
    out = pd.concat(parts, ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"  [saved] {len(out):,} rows ({len(parts)} surface(s)) → {args.output}")
    print(f"  correction column: {CORR_COL!r}  (use --poster-model {CORR_COL})")


if __name__ == '__main__':
    main()
