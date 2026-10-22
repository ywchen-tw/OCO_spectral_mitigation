"""predict_pool.py — step 2: out-of-fold mu_raw on the training-date pool.

For every pool station-day, runs the production plot-data builder
(workspace/build_deepens_plot_data.py, imported in-process) with ONLY the raw
land fold whose held set contains the date (pool_manifest.held_fold), so the
prediction is out-of-fold and the training-date leakage guard passes by
itself (no --allow-train-overlap).  Output per case:

    results/anchor_lr/pool/combined_<date>_<site>/plot_data.parquet

with deep_ensemble_corrected_xco2 = xco2_raw − mu_raw (guards as in production).
Note: 5 members (one fold) per date instead of the 25-member evaluation mix.

Run:  python workspace/anchor_lr/predict_pool.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                                   # noqa: E402
import build_deepens_plot_data as bdp                        # noqa: E402
from models.leakage_guard import check_training_overlap      # noqa: E402

DK = dict(ood_thresh=8.0, max_ood_frac=0.02, strict=False)   # builder defaults


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--climatology-max-ppm', type=float, default=50.0)
    ap.add_argument('--max-abs-anomaly', type=float, default=25.0)
    args = ap.parse_args()

    man = pd.read_csv(ac.MANIFEST)
    done, failed = 0, []
    diag = []
    for r in man.itertuples():
        src = ac.ROOT / r.source
        out = src.parent / 'plot_data.parquet'
        fold = ac.raw_fold_dir(int(r.held_fold))
        if out.exists() and not args.force:
            print(f'  skip {src.parent.name} (exists)')
        else:
            print(f'\n=== {src.parent.name}  fold f{r.held_fold}')
            df = pd.read_parquet(src)
            check_training_overlap([fold], input_paths=[src],
                                   times=df['time'].to_numpy(), allow=False, tag='sfc1')
            part = bdp._build_surface(df, [fold], 1, dk=DK,
                                      clim_max_ppm=args.climatology_max_ppm,
                                      max_abs_anomaly=args.max_abs_anomaly,
                                      base_col='xco2_raw', truth_col='xco2_raw_anomaly')
            if part is None or len(part) == 0:
                failed.append(src.parent.name)
                continue
            part.to_parquet(out, index=False)
            done += 1
        # verification: mu_raw vs the r15 raw target (the fold's own training target)
        pdf = pd.read_parquet(out, columns=['time', 'lon', 'lat', 'pred_anomaly',
                                            'clim_guard', 'anomaly_guard'])
        sdf = pd.read_parquet(src, columns=ac.MERGE_KEYS + ['xco2_raw_anomaly_r15'])
        m = pdf.merge(sdf.drop_duplicates(ac.MERGE_KEYS), on=ac.MERGE_KEYS, how='left')
        ok = (~(m.clim_guard | m.anomaly_guard)) & np.isfinite(m.xco2_raw_anomaly_r15)
        y = m.loc[ok, 'xco2_raw_anomaly_r15'].to_numpy(float)
        mu = m.loc[ok, 'pred_anomaly'].to_numpy(float)
        diag.append(dict(case=src.parent.name, fold=int(r.held_fold), n_label=int(ok.sum()),
                         rms_before=float(np.sqrt(np.mean(y ** 2))) if ok.any() else np.nan,
                         rms_after=float(np.sqrt(np.mean((y - mu) ** 2))) if ok.any() else np.nan))
    d = pd.DataFrame(diag)
    d.to_csv(ac.OUT_ROOT / 'pool_prediction_diagnostics.csv', index=False)
    n = d.n_label.to_numpy(float)
    ok = np.isfinite(d.rms_before) & (n > 0)
    pooled_b = np.sqrt(np.sum(n[ok] * d.rms_before[ok] ** 2) / n[ok].sum())
    pooled_a = np.sqrt(np.sum(n[ok] * d.rms_after[ok] ** 2) / n[ok].sum())
    print(f'\nbuilt {done} case(s), {len(failed)} failed {failed}')
    print(f'labelled pool rows (r15 target, unguarded): {int(n.sum()):,}; '
          f'RMS {pooled_b:.3f} → {pooled_a:.3f} ppm ({100 * (1 - pooled_a / pooled_b):+.1f}%) '
          f'[out-of-fold, one fold per date]')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
