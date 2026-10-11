"""Pooled out-of-fold correction-effectiveness report.

Under date_kfold every date is in the held-out (test) fold exactly once, so
concatenating the per-fold ``held_out_predictions.parquet`` files yields one
out-of-sample prediction for every sounding in the dataset — the cleanest
estimate of how the correction behaves in deployment.

For each cloud-distance bin it reports the anomaly magnitude BEFORE the
correction (|y|) vs AFTER (|y - mu|), and additionally breaks the near-cloud
(<=10 km) soundings into bulk vs left-tail so you can see how well the
correction does on the large anomalies you actually care about.

Usage:
    PYTHONPATH=src python -m models.correction_report \
        --glob 'results/model_deep_ensemble/de_ocean_f*/held_out_predictions.parquet' \
        --label DeepEns-Ocean \
        --out results/model_comparison/deep_ensemble_ocean_correction.md
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from . import diagnostics as diag


def _md_table(df: pd.DataFrame, floatfmt: str = '.4f') -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    cols = list(df.columns)
    head = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join('---' for _ in cols) + ' |'
    body = ['| ' + ' | '.join(fmt(v) for v in row) + ' |'
            for row in df.itertuples(index=False, name=None)]
    return '\n'.join([head, sep, *body])


def _tail_breakdown(df: pd.DataFrame, near_km: float = 10.0,
                    tail_fracs=(0.05, 0.10)) -> pd.DataFrame:
    """Correction RMS pre/post for near-cloud bulk vs left tail (global tail)."""
    y = df['y_true'].to_numpy(float)
    resid = y - df['mu'].to_numpy(float)
    cd = df['cld_dist_km'].to_numpy(float)
    near = (cd <= near_km) & np.isfinite(cd)
    order = np.argsort(y)
    n = len(y)

    def _row(label, mask):
        yy, rr = y[mask], resid[mask]
        pre = float(np.sqrt(np.mean(yy ** 2)))
        post = float(np.sqrt(np.mean(rr ** 2)))
        return {'group': label, 'n': int(mask.sum()),
                'pre_rms': pre, 'post_rms': post,
                'rms_reduction_pct': 100.0 * (1 - post / pre) if pre > 0 else np.nan,
                'post_bias': float(rr.mean())}

    rows = [_row(f'near_cloud(<={near_km:g}km) all', near)]
    tail_any = np.zeros(n, dtype=bool)
    for frac in tail_fracs:
        k = max(1, int(np.floor(frac * n)))
        tail = np.zeros(n, dtype=bool)
        tail[order[:k]] = True
        rows.append(_row(f'near & bottom_{int(frac * 100)}pct', near & tail))
        if frac == max(tail_fracs):
            tail_any = tail
    rows.append(_row(f'near & bulk(rest)', near & ~tail_any))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True,
                    help="glob for per-fold held_out_predictions.parquet files")
    ap.add_argument('--label', default='model')
    ap.add_argument('--out', default=None, help='write a markdown report here')
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob!r}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"pooled {len(files)} folds → {len(df):,} out-of-fold soundings")

    bins = diag.correction_by_cloud_distance(df, df['y_true'].to_numpy(float),
                                             df['mu'].to_numpy(float))
    tail = _tail_breakdown(df)

    y = df['y_true'].to_numpy(float)
    resid = y - df['mu'].to_numpy(float)
    overall = {'pre_rms': float(np.sqrt(np.mean(y ** 2))),
               'post_rms': float(np.sqrt(np.mean(resid ** 2)))}
    overall['rms_reduction_pct'] = 100.0 * (1 - overall['post_rms'] / overall['pre_rms'])

    lines = [f"# Correction effectiveness (pooled out-of-fold) — {args.label}", '',
             f"Pooled {len(files)} folds, {len(df):,} soundings. `pre` = |anomaly| before "
             "correction, `post` = |anomaly − μ̂| after. RMS in ppm.", '',
             f"**Overall**: pre_rms={overall['pre_rms']:.4f} → post_rms="
             f"{overall['post_rms']:.4f}  ({overall['rms_reduction_pct']:+.1f}%)", '',
             '## By cloud distance', '',
             _md_table(bins[['bin', 'n', 'pre_rms', 'post_rms', 'rms_reduction_pct',
                             'post_bias', 'r2']]), '',
             '## Near-cloud: bulk vs left tail', '',
             _md_table(tail), '']
    report = '\n'.join(lines)
    print('\n' + report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[saved → {args.out}]")


if __name__ == '__main__':
    main()
