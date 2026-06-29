"""Aggregate the land DE feature-set comparison (re-run WITH snow=1 data).

Reads the per-fold output dirs produced by curc_shell_blanca_de_land_snow_features.sh
(de_land_snow_{feature_set}_f{0..4}) and reports, per feature set, the fold
mean +/- std of:
  - global metrics (RMSE, MAE, R2, coverage_90)               [de_mondrian_*_metrics.json -> 'global']
  - near-cloud metrics (<=10km RMSE/R2, near&bottom_5pct R2)  [*_stratified_metrics.csv]
  - closest-bin correction ([0,2)km RMS-reduction %)         [de_correction_clddist_*.csv]

For before/after context it also aggregates the OLD no-snow land dirs
(de_land_{feature_set}_f*) when present, so the snow effect is visible per feature set.

Usage (after the array finishes):
    PYTHONPATH=src python workspace/aggregate_land_snow_features.py \
        [--de-dir results/model_deep_ensemble] \
        [--out results/model_comparison/de_land_snow_feature_comparison.md]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SETS = ['full_contam', 'no_spec', 'no_xco2', 'no_xco2_and_spec']
PREFIX = 'de_mondrian_date_kfold'                 # headline metric set (Mondrian-CQR)
STRAT = f'{PREFIX}_stratified_metrics.csv'
GJSON = f'{PREFIX}_metrics.json'
CLDDIST = 'de_correction_clddist_date_kfold.csv'


def _fold_dirs(de_dir: Path, pattern: str) -> list:
    return sorted(p for p in glob.glob(str(de_dir / pattern)) if Path(p).is_dir())


def _collect(dirs: list) -> dict:
    """Return {metric: [per-fold values]} for global + near-cloud + closest-bin."""
    acc: dict = {k: [] for k in
                 ['rmse', 'mae', 'r2', 'coverage_90',
                  'near_rmse', 'near_r2', 'neartail5_r2', 'bin02_rms_red']}
    for d in dirs:
        d = Path(d)
        gj = d / GJSON
        if gj.exists():
            g = json.load(open(gj)).get('global', {})
            for k in ('rmse', 'mae', 'r2', 'coverage_90'):
                if isinstance(g.get(k), (int, float)):
                    acc[k].append(float(g[k]))
        sc = d / STRAT
        if sc.exists():
            s = pd.read_csv(sc)
            def _val(regime, group, col):
                r = s[(s.regime == regime) & (s.group == group)]
                return float(r.iloc[0][col]) if len(r) else np.nan
            acc['near_rmse'].append(_val('cloud_proximity', 'near_cloud(<=10km)', 'rmse'))
            acc['near_r2'].append(_val('cloud_proximity', 'near_cloud(<=10km)', 'r2'))
            acc['neartail5_r2'].append(_val('near_cloud_tail', 'near&bottom_5pct', 'r2'))
        cc = d / CLDDIST
        if cc.exists():
            c = pd.read_csv(cc)
            r = c[c.bin == '[0,2)km']
            if len(r):
                acc['bin02_rms_red'].append(float(r.iloc[0]['rms_reduction_pct']))
    return acc


def _ms(vals: list) -> 'tuple[float, float, int]':
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan, 0)
    return (float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0), int(a.size))


def _fmt(vals: list, p: int = 4) -> str:
    m, s, n = _ms(vals)
    if n == 0:
        return '—'
    return f'{m:.{p}f} ± {s:.{p}f}'


def _row(label: str, acc: dict) -> str:
    n = max((_ms(acc[k])[2] for k in acc), default=0)
    return (f"| {label} (n={n}) | {_fmt(acc['rmse'])} | {_fmt(acc['r2'])} | "
            f"{_fmt(acc['near_rmse'])} | {_fmt(acc['near_r2'])} | "
            f"{_fmt(acc['neartail5_r2'])} | {_fmt(acc['bin02_rms_red'], 2)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--de-dir', default='results/model_deep_ensemble')
    ap.add_argument('--out', default='results/model_comparison/de_land_snow_feature_comparison.md')
    args = ap.parse_args()
    de_dir = Path(args.de_dir)

    lines = [
        "# Land DE feature-set comparison — WITH snow=1 data (`--include_snow`)",
        "",
        "5-fold `date_kfold`, beta_nll (beta=1.0), M=5, Mondrian-CQR by `cld_dist_km`, "
        "`near_cloud_target=0.98`. Recipe identical to the current ocean comparison; "
        "the only change vs the old land comparison is keeping snow/ice footprints.",
        "",
        "Metrics are fold mean ± std (headline = `de_mondrian` set). `near` = soundings "
        "within 10 km of cloud; `[0,2)km RMS↓` = RMS-reduction % in the closest bin.",
        "",
        "| feature set | RMSE | R² | near≤10km RMSE | near≤10km R² | near&tail5% R² | [0,2)km RMS↓ % |",
        "|---|---|---|---|---|---|---|",
    ]
    found_any = False
    for fs in FEATURE_SETS:
        dirs = _fold_dirs(de_dir, f'de_land_snow_{fs}_f*')
        if not dirs:
            lines.append(f"| {fs} | _no runs found_ | | | | | |")
            continue
        found_any = True
        lines.append(_row(fs, _collect(dirs)))
    lines.append("")

    # Before/after: old no-snow land runs, same feature sets, for context.
    old_rows = []
    for fs in FEATURE_SETS:
        dirs = _fold_dirs(de_dir, f'de_land_{fs}_f*')
        # exclude the snow dirs that share the de_land_ prefix
        dirs = [d for d in dirs if '_snow_' not in Path(d).name]
        if dirs:
            old_rows.append(_row(f'{fs} (old, no-snow)', _collect(dirs)))
    if old_rows:
        lines += [
            "## Reference: old no-snow land comparison (same feature sets)",
            "",
            "| feature set | RMSE | R² | near≤10km RMSE | near≤10km R² | near&tail5% R² | [0,2)km RMS↓ % |",
            "|---|---|---|---|---|---|---|",
            *old_rows,
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if found_any:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"\n[saved → {args.out}]")
    else:
        print("\n[no de_land_snow_* runs found — run the SLURM array first]")


if __name__ == '__main__':
    main()
