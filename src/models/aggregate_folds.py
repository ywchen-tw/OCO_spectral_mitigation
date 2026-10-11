"""Aggregate per-fold (or per-seed) run metrics into mean ± std.

Reference: no external reference (standard cross-validation aggregation).

The date_kfold runner writes one output dir per fold (one fold per invocation,
distinct ``--suffix``).  This script collects the ``global`` block and the
per-regime stratified CSV across those dirs and reports mean ± std, so the
marginal date-split edges can be judged against fold variance instead of a
single draw.  Also works for seed sweeps (point each ``--dirs`` glob at the
``*_s{0,1,2}`` dirs).

Usage
-----
    PYTHONPATH=src python -m models.aggregate_folds \
        --dirs 'results/model_tabm/tabm_ocean_kfold_f*' \
        --label 'TabM k16 (date_kfold)' \
        --out results/model_comparison/tabm_ocean_kfold_agg.md

    # compare several models in one report (repeat --dirs/--label in order):
    PYTHONPATH=src python -m models.aggregate_folds \
        --dirs 'results/model_tabm/tabm_ocean_kfold_f*'  --label TabM \
        --dirs 'results/model_gbdt/gbdt_ocean_kfold_f*'  --label XGB \
        --dirs 'results/model_mlp_baseline/mlp_ocean_kfold_f*' --label MLP \
        --out results/model_comparison/ocean_kfold_agg.md
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Global metrics worth reporting (skip if absent, e.g. interval metrics for MLP).
_GLOBAL_KEYS = ['rmse', 'mae', 'r2', 'coverage_90', 'mean_interval_width',
                'crossing_rate', 'pinball_q05', 'pinball_q50', 'pinball_q95']
_STRAT_METRICS = ['rmse', 'mae', 'r2', 'coverage_90']


def _find_metrics_json(d: Path) -> 'Path | None':
    """Return the primary metrics JSON in dir ``d`` (exclude rearranged/stratified)."""
    cands = [p for p in d.glob('*_metrics.json')
             if 'rearranged' not in p.name and 'stratified' not in p.name]
    return sorted(cands)[0] if cands else None


def _find_stratified_csv(d: Path) -> 'Path | None':
    cands = sorted(d.glob('*_stratified_metrics.csv'))
    return cands[0] if cands else None


def _collect(dirs: list) -> tuple:
    """Return (global_stack: dict[key]->list of (dirname, value), strat_frames, used_dirs).

    Values are kept paired with their fold dir so a divergent fold can be named
    in the outlier flag (see _mean_std).
    """
    global_stack: dict = {}
    strat_frames = []
    used = []
    for d in dirs:
        d = Path(d)
        mj = _find_metrics_json(d)
        if mj is None:
            continue
        g = json.load(open(mj)).get('global', {})
        if not g:
            continue
        used.append(d.name)
        for k in _GLOBAL_KEYS:
            v = g.get(k)
            if isinstance(v, (int, float)):
                global_stack.setdefault(k, []).append((d.name, float(v)))
        sc = _find_stratified_csv(d)
        if sc is not None:
            strat_frames.append(pd.read_csv(sc))
    return global_stack, strat_frames, used


def _mean_std(stack: dict) -> dict:
    """Per-metric mean/std/median/min/max plus a Tukey far-out outlier flag.

    ``median`` is the outlier-robust summary: a single pathological fold (e.g. a
    lone out-of-domain footprint blowing up a linear model's held-fold RMSE to
    1e3+) wrecks the mean but not the median.  ``outliers`` lists (dir, value)
    for folds that are an *order of magnitude beyond the median's scale*
    (|v-median| > 10·(|median|+1)).  This scale-aware rule fires only on genuine
    blowups (RMSE 2486 vs median 0.69) and NOT on normal fold-to-fold variance
    (an IQR/std fence would false-positive at n=5), so a benign spread is never
    mislabelled while a poisoned fold is always caught and named.
    """
    out = {}
    for k, pairs in stack.items():
        names = [p[0] for p in pairs]
        a = np.asarray([p[1] for p in pairs], dtype=float)
        med = float(np.median(a))
        thresh = 10.0 * (abs(med) + 1.0)
        outliers = [(names[i], float(a[i])) for i in range(len(a))
                    if abs(a[i] - med) > thresh]
        out[k] = {'mean': float(a.mean()), 'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
                  'n': int(len(a)), 'min': float(a.min()), 'max': float(a.max()),
                  'median': med, 'outliers': outliers}
    return out


def _agg_stratified(frames: list) -> 'pd.DataFrame | None':
    """Stack per-fold stratified CSVs → mean/std per (regime, group, metric)."""
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True)
    metrics = [m for m in _STRAT_METRICS if m in big.columns]
    g = big.groupby(['regime', 'group'])[metrics].agg(['mean', 'std', 'count'])
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    return g.reset_index()


def _fmt(stat: dict, p: int = 4) -> str:
    return f"{stat['mean']:.{p}f} ± {stat['std']:.{p}f}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-fold/seed metrics into mean ± std.")
    ap.add_argument('--dirs', action='append', required=True,
                    help='Glob of fold/seed output dirs. Repeat with --label per model.')
    ap.add_argument('--label', action='append', default=None,
                    help='Label for the matching --dirs (repeat in the same order).')
    ap.add_argument('--out', type=str, default=None, help='Write a markdown report here.')
    args = ap.parse_args()

    labels = args.label or [f"group{i}" for i in range(len(args.dirs))]
    if len(labels) != len(args.dirs):
        ap.error(f"got {len(args.dirs)} --dirs but {len(labels)} --label (must match)")

    lines = ["# Fold / seed aggregation (mean ± std)", ""]
    per_label_global = {}
    per_label_strat = {}
    for pattern, label in zip(args.dirs, labels):
        matched = sorted(glob.glob(pattern))
        gstack, sframes, used = _collect(matched)
        if not used:
            lines.append(f"## {label}\n\n_No metrics found for_ `{pattern}` "
                         f"(matched {len(matched)} dir(s)).\n")
            continue
        gms = _mean_std(gstack)
        per_label_global[label] = gms
        per_label_strat[label] = _agg_stratified(sframes)
        n = next(iter(gms.values()))['n'] if gms else 0
        lines.append(f"## {label}  (n={n} runs: {', '.join(used)})\n")
        lines.append("| metric | mean ± std | median | min | max |")
        lines.append("|---|---|---|---|---|")
        flagged = []
        for k in _GLOBAL_KEYS:
            if k in gms:
                s = gms[k]
                lines.append(f"| {k} | {_fmt(s)} | {s['median']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
                if s['outliers']:
                    flagged.append((k, s))
        lines.append("")
        for k, s in flagged:
            bad = ", ".join(f"{nm} ({val:.4g})" for nm, val in s['outliers'])
            lines.append(f"> ⚠ **{k}** has outlier fold(s): {bad} — mean is not "
                         f"robust here; use the **median ({s['median']:.4f})**. "
                         f"(A lone out-of-domain footprint can blow up a fold's "
                         f"held-out {k}.)")
        if flagged:
            lines.append("")

    # Side-by-side global comparison if >1 label resolved.  Cells show mean ± std,
    # with the robust median in parentheses when it diverges from the mean (an
    # outlier fold is present) so a poisoned mean is never read at face value.
    if len(per_label_global) > 1:
        lines.append("## Global comparison (mean ± std; median in () when an outlier fold skews the mean)\n")
        labs = list(per_label_global)
        lines.append("| metric | " + " | ".join(labs) + " |")
        lines.append("|---" * (len(labs) + 1) + "|")
        for k in _GLOBAL_KEYS:
            cells = []
            present = False
            for lab in labs:
                gm = per_label_global[lab]
                if k in gm:
                    s = gm[k]; cell = _fmt(s)
                    if s['outliers']:
                        cell += f" (med {s['median']:.4f})"
                    cells.append(cell); present = True
                else:
                    cells.append("—")
            if present:
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"\n[saved → {args.out}]")


if __name__ == '__main__':
    main()
