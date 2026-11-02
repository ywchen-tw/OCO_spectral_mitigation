"""Collect the 2020 fit-method comparison (ridge, GBDT, deep ensemble).

Runs: results_doppler/model_<model>/<fit_method>/fm2020_<surface>_<feature_set>_f<k>/
(launched with --val_split date_kfold --n_folds 4; ocean target r05, land r15).
Each run's run_summary.json points to its held-out metrics json; the point-skill
numbers (global rmse / r2 / mae / n) are read from there.

Prints, per model x surface, every config's fold-mean RMSE, R2, MAE, and against the
no-spectral floor (am + no_spec), the agreed model (am + full) and, for the
fit-uncertainty configs (full_sig, spec_only_sig), the same config without the
uncertainties: the fold-mean RMSE difference and the number of folds in which the
config has the lower RMSE.  The folds hold out the same dates for every config, so
the comparison is paired.  The no-dXCO2 configs (no_xco2, no_xco2_sig) are also
compared with the no-dXCO2 no-spectral floor (am + no_xco2_and_spec), and a
fit method x feature set RMSE grid (full / no_xco2 / *_sig / spec_only) closes the
printout.

The same configurations on the 2016-2020 data use the run prefix fm1620 (--prefix fm1620).

Usage: python workspace/collect_fit_method_2020.py [--prefix fm2020|fm1620] [--out results.csv]
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

ROOT = '/scratch/alpine/yuch8913/oco/spec_anal/results_doppler'
MODELS = {'linear': 'model_linear_baseline', 'gbdt': 'model_gbdt', 'deep_ensemble': 'model_deep_ensemble'}


def collect(prefix: str = 'fm2020') -> pd.DataFrame:
    run_re = re.compile(rf'{prefix}_(ocean|land)_(.+)_f(\d)$')
    rows = []
    for model, mdir in MODELS.items():
        for summ in glob.glob(f'{ROOT}/{mdir}/*/{prefix}_*/run_summary.json'):
            run_dir = os.path.dirname(summ)
            m = run_re.match(os.path.basename(run_dir))
            if not m:
                continue
            surface, feature_set, fold = m.group(1), m.group(2), int(m.group(3))
            method = os.path.basename(os.path.dirname(run_dir))
            with open(summ) as f:
                s = json.load(f)
            mpath = s.get('artifacts', {}).get('metrics_json')
            if not mpath or not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                g = json.load(f)['global']
            rows.append(dict(model=model, surface=surface, fit_method=method,
                             feature_set=feature_set, fold=fold, rmse=g['rmse'], r2=g['r2'],
                             mae=g['mae'], n=g['n']))
    return pd.DataFrame(rows)


def config_name(r) -> str:
    return r.fit_method if r.feature_set == 'full' else f'{r.fit_method}+{r.feature_set}'


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(config=df.apply(config_name, axis=1))
    out = []
    for (model, surface), g in df.groupby(['model', 'surface']):
        piv = g.pivot_table(index='fold', columns='config', values='rmse')
        for config, sub in g.groupby('config'):
            row = dict(model=model, surface=surface, config=config, n_folds=len(sub),
                       rmse=sub.rmse.mean(), rmse_sd=sub.rmse.std(), r2=sub.r2.mean(),
                       mae=sub.mae.mean())
            # *_sig configs are also compared with the same config without uncertainties
            nosig = (config.replace('+full_sig', '').replace('+spec_only_sig', '+spec_only')
                     .replace('+no_xco2_sig', '+no_xco2') if config.endswith('_sig') else None)
            # no-dXCO2 configs are also compared with the no-dXCO2, no-spectral floor
            noxfloor = 'am+no_xco2_and_spec' if '+no_xco2' in config else None
            for ref, tag in (('am+no_spec', 'nospec'), ('am', 'am'), (nosig, 'nosig'),
                             (noxfloor, 'noxfloor')):
                if ref is not None and ref in piv and config != ref:
                    d = (piv[config] - piv[ref]).dropna()
                    row[f'd_rmse_vs_{tag}'] = d.mean() if len(d) else float('nan')
                    row[f'wins_vs_{tag}'] = f'{int((d < 0).sum())}/{len(d)}'
            out.append(row)
    return pd.DataFrame(out).sort_values(['model', 'surface', 'rmse'])


GRID_SETS = ['full', 'no_xco2', 'full_sig', 'no_xco2_sig', 'spec_only', 'spec_only_sig']


def grid(df: pd.DataFrame) -> None:
    """Fold-mean RMSE as fit method x feature set, per model x surface, with the
    two no-spectral floors (with and without dXCO2) underneath."""
    for (model, surface), g in df.groupby(['model', 'surface']):
        piv = g.pivot_table(index='fit_method', columns='feature_set', values='rmse',
                            aggfunc='mean')
        cols = [c for c in GRID_SETS if c in piv]
        if 'no_xco2' not in cols:
            continue
        print(f'--- RMSE grid: {model} / {surface}')
        print(piv[cols].reindex(['am', 'o2', 'o3', 't2', 'legacy']).to_string(
            float_format=lambda v: f'{v:.4f}'))
        am = g[g.fit_method == 'am'].groupby('feature_set').rmse.mean()
        print('  floors (am row):  no_spec', f"{am.get('no_spec', float('nan')):.4f}",
              '  no_xco2_and_spec', f"{am.get('no_xco2_and_spec', float('nan')):.4f}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None, help='write the per-fold table to this CSV')
    ap.add_argument('--prefix', default='fm2020',
                    help='run-directory prefix: fm2020 (2020 data) or fm1620 (2016-2020 data)')
    args = ap.parse_args()
    df = collect(args.prefix)
    if df.empty:
        print(f'no {args.prefix} runs found yet')
        return
    if args.out:
        df.to_csv(args.out, index=False)
    print(df.groupby(['model', 'surface']).size().rename('runs').to_string(), '\n')
    pd.set_option('display.width', 200)
    s = summarise(df)
    for (model, surface), g in s.groupby(['model', 'surface'], sort=False):
        print(f'=== {model} / {surface}')
        print(g.drop(columns=['model', 'surface']).to_string(index=False,
                                                              float_format=lambda v: f'{v:.4f}'))
        print()
    grid(df)


if __name__ == '__main__':
    main()
