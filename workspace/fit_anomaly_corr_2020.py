"""Correlation of the Doppler-variant fit variables with the XCO2 anomaly, land vs ocean.

Input: results_doppler/csv_collection/combined_2020_dates.parquet (fit_schema
doppler_am_t2_v1; build_feature_dataset.py --date-set training_2020).

For each surface (ocean = sfc_type 0, glint; land = sfc_type 1, all modes) and
each candidate feature it reports, against the bias-corrected XCO2 anomaly of
the matching clear-sky pool:
  rho      Spearman rank correlation over all soundings (robust to the heavy
           land anomaly tails)
  r_w      Pearson r after clipping feature and anomaly to their 1-99 %
  rho_day  median of the per-date Spearman rho, and sign_frac, the fraction of
           dates whose rho has the sign of the pooled rho (guards against one
           date driving the pooled number)

Features: every fitted coefficient of the order-2/3 cumulant fits (_o2, _o3;
the counterpart of the old production k1/k2, now unsmoothed), the two-tau fit
(_t2) and the agreed model (_am); for the _am/_t2 coefficients with stored
clear-sky statistics also d_<x> = x - <pfx>_<x>_mean and z_<x> = d / std;
fit uncertainties (sig_*), fit RMS, the fit-derived ratios (k2/k1, exp_int_am vs
albedo), and baseline Lite /
preprocessor predictors for scale.

Targets: xco2_bc_anomaly (ref pool) and the pool each surface's models use
(land r15 = min cloud distance 15 km, ocean r05 = 5 km).  Default subset is
xco2_qf == 0; --all-qf keeps every sounding.

Usage:
  python workspace/fit_anomaly_corr_2020.py [--parquet P] [--out DIR] [--all-qf]
Writes <out>/corr_<surface>_<target>.csv and a summary to stdout.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

DEFAULT_PARQUET = ('/scratch/alpine/yuch8913/oco/spec_anal/results_doppler/'
                   'csv_collection/combined_2020_dates.parquet')
BANDS = ('o2a', 'wco2', 'sco2')
SURFACES = {'ocean': (0, 'r05'), 'land': (1, 'r15')}
BASELINE = ['dp', 'dp_o2a', 'dp_sco2', 'dpfrac', 'co2_grad_del', 'alb_o2a', 'alb_wco2',
            'alb_sco2', 'aod_total', 'h2o_ratio_bc', 'co2_ratio_bc', 'fs_rel', 'airmass',
            'delT', 'dws', 'csnr_o2a', 'csnr_wco2', 'csnr_sco2', 'h_cont_o2a', 'h_cont_wco2',
            'h_cont_sco2', 'chi2_o2a', 'chi2_wco2', 'chi2_sco2', 'rms_rel_o2a',
            'rms_rel_wco2', 'rms_rel_sco2', 'cld_dist_km']


def family(name):
    """Feature family label used in the summary."""
    for tag, fam in (('_o2', 'cumulant o2'), ('_o3', 'cumulant o3'), ('_t2', 'two-tau t2'),
                     ('_am', 'agreed model am')):
        if name.endswith(tag) or name.endswith(tag + '_mean'):
            base = fam
            break
    else:
        base = None
    if name.startswith(('d_', 'z_')):
        return f'{base} (vs clear-sky ref)'
    if '_sig_' in name:
        return 'fit sigma'
    if '_rms_' in name:
        return 'fit rms'
    if 'over' in name or 'minus' in name:
        return 'fit-derived ratios'
    if name in BASELINE:
        return 'Lite / preprocessor baseline'
    return base or 'other'


def spearman(x, y):
    """Spearman rho of two float arrays (NaNs removed pairwise); (rho, n)."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 200:
        return np.nan, n
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), n


def pearson_clipped(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return np.nan
    xs, ys = x[m], y[m]
    xs = np.clip(xs, *np.percentile(xs, [1, 99]))
    ys = np.clip(ys, *np.percentile(ys, [1, 99]))
    if xs.std() == 0 or ys.std() == 0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def build_features(df, pfx):
    """Candidate feature columns -> float64 arrays."""
    feats = {}
    for c in df.columns:
        band, _, field = c.partition('_')
        if band in BANDS and field not in ('nchan', 'ok_am'):
            feats[c] = df[c].to_numpy(np.float64)          # coefficients, sig_*, rms_*
        elif c in BASELINE or ('over' in c or 'minus' in c) and ('k2' in c or 'exp_int' in c):
            feats[c] = df[c].to_numpy(np.float64)
    for b in BANDS:
        for c in list(df.columns):
            if not (c.startswith(f'{pfx}_{b}_') and c.endswith('_mean')):
                continue
            field = c[len(f'{pfx}_{b}_'):-len('_mean')]
            x = f'{b}_{field}'
            if x not in df.columns:
                continue
            d = df[x].to_numpy(np.float64) - df[c].to_numpy(np.float64)
            s = df[f'{pfx}_{b}_{field}_std'].to_numpy(np.float64)
            feats[f'd_{x}'] = d
            with np.errstate(divide='ignore', invalid='ignore'):
                feats[f'z_{x}'] = np.where(s > 0, d / s, np.nan)
    return feats


def analyse(df, target, pfx):
    y = df[target].to_numpy(np.float64)
    dates = df['date_str'].to_numpy()
    rows = []
    for name, x in build_features(df, pfx).items():
        rho, n = spearman(x, y)
        if not np.isfinite(rho):
            continue
        per = []
        for d in np.unique(dates):
            m = dates == d
            r, k = spearman(x[m], y[m])
            if np.isfinite(r):
                per.append(r)
        per = np.array(per)
        rows.append(dict(feature=name, family=family(name), n=n, rho=rho,
                         r_w=pearson_clipped(x, y),
                         rho_day=float(np.median(per)) if per.size else np.nan,
                         sign_frac=float(np.mean(np.sign(per) == np.sign(rho))) if per.size else np.nan,
                         n_days=int(per.size)))
    out = pd.DataFrame(rows)
    out['abs_rho'] = out.rho.abs()
    return out.sort_values('abs_rho', ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--parquet', default=DEFAULT_PARQUET)
    ap.add_argument('--out', default=None, help='output dir (default: next to this script, '
                                                'fit_anomaly_corr_2020/)')
    ap.add_argument('--all-qf', action='store_true', help='keep xco2_qf == 1 soundings too')
    args = ap.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'fit_anomaly_corr_2020' + ('_allqf' if args.all_qf else ''))
    os.makedirs(out_dir, exist_ok=True)

    dset = ds.dataset(args.parquet, format='parquet')
    names = dset.schema.names
    for surf, (code, pool) in SURFACES.items():
        flt = ds.field('sfc_type') == code
        if not args.all_qf:
            flt = flt & (ds.field('xco2_qf') == 0)
        keep_ref = ('ref_', f'{pool}_')
        cols = [c for c in names
                if c.split('_')[0] in BANDS or c in BASELINE or c.startswith(keep_ref)
                or (('over' in c or 'minus' in c) and ('k2' in c or 'exp_int' in c))
                or c in ('date', 'xco2_bc_anomaly',
                                                          f'xco2_bc_anomaly_{pool}')]
        df = dset.to_table(columns=sorted(set(cols)), filter=flt).to_pandas()
        df['date_str'] = df['date'].map(lambda v: v.decode() if isinstance(v, bytes) else str(v)).str[:10]
        print(f'\n=== {surf}: {len(df):,} soundings'
              f'{"" if args.all_qf else " (xco2_qf == 0)"}, {df.date_str.nunique()} dates')
        for target, pfx in (('xco2_bc_anomaly', 'ref'), (f'xco2_bc_anomaly_{pool}', pool)):
            res = analyse(df, target, pfx)
            res.to_csv(os.path.join(out_dir, f'corr_{surf}_{target}.csv'), index=False)
            print(f'\n--- {surf}, target {target} (finite: {np.isfinite(df[target]).sum():,})')
            best = (res.sort_values('abs_rho', ascending=False)
                       .groupby('family', sort=False).head(1)
                       .sort_values('abs_rho', ascending=False))
            print('best |rho| per family:')
            print(best[['family', 'feature', 'rho', 'r_w', 'rho_day', 'sign_frac', 'n']]
                  .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
            print('top 15 features:')
            print(res.head(15)[['feature', 'family', 'rho', 'r_w', 'rho_day', 'sign_frac']]
                  .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        del df
    print(f'\nfull tables: {out_dir}')


if __name__ == '__main__':
    main()
