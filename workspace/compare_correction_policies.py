"""Compare three near-cloud XCO2 correction policies on OUT-OF-DISTRIBUTION data.

For each surface (ocean sfc=0, land sfc=1) the trained date_kfold fold models are
applied to their OWN held-out date block, so every sounding is scored by a model
that never saw its date.  Two models are run jointly on the SAME rows (a shared
finite-feature mask), so the policies are compared apples-to-apples:

  mu      = deep-ensemble predicted xco2_bc_anomaly  (the correction magnitude)
  P(near) = xgb_cloud P(cld_dist_km <= gate)         (deployable, no MODIS)

Policies (correction subtracted from xco2_bc; equivalently residual = y - corr):
  (1) corr = mu                       — current correction (full DE everywhere)
  (2) corr = P(near) * mu             — confidence-weighted (FT1)
  (3) corr = mu if P(near) > 0.5      — hard gate

Metric: how close the corrected anomaly is to truth.  Lower residual RMS / higher
R2 is better.  R2 uses the same demeaned convention as diagnostics.compute_metrics
so the numbers line up with the plan's near-cloud R2 of record.  "do-nothing" RMS
(= RMS of y about 0) is the uncorrected error the policy must beat.

The fold models hold out the same date blocks for DE and xgb (both call
split_dataframe(date_kfold) on identically-preprocessed frames), so per fold the
held block is identical; only the finite-feature mask differs -> we intersect it.

Run (local): PYTHONPATH=src python workspace/compare_correction_policies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from models.pipeline import (FeaturePipeline, _ensure_derived_features,  # noqa: E402
                             filter_target_outliers)
from models.splits import split_dataframe  # noqa: E402
from models.deep_ensemble import ensemble_predict  # noqa: E402
from apply.apply_deep_ensemble import _load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'results/csv_collection/combined_2016_2020_dates.parquet'
OUTDIR = ROOT / 'results/model_comparison'
N_FOLDS = 5
DEVICE = torch.device('cpu')

# Per-surface config: the FINAL deployable models (DE beta_nll 64,32 + xgb final
# expanded-feature classifier at the per-surface gate threshold ocean5/land15).
SURFACES = [
    dict(name='ocean', sfc=0, gate=5.0,
         de='de_ocean_beta_nll', xgb='xgbcloud_final_ocean'),
    dict(name='land', sfc=1, gate=15.0,
         de='de_land_beta_nll', xgb='xgbcloud_final_land'),
]

# Distance strata for the report (km).  None = no upper bound.
BINS = [(0, 2), (2, 5), (5, 10), (10, 15), (15, None)]


def _xgb_features(pipeline, frame, cloud_cols):
    X = pipeline.transform(frame)
    if cloud_cols:
        X = np.concatenate(
            [X, frame[list(cloud_cols)].to_numpy(dtype=np.float32)], axis=1)
    return X


def _metrics(y, corr):
    """Residual = y - corr (corr is the predicted anomaly to remove)."""
    resid = y - corr
    n = y.size
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))          # residual scatter (about pred)
    bias = float(np.mean(resid))
    prerms = float(np.sqrt(np.mean(y ** 2)))            # do-nothing error (about 0)
    postrms = rmse                                      # residual already about 0-target
    red = 100.0 * (1 - postrms / prerms) if prerms > 0 else np.nan
    return dict(n=n, r2=r2, rmse=rmse, bias=bias,
                prerms=prerms, postrms=postrms, reduction_pct=red)


def _collect_surface(cfg):
    """Return pooled OOD arrays (y, cd, mu, P) over all folds for one surface."""
    print(f"\n=== {cfg['name']} (sfc={cfg['sfc']}, gate={cfg['gate']}km) ===")
    df = pd.read_parquet(DATA, filters=[('sfc_type', '==', cfg['sfc']),
                                        ('snow_flag', '==', 0)])
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)
    print(f"  loaded {len(df):,} rows after snow/outlier filter")

    ys, cds, mus, Ps = [], [], [], []
    for f in range(N_FOLDS):
        _, held = split_dataframe(df, mode='date_kfold', n_folds=N_FOLDS, fold=f)

        de_dir = ROOT / 'results/model_deep_ensemble' / f"{cfg['de']}_f{f}"
        xg_dir = ROOT / 'results/model_xgb_cloud' / f"{cfg['xgb']}_f{f}"

        de_pipe, members, meta = _load_model(de_dir)
        xg_pipe = FeaturePipeline.load(xg_dir / 'xgb_cloud_pipeline.pkl')
        import pickle
        xg_meta = pickle.load(open(xg_dir / 'xgb_cloud_meta.pkl', 'rb'))
        clf = xgb.XGBClassifier()
        clf.load_model(str(xg_dir / 'xgb_cloud_model.json'))

        X_de = de_pipe.transform(held)
        X_xg = _xgb_features(xg_pipe, held, xg_meta.get('cloud_cols', ()))
        y = held['xco2_bc_anomaly'].to_numpy(dtype=float)
        cd = held['cld_dist_km'].to_numpy(dtype=float)

        shared = (np.all(np.isfinite(X_de), axis=1)
                  & np.all(np.isfinite(X_xg), axis=1)
                  & np.isfinite(y))
        mu, _ = ensemble_predict(members, X_de[shared], DEVICE,
                                 loss=meta.get('loss', 'gaussian_nll'),
                                 nu=meta.get('nu', 4.0))
        P = clf.predict_proba(X_xg[shared])[:, 1]

        ys.append(y[shared]); cds.append(cd[shared])
        mus.append(mu.astype(float)); Ps.append(P.astype(float))
        print(f"  fold {f}: {int(shared.sum()):,} OOD soundings scored "
              f"(dropped {int((~shared).sum()):,})")

    return (np.concatenate(ys), np.concatenate(cds),
            np.concatenate(mus), np.concatenate(Ps))


def _stratum_rows(name, surf, y, cd, mu, P):
    """One (surface, stratum) -> rows for all 3 policies + do-nothing."""
    gate = surf['gate']
    policies = {
        '1_full_mu': mu,
        '2_Pnear_mu': P * mu,
        '3_gate_mu': np.where(P > 0.5, mu, 0.0),
    }
    strata = {}
    for lo, hi in BINS:
        m = (cd >= lo) if lo is not None else np.ones_like(cd, bool)
        if hi is not None:
            m &= (cd < hi)
        # NaN cld_dist -> treat as far (>15): only matches the (15,None) bin
        if hi is None and lo == 15:
            m |= ~np.isfinite(cd)
        strata[f'[{lo},{hi if hi is not None else "inf"})'] = m
    strata['near(<=gate)'] = np.isfinite(cd) & (cd <= gate)
    strata['<=10km'] = np.isfinite(cd) & (cd <= 10.0)
    strata['far(>gate)'] = (~np.isfinite(cd)) | (cd > gate)
    strata['all'] = np.ones_like(cd, bool)

    rows = []
    for sname, m in strata.items():
        if m.sum() < 50:
            continue
        for pname, corr in policies.items():
            r = _metrics(y[m], corr[m])
            rows.append(dict(surface=name, stratum=sname, policy=pname, **r))
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for surf in SURFACES:
        y, cd, mu, P = _collect_surface(surf)
        all_rows += _stratum_rows(surf['name'], surf, y, cd, mu, P)

    res = pd.DataFrame(all_rows)
    out = OUTDIR / 'correction_policy_comparison.csv'
    res.to_csv(out, index=False)

    # Pretty print: focus strata, R2 + residual RMS + bias per policy.
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 200)
    show = res[res['stratum'].isin(
        ['near(<=gate)', '<=10km', 'far(>gate)', 'all'])].copy()
    for c in ('r2', 'rmse', 'bias', 'prerms', 'reduction_pct'):
        show[c] = show[c].round(4)
    print("\n================ CORRECTION POLICY COMPARISON (OOD) ================")
    print(show[['surface', 'stratum', 'policy', 'n', 'prerms',
                'rmse', 'reduction_pct', 'r2', 'bias']].to_string(index=False))
    print(f"\nFull per-bin table -> {out}")


if __name__ == '__main__':
    main()
