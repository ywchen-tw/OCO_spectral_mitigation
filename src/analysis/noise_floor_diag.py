"""Noise-floor / residual-structure diagnostic for the XCO2 anomaly target.

Answers: is the ~0.53 unseen-date R² plateau a fundamental ceiling (irreducible
noise / missing features) or a model gap (recoverable)?

(1) kNN noise floor — model-free.  For soundings that are near-identical in
    feature space, how much does y still vary?  That within-neighborhood variance
    is the irreducible noise GIVEN the features → implies a max achievable R².
(2) Residual structure — fit a flexible model, then test whether the held-out
    residuals are still predictable from / correlated with observables.  If yes,
    signal is left on the table; if not, the primary model already extracted it.

Run:  PYTHONPATH=src python -m analysis.noise_floor_diag --data <parquet> --sfc_type 0
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from models.pipeline import FeaturePipeline, _ensure_derived_features, filter_target_outliers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--sfc_type', type=int, default=0)
    ap.add_argument('--knn_sample', type=int, default=40000)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.data)
    df = df[df['sfc_type'] == args.sfc_type]
    df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    df = filter_target_outliers(df)

    pipe = FeaturePipeline.fit(df, sfc_type=args.sfc_type, feature_set='full')
    X = pipe.transform(df)
    y = df['xco2_bc_anomaly'].to_numpy(dtype=np.float32)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]
    dfv = df.loc[valid]
    var_y = float(np.var(y))
    print(f"n={len(y)}  Var(y)={var_y:.4f}  std={np.sqrt(var_y):.4f}")

    # ── (1) kNN noise floor ────────────────────────────────────────────────────
    idx = rng.choice(len(y), min(args.knn_sample, len(y)), replace=False)
    Xs, ys = X[idx], y[idx]
    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(Xs)
    _, nbr = nn.kneighbors(Xs)
    nbr = nbr[:, 1:]                                   # drop self
    local_var = ys[nbr].var(axis=1)                    # within-neighborhood y variance
    irreducible = float(local_var.mean())
    max_r2 = 1.0 - irreducible / var_y
    print("\n=== (1) kNN noise floor (in-distribution, given features) ===")
    print(f"  mean within-{args.k}NN Var(y) = {irreducible:.4f}  (irreducible)")
    print(f"  implied MAX achievable R²    = {max_r2:.3f}")
    # conditional: tail vs bulk
    thr = np.quantile(ys, 0.05)
    tail = ys <= thr
    print(f"  left-tail (bottom 5%) local Var(y) = {local_var[tail].mean():.4f}  "
          f"vs bulk = {local_var[~tail].mean():.4f}")

    # ── (2) residual structure ─────────────────────────────────────────────────
    Xtr, Xte, ytr, yte, _, dte = train_test_split(
        X, y, dfv, test_size=0.25, random_state=args.seed)
    base = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_depth=6, random_state=args.seed)
    base.fit(Xtr, ytr)
    pred = base.predict(Xte)
    resid = yte - pred
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((yte - yte.mean()) ** 2)
    print("\n=== (2) residual structure (random-split HistGBR) ===")
    print(f"  base model R² (random split) = {r2:.3f}")

    # 2a. can a 2nd model predict the residual from the SAME features? (underfit test)
    r_tr = ytr - base.predict(Xtr)
    res_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              max_depth=6, random_state=args.seed + 1)
    res_model.fit(Xtr, r_tr)
    r_pred = res_model.predict(Xte)
    res_r2 = 1.0 - np.sum((resid - r_pred) ** 2) / np.sum((resid - resid.mean()) ** 2)
    print(f"  residual-on-features R²       = {res_r2:.3f}  "
          f"(≈0 → base extracted feature signal; >0 → base underfit)")

    # 2b. |resid| correlation with observables (where does error concentrate?)
    print("  |residual| correlation with observables:")
    for c in ['cld_dist_km', 'aod_total', 'sza', 'vza', 'glint_angle']:
        if c in dte.columns:
            v = dte[c].to_numpy(dtype=float)
            m = np.isfinite(v)
            if m.sum() > 100:
                r = np.corrcoef(np.abs(resid[m]), v[m])[0, 1]
                print(f"    {c:16} r={r:+.3f}")

    # 2c. residual std by cloud proximity (heteroscedasticity → should sigma vary?)
    cd = dte['cld_dist_km'].to_numpy(dtype=float) if 'cld_dist_km' in dte.columns else None
    if cd is not None:
        near = cd <= 10.0
        print(f"  residual std: near_cloud(<=10km)={resid[near].std():.4f}  "
              f"far={resid[~near].std():.4f}")
    print(f"  residual std: left-tail(bottom5% y)={resid[yte<=np.quantile(yte,0.05)].std():.4f}  "
          f"bulk={resid[yte>np.quantile(yte,0.05)].std():.4f}")


if __name__ == '__main__':
    main()
