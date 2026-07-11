"""Permutation feature importance for the DE / XGB-mean / Ridge model trio.

Motivation (manuscript comparison, results/model_comparison/
MODEL_COMPARISON_manuscript_DE_XGB_LinReg.md): the three models are point-
comparable only through mu, so the one importance method defined identically
for all of them is permutation importance — ΔRMSE on a fold's held-out dates
when one feature (or feature group) is shuffled.  Model-native importances
(Ridge |coef|, XGB total_gain) are emitted alongside as corroboration; the DE
has no native equivalent, which is why permutation is the common currency.

Protocol
--------
- Data: each date_kfold fold's held dates (from the fold dir's
  training_dates.json) pulled from the combined 2016-2020 parquet — unseen by
  that fold's model; the 5 folds tile all 116 dates.
- Each model is transformed by its OWN fold pipeline pkl; the three pipelines
  are asserted feature-identical per fold, and the reconstructed held set must
  REPRODUCE the stored held RMSE (run_summary.json) before any permutation is
  trusted (--rmse-tol gate).
- Groups reuse pipeline.py's own ablation sets (xco2 / spec / contam) plus
  geometry, profile-EOF block, fp one-hots; the remainder is 'met_other'.
  Group permutation (columns shuffled jointly) is the headline number because
  the features are collinear; per-feature importance is kept for relative
  individual contributions.
- Strata: 'global' (all held rows) and 'nearcloud' (0 <= cld_dist_km <= 10).
  Shuffling is WITHIN the evaluated stratum.
- The same permutation index arrays (seeded by fold/stratum/repeat) are used
  across models and features, so cross-model differences are not shuffle noise.

Usage
-----
  # one fold, all three models (validation gate + importance CSVs):
  PYTHONPATH=src python -m models.feature_importance \
      --surface ocean --fold 0 --models de,xgb,linreg

  # all folds (loop), then aggregate across folds into the comparison table:
  PYTHONPATH=src python -m models.feature_importance --surface ocean --aggregate

Outputs (under <storage>/results/model_comparison/feature_importance/<surface>/)
  importance_{model}_{surface}_f{fold}.csv     per-run long format
  native_{model}_{surface}_f{fold}.csv         Ridge coefs / XGB gain
  importance_{model}_{surface}_agg.csv         mean ± sd across folds
  FEATURE_IMPORTANCE_TABLE_{surface}.md        3-model comparison table
"""

import argparse
import json
import logging
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .pipeline import (FeaturePipeline, _ensure_derived_features,
                       filter_target_outliers,
                       XCO2_FEATURES, SPEC_FEATURES, CONTAM_FEATURES)

from utils import get_storage_dir

logger = logging.getLogger(__name__)

SEED = 42
NEAR_CLOUD_KM = 10.0

# Geometry / viewing-angle block (union over surfaces; names absent from a
# pipeline's feature list are ignored at assignment time).
GEOMETRY_FEATURES = frozenset([
    'cos_glint_angle', '1_over_cos_sza', '1_over_cos_vza', 'sin_raa',
    'pol_ang_rad', 's31', 'fp_area_km2',
])

# Fold-dir name templates (production date_kfold runs; ocean = r05 target,
# land = r15 target — see curc_shell_blanca_*_foldpca_r{05,15}.sh).
MODEL_DIRS = {
    'de':     'model_deep_ensemble/de_{surf}_beta_nll_prof_reg_foldpca_{r}_f{fold}',
    'xgb':    'model_gbdt/xgb_{surf}_full_prof_foldpca_{r}_f{fold}',
    'linreg': 'model_linear_baseline/linreg_{surf}_full_prof_foldpca_{r}_f{fold}',
}
PIPELINE_FILES = {'de': 'deep_ensemble_pipeline.pkl',
                  'xgb': 'gbdt_pipeline.pkl',
                  'linreg': 'linear_pipeline.pkl'}
SURF = {'ocean': dict(sfc_type=0, r='r05', target='xco2_bc_anomaly_r05'),
        'land':  dict(sfc_type=1, r='r15', target='xco2_bc_anomaly_r15')}
N_FOLDS = 5


# ─── Model loading → uniform predict_mu(X) ─────────────────────────────────────

def _load_de(model_dir: Path):
    """Deep ensemble: member_*.pt + meta → predict_mu via ensemble mixture mean.

    Mirrors apply_deep_ensemble._load_model (kept local: that module is a CLI
    script under src/apply, not an importable sibling of models/).
    """
    import pickle
    import torch
    from .deep_ensemble import GaussianMLP, ensemble_predict

    with open(model_dir / 'deep_ensemble_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    if meta.get('cloud_bin_feature', 'none') not in ('none', None, False):
        raise SystemExit(f"{model_dir}: cloud_bin_feature models need cloud "
                         "distance at inference — not supported here.")
    pipeline = FeaturePipeline.load(model_dir / PIPELINE_FILES['de'])
    hidden_dims = tuple(meta.get('hidden_dims', (64, 32)))
    members = []
    for p in sorted(model_dir.glob('member_*.pt')):
        m = GaussianMLP(pipeline.n_features, hidden_dims=hidden_dims,
                        aux_cloud=bool(meta.get('aux_cloud', False)),
                        dropout=float(meta.get('dropout', 0.0)),
                        norm=meta.get('norm', 'none'))
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        m.eval()
        members.append(m)
    if not members:
        raise SystemExit(f"no member_*.pt in {model_dir}")
    device = torch.device('cpu')
    loss, nu = meta.get('loss', 'gaussian_nll'), meta.get('nu', 4.0)

    def predict_mu(X):
        mu, _ = ensemble_predict(members, X, device, loss=loss, nu=nu)
        return mu

    return pipeline, predict_mu


def _load_xgb(model_dir: Path):
    pipeline = FeaturePipeline.load(model_dir / PIPELINE_FILES['xgb'])
    model = joblib.load(model_dir / 'model_mean_xgboost.joblib')
    return pipeline, lambda X: model.predict(X)


def _load_linreg(model_dir: Path):
    pipeline = FeaturePipeline.load(model_dir / PIPELINE_FILES['linreg'])
    model = joblib.load(model_dir / 'model_ridge.joblib')
    return pipeline, lambda X: model.predict(X)


_LOADERS = {'de': _load_de, 'xgb': _load_xgb, 'linreg': _load_linreg}


def _stored_held_rmse(model_dir: Path) -> 'float | None':
    """Held-fold RMSE recorded at train time (run_summary primary metric)."""
    try:
        with open(model_dir / 'run_summary.json', encoding='utf-8') as f:
            return float(json.load(f)['primary_metric_value'])
    except (OSError, KeyError, ValueError):
        return None


# ─── Held-set reconstruction ───────────────────────────────────────────────────

def load_held_frame(data_path, held_dates: list, sfc_type: int,
                    target_col: str, exclude_snow: bool = False) -> pd.DataFrame:
    """Rows of the combined parquet belonging to the fold's held dates, with the
    same filter chain the trainers applied (sfc_type → [snow] → derived features
    → target outliers).  Date-filter pushdown avoids reading the full 26 GB."""
    date_vals = [d.encode() for d in held_dates]      # 'date' is stored binary
    df = pd.read_parquet(
        data_path,
        filters=[('date', 'in', date_vals), ('sfc_type', '==', float(sfc_type))],
    )
    logger.info("held frame: %d rows, %d dates", len(df), len(held_dates))
    if exclude_snow:
        df = df[df['snow_flag'] == 0]
    df = _ensure_derived_features(df)
    if target_col not in df.columns:
        raise SystemExit(f"target column {target_col!r} missing from {data_path}")
    df = filter_target_outliers(df, target_col=target_col)
    return df


# ─── Grouping ──────────────────────────────────────────────────────────────────

def assign_groups(pipeline: FeaturePipeline) -> dict:
    """feature name → group name, exhaustively over pipeline.features."""
    profile = set(getattr(pipeline, 'profile_names', []) or [])
    fp_cols = set(pipeline.fp_cols)
    out = {}
    for f in pipeline.features:
        if f in XCO2_FEATURES:
            out[f] = 'xco2'
        elif f in SPEC_FEATURES:
            out[f] = 'spec'
        elif f in CONTAM_FEATURES:
            out[f] = 'contam'
        elif f in GEOMETRY_FEATURES:
            out[f] = 'geometry'
        elif f in profile:
            out[f] = 'profile'
        elif f in fp_cols:
            out[f] = 'fp_onehot'
        else:
            out[f] = 'met_other'
    return out


# ─── Permutation engine ────────────────────────────────────────────────────────

def _rmse(y, mu):
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def _r2(y, mu):
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - mu) ** 2)) / ss if ss > 0 else np.nan


def permutation_importance(predict_mu, X, y, features: list, groups: dict,
                           strata: dict, n_repeats: int, fold: int,
                           seed: int = SEED) -> pd.DataFrame:
    """Permutation ΔRMSE/ΔR² per feature and per group, per stratum.

    strata : {name: bool row mask over X}.  Shuffling happens within the
    stratum's rows only, and evaluation is on those rows.  Permutation index
    arrays are seeded by (seed, fold, stratum, repeat) ONLY — identical across
    models and across permuted column sets, so model-to-model differences are
    never shuffle noise.
    """
    col_idx = {f: i for i, f in enumerate(features)}
    units = [(f, 'feature', groups[f], [col_idx[f]]) for f in features]
    for gname in sorted(set(groups.values())):
        cols = [col_idx[f] for f in features if groups[f] == gname]
        units.append((gname, 'group', gname, cols))

    rows = []
    for sname, mask in strata.items():
        Xs, ys = X[mask], y[mask]
        n = len(ys)
        if n < 1000:
            logger.warning("stratum %s has only %d rows — skipped", sname, n)
            continue
        base_mu = predict_mu(Xs)
        base_rmse, base_r2 = _rmse(ys, base_mu), _r2(ys, base_mu)
        logger.info("stratum %-9s n=%d  base RMSE=%.4f  R²=%.4f",
                    sname, n, base_rmse, base_r2)
        perms = []
        for rep in range(n_repeats):
            # zlib.crc32: stable across processes (hash() is salted per process)
            rng = np.random.default_rng([seed, fold, zlib.crc32(sname.encode()), rep])
            perms.append(rng.permutation(n))
        t0 = time.monotonic()
        for u_i, (name, scope, group, cols) in enumerate(units, 1):
            d_rmse, d_r2 = [], []
            saved = Xs[:, cols].copy()
            for perm in perms:
                Xs[:, cols] = saved[perm]
                mu = predict_mu(Xs)
                d_rmse.append(_rmse(ys, mu) - base_rmse)
                d_r2.append(base_r2 - _r2(ys, mu))
            Xs[:, cols] = saved
            rows.append(dict(stratum=sname, scope=scope, name=name, group=group,
                             n_cols=len(cols), n_rows=n, base_rmse=base_rmse,
                             base_r2=base_r2,
                             delta_rmse=float(np.mean(d_rmse)),
                             delta_rmse_sd=float(np.std(d_rmse)),
                             delta_r2=float(np.mean(d_r2))))
            if u_i % 10 == 0 or u_i == len(units):
                logger.info("  [%s] %d/%d units (%.1f min)", sname, u_i,
                            len(units), (time.monotonic() - t0) / 60)
    return pd.DataFrame(rows)


# ─── Native importances (corroboration) ────────────────────────────────────────

def native_importance(model_key: str, model_dir: Path,
                      features: list) -> 'pd.DataFrame | None':
    """Ridge |coef| (comparable scales: QT/robust-scaled inputs) or XGB total_gain."""
    if model_key == 'linreg':
        with open(model_dir / 'coef.json', encoding='utf-8') as f:
            d = json.load(f)
        return pd.DataFrame({'name': d['features'],
                             'native': np.abs(d['coef']),
                             'kind': 'ridge_abs_coef'})
    if model_key == 'xgb':
        model = joblib.load(model_dir / 'model_mean_xgboost.joblib')
        gain = model.get_booster().get_score(importance_type='total_gain')
        vals = [gain.get(f'f{i}', 0.0) for i in range(len(features))]
        return pd.DataFrame({'name': features, 'native': vals,
                             'kind': 'xgb_total_gain'})
    return None   # DE: permutation only


# ─── Aggregation + comparison table ────────────────────────────────────────────

def aggregate(out_dir: Path, surface: str, models: list) -> None:
    """Mean ± sd across folds per model; write agg CSVs + the 3-model MD table."""
    agg = {}
    for mk in models:
        paths = sorted(out_dir.glob(f'importance_{mk}_{surface}_f*.csv'))
        if not paths:
            logger.warning("no fold CSVs for %s — skipped in table", mk)
            continue
        df = pd.concat([pd.read_csv(p).assign(fold=i)
                        for i, p in enumerate(paths)], ignore_index=True)
        g = (df.groupby(['stratum', 'scope', 'name', 'group'], as_index=False)
               .agg(delta_rmse=('delta_rmse', 'mean'),
                    fold_sd=('delta_rmse', 'std'),
                    delta_r2=('delta_r2', 'mean'),
                    base_rmse=('base_rmse', 'mean'),
                    n_folds=('fold', 'nunique')))
        # share of summed per-feature ΔRMSE within (stratum) — relative
        # individual contribution
        for st in g['stratum'].unique():
            m = (g['stratum'] == st) & (g['scope'] == 'feature')
            tot = g.loc[m, 'delta_rmse'].clip(lower=0).sum()
            g.loc[m, 'share'] = g.loc[m, 'delta_rmse'].clip(lower=0) / tot if tot > 0 else np.nan
            mg = (g['stratum'] == st) & (g['scope'] == 'group')
            totg = g.loc[mg, 'delta_rmse'].clip(lower=0).sum()
            g.loc[mg, 'share'] = g.loc[mg, 'delta_rmse'].clip(lower=0) / totg if totg > 0 else np.nan
        g.to_csv(out_dir / f'importance_{mk}_{surface}_agg.csv', index=False)
        agg[mk] = g

    if not agg:
        raise SystemExit("nothing to aggregate — run per-fold importance first")

    lines = [f"# Permutation feature importance — {surface} "
             f"(date_kfold held folds, {N_FOLDS} folds)",
             "",
             "ΔRMSE (ppm) = increase in held-fold RMSE when the feature/group is "
             "permuted within the evaluated stratum; mean over folds (± sd across "
             "folds). share = fraction of the model's summed positive per-"
             "feature (or per-group) ΔRMSE. Groups are permuted jointly — the "
             "honest number under collinearity; per-feature rows give relative "
             "individual contributions.",
             "",
             "> Caveat (per the QF ablation, log/SPEC_EMPHASIS_STATUS_2026-07-08.md): "
             "held-out CV importance over-credits blocks that are TCCON-neutral "
             "(contam, spec). Cross-reference the 6-set TCCON feature-set ablation "
             "before quoting any block as load-bearing.",
             ""]

    def _fmt(v, sd=None):
        if pd.isna(v):
            return "—"
        s = f"{v:+.4f}"
        if sd is not None and not pd.isna(sd):
            s += f" ± {sd:.4f}"
        return s

    for st in ('global', 'nearcloud'):
        for scope, title, top in (('group', 'Group importance', None),
                                  ('feature', 'Per-feature importance (top 20 by DE)', 20)):
            sub = {mk: g[(g['stratum'] == st) & (g['scope'] == scope)]
                       .set_index('name') for mk, g in agg.items()}
            if all(s.empty for s in sub.values()):
                continue
            ref = 'de' if 'de' in sub and not sub['de'].empty else list(sub)[0]
            order = sub[ref].sort_values('delta_rmse', ascending=False)
            names = list(order.index[:top]) if top else list(order.index)
            first = 'feature' if scope == 'feature' else 'group'
            hdr = f"| {first} | block | " + " | ".join(
                f"{mk} ΔRMSE | {mk} share" for mk in sub) + " |"
            sep = "|" + "---|" * (2 + 2 * len(sub))
            lines += [f"## {title} — stratum: {st}", "", hdr, sep]
            for name in names:
                grp = order.loc[name, 'group'] if name in order.index else ""
                cells = []
                for mk in sub:
                    if name in sub[mk].index:
                        r = sub[mk].loc[name]
                        cells += [_fmt(r['delta_rmse'], r['fold_sd']),
                                  f"{r['share']*100:.1f}%" if pd.notna(r['share']) else "—"]
                    else:
                        cells += ["—", "—"]
                lines.append(f"| {name} | {grp} | " + " | ".join(cells) + " |")
            base = ", ".join(f"{mk} {sub[mk]['base_rmse'].iloc[0]:.4f}"
                             for mk in sub if not sub[mk].empty)
            lines += ["", f"Base held RMSE (ppm, fold mean): {base}", ""]

    out_md = out_dir / f'FEATURE_IMPORTANCE_TABLE_{surface}.md'
    out_md.write_text("\n".join(lines), encoding='utf-8')
    print(f"[aggregate] table → {out_md}")


# ─── Per-fold driver ───────────────────────────────────────────────────────────

def run_fold(surface: str, fold: int, models: list, data_path, out_dir: Path,
             n_repeats: int, n_rows: int, rmse_tol: float,
             exclude_snow: bool) -> None:
    cfg = SURF[surface]
    storage = get_storage_dir()

    loaded = {}
    for mk in models:
        mdir = storage / 'results' / MODEL_DIRS[mk].format(
            surf=surface, r=cfg['r'], fold=fold)
        if not mdir.is_dir():
            raise SystemExit(f"model dir missing: {mdir}")
        pipeline, predict_mu = _LOADERS[mk](mdir)
        loaded[mk] = dict(dir=mdir, pipeline=pipeline, predict=predict_mu,
                          stored_rmse=_stored_held_rmse(mdir))

    # Gate 1 — identical feature lists across models within the fold.
    feats0 = loaded[models[0]]['pipeline'].features
    for mk in models[1:]:
        if loaded[mk]['pipeline'].features != feats0:
            raise SystemExit(
                f"feature-list mismatch {models[0]} vs {mk} (fold {fold}) — "
                "the cross-model comparison would be invalid.")
    groups = assign_groups(loaded[models[0]]['pipeline'])
    logger.info("fold %d: %d features, groups: %s", fold, len(feats0),
                {g: sum(v == g for v in groups.values()) for g in sorted(set(groups.values()))})

    # Held dates from the first model's manifest (asserted identical).
    manifests = {}
    for mk in models:
        with open(loaded[mk]['dir'] / 'training_dates.json', encoding='utf-8') as f:
            manifests[mk] = json.load(f)['held_dates']
    for mk in models[1:]:
        if manifests[mk] != manifests[models[0]]:
            raise SystemExit(f"held_dates mismatch {models[0]} vs {mk} (fold {fold})")

    df = load_held_frame(data_path, manifests[models[0]], cfg['sfc_type'],
                         cfg['target'], exclude_snow=exclude_snow)
    y_all = df[cfg['target']].to_numpy(dtype=np.float32)

    for mk in models:
        pipeline, predict_mu = loaded[mk]['pipeline'], loaded[mk]['predict']
        X = pipeline.transform(df)
        valid = np.isfinite(y_all) & np.all(np.isfinite(X), axis=1)
        Xv, yv = X[valid], y_all[valid]
        cld = df.loc[valid, 'cld_dist_km'].to_numpy(dtype=np.float32)
        del X

        # Gate 2 — reproduce the stored held RMSE before permuting anything.
        rmse_full = _rmse(yv, predict_mu(Xv))
        stored = loaded[mk]['stored_rmse']
        if stored is not None:
            rel = abs(rmse_full - stored) / stored
            msg = (f"[{mk} f{fold}] reconstructed held RMSE {rmse_full:.4f} vs "
                   f"stored {stored:.4f} (rel diff {rel:.2e})")
            if rel > rmse_tol:
                raise SystemExit(f"GATE FAILED: {msg} > tol {rmse_tol:g} — the "
                                 "held-set reconstruction does not match training; "
                                 "importance numbers would be invalid.")
            print(f"GATE OK: {msg}")
        else:
            logger.warning("[%s f%d] no stored RMSE — gate skipped", mk, fold)

        # Optional stratified subsample (local smoke runs).
        if n_rows and len(yv) > n_rows:
            rng = np.random.default_rng([SEED, fold])
            near = (cld >= 0) & (cld <= NEAR_CLOUD_KM)
            k_near = max(int(n_rows * near.mean()), min(near.sum(), 50_000))
            idx = np.concatenate([
                rng.choice(np.where(near)[0], size=min(k_near, near.sum()), replace=False),
                rng.choice(np.where(~near)[0], size=min(n_rows - k_near, (~near).sum()),
                           replace=False)])
            idx.sort()
            Xv, yv, cld = Xv[idx], yv[idx], cld[idx]
            logger.info("[%s f%d] subsampled to %d rows (near-cloud kept: %d)",
                        mk, fold, len(yv), int(((cld >= 0) & (cld <= NEAR_CLOUD_KM)).sum()))

        strata = {'global': np.ones(len(yv), dtype=bool),
                  'nearcloud': (cld >= 0) & (cld <= NEAR_CLOUD_KM)}
        res = permutation_importance(predict_mu, Xv, yv, feats0, groups,
                                     strata, n_repeats, fold)
        res.insert(0, 'model', mk)
        res.insert(1, 'surface', surface)
        res.insert(2, 'fold', fold)
        out_csv = out_dir / f'importance_{mk}_{surface}_f{fold}.csv'
        res.to_csv(out_csv, index=False)
        print(f"[{mk} f{fold}] importance → {out_csv}")

        nat = native_importance(mk, loaded[mk]['dir'], feats0)
        if nat is not None:
            nat.to_csv(out_dir / f'native_{mk}_{surface}_f{fold}.csv', index=False)
        del Xv


def main():
    parser = argparse.ArgumentParser(
        description="Permutation feature importance for DE / XGB-mean / Ridge "
                    "on date_kfold held folds")
    parser.add_argument('--surface', required=True, choices=['ocean', 'land'])
    parser.add_argument('--models', type=str, default='de,xgb,linreg')
    parser.add_argument('--fold', type=int, default=None,
                        help='Single fold (0-based). Default: all 5.')
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--n-rows', type=int, default=0,
                        help='Stratified subsample size for local smoke runs; '
                             '0 = full held fold (CURC).')
    parser.add_argument('--data', type=str, default=None,
                        help='Combined parquet override (default: '
                             'results/csv_collection/combined_2016_2020_dates.parquet).')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir (default: results/model_comparison/'
                             'feature_importance/<surface>/).')
    parser.add_argument('--rmse-tol', type=float, default=2e-3,
                        help='Relative tolerance for the stored-vs-reconstructed '
                             'held-RMSE gate.')
    parser.add_argument('--exclude-snow', action='store_true',
                        help='Match a trainer run that used --exclude_snow '
                             '(production kept snow — leave unset).')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate existing per-fold CSVs into the '
                             'comparison table instead of computing.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        if m not in MODEL_DIRS:
            raise SystemExit(f"unknown model {m!r} (choose from {sorted(MODEL_DIRS)})")

    storage = get_storage_dir()
    out_dir = Path(args.out) if args.out else (
        storage / 'results/model_comparison/feature_importance' / args.surface)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate(out_dir, args.surface, models)
        return

    data_path = Path(args.data) if args.data else (
        storage / 'results/csv_collection/combined_2016_2020_dates.parquet')
    if not data_path.is_file():
        raise SystemExit(f"data parquet not found: {data_path}")

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    for fold in folds:
        run_fold(args.surface, fold, models, data_path, out_dir,
                 args.n_repeats, args.n_rows, args.rmse_tol, args.exclude_snow)


if __name__ == '__main__':
    main()
