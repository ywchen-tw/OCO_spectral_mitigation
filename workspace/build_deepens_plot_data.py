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

Uncertainty columns (Phase 0 of the uncertainty-aware TCCON comparison; see
src/analysis/UNCERTAINTY_AWARE_TCCON_COMPARISON.md) are emitted per footprint:

    de_sigma            total predictive std (mixture: aleatoric + epistemic)
    de_epistemic_sigma  ensemble disagreement  = std_m(mu_m)      (does NOT avg down)
    de_aleatoric_sigma  mean predicted noise   = sqrt(mean_m(var_m))
    xco2_uncertainty    OCO-2 L2 retrieval posterior (passthrough from the parquet)

By construction de_sigma^2 == de_epistemic_sigma^2 + de_aleatoric_sigma^2
(checked at write time).  These are the three candidate Side-A components the
calibration study (§4.2) picks between; the point correction (ensemble mean mu)
alone still suffices for the poster figure.

Example (both surfaces):
    PYTHONPATH=src python workspace/build_deepens_plot_data.py \
        --ocean-model-dir results/model_deep_ensemble/de_ocean_beta_nll_f0 \
        --land-model-dir  results/model_deep_ensemble/de_land_beta_nll_f0 \
        --input results/csv_collection/combined_2018-09-02_all_orbits.parquet \
        --output .../deep_ensemble/plot_data.parquet
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import xgboost as xgb

from models.pipeline import FeaturePipeline, _ensure_derived_features
from models.deep_ensemble import GaussianMLP, _member_predict
from models.leakage_guard import check_training_overlap
from apply.apply_deep_ensemble import _check_required_columns, _domain_report

import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoother_null import add_smoother_columns, DEFAULT_WINDOWS_S  # noqa: E402

CORR_COL = 'deep_ensemble_corrected_xco2'              # (1) full mu          xco2_bc - mu
CORR_COL_PNEAR = 'deepens_pnear_corrected_xco2'        # (2) P(near)*mu       xco2_bc - P*mu
CORR_COL_GATE = 'deepens_gate_corrected_xco2'          # (3) mu*1[P>0.5]      xco2_bc - mu*(P>0.5)
KEEP_COLS = ('time', 'lon', 'lat', 'cld_dist_km', 'sfc_type', 'fp',
             'xco2_bc', 'xco2_bc_anomaly', 'xco2_apriori',
             'xco2_raw', 'xco2_raw_anomaly', 'xco2_uncertainty',
             # scene diagnostics for uncertainty-relationship stratification (§4.1);
             # all optional (kept only if present in the input parquet).
             'aod_total', 'sza', 'snr_o2a', 'snr_wco2', 'snr_sco2')


def _load_cloud_fold(model_dir):
    """Load one xgb_cloud fold: (pipeline, classifier, cloud_cols)."""
    md = Path(model_dir)
    pipe = FeaturePipeline.load(md / 'xgb_cloud_pipeline.pkl')
    meta = pickle.load(open(md / 'xgb_cloud_meta.pkl', 'rb'))
    clf = xgb.XGBClassifier()
    clf.load_model(str(md / 'xgb_cloud_model.json'))
    return pipe, clf, tuple(meta.get('cloud_cols', ()))


def _predict_cloud_pooled(kept_df, cloud_fold_dirs):
    """Pooled near-cloud probability P(near) over xgb_cloud folds for the rows in
    `kept_df` (already surface-filtered + DE-valid).  Each fold uses its own fitted
    pipeline; P(near) = mean of the per-fold predicted probabilities."""
    folds = [_load_cloud_fold(d) for d in cloud_fold_dirs]
    probs = []
    for pipe, clf, cloud_cols in folds:
        X = pipe.transform(kept_df)
        if cloud_cols:
            X = np.concatenate(
                [X, kept_df[list(cloud_cols)].to_numpy(dtype=np.float32)], axis=1)
        probs.append(clf.predict_proba(X.astype(np.float32))[:, 1])
    p = np.mean(np.stack(probs), axis=0)
    print(f"  [cloud] pooled {len(folds)} fold(s) → P(near): "
          f"mean={p.mean():.3f} frac>0.5={np.mean(p > 0.5):.3f}")
    return p.astype(np.float32)


def _load_fold(model_dir):
    """Load one fold's (pipeline, members, meta)."""
    md = Path(model_dir)
    meta = pickle.load(open(md / 'deep_ensemble_meta.pkl', 'rb'))
    pipe = FeaturePipeline.load(md / 'deep_ensemble_pipeline.pkl')
    aux_cloud = bool(meta.get('aux_cloud', False))
    hidden_dims = tuple(meta.get('hidden_dims', (64, 32)))
    dropout = float(meta.get('dropout', 0.0))
    norm = meta.get('norm', 'none')
    members = []
    for p in sorted(md.glob('member_*.pt')):
        m = GaussianMLP(pipe.n_features, hidden_dims=hidden_dims,
                        aux_cloud=aux_cloud, dropout=dropout, norm=norm)
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        m.eval(); members.append(m)
    if not members:
        raise SystemExit(f"no member_*.pt in {md}")
    return pipe, members, meta


def _predict_pooled(df, fold_dirs, sfc_type, *, ood_thresh=8.0, max_ood_frac=0.02,
                    strict=False, tag='input'):
    """Cross-fold ensemble: pool every member of every fold.

    Each fold has its OWN fitted scaler, so X is transformed per fold before
    prediction.  Returns (mu, sigma, epi_sigma, alea_sigma, mu_members, kept_df, meta0):
      mu          mean over all pooled members,
      sigma       total predictive std (mixture: aleatoric + epistemic),
      epi_sigma   epistemic std = std_m(mu_m)   (ensemble disagreement),
      alea_sigma  aleatoric std = sqrt(mean_m(var_m))   (mean predicted noise),
      mu_members  per-member mean stack [M, N] (M pooled members) — the raw
                  material for the EXACT case-level epistemic Var_m(x̄_m); only
                  attached to the output when --emit-members is set,
    all computed over the full pooled member set.  By the mixture identity
    sigma^2 == epi_sigma^2 + alea_sigma^2 (up to the 1e-12 var floor).
    """
    folds = [_load_fold(d) for d in fold_dirs]
    pipe0, _, meta0 = folds[0]
    loss = meta0.get('loss', 'gaussian_nll'); nu = meta0.get('nu', 4.0)
    n_mem = sum(len(m) for _, m, _ in folds)
    print(f"  [loaded] {len(folds)} fold(s) × members = {n_mem} total, "
          f"{pipe0.n_features} features, loss={loss}")

    _check_required_columns(df, pipe0)
    df = df[df['sfc_type'] == sfc_type].copy()
    df = _ensure_derived_features(df)
    X0 = pipe0.transform(df)
    valid = np.all(np.isfinite(X0), axis=1)
    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        return None, None, None, None, None, df, meta0

    overall, worst = _domain_report(X0[valid], pipe0, thresh=ood_thresh)
    if overall > max_ood_frac:
        flagged = [f"{n}({f:.0%})" for n, f in worst[:5] if f > max_ood_frac]
        msg = (f"  ⚠ DOMAIN WARNING [{tag}]: {overall:.1%} of feature values OOD "
               f"(|z|>{ood_thresh:g}); worst: {flagged} → predictions UNRELIABLE")
        if strict:
            raise SystemExit(msg + "\n  (refused: --strict)")
        print(msg)
    else:
        print(f"  domain check [{tag}]: OK ({overall:.2%} OOD)")

    dev = torch.device('cpu')
    mu_stack, var_stack = [], []
    for pipe, members, _ in folds:
        Xi = pipe.transform(df)          # this fold's own scaler
        for m in members:
            mu_i, var_i = _member_predict(m, Xi, dev, loss=loss, nu=nu)
            mu_stack.append(mu_i); var_stack.append(var_i)
    mu_stack = np.stack(mu_stack); var_stack = np.stack(var_stack)
    mu = mu_stack.mean(0)
    epi_var = mu_stack.var(0)                               # Var_m(mu_m): epistemic
    alea_var = var_stack.mean(0)                            # mean_m(var_m): aleatoric
    var = epi_var + alea_var                                # mixture total variance
    sigma = np.sqrt(np.maximum(var, 1e-12))
    epi_sigma = np.sqrt(np.maximum(epi_var, 1e-12))
    alea_sigma = np.sqrt(np.maximum(alea_var, 1e-12))
    return (mu.astype(np.float32), sigma.astype(np.float32),
            epi_sigma.astype(np.float32), alea_sigma.astype(np.float32),
            mu_stack.astype(np.float32), df, meta0)


def _build_surface(df, fold_dirs, sfc_type, *, dk, clim_max_ppm=50.0,
                   max_abs_anomaly=25.0, cloud_fold_dirs=None,
                   base_col='xco2_bc', truth_col='xco2_bc_anomaly',
                   emit_members=False):
    """Predict one surface (pooling all folds); return a plot_data frame or None.

    ``base_col`` is the XCO2 column the predicted anomaly is subtracted from
    (xco2_bc for models trained on xco2_bc_anomaly*, xco2_raw for models trained
    on xco2_raw_anomaly*); ``truth_col`` is the matching anomaly column used only
    for the printed RMS diagnostic.  The corrected column is always named
    deep_ensemble_corrected_xco2 (= base_col − predicted anomaly).

    Two guards skip the correction (set mu=0 → corrected XCO2 == raw base_col) and
    flag the row, so non-physical points don't pollute the histogram:
      • clim_guard    — INPUT guard: base_col > xco2_apriori + `clim_max_ppm`
                        (fill values / failed retrievals).
      • anomaly_guard — OUTPUT guard: |predicted anomaly| > `max_abs_anomaly`
                        (model blow-ups; a bias correction should be a few ppm,
                        not hundreds).  No effect when `max_abs_anomaly` is None.
    """
    mu, sigma, epi_sigma, alea_sigma, mu_members, kept, meta = _predict_pooled(
        df, fold_dirs, sfc_type, tag=f'sfc{sfc_type}', **dk)
    if mu is None or len(kept) == 0:
        print(f"  sfc={sfc_type}: no rows after filter — skipped")
        return None
    out = kept[[c for c in KEEP_COLS if c in kept.columns]].reset_index(drop=True).copy()
    mu = np.asarray(mu, dtype=float)

    # INPUT guard (climatology)
    clim_guard = np.zeros(len(out), dtype=bool)
    if 'xco2_apriori' in out.columns:
        diff = out[base_col].to_numpy(float) - out['xco2_apriori'].to_numpy(float)
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
    # Side-A uncertainty components (emitted per footprint, NOT zeroed by the
    # guards — downstream filters on clim_guard/anomaly_guard). `sigma` kept for
    # backward compatibility; de_sigma is its explicit alias.
    out['sigma'] = sigma
    out['de_sigma'] = sigma
    out['de_epistemic_sigma'] = epi_sigma
    out['de_aleatoric_sigma'] = alea_sigma
    # Per-member mean columns mu_00… — only when --emit-members (M extra columns
    # per footprint).  They give the EXACT case-level epistemic Var_m(x̄_m) in the
    # uncertainty comparison; without them the fully-correlated fallback (a mild
    # over-estimate) is used.  NOT zeroed by the guards (raw member diagnostics).
    if emit_members and mu_members is not None:
        M = mu_members.shape[0]
        w = max(2, len(str(M - 1)))
        for j in range(M):
            out[f'mu_{j:0{w}d}'] = np.asarray(mu_members[j], dtype=np.float32)
    ident = float(np.nanmax(np.abs(
        np.asarray(sigma, float) ** 2
        - np.asarray(epi_sigma, float) ** 2 - np.asarray(alea_sigma, float) ** 2)))
    print(f"  sfc={sfc_type}: uncertainty σ (ppm) — total {np.nanmean(sigma):.3f}, "
          f"epistemic {np.nanmean(epi_sigma):.3f}, aleatoric {np.nanmean(alea_sigma):.3f}"
          f"  [mixture identity max|Δvar|={ident:.2e}]")
    xb = out[base_col].to_numpy(dtype=float)
    out[CORR_COL] = xb - mu                                   # (1) full mu

    # ── cloud-distance correction policies (need the xgb cloud classifier) ─────
    if cloud_fold_dirs:
        p_near = np.asarray(_predict_cloud_pooled(kept, cloud_fold_dirs), dtype=float)
        # mu already has guards zeroed → all three policies collapse to xco2_bc there
        out['p_near'] = p_near
        out[CORR_COL_PNEAR] = xb - p_near * mu                # (2) P(near)*mu
        out[CORR_COL_GATE] = xb - np.where(p_near > 0.5, mu, 0.0)  # (3) mu*1[P>0.5]

    if truth_col in out.columns:
        y = out[truth_col].to_numpy(float)
        keep = ~guard                      # report only where correction was applied
        pre = np.sqrt(np.nanmean(y[keep] ** 2))
        post = np.sqrt(np.nanmean((y[keep] - mu[keep]) ** 2))
        print(f"  sfc={sfc_type}: {int(keep.sum()):,}/{len(out):,} corrected  {truth_col} RMS "
              f"{pre:.3f} → {post:.3f} ppm ({100*(1-post/pre):+.1f}%)")
    else:
        print(f"  sfc={sfc_type}: {len(out):,} soundings (no truth column)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ocean-model-dir', nargs='+', default=None,
                    help="Deep-ensemble fold dir(s) trained on ocean (sfc_type==0). "
                         "Pass all folds (f0..f4) to pool the full cross-fold ensemble.")
    ap.add_argument('--land-model-dir', nargs='+', default=None,
                    help="Deep-ensemble fold dir(s) trained on land (sfc_type==1). "
                         "Pass all folds to pool the full cross-fold ensemble.")
    ap.add_argument('--ocean-cloud-model-dir', nargs='+', default=None,
                    help="xgb_cloud fold dir(s) for ocean (sfc_type==0).  When given, "
                         "emits P(near) and the two extra correction columns "
                         f"{CORR_COL_PNEAR!r} (P*mu) and {CORR_COL_GATE!r} (gate).")
    ap.add_argument('--land-cloud-model-dir', nargs='+', default=None,
                    help="xgb_cloud fold dir(s) for land (sfc_type==1).  Same effect "
                         "as --ocean-cloud-model-dir, for land footprints.")
    # backward-compatible single-surface form
    ap.add_argument('--model-dir', nargs='+', default=None, help="(single-surface) model dir(s).")
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
    ap.add_argument('--max-abs-anomaly', type=float, default=25.0,
                    help="Skip correction where |predicted anomaly| exceeds this many ppm "
                         "(model blow-ups); default 25. Set <=0 to disable.")
    ap.add_argument('--emit-members', action='store_true',
                    help="Also write per-member mean columns mu_00… (M extra columns "
                         "per footprint) for the EXACT case-level epistemic term in the "
                         "uncertainty-aware TCCON comparison. Off by default (keeps file "
                         "size unchanged); enable for the uncertainty runs.")
    ap.add_argument('--allow-train-overlap', action='store_true',
                    help="Downgrade the training-date leakage guard from refusal "
                         "to a loud warning (deliberate in-sample diagnostics "
                         "only — never for validation numbers).")
    ap.add_argument('--smoother-windows-s', default=','.join(
                        f'{w:g}' for w in DEFAULT_WINDOWS_S),
                    help="Pure-smoother null (M4): also emit feature-free "
                         "smoother_w{W}_corrected_xco2 columns, W = running-mean "
                         "half-width in seconds over the orbit segment (see "
                         "smoother_null.py). Comma-separated; '' disables.")
    ap.add_argument('--correction-base', choices=('bc', 'raw'), default='bc',
                    help="XCO2 column the predicted anomaly is subtracted from: 'bc' "
                         "(xco2_bc, for models trained on xco2_bc_anomaly*) or 'raw' "
                         "(xco2_raw, for models trained on xco2_raw_anomaly*). The guard "
                         "and RMS diagnostic follow the base. Default bc.")
    args = ap.parse_args()
    dk = dict(ood_thresh=args.ood_thresh, max_ood_frac=args.max_ood_frac, strict=args.strict)
    base_col = 'xco2_raw' if args.correction_base == 'raw' else 'xco2_bc'
    truth_col = 'xco2_raw_anomaly' if args.correction_base == 'raw' else 'xco2_bc_anomaly'

    surfaces = []  # (model_dir, sfc_type, cloud_model_dir)
    if args.ocean_model_dir:
        surfaces.append((args.ocean_model_dir, 0, args.ocean_cloud_model_dir))
    if args.land_model_dir:
        surfaces.append((args.land_model_dir, 1, args.land_cloud_model_dir))
    if args.model_dir is not None:
        if args.sfc_type is None:
            raise SystemExit("--model-dir requires --sfc_type")
        cmd = args.ocean_cloud_model_dir if args.sfc_type == 0 else args.land_cloud_model_dir
        surfaces.append((args.model_dir, args.sfc_type, cmd))
    if not surfaces:
        raise SystemExit("pass --ocean-model-dir and/or --land-model-dir "
                         "(or --model-dir with --sfc_type)")

    df = pd.concat([pd.read_parquet(p) for p in args.input], ignore_index=True)
    print(f"  read {len(df):,} rows from {len(args.input)} file(s)")

    # Training-date leakage guard: refuse when an evaluation date appears in
    # any model dir's training/calibration manifest (training_dates.json).
    for md, sfc, cmd in surfaces:
        check_training_overlap(
            list(md) + list(cmd or ()), input_paths=args.input,
            times=df['time'].to_numpy() if 'time' in df.columns else None,
            allow=args.allow_train_overlap, tag=f'sfc{sfc}')

    max_abs_anom = args.max_abs_anomaly if args.max_abs_anomaly > 0 else None
    parts = [p for p in (_build_surface(df, md, sfc, dk=dk,
                                        clim_max_ppm=args.climatology_max_ppm,
                                        max_abs_anomaly=max_abs_anom,
                                        cloud_fold_dirs=cmd,
                                        base_col=base_col, truth_col=truth_col,
                                        emit_members=args.emit_members)
                         for md, sfc, cmd in surfaces)
             if p is not None]
    if not parts:
        raise SystemExit("no soundings predicted on any surface")
    out = pd.concat(parts, ignore_index=True)

    # Pure-smoother null columns (M4): feature-free running-mean "corrections"
    # the TCCON report can be pointed at via --corr-col to show that merely
    # smoothing XCO2 collapses footprint scatter but does not move the bias.
    smoother_ws = [float(x) for x in args.smoother_windows_s.split(',') if x.strip()]
    if smoother_ws:
        out = add_smoother_columns(out, windows_s=smoother_ws, base_col=base_col,
                                   clim_max_ppm=args.climatology_max_ppm)
        print(f"  smoother-null columns added (±{smoother_ws} s, base {base_col})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"  [saved] {len(out):,} rows ({len(parts)} surface(s)) → {args.output}")
    extra = [c for c in (CORR_COL_PNEAR, CORR_COL_GATE) if c in out.columns]
    print(f"  correction columns: {CORR_COL!r}"
          + (f" + {extra}" if extra else "") + f"  (use --poster-model {CORR_COL})")


if __name__ == '__main__':
    main()
