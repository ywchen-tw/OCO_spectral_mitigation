"""apply_anchor.py — step 4: add the anchored product column to the ML(raw) tree.

For every case under results/model_comparison/deep_ensemble/de_prof_reg_mix_raw<RUN_SUFFIX>/
(combined_<date>_<site>/plot_data.parquet) merge the B11-style predictors from
the per-date source parquet (time/lon/lat key, as the report does), evaluate
the fitted LR on LAND rows and write

    anchor_lr_increment      = LR(f)                (NaN on ocean / missing inputs)
    anchor_lr_guard          = |LR(f)| > --max-abs-increment (fill-value predictors)
    anchor_lr_corrected_xco2 = deep_ensemble_corrected_xco2 − anchor_lr_increment

Applied to QF 0 and 1 and to guarded rows alike (B11 is applied to every
sounding).  Atomic rewrite; idempotent unless --force (smoother_null pattern).

Diagnostic: per case and pooled over land footprints, the correlation and
slope of the LR increment against the operational increment
xco2_raw − xco2_bc — the "does the LR mimic B11" check — written to
results/anchor_lr/apply_diagnostics.csv.

Then run the unchanged report on the new column:
    python workspace/tccon_comparison_report.py --script curc_shell_blanca_plot_corr_xco2_deepens.sh \
        --out-base <raw tree> --output-dir <raw tree>/anchor_lr \
        --corr-col anchor_lr_corrected_xco2 --fname-suffix _anchor_lr_r100km \
        --exclude-sites ny --radius-km 100 --window-min 60 --cld-edges 0,10,inf --ak-harmonize

Run:  python workspace/anchor_lr/apply_anchor.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                       # noqa: E402

def new_cols(variant: str) -> tuple[str, str, str]:
    return ac.inc_col(variant), ac.guard_col(variant), ac.corr_col(variant)


def _ols(y, x):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.var() == 0:
        return np.nan, np.nan
    return float(np.cov(x, y, bias=True)[0, 1] / x.var()), float(np.corrcoef(x, y)[0, 1])


def augment_case(path: Path, model: dict, *, max_abs: float, force: bool):
    INC, GUARD, CORR = new_cols(model.get('variant', ''))
    df = pd.read_parquet(path)
    if not force and all(c in df.columns for c in (INC, GUARD, CORR)):
        return 'skip (columns present)', None
    date = ac.date_from_name(path.parent.name)
    sp = ac.source_parquet_for_date(date) if date else None
    if sp is None:
        return f'skip (no source parquet for {date})', None
    need = [c for c in ac.SOURCE_COLS if c not in df.columns]
    src = pd.read_parquet(sp, columns=ac.MERGE_KEYS + need).drop_duplicates(ac.MERGE_KEYS)
    m = df.merge(src, on=ac.MERGE_KEYS, how='left')
    assert len(m) == len(df)
    land = m['sfc_type'].to_numpy(float) == 1
    inc = ac.predict_increment(m, model)
    inc[~land] = np.nan
    guard = np.isfinite(inc) & (np.abs(inc) > max_abs)
    inc[guard] = np.nan
    df = df.drop(columns=[c for c in (INC, GUARD, CORR) if c in df.columns])
    df[INC] = inc.astype(np.float32)
    df[GUARD] = guard
    df[CORR] = (df[ac.ML_RAW_CORR_COL].to_numpy(float) - inc).astype(np.float32)
    tmp = path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    opinc = df['xco2_raw'].to_numpy(float) - df['xco2_bc'].to_numpy(float)
    ok = land & np.isfinite(inc) & np.isfinite(opinc)
    slope, r = _ols(inc[ok], opinc[ok])
    diag = dict(case=path.parent.name, n_land=int(land.sum()), n_applied=int(ok.sum()),
                n_guard=int(guard.sum()), inc_mean=float(np.nanmean(inc[ok])) if ok.any() else np.nan,
                opinc_mean=float(opinc[ok].mean()) if ok.any() else np.nan,
                r_inc_opinc=r, slope_inc_on_opinc=slope)
    pairs = pd.DataFrame(dict(inc=inc[ok], opinc=opinc[ok]))
    return (f'wrote n_land={diag["n_land"]} applied={diag["n_applied"]} '
            f'guard={diag["n_guard"]} r(LR,B11 inc)={r:+.3f}'), (diag, pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tree', type=Path, default=ac.RAW_TREE)
    ap.add_argument('--variant', default='',
                    help="'' (b11 default) or 'nocontam': picks the model JSON and the "
                         "output column names anchor_lr[_<variant>]_*")
    ap.add_argument('--model', type=Path, default=None,
                    help='model JSON (default from --variant)')
    ap.add_argument('--max-abs-increment', type=float, default=25.0,
                    help='|LR(f)| above this (ppm) → NaN + anchor_lr_guard (mirrors the '
                         'production |mu| guard; catches fill-value predictors)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    model = ac.load_model(args.model or ac.coef_json(args.variant))
    print(f"variant {model.get('variant', '') or 'b11'}: terms {model['terms']['continuous']} "
          f"→ columns {new_cols(model.get('variant', ''))}")
    diags, pairs = [], []
    for pq in sorted(args.tree.glob('combined_*/plot_data.parquet')):
        msg, out = augment_case(pq, model, max_abs=args.max_abs_increment, force=args.force)
        print(f'  {pq.parent.name}: {msg}')
        if out is not None:
            diags.append(out[0]); pairs.append(out[1])
    if not diags:
        print('nothing written (all cases already augmented? use --force)')
        return 0
    d = pd.DataFrame(diags)
    pp = pd.concat(pairs, ignore_index=True)
    slope, r = _ols(pp.inc, pp.opinc)
    pooled = dict(case='POOLED', n_land=int(d.n_land.sum()), n_applied=int(len(pp)),
                  n_guard=int(d.n_guard.sum()), inc_mean=float(pp.inc.mean()),
                  opinc_mean=float(pp.opinc.mean()), r_inc_opinc=r, slope_inc_on_opinc=slope)
    d = pd.concat([d, pd.DataFrame([pooled])], ignore_index=True)
    ac.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    d.to_csv(ac.OUT_ROOT / f"apply_diagnostics{ac.variant_tag(model.get('variant', ''))}.csv",
             index=False)
    print(f'\npooled land ({len(pp):,} fp): LR increment mean {pp.inc.mean():+.3f} '
          f'(σ {pp.inc.std():.3f}) vs operational xco2_raw−xco2_bc mean {pp.opinc.mean():+.3f} '
          f'(σ {pp.opinc.std():.3f}); r = {r:+.3f}, slope(LR on B11) = {slope:+.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
