"""Label-noise ceiling for the clear-sky-anomaly target (TODO_ACCOMPLISH §2-4).

The target is y = xco2_bc − mean(clear-sky reference), so even a perfect model
cannot beat the label's own noise:

  σ²_ref = ref_std² / n_ref     reference-mean sampling error (recomputed here
                                per sounding with the production reference
                                definition, grouped by orbit_id exactly as at
                                build time — recomputed anomaly is checked
                                against the stored column),
  σ²_ret = xco2_uncertainty²    single-sounding retrieval posterior σ (the
                                random part of the sounding's own XCO2).

Implied maximum R² on the same rows:  R²_max = 1 − E[σ²_noise] / Var(y),
reported for σ²_ref alone (strict lower bound on noise → weak upper bound on
the ceiling), σ²_ref + σ²_ret (headline), and an EMPIRICAL alternative that
needs no error model: far-field anomalies (cld_dist ≥ far_km) carry no cloud
signal, so Var(y | far) estimates the noise floor including everything the
posterior σ misses → R²_max,emp = 1 − Var(y|far)/Var(y).  The three bracket
the ceiling; quote the headline with the empirical one as robustness.

Population matches training: per surface, filter_target_outliers (|y| ≤ 100),
NaN targets dropped; near-cloud (< near_km) subset also reported since the
correction is decided there.

Run (per-surface targets, as in production: ocean r05 / land r15):
  PYTHONPATH=src python -m analysis.label_noise_ceiling \
      --data results/csv_collection/combined_2018-*-*_all_orbits.parquet \
      --out results/label_noise_ceiling.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral.anomaly import compute_xco2_anomaly           # noqa: E402
from constants import anomaly_args                          # noqa: E402

# (target column, min_cld_dist km) — production per-surface targets + default.
TARGETS = {
    'r10': ('xco2_bc_anomaly', 10.0),
    'r05': ('xco2_bc_anomaly_r05', 5.0),
    'r15': ('xco2_bc_anomaly_r15', 15.0),
}
LOAD_COLS = ['date', 'orbit_id', 'lat', 'cld_dist_km', 'sfc_type', 'snow_flag',
             'xco2_bc', 'xco2_uncertainty'] + [c for c, _ in TARGETS.values()]
MAX_ABS_PPM = 100.0            # models.pipeline.MAX_ABS_ANOMALY_PPM


def per_file_rows(path: Path, targets) -> list[pd.DataFrame]:
    df = pd.read_parquet(path, columns=[c for c in LOAD_COLS])
    parts = []
    for _, g in df.groupby('orbit_id', sort=False):
        lat = g['lat'].to_numpy(float)
        cld = g['cld_dist_km'].to_numpy(float)
        x = g['xco2_bc'].to_numpy(float)
        out = g[['date', 'sfc_type', 'snow_flag', 'cld_dist_km',
                 'xco2_uncertainty']].copy()
        for tkey in targets:
            tcol, min_cld = TARGETS[tkey]
            anom, _rm, rstd, nref = compute_xco2_anomaly(
                lat, cld, x, return_ref_stats=True,
                **anomaly_args(min_cld_dist=min_cld))
            out[f'y_{tkey}'] = g[tcol].to_numpy(float)
            out[f'yrc_{tkey}'] = anom              # recomputed (cross-check)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'sref2_{tkey}'] = np.where(nref > 0, rstd ** 2 / np.maximum(nref, 1),
                                                np.nan)
        parts.append(out)
    return parts


def summarize(rows: pd.DataFrame, tkey: str, near_km: float, far_km: float):
    """One (surface × target) ceiling block over the pooled rows."""
    y = rows[f'y_{tkey}'].to_numpy(float)
    ok = np.isfinite(y) & (np.abs(y) <= MAX_ABS_PPM)
    r = rows[ok]
    y = y[ok]
    sref2 = r[f'sref2_{tkey}'].to_numpy(float)
    sret2 = r['xco2_uncertainty'].to_numpy(float) ** 2
    cld = r['cld_dist_km'].to_numpy(float)

    def block(mask, label):
        yv = y[mask]
        if len(yv) < 100:
            return None
        var_y = float(np.var(yv))
        e_ref = float(np.nanmean(sref2[mask]))
        e_ret = float(np.nanmean(sret2[mask]))
        var_far = float(np.var(y[(cld >= far_km)])) if label == 'all' else np.nan
        return dict(
            subset=label, n=int(len(yv)), var_y=var_y,
            noise_ref=e_ref, noise_ret=e_ret,
            r2max_ref=1 - e_ref / var_y,
            r2max_ref_ret=1 - (e_ref + e_ret) / var_y,
            var_far=var_far,
            r2max_emp=(1 - var_far / var_y) if np.isfinite(var_far) else np.nan)

    out = []
    for lab, m in (('all', np.ones(len(y), bool)), (f'near<{near_km:g}km', cld < near_km)):
        b = block(m, lab)
        if b:
            out.append(b)
    # recompute cross-check on this population
    yrc = r[f'yrc_{tkey}'].to_numpy(float)
    both = np.isfinite(yrc)
    dmax = float(np.nanmax(np.abs(yrc[both] - y[both]))) if both.any() else np.nan
    frac_match = float(np.mean(both))
    return out, dmax, frac_match


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--targets', default='r10,r05,r15')
    ap.add_argument('--near-km', type=float, default=10.0)
    ap.add_argument('--far-km', type=float, default=20.0)
    ap.add_argument('--out', default='results/label_noise_ceiling.csv')
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(',') if t.strip()]

    parts = []
    for i, p in enumerate(map(Path, args.data), 1):
        parts += per_file_rows(p, targets)
        print(f"  [{i}/{len(args.data)}] {p.name}")
    rows = pd.concat(parts, ignore_index=True)
    print(f"pooled {len(rows):,} soundings from {len(args.data)} date file(s)")

    recs = []
    for sfc, sname in ((0, 'ocean'), (1, 'land')):
        sub = rows[rows['sfc_type'] == sfc]
        for tkey in targets:
            blocks, dmax, frac = summarize(sub, tkey, args.near_km, args.far_km)
            print(f"\n== {sname} × {tkey}  (recomputed-vs-stored max|Δ| "
                  f"{dmax:.2e} ppm on {frac:.1%} of rows)")
            for b in blocks:
                recs.append(dict(surface=sname, target=tkey, **b))
                print(f"  {b['subset']:>12}: n={b['n']:>9,}  Var(y)={b['var_y']:.3f}  "
                      f"E[σ²_ref]={b['noise_ref']:.4f}  E[σ²_ret]={b['noise_ret']:.3f}  "
                      f"R²max(ref)={b['r2max_ref']:.3f}  "
                      f"R²max(ref+ret)={b['r2max_ref_ret']:.3f}  "
                      + (f"R²max(emp far≥{args.far_km:g}km)={b['r2max_emp']:.3f}"
                         if np.isfinite(b['var_far']) else ''))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
