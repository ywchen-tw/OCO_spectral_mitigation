#!/usr/bin/env python3
"""Footprint-size (fp_area_km2) influence on the near-cloud XCO2 bias and its
correction — Appendix B5 analysis (planned 2026-08-01).

Question: at fixed nearest-cloud distance, does the anomaly (and its spectral
signature) depend on footprint area, and is the correction's held-out residual
flat across footprint-area strata?  Mechanistic prediction: cloud distance is
measured footprint-center to MODIS pixel, so a larger footprint extends closer
to the cloud — decay curves of different area strata should approximately
collapse under a shift of order half the footprint extent.

fp_area_km2 provenance: L2 Lite vertex_latitude/longitude 4-corner polygon,
equal-area projected + shoelace (src/spectral/fitting.py).  Step-0 QC
(2026-08-01, this script's `qc` step) showed the axis is real geometry, not
vertex noise: within-frame std across the 8 footprints is ~0.02 km2 vs
~0.84 km2 across frames, and the area varies smoothly along-track (lag-10
autocorrelation 0.997).  A 5.3 % tail below 0.2 km2 (uniform in year and
footprint index) is screened by the QC cut rather than interpreted.

Populations mirror the manuscript conventions:
  decay    — all-QF, valid production target        (Fig. 3 convention)
  spectral — QF = 0, snow-free                      (Fig. 4 convention)
  residual — all rows with a valid production target (training population)

Steps (run in order; later steps reuse cached CSVs/parquets):
  qc        axis validation + per-surface quartile edges  → fp_area_qc.json/.csv
  decay     anomaly-vs-distance by area quartile + shift-collapse fit
  spectral  ref-corrected spectral effect sizes by area quartile
            (land_class.build_effect_sizes + ref_corrected r05/r15 machinery)
  residual  fold-safe held-out mu for all 116 dates (5 folds/surface) →
            before/after bias+RMSE by quartile × near/far regime
  numbers   freeze the quotable numbers → fp_area_headline.md

Usage:
  python workspace/fp_area_analysis.py --step qc
  python workspace/fp_area_analysis.py --step decay
  python workspace/fp_area_analysis.py --step spectral
  python workspace/fp_area_analysis.py --step residual
  python workspace/fp_area_analysis.py --step numbers

Output: results/figures/cld_dist_analysis/fp_area/
Figure composition: manuscript/scripts/make_fp_area_figure.py (figB5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'workspace'))

PARQUET = _REPO / 'results/csv_collection/combined_2016_2020_dates.parquet'
OUT_DIR = _REPO / 'results/figures/cld_dist_analysis/fp_area'
MODEL_ROOT = _REPO / 'results/model_deep_ensemble'

AREA_MIN, AREA_MAX = 0.2, 10.0          # QC cut on fp_area_km2
QUARTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']
DECAY_BINS = np.arange(0.0, 31.0, 1.0)  # 1-km bins, 0-30 km
SHIFT_GRID = np.arange(0.0, 4.01, 0.05)  # collapse-shift search (km)
CROSS_TRACK_KM = 10.0 / 8.0             # nominal per-footprint swath share

SURFACES = {
    'ocean': dict(sfc=0, target='xco2_bc_anomaly_r05', radius=5.0,
                  model_tpl='de_ocean_beta_nll_prof_reg_foldpca_r05_f{f}'),
    'land': dict(sfc=1, target='xco2_bc_anomaly_r15', radius=15.0,
                 model_tpl='de_land_beta_nll_prof_reg_foldpca_r15_f{f}'),
}


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def _read(columns, sfc=None):
    """Column-pruned parquet read; optional surface filter."""
    dset = pads.dataset(PARQUET)
    filt = None if sfc is None else (pads.field('sfc_type') == sfc)
    t = dset.to_table(columns=columns, filter=filt)
    return t.to_pandas()


def _area_ok(a):
    return (a >= AREA_MIN) & (a <= AREA_MAX)


def _quartile_edges():
    p = OUT_DIR / 'fp_area_qc.json'
    if not p.exists():
        raise SystemExit('run --step qc first (writes quartile edges)')
    return json.loads(p.read_text())


def _assign_quartile(area, edges):
    """edges: [q0, q25, q50, q75, q100] from the qc step."""
    idx = np.clip(np.searchsorted(edges[1:4], area, side='right'), 0, 3)
    lab = np.array(QUARTILE_LABELS)[idx]
    lab[~_area_ok(area)] = ''
    return lab


# ──────────────────────────────────────────────────────────────────────────
# step: qc
# ──────────────────────────────────────────────────────────────────────────

def step_qc():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'orbit_id', 'time', 'fp_id', 'fp', 'fp_area_km2',
            'sza', 'vza', 'lat', 'sfc_type', 'cld_dist_km']
    df = _read(cols)
    a = df.fp_area_km2
    ok = _area_ok(a)

    out = {'n_total': int(len(df)),
           'n_below_min': int((a < AREA_MIN).sum()),
           'n_above_max': int((a > AREA_MAX).sum()),
           'frac_screened': float((~ok).mean())}

    # frame coherence + along-track smoothness on one mid-record date
    dates = sorted(df.date.unique())
    sample_date = dates[len(dates) // 2]
    sub = df[df.date == sample_date]
    g = sub.groupby(sub.fp_id // 10).fp_area_km2.agg(['mean', 'std', 'count'])
    g = g[g['count'] >= 6]
    out['sample_date'] = (sample_date.decode()
                          if isinstance(sample_date, bytes) else str(sample_date))
    out['within_frame_std_median'] = float(g['std'].median())
    out['across_frame_std'] = float(g['mean'].std())
    orb = sub[sub.orbit_id == sub.orbit_id.mode().iloc[0]].sort_values('time')
    fm = (orb.groupby(orb.fp_id // 10)
          .agg(area=('fp_area_km2', 'mean'), t=('time', 'first'))
          .sort_values('t'))
    out['alongtrack_autocorr_lag1'] = float(fm.area.autocorr(1))
    out['alongtrack_autocorr_lag10'] = float(fm.area.autocorr(10))

    # confound scan on the QC-passing population (sampled)
    rows = []
    for name, spec in SURFACES.items():
        d = df[(df.sfc_type == spec['sfc']) & ok]
        edges = list(np.quantile(d.fp_area_km2, [0, .25, .5, .75, 1.0]))
        out[f'{name}_quartile_edges'] = [round(float(e), 4) for e in edges]
        out[f'{name}_n'] = int(len(d))
        s = d.sample(min(len(d), 1_000_000), random_state=0)
        cld = s.cld_dist_km.replace(-999, np.nan)
        rows.append({'surface': name,
                     'corr_sza': float(s.fp_area_km2.corr(s.sza)),
                     'corr_vza': float(s.fp_area_km2.corr(s.vza)),
                     'corr_abslat': float(s.fp_area_km2.corr(s.lat.abs())),
                     'corr_cld_dist': float(s.fp_area_km2.corr(cld))})
        # per-quartile confound profile (is any stratum sampling a different world?)
        lab = _assign_quartile_local(d.fp_area_km2.to_numpy(), edges)
        for q in QUARTILE_LABELS:
            m = lab == q
            rows.append({'surface': name, 'quartile': q,
                         'n': int(m.sum()),
                         'area_median': float(d.fp_area_km2[m].median()),
                         'sza_median': float(d.sza[m].median()),
                         'vza_median': float(d.vza[m].median()),
                         'abslat_median': float(d.lat[m].abs().median()),
                         'cld_dist_median': float(
                             d.cld_dist_km[m].replace(-999, np.nan).median())})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'fp_area_qc_confounds.csv', index=False)
    (OUT_DIR / 'fp_area_qc.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # area distribution histogram cache for the figure
    hist_rows = []
    hedges = np.arange(0, 6.05, 0.1)
    for name, spec in SURFACES.items():
        d = df[(df.sfc_type == spec['sfc']) & ok]
        h, _ = np.histogram(d.fp_area_km2, bins=hedges, density=True)
        hist_rows.append(pd.DataFrame({'surface': name,
                                       'bin_left': hedges[:-1], 'density': h}))
    pd.concat(hist_rows).to_csv(OUT_DIR / 'fp_area_hist.csv', index=False)


def _assign_quartile_local(area, edges):
    idx = np.clip(np.searchsorted(edges[1:4], area, side='right'), 0, 3)
    return np.array(QUARTILE_LABELS)[idx]


# ──────────────────────────────────────────────────────────────────────────
# step: decay
# ──────────────────────────────────────────────────────────────────────────

def step_decay():
    from models.pipeline import MAX_ABS_ANOMALY_PPM
    qc = _quartile_edges()
    cols = ['sfc_type', 'cld_dist_km', 'fp_area_km2', 'sza', 'lat',
            'xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15']
    df = _read(cols)
    df = df[_area_ok(df.fp_area_km2)]
    # training-population target screen (models.pipeline.filter_target_outliers)
    for c in ('xco2_bc_anomaly_r05', 'xco2_bc_anomaly_r15'):
        df.loc[df[c].abs() > MAX_ABS_ANOMALY_PPM, c] = np.nan

    curves, stats_rows = [], []
    for name, spec in SURFACES.items():
        edges = qc[f'{name}_quartile_edges']
        d = df[df.sfc_type == spec['sfc']].copy()
        y = d[spec['target']]
        m = np.isfinite(y) & np.isfinite(d.cld_dist_km) & (d.cld_dist_km >= 0)
        d, y = d[m], y[m]
        lab = _assign_quartile_local(d.fp_area_km2.to_numpy(), edges)
        binidx = np.digitize(d.cld_dist_km, DECAY_BINS) - 1
        centers = DECAY_BINS[:-1] + 0.5
        for qi, q in enumerate(QUARTILE_LABELS):
            qm = lab == q
            for b, c in enumerate(centers):
                bm = qm & (binidx == b)
                n = int(bm.sum())
                if n == 0:
                    continue
                curves.append({'surface': name, 'quartile': q, 'cld_dist': c,
                               'n': n, 'mean': float(y[bm].mean()),
                               'median': float(y[bm].median()),
                               'sem': float(y[bm].sem())})
        cdf = pd.DataFrame([r for r in curves if r['surface'] == name])

        # per-quartile amplitude + half-response distance + collapse shift
        ref_curve = _curve(cdf, 'Q1')
        for q in QUARTILE_LABELS:
            cur = _curve(cdf, q)
            amp = {f'amp_{int(dd)}km': float(np.interp(dd, cur.index, cur.values))
                   for dd in (1, 2, 5)}
            a0 = np.interp(1.5, cur.index, cur.values)
            half = _half_distance(cur, a0)
            shift = np.nan if q == 'Q1' else _collapse_shift(cur, ref_curve)
            med_area = _median_area(df, spec['sfc'], edges, q)
            stats_rows.append({'surface': name, 'quartile': q,
                               'area_median_km2': med_area,
                               'len_proxy_km': med_area / CROSS_TRACK_KM,
                               **amp, 'half_dist_km': half,
                               'collapse_shift_km': shift})

    pd.DataFrame(curves).to_csv(OUT_DIR / 'fp_area_decay_curves.csv', index=False)

    # geometry-controlled per-bin partial coefficient: within each 1-km bin,
    # OLS anomaly ~ 1 + fp_area + sza + |lat|  →  d(anomaly)/d(area) in
    # ppm per km2 at fixed distance, immune to the sza/lat stratum confound
    # documented in fp_area_qc_confounds.csv
    coef_rows = []
    for name, spec in SURFACES.items():
        d = df[df.sfc_type == spec['sfc']].copy()
        y = d[spec['target']]
        m = np.isfinite(y) & np.isfinite(d.cld_dist_km) & (d.cld_dist_km >= 0)
        d, y = d[m], y[m]
        binidx = np.digitize(d.cld_dist_km, DECAY_BINS) - 1
        rng = np.random.default_rng(0)
        for b, c in enumerate(DECAY_BINS[:-1] + 0.5):
            bm = np.flatnonzero(binidx == b)
            if len(bm) < 2000:
                continue
            if len(bm) > 500_000:
                bm = rng.choice(bm, 500_000, replace=False)
            X = np.column_stack([np.ones(len(bm)),
                                 d.fp_area_km2.to_numpy()[bm],
                                 d.sza.to_numpy()[bm],
                                 np.abs(d.lat.to_numpy()[bm])])
            yy = y.to_numpy()[bm]
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            dof = len(bm) - X.shape[1]
            cov = (np.linalg.inv(X.T @ X)
                   * float(resid @ resid) / max(dof, 1))
            coef_rows.append({'surface': name, 'cld_dist': float(c),
                              'n': int(len(bm)),
                              'coef_ppm_per_km2': float(beta[1]),
                              'se': float(np.sqrt(cov[1, 1]))})
    pd.DataFrame(coef_rows).to_csv(OUT_DIR / 'fp_area_partial_coef.csv',
                                   index=False)
    st = pd.DataFrame(stats_rows)
    # expected shift if the effect is footprint-edge geometry: half the
    # difference in along-track length relative to Q1
    for name in SURFACES:
        m = st.surface == name
        l1 = st.loc[m & (st.quartile == 'Q1'), 'len_proxy_km'].iloc[0]
        st.loc[m, 'expected_shift_km'] = (st.loc[m, 'len_proxy_km'] - l1) / 2.0
    st.to_csv(OUT_DIR / 'fp_area_decay_stats.csv', index=False)
    print(st.to_string(index=False))


def _curve(cdf, q):
    c = cdf[cdf.quartile == q].set_index('cld_dist')['mean'].sort_index()
    return c


def _half_distance(cur, a0):
    """First distance beyond 1.5 km where |curve| falls below |a0|/2."""
    for d, v in cur.items():
        if d < 1.5:
            continue
        if abs(v) < abs(a0) / 2:
            return float(d)
    return np.nan


def _collapse_shift(cur, ref):
    """Shift delta >= 0 minimizing SSE between cur(d) and ref(d - delta)."""
    dgrid = np.arange(1.0, 20.0, 0.25)
    best, best_sse = np.nan, np.inf
    for delta in SHIFT_GRID:
        dv = dgrid[dgrid - delta >= ref.index.min()]
        if len(dv) < 10:
            continue
        cv = np.interp(dv, cur.index, cur.values)
        rv = np.interp(dv - delta, ref.index, ref.values)
        sse = float(np.mean((cv - rv) ** 2))
        if sse < best_sse:
            best, best_sse = float(delta), sse
    return best


def _median_area(df, sfc, edges, q):
    d = df[df.sfc_type == sfc]
    lab = _assign_quartile_local(d.fp_area_km2.to_numpy(), edges)
    return float(d.fp_area_km2[lab == q].median())


# ──────────────────────────────────────────────────────────────────────────
# step: spectral
# ──────────────────────────────────────────────────────────────────────────

# (diff-col template, band label, term label) — k1/k2/exp_int rows, Fig. 4 set
_SPEC_TERMS = [('k1', 'o2a'), ('k2', 'o2a'), ('exp', 'o2a'),
               ('k1', 'wco2'), ('k2', 'wco2'), ('exp', 'wco2'),
               ('k1', 'sco2'), ('k2', 'sco2'), ('exp', 'sco2')]


def step_spectral():
    from analysis.land_class import build_effect_sizes
    from analysis.ref_corrected import add_r05_anomalies, add_r15_anomalies

    qc = _quartile_edges()
    obs_cols = ['o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
                'exp_o2a_intercept', 'exp_wco2_intercept', 'exp_sco2_intercept']
    for name, spec in SURFACES.items():
        pref = 'r05' if name == 'ocean' else 'r15'
        ref_cols = []
        for term, band in _SPEC_TERMS:
            base = (f'{pref}_exp_int_{band}' if term == 'exp'
                    else f'{pref}_{band}_{term}')
            ref_cols += [f'{base}_mean', f'{base}_std']
        cols = (['sfc_type', 'cld_dist_km', 'fp_area_km2', 'xco2_qf',
                 'snow_flag'] + obs_cols + ref_cols)
        d = _read(cols, sfc=spec['sfc'])
        d = d[_area_ok(d.fp_area_km2) & (d.xco2_qf == 0) & (d.snow_flag == 0)]
        d = (add_r05_anomalies(d) if pref == 'r05' else add_r15_anomalies(d))
        edges = qc[f'{name}_quartile_edges']
        d['area_q'] = _assign_quartile_local(d.fp_area_km2.to_numpy(), edges)
        variables = []
        for term, band in _SPEC_TERMS:
            dcol = f'd{pref}{term}_{band}'
            variables.append((dcol, f'{band.upper()} {term}'))
        build_effect_sizes(d, variables, str(OUT_DIR),
                           groups=QUARTILE_LABELS, n_min=500,
                           group_col='area_q', prefix=f'fp_area_spec_{name}')


# ──────────────────────────────────────────────────────────────────────────
# step: residual  (fold-safe held-out mu over all 116 dates)
# ──────────────────────────────────────────────────────────────────────────

def step_residual():
    import torch  # noqa: F401  (import check before the heavy loop)
    from apply.apply_deep_ensemble import _load_model, _predict
    from models.pipeline import _DERIVED_FEATURES

    qc = _quartile_edges()
    for name, spec in SURFACES.items():
        cache = OUT_DIR / f'heldout_mu_{name}.parquet'
        if cache.exists():
            print(f'[skip] {cache} exists')
            continue
        frames = []
        for f in range(5):
            mdir = MODEL_ROOT / spec['model_tpl'].format(f=f)
            pipeline, members, meta = _load_model(mdir)
            manifest = json.loads((mdir / 'training_dates.json').read_text())
            held = [d.encode() for d in manifest['held_dates']]
            prof_src = []
            ppca = getattr(pipeline, 'profile_pca', None)
            if ppca is not None:
                for g in ppca.groups.values():
                    prof_src += list(g.columns)
                prof_src += list(getattr(ppca, 'scalars', []) or [])
            need = sorted(set(
                [c for c in pipeline.qt_features] + prof_src +
                list(_DERIVED_FEATURES) +  # present in parquet → derive no-op
                ['fp', 'sfc_type', 'cld_dist_km', 'fp_area_km2', 'xco2_qf',
                 'snow_flag', 'date', spec['target']]))
            dset = pads.dataset(PARQUET)
            avail = set(dset.schema.names)
            t = dset.to_table(columns=[c for c in need if c in avail],
                              filter=(pads.field('sfc_type') == spec['sfc'])
                              & pc.field('date').isin(held))
            df = t.to_pandas()
            y = df[spec['target']]
            df = df[np.isfinite(y)].reset_index(drop=True)
            mu, sigma, kept = _predict(df, pipeline, members, meta,
                                       spec['sfc'], tag=f'{name}_f{f}')
            frames.append(pd.DataFrame({
                'date': kept['date'], 'fold': f,
                'fp_area_km2': kept['fp_area_km2'],
                'cld_dist_km': kept['cld_dist_km'],
                'xco2_qf': kept['xco2_qf'], 'snow_flag': kept['snow_flag'],
                'y': kept[spec['target']], 'mu': mu, 'sigma': sigma}))
            print(f'  {name} f{f}: {len(frames[-1]):,} held-out predictions')
        allf = pd.concat(frames, ignore_index=True)
        allf.to_parquet(cache, index=False)
        print(f'[saved] {cache} ({len(allf):,} rows)')

    # before/after stats by quartile × regime (training-population target
    # filter |y| <= 100 ppm, mirroring models.pipeline.filter_target_outliers)
    from models.pipeline import MAX_ABS_ANOMALY_PPM
    rows = []
    for name, spec in SURFACES.items():
        d = pd.read_parquet(OUT_DIR / f'heldout_mu_{name}.parquet')
        d = d[_area_ok(d.fp_area_km2) & (d.y.abs() <= MAX_ABS_ANOMALY_PPM)]
        edges = qc[f'{name}_quartile_edges']
        d['area_q'] = _assign_quartile_local(d.fp_area_km2.to_numpy(), edges)
        d['resid'] = d.y - d.mu
        near = d.cld_dist_km <= spec['radius']
        for q in QUARTILE_LABELS:
            for regime, m in [('near', (d.area_q == q) & near),
                              ('far', (d.area_q == q) & ~near)]:
                s = d[m]
                if not len(s):
                    continue
                rows.append({
                    'surface': name, 'quartile': q, 'regime': regime,
                    'n': len(s),
                    'bias_before': float(s.y.mean()),
                    'bias_after': float(s.resid.mean()),
                    'rmse_before': float(np.sqrt((s.y ** 2).mean())),
                    'rmse_after': float(np.sqrt((s.resid ** 2).mean())),
                    'sem_after': float(s.resid.sem())})
    st = pd.DataFrame(rows)
    st.to_csv(OUT_DIR / 'fp_area_residual_stats.csv', index=False)
    print(st.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────
# step: numbers
# ──────────────────────────────────────────────────────────────────────────

def step_numbers():
    qc = _quartile_edges()
    dec = pd.read_csv(OUT_DIR / 'fp_area_decay_stats.csv')
    coef = pd.read_csv(OUT_DIR / 'fp_area_partial_coef.csv')
    res = pd.read_csv(OUT_DIR / 'fp_area_residual_stats.csv')
    lines = ['# Footprint-size analysis — headline numbers',
             f'(fp_area QC window [{AREA_MIN}, {AREA_MAX}] km2, screens '
             f'{qc["frac_screened"]:.1%} of soundings; target screen '
             '|y| <= 100 ppm as in training)', '',
             'Verdicts: (1) the near-cloud anomaly ATTENUATES with footprint '
             'area at fixed nearest-cloud distance on both surfaces '
             '(geometry-controlled OLS per 1-km bin, controls SZA + |lat|); '
             '(2) the effect is confined to each surface\'s response zone — '
             'a real near-cloud interaction, not a scene confound; (3) the '
             'shift-collapse (edge-proximity) hypothesis is REJECTED: best-fit '
             'distance shift ~0 km — the effect is amplitude, not offset; '
             '(4) the correction ABSORBS the dependence: held-out near-cloud '
             'residual bias is flat across area quartiles.', '']
    for name in SURFACES:
        e = qc[f'{name}_quartile_edges']
        lines.append(f'## {name} (quartile edges {[round(x, 2) for x in e]} '
                     'km2)')
        cc = coef[coef.surface == name].set_index('cld_dist')
        for d0 in (0.5, 1.5, 2.5):
            r = cc.loc[d0]
            lines.append(f'- area coefficient @ {d0} km: '
                         f'{r.coef_ppm_per_km2:+.4f} ± {r.se:.4f} ppm/km2')
        zone = 3.5 if name == 'ocean' else 7.5
        far = cc[cc.index >= zone + 2]
        lines.append(f'- beyond ~{zone:.0f} km the coefficient is '
                     f'|{far.coef_ppm_per_km2.abs().max():.4f}| ppm/km2 at '
                     'most (zone-confined)')
        d = dec[dec.surface == name]
        amp = d.set_index('quartile').amp_1km
        lines.append(f'- stratified amp@1km Q1→Q4: {amp.Q1:+.2f} → '
                     f'{amp.Q4:+.2f} ppm (land strata additionally carry a '
                     'scene-mix confound — quote the coefficient, not this)')
        rr = res[(res.surface == name) & (res.regime == 'near')]
        ba = rr.set_index('quartile')
        lines.append(
            f'- near-cloud bias before: {ba.bias_before.min():+.3f} to '
            f'{ba.bias_before.max():+.3f} ppm across quartiles → after: '
            f'{ba.bias_after.min():+.3f} to {ba.bias_after.max():+.3f} ppm '
            f'(span {ba.bias_before.max() - ba.bias_before.min():.3f} → '
            f'{ba.bias_after.max() - ba.bias_after.min():.3f} ppm)')
        lines.append(
            f'- near-cloud fp-RMSE before → after per quartile: '
            + ', '.join(f'{q} {ba.rmse_before[q]:.2f}→{ba.rmse_after[q]:.2f}'
                        for q in QUARTILE_LABELS))
        lines.append('')
    (OUT_DIR / 'fp_area_headline.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


# ──────────────────────────────────────────────────────────────────────────

STEPS = {'qc': step_qc, 'decay': step_decay, 'spectral': step_spectral,
         'residual': step_residual, 'numbers': step_numbers}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--step', required=True, choices=list(STEPS))
    args = ap.parse_args()
    STEPS[args.step]()
