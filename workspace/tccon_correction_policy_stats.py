"""tccon_correction_policy_stats.py — compare 3 correction policies vs TCCON.

For each `run_case` in curc_shell_blanca_plot_corr_xco2_deepens.sh that has a local
parquet + TCCON file, this builds the deep-ensemble + xgb-cloud corrections (via
build_deepens_plot_data.py) and matches OCO-2 footprints to the TCCON station the
SAME way plot_corrected_xco2.py does (lon/lat box → target day → OCO footprint time
window ±60 min → station = median TCCON lon/lat → footprints within --radius-km).

It then compares, per footprint and pooled across all cases, how close each
corrected XCO2 is to the coincident TCCON window-mean:

  uncorrected   xco2_bc
  full_mu       xco2_bc - mu                 (current correction)
  latgate{L}    xco2_bc - mu*1[|lat|<=L]     (skip correction at |lat| > L)

No dependency on the parquet truth anomaly (xco2_bc_anomaly); footprints are kept
by a sane band around the station only.

Outputs: a per-case CSV and a pooled-footprint summary (overall + latitude bands).

Run: PYTHONPATH=src python workspace/tccon_correction_policy_stats.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc4

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / 'curc_shell_blanca_plot_corr_xco2_deepens.sh'
CSV_DIR = ROOT / 'results/csv_collection'
TCCON_DIR = ROOT / 'data/TCCON'
CACHE = ROOT / 'results/model_comparison/tccon_policy/plotdata'
OUTDIR = ROOT / 'results/model_comparison/tccon_policy'

DE_OCEAN = sorted((ROOT / 'results/model_deep_ensemble').glob('de_ocean_beta_nll_f*'))
DE_LAND = sorted((ROOT / 'results/model_deep_ensemble').glob('de_land_beta_nll_f*'))
XGB_OCEAN = sorted((ROOT / 'results/model_xgb_cloud').glob('xgbcloud_final_ocean_f*'))
XGB_LAND = sorted((ROOT / 'results/model_xgb_cloud').glob('xgbcloud_final_land_f*'))

# Latitude gates: skip the correction where |lat| > L (degrees), N and S.
LAT_GATES = (80.0, 75.0, 70.0)

# scenario label -> column in plot_data (lat-gate columns are derived in match_case)
SCEN = {
    'uncorrected': 'xco2_bc',
    'full_mu': 'deep_ensemble_corrected_xco2',
    **{f'latgate{int(L)}': f'latgate{int(L)}' for L in LAT_GATES},
}
GATE = {0: 5.0, 1: 15.0}   # per-surface near-cloud threshold (km)

_RUN = re.compile(r'^\s*run_case\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'
                  r'\s+(\S+)\s+(\S+)\s+(\S+)?')


def parse_cases():
    cases = []
    for line in SH.read_text().splitlines():
        if line.lstrip().startswith('#'):
            continue
        m = _RUN.match(line)
        if not m:
            continue
        date, tccon, lonmin, lonmax, latmin, latmax, _vmin, _vmax, surf = m.groups()
        cases.append(dict(date=date, tccon=tccon,
                          lon=(float(lonmin), float(lonmax)),
                          lat=(float(latmin), float(latmax)),
                          surf=surf or 'both'))
    return cases


def _haversine_km(lon, lat, lon0, lat0):
    R = 6371.0088
    lon = np.radians(np.asarray(lon, float)); lat = np.radians(np.asarray(lat, float))
    lon0 = np.radians(lon0); lat0 = np.radians(lat0)
    dlon = lon - lon0; dlat = lat - lat0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_tccon(path):
    with nc4.Dataset(path, 'r') as ds:
        t = np.ma.filled(ds.variables['time'][:], np.nan).astype(float)
        lat = np.ma.filled(ds.variables['lat'][:], np.nan).astype(float)
        lon = np.ma.filled(ds.variables['long'][:], np.nan).astype(float)
        x = np.ma.filled(ds.variables['xco2'][:], np.nan).astype(float)
    times = pd.Timestamp('1970-01-01', tz='UTC') + pd.to_timedelta(t, unit='s', errors='coerce')
    df = pd.DataFrame({'time': times, 'lat': lat, 'lon': lon, 'xco2': x})
    return df[(df['xco2'] > 300) & (df['xco2'] < 550) & df['xco2'].notna()].reset_index(drop=True)


def precheck(case, radius_km=100.0, window_min=60.0):
    """Cheap coincidence test on the RAW parquet (no model build): is there TCCON
    on the day within ±window of the scene pass, and any footprint within radius?"""
    inp = CSV_DIR / f"combined_{case['date']}_all_orbits.parquet"
    raw = pd.read_parquet(inp, columns=['time', 'lon', 'lat'])
    (lo0, lo1), (la0, la1) = case['lon'], case['lat']
    raw = raw[(raw['lon'] >= lo0) & (raw['lon'] <= lo1)
              & (raw['lat'] >= la0) & (raw['lat'] <= la1)]
    if raw.empty:
        return False
    t = pd.to_datetime(raw['time'], unit='s', utc=True)
    tcc = load_tccon(TCCON_DIR / case['tccon'])
    day = pd.Timestamp(case['date'], tz='UTC').normalize()
    tcc = tcc[(tcc['time'] >= day) & (tcc['time'] < day + pd.Timedelta(days=1))]
    buf = pd.Timedelta(minutes=window_min)
    tcc = tcc[(tcc['time'] >= t.min() - buf) & (tcc['time'] <= t.max() + buf)]
    if tcc.empty:
        return False
    st_lon, st_lat = float(tcc['lon'].median()), float(tcc['lat'].median())
    d = _haversine_km(raw['lon'].values, raw['lat'].values, st_lon, st_lat)
    return bool(np.any(d <= radius_km))


def build_plotdata(case, force=False):
    date, surf = case['date'], case['surf']
    out = CACHE / f"plotdata_{date}_{Path(case['tccon']).name[:2]}.parquet"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    inp = CSV_DIR / f"combined_{date}_all_orbits.parquet"
    cmd = [sys.executable, str(ROOT / 'workspace/build_deepens_plot_data.py'),
           '--input', str(inp), '--output', str(out)]
    if surf in ('both', 'ocean'):
        cmd += ['--ocean-model-dir', *map(str, DE_OCEAN),
                '--ocean-cloud-model-dir', *map(str, XGB_OCEAN)]
    if surf in ('both', 'land'):
        cmd += ['--land-model-dir', *map(str, DE_LAND),
                '--land-cloud-model-dir', *map(str, XGB_LAND)]
    env = {'PYTHONPATH': 'src'}
    import os
    r = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, **env},
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"  build FAILED for {date}: {r.stderr.strip().splitlines()[-1:]}")
        return None
    return out


def match_case(case, plotdata, radius_km=100.0, window_min=60.0):
    """Return per-footprint frame near the TCCON station with a 'tccon_ref' column,
    or None if no coincident TCCON."""
    oco = pd.read_parquet(plotdata)
    (lo0, lo1), (la0, la1) = case['lon'], case['lat']
    oco = oco[(oco['lon'] >= lo0) & (oco['lon'] <= lo1)
              & (oco['lat'] >= la0) & (oco['lat'] <= la1)].copy()
    if oco.empty:
        return None
    oco_t = pd.to_datetime(oco['time'], unit='s', utc=True)

    tcc = load_tccon(TCCON_DIR / case['tccon'])
    day = pd.Timestamp(case['date'], tz='UTC').normalize()
    tcc = tcc[(tcc['time'] >= day) & (tcc['time'] < day + pd.Timedelta(days=1))]
    if tcc.empty:
        return None
    buf = pd.Timedelta(minutes=window_min)
    tcc_w = tcc[(tcc['time'] >= oco_t.min() - buf) & (tcc['time'] <= oco_t.max() + buf)]
    if tcc_w.empty:
        return None
    st_lon, st_lat = float(tcc_w['lon'].median()), float(tcc_w['lat'].median())
    tref = float(tcc_w['xco2'].mean())

    d = _haversine_km(oco['lon'].values, oco['lat'].values, st_lon, st_lat)
    near = oco[d <= radius_km].copy()
    # Quality filter: a sane band around the station only (does NOT use the parquet
    # truth anomaly).  50 ppm is wide enough to retain genuine large near-cloud
    # anomalies while still dropping gross fill values / failed retrievals.
    near = near[np.abs(near['xco2_bc'].to_numpy(float) - tref) < 50.0]
    if near.empty:
        return None
    near['tccon_ref'] = tref
    near['tccon_sd'] = float(tcc_w['xco2'].std())
    near['tccon_n'] = len(tcc_w)
    near['date'] = case['date']
    near['site'] = Path(case['tccon']).name[:2]
    near['gate_km'] = near['sfc_type'].map(GATE)
    near['is_near_cloud'] = near['cld_dist_km'].le(near['gate_km'])
    # latitude-gated corrections: apply mu only where |lat| <= L
    xb = near['xco2_bc'].to_numpy(float)
    mu = near['pred_anomaly'].to_numpy(float)
    alat = near['lat'].abs().to_numpy(float)
    for L in LAT_GATES:
        near[f'latgate{int(L)}'] = xb - np.where(alat <= L, mu, 0.0)
    return near


def _stats(frame):
    """Per-scenario bias/RMSE/std of (corrected - tccon_ref) over the footprints."""
    rows = {}
    ref = frame['tccon_ref'].to_numpy(float)
    for name, col in SCEN.items():
        if col not in frame:
            continue
        v = frame[col].to_numpy(float)
        d = v - ref
        m = np.isfinite(d)
        rows[name] = dict(n=int(m.sum()),
                          bias=float(np.mean(d[m])),
                          rmse=float(np.sqrt(np.mean(d[m] ** 2))),
                          std=float(np.std(d[m])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--force', action='store_true', help="rebuild plot_data cache")
    ap.add_argument('--limit', type=int, default=None, help="first N runnable cases")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cases = parse_cases()
    runnable = [c for c in cases
                if (CSV_DIR / f"combined_{c['date']}_all_orbits.parquet").exists()
                and (TCCON_DIR / c['tccon']).exists()]
    print(f"{len(cases)} run_case lines; {len(runnable)} have local parquet + TCCON")
    if args.limit:
        runnable = runnable[:args.limit]

    per_case, pooled, case_means = [], [], []
    for i, c in enumerate(runnable, 1):
        print(f"\n[{i}/{len(runnable)}] {c['date']} {c['tccon'][:2]} ({c['surf']})")
        if not precheck(c, args.radius_km, args.window_min):
            print("  no coincident TCCON in window/radius — skip (no build)")
            continue
        pd_path = build_plotdata(c, force=args.force)
        if pd_path is None:
            continue
        near = match_case(c, pd_path, args.radius_km, args.window_min)
        if near is None:
            print("  no coincident TCCON / no footprints in radius — skipped")
            continue
        st = _stats(near)
        nhi = int((near['lat'].abs() > min(LAT_GATES)).sum())
        print(f"  {len(near):,} footprints ≤{args.radius_km:g}km ({nhi} with |lat|>{int(min(LAT_GATES))}), "
              f"TCCON_ref={near['tccon_ref'].iloc[0]:.2f}ppm  | bias: "
              + "  ".join(f"{k}={st[k]['bias']:+.2f}" for k in
                          ('uncorrected', 'full_mu', f'latgate{int(min(LAT_GATES))}') if k in st))
        for name, s in st.items():
            per_case.append(dict(date=c['date'], site=near['site'].iloc[0],
                                 surf=c['surf'], scenario=name, **s))
        # one (mean OCO vs TCCON) point per station-day for the calibration regression
        row = dict(date=c['date'], site=near['site'].iloc[0], surf=c['surf'],
                   n=len(near), tccon_ref=float(near['tccon_ref'].iloc[0]),
                   tccon_sd=float(near['tccon_sd'].iloc[0]),
                   abs_lat_max=float(near['lat'].abs().max()))
        for name, col in SCEN.items():
            if col in near:
                row[name] = float(near[col].mean())
                row[f'{name}_sd'] = float(near[col].std())
        case_means.append(row)
        pooled.append(near)

    if not pooled:
        print("\nNo matched cases."); return
    allfp = pd.concat(pooled, ignore_index=True)
    pd.DataFrame(per_case).to_csv(OUTDIR / 'tccon_policy_per_case.csv', index=False)

    # ── pooled footprint-level summary: overall + near/far cloud ───────────────
    def summarize(frame, label):
        out = []
        for name, s in _stats(frame).items():
            out.append(dict(subset=label, scenario=name, **s))
        return out
    alat = allfp['lat'].abs()
    summary = []
    summary += summarize(allfp, 'all')
    for L in LAT_GATES:
        summary += summarize(allfp[alat > L], f'lat>{int(L)}')
    summary += summarize(allfp[alat <= min(LAT_GATES)], f'lat<={int(min(LAT_GATES))}')
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTDIR / 'tccon_policy_pooled_summary.csv', index=False)

    order = list(SCEN.keys())
    sdf['scenario'] = pd.Categorical(sdf['scenario'], order, ordered=True)
    sdf = sdf.sort_values(['subset', 'scenario'])
    for c in ('bias', 'rmse', 'std'):
        sdf[c] = sdf[c].round(3)
    print(f"\n===== POOLED OCO−TCCON BY POLICY ({len(allfp):,} footprints, "
          f"{allfp['date'].nunique()} case-days) =====")
    print(sdf[['subset', 'scenario', 'n', 'bias', 'rmse', 'std']].to_string(index=False))
    # ── station-day calibration: regress mean corrected OCO on TCCON ───────────
    cm = pd.DataFrame(case_means)
    cm.to_csv(OUTDIR / 'tccon_policy_station_means.csv', index=False)
    reg = []
    for name in SCEN:
        if name not in cm.columns:
            continue
        sub = cm.dropna(subset=['tccon_ref', name])
        x = sub['tccon_ref'].to_numpy(float); y = sub[name].to_numpy(float)
        if len(x) < 3:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        reg.append(dict(scenario=name, n_stationdays=len(x),
                        slope=float(slope), intercept=float(intercept), r2=r * r,
                        mean_resid=float(np.mean(y - x)),
                        rms_resid=float(np.sqrt(np.mean((y - x) ** 2)))))
    rdf = pd.DataFrame(reg)
    rdf.to_csv(OUTDIR / 'tccon_policy_station_regression.csv', index=False)
    rdf['scenario'] = pd.Categorical(rdf['scenario'], order, ordered=True)
    rdf = rdf.sort_values('scenario')
    for cc in ('slope', 'intercept', 'r2', 'mean_resid', 'rms_resid'):
        rdf[cc] = rdf[cc].round(4)
    print(f"\n===== STATION-DAY CALIBRATION: mean corrected OCO vs TCCON "
          f"({len(cm)} station-days) =====")
    print(rdf[['scenario', 'n_stationdays', 'slope', 'intercept', 'r2',
               'mean_resid', 'rms_resid']].to_string(index=False))

    print(f"\nPer-case  → {OUTDIR / 'tccon_policy_per_case.csv'}")
    print(f"Summary   → {OUTDIR / 'tccon_policy_pooled_summary.csv'}")
    print(f"Station regression → {OUTDIR / 'tccon_policy_station_regression.csv'}")


if __name__ == '__main__':
    main()
