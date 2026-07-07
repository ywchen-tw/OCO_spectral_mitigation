"""tccon_correction_policy_stats.py — uncorrected vs corrected XCO2 vs TCCON.

For each `run_case` in curc_shell_blanca_plot_corr_xco2_deepens.sh that has a local
parquet + TCCON file, this builds the deep-ensemble + xgb-cloud corrections (via
build_deepens_plot_data.py) and matches OCO-2 footprints to the TCCON station the
SAME way plot_corrected_xco2.py does (lon/lat box → target day → OCO footprint time
window ±60 min → station = median TCCON lon/lat → footprints within --radius-km).

It then compares, per footprint and pooled across all cases, how close each
corrected XCO2 is to the coincident TCCON window-mean:

  uncorrected   xco2_bc
  full_mu       xco2_bc - mu                 (the deep-ensemble correction)

(The |lat|>L lat-gate scenarios were dropped — the radius sweep showed the full
correction beats the gate at every radius.)  No dependency on the parquet truth
anomaly (xco2_bc_anomaly); footprints are kept by a sane band around the station.

Outputs: a per-case CSV and a pooled-footprint summary (overall + near/far cloud).

Run: PYTHONPATH=src python workspace/tccon_correction_policy_stats.py
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Shared collocation + TCCON reader so this report and tccon_comparison_report.py
# select the IDENTICAL footprints from the SAME build_deepens_plot_data output.
from tccon_collocate import collocate, find_plotdata, load_tccon, _haversine_km

ROOT = Path(__file__).resolve().parent.parent
# On CURC the results/ tree and data/ (TCCON) live under scratch, not the repo
# checkout.  Mirror get_storage_dir()/curc_shell_blanca_general.sh: prefer
# CURC_DATA_ROOT, then OCO2_DATAROOT, else the repo root (local, unchanged).
DATA_ROOT = Path(os.environ.get('CURC_DATA_ROOT')
                 or os.environ.get('OCO2_DATAROOT') or ROOT)
SH = ROOT / 'curc_shell_blanca_plot_corr_xco2_deepens.sh'   # script lives in the repo
CSV_DIR = DATA_ROOT / 'results/csv_collection'
TCCON_DIR = DATA_ROOT / 'data/TCCON'
# Default output dir (overridden by --output-dir, which the plot script points at
# OUT_BASE so everything lands under deep_ensemble/<MODEL_TAG>/).
OUTDIR = DATA_ROOT / 'results/model_comparison/deep_ensemble'

# scenario label -> column in plot_data.  (The |lat|>L lat-gate scenarios were
# dropped — the radius sweep showed the full correction beats the gate everywhere.)
SCEN = {
    'uncorrected': 'xco2_bc',
    'full_mu': 'deep_ensemble_corrected_xco2',
}
GATE = {0: 5.0, 1: 15.0}   # per-surface near-cloud threshold (km)

_RUN = re.compile(r'^\s*run_case\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'
                  r'\s+(\S+)\s+(\S+)\s+(\S+)?')


def parse_cases(script=SH):
    cases = []
    for line in Path(script).read_text().splitlines():
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


def match_case(case, plotdata, radius_km=100.0, window_min=60.0):
    """Return per-footprint frame near the TCCON station (guarded footprints KEPT
    and flagged ``is_guarded``) with tccon_ref/site/date/near-cloud columns, or None
    if no coincident TCCON.  Uses the SHARED collocator, so the footprint set matches
    tccon_comparison_report.py exactly."""
    oco = pd.read_parquet(plotdata)
    tcc = load_tccon(TCCON_DIR / case['tccon'])
    col = collocate(oco, tcc,
                    box=(case['lon'][0], case['lon'][1], case['lat'][0], case['lat'][1]),
                    radius_km=radius_km, window_min=window_min)
    near = col['near']
    if not len(near) or not np.isfinite(col['tccon_ref']):
        return None
    near = near.copy()
    near['tccon_ref'] = col['tccon_ref']
    near['tccon_sd'] = col['tccon_sd']
    near['tccon_n'] = col['n_tccon']
    near['date'] = case['date']
    near['site'] = Path(case['tccon']).name[:2]
    near['gate_km'] = near['sfc_type'].map(GATE)
    near['is_near_cloud'] = near['cld_dist_km'].le(near['gate_km'])
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
    ap.add_argument('--limit', type=int, default=None, help="first N runnable cases")
    ap.add_argument('--plotdata-base', required=True,
                    help="Read each case's processed XCO2 from "
                         "<base>/combined_<date>_<site>/plot_data.parquet "
                         "(the plot script's OUT_BASE). No model is re-run here.")
    ap.add_argument('--output-dir', default=None,
                    help="Where to write the policy CSVs (default: --plotdata-base, so "
                         "everything lands under deep_ensemble/<MODEL_TAG>/).")
    ap.add_argument('--fname-suffix', default='',
                    help="Appended before the extension of every output CSV filename "
                         "(e.g. '_r100km') so a radius sweep's stats coexist.")
    ap.add_argument('--script', default=str(SH),
                    help="Shell script whose run_case lines to read (default: the "
                         "non-drift deepens script).")
    ap.add_argument('--corr-col', default='deep_ensemble_corrected_xco2',
                    help="plot_data column holding the corrected XCO2 for the "
                         "'full_mu' scenario (default 'deep_ensemble_corrected_xco2'; "
                         "use 'tabm_corrected_xco2' for TabM).")
    args = ap.parse_args()
    sfx = args.fname_suffix
    SCEN['full_mu'] = args.corr_col   # point the correction scenario at the chosen model column

    global OUTDIR
    plotdata_base = Path(args.plotdata_base)
    OUTDIR = Path(args.output_dir) if args.output_dir else plotdata_base
    print(f"reading plot_data from {plotdata_base}; out → {OUTDIR}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cases = parse_cases(args.script)
    runnable = [c for c in cases
                if (CSV_DIR / f"combined_{c['date']}_all_orbits.parquet").exists()
                and (TCCON_DIR / c['tccon']).exists()]
    print(f"{len(cases)} run_case lines; {len(runnable)} have local parquet + TCCON")
    if args.limit:
        runnable = runnable[:args.limit]

    per_case, pooled, case_means, surf_means = [], [], [], []
    for i, c in enumerate(runnable, 1):
        print(f"\n[{i}/{len(runnable)}] {c['date']} {c['tccon'][:2]} ({c['surf']})")
        if not precheck(c, args.radius_km, args.window_min):
            print("  no coincident TCCON in window/radius — skip")
            continue
        pd_path = find_plotdata(plotdata_base, c['date'], Path(c['tccon']).name[:2])
        if pd_path is None:
            print("  no plot_data.parquet under --plotdata-base — skip")
            continue
        near = match_case(c, pd_path, args.radius_km, args.window_min)
        if near is None:
            print("  no coincident TCCON / no footprints in radius — skipped")
            continue
        st = _stats(near)
        print(f"  {len(near):,} footprints ≤{args.radius_km:g}km, "
              f"TCCON_ref={near['tccon_ref'].iloc[0]:.2f}ppm  | bias: "
              + "  ".join(f"{k}={st[k]['bias']:+.2f}" for k in
                          ('uncorrected', 'full_mu') if k in st))
        for name, s in st.items():
            per_case.append(dict(date=c['date'], site=near['site'].iloc[0],
                                 surf=c['surf'], scenario=name, **s))
        # station-day means (mean OCO vs TCCON) for the calibration regression and
        # the comparison figures: one overall row + one per surface (ocean/land).
        def _means_row(frame, **extra):
            r = dict(date=c['date'], site=near['site'].iloc[0],
                     n=len(frame), tccon_ref=float(frame['tccon_ref'].iloc[0]),
                     tccon_sd=float(frame['tccon_sd'].iloc[0]),
                     abs_lat_max=float(frame['lat'].abs().max()), **extra)
            for name, col in SCEN.items():
                if col in frame:
                    r[name] = float(frame[col].mean())
                    r[f'{name}_sd'] = float(frame[col].std())
            return r
        case_means.append(_means_row(near, surf=c['surf']))
        for _sfc, _sname in ((0, 'ocean'), (1, 'land')):
            sub = near[near['sfc_type'] == _sfc]
            if len(sub):
                surf_means.append(_means_row(sub, surface=_sname))
        pooled.append(near)

    if not pooled:
        print("\nNo matched cases."); return
    allfp = pd.concat(pooled, ignore_index=True)
    pd.DataFrame(per_case).to_csv(OUTDIR / f'tccon_policy_per_case{sfx}.csv', index=False)

    # ── pooled footprint-level summary: overall + near/far cloud + drop-guards ──
    # 'all' KEEPS guarded footprints (correction skipped there → corrected = raw
    # xco2_bc): the end-to-end number.  'drop_guards' excludes them (correction
    # quality where the model acted) — the two together are the "report both" view.
    def summarize(frame, label):
        out = []
        for name, s in _stats(frame).items():
            out.append(dict(subset=label, scenario=name, **s))
        return out
    summary = []
    summary += summarize(allfp, 'all')
    if 'is_near_cloud' in allfp:
        summary += summarize(allfp[allfp['is_near_cloud']], 'near_cloud')
        summary += summarize(allfp[~allfp['is_near_cloud'].astype(bool)], 'far_cloud')
    if 'is_guarded' in allfp:
        summary += summarize(allfp[~allfp['is_guarded'].astype(bool)], 'drop_guards')
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTDIR / f'tccon_policy_pooled_summary{sfx}.csv', index=False)

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
    cm.to_csv(OUTDIR / f'tccon_policy_station_means{sfx}.csv', index=False)
    # per-surface station-day means (ocean/land) for the surface-separated figure
    pd.DataFrame(surf_means).to_csv(
        OUTDIR / f'tccon_policy_station_means_by_surface{sfx}.csv', index=False)
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
    rdf.to_csv(OUTDIR / f'tccon_policy_station_regression{sfx}.csv', index=False)
    rdf['scenario'] = pd.Categorical(rdf['scenario'], order, ordered=True)
    rdf = rdf.sort_values('scenario')
    for cc in ('slope', 'intercept', 'r2', 'mean_resid', 'rms_resid'):
        rdf[cc] = rdf[cc].round(4)
    print(f"\n===== STATION-DAY CALIBRATION: mean corrected OCO vs TCCON "
          f"({len(cm)} station-days) =====")
    print(rdf[['scenario', 'n_stationdays', 'slope', 'intercept', 'r2',
               'mean_resid', 'rms_resid']].to_string(index=False))

    print(f"\nPer-case  → {OUTDIR / f'tccon_policy_per_case{sfx}.csv'}")
    print(f"Summary   → {OUTDIR / f'tccon_policy_pooled_summary{sfx}.csv'}")
    print(f"Station regression → {OUTDIR / f'tccon_policy_station_regression{sfx}.csv'}")


if __name__ == '__main__':
    main()
