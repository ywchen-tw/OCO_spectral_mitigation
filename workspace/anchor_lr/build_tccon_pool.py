"""build_tccon_pool.py — step 1: TCCON station-days on the TRAINING dates.

Scans results/csv_collection/combined_2016_2020_dates.parquet (one row group
per training date) for land footprints within --radius-km of any TCCON station
that has a GGG2020 file under data/TCCON, collocates them with the shared
tccon_collocate.collocate() (same box/radius/window/sanity rules as the
evaluation report) and writes, per (date, site):

    results/anchor_lr/pool/combined_<date>_<site>/combined_<date>_<site>_source.parquet

with ALL columns of the kept rows, so build_deepens_plot_data.py and the AK
operator (ak_NN/pwf_NN/co2_ap_NN/plev_NN) need no further lookup.  A manifest
(results/anchor_lr/pool_manifest.csv) records counts, the direct TCCON
reference and the raw land fold whose HELD set contains the date (exactly one
— that fold gives out-of-fold mu_raw in step 2).

Checks: pool dates are disjoint from the active run_case evaluation dates.

Run:  python workspace/anchor_lr/build_tccon_pool.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_common as ac                                       # noqa: E402
from tccon_collocate import SITE_COORDS, collocate, _haversine_km  # noqa: E402
from plot_corrected_xco2 import load_tccon                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--min-fp', type=int, default=10,
                    help='minimum collocated land footprints per station-day (report default)')
    ap.add_argument('--box-deg', type=float, default=2.0,
                    help='half-width of the lon/lat pre-selection box around the station')
    ap.add_argument('--force', action='store_true', help='rewrite existing case parquets')
    args = ap.parse_args()

    stations = {}
    for code, (lon, lat) in SITE_COORDS.items():
        f = ac.tccon_file(code)
        if f is not None:
            stations[code] = (lon, lat, f)
    print(f'{len(stations)} stations with a GGG2020 file: {sorted(stations)}')
    eval_dates = ac.active_run_case_dates()

    pf = pq.ParquetFile(ac.TRAIN_PARQUET)
    tccon_cache: dict[str, pd.DataFrame] = {}
    rows = []
    t_start = time.time()
    for rg in range(pf.num_row_groups):
        light = pf.read_row_group(rg, columns=['date', 'lon', 'lat', 'sfc_type']).to_pandas()
        dates = light['date'].map(lambda b: b.decode() if isinstance(b, bytes) else str(b)).unique()
        if len(dates) != 1:
            raise SystemExit(f'row group {rg}: {len(dates)} dates {dates[:5]} — expected one per group')
        date = str(dates[0])
        land = (light['sfc_type'].to_numpy(float) == 1)
        lon = light['lon'].to_numpy(float); lat = light['lat'].to_numpy(float)
        cand = {}
        for code, (slon, slat, _) in stations.items():
            d = _haversine_km(lon, lat, slon, slat)
            idx = np.flatnonzero(land & (d <= args.radius_km))
            if len(idx) >= args.min_fp:
                cand[code] = idx
        if not cand:
            continue
        full = pf.read_row_group(rg).to_pandas()
        for code, idx in cand.items():
            slon, slat, tfile = stations[code]
            if code not in tccon_cache:
                tccon_cache[code] = load_tccon(str(tfile))
            sub = full.iloc[idx]
            col = collocate(sub, tccon_cache[code],
                            box=(slon - args.box_deg, slon + args.box_deg,
                                 slat - args.box_deg, slat + args.box_deg),
                            radius_km=args.radius_km, window_min=args.window_min, site=code)
            near = col['near']
            if col['n_tccon'] == 0 or len(near) < args.min_fp:
                continue
            near = near.drop(columns=['is_guarded'], errors='ignore')
            name = ac.case_name(date, code)
            out_dir = ac.POOL_DIR / name
            out_path = out_dir / f'{name}_source.parquet'
            if args.force or not out_path.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                near.to_parquet(out_path, index=False)
            qf = near['xco2_qf'].to_numpy(float)
            rows.append(dict(
                date=date, site=code, n_land=int(len(near)), n_qf0=int((qf == 0).sum()),
                n_tccon=int(col['n_tccon']), tccon_ref_direct=float(col['tccon_ref']),
                tccon_sd=float(col['tccon_sd']), held_fold=ac.held_fold_for_date(date),
                st_lon=float(col['st_lon']), st_lat=float(col['st_lat']),
                source=str(out_path.relative_to(ac.ROOT))))
            print(f'  {date} {code}: n_land={len(near):5d} qf0={int((qf == 0).sum()):5d} '
                  f'n_tccon={col["n_tccon"]:4d} ref={col["tccon_ref"]:.2f} fold={rows[-1]["held_fold"]}')
        print(f'[{rg + 1}/{pf.num_row_groups}] {date}: {len(cand)} candidate station(s) '
              f'({time.time() - t_start:.0f} s)')

    man = pd.DataFrame(rows).sort_values(['date', 'site']).reset_index(drop=True)
    overlap = sorted(set(man['date']) & eval_dates)
    if overlap:
        raise SystemExit(f'pool dates overlap active evaluation dates: {overlap}')
    ac.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    man.to_csv(ac.MANIFEST, index=False)
    print(f'\npool: {len(man)} station-days, {man.site.nunique()} sites, '
          f'{man.date.nunique()} dates, {int(man.n_land.sum()):,} land fp '
          f'({int(man.n_qf0.sum()):,} QF=0); disjoint from {len(eval_dates)} evaluation dates')
    print(man.groupby('site').size().sort_values(ascending=False).to_string())
    print(f'→ {ac.MANIFEST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
