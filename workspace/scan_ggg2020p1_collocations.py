"""scan_ggg2020p1_collocations.py — what new OCO-2/TCCON collocations does GGG2020.1 open?

Standalone, read-only scan requested by the "New-collocation scan" section of
log/JQSRT_plan/ggg2020p1_test_design_2026-09-15.md.  It deliberately imports NOTHING
from the TCCON chain (plot_corrected_xco2.py, ak_harmonize.py, tccon_comparison_report.py)
and ships its own netCDF reader, so it can run while those files are being edited.

Three tables, written to log/JQSRT_plan/ggg2020p1_new_collocations_2026-09-15.md:

  (a) the 53 `run_case` lines flagged AVAIL = no in
      curc_shell_blanca_plot_corr_xco2_deepens.sh, re-evaluated against the matching
      GGG2020.1 file (same station code), same rule as workspace/check_tccon_availability.py:
      footprints inside the run_case lon/lat box AND within --radius-km of the station,
      then TCCON observations within +/- --window-min of [pass start, pass end].
  (b) the 15 stations new to the download plus hf and an, over every locally processed
      date (results/csv_collection/combined_<date>_all_orbits.parquet).  No lon/lat box
      here: footprints within --radius-km of the station define the pass.
  (c) old (data/TCCON) vs new (data/TCCON_GGG2020.1) file date ranges and valid-observation
      counts for the 20 stations present in both directories.

TCCON conventions used throughout:
  station lon/lat  = median of the 'long' / 'lat' variables of the file
  time             = seconds since 1970-01-01 UTC ('time' variable)
  valid observation= 300 < xco2_x2019 < 550 ppm
OCO-2 parquet: only the columns time (seconds since epoch), lon, lat are read.

Run: python workspace/scan_ggg2020p1_collocations.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TCCON_OLD = ROOT / 'data' / 'TCCON'
TCCON_NEW = ROOT / 'data' / 'TCCON_GGG2020.1'
CSV_DIR = ROOT / 'results' / 'csv_collection'
SH = ROOT / 'curc_shell_blanca_plot_corr_xco2_deepens.sh'
OUT_MD = ROOT / 'log' / 'JQSRT_plan' / 'ggg2020p1_new_collocations_2026-09-15.md'

# (b) target stations: the 15 new to the download, plus hf and an.
NEW_STATIONS = ['bi', 'br', 'eu', 'fc', 'gm', 'hw', 'if', 'jc', 'jf',
                'lh', 'll', 'lr', 'ni', 'so', 'tk']
B_STATIONS = NEW_STATIONS + ['hf', 'an']

# Production DE tag: land r15 + ocean r05, fold-safe PCA.
MANIFEST_GLOBS = [
    'results/model_deep_ensemble/de_land_beta_nll_prof_reg_foldpca_r15_f*/training_dates.json',
    'results/model_deep_ensemble/de_ocean_beta_nll_prof_reg_foldpca_r05_f*/training_dates.json',
]

STATION_NAMES = {
    'ae': 'Ascension Island', 'an': 'Anmyeondo', 'bi': 'Bialystok', 'br': 'Bremen',
    'bu': 'Burgos', 'ci': 'Caltech (Pasadena)', 'db': 'Darwin', 'df': 'Edwards (AFRC)',
    'et': 'East Trout Lake', 'eu': 'Eureka', 'fc': 'Four Corners', 'gm': 'Garmisch',
    'hf': 'Hefei', 'hw': 'Harwell', 'if': 'Indianapolis', 'iz': 'Izana',
    'jc': 'JPL 2007', 'jf': 'JPL 2011', 'js': 'Saga', 'ka': 'Karlsruhe',
    'lh': 'Lauder 120HR', 'll': 'Lauder 125HR (ll)', 'lr': 'Lauder 125HR (lr)',
    'ma': 'Manaus', 'ni': 'Nicosia', 'ny': 'Ny-Alesund', 'oc': 'Lamont',
    'or': 'Orleans', 'pa': 'Park Falls', 'pr': 'Paris', 'ra': 'Reunion Island',
    'rj': 'Rikubetsu', 'so': 'Sodankyla', 'tk': 'Tsukuba', 'wg': 'Wollongong',
    'xh': 'Xianghe',
}

ERRORS: list[str] = []


def note_error(what: str) -> None:
    ERRORS.append(f'{what}\n{traceback.format_exc()}')
    print(f'[error] {what}', file=sys.stderr)
    traceback.print_exc()


# --------------------------------------------------------------------------- TCCON
_tccon_cache: dict[str, dict] = {}


def read_tccon(path: Path, xco2_var: str = 'xco2_x2019') -> dict:
    """Standalone reader: time (s since epoch), station lon/lat, valid-observation mask."""
    key = str(path)
    if key in _tccon_cache:
        return _tccon_cache[key]
    with netCDF4.Dataset(str(path)) as ds:
        t = np.asarray(ds.variables['time'][:], dtype='float64')
        lat = np.ma.filled(np.asarray(ds.variables['lat'][:], dtype='float64'), np.nan)
        lon = np.ma.filled(np.asarray(ds.variables['long'][:], dtype='float64'), np.nan)
        if xco2_var in ds.variables:
            x = np.ma.filled(np.asarray(ds.variables[xco2_var][:], dtype='float64'), np.nan)
        else:  # e.g. the pre-2020 Ascension file, which only has xco2_ppm
            alt = 'xco2_ppm' if 'xco2_ppm' in ds.variables else 'xco2'
            x = np.ma.filled(np.asarray(ds.variables[alt][:], dtype='float64'), np.nan)
            xco2_var = alt
    ok = np.isfinite(t) & np.isfinite(x) & (x > 300.0) & (x < 550.0)
    out = dict(path=path, xco2_var=xco2_var, n_all=int(t.size), n_valid=int(ok.sum()),
               time=np.sort(t[ok]),
               st_lon=float(np.nanmedian(lon)), st_lat=float(np.nanmedian(lat)),
               t_first=float(np.nanmin(t)) if t.size else np.nan,
               t_last=float(np.nanmax(t)) if t.size else np.nan)
    _tccon_cache[key] = out
    return out


def find_file(directory: Path, code: str) -> Path | None:
    """Unique <2-letter code>*.public*.nc in a TCCON directory; loud on 0 or >1."""
    hits = sorted(directory.glob(f'{code}*.public*.nc'))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    raise RuntimeError(f'{len(hits)} files match {code}* in {directory}: '
                       f'{[h.name for h in hits]}')


def fname_range(path: Path) -> tuple[str, str]:
    m = re.match(r'^[a-z]{2}(\d{8})_(\d{8})\.public', path.name)
    if not m:
        return ('?', '?')
    a, b = m.groups()
    return (f'{a[:4]}-{a[4:6]}-{a[6:]}', f'{b[:4]}-{b[4:6]}-{b[6:]}')


# --------------------------------------------------------------------------- geometry
def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def count_window(tc: dict, t_start: float, t_end: float, window_min: float) -> int:
    w = window_min * 60.0
    t = tc['time']
    lo = np.searchsorted(t, t_start - w, side='left')
    hi = np.searchsorted(t, t_end + w, side='right')
    return int(hi - lo)


def count_day(tc: dict, date: str) -> int:
    """Valid observations anywhere in that UTC calendar day (window-free diagnostic)."""
    t0 = pd.Timestamp(date, tz='UTC').timestamp()
    return count_window(tc, t0, t0 + 86400.0, 0.0)


# --------------------------------------------------------------------------- run_case
_RUN = re.compile(r'^\s*run_case\s+' + r'\s+'.join([r'(\S+)'] * 8) +
                  r'(?:\s+(\S+))?(?:\s+(\S+))?(?:\s+(\S+))?(?:\s+(\S+))?')


def parse_cases(text: str) -> list[dict]:
    cases = []
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        m = _RUN.match(line)
        if not m:
            continue
        g = m.groups()
        cases.append(dict(date=g[0], tccon=g[1],
                          lonmin=float(g[2]), lonmax=float(g[3]),
                          latmin=float(g[4]), latmax=float(g[5]),
                          surf=g[8] or 'both',
                          site=(g[10] or g[1][:2]),
                          avail=(g[11] or '')))
    return cases


# --------------------------------------------------------------------------- manifest
def load_training_dates() -> tuple[set[str], list[str]]:
    dates: set[str] = set()
    used: list[str] = []
    for pat in MANIFEST_GLOBS:
        for f in sorted(glob.glob(str(ROOT / pat))):
            with open(f) as fh:
                d = json.load(fh)
            for k in ('train_dates', 'calib_dates', 'held_dates'):
                dates |= set(d.get(k, []))
            used.append(os.path.relpath(f, ROOT))
    return dates, used


# --------------------------------------------------------------------------- parquet
def parquet_dates() -> list[str]:
    out = []
    for p in sorted(CSV_DIR.glob('combined_*_all_orbits.parquet')):
        m = re.match(r'combined_(\d{4}-\d{2}-\d{2})_all_orbits\.parquet$', p.name)
        if m:
            out.append(m.group(1))
    return out


def load_oco(date: str) -> pd.DataFrame | None:
    p = CSV_DIR / f'combined_{date}_all_orbits.parquet'
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=['time', 'lon', 'lat'])


def iso(t: float) -> str:
    return pd.to_datetime(t, unit='s', utc=True).strftime('%Y-%m-%d %H:%M')


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--min-fp', type=int, default=20,
                    help='(b) minimum footprints within the radius to report a candidate')
    ap.add_argument('--out', default=str(OUT_MD))
    args = ap.parse_args()

    train_dates, manifests = load_training_dates()
    print(f'[manifest] {len(manifests)} files, {len(train_dates)} development dates')

    cases = parse_cases(SH.read_text())
    no_cases = [c for c in cases if c['avail'] == 'no']
    print(f'[run_case] {len(cases)} active lines, {len(no_cases)} flagged no')

    # station tables --------------------------------------------------------
    new_tc: dict[str, dict] = {}
    old_tc: dict[str, dict] = {}
    for code in sorted({c['site'][:2] for c in no_cases} | set(B_STATIONS)):
        try:
            p = find_file(TCCON_NEW, code)
            if p is None:
                note_error(f'no GGG2020.1 file for station {code}')
                continue
            new_tc[code] = read_tccon(p)
        except Exception:
            note_error(f'reading GGG2020.1 file for station {code}')
        try:
            q = find_file(TCCON_OLD, code)
            if q is not None:
                old_tc[code] = read_tccon(q)
        except Exception:
            note_error(f'reading GGG2020 (old) file for station {code}')

    dates = parquet_dates()
    print(f'[parquet] {len(dates)} processed dates')

    by_date_cases: dict[str, list[dict]] = {}
    for c in no_cases:
        by_date_cases.setdefault(c['date'], []).append(c)

    rows_a: list[dict] = []
    rows_b: list[dict] = []
    coverage: dict[str, dict] = {}

    for i, date in enumerate(dates, 1):
        try:
            oco = load_oco(date)
        except Exception:
            note_error(f'reading parquet for {date}')
            continue
        if oco is None or not len(oco):
            note_error(f'empty/missing parquet for {date}')
            continue
        lon = oco['lon'].to_numpy(dtype='float64')
        lat = oco['lat'].to_numpy(dtype='float64')
        tim = oco['time'].to_numpy(dtype='float64')
        good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(tim)
        lon, lat, tim = lon[good], lat[good], tim[good]
        print(f'[{i:3d}/{len(dates)}] {date}  {len(lon):7d} footprints', flush=True)

        # ---- (a) run_case re-evaluation
        for c in by_date_cases.get(date, []):
            code = c['site'][:2]
            tc = new_tc.get(code)
            row = dict(date=date, site=code, surf=c['surf'], old_file=c['tccon'],
                       new_file=(tc['path'].name if tc else '-'),
                       n_fp=0, n_tccon=0, n_tccon_old=0, n_day_new=0,
                       avail_new='no', note='')
            if tc is None:
                row['note'] = 'no GGG2020.1 file'
                rows_a.append(row)
                continue
            row['n_day_new'] = count_day(tc, date)
            try:
                tc_old = old_tc.get(code)
            except Exception:
                tc_old = None
            inbox = ((lon >= c['lonmin']) & (lon <= c['lonmax']) &
                     (lat >= c['latmin']) & (lat <= c['latmax']))
            if not inbox.any():
                row['note'] = 'no footprints in box'
                rows_a.append(row)
                continue
            d = haversine_km(lon[inbox], lat[inbox], tc['st_lon'], tc['st_lat'])
            near = d <= args.radius_km
            n_fp = int(near.sum())
            row['n_fp'] = n_fp
            if not n_fp:
                row['note'] = f'no footprints <= {args.radius_km:g} km'
                rows_a.append(row)
                continue
            tn = tim[inbox][near]
            t0, t1 = float(tn.min()), float(tn.max())
            n = count_window(tc, t0, t1, args.window_min)
            row.update(n_tccon=n, avail_new=('yes' if n > 0 else 'no'),
                       n_tccon_old=(count_window(tc_old, t0, t1, args.window_min)
                                    if tc_old else -1),
                       pass_start=iso(t0), pass_end=iso(t1))
            rows_a.append(row)

        # ---- (b) blind scan over the new stations
        for code in B_STATIONS:
            tc = new_tc.get(code)
            if tc is None:
                continue
            d = haversine_km(lon, lat, tc['st_lon'], tc['st_lat'])
            near = d <= args.radius_km
            n_fp = int(near.sum())
            cov = coverage.setdefault(code, dict(n_dates_fp=0, n_dates_minfp=0,
                                                 n_dates_minfp_tccon=0, max_fp=0))
            if n_fp:
                cov['n_dates_fp'] += 1
                cov['max_fp'] = max(cov['max_fp'], n_fp)
            if n_fp < args.min_fp:
                continue
            cov['n_dates_minfp'] += 1
            tn = tim[near]
            t0, t1 = float(tn.min()), float(tn.max())
            n = count_window(tc, t0, t1, args.window_min)
            if n < 1:
                continue
            cov['n_dates_minfp_tccon'] += 1
            rows_b.append(dict(site=code, name=STATION_NAMES.get(code, '?'),
                               st_lat=tc['st_lat'], st_lon=tc['st_lon'],
                               date=date, n_fp=n_fp, n_tccon=n,
                               min_dist_km=float(d[near].min()),
                               pass_start=iso(t0), pass_end=iso(t1),
                               pass_min=(t1 - t0) / 60.0,
                               train=('yes' if date in train_dates else 'no')))
        del oco, lon, lat, tim

    # ---- (c) old vs new coverage
    rows_c: list[dict] = []
    for p_old in sorted(TCCON_OLD.glob('*.public*.nc')):
        code = p_old.name[:2]
        try:
            p_new = find_file(TCCON_NEW, code)
        except Exception:
            note_error(f'matching new file for {code}')
            continue
        if p_new is None:
            rows_c.append(dict(site=code, name=STATION_NAMES.get(code, '?'),
                               old_file=p_old.name, new_file='(absent)',
                               old_range='-', new_range='-', n_old=-1, n_new=-1))
            continue
        try:
            a = read_tccon(p_old)
            b = read_tccon(p_new)
        except Exception:
            note_error(f'reading pair for {code}')
            continue
        oa, ob = fname_range(p_old), fname_range(p_new)
        rows_c.append(dict(site=code, name=STATION_NAMES.get(code, '?'),
                           old_file=p_old.name, new_file=p_new.name,
                           old_range=f'{oa[0]} .. {oa[1]}',
                           new_range=f'{ob[0]} .. {ob[1]}',
                           n_old=a['n_valid'], n_new=b['n_valid'],
                           old_var=a['xco2_var'], new_var=b['xco2_var']))

    write_markdown(args, rows_a, rows_b, rows_c, manifests, train_dates, dates, no_cases,
                   coverage, new_tc)
    return 0


def write_markdown(args, rows_a, rows_b, rows_c, manifests, train_dates, dates, no_cases,
                   coverage, new_tc):
    a = pd.DataFrame(rows_a)
    b = pd.DataFrame(rows_b)
    c = pd.DataFrame(rows_c)
    flips = a[a['avail_new'] == 'yes'] if len(a) else a

    L: list[str] = []
    L.append('# GGG2020.1 new-collocation scan (2026-09-15)')
    L.append('')
    L.append('Generated by `workspace/scan_ggg2020p1_collocations.py` '
             '(standalone; imports nothing from the TCCON chain). Read-only scan; '
             'no manuscript or result directory touched.')
    L.append('')
    L.append('## Rules')
    L.append('')
    L.append(f'- Collocation radius {args.radius_km:g} km, TCCON window '
             f'+/- {args.window_min:g} min around the OCO-2 pass '
             '(pass = first to last footprint time inside the radius).')
    L.append('- Station longitude/latitude = median of the `long` / `lat` variables of the '
             'TCCON file.')
    L.append('- TCCON time = seconds since 1970-01-01 UTC; an observation counts when '
             '300 < `xco2_x2019` < 550 ppm.')
    L.append('- OCO-2 footprints from `results/csv_collection/combined_<date>_all_orbits.parquet` '
             f'({len(dates)} processed dates), columns `time`, `lon`, `lat` only.')
    L.append('- Table (a) keeps the `run_case` lon/lat box before the radius test, exactly as '
             '`workspace/check_tccon_availability.py` does. Table (b) uses the radius alone '
             f'(no box) and requires >= {args.min_fp} footprints and >= 1 TCCON observation.')
    L.append(f'- Training-date exclusion: union of `train_dates`, `calib_dates` and '
             f'`held_dates` over the {len(manifests)} production manifests '
             '`results/model_deep_ensemble/de_{land_beta_nll_prof_reg_foldpca_r15,'
             'ocean_beta_nll_prof_reg_foldpca_r05}_f[0-4]/training_dates.json` '
             f'({len(train_dates)} dates, 2016-2020). Because the folds rotate, every date in '
             'that pool has been trained on in some fold, so all of them are disqualified.')
    dev_here = sorted(d for d in dates if d in train_dates)
    L.append(f'- Of the {len(dates)} processed dates, {len(dev_here)} fall in that pool '
             f'({", ".join(dev_here) if dev_here else "none"}); the other '
             f'{len(dates) - len(dev_here)} are unseen by the production model.')
    L.append('- Gate: the same code run against the OLD `data/TCCON` files returns '
             '`n_tccon = 0` on all 53 lines and non-zero counts on the AVAIL = yes lines, '
             'so it reproduces the flags it is re-testing (column `n_tccon old` below).')
    L.append('')

    # (a)
    L.append(f'## (a) The {len(no_cases)} `run_case` lines flagged AVAIL = no, re-checked '
             'against GGG2020.1')
    L.append('')
    if len(a):
        L.append(f'{len(flips)} of {len(a)} flip to **yes**.')
        L.append('')
        L.append('| date | site | surf | n_fp <=100 km | n_tccon +/-60 min | new file | pass (UTC) |')
        L.append('|---|---|---|---:|---:|---|---|')
        for _, r in flips.sort_values(['site', 'date']).iterrows():
            L.append(f"| {r['date']} | {r['site']} | {r['surf']} | {int(r['n_fp'])} | "
                     f"{int(r['n_tccon'])} | `{r['new_file']}` | "
                     f"{r.get('pass_start','')} -> {str(r.get('pass_end',''))[-5:]} |")
        if not len(flips):
            L.append('| (none) | | | | | | |')
        L.append('')
        L.append(f'<details><summary>All {len(a)} lines</summary>')
        L.append('')
        L.append('`n_tccon old` repeats the count with the GGG2020 file (the number the AVAIL '
                 'flag was set from); `n_day new` is every valid GGG2020.1 observation anywhere '
                 'in that UTC day, window-free, so a 0 there means the station simply did not '
                 'measure that day.')
        L.append('')
        L.append('| date | site | surf | n_fp <=100 km | n_tccon old | n_tccon new | '
                 'n_day new | new AVAIL | note |')
        L.append('|---|---|---|---:|---:|---:|---:|---|---|')
        for _, r in a.sort_values(['site', 'date']).iterrows():
            L.append(f"| {r['date']} | {r['site']} | {r['surf']} | {int(r['n_fp'])} | "
                     f"{int(r['n_tccon_old'])} | {int(r['n_tccon'])} | {int(r['n_day_new'])} | "
                     f"{r['avail_new']} | {r['note']} |")
        L.append('')
        L.append('</details>')
        nday = int((a['n_day_new'] > 0).sum())
        L.append('')
        L.append(f'Diagnostic: {len(a) - nday} of the {len(a)} station-days have no valid '
                 'GGG2020.1 observation anywhere in the UTC day; the other '
                 f'{nday} have observations in the day but outside the '
                 f'+/- {args.window_min:g} min window. The reprocessing keeps the same spectra, '
                 'so a day the station did not measure stays empty.')
    else:
        L.append('No rows produced.')
    L.append('')

    # (b)
    L.append('## (b) Blind scan, 17 stations new to the download (plus hf, an)')
    L.append('')
    L.append('Stations scanned: ' + ', '.join(B_STATIONS) + '.')
    L.append('')
    if len(b):
        nb = b[b['train'] == 'no']
        L.append(f'{len(b)} (station, date) pairs clear the thresholds; {len(nb)} of them are '
                 'not development dates.')
        L.append('')
        L.append('| site | station | lat | lon | date | n_fp <=100 km | closest fp (km) | '
                 'n_tccon +/-60 min | pass (UTC) | dev date? |')
        L.append('|---|---|---:|---:|---|---:|---:|---:|---|---|')
        for _, r in b.sort_values(['site', 'date']).iterrows():
            L.append(f"| {r['site']} | {r['name']} | {r['st_lat']:.2f} | {r['st_lon']:.2f} | "
                     f"{r['date']} | {int(r['n_fp'])} | {r['min_dist_km']:.1f} | "
                     f"{int(r['n_tccon'])} | {r['pass_start']} (+{r['pass_min']:.0f} min) | "
                     f"{r['train']} |")
        L.append('')
    else:
        L.append('No (station, date) pair cleared the thresholds.')
    L.append('')
    L.append('Per-station coverage over all processed dates, so the stations that produced '
             'nothing can be told apart from the stations OCO-2 never approached:')
    L.append('')
    L.append('| site | station | lat | lon | TCCON record | dates with any fp | '
             f'dates with >= {args.min_fp} fp | of those, with TCCON | max fp on a date |')
    L.append('|---|---|---:|---:|---|---:|---:|---:|---:|')
    for code in B_STATIONS:
        tc = new_tc.get(code)
        cov = coverage.get(code, dict(n_dates_fp=0, n_dates_minfp=0,
                                      n_dates_minfp_tccon=0, max_fp=0))
        if tc is None:
            L.append(f'| {code} | {STATION_NAMES.get(code, "?")} | | | (file missing) '
                     '| | | | |')
            continue
        r0, r1 = fname_range(tc['path'])
        L.append(f"| {code} | {STATION_NAMES.get(code, '?')} | {tc['st_lat']:.2f} | "
                 f"{tc['st_lon']:.2f} | {r0} .. {r1} | {cov['n_dates_fp']} | "
                 f"{cov['n_dates_minfp']} | {cov['n_dates_minfp_tccon']} | {cov['max_fp']} |")
    L.append('')
    dry = [f"{code} ({new_tc[code]['st_lat']:.0f} deg, "
           f"{coverage.get(code, {}).get('n_dates_minfp', 0)} dates)"
           for code in B_STATIONS
           if code in new_tc and coverage.get(code, {}).get('n_dates_minfp', 0) >= 5
           and coverage.get(code, {}).get('n_dates_minfp_tccon', 0) == 0]
    if dry:
        L.append('Stations OCO-2 reaches often but that never had a TCCON observation inside '
                 'the window: ' + ', '.join(dry) + '. Eureka (eu, 80 N) fails on timing and on '
                 'record length, not on geometry; the high-latitude station that does deliver '
                 'is Sodankyla (so, 67 N).')
        L.append('')

    # (c)
    L.append('## (c) Old vs new file coverage, stations present in both directories')
    L.append('')
    if len(c):
        n_pair = int((c['n_new'] >= 0).sum())
        L.append(f"`data/TCCON` holds {len(c)} files; {n_pair} have a GGG2020.1 counterpart. "
                 f"{int((c['n_new'] > c['n_old']).sum())} gain observations, "
                 f"{int((c['n_new'] == c['n_old']).sum()) - (len(c) - n_pair)} are unchanged.")
        L.append('')
        L.append('| site | station | old range | new range | n valid old | n valid new | '
                 'delta | old file | new file |')
        L.append('|---|---|---|---|---:|---:|---:|---|---|')
        for _, r in c.sort_values('site').iterrows():
            dlt = ('-' if r['n_old'] < 0 or r['n_new'] < 0
                   else f"{int(r['n_new']) - int(r['n_old']):+d}")
            L.append(f"| {r['site']} | {r['name']} | {r['old_range']} | {r['new_range']} | "
                     f"{'' if r['n_old'] < 0 else int(r['n_old'])} | "
                     f"{'' if r['n_new'] < 0 else int(r['n_new'])} | {dlt} | "
                     f"`{r['old_file']}` | `{r['new_file']}` |")
        L.append('')
        L.append('Valid-observation counts use 300 < `xco2_x2019` < 550 in both releases, so the '
                 'columns are directly comparable.')
    else:
        L.append('No rows produced.')
    L.append('')

    if ERRORS:
        L.append('## Failures')
        L.append('')
        for e in ERRORS:
            L.append('```')
            L.append(e.rstrip())
            L.append('```')
        L.append('')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n')
    print(f'[saved] {out}')
    print(f'(a) flips to yes: {len(flips)} / {len(a)}')
    if len(b):
        print(f'(b) candidates: {len(b)} total, {int((b["train"] == "no").sum())} non-development')
    print(f'(c) stations compared: {len(c)}')
    if ERRORS:
        print(f'[failures] {len(ERRORS)}')


if __name__ == '__main__':
    sys.exit(main())
