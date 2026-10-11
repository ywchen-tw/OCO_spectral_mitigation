"""check_tccon_availability.py — does each run_case have TCCON at the OCO pass time?

Mirrors the n_tccon logic in tccon_comparison_report.py: for each active run_case in
curc_shell_blanca_plot_corr_xco2_deepens.sh, load the input parquet (lon/lat/time),
keep footprints in the lon/lat box and ≤--radius-km of the TCCON station, then count
TCCON observations within ±--window-min of [pass start, pass end].  AVAIL = (count>0).

Uses only the INPUT parquet + TCCON file, so it does NOT depend on the plotting run.
Prints one line per case and writes a CSV; with --emit-flags it prints the yes/no token
in run_case column order for pasting into the script's 12th (TCCON_AVAIL) column.

Run: PYTHONPATH=src python workspace/check_tccon_availability.py
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plot_corrected_xco2 import load_tccon, _haversine_km, get_storage_dir

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get('CURC_DATA_ROOT')
                 or os.environ.get('OCO2_DATAROOT') or ROOT)
SH = ROOT / 'curc_shell_blanca_plot_corr_xco2_deepens.sh'
CSV_DIR = DATA_ROOT / 'results/csv_collection'

# run_case DATE TCCON LONMIN LONMAX LATMIN LATMAX VMIN VMAX [SURF] [POSTER] [SITE] [AVAIL]
_RUN = re.compile(r'^\s*run_case\s+' + r'\s+'.join([r'(\S+)'] * 8) +
                  r'(?:\s+(\S+))?(?:\s+(\S+))?(?:\s+(\S+))?(?:\s+(\S+))?')


def parse_cases(text):
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
                          surf=g[8] or 'both', site=g[10] or g[1][:2]))
    return cases


_tccon_cache = {}
def tccon_df(name):
    if name not in _tccon_cache:
        p = DATA_ROOT / 'data/TCCON' / name
        if not p.exists():
            p = get_storage_dir() / 'data/TCCON' / name
        _tccon_cache[name] = load_tccon(str(p)) if p.exists() else None
    return _tccon_cache[name]


def check(c, radius_km, window_min):
    pq = CSV_DIR / f"combined_{c['date']}_all_orbits.parquet"
    if not pq.exists():
        return dict(n_tccon=-1, avail='unknown', note='no input parquet')
    oco = pd.read_parquet(pq, columns=['lon', 'lat', 'time'])
    oco = oco[(oco['lon'] >= c['lonmin']) & (oco['lon'] <= c['lonmax']) &
              (oco['lat'] >= c['latmin']) & (oco['lat'] <= c['latmax'])]
    if len(oco) == 0:
        return dict(n_tccon=0, avail='no', note='no OCO footprints in box')
    tc = tccon_df(c['tccon'])
    if tc is None or not len(tc):
        return dict(n_tccon=0, avail='no', note='TCCON file missing/empty')
    st_lon, st_lat = float(tc['lon'].median()), float(tc['lat'].median())
    d = _haversine_km(oco['lon'].values, oco['lat'].values, st_lon, st_lat)
    near = oco[d <= radius_km]
    if len(near) == 0:
        return dict(n_tccon=0, avail='no', note=f'no footprints ≤{radius_km:g}km of station')
    ot = pd.to_datetime(near['time'], unit='s', utc=True, errors='coerce').dropna()
    if not len(ot):
        return dict(n_tccon=0, avail='no', note='no valid OCO times')
    w = pd.Timedelta(minutes=window_min)
    sub = tc[(tc['time'] >= ot.min() - w) & (tc['time'] <= ot.max() + w)]
    n = int(len(sub))
    return dict(n_tccon=n, avail=('yes' if n > 0 else 'no'),
                note=f'{len(near)} fp ≤{radius_km:g}km, ±{window_min:g}min')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius-km', type=float, default=100.0)
    ap.add_argument('--window-min', type=float, default=60.0)
    ap.add_argument('--emit-flags', action='store_true',
                    help="Print just 'DATE SITE AVAIL' tokens for the 12th script column.")
    ap.add_argument('--script', default=str(SH),
                    help="Shell script whose run_case lines to check (default: the "
                         "non-drift deepens script).")
    args = ap.parse_args()

    cases = parse_cases(Path(args.script).read_text())
    rows = []
    print(f"{'date':12s} {'site':4s} {'surf':5s} {'n_tccon':>7s} {'AVAIL':>7s}  note")
    print('-' * 70)
    for c in cases:
        r = check(c, args.radius_km, args.window_min)
        rows.append(dict(date=c['date'], site=c['site'], surf=c['surf'], **r))
        print(f"{c['date']:12s} {c['site']:4s} {c['surf']:5s} "
              f"{r['n_tccon']:>7d} {r['avail']:>7s}  {r['note']}")

    df = pd.DataFrame(rows)
    out = DATA_ROOT / 'results/model_comparison/tccon_availability.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_no = int((df['avail'] == 'no').sum()); n_unk = int((df['avail'] == 'unknown').sum())
    print('-' * 70)
    print(f"{len(df)} cases: {int((df['avail']=='yes').sum())} avail, {n_no} NO-TCCON, {n_unk} unknown")
    print(f"[saved] {out}")
    if args.emit_flags:
        print("\n# date site avail (no-TCCON cases only):")
        for _, r in df[df['avail'] != 'yes'].iterrows():
            print(f"#   {r['date']}  {r['site']}  -> {r['avail']}")


if __name__ == '__main__':
    main()
