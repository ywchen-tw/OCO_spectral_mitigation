"""Identify which Lite version (11.1r/11.2r/11.3r) each drift-case parquet used.

For each date, fetch an OPeNDAP subset of the candidate Lite collections and
compare xco2 to the parquet xco2_bc on lat/lon-matched soundings.
"""
import json, subprocess, sys, urllib.request, urllib.parse
from pathlib import Path
import h5py, numpy as np, pandas as pd

S = Path(sys.argv[1]); R = Path(sys.argv[2]); dates = sys.argv[3:]
CID = {'11.1r': None, '11.2r': 'C2912085112-GES_DISC', '11.3r': 'C4203038017-GES_DISC'}
cmr = json.load(urllib.request.urlopen(
    'https://cmr.earthdata.nasa.gov/search/collections.json?short_name=OCO2_L2_Lite_FP&version=11.1r'))
CID['11.1r'] = cmr['feed']['entry'][0]['id'] if cmr['feed']['entry'] else None

def granule(cid, date):
    q = urllib.parse.urlencode({'collection_concept_id': cid, 'temporal': f'{date}T00:00:00Z,{date}T23:59:59Z', 'page_size': 5})
    for e in json.load(urllib.request.urlopen(f'https://cmr.earthdata.nasa.gov/search/granules.json?{q}'))['feed']['entry']:
        if e['time_start'][:10] == date:
            return e['title']
    return None

def fetch(ver, date):
    out = S / f'{date}_{ver}.nc4'
    if out.exists():
        return out
    cid = CID[ver]; title = cid and granule(cid, date)
    if not title:
        return None
    zlo = ';/Retrieval/xco2_zlo_bias' if ver == '11.3r' else ''
    url = (f'https://opendap.earthdata.nasa.gov/collections/{cid}/granules/{urllib.parse.quote(title)}'
           f'.dap.nc4?dap4.ce=/latitude;/longitude;/xco2{zlo}')
    subprocess.run(['curl', '-s', '-n', '-L', '-c', str(S / 'cj'), '-b', str(S / 'cj'), '-o', str(out), url], check=True)
    with open(out, 'rb') as f:
        if f.read(4) != b'\x89HDF':
            out.unlink(); return None
    return out

rows = []
for date in dates:
    p = pd.read_parquet(R / f'results/csv_collection/combined_{date}_all_orbits.parquet', columns=['lat', 'lon', 'xco2_bc'])
    p['k'] = list(zip(np.round(p.lat.astype('f4'), 4), np.round(p.lon.astype('f4'), 4)))
    res = {'date': date, 'n_parquet': len(p)}
    vers = ['11.3r', '11.2r'] if date >= '2024' else ['11.2r', '11.1r']
    for ver in vers:
        f = fetch(ver, date)
        if f is None:
            res[f'med_{ver}'] = np.nan; continue
        with h5py.File(f) as h:
            d = pd.DataFrame({'lat': h['latitude'][:], 'lon': h['longitude'][:], 'x': h['xco2'][:]})
            if ver == '11.3r':
                d['zlo'] = h['Retrieval/xco2_zlo_bias'][:]
        d['k'] = list(zip(np.round(d.lat.astype('f4'), 4), np.round(d.lon.astype('f4'), 4)))
        m = p.merge(d.drop(columns=['lat', 'lon']), on='k')
        res[f'n_{ver}'] = len(m)
        res[f'med_{ver}'] = float(np.median(np.abs(m.xco2_bc - m.x)))
        if ver == '11.3r':
            res['zlo_mean'] = float(m.zlo.mean())
    ok = {v: res.get(f'med_{v}') for v in vers}
    res['lite_version'] = next((v for v in vers if ok[v] is not None and ok[v] < 1e-3), 'unmatched')
    print(res, flush=True); rows.append(res)
pd.DataFrame(rows).to_csv(S / 'lite_versions.csv', index=False)
