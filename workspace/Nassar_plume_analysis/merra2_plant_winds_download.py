#!/usr/bin/env python
"""Download MERRA-2 winds at the US power plants for us_plant_emissions.py.

Nassar et al. (2021, 2022) take the wind at the plume midline (stack height +
250 m) as the vector average of ERA5 and MERRA-2 (0.5 x 0.625 deg, 3-h
average).  The 3-h average pressure-level product (M2T3NPASM) is not in CMR, so
this script pulls the 3-hourly instantaneous one (M2I3NPASM, inst3_3d_asm_Np:
U, V, H) at the grid point nearest each plant for the overpass day, 1000-700
hPa; us_plant_emissions.py interpolates it linearly in time to the overpass.

GES DISC's on-premises OPeNDAP is retired, so the granule is found through CMR
with earthaccess and read through its advertised (cloud) OPeNDAP endpoint,
falling back to a full-granule download.  Needs earthaccess, xarray and an
Earthdata login in ~/.netrc (run with the era5 env).

Usage:  python workspace/Nassar_plume_analysis/merra2_plant_winds_download.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

import earthaccess
import xarray as xr

DEFAULT_OUT = Path("results/model_comparison/deep_ensemble/"
                   "de_beta_nll_prof_reg_foldpca_o05l15_m5/nassar_plumes/us_emissions/_cache/merra2")
REQUESTS = (
    ("westar", 39.28556, -96.11701, "2015-12-04"),
    ("ghent", 38.74753, -85.03433, "2015-08-13"),
    ("westar", 39.28556, -96.11701, "2023-03-13"),
    ("westar", 39.28556, -96.11701, "2023-06-26"),
    ("ghent", 38.74753, -85.03433, "2024-04-15"),
)
SHORT_NAME, VERSION = "M2I3NPASM", "5.12.4"


def granule_for(date):
    d = dt.date.fromisoformat(date)
    res = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION,
                                  temporal=(d.isoformat(), (d + dt.timedelta(days=1)).isoformat()))
    tag = f".{d:%Y%m%d}."
    hits = [g for g in res if tag in g["umm"]["GranuleUR"]]
    if not hits:
        raise FileNotFoundError(f"no {SHORT_NAME} granule for {date}")
    return hits[0]


def opendap_url(granule):
    for entry in granule["umm"].get("RelatedUrls", []):
        url = entry.get("URL", "")
        if "opendap" in url.lower() and (entry.get("Subtype", "") == "OPENDAP DATA"
                                         or entry.get("Type", "") == "USE SERVICE API"):
            return url
    return None


def point_subset(ds, lat, lon):
    return (ds[["U", "V", "H"]].sel(lat=lat, lon=lon, method="nearest")
            .sel(lev=slice(1000, 700)).load())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="netrc")
    for tag, lat, lon, date in REQUESTS:
        out = args.out_dir / f"merra2_{tag}_{date}.nc"
        if out.exists() and out.stat().st_size > 0:
            print(f"  have {out.name}")
            continue
        g = granule_for(date)
        sub = None
        url = opendap_url(g)
        if url:
            try:
                store = xr.backends.PydapDataStore.open(
                    url, session=earthaccess.get_requests_https_session())
                sub = point_subset(xr.open_dataset(store), lat, lon)
                print(f"  {date} {tag}: OPeNDAP subset", flush=True)
            except Exception as exc:
                print(f"  {date} {tag}: OPeNDAP failed ({exc}); full granule", flush=True)
        if sub is None:
            with tempfile.TemporaryDirectory() as tmp:
                f = earthaccess.download([g], tmp, threads=1)[0]
                with xr.open_dataset(f) as full:
                    sub = point_subset(full, lat, lon)
        sub.attrs = {}                       # cloud OPeNDAP adds non-serializable attrs
        for v in sub.variables.values():
            v.attrs = {k: x for k, x in v.attrs.items() if isinstance(x, (str, int, float))}
        sub.to_netcdf(out)
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
