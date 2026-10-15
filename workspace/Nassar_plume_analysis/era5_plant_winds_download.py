#!/usr/bin/env python
"""Download ERA5 winds around the US power plants for us_plant_emissions.py.

Per overpass, two small CDS requests (0.5 deg box, the three hours bracketing
the overpass):
  pressure levels 1000-850 hPa : u, v, geopotential
  single levels               : surface geopotential, 100 m u/v, surface
                                pressure, boundary-layer height
us_plant_emissions.py interpolates them to heights above ground (e.g. 300 m).

Needs cdsapi and ~/.cdsapirc (run with the era5 env).

Usage:  python workspace/Nassar_plume_analysis/era5_plant_winds_download.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

DEFAULT_OUT = Path("results/model_comparison/deep_ensemble/"
                   "de_beta_nll_prof_reg_foldpca_o05l15_m5/nassar_plumes/us_emissions/_cache/era5_cds")
# (tag, lat, lon, date, UTC hours bracketing the overpass)
REQUESTS = (
    ("westar", 39.28556, -96.11701, "2015-12-04", ("18:00", "19:00", "20:00")),
    ("ghent", 38.74753, -85.03433, "2015-08-13", ("17:00", "18:00", "19:00")),
    ("westar", 39.28556, -96.11701, "2023-03-13", ("18:00", "19:00", "20:00")),
    ("westar", 39.28556, -96.11701, "2023-06-26", ("18:00", "19:00", "20:00")),
    ("ghent", 38.74753, -85.03433, "2024-04-15", ("17:00", "18:00", "19:00")),
)
LEVELS = ["850", "875", "900", "925", "950", "975", "1000"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False)
    for tag, lat, lon, date, hours in REQUESTS:
        y, m, d = date.split("-")
        common = dict(product_type=["reanalysis"], year=[y], month=[m], day=[d],
                      time=list(hours), data_format="netcdf", download_format="unarchived",
                      area=[lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25])
        for kind, ds, extra in (
                ("plev", "reanalysis-era5-pressure-levels",
                 dict(variable=["u_component_of_wind", "v_component_of_wind", "geopotential"],
                      pressure_level=LEVELS)),
                ("sfc", "reanalysis-era5-single-levels",
                 dict(variable=["geopotential", "100m_u_component_of_wind",
                                "100m_v_component_of_wind", "surface_pressure",
                                "boundary_layer_height"]))):
            out = args.out_dir / f"era5_{kind}_{tag}_{date}.nc"
            if out.exists() and out.stat().st_size > 0:
                print(f"  have {out.name}")
                continue
            print(f"  requesting {out.name} ...", flush=True)
            client.retrieve(ds, {**common, **extra}).download(str(out))
            print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
