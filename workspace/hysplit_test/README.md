# HYSPLIT practice (not for the manuscript)

Goal: run a NOAA HYSPLIT dispersion for one US power-plant overpass and compare the
modelled column CO2 enhancement with the OCO-2 transect before/after the DE correction.
The emission-vs-enhancement analysis itself lives in
`results/.../us_coal_plumes/screen/` and does not depend on HYSPLIT.

## 1. Install (macOS, unregistered trial build)
1. Open https://www.ready.noaa.gov/HYSPLIT_machysplit.php, accept the use agreement, download the .dmg.
2. Copy `hysplit/` (with `exec/`) to `~/hysplit` or into `workspace/hysplit_test/hysplit_install/` (git-ignored).
   The trial build runs trajectories and dispersion with ARCHIVED meteorology (all we need).
3. Alternative on CURC: the Linux trial tarball from https://www.ready.noaa.gov/HYSPLIT_linuxtrial.php.

## 2. Meteorology (already fetched by script)
`met/20190914_nam12` — NAM 12 km ARL file (446 MB) from https://www.ready.noaa.gov/data/archives/nam12/
Higher resolution option: HRRR 3 km ARL, 6-h chunks of 3.4 GB, e.g. `.../archives/hrrr/20190914_18-23_hrrr` (needs the 12-17 chunk too for a 15 UTC start).

## 3. Run
```
cd runs/prairie_state_2019-09-14
./run.sh ~/hysplit/exec          # trajectory test, then dispersion, then con2asc
conda run -n ml310 python postprocess.py   # samples the column grid at the OCO-2 footprints, makes hysplit_vs_oco2.png
```
CONTROL.disp: release 2019-09-14 15-21 UTC at 213 m AGL, 853,200 kg/h (EPA CAMPD 12-14 LST = 237 kg/s),
concentration grid 0.01 deg over +-0.5 deg, one layer 0-5000 m (column), 30-min average 19:00-19:30 UTC
(overpass 19:08). Column kg/m2 = conc * 5000 m; ppm = column / 0.0154.

## 4. Other cases
Copy the run directory and edit start time, lat/lon, stack height, emission, met file. Pre-2019 cases have
no HRRR ARL archive; use `nam12` (2010+) or `gdas1`. Winds/emissions per case:
`results/.../us_coal_plumes/screen/final_case_table.csv`, `plume_height_wind.csv`, `campd_hourly.csv`.
