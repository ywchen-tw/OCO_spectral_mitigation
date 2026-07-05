# Shipborne EM27/SUN ocean-glint XCO2 comparison

Independent **open-ocean anchor** for the near-cloud XCO2 correction — the ocean
counterpart to the land TCCON stations (which never see open water). Two ship-going
EM27/SUN cruises measured column XCO2 over the Pacific during our OCO-2 data years:

| Campaign | Vessel | Dates | Region | raw .tab |
|---|---|---|---|---|
| MORE-2 (`so268`) | RV *Sonne* | 2019-06-04…29 | Pacific ~30°N (spans basin) | `data/Other/SO268-3_track_XCO2_XCH4_XCO.tab` |
| MR21-01 (`mr2101`) | RV *Mirai* | 2021-02-13…03-22 | Western N Pacific | `data/Other/Hanft_2021_XCO2_XCH4_XCO.tab` |

Both are TCCON-Karlsruhe-tied EM27/SUN columns (Knapp et al. 2021 ESSD;
Butz et al. 2022 Front. Remote Sens.).

> **Why a separate module (not the TCCON launcher).** A ship is a *moving* platform
> with no fixed station, so it does not fit `run_case` in
> `curc_shell_blanca_plot_corr_xco2_deepens.sh` and would pollute the TCCON aggregate
> reports. Same rationale and layout as `workspace/ATom_analysis/`.

## Stage 1 — overlap screen  ·  `ship_lite_collocate.py`

Before any pipeline compute, screen each ship day for OCO-2 **ocean-glint, good-QF**
soundings near the track using only the global daily **L2 Lite** granules (CMR +
Earthdata auth; cached in `data/Other/lite_cache/`, shared with the ATom scan).
Matches the *moving* track in space **and** time; counts soundings within
{50,100,250} km × ±2 h.

```
python workspace/Ship_analysis/ship_lite_collocate.py              # download (cached) + screen
python workspace/Ship_analysis/ship_lite_collocate.py --no-download # cached Lite only
```

Outputs → `output/ship_oco2_collocation.csv` (all 47 ship days) and
`output/process_dates.txt` (days clearing 100 km / ±2 h / good-QF).

**Outcome (47 ship days):** only **4** clear the strict gate —
`2019-06-09 2019-06-14 2019-06-22` (MORE-2) and `2021-03-15` (MR21-01).
Binding limit: OCO-2's ~13:30 LT overpass vs the ~10 km swath crossing a moving ship.
`2021-03-18` is a near-miss (nearest pass ~212 km). **2019-06-24…27 have no OCO-2 Lite
granule at all** (satellite outage). Full table + the raw numbers:
`data/Other/ship_validation_overlap.md` and `data/Other/ship_overlap_results.csv`.

## Stage 1.5 — footprint collocation + box  ·  `ship_footprint_collocate.py`

Collocates the ship track against the **processed** OCO-2 parquets
(`results/csv_collection/combined_<date>_all_orbits.parquet`), ocean-glint good-QF
(`sfc_type==0 & xco2_qf==0`), 100 km / ±2 h. Reports per date the footprint bbox,
VMIN/VMAX, **cloud-distance coverage** (the near-cloud second filter), and OCO-2 vs
ship XCO2 medians. Prints ready-to-paste `ship_case` lines.

```
python workspace/Ship_analysis/ship_footprint_collocate.py [--radius-km 100 --window-min 120]
```

**Near-cloud coverage matters here:** 2019-06-09 is 494/494 footprints ≤10 km from
cloud (a true near-cloud test), while 2019-06-22 is essentially clear-sky (1/460) — a
clear-reference day. Read the comparison in that light.

## Stage 2 — correction + comparison  ·  `curc_shell_blanca_ship_deepens.sh` + `plot_ship_comparison.py`

Standalone Blanca runner. Per date it applies the **same production M=5 deep ensemble**
as the TCCON launcher — ocean **r05** profile+reg variant
(`de_ocean_beta_nll_prof_reg_r05_f*`) — via `build_deepens_plot_data.py`, then draws the
ship-native comparison figure (`plot_ship_comparison.py`): DeepEns-corrected map + ship
track, OCO-2-original-vs-corrected-vs-ship histogram, collocated cloud-distance panel,
and the ship XCO2 time series with the overpass shaded. Proper ship labels throughout —
no TCCON machinery. Map panels carry a **MODIS Aqua true-colour background** (NASA GIBS,
`--modis-auto`, same fetcher as `plot_corrected_xco2.py`) so footprints can be read
against the actual cloud field.

```
sbatch workspace/Ship_analysis/curc_shell_blanca_ship_deepens.sh     # submit from repo root
```

Output → `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/ship/combined_<date>_<ship>/`
(`plot_data.parquet` + `ship_comparison_<ship>_<date>.png`).

**Local sanity run (2019-06-22, ocean r05):** 644 footprints ≤100 km; ship median
411.22 ppm vs OCO-2 corrected 412.03 → Δ +0.81 ppm (original +0.77). On this clear-sky
day the correction barely moves the ocean bias — expected, since it targets near-cloud
anomalies; the near-cloud day 2019-06-09 is the real test.

## References
- Knapp et al. 2021, ESSD 13, 199–211 (MORE-2). PANGAEA 917240.
- Butz et al. 2022, Front. Remote Sens. 2, 775805 (MR21-01). PANGAEA 937933.
