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

Per-case figure panels: DeepEns-corrected map, **before-correction `xco2_bc` map** (same
colour scale, side-by-side), OCO-vs-corrected-vs-ship histogram with **μ and σ** (bias
title carries ±σ(OCO)⊕σ(ship)), and the ship time series. `plot_ship_summary.py` then
draws a cross-case summary (`ship_comparison_summary.png`) mirroring
`atom_pseudo_column_summary.png`: per-case bias dumbbell `xco2_bc → DE-corrected` and
bias-vs-cloud-distance, both with ±1σ error bars.

Output → `results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/ship/`:
per case `combined_<date>_<ship>/` (`plot_data.parquet` + `ship_comparison_<ship>_<date>.png`),
plus `ship_comparison_summary.png` + `.csv` at the top.

**Result across the 4 cases:** the correction collapses footprint scatter (mean σ
0.63→0.27 ppm) but leaves the absolute ocean offset (~+1 ppm) — it targets OCO-2's
*relative* near-cloud anomaly, so it tightens spread without removing an absolute bias.

**Local sanity run (2019-06-22, ocean r05):** 644 footprints ≤100 km; ship median
411.22 ppm vs OCO-2 corrected 412.03 → Δ +0.81 ppm (original +0.77). On this clear-sky
day the correction barely moves the ocean bias — expected, since it targets near-cloud
anomalies; the near-cloud day 2019-06-09 is the real test.

## References
- Knapp et al. 2021, ESSD 13, 199–211 (MORE-2). PANGAEA 917240.
- Butz et al. 2022, Front. Remote Sens. 2, 775805 (MR21-01). PANGAEA 937933.

## Interpreting the corrected-OCO − ship offset (added 2026-07-07)

Per-day residuals (OCO − ship, ppm; `ship_comparison_summary.csv`):

| date | n | cld_med (km) | resid `xco2_bc` | resid corrected | reading |
|---|--:|--:|--:|--:|---|
| 2019-06-09 | 586 | 2.2 | +1.09 | +1.13 | true near-cloud case |
| 2019-06-14 | 243 | 4.9 | +0.22 | +0.47 | mixed |
| 2019-06-22 | 644 | 41.3 | +0.77 | +0.81 | clear-sky control |
| 2021-03-15 | 21 | 2.9 | +1.89 | +2.23 | Mirai; n=21, winter, weak constraint |

**Key triangulation (2026-07-07, post AK-fix TCCON report):** ocean-glint
footprints near TCCON show the SAME offset in the same direction — signed
after-correction bias **+0.61 ± 0.87 ppm direct**, collapsing to
**+0.12 ± 1.21 ppm AK-harmonized** (20 cases, n_oco ≥ 20; land: −0.10 direct /
−0.43 AK). The ship clear-sky control (+0.81 direct) is statistically the same
number as the TCCON-ocean direct offset. So the ship gap is a general property
of *direct* ocean-glint comparisons of this product — not a ship-data artifact
and not created by the ML correction.

**Proposed causes, ranked (no code changes made — analysis only):**

1. **Missing AK/prior harmonization (~+0.5 ppm, dominant).** This module
   compares `xco2_bc`/corrected directly to EM27 XCO₂; no Rodgers–Connor
   adjustment. The TCCON-ocean evidence above implies harmonization would
   remove most of the offset. Precedent: Klappenbach et al. (2015, AMT)
   applied AK harmonization for exactly this shipborne-EM27-vs-OCO-2 setup.
2. **EM27 calibration vintage (±0.2–0.4 ppm).** Both cruises are tied to
   TCCON Karlsruhe under pre-GGG2020 processing (Knapp et al. 2021; Butz et
   al. 2022), while B11's absolute scale is anchored to GGG2020 (B11 DUG
   §4.2.3) — a scale-epoch mismatch (GGG2014→GGG2020 + X2007→X2019) of a few
   tenths ppm.
3. **Residual B11 ocean-glint regional bias (±0.3 ppm).** The ocean divisor is
   set globally (coastline crossings); regional/seasonal structure at the
   0.3–0.5 ppm level is documented (O'Dell et al. 2018; Taylor et al. 2023;
   Das et al. 2025) and is not removed by a global anchor.
4. **The ML correction slightly widens the direct gap (+0.04…+0.35 ppm) by
   design.** Near-cloud ocean μ < 0 (the near-cloud low anomaly), so
   corrected = bc − μ sits above bc. It repairs the *relative* near-cloud
   anomaly against the OCO clear-sky field; the clear-sky control proves the
   base offset predates the correction.
5. **Sampling.** Four days, one cruise dominating; the Mirai +2.2 (n=21,
   winter, high SZA) should be read as a weak constraint, not a trend.

**Cheap follow-up if a reviewer pushes:** harmonize the ship reference with
the (now wet/dry-fixed) `ak_harmonize.py` operator — the collocated parquets
already carry the flattened ak/pwf/prior columns; the EM27 side needs only a
γ-scaled prior proxy (PROFFAST priors, or bound the term using the OCO-side
operator alone). Expected shift is the ~+0.5 ppm seen at the TCCON ocean
subset, which would bring the clear-sky control near zero.
