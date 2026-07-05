# ATom ocean-glint pseudo-column comparison

Independent **in-situ ocean anchor** for the near-cloud XCO2 correction
(PROJECT_REVIEW_2026-07-03.md → M7-3, resolution item **#3**).

ATom flew four seasonal Pacific/Atlantic circuits of continuous 0.2–12 km vertical
profiles (2016–2018), overlapping our OCO-2 data years. The goal is to build XCO2
**pseudo-columns** from these profiles and compare them to co-located OCO-2 ocean-glint
soundings.

> **Priority note.** This is item **#3** of five in the ocean-validation plan. Items
> #1 (TCCON ocean-glint stratification — free, reuses `tccon_collocate.py`) and
> #2 (shipborne EM27/SUN columns) rank higher and are expected to *suffice*. ATom is
> a labor-intensive spot-check whose binding limit is the **scarcity of tight OCO-2
> glint coincidences** — treat any result as illustrative, not statistical. Pursue
> only if a reviewer explicitly wants an independent in-situ ocean anchor.

## Outputs location

All outputs live under `results/model_comparison/deep_ensemble/<MODEL_TAG>/atom/`
(`MODEL_TAG = de_beta_nll_prof_reg_o05l15_m5`), mirroring the TCCON comparison layout —
this dir is `$OUT` below. The scripts are in `workspace/ATom_analysis/`.

```
$OUT/
  combined_<date>_atom/           # per-case (like TCCON combined_<date>_<site>/)
    plot_data.parquet             #   DE-corrected XCO2 (curc_shell_blanca_atom_deepens.sh)
    atom_modis_<date>.png         #   MODIS Aqua overlay
  atom_merged/                    # Stage-1 merged ATom profiles + profile plots (input)
  _modis_tiles/                   # cached GIBS Aqua tiles
  atom_oco2_collocation.csv       # aggregate: Lite screen
  process_dates.txt               # aggregate: usability status per date
  atom_pseudo_column_results.csv  # aggregate: Stage 2/3 results (+ _summary.png)
```

## Data (raw)

- `data/Other/ATom_Picarro_Instrument_Data_1732/` — NOAA-Picarro CO2/CH4/CO, ICARTT
  FFI 1001, 1 Hz. **Position-free**: only `UTC_Start, CO2, CH4, CO`. WMO X2007 CO2 scale.
- `data/Other/ATom_nav_1613/` — `ATom{1..4}_flight_tracks.csv`, 10-s nav with
  lat/lon/alt and **measured static pressure + temperature** (MMS). Shares the Picarro
  time base (seconds since midnight UTC, no post-midnight rollover), so the join is direct.

## Stage 1 — `merge_atom_profiles.py`  ✅ done

Joins Picarro CO2 to nav (interpolated onto the 1-Hz CO2 timestamps) and segments each
flight into ascending/descending legs. Using the nav's *measured* pressure means no
altitude→pressure conversion is needed downstream.

```
python merge_atom_profiles.py                 # all flights
python merge_atom_profiles.py --date 20171001 --plot
```

Output → `output/atom_merged_<date>.parquet`, columns:
`time_utc_s, co2_ppm, lat, lon, alt_m, p_hpa, t_k, profile_id, leg_dir`.

**Run status (48 flights, 644 profiles, ~890k pts):**
- 4 flights dropped — CO2 fully flagged ("material interaction"): 20160726/29, 20160801, 20170111.
- 20170124 dropped — valid CO2 but no nav (ATom-2 Palmdale transit; nav covers science flights only).

Segmentation knobs at the top of the script: `MIN_LEG_SPAN_M=3000`,
`SMOOTH_WIN_S=60`, `PEAK_PROMINENCE_M=2000`.

## Stage 1.5 — OCO-2 coincidence screen  ✅ done → `atom_lite_collocate.py`

Before spending pipeline compute, screen each flight for OCO-2 **ocean-glint, good-QF**
soundings near the track. Downloads daily Lite granules (cached in
`data/Other/lite_cache/`, shared with the ship scan), filters
`operation_mode==1 & land_water_indicator==1 & xco2_quality_flag==0`, counts soundings
within {50,100,250} km × {2,6,24} h.

```
python atom_lite_collocate.py            # all flights -> output/atom_oco2_collocation.csv
python atom_lite_collocate.py --no-download   # cached Lite only
```

**Outcome (45 CO2-valid flights):** 9 dates meet the strict 100 km/±2 h ocean-glint gate,
13 at 100 km/±6 h, 21 at 250 km/same-day. Matches cluster in ATom-2/3/4; ATom-1 (2016)
has none. Binding limit: OCO-2's fixed ~13:30 LT overpass vs ATom's arbitrary local time.
`min_km`/`tgap_min` in the CSV is the spatially-closest sounding *ignoring time*; the
`n_*` counts require space **and** time — use the counts.

**Selected set (strict, 2026-07-04):** the 9 dates in `output/process_dates.txt` →
`20170126 20170203 20170205 20170210 20171008 20171020 20171027 20180501 20180512`.
NB: this guarantees only an ocean-glint *column* coincidence; the near-cloud requirement
is a second filter applied after the cloud-distance pipeline runs on these dates — some
may thin out there.

## Stage 1.5b — footprint collocation + correction runner  ✅ done

`atom_footprint_collocate.py` — collocates the ATom track against the **processed**
OCO-2 parquets (`results/csv_collection/combined_<date>_all_orbits.parquet`),
ocean-glint good-QF (`sfc_type==0 & xco2_qf==0`), 100 km / ±2 h. Loads every UTC day
the flight spans (dateline crossers span two days). Reports per date: footprint count,
bounding box, VMIN/VMAX, and **cloud-distance coverage** (the near-cloud second filter).

**Outcome — 6 usable dates** (good near-cloud coverage): 2017-01-26 (489 fp, all
near-cloud), 2017-02-10 (471/275), 2017-10-20 (433/331), 2017-10-27 (891/158),
2018-05-12 (16/16), and **2017-02-06** (409/409 — recovered 2026-07; this is the OCO day-2
of the dateline-crossing **2017-02-05** flight, so `OCO_TO_FLIGHT` maps it to the
2017-02-05 merged profile). See `$OUT/process_dates.txt` for boxes. **1 stubbed**
(2017-10-08→needs `combined_2017-10-09`, 2nd UTC day). **Unavailable**: 2017-02-03
(its coincidence is on 2017-02-04, unprocessed — 02-03 alone gives 0); 2018-05-01 (training).

`curc_shell_blanca_atom_deepens.sh` — **standalone** runner (NOT wired into the TCCON
launcher, since ATom has no station). Applies the ocean **r05** DE model
(`de_ocean_beta_nll_prof_reg_r05_f*`, M=5) — same OCEAN model as the ship/TCCON launcher
(which pairs ocean r05 + land r15) — via `build_deepens_plot_data.py` to each date →
`…/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/atom/combined_<date>_atom/plot_data.parquet`.
Models are local, so it runs on a laptop OR CURC (module/conda load guarded to Linux).
**Verified locally** on 2017-10-20: anomaly RMS 0.632→0.280 ppm (+55.6%). 6 active + 1 stubbed.

## Stage 2/3 — pseudo-column + AK + comparison  ✅ done → `atom_pseudo_column.py`

Per usable date, per profile leg with collocated OCO-2 footprints (100 km / ±2 h):
1. **Pseudo-column profile** from the leg (measured `p_hpa`, `co2_ppm`), binned to a
   monotonic pressure profile; **below floor** hold lowest value; **above ceiling** use
   the OCO-2 prior (`co2_ap_NN`) so unmeasured stratosphere contributes only the prior.
2. **OCO-2 column operator** from the collocated footprints via
   `ak_harmonize.operator_from_dataframe` (parquet `ak/pwf/co2_ap/plev_NN` + `xco2_apriori`).
3. **AK-smoothed pseudo-column** `c_ak = c_a + Σ h·a·(x − x_a)` (Rodgers & Connor 2003 /
   Wunch 2017), compared to collocated OCO-2 `xco2_bc` and `deep_ensemble_corrected_xco2`.

Output `$OUT/atom_pseudo_column_results.csv` + `atom_pseudo_column_summary.png`
(2-panel bias plot with **±1σ error bars** = spread of the collocated OCO-2 soundings,
per-leg bc→corrected and bias-vs-cloud-distance — same σ-errorbar style as
`tccon_comparison_report`; the corrected σ is visibly tighter than bc on high-bias legs).

**Result (6 dates, 13 collocated legs):** the DE correction reduces |residual| vs the
ATom pseudo-column — **near-cloud legs (n=12): 0.592 → 0.506 ppm** (~15%). Biases are
signed and partly cancel across dates (2017-10-20/27 positive, 2017-02-06/10 + 2018-05-12
negative), so mean bias stays near +0.23; |residual| is the honest headline. Clearest
near-cloud date 2017-10-20 (legs 8–11) pulls OCO-2 down toward the aircraft column (leg 11,
cloud 1.1 km: +1.86 → +1.53 ppm); 2017-02-06 pulls a *negative* bias up (−0.31 → −0.15).
Independent in-situ confirmation of the near-cloud ocean bias reduction. Spot-check (small n).

Reuses `workspace/ak_harmonize.py` (the M2 operator). To recover 2 more dates,
process `combined_2017-02-06` / `combined_2017-10-09` and rerun the runner + this.

## Ship-style 4-panel comparison  ✅ → `plot_atom_comparison.py`

The ATom analog of `Ship_analysis/plot_ship_comparison.py` (same layout), per date →
`combined_<date>_atom/atom_comparison_<date>.png`:
1. DeepEns-corrected XCO2 footprints + ATom track (MODIS Aqua bg); each collocated leg
   marked by a diamond coloured by its AK pseudo-column (same scale);
2. histogram OCO-2 original vs DeepEns-corrected (pooled collocated footprints) with each
   leg's AK pseudo-column as a red reference line; `Δmedian(OCO−ATom)` orig→corr in title;
3. original XCO2_bc map (same colour scale, before/after side-by-side);
4. the aircraft CO2 profile(s) that built the pseudo-column (CO2 vs pressure) + OCO-2
   prior profile + column-value reference lines (pseudo-column / OCO orig & corr medians).

The aircraft reference is the AK pseudo-column (per leg), not a continuous XCO2 field —
so panel 2 uses reference *lines* (not a ship-like reference histogram) and panel 4 shows
the profile instead of the ship time series. Reuses the collocation/AK machinery from
`atom_pseudo_column.py`. `Δmedian` per date: 2017-10-20 +0.70→+0.50, 2017-10-27 +0.38→+0.45,
2017-02-06 −0.31→−0.15 (negative bias pulled up), others already near zero. Dateline
dates (2017-02-06) map cleanly since the collocation sits east of 180° in the day-2 parquet.

```
python plot_atom_comparison.py            # all 5 dates
python plot_atom_comparison.py --no-modis # skip GIBS
```

## MODIS Aqua overlay maps  ✅ → `atom_modis_overlay.py`

Per-date map mirroring `plot_corrected_xco2.py --modis-auto`: MODIS Aqua true-colour
composite (NASA GIBS, reusing `plot_corrected_xco2.download_modis_rgb`) as background,
OCO-2 ocean soundings coloured by DE-corrected XCO2 (collocated set full-opacity, rest
faint context), ATom flight track in magenta. Puts aircraft + soundings + actual cloud
field in one frame — the near-cloud collocation is visually explicit (e.g. 2017-02-10:
the OCO swath crosses the track at a cloud band, near-cloud end reads higher XCO2).

```
python atom_modis_overlay.py            # all 5 dates → output/atom_modis_<date>.png
python atom_modis_overlay.py --no-modis # skip GIBS (points-only)
```
Needs `owslib`+`cartopy` (present locally) and network for the GIBS tile (cached in
`output/aqua_rgb_*.png` and reused).

## References (method precedent)

- Wunch et al. 2010, AMT — TCCON calibration via aircraft profile pseudo-columns.
- Frankenberg et al. 2016 — HIPPO profiles → satellite XCO2 validation (same Wofsy platform lineage).
- Wofsy et al., ATom mission (ORNL DAAC).
