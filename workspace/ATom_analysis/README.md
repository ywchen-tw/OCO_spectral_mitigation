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

## Stage 2 — pseudo-column + AK  ⬜ not started (the hard half)

1. **Profile → pressure grid** per leg (use measured `p_hpa` directly).
2. **Extend below floor** (~0.2 km → surface): over ocean, hold lowest valid value down.
   Small mass, small error.
3. **Extend above ceiling** (~12 km → TOA): the dominant uncertainty. Stitch to a model/
   prior stratosphere — cleanest is the **OCO-2 retrieval a-priori profile** above the
   aircraft top (alt: CAMS / CarbonTracker). Bounds the error to the ~15% column mass
   not measured.
4. **Pressure-weight** → `XCO2_pseudo = Σ hᵢ·CO2ᵢ`.
5. **Apply OCO-2 averaging kernel + prior** → `XCO2_AK = XCO2_prior + Σ hᵢ·aᵢ·(profileᵢ − priorᵢ)`.
   **Required, not optional** — comparing a true column to an AK-weighted retrieval
   otherwise leaves a tenths-of-ppm mismatch. This is the same **M2** operator the TCCON
   chain still needs; build once, use for both.

## Stage 3 — collocation with OCO-2 ocean-glint  ⬜ not started

Match each pseudo-column (mean lat/lon/time of its leg) to OCO-2 `sfc_type==0` glint
soundings in a space/time window. Expect **few** tight coincidences → report as spot-check.

## References (method precedent)

- Wunch et al. 2010, AMT — TCCON calibration via aircraft profile pseudo-columns.
- Frankenberg et al. 2016 — HIPPO profiles → satellite XCO2 validation (same Wofsy platform lineage).
- Wofsy et al., ATom mission (ORNL DAAC).
