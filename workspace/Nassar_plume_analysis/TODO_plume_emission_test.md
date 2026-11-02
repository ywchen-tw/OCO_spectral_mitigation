# TO-DO (hand-over to the local agent): plume emission-rate test

Written 2026-11-02 on CURC. Not started locally yet. Kozienice is OUT of scope (no emission-rate test for it).

## Why
The bias corrections reduce real plume enhancements. Sasan 2014-10-23 and Kozienice 2021-09-06 are
nearly clear-sky cases. Feature attribution shows the cause is the CO2-band k1 (`sco2_k1_am`, `wco2_k1_am`):
it scales with the true CO2 amount, and the local clear-sky training target labels real local CO2 excess
as bias. The formal model is now tuned XGBoost on `no_xco2_sig_shape`, which drops the CO2-band k1/k2 and
adds the g-invariant ratios k2/k1^2 and k1_S/k1_W. It keeps about 98 % of plume enhancements, against
84 % for `no_xco2_sig`.

The open question is whether the plume reduction by `no_xco2_sig` is real CO2 loss or legitimate
correction. The independent check is the emission rate inferred from the plume, compared with the
plant's reported hourly emission (EPA CAMPD).

## What exists (CURC, 2026-11-02, untuned XGB models)
Method: `workspace/Nassar_plume_analysis/us_plant_emissions.py`, run unchanged (Nassar et al. 2017
Gaussian plume, slant path). The winds were Open-Meteo ERA5 100 m only; there was no CDS profile.

| series | Ghent 2015-08-13 Q/Q_rep | Westar 2015-12-04 Q/Q_rep |
|---|---|---|
| xco2_bc | 0.73 | 0.80 |
| XGB no_xco2_sig | 0.62 | 0.66 |
| XGB shape | 0.71 | 0.80 |
| TabM no_xco2_sig | 0.64 | 0.73 |

Values use the best-fit wind direction (±30°) and the 100 m wind.
- All series are low, most likely because the 100 m wind is slower than the wind at plume height.
- The standard corrections lower Q by a further 10–18 %. `shape` stays at the xco2_bc level.
- This is not conclusive:
  - 10 m vs 100 m wind alone changes Q by 20–27 %;
  - Ghent's wind is below 1 m/s;
  - Westar's wind runs along the track (within 3°).

## To do
1. **Plume-height wind.** Use the ERA5 pressure-level profile from the Copernicus CDS (needs
   `~/.cdsapirc`): `workspace/Nassar_plume_analysis/era5_plant_winds_download.py`, then rerun
   `us_plant_emissions.py` with the stack+250 m wind (heights `stack250`, `stack250_mix` with MERRA-2;
   MERRA-2 needs Earthdata and earthaccess, `merra2_plant_winds_download.py`).
2. **More cases.**
   - Screen all available dates for US plants with EPA CAMPD hourly data, using these rules:
     - closest OCO-2 footprint ≤ ~5 km from the plant;
     - plume-height wind 2–8 m/s;
     - wind crossing the track by more than 30°;
     - not a training date: the 116 dates listed in any `training_dates.json`.
   - `screen_nassar_plumes.py` / `nassar_power_plants.csv` are the starting point.
   - Aim for a handful of good-geometry overpasses, so this becomes a population test.
3. **Series to compare per case:**
   - xco2_bc;
   - final XGB `no_xco2_sig_shape`;
   - XGB `no_xco2_sig`.

   All XGB series are tuned XGB (config c41), prefix `fm1620k5v2xt41`. Report Q, Q/Q_rep, the fit
   residual, and the spread across wind heights.

## Inputs to copy from CURC (scratch: /scratch/alpine/yuch8913/oco/spec_anal/results_doppler)
- Per-case plot_data:
  - `model_comparison/xgb/<run>/nassar_plumes/combined_<date>/plot_data.parquet`, where `<run>` is
    `fm1620k5v2xt41_am_no_xco2_sig_shape` (final) or `fm1620k5v2xt41_am_no_xco2_sig`. Both exist for
    Ghent (`combined_2015-08-13`) and Westar (`combined_2015-12-04`), built 2026-11-02.
  - Untuned references: `xgb/fm1620k5v2_am_no_xco2_sig/nassar_plumes/combined_<date>/` (no_xco2_sig),
    `xgb/fm1620k5v2_am_no_xco2_sig/nassar_plumes_variants/shape/combined_<date>/` (shape arm),
    `tabm/fm1620k5v2tuneC_am_no_xco2_sig/nassar_plumes/combined_<date>/`.
  - Final-model retention for context (10 km core, xco2_bc → corrected): tuned XGB shape keeps Ghent 90 %,
    Westar 80 %; tuned XGB no_xco2_sig keeps 86 % / 70 % (scratchpad `drift_plume/plume_summary_final.md`).
  - The corrected column is `xgb_corrected_xco2`. `us_plant_emissions.py` reads
    `deep_ensemble_corrected_xco2`, so rename the column (the CURC runs used a wrapper,
    scratchpad `drift_plume/nassar_adapt.py`) or add a `--corr-col` option.
- Existing wind / CAMPD caches (Open-Meteo 10 m / 100 m, CAMPD hour):
  `model_comparison/xgb/fm1620k5v2_am_no_xco2_sig/nassar_plumes_variants/emissions/_cache_wind{100m,10m}/`.
- Results so far:
  `model_comparison/{xgb/fm1620k5v2_am_no_xco2_sig,tabm/fm1620k5v2tuneC_am_no_xco2_sig}/nassar_plumes_variants/emissions/`.
- New case dates need per-date parquets (`csv_collection/combined_<date>_all_orbits.parquet`, about
  0.3–1 GB each) and the 10 fold models of each XGB run (`model_gbdt/am/fm1620k5v2xt41_*`).
