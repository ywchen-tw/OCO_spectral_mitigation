# Nassar Power-Plant Plume Controls

Purpose: close M1 by testing whether the cloud-proximity correction leaves real
CO2 plume enhancements intact.  These cases are not TCCON station-days and should
not be mixed into the TCCON aggregate scripts.

## Inputs

`nassar_power_plants.csv` contains only source names and coordinates from the
Nassar et al. (2021) table.  Dates are intentionally not part of this catalog;
choose dates from the local OCO-2 inventory or a separate experiment list.

## Workflow

1. Pick the OCO-2 dates to test separately from the plant-coordinate catalog.
   Process those dates through the normal OCO-2/MODIS/spectral/feature pipeline
   until this exists for each date:

   ```bash
   results/csv_collection/combined_<YYYY-MM-DD>_all_orbits.parquet
   ```

2. Apply the production deep ensemble:

   ```bash
   NASSAR_DATES="2018-06-03 2021-04-24" \
     sbatch workspace/Nassar_plume_analysis/curc_shell_blanca_nassar_deepens.sh
   ```

   Or use `NASSAR_DATES_FILE=path/to/date_list.txt`, with one `YYYY-MM-DD` or
   `YYYYMMDD` date per line.

3. Inspect the screen:

   ```bash
   python workspace/Nassar_plume_analysis/screen_nassar_plumes.py \
     --dates 2018-06-03 2021-04-24
   ```

   Outputs:

   ```bash
   results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/nassar_plume_screen.csv
   results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/nassar_plume_screen.md
   ```

## Selection Criteria

Rank cases by:

- source proximity: many footprints within 10-25 km, or the minimum distance is
  close to the Nassar overpass distance;
- clear-sky coverage: many near-source footprints with `cld_dist_km >= 20`;
- plume signal: `xco2_bc` has a coherent along-track enhancement;
- correction inertness: predicted `mu` is flat/small across the plume;
- spectral sanity: `o2a_k1` stays flat while CO2-band k1 changes only at the
  small level expected from the real CO2 enhancement.

## Plume-Preservation Diagnostic

The plume-preservation analysis is deliberately not a flux inversion.  It checks
whether the correction preserves local XCO2 structure near the power plant and
reports the XCO2 ranges requested for footprints within 25 and 50 km:

```bash
conda run -n ml310 python workspace/Nassar_plume_analysis/analyze_nassar_plume_preservation.py
```

Outputs:

```bash
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/plume_preservation/nassar_plume_preservation_summary.csv
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/plume_preservation/nassar_plume_preservation_summary.md
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/plume_preservation/nassar_plume_preservation_by_footprint.csv
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/plume_preservation/nassar_plume_preservation_by_footprint.md
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/nassar_plumes/plume_preservation/footprints/
```

The case and footprint summaries also report cloud-distance context for the
spectral coefficients: `cld_dist_km` spread, a simple cloud-distance regime
label, and correlations between cloud distance and `o2a_k1`, `o2a_k2`,
`wco2_k1`, and `wco2_k2`.  Use these fields when interpreting spectral std:
small k-parameter std is only strong evidence for spectral stability when the
local footprints also sample a meaningful range of cloud distances.

Downwind fractions require wind direction, which is not currently stored in the
combined feature parquets.  Supply directions as degrees clockwise from north:

```bash
conda run -n ml310 python workspace/Nassar_plume_analysis/analyze_nassar_plume_preservation.py \
  --wind-from ghent:2024-04-15:270 \
  --wind-to comanche:2020-09-06:135
```

`--wind-from` follows the meteorological convention; the script converts it to
the downwind direction internally.  `--wind-to` is already the transport
direction.
