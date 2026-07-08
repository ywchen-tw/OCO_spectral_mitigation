# Nassar Power-Plant Plume Controls

Purpose: close M1 by testing whether the cloud-proximity correction leaves real
CO2 plume enhancements intact.  These cases are not TCCON station-days and should
not be mixed into the TCCON aggregate scripts.

## Inputs

`nassar_power_plant_cases.csv` contains the exact source/date/coordinate table
from Nassar et al. (2021).  One row is one source on one OCO-2 date; the
Vindhyachal/Sasan joint overpass is represented as two source rows on the same
date.

## Workflow

1. Process the listed dates through the normal OCO-2/MODIS/spectral/feature
   pipeline until this exists for each date:

   ```bash
   results/csv_collection/combined_<YYYY-MM-DD>_all_orbits.parquet
   ```

2. Apply the production deep ensemble:

   ```bash
   sbatch workspace/Nassar_plume_analysis/curc_shell_blanca_nassar_deepens.sh
   ```

3. Inspect the screen:

   ```bash
   python workspace/Nassar_plume_analysis/screen_nassar_plumes.py
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
