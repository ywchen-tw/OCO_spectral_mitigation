# combined_analyze.py — Refactoring & Analysis Expansion Plan

**Owner**: Yu-Cheng
**Last updated**: 2026-03-10
**Status legend**: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Part 1 — Code Split (Refactoring)

Split the monolithic `combined_analyze.py` (~3 277 lines, 53 functions) into focused modules.
All modules live in `src/`. `combined_analyze.py` becomes an entry-point only.

### Target layout

```
src/
  ca_utils.py           ~100 lines   shared helpers, no plots
  ca_signal.py          ~250 lines   Section 1: signal hierarchy
  ca_exp_alb.py         ~600 lines   Section 2: exp_intercept / albedo
  ca_k_coeff.py         ~650 lines   Section 3: k1/k2/k3 + distributions
  ca_stratified.py      ~400 lines   Section 4: stratified analysis
  ca_xco2.py            ~450 lines   Section 5: XCO2 anomaly (+ sign split)
  ca_ref_corrected.py   ~700 lines   Ref-corrected R0–R7 (+ R8–R12 new)
  combined_analyze.py   ~150 lines   main() + imports only
```

### Task checklist

#### ca_utils.py
Functions to move: `get_storage_dir`, `load_data`, `apply_quality_filter`,
`split_by_surface`, `cld_dist_bins`, `bin_by_cld_dist`, `_save`,
`rolling_median_iqr`, `print_summary_stats`

- [x] Create `ca_utils.py` with above functions
- [x] Add module-level imports: `numpy`, `pandas`, `matplotlib`, `scipy.stats`, `logging`, `Path`, `Config`

#### ca_signal.py
Functions: `plot_signal_hierarchy`, `plot_residual_signal_hierarchy`

- [x] Create `ca_signal.py`
- [x] Import from `ca_utils`: `_save`, `rolling_median_iqr`, `bin_by_cld_dist`

#### ca_exp_alb.py
Functions: `plot_alb_vs_exp_intercept`, `plot_alb_vs_exp_intercept_cross`,
`plot_intercept_binned_profile`, `plot_exp_intercept_interband_coherence`,
`plot_alb_exp_divergence`, `plot_exp_intercept_albedo_residuals`,
`plot_exp_alb_ratio_residuals`, `plot_alb_binned_profile`

- [x] Create `ca_exp_alb.py`

#### ca_k_coeff.py
Functions: `plot_distributions_vs_cld_dist`, `plot_k1_k2_vs_cld_dist`,
`plot_k2_over_k1_vs_cld_dist`, `plot_k1_k2_binned_profile`, `plot_k1_k2_joint`,
`plot_higher_order_k_profiles`, `plot_k_albedo_residuals`,
`plot_cross_band_k_combinations`

- [x] Create `ca_k_coeff.py`

#### ca_stratified.py
Objects/functions: `STRAT_CONFIG`, `_safe_label`, `_build_strata`,
`plot_k1_k2_overlay`, `plot_intercept_overlay`,
`plot_xco2_anomaly_binned_overlay`, `run_stratified_analysis`

- [x] Create `ca_stratified.py`

#### ca_xco2.py
Functions: `plot_xco2_anomaly_correlations`, `plot_xco2_anomaly_vs_key_vars`,
`plot_xco2_anomaly_vs_cld_dist_binned`, `plot_xco2_anomaly_partial`,
`plot_xco2_derived_vs_cld_dist_binned`, `plot_xco2_derived_vs_bc_anomaly`
(+ new sign-split function from Part 3)

- [x] Create `ca_xco2.py` with existing functions
- [x] Add `run_xco2_sign_analysis()` (see Part 3)

#### ca_ref_corrected.py
Objects/functions: `_REF_PAIRS`, `_R25_PAIRS`, `add_ref_anomalies`,
`add_r25_anomalies`, `_has_ref_data`, `_has_r25_data`, `_binned_ref_profile`,
`plot_ref_diff_vs_cld_dist` (R0), `plot_ref_coverage_bias` (R1),
`plot_ref_std_profiles` (R2), `plot_ref_corrected_profiles` (R3),
`plot_ref_zscore_profiles` (R4), `plot_ref_signal_hierarchy` (R5),
`plot_ref_alb_decoupled_exp` (R6), `plot_obs_vs_ref_scatter` (R7)
(+ new R8–R12 from Part 2)

- [x] Create `ca_ref_corrected.py` with existing R0–R7 functions
- [ ] Add R8–R12 new functions (see Part 2)

#### combined_analyze.py (entry point)
- [x] Replace function bodies with imports from new modules
- [x] Keep only `main()` and top-level `if __name__ == '__main__'`
- [x] Verify end-to-end run produces identical output (365 figures, 0 errors, 2020-01-01 parquet)

---

## Part 2 — New fp-vs-ref Difference Analyses (R8–R13)

Extend the existing R0–R7 ref-corrected block in `ca_ref_corrected.py`.
All new functions accept `pairs=` and `tag=` so they work for both `ref` and `r25` references.
Output goes to `ref_corrected/` and `r25_corrected/` respectively.
R13 (`fp_area_km` analysis) lives in `ca_k_coeff.py` and is called from the per-surface loop in `main()`.

### R8 — Multi-variable delta comparison (per surface type)

**Goal**: On one figure, overlay the binned-mean profiles of all delta columns
(`dk1_o2a`, `dk2_o2a`, `dexp_o2a`, `dalb_o2a`, …) per surface type to compare
the magnitude and direction of cloud-induced perturbation across variables in a
single view.

**Function**: `plot_ref_delta_multivar(df, bins, labels, outdir, pairs, tag)`
**Output**: `{tag}_delta_multivar_{ocean,land}.png`

- [x] Implement `plot_ref_delta_multivar`
- [x] Call from `main()` inside `_has_ref_data` and `_has_r25_data` blocks

### R9 — Cross-band delta coherence

**Goal**: Scatter matrix of `dk1_o2a` vs `dk1_wco2` vs `dk1_sco2` (and same for
`dk2`, `dexp`, `dalb`) colored by `cld_dist_km`. Mirrors the existing
`plot_cross_band_k_combinations` but operates on reference-corrected deltas.
Reveals whether cloud contamination is coherent across bands.

**Function**: `plot_ref_cross_band_delta(df, outdir, pairs, tag)`
**Output**: `{tag}_cross_band_delta_k1.png`, `{tag}_cross_band_delta_k2.png`,
`{tag}_cross_band_delta_alb.png`, `{tag}_cross_band_delta_exp.png`

- [x] Implement `plot_ref_cross_band_delta`
- [x] Call from `main()`

### R10 — Delta decay length scale

**Goal**: For each delta variable, fit an exponential decay `A·exp(−d/τ) + C`
to the binned mean vs `cld_dist_km`. Report fitted τ (km) and amplitude A per
variable and surface type as a table and a grouped bar chart.

**Function**: `plot_ref_delta_decay(df, bins, labels, outdir, pairs, tag)`
**Output**: `{tag}_delta_decay_lengths.png` + `{tag}_delta_decay_table.csv`

- [x] Implement `plot_ref_delta_decay` (use `scipy.optimize.curve_fit`)
- [x] Call from `main()`

### R11 — Delta vs XCO2 BC anomaly joint analysis

**Goal**: Scatter of each `dk1`/`dk2`/`dexp` vs `xco2_bc_anomaly`, colored by
`cld_dist_km`, per surface type. Rolling median overlay. Tests whether the
reference-corrected cloud signal predicts XCO2 bias.

**Function**: `plot_ref_delta_vs_xco2(df, outdir, pairs, tag, max_dist=50)`
**Output**: `{tag}_delta_vs_xco2_{k1,k2,exp,alb}.png`

- [x] Implement `plot_ref_delta_vs_xco2`
- [x] Call from `main()`

### R12 — Partial correlation of delta variables with XCO2 anomaly

**Goal**: Pearson partial-r of each delta column vs `xco2_bc_anomaly` after
OLS-removing `alb_*` + `airmass` + `cos(SZA)` confounders. Produces a bar
chart analogous to `plot_residual_signal_hierarchy` but restricted to delta
columns. Separates ocean and land.

**Function**: `plot_ref_delta_partial_xco2(df, outdir, pairs, tag)`
**Output**: `{tag}_delta_partial_xco2.png`

- [x] Implement `plot_ref_delta_partial_xco2`
- [x] Call from `main()`

### R14 — Ref-corrected profiles stratified by footprint area

**Goal**: Mirror R3 (`ref_corrected_k1_profiles.png`) but replace the single mean line per
subplot with one coloured line per `fp_area_km2` quintile. Shows whether the cloud-adjacency
signal in `obs − ref` depends on footprint size.

**Six variable groups**: k1, k2, albedo, exp_intercept, exp_intercept − albedo, exp_intercept / albedo.
Derived diffs computed inline:
- `Δ(exp − alb)_{band}` = `dexp_{band} − dalb_{band}`
- `Δ(exp / alb)_{band}` = `(obs_exp / obs_alb) − (ref_exp_mean / ref_alb_mean)`

**Function**: `plot_ref_corrected_profiles_by_fp_area(df, bins, labels, outdir, pairs, tag)`
**Module**: `ca_ref_corrected.py`
**Output** (under `{outdir}/fp_area/`):
- `{tag}_corrected_{k1,k2,alb,exp,exp_minus_alb,exp_over_alb}_profiles_by_fp_area.png`

- [x] Implement `plot_ref_corrected_profiles_by_fp_area` in `ca_ref_corrected.py`
- [x] Export and import in `combined_analyze.py`
- [x] Call from `main()` inside `_has_ref_data` block as R14

---

### R13 — Footprint area vs spectral variables within cloud-distance groups

**Goal**: Disentangle the footprint-size effect from the cloud-proximity effect.
Within each cloud-distance bin, compare `k1`, `k2`, `exp_intercept`, `albedo`,
and `xco2_bc_anomaly` as a function of `fp_area_km` (footprint area in km²).
Answers whether larger footprints (more area-averaged signal) behave differently
from smaller ones at the same cloud distance.

**Analyses**:
1. **Binned profile per cld_dist bin** — for each bin, plot mean ± SEM of each
   spectral variable vs `fp_area_km` quantile bins (5 equal-count bins).
   One figure per spectral variable; rows = cld_dist bins, single x-axis = fp_area_km.
2. **2-D hexbin** — `fp_area_km` (x) vs each spectral variable (y), colored by
   `cld_dist_km`. Rolling median per cld_dist quintile overlaid.
   Reveals whether the cloud-proximity trend is modulated by footprint size.
3. **Partial correlation bar chart** — Pearson r of each spectral variable vs
   `fp_area_km` within each cld_dist bin, side-by-side bars. Shows whether the
   fp-area correlation changes sign or magnitude closer to clouds.
4. **Interaction heatmap** — 2-D grid of mean `xco2_bc_anomaly` indexed by
   (cld_dist bin × fp_area_km quintile). Exposes the joint effect.

**Function**: `plot_fp_area_analysis(df, bins, labels, outdir, max_dist=50)`
**Module**: `ca_k_coeff.py`
**Output** (under `{sfc_outdir}/fp_area/`):
- `fp_area_binned_{k1,k2,exp,alb,xco2}.png`
- `fp_area_hexbin_{k1,k2,exp,alb,xco2}.png`
- `fp_area_partial_r.png`
- `fp_area_xco2_interaction_heatmap.png`

**Notes**:
- Skip silently if `fp_area_km` column is absent.
- Use `observed=True` in all `groupby` calls (categorical cld_dist bins).
- Clip `fp_area_km` at 1st–99th percentile before binning to remove outliers.

- [x] Implement `plot_fp_area_analysis` in `ca_k_coeff.py`
- [x] Export from `ca_k_coeff.py` and import in `combined_analyze.py`
- [x] Call from the per-surface loop in `main()` after `plot_cross_band_k_combinations`
- [x] Add `fp_area_km2` to `plot_distributions_vs_cld_dist` variable list

**Note**: column renamed `fp_area_km` → `fp_area_km2` (km²) in pipeline; R13 uses `fp_area_km2`.
**Bug fixed**: `tick_labels=` → `labels=` in `ax.boxplot()` for matplotlib < 3.9 compatibility.

---

## Part 3 — Stratification by XCO2 BC Anomaly Sign

Run the full core analysis suite separately for positive and negative
`xco2_bc_anomaly` subsets, within each surface type.

### Motivation

Ocean and land soundings with positive XCO2 BC anomaly (retrieved XCO2 > mean)
near clouds may have a different spectral fingerprint than those with negative
anomaly. Separating by sign isolates directional biases.

### Output directory structure

```
figures/cld_dist_analysis/
  ocean/
    xco2_sign/
      pos/    (xco2_bc_anomaly >= 0)
      neg/    (xco2_bc_anomaly <  0)
  land/
    xco2_sign/
      pos/
      neg/
```

### Figures produced per sign×surface subset

The same core plot suite as the main `sfc_name` loop:

| Section | Plot |
|---------|------|
| Supp    | `dist_vs_cld_dist_boxplot.png` |
| 2c      | `exp_intercept_binned_profile.png` |
| 2a      | `alb_vs_exp_intercept.png`, `alb_vs_exp_intercept_cross.png` |
| 3a      | `k1_k2_binned_profile.png`, `k1_k2_vs_cld_dist.png` |
| 3b      | `k2_over_k1_vs_cld_dist.png` |
| 3d      | `k1_vs_k2_joint_cld_dist.png` |
| Supp    | `alb_binned_profile.png` |
| 5       | `xco2_{bc,raw}_anomaly_vs_cld_dist_binned.png`, correlation heatmap |
| R0–R7  | (if ref data present) ref-corrected profiles for this sign subset |

Additionally, one **comparison overlay** figure per surface type:

**`xco2_sign_comparison.png`** — overlay the binned mean profiles of `dk1`,
`dk2`, `dexp` for the pos and neg subsets on the same axes (4 panels, one per
band+variable), to visualize direction differences directly.

### New function in `ca_xco2.py`

```python
def run_xco2_sign_analysis(
    df: pd.DataFrame,
    bins, labels,
    sfc_outdir: str,
    run_ref: bool = False,
    ref_pairs=None, r25_pairs=None,
) -> None:
    """Split df by sign of xco2_bc_anomaly and run core plots for each half."""
```

### Task checklist

- [x] Implement `run_xco2_sign_analysis` in `ca_xco2.py`
  - [x] Split: `pos = df[df['xco2_bc_anomaly'] >= 0]`, `neg = df[df['xco2_bc_anomaly'] < 0]`
  - [x] Log counts for each subset; skip if < 500 soundings
  - [x] Call full core plot suite for `pos` → `sfc_outdir/xco2_sign/pos/`
  - [x] Call full core plot suite for `neg` → `sfc_outdir/xco2_sign/neg/`
  - [x] Run ref-corrected R0–R7 per subset if `run_ref=True` and columns present
- [x] Implement `plot_xco2_sign_comparison` in `ca_xco2.py`
  - [x] 2×2 grid: one panel per delta variable (dk1, dk2, dexp, dalb); pos=solid, neg=dashed; bands colored
  - [x] Save to `sfc_outdir/xco2_sign/sign_comparison.png`
- [x] Call `run_xco2_sign_analysis` from `main()` inside the surface-type loop,
  after all existing per-surface plots:
  ```python
  run_xco2_sign_analysis(sdf, bins, labels, sfc_outdir,
                         run_ref=_has_ref_data(sdf))
  ```

---

## Bug Fixes Applied

| # | File | Fix |
|---|------|-----|
| B1 | `ca_ref_corrected.py` | `isinstance()` arg 2 was a bool — removed dead `is_derived` variable |
| B2 | `ca_ref_corrected.py` | `sfc_subsets` sliced before derived diff columns added → moved to after derived column computation |
| B3 | `combined_analyze.py` | Stratified loop skipped `mu_sza` when `sza` present (and vice-versa) → replaced with `if strat_var not in sdf.columns: continue` so both run when both columns exist |

---

## Integration Checklist

- [ ] All new `ca_*.py` modules tested independently with a small parquet sample
- [x] New R8–R14 figures appear in `ref_corrected/fp_area/`
- [x] Sign-split figures appear under `ocean/xco2_sign/` and `land/xco2_sign/`
- [x] Both `sza` and `mu_sza` stratifications run when both columns present
- [ ] CURC shell script (`curc_shell_blanca_combined_analysis.sh`) updated if
  the entry-point module name changes
- [x] This plan file updated to `[x]` as tasks are completed
