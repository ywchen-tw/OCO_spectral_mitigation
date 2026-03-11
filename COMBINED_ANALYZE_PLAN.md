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

- [ ] Create `ca_utils.py` with above functions
- [ ] Add module-level imports: `numpy`, `pandas`, `matplotlib`, `scipy.stats`, `logging`, `Path`, `Config`

#### ca_signal.py
Functions: `plot_signal_hierarchy`, `plot_residual_signal_hierarchy`

- [ ] Create `ca_signal.py`
- [ ] Import from `ca_utils`: `_save`, `rolling_median_iqr`, `bin_by_cld_dist`

#### ca_exp_alb.py
Functions: `plot_alb_vs_exp_intercept`, `plot_alb_vs_exp_intercept_cross`,
`plot_intercept_binned_profile`, `plot_exp_intercept_interband_coherence`,
`plot_alb_exp_divergence`, `plot_exp_intercept_albedo_residuals`,
`plot_exp_alb_ratio_residuals`, `plot_alb_binned_profile`

- [ ] Create `ca_exp_alb.py`

#### ca_k_coeff.py
Functions: `plot_distributions_vs_cld_dist`, `plot_k1_k2_vs_cld_dist`,
`plot_k2_over_k1_vs_cld_dist`, `plot_k1_k2_binned_profile`, `plot_k1_k2_joint`,
`plot_higher_order_k_profiles`, `plot_k_albedo_residuals`,
`plot_cross_band_k_combinations`

- [ ] Create `ca_k_coeff.py`

#### ca_stratified.py
Objects/functions: `STRAT_CONFIG`, `_safe_label`, `_build_strata`,
`plot_k1_k2_overlay`, `plot_intercept_overlay`,
`plot_xco2_anomaly_binned_overlay`, `run_stratified_analysis`

- [ ] Create `ca_stratified.py`

#### ca_xco2.py
Functions: `plot_xco2_anomaly_correlations`, `plot_xco2_anomaly_vs_key_vars`,
`plot_xco2_anomaly_vs_cld_dist_binned`, `plot_xco2_anomaly_partial`,
`plot_xco2_derived_vs_cld_dist_binned`, `plot_xco2_derived_vs_bc_anomaly`
(+ new sign-split function from Part 3)

- [ ] Create `ca_xco2.py` with existing functions
- [ ] Add `run_xco2_sign_analysis()` (see Part 3)

#### ca_ref_corrected.py
Objects/functions: `_REF_PAIRS`, `_R25_PAIRS`, `add_ref_anomalies`,
`add_r25_anomalies`, `_has_ref_data`, `_has_r25_data`, `_binned_ref_profile`,
`plot_ref_diff_vs_cld_dist` (R0), `plot_ref_coverage_bias` (R1),
`plot_ref_std_profiles` (R2), `plot_ref_corrected_profiles` (R3),
`plot_ref_zscore_profiles` (R4), `plot_ref_signal_hierarchy` (R5),
`plot_ref_alb_decoupled_exp` (R6), `plot_obs_vs_ref_scatter` (R7)
(+ new R8–R12 from Part 2)

- [ ] Create `ca_ref_corrected.py` with existing R0–R7 functions
- [ ] Add R8–R12 new functions (see Part 2)

#### combined_analyze.py (entry point)
- [ ] Replace function bodies with imports from new modules
- [ ] Keep only `main()` and top-level `if __name__ == '__main__'`
- [ ] Verify end-to-end run produces identical output

---

## Part 2 — New fp-vs-ref Difference Analyses (R8–R12)

Extend the existing R0–R7 ref-corrected block in `ca_ref_corrected.py`.
All new functions accept `pairs=` and `tag=` so they work for both `ref` and `r25` references.
Output goes to `ref_corrected/` and `r25_corrected/` respectively.

### R8 — Multi-variable delta comparison (per surface type)

**Goal**: On one figure, overlay the binned-mean profiles of all delta columns
(`dk1_o2a`, `dk2_o2a`, `dexp_o2a`, `dalb_o2a`, …) per surface type to compare
the magnitude and direction of cloud-induced perturbation across variables in a
single view.

**Function**: `plot_ref_delta_multivar(df, bins, labels, outdir, pairs, tag)`
**Output**: `{tag}_delta_multivar_{ocean,land}.png`

- [ ] Implement `plot_ref_delta_multivar`
- [ ] Call from `main()` inside `_has_ref_data` and `_has_r25_data` blocks

### R9 — Cross-band delta coherence

**Goal**: Scatter matrix of `dk1_o2a` vs `dk1_wco2` vs `dk1_sco2` (and same for
`dk2`, `dexp`, `dalb`) colored by `cld_dist_km`. Mirrors the existing
`plot_cross_band_k_combinations` but operates on reference-corrected deltas.
Reveals whether cloud contamination is coherent across bands.

**Function**: `plot_ref_cross_band_delta(df, outdir, pairs, tag)`
**Output**: `{tag}_cross_band_delta_k1.png`, `{tag}_cross_band_delta_k2.png`,
`{tag}_cross_band_delta_alb.png`, `{tag}_cross_band_delta_exp.png`

- [ ] Implement `plot_ref_cross_band_delta`
- [ ] Call from `main()`

### R10 — Delta decay length scale

**Goal**: For each delta variable, fit an exponential decay `A·exp(−d/τ) + C`
to the binned mean vs `cld_dist_km`. Report fitted τ (km) and amplitude A per
variable and surface type as a table and a grouped bar chart.

**Function**: `plot_ref_delta_decay(df, bins, labels, outdir, pairs, tag)`
**Output**: `{tag}_delta_decay_lengths.png` + `{tag}_delta_decay_table.csv`

- [ ] Implement `plot_ref_delta_decay` (use `scipy.optimize.curve_fit`)
- [ ] Call from `main()`

### R11 — Delta vs XCO2 BC anomaly joint analysis

**Goal**: Scatter of each `dk1`/`dk2`/`dexp` vs `xco2_bc_anomaly`, colored by
`cld_dist_km`, per surface type. Rolling median overlay. Tests whether the
reference-corrected cloud signal predicts XCO2 bias.

**Function**: `plot_ref_delta_vs_xco2(df, outdir, pairs, tag, max_dist=50)`
**Output**: `{tag}_delta_vs_xco2_{k1,k2,exp,alb}.png`

- [ ] Implement `plot_ref_delta_vs_xco2`
- [ ] Call from `main()`

### R12 — Partial correlation of delta variables with XCO2 anomaly

**Goal**: Pearson partial-r of each delta column vs `xco2_bc_anomaly` after
OLS-removing `alb_*` + `airmass` + `cos(SZA)` confounders. Produces a bar
chart analogous to `plot_residual_signal_hierarchy` but restricted to delta
columns. Separates ocean and land.

**Function**: `plot_ref_delta_partial_xco2(df, outdir, pairs, tag)`
**Output**: `{tag}_delta_partial_xco2.png`

- [ ] Implement `plot_ref_delta_partial_xco2`
- [ ] Call from `main()`

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

- [ ] Implement `run_xco2_sign_analysis` in `ca_xco2.py`
  - [ ] Split: `pos = df[df['xco2_bc_anomaly'] >= 0]`, `neg = df[df['xco2_bc_anomaly'] < 0]`
  - [ ] Log counts for each subset; skip if < 500 soundings
  - [ ] Call full core plot suite for `pos` → `sfc_outdir/xco2_sign/pos/`
  - [ ] Call full core plot suite for `neg` → `sfc_outdir/xco2_sign/neg/`
  - [ ] Run ref-corrected R0–R7 per subset if `run_ref=True` and columns present
- [ ] Implement `plot_xco2_sign_comparison` in `ca_xco2.py`
  - [ ] 2×2 or 3×2 grid: one panel per (band, delta variable); pos=blue, neg=red
  - [ ] Save to `sfc_outdir/xco2_sign/sign_comparison.png`
- [ ] Call `run_xco2_sign_analysis` from `main()` inside the surface-type loop,
  after all existing per-surface plots:
  ```python
  run_xco2_sign_analysis(sdf, bins, labels, sfc_outdir,
                         run_ref=_has_ref_data(sdf))
  ```

---

## Integration Checklist

- [ ] All new `ca_*.py` modules tested independently with a small parquet sample
- [ ] `main()` in refactored `combined_analyze.py` produces same figures as original
  (diff the figure list or use `md5sum`)
- [ ] New R8–R12 figures appear in `ref_corrected/` and `r25_corrected/`
- [ ] Sign-split figures appear under `ocean/xco2_sign/` and `land/xco2_sign/`
- [ ] CURC shell script (`curc_shell_blanca_combined_analysis.sh`) updated if
  the entry-point module name changes
- [ ] This plan file updated to `[x]` as tasks are completed
