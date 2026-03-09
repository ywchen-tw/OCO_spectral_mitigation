# combined_csv_analyze.py — Update Log

**File**: `src/combined_csv_analyze.py`
**Purpose**: Analyze parquet output from `fitting_data_correction.py`.
Loads combined `*_all_orbits.parquet` files, applies QF+snow filters, and
generates all diagnostic / science figures for the cloud-proximity analysis.

---

## Analysis Sections (current state)

### Core infrastructure
| Function | Description |
|---|---|
| `load_data()` | Loads combined or per-date parquet files from `results/csv_collection/` |
| `apply_quality_filter()` | Keeps `xco2_bc > 0`, `xco2_qf == 0`, `snow_flag == 0` |
| `split_by_surface()` | Splits into `ocean` (sfc_type=0) and `land` (sfc_type=1) subsets |
| `cld_dist_bins()` / `bin_by_cld_dist()` | Bins `cld_dist_km` into fixed-edge intervals |
| `rolling_median_iqr()` | Fast O(n) binned rolling median for scatter overlays |
| `print_summary_stats()` | Pearson r table + binned means printed to stdout |
| `STRAT_CONFIG` | Dict of stratification variables and bin edges for Section 4 |

---

## Section History

### Section 1 — Signal hierarchy (`plot_signal_hierarchy`)
Bar chart of Pearson r(cld_dist_km) for k1, k2, k3, exp_intercept, and exp/alb
ratio across all three bands; ocean vs land side-by-side.
Output: `signal_hierarchy.png`

### Section 1b — Residual signal hierarchy (`plot_residual_signal_hierarchy`)
Same bar chart after OLS-removing band-matched albedo + airmass + cos(SZA) from
each spectral coefficient, and airmass + cos(SZA) + AOD from the exp/alb ratio.
Identifies variables with genuine cloud-proximity signal independent of scene
co-variation.
Key result: SCO₂ k₁/k₂/k₃ on land retain r ≈ +0.20–0.28; ocean O₂A exp/alb ≈ −0.16.
Output: `residual_signal_hierarchy.png`

### Section 2a — Albedo vs exp_intercept scatter
- `plot_alb_vs_exp_intercept`: hexbin + rolling median per band (same-band)
- `plot_alb_vs_exp_intercept_cross`: 3×3 cross-band scatter matrix
Outputs: `alb_vs_exp_intercept.png`, `alb_vs_exp_intercept_cross.png`

### Section 2b — exp_intercept albedo residuals (`plot_exp_intercept_albedo_residuals`)
OLS-removes albedo + airmass + cos(SZA) from each exp_intercept; plots residuals
vs cloud distance (binned mean ± SEM/std), ocean/land separate.
Shows how much cloud-distance signal in exp_intercept survives confounder removal.
Output: `exp_intercept_albedo_residuals.png`

### Section 2c — exp_intercept binned profiles (`plot_intercept_binned_profile`)
Mean ± SEM (bars) / ± std (shading) of spectral exp_intercept per band vs cld_dist bin.
Output: `exp_intercept_binned_profile.png`

### Section 2d — exp_intercept inter-band coherence (`plot_exp_intercept_interband_coherence`)
Pairwise scatter (O₂A vs WCO₂, O₂A vs SCO₂, WCO₂ vs SCO₂) colored by cld_dist_km;
ocean and land rows.  Quantifies shared vs band-specific cloud signal.
Output: `exp_intercept_interband_coherence.png`

### Section 2e — Albedo vs exp_intercept divergence (`plot_alb_exp_divergence`)
% change from far-cloud reference for albedo vs exp_intercept (Fig 1) and the
exp/alb ratio (Fig 2) — ocean/land columns, per band.
Ocean finding: exp/alb rises near clouds (cloud-edge scattered light).
Land finding: exp/alb collapses near clouds (anomalous suppression).
Outputs: `alb_exp_pct_change_vs_cld_dist.png`, `alb_exp_ratio_divergence_vs_cld_dist.png`

### Section 2f — exp/alb ratio residuals (`plot_exp_alb_ratio_residuals`)
OLS-removes airmass + cos(SZA) + AOD from the exp/alb ratio (albedo already
divided out), then plots residuals vs cloud distance.
Result: ocean O₂A retains ~67% of raw signal; land shows sign flip after correction.
Output: `exp_alb_ratio_residuals.png`

### Section 3a — k1/k2 scatter and binned profiles
- `plot_k1_k2_vs_cld_dist`: hexbin + rolling median/IQR vs cld_dist (per band, k1 and k2)
- `plot_k1_k2_binned_profile`: mean ± SEM / ± std binned profile per band
Outputs: `k1_k2_vs_cld_dist.png`, `k1_k2_binned_profile.png`

### Section 3b — k₂/k₁ ratio (`plot_k2_over_k1_vs_cld_dist`)
Hexbin + rolling median of k₂/k₁ ratio per band vs cloud distance.
Scattering asymmetry proxy.
Output: `k2_over_k1_vs_cld_dist.png`

### Section 3c — Higher-order k profiles (`plot_higher_order_k_profiles`)
Binned mean ± SEM/std for k₃ (SCO₂ and WCO₂ only; O₂A k₃/k₄/k₅ negligible).
Ocean and land in separate columns.
Output: `higher_order_k_profiles.png`

### Section 3d — k₁ vs k₂ joint scatter (`plot_k1_k2_joint`)
k₁ vs k₂ scatter colored by cld_dist_km per band.
Output: `k1_vs_k2_joint_cld_dist.png`

### Section 3e — k1/k2/k3 albedo residuals (`plot_k_albedo_residuals`)
OLS-removes alb_{band} + airmass + cos(SZA) from each k coefficient; plots
residuals vs cld_dist, one output file per k term.
Key finding: SCO₂ k₁/k₂/k₃ on land retain r ≈ +0.20–0.28; O₂A and WCO₂ collapse.
Outputs: `k1_albedo_residuals.png`, `k2_albedo_residuals.png`, `k3_albedo_residuals.png`

### Section 4 — Stratified analysis (`run_stratified_analysis`)
Repeats core plots on fixed-edge strata of conditioning variables defined in
`STRAT_CONFIG`: `mu_sza`, `alb_o2a`, `glint_angle`, `aod_total`, `dp`.
Per-stratum figures: `figures/cld_dist_analysis/{sfc_type}/stratified/by_{var}/{bin}/`
Overlay figures:    `figures/cld_dist_analysis/{sfc_type}/stratified/by_{var}/`
- `plot_k1_k2_overlay` — all strata on one k1/k2 profile plot
- `plot_intercept_overlay` — all strata on one exp_intercept profile
- `plot_xco2_anomaly_binned_overlay` — all strata on one XCO₂ anomaly profile

### Section 5 — XCO₂ anomaly partial correlation (`plot_xco2_anomaly_partial`)
Partial correlation of xco2_bc_anomaly with cloud distance after OLS-removing
albedo (all bands) + airmass + cos(SZA) + AOD + ΔP + CO₂_grad + H₂O + dpfrac.
Result: r_resid ≈ 0 — no detectable cloud-proximity bias remains in XCO₂.
Output: `xco2_anomaly_partial_vs_cld_dist.png`

Also per surface type:
- `plot_distributions_vs_cld_dist` → `dist_vs_cld_dist_boxplot.png`
- `plot_xco2_anomaly_correlations` → `xco2_anomaly_correlation_heatmap.png`
- `plot_xco2_anomaly_vs_key_vars` → `xco2_bc_anomaly_vs_predictors.png`
- `plot_xco2_anomaly_vs_cld_dist_binned` → `xco2_{bc,raw}_anomaly_vs_cld_dist_binned.png`
- `plot_alb_binned_profile` → `alb_binned_profile.png`

---

## 2026-03-08 — Ref-corrected analyses (Sections R1–R7)

**Motivation**: Updated parquet files now include 24 new `ref_*` columns containing
clear-sky pixel statistics (mean and σ) for k1, k2, albedo, and exp_intercept for
each spectral band.  These are collocated reference values from cloud-free soundings
nearby in the same overpass, allowing direct obs − ref comparisons that bypass
confounders like local surface type, geometry, and season.

**Data note**: ref coverage is inverted relative to cloud proximity:
- 0–2 km:  ~14% of soundings have a ref value
- 20–50 km: ~96% of soundings have a ref value
Soundings without a ref are excluded from Sections R3–R7 (NaN in diff columns).

**π-scaling**: `ref_exp_int_*_mean` and `ref_exp_int_*_std` are now also scaled
by π in `main()` alongside `exp_*_intercept`, ensuring diffs stay on the same scale.

### New infrastructure
| Symbol | Description |
|---|---|
| `_REF_PAIRS` | Registry of 12 `(obs, ref_mean, ref_std, diff_col, band, term, color)` tuples |
| `add_ref_anomalies(df)` | Appends `d{term}_{band}` (obs−ref) and `z{term}_{band}` ((obs−ref)/σ_ref) columns |
| `_has_ref_data(df)` | Guards all R sections; skips with a warning if no `ref_*` columns present |
| `_binned_ref_profile(ax, …)` | Shared helper: binned mean±SEM (errorbar) / ±std (fill_between) with axhline(0) |

### R1 — Coverage bias (`plot_ref_coverage_bias`)
Grouped bar chart comparing soundings with vs without ref across 6 key variables
per cld_dist bin (mean ± SEM).  Reveals selection bias in the near-cloud subset.
Includes a coverage% panel showing the 14%→96% gradient.
Output: `ref_corrected/ref_coverage_bias.png`

### R2 — Ref σ profiles (`plot_ref_std_profiles`)
Plots ref_std (intra-reference variability) vs cld_dist for k₁, albedo, exp_int
(O₂A and SCO₂); ocean/land columns.  Decreasing σ near clouds may indicate
fewer ref pixels rather than genuine scene homogeneity.
Output: `ref_corrected/ref_std_profiles.png`

### R3 — Ref-corrected anomaly profiles (`plot_ref_corrected_profiles`)
Four figures (k₁, k₂, albedo, exp_int), each 3 bands × 2 surface types.
Binned mean ± SEM/std of `obs − ref_mean`.  y = 0 is the clear-sky baseline.
Outputs: `ref_corrected/ref_corrected_{k1,k2,alb,exp}_profiles.png`

### R4 — Ref z-score profiles (`plot_ref_zscore_profiles`)
Same layout as R3 but y-axis is `(obs − ref_mean) / ref_std`.
Units of natural clear-sky variability; enables cross-band / cross-variable comparison.
Outputs: `ref_corrected/ref_zscore_{k1,k2,alb,exp}_profiles.png`

### R5 — Ref-corrected signal hierarchy (`plot_ref_signal_hierarchy`)
Bar chart of Pearson r(cld_dist, obs−ref) for all 12 diff variables, ocean vs land.
Companion to Section 1.  Variables retaining large |r| after ref subtraction carry
genuine cloud-adjacency signal not explained by scene co-variation.
Output: `ref_corrected/ref_signal_hierarchy.png`

### R6 — Albedo-decoupled exp_int in ref-corrected space (`plot_ref_alb_decoupled_exp`)
OLS: `Δexp ~ const + Δalb` (per band, per surface type).
Plots `Δexp` residuals vs cld_dist; isolates photon-transport effects independent
of surface reflectance changes near clouds.
Reports r_raw(Δexp) → r_resid for each panel.
Output: `ref_corrected/ref_alb_decoupled_exp_residuals.png`

### R7 — Obs vs ref scatter (`plot_obs_vs_ref_scatter`)
Hexbin scatter of obs vs ref_mean with 1:1 line + rolling median / IQR.
Variables: k₁ and exp_int for O₂A and SCO₂; two figures (ocean, land).
Systematic above/below-1:1 departures reveal direction of cloud-adjacency bias.
Outputs: `ref_corrected/obs_vs_ref_scatter_{ocean,land}.png`

---

## Output directory layout

```
results/figures/cld_dist_analysis/
├── signal_hierarchy.png
├── residual_signal_hierarchy.png
├── exp_intercept_albedo_residuals.png
├── alb_exp_pct_change_vs_cld_dist.png
├── alb_exp_ratio_divergence_vs_cld_dist.png
├── exp_alb_ratio_residuals.png
├── k{1,2,3}_albedo_residuals.png
├── exp_intercept_interband_coherence.png
├── higher_order_k_profiles.png
├── ref_corrected/                        ← NEW 2026-03-08
│   ├── ref_coverage_bias.png
│   ├── ref_std_profiles.png
│   ├── ref_corrected_{k1,k2,alb,exp}_profiles.png
│   ├── ref_zscore_{k1,k2,alb,exp}_profiles.png
│   ├── ref_signal_hierarchy.png
│   ├── ref_alb_decoupled_exp_residuals.png
│   ├── obs_vs_ref_scatter_ocean.png
│   └── obs_vs_ref_scatter_land.png
├── ocean/
│   ├── dist_vs_cld_dist_boxplot.png
│   ├── k1_k2_binned_profile.png
│   ├── k1_k2_vs_cld_dist.png
│   ├── k2_over_k1_vs_cld_dist.png
│   ├── k1_vs_k2_joint_cld_dist.png
│   ├── exp_intercept_binned_profile.png
│   ├── alb_vs_exp_intercept.png
│   ├── alb_vs_exp_intercept_cross.png
│   ├── alb_binned_profile.png
│   ├── xco2_anomaly_correlation_heatmap.png
│   ├── xco2_bc_anomaly_vs_predictors.png
│   ├── xco2_{bc,raw}_anomaly_vs_cld_dist_binned.png
│   ├── xco2_anomaly_partial_vs_cld_dist.png
│   └── stratified/by_{var}/{bin}/…
└── land/
    └── (same layout as ocean/)
```
