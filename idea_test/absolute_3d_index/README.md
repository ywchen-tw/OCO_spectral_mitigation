# absolute_3d_index: can absolute per-footprint quantities show the near-cloud 3D effect without the clear-sky reference?

Status: exploratory, 2026-11-05. Nothing here touches `manuscript/`, `src/` or the existing results.

## Verdict

No absolute quantity shows the near-cloud effect as cleanly as its Δ version.

- **Ocean O2A ⟨l'⟩** comes closest. The absolute E_z is −0.51 on all footprints and −0.35 on the reference population. The Δ E_z is −0.66.
  - The absolute near–far difference is −0.0045. That is 1.7 times the Δ difference of −0.0026.
  - The far-window SD of the absolute value is 3.2 times the Δ SD, so between-scene variation dominates the absolute value.
  - A geometry/prior baseline (T3) brings the residual near–far difference to −0.0029, almost exactly the Δ amplitude. The residual SD is still twice the Δ SD, so E_z is −0.38 against −0.66.
- **Land O2A ⟨l'⟩** fails.
  - The absolute E_z has the wrong sign: −0.35 against Δ +0.48. Near-cloud land scenes differ climatologically from far ones.
  - After the baseline, it is +0.07 on the reference population. Adding the retrieved albedo removes it completely.
- **Land SCO2/WCO2 k1 ratio, after the baseline**, is the most promising absolute-based index on land.
  - Residual E_z is −0.29 (reference population) and −0.46 (all footprints). Δ is −0.53.
  - The baseline recovers about 70 % of the Δ near–far difference: −0.0015 against −0.0022.
  - The albedo regressor does not absorb it.
  - Without the baseline, the absolute ratio shows nothing (E_z −0.01), because composition cancels it.
  - It loses the shadow/brightening contrast that Δ has: −0.30, −0.23, −0.29 by class, against −0.72, −0.31, −0.39 for Δ.
- **f·γ/X and f·A/X** do not respond to illumination. The slab and the data agree on this.
  - Absolute f·γ/X has E_z −0.13, −0.09, −0.11 for shadowed, neutral and brightened land.
  - Its ocean near-cloud rise (E_z +0.27, against Δ +0.09) builds up gradually over about 30 km (T2). It looks like a scene-scale effect, not a local 3D one.
- **Coverage** is the one real advantage. Over land, the reference population holds only 10 % of the footprints within 5 km of cloud (219 k of 2.20 M).

## Data and definitions

- **Dates.** The 139 evaluation dates in `results_doppler/model_comparison/xgb/fm1620k5v2xt41_eval_dates/per_date_nok2diag/`. Source: `results_doppler/csv_collection/combined_<date>_all_orbits.parquet`.
- **Footprints.** 20.85 M snow-free footprints with `xco2_bc > 0` and a cloud distance. The distance is capped at 50 km, so the 20–50 km window excludes the cap.
- **Reused code** (`common.py`):
  - `analysis.utils.apply_quality_filter(require_qf0=False)`
  - `analysis.spec_sensitivity._apply_production_reference(df, 'v2')`: ocean r05 / land r15, Δ and z columns, the |anomaly| ≤ 100 ppm screen
  - `workspace/mixed_surface_reference_count.load/analyse`: the mixed-surface reference-window flag, run per date with targets out to 50 km so the far window gets the same screen
  - `analysis.land_cover.assign_land_cover`: the MCD12C1 class
  - The E_z definition matches `analysis.land_class.build_effect_sizes` exactly. Cross-check on land Δ⟨l'⟩: 0.47718 in both.
- **Populations.**
  - `ref`: finite production z_γ and anomaly, and no mixed-surface window. This is the Fig. 5 population.
  - `all`: every footprint with a cloud distance.
- **Illumination classes** (land, `ref` only): z_γ > 0.5 brightened, |z_γ| ≤ 0.5 neutral, z_γ < −0.5 shadowed. Each class is compared with two far windows:
  - "same-class far": far footprints of the same z_γ class. This matches the selection.
  - "all far": the whole `ref` far window.
- **Composite Δ** (k2/k1², the k1 ratio, f·γ/X, f·A/X): the reference value is the same combination of the stored reference means. This is first order only, as in `make_landclass_heatmap_figure.SHAPE_ROWS`. X is the footprint's own geometry.
- **E_z** = (mean over 0–5 km − mean over 20–50 km) / SD over 20–50 km, with n ≥ 500 in each window.

## T0. Rayleigh geometry X (gate for every X-based quantity)

**Convention.** The Lite `saa` and `vaa` are both azimuths as seen from the footprint.
- Glint soundings have raa ≈ 180°. cos(glint) = μ0μ − s0·s·cos(raa) reproduces `glint_angle` to 1e-4°, with correlation 1.000000.
- The scattering angle is therefore cos Θ = −μ0μ − s0·s·cos(raa).
- The stored column `cos_theta` (and `Phi_cos_theta`, `R_rs_factor`), from `analysis/build_feature_dataset.py:473`, has the opposite sign on the azimuth term. It is wrong for glint. It is not a feature of the current model; it appears only in `models/test_unused_features.py`.

X = 0.0255 (p_prior/1013.25) · 0.75(1 + cos²Θ) / (4μμ0). Checked on QF 0 footprints more than 10 km from cloud, all 139 dates (`t0_xcheck.py`, `out/t0_xcheck.csv`):

| | f·A/X median | μ0-quintile spread, f·A/X | μ0-quintile spread, f·A·μ0 | IQR/median, f·A/X vs f·A·μ0 | obs-mode spread (land), f·A/X vs f·A·μ0 |
|---|---|---|---|---|---|
| S1.6 land | 0.22 | 16 % | 52 % | 16 % vs 33 % | 7 % vs 33 % |
| here, land | **0.222** | 22.5 % (20.6 % without the pressure scaling) | 45.9 % | 18 % vs 36 % | 6.3 % vs 25 % |
| S1.6 water | 0.11 | 16 % | 47 % | — | — |
| here, water | **0.107** | **15.6 %** | **47.8 %** | 15 % vs 33 % | — |
| wrong sign (land / water) | 0.196 / 0.069 | 31 % / 65 % | — | 54 % / 43 % | 46 % |

Spread is (max − min) / median of the five quintile medians.

**Result.** The check passes:
- The water numbers reproduce S1.6. The medians match on both surfaces.
- Over land, the quintile spread is 22 % rather than 16 %. Random 6-date, 15 k-footprint subsamples give 18–26 % (interquartile range), so the 6-date S1.6 sample sits at the low end.
- The opposite sign is clearly rejected.

The X-based quantities are therefore used below.

## T1. Effect size: absolute vs Δ (`t1_t2_effect.py`)

E_z, with both QF pooled. "abs all" uses every footprint. "abs ref" and "Δ ref" use the same Fig. 5 reference footprints. "raw" is the near–far difference in native units. "SD ratio" is the far SD of the absolute value over the far SD of Δ. Analytic 95 % CIs are about ±0.002 (all), ±0.0025 (ocean ref) and ±0.01–0.017 (land ref).

| surface | quantity | E_z abs all | E_z abs ref | E_z Δ ref | raw abs ref | raw Δ ref | SD ratio |
|---|---|---|---|---|---|---|---|
| ocean | O2A ⟨l'⟩ | −0.51 | −0.35 | −0.66 | −0.0045 | −0.0026 | 3.2 |
| ocean | SCO2 k2/k1² | +0.16 | +0.13 | −0.11 | +0.0008 | −0.0003 | 2.1 |
| ocean | k1 SCO2/WCO2 | −0.23 | −0.15 | −0.51 | −0.0010 | −0.0016 | 2.1 |
| ocean | f·γ/X | +0.27 | +0.26 | +0.09 | +0.013 | +0.0008 | 5.7 |
| ocean | f·A/X | +0.12 | +0.09 | +0.17 | +0.0019 | +0.0012 | 3.0 |
| ocean | f | +0.58 | +0.42 | +0.30 | +0.0015 | +0.0002 | 4.7 |
| ocean | γ | −0.24 | −0.16 | −0.30 | −0.0092 | −0.0028 | 6.1 |
| ocean | f·γ (no X) | +0.25 | +0.24 | — | | | |
| land | O2A ⟨l'⟩ | −0.35 | −0.35 | **+0.48** | −0.0063 | +0.0023 | 3.8 |
| land | SCO2 k2/k1² | +0.35 | +0.34 | +0.18 | +0.0022 | +0.0007 | 1.6 |
| land | k1 SCO2/WCO2 | −0.05 | −0.01 | −0.53 | −0.00004 | −0.0022 | 1.7 |
| land | f·γ/X | −0.19 | −0.12 | −0.06 | −0.0076 | −0.0008 | 4.8 |
| land | f·A/X | 0.00 | +0.05 | −0.34 | +0.0025 | −0.0066 | 2.4 |
| land | f | +0.23 | +0.35 | −0.13 | +0.0013 | −0.0002 | 3.1 |
| land | γ | −0.29 | −0.25 | −0.18 | −0.024 | −0.0031 | 5.7 |
| land | f·γ (no X) | −0.15 | −0.06 | — | | | |

**QF split** (full table in `out/t1_effect_sizes.csv`):
- Ocean O2A ⟨l'⟩, QF 0: abs −0.41, Δ −0.28. QF 1: abs −0.44, Δ −1.17.
- Land O2A ⟨l'⟩, QF 0: abs −0.33, Δ +0.79. QF 1: abs −0.48, Δ +0.14.
- Land k1 ratio, QF 0: abs +0.29, Δ −0.27. QF 1: abs −0.08, Δ −0.88.

### Coverage gain (`out/t1_coverage.csv`)

| surface | window | all | ref | gain | ref share |
|---|---|---|---|---|---|
| ocean | 0–5 km | 7.49 M | 4.30 M | +3.19 M (+74 %) | 57 % |
| ocean | 0–50 km | 12.06 M | 8.75 M | +3.32 M (+38 %) | 73 % |
| land | 0–5 km | 2.20 M | 0.22 M | **+1.98 M (+903 %)** | **10 %** |
| land | 0–50 km | 5.69 M | 2.28 M | +3.41 M (+149 %) | 40 % |

- Land QF 0, 0–5 km: 874 k against 121 k (14 %).
- Land QF 1, 0–5 km: 1.33 M against 99 k (7 %).

### Land by illumination class (`out/t1_illum.csv`, both QF)

n near is 94 k shadowed, 35 k neutral and 90 k brightened.

| quantity | kind | shadowed | neutral | brightened |
|---|---|---|---|---|
| O2A ⟨l'⟩ | abs, all far | −0.51 | −0.33 | −0.20 |
| | Δ, all far | +0.11 | +0.08 | +1.02 |
| | abs, same-class far | −0.31 | −0.34 | −0.40 |
| | Δ, same-class far | +0.64 | +0.06 | +0.56 |
| k1 SCO2/WCO2 | abs / Δ (all far) | −0.01 / −0.78 | +0.04 / −0.30 | −0.02 / −0.37 |
| SCO2 k2/k1² | abs / Δ (all far) | +0.45 / +0.24 | +0.26 / +0.22 | +0.25 / +0.11 |
| f·γ/X | abs / Δ (all far) | −0.13 / −0.66 | −0.09 / +0.24 | −0.11 / +0.45 |
| f·A/X | abs / Δ (all far) | 0.00 / −1.35 | +0.02 / +0.21 | +0.12 / +0.52 |
| γ | abs / Δ (all far) | −0.39 / −1.69 | −0.25 / −0.01 | −0.10 / +1.34 |

- Absolute f·γ/X does not separate the classes. The Δ version does, but that Δ uses the first-order reference f̄·γ̄. Part of its class dependence may come from that approximation rather than from a failure of f to compensate γ.

### MCD12C1 land class (`out/t1_landclass.csv`, classes with ≥ 5 000 near footprints)

The absolute E_z does not track the Δ E_z across classes for O2A ⟨l'⟩:
- cropland: abs −0.45, Δ +2.28
- barren: abs −0.61, Δ −0.01
- forest: abs +0.01, Δ +0.51

They agree only for SCO2 k2/k1² in grassland (+0.35 / +0.36) and shrubland (+0.40 / +0.44).

## T2. Distance curves (`out/t2_distance_curves.png`, `out/t2_curves.csv`)

Medians with IQR shading in 1-km bins.

**Ocean.**
- O2A ⟨l'⟩ is flat at 0.979 beyond about 15 km. It drops sharply inside 3 km, to 0.969 at 0–1 km.
- f rises toward cloud: 0.0104 far, 0.0134 at 0–1 km.
- γ falls: 0.110 far, 0.096 at 0–1 km.
- f·γ/X climbs gradually from 0.141 at 50 km to 0.162 at 2 km. The slope is broad, not local.
- The k1 ratio drops inside 2 km, from 1.011 to 1.008.
- k2/k1² peaks at about 3 km.
- f·A/X stays flat, 0.108–0.110.

**Land.**
- O2A ⟨l'⟩ drifts down slowly over the whole 50 km, from 0.998 to 0.993. It then drops to 0.989 in the first bin. The slow drift is the composition change that flips the sign of the absolute E_z.
- f·γ/X, f·A/X and γ are flat to about 2 km. They then dip sharply in the 0–1 km bin (f·γ/X from 0.155 to 0.10).
- k2/k1² rises steadily inside 15 km.

**Land by illumination class.**
- The brightened, neutral and shadowed curves stay apart at 50 km in ⟨l'⟩, f and γ. In ⟨l'⟩ the brightened curve sits about 0.005 above the shadowed one.
- So in absolute terms the class split is mostly a selection effect, present at every distance.
- The near-cloud widening of the split is small against the IQR.

## T3. Global clear-sky baseline (`t3_baseline.py`)

**Setup.**
- XGBoost: 300 trees, depth 8, learning rate 0.05. One model per surface and quantity.
- Training: footprints more than 20 km from cloud, on half the dates. Folds are by date, 70 and 69 dates. Each half is held out once and the residuals are pooled.
- Regressors: SZA, VZA, cos Θ, glint angle, airmass, observation mode, footprint index, latitude, longitude, month, altitude and its SD, prior p_sfc, T700, plus prior wind speed (ocean) or the IGBP class (land).
- Contrast model: the same regressors plus the Lite O2A albedo.
- Held-out far-window R²:
  - O2A ⟨l'⟩: ocean 0.64 (0.73 with albedo), land 0.73 (0.80)
  - k2/k1²: 0.28–0.37
  - k1 ratio: 0.38–0.54
- A first run with early stopping was discarded: land fold 1 stopped at 6 trees (`out/t3_earlystop_run.log`).

Residual E_z, both QF (`out/t3_residual_effect.csv`). Each cell is all footprints / ref population. Δ uses the ref population only. Raw values are for the ref population.

| surface | quantity | abs E_z | resid E_z (geom) | resid E_z (+albedo) | Δ E_z | raw resid geom / +alb / Δ | far SD resid / Δ |
|---|---|---|---|---|---|---|---|
| ocean | O2A ⟨l'⟩ | −0.51 / −0.35 | −0.52 / −0.38 | −0.46 / −0.32 | −0.66 | −0.0029 / −0.0021 / −0.0026 | 0.0077 / 0.0039 |
| ocean | k2/k1² | +0.16 / +0.13 | +0.17 / +0.16 | +0.15 / +0.14 | −0.11 | +0.0008 / +0.0007 / −0.0003 | 0.0047 / 0.0027 |
| ocean | k1 ratio | −0.23 / −0.15 | −0.33 / −0.20 | −0.34 / −0.21 | −0.51 | −0.0009 / −0.0010 / −0.0016 | 0.0047 / 0.0031 |
| land | O2A ⟨l'⟩ | −0.35 / −0.35 | **+0.43** / +0.07 | +0.21 / −0.01 | +0.48 | +0.0006 / −0.0001 / +0.0023 | 0.0091 / 0.0048 |
| land | k2/k1² | +0.35 / +0.34 | +0.24 / +0.25 | +0.28 / +0.25 | +0.18 | +0.0014 / +0.0014 / +0.0007 | 0.0054 / 0.0041 |
| land | k1 ratio | −0.05 / −0.01 | **−0.46 / −0.29** | −0.45 / −0.29 | −0.53 | −0.0015 / −0.0015 / −0.0022 | 0.0053 / 0.0042 |

**QF split.**
- Ocean O2A ⟨l'⟩: QF 1 −0.72 / −0.62 (Δ −1.17); QF 0 −0.24 / −0.17 (Δ −0.28).
- Land k1 ratio: QF 1 −0.58 / −0.46 (Δ −0.88); QF 0 −0.06 / −0.06 (Δ −0.27).

Land by illumination class, residual E_z (all far / same-class far; `out/t3_illum.csv`):

| quantity | model | shadowed | neutral | brightened |
|---|---|---|---|---|
| O2A ⟨l'⟩ | geom | −0.17 / +0.13 | +0.01 / −0.01 | +0.34 / +0.07 |
| O2A ⟨l'⟩ | +albedo | +0.01 / −0.01 | −0.04 / −0.04 | −0.02 / 0.00 |
| O2A ⟨l'⟩ | Δ | +0.11 / +0.64 | +0.08 / +0.06 | +1.02 / +0.56 |
| k1 ratio | geom (+alb nearly identical) | −0.32 / −0.30 | −0.21 / −0.23 | −0.29 / −0.29 |
| k1 ratio | Δ | −0.78 / −0.72 | −0.30 / −0.31 | −0.37 / −0.39 |

**How much the albedo regressor absorbs.**
- O2A ⟨l'⟩, ocean: about 27 % of the raw residual signal (−0.0029 to −0.0021).
- O2A ⟨l'⟩, land reference population: all of it. The brightened-class residual goes from +0.34 to −0.02.
- This fits the albedo seeing the 3D-perturbed continuum.
- The CO2-band shape ratios are essentially untouched.

## T4. Slab f·γ (`t4_slab.py`, `out/t4_slab_profiles.png`, `out/t4_slab_values.csv`)

1e9-photon production run, 3D solver. The IPA stays at the far value everywhere outside the cloud: 0.00338 dark, 0.00331 bright.

| region | dark f·γ | dark f / γ | bright f·γ | bright f / γ |
|---|---|---|---|---|
| far (x < 5 km, median) | 0.00340 | 0.083 / 0.041 | 0.00334 | 0.0107 / 0.312 |
| sunlit side 5–9.5 km (mean) | 0.00341 | 0.078 / 0.044 | 0.00337 | 0.0098 / 0.343 |
| sunlit edge, x = 9.25 km | 0.00343 (+1 %) | 0.073 / 0.047 | 0.00340 (+2 %) | 0.0092 / 0.369 (γ +18 %) |
| shadow 15–22 km (mean) | 0.00279 (−18 %) | 0.186 / 0.016 | 0.00364 (+9 %) | 0.061 / 0.085 |
| shadow, min-γ column | 0.00257 at x = 18.25 (−24 %) | **0.200 (bound)** / 0.013 | 0.00368 at x = 20.75 (+10 %) | 0.086 / 0.043 |
| in cloud 9.5–14.5 km | ≈ 0 (f at its lower bound 0) | | 0.0079 | 0.013 / 0.60 |

- **Brightened (sunlit) side.** f·γ does not rise: +1 % (dark) and +2 % (bright), while γ rises 15–18 %. f falls in proportion.
- **Dark-surface shadow.** f hits its upper bound of 0.2 (`spectral.cumulant_fit._AM_F_MAX`) in 12 of the 14 shadow columns, x ≈ 15.75–21.25 km.
  - f·γ falls to 0.0026–0.0028. This is a clipping artifact: holding f·γ at 0.0034 with γ = 0.0128 would need f ≈ 0.27.
- **Bright-surface shadow.** f stays below the bound (max 0.086). f·γ rises 8–10 %, to 0.0036–0.0037.
- f·γ is close to an illumination-invariant air-path radiance. It does not flag shadows or brightening, except through the bound.

## Caveats

- **CIs.** The analytic 95 % CIs treat footprints as independent. Neighbouring footprints are strongly correlated, so the true uncertainties are several times larger. Differences in E_z below about 0.1 should not be read as real.
- **Land Δ sample sizes.** The reference population is thin near cloud. Over 0–5 km it holds 219 k footprints (QF 0: 121 k, QF 1: 99 k). The illumination classes hold 35–94 k each.
- **Composite Δ.** The Δ of the ratio and composite quantities uses the ratio of reference means (first order). The Δ f·γ/X class dependence may partly be that approximation.
- **Mixed-surface screen.** Here it is applied to the whole 0–50 km reference population, both near and far. Fig. 5 screens targets within 15 km only.
- **Illumination class in the far window.** The class is defined by z_γ in the far window as well. "Same-class far" therefore matches the selection, and "all far" mixes the class selection with the near-cloud effect. Both are given.
- **Baseline quality.** The baselines leave 2× the Δ scatter for O2A ⟨l'⟩. For k2/k1² and the k1 ratio, R² is only 0.3–0.5. A richer prior set (aerosol or humidity priors not taken from the retrieval) might narrow the gap. Retrieved AOD, TCWV and QF were excluded on purpose.
- **Stored cos_theta.** The stored `cos_theta` / `R_rs_factor` columns of the feature build have the wrong azimuth sign. This is noted only; nothing in `src/` was changed.
- **Disk.** `out/cache/` holds 6.3 GB: 139 per-date parquets plus `eval_footprints.parquet`. Delete it when finished. `python common.py` rebuilds it in a few minutes (6 processes).

## Files

- `common.py`: builds the cached footprint table, and holds the E_z helper and the class helper
- `t0_xcheck.py`: the X geometry check
- `t1_t2_effect.py`: T1 tables and the T2 figure
- `t3_baseline.py`: T3 baseline and residual E_z
- `t4_slab.py`: T4 slab profiles
- `out/`: CSVs, PNGs and logs, as listed above
