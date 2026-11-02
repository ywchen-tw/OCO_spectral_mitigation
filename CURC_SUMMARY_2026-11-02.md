# CURC progress since the 2026-10-30 hand-overs (status 2026-11-02)

This covers the CURC work after the local agent's hand-overs (telluric Doppler, two-τ H2O, agreed fit model). The code for those hand-overs is in commit `5601e4b`. Everything below came after it.

- **Storage root:** `/scratch/alpine/yuch8913/oco/spec_anal/results_doppler` (`RD` below).
- **Job scripts:** the Slurm scripts and small collectors for these runs live in the Claude session scratchpad, not in the repo (see §10).

---

## 1. Data and fit pipeline

### Doppler τ
- The τ is now built with telluric and solar Doppler shifts (stamp `transmission_ils_rayleigh_hr_v4_telluric_solar_doppler`).
- Production τ ignored the telluric shift λ_rest·(1 − v/c). That was the main cumulant-fit misfit:
  - ln T residual rms on production τ: 0.17 / 0.05 / 0.23 (O2A / WCO2 / SCO2);
  - with Doppler: 0.03 / 0.02 / 0.02.
- `abs_util/toa_solar.py` and `--toa-only` upgrade older v3 files in minutes instead of a full τ recompute.

### Fit schema `doppler_am_t2_v2` (`spectral/cumulant_fit.py`, `fitting.py`)
- Fits use only the v4 τ and raw ln T: no Savitzky–Golay and no production-order fit.
- Stored per band:
  - order 2 / order 3, two-τ (`_t2`) and the agreed model (`_am`, plus `exp_int_am`);
  - every coefficient gets a formal 1σ (`*_sig_*`);
  - clear-sky reference statistics (`ref_` / `r05_` / `r15_`).
- A fit takes about 1 min per orbit.
- **v2 land rule:** land means L1B `land_fraction >= 80`, matching the Lite `surface_type`.
  - v1 used > 90. As a result, 11,640 target-valid land soundings (80–90 % land, 54 % north of 60° N) had a NaN SCO2 x² term.
  - All 5,605 orbits were refit. `fitting_details` is now v2, and the v1 files are deleted.
  - **Check:** all 360 non-agreed-model v1 land models reproduce their held-out RMSE on v2 data (max relative difference 2e-6). So only the agreed-model land runs were retrained (prefix `fm1620k5v2`). The other land runs are symlinked under that prefix.

### Feature datasets (`analysis/build_feature_dataset.py`, redesigned)
- Reads `RD/fitting_details` and writes `RD/csv_collection/`:
  - 116 training dates;
  - `combined_2016_2020_dates.parquet` (20 GB);
  - `combined_2020_dates.parquet`;
  - the evaluation dates;
  - 110 TCCON case dates.
- The old `results/csv_collection` consumers (`results.py`, `land_class.py`, about 25 scripts) still expect the removed `*_fitting` / `*_nosg` fields. They fail loudly, by design.

### Storage
- MODIS swaths deleted (921 GB).
- Production `fp_tau_combined` files deleted (2.1 TB).
- τ and L1B are kept for refits. Never delete `fitting_details`.
- The 2014–2016 OCO-2 data moved to PetaLibrary, with symlinks.

### Missingness, 2016–2020 combined
| | rows | target NaN | kept after filtering |
|---|---|---|---|
| ocean | 10.5 M | 26.7 % | 7.73 M |
| land | 7.2 M | 47.7 % | 3.76 M |

Feature NaNs are now about 0 after the v2 refit.

---

## 2. Which fit-method coefficients carry the signal (24 dates of 2020)

- The agreed-model k1 deviation from the clear-sky reference (`d_*_k1_am`) is the strongest correlate of the anomaly target on both surfaces, with the same sign on every date:
  - ocean `d_sco2_k1_am` ρ = 0.32;
  - land `d_wco2_k1_am` ρ = 0.29.
- Weaker correlates:
  - two-τ k1: 0.26 ocean;
  - order-2/3 k1: ≤ 0.09;
  - σ and rms: ≤ 0.06;
  - cloud distance itself: 0.12 ocean.
- The ocean CO2-band k1 correlation stays at about 0.22 beyond 50 km from cloud. This first hinted that k1 also carries real CO2 amount (see §5).

---

## 3. Fit-method and feature-set comparison (5-fold date k-fold, 2016–2020)

### Folds and fit methods
- 5 folds by date (`kfold-2016-2020`).
- Fit methods compared: am (agreed model), o2, o3, t2, legacy. Models compared: Ridge, XGBoost, DE.
- **The agreed model is best for every model, surface and feature set** (36/36 for the deployment set).

### Held-out RMSE, ppm (am, before tuning)
| feature set | DE ocean | XGB ocean | Ridge ocean | DE land | XGB land | Ridge land |
|---|---|---|---|---|---|---|
| full | 0.394 | 0.404 | 0.507 | 0.543 | 0.550 | 0.674 |
| full_sig | 0.396 | 0.401 | 0.502 | 0.538 | 0.547 | 0.672 |
| no_xco2_sig | 0.397 | 0.418 | 0.535 | 0.550 | 0.558 | 0.709 |
| no_xco2 | 0.393 | 0.422 | 0.541 | 0.549 | 0.562 | 0.713 |
| no_spec | 0.424 | 0.426 | 0.555 | 0.560 | 0.562 | 0.687 |
| no_xco2_and_spec | 0.519 | 0.534 | 0.666 | 0.634 | 0.638 | 0.766 |

- **Team rule:** no XCO2-prior / dXCO2 predictor in deployment sets. `full*` sets are comparison-only.
- **Spectral uncertainties:** `*_sig` helps XGB and Ridge consistently (about −0.004). So `no_xco2_sig` was preferred over `no_xco2`.
- **2020-only runs:** dXCO2 is season-dependent and hurts in Jul–Sep.

### DE group-drop ablation (no_xco2_sig base, ΔRMSE ocean / land)
| group dropped | Δ ocean | Δ land |
|---|---|---|
| CO2-band spectral | +0.085 | +0.075 |
| both spectral | +0.122 | +0.084 |
| contamination | +0.037 | +0.029 |
| footprint cloud | +0.023 | +0.020 |
| O2A spectral | +0.011 | +0.003 |
| inhomogeneity | +0.010 | +0.001 |
| screening | +0.001 | +0.005 |
| σ | ≈ 0 | ≈ 0 |

### TabM vs DE
- TabM was tuned on 2 folds; config C won both. TabM C no_xco2_sig scores ocean 0.407 / land 0.550, against DE 0.397 / 0.550.
- **Seed spread:**
  - CV: about 0.001–0.002;
  - TCCON fp-RMSE: ±0.05 (1.46–1.57);
  - XGB in comparison: < 0.01.
- **Plume retention:** TabM 0.81, DE 0.70.

### Model decision for the JQSRT manuscript
- XGBoost in the main text, with Ridge as the baseline.
- TabM and DE go in one SI table plus a "tried other models" sentence.
- DE is dropped.

---

## 4. XGBoost tuning

- **Method:** random search with 40 configs on no_xco2_sig, 5 folds. Selection used the calibration-block RMSE inside each fold's training years. That block is also the early-stopping set, so selection is slightly optimistic.
- **Result:** c30 gave held-out RMSE ocean 0.4092 / land 0.5511. The defaults (c40) gave 0.4180 / 0.5578 and hit the 2,000-tree cap on every fold.
- **Final config c41** = c30 with up to 10,000 trees. Its ocean RMSE was 0.40919, a negligible change; only one fold needed more than 5,000 trees.

  `{"max_depth": 8, "min_child_weight": 9.23, "learning_rate": 0.0287, "subsample": 0.74, "colsample_bytree": 0.56, "reg_lambda": 0.371, "reg_alpha": 0.917, "gamma": 0.705, "n_estimators": 10000, "early_stopping_rounds": 100}`

- **New code:** `gbdt_baselines.py --xgb-params <json>` (keys checked against an allow-list), plus `reg_alpha`, `min_child_weight` and `gamma` read from the config. The selected calib RMSE is stored in `run_summary.json`.

---

## 5. Plume loss and the CO2-amount-invariant ("shape") feature set

### Problem
- The corrections removed part of real power-plant plumes: XGB no_xco2_sig kept a median 0.84 of the enhancement.
- **Cause:** the τ is built with prior CO2, so the band cumulants scale as k_n = gⁿ·κ_n. The CO2-band k1 therefore follows the true CO2 amount.
- The local clear-sky reference target labels real local excess as bias, so the model learns to remove it.
- **Evidence:**
  - TreeSHAP on Sasan (2014-10-23) puts the plume reduction on `sco2_k1_am` / `wco2_k1_am`;
  - dropping k1 restores retention to 0.98.

### Fix: `no_xco2_sig_shape` (the formal set)
- Drops the CO2-band k1 and k2.
- Adds three ratios that cancel g:
  - `sco2_k2_over_k1sq_am` (= var/⟨l⟩²);
  - `wco2_k2diag_over_k1sq_am`;
  - `sco2_over_wco2_k1_am`.
- Weak-line `k1w` and O2A terms are kept.
- New sets in `models/pipeline.py`: `no_xco2_sig_nok1`, `no_xco2_sig_nok1k1w`, `no_xco2_sig_shape`, `no_xco2_sig_plus_shape`, `full_sig_shape`, plus the `no_xco2_sig_shape_no_<group>` ablation sets.
- Adding the ratios on top of no_xco2_sig (`plus_shape`) changes nothing (−0.0003).

### Choice
The user chose `shape` as the formal version. It costs about 10 % of CV RMSE but preserves plumes, and the near-cloud TCCON bias goes to about 0.

---

## 6. Final models: tuned XGBoost c41 (prefix `fm1620k5v2xt41`)

### 6.1 Held-out CV RMSE, ppm
| feature set | ocean | land |
|---|---|---|
| no_xco2_sig | 0.409 | 0.551 |
| **no_xco2_sig_shape (formal)** | **0.448** | **0.604** |
| full_sig (dXCO2, comparison) | 0.398 | 0.540 |
| full_sig_shape (dXCO2, comparison) | 0.400 | 0.542 |
| shape before tuning | 0.465 | 0.611 |
| Ridge shape | 0.589 | 0.755 |
| TabM C shape | 0.4545 | 0.5985 |

- **Seeds 0/1/2 (shape):** ocean 0.4477 / 0.4478 / 0.4477; land 0.6037 / 0.6028 / 0.6029.
- **Fit methods (tuned no_xco2_sig):**

  | | am | t2 | o2 | o3 | legacy |
  |---|---|---|---|---|---|
  | ocean | **0.409** | 0.424 | 0.436 | 0.439 | 0.439 |
  | land | **0.551** | 0.568 | 0.575 | 0.589 | 0.582 |

- **Group-drop ablation of shape (ΔRMSE ocean / land):**

  | group dropped | Δ ocean | Δ land |
  |---|---|---|
  | contamination | +0.045 | +0.059 |
  | CO2 spectral | +0.043 | +0.022 |
  | footprint cloud | +0.025 | +0.038 |
  | inhomogeneity | +0.017 | +0.004 |
  | O2A spectral | +0.010 | +0.001 |
  | screening | +0.002 | +0.006 |

  Saved as `RD/model_comparison/feature_importance_fm1620k5v2xt41_no_xco2_sig_shape/ABLATION_xt41_shape.md`.

### 6.2 TCCON 2014–2021
183 station-days, AK-harmonised. Values are before → after, from `tccon_metrics_ak` (all QF). Near / far split: cloud distance ≤ 5 km ocean / ≤ 15 km land.

| model | footprint RMSE | mean \|station-day bias\| | near bias / RMSE | far bias / RMSE |
|---|---|---|---|---|
| uncorrected | 3.10 | 1.08 | −0.72 / 3.49 | −0.11 / 1.14 |
| XGB no_xco2_sig | 1.48 | 0.81 | −0.07 / 1.55 | −0.15 / 0.95 |
| **XGB shape** | **1.74** | **0.85** | **−0.00 / 1.87** | **−0.09 / 1.08** |
| XGB full_sig | 1.28 | 0.80 | −0.17 / 1.31 | −0.25 / 0.95 |
| XGB full_sig_shape | 1.29 | 0.80 | −0.18 / 1.33 | −0.26 / 0.95 |
| TabM C shape | 1.73 | 0.85 | −0.04 / 1.84 | −0.15 / 1.10 |
| Ridge shape | 2.82 | 1.02 | −0.52 / 3.16 | −0.08 / 1.17 |

- Footprint RMSE rewards smoothing.
- Station-day bias is almost unchanged with or without dXCO2; dXCO2 mainly cuts footprint scatter.
- The dXCO2 sets add a −0.2 ppm offset.

### 6.3 Drift TCCON 2022–2025
111 station-days, 20 sites, AK-harmonised.

| model | footprint RMSE | mean \|station-day residual\| (Δ, p) |
|---|---|---|
| uncorrected | 1.77 | 0.94 |
| XGB shape | 1.36 | 0.79 (−0.15, p = 0.03) |
| XGB no_xco2_sig | 1.17 | 0.73 |
| XGB full_sig / full_sig_shape | 1.09 | 0.74 / 0.73 |
| TabM shape | 1.37 | 0.82 (p = 0.06) |
| Ridge shape | 1.70 | 0.95 |

- The improvement holds for 2023+ and without Ny-Ålesund.
- **GEOS split for XGB shape:** FP-IT −0.18 (p = 0.13); GEOS-IT −0.12 (p = 0.05).

### 6.4 Plume tests
- 28 pairs on 25 dates. "Real plume" means closest pass ≤ 12 km and xco2_bc enhancement ≥ 0.5 ppm.
- Retention = median fraction of the enhancement kept, for a 10 km core / 20–60 km background (or a 15 km core / 30–100 km background).

| | without dXCO2 | with dXCO2 |
|---|---|---|
| CO2-band k1/k2 | no_xco2_sig 0.84 / 0.75 | full_sig 0.81 / 0.77 |
| shape ratios | **shape 0.99 / 0.91** | full_sig_shape 0.81 / 0.80 |

- **Shape restores plumes only without dXCO2.** dXCO2 contains the plume itself.
- **XGB shape vs same-date pseudo-sources:** not distinguishable with the 10 km core (Wilcoxon p = 0.41). All k1 and dXCO2 models remove plume signal (p < 0.003).
- **Per case, XGB shape:** Sasan 104 %, Kozienice 2021-09-06 101 %, Ghent 90 %, Westar 80 % (only a +0.3 ppm enhancement).
- **Control-region null:** no XGB model is flagged. corr(xco2_bc, μ): shape 0.61, others 0.81–0.86.
- **Mauceri et al. cases:** Ghent is kept; Westar is too small to judge; Matimba can't be judged (nearest footprint 23 km away).
- **Emission rates** (Gaussian plume vs EPA CAMPD, 100 m wind, before tuning):
  - Ghent Q/Q_rep: xco2_bc 0.73, shape 0.71, no_xco2_sig 0.62;
  - Westar: 0.80 / 0.80 / 0.66;
  - not conclusive (wind uncertainty 20–27 %).
  - Moved to the local agent: see `workspace/Nassar_plume_analysis/TODO_plume_emission_test.md`.
  - The Kozienice emission test is dropped.

### 6.5 Evaluation dates 2014–2021, recomputed corrected anomaly
- 133 dates, no training dates. Output: `RD/model_comparison/xgb/fm1620k5v2xt41_eval_dates/`.
- Within the reference radius, QF0:

  | | shape | no_xco2_sig |
  |---|---|---|
  | ocean mean bias removed | 69 % | 74 % |
  | ocean RMS reduction | 32 % | 43 % |
  | land mean bias removed | 33 % | 39 % |
  | land RMS reduction | 20 % | 32 % |

- The spread gap mirrors the CV gap, because this metric uses the training target's reference. Removing the mean near-cloud bias is close between the two.
- **Weak spot:** 2014 ocean, where shape removes only 38 % of the bias. `pol_ang_rad` is out of the training range on two dates.

### 6.6 Attribution (tuned shape)
- **Permutation, per feature:**
  - ocean is led by `wco2_k1w_am` +0.62, `alb_sco2_over_wco2` +0.32, `sco2_k1w_am` +0.28;
  - land is led by dust AOD +0.19, dust height +0.15, stratospheric aerosol +0.07.
- **TreeSHAP, land:**
  - footprint cloud/aerosol is the largest group (0.26 ppm mean |SHAP|), but its net push is about 0;
  - the spectral group gives the signed near-cloud push: −0.24 ppm at < 2 km, fading to about 0 beyond 20 km;
  - far from cloud, every group's net push is about 0.01 ppm or less.
- **Bug fixed 2026-11-02:** `feature_importance.assign_groups` put the shape ratios in `met_other`. They are now in `spec`. The permutation and SHAP group tables for xt41 shape were rerun.
- The conformal intervals and the ocean SHAP table come from that rerun and are still pending.

### 6.7 Manuscript figures
- Regenerated from XGB shape into `RD/manuscript_figures_xgb_shape/`: TCCON, smoother null, drift, plumes and Mauceri panels, anomaly decay. Feature-importance figures and tables are pending.
- Six figure scripts got `--model-label` / `--out-dir`-type options. Their defaults reproduce the DE figures.
- These figures stay DE-only, because they need DE σ or DE-only models:
  - the failure-mode analysis;
  - the footprint-area residual step;
  - the spectral-variant plume comparison;
  - the σ-based uncertainty figures.
- **Fonts:** CURC has neither Arial nor Times. For final typography, re-render with `fonts/`.

---

## 7. Side study: non-ML direct XCO2 from the cumulants (PPDF), paused

- Folder: `idea_test/ppdf_direct_xco2/`. Read `PLAN.md` → "Status (2026-11-01)".
- **Round 2:** a pooled path-model blend explained about 39 % of bc − prior out of sample (far-cloud error 0.78 vs 1.07 ppm for the prior alone).
- **Physical rounds 3–5 all failed the O2A truth check** (O2A-implied path scale rsd 0.6–1.0 %, needed ≤ 0.3 %).
- **Conclusion:** within one band, g is degenerate with the PPDF scale. Light in O2A, WCO2 and SCO2 does not share photon paths: the SCO2/WCO2 k1 ratio has a 1 % offset and 0.6 % scatter. The limit is forward-model path physics, not τ resolution.

---

## 8. Code changes in this commit (summary)

- **`src/spectral/`:** Doppler-only fit schema v2, `sig_*` for every coefficient, land rule ≥ 80, `--fit-subdir`, plots trimmed.
- **`src/analysis/build_feature_dataset.py`:** reads `results_doppler/fitting_details` and the new field names.
- **`src/models/pipeline.py`:**
  - `--fit-method {am,o2,o3,t2,legacy}`;
  - new feature sets: sig, shape, nok1, ablation;
  - derived shape ratios (`_safe_ratio`);
  - `CO2_SHAPE_FEATURES`.
- **`src/models/gbdt_baselines.py`:** `--xgb-params`, extra XGB hyperparameters, calib RMSE in `run_summary`, and `--feature_set` choices from `_FEATURE_SETS`. Same change to `linear_baseline.py`.
- **All trainers** (`deep_ensemble`, `tabm`, `xgb`, `mlp_baseline`, `geom_mlp`, `structured_*`, `xgb_cloud_classifier`, `profile_pca_kfold`):
  - `--fit-method`;
  - data read from `results_doppler/csv_collection`, one surface at a time, with the analysis-only reference columns skipped;
  - outputs go to `model_<x>/<fit_method>/<suffix>`.
- **`tabm_eval.py`:** the missing `tqdm` / `gc` imports.
- **`src/models/feature_importance.py`:** v2 loaders; shape ratios grouped as `spec`.
- **`workspace/`:**
  - figure scripts with model-label / out-dir options;
  - `collect_fit_method_2020.py`;
  - `fit_anomaly_corr_2020.py`;
  - `tccon_correction_policy_stats.py`, `test_set_anomaly_decay.py` updates;
  - `TODO_plume_emission_test.md`.
- **Tests:** `tests/test_doppler_fit_output.py`, plus an updated `test_fitting_orbit_sharing.py`.
- **Repo cleanup:** removed the empty `phd_desertation` gitlink.

---

## 9. Open items

1. **Pending on CURC:**
   - xt41 shape SHAP and conformal tables, then the feature-importance figures (`ms_xgb_fi`);
   - the TabM permutation aggregate and the last TabM ablation task (SI only).
2. **Local agent:** plume emission-rate test (CDS plume-height winds, more US CAMPD cases). See the TODO file.
3. **Manuscript choices still open:**
   - which per-case TCCON maps to show;
   - display names for the `cg_fpcloud`, `cg_inhom` and `cg_screen` groups in the FI figures.
4. **Not done:**
   - ATom on the Doppler fits (only 3 of 8 dates have fits);
   - redesigning the legacy `results/` consumers;
   - the 48 deferred non-2016–2020 event dates.

## 10. Where the job scripts are

The Slurm scripts and small collectors are in the Claude session scratchpad (`/projects/yuch8913/code-server/tmp/claude-777013/-projects-yuch8913-OCO-spectral-mitigation/dbbff218-…/scratchpad/`):

| purpose | scripts |
|---|---|
| final XGB training | `fm1620k5v2_xgb41.sh` |
| tuning configs and summary | `xgb_tune/c*.json`, `xgb_tune/tune_summary.csv` |
| SHAP and conformal | `xgb_extras.py` |
| permutation | `perm_xgb41_shape.sh` |
| drift and plume summaries | `drift_plume/{drift,plume}_summary_final.py` |
| evaluation-date anomalies | `eval_dates/` |
| figure builder | `figures_xgb/` |
| job registry | `fm_arrays.txt` |

That directory is session-specific. Copy what should be kept into the repo before it is cleaned up.
