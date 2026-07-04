# Project Review — OCO-2 Cloud-Proximity XCO2 Bias Correction

**Date:** 2026-07-03
**Scope reviewed:** data pipeline (`workspace/demo_combined.py`, `src/pipeline/phase_03_processing.py`, `src/pipeline/phase_04_geometry.py`), spectral fitting (`src/spectral/fitting.py`), feature/ML stack (`src/analysis/build_feature_dataset.py`, `src/models/*`), analysis (`src/analysis/run_all.py`), TCCON validation chain (`curc_shell_blanca_plot_corr_xco2_deepens.sh`, `workspace/tccon_*`, `workspace/build_deepens_plot_data.py`).
**Purpose:** scientist + journal-reviewer critique, improvement plan, manuscript storyline, target journal.

---

## 1. System summary (as implemented)

- **Cloud distance:** OCO-2 glint soundings collocated with Aqua-MODIS MYD35 (byte 0 bits 1–2; Cloudy `00` + Uncertain `01` retained; night granules rejected by bit-3 majority vote). Nearest-cloud distance via ECEF KD-tree (WGS84, altitude fixed to 0), chord distance capped at 50 km; also an inverse-square `weighted_cloud_distance_km`. Temporal buffer ±10 min (year < 2022) / ±20 min (≥ 2022); post-2022 no-MODIS soundings get `-999` / `NoMODIS` sentinels.
- **Spectral features:** per sounding, per band, fit `ln T = −k1·τ + ½k2·τ² − …` (cumulant expansion) against slant optical depth. Physical meaning: `k1` = mean photon path-length enhancement, `k2` = path-length variance, intercept = ln(reflectance); κ = k1²/k2 = gamma shape. Fit orders (7, 3, 7) for O2A/WCO2/SCO2 (WCO2 capped by τ dynamic range ~0.7–1.1). Savitzky-Golay (51, 3) pre-smoothing; `curve_fit` with k1,k2 ≥ 0 bounds; edge channels dropped; T > 1 masked.
- **Target:** `xco2_bc_anomaly` = `xco2_bc(i)` − mean `xco2_bc` of same-orbit clear-sky (cld_dist > 10 km) neighbors within ±0.25° lat, kept only if reference std ≤ 1.0 ppm (`src/spectral/fitting.py:562–671`; production params at `:1066,1085`). Alternates: r05 (5 km), r15 (15 km).
- **Features (per surface):** `xco2_raw_minus_apriori`; spectral cumulants (o2a_k1/k2, wco2_k1/k2/k3, sco2_k1/k2), exp-intercept and intercept−albedo mismatches, cross-band albedo ratio; geometry (glint angle / 1/cos SZA / 1/cos VZA / sin RAA / pol angle); met/surface (log_P, dp ratios, h2o_scale, delT, co2_grad_del, tcwv, t700, fp_area); contamination block (AOD components, dp_abp, continuum/SNR, heights, dpfrac, alt_std…); fp_0–7 one-hots; optional ProfilePCA block (4 EOFs each of σ-grid T/q/CO2-prior + tropopause σ/T). `cld_dist_km` is **not** an input feature (used for labels, loss weighting, stratified reporting, optional oracle bin).
- **Model:** per-surface deep ensemble, M = 5, `n → 64 → 32 → (mu, log_var)`, production loss **beta-NLL (β = 0.5)** (CLI default is gaussian_nll — inconsistency), AdamW + OneCycle, early stop on a date-split calibration block; ensemble mixture mean/variance; split + **Mondrian conformal** 90% intervals. Splits: random (leaky), `date` (trailing block), `date_kfold` (contiguous date blocks — the defensible estimate).
- **Validation:** `curc_shell_blanca_plot_corr_xco2_deepens.sh` runs ~90 `run_case` lines / ~25 TCCON stations with 25 pooled members (5 folds × M=5), tag `de_beta_nll_prof_m5`, `--profile-pca`. Collocation: radius 100 km (primary) / 50 km, TCCON window ±60 min, 50-ppm sanity drop; correction guards (clim 50 ppm, |mu| 25 ppm) kept + flagged; train dates manually commented out. Correction: `corrected = xco2_bc − mu`.

### Headline numbers

| Metric | Before | After |
|---|---|---|
| date_kfold R² ocean / land | — | 0.526 ± 0.104 / 0.387 ± 0.071 |
| Random-split R² (TabM) — leakage artifact | 0.821 | 0.530 under date_kfold (Δ ≈ 0.29) |
| TCCON r=100 km, 70 station-days: mean \|bias\| | 1.09 ppm | 0.69 ppm |
| … RMS bias | 1.70 ppm | 0.89 ppm |
| … per-footprint RMSE to TCCON | 2.54 ppm | 1.22 ppm (improved 64/70) |
| … OCO σ (scatter) | 2.20 ppm | 0.89 ppm |
| TCCON r=50 km, 64 station-days: RMS bias | 2.88 ppm | 0.95 ppm |
| Excl. Ny-Ålesund (65 days): RMS bias | 1.30 ppm | 0.725 ppm (−44%); calib R² 0.963→0.988 |
| Pooled footprint (older 33-day set): bias / RMSE | −0.391 / 1.636 | −0.088 / 1.073 (ideal 0.583) |

Profile-EOF A/B (profile-present subset): land global R² +0.065, land near&tail-5% 0.15 → 0.45. Loss ablation: beta-NLL +1.0 tail R² land vs gaussian; student-t worse.

---

## 2. Strengths

1. **Path-length cumulants are the scientific novelty** — k1/k2 derived from the Laplace-transform view of the photon path-length PDF (`src/spectral/FITTING_DERIVATION.md`) give a mechanistic spine; `run_all.py` shows k-vs-cloud-distance phenomenology before any ML.
2. **Split-design honesty** — the documented ~0.29 R² random-split inflation and adoption of date_kfold is itself a publishable methodological finding.
3. **Independent TCCON validation prioritized over training metrics** — modest honest CV skill (0.53/0.39) yet large, consistent TCCON improvement; robustness across radius, guard handling, site exclusion.
4. Sound leakage discipline in code (fit scalers/PCA on train only), ablation machinery (`no_xco2`, `no_spec`), conformal UQ, per-surface modeling.

---

## 3. Major reviewer concerns

- **M1 — Target circularity + selection effect.** Label = OCO-2 minus OCO-2 clear-sky neighbors (both `xco2_bc`); no independent anchor. Real sub-0.25° CO2 gradients are labeled "cloud anomaly." `std_thres` drops the hardest (heavily cloudy) scenes from the label set.
  - **Suggestion:** Run the negative control explicitly: show predicted mu ≈ 0 for far-cloud clear-sky soundings and no degradation over TCCON on clear days. Bound real-gradient contamination by regressing the target against a transport-model XCO2 field (CAMS / CarbonTracker) — the unexplained-by-model fraction is the defensible "cloud" part. Report target-definition sensitivity using the existing r05/r15 alternates. Quantify per-row label noise as `ref_std/√ref_count` and cite it as the R² ceiling.
- **M2 — TCCON protocol below community standard.** No averaging-kernel / a-priori harmonization (Wunch et al. 2011; Rodgers & Connor 2003), no TCCON QC flags beyond 300–550 ppm, no altitude matching (Izaña 2.4 km), station = median obs lat/lon, single ±60 min window, only two radii. (`workspace/plot_corrected_xco2.py:119–143`.)
  - **Suggestion:** Adopt the Wunch et al. (2017) AK-harmonized protocol: adjust OCO-2 and TCCON to a common prior via Rodgers & Connor (2003) using the Lite-file averaging kernels and prior profiles. Apply GGG2020 QC flags. Drop or altitude-match high-altitude sites (Izaña). Use the fixed published station coordinates instead of median obs lat/lon. Add a coincidence-sensitivity table (radius 25/50/100 km × window ±30/60/120 min) to the supplement.
- **M3 — No significance testing.** 43/70 station-days improved in |bias| is near chance; days clustered by site (Réunion 14, Ny-Ålesund 9, Wollongong 7) violate independence. Five sites worsen in mean bias after correction (et, js, ka, or, rj).
  - **Suggestion:** Paired Wilcoxon signed-rank on station-day |bias| before vs after; site-clustered (block) bootstrap CIs on the RMS-bias reduction; pool per-site (not per-station-day) for the headline so data-rich sites don't dominate. Present the five worsening sites openly with a per-site paragraph (they mostly still improve in RMSE — say so) rather than leaving them discoverable in the CSV.
- **M4 — σ-collapse is partly tautological.** Any smoother shrinks scatter; lead with bias/RMSE to TCCON, present σ reduction as secondary.
  - **Suggestion:** Reorder headline metrics (bias/RMSE first, σ second with the caveat stated). Add a "pure smoother" null baseline — e.g., replace mu with the orbit-local running-mean anomaly — and show it collapses σ similarly but does NOT reproduce the TCCON bias/RMSE reduction. That single figure defuses the objection.
- **M5 — Cloud-distance systematics unquantified.** No parallax/cloud-height correction (all pixels at alt 0); ±10–20 min advection (~6–12+ km); Cloudy+Uncertain pooled with no sensitivity test; MYD35 "determined" bit unchecked; 1-km quantization floor.
  - **Suggestion:** Sensitivity appendix: (a) rerun distances with Cloudy-only vs pooled and show anomaly-vs-distance curves are stable; (b) bound parallax on one representative month using MYD06 cloud-top height × viewing geometry; (c) bound advection as window × climatological cloud-level wind (state the km number); (d) add the "determined"-bit check to `_unpack_cloud_mask`; (e) state the 1-km quantization floor in methods. None of these need to change the production data — bounds + one sensitivity rerun suffice.
- **M6 — `xco2_raw_minus_apriori` leakage risk.** Nearly collinear with `xco2_bc`, half the target definition.
  - **Suggestion:** Re-run the full TCCON validation chain with the `no_xco2` model and report both configurations side-by-side in the comparison table. If they agree, the concern dissolves in one row; if they diverge, promote `no_xco2` to the headline configuration and move `full` to the appendix as an upper bound.
- **M7 — Ocean near-cloud essentially unvalidated.** TCCON is land/midlat/clean-air; ocean gate is 5 km and barely probed; Ny-Ålesund (worst site: RMSE 4.67→2.35, only 4/9 improved) excluded from headlines. State plainly: TCCON validates the land correction.
  - **Suggestion:** Stratify the TCCON validation to ocean-glint footprints near island/coastal sites (Réunion, Burgos, Saga, Darwin, Izaña) and report that subset separately, even if n is small. Explore ObsPack ship/tower or AirCore profiles for ocean spot-checks. Give Ny-Ålesund a dedicated high-latitude subsection (with the snow/high-SZA discussion) instead of silent `--exclude-sites ny` — a reviewer respects an owned limitation far more than a discovered exclusion.
- **M8 — Profile-EOF benefit confounded with date.** ~51% of 2016–2020 rows lack profiles; missingness tracks processing batch (`src/models/compare_profile_features.py:14`). Land tail gains must be shown on date-matched profile-present subsets.
  - **Suggestion:** Redo the profile A/B with both arms restricted to the identical profile-present row set (same dates, same rows) so the only difference is the feature block. Better long-term: backfill the missing profiles by re-running `build_feature_dataset.py` extraction for the mid-month dates (the raw Met/CO2Prior files exist), which removes the confound entirely and grows the training set.
- **M9 — Spectral-fit details.** Savgol pre-smoothing biases k2 and discards covariance; k1,k2 ≥ 0 boundary solutions unflagged; cumulant series outside convergence radius at O2A line cores (τ→8.7); (7,3,7) tuned on one orbit; Lambertian-γ assumption weakest in glint (wind-dependent BRDF unused).
  - **Suggestion:** (a) Refit a subsample without Savitzky-Golay and report the k1/k2 shift — if small, one sentence closes the issue; (b) keep the `curve_fit` covariance and export σ(k1), σ(k2) as fit-quality features/QC; (c) flag k2-at-boundary (k2≈0) solutions and mask derived κ there; (d) redo order selection on a multi-orbit, multi-surface sample with held-out channels instead of smoothed-curve BIC; (e) add a robustness fit with the closed-form gamma model (already coded, unused) as an appendix cross-check; (f) caveat the convergence radius at O2A line cores in the text — k3+ are effective, not physical, cumulants there.

### Minor / hygiene (fix pre-submission)

- Temporal buffer threshold: 2022 (`workspace/demo_combined.py:738`) vs 2023 (`phase_03_processing.py:1540` docstring/default) — **fix:** pick one year (2022, matching the active path) and hoist it into a single shared constant both call sites import; update the docstring.
- Target-param doc drift: production 0.25°/1.0 ppm/10 km vs 0.5/2.0/10 in `build_feature_dataset.py:52` and `pipeline.py:736` — **fix:** define the production anomaly params once (constants module or config dict) and reference it from both docstrings; the manuscript methods section should quote that single source.
- CLI loss default `gaussian_nll` vs production `beta_nll` (`deep_ensemble.py:331`) — **fix:** change the CLI default to `beta_nll` (the documented adopted choice) so a bare invocation reproduces the production model; note the change in the ablation doc.
- Band-width/overlap defaults differ across three call sites (2.5/0.5 vs 10/1 vs 5/1); `process_orbit` default order (7,2,7) vs used (7,3,7) — **fix:** centralize numeric defaults in one constants module; make callers pass explicitly or import; assert the fit order in `run_simulation` equals the constant.
- Phase-3 blanket `try/except: continue` hides per-sounding failure accounting — **fix:** count and log failures per exception type; emit a `failure_summary` block in the per-granule/results JSON so silent data loss is visible in synthesis stats.
- Train-date exclusion from TCCON eval is manual comment-based — **fix:** write a training-date manifest (JSON) at training time; have the driver / `run_case` check each case date against it and refuse (or flag) overlapping cases programmatically. This also gives the paper a one-sentence leakage guarantee.

---

## 4. Prioritized improvements (highest leverage first)

1. **AK/prior harmonization of TCCON comparison** (M2) — the single change that makes the validation publication-grade; follow O'Dell et al. (2018) / Kiel et al. (2019).
2. **Significance testing with site-clustered resampling** (M3) — cheap; effect sizes likely survive, yielding a headline sentence.
3. **Negative-control experiment** (M1): predicted correction ≈ 0 for far-cloud clear sky; no degradation at TCCON on clear days.
4. **Report TCCON results for `no_xco2` feature set** alongside `full` (M6).
5. **Cloud-distance sensitivity appendix** (M5): Cloudy-only vs pooled; parallax bound via MYD06 cloud-top heights (subset); advection bound from window × typical winds.
6. **Position against prior literature**: Massie et al. (2017, 2021); Emde / Merrelli 3D-RT studies; **Mauceri, Massie & Schmidt (AMT 2023)** — closest prior work (ML correction of OCO-2 3D cloud biases). Differentiators: physically derived path-length cumulants, per-surface heteroscedastic DE + conformal UQ, larger TCCON validation. Make it a comparison table.
7. Orbit-grouped / spatially-blocked CV as a robustness row next to date_kfold.
8. **Data-recovery framing**: quantify how many quality-flagged near-cloud soundings become usable post-correction (throughput in cloudy tropics → flux-inversion impact).

---

## 5. Manuscript storyline

**Title shape:** *"Correcting cloud-proximity biases in OCO-2 XCO2 using photon path-length statistics and deep ensembles, validated against TCCON."*

1. **Motivation** — 3D cloud effects bias XCO2 near clouds at the ~1 ppm level (comparable to flux-inversion signals); QF filtering discards near-cloud soundings disproportionately in the cloudy tropics.
2. **Observable** — MODIS-collocated cloud distance, 2014–2021; raw anomaly-vs-distance decay curve (run_all.py Section 5) as the phenomenon before ML.
3. **Physics** — cumulant expansion of ln T samples the photon path-length PDF; k1/k2 respond systematically to cloud proximity (correlations, decay lengths, k2/k1 asymmetry).
4. **Model** — within-orbit clear-sky anomaly target (limitations stated openly); per-surface DE, beta-NLL, Mondrian conformal; ablations show spectral cumulants carry independent skill.
5. **Methodological interlude** — random-split R² 0.82 collapses to 0.53 under date-blocked CV; cautionary result for satellite-ML literature.
6. **TCCON validation (payoff)** — RMS station-day bias −48% (1.70→0.89 ppm), footprint RMSE −52%, improved in 64/70 station-days; robust to radius, guards; significant under site-clustered bootstrap (after fix); calibration slope/R² improves.
7. **Discussion** — land-driven validation; ocean near-cloud and polar limits; parallax/advection bounds; relation to operational B11 correction (complementary residual layer); data-recovery estimate; applicability to OCO-3 / GOSAT-GW / CO2M.

---

## 6. Target journal

**Primary: *Atmospheric Measurement Techniques* (AMT, EGU).** The OCO-2 bias-correction + TCCON-validation canon lives there (O'Dell 2018; Kiel 2019; Wunch 2017; Mauceri 2023). Right referees, open review suits a methods paper, no length pressure for sensitivity appendices.

Alternatives by framing:
- **Remote Sensing of Environment** — if emphasizing phenomenology + data-recovery impact.
- **JGR: Atmospheres** — if expanding the science interpretation of k1/k2 fields.
- **GRL** — only as a later compressed companion letter once the AMT methods paper exists.
- **AIES / Environmental Data Science** — only for an ML-methodology pivot (not recommended as the lead; the differentiator is the physics).

**Bottom line:** AMT. The two pre-submission must-fixes (AK harmonization, significance testing) are exactly what AMT referees check first; with those done, the core claim — TCCON-validated ~50% error reduction near clouds — is strong, novel, and defensible.
