# Project Review — OCO-2 Cloud-Proximity XCO2 Bias Correction

**Date:** 2026-07-03
**Scope reviewed:** data pipeline (`workspace/demo_combined.py`, `src/pipeline/phase_03_processing.py`, `src/pipeline/phase_04_geometry.py`), spectral fitting (`src/spectral/fitting.py`), feature/ML stack (`src/analysis/build_feature_dataset.py`, `src/models/*`), analysis (`src/analysis/run_all.py`), TCCON validation chain (`curc_shell_blanca_plot_corr_xco2_deepens.sh`, `workspace/tccon_*`, `workspace/build_deepens_plot_data.py`).
**Purpose:** scientist + journal-reviewer critique, improvement plan, manuscript storyline, target journal.
**Updated 2026-07-04:** discussion outcomes folded in — per-footprint selling point added to Strengths; resolution plans added under M1, M2, M5, M6, M7; M8 marked resolved. Added §7 (engineering improvements: download stability, processing efficiency/memory, ML training structure, code clarity) from a code scan.
**Updated 2026-07-04 (implementation, commit `7809fe9`):** §7.1 stability fixes, §7.2 (full), §7.3 Phases 0/1/2/4, and §7.4 constants module are IMPLEMENTED and verified — see the status blocks in each subsection. Four of the six §3-Minor hygiene items are closed by `src/constants.py` + the `--loss` default fix; the training-date manifest is written per run (TCCON `run_case` check still to wire). Outstanding: §7.1 efficiency (parallel downloads, listing memoization), §7.3 Phase 3 ablation (launcher ready: `curc_shell_blanca_de_reg_ablation.sh`) + Phase 5 verification, §3-Minor Phase-3 failure accounting + TCCON `run_case` manifest check. §7.4 dead-code/file-split hygiene RESOLVED 2026-07-04 (fitting.py → 4 submodules + facade, demo_combined.py → utils + phase runners; both verified bit-for-bit; pickle alias kept deliberately — see §7.4).

---

## 1. System summary (as implemented)

- **Cloud distance:** OCO-2 glint soundings collocated with Aqua-MODIS MYD35 (byte 0 bits 1–2; Cloudy `00` + Uncertain `01` retained; night granules rejected by bit-3 majority vote). Nearest-cloud distance via ECEF KD-tree (WGS84, altitude fixed to 0), chord distance capped at 50 km; also an inverse-square `weighted_cloud_distance_km`. Temporal buffer ±10 min (year < 2022) / ±20 min (≥ 2022); post-2022 no-MODIS soundings get `-999` / `NoMODIS` sentinels.
- **Spectral features:** per sounding, per band, fit `ln T = −k1·τ + ½k2·τ² − …` (cumulant expansion) against slant optical depth. Physical meaning: `k1` = mean photon path-length enhancement, `k2` = path-length variance, intercept = ln(reflectance); κ = k1²/k2 = gamma shape. Fit orders (7, 3, 7) for O2A/WCO2/SCO2 (WCO2 capped by τ dynamic range ~0.7–1.1). Savitzky-Golay (51, 3) pre-smoothing; exact linear least-squares solve with k1,k2 ≥ 0 (lstsq + BVLS fallback; replaced iterative `curve_fit` 2026-07-04, same optimum ×14 faster); edge channels dropped; T > 1 masked.
- **Target:** `xco2_bc_anomaly` = `xco2_bc(i)` − mean `xco2_bc` of same-orbit clear-sky (cld_dist > 10 km) neighbors within ±0.25° lat, kept only if reference std ≤ 1.0 ppm (`src/spectral/fitting.py:562–671`; production params at `:1066,1085`). Alternates: r05 (5 km), r15 (15 km).
- **Features (per surface):** `xco2_raw_minus_apriori`; spectral cumulants (o2a_k1/k2, wco2_k1/k2/k3, sco2_k1/k2), exp-intercept and intercept−albedo mismatches, cross-band albedo ratio; geometry (glint angle / 1/cos SZA / 1/cos VZA / sin RAA / pol angle); met/surface (log_P, dp ratios, h2o_scale, delT, co2_grad_del, tcwv, t700, fp_area); contamination block (AOD components, dp_abp, continuum/SNR, heights, dpfrac, alt_std…); fp_0–7 one-hots; optional ProfilePCA block (4 EOFs each of σ-grid T/q/CO2-prior + tropopause σ/T). `cld_dist_km` is **not** an input feature (used for labels, loss weighting, stratified reporting, optional oracle bin).
- **Model:** per-surface deep ensemble, M = 5, `n → 64 → 32 → (mu, log_var)`, production loss **beta-NLL (β = 0.5)** (CLI default fixed to beta_nll 2026-07-04), AdamW + OneCycle, early stop on a date-split calibration block; ensemble mixture mean/variance; split + **Mondrian conformal** 90% intervals. Splits: random (leaky), `date` (trailing block), `date_kfold` (contiguous date blocks — the defensible estimate).
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
5. **Per-footprint, cloud-info-free deployment (key selling point — state in abstract).** The production model uses no cloud information and no neighboring-footprint information at inference: `cld_dist_km` enters only label construction, loss weighting, and evaluation stratification. The correction therefore runs sounding-by-sounding, is deployable without imager collocation, is immune to the post-2022 Aqua free-drift NoMODIS gap, and directly differentiates this work from the MODIS-dependent Massie/Mauceri line. Any proposed model change must preserve per-footprint independence (sequence/context models are diagnostics only — see the ML ideas note §2.1).

---

## 3. Major reviewer concerns

- **M1 — Target circularity + selection effect.** Label = OCO-2 minus OCO-2 clear-sky neighbors (both `xco2_bc`); no independent anchor. Real sub-0.25° CO2 gradients are labeled "cloud anomaly." `std_thres` drops the hardest (heavily cloudy) scenes from the label set.
  - **Suggestion:** Run the negative control explicitly: show predicted mu ≈ 0 for far-cloud clear-sky soundings and no degradation over TCCON on clear days. Bound real-gradient contamination by regressing the target against a transport-model XCO2 field (CAMS / CarbonTracker) — the unexplained-by-model fraction is the defensible "cloud" part. Report target-definition sensitivity using the existing r05/r15 alternates. Quantify per-row label noise as `ref_std/√ref_count` and cite it as the R² ceiling.
  - **Resolution plan (2026-07-04) — test cases for "correction doesn't remove real CO2 signal":**
    - **Power-plant plumes under clear sky:** use the Nassar et al. (2017, 2021) OCO-2 overpass lists (Bełchatów, Ghent, US plants); select QF0 soundings with `cld_dist_km` > 20–30 km crossing the plume; show `xco2_bc` rises 1–3 ppm while predicted mu stays flat.
    - **Urban enhancements:** LA basin overpasses (Caltech TCCON in-plume vs Edwards desert background); OCO-2/OCO-3 target/SAM city acquisitions.
    - **EaR3T OSSE (strongest):** inject a CO2 enhancement into clear-sky synthetic spectra with our own 3D-RT toolbox (Chen et al. 2025) and verify fitted k1/k2 invariance rigorously.
    - **Physics nuance to quantify first:** τ (SOD) uses the *prior* CO2 profile, so a real plume shifts `wco2_k1`/`sco2_k1` up by ~ΔCO2/CO2 (~1% for 4 ppm) while `o2a_k1` stays exactly flat (O2 insensitive to CO2). That contrast is itself the test: show o2a_k1 flat, CO2-band k1 shift small and consistent with the enhancement, then compute ∂mu/∂k1 to bound how much plume the model could wrongly "correct away" (expected negligible vs cloud-induced k1 variation — but report the number).
- **M2 — TCCON protocol below community standard.** No averaging-kernel / a-priori harmonization (Wunch et al. 2011; Rodgers & Connor 2003), no TCCON QC flags beyond 300–550 ppm, no altitude matching (Izaña 2.4 km), station = median obs lat/lon, single ±60 min window, only two radii. (`workspace/plot_corrected_xco2.py:119–143`.)
  - **Suggestion:** Adopt the Wunch et al. (2017) AK-harmonized protocol: adjust OCO-2 and TCCON to a common prior via Rodgers & Connor (2003) using the Lite-file averaging kernels and prior profiles. Apply GGG2020 QC flags. Drop or altitude-match high-altitude sites (Izaña). Use the fixed published station coordinates instead of median obs lat/lon. Add a coincidence-sensitivity table (radius 25/50/100 km × window ±30/60/120 min) to the supplement.
  - **Explanation (2026-07-04) — what AK harmonization means:** OCO-2 and TCCON each report XCO2 pulled toward *their own prior profile* and weighted by *their own averaging kernel* (OCO-2's varies with pressure/SZA; TCCON's differs). When the true profile deviates from the priors (boundary-layer enhancement, stratospheric intrusion), the two instruments report different XCO2 even with perfect measurements — "smoothing error," typically 0.1–0.5 ppm, structured in SZA/season/latitude — the same order as our bias improvements. Fix is mechanical: use Lite-file `xco2_averaging_kernel`, `pressure_weight`, and prior profile plus the TCCON prior to adjust both to a common basis (Rodgers & Connor 2003; Wunch et al. 2017 appendix).
  - **Key insight — improvement metrics are AK-invariant:** the AK/prior mismatch is *identical* for `xco2_bc` and `xco2_bc − mu` at the same sounding, so it cancels in every before-vs-after difference. Δbias, ΔRMSE, and 64/70 improved are unaffected; only *absolute* bias numbers (e.g., "0.69 ppm after") shift. Implement AK adjustment for the absolute numbers, and state the invariance explicitly in the paper — that sentence defuses the whole objection.
- **M3 — No significance testing.** 43/70 station-days improved in |bias| is near chance; days clustered by site (Réunion 14, Ny-Ålesund 9, Wollongong 7) violate independence. Five sites worsen in mean bias after correction (et, js, ka, or, rj).
  - **Suggestion:** Paired Wilcoxon signed-rank on station-day |bias| before vs after; site-clustered (block) bootstrap CIs on the RMS-bias reduction; pool per-site (not per-station-day) for the headline so data-rich sites don't dominate. Present the five worsening sites openly with a per-site paragraph (they mostly still improve in RMSE — say so) rather than leaving them discoverable in the CSV.
- **M4 — σ-collapse is partly tautological.** Any smoother shrinks scatter; lead with bias/RMSE to TCCON, present σ reduction as secondary.
  - **Suggestion:** Reorder headline metrics (bias/RMSE first, σ second with the caveat stated). Add a "pure smoother" null baseline — e.g., replace mu with the orbit-local running-mean anomaly — and show it collapses σ similarly but does NOT reproduce the TCCON bias/RMSE reduction. That single figure defuses the objection.
- **M5 — Cloud-distance systematics unquantified.** No parallax/cloud-height correction (all pixels at alt 0); ±10–20 min advection (~6–12+ km); Cloudy+Uncertain pooled with no sensitivity test; MYD35 "determined" bit unchecked; 1-km quantization floor.
  - **Suggestion:** Sensitivity appendix: (a) rerun distances with Cloudy-only vs pooled and show anomaly-vs-distance curves are stable; (b) bound parallax on one representative month using MYD06 cloud-top height × viewing geometry; (c) bound advection as window × climatological cloud-level wind (state the km number); (d) add the "determined"-bit check to `_unpack_cloud_mask`; (e) state the 1-km quantization floor in methods. None of these need to change the production data — bounds + one sensitivity rerun suffice.
  - **Resolution (2026-07-04) — parallax/cloud-height correction declared out of scope; agreed.** The anomaly-vs-distance trend is already clear over both surfaces, and no correction is needed — only two pre-emptive sentences in the paper: (1) parallax and advection add O(1–10 km) *noise* to cloud distance, which can only **blur** the anomaly–distance relationship, not create it, so the observed decay if anything underestimates the true sharpness; (2) since cloud distance is never a model input, these errors touch only the label's clear-sky reference selection (where the 10 km threshold exceeds typical displacement) and the evaluation binning. The Cloudy-only vs pooled rerun stays on the optional list (one flag change, one figure).
- **M6 — `xco2_raw_minus_apriori` leakage risk.** Nearly collinear with `xco2_bc`, half the target definition.
  - **Suggestion:** Re-run the full TCCON validation chain with the `no_xco2` model and report both configurations side-by-side in the comparison table. If they agree, the concern dissolves in one row; if they diverge, promote `no_xco2` to the headline configuration and move `full` to the appendix as an upper bound.
  - **Resolution (2026-07-04) — `no_xco2` performs worse than `full` in CV: that is expected and okay.** `xco2_raw − apriori` legitimately *contains* the retrieval error being hunted, so removing it must cost skill. The leakage question is not "does full beat no_xco2 in CV" but "does full's advantage survive independent validation." Decisive check: run the `no_xco2` model through the TCCON chain once — if `full` also wins (or no_xco2 doesn't close the gap) on TCCON bias/RMSE, the feature carries real signal and `full` stays production; report both in one table row. Precedent to cite: the operational B11 correction itself uses retrieval-internal predictors (dp, co2_grad_del), and Keely et al. (2023) do the same in the ML setting. Only "full wins CV but ties on TCCON" would indicate leakage.
- **M7 — Ocean near-cloud essentially unvalidated.** TCCON is land/midlat/clean-air; ocean gate is 5 km and barely probed; Ny-Ålesund (worst site: RMSE 4.67→2.35, only 4/9 improved) excluded from headlines. State plainly: TCCON validates the land correction.
  - **Suggestion:** Stratify the TCCON validation to ocean-glint footprints near island/coastal sites (Réunion, Burgos, Saga, Darwin, Izaña) and report that subset separately, even if n is small. Explore ObsPack ship/tower or AirCore profiles for ocean spot-checks. Give Ny-Ålesund a dedicated high-latitude subsection (with the snow/high-SZA discussion) instead of silent `--exclude-sites ny` — a reviewer respects an owned limitation far more than a discovered exclusion.
  - **Resolution plan (2026-07-04) — ocean validation data, ranked by practicality:**
    1. **TCCON ocean-glint stratification (do first, free):** split existing collocations by `sfc_type==0` glint footprints within the radius. Best ocean-like sites: **Ascension Island** (8°S mid-Atlantic, the most ocean-dominated TCCON site — check it is in the run_case list), Réunion, Burgos, Darwin, Saga, Wollongong; Izaña only with the altitude caveat.
    2. **Shipborne EM27/SUN columns (direct ocean XCO2):** Klappenbach et al. 2015 (AMT — R/V Polarstern Atlantic transect, built to validate satellite glint XCO2 over ocean) and Knapp et al. 2021 (R/V Sonne Pacific transect). Data archived on PANGAEA; reuse tccon_collocate with a moving "station."
    3. **ATom aircraft pseudo-columns (2016–2018):** four seasonal Pacific/Atlantic circuits of 0.2–12 km profiles; build XCO2 pseudo-columns with stratospheric extension and compare to ocean-glint overpasses (published precedent; overlaps our data years).
    4. **NOAA AirCore:** true columns but continental launch sites (Colorado, Sodankylä, Traînou) — land cross-check only.
    5. **ObsPack ship lines:** surface-only, validates a column product only through a transport model — skip unless a reviewer insists.
    Items 1+2 suffice to turn "ocean unvalidated" into "ocean validated at reduced n, consistent with land."
- **M8 — Profile-EOF benefit confounded with date.** ~~51% of 2016–2020 rows lack profiles~~ **RESOLVED (2026-07):** the mid-month reprocessing backfilled all profiles — `combined_2016_2020_dates.parquet` is 100% profile-complete (116 dates, 17.77M rows), and the A/B rerun on full-data ProfilePCA transformers reproduced the gains unchanged (ocean ΔR² +0.076, land +0.056). The date confound no longer exists; stale claims remain in `compare_profile_features.py:14` comments and old per-date parquets — clean those up so a reviewer doesn't rediscover the retracted caveat.
- **M9 — Spectral-fit details.** Savgol pre-smoothing biases k2 and discards covariance; k1,k2 ≥ 0 boundary solutions unflagged; cumulant series outside convergence radius at O2A line cores (τ→8.7); (7,3,7) tuned on one orbit; Lambertian-γ assumption weakest in glint (wind-dependent BRDF unused).
  - **Suggestion:** (a) Refit a subsample without Savitzky-Golay and report the k1/k2 shift — if small, one sentence closes the issue; (b) keep the `curve_fit` covariance and export σ(k1), σ(k2) as fit-quality features/QC; (c) flag k2-at-boundary (k2≈0) solutions and mask derived κ there; (d) redo order selection on a multi-orbit, multi-surface sample with held-out channels instead of smoothed-curve BIC; (e) add a robustness fit with the closed-form gamma model (already coded, unused) as an appendix cross-check; (f) caveat the convergence radius at O2A line cores in the text — k3+ are effective, not physical, cumulants there.

### Minor / hygiene (fix pre-submission)

- ~~Temporal buffer threshold: 2022 (`workspace/demo_combined.py:738`) vs 2023 (`phase_03_processing.py:1540` docstring/default)~~ **FIXED (2026-07-04):** `constants.AQUA_FREE_DRIFT_YEAR = 2022` (the active path); both call sites import it and the phase_03 docstring references the constant.
- ~~Target-param doc drift: production 0.25°/1.0 ppm/10 km vs 0.5/2.0/10~~ **FIXED (2026-07-04):** `constants.ANOMALY_LAT_THRES_DEG/STD_THRES_PPM/MIN_CLD_DIST_KM = 0.25/1.0/10` chosen (production values, used by every call site; 0.5/2.0 were stale unused defaults); function defaults in `fitting.py`, `fitting_with_ring_effect.py`, `models/pipeline.py` now import them. The manuscript methods section should quote `src/constants.py`.
- ~~CLI loss default `gaussian_nll` vs production `beta_nll` (`deep_ensemble.py:331`)~~ **FIXED (2026-07-04):** CLI default is `beta_nll` (§7.3 Phase 0).
- ~~Band-width/overlap defaults differ across three call sites (2.5/0.5 vs 10/1 vs 5/1); `process_orbit` default order (7,2,7) vs used (7,3,7)~~ **FIXED (2026-07-04):** `constants.CLOUD_DIST_BAND_WIDTH_DEG/OVERLAP_DEG = 2.5/0.5` and `constants.FIT_ORDER = (7,3,7)`; all call sites (both demos, phase_04 default, `process_orbit`, `run_simulation`) import them.
- Phase-3 blanket `try/except: continue` hides per-sounding failure accounting — **fix:** count and log failures per exception type; emit a `failure_summary` block in the per-granule/results JSON so silent data loss is visible in synthesis stats.
- Train-date exclusion from TCCON eval is manual comment-based — **HALF-FIXED (2026-07-04):** every `deep_ensemble` run now writes `training_dates.json` (train/calib/held date lists) alongside its checkpoints (§7.3 Phase 1). **Still to do:** have the TCCON driver / `run_case` check each case date against the manifest and refuse (or flag) overlaps programmatically — that check is the paper's one-sentence leakage guarantee.

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

---

## 7. Engineering improvements (added 2026-07-04)

Separate from the reviewer-facing concerns above: code-level suggestions in four areas, from a targeted scan of the download, processing, and ML code. None change scientific results; they buy reliability, wall-clock, RAM, and maintainability — worth doing before the pre-submission reprocessing runs so the reruns are cheap.

### 7.1 Data download (`workspace/demo_combined.py`, `src/pipeline/phase_02_ingestion.py`) — stability, efficiency, memory

**Stability items IMPLEMENTED (2026-07-04).** `_download_file` now streams to `<name>.part` and publishes via `os.replace()` only after the byte count matches `Content-Length` (or the `Content-Range` total); mid-stream failures and short reads retry up to 3× with `Range: bytes=<partial>-` resume; `OSError` (disk full) is caught and the partial removed; `_check_file_exists_remote` returns `(bool, float)` on every path; `_check_day_night_from_api` goes through `_get_with_retry` (3 attempts); MODIS existing files get a new `_is_readable_hdf4` probe (pyhdf open + dataset listing, mirroring the OCO-2 HDF5 probe) with delete-and-redownload on failure. Verified with stub-session functional tests (happy path, short response rejected, Range resume, Range-ignored restart, stale-`.part` discard) plus a real vs truncated MYD35 file; the 6 existing resilience tests pass. Efficiency items (parallel downloads, listing memoization) remain open.

**Stability (highest priority — these cause the silent-corruption bugs already seen, cf. fix #8):**
- **No atomic writes:** `_download_file` streams directly into the final `output_path` (`phase_02_ingestion.py:382-398`); an interrupted run leaves a truncated file at the real destination. Fix: download to `<name>.part`, `os.rename()` on success.
- **No completeness verification:** the downloaded byte count is never compared to the `Content-Length` header, and no checksum is checked — a short response returns `success=True` (`:376-398`). MODIS existing-file check is `exists()` + size only (`:1375-1400`), unlike the OCO-2 path which validates HDF5 readability (`:989-996`); a truncated `.hdf` from a prior crash is silently accepted forever. Fix: size-vs-header check in `_download_file`; extend the readability probe to MODIS.
- **No resume:** every retry restarts from byte 0, though `Range` requests are already used for existence checks (`:300` vs `:382-386`). Add `Range: bytes=<partial>-` on retry.
- **Error-handling gaps:** `_download_file` catches only `RequestException`, so an `OSError` during `f.write` (disk full) aborts the whole run leaving a partial file (`:400-402`); `_check_file_exists_remote` has fall-through paths returning `None` that break tuple unpacking in the dry-run branch (`:275-322`, `:343`); `_check_day_night_from_api` bypasses `_get_with_retry`, so one transient 5xx forces an unnecessary full download (`:1232-1234`).

**Efficiency:**
- Downloads are strictly serial across independent granules (OCO-2, then MYD35, then MYD03; `:1742-1818`) — throughput is latency-bound. A small `ThreadPoolExecutor` (4–6 workers, downloads are I/O-bound) would cut Phase-2 wall-clock several-fold.
- Directory listings are re-fetched per granule/product with no memoization: `find_modis_granules` runs per OCO-2 granule and again per MODIS product (`:1762-1774`); the same date's Met/CO2Prior GES DISC listing is re-downloaded for every orbit (`:1161-1194`, `:465-607`); the version × data-token URL candidate expansion issues multiple sequential GETs even when the first listing already has the file (`:441-463`, `:486-489`). Fix: an in-run `dict` cache keyed by listing URL.

**Memory:** the download path itself is already sound — files stream via `iter_content` and only lightweight metadata is accumulated; no change needed (the big `Cloud_Mask` full-array read at `demo_combined.py:769` belongs to Phase 3, see 7.2).

### 7.2 Processing pipeline (`src/spectral/fitting.py`, `src/analysis/build_feature_dataset.py`) — efficiency, memory

**IMPLEMENTED (2026-07-04).** All items below are in the code, verified numerically against the pre-change implementation on orbit 22845a_GL (2018-10-18): fit stage ×14 faster (curve_fit → exact lstsq/BVLS; k1 agreement ~1e-7 relative, residuals ≤ old in every worst case), `load_orbit_data` 20.3→3.3 s, per-orbit arrays float32, `fitting_details.h5` float32+gzip (~5× smaller), per-date parquet old-vs-new identical (403 columns within float32 rounding), multi-date combine streams via ParquetWriter (bit-exact vs pd.concat). New `--fit-workers` CLI (default auto = cores−1) parallelizes the fit stage with identical results.

**CPU (biggest wins first):**
- The core fit is a serial Python loop over all soundings × 3 bands, each calling `curve_fit` — twice when `dual_fit` (`fitting.py:990-1033`) — and orbits are also processed serially (`:1835-1845`). This is the dominant wall-clock cost and embarrassingly parallel: a `multiprocessing.Pool` over orbits (or chunks of soundings) is the single highest-leverage change for CURC runs.
- The model is linear in its parameters (polynomial in τ), yet fitted with iterative Levenberg–Marquardt per sounding (`:507-545`). A batched `np.linalg.lstsq` on the design matrix — `get_design_matrix` at `:433` already exists, unused — with the k1,k2 ≥ 0 constraint handled as a post-check/refit would vectorize across soundings and remove per-sounding solver overhead entirely.
- Redundant work in the loop: `argsort` + `savgol_filter` recomputed per sounding per band, and the smoothed/unsmoothed dual-fit paths re-sort and re-fit independently (`:528-532`, `:1008`, `:1019`); the `l1b_index` dict is rebuilt with a nested Python loop per orbit (`:363-372`); per-footprint `pyproj` transform + shapely area run in a Python loop although `Transformer.transform` accepts arrays (`:1159-1161`).
- `build_feature_dataset.py:97-98` — σ-grid profile interpolation is a per-sounding Python loop (`_interp_profile_to_sigma` re-doing `asarray`/`argsort`/`interp` per row); vectorizable across soundings.

**Memory:**
- `fitting.py` holds five simultaneous `[3, N, 1016]` float64 arrays per orbit (`radiances`/`tau`/`toa_sol` at `:375-396`, `T_all`/`ln_T_all` at `:983-984`); float32 halves all of them with no impact on fit quality at these SNRs.
- `load_shared_data` reads the entire Lite NetCDF into ~90 resident float64 arrays for the whole run (`:176-267`); load only needed variables, downcast, or use lazy per-orbit slicing.
- Output HDF5 written uncompressed, float64, ~200 datasets (`:1445-1447`) — add `compression="gzip"` (or lzf) + float32 for derived quantities; shrinks `fitting_details.h5` severalfold and speeds the downstream reads.
- `build_feature_dataset.py` materializes each day twice (per-orbit dict list + concatenated dict, `:275-300`) and builds the entire feature frame in float64 (`:87-95`, `:462-498`) before writing parquet; concatenate incrementally and downcast to float32 at frame build. In `raw_processing_multipe_dates` all per-date DataFrames coexist at peak before one big `pd.concat` (`:668-671`) — for the 117.7M-row combined build, stream to a parquet writer per date instead.
- Correctness-adjacent: `:467-468` mutates `combined.get('fs_rel')` in place, aliasing the source array — make the copy explicit.

### 7.3 ML model structure and training (`src/models/`)

**Implementation plan (tracking, added 2026-07-04).** Hard constraint for every phase: with all new options at defaults, the constructed `GaussianMLP` and its `state_dict` keys are byte-identical to today's, so the production TCCON checkpoints (25× `member_*.pt`) keep loading.

- [x] **Phase 0 — hygiene:** delete `deep_ensemble.py.bak`; CLI `--loss` default → `beta_nll` (production choice); `torch.load(weights_only=True)`; `mlp_baseline` gets the `_cuda_device_supported` guard.
- [x] **Phase 1 — shared trainer:** new `src/models/train_common.py` with `TrainConfig` (all optimizer/schedule/early-stop literals in one place, platform defaults made explicit), `select_device()`, full seeding (`set_seeds` + DataLoader generator/`worker_init_fn`, opt-in `--deterministic`), and a generic `train_model()` loop. `deep_ensemble._train_member` and `mlp_baseline.train_mlp` move onto it fully; `tabm.py` adopts the shared device/seed/optimizer helpers but keeps its specialized inner loop (AMP + per-member quantile losses; experimental model, not worth destabilizing). Training-date manifest JSON written at train time (feeds the TCCON leakage guard from §3-Minor).
- [x] **Phase 2 — regularization options (accuracy/overfitting):** `GaussianMLP(dropout=0.0, norm='none'|'layer'|'batch')`, inserted only when non-default (checkpoint-key compat); CLI `--dropout` / `--norm`; recorded in meta/run summary. No MC-dropout at inference (ensemble provides the spread).
- [x] **Phase 4 — training/predict performance:** GPU-resident tensor batches when the train set fits device memory (DataLoader fallback); `ensemble_predict` streams running moments instead of `[M, N]` stacks.
- [ ] **Phase 3 — regularization ablation (CURC):** arms = baseline / dropout 0.1 / dropout 0.3 / LayerNorm / LN+dropout 0.1 / BatchNorm / `256,128,64`+LN+dropout 0.1; date_kfold 5 folds × both surfaces, production config otherwise; judge on held R² **and** near-cloud (≤10 km) + bottom-5% tail metrics **and** coverage_90. Adopt only if > fold-σ better on both surfaces without coverage regression (TabM-HPO flat-landscape lesson: don't chase noise). Expectation: at 17M rows a 64→32 MLP likely underfits — the capacity+regularization arm is the one to watch; dropout's real chance is the data-scarce near-cloud tail. **Launcher ready:** `sbatch curc_shell_blanca_de_reg_ablation.sh` (70 array tasks; aggregation one-liner in the script header).
- [ ] **Phase 5 — verification:** arm 0 (`base`) of the Phase 3 ablation IS this check — it retrains the exact production config under the new trainer; its fold metrics must land within fold noise of the existing `de_*_beta_nll_prof` runs (ocean ~0.526 / land ~0.387). Then one TCCON case regenerated as spot check. Only Phase 3's decision rule may change the production model.

**Structure:**
- The train loop (DataLoader setup, AdamW + OneCycle with identical literals, grad-clip 1.0, patience 50, best-val checkpointing) is triplicated across `deep_ensemble.py:183-249`, `mlp_baseline.py:79-163`, and `tabm.py:321+`; `mlp_baseline` even hardcodes the same 64→32 stack literally (`:55-61`). Extract one shared trainer + one `TrainConfig` dataclass; the three models become thin heads/losses. This also fixes the drift where only `deep_ensemble` guards against unsupported CUDA cards (`:567-572`) while `mlp_baseline` (`:87-92`) can still crash mid-run.
- Hyperparameters are scattered magic numbers: optimizer/schedule literals repeated in three files (`deep_ensemble.py:214-217`); `log_var` clamp ±10 (`:115`); platform-branched epochs/batch (`(100,2048)` Darwin vs `(500,4096)`, `:573`) and dataset filename chosen by `platform.system()` (`pipeline.py:426-427`). Move to a config object/file; ~30 argparse flags in `main()` (`deep_ensemble.py:320-421`) should mostly become a saved config the run records — which also gives the paper an exact-reproducibility artifact.
- `GaussianMLP` has no dropout/normalization/weight-init and a shared `Linear(·,2)` head for (mu, log_var) (`deep_ensemble.py:83-118`); fine as the production choice, but make it explicit in the config so ablations (separate heads, softplus variance) are one flag, not code edits.

**Training process:**
- Reproducibility is incomplete: only `torch.manual_seed`/`np.random.seed`, with `shuffle=True` + `num_workers>0` and no DataLoader `generator`/`worker_init_fn` or `cudnn.deterministic` (`deep_ensemble.py:190,204-207`) — member training is not bit-reproducible. Matters for a paper claiming a specific ensemble.
- Ensemble members train strictly serially (`:591-603`) though fully independent — trivially parallel across CURC array tasks or processes.
- Tabular data goes through worker-process DataLoaders with per-batch host→device copies (`:199-207`); at these sizes, keeping tensors GPU-resident and slicing indices is faster and simpler.
- Checkpoints: `member_{m}.pt` paths reused with no run isolation, and `torch.load` without `weights_only=True` (`:243,248`); write per-run directories keyed by the saved config hash.
- `ensemble_predict` re-does `model.to(device)` per member and materializes full `[M, N]` mu/var stacks (`:264-275`); chunk over N for the 17M-row scoring runs.
- Hygiene: delete `deep_ensemble.py.bak` (35 KB stale copy in-tree); `mlp_baseline.py:145-148` recomputes R²/MAE by hand instead of the shared `diag.compute_metrics` used everywhere else.

### 7.4 Overall code readability and clarity

- **Constants drift is the recurring disease** — the same number defined in 2–4 places is behind most items in §3-Minor (buffer year 2022 vs 2023, anomaly params 0.25/1.0 vs 0.5/2.0, loss default, band widths, fit order) and §7.3 (optimizer literals ×3). One `src/constants.py` (or a small config module) that every call site imports resolves the whole class at once; do this first, the rest of the hygiene list becomes mechanical.
  - **IMPLEMENTED (2026-07-04):** `src/constants.py` created and wired into every call site. Decisions: buffer year = **2022** (`AQUA_FREE_DRIFT_YEAR`; matches the active demo_combined path — phase_03's 2023 docstring/branch updated, so 2022 data now matches at ±20 min there too); anomaly params = **0.25°/1.0 ppm/10 km** (`ANOMALY_*` + `anomaly_args()`; these are what every production call site already passed — the 0.5/2.0 were stale unused function defaults, now aligned so a bare call reproduces production); band width/overlap = **2.5°/0.5°** (`CLOUD_DIST_BAND_*`, the active CLI defaults; phase_04's 5°/1° default and 10° docstring updated); fit order = **(7,3,7)** (`FIT_ORDER`, now also `process_orbit`'s default, fixing the (7,2,7) drift); ML optimizer/schedule literals deliberately live in `models/train_common.TrainConfig` (single torch-specific home, §7.3 Phase 1) — `tabm.py`/`mlp_baseline.py` residual literals now reference `TrainConfig` attributes.
- **Dead/stale code misleads readers — RESOLVED (2026-07-04):** `deep_ensemble.py.bak` deleted; `get_design_matrix` is now the core of the §7.2 linear solver; the retracted M8 caveat in `compare_profile_features.py` rewritten as a historical note; `pipeline.py` feature lists stripped to active entries only (verified identical lists; SHAP rationales kept in a header comment); `fitting.py` lost its dead `universal_quantile_loss`, `fit_bottom_spline`, the 140-line `if 0:` visualization block, and the commented k1k2 call tail; `demo_combined.py` lost the vestigial module-level `run_phase_035` stub.
- **Fragile idioms — assessed 2026-07-04:** the `sys.modules['pipeline']` alias is **kept deliberately** — it is the backward-compat mechanism itself: every existing `deep_ensemble_pipeline.pkl` (local + CURC production folds) was serialized with `__module__ = 'pipeline'`, so removing the alias would orphan them all; the block is fully commented in place. Blanket `try/except: continue` in Phase 3 remains open (§3-Minor failure accounting); `_check_file_exists_remote` returns were fixed under §7.1.
- **File size — RESOLVED (2026-07-04):** `fitting.py` split along its seams into `spectral/cumulant_fit.py` (fit core: models + exact solver + chunk worker), `spectral/orbit_data.py` (per-date/per-orbit loading + validation), `spectral/anomaly.py` (the target definition), `spectral/fit_plots.py` (example plots), with `fitting.py` (2051→824 lines) keeping orchestration/CLI and re-exporting every public name — verified bit-for-bit on orbit 22845a_GL (all 234 output datasets identical, including the multi-process fit path). `demo_combined.py` (1658→471 lines) split into `workspace/demo_utils.py` (helpers) + `workspace/pipeline_phases.py` (run_phase_1–5) with the orchestrator/CLI unchanged — verified by a full cached-data pipeline run reproducing `results_2018-10-18.h5` bit-for-bit (10 datasets, 53,549 soundings).
- **Docstrings as spec:** several docstrings state parameter values the code no longer uses (§3-Minor items 1–2, `process_orbit` order (7,2,7) vs used (7,3,7)); after centralizing constants, have docstrings reference the constant name instead of repeating numbers.

**Suggested order:** 7.1 stability fixes (an afternoon, prevents corrupt-data reruns) → constants module (7.4) → fitting parallelization + float32 (7.2, biggest wall-clock/RAM win for the reprocessing runs) → shared trainer + run configs (7.3) → file splits (7.4, last, purely cosmetic).

**Status (2026-07-04): the entire suggested order above is executed and verified** (commits `7809fe9`, `223b2cb`). Remaining engineering items, in priority order: (1) wire the `training_dates.json` check into the TCCON `run_case` driver (§3-Minor leakage guarantee — one sentence in the paper); (2) launch the §7.3 Phase 3 regularization ablation (`sbatch curc_shell_blanca_de_reg_ablation.sh`; its `base` arm is the Phase 5 new-trainer verification); (3) Phase-3 per-sounding failure accounting (§3-Minor); (4) §7.1 efficiency (parallel downloads + listing memoization) — nice-to-have, stability was the load-bearing part.

**Status (2026-07-04):** 7.1 stability ✅, constants module (7.4) ✅, 7.2 ✅, 7.3 code phases (0/1/2/4) ✅ — remaining: 7.1 efficiency (parallel downloads, listing cache), 7.3 Phase 3 ablation + Phase 5 verification runs on CURC, 7.4 dead-code/file-split hygiene.
