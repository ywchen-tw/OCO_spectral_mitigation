# Near-Cloud XCO2 Accuracy + Cloud-Distance Feature — Plan & Results Tracker

Started: 2026-06-26
Owner: Yu-Wen + Claude

## Goal & constraints
- **Primary:** improve point accuracy of `xco2_bc_anomaly` for **near-cloud (cld_dist_km ≤ 10 km)** footprints.
- **Secondary:** predict the cloud-distance group (5 bins: [0,2)(2,5)(5,10)(10,15)[15,∞) km).
- **Deployment constraint:** MODIS-derived `cld_dist_km` is **NOT available at inference** → any cloud-distance signal used as a model input must come from a **predicted** (spectra-based) bin, not the true value.

## Test harness conventions (local)
- Data: `results/csv_collection/combined_2020_dates.parquet` (12 dates, ~1.7M rows; ocean 76% near, land 38% near).
- Proven non-degenerate local hp: **batch 2048, 50 epochs, n_members 2, beta_nll β=1.0, seed 42, val_split random**.
  - ⚠️ batch 8192 / ~40 epochs UNDERFITS this data (R²≈0); batch 2048 / 50 ep recovers.
  - ⚠️ `date_kfold` on only 12 dates is **degenerate** (best epoch 0) — the honest out-of-distribution test must run on CURC full data (66 dates).
- Metric of record: **near-cloud (≤10km) R²** from `de_raw_*_stratified_metrics.csv`, regime `cloud_proximity`.

---

## Established findings (DONE)

### F1 — Multi-task auxiliary cloud head (cloud as OUTPUT): ❌ hurts out-of-distribution
tabM, CURC date_kfold, paired on common folds f2,f3,f4:

| λ_cloud | ocean near R² | land near R² |
|---|---|---|
| 0.0 | 0.640 | 0.583 |
| 0.1 | 0.632 | 0.542 |
| 0.3 | 0.609 | 0.442 |

Monotone decline both surfaces. The in-distribution single-date win (+0.04) **sign-flipped** under date-shift. DE (small backbone) also could not benefit (capacity). **Multi-task aux = abandoned.**

### F2 — Oracle: TRUE cld_dist bin as INPUT feature: ✅ large, but MODIS-only
DE, 12-date land, random split, proven hp:

| | near R² | far R² | global R² |
|---|---|---|---|
| baseline | 0.510 | 0.376 | 0.445 |
| + true cld_dist one-hot (oracle) | **0.684** | 0.468 | 0.580 |

+0.175 near R². The gain is largely **independent MODIS info not in the spectra** → a predicted (spectra-based) bin **cannot fully recover it**. Oracle is deployable only if MODIS is available (it is not) → oracle is a **ceiling reference only**.

---

## Forward plan

### Phase 1a — DE capacity sweep (does the small backbone underfit?)  ✅ DONE 2026-06-26
Config: land, 12-date, random split, proven hp (batch 2048, 50 ep, M=2, β=1.0, seed 42).

| config | hidden_dims | near R² | far R² | global R² | cov90 | raw width |
|---|---|---|---|---|---|---|
| current | 64,32 | 0.510 | 0.376 | 0.445 | 0.976 | 2.39 |
| wider | 128,64 | 0.718 | 0.510 | 0.617 | 0.900 | 1.39 |
| **+1 layer (best)** | **128,64,32** | **0.753** | **0.557** | **0.659** | 0.900 | 1.30 |
| bigger | 256,128,64 | 0.745 | 0.527 | 0.640 | 0.887 | 1.29 |

**RESULT: the DE was badly underfitting.** Capacity is the dominant lever:
- near R² **+0.243** (0.510 → 0.753) at 128,64,32 — **larger than the MODIS oracle (+0.175)** and free (no MODIS/classifier/cloud feature).
- 256,128,64 slightly worse → 128,64,32 is the sweet spot (in-distribution).
- Bonus: the underfit 64,32 had inflated σ (raw width 2.39, cov90 0.976 = over-covered); bigger models are tighter AND well-calibrated (cov90≈0.90).

**Reinterprets prior findings:** the multi-task aux hurt and the oracle "helped" largely because the 64,32 model was capacity-starved. A right-sized model may extract the cloud signal from spectra itself.

⚠️ **IN-DISTRIBUTION only.** Bigger models can overfit training dates → the +0.243 MUST be confirmed under date_kfold (Phase 2). This is now the critical risk.

### Phase 2 — CURC date_kfold CAPACITY confirmation  ✅ DONE 2026-06-27 — NEGATIVE
Full 66-date, single-task, only `--hidden_dims` varies. Near-cloud XCO2 R² (5-fold OOD):

| surface | hidden_dims | near R² (5-fold OOD mean±std) | vs 64,32 |
|---|---|---|---|
| ocean | 64,32 (ref) | 0.631 ± 0.055 | — |
| ocean | 128,64,32 | 0.627 ± 0.053 | −0.004 |
| ocean | 256,128,64 | 0.628 ± 0.059 | −0.003 |
| land | 64,32 (ref) | 0.603 ± 0.040 | — |
| land | 128,64,32 | 0.608 ± 0.041 | +0.005 |
| land | 256,128,64 | 0.603 ± 0.040 | −0.001 |

**RESULT: the +0.243 capacity win does NOT survive OOD.** On held-out dates, backbone
size makes no difference (ocean −0.004, land +0.005 — both inside the ±0.05 fold noise).
The in-distribution +0.243 was a RANDOM-SPLIT ARTIFACT — same autocorrelation leakage
that inflated the cloud classifier (0.99→0.85) and flipped multi-task. The bigger
backbone overfit training dates. **→ keep the 64,32 backbone (no capacity change).**

**Sobering meta-finding:** EVERY near-cloud XCO2 lever tried — multi-task aux,
cloud-bin input, backbone capacity — was an in-distribution mirage that vanished OOD.
The honest near-cloud XCO2 R² is ~0.60–0.63 on unseen dates regardless, and appears to
be a ceiling for these features. Only date_kfold numbers are trustworthy.

### Phase 1b — cloud-bin screen at the BEST capacity (128,64,32)  ✅ DONE 2026-06-26
Local, 12-date land, random split. At the winning capacity, near-cloud R² (seed 42):

| cloud_bin_feature | near R² | far R² | classifier 5-class acc |
|---|---|---|---|
| none (baseline) | **0.753** | 0.557 | — |
| predicted (GBDT on spectra) | 0.666 | 0.467 | 0.835 |
| oracle (true, ceiling) | 0.671 | 0.469 | — |

Multi-seed verification (none vs oracle near R²): seed42 −0.082, seed1 −0.023, seed2 −0.029.

**RESULT: at proper capacity, the cloud-bin input does NOT help — even the ORACLE consistently HURTS** (mean ≈ −0.045 across 3 seeds). The +0.175 oracle gain seen at the underfit 64,32 was a **capacity crutch, not independent-information value**: once the backbone is big enough to extract cloud structure from the spectra itself, the explicit coarse bin is redundant and mildly harmful. Since predicted ≈ oracle ≈ below baseline, the **predicted-bin idea is dead** (a noisier version of a feature that doesn't help even when perfect).

**Conclusion: capacity is the lever; the cloud-distance input feature is dropped.** No need to build the cloud classifier or the apply-side augmentation. The cloud-bin code stays in the tree (default-off) as a documented negative result.

### Phase 1c — multi-task aux head re-tested AT proper capacity (128,64,32)  ✅ DONE 2026-06-26
Re-run of F1 (which was on the underfit 64,32) now that capacity is fixed. Local, 12-date land, random split, seed 42.

| λ_cloud @ 128,64,32 | near XCO2 R² | far R² | cloud AUC |
|---|---|---|---|
| 0.0 | 0.753 | 0.557 | — |
| 0.05 | 0.744 | 0.504 | 0.898 |
| 0.1 | 0.758 | 0.521 | 0.918 |
| 0.3 | 0.734 | 0.524 | 0.936 |

**RESULT: capacity-competition confirmed.** On 64,32 the aux head monotonically DESTROYED XCO2 (0.510→0.469); at 128,64,32 it is **roughly neutral for XCO2** (±0.01, within noise) AND yields a strong cloud classifier (AUC ~0.92 at λ=0.1). So at proper capacity you can get **both predictions from one model at ~no XCO2 cost** — not because the cloud task helps XCO2 (capacity does that), but because it's now "free."

⚠️ IN-DISTRIBUTION. At 64,32 multi-task helped in-distribution but flipped to hurting OOD (F1). Must confirm under date_kfold that (a) λ>0 stays XCO2-neutral at proper capacity and (b) the cloud AUC generalizes to unseen dates.

### Phase 1d — SEPARATE cloud<10km classifier vs the bundled multi-task head  ✅ DONE 2026-06-26
Standalone GBDT + MLP classifiers (same FeaturePipeline features, same 12-date random split / held set as the DE runs). Target: binary cld_dist_km < 10km.

| surface | model | AUC | AP | recall@0.5 |
|---|---|---|---|---|
| land (pos 17%) | GBDT | **0.973** | 0.871 | 0.943 |
| land | MLP | 0.970 | 0.855 | 0.935 |
| ocean (pos 49%) | GBDT | 0.884 | 0.871 | 0.802 |
| ocean | MLP | 0.894 | 0.880 | 0.823 |

Bundled multi-task head (land, 128,64,32): AUC 0.918 (λ=0.1) / 0.936 (λ=0.3).

**RESULT: a dedicated classifier predicts cloud<10km notably better than the bundled head** (land +0.05 AUC: 0.973 vs 0.918). GBDT ≈ MLP (within 0.01; GBDT better on land, MLP on ocean). Cloud<10km is highly predictable from spectra on land (AUC ~0.97), harder on ocean (~0.89). No ocean bundled number to compare (1c was land-only).

**Conclusion → TWO separate models for the dual goal:** (1) a bigger single-task DE for XCO2 (capacity is the lever, +0.243), and (2) a dedicated GBDT/MLP for cloud<10km (beats bundling by +0.05 AUC). Multi-task loses on both axes — it doesn't help XCO2 (capacity does) and predicts cloud worse than a dedicated model. ⚠️ All in-distribution; confirm both on CURC date_kfold.

### Phase 1e — push cloud<10km AUC (tuned models)  ✅ DONE 2026-06-26
Same 12-date held set. Tuned HGB (early stopping), XGBoost (early stopping, ~2000 trees), bigger MLP (256,128,64), XGB+MLP rank-ensemble.

| surface | HGB tuned | **XGBoost** | MLP big | ENS | prev best |
|---|---|---|---|---|---|
| land (pos 17%) | 0.987 | **0.9935** | 0.985 | 0.992 | 0.973 |
| ocean (pos 49%) | 0.928 | **0.9538** | 0.933 | 0.951 | 0.894 |

**RESULT: XGBoost wins both** — land 0.9935 (near-ceiling, +0.02), ocean 0.9538 (+0.06, big jump on the harder surface). Ensemble doesn't beat XGB alone. vs the bundled multi-task head (land 0.918): **+0.075**. Dedicated XGBoost is the cloud classifier. XGB hit the ~2000-tree cap → marginal headroom with more trees. ⚠️ in-distribution; OOD date_kfold next (2000 trees risks date overfit).

### Phase 1f — cloud<10km classifier OUT OF DISTRIBUTION (date_kfold, local 12-date)  ✅ DONE 2026-06-26
XGBoost, random (in-dist) vs date_kfold (OOD, 4 folds), both surfaces:

| surface | random (in-dist) | date_kfold OOD | drop |
|---|---|---|---|
| land | 0.9935 | 0.824 ± 0.05 | −0.17 |
| ocean | 0.9538 | 0.654 ± 0.01 | −0.30 |

**RESULT: the in-distribution AUCs were largely an ILLUSION** (spatial/temporal autocorrelation leakage when train/test share dates). On unseen dates: land 0.82 (useful), ocean 0.65 (near chance). This also inflates every prior in-dist number (bundled head 0.918 included). Regularization (400 trees, depth 4, strong reg) helps modestly — land 0.824→0.843, ocean 0.654→0.688 — so the collapse is *partly* overfit, *mostly* fundamental at 12 dates (only ~9 train dates/fold). **Key hope: the full 66-date set has many more train dates/fold → should generalize better. → CURC test.**

### Phase 2b — CURC XGBoost cloud<10km date_kfold (full 66-date data)  ✅ DONE 2026-06-27
Single regularized XGBoost per fold (no ensemble, CPU). 5-fold OOD AUC:

| surface | OOD AUC | recall@0.5 | precision@0.5 | vs local 12-date OOD |
|---|---|---|---|---|
| land | **0.845 ± 0.012** | 0.75–0.81 | ~0.42 | +0.005 (flat) |
| ocean | **0.720 ± 0.006** | 0.59–0.66 | ~0.65 | +0.030 |

**RESULT: more data did NOT recover it.** 12→66 train dates left land flat (~0.845) and ocean barely up (~0.72). The OOD collapse is FUNDAMENTAL, not too-few-dates — the in-dist 0.99/0.95 was autocorrelation leakage; **~0.85 land / ~0.72 ocean is the true spectra-only ceiling** on unseen dates. (All folds hit the 400-tree cap; the local overfit 2000-tree config was worse OOD → regularized is right.)

**Deployment read:** land usable as a recall-tuned gate (recall 0.77–0.81 @ 0.5, higher at lower threshold; false positives cheap since mu≈0 far away). Ocean marginal — misses ~40% of near-cloud @ 0.5; needs a low threshold and stays noisy. Cloud proximity is just weakly encoded in ocean spectra.

### Phase 2c — expanded cloud-diagnostic features for the classifier  ✅ DONE 2026-06-27
Hypothesis: ocean's weak OOD AUC is a FEATURE-SET problem (reuses XCO2 features,
omits cloud diagnostics), not a model limit. Added a cloud-diagnostic block to the
base features: fit quality (chi2_*, rms_rel_*), scattering (dp, dp_abp, dpfrac,
dp_o2a, dp_sco2), cloud aerosols (aod_ice/water/total, ice/water_height),
brightness/SNR (alb_*, snr_*, csnr_*, max_declock_o2a), continuum (h_cont_*), glint
(glint_prox). All deployable (per-sounding L2, no MODIS, not cld_dist-derived).
Excluded leaky/MODIS-built r25_*/ref_*/weighted_cloud_dist_km. XGBoost, local
12-date date_kfold:

| surface | BASE | EXPANDED | gain |
|---|---|---|---|
| land | 0.840 (34 feat) | 0.841 (60 feat) | +0.001 |
| ocean | 0.688 (28 feat) | **0.764 (57 feat)** | **+0.075** |

**RESULT: confirmed — ocean was feature-starved.** Cloud diagnostics lift ocean OOD
AUC 0.69→0.76 (recall 0.66→0.69); land unmoved (already at ceiling via its
bright/structured surface). Adopt expanded features for the ocean classifier.

### Phase 2d — expanded features on CURC full 66-date OOD  ✅ DONE 2026-06-27
Wired --cloud_features into xgb_cloud_classifier; CURC date_kfold (suffix xgbcloud_exp_*):

| surface | BASE (full-66) | EXPANDED (full-66) | gain | recall@0.5 |
|---|---|---|---|---|
| ocean | 0.720 | **0.781** | +0.061 | 0.62→0.67 |
| land | 0.845 | 0.850 | +0.005 | 0.78→0.79 |

**RESULT: the lift HOLDS out-of-distribution on full data.** Ocean +0.061 (per-fold
0.768–0.788, std 0.007 — rock-steady), land unchanged. **Final cloud<10km classifier:
land ~0.85 / ocean ~0.78 OOD AUC**, both usable as recall-tuned gates. This is the
deployable cloud-prediction model.

### Phase 2e — threshold sweep (<5 / <7 / <10 km, expanded features)  ✅ DONE 2026-06-27
Local 12-date date_kfold OOD AUC by threshold:

| surface | <5km | <7km | <10km |
|---|---|---|---|
| land | 0.847 | 0.846 | 0.841 |
| ocean | **0.819** | 0.793 | 0.764 |

**RESULT: ocean is MORE predictable at tighter thresholds** (monotone 0.764→0.793→0.819,
+0.055 from <10 to <5) — very-near-cloud carries a stronger signature, and it matches
the boxplot (ocean bias concentrated <~7km). Land is flat (~0.845, threshold-free).
(AP drops at tighter thresholds = base-rate artifact; AUC is the fair comparison.)
**→ surface-specific gate thresholds: ocean ~5km, land ~10km.** A <5km ocean gate is
both more accurate AND only flags where ocean bias lives.

### Phase 2f — FINAL thresholds from 1km bias-decay (full 66-date, 5.96M soundings)  ✅ DONE 2026-06-27
Finer 1km boxplot of xco2_bc_anomaly vs cld_dist (0-15km + 15-50 ref), full data.
Fig: results/figures/cld_dist_analysis/xco2_bc_anomaly_ocean_land_boxplot_1km_0to15_66date.png.
Median bias by bin:

| | 0-1 | 4-5 | 6-7 | 8-9 | 10-11 | 14-15 | 15-50 ref |
|---|---|---|---|---|---|---|---|
| ocean | -0.199 | -0.038 | -0.016 | -0.005 | -0.0003 | -0.0004 | +0.0003 |
| land | +0.181 | +0.128 | +0.097 | +0.085 | +0.059 | +0.040 | +0.019 |

**FINAL: ocean ~5km, land ~15km** (NOT a shared 10km).
- Ocean bias gone (|med|<0.01) by 7-8km, ~0 by 9km; bulk <5km. Classifier best at <5km
  → ocean gate **5km** (or 7km to catch the small 5-7km residual).
- Land bias very persistent: still +0.040 at 14-15km; far-ref +0.019 = baseline land
  offset. Cloud-attributable excess (med-0.019) ~+0.04 at 10-11km → 10km leaves bias
  uncorrected; land gate **~15km** (classifier AUC flat across thresholds, so free).

Reusable boxplot generator saved: `workspace/plot_cld_dist_boxplot_fine.py`
(configurable parquet / col / bin-width / max-km).

### Phase 2g — local validation of FINAL per-surface thresholds  ✅ DONE 2026-06-27
Expanded-feature XGBoost, local 12-date date_kfold OOD, at the chosen thresholds:

| config | AUC | AP | recall | precision | pos_rate |
|---|---|---|---|---|---|
| ocean @5km (FINAL) | 0.819 | 0.66 | 0.71 | 0.53 | 0.26 |
| land @15km (FINAL) | 0.839 | 0.64 | 0.76 | 0.53 | 0.25 |
| land @10km (compare) | 0.841 | 0.54 | 0.75 | 0.42 | 0.17 |

**Both validate.** land@15km matches @10km on AUC (0.839 vs 0.841, flat as expected) but
with BETTER precision/AP (more balanced) — so 15km captures the 10-15km bias at no AUC
cost. Ready for the final CURC run (ocean --near_cloud_km 5.0, land --near_cloud_km 15.0).

### Phase 2h — FINAL per-surface cloud classifier on CURC full-66 OOD  ✅ DONE 2026-06-27
xgbcloud_final_* (expanded features, ocean@5km / land@15km), date_kfold 5-fold:

| surface | threshold | AUC | AP | recall@0.5 | precision@0.5 |
|---|---|---|---|---|---|
| ocean | 5km | **0.826 ± 0.004** | 0.67 | 0.71 | 0.54 |
| land | 15km | **0.849 ± 0.011** | 0.68 | 0.78 | 0.55 |

**FINAL deployable cloud classifier.** Rock-steady (ocean std 0.004). Ocean@5km beats
ocean@10km (0.781) by +0.045 — tighter-threshold finding confirmed on full data. Land@15km
≈ @10km AUC but better AP/precision and captures the 10-15km bias. This is the production
cloud gate (footprint-independent, no MODIS).

---

## Correction / deployment policy

### FT1 — confidence-weighted vs gated vs full correction  ✅ DONE 2026-06-27 — full `mu` WINS
Joint OOD comparison of three policies on the full 114-date set, date_kfold (every
sounding scored by the fold model that held out its date). DE `mu` (beta_nll 64,32)
and xgb `P(near)` (final expanded classifier, ocean@5km / land@15km) predicted on the
SAME rows (shared finite-feature mask). Metric: residual `y − corr` vs truth.
Script: `workspace/compare_correction_policies.py`; full table
`results/model_comparison/correction_policy_comparison.csv`.
Policies: (1) `mu` (current), (2) `P(near)·mu`, (3) `mu·1[P>0.5]`. Near-cloud R²:

| surface | stratum | (1) full mu | (2) P·mu | (3) gate |
|---|---|---|---|---|
| ocean | near (≤5km) | **0.685** | 0.673 | 0.672 |
| ocean | ≤10km | **0.633** | 0.626 | 0.615 |
| ocean | far (>5km) | 0.191 | **0.230** | 0.138 |
| ocean | all | 0.546 | **0.551** | 0.522 |
| land | near (≤15km) | **0.566** | 0.549 | 0.543 |
| land | ≤10km | **0.604** | 0.582 | 0.584 |
| land | far (>15km) | **0.266** | 0.188 | 0.080 |
| land | all | **0.458** | 0.418 | 0.376 |

(Sanity: policy-1 ≤10km R² = 0.633 ocean / 0.604 land reproduces Phase-2 0.631/0.603.)

**RESULT: keep the current full-`mu` correction. Neither down-weighting scheme helps
the primary near-cloud goal — both HURT it on both surfaces, most at 0–2km where the
bias is largest** (ocean 0.726→0.712 P·mu; land 0.745→0.727), because `P(near)<1`
down-weights real corrections exactly where they matter. The FT1 premise (far anomaly
≈ 0, so don't correct it) is only weakly true on **ocean beyond ~5km** — there P·mu
shaves noise (far R² 0.19→0.23; fine bins: 10–15km 0.13→0.19, >15km 0.12→0.18) — but
even there full `mu` keeps a positive ~6% RMS reduction (so `mu` is NOT pure noise far
out), and the net ocean-all gain is negligible (+0.004 R²). On **land the premise is
false**: the bias genuinely extends far (boxplot +0.04 at 14–15km), so full `mu` is
best everywhere incl. far (>15km 0.266 vs 0.188) — any taper throws away real signal.
The hard gate (3) is worst nearly everywhere (zeros corrections for near-cloud
soundings below the 0.5 threshold; ocean recall@0.5 only 0.71). **The DE's `mu`
already self-attenuates where there's no bias; an external `P(near)` gate removes more
real signal than noise. Confidence-weighting and gating are not adopted.**

### FT1-TCCON — independent confirmation vs TCCON ground stations  ✅ DONE 2026-06-27
Same three policies, but validated against TCCON (absolute ground truth, not the OCO
label) over the deep-ensemble TCCON cases. Per case: build DE `mu` + xgb `P(near)`
(pooled folds — most case dates are outside the 2016–2020 training dates → genuinely
unseen), match OCO footprints ≤100km of the station within ±60min of the pass, compare
each corrected XCO2 to the TCCON window-mean. Pooled over 57,670 correctable footprints
/ 33 station-days (`workspace/tccon_correction_policy_stats.py`,
`results/model_comparison/tccon_policy/`):

| subset | metric | uncorrected | full_mu | P·mu | gate | ideal |
|---|---|---|---|---|---|---|
| all | bias | −0.391 | **−0.088** | −0.210 | −0.230 | −0.123 |
| all | RMSE | 1.636 | **1.073** | 1.090 | 1.107 | 0.583 |
| near-cloud | bias | −0.478 | **−0.135** | −0.229 | −0.236 | −0.050 |
| near-cloud | RMSE | 1.974 | **1.252** | 1.267 | 1.275 | 0.609 |
| far-cloud | bias | −0.239 | **−0.007** | −0.178 | −0.220 | −0.250 |
| far-cloud | RMSE | 0.738 | **0.649** | 0.679 | 0.726 | 0.533 |

**RESULT: TCCON independently confirms full `mu` wins on every subset** — lowest |bias|
AND RMSE vs absolute truth. P·mu and gate UNDER-correct (residual bias −0.21/−0.23 vs
full_mu −0.09). Decisively, **even far-cloud full `mu` wins** (bias −0.007 vs P·mu
−0.178): down-weighting far corrections leaves the −0.24 uncorrected bias mostly
in place, because `mu` is doing real bias removal there — exactly opposite to the FT1
premise. Two independent evaluations (held-out OCO label + TCCON) agree: **keep full
`mu`; cloud-probability weighting/gating does not help.** (`build_deepens_plot_data.py`
now emits `p_near` + the two policy columns when xgb cloud dirs are passed; the
`.sh` harness is wired but optional.)

### FT2 — LATITUDE gate (skip correction where |lat| > L)  ✅ DONE 2026-06-27 — gate NOT adopted
Tested skipping the DE correction for high-latitude footprints (|lat| > 80/75/70°,
N&S) vs TCCON (`tccon_correction_policy_stats.py`, lat-gate columns derived from
`lat`/`pred_anomaly`). The only high-lat TCCON footprints are at 75–80°N (Ny-Ålesund,
Svalbard); none in 70–75° or >80°, so latgate70≡latgate75 and latgate80≡full_mu.

⚠️ **Initial result (gate "helps") was a FILTER ARTIFACT and is RETRACTED.** The first
pass filtered footprints to finite `xco2_bc_anomaly`, which kept only **2 of the 8**
Ny-Ålesund station-days — and those 2 happened to be ones where `mu` overcorrects, so
gating looked like a clean win (latgate75 R² 0.984 vs full_mu 0.973). Dropping the
anomaly dependency (band-only QC, |xco2_bc−TCCON|<50ppm) restores all **8 high-lat
station-days** and flips the conclusion.

Full 73-station-day calibration (mean OCO vs TCCON), no anomaly filter:

| policy | slope | R² | RMS resid |
|---|---|---|---|
| uncorrected | 1.0076 | 0.9384 | 1.661 |
| **full_mu (no gate, = latgate80)** | 1.0222 | **0.9698** | **1.139** |
| latgate75 (= latgate70) | 1.0258 | 0.9609 | 1.312 |

The 8 high-lat (Ny-Ålesund) days, |bias to TCCON|:

| metric (8 days) | uncorrected (=gated) | full_mu (no gate) |
|---|---|---|
| RMS bias | 3.382 | **2.749** |
| mean \|bias\| | **2.315** | 2.365 |

**RESULT: the |lat|>75 gate is NOT beneficial — full `mu` (no gate) has the higher
R² (0.970 vs 0.961) and lower station-mean RMS.** At high latitude the correction is
mixed but net-helpful by RMS: it overcorrects several small-bias days (gate would help
those by ~0.3–1.3 ppm) BUT fixes one large bias — **2017-06-17 Ny-Ålesund: −8.1 → −4.2
ppm** — which dominates RMS. So gating throws away a big real correction to shave
several small ones. **Do NOT add a latitude gate; keep full `mu` everywhere.** (Methods
note: never select the evaluation footprints on a truth column — it cherry-picked the
2 worsening days. Band-only QC is the right filter.)

**Cross-check excluding Ny-Ålesund entirely (65 non-polar station-days,
`plot_latgate_tccon_comparison.py --exclude-sites ny`): the correction is strongly
beneficial** — mean|bias| 0.907→0.577, RMS bias 1.301→**0.725** (−44%), R²
0.963→**0.988**, slope 0.992→1.010. Ny-Ålesund alone inflates the corrected RMS bias
0.725→1.139 and drops R² 0.988→0.970 — one genuinely hard polar site (extreme SZA,
snow/ice), not a reason to gate. Fig: `tccon_latgate75_comparison_excl_ny.png`.

### FT1-orig — original motivation (superseded by the result above)
Apply the DE correction `mu` scaled by the classifier's CONTINUOUS near-cloud
probability `P(near)`, instead of a hard distance gate.

**Motivation** (`results/figures/cld_dist_analysis/xco2_bc_anomaly_ocean_land_boxplot.png`):
the cloud BIAS (median anomaly) is concentrated at short range — ocean median
−0.18 (0–2km) → ~0 by 10km; land +0.15 → ~0 by ~15km. Beyond that the anomaly is
centered on zero (no cloud-attributable signal), so far-cloud footprints should
NOT be corrected. `P(near)·mu` does this smoothly: full correction where the model
is confident it's near-cloud, tapering to zero far away — no hard-bin discontinuities
(which would inject spatial-gradient artifacts in the XCO2 field), and the "partial"
zone emerges naturally where `P≈0.5`.

**vs the discrete 3-group policy** (full <10 / 50% 10–15 / none >15): the continuous
form is preferred — smoother, encodes confidence, and avoids the arbitrary 50% step
(the boxplot says land's 10–15km residual is ~⅓ of peak, not ½). Discrete remains a
simpler, more communicable fallback.

**Why the noisy gate is tolerable here:** the gate's errors are asymmetrically cheap.
far→near misclassification applies `mu ≈ 0` (model self-zeros far away) → low harm;
the costly error is near→far (missed correction). So tune the classifier for high
RECALL on near-cloud, not balanced accuracy (land recall@0.5 already 0.94).

**Policies to compare** (metric: corrected-XCO2 residual vs truth on held-out dates,
and vs TCCON; watch for gradient artifacts):
  (a) full `mu` everywhere   (b) binary near-cloud gate
  (c) discrete 3-group taper (d) continuous `P(near)·mu`  ← the one to beat

**Caveats / prerequisites:** needs CALIBRATED `P(near)` — especially OOD, where the
classifier collapses on ocean (Phase 1f/2b). Likely surface-specific. Blocked on:
DE capacity confirmation (Phase 2) + cloud classifier OOD (Phase 2b). Also worth
checking first whether `mu` already self-zeros beyond ~10–15km (if so, the gate
mostly guards against model overfitting in the no-signal regime).

---

## Decision log
- 2026-06-26: Multi-task aux abandoned (F1). Oracle proves cloud-distance is informative but MODIS-only (F2). Pivot to: (1a) test DE capacity, (1b) predicted-bin vs capacity, then (2) confirm on CURC.
- 2026-06-26: 1a — DE was badly underfitting; 128,64,32 gives +0.243 near R² (free, no MODIS). 1b — at that capacity, cloud-bin (even oracle) HURTS → the oracle gain was a capacity crutch; predicted-bin idea dropped. **Net: increase DE capacity (single-task), confirm OOD on CURC; cloud-distance as input/output is closed with evidence.**
- 2026-06-26: 1c/1d/1e — multi-task XCO2-neutral at proper capacity but bundled cloud AUC (0.918) < dedicated XGBoost (0.99 land in-dist). 1f — cloud AUC COLLAPSES OOD (date_kfold: 0.84 land / 0.69 ocean) = in-dist was autocorrelation leakage. Built xgb_cloud_classifier + CURC OOD test (Phase 2b). Boxplot motivates gating off >10km corrections (bias concentrated <10km ocean / <15km land). **Logged FT1: confidence-weighted correction `xco2_bc − P(near)·mu` for future test once the two CURC OOD results land.**
- 2026-06-27: FT1 — RESOLVED. Joint OOD comparison (full `mu` vs `P·mu` vs `mu·1[P>0.5]`) on the full 114-date date_kfold set. **Full `mu` (the current correction) wins the primary near-cloud goal on both surfaces;** both down-weighting schemes hurt it, worst at 0–2km. `P·mu` only helps ocean far-cloud (a noise regime that doesn't matter and barely moves ocean-all, +0.004 R²); on land it strictly hurts (land bias extends far). Hard gate is worst. **Keep full-`mu` correction; cloud-distance probability adds no point-accuracy value on top of the DE — consistent with the project's meta-finding. The near-cloud accuracy program is closed.** (Note: the canonical training parquet is now `combined_2016_2020_dates.parquet` = 114 dates / 17.5M rows; DE `full` mask drops ~53% of ocean rows as non-finite — fold counts reproduce exactly.)
