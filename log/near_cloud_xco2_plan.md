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

### Phase 2 — CURC date_kfold CAPACITY confirmation  [READY — promoted ahead of 1b]
Given the magnitude of the 1a capacity win (+0.243, larger than the cloud feature), confirm it out-of-distribution BEFORE spending on 1b. Script: `curc_shell_blanca_de_capacity_ablation.sh` (30-task array = {ocean,land} × hidden_dims{64,32 | 128,64,32 | 256,128,64} × fold{0..4}). **Single-task, current style** (xco2 anomaly only; no cloud head, no cloud-bin input) — only `--hidden_dims` varies. Production settings otherwise (beta_nll β=1.0, M=5, batch 8192, near_cloud_target 0.98).

| surface | hidden_dims | near R² (5-fold mean±std) | status |
|---|---|---|---|
| ocean | 64,32 (ref) | _tbd_ | pending |
| ocean | 128,64,32 | _tbd_ | pending |
| ocean | 256,128,64 | _tbd_ | pending |
| land | 64,32 (ref) | _tbd_ | pending |
| land | 128,64,32 | _tbd_ | pending |
| land | 256,128,64 | _tbd_ | pending |

Decision: if 128,64,32 still beats 64,32 OOD (and 256,128,64 doesn't overfit) → adopt the bigger backbone as the new DE baseline, THEN run 1b (predicted bin) on top of it.

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
deployable cloud-prediction model. Optional further gains: feature-importance pruning
of the 29-feature block, shorter ocean threshold (<5km).

---

## Future tests — correction / deployment policy

### FT1 — confidence-weighted correction: `corrected = xco2_bc − P(near)·mu`  [FUTURE]
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
