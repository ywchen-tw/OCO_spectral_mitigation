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

### Phase 1b — cloud-bin screen at the BEST capacity from 1a  [BLOCKED on 1a]
At the winning architecture, compare near-cloud R²:

| cloud_bin_feature | near R² | classifier 5-class acc | status |
|---|---|---|---|
| none (baseline) | _tbd_ | — | pending |
| predicted (GBDT on spectra) | _tbd_ | _tbd_ | pending |
| oracle (true, ceiling) | _tbd_ | — | pending |

Key question: does `predicted` beat just the best-capacity baseline? (distillation vs capacity)

### Phase 2 — CURC date_kfold confirmation  [BLOCKED on 1a/1b decision gate]
Winners from 1a/1b, 5 folds, both surfaces, real date_kfold (the only honest OOD test). Watch bigger models for date-shift overfitting.

---

## Decision log
- 2026-06-26: Multi-task aux abandoned (F1). Oracle proves cloud-distance is informative but MODIS-only (F2). Pivot to: (1a) test DE capacity, (1b) predicted-bin vs capacity, then (2) confirm on CURC.
