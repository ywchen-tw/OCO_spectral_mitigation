# TCCON Bias and fp-RMSE Improvement Plan

Date: 2026-07-06

## Goal

Improve corrected XCO2 agreement with TCCON, with two primary metrics:

- lower after-correction mean absolute station bias;
- lower per-footprint RMSE around TCCON stations.

**Metric caveat (added 2026-07-07; revised same day after the wet/dry fix):**
under the AK-harmonized reference the station-mean bias has a floor of roughly
|ak_delta| ≈ **0.3–0.4 ppm** that no per-sounding anomaly model can cross — an
absolute-scale offset inherited from the operational product's direct-TCCON
anchoring plus the genuine smoothing term, not model error (see the final
section below; the originally observed ≈0.9 ppm floor was dominated by the
`ak_harmonize.py` wet/dry bug, fixed 2026-07-07 — CRITICAL_FIXES #11). Judge
blending/calibration experiments on fp-RMSE and on AK-invariant before→after
improvements, not on the absolute AK-harmonized station bias.

The current best result is still the regularized deep ensemble. The structured
residual model is close, but does not yet clearly beat DE on the TCCON objective.

## Current reference results

> **Superseded 2026-07-07.** The table below was generated on the pre-fix
> (buggy) AK reference — see CRITICAL_FIXES #11 and the regenerated table
> that follows it.

For the local r50 km TCCON comparison:

| model set | after mean \|bias\| | after fp-RMSE | note |
| --- | ---: | ---: | --- |
| DE o05/l15 | ~0.93 ppm | ~1.36 ppm | current best |
| Structured o10/l10 | ~1.03 ppm | ~1.53 ppm | shared cloud-distance target |
| Structured o05/l15 | ~1.00 ppm | ~1.51 ppm | closer to DE |
| Structured o05/l15, drop-guarded | ~0.96 ppm | ~1.37 ppm | close to DE after excluding guarded footprints |
| TabM o05/l15 | ~1.09 ppm | ~1.78 ppm | weaker than DE and structured |

### Regenerated results (2026-07-07, fixed AK reference; all model reports rerun)

r50 km (69 cases), after-correction, AK-harmonized | direct:

| model set | AK \|bias\| | AK fp-RMSE | direct \|bias\| | direct fp-RMSE |
| --- | ---: | ---: | ---: | ---: |
| DE o05/l15 | 0.79 | **1.19** | 0.63 | **1.08** |
| Structured o10/l10 | **0.74** | 1.26 | 0.63 | 1.19 |
| Structured o05/l15 | 0.85 | 1.33 | 0.66 | 1.22 |
| Struct + cal (surface/\|mu\|/sigma) | 0.84 | 1.29 | 0.62 | 1.16 |
| Regime structured o05/l15 | 0.83 | 1.36 | 0.67 | 1.25 |
| TabM o05/l15 | 0.97 | 1.62 | 0.81 | 1.53 |

r100 km (75 cases): DE 0.81/1.19 (AK), 0.63/1.08 (direct); Structured o10/l10
0.78/1.26; Structured o05/l15 0.86/1.33; TabM o05/l15 0.90/1.53.

Interpretation (updated): **DE keeps a clear fp-RMSE lead at every radius and
reference** — the footprint-level metric that reflects the near-cloud tail.
On station-mean |bias|, Structured o10/l10 now marginally edges DE
(0.74 vs 0.79 AK r50; 0.78 vs 0.81 r100), but the gap (~0.04 ppm) is well
inside the site-clustered CI (±0.3) — treat as a tie. The calibration variants
remain indistinguishable from uncalibrated structured. TabM stays clearly
behind. Conclusion unchanged: DE stays production; with the AK reference fixed
and DE performing well on both references, further model-side improvement is
de-prioritized (2026-07-07 decision) — the remaining gaps are reference-chain
properties, not model error.

## Main conclusion

The next improvement should probably target calibration, blending, and regime
robustness before adding a larger backbone.

A transformer-style structured model is worth testing, but only if the attention
is organized around physical feature blocks. A CNN is not a good first choice
for the current scalar `FeaturePipeline` inputs because the feature order is
semantic, not spatial, spectral, or vertical.

## Recommended experiment order

### 1. Blend DE and structured residual predictions

This is the lowest-risk next step.

Use existing DE and structured model outputs and fit a simple cross-validated
blender:

```text
correction = w_de * de_mu + w_struct * struct_mu + b
```

or per surface:

```text
ocean correction = w_de_ocean * de_mu + w_struct_ocean * struct_mu + b_ocean
land  correction = w_de_land  * de_mu + w_struct_land  * struct_mu + b_land
```

Useful constraints:

- nonnegative weights;
- optionally force weights to sum to one;
- use ridge regularization;
- fit on training/calibration folds, not on TCCON, if TCCON should remain a
  fully independent validation target.

Why this is promising:

- DE is best overall, but structured residual has a different inductive bias.
- If their errors are not perfectly correlated, a conservative blend may reduce
  station bias and fp-RMSE without changing either base model.
- This is easy to test locally using already downloaded prediction products.

### 2. Add bias-aware calibration

The current models are trained to minimize per-sounding anomaly error, while
TCCON evaluates grouped station/date behavior. Add a small calibration layer
using held-out folds:

```text
xco2_corrected = xco2_raw - alpha_regime * mu - beta_regime
```

Candidate regimes:

- surface type;
- cloud-distance target pair;
- near-cloud versus far-cloud;
- footprint index;
- glint/nadir mode;
- broad latitude band;
- high/low aerosol or surface heterogeneity.

This is manuscript-friendly because it is interpretable: the physical correction
is calibrated by validation-derived regimes, not tuned directly on TCCON.

### 3. Try a regime-aware structured residual model

Use a small mixture-of-experts structure:

```text
mu = gate_1(x) * expert_1(x) + gate_2(x) * expert_2(x) + ...
```

Likely regimes:

- near-cloud versus far-cloud;
- ocean glint versus non-glint;
- land high-aerosol or high-heterogeneity;
- high versus low solar zenith;
- high versus low profile uncertainty.

This is more targeted than simply making the MLP deeper, because the observed
TCCON degradation appears regime-dependent.

### 4. Test a small feature-token transformer

This is the most reasonable neural-backbone extension.

Recommended design:

```text
physical feature blocks:
  xco2 anchor
  spectroscopy
  contamination/cloud/aerosol
  geometry
  state
  profile
  footprint

each block -> token embedding
tokens -> 1-2 layer transformer encoder
pooled tokens + explicit xco2 anchor path -> mean/sigma head
```

Suggested starting configuration:

- token dimension: 16 or 32;
- transformer layers: 1 or 2;
- attention heads: 2 or 4;
- dropout: 0.05 to 0.10;
- retain the explicit `xco2_raw_minus_apriori` anchor path;
- retain fold-specific profile PCA;
- implement in a separate model file rather than modifying production
  `deep_ensemble.py`.

This model can learn interactions such as spectroscopy × geometry × cloud
contamination more flexibly than the current block-concat residual MLP.

### 5. Do not prioritize CNN unless ordered inputs are added

A CNN is not a good match for the current scalar feature vector. The existing
feature order in `src/models/pipeline.py` is not physically ordered in a way
that makes convolution meaningful.

CNN becomes reasonable only if one of these ordered inputs is used:

- full spectra;
- compact binned spectral windows;
- raw vertical profile levels instead of profile PCA;
- a deliberately ordered compact spectral-summary sequence.

If full spectra are too large, a future compromise could be a compact spectral
summary encoder for O2 A-band, weak CO2, and strong CO2 features. With only the
current k-coefficients and scalar summaries, attention or gating is a better
fit than convolution.

## Practical next step

Start with a local comparison using existing downloaded products:

1. DE baseline.
2. Structured residual baseline.
3. DE + structured blend.
4. DE + structured blend with per-surface calibration.
5. Optional: DE + structured + TabM blend.

Evaluate each with the same TCCON r50 and r100 reports. If the blend improves
mean absolute station bias and fp-RMSE, then move the workflow to CURC. If the
blend does not improve, the next model-side experiment should be the small
feature-token transformer, not a CNN.

## First calibration test

Post-training calibration was tested for the structured o05/l15 model using
date-kfold held-out predictions, without using TCCON in the fit. The fitted
global relationship was approximately:

```text
y_true ≈ 1.077 * mu - 0.004
```

This indicates the structured model is slightly under-amplified relative to the
held-out OCO-2 anomaly target. However, when evaluated against TCCON r50 km,
the calibration did not improve the main external target:

| model set | after mean \|bias\| | after fp-RMSE | drop-guard mean \|bias\| | drop-guard fp-RMSE |
| --- | ---: | ---: | ---: | ---: |
| DE o05/l15 | 0.93 ppm | 1.36 ppm | 0.92 ppm | 1.31 ppm |
| Structured o05/l15 | 1.00 ppm | 1.51 ppm | 0.96 ppm | 1.37 ppm |
| Structured + surface calibration | 1.04 ppm | 1.49 ppm | 1.02 ppm | 1.36 ppm |
| Structured + surface/abs(mu) calibration | 1.04 ppm | 1.49 ppm | 1.02 ppm | 1.36 ppm |
| Structured + surface/sigma calibration | 1.04 ppm | 1.49 ppm | 1.02 ppm | 1.35 ppm |

Interpretation: OCO-2 held-out anomaly calibration slightly reduces
per-footprint spread but worsens station mean bias against TCCON. The TCCON gap
is therefore probably not a simple global scaling/intercept problem. More
promising next steps are:

- preserve footprint index in generated plot-data and test surface+footprint
  calibration;
- try a regime-aware structured model with learned experts;
- test a DE + structured blend before adding more backbone complexity.

## Regime-aware structured residual implementation

Option 3 has been implemented as a separate experimental backbone:

```text
src/models/regime_structured_residual.py
```

The model keeps the same per-sounding input constraint as the structured
residual model. It does not use spectra arrays, neighboring soundings, TCCON, or
retrieval examples at inference. It uses the existing `FeaturePipeline` scalar
features and fold-specific profile PCA.

Architecture:

```text
physical feature blocks -> block encoders -> shared body

gate = softmax(gate_head(body))
expert_mu = expert_mu_head(body)

mu = anchor(xco2_raw_minus_apriori) + sum_k gate_k * expert_mu_k
raw2 = variance_head(body)
```

The first configuration uses:

```text
hidden_dims = 64,32
block_dim = 8
n_experts = 4
dropout = 0.10
loss = beta_nll
beta = 1.0
```

The existing structured trainer now supports:

```text
--backbone regime_structured_residual
--n_experts 4
```

CURC launchers were added for the same fold-PCA, surface-separated setup:

```text
sbatch curc_shell_blanca_train_regime_structured_shared_foldpca_2016_2020_r05.sh
sbatch curc_shell_blanca_train_regime_structured_shared_foldpca_2016_2020_r10.sh
sbatch curc_shell_blanca_train_regime_structured_shared_foldpca_2016_2020_r15.sh
```

For the current best mixed cloud-distance comparison, use ocean from r05 and
land from r15 after the jobs finish:

```text
de2016_2020_regime_structured_shared_h64x32_b8_e4_foldpca_r05_ocean_f*
de2016_2020_regime_structured_shared_h64x32_b8_e4_foldpca_r15_land_f*
```

## Direct vs AK-harmonized station bias (added 2026-07-07 — RESOLVED same day)

Observation: after correction, mean |station bias| was ~0.63 ppm against the
direct TCCON window mean but ~1.05 ppm against the AK/prior-harmonized
reference (r100), while fp-RMSE improved under both references.

**Resolution: the planned diagnostics ran and found a bug** (CRITICAL_FIXES
#11). The a≡1 null check failed — forcing the averaging kernel to 1 left the
delta unchanged (−0.933 vs −0.927), so the shift was never an AK effect (true
smoothing term: +0.006 ± 0.227 ppm, literature-scale). The near-cloud vs
far-cloud operator-population hypothesis was refuted (+0.033 ± 0.110 ppm
paired). The entire shift sat in the truth-proxy column: GGG2020 `prior_co2`
is a **wet** mole fraction and `ak_harmonize.py` used it as dry against the
dry-air OCO-2 operator — an error of ≈ −1.3 ppm × column-H₂O fraction,
confirmed by H₂O correlation (r = −0.66), level-by-level accumulation in the
humid layers, and wet/dry closure tests across all 20 TCCON sites.

Fix (`x_tc = gamma·prior_co2/(1 − prior_h2o)`) applied; r100 + r50 reports
regenerated 2026-07-07:

| | direct | AK (buggy) | AK (fixed) |
| --- | ---: | ---: | ---: |
| ak_delta | — | −0.93 ± 0.74 | **+0.34 ± 0.55** |
| after mean \|bias\| (r100) | 0.63 | 1.05 | **0.81** |
| before mean \|bias\| (r100) | 1.08 | 1.39 | 1.26 |
| after fp-RMSE (r100) | 1.08 | 1.44 | **1.19** |

Bonus: on the fixed reference the station-day |bias| Wilcoxon — previously the
weak axis (p = 0.17) — is significant (**p = 0.0064**); site-clustered
bootstrap Δ mean |bias| −0.43 [−0.71, −0.13], p = 0.004.

The residual ~0.3 ppm direct-vs-AK gap is real and literature-consistent: the
operational absolute scale is anchored to TCCON by **direct, non-harmonized**
zero-intercept regression at every version — Wunch et al. (2017) for B7;
O'Dell et al. (2018) for v8 (AK correction explicitly neglected in the global
divisors, ~0.1 ppm mean; land 0.2–0.3 ppm apparent bias partly from this);
B10 DUG §3.2.3 / B11 DUG §4.2.3 (TCCON_Adjust, GGG2020 in B11). Ship/ATom
chains audited — NOT affected (ship compares `xco2_bc` directly to shipborne
EM27 XCO₂, Knapp et al. 2021 ESSD; ATom uses the verified-clean OCO-side
operator on in-situ dry profiles), so the ship ~+1 ppm ocean offset stands as
an independent direct-comparison observation.

Consequences for this plan:

- Do NOT try to close the AK-harmonized station bias with model-side changes.
  A per-sounding anomaly model cannot move the absolute scale; features that
  are common-mode within an orbit (e.g. sun–Earth distance, 11-yr solar cycle —
  evaluated and rejected 2026-07-07) cancel exactly in the target and act only
  as date proxies.
- Blend / calibration / regime experiments above should therefore be scored on
  fp-RMSE and on AK-invariant before→after deltas; the residual absolute
  AK-harmonized station bias (~0.3 ppm floor) is a property of the reference
  chain, to be *explained* in the manuscript (anchoring citations: Wunch 2017 →
  O'Dell 2018 → B10/B11 DUG), not optimized away.
- Any comparison table quoting AK-referenced numbers generated before
  2026-07-07 (including the TabM / structured-residual reports and the
  cloud-distance-grouped tables) mixes in the buggy reference — regenerate
  before quoting AK columns; direct-reference columns are unaffected.
