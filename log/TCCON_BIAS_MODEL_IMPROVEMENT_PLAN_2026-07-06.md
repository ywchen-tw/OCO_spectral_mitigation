# TCCON Bias and fp-RMSE Improvement Plan

Date: 2026-07-06

## Goal

Improve corrected XCO2 agreement with TCCON, with two primary metrics:

- lower after-correction mean absolute station bias;
- lower per-footprint RMSE around TCCON stations.

**Metric caveat (added 2026-07-07):** under the AK-harmonized reference, the
station-mean bias has a floor of roughly |ak_delta| ≈ 0.9 ppm that no
per-sounding anomaly model can cross — it is an absolute-scale offset inherited
from the operational product's direct-TCCON anchoring, not model error (see the
final section below). Judge blending/calibration experiments on fp-RMSE and on
AK-invariant before→after improvements, not on the absolute AK-harmonized
station bias.

The current best result is still the regularized deep ensemble. The structured
residual model is close, but does not yet clearly beat DE on the TCCON objective.

## Current reference results

For the local r50 km TCCON comparison:

| model set | after mean \|bias\| | after fp-RMSE | note |
| --- | ---: | ---: | --- |
| DE o05/l15 | ~0.93 ppm | ~1.36 ppm | current best |
| Structured o10/l10 | ~1.03 ppm | ~1.53 ppm | shared cloud-distance target |
| Structured o05/l15 | ~1.00 ppm | ~1.51 ppm | closer to DE |
| Structured o05/l15, drop-guarded | ~0.96 ppm | ~1.37 ppm | close to DE after excluding guarded footprints |
| TabM o05/l15 | ~1.09 ppm | ~1.78 ppm | weaker than DE and structured |

Interpretation: the structured residual model is not fundamentally failing.
It is close to DE, especially after guard filtering, but it appears slightly
less robust for station-level bias and high-risk footprints.

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

## Direct vs AK-harmonized station bias (added 2026-07-07)

Observation: after correction, mean |station bias| is ~0.63 ppm against the
direct TCCON window mean but ~1.05 ppm against the AK/prior-harmonized
reference (r100), while fp-RMSE improves under both references.

Diagnosis — this is arithmetic, not a model failure:

- Harmonization shifts the TCCON reference by ak_delta = −0.93 ± 0.74 ppm.
- Before→after improvements are exactly AK-invariant (same soundings, same
  operator on both sides of the difference).
- Once the correction pulls OCO-2 close to the *direct* TCCON mean, the small
  residuals are dominated by the common-mode shift, so the AK-harmonized
  after-|bias| lands near |ak_delta|. RMSE still improves because it is
  dominated by the scatter collapse; a constant shift only adds in quadrature.

Root cause (literature-verified 2026-07-07, primary sources read): the
operational product's absolute scale is anchored to TCCON by **direct,
non-AK-harmonized** zero-intercept regression at every version — Wunch et al.
(2017) for B7; O'Dell et al. (2018) for v8, with the explicit statement that
the AK correction was neglected when solving the global divisors (~0.1 ppm
mean; land 0.2–0.3 ppm apparent bias partly from this); B10 DUG §3.2.3 and B11
DUG §4.2.3 (TCCON_Adjust, GGG2020 in B11) carry the same procedure forward. So
agreement with direct TCCON is partly built into `xco2_bc`, and harmonization
re-exposes an offset. The ship EM27 ~+1 ppm ocean offset independently
corroborates the harmonized picture.

Magnitude caveat: the literature's neglected-AK term (~0.1–0.3 ppm) covers only
part of our −0.93; the remainder is plausibly the QF0+1 near-cloud operator
population in `ak_harmonize.py` (near-cloud AKs deviate far more from 1) and/or
the gamma-scaled GGG2020 prior proxy. Three diagnostics before the manuscript:

1. **a≡1 null check** — recompute `c_est` with the averaging kernel forced to
   1; must give delta ≈ 0 (nonzero ⇒ units/pwf/interpolation bug).
2. **QF0-only clear-sky/far-cloud ak_delta subset** — should land near
   0.1–0.3 ppm (matching O'Dell); the gap to −0.93 is then attributable to the
   near-cloud population and is itself a quotable result.
3. **Per-case signed identity** — bias_AK vs (bias_direct − ak_delta) must fall
   on the 1:1 line, confirming pure common-mode rather than regime dependence.

Consequences for this plan:

- Do NOT try to close the AK-harmonized station bias with model-side changes.
  A per-sounding anomaly model cannot move the absolute scale; features that
  are common-mode within an orbit (e.g. sun–Earth distance, 11-yr solar cycle —
  evaluated and rejected 2026-07-07) cancel exactly in the target and act only
  as date proxies.
- Blend / calibration / regime experiments above should therefore be scored on
  fp-RMSE and on AK-invariant before→after deltas; the absolute AK-harmonized
  station bias is a property of the reference chain, to be *explained* in the
  manuscript (anchoring citations: Wunch 2017 → O'Dell 2018 → B10/B11 DUG),
  not optimized away.
