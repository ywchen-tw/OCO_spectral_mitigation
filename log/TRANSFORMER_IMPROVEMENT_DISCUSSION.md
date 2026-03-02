# FT-Transformer Improvement Proposals — Discussion Log

**Date:** 2026-03-02
**Context:** `src/models_transformer.py` — `UncertainFTTransformerRefined` architecture

---

## Status Summary (as of 2026-03-02)

| Proposal | Status | Where |
|---|---|---|
| GRN as FFN replacement | **Done** | `AdvancedTransformerBlock` — `self.ff = GatedResidualNetwork(...)` |
| Robust Scaling (RobustScaler + StandardScaler) | **Done** | `pipeline.py` — `RobustStandardScaler` + `ClipFreeQuantileTransformer` |
| ResNet-style MLP | **Done** | `model_adapters.py` — `_MLP` uses 4 × `_ResBlock(LN→GELU→Linear→LN→GELU→Dropout→Linear + skip)` |
| Group / Segment Embeddings (Transformer) | **Done** | `models_transformer.py` — `_FEATURE_GROUPS`, `_build_feature_to_group`, `group_emb` in `UncertainFTTransformerRefined` |

Open proposals from the 2026-03-02 review are documented below.

---

## DONE — Proposal 1: GRN as FFN Replacement

**Verdict: Already implemented.**

`AdvancedTransformerBlock` (`src/models_transformer.py:212-235`) replaces the standard FFN with
`GatedResidualNetwork`. The per-sample gating suppresses uninformative tokens (e.g. saturated O2-A
band under heavy aerosol). The older `TransformerBlock` and `TransformerBlockWithExtraction`
classes retain plain FFNs but are not used in the production `UncertainFTTransformerRefined`.

### Implementation notes (for reference)

- The `GatedResidualNetwork` has an internal `LayerNorm`. `AdvancedTransformerBlock` applies
  `self.norm2` before calling the GRN (`x + drop_path(self.ff(self.norm2(x)), ...)`), so
  double-normalization is avoided by the GRN's internal norm acting on the already-normalised input.
- `input_size == output_size == d_token`: the skip linear is square (no dimension change) and
  the residual stream is preserved.

---

## DONE — Proposal 2: QuantileTransformer Saturation / OOD Scaling

**Verdict: Already implemented.**

`pipeline.py` provides two OOD-safe scalers, both fully integrated:

- **`ClipFreeQuantileTransformer`** (L43-119): wraps `QuantileTransformer` and replaces hard
  boundary clipping with linear extrapolation using the local slope at the training support edge.
  Preserves the normal-distribution output of QT for in-distribution data while avoiding
  "attention blindness" at the OOD boundary.

- **`RobustStandardScaler`** (L122-154): `RobustScaler(median/IQR)` → `StandardScaler`.
  OOD values extrapolate linearly; re-equalised feature variances prevent high-IQR features
  (e.g. `aod_total`) from dominating attention dot-products.

`FeaturePipeline.fit()` currently uses `RobustStandardScaler` (line 301). The
`ClipFreeQuantileTransformer` is available as an alternative for comparison runs.

---

## DONE — Proposal 3: ResNet-style MLP

**Verdict: Already implemented.**

`_MLP` in `model_adapters.py` (L59-83) uses:
```
Linear projection → LayerNorm → GELU → Dropout(0.1)
→ 4 × _ResBlock(LN → GELU → Linear → LN → GELU → Dropout → Linear + skip)
→ Linear head
```
Pre-activation residual connections (LN before the non-linearity) are exactly the "physical
anchor" described in the proposal. Target is median-centered + IQR-scaled before training.

---

## DONE — Proposal 4: Group / Segment Embeddings (Transformer)

**Verdict: Implemented 2026-03-02.**

### Motivation

The pipeline features span 8 physical domains (Spectral k-coeff, Geometry, Pressure & Meteo,
Surface Albedo, CO₂ Retrieval, SNR & Detector, AOD & Aerosol, Footprint). Self-attention treats
all tokens as structurally equivalent. A learnable group embedding (BERT segment-embedding
analogue) injects the prior that `o2a_k1`/`o2a_k2`/`wco2_k1`/... share a spectral retrieval
context, and that `mu_sza`/`mu_vza`/`cos_glint_angle` share a geometric context.

### Feature → group mapping (derived from `pipeline.py _FEATURES_SFC0`)

| Group | Features |
|---|---|
| Spectral k-coeff | `o2a_k1`, `o2a_k2`, `wco2_k1`, `wco2_k2`, `sco2_k1`, `sco2_k2`, `o2a_intercept`, `wco2_intercept`, `sco2_intercept` |
| Geometry | `mu_sza`, `mu_vza`, `cos_glint_angle`, `pol_ang_rad` (+ `sin_raa`, `cos_raa` for sfc_type=1) |
| Pressure & Meteo | `log_P`, `dp`, `h2o_scale`, `delT`, `co2_grad_del`, `ws`, `s31`, `s32`, `airmass_sq` |
| Surface Albedo | `alb_o2a`, `alb_wco2`, `alb_sco2`, `alb_o2a_over_cos_sza`, `alb_wco2_over_cos_sza`, `alb_sco2_over_cos_sza` |
| CO₂ Retrieval | `xco2_raw_minus_apriori`, `co2_ratio_bc`, `h2o_ratio_bc`, `xco2_strong_idp`, `xco2_weak_idp` |
| SNR & Detector | `csnr_o2a`, `csnr_wco2`, `csnr_sco2`, `snr_o2a`, `snr_wco2`, `snr_sco2`, `h_cont_o2a`, `h_cont_wco2`, `h_cont_sco2`, `max_declock_o2a`, `max_declock_wco2`, `max_declock_sco2` |
| AOD & Aerosol | `aod_total` |
| Footprint | `fp_0` … `fp_7` |

### Implementation (as deployed)

**`src/models_transformer.py`**

- `_FEATURE_GROUPS` (L30–52): module-level ordered dict — single source of truth for both the
  model and `plot_attention_map`. Replaces the stale local `groups` dict that referenced many
  features no longer in the pipeline.
- `_build_feature_to_group(feature_names)` (L55–84): maps each ordered feature name to its group
  index (0-based, insertion order of `_FEATURE_GROUPS`). Warns on unknown features.
- `UncertainFTTransformerRefined.__init__` gains `feature_names: list | None = None`:
  - Creates `nn.Embedding(8, d_token)` initialised with `trunc_normal_(std=0.02)`
  - Registers `feature_to_group` as a device buffer (moves with `.to(device)` automatically)
- `forward()`: `x = x + self.group_emb(self.feature_to_group).unsqueeze(0)` inserted
  after tokenizer output, before [CLS] prepend

**`src/model_adapters.py`**

- `FTAdapter._ARCH_VERSION` bumped to **3** — stale v2 checkpoints auto-rejected, forcing retrain
- `FTAdapter.__init__` / `save()` / `load()`: `feature_names` stored in `ft_meta.pkl` and
  passed back to the model constructor on load

**Parameter cost**: 8 × 256 = **2 048** new learnable parameters (negligible).
**Backward compat**: `feature_names=None` → `group_emb = None`, no-op in `forward()`.
**`plot_attention_map`**: local `groups` dict replaced with `groups = _FEATURE_GROUPS`.

---

## Open — Proposal 5: Attention Pooling (replaces [CLS])

**Verdict: Moderate priority — implement after segment embeddings.**

The learnable `[CLS]` token is updated during training and adapts to the training distribution,
but its aggregation strategy is fixed at inference time and may not generalise optimally to OOD
scenes. Attention pooling generates a content-adaptive query from the mean of the token sequence:

```python
# Replaces x[:, 0] (CLS extraction) after the last transformer block
query = x.mean(dim=1, keepdim=True)           # [batch, 1, d_token]
attn_pool = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
x_global, _ = attn_pool(query, x, x)          # [batch, 1, d_token]
x_global = x_global.squeeze(1)                # [batch, d_token]
```

**Caution:** adds one extra attention operation per forward pass. The OOD concern is real but
minor; check whether segment embeddings alone resolve the scatter "bloat" first.

---

## Open — Proposal 6: DCNv2 Cross-Layers (MLP)

**Verdict: Highest-priority new MLP suggestion — physically motivated.**

The `_MLP` learns feature interactions implicitly through depth. DCNv2 parallel cross-layers
explicitly model multiplicative interactions (e.g. `airmass_sq × aod_total`,
`cos_glint_angle × alb_o2a_over_cos_sza`) which appear directly in radiative transfer equations
and are inefficiently approximated by stacked residual blocks.

### Architecture sketch

```
input x
  ├── Cross stream: x₀ ⊗ (W_c @ x + b_c) applied for L_c layers
  └── Deep stream:  existing _MLP residual blocks
  └── concat(cross_out, deep_out) → Linear → 1
```

This does not replace `_MLP` — run as a parallel experiment first to compare R² on the
validation set before committing.

---

## Open — Proposal 7: PLR Feature Embeddings (MLP)

**Verdict: Low priority given current pipeline.**

Periodic + Linear + ReLU (PLR) embeddings help MLPs learn high-frequency functions. With
`RobustStandardScaler` already providing smooth OOD extrapolation, and `_ResBlock` using GELU
(which provides some frequency expansion), the marginal benefit is expected to be small for
this dataset. The sharp physical transitions are in `cld_dist_km` which is not a model input.
Revisit if the MLP shows underfitting on narrow feature-value ranges.

---

## Open — Proposal 8: TabM / BatchEnsemble (MLP)

**Verdict: Low priority — check permutation importance plots first.**

BatchEnsemble simulates an ensemble of MLPs via rank-1 member-specific adapters, improving
generalisation on 1M+ tabular samples. With patience=30 early stopping already applied, a
single `_MLP` is unlikely to catastrophically overfit. Justified only if regime-specific
overfitting is observed in the permutation importance or the 3×5 evaluation plots (e.g. good
R² on clear-sky FPs, poor on cloud-affected FPs). Profile first.

---

## Open — Proposal 9: Prior-Plus-Residual Target Decomposition

**Verdict: Medium-low priority — partially done; full version requires new data.**

`xco2_bc_anomaly` is already a deviation from the orbit-local clear-sky mean, removing
large-scale gradients when sufficient clear-sky soundings are available. The next step —
predicting `xco2_bc − xco2_ct2019` where `xco2_ct2019` is a CarbonTracker / GEOS-Chem prior
— would additionally remove:
- The north-south CO₂ gradient (~5–10 ppm pole-to-equator)
- The seasonal cycle (~8 ppm at NH mid-latitudes)

These appear in `xco2_bc_anomaly` when short glint-mode orbits lack clear-sky soundings.
Cost: requires collocating CT2019B model output with every sounding — non-trivial new data
engineering not currently in the pipeline.

---

## Recommended Implementation Order

| Priority | Change | File(s) | Complexity | Expected Impact |
|---|---|---|---|---|
| ~~1~~ | ~~Group / segment embeddings~~ | ~~`src/models_transformer.py`~~ | ~~Low~~ | ~~Done~~ |
| 1 | DCNv2 cross-layers (MLP) | `src/model_adapters.py` | Medium | Medium-High — explicit interaction terms |
| 2 | Attention pooling (replace [CLS]) | `src/models_transformer.py` | Low | Low-Medium — assess after segment emb training results |
| 3 | PLR embeddings (MLP) | `src/model_adapters.py` | Medium | Low (redundant with current scaler) |
| 4 | TabM / BatchEnsemble (MLP) | `src/model_adapters.py` | High | Low-Medium (profile first) |
| 5 | Prior-Plus-Residual (CT2019B) | Pipeline + data eng | High | Medium (requires external data) |

---

## Related Files

- `src/models_transformer.py` — `UncertainFTTransformerRefined`, `AdvancedTransformerBlock`, `GatedResidualNetwork`, `plot_attention_map`
- `src/model_adapters.py` — `_MLP`, `_ResBlock`, `RidgeAdapter`, `MLPAdapter`, `FTAdapter`
- `src/pipeline.py` — `FeaturePipeline`, `RobustStandardScaler`, `ClipFreeQuantileTransformer`
- `src/mlp_lr_models.py` — training loop, `mitigation_test`, `compute_xco2_anomaly_date_id`
