# FT-Transformer development line (historical, consolidated 2026-07-06)

> Consolidated from `TRANSFORMER_IMPROVEMENT_DISCUSSION.md` (2026-03-02) and
> `TRANSFORMER_GRN_UPGRADE.md` (~2026-03). The FT-Transformer is no longer in the tree
> (`src/models/transformer.py` removed 2026-07-03; see PIPELINE_CHANGELOG.md) — production is
> the per-surface deep ensemble (`de_*_beta_nll_prof_reg`). Kept for reference.

---

# Part 1 — Improvement discussion (nine proposals)

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

---

# Part 2 — GRN + per-feature tokenizer upgrade plan (implements Proposal 1)

# Transformer Upgrade: GRN Head + MLP Tokenizer

**Goal**: Improve `UncertainFTTransformerRefined` to reduce scatter on the
predicted-vs-true XCO2 anomaly plot by replacing the linear feature tokenizer
with a per-feature MLP tokenizer and replacing the regression head with a
Gated Residual Network (GRN).

**File**: `src/models_transformer.py` (primary), `src/model_adapters.py`,
`src/fitting_data_correction.py`

---

## Summary of Changes

| # | Component | Before | After | Status |
|---|---|---|---|---|
| 1 | `GatedResidualNetwork` class | (missing) | New class added | [x] |
| 2 | `MLPTokenizer` class | (missing) | New class added | [x] |
| 3a | `UncertainFTTransformerRefined` tokenizer | `FeatureTokenizer` | `MLPTokenizer` (with `tokenizer_type` param) | [x] |
| 3b | `UncertainFTTransformerRefined` head | 2-layer GELU MLP | `GatedResidualNetwork` | [x] |
| 4 | `FTAdapter` | no `tokenizer_type` | propagate `tokenizer_type` in save/load | [x] |
| 5 | `models_transformer.py` call site | no `tokenizer_type` | `tokenizer_type='mlp'` | [x] |

Pre-LayerNorm and GELU in `AdvancedTransformerBlock` are **already implemented** — no change needed.

---

## Detailed Steps

### Step 1 — Add `GatedResidualNetwork` (after `PreNormResidual`, ~line 160)

File: `src/models_transformer.py`

```python
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.skip_connection = nn.Linear(input_size, output_size)
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip_connection(x)
        gated_output = self.gate(x) * self.dense(x)
        return self.norm(residual + gated_output)
```

**Why**: The gated skip connection lets the Transformer selectively suppress
high-frequency noise in the prediction, tightening the 1:1 scatter alignment.

Status: [ ]

---

### Step 2 — Add `MLPTokenizer` (after `FeatureTokenizer`, ~line 150)

File: `src/models_transformer.py`

Keep `FeatureTokenizer` for backward-compat loading of existing checkpoints.
Add `MLPTokenizer` as a new class:

```python
class MLPTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_token // 2),
                nn.GELU(),
                nn.Linear(d_token // 2, d_token)
            ) for _ in range(n_features)
        ])

    def forward(self, x):
        # x: [batch, n_features]
        tokens = [self.tokenizers[i](x[:, i:i+1]) for i in range(len(self.tokenizers))]
        return torch.stack(tokens, dim=1)   # [batch, n_features, d_token]
```

**Why**: A single linear projection of a scalar into a 64-D / 256-D token is
unstable for noisy inputs like cloud distance. Two layers with GELU allow each
physical feature to build a richer initial representation before attention.

Status: [ ]

---

### Step 3a — Swap tokenizer in `UncertainFTTransformerRefined.__init__` (~line 201)

File: `src/models_transformer.py`

Add `tokenizer_type: str = 'mlp'` parameter; choose tokenizer accordingly:

```python
# Before:
self.tokenizer = FeatureTokenizer(n_features, d_token)

# After:
if tokenizer_type == 'mlp':
    self.tokenizer = MLPTokenizer(n_features, d_token)
else:
    self.tokenizer = FeatureTokenizer(n_features, d_token)
self.tokenizer_type = tokenizer_type
```

Status: [ ]

---

### Step 3b — Replace head in `UncertainFTTransformerRefined.__init__` (~lines 210-214)

File: `src/models_transformer.py`

```python
# Before:
self.head = nn.Sequential(
    nn.Linear(d_token, d_token // 2),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(d_token // 2, 3),
)

# After:
self.head = GatedResidualNetwork(
    input_size=d_token, hidden_size=d_token // 2,
    output_size=3, dropout=0.1
)
```

`forward()` is **unchanged** — `self.head(x)` still returns `[batch, 3]`.

Status: [ ]

---

### Step 4 — Update `FTAdapter` to propagate `tokenizer_type`

File: `src/model_adapters.py`

Three locations:

**4a. `__init__`** (~line 253): add parameter
```python
def __init__(self, model, n_features, d_token=128, n_heads=8,
             n_layers=4, d_ff=256, tokenizer_type='mlp'):
    ...
    self.tokenizer_type = tokenizer_type
```

**4b. `save()`** (~line 287): add to meta dict
```python
meta = {
    'n_features':     self.n_features,
    'd_token':        self.d_token,
    'n_heads':        self.n_heads,
    'n_layers':       self.n_layers,
    'd_ff':           self.d_ff,
    'tokenizer_type': self.tokenizer_type,   # ← new
}
```

**4c. `load()`** (~line 324): pass to model constructor with safe default for
old checkpoints that pre-date this change:
```python
model = UncertainFTTransformerRefined(
    n_features=meta['n_features'],
    d_token=meta['d_token'],
    n_heads=meta['n_heads'],
    n_layers=meta['n_layers'],
    d_ff=meta['d_ff'],
    tokenizer_type=meta.get('tokenizer_type', 'linear'),  # 'linear' = old FeatureTokenizer
)
```

Status: [ ]

---

### Step 5 — Update training call site in `fitting_data_correction.py` (~line 1435)

File: `src/fitting_data_correction.py`

```python
# Before:
FTAdapter(model, n_features=pipeline.n_features,
          d_token=256, n_heads=8, n_layers=4, d_ff=512).save(output_dir)

# After:
FTAdapter(model, n_features=pipeline.n_features,
          d_token=256, n_heads=8, n_layers=4, d_ff=512,
          tokenizer_type='mlp').save(output_dir)
```

Status: [ ]

---

## Backward Compatibility Notes

- **`FeatureTokenizer` is kept** in the file so that old `model_best.pt`
  checkpoints still load when `ft_meta.pkl` has `tokenizer_type='linear'` or
  is absent (defaults to `'linear'` in `FTAdapter.load`).
- **`plot_attention_map`** accesses `model.tokenizer(sample_batch)` — the
  output shape `[batch, n_features, d_token]` is identical for both tokenizer
  types, so no changes needed there.
- **New `model_best.pt` checkpoints are not backward-compatible** with old
  `FTAdapter.load` builds because the head state-dict keys differ. Existing
  trained models must be retrained after this change.

---

## Parameter Count Impact (d_token=256, n_features≈80)

| Component | Before | After | Delta |
|---|---|---|---|
| Tokenizer | `80 × 256 × 2` ≈ 41 K | `80 × (1×128 + 128×256)` ≈ 2.6 M | +2.6 M |
| Head | `256×128 + 128×3` ≈ 33 K | `256×3 × 3` ≈ 2.3 K | ≈ same |

The MLP tokenizer is the dominant cost — verify GPU memory on Blanca is
sufficient before kicking off a full training run.

---

## Testing Checklist (after implementation)

- [x] `python src/models_transformer.py` imports without error
- [x] `UncertainFTTransformerRefined(n_features=12)` forward pass produces `[batch, 3]`
- [x] `MLPTokenizer` forward produces `[batch, n_features, d_token]`
- [x] `GatedResidualNetwork(D, D//2, 3)` forward produces `[batch, 3]`
- [x] `return_attn=True` path returns correct attn shape `[batch, n_features, n_features]`
- [x] `tokenizer_type='linear'` backward-compat path still works
- [x] `FTAdapter.save` writes `tokenizer_type` into `ft_meta.pkl`
- [x] `FTAdapter.load` on old checkpoint (no `tokenizer_type`) defaults to `'linear'`
- [ ] `plot_attention_map` runs end-to-end without shape errors
- [ ] Training run on local machine (50 epochs) converges — compare val MAE vs old head
