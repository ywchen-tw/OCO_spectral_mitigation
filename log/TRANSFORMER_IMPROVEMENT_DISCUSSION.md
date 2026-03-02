# FT-Transformer Improvement Proposals — Discussion Log

**Date:** 2026-03-02
**Context:** `src/models_transformer.py` — `UncertainFTTransformerRefined` architecture

---

## Proposal 1: GRN as FFN Replacement in Transformer Blocks

**Proposal summary:** Replace the standard FFN (`self.ff`) in `AdvancedTransformerBlock` with a
`GatedResidualNetwork`, preserving the parameter budget while adding per-sample gating.

**Verdict: Medium priority — implement after fixing OOD scaling.**

The physics motivation is sound: in glint-mode OCO-2, the O2-A band can become nearly
uninformative under heavy aerosol loading or near cloud edges. A gating mechanism allows the
model to dynamically suppress uninformative tokens on a per-sample basis.

### Implementation Notes

- The existing `GatedResidualNetwork` (L152-171) has an **internal `LayerNorm`**. Since
  `AdvancedTransformerBlock` already applies `self.norm2` before the FFN (pre-norm pattern),
  dropping GRN in as-is causes double-normalization. Fix by either:
  - Stripping `self.norm` from `GatedResidualNetwork` when used as FFN replacement, or
  - Adding an `apply_norm=False` constructor flag.
- Set `input_size == output_size == d_token` so the skip linear is identity-like (no extra
  parameters in the skip path).
- Do **not** stack GRN on top of the existing FFN — replace it. Stacking doubles parameter
  count without proportional benefit.

### GRN path (consistent with existing code / TFT convention)

```
GRN(x) = LayerNorm(skip(x) + gate(x) * dense(x))
```

where `dense = Linear → GELU → Linear → Dropout` and `gate = Linear → Sigmoid`.

---

## Proposal 2: QuantileTransformer Saturation / OOD Data

**Proposal summary:** The QT maps OOD feature values (higher albedo, AOD, etc. than seen in
training) to the 0 or 1 boundary. Multiple physically distinct states collapse to identical
embeddings, blinding self-attention to physical gradients. Switch to `RobustScaler` or
Yeo-Johnson Power Transform.

**Verdict: Highest priority — most likely cause of observed variance bloat in scatter plots.**

### Diagnosis

The `QuantileTransformer` maps each feature's empirical CDF to [0, 1] (with `clip=True` by
default in sklearn). Any value outside the training support is clipped to exactly 0.0 or 1.0.
The `MLPTokenizer`'s `Linear(1, d_token//2)` then produces an **identical embedding** for all
clipped samples — self-attention becomes blind to physical gradients in those regions.

The MLP is less affected because it operates on the full concatenated input vector and can still
extract signal from relative differences across features even when individual features are
clipped.

### Recommended Fix — Step by Step

1. **Zero-cost diagnostic first:** set `clip=False` in `QuantileTransformer`. If scatter
   variance decreases, the clipping hypothesis is confirmed.
2. If confirmed, switch the scaler in `FeaturePipeline` (`src/pipeline.py`):
   - **`RobustScaler`** (preferred first step): median/IQR centering, no clipping, OOD values
     extrapolate linearly. Simple and immediately effective.
   - **Yeo-Johnson** (`PowerTransformer`): handles skewness better; adds per-feature parameter
     fit. More complex to validate across dates.
3. Follow `RobustScaler` with `StandardScaler` (or `normalize=True`) to re-equalise feature
   variances. Without this, high-variance features (e.g. `aod_total`) dominate dot-product
   similarity in the attention computation.

### Trade-off to keep in mind

The QT's uniform marginal distribution is part of why all features contribute equally to
attention similarity. `RobustScaler` output is **not** uniformly distributed — re-tuning
`d_token` or adding a LayerNorm after the tokenizer may be needed.

---

## Proposal 3: Spectral-Order-Aware Positional Biases (RoPE / Group Embeddings)

**Proposal summary:** Add Rotary Positional Embedding (RoPE) or 1D positional encoding to
spectral tokens (O2A, WCO2, SCO2) to provide a "spectral continuity" prior. Optionally, use
hierarchical feature injection (geometric vs spectral group separation).

**Verdict: Lowest priority — RoPE is the wrong tool; group embeddings are the right analogue.**

### Why RoPE is a poor fit

RoPE encodes **relative position** in a continuous sequence — adjacent tokens in a sentence,
adjacent patches in an image. OCO-2 features are not a continuous sequence: O2A, WCO2, SCO2
are three discrete channels at 760 nm, 1600 nm, 2300 nm. There is no meaningful notion of
"adjacent" that RoPE can exploit.

### Better alternative: Group / Segment Embeddings

Add a learnable embedding vector per feature group (8 groups), summed into each token after
tokenization — the BERT segment-embedding analogue. This:
- Is simpler than RoPE
- Is physically grounded (groups defined semantically, not positionally)
- Aligns with the `groups` dict already defined in `plot_attention_map` (L578-596):
  `Spectral k-coeff`, `Geometry`, `Pressure & Meteo`, `Surface Albedo`, `CO2 Retrieval`,
  `SNR & Detector`, `AOD & Aerosol`, `Footprint`

Implementation: build a `nn.Embedding(n_groups, d_token)` and a `feature_to_group` index
tensor; add `group_emb[feature_to_group]` to the tokenizer output before the transformer layers.

---

## Proposal 4: Prior-Plus-Residual Target Decomposition

**Proposal summary:** Instead of predicting `xco2_bc_anomaly` directly, predict the residual
from a physically-derived prior (e.g. CarbonTracker / GEOS-Chem CO2 background). Reduces output
variance and keeps the model physically grounded for OOD scenes.

**Verdict: Medium-low priority — partially implemented already; full version requires new data.**

### What's already done

The current target `xco2_bc_anomaly` is already a deviation from the orbit-mean bias-corrected
XCO2 — it is not the raw retrieval. The model is already doing a form of residual prediction
relative to the ACOS bias-corrected product.

### What would be genuinely new

Predict `xco2_bc - xco2_model_prior` where `xco2_model_prior` is a spatially and temporally
varying CO2 background from a transport model (e.g. CT2019B, GEOS-Chem). This removes:
- The large-scale north-south CO2 gradient (~5–10 ppm pole-to-equator)
- The seasonal cycle (~8 ppm amplitude at NH mid-latitudes)

Both of which are currently absorbed into `xco2_bc_anomaly` when orbit-mean detrending fails
(e.g. short glint-mode orbits with few clear-sky soundings).

**Cost:** requires collocating CT/GEOS-Chem model output with each sounding — non-trivial data
engineering step not currently in the pipeline.

---

## Recommended Implementation Order

| Priority | Change | File(s) | Complexity | Expected Impact |
|---|---|---|---|---|
| 1 | QT `clip=False` diagnostic | `src/pipeline.py` | Trivial | Confirms OOD hypothesis |
| 2 | QT → RobustScaler + StandardScaler | `src/pipeline.py` | Low | High — fixes OOD boundary clipping |
| 3 | GRN-FFN swap in `AdvancedTransformerBlock` | `src/models_transformer.py` | Low | Medium — per-sample gating |
| 4 | Group embeddings (8 groups, learnable) | `src/models_transformer.py` | Medium | Low-Medium — spectral structure prior |
| 5 | Prior-Plus-Residual target (CT2019B prior) | Pipeline + data eng | High | Medium — only if CT prior available |

---

## Related Files

- `src/models_transformer.py` — model architecture (`UncertainFTTransformerRefined`, `AdvancedTransformerBlock`, `GatedResidualNetwork`)
- `src/pipeline.py` — `FeaturePipeline` (QT fitting and transform)
- `log/TRANSFORMER_GRN_UPGRADE.md` — earlier GRN upgrade notes
