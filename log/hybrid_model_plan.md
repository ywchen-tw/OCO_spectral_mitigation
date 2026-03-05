# Hybrid Dual-Tower Model Plan

## Background

The existing `UncertainFTTransformerRefined` (models_transformer.py) already implements:
- GRN-as-FFN in each Transformer block (AdvancedTransformerBlock:293)
- 8 domain-group segment embeddings (UncertainFTTransformerRefined:338)
- MLPTokenizer — per-feature non-linear tokenization (:244)
- [CLS] token aggregation (:349)
- 3-quantile output [q05, q50, q95] with Huber+pinball loss

Proposals for GRN-Transformer and group embeddings are **already implemented**.
The genuinely new value is the **Dual-Tower fusion** (MLP anchor + FT-Transformer).

---

## Architecture: HybridDualTower

```
Input [N, n_features]
  │
  ├─── MLP Branch ──────────────────────────────────→ repr [N, 128]
  │     Linear(n_feat → 256) → LayerNorm → GELU
  │     Linear(256 → 256)    → LayerNorm → GELU → Dropout(0.1)
  │     Linear(256 → 128)    → LayerNorm → GELU
  │     (stable linear anchor; captures first-order relationships)
  │
  └─── FT-Transformer Branch ───────────────────────→ CLS [N, d_token]
        MLPTokenizer → GroupEmbeddings → [CLS] prepend
        → N x AdvancedTransformerBlock (PreNorm + MHA + GRN-FFN + DropPath)
        → extract CLS token → GRN head → repr [N, d_token//2]
        (cross-feature interaction modeling)
        │
        └── project to [N, 128] via Linear(d_token//2 → 128)

Fusion [N, 256]  ← Concat([mlp_repr, ft_repr])
  │
  GRN(256 → 128)   ← adaptive per-sample weighting of both branches
  │
  Linear(128 → 3)  ← [q05, q50, q95] quantile outputs
```

### Design rationale

- **MLP branch**: provides a stable, regularised linear anchor. Even if the Transformer
  branch collapses during training, the MLP path ensures a reasonable baseline prediction
  is always available. Deep enough (3 layers) to capture non-linear feature interactions
  but shallow enough to avoid overfitting.

- **FT-Transformer branch**: reuses the existing `UncertainFTTransformerRefined` backbone
  (without the head) so group embeddings, MLPTokenizer, GRN-FFN, and DropPath are all
  preserved. Only the regression head is stripped and replaced by the shared fusion head.

- **GRN fusion**: a single `GatedResidualNetwork(256 → 128)` adaptively weights the two
  branches per sample. This is equivalent to content-adaptive attention pooling over the
  [MLP_repr, CLS_repr] pair but is cheaper and more stable in practice.

- **Single quantile head**: both branches share one [q05, q50, q95] head, so the
  uncertainty calibration training signal propagates back through both branches.

---

## What is deliberately excluded

| Proposal | Reason excluded |
|---|---|
| BatchEnsemble (k members) | k× memory/compute overhead; mainly improves calibration not R²; revisit if distribution-shift is a confirmed problem |
| DCNv2 cross-layers | Transformer self-attention already models cross-feature products; DCNv2 alongside it creates redundancy |
| PLR embeddings (Periodic + Linear + ReLU) | QT normalisation + MLPTokenizer already handle non-linearity; add only if MLP branch underperforms |
| Attention Pooling for fusion | Over a 2-vector pool it degenerates to weighting one branch; GRN is equally expressive and more stable |
| GRN-as-FFN swap | Already implemented in existing AdvancedTransformerBlock |
| Group embeddings | Already implemented in existing UncertainFTTransformerRefined |

---

## Stage Plan

### Stage 1 — Dual-Tower (implement now)

1. **New file**: `src/models_hybrid.py`
   - `HybridDualTower(nn.Module)`: MLP branch + FT-Transformer backbone + GRN fusion + linear head
   - FT-Transformer backbone = `UncertainFTTransformerRefined` with head replaced by a projection layer
   - Reuse `GatedResidualNetwork`, `AdvancedTransformerBlock`, `MLPTokenizer` from `models_transformer.py`
   - Reuse `_MLP` / `_ResBlock` from `model_adapters.py` or implement a new `_MLPBranch`
   - Loss functions: import `huber_pinball_loss` / `quantile_loss` from `models_transformer.py`
   - Visualisation: `plot_evaluation_by_regime`, `plot_permutation_importance`, `plot_attention_map`
     — adapted versions that work with the dual-tower forward signature

2. **New adapter**: add `HybridAdapter` to `src/model_adapters.py`
   - Checkpoint: `model_hybrid_best.pt` + `hybrid_adapter.json` (stores n_features, d_token, n_heads, n_layers, d_ff, mlp_hidden, tokenizer_type, feature_names)
   - `can_load(dir)`, `load(dir)`, `save(dir)` — same pattern as `FTAdapter`

3. **New training script**: `src/hybrid_models.py`
   - CLI mirroring `models_transformer.py`: `--sfc_type`, `--suffix`, `--pipeline`, `--loss`, `--huber-delta`
   - Output dir: `results/model_hybrid/<suffix>/`
   - Saves: `model_hybrid_best.pt`, `hybrid_adapter.json`, `pipeline.pkl` (if fitted)

4. **Update `src/apply_models.py`**:
   - Add `--hybrid-dir` argument
   - Load `HybridAdapter` and run inference alongside Ridge/MLP/FT/XGB

5. **Update shell scripts**:
   - Add `hybrid_models.py` invocation to `curc_shell_blanca_train_compare.sh`
   - Add `--hybrid-dir` to `apply_models.py` invocation

### Stage 2 — PLR embeddings (conditional)

Add only if Stage 1 MLP branch shows poor performance on high-frequency features
(e.g. cloud-boundary transitions in `cld_dist_km`, sharp `aod_total` regime changes).

Implementation: `PeriodicEmbedding(n_features, n_frequencies)` — applies
`[sin(2π f x), cos(2π f x)]` for learnable frequencies f, then a linear projection to
d_token. Applied to the MLP branch's input only (Transformer branch already has MLPTokenizer).

---

## Key hyperparameters (initial values)

| Parameter | Value | Notes |
|---|---|---|
| MLP hidden dims | [256, 256, 128] | |
| FT d_token | 128 | Reduced from 256 to save memory (MLP branch carries some capacity) |
| FT n_heads | 8 | |
| FT n_layers | 4 | |
| FT d_ff | 256 | |
| fusion hidden | 128 | GRN output dim |
| batch size | 1024 | |
| optimizer | AdamW(lr=1e-4, wd=1e-4) | |
| LR schedule | CosineAnnealingLR(T_max=patience, eta_min=1e-6) | |
| early stop patience | 50 | |
| max epochs | 500 (CURC) / 100 (local) | |
| loss | huber_pinball (default) | Huber δ=1.0 for q50, pinball for q05/q95 |

---

## File changes summary

| File | Change |
|---|---|
| `src/models_hybrid.py` | **New** — HybridDualTower class + training + evaluation |
| `src/model_adapters.py` | Add HybridAdapter |
| `src/apply_models.py` | Add --hybrid-dir argument and inference path |
| `curc_shell_blanca_train_compare.sh` | Add hybrid training + apply_models invocation |
