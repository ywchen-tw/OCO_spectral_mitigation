# PCA Integration Plan — XCO2 Bias Correction Models

**Date**: 2026-04-06
**Author**: derived from PCA analysis of land/ocean OCO-2 glint soundings
**Status**: Planning

---

## 1. Motivation

PCA of the 43-feature retrieval space (full) and 28-feature reduced space reveals
physically interpretable axes that correlate strongly with `cld_dist_km`:

| Surface | PC | r(cld\_dist) | Dominant features | Physical meaning |
|---------|----|-------------|-------------------|-----------------|
| Land    | PC1 (18.2%) | **+0.358** | exp\_intercepts, O₂A k₂/k₃ | Surface brightness |
| Land    | PC4 (7.4%)  | **+0.215** | SCO₂ k₂/k₃, log\_P, tcwv  | Pressure/moisture regime |
| Land    | PC8 (3.7%)  | **−0.208** | csnr\_o2a, csnr\_sco2      | SNR (clouds scatter signal) |
| Ocean   | PC3 (8.3%)  | **+0.203** | SCO₂ vs WCO₂ contrast      | Band absorption contrast |
| Ocean   | PC6 (4.2%)  | **−0.216** | csnr\_o2a, csnr\_sco2      | SNR |
| Ocean (reduced) | PC4 (6.7%) | **−0.310** | csnr\_o2a, csnr\_sco2 | SNR (strongest single predictor) |

Key findings driving the integration:
1. **Surface brightness is the dominant land covariate** — bright arid surfaces are
   geographically anti-correlated with clouds. The model must see this axis explicitly.
2. **SNR is a universal cloud-proximity signal** on both surfaces (r ≈ −0.21 to −0.31).
3. **Collinear feature groups** (k₁/k₂/k₃ per band, albedo channels, dp/dp\_psfc\_prior\_ratio)
   create redundant gradient directions that slow MLP/Ridge convergence.
4. **XGBoost is immune** to collinearity via its split-gain criterion — PCA whitening
   provides no benefit and hurts SHAP interpretability for XGBoost.
5. **FT-Transformer tokenises each feature independently** — PCA whitening destroys
   per-feature token identity and segment embeddings; PC scores must be *appended*
   as additional tokens rather than replacing raw features.

---

## 2. Architecture Overview

All five regression scripts and two classifier scripts share a single
`FeaturePipeline` from `pipeline.py`:

```
FeaturePipeline
  ├── qt_features   ~25–30 continuous features (sfc-type specific)
  ├── qt            RobustStandardScaler  ← primary change point
  ├── fp_cols       fp_0..7 one-hot (appended raw, untransformed)
  └── transform()  → [qt_scaled | fp_onehot]   shape [N, ~33–38]
```

Downstream consumers:

| Script | Model | PCA strategy |
|--------|-------|-------------|
| `mlp_lr_models.py` | `_MLP` (ResBlock) + Ridge | PCAWhitening OR PCAScoreAppender |
| `xgb_models.py` | XGBRegressor | PCAScoreAppender only (no whitening) |
| `models_transformer.py` | `UncertainFTTransformerRefined` | PCAScoreAppender only |
| `models_hybrid.py` | `HybridDualTower` (MLP + FT) | PCAScoreAppender for both branches |
| `mlp_lr_cloud_classifier.py` | `_MLP` classifier + LogisticRegression | PCAWhitening OR PCAScoreAppender |
| `transformer_cloud_classifier.py` | `FTClassifier` | PCAScoreAppender only |

---

## 3. New Classes in `pipeline.py`

### 3.1 `PCAWhitening`

Drop-in replacement for `RobustStandardScaler` stored in `FeaturePipeline.qt`.
Chains `RobustScaler → PCA(n_components, whiten=True)`.

```python
class PCAWhitening:
    """RobustScaler → PCA(n_components, whiten=True).

    Output dimensionality = n_components (≤ n_qt_features).
    Intended for MLP / Ridge / Logistic — models that treat input as a flat
    vector and benefit from decorrelated, unit-variance features.
    NOT suitable for FT-Transformer (destroys per-feature token identity).
    """
    __module__ = 'pipeline'

    def __init__(self, n_components: int | float = 0.90):
        # n_components: int → exact count; float in (0,1) → variance fraction
        self._robust = RobustScaler()
        self._pca    = PCA(n_components=n_components, whiten=True, random_state=42)

    def fit(self, X: np.ndarray) -> 'PCAWhitening': ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def n_components_(self) -> int: ...       # actual number of PCs retained
    @property
    def explained_variance_ratio_(self): ...  # forwarded from inner PCA
```

**Impact on `FeaturePipeline`**:
- `pipeline.qt_features` still records original named features (needed by `transform()`
  for column selection).
- `pipeline.features` becomes `['pca_pc1', ..., 'pca_pcN'] + fp_cols`.
- `pipeline.n_features` changes from ~33–38 to `n_pca_components + 8`.
- `_MLP(n)` and `RidgeAdapter` accept any `n` — no model code changes.

### 3.2 `PCAScoreAppender`

Fits a PCA on top of already-scaled features; appends selected PC scores to the
feature matrix. Preserves all original features (token stream intact for FT-Transformer).

```python
class PCAScoreAppender:
    """Appends selected PCA score columns to the pipeline output.

    After fitting, FeaturePipeline.transform() produces:
      [qt_scaled_features | selected_pc_scores | fp_onehot]

    Selected PCs are chosen per sfc_type based on their correlation with
    cld_dist_km (from the PCA analysis):
      sfc_type=1 (land):  PC1, PC4, PC8  (r = +0.36, +0.22, −0.21)
      sfc_type=0 (ocean): PC3, PC6       (r = +0.20, −0.22)
    """
    __module__ = 'pipeline'

    # 0-indexed PC columns to append, keyed by sfc_type
    _DEFAULT_PC_IDX: dict = {
        1: [0, 3, 7],   # land:  PC1, PC4, PC8
        0: [2, 5],      # ocean: PC3, PC6
    }

    def __init__(self, pc_idx: list[int] | None = None):
        self._pca    = PCA(n_components=8, random_state=42)
        self._robust = RobustScaler()   # same scaling as pipeline.qt before PCA
        self._pc_idx = pc_idx           # None → use _DEFAULT_PC_IDX[sfc_type]

    def fit(self, X_scaled: np.ndarray, sfc_type: int) -> 'PCAScoreAppender': ...
    def transform_append(self, X_scaled: np.ndarray,
                         sfc_type: int) -> np.ndarray: ...
    # Returns [X_scaled | selected_scores], shape [N, n_qt + len(pc_idx)]

    @property
    def pc_names(self) -> list[str]: ...   # ['pca_pc1', 'pca_pc4', 'pca_pc8']
    @property
    def pc1_col_offset(self) -> int: ...   # index of first appended PC in full feature vec
```

### 3.3 `FeaturePipeline.fit()` new parameters

```python
@classmethod
def fit(cls, df: pd.DataFrame,
        sfc_type: int = 1,
        scaler: str = 'robust_standard',   # NEW: 'robust_standard' | 'pca_whitening'
        pca_augment: bool = False,          # NEW: append selected PC scores
        ) -> 'FeaturePipeline':
```

New stored attributes on `FeaturePipeline`:
```python
self.scaler_type    = scaler          # 'robust_standard' | 'pca_whitening'
self.pca_appender   = None            # PCAScoreAppender if pca_augment=True
self.pc1_col_idx    = None            # int index of PC1 col in transform() output
```

`FeaturePipeline.transform()` checks `self.pca_appender` and appends scores
between the QT-scaled block and `fp_cols`:
```
[qt_scaled | pca_scores (if appender set) | fp_onehot]
```

---

## 4. FT-Transformer: new feature group

**File**: `models_transformer.py`

Add `'PCA\nScores'` as the 9th group in `_FEATURE_GROUPS`:

```python
_FEATURE_GROUPS: dict = {
    'Spectral\nk-coeff':  [...],
    'Geometry':           [...],
    'Pressure\n& Meteo':  [...],
    'Surface\nAlbedo':    [...],
    'CO₂\nRetrieval':     [...],
    'SNR &\nDetector':    [...],
    'AOD &\nAerosol':     [...],
    'Footprint':          [...],
    'PCA\nScores':        ['pca_pc1', 'pca_pc2', 'pca_pc3',
                           'pca_pc4', 'pca_pc5', 'pca_pc6',
                           'pca_pc7', 'pca_pc8'],   # all 8 possible PC names
}
```

The FT tokenizer will embed each appended PC score as its own token. The segment
embedding for this group provides a learnable additive shift that distinguishes
PC-score tokens from raw retrieval tokens, while the CLS token can attend to them
through the standard self-attention mechanism.

`HybridDualTower` in `models_hybrid.py` instantiates `UncertainFTTransformerRefined`
internally and shares the same `_FEATURE_GROUPS` — **no changes needed** in
`models_hybrid.py` for the group mapping.

---

## 5. Arch version bumps in `model_adapters.py`

Required because `group_emb` changes shape (8 → 9 groups):

| Adapter | Current | New | Reason |
|---------|---------|-----|--------|
| `FTAdapter` | 3 | 4 | `_FEATURE_GROUPS` has 9 groups → `group_emb` embedding dim changes |
| `HybridAdapter` | 1 | 2 | FT backbone inside `HybridDualTower` has same `group_emb` |
| `FTClassifierAdapter` | 1 | 2 | `FTClassifier` copies same FT backbone |

`MLPAdapter` has no version field — `_MLP(n)` accepts any input width; no bump needed.

Stale checkpoints are detected automatically by `can_load()` → they fall through
to a fresh training run without crashing.

---

## 6. PC1-stratified train/val split (all 7 scripts)

Currently all scripts use unstratified `train_test_split(random_state=42)`.
Replace with quantile-based stratification on PC1:

```python
# After pipeline.transform():
if pipeline.pc1_col_idx is not None:
    _pc1      = X_all[:, pipeline.pc1_col_idx]
    _pc1_bins = pd.qcut(_pc1, q=5, labels=False, duplicates='drop')
    stratify  = _pc1_bins
else:
    stratify = None   # fallback: unstratified (pca_augment not used)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=stratify
)
```

**Why PC1**: On land PC1 captures the brightness/cloud-proximity axis (r=+0.36).
Random splits can under-sample bright-surface soundings in the test set, causing
optimistic evaluation on those scenes. Stratifying ensures each brightness quintile
is proportionally represented.

For the classifiers (`cld_dist_km < 10` target), the existing `stratify=y_all`
is kept alongside the PC1 stratification:
```python
# Classifiers: stratify on both class label AND PC1 bin
stratify = y_all.astype(str) + '_' + _pc1_bins.astype(str)
```

---

## 7. Per-PC-regime evaluation block

Add to the evaluation section of all regression scripts (after computing `y_pred`
on the full test set):

```python
def _pc1_stratified_rmse(y_true, y_pred, pc1_vals, n_bins=5, label='MLP'):
    """Print and return RMSE / MAE per PC1 quintile."""
    bins = pd.qcut(pc1_vals, q=n_bins, labels=False, duplicates='drop')
    rows = []
    for q in range(n_bins):
        mask = bins == q
        if mask.sum() < 10:
            continue
        rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
        mae  = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        rows.append({'model': label, 'pc1_quintile': q + 1,
                     'n': mask.sum(), 'rmse': rmse, 'mae': mae})
        print(f"  {label} PC1-Q{q+1}: RMSE={rmse:.4f}  MAE={mae:.4f}  n={mask.sum():,}")
    return pd.DataFrame(rows)

# After predictions:
if pipeline.pc1_col_idx is not None:
    pc1_test = X_test[:, pipeline.pc1_col_idx]
    strat_df = _pc1_stratified_rmse(y_test, y_pred, pc1_test, label='MLP')
    strat_df.to_csv(output_dir / 'stratified_pc1_rmse.csv', index=False)
```

---

## 8. CLI flags per script

### `mlp_lr_models.py`
```
--scaler        {robust_standard, pca_whitening}  default: robust_standard
--pca-augment   flag: append PC scores to feature vector
```

### `xgb_models.py`
```
--pca-augment   flag: append PC scores (no --scaler; whitening not applicable)
```

### `models_transformer.py`
```
--pca-augment   flag: append PC scores (no --scaler)
```

### `models_hybrid.py`
```
--scaler        {robust_standard, pca_whitening}  (MLP branch benefits)
--pca-augment   flag: append PC scores (FT token stream)
```

### `mlp_lr_cloud_classifier.py`
```
--scaler        {robust_standard, pca_whitening}
--pca-augment   flag
```

### `transformer_cloud_classifier.py`
```
--pca-augment   flag
```

All flags are **opt-in** — default behaviour (no flags) is identical to the current
codebase. This preserves backward compatibility with existing saved pipelines.

---

## 9. File change summary

| File | Type of change |
|------|---------------|
| `src/pipeline.py` | Add `PCAWhitening`, `PCAScoreAppender`; add `scaler` + `pca_augment` to `FeaturePipeline.fit()`; update `transform()` |
| `src/models_transformer.py` | Add `'PCA\nScores'` group to `_FEATURE_GROUPS` (8 → 9 groups) |
| `src/model_adapters.py` | Bump `FTAdapter._ARCH_VERSION` 3→4, `HybridAdapter._ARCH_VERSION` 1→2, `FTClassifierAdapter._ARCH_VERSION` 1→2 |
| `src/mlp_lr_models.py` | Add `--scaler`, `--pca-augment`; PC1-stratified split; PC1-regime eval |
| `src/xgb_models.py` | Add `--pca-augment`; PC1-stratified split; PC1-regime eval |
| `src/models_hybrid.py` | Add `--scaler`, `--pca-augment`; PC1-stratified split; PC1-regime eval |
| `src/mlp_lr_cloud_classifier.py` | Add `--scaler`, `--pca-augment`; PC1-stratified split |
| `src/transformer_cloud_classifier.py` | Add `--pca-augment`; PC1-stratified split |

**No changes needed** in: `model_adapters._MLP`, `models_hybrid.HybridDualTower`,
`apply_models.py`, `apply_model_with_cld.py` (inference scripts pick up the updated
pipeline automatically via `FeaturePipeline.load()`).

---

## 10. Implementation order

Steps 1–3 are hard prerequisites; steps 4–8 are independent of each other.

```
Step 1  pipeline.py          PCAWhitening + PCAScoreAppender + FeaturePipeline params
Step 2  models_transformer.py  Add 'PCA\nScores' to _FEATURE_GROUPS
Step 3  model_adapters.py    Bump arch versions (FTAdapter→4, HybridAdapter→2, FTClassifierAdapter→2)
────────────────────────────────────── prerequisites above ──────────────────────────────────────
Step 4  mlp_lr_models.py     --scaler, --pca-augment, stratified split, PC1-regime eval
Step 5  xgb_models.py        --pca-augment, stratified split, PC1-regime eval
Step 6  models_hybrid.py     --scaler, --pca-augment, stratified split, PC1-regime eval
Step 7  mlp_lr_cloud_classifier.py      --scaler, --pca-augment, stratified split
Step 8  transformer_cloud_classifier.py --pca-augment, stratified split
```

---

## 11. Experiment matrix (recommended runs after implementation)

| Run | sfc\_type | scaler | pca\_augment | Purpose |
|-----|-----------|--------|-------------|---------|
| baseline | 0 / 1 | robust\_standard | false | Reference (current behaviour) |
| whitening | 0 / 1 | pca\_whitening | false | Decorrelated input for MLP/Ridge |
| augment | 0 / 1 | robust\_standard | true | Cloud-correlated PCs as extra tokens |
| both | 0 / 1 | pca\_whitening | true | Combined (MLP only; FT uses augment) |

Compare RMSE, MAE, and `stratified_pc1_rmse.csv` across runs.
Key question: does the model's error on low-PC1 (cloud-adjacent, low-albedo) scenes
decrease when PC scores are explicitly provided?

---

## 12. Notes and constraints

- **Pickle compatibility**: `PCAWhitening` and `PCAScoreAppender` must set
  `__module__ = 'pipeline'` (same pattern as `ClipFreeQuantileTransformer` and
  `RobustStandardScaler`) so that saved pipelines can be loaded on CURC even when
  `pipeline.py` is run as `__main__`.
- **Float32 consistency**: `PCA.transform()` in sklearn returns float64 by default.
  Cast output to float32 before concatenation (consistent with existing pipeline).
- **CURC**: The new `--scaler pca_whitening` runs need more memory for the PCA fit
  on large datasets. `n_components=0.90` (variance fraction) is safer than a fixed
  int on CURC because the effective rank varies by date subset.
- **Inference scripts** (`apply_models.py`, `apply_model_with_cld.py`): load
  `FeaturePipeline` via `.load()` — the updated `transform()` will automatically
  apply the appender if `pipeline.pca_appender` is set. No changes needed.
- **XGBoost + PCAWhitening**: explicitly blocked — if `--scaler pca_whitening` is
  passed to `xgb_models.py`, raise a clear `ValueError` with explanation.
