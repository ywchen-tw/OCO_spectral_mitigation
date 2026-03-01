# Refactoring Tracker: Shared Pipeline + Model Adapters + Inference CLI

**Status**: ✅ Complete — all steps done
**Last updated**: 2026-02-27

### Current state
- `src/pipeline.py` ✅ written and complete
- `src/model_adapters.py` ✅ written and complete (`_ResBlock`, `_MLP` at module level; `RidgeAdapter`, `MLPAdapter`, `FTAdapter` all implemented)
- `src/apply_models.py` ✅ written and complete
- `src/mlp_lr_models.py` ✅ **fully modified** — all A1–A8 complete
- `src/models_transformer.py` 🔄 **in progress** — B1–B3e pending

### Next action
All steps complete. Refactoring done.

---

## Goal

Both `mlp_lr_models.py` and `models_transformer.py` have identical feature lists, QT-fitting code, and fp_ one-hot logic. Artifact formats are incompatible, making unified new-data inference and cross-model comparison impossible.

**Solution**: Extract the shared pipeline, wrap models in adapters, add `apply_models.py` CLI.
**Constraint**: Training remains independent — Ridge+MLP via `mlp_lr_models.py`, FT via `models_transformer.py`.

---

## New Files

| File | Status | Purpose |
|------|--------|---------|
| `src/pipeline.py` | ✅ Done | `FeaturePipeline` class — fit/transform/save/load + CLI |
| `src/model_adapters.py` | ✅ Done | `ModelAdapter` ABC + `RidgeAdapter`, `MLPAdapter`, `FTAdapter` |
| `src/apply_models.py` | ✅ Done | Inference + comparison CLI |

## Modified Files

| File | Status | Changes |
|------|--------|---------|
| `src/mlp_lr_models.py` | ✅ Done | Remove `training_data_load_preselect()`; extract `_MLP`/`_ResBlock`; use adapters |
| `src/models_transformer.py` | ✅ Done | Remove `training_data_load()`; use `FTAdapter`; add `--pipeline` arg |

---

## Detailed To-Do Checklist

### Step 1 — `src/pipeline.py`
- [x] `FeaturePipeline` class with fields: `sfc_type`, `qt`, `qt_features`, `fp_cols`, `features`
- [x] `fit(cls, df, sfc_type=1)` — copies feature lists from existing files (identical in both), fits QT, creates fp_{i} columns from `df['fp']` if not present
- [x] `transform(self, df) -> np.ndarray` — creates fp_{i} if needed, applies QT, appends raw fp one-hots; returns `X [N, n_features]`
- [x] `save(path)` / `load(path)` via pickle
- [x] `@property n_features` and `@property feature_names` (alias for `self.features`)
- [x] `__main__` CLI: `python src/pipeline.py --data <csv> --sfc-type 1 --out <pipeline.pkl>`
  - Loads CSV, filters `sfc_type` + `snow_flag==0`, calls `fit()`, saves

### Step 2 — `src/model_adapters.py`
- [x] `ModelAdapter(ABC)`: `predict(X)`, `predict_quantiles(X)` (default: NotImplementedError), `save(dir)`, `load(dir)`
- [x] **Extract** `_ResBlock` and `_MLP` from `mitigation_test()` in `mlp_lr_models.py:394–429` to module-level here (architecture unchanged)
- [x] `RidgeAdapter`: wraps sklearn `Ridge`; `FILENAME='ridge_model.pkl'`; pickles whole adapter
- [x] `MLPAdapter`: wraps `_MLP` + `y_mean`/`y_std`/`device`; batched predict with denorm; `WEIGHTS_FILE='mlp_weights.pt'`, `META_FILE='mlp_meta.pkl'`
- [x] `FTAdapter`: wraps `UncertainFTTransformerRefined` (imported from `models_transformer`); `predict()` → q50; `predict_quantiles()` → (q05, q50, q95); `CHECKPOINT_FILE='model_best.pt'` (format unchanged); `META_FILE='ft_meta.pkl'`; `can_load(dir)` classmethod
- [x] All adapters: `save()` and `load()` methods

### Step 3 — `src/apply_models.py`
- [x] CLI args: `--pipeline` (required), `--ridge-dir`, `--mlp-dir`, `--ft-dir` (optional), `--input` (required), `--output` (required), `--plot-dir` (optional), `--sfc-type` (default 1)
- [x] Load pipeline → `FeaturePipeline.load()`
- [x] Load available adapters (skip silently if dir not supplied or files missing)
- [x] Filter CSV by `sfc_type` + `snow_flag==0`, apply `pipeline.transform(df)`
- [x] Predict with each adapter; add columns: `ridge_pred`, `mlp_pred`, `ft_q05`, `ft_q50`, `ft_q95`, `ft_uncertainty`
- [x] Anomaly re-computation (when `date`, `orbit_id`, `lat`, `cld_dist_km` present):
  - `from mlp_lr_models import compute_xco2_anomaly_date_id`
  - `xco2_bc_corrected = df['xco2_bc'] - pred` per model
  - Add `ridge_anomaly`, `mlp_anomaly`, `ft_anomaly` columns
  - Params: `lat_thres=0.25, std_thres=1.0, min_cld_dist=10.0`
- [x] Save output CSV
- [x] Comparison plots when `--plot-dir` + `xco2_bc_anomaly` in CSV:
  - `comparison_scatter.png` — 1×N scatter per model
  - `comparison_hist.png` — overlaid anomaly distributions
  - `metrics.csv` — R², MAE, σ per model

### Step 4 — `src/mlp_lr_models.py`
- [x] A1: Remove `from sklearn.preprocessing import QuantileTransformer`; add `from pipeline import FeaturePipeline` and `from model_adapters import _ResBlock, _MLP, RidgeAdapter, MLPAdapter`
- [x] A2: Delete `training_data_load_preselect()` (lines 26–133)
- [x] A3: Add `pipeline: FeaturePipeline` parameter to `mitigation_test()`
- [x] A4: Replace `training_data_load_preselect(df)` call with `pipeline.transform(df)` + valid-row masking + `train_test_split`
- [x] A5: Remove `_transform(df)` helper inside `mitigation_test()`; use `pipeline.transform()` instead
- [x] A6: Delete nested `_ResBlock` + `_MLP` from `mitigation_test()`
- [x] A7: Replace artifact save/load block with adapter-based save/load (`RidgeAdapter.can_load` / `MLPAdapter.can_load`; `RidgeAdapter(model).save()` / `MLPAdapter(mlp, y_mean, y_std).save()`)
- [x] A8: `main()`: add `--pipeline` arg; load or fit+save pipeline; remove manual fp_dummies; pass `pipeline` to `mitigation_test()`
- [x] Keep `compute_xco2_anomaly_date_id` unchanged

### Step 5 — `src/models_transformer.py`
- [x] B1: Add `from pipeline import FeaturePipeline` and `from model_adapters import FTAdapter`
- [x] B2: Delete `training_data_load()` (lines 614–734)
- [x] B3a: `main()`: add `--pipeline` arg; load or fit+save pipeline
- [x] B3b: Replace inline data loading with `pipeline.transform(df)` + valid-row masking + `train_test_split`
- [x] B3c: Replace checkpoint load: `if FTAdapter.can_load(output_dir): adapter = FTAdapter.load(output_dir); model = adapter.model`
- [x] B3d: Replace checkpoint save (after training): `FTAdapter(model, n_features=...).save(output_dir)` (saves `ft_meta.pkl` only; `model_best.pt` already written by training loop)
- [x] B3e: Remove `pickle.dump/load(qt, ...)` blocks (QT lives in pipeline now)

---

## Key Architecture Decisions

| Decision | Choice |
|----------|--------|
| Pipeline format break | **Clean break** — no backward compat. Existing dirs need one re-run of `pipeline.py`. |
| `model_best.pt` format | **Unchanged** — `FTAdapter.load()` reads it directly |
| Anomaly re-computation in apply | **Yes** — when `date/orbit_id/lat/cld_dist_km` columns present |
| `_MLP`/`_ResBlock` location | Moved to **`model_adapters.py`** at module level; imported back into `mlp_lr_models.py` |
| `compute_xco2_anomaly_date_id` | Stays in `mlp_lr_models.py`; imported by `models_transformer.py` (unchanged) and new `apply_models.py` |

---

## Post-Refactoring Workflow

```bash
# 1. Fit pipeline once
python src/pipeline.py --data results/csv_collection/combined_2017_2021_dates.csv --sfc-type 0 --out results/train_data/exp_v1/pipeline_ocean_2017_2020.pkl

# 2a. Train Ridge + MLP
python src/mlp_lr_models.py --pipeline results/train_data/exp_v1/pipeline_202001.pkl --suffix exp_v1

# 2b. Train FT-Transformer (independently, e.g. on HPC)
python src/models_transformer.py --pipeline results/train_data/exp_v1/pipeline_202001.pkl --suffix exp_v1

# 3. Inference + comparison on new data
python src/apply_models.py \
  --pipeline  results/train_data/exp_v1/pipeline_202001.pkl \
  --ridge-dir results/model_mlp_lr/exp_v1/ \
  --mlp-dir   results/model_mlp_lr/exp_v1/ \
  --ft-dir    results/model_ft_transformer/exp_v1/ \
  --input     results/csv_collection/combined_2019-01-01_all_orbits.csv  \
  --output    corrected.csv \
  --plot-dir  results/exp_v1/comparison/
```

---

## Key File Line References (for implementing agents)

| Location | Lines (original) | Action | Done? |
|----------|-----------------|--------|-------|
| `mlp_lr_models.py` | 26–133 | DELETE — `training_data_load_preselect()` | ✅ |
| `mlp_lr_models.py` | 394–429 | MOVE to `model_adapters.py` — `_ResBlock`, `_MLP` | ✅ |
| `mlp_lr_models.py` | 439–570 | REPLACE — artifact save/load block | ✅ |
| `mlp_lr_models.py` | 186–270 | KEEP — `compute_xco2_anomaly_date_id()` | ✅ |
| `models_transformer.py` | 614–734 | DELETE — `training_data_load()` | ✅ |
| `models_transformer.py` | 1364–1393 | REPLACE — checkpoint load/save in `main()` | ✅ |
| `models_transformer.py` | 163–241 | KEEP — `UncertainFTTransformerRefined` (imported by `FTAdapter`) | ✅ |
