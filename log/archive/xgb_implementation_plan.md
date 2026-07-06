# XGBoost Model Integration Plan

**Date**: 2026-03-03
**Status**: ✅ Complete

## Progress

| File | Action | Status |
|------|--------|--------|
| `src/xgb_models.py` | CREATE | ✅ Done |
| `src/model_adapters.py` | MODIFY — add `XGBoostAdapter` | ✅ Done |
| `src/apply_models.py` | MODIFY — `--xgb-dir` + predictions + plots | ✅ Done |
| `curc_shell_blanca_train_compare.sh` | MODIFY — XGB training + `--xgb-dir` | ✅ Done |

---

## Objective

Add XGBoost as a fourth bias-correction model alongside Ridge, MLP, and FT-Transformer.
Uses the same `FeaturePipeline`, `ModelAdapter` interface, and `apply_models.py` CLI.
Primary feature importance method: **SHAP** (TreeExplainer).

---

## Files to Create

### `src/xgb_models.py` — Standalone training script

CLI:
```
python src/xgb_models.py [--sfc_type 0|1] [--suffix <str>] [--pipeline <path>]
```

Training flow mirrors `mlp_lr_models.py`:
1. Load parquet (Linux: `combined_2016_2020_dates.parquet`, Darwin: single-date)
2. Filter `sfc_type` + `snow_flag == 0`
3. Load or fit `FeaturePipeline` (same `--pipeline` arg pattern)
4. `pipeline.transform(df)` → `X [N, n_features]`
5. `valid_rows = ~df['xco2_bc_anomaly'].isna()`
6. `train_test_split(X, y, test_size=0.2, random_state=42)`
7. Fit `XGBRegressor` with early stopping
8. Print R² on test set
9. Run importance plots (builtin + SHAP)
10. `XGBoostAdapter(model, feature_names=features).save(output_dir)`

**Hyperparameters:**
```python
xgb.XGBRegressor(
    n_estimators        = 2000,
    max_depth           = 6,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 5,
    gamma               = 0.1,
    reg_lambda          = 1.0,
    objective           = 'reg:squarederror',
    tree_method         = 'hist',        # fast on large N
    device              = 'cuda' | 'cpu',
    early_stopping_rounds = 50,
    eval_metric         = 'rmse',
    n_jobs              = -1,
)
```

**`plot_xgb_importance_builtin(model, features, output_dir)`:**
- 3-panel: gain / cover / weight, horizontal bar, all features sorted by gain
- Output: `xgb_importance_builtin.png`

**`plot_xgb_shap(model, X_test, features, output_dir)`** (primary importance):
```python
import shap
explainer   = shap.TreeExplainer(model)      # native XGBoost path, O(TLD)
shap_values = explainer.shap_values(X_test)  # [N, n_features], signed
```

Outputs:
| File | Content |
|------|---------|
| `xgb_shap_bar.png` | `mean(\|SHAP\|)` per feature, horizontal bar, all features |
| `xgb_shap_importance.csv` | `feature, mean_abs_shap, std_abs_shap` (comparable to `feature_importance_xco2_bc_anomaly.csv` from MLP) |
| `xgb_shap_beeswarm.png` | `shap.summary_plot(..., plot_type='dot')`, top 30 features, shows direction |
| `xgb_shap_dependence_<feat>.png` | `shap.dependence_plot(feat, shap_values, X_test, interaction_index='auto')` for top 5 features by mean\|SHAP\| |
| `xgb_shap_heatmap.png` | `shap.plots.heatmap(explanation[:500])`, reveals cloud-regime patterns |

Subsampling: beeswarm/heatmap use `X_shap = X_test[:5000]`; CSV/bar use full test set.

**`xgb_learning_curve.png`**: train vs eval RMSE from `model.evals_result_` vs boosting round.

**Output dir**: `<storage_dir>/results/model_xgb/<suffix>/`

---

## Files to Modify

### `src/model_adapters.py` — Add `XGBoostAdapter`

Add after `MLPAdapter`, before `FTAdapter`:

```python
class XGBoostAdapter(ModelAdapter):
    WEIGHTS_FILE = 'xgb_model.ubj'   # XGBoost native binary
    META_FILE    = 'xgb_meta.pkl'    # n_features + feature_names

    def __init__(self, model, feature_names=None):
        self.model         = model
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)   # [N] float32

    def save(self, output_dir) -> None: ...   # save .ubj + pickle meta
    def load(cls, output_dir) -> 'XGBoostAdapter': ...
    def can_load(cls, output_dir) -> bool: ...
```

### `src/apply_models.py` — Add `--xgb-dir` + predictions + plots

- Add `--xgb-dir` CLI argument (resolved via `_abs()` helper)
- Load `XGBoostAdapter` via `_try_load(XGBoostAdapter, xgb_dir, 'XGBoost')`
- In `_process_one_file()`: add `xgb_pred`, `xgb_corrected_xco2`, `xgb_anomaly` columns
- In all comparison plots: add `'XGBoost': ('xgb_pred', 'crimson')` to `pred_cols`
- Include XGBoost in `metrics.csv` (R², MAE, σ, n)
- Include in `plot_data<suffix>.parquet`

### `curc_shell_blanca_train_compare.sh`

Add XGBoost training block after FT-Transformer:
```bash
python src/xgb_models.py \
  --sfc_type 1 \
  --suffix land_2016_2020 \
  --pipeline results/train_data/pipeline_land_2016_2020.pkl
```

Add `--xgb-dir results/model_xgb/land_2016_2020/` to existing `apply_models.py` call.

---

## File Change Summary

| File | Action | Key Change |
|------|--------|-----------|
| `src/xgb_models.py` | **CREATE** | Training + built-in importance + SHAP (bar, beeswarm, dependence, heatmap) + learning curve |
| `src/model_adapters.py` | **MODIFY** | Add `XGBoostAdapter` (`.ubj` weights + `.pkl` meta) |
| `src/apply_models.py` | **MODIFY** | `--xgb-dir` + predictions + comparison plots + metrics.csv |
| `curc_shell_blanca_train_compare.sh` | **MODIFY** | XGB training step + `--xgb-dir` in apply call |

---

## Verification

```bash
# 1. Train (local smoke test)
python src/xgb_models.py --sfc_type 0
# Expect: xgb_model.ubj, xgb_meta.pkl, xgb_importance_builtin.png,
#         xgb_shap_bar.png, xgb_shap_beeswarm.png,
#         xgb_shap_dependence_*.png, xgb_shap_heatmap.png,
#         xgb_shap_importance.csv, xgb_learning_curve.png

# 2. Apply
python src/apply_models.py \
  --pipeline results/model_mlp_lr/pipeline.pkl \
  --ridge-dir results/model_mlp_lr/ --mlp-dir results/model_mlp_lr/ \
  --ft-dir results/model_ft_transformer/ --xgb-dir results/model_xgb/ \
  --input combined_2020-01-01_all_orbits.parquet \
  --plot-dir results/model_comparison/
# Expect: corrected.parquet has xgb_pred, xgb_corrected_xco2, xgb_anomaly
#         comparison_scatter.png has 4 panels (ridge, mlp, ft, xgb)

# 3. Importance sanity check
# xgb_shap_importance.csv should rank o2a_k1, o2a_k2 near top
# Dependence plots should show cld_dist_km as auto-selected interaction feature
```
