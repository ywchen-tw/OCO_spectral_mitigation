# MLP Model Integration — `result_ana.py`

**Date**: 2026-02-17
**File modified**: `src/result_ana.py`
**Purpose**: Add a PyTorch MLP as a non-linear alternative to the existing `LinearRegression` baseline, running both models in parallel for comparison.

---

## Overview

Two functions in `result_ana.py` were extended to run a PyTorch MLP (multi-layer perceptron) alongside the existing `sklearn.LinearRegression` baseline.
The linear model is **not replaced** — it serves as a baseline; the MLP results are stored, plotted, and saved separately.

---

## Changes

### 1. Imports (line ~12)

```python
import torch
import torch.nn as nn
```

---

### 2. `plot_comparison()` — regression analysis per footprint

#### New arrays before the loop

```python
R2_scores_mlp    = np.full(len(y_set), np.nan)
slopes_mlp       = np.full(len(y_set), np.nan)
intercepts_mlp   = np.full(len(y_set), np.nan)
```

#### MLP block added after the linear model (inside `for j, y in enumerate(y_set)`)

- Defines an inner `_MLP` class: `n_features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)`
- Standardizes inputs using per-training-set `mean` and `std` (separate from the existing `X / nanmax` normalization)
- Trains with Adam for 1000 steps, MSE loss
- Computes R² manually and stores into `R2_scores_mlp[j]`, `slopes_mlp[j]`, `intercepts_mlp[j]`
- `model_weights` for MLP is set to `np.nan` (MLP has no single `coef_` vector)
- Generates a separate scatter plot saved as:
  `{y_description}_X{i+1}_mlp_{fp}.png`

#### CSV output extended

New columns added to `regression_results_{fp}.csv`:
| Column | Description |
|---|---|
| `R2_score_mlp` | MLP R² on training set |
| `slope_mlp` | slope of actual vs MLP-predicted |
| `intercept_mlp` | intercept of actual vs MLP-predicted |

---

### 3. `mitigation_test()` — per-footprint bias correction

#### New correction arrays before the loop

```python
xco2_bc_pred_anomaly_mlp  = np.full(len(xco2_bc), np.nan)
xco2_raw_pred_anomaly_mlp = np.full(len(xco2_raw), np.nan)
xco2_bc_corrected_mlp     = copy.deepcopy(xco2_bc)
xco2_raw_corrected_mlp    = copy.deepcopy(xco2_raw)
```

#### Inner `zip` extended from 4 to 6 iterables

The per-`(bc, raw)` inner loop now carries the MLP arrays alongside the linear ones:

```python
for (y, xco2,
     xco2_predict_anomaly,     xco2_corrected,
     xco2_predict_anomaly_mlp, xco2_corrected_mlp) in zip(
        [train_xco2_bc_anomaly,    train_xco2_raw_anomaly],
        [xco2_bc,                  xco2_raw],
        [xco2_bc_pred_anomaly,     xco2_raw_pred_anomaly],
        [xco2_bc_corrected,        xco2_raw_corrected],
        [xco2_bc_pred_anomaly_mlp, xco2_raw_pred_anomaly_mlp],
        [xco2_bc_corrected_mlp,    xco2_raw_corrected_mlp]):
```

#### MLP train/predict block (per footprint, per target)

Same architecture as above (`n_features → 64 → 32 → 1`), trained on `X_train`/`y_train` from the same 80/20 split.
Prints R² alongside the linear model's R² for direct console comparison.
Writes predictions into `xco2_predict_anomaly_mlp` and `xco2_corrected_mlp`.

#### Scatter + histogram plot expanded (1×2 → 1×3)

| Panel | Content |
|---|---|
| Left | Original vs LR corrected (blue) |
| Centre | Original vs MLP corrected (green) |
| Right | Histogram: original (blue), LR (orange), MLP (green) |

Output filename: `LR_MLP_correction_lt_xco2_scatter[_reference_...].png`

#### Map expanded (2×3 → 2×4)

| Position | Content |
|---|---|
| Top-left | O2A k1 |
| Top-centre-left | O2A k2 |
| Top-centre-right | LR predicted anomaly |
| Top-right | **MLP predicted anomaly** *(new)* |
| Bottom-left | XCO2 raw |
| Bottom-centre-left | XCO2 bias-corrected |
| Bottom-centre-right | LR corrected XCO2 |
| Bottom-right | **MLP corrected XCO2** *(new)* |

Output filename: `LR_MLP_correction_lt_xco2_map[_reference_...].png`

**Bug fix (incidental)**: the old map used the loop variable `xco2_predict_anomaly` (holding only the last loop iteration's data). It now correctly uses `xco2_bc_pred_anomaly`, the full pre-allocated array.

---

## MLP Architecture

```
Input (n_features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| Hidden layers | 64, 32 | Moderate capacity for ~15 geophysical features |
| Activation | ReLU | Avoids vanishing gradients with normalized inputs |
| Optimizer | Adam, lr=1e-3 | Robust default for small tabular datasets |
| Epochs | 1000 | Sufficient convergence; no early stopping (small N) |
| Input scaling | StandardScaler per training fold | MLP is sensitive to scale even after `X/max` normalization |
