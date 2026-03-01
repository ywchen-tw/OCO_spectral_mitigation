# OCO-FP Analysis Project Notes

## Project Overview
OCO-2 glint-mode footprint cloud-proximity pipeline. Collocates OCO-2 soundings with Aqua-MODIS cloud masks; calculates nearest-cloud distance (km) per sounding_id. Five phases: Metadata → Ingestion → Processing → Geometry → Synthesis.

**Working dir**: `/Users/yuch8913/programming/oco_fp_analysis`
**Key entry point**: `workspace/demo_combined.py --date YYYY-MM-DD`
**Test date**: 2018-10-18

---

## Architecture

### Source Files — Pipeline (Phases 1–4)
- `src/phase_01_metadata.py` — OCO2MetadataRetriever; CMR + GES DISC XML fetch
- `src/phase_02_ingestion.py` — DataIngestionManager; OCO-2 + MODIS download
- `src/phase_03_processing.py` — SpatialProcessor; footprint + cloud mask extraction
- `src/phase_04_geometry.py` — GeometryProcessor; ECEF, KD-tree, distances
- `src/oco_fp_spec_anal.py` — Spectral analysis / transmittance fitting
- `src/abs_util/fp_abs_coeff.py` — Absorption coefficient calc (Doppler, solar H5)
- `workspace/demo_combined.py` — End-to-end pipeline runner

### Source Files — Bias Correction (ML)
- `src/utils.py` — shared utilities incl. `get_storage_dir()` (single canonical definition)
- `src/pipeline.py` — `FeaturePipeline`: shared QT fitting + fp one-hot; CLI to fit/save
- `src/model_adapters.py` — `ModelAdapter` ABC; `RidgeAdapter`, `MLPAdapter`, `FTAdapter`; `_ResBlock`, `_MLP` at module level
- `src/mlp_lr_models.py` — trains Ridge + residual MLP; uses `FeaturePipeline` + adapters
- `src/models_transformer.py` — trains FT-Transformer (`UncertainFTTransformerRefined`); uses `FeaturePipeline` + `FTAdapter`
- `src/apply_models.py` — inference + cross-model comparison CLI
- `src/result_ana.py` — k1k2 analysis + legacy LR/MLP comparison

### Data Layout
```
data/OCO2/{year}/{doy:03d}/{granule_id}/   ← OCO-2 files + sat_data_status.json
data/MODIS/{year}/{doy:03d}/               ← MYD35_L2 + MYD03 HDF files
data/processing/{year}/{doy:03d}/{granule_id}/  ← footprints.pkl, clouds.pkl, phase4_results.pkl
data/processing/{year}/{doy:03d}/lite_sounding_ids.pkl  ← day-level cache
results_{date}.h5 / results_{date}.csv     ← Phase 4/5 output
results/csv_collection/combined_*.csv      ← training data for ML models
results/model_mlp_lr/<suffix>/             ← Ridge + MLP artifacts
results/model_ft_transformer/<suffix>/     ← FT-Transformer artifacts
```

---

## Key Technical Details

### Temporal Matching
- Use **±20 min** buffer for all years (Aqua free-drift since 2023)
- Phase 2 always downloads with full ±20 min; Phase 3 matching uses adaptive: `year < 2022` → ±10 min, `year ≥ 2022` → ±20 min
- MODIS granule times are **naive UTC** (no tzinfo); ensure `.replace(tzinfo=None)` before comparisons

### OCO-2 L1B Version Switching
- Before 2024-04-01: `OCO2_L1B_Science_11r`
- On/after 2024-04-01: `OCO2_L1B_Science_11.2r`
- Automatic in Phase 1 — no manual config needed

### Cloud Mask Unpacking (MYD35_L2 Byte 1)
- Bits 1-2: `00`=Cloudy, `01`=Uncertain, `10`=Prob. Clear, `11`=Clear
- Extract: `(byte >> 1) & 0b11`
- Only Cloudy + Uncertain pixels are kept for distance calc
- Night passes: check bit 3 of byte 0; skip granules where night > day pixels

### MODISCloudMask Dataclass
```python
MODISCloudMask(granule_id, observation_time,
               lon: np.ndarray, lat: np.ndarray, cloud_flag: np.ndarray)
# cloud_flag: 0=Uncertain, 1=Cloudy (uint8)
```

### Phase 4: Array-Based Processing
Pass numpy arrays directly to `build_kdtree(cloud_lons=, cloud_lats=, cloud_flags=)`.
Legacy object mode still supported for backward compat.

---

## Bias Correction ML Architecture

### Platform-Aware Path Resolution (`src/utils.py`)
`get_storage_dir()` is the **single canonical definition** of the storage root.
All ML scripts import it from here — no local copies.

```python
from utils import get_storage_dir
storage_dir = get_storage_dir()   # Path; macOS→local, Linux→CURC, else→default
```

Every script derives its data/results paths from `storage_dir`:
```
storage_dir / 'results/csv_collection/combined_2020_dates.csv'  ← training CSV
storage_dir / 'results/model_mlp_lr/<suffix>/'                  ← Ridge+MLP artifacts
storage_dir / 'results/model_ft_transformer/<suffix>/'          ← FT artifacts
storage_dir / 'results/comparison/<suffix>/'                    ← comparison plots
```

### FeaturePipeline (`src/pipeline.py`)
Single source of truth for feature selection and QT fitting. Replaces duplicated
`training_data_load_preselect()` / `training_data_load()` that previously lived in
both training files.

```python
pipeline = FeaturePipeline.fit(df, sfc_type=1)   # fit once on training data
pipeline.save(path)

X = pipeline.transform(df)                        # apply to any DataFrame
```

- `sfc_type=0` → ocean feature set; `sfc_type=1` → glint/land feature set
- QT features + `fp_{0..7}` one-hots appended raw = `pipeline.n_features` total
- All path args are **optional** — defaults derived from `get_storage_dir()`:

```bash
# Minimal: all paths resolved from storage_dir
python src/pipeline.py --sfc-type 1 --suffix exp_v1

# Full explicit override
python src/pipeline.py --data /path/to/data.csv --sfc-type 1 --out /path/to/pipeline.pkl
```

### Model Adapters (`src/model_adapters.py`)
Uniform `predict(X)` / `predict_quantiles(X)` / `save(dir)` / `load(dir)` / `can_load(dir)` interface.

| Adapter | Wraps | Artifact files |
|---------|-------|---------------|
| `RidgeAdapter` | sklearn `Ridge` | `ridge_model.pkl` |
| `MLPAdapter` | `_MLP` (residual) | `mlp_weights.pt`, `mlp_meta.pkl` |
| `FTAdapter` | `UncertainFTTransformerRefined` | `model_best.pt`, `ft_meta.pkl` |

`_ResBlock` and `_MLP` are defined at module level here (not nested in training scripts).

### Model Architecture Improvements (2026-03-01)

Four targeted fixes applied to address poor clear-sky R² (≈0.048) and cloud-regime spread:

#### MLP (`src/model_adapters.py`, `src/mlp_lr_models.py`)

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Norm in `_ResBlock` | `BatchNorm1d` | `LayerNorm` | BN uses running stats at eval → misscales OOD cloud-affected samples |
| Norm in `_MLP.input_proj` | `BatchNorm1d` | `LayerNorm` | Same issue at the input projection |
| y-normalization | `mean / std` | `median / (IQR / 1.3490)` | Robust to heavy left tail; `1.3490 = IQR of N(0,1) = 2 × Φ⁻¹(0.75)`, so `IQR/1.3490 ≈ σ` |

#### FT-Transformer (`src/models_transformer.py`)

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Head aggregation | `x.flatten(1)` → `Linear(n_feat×d_token, d_ff)` | `x.mean(dim=1)` → `Linear(d_token, d_token//2)` → `Linear(d_token//2, 3)` | Flatten head had 2.4M params for a single projection; mean-pool drops this to 32K and is more stable |
| FFN width `d_ff` | 256 (= d_token) | 512 (= 2 × d_token) | Standard FT-Transformer recommendation; equal sizes give FFN no capacity over attention residual |
| LR schedule | None (fixed 1e-4) | `CosineAnnealingLR(T_max=n_epochs, eta_min=1e-6)` | Prevents lr from staying high late in training |

Existing checkpoints (`model_best.pt`) are **incompatible** with the new FT architecture — retrain required.

### Training Scripts
- **`mlp_lr_models.py`**: `--pipeline <path>` optional; loads or fits+saves pipeline; trains Ridge + `_MLP`; saves via adapters; paths via `get_storage_dir()`
- **`models_transformer.py`**: `--pipeline <path>` optional; auto-fits if absent; trains FT-Transformer; saves `ft_meta.pkl` via `FTAdapter.save()` (`model_best.pt` written by training loop); paths via `get_storage_dir()`
- Training remains **independent** — Ridge/MLP and FT can be trained on different machines

### Inference CLI (`src/apply_models.py`)
All path args are **optional** — defaults derived from `get_storage_dir()` + `--suffix`.

```bash
# Minimal: all paths resolved from storage_dir + suffix
python src/apply_models.py --suffix exp_v1

# Full explicit override
python src/apply_models.py \
  --suffix     exp_v1 \
  --pipeline   /custom/pipeline.pkl \
  --ridge-dir  /custom/ridge/ \
  --mlp-dir    /custom/mlp/ \
  --ft-dir     /custom/ft/ \
  --input      new_data.csv \
  --output     corrected.csv \
  --plot-dir   results/comparison/
```

Output CSV adds: `ridge_pred`, `mlp_pred`, `ft_q05`, `ft_q50`, `ft_q95`, `ft_uncertainty`
(+ `ridge_anomaly`, `mlp_anomaly`, `ft_anomaly` when orbit metadata columns present)

Comparison plots (require ground-truth `xco2_bc_anomaly` in input CSV):
- `comparison_scatter.png` — 1×N scatter per active model, R²/MAE annotated
- `comparison_hist.png` — overlaid anomaly distributions
- `metrics.csv` — R², MAE, σ, n per model

### Standard Workflow
```bash
# 1. Fit pipeline once (--suffix controls output location)
python src/pipeline.py --sfc-type 1 --suffix exp_v1

# 2a. Train Ridge + MLP  (uses same suffix to find data + pipeline)
python src/mlp_lr_models.py --suffix exp_v1

# 2b. Train FT-Transformer (independently, e.g. on HPC)
python src/models_transformer.py --suffix exp_v1

# 3. Inference + comparison on new/held-out data
python src/apply_models.py --suffix exp_v1
```

---

## Critical Bugs Fixed (see `log/CRITICAL_FIXES.md` for full details)

| # | File | Issue |
|---|---|---|
| 1 | phase_02_ingestion.py | Met/CO2Prior always used first orbit file |
| 2 | phase_02_ingestion.py | skip_existing returned early without verifying files |
| 3 | demo_combined.py | hardcoded `data_dir="./data"` in run_phase_5 |
| 7 | phase_02_ingestion.py | Dual L2 Lite from cross-midnight CMR granule |
| 8 | phase_02_ingestion.py | HTML login page saved as .nc4 (cookie expiry) |
| 9A | phase_02_ingestion.py | Timezone naive/aware mismatch → zero MODIS downloads on GES DISC runs |
| 9D | demo_combined.py | Phase 3 cache not invalidated after Phase 2 re-downloads |
| 9F | demo_combined.py + phase_03 | Night-pass MODIS granules included in cloud collocation |
| 10 | demo_combined.py `run_phase_3` | Cross-date granule: L2 Lite from target date has no IDs for previous-date orbit → empty footprints → footprints.pkl never written → granule never cached |

---

## oco_fp_spec_anal.py Key Points
- `load_shared_data(sat)` reads Lite + cloud-dist HDF5 **once**; builds O(1) dicts
- `process_orbit(sat, orbit_id, shared_data)` — one orbit at a time (old `cal_mca_rad_oco2` was shadowing orbit_id param)
- Cloud-dist path: `f"{sat['result_dir']}/results_{date}.h5"` (was hardcoded 2018-10-18)
- Output: `fitting_details.h5` with 37 dataset keys unchanged

## fp_abs_coeff.py Key Points
- Solar irradiance: use `solar.h5` (not `solar.txt`); computed per-sounding in solar rest frame
- Doppler chain: atmosphere frame → instrument frame → solar rest frame
- Rayleigh cross-section: computed per-sounding using `wloco_atm = wloco * (1 + v_inst/c)`

## result_ana.py Key Points
- MLP architecture: `n_features → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)`
- Runs alongside LinearRegression as baseline; results saved to CSV + plots
- Output: `LR_MLP_correction_lt_xco2_scatter.png`, `LR_MLP_correction_lt_xco2_map.png`

---

## Workspace / HPC
- Local: standard Python env
- CURC (Blanca): see `curc_shell_cld_dist_blanca_general.sh`
- Phase 5 (Synthesis): next phase — not yet implemented
