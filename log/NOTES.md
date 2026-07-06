# OCO-FP Analysis Project Notes

*Updated 2026-07-06: paths refreshed to the current `src/` package layout; the
Ridge/MLP/FT-Transformer section (removed 2026-07-03, see PIPELINE_CHANGELOG.md)
replaced by the current deep-ensemble stack. The old FT-Transformer notes live in
`log/archive/FT_TRANSFORMER_HISTORICAL.md`.*

## Project Overview
OCO-2 glint-mode footprint cloud-proximity pipeline + XCO2 bias-correction stack.
Collocates OCO-2 soundings with Aqua-MODIS cloud masks; calculates nearest-cloud
distance (km) per sounding_id; fits photon path-length cumulants from the L1B
spectra; trains per-surface deep ensembles to correct the near-cloud XCO2 anomaly.

**Working dir**: `/Users/yuch8913/programming/oco_fp_analysis`
**Key entry point**: `workspace/demo_combined.py --date YYYY-MM-DD`
**Test date**: 2018-10-18

---

## Architecture

### Source Files — Data Pipeline (Phases 1–4)
- `src/pipeline/phase_01_metadata.py` — OCO2MetadataRetriever; CMR + GES DISC XML fetch
- `src/pipeline/phase_02_ingestion.py` — DataIngestionManager; OCO-2 + MODIS download
- `src/pipeline/phase_03_processing.py` — SpatialProcessor; footprint + cloud mask extraction
- `src/pipeline/phase_035_embedding.py` — opt-in GEE satellite-embedding extraction
- `src/pipeline/phase_04_geometry.py` — GeometryProcessor; ECEF, KD-tree, distances
- `src/abs_util/fp_abs_coeff.py` — Absorption coefficient calc (Doppler, solar H5)
- `workspace/demo_combined.py` — End-to-end runner (helpers in `demo_utils.py`,
  phase runners in `pipeline_phases.py`)
- `src/constants.py` — single source of pipeline numbers (buffer year, anomaly
  params, band widths, `FIT_ORDER`)

### Source Files — Spectral fitting + features
- `src/spectral/` — cumulant fit (`cumulant_fit.py` exact lstsq/BVLS solver,
  `orbit_data.py` loading, `anomaly.py` target definition, `fitting.py`
  facade/CLI). Physics: `FITTING_DERIVATION.md`; order choice: `FIT_ORDER_EXPERIMENT.md`.
  Output: per-date `fitting_details.h5` (float32+gzip).
- `src/analysis/build_feature_dataset.py` — per-date feature parquet
  (`results/csv_collection/combined_<date>_all_orbits.parquet`) incl. AK columns
  and sigma-grid profile columns; `src/analysis/run_all.py` — science figures.

### Source Files — Bias Correction (ML, current)
- `src/models/pipeline.py` — `FeaturePipeline`; feature sets `full / no_xco2 /
  no_spec / no_xco2_and_spec / no_contam / no_contam_and_xco2`; no-SG k-features
  default (`_USE_NOSG_K`)
- `src/models/deep_ensemble.py` — **production**: per-surface `GaussianMLP`
  64→32→(mu, log_var), M=5, beta-NLL β=1.0, `--norm layer --dropout 0.1`,
  `--profile-pca`, split + cloud-Mondrian conformal (`--near_cloud_target 0.98`);
  tags `de_{ocean,land}_beta_nll_prof_reg_{r05,r15}`; docs in
  `deep_ensemble_ARCHITECTURE.md`, plan status in `FINE_TUNE_PLAN.md`
- `src/models/train_common.py` — shared trainer/`TrainConfig` (all torch literals)
- `src/models/tabm.py`, `xgb.py`, `gbdt_baselines.py`, `mlp_baseline.py` — comparison models
- `src/models/profile_pca.py` — per-surface EOF compression of T/q/CO2 profiles
- `src/apply/apply_deep_ensemble.py` — inference bridge
- `workspace/build_deepens_plot_data.py` → `tccon_collocate.py` → `ak_harmonize.py`
  → `tccon_comparison_report.py` — TCCON validation chain; ocean anchors in
  `workspace/Ship_analysis/` + `workspace/ATom_analysis/`

### Data Layout
```
data/OCO2/{year}/{doy:03d}/{granule_id}/   ← OCO-2 files + sat_data_status.json
data/MODIS/{year}/{doy:03d}/               ← MYD35_L2 + MYD03 HDF files
data/processing/{year}/{doy:03d}/{granule_id}/  ← footprints.pkl, clouds.pkl, phase4_results.pkl
data/processing/{year}/{doy:03d}/lite_sounding_ids.pkl  ← day-level cache
results_{date}.h5                          ← Phase 4/5 output
results/csv_collection/combined_*_all_orbits.parquet  ← per-date feature data
results/csv_collection/combined_2016_2020_dates.parquet ← combined training data (117.7M rows)
results/model_comparison/                  ← experiments; see EXPERIMENTS.md index
results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_o05l15_m5/ ← production TCCON/atom/ship trees
```

---

## Key Technical Details

### Temporal Matching
- Phase 2 always downloads with full ±20 min; Phase 3 matching is adaptive:
  `year < constants.AQUA_FREE_DRIFT_YEAR (=2022)` → ±10 min, else ±20 min
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

### Platform-Aware Path Resolution (`src/utils.py`)
`get_storage_dir()` is the **single canonical definition** of the storage root
(macOS→local, Linux→CURC). All ML scripts derive data/results paths from it.

### ML target + validation conventions
- Target: `xco2_bc_anomaly` vs same-orbit clear-sky neighbors
  (`constants.ANOMALY_*` = 0.25° / 1.0 ppm / 10 km); per-surface radius variants
  r05 (ocean production) / r15 (land production)
- Honest validation split: `date_kfold` (random split leaks ~0.29 R²)
- Deployment constraint: correction is per-footprint — no cloud info, no
  neighboring-footprint info at inference

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

(2026-07-04 hardening: atomic `.part` downloads + size verification + Range
resume + MODIS HDF4 readability probe — see PROJECT_REVIEW §7.1.)

---

## fp_abs_coeff.py Key Points
- Solar irradiance: use `solar.h5` (not `solar.txt`); computed per-sounding in solar rest frame
- Doppler chain: atmosphere frame → instrument frame → solar rest frame
- Rayleigh cross-section: computed per-sounding using `wloco_atm = wloco * (1 + v_inst/c)`

---

## Workspace / HPC
- Local: standard Python env
- CURC (Blanca): pipeline `curc_shell_cld_dist_blanca_general.sh`; DE training
  `curc_shell_blanca_de_profile*.sh`; TCCON validation
  `curc_shell_blanca_plot_corr_xco2_deepens.sh`; uncertainty regen
  `curc_shell_blanca_deepens_uncertainty.sh`
