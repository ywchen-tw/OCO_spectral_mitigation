# OCO-FP Analysis Project Memory

## Project Overview
OCO-2 glint-mode footprint cloud-proximity pipeline. Collocates OCO-2 soundings with Aqua-MODIS cloud masks; calculates nearest-cloud distance (km) per sounding_id. Five phases: Metadata → Ingestion → Processing → Geometry → Synthesis.

**Working dir**: `/Users/yuch8913/programming/oco_fp_analysis`
**Key entry point**: `workspace/demo_combined.py --date YYYY-MM-DD`
**Test date**: 2018-10-18

---

## Architecture

### Source Files
- `src/phase_01_metadata.py` — OCO2MetadataRetriever; CMR + GES DISC XML fetch
- `src/phase_02_ingestion.py` — DataIngestionManager; OCO-2 + MODIS download
- `src/phase_03_processing.py` — SpatialProcessor; footprint + cloud mask extraction
- `src/phase_04_geometry.py` — GeometryProcessor; ECEF, KD-tree, distances
- `src/oco_fp_spec_anal.py` — Spectral analysis / transmittance fitting
- `src/result_ana.py` — k1k2 analysis + MLP vs LinearRegression comparison
- `src/abs_util/fp_abs_coeff.py` — Absorption coefficient calc (Doppler, solar H5)
- `workspace/demo_combined.py` — End-to-end pipeline runner

### Data Layout
```
data/OCO2/{year}/{doy:03d}/{granule_id}/   ← OCO-2 files + sat_data_status.json
data/MODIS/{year}/{doy:03d}/               ← MYD35_L2 + MYD03 HDF files
data/processing/{year}/{doy:03d}/{granule_id}/  ← footprints.pkl, clouds.pkl, phase4_results.pkl
data/processing/{year}/{doy:03d}/lite_sounding_ids.pkl  ← day-level cache
results_{date}.h5 / results_{date}.csv     ← Phase 4/5 output
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
