# Original phase prompts + current pipeline state (then vs now)

> **Consolidated 2026-07-09** from the six files of the gitignored `prompts/`
> directory (deleted): `OCO2_data_analyst.md`, `Phase_01_Metadata.md`,
> `Phase_02_Ingestion.md`, `Phase_03_Processing.md`, `Phase_04_Geometry.md`,
> `Phase_05_Integration.md`. These were the Feb-2026 commissioning specs handed
> to the agent that built the collocation pipeline. Part 1 preserves them
> verbatim; Part 2 records the implementation as of 2026-07-09; Part 3 lists
> where reality diverged from the spec.
>
> **Live successors:** `workspace/oco_modis_cloud_distance.py` (runner),
> `src/pipeline/step_01..04 + step_035` (modules), `log/NOTES.md`
> (architecture reference), `README.md` (quickstart).

---

# Part 1 — Original prompts (verbatim, Feb 2026)

## Role: Principal Scientific Data Analyst and Programmer (`OCO2_data_analyst.md`)

### Domain Context
Role: You are an expert Atmospheric Data Science Agent specialized in satellite collocation and radiative transfer.

Objective: Execute a phased workflow to collocate OCO-2 glint-mode footprints with Aqua-MODIS cloud masks and calculate the distance to the nearest cloud pixel for a user-specified date.

### Technical Context
- Aqua is in a "free-drift" phase (2023+), meaning the temporal gap with OCO-2 has increased. You must use a $\pm 20$-minute temporal window for granule matching.
- MODIS MYD35_L2 geolocation is insufficient at 5 km; you must ingest MYD03 for 1 km resolution. Use ECEF coordinate transformations to avoid polar distortions during distance calculation.
- Implement KD-Trees for efficient $O(N \log M)$ nearest-neighbor searches.

### Workflow Instructions
- Never write or delete files outside this parent folder
- Never output the results to /tmp folder
- Put the test, debug, check codes and files under tests/ folder
- Keep all markdown file in log/ folder
- Initialize: Retrieve the OCO-2 L1B XML for the provided date.
- Ingest: Download OCO-2 L1B, L2 Lite, L2 Met, and L2 CO2Prior, along with MODIS MYD35_L2 and MYD03.
- Process: Unpack the MODIS 48-bit mask (Byte 1, bits 1-2) to identify cloud targets.
- Compute: Convert all coordinates to ECEF and query the KD-Tree for the nearest cloud pixel for each OCO-2 sounding ID.
- Synthesize: Merge results into a single dataset containing sounding_id, nearest_cloud_dist_km, and the temporal difference between the OCO-2 and MODIS observations.

## Phase 1: Metadata Acquisition and Temporal Filtering
The objective is to establish the temporal and orbital boundaries for a specific observation date.

1. XML Retrieval: Programmatically fetch the OCO-2 L1B Science XML for the target date from the GES DISC archive to identify available granules.
2. Orbit Analysis: Extract the specific orbit number, viewing mode (Glint "GL" or Nadir "ND"), and software version (e.g., B11).
3. Temporal Windowing: Identify the exact start and end timestamps for the OCO-2 orbit to define the initial search window for MODIS data.

## Phase 2: Targeted Data Ingestion
Download all required hyperspectral and imagery products for the identified temporal window.

1. OCO-2 Product Suite: Acquire HDF5 files for L1B Science (radiances), L2 Lite (bias-corrected $XCO_2$), L2 Met (meteorology), and L2 CO2Prior (a priori profiles). MODIS Product Suite: Retrieve Aqua MODIS files for the identified window.
2. MYD35_L2: The 48-bit cloud mask. MYD03: 1 km Geolocation Fields. This is required because MYD35_L2 only provides geolocation at 5 km resolution; MYD03 provides the high-resolution 1 km pixel centers needed for precision distance calculation.

## Phase 3: Spatial and Bitmask Processing
Prepare the datasets for geometric collocation by extracting coordinates and identifying cloud targets.

1. OCO-2 Footprint Extraction: Parse the OCO2 Lite nc file to extract footprint_latitude, footprint_longitude, and sounding_time, indexed by sounding_id.
2. Drift-Corrected Alignment: Match OCO-2 footprint geolocation to MODIS 5-minute granules. Use an expanded search window of $\pm 20$ minutes to account for the orbital drift of Aqua in 2023 and beyond.
3. Cloud Mask Unpacking: Extract the Cloud_Mask SDS from MYD35_L2. Unpack Byte 1 and isolate bits 1 and 2 to identify "Cloudy" (00) and "Uncertain" (01) pixels.

## Phase 4: High-Performance Computational Geometry
Perform the nearest-neighbor search using optimized spatial data structures.

1. ECEF Transformation: Convert geodetic coordinates (latitude, longitude, altitude) for both OCO-2 footprints and MODIS cloud pixels into Earth-Centered Earth-Fixed (ECEF) meters using the WGS84 ellipsoid.
2. KD-Tree Construction: Build a spatial KD-Tree using the ECEF coordinates of all MODIS pixels flagged as cloudy or uncertain.
3. Distance Querying: For every OCO-2 footprint center, query the KD-Tree for the $k=1$ nearest neighbor to determine the Euclidean distance ($d$) in 3D space.

## Phase 5: Data Export
Consolidate results into a research-ready format.

1. Dataset Merging: Produce a final HDF5 or CSV file. Map each sounding_id to its calculated nearest_cloud_dist_km.

---

# Part 2 — Current implementation (as of 2026-07-09)

## Runner

`workspace/oco_modis_cloud_distance.py` (renamed from `demo_combined.py`,
commit `f6d7bf4`; helpers split into `workspace/pipeline_phases.py` — the five
`run_phase_1..5` runners — and `workspace/pipeline_utils.py` — Lite-version
handling, cache invalidation, MODIS cleanup, orbit parsing, banners).

```
python workspace/oco_modis_cloud_distance.py --date YYYY-MM-DD
```

Key flags: `--max-distance` (50 km), `--band-width` 2.5° / `--band-overlap`
0.5°, `--skip-phase N`, `--force-recompute`, `--force-download`,
`--force-recompute-if-lite-before VERSION` (invalidates Lite-derived caches when
the local L2 Lite file predates a given version), `--orbit` / `--mode {GL,ND,TG}`
/ `--limit-granules`, `--delete-modis`, `--gcp-project [ID]` (+
`--embedding-batch`, `--embedding-limit-orbits`) for the opt-in Phase 3.5.

Live flow: **Phase 1 is commented out in the orchestrator** (metadata is
resolved inside Phase 2's download path) → Phase 2 → Phase 3 → Phase 3.5
(only with `--gcp-project`; non-fatal on failure) → Phase 4 → Phase 5.
Per-orbit cache under `data/processing/{year}/{doy:03d}/{orbit_id}/`
(`footprints.pkl`, `clouds.pkl`, `granule_combined_*.pkl`,
`phase4_results.pkl`); post-2022 orbits with no MODIS overlap get a `-999`
placeholder written directly to `phase4_results.pkl`.

## Step modules (`src/pipeline/`)

- **`step_01_metadata.py`** — `OCO2MetadataRetriever`: GES DISC XML listing with
  CMR fallback, retry/backoff, login-page detection. **L1B version switching:**
  `VERSION_CHANGE_DATE = 2024-04-01` → `11r` before, `11.2r` on/after (matching
  CMR concept-ids). Parses orbit, GL/ND/TG mode, temporal window.
- **`step_02_ingestion.py`** — `DataIngestionManager.download_all_for_date`:
  OCO-2 L1B / L2 Lite / L2 Met / L2 CO2Prior + MODIS MYD35_L2 + MYD03.
  L2 Lite has a different archive layout (`/YYYY/` not `/YYYY/DOY/`). Streaming
  downloads with resume, 429/5xx backoff, cookie refresh on 401, and
  `_is_readable_hdf5/_is_readable_hdf4` integrity gates.
- **`step_03_processing.py`** — `SpatialProcessor`: footprint extraction from
  Lite, MYD35 unpacking `(byte0 >> 1) & 0b11` (00=Cloudy, 01=Uncertain,
  10=Prob. Clear, 11=Clear; only Cloudy+Uncertain kept, stored float32 lon/lat +
  uint8 flag 0=Uncertain/1=Cloudy). Night passes rejected via bit 3 of byte 0
  (majority vote; re-checked in `match_temporal_windows`). **Drift-aware
  matching buffer** via `constants.modis_match_buffer_minutes(year)`: ±10 min
  before 2022, ±20 min from 2022 (Aqua free drift).
- **`step_035_embedding.py`** — opt-in Phase 3.5: Google Satellite Embedding V1
  Annual (64-D, 10 m) mean+stdDev per footprint polygon via GEE; writes
  `embedding_stats_{date}.parquet` (sync `getInfo` or Drive batch export). See
  `SATELLITE_EMBEDDING_PLAN.md` — model-side consumption still pending.
- **`step_04_geometry.py`** — `GeometryProcessor`: WGS84 geodetic→ECEF,
  `cKDTree` over cloud pixels, **banded sweep** (2.5° lat bands, ±0.5° overlap)
  per granule for memory; also a `1/d²`-weighted cloud distance capped at
  `max_distance_km`. `-999` sentinels excluded from statistics.
- **Phase 5** — no module; `pipeline_phases.run_phase_5` writes
  `results_{date}.h5` (per-sounding datasets incl. `nearest_cloud_distance_km`,
  `weighted_cloud_distance_km`, cloud lat/lon/class) + `results_{date}.json`
  (statistics). A `export_results_csv` method exists but is not called.

---

# Part 3 — Where reality diverged from the spec

| Spec (Feb 2026) | As implemented (Jul 2026) |
|---|---|
| Fixed ±20 min matching window | Drift-aware: ±10 min pre-2022, ±20 min from 2022; Phase 2 always downloads the full ±20 min |
| "Unpack Byte 1, bits 1-2" | Byte **0** of the 6-byte mask, bits 1–2 (the spec's "Byte 1" was 1-indexed) + bit 3 day/night gate the spec never mentioned |
| Five phases, all run | Phase 1 commented out of the orchestrator; Phase 3.5 (GEE embeddings) added, opt-in |
| Single L1B version | Automatic 11r → 11.2r switch at 2024-04-01 |
| Plain KD-tree over all pixels | Per-granule KD-trees + banded latitude sweep + weighted-distance variant |
| "HDF5 or CSV" output | HDF5 + JSON stats (`results_{date}.h5` / `results_{date}.json`); CSV path exists but unused |
| sounding_id → distance only | Output also carries nearest-cloud lat/lon/classification, weighted distance, viewing mode |
| (not in spec) | Caching/skip logic, cache invalidation on Lite-version change, `-999` no-collocation sentinels, night-pass rejection, download integrity gates — see `log/CRITICAL_FIXES.md` Fixes 1–10 for the bugs that forced them |

The original spec's markdown-in-`log/`, tests-in-`tests/`, and
never-write-outside-parent rules are still the house conventions.
