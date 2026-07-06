# Satellite Embedding Feature Integration Plan

**Goal**: Enrich the XCO2 bias-correction models with high-resolution (10 m) surface spectral
albedo and sub-pixel heterogeneity features derived from the
**Google Satellite Embedding V1 Annual** dataset
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` on Google Earth Engine).

---

## Motivation

### Why this dataset

The OCO-2 retrieved albedos (`alb_o2a`, `alb_wco2`, `alb_sco2`) are integrated over the
~1.3 × 2.25 km footprint and are contaminated by the atmospheric state at measurement time
(cloud scattering, aerosol).  The annual Google/DeepMind satellite embedding provides:

1. **Clean surface spectral albedo** at 10 m — derived from cloud-screened Sentinel-2/Landsat
   composites, independent of measurement-time atmosphere
2. **Sub-pixel surface heterogeneity** — variance of the 10 m embeddings within the OCO-2
   footprint boundary quantifies land-cover patchiness / coastal mixing that the retrieved
   albedo collapses to a single value
3. **Spectral overlap with OCO-2 bands** — Sentinel-2 B11 (1610 nm) overlaps WCO2 exactly;
   B7/B8A (~740–865 nm) covers O2-A; B12 (2190 nm) is near SCO2

### What it does NOT provide

- Instantaneous cloud field morphology (annual composite → not useful for cloud-proximity bias)
- Atmospheric state at measurement time
- Useful signal over homogeneous open ocean away from coastlines (embedding is near-uniform)

Primary value is for **coastal ocean soundings** (`sfc_type=0` near land) and
**land soundings** (`sfc_type=1`).

---

## Dataset Access

| Item | Detail |
|---|---|
| GEE dataset ID | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| Bands | `A00` – `A63` (64 dimensions, unit-length vectors on the unit sphere) |
| Resolution | 10 m (UTM projection) |
| Temporal coverage | 2017 – 2024 (annual composites) |
| Year matching | OCO-2 obs year → same year embedding; 2016 obs → use 2017 (earliest available) |
| License | CC with attribution to Google / Google DeepMind |
| Access | Free GEE account (research/education); no cost for GEE-side compute |
| Cloud Storage | `gs://alphaearth_foundations` — Requester Pays, avoid for this use case |

### One-time setup

```bash
pip install earthengine-api
earthengine authenticate   # opens browser OAuth
```

```python
import ee
ee.Initialize(project='<your-gcp-project-id>')  # free GCP project, no billing needed
```

---

## Implementation Status

| Phase | Status |
|---|---|
| A — OCO2Footprint vertex geometry | ✅ Complete |
| B — Phase 3.5 GEE extraction module | ⚠ Blocked (see Known Issues) |
| C — demo_combined.py wiring | ✅ Complete |
| C — FeaturePipeline integration (pipeline.py / transformer.py) | ⬜ Pending |
| C — Dimensionality reduction (PCA / projection) | ⬜ Pending (after permutation importance) |

---

## Implementation Phases

### Phase A — Extend `OCO2Footprint` with vertex geometry ✅
**File**: `src/pipeline/phase_03_processing.py`

#### A1. Add vertex fields to the dataclass

```python
@dataclass
class OCO2Footprint:
    sounding_id: int
    granule_id: str
    short_orbit_id: str
    latitude: float
    longitude: float
    sounding_time: datetime
    viewing_mode: str
    vertex_lon: np.ndarray = None   # shape (4,) — from L2 Lite vertex_longitude
    vertex_lat: np.ndarray = None   # shape (4,) — from L2 Lite vertex_latitude
```

Fields are optional (`= None`) so all existing `OCO2Footprint` construction sites
continue to work without modification.

#### A2. New method `_extract_vertex_data_from_lite`

Opens the L2 Lite file (already downloaded in Phase 2, already opened in
`_extract_sounding_ids_from_lite`) and reads vertex arrays.  Cached separately
as `lite_vertex_data.pkl` at the same day-level directory as `lite_sounding_ids.pkl`.

```python
def _extract_vertex_data_from_lite(self, filepath: Path,
                                   use_cache: bool = True
                                   ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    # cache path: data/processing/{year}/{doy:03d}/lite_vertex_data.pkl
    ...
    with nc.Dataset(filepath, 'r') as ds:
        ids  = ds.variables['sounding_id'][:]        # [N]
        vlon = ds.variables['vertex_longitude'][:]   # [N, 4]
        vlat = ds.variables['vertex_latitude'][:]    # [N, 4]
    return {int(sid): (np.array(vlon[i]), np.array(vlat[i]))
            for i, sid in enumerate(ids)}
```

**Do not merge with `lite_sounding_ids.pkl`** — that cache is a `set`; this is a `dict`
of arrays.  Merging would require invalidating all existing caches.

#### A3. Attach vertex data in `extract_oco2_footprints`

After the L1B footprint loop, one lookup pass:

```python
vertex_data = self._extract_vertex_data_from_lite(lite_file_path)
for sid, fp in footprints.items():
    if sid in vertex_data:
        fp.vertex_lon, fp.vertex_lat = vertex_data[sid]
```

---

### Phase B — Phase 3.5: GEE Embedding Extraction ✅
**New file**: `src/pipeline/phase_035_embedding.py`

Runs once per date after Phase 3, before Phase 4.  Reads `footprints.pkl`, extracts
64D mean + 64D std per footprint polygon from GEE, writes
`embedding_stats_{date}.parquet`.

#### B1. Footprint → GEE FeatureCollection

```python
features = []
for sid, fp in footprints.items():
    if fp.vertex_lon is None:
        continue
    coords = list(zip(fp.vertex_lon.tolist(), fp.vertex_lat.tolist()))
    # geodesic=False (3rd positional arg): planar geometry prevents GEE from
    # misreading a small polygon as its global complement due to winding order.
    poly = ee.Geometry.Polygon([coords], None, False)
    features.append(ee.Feature(poly, {'sounding_id': sid,
                                      'year': obs_year}))
fc = ee.FeatureCollection(features)
```

#### B2. Load annual embedding image

The collection uses `system:time_start` (standard GEE temporal property), **not** a
custom `year` metadata field.  Use `filterDate()`:

```python
obs_year = max(int(date[:4]), 2017)   # 2016 obs → 2017 embedding
embedding = (
    ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(f'{obs_year}-01-01', f'{obs_year + 1}-01-01')
    .first()
)
```

> **Bug fixed**: `.filter(ee.Filter.eq('year', obs_year))` returns an empty collection
> (no images have a `year` property) → `.first()` returns GEE null →
> `"Parameter 'image' is required and may not be null"` at map time.

#### B3. `reduceRegion` — mean + std per footprint

```python
def add_stats(feat):
    stats = embedding.reduceRegion(
        reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), sharedInputs=True),
        geometry=feat.geometry(),
        scale=10,
        maxPixels=50000,   # ~29 000 pixels per ~1.3×2.25 km footprint at 10 m
    )
    return feat.set(stats)

result_fc = fc.map(add_stats)
```

Output columns per footprint: `A00_mean … A63_mean`, `A00_stdDev … A63_stdDev`
(128 values total).

> **Bug fixed**: original `maxPixels=1000` was too small (~300 was wrong by 100×;
> correct count is ~29 000 pixels at native 10 m resolution).

#### B4. Export strategy (scale-aware)

| Dataset size | Strategy |
|---|---|
| Single orbit / dev | `result_fc.getInfo()` — synchronous, fast enough |
| Full dataset (84K soundings) | `ee.batch.Export.table.toDrive(result_fc, ...)` — one task per orbit or per date; poll with `ee.data.getTaskStatus()` |

Batch export produces a CSV per orbit in Google Drive; a post-processing step
concatenates them into `embedding_stats_{date}.parquet`.

#### B5. Output schema

```
sounding_id    int64
A00_mean       float32
...
A63_mean       float32
A00_stdDev     float32
...
A63_stdDev     float32
```

Saved to: `data/processing/{year}/{doy:03d}/embedding_stats_{date}.parquet`

---

### Phase C — Integrate into FeaturePipeline
**File**: `src/models/pipeline.py`

> **demo_combined.py wiring** ✅ complete — `run_phase_035()` is called between
> Phase 3 and Phase 4 **only when `--gcp-project` is explicitly supplied**.
> Non-fatal: a failed GEE call logs a warning and the pipeline continues
> without embeddings.
>
> **`--gcp-project` flag behaviour** (three modes):
> - *(flag omitted)* → Phase 3.5 skipped entirely
> - `--gcp-project` (flag, no value) → reads `GEE_PROJECT` env var; errors if unset
> - `--gcp-project my-id` → uses `my-id` directly
>
> The `GEE_PROJECT` env var alone (without the flag) does **not** trigger Phase 3.5.

#### C1. Load and merge embedding stats

In `FeaturePipeline.fit()` and `.transform()`, left-join embedding stats onto the
main dataframe by `sounding_id`.  Missing rows (open ocean far from coast where
embedding adds nothing) remain NaN — the existing `RobustScaler` + imputation
handles this.

#### C2. ~~New feature group in `_FEATURE_GROUPS` (transformer.py)~~ — RETARGET (2026-07-06)

> `src/models/transformer.py` was **deleted** in the 2026-07-03 model
> consolidation (see PIPELINE_CHANGELOG.md); the production model is the
> per-surface deep ensemble. If this integration is revived, the embedding
> block should instead be added as a feature block in
> `src/models/pipeline.py::_FEATURE_SETS` (like the contamination block) or —
> better matching its 128-wide correlated structure — compressed per surface
> via the ProfilePCA pattern (`src/models/profile_pca.py`) into a few EOF
> scores before entering the `full` set.

#### C3. Dimensionality reduction (optional)

128 new features increases model width significantly.  Options:
- **PCA on embedding columns** (fitted on training set only) → keep top 16–32 PCs
- **Linear projection layer** trained end-to-end in the transformer head
- Start without reduction; permutation importance will identify which dims matter

---

## Validation Experiment

After integration, run the following to confirm marginal value:

1. Train FT-Transformer with and without embedding features on the same
   train/test split
2. Compare test RMSE, MAE, R²
3. Check **permutation importance** — if embedding dims cluster near the top,
   they carry non-redundant information; if they rank below `cld_dist_km` and
   `alb_wco2`, the value is marginal
4. Stratify residuals by `cld_dist_km` bin — embedding should reduce residuals
   in coastal/heterogeneous footprints more than open-ocean footprints
5. Check separately for `sfc_type=0` (ocean) vs `sfc_type=1` (land) —
   expect larger gain on land

---

## File Change Summary

| File | Change | Status |
|---|---|---|
| `src/pipeline/phase_03_processing.py` | Add `vertex_lon/lat` to `OCO2Footprint`; new `_extract_vertex_data_from_lite`; attach in `extract_oco2_footprints` | ✅ Done |
| `src/pipeline/phase_035_embedding.py` | **New file** — GEE extraction script; `filterDate` fix; `geodesic=False`; `maxPixels=50000` | ✅ Done |
| `workspace/demo_combined.py` | Wire Phase 3.5 (lazy import); `--gcp-project` flag with env-var fallback only when flag is present | ✅ Done |
| `src/models/pipeline.py` | Load and merge embedding stats parquet | ⬜ Pending |
| ~~`src/models/transformer.py`~~ | ~~Add `'Surface\nEmbedding'` to `_FEATURE_GROUPS`~~ file deleted 2026-07-03; retarget to `_FEATURE_SETS` / ProfilePCA-style EOF block (see C2) | ⬜ Pending |

---

## Known Issues

### B3 — `reduceRegion` reports ~783 M pixels (unresolved)

After switching to `filterDate` and `geodesic=False`, GEE still reports
~783 million pixels for the first footprint in granule 22845a
(`"Too many pixels in the region. Found 783477249, but maxPixels allows only 50000"`).

Vertex data for this footprint is correct (lon span ~0.05°, lat span ~0.02°,
expected ~29 000 pixels at 10 m). Root cause unknown. Hypotheses investigated
and ruled out:
- **Anti-meridian crossing** — no footprints with `lon_span > 90°` in the dataset
- **Wrong filter property** — fixed by switching to `filterDate()`
- **Geodesic winding-order complement** — `geodesic=False` applied but did not
  reduce pixel count

**Current workaround**: Phase 3.5 is left as an opt-in step (requires explicit
`--gcp-project` flag). The main pipeline runs Phase 4 without embedding features.

**Next debugging steps** when revisiting:
1. Run `poly.area(1).getInfo()` on the first GEE polygon to confirm whether GEE
   sees a small or globe-sized area — this will determine if the issue is in the
   polygon geometry or in the image/scale interaction.
2. Try `bestEffort=True` in `reduceRegion` to let GEE auto-select scale.
3. Check whether the embedding image has data over the Southern Ocean
   (granule 22845a is near −62° latitude); the image may behave unexpectedly
   over areas with no data.

---

## Open Questions

- [ ] Does the free GEE account quota cover ~84K `reduceRegion` calls per batch export?
      (Expected: yes — well within free-tier limits once the pixel-count issue is resolved)
- [x] Anti-meridian handling: `geodesic=True` causes a CRS type error in current
      GEE Python API (the kwarg lands in the `proj` slot). Fixed: use
      `ee.Geometry.Polygon([coords], None, False)` (positional args).
- [ ] Root cause of 783 M pixel count in `reduceRegion` (see Known Issues above)
- [ ] For `sfc_type=0` open ocean: confirm embedding variance is near-zero and
      consider masking these features to zero rather than letting them add noise
- [ ] PCA reduction threshold: TBD after first permutation importance run
