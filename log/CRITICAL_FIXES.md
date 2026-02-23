# Critical Fixes

Documentation of bugs and code quality issues found and fixed across the pipeline.

---

## Fix 1 — Met/CO2Prior files always downloaded from first orbit (2026-02-17)

**File**: `src/phase_02_ingestion.py`
**Method**: `_query_ges_disc_directory`

### Problem

For `L2_Met` and `L2_CO2Prior` products, the directory query returned `matches[0]` — the
first `.h5` file in the daily GES DISC directory — regardless of which orbit was being
requested. Because the result was always the same first-orbit file, every orbit folder
received identical (and incorrect) Met/CO2Prior files.

`L1B` was unaffected because it uses a direct URL from Phase 1 metadata.
`L2_Lite` was unaffected because it already filtered matches by date string.

### Root Cause

```python
# Before fix — always picked the first file in the day directory
else:
    filename = matches[0]
```

### Fix

Added an `elif granule_id` branch (parallel to the existing `L2_Lite` branch) that
extracts the short orbit ID (e.g., `22845a`) from `granule_id.split('_')[2]` and
filters `matches` to find the file containing that orbit ID. Falls back to `matches[0]`
with a warning if no orbit-specific match is found.

```python
elif granule_id:
    parts = granule_id.split('_')
    if len(parts) >= 3:
        orbit_id = parts[2]  # e.g., "22845a"
        orbit_matches = [m for m in matches if orbit_id in m]
        if orbit_matches:
            filename = orbit_matches[0]
        else:
            filename = matches[0]  # fallback with warning
```

---

## Fix 2 — `skip_existing` bypassed download even when files were missing (2026-02-17)

**File**: `src/phase_02_ingestion.py`
**Method**: `download_all_for_date`

### Problem

When `skip_existing=True` (the default), the method checked for any
`sat_data_status.json` in the day's OCO2 directory. If one was found it returned
immediately — without verifying that the actual data files existed in each orbit folder.

After Fix 1, the incorrectly-placed Met/CO2Prior files from other orbits were removed,
but the status files remained. Subsequent runs (without `--force-download`) saw the
status files and short-circuited, never attempting to download the now-missing files.

### Root Cause

```python
if existing_status:
    # ... fetched metadata, listed existing files, then:
    return {... 'loaded_from_cache': True}   # <-- never re-downloaded missing files
```

### Fix

Removed the early return. The status check now only logs that a previous run was
detected, then falls through to the normal download loop. Per-file `output_path.exists()`
checks inside `download_oco2_granule` and `download_modis_granule` handle skipping
files that are already present and downloading those that are missing.

```python
if existing_status:
    logger.info("✓ Previous download status found. Verifying files and downloading any missing ones...")
# Always fall through to the download loop so per-file existence checks
# catch any files missing from orbit folders (e.g. after a partial run).
```

---

## Fix 3 — `run_phase_5` used hardcoded `data_dir="./data"` (2026-02-17)

**File**: `workspace/demo_combined.py`
**Function**: `run_phase_5`

### Problem

`GeometryProcessor` was instantiated with a hardcoded `data_dir="./data"`, ignoring the
platform-specific data directory resolved in `main()` (local vs. CURC path).

### Fix

Added `data_dir: Path` as a parameter to `run_phase_5` and updated the instantiation:

```python
# Before
geometry_processor = GeometryProcessor(data_dir="./data")

# After
geometry_processor = GeometryProcessor(data_dir=str(data_dir))
```

Call in `main()` updated to pass `data_dir=data_dir`.

---

## Fix 4 — Dead parameters in `run_phase_2` (2026-02-17)

**File**: `workspace/demo_combined.py`
**Function**: `run_phase_2`

### Problem

Two parameters were never functional:

- `metadata: Dict` — accepted but never referenced in the function body.
- `skip_phase: bool` — its guard `if skip_phase: return` was unreachable because
  `main()` only calls `run_phase_2` when `2 not in skip_phases`, so the argument
  `skip_phase=(2 in skip_phases)` was always `False`.

### Fix

Removed both parameters from the signature, the docstring, and the call site in `main()`.

---

## Fix 5 — `run_phase_3` accepted `visualize`/`viz_dir` but never used them (2026-02-17)

**File**: `workspace/demo_combined.py`
**Function**: `run_phase_3`

### Problem

The function signature included `visualize: bool = False` and `viz_dir: Optional[Path] = None`,
but no visualization code existed in the function body. The parameters were silently ignored
on every call.

### Fix

Removed both parameters from the signature, the docstring, and the call site in `main()`:

```python
# Before
def run_phase_3(target_date, data_dir, visualize=False, viz_dir=None): ...
processing_info, success = run_phase_3(target_date, data_dir, visualize=args.visualize, viz_dir=viz_dir)

# After
def run_phase_3(target_date, data_dir): ...
processing_info, success = run_phase_3(target_date, data_dir)
```

---

## Fix 6 — `import traceback` repeated inline in four except blocks (2026-02-17)

**File**: `workspace/demo_combined.py`

### Problem

`import traceback` appeared inside four separate `except` blocks
(`cleanup_modis_data`, `run_phase_3` ×2, `run_phase_4`, `run_phase_5`).
`traceback` is a stdlib module and should be imported once at the top level.

### Fix

Added `import traceback` to the top-level imports and removed all four inline imports.

---

## Fix 7 — Dual L2 Lite files downloaded when CMR returns cross-midnight orbit (2026-02-20)

**File**: `src/phase_02_ingestion.py`
**Method**: `download_all_for_date`

### Problem

When `fetch_oco2_xml_from_cmr` is used (fallback when GES DISC directory query fails), CMR
uses a **temporal-overlap** search: any granule whose time extent overlaps
`[target_date 00:00Z, target_date+1 00:00Z]` is returned.

The last orbit of the *previous* day (e.g. 2020-07-31 22:50 UTC → 2020-08-01 00:30 UTC)
overlaps with 2020-08-01's window and is returned by CMR.  For that orbit:

- `granule.start_time` = 2020-07-31 → `doy = 213`, `date_str = '200731'`
- `download_oco2_granule` downloads `oco2_LtCO2_200731_...nc4` into `data/OCO2/2020/213/`

The genuine Aug-01 orbits download `oco2_LtCO2_200801_...nc4` into `data/OCO2/2020/214/`.
Result: two L2 Lite files in the data tree for a single requested date.

### Fix

Added an off-day filter in `download_all_for_date` immediately after `parse_orbit_info`,
before any orbit/mode/limit filters and before any downloads:

```python
target_date_only = target_date.date()
off_day = [g for g in granules if g.start_time.date() != target_date_only]
if off_day:
    logger.warning(
        f"Dropping {len(off_day)} granule(s) whose start_time is not on "
        f"{target_date_only}: {[g.granule_id for g in off_day]}"
    )
    granules = [g for g in granules if g.start_time.date() == target_date_only]
```

Works for both timezone-aware datetimes (CMR returns `+00:00`) and naive datetimes
(GES DISC directory path), since `.date()` returns the UTC calendar date in both cases.
The GES DISC directory path is not affected (it never returns off-day files); the filter
is a no-op there and adds only negligible overhead.

---

## Fix 8 — Corrupted L2 Lite `.nc4` silently accepted; cryptic HDF error in spec anal (2026-02-20)

**Files**: `src/phase_02_ingestion.py`, `src/oco_fp_spec_anal.py`

### Problem

GES DISC returns HTTP 200 with an HTML login page (instead of 401/403) when the session
cookie expires. `_download_file` only called `raise_for_status()` — which passes on a
200 — so the HTML page was written to disk as a `.nc4` file. On the next run,
`download_oco2_granule`'s `output_path.exists()` check accepted the corrupt file without
re-downloading. When `oco_fp_spec_anal.py` later opened it via `netCDF4.Dataset`, the
HDF5 library returned an opaque `OSError: [Errno -101] NetCDF: HDF error`.

### Fixes (three layers)

**A. `_download_file`** — reject HTML responses before writing to disk:
check `Content-Type: text/html` and raise immediately.

**B. `download_oco2_granule` "file exists" branch** — validate HDF5 header with
`h5py.is_hdf5()` for `.nc4/.h5/.hdf5` files; delete and re-download if invalid.

**C. `preprocess` in `oco_fp_spec_anal.py`** — validate L2 Lite before passing it to
`oco_fp_atm_abs`; raise an `OSError` with the filename and remediation command.
Also fixed glob pattern from `*nc4` to `*.nc4`.

### Immediate remediation for already-corrupted files

```bash
rm /path/to/data/OCO2/YYYY/DOY/oco2_LtCO2_*.nc4
python workspace/demo_combined.py --date YYYY-MM-DD
```

---

## Fix 9 — MODIS granule matching intermittent / coverage gaps (2026-02-23)

**Files**: `src/phase_02_ingestion.py`, `src/phase_03_processing.py`, `workspace/demo_combined.py`

Six separate problems caused MODIS granules to be inconsistently or incompletely
matched to OCO-2 granules across pipeline runs.

---

### Fix 9-A — Timezone mismatch caused zero MODIS downloads on GES DISC runs

**File**: `src/phase_02_ingestion.py` — `find_modis_granules`

**Root cause**: The MODIS `granule_time` was created with `tzinfo=timezone.utc`
(timezone-aware), but `search_start`/`search_end` inherited the timezone of the
OCO-2 `granule.start_time`. The GES DISC XML path produces **naive** datetimes
(the `.replace('Z', '+00:00')` is a no-op when the time string has no `Z` suffix).
The resulting `TypeError: can't compare offset-naive and offset-aware datetimes`
was silently swallowed by the `except Exception` in the LAADS query loop, so
**zero MODIS granules were found** whenever GES DISC was the Phase 1 source.
When GES DISC failed and CMR fallback was used, times were `+00:00`-aware and the
comparison succeeded — explaining the run-to-run intermittency for the same date.

**Fix**: Added `_to_naive_utc()` helper that strips `tzinfo` from both
`start_time` / `end_time` bounds before computing `search_start`/`search_end`.
Removed `tzinfo=timezone.utc` from the MODIS `granule_time` construction so both
sides are always naive UTC.

```python
def _to_naive_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

start_naive = _to_naive_utc(start_time)
end_naive   = _to_naive_utc(end_time)
search_start = start_naive - timedelta(minutes=effective_buffer)
search_end   = end_naive   + timedelta(minutes=effective_buffer)

# Granule time — naive UTC
granule_date = datetime(g_year, 1, 1) + timedelta(days=g_doy - 1)
granule_time = granule_date.replace(hour=hour, minute=minute)
```

---

### Fix 9-B — Phase 2 download buffer reduced unnecessarily, risking coverage gaps

**File**: `src/phase_02_ingestion.py` — `find_modis_granules`

**Root cause**: `find_modis_granules` reduced the download buffer to ±10 min for
`year < 2022`. Phase 2 uses nominal OCO-2 times from Phase 1 XML metadata, which
can differ from actual L1B sounding times by several minutes. With only ±10 min of
margin, MODIS granules at the edge of the actual observation window were not
downloaded, leaving silent cloud-data gaps at the start and end of each orbit.

**Fix**: Removed the adaptive reduction from the download path entirely. Phase 2
now always downloads with the full `±buffer_minutes` (default ±20 min). The
adaptive tighter window is left to the Phase 3 matching step, where it governs
accuracy of collocation, not completeness of the download.

---

### Fix 9-C — Adaptive buffer cutoff year was 2022 in Phase 2 but the intent was 2022

**File**: `workspace/demo_combined.py` — `run_phase_3`

**Root cause**: The hardcoded `buffer_seconds = 20 * 60` in `run_phase_3` ignored
the adaptive logic present in `match_temporal_windows` (`year < 2023` → ±10 min).
This mismatch meant the demo's matching window differed from the module's matching
window for the same data.

**Fix**: Replaced the hardcoded constant with adaptive logic that mirrors the
project standard: `year < 2022` → ±10 min, `year ≥ 2022` → ±20 min.

```python
buffer_minutes = 10 if year < 2022 else 20
buffer_seconds = buffer_minutes * 60
```

---

### Fix 9-D — Phase 3 cache never invalidated after Phase 2 re-downloads

**File**: `workspace/demo_combined.py` — `run_phase_3`

**Root cause**: The cache-validity check only tested whether
`granule_combined_*.pkl` existed on disk. If the cache was created during a
partial Phase 2 run (some MODIS files missing), it was reused on every subsequent
run even after Phase 2 downloaded the missing files.

**Fix**: Before the early-return cache hit, the newest mtime across all MODIS HDF
files on disk is computed. Any cache file older than that mtime is treated as
stale, the granule is added back to `missing_granules`, and the cache is rebuilt
with the current complete MODIS file set.

```python
latest_modis_mtime = 0.0
for modis_dir in [MYD35_L2_dir, MYD03_dir]:
    for hdf in modis_dir.glob("*.hdf"):
        latest_modis_mtime = max(latest_modis_mtime, hdf.stat().st_mtime)

cache_mtime = min(f.stat().st_mtime for f in combined_files)
if latest_modis_mtime > cache_mtime:
    # invalidate — granule added to missing_granules for reprocessing
```

---

### Fix 9-E — No warning when MODIS granule coverage had gaps

**File**: `workspace/demo_combined.py` — `run_phase_3`

**Root cause**: After the MODIS–OCO-2 matching loop there was no check that MODIS
granules covered every ~5-minute slot in the OCO-2 orbit. A gap (e.g. one missing
granule between 14:35 and 14:45) caused all soundings in that interval to receive
no cloud data, silently.

**Fix**: After the matching loop, for each OCO-2 granule the matched MODIS times
are sorted and consecutive gaps > 6 minutes are logged as warnings. Granules with
zero MODIS matches are also warned.

```
⚠ MODIS coverage gap: 14:35 → 14:45 (10 min) for oco2_L1bScGL_22845a_...
⚠ No MODIS granules matched for oco2_L1bScGL_22846a_...
```

---

### Fix 9-F — Night-pass MODIS granules included in cloud collocation

**Files**: `workspace/demo_combined.py` — `run_phase_3` matching loop;
`src/phase_03_processing.py` — `_unpack_cloud_mask`

**Root cause**: The matching code in `run_phase_3` included all MYD35_L2 files
whose start time fell in the OCO-2 ±buffer window, without checking the day/night
flag. Near orbit boundaries a MODIS night granule can have a start timestamp
inside the window, introducing cloud pixels from a completely different scene.

**Fix (two layers)**:

1. `run_phase_3` — before adding a MODIS file to `modis_to_oco2_mapping`, the
   code reads byte 0 of the cloud mask and checks bit 3 (day/night flag). Files
   where night pixels outnumber day pixels are skipped.

2. `_unpack_cloud_mask` — after computing `is_day_pass`, night passes now
   immediately return an empty `MODISCloudMask` (zero pixels) rather than
   processing all cloud pixels from the wrong scene. This acts as a second safety
   net for any code path that reaches `extract_modis_cloud_mask` directly.

---

## Fix 10 — Cross-date granule: `footprints.pkl` never written; granule never processed (2026-02-23)

**File**: `workspace/demo_combined.py` — `run_phase_3`

### Problem

The first OCO-2 orbit of every target date typically starts on the *previous* calendar
day (e.g., target date 2020-08-01 but L1B filename contains `200731`). These granules
are downloaded into the target date's OCO-2 directory and appear in `missing_granules`,
but they are never processed:

1. OCO-2 L2 Lite files are organised by **orbit date** (the date in the filename), not
   by individual sounding UTC time. All sounding IDs for orbit `22845a` (starting Jul 31)
   are in `oco2_LtCO2_200731_...nc4` — stored in the **Jul 31 directory** — not in the
   Aug 1 Lite file.
2. `run_phase_3` loads only the Lite file from the target date's directory
   (`data/OCO2/2020/214/`), so `valid_sounding_ids` contains no IDs from orbit 22845.
3. All footprints from the cross-date L1B are filtered out → `l1b_footprints` is empty.
4. `if use_cache and l1b_footprints:` is `False` → `footprints.pkl` is never written.
5. No footprints → no MODIS matching → no `granule_combined_*.pkl` → granule stays in
   `missing_granules` on every subsequent run.

### Fix

Added cross-date detection in `run_phase_3` inside the `oco2_files` construction loop.
When an L1B filename date ≠ `target_date`, the previous day's Lite file is located in
`data/OCO2/{l1b_year}/{l1b_doy:03d}/` and added to `oco2_files` so
`extract_oco2_footprints` can find the correct sounding IDs for quality filtering.

A `_lite_dirs_added` set prevents the same directory from being added multiple times if
several cross-date granules share the same previous date.

If the previous day's directory does not exist (e.g., that date was never downloaded),
a warning is logged and the granule proceeds without Lite-based quality filtering.

```python
_m = re.search(r'_(\d{6})_B', file_path.name)
if _m and _m.group(1) != target_date.strftime("%y%m%d"):
    _prev_dir = data_dir / "OCO2" / str(_l1b_dt.year) / f"{_l1b_doy:03d}"
    if _prev_dir.exists() and _prev_dir not in _lite_dirs_added:
        for _lp in _prev_dir.glob("*.nc4"):
            if _lp.is_file() and "Lt" in _lp.name:
                oco2_files.append(DownloadedFile(..., product_type="L2_Lite"))
        _lite_dirs_added.add(_prev_dir)
```
