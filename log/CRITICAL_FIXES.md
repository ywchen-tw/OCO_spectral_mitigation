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
