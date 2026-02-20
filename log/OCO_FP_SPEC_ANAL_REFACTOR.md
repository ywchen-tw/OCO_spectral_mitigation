# Refactor: `oco_fp_spec_anal.py`

**Date**: 2026-02-17
**File modified**: `src/oco_fp_spec_anal.py`
**Lines**: 925 → 698 (−24%)

---

## Summary

`oco_fp_spec_anal.py` was refactored for efficiency and readability.  The 460-line
monolithic `cal_mca_rad_oco2` function was decomposed into focused, testable units.
Redundant I/O (Lite file + cloud-distance file re-read on every orbit call) was
eliminated, and three separate O(N) sounding-ID linear scans per footprint were
replaced with O(1) dict lookups built once.

---

## Bug Fixes

### 1. Shadowed `orbit_id` parameter (silent wrong-orbit processing)

**Location**: old `cal_mca_rad_oco2`, line 217
**Problem**: The function signature was `cal_mca_rad_oco2(sat, orbit_id, ...)`, but
the body immediately opened `for orbit_id in sat["orbit_list"]:`, overwriting the
parameter.  `run_simulation` also iterated the orbit list and called this function
per orbit, so on every call the inner loop re-iterated *all* orbits under the wrong
`orbit_id` variable.
**Fix**: `cal_mca_rad_oco2` replaced by `process_orbit(sat, orbit_id, shared_data,
...)` which operates on exactly one orbit.  `run_simulation` drives the orbit loop.

### 2. Hardcoded date string in cloud-distance file path

**Location**: old line 188
**Problem**: `f"{sat['result_dir']}/results_2018-10-18.h5"` — date hardcoded,
breaking every run on any other date.
**Fix**: `f"{sat['result_dir']}/results_{date}.h5"` using the formatted `date`
variable already in scope.

### 3. `fp_lat_j is not np.nan` identity comparison

**Location**: old XCO2 anomaly loop, line 495
**Problem**: `is not np.nan` is an identity check, not a value check; it is always
`True` for NaN floats, so the latitude guard never filtered invalid entries.
**Fix**: `~np.isnan(fp_lat)` used as a boolean mask in the vectorised replacement.

---

## Performance Changes

### Shared data hoisted out of per-orbit calls

| Data source | Old behaviour | New behaviour |
|---|---|---|
| `results_{date}.h5` (cloud distances) | Opened and fully read on every `cal_mca_rad_oco2` call | Read once in `load_shared_data(sat)`, result passed to all orbit calls |
| OCO-2 Lite `.nc4` (18 variables) | Opened and fully read on every call | Read once in `load_shared_data(sat)` |

Both are indexed into O(1) dicts:
```python
cld_dist_index = {sounding_id: nearest_cloud_distance_km}
lite_index     = {sounding_id: row_index_in_lite_arrays}
```

### O(N) sounding-ID scans replaced with O(1) dict lookups

Old code ran three separate linear scans per footprint `j`:
```python
if sounding_ind in snd_id_all[:, fp]:    # O(N) — L1B track scan
if sounding_ind in cld_snd_id:           # O(N) — cloud dist scan
if sounding_ind in oco_lt_id:            # O(N) — Lite ID scan
```

New code uses pre-built dicts for all three lookups:
```python
l1b_index    = {(sid, fp): track_ind}   # built once in load_orbit_data
cld_idx      = shared_data["cld_dist_index"]
lite_idx     = shared_data["lite_index"]
```

L1B radiance and position arrays are then filled with a single NumPy fancy-index
call (one per band) rather than per-sounding scalar assignments.

### XCO2 anomaly vectorised

Old code: Python double-loop — outer loop over N footprints, inner loop searching
`lat_diff` for each one.

New `compute_xco2_anomaly()`: builds an `[N, N]` pairwise latitude-difference
matrix, applies a clear-sky boolean mask, then calls `np.nanmean` / `np.nanstd`
once along axis 1:

```python
lat_mat  = np.abs(fp_lat[:, None] - fp_lat[None, :])   # [N, N]
ref_mask = (lat_mat <= lat_thres) & clear_mask[None, :]
ref_mean = np.nanmean(np.where(ref_mask, xco2[None, :], np.nan), axis=1)
```

> **Memory note**: the `[N, N]` matrix uses ~N²×8 bytes.  For N > ~5 000
> soundings per orbit consider a chunked implementation.

### Transmittance computed for all soundings at once

Old code computed `T = rad / toa_sol` per sounding in the fitting loop.
New `compute_transmittance(radiances, toa_sol)` operates on the full `[3, N, 1016]`
arrays in one NumPy call, with a single vectorised `T[T > 1] = np.nan` mask.

---

## Readability Changes

### Decomposition of `cal_mca_rad_oco2` (460 lines → 7 focused functions)

| New function | Responsibility | Approx. lines |
|---|---|---|
| `load_shared_data(sat)` | Read Lite + cloud-dist once; build index dicts | 45 |
| `load_orbit_data(sat, orbit_id)` | Read tau file + L1B; vectorised extraction | 55 |
| `compute_transmittance(radiances, toa_sol)` | T = rad/toa, mask T>1 | 10 |
| `fit_spectral_model(tau, ln_T, fit_order)` | SG-smooth + curve_fit | 10 |
| `compute_xco2_anomaly(...)` | Vectorised lat-window anomaly | 25 |
| `plot_fitting_example(...)` | Save example PNG plots | 40 |
| `process_orbit(sat, orbit_id, shared_data, ...)` | Orchestrates steps 1–7 | 160 |

`process_orbit` labels each step explicitly:
```
── 1. Load orbit data
── 2. Transmittance for all soundings at once
── 3. Per-sounding spectral fitting
── 4. Lite variable extraction (vectorised)
── 5. Cloud distance per sounding (O(1) dict lookup)
── 6. XCO2 anomaly (vectorised lat-window)
── 7. Write output HDF5
```

### Model dispatch dict replaces if/elif chain

Old (4-branch if/elif):
```python
if gas_fit_order == 1:
    log_transmittance_model = log_transmittance_model_1
elif gas_fit_order == 2:
    log_transmittance_model = log_transmittance_model_2
...
```

New (one-line dict lookup):
```python
LOG_TRANSMITTANCE_MODELS = {1: ..., 2: ..., 5: ..., 7: ...}
model_func = LOG_TRANSMITTANCE_MODELS[fit_order]
```

Adding a new truncation order now requires only one dict entry.

### `transmittance_model` signature simplified

Old: `transmittance_model(mean_ext, k1, k2, k3, k4, k5, k6, k7, intercept)` — only
`k1`, `k2`, and `intercept` were used; k3–k7 were ignored.
New: `transmittance_model(tau, k1, k2, intercept)` — matches actual computation.
`curve_fit` now estimates 3 parameters instead of 8.

### Plot code extracted to `plot_fitting_example()`

The plotting block (old lines 372–411) was entangled inside the per-sounding, per-band
fitting loop, controlled by three boolean flags with a typo (`plot_expample`).  It is
now a standalone function called with a single line when `not plot_done[tag]`.
`plt.close(fig)` added to prevent figure memory leaks on long runs.

### Lite variable extraction simplified via `_lite()` closure

Old: explicit `if sounding_ind in oco_lt_id: ... append(...)` blocks with 17 repeated
`list.append` + `else: list.append(np.nan)` pairs.

New: one closure + vectorised indexing covers all 17 variables:
```python
def _lite(key):
    out = np.full(N, np.nan)
    if valid_lt.any():
        out[valid_lt] = lite[key][row_inds[valid_lt]]
    return out
```

---

## Dead Code Removed

| Item | Reason |
|---|---|
| `fp_info` class | Defined but never used anywhere in the file |
| `cld_position_func()` | Defined but never called |
| `transmittance_model_3()` | Unused variant (3-cumulant, non-log form) |
| `log_transmittance_model_9()` | Unused 9th-order variant |
| Commented-out `else:` block (lines 586–609) | Re-reading output file — superseded by `overwrite` flag logic |
| `sys.exit()` before `k1k2_analysis` (line 725) | Permanently blocked downstream analysis; removed |
| ~80 commented-out `run_simulation(cfg, ...)` calls | Experiment log; not needed in source |
| Commented-out `cal_mca_rad_oco2` setup block (lines 163–178) | Leftover from earlier refactor attempt |

### Unused imports removed

`geopy`, `matplotlib.image`, `scipy.stats`, `cdata_sat_raw`, `path_dir`,
`haversine_vector`, `Unit` (haversine), `pandas`

---

## `run_simulation` changes

```python
# Before
sat0 = preprocess(...)
for orbit_id in orbit_list:
    cal_mca_rad_oco2(sat0, orbit_id=orbit_id, ...)  # re-reads Lite + cloud dist inside
sys.exit()          # k1k2_analysis never reached
k1k2_analysis(sat0)

# After
sat0        = preprocess(...)
shared_data = load_shared_data(sat0)   # read Lite + cloud dist ONCE
for orbit_id in sat0["orbit_list"]:
    process_orbit(sat0, orbit_id, shared_data, ...)
k1k2_analysis(sat0)                    # now reachable
```

---

## Output Unchanged

The schema of the output `fitting_details.h5` file is identical to the original:
all 37 dataset keys (`o2a_k1_fitting` … `ws_apriori_lt`) are preserved with the
same names, shapes, and fill-value conventions (NaN for missing).  Downstream
consumers (`result_ana.k1k2_analysis`) are unaffected.
