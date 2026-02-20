# fp_abs_coeff.py — Refactor Log

**File:** `src/abs_util/fp_abs_coeff.py`
**Date:** 2026-02-18
**Scope:** Doppler corrections, solar spectrum source, Rayleigh calculation

---

## Summary of Changes

| # | Change | Location |
|---|---|---|
| 1 | Fixed argument order in `cal_mol_ext` call | `_process_track_all_bands` |
| 2 | Replaced `solar.txt` with solar H5 model | `oco_fp_abs_all_bands` + worker |
| 3 | Added per-sounding Doppler shift for gas/H2O ILS (`v_inst`) | `_process_track_all_bands` |
| 4 | Fixed solar Doppler shift starting frame | `_process_track_all_bands` |
| 5 | Moved Rayleigh cross-section to per-sounding with atmosphere-frame λ | `_process_track_all_bands` |
| 6 | Added `v_solar`, `v_inst`, `dist_au` to per-sounding `track_args` | `oco_fp_abs_all_bands` |

---

## Wavelength / Wavenumber Grids — Definitive Reference

Four distinct grids exist in `_process_track_all_bands`. Every calculation is mapped to exactly one of them.

### Grid definitions

| Variable | Frame | Units | Source |
|---|---|---|---|
| `wavedat` | Atmosphere rest frame | cm⁻¹ | `find_boundary()` on ABSCO table |
| `wavedat_inst` | Instrument frame | cm⁻¹ | `wavedat / (1 + v_inst/c)` |
| `wavedat_rest` | Solar rest frame | cm⁻¹ | `wavedat_inst * (1 + (v_solar+v_inst)/c)` |
| `wloco` | Instrument frame | µm | `oco_wv_absco()` — dispersion polynomial |
| `wlabsco` | Instrument frame | µm | `10000 / wavedat_inst` |
| `wloco_atm` | Atmosphere rest frame | µm | `wloco * (1 + v_inst/c)` |

### Doppler shift chain

```
wavedat  (atmosphere rest frame)
  │
  │  ÷ (1 + v_inst/c)
  ▼
wavedat_inst  (instrument frame)  ◄── same frame as wloco, wlabsco
  │
  │  × (1 + (v_solar + v_inst)/c)
  ▼
wavedat_rest  (solar rest frame)
```

```
wloco  (instrument frame, µm)
  │
  │  × (1 + v_inst/c)
  ▼
wloco_atm  (atmosphere rest frame, µm)
```

---

## Per-Calculation Wavelength Usage

### 1. ABSCO T/P lookup (`rdabsco_species_absco`)

**Grid:** `wavedat` — **atmosphere rest frame**

```python
absco = rdabsco_species_absco(
    absco_data_gas, ..., iwcm1, iwcm2, ...
)
```

No Doppler correction is applied. Absorption cross-sections σ(T, P, ν) are physical properties of the gas in the atmosphere rest frame. The indices `iwcm1`, `iwcm2` select the band's wavenumber range from the ABSCO table (also in atmosphere frame).

---

### 2. ILS pixel-to-ABSCO mapping (`oco_conv_absco` → `indlr`)

**Grid:** `wavedat_inst` — **instrument frame**

```python
wavedat_inst = wavedat / (1.0 + v_inst / C_LIGHT)
indlr = oco_conv_absco(wloco, xx, yy, wavedat_inst, nwav)
```

`oco_conv_absco` determines, for each instrument pixel `wloco[ind]` (instrument frame), which ABSCO array indices fall within that pixel's ILS support. Since `wloco` is in the instrument frame, the ABSCO wavenumber grid must also be in the instrument frame for the mapping to be correct.

---

### 3. ILS weight evaluation (`ilg0`)

**Grids:** `wlabsco` and `wloco` — both **instrument frame**

```python
wlabsco = 10000.0 / wavedat_inst.astype(np.float64)
...
ilg0 = np.interp(wlabsco[start:end], xx[ind, :] + wloco[ind], yy[ind, :])
```

The ILS function is defined by (`xx` offsets, `yy` response), both centred on `wloco[ind]` in the instrument frame. Evaluating it at `wlabsco[start:end]` requires both arrays to be in the same frame (instrument).

---

### 4. Gas + H2O extinction ILS convolution (`ext_profile`)

**Grid:** ABSCO array indices from `indlr` (derived from `wavedat_inst`)

```python
ext_profile[ind, :] = ext[start:end, :].T @ ilg0 / ilg0_sum
```

`ext[:, iz]` is indexed by ABSCO wavenumber position (atmosphere frame), but the ILS weights `ilg0` were computed using the Doppler-shifted index mapping (`indlr` from `wavedat_inst`), so the convolution correctly projects the extinction profile onto instrument-frame pixels.

---

### 5. TOA solar irradiance (`fsol`, `toa_sol`)

**Grid:** `wavedat_rest` — **solar rest frame**

```python
beta_total   = (v_solar + v_inst) / C_LIGHT
wavedat_rest = wavedat_inst * (1.0 + beta_total)   # instrument → solar rest
fsol = (np.interp(wavedat_rest, sol_abs_nu, sol_abs_val, left=1.0, right=1.0) *
        np.interp(wavedat_rest, sol_cont_nu, sol_cont_val)                      *
        (1.0 / dist_au) ** 2)
...
toa_sol[ind] = np.dot(fsol[start:end], ilg0) / ilg0_sum * musolzen
```

The solar H5 tables (`sol_abs_nu`, `sol_cont_nu`) are in the **solar rest frame**. To look up the correct solar value for each observed ABSCO point, the ABSCO instrument-frame wavenumbers are shifted to the solar rest frame using both `v_solar` (Earth orbital velocity around Sun) and `v_inst` (spacecraft velocity relative to Earth), consistent with the sign convention in `derive_oco2_solar_radiance`.

`toa_sol` then applies the same `ilg0` ILS weights as `ext_profile`, since the solar irradiance is also convolved through the instrument lineshape.

**Previous (incorrect) implementation:**
```python
# BUG: wavedat is atmosphere-frame, but v_solar+v_inst is the instrument→solar correction
wavedat_rest = wavedat * (1.0 + beta_total)   # double-counted v_inst
```
Error magnitude: O(v_solar × v_inst / c²) ≈ 7×10⁻⁹ (numerically negligible but conceptually wrong).

---

### 6. Rayleigh optical depth (`tau_molec_ext_lays`)

**Grid:** `wloco_atm` — **atmosphere rest frame** (per sounding)

```python
wloco_atm    = wloco * (1.0 + v_inst / C_LIGHT)   # instrument µm → atmosphere µm
molec_ext_fp = mol_ext_wvl(wloco_atm)              # σ_Rayleigh at atm-frame λ
tau_molec_ext_lays = np.sum(cal_mol_ext(molec_ext_fp, d_air_lay, dzf), axis=1)
```

`mol_ext_wvl` implements the Bodhaine et al. (1999) Eq. 29 cross-section formula. The atmosphere scatters the photon at its **rest-frame** wavelength, so `wloco_atm` (atmosphere frame) is the physically correct input.

**Previous implementation:** used `wloco` (instrument frame), computed once per footprint at setup time. Ignored the per-sounding `v_inst` Doppler correction. Error: Δσ/σ ≈ 4 × |v_inst|/c ≈ 0.01%, negligible in magnitude but now corrected for consistency.

---

## Changes to `oco_fp_abs_all_bands` (Setup)

### Removed

| Item | Reason |
|---|---|
| `from abs_util.abs.solar import solar` | `solar.txt` replaced by solar H5 model |
| `wlsol0, fsol0 = solar(pathinp + "sol/solar.txt")` | Same |
| `fsol = np.interp(wavedat, wlsol0, fsol0)` per band | Same; solar now computed per sounding |
| `'fsol': fsol` in `band_states` | No longer a static array |
| `molec_ext = np.empty((8, 1016), ...)` | Moved to per-sounding |
| `molec_ext[fp] = mol_ext_wvl(wloco_fps[fp])` | Moved to per-sounding |
| `'molec_ext': molec_ext` in `band_states` | Same |
| `atm_dict["doy"]` in `track_args` | `cal_sol_fac(doy)` no longer used |

### Added

| Item | Reason |
|---|---|
| `solar_h5_path = pathinp + "sol/solar.h5"` | H5 solar model path |
| Read `sol_abs_nu/val`, `sol_cont_nu/val` per band from H5 | Pre-read once; stored in `band_states` |
| `'solar_h5_path'` in `shared_state` | Accessible to workers |
| `float(atm_dict["v_solar"][i])` in `track_args` | Per-sounding solar Doppler |
| `float(atm_dict["v_inst"][i])` in `track_args` | Per-sounding instrument Doppler |
| `float(atm_dict["dist_au"][i])` in `track_args` | Per-sounding Earth-Sun distance |
| `else: raise RuntimeError(...)` for unknown platform | Prevents silent `pathinp` unbound error |

---

## Bug Fixed: `cal_mol_ext` Argument Order

**Before:**
```python
tau_molec_ext_lays = np.sum(cal_mol_ext(molec_ext_fp, dzf, d_air_lay), axis=1)
#                                                      ^^^  ^^^^^^^^^^
#                                                      ← swapped →
```

**After:**
```python
tau_molec_ext_lays = np.sum(cal_mol_ext(molec_ext_fp, d_air_lay, dzf), axis=1)
```

Signature: `cal_mol_ext(mol_ext_wvl_array, d_air_lay [molec/cm³], dz [km])`.
The result was numerically identical (multiplication is commutative), but the argument names were semantically swapped.
