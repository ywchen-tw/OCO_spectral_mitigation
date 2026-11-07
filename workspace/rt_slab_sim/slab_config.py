"""
Configuration for the Appendix G x-z slab RT simulation (single source of
truth -- Table G1 is generated from this module).

Scene: 2-D (x-z) boundary-layer water cloud in a periodic MCARaTS domain
(Ny=1, medium invariant along y; photons keep full 3-D angular freedom),
run under full 3-D transport and ICA ('ipa' solver) on an identical scene,
over a dark and a bright Lambertian surface.
"""
import os

import numpy as np

# --------------------------------------------------------------------------
# Input granule / footprint
# --------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# data root: CURC_DATA_ROOT, then OCO2_DATAROOT, else the repo root -- the
# parent of both data/ and results/ (src/config.py convention).  Laptop:
# <repo>/data/OCO2/...; CURC: $OCO2_DATAROOT/data/OCO2/...
_out_root = (os.environ.get("CURC_DATA_ROOT")
             or os.environ.get("OCO2_DATAROOT") or REPO_ROOT)
GRANULE_DIR = os.path.join(_out_root, "data", "OCO2", "2020", "001", "29252a_GL")

OCO_FILES = {
    "l1b": os.path.join(GRANULE_DIR, "oco2_L1bScGL_29252a_191231_B11006r_220715045716.h5"),
    "met": os.path.join(GRANULE_DIR, "oco2_L2MetGL_29252a_191231_B11006r_220624111442.h5"),
    "co2prior": os.path.join(GRANULE_DIR, "oco2_L2CPrGL_29252a_191231_B11006r_220624112210.h5"),
}

# Specific sounding to take Met/CO2-prior profiles + geometry from.
# None -> the median-index sounding with fully valid profiles (reported by
# build_atmosphere.py; freeze the chosen id here afterwards).
# 2020010100281632: SZA 55.0 deg (subtropical N Pacific, 29.9 N) -- chosen so
# the cloud-top shadow (~2 km * tan 55 = 2.9 km, ~6 columns) is resolvable;
# the first candidate (median sounding, SZA 22.7) put the shadow inside two
# columns and the demo showed no shadow side.
SOUNDING_ID = 2020010100281632

O2MIX = 0.20935  # matches fp_atm.py

# --------------------------------------------------------------------------
# Vertical grid (km above the footprint surface; 21 layers / 22 levels)
# 0-10 km @ 1 km, 10-20 @ 2 km, 20-40 @ 5 km, 40-60 @ 10 km
# --------------------------------------------------------------------------
LEVELS_KM = np.concatenate([
    np.arange(0.0, 10.5, 1.0),
    np.arange(12.0, 20.5, 2.0),
    np.array([25.0, 30.0, 35.0, 40.0, 50.0, 60.0]),
])

# --------------------------------------------------------------------------
# Horizontal slab domain (widened 24->32 km 2026-07-24 with the 3-4 km cloud:
# the SZA-55 shadow band reaches x ~ 20 km and needs clear far-field before
# the periodic wrap)
# --------------------------------------------------------------------------
NX = 64
DX_KM = 0.5          # 32 km domain
NY = 1
DY_KM = 1.0

# --------------------------------------------------------------------------
# Cloud (occupies the 3-4 km model layer; raised from 1-2 km 2026-07-24 so
# the SZA-55 shadow spans ~4-6 km from the cloud edge, well resolved)
# --------------------------------------------------------------------------
CLOUD_X_KM = (9.5, 14.5)   # 5 km wide, centered mid-domain
CLOUD_BASE_KM = 3.0
CLOUD_TOP_KM = 4.0
# Effective height in the MCARaTS runs (found 2026-10-10): er3t rtm/mca/mca_atm.py writes
# Atm_iz3l = (lay_index + 1) + 1, so the 3D layer lands one model layer (1 km here) above
# CLOUD_BASE_KM/CLOUD_TOP_KM. The shadow edges in slab_rad.h5 (x ~ 15.2-21.6 km at SZA 55)
# confirm 4-5 km. Report these values (CLOUD_*_EFF_KM, set after the scene
# override below); keep the requested ones so reruns reproduce the data.
CLOUD_COD = 10.0
CLOUD_CER_UM = 10.0        # micron, water cloud (pha_mie_wc)
# Phase function (2026-11-06): er3t's mca_atm_3d writes the Mie ASYMMETRY
# PARAMETER g(r_eff, lambda) into Atm_apfp3d, which MCARaTS reads as a
# Henyey-Greenstein phase function (apf in (-1, 1)); the tabulated Mie phase
# functions in mca_sca.bin are only used when apf >= 1 (table index).  Every
# run before 2026-11-06 therefore used HG with g = 0.86 (O2A, 10 um) -- the
# Table G1 wording "Mie phase function" is inaccurate.  "mie" switches the
# cloud cells to the tabulated Mie phase function of CLOUD_CER_UM.
CLOUD_PHASE = "hg"
# Boundary-layer aerosol as a second horizontally homogeneous 3-D
# constituent (None = no aerosol): aod at 768 nm (Angstrom-scaled to the
# band's Mie wavelength), requested base/top km (effective +1 km, like the
# cloud), single-scattering albedo and HG asymmetry parameter.
AEROSOL = None
# Lambertian albedo step along x (None = uniform): albedo_right for x >= x_km,
# the surface's own albedo to the left (surface-heterogeneity test).
SURFACE_STEP = None

# --------------------------------------------------------------------------
# Geometry / surfaces
# --------------------------------------------------------------------------
# SZA comes from the selected footprint; solar azimuth must put horizontal
# transport in the resolved x-dimension.  er3t takes COMPASS azimuth and
# converts via mca = 270 - compass (MCARaTS 0 = sun shining from west,
# photons travel toward +x).  Compass 270 (sun due west) -> mca 0: photons
# along +x, illuminated edge on the -x side, shadow band on the +x side.
# (Compass 0 = along y, the invariant dimension -- no resolvable shadow;
# that was the 2026-07-24 first-pass mistake.)
SOLAR_AZIMUTH = 270.0
SENSOR_ZENITH = 0.0        # nadir
SENSOR_AZIMUTH = 0.0
SENSOR_ALTITUDE_M = 705000.0
SURFACE_ALBEDOS = {"dark": 0.03, "bright": 0.30}
# Surface types for the band scenes (2026-11-06): Lambertian albedo per
# band (o2a / wco2 / sco2) representative of OCO-2 nadir/glint-free scenes --
# ocean (non-glint, ~0.02-0.03 everywhere), dense vegetation (NIR plateau at
# 765 nm, falling through the SWIR water-absorption bands), bare soil/desert
# (rising into the SWIR), snow (bright at 765 nm, dark at 1.6 and 2.06 um).
# "ocean" in O2A equals the Appendix G dark surface (0.03).
SURFACE_TYPES = {
    "ocean": {"o2a": 0.03, "wco2": 0.02, "sco2": 0.02},
    "veg":   {"o2a": 0.45, "wco2": 0.25, "sco2": 0.12},
    "soil":  {"o2a": 0.28, "wco2": 0.42, "sco2": 0.32},
    "snow":  {"o2a": 0.90, "wco2": 0.12, "sco2": 0.03},
}


def surface_set(band):
    """{surface type: albedo in this band}."""
    return {k: v[band] for k, v in SURFACE_TYPES.items()}

# Rayleigh scattering in the 1-D background (False -> er3t cal_mol_ext_atm is
# zeroed in run_slab; pure Beer-Lambert control, verified 2026-11-05)
RAYLEIGH = True
# SZA override (deg): None -> the footprint SZA; a float replaces meta/sza in
# slab_atm.h5 (build_atmosphere), so the airmass, the wavelength selection
# (gas_od) and the MCARaTS source direction (run_slab) all follow.
SZA_OVERRIDE = None
SOUNDING_SZA = 55.0        # footprint SZA of SOUNDING_ID, for geometry planning

# --------------------------------------------------------------------------
# Scenes (2026-11-06, reviewer-robustness sweep).  SLAB_SCENE=<name> selects
# one; unset/"base" reproduces the Appendix G configuration above exactly.
# Each entry overrides module constants; everything derived (effective cloud
# height, path-length bins, OUT_DIR) is computed after the override.
# Sensor azimuth: er3t compass 270 -> MCARaTS phi 0 = radiance direction
# along +x, i.e. the sensor sits down-sun of the cloud (forward scattering,
# glint-like); compass 90 puts it up-sun (backscatter).
# --------------------------------------------------------------------------
SCENES = {
    "base": {},
    "ref": {},          # base conditions, written to scenes/ref (same-budget control for the sweep)
    "sza25": {"SZA_OVERRIDE": 25.0},
    "sza70": {"SZA_OVERRIDE": 70.0, "NX": 96},          # 48 km: shadow to x ~ 28 km
    "vza43_fwd": {"SENSOR_ZENITH": 42.8, "SENSOR_AZIMUTH": 270.0},
    "vza43_back": {"SENSOR_ZENITH": 42.8, "SENSOR_AZIMUTH": 90.0},
    "cloud_low": {"CLOUD_BASE_KM": 0.0, "CLOUD_TOP_KM": 1.0},   # effective 1-2 km
    "cod2": {"CLOUD_COD": 2.0},
    "cod30": {"CLOUD_COD": 30.0},
    "rayleigh_off": {"RAYLEIGH": False},
    "width1": {"CLOUD_X_KM": (9.5, 10.5)},
    "width15": {"CLOUD_X_KM": (9.5, 24.5), "NX": 96},
    # CO2 bands on the base scene (BAND is read by the band table below, not
    # a module constant): does the path-length signature transfer across bands?
    # each band scene carries the four surface types at that band's albedo
    # (o2a_sfc = base geometry/cloud with the surface types)
    "o2a_sfc": {"SURFACE_ALBEDOS": surface_set("o2a")},
    "wco2": {"BAND": "wco2", "SURFACE_ALBEDOS": surface_set("wco2")},
    "sco2": {"BAND": "sco2", "SURFACE_ALBEDOS": surface_set("sco2")},
}
# Physics variants (2026-11-06, second batch): phase function, droplet size,
# thin/thick high cloud (eff. 9-10 km, 48-km domain: shadow reaches x ~ 29 km),
# boundary-layer aerosol (eff. 1-3 km), and a cloud-free albedo step (coast).
SCENES.update({
    "mie": {"CLOUD_PHASE": "mie"},
    "cer5": {"CLOUD_CER_UM": 5.0},
    "cer20": {"CLOUD_CER_UM": 20.0},
    "high_thin": {"CLOUD_BASE_KM": 8.0, "CLOUD_TOP_KM": 9.0, "CLOUD_COD": 1.0, "NX": 96},
    "high_cod10": {"CLOUD_BASE_KM": 8.0, "CLOUD_TOP_KM": 9.0, "NX": 96},
    "aerosol": {"AEROSOL": {"aod": 0.15, "base_km": 0.0, "top_km": 2.0,
                            "ssa": 0.95, "g": 0.70, "angstrom": 1.3}},
    "coast": {"CLOUD_COD": 0.0, "SURFACE_STEP": {"x_km": 16.0, "albedo_right": 0.30},
              "SURFACE_ALBEDOS": {"dark": 0.03}},
})
# The O2A condition ladder repeated for the CO2 bands (2026-11-06): scene
# "<band>_<condition>" = that condition with BAND switched and the band's
# dark/bright end-member surfaces (ocean, soil).
_LADDER = ("sza25", "sza70", "vza43_fwd", "vza43_back", "cloud_low", "cod2",
           "cod30", "rayleigh_off", "width1", "width15")
for _b in ("wco2", "sco2"):
    for _n in _LADDER:
        SCENES[f"{_b}_{_n}"] = dict(SCENES[_n], BAND=_b, SURFACE_ALBEDOS={
            _s: SURFACE_TYPES[_s][_b] for _s in ("ocean", "soil")})
# Sun glint (2026-11-06, third batch).  OCO-2 glint mode views the specular
# point over ocean: VZA = SZA, sensor down-sun of the footprint.  Surface =
# MCARaTS DSM model (isfc = 2, diffuse-specular mixture): Fresnel reflection
# off Cox-Munk wave facets (slope variance sigma^2 = 0.003 + 0.00512 U) plus
# a small Lambertian water-leaving/foam term f_d * alpha_d = 0.02.  A surface
# value that is a dict is a DSM spec (run_slab.SlabSfc2d); scalars stay
# Lambertian.  Refractive index of water per band (imaginary part negligible).
WATER_REFR = {"o2a": 1.33, "wco2": 1.318, "sco2": 1.30}


def glint_surface(band, wind_ms, alpha_d=0.40, f_d=0.05):
    return {"model": "dsm", "alpha_d": alpha_d, "f_d": f_d,
            "n_real": WATER_REFR[band], "n_imag": 0.0,
            "sigma2": 0.003 + 0.00512 * wind_ms, "wind_ms": wind_ms}


def glint_set(band, winds=(3, 7, 12)):
    return {f"u{w}": glint_surface(band, w) for w in winds}


# Sensor azimuth for the specular point: compass 90 (verified 2026-11-06 by
# the azimuth smoke test on glint_cod0: compass 90 -> effective reflectance
# 0.18 = Cox-Munk estimate rho_F/(4 mu0 muv sigma^2); compass 270 -> 0.03,
# diffuse only).  er3t's sensor azimuth is therefore the direction of the
# *view ray* (sensor -> ground), not of the radiance: compass 90 = view ray
# travelling -x, sensor on the +x (down-sun) side.  NB this means the
# vza43_fwd / vza43_back scene names are swapped physically: compass 270
# (vza43_fwd) puts the sensor up-sun (backscatter), compass 90 (vza43_back)
# down-sun (forward scatter).  SLAB_GLINT_AZ overrides for tests.
GLINT_SENSOR_AZIMUTH = float(os.environ.get("SLAB_GLINT_AZ", 90.0))
_GLINT = {"SZA_OVERRIDE": 30.0, "SENSOR_ZENITH": 30.0,
          "SENSOR_AZIMUTH": GLINT_SENSOR_AZIMUTH}
SCENES.update({
    "glint": dict(_GLINT, SURFACE_ALBEDOS=glint_set("o2a")),
    "glint_cod0": dict(_GLINT, CLOUD_COD=0.0, SURFACE_ALBEDOS={"u7": glint_surface("o2a", 7)}),
    "glint_sza15": dict(_GLINT, SZA_OVERRIDE=15.0, SENSOR_ZENITH=15.0, SURFACE_ALBEDOS=glint_set("o2a")),
    "glint_sza45": dict(_GLINT, SZA_OVERRIDE=45.0, SENSOR_ZENITH=45.0, SURFACE_ALBEDOS=glint_set("o2a")),
    "glint_cod2": dict(_GLINT, CLOUD_COD=2.0, SURFACE_ALBEDOS=glint_set("o2a")),
    "glint_cloud_low": dict(_GLINT, CLOUD_BASE_KM=0.0, CLOUD_TOP_KM=1.0, SURFACE_ALBEDOS=glint_set("o2a")),
    "wco2_glint": dict(_GLINT, BAND="wco2", SURFACE_ALBEDOS=glint_set("wco2")),
    "sco2_glint": dict(_GLINT, BAND="sco2", SURFACE_ALBEDOS=glint_set("sco2")),
})
SCENE = os.environ.get("SLAB_SCENE", "base")
if SCENE not in SCENES:
    raise KeyError(f"SLAB_SCENE={SCENE!r}; known scenes: {sorted(SCENES)}")
for _k, _v in SCENES[SCENE].items():
    if _k == "BAND":
        continue
    assert _k in globals(), _k
    globals()[_k] = _v
CLOUD_BASE_EFF_KM = CLOUD_BASE_KM + 1.0
CLOUD_TOP_EFF_KM = CLOUD_TOP_KM + 1.0
SCENE_SZA = SOUNDING_SZA if SZA_OVERRIDE is None else SZA_OVERRIDE

# --------------------------------------------------------------------------
# Spectral sampling (O2A band, monochromatic ABSCO wavelengths)
# --------------------------------------------------------------------------
# Band table (2026-11-06): windows match fp_abs_coeff xr_dict; absorber,
# ABSCO file/key and the density column of slab_atm.h5 per band.  The cloud
# COD stays at its 768 nm value in every band (water-droplet extinction
# efficiency ~2 across 0.76-2.1 um); droplet single-scattering albedo at the
# band wavelength comes from the er3t Mie table (absorbing at 1.6/2.06 um).
BANDS = {
    "o2a":  {"wvl_um": (0.755, 0.784), "gas": "o2",  "absco": "o2_v52.hdf",
             "gas_key": "Gas_07_Absorption", "dens": "d_o2_cm3",
             "mie_nm": 768.0, "fit_order": 7},
    "wco2": {"wvl_um": (1.587, 1.629), "gas": "co2", "absco": "co2_v52.hdf",
             "gas_key": "Gas_02_Absorption", "dens": "d_co2_cm3",
             "mie_nm": 1608.0, "fit_order": 3},
    "sco2": {"wvl_um": (2.020, 2.101), "gas": "co2", "absco": "co2_v52.hdf",
             "gas_key": "Gas_02_Absorption", "dens": "d_co2_cm3",
             "mie_nm": 2060.0, "fit_order": 7},
}
BAND = SCENES[SCENE].get("BAND", "o2a")
BAND_WVL_RANGE_UM = BANDS[BAND]["wvl_um"]
N_TAU_SAMPLES = 30                     # log-spaced in slant tau
SLANT_TAU_RANGE = (0.03, 8.0)          # production fit range (M9f caveat)
N_CONTINUUM = 3                        # near-zero-tau anchor points
PHA_MIE_WVL_NM = BANDS[BAND]["mie_nm"] # single Mie table for the whole band

# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
PHOTONS_LOCAL = 1.0e6
PHOTONS_PROD = 1.0e9
NRUN = 3
SOLVERS = ("3d", "ipa")                # identical scene; single-kwarg switch

# --------------------------------------------------------------------------
# Photon path-length statistics -- MCARaTS' built-in tally, enabled purely
# by namelist (injected via run_slab.SlabMcarats.extra_nml, where the
# mechanics are documented).  Rad_mplen=3 bins every radiance contribution's
# TOTAL geometric path (TOA -> scattering events -> sensor, summed from the
# per-layer Pho_plen/PhoV_plen arrays of the local-estimation trace) into
# PLEN_NBIN bins over [PLEN_MIN_M, PLEN_MAX_M], radiance-weighted and
# normalized per pixel -> the per-column PPDF.  Totals include the constant
# vacuum leg TOA(60 km) -> sensor(705 km) = 645 km for the nadir view, plus
# the injection-plane accounting; bins must bracket that offset (it cancels
# in all anomalies/variances).
# --------------------------------------------------------------------------
PLEN_MODE = 3
PLEN_NBIN = 2500
if SCENE == "base":
    PLEN_MIN_M = 640.0e3
    PLEN_MAX_M = 1140.0e3              # 200 m bins (historic Appendix G range)
else:
    # expected tallied direct path (km): vacuum leg to the sensor + the
    # injection-plane accounting (2 x TOA slant) + the nadir/slant up leg;
    # base: 645 + 209.2 + 60 = 914.2, the measured far-field L_peak 914.1
    _mu0 = np.cos(np.deg2rad(SCENE_SZA))
    _muv = np.cos(np.deg2rad(SENSOR_ZENITH))
    _toa = LEVELS_KM[-1]
    L_DIRECT_EXPECTED_KM = ((SENSOR_ALTITUDE_M / 1e3 - _toa) / _muv
                            + 2.0 * _toa / _mu0 + _toa / _muv)
    PLEN_MIN_M = np.floor((L_DIRECT_EXPECTED_KM - 60.0) / 10.0) * 10.0 * 1e3
    PLEN_MAX_M = PLEN_MIN_M + 500.0e3  # same 200 m bins

# --------------------------------------------------------------------------
# Production spectral estimator (mirror src/spectral defaults)
# --------------------------------------------------------------------------
FIT_ORDER = BANDS[BAND]["fit_order"]   # constants.FIT_ORDER = (7, 3, 7)
AEROSOL_AOD = (None if AEROSOL is None else
               AEROSOL["aod"] * (PHA_MIE_WVL_NM / 768.0) ** (-AEROSOL["angstrom"]))
FIT_SMOOTH = False                     # no-SG production path

# --------------------------------------------------------------------------
# Outputs.  Same data-root convention as the rest of the repo
# (src/config.py, workspace/tccon_* scripts): CURC_DATA_ROOT, then
# OCO2_DATAROOT, else the repo root, as the parent of results/.  On CURC the
# launcher exports OCO2_DATAROOT=/scratch/alpine/$USER/oco2_data, so
# everything lives under .../oco2_data/results/rt_slab_sim.
# --------------------------------------------------------------------------
OUT_DIR_BASE = os.path.join(_out_root, "results", "rt_slab_sim")
# scene trees: results/rt_slab_sim/<SCENES_SUBDIR>/<scene>; the 1e8 screening
# sweep lives in "scenes", the 1e9 production promotion in "scenes_1e9"
SCENES_SUBDIR = os.environ.get("SLAB_SCENES_SUBDIR", "scenes")
SCENES_DIR = os.path.join(OUT_DIR_BASE, SCENES_SUBDIR)
OUT_DIR = OUT_DIR_BASE if SCENE == "base" else os.path.join(SCENES_DIR, SCENE)
ATM_FILE = os.path.join(OUT_DIR, "slab_atm.h5")
OD_FILE = os.path.join(OUT_DIR, "slab_od.h5")
RAD_FILE = os.path.join(OUT_DIR, "slab_rad.h5")
TMP_DIR = os.path.join(OUT_DIR, "tmp-mca")
