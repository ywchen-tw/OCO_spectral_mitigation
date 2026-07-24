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
GRANULE_DIR = os.path.join(REPO_ROOT, "data", "OCO2", "2020", "001", "29252a_GL")

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
CLOUD_COD = 10.0
CLOUD_CER_UM = 10.0        # micron, water cloud (pha_mie_wc)

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

# --------------------------------------------------------------------------
# Spectral sampling (O2A band, monochromatic ABSCO wavelengths)
# --------------------------------------------------------------------------
BAND = "o2a"
BAND_WVL_RANGE_UM = (0.755, 0.784)     # matches fp_abs_coeff xr_dict[0]
N_TAU_SAMPLES = 30                     # log-spaced in slant tau
SLANT_TAU_RANGE = (0.03, 8.0)          # production fit range (M9f caveat)
N_CONTINUUM = 3                        # near-zero-tau anchor points
PHA_MIE_WVL_NM = 768.0                 # single Mie table for the whole band

# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
PHOTONS_LOCAL = 1.0e6
PHOTONS_PROD = 1.0e9
NRUN = 3
SOLVERS = ("3d", "ipa")                # identical scene; single-kwarg switch

# --------------------------------------------------------------------------
# Photon path-length statistics (MCARaTS Rad_mplen=3: per-pixel histogram of
# total geometric path, contribution-weighted, normalized by pixel radiance).
# Total path includes the constant vacuum leg TOA(60 km) -> sensor(705 km)
# = 645 km for the nadir view; bins must bracket it.
# --------------------------------------------------------------------------
PLEN_MODE = 3
PLEN_NBIN = 2500
PLEN_MIN_M = 640.0e3
PLEN_MAX_M = 1140.0e3                  # 200 m bins

# --------------------------------------------------------------------------
# Production spectral estimator (mirror src/spectral defaults)
# --------------------------------------------------------------------------
FIT_ORDER = 7                          # constants.FIT_ORDER[0] (O2A)
FIT_SMOOTH = False                     # no-SG production path

# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
OUT_DIR = os.path.join(REPO_ROOT, "results", "rt_slab_sim")
ATM_FILE = os.path.join(OUT_DIR, "slab_atm.h5")
OD_FILE = os.path.join(OUT_DIR, "slab_od.h5")
RAD_FILE = os.path.join(OUT_DIR, "slab_rad.h5")
TMP_DIR = os.path.join(OUT_DIR, "tmp-mca")
