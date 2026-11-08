"""
Single source of truth for numeric constants shared across the pipeline
========================================================================

Every value here was previously duplicated (with drift) across 2-4 call
sites; see log/PROJECT_REVIEW.md §7.4. Import from this module
instead of re-declaring literals, and have docstrings reference the
constant name rather than repeating the number.
"""
import os

# ---------------------------------------------------------------------------
# Phases 2-3: OCO-2 <-> MODIS temporal matching
# ---------------------------------------------------------------------------
# Aqua entered free drift and its equator-crossing time began moving in 2022,
# so collocation needs a wider time window from that year on. Phase 2 always
# *downloads* with the full post-drift buffer; Phase 3 *matching* is adaptive
# via modis_match_buffer_minutes().
AQUA_FREE_DRIFT_YEAR = 2022
MODIS_MATCH_BUFFER_PRE_DRIFT_MIN = 10   # ± minutes, years < AQUA_FREE_DRIFT_YEAR
MODIS_MATCH_BUFFER_MIN = 20             # ± minutes, free-drift era + Phase-2 downloads


def modis_match_buffer_minutes(year: int) -> int:
    """Adaptive Phase-3 matching buffer (± minutes) for an observation year."""
    if year < AQUA_FREE_DRIFT_YEAR:
        return MODIS_MATCH_BUFFER_PRE_DRIFT_MIN
    return MODIS_MATCH_BUFFER_MIN


# ---------------------------------------------------------------------------
# Phase 4: banded nearest-cloud distance calculation
# ---------------------------------------------------------------------------
# Production values are the oco_modis_cloud_distance.py CLI defaults (the active path).
# The overlap buffer must comfortably exceed the largest cloud distance of
# interest so band edges cannot truncate a nearest-cloud search (0.5 deg
# latitude is roughly 55 km).
CLOUD_DIST_BAND_WIDTH_DEG = 2.5
CLOUD_DIST_BAND_OVERLAP_DEG = 0.5

# ---------------------------------------------------------------------------
# Spectral fitting: cumulant truncation order per band (o2a, wco2, sco2)
# ---------------------------------------------------------------------------
# (7, 3, 7) is the production order: WCO2 slant optical depth maxes at ~1 so
# its order stays low; higher orders on the smoothed curve overfit low-tau
# bands (see memory note fitting-order-per-band).
FIT_ORDER = (7, 3, 7)

# Fit variant the phenomenology analyses read: '_nosg' = the fit to the
# unsmoothed spectrum (Sect. 2.2, Fig. 2, the model predictors), '' = the
# SG(51, 3)-smoothed production fit.  Applies to k1-k3 and gamma; their _nosg
# clear-sky reference statistics come from analysis.nosg_reference (sidecar).
# models.pipeline selects its predictors by fit_method (FIT_METHODS) since 2026-10-31.
PATH_LENGTH_FIT_SUFFIX = '_nosg'

# ---------------------------------------------------------------------------
# Model run version (log/PIPELINE_FIX_RERUN_PLAN_2026-10-14.md)
# ---------------------------------------------------------------------------
# Same convention as the launchers' RUN_VERSION: default 'v2' (fill masking,
# RobustClipScaler, unsmoothed gamma, no t700); an empty value selects the v1
# names.  The suffix sits before the fold tag (..._r05_v2_f0) and at the end of
# every evaluation-tree name (..._o05l15_m5_v2, de_prof_mix_<arm>_v2).
RUN_VERSION = os.environ.get('RUN_VERSION', 'v2')
RUN_SUFFIX = f'_{RUN_VERSION}' if RUN_VERSION else ''
DE_PRODUCTION_TAG = 'de_beta_nll_prof_reg_foldpca_o05l15_m5' + RUN_SUFFIX

# Manuscript (JQSRT) model since 2026-11-03: tuned XGBoost c41 (run prefix
# fm1620k5v2xt41, fit method am) with feature set no_xco2_sig_shape_nok2diag,
# on the fit-schema-v2 data.  Its evaluation tree is
# results_doppler/model_comparison/xgb/<XGB_PRODUCTION_TAG>/{atrain,drift};
# the ablation arms append _<arm>, the ridge tree is
# results_doppler/model_comparison/linreg/<LINREG_PRODUCTION_TAG>.  The
# manuscript generators read these names; do not hard-code them elsewhere.
# (no_xco2_sig_shape, the 2026-11-02 set, is retired.)
XGB_PRODUCTION_FEATURE_SET = 'no_xco2_sig_shape_nok2diag'
XGB_PRODUCTION_TAG = f'fm1620k5v2xt41_am_{XGB_PRODUCTION_FEATURE_SET}'
LINREG_PRODUCTION_TAG = f'fm1620k5v2_am_{XGB_PRODUCTION_FEATURE_SET}'
# Same model, settings, folds and feature set trained on the raw-XCO2 anomaly
# (target xco2_raw_anomaly_r05 / _r15 instead of xco2_bc_anomaly_*): the
# ML(raw) series of Table C8 (SI "Model trained on the raw product").
XGB_RAW_PRODUCTION_TAG = f'fm1620k5v2xt41raw_am_{XGB_PRODUCTION_FEATURE_SET}'
# TabM (config C) on the same feature set: comparison column of Table 1 and
# Table C5 (results_doppler/model_comparison/tabm/<TABM_PRODUCTION_TAG>).
TABM_PRODUCTION_TAG = f'fm1620k5v2tuneC_am_{XGB_PRODUCTION_FEATURE_SET}'

# ---------------------------------------------------------------------------
# XCO2 clear-sky anomaly (the ML training target)
# ---------------------------------------------------------------------------
# Production parameters, used by every call site that builds the primary
# target (fitting.py, results.py, tabm_eval.py). The r05/r15/r25 alternate
# reference sets vary only min_cld_dist and pass it explicitly.
ANOMALY_LAT_THRES_DEG = 0.25    # half-width of the latitude search window
ANOMALY_STD_THRES_PPM = 1.0     # max std of the clear-sky reference set
ANOMALY_MIN_CLD_DIST_KM = 10.0  # min cloud distance for a reference sounding


def anomaly_args(min_cld_dist: float = ANOMALY_MIN_CLD_DIST_KM) -> dict:
    """Fresh kwargs dict for compute_xco2_anomaly* with production values."""
    return {
        'lat_thres': ANOMALY_LAT_THRES_DEG,
        'std_thres': ANOMALY_STD_THRES_PPM,
        'min_cld_dist': min_cld_dist,
    }


# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------
# Optimizer / schedule / early-stop literals live in
# models/train_common.TrainConfig (torch-specific, one dataclass) — import
# from there, not here, so there is exactly one home for them.
