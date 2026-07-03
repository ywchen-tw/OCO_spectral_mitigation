"""FeaturePipeline — shared QT fitting and feature transformation.

Encapsulates the feature-engineering steps that were previously duplicated
in training_data_load_preselect() (mlp_lr_models.py) and training_data_load()
(models_transformer.py).  Both files produced identical feature lists and
identical fp_ one-hot append logic; this class is the single source of truth.

Typical usage (training):
    pipeline = FeaturePipeline.fit(df, sfc_type=1)
    pipeline.save("results/exp_v1/pipeline.pkl")

    X_all    = pipeline.transform(df)
    valid    = ~df["xco2_bc_anomaly"].isna()
    X, y     = X_all[valid.values], df["xco2_bc_anomaly"][valid].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

Typical usage (inference on new data):
    pipeline = FeaturePipeline.load("results/exp_v1/pipeline.pkl")
    X = pipeline.transform(new_df)
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Optional

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from utils import get_storage_dir

logger = logging.getLogger(__name__)

# The scaler/transformer classes below force ``__module__ = 'pipeline'`` so that
# pickles are portable across however this file gets imported.  For that to work
# at *save* and *load* time, a module named ``pipeline`` must exist in
# sys.modules.  Register this module under that canonical name regardless of how
# it was imported:
#   - directly as __main__  (python src/models/pipeline.py)
#   - as models.pipeline    (python -m models.tabm, which fits+saves inline)
# setdefault avoids clobbering a real top-level ``pipeline`` module when
# src/models is already on sys.path (legacy CURC launch layout).
if __name__ == '__main__':
    # Force-overwrite: pipeline.py may have already been imported as models.pipeline
    # (via __init__.py → tabm → pipeline) before Python re-executes it as __main__,
    # leaving a stale entry. setdefault would silently keep the old module, causing
    # the two class objects to diverge and breaking pickle.
    sys.modules['pipeline'] = sys.modules['__main__']
else:
    sys.modules.setdefault('pipeline', sys.modules[__name__])


class ClipFreeQuantileTransformer:
    """QuantileTransformer with linear extrapolation beyond training support.

    sklearn 1.x removed the ``clip`` parameter.  This wrapper replicates the
    old ``clip=False`` behaviour: values outside the fitted quantile boundaries
    are extrapolated linearly using the local slope at the nearest boundary,
    rather than being clamped to the boundary value.

    All ``**qt_kwargs`` are forwarded to ``QuantileTransformer``.
    The wrapper is fully picklable and exposes the same ``fit`` / ``transform``
    interface so it can be stored in ``FeaturePipeline.qt`` unchanged.
    """

    # Force pickle to always record this as 'pipeline.ClipFreeQuantileTransformer'
    # so that loading works regardless of whether pipeline.py was run as __main__.
    __module__ = 'pipeline'

    def __init__(self, **qt_kwargs):
        self._qt = QuantileTransformer(**qt_kwargs)

    def fit(self, X: np.ndarray) -> 'ClipFreeQuantileTransformer':
        X = np.asarray(X, dtype=float)
        self._qt.fit(X)

        # quantiles_ shape: [n_quantiles, n_features]
        quants = self._qt.quantiles_

        # Transformed boundary values (these ARE clipped by the inner QT,
        # but they are the correct boundary output values).
        self._x_lo = quants[0].copy()     # [n_features] — training min quantile
        self._x_hi = quants[-1].copy()    # [n_features] — training max quantile
        self._y_lo = self._qt.transform(quants[0:1])[0].copy()   # [n_features]
        self._y_hi = self._qt.transform(quants[-1:])[0].copy()   # [n_features]

        # Local slope at each boundary estimated from the adjacent quantile pair.
        eps = 1e-10
        dx_lo = quants[1] - quants[0]
        dy_lo = self._qt.transform(quants[1:2])[0] - self._y_lo
        self._slope_lo = np.where(np.abs(dx_lo) > eps, dy_lo / dx_lo, 0.0)

        dx_hi = quants[-1] - quants[-2]
        dy_hi = self._y_hi - self._qt.transform(quants[-2:-1])[0]
        self._slope_hi = np.where(np.abs(dx_hi) > eps, dy_hi / dx_hi, 0.0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Preserve float32 input — only upcast non-floating types (sklearn QT supports float32 natively)
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        Xt = self._qt.transform(X)   # inner QT clips OOD values to boundary

        # Overwrite clipped boundary values with linear extrapolation per feature.
        # Vectorised: one broadcast per feature boundary instead of a Python loop.
        lo_mask = X < self._x_lo[np.newaxis, :]   # [N, F]
        hi_mask = X > self._x_hi[np.newaxis, :]   # [N, F]
        if lo_mask.any():
            Xt = np.where(lo_mask,
                          self._y_lo + self._slope_lo * (X - self._x_lo),
                          Xt)
        if hi_mask.any():
            Xt = np.where(hi_mask,
                          self._y_hi + self._slope_hi * (X - self._x_hi),
                          Xt)
        return Xt

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # Delegate attribute look-ups to the inner QT so callers that access
    # qt.quantiles_, qt.references_, etc. still work transparently.
    # Guard against _qt itself to avoid infinite recursion during unpickling
    # (pickle uses __new__ without __init__, so _qt may not exist yet).
    def __getattr__(self, name):
        if name == '_qt':
            raise AttributeError('_qt')
        return getattr(self._qt, name)

    def __repr__(self) -> str:
        return f"ClipFreeQuantileTransformer(inner={self._qt!r})"


class RobustStandardScaler:
    """Two-stage scaler: RobustScaler (median/IQR) then StandardScaler.

    RobustScaler never clips OOD values — features outside the training IQR
    are scaled linearly rather than clamped, avoiding the attention-blindness
    caused by QuantileTransformer boundary saturation.

    StandardScaler (fitted on RobustScaler output) re-equalises feature
    variances so that high-IQR features do not dominate attention dot-products.

    Fully picklable; exposes the same fit / transform interface as
    QuantileTransformer so it can be stored in FeaturePipeline.qt unchanged.
    """

    __module__ = 'pipeline'

    def __init__(self):
        self._robust = RobustScaler()
        self._std    = StandardScaler()

    def fit(self, X: np.ndarray) -> 'RobustStandardScaler':
        X = np.asarray(X, dtype=float)
        self._std.fit(self._robust.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Preserve float32 — only upcast non-floating types (sklearn scalers support float32 natively)
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        return self._std.transform(self._robust.transform(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return 'RobustStandardScaler()'


class PCAWhitening:
    """RobustScaler → PCA(n_components, whiten=True).

    Output dimensionality = n_components (≤ n_qt_features).
    Intended for MLP / Ridge / Logistic — models that treat input as a flat
    vector and benefit from decorrelated, unit-variance features.
    NOT suitable for FT-Transformer (destroys per-feature token identity).
    """

    __module__ = 'pipeline'

    def __init__(self, n_components=0.90):
        self._n_components = n_components
        self._robust = RobustScaler()
        self._pca    = PCA(n_components=n_components, whiten=True, random_state=42)

    def fit(self, X: np.ndarray) -> 'PCAWhitening':
        X = np.asarray(X, dtype=float)
        self._pca.fit(self._robust.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        return self._pca.transform(self._robust.transform(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_components_(self) -> int:
        return self._pca.n_components_

    @property
    def explained_variance_ratio_(self):
        return self._pca.explained_variance_ratio_

    def __repr__(self) -> str:
        return f'PCAWhitening(n_components={self._n_components})'


class PCAScoreAppender:
    """Appends selected PCA score columns to the pipeline output.

    After fitting, FeaturePipeline.transform() produces:
      [qt_scaled_features | selected_pc_scores | fp_onehot]

    Selected PCs are chosen per sfc_type based on their correlation with
    cld_dist_km (from the PCA analysis):
      sfc_type=1 (land):  PC1, PC4, PC8  (r = +0.36, +0.22, −0.21)
      sfc_type=0 (ocean): PC3, PC6       (r = +0.20, −0.22)
    """

    __module__ = 'pipeline'

    # 0-indexed PC columns to append, keyed by sfc_type
    _DEFAULT_PC_IDX: dict = {
        1: [0, 3, 7],   # land:  PC1, PC4, PC8
        0: [2, 5],      # ocean: PC3, PC6
    }

    def __init__(self, pc_idx=None):
        self._pca      = PCA(n_components=8, random_state=42, svd_solver='randomized')
        self._robust   = RobustScaler()
        self._pc_idx   = pc_idx    # None → use _DEFAULT_PC_IDX[sfc_type]
        self._sfc_type = None

    def _resolve_pc_idx(self, sfc_type: int) -> list:
        if self._pc_idx is not None:
            return list(self._pc_idx)
        return list(self._DEFAULT_PC_IDX.get(sfc_type, self._DEFAULT_PC_IDX[1]))

    def fit(self, X_scaled: np.ndarray, sfc_type: int) -> 'PCAScoreAppender':
        """Fit PCA on already-QT-scaled features."""
        self._sfc_type = sfc_type
        X = np.asarray(X_scaled, dtype=float)
        X_r = self._robust.fit_transform(X)
        n_comp = min(8, X_r.shape[0], X_r.shape[1])
        if n_comp != self._pca.n_components:
            self._pca = PCA(n_components=n_comp, random_state=42, svd_solver='randomized')
        self._pca.fit(X_r)
        return self

    def transform_append(self, X_scaled: np.ndarray, sfc_type: int) -> np.ndarray:
        """Return [X_scaled | selected_pc_scores], shape [N, n_qt + len(pc_idx)]."""
        X = np.asarray(X_scaled)
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        scores   = self._pca.transform(self._robust.transform(X)).astype(np.float32)  # [N, 8]
        selected = scores[:, self._resolve_pc_idx(sfc_type)]                           # [N, k]
        return np.concatenate([X_scaled.astype(np.float32), selected], axis=1)

    @property
    def pc_names(self) -> list:
        """Names of the appended PC columns, e.g. ['pca_pc1', 'pca_pc4', 'pca_pc8']."""
        sfc = self._sfc_type if self._sfc_type is not None else 1
        return [f'pca_pc{i + 1}' for i in self._resolve_pc_idx(sfc)]

    def __repr__(self) -> str:
        return f'PCAScoreAppender(pc_idx={self._pc_idx})'


# ── Feature definitions (identical in both training files — single source of truth) ──

_FEATURES_SFC0 = [
    'xco2_raw_minus_apriori',
    # 'xco2_bc_minus_raw',
    # 'xco2_raw_minus-xco2_strong_idp_minus',
    # 'airmass_sq',
    'fp_area_km2',
    # 'alb_o2a_over_cos_sza', 
    # 'alb_wco2_over_cos_sza', 'alb_sco2_over_cos_sza',
    # 'o2a_intercept', 
    # 'wco2_intercept', 'sco2_intercept',
    'exp_o2a_intercept', 
    # 'exp_wco2_intercept', 
    # 'exp_sco2_intercept',
    'o2a_exp_intercept-alb',
    # 'wco2_exp_intercept- alb',
    # 'sco2_exp_intercept-alb',
    'o2a_k1', 'o2a_k2', #'o2a_k3',
    'wco2_k1', 'wco2_k2', 'wco2_k3',
    'sco2_k1', 'sco2_k2', #'sco2_k3',
    # 'o2a_k2_over_k1',
    # 'wco2_k2_over_k1',
    # 'sco2_k2_over_k1',
    # '1_over_cos_sza', '1_over_cos_vza',
    # 'mu_sza', 'mu_vza',
    # 'sin_raa', 'cos_raa',
    # 'cos_theta',
    # 'Phi_cos_theta',
    # 'R_rs_factor',
    'cos_glint_angle',
    # 'glint_prox',
    # 'alt', 'alt_std',
    # 'ws',
    'log_P',
    # 'airmass',
    # 'dp',
    # 'dp_abp',
    # 'dp_psfc_prior',
    'dp_psfc_prior_ratio',
    # 'dpfrac',
    'h2o_scale', 'delT',
    'co2_grad_del',
    # 'alb_o2a',
    # 'alb_wco2', 'alb_sco2',
    # 'fs_rel_0',
    # 'co2_ratio_bc',   # SHAP≈0.001 — near-zero, dropped
    'h2o_ratio_bc',
    # 'csnr_o2a', 'csnr_wco2',
    'csnr_sco2',
    # 'h_cont_o2a',
    # 'h_cont_wco2',
    # 'h_cont_sco2',
    # 'max_declock_o2a', 'max_declock_wco2',
    # 'max_declock_sco2',
    # 'xco2_strong_idp',
    # 'xco2_weak_idp',
    # 'aod_total',      # SHAP=0.006 — redundant with individual components below
    # 'aod_bc',         # SHAP=0.000 — zero importance
    'aod_dust', 
    # 'aod_ice', 
    # 'aod_water',
    'aod_oc', 'aod_seasalt', 
    'aod_strataer', 
    'aod_sulfate',
    # 'dws',
    # 'dust_height', 'ice_height', 'water_height',
    # 'snr_o2a', 'snr_wco2', 'snr_sco2',
    'pol_ang_rad',
    's31',
    # 's32',
    'tcwv',
    # ── cloud/aerosol contamination (merged from CONTAM_FEATURES; ocean) ──
    'max_declock_wco2', 'aod_water', 'dp_abp', 'h_cont_wco2', 't700',
    'water_height', 'alb_sco2_over_wco2',
]

_FEATURES_SFC1 = [
    'xco2_raw_minus_apriori',
    # 'xco2_bc_minus_raw',
    # 'airmass_sq',
    'fp_area_km2',
    # 'alb_o2a_over_cos_sza', 
    # 'alb_wco2_over_cos_sza', 'alb_sco2_over_cos_sza',
    # 'o2a_intercept', 
    # 'wco2_intercept', 
    # 'sco2_intercept',
   'exp_o2a_intercept', 
    # 'exp_wco2_intercept', 
    # 'exp_sco2_intercept',
    'o2a_exp_intercept-alb',
    'wco2_exp_intercept-alb',
    # 'sco2_exp_intercept-alb',
    'o2a_k1', 'o2a_k2', #'o2a_k3',
    'wco2_k1', 'wco2_k2', 'wco2_k3',
    'sco2_k1', 'sco2_k2', #'sco2_k3',
    # 'o2a_k2_over_k1',
    # 'wco2_k2_over_k1',
    # 'sco2_k2_over_k1',
    '1_over_cos_sza', '1_over_cos_vza',
    # 'mu_sza', 'mu_vza',
    'sin_raa', 
    # 'cos_raa',
    # 'cos_theta',
    # 'Phi_cos_theta',
    # 'R_rs_factor',
    # 'cos_glint_angle',
    # 'glint_prox',
    # 'alt', 'alt_std',
    # 'ws',
    'log_P',
    # 'airmass',
    # 'dp',
    # 'dp_abp',
    # 'dp_psfc_ratio',
    'dp_psfc_prior_ratio',
    # 'dpfrac',
    'h2o_scale', 'delT',
    'co2_grad_del',
    # 'alb_o2a',
    # 'alb_wco2', 'alb_sco2',
    # 'fs_rel_0',
    'co2_ratio_bc', 'h2o_ratio_bc',
    'csnr_o2a', 
    # 'csnr_wco2', 'csnr_sco2',
    'h_cont_o2a',
    # 'h_cont_wco2',
    'h_cont_sco2',
    # 'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
    # 'xco2_strong_idp', 
    # 'xco2_weak_idp',
    # 'aod_total',
    # 'aod_bc', 
    'aod_dust', 
    # 'aod_ice', 'aod_water',
    'aod_oc', 'aod_seasalt',
    'aod_strataer', 'aod_sulfate', 
    # 'dws',
    # 'dust_height', 'ice_height', 'water_height',
    # 'snr_o2a', 'snr_wco2', 'snr_sco2',
    'pol_ang_rad',
    's31',
    # 's32',
    'tcwv',
    # ── cloud/aerosol contamination (merged from CONTAM_FEATURES; land) ──
    'dpfrac', 'fs_rel_0', 'dust_height', 'aod_ice', 'ice_height', 'dp_abp',
    'water_height', 'aod_water', 't700', 'h_cont_wco2', 'alt_std',
    'alb_sco2_over_wco2',
]

_FEATURE_MAP = {0: _FEATURES_SFC0, 1: _FEATURES_SFC1}
_FP_COLS     = [f'fp_{i}' for i in range(8)]

# ── Feature-set ablations (see src/models/TABM_PLAN.md "Feature set ablations") ──
# Named groups of continuous features that can be dropped from _FEATURE_MAP[sfc_type]
# before fitting.  Dropping is applied identically across all models (TabM,
# FT-Transformer, GBDT baselines) so the active set is explicit and reproducible.
# Names are listed for both surface types; any name absent from the active
# sfc_type's list is silently ignored.

# Set 2 — drop xco2-derived features (retrieval-model bias that may leak target info)
XCO2_FEATURES = frozenset([
    'xco2_raw_minus_apriori',
])

# Set 3 — drop k1/k2/k3 spectroscopic coefficients and exp_intercept / exp_intercept-alb
SPEC_FEATURES = frozenset([
    'exp_o2a_intercept',
    'o2a_exp_intercept-alb',
    'wco2_exp_intercept-alb',     # sfc_type=1 only
    'o2a_k1', 'o2a_k2',
    'wco2_k1', 'wco2_k2', 'wco2_k3',
    'sco2_k1', 'sco2_k2',
])

# Cloud/aerosol contamination features are part of the base _FEATURES_SFC0/1 above
# (always active) — forward-selected from the disabled-feature pool and validated with
# date-blocked (date_kfold) CV: ~+0.06 (ocean) / +0.07 (land) held-out R² over the
# pre-contam base (declocking/water aerosol on ocean; dpfrac/humidity/dust+ice on land).
# They are permanent base features, so there is no separate CONTAM_FEATURES group and no
# contam ablation.

# snow_flag is NOT a model feature.  The land A/B (test_snow_features.py) showed it is
# neutral — ~0 ΔR²/ΔRMSE even with snow footprints present — so it is not added to any
# feature set.  The snow footprints themselves (the DATA) are kept by default in the
# trainers (opt out with --exclude-snow); snow_flag is only used there as a filter column.

# Maps --feature_set name → spec.  The resolver (_resolve_feature_set) supports a
# 'drop' set (remove features), plus 'add'/'add_per_sfc' appends for future use;
# every current set is drop-only (or the ``None`` sentinel = base unchanged).
# Appends, if present, run BEFORE drops so a drop can remove an appended feature.
#
# Contamination features are part of the base _FEATURES_SFC0/1, so the base IS the
# active feature set: 'full' == base (``None`` sentinel = unchanged).  The
# no_xco2 / no_spec / no_xco2_and_spec ablations drop the named group from full.
#
# PROFILE BLOCK (profile EOFs + tropopause) is ORTHOGONAL to these sets — it is
# not a raw column so it lives outside _FEATURE_SETS, supplied via
# FeaturePipeline.fit(..., profile_pca=...).  When supplied it is appended to
# EVERY set and never dropped, so `full`+profile_pca is the "new full" (active
# features + profile EOFs + tropopause) and no_xco2/no_spec/no_xco2_and_spec
# remove only the xco2 / spectroscopy raw features from that new full.
_FEATURE_SETS: dict = {
    'full':             None,
    'no_xco2':          {'drop': XCO2_FEATURES},
    'no_spec':          {'drop': SPEC_FEATURES},
    'no_xco2_and_spec': {'drop': XCO2_FEATURES | SPEC_FEATURES},
}


def _resolve_feature_set(qt_features: list, feature_set: str,
                         sfc_type: 'int | None' = None) -> list:
    """Return qt_features with the named ablation applied (order preserved).

    A spec may combine appends and a drop.  Appends run first: 'add' appends the
    same features for both surfaces, 'add_per_sfc' selects the append list by
    ``sfc_type`` (appended columns must exist in the dataframe at fit time).  A
    'drop' set then removes features (including any just appended).  ``None`` is a
    sentinel leaving the base list unchanged.
    """
    if feature_set not in _FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {sorted(_FEATURE_SETS)}, got {feature_set!r}"
        )
    spec = _FEATURE_SETS[feature_set]
    if spec is None:                      # sentinel — base list unchanged
        return list(qt_features)
    result = list(qt_features)
    if 'add_per_sfc' in spec:             # per-surface append
        add_src = spec['add_per_sfc'].get(sfc_type, []) if sfc_type is not None else []
        result += [f for f in add_src if f not in result]
    if 'add' in spec:                     # append (both surfaces)
        result += [f for f in spec['add'] if f not in result]
    if 'drop' in spec:                    # remove named group(s)
        drop = spec['drop']
        result = [f for f in result if f not in drop]
    return result

# Features with heavy right tails (orders-of-magnitude spread) that benefit
# from log1p compression before the scaler. Covers all possible AOD components
# plus footprint area; inactive/commented-out features are harmless to list here.
_LOG1P_FEATURES = frozenset([
    'fp_area_km2',
    'aod_total', 'aod_bc',
    'aod_dust', 'aod_ice', 'aod_water',
    'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate',
])


def _apply_log1p(X: np.ndarray, idx: list) -> np.ndarray:
    """log1p-transform selected column indices in-place (clamps negatives to 0)."""
    if not idx:
        return X
    X = X.copy()
    X[:, idx] = np.log1p(np.clip(X[:, idx], 0.0, None))
    return X


def _ensure_fp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fp_{0..7} one-hot columns from df['fp'] if not already present."""
    if all(c in df.columns for c in _FP_COLS):
        return df
    df = df.copy()
    for i in range(8):
        df[f'fp_{i}'] = (df['fp'] == i).astype(int)
    return df


# Derived features that older CSVs may be missing: maps column name → (formula, base cols).
# Computed lazily so stale CSVs regenerated from raw base columns still work.
_DERIVED_FEATURES = {
    'xco2_raw_minus_apriori': ('xco2_raw - xco2_apriori',  ['xco2_raw', 'xco2_apriori']),
    'xco2_raw_minus-xco2_strong_idp_minus': ('xco2_raw - xco2_strong_idp_minus', ['xco2_raw', 'xco2_strong_idp_minus']),
    'airmass_sq':             ('airmass ** 2',              ['airmass']),
    'alb_o2a_over_cos_sza':   ('alb_o2a / cos(sza)',        ['alb_o2a', 'sza']),
    'alb_wco2_over_cos_sza':  ('alb_wco2 / cos(sza)',       ['alb_wco2', 'sza']),
    'alb_sco2_over_cos_sza':  ('alb_sco2 / cos(sza)',       ['alb_sco2', 'sza']),
    'alb_sco2_over_wco2':     ('alb_sco2 / alb_wco2',       ['alb_sco2', 'alb_wco2']),
}


def _ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute any missing derived feature columns from their base columns.

    Handles CSVs generated before these engineered features were added to
    fitting_data_correction.py.  A copy is made only if at least one column
    needs to be added.

    Raises
    ------
    ValueError
        If a derived column is missing AND one or more of its required base
        columns are also absent.  This means the CSV is too old to recover
        from and must be regenerated via fitting_data_correction.py.
    """
    missing = [c for c in _DERIVED_FEATURES if c not in df.columns]
    if not missing:
        return df

    # Check that all base columns exist before touching df.
    absent_base: dict[str, list[str]] = {}
    for col in missing:
        _, base_cols = _DERIVED_FEATURES[col]
        bad = [b for b in base_cols if b not in df.columns]
        if bad:
            absent_base[col] = bad
    if absent_base:
        lines = "\n".join(
            f"  {col!r} needs base column(s): {bases}"
            for col, bases in absent_base.items()
        )
        raise ValueError(
            f"Cannot derive {len(absent_base)} feature(s) — required base columns are "
            f"missing from the CSV.\n{lines}\n\n"
            "The CSV was generated before these columns existed.  "
            "Re-run fitting_data_correction.py for all dates and rebuild the "
            "combined CSV (raw_processing_multipe_dates) to fix this."
        )

    df = df.copy()
    cos_sza = None  # compute once if needed by alb_*_over_cos_sza features
    for col in missing:
        if col == 'xco2_raw_minus_apriori':
            df[col] = df['xco2_raw'] - df['xco2_apriori']
        elif col == 'airmass_sq':
            df[col] = df['airmass'] ** 2
        elif col in ('alb_o2a_over_cos_sza', 'alb_wco2_over_cos_sza', 'alb_sco2_over_cos_sza'):
            if cos_sza is None:
                cos_sza = np.cos(np.radians(df['sza'].to_numpy(dtype=float)))
            band = col.replace('_over_cos_sza', '')   # 'alb_o2a' / 'alb_wco2' / 'alb_sco2'
            df[col] = df[band].to_numpy(dtype=float) / cos_sza
        elif col == 'alb_sco2_over_wco2':
            df[col] = df['alb_sco2'].to_numpy(dtype=float) / df['alb_wco2'].to_numpy(dtype=float)
        logger.debug("_ensure_derived_features: computed '%s' from base columns", col)
    logger.info(
        "_ensure_derived_features: computed %d missing derived column(s): %s",
        len(missing), missing,
    )
    return df


# Soundings with |xco2_bc_anomaly| beyond this (ppm) are not physical retrieval
# error — they are quality-flag escapees / fill-value leakage and are NOT the
# target we model.  A handful of them dominate squared error (RMSE blows up,
# R² collapses to ~0) while leaving MAE almost untouched — exactly the land
# signature.  Drop them from the RAW dataframe BEFORE the train/held split so
# every model family (and both splits) sees the identical, consistent target.
MAX_ABS_ANOMALY_PPM = 100.0

# Dedicated location for the per-surface ProfilePCA transformers
# (profile_pca_ocean.pkl / profile_pca_land.pkl), relative to get_storage_dir().
# Kept separate from any model output subfolder so the profile-EOF block is a
# shared, model-agnostic artifact.
PROFILE_PCA_DIR = 'results/profile_pca'


# ── Regression target selection ───────────────────────────────────────────────
# The model target is the clear-sky XCO2 anomaly.  Two clear-sky reference sets
# exist in the combined parquet (produced by spectral/fitting.py): the default
# 10 km set and the stricter 15 km set (xco2_bc_anomaly_r15).  Trainers resolve
# the column through resolve_target_col() so the choice lives here, not hardcoded
# per model.  Pass a short name ('10km'/'15km') or an explicit column name.
DEFAULT_TARGET_COL = 'xco2_bc_anomaly'          # 10 km clear-sky reference
TARGET_COLS = {
    '10km': 'xco2_bc_anomaly',
    '15km': 'xco2_bc_anomaly_r15',
}


def resolve_target_col(target: Optional[str] = None) -> str:
    """Map a short name ('10km'/'15km') or explicit column to the target column.

    ``None``/empty → DEFAULT_TARGET_COL.  An unrecognised string is returned
    verbatim so callers may pass any column present in the parquet.
    """
    if not target:
        return DEFAULT_TARGET_COL
    return TARGET_COLS.get(target, target)


def filter_target_outliers(df: pd.DataFrame, max_abs_ppm: float = MAX_ABS_ANOMALY_PPM,
                           target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """Drop rows whose target magnitude exceeds ``max_abs_ppm``.

    NaN targets are kept here (the per-model ``isfinite`` mask handles those);
    this only removes finite-but-non-physical extremes.  Returns ``df`` unchanged
    (no copy) when nothing is dropped.
    """
    if target_col not in df.columns:
        return df
    y = df[target_col].to_numpy(dtype=float)
    drop = np.isfinite(y) & (np.abs(y) > max_abs_ppm)
    n_drop = int(drop.sum())
    if n_drop == 0:
        return df
    logger.info(
        "filter_target_outliers: dropped %d/%d rows with |%s| > %g ppm",
        n_drop, len(df), target_col, max_abs_ppm,
    )
    return df[~drop]


def _compute_anomaly_group(idx_list, fp_lat, cld_dist_km, xco2,
                           lat_thres, std_thres, min_cld_dist, chunk_size):
    """Compute XCO2 anomaly for a single (date, orbit_id) group.

    Top-level function so joblib can pickle it for parallel dispatch.

    Returns
    -------
    (idx, anomaly_vals) : indices into the original array and their anomaly values
    """
    idx    = np.array(idx_list)
    g_lat  = fp_lat[idx]
    g_dist = cld_dist_km[idx]
    g_xco2 = xco2[idx]
    M      = len(idx)

    valid_lat  = ~np.isnan(g_lat)
    clear_mask = valid_lat & (g_dist > min_cld_dist)

    ref_lat  = np.where(clear_mask, g_lat,  np.nan)
    ref_xco2 = np.where(clear_mask, g_xco2, np.nan)

    g_ref_mean = np.full(M, np.nan)
    g_ref_std  = np.full(M, np.nan)

    for start in range(0, M, chunk_size):
        end   = min(start + chunk_size, M)
        q_lat = g_lat[start:end]

        lat_diff  = np.abs(q_lat[:, None] - ref_lat[None, :])
        in_window = lat_diff <= lat_thres
        xco2_win  = np.where(in_window, ref_xco2[None, :], np.nan)
        has_refs  = in_window.any(axis=1)

        c_mean = np.full(end - start, np.nan)
        c_std  = np.full(end - start, np.nan)
        if has_refs.any():
            c_mean[has_refs] = np.nanmean(xco2_win[has_refs], axis=1)
            c_std[has_refs]  = np.nanstd( xco2_win[has_refs], axis=1)

        g_ref_mean[start:end] = c_mean
        g_ref_std[start:end]  = c_std
        del lat_diff, in_window, xco2_win

    valid = valid_lat & ~np.isnan(g_ref_mean) & (g_ref_std <= std_thres)
    return idx[valid], g_xco2[valid] - g_ref_mean[valid]


def compute_xco2_anomaly_date_id(fp_date, fp_orbit_id, fp_lat, cld_dist_km, xco2,
                         lat_thres=0.5, std_thres=2.0, min_cld_dist=10.0,
                         chunk_size=512, n_jobs=-1):
    """XCO2 anomaly relative to nearby clear-sky soundings within the same orbit.

    Groups footprints by (date, orbit_id) and processes each group
    independently, so the pairwise broadcast is O(M²) per orbit rather than
    O(N²) over the full dataset.  Within each group every footprint already
    shares the same date and orbit_id, so no cross-group comparisons are made.

    Parameters
    ----------
    fp_date      : [N] footprint date (string or any equality-comparable type)
    fp_orbit_id  : [N] orbit ID (int, str, or float)
    fp_lat       : [N] footprint latitudes (may contain NaN)
    cld_dist_km  : [N] nearest-cloud distance in km (may contain NaN)
    xco2         : [N] XCO2 values (may contain NaN)
    lat_thres    : float, half-width of latitude search window (degrees)
    std_thres    : float, maximum allowed std of reference XCO2 (ppm)
    min_cld_dist : float, minimum cloud distance for clear-sky reference (km)
    chunk_size   : int, query rows per iteration within each group.
                   Each group typically has O(1 000) rows, so the default 512
                   usually means 1-2 iterations per group with small arrays.
    n_jobs       : int, number of parallel workers (default: -1 = all cores).

    Returns
    -------
    anomaly : [N] float array, NaN where reference is unavailable or noisy
    """
    from joblib import Parallel, delayed

    fp_date     = np.asarray(fp_date)
    fp_orbit_id = np.asarray(fp_orbit_id)
    fp_lat      = np.asarray(fp_lat,      dtype=float)
    xco2        = np.asarray(xco2,        dtype=float)
    cld_dist_km = np.asarray(cld_dist_km, dtype=float)

    chunk_size = int(chunk_size)
    anomaly    = np.full(len(fp_lat), np.nan)

    # Build index groups: {(date, orbit_id): [row indices]}
    groups: dict = {}
    for i, key in enumerate(zip(fp_date, fp_orbit_id)):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_compute_anomaly_group)(
            idx_list, fp_lat, cld_dist_km, xco2,
            lat_thres, std_thres, min_cld_dist, chunk_size
        )
        for idx_list in groups.values()
    )

    for valid_idx, valid_vals in results:
        anomaly[valid_idx] = valid_vals

    return anomaly


def _resolve_profile_pca(spec, sfc_type: int):
    """Resolve the ``profile_pca`` argument to a fitted ProfilePCA or None.

    Accepts a ProfilePCA instance, a path (str/Path), ``True`` (load the default
    per-surface pkl <storage>/results/profile_pca/profile_pca_<surface>.pkl), or
    None/False (no profile block).
    """
    if spec is None or spec is False:
        return None
    from .profile_pca import ProfilePCA
    if isinstance(spec, ProfilePCA):
        return spec
    if spec is True:
        from utils import get_storage_dir
        surf = 'ocean' if sfc_type == 0 else 'land'
        spec = get_storage_dir() / PROFILE_PCA_DIR / f'profile_pca_{surf}.pkl'
    return ProfilePCA.load(spec)


class FeaturePipeline:
    """Shared feature pipeline for all XCO2 bias-correction models.

    Encapsulates:
    - Feature selection (sfc_type-specific continuous features)
    - QuantileTransformer fitting on continuous features
    - Optional profile-EOF / tropopause block (ProfilePCA), standardized and
      appended after the scaled features (see ``profile_pca`` in ``fit``)
    - fp_{0..7} one-hot encoding appended raw (not QT-transformed)

    The ``features`` attribute contains the full ordered list used as model
    input: ``qt_features [+ pc_names] [+ profile_names] + fp_cols``.
    """

    def __init__(self,
                 sfc_type: int,
                 qt: 'QuantileTransformer | ClipFreeQuantileTransformer | RobustStandardScaler | PCAWhitening',
                 qt_features: list,
                 fp_cols: list,
                 features: list,
                 *,
                 scaler_type: str = 'robust_standard',
                 pca_appender: 'PCAScoreAppender | None' = None,
                 pc1_col_idx: 'int | None' = None,
                 log1p_cols: 'list | None' = None,
                 feature_set: str = 'full',
                 profile_pca=None,
                 profile_scaler=None,
                 profile_names: 'list | None' = None):
        self.sfc_type    = sfc_type
        self.qt          = qt
        self.qt_features = qt_features
        self.fp_cols     = fp_cols
        self.features    = features      # full model input feature names
        self.scaler_type = scaler_type
        self.pca_appender  = pca_appender
        self.pc1_col_idx   = pc1_col_idx
        self.log1p_cols    = log1p_cols or []   # qt_feature names pre-transformed by log1p
        self.feature_set   = feature_set        # named ablation set ('full', 'no_xco2', 'no_spec')
        # Profile-EOF/tropopause block (ProfilePCA + its own StandardScaler); None
        # when profiles are disabled.  Orthogonal to feature_set — always carried
        # through, never dropped by the no_xco2/no_spec ablations.
        self.profile_pca    = profile_pca
        self.profile_scaler = profile_scaler
        self.profile_names  = profile_names or []

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, df: pd.DataFrame, sfc_type: int = 1,
            scaler: str = 'robust_standard',
            pca_augment: bool = False,
            feature_set: str = 'full',
            profile_pca=None) -> 'FeaturePipeline':
        """Fit a new pipeline on df.

        Parameters
        ----------
        df : DataFrame containing all feature columns for sfc_type plus 'fp'.
        sfc_type : 0=ocean, 1=land.
        scaler : 'robust_standard' (default) or 'pca_whitening'.
            'pca_whitening' chains RobustScaler → PCA(whiten=True); output
            dimensionality is reduced to explain 90% variance.  Not suitable
            for FT-Transformer (destroys per-feature token identity).
        pca_augment : if True, append selected PC scores after the scaled
            features (land: PC1/PC4/PC8; ocean: PC3/PC6).  Compatible with
            all model types; for FT-Transformer this is the only PCA mode.
        feature_set : named ablation set selecting which continuous features
            survive.  'full' (default) is the base, which now includes the
            per-surface contamination features (snow_flag is NOT a feature).
            'no_xco2' / 'no_spec' / 'no_xco2_and_spec' start from full and drop
            xco2_raw_minus_apriori / the k1-k3 + exp_intercept spectroscopy
            group / both.  See _FEATURE_SETS.  The
            resulting ``n_features`` / ``features`` remain the single
            authoritative source — callers must never hard-code feature counts.
        profile_pca : profile-EOF/tropopause block, ORTHOGONAL to feature_set.
            A ProfilePCA instance, a path to a saved per-surface pkl, ``True``
            (load the default <storage>/results/profile_pca/profile_pca_<surface>.pkl),
            or None to disable.  When supplied, the block's score columns
            (t_pc01…, q_pc01…, co2prior_pc01…, tropopause_sigma, tropopause_temp)
            are standardized by their own StandardScaler and appended after the
            scaled features — and are carried through EVERY feature set: the
            no_xco2/no_spec ablations drop only qt raw features, never profiles.
            So `full`+profile_pca is the "new full" (active features + profile
            EOFs + tropopause); `no_xco2`+profile_pca removes only the xco2 term
            from it; etc.  NOTE: rows whose profiles are NaN (e.g. the 2016-2020
            mid-month dates built before fitting.py emitted profiles) become
            NaN in the block and are dropped downstream by each trainer's
            finite-feature filter — regenerate those dates first for full coverage.

        IMPORTANT (leakage discipline): for blocked-split validation, fit on the
        train split only.  Call ``split_dataframe(df, mode=...)`` first, then
        ``FeaturePipeline.fit(train_df, ...)``; never fit on the full dataset
        before splitting (leaks scaler/quantile statistics into the held-out set).
        """
        if sfc_type not in _FEATURE_MAP:
            raise ValueError(f"sfc_type must be 0 or 1, got {sfc_type}")
        if scaler not in ('robust_standard', 'pca_whitening'):
            raise ValueError(
                f"scaler must be 'robust_standard' or 'pca_whitening', got {scaler!r}"
            )

        # Continuous features for this sfc_type, with the named ablation applied.
        qt_features = _resolve_feature_set(_FEATURE_MAP[sfc_type], feature_set, sfc_type)
        fp_cols     = list(_FP_COLS)

        df = _ensure_derived_features(df)
        df = _ensure_fp_columns(df)

        log1p_cols = [f for f in qt_features if f in _LOG1P_FEATURES]
        log1p_idx  = [qt_features.index(f) for f in log1p_cols]

        X_qt_raw = df[qt_features].to_numpy(dtype=float)
        X_qt_raw = _apply_log1p(X_qt_raw, log1p_idx)

        if scaler == 'pca_whitening':
            qt = PCAWhitening()
            qt.fit(X_qt_raw)
            n_pca    = qt.n_components_
            features = [f'pca_pc{i + 1}' for i in range(n_pca)] + fp_cols
        else:
            qt = RobustStandardScaler()
            qt.fit(X_qt_raw)
            features = qt_features + fp_cols

        # Optionally fit PCAScoreAppender and insert PC score columns between
        # the scaled block and fp_onehot.
        pca_appender = None
        pc1_col_idx  = None
        if pca_augment:
            X_scaled     = qt.transform(X_qt_raw).astype(np.float32)
            pca_appender = PCAScoreAppender()
            pca_appender.fit(X_scaled, sfc_type)
            pc_names    = pca_appender.pc_names
            # Insert PC names between scaled block and fp_cols
            features    = [f for f in features if f not in fp_cols] + pc_names + fp_cols
            # pc1_col_idx: column index of the first appended PC in transform() output
            pc1_col_idx = len(features) - len(fp_cols) - len(pc_names)

        # Optional profile-EOF/tropopause block: standardized and appended between
        # the scaled/PC block and the fp one-hots.  Carried through all feature sets.
        profile_obj    = _resolve_profile_pca(profile_pca, sfc_type)
        profile_scaler = None
        profile_names  = []
        if profile_obj is not None:
            Xp = profile_obj.transform(df)                          # [N, k], may hold NaN
            finite = np.isfinite(Xp).all(axis=1)
            if finite.sum() < 2:
                raise ValueError(
                    "profile_pca block has <2 finite rows — the profile/tropopause "
                    "columns are (nearly) all NaN for this data (older fitting.py?).")
            profile_scaler = StandardScaler().fit(Xp[finite])
            profile_names  = list(profile_obj.feature_names)
            features       = [f for f in features if f not in fp_cols] + profile_names + fp_cols

        logger.info(
            "FeaturePipeline fitted: sfc_type=%d, scaler=%s, pca_augment=%s, "
            "feature_set=%s, %d qt_features + %d profile + %d fp cols = %d total",
            sfc_type, scaler, pca_augment, feature_set,
            len(qt_features), len(profile_names), len(fp_cols), len(features),
        )
        logger.info(
            "FeaturePipeline log1p pre-transform: %s", log1p_cols or "none"
        )
        if profile_names:
            logger.info("FeaturePipeline profile block (%d cols): %s",
                        len(profile_names), profile_names)
        return cls(sfc_type=sfc_type, qt=qt,
                   qt_features=qt_features, fp_cols=fp_cols, features=features,
                   scaler_type=scaler, pca_appender=pca_appender, pc1_col_idx=pc1_col_idx,
                   log1p_cols=log1p_cols, feature_set=feature_set,
                   profile_pca=profile_obj, profile_scaler=profile_scaler,
                   profile_names=profile_names)

    # ── Transformation ────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply scaler to continuous features, optionally append PC scores, then fp one-hots.

        Returns
        -------
        X : np.ndarray, shape [N, n_features], dtype float32
            Ready to pass to any adapter's predict().
        """
        df = _ensure_derived_features(df)
        df = _ensure_fp_columns(df)
        # Extract as float32 — halves memory vs float64 (scalers handle float32 natively)
        X_qt_raw = df[self.qt_features].to_numpy(dtype=np.float32)
        # log1p pre-transform for skewed AOD / area features (backward compat: old
        # pickles without log1p_cols skip this step transparently)
        log1p_cols = getattr(self, 'log1p_cols', [])
        if log1p_cols:
            log1p_idx = [self.qt_features.index(f) for f in log1p_cols]
            X_qt_raw  = _apply_log1p(X_qt_raw, log1p_idx)
        X_qt     = self.qt.transform(X_qt_raw).astype(np.float32)
        del X_qt_raw   # free before concatenation
        # Append PC scores if appender present (backward compat: old pickles lack this attr)
        pca_appender = getattr(self, 'pca_appender', None)
        if pca_appender is not None:
            X_qt = pca_appender.transform_append(X_qt, self.sfc_type)
        # Append the standardized profile-EOF/tropopause block, if present
        # (backward compat: old pickles lack these attrs).  Order matches fit:
        # [scaled qt | pc scores | profile block | fp one-hots].
        profile_pca = getattr(self, 'profile_pca', None)
        if profile_pca is not None:
            Xp = profile_pca.transform(df).astype(np.float32)
            Xp = self.profile_scaler.transform(Xp).astype(np.float32)
            X_qt = np.concatenate([X_qt, Xp], axis=1)
        X_fp = df[self.fp_cols].to_numpy(dtype=np.float32)
        result = np.concatenate([X_qt, X_fp], axis=1)
        del X_qt       # free before return
        return result

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> list:
        """Alias for self.features — full ordered feature list."""
        return self.features

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Pickle this pipeline to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info("FeaturePipeline saved → %s", path)

    @classmethod
    def load(cls, path) -> 'FeaturePipeline':
        """Load a previously saved pipeline.

        Relative paths are resolved against ``get_storage_dir()`` so that
        callers on CURC (where storage_dir ≠ CWD) find the correct file.
        """
        path = Path(path)
        if not path.is_absolute():
            path = get_storage_dir() / path
        # A pipeline may embed a ProfilePCA whose helper classes pin
        # __module__='profile_pca'.  Import profile_pca.py first so that module
        # name is registered in sys.modules, else pickle.load raises
        # ModuleNotFoundError when profile_pca hasn't been imported yet.
        try:
            from . import profile_pca as _pp  # noqa: F401
        except Exception:
            try:
                import profile_pca as _pp     # noqa: F401
            except Exception:
                pass
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FeaturePipeline, got {type(obj)}")
        logger.info("FeaturePipeline loaded ← %s  (sfc_type=%d, %d features)",
                    path, obj.sfc_type, obj.n_features)
        return obj

    def __repr__(self) -> str:
        scaler_type = getattr(self, 'scaler_type', 'robust_standard')
        pca_augment = getattr(self, 'pca_appender', None) is not None
        log1p_cols  = getattr(self, 'log1p_cols', [])
        feature_set = getattr(self, 'feature_set', 'full')
        n_profile   = len(getattr(self, 'profile_names', []))
        return (f"FeaturePipeline(sfc_type={self.sfc_type}, "
                f"scaler={scaler_type!r}, pca_augment={pca_augment}, "
                f"feature_set={feature_set!r}, profile={n_profile or 'off'}, "
                f"log1p={log1p_cols or 'none'}, "
                f"n_qt_features={len(self.qt_features)}, "
                f"n_features={self.n_features})")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit and save a FeaturePipeline from a labelled CSV."
    )
    parser.add_argument('--data',     default=None,
                        help='Path to input file (.parquet or .csv).  '
                             'Defaults to <storage_dir>/results/csv_collection/combined_2020_dates.parquet')
    parser.add_argument('--sfc-type', type=int, default=1, choices=[0, 1],
                        help='Surface type filter: 0=ocean, 1=land (default: 1)')
    parser.add_argument('--suffix',   type=str, default='',
                        help='Subfolder under results/feature_pipeline/ used to derive the default '
                             '--out path.  Ignored when --out is supplied explicitly.')
    parser.add_argument('--out',      default=None,
                        help='Output path for pipeline.pkl.  '
                             'Defaults to <storage_dir>/results/feature_pipeline/<suffix>/pipeline.pkl')
    parser.add_argument('--scaler',   default='robust_standard',
                        choices=['robust_standard', 'pca_whitening'],
                        help='Scaler type: robust_standard (default) or pca_whitening '
                             '(RobustScaler → PCA(whiten=True)).  Not for FT-Transformer.')
    parser.add_argument('--pca-augment', action='store_true',
                        help='Append selected PC scores after scaled features '
                             '(land: PC1/PC4/PC8; ocean: PC3/PC6).')
    parser.add_argument('--feature-set', default='full',
                        choices=sorted(_FEATURE_SETS),
                        help="Feature ablation set. 'full' (default) = base "
                             "(includes per-surface contamination; snow_flag is not a "
                             "feature). 'no_xco2'/'no_spec'/'no_xco2_and_spec' drop "
                             "xco2_raw_minus_apriori / k1-k3+exp_intercept / both "
                             "from full.")
    parser.add_argument('--profile-pca', dest='profile_pca', nargs='?', const='auto', default=None,
                        help='Append the profile-EOF + tropopause block (ProfilePCA). Bare flag / '
                             '"auto" loads results/profile_pca/profile_pca_<surface>.pkl; or pass a '
                             '.pkl path. Carried through every feature set (no_xco2/no_spec keep it).')
    parser.add_argument('--exclude-snow', dest='exclude_snow', action='store_true',
                        help='Filter OUT snow/ice footprints (snow_flag==1). Default: KEEP snow.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    storage_dir = get_storage_dir()

    def _resolve(arg, default):
        """Return arg resolved against storage_dir if relative, else default."""
        if arg is None:
            return str(default)
        p = Path(arg)
        return str(p if p.is_absolute() else storage_dir / p)

    data_path = _resolve(args.data,
                         storage_dir / 'results/csv_collection/combined_2020_dates.parquet')
    base_dir  = storage_dir / 'results/feature_pipeline'
    out_dir   = base_dir / args.suffix if args.suffix else base_dir
    out_path  = _resolve(args.out, out_dir / 'pipeline.pkl')

    print(f"Loading data: {data_path}", flush=True)
    df = pd.read_parquet(data_path) if str(data_path).endswith('.parquet') else pd.read_csv(data_path)
    print(f"  Rows before filtering: {len(df):,}", flush=True)

    df = df[df['sfc_type'] == args.sfc_type]
    # Snow footprints KEPT by default (--exclude-snow to drop them).
    if args.exclude_snow:
        df = df[df['snow_flag'] == 0]
    print(f"  Rows after sfc_type={args.sfc_type} filter "
          f"(snow {'excluded' if args.exclude_snow else 'KEPT'}): {len(df):,}", flush=True)

    _prof = True if args.profile_pca == 'auto' else args.profile_pca
    pipeline = FeaturePipeline.fit(df, sfc_type=args.sfc_type,
                                   scaler=args.scaler,
                                   pca_augment=args.pca_augment,
                                   feature_set=args.feature_set,
                                   profile_pca=_prof)
    print(f"  {pipeline}", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(out_path)
    print(f"Saved pipeline → {out_path}", flush=True)


if __name__ == '__main__':
    main()
