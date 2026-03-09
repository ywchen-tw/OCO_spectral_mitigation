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

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from utils import get_storage_dir

logger = logging.getLogger(__name__)

# When pipeline.py is executed directly (as __main__), register it under the
# canonical module name so pickle can resolve ClipFreeQuantileTransformer via
# 'pipeline.ClipFreeQuantileTransformer' on both save and load.
if __name__ == '__main__':
    sys.modules.setdefault('pipeline', sys.modules['__main__'])


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


# ── Feature definitions (identical in both training files — single source of truth) ──

_FEATURES_SFC0 = [
    # 'xco2_raw_minus_apriori',
    # 'xco2_bc_minus_raw',
    # 'xco2_raw_minus-xco2_strong_idp_minus',
    # 'airmass_sq',
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
    'o2a_k2_over_k1',
    'wco2_k2_over_k1',
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
    # 't700',
    'tcwv',
]

_FEATURES_SFC1 = [
    # 'xco2_raw_minus_apriori',
    # 'xco2_bc_minus_raw',
    # 'airmass_sq',
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
    'o2a_k2_over_k1',
    'wco2_k2_over_k1',
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
    # 't700', 
    'tcwv',
]

_FEATURE_MAP = {0: _FEATURES_SFC0, 1: _FEATURES_SFC1}
_FP_COLS     = [f'fp_{i}' for i in range(8)]


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
        logger.debug("_ensure_derived_features: computed '%s' from base columns", col)
    logger.info(
        "_ensure_derived_features: computed %d missing derived column(s): %s",
        len(missing), missing,
    )
    return df


class FeaturePipeline:
    """Shared feature pipeline for all XCO2 bias-correction models.

    Encapsulates:
    - Feature selection (sfc_type-specific continuous features)
    - QuantileTransformer fitting on continuous features
    - fp_{0..7} one-hot encoding appended raw (not QT-transformed)

    The ``features`` attribute contains the full ordered list used as model
    input: ``qt_features + fp_cols``.
    """

    def __init__(self,
                 sfc_type: int,
                 qt: 'QuantileTransformer | ClipFreeQuantileTransformer | RobustStandardScaler',
                 qt_features: list,
                 fp_cols: list,
                 features: list):
        self.sfc_type   = sfc_type
        self.qt         = qt
        self.qt_features = qt_features
        self.fp_cols    = fp_cols
        self.features   = features   # qt_features + fp_cols

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, df: pd.DataFrame, sfc_type: int = 1) -> 'FeaturePipeline':
        """Fit a new pipeline on df.

        df must contain all feature columns for the given sfc_type plus a
        'fp' column (integer 0–7) for footprint one-hot encoding.
        """
        if sfc_type not in _FEATURE_MAP:
            raise ValueError(f"sfc_type must be 0 or 1, got {sfc_type}")

        qt_features = list(_FEATURE_MAP[sfc_type])   # continuous features only
        fp_cols     = list(_FP_COLS)
        features    = qt_features + fp_cols

        df = _ensure_derived_features(df)
        df = _ensure_fp_columns(df)

        qt = RobustStandardScaler()
        qt.fit(df[qt_features].to_numpy(dtype=float))

        logger.info(
            "FeaturePipeline fitted: sfc_type=%d, %d qt_features + 8 fp cols = %d total",
            sfc_type, len(qt_features), len(features),
        )
        return cls(sfc_type=sfc_type, qt=qt,
                   qt_features=qt_features, fp_cols=fp_cols, features=features)

    # ── Transformation ────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply QT to continuous features then append raw fp one-hots.

        Returns
        -------
        X : np.ndarray, shape [N, n_features], dtype float32
            Ready to pass to any adapter's predict().
        """
        df = _ensure_derived_features(df)
        df = _ensure_fp_columns(df)
        # Extract as float32 — halves memory vs float64 (scalers handle float32 natively)
        X_qt_raw = df[self.qt_features].to_numpy(dtype=np.float32)
        X_qt     = self.qt.transform(X_qt_raw)
        del X_qt_raw   # free before concatenation
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
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FeaturePipeline, got {type(obj)}")
        logger.info("FeaturePipeline loaded ← %s  (sfc_type=%d, %d features)",
                    path, obj.sfc_type, obj.n_features)
        return obj

    def __repr__(self) -> str:
        return (f"FeaturePipeline(sfc_type={self.sfc_type}, "
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
                        help='Subfolder under results/model_mlp_lr/ used to derive the default '
                             '--out path.  Ignored when --out is supplied explicitly.')
    parser.add_argument('--out',      default=None,
                        help='Output path for pipeline.pkl.  '
                             'Defaults to <storage_dir>/results/model_mlp_lr/<suffix>/pipeline.pkl')
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
    base_dir  = storage_dir / 'results/model_mlp_lr'
    out_dir   = base_dir / args.suffix if args.suffix else base_dir
    out_path  = _resolve(args.out, out_dir / 'pipeline.pkl')

    print(f"Loading data: {data_path}", flush=True)
    df = pd.read_parquet(data_path) if str(data_path).endswith('.parquet') else pd.read_csv(data_path)
    print(f"  Rows before filtering: {len(df):,}", flush=True)

    df = df[df['sfc_type'] == args.sfc_type]
    df = df[df['snow_flag'] == 0]
    print(f"  Rows after sfc_type={args.sfc_type} + snow_flag==0: {len(df):,}", flush=True)

    pipeline = FeaturePipeline.fit(df, sfc_type=args.sfc_type)
    print(f"  {pipeline}", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(out_path)
    print(f"Saved pipeline → {out_path}", flush=True)


if __name__ == '__main__':
    main()
