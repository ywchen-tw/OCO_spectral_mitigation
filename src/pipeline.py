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

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from utils import get_storage_dir

logger = logging.getLogger(__name__)

# ── Feature definitions (identical in both training files — single source of truth) ──

_FEATURES_SFC0 = [
    'o2a_intercept', 'wco2_intercept', 'sco2_intercept',
    'o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
    'mu_sza', 'mu_vza',
    # 'sin_raa', 'cos_raa',
    # 'cos_theta',
    # 'Phi_cos_theta',
    # 'R_rs_factor',
    'cos_glint_angle',
    # 'glint_prox',
    # 'alt', 'alt_std',
    'ws',
    'log_P',
    # 'airmass',
    'dp',
    # 'dp_abp',
    # 'dp_psfc_ratio',
    # 'dpfrac',
    'h2o_scale', 'delT',
    'co2_grad_del',
    'alb_o2a',
    # 'alb_wco2', 'alb_sco2',
    # 'fs_rel_0',
    'co2_ratio_bc', 'h2o_ratio_bc',
    # 'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
    'h_cont_o2a',
    # 'h_cont_wco2',
    'h_cont_sco2',
    # 'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
    'xco2_strong_idp', 'xco2_weak_idp',
    'aod_total',
    # 'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
    # 'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate', 'dws',
    # 'dust_height', 'ice_height', 'water_height',
    # 'snr_o2a', 'snr_wco2', 'snr_sco2',
    'pol_ang_rad',
    's31', 's32',
    # 't700', 'tcwv',
]

_FEATURES_SFC1 = [
    'o2a_intercept', 'wco2_intercept', 'sco2_intercept',
    'o2a_k1', 'o2a_k2', 'wco2_k1', 'wco2_k2', 'sco2_k1', 'sco2_k2',
    'mu_sza', 'mu_vza',
    'sin_raa', 'cos_raa',
    # 'cos_theta',
    # 'Phi_cos_theta',
    # 'R_rs_factor',
    # 'cos_glint_angle',
    # 'glint_prox',
    # 'alt', 'alt_std',
    # 'ws',
    'log_P',
    # 'airmass',
    'dp',
    # 'dp_abp',
    # 'dp_psfc_ratio',
    # 'dpfrac',
    'h2o_scale', 'delT',
    'co2_grad_del',
    'alb_o2a',
    # 'alb_wco2', 'alb_sco2',
    # 'fs_rel_0',
    'co2_ratio_bc', 'h2o_ratio_bc',
    # 'csnr_o2a', 'csnr_wco2', 'csnr_sco2',
    'h_cont_o2a',
    # 'h_cont_wco2',
    'h_cont_sco2',
    # 'max_declock_o2a', 'max_declock_wco2', 'max_declock_sco2',
    'xco2_strong_idp', 'xco2_weak_idp',
    'aod_total',
    # 'aod_bc', 'aod_dust', 'aod_ice', 'aod_water',
    # 'aod_oc', 'aod_seasalt', 'aod_strataer', 'aod_sulfate', 'dws',
    # 'dust_height', 'ice_height', 'water_height',
    # 'snr_o2a', 'snr_wco2', 'snr_sco2',
    'pol_ang_rad',
    's31', 's32',
    # 't700', 'tcwv',
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
                 qt: QuantileTransformer,
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

        df = _ensure_fp_columns(df)

        qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
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
        X : np.ndarray, shape [N, n_features]
            Ready to pass to any adapter's predict().
        """
        df = _ensure_fp_columns(df)
        X_qt = self.qt.transform(df[self.qt_features].to_numpy(dtype=float))
        X_fp = df[self.fp_cols].to_numpy(dtype=float)
        return np.hstack([X_qt, X_fp])

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
        """Load a previously saved pipeline."""
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
                        help='Path to input CSV.  '
                             'Defaults to <storage_dir>/results/csv_collection/combined_2020_dates.csv')
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
                         storage_dir / 'results/csv_collection/combined_2020_dates.csv')
    base_dir  = storage_dir / 'results/model_mlp_lr'
    out_dir   = base_dir / args.suffix if args.suffix else base_dir
    out_path  = _resolve(args.out, out_dir / 'pipeline.pkl')

    print(f"Loading data: {data_path}", flush=True)
    df = pd.read_csv(data_path)
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
