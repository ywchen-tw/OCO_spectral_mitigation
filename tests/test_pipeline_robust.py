"""FeaturePipeline fill/spike masking, RobustClipScaler and the _nosg gamma predictors.

Run:  PYTHONPATH=src python -m unittest tests.test_pipeline_robust -v
"""
from __future__ import annotations

import pickle
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ``models.__init__`` imports TabM, whose progress-bar dependency is optional.
try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__spec__ = ModuleSpec("tqdm", loader=None)
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from models import pipeline as P  # noqa: E402

LAND = 1
N = 6000
FILL = -999999.0


def _land_frame(seed: int = 0) -> pd.DataFrame:
    """Synthetic land frame with every 'full' predictor, valid values only.

    The _nosg gamma predictors are NOT provided: only their base columns, so
    every fit exercises the derivation.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"fp": rng.integers(0, 8, N)})
    for name in P._FEATURE_MAP[LAND]:
        if name in P._DERIVED_FEATURES and name.endswith("_nosg"):
            continue
        df[name] = rng.normal(5.0, 1.0, N)
    df["o2a_intercept_nosg"] = rng.normal(-2.2, 0.3, N)
    df["wco2_intercept_nosg"] = rng.normal(-2.2, 0.3, N)
    df["alb_o2a"] = rng.normal(0.12, 0.03, N)
    df["alb_wco2"] = rng.normal(0.11, 0.03, N)
    # Predictors with a valid range get values inside it.
    df["alb_sco2_over_wco2"] = rng.normal(0.9, 0.1, N)
    df["dpfrac"] = rng.normal(0.0, 2.0, N)
    df["dp_abp"] = rng.normal(0.0, 5.0, N)
    df["h_cont_sco2"] = rng.lognormal(1.7, 0.6, N)
    df["fp_area_km2"] = rng.normal(1.5, 0.2, N)
    df["alt_std"] = rng.gamma(1.5, 10.0, N)        # roughness in m, IQR ~ 15 m
    return df


def _with_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inject the fill and spike types seen in the real parquet."""
    df = df.copy()
    df.loc[:199, "alt_std"] = FILL
    df.loc[200:209, "h_cont_sco2"] = 5.0e6
    df.loc[210:229, "dp_abp"] = 1.1e4
    df.loc[230:249, "dpfrac"] = 4.0e3
    df.loc[250:259, "fp_area_km2"] = 8.0e4
    df.loc[260:264, "alb_sco2_over_wco2"] = -1295.0
    return df


def _qt_block(pipe: P.FeaturePipeline, X: np.ndarray) -> np.ndarray:
    return X[:, :len(pipe.qt_features)]


class GammaNosgDerivationTests(unittest.TestCase):
    def test_gamma_nosg_columns_follow_the_smoothed_formulas(self):
        df = P._ensure_derived_features(_land_frame())
        np.testing.assert_allclose(df["exp_o2a_intercept_nosg"],
                                   np.exp(df["o2a_intercept_nosg"]))
        np.testing.assert_allclose(df["o2a_exp_intercept-alb_nosg"],
                                   np.exp(df["o2a_intercept_nosg"]) - df["alb_o2a"])
        np.testing.assert_allclose(df["wco2_exp_intercept-alb_nosg"],
                                   np.exp(df["wco2_intercept_nosg"]) - df["alb_wco2"])

    def test_gamma_predictors_use_the_unsmoothed_fit(self):
        gamma = [f for f in P._FEATURE_MAP[LAND] if "intercept" in f]
        self.assertTrue(gamma)
        self.assertTrue(all(f.endswith("_nosg") for f in gamma), gamma)
        self.assertNotIn("t700", P._FEATURE_MAP[0] + P._FEATURE_MAP[LAND])

    def test_underivable_feature_is_skipped_not_raised(self):
        df = _land_frame().drop(columns=["o2a_intercept_nosg"])
        out = P._ensure_derived_features(df)       # must not raise
        self.assertNotIn("exp_o2a_intercept_nosg", out.columns)
        with self.assertRaisesRegex(KeyError, "o2a_intercept_nosg"):
            P.FeaturePipeline.fit(df, sfc_type=LAND)


class MaskingAndImputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = _land_frame()
        cls.dirty = _with_bad_values(cls.clean)
        cls.pipe = P.FeaturePipeline.fit(cls.dirty, sfc_type=LAND)

    def _col(self, name: str) -> int:
        return self.pipe.qt_features.index(name)

    def test_impute_value_is_median_of_valid_training_values(self):
        valid = self.dirty["alt_std"][self.dirty["alt_std"] > P._FILL_THRESHOLD]
        self.assertAlmostEqual(self.pipe.impute_values[self._col("alt_std")],
                               float(np.median(valid)), places=6)

    def test_bad_values_map_to_the_imputed_median(self):
        X = _qt_block(self.pipe, self.pipe.transform(self.dirty))
        for name, row in [("alt_std", 0), ("h_cont_sco2", 200), ("dp_abp", 210),
                          ("dpfrac", 230), ("fp_area_km2", 250),
                          ("alb_sco2_over_wco2", 260)]:
            probe = self.dirty.iloc[[row]].copy()
            probe[name] = self.pipe.impute_values[self._col(name)]
            expected = _qt_block(self.pipe, self.pipe.transform(probe))[0, self._col(name)]
            self.assertAlmostEqual(X[row, self._col(name)], expected, places=5, msg=name)

    def test_existing_nan_is_not_imputed(self):
        probe = self.clean.iloc[:3].copy()
        probe.loc[probe.index[1], "csnr_o2a"] = np.nan
        X = _qt_block(self.pipe, self.pipe.transform(probe))
        self.assertTrue(np.isnan(X[1, self._col("csnr_o2a")]))
        self.assertTrue(np.isfinite(np.delete(X[1], self._col("csnr_o2a"))).all())

    def test_scaled_predictors_have_sd_of_order_one(self):
        X = _qt_block(self.pipe, self.pipe.transform(self.dirty))
        sd = X.std(axis=0)
        bad = {f: round(float(s), 3) for f, s in zip(self.pipe.qt_features, sd)
               if not 0.3 <= s <= 3.0}
        self.assertEqual(bad, {})

    def test_alt_std_keeps_its_resolution(self):
        probe = self.clean.iloc[:2].copy()
        probe["alt_std"] = [5.0, 50.0]
        z = _qt_block(self.pipe, self.pipe.transform(probe))[:, self._col("alt_std")]
        self.assertGreater(z[1] - z[0], 1.0)

    def test_out_of_support_values_are_clipped(self):
        probe = self.clean.iloc[:1].copy()
        probe["csnr_o2a"] = 1.0e9                  # no valid range: reaches the scaler
        z = _qt_block(self.pipe, self.pipe.transform(probe))[0, self._col("csnr_o2a")]
        self.assertEqual(z, self.pipe.qt.clip)

    def test_pickle_round_trip_keeps_the_rules(self):
        clone = pickle.loads(pickle.dumps(self.pipe))
        np.testing.assert_array_equal(clone.transform(self.dirty),
                                      self.pipe.transform(self.dirty))


class LegacyPickleTests(unittest.TestCase):
    def test_pre_fix_pipeline_transform_is_unchanged(self):
        dirty = _with_bad_values(_land_frame())
        pipe = P.FeaturePipeline.fit(dirty, sfc_type=LAND, scaler="robust_standard")
        # A pickle written before 2026-10-14 has no masking attributes.
        del pipe.impute_values, pipe.valid_range
        legacy = pickle.loads(pickle.dumps(pipe))

        frame = P._ensure_derived_features(dirty)
        X_raw = frame[legacy.qt_features].to_numpy(dtype=np.float32)
        idx = [legacy.qt_features.index(f) for f in legacy.log1p_cols]
        expected = legacy.qt.transform(P._apply_log1p(X_raw, idx)).astype(np.float32)
        np.testing.assert_array_equal(_qt_block(legacy, legacy.transform(dirty)), expected)


class ContaminationSubgroupTests(unittest.TestCase):
    # The 2026-07-25 contamination group, before the 2026-10-14 split by origin.
    V1_CONTAM = {
        'aod_water', 'aod_ice', 'aod_dust', 'aod_oc', 'aod_seasalt', 'aod_sulfate',
        'aod_strataer', 'water_height', 'ice_height', 'dust_height',
        'h_cont_o2a', 'h_cont_wco2', 'h_cont_sco2', 'csnr_o2a', 'csnr_sco2',
        'max_declock_wco2', 'dp_abp', 'co2_ratio_bc', 'h2o_ratio_bc',
    }

    def test_split_keeps_the_v1_membership(self):
        self.assertEqual(set(P.CONTAM_FEATURES), self.V1_CONTAM)

    def test_subgroups_are_disjoint(self):
        subs = (P.CG_FULLPHYS_FEATURES, P.CG_INHOM_FEATURES, P.CG_SCREEN_FEATURES)
        self.assertEqual(sum(len(s) for s in subs), len(P.CONTAM_FEATURES))

    def test_drop_arms_remove_exactly_one_subgroup(self):
        for name, sub in (("no_fpcloud", P.CG_FULLPHYS_FEATURES),
                          ("no_inhom", P.CG_INHOM_FEATURES),
                          ("no_screen", P.CG_SCREEN_FEATURES)):
            for sfc in (0, LAND):
                base = P._FEATURE_MAP[sfc]
                kept = P._resolve_feature_set(base, name, sfc)
                self.assertEqual(set(base) - set(kept), set(base) & sub, (name, sfc))


class RobustClipScalerTests(unittest.TestCase):
    def test_zero_iqr_falls_back_to_a_wider_quantile_range(self):
        rng = np.random.default_rng(1)
        x = np.where(rng.random(20000) < 0.85, 0.0, rng.gamma(2.0, 3.0, 20000))
        X = np.c_[x, np.full_like(x, 7.0)]         # zero-inflated and constant
        sc = P.RobustClipScaler().fit(X)
        q10, q90 = np.percentile(x, [10, 90])
        self.assertAlmostEqual(sc.scale_[0], q90 - q10)
        self.assertEqual(sc.scale_[1], 1.0)
        out = sc.transform(np.array([[0.5, 7.0], [5.0, 7.0]]))
        self.assertGreater(out[1, 0] - out[0, 0], 0.1)
        np.testing.assert_array_equal(out[:, 1], [0.0, 0.0])

    def test_spikes_do_not_set_the_scale(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.0, 1.0, 50000)
        spiked = x.copy()
        spiked[:10] = 5.0e6
        a = P.RobustClipScaler().fit(x[:, None])
        b = P.RobustClipScaler().fit(spiked[:, None])
        self.assertAlmostEqual(b.sd_[0] / a.sd_[0], 1.0, delta=0.01)

    def test_float32_input_stays_float32_and_nan_passes(self):
        X = np.random.default_rng(3).normal(size=(1000, 2)).astype(np.float32)
        sc = P.RobustClipScaler().fit(X)
        X[0, 0] = np.nan
        out = sc.transform(X)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isnan(out[0, 0]))


if __name__ == "__main__":
    unittest.main()
