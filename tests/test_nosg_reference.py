"""analysis.nosg_reference: fast clear-sky reference == spectral.anomaly.

Run:  PYTHONPATH=src python -m unittest tests.test_nosg_reference -v
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import nosg_reference as R  # noqa: E402
from constants import anomaly_args  # noqa: E402

# spectral/__init__ imports pyproj; load the one module directly.
_spec = importlib.util.spec_from_file_location("anomaly", SRC_DIR / "spectral" / "anomaly.py")
anomaly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anomaly)


def _orbit(seed: int, n: int = 3000) -> dict[str, np.ndarray]:
    """One synthetic orbit: float32 like the parquet, with NaNs, ties and a
    latitude grid on which many pairs sit exactly on the 0.25 deg edge."""
    rng = np.random.default_rng(seed)
    lat = np.round(rng.uniform(-30, 30, n) * 8) / 8          # 0.125 deg steps: exact edges
    lat[rng.random(n) < 0.01] = np.nan
    cld = rng.exponential(12.0, n)
    cld[rng.random(n) < 0.02] = np.nan
    xco2 = 410 + rng.normal(0, 0.6, n) + 0.02 * np.nan_to_num(lat)
    xco2[rng.random(n) < 0.03] = np.nan
    k1 = 1.02 + rng.normal(0, 0.03, n)
    k1[rng.random(n) < 0.02] = np.nan
    gam = 0.3 + rng.normal(0, 0.01, n)
    f32 = np.float32
    return {"lat": lat.astype(f32), "cld": cld.astype(f32), "xco2": xco2.astype(f32),
            "k1": k1.astype(f32), "gam": gam.astype(f32)}


class FastReferenceEquivalenceTests(unittest.TestCase):
    def _compare(self, o: dict, min_cld_dist: float, std_thres: float | None = None):
        args = anomaly_args(min_cld_dist=min_cld_dist)
        if std_thres is not None:
            args["std_thres"] = std_thres
        ev = {"k1": o["k1"], "gam": o["gam"]}
        a0, m0, s0 = anomaly.compute_xco2_anomaly(o["lat"], o["cld"], o["xco2"],
                                                  extra_vars=ev, **args)
        a1, m1, s1 = R.clear_sky_reference(o["lat"], o["cld"], o["xco2"], ev, **args)
        np.testing.assert_array_equal(np.isfinite(a1), np.isfinite(a0))
        np.testing.assert_allclose(a1, a0, atol=5e-4, equal_nan=True)
        for k in ev:
            np.testing.assert_array_equal(np.isfinite(m1[k]), np.isfinite(m0[k]))
            np.testing.assert_allclose(m1[k], m0[k], rtol=1e-5, equal_nan=True)
            np.testing.assert_allclose(s1[k], s0[k], rtol=1e-4, atol=1e-6, equal_nan=True)
        return np.isfinite(a0).sum()

    def test_matches_production_function(self):
        for seed in range(3):
            for d in R.REFERENCE_SETS.values():
                self.assertGreater(self._compare(_orbit(seed), d), 0)

    def test_std_threshold_rejects_noisy_references(self):
        o = _orbit(7)
        n_tight = self._compare(o, 5.0, std_thres=0.55)
        n_loose = self._compare(o, 5.0)
        self.assertLess(n_tight, n_loose)

    def test_empty_pool_gives_all_nan(self):
        o = _orbit(1, 200)
        o["cld"][:] = 0.0
        a, m, s = R.clear_sky_reference(o["lat"], o["cld"], o["xco2"], {"k1": o["k1"]},
                                        **anomaly_args(min_cld_dist=5.0))
        self.assertTrue(np.isnan(a).all() and np.isnan(m["k1"]).all())


class NamingTests(unittest.TestCase):
    def test_output_columns_follow_the_stored_naming(self):
        cols = set(R.output_columns())
        for c in ("exp_o2a_intercept_nosg", "r05_o2a_k1_nosg_mean", "r15_sco2_k2_nosg_std",
                  "ref_wco2_k3_nosg_mean", "r15_exp_int_o2a_nosg_mean", "r05_exp_int_sco2_nosg_std"):
            self.assertIn(c, cols)
        self.assertEqual(R._stat_stem("exp_o2a_intercept"), "exp_int_o2a")
        self.assertEqual(len(cols), 3 + 3 * 12 * 2)


class ReadCombinedTests(unittest.TestCase):
    def test_join_and_misalignment_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "combined.parquet"
            pd.DataFrame({"fp_id": [3, 1, 2], "lat": [1.0, 2.0, 3.0]}).to_parquet(main)
            side = R.sidecar_path(main)
            pd.DataFrame({"fp_id": [3, 1, 2], "r05_o2a_k1_nosg_mean": [0.1, 0.2, 0.3]}
                         ).to_parquet(side)
            df = R.read_combined(main, ["lat", "r05_o2a_k1_nosg_mean"])
            self.assertEqual(list(df.columns), ["lat", "r05_o2a_k1_nosg_mean"])
            np.testing.assert_allclose(df["r05_o2a_k1_nosg_mean"], [0.1, 0.2, 0.3])

            pd.DataFrame({"fp_id": [1, 3, 2], "r05_o2a_k1_nosg_mean": [0.2, 0.1, 0.3]}
                         ).to_parquet(side)
            with self.assertRaisesRegex(ValueError, "row-aligned"):
                R.read_combined(main, ["lat", "r05_o2a_k1_nosg_mean"])
            with self.assertRaises(KeyError):
                R.read_combined(main, ["lat", "no_such_column"])


if __name__ == "__main__":
    unittest.main()
