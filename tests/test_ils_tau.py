from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ILS_TAU_PATH = Path(__file__).resolve().parents[1] / "src" / "abs_util" / "ils_tau.py"
spec = importlib.util.spec_from_file_location("ils_tau", ILS_TAU_PATH)
ils_tau = importlib.util.module_from_spec(spec)
sys.modules["ils_tau"] = ils_tau
spec.loader.exec_module(ils_tau)

convolve_optical_depth_in_transmission = ils_tau.convolve_optical_depth_in_transmission


class IlsTauTests(unittest.TestCase):
    def test_weak_absorption_matches_weighted_mean_tau(self) -> None:
        ext_window = np.array([[0.001], [0.002], [0.003]])
        dz = np.array([1.0])
        weights = np.array([1.0, 2.0, 1.0])
        tau_eff = convolve_optical_depth_in_transmission(ext_window, dz, weights, 1.0)
        tau_mean = np.dot(ext_window[:, 0], weights) / weights.sum()
        self.assertAlmostEqual(tau_eff, tau_mean, places=6)

    def test_saturated_absorption_convolves_transmission_not_tau(self) -> None:
        ext_window = np.array([[0.0], [10.0]])
        dz = np.array([1.0])
        weights = np.array([1.0, 1.0])
        tau_eff = convolve_optical_depth_in_transmission(ext_window, dz, weights, 1.0)
        expected = -math.log((1.0 + math.exp(-10.0)) / 2.0)
        self.assertAlmostEqual(tau_eff, expected)
        self.assertLess(tau_eff, 1.0)

    def test_rayleigh_tau_is_inside_transmission_convolution(self) -> None:
        ext_window = np.array([[0.0], [10.0]])
        dz = np.array([1.0])
        weights = np.array([1.0, 1.0])
        rayleigh_tau = np.array([0.1, 0.2])
        tau_eff = convolve_optical_depth_in_transmission(
            ext_window,
            dz,
            weights,
            1.0,
            rayleigh_tau_window=rayleigh_tau,
        )
        expected = -math.log((math.exp(-0.1) + math.exp(-10.2)) / 2.0)
        self.assertAlmostEqual(tau_eff, expected)

    def test_opaque_window_stays_finite(self) -> None:
        ext_window = np.array([[1000.0], [1000.0]])
        dz = np.array([1.0])
        weights = np.array([1.0, 1.0])
        tau_eff = convolve_optical_depth_in_transmission(ext_window, dz, weights, 1.0)
        self.assertTrue(np.isfinite(tau_eff))


if __name__ == "__main__":
    unittest.main()
