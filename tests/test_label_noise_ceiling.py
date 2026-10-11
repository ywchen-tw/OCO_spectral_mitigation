from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.label_noise_ceiling import reference_stats_sorted


def reference_stats_brute_force(lat, cld, xco2, *, lat_thres, std_thres,
                                min_cld_dist):
    valid_lat = np.isfinite(lat)
    clear = valid_lat & (cld > min_cld_dist) & np.isfinite(xco2)
    anomaly = np.full(len(lat), np.nan)
    mean = np.full(len(lat), np.nan)
    std = np.full(len(lat), np.nan)
    nref = np.zeros(len(lat), dtype=int)
    for i, query_lat in enumerate(lat):
        refs = clear & (np.abs(lat - query_lat) <= lat_thres)
        values = xco2[refs]
        if valid_lat[i] and len(values) >= 5 and np.std(values) <= std_thres:
            mean[i] = np.mean(values)
            std[i] = np.std(values)
            nref[i] = len(values)
            anomaly[i] = xco2[i] - mean[i]
    return anomaly, mean, std, nref


class LabelNoiseCeilingTests(unittest.TestCase):
    def test_sorted_reference_stats_match_brute_force(self) -> None:
        rng = np.random.default_rng(42)
        lat = rng.uniform(-2.0, 2.0, 500)
        cld = rng.uniform(0.0, 30.0, 500)
        xco2 = 410.0 + rng.normal(0.0, 0.7, 500)
        lat[::47] = np.nan
        cld[::53] = np.nan
        xco2[::59] = np.nan
        kwargs = {
            "lat_thres": 0.25,
            "std_thres": 1.0,
            "min_cld_dist": 5.0,
        }

        expected = reference_stats_brute_force(lat, cld, xco2, **kwargs)
        actual = reference_stats_sorted(lat, cld, xco2, **kwargs)

        for expected_values, actual_values in zip(expected[:3], actual[:3]):
            np.testing.assert_allclose(
                actual_values, expected_values, rtol=0.0, atol=2e-12,
                equal_nan=True)
        np.testing.assert_array_equal(actual[3], expected[3])


if __name__ == "__main__":
    unittest.main()
