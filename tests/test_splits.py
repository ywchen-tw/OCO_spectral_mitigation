from __future__ import annotations

import math
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

# ``models.__init__`` imports TabM, whose progress-bar dependency is optional
# in minimal test environments and irrelevant for split-only tests.
try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__spec__ = ModuleSpec("tqdm", loader=None)
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from models.splits import split_dataframe, split_date_kfold_train_calib_test  # noqa: E402


def _date_values(frame: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(frame["date"]).unique().tolist())


class DateKfoldTrainCalibTestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n_dates = 116
        self.n_folds = 5
        self.calib_frac = 0.15
        dates = pd.date_range("2016-01-01", periods=self.n_dates, freq="15D")
        self.frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "value": np.arange(self.n_dates),
            }
        )

    def test_preserves_date_kfold_held_blocks_and_calib_ratio(self) -> None:
        all_dates = set(_date_values(self.frame))
        calib_sets: list[set[pd.Timestamp]] = []

        for fold in range(self.n_folds):
            proper_df, calib_df, held_df = split_date_kfold_train_calib_test(
                self.frame,
                n_folds=self.n_folds,
                fold=fold,
                calib_frac=self.calib_frac,
            )
            _, expected_held_df = split_dataframe(
                self.frame,
                mode="date_kfold",
                n_folds=self.n_folds,
                fold=fold,
            )

            proper_dates = set(_date_values(proper_df))
            calib_dates = set(_date_values(calib_df))
            held_dates = set(_date_values(held_df))
            expected_held_dates = set(_date_values(expected_held_df))

            self.assertEqual(held_dates, expected_held_dates)
            self.assertFalse(proper_dates & calib_dates)
            self.assertFalse(proper_dates & held_dates)
            self.assertFalse(calib_dates & held_dates)
            self.assertEqual(proper_dates | calib_dates | held_dates, all_dates)

            expected_calib_dates = math.ceil(
                self.calib_frac * (self.n_dates - len(held_dates))
            )
            self.assertEqual(len(calib_dates), expected_calib_dates)
            calib_sets.append(calib_dates)

        self.assertEqual(len({frozenset(dates) for dates in calib_sets}), self.n_folds)

    def test_default_calibration_block_is_adjacent_to_held_block(self) -> None:
        sorted_dates = _date_values(self.frame)
        date_index = {date: idx for idx, date in enumerate(sorted_dates)}

        for fold in range(self.n_folds):
            _, calib_df, held_df = split_date_kfold_train_calib_test(
                self.frame,
                n_folds=self.n_folds,
                fold=fold,
                calib_frac=self.calib_frac,
            )
            calib_idx = sorted(date_index[date] for date in _date_values(calib_df))
            held_idx = sorted(date_index[date] for date in _date_values(held_df))

            self.assertEqual(calib_idx, list(range(calib_idx[0], calib_idx[-1] + 1)))
            self.assertTrue(
                calib_idx[0] == held_idx[-1] + 1
                or calib_idx[-1] == held_idx[0] - 1
            )


if __name__ == "__main__":
    unittest.main()
