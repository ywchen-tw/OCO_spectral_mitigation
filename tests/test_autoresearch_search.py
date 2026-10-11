from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearch_search import (  # noqa: E402
    _extract_primary_metric,
    build_run_config,
    build_trial_command,
    sample_config,
)


class AutoresearchSearchTests(unittest.TestCase):
    def test_sample_config_is_deterministic(self) -> None:
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        cfg1 = sample_config("mlp_lr", rng1)
        cfg2 = sample_config("mlp_lr", rng2)
        self.assertEqual(cfg1, cfg2)
        self.assertIn("mlp.lr", cfg1)
        self.assertIn("ridge.alpha", cfg1)
        self.assertIn("pipeline.scaler", cfg1)

    def test_build_run_config_nests_sections(self) -> None:
        flat = {
            "mlp.lr": 1e-3,
            "ridge.alpha": 0.5,
            "pipeline.scaler": "robust_standard",
            "plain": 7,
        }
        nested = build_run_config("mlp_lr", flat)
        self.assertEqual(nested["mlp"]["lr"], 1e-3)
        self.assertEqual(nested["ridge"]["alpha"], 0.5)
        self.assertEqual(nested["pipeline"]["scaler"], "robust_standard")
        self.assertEqual(nested["plain"], 7)

    def test_build_trial_command_targets_mlp_script(self) -> None:
        repo_root = Path("/tmp/repo")
        cmd = build_trial_command(repo_root, "mlp_lr", "/tmp/config.json", "trial123")
        self.assertTrue(cmd[0].endswith("python") or "python" in cmd[0])
        self.assertIn(str(repo_root / "src" / "mlp_lr_models.py"), cmd)
        self.assertIn("--config", cmd)
        self.assertIn("/tmp/config.json", cmd)

    def test_extract_primary_metric_returns_nan_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metric = _extract_primary_metric(Path(tmpdir), "ft_transformer")
            self.assertTrue(math.isnan(metric))

    def test_extract_primary_metric_averages_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            csv_path = out_dir / "stratified_pc1_rmse_Transformer.csv"
            csv_path.write_text("model,pc1_quintile,n,rmse,mae\nTransformer,1,10,1.0,0.5\nTransformer,2,12,2.0,0.8\n", encoding="utf-8")
            metric = _extract_primary_metric(out_dir, "ft_transformer")
            self.assertAlmostEqual(metric, 1.5)


if __name__ == "__main__":
    unittest.main()
