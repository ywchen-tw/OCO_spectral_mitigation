from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearch_confirm import ConfirmationConfig, confirm_and_promote  # noqa: E402
from autoresearch_search import TrialResult  # noqa: E402
from experiment_tracking import RunSummary  # noqa: E402


class AutoresearchConfirmTests(unittest.TestCase):
    def _summary(self, run_id: str, metric: float, runtime: float = 10.0, memory: float = 1000.0) -> RunSummary:
        return RunSummary(
            run_id=run_id,
            script_name="mlp_lr_models.py",
            model_family="mlp_lr",
            commit="abc1234",
            status="success",
            primary_metric_name="mlp_test_rmse",
            primary_metric_value=metric,
            secondary_metrics={"num_params_M": 1.0},
            peak_memory_mb=memory,
            runtime_seconds=runtime,
            description="baseline",
            artifacts={"model_dir": "/tmp/model"},
            config={"mlp": {"lr": 1e-3}},
        )

    def _trial(self, suffix: str, summary: RunSummary) -> TrialResult:
        return TrialResult(
            run_id=suffix,
            family="mlp_lr",
            suffix=suffix,
            config_path="/tmp/config.json",
            output_dir="/tmp/out",
            status="success",
            summary_path="/tmp/run_summary.json",
            summary=summary.to_dict(),
        )

    def test_promotes_better_stable_candidate(self) -> None:
        candidate = self._summary("cand", 0.9500, runtime=9.0, memory=900.0)
        incumbent = self._summary("inc", 0.9800, runtime=10.0, memory=1000.0)

        candidate_trials = [self._trial(f"candidate_{i}", self._summary(f"cand_{i}", 0.9500 + i * 0.0001, runtime=9.0, memory=900.0)) for i in range(3)]
        incumbent_trials = [self._trial(f"incumbent_{i}", self._summary(f"inc_{i}", 0.9800 + i * 0.0001, runtime=10.0, memory=1000.0)) for i in range(3)]
        trial_queue = candidate_trials + incumbent_trials

        def fake_runner(repo_root, family, config, suffix):  # noqa: ANN001
            return trial_queue.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfirmationConfig(repo_root=tmpdir, repeats=3, max_primary_metric_std=0.01)
            snapshot = confirm_and_promote(candidate, incumbent, config, trial_runner=fake_runner)

        self.assertEqual(snapshot.decision, "keep")
        self.assertTrue(snapshot.candidate_guardrails_ok)
        self.assertTrue(snapshot.incumbent_guardrails_ok)
        self.assertIn("improved", snapshot.reason)

    def test_discards_unstable_candidate(self) -> None:
        candidate = self._summary("cand", 0.9500, runtime=9.0, memory=900.0)
        incumbent = self._summary("inc", 0.9800, runtime=10.0, memory=1000.0)

        candidate_trials = [
            self._trial("candidate_0", self._summary("cand_0", 0.9400, runtime=9.0, memory=900.0)),
            self._trial("candidate_1", self._summary("cand_1", 1.0400, runtime=9.0, memory=900.0)),
            self._trial("candidate_2", self._summary("cand_2", 0.9600, runtime=9.0, memory=900.0)),
        ]
        incumbent_trials = [self._trial(f"incumbent_{i}", self._summary(f"inc_{i}", 0.9800, runtime=10.0, memory=1000.0)) for i in range(3)]
        trial_queue = candidate_trials + incumbent_trials

        def fake_runner(repo_root, family, config, suffix):  # noqa: ANN001
            return trial_queue.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfirmationConfig(repo_root=tmpdir, repeats=3, max_primary_metric_std=0.001)
            snapshot = confirm_and_promote(candidate, incumbent, config, trial_runner=fake_runner)

        self.assertEqual(snapshot.decision, "discard")
        self.assertFalse(snapshot.candidate_guardrails_ok)


if __name__ == "__main__":
    unittest.main()
