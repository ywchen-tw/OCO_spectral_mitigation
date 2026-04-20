from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearch_loop import LoopConfig, run_autoresearch_loop, should_stop  # noqa: E402
from autoresearch_search import TrialResult  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class AutoresearchLoopTests(unittest.TestCase):
    def _trial_result(self, status: str, metric: float | None = None, trial_id: str = "trial") -> TrialResult:
        summary = None
        if status == "success" and metric is not None:
            summary = {
                "run_id": trial_id,
                "script_name": "mlp_lr_models.py",
                "model_family": "mlp_lr",
                "commit": "abc1234",
                "status": "success",
                "primary_metric_name": "mlp_test_rmse",
                "primary_metric_value": metric,
                "secondary_metrics": {"num_params_M": 1.0},
                "peak_memory_mb": 100.0,
                "runtime_seconds": 1.0,
                "description": "",
                "artifacts": {},
                "config": {},
                "timestamp_utc": "2026-04-14T00:00:00+00:00",
            }
        return TrialResult(
            run_id=trial_id,
            family="mlp_lr",
            suffix=trial_id,
            config_path="/tmp/config.json",
            output_dir="/tmp/out",
            status=status,
            summary_path="/tmp/run_summary.json" if summary is not None else None,
            summary=summary,
        )

    def test_should_stop_on_time_budget(self) -> None:
        config = LoopConfig(max_duration_seconds=3.0, max_trials=10)
        from autoresearch_loop import LoopState

        state = LoopState(trial_index=1, consecutive_crashes=0, consecutive_non_improve=0, start_monotonic=0.0)
        reason = should_stop(elapsed_seconds=3.5, config=config, state=state)
        self.assertIsNotNone(reason)
        self.assertIn("time budget", reason)

    def test_loop_stops_on_non_improvement_streak(self) -> None:
        clock = FakeClock(0.0)
        results = [
            self._trial_result("success", metric=1.0, trial_id="t0"),
            self._trial_result("success", metric=1.2, trial_id="t1"),
            self._trial_result("success", metric=1.3, trial_id="t2"),
        ]

        def fake_runner(repo_root, family, config, suffix):  # noqa: ANN001
            clock.advance(0.5)
            return results.pop(0)

        config = LoopConfig(max_trials=5, max_duration_seconds=10.0, max_consecutive_non_improve=1, repo_root=str(Path("/tmp")))
        report = run_autoresearch_loop(config, trial_runner=fake_runner, clock=clock.now)
        self.assertIn("non-improvement streak", report.stopped_reason)
        self.assertEqual(report.state["trial_index"], 2)
        self.assertEqual(len(report.trials), 2)

    def test_loop_stops_on_time_budget(self) -> None:
        clock = FakeClock(0.0)
        results = [
            self._trial_result("success", metric=1.0, trial_id="t0"),
            self._trial_result("success", metric=0.9, trial_id="t1"),
        ]

        def fake_runner(repo_root, family, config, suffix):  # noqa: ANN001
            clock.advance(2.1)
            return results.pop(0)

        config = LoopConfig(max_trials=5, max_duration_seconds=3.0, max_consecutive_non_improve=5, repo_root=str(Path("/tmp")))
        report = run_autoresearch_loop(config, trial_runner=fake_runner, clock=clock.now)
        self.assertIn("time budget", report.stopped_reason)
        self.assertGreaterEqual(report.elapsed_seconds, 4.0)
        self.assertEqual(len(report.trials), 2)


if __name__ == "__main__":
    unittest.main()
