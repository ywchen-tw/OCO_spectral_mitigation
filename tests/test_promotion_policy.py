from __future__ import annotations

import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_tracking import RunSummary  # noqa: E402
from promotion_policy import decide_promotion  # noqa: E402


class PromotionPolicyTests(unittest.TestCase):
    def _summary(self, value: float, status: str = "success") -> RunSummary:
        return RunSummary(
            run_id="run",
            script_name="train.py",
            model_family="mlp_lr",
            commit="abc1234",
            status=status,
            primary_metric_name="mlp_test_rmse",
            primary_metric_value=value,
        )

    def test_keep_when_metric_improves(self) -> None:
        incumbent = self._summary(1.0000)
        current = self._summary(0.9900)
        decision = decide_promotion(current, incumbent, current_complexity=10.0, incumbent_complexity=10.0)
        self.assertEqual(decision.decision, "keep")
        self.assertTrue(decision.keep)

    def test_keep_on_tie_with_lower_complexity(self) -> None:
        incumbent = self._summary(1.0000)
        current = self._summary(1.0000)
        decision = decide_promotion(current, incumbent, current_complexity=9.0, incumbent_complexity=10.0)
        self.assertEqual(decision.decision, "keep")

    def test_discard_when_current_fails_guardrails(self) -> None:
        incumbent = self._summary(1.0000)
        current = self._summary(0.9800, status="success")
        decision = decide_promotion(
            current,
            incumbent,
            current_complexity=9.0,
            incumbent_complexity=10.0,
            current_guardrails_ok=False,
            incumbent_guardrails_ok=True,
        )
        self.assertEqual(decision.decision, "discard")


if __name__ == "__main__":
    unittest.main()
