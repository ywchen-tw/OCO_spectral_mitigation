"""Deterministic keep/discard promotion policy for experiment runs.

This module is intentionally simple and side-effect free so it can be unit-tested
and reused by any trainer or orchestration loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tracking import RunSummary


@dataclass(frozen=True)
class PromotionDecision:
    decision: str
    reason: str
    metric_delta: float
    complexity_delta: float

    @property
    def keep(self) -> bool:
        return self.decision == "keep"


def decide_promotion(
    current: RunSummary,
    incumbent: RunSummary,
    *,
    epsilon: float = 1e-4,
    current_complexity: float = 0.0,
    incumbent_complexity: float = 0.0,
    current_guardrails_ok: bool = True,
    incumbent_guardrails_ok: bool = True,
) -> PromotionDecision:
    """Decide whether to keep a current run over an incumbent.

    Lower primary-metric values are better.

    Rules
    -----
    1. If current fails guardrails and incumbent passes, discard.
    2. If current improves primary metric by at least epsilon, keep.
    3. If the primary metric ties within epsilon, keep only if complexity
       improves or guardrails improve.
    4. Otherwise discard.
    """

    current_metric = float(current.primary_metric_value)
    incumbent_metric = float(incumbent.primary_metric_value)
    metric_delta = incumbent_metric - current_metric
    complexity_delta = float(incumbent_complexity) - float(current_complexity)

    if not current_guardrails_ok and incumbent_guardrails_ok:
        return PromotionDecision(
            decision="discard",
            reason="current run failed guardrails while incumbent passed",
            metric_delta=metric_delta,
            complexity_delta=complexity_delta,
        )

    if metric_delta > epsilon:
        return PromotionDecision(
            decision="keep",
            reason=f"primary metric improved by {metric_delta:.6f}",
            metric_delta=metric_delta,
            complexity_delta=complexity_delta,
        )

    metric_tie = abs(metric_delta) <= epsilon
    if metric_tie and (complexity_delta > 0 or (current_guardrails_ok and not incumbent_guardrails_ok)):
        reasons = []
        if complexity_delta > 0:
            reasons.append(f"complexity improved by {complexity_delta:.3f}")
        if current_guardrails_ok and not incumbent_guardrails_ok:
            reasons.append("guardrails improved")
        return PromotionDecision(
            decision="keep",
            reason="; ".join(reasons) if reasons else "tie-break promotion",
            metric_delta=metric_delta,
            complexity_delta=complexity_delta,
        )

    if not current_guardrails_ok:
        return PromotionDecision(
            decision="discard",
            reason="current run failed guardrails",
            metric_delta=metric_delta,
            complexity_delta=complexity_delta,
        )

    return PromotionDecision(
        decision="discard",
        reason="no sufficient improvement over incumbent",
        metric_delta=metric_delta,
        complexity_delta=complexity_delta,
    )
