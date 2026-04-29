"""Time-budgeted autonomous search loop.

The loop reuses the autoresearch search layer and adds stop guards for:
- wall-clock budget
- maximum number of trials
- consecutive crash limit
- consecutive non-improvement limit

It is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .tracking import RunSummary
from .promotion import decide_promotion
from .search import (
    SUPPORTED_FAMILIES,
    build_run_config,
    build_trial_suffix,
    sample_config,
    run_trial,
    TrialResult,
)


@dataclass(frozen=True)
class LoopConfig:
    family: str = "mlp_lr"
    seed: int = 42
    max_trials: int = 12
    max_duration_seconds: float = 300.0
    max_consecutive_crashes: int = 3
    max_consecutive_non_improve: int = 5
    epsilon: float = 1e-4
    repo_root: str = "."


@dataclass
class LoopState:
    trial_index: int = 0
    consecutive_crashes: int = 0
    consecutive_non_improve: int = 0
    start_monotonic: float = 0.0
    incumbent: RunSummary | None = None
    incumbent_result: TrialResult | None = None
    stopped_reason: str = ""


@dataclass(frozen=True)
class LoopReport:
    config: dict[str, Any]
    state: dict[str, Any]
    trials: list[dict[str, Any]]
    incumbent: dict[str, Any] | None
    incumbent_result: dict[str, Any] | None
    stopped_reason: str
    elapsed_seconds: float


TrialRunner = Callable[[Path, str, dict[str, Any], str], TrialResult]
Clock = Callable[[], float]


def _default_output_dir(repo_root: Path, family: str, seed: int) -> Path:
    return repo_root / "results" / "autoresearch_loop" / f"{family}_seed{seed}"


def _load_summary_as_run(summary: dict[str, Any]) -> RunSummary:
    return RunSummary(**summary)


def should_stop(
    *,
    elapsed_seconds: float,
    config: LoopConfig,
    state: LoopState,
) -> str | None:
    if elapsed_seconds >= config.max_duration_seconds:
        return f"time budget reached ({elapsed_seconds:.1f}s >= {config.max_duration_seconds:.1f}s)"
    if state.trial_index >= config.max_trials:
        return f"trial budget reached ({state.trial_index} >= {config.max_trials})"
    if state.consecutive_crashes >= config.max_consecutive_crashes:
        return f"crash streak limit reached ({state.consecutive_crashes} >= {config.max_consecutive_crashes})"
    if state.consecutive_non_improve >= config.max_consecutive_non_improve:
        return (
            "non-improvement streak limit reached "
            f"({state.consecutive_non_improve} >= {config.max_consecutive_non_improve})"
        )
    return None


def run_autoresearch_loop(
    config: LoopConfig,
    *,
    trial_runner: TrialRunner = run_trial,
    clock: Clock = time.monotonic,
) -> LoopReport:
    repo_root = Path(config.repo_root)
    loop_dir = _default_output_dir(repo_root, config.family, config.seed)
    loop_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = loop_dir / "loop_manifest.json"

    rng_seed_tag = f"seed{config.seed}"
    import random

    rng = random.Random(config.seed)
    state = LoopState(start_monotonic=clock())
    trial_records: list[dict[str, Any]] = []

    while True:
        elapsed = clock() - state.start_monotonic
        stop_reason = should_stop(elapsed_seconds=elapsed, config=config, state=state)
        if stop_reason is not None:
            state.stopped_reason = stop_reason
            break

        flat_config = sample_config(config.family, rng)
        nested_config = build_run_config(config.family, flat_config)
        suffix = build_trial_suffix(rng_seed_tag, state.trial_index)

        try:
            result = trial_runner(repo_root, config.family, nested_config, suffix)
        except Exception as exc:  # pragma: no cover - safety net for external runner failures
            result = TrialResult(
                run_id=suffix,
                family=config.family,
                suffix=suffix,
                config_path="",
                output_dir=str(loop_dir / suffix),
                status="crash",
                summary_path=None,
                summary={"error": str(exc)},
            )

        trial_records.append(asdict(result))
        state.trial_index += 1

        if result.status != "success" or result.summary is None:
            state.consecutive_crashes += 1
            state.consecutive_non_improve += 1
            continue

        current = _load_summary_as_run(result.summary)
        current_complexity = float(current.secondary_metrics.get("num_params_M", 0.0) or 0.0)
        incumbent_complexity = (
            float(state.incumbent.secondary_metrics.get("num_params_M", 0.0) or 0.0)
            if state.incumbent is not None
            else current_complexity
        )

        if state.incumbent is None:
            state.incumbent = current
            state.incumbent_result = result
            state.consecutive_crashes = 0
            state.consecutive_non_improve = 0
            continue

        decision = decide_promotion(
            current,
            state.incumbent,
            epsilon=config.epsilon,
            current_complexity=current_complexity,
            incumbent_complexity=incumbent_complexity,
            current_guardrails_ok=True,
            incumbent_guardrails_ok=True,
        )
        if decision.keep:
            state.incumbent = current
            state.incumbent_result = result
            state.consecutive_non_improve = 0
        else:
            state.consecutive_non_improve += 1

        state.consecutive_crashes = 0

    report = LoopReport(
        config=asdict(config),
        state=asdict(state),
        trials=trial_records,
        incumbent=state.incumbent.to_dict() if state.incumbent is not None else None,
        incumbent_result=asdict(state.incumbent_result) if state.incumbent_result is not None else None,
        stopped_reason=state.stopped_reason,
        elapsed_seconds=clock() - state.start_monotonic,
    )
    manifest_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time-budgeted autoresearch loop")
    parser.add_argument("--family", choices=SUPPORTED_FAMILIES, default="mlp_lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    parser.add_argument("--max-consecutive-crashes", type=int, default=3)
    parser.add_argument("--max-consecutive-non-improve", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = LoopConfig(
        family=args.family,
        seed=args.seed,
        max_trials=args.max_trials,
        max_duration_seconds=args.max_duration_seconds,
        max_consecutive_crashes=args.max_consecutive_crashes,
        max_consecutive_non_improve=args.max_consecutive_non_improve,
        epsilon=args.epsilon,
        repo_root=args.repo_root,
    )
    run_autoresearch_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
