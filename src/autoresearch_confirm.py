"""Confirmation and promotion gate for autoresearch runs.

This module reruns a candidate and incumbent configuration, aggregates the
results, checks guardrails, and writes a promotion snapshot that can be used as
an audit trail for the final keep/discard decision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from experiment_tracking import RunSummary
from promotion_policy import decide_promotion
from autoresearch_search import TrialResult, run_trial


@dataclass(frozen=True)
class ConfirmationConfig:
    family: str = "mlp_lr"
    repo_root: str = "."
    repeats: int = 3
    epsilon: float = 1e-4
    max_primary_metric_std: float = 0.02
    max_runtime_regression_frac: float = 0.25
    max_memory_regression_frac: float = 0.25
    require_no_crashes: bool = True


@dataclass(frozen=True)
class ConfirmationStats:
    metric_name: str
    mean_primary_metric: float
    std_primary_metric: float
    min_primary_metric: float
    max_primary_metric: float
    mean_runtime_seconds: float
    mean_peak_memory_mb: float
    n_runs: int
    crashed_runs: int


@dataclass(frozen=True)
class ConfirmationRun:
    role: str
    suffix: str
    result: dict[str, Any]


@dataclass(frozen=True)
class PromotionSnapshot:
    config: dict[str, Any]
    candidate: dict[str, Any]
    incumbent: dict[str, Any]
    candidate_stats: dict[str, Any]
    incumbent_stats: dict[str, Any]
    candidate_guardrails_ok: bool
    incumbent_guardrails_ok: bool
    decision: str
    reason: str
    runs: list[dict[str, Any]]
    snapshot_path: str


TrialRunner = Callable[[Path, str, dict[str, Any], str], TrialResult]


def _load_summary(summary: dict[str, Any]) -> RunSummary:
    return RunSummary(**summary)


def _base_output_dir(repo_root: Path, family: str) -> Path:
    return repo_root / "results" / "autoresearch_confirmation" / family


def _build_suffix(prefix: str, role: str, repeat_idx: int) -> str:
    return f"{prefix}_{role}_r{repeat_idx:02d}"


def _load_repeat_results(
    repo_root: Path,
    family: str,
    config: dict[str, Any],
    repeats: int,
    prefix: str,
    role: str,
    trial_runner: TrialRunner,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for repeat_idx in range(repeats):
        suffix = _build_suffix(prefix, role, repeat_idx)
        results.append(trial_runner(repo_root, family, config, suffix))
    return results


def _summary_from_runs(base: RunSummary, stats: ConfirmationStats) -> RunSummary:
    merged_secondary = dict(base.secondary_metrics)
    merged_secondary.update({
        "confirm_mean_primary_metric": stats.mean_primary_metric,
        "confirm_std_primary_metric": stats.std_primary_metric,
        "confirm_min_primary_metric": stats.min_primary_metric,
        "confirm_max_primary_metric": stats.max_primary_metric,
        "confirm_mean_runtime_seconds": stats.mean_runtime_seconds,
        "confirm_mean_peak_memory_mb": stats.mean_peak_memory_mb,
        "confirm_n_runs": stats.n_runs,
        "confirm_crashed_runs": stats.crashed_runs,
    })
    return RunSummary(
        run_id=base.run_id,
        script_name=base.script_name,
        model_family=base.model_family,
        commit=base.commit,
        status=base.status,
        primary_metric_name=base.primary_metric_name,
        primary_metric_value=stats.mean_primary_metric,
        secondary_metrics=merged_secondary,
        peak_memory_mb=stats.mean_peak_memory_mb,
        runtime_seconds=stats.mean_runtime_seconds,
        description=base.description,
        artifacts=base.artifacts,
        config=base.config,
        timestamp_utc=base.timestamp_utc,
    )


def _aggregate_results(results: list[TrialResult], metric_name: str) -> tuple[ConfirmationStats, list[dict[str, Any]]]:
    metric_values: list[float] = []
    runtime_values: list[float] = []
    memory_values: list[float] = []
    crashed_runs = 0
    records: list[dict[str, Any]] = []

    for result in results:
        summary = result.summary
        if result.status != "success" or summary is None:
            crashed_runs += 1
            records.append({"run_id": result.run_id, "status": result.status, "summary": None})
            continue

        run = _load_summary(summary)
        metric_values.append(float(run.primary_metric_value))
        runtime_values.append(float(run.runtime_seconds))
        memory_values.append(float(run.peak_memory_mb))
        records.append({
            "run_id": result.run_id,
            "status": result.status,
            "summary": run.to_dict(),
        })

    if metric_values:
        metric_mean = statistics.fmean(metric_values)
        metric_std = statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0
        metric_min = min(metric_values)
        metric_max = max(metric_values)
    else:
        metric_mean = float("nan")
        metric_std = float("nan")
        metric_min = float("nan")
        metric_max = float("nan")

    stats = ConfirmationStats(
        metric_name=metric_name,
        mean_primary_metric=metric_mean,
        std_primary_metric=metric_std,
        min_primary_metric=metric_min,
        max_primary_metric=metric_max,
        mean_runtime_seconds=statistics.fmean(runtime_values) if runtime_values else float("nan"),
        mean_peak_memory_mb=statistics.fmean(memory_values) if memory_values else float("nan"),
        n_runs=len(results),
        crashed_runs=crashed_runs,
    )
    return stats, records


def confirm_and_promote(
    candidate: RunSummary,
    incumbent: RunSummary,
    config: ConfirmationConfig,
    *,
    trial_runner: TrialRunner = run_trial,
) -> PromotionSnapshot:
    repo_root = Path(config.repo_root)
    out_dir = _base_output_dir(repo_root, config.family)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / f"{candidate.run_id}_vs_{incumbent.run_id}.json"

    if candidate.config is None or incumbent.config is None:
        raise ValueError("Both candidate and incumbent must contain a config payload")

    candidate_results = _load_repeat_results(
        repo_root,
        config.family,
        candidate.config,
        config.repeats,
        f"confirm_{candidate.run_id}",
        "candidate",
        trial_runner,
    )
    incumbent_results = _load_repeat_results(
        repo_root,
        config.family,
        incumbent.config,
        config.repeats,
        f"confirm_{incumbent.run_id}",
        "incumbent",
        trial_runner,
    )

    candidate_stats, candidate_records = _aggregate_results(candidate_results, candidate.primary_metric_name)
    incumbent_stats, incumbent_records = _aggregate_results(incumbent_results, incumbent.primary_metric_name)

    candidate_guardrails_ok = True
    incumbent_guardrails_ok = True

    if config.require_no_crashes:
        candidate_guardrails_ok = candidate_guardrails_ok and candidate_stats.crashed_runs == 0
        incumbent_guardrails_ok = incumbent_guardrails_ok and incumbent_stats.crashed_runs == 0

    candidate_has_success = candidate_stats.crashed_runs < candidate_stats.n_runs
    incumbent_has_success = incumbent_stats.crashed_runs < incumbent_stats.n_runs

    if candidate_has_success:
        candidate_guardrails_ok = candidate_guardrails_ok and not math.isnan(candidate_stats.mean_primary_metric)
        candidate_guardrails_ok = candidate_guardrails_ok and candidate_stats.std_primary_metric <= config.max_primary_metric_std
        if incumbent_has_success and not math.isnan(incumbent_stats.mean_runtime_seconds) and not math.isnan(candidate_stats.mean_runtime_seconds):
            candidate_guardrails_ok = candidate_guardrails_ok and (
                candidate_stats.mean_runtime_seconds <= incumbent_stats.mean_runtime_seconds * (1.0 + config.max_runtime_regression_frac)
            )
        if incumbent_has_success and not math.isnan(incumbent_stats.mean_peak_memory_mb) and not math.isnan(candidate_stats.mean_peak_memory_mb):
            candidate_guardrails_ok = candidate_guardrails_ok and (
                candidate_stats.mean_peak_memory_mb <= incumbent_stats.mean_peak_memory_mb * (1.0 + config.max_memory_regression_frac)
            )

    if incumbent_has_success:
        incumbent_guardrails_ok = incumbent_guardrails_ok and not math.isnan(incumbent_stats.mean_primary_metric)
        incumbent_guardrails_ok = incumbent_guardrails_ok and incumbent_stats.std_primary_metric <= config.max_primary_metric_std
    else:
        incumbent_guardrails_ok = False

    confirmed_candidate = _summary_from_runs(candidate, candidate_stats)
    confirmed_incumbent = _summary_from_runs(incumbent, incumbent_stats)

    decision = decide_promotion(
        confirmed_candidate,
        confirmed_incumbent,
        epsilon=config.epsilon,
        current_complexity=float(candidate.secondary_metrics.get("num_params_M", 0.0) or 0.0),
        incumbent_complexity=float(incumbent.secondary_metrics.get("num_params_M", 0.0) or 0.0),
        current_guardrails_ok=candidate_guardrails_ok,
        incumbent_guardrails_ok=incumbent_guardrails_ok,
    )

    runs = (
        [{"role": "candidate", **record} for record in candidate_records]
        + [{"role": "incumbent", **record} for record in incumbent_records]
    )
    snapshot = PromotionSnapshot(
        config=asdict(config),
        candidate=confirmed_candidate.to_dict(),
        incumbent=confirmed_incumbent.to_dict(),
        candidate_stats=asdict(candidate_stats),
        incumbent_stats=asdict(incumbent_stats),
        candidate_guardrails_ok=candidate_guardrails_ok,
        incumbent_guardrails_ok=incumbent_guardrails_ok,
        decision=decision.decision,
        reason=decision.reason,
        runs=runs,
        snapshot_path=str(snapshot_path),
    )
    snapshot_path.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True), encoding="utf-8")
    return snapshot


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confirmation and promotion gate")
    parser.add_argument("--candidate-summary", type=str, required=True)
    parser.add_argument("--incumbent-summary", type=str, required=True)
    parser.add_argument("--family", choices=["mlp_lr", "ft_transformer", "hybrid", "xgb"], default="mlp_lr")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--max-primary-metric-std", type=float, default=0.02)
    parser.add_argument("--max-runtime-regression-frac", type=float, default=0.25)
    parser.add_argument("--max-memory-regression-frac", type=float, default=0.25)
    parser.add_argument("--allow-crashes", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    candidate = RunSummary(**json.loads(Path(args.candidate_summary).read_text(encoding="utf-8")))
    incumbent = RunSummary(**json.loads(Path(args.incumbent_summary).read_text(encoding="utf-8")))
    config = ConfirmationConfig(
        family=args.family,
        repo_root=args.repo_root,
        repeats=args.repeats,
        epsilon=args.epsilon,
        max_primary_metric_std=args.max_primary_metric_std,
        max_runtime_regression_frac=args.max_runtime_regression_frac,
        max_memory_regression_frac=args.max_memory_regression_frac,
        require_no_crashes=not args.allow_crashes,
    )
    snapshot = confirm_and_promote(candidate, incumbent, config)
    print(json.dumps(asdict(snapshot), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
