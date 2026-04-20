"""Autoresearch-style search orchestration.

This module samples bounded hyperparameter configurations and can launch
config-driven training runs. It is intentionally dependency-free and only
executes model families that already expose a JSON config interface.
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from experiment_tracking import RunSummary, get_git_commit_hash
from promotion_policy import decide_promotion


SUPPORTED_FAMILIES = ("mlp_lr", "ft_transformer", "hybrid", "xgb")
EXECUTABLE_FAMILIES = {"mlp_lr", "ft_transformer", "hybrid", "xgb"}


@dataclass(frozen=True)
class SearchSpace:
    family: str
    integer_ranges: dict[str, tuple[int, int]]
    float_log_ranges: dict[str, tuple[float, float]]
    float_ranges: dict[str, tuple[float, float]]
    categorical: dict[str, tuple[Any, ...]]

    def sample(self, rng: random.Random) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for key, (lo, hi) in self.integer_ranges.items():
            config[key] = rng.randint(lo, hi)
        for key, (lo, hi) in self.float_ranges.items():
            config[key] = rng.uniform(lo, hi)
        for key, (lo, hi) in self.float_log_ranges.items():
            if lo <= 0 or hi <= 0:
                raise ValueError(f"Log-range bounds must be positive for {key}")
            config[key] = 10 ** rng.uniform(_log10(lo), _log10(hi))
        for key, choices in self.categorical.items():
            config[key] = rng.choice(list(choices))
        return config


@dataclass(frozen=True)
class TrialResult:
    run_id: str
    family: str
    suffix: str
    config_path: str
    output_dir: str
    status: str
    summary_path: str | None
    summary: dict[str, Any] | None


@dataclass(frozen=True)
class SearchPlan:
    family: str
    seed: int
    n_trials: int
    output_dir: str
    incumbent_path: str | None = None


MLP_SPACE = SearchSpace(
    family="mlp_lr",
    integer_ranges={
        "mlp.batch_size": (2048, 8192),
        "mlp.max_epochs": (250, 500),
        "mlp.patience": (20, 40),
        "mlp.cosine_t_max": (80, 180),
        "importance.perm_max_rows": (2000, 8000),
        "importance.perm_repeats": (3, 7),
    },
    float_log_ranges={
        "ridge.alpha": (1e-2, 10.0),
        "mlp.lr": (1e-4, 2e-3),
        "mlp.weight_decay": (1e-6, 3e-3),
        "mlp.cosine_eta_min": (1e-7, 1e-5),
    },
    float_ranges={
        "mlp.grad_clip_norm": (0.5, 2.0),
        "loss.huber_delta": (0.5, 2.0),
        "loss.pred_l1_weight": (0.0, 0.1),
    },
    categorical={
        "split.stratify_by_pc1": (True, False),
        "pipeline.scaler": ("robust_standard", "pca_whitening"),
        "pipeline.pca_augment": (True, False),
    },
)

FT_SPACE = SearchSpace(
    family="ft_transformer",
    integer_ranges={
        "d_token": (64, 160),
        "n_layers": (3, 6),
        "n_heads": (4, 8),
        "d_ff": (128, 512),
    },
    float_log_ranges={
        "lr": (3e-4, 2e-3),
        "weight_decay": (1e-6, 1e-3),
    },
    float_ranges={
        "range_loss_weight": (0.0, 0.05),
    },
    categorical={
        "loss_fn": ("quantile", "huber"),
        "range_loss_type": ("variance", "mmd"),
    },
)

HYBRID_SPACE = SearchSpace(
    family="hybrid",
    integer_ranges={
        "d_token": (64, 160),
        "n_layers": (3, 6),
        "n_heads": (4, 8),
        "d_ff": (128, 512),
        "mlp_hidden": (128, 384),
        "fusion_dim": (64, 192),
    },
    float_log_ranges={
        "lr": (1e-4, 8e-4),
        "weight_decay": (1e-6, 1e-3),
    },
    float_ranges={
        "range_loss_weight": (0.0, 0.05),
    },
    categorical={
        "loss_fn": ("quantile", "huber"),
        "range_loss_type": ("variance", "mmd"),
    },
)

XGB_SPACE = SearchSpace(
    family="xgb",
    integer_ranges={
        "max_depth": (4, 10),
        "n_estimators": (300, 1200),
    },
    float_log_ranges={
        "learning_rate": (0.01, 0.2),
        "reg_lambda": (1e-3, 10.0),
    },
    float_ranges={
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
    },
    categorical={},
)

SEARCH_SPACES = {
    "mlp_lr": MLP_SPACE,
    "ft_transformer": FT_SPACE,
    "hybrid": HYBRID_SPACE,
    "xgb": XGB_SPACE,
}


def _log10(x: float) -> float:
    import math

    return math.log10(x)


def family_search_space(family: str) -> SearchSpace:
    try:
        return SEARCH_SPACES[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported family: {family}") from exc


def sample_config(family: str, rng: random.Random) -> dict[str, Any]:
    sampled = family_search_space(family).sample(rng)
    if family in {"ft_transformer", "hybrid"}:
        sampled = _ensure_transformer_head_compat(sampled)
    return sampled


def _ensure_transformer_head_compat(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure sampled transformer configs satisfy d_token % n_heads == 0.

    The model constructor requires embed_dim to be divisible by num_heads.
    Search spaces may sample invalid pairs independently, so we repair them
    deterministically before writing search configs.
    """
    fixed = dict(config)
    d_token = int(fixed.get("d_token", 128))
    n_heads = int(fixed.get("n_heads", 8))

    if n_heads <= 0:
        n_heads = 1

    if d_token % n_heads != 0:
        divisors = [h for h in range(1, min(d_token, 32) + 1) if d_token % h == 0]
        if not divisors:
            n_heads = 1
        else:
            at_most = [h for h in divisors if h <= n_heads]
            if at_most:
                n_heads = max(at_most)
            else:
                n_heads = min(divisors)

    fixed["d_token"] = d_token
    fixed["n_heads"] = n_heads
    return fixed


def _nest_config(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        if "." not in key:
            nested[key] = value
            continue
        section, field = key.split(".", 1)
        nested.setdefault(section, {})[field] = value
    return nested


def build_run_config(family: str, flat_config: dict[str, Any]) -> dict[str, Any]:
    nested = _nest_config(flat_config)
    if family == "mlp_lr":
        return nested
    if family == "ft_transformer":
        return {
            "model": {
                "d_token": int(flat_config["d_token"]),
                "n_heads": int(flat_config["n_heads"]),
                "n_layers": int(flat_config["n_layers"]),
                "d_ff": int(flat_config["d_ff"]),
            },
            "loss": {
                "loss": str(flat_config["loss_fn"]),
                "range_loss_weight": float(flat_config["range_loss_weight"]),
                "range_loss_type": str(flat_config["range_loss_type"]),
            },
            "search": {
                "lr": float(flat_config["lr"]),
                "weight_decay": float(flat_config["weight_decay"]),
            },
        }
    if family == "hybrid":
        return {
            "model": {
                "d_token": int(flat_config["d_token"]),
                "n_heads": int(flat_config["n_heads"]),
                "n_layers": int(flat_config["n_layers"]),
                "d_ff": int(flat_config["d_ff"]),
                "mlp_hidden": int(flat_config["mlp_hidden"]),
                "fusion_dim": int(flat_config["fusion_dim"]),
            },
            "loss": {
                "loss": str(flat_config["loss_fn"]),
                "range_loss_weight": float(flat_config["range_loss_weight"]),
                "range_loss_type": str(flat_config["range_loss_type"]),
            },
            "search": {
                "lr": float(flat_config["lr"]),
                "weight_decay": float(flat_config["weight_decay"]),
            },
        }
    if family == "xgb":
        return {
            "xgb": {
                "max_depth": int(flat_config["max_depth"]),
                "n_estimators": int(flat_config["n_estimators"]),
                "learning_rate": float(flat_config["learning_rate"]),
                "reg_lambda": float(flat_config["reg_lambda"]),
                "subsample": float(flat_config["subsample"]),
                "colsample_bytree": float(flat_config["colsample_bytree"]),
            },
        }
    return nested


def build_trial_suffix(run_id: str, trial_index: int) -> str:
    return f"autoresearch_{run_id}_t{trial_index:03d}"


def build_trial_command(repo_root: str | Path, family: str, config_path: str, suffix: str) -> list[str]:
    repo_root = Path(repo_root)
    if family == "mlp_lr":
        script = repo_root / "src" / "mlp_lr_models.py"
        return [sys.executable, str(script), "--suffix", suffix, "--config", config_path]
    if family == "ft_transformer":
        script = repo_root / "src" / "models_transformer.py"
        return [sys.executable, str(script), "--suffix", suffix, "--config", config_path]
    if family == "hybrid":
        script = repo_root / "src" / "models_hybrid.py"
        return [sys.executable, str(script), "--suffix", suffix, "--config", config_path]
    if family == "xgb":
        script = repo_root / "src" / "xgb_models.py"
        return [sys.executable, str(script), "--suffix", suffix, "--config", config_path]
    raise NotImplementedError(f"Family {family} does not yet have an executable trainer")


def _family_output_dir(repo_root: Path, family: str, suffix: str) -> Path:
    if family == "mlp_lr":
        return repo_root / "results" / "model_mlp_lr" / suffix
    if family == "ft_transformer":
        return repo_root / "results" / "model_ft_transformer" / suffix
    if family == "hybrid":
        return repo_root / "results" / "model_hybrid" / suffix
    if family == "xgb":
        return repo_root / "results" / "model_xgb" / suffix
    raise ValueError(f"Unsupported family: {family}")


def _primary_metric_name(family: str) -> str:
    if family == "mlp_lr":
        return "mlp_test_rmse"
    if family == "ft_transformer":
        return "ft_test_rmse"
    if family == "hybrid":
        return "hybrid_test_rmse"
    if family == "xgb":
        return "xgb_test_rmse"
    return "test_rmse"


def _extract_primary_metric(output_dir: Path, family: str) -> float:
    csv_by_family = {
        "ft_transformer": output_dir / "stratified_pc1_rmse_Transformer.csv",
        "hybrid": output_dir / "stratified_pc1_rmse_Hybrid.csv",
        "xgb": output_dir / "stratified_pc1_rmse_XGB.csv",
    }
    path = csv_by_family.get(family)
    if path is None or not path.exists():
        return float("nan")

    vals: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals.append(float(row["rmse"]))
            except Exception:
                continue
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_trial(repo_root: str | Path, family: str, config: dict[str, Any], suffix: str) -> TrialResult:
    if family not in EXECUTABLE_FAMILIES:
        raise NotImplementedError(f"Family {family} is not executable yet")

    repo_root = Path(repo_root)
    output_dir = _family_output_dir(repo_root, family, suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "search_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    cmd = build_trial_command(repo_root, family, str(config_path), suffix)
    start_ts = time.monotonic()

    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        summary_path.unlink()

    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    runtime_seconds = time.monotonic() - start_ts

    if summary_path.exists():
        summary = load_summary(summary_path)
    elif proc.returncode == 0:
        primary_metric = _extract_primary_metric(output_dir, family)
        if math.isfinite(primary_metric):
            fallback = RunSummary(
                run_id=suffix,
                script_name=Path(cmd[1]).name,
                model_family=family,
                commit=get_git_commit_hash(repo_root),
                status="success",
                primary_metric_name=_primary_metric_name(family),
                primary_metric_value=float(primary_metric),
                secondary_metrics={
                    "summary_source": "autoresearch_fallback",
                    "fallback_metric_from": "stratified_pc1_rmse_csv_or_default",
                },
                peak_memory_mb=0.0,
                runtime_seconds=float(runtime_seconds),
                description=f"Autoresearch fallback summary for {family}",
                artifacts={
                    "output_dir": str(output_dir),
                    "run_config": str(config_path),
                },
                config=config,
            )
            summary = fallback.to_dict()
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        else:
            summary = None
    else:
        summary = None

    status = "success" if proc.returncode == 0 and summary is not None else "crash"
    return TrialResult(
        run_id=suffix,
        family=family,
        suffix=suffix,
        config_path=str(config_path),
        output_dir=str(output_dir),
        status=status,
        summary_path=str(summary_path) if summary_path.exists() else None,
        summary=summary,
    )


def run_search(repo_root: str | Path, plan: SearchPlan) -> list[TrialResult]:
    repo_root = Path(repo_root)
    rng = random.Random(plan.seed)
    results: list[TrialResult] = []
    incumbent: RunSummary | None = None
    incumbent_result: TrialResult | None = None

    for trial_index in range(plan.n_trials):
        config = build_run_config(plan.family, sample_config(plan.family, rng))
        suffix = build_trial_suffix(f"seed{plan.seed}", trial_index)
        result = run_trial(repo_root, plan.family, config, suffix)
        results.append(result)

        if result.summary is None:
            continue

        current = RunSummary(**result.summary)
        if incumbent is None:
            incumbent = current
            incumbent_result = result
            continue

        decision = decide_promotion(current, incumbent)
        if decision.keep:
            incumbent = current
            incumbent_result = result

    _write_search_manifest(repo_root, plan, results, incumbent_result)
    return results


def _write_search_manifest(repo_root: Path, plan: SearchPlan, results: list[TrialResult], incumbent_result: TrialResult | None) -> Path:
    manifest_dir = repo_root / "results" / "autoresearch_search"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{plan.family}_seed{plan.seed}.json"
    payload = {
        "plan": asdict(plan),
        "incumbent": asdict(incumbent_result) if incumbent_result is not None else None,
        "results": [asdict(r) for r in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch-style hyperparameter search")
    parser.add_argument("--family", choices=SUPPORTED_FAMILIES, default="mlp_lr")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    plan = SearchPlan(
        family=args.family,
        seed=args.seed,
        n_trials=args.n_trials,
        output_dir=args.output_dir or str(Path(args.repo_root) / "results" / "autoresearch_search"),
    )
    run_search(args.repo_root, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
