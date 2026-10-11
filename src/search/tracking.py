"""Experiment tracking helpers for autoresearch-style runs.

Provides a unified run-summary schema and an append-only TSV ledger so model
training scripts can emit comparable records without duplicating logging logic.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_HEADER = [
    "run_id",
    "timestamp_utc",
    "script_name",
    "model_family",
    "commit",
    "status",
    "primary_metric_name",
    "primary_metric_value",
    "secondary_metrics_json",
    "peak_memory_mb",
    "runtime_seconds",
    "description",
    "artifacts_json",
    "config_json",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_git_commit_hash(repo_path: str | Path | None = None) -> str:
    cwd = str(repo_path) if repo_path is not None else None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    commit = (proc.stdout or "").strip()
    return commit or "unknown"


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _tsv_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


@dataclass
class RunSummary:
    run_id: str
    script_name: str
    model_family: str
    commit: str
    status: str
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    runtime_seconds: float = 0.0
    description: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_ledger_row(self) -> dict[str, str]:
        return {
            "run_id": _tsv_escape(self.run_id),
            "timestamp_utc": _tsv_escape(self.timestamp_utc),
            "script_name": _tsv_escape(self.script_name),
            "model_family": _tsv_escape(self.model_family),
            "commit": _tsv_escape(self.commit),
            "status": _tsv_escape(self.status),
            "primary_metric_name": _tsv_escape(self.primary_metric_name),
            "primary_metric_value": f"{self.primary_metric_value:.6f}",
            "secondary_metrics_json": _tsv_escape(_compact_json(self.secondary_metrics)),
            "peak_memory_mb": f"{self.peak_memory_mb:.1f}",
            "runtime_seconds": f"{self.runtime_seconds:.1f}",
            "description": _tsv_escape(self.description),
            "artifacts_json": _tsv_escape(_compact_json(self.artifacts)),
            "config_json": _tsv_escape(_compact_json(self.config)),
        }


def write_run_summary(summary: RunSummary, output_dir: str | Path, ledger_path: str | Path) -> Path:
    output_dir = Path(output_dir)
    ledger_path = Path(ledger_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    write_header = not ledger_path.exists()
    with ledger_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(LEDGER_HEADER) + "\n")
        row = summary.to_ledger_row()
        f.write("\t".join(row[col] for col in LEDGER_HEADER) + "\n")

    return summary_path