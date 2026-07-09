"""leakage_guard.py — training-date overlap guard for evaluation runs.

Every trainer (deep_ensemble, gbdt_baselines, linear_baseline,
structured_dcn_ensemble, tabm) writes a ``training_dates.json`` manifest
{train_dates, calib_dates, held_dates} into its model dir.  This module is
the enforcement side: evaluation drivers (build_*_plot_data.py, i.e. the
``run_case`` chain) call :func:`check_training_overlap` with the model dirs
and the evaluation input, and the run REFUSES when an evaluation date was
seen in training or calibration — the manuscript's one-sentence leakage
guarantee ("no evaluation date appears in any training manifest").

Date identity follows the parquet DATE LABEL convention used at training
time (combined_YYYY-MM-DD…), so cross-midnight spill inside a labeled file
does not spuriously trigger; when no label is parseable the UTC dates of the
``time`` column are used instead.

Model dirs without a manifest (pre-2026-07 checkpoints) produce a WARNING,
not a refusal — the guarantee is then explicitly unverifiable for that dir.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def dates_from_paths(paths) -> set:
    """Date labels (YYYY-MM-DD) parsed from file names like
    combined_2018-09-02_all_orbits.parquet."""
    out = set()
    for p in paths or ():
        m = _DATE_RE.search(Path(str(p)).name)
        if m:
            out.add(m.group(1))
    return out


def dates_from_times(times) -> set:
    """UTC dates covered by an epoch-seconds time array (fallback identity)."""
    out = set()
    for t in {float(t) // 86400 for t in times if t == t}:
        out.add(datetime.fromtimestamp(t * 86400, tz=timezone.utc)
                .strftime("%Y-%m-%d"))
    return out


def load_training_dates(model_dirs, *, include_calib: bool = True):
    """Union of train (+calib) dates over model dirs.

    Returns (dates: set[str], missing: list[str]) where ``missing`` lists dirs
    without a readable training_dates.json manifest."""
    dates, missing = set(), []
    for d in model_dirs or ():
        p = Path(d) / "training_dates.json"
        try:
            man = json.loads(p.read_text())
        except (OSError, ValueError):
            missing.append(str(d))
            continue
        dates.update(man.get("train_dates") or ())
        if include_calib:
            dates.update(man.get("calib_dates") or ())
    return dates, missing


def check_training_overlap(model_dirs, *, input_paths=None, times=None,
                           include_calib: bool = True, allow: bool = False,
                           tag: str = "") -> dict:
    """Refuse (SystemExit) when an evaluation date overlaps the training
    manifest of any model dir; ``allow=True`` downgrades to a loud warning.

    Evaluation dates come from ``input_paths`` file-name labels when
    parseable, else from ``times`` (epoch seconds).  Returns a summary dict
    {eval_dates, train_dates, overlap, unverified} for logging/tests."""
    eval_dates = dates_from_paths(input_paths)
    src = "filename labels"
    if not eval_dates and times is not None:
        eval_dates = dates_from_times(times)
        src = "time column (UTC)"
    train_dates, missing = load_training_dates(model_dirs,
                                               include_calib=include_calib)
    overlap = sorted(eval_dates & train_dates)
    label = f" [{tag}]" if tag else ""
    if missing:
        print(f"  ⚠ leakage guard{label}: no training_dates.json in "
              f"{len(missing)} model dir(s) — overlap UNVERIFIABLE there: "
              + ", ".join(Path(m).name for m in missing))
    if overlap:
        msg = (f"leakage guard{label}: evaluation date(s) {overlap} ({src}) "
               f"appear in the training/calibration manifest "
               f"({len(train_dates)} manifest dates)")
        if not allow:
            raise SystemExit(
                "  ✗ " + msg + "\n  (refused; pass --allow-train-overlap to "
                "override for a deliberate in-sample diagnostic)")
        print(f"  ⚠ {msg} — OVERRIDDEN by --allow-train-overlap; this run is "
              f"IN-SAMPLE and must not be quoted as validation")
    elif eval_dates:
        print(f"  leakage guard{label}: OK — {len(eval_dates)} evaluation "
              f"date(s) ({src}) disjoint from {len(train_dates)} manifest "
              f"date(s)")
    else:
        print(f"  ⚠ leakage guard{label}: could not determine evaluation "
              f"dates — overlap not checked")
    return dict(eval_dates=sorted(eval_dates), train_dates=sorted(train_dates),
                overlap=overlap, unverified=missing)
