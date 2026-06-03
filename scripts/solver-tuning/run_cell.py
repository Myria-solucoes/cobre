#!/usr/bin/env python3
"""Run one tuning cell (one manifest line): set the COBRE_TUNE_* env, invoke
``cobre run`` on the cell's case copy, and write ``tune_params.json`` (the exact
parameters used) and ``result.json`` (parsed metrics) into the cell output dir.

Standalone (no SLURM):
  run_cell.py --manifest m.jsonl --index 0 --cases CASES --runs RUNS --binary BIN

Under SLURM the array wrapper passes ``--index "$SLURM_ARRAY_TASK_ID"``.

Idempotent/resumable: a cell whose ``result.json`` already exists is skipped.
The correctness gate (final-LB vs the baseline cell) is computed later by
``aggregate.py``; this script only records each cell's metrics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def extract_metrics(out_dir: Path) -> dict[str, Any]:
    """Pull the comparison metrics from cobre's training (+ simulation) metadata.

    Missing values come back as ``None`` (e.g. when a run failed before writing
    metadata), so a failed cell still produces a ``result.json``.
    """
    training = _load_json(out_dir / "training" / "metadata.json") or {}
    solve = (
        training.get("solve_stats")
        if isinstance(training.get("solve_stats"), dict)
        else {}
    )
    bounds = training.get("bounds") if isinstance(training.get("bounds"), dict) else {}
    sim = _load_json(out_dir / "simulation" / "metadata.json") or {}

    return {
        "duration_seconds": training.get("duration_seconds"),
        "forward_solve_seconds": solve.get("forward_solve_seconds"),
        "backward_solve_seconds": solve.get("backward_solve_seconds"),
        "total_lp_solves": solve.get("total_lp_solves"),
        "retried": solve.get("retried"),
        "failed": solve.get("failed"),
        # final_lower_bound has lived both at top level and under "bounds".
        "final_lower_bound": bounds.get(
            "final_lower_bound", training.get("final_lower_bound")
        ),
        "simulation_duration_seconds": sim.get("duration_seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument(
        "--cases", required=True, type=Path, help="dir with per-method case copies"
    )
    ap.add_argument("--runs", required=True, type=Path, help="output root for cells")
    ap.add_argument(
        "--binary", required=True, type=Path, help="path to the cobre binary"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("COBRE_BENCH_THREADS", "96")),
    )
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()

    lines = [ln for ln in args.manifest.read_text().splitlines() if ln.strip()]
    if not 0 <= args.index < len(lines):
        print(f"index {args.index} out of range (0..{len(lines) - 1})", file=sys.stderr)
        return 2
    cell = json.loads(lines[args.index])

    cell_out = args.runs / cell["backend"] / f"s{cell['stage']}" / cell["run_id"]
    if (cell_out / "result.json").exists():
        print(f"[skip] {cell['run_id']} already complete")
        return 0
    cell_out.mkdir(parents=True, exist_ok=True)

    case_dir = args.cases / cell["cut_sel"]
    if not case_dir.is_dir():
        print(f"missing case copy {case_dir} (run prep_cases.sh)", file=sys.stderr)
        return 3

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cell["env"].items()})

    started = _utcnow()
    params = {
        **cell,
        "binary": str(args.binary),
        "threads": args.threads,
        "case_dir": str(case_dir),
        "git_commit": _git_commit(args.repo),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": started,
    }
    (cell_out / "tune_params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n"
    )

    out_dir = cell_out / "output"
    cmd = [
        str(args.binary),
        "run",
        str(case_dir),
        "--output",
        str(out_dir),
        "--threads",
        str(args.threads),
        "--quiet",
    ]
    env_echo = (
        " ".join(f"{k}={v}" for k, v in sorted(cell["env"].items())) or "(defaults)"
    )
    print(f"[run] {cell['run_id']}  cut_sel={cell['cut_sel']}  env: {env_echo}")

    proc = subprocess.run(cmd, env=env)
    ended = _utcnow()

    result: dict[str, Any] = {
        "run_id": cell["run_id"],
        "cell": cell["cell"],
        "stage": cell["stage"],
        "backend": cell["backend"],
        "rep": cell["rep"],
        "label": cell["label"],
        "warmstart": cell["warmstart"],
        "cut_sel": cell["cut_sel"],
        "env": cell["env"],
        "exit_status": proc.returncode,
        "started_at": started,
        "ended_at": ended,
    }
    result.update(extract_metrics(out_dir))
    (cell_out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"[done] {cell['run_id']} exit={proc.returncode} "
        f"backward_solve_s={result.get('backward_solve_seconds')} "
        f"duration_s={result.get('duration_seconds')}"
    )
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
