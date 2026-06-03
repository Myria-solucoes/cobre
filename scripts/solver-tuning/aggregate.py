#!/usr/bin/env python3
"""Aggregate tuning cells into a CSV + a ranked console summary, and flag each
cell's correctness against the stage baseline.

  aggregate.py --runs runs --backend highs --stage 1 [--ref-tol 1e-6]

Reads every ``runs/<backend>/s<stage>/*/result.json``, groups repeats by cell,
reports min/median ``backward_solve_seconds`` and ``duration_seconds`` with the
percent delta vs the ``baseline`` cell, and marks a cell ``correctness=PASS``
when it exited 0, had no failed solves, and its ``final_lower_bound`` is within
``--ref-tol`` (relative) of the baseline's. Writes ``results.csv`` in the stage
dir. For stage 1 it also writes ``suggested_winner.json`` (the env of the
fastest correctness-passing cell) to feed ``grid.py --stage 2 --winner``.

Metrics-only; it never runs cobre.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(x: float | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def _first_num(reps: list[dict[str, Any]], key: str) -> float | None:
    return next((r[key] for r in reps if isinstance(r.get(key), (int, float))), None)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=("highs", "clp"))
    ap.add_argument("--stage", required=True, type=int, choices=(1, 2))
    ap.add_argument(
        "--ref-tol", type=float, default=1e-6, help="relative LB tolerance vs baseline"
    )
    args = ap.parse_args()

    stage_dir = args.runs / args.backend / f"s{args.stage}"
    results = [
        r
        for p in sorted(stage_dir.glob("*/result.json"))
        if (r := _load(p)) is not None
    ]
    if not results:
        print(f"no result.json under {stage_dir}")
        return 1

    # Group repeats by cell.
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cells.setdefault(r["cell"], []).append(r)

    def agg(reps: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
        vals = [r[key] for r in reps if isinstance(r.get(key), (int, float))]
        if not vals:
            return None, None
        return min(vals), statistics.median(vals)

    # Reference cell, flagged by grid.py (stage-1 "floor"; stage-2 "ws-full.sel-none").
    baseline = next((c for c, reps in cells.items() if reps[0].get("reference")), None)
    ref_lb: float | None = None
    base_bwd_min: float | None = None
    if baseline is not None:
        ref_lb = _first_num(cells[baseline], "final_lower_bound")
        base_bwd_min, _ = agg(cells[baseline], "backward_solve_seconds")

    rows: list[dict[str, Any]] = []
    for cell, reps in cells.items():
        bwd_min, bwd_med = agg(reps, "backward_solve_seconds")
        dur_min, dur_med = agg(reps, "duration_seconds")
        lb = _first_num(reps, "final_lower_bound")
        ub = _first_num(reps, "final_upper_bound")
        # Convergence gap computed from the bounds (robust to metadata layout).
        gap = (
            None
            if (lb is None or ub is None)
            else 100.0 * (ub - lb) / max(1.0, abs(ub))
        )
        ok_exit = all(r.get("exit_status") == 0 for r in reps)
        ok_fail = all((r.get("failed") in (0, None)) for r in reps)
        # Cut-validity gate: at convergence LB <= UB (gap >= 0). LB exceeding UB
        # beyond tolerance means a cut sliced off the true optimum (perturbation
        # or loose dual tolerance) — the report's primary correctness failure.
        invalid_cut = (
            lb is not None
            and ub is not None
            and lb - ub > args.ref_tol * max(1.0, abs(ub))
        )
        if invalid_cut:
            correctness = "INVALID"
        elif not (ok_exit and ok_fail):
            correctness = "FAIL"
        else:
            correctness = "PASS"
        delta = (
            None
            if (bwd_min is None or not base_bwd_min)
            else 100.0 * (bwd_min - base_bwd_min) / base_bwd_min
        )
        rows.append(
            {
                "cell": cell,
                "label": reps[0]["label"],
                "warmstart": reps[0]["warmstart"],
                "cut_sel": reps[0]["cut_sel"],
                "reps": len(reps),
                "backward_solve_s_min": bwd_min,
                "backward_solve_s_med": bwd_med,
                "duration_s_min": dur_min,
                "duration_s_med": dur_med,
                "delta_pct_vs_baseline": delta,
                "final_lower_bound": lb,
                "final_gap_percent": gap,
                "correctness": correctness,
                "env": json.dumps(reps[0]["env"], sort_keys=True),
            }
        )

    rows.sort(
        key=lambda r: (
            r["backward_solve_s_min"] is None,
            r["backward_solve_s_min"] or 0.0,
        )
    )

    # CSV.
    csv_path = stage_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Console summary.
    print(
        f"\n{args.backend} stage {args.stage}  (ref LB = {_fmt(ref_lb, 4)}, tol = {args.ref_tol})\n"
    )
    hdr = (
        f"{'cell':28} {'bwd_min':>9} {'Δ% base':>8} {'dur_min':>9} "
        f"{'gap%':>7} {'verdict':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['cell']:28} {_fmt(r['backward_solve_s_min']):>9} "
            f"{_fmt(r['delta_pct_vs_baseline'], 1):>8} {_fmt(r['duration_s_min']):>9} "
            f"{_fmt(r['final_gap_percent'], 3):>7} {r['correctness']:>8}"
        )
    print(f"\nwrote {csv_path}")
    print(
        "verdict: PASS ok · FAIL exit/solve error · INVALID LB>UB (cut sliced the optimum)"
    )

    # Stage 1: suggest the fastest correctness-passing cell's env (the user makes
    # the final call at the manual gate).
    if args.stage == 1:
        passing = [
            r
            for r in rows
            if r["correctness"] == "PASS" and r["backward_solve_s_min"] is not None
        ]
        if passing:
            winner = passing[0]
            winner_env: dict[str, str] = json.loads(winner["env"])
            win_path = stage_dir / "suggested_winner.json"
            win_path.write_text(json.dumps(winner_env, indent=2) + "\n")
            note = (
                " (baseline — no override beat the defaults)" if not winner_env else ""
            )
            print(
                f"\nsuggested winner: {winner['cell']}{note} "
                f"(Δ {_fmt(winner['delta_pct_vs_baseline'], 1)}% vs baseline)\n"
                f"  env written to {win_path} — review/edit, then:\n"
                f"  grid.py --backend {args.backend} --stage 2 --winner {win_path} "
                f"> manifests/{args.backend}.s2.jsonl"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
