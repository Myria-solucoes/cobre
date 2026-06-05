#!/usr/bin/env python3
"""Aggregate tuning cells into a CSV + a ranked console summary, and flag each
cell's correctness against the stage baseline.

  aggregate.py --runs runs --backend highs --stage 1 [--ref-tol 1e-6] [--cases DIR]

Reads every ``runs/<backend>/s<stage>/*/result.json``, groups repeats by cell,
reports min/median ``backward_solve_seconds`` and ``duration_seconds`` with the
percent delta vs the ``baseline`` cell, and applies a **risk-aware cut-validity**
gate. A cell is ``INVALID`` on an invalid cut, ``FAIL`` when the run errored or
reported failed solves, else ``PASS``.

Detecting an invalid cut depends on the risk measure (read from result.json's
``risk_averse`` field, or from each cell's ``stages.json`` under ``--cases``):
  * risk-neutral: ``LB > UB`` beyond ``--ref-tol`` (a cut sliced the optimum),
    AND a non-monotone (decreasing) per-iteration lower bound.
  * risk-averse (CVaR): ``LB > UB`` is EXPECTED (the forward-pass mean-cost "UB"
    is not a valid upper bound for a CVaR objective), so only the LB-monotonicity
    check applies. Without ``convergence.parquet`` (needs pandas/pyarrow) that
    check can't run, so the cell is reported ``PASS?`` (cut-validity *unchecked*).

(Cross-config LB *drift* via alternate optima is expected, reported as gap%, not
failed.) A run that errored before ``max_iterations`` is sorted below the passing
cells and flagged ``*``. Writes ``results.csv`` in the stage dir. For stage 1 it
also writes ``suggested_winner.json`` (the env of the fastest passing cell) to
feed ``grid.py --stage 2 --winner``.

Metrics-only; it never runs cobre.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

try:
    # Sibling module (same dir is on sys.path[0] when this script is invoked).
    # Reused so the stages.json risk-measure schema has a single source of truth.
    from run_cell import detect_risk_averse
except ImportError:  # pragma: no cover - defensive; result.json path still works
    detect_risk_averse = None  # type: ignore[assignment]


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cell_risk_averse(
    reps: list[dict[str, Any]], cases: Path | None, cut_sel: str
) -> bool:
    """True if the cell is a risk-averse (CVaR) run. Prefer the ``risk_averse``
    field recorded in result.json; fall back to reading the case's stages.json
    under ``--cases`` (so runs that predate risk-averse detection still work)."""
    for r in reps:
        if isinstance(r.get("risk_averse"), bool):
            return r["risk_averse"]
    if cases is not None and detect_risk_averse is not None:
        return bool(detect_risk_averse(cases / cut_sel).get("risk_averse"))
    return False


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
        "--ref-tol",
        type=float,
        default=1e-6,
        help="relative tolerance for the cut-validity gate (LB>UB risk-neutral, "
        "LB-monotonicity risk-averse)",
    )
    ap.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="per-method case dir (e.g. cases/highs). Used to read each cell's "
        "stages.json and detect a risk-averse (CVaR) run when result.json predates "
        "risk-averse detection. Without it, risk-aversion is read from result.json.",
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
        # Prefer the convergence-parquet bounds (per-iteration, authoritative)
        # over the metadata summary; fall back when the parquet is unavailable.
        lb = _first_num(reps, "conv_final_lower_bound") or _first_num(
            reps, "final_lower_bound"
        )
        ub = _first_num(reps, "conv_final_upper_bound") or _first_num(
            reps, "final_upper_bound"
        )
        gap = (
            None
            if (lb is None or ub is None)
            else 100.0 * (ub - lb) / max(1.0, abs(ub))
        )
        # Per-phase diagnostics (parquet): pivots-per-resolve & basis-reject rate.
        pivots = _first_num(reps, "bwd_pivots_per_solve")
        reject = _first_num(reps, "bwd_basis_reject_rate")
        ok_exit = all(r.get("exit_status") == 0 for r in reps)
        ok_fail = all((r.get("failed") in (0, None)) for r in reps)
        risk_averse = _cell_risk_averse(reps, args.cases, reps[0]["cut_sel"])
        # Cut-validity, signal 1 — LB monotonicity. The SDDP lower bound is
        # non-decreasing as cuts accumulate; conv_max_lb_decrease (from the
        # parquet) is the worst per-iteration drop, and a value beyond tolerance
        # means an invalid cut. Holds for BOTH risk-neutral and risk-averse runs.
        lb_decrease = _first_num(reps, "conv_max_lb_decrease")
        mono_tol = args.ref_tol * max(1.0, abs(lb)) if lb is not None else args.ref_tol
        bad_monotonic = lb_decrease is not None and lb_decrease > mono_tol
        # Cut-validity, signal 2 — LB <= UB, but ONLY for risk-NEUTRAL runs. Under
        # a CVaR objective the LB converges to the risk-adjusted optimum while the
        # forward-pass "UB" estimates the mean cost, so LB > UB is EXPECTED and is
        # NOT a failure. Disabling this check for risk-averse runs is the whole
        # point of detecting the risk measure.
        worst_viol = _first_num(reps, "conv_max_lb_minus_ub")
        if worst_viol is None and lb is not None and ub is not None:
            worst_viol = lb - ub
        tol_abs = args.ref_tol * max(1.0, abs(ub)) if ub is not None else args.ref_tol
        bad_lb_ub = (
            (not risk_averse) and worst_viol is not None and worst_viol > tol_abs
        )
        # Which validity signal actually applied — surfaced so a risk-averse run
        # with no convergence parquet (LB<=UB invalid here, monotonicity needs
        # pyarrow) reads as "unchecked", never a silent PASS.
        if risk_averse:
            cut_check = "monotonic" if lb_decrease is not None else "unchecked"
        else:
            cut_check = "lb<=ub+mono"
        if bad_lb_ub or bad_monotonic:
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
        # Iteration progress. A cell that errored mid-training stops short of
        # max_iterations, so its absolute timing/LB are NOT comparable to a full
        # run — they only look "fast" because less work was done. Surface this and
        # de-rank non-PASS rows so a partial run can't masquerade as the fastest.
        iters_done = _first_num(reps, "iterations_completed")
        iters_max = _first_num(reps, "max_iterations")
        partial = (
            iters_done is not None and iters_max is not None and iters_done < iters_max
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
                "bwd_pivots_per_solve": pivots,
                "bwd_basis_reject_rate": reject,
                "final_lower_bound": lb,
                "final_gap_percent": gap,
                "iterations_completed": iters_done,
                "max_iterations": iters_max,
                "partial": partial,
                "risk_averse": risk_averse,
                "cut_check": cut_check,
                "correctness": correctness,
                "env": json.dumps(reps[0]["env"], sort_keys=True),
            }
        )

    # PASS cells first (ranked by speed), then FAIL/INVALID — a partial run that
    # exits with an error must never sort above a complete one on raw timing.
    rows.sort(
        key=lambda r: (
            r["correctness"] != "PASS",
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
        f"{'cell':28} {'bwd_min':>9} {'Δ% base':>8} {'piv/slv':>8} "
        f"{'rej%':>6} {'gap%':>7} {'iters':>7} {'verdict':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        reject_pct = (
            None
            if r["bwd_basis_reject_rate"] is None
            else 100.0 * r["bwd_basis_reject_rate"]
        )
        if r["iterations_completed"] is None:
            iters_str = "n/a"
        elif r["max_iterations"] is None:
            iters_str = str(int(r["iterations_completed"]))
        else:
            iters_str = f"{int(r['iterations_completed'])}/{int(r['max_iterations'])}"
        # Verdict markers: `*` partial run (timing/LB not comparable); `?`
        # risk-averse cell whose cut-validity could not be checked (no parquet).
        verdict = r["correctness"]
        if r["partial"]:
            verdict += "*"
        if r["cut_check"] == "unchecked":
            verdict += "?"
        print(
            f"{r['cell']:28} {_fmt(r['backward_solve_s_min']):>9} "
            f"{_fmt(r['delta_pct_vs_baseline'], 1):>8} {_fmt(r['bwd_pivots_per_solve'], 1):>8} "
            f"{_fmt(reject_pct, 1):>6} {_fmt(r['final_gap_percent'], 3):>7} "
            f"{iters_str:>7} {verdict:>9}"
        )
    print(f"\nwrote {csv_path}")
    print(
        "cols: bwd_min=backward solve s (min over reps) · piv/slv=backward "
        "pivots-per-resolve · rej%=basis-rejection rate · iters=completed/max · "
        "verdict PASS/FAIL/INVALID"
    )
    print(
        "verdict: PASS ok · FAIL exit/solve error · INVALID invalid cut (LB>UB "
        "[risk-neutral] or LB decreased across iterations) · * partial run · "
        "? risk-averse, cut-validity unchecked"
    )
    n_risk = sum(1 for r in rows if r["risk_averse"])
    if n_risk:
        n_unchecked = sum(1 for r in rows if r["cut_check"] == "unchecked")
        note = (
            f"NOTE: {n_risk}/{len(rows)} cell(s) risk-averse (CVaR) → LB≤UB check "
            "OFF (LB>UB is EXPECTED under CVaR); cut-validity via LB-monotonicity. "
            "gap% here is LB-vs-mean-cost, not a convergence gap."
        )
        if n_unchecked:
            note += (
                f" {n_unchecked} UNCHECKED — install pandas+pyarrow and re-run "
                "aggregate to read convergence.parquet and enable the check."
            )
        print(note)

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
