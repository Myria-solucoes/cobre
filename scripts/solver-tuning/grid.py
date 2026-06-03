#!/usr/bin/env python3
"""Generate the solver/accelerator tuning manifest (one JSONL line per run).

The tuning campaign is staged, one backend at a time:

  Stage 1  OFAT over solver parameters (warm-start ``full``, cut selection
           ``none``); the ``baseline`` cell carries no overrides and is the
           correctness reference.
  Stage 2  Accelerator matrix (warm-start x cut-selection) with Stage-1's
           winning solver env fixed (passed via ``--winner``).

Each emitted line is one ``(cell, repeat)``. The SLURM array maps task id ->
line. Env values are the ``COBRE_TUNE_*`` overrides applied on top of the
compiled defaults; ``cut_sel`` selects the per-method case copy (see
``prep_cases.sh``), not an env var.

Usage:
  grid.py --backend highs --stage 1 [--reps 1]            > manifests/highs.s1.jsonl
  grid.py --backend highs --stage 2 --winner winner.json  > manifests/highs.s2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Stage-1 OFAT solver-parameter cells, per backend ------------------------
# Stage-1 cells per backend: a correctness-FLOOR baseline (the reference) plus
# OFAT *pure-speed* levers applied ON TOP of the floor.
#
# The floor is the config the source-grounded study mandates for a warm-started
# SDDP backward pass: dual simplex + warm basis + presolve OFF + perturbation
# OFF + tight dual tolerance (these protect cut validity AND the warm start;
# they are set, not tuned). cobre's compiled defaults already satisfy the floor
# EXCEPT HiGHS presolve (cobre ships presolve=on per spec SS4.1), so the HiGHS
# floor sets presolve=off and one confirmatory cell measures the cost of
# presolve=on.
#
# Enum values confirmed from the vendored HiGHS source
# (highs/simplex/SimplexConst.h): SimplexPriceStrategy = {0 Col, 1 Row,
# 2 RowSwitch, 3 RowSwitchColSwitch}; SimplexEdgeWeightStrategy = {-1 Choose,
# 0 Dantzig, 1 Devex, 2 SteepestEdge}. (Confirm the HiGHS scale-strategy enum
# value for equilibration before trusting the scale cell.)
#
# Each entry is (label, env-overrides, is_reference).
STAGE1: dict[str, list[tuple[str, dict[str, str], bool]]] = {
    "highs": [
        # Correctness floor (reference): presolve off. perturbation=0.0, dual
        # feasibility tol=1e-9, and dual-serial simplex are already compiled in.
        ("floor", {"COBRE_TUNE_HIGHS_PRESOLVE": "off"}, True),
        # Confirmatory: cost of the shipped presolve=on (spec SS4.1) vs the floor.
        ("presolve-on", {"COBRE_TUNE_HIGHS_PRESOLVE": "on"}, False),
        # --- pure speed levers, each on top of the floor (presolve off) ---
        # cobre default edge weight is 1 (Devex); test choose (-1) and SteepestEdge (2).
        (
            "edge-choose",
            {"COBRE_TUNE_HIGHS_PRESOLVE": "off", "COBRE_TUNE_HIGHS_EDGE_WEIGHT": "-1"},
            False,
        ),
        (
            "edge-steepest",
            {"COBRE_TUNE_HIGHS_PRESOLVE": "off", "COBRE_TUNE_HIGHS_EDGE_WEIGHT": "2"},
            False,
        ),
        # cobre uses scale=0 (its prescaler conditions the matrix); test HiGHS equilibration.
        (
            "scale-equil",
            {"COBRE_TUNE_HIGHS_PRESOLVE": "off", "COBRE_TUNE_HIGHS_SCALE": "2"},
            False,
        ),
        # cobre backward uses price=2 (RowSwitch); 3 = RowSwitchColSwitch (HiGHS default).
        (
            "price-rsc",
            {"COBRE_TUNE_HIGHS_PRESOLVE": "off", "COBRE_TUNE_HIGHS_PRICE": "3"},
            False,
        ),
    ],
    "clp": [
        # cobre's CLP compiled defaults already meet the floor (perturbation 102,
        # dual algorithm, dual tol 1e-9, no presolve on the dual() path).
        ("floor", {}, True),
        ("scaling-equil", {"COBRE_TUNE_CLP_SCALING": "1"}, False),
        # Uninitialized DSE weights — the study's warm-start hypothesis (full
        # weight init is under-amortized over a few-pivot resolve).
        ("pricing-uninit", {"COBRE_TUNE_CLP_PRICING_MODE": "0"}, False),
        ("pricing-full", {"COBRE_TUNE_CLP_PRICING_MODE": "1"}, False),
        # Factorization frequency is expected near-irrelevant for few-pivot
        # resolves; one cell confirms.
        ("factor-100", {"COBRE_TUNE_CLP_FACTOR_FREQ": "100"}, False),
    ],
}

WARMSTARTS = ("full", "core", "off")
CUT_SELS = ("none", "level1", "dominated")


def stage1_cells(backend: str) -> list[dict]:
    return [
        {
            "label": label,
            "warmstart": "full",
            "cut_sel": "none",
            "env": dict(env),
            "reference": is_ref,
        }
        for label, env, is_ref in STAGE1[backend]
    ]


def stage2_cells(winner_env: dict[str, str]) -> list[dict]:
    """Full 3x3 warm-start x cut-selection matrix on top of the winning solver env.

    ``ws-full.sel-none`` is the reference (the Stage-1 winner with no accelerator
    override). The ``ws-off.sel-{level1,dominated}`` corner is the SPTcpp-style
    configuration (cold-solve a small LP).
    """
    cells: list[dict] = []
    for ws in WARMSTARTS:
        for sel in CUT_SELS:
            env = dict(winner_env)
            # `full` is the default; only set the override for core/off so the
            # reference cell's env stays minimal.
            if ws != "full":
                env["COBRE_TUNE_WARMSTART"] = ws
            cells.append(
                {
                    "label": f"ws-{ws}.sel-{sel}",
                    "warmstart": ws,
                    "cut_sel": sel,
                    "env": env,
                    "reference": ws == "full" and sel == "none",
                }
            )
    return cells


def emit(cells: list[dict], backend: str, stage: int, reps: int) -> None:
    for cell in cells:
        base_id = f"s{stage}-{cell['label']}"
        for rep in range(reps):
            rec = {
                "run_id": f"{base_id}__rep{rep}",
                "cell": base_id,
                "stage": stage,
                "backend": backend,
                "rep": rep,
                "label": cell["label"],
                "warmstart": cell["warmstart"],
                "cut_sel": cell["cut_sel"],
                "reference": cell["reference"],
                "env": cell["env"],
            }
            print(json.dumps(rec, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", required=True, choices=("highs", "clp"))
    ap.add_argument("--stage", required=True, type=int, choices=(1, 2))
    ap.add_argument("--reps", type=int, default=1, help="repeats per cell (default 1)")
    ap.add_argument(
        "--winner",
        type=Path,
        help="Stage 2 only: JSON file with the chosen solver env, e.g. "
        '{"COBRE_TUNE_HIGHS_PRESOLVE": "off"}',
    )
    args = ap.parse_args()

    if args.stage == 1:
        cells = stage1_cells(args.backend)
    else:
        if not args.winner:
            print("--winner <json> is required for stage 2", file=sys.stderr)
            return 2
        winner_env = json.loads(args.winner.read_text())
        if not isinstance(winner_env, dict):
            print("--winner JSON must be an object of env vars", file=sys.stderr)
            return 2
        cells = stage2_cells({str(k): str(v) for k, v in winner_env.items()})

    emit(cells, args.backend, args.stage, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
