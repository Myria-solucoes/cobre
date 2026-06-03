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
# (label, env-overrides-on-top-of-defaults). The baseline has an empty env
# (= current compiled defaults) and is the correctness reference for the stage.
#
# NOTE: confirm COBRE_TUNE_HIGHS_SCALE's equilibration value against the linked
# HiGHS version before trusting that cell (the ideas doc says 2; a code comment
# suggested 4). Likewise confirm the price-strategy enum labels.
STAGE1: dict[str, list[tuple[str, dict[str, str]]]] = {
    "highs": [
        ("baseline", {}),
        ("presolve-off", {"COBRE_TUNE_HIGHS_PRESOLVE": "off"}),  # headline experiment
        ("edge-se", {"COBRE_TUNE_HIGHS_EDGE_WEIGHT": "2"}),  # SteepestEdge
        (
            "scale-equil",
            {"COBRE_TUNE_HIGHS_SCALE": "2"},
        ),  # equilibration (confirm enum)
        ("price-rsc", {"COBRE_TUNE_HIGHS_PRICE": "3"}),  # RowSwitchColSwitch
    ],
    "clp": [
        ("baseline", {}),
        ("scaling-equil", {"COBRE_TUNE_CLP_SCALING": "1"}),
        ("pricing-full", {"COBRE_TUNE_CLP_PRICING_MODE": "1"}),  # full DSE, all phases
        ("factor-100", {"COBRE_TUNE_CLP_FACTOR_FREQ": "100"}),
        ("factor-400", {"COBRE_TUNE_CLP_FACTOR_FREQ": "400"}),
    ],
}

WARMSTARTS = ("full", "core", "off")
CUT_SELS = ("none", "level1", "dominated")


def stage1_cells(backend: str) -> list[dict]:
    return [
        {"label": label, "warmstart": "full", "cut_sel": "none", "env": dict(env)}
        for label, env in STAGE1[backend]
    ]


def stage2_cells(winner_env: dict[str, str]) -> list[dict]:
    """Full 3x3 warm-start x cut-selection matrix on top of the winning solver env."""
    cells: list[dict] = []
    for ws in WARMSTARTS:
        for sel in CUT_SELS:
            env = dict(winner_env)
            # `full` is the default; only set the override for core/off so the
            # baseline cell's env stays minimal.
            if ws != "full":
                env["COBRE_TUNE_WARMSTART"] = ws
            cells.append(
                {
                    "label": f"ws-{ws}.sel-{sel}",
                    "warmstart": ws,
                    "cut_sel": sel,
                    "env": env,
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
