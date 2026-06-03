#!/usr/bin/env python3
"""Patch a copied case's ``config.json`` to set ``training.cut_selection`` for one
selection method, leaving every other field untouched.

  patch_config.py <case>/config.json --method none|level1|dominated

Only the selection keys are managed; other keys under ``cut_selection`` (e.g.
``max_active_per_stage``) are preserved. ``domination_epsilon`` for the
``dominated`` method is a placeholder default — confirm/tune it for the case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keys this script owns (reset before applying the method).
_MANAGED = (
    "enabled",
    "method",
    "check_frequency",
    "tie_tolerance",
    "domination_epsilon",
)

METHODS: dict[str, dict[str, object]] = {
    "none": {"enabled": False},
    "level1": {
        "enabled": True,
        "method": "level1",
        "check_frequency": 5,
        "tie_tolerance": 1e-10,
    },
    "dominated": {
        "enabled": True,
        "method": "domination",
        "check_frequency": 5,
        "domination_epsilon": 1e-6,  # placeholder — confirm for the case
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    training = cfg.setdefault("training", {})
    cut_selection = training.get("cut_selection")
    if not isinstance(cut_selection, dict):
        cut_selection = {}
    for key in _MANAGED:
        cut_selection.pop(key, None)
    cut_selection.update(METHODS[args.method])
    training["cut_selection"] = cut_selection

    args.config.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"patched {args.config} -> cut_selection method={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
