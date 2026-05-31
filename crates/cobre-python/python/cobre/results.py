"""Result loading and inspection — the public ``cobre.results`` module.

This pure-Python wrapper sits on top of the compiled ``cobre._native.results``
extension. It re-exports every compiled ``load_*`` function and ``report`` so
that ``cobre.results.load_simulation``, ``cobre.results.report`` and friends keep
resolving exactly as before, and it adds a human-readable :func:`summary`
renderer.

The split exists because :func:`summary` formats a multi-section plain-text
report from the structured dict that the compiled ``report`` produces — a pure
presentation concern best authored in Python. All disk reads stay in Rust:
``summary`` never touches the filesystem itself, it only formats whatever dict
``report`` returns (delegating any path/IO error to ``report``).
"""

from __future__ import annotations

import os
from typing import Any

# Re-export the compiled result-loading surface so the Python wrapper is a
# drop-in replacement for the bare `_native.results` module. No public name may
# regress: every function the compiled module exposed must be visible here.
from cobre._native.results import (
    Stochastic as Stochastic,
    load_convergence as load_convergence,
    load_convergence_arrow as load_convergence_arrow,
    load_policy as load_policy,
    load_results as load_results,
    load_simulation as load_simulation,
    load_simulation_arrow as load_simulation_arrow,
    load_stochastic as load_stochastic,
    report as report,
)

__all__ = [
    "Stochastic",
    "load_results",
    "load_convergence",
    "load_convergence_arrow",
    "load_simulation",
    "load_simulation_arrow",
    "load_policy",
    "load_stochastic",
    "report",
    "summary",
]

# Sentinel rendered in place of a value that the report dict did not carry. Used
# so a missing optional key produces a clearly-labelled absent line rather than a
# KeyError or a misleading numeric default.
_ABSENT = "n/a"


def _fmt_float(value: Any, *, precision: int = 2) -> str:
    """Format ``value`` as a fixed-point float, or ``_ABSENT`` when missing.

    Returns the absent sentinel for ``None`` or for any value that is not a
    real number, so a partially-populated dict never raises.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _ABSENT
    return f"{float(value):.{precision}f}"


def _fmt_int(value: Any) -> str:
    """Format ``value`` as an integer, or ``_ABSENT`` when missing/non-integral."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _ABSENT
    return f"{int(value)}"


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, otherwise an empty dict.

    Lets callers chain ``.get(...)`` on nested sections without first checking
    that the section is present and dict-shaped.
    """
    return value if isinstance(value, dict) else {}


def _render_training(report_dict: dict[str, Any], lines: list[str]) -> None:
    """Append the always-present Training section to ``lines``.

    Headline bounds come from ``report["bounds"]``; iterations and convergence
    come from the full ``report["training"]`` metadata. Every field is read
    defensively, so a missing key renders an absent line rather than raising.
    """
    bounds = _as_mapping(report_dict.get("bounds"))
    training = _as_mapping(report_dict.get("training"))
    iterations = _as_mapping(training.get("iterations"))
    convergence = _as_mapping(training.get("convergence"))

    lines.append("Training")

    lower = _fmt_float(bounds.get("final_lower_bound"))
    lines.append(f"  Final lower bound: {lower}")

    upper = _fmt_float(bounds.get("final_upper_bound"))
    upper_std = bounds.get("final_upper_bound_std")
    if upper_std is None:
        lines.append(f"  Final upper bound: {upper}")
    else:
        lines.append(f"  Final upper bound: {upper} +/- {_fmt_float(upper_std)}")

    gap = convergence.get("final_gap_percent")
    if gap is None:
        lines.append(f"  Gap: {_ABSENT}")
    else:
        lines.append(f"  Gap: {_fmt_float(gap, precision=2)}%")

    lines.append(f"  Iterations: {_fmt_int(iterations.get('completed'))}")

    achieved = convergence.get("achieved")
    if achieved is None:
        converged = _ABSENT
    else:
        converged = "yes" if achieved else "no"
    lines.append(f"  Converged: {converged}")

    reason = convergence.get("termination_reason")
    reason_text = reason if isinstance(reason, str) and reason else _ABSENT
    lines.append(f"  Termination reason: {reason_text}")


def _render_simulation(report_dict: dict[str, Any], lines: list[str]) -> None:
    """Append the Simulation section to ``lines`` when simulation data exists.

    Renders nothing when ``report["simulation"]`` is ``None``/absent. Cost lines
    are emitted only when ``report["cost"]`` is present. All reads are defensive.
    """
    simulation = report_dict.get("simulation")
    if not isinstance(simulation, dict):
        return

    scenarios = _as_mapping(simulation.get("scenarios"))

    lines.append("")
    lines.append("Simulation")
    lines.append(f"  Scenarios total: {_fmt_int(scenarios.get('total'))}")
    lines.append(f"  Scenarios completed: {_fmt_int(scenarios.get('completed'))}")
    lines.append(f"  Scenarios failed: {_fmt_int(scenarios.get('failed'))}")

    cost = report_dict.get("cost")
    if isinstance(cost, dict):
        lines.append(f"  Mean cost: {_fmt_float(cost.get('mean_cost'), precision=5)}")
        lines.append(f"  Std cost: {_fmt_float(cost.get('std_cost'), precision=5)}")
    else:
        lines.append(f"  Mean cost: {_ABSENT}")
        lines.append(f"  Std cost: {_ABSENT}")


def summary(output_dir: str | os.PathLike[str]) -> str:
    """Render a human-readable plain-text summary of a completed run.

    Calls :func:`report` to read the structured run metadata from
    ``output_dir`` (the disk read happens in Rust) and formats the resulting
    dict into a multi-section string: a header with the resolved output
    directory and status, an always-present Training section (final bounds, gap,
    iterations, convergence), and a Simulation section that appears only when the
    run produced simulation metadata.

    All dict keys are read defensively, so a partially-populated report renders a
    labelled ``n/a`` line for any missing optional value rather than raising
    ``KeyError``. Path and I/O errors (a missing or malformed
    ``training/metadata.json``) are delegated unchanged from :func:`report`,
    which raises ``FileNotFoundError`` or ``ValueError`` respectively.

    Args:
        output_dir: Path to a completed run's output directory — the same
            contract as :func:`report`.

    Returns:
        A multi-section human-readable summary string.

    Raises:
        FileNotFoundError: If ``training/metadata.json`` is absent (delegated
            from :func:`report`).
        ValueError: If a metadata file contains malformed JSON (delegated from
            :func:`report`).
        OSError: For other I/O failures (delegated from :func:`report`).
    """
    report_dict = report(output_dir)

    output_directory = report_dict.get("output_directory")
    directory_text = (
        output_directory if isinstance(output_directory, str) else str(output_dir)
    )
    status = report_dict.get("status")
    status_text = status if isinstance(status, str) and status else _ABSENT

    lines: list[str] = [
        f"Run summary for {directory_text}",
        f"Status: {status_text}",
        "",
    ]

    _render_training(report_dict, lines)
    _render_simulation(report_dict, lines)

    return "\n".join(lines)
