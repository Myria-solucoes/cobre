"""Integration tests for cobre.write_policy_checkpoint.

Verifies that a policy checkpoint authored from plain Python dicts round-trips
through cobre.results.load_policy — the read path whose emitted dict shapes
write_policy_checkpoint's input mirrors — and that malformed input raises a
clear error naming the offending stage/cut.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_write_policy_checkpoint.py
"""

import pathlib
from typing import Any, Optional

import pytest


def _make_metadata(cost_scale_factor: Optional[float] = 2_500_000.0) -> dict[str, Any]:
    producer: dict[str, Any] = {
        "completed_iterations": 5,
        "final_lower_bound": 123.45,
        "best_upper_bound": 130.0,
        "max_iterations": 10,
        "forward_passes": 4,
        "warm_start_cuts": 0,
        "warm_start_counts": [0],
        "rng_seed": 42,
        "total_visited_states": 0,
        "training_block_mode": "parallel",
        "training_block_mode_per_stage": [],
    }
    if cost_scale_factor is not None:
        producer["cost_scale_factor"] = cost_scale_factor
    # The neutral core: format_version defaults when omitted; the graph manifest
    # is optional (a checkpoint authored from raw records carries none).
    return {
        "cobre_version": "0.13.0",
        "created_at": "2026-07-30T00:00:00Z",
        "num_stages": 1,
        "producer": producer,
    }


def _make_stage_cuts() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": 0,
            "state_dimension": 3,
            "capacity": 10,
            "cuts": [
                {
                    "cut_id": 1,
                    "slot_index": 0,
                    "iteration": 1,
                    "forward_pass_index": 0,
                    "intercept": 42.0,
                    "coefficients": [1.0, 2.0, 3.0],
                    "is_active": True,
                },
                {
                    "cut_id": 2,
                    "slot_index": 1,
                    "iteration": 1,
                    "forward_pass_index": 1,
                    "intercept": 10.5,
                    "coefficients": [0.5, -1.5, 2.5],
                    "is_active": True,
                },
            ],
        }
    ]


def test_write_policy_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    """A synthetic checkpoint written from dicts reads back with matching cuts
    and metadata, including the cost_scale_factor provenance marker.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), _make_stage_cuts(), _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))

    assert loaded["metadata"]["format_version"] == 1
    assert loaded["metadata"]["producer"]["cost_scale_factor"] == pytest.approx(
        2_500_000.0
    )
    assert loaded["metadata"]["producer"]["completed_iterations"] == 5

    assert len(loaded["stage_cuts"]) == 1
    stage = loaded["stage_cuts"][0]
    assert stage["stage_id"] == 0
    assert stage["state_dimension"] == 3
    assert stage["capacity"] == 10
    assert stage["populated_count"] == 2, "populated_count must default to len(cuts)"

    cuts = stage["cuts"]
    assert len(cuts) == 2
    assert cuts[0]["cut_id"] == 1
    assert cuts[0]["intercept"] == pytest.approx(42.0)
    assert cuts[0]["coefficients"] == pytest.approx([1.0, 2.0, 3.0])
    assert cuts[1]["cut_id"] == 2
    assert cuts[1]["coefficients"] == pytest.approx([0.5, -1.5, 2.5])


def test_write_policy_checkpoint_cost_scale_factor_omitted_reads_as_none(
    tmp_path: pathlib.Path,
) -> None:
    """Omitting cost_scale_factor from the metadata dict reads back as None,
    matching a legacy (pre-marker) checkpoint.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"),
        _make_stage_cuts(),
        _make_metadata(cost_scale_factor=None),
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["metadata"]["producer"]["cost_scale_factor"] is None


def test_write_policy_checkpoint_coefficient_length_mismatch_raises(
    tmp_path: pathlib.Path,
) -> None:
    """A cut whose coefficients length disagrees with its stage's
    state_dimension raises ValueError naming the stage and cut.
    """
    import cobre  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    stage_cuts[0]["cuts"][0]["coefficients"] = [1.0, 2.0]  # state_dimension is 3

    with pytest.raises(ValueError, match=r"stage 0 cut 1.*coefficients"):
        cobre.write_policy_checkpoint(
            str(tmp_path / "policy"), stage_cuts, _make_metadata()
        )


def test_write_policy_checkpoint_defaults_apply_when_keys_omitted(
    tmp_path: pathlib.Path,
) -> None:
    """warm_start_count, active_cut_indices, and entity_manifest all default
    when their keys are absent from the stage_cuts dict.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    stage_cuts = _make_stage_cuts()
    assert "warm_start_count" not in stage_cuts[0]
    assert "active_cut_indices" not in stage_cuts[0]
    assert "entity_manifest" not in stage_cuts[0]

    cobre.write_policy_checkpoint(
        str(tmp_path / "policy"), stage_cuts, _make_metadata()
    )

    loaded = cobre.results.load_policy(str(tmp_path))
    assert loaded["stage_cuts"][0]["warm_start_count"] == 0
