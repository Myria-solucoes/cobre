"""Tests for the cobre.io.validate full pre-solver pipeline.

Verifies that validate() exercises all ten phases (path check + six cobre-io
layers + three SDDP preparation phases) and returns a correctly shaped result
dict without ever raising.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_validate.py -v
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import pytest

VALID_CASE_1DTOY = "examples/1dtoy"
VALID_CASE_4REE = "examples/4ree"


# ── helper ────────────────────────────────────────────────────────────────────


def copy_case_to_tempdir(src: str) -> pathlib.Path:
    """Copy a case directory to a fresh temp dir and return the new path."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    dest = tmp / pathlib.Path(src).name
    shutil.copytree(src, dest)
    return dest


# ── result shape contract ─────────────────────────────────────────────────────


def test_validate_result_has_required_keys() -> None:
    """validate() always returns a dict with valid, errors, and warnings keys."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "errors" in result
    assert "warnings" in result


def test_validate_warning_entries_have_required_fields() -> None:
    """Each warning entry has kind, message, file, and entity fields."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    for w in result["warnings"]:
        assert "kind" in w, f"missing 'kind' in warning: {w}"
        assert "message" in w, f"missing 'message' in warning: {w}"
        assert "file" in w, f"missing 'file' in warning: {w}"
        assert "entity" in w, f"missing 'entity' in warning: {w}"


# ── clean-case tests ──────────────────────────────────────────────────────────


def test_validate_clean_case_returns_valid_true() -> None:
    """validate(examples/1dtoy) returns valid=True with no errors."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert result["valid"] is True, (
        f"expected valid=True, got errors: {result['errors']}"
    )
    assert result["errors"] == []


def test_validate_emits_penalty_ordering_warning() -> None:
    """validate() surfaces a penalty-ordering warning end-to-end.

    Rather than relying on a shipped example happening to violate the penalty
    hierarchy (which is curated to be well-formed), this constructs a case that
    deliberately re-inverts the ordering: the bus deficit-segment cost is set
    BELOW the maximum hydro constraint-violation cost (``evaporation_violation_cost``
    stays at 5000). That trips semantic "Check 8"
    (``max(deficit_segment_costs) <= max(constraint_violation_costs)``), which
    emits a ``ModelQuality`` penalty-ordering warning.

    This pins the warnings plumbing (semantic layer → ReportEntry → Python dict)
    end-to-end and is robust to future curation of the shipped example. A
    warning must NOT invalidate the case, so ``valid`` stays ``True``.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        # Re-invert the penalty hierarchy: drive the bus deficit-segment cost
        # below the max constraint-violation cost (evaporation_violation_cost =
        # 5000 in penalties.json). 1dtoy carries an entity-level deficit segment
        # on its single bus, so system/buses.json is the resolved cost source.
        buses_path = case_dir / "system" / "buses.json"
        with buses_path.open() as f:
            buses = json.load(f)
        for bus in buses["buses"]:
            for segment in bus["deficit_segments"]:
                segment["cost"] = 1000.0
        with buses_path.open("w") as f:
            json.dump(buses, f, indent=2)

        result = cobre.io.validate(str(case_dir))

        # Warnings never invalidate a case.
        assert result["valid"] is True, (
            f"warnings must not invalidate; got errors: {result['errors']}"
        )
        assert isinstance(result["warnings"], list)
        assert len(result["warnings"]) >= 1, (
            "expected at least one warning after re-inverting the penalty "
            "hierarchy, got zero — warnings are being dropped"
        )

        penalty_warnings = [
            w
            for w in result["warnings"]
            if w["kind"] == "ModelQuality"
            and (
                "penalty" in w["message"].lower() or "ordering" in w["message"].lower()
            )
        ]
        assert penalty_warnings, (
            "expected a ModelQuality penalty-ordering warning, got: "
            f"{result['warnings']}"
        )
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


def test_validate_clean_case_accepts_pathlib_path() -> None:
    """validate() accepts a pathlib.Path, not just a str."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(pathlib.Path(VALID_CASE_1DTOY))
    assert result["valid"] is True


def test_validate_4ree_warnings_populated() -> None:
    """validate(examples/4ree) also populates warnings from the pipeline."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_4REE)
    assert result["valid"] is True, f"4ree should be valid, errors: {result['errors']}"
    # 4ree may emit zero warnings — we only assert the key is present and a list.
    assert isinstance(result["warnings"], list)


# ── missing-directory test ────────────────────────────────────────────────────


def test_validate_missing_directory_returns_invalid() -> None:
    """validate() returns valid=False for a non-existent path without raising."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate("/tmp/nonexistent_cobre_case_abc999xyz")
    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    err = result["errors"][0]
    assert "kind" in err
    assert "message" in err


# ── Phase 8: StudyParams::from_config (ConfigValidationError) ─────────────────


def test_validate_basis_activity_window_is_rejected() -> None:
    """basis_activity_window was removed in the cut_selection restructure.

    cut_selection (RowSelectionConfig) uses deny_unknown_fields, so a config
    that still carries the removed basis_activity_window field fails to load.
    validate() must surface this as valid=False and name the offending field,
    rather than silently ignoring it.
    """
    import cobre.io  # noqa: PLC0415

    case_dir = copy_case_to_tempdir(VALID_CASE_1DTOY)
    try:
        config_path = case_dir / "config.json"
        with config_path.open() as f:
            config = json.load(f)

        config.setdefault("training", {}).setdefault("cut_selection", {})[
            "basis_activity_window"
        ] = 100

        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        result = cobre.io.validate(str(case_dir))
        assert result["valid"] is False, (
            "expected valid=False for the removed basis_activity_window field, "
            f"got: {result!r}"
        )
        assert any(
            "basis_activity_window" in err["message"] for err in result["errors"]
        ), f"an error must name the removed field, got: {result['errors']!r}"
    finally:
        shutil.rmtree(case_dir.parent, ignore_errors=True)


# ── Phase 1-6: cobre-io pipeline (ConstraintError / IoError) ──────────────────


def test_validate_missing_config_returns_error() -> None:
    """An empty directory (no config.json) returns valid=False with an error."""
    import cobre.io  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        result = cobre.io.validate(tmp)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
        err = result["errors"][0]
        # The cobre-io structural layer reports missing required files as
        # ConstraintError (aggregated) or IoError (direct read failure).
        assert err["kind"] in (
            "ConstraintError",
            "IoError",
            "ParseError",
            "SchemaError",
        ), f"unexpected kind for missing-config case: {err['kind']!r}"


def test_validate_never_raises_for_missing_case() -> None:
    """validate() must not raise any exception, even for bad inputs."""
    import cobre.io  # noqa: PLC0415

    # Should not raise — must return a dict with valid=False.
    result = cobre.io.validate("/dev/null/this/cannot/exist")
    assert isinstance(result, dict)
    assert result["valid"] is False


def test_validate_never_raises_for_valid_case() -> None:
    """validate() must not raise any exception for a valid case."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY)
    assert isinstance(result, dict)


# ── config_overrides ──────────────────────────────────────────────────────────


def test_validate_config_overrides_none_is_valid() -> None:
    """config_overrides=None reproduces the default valid result."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(VALID_CASE_1DTOY, config_overrides=None)
    assert result["valid"] is True, f"errors: {result['errors']}"


def test_validate_config_overrides_invalid_value_surfaces_schema_error() -> None:
    """A semantically-invalid override surfaces in the dict as a SchemaError, not a raise.

    Setting `historical_years` while no scenario class uses the `historical`
    scheme violates validate_config, which `Config::with_overrides` runs
    identically to an edited config.json. 1dtoy uses `in_sample` for every
    class, so the override is rejected with the offending field named.
    """
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(
        VALID_CASE_1DTOY,
        config_overrides={"training.scenario_source.historical_years": [1990, 1991]},
    )
    assert result["valid"] is False, "an invalid override must mark the case invalid"
    assert len(result["errors"]) >= 1
    err = result["errors"][0]
    assert err["kind"] == "SchemaError", f"expected SchemaError, got {err['kind']!r}"
    assert "historical_years" in err["message"], (
        f"error message must name the offending field, got: {err['message']!r}"
    )


def test_validate_config_overrides_typo_surfaces_schema_error() -> None:
    """A typo override key is rejected via deny_unknown_fields as a SchemaError."""
    import cobre.io  # noqa: PLC0415

    result = cobre.io.validate(
        VALID_CASE_1DTOY,
        config_overrides={"trainning.tree_seed": 7},
    )
    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    assert result["errors"][0]["kind"] == "SchemaError"


def test_validate_config_overrides_unsupported_value_raises_value_error() -> None:
    """A malformed override value (no JSON form) raises ValueError before merging.

    Unlike case-validation failures (returned as data), a malformed call payload
    is a programming error and is raised under the GIL before py.detach.
    """
    import cobre.io  # noqa: PLC0415

    with pytest.raises(ValueError):
        cobre.io.validate(
            VALID_CASE_1DTOY,
            config_overrides={"training.cut_selection.row_activity_tolerance": {1, 2}},
        )
