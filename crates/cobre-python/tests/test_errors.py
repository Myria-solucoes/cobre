"""Tests for error propagation across the FFI boundary.

These tests verify that Rust errors are correctly translated into appropriate
Python exceptions with meaningful error messages.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_errors.py
"""

import pathlib

import pytest


MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


def test_errors_importable_and_subclass_builtins() -> None:
    """The seven classes import from cobre.errors and subclass the right builtins."""
    import cobre.errors as e  # noqa: PLC0415

    # All seven names resolve from cobre.errors.
    assert e.CobreError is not None
    assert e.ValidationError is not None
    assert e.CaseIoError is not None
    assert e.PolicyIncompatibleError is not None
    assert e.SolverError is not None
    assert e.SimulationError is not None
    assert e.OutputError is not None

    # CobreError is the common base.
    assert issubclass(e.CobreError, Exception)

    # Dual-base subclassing: the matching builtin AND CobreError.
    assert issubclass(e.ValidationError, ValueError)
    assert issubclass(e.ValidationError, e.CobreError)
    assert issubclass(e.PolicyIncompatibleError, ValueError)
    assert issubclass(e.PolicyIncompatibleError, e.CobreError)
    assert issubclass(e.CaseIoError, OSError)
    assert issubclass(e.CaseIoError, e.CobreError)
    assert issubclass(e.OutputError, OSError)
    assert issubclass(e.OutputError, e.CobreError)
    assert issubclass(e.SolverError, RuntimeError)
    assert issubclass(e.SolverError, e.CobreError)
    assert issubclass(e.SimulationError, RuntimeError)
    assert issubclass(e.SimulationError, e.CobreError)

    # Qualified names read cobre.errors.<Name> (so tracebacks are unambiguous).
    assert e.SolverError.__module__ == "cobre.errors"
    assert e.SolverError.__qualname__ == "SolverError"


def test_run_nonexistent_dir_raises_oserror(tmp_path: pathlib.Path) -> None:
    """run() raises OSError with a descriptive message for a non-existent directory."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        cobre.run.run(MISSING_CASE, output_dir=str(tmp_path))


def test_run_empty_dir_raises_runtime_error(tmp_path: pathlib.Path) -> None:
    """run() raises RuntimeError for a directory missing required case files."""
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="constraint violation"):
        cobre.run.run(str(empty_case), output_dir=str(output))


def test_run_empty_dir_error_mentions_missing_files(tmp_path: pathlib.Path) -> None:
    """The RuntimeError for an empty case lists specific missing files."""
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(RuntimeError) as exc_info:
        cobre.run.run(str(empty_case), output_dir=str(output))

    msg = str(exc_info.value)
    assert "config.json" in msg, "error must mention config.json"
    assert "FileNotFound" in msg, "error must include FileNotFound kind"


def test_load_case_nonexistent_raises_oserror() -> None:
    """load_case raises OSError for a non-existent path."""
    import cobre.io  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        cobre.io.load_case(MISSING_CASE)


def test_load_results_empty_dir_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    """load_results raises FileNotFoundError for a directory without _SUCCESS."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results(str(tmp_path))


def test_validation_failure_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """A schema/constraint load failure raises ValidationError, also a ValueError.

    An empty case directory fails the structural-validation layer with a
    constraint violation, which maps to ``cobre.errors.ValidationError``.
    """
    import cobre.errors  # noqa: PLC0415
    import cobre.io  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()

    # Raises ValidationError with the verbatim "constraint violation" message.
    with pytest.raises(cobre.errors.ValidationError, match="constraint violation"):
        cobre.io.load_case(str(empty_case))

    # The same failure is catchable as the builtin ValueError (dual base intact)
    # and as the common CobreError base.
    with pytest.raises(ValueError):
        cobre.io.load_case(str(empty_case))
    with pytest.raises(cobre.errors.CobreError):
        cobre.io.load_case(str(empty_case))


def test_io_failure_raises_caseio_error(tmp_path: pathlib.Path) -> None:
    """An output-write I/O failure raises CaseIoError, also catchable as OSError.

    Running 1dtoy against a read-only output directory makes the first sidecar
    write fail with an "output write error", which maps to
    ``cobre.errors.CaseIoError``.
    """
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    case_dir = pathlib.Path(__file__).resolve().parents[3] / "examples" / "1dtoy"
    if not case_dir.exists():
        pytest.skip(f"examples/1dtoy not found at {case_dir}")

    out = tmp_path / "ro_output"
    out.mkdir()
    import os  # noqa: PLC0415

    os.chmod(out, 0o500)  # read + execute, no write
    try:
        # Raises CaseIoError carrying the verbatim "output write error" message.
        with pytest.raises(cobre.errors.CaseIoError, match="output write error"):
            cobre.run.run(str(case_dir), output_dir=str(out))

        # The same failure is catchable as the builtin OSError (dual base intact).
        with pytest.raises(OSError):
            cobre.run.run(str(case_dir), output_dir=str(out))
    finally:
        os.chmod(out, 0o700)
