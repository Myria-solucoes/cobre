"""Tests for error propagation across the FFI boundary.

These tests verify that Rust errors are correctly translated into appropriate
Python exceptions with meaningful error messages.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_errors.py
"""

import pathlib
import shutil
import struct
import sys
from typing import Any

import pytest


MISSING_CASE = "/tmp/nonexistent_cobre_case_xzy123"


@pytest.fixture(scope="module")
def policy_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    import cobre.run

    output = tmp_path_factory.mktemp("checkpoint-error-source")
    cobre.run.run(
        "examples/1dtoy",
        output_dir=str(output),
        threads=1,
        config_overrides={
            "training.stopping_rules": [{"type": "iteration_limit", "limit": 2}],
            "simulation.enabled": False,
        },
    )
    return output


@pytest.mark.parametrize(
    ("entry_point", "fault"),
    [
        (entry, fault)
        for entry in (
            "results", "study_load", "run_simulation", "run_warm_start",
            "study_warm_start", "run_resume", "study_resume",
        )
        for fault in (
            "missing_directory", "missing_manifest", "unsupported_version",
            "corrupt_manifest", "manifest_is_directory", "incompatible_dimension",
        )
        if not (entry == "results" and fault == "incompatible_dimension")
    ],
)
def test_checkpoint_errors_preserve_kind_across_entry_points(
    tmp_path: pathlib.Path, policy_checkpoint: pathlib.Path,
    entry_point: str, fault: str,
) -> None:
    import cobre
    import cobre.errors
    import cobre.results
    import cobre.run

    policy_dir = tmp_path / "policy"
    if fault != "missing_directory":
        shutil.copytree(policy_checkpoint / "policy", policy_dir)
    if fault == "missing_manifest":
        (policy_dir / "manifest.bin").unlink()
    elif fault == "manifest_is_directory":
        (policy_dir / "manifest.bin").unlink()
        (policy_dir / "manifest.bin").mkdir()
    elif fault == "corrupt_manifest":
        (policy_dir / "manifest.bin").write_bytes(b"invalid checkpoint")
    elif fault == "unsupported_version":
        path = policy_dir / "manifest.bin"
        data = bytearray(path.read_bytes())
        table = struct.unpack_from("<I", data, 0)[0]
        vtable = table - struct.unpack_from("<i", data, table)[0]
        version_offset = struct.unpack_from("<H", data, vtable + 4)[0]
        assert version_offset != 0
        struct.pack_into("<I", data, table + version_offset, 999)
        path.write_bytes(data)
    elif fault == "incompatible_dimension":
        checkpoint = cobre.results.load_policy(str(tmp_path))
        for stage in checkpoint["stage_cuts"]:
            stage["state_dimension"] += 1
            stage["entity_manifest"] = []
            for cut in stage["cuts"]:
                cut["coefficients"].append(0.0)
        cobre.write_policy_checkpoint(
            str(policy_dir), checkpoint["stage_cuts"], checkpoint["metadata"],
        )
        # The artifact is readable; only loading it against this study rejects it.
        cobre.results.load_policy(str(tmp_path))

    expected = (
        FileNotFoundError if fault.startswith("missing_")
        else cobre.errors.PolicyIncompatibleError if fault == "incompatible_dimension"
        else cobre.errors.CaseIoError if fault == "manifest_is_directory"
        else cobre.errors.OutputError
    )
    overrides: dict[str, Any] = {"simulation.enabled": False}
    if entry_point == "run_simulation":
        overrides = {"training.enabled": False, "simulation.enabled": True}
    elif entry_point.endswith("warm_start"):
        overrides["policy.mode"] = "warm_start"
    elif entry_point.endswith("resume"):
        overrides["policy.mode"] = "resume"

    with pytest.raises(expected) as caught:
        if entry_point == "results":
            cobre.results.load_policy(str(tmp_path))
        elif entry_point.startswith("run_"):
            cobre.run.run("examples/1dtoy", output_dir=str(tmp_path),
                          threads=1, config_overrides=overrides)
        else:
            study = cobre.Study("examples/1dtoy", output_dir=str(tmp_path),
                                threads=1, config_overrides=overrides)
            if entry_point == "study_load":
                study.load_policy()
            else:
                study.train()
    assert type(caught.value) is expected
    if fault == "incompatible_dimension":
        message = str(caught.value)
        assert "state_dimension" in message
        assert "policy has" in message and "system has" in message
    elif fault == "unsupported_version":
        assert "format_version 999" in str(caught.value)


def test_errors_importable_and_subclass_builtins() -> None:
    """The eight classes import from cobre.errors and subclass the right builtins."""
    import cobre.errors as e  # noqa: PLC0415

    assert e.CobreError is not None
    assert e.ValidationError is not None
    assert e.CaseIoError is not None
    assert e.PolicyIncompatibleError is not None
    assert e.SolverError is not None
    assert e.SimulationError is not None
    assert e.OutputError is not None
    assert e.InternalError is not None

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
    assert issubclass(e.InternalError, RuntimeError)
    assert issubclass(e.InternalError, e.CobreError)
    # InternalError is a sibling, not a subclass of SolverError.
    assert not issubclass(e.InternalError, e.SolverError)

    # Qualified names read cobre.errors.<Name> (so tracebacks are unambiguous).
    assert e.SolverError.__module__ == "cobre.errors"
    assert e.SolverError.__qualname__ == "SolverError"
    assert e.InternalError.__module__ == "cobre.errors"


def test_run_nonexistent_dir_raises_oserror(tmp_path: pathlib.Path) -> None:
    """run() raises OSError with a descriptive message for a non-existent directory."""
    import cobre.run  # noqa: PLC0415

    with pytest.raises(OSError, match="does not exist"):
        cobre.run.run(MISSING_CASE, output_dir=str(tmp_path))


def test_run_empty_dir_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """run() raises ValidationError for a directory missing required case files."""
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(cobre.errors.ValidationError, match="constraint violation"):
        cobre.run.run(str(empty_case), output_dir=str(output))

    # The same failure is catchable as the builtin ValueError (dual base intact).
    with pytest.raises(ValueError):
        cobre.run.run(str(empty_case), output_dir=str(output))


def test_run_empty_dir_error_mentions_missing_files(tmp_path: pathlib.Path) -> None:
    """The ValidationError for an empty case lists specific missing files."""
    import cobre.errors  # noqa: PLC0415
    import cobre.run  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(cobre.errors.ValidationError) as exc_info:
        cobre.run.run(str(empty_case), output_dir=str(output))

    msg = str(exc_info.value)
    assert "config.json" in msg, "error must mention config.json"
    assert "FileNotFound" in msg, "error must include FileNotFound kind"


def test_study_empty_dir_raises_validation_error(tmp_path: pathlib.Path) -> None:
    """Study() construction raises ValidationError for a directory missing required case files.

    Proves that the constructor and run() now agree on the exception class.
    """
    import cobre.errors  # noqa: PLC0415

    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()
    output = tmp_path / "output"

    with pytest.raises(cobre.errors.ValidationError, match="constraint violation"):
        cobre.Study(str(empty_case), output_dir=str(output))


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

    with pytest.raises(cobre.errors.ValidationError, match="constraint violation"):
        cobre.io.load_case(str(empty_case))

    # The same failure is catchable as the builtin ValueError (dual base intact)
    # and as the common CobreError base.
    with pytest.raises(ValueError):
        cobre.io.load_case(str(empty_case))
    with pytest.raises(cobre.errors.CobreError):
        cobre.io.load_case(str(empty_case))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="forces the write failure via os.chmod(dir, 0o500), which does not make "
    "a directory read-only on Windows, so the output-write error path cannot be "
    "exercised here; cobre.run's CaseIoError behavior itself is platform-independent",
)
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
        with pytest.raises(cobre.errors.CaseIoError, match="output write error"):
            cobre.run.run(str(case_dir), output_dir=str(out))

        # The same failure is catchable as the builtin OSError (dual base intact).
        with pytest.raises(OSError):
            cobre.run.run(str(case_dir), output_dir=str(out))
    finally:
        os.chmod(out, 0o700)
