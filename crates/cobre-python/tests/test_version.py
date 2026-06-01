"""Tests for the top-level cobre.version_info function.

Verify the dict shape and the single-process invariant (comm is always
"local"), and that the solver string carries a live HiGHS version.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_version.py
"""

from __future__ import annotations


def test_version_info_returns_dict() -> None:
    """version_info returns a dict with the expected keys."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert isinstance(info, dict)
    for key in ("version", "solver", "comm", "zstd", "arch", "build"):
        assert key in info


def test_version_matches_module_version() -> None:
    """version_info()["version"] equals cobre.__version__."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["version"] == cobre.__version__


def test_comm_is_always_local() -> None:
    """The single-process binding always reports comm == "local"."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["comm"] == "local"


def test_solver_string_format() -> None:
    """The solver field starts with "HiGHS " and contains a version dot."""
    import cobre  # noqa: PLC0415

    solver = cobre.version_info()["solver"]

    assert solver.startswith("HiGHS ")
    assert "." in solver


def test_zstd_enabled_and_build_known() -> None:
    """zstd is enabled and build is one of debug/release."""
    import cobre  # noqa: PLC0415

    info = cobre.version_info()

    assert info["zstd"] == "enabled"
    assert info["build"] in ("debug", "release")
