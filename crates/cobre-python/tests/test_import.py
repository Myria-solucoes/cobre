"""Smoke tests for the cobre Python extension module foundation.

These tests verify that the PyO3 extension module loads correctly and that
the top-level module and its empty sub-modules are importable. They are
intended to be run after `maturin develop --uv` installs the extension.

Run with:
    pytest crates/cobre-python/tests/
"""


def test_import_cobre() -> None:
    """Importing cobre must succeed without errors."""
    import cobre  # noqa: F401, PLC0415


def test_version() -> None:
    """cobre.__version__ must be a non-empty string."""
    import cobre  # noqa: PLC0415

    assert isinstance(cobre.__version__, str)
    assert len(cobre.__version__) > 0


def test_submodules_exist() -> None:
    """cobre.model, cobre.io, cobre.run, and cobre.results must be importable."""
    import cobre.io  # noqa: F401, PLC0415
    import cobre.model  # noqa: F401, PLC0415
    import cobre.results  # noqa: F401, PLC0415
    import cobre.run  # noqa: F401, PLC0415


def test_native_module_importable() -> None:
    """The private compiled module cobre._native must import cleanly."""
    import cobre._native  # noqa: F401, PLC0415


def test_public_submodules_are_the_native_modules() -> None:
    """The compiled-only public cobre.* submodules ARE the _native modules.

    The package re-exports the compiled submodules rather than copying them, so
    identity must hold for the modules that have no Python wrapper. ``results``
    is the exception: it is a pure-Python wrapper (``results.py``) that
    re-exports the compiled ``load_*``/``report`` and adds the ``summary``
    renderer, so it is intentionally NOT the compiled module — see
    :func:`test_results_is_python_wrapper`.
    """
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.run is cobre._native.run
    assert cobre.io is cobre._native.io
    assert cobre.model is cobre._native.model
    assert cobre.schema is cobre._native.schema


def test_results_is_python_wrapper() -> None:
    """The public cobre.results is the Python wrapper, not the compiled module.

    Requirement 3 of the summary-renderer work makes ``cobre.results`` resolve to
    the pure-Python ``results.py`` wrapper. The wrapper must re-export every
    compiled ``load_*``/``report`` function (so no public name regresses) and add
    the ``summary`` renderer that the compiled module does not provide.
    """
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.results is not cobre._native.results, (
        "cobre.results must be the Python wrapper, not the compiled module"
    )
    assert getattr(cobre.results, "__file__", "").endswith("results.py"), (
        "cobre.results must resolve to the results.py wrapper"
    )
    # The wrapper adds `summary`, which the compiled module no longer exposes.
    assert hasattr(cobre.results, "summary")
    assert not hasattr(cobre._native.results, "summary")
    # No public name regresses: every compiled result loader is re-exported.
    for name in (
        "load_results",
        "load_convergence",
        "load_convergence_arrow",
        "load_simulation",
        "load_simulation_arrow",
        "load_policy",
        "report",
    ):
        assert hasattr(cobre.results, name), f"wrapper must re-export {name}"
        assert getattr(cobre.results, name) is getattr(cobre._native.results, name), (
            f"{name} must be the same compiled function object"
        )


def test_version_matches_native() -> None:
    """The public __version__ mirrors the compiled module's __version__."""
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.__version__ == cobre._native.__version__
