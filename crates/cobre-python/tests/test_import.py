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
    """The public cobre.* submodules ARE the _native compiled modules.

    The package re-exports the compiled submodules rather than copying them, so
    identity must hold for all compiled-only modules.
    """
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.run is cobre._native.run
    assert cobre.io is cobre._native.io
    assert cobre.model is cobre._native.model
    assert cobre.results is cobre._native.results
    assert cobre.schema is cobre._native.schema


def test_version_matches_native() -> None:
    """The public __version__ mirrors the compiled module's __version__."""
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.__version__ == cobre._native.__version__
