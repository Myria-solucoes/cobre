"""Smoke tests for the cobre Python extension module foundation.

These tests verify that the PyO3 extension module loads correctly and that
the top-level module and its empty sub-modules are importable. They are
intended to be run after `maturin develop --uv` installs the extension.

Run with:
    pytest crates/cobre-python/tests/
"""


def test_import_cobre() -> None:
    import cobre  # noqa: F401, PLC0415


def test_version() -> None:
    import cobre  # noqa: PLC0415

    assert isinstance(cobre.__version__, str)
    assert len(cobre.__version__) > 0


def test_submodules_exist() -> None:
    import cobre.io  # noqa: F401, PLC0415
    import cobre.model  # noqa: F401, PLC0415
    import cobre.results  # noqa: F401, PLC0415
    import cobre.run  # noqa: F401, PLC0415


def test_native_module_importable() -> None:
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
    import cobre  # noqa: PLC0415
    import cobre._native  # noqa: PLC0415

    assert cobre.__version__ == cobre._native.__version__
