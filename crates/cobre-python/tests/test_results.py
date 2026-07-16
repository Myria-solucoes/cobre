"""Integration tests for cobre.results — result loading and inspection.

These tests verify that, after a completed run, the result loading functions
return correctly-shaped Python objects.

Run with (from the repo root):
    pytest crates/cobre-python/tests/test_results.py

Note: tests that invoke run() write to a temporary directory created by
pytest's tmp_path fixture. The 1dtoy case is small enough that tests complete
in a few seconds.
"""

import pathlib

import pytest


VALID_CASE = "examples/1dtoy"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Run the 1dtoy case once and return the output directory.

    Module-scoped so the solver only runs once per test session.
    """
    import cobre.run  # noqa: PLC0415

    output_dir = tmp_path_factory.mktemp("results_output")
    cobre.run.run(VALID_CASE, output_dir=str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# load_results tests
# ---------------------------------------------------------------------------


def test_load_results_after_run(run_output: pathlib.Path) -> None:
    """load_results() returns a dict with training.complete == True."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))

    assert isinstance(result, dict), "load_results must return a dict"
    assert "training" in result, "result must have 'training' key"
    assert result["training"]["complete"] is True, "training.complete must be True"


def test_load_results_manifest_keys(run_output: pathlib.Path) -> None:
    """result['training']['manifest'] contains required top-level keys."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    manifest = result["training"]["manifest"]

    assert isinstance(manifest, dict), "manifest must be a dict"
    assert "cobre_version" in manifest, "manifest must contain 'cobre_version'"
    assert "status" in manifest, "manifest must contain 'status'"
    assert "convergence" in manifest, "manifest must contain 'convergence'"


def test_load_results_metadata_present(run_output: pathlib.Path) -> None:
    """result['training']['metadata'] is a non-empty dict."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    metadata = result["training"]["metadata"]

    assert isinstance(metadata, dict), "metadata must be a dict"
    assert len(metadata) > 0, "metadata must not be empty"


def test_load_results_convergence_path_is_file(run_output: pathlib.Path) -> None:
    """result['training']['convergence_path'] points to an existing file."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    convergence_path = result["training"]["convergence_path"]

    assert isinstance(convergence_path, str), "convergence_path must be a str"
    assert pathlib.Path(convergence_path).is_file(), (
        f"convergence_path must point to an existing file: {convergence_path}"
    )


def test_load_results_timing_path_is_file(run_output: pathlib.Path) -> None:
    """result['training']['timing_path'] points to an existing file."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    timing_path = result["training"]["timing_path"]

    assert isinstance(timing_path, str), "timing_path must be a str"
    assert pathlib.Path(timing_path).is_file(), (
        f"timing_path must point to an existing file: {timing_path}"
    )


def test_load_results_simulation_section_present(run_output: pathlib.Path) -> None:
    """result['simulation'] is a dict with 'manifest' and 'complete' keys."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))

    assert "simulation" in result, "result must have 'simulation' key"
    sim = result["simulation"]
    assert isinstance(sim, dict), "simulation must be a dict"
    assert "manifest" in sim, "simulation must have 'manifest' key"
    assert "complete" in sim, "simulation must have 'complete' key"


def test_load_results_simulation_ran(
    run_output: pathlib.Path,
) -> None:
    """1dtoy has simulation.enabled=true, so simulation results should exist."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    sim = result["simulation"]
    assert sim["complete"] is True, "simulation must be complete after a successful run"
    assert isinstance(sim["manifest"], dict), "simulation manifest must be a dict"


def test_load_results_no_success_raises(tmp_path: pathlib.Path) -> None:
    """load_results() raises FileNotFoundError when training/_SUCCESS is absent."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results(str(tmp_path))


def test_load_results_nonexistent_dir_raises() -> None:
    """load_results() raises FileNotFoundError for a non-existent directory."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_results("/tmp/nonexistent_cobre_output_xzy123")


# ---------------------------------------------------------------------------
# load_convergence tests
# ---------------------------------------------------------------------------


def test_load_convergence_returns_list(run_output: pathlib.Path) -> None:
    """load_convergence() returns a non-empty list of dicts."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))

    assert isinstance(rows, list), "load_convergence must return a list"
    assert len(rows) > 0, "convergence list must be non-empty after a real run"


def test_load_convergence_dict_keys(run_output: pathlib.Path) -> None:
    """Each dict in the convergence list has the required keys."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    required_keys = {
        "iteration",
        "lower_bound",
        "upper_bound_mean",
        "upper_bound_std",
        "gap_percent",
        "cuts_added",
        "cuts_removed",
        "cuts_active",
        "time_forward_ms",
        "time_backward_ms",
        "time_total_ms",
        "forward_passes",
        "lp_solves",
    }

    for i, row in enumerate(rows):
        assert isinstance(row, dict), f"row {i} must be a dict"
        missing = required_keys - row.keys()
        assert not missing, f"row {i} is missing keys: {missing}"


def test_load_convergence_value_types(run_output: pathlib.Path) -> None:
    """Convergence rows have correct Python types for key columns."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "must have at least one row"

    row = rows[0]
    assert isinstance(row["iteration"], int), "iteration must be int"
    assert isinstance(row["lower_bound"], float), "lower_bound must be float"
    assert isinstance(row["upper_bound_mean"], float), "upper_bound_mean must be float"
    assert isinstance(row["upper_bound_std"], float), "upper_bound_std must be float"
    # gap_percent may be None or float
    assert row["gap_percent"] is None or isinstance(row["gap_percent"], float), (
        "gap_percent must be float or None"
    )
    assert isinstance(row["cuts_added"], int), "cuts_added must be int"
    assert isinstance(row["cuts_active"], int), "cuts_active must be int"
    assert isinstance(row["time_total_ms"], int), "time_total_ms must be int"


def test_load_convergence_iteration_is_one_based(run_output: pathlib.Path) -> None:
    """The first iteration row has iteration == 1."""
    import cobre.results  # noqa: PLC0415

    rows = cobre.results.load_convergence(str(run_output))
    assert rows, "must have at least one row"
    assert rows[0]["iteration"] == 1, "first iteration must be 1-based"


def test_load_convergence_empty_dir_raises(tmp_path: pathlib.Path) -> None:
    """load_convergence() raises FileNotFoundError for a directory without Parquet."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_convergence(str(tmp_path))


def test_convergence_path_is_readable(run_output: pathlib.Path) -> None:
    """The convergence_path from load_results() is a valid, non-empty Parquet path."""
    import cobre.results  # noqa: PLC0415

    result = cobre.results.load_results(str(run_output))
    path = pathlib.Path(result["training"]["convergence_path"])

    assert path.exists(), "convergence_path must exist"
    assert path.stat().st_size > 0, "convergence.parquet must not be empty"


# ---------------------------------------------------------------------------
# load_policy tests
# ---------------------------------------------------------------------------


def test_load_policy_reads_real_run_output(run_output: pathlib.Path) -> None:
    """load_policy() reads the policy checkpoint a standard run writes.

    A run writes the checkpoint to <output_dir>/policy (the default policy_path
    "./policy"), so the default policy_subdir="policy" must resolve it. The
    returned dict must carry per-stage cut pools with non-empty stage-0 cuts.
    """
    import cobre.results  # noqa: PLC0415

    policy = cobre.results.load_policy(str(run_output))

    assert isinstance(policy, dict), "load_policy must return a dict"
    assert "stage_cuts" in policy, "policy dict must have a 'stage_cuts' key"
    assert len(policy["stage_cuts"]) > 0, "a trained policy must have stage cuts"
    assert len(policy["stage_cuts"][0]["cuts"]) > 0, (
        "stage 0 must carry at least one cut after a real run"
    )


def test_load_policy_missing_dir_raises(tmp_path: pathlib.Path) -> None:
    """load_policy() raises FileNotFoundError when the policy dir is absent."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.load_policy(str(tmp_path))


# ---------------------------------------------------------------------------
# report / summary tests
# ---------------------------------------------------------------------------


def test_report_top_level_keys(run_output: pathlib.Path) -> None:
    """report() returns a dict with the six expected top-level keys."""
    import cobre.results  # noqa: PLC0415

    report = cobre.results.report(str(run_output))

    assert isinstance(report, dict), "report must return a dict"
    expected_keys = {
        "output_directory",
        "status",
        "bounds",
        "training",
        "cost",
        "simulation",
    }
    missing = expected_keys - report.keys()
    assert not missing, f"report is missing top-level keys: {missing}"


def test_report_bounds_hoist_is_consistent(run_output: pathlib.Path) -> None:
    """report()['bounds'] mirrors report()['training']['bounds']."""
    import cobre.results  # noqa: PLC0415

    report = cobre.results.report(str(run_output))

    assert (
        report["bounds"]["final_lower_bound"]
        == (report["training"]["bounds"]["final_lower_bound"])
    ), "top-level bounds must match nested training.bounds"


def test_report_cost_hoist_is_consistent(run_output: pathlib.Path) -> None:
    """report()['cost'] mirrors report()['simulation']['cost'] when simulation ran."""
    import cobre.results  # noqa: PLC0415

    report = cobre.results.report(str(run_output))

    assert report["simulation"] is not None, "1dtoy runs simulation"
    assert report["cost"] is not None, "cost must be present when simulation ran"
    assert report["cost"]["mean_cost"] == (report["simulation"]["cost"]["mean_cost"]), (
        "top-level cost must match nested simulation.cost"
    )


def test_report_simulation_none_when_absent(run_output: pathlib.Path) -> None:
    """report() returns None for cost/simulation when simulation metadata is absent."""
    import shutil

    import cobre.results  # noqa: PLC0415

    # Copy the run output and strip the simulation directory.
    stripped = run_output.parent / "report_no_simulation"
    if stripped.exists():
        shutil.rmtree(stripped)
    shutil.copytree(run_output, stripped)
    shutil.rmtree(stripped / "simulation", ignore_errors=True)

    report = cobre.results.report(str(stripped))

    assert report["simulation"] is None, "simulation must be None when metadata absent"
    assert report["cost"] is None, "cost must be None when simulation metadata absent"


def test_report_missing_training_raises(tmp_path: pathlib.Path) -> None:
    """report() raises FileNotFoundError when training/metadata.json is absent."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.report(str(tmp_path))


def test_summary_returns_string(run_output: pathlib.Path) -> None:
    """summary() returns a non-empty str containing a recognizable bounds label."""
    import cobre.results  # noqa: PLC0415

    text = cobre.results.summary(str(run_output))

    assert isinstance(text, str), "summary must return a str"
    assert len(text) > 0, "summary string must be non-empty"
    assert "lower bound" in text.lower(), (
        "summary must include the training bounds section"
    )


def test_report_still_returns_dict(run_output: pathlib.Path) -> None:
    """report() still returns the structured dict with bounds + training keys."""
    import cobre.results  # noqa: PLC0415

    report = cobre.results.report(str(run_output))

    assert isinstance(report, dict), "report must still return a dict"
    assert "bounds" in report, "report must contain 'bounds'"
    assert "training" in report, "report must contain 'training'"


def test_summary_missing_dir_raises() -> None:
    """summary() raises FileNotFoundError, delegated from report()."""
    import cobre.results  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        cobre.results.summary("/tmp/nonexistent_cobre_output_xzy123")


# ---------------------------------------------------------------------------
# load_stochastic tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stochastic_output(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Train the 1dtoy case with ``exports.stochastic`` enabled, once.

    The default 1dtoy config does NOT export stochastic artifacts, so the
    override is required to produce ``stochastic/inflow_ar_coefficients.parquet``
    and ``stochastic/noise_openings.parquet``. Module-scoped so the solver runs
    only once per session.
    """
    import cobre  # noqa: PLC0415

    output_dir = tmp_path_factory.mktemp("stochastic_output")
    cobre.Study(
        VALID_CASE,
        output_dir=str(output_dir),
        config_overrides={"exports.stochastic": True},
    ).train()
    return output_dir


def test_load_stochastic_par_coefficients_shape(
    stochastic_output: pathlib.Path,
) -> None:
    """par_coefficients() returns a 2-D (n_rows, 4) float64 array."""
    numpy = pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    arr = cobre.results.load_stochastic(str(stochastic_output)).par_coefficients()

    assert arr.ndim == 2, "par_coefficients must be 2-D"
    assert arr.shape[1] == 4, "par_coefficients must have 4 columns"
    assert arr.dtype == numpy.float64, "par_coefficients must be float64"


def test_load_stochastic_opening_tree_shape(
    stochastic_output: pathlib.Path,
) -> None:
    """opening_tree(0) returns a 2-D float64 array; shape[1] == stage-0 noise dim."""
    numpy = pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    stoch = cobre.results.load_stochastic(str(stochastic_output))
    arr = stoch.opening_tree(0)

    assert arr.ndim == 2, "opening_tree must be 2-D"
    assert arr.dtype == numpy.float64, "opening_tree must be float64"
    # shape[1] is the number of distinct entity_index values at stage 0, i.e.
    # the noise dimension (1 hydro for the 1dtoy single-reservoir case).
    assert arr.shape[1] >= 1, "noise dimension must be at least 1"
    assert arr.shape[0] >= 1, "stage 0 must have at least one opening"


def test_load_stochastic_missing_artifacts_raises(tmp_path: pathlib.Path) -> None:
    """load_stochastic() raises FileNotFoundError on a default (no-exports) run.

    The 1dtoy default config does not set ``exports.stochastic``, so the
    ``stochastic/`` artifacts are absent and the error message must point the
    caller at the required export flag.
    """
    import cobre  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    cobre.Study(VALID_CASE, output_dir=str(tmp_path)).train()

    with pytest.raises(FileNotFoundError, match="exports.stochastic"):
        cobre.results.load_stochastic(str(tmp_path))


def test_load_stochastic_opening_tree_bad_stage_raises(
    stochastic_output: pathlib.Path,
) -> None:
    """opening_tree(999) raises IndexError for an absent stage."""
    pytest.importorskip("numpy")
    import cobre.results  # noqa: PLC0415

    stoch = cobre.results.load_stochastic(str(stochastic_output))

    with pytest.raises(IndexError):
        stoch.opening_tree(999)


def test_load_stochastic_reexport_identity() -> None:
    """load_stochastic is in __all__ and is the compiled function (identity)."""
    import cobre._native.results  # noqa: PLC0415
    import cobre.results  # noqa: PLC0415

    assert "load_stochastic" in cobre.results.__all__
    assert cobre.results.load_stochastic is cobre._native.results.load_stochastic
