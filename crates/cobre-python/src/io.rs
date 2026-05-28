//! I/O helpers for loading Cobre case directories from Python.
//!
//! Exposes [`load_case`] and [`validate`] in the `cobre.io` sub-module.
//! These are the primary entry points for Python scripts and Jupyter notebooks
//! that need to read and inspect Cobre power-system cases.
//!
//! ## Error mapping
//!
//! [`cobre_io::LoadError`] variants are converted to Python exceptions as follows:
//!
//! | Rust variant                        | Python exception |
//! |-------------------------------------|-----------------|
//! | `LoadError::IoError`                | `OSError`       |
//! | `LoadError::ParseError`             | `ValueError`    |
//! | `LoadError::SchemaError`            | `ValueError`    |
//! | `LoadError::CrossReferenceError`    | `ValueError`    |
//! | `LoadError::ConstraintError`        | `ValueError`    |
//! | `LoadError::PolicyIncompatible`     | `ValueError`    |
//!
//! The [`validate`] function never raises — errors are returned as data in a
//! Python dict so that callers see all problems at once.

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cobre_io::LoadError;
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::validate_phases::{prep_phase_metadata, PrepPhase};
use cobre_sddp::{prepare_stochastic, StudyParams};

use crate::model::PySystem;

// ── Error conversion ──────────────────────────────────────────────────────────

/// Map a [`LoadError`] variant to its `&'static str` kind name.
fn load_error_kind(err: &LoadError) -> &'static str {
    match err {
        LoadError::IoError { .. } => "IoError",
        LoadError::ParseError { .. } => "ParseError",
        LoadError::SchemaError { .. } => "SchemaError",
        LoadError::CrossReferenceError { .. } => "CrossReferenceError",
        LoadError::ConstraintError { .. } => "ConstraintError",
        LoadError::PolicyIncompatible { .. } => "PolicyIncompatible",
    }
}

/// Convert a [`LoadError`] to the appropriate Python exception.
///
/// The mapping preserves as much context as possible in the exception message
/// without exposing Rust-internal type names in the Python API.
fn convert_load_error(err: &LoadError) -> PyErr {
    match err {
        LoadError::IoError { .. } => PyOSError::new_err(err.to_string()),
        LoadError::ParseError { .. }
        | LoadError::SchemaError { .. }
        | LoadError::CrossReferenceError { .. }
        | LoadError::ConstraintError { .. }
        | LoadError::PolicyIncompatible { .. } => PyValueError::new_err(err.to_string()),
    }
}

// ── load_case ────────────────────────────────────────────────────────────────

/// Load a Cobre case directory and return a validated `System`.
///
/// Executes the six-layer validation pipeline (structural, schema, referential
/// integrity, dimensional consistency, semantic, and cross-file resolution).
/// Returns a fully-validated `cobre.model.System` on success or raises a Python
/// exception on failure.
///
/// # Arguments
///
/// * `path` — path to the case directory, as a `str` or `pathlib.Path`.
///   Relative paths are resolved from the process working directory.
///
/// # Raises
///
/// * `OSError` — a required file is missing or cannot be read.
/// * `ValueError` — the case data fails schema, referential integrity,
///   dimensional consistency, or semantic validation.
///
/// # Examples
///
/// ```python
/// import cobre.io
/// system = cobre.io.load_case("examples/1dtoy")
/// print(system.n_buses)
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn load_case(path: PathBuf) -> PyResult<PySystem> {
    if !path.exists() {
        return Err(PyOSError::new_err(format!(
            "case directory does not exist: {}",
            path.display()
        )));
    }
    let system = cobre_io::load_case(&path).map_err(|e| convert_load_error(&e))?;
    Ok(PySystem::from_rust(system))
}

// ── validate ─────────────────────────────────────────────────────────────────

/// Validate a Cobre case directory and return a structured report dict.
///
/// Unlike [`load_case`], this function **never raises** — all errors are
/// returned as data in the result dict. This is intentional: Jupyter workflows
/// need to see all validation problems at once rather than stopping at the
/// first failure.
///
/// The pipeline executes ten phases:
///
/// 1. Path existence check
/// 2–6. `cobre-io` six-layer pipeline (structural, schema, referential,
///      dimensional, semantic, cross-file resolution)
/// 7. `config.json` parse
/// 8. [`StudyParams::from_config`] — validates solver-level config fields
///    and surfaces deprecation warnings for fields scheduled for removal
/// 9. [`prepare_stochastic`] — PAR estimation, opening trees, stochastic context
/// 10. [`prepare_hydro_models_from_artifacts`] — production/evaporation models
///
/// If any phase fails, the remaining phases are skipped and the error is
/// returned immediately (short-circuit semantics matching `cobre validate`).
///
/// # Arguments
///
/// * `path` — path to the case directory, as a `str` or `pathlib.Path`.
///
/// # Returns
///
/// A dict with the following keys:
///
/// * `"valid"` (`bool`) — `True` when all ten phases completed without errors.
/// * `"errors"` (`list[dict]`) — list of error dicts, each with `"kind"` and
///   `"message"` string fields. Empty when `valid` is `True`.
/// * `"warnings"` (`list[dict]`) — list of warning dicts, each with `"kind"`,
///   `"message"`, `"file"`, and `"entity"` string fields.
///   Warnings do not affect the `valid` flag.
///
/// # Examples
///
/// ```python
/// import cobre.io
/// result = cobre.io.validate("examples/1dtoy")
/// assert result["valid"] is True
/// assert result["errors"] == []
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn validate(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    /// Short-circuit helper: populate the result dict with a single error entry
    /// and return immediately. Warnings are always empty on error paths because
    /// the pipeline aborts before the warning-collection stage.
    macro_rules! return_error {
        ($kind:expr, $message:expr) => {{
            dict.set_item("valid", false)?;
            let entry = PyDict::new(py);
            entry.set_item("kind", $kind)?;
            entry.set_item("message", $message)?;
            let errors = PyList::new(py, [entry.as_any()])?;
            dict.set_item("errors", errors)?;
            dict.set_item("warnings", PyList::empty(py))?;
            return Ok(dict.into());
        }};
    }

    // Phase 0: path existence check.
    if !path.exists() {
        return_error!(
            "IoError",
            format!("case directory does not exist: {}", path.display())
        );
    }

    // Phases 1–6: cobre-io six-layer validation pipeline.
    // Use the _with_artifacts variant to obtain the pre-parsed CaseArtifacts
    // bundle needed by phase 10 without re-reading disk.
    let (loaded, report) = match cobre_io::validate_case_with_artifacts(&path) {
        Ok(result) => result,
        Err(err) => {
            return_error!(load_error_kind(&err), err.to_string());
        }
    };

    let system = loaded.system;
    let artifacts = loaded.artifacts;

    // Phase 7: parse config.json.
    let config_path = path.join("config.json");
    let config = match cobre_io::parse_config(&config_path) {
        Ok(c) => c,
        Err(ref err) => {
            return_error!(load_error_kind(err), err.to_string());
        }
    };

    // Phase 8: StudyParams::from_config — validates solver-level fields such as
    // stopping rules and cut-selection parameters that are only checked at
    // algorithm startup, and emits deprecation warnings for fields scheduled
    // for removal.
    let study_params = match StudyParams::from_config(&config) {
        Ok(p) => p,
        Err(ref err) => {
            let (kind, file_label) = prep_phase_metadata(PrepPhase::Config, err);
            return_error!(kind, format!("{file_label}: {err}"));
        }
    };

    let seed = study_params.seed;

    // Extract the training scenario source (needed by prepare_stochastic).
    let training_source = match config.training_scenario_source(&config_path) {
        Ok(s) => s,
        Err(ref err) => {
            return_error!(load_error_kind(err), err.to_string());
        }
    };

    // Phase 9: prepare_stochastic — runs PAR estimation from inflow history,
    // loads user opening trees, builds the stochastic context.
    let prepared = match prepare_stochastic(system, &path, &config, seed, &training_source) {
        Ok(p) => p,
        Err(ref err) => {
            let (kind, file_label) = prep_phase_metadata(PrepPhase::Stochastic, err);
            return_error!(kind, format!("{file_label}: {err}"));
        }
    };

    // Phase 10: prepare_hydro_models_from_artifacts — resolves production and
    // evaporation models. Uses the pre-parsed artifacts bundle from phases 1–6.
    if let Err(ref err) = prepare_hydro_models_from_artifacts(&prepared.system, &artifacts) {
        let (kind, file_label) = prep_phase_metadata(PrepPhase::HydroModels, err);
        return_error!(kind, format!("{file_label}: {err}"));
    }

    // All phases passed — populate warnings from the cobre-io pipeline report.
    dict.set_item("valid", true)?;
    dict.set_item("errors", PyList::empty(py))?;

    let warnings_list = PyList::empty(py);
    for entry in &report.warnings {
        let w = PyDict::new(py);
        w.set_item("kind", &entry.kind)?;
        w.set_item("message", &entry.message)?;
        w.set_item("file", &entry.file)?;
        w.set_item("entity", entry.entity.as_deref())?;
        warnings_list.append(w)?;
    }
    dict.set_item("warnings", warnings_list)?;

    Ok(dict.into())
}
