//! The `cobre.errors` exception hierarchy and the single error-mapping site.
//!
//! Cobre's Rust layer fails with a handful of typed error enums
//! ([`cobre_io::LoadError`], [`cobre_io::OutputError`], [`cobre_sddp::SddpError`])
//! and a few string-prefixed messages produced by the run/study front ends.
//! This module funnels every one of those failures through a single
//! [`convert_error`] function that maps them to a structured Python exception
//! hierarchy rooted at `cobre.errors.CobreError`.
//!
//! ## Hierarchy
//!
//! Every leaf class subclasses BOTH [`CobreError`] and the matching builtin, so
//! existing `except OSError` / `except ValueError` / `except RuntimeError` code
//! keeps catching while new code can catch the typed class or the common
//! `CobreError` base:
//!
//! ```text
//! CobreError(Exception)
//! ├── ValidationError(CobreError, ValueError)
//! ├── PolicyIncompatibleError(CobreError, ValueError)
//! ├── CaseIoError(CobreError, OSError)
//! ├── OutputError(CobreError, OSError)
//! ├── SolverError(CobreError, RuntimeError)        ← stage/iteration/scenario
//! └── SimulationError(CobreError, RuntimeError)
//! ```
//!
//! The qualified name of every class is `cobre.errors.<Name>` so tracebacks read
//! `cobre.errors.SolverError`.
//!
//! ## Single mapping site
//!
//! [`convert_error`] is the ONLY place Rust errors become Python exceptions for
//! the raising paths. It accepts a concrete [`ErrorSource`] enum (never a
//! `Box<dyn Trait>`) so the match is exhaustive: a newly added `LoadError` /
//! `OutputError` variant fails the build, surfacing the need to map it. The
//! `SddpError` match keeps explicit `Infeasible`/`Simulation` arms plus the
//! documented total `other => SolverError(None)` fallthrough (every other
//! `SddpError` reaches this site only through the training/simulation phase
//! helpers, whose failures are `RuntimeError`-shaped).
//!
//! The `cobre.io.validate` data-report `kind` field is intentionally NOT routed
//! here — it is a stable data contract decoupled from these class names.

use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyTuple, PyType};

use cobre_io::{LoadError, OutputError};
use cobre_sddp::SddpError;

pyo3::create_exception!(
    errors,
    CobreError,
    PyException,
    "Base class for every Cobre exception (subclasses `Exception`)."
);

/// A leaf exception class in the `cobre.errors` hierarchy.
///
/// Each leaf subclasses BOTH [`CobreError`] and a builtin, so the dual-base
/// type object is built once via Python's `type(name, bases, dict)` and cached
/// for the life of the interpreter.
struct LeafClass {
    /// Unqualified class name (e.g. `"ValidationError"`).
    name: &'static str,
    /// The builtin co-base alongside [`CobreError`] (e.g. `PyValueError`).
    builtin_base: BuiltinBase,
    /// Docstring exposed as `__doc__`.
    doc: &'static str,
    /// The lazily built, interpreter-lifetime type object.
    cell: PyOnceLock<Py<PyType>>,
}

/// The builtin co-base for a [`LeafClass`].
#[derive(Clone, Copy)]
enum BuiltinBase {
    Value,
    Os,
    Runtime,
}

impl LeafClass {
    const fn new(name: &'static str, builtin_base: BuiltinBase, doc: &'static str) -> Self {
        Self {
            name,
            builtin_base,
            doc,
            cell: PyOnceLock::new(),
        }
    }

    /// Resolve (building once) the dual-base type object under the GIL.
    fn get(&self, py: Python<'_>) -> PyResult<&Py<PyType>> {
        self.cell.get_or_try_init(py, || self.build(py))
    }

    /// Build the dual-base class object: `type(name, (CobreError, Builtin), {})`
    /// with `__module__ = "cobre.errors"` and `__doc__ = doc`, so the qualified
    /// name reads `cobre.errors.<Name>`.
    fn build(&self, py: Python<'_>) -> PyResult<Py<PyType>> {
        let cobre_base = py.get_type::<CobreError>();
        let builtin: Bound<'_, PyType> = match self.builtin_base {
            BuiltinBase::Value => py.get_type::<PyValueError>(),
            BuiltinBase::Os => py.get_type::<PyOSError>(),
            BuiltinBase::Runtime => py.get_type::<PyRuntimeError>(),
        };
        let bases = PyTuple::new(py, [cobre_base, builtin])?;
        let namespace = pyo3::types::PyDict::new(py);
        namespace.set_item("__module__", "cobre.errors")?;
        namespace.set_item("__doc__", self.doc)?;

        let type_builtin = py.get_type::<PyType>();
        let class = type_builtin.call1((self.name, bases, namespace))?;
        let class_type = class.cast_into::<PyType>().map_err(PyErr::from)?;
        Ok(class_type.unbind())
    }
}

/// `ValidationError(CobreError, ValueError)` — schema / parse / constraint /
/// config-override load failures.
static VALIDATION_ERROR: LeafClass = LeafClass::new(
    "ValidationError",
    BuiltinBase::Value,
    "Raised when case data or configuration fails validation (subclasses ValueError).",
);

/// `PolicyIncompatibleError(CobreError, ValueError)` — warm-start / resume
/// policy incompatibility.
static POLICY_INCOMPATIBLE_ERROR: LeafClass = LeafClass::new(
    "PolicyIncompatibleError",
    BuiltinBase::Value,
    "Raised when a warm-start policy is incompatible with the current system \
     (subclasses ValueError).",
);

/// `CaseIoError(CobreError, OSError)` — filesystem read/write failures.
static CASE_IO_ERROR: LeafClass = LeafClass::new(
    "CaseIoError",
    BuiltinBase::Os,
    "Raised on a filesystem read or write failure while loading or writing a case \
     (subclasses OSError).",
);

/// `OutputError(CobreError, OSError)` — output serialization / schema / manifest
/// failures on the write path.
static OUTPUT_ERROR: LeafClass = LeafClass::new(
    "OutputError",
    BuiltinBase::Os,
    "Raised on an output serialization, schema, or manifest failure while writing \
     results (subclasses OSError).",
);

/// `SolverError(CobreError, RuntimeError)` — training/solver failures. Carries
/// `stage`/`iteration`/`scenario` int attributes for an infeasible subproblem,
/// `None` otherwise.
static SOLVER_ERROR: LeafClass = LeafClass::new(
    "SolverError",
    BuiltinBase::Runtime,
    "Raised on a training or solver failure (subclasses RuntimeError). For an \
     infeasible subproblem, exposes integer `stage`/`iteration`/`scenario` \
     attributes; otherwise they are None.",
);

/// `SimulationError(CobreError, RuntimeError)` — simulation-phase failures.
static SIMULATION_ERROR: LeafClass = LeafClass::new(
    "SimulationError",
    BuiltinBase::Runtime,
    "Raised on a simulation-phase failure (subclasses RuntimeError).",
);

/// The owned/borrowed source of an error to be mapped to a Python exception.
///
/// A concrete enum (NOT a `Box<dyn Trait>`, per the hard rules) so every call
/// site funnels through [`convert_error`] with an exhaustive match.
pub(crate) enum ErrorSource<'a> {
    /// A case-load failure from `cobre-io`.
    Load(&'a LoadError),
    /// An output-write / metadata-read failure from `cobre-io`.
    Output(&'a OutputError),
    /// A typed SDDP error carried verbatim from a phase helper, paired with the
    /// descriptive message that today's string path would have produced.
    ///
    /// `message` preserves the exact text (so `match=` assertions pass); the
    /// [`SddpError`] supplies the structured fields (e.g. `Infeasible`).
    Sddp {
        /// The typed SDDP error (borrowed; `Send + Sync + 'static`).
        error: &'a SddpError,
        /// The verbatim descriptive message.
        message: String,
    },
    /// A string-prefixed message from run/study with no typed source.
    Message(String),
}

/// Build a [`CaseIoError`] `PyErr` from a message.
fn case_io_error(py: Python<'_>, message: &str) -> PyErr {
    new_leaf_err(py, &CASE_IO_ERROR, message)
}

/// Build a [`ValidationError`] `PyErr` from a message.
fn validation_error(py: Python<'_>, message: &str) -> PyErr {
    new_leaf_err(py, &VALIDATION_ERROR, message)
}

/// Build a `PyErr` for the given leaf class with the supplied message.
///
/// If the dual-base class fails to build (it never should, the bases are
/// builtins), the returned `PyErr` is that build failure rather than a
/// mis-typed exception — it cannot be silently swallowed.
fn new_leaf_err(py: Python<'_>, leaf: &LeafClass, message: &str) -> PyErr {
    let class = match leaf.get(py) {
        Ok(class) => class,
        Err(err) => return err,
    };
    match class.bind(py).call1((message,)) {
        Ok(instance) => PyErr::from_value(instance),
        Err(err) => err,
    }
}

/// Map an [`ErrorSource`] to the appropriate `cobre.errors` Python exception.
///
/// This is the single mapping site for every raising path. The `ErrorSource`,
/// `LoadError`, and `OutputError` matches are exhaustive with explicit arms (a
/// new variant fails the build). The `SddpError` match keeps explicit
/// `Infeasible`/`Simulation` arms plus the documented total `other =>
/// SolverError(None)` fallthrough.
///
/// Requires a GIL token internally (it constructs Python exception instances and
/// `setattr`s structured fields), but its signature takes only Rust enums and
/// strings, so it is callable from a GIL-bound Rust `#[cfg(test)]` boundary test
/// as well as from the binding entry points.
pub(crate) fn convert_error(source: ErrorSource<'_>) -> PyErr {
    Python::attach(|py| convert_error_with(py, source))
}

/// GIL-bound body of [`convert_error`]. Split out so a test (already holding the
/// GIL) can call it directly with its own `py` token.
fn convert_error_with(py: Python<'_>, source: ErrorSource<'_>) -> PyErr {
    match source {
        ErrorSource::Load(err) => match err {
            LoadError::IoError { .. } => case_io_error(py, &err.to_string()),
            LoadError::ParseError { .. }
            | LoadError::SchemaError { .. }
            | LoadError::CrossReferenceError { .. }
            | LoadError::ConstraintError { .. } => validation_error(py, &err.to_string()),
            LoadError::PolicyIncompatible { .. } => {
                new_leaf_err(py, &POLICY_INCOMPATIBLE_ERROR, &err.to_string())
            }
        },
        ErrorSource::Output(err) => match err {
            // Preserve today's read-path behavior: a NotFound I/O error maps to
            // the builtin FileNotFoundError (no typed class for it).
            OutputError::IoError { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                PyFileNotFoundError::new_err(err.to_string())
            }
            OutputError::IoError { .. } => case_io_error(py, &err.to_string()),
            OutputError::SerializationError { .. } | OutputError::SchemaError { .. } => {
                new_leaf_err(py, &OUTPUT_ERROR, &err.to_string())
            }
            // Preserve today's `metadata_error_to_py` -> ValueError behavior.
            OutputError::ManifestError { .. } => validation_error(py, &err.to_string()),
        },
        ErrorSource::Sddp { error, message } => match error {
            SddpError::Infeasible {
                stage,
                iteration,
                scenario,
            } => solver_error_infeasible(py, &message, *stage, *iteration, *scenario),
            SddpError::Simulation(_) => new_leaf_err(py, &SIMULATION_ERROR, &message),
            // Documented total arm: every other SddpError (Solver, Validation,
            // Io, Stochastic, Communication, WireVersionMismatch) is
            // RuntimeError-shaped today, so it maps to a SolverError with the
            // three structured attributes set to None.
            _other => solver_error_plain(py, &message),
        },
        ErrorSource::Message(msg) => message_prefix_to_pyerr(py, &msg),
    }
}

/// Reproduce the historical message-prefix dispatch table, mapping the
/// string-prefixed run/study messages to the typed classes. Every message string
/// is passed through unchanged.
fn message_prefix_to_pyerr(py: Python<'_>, msg: &str) -> PyErr {
    if msg.starts_with("output write error") || msg.starts_with("policy checkpoint error") {
        case_io_error(py, msg)
    } else if msg.starts_with("config override error")
        || msg.starts_with("config parse error")
        || msg.starts_with("config read error")
    {
        validation_error(py, msg)
    } else if msg.starts_with("policy validation error") {
        new_leaf_err(py, &POLICY_INCOMPATIBLE_ERROR, msg)
    } else if msg.starts_with("simulation error") {
        new_leaf_err(py, &SIMULATION_ERROR, msg)
    } else {
        // "training error" / "training failed after" and every other message
        // fall through to SolverError, preserving today's PyRuntimeError default.
        solver_error_plain(py, msg)
    }
}

/// Build a [`SolverError`] for a non-infeasible failure: all three structured
/// attributes are `None`.
fn solver_error_plain(py: Python<'_>, message: &str) -> PyErr {
    build_solver_error(py, message, None)
}

/// Build a [`SolverError`] for an infeasible subproblem, attaching the
/// `stage`/`iteration`/`scenario` int attributes.
fn solver_error_infeasible(
    py: Python<'_>,
    message: &str,
    stage: usize,
    iteration: u64,
    scenario: usize,
) -> PyErr {
    build_solver_error(py, message, Some((stage, iteration, scenario)))
}

/// Construct a `SolverError` instance, setting `stage`/`iteration`/`scenario`
/// either to the supplied ints (infeasible) or to `None` (every other failure)
/// so the three attributes are always present on the instance.
///
/// A `setattr` failure propagates as the returned `PyErr` — it is never
/// swallowed.
fn build_solver_error(
    py: Python<'_>,
    message: &str,
    infeasible: Option<(usize, u64, usize)>,
) -> PyErr {
    let class = match SOLVER_ERROR.get(py) {
        Ok(class) => class,
        Err(err) => return err,
    };
    let instance = match class.bind(py).call1((message,)) {
        Ok(instance) => instance,
        Err(err) => return err,
    };

    let set_fields = || -> PyResult<()> {
        if let Some((stage, iteration, scenario)) = infeasible {
            instance.setattr("stage", stage)?;
            instance.setattr("iteration", iteration)?;
            instance.setattr("scenario", scenario)?;
        } else {
            instance.setattr("stage", py.None())?;
            instance.setattr("iteration", py.None())?;
            instance.setattr("scenario", py.None())?;
        }
        Ok(())
    };

    if let Err(err) = set_fields() {
        return err;
    }
    PyErr::from_value(instance)
}

/// Register the seven exception classes into the `errors` submodule.
///
/// Adds `CobreError` (built by `create_exception!`) and the six dual-base leaves
/// (built lazily via `type()`), so `from cobre.errors import …` resolves all
/// seven and `issubclass` reflects the documented hierarchy.
pub(crate) fn register_errors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "__doc__",
        "Structured exception hierarchy for Cobre errors.",
    )?;
    m.add("CobreError", py.get_type::<CobreError>())?;
    for leaf in [
        &VALIDATION_ERROR,
        &POLICY_INCOMPATIBLE_ERROR,
        &CASE_IO_ERROR,
        &OUTPUT_ERROR,
        &SOLVER_ERROR,
        &SIMULATION_ERROR,
    ] {
        let class = leaf.get(py)?;
        m.add(leaf.name, class.clone_ref(py))?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{ErrorSource, LeafClass, SIMULATION_ERROR, SOLVER_ERROR, convert_error_with};
    use cobre_sddp::SddpError;
    use pyo3::prelude::*;

    /// Assert the bound `PyErr` value is an instance of the supplied leaf class
    /// and that its qualified name reads `cobre.errors.<Name>`.
    ///
    /// Resolves the type object directly from the leaf static (no `sys.modules`
    /// or `import cobre.errors`, which would require the parent `cobre` package
    /// to exist in the standalone test binary).
    fn assert_leaf(py: Python<'_>, err: &PyErr, leaf: &LeafClass) {
        let class = leaf.get(py).expect("leaf class builds").bind(py);
        let value = err.value(py);
        assert!(
            value.is_instance(class).expect("isinstance check"),
            "expected cobre.errors.{}, got {value:?}",
            leaf.name
        );
        // The qualified name must read `cobre.errors.<Name>`.
        let qualname: String = class
            .getattr("__module__")
            .expect("__module__")
            .extract()
            .expect("str");
        assert_eq!(qualname, "cobre.errors", "{} __module__", leaf.name);
    }

    /// The authoritative structured-field assertion (Acceptance Criteria 3):
    /// an `Infeasible` SDDP error maps to `SolverError` with the three int
    /// attributes set and the verbatim message preserved.
    #[test]
    fn convert_error_infeasible() {
        Python::initialize();
        Python::attach(|py| {
            let message = "infeasible subproblem at stage 5, iteration 42, scenario 3".to_string();
            let err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Infeasible {
                        stage: 5,
                        iteration: 42,
                        scenario: 3,
                    },
                    message: message.clone(),
                },
            );

            assert_leaf(py, &err, &SOLVER_ERROR);

            let value = err.value(py);
            let stage: usize = value.getattr("stage").unwrap().extract().unwrap();
            let iteration: u64 = value.getattr("iteration").unwrap().extract().unwrap();
            let scenario: usize = value.getattr("scenario").unwrap().extract().unwrap();
            assert_eq!(stage, 5);
            assert_eq!(iteration, 42);
            assert_eq!(scenario, 3);

            let rendered: String = value.str().unwrap().extract().unwrap();
            assert_eq!(rendered, message);
        });
    }

    /// Sibling: the non-`Infeasible` SDDP arms. `Simulation` maps to
    /// `SimulationError`; a generic `Solver` maps to `SolverError` with the three
    /// attributes `None`. Messages are preserved verbatim.
    #[test]
    fn convert_error_non_infeasible_arms() {
        Python::initialize();
        Python::attach(|py| {
            // Simulation -> SimulationError.
            let sim_msg = "simulation error: x".to_string();
            let sim_err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Simulation("x".to_string()),
                    message: sim_msg.clone(),
                },
            );
            assert_leaf(py, &sim_err, &SIMULATION_ERROR);
            let sim_rendered: String = sim_err.value(py).str().unwrap().extract().unwrap();
            assert_eq!(sim_rendered, sim_msg);

            // Generic non-Infeasible/non-Simulation SddpError -> SolverError with
            // attributes None.
            let solver_msg = "training error: solver error: numerical failure".to_string();
            let solver_err = convert_error_with(
                py,
                ErrorSource::Sddp {
                    error: &SddpError::Validation("numerical failure".to_string()),
                    message: solver_msg.clone(),
                },
            );
            assert_leaf(py, &solver_err, &SOLVER_ERROR);
            let value = solver_err.value(py);
            assert!(value.getattr("stage").unwrap().is_none());
            assert!(value.getattr("iteration").unwrap().is_none());
            assert!(value.getattr("scenario").unwrap().is_none());
            let solver_rendered: String = value.str().unwrap().extract().unwrap();
            assert_eq!(solver_rendered, solver_msg);
        });
    }
}
