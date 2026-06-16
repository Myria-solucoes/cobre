//! The `cobre.Study` pyclass — a live, in-memory study loaded once from a case
//! directory and reused across the solve lifecycle.
//!
//! `Study.__new__` runs the front half of the solve lifecycle (load →
//! stochastic preprocessing → hydro models → `StudySetup` construction →
//! provenance/summary build + sidecar writes) exactly once via the shared
//! [`crate::run::build_study_setup`] helper, then stores the live [`StudySetup`]
//! and the adjacent immutable state so later `train`/`simulate` methods need no
//! reload. It also exposes [`Study::validate`], which replays the validation
//! warnings captured during construction without re-reading disk.
//!
//! ## Single-process only
//!
//! Like [`crate::run`], this module uses [`cobre_comm::LocalBackend`] exclusively
//! and never initializes MPI.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use cobre_sddp::{
    FutureCostFunction, HydroModelSummary, ModelProvenanceReport, StochasticSummary, StudySetup,
    TrainingResult,
};

use crate::errors::{ErrorSource, convert_error};
use crate::model::PySystem;
use crate::run::{
    LoadedStudy, PhaseError, SimSummary, apply_training_policy_mode, build_study_setup,
    reconstruct_policy_from_checkpoint, run_in_scoped_pool, run_simulation_phase_py,
    run_training_phase_py, run_training_phase_py_streaming, write_evaporation_models_if_any,
    write_fpha_hyperplanes_if_any, write_training_artifacts,
};

/// Map a [`PhaseError`] to a Python exception through the single
/// [`convert_error`] mapping site.
///
/// A typed `Sddp` arm routes through [`ErrorSource::Sddp`] so structured fields
/// (e.g. `Infeasible`'s stage/iteration/scenario) reach Python as `SolverError`
/// attributes; a `Message` arm routes through [`ErrorSource::Message`]. Both
/// front ends (`run_via_study` and the `Study` methods) thus map identically.
fn phase_error_to_pyerr(err: PhaseError) -> PyErr {
    match err {
        PhaseError::Message(msg) => convert_error(ErrorSource::Message(msg)),
        PhaseError::Sddp { error, message } => convert_error(ErrorSource::Sddp {
            error: &error,
            message,
        }),
    }
}

/// A loaded study: the live [`StudySetup`] plus the immutable state produced by
/// the front half of the solve lifecycle.
///
/// Constructed once via [`Study::__new__`] (which runs
/// [`crate::run::build_study_setup`]); `train`/`simulate` (later tickets) reuse
/// the stored state, and [`Study::validate`] replays the captured warnings.
/// The `output_dir` is fixed at construction time so every artifact this study
/// writes lands in the same directory (the always-writes contract).
// Some fields are stored at construction time but only consumed by the
// `simulate` method added in a later ticket; they are intentionally retained
// here (the always-load-once contract) and annotated `dead_code` until that
// method lands.
#[pyclass(name = "Study")]
pub struct Study {
    /// The live, fully prepared study setup. Consumed by `train` (and
    /// `simulate` in a later ticket).
    setup: StudySetup,
    /// The system after stochastic preprocessing. Shared (not copied) with the
    /// `cobre.model.System` view returned by the `system` getter; `train`/
    /// `simulate` borrow `&*self.system`.
    system: Arc<cobre_core::System>,
    /// The effective (post-override) configuration. Consumed by `train` (and
    /// `simulate` in a later ticket).
    config: cobre_io::Config,
    /// The resolved tree seed. Consumed by `train` (and `simulate` in a later
    /// ticket).
    seed: u64,
    /// The model-provenance report. Consumed by `simulate` in a later ticket.
    // Rationale: stored at construction time under the always-load-once
    // contract; the `simulate` PyO3 method reads it to populate the simulation
    // output report. Removing it would force a second expensive case load when
    // simulate is called.
    #[allow(dead_code)]
    provenance: ModelProvenanceReport,
    /// The structural stochastic summary. Consumed by `simulate` in a later
    /// ticket.
    // Rationale: captured once during construction so the `simulate` method
    // can surface stochastic preprocessing metadata in its output without
    // re-running the preprocessing pipeline.
    #[allow(dead_code)]
    stochastic_summary: StochasticSummary,
    /// The structural hydro-model summary. Consumed by `simulate` in a later
    /// ticket.
    // Rationale: captured once during construction so the `simulate` method
    // can include hydro-model provenance in its output report without
    // re-running the hydro preprocessing pipeline.
    #[allow(dead_code)]
    hydro_models_summary: HydroModelSummary,
    /// Validation-pipeline warnings captured during the case load, replayed by
    /// [`Study::validate`].
    warnings: Vec<cobre_io::ReportEntry>,
    /// The output directory fixed at construction time.
    output_dir: PathBuf,
    /// The requested thread count, stored for later `train`/`simulate` calls.
    threads: Option<u32>,
}

/// The output of [`Study::train`]: an in-memory trained-policy handle.
///
/// `Policy` carries enough state to drive `simulate` directly without reloading
/// a checkpoint from disk: the [`TrainingResult`] (which owns the per-stage
/// basis cache and the baked stage templates) and a clone of the trained
/// [`FutureCostFunction`] (the cut pool). A `Policy.load`-ed handle (a later
/// ticket) and a trained one therefore expose the same shape.
///
/// The read-only getters surface the headline convergence figures
/// (`iterations`, `final_lower_bound`, `final_upper_bound`); richer accessors
/// land in a later ticket.
#[pyclass(name = "Policy")]
pub struct Policy {
    /// The full training result: convergence metrics, basis cache, baked
    /// templates, solver-stats log. Moved out of the training phase and into
    /// the handle so `simulate` can warm-start from it.
    training_result: TrainingResult,
    /// A clone of the trained (or loaded) study FCF (the cut pool).
    /// `Study::simulate` reads this FCF and `replace_fcf`s it into the study
    /// before simulating, so a trained `Policy` and a loaded one feed the
    /// identical simulate path.
    fcf: FutureCostFunction,
}

#[pymethods]
impl Policy {
    /// Number of completed training iterations.
    #[getter]
    fn iterations(&self) -> u64 {
        self.training_result.iterations
    }

    /// Final lower bound at termination.
    #[getter]
    fn final_lower_bound(&self) -> f64 {
        self.training_result.final_lb
    }

    /// Final (mean) upper bound at termination.
    #[getter]
    fn final_upper_bound(&self) -> f64 {
        self.training_result.final_ub
    }

    /// Evaluate the future-cost function at `state` for the given 0-based `stage`.
    ///
    /// Returns `max_k(intercept_k + coeffs_k · state)` over the stage's active
    /// Benders cuts — the FCF lower-bound value at that state. The coefficients
    /// are the stored cut gradients (the raw `HiGHS` duals; see [`Policy::cut_matrix`]),
    /// so the cut is read as `θ ≥ intercept + coeffs · state`. A stage with no
    /// active cuts returns `float('-inf')` (NOT an error).
    ///
    /// `stage` uses the FCF's 0-based stage indexing (stage `t - 1` for the
    /// 1-based SDDP stage `t`).
    ///
    /// # Errors
    ///
    /// - `IndexError` if `stage` is out of range.
    /// - `ValueError` if `state` does not have the policy's state dimension.
    // `state: Vec<f64>` is required so PyO3 can extract from an arbitrary Python
    // sequence; only `&state` is read, hence the `needless_pass_by_value` allow.
    // `stage`/`state` are the natural API names, hence the `similar_names` allow.
    #[allow(clippy::needless_pass_by_value, clippy::similar_names)]
    fn evaluate(&self, stage: usize, state: Vec<f64>) -> PyResult<f64> {
        let n_stages = self.fcf.pools.len();
        if stage >= n_stages {
            return Err(PyIndexError::new_err(format!(
                "stage {stage} out of range (policy has {n_stages} stages)"
            )));
        }
        let dim = self.fcf.state_dimension;
        if state.len() != dim {
            return Err(PyValueError::new_err(format!(
                "state has length {}, expected {dim} (policy state dimension)",
                state.len()
            )));
        }
        Ok(self.fcf.evaluate_at_state(stage, &state))
    }

    /// Return the stage's active Benders cuts as two `NumPy` arrays.
    ///
    /// The result is the 2-tuple `(intercepts, coeffs)` where `intercepts` has
    /// shape `(n_cuts,)` and `coeffs` has shape `(n_cuts, dim)`, both `float64`.
    /// Row `k` of `coeffs` is the gradient of cut `k`, and `intercepts[k]` its
    /// constant term; the active cuts are emitted in the FCF's native ascending
    /// slot order (the deterministic pool order). `dim` is the policy state
    /// dimension and is the column count even when `n_cuts == 0` (shapes `(0,)`
    /// and `(0, dim)`).
    ///
    /// `stage` uses the FCF's 0-based stage indexing (stage `t - 1` for the
    /// 1-based SDDP stage `t`).
    ///
    /// ## Sign convention
    ///
    /// Coefficients are returned **exactly as stored** — the raw `HiGHS` dual of
    /// the state-fixing rows, used directly as the FCF gradient. They are **NOT**
    /// negated. A downstream consumer reconstructs each cut as
    /// `θ ≥ intercept + coeffs · state`, consistent with [`Policy::evaluate`].
    /// (The LP-assembly negation in `build_cut_row_batch` is an internal detail
    /// of solving and does not affect the values surfaced here.)
    ///
    /// # Errors
    ///
    /// - `IndexError` if `stage` is out of range.
    /// - `ImportError` if `NumPy` is not installed (propagated verbatim from the
    ///   lazy `import numpy`; `NumPy` is a soft, lazily imported dependency).
    fn cut_matrix(&self, py: Python<'_>, stage: usize) -> PyResult<Py<PyAny>> {
        let n_stages = self.fcf.pools.len();
        if stage >= n_stages {
            return Err(PyIndexError::new_err(format!(
                "stage {stage} out of range (policy has {n_stages} stages)"
            )));
        }
        let dim = self.fcf.state_dimension;

        let mut intercepts: Vec<f64> = Vec::new();
        let mut coeffs_flat: Vec<f64> = Vec::new();
        for (_slot, intercept, coeffs) in self.fcf.active_cuts(stage) {
            intercepts.push(intercept);
            coeffs_flat.extend_from_slice(coeffs);
        }
        let n_cuts = intercepts.len();

        // Lazy soft import; ImportError propagates verbatim if NumPy is absent.
        let np = py.import("numpy")?;
        let intercepts_arr = np.call_method1("asarray", (intercepts,))?;
        // Build the (n_cuts, dim) coefficient matrix from the flat C-order
        // buffer by reshaping the 1-D array. `dim` fixes the column count even
        // when `n_cuts == 0`, giving shape `(0, dim)`.
        let coeffs_1d = np.call_method1("asarray", (coeffs_flat,))?;
        let coeffs_arr = coeffs_1d.call_method1("reshape", ((n_cuts, dim),))?;

        Ok((intercepts_arr, coeffs_arr)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

#[pymethods]
impl Study {
    /// Load a case directory into a live, reusable [`Study`].
    ///
    /// Runs the front half of the solve lifecycle once: loads the case, resolves
    /// the effective config (deep-merging `config_overrides` when present), runs
    /// stochastic and hydro-model preprocessing, builds the [`StudySetup`], and
    /// writes the front-half sidecars (`training/scaling_report.json`,
    /// `training/model_provenance.json`, `training/hydro_models.json`, and the
    /// stochastic exports when enabled) to `output_dir`.
    ///
    /// `output_dir` defaults to `case_dir/output` when `None`. `threads` is
    /// stored for later `train`/`simulate` calls and is not used during load.
    /// `config_overrides` is a flat dotted-key mapping (e.g.
    /// `{"training.tree_seed": 7}`) converted under the GIL before the load runs
    /// with the GIL released.
    ///
    /// # Errors
    ///
    /// - Raises `OSError` if `case_dir` does not exist (before any work).
    /// - Raises `ValueError` on a malformed override dict (non-str key or
    ///   unsupported value type), or on a config override/parse/read failure.
    /// - Raises `OSError` on a sidecar write failure, and `RuntimeError` on any
    ///   other load/preprocessing/construction failure.
    #[new]
    #[pyo3(signature = (case_dir, output_dir=None, threads=None, config_overrides=None))]
    // PyO3 `#[new]` extracts owned argument types; the path values are then used
    // only by reference here, so clippy's pass-by-value lint does not apply.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        py: Python<'_>,
        case_dir: PathBuf,
        output_dir: Option<PathBuf>,
        threads: Option<u32>,
        config_overrides: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if !case_dir.exists() {
            return Err(PyOSError::new_err(format!(
                "case directory does not exist: {}",
                case_dir.display()
            )));
        }

        let resolved_output = output_dir.unwrap_or_else(|| case_dir.join("output"));

        // Convert the override dict UNDER THE GIL, before py.detach releases it.
        // An unsupported value type or non-str key raises PyValueError here. The
        // resulting `serde_json::Map` is owned (Send), so it crosses the
        // py.detach boundary into the GIL-free load.
        let overrides = config_overrides
            .map(|dict| crate::convert::pydict_to_json_map(&dict))
            .transpose()?;

        // The load runs PAR estimation and can be slow; release the GIL while it
        // runs. No rayon work happens here, so no scoped pool is created (the
        // per-call pool is built in `train`/`simulate`).
        let loaded: Result<LoadedStudy, String> =
            py.detach(|| build_study_setup(&case_dir, &resolved_output, overrides.as_ref()));

        let LoadedStudy {
            setup,
            system,
            config,
            seed,
            provenance,
            stochastic_summary,
            hydro_models_summary,
            warnings,
        } = loaded.map_err(|msg| convert_error(ErrorSource::Message(msg)))?;

        Ok(Study {
            setup,
            system: Arc::new(system),
            config,
            seed,
            provenance,
            stochastic_summary,
            hydro_models_summary,
            warnings,
            output_dir: resolved_output,
            threads,
        })
    }

    /// The resolved output directory as a string.
    #[getter]
    fn output_dir(&self) -> String {
        self.output_dir.to_string_lossy().into_owned()
    }

    /// The loaded [`cobre_core::System`] (as `cobre.model.System`).
    ///
    /// Lets callers introspect the loaded study without a reload. Returned via a
    /// cheap [`Arc`] refcount bump — the underlying `System` is shared, not
    /// copied.
    #[getter]
    fn system(&self) -> PySystem {
        PySystem::from_arc(Arc::clone(&self.system))
    }

    /// Validate the loaded study, returning the same report dict shape as
    /// `cobre.io.validate`: keys `"valid"` (bool), `"errors"` (`list[dict]`),
    /// `"warnings"` (`list[dict]`).
    ///
    /// Because `__new__` already ran every `cobre.io.validate` phase (a failure
    /// there would have raised), the study is known valid here: this returns
    /// `{"valid": True, "errors": [], "warnings": [...]}`, where the warnings are
    /// the `cobre-io` pipeline warnings captured during construction. It never
    /// re-reads disk and never raises.
    fn validate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("valid", true)?;
        dict.set_item("errors", PyList::empty(py))?;
        dict.set_item(
            "warnings",
            crate::io::build_warnings_list(py, &self.warnings)?,
        )?;

        Ok(dict)
    }

    /// Train an SDDP policy against this study's in-memory [`StudySetup`],
    /// writing the training artifacts and returning a [`Policy`] handle.
    ///
    /// Runs the SDDP training loop against the live setup built once in
    /// `__new__`, honoring `config.policy.mode` (warm-start / resume /
    /// boundary cuts) before training. The whole Rust computation runs with the
    /// GIL released; when an `on_iteration` callback is provided it is invoked
    /// once per training-iteration boundary in a dedicated drain thread that
    /// reacquires the GIL only at those boundaries (never in the solver's hot
    /// loop).
    ///
    /// Writes `training/policy/`, `training/solver_stats.parquet` (when
    /// non-empty), `training/cut_selection.parquet` (when non-empty),
    /// `training/metadata.json`, `training/_SUCCESS`, and
    /// `hydro_models/fpha_hyperplanes.parquet` (when non-empty) — bit-identical
    /// to `cobre.run.run`. Artifacts are always written before any captured
    /// callback exception is propagated, so a stopped or raising run still
    /// persists what it completed.
    ///
    /// When `config.training.enabled` is `false`, this is a no-op that returns
    /// a [`Policy`] whose `TrainingResult` is a synthetic zero-iteration result
    /// carrying a fresh zero-cut FCF. Such a policy cannot be simulated: pass it
    /// to [`Study::simulate`] and it raises, because simulating with no Benders
    /// cuts would silently produce a wrong result. When training is disabled, use
    /// [`Study::load_policy`] to load a previously trained policy from disk before
    /// calling [`Study::simulate`].
    ///
    /// # Arguments
    ///
    /// - `on_iteration` — optional Python callable invoked once per training
    ///   iteration boundary with a dict (`"kind"`, `"iteration"`,
    ///   `"lower_bound"`, `"upper_bound"`, `"gap"`, `"wall_time_ms"`); `gap` is
    ///   the raw relative gap (NOT scaled by 100). A truthy return requests a
    ///   cooperative stop at a later iteration boundary (asynchronous); a
    ///   raising callback propagates as this method's exception after artifacts
    ///   are written.
    ///
    /// # Errors
    ///
    /// - `RuntimeError` / `OSError` on `HiGHS` init failure, a training error, a
    ///   drain-thread panic, or a policy-mode failure (e.g. a missing prior
    ///   policy directory under `WarmStart`/`Resume`), mapped from the
    ///   descriptive message.
    /// - The original exception raised by a callback (or `KeyboardInterrupt`)
    ///   re-raised verbatim AFTER the training artifacts are written.
    #[pyo3(signature = (on_iteration=None))]
    fn train(&mut self, py: Python<'_>, on_iteration: Option<Py<PyAny>>) -> PyResult<Policy> {
        // Training-disabled: explicit no-op that returns a synthetic
        // zero-iteration policy. The simulation-only path loads a policy
        // elsewhere (`simulate`/`Policy.load`, a later ticket), so there is
        // nothing to train or write here. This mirrors the training-disabled
        // branch of `run_via_study`.
        if !self.config.training.enabled {
            let synthetic = TrainingResult::new(
                0.0,
                f64::INFINITY,
                0.0,
                0.0,
                0,
                "training disabled".to_string(),
                0,
                Vec::new(),
                Vec::new(),
                None,
                None,
            );
            return Ok(Policy {
                training_result: synthetic,
                fcf: self.setup.fcf.clone(),
            });
        }

        // Borrow the immutable study state for the detached closure. The
        // mutable `setup` is the only field the training loop mutates; the
        // rest are read-only.
        let seed = self.seed;
        let output_dir = self.output_dir.clone();
        let threads = self.threads;
        let setup = &mut self.setup;
        let system = self.system.as_ref();
        let config = &self.config;

        // `on_iteration` is already a `Py<PyAny>` (a GIL-independent owned
        // handle), so it can cross the `py.detach` boundary into the drain
        // thread and be re-bound under `Python::attach`. `Py<PyAny>` and
        // `PyErr` are `Send`, so returning `(TrainingPhaseResult, Option<PyErr>)`
        // from the scoped-pool closure is sound.
        // The closure unifies on `PhaseError`: the two training-phase helpers
        // yield it directly (carrying a typed `SddpError` for a hard `train`
        // failure), while the `String`-returning helpers (policy-mode,
        // artifact/FPHA writes, pool construction) convert via `From<String>`.
        // The error is mapped through the single `convert_error` site below, so
        // both front ends (`run_via_study` and this method) map identically.
        let phase_result: Result<(crate::run::TrainingPhaseResult, Option<PyErr>), PhaseError> = py
            .detach(|| {
                // `run_in_scoped_pool` returns `Result<closure_output, String>`,
                // where `closure_output` is itself a `Result<_, PhaseError>`; the
                // outer `?` surfaces pool-construction failure (a `String`, lifted
                // to `PhaseError::Message`), the inner result is the
                // training/artifact outcome.
                run_in_scoped_pool(threads, |n| {
                    // Warm-start / resume / boundary-cut FCF replacement BEFORE
                    // training — shared verbatim with `run_via_study`.
                    apply_training_policy_mode(setup, system, config, &output_dir)?;

                    // Bifurcate on the callback: `None` keeps the historical
                    // collect-after-return path bit-identical (the no-callback
                    // golden parity anchor); `Some` uses the streaming drain
                    // thread.
                    let (training, callback_error) = match on_iteration {
                        Some(callback) => run_training_phase_py_streaming(setup, n, callback)?,
                        None => (run_training_phase_py(setup, n)?, None),
                    };

                    // Write ALL artifacts FIRST (always-writes), then surface a
                    // captured callback error — ordering identical to
                    // `run_via_study`, so a stopped/raising run still persists its
                    // partial artifacts.
                    write_training_artifacts(
                        &output_dir,
                        system,
                        config,
                        setup,
                        &training,
                        seed,
                        n,
                    )?;
                    write_fpha_hyperplanes_if_any(&output_dir, setup)?;
                    write_evaporation_models_if_any(&output_dir, setup, system)?;

                    Ok::<_, PhaseError>((training, callback_error))
                })?
            });

        let (mut training, callback_error) = phase_result.map_err(phase_error_to_pyerr)?;

        // Propagate a captured callback exception (or KeyboardInterrupt) only
        // now that all training artifacts have been written.
        if let Some(err) = callback_error {
            return Err(err);
        }

        // A genuine training failure becomes a `SolverError` (a `RuntimeError`
        // subclass), mirroring `run_via_study`'s post-artifacts mapping. The
        // typed error is moved out of the carrier so its structured fields (e.g.
        // `Infeasible`) survive to `convert_error`, with the message preserved.
        if let Some(error) = training.error.take() {
            let iterations = training.result.iterations;
            let message = format!("training failed after {iterations} iterations: {error}");
            return Err(convert_error(ErrorSource::Sddp {
                error: &error,
                message,
            }));
        }

        // Build the policy handle: clone the trained FCF and move the training
        // result out of the (now owned) phase carrier.
        Ok(Policy {
            training_result: training.result,
            fcf: setup.fcf.clone(),
        })
    }

    /// Reconstruct a [`Policy`] from an on-disk policy checkpoint so a loaded
    /// policy and a trained one converge on the IDENTICAL [`Study::simulate`]
    /// entry point.
    ///
    /// Reads `<output_dir>/<policy_path>/` (`output_dir` defaults to this study's
    /// construction-time `output_dir`), reconstructs the
    /// [`FutureCostFunction`] and a synthetic [`TrainingResult`] via the shared
    /// [`reconstruct_policy_from_checkpoint`] helper, and packages them into a
    /// [`Policy`]. The returned policy carries `baked_templates = None`;
    /// [`Study::simulate`] re-bakes the stage templates from the FCF at startup,
    /// exactly as the monolithic simulation-only path does.
    ///
    /// `validate_compatibility` overrides `config.policy.validate_compatibility`
    /// when supplied; otherwise the config value (default `true`) controls
    /// whether the checkpoint metadata is validated against this study's state
    /// dimension and stage count.
    ///
    /// # Errors
    ///
    /// - `RuntimeError` when the policy directory does not exist (message
    ///   containing `"Policy directory not found"`), when the checkpoint cannot
    ///   be read, or when FCF reconstruction fails.
    /// - `ValueError` when policy validation fails (incompatible state dimension
    ///   or stage count), mapped from the descriptive message.
    #[pyo3(signature = (output_dir=None, validate_compatibility=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn load_policy(
        &self,
        py: Python<'_>,
        output_dir: Option<PathBuf>,
        validate_compatibility: Option<bool>,
    ) -> PyResult<Policy> {
        let out_dir = output_dir.unwrap_or_else(|| self.output_dir.clone());
        let policy_dir = out_dir.join(&self.setup.policy_path);

        // Apply the optional validate_compatibility override against a config
        // clone so the shared helper reads the effective value (the helper keys
        // off `config.policy.validate_compatibility`). When `None`, the study's
        // config value (default `true`) is used unchanged.
        let mut config = self.config.clone();
        if let Some(validate) = validate_compatibility {
            config.policy.validate_compatibility = validate;
        }

        let setup = &self.setup;
        let system = self.system.as_ref();

        // The reconstruction reads parquet/JSON from disk; release the GIL.
        let reconstructed: Result<(FutureCostFunction, TrainingResult), String> =
            py.detach(|| reconstruct_policy_from_checkpoint(setup, system, &config, &policy_dir));

        let (fcf, training_result) =
            reconstructed.map_err(|msg| convert_error(ErrorSource::Message(msg)))?;
        Ok(Policy {
            training_result,
            fcf,
        })
    }

    /// Run the simulation phase against this study's in-memory [`StudySetup`]
    /// using the supplied [`Policy`], writing the `simulation/` artifacts and
    /// returning a `{"n_scenarios", "completed"}` dict.
    ///
    /// The policy's FCF is installed into the study via
    /// [`StudySetup::replace_fcf`] before simulating, so a trained `Policy` (from
    /// [`Study::train`]) and a loaded `Policy` (from [`Study::load_policy`]) feed
    /// the IDENTICAL simulate path: the unchanged [`run_simulation_phase_py`]
    /// reads the policy's `baked_templates` and `basis_cache`. A trained policy
    /// carries `baked_templates = Some(...)`; a loaded one carries `None` and the
    /// study re-bakes the stage templates from the FCF at startup — exactly the
    /// monolithic behavior (P3).
    ///
    /// `output_dir` defaults to this study's construction-time `output_dir`.
    /// Each call writes a fresh `simulation/` output set; the method may be called
    /// repeatedly against one [`Policy`] with no reload between calls. Each call
    /// installs the supplied policy's FCF into the study via
    /// [`StudySetup::replace_fcf`] before simulating, so repeated calls remain
    /// deterministic (each one re-installs the same cut pool).
    ///
    /// Writes `simulation/metadata.json`, the per-scenario parquet,
    /// `simulation/_SUCCESS`, and `simulation/solver_stats.parquet` (when the
    /// solver log is non-empty) to `output_dir` — bit-identical to
    /// `cobre.run.run`'s simulation phase.
    ///
    /// # Errors
    ///
    /// - `RuntimeError` when `policy` carries no Benders cuts (zero active cuts).
    ///   A cut-less policy — e.g. the synthetic handle [`Study::train`] returns
    ///   when `config.training.enabled` is `false` — would silently simulate a
    ///   wrong result, so this guard rejects it up front and asks the caller to
    ///   load a trained policy via [`Study::load_policy`] first.
    /// - `OSError` on a simulation workspace-pool (`HiGHS`) init failure.
    /// - `RuntimeError` on a simulation error (infeasibility) or a writer/output
    ///   failure, mapped from the descriptive message.
    #[pyo3(signature = (policy, output_dir=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn simulate(
        &mut self,
        py: Python<'_>,
        policy: PyRef<'_, Policy>,
        output_dir: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let out_dir = output_dir.unwrap_or_else(|| self.output_dir.clone());

        // Reject a cut-less policy up front. A policy with zero active Benders
        // cuts — e.g. the synthetic zero-cut handle `train()` returns when
        // `config.training.enabled` is false — would let `StudySetup::simulate`
        // run with no future-cost approximation and silently emit a wrong result.
        // The monolithic `run_via_study` path never hits this because its
        // training-disabled+simulate branch loads the on-disk FCF via
        // `reconstruct_policy_from_checkpoint`; this guard covers the direct
        // `Study.train() -> Study.simulate()` misuse (and any other zero-cut
        // policy) explicitly.
        if policy.fcf.total_active_cuts() == 0 {
            // Routed through the single mapping site: the message has no
            // recognized prefix, so it falls through to `SolverError` (a
            // `RuntimeError` subclass), so `except RuntimeError` still catches and
            // the message is preserved verbatim.
            return Err(convert_error(ErrorSource::Message(
                "Policy has no cuts to simulate; when training is disabled, call \
                 Study.load_policy() to load a trained policy before simulate()"
                    .to_string(),
            )));
        }

        // Install the policy's FCF into the study so `StudySetup::simulate`
        // (which reads `self.fcf`) uses the supplied policy. This makes a trained
        // `Policy` and a loaded `Policy` feed the identical simulate path.
        self.setup.replace_fcf(policy.fcf.clone());

        let threads = self.threads;
        let setup = &mut self.setup;
        let system = self.system.as_ref();
        let training_result = &policy.training_result;

        // The simulation runs the full scenario sweep; release the GIL for the
        // entire Rust computation. The scoped rayon pool is built per call so two
        // sequential `simulate` invocations each honor their own thread count.
        let summary: Result<SimSummary, PhaseError> = py.detach(|| {
            // `run_in_scoped_pool` returns `Result<closure_output, String>`,
            // where `closure_output` is itself `Result<SimSummary, PhaseError>`;
            // the outer `?` surfaces pool-construction failure (a `String`, lifted
            // to `PhaseError::Message`), the inner result is the simulation
            // outcome (a typed `Sddp` arm on a hard `simulate` failure).
            run_in_scoped_pool(threads, |n| {
                run_simulation_phase_py(setup, &out_dir, system, training_result, n)
            })?
        });

        let summary = summary.map_err(phase_error_to_pyerr)?;

        let dict = PyDict::new(py);
        dict.set_item("n_scenarios", summary.n_scenarios)?;
        dict.set_item("completed", summary.completed)?;
        Ok(dict.into())
    }
}
