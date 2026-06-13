//! The [`HighsSolver`] handle wrapper, its lifecycle/solve primitives, and the
//! `highs_version` free function.
//!
//! Owns the `HighsSolver` struct definition, the `unsafe impl Send`, the
//! construction/configuration/solve helpers, the warm-start `solve_inner`
//! orchestration (determinism-sensitive), the `Drop` handle teardown, the
//! `highs_version` query, and the `test-support` accessor impl. The escalation
//! ladder (`retry`) and the `SolverInterface` impl (`interface`) are additional
//! `impl HighsSolver` blocks that reach this struct's fields and helpers via the
//! child-module hierarchy.

use std::os::raw::c_void;
use std::time::Instant;

use super::config::{HighsProfile, default_options};
use crate::{
    DEFAULT_PROFILE_HEURISTIC_SENTINEL, DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL, ffi,
    types::{SolutionView, SolverError, SolverStatistics},
};

/// `HiGHS` LP solver instance implementing [`SolverInterface`](crate::SolverInterface).
///
/// Owns an opaque `HiGHS` handle and pre-allocated buffers for solution
/// extraction, scratch i32 index conversion, and statistics accumulation.
///
/// Construct with [`HighsSolver::new`]. The handle is destroyed automatically
/// when the instance is dropped.
///
/// # Example
///
/// ```rust
/// use cobre_solver::{HighsSolver, SolverInterface};
///
/// let solver = HighsSolver::new().expect("HiGHS initialisation failed");
/// assert_eq!(solver.name(), "HiGHS");
/// ```
pub struct HighsSolver {
    /// Opaque pointer to the `HiGHS` C++ instance, obtained from `cobre_highs_create()`.
    pub(super) handle: *mut c_void,
    /// Pre-allocated buffer for primal column values extracted after each solve.
    /// Resized in `load_model`; reused across solves to avoid per-solve allocation.
    pub(super) col_value: Vec<f64>,
    /// Pre-allocated buffer for column dual values (reduced costs from `HiGHS` perspective).
    /// Resized in `load_model`.
    pub(super) col_dual: Vec<f64>,
    /// Pre-allocated buffer for row primal values (constraint activity).
    /// Resized in `load_model`.
    pub(super) row_value: Vec<f64>,
    /// Pre-allocated buffer for row dual multipliers (shadow prices).
    /// Resized in `load_model`.
    pub(super) row_dual: Vec<f64>,
    /// Scratch buffer for converting `usize` indices to `i32` for the `HiGHS` C API.
    /// Used by `add_rows`, `set_row_bounds`, and `set_col_bounds`.
    /// Never shrunk -- only grows -- to prevent reallocation churn on the hot path.
    pub(super) scratch_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for column basis status codes.
    /// Reused across warm-start `solve` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_cols`; never shrunk.
    pub(super) basis_col_i32: Vec<i32>,
    /// Pre-allocated i32 buffer for row basis status codes.
    /// Reused across warm-start `solve` and `get_basis` calls to avoid per-call allocation.
    /// Resized in `load_model` to `num_rows` and grown in `add_rows`.
    pub(super) basis_row_i32: Vec<i32>,
    /// Scratch buffer for dual-ray extraction in `interpret_terminal_status` (dual).
    /// Grown lazily to `num_rows` via `resize`; contents are discarded after classification.
    /// Retained across calls so repeated non-optimal solves do not re-allocate.
    pub(super) terminal_status_dual_scratch: Vec<f64>,
    /// Scratch buffer for primal-ray extraction in `interpret_terminal_status` (primal).
    /// Grown lazily to `num_cols` via `resize`; contents are discarded after classification.
    /// Retained across calls so repeated non-optimal solves do not re-allocate.
    pub(super) terminal_status_primal_scratch: Vec<f64>,
    /// Current number of LP columns (decision variables), updated by `load_model` and `add_rows`.
    pub(super) num_cols: usize,
    /// Current number of LP rows (constraints), updated by `load_model` and `add_rows`.
    pub(super) num_rows: usize,
    /// Whether a model is currently loaded. Set to `true` in `load_model`,
    /// `false` in `reset` and `new`. Guards `solve`/`get_basis` contract.
    pub(super) has_model: bool,
    /// Accumulated solver statistics. Counters grow monotonically from zero;
    /// not reset by `reset()`.
    pub(super) stats: SolverStatistics,
    /// Cached solver profile applied by the last `set_*_profile` call.
    ///
    /// Initialised to `HighsProfile::default()` at construction, which
    /// preserves the historical hardcoded behaviour bit-for-bit. Updated by
    /// the four `SolverInterface` profile setter methods; read by
    /// `set_iteration_limits` on every solve attempt.
    pub(super) current_profile: HighsProfile,
}

// SAFETY: `HighsSolver` holds a raw pointer to a `HiGHS` C++ object. The `HiGHS`
// handle is not thread-safe for concurrent access, but exclusive ownership is
// maintained at all times -- exactly one `HighsSolver` instance owns each
// handle and no shared references to the handle exist. Transferring the
// `HighsSolver` to another thread (via `Send`) is safe because there is no
// concurrent access; the new thread has exclusive ownership. `Sync` is
// intentionally NOT implemented per `HiGHS` Implementation SS6.3.
unsafe impl Send for HighsSolver {}

impl HighsSolver {
    /// Creates a new `HiGHS` solver instance with performance-tuned defaults.
    ///
    /// Calls `cobre_highs_create()` to allocate the `HiGHS` handle, then applies
    /// the thirteen default options defined in `HiGHS` Implementation SS4.1:
    ///
    /// | Option                                      | Value       | Type   |
    /// |---------------------------------------------|-------------|--------|
    /// | `solver`                                    | `"simplex"` | string |
    /// | `simplex_strategy`                          | `1`         | int    |
    /// | `simplex_scale_strategy`                    | `0`         | int    |
    /// | `presolve`                                  | `"on"`      | string |
    /// | `parallel`                                  | `"off"`     | string |
    /// | `output_flag`                               | `0`         | bool   |
    /// | `primal_feasibility_tolerance`              | `1e-9`      | double |
    /// | `dual_feasibility_tolerance`                | `1e-9`      | double |
    /// | `simplex_dual_edge_weight_strategy`         | `1`         | int    |
    /// | `dual_simplex_cost_perturbation_multiplier` | `0.0`       | double |
    /// | `simplex_initial_condition_check`           | `0`         | bool   |
    /// | `simplex_price_strategy`                    | `1`         | int    |
    /// | `rebuild_refactor_solution_error_tolerance` | `1e-6`      | double |
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if:
    /// - `cobre_highs_create()` returns a null pointer.
    /// - Any configuration call returns `HIGHS_STATUS_ERROR`.
    ///
    /// In both failure cases the `HiGHS` handle is destroyed before returning to
    /// prevent a resource leak.
    pub fn new() -> Result<Self, SolverError> {
        // SAFETY: `cobre_highs_create` is a C function with no preconditions.
        // It allocates and returns a new `HiGHS` instance, or null on allocation
        // failure. The returned pointer is opaque and must be passed back to
        // `HiGHS` API functions.
        let handle = unsafe { ffi::cobre_highs_create() };

        if handle.is_null() {
            return Err(SolverError::InternalError {
                message: "HiGHS instance creation failed: Highs_create() returned null".to_string(),
                error_code: None,
            });
        }

        // Apply performance-tuned configuration. On any failure, destroy the
        // handle before returning to prevent a resource leak.
        if let Err(e) = Self::apply_default_config(handle) {
            // SAFETY: `handle` is a valid, non-null pointer obtained from
            // `cobre_highs_create()` in this same function. It has not been
            // passed to `cobre_highs_destroy()` yet. After this call, `handle`
            // must not be used again -- this function returns immediately with Err.
            unsafe { ffi::cobre_highs_destroy(handle) };
            return Err(e);
        }

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_value: Vec::new(),
            row_dual: Vec::new(),
            scratch_i32: Vec::new(),
            basis_col_i32: Vec::new(),
            basis_row_i32: Vec::new(),
            terminal_status_dual_scratch: Vec::new(),
            terminal_status_primal_scratch: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics {
                retry_level_histogram: vec![0u64; 12],
                ..SolverStatistics::default()
            },
            current_profile: HighsProfile::default(),
        })
    }

    /// Applies the thirteen performance-tuned `HiGHS` configuration options.
    ///
    /// Called once during construction. Returns `Ok(())` if all options are set
    /// successfully, or `Err(SolverError::InternalError)` with the failing
    /// option name if any configuration call returns `HIGHS_STATUS_ERROR`.
    fn apply_default_config(handle: *mut c_void) -> Result<(), SolverError> {
        for opt in &default_options() {
            // SAFETY: `handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(handle) };
            if status == ffi::HIGHS_STATUS_ERROR {
                return Err(SolverError::InternalError {
                    message: format!(
                        "HiGHS configuration failed: {}",
                        opt.name.to_str().unwrap_or("?")
                    ),
                    error_code: Some(status),
                });
            }
        }
        Ok(())
    }

    /// Extracts the optimal solution from `HiGHS` into pre-allocated buffers and returns
    /// a [`SolutionView`] borrowing directly from those buffers.
    ///
    /// The returned view borrows solver-internal buffers and is valid until the next
    /// `&mut self` call. `col_dual` is the reduced cost vector. Row duals follow the
    /// canonical sign convention (per Solver Abstraction SS8).
    pub(super) fn extract_solution_view(&mut self, solve_time_seconds: f64) -> SolutionView<'_> {
        // SAFETY: buffers resized in `load_model`/`add_rows`; HiGHS writes within bounds.
        let status = unsafe {
            ffi::cobre_highs_get_solution(
                self.handle,
                self.col_value.as_mut_ptr(),
                self.col_dual.as_mut_ptr(),
                self.row_value.as_mut_ptr(),
                self.row_dual.as_mut_ptr(),
            )
        };
        // HiGHS documentation guarantees `cobre_highs_get_solution` returns
        // non-ERROR status after `OPTIMAL` model status; this is a
        // debug-build-only invariant check.
        debug_assert_ne!(
            status,
            ffi::HIGHS_STATUS_ERROR,
            "cobre_highs_get_solution failed after optimal solve; HiGHS invariant violation"
        );

        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let objective = unsafe { ffi::cobre_highs_get_objective_value(self.handle) };

        // SAFETY: iteration count is non-negative so cast is safe.
        #[allow(clippy::cast_sign_loss)]
        let iterations =
            unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;

        SolutionView {
            objective,
            primal: &self.col_value[..self.num_cols],
            dual: &self.row_dual[..self.num_rows],
            reduced_costs: &self.col_dual[..self.num_cols],
            iterations,
            solve_time_seconds,
        }
    }

    /// Re-applies the current profile's feasibility tolerances to the `HiGHS` instance.
    ///
    /// Called immediately after `restore_default_settings()` in the retry-escalation
    /// finalization path so that `HiGHS` state and `current_profile` remain in sync.
    /// `restore_default_settings` resets the tolerances to the hardcoded table values
    /// (`1e-9`); this helper layers the caller's profile values on top.
    ///
    /// The iteration limits are not re-applied here because `restore_iteration_limits`
    /// always follows immediately and sets them to `i32::MAX` (unconstrained for the
    /// post-retry default-attempt path).
    pub(super) fn apply_profile_tolerances(&mut self) {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer obtained from
        // `cobre_highs_create()`. Option names are static C string literals with no
        // retained pointer after the call returns.
        unsafe {
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"primal_feasibility_tolerance".as_ptr(),
                self.current_profile.primal_feasibility_tolerance,
            );
            ffi::cobre_highs_set_double_option(
                self.handle,
                c"dual_feasibility_tolerance".as_ptr(),
                self.current_profile.dual_feasibility_tolerance,
            );
            // Also re-apply the algorithmic strategy int options. Post-retry
            // `restore_default_settings` resets these to HiGHS defaults; the
            // profile values must be reinstalled before the default-attempt
            // path runs so backward-tuned profiles survive the retry boundary.
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_dual_edge_weight_strategy".as_ptr(),
                self.current_profile.simplex_dual_edge_weight_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_scale_strategy".as_ptr(),
                self.current_profile.simplex_scale_strategy,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_price_strategy".as_ptr(),
                self.current_profile.simplex_price_strategy,
            );
        }
    }

    /// Restores default options after retry escalation.
    ///
    /// Status codes are checked via `debug_assert!` to catch programming
    /// errors during development (e.g., invalid option name). In release
    /// builds, failures are silently ignored since we are already on the
    /// recovery path.
    pub(super) fn restore_default_settings(&mut self) {
        for opt in &default_options() {
            // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
            let status = unsafe { opt.apply(self.handle) };
            debug_assert_eq!(
                status,
                ffi::HIGHS_STATUS_OK,
                "restore_default_settings: option {:?} failed with status {status}",
                opt.name,
            );
        }
    }

    /// Runs the solver once and returns the raw `HiGHS` model status.
    pub(super) fn run_once(&mut self) -> i32 {
        // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
        let run_status = unsafe { ffi::cobre_highs_run(self.handle) };
        if run_status == ffi::HIGHS_STATUS_ERROR {
            return ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR;
        }
        // SAFETY: same.
        unsafe { ffi::cobre_highs_get_model_status(self.handle) }
    }

    /// Sets per-solve iteration limits before a `run_once()` call.
    ///
    /// Simplex cap: if `current_profile.simplex_iteration_limit` equals
    /// [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`), the historical heuristic
    /// `max(100_000, 50 × num_cols)` is used. Any non-zero profile value is
    /// applied verbatim (clamped to `i32::MAX` for the FFI call).
    ///
    /// IPM cap: if `current_profile.ipm_iteration_limit` equals
    /// [`DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL`] (`0`), `i32::MAX` is sent to
    /// `HiGHS` (no cap). Any positive value is applied verbatim (clamped to
    /// `i32::MAX` for the FFI call). The `Default` value is `10_000`, so
    /// existing callers see no behavioural change.
    ///
    /// **Note on `time_limit`**: `HiGHS` tracks elapsed time cumulatively from
    /// instance creation, not per-`run()` call — neither `clear_solver()` nor
    /// option changes reset the internal timer. This makes `time_limit`
    /// unusable for the scenario-loop pattern (thousands of solves per
    /// instance). Wall-clock measurement via `Instant` is used instead for
    /// time-based budget management.
    pub(super) fn set_iteration_limits(&mut self) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let simplex_iter_limit: i32 =
            if self.current_profile.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL {
                // Heuristic fallback: scale with LP size to avoid runaway cycling.
                let heuristic = self.num_cols.saturating_mul(50).max(100_000);
                // `heuristic` is bounded by `usize::MAX`, but realistic LP sizes
                // are well below `i32::MAX` (≈2.1 × 10^9 cols). Clamp defensively.
                (heuristic.min(i32::MAX as usize)) as i32
            } else {
                // Profile literal value: apply verbatim, clamped for FFI cast.
                // `.min(i32::MAX as u32)` ensures the value fits; the cast cannot wrap.
                (self
                    .current_profile
                    .simplex_iteration_limit
                    .min(i32::MAX as u32)) as i32
            };

        // IPM cap: 0 is the "unbounded" sentinel per trait contract; map it to
        // i32::MAX so HiGHS does not interpret 0 as "no iterations allowed".
        // Any positive value is applied verbatim (clamped for the FFI cast).
        #[allow(clippy::cast_possible_wrap)]
        let ipm_iter_limit: i32 =
            if self.current_profile.ipm_iteration_limit == DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL {
                i32::MAX // "unbounded" per trait contract
            } else {
                (self
                    .current_profile
                    .ipm_iteration_limit
                    .min(i32::MAX as u32)) as i32
            };

        // SAFETY: handle is valid non-null HiGHS pointer; option names are
        // static C strings with no retained pointers.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                simplex_iter_limit,
            );
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"ipm_iteration_limit".as_ptr(),
                ipm_iter_limit,
            );
        }
    }

    /// Restores iteration limits to their unconstrained defaults.
    ///
    /// Called after `retry_escalation` completes (regardless of outcome).
    pub(super) fn restore_iteration_limits(&mut self) {
        // SAFETY: handle is valid non-null HiGHS pointer.
        unsafe {
            ffi::cobre_highs_set_int_option(
                self.handle,
                c"simplex_iteration_limit".as_ptr(),
                i32::MAX,
            );
            ffi::cobre_highs_set_int_option(self.handle, c"ipm_iteration_limit".as_ptr(), i32::MAX);
        }
    }

    /// Interprets a non-optimal status as a terminal `SolverError`.
    ///
    /// Returns `None` for `SOLVE_ERROR` or `UNKNOWN` (retry continues),
    /// or `Some(error)` for terminal statuses.
    pub(super) fn interpret_terminal_status(
        &mut self,
        status: i32,
        solve_time_seconds: f64,
    ) -> Option<SolverError> {
        match status {
            ffi::HIGHS_MODEL_STATUS_OPTIMAL => {
                // Caller should have handled optimal before reaching here.
                None
            }
            ffi::HIGHS_MODEL_STATUS_INFEASIBLE => Some(SolverError::Infeasible),
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => {
                // Probe for a dual ray to classify as Infeasible, then a primal
                // ray to classify as Unbounded. The ray values are not stored in
                // the error -- only the classification matters.
                //
                // `num_rows` and `num_cols` are up-to-date because `load_model`
                // and `add_rows` always update them before any solve that could
                // reach this branch. The `resize` below matches the exact count
                // that HiGHS writes into the buffer.
                let mut has_dual_ray: i32 = 0;
                self.terminal_status_dual_scratch.resize(self.num_rows, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_dual_scratch` has been resized to at least
                // `self.num_rows` elements; HiGHS writes exactly `num_rows` values.
                let dual_status = unsafe {
                    ffi::cobre_highs_get_dual_ray(
                        self.handle,
                        &raw mut has_dual_ray,
                        self.terminal_status_dual_scratch.as_mut_ptr(),
                    )
                };
                if dual_status != ffi::HIGHS_STATUS_ERROR && has_dual_ray != 0 {
                    return Some(SolverError::Infeasible);
                }
                let mut has_primal_ray: i32 = 0;
                self.terminal_status_primal_scratch
                    .resize(self.num_cols, 0.0);
                // SAFETY: `self.handle` is a valid, non-null HiGHS pointer.
                // `terminal_status_primal_scratch` has been resized to at least
                // `self.num_cols` elements; HiGHS writes exactly `num_cols` values.
                let primal_status = unsafe {
                    ffi::cobre_highs_get_primal_ray(
                        self.handle,
                        &raw mut has_primal_ray,
                        self.terminal_status_primal_scratch.as_mut_ptr(),
                    )
                };
                if primal_status != ffi::HIGHS_STATUS_ERROR && has_primal_ray != 0 {
                    return Some(SolverError::Unbounded);
                }
                Some(SolverError::Infeasible)
            }
            ffi::HIGHS_MODEL_STATUS_UNBOUNDED => Some(SolverError::Unbounded),
            ffi::HIGHS_MODEL_STATUS_TIME_LIMIT => Some(SolverError::TimeLimitExceeded {
                elapsed_seconds: solve_time_seconds,
            }),
            ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT => {
                // SAFETY: handle is valid non-null pointer; iteration count is non-negative.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                Some(SolverError::IterationLimit { iterations })
            }
            ffi::HIGHS_MODEL_STATUS_SOLVE_ERROR | ffi::HIGHS_MODEL_STATUS_UNKNOWN => {
                // Signal to the caller that retry should continue.
                None
            }
            other => Some(SolverError::InternalError {
                message: format!("HiGHS returned unexpected model status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Converts `usize` indices to `i32` in the internal scratch buffer.
    ///
    /// Grows but never shrinks the buffer. Each element is debug-asserted to fit in i32.
    pub(super) fn convert_to_i32_scratch(&mut self, source: &[usize]) -> &[i32] {
        if source.len() > self.scratch_i32.len() {
            self.scratch_i32.resize(source.len(), 0);
        }
        for (i, &v) in source.iter().enumerate() {
            debug_assert!(
                i32::try_from(v).is_ok(),
                "usize index {v} overflows i32::MAX at position {i}"
            );
            // SAFETY: debug_assert verifies v fits in i32; cast to HiGHS C API i32.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                self.scratch_i32[i] = v as i32;
            }
        }
        &self.scratch_i32[..source.len()]
    }

    /// Internal helper: run the simplex and update stats.
    ///
    /// Core simplex execution, called after (for warm-start) the basis has been
    /// installed. `HiGHS` retains its internal simplex basis across consecutive
    /// `solve_inner` calls on the same LP shape, which is the primary warm-start
    /// mechanism for the backward pass. No `Highs_clearSolver` call is issued —
    /// clearing the solver discards the retained basis and forfeits the warm start.
    pub(super) fn solve_inner(&mut self) -> Result<SolutionView<'_>, SolverError> {
        // Safeguard: apply iteration limits before the initial attempt.
        // Time limits are NOT set here — HiGHS tracks time cumulatively from
        // instance creation, so a per-solve time_limit would fire spuriously
        // on long-running solver instances. Instead, wall-clock time is checked
        // after run_once() to detect stuck solves.
        self.set_iteration_limits();

        let t0 = Instant::now();
        let model_status = self.run_once();
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if model_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
            // Read iteration count from FFI BEFORE establishing the shared borrow
            // via extract_solution_view, so stats can be updated without violating
            // the aliasing rules.
            // SAFETY: handle is valid non-null HiGHS pointer.
            #[allow(clippy::cast_sign_loss)]
            let iterations =
                unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
            self.stats.success_count += 1;
            self.stats.first_try_successes += 1;
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;
            self.restore_iteration_limits();
            return Ok(self.extract_solution_view(solve_time));
        }

        // Check for a definitive terminal status (not a retry-able error).
        // UNBOUNDED is retried: HiGHS dual simplex can report spurious UNBOUNDED
        // on numerically difficult LPs with wide coefficient ranges. The retry
        // escalation (especially presolve in the core sequence) often resolves these.
        // ITERATION_LIMIT from the initial attempt is retryable — the retry
        // sequence uses different strategies that may converge faster.
        // TIME_LIMIT is retryable — HiGHS tracks time cumulatively from instance
        // creation; a spurious TIME_LIMIT can fire even with time_limit=Infinity
        // in edge cases. Retry level 0 (cold restart) recovers from this.
        // Wall-clock > 15s is also retryable — detects stuck initial solves.
        let is_unbounded = model_status == ffi::HIGHS_MODEL_STATUS_UNBOUNDED;
        // INFEASIBLE from the initial attempt is retryable. A warm-started dual
        // simplex can FALSELY report INFEASIBLE on numerically difficult LPs --
        // the same warm-start fragility that surfaces as spurious UNBOUNDED or
        // ITERATION_LIMIT (retried above). Routing INFEASIBLE through the
        // escalation runs level 0 first, which clears the warm basis
        // (`cobre_highs_clear_solver`) and re-solves cold. A *genuinely* infeasible
        // LP is confirmed by that cold solve and the escalation stops immediately
        // (INFEASIBLE is terminal inside `retry_escalation`), so a true infeasible
        // still returns `Err(Infeasible)` -- only one extra cold solve. A
        // warm-start-only false infeasible is rescued (returns optimal). Without
        // this, a false INFEASIBLE bypassed the escalation entirely and became a
        // fatal training error. Mirrors the cold-solve escalation on the CLP path.
        let is_infeasible = model_status == ffi::HIGHS_MODEL_STATUS_INFEASIBLE;
        let initial_retryable = is_unbounded
            || is_infeasible
            || model_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
            || model_status == ffi::HIGHS_MODEL_STATUS_TIME_LIMIT
            || solve_time > 15.0;
        if !initial_retryable
            && let Some(terminal_err) = self.interpret_terminal_status(model_status, solve_time)
        {
            self.restore_iteration_limits();
            self.stats.failure_count += 1;
            return Err(terminal_err);
        }

        // Delegate to the retry escalation method (restores limits internally).
        match self.retry_escalation(is_unbounded) {
            Ok(outcome) => {
                self.stats.retry_count += outcome.attempts;
                self.stats.success_count += 1;
                self.stats.total_iterations += outcome.iterations;
                self.stats.total_solve_time_seconds += outcome.solve_time;
                self.stats.retry_level_histogram[outcome.level as usize] += 1;
                Ok(self.extract_solution_view(outcome.solve_time))
            }
            Err((attempts, err)) => {
                self.stats.retry_count += attempts;
                self.stats.failure_count += 1;
                Err(err)
            }
        }
    }
}

impl Drop for HighsSolver {
    fn drop(&mut self) {
        // SAFETY: valid HiGHS pointer from construction, called once per instance.
        unsafe { ffi::cobre_highs_destroy(self.handle) };
    }
}

/// Returns the `HiGHS` version as a `"major.minor.patch"` string.
///
/// This is a free function — no solver instance is required.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "highs")]
/// # {
/// let v = cobre_solver::highs_version();
/// assert!(v.contains('.'), "version string should be 'major.minor.patch'");
/// # }
/// ```
#[must_use]
pub fn highs_version() -> String {
    // SAFETY: These are pure query functions with no arguments. The HiGHS C API
    // documents them as safe to call without any prior initialisation; they read
    // only compile-time constants embedded in the library.
    let major = unsafe { crate::ffi::cobre_highs_version_major() };
    let minor = unsafe { crate::ffi::cobre_highs_version_minor() };
    let patch = unsafe { crate::ffi::cobre_highs_version_patch() };
    format!("{major}.{minor}.{patch}")
}

/// Test-support accessors for integration tests that need to set raw `HiGHS` options.
///
/// Gated behind the `test-support` feature. The raw handle is intentionally not
/// part of the public API — callers use these methods to configure time/iteration
/// limits before a solve without going through the safe wrapper.
#[cfg(feature = "test-support")]
impl HighsSolver {
    /// Returns the raw `HiGHS` handle for use with test-support FFI helpers.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of `self`. The caller must
    /// not store the pointer beyond that lifetime, must not call
    /// `cobre_highs_destroy` on it, and must not alias it across threads.
    #[must_use]
    pub fn raw_handle(&self) -> *mut std::os::raw::c_void {
        self.handle
    }

    /// Invoke `apply_retry_level_options` for a given level.
    ///
    /// Thin test-support wrapper that exposes the private retry-level applier
    /// so integration tests can verify option composition without driving a
    /// full solve through the retry ladder.
    ///
    /// Only levels 0-4 are routed through this method; levels 5-11 delegate to
    /// `apply_extended_retry_options_for_test`. Call the appropriate method
    /// based on the level you want to test.
    pub fn apply_retry_level_options_for_test(&mut self, level: u32) {
        self.apply_retry_level_options(level);
    }

    /// Invoke `apply_extended_retry_options` for a given level (5-11).
    ///
    /// Thin test-support wrapper that exposes the private extended retry-level
    /// applier so integration tests can verify option composition without
    /// driving a full solve.
    pub fn apply_extended_retry_options_for_test(&mut self, level: u32) {
        self.apply_extended_retry_options(level);
    }

    /// Invoke `restore_default_settings` then `apply_profile_tolerances` in sequence.
    ///
    /// Mirrors the finalization path in `retry_escalation` so integration tests
    /// can verify that profile tolerances survive a defaults-restore without
    /// driving a full retry through a failing LP.
    pub fn restore_defaults_then_apply_profile_for_test(&mut self) {
        self.restore_default_settings();
        self.apply_profile_tolerances();
    }

    /// Read a double-valued `HiGHS` option by name.
    ///
    /// Returns `None` if the option name is unknown to `HiGHS`; `Some(value)`
    /// on success.
    #[must_use]
    pub fn get_double_option(&self, option: &std::ffi::CStr) -> Option<f64> {
        let mut out = 0.0_f64;
        // SAFETY: handle is valid non-null HiGHS pointer; option is a valid
        // null-terminated C string borrowed for the duration of the call;
        // `out` is stack-allocated and written by HiGHS on success.
        let status = unsafe {
            ffi::cobre_highs_get_double_option(self.handle, option.as_ptr(), &raw mut out)
        };
        if status == ffi::HIGHS_STATUS_ERROR {
            None
        } else {
            Some(out)
        }
    }

    /// Read an integer-valued `HiGHS` option by name.
    ///
    /// Returns `None` if the option name is unknown to `HiGHS`; `Some(value)`
    /// on success.
    #[must_use]
    pub fn get_int_option(&self, option: &std::ffi::CStr) -> Option<i32> {
        let mut out = 0_i32;
        // SAFETY: handle is valid non-null HiGHS pointer; option is a valid
        // null-terminated C string borrowed for the duration of the call;
        // `out` is stack-allocated and written by HiGHS on success.
        let status =
            unsafe { ffi::cobre_highs_get_int_option(self.handle, option.as_ptr(), &raw mut out) };
        if status == ffi::HIGHS_STATUS_ERROR {
            None
        } else {
            Some(out)
        }
    }
}
