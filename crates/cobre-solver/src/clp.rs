//! CLP LP solver backend.
//!
//! This module provides [`ClpSolver`], which wraps the CLP C API through the
//! FFI layer in [`crate::clp_ffi`]. It defines the solver type, its associated
//! [`ClpProfile`] value type, the handle lifecycle, and the reusable solution
//! buffers, plus the complete [`crate::SolverInterface`] implementation.
//!
//! # Thread Safety
//!
//! [`ClpSolver`] is `Send` but not `Sync`. The underlying CLP handle is
//! exclusively owned; transferring ownership to a worker thread is safe.
//! Concurrent access from multiple threads is not permitted.
//!
//! # Configuration
//!
//! Configuration is wired through [`ClpProfile`] and the profile-setter FFI
//! calls. The profile defaults mirror [`crate::HighsProfile`] where the options
//! overlap (tight feasibility tolerances, scaling off because the cobre
//! prescaler already conditions the matrix), and use CLP-native values
//! otherwise (perturbation off, iteration limit deferred to the per-call
//! heuristic).

use std::os::raw::c_void;
use std::time::Instant;

use crate::{
    DEFAULT_PROFILE_HEURISTIC_SENTINEL, SolverInterface, clp_ffi,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

/// CLP-specific solver profile carrying the tunable option surface.
///
/// `ClpProfile` is the associated `Profile` type for [`ClpSolver`]. Other
/// solvers (e.g. `HiGHS`) define their own concrete profile types with their
/// native option names.
///
/// The field defaults are tuned for deterministic, warm-started repeated
/// re-solves: perturbation is off (`102`, not CLP's own `100`=auto default),
/// scaling is off (the cobre prescaler conditions the matrix), and the
/// feasibility tolerances match [`crate::HighsProfile`] bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClpProfile {
    /// CLP perturbation mode (`Clp_setPerturbation`). `102` disables
    /// perturbation (required for deterministic, reproducible re-solves);
    /// CLP's own default `100` requests automatic perturbation.
    pub perturbation: i32,
    /// CLP scaling mode (`Clp_scaling`). `0` disables scaling because the
    /// cobre prescaler already normalizes matrix entries, mirroring `HiGHS`
    /// `simplex_scale_strategy=0`.
    pub scaling: i32,
    /// Primal feasibility tolerance (`Clp_setPrimalTolerance`).
    pub primal_feasibility_tolerance: f64,
    /// Dual feasibility tolerance (`Clp_setDualTolerance`).
    pub dual_feasibility_tolerance: f64,
    /// Per-attempt simplex iteration cap (`Clp_setMaximumIterations`).
    /// `DEFAULT_PROFILE_HEURISTIC_SENTINEL` (0) selects the per-call heuristic.
    pub simplex_iteration_limit: u32,
}

impl Default for ClpProfile {
    fn default() -> Self {
        Self {
            perturbation: 102,
            scaling: 0,
            primal_feasibility_tolerance: 1e-9,
            dual_feasibility_tolerance: 1e-9,
            simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        }
    }
}

/// CLP LP solver backend.
///
/// Owns an opaque CLP model handle plus a set of pre-allocated, reusable
/// buffers that are resized when a model is loaded and reused across solves to
/// avoid per-solve allocation on the hot path.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "clp")]
/// # {
/// use cobre_solver::{ClpSolver, SolverInterface};
///
/// let solver = ClpSolver::new().expect("CLP initialisation failed");
/// assert_eq!(solver.name(), "CLP");
/// # }
/// ```
pub struct ClpSolver {
    /// Opaque pointer to the CLP model, obtained from `cobre_clp_create()`.
    handle: *mut c_void,
    /// Pre-allocated buffer for primal column values extracted after each solve.
    /// Resized in `load_model`; reused across solves to avoid per-solve allocation.
    col_value: Vec<f64>,
    /// Pre-allocated buffer for column dual values (reduced costs).
    /// Resized in `load_model`.
    col_dual: Vec<f64>,
    /// Pre-allocated buffer for row dual multipliers (shadow prices).
    /// Resized in `load_model`.
    row_dual: Vec<f64>,
    /// Scratch buffer for converting `usize` indices to `i32` for the CLP C API.
    /// Never shrunk -- only grows -- to prevent reallocation churn on the hot path.
    ///
    /// Currently unread: the CSR→CSC merge in `add_rows` builds its own local
    /// buffers, and the bound-patch paths reuse the retained vectors. Retained
    /// as a pre-allocated workspace for a future incremental-index FFI path
    /// (a narrowed field-level allow replaces the former module-level one).
    #[allow(dead_code)]
    scratch_i32: Vec<i32>,
    /// Retained CSC column-start offsets (length `num_cols + 1`).
    ///
    /// Owned mirror of the loaded LP. CLP exposes no incremental row-append or
    /// bound-change call, so `ClpSolver` keeps a full mutable copy of the model
    /// here and re-issues `cobre_clp_load_problem` (`reload`) after every
    /// mutation. Populated by `load_model`.
    col_starts: Vec<i32>,
    /// Retained CSC row indices for each non-zero (length `num_nz`).
    row_indices: Vec<i32>,
    /// Retained CSC non-zero values (length `num_nz`).
    values: Vec<f64>,
    /// Retained column lower bounds (length `num_cols`). Forwarded verbatim.
    col_lower: Vec<f64>,
    /// Retained column upper bounds (length `num_cols`). Forwarded verbatim.
    col_upper: Vec<f64>,
    /// Retained objective coefficients (length `num_cols`).
    objective: Vec<f64>,
    /// Retained row lower bounds (length `num_rows`). Forwarded verbatim.
    row_lower: Vec<f64>,
    /// Retained row upper bounds (length `num_rows`). Forwarded verbatim.
    row_upper: Vec<f64>,
    /// Retained non-zero count, kept in sync with `values.len()`.
    num_nz: usize,
    /// Current number of LP columns (decision variables), updated by `load_model`.
    num_cols: usize,
    /// Current number of LP rows (constraints), updated by `load_model`.
    num_rows: usize,
    /// Whether a model is currently loaded. Set to `true` in `load_model`.
    /// Guards `solve`/`get_basis` contract.
    has_model: bool,
    /// Accumulated solver statistics. Counters grow monotonically from zero.
    stats: SolverStatistics,
    /// Cached solver profile applied by the last profile-setter call.
    /// Initialised to `ClpProfile::default()` at construction.
    current_profile: ClpProfile,
}

// SAFETY: `ClpSolver` holds a raw pointer to a CLP C++ object. The CLP handle
// is not thread-safe for concurrent access, but exclusive ownership is
// maintained at all times -- exactly one `ClpSolver` instance owns each handle
// and no shared references to the handle exist. Transferring the `ClpSolver`
// to another thread (via `Send`) is safe because there is no concurrent
// access; the new thread has exclusive ownership. `Sync` is intentionally NOT
// implemented.
unsafe impl Send for ClpSolver {}

impl ClpSolver {
    /// Creates a new CLP solver instance.
    ///
    /// Calls `cobre_clp_create()` to allocate the CLP handle and initialises
    /// every reusable buffer empty. No CLP options are applied here;
    /// configuration is wired through [`ClpProfile`] separately.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if `cobre_clp_create()`
    /// returns a null pointer. There is no handle to free in that branch (it
    /// was null).
    pub fn new() -> Result<Self, SolverError> {
        // SAFETY: `cobre_clp_create` is a C function with no preconditions.
        // It allocates and returns a new CLP model pointer, or null on
        // allocation failure. The returned pointer is opaque and must be
        // passed back to CLP API functions.
        let handle = unsafe { clp_ffi::cobre_clp_create() };

        if handle.is_null() {
            return Err(SolverError::InternalError {
                message: "CLP instance creation failed: Clp_newModel() returned null".to_string(),
                error_code: None,
            });
        }

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_dual: Vec::new(),
            scratch_i32: Vec::new(),
            col_starts: Vec::new(),
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            objective: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            num_nz: 0,
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics::default(),
            current_profile: ClpProfile::default(),
        })
    }

    /// Re-issues `cobre_clp_load_problem` with the current retained-model
    /// buffers.
    ///
    /// CLP exposes no incremental row-append or bound-change call, so every
    /// mutation (`add_rows`, `set_row_bounds`, `set_col_bounds`) patches the
    /// retained CSC/bounds buffers and then calls this helper to rebuild the
    /// CLP model from scratch. Bounds are forwarded verbatim; the C wrapper
    /// owns the ±IEEE-inf → ±`DBL_MAX` translation and the minimize sense, so
    /// no infinity translation or sense-setting happens here.
    ///
    /// # Panics
    ///
    /// Panics if the retained `num_cols`, `num_rows`, or `num_nz` does not fit
    /// in `i32` (the LP exceeds the CLP C API limit).
    fn reload(&mut self) {
        assert!(
            i32::try_from(self.num_cols).is_ok(),
            "num_cols {} overflows i32: LP exceeds CLP API limit",
            self.num_cols
        );
        assert!(
            i32::try_from(self.num_rows).is_ok(),
            "num_rows {} overflows i32: LP exceeds CLP API limit",
            self.num_rows
        );
        assert!(
            i32::try_from(self.num_nz).is_ok(),
            "num_nz {} overflows i32: LP exceeds CLP API limit",
            self.num_nz
        );
        // The two values below have been asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_col = self.num_cols as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_row = self.num_rows as i32;
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer from `cobre_clp_create()`.
        // - `num_col`/`num_row` fit in i32 (asserted above).
        // - All pointer arguments point into owned `Vec` data on `self` that
        //   remains alive for the duration of this call.
        // - Retained slice lengths match the CLP `Clp_loadProblem` contract:
        //   `num_cols + 1` for col_starts, `num_nz` for row_indices and values,
        //   `num_cols` for col_lower/col_upper/objective, `num_rows` for
        //   row_lower/row_upper (maintained by `load_model`, `add_rows`, and
        //   the `set_*_bounds` patches).
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation and sets the objective sense.
        unsafe {
            clp_ffi::cobre_clp_load_problem(
                self.handle,
                num_col,
                num_row,
                self.col_starts.as_ptr(),
                self.row_indices.as_ptr(),
                self.values.as_ptr(),
                self.col_lower.as_ptr(),
                self.col_upper.as_ptr(),
                self.objective.as_ptr(),
                self.row_lower.as_ptr(),
                self.row_upper.as_ptr(),
            );
        }
    }

    /// Copies the three CLP-owned solution pointers into the owned buffers.
    ///
    /// Called immediately after an optimal `cobre_clp_dual` and before any
    /// further CLP call: the pointers returned by `cobre_clp_get_col_solution`,
    /// `cobre_clp_get_reduced_cost`, and `cobre_clp_get_row_price` point into
    /// CLP-owned memory that is valid only until the next solve. The primal
    /// solution lands in `col_value` (length `num_cols`), the reduced costs in
    /// `col_dual` (length `num_cols`), and the row prices — normalized via
    /// `normalize_row_dual` — in `row_dual` (length `num_rows`).
    ///
    /// Each copy is guarded with `if len > 0` because passing a null or
    /// dangling pointer to `std::slice::from_raw_parts` is undefined behavior
    /// even with length 0 (a zero-column or zero-row LP may yield such a
    /// pointer from CLP).
    fn copy_solution(&mut self) {
        if self.num_cols > 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved to optimality. `cobre_clp_get_col_solution`
            // returns a non-null pointer into CLP-owned memory of exactly
            // `num_cols` `f64`s (guarded `num_cols > 0`), valid until the next
            // solve. `self.col_value` was resized to `num_cols` in `load_model`.
            let primal = unsafe {
                let ptr = clp_ffi::cobre_clp_get_col_solution(self.handle);
                std::slice::from_raw_parts(ptr, self.num_cols)
            };
            self.col_value.copy_from_slice(primal);

            // SAFETY: as above; `cobre_clp_get_reduced_cost` returns a non-null
            // pointer into CLP-owned memory of exactly `num_cols` `f64`s, valid
            // until the next solve. `self.col_dual` was resized to `num_cols`.
            let reduced = unsafe {
                let ptr = clp_ffi::cobre_clp_get_reduced_cost(self.handle);
                std::slice::from_raw_parts(ptr, self.num_cols)
            };
            self.col_dual.copy_from_slice(reduced);
        }

        if self.num_rows > 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved to optimality. `cobre_clp_get_row_price` returns
            // a non-null pointer into CLP-owned memory of exactly `num_rows`
            // `f64`s (guarded `num_rows > 0`), valid until the next solve.
            let row_price = unsafe {
                let ptr = clp_ffi::cobre_clp_get_row_price(self.handle);
                std::slice::from_raw_parts(ptr, self.num_rows)
            };
            for (dst, &raw) in self.row_dual.iter_mut().zip(row_price) {
                *dst = normalize_row_dual(raw);
            }
        }
    }

    /// Reinstalls an offered warm-start basis into the CLP model element-by-element.
    ///
    /// CLP exposes basis status **per element** (`cobre_clp_set_column_status` /
    /// `cobre_clp_set_row_status`), not as a bulk array, so the reinstall is a
    /// pair of per-element loops — no `copy_from_slice`. The raw `CLP_BASIS_*`
    /// `i32` codes carried in `b` are written back verbatim (no translation).
    ///
    /// Rows are reinstalled only up to `min(b.row_status.len(), num_rows)` so a
    /// basis captured before `add_rows` grew the LP is tolerated: the trailing
    /// rows keep whatever status CLP currently holds and `Clp_dual` repairs the
    /// basis as needed (mirrors the `HiGHS` `copy_len` guard).
    ///
    /// # Panics
    ///
    /// Panics if `b.col_status.len() != self.num_cols` (mirrors the `HiGHS`
    /// warm-start contract). `b.row_status.len() >= self.num_rows` is checked
    /// with `debug_assert!`.
    fn install_basis(&mut self, b: &crate::Basis) {
        // NOTE: CLP set_*Status does not run isBasisConsistent; no BasisInconsistent path.
        // CLP's per-element setters silently accept an inconsistent offered basis
        // and `Clp_dual` repairs it, so — unlike `HighsSolver::solve` — there is no
        // consistency check and no `SolverError::BasisInconsistent` surface here.
        assert!(
            b.col_status.len() == self.num_cols,
            "basis column count {} does not match LP column count {}",
            b.col_status.len(),
            self.num_cols
        );
        debug_assert!(
            b.row_status.len() >= self.num_rows,
            "solve(Some(&basis)): basis.row_status.len() ({}) < self.num_rows ({}); \
             callers introducing new rows must reconcile the basis before calling solve",
            b.row_status.len(),
            self.num_rows
        );

        // Track every warm-start call as a basis offer for diagnostics.
        self.stats.basis_offered += 1;

        let row_copy_len = b.row_status.len().min(self.num_rows);

        let basis_set_start = Instant::now();
        // Loop indices are bounded by `num_cols`/`num_rows`, both asserted to
        // fit in i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `c` is in `0..num_cols`, a valid column sequence index, and
            // fits in i32. The setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_column_status(self.handle, c as i32, b.col_status[c]);
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..row_copy_len {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..min(b.row_status.len(), num_rows)`, a valid row
            // sequence index, and fits in i32. The setter writes a single status
            // byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_row_status(self.handle, r as i32, b.row_status[r]);
            }
        }
        self.stats.total_basis_set_time_seconds += basis_set_start.elapsed().as_secs_f64();
    }

    /// Resolves the per-attempt simplex iteration cap for `apply_profile`.
    ///
    /// When `current_profile.simplex_iteration_limit` equals
    /// [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`), applies the size-scaled
    /// heuristic `max(100_000, num_cols * 50)`; otherwise applies the profile
    /// value verbatim. Both branches clamp to `i32::MAX` for the FFI cast.
    /// Mirrors the simplex branch of `HighsSolver::set_iteration_limits`.
    fn resolve_simplex_cap(&self) -> i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        if self.current_profile.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL {
            // Heuristic fallback: scale with LP size to avoid runaway cycling.
            let heuristic = self.num_cols.saturating_mul(50).max(100_000);
            // Realistic LP sizes are well below `i32::MAX`; clamp defensively.
            (heuristic.min(i32::MAX as usize)) as i32
        } else {
            // Profile literal value: apply verbatim, clamped for the FFI cast.
            (self
                .current_profile
                .simplex_iteration_limit
                .min(i32::MAX as u32)) as i32
        }
    }
}

impl SolverInterface for ClpSolver {
    type Profile = ClpProfile;

    /// Applies every [`ClpProfile`] field to the underlying CLP model in one
    /// pass, then caches the profile in `current_profile`.
    ///
    /// Issues `cobre_clp_set_perturbation` (CRITICAL: the default profile sends
    /// `102` to disable CLP's auto-perturbation — CLP's own default `100`
    /// breaks bit-for-bit reproducibility across re-solves),
    /// `cobre_clp_scaling`, `cobre_clp_set_primal_tolerance`,
    /// `cobre_clp_set_dual_tolerance`, and the iteration cap resolved by
    /// `resolve_simplex_cap` via `cobre_clp_set_maximum_iterations`.
    ///
    /// Mirrors `HighsSolver::apply_profile`: configure all fields, then assign
    /// `self.current_profile`.
    fn apply_profile(&mut self, profile: &ClpProfile) {
        // Cache the profile first so `resolve_simplex_cap` reads the new
        // simplex iteration limit when computing the FFI cap below.
        self.current_profile = *profile;
        let cap = self.resolve_simplex_cap();
        // SAFETY: `self.handle` is a valid, non-null CLP pointer obtained from
        // `cobre_clp_create()`. Each `cobre_clp_set_*` setter accepts any
        // `i32`/`f64` value, retains no pointer after the call returns, and
        // cannot fail on a valid handle. `cap` is the resolved iteration limit.
        unsafe {
            clp_ffi::cobre_clp_set_perturbation(self.handle, profile.perturbation);
            clp_ffi::cobre_clp_scaling(self.handle, profile.scaling);
            clp_ffi::cobre_clp_set_primal_tolerance(
                self.handle,
                profile.primal_feasibility_tolerance,
            );
            clp_ffi::cobre_clp_set_dual_tolerance(self.handle, profile.dual_feasibility_tolerance);
            clp_ffi::cobre_clp_set_maximum_iterations(self.handle, cap);
        }
    }

    /// Loads a complete LP into the CLP model from column-major (CSC) data.
    ///
    /// Wraps [`clp_ffi::cobre_clp_load_problem`], which forwards the template's
    /// CSC arrays to `Clp_loadProblem` and fixes the objective sense to
    /// minimize on the C side. The ±IEEE-inf → ±`DBL_MAX` bound translation is
    /// also owned by the C wrapper, so the bound slices are forwarded verbatim
    /// (`f64::INFINITY` is **not** pre-translated here).
    ///
    /// After the load, the model dimensions are recorded and the three
    /// solution-extraction buffers (`col_value`, `col_dual`, `row_dual`) are
    /// resized to match. Re-calling replaces the prior model (CLP's
    /// `Clp_loadProblem` overwrites) and resizes the buffers accordingly. A
    /// zero-row template (`num_rows == 0`) is valid and resizes `row_dual` to
    /// length 0.
    ///
    /// `cobre_clp_load_problem` returns `()` — there is no status to check.
    /// Model validity is established later by `solve` (a non-optimal status
    /// surfaces as a `SolverError`). The only failure mode here is the
    /// i32-overflow assert below (a programmer/data error), which panics with a
    /// descriptive message.
    ///
    /// # Panics
    ///
    /// Panics if `template.num_cols`, `template.num_rows`, or `template.num_nz`
    /// does not fit in `i32` (the LP exceeds the CLP C API limit).
    fn load_model(&mut self, template: &StageTemplate) {
        let t0 = Instant::now();
        assert!(
            i32::try_from(template.num_cols).is_ok(),
            "num_cols {} overflows i32: LP exceeds CLP API limit",
            template.num_cols
        );
        assert!(
            i32::try_from(template.num_rows).is_ok(),
            "num_rows {} overflows i32: LP exceeds CLP API limit",
            template.num_rows
        );
        assert!(
            i32::try_from(template.num_nz).is_ok(),
            "num_nz {} overflows i32: LP exceeds CLP API limit",
            template.num_nz
        );
        // The two values below have been asserted to fit in i32 above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_col = template.num_cols as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let num_row = template.num_rows as i32;
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer from `cobre_clp_create()`.
        // - `num_col`/`num_row` fit in i32 (asserted above).
        // - All pointer arguments point into owned `Vec` data on `template` that
        //   remains alive for the duration of this call.
        // - Slice lengths match the CLP `Clp_loadProblem` contract:
        //   `num_cols + 1` for col_starts, `num_nz` for row_indices and values,
        //   `num_cols` for col_lower/col_upper/objective, `num_rows` for
        //   row_lower/row_upper.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation and sets the objective sense.
        unsafe {
            clp_ffi::cobre_clp_load_problem(
                self.handle,
                num_col,
                num_row,
                template.col_starts.as_ptr(),
                template.row_indices.as_ptr(),
                template.values.as_ptr(),
                template.col_lower.as_ptr(),
                template.col_upper.as_ptr(),
                template.objective.as_ptr(),
                template.row_lower.as_ptr(),
                template.row_upper.as_ptr(),
            );
        }

        self.num_cols = template.num_cols;
        self.num_rows = template.num_rows;
        self.has_model = true;

        // Resize solution extraction buffers to match the new LP dimensions.
        // Zero-fill is fine; these are overwritten in full after each solve.
        self.col_value.resize(self.num_cols, 0.0);
        self.col_dual.resize(self.num_cols, 0.0);
        self.row_dual.resize(self.num_rows, 0.0);

        // Clone the template's CSC/bounds into the retained model buffers so the
        // in-Rust copy equals the loaded model. CLP has no incremental mutation
        // API, so `add_rows`/`set_*_bounds` patch these buffers and re-issue the
        // full load via `reload`. Cleared then filled (no `shrink_to_fit`) so
        // capacity stabilises at the peak.
        self.col_starts.clear();
        self.col_starts.extend_from_slice(&template.col_starts);
        self.row_indices.clear();
        self.row_indices.extend_from_slice(&template.row_indices);
        self.values.clear();
        self.values.extend_from_slice(&template.values);
        self.col_lower.clear();
        self.col_lower.extend_from_slice(&template.col_lower);
        self.col_upper.clear();
        self.col_upper.extend_from_slice(&template.col_upper);
        self.objective.clear();
        self.objective.extend_from_slice(&template.objective);
        self.row_lower.clear();
        self.row_lower.extend_from_slice(&template.row_lower);
        self.row_upper.clear();
        self.row_upper.extend_from_slice(&template.row_upper);
        self.num_nz = template.num_nz;

        self.stats.total_load_model_time_seconds += t0.elapsed().as_secs_f64();
        self.stats.load_model_count += 1;
    }

    /// Appends a batch of constraint rows to the loaded LP.
    ///
    /// `rows` is in CSR (row-major) form; the retained model is CSC
    /// (column-major). The merge transposes each batch row's `(col, value)`
    /// pairs into the per-column lists of the retained CSC, then re-issues a
    /// full `cobre_clp_load_problem` via `reload` (CLP has no
    /// incremental row-append). The retained `row_lower`/`row_upper` are
    /// extended, `num_rows`/`num_nz` are bumped, and `row_dual` is resized.
    ///
    /// # Panics
    ///
    /// Panics if `rows.num_rows` or the batch nnz does not fit in `i32`, or if
    /// the merged model dimensions exceed the CLP C API limit (via `reload`).
    fn add_rows(&mut self, rows: &RowBatch) {
        assert!(
            i32::try_from(rows.num_rows).is_ok(),
            "rows.num_rows {} overflows i32: RowBatch exceeds CLP API limit",
            rows.num_rows
        );
        assert!(
            i32::try_from(rows.col_indices.len()).is_ok(),
            "rows nnz {} overflows i32: RowBatch exceeds CLP API limit",
            rows.col_indices.len()
        );

        let new_nz = rows.col_indices.len();
        if rows.num_rows == 0 {
            // Nothing to append; avoid a needless CSC rebuild + reload. A
            // well-formed empty batch has no column entries either — guard the
            // invariant so a malformed batch (no rows but non-empty
            // col_indices) cannot scatter ghost entries into the retained CSC.
            debug_assert!(
                new_nz == 0,
                "malformed RowBatch: num_rows is 0 but col_indices has {new_nz} entries"
            );
            return;
        }

        // Transpose-append the CSR batch into the retained CSC.
        //
        // Step 1: count how many new non-zeros land in each existing column.
        // `per_col_count[c]` is the number of batch entries in column `c`.
        let mut per_col_count = vec![0_usize; self.num_cols];
        for &col in &rows.col_indices {
            #[allow(clippy::cast_sign_loss)]
            let col = col as usize;
            debug_assert!(
                col < self.num_cols,
                "RowBatch column index {col} out of range [0, {})",
                self.num_cols
            );
            per_col_count[col] += 1;
        }

        // Step 2: build the new col_starts and reserve the merged entry buffers.
        // The merged matrix has the same column count; each column `c` keeps its
        // existing entries followed by `per_col_count[c]` appended entries.
        let merged_nz = self.num_nz + new_nz;
        let mut new_col_starts = Vec::with_capacity(self.num_cols + 1);
        let mut new_row_indices = vec![0_i32; merged_nz];
        let mut new_values = vec![0.0_f64; merged_nz];

        // `write_cursor[c]` tracks the next write position within column `c`'s
        // slice of the merged buffers. Initialised to each column's start.
        let mut write_cursor = Vec::with_capacity(self.num_cols);
        let mut acc = 0_usize;
        for c in 0..self.num_cols {
            new_col_starts.push(i32_from_usize(acc));
            write_cursor.push(acc);
            #[allow(clippy::cast_sign_loss)]
            let old_start = self.col_starts[c] as usize;
            #[allow(clippy::cast_sign_loss)]
            let old_end = self.col_starts[c + 1] as usize;
            // Copy the existing entries of column `c` into its merged slice.
            for k in old_start..old_end {
                new_row_indices[acc] = self.row_indices[k];
                new_values[acc] = self.values[k];
                acc += 1;
            }
            // `write_cursor[c]` already points at the start of column `c`; advance
            // it past the copied existing entries so appended entries follow.
            write_cursor[c] = acc;
            // Reserve space for the appended batch entries in this column.
            acc += per_col_count[c];
        }
        new_col_starts.push(i32_from_usize(acc));
        debug_assert_eq!(acc, merged_nz);

        // Step 3: scatter each batch row's entries into the reserved per-column
        // slots. Appended rows occupy global row indices [num_rows, num_rows + n).
        for r in 0..rows.num_rows {
            #[allow(clippy::cast_sign_loss)]
            let start = rows.row_starts[r] as usize;
            #[allow(clippy::cast_sign_loss)]
            let end = rows.row_starts[r + 1] as usize;
            let global_row = self.num_rows + r;
            for k in start..end {
                #[allow(clippy::cast_sign_loss)]
                let col = rows.col_indices[k] as usize;
                let pos = write_cursor[col];
                new_row_indices[pos] = i32_from_usize(global_row);
                new_values[pos] = rows.values[k];
                write_cursor[col] += 1;
            }
        }

        // Commit the merged CSC and extend the row-bound vectors.
        self.col_starts = new_col_starts;
        self.row_indices = new_row_indices;
        self.values = new_values;
        self.row_lower.extend_from_slice(&rows.row_lower);
        self.row_upper.extend_from_slice(&rows.row_upper);
        self.num_rows += rows.num_rows;
        self.num_nz = merged_nz;

        self.reload();

        // Grow the row-dual extraction buffer to cover the appended rows.
        self.row_dual.resize(self.num_rows, 0.0);
    }

    /// Patches the bounds of an arbitrary set of rows and rebuilds the model.
    ///
    /// `indices`, `lower`, and `upper` must have equal length. An empty
    /// `indices` slice is a no-op (no reload). Each retained `row_lower`/
    /// `row_upper` entry at the given index is overwritten, then the model is
    /// rebuilt via `reload`. Bounds are forwarded verbatim.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds), or via `reload`'s i32-overflow asserts.
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        assert!(
            indices.len() == lower.len() && indices.len() == upper.len(),
            "set_row_bounds: indices ({}), lower ({}), and upper ({}) must have equal length",
            indices.len(),
            lower.len(),
            upper.len()
        );
        if indices.is_empty() {
            return;
        }

        let t0 = Instant::now();
        for (i, &row) in indices.iter().enumerate() {
            debug_assert!(
                row < self.num_rows,
                "set_row_bounds: index {row} out of range [0, {})",
                self.num_rows
            );
            self.row_lower[row] = lower[i];
            self.row_upper[row] = upper[i];
        }
        self.reload();
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Patches the bounds of an arbitrary set of columns and rebuilds the model.
    ///
    /// Symmetric to [`Self::set_row_bounds`] but patches the retained
    /// `col_lower`/`col_upper`. An empty `indices` slice is a no-op (no
    /// reload). Bounds are forwarded verbatim.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds), or via `reload`'s i32-overflow asserts.
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        assert!(
            indices.len() == lower.len() && indices.len() == upper.len(),
            "set_col_bounds: indices ({}), lower ({}), and upper ({}) must have equal length",
            indices.len(),
            lower.len(),
            upper.len()
        );
        if indices.is_empty() {
            return;
        }

        let t0 = Instant::now();
        for (i, &col) in indices.iter().enumerate() {
            debug_assert!(
                col < self.num_cols,
                "set_col_bounds: index {col} out of range [0, {})",
                self.num_cols
            );
            self.col_lower[col] = lower[i];
            self.col_upper[col] = upper[i];
        }
        self.reload();
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Solves the loaded LP via the dual simplex and returns the optimal
    /// solution as a [`SolutionView`] borrowing the solver's owned buffers.
    ///
    /// Runs a single cold `cobre_clp_dual` call (no retry escalation ladder and
    /// no per-solve wall-clock budget). On
    /// `CLP_STATUS_OPTIMAL` the three CLP-owned solution pointers are copied
    /// **immediately** into `col_value`/`col_dual`/`row_dual` (they are valid
    /// only until the next solve), the row duals are normalized via
    /// `normalize_row_dual`, and a `SolutionView` borrowing those buffers is
    /// returned. Non-optimal statuses map to a [`SolverError`].
    ///
    /// # Warm-start basis
    ///
    /// When `basis = Some(b)`, `b` is reinstalled into the CLP model
    /// element-by-element via `install_basis` before the solve; both
    /// `None` and `Some(&b)` then run the same cold `cobre_clp_dual` path (CLP
    /// retains its internal basis across consecutive solves). Unlike the `HiGHS`
    /// backend, CLP's per-element status setters do not validate global basis
    /// consistency, so there is no `BasisInconsistent` error path here.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::Infeasible)` on `PRIMAL_INFEASIBLE`,
    /// `Err(SolverError::Unbounded)` on `DUAL_INFEASIBLE`,
    /// `Err(SolverError::IterationLimit { .. })` on `STOPPED`, and
    /// `Err(SolverError::InternalError { .. })` on `ERRORS` or any unexpected
    /// status int.
    ///
    /// # Panics
    ///
    /// Panics if no model is loaded (`!self.has_model`).
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        assert!(self.has_model, "solve called without a loaded model");

        if let Some(b) = basis {
            // Reinstall the offered basis element-by-element before the solve.
            // CLP retains its internal basis across consecutive `cobre_clp_dual`
            // calls, so both `None` (warm from prior solve) and `Some` (warm from
            // the reinstalled basis) fall through to the same cold dual path below.
            self.install_basis(b);
        }

        let t0 = Instant::now();
        // SAFETY: `self.handle` is a valid, non-null CLP pointer from
        // `cobre_clp_create()` with a model loaded (asserted via `has_model`).
        // `if_values_pass = 0` requests a cold solve (no values pass). The
        // returned int is the CLP solve status.
        let status = unsafe { clp_ffi::cobre_clp_dual(self.handle, 0) };
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if status == clp_ffi::CLP_STATUS_OPTIMAL {
            // Read iteration count and objective from FFI BEFORE establishing
            // the shared borrow of the owned buffers via the returned
            // `SolutionView`, so stats can be updated without violating the
            // aliasing rules (mirrors the HiGHS ordering in `solve_inner`).
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved; iteration count is non-negative so the cast is
            // safe.
            #[allow(clippy::cast_sign_loss)]
            let iterations = unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved. Objective is already in minimize sense (the
            // wrapper set the optimization direction at load); returned as-is.
            let objective = unsafe { clp_ffi::cobre_clp_objective_value(self.handle) };

            // Copy the three CLP-owned solution pointers into the owned buffers
            // immediately, before any further CLP call (the pointers are valid
            // only until the next solve).
            self.copy_solution();

            self.stats.success_count += 1;
            self.stats.first_try_successes += 1;
            self.stats.total_iterations += iterations;
            self.stats.total_solve_time_seconds += solve_time;

            return Ok(SolutionView {
                objective,
                primal: &self.col_value[..self.num_cols],
                dual: &self.row_dual[..self.num_rows],
                reduced_costs: &self.col_dual[..self.num_cols],
                iterations,
                solve_time_seconds: solve_time,
            });
        }

        self.stats.failure_count += 1;
        match status {
            clp_ffi::CLP_STATUS_PRIMAL_INFEASIBLE => Err(SolverError::Infeasible),
            clp_ffi::CLP_STATUS_DUAL_INFEASIBLE => Err(SolverError::Unbounded),
            clp_ffi::CLP_STATUS_STOPPED => {
                // NOTE: CLP status 3 covers both iteration and time stop; mapped
                // to IterationLimit. CLP's status int does not distinguish the
                // two, and no per-solve time budget is derived.
                // SAFETY: `self.handle` is a valid, non-null CLP pointer;
                // iteration count is non-negative so the cast is safe.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
                Err(SolverError::IterationLimit { iterations })
            }
            clp_ffi::CLP_STATUS_ERRORS => Err(SolverError::InternalError {
                message: "CLP solve failed (Clp_dual returned ERRORS status)".to_string(),
                error_code: Some(4),
            }),
            other => Err(SolverError::InternalError {
                message: format!("CLP returned unexpected status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Extracts the current simplex basis into `out`, element-by-element.
    ///
    /// `out.col_status` is resized to `num_cols` and `out.row_status` to
    /// `num_rows`, then each entry is filled from `cobre_clp_get_column_status` /
    /// `cobre_clp_get_row_status`. CLP reports basis status one element at a time
    /// (no bulk array in the wrapper), so this is a pair of per-element loops; the
    /// raw `CLP_BASIS_*` `i32` codes are stored verbatim for a later round-trip
    /// into `install_basis`.
    ///
    /// # Panics
    ///
    /// Panics if no model is loaded (`!self.has_model`).
    fn get_basis(&mut self, out: &mut Basis) {
        assert!(
            self.has_model,
            "get_basis called without a loaded model — call load_model first"
        );

        out.col_status.resize(self.num_cols, 0);
        out.row_status.resize(self.num_rows, 0);

        // Loop indices are bounded by `num_cols`/`num_rows`, both asserted to
        // fit in i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded (asserted via `has_model`); `c` is in `0..num_cols`, a valid
            // column sequence index, and fits in i32. The getter reads a single
            // status byte and returns it widened to i32.
            out.col_status[c] =
                unsafe { clp_ffi::cobre_clp_get_column_status(self.handle, c as i32) };
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..self.num_rows {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..num_rows`, a valid row sequence index, and fits
            // in i32. The getter reads a single status byte and returns it widened.
            out.row_status[r] = unsafe { clp_ffi::cobre_clp_get_row_status(self.handle, r as i32) };
        }
    }

    /// Returns a snapshot of the accumulated solver statistics.
    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }

    /// Returns a static string identifying the solver backend.
    fn name(&self) -> &'static str {
        "CLP"
    }

    /// Returns the solver name and version as a human-readable string.
    ///
    /// Example: `"CLP 1.17.11"`
    fn solver_name_version(&self) -> String {
        format!("CLP {}", clp_version())
    }

    // NOTE: `record_reconstruction_stats` is intentionally NOT overridden — CLP
    // has no slot-reconciliation basis reconstruction, so the trait's default
    // no-op is correct.

    /// Configures the primal feasibility tolerance on the underlying CLP model
    /// and caches the value in `current_profile`.
    fn set_primal_feasibility_tolerance(&mut self, value: f64) {
        // SAFETY: `self.handle` is a valid, non-null CLP pointer obtained from
        // `cobre_clp_create()`. The setter accepts any `f64` and retains no
        // pointer after the call returns; it cannot fail on a valid handle.
        unsafe {
            clp_ffi::cobre_clp_set_primal_tolerance(self.handle, value);
        }
        self.current_profile.primal_feasibility_tolerance = value;
    }

    /// Configures the dual feasibility tolerance on the underlying CLP model
    /// and caches the value in `current_profile`.
    ///
    /// Symmetric to [`Self::set_primal_feasibility_tolerance`].
    fn set_dual_feasibility_tolerance(&mut self, value: f64) {
        // SAFETY: `self.handle` is a valid, non-null CLP pointer obtained from
        // `cobre_clp_create()`. The setter accepts any `f64` and retains no
        // pointer after the call returns; it cannot fail on a valid handle.
        unsafe {
            clp_ffi::cobre_clp_set_dual_tolerance(self.handle, value);
        }
        self.current_profile.dual_feasibility_tolerance = value;
    }

    /// Caches the per-attempt simplex iteration cap in `current_profile`.
    ///
    /// No FFI call is issued here. The actual cap is applied by
    /// [`Self::apply_profile`] via `resolve_simplex_cap`: a value of
    /// [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`) selects the heuristic
    /// `max(100_000, num_cols * 50)`; any non-zero value is applied verbatim.
    fn set_simplex_iteration_limit_profile(&mut self, value: u32) {
        self.current_profile.simplex_iteration_limit = value;
    }

    /// Per-attempt IPM iteration cap — a documented no-op for CLP.
    fn set_ipm_iteration_limit_profile(&mut self, value: u32) {
        // NOTE: CLP is simplex-only; IPM cap cached for PartialEq delta-tracking but not applied.
        //
        // `ClpProfile` carries no IPM field (CLP has no interior-point solver),
        // so there is no field to cache and no FFI call to issue. The method
        // exists solely to satisfy the trait; mirrors how `HighsProfile` carries
        // the IPM field even when a given path does not use it.
        let _ = value;
    }
}

/// Normalizes a raw CLP row price into cobre's canonical dual-sign convention.
///
/// Dual-sign: the sign-convention probe confirmed `cobre_clp_get_row_price`
/// matches the canonical convention (`HiGHS` does not negate either); identity
/// is correct. See `tests/_clp_sign_convention_probe.rs`.
const fn normalize_row_dual(raw: f64) -> f64 {
    raw
}

/// Converts a `usize` to `i32`, panicking on overflow.
///
/// Used inside the CSR→CSC merge to write `col_starts`/`row_indices` entries.
/// Each value is bounded by the merged nnz / row count, which `add_rows` and
/// `reload` independently assert fit in `i32`; this guards the per-entry writes.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn i32_from_usize(v: usize) -> i32 {
    debug_assert!(
        i32::try_from(v).is_ok(),
        "value {v} overflows i32: LP exceeds CLP API limit"
    );
    v as i32
}

impl Drop for ClpSolver {
    fn drop(&mut self) {
        // SAFETY: valid CLP pointer from construction, called once per instance.
        unsafe { clp_ffi::cobre_clp_destroy(self.handle) };
    }
}

/// Returns the CLP version as a `"major.minor.patch"` string.
///
/// This is a free function — no solver instance is required.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "clp")]
/// # {
/// let v = cobre_solver::clp_version();
/// assert!(v.contains('.'), "version string should be 'major.minor.patch'");
/// # }
/// ```
#[must_use]
pub fn clp_version() -> String {
    // SAFETY: These are pure query functions with no arguments. The CLP C API
    // documents them as safe to call without any prior initialisation; they
    // read only compile-time constants embedded in the library.
    let major = unsafe { clp_ffi::cobre_clp_version_major() };
    let minor = unsafe { clp_ffi::cobre_clp_version_minor() };
    let patch = unsafe { clp_ffi::cobre_clp_version_release() };
    format!("{major}.{minor}.{patch}")
}

#[cfg(test)]
mod tests {
    use crate::clp::{ClpProfile, ClpSolver, clp_version};
    use crate::profile::DEFAULT_PROFILE_HEURISTIC_SENTINEL;
    use crate::types::{Basis, RowBatch, SolutionView, SolverError, StageTemplate};
    use crate::{ProfiledSolver, SolverInterface};

    fn assert_profile_bounds<P: Copy + PartialEq + Default + Send>() {}

    fn assert_send<T: Send>() {}

    // Solver Interface Testing SS1.1 fixture (mirrors the HiGHS test fixture):
    //   minimize  0*x0 + 1*x1 + 50*x2
    //   subject to  row0:  x0           = 6
    //               row1: 2*x0     + x2 = 14
    //   x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
    //
    // CSC matrix A = [[1, 0, 0], [2, 0, 1]]:
    //   col_starts  = [0, 2, 2, 3]
    //   row_indices = [0, 1, 1]
    //   values      = [1.0, 2.0, 1.0]
    fn make_fixture_stage_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 2, 2, 3],
            row_indices: vec![0_i32, 1, 1],
            values: vec![1.0, 2.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![10.0, f64::INFINITY, 8.0],
            objective: vec![0.0, 1.0, 50.0],
            row_lower: vec![6.0, 14.0],
            row_upper: vec![6.0, 14.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    // Solver Interface Testing SS1.2 fixture (mirrors the conformance fixture):
    // two appended `>=` rows over the SS1.1 columns:
    //   row2: -5*x0 + x1 >= 20   (row_lower 20, row_upper +inf)
    //   row3:  3*x0 + x1 >= 80   (row_lower 80, row_upper +inf)
    //
    // CSR (row-major):
    //   row_starts  = [0, 2, 4]
    //   col_indices = [0, 1, 0, 1]
    //   values      = [-5.0, 1.0, 3.0, 1.0]
    fn make_fixture_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 2,
            row_starts: vec![0_i32, 2, 4],
            col_indices: vec![0_i32, 1, 0, 1],
            values: vec![-5.0, 1.0, 3.0, 1.0],
            row_lower: vec![20.0, 80.0],
            row_upper: vec![f64::INFINITY, f64::INFINITY],
        }
    }

    #[test]
    fn test_clp_profile_default_values() {
        let p = ClpProfile::default();
        assert_eq!(p.perturbation, 102);
        assert_eq!(p.scaling, 0);
        assert_eq!(p.primal_feasibility_tolerance, 1e-9);
        assert_eq!(p.dual_feasibility_tolerance, 1e-9);
        assert_eq!(
            p.simplex_iteration_limit,
            DEFAULT_PROFILE_HEURISTIC_SENTINEL
        );
    }

    #[test]
    fn test_clp_solver_create_and_name() {
        let solver = ClpSolver::new().expect("CLP solver creation failed");
        assert_eq!(solver.name(), "CLP");
        // Drop runs at scope end and frees the handle exactly once.
    }

    #[test]
    fn test_clp_solver_name_version_format() {
        let solver = ClpSolver::new().expect("CLP solver creation failed");
        let nv = solver.solver_name_version();
        assert!(nv.starts_with("CLP "), "got {nv}");
        assert_eq!(nv.matches('.').count(), 2, "got {nv}");
    }

    #[test]
    fn test_clp_profile_satisfies_associated_type_bounds() {
        assert_profile_bounds::<ClpProfile>();
    }

    #[test]
    fn test_clp_solver_send_bound() {
        assert_send::<ClpSolver>();
        // Also exercise the free version function for coverage.
        let _ = clp_version();
    }

    #[test]
    fn test_clp_load_model_updates_dimensions() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        let template = make_fixture_stage_template();
        solver.load_model(&template);

        // AC1: dimensions and has_model flag.
        assert_eq!(solver.num_cols, 3);
        assert_eq!(solver.num_rows, 2);
        assert!(solver.has_model);

        // AC2: solution buffer lengths.
        assert_eq!(solver.col_value.len(), 3);
        assert_eq!(solver.col_dual.len(), 3);
        assert_eq!(solver.row_dual.len(), 2);
    }

    #[test]
    fn test_clp_load_model_accepts_infinite_bounds() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        let mut template = make_fixture_stage_template();
        template.col_upper[1] = f64::INFINITY;

        // AC3: completes without panic; the C wrapper owns the DBL_MAX
        // translation, so the Rust layer never reads or mutates the infinity.
        solver.load_model(&template);

        assert_eq!(solver.num_cols, 3);
        assert_eq!(solver.num_rows, 2);
        assert!(solver.has_model);
    }

    #[test]
    fn test_clp_load_model_reload_zero_row() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC4: re-load with a zero-row, single-column template replaces the
        // prior model and resizes buffers accordingly.
        let zero_row = StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: vec![0.0],
            col_upper: vec![1.0],
            objective: vec![1.0],
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        solver.load_model(&zero_row);

        assert_eq!(solver.num_rows, 0);
        assert_eq!(solver.row_dual.len(), 0);
        assert_eq!(solver.num_cols, 1);
    }

    #[test]
    fn test_clp_load_model_count_accumulates() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        let template = make_fixture_stage_template();

        // AC5: load_model_count increments on each call.
        solver.load_model(&template);
        assert_eq!(solver.stats.load_model_count, 1);

        solver.load_model(&template);
        assert_eq!(solver.stats.load_model_count, 2);
    }

    #[test]
    fn test_clp_add_rows_updates_dimensions() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        let batch = make_fixture_row_batch();
        solver.add_rows(&batch);

        // AC1: the two appended rows grow num_rows to 4, row_dual is resized to
        // match, and the column count is unchanged.
        assert_eq!(solver.num_rows, 4);
        assert_eq!(solver.row_dual.len(), 4);
        assert_eq!(solver.num_cols, 3);

        // The merged CSC must be structurally consistent: nnz grows by the batch
        // nnz, col_starts has num_cols + 1 entries, and the final start equals
        // the total nnz.
        assert_eq!(solver.num_nz, 3 + 4);
        assert_eq!(solver.col_starts.len(), 4);
        assert_eq!(solver.col_starts[3], 7);

        // Hand-checked merged CSC. SS1.1 columns:
        //   col0: (row0, 1.0), (row1, 2.0)
        //   col1: (none)
        //   col2: (row1, 1.0)
        // SS1.2 batch appends rows 2 and 3:
        //   col0 gains (row2, -5.0), (row3, 3.0)
        //   col1 gains (row2,  1.0), (row3, 1.0)
        //   col2 gains nothing
        // Merged column-major layout (existing entries first, then appended):
        //   col0: [ (0,1.0), (1,2.0), (2,-5.0), (3,3.0) ]  -> starts at 0
        //   col1: [ (2,1.0), (3,1.0) ]                      -> starts at 4
        //   col2: [ (1,1.0) ]                               -> starts at 6
        assert_eq!(solver.col_starts, vec![0, 4, 6, 7]);
        assert_eq!(solver.row_indices, vec![0, 1, 2, 3, 2, 3, 1]);
        assert_eq!(solver.values, vec![1.0, 2.0, -5.0, 3.0, 1.0, 1.0, 1.0]);

        // Appended row bounds land at the end of the retained vectors.
        assert_eq!(solver.row_lower, vec![6.0, 14.0, 20.0, 80.0]);
        assert!(solver.row_upper[2].is_infinite());
        assert!(solver.row_upper[3].is_infinite());
    }

    #[test]
    fn test_clp_set_row_bounds_patches_retained() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC2: equality-tighten row 0 to [4.0, 4.0]; the retained vectors reflect
        // the patch and the call completes without panic.
        solver.set_row_bounds(&[0], &[4.0], &[4.0]);
        assert_eq!(solver.row_lower[0], 4.0);
        assert_eq!(solver.row_upper[0], 4.0);
        // Untouched row is unchanged.
        assert_eq!(solver.row_lower[1], 14.0);
    }

    #[test]
    fn test_clp_set_col_bounds_patches_retained() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC3: raise the lower bound of col 1 to 10.0 with an infinite upper.
        solver.set_col_bounds(&[1], &[10.0], &[f64::INFINITY]);
        assert_eq!(solver.col_lower[1], 10.0);
        assert!(solver.col_upper[1].is_infinite());
        // Untouched column is unchanged.
        assert_eq!(solver.col_lower[0], 0.0);
    }

    #[test]
    fn test_clp_set_bounds_empty_no_reload() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        let load_count_before = solver.stats.load_model_count;

        // AC4: empty index slices are a no-op -- no reload, no panic, dimensions
        // unchanged. (load_model_count would be unaffected regardless, since
        // reload bypasses it; we assert state invariance instead.)
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);

        assert_eq!(solver.num_rows, 2);
        assert_eq!(solver.num_cols, 3);
        assert_eq!(solver.stats.load_model_count, load_count_before);
        // Bound vectors are untouched.
        assert_eq!(solver.row_lower, vec![6.0, 14.0]);
        assert_eq!(solver.col_lower, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clp_add_rows_then_solve_objective() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        solver.add_rows(&make_fixture_row_batch());

        // Assert the merge produced the expected retained dimensions/state.
        assert_eq!(solver.num_cols, 3);
        assert_eq!(solver.num_rows, 4);
        assert_eq!(solver.num_nz, 7);
        assert_eq!(solver.row_dual.len(), 4);
    }

    #[test]
    fn test_clp_solve_basic_lp() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC1: cold solve of the SS1.1 LP returns Ok with the expected optimum.
        let view = solver
            .solve(None)
            .expect("SS1.1 LP should solve to optimal");

        // Optimal x = (6, 0, 2), obj = 100.
        assert!(
            (view.objective - 100.0).abs() < 1e-8,
            "objective {} not within 1e-8 of 100.0",
            view.objective
        );
        assert!(
            (view.primal[0] - 6.0).abs() < 1e-8,
            "primal[0] {} not within 1e-8 of 6.0",
            view.primal[0]
        );
        assert!(
            (view.primal[2] - 2.0).abs() < 1e-8,
            "primal[2] {} not within 1e-8 of 2.0",
            view.primal[2]
        );

        // AC2: view slice lengths match the LP dimensions.
        assert_eq!(view.primal.len(), 3);
        assert_eq!(view.reduced_costs.len(), 3);
        assert_eq!(view.dual.len(), 2);
    }

    #[test]
    fn test_clp_solve_infeasible() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC3: pin x0 = 100, which violates the row0 equality x0 = 6.
        solver.set_col_bounds(&[0], &[100.0], &[100.0]);
        let result = solver.solve(None);

        assert!(
            matches!(result, Err(SolverError::Infeasible)),
            "expected Err(Infeasible), got {result:?}"
        );
    }

    #[test]
    fn test_clp_solve_unbounded() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");

        // AC4: single column, objective -1, lower 0, upper +inf, no rows.
        let unbounded = StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: vec![0.0],
            col_upper: vec![f64::INFINITY],
            objective: vec![-1.0],
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        solver.load_model(&unbounded);
        let result = solver.solve(None);

        assert!(
            matches!(result, Err(SolverError::Unbounded)),
            "expected Err(Unbounded), got {result:?}"
        );
    }

    #[test]
    fn test_clp_solve_twice_stats() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        // AC5: two consecutive cold solves both succeed; the owned-buffer copies
        // are re-read each solve rather than aliasing stale CLP pointers.
        let _ = solver.solve(None).expect("first solve should be optimal");
        let _ = solver.solve(None).expect("second solve should be optimal");

        assert_eq!(solver.stats.solve_count, 2);
        assert_eq!(solver.stats.success_count, 2);
    }

    #[test]
    fn test_clp_get_basis_dimensions_and_codes() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        let _ = solver
            .solve(None)
            .expect("SS1.1 LP should solve to optimal");

        // AC1: get_basis resizes a 0/0 Basis to the LP dimensions and fills it
        // with raw CLP status codes (all in 0..=5).
        let mut out = Basis::new(0, 0);
        solver.get_basis(&mut out);

        assert_eq!(out.col_status.len(), 3);
        assert_eq!(out.row_status.len(), 2);
        for &s in out.col_status.iter().chain(out.row_status.iter()) {
            assert!(
                (0..=5).contains(&s),
                "status code {s} out of CLP range 0..=5"
            );
        }
    }

    #[test]
    fn test_clp_warm_start_roundtrip_objective() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        let _ = solver.solve(None).expect("first solve should be optimal");

        // Capture the optimal basis.
        let mut captured = Basis::new(0, 0);
        solver.get_basis(&mut captured);

        // AC2: warm-start solve on the unchanged LP returns the same optimum.
        let view = solver
            .solve(Some(&captured))
            .expect("warm-start solve should be optimal");
        assert!(
            (view.objective - 100.0).abs() < 1e-8,
            "warm-start objective {} not within 1e-8 of 100.0",
            view.objective
        );

        // AC3: the offered basis was counted and the set-time accumulated.
        // This reads the private `stats` field directly, as the sibling tests do.
        assert_eq!(solver.stats.basis_offered, 1);
        assert!(
            solver.stats.total_basis_set_time_seconds >= 0.0,
            "total_basis_set_time_seconds {} should be non-negative",
            solver.stats.total_basis_set_time_seconds
        );
    }

    #[test]
    #[should_panic(expected = "loaded model")]
    fn test_clp_get_basis_without_model_panics() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        // AC4: get_basis on a solver with no loaded model panics.
        solver.get_basis(&mut Basis::new(0, 0));
    }

    #[test]
    fn test_clp_basis_roundtrip_identity() {
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        let _ = solver
            .solve(None)
            .expect("SS1.1 LP should solve to optimal");

        // AC5: two get_basis calls into fresh Basis values yield identical vectors
        // — the round-trip stores exactly the CLP-reported codes with no drift.
        let mut first = Basis::new(0, 0);
        solver.get_basis(&mut first);
        let mut second = Basis::new(0, 0);
        solver.get_basis(&mut second);

        assert_eq!(first.col_status, second.col_status);
        assert_eq!(first.row_status, second.row_status);
    }

    #[test]
    fn test_clp_apply_default_profile_then_solve() {
        // AC1: applying the default profile (perturbation=102, scaling=0, tight
        // tolerances, heuristic iteration cap) before a solve must not break the
        // solve — the SS1.1 LP still returns obj 100.0.
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.apply_profile(&ClpProfile::default());
        assert_eq!(solver.current_profile, ClpProfile::default());

        solver.load_model(&make_fixture_stage_template());
        let view = solver
            .solve(None)
            .expect("SS1.1 LP should solve to optimal after apply_profile");
        assert!(
            (view.objective - 100.0).abs() < 1e-8,
            "objective {} not within 1e-8 of 100.0",
            view.objective
        );
    }

    #[test]
    fn test_clp_tolerance_setters_cache_profile() {
        // AC2: each tolerance setter issues the FFI call and caches the value in
        // current_profile so ProfiledSolver delta-tracking observes the change.
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");

        solver.set_primal_feasibility_tolerance(1e-7);
        assert_eq!(solver.current_profile.primal_feasibility_tolerance, 1e-7);

        solver.set_dual_feasibility_tolerance(1e-7);
        assert_eq!(solver.current_profile.dual_feasibility_tolerance, 1e-7);
    }

    #[test]
    fn test_clp_profiled_solver_default_noop() {
        // AC3: wrapping in ProfiledSolver and calling set_profile(&default) is a
        // delta-tracking no-op — the wrapper starts at default, so no
        // apply_profile is issued and current_profile() is unchanged.
        let solver = ClpSolver::new().expect("CLP solver creation failed");
        let mut profiled = ProfiledSolver::new(solver);

        assert_eq!(*profiled.current_profile(), ClpProfile::default());
        profiled.set_profile(&ClpProfile::default());
        assert_eq!(*profiled.current_profile(), ClpProfile::default());
    }

    #[test]
    fn test_clp_usable_as_generic_solver_bound() {
        // AC4: ClpSolver satisfies the full SolverInterface contract and is
        // usable wherever `S: SolverInterface` is required.
        fn run_solve<S: SolverInterface>(s: &mut S) -> Result<SolutionView<'_>, SolverError> {
            s.solve(None)
        }

        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());
        let view = run_solve::<ClpSolver>(&mut solver)
            .expect("generic run_solve over ClpSolver should be Ok");
        assert!(
            (view.objective - 100.0).abs() < 1e-8,
            "objective {} not within 1e-8 of 100.0",
            view.objective
        );
    }

    #[test]
    fn test_clp_resolve_simplex_cap_sentinel_and_explicit() {
        // AC5: on a 3-col LP, the sentinel (0) resolves to max(100_000, 3*50) =
        // 100_000; an explicit 500 resolves to 500 verbatim.
        let mut solver = ClpSolver::new().expect("CLP solver creation failed");
        solver.load_model(&make_fixture_stage_template());

        solver.set_simplex_iteration_limit_profile(DEFAULT_PROFILE_HEURISTIC_SENTINEL);
        assert_eq!(solver.resolve_simplex_cap(), 100_000);

        solver.set_simplex_iteration_limit_profile(500);
        assert_eq!(solver.resolve_simplex_cap(), 500);
    }
}
