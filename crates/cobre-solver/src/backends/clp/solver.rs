//! The [`ClpSolver`] handle wrapper, its lifecycle/hot-start primitives, and the
//! [`clp_version`] free function.

use std::os::raw::c_void;
use std::time::Instant;

use super::config::ClpProfile;
use crate::{
    DEFAULT_PROFILE_HEURISTIC_SENTINEL, clp_ffi,
    types::{SolutionView, SolverError, SolverStatistics},
};

/// CLP basis status code for a column resting at its lower bound (nonbasic).
///
/// Status codes follow `ClpSimplex.hpp`: `0 = free, 1 = basic, 2 = atUpper,
/// 3 = atLower, 4 = superbasic, 5 = fixed`.
const CLP_BASIS_AT_LOWER: i32 = 3;

/// CLP basis status code for a basic variable (see [`CLP_BASIS_AT_LOWER`] for the
/// full enum).
const CLP_BASIS_BASIC: i32 = 1;

/// CLP LP solver backend.
///
/// Owns an opaque CLP model handle plus pre-allocated, reusable buffers resized
/// by `load_model` and reused across solves to avoid per-solve allocation. The
/// retained CSC arrays and bound vectors are the canonical, declaration-ordered
/// mirror of the loaded LP: `add_rows`/`set_*_bounds` patch them and reconcile
/// the change into CLP natively (`cobre_clp_add_rows` / `cobre_clp_chg_*`)
/// without rebuilding the model, preserving CLP's factorization/basis. Bound
/// vectors are forwarded to CLP verbatim.
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
    pub(super) handle: *mut c_void,
    /// Primal column values extracted after each solve.
    pub(super) col_value: Vec<f64>,
    /// Column dual values (reduced costs).
    pub(super) col_dual: Vec<f64>,
    /// Row dual multipliers (shadow prices).
    pub(super) row_dual: Vec<f64>,
    /// Retained CSC column-start offsets (length `num_cols + 1`).
    pub(super) col_starts: Vec<i32>,
    /// Retained CSC row indices for each non-zero (length `num_nz`).
    pub(super) row_indices: Vec<i32>,
    /// Retained CSC non-zero values (length `num_nz`).
    pub(super) values: Vec<f64>,
    /// Retained column lower bounds (length `num_cols`).
    pub(super) col_lower: Vec<f64>,
    /// Retained column upper bounds (length `num_cols`).
    pub(super) col_upper: Vec<f64>,
    /// Retained row lower bounds (length `num_rows`).
    pub(super) row_lower: Vec<f64>,
    /// Retained row upper bounds (length `num_rows`).
    pub(super) row_upper: Vec<f64>,
    /// Retained non-zero count, kept in sync with `values.len()`.
    pub(super) num_nz: usize,
    /// Current number of LP columns (decision variables).
    pub(super) num_cols: usize,
    /// Current number of LP rows (constraints).
    pub(super) num_rows: usize,
    /// Whether a model is currently loaded; guards the `solve`/`get_basis` contract.
    pub(super) has_model: bool,
    /// Accumulated solver statistics. Counters grow monotonically from zero.
    pub(super) stats: SolverStatistics,
    /// Cached solver profile applied by the last profile-setter call.
    pub(super) current_profile: ClpProfile,
    /// Opaque CLP-owned hot-start snapshot token, or null when no snapshot is
    /// active. It is **never dereferenced** on the Rust side — only threaded back
    /// into the hot-start FFI on this same handle. `Drop` releases a still-held
    /// token so every `mark` is paired with exactly one `unmark`.
    pub(super) hot_start_token: *mut c_void,
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
    /// # Errors
    ///
    /// Returns `Err(SolverError::InternalError { .. })` if `cobre_clp_create()`
    /// returns a null pointer.
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

        // CLP ships with log level 1 (per-solve progress to stdout), which would
        // pollute CLI/Python output on every solve; force level 0.
        //
        // SAFETY: `handle` is the non-null model just returned by
        // `cobre_clp_create`; `cobre_clp_set_log_level` only forwards the level
        // to `Clp_setLogLevel` on that model and has no other preconditions.
        unsafe { clp_ffi::cobre_clp_set_log_level(handle, 0) };

        Ok(Self {
            handle,
            col_value: Vec::new(),
            col_dual: Vec::new(),
            row_dual: Vec::new(),
            col_starts: Vec::new(),
            row_indices: Vec::new(),
            values: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            num_nz: 0,
            num_cols: 0,
            num_rows: 0,
            has_model: false,
            stats: SolverStatistics::default(),
            current_profile: ClpProfile::default(),
            hot_start_token: std::ptr::null_mut(),
        })
    }

    /// Copies the three CLP-owned solution pointers into the owned buffers.
    ///
    /// Called immediately after an optimal solve and before any further CLP call:
    /// the CLP-owned pointers are valid only until the next solve. Row prices are
    /// normalized via `normalize_row_dual`.
    ///
    /// Each copy is guarded with `if len > 0` because passing a null or dangling
    /// pointer to `std::slice::from_raw_parts` is undefined behavior even with
    /// length 0 (a zero-column or zero-row LP may yield such a pointer from CLP).
    pub(super) fn copy_solution(&mut self) {
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
    /// CLP exposes basis status **per element**, not as a bulk array. Each
    /// canonical status in `b` is mapped via `to_clp_code` before the per-element
    /// write. An oversized row basis is tolerated (reinstalled up to
    /// `min(len, num_rows)`); an undersized one cannot be padded soundly and is
    /// rejected.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::BasisRowCountMismatch { lp_rows, basis_rows })`
    /// when `b.row_status.len() < self.num_rows`. On that path
    /// `basis_consistency_failures` is incremented and the basis is not offered
    /// to the solver.
    ///
    /// # Panics
    ///
    /// Panics if `b.col_status.len() != self.num_cols`; the LP column count is
    /// fixed at `load_model`, so a column mismatch is a genuine shape bug.
    pub(super) fn install_basis(&mut self, b: &crate::Basis) -> Result<(), SolverError> {
        // CLP's per-element setters silently accept an inconsistent offered basis
        // and `Clp_dual` repairs it, so — unlike `HighsSolver::solve` — there is no
        // consistency check and no `SolverError::BasisInconsistent` surface here.
        assert!(
            b.col_status.len() == self.num_cols,
            "basis column count {} does not match LP column count {}",
            b.col_status.len(),
            self.num_cols
        );
        // An undersized row basis cannot be padded soundly: the missing tail rows
        // would keep whatever (wrong) status CLP currently holds. Reject before
        // `basis_offered` is incremented — a rejected basis was never offered.
        if b.row_status.len() < self.num_rows {
            self.stats.basis_consistency_failures += 1;
            return Err(SolverError::BasisRowCountMismatch {
                lp_rows: self.num_rows,
                basis_rows: b.row_status.len(),
            });
        }

        self.stats.basis_offered += 1;

        let row_copy_len = b.row_status.len().min(self.num_rows);

        let basis_set_start = Instant::now();
        // Rationale: indices bounded by `num_cols`/`num_rows`, asserted to fit in
        // i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `c` is in `0..num_cols`, a valid column sequence index, and
            // fits in i32. The setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_column_status(
                    self.handle,
                    c as i32,
                    b.col_status[c].to_clp_code(),
                );
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..row_copy_len {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..min(b.row_status.len(), num_rows)`, a valid row
            // sequence index, and fits in i32. The setter writes a single status
            // byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_row_status(
                    self.handle,
                    r as i32,
                    b.row_status[r].to_clp_code(),
                );
            }
        }
        self.stats.total_basis_set_time_seconds += basis_set_start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Resets the CLP model to a clean all-slack (cold) starting basis.
    ///
    /// After a failed `Clp_dual` the model retains CLP's failed internal basis;
    /// re-solving on it can inherit the bad state, so each escalation rung first
    /// drives every structural column nonbasic ([`CLP_BASIS_AT_LOWER`]) and every
    /// row slack basic ([`CLP_BASIS_BASIC`]). Fully deterministic — the same
    /// status codes are written every time — so it cannot perturb bit-for-bit
    /// reproducibility.
    pub(super) fn reset_cold_basis(&mut self) {
        // Rationale: indices bounded by `num_cols`/`num_rows`, asserted to fit in
        // i32 by `load_model`; the casts cannot truncate or wrap.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for c in 0..self.num_cols {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded (asserted via `has_model` in `solve`); `c` is in
            // `0..num_cols`, a valid column sequence index, and fits in i32. The
            // setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_column_status(self.handle, c as i32, CLP_BASIS_AT_LOWER);
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        for r in 0..self.num_rows {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded; `r` is in `0..num_rows`, a valid row sequence index, and
            // fits in i32. The setter writes a single status byte; no aliasing.
            unsafe {
                clp_ffi::cobre_clp_set_row_status(self.handle, r as i32, CLP_BASIS_BASIC);
            }
        }
    }
    /// Resolves the per-attempt simplex iteration cap for `apply_profile`.
    ///
    /// At the [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`) applies the size-scaled
    /// heuristic `max(100_000, num_cols * 50)`; otherwise the profile value
    /// verbatim. Both branches clamp to `i32::MAX` for the FFI cast.
    pub(super) fn resolve_simplex_cap(&self) -> i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        if self.current_profile.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL {
            // Scale with LP size to avoid runaway cycling.
            let heuristic = self.num_cols.saturating_mul(50).max(100_000);
            (heuristic.min(i32::MAX as usize)) as i32
        } else {
            (self
                .current_profile
                .simplex_iteration_limit
                .min(i32::MAX as u32)) as i32
        }
    }

    /// Selects the dual-steepest-edge pricing rule on the underlying CLP model.
    ///
    /// `mode` 1 selects full DSE; 3 is the `ClpDualRowSteepest` default. Driven by
    /// [`Self::apply_profile`] only for a non-default mode (the `== 3` sentinel
    /// skips the call to keep the default profile byte-identical). The setter is
    /// idempotent and issues no solve.
    pub(super) fn set_dual_row_steepest(&mut self, mode: i32) {
        // SAFETY: `self.handle` is a valid, non-null CLP pointer from
        // `cobre_clp_create()`. The shim constructs a stack `ClpDualRowSteepest`
        // and installs it; it retains no pointer after the call returns and
        // cannot fail on a valid handle.
        unsafe {
            clp_ffi::cobre_clp_set_dual_row_steepest(self.handle, mode);
        }
    }

    /// Snapshots the simplex rim/factorization for hot-started re-solves.
    ///
    /// A model must be loaded; a prior `solve` is **not** required — CLP's
    /// `markHotStart` re-solves internally to establish the rim/factorization it
    /// snapshots. Calling while a snapshot is already active first releases the
    /// prior token, so at most one token is ever live. Perturbation and scaling
    /// stay off across the whole hot-start lifetime — the determinism
    /// preconditions are inherited from the applied profile, never re-enabled here.
    ///
    /// # Panics
    ///
    /// Panics only if no model is loaded (`!self.has_model`).
    pub fn mark_hot_start(&mut self) {
        assert!(
            self.has_model,
            "mark_hot_start called without a loaded model — call load_model first"
        );
        if !self.hot_start_token.is_null() {
            self.unmark_hot_start();
        }
        // SAFETY: `self.handle` is a valid, non-null CLP pointer from
        // `cobre_clp_create()` with a solved model loaded (asserted via
        // `has_model`). The shim reaches the live `ClpSimplex` through the
        // wrapper's `model_` member, calls `markHotStart`, and returns the
        // CLP-allocated `saveStuff` token. The token is CLP-owned; we retain it
        // opaquely and release it via `unmark_hot_start` (or `Drop`).
        let token = unsafe { clp_ffi::cobre_clp_mark_hot_start(self.handle) };
        debug_assert!(
            !token.is_null(),
            "markHotStart returned a null saveStuff token"
        );
        self.hot_start_token = token;
    }

    /// Re-solves the (bound-patched) model from the active hot-start snapshot.
    ///
    /// A snapshot must be active ([`Self::mark_hot_start`] called and not yet
    /// released). The intended composition is: solve once cold, `mark_hot_start`,
    /// then per re-solve patch bounds (which mutate CLP natively, preserving the
    /// factorization) and call this. On `CLP_STATUS_OPTIMAL` the CLP-owned
    /// solution pointers are copied immediately (valid only until the next solve).
    ///
    /// The determinism contract for this path is **self-consistent**
    /// reproducibility (run-to-run + cross-instance bit-for-bit) plus
    /// declaration-order invariance — it is NOT required to land on the same
    /// dual vertex as a cold solve of the same model.
    ///
    /// # Errors
    ///
    /// Mirrors [`SolverInterface::solve`](crate::SolverInterface::solve): `Infeasible` on `PRIMAL_INFEASIBLE`,
    /// `Unbounded` on `DUAL_INFEASIBLE`, `IterationLimit` on `STOPPED`,
    /// `InternalError` on `ERRORS` or any unexpected status int.
    ///
    /// # Panics
    ///
    /// Panics if no snapshot is active (`hot_start_token` is null).
    pub fn solve_from_hot_start(&mut self) -> Result<SolutionView<'_>, SolverError> {
        assert!(
            !self.hot_start_token.is_null(),
            "solve_from_hot_start called without an active snapshot — call mark_hot_start first"
        );

        let t0 = Instant::now();
        // SAFETY: `self.handle` is a valid, non-null CLP pointer with a solved
        // model loaded. `self.hot_start_token` is the non-null token from a prior
        // `mark_hot_start` on this same handle (asserted above); it is forwarded
        // to CLP unchanged and never dereferenced here. The returned int is the
        // CLP solve status.
        let status =
            unsafe { clp_ffi::cobre_clp_solve_from_hot_start(self.handle, self.hot_start_token) };
        let solve_time = t0.elapsed().as_secs_f64();

        self.stats.solve_count += 1;

        if status == clp_ffi::CLP_STATUS_OPTIMAL {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved; iteration count is non-negative so the cast is
            // safe.
            #[allow(clippy::cast_sign_loss)]
            let iterations = unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
            // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
            // just been solved. Objective is already in minimize sense.
            let objective = unsafe { clp_ffi::cobre_clp_objective_value(self.handle) };

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
                // SAFETY: `self.handle` is a valid, non-null CLP pointer;
                // iteration count is non-negative so the cast is safe.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
                Err(SolverError::IterationLimit { iterations })
            }
            clp_ffi::CLP_STATUS_ERRORS => Err(SolverError::InternalError {
                message: "CLP hot-start solve failed (simplex returned ERRORS status)".to_string(),
                error_code: Some(4),
            }),
            other => Err(SolverError::InternalError {
                message: format!("CLP hot-start returned unexpected status {other}"),
                error_code: Some(other),
            }),
        }
    }

    /// Releases the active hot-start snapshot, freeing the `saveStuff` token and
    /// resetting `hot_start_token` to null.
    ///
    /// A no-op when no snapshot is active, so it is always safe to call (and is
    /// called by `Drop`).
    pub fn unmark_hot_start(&mut self) {
        if self.hot_start_token.is_null() {
            return;
        }
        // SAFETY: `self.handle` is a valid, non-null CLP pointer.
        // `self.hot_start_token` is the non-null token from a prior
        // `mark_hot_start` on this same handle (guarded above); the shim forwards
        // it to `unmarkHotStart`, which frees it. It is never dereferenced here
        // and is nulled immediately so it cannot be reused after the free.
        unsafe {
            clp_ffi::cobre_clp_unmark_hot_start(self.handle, self.hot_start_token);
        }
        self.hot_start_token = std::ptr::null_mut();
    }
}

/// Normalizes a raw CLP row price into cobre's canonical dual-sign convention.
///
/// Identity is correct: `cobre_clp_get_row_price` already matches the canonical
/// convention (`HiGHS` does not negate either). See
/// `tests/_clp_sign_convention_probe.rs`.
const fn normalize_row_dual(raw: f64) -> f64 {
    raw
}

/// Converts a `usize` to `i32`, debug-asserting on overflow.
///
/// Each value is bounded by the merged nnz / row count, which `add_rows` asserts
/// fits in `i32`; this guards the per-entry writes in the CSR→CSC merge.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn i32_from_usize(v: usize) -> i32 {
    debug_assert!(
        i32::try_from(v).is_ok(),
        "value {v} overflows i32: LP exceeds CLP API limit"
    );
    v as i32
}

impl Drop for ClpSolver {
    fn drop(&mut self) {
        // Release a still-held hot-start snapshot before destroying the model so
        // no `saveStuff` leaks (every `mark_hot_start` gets exactly one release).
        self.unmark_hot_start();
        // SAFETY: valid CLP pointer from construction, called once per instance.
        unsafe { clp_ffi::cobre_clp_destroy(self.handle) };
    }
}

/// Returns the CLP version as a `"major.minor.patch"` string.
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
