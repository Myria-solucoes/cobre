//! Raw FFI bindings to the CLP C wrapper layer.
//!
//! These are low-level unsafe functions that map 1:1 to the `cobre_clp_*`
//! functions declared in `csrc/clp_wrapper.h`.  Use the safe wrappers in
//! the parent module rather than calling these directly.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::os::raw::{c_double, c_void};

/// C `int32_t` mapped to Rust `i32`.
pub type int32_t = i32;

// ============================================================
// CLP solve status constants (Clp_status)
// ============================================================

/// `Clp_status` == 0 — the model was solved to optimality.
pub const CLP_STATUS_OPTIMAL: i32 = 0;
/// `Clp_status` == 1 — the model is primal infeasible.
pub const CLP_STATUS_PRIMAL_INFEASIBLE: i32 = 1;
/// `Clp_status` == 2 — the model is dual infeasible (unbounded).
pub const CLP_STATUS_DUAL_INFEASIBLE: i32 = 2;
/// `Clp_status` == 3 — stopped on iterations / time.
pub const CLP_STATUS_STOPPED: i32 = 3;
/// `Clp_status` == 4 — stopped due to errors.
pub const CLP_STATUS_ERRORS: i32 = 4;

// Per-element basis status codes (`Clp_getColumnStatus` / `Clp_getRowStatus`,
// per `ClpSimplex.hpp`): 0 free, 1 basic, 2 at-upper, 3 at-lower, 4 superbasic,
// 5 fixed. These are round-tripped verbatim as raw `i32` (see `ClpSolver`'s
// basis capture/install paths) and never compared against named constants, so
// no symbolic definitions are kept here.

unsafe extern "C" {
    // ============================================================
    // Lifecycle
    // ============================================================

    /// Create a CLP simplex model. Returns an opaque pointer; caller owns it.
    /// Wraps `Clp_newModel()`.
    pub fn cobre_clp_create() -> *mut c_void;

    /// Destroy a CLP simplex model and free all associated memory.
    /// Wraps `Clp_deleteModel()`.
    pub fn cobre_clp_destroy(model: *mut c_void);

    /// Set the model's logging verbosity. Wraps `Clp_setLogLevel()`.
    /// Level `0` is silent; cobre applies it at construction so CLP does not
    /// print per-solve progress to stdout (mirrors `HiGHS` `output_flag=0`).
    pub fn cobre_clp_set_log_level(model: *mut c_void, value: int32_t);

    // ============================================================
    // Model Loading
    // ============================================================

    /// Load a complete LP into the model from column-major (CSC) data.
    /// Wraps `Clp_loadProblem()`, then fixes the objective sense to minimize.
    pub fn cobre_clp_load_problem(
        model: *mut c_void,
        num_cols: int32_t,
        num_rows: int32_t,
        col_starts: *const int32_t,
        row_indices: *const int32_t,
        values: *const c_double,
        col_lower: *const c_double,
        col_upper: *const c_double,
        objective: *const c_double,
        row_lower: *const c_double,
        row_upper: *const c_double,
    );

    // ============================================================
    // Incremental Mutation
    //
    // These mutate a loaded model in place (preserving CLP's factorization and
    // basis across the change) instead of rebuilding it. The C wrapper owns the
    // ±IEEE-infinity → ±COIN_DBL_MAX bound translation, so the bound slices are
    // forwarded verbatim. `row_starts`/`columns` are `*const i32`, matching the
    // `CoinBigIndex == int` build asserted at compile time in the C wrapper.
    // ============================================================

    /// Append `number` constraint rows from row-major (CSR) data.
    /// Wraps `Clp_addRows()`. Bounds are infinity-translated by the C wrapper.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// `row_lower`/`row_upper` must be valid for `number` reads; `row_starts`
    /// for `number + 1` reads; `columns`/`elements` for `row_starts[number]`
    /// reads. All pointers must outlive the call. `number` must be non-negative.
    pub fn cobre_clp_add_rows(
        model: *mut c_void,
        number: int32_t,
        row_lower: *const c_double,
        row_upper: *const c_double,
        row_starts: *const int32_t,
        columns: *const int32_t,
        elements: *const c_double,
    );

    /// Replace the model's row lower bounds with a full-length array.
    /// Wraps `Clp_chgRowLower()`. Bounds are infinity-translated by the wrapper.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// `row_lower` must be valid for reads of the model's current row count
    /// `f64`s and must outlive the call.
    pub fn cobre_clp_chg_row_lower(model: *mut c_void, row_lower: *const c_double);

    /// Replace the model's row upper bounds with a full-length array.
    /// Wraps `Clp_chgRowUpper()`. Bounds are infinity-translated by the wrapper.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// `row_upper` must be valid for reads of the model's current row count
    /// `f64`s and must outlive the call.
    pub fn cobre_clp_chg_row_upper(model: *mut c_void, row_upper: *const c_double);

    /// Replace the model's column lower bounds with a full-length array.
    /// Wraps `Clp_chgColumnLower()`. Bounds are infinity-translated by the wrapper.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// `column_lower` must be valid for reads of the model's current column count
    /// `f64`s and must outlive the call.
    pub fn cobre_clp_chg_column_lower(model: *mut c_void, column_lower: *const c_double);

    /// Replace the model's column upper bounds with a full-length array.
    /// Wraps `Clp_chgColumnUpper()`. Bounds are infinity-translated by the wrapper.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// `column_upper` must be valid for reads of the model's current column count
    /// `f64`s and must outlive the call.
    pub fn cobre_clp_chg_column_upper(model: *mut c_void, column_upper: *const c_double);

    // ============================================================
    // Solving
    // ============================================================

    /// Run the dual simplex algorithm. Wraps `Clp_dual()`.
    /// Returns the CLP solve status int (0 = optimal).
    pub fn cobre_clp_dual(model: *mut c_void, if_values_pass: int32_t) -> int32_t;

    /// Run the primal simplex algorithm. Wraps `Clp_primal()`.
    /// Returns the CLP solve status int (0 = optimal).
    pub fn cobre_clp_primal(model: *mut c_void, if_values_pass: int32_t) -> int32_t;

    // ============================================================
    // Solution Extraction
    // ============================================================

    /// Get the objective function value. Wraps `Clp_objectiveValue()`.
    pub fn cobre_clp_objective_value(model: *const c_void) -> c_double;

    /// Get the problem status. Wraps `Clp_status()`.
    pub fn cobre_clp_status(model: *const c_void) -> int32_t;

    /// Get the simplex iteration count from the most recent solve.
    /// Wraps `Clp_numberIterations()`.
    pub fn cobre_clp_number_iterations(model: *const c_void) -> int32_t;

    /// Get the primal column solution. Wraps `Clp_getColSolution()`.
    /// Returns a pointer into CLP-owned memory (length `num_cols`) valid only
    /// until the next solve; do NOT free it.
    pub fn cobre_clp_get_col_solution(model: *const c_void) -> *const c_double;

    /// Get the dual row solution (row prices). Wraps `Clp_getRowPrice()`.
    /// Returns a pointer into CLP-owned memory (length `num_rows`) valid only
    /// until the next solve; do NOT free it.
    pub fn cobre_clp_get_row_price(model: *const c_void) -> *const c_double;

    /// Get the reduced costs (dual column solution).
    /// Wraps `Clp_getReducedCost()`.
    /// Returns a pointer into CLP-owned memory (length `num_cols`) valid only
    /// until the next solve; do NOT free it.
    pub fn cobre_clp_get_reduced_cost(model: *const c_void) -> *const c_double;

    // ============================================================
    // Configuration
    // ============================================================

    /// Set the perturbation mode. Wraps `Clp_setPerturbation()`.
    pub fn cobre_clp_set_perturbation(model: *mut c_void, value: int32_t);

    /// Set the scaling mode. Wraps `Clp_scaling()`.
    pub fn cobre_clp_scaling(model: *mut c_void, mode: int32_t);

    /// Set the primal feasibility tolerance. Wraps `Clp_setPrimalTolerance()`.
    pub fn cobre_clp_set_primal_tolerance(model: *mut c_void, value: c_double);

    /// Set the dual feasibility tolerance. Wraps `Clp_setDualTolerance()`.
    pub fn cobre_clp_set_dual_tolerance(model: *mut c_void, value: c_double);

    /// Set the maximum number of simplex iterations.
    /// Wraps `Clp_setMaximumIterations()`.
    pub fn cobre_clp_set_maximum_iterations(model: *mut c_void, value: int32_t);

    // ============================================================
    // Basis Management
    // ============================================================

    /// Get the basis status of a column (variable).
    /// Wraps `Clp_getColumnStatus()`.
    pub fn cobre_clp_get_column_status(model: *const c_void, sequence: int32_t) -> int32_t;

    /// Set the basis status of a column (variable).
    /// Wraps `Clp_setColumnStatus()`.
    pub fn cobre_clp_set_column_status(model: *mut c_void, sequence: int32_t, value: int32_t);

    /// Get the basis status of a row (slack). Wraps `Clp_getRowStatus()`.
    pub fn cobre_clp_get_row_status(model: *const c_void, sequence: int32_t) -> int32_t;

    /// Set the basis status of a row (slack). Wraps `Clp_setRowStatus()`.
    pub fn cobre_clp_set_row_status(model: *mut c_void, sequence: int32_t, value: int32_t);

    // ============================================================
    // C++ class-only knobs (implemented in clp_wrapper_cpp.cpp)
    //
    // These reach methods that exist only on the C++ `ClpSimplex` class and are
    // not in the CLP C interface: dual-steepest-edge pricing, factorization
    // frequency, and the hot-start snapshot/restore trio. The C++ shim
    // static-casts the opaque `model` handle to the concrete `Clp_Simplex`
    // wrapper struct (what `cobre_clp_create` returns) and reads `->model_` to
    // reach the live `ClpSimplex`, matching how every `Clp_*` C-API call reaches
    // the model. The `save_stuff` token is CLP-owned and kept opaque on the Rust
    // side — never dereferenced, always paired (mark/unmark) on the same model
    // instance.
    // ============================================================

    /// Select dual-steepest-edge pricing. Wraps
    /// `ClpSimplex::setDualRowPivotAlgorithm(ClpDualRowSteepest(mode))`.
    /// `mode` 1 = full DSE; 3 = the `ClpDualRowSteepest` default.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer from
    /// `cobre_clp_create`.
    pub fn cobre_clp_set_dual_row_steepest(model: *mut c_void, mode: int32_t);

    /// Set the simplex factorization refactor cadence.
    /// Wraps `ClpSimplex::setFactorizationFrequency()`.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer from
    /// `cobre_clp_create`.
    pub fn cobre_clp_set_factorization_frequency(model: *mut c_void, value: int32_t);

    /// Snapshot the model for hot-started re-solves. Wraps
    /// `ClpSimplex::markHotStart` and returns the opaque CLP-allocated
    /// `saveStuff` token.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer with a model loaded.
    /// The returned token is CLP-owned; keep it opaque and release it with
    /// `cobre_clp_unmark_hot_start` on the same `model`.
    pub fn cobre_clp_mark_hot_start(model: *mut c_void) -> *mut c_void;

    /// Re-solve the model from the hot-start snapshot. Wraps
    /// `ClpSimplex::solveFromHotStart` and returns the CLP solve status int
    /// (0 = optimal; same space as `cobre_clp_status`).
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer. `save_stuff` must be
    /// a non-null token from a prior `cobre_clp_mark_hot_start` on this same
    /// `model`. It is forwarded to CLP unchanged and never dereferenced by Rust.
    pub fn cobre_clp_solve_from_hot_start(model: *mut c_void, save_stuff: *mut c_void) -> int32_t;

    /// Release a hot-start snapshot, freeing the `saveStuff` token. Wraps
    /// `ClpSimplex::unmarkHotStart`.
    ///
    /// # Safety
    ///
    /// `model` must be a valid, non-null CLP model pointer. `save_stuff` must be
    /// a non-null token from a prior `cobre_clp_mark_hot_start` on this same
    /// `model`; after this call it is freed and must not be reused.
    pub fn cobre_clp_unmark_hot_start(model: *mut c_void, save_stuff: *mut c_void);

    // ============================================================
    // Version query (no instance required)
    // ============================================================

    /// Return the CLP major version number. Wraps `Clp_VersionMajor()`.
    pub fn cobre_clp_version_major() -> int32_t;

    /// Return the CLP minor version number. Wraps `Clp_VersionMinor()`.
    pub fn cobre_clp_version_minor() -> int32_t;

    /// Return the CLP release version number. Wraps `Clp_VersionRelease()`.
    pub fn cobre_clp_version_release() -> int32_t;
}

#[cfg(test)]
mod tests {
    use super::{
        CLP_STATUS_OPTIMAL, cobre_clp_create, cobre_clp_destroy, cobre_clp_dual,
        cobre_clp_load_problem, cobre_clp_objective_value, cobre_clp_set_log_level,
        cobre_clp_status,
    };

    /// Smoke test: create a CLP model, load a trivial 1-variable LP
    /// (minimize x, x ∈ [0, 10], no constraints), solve it, verify optimality
    /// and objective value, then destroy the model.
    ///
    /// This validates the full FFI pipeline: C wrapper compiles and links against
    /// CLP, Rust declarations match the C signatures, and a basic solve works
    /// end-to-end.
    #[test]
    fn test_clp_ffi_smoke_create_solve_destroy() {
        // SAFETY: `cobre_clp_create` takes no arguments and returns a freshly
        // allocated, caller-owned CLP model pointer (or null on allocation
        // failure). We check for null immediately below.
        let model = unsafe { cobre_clp_create() };
        assert!(!model.is_null(), "cobre_clp_create() returned null");

        // Silence CLP's default level-1 progress output (this raw-FFI test does
        // not go through `ClpSolver::new`, which silences it for production).
        // SAFETY: `model` is the non-null handle just created above.
        unsafe { cobre_clp_set_log_level(model, 0) };

        // Minimize x where x ∈ [0, 10] with no constraints. Expected: x* = 0, obj = 0.
        // CSC form for a 1-column, 0-row, 0-nonzero matrix:
        //   col_starts has length num_cols + 1 = 2, both offsets are 0,
        //   row_indices and values are empty.
        let col_starts: [i32; 2] = [0, 0];
        let row_indices: [i32; 0] = [];
        let values: [f64; 0] = [];
        let col_lower: [f64; 1] = [0.0];
        let col_upper: [f64; 1] = [10.0];
        let objective: [f64; 1] = [1.0];
        let row_lower: [f64; 0] = [];
        let row_upper: [f64; 0] = [];

        // SAFETY: `model` is the non-null pointer from `cobre_clp_create`. All
        // array pointers are valid for the dimensions passed: `col_starts` has
        // `num_cols + 1 = 2` entries; `col_lower`/`col_upper`/`objective` have
        // `num_cols = 1` entry; `row_indices`/`values`/`row_lower`/`row_upper`
        // are empty (length 0) matching `num_rows = 0` and `col_starts[1] = 0`
        // nonzeros. The pointers outlive the call.
        unsafe {
            cobre_clp_load_problem(
                model,
                1, // num_cols
                0, // num_rows
                col_starts.as_ptr(),
                row_indices.as_ptr(),
                values.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                objective.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
            );
        }

        // SAFETY: `model` is a valid CLP model with a loaded problem.
        // `if_values_pass = 0` requests no values pass.
        let solve_status = unsafe { cobre_clp_dual(model, 0) };
        assert_eq!(
            solve_status, CLP_STATUS_OPTIMAL,
            "cobre_clp_dual() returned solve status {solve_status}"
        );

        // SAFETY: `model` is a valid CLP model that has just been solved.
        let status = unsafe { cobre_clp_status(model) };
        assert_eq!(
            status, CLP_STATUS_OPTIMAL,
            "expected Optimal status, got {status}"
        );

        // Read the CLP-owned objective value BEFORE destroying the model.
        // SAFETY: `model` is a valid solved CLP model.
        let obj = unsafe { cobre_clp_objective_value(model) };
        assert!(
            (obj - 0.0_f64).abs() < 1e-10,
            "objective value {obj} is not within 1e-10 of expected 0.0"
        );

        // SAFETY: `model` is a valid CLP model created by `cobre_clp_create`
        // and not yet destroyed; this transfers ownership back to CLP for
        // deallocation. It is not used after this call.
        unsafe { cobre_clp_destroy(model) };
    }
}
