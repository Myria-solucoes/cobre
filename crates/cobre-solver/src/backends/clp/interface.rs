//! `impl SolverInterface for ClpSolver`.
//!
//! Additional `impl` block (the struct and its lifecycle/hot-start primitives
//! are owned by `solver`): the public [`SolverInterface`](crate::SolverInterface)
//! surface — profile application, model loading, row/bound mutation, the warm-start
//! `solve` entry point (which routes spurious `PRIMAL_INFEASIBLE`/`STOPPED`
//! through `retry`'s escalation ladder), basis extraction, and statistics
//! reporting.

use std::time::Instant;

use super::config::{ClpAlgorithm, ClpProfile};
use super::retry::LADDER_RUNGS;
use super::solver::{ClpSolver, clp_version, i32_from_usize};
use crate::{
    SolverInterface, clp_ffi,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

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
    /// It then drives the two C++-class-only knobs through the shim, each behind
    /// a behavior-neutral sentinel so the default profile stays byte-identical to
    /// a build that never issued either setter:
    ///
    /// - **Dual-row pricing** ([`ClpProfile::dual_pricing_mode`]): a non-default
    ///   mode (`!= 3`) installs that pricing rule via `set_dual_row_steepest`;
    ///   mode `3` is CLP's own `ClpDualRowSteepest` constructor default, so it is
    ///   the "leave CLP default — do not call" sentinel and issues no shim call.
    ///   A tuned profile pinning full DSE (`dual_pricing_mode = 1`) therefore
    ///   drives the setter; the default/forward/simulation profiles do not.
    /// - **Refactorization cadence** ([`ClpProfile::factorization_frequency`]): a
    ///   non-zero value calls `cobre_clp_set_factorization_frequency`; `0` is the
    ///   "leave CLP's internal default — do not call" sentinel and issues no call.
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

        // Dual-row pricing: skip mode 3 (CLP's own steepest-edge ctor default)
        // so the default/forward/simulation profiles issue no shim call and stay
        // byte-identical to today; any other mode (e.g. full DSE = 1) installs
        // the pricing rule via the shim.
        if profile.dual_pricing_mode != 3 {
            self.set_dual_row_steepest(profile.dual_pricing_mode);
        }

        // Refactorization cadence: 0 is the "leave CLP's internal default"
        // sentinel (no call); any non-zero value sets the cadence via the shim.
        if profile.factorization_frequency != 0 {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer from
            // `cobre_clp_create()`. The shim reaches the live `ClpSimplex`
            // through the wrapper's `model_` member and calls
            // `setFactorizationFrequency`, which stores the cadence on the
            // factorization object; it retains no pointer and cannot fail on a
            // valid handle.
            unsafe {
                clp_ffi::cobre_clp_set_factorization_frequency(
                    self.handle,
                    profile.factorization_frequency,
                );
            }
        }
    }

    /// Fully reset the CLP simplex state by recreating the underlying model.
    ///
    /// `Clp_loadProblem` swaps the model data but does NOT heal the
    /// `ClpSimplex`-level rim/pricing state, so stale steepest-edge reference
    /// weights persist across solves and make the landed vertex on
    /// alternative-optima LPs depend on the order a worker processed prior
    /// scenarios — breaking thread/rank-count determinism. Recreating the
    /// `ClpSimplex` (a fresh, empty model) discards that state entirely: the CLP
    /// analogue of the clean state `HiGHS` gets for free on every `passLp`. The
    /// next `load_model` repopulates the model; the cached profile is re-applied
    /// so perturbation/scaling/tolerance/pricing/iteration-cap configuration
    /// survives the swap.
    fn reset_solver_state(&mut self) {
        // SAFETY: `cobre_clp_create` has no preconditions; it allocates a new
        // empty CLP model or returns null on allocation failure.
        let new_handle = unsafe { clp_ffi::cobre_clp_create() };
        if new_handle.is_null() {
            // Allocating an empty model failed (extremely unlikely). Keep the
            // existing handle rather than abort; determinism degrades but the
            // run continues.
            return;
        }
        // Release any hot-start snapshot bound to the OLD handle before it is
        // destroyed — the `saveStuff` token belongs to the old model.
        if !self.hot_start_token.is_null() {
            self.unmark_hot_start();
        }
        // SAFETY: `self.handle` is the valid handle from construction (or a prior
        // reset); `cobre_clp_destroy` frees it. It is immediately replaced by the
        // freshly created, non-null `new_handle` before any further use.
        unsafe { clp_ffi::cobre_clp_destroy(self.handle) };
        self.handle = new_handle;
        self.has_model = false;
        // SAFETY: `self.handle` is the just-created non-null model; mirror
        // `new()` by silencing CLP's per-solve logging.
        unsafe { clp_ffi::cobre_clp_set_log_level(self.handle, 0) };
        // Re-apply the cached profile so the fresh model carries the same
        // configuration the previous one had (ClpProfile is `Copy`).
        let profile = self.current_profile;
        self.apply_profile(&profile);
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
        // Release any active hot-start snapshot before replacing the model — the
        // saveStuff belongs to the old model's factorization and is invalid after
        // Clp_loadProblem. Mirrors the self-heal guard in `mark_hot_start`. This
        // releasing-before-reload is the correct ordering and keeps `Drop` from
        // unmarking a stale token after the model is swapped. (Note: vendored CLP
        // leaves `ClpSimplex::factorization_` dangling after `unmarkHotStart` and
        // `Clp_loadProblem` does not heal the ClpSimplex-level rim, so a *solve*
        // after a reload-following-a-hot-start is unsafe at the CLP level — out of
        // this guard's scope, and not on any persistent-solver path, which never
        // reloads a fresh model after marking.)
        if !self.hot_start_token.is_null() {
            self.unmark_hot_start();
        }
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
        // in-Rust copy equals the loaded model. `add_rows`/`set_*_bounds` patch
        // these buffers (keeping them the canonical, declaration-ordered mirror)
        // and reconcile the change into CLP natively via `cobre_clp_add_rows` /
        // `cobre_clp_chg_*`. Cleared then filled (no `shrink_to_fit`) so capacity
        // stabilises at the peak.
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
    /// pairs into the per-column lists of the retained CSC (keeping the retained
    /// buffers the canonical mirror), then appends the rows into CLP **natively**
    /// via `cobre_clp_add_rows` — which takes the CSR batch directly, so the
    /// CSC transpose feeds only the retained mirror, not the FFI call. The
    /// native append preserves CLP's persistent simplex basis across the append
    /// (no full rebuild). The retained `row_lower`/`row_upper` are extended,
    /// `num_rows`/`num_nz` are bumped, and `row_dual` is resized.
    ///
    /// A non-empty append does, however, **release any captured hot-start
    /// snapshot**: the snapshot's saveStuff pins the pre-append factorization/rim
    /// and is stale once the row dimension changes. This mirrors the self-heal
    /// guard in [`Self::load_model`]; release is performed via
    /// [`Self::unmark_hot_start`]. An empty batch (`num_rows == 0`) makes no
    /// structural change and leaves an active snapshot intact.
    ///
    /// # Panics
    ///
    /// Panics if `rows.num_rows` or the batch nnz does not fit in `i32`.
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

        // A non-empty add_rows changes the row dimension, so any captured hot-start
        // snapshot (which pins the pre-append factorization/rim) is now stale. Release
        // it before mutating — mirrors the self-heal guard in `load_model`.
        if !self.hot_start_token.is_null() {
            self.unmark_hot_start();
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

        // `rows.num_rows` was asserted to fit in i32 at the top of this method.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let number = rows.num_rows as i32;
        // Append the rows into CLP natively from the CSR batch (NOT the retained
        // CSC). `Clp_addRows` takes CSR directly, so the batch's own
        // `row_starts`/`col_indices`/`values` are forwarded verbatim; the CSC
        // transpose above only refreshes the retained mirror. The native append
        // preserves CLP's factorization/basis (no full reload).
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer from
        //   `cobre_clp_create()` with a model loaded.
        // - `number` (== `rows.num_rows`) is non-negative and fits in i32
        //   (asserted at the top of this method).
        // - The pointer arguments point into the caller's `rows` CSR slices,
        //   which outlive this call: `row_lower`/`row_upper` have `num_rows`
        //   entries, `row_starts` has `num_rows + 1` entries, and
        //   `col_indices`/`values` have `row_starts[num_rows]` entries (the
        //   `RowBatch` CSR contract). The batch nnz was asserted to fit in i32.
        // - Row bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_add_rows(
                self.handle,
                number,
                rows.row_lower.as_ptr(),
                rows.row_upper.as_ptr(),
                rows.row_starts.as_ptr(),
                rows.col_indices.as_ptr(),
                rows.values.as_ptr(),
            );
        }

        // Grow the row-dual extraction buffer to cover the appended rows.
        self.row_dual.resize(self.num_rows, 0.0);
    }

    /// Patches the bounds of an arbitrary set of rows in place.
    ///
    /// `indices`, `lower`, and `upper` must have equal length. An empty
    /// `indices` slice is a no-op. Each retained `row_lower`/`row_upper` entry at
    /// the given index is overwritten (keeping the retained vectors the canonical
    /// mirror), then the **full** patched bound vectors are pushed into CLP
    /// natively via `cobre_clp_chg_row_lower`/`cobre_clp_chg_row_upper` — which
    /// each take the whole array, not an index subset. This preserves CLP's
    /// factorization/basis across the patch. Bounds are forwarded verbatim.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds).
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
        // Push the full patched row-bound vectors into CLP. `Clp_chgRowLower`/
        // `Upper` replace the model's entire bound array (not a subset), so the
        // retained vectors — already patched above — are forwarded in full.
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer with a model loaded.
        // - `self.row_lower`/`self.row_upper` each have exactly `self.num_rows`
        //   entries (maintained by `load_model`/`add_rows`), matching the model's
        //   current row count that the C wrapper queries to size its translation.
        // - The pointers reference owned `Vec` data alive for the call.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_chg_row_lower(self.handle, self.row_lower.as_ptr());
            clp_ffi::cobre_clp_chg_row_upper(self.handle, self.row_upper.as_ptr());
        }
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Patches the bounds of an arbitrary set of columns in place.
    ///
    /// Symmetric to [`Self::set_row_bounds`] but patches the retained
    /// `col_lower`/`col_upper` and pushes them into CLP natively via
    /// `cobre_clp_chg_column_lower`/`cobre_clp_chg_column_upper` (each taking the
    /// full bound array). An empty `indices` slice is a no-op. This preserves
    /// CLP's factorization/basis across the patch. Bounds are forwarded verbatim.
    ///
    /// # Panics
    ///
    /// Panics if the three slices differ in length, or if any index is out of
    /// range (debug builds).
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
        // Push the full patched column-bound vectors into CLP. `Clp_chgColumn*`
        // replace the model's entire bound array (not a subset), so the retained
        // vectors — already patched above — are forwarded in full.
        // SAFETY:
        // - `self.handle` is a valid, non-null CLP pointer with a model loaded.
        // - `self.col_lower`/`self.col_upper` each have exactly `self.num_cols`
        //   entries (maintained by `load_model`), matching the model's current
        //   column count that the C wrapper queries to size its translation.
        // - The pointers reference owned `Vec` data alive for the call.
        // - Bounds are forwarded verbatim; the C wrapper owns the
        //   ±IEEE-inf → ±DBL_MAX translation.
        unsafe {
            clp_ffi::cobre_clp_chg_column_lower(self.handle, self.col_lower.as_ptr());
            clp_ffi::cobre_clp_chg_column_upper(self.handle, self.col_upper.as_ptr());
        }
        self.stats.total_set_bounds_time_seconds += t0.elapsed().as_secs_f64();
    }

    /// Solves the loaded LP via the dual simplex and returns the optimal
    /// solution as a [`SolutionView`] borrowing the solver's owned buffers.
    ///
    /// Runs a single cold `cobre_clp_dual` call (no per-solve wall-clock
    /// budget). On `CLP_STATUS_OPTIMAL` the three CLP-owned solution pointers
    /// are copied **immediately** into `col_value`/`col_dual`/`row_dual` (they
    /// are valid only until the next solve), the row duals are normalized via
    /// `normalize_row_dual`, and a `SolutionView` borrowing those buffers is
    /// returned. This first-solve happy path is byte-identical to a build with
    /// no escalation ladder.
    ///
    /// # Escalation ladder
    ///
    /// CLP's bare dual simplex can spuriously report `PRIMAL_INFEASIBLE` on
    /// numerically delicate feasible LPs. On `PRIMAL_INFEASIBLE` or `STOPPED`
    /// the failure path runs `escalate_solve` — re-solving the same
    /// already-loaded model (bounds plus any installed basis) with primal
    /// simplex, then perturbation, then scaling — before surfacing the error. A
    /// solve recovered by the ladder returns `Ok` and counts as a retried
    /// success (`success_count` + `retry_count`, never `first_try_successes`).
    /// `DUAL_INFEASIBLE`, `ERRORS`, and unexpected statuses stay terminal. The
    /// floor (deterministic) settings are re-applied after the ladder runs
    /// regardless of outcome, so the next `solve` starts from the clean config.
    /// Non-recovered statuses map to a [`SolverError`].
    ///
    /// # Warm-start basis
    ///
    /// When `basis = Some(b)`, `b` is reinstalled into the CLP model
    /// element-by-element via `install_basis` before the solve; both
    /// `None` and `Some(&b)` then run the same cold `cobre_clp_dual` path (CLP
    /// retains its internal basis across consecutive solves). Unlike the `HiGHS`
    /// backend, CLP's per-element status setters do not validate global basis
    /// consistency, so there is no `BasisInconsistent` error path here. An
    /// undersized row basis (`b.row_status.len() < self.num_rows`) is rejected
    /// by `install_basis` with `Err(SolverError::BasisRowCountMismatch)` rather
    /// than being silently short-copied; the caller should fall back to a cold
    /// solve.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError::Infeasible)` on `PRIMAL_INFEASIBLE` that the
    /// escalation ladder could not recover, `Err(SolverError::Unbounded)` on
    /// `DUAL_INFEASIBLE`, `Err(SolverError::IterationLimit { .. })` on `STOPPED`
    /// the ladder could not recover, and `Err(SolverError::InternalError { .. })`
    /// on `ERRORS` or any unexpected status int. Returns
    /// `Err(SolverError::BasisRowCountMismatch { .. })` when an offered warm-start
    /// basis has fewer row entries than the LP has rows.
    ///
    /// # Panics
    ///
    /// Panics if no model is loaded (`!self.has_model`).
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        assert!(self.has_model, "solve called without a loaded model");

        if let Some(b) = basis {
            // Reinstall the offered basis element-by-element before the solve.
            // CLP retains its internal basis across consecutive simplex calls,
            // so both `None` (warm from prior solve) and `Some` (warm from the
            // reinstalled basis) fall through to the same cold solve below.
            self.install_basis(b)?;
        }

        let t0 = Instant::now();
        // Dispatch on the profile's algorithm selection. Both the dual and
        // primal simplex return the same `Clp_status` int space, so the status
        // mapping, `copy_solution`, and `SolutionView` construction below are
        // shared and unchanged.
        let status = match self.current_profile.algorithm {
            ClpAlgorithm::Dual => {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer from
                // `cobre_clp_create()` with a model loaded (asserted via
                // `has_model`). `if_values_pass = 0` requests a cold solve (no
                // values pass). The returned int is the CLP solve status.
                unsafe { clp_ffi::cobre_clp_dual(self.handle, 0) }
            }
            ClpAlgorithm::Primal => {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer from
                // `cobre_clp_create()` with a model loaded (asserted via
                // `has_model`). `if_values_pass = 0` requests a cold solve (no
                // values pass). The returned int is the CLP solve status.
                unsafe { clp_ffi::cobre_clp_primal(self.handle, 0) }
            }
        };
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

        // Failure path. `PRIMAL_INFEASIBLE` (the confirmed false-infeasible from
        // CLP's bare dual simplex) and `STOPPED` (iteration/time stop, which a
        // different algorithm may converge through) are routed through the
        // escalation ladder before surfacing the error. `DUAL_INFEASIBLE`,
        // `ERRORS`, and any unexpected status int stay terminal (a genuine
        // unbounded/error is not retry-recoverable), mirroring how the `HiGHS`
        // backend treats a genuine `INFEASIBLE` as terminal.
        //
        // `failure_count` is deliberately NOT incremented before the ladder
        // runs: a solve recovered by escalation counts only as a (retried)
        // success, never as a failure. It is incremented on the terminal paths
        // below and on ladder exhaustion (mirrors `HiGHS` `solve_inner`, where
        // `failure_count` rises only on the early terminal return or the
        // escalation `Err` arm).
        if status == clp_ffi::CLP_STATUS_PRIMAL_INFEASIBLE || status == clp_ffi::CLP_STATUS_STOPPED
        {
            let outcome = self.escalate_solve();

            // Restore the floor (deterministic) settings unconditionally —
            // success OR exhaustion — so the NEXT `solve` starts from the clean
            // config the happy path depends on. Re-applying `current_profile`
            // resets perturbation (-> profile value, 102/off by default),
            // scaling (-> off), and both feasibility tolerances in one pass via
            // `apply_profile`. `apply_profile` reassigns `self.current_profile`
            // to the same value it already holds (no behavioral change), so the
            // floor is byte-identical to a build that never escalated. Mirrors
            // the unconditional `restore_default_settings()` in `HiGHS`
            // `retry_escalation`.
            let profile = self.current_profile;
            self.apply_profile(&profile);

            if let Some(escalation) = outcome {
                // Recovered. The solution was copied into the owned buffers by
                // `escalate_solve` on the rung that returned OPTIMAL. Count it
                // as a retried success: bump `success_count` and `retry_count`,
                // but NOT `first_try_successes` (the dual first solve failed).
                self.stats.success_count += 1;
                self.stats.retry_count += escalation.attempts;
                self.stats.total_iterations += escalation.iterations;
                self.stats.total_solve_time_seconds += escalation.solve_time;

                return Ok(SolutionView {
                    objective: escalation.objective,
                    primal: &self.col_value[..self.num_cols],
                    dual: &self.row_dual[..self.num_rows],
                    reduced_costs: &self.col_dual[..self.num_cols],
                    iterations: escalation.iterations,
                    solve_time_seconds: escalation.solve_time,
                });
            }

            // Ladder exhausted: surface the ORIGINAL error, exactly as today.
            // Count the attempted rungs and the final failure. `LADDER_RUNGS`
            // (<= 5) widens losslessly to the `u64` `retry_count`.
            self.stats.retry_count += LADDER_RUNGS as u64;
            self.stats.failure_count += 1;
            if status == clp_ffi::CLP_STATUS_PRIMAL_INFEASIBLE {
                return Err(SolverError::Infeasible);
            }
            // STOPPED: map to IterationLimit using the last solve's iteration
            // count (mirrors the terminal branch below).
            // SAFETY: `self.handle` is a valid, non-null CLP pointer; iteration
            // count is non-negative so the cast is safe.
            #[allow(clippy::cast_sign_loss)]
            let iterations = unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
            return Err(SolverError::IterationLimit { iterations });
        }

        self.stats.failure_count += 1;
        match status {
            clp_ffi::CLP_STATUS_DUAL_INFEASIBLE => Err(SolverError::Unbounded),
            clp_ffi::CLP_STATUS_ERRORS => Err(SolverError::InternalError {
                message: "CLP solve failed (simplex returned ERRORS status)".to_string(),
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

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.stats);
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
}
