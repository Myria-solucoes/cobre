//! Benders cut-row construction for the SDDP LP.
//!
//! Builds the LP rows that encode the Future Cost Function's cuts. The cut-sign
//! convention is owned here: [`push_scaled_coefficient`] negates the stored raw
//! subgradient so each row reads `−∇·x + θ ≥ intercept`, yielding the Benders
//! cut `θ ≥ Q(x̂) + π'(x − x̂)`. See [`push_scaled_coefficient`] for the
//! cut-sign negation contract and [`crate::backward::extract_duals_from_view`]
//! for the subgradient-extraction side of the contract.
//!
//! Consumers: the backward pass, simulation, lower-bound evaluation, and DCS
//! (Dynamic Cut Selection). The forward training loop uses pre-baked templates
//! and does not call these builders.

use cobre_solver::{RowBatch, SolverInterface};

use crate::cut::FutureCostFunction;
use crate::indexer::StageIndexer;

/// Push one negated, scaled coefficient entry into the cut row batch.
///
/// Shared per-coefficient emit helper. Used by [`build_cut_row_batch_into`]
/// (the unified mask-driven path, via [`StageIndexer::lp_column_for_state`]) and
/// by [`build_delta_cut_row_batch_into`](crate::forward::build_delta_cut_row_batch_into)
/// / [`push_cut_row`] (which retain their own sparse/dense branching over
/// [`StageIndexer::state_to_lp_column`]). All three apply the same
/// negate-and-divide-by-scale rule here so it cannot drift apart during
/// maintenance.
#[inline]
pub(crate) fn push_scaled_coefficient(
    batch: &mut RowBatch,
    j: usize,
    coeff: f64,
    col_scale: &[f64],
) {
    debug_assert!(
        i32::try_from(j).is_ok(),
        "column index j={j} exceeds i32::MAX"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    batch.col_indices.push(j as i32);
    let d = if col_scale.is_empty() {
        1.0
    } else {
        col_scale[j]
    };
    batch.values.push(-coeff * d);
}

/// Append one Benders cut row to `batch` in CSR form.
///
/// Emits the negated-scaled state coefficients (via [`push_scaled_coefficient`],
/// sparse over `indexer.nonzero_state_indices` when non-empty, else dense over
/// all state indices), the positive scaled `theta` column entry, and the row
/// bounds `row_lower = intercept`, `row_upper = +INFINITY` — exactly the layout
/// [`build_cut_row_batch_into`] and [`append_new_cuts_to_lp`] use.
///
/// The caller pushes the `row_starts` offset for this row before calling (and
/// the final terminator / `num_rows` / `add_rows` afterward); this helper only
/// appends the row's non-zeros and bounds. Shared by `append_new_cuts_to_lp`
/// ("all not-yet-resident" cuts) and the DCS `append_slots_to_lp` (an explicit
/// slot set) so the two cannot drift apart.
#[inline]
pub(crate) fn push_cut_row(
    batch: &mut RowBatch,
    intercept: f64,
    coefficients: &[f64],
    indexer: &StageIndexer,
    col_scale: &[f64],
) {
    let theta_col = indexer.theta;
    let mask = &indexer.nonzero_state_indices;

    if mask.is_empty() {
        for (j, &c) in coefficients.iter().enumerate() {
            let lp_col = indexer.state_to_lp_column(j);
            // Padding-slot invariant: when state_to_lp_column returns j unchanged
            // and j falls inside the anticipated-state block, the slot is a padding
            // slot (slot >= k_p for its plant). The INVARIANT comment in
            // state_to_lp_column explains the 5-step chain that guarantees the
            // corresponding cut coefficient is 0.0. Assert this in debug builds.
            debug_assert!(
                !(lp_col == j
                    && indexer.n_anticipated > 0
                    && j >= indexer.anticipated_state.start
                    && j < indexer.anticipated_state.start + indexer.n_anticipated * indexer.k_max)
                    || c == 0.0,
                "padding-slot j={j} has non-zero cut coefficient {c}; \
                 shift_anticipated_state must have seeded a non-zero into a padding slot"
            );
            push_scaled_coefficient(batch, lp_col, c, col_scale);
        }
    } else {
        for &j in mask {
            let lp_col = indexer.state_to_lp_column(j);
            push_scaled_coefficient(batch, lp_col, coefficients[j], col_scale);
        }
    }

    debug_assert!(
        i32::try_from(theta_col).is_ok(),
        "theta_col={theta_col} exceeds i32::MAX"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    batch.col_indices.push(theta_col as i32);
    let d_theta = if col_scale.is_empty() {
        1.0
    } else {
        col_scale[theta_col]
    };
    batch.values.push(d_theta);

    batch.row_lower.push(intercept);
    batch.row_upper.push(f64::INFINITY);
}

/// Fill a pre-allocated [`RowBatch`] with Benders cut rows from the FCF.
///
/// Clears `batch` and repopulates it with active cuts from `fcf` at the
/// given `stage`. The buffers inside `batch` retain their allocated capacity
/// across calls, eliminating heap allocation on the hot path.
///
/// This is the allocation-free core used by `build_cut_row_batch`.
///
/// # Panics
///
/// Panics if the total number of non-zeros exceeds `i32::MAX` (the `HiGHS`
/// API limit for CSR indices).
pub fn build_cut_row_batch_into(
    batch: &mut RowBatch,
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
) {
    batch.clear();

    let n_state = indexer.n_state;
    let theta_col = indexer.theta;
    let mask = &indexer.nonzero_state_indices;

    // The cut-row loop is mask-driven only. `setup/mod.rs` populates the mask
    // unconditionally (storage-only → [0, n_state) ascending; pure-thermal →
    // empty); the precompute map is finalized in the same place. Guard the
    // missed-finalize regression — correctness is covered by the
    // `lp_column_for_state` fallback, this catches a setup wiring miss.
    debug_assert!(
        !indexer.state_to_lp_column_map.is_empty() || indexer.n_state == 0,
        "state_to_lp_column_map not finalized before build_cut_row_batch_into"
    );

    let num_cuts: usize = fcf.pools[stage].active_count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    // NNZ per cut = nonzero state entries + theta. For storage-only the mask is
    // the full [0, n_state) range, so this equals the old dense `n_state + 1`;
    // for pure-thermal it is `0 + 1`.
    let nnz_per_cut = mask.len() + 1;
    let total_nnz = num_cuts * nnz_per_cut;

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.active_cuts(stage) {
        debug_assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length {got} != n_state {expected}",
            got = coefficients.len(),
            expected = n_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.row_starts.push(nz_offset as i32);

        // Single mask-driven state coefficient loop. The mask carries the
        // nonzero state indices in ascending order; `lp_column_for_state`
        // reads the precomputed `state_to_lp_column(j)`.
        //
        // state_to_lp_column remaps outgoing-state indices to LP columns.
        // For storage (j < N) the mapping is identity. For lag dimensions
        // the outgoing state after shift_lag_state stores z_inflow at lag 0
        // and shifted incoming lags at lag 1+, so the cut must reference the
        // corresponding LP columns (z_inflow and incoming lag l−1).
        //
        // No padding-slot assert is needed here: the mask omits anticipated
        // padding slots (`set_nonzero_mask` emits only `slot < k_i`, exactly
        // the slots where `state_to_lp_column` is non-identity), so this loop
        // never visits a padding slot.
        for &j in mask {
            let lp_col = indexer.lp_column_for_state(j);
            push_scaled_coefficient(batch, lp_col, coefficients[j], col_scale);
        }

        debug_assert!(
            i32::try_from(theta_col).is_ok(),
            "theta_col={theta_col} exceeds i32::MAX"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.col_indices.push(theta_col as i32);
        let d_theta = if col_scale.is_empty() {
            1.0
        } else {
            col_scale[theta_col]
        };
        batch.values.push(d_theta);

        batch.row_lower.push(intercept);
        batch.row_upper.push(f64::INFINITY);

        nz_offset += nnz_per_cut;
    }

    #[allow(clippy::expect_used)]
    batch.row_starts.push(
        i32::try_from(total_nnz).expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
    );

    batch.num_rows = num_cuts;
}

/// Build a fresh [`RowBatch`] of Benders cut rows from the FCF.
///
/// Convenience wrapper around [`build_cut_row_batch_into`] that allocates a
/// new `RowBatch`. For allocation-free usage on the hot path, prefer calling
/// [`build_cut_row_batch_into`] with a pre-allocated batch.
#[must_use]
pub fn build_cut_row_batch(
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
) -> RowBatch {
    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, fcf, stage, indexer, col_scale);
    batch
}

/// Append only the newly active cuts (not yet in the LP) to a live solver.
///
/// Iterates over all active cuts in `fcf.pools[stage]`, checks `row_map` to
/// determine which are already present in the LP, builds a small [`RowBatch`]
/// containing only the new cuts, and calls `solver.add_rows()`. Updates
/// `row_map` with the new LP row indices.
///
/// Returns the number of new cuts appended (0 if none).
///
/// The LP rows produced use the same coefficient transformation as
/// [`build_cut_row_batch_into`]: negated state coefficients with column
/// scaling and a positive theta column entry.
///
/// # Callers and design invariant
///
/// There are three production callers of [`SolverInterface::add_rows`]: this
/// function (lower-bound LP), the backward pass delta-cut append in
/// `cobre_sddp::backward::load_backward_lp`, and a test-only fallback path in
/// `cobre_sddp::lower_bound`. The training-loop forward pass does not call
/// `add_rows`; it uses pre-baked templates exclusively. The lower-bound LP
/// grows monotonically across iterations (new cuts are appended; nothing is
/// ever removed), and re-baking its template each iteration would increase
/// cumulative setup cost from `O(n_iters)` to `O(n_iters^2)`. The
/// append-only design is intentional.
///
/// # Arguments
///
/// - `solver`: the live LP solver instance with a loaded model.
/// - `fcf`: the Future Cost Function containing all cut pools.
/// - `stage`: 0-based stage index.
/// - `indexer`: provides `n_state` and `theta` column index.
/// - `col_scale`: column scaling factors (empty slice if no scaling).
/// - `row_map`: per-stage [`CutRowMap`] to update.
/// - `batch_buf`: reusable [`RowBatch`] buffer for constructing the new cut rows.
///
/// # Panics
///
/// Panics if `total_nnz` exceeds `i32::MAX` (LP exceeds the `HiGHS` API limit).
/// In debug builds, also panics if `stage >= fcf.pools.len()`.
///
/// [`CutRowMap`]: crate::cut::CutRowMap
pub fn append_new_cuts_to_lp<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    stage: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    row_map: &mut crate::cut::CutRowMap,
    batch_buf: &mut RowBatch,
) -> usize {
    batch_buf.clear();

    let n_state = indexer.n_state;
    let mask = &indexer.nonzero_state_indices;
    let is_sparse = !mask.is_empty();
    let nnz_per_cut = if is_sparse {
        mask.len() + 1
    } else {
        n_state + 1
    };

    let mut new_count = 0usize;
    let mut nz_offset = 0usize;

    for (slot, intercept, coefficients) in fcf.active_cuts(stage) {
        // Skip cuts already present in the LP.
        if row_map.lp_row_for_slot(slot).is_some() {
            continue;
        }

        debug_assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length {got} != n_state {expected}",
            got = coefficients.len(),
            expected = n_state,
        );

        // Build the row using the shared cut-row constructor (negated-scaled
        // state coefficients + positive theta column + row bounds), the same
        // transformation as build_cut_row_batch_into.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        push_cut_row(batch_buf, intercept, coefficients, indexer, col_scale);

        row_map.insert(slot);
        new_count += 1;
        nz_offset += nnz_per_cut;
    }

    if new_count > 0 {
        let total_nnz = new_count * nnz_per_cut;
        #[allow(clippy::expect_used)]
        batch_buf.row_starts.push(
            i32::try_from(total_nnz)
                .expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
        );
        batch_buf.num_rows = new_count;
        solver.add_rows(batch_buf);
    }

    new_count
}

/// Append an explicit set of cut slots from a [`CutPool`] to a live solver.
///
/// The DCS analogue of [`append_new_cuts_to_lp`]: instead of "all active cuts
/// not yet resident", it adds exactly the slots in `slots`, skipping any that
/// are inactive or already resident in `row_map`. Each added row is built with
/// the shared [`push_cut_row`] constructor (identical layout to
/// [`append_new_cuts_to_lp`]) and recorded in `row_map`.
///
/// Returns the number of cut rows actually appended (`0` if none — `slots` was
/// empty or every slot was inactive/already resident, in which case no
/// `add_rows` call is made).
///
/// # Parameters
///
/// - `solver`: live LP solver with a loaded model.
/// - `pool`: the cut pool to read intercepts/coefficients/active flags from.
/// - `slots`: the slot ids to append, in caller order (the LP row order follows
///   this order for the appended subset).
/// - `indexer`: provides `n_state`, `theta`, and the state→column mapping.
/// - `col_scale`: column scaling factors (empty slice ⇒ no scaling).
/// - `row_map`: per-(stage, solve) [`CutRowMap`] to update.
/// - `batch_buf`: reusable [`RowBatch`] buffer.
///
/// # Panics
///
/// Panics if the total non-zero count exceeds `i32::MAX` (the `HiGHS` API
/// limit), matching [`append_new_cuts_to_lp`].
///
/// [`CutPool`]: crate::cut::CutPool
pub fn append_slots_to_lp<S: SolverInterface>(
    solver: &mut S,
    pool: &crate::cut::CutPool,
    slots: &[u32],
    indexer: &StageIndexer,
    col_scale: &[f64],
    row_map: &mut crate::cut::CutRowMap,
    batch_buf: &mut RowBatch,
) -> usize {
    batch_buf.clear();

    let n_state = indexer.n_state;
    let mask = &indexer.nonzero_state_indices;
    let is_sparse = !mask.is_empty();
    let nnz_per_cut = if is_sparse {
        mask.len() + 1
    } else {
        n_state + 1
    };

    let mut new_count = 0usize;
    let mut nz_offset = 0usize;

    for &slot in slots {
        let slot_usize = slot as usize;

        // Skip inactive cuts and cuts already resident in the LP.
        if slot_usize >= pool.populated_count
            || !pool.active[slot_usize]
            || row_map.lp_row_for_slot(slot_usize).is_some()
        {
            continue;
        }

        let intercept = pool.intercepts[slot_usize];
        let start = slot_usize * n_state;
        let coefficients = &pool.coefficients[start..start + n_state];

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        push_cut_row(batch_buf, intercept, coefficients, indexer, col_scale);

        row_map.insert(slot_usize);
        new_count += 1;
        nz_offset += nnz_per_cut;
    }

    if new_count > 0 {
        let total_nnz = new_count * nnz_per_cut;
        #[allow(clippy::expect_used)]
        batch_buf.row_starts.push(
            i32::try_from(total_nnz)
                .expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
        );
        batch_buf.num_rows = new_count;
        solver.add_rows(batch_buf);
    }

    new_count
}

#[cfg(test)]
mod tests {
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };

    use super::{append_new_cuts_to_lp, build_cut_row_batch, build_cut_row_batch_into};
    use crate::cut::FutureCostFunction;
    use crate::indexer::StageIndexer;

    // ── Unit tests: build_cut_row_batch ──────────────────────────────────────

    #[test]
    fn build_cut_row_batch_empty_cuts_returns_empty_batch() {
        let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let indexer = {
            let mut ix = StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.row_starts, vec![0]);
        assert!(batch.col_indices.is_empty());
        assert!(batch.values.is_empty());
        assert!(batch.row_lower.is_empty());
        assert!(batch.row_upper.is_empty());
    }

    #[test]
    fn build_cut_row_batch_one_cut_correct_structure() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 5.0, &[2.0]);
        let indexer = {
            let mut ix = StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_starts, vec![0, 2]);
        assert_eq!(batch.col_indices, vec![0, 3]); // theta at col N*(3+L) = 3
        assert_eq!(batch.values, vec![-2.0, 1.0]);
        assert_eq!(batch.row_lower, vec![5.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_two_cuts_correct_row_starts() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 3.0]);
        fcf.add_cut(1, 1, 0, 20.0, &[2.0, 4.0]);
        let indexer = {
            let mut ix = StageIndexer::new(1, 1);
            ix.finalize_for_test();
            ix
        };
        let batch = build_cut_row_batch(&fcf, 1, &indexer, &[]);

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts, vec![0, 3, 6]);
        assert_eq!(batch.col_indices[0], 0); // storage col 0
        assert_eq!(batch.col_indices[1], 2); // lag 0 → z_inflow col N*(1+L)=2
        assert_eq!(batch.col_indices[2], 4); // theta at N*(3+L) = 1*(3+1) = 4
        assert_eq!(batch.values[0], -1.0);
        assert_eq!(batch.values[1], -3.0);
        assert_eq!(batch.values[2], 1.0);
        assert_eq!(batch.col_indices[3], 0); // storage col 0
        assert_eq!(batch.col_indices[4], 2); // lag 0 → z_inflow col 2
        assert_eq!(batch.col_indices[5], 4); // theta at N*(3+L) = 4
        assert_eq!(batch.values[3], -2.0);
        assert_eq!(batch.values[4], -4.0);
        assert_eq!(batch.values[5], 1.0);
        assert_eq!(batch.row_lower, vec![10.0, 20.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_zero_coefficient_state_variable() {
        let mut fcf = FutureCostFunction::new(1, 2, 1, 5, &[0; 1]);
        fcf.add_cut(0, 0, 0, 3.0, &[0.0, 7.0]);
        let indexer = {
            let mut ix = StageIndexer::new(1, 1);
            ix.finalize_for_test();
            ix
        };
        let batch = build_cut_row_batch(&fcf, 0, &indexer, &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.col_indices, vec![0, 2, 4]); // lag 0 → z_inflow col 2; theta at 4
        assert_eq!(batch.values, vec![0.0, -7.0, 1.0]);
        assert_eq!(batch.row_lower, vec![3.0]);
    }

    // ── Tests for append_new_cuts_to_lp ─────────────────────────────────

    /// Mock solver that records the last `add_rows` call for verification.
    struct RecordingMockSolver {
        last_batch: Option<RowBatch>,
        add_rows_count: usize,
    }

    impl RecordingMockSolver {
        fn new() -> Self {
            Self {
                last_batch: None,
                add_rows_count: 0,
            }
        }
    }

    impl SolverInterface for RecordingMockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, cuts: &RowBatch) {
            self.last_batch = Some(RowBatch {
                num_rows: cuts.num_rows,
                row_starts: cuts.row_starts.clone(),
                col_indices: cuts.col_indices.clone(),
                values: cuts.values.clone(),
                row_lower: cuts.row_lower.clone(),
                row_upper: cuts.row_upper.clone(),
            });
            self.add_rows_count += 1;
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "not implemented for test".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }

        fn name(&self) -> &'static str {
            "RecordingMock"
        }
    }

    fn empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    #[test]
    fn append_new_cuts_returns_zero_when_no_new_cuts() {
        use crate::cut::CutRowMap;

        let fcf = crate::cut::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let indexer = {
            let mut ix = crate::indexer::StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        // No active cuts -> should return 0 and not call add_rows.
        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );
        assert_eq!(count, 0);
        assert_eq!(solver.add_rows_count, 0);
    }

    #[test]
    fn append_new_cuts_appends_all_on_empty_row_map() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::cut::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = {
            let mut ix = crate::indexer::StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        assert_eq!(count, 2);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert_eq!(row_map.lp_row_for_slot(0), Some(5));
        assert_eq!(row_map.lp_row_for_slot(1), Some(6));
    }

    #[test]
    fn append_new_cuts_skips_already_mapped_cuts() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::cut::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = {
            let mut ix = crate::indexer::StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        let mut row_map = CutRowMap::new(10, 5);
        // Pre-insert slot 0 as if it was already in the LP.
        row_map.insert(0);

        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        // Only slot 1 should be appended (slot 0 was already mapped).
        assert_eq!(count, 1);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert!(solver.last_batch.as_ref().is_some_and(|b| b.num_rows == 1));
    }

    #[test]
    fn append_new_cuts_matches_build_cut_row_batch_into() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::cut::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 20.0, &[3.0]); // slot 1

        let indexer = {
            let mut ix = crate::indexer::StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };

        // Build via build_cut_row_batch_into.
        let mut expected_batch = empty_row_batch();
        build_cut_row_batch_into(&mut expected_batch, &fcf, 0, &indexer, &[]);

        // Build via append_new_cuts_to_lp (empty row_map, so all cuts are new).
        let mut row_map = CutRowMap::new(10, 5);
        let mut actual_batch = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &[],
            &mut row_map,
            &mut actual_batch,
        );

        // The batch passed to add_rows must match build_cut_row_batch_into.
        assert_eq!(actual_batch.num_rows, expected_batch.num_rows);
        assert_eq!(actual_batch.row_starts, expected_batch.row_starts);
        assert_eq!(actual_batch.col_indices, expected_batch.col_indices);
        assert_eq!(actual_batch.values, expected_batch.values);
        assert_eq!(actual_batch.row_lower, expected_batch.row_lower);
        assert_eq!(actual_batch.row_upper, expected_batch.row_upper);
    }

    #[test]
    fn append_new_cuts_with_scaling_matches_build() {
        use crate::cut::CutRowMap;

        let mut fcf = crate::cut::FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 10.0, &[1.0]);

        let indexer = {
            let mut ix = crate::indexer::StageIndexer::new(1, 0);
            ix.finalize_for_test();
            ix
        };
        // col_scale must have at least theta+1 = 4 entries.
        let col_scale = vec![0.5, 2.0, 1.0, 0.1];

        let mut expected = empty_row_batch();
        build_cut_row_batch_into(&mut expected, &fcf, 0, &indexer, &col_scale);

        let mut row_map = CutRowMap::new(10, 5);
        let mut actual = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &indexer,
            &col_scale,
            &mut row_map,
            &mut actual,
        );

        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.col_indices, expected.col_indices);
    }
}
