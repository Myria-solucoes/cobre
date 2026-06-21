//! Delta-cut `RowBatch` construction for baked-template appends.
//!
//! Owns `build_delta_cut_row_batch_into`: fills a pre-allocated `RowBatch` with
//! only the cut rows generated in the current iteration, for appending to a
//! baked template via `add_rows`.

use cobre_solver::RowBatch;

use crate::cut::FutureCostFunction;
use crate::cut::row::push_scaled_coefficient;
use crate::indexer::StateLayout;

/// Fill a pre-allocated [`RowBatch`] with only the Benders cut rows generated
/// in `current_iteration`.
///
/// Clears `batch` and repopulates it with the subset of active cuts from
/// `fcf.pools[stage]` whose `iteration_generated` metadata field equals
/// `current_iteration`. Warm-start cuts (sentinel `iteration_generated ==
/// u64::MAX`) are always excluded.
///
/// Delta-cut variant of [`build_cut_row_batch_into`](crate::cut::row::build_cut_row_batch_into)
/// for use with baked templates.
///
/// When a baked template contains all cuts from previous iterations, this
/// function builds only the new cuts from `current_iteration` for appending
/// via `add_rows`. The CSR layout and coefficient transformation are identical
/// to [`build_cut_row_batch_into`](crate::cut::row::build_cut_row_batch_into);
/// when the pool contains only cuts from `current_iteration`, both functions
/// produce byte-identical output.
///
/// # Panics
///
/// Panics if total non-zeros exceeds `i32::MAX` (`HiGHS` API limit).
// Rationale: clippy::similar_names flags the role-(a) `state` handle next to the
// `stage` index; both are established names, so renaming either would obscure intent.
#[allow(clippy::similar_names)]
pub fn build_delta_cut_row_batch_into(
    batch: &mut RowBatch,
    fcf: &FutureCostFunction,
    stage: usize,
    state: &StateLayout,
    col_scale: &[f64],
    current_iteration: u64,
) {
    batch.clear();

    let n_state = state.n_state;
    let theta_col = state.theta;
    let mask = &state.nonzero_state_indices;

    // Count delta cuts with a lightweight scan to avoid double-iteration
    // overhead in the common case of zero delta cuts (early return).
    let num_cuts: usize = fcf.pools[stage]
        .active_delta_cuts(current_iteration)
        .count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    // NNZ per cut = nonzero state entries + theta. The mask is always finalized
    // (storage-only ⇒ full `[0, n_state)` range; pure-thermal ⇒ empty with
    // `n_state == 0`), so this matches the sparse-only authority
    // `cut::row::build_cut_row_batch_into`.
    let nnz_per_cut = mask.len() + 1;
    let total_nnz = num_cuts * nnz_per_cut;

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.pools[stage].active_delta_cuts(current_iteration) {
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
        // nonzero state indices in ascending order; padding slots are excluded
        // by `set_nonzero_mask` (it emits only `slot < k_i`), so this loop
        // never visits a padding slot.
        //
        // state_to_lp_column remaps outgoing-state indices to LP columns.
        // For storage (j < N) the mapping is identity. For lag dimensions
        // the outgoing state after shift_lag_state stores z_inflow at lag 0
        // and shifted incoming lags at lag 1+, so the cut must reference the
        // corresponding LP columns (z_inflow and incoming lag l−1).
        for &j in mask {
            let lp_col = state.state_to_lp_column(j);
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
