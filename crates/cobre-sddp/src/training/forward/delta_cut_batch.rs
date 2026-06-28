//! Delta-cut `RowBatch` construction for baked-template appends.

use cobre_solver::RowBatch;

use crate::cut::FutureCostFunction;
use crate::cut::row::push_scaled_coefficient;
use crate::indexer::{CutStateProjection, StateLayout};

/// Fill a pre-allocated [`RowBatch`] with only the Benders cut rows generated
/// in `current_iteration`, for appending to a baked template via `add_rows`.
///
/// Warm-start cuts (sentinel `iteration_generated == u64::MAX`) are always
/// excluded. The CSR layout and coefficient transformation mirror
/// [`build_cut_row_batch_into`](crate::cut::row::build_cut_row_batch_into); when
/// the pool holds only `current_iteration` cuts the two produce byte-identical
/// output. `cut_state` is pool `stage`'s projection; `coefficients` has length
/// `cut_state.n_state()`.
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
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    current_iteration: u64,
) {
    batch.clear();

    let n_cut_state = cut_state.n_state();
    let theta_col = state.theta;

    // Count first: cheap scan that early-returns on the common zero-delta case
    // before the heavier coefficient loop.
    let num_cuts: usize = fcf.pools[stage]
        .active_delta_cuts(current_iteration)
        .count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    // NNZ per cut = nonzero state entries + theta, matching the sparse-only
    // authority `cut::row::build_cut_row_batch_into`.
    let nnz_per_cut = cut_state.render_len() + 1;
    let total_nnz = num_cuts * nnz_per_cut;

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.pools[stage].active_delta_cuts(current_iteration) {
        debug_assert_eq!(
            coefficients.len(),
            n_cut_state,
            "cut coefficients length {got} != pool n_state {expected}",
            got = coefficients.len(),
            expected = n_cut_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.row_starts.push(nz_offset as i32);

        // render_pairs maps each enabled non-padding reduced index to its outgoing
        // LP column: identity for storage; for lag dimensions the outgoing state
        // stores z_inflow at lag 0 and shifted incoming lags at lag 1+, so the cut
        // references z_inflow and incoming lag l−1, not the outgoing slot.
        for (j, lp_col) in cut_state.render_pairs() {
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
