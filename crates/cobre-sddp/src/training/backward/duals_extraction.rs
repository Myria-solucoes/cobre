//! Benders subgradient extraction from a solved backward LP (`SEALED`).
//!
//! Owns the cut-sign / `col_scale`-division contract via
//! [`extract_duals_from_view`]: the subgradient is `rc_scaled / col_scale[col]` —
//! divided, not multiplied — because the incoming-state column pin sets
//! `v_scaled = v_orig / col_scale`. Both functions are `pub(crate)` so the
//! per-opening solve dispatch in [`super::StageOpeningSolver`] (`trial_point`)
//! can call them across the submodule boundary; their bodies and Contract
//! rustdoc move verbatim and must not be edited.

use cobre_solver::SolutionView;

use crate::indexer::StageIndexer;

use super::SuccessorSpec;

/// Extract state and cut duals from the solver view into pre-warmed scratch buffers.
///
/// Called while `view` is still live (borrowing the solver). The output buffers
/// (`state_duals`, `cut_duals`) were taken out of `ws.backward_accum` before the
/// solve and are passed here directly so that no `ws` borrow is needed.
///
/// Returns the LP objective value.
///
/// # Dual-fill layout
///
/// `state_duals`: unscaled reduced costs at the LP columns pinned by
/// `fill_col_state_patches`, one entry per state-vector index.
/// Scaling: `rc_original[j] = rc_scaled[j] / col_scale[col]` (the pin sets
/// `v_scaled = v_orig / col_scale`, so the subgradient w.r.t. the original
/// state divides by `col_scale`); when `col_scale` is empty the raw reduced
/// costs are used directly.
///
/// `cut_duals`: raw duals for cut rows `[template_num_rows, template_num_rows + num_cuts)`.
/// These always have implicit `row_scale = 1.0`.
pub(crate) fn extract_duals_from_view(
    view: &SolutionView<'_>,
    n_state: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    succ: &SuccessorSpec<'_>,
    state_duals: &mut Vec<f64>,
    cut_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    // Unscale state-fixing-column reduced costs from scaled to original units.
    // The incoming-state column is pinned in scaled space as `v_scaled =
    // v_orig / col_scale[col]` (see `fill_col_state_patches` and
    // `apply_col_scale`, which divides col bounds by `col_scale`). The reduced
    // cost HiGHS reports is `rc_scaled = dQ/dv_scaled`; the cut subgradient we
    // need is `dQ/dv_orig = rc_scaled * dv_scaled/dv_orig = rc_scaled /
    // col_scale[col]`. Dividing (not multiplying) keeps parity with the legacy
    // fixing-row dual, whose single-entry row carried `row_scale = 1/col_scale`.
    // `state_duals` carries pre-warmed capacity; `clear` + `push` reuses it.
    state_duals.clear();
    for j in 0..n_state {
        let col = indexer.state_to_lp_incoming_column(j);
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
    debug_assert_eq!(
        state_duals.len(),
        n_state,
        "state_duals must contain exactly n_state entries after fill"
    );

    // Fill cut duals from the cut-row slice.
    //
    // Layout: [0, template_num_rows) — structural rows;
    //         [template_num_rows, template_num_rows + num_cuts) — cut rows (baked then delta).
    cut_duals.clear();
    if succ.num_cuts_at_successor > 0 {
        cut_duals.extend_from_slice(
            &view.dual[succ.template_num_rows..succ.template_num_rows + succ.num_cuts_at_successor],
        );
    }

    objective
}

/// Extract only the cut gradient (state-fixing-column reduced costs) and return
/// the objective, for the lazy-solve path.
///
/// Identical to the state-dual half of [`extract_duals_from_view`]: it fills
/// `state_duals[j] = rc_scaled[col] / col_scale[col]` (raw reduced costs when
/// `col_scale` is empty) at the incoming-state column for each state index `j`,
/// where the negation into the `−∇·x + θ ≥ intercept` row happens later in
/// cut-row construction. The Benders gradient and intercept come solely from
/// these structural state columns, which are identical in the all-cuts LP and
/// the lazy-solve LP, so the resulting cut matches the all-cuts cut by
/// exactness.
///
/// Unlike [`extract_duals_from_view`], it does NOT read cut-row duals: under the
/// lazy-solve path the resident cut rows are a subset in row-map insertion
/// order, so the all-cuts cut-row→slot mapping does not apply. Binding-count
/// metadata and basis capture for that layout are handled separately and are
/// not driven from this function.
pub(crate) fn extract_state_duals_only(
    view: &SolutionView<'_>,
    n_state: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    state_duals.clear();
    for j in 0..n_state {
        let col = indexer.state_to_lp_incoming_column(j);
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
    debug_assert_eq!(
        state_duals.len(),
        n_state,
        "state_duals must contain exactly n_state entries after fill"
    );

    objective
}
