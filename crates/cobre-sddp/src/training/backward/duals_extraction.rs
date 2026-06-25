//! Benders subgradient extraction from a solved backward LP.
//!
//! Owns the cut-sign / `col_scale`-division contract ([`extract_duals_from_view`]):
//! the subgradient is `rc_scaled / col_scale[col]` — **divided, not multiplied** —
//! because the incoming-state column pin sets `v_scaled = v_orig / col_scale`.
//! Both functions are `pub(crate)` for the cross-submodule call from
//! [`super::StageOpeningSolver`].

use cobre_solver::SolutionView;

use crate::indexer::StateLayout;

use super::SuccessorSpec;

/// Extract state and cut duals from the live solver view into pre-warmed scratch
/// buffers (`state_duals`, `cut_duals` — taken from `ws.backward_accum` before the
/// solve so no `ws` borrow is held here). Returns the LP objective.
///
/// `state_duals` holds the unscaled incoming-state reduced costs (the module-level
/// `col_scale`-division contract); `cut_duals` holds the cut-row slice
/// `[template_num_rows, template_num_rows + num_cuts)` (implicit `row_scale = 1`).
pub(crate) fn extract_duals_from_view(
    view: &SolutionView<'_>,
    n_state: usize,
    state: &StateLayout,
    col_scale: &[f64],
    succ: &SuccessorSpec<'_>,
    state_duals: &mut Vec<f64>,
    cut_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    // Unscale to original units: the pin sets v_scaled = v_orig / col_scale[col]
    // (see fill_col_state_patches), so the subgradient dQ/dv_orig is
    // rc_scaled / col_scale[col] — divided, not multiplied. (col_scale empty ⇒ raw rc.)
    state_duals.clear();
    for j in 0..n_state {
        let col = state.state_to_lp_incoming_column(j);
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

    cut_duals.clear();
    if succ.num_cuts_at_successor > 0 {
        cut_duals.extend_from_slice(
            &view.dual[succ.template_num_rows..succ.template_num_rows + succ.num_cuts_at_successor],
        );
    }

    objective
}

/// State-dual half of [`extract_duals_from_view`] for the lazy-solve path: fills
/// `state_duals` with the unscaled incoming-state reduced costs (the negation into
/// the `−∇·x + θ ≥ intercept` row happens later, in cut-row construction).
///
/// The Benders gradient comes solely from the structural state columns, which are
/// identical in the all-cuts and lazy-solve LPs, so the cut matches by exactness.
/// Unlike [`extract_duals_from_view`] it does NOT read cut-row duals — under
/// lazy-solve the resident cut rows are an insertion-order subset, so the
/// cut-row→slot mapping does not apply.
pub(crate) fn extract_state_duals_only(
    view: &SolutionView<'_>,
    n_state: usize,
    state: &StateLayout,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    state_duals.clear();
    for j in 0..n_state {
        let col = state.state_to_lp_incoming_column(j);
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
