//! Benders subgradient extraction from a solved backward LP.
//!
//! The subgradient is `rc_scaled / col_scale[col]` — **divided, not multiplied** —
//! because the incoming-state column pin sets `v_scaled = v_orig / col_scale`
//! (sddp.md "Benders cut sign & subgradient extraction").

use cobre_solver::SolutionView;

use crate::indexer::StateLayout;

use super::SuccessorSpec;

/// Extract state and cut duals from the live solver view into pre-warmed scratch
/// buffers, returning the LP objective. `state_duals` holds the unscaled
/// incoming-state reduced costs (the module-level `col_scale`-division contract);
/// `cut_duals` holds the cut-row slice
/// `[template_num_rows, template_num_rows + num_cuts)`.
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

    // Unscale: the subgradient is rc_scaled / col_scale[col] — divided, not
    // multiplied (the pin sets v_scaled = v_orig / col_scale; see
    // fill_col_state_patches). Empty col_scale ⇒ raw rc.
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
/// `state_duals` with the unscaled incoming-state reduced costs.
///
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
