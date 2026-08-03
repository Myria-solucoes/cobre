//! Replicated-aggregation slice solve for a singleton-trial-state group: ranks
//! split the successor outcome set, each solves its opening slice, and the
//! outcomes are allgathered so every rank runs the identical flat aggregation and
//! appends the identical cut.
//!
//! Reserved: inert until the forward pass emits a DEDUPED trial-state count
//! distinct from the scenario count — the sampled walk resolves a node per
//! trajectory, but no per-node trial-state count exists yet, so no
//! dedup/recombining node is trainable end-to-end. The orchestrator and its
//! activation condition live at `run_backward_node_replicated`
//! (`training::backward_pass_state`); the aggregation identity is pinned by the
//! backward-pass unit tests.

use cobre_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    stage_solve::{StageInputs, run_stage_solve},
    state_exchange::ExchangeBuffers,
    workspace::SolverWorkspace,
};

use super::{
    SuccessorSpec, duals_extraction::extract_state_duals_only, lp_setup::patch_opening_bounds,
};

/// Per-outcome payload width in [`solve_replicated_outcome_slice`]'s output:
/// one objective followed by the `n_state` subgradient entries.
///
/// Reserved seam — see the module doc.
#[must_use]
#[allow(dead_code)]
pub(crate) fn outcome_stride(n_state: usize) -> usize {
    1 + n_state
}

/// Solve trial point `m`'s successor-opening slice `[o_start, o_end)` in
/// canonical ω order, appending each opening's `objective ‖ subgradient`
/// (`1 + cut_state.n_slots()` doubles) to `out`. The frozen LP loads once; the
/// duals feed the replicated `allgather_outcomes` + `aggregate_cut_into` the
/// caller runs over the whole set on every rank.
///
/// Openings are solved in canonical order (not the warm-start `solve_order`):
/// aggregation is over canonical `(m, ψ)`, and the warm chain is a
/// performance-only concern this inert path does not optimize.
///
/// Reserved seam — see the module doc.
///
/// # Errors
///
/// Propagates `SddpError::Infeasible` / `SddpError::Solver` from the slice's
/// solves and `SddpError::AnticipatedCommitmentOutOfBounds` from the
/// bound-patch.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn solve_replicated_outcome_slice<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    exchange: &ExchangeBuffers,
    succ: &SuccessorSpec<'_>,
    m: usize,
    o_start: usize,
    o_end: usize,
    scenario_base: usize,
    iteration: u64,
    out: &mut Vec<f64>,
) -> Result<(), SddpError> {
    let s = succ.successor;
    let n_state = succ.cut_state.n_slots();
    let x_hat = exchange.state_at(succ.my_rank, m);
    let tree_view = training_ctx.stochastic.tree_view();

    ws.solver.reset_solver_state();
    super::lp_setup::load_backward_lp(ws, succ);

    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
    for omega in o_start..o_end {
        let raw_noise = tree_view.opening(s, omega);
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s)?;
        let inputs = StageInputs {
            stage_context: ctx,
            pool: succ.successor_pool,
            stored_basis: None,
            stage_index: s,
            scenario_index: scenario_base + m,
            iteration: Some(iteration),
            node_id: succ.successor_node_id,
        };
        let view = run_stage_solve(ws, &inputs)?;
        let objective = extract_state_duals_only(
            &view,
            succ.cut_state,
            &ctx.templates[s].col_scale,
            &mut state_duals,
        );
        let _ = view;
        out.push(objective);
        out.extend_from_slice(&state_duals[..n_state]);
    }
    ws.backward_accum.state_duals_buf = state_duals;
    Ok(())
}
