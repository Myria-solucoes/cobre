//! Per-(scenario, stage) LP solve kernel for the forward pass.
//!
//! All scratch is owned by the workspace and reused across scenarios; no
//! allocation occurs on this hot path.

use cobre_solver::SolverInterface;
use cobre_stochastic::par::resolve_stage_lag_transition;

use crate::{
    context::{StageContext, TrainingContext},
    dcs::{DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    error::SddpError,
    noise::{DownstreamAccumState, LagAccumState, accumulate_and_shift_lag_state},
    stage_solve::{
        StageInputs, debug_assert_bucket_copy_gap_intact, fill_unscaled, run_stage_solve,
    },
    training::stage_solve_prep::{
        InflowNoise, LoadNoise, OpeningMode, StageSolvePrep, StageSolvePrepParams, StateSource,
    },
    trajectory::TrajectoryRecord,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::{StageKey, write_capture_metadata};

/// Execute the stage-level LP solve for one (scenario, stage) pair.
///
/// Returns the stage cost on success, or propagates the solver error.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible)` when the stage LP is infeasible, or
/// `Err(SddpError::Solver)` for any other terminal solver failure.
// RATIONALE: covers the complete sequence of patch→solve→record→advance for one
// forward stage. The body is a single linear pipeline whose numerical work is
// already delegated to free helpers; the residual is orchestration glue threading
// disjoint partial borrows of one `&mut SolverWorkspace`. Any further extraction
// would pass either the whole workspace (no gain) or many borrowed fields
// (reintroducing too_many_arguments), without reducing complexity.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_forward_stage<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    key: &StageKey<'_>,
    worker_records: &mut [TrajectoryRecord],
) -> Result<f64, SddpError> {
    let StageKey {
        t,
        m,
        local_m,
        num_stages,
        iteration,
        raw_noise,
        basis_row_capacity,
        terminal_has_boundary_cuts,
        pool,
        dcs,
        node,
    } = *key;
    let node_graph = training_ctx.node_graph;
    let pool_id = node_graph.nodes[node].pool_id;
    let node_id = node_graph.node_ids[node];
    let state = training_ctx.state;
    let horizon = training_ctx.horizon;

    // DCS path: load the cut-free base template here (the caller skips its frozen
    // load). Loading the frozen template instead would make the lazy loop's fresh
    // CutRowMap double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(&ctx.templates[t]);
    }

    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        opening_mode: OpeningMode::SingleRealized,
        load_noise: LoadNoise::Present,
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        t,
        &prep_params,
    )?;
    // Zero theta at the terminal stage (no successor to penalise), but NOT when
    // boundary cuts are loaded — those constrain theta from below and must stay
    // visible in the objective.
    if horizon.is_terminal(t + 1) && !terminal_has_boundary_cuts {
        ws.solver.set_col_bounds(&[state.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.templates[t].col_scale;

    // `mem::take` the scratch buffer out before the solve borrows ws, so it can
    // be filled from `view` slices tied to ws while `&mut ws` is live; restored
    // after the last read so the next stage reuses the warmed allocation.
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);

    // DCS branch solves the cut pool lazily from the cut-free base loaded above
    // (extracting the primal, not the dual); frozen branch solves the all-cuts LP
    // via `run_stage_solve`.
    let view_objective: f64 = if let Some(params) = dcs {
        // Metadata-seeded initial resident subset (deterministic, rank-invariant).
        build_initial_resident_set(
            pool,
            iteration,
            params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
        let dcs_ctx = DcsSolveContext {
            stage_index: t,
            scenario_index: m,
            iteration: Some(iteration),
            // Forward pass solves one LP per (stage, scenario): always a fresh
            // solve, never a carried continuation.
            continue_carry: false,
            node_id,
        };
        lazy_solve_preloaded(
            &mut ws.solver,
            &ctx.templates[t],
            pool,
            state,
            &training_ctx.cut_state_layouts[pool_id],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();
        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        let _ = view;
        objective
    } else {
        let inputs = StageInputs {
            stage_context: ctx,
            pool,
            stored_basis: basis_slice.get_mut(m, t).as_ref(),
            stage_index: t,
            scenario_index: m,
            iteration: Some(iteration),
            node_id,
        };

        let view = run_stage_solve(ws, &inputs).map_err(|e| {
            // Invalidate the stored basis on Infeasible so the next warm-start
            // attempt cold-solves.
            if matches!(e, SddpError::Infeasible { .. }) {
                *basis_slice.get_mut(m, t) = None;
            }
            e
        })?;

        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        let _ = view;
        objective
    };

    let d_t = ctx.discount_factors.get(t).copied().unwrap_or(1.0);
    let stage_cost = (view_objective - d_t * unscaled_primal[state.theta]) * ctx.cost_scale_factor;
    let rec = &mut worker_records[local_m * num_stages + t];
    // rec.primal/dual stay empty: only state and node_id feed downstream
    // consumers (the backward pass reads state; node_id tags the visit for
    // per-node output) — simulation reads primal/dual directly from the solver.
    rec.primal.clear();
    rec.dual.clear();
    rec.stage_cost = stage_cost;
    rec.node_id = node_id;

    // Save incoming lag values before overwriting state with primal.
    let lag_start = state.inflow_lags.start;
    let lag_len = state.hydro_count * state.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&unscaled_primal[..state.n_state]);
    let stage_lag = resolve_stage_lag_transition(ctx.stage_lag_transitions, t);
    let downstream_par_order = ws
        .scratch
        .downstream_completed_lags
        .len()
        .checked_div(ws.scratch.lag_accumulator.len())
        .unwrap_or(0);
    accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        &unscaled_primal,
        state,
        &stage_lag,
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    debug_assert_bucket_copy_gap_intact(&ws.current_state, &unscaled_primal, state);
    // Last read of `unscaled_primal`; restore it so the next stage reuses the
    // warmed allocation.
    ws.scratch.unscaled_primal = unscaled_primal;
    rec.state.clear();
    rec.state.extend_from_slice(&ws.current_state);
    // Capture the post-solve basis for next iteration's warm-start — frozen arm
    // ONLY. A DCS-solve basis describes the lazy resident-subset row layout, not
    // the frozen layout the warm-start reconstruction expects, so capturing it
    // would corrupt the warm-start; the DCS path leaves the (m, t) slot untouched.
    if dcs.is_none() {
        let cut_row_count = basis_row_capacity.saturating_sub(ctx.templates[t].num_rows);
        let captured = basis_slice.get_mut(m, t).get_or_insert_with(|| {
            CapturedBasis::new(
                ctx.templates[t].num_cols,
                basis_row_capacity,
                ctx.templates[t].num_rows,
                cut_row_count,
                state.n_state,
                node_id,
            )
        });
        ws.solver.get_basis(&mut captured.basis);
        write_capture_metadata(
            captured,
            pool,
            ctx.templates[t].num_rows,
            cut_row_count,
            &ws.current_state[..state.n_state],
            node_id,
        );
    }
    Ok(stage_cost)
}
