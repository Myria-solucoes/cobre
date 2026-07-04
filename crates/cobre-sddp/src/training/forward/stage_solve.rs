//! Per-(scenario, stage) LP solve kernel for the forward pass.
//!
//! All scratch is owned by the workspace and reused across scenarios; no
//! allocation occurs on this hot path.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    dcs::{DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    error::SddpError,
    indexer::BlockGrid,
    lp_builder::COST_SCALE_FACTOR,
    noise::{NcsNoiseOffsets, transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    trajectory::TrajectoryRecord,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::{StageKey, write_capture_metadata};

/// Execute the stage-level LP solve for one (scenario, stage) pair.
///
/// Applies noise patches, warm-starts the solver, records the trajectory step,
/// and updates the current state and basis store for the next stage.
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
    } = *key;
    let n_hydros = ctx.n_hydros;
    let n_load_buses = ctx.n_load_buses;
    let study_dims = training_ctx.study_dims;
    let state = training_ctx.state;
    let stochastic = training_ctx.stochastic;
    let horizon = training_ctx.horizon;

    // DCS path: load the cut-free base template here (the caller skips its frozen
    // load). Loading the frozen template instead would make the lazy loop's fresh
    // CutRowMap double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(&ctx.templates[t]);
    }

    let (state_ref, scratch) = (&ws.current_state[..], &mut ws.scratch);
    transform_inflow_noise(raw_noise, t, state_ref, ctx, training_ctx, scratch);
    let blk = if n_load_buses > 0 {
        ctx.block_counts_per_stage[t]
    } else {
        0
    };
    transform_load_noise(
        raw_noise,
        n_hydros,
        n_load_buses,
        stochastic,
        t,
        blk,
        &mut ws.scratch.load_rhs_buf,
    );
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();
    if n_stochastic_ncs > 0 {
        transform_ncs_noise(
            raw_noise,
            &NcsNoiseOffsets {
                n_hydros,
                n_load_buses,
            },
            stochastic,
            t,
            ctx.block_counts_per_stage[t],
            ctx.ncs_max_gen,
            ctx.ncs_allow_curtailment,
            &mut ws.scratch.ncs_col_lower_buf,
            &mut ws.scratch.ncs_col_upper_buf,
        );
    }

    ws.patch_buf.fill_col_state_patches(
        training_ctx.state,
        &ws.current_state,
        &ctx.templates[t].col_scale,
    );
    ws.patch_buf.fill_forward_patches(
        state,
        &ws.current_state,
        &ws.scratch.noise_buf,
        ctx.base_rows[t],
        &ctx.templates[t].row_scale,
    );
    if n_load_buses > 0 {
        // Per-stage block count, NOT the global `indexer.n_blks` (the
        // nonuniform-block extraction trap). `max_deficit_segments` is
        // study-invariant, read from `study_dims`.
        let grid = BlockGrid::new(
            ctx.block_counts_per_stage[t],
            study_dims.max_deficit_segments,
        );
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[t],
            grid,
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[t].row_scale,
        );
    }
    // Per-stage `geometry.z_inflow_row_start`, always 0: state pinning uses
    // column bounds, so no rows precede the z-inflow block. Empty
    // `geometry_per_stage` (synthetic tests) falls back to 0.
    let z_inflow_row_start = ctx
        .geometry_per_stage
        .get(t)
        .map_or(0, |g| g.z_inflow_row_start);
    ws.patch_buf.fill_z_inflow_patches(
        z_inflow_row_start,
        &ws.scratch.z_inflow_rhs_buf,
        &ctx.templates[t].row_scale,
    );
    let cp = ws.patch_buf.state_col_patch_count();
    ws.solver.set_col_bounds(
        &ws.patch_buf.col_indices[..cp],
        &ws.patch_buf.col_lower[..cp],
        &ws.patch_buf.col_upper[..cp],
    );
    let pc = ws.patch_buf.forward_patch_count();
    ws.solver.set_row_bounds(
        &ws.patch_buf.indices[..pc],
        &ws.patch_buf.lower[..pc],
        &ws.patch_buf.upper[..pc],
    );
    // Patch NCS availability bounds onto this stage's dense NCS columns. The
    // gather forces `[0, 0]` for a slot whose `ncs_stochastic_windows[slot]`
    // excludes this stage's id, so a not-yet-commissioned source dispatches
    // nothing. The index buffer is constant across openings (rebuilt lazily on a
    // stage transition); the bounds change every scenario, gathered every solve.
    if n_stochastic_ncs > 0 && study_dims.has_ncs {
        let n_blks = ctx.block_counts_per_stage[t];
        let dense_col = ctx.ncs_stochastic_dense_col;
        let windows = ctx.ncs_stochastic_windows;
        // Stage id is the dormancy key (NOT the index `t`; filtered placeholder
        // stages can shift the id off the index).
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let stage_id = training_ctx
            .stages
            .get(t)
            .map_or(t as i32, |stage| stage.id);
        let ncs_col_start = ctx.ncs_col_starts[t];
        let expected_len = dense_col.len() * n_blks;
        // Rebuild the index buffer when the column **start** changes, not only its
        // length: two stages can share a length yet address different columns, so
        // keying on length alone would set bounds on the previous stage's columns.
        if ws.scratch.last_ncs_col_start != ncs_col_start
            || ws.scratch.ncs_col_indices_buf.len() != expected_len
        {
            crate::noise::build_dense_ncs_col_indices(
                dense_col,
                ncs_col_start,
                n_blks,
                &mut ws.scratch.ncs_col_indices_buf,
            );
            ws.scratch.last_ncs_col_start = ncs_col_start;
        }
        crate::noise::gather_dense_ncs_bounds(
            windows,
            stage_id,
            n_blks,
            &ws.scratch.ncs_col_lower_buf,
            &ws.scratch.ncs_col_upper_buf,
            &mut ws.scratch.ncs_col_lower_active_buf,
            &mut ws.scratch.ncs_col_upper_active_buf,
        );
        ws.solver.set_col_bounds(
            &ws.scratch.ncs_col_indices_buf,
            &ws.scratch.ncs_col_lower_active_buf,
            &ws.scratch.ncs_col_upper_active_buf,
        );
    }
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
        };
        lazy_solve_preloaded(
            &mut ws.solver,
            &ctx.templates[t],
            pool,
            state,
            &training_ctx.cut_state_layouts[t],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();
        let objective = view.objective;
        crate::stage_solve::fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        let _ = view;
        objective
    } else {
        let inputs = crate::stage_solve::StageInputs {
            stage_context: ctx,
            pool,
            stored_basis: basis_slice.get_mut(m, t).as_ref(),
            stage_index: t,
            scenario_index: m,
            iteration: Some(iteration),
        };

        let view = crate::stage_solve::run_stage_solve(ws, &inputs).map_err(|e| {
            // Invalidate the stored basis on Infeasible so the next warm-start
            // attempt cold-solves.
            if matches!(e, SddpError::Infeasible { .. }) {
                *basis_slice.get_mut(m, t) = None;
            }
            e
        })?;

        let objective = view.objective;
        crate::stage_solve::fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        let _ = view;
        objective
    };

    let d_t = ctx.discount_factors.get(t).copied().unwrap_or(1.0);
    let stage_cost = (view_objective - d_t * unscaled_primal[state.theta]) * COST_SCALE_FACTOR;
    let rec = &mut worker_records[local_m * num_stages + t];
    // Only rec.state is consumed downstream; the backward pass needs no primal
    // or dual here, and simulation reads them directly from the solver.
    rec.primal.clear();
    rec.dual.clear();
    rec.stage_cost = stage_cost;

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
    let stage_lag = ctx.stage_lag_transitions.get(t).copied().unwrap_or(
        cobre_core::temporal::StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        },
    );
    let downstream_par_order = ws
        .scratch
        .downstream_completed_lags
        .len()
        .checked_div(ws.scratch.lag_accumulator.len())
        .unwrap_or(0);
    crate::noise::accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        &unscaled_primal,
        state,
        &stage_lag,
        &mut crate::noise::LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut crate::noise::DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    crate::stage_solve::debug_assert_bucket_copy_gap_intact(
        &ws.current_state,
        &unscaled_primal,
        state,
    );
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
        if let Some(captured) = basis_slice.get_mut(m, t) {
            ws.solver.get_basis(&mut captured.basis);
            write_capture_metadata(
                captured,
                pool,
                ctx.templates[t].num_rows,
                cut_row_count,
                &ws.current_state[..state.n_state],
            );
        } else {
            let mut captured = CapturedBasis::new(
                ctx.templates[t].num_cols,
                basis_row_capacity,
                ctx.templates[t].num_rows,
                cut_row_count,
                state.n_state,
            );
            ws.solver.get_basis(&mut captured.basis);
            write_capture_metadata(
                &mut captured,
                pool,
                ctx.templates[t].num_rows,
                cut_row_count,
                &ws.current_state[..state.n_state],
            );
            *basis_slice.get_mut(m, t) = Some(captured);
        }
    }
    Ok(stage_cost)
}
