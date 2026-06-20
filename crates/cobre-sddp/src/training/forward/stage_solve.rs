//! Per-(scenario, stage) LP solve kernel for the forward pass.
//!
//! Owns `run_forward_stage`: applies the noise patches, warm-starts the solver,
//! records the trajectory step, and advances the current state and basis store.
//! All scratch is owned by the workspace and reused across scenarios;
//! `run_forward_stage` reuses the `ws.scratch.unscaled_primal` buffer via
//! `mem::take`/restore (taken out before the solve so it can be filled while the
//! `view` borrow of `ws` is live, then restored after the last read), the same
//! pattern as `simulation/pipeline.rs`.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    dcs::{DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    error::SddpError,
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
    let indexer = training_ctx.indexer;
    let stochastic = training_ctx.stochastic;
    let horizon = training_ctx.horizon;

    // On the DCS path, load the cut-free base structural template here (the
    // caller skips its baked load), so the bounds patch below applies to the
    // cut-free LP that `lazy_solve_preloaded` then grows with the resident cut
    // subset. Loading the baked template would make the lazy loop's fresh
    // CutRowMap double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(&ctx.templates[t]);
    }

    // Split borrows: current_state and scratch are distinct fields of ws.
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

    ws.patch_buf
        .fill_col_state_patches(indexer, &ws.current_state, &ctx.templates[t].col_scale);
    ws.patch_buf.fill_forward_patches(
        indexer,
        &ws.current_state,
        &ws.scratch.noise_buf,
        ctx.base_rows[t],
        &ctx.templates[t].row_scale,
    );
    if n_load_buses > 0 {
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[t],
            ctx.block_counts_per_stage[t],
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[t].row_scale,
        );
    }
    ws.patch_buf.fill_z_inflow_patches(
        indexer.z_inflow_row_start,
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
    // Patch NCS availability bounds onto this stage's **active** NCS columns,
    // indexed by `ncs_local` within `active_ncs_indices[t]` via the per-stage
    // `ctx.ncs_active_slot_to_local[t]` map — a stochastic NCS inactive at this
    // stage contributes no column and no bound. `transform_ncs_noise` above wrote
    // `ncs_col_{lower,upper}_buf` in full stochastic-slot order; the gather copies
    // only the active slots' bounds (parallel to the index buffer), since a subset
    // stage allocates only `(active stochastic NCS) * n_blks` columns and a full
    // `n_stochastic_ncs` block would overrun. The index buffer is constant across
    // openings, so it is rebuilt lazily on a stage transition (when the per-stage
    // NCS column start or the active-subset length changes); the bounds change
    // every scenario, so they are gathered every solve.
    if n_stochastic_ncs > 0 && indexer.has_ncs {
        let n_blks = ctx.block_counts_per_stage[t];
        let slot_to_local = &ctx.ncs_active_slot_to_local[t];
        let ncs_col_start = ctx.ncs_col_starts[t];
        let expected_len = crate::noise::active_ncs_slot_count(slot_to_local) * n_blks;
        // The index buffer must be rebuilt when the per-stage NCS column **start**
        // changes, not only when its length changes: `ncs_col_starts[t]` varies per
        // stage, so two stages can share `active_count * n_blks` yet address
        // different columns. Keying on length alone (the forbidden alternative)
        // would retain the previous stage's indices and set bounds on the wrong
        // columns.
        if ws.scratch.last_ncs_col_start != ncs_col_start
            || ws.scratch.ncs_col_indices_buf.len() != expected_len
        {
            crate::noise::build_active_ncs_col_indices(
                slot_to_local,
                ncs_col_start,
                n_blks,
                &mut ws.scratch.ncs_col_indices_buf,
            );
            ws.scratch.last_ncs_col_start = ncs_col_start;
        }
        crate::noise::gather_active_ncs_bounds(
            slot_to_local,
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
    // Zero out the theta column at the terminal stage (the last study stage,
    // `T-1`) so that the LP does not penalise future cost when there is no
    // successor.  The zeroing is skipped when boundary (warm-start) cuts have
    // been loaded for the terminal stage: in that case the cuts constrain
    // theta from below and their contribution to the objective must remain
    // visible to the solver.
    if horizon.is_terminal(t + 1) && !terminal_has_boundary_cuts {
        ws.solver.set_col_bounds(&[indexer.theta], &[0.0], &[0.0]);
    }

    let col_scale = &ctx.templates[t].col_scale;

    // Take the reusable scratch buffer out of ws before the solve borrows ws.
    // The `mem::take` pattern (capacity retained, empty vec left in field) lets
    // us fill unscaled_primal from `view` slices that are still tied to 'ws
    // while `&mut ws` is also live. Restored to ws.scratch after the last read
    // so the next stage reuses the warmed allocation (mirrors pipeline.rs).
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);

    // Solve and fill the unscaled primal buffer. The DCS branch solves the cut
    // pool lazily from the cut-free base loaded above (mirroring
    // `StageOpeningSolver::solve_lazy`, but extracting the primal rather than the
    // dual); the baked branch solves the already-loaded all-cuts LP via
    // `run_stage_solve`. Each branch fills `unscaled_primal` from `view` (tied to
    // `ws`) and captures the objective before ending the view borrow
    // (`SolutionView` is `Copy`).
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
        // Disjoint borrows of `ws`: `solver`, `backward_accum.dcs_initial_resident`
        // (shared), and `backward_accum.dcs_solve` (mut) are distinct fields.
        // `lazy_solve_preloaded` copies the final solve into `dcs_solve`'s result
        // buffers and returns `Result<()>`; the zero-cost view is rebuilt below.
        lazy_solve_preloaded(
            &mut ws.solver,
            &ctx.templates[t],
            pool,
            indexer,
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
            // Preserve today's behavior: invalidate the stored basis on
            // Infeasible so the next warm-start attempt cold-solves.
            if matches!(e, SddpError::Infeasible { .. }) {
                *basis_slice.get_mut(m, t) = None;
            }
            e
        })?;

        // Unscale primal values and capture the objective before the ws borrow
        // held by `view` ends its overlap with subsequent ws.scratch accesses.
        let objective = view.objective;
        crate::stage_solve::fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        // `view` is Copy, so end its borrow of ws before the mutable borrows below.
        let _ = view;
        objective
    };

    let d_t = ctx.discount_factors.get(t).copied().unwrap_or(1.0);
    let stage_cost = (view_objective - d_t * unscaled_primal[indexer.theta]) * COST_SCALE_FACTOR;
    let rec = &mut worker_records[local_m * num_stages + t];
    // Skip primal storage: only rec.state is consumed downstream; primals
    // are read directly from the solver when needed (simulation).
    rec.primal.clear();
    // Skip dual storage: rec.dual is not read by the backward pass or any
    // training-path code. Simulation reads duals directly from the solver view.
    rec.dual.clear();
    rec.stage_cost = stage_cost;

    // Save incoming lag values before overwriting state with primal.
    // Uses the pre-allocated lag_matrix_buf scratch buffer (no allocation).
    let lag_start = indexer.inflow_lags.start;
    let lag_len = indexer.hydro_count * indexer.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    // Save incoming anticipated-state slice before overwriting state with primal.
    // Uses the pre-allocated anticipated_state_buf scratch buffer (no allocation).
    let ant_start = indexer.anticipated_state.start;
    let ant_len = indexer.n_anticipated * indexer.k_max;
    ws.scratch.anticipated_state_buf.clear();
    ws.scratch
        .anticipated_state_buf
        .extend_from_slice(&ws.current_state[ant_start..ant_start + ant_len]);

    // Compute shifted lag state once into ws.current_state, then copy to rec.state.
    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&unscaled_primal[..indexer.n_state]);
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
        indexer,
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
    crate::noise::shift_anticipated_state(
        &mut ws.current_state,
        &ws.scratch.anticipated_state_buf,
        &unscaled_primal,
        indexer,
    );
    // Restore the scratch buffer so the next stage reuses the warmed allocation.
    // This is the last read of `unscaled_primal`.
    ws.scratch.unscaled_primal = unscaled_primal;
    rec.state.clear();
    rec.state.extend_from_slice(&ws.current_state);
    // Cross-iteration forward warm-start applies to the baked arm only: it
    // captures the post-solve basis here so the next iteration's forward solve
    // at this (m, t) slot can warm-start from it. The DCS arm makes no such
    // capture (it passes `stored_basis = None`) — a DCS-solve basis describes
    // the lazy resident-subset row layout, not the baked layout the warm-start
    // reconstruction expects, so the DCS path leaves the (m, t) slot untouched.
    if dcs.is_none() {
        let cut_row_count = basis_row_capacity.saturating_sub(ctx.templates[t].num_rows);
        if let Some(captured) = basis_slice.get_mut(m, t) {
            ws.solver.get_basis(&mut captured.basis);
            write_capture_metadata(
                captured,
                pool,
                ctx.templates[t].num_rows,
                cut_row_count,
                &ws.current_state[..indexer.n_state],
            );
        } else {
            let mut captured = CapturedBasis::new(
                ctx.templates[t].num_cols,
                basis_row_capacity,
                ctx.templates[t].num_rows,
                cut_row_count,
                indexer.n_state,
            );
            ws.solver.get_basis(&mut captured.basis);
            write_capture_metadata(
                &mut captured,
                pool,
                ctx.templates[t].num_rows,
                cut_row_count,
                &ws.current_state[..indexer.n_state],
            );
            *basis_slice.get_mut(m, t) = Some(captured);
        }
    }
    Ok(stage_cost)
}
