//! LP load and bound-patch seam for the backward pass.
//!
//! Invoked at the top of each trial-point iteration (and per opening) to reset
//! the stage LP to its structural template, append delta cuts, and patch the
//! noise-dependent row/column bounds for one opening. The functions are
//! `pub(crate)` because the per-opening solve dispatch in
//! [`super::StageOpeningSolver`] (`trial_point`) drives them across the
//! submodule boundary.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    indexer::BlockGrid,
    noise::{NcsNoiseOffsets, transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorSpec;

/// Load the stage LP template and append delta cuts.
///
/// Called at the top of every trial-point iteration in `process_stage_backward`
/// to reset `HiGHS`'s retained simplex basis, factorization, and RNG position so
/// that results do not depend on the scenario-to-worker partition. Within a
/// trial point the LP structure is identical across openings — only the
/// noise-dependent bounds change, so only bound patching happens per opening.
pub(crate) fn load_backward_lp<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorSpec<'_>,
) {
    ws.solver.load_model(succ.baked_template);
    if succ.cut_batch.num_rows > 0 {
        ws.solver.add_rows(succ.cut_batch);
    }
}

/// Transform opening noise and patch LP bounds for one backward opening.
///
/// Called once per opening inside [`process_trial_point_backward`](super::process_trial_point_backward).  The LP
/// structure is already loaded by [`load_backward_lp`]; this function only
/// updates noise-dependent row and column bounds via `set_row_bounds` /
/// `set_col_bounds`.
pub(crate) fn patch_opening_bounds<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    raw_noise: &[f64],
    x_hat: &[f64],
    s: usize,
) {
    let n_blks = if ctx.n_load_buses > 0 {
        ctx.block_counts_per_stage[s]
    } else {
        0
    };
    transform_inflow_noise(raw_noise, s, x_hat, ctx, training_ctx, &mut ws.scratch);
    transform_load_noise(
        raw_noise,
        ctx.n_hydros,
        ctx.n_load_buses,
        training_ctx.stochastic,
        s,
        n_blks,
        &mut ws.scratch.load_rhs_buf,
    );
    let n_stochastic_ncs = training_ctx.stochastic.n_stochastic_ncs();
    if n_stochastic_ncs > 0 {
        transform_ncs_noise(
            raw_noise,
            &NcsNoiseOffsets {
                n_hydros: ctx.n_hydros,
                n_load_buses: ctx.n_load_buses,
            },
            training_ctx.stochastic,
            s,
            ctx.block_counts_per_stage[s],
            ctx.ncs_max_gen,
            ctx.ncs_allow_curtailment,
            &mut ws.scratch.ncs_col_lower_buf,
            &mut ws.scratch.ncs_col_upper_buf,
        );
    }
    // No shift_anticipated_state call here: the backward pass solves each
    // opening at a fixed trial point produced by the forward sampler. The
    // ring-buffer advance happens once in the forward pass; the backward
    // and simulation paths reuse those slot values without re-shifting.
    ws.patch_buf
        .fill_col_state_patches(training_ctx.state, x_hat, &ctx.templates[s].col_scale);
    ws.patch_buf.fill_forward_patches(
        training_ctx.state,
        x_hat,
        &ws.scratch.noise_buf,
        ctx.base_rows[s],
        &ctx.templates[s].row_scale,
    );
    if ctx.n_load_buses > 0 {
        // Per-stage grid: it must carry this stage's block count, not the global
        // `indexer.n_blks`. `max_deficit_segments` is study-invariant, so it reads
        // from the single `study_dims` owner, not the global stage-0 `indexer`.
        let grid = BlockGrid::new(n_blks, training_ctx.study_dims.max_deficit_segments);
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[s],
            grid,
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[s].row_scale,
        );
    }
    // z-inflow rows start at the per-stage `geometry.z_inflow_row_start` (always 0:
    // state pinning uses column bounds, so no rows precede the z-inflow block),
    // read from the per-stage geometry table rather than the global stage-0
    // `indexer`. Empty `geometry_per_stage` in a synthetic test falls back to the
    // all-empty `StageGeometry::default` (also 0), matching the sibling per-stage
    // slices.
    let z_inflow_row_start = ctx
        .geometry_per_stage
        .get(s)
        .map_or(0, |g| g.z_inflow_row_start);
    ws.patch_buf.fill_z_inflow_patches(
        z_inflow_row_start,
        &ws.scratch.z_inflow_rhs_buf,
        &ctx.templates[s].row_scale,
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
    // Patch NCS availability bounds onto this stage's dense NCS columns, indexed
    // by `ncs_stochastic_dense_col[slot]` (the slot's NCS system index — the dense
    // column position, identical at every stage). `transform_ncs_noise` above wrote
    // the bound buffers in full stochastic-slot order; the gather copies every
    // slot's bounds, forcing `[0, 0]` for a slot whose
    // `ncs_stochastic_windows[slot]` excludes this stage's id (dormancy computed
    // inline by the gather). This is the same zeroing the forward and lower-bound
    // patch sites apply — keeping the three identical is the
    // `.claude/rules/sddp.md` "patch NCS identically" contract (a divergence
    // understates the bound). Index buffer rebuilt lazily on a stage transition
    // (when the per-stage NCS column start changes); bounds gathered every opening.
    if n_stochastic_ncs > 0 && training_ctx.study_dims.has_ncs {
        let n_blks_stage = ctx.block_counts_per_stage[s];
        let dense_col = ctx.ncs_stochastic_dense_col;
        let windows = ctx.ncs_stochastic_windows;
        // Stage id is the dormancy key (NOT the index `s`; filtered placeholder
        // stages can shift the id off the index).
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let stage_id = training_ctx
            .stages
            .get(s)
            .map_or(s as i32, |stage| stage.id);
        let ncs_col_start = ctx.ncs_col_starts[s];
        let expected_len = dense_col.len() * n_blks_stage;
        // The index buffer must be rebuilt when the per-stage NCS column **start**
        // changes, not only when its length changes: `ncs_col_starts[s]` varies per
        // stage, so two stages can share the same length yet address different
        // columns. Keying on length alone (the forbidden alternative) would retain
        // the previous stage's indices and set bounds on the wrong columns.
        if ws.scratch.last_ncs_col_start != ncs_col_start
            || ws.scratch.ncs_col_indices_buf.len() != expected_len
        {
            crate::noise::build_dense_ncs_col_indices(
                dense_col,
                ncs_col_start,
                n_blks_stage,
                &mut ws.scratch.ncs_col_indices_buf,
            );
            ws.scratch.last_ncs_col_start = ncs_col_start;
        }
        crate::noise::gather_dense_ncs_bounds(
            windows,
            stage_id,
            n_blks_stage,
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
}

/// Resolve the ω=0 warm-start basis from the worker's `BasisStoreSliceMut`.
///
/// Returns `None` when the slot is empty (cold start or no prior capture).
#[inline]
pub(crate) fn resolve_backward_basis<'a>(
    basis_slice: &'a BasisStoreSliceMut<'_>,
    m: usize,
    s: usize,
) -> Option<&'a CapturedBasis> {
    basis_slice.get(m, s)
}
