//! Shared stage-LP solve-preparation pipeline (`pin → row_patches → ncs_patch →
//! commit`): the single owner the forward pass, backward pass, simulation
//! pipeline, and lower-bound evaluation all route through, so every solve site
//! patches the LP identically (the "patch NCS identically" contract, D15).
//!
//! Home is `training/`, not `lp/builder/`: the pipeline needs `crate::context`,
//! `crate::workspace`, and `crate::noise`, which `training/` already depends on,
//! whereas `lp/builder/` sits below `training/` in the crate layering — hosting it
//! there would invert the dependency direction.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    indexer::BlockGrid,
    lp_builder::PatchBuffer,
    noise::{
        NcsNoiseOffsets, apply_ncs_col_bounds, transform_inflow_noise, transform_load_noise,
        transform_ncs_noise,
    },
    workspace::ScratchBuffers,
};

/// The state slice this stage solve pins as incoming state; the caller resolves
/// it per site (forward `current_state`, backward `x_hat`, lower-bound
/// `initial_state`, simulation `current_state`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateSource<'a>(pub &'a [f64]);

/// Whether the caller drives one realized stage solve (forward pass,
/// simulation) or a per-opening loop (backward pass, lower bound).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpeningMode {
    /// One realized noise draw per stage.
    SingleRealized,
    /// One [`StageSolvePrep::run`] call per opening in the caller's loop.
    PerOpening,
}

/// Whether this stage solve patches stochastic load-bus bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadNoise {
    /// Patch stochastic load-bus bounds (forward, backward, simulation).
    Present,
    /// Skip them: the lower bound has no load-bus noise dimension.
    Absent,
}

/// How the water-balance noise buffers this solve reads
/// (`scratch.noise_buf`, `scratch.z_inflow_rhs_buf`) are populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InflowNoise {
    /// Fill the buffers from `raw_noise` (forward, backward, simulation).
    Transform,
    /// The caller has already filled the buffers (the lower bound's PAR-batch
    /// precompute).
    PreBuilt,
}

/// Per-call variation points for [`StageSolvePrep::run`] — the divergences
/// among the four solve sites.
///
/// The NCS availability patch is not a variation point: [`StageSolvePrep::run`]
/// derives its own gate (`n_stochastic_ncs() > 0`, `has_ncs`) internally, the
/// same gate every solve site applies.
pub(crate) struct StageSolvePrepParams<'a> {
    /// Which slice this solve pins as incoming state.
    pub state_source: StateSource<'a>,
    /// One realized solve vs. a per-opening loop; see [`OpeningMode`].
    // Rationale: caller-declared intent only — `run` always prepares exactly
    // one solve regardless of the caller's loop shape, so the field is never
    // read internally.
    #[allow(dead_code)]
    pub opening_mode: OpeningMode,
    /// Whether to patch stochastic load-bus bounds.
    pub load_noise: LoadNoise,
    /// How the inflow-noise buffers are populated.
    pub inflow_noise: InflowNoise,
    /// This solve's realized noise draw, laid out `[hydro | load-bus | NCS]`.
    pub raw_noise: &'a [f64],
}

/// The single owner of the shared stage-LP solve-preparation pipeline.
pub(crate) struct StageSolvePrep;

impl StageSolvePrep {
    /// Executes `pin → row_patches → ncs_patch → commit` over caller-owned
    /// `&mut` scratch: allocates nothing.
    pub(crate) fn run<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: usize,
        params: &StageSolvePrepParams<'_>,
    ) where
        S: SolverInterface,
    {
        let pinned_state = params.state_source.0;

        if params.inflow_noise == InflowNoise::Transform {
            transform_inflow_noise(
                params.raw_noise,
                stage,
                pinned_state,
                ctx,
                training_ctx,
                scratch,
            );
        }

        let load_blocks = if ctx.n_load_buses > 0 {
            ctx.block_counts_per_stage[stage]
        } else {
            0
        };
        if params.load_noise == LoadNoise::Present {
            transform_load_noise(
                params.raw_noise,
                ctx.n_hydros,
                ctx.n_load_buses,
                training_ctx.stochastic,
                stage,
                load_blocks,
                &mut scratch.load_rhs_buf,
            );
        }

        patch_buf.fill_col_state_patches(
            training_ctx.state,
            pinned_state,
            &ctx.templates[stage].col_scale,
        );
        patch_buf.fill_forward_patches(
            training_ctx.state,
            pinned_state,
            &scratch.noise_buf,
            ctx.base_rows[stage],
            &ctx.templates[stage].row_scale,
        );
        if params.load_noise == LoadNoise::Present && ctx.n_load_buses > 0 {
            let grid = BlockGrid::new(load_blocks, training_ctx.study_dims.max_deficit_segments);
            patch_buf.fill_load_patches(
                ctx.load_balance_row_starts[stage],
                grid,
                &scratch.load_rhs_buf,
                ctx.load_bus_indices,
                &ctx.templates[stage].row_scale,
            );
        }
        let z_inflow_row_start = ctx
            .geometry_per_stage
            .get(stage)
            .map_or(0, |g| g.z_inflow_row_start);
        patch_buf.fill_z_inflow_patches(
            z_inflow_row_start,
            &scratch.z_inflow_rhs_buf,
            &ctx.templates[stage].row_scale,
        );

        let cp = patch_buf.state_col_patch_count();
        solver.set_col_bounds(
            &patch_buf.col_indices[..cp],
            &patch_buf.col_lower[..cp],
            &patch_buf.col_upper[..cp],
        );

        let pc = patch_buf.forward_patch_count();
        solver.set_row_bounds(
            &patch_buf.indices[..pc],
            &patch_buf.lower[..pc],
            &patch_buf.upper[..pc],
        );

        let n_stochastic_ncs = training_ctx.stochastic.n_stochastic_ncs();
        if n_stochastic_ncs > 0 {
            transform_ncs_noise(
                params.raw_noise,
                &NcsNoiseOffsets {
                    n_hydros: ctx.n_hydros,
                    n_load_buses: ctx.n_load_buses,
                },
                training_ctx.stochastic,
                stage,
                ctx.block_counts_per_stage[stage],
                ctx.ncs_max_gen,
                ctx.ncs_allow_curtailment,
                &mut scratch.ncs_col_lower_buf,
                &mut scratch.ncs_col_upper_buf,
            );
            if training_ctx.study_dims.has_ncs {
                // Stage id is the dormancy key (NOT the index `stage`; filtered
                // placeholder stages can shift the id off the index).
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let stage_id = training_ctx
                    .stages
                    .get(stage)
                    .map_or(stage as i32, |s| s.id);
                apply_ncs_col_bounds(
                    solver,
                    scratch,
                    ctx.ncs_col_starts[stage],
                    ctx.ncs_stochastic_dense_col,
                    ctx.ncs_stochastic_windows,
                    stage_id,
                    ctx.block_counts_per_stage[stage],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
