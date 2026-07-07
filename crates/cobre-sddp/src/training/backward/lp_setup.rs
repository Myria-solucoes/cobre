//! LP load and bound-patch seam for the backward pass: reset the stage LP to its
//! structural template, append delta cuts, and patch the noise-dependent
//! row/column bounds for one opening.

use cobre_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    lp_builder::AnticipatedGenWidenCtx,
    training::stage_solve_prep::{
        InflowNoise, LoadNoise, OpeningMode, StageSolvePrep, StageSolvePrepParams, StateSource,
    },
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorSpec;

/// Load the stage LP template and append delta cuts.
///
/// Resets the solver's retained basis, factorization, and RNG position so results
/// do not depend on the scenario-to-worker partition. The LP structure is
/// identical across a trial point's openings, so only bound patching runs per
/// opening.
pub(crate) fn load_backward_lp<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorSpec<'_>,
) {
    ws.solver.load_model(succ.frozen_template);
    if succ.cut_batch.num_rows > 0 {
        ws.solver.add_rows(succ.cut_batch);
    }
}

/// Transform opening noise and patch LP bounds for one backward opening.
///
/// The LP structure is already loaded by [`load_backward_lp`]; this delegates to
/// [`StageSolvePrep::run`], pinning `x_hat` as the incoming state.
pub(crate) fn patch_opening_bounds<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    raw_noise: &[f64],
    x_hat: &[f64],
    s: usize,
) {
    let widen_ctx = if training_ctx.state.n_anticipated > 0 {
        ctx.geometry_per_stage.get(s).map(|geom| {
            let template = &ctx.templates[s];
            AnticipatedGenWidenCtx {
                state_layout: training_ctx.state,
                state: x_hat,
                anticipated_thermal_indices: &training_ctx.study_dims.anticipated_thermal_indices,
                col_scale: &template.col_scale,
                col_lower: &template.col_lower,
                col_upper: &template.col_upper,
                thermal_col_start: geom.thermal.start,
                n_blks: ctx.block_counts_per_stage[s],
                stage_idx: s,
                n_stages: ctx.templates.len(),
            }
        })
    } else {
        None
    };
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(x_hat),
        opening_mode: OpeningMode::PerOpening,
        load_noise: LoadNoise::Present,
        inflow_noise: InflowNoise::Transform,
        widen_ctx,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        s,
        &prep_params,
    );
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
