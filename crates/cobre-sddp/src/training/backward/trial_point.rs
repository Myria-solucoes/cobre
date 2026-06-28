//! Per-trial-point opening-solve dispatch and the deterministic trial-point kernel.
//!
//! [`StageOpeningSolver`] is the closed two-variant per-opening solve strategy
//! (baked all-cuts vs lazy resident-set DCS), and `process_trial_point_backward`
//! drives it: it solves a trial point's openings in a run-constant, rank-invariant
//! `solve_order` permutation but writes and aggregates outcomes by canonical ω, so
//! the generated cut is bit-identical regardless of solve order.

use cobre_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    risk_measure::RiskMeasure,
    state_exchange::ExchangeBuffers,
    workspace::{BasisStoreSliceMut, SolverWorkspace},
};

use super::{
    StagedCut, SuccessorSpec,
    duals_extraction::{extract_duals_from_view, extract_state_duals_only},
    lp_setup::{load_backward_lp, patch_opening_bounds, resolve_backward_basis},
    outcome_aggregation::{
        accumulate_dcs_binding_counts, accumulate_opening_outcome, save_basis_at_omega_zero,
        write_opening_outcome,
    },
};

/// Per-trial-point opening-solve strategy for the backward pass.
///
/// A closed two-variant sum type dispatched by `match` — never a trait object
/// (CLAUDE.md closed-variant-set rule). The two variants are the two per-opening
/// solve paths:
///
/// - [`StageOpeningSolver::Baked`]: the baked all-cuts LP. Cross-iteration warm
///   basis on the first-solved opening, full state+cut dual extraction,
///   baked-order slot bumps, and first-solved basis capture.
/// - [`StageOpeningSolver::Lazy`]: the resident-set LP (DCS). A cut-free core is
///   loaded once per trial point and reused across the openings; the first-solved
///   opening solves fresh, the rest warm-carry; state-dual-only extraction;
///   row-map-correct slot bumps; no basis capture.
pub(crate) enum StageOpeningSolver {
    /// Baked all-cuts LP path (no DCS).
    Baked,
    /// Lazy resident-set LP path (Dynamic Cut Selection). Carries the active
    /// [`DcsParams`] so the per-opening call needs no separate `Option` test.
    Lazy(DcsParams),
}

impl StageOpeningSolver {
    /// Choose the strategy from the already-`is_active`-filtered `dcs_params`:
    /// `Some` → [`StageOpeningSolver::Lazy`], `None` → [`StageOpeningSolver::Baked`].
    pub(crate) fn from_dcs_params(dcs_params: Option<DcsParams>) -> Self {
        match dcs_params {
            Some(params) => StageOpeningSolver::Lazy(params),
            None => StageOpeningSolver::Baked,
        }
    }

    /// Per-trial-point LP load, issued once after `reset_solver_state()` and before
    /// any opening solve; each variant owns its own load.
    ///
    /// - [`StageOpeningSolver::Baked`]: load the baked all-cuts LP via
    ///   [`load_backward_lp`].
    /// - [`StageOpeningSolver::Lazy`]: load the cut-free core and build the metadata
    ///   seed ONCE here, then reuse the loaded LP across this trial point's
    ///   openings. This core load also serves as the per-trial-point reset that
    ///   keeps state from carrying across trial points (rank-invariance).
    pub(crate) fn prepare<S: SolverInterface + Send>(
        &self,
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        succ: &SuccessorSpec<'_>,
        iteration: u64,
    ) {
        match self {
            StageOpeningSolver::Baked => {
                load_backward_lp(ws, succ);
            }
            StageOpeningSolver::Lazy(params) => {
                ws.solver.load_model(&ctx.templates[succ.successor]);
                build_initial_resident_set(
                    succ.successor_pool,
                    iteration,
                    params.k2,
                    &mut ws.backward_accum.dcs_initial_resident,
                );
            }
        }
    }

    /// Solve one backward opening and accumulate its outcome, dispatching to the
    /// active variant's path.
    ///
    /// `is_first` is `true` for the trial point's **first-solved** opening — ω=0
    /// under the identity order, else the first entry of the solve order. The baked
    /// path loads and captures the per-(m, s) stored basis only on it; the lazy
    /// path passes `continue_carry == !is_first`. Decoupling basis identity from the
    /// literal ω=0 is what lets the openings be solved in any order while the
    /// per-(m, s) basis store stays consistent with the actual first solve.
    // Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ,
    // basis_slice) and per-opening scalars (raw_noise, x_hat, s, scenario,
    // iteration, m, omega, is_first); no natural grouping reduces caller-side
    // borrows.
    #[allow(clippy::too_many_arguments)]
    fn solve_opening<S: SolverInterface + Send>(
        &self,
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        basis_slice: &mut BasisStoreSliceMut<'_>,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: usize,
        scenario: usize,
        iteration: u64,
        m: usize,
        omega: usize,
        is_first: bool,
    ) -> Result<(), SddpError> {
        match self {
            StageOpeningSolver::Baked => Self::solve_baked(
                ws,
                ctx,
                training_ctx,
                succ,
                basis_slice,
                raw_noise,
                x_hat,
                s,
                scenario,
                iteration,
                m,
                omega,
                is_first,
            ),
            StageOpeningSolver::Lazy(params) => Self::solve_lazy(
                ws,
                ctx,
                training_ctx,
                succ,
                *params,
                raw_noise,
                x_hat,
                s,
                scenario,
                iteration,
                omega,
                !is_first,
            ),
        }
    }

    /// Baked all-cuts per-opening solve: patch the opening bounds, reconstruct +
    /// solve, extract state and cut duals, accumulate the outcome (including the
    /// `slot_increments` update), and capture the first-solved opening's basis.
    // Rationale: see [`StageOpeningSolver::solve_opening`] — disjoint-borrow args
    // with no natural grouping.
    #[allow(clippy::too_many_arguments)]
    fn solve_baked<S: SolverInterface + Send>(
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        basis_slice: &mut BasisStoreSliceMut<'_>,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: usize,
        scenario: usize,
        iteration: u64,
        m: usize,
        omega: usize,
        is_first: bool,
    ) -> Result<(), SddpError> {
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        // Moved out before the solve to avoid a borrow conflict with `view`'s
        // lifetime; pre-warmed capacity is reused across openings.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
        let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);

        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        let stored_basis = if is_first {
            resolve_backward_basis(basis_slice, m, s)
        } else {
            None
        };
        let inputs = crate::stage_solve::StageInputs {
            stage_context: ctx,
            pool: succ.successor_pool,
            stored_basis,
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
        };

        let view = crate::stage_solve::run_stage_solve(ws, &inputs)?;

        // Statistics must be captured after `view` is dropped (the `let _ = view`
        // below).
        let objective = extract_duals_from_view(
            &view,
            succ.cut_state,
            &ctx.templates[s].col_scale,
            succ,
            &mut state_duals,
            &mut cut_duals,
        );
        let _ = view;

        ws.backward_accum.state_duals_buf = state_duals;
        ws.backward_accum.cut_duals_buf = cut_duals;

        let mut stats_after_omega = std::mem::take(&mut ws.backward_accum.stats_after_buf);
        ws.solver.statistics_into(&mut stats_after_omega);

        accumulate_opening_outcome(
            ws,
            succ,
            omega,
            objective,
            x_hat,
            &stats_before_omega,
            &stats_after_omega,
        );

        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        if is_first {
            save_basis_at_omega_zero(ws, succ, basis_slice, m, x_hat);
        }

        Ok(())
    }

    /// Lazy resident-set per-opening solve under Dynamic Cut Selection.
    ///
    /// The cut-free core and metadata seed are loaded ONCE per trial point by
    /// [`StageOpeningSolver::prepare`]; this routine never reloads or re-seeds, only
    /// patches the opening's bounds and runs [`lazy_solve_preloaded`]:
    ///
    /// - `continue_carry == false` (first-solved opening): the lazy loop resets the
    ///   carried row map, appends the seed, and solves with `stored_basis = None`
    ///   (no warm per-opening basis, but reusing the loaded cut-free core).
    /// - `continue_carry == true` (subsequent openings): warm-carry the prior
    ///   opening's LP, basis, and monotonically grown resident cut set, adding only
    ///   the cuts this opening additionally violates (the paper's §3.4 base
    ///   recovery, extended across the openings).
    ///
    /// The gradient and intercept are read from the final all-satisfied LP via
    /// [`extract_state_duals_only`] in both cases. Binding-count metadata is
    /// maintained slot-correct under the lazy layout via
    /// [`accumulate_dcs_binding_counts`], feeding the per-stage
    /// `metadata_sync_contribution` allreduce without altering the gradient.
    ///
    /// First-solved basis capture is intentionally NOT performed: a captured basis
    /// would describe the baked layout, not the DCS resident subset.
    // Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ) and
    // per-opening scalars (params, raw_noise, x_hat, s, scenario, iteration,
    // omega); no natural grouping reduces caller-side borrows.
    #[allow(clippy::too_many_arguments)]
    fn solve_lazy<S: SolverInterface + Send>(
        ws: &mut SolverWorkspace<S>,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        succ: &SuccessorSpec<'_>,
        params: DcsParams,
        raw_noise: &[f64],
        x_hat: &[f64],
        s: usize,
        scenario: usize,
        iteration: u64,
        omega: usize,
        continue_carry: bool,
    ) -> Result<(), SddpError> {
        let state = training_ctx.state;
        // The DCS LP must start from the cut-free base template (`ctx.templates[s]`),
        // NOT `succ.baked_template`: the baked template already carries the active
        // cut rows, and loading it would make the lazy loop's fresh CutRowMap treat
        // those slots as non-resident and append them again (duplicate rows, broken
        // laziness).
        let core = &ctx.templates[s];
        let col_scale = &ctx.templates[s].col_scale;

        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        // Moved out so `extract_state_duals_only` can fill it while `view` holds an
        // immutable borrow of the sibling `dcs_solve` field; restored at the end.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);

        let dcs_ctx = DcsSolveContext {
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
            continue_carry,
        };
        // The DCS LP renders `successor_pool`'s cuts into stage `s` (== successor);
        // its projection is pool `successor`'s, NOT `succ.cut_state` (pool `t`'s,
        // used only for the incoming extraction below).
        let successor_cut_layout = &training_ctx.cut_state_layouts[succ.successor];
        lazy_solve_preloaded(
            &mut ws.solver,
            core,
            succ.successor_pool,
            state,
            successor_cut_layout,
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        let view = ws.backward_accum.dcs_solve.result_view();

        let objective =
            extract_state_duals_only(&view, succ.cut_state, col_scale, &mut state_duals);

        // `view` and `dcs_solve.row_map` both borrow `dcs_solve` immutably (so they
        // coexist); `slot_increments` is a distinct field borrowed mutably.
        accumulate_dcs_binding_counts(
            view.dual,
            &ws.backward_accum.dcs_solve.row_map,
            succ.successor_pool,
            succ.cut_activity_tolerance,
            &mut ws.backward_accum.slot_increments,
        );
        let _ = view;

        ws.backward_accum.state_duals_buf = state_duals;

        let mut stats_after_omega = std::mem::take(&mut ws.backward_accum.stats_after_buf);
        ws.solver.statistics_into(&mut stats_after_omega);

        write_opening_outcome(
            ws,
            omega,
            objective,
            x_hat,
            &stats_before_omega,
            &stats_after_omega,
        );

        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        Ok(())
    }
}

/// Process one trial point `m` in the backward pass, iterating over all openings.
///
/// On the baked path the post-solve basis is written into `basis_slice` only at the
/// first-solved opening; later writes are forbidden (retained-LU corruption). The
/// DCS arm skips capture (see [`StageOpeningSolver::solve_lazy`]).
// RATIONALE: 12 args required — each is a disjoint borrow (ws, ctx, training_ctx, exchange,
// succ, basis_slice, opening_solver) or a plain scalar (fwd_offset, iteration, m, arena_offset)
// or a risk slice. Merging into a struct would add indirection without reducing the caller's
// borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_trial_point_backward<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    opening_solver: &StageOpeningSolver,
    m: usize,
    arena_offset: usize,
) -> Result<StagedCut, SddpError> {
    let tree_view = training_ctx.stochastic.tree_view();
    let x_hat = exchange.state_at(succ.my_rank, m);
    let scenario = fwd_offset + m;
    let s = succ.successor;

    debug_assert_eq!(
        ws.backward_accum.per_opening_stats.len(),
        succ.probabilities.len(),
        "per_opening_stats must be initialised to n_openings before each stage's trial-point loop"
    );

    // Env-gated backward `noise_key` diagnostic; `None` on the default path.
    let noise_key_diag = training_ctx.noise_key_diag;

    // Openings are SOLVED in `solve_order(s)` (a run-constant, rank-invariant
    // permutation) but written and aggregated by CANONICAL ω below, so the generated
    // cut is bit-identical regardless of solve order: the reorder changes only the
    // warm-start chain, never which cuts are produced. Defaults to the identity
    // permutation when no order is installed.
    let solve_order = tree_view.solve_order_data(s);
    debug_assert_eq!(
        solve_order.len(),
        succ.probabilities.len(),
        "solve_order(s) must be a permutation of 0..n_openings"
    );
    // First-solved opening: it owns the per-(m, s) basis load/capture and the fresh
    // (non-warm-carry) solve.
    let first = solve_order[0] as usize;

    let mut omega_position = 0usize;
    while omega_position < succ.probabilities.len() {
        let omega = solve_order[omega_position] as usize;
        omega_position += 1;

        let raw_noise = tree_view.opening(s, omega);
        let is_first = omega == first;

        opening_solver.solve_opening(
            ws,
            ctx,
            training_ctx,
            succ,
            basis_slice,
            raw_noise,
            x_hat,
            s,
            scenario,
            iteration,
            m,
            omega,
            is_first,
        )?;

        if let Some(diag) = noise_key_diag {
            let simplex_iterations = ws.backward_accum.per_opening_stats[omega].simplex_iterations;
            let noise_key = diag.key(s, omega).unwrap_or(f64::NAN);
            eprintln!(
                "COBRE_W1_DIAG\tstage={s}\ttrial={scenario}\tomega={omega}\t\
                 noise_key={noise_key:.17e}\tsimplex_iterations={simplex_iterations}"
            );
        }
    }

    // Aggregate into `agg_coefficients`, then copy into this trial point's arena
    // slot so the bytes outlive the parallel closure without a per-cut allocation.
    let n_openings = succ.probabilities.len();
    let mut agg_intercept = 0.0_f64;
    risk_measures[succ.t].aggregate_cut_into(
        &ws.backward_accum.outcomes[..n_openings],
        succ.probabilities,
        &mut agg_intercept,
        &mut ws.backward_accum.agg_coefficients,
        &mut ws.backward_accum.risk_scratch,
    );
    let n_state = ws.backward_accum.agg_coefficients.len();
    let coefficients_range = arena_offset..arena_offset + n_state;
    debug_assert!(
        coefficients_range.end <= ws.backward_accum.agg_arena.len(),
        "agg_arena must be sized to cover this trial point's slot before the solve"
    );
    ws.backward_accum.agg_arena[coefficients_range.clone()]
        .copy_from_slice(&ws.backward_accum.agg_coefficients[..n_state]);
    debug_assert!(
        u32::try_from(scenario).is_ok(),
        "global scenario index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = scenario as u32;
    let pop = ws.backward_accum.slot_increments.len();
    for slot in 0..pop {
        let count = ws.backward_accum.slot_increments[slot];
        if count > 0 {
            ws.backward_accum.metadata_sync_contribution[slot] += count;
        }
    }
    Ok(StagedCut {
        trial_point_idx: m,
        intercept: agg_intercept,
        coefficients_range,
        forward_pass_index,
    })
}
