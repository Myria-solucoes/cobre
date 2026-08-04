//! Per-trial-point opening-solve dispatch and the deterministic by-scenario kernel.
//!
//! [`StageOpeningSolver`] is the closed two-variant per-opening solve strategy
//! (frozen all-cuts vs lazy resident-set DCS), and `process_by_scenario_backward`
//! drives it: it solves a trial point's openings in a run-constant, rank-invariant
//! `solve_order` permutation but writes and aggregates outcomes by canonical ω, so
//! the generated cut is bit-identical regardless of solve order.

use cobre_solver::SolverInterface;

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    risk_measure::RiskMeasure,
    setup::node_graph::OpeningSource,
    stage_solve::{StageInputs, run_stage_solve},
    state_exchange::ExchangeBuffers,
    workspace::{BasisStoreSliceMut, SolverWorkspace},
};

use super::{
    StagedCut, SuccessorSpec,
    duals_extraction::{extract_duals_from_view, extract_state_duals_only},
    lp_setup::{
        fill_external_opening_noise, load_backward_lp, patch_opening_bounds, resolve_backward_basis,
    },
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
/// - [`StageOpeningSolver::Frozen`]: the frozen all-cuts LP. Cross-iteration warm
///   basis on the first-solved opening, full state+cut dual extraction,
///   frozen-order slot bumps, and first-solved basis capture.
/// - [`StageOpeningSolver::Lazy`]: the resident-set LP (DCS). A cut-free core is
///   loaded once per trial point and reused across the openings; the first-solved
///   opening solves fresh, the rest warm-carry; state-dual-only extraction;
///   row-map-correct slot bumps; no basis capture.
pub(crate) enum StageOpeningSolver {
    /// Frozen all-cuts LP path (no DCS).
    Frozen,
    /// Lazy resident-set LP path (Dynamic Cut Selection). Carries the active
    /// [`DcsParams`] so the per-opening call needs no separate `Option` test.
    Lazy(DcsParams),
}

impl StageOpeningSolver {
    /// Choose the strategy from the already-`is_active`-filtered `dcs_params`:
    /// `Some` → [`StageOpeningSolver::Lazy`], `None` → [`StageOpeningSolver::Frozen`].
    pub(crate) fn from_dcs_params(dcs_params: Option<DcsParams>) -> Self {
        match dcs_params {
            Some(params) => StageOpeningSolver::Lazy(params),
            None => StageOpeningSolver::Frozen,
        }
    }

    /// Per-trial-point LP load, issued once after `reset_solver_state()` and before
    /// any opening solve; each variant owns its own load.
    ///
    /// - [`StageOpeningSolver::Frozen`]: load the frozen all-cuts LP via
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
            StageOpeningSolver::Frozen => {
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
    /// under the identity order, else the first entry of the solve order. The frozen
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
            StageOpeningSolver::Frozen => Self::solve_frozen(
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

    /// Frozen all-cuts per-opening solve: patch the opening bounds, reconstruct +
    /// solve, extract state and cut duals, accumulate the outcome (including the
    /// `slot_increments` update), and capture the first-solved opening's basis.
    // Rationale: see [`StageOpeningSolver::solve_opening`] — disjoint-borrow args
    // with no natural grouping.
    #[allow(clippy::too_many_arguments)]
    fn solve_frozen<S: SolverInterface + Send>(
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
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s)?;

        // Moved out before the solve to avoid a borrow conflict with `view`'s
        // lifetime; pre-warmed capacity is reused across openings.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
        let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);

        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        let stored_basis = if is_first {
            resolve_backward_basis(basis_slice, m, succ.successor_node)
        } else {
            None
        };
        let inputs = StageInputs {
            stage_context: ctx,
            pool: succ.successor_pool,
            stored_basis,
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
            node_id: succ.successor_node_id,
        };

        let view = run_stage_solve(ws, &inputs)?;

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
    /// would describe the frozen layout, not the DCS resident subset.
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
        // NOT `succ.frozen_template`: the frozen template already carries the active
        // cut rows, and loading it would make the lazy loop's fresh CutRowMap treat
        // those slots as non-resident and append them again (duplicate rows, broken
        // laziness).
        let core = &ctx.templates[s];
        let col_scale = &ctx.templates[s].col_scale;

        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s)?;

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
            node_id: succ.successor_node_id,
        };
        // The DCS LP renders `successor_pool`'s cuts into stage `s` (== successor);
        // its projection is the successor node's pool's, NOT `succ.cut_state`
        // (the pool `t`'s node resolves to, used only for the incoming
        // extraction below).
        let successor_pool_id = training_ctx.node_graph.nodes[succ.successor].pool_id;
        let successor_cut_layout = &training_ctx.cut_state_layouts[successor_pool_id];
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
/// On the frozen path the post-solve basis is written into `basis_slice` only at the
/// first-solved opening; later writes are forbidden (retained-LU corruption). The
/// DCS arm skips capture (see [`StageOpeningSolver::solve_lazy`]).
// RATIONALE: 12 args required — each is a disjoint borrow (ws, ctx, training_ctx, exchange,
// succ, basis_slice, opening_solver) or a plain scalar (fwd_offset, iteration, m, arena_offset)
// or a risk slice. Merging into a struct would add indirection without reducing the caller's
// borrow count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_by_scenario_backward<S: SolverInterface + Send>(
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
    compacted: usize,
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

    // Openings are SOLVED in `solve_order(s)` (a run-constant, rank-invariant
    // permutation) but written and aggregated by CANONICAL ω below, so the generated
    // cut is bit-identical regardless of solve order: the reorder changes only the
    // warm-start chain, never which cuts are produced. Defaults to the identity
    // permutation when no order is installed.
    let solve_order = tree_view.solve_order_data(s);
    // `succ.probabilities` flattens EVERY successor of the current node in
    // canonical order (ascending child id, then within-child ω —
    // `assemble_outcome_weights`), but `solve_order(s)`/`tree_view.opening(s,
    // _)` only understand a single successor's own within-stage index — so a
    // flattened `omega` is split into a successor block and a LOCAL within-
    // stage index below. A single-successor node has exactly one block, whose
    // local index equals `omega` unconditionally: chain parity depends on that
    // degeneracy, not on a separate code path.
    let n_local_openings = tree_view.n_openings(s);
    debug_assert_eq!(
        succ.probabilities.len() % n_local_openings,
        0,
        "process_by_scenario_backward: {} flattened outcomes must be an exact multiple of \
         stage {s}'s own {n_local_openings} openings",
        succ.probabilities.len()
    );
    debug_assert_eq!(
        solve_order.len(),
        n_local_openings,
        "solve_order(s) must be a permutation of 0..n_local_openings"
    );
    // First-solved opening: it owns the per-(m, s) basis load/capture and the fresh
    // (non-warm-carry) solve.
    let first = solve_order[0] as usize;

    // An External successor reads its declared column `eta_slice(s, k)` (assembled
    // once — invariant across the loop); a Generated successor keeps
    // `tree_view.opening` byte-for-byte. Match on OpeningSource, never a global
    // switch, so every sampled/generated study is unchanged.
    let mut external_noise: Option<Vec<f64>> = None;
    if training_ctx.node_graph.nodes[succ.successor_node]
        .openings
        .source
        == OpeningSource::External
    {
        let k = training_ctx.node_graph.nodes[succ.successor_node]
            .openings
            .offset;
        let mut buf = std::mem::take(&mut ws.backward_accum.external_noise_buf);
        fill_external_opening_noise(training_ctx, ctx, s, k, succ.successor_node_id, &mut buf)?;
        external_noise = Some(buf);
    }

    for omega_position in 0..succ.probabilities.len() {
        let block = omega_position / n_local_openings;
        let local_pos = omega_position % n_local_openings;
        let local_omega = solve_order[local_pos] as usize;
        let omega = block * n_local_openings + local_omega;

        let raw_noise: &[f64] = match &external_noise {
            Some(buf) => buf,
            None => tree_view.opening(s, local_omega),
        };
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
    }

    if let Some(buf) = external_noise {
        ws.backward_accum.external_noise_buf = buf;
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
    // The cut's `forward_pass_index` addresses `CutPool`'s per-iteration slot
    // block at stride `visit_bound[pool]` (the node's own routed visit count),
    // NOT `forward_passes`, so it must be the position within THIS node's routed
    // subset: `fwd_offset + compacted`. On a single-node level the subset is the
    // full global range and `compacted == m`, so this reduces to the global
    // `scenario` (byte-neutral, and cross-rank-disjoint on a chain). The global
    // `scenario` is the wrong-but-compiling alternative — on a fan its stride
    // overshoots and two trajectories on one node collide on the same slot.
    let node_relative_index = fwd_offset + compacted;
    debug_assert!(
        u32::try_from(node_relative_index).is_ok(),
        "node-relative forward-pass index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = node_relative_index as u32;
    let pop = ws.backward_accum.slot_increments.len();
    for slot in 0..pop {
        let count = ws.backward_accum.slot_increments[slot];
        if count > 0 {
            ws.backward_accum.metadata_sync_contribution[slot] += count;
        }
    }
    Ok(StagedCut {
        trial_state_idx: m,
        intercept: agg_intercept,
        coefficients_range,
        forward_pass_index,
    })
}
