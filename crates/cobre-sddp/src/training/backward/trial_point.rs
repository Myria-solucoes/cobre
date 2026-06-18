//! Per-trial-point opening-solve dispatch and the deterministic trial-point kernel.
//!
//! [`StageOpeningSolver`] is the closed two-variant per-opening solve strategy
//! (baked all-cuts vs lazy resident-set DCS), and `process_trial_point_backward`
//! is the deterministic hot-path kernel that drives it: it solves a trial point's
//! openings in a run-constant, rank-invariant `solve_order` permutation but writes
//! and aggregates outcomes by canonical ω, so the generated cut is bit-identical
//! regardless of solve order. The two co-locate because the kernel calls
//! `opening_solver.solve_opening(...)` and [`StageOpeningSolver::prepare`]; both
//! are hot-path behaviour-sensitive and move verbatim.

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
/// Constructed once per trial point from the Dynamic Cut Selection (DCS)
/// activation decision and dispatches each opening's solve through one uniform
/// call site ([`StageOpeningSolver::solve_opening`]), with a per-trial-point LP
/// load step ([`StageOpeningSolver::prepare`]) issued once before the openings
/// run. This is a closed two-variant sum type dispatched by `match` — never a
/// trait object (CLAUDE.md closed-variant-set rule), mirroring the
/// [`crate::cut_selection::CutSelectionStrategy`] / [`DcsParams::from_strategy`]
/// pattern already used in the crate.
///
/// The two variants are the two pre-existing, drifted per-opening solve paths:
///
/// - [`StageOpeningSolver::Baked`]: the baked all-cuts LP. Cross-iteration warm
///   basis on the first-solved opening, full state+cut dual extraction,
///   baked-order slot bumps, and first-solved basis capture.
/// - [`StageOpeningSolver::Lazy`]: the resident-set LP (DCS). A cut-free core is
///   loaded once per trial point and reused across the openings; the first-solved
///   opening solves fresh, the rest warm-carry; state-dual-only extraction;
///   row-map-correct slot bumps; no basis capture.
pub(crate) enum StageOpeningSolver {
    /// Baked all-cuts LP path (no DCS). See the type-level docs for the
    /// five behavioral axes.
    Baked,
    /// Lazy resident-set LP path (Dynamic Cut Selection). Carries the active
    /// [`DcsParams`] so the per-opening call needs no separate `Option` test.
    Lazy(DcsParams),
}

impl StageOpeningSolver {
    /// Choose the per-trial-point strategy from the already-`is_active`-filtered
    /// `dcs_params`: `Some(params)` selects [`StageOpeningSolver::Lazy`], `None`
    /// selects [`StageOpeningSolver::Baked`]. The decision is constant across a
    /// stage's trial points and openings for a given iteration.
    pub(crate) fn from_dcs_params(dcs_params: Option<DcsParams>) -> Self {
        match dcs_params {
            Some(params) => StageOpeningSolver::Lazy(params),
            None => StageOpeningSolver::Baked,
        }
    }

    /// Per-trial-point LP load. Issued once by the driver after
    /// `reset_solver_state()` and before any opening solve, replacing the former
    /// `dcs_active`-gated load skips: the driver no longer asks "is DCS active,
    /// so should I skip my normal load?" — both variants own their own load.
    ///
    /// - [`StageOpeningSolver::Baked`]: load the baked all-cuts LP via
    ///   [`load_backward_lp`] (the former per-trial-point load that paired with
    ///   the per-opening solver-state reset).
    /// - [`StageOpeningSolver::Lazy`]: load the cut-free core and build the
    ///   metadata seed ONCE here, then reuse the loaded LP across this trial
    ///   point's openings — the first-solved opening reuses the loaded cut-free
    ///   core with the appended seed and solves with no carried per-opening basis
    ///   (not a cold SDDP solve, it merely lacks a warm per-opening basis to
    ///   carry); the rest warm-carry. The core load also serves as the
    ///   per-trial-point `HiGHS` reset that preserves rank-invariance: state is
    ///   reset at every trial-point boundary and never carried across trial
    ///   points.
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
    /// active variant's path. The variant arms are the two pre-existing
    /// per-opening solve functions, re-homed verbatim.
    ///
    /// `is_first` is `true` for the trial point's **first-solved** opening — ω=0
    /// in canonical order, or the first entry of the solve order when reordering
    /// is enabled. The baked path uses it to load the per-(m, s) stored basis and
    /// capture the post-solve basis back into the store (only on the first-solved
    /// opening; the rest warm re-solve on the retained LU). The lazy path uses
    /// `continue_carry == !is_first`: the first-solved opening starts with no
    /// carried per-opening basis — it reuses the loaded cut-free core with the
    /// appended seed (not a cold SDDP solve, it merely lacks a warm per-opening
    /// basis to carry) — while the rest warm-carry the LP. Decoupling basis
    /// identity from the literal ω=0 is
    /// what lets the openings be solved in any order while the per-(m, s) basis
    /// store stays consistent with the actual first solve.
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
                // The first-solved opening starts with no carried per-opening
                // basis (it reuses the loaded cut-free core with the appended
                // seed — not a cold SDDP solve); subsequent openings warm-carry
                // the LP.
                !is_first,
            ),
        }
    }

    /// Baked all-cuts per-opening solve. Patch the opening bounds on the
    /// already-loaded baked/delta LP, reconstruct + solve via `run_stage_solve`,
    /// extract both state and cut duals, accumulate the outcome (including the
    /// binding-count `slot_increments` update), and capture the first-solved
    /// opening's basis. Body re-homed verbatim from the former free function.
    // Rationale: see [`StageOpeningSolver::solve_opening`]; the disjoint-borrow
    // argument list is unchanged from the former free function.
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
        let indexer = training_ctx.indexer;
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        // Scratch buffers are moved out before the solve to avoid borrow conflicts
        // with `view`'s lifetime. Pre-warmed capacity is reused across openings.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
        let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);

        // Reuse the per-worker snapshot buffer (histogram allocation is retained
        // across openings) instead of cloning via `statistics()`.
        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        // The per-(m, s) stored basis is loaded for, and captured from, the
        // FIRST-SOLVED opening of this trial point — not canonical ω=0. When the
        // openings are solved in canonical order these coincide; when solved in a
        // warm-start order they differ, and basis identity must follow the actual
        // first solve. See `process_trial_point_backward`.
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

        // Extract duals from view (which borrows ws for 'ws).
        // Statistics must be captured after view is dropped.
        let objective = extract_duals_from_view(
            &view,
            indexer.n_state,
            indexer,
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

        // Restore the snapshot buffers (with their grown histogram capacity) for
        // the next opening.
        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        if is_first {
            save_basis_at_omega_zero(ws, succ, basis_slice, m, x_hat);
        }

        Ok(())
    }

    /// Lazy resident-set per-opening solve under Dynamic Cut Selection.
    ///
    /// All openings of a trial point are processed by one worker, sharing the same
    /// pinned incoming state `x̂` and differing only in their noise RHS, so the
    /// cut-free core and the metadata seed are loaded/built ONCE per trial point
    /// by [`StageOpeningSolver::prepare`] and the LP is reused across the
    /// openings. This routine therefore never reloads or re-seeds; it only patches
    /// the opening's bounds and runs the lazy loop via [`lazy_solve_preloaded`]:
    ///
    /// - `continue_carry == false` (the first-solved opening): the lazy loop
    ///   resets the carried row map, appends the seed, and solves with no carried
    ///   per-opening basis (`stored_basis = None`) — it still reuses the loaded
    ///   cut-free core, so this is not a cold SDDP solve, it merely lacks a warm
    ///   per-opening basis to carry. The cut produced is identical to the (former)
    ///   per-opening path.
    /// - `continue_carry == true` (subsequent openings): warm-carry the prior
    ///   opening's LP, basis, and (monotonically grown) resident cut set; re-solve
    ///   warm under the new noise and add only the cuts this opening additionally
    ///   violates. This is what the paper's §3.4 "base recovery" buys, extended
    ///   across the trial point's openings.
    ///
    /// The Benders cut gradient and intercept are read from the final
    /// all-satisfied LP via [`extract_state_duals_only`] in both cases.
    ///
    /// **Binding-count metadata** (`slot_increments`) IS maintained here, but
    /// slot-correct under the lazy layout: the baked path bumps by baked cut-row
    /// order, whereas [`accumulate_dcs_binding_counts`] maps each resident cut row
    /// back to its pool slot via the final [`CutRowMap`](crate::cut::CutRowMap) and bumps
    /// `slot_increments[slot]` for residents whose cut-row dual exceeds
    /// `cut_activity_tolerance` (the same criterion the baked path uses). This
    /// feeds the existing per-stage `metadata_sync_contribution` allreduce, which
    /// advances `last_active_iter` rank-invariantly and so restores the §3.1
    /// clause-1 seed (cuts **binding** in the last `k2` iterations) — without
    /// altering the extracted gradient/intercept (those come solely from the
    /// structural state-column reduced costs and are identical to the all-cuts cut
    /// by exactness).
    ///
    /// One piece of the baked path is intentionally NOT performed here:
    ///
    /// - **first-solved basis capture**: a captured basis would describe the baked
    ///   layout, not the DCS resident subset, so it is skipped; the next DCS solve
    ///   cold-starts its initial solve, which [`lazy_solve_preloaded`] supports
    ///   with `stored_basis = None`.
    ///
    /// Body re-homed verbatim from the former free function.
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
        let indexer = training_ctx.indexer;
        // The DCS LP must start from the cut-free base structural template, NOT
        // `succ.baked_template`: when baking is active the baked template already
        // carries the active cut rows, and loading it would make the lazy loop's
        // fresh CutRowMap treat those baked slots as non-resident and append them a
        // second time (duplicate rows, broken laziness). `ctx.templates[s]` is the
        // cut-free base (its `num_rows` equals `succ.template_num_rows`), so the
        // CutRowMap's base offset matches and the lazy loop exclusively owns the
        // cut rows.
        let core = &ctx.templates[s];
        let col_scale = &ctx.templates[s].col_scale;

        // [`StageOpeningSolver::prepare`] has already loaded the cut-free core and
        // built the metadata seed once for this trial point; here we only apply
        // this opening's state-pin + noise patch. `lazy_solve_preloaded` appends
        // cut rows to the loaded, patched LP and never reloads, so the patch — and,
        // on continued openings, the carried cut rows + warm basis — survive the
        // whole lazy loop.
        patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

        // Reuse the per-worker snapshot buffer (histogram allocation is retained
        // across openings) instead of cloning via `statistics()`.
        let mut stats_before_omega = std::mem::take(&mut ws.backward_accum.stats_before_buf);
        ws.solver.statistics_into(&mut stats_before_omega);

        // Move the state-duals buffer out of `ws.backward_accum` so it can be
        // filled by `extract_state_duals_only` (a `&mut Vec` sink) while `view`
        // holds an immutable borrow of the sibling `dcs_solve` field; restored at
        // the end.
        let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);

        let dcs_ctx = DcsSolveContext {
            stage_index: s,
            scenario_index: scenario,
            iteration: Some(iteration),
            // The first-solved opening of the trial point starts without a
            // carried per-opening basis (`stored_basis = None` below): the lazy
            // loop resets the carried row map and appends the metadata seed
            // before solving. Every subsequent opening warm-carries the prior
            // opening's LP, basis, and resident cut set.
            continue_carry,
        };
        // Disjoint borrows of `ws`: `solver`, `backward_accum.dcs_initial_resident`
        // (shared), and `backward_accum.dcs_solve` (mut) are distinct fields.
        // `lazy_solve_preloaded` copies the final solve into `dcs_solve`'s result
        // buffers and returns `Result<()>`; the zero-cost view is rebuilt below.
        lazy_solve_preloaded(
            &mut ws.solver,
            core,
            succ.successor_pool,
            indexer,
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )?;
        // View over the result buffers (borrows `dcs_solve` immutably; coexists
        // with the immutable `dcs_solve.row_map` read in
        // `accumulate_dcs_binding_counts`).
        let view = ws.backward_accum.dcs_solve.result_view();

        // Cut gradient + intercept from the final all-satisfied LP only. The
        // gradient/intercept come solely from the structural state-column reduced
        // costs, identical in the all-cuts and lazy-solve LPs — exactness is
        // unaffected by the binding-count bookkeeping below.
        let objective =
            extract_state_duals_only(&view, indexer.n_state, indexer, col_scale, &mut state_duals);

        // Binding-count metadata, slot-correct under the resident `CutRowMap`. The
        // final all-satisfied solve's cut-row duals (`view.dual`, the full dual
        // copied into `dcs_solve.res_dual`) are read here and mapped slot→row via
        // the persistent row map that `lazy_solve_preloaded` just finalized.
        // Borrows: `view` and `dcs_solve.row_map` both borrow
        // `ws.backward_accum.dcs_solve` immutably (so they coexist);
        // `slot_increments` is a distinct field of `ws.backward_accum` borrowed
        // mutably.
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

        // Outcome only — no first-solved basis capture (that captured basis would
        // describe the baked layout, not the DCS resident subset). The
        // binding-count update is done above, slot-correct under the lazy layout.
        write_opening_outcome(
            ws,
            omega,
            objective,
            x_hat,
            &stats_before_omega,
            &stats_after_omega,
        );

        // Restore the snapshot buffers (with their grown histogram capacity) for
        // the next opening.
        ws.backward_accum.stats_before_buf = stats_before_omega;
        ws.backward_accum.stats_after_buf = stats_after_omega;

        Ok(())
    }
}

/// Process one trial point `m` in the backward pass, iterating over all openings.
///
/// Solves at each (scenario, opening) and accumulates duals into `per_opening_stats`.
/// On the **baked path** only, at the first-solved opening (`solve_order[0]`, which
/// equals canonical ω=0 only under the identity order), writes the post-solve basis
/// into `basis_slice`; later writes are forbidden (retained-LU corruption risk), and
/// infeasibility at the first-solved opening leaves the slot unchanged. The DCS arm
/// skips first-solved basis capture by design — a captured basis would describe the
/// baked layout, not the DCS resident subset (see [`StageOpeningSolver::solve_lazy`])
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

    // Throwaway, env-gated backward `noise_key` diagnostic. `None` on the
    // default path (single hoisted boolean check; no key compute, no
    // allocation). When `Some`, after each opening solve we pair the precomputed
    // per-(stage, ω) noise key with that opening's just-consumed
    // `simplex_iterations` and emit one record to stderr.
    let noise_key_diag = training_ctx.noise_key_diag;

    // Solve order for this stage's openings. Openings are SOLVED in
    // `solve_order(s)` — a run-constant, rank-invariant permutation precomputed at
    // setup (always installed descending; see `StudySetup::from_broadcast_params`).
    // Per-opening outcomes are written and aggregated by CANONICAL ω below, so the
    // generated cut is bit-identical regardless of the solve order: the reorder
    // changes only the warm-start chain, never which cuts are produced.
    //
    // `solve_order(s)` defaults to the identity permutation when no order is
    // installed (e.g. in tests that build the context directly), so the loop
    // degrades to canonical `0..n` order in that case.
    let solve_order = tree_view.solve_order_data(s);
    debug_assert_eq!(
        solve_order.len(),
        succ.probabilities.len(),
        "solve_order(s) must be a permutation of 0..n_openings"
    );
    // The FIRST-SOLVED opening of the trial point: it owns the per-(m, s) basis
    // load/capture (baked path) and the fresh (non-warm-carry) solve (DCS path).
    // It is the first entry of the solve-order permutation (ω=0 under the identity
    // default).
    // `u32 -> usize` is a lossless widening on every supported target.
    let first = solve_order[0] as usize;

    let mut omega_position = 0usize;
    while omega_position < succ.probabilities.len() {
        // Canonical ω for this iteration: index the precomputed solve-order
        // permutation (the identity `0..n` when no order is installed).
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

        // Env-gated throwaway emit: one record per backward opening pairing the
        // precomputed σ-weighted noise key with the opening's just-consumed
        // simplex iterations (the only live value pulled from the solve). The
        // outer `if let Some` is the single hoisted boolean check; the default
        // (`None`) path does nothing.
        if let Some(diag) = noise_key_diag {
            let simplex_iterations = ws.backward_accum.per_opening_stats[omega].simplex_iterations;
            let noise_key = diag.key(s, omega).unwrap_or(f64::NAN);
            eprintln!(
                "COBRE_W1_DIAG\tstage={s}\ttrial={scenario}\tomega={omega}\t\
                 noise_key={noise_key:.17e}\tsimplex_iterations={simplex_iterations}"
            );
        }
    }

    // Aggregate into the per-worker `agg_coefficients` scratch (unchanged
    // order/values), then copy the result verbatim into this trial point's slot
    // of the per-worker arena so the bytes outlive the parallel closure without
    // a per-cut heap allocation (see module-level docs and `agg_arena`).
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
    // `copy_from_slice` writes exactly the bytes the former `.clone()` produced;
    // the arena is storage only — no value transform, no reordering.
    ws.backward_accum.agg_arena[coefficients_range.clone()]
        .copy_from_slice(&ws.backward_accum.agg_coefficients[..n_state]);
    debug_assert!(
        u32::try_from(scenario).is_ok(),
        "global scenario index overflows u32"
    );
    #[allow(clippy::cast_possible_truncation)]
    let forward_pass_index = scenario as u32;
    // Accumulate binding counts into the metadata buffer for later merge.
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
