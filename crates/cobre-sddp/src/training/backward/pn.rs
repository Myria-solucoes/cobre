//! Parallel-by-node (PN) backward stage scheduler: the work unit is an
//! opening-block of one trial point (`B_s` consecutive `solve_order` positions),
//! not a whole trial point. Units are enumerated in canonical (m-major,
//! block-minor) order and claimed dynamically by workers from a shared atomic
//! counter (work-stealing dispatch); each unit's chain is self-contained (head
//! anchored on the forward capture at `(m, s)`, warm-continue inside the block),
//! so results are independent of the worker count and of which worker claims
//! which unit, by construction. Outcomes are accumulated worker-locally and
//! scattered into a per-`(m, ω)` arena after the parallel region; cut
//! aggregation runs canonically over ω per trial point, in ascending m
//! (sddp.md "PN opening-block scheduler is warm-start-only").

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use cobre_solver::SolverInterface;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    risk_measure::{BackwardOutcome, RiskMeasure},
    solver_stats::SolverStatsDelta,
    stage_solve::{StageInputs, run_stage_solve},
    state_exchange::ExchangeBuffers,
    workspace::{BackwardPnScratch, BasisStore, SolverWorkspace},
};

use super::{
    SuccessorSpec,
    duals_extraction::extract_duals_from_view,
    lp_setup::{load_backward_lp, patch_opening_bounds},
    outcome_aggregation::accumulate_opening_outcome,
};

/// One solved (trial point, opening) outcome produced by a PN unit.
pub(crate) struct PnOutcome {
    pub(crate) m: usize,
    pub(crate) omega: usize,
    pub(crate) outcome: BackwardOutcome,
}

/// Resolve the per-stage opening-block size `B_s` (D5): the caller-configured
/// size clamped to `n_openings`, or half the openings (rounded up) when unset.
/// Never a function of worker or rank count.
pub(crate) fn resolve_block_size(
    n_openings: usize,
    opening_block_size: Option<NonZeroUsize>,
) -> usize {
    let requested = opening_block_size.map_or(n_openings.div_ceil(2), NonZeroUsize::get);
    requested.min(n_openings)
}

/// Solve one backward stage PN-style; returns, per worker, either the error
/// that aborted its claim loop or `(worker_index, outcome_count)` — the count
/// of entries this worker recorded into its own
/// `ws.backward_accum.pn_outcomes_buf[..outcome_count]`, which
/// [`pn_finish`] resolves back through `workspaces[worker_index]`.
// Rationale: mirrors `process_stage_backward`'s disjoint-borrow argument list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_stage_backward_pn<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    local_work: usize,
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    iteration: u64,
    succ: &SuccessorSpec<'_>,
    basis_store: &BasisStore,
    block_size: usize,
) -> Vec<Result<(usize, usize), SddpError>> {
    let n_openings = succ.probabilities.len();
    let cut_n_state = succ.cut_state.n_slots();
    let pop = succ.successor_populated_count;
    let n_blocks = n_openings.div_ceil(block_size.max(1));
    let total_units = local_work * n_blocks;
    let tree_view = training_ctx.stochastic.tree_view();
    let solve_order = tree_view.solve_order_data(succ.successor);
    let next_unit = AtomicUsize::new(0);

    workspaces
        .par_iter_mut()
        .enumerate()
        .map(|(w, ws)| {
            while ws.backward_accum.outcomes.len() < n_openings {
                ws.backward_accum.outcomes.push(BackwardOutcome {
                    intercept: 0.0,
                    coefficients: vec![0.0_f64; cut_n_state],
                    objective_value: 0.0,
                });
            }
            for outcome in &mut ws.backward_accum.outcomes[..n_openings] {
                outcome.coefficients.resize(cut_n_state, 0.0_f64);
            }
            if ws.backward_accum.slot_increments.len() < pop {
                ws.backward_accum.slot_increments.resize(pop, 0u64);
            }
            if ws.backward_accum.metadata_sync_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_contribution
                    .resize(pop, 0u64);
            }
            ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
            ws.backward_accum.slot_increments[..pop].fill(0);
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }

            let s = succ.successor;
            let mut count = 0usize;

            loop {
                let u = next_unit.fetch_add(1, Ordering::Relaxed);
                if u >= total_units {
                    break;
                }
                let m = u / n_blocks;
                let b = u % n_blocks;
                let x_hat = exchange.state_at(succ.my_rank, m);
                let scenario = fwd_offset + m;

                ws.solver.reset_solver_state();
                load_backward_lp(ws, succ);

                let pos_start = b * block_size;
                let pos_end = (pos_start + block_size).min(n_openings);
                for (local_idx, &omega_u32) in solve_order[pos_start..pos_end].iter().enumerate() {
                    let omega = omega_u32 as usize;
                    let raw_noise = tree_view.opening(s, omega);
                    patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s)?;

                    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);
                    let mut cut_duals = std::mem::take(&mut ws.backward_accum.cut_duals_buf);
                    let mut stats_before = std::mem::take(&mut ws.backward_accum.stats_before_buf);
                    ws.solver.statistics_into(&mut stats_before);

                    // Only the block's first position warm-starts from the
                    // captured `(m, s)` basis; later positions warm-continue the
                    // block's own retained factorization (the same
                    // first-solved-only rule the trial-point path enforces).
                    let stored = if local_idx == 0 {
                        basis_store.get(m, s)
                    } else {
                        None
                    };
                    let inputs = StageInputs {
                        stage_context: ctx,
                        pool: succ.successor_pool,
                        stored_basis: stored,
                        stage_index: s,
                        scenario_index: scenario,
                        iteration: Some(iteration),
                    };
                    let view = run_stage_solve(ws, &inputs)?;
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

                    let mut stats_after = std::mem::take(&mut ws.backward_accum.stats_after_buf);
                    ws.solver.statistics_into(&mut stats_after);
                    accumulate_opening_outcome(
                        ws,
                        succ,
                        omega,
                        objective,
                        x_hat,
                        &stats_before,
                        &stats_after,
                    );
                    ws.backward_accum.stats_before_buf = stats_before;
                    ws.backward_accum.stats_after_buf = stats_after;

                    // Grow-once record into the pre-allocated per-worker
                    // out-buffer: a fresh slot is pushed only the first time
                    // `count` exceeds every prior stage's high-water mark; every
                    // subsequent stage's writes land on an already-allocated
                    // slot via `copy_from_slice`, never a clone.
                    if ws.backward_accum.pn_outcomes_buf.len() <= count {
                        ws.backward_accum.pn_outcomes_buf.push(PnOutcome {
                            m: 0,
                            omega: 0,
                            outcome: BackwardOutcome {
                                intercept: 0.0,
                                coefficients: vec![0.0_f64; cut_n_state],
                                objective_value: 0.0,
                            },
                        });
                    }
                    let recorded = &mut ws.backward_accum.pn_outcomes_buf[count];
                    recorded.m = m;
                    recorded.omega = omega;
                    recorded
                        .outcome
                        .coefficients
                        .copy_from_slice(&ws.backward_accum.outcomes[omega].coefficients);
                    recorded.outcome.intercept = ws.backward_accum.outcomes[omega].intercept;
                    recorded.outcome.objective_value =
                        ws.backward_accum.outcomes[omega].objective_value;
                    count += 1;
                }
            }

            for slot in 0..pop {
                let increment = ws.backward_accum.slot_increments[slot];
                if increment > 0 {
                    ws.backward_accum.metadata_sync_contribution[slot] += increment;
                }
            }

            Ok((w, count))
        })
        .collect()
}

/// Scatter worker outcomes into the pre-allocated per-`(m, ω)` arena, aggregate
/// each trial point's cut canonically over ω, and insert into the FCF in
/// ascending m (sddp.md "PN opening-block scheduler is warm-start-only" — never
/// claim order).
///
/// `scratch` is `BackwardPassState::pn_scratch`, sized once by `set_scheduler`;
/// the scatter overwrites `arena[0..local_work * n_openings]` in full before the
/// aggregation loop reads it, so no clear pass is required between stages.
// Rationale: mirrors the staged-cut merge's disjoint-borrow argument list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pn_finish<S: SolverInterface>(
    worker_out: Vec<Result<(usize, usize), SddpError>>,
    workspaces: &[SolverWorkspace<S>],
    local_work: usize,
    n_openings: usize,
    cut_n_state: usize,
    probabilities: &[f64],
    risk_measure: &RiskMeasure,
    fcf: &mut FutureCostFunction,
    t: usize,
    iteration: u64,
    fwd_offset: usize,
    scratch: &mut BackwardPnScratch,
) -> Result<usize, SddpError> {
    let arena_len = local_work * n_openings;
    debug_assert!(
        scratch.arena.len() >= arena_len && scratch.coeffs_buf.len() == cut_n_state,
        "BackwardPnScratch must already cover local_work * n_openings ({arena_len}) with \
         n_state = {cut_n_state} coefficients before the PN scatter (arena len = {}, coeffs_buf \
         len = {})",
        scratch.arena.len(),
        scratch.coeffs_buf.len(),
    );
    for res in worker_out {
        let (w, count) = res?;
        for recorded in &workspaces[w].backward_accum.pn_outcomes_buf[..count] {
            let slot = &mut scratch.arena[recorded.m * n_openings + recorded.omega];
            slot.coefficients
                .copy_from_slice(&recorded.outcome.coefficients);
            slot.intercept = recorded.outcome.intercept;
            slot.objective_value = recorded.outcome.objective_value;
        }
    }
    let mut cuts_added = 0usize;
    for m in 0..local_work {
        let mut intercept = 0.0_f64;
        risk_measure.aggregate_cut_into(
            &scratch.arena[m * n_openings..(m + 1) * n_openings],
            probabilities,
            &mut intercept,
            &mut scratch.coeffs_buf,
            &mut scratch.risk_scratch,
        );
        debug_assert!(
            u32::try_from(fwd_offset + m).is_ok(),
            "global scenario index overflows u32"
        );
        #[allow(clippy::cast_possible_truncation)]
        fcf.add_cut(
            t,
            iteration,
            (fwd_offset + m) as u32,
            intercept,
            &scratch.coeffs_buf,
        );
        cuts_added += 1;
    }
    debug_assert_eq!(
        cuts_added, local_work,
        "pn_finish must add exactly local_work cuts, matching the trial-point path"
    );
    Ok(cuts_added)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::resolve_block_size;

    #[test]
    fn resolve_block_size_defaults_to_half_openings_rounded_up() {
        assert_eq!(resolve_block_size(7, None), 4);
        assert_eq!(resolve_block_size(8, None), 4);
        assert_eq!(resolve_block_size(1, None), 1);
    }

    #[test]
    fn resolve_block_size_clamps_configured_size_to_n_openings() {
        assert_eq!(resolve_block_size(7, NonZeroUsize::new(3)), 3);
        assert_eq!(resolve_block_size(7, NonZeroUsize::new(100)), 7);
    }
}
