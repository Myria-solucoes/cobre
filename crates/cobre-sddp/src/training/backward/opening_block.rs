//! Opening-block backward stage scheduler: the work unit is an
//! opening-block of one trial point (`B_s` consecutive `solve_order` positions),
//! not a whole trial point. Units are claimed dynamically by workers from a
//! shared atomic counter (work-stealing dispatch) in block-major order — every
//! `m` of one block is claimed before the next block, in the `block_order`
//! permutation ([`hardest_first_block_order`]: hardest-first by the previous iteration's
//! mean pivot, or [`identity_block_order`] when hardest-first is disabled).
//! Each unit's
//! chain is self-contained (head anchored on the forward capture at `(m, s)`,
//! warm-continue inside the block), so results are independent of the worker
//! count and of the claim/block order, by construction. Outcomes are
//! accumulated worker-locally and scattered into a per-`(m, ω)` arena after the
//! parallel region; cut aggregation runs canonically over ω per trial point, in
//! ascending m (sddp.md "Opening-block scheduler is warm-start-only").

use std::cmp;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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
    workspace::{BasisStore, OpeningBlockScratch, SolverWorkspace},
};

use super::{
    SuccessorSpec,
    duals_extraction::extract_duals_from_view,
    lp_setup::{load_backward_lp, patch_opening_bounds},
    outcome_aggregation::accumulate_opening_outcome,
};

/// One solved (trial point, opening) outcome produced by an opening-block unit.
pub(crate) struct OpeningOutcome {
    pub(crate) m: usize,
    pub(crate) omega: usize,
    pub(crate) outcome: BackwardOutcome,
}

/// Resolve the per-stage opening-block size `B_s`: the caller-configured
/// size clamped to `n_openings`, or half the openings (rounded up) when unset.
/// Never a function of worker or rank count.
pub(crate) fn resolve_block_size(
    n_openings: usize,
    opening_block_size: Option<NonZeroUsize>,
) -> usize {
    let requested = opening_block_size.map_or(n_openings.div_ceil(2), NonZeroUsize::get);
    requested.min(n_openings)
}

/// Resolve the number of opening-blocks `n_blocks` a stage's `n_openings`
/// split into under `block_size` — the single owner of this formula, shared
/// by [`process_stage_backward_opening_block`] (the claim-loop unit count) and its
/// caller (the block-pivot merge's per-stage block count).
pub(crate) fn opening_block_count(n_openings: usize, block_size: usize) -> usize {
    n_openings.div_ceil(block_size.max(1))
}

/// Fill `out[..n_blocks]` with the identity permutation `0..n_blocks` — the
/// canonical claim order (hardest-first disabled) and [`hardest_first_block_order`]'s pre-sort
/// seed. `out` is pre-allocated (`OpeningBlockScratch::block_order`, reserved
/// capacity `n_blocks_max`); cleared and refilled in place, never reallocated
/// on the hot path.
pub(crate) fn identity_block_order(n_blocks: usize, out: &mut Vec<u32>) {
    debug_assert!(
        n_blocks <= out.capacity(),
        "n_blocks ({n_blocks}) exceeds the block_order scratch's reserved capacity ({}) — \
         resizing must never reallocate on the opening-block hot path",
        out.capacity()
    );
    out.clear();
    #[allow(clippy::cast_possible_truncation)]
    out.extend((0..n_blocks).map(|b| b as u32));
}

/// Compute the hardest-block-first (longest-processing-time, LPT) claim
/// order for one stage's `n_blocks`
/// opening-blocks from the previous iteration's per-block mean
/// `(simplex_iterations sum, count)` pivot row: block indices sorted by
/// descending mean, ties — including an all-zero row (iteration 1, or any
/// block with no prior data) — broken by ascending index, a total order.
/// Compares means by cross-multiplying `sum`/`count` in `u128` rather than
/// dividing `f64`, avoiding both the `count == 0` divide-by-zero and a
/// float-compare edge case.
pub(crate) fn hardest_first_block_order(
    prev_row: &[(u64, u64)],
    n_blocks: usize,
    out: &mut Vec<u32>,
) {
    debug_assert!(prev_row.len() >= n_blocks);
    identity_block_order(n_blocks, out);
    out.sort_by(|&a, &b| {
        let (sum_a, count_a) = prev_row[a as usize];
        let (sum_b, count_b) = prev_row[b as usize];
        match (count_a == 0, count_b == 0) {
            (true, true) => a.cmp(&b),
            (true, false) => cmp::Ordering::Greater,
            (false, true) => cmp::Ordering::Less,
            (false, false) => {
                let cross_a = u128::from(sum_a) * u128::from(count_b);
                let cross_b = u128::from(sum_b) * u128::from(count_a);
                cross_b.cmp(&cross_a).then_with(|| a.cmp(&b))
            }
        }
    });
}

/// Solve one backward stage opening-block-style; returns, per worker, either the error
/// that aborted its claim loop or `(worker_index, outcome_count)` — the count
/// of entries this worker recorded into its own
/// `ws.backward_accum.opening_outcomes_buf[..outcome_count]`, which
/// [`opening_block_finish`] resolves back through `workspaces[worker_index]`.
// Rationale: mirrors `process_stage_backward`'s disjoint-borrow argument list;
// too_many_lines is the claim loop's single sequential region (LP load, warm-chain
// solve, dual extraction, pivot accumulation, outcome recording) — splitting it
// would scatter state the next step in the same loop iteration reads.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn process_stage_backward_opening_block<S: SolverInterface + Send>(
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
    block_order: &[u32],
) -> Vec<Result<(usize, usize), SddpError>> {
    let n_openings = succ.probabilities.len();
    let cut_n_state = succ.cut_state.n_slots();
    let pop = succ.successor_populated_count;
    let n_blocks = opening_block_count(n_openings, block_size);
    debug_assert_eq!(
        block_order.len(),
        n_blocks,
        "block_order must hold exactly n_blocks entries"
    );
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
            if ws.backward_accum.block_pivot_sum.len() < n_blocks {
                ws.backward_accum.block_pivot_sum.resize(n_blocks, 0u64);
                ws.backward_accum.block_pivot_count.resize(n_blocks, 0u64);
            }
            ws.backward_accum.block_pivot_sum[..n_blocks].fill(0);
            ws.backward_accum.block_pivot_count[..n_blocks].fill(0);

            let worker_stage_wall_start = Instant::now();
            let s = succ.successor;
            let mut count = 0usize;

            loop {
                let u = next_unit.fetch_add(1, Ordering::Relaxed);
                if u >= total_units {
                    break;
                }
                let b = block_order[u / local_work] as usize;
                let m = u % local_work;
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
                    // Counters are monotonically increasing (mirrors
                    // `SolverStatsDelta::from_snapshots`); the running sum below
                    // saturates against overflow across many iterations.
                    let opening_simplex_iters =
                        stats_after.total_iterations - stats_before.total_iterations;
                    ws.backward_accum.block_pivot_sum[b] =
                        ws.backward_accum.block_pivot_sum[b].saturating_add(opening_simplex_iters);
                    ws.backward_accum.block_pivot_count[b] =
                        ws.backward_accum.block_pivot_count[b].saturating_add(1);
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

                    // A reused slot's `coefficients` may carry a DIFFERENT prior
                    // stage's `cut_n_state` (a successor disabling a state group
                    // shrinks it, a later stage regrows it), so it is resized to
                    // THIS stage's `cut_n_state` before every write — otherwise
                    // `copy_from_slice` panics on a length mismatch.
                    if ws.backward_accum.opening_outcomes_buf.len() <= count {
                        ws.backward_accum.opening_outcomes_buf.push(OpeningOutcome {
                            m: 0,
                            omega: 0,
                            outcome: BackwardOutcome {
                                intercept: 0.0,
                                coefficients: vec![0.0_f64; cut_n_state],
                                objective_value: 0.0,
                            },
                        });
                    }
                    let recorded = &mut ws.backward_accum.opening_outcomes_buf[count];
                    recorded.outcome.coefficients.resize(cut_n_state, 0.0_f64);
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

            ws.worker_timing_buf.backward_wall_ms +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;

            Ok((w, count))
        })
        .collect()
}

/// Scatter worker outcomes into the pre-allocated per-`(m, ω)` arena, aggregate
/// each trial point's cut canonically over ω, and insert into the FCF in
/// ascending m (sddp.md "Opening-block scheduler is warm-start-only" — never
/// claim order).
///
/// `scratch` is `BackwardPassState::opening_block_scratch`, sized once by `set_scheduler`;
/// the scatter overwrites `arena[0..local_work * n_openings]` in full before the
/// aggregation loop reads it, so no clear pass is required between stages. Each
/// touched slot's `coefficients` (and `coeffs_buf`) resize to THIS stage's
/// `cut_n_state`, which may differ from a prior stage's — always within the
/// capacity `OpeningBlockScratch::sized` reserved at the run's global `n_state`, so
/// this never reallocates on the hot path.
// Rationale: mirrors the staged-cut merge's disjoint-borrow argument list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn opening_block_finish<S: SolverInterface>(
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
    scratch: &mut OpeningBlockScratch,
) -> Result<usize, SddpError> {
    let arena_len = local_work * n_openings;
    debug_assert!(
        scratch.arena.len() >= arena_len,
        "OpeningBlockScratch arena must already cover local_work * n_openings ({arena_len}); got \
         arena len = {}",
        scratch.arena.len(),
    );
    debug_assert!(
        cut_n_state <= scratch.coeffs_buf.capacity(),
        "cut_n_state ({cut_n_state}) exceeds OpeningBlockScratch::coeffs_buf's reserved capacity \
         ({}) — resizing must never reallocate on the opening-block hot path",
        scratch.coeffs_buf.capacity()
    );
    scratch.coeffs_buf.resize(cut_n_state, 0.0_f64);
    for res in worker_out {
        let (w, count) = res?;
        for recorded in &workspaces[w].backward_accum.opening_outcomes_buf[..count] {
            let slot = &mut scratch.arena[recorded.m * n_openings + recorded.omega];
            debug_assert!(
                cut_n_state <= slot.coefficients.capacity(),
                "cut_n_state ({cut_n_state}) exceeds this arena slot's reserved capacity ({}) — \
                 resizing must never reallocate on the opening-block hot path",
                slot.coefficients.capacity()
            );
            slot.coefficients.resize(cut_n_state, 0.0_f64);
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
        "opening_block_finish must add exactly local_work cuts, matching the trial-point path"
    );
    Ok(cuts_added)
}

/// Merge each worker's per-block `(simplex_iterations sum, count)` — filled
/// during [`process_stage_backward_opening_block`]'s claim loop, aggregated over every
/// trial point `m` a worker claimed for a block by construction — into
/// [`OpeningBlockScratch::block_pivots`] at `stage_key`'s row. Telemetry-only:
/// touches no cut-generation state, so it is cut-neutral by construction
/// (disjoint from `opening_block_finish`'s per-`(m, ω)` arena and FCF insertion).
///
/// `per_worker` yields each worker's `(sums, counts)` slices, both indexed
/// `[0, n_blocks)`; `n_blocks <= scratch.n_blocks_max` is the caller's
/// contract (checked in debug builds).
pub(crate) fn merge_block_pivots<'a>(
    per_worker: impl Iterator<Item = (&'a [u64], &'a [u64])>,
    n_blocks: usize,
    stage_key: usize,
    scratch: &mut OpeningBlockScratch,
) {
    debug_assert!(
        n_blocks <= scratch.n_blocks_max,
        "n_blocks ({n_blocks}) must not exceed scratch.n_blocks_max ({})",
        scratch.n_blocks_max
    );
    debug_assert!(
        stage_key * scratch.n_blocks_max + n_blocks <= scratch.block_pivots.len(),
        "stage_key ({stage_key}) row must fit scratch.block_pivots (len {})",
        scratch.block_pivots.len()
    );
    for (sums, counts) in per_worker {
        debug_assert!(sums.len() >= n_blocks && counts.len() >= n_blocks);
        for b in 0..n_blocks {
            let idx = stage_key * scratch.n_blocks_max + b;
            let (sum, count) = scratch.block_pivots[idx];
            scratch.block_pivots[idx] =
                (sum.saturating_add(sums[b]), count.saturating_add(counts[b]));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        hardest_first_block_order, identity_block_order, merge_block_pivots, opening_block_count,
        resolve_block_size,
    };
    use crate::workspace::OpeningBlockScratch;

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

    #[test]
    fn opening_block_count_matches_div_ceil() {
        assert_eq!(opening_block_count(7, 4), 2);
        assert_eq!(opening_block_count(8, 4), 2);
        assert_eq!(opening_block_count(1, 1), 1);
    }

    /// Two workers' per-block `simplex_iterations` sums/counts merged into the
    /// same `(stage, block)` bucket yield the expected combined sum, count,
    /// and mean — the aggregate-over-`m` contract (sddp.md "Hardest-first claim
    /// order is result-neutral") at its smallest scale.
    #[test]
    fn merge_block_pivots_sums_two_workers_into_one_bucket() {
        let n_blocks = 2_usize;
        let bwd_max_openings = 4_usize;
        let num_stages = 3_usize;
        let mut scratch = OpeningBlockScratch::sized(1, bwd_max_openings, 1, num_stages);
        let stage_key = 1_usize;

        let worker0_sum = [10_u64, 0];
        let worker0_count = [2_u64, 0];
        let worker1_sum = [5_u64, 0];
        let worker1_count = [1_u64, 0];
        let per_worker = [
            (worker0_sum.as_slice(), worker0_count.as_slice()),
            (worker1_sum.as_slice(), worker1_count.as_slice()),
        ];

        merge_block_pivots(per_worker.into_iter(), n_blocks, stage_key, &mut scratch);

        let idx = stage_key * bwd_max_openings;
        assert_eq!(
            scratch.block_pivots[idx],
            (15, 3),
            "block 0 must hold the summed (sum, count) across both workers"
        );
        #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
        {
            assert_eq!(
                scratch.block_pivots[idx].0 as f64 / scratch.block_pivots[idx].1 as f64,
                5.0
            );
        }
        assert_eq!(
            scratch.block_pivots[idx + 1],
            (0, 0),
            "block 1 (untouched by either worker) must stay zeroed"
        );
    }

    #[test]
    fn identity_block_order_fills_ascending_prefix() {
        let mut out = vec![9_u32; 5];
        identity_block_order(3, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn hardest_first_block_order_sorts_descending_mean_ties_by_index() {
        // Means: block 0 = 5.0, block 1 = 10.0, block 2 = 4.0, block 3 = 5.0 (ties block 0).
        let prev_row = [(10_u64, 2_u64), (30, 3), (8, 2), (10, 2)];
        let mut out = vec![0_u32; 4];
        hardest_first_block_order(&prev_row, 4, &mut out);
        assert_eq!(
            out,
            vec![1, 0, 3, 2],
            "blocks sort by descending mean; the (block 0, block 3) tie at mean 5.0 breaks by \
             ascending index"
        );
    }

    #[test]
    fn hardest_first_block_order_falls_back_to_identity_when_no_prior_data() {
        let prev_row = [(0_u64, 0_u64); 3];
        let mut out = vec![0_u32; 3];
        hardest_first_block_order(&prev_row, 3, &mut out);
        assert_eq!(
            out,
            vec![0, 1, 2],
            "an all-zero previous-iteration row (iteration 1) must yield the identity order"
        );
    }

    #[test]
    fn hardest_first_block_order_sorts_zero_count_block_as_least_hard() {
        // Block 1 has no prior data (undefined mean); blocks 0 and 2 have real means
        // 5.0 and 10.0. Block 1 must sort last, never panic on the count == 0 divide.
        let prev_row = [(10_u64, 2_u64), (0, 0), (30, 3)];
        let mut out = vec![0_u32; 3];
        hardest_first_block_order(&prev_row, 3, &mut out);
        assert_eq!(
            out,
            vec![2, 0, 1],
            "a zero-count block sorts least-hard, after every block with real prior data"
        );
    }
}
