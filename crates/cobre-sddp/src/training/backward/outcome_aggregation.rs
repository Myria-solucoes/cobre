//! Per-opening outcome, binding-count, and basis-capture accumulators.
//!
//! After each backward opening solve, these helpers write the per-opening stats
//! delta and aggregated outcome into the workspace accumulators, bump binding-cut
//! slot increments (baked-order or, for the DCS lazy layout, row-map-correct), and
//! capture the first-solved opening's basis. The functions are `pub(crate)`
//! because the per-opening solve dispatch in [`super::StageOpeningSolver`]
//! (`trial_point`) drives them across the submodule boundary.

use cobre_solver::{SolverInterface, SolverStatistics};

use crate::{
    cut::{CutRowMap, pool::CutPool},
    forward::write_capture_metadata,
    solver_stats::SolverStatsDelta,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorSpec;

/// Accumulate one opening's solve result into the workspace accumulators.
///
/// Called after `view` is dropped (so `ws` is freely borrowable). Writes:
/// - per-opening stats delta into `ws.backward_accum.per_opening_stats[omega]`
/// - outcome coefficients, objective, and intercept into `ws.backward_accum.outcomes[omega]`
/// - binding-cut slot increments into `ws.backward_accum.slot_increments`
pub(crate) fn accumulate_opening_outcome<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorSpec<'_>,
    omega: usize,
    objective: f64,
    x_hat: &[f64],
    stats_before: &SolverStatistics,
    stats_after: &SolverStatistics,
) {
    write_opening_outcome(ws, omega, objective, x_hat, stats_before, stats_after);

    // Update binding-cut slot increments from cut duals.
    for (cut_idx, &slot) in succ.successor_active_slots.iter().enumerate() {
        if ws
            .backward_accum
            .cut_duals_buf
            .get(cut_idx)
            .is_some_and(|&d| d > succ.cut_activity_tolerance)
        {
            ws.backward_accum.slot_increments[slot] += 1;
        }
    }
}

/// Accumulate binding-cut slot increments from the DCS lazy solve's final
/// all-satisfied LP, slot-correct under the resident [`CutRowMap`] layout.
///
/// The baked path ([`accumulate_opening_outcome`]) bumps `slot_increments` by
/// **baked cut-row order** (`successor_active_slots[cut_idx]` ↔
/// `cut_duals[cut_idx]`). Under DCS the resident cut rows are a row-map-ordered
/// subset, so that mapping does not apply: a slot's cut row, when resident, lives
/// at the LP row `row_map.lp_row_for_slot(slot)` (an index `>= base_row_offset =
/// core.num_rows`), and its dual is `dual[lp_row]`. This routine therefore
/// iterates the pool's populated slots, and for each slot that is **resident**
/// at the converged optimum bumps `slot_increments[slot]` when its cut-row dual
/// exceeds `cut_activity_tolerance` — the *same* binding criterion the baked path
/// uses (raw dual, not magnitude), applied to the lazy layout.
///
/// A non-resident slot did not bind (by exactness it was not violated at the
/// optimum, else the lazy loop would have added it), so leaving it uncounted is
/// correct. `dual` must be the FINAL all-satisfied solve's dual vector, read
/// before the [`SolutionView`] is dropped; `row_map` must be the residency that
/// produced that solve (the persistent `DcsSolveScratch.row_map` after
/// `lazy_solve_preloaded` returns). The bump is a deterministic function of the
/// resident map and the cut-row duals only — no worker id, rank, or trace — so
/// the order-insensitive metadata allreduce preserves rank-invariance.
///
/// `slot_increments` accumulates across the trial point's openings (summed),
/// matching the baked path's per-(trial-point) accumulation; the per-trial-point
/// reset of `slot_increments` happens in the stage loop before the openings run.
pub(crate) fn accumulate_dcs_binding_counts(
    dual: &[f64],
    row_map: &CutRowMap,
    pool: &CutPool,
    cut_activity_tolerance: f64,
    slot_increments: &mut [u64],
) {
    for (slot, increment) in slot_increments
        .iter_mut()
        .enumerate()
        .take(pool.populated_count)
    {
        let Some(lp_row) = row_map.lp_row_for_slot(slot) else {
            continue;
        };
        if dual
            .get(lp_row)
            .is_some_and(|&d| d > cut_activity_tolerance)
        {
            *increment += 1;
        }
    }
}

/// Write one opening's stats delta and outcome (coefficients + intercept) into
/// the workspace accumulators, without touching binding-count metadata.
///
/// Shared by the all-cuts path ([`accumulate_opening_outcome`], which adds the
/// cut-dual→`slot_increments` update) and the lazy-solve path (which skips that
/// update because its resident cut rows are a row-map-ordered subset whose
/// cut-row→slot mapping differs from the all-cuts layout). The cut gradient and
/// intercept come from the state duals and are identical either way.
pub(crate) fn write_opening_outcome<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    omega: usize,
    objective: f64,
    x_hat: &[f64],
    stats_before: &SolverStatistics,
    stats_after: &SolverStatistics,
) {
    // Per-opening stats delta.
    let opening_delta = SolverStatsDelta::from_snapshots(stats_before, stats_after);
    SolverStatsDelta::accumulate_into(
        &mut ws.backward_accum.per_opening_stats[omega],
        &opening_delta,
    );

    // Copy state duals into outcome coefficients, then compute the intercept.
    // Simultaneous access to `outcomes[omega]` (mutable) and `state_duals_buf`
    // (immutable) is safe because they are distinct fields of `BackwardAccumulators`.
    let out = &mut ws.backward_accum.outcomes[omega];
    out.coefficients
        .copy_from_slice(&ws.backward_accum.state_duals_buf);
    out.objective_value = objective;
    // Intercept: alpha = Q_scaled - pi' * x_hat.
    // All terms are in scaled cost units (LP duals inherit cost scaling).
    out.intercept = objective
        - out
            .coefficients
            .iter()
            .zip(x_hat)
            .map(|(pi, x)| pi * x)
            .sum::<f64>();
}

/// Capture the post-solve basis at the first-solved opening into `basis_slice[m, s]`.
///
/// Only called for the first-solved opening of each trial point (identified by
/// `is_first`, i.e. `solve_order[0]` — which equals canonical ω=0 only under the
/// identity order); writes after the first solve are forbidden because the
/// retained LU factorization would be overwritten by subsequent opening solves,
/// making the stored basis stale and potentially infeasible when reloaded.
///
/// Reuses an existing slot in-place when present (avoids reallocation on
/// subsequent iterations). Allocates a new `CapturedBasis` only on the first
/// capture for this `(m, s)` pair.
pub(crate) fn save_basis_at_omega_zero<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    succ: &SuccessorSpec<'_>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    m: usize,
    x_hat: &[f64],
) {
    let s = succ.successor;
    let num_cols = succ.baked_template.num_cols;
    let base_row_count = succ.template_num_rows;
    let cut_row_count = succ.num_cuts_at_successor;
    let basis_row_capacity = base_row_count + cut_row_count;
    if let Some(captured) = basis_slice.get_mut(m, s).as_mut() {
        ws.solver.get_basis(&mut captured.basis);
        write_capture_metadata(
            captured,
            succ.successor_pool,
            base_row_count,
            cut_row_count,
            x_hat,
        );
    } else {
        let mut captured = CapturedBasis::new(
            num_cols,
            basis_row_capacity,
            base_row_count,
            cut_row_count,
            x_hat.len(),
        );
        ws.solver.get_basis(&mut captured.basis);
        write_capture_metadata(
            &mut captured,
            succ.successor_pool,
            base_row_count,
            cut_row_count,
            x_hat,
        );
        *basis_slice.get_mut(m, s) = Some(captured);
    }
}
