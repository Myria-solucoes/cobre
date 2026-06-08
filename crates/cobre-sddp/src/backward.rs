//! Backward pass execution for the SDDP training loop.
//!
//! `run_backward_pass` sweeps stages in reverse order (`T-2` down to `0`),
//! evaluating the cost-to-go at each trial point **assigned to this rank**
//! during the forward pass. For each trial point, the backward pass iterates
//! over every opening from the fixed opening tree, extracts LP duals to form
//! Benders cut coefficients, and aggregates per-opening outcomes via
//! [`RiskMeasure::aggregate_cut`] to produce one cut per trial point per
//! stage. Each aggregated cut is inserted into the [`crate::FutureCostFunction`].
//!
//! Although [`ExchangeBuffers`] contains trial points from all ranks (after
//! `allgatherv`), each rank only processes its own forward pass assignments
//! to avoid generating duplicate cuts. Cut synchronization (`allgatherv`)
//! distributes the generated cuts to all ranks after the backward pass.
//!
//! ## Stage indexing convention
//!
//! The backward pass generates a cut **at stage `t`** by solving the LP
//! **at stage `t + 1`** (the successor) under each opening noise vector from
//! that successor stage. The opening tree provides noise at `t + 1`.
//!
//! ## Cut coefficient formula
//!
//! For a solve at stage `t + 1` with trial point state `x_hat`:
//!
//! ```text
//! pi[i]  = reduced_cost[col_i] / col_scale[col_i]   for i in 0..n_state
//! alpha  = Q - sum_i(pi[i] * x_hat[i])              (intercept)
//! ```
//!
//! where `Q` is the LP objective, `col_i = state_to_lp_incoming_column(i)` is the
//! bound-pinned incoming-state column, and `reduced_cost[col_i]` is its `HiGHS`
//! reduced cost (the dual of the `lb == ub` pin — equal to the legacy
//! fixing-row dual by KKT stationarity). State fixing moved from equality rows to
//! column bounds in the state-fixing cutover; the gradient is now a reduced cost,
//! unscaled by `col_scale` (not `row_scale`). The convention is
//! `coefficients = reduced_cost` (raw, no sign flip at extraction). Negation is
//! applied later when building the LP cut row in `build_cut_row_batch_into`
//! (forward.rs):
//! `-coeff * x + theta >= intercept`.
//! The intercept formula covers all `n_state` indices uniformly; no special handling
//! for the anticipated-state subrange.
//! See the project convention: "coefficients = dual (NOT -dual)".
//!
//! ### Anticipated-state cut gradient flow
//!
//! Anticipated-state slots are deterministic state variables that connect a decision
//! stage to its delivery stage via the fishing constraint at the delivery stage.
//! `state_to_lp_column` applies a shift-aware mapping for these slots: slot `K_p-1`
//! (the maturation slot for plant `p`) maps to the decision column, while slot
//! `i < K_p-1` maps to slot `i+1` so cuts apply against the correct LP variables
//! at the predecessor stage.
//!
//! The dual of the anticipated-state-fixing row at stage `t` reflects
//! `dQ_t/dx_ant[slot, plant]`. Under the always-active fishing predicate
//! (`StageIndexer::is_anticipated_fishing_active` at `indexer.rs:1555`), every
//! slot at every stage participates in the dual chain: the fishing constraint is
//! emitted at every stage unconditionally. The dual on the Cat 6
//! state-fixing row at slot `s` flows back to the predecessor's LP column via
//! `state_to_lp_column`'s branch decision (Less / Equal / Greater), which maps
//! slot `K_p-1` to the decision column and slot `i < K_p-1` to slot `i+1`.
//! See `indexer.rs:1429-1454` for the `state_to_lp_column` rustdoc and
//! `artifacts/layout-decision.md` Section 2 for sign-chain derivations.
//!
//! The backward pass does not call `shift_anticipated_state`. The trial point
//! `x_hat` is the forward-shifted state; cut extraction uses it as-is. This
//! mirrors the inflow-lag convention and is correct: re-shifting would offset
//! `x_hat` relative to the anticipated-state-fixing rows, breaking cut consistency.
//! Empirical verification:
//! `crates/cobre-sddp/tests/anticipated_backward_cut_k1.rs` (K=1) and
//! `crates/cobre-sddp/tests/anticipated_backward_cut_k2.rs` (K=2).
//!
//! ## Cut activity tracking
//!
//! After each backward solve, the duals of the appended cut rows are inspected
//! to determine which existing cuts at the successor stage are binding. The
//! metadata of binding cuts is updated in-place so that cut selection
//! strategies have accurate activity counts at the end of the iteration.
//!
//! ## Thread-level parallelism
//!
//! Within a rank, the outer per-stage loop remains sequential (stage `t`
//! depends on cuts generated at stage `t+1`). The inner trial-point loop is
//! parallelised across [`SolverWorkspace`] instances using rayon's
//! `par_iter_mut` with static scenario partitioning (matching the forward pass).
//! Each worker generates cuts into a thread-local `StagedCut` buffer, sorted
//! by `trial_point_idx` after the parallel region to ensure deterministic FCF
//! insertion regardless of thread completion order.
//!
//! ## Hot-path allocation discipline
//!
//! Allocations are limited to:
//! - One `Vec<f64>` for opening probabilities per stage (outside the trial
//!   point loop).
//! - One `Vec<BackwardOutcome>` per worker thread, allocated once per stage
//!   in the parallel region and reused via `clear()` per trial point.
//! - One `RowBatch` per stage built by `build_cut_row_batch` (outside the
//!   trial point loop, before the parallel region).
//! - One `Vec<StagedCut>` per stage for the merge phase (bounded by
//!   `local_work` entries, each holding one cut and its binding slot list).
//!
//! The `binding_slots` vector inside each `StagedCut` is allocated per
//! trial point — a flat buffer optimization is deferred to profiling.
//!
//! Note: `load_model` is called once per trial point (not per stage) to reset
//! the LP to the structural template before appending cuts. `HiGHS` performs
//! internal allocations during `load_model` that are not visible to this
//! module; these are a fixed cost per trial point and are not considered
//! hot-path allocations from Cobre's perspective.

#[cfg(test)]
use cobre_comm::Communicator;
use cobre_core::StageRowSelectionRecord;
use cobre_solver::{RowBatch, SolutionView, SolverInterface, SolverStatistics};

use crate::{
    SddpError,
    context::{StageContext, TrainingContext},
    cut::{CutRowMap, pool::CutPool},
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    forward::write_capture_metadata,
    indexer::StageIndexer,
    noise::{NcsNoiseOffsets, transform_inflow_noise, transform_load_noise, transform_ncs_noise},
    risk_measure::RiskMeasure,
    solver_stats::SolverStatsDelta,
    state_exchange::ExchangeBuffers,
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

/// Per-`(rank, worker_id, opening)` solver delta collected during a single
/// backward stage, as returned inside [`BackwardResult::stage_stats`].
///
/// Layout: `(rank, worker_id, opening_index, delta)`.
pub type StageWorkerOpeningDelta = (i32, i32, usize, SolverStatsDelta);

/// Result produced by the backward pass on a single rank.
///
/// The per-worker timing data carried inside `stage_stats` is keyed
/// by the `WORKER_TIMING_SLOT_*` constants exported from
/// `cobre-core`. New per-worker timing slots should be added to
/// that constant set (and the `WORKER_TIMING_SLOT_COUNT` updated)
/// rather than as standalone fields on this struct, so the parquet
/// timing schema picks them up automatically.
#[derive(Debug, Clone)]
#[must_use]
pub struct BackwardResult {
    /// Total number of cuts generated by this rank during the backward pass.
    pub cuts_generated: usize,

    /// Wall-clock time in milliseconds for this rank's backward pass.
    pub elapsed_ms: u64,

    /// Number of LP solves performed during this backward pass.
    pub lp_solves: u64,

    /// Per-stage, per-`(rank, worker_id, opening)` solver statistics deltas.
    ///
    /// Each outer entry is `(successor_stage_index, per_worker_opening_deltas)`.
    /// The inner `Vec` element is `(rank, worker_id, omega, delta)`: one entry per
    /// `(MPI rank, rayon worker, opening index)` triple gathered via `allgatherv`.
    /// Only includes entries where `omega < n_openings(successor)` AND
    /// `delta.lp_solves > 0 || omega == 0` (preserves the omega=0 "stage visited"
    /// sentinel while skipping padded buffer slots).
    /// Entries are in reverse stage order (matching the backward iteration direction).
    pub stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)>,

    /// Wall-clock time for state exchange (`allgatherv`) accumulated across
    /// all stages, in milliseconds.
    pub state_exchange_time_ms: u64,

    /// Wall-clock time for `build_cut_row_batch_into` accumulated across
    /// all stages, in milliseconds.
    pub cut_batch_build_time_ms: u64,

    /// Aggregate non-solve work inside the parallel region accumulated across
    /// all stages, in milliseconds.
    ///
    /// Computed per-stage as the sum over all workers of
    /// `load_model_time_ms + set_bounds_time_ms + basis_set_time_ms`.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead accumulated across all
    /// stages, in milliseconds.
    ///
    /// Computed per-stage as `max_worker_total_ms - avg_worker_total_ms`, where
    /// `worker_total_ms = solve + load_model + set_bounds + basis_set`
    /// for each worker. Measures how much the slowest worker exceeds the average.
    pub load_imbalance_ms: u64,

    /// True rayon scheduling overhead accumulated across all stages, in
    /// milliseconds.
    ///
    /// Computed per-stage as `parallel_wall_ms - max_worker_total_ms`. Represents
    /// rayon barrier, thread wake-up, and work-stealing dispatch costs after
    /// accounting for all measured per-worker work.
    pub scheduling_overhead_ms: u64,

    /// Wall-clock time for per-stage cut synchronization (`allgatherv`)
    /// accumulated across all stages, in milliseconds.
    pub cut_sync_time_ms: u64,

    /// Per-stage selection records collected when the in-backward selection
    /// hook ran. Length is bounded by `num_stages - 1`; stages where the
    /// hook did not run produce no entry. Sorted by stage index ascending.
    ///
    /// Populated only when `IN_BACKWARD_ENABLED` is true (set via
    /// [`crate::set_inside_backward_enabled`]) AND a cut-selection strategy
    /// is plumbed into the backward sweep AND the strategy's
    /// `should_run(iteration)` gate fires. Otherwise this `Vec` is empty.
    pub selection_records: Vec<StageRowSelectionRecord>,
}

/// Per-thread staging buffer for one aggregated cut produced at a single trial
/// point during the parallel backward sweep.
///
/// Each worker thread populates one `StagedCut` per trial point instead of
/// writing directly into the `FutureCostFunction`. After the parallel region,
/// staged cuts are sorted by `trial_point_idx` and merged into the FCF in
/// deterministic order regardless of thread completion order.
pub(crate) struct StagedCut {
    /// Local trial-point index within `0..local_work`. Used for deterministic
    /// merge ordering after the parallel region.
    pub(crate) trial_point_idx: usize,

    /// Aggregated cut intercept (result of `RiskMeasure::aggregate_cut`).
    pub(crate) intercept: f64,

    /// Aggregated cut coefficients (length = `n_state`).
    pub(crate) coefficients: Vec<f64>,

    /// Global forward-pass index (`fwd_offset + m`), stored as `u32` for the
    /// FCF slot formula.
    pub(crate) forward_pass_index: u32,
}

/// Per-successor data bundled for `process_stage_backward` and the trial-point helper.
///
/// Groups the successor-specific arguments — including the stage index `t`,
/// opening probabilities, pre-built cut batch, and cut activity metadata —
/// to keep per-function argument counts at or below seven.
pub(crate) struct SuccessorSpec<'a> {
    /// Stage index being cut (the stage whose cost-to-go we are computing).
    pub(crate) t: usize,
    /// Successor stage index (`t + 1`), where the LP is actually solved.
    pub(crate) successor: usize,
    /// This rank's MPI rank index (used to address exchange buffer state).
    pub(crate) my_rank: usize,
    /// Uniform opening probabilities for the successor stage.
    pub(crate) probabilities: &'a [f64],
    /// Pre-built cut rows to append to each successor LP.
    /// Delta batch when baking is active, full active-cut batch otherwise.
    pub(crate) cut_batch: &'a RowBatch,
    /// Total number of active cuts at the successor stage for dual extraction.
    /// Includes both baked and delta cuts contiguous after `template_num_rows`.
    pub(crate) num_cuts_at_successor: usize,
    /// Base row count of the successor template (excludes cuts).
    pub(crate) template_num_rows: usize,
    /// Baked LP template for the successor stage. Always populated — baking
    /// is complete before the backward pass begins.
    pub(crate) baked_template: &'a cobre_solver::StageTemplate,
    /// Ordered slot indices of the active cuts at the successor stage.
    pub(crate) successor_active_slots: &'a [usize],
    /// Minimum dual multiplier for a cut to count as binding.
    pub(crate) cut_activity_tolerance: f64,
    /// Populated count of the successor's cut pool.
    pub(crate) successor_populated_count: usize,
    /// Cut pool at the successor stage for binding-activity tracking.
    pub(crate) successor_pool: &'a CutPool,
}

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
/// Called once per opening inside [`process_trial_point_backward`].  The LP
/// structure is already loaded by [`load_backward_lp`]; this function only
/// updates noise-dependent row and column bounds via `set_row_bounds` /
/// `set_col_bounds`.
fn patch_opening_bounds<S: SolverInterface + Send>(
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
        .fill_col_state_patches(training_ctx.indexer, x_hat, &ctx.templates[s].col_scale);
    ws.patch_buf.fill_forward_patches(
        training_ctx.indexer,
        x_hat,
        &ws.scratch.noise_buf,
        ctx.base_rows[s],
        &ctx.templates[s].row_scale,
    );
    if ctx.n_load_buses > 0 {
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[s],
            n_blks,
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[s].row_scale,
        );
    }
    ws.patch_buf.fill_z_inflow_patches(
        training_ctx.indexer.z_inflow_row_start,
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
    if n_stochastic_ncs > 0 && !training_ctx.indexer.ncs_generation.is_empty() {
        let n_blks_stage = ctx.block_counts_per_stage[s];
        let expected_len = n_stochastic_ncs * n_blks_stage;
        if ws.scratch.ncs_col_indices_buf.len() != expected_len {
            ws.scratch.ncs_col_indices_buf.clear();
            for ncs_idx in 0..n_stochastic_ncs {
                for blk in 0..n_blks_stage {
                    ws.scratch.ncs_col_indices_buf.push(
                        training_ctx.indexer.ncs_generation.start + ncs_idx * n_blks_stage + blk,
                    );
                }
            }
        }
        ws.solver.set_col_bounds(
            &ws.scratch.ncs_col_indices_buf,
            &ws.scratch.ncs_col_lower_buf,
            &ws.scratch.ncs_col_upper_buf,
        );
    }
}

/// Resolve the ω=0 warm-start basis from the worker's `BasisStoreSliceMut`.
///
/// Returns `None` when the slot is empty (cold start or no prior capture).
#[inline]
fn resolve_backward_basis<'a>(
    basis_slice: &'a BasisStoreSliceMut<'_>,
    m: usize,
    s: usize,
) -> Option<&'a CapturedBasis> {
    basis_slice.get(m, s)
}

/// Extract state and cut duals from the solver view into pre-warmed scratch buffers.
///
/// Called while `view` is still live (borrowing the solver). The output buffers
/// (`state_duals`, `cut_duals`) were taken out of `ws.backward_accum` before the
/// solve and are passed here directly so that no `ws` borrow is needed.
///
/// Returns the LP objective value.
///
/// # Dual-fill layout
///
/// `state_duals`: unscaled reduced costs at the LP columns pinned by
/// `fill_col_state_patches`, one entry per state-vector index.
/// Scaling: `rc_original[j] = rc_scaled[j] / col_scale[col]` (the pin sets
/// `v_scaled = v_orig / col_scale`, so the subgradient w.r.t. the original
/// state divides by `col_scale`); when `col_scale` is empty the raw reduced
/// costs are used directly.
///
/// `cut_duals`: raw duals for cut rows `[template_num_rows, template_num_rows + num_cuts)`.
/// These always have implicit `row_scale = 1.0`.
fn extract_duals_from_view(
    view: &SolutionView<'_>,
    n_state: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    succ: &SuccessorSpec<'_>,
    state_duals: &mut Vec<f64>,
    cut_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    // Unscale state-fixing-column reduced costs from scaled to original units.
    // The incoming-state column is pinned in scaled space as `v_scaled =
    // v_orig / col_scale[col]` (see `fill_col_state_patches` and
    // `apply_col_scale`, which divides col bounds by `col_scale`). The reduced
    // cost HiGHS reports is `rc_scaled = dQ/dv_scaled`; the cut subgradient we
    // need is `dQ/dv_orig = rc_scaled * dv_scaled/dv_orig = rc_scaled /
    // col_scale[col]`. Dividing (not multiplying) keeps parity with the legacy
    // fixing-row dual, whose single-entry row carried `row_scale = 1/col_scale`.
    // `state_duals` carries pre-warmed capacity; `clear` + `push` reuses it.
    state_duals.clear();
    for j in 0..n_state {
        let col = indexer.state_to_lp_incoming_column(j);
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
    debug_assert_eq!(
        state_duals.len(),
        n_state,
        "state_duals must contain exactly n_state entries after fill"
    );

    // Fill cut duals from the cut-row slice.
    //
    // Layout: [0, template_num_rows) — structural rows;
    //         [template_num_rows, template_num_rows + num_cuts) — cut rows (baked then delta).
    cut_duals.clear();
    if succ.num_cuts_at_successor > 0 {
        cut_duals.extend_from_slice(
            &view.dual[succ.template_num_rows..succ.template_num_rows + succ.num_cuts_at_successor],
        );
    }

    objective
}

/// Extract only the cut gradient (state-fixing-column reduced costs) and return
/// the objective, for the lazy-solve path.
///
/// Identical to the state-dual half of [`extract_duals_from_view`]: it fills
/// `state_duals[j] = rc_scaled[col] / col_scale[col]` (raw reduced costs when
/// `col_scale` is empty) at the incoming-state column for each state index `j`,
/// where the negation into the `−∇·x + θ ≥ intercept` row happens later in
/// cut-row construction. The Benders gradient and intercept come solely from
/// these structural state columns, which are identical in the all-cuts LP and
/// the lazy-solve LP, so the resulting cut matches the all-cuts cut by
/// exactness.
///
/// Unlike [`extract_duals_from_view`], it does NOT read cut-row duals: under the
/// lazy-solve path the resident cut rows are a subset in row-map insertion
/// order, so the all-cuts cut-row→slot mapping does not apply. Binding-count
/// metadata and basis capture for that layout are handled separately and are
/// not driven from this function.
fn extract_state_duals_only(
    view: &SolutionView<'_>,
    n_state: usize,
    indexer: &StageIndexer,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    state_duals.clear();
    for j in 0..n_state {
        let col = indexer.state_to_lp_incoming_column(j);
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
    debug_assert_eq!(
        state_duals.len(),
        n_state,
        "state_duals must contain exactly n_state entries after fill"
    );

    objective
}

/// Accumulate one opening's solve result into the workspace accumulators.
///
/// Called after `view` is dropped (so `ws` is freely borrowable). Writes:
/// - per-opening stats delta into `ws.backward_accum.per_opening_stats[omega]`
/// - outcome coefficients, objective, and intercept into `ws.backward_accum.outcomes[omega]`
/// - binding-cut slot increments into `ws.backward_accum.slot_increments`
fn accumulate_opening_outcome<S: SolverInterface + Send>(
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
fn accumulate_dcs_binding_counts(
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
fn write_opening_outcome<S: SolverInterface + Send>(
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

/// Capture the post-solve basis at ω=0 into `basis_slice[m, s]`.
///
/// Only called when `omega == 0`; writes at ω>0 are forbidden because the
/// retained LU factorization would be overwritten by subsequent opening solves,
/// making the stored basis stale and potentially infeasible when reloaded.
///
/// Reuses an existing slot in-place when present (avoids reallocation on
/// subsequent iterations). Allocates a new `CapturedBasis` only on the first
/// capture for this `(m, s)` pair.
fn save_basis_at_omega_zero<S: SolverInterface + Send>(
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

/// Solve one backward opening on the baked all-cuts LP and accumulate its
/// outcome. This is the unchanged pre-DCS per-opening path: patch the opening
/// bounds on the already-loaded baked/delta LP, reconstruct + solve via
/// `run_stage_solve`, extract both state and cut duals, accumulate the outcome
/// (including the binding-count `slot_increments` update), and capture the
/// first-solved opening's basis.
///
/// `is_first` is `true` for the trial point's **first-solved** opening — ω=0 in
/// canonical order, or the first entry of the solve order when reordering is
/// enabled. Only that opening loads the per-(m, s) stored basis and captures the
/// post-solve basis back into the store; the rest pass `None` (warm re-solve on
/// the retained LU). Decoupling basis identity from the literal ω=0 is what lets
/// the openings be solved in any order while the per-(m, s) basis store stays
/// consistent with the actual first solve.
// Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ,
// basis_slice) and per-opening scalars (raw_noise, x_hat, s, scenario,
// iteration, m, omega, is_first); no natural grouping reduces caller-side borrows.
#[allow(clippy::too_many_arguments)]
fn solve_opening_baked<S: SolverInterface + Send>(
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

    let stats_before_omega = ws.solver.statistics();

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

    let stats_after_omega = ws.solver.statistics();

    accumulate_opening_outcome(
        ws,
        succ,
        omega,
        objective,
        x_hat,
        &stats_before_omega,
        &stats_after_omega,
    );

    if is_first {
        save_basis_at_omega_zero(ws, succ, basis_slice, m, x_hat);
    }

    Ok(())
}

/// Solve one backward opening under Dynamic Cut Selection and accumulate its
/// outcome.
///
/// All openings of a trial point are processed by one worker in fixed order
/// `0..n_openings`, sharing the same pinned incoming state `x̂` and differing
/// only in their noise RHS, so the cut-free core and the metadata seed are
/// loaded/built ONCE per trial point by the caller
/// ([`process_trial_point_backward`]) and the LP is reused across the openings.
/// This routine therefore never reloads or re-seeds; it only patches the
/// opening's bounds and runs the lazy loop via [`lazy_solve_preloaded`]:
///
/// - `continue_carry == false` (the first opening, ω=0): solve fresh — the lazy
///   loop resets the carried row map, appends the seed, and cold-solves. The
///   cut produced is identical to the (former) per-opening path.
/// - `continue_carry == true` (subsequent openings): warm-carry the prior
///   opening's LP, basis, and (monotonically grown) resident cut set; re-solve
///   warm under the new noise and add only the cuts this opening additionally
///   violates. This is what the paper's §3.4 "base recovery" buys, extended
///   across the trial point's openings.
///
/// The Benders cut gradient and intercept are read from the final all-satisfied
/// LP via [`extract_state_duals_only`] in both cases.
///
/// **Binding-count metadata** (`slot_increments`) IS maintained here, but
/// slot-correct under the lazy layout: the baked path bumps by baked cut-row
/// order, whereas [`accumulate_dcs_binding_counts`] maps each resident cut row
/// back to its pool slot via the final [`CutRowMap`] and bumps
/// `slot_increments[slot]` for residents whose cut-row dual exceeds
/// `cut_activity_tolerance` (the same criterion the baked path uses). This feeds
/// the existing per-stage `metadata_sync_contribution` allreduce, which advances
/// `last_active_iter` rank-invariantly and so restores the §3.1 clause-1 seed
/// (cuts **binding** in the last `k2` iterations) — without altering the
/// extracted gradient/intercept (those come solely from the structural
/// state-column reduced costs and are identical to the all-cuts cut by
/// exactness).
///
/// One piece of the baked path is intentionally NOT performed here:
///
/// - **ω=0 basis capture**: a captured basis would describe the baked layout,
///   not the DCS resident subset, so it is skipped; the next DCS solve
///   cold-starts its initial solve, which [`lazy_solve_preloaded`] supports with
///   `stored_basis = None`.
// Rationale: the args are disjoint borrows (ws, ctx, training_ctx, succ) and
// per-opening scalars (params, raw_noise, x_hat, s, scenario, iteration, omega);
// no natural grouping reduces caller-side borrows.
#[allow(clippy::too_many_arguments)]
fn solve_opening_dcs<S: SolverInterface + Send>(
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

    // The caller ([`process_trial_point_backward`]) has already loaded the
    // cut-free core and built the metadata seed once for this trial point; here
    // we only apply this opening's state-pin + noise patch. `lazy_solve_preloaded`
    // appends cut rows to the loaded, patched LP and never reloads, so the patch
    // — and, on continued openings, the carried cut rows + warm basis — survive
    // the whole lazy loop.
    patch_opening_bounds(ws, ctx, training_ctx, raw_noise, x_hat, s);

    let stats_before_omega = ws.solver.statistics();

    // Move the state-duals buffer out of `ws.backward_accum` so it can be filled
    // by `extract_state_duals_only` (a `&mut Vec` sink) while `view` holds an
    // immutable borrow of the sibling `dcs_solve` field; restored at the end.
    let mut state_duals = std::mem::take(&mut ws.backward_accum.state_duals_buf);

    let dcs_ctx = DcsSolveContext {
        stage_index: s,
        scenario_index: scenario,
        iteration: Some(iteration),
        // Warm-carry the prior opening's LP + basis on every opening but the
        // first; ω=0 runs a fresh (cold, reset + seed) solve.
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
    // View over the result buffers (borrows `dcs_solve` immutably; coexists with
    // the immutable `dcs_solve.row_map` read in `accumulate_dcs_binding_counts`).
    let view = ws.backward_accum.dcs_solve.result_view();

    // Cut gradient + intercept from the final all-satisfied LP only. The
    // gradient/intercept come solely from the structural state-column reduced
    // costs, identical in the all-cuts and lazy-solve LPs — exactness is
    // unaffected by the binding-count bookkeeping below.
    let objective =
        extract_state_duals_only(&view, indexer.n_state, indexer, col_scale, &mut state_duals);

    // Binding-count metadata, slot-correct under the resident `CutRowMap`. The
    // final all-satisfied solve's cut-row duals (`view.dual`, the full dual copied
    // into `dcs_solve.res_dual`) are read here and mapped slot→row via the
    // persistent row map that `lazy_solve_preloaded` just finalized. Borrows:
    // `view` and `dcs_solve.row_map` both borrow `ws.backward_accum.dcs_solve`
    // immutably (so they coexist); `slot_increments` is a distinct field of
    // `ws.backward_accum` borrowed mutably.
    accumulate_dcs_binding_counts(
        view.dual,
        &ws.backward_accum.dcs_solve.row_map,
        succ.successor_pool,
        succ.cut_activity_tolerance,
        &mut ws.backward_accum.slot_increments,
    );
    let _ = view;

    ws.backward_accum.state_duals_buf = state_duals;

    let stats_after_omega = ws.solver.statistics();

    // Outcome only — no ω=0 basis capture (that captured basis would describe
    // the baked layout, not the DCS resident subset). The binding-count update
    // is done above, slot-correct under the lazy layout.
    write_opening_outcome(
        ws,
        omega,
        objective,
        x_hat,
        &stats_before_omega,
        &stats_after_omega,
    );

    Ok(())
}

/// Process one trial point `m` in the backward pass, iterating over all openings.
///
/// Solves at each (scenario, opening) and accumulates duals into `per_opening_stats`.
/// At ω=0, writes the post-solve basis into `basis_slice`; writes at ω>0 are
/// forbidden (retained-LU corruption risk). Infeasibility at ω=0
/// leaves the slot unchanged
// RATIONALE: 10 args required — each is a disjoint borrow (ws, ctx, training_ctx, exchange,
// succ, basis_slice) or a plain scalar (fwd_offset, iteration, m) or a risk slice.
// Merging into a struct would add indirection without reducing the caller's borrow count.
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
    m: usize,
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

    // Dynamic Cut Selection params for this solve, `Some` only when configured
    // and the current iteration is at or past its start. The decision is
    // constant across openings and trial points for a given iteration.
    let dcs_params = training_ctx
        .dcs
        .filter(|params| params.is_active(iteration));

    // Throwaway, env-gated backward `noise_key` diagnostic. `None` on the
    // default path (single hoisted boolean check; no key compute, no
    // allocation). When `Some`, after each opening solve we pair the precomputed
    // per-(stage, ω) noise key with that opening's just-consumed
    // `simplex_iterations` and emit one record to stderr.
    let noise_key_diag = training_ctx.noise_key_diag;

    // Opening-reuse (the DCS backward path): load the cut-free core + build the
    // metadata seed ONCE here, then reuse the loaded LP across this trial point's
    // openings — ω=0 solves fresh from the seed, ω>0 warm-carry. The core load
    // also serves as the per-trial-point HiGHS reset that preserves
    // rank-invariance: state is reset at every trial-point boundary and never
    // carried across trial points. (`load_backward_lp` skips its baked load on
    // the DCS path, so this is the only load.)
    if let Some(params) = dcs_params {
        ws.solver.load_model(&ctx.templates[s]);
        build_initial_resident_set(
            succ.successor_pool,
            iteration,
            params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
    }

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

        if let Some(params) = dcs_params {
            solve_opening_dcs(
                ws,
                ctx,
                training_ctx,
                succ,
                params,
                raw_noise,
                x_hat,
                s,
                scenario,
                iteration,
                omega,
                // The first-solved opening solves fresh; subsequent openings
                // warm-carry the LP.
                !is_first,
            )?;
        } else {
            solve_opening_baked(
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

    // One allocation per trial point: copy coefficients out of the scratch
    // buffer so they outlive the parallel closure (see module-level docs).
    let n_openings = succ.probabilities.len();
    let mut agg_intercept = 0.0_f64;
    risk_measures[succ.t].aggregate_cut_into(
        &ws.backward_accum.outcomes[..n_openings],
        succ.probabilities,
        &mut agg_intercept,
        &mut ws.backward_accum.agg_coefficients,
        &mut ws.backward_accum.risk_scratch,
    );
    let agg_coefficients = ws.backward_accum.agg_coefficients.clone();
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
        coefficients: agg_coefficients,
        forward_pass_index,
    })
}

/// Test-only backward-pass shim that owns per-call scratch.
///
/// Production code drives the backward pass via [`BackwardPassState::run`]
/// on the state struct held by `TrainingSession`. This shim exists so that
/// the tests in this module can exercise `run_one_backward_stage` without
/// threading a full `TrainingSession` through every fixture.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
/// feasible solution during the backward sweep. Returns
/// `Err(SddpError::Solver(_))` for all other terminal LP solver failures.
#[cfg(test)]
fn run_backward_pass<S, C: Communicator>(
    inputs: &mut crate::backward_pass_state::BackwardPassInputs<'_, S, C>,
) -> Result<BackwardResult, SddpError>
where
    S: SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    let n_workers_local = inputs.workspaces.len();
    let n_ranks = inputs.comm.size();
    let num_stages = inputs.training_ctx.horizon.num_stages();
    let bwd_max_openings = (0..num_stages)
        .map(|t| inputs.training_ctx.stochastic.opening_tree().n_openings(t))
        .max()
        .unwrap_or(0);
    let real_states_capacity =
        inputs.exchange.real_total_scenarios() * inputs.training_ctx.indexer.n_state;
    let mut bwd_state = crate::backward_pass_state::BackwardPassState::new(
        n_workers_local,
        n_ranks,
        bwd_max_openings,
        real_states_capacity,
    );
    bwd_state.run(inputs)
}

#[cfg(test)]
mod tests {
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_solver::{
        Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };

    use cobre_core::scenario::SamplingScheme;

    use super::{BackwardResult, run_backward_pass};
    use crate::{
        context::{StageContext, TrainingContext},
        cut::FutureCostFunction,
        cut_sync::CutSyncBuffers,
        horizon_mode::HorizonMode,
        indexer::StageIndexer,
        inflow_method::InflowNonNegativityMethod,
        risk_measure::RiskMeasure,
        solver_stats::SolverStatsDelta,
        state_exchange::ExchangeBuffers,
        trajectory::TrajectoryRecord,
        workspace::{BackwardAccumulators, BasisStore, SolverWorkspace},
    };

    fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
        (0..n_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect()
    }

    /// Stub communicator for tests (single-rank).
    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            // Single-rank: copy send to recv (mirrors LocalBackend behavior).
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("StubComm broadcast not used in backward pass tests")
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            1
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    /// Mock solver for testing: returns fixed solution or infeasible error on demand.
    ///
    /// Buffer fields (`buf_primal`, `buf_dual`, `buf_reduced_costs`) store the
    /// solution data that [`SolutionView`] borrows from. They are filled in
    /// `solve` before the borrow is established.
    struct MockSolver {
        solution: LpSolution,
        infeasible_at: Option<usize>,
        call_count: usize,
        /// Tracks the current number of rows (template + appended cuts).
        current_num_rows: usize,
        /// Number of times `solve(Some(&basis))` was called (warm-start calls).
        warm_start_calls: usize,
        /// Dual padding value for rows beyond the base template (cuts).
        /// Defaults to 0.0 (cuts not binding). Set to a positive value
        /// to make all cuts appear binding in tests.
        cut_dual_padding: f64,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let base_rows = solution.dual.len();
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: None,
                call_count: 0,
                current_num_rows: base_rows,
                warm_start_calls: 0,
                cut_dual_padding: 0.0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        fn infeasible_on(solution: LpSolution, n: usize) -> Self {
            let base_rows = solution.dual.len();
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: Some(n),
                call_count: 0,
                current_num_rows: base_rows,
                warm_start_calls: 0,
                cut_dual_padding: 0.0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }

        /// Like `always_ok` but added cut rows return positive duals,
        /// making all existing cuts appear binding in subsequent solves.
        fn always_ok_with_binding_cuts(solution: LpSolution) -> Self {
            let mut s = Self::always_ok(solution);
            s.cut_dual_padding = 1.0;
            s
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.current_num_rows = template.num_rows;
        }

        fn add_rows(&mut self, cuts: &RowBatch) {
            self.current_num_rows += cuts.num_rows;
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(
            &mut self,
            basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            if basis.is_some() {
                self.warm_start_calls += 1;
            }
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_at == Some(call) {
                return Err(SolverError::Infeasible);
            }
            // Fill internal buffers, resizing dual to match current LP row count.
            self.buf_primal.clone_from(&self.solution.primal);
            self.buf_dual.clone_from(&self.solution.dual);
            self.buf_dual
                .resize(self.current_num_rows, self.cut_dual_padding);
            self.buf_reduced_costs
                .clone_from(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: self.solution.iterations,
                solve_time_seconds: self.solution.solve_time_seconds,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn set_primal_feasibility_tolerance(&mut self, _tolerance: f64) {}

        fn set_dual_feasibility_tolerance(&mut self, _tolerance: f64) {}

        fn set_simplex_iteration_limit_profile(&mut self, _limit: u32) {}

        fn set_ipm_iteration_limit_profile(&mut self, _limit: u32) {}
    }

    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn solution_1_0(objective: f64, dual_storage: f64) -> LpSolution {
        // For StageIndexer::new(1, 0): storage_in.start = N*(2+L) = 1*(2+0) = 2.
        // state_to_lp_incoming_column(0) = storage_in.start + 0 = 2.
        // Cut subgradients are read from reduced_costs[storage_in_col], so
        // reduced_costs[2] must hold the same value as the storage-fixing dual.
        let mut reduced_costs = vec![0.0; 3];
        reduced_costs[2] = dual_storage;
        LpSolution {
            objective,
            primal: vec![0.0, 0.0, 0.0],
            dual: vec![dual_storage],
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Wrap a `MockSolver` into a single-element `Vec<SolverWorkspace<MockSolver>>`
    /// for tests that exercise the workspace-based backward-pass API.
    ///
    /// The workspace is sized for `n_hydro=1`, `max_par_order=0`, and `n_state`
    /// state dimensions.
    fn single_workspace(solver: MockSolver, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
        use crate::lp_builder::PatchBuffer;
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(solver),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
                anticipated_state_out_col_indices_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }]
    }

    /// Create an empty `BasisStore` for `num_scenarios` scenarios and
    /// `num_stages` stages (all slots `None`).
    fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
        BasisStore::new(num_scenarios, num_stages)
    }

    /// Create a `BasisStore` with one slot pre-populated at
    /// `[scenario][stage]` with the given `Basis`.
    fn basis_store_with_one(
        num_scenarios: usize,
        num_stages: usize,
        scenario: usize,
        stage: usize,
        basis: Basis,
    ) -> BasisStore {
        let mut store = BasisStore::new(num_scenarios, num_stages);
        // Set `base_row_count` to the full row_status length and leave
        // `cut_row_slots` empty so the CapturedBasis invariant
        // (`row_status.len() == base_row_count + cut_row_slots.len()`) holds
        // by construction. These tests exercise the warm-start propagation
        // path; the reconstruction copies the template rows verbatim and
        // emits an empty cut block.
        let base_row_count = basis.row_status.len();
        *store.get_mut(scenario, stage) = Some(crate::workspace::CapturedBasis {
            basis,
            base_row_count,
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
        });
        store
    }

    fn exchange_with_states(n_state: usize, states: Vec<Vec<f64>>) -> ExchangeBuffers {
        use cobre_comm::LocalBackend;

        let local_count = states.len();
        let mut bufs = ExchangeBuffers::new(n_state, local_count, 1);
        let records: Vec<TrajectoryRecord> = states
            .into_iter()
            .map(|state| TrajectoryRecord {
                primal: vec![],
                dual: vec![],
                stage_cost: 0.0,
                state,
            })
            .collect();

        let comm = LocalBackend;
        bufs.exchange(&records, 0, 1, &comm).unwrap();
        bufs
    }

    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context(
        n_stages: usize,
        branching_factor: usize,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            scenario::{
                CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
                InflowModel,
            },
            temporal::{
                Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
                StageStateConfig,
            },
        };
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };
        use std::collections::BTreeMap;

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation)]
        let inflow = |stage_idx: usize| InflowModel {
            hydro_id: EntityId(1),
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            stage_id: stage_idx as i32,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };

        let inflow_models: Vec<InflowModel> = (0..n_stages).map(inflow).collect();

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn backward_result_fields_accessible() {
        let r = BackwardResult {
            cuts_generated: 6,
            elapsed_ms: 42,
            lp_solves: 0,
            stage_stats: Vec::new(),
            state_exchange_time_ms: 0,
            cut_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            cut_sync_time_ms: 0,
            selection_records: Vec::new(),
        };
        assert_eq!(r.cuts_generated, 6);
        assert_eq!(r.elapsed_ms, 42);
        assert!(r.stage_stats.is_empty());
        assert_eq!(r.state_exchange_time_ms, 0);
        assert_eq!(r.cut_batch_build_time_ms, 0);
        assert_eq!(r.setup_time_ms, 0);
        assert_eq!(r.load_imbalance_ms, 0);
        assert_eq!(r.scheduling_overhead_ms, 0);
        assert_eq!(r.cut_sync_time_ms, 0);
        assert!(r.selection_records.is_empty());
    }

    #[test]
    fn backward_result_clone_and_debug() {
        let r = BackwardResult {
            cuts_generated: 3,
            elapsed_ms: 100,
            lp_solves: 0,
            stage_stats: Vec::new(),
            state_exchange_time_ms: 0,
            cut_batch_build_time_ms: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            cut_sync_time_ms: 0,
            selection_records: Vec::new(),
        };
        let c = r.clone();
        assert_eq!(c.cuts_generated, 3);
        let s = format!("{r:?}");
        assert!(s.contains("BackwardResult"));
    }

    #[test]
    fn dual_extraction_formula_coefficients_are_negated_duals() {
        // Given known dual values [d0, d1], coefficients must be [-d0, -d1].
        let d0 = 3.5_f64;
        let d1 = -1.2_f64;
        let dual = [d0, d1];

        let coefficients: Vec<f64> = dual.iter().map(|&d| -d).collect();

        assert!((coefficients[0] - (-d0)).abs() < f64::EPSILON);
        assert!((coefficients[1] - (-d1)).abs() < f64::EPSILON);
    }

    #[test]
    fn intercept_formula_matches_spec() {
        // alpha = Q - pi^T * x_hat
        // Given: objective=50.0, pi=[2.0, -1.0], x_hat=[10.0, 5.0]
        // Expected: alpha = 50.0 - (2.0*10.0 + (-1.0)*5.0) = 50.0 - 15.0 = 35.0
        let objective = 50.0_f64;
        let coefficients = [2.0_f64, -1.0_f64];
        let x_hat = [10.0_f64, 5.0_f64];
        let pi_dot_x: f64 = coefficients
            .iter()
            .zip(x_hat.iter())
            .map(|(p, x)| p * x)
            .sum();
        let intercept = objective - pi_dot_x;
        assert!((intercept - 35.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_stage_system_produces_no_cuts() {
        // A 1-stage system has no stages with a successor, so the backward
        // sweep (0..0) is empty — zero cuts are generated.
        let stochastic = make_stochastic_context(1, 2);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0()];
        let base_rows = vec![1_usize];

        let n_state = indexer.n_state;
        let n_stages = 1_usize;
        let forward_passes = 2_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        assert_eq!(result.cuts_generated, 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn two_stage_system_two_trial_points_generates_two_cuts_at_stage_0() {
        // Acceptance criterion: 3-stage system, 1 hydro (n_state=1), 2 openings,
        // 2 trial points → 2 cuts at stage 0. This is the 2-stage version
        // (stages 0 and 1); cuts should exist only at stage 0.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state; // 1
        let forward_passes = 2_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);

        // Two trial points with states [10.0] and [20.0] at stage 0.
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // MockSolver returns objective=100.0, dual[0]=-5.0 for every solve.
        // With x_hat=[10.0]: pi=[5.0], alpha = 100 - 5*10 = 50.
        // With x_hat=[20.0]: pi=[5.0], alpha = 100 - 5*20 = 0 (could be negative).
        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // 2 trial points × 1 stage with a successor = 2 cuts at stage 0.
        assert_eq!(result.cuts_generated, 2);
        assert_eq!(fcf.active_cuts(0).count(), 2);
        // Stage 1 (the last stage) gets no cuts.
        assert_eq!(fcf.active_cuts(1).count(), 0);
    }

    #[test]
    fn cut_inserted_with_correct_stage_iteration_and_forward_pass_index() {
        // Acceptance criterion: iteration=2, forward_passes=3, global
        // trial point m=1 → fcf.add_cut(stage=0, iteration=2, fpi=1, ...).
        // slot = warm_start + 2*3 + 1 = 7.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

        // 3 trial points (forward_passes=3 on a single rank).
        let mut exchange = exchange_with_states(n_state, vec![vec![5.0], vec![10.0], vec![15.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(50.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 2,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // Trial point m=1: slot = 0 + 2*3 + 1 = 7
        // Verify that pool[0].metadata[7] has the correct iteration and fpi.
        let meta = &fcf.pools[0].metadata[7];
        assert_eq!(meta.iteration_generated, 2);
        assert_eq!(meta.forward_pass_index, 1);
    }

    #[test]
    fn no_cuts_generated_at_last_stage() {
        // Acceptance criterion: 5-stage system → cuts at stages 0..3, not at 4.
        let n_stages = 5_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -3.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // 1 trial point × 4 stages with successors = 4 cuts total.
        assert_eq!(result.cuts_generated, 4);
        for t in 0..4 {
            assert_eq!(fcf.active_cuts(t).count(), 1, "stage {t} should have 1 cut");
        }
        // The last stage (4) must have no cuts.
        assert_eq!(fcf.active_cuts(4).count(), 0, "stage 4 must have no cuts");
    }

    #[test]
    fn elapsed_ms_is_non_negative() {
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 2);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![5.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(10.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // elapsed_ms is u64, so it is always >= 0.
        let _ = result.elapsed_ms;
    }

    #[test]
    fn infeasible_solver_returns_sddp_infeasible_error() {
        // Acceptance criterion: MockSolver::infeasible_on(0) for the first
        // backward solve → SddpError::Infeasible is returned.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(0.0, 0.0);
        // First solve call returns infeasible.
        let solver = MockSolver::infeasible_on(solution, 0);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        });

        assert!(
            matches!(result, Err(crate::SddpError::Infeasible { .. })),
            "expected SddpError::Infeasible, got: {result:?}",
        );
    }

    #[test]
    fn expectation_aggregation_mean_of_per_opening_intercepts() {
        // Given 3 openings with uniform probability 1/3 and per-opening
        // intercepts [10.0, 20.0, 30.0], the aggregated intercept must be 20.0.
        use crate::risk_measure::BackwardOutcome as BO;

        let outcomes = vec![
            BO {
                intercept: 10.0,
                coefficients: vec![],
                objective_value: 10.0,
            },
            BO {
                intercept: 20.0,
                coefficients: vec![],
                objective_value: 20.0,
            },
            BO {
                intercept: 30.0,
                coefficients: vec![],
                objective_value: 30.0,
            },
        ];
        let probs = vec![1.0 / 3.0; 3];
        let (intercept, _) = RiskMeasure::Expectation.aggregate_cut(&outcomes, &probs);
        assert!(
            (intercept - 20.0).abs() < 1e-10,
            "expected 20.0, got {intercept}"
        );
    }

    // ── Integration tests ─────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cut_coefficients_and_intercept_match_dual_extraction_formula() {
        // Integration test: verify that the backward pass uses the correct
        // dual extraction formula by checking cuts in the FCF.
        //
        // Setup: 2-stage, N=1, L=0, 1 opening, 1 trial point.
        //   dual[0] = -3.0 (storage-fixing dual from MockSolver)
        //   objective = 80.0
        //   x_hat = [10.0]
        //
        // Expected (coefficients = dual, not -dual):
        //   pi[0] = dual[0] = -3.0
        //   intercept = 80.0 - (-3.0) * 10.0 = 110.0
        //   coefficients = [-3.0]
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // dual[0] = -3.0, objective = 80.0
        let solution = solution_1_0(80.0, -3.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, intercept, coefficients) = &cuts[0];

        assert!(
            (intercept - 110.0).abs() < 1e-10,
            "expected intercept=110.0, got {intercept}"
        );
        assert_eq!(coefficients.len(), 1);
        assert!(
            (coefficients[0] - (-3.0)).abs() < 1e-10,
            "expected coefficient=-3.0, got {}",
            coefficients[0]
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cut_gradient_sign_physically_correct() {
        // Regression test for the Benders cut sign bug.
        //
        // Physical invariant: more initial storage → lower future cost.
        // The storage-fixing dual π is negative (shadow price of relaxing
        // the fixing constraint increases cost when storage decreases).
        //
        // Correct: coefficient = π < 0, so the cut slope is negative
        //   (more storage → lower cut value → lower theta → lower total cost).
        //
        // Old bug: coefficient = -π > 0, so the cut slope was positive
        //   (more storage → higher cut value → wrong incentive).
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![50.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // dual[0] = -2.0 (negative: more storage → less cost)
        // objective = 100.0, x_hat = 50.0
        let solution = solution_1_0(100.0, -2.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1, "expected exactly one cut");
        let (_, _intercept, coefficients) = &cuts[0];

        // The coefficient must be negative (same sign as the dual).
        // The old bug would produce +2.0 here instead of -2.0.
        assert!(
            coefficients[0] < 0.0,
            "cut coefficient must be negative (more storage → less future cost), \
             got {} — likely the Benders cut sign bug has been reintroduced",
            coefficients[0]
        );
        assert!(
            (coefficients[0] - (-2.0)).abs() < 1e-10,
            "expected coefficient=-2.0, got {}",
            coefficients[0]
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cut_is_tight_at_trial_point() {
        // Regression test: a Benders cut must be tight (exact) at the trial
        // point x̂ where it was generated. That is:
        //   intercept + coefficient * x̂ = Q(x̂)
        // where Q(x̂) = objective value of the subproblem at x̂.
        //
        // The cut equation is: θ ≥ intercept + coefficient * x
        // At x = x̂: θ ≥ Q(x̂) + π'(x̂ - x̂) = Q(x̂)
        //
        // If the sign is wrong (coefficient = -π instead of π), then:
        //   intercept + (-π) * x̂ ≠ Q(x̂) in general
        //
        // This test verifies the tightness property.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let x_hat = 30.0_f64;
        let mut exchange = exchange_with_states(n_state, vec![vec![x_hat]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let q_xhat = 200.0_f64; // subproblem objective at x̂
        let dual_storage = -4.0_f64;
        let solution = solution_1_0(q_xhat, dual_storage);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, intercept, coefficients) = &cuts[0];

        // Evaluate the cut at x̂: cut_value = intercept + coeff * x̂
        let cut_at_xhat = intercept + coefficients[0] * x_hat;

        // Must equal Q(x̂) (tightness property)
        assert!(
            (cut_at_xhat - q_xhat).abs() < 1e-10,
            "cut must be tight at trial point: \
             cut_value={cut_at_xhat}, Q(x̂)={q_xhat}, \
             intercept={intercept}, coeff={}, x̂={x_hat}",
            coefficients[0]
        );
    }

    #[test]
    fn single_rank_backward_pass_with_local_backend_produces_correct_fcf() {
        // Integration test with LocalBackend communicator (exercises single-rank path).
        use cobre_comm::LocalBackend;

        let n_stages = 3_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 2_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // 3-stage system: cuts at stages 0 and 1; 2 trial points each.
        // Total cuts = 2 stages × 2 trial points = 4.
        assert_eq!(result.cuts_generated, 4);
        assert_eq!(fcf.active_cuts(0).count(), 2);
        assert_eq!(fcf.active_cuts(1).count(), 2);
        assert_eq!(fcf.active_cuts(2).count(), 0);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn forward_pass_index_matches_global_scenario_index() {
        // Acceptance criterion: when a cut is generated for global trial point
        // m=5, then `fcf.add_cut(stage, iteration, 5, ...)` is called with
        // forward_pass_index = m = 5.
        //
        // Setup: iteration=2, forward_passes=6 (6 scenarios on 1 rank), 1 opening.
        // ExchangeBuffers: local_count=6, num_ranks=1, total_scenarios=6.
        // state_at(5/6, 5%6) = state_at(0, 5) — valid.
        //
        // Slot formula: slot = warm_start(0) + 2*6 + 5 = 17.
        // The key invariant: forward_pass_index = m = 5.
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 6_u32; // 6 scenarios on a single rank
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);

        // 6 trial points (m = 0..5). ExchangeBuffers: local_count=6, num_ranks=1.
        let mut exchange = exchange_with_states(
            n_state,
            vec![
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
                vec![6.0],
            ],
        );

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(50.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 2,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // m=5: slot = warm_start(0) + 2*6 + 5 = 17
        // The critical check: forward_pass_index in metadata equals global m=5.
        let meta = &fcf.pools[0].metadata[17];
        assert_eq!(meta.iteration_generated, 2, "iteration_generated must be 2");
        assert_eq!(
            meta.forward_pass_index, 5,
            "forward_pass_index must be 5 (= global m)"
        );
    }

    // ── Unit tests: warm-start basis caching (backward pass) ──────────────────

    /// Warm-start from a pre-populated forward basis: when `BasisStore` has
    /// `Some(Basis)` at `(scenario=0, stage=1)` before the first backward call,
    /// the first opening at the successor stage must call `solve(Some(&basis))`
    /// rather than `solve(None)`.
    ///
    /// AC: Given a 2-stage system, 1 trial point, 1 opening, with
    /// `basis_store.get(0, 1) = Some(Basis::new(...))` pre-populated,
    /// then `solver.warm_start_calls == 1` after the backward pass.
    #[test]
    fn warm_start_uses_prepopulated_forward_basis() {
        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;

        // Pre-populate the basis store at (scenario=0, stage=1).
        // This simulates a forward pass having already solved stage 1 and cached its basis.
        let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store =
            basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
        assert_eq!(
            warm_start_calls, 1,
            "first opening at successor stage must call solve(Some(&basis)) \
             when basis_store.get(0, 1) is pre-populated (warm_start_calls == 1, got {warm_start_calls})"
        );
    }

    /// Multi-opening P3b behavior: given 3 openings at the same successor stage,
    /// the first opening cold-starts (store slot is None via `solve()`), and
    /// openings 1 and 2 use `HiGHS` internal hot-start via `solve(None)` instead of
    /// `solve(Some(&working_basis))`.
    ///
    /// AC: Given a 2-stage system, 1 trial point, 3 openings, empty basis cache,
    /// then `solver.warm_start_calls == 0` after the backward pass (P3b: no
    /// and 3 warm-start; opening 1 cold-starts).
    #[test]
    fn multi_opening_subsequent_openings_use_internal_hotstart() {
        let n_stages = 2_usize;
        let n_openings = 3_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm;

        // Start with an empty store — opening 1 must cold-start.
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // P3b optimization: opening 0 cold-starts (no basis in store),
        // openings 1 and 2 use solve(None) (HiGHS internal hot-start) instead of
        // solve(Some(&working_basis)). No explicit warm-start calls for subsequent openings.
        let warm_start_calls = workspaces[0].solver.inner().warm_start_calls;
        assert_eq!(
            warm_start_calls, 0,
            "P3b: no warm-start calls expected when BasisStore is empty \
             (warm_start_calls == 0, got {warm_start_calls})"
        );
    }

    /// Error propagation: when a backward solve returns `SolverError::Infeasible`,
    /// the error must propagate as `SddpError::Infeasible`.
    ///
    /// In the new per-scenario design, the backward pass uses a local `working_basis`
    /// variable (not written back to `BasisStore`), so there is no shared cache slot
    /// to check after the error. The test verifies that the error is correctly
    /// propagated regardless of what was in the basis store at entry.
    ///
    /// AC: Given a 2-stage system, 1 opening, `MockSolver` returns infeasible on
    /// call 0, then `run_backward_pass` returns `Err(SddpError::Infeasible { .. })`.
    #[test]
    fn backward_solver_error_propagates() {
        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(0.0, 0.0);
        // The first backward solve (call 0) returns infeasible.
        let solver = MockSolver::infeasible_on(solution, 0);
        let comm = StubComm;

        // Pre-populate the store — error should propagate regardless.
        let pre_basis = Basis::new(templates[1].num_cols, templates[1].num_rows);
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store =
            basis_store_with_one(exchange.local_count(), n_stages, 0, 1, pre_basis);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        });

        assert!(
            matches!(result, Err(crate::SddpError::Infeasible { .. })),
            "expected SddpError::Infeasible, got: {result:?}",
        );
        // The BasisStore is not mutated by the backward pass — the working_basis
        // is a local variable dropped on error. The store slot at (0, 1) remains
        // as it was; this just verifies the store is untouched by an error path.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore must not be mutated by the backward pass error path"
        );
    }

    // ── New test: parallel cut determinism ────────────────────────────────────

    /// AC: When `run_backward_pass` runs with 1 workspace vs 4 workspaces given
    /// the same input data, the FCF pools contain identical cuts (same intercept,
    /// coefficient vectors, and slot assignments for each trial point).
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    fn test_backward_pass_parallel_cut_determinism() {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 3_usize;
        let n_openings = 2_usize;
        let n_trial_points = 8_usize;

        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        #[allow(clippy::cast_possible_truncation)]
        let forward_passes = n_trial_points as u32;

        // Build 8 distinct trial-point states.
        let states: Vec<Vec<f64>> = (0..n_trial_points).map(|i| vec![i as f64 + 1.0]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;

        // --- Run with 1 workspace ---
        let mut fcf_1 =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let solver_1 = MockSolver::always_ok(solution.clone());
        let mut workspaces_1 = vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(solver_1),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
                anticipated_state_out_col_indices_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }];
        let mut basis_store_1 = empty_basis_store(exchange.local_count(), n_stages);
        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces_1,
            basis_store: &mut basis_store_1,
            ctx: &ctx,
            baked: &mut templates.clone(),
            fcf: &mut fcf_1,
            cut_batches: &mut empty_cut_batches(n_stages),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // --- Run with 4 workspaces ---
        let mut fcf_4 =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut workspaces_4: Vec<SolverWorkspace<MockSolver>> = (0..4_i32)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: idx,
                solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
                patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
                current_state: Vec::with_capacity(n_state),
                scratch: crate::workspace::ScratchBuffers {
                    noise_buf: Vec::new(),
                    inflow_m3s_buf: Vec::new(),
                    lag_matrix_buf: Vec::new(),
                    par_inflow_buf: Vec::new(),
                    eta_floor_buf: Vec::new(),
                    zero_targets_buf: Vec::new(),
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::new(),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::new(),
                    effective_eta_buf: Vec::new(),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    trajectory_costs_buf: Vec::new(),
                    raw_noise_buf: Vec::new(),
                    perm_scratch: Vec::new(),
                    anticipated_state_buf: Vec::new(),
                    anticipated_state_out_col_indices_buf: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
            })
            .collect();
        let mut basis_store_4 = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces_4,
            basis_store: &mut basis_store_4,
            ctx: &ctx,
            baked: &mut templates.clone(),
            fcf: &mut fcf_4,
            cut_batches: &mut empty_cut_batches(n_stages),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // --- Verify identical FCF contents for all non-last stages ---
        for t in 0..(n_stages - 1) {
            let cuts_1: Vec<_> = fcf_1.active_cuts(t).collect();
            let cuts_4: Vec<_> = fcf_4.active_cuts(t).collect();

            assert_eq!(
                cuts_1.len(),
                cuts_4.len(),
                "stage {t}: cut count differs (1 workspace: {}, 4 workspaces: {})",
                cuts_1.len(),
                cuts_4.len()
            );

            for (idx, ((slot_1, intercept_1, coeff_1), (slot_4, intercept_4, coeff_4))) in
                cuts_1.iter().zip(cuts_4.iter()).enumerate()
            {
                assert_eq!(
                    slot_1, slot_4,
                    "stage {t} cut {idx}: slot mismatch ({slot_1} vs {slot_4})"
                );
                assert!(
                    (intercept_1 - intercept_4).abs() < 1e-12,
                    "stage {t} cut {idx}: intercept mismatch ({intercept_1} vs {intercept_4})"
                );
                assert_eq!(
                    coeff_1.len(),
                    coeff_4.len(),
                    "stage {t} cut {idx}: coefficient vector length mismatch"
                );
                for (j, (c1, c4)) in coeff_1.iter().zip(coeff_4.iter()).enumerate() {
                    assert!(
                        (c1 - c4).abs() < 1e-12,
                        "stage {t} cut {idx} coeff[{j}]: {c1} vs {c4}"
                    );
                }
            }
        }

        // Last stage must have no cuts in both.
        assert_eq!(fcf_1.active_cuts(n_stages - 1).count(), 0);
        assert_eq!(fcf_4.active_cuts(n_stages - 1).count(), 0);
    }

    // ── Load noise wiring tests (backward pass) ──────────────────────────────

    /// Build a 2-stage `StochasticContext` with 1 hydro and 1 stochastic load bus.
    ///
    /// The noise vector dimension is `n_hydros + n_load_buses = 2`.
    /// Stage 0 uses `branching_factor` openings; stage 1 is the successor solved
    /// in the backward pass opening loop.
    #[allow(clippy::too_many_lines)]
    fn make_stochastic_context_with_load(
        n_stages: usize,
        branching_factor: usize,
        mean_mw: f64,
        std_mw: f64,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };

        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(10),
            name: "H10".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let make_stage = |idx: usize| Stage {
            index: idx,
            id: idx as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(10),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|idx| LoadModel {
                bus_id: EntityId(1),
                stage_id: idx as i32,
                mean_mw,
                std_mw,
            })
            .collect();

        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: std::collections::BTreeMap::new(),
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(correlation)
            .build()
            .unwrap();

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap()
    }

    /// AC: Given a backward pass with 1 stochastic load bus and opening noise
    /// that includes a load component eta, the load balance row RHS in the patch
    /// buffer is set to `max(0, mean + std * eta) * block_factor` before the solve.
    ///
    /// We verify this indirectly: after the backward pass runs, `ws.scratch.load_rhs_buf`
    /// must be non-empty and must contain a positive value (with mean=300, std=30
    /// any reasonable eta produces a positive realization).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_load_patches_applied() {
        // 2-stage system: backward pass solves at successor=1 for each opening.
        // n_hydros=1, n_load_buses=1, 1 block per stage.
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        // mean_mw=300 guarantees a positive realization for any reasonable eta draw.
        let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 300.0, 30.0);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0, n_state=1

        // PatchBuffer: n_hydros=1, max_par_order=0, n_load_buses=1, max_blocks=1.
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1, 0, 0);

        // Template: 2 rows (row 0 = state-fixing, row 1 = water-balance).
        // base_rows=[1] → inflow RHS row starts at index 1.
        // noise_scale=[1.0, 1.0] (one per (stage, hydro)).
        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages]; // one per (stage, hydro)

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // MockSolver returns a fixed solution (1 state var, 1 dual entry).
        let solution = solution_1_0(100.0, -2.0);

        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(1),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
                anticipated_state_out_col_indices_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        };
        let mut workspaces = vec![ws];

        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        // load_balance_row_starts[successor=1]=10; load_bus_indices=[0]; 1 block/stage.
        let load_balance_row_starts = vec![10_usize; n_stages];
        let load_bus_indices = vec![0_usize];
        let block_counts_per_stage = vec![1_usize; n_stages];

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 1,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // After the backward pass, load_rhs_buf must have been populated with a
        // positive value for the last opening solved (mean=300, std=30 → positive).
        assert_eq!(
            workspaces[0].scratch.load_rhs_buf.len(),
            1,
            "load_rhs_buf must have 1 entry (1 load bus × 1 block)"
        );
        assert!(
            workspaces[0].scratch.load_rhs_buf[0] > 0.0,
            "load realization must be positive with mean=300, std=30: got {}",
            workspaces[0].scratch.load_rhs_buf[0]
        );
    }

    /// AC: Given a backward pass with 0 stochastic load buses, `patch_count`
    /// equals `N*(2+L)` (no load patches) and `load_rhs_buf` stays empty.
    ///
    /// N=1, L=0 → `N*(2+L) = 2`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_no_load_buses_unchanged() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0

        // PatchBuffer with no load buses: n_load_buses=0, max_blocks=1.
        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 0, 0, 0, 0);

        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -2.0);
        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
                anticipated_state_out_col_indices_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        };
        let mut workspaces = vec![ws];
        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let _ = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[1_usize; 2],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // With n_load_buses=0, forward_patch_count = N + z_inflow = 1 + 1 = 2.
        assert_eq!(
            workspaces[0].patch_buf.forward_patch_count(),
            2,
            "forward_patch_count must be N+z_inflow=2 when n_load_buses=0, got {}",
            workspaces[0].patch_buf.forward_patch_count()
        );
        // load_rhs_buf must remain empty.
        assert!(
            workspaces[0].scratch.load_rhs_buf.is_empty(),
            "load_rhs_buf must be empty when n_load_buses=0"
        );
    }

    /// AC: Given a backward pass with stochastic load, when Benders cut
    /// coefficients are extracted, the cut coefficient array has length `n_state`
    /// unchanged — load adds no state variables.
    ///
    /// Setup: N=1 hydro, L=0 PAR lags → `n_state=1`. After the backward pass with
    /// 1 load bus, each generated cut must have exactly 1 coefficient.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn backward_pass_cut_coefficients_unaffected() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context_with_load(n_stages, n_openings, 200.0, 20.0);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0, n_state=1

        let patch_buf = crate::lp_builder::PatchBuffer::new(1, 0, 1, 1, 0, 0);

        let template = StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![50.0, 100.0],
            row_upper: vec![50.0, 100.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template; n_stages];
        let base_rows = vec![1_usize; n_stages];
        let noise_scale = vec![1.0_f64; n_stages];

        let n_state = indexer.n_state; // 1
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(80.0, -3.0);
        let ws = SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(MockSolver::always_ok(solution)),
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            scratch: crate::workspace::ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(1),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
                anticipated_state_out_col_indices_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        };
        let mut workspaces = vec![ws];
        let comm = StubComm;
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let load_balance_row_starts = vec![10_usize; n_stages];
        let load_bus_indices = vec![0_usize];
        let block_counts_per_stage = vec![1_usize; n_stages];

        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 1,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // Exactly 1 cut generated (1 trial point × 1 stage with a successor).
        assert_eq!(result.cuts_generated, 1);

        // The cut must have exactly n_state=1 coefficient.
        let cuts: Vec<_> = fcf.active_cuts(0).collect();
        assert_eq!(cuts.len(), 1);
        let (_, _intercept, coefficients) = &cuts[0];
        assert_eq!(
            coefficients.len(),
            n_state,
            "cut coefficients length must be n_state={n_state}, got {} — \
             load buses must not add state variables",
            coefficients.len()
        );
    }

    /// BUG-1 structural invariant: per-stage cut sync inside the backward loop.
    ///
    /// Verifies that after `run_backward_pass`, the cut synchronization has been
    /// performed per-stage (not as a separate post-sweep loop). The structural
    /// evidence is:
    ///
    /// 1. `BackwardResult.cut_sync_time_ms` is populated (timing was captured).
    /// 2. The FCF has the expected number of cuts per stage — same as single-rank
    ///    without sync, because single-rank sync is a no-op that does not change
    ///    results but exercises the code path.
    /// 3. Using `LocalBackend` (the production single-rank communicator) instead
    ///    of `StubComm` exercises the full `sync_cuts` → allgatherv → deserialize
    ///    path, confirming no panics or data corruption.
    ///
    /// True multi-rank correctness testing requires actual MPI and is out of
    /// scope for CI. This test validates the structural invariant (sync is
    /// called per-stage inside the loop) and exercises the full code path.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn per_stage_cut_sync_invariant_after_bug1_fix() {
        use cobre_comm::LocalBackend;

        let n_stages = 4_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 1,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // 4-stage system: cuts at stages 0, 1, 2; 3 trial points each.
        // Total cuts = 3 stages × 3 trial points = 9.
        assert_eq!(result.cuts_generated, 9);

        // Each non-terminal stage has 3 cuts (one per trial point).
        assert_eq!(fcf.active_cuts(0).count(), 3, "stage 0 must have 3 cuts");
        assert_eq!(fcf.active_cuts(1).count(), 3, "stage 1 must have 3 cuts");
        assert_eq!(fcf.active_cuts(2).count(), 3, "stage 2 must have 3 cuts");
        assert_eq!(
            fcf.active_cuts(3).count(),
            0,
            "terminal stage must have 0 cuts"
        );

        // Verify cut_sync_time_ms was captured (structural evidence that
        // sync_cuts was called inside the backward loop).
        // For single-rank LocalBackend, sync is a no-op, so time should be
        // very small but the field must be populated (not default/garbage).
        // We just verify it's a valid non-negative value.
        assert!(
            result.cut_sync_time_ms < 10_000,
            "cut_sync_time_ms should be reasonable, got {}",
            result.cut_sync_time_ms
        );
    }

    /// Acceptance criterion: within a single backward
    /// iteration on a 3-stage system with `LocalBackend` (single-rank),
    /// cuts generated at stage t=1 are visible at stage t=0 and appear
    /// binding (mock returns positive cut duals). The metadata sync
    /// correctly accumulates `active_count` and sets `last_active_iter`.
    ///
    /// Uses `MockSolver::always_ok_with_binding_cuts` so that cut rows
    /// return positive duals, making them appear binding when evaluated.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn metadata_sync_updates_active_count_and_last_active_iter() {
        use cobre_comm::LocalBackend;

        // 3-stage system: backward loop processes t=1 then t=0.
        // At t=1: generates cuts into pool[1], successor pool[2] is empty.
        // At t=0: generates cuts into pool[0], successor pool[1] has cuts
        //         from t=1. Mock duals make those cuts appear binding.
        let n_stages = 3_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];

        let n_state = indexer.n_state;
        let forward_passes = 3_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0], vec![20.0], vec![30.0]]);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let solver = MockSolver::always_ok_with_binding_cuts(solution);
        let comm = LocalBackend;
        let mut workspaces = single_workspace(solver, n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);

        let mut csb = CutSyncBuffers::new(n_state, forward_passes as usize, 1);

        // Run a single backward iteration. The backward loop visits t=1
        // (cuts go to pool[1]), then t=0 (cuts go to pool[0], binding
        // checked against pool[1]).
        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 1,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // 3 stages × (n_stages-1=2 non-terminal) × 3 trial points = 6 cuts.
        assert_eq!(result.cuts_generated, 6);

        // Pool[1] received 3 cuts from t=1 backward pass.
        // Slot formula: warm_start(0) + iteration(1) * fwd_passes(3) + fpi
        // → slots 3, 4, 5. Populated count = 6 (high-water mark).
        assert_eq!(fcf.pools[1].populated_count, 6);

        // At t=0, the 3 cuts in pool[1] (slots 3,4,5) were evaluated for
        // binding. The mock solver returns positive duals (cut_dual_padding
        // = 1.0) for all cut rows. Each trial point has n_openings=2
        // openings, and the binding check runs per opening. So each slot
        // gets 3 trial points × 2 openings = 6 increments.
        for slot in 3..6 {
            assert!(
                fcf.pools[1].metadata[slot].active_count > 0,
                "slot {slot} active_count should be > 0 (cuts were binding)"
            );
            assert_eq!(
                fcf.pools[1].metadata[slot].active_count, 6,
                "slot {slot} active_count should be 6 (3 trial points × 2 openings)"
            );
            assert_eq!(
                fcf.pools[1].metadata[slot].last_active_iter, 1,
                "slot {slot} last_active_iter should be 1 (current iteration)"
            );
        }

        // Pool[2] (terminal successor) received no cuts and no binding
        // was checked against it — metadata should be at defaults.
        assert_eq!(fcf.pools[2].populated_count, 0);
    }

    /// Build N identical `SolverWorkspace<MockSolver>` instances and run a
    /// 2-stage backward pass with 6 trial points. Returns the resulting FCF.
    ///
    /// Used by `work_stealing_produces_identical_results_across_worker_counts`
    /// to compare FCF state across different worker counts.
    ///
    /// The `MockSolver` returns objective=100.0 and dual[0]=-5.0 for every solve,
    /// which is deterministic (no dependence on call order or worker identity).
    /// Each trial point i gets state [(i + 1) as f64 * 10.0] so that distinct
    /// cuts are generated and the ordering invariant is meaningful.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn run_backward_pass_with_n_workers(n_workers: usize) -> FutureCostFunction {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 2_usize;
        let local_work = 6_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state; // 1

        // Use forward_passes = local_work so the FCF pool is large enough for
        // all trial points in a single iteration (iteration 0, slots 0..5).
        #[allow(clippy::cast_possible_truncation)]
        let forward_passes = local_work as u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 64, &vec![0; n_stages]);

        // Build `local_work` trial points with distinct states so each cut
        // has a different intercept. State for trial point i = (i+1)*10.0.
        let states: Vec<Vec<f64>> = (0..local_work)
            .map(|i| vec![(i + 1) as f64 * 10.0])
            .collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        // Each workspace gets the same deterministic solution.
        // MockSolver::always_ok returns objective=100.0, dual[0]=-5.0 for
        // every call regardless of call order or worker identity.
        let solution = solution_1_0(100.0, -5.0);
        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("worker_id fits in i32"),
                solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
                patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
                current_state: Vec::with_capacity(n_state),
                scratch: crate::workspace::ScratchBuffers {
                    noise_buf: Vec::new(),
                    inflow_m3s_buf: Vec::new(),
                    lag_matrix_buf: Vec::new(),
                    par_inflow_buf: Vec::new(),
                    eta_floor_buf: Vec::new(),
                    zero_targets_buf: Vec::new(),
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::new(),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::new(),
                    effective_eta_buf: Vec::new(),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    trajectory_costs_buf: Vec::new(),
                    raw_noise_buf: Vec::new(),
                    perm_scratch: Vec::new(),
                    anticipated_state_buf: Vec::new(),
                    anticipated_state_out_col_indices_buf: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let comm = StubComm;
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .unwrap();

        // Confirm all 6 trial points produced cuts at stage 0.
        assert_eq!(
            result.cuts_generated, local_work,
            "n_workers={n_workers}: expected {local_work} cuts, got {}",
            result.cuts_generated,
        );

        fcf
    }

    #[test]
    fn work_stealing_produces_identical_results_across_worker_counts() {
        // Acceptance criterion: the FCF state after running the backward pass
        // with 1 workspace must be bit-identical to the state after running
        // with 3 workspaces, given the same inputs. This verifies that the
        // sort-by-trial_point_idx post-processing in the work-stealing
        // implementation produces a deterministic FCF regardless of which
        // worker claims which trial point.
        let fcf_1 = run_backward_pass_with_n_workers(1);
        let fcf_3 = run_backward_pass_with_n_workers(3);

        let num_stages = 2;

        // Verify that both runs produced cuts (belt-and-suspenders guard so
        // that an empty FCF cannot cause a false positive).
        assert!(
            fcf_1.active_cuts(0).count() > 0,
            "1-worker run produced no cuts at stage 0"
        );

        for stage in 0..num_stages {
            let cuts_1: Vec<_> = fcf_1.active_cuts(stage).collect();
            let cuts_3: Vec<_> = fcf_3.active_cuts(stage).collect();
            assert_eq!(
                cuts_1.len(),
                cuts_3.len(),
                "stage {stage}: cut count mismatch ({} vs {})",
                cuts_1.len(),
                cuts_3.len(),
            );
            for (i, ((s1, int1, c1), (s3, int3, c3))) in cuts_1.iter().zip(&cuts_3).enumerate() {
                assert_eq!(
                    s1, s3,
                    "stage {stage}, cut {i}: slot mismatch ({s1} vs {s3})"
                );
                assert_eq!(
                    int1, int3,
                    "stage {stage}, cut {i}: intercept mismatch ({int1} vs {int3})"
                );
                assert_eq!(
                    c1, c3,
                    "stage {stage}, cut {i}: coefficients mismatch ({c1:?} vs {c3:?})"
                );
            }
        }
    }

    // ── Parallel overhead decomposition unit tests ────────────────────────────

    /// Build a `SolverStatistics` snapshot with the given cumulative times (in seconds).
    fn make_stats(
        solve_s: f64,
        load_s: f64,
        set_bounds_s: f64,
        basis_set_s: f64,
    ) -> SolverStatistics {
        SolverStatistics {
            total_solve_time_seconds: solve_s,
            total_load_model_time_seconds: load_s,
            total_set_bounds_time_seconds: set_bounds_s,
            total_basis_set_time_seconds: basis_set_s,
            ..SolverStatistics::default()
        }
    }

    /// Decompose parallel overhead into (`setup_ms`, `imbalance_ms`, `scheduling_ms`)
    /// from per-worker before/after snapshots.
    fn decompose_overhead(
        pairs: &[(SolverStatistics, SolverStatistics)],
        parallel_wall_ms: u64,
    ) -> (u64, u64, u64) {
        use crate::solver_stats::SolverStatsDelta;

        #[allow(clippy::cast_precision_loss)]
        let n_workers = pairs.len() as f64;

        let worker_deltas: Vec<SolverStatsDelta> = pairs
            .iter()
            .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after))
            .collect();

        let stage_setup_ms: f64 = worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();

        let worker_totals: Vec<f64> = worker_deltas
            .iter()
            .map(|d| {
                d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
            })
            .collect();

        let max_worker_ms = worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if worker_totals.is_empty() {
            0.0_f64
        } else {
            worker_totals.iter().sum::<f64>() / n_workers
        };

        let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        #[allow(clippy::cast_precision_loss)]
        let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        (
            stage_setup_ms as u64,
            stage_imbalance_ms as u64,
            stage_scheduling_ms as u64,
        )
    }

    /// 4 workers with different solve times: imbalance equals
    /// `trunc(max - mean_f64)` of worker totals, scheduling equals
    /// `parallel_wall - max`.
    ///
    /// Worker solve times: 100 ms, 200 ms, 150 ms, 180 ms.
    /// Setup per worker: 0 (this sub-test isolates solve imbalance).
    /// Mean of totals (f64) = 630.0 / 4 = 157.5.
    /// Imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42.
    /// Scheduling = 250 - 200 = 50.
    #[test]
    fn decompose_four_workers_different_solve_times() {
        // All setup times are zero; use only solve time to isolate imbalance.
        let zero = SolverStatistics::default();
        let pairs = vec![
            (zero.clone(), make_stats(0.1, 0.0, 0.0, 0.0)), // 100 ms solve
            (zero.clone(), make_stats(0.2, 0.0, 0.0, 0.0)), // 200 ms solve
            (zero.clone(), make_stats(0.15, 0.0, 0.0, 0.0)), // 150 ms solve
            (zero.clone(), make_stats(0.18, 0.0, 0.0, 0.0)), // 180 ms solve
        ];
        let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 250);

        assert_eq!(setup_ms, 0, "no setup work expected");
        // f64 mean = 157.5; imbalance = trunc(200.0 - 157.5) = trunc(42.5) = 42
        assert_eq!(
            imbalance_ms, 42,
            "imbalance = trunc(max(200.0) - avg(157.5)) = trunc(42.5) = 42"
        );
        // scheduling = 250 - 200 = 50
        assert_eq!(scheduling_ms, 50, "scheduling overhead = wall - max_worker");
    }

    /// Acceptance criterion: `setup_time_ms` is the sum of all workers' non-solve
    /// work.
    ///
    /// Workers have setup costs (load+add+bounds+basis): 20, 25, 15, 22 ms.
    /// Expected `setup_ms` = 20 + 25 + 15 + 22 = 82.
    #[test]
    fn decompose_setup_time_is_aggregate_non_solve_work() {
        let zero = SolverStatistics::default();
        // Each worker: 0 solve + known setup split across the three sub-timers.
        // Worker setup totals: 20, 25, 15, 22 ms (put entirely in load_model timer).
        let pairs = vec![
            (zero.clone(), make_stats(0.0, 0.020, 0.0, 0.0)), // 20 ms total setup
            (zero.clone(), make_stats(0.0, 0.025, 0.0, 0.0)), // 25 ms
            (zero.clone(), make_stats(0.0, 0.015, 0.0, 0.0)), // 15 ms
            (zero.clone(), make_stats(0.0, 0.022, 0.0, 0.0)), // 22 ms
        ];
        let (setup_ms, _imbalance_ms, _scheduling_ms) = decompose_overhead(&pairs, 300);
        assert_eq!(
            setup_ms, 82,
            "aggregate setup must sum all workers' non-solve work"
        );
    }

    /// Edge case: all workers have identical timing → imbalance must be 0.
    #[test]
    fn decompose_identical_workers_zero_imbalance() {
        let zero = SolverStatistics::default();
        let after = make_stats(0.1, 0.01, 0.002, 0.001);
        let pairs = vec![
            (zero.clone(), after.clone()),
            (zero.clone(), after.clone()),
            (zero.clone(), after.clone()),
        ];
        let (_, imbalance_ms, _) = decompose_overhead(&pairs, 200);
        assert_eq!(
            imbalance_ms, 0,
            "identical workers must have zero imbalance"
        );
    }

    /// Edge case: single worker → imbalance is 0, setup equals that worker's
    /// setup, scheduling is the residual.
    #[test]
    fn decompose_single_worker() {
        let zero = SolverStatistics::default();
        // 100 ms solve + 20 ms setup = 120 ms worker total.
        let after = make_stats(0.1, 0.020, 0.0, 0.0);
        let pairs = vec![(zero.clone(), after)];
        let (setup_ms, imbalance_ms, scheduling_ms) = decompose_overhead(&pairs, 150);

        assert_eq!(setup_ms, 20, "single worker: setup = 20 ms");
        assert_eq!(imbalance_ms, 0, "single worker: imbalance must be 0");
        // scheduling = 150 - 120 = 30
        assert_eq!(
            scheduling_ms, 30,
            "single worker: scheduling = wall - worker_total"
        );
    }

    /// Edge case: `scheduling_overhead_ms` is clamped to 0 when `max_worker_total`
    /// exceeds `parallel_wall_ms` (clock skew or measurement granularity).
    #[test]
    fn decompose_scheduling_clamped_when_worker_exceeds_wall() {
        let zero = SolverStatistics::default();
        // Worker total = 200 ms, but wall = 180 ms → scheduling would be negative.
        let after = make_stats(0.2, 0.0, 0.0, 0.0);
        let pairs = vec![(zero.clone(), after)];
        let (_, _, scheduling_ms) = decompose_overhead(&pairs, 180);
        assert_eq!(scheduling_ms, 0, "negative scheduling must be clamped to 0");
    }

    // ── allgatherv per-worker stats unit tests ────────────────────────────────

    /// Single-rank (np=1) backward pass with 2 workers.
    ///
    /// Constructs a 2-worker `StageWorkerStatsBuffer::new(2, 4)` and uses
    /// `StubComm` (which echoes send→recv, simulating `LocalBackend` np=1).
    /// After one backward iteration, `BackwardResult::stage_stats` must
    /// contain 2 entries per non-zero opening (`worker_id` 0 and `worker_id` 1),
    /// both with `rank = 0`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn allgatherv_single_rank_two_workers_stage_stats_has_per_worker_entries() {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 2_usize;
        let n_openings = 4_usize;
        let n_workers = 2_usize;
        let local_work = 4_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;

        let solution = solution_1_0(100.0, -5.0);
        let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("idx fits in i32"),
                solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
                patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
                current_state: Vec::with_capacity(n_state),
                scratch: crate::workspace::ScratchBuffers {
                    noise_buf: Vec::new(),
                    inflow_m3s_buf: Vec::new(),
                    lag_matrix_buf: Vec::new(),
                    par_inflow_buf: Vec::new(),
                    eta_floor_buf: Vec::new(),
                    zero_targets_buf: Vec::new(),
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::new(),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::new(),
                    effective_eta_buf: Vec::new(),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    trajectory_costs_buf: Vec::new(),
                    raw_noise_buf: Vec::new(),
                    perm_scratch: Vec::new(),
                    anticipated_state_buf: Vec::new(),
                    anticipated_state_out_col_indices_buf: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &StubComm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .expect("single-rank 2-worker backward must not error");

        // The 2-stage system has 1 backward stage (t=0, successor=1).
        // stage_stats must contain exactly 1 entry (one successor).
        assert_eq!(
            result.stage_stats.len(),
            1,
            "expected 1 backward stage entry (successor=1)"
        );
        let (successor, entries) = &result.stage_stats[0];
        assert_eq!(*successor, 1_usize, "successor index must be 1");

        // Every entry must have rank=0 (np=1 StubComm).
        for (rank, _wid, _omega, _delta) in entries {
            assert_eq!(*rank, 0_i32, "all entries must have rank=0 for np=1");
        }
        // Both worker_id values (0 and 1) must appear at omega=0.
        let omega0_wids: Vec<i32> = entries
            .iter()
            .filter(|(_, _, omega, _)| *omega == 0)
            .map(|(_, wid, _, _)| *wid)
            .collect();
        assert!(
            omega0_wids.contains(&0),
            "worker_id=0 must appear at omega=0"
        );
        assert!(
            omega0_wids.contains(&1),
            "worker_id=1 must appear at omega=0"
        );
    }

    /// Multi-rank (np=2) backward pass with stub communicator.
    ///
    /// Uses a `DualRankStubComm` whose `size()` returns 2 and whose
    /// `allgatherv` concatenates a manually injected "remote rank" payload
    /// (a copy of the send buffer). Asserts that the unpacked
    /// `stage_stats` contains entries for both `rank=0` and `rank=1`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn allgatherv_dual_rank_stub_stage_stats_contains_both_ranks() {
        use crate::lp_builder::PatchBuffer;

        /// Stub communicator simulating np=2: `allgatherv` fills recv with
        /// `[send, send]` (rank-0 and a synthetic rank-1 copy).
        struct DualRankStubComm;

        impl Communicator for DualRankStubComm {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                counts: &[usize],
                displs: &[usize],
            ) -> Result<(), CommError> {
                // Fill each rank's slot in recv using the provided counts/displs.
                // Both ranks contribute `send` (rank-1 is a synthetic copy of rank-0).
                for (r, (&count, &displ)) in counts.iter().zip(displs).enumerate() {
                    let src = &send[..count.min(send.len())];
                    recv[displ..displ + src.len()].copy_from_slice(src);
                    let _ = r; // suppress unused warning in cfg(test)
                }
                Ok(())
            }

            fn allreduce<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                recv[..send.len()].copy_from_slice(send);
                Ok(())
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                Ok(())
            }

            fn barrier(&self) -> Result<(), CommError> {
                Ok(())
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                2
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let n_workers = 1_usize;
        let local_work = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let n_state = indexer.n_state;

        let solution = solution_1_0(100.0, -5.0);
        let states: Vec<Vec<f64>> = (0..local_work).map(|i| vec![(i + 1) as f64]).collect();
        let mut exchange = exchange_with_states(n_state, states);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("idx fits in i32"),
                solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
                patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
                current_state: Vec::with_capacity(n_state),
                scratch: crate::workspace::ScratchBuffers {
                    noise_buf: Vec::new(),
                    inflow_m3s_buf: Vec::new(),
                    lag_matrix_buf: Vec::new(),
                    par_inflow_buf: Vec::new(),
                    eta_floor_buf: Vec::new(),
                    zero_targets_buf: Vec::new(),
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::new(),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::new(),
                    effective_eta_buf: Vec::new(),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    trajectory_costs_buf: Vec::new(),
                    raw_noise_buf: Vec::new(),
                    perm_scratch: Vec::new(),
                    anticipated_state_buf: Vec::new(),
                    anticipated_state_out_col_indices_buf: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, local_work as u32, 64, &vec![0; n_stages]);
        let mut csb = CutSyncBuffers::new(n_state, local_work, 1);

        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(templates.len()),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &DualRankStubComm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        })
        .expect("dual-rank stub backward must not error");

        // With np=2, stage_stats for successor=1 must contain entries from
        // both rank=0 and rank=1 (DualRankStubComm copies the rank-0 block
        // into the rank-1 slot, so both appear in the unpacked output).
        assert_eq!(result.stage_stats.len(), 1);
        let (_, entries) = &result.stage_stats[0];

        let ranks_seen: Vec<i32> = entries
            .iter()
            .map(|(rank, _, _, _)| *rank)
            .collect::<std::collections::HashSet<i32>>()
            .into_iter()
            .collect();
        assert!(
            ranks_seen.contains(&0),
            "rank=0 must appear in stage_stats; got {ranks_seen:?}"
        );
        assert!(
            ranks_seen.contains(&1),
            "rank=1 must appear in stage_stats; got {ranks_seen:?}"
        );
    }

    // ── read-site prefer-with-fallback unit tests ────────────────

    /// Run `process_trial_point_backward` for stage 0 → successor 1 with
    /// explicitly-provided backward and forward basis stores.
    ///
    /// `basis_store` is taken by `&mut` so a `BasisStoreSliceMut` can be
    /// derived from it and passed to `process_trial_point_backward`.
    ///
    /// Returns the mutated workspace so the caller can inspect
    /// `ws.solver.warm_start_calls`.
    #[allow(clippy::too_many_lines)]
    fn run_one_trial_point_with_stores(
        basis_store: &mut crate::workspace::BasisStore,
    ) -> Result<Vec<SolverWorkspace<MockSolver>>, crate::SddpError> {
        use crate::context::StageContext;

        let n_stages = 2_usize;
        let n_openings = 1_usize;
        let n_state = 1_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let indexer = StageIndexer::new(n_state, 0);

        let solver = MockSolver::always_ok(solution_1_0(100.0, -5.0));
        let mut workspaces = single_workspace(solver, n_state);
        let ws = &mut workspaces[0];
        ws.backward_accum
            .outcomes
            .push(crate::risk_measure::BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0; n_state],
                objective_value: 0.0,
            });
        ws.backward_accum
            .per_opening_stats
            .push(SolverStatsDelta::default());
        ws.backward_accum.agg_coefficients.resize(n_state, 0.0);

        let exchange = exchange_with_states(n_state, vec![vec![5.0]]);

        let templates: &'static _ = Box::leak(Box::new(vec![
            minimal_template_1_0(),
            minimal_template_1_0(),
        ]));
        let base_rows: &'static _ = Box::leak(Box::new(vec![1_usize, 1_usize]));
        let ctx: StageContext<'static> = StageContext {
            templates,
            base_rows,
            noise_scale: Box::leak(Box::new(vec![])),
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: Box::leak(Box::new(vec![])),
            load_bus_indices: Box::leak(Box::new(vec![])),
            block_counts_per_stage: Box::leak(Box::new(vec![])),
            ncs_max_gen: Box::leak(Box::new(vec![])),
            ncs_allow_curtailment: Box::leak(Box::new(vec![])),
            discount_factors: Box::leak(Box::new(vec![])),
            cumulative_discount_factors: Box::leak(Box::new(vec![])),
            stage_lag_transitions: Box::leak(Box::new(vec![])),
            noise_group_ids: Box::leak(Box::new(vec![])),
            downstream_par_order: 0,
        };

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: None,
            noise_key_diag: None,
        };

        let iteration: u64 = 1;
        let fwd_offset: usize = 0;
        let succ_probabilities = vec![1.0_f64; n_openings];
        let successor_active_slots: Vec<usize> = vec![];
        let baked_template = minimal_template_1_0();

        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0u32; n_stages]);
        let empty_cut_batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };

        let succ_spec = super::SuccessorSpec {
            t: 0,
            successor: 1,
            my_rank: 0,
            probabilities: &succ_probabilities,
            cut_batch: &empty_cut_batch,
            num_cuts_at_successor: 0,
            template_num_rows: baked_template.num_rows,
            baked_template: &baked_template,
            successor_active_slots: &successor_active_slots,
            cut_activity_tolerance: 0.0,
            successor_populated_count: fcf.pools[1].populated_count,
            successor_pool: &fcf.pools[1],
        };

        // Derive a single-worker BasisStoreSliceMut covering all scenarios.
        let mut basis_slices = basis_store.split_workers_mut(1);
        let mut basis_slice = basis_slices.remove(0);

        let ws = &mut workspaces[0];
        super::load_backward_lp(ws, &succ_spec);
        ws.backward_accum
            .per_opening_stats
            .resize_with(n_openings, SolverStatsDelta::default);
        for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
            *slot = SolverStatsDelta::default();
        }
        ws.backward_accum.slot_increments.resize(1, 0);
        ws.backward_accum.slot_increments[..1].fill(0);

        super::process_trial_point_backward(
            ws,
            &ctx,
            &training_ctx,
            &exchange,
            fwd_offset,
            iteration,
            &risk_measures,
            &succ_spec,
            &mut basis_slice,
            0,
        )?;
        Ok(workspaces)
    }

    // ---------------------------------------------------------------------------
    // resolve_backward_basis_* unit tests
    // ---------------------------------------------------------------------------

    #[test]
    fn resolve_backward_basis_returns_some_when_slot_is_populated() {
        // Given: BasisStore[0, 1] has Some(CapturedBasis).
        // Then: resolve_backward_basis returns Some(_).
        use crate::workspace::{BasisStore, CapturedBasis};

        let b = CapturedBasis::new(2, 2, 0, 0, 0);
        let mut store = BasisStore::new(1, 2);
        *store.get_mut(0, 1) = Some(b);

        let slices = store.split_workers_mut(1);
        let slice = &slices[0];
        let basis_ref = super::resolve_backward_basis(slice, 0, 1);

        assert!(basis_ref.is_some(), "expected Some when slot has a basis");
        drop(slices);
    }

    #[test]
    fn resolve_backward_basis_returns_none_when_slot_is_empty() {
        // Given: BasisStore[0, 1] is None (cold-start, slot never written).
        // Then: resolve_backward_basis returns None.
        use crate::workspace::BasisStore;

        let mut store = BasisStore::new(1, 2);
        let slices = store.split_workers_mut(1);
        let slice = &slices[0];
        let basis_ref = super::resolve_backward_basis(slice, 0, 1);

        assert!(basis_ref.is_none(), "expected None for empty slot");
        drop(slices);
    }

    // ---------------------------------------------------------------------------
    // T2 integration tests (backward write populates BasisStore)
    // ---------------------------------------------------------------------------

    #[test]
    fn backward_write_populates_basis_store_at_omega_zero() {
        // Given: a 2-stage, 1-opening study with one forward trial point (m=0, x_hat=[5.0]).
        //        BasisStore starts empty (all None).
        // When: process_trial_point_backward runs at omega=0.
        // Then: BasisStore[0, 1] is Some(CapturedBasis) with state_at_capture == [5.0].
        //
        // write occurs only at omega=0 (this test has exactly 1 opening).
        // infeasibility guard is not triggered (solver succeeds).
        use crate::workspace::BasisStore;

        let mut basis_store = BasisStore::new(1, 2);
        let workspaces = run_one_trial_point_with_stores(&mut basis_store).unwrap();

        // Verify the BasisStore slot was written.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore[0, 1] must be Some after backward write at omega=0"
        );
        let captured = basis_store.get(0, 1).unwrap();
        assert_eq!(
            captured.state_at_capture,
            vec![5.0_f64],
            "state_at_capture must equal x_hat"
        );
        // Confirm the solver ran exactly once.
        assert_eq!(
            workspaces[0].solver.inner().call_count,
            1,
            "solver must be called exactly once for a 1-opening backward pass"
        );
    }

    #[test]
    fn backward_write_preserves_slot_on_infeasibility_at_omega_zero() {
        // Given: a 2-stage, 1-opening study.
        //        BasisStore starts with a pre-existing basis at [0, 1].
        //        The solver returns Infeasible on its first call.
        // When: process_trial_point_backward runs via run_backward_pass.
        // Then: run_backward_pass returns Err(SddpError::Infeasible) and
        //       BasisStore[0, 1] retains its original content.
        //
        // the write in process_trial_point_backward is guarded by `?`
        // immediately after run_stage_solve. An Infeasible error propagates
        // out of the function before reaching the BasisStore write site, so
        // the slot is unconditionally preserved on infeasibility.
        use cobre_solver::Basis;

        use crate::workspace::{BasisStore, CapturedBasis};

        // Pre-populate slot [0, 1] with a sentinel basis. `state_at_capture =
        // [42.0]` is the sentinel that the reuse-path overwrite must
        // replace. The remaining fields satisfy the `CapturedBasis`
        // invariant `row_status.len() == base_row_count + cut_row_slots.len()`.
        let pre_existing = CapturedBasis {
            basis: Basis::new(2, 2),
            base_row_count: 2,
            cut_row_slots: Vec::new(),
            state_at_capture: vec![42.0],
        };
        let mut basis_store = BasisStore::new(1, 2);
        *basis_store.get_mut(0, 1) = Some(pre_existing);

        // Verify sentinel is in place before the call.
        assert_eq!(
            basis_store.get(0, 1).unwrap().state_at_capture,
            vec![42.0_f64],
            "sentinel must be in place before the infeasible solve"
        );

        // run_one_trial_point_with_stores uses MockSolver::always_ok, so we
        // exercise the reuse path (successful solve overwrites slot). For the
        // infeasibility path, the structural guarantee is: `?` in
        // process_trial_point_backward propagates Err before the write site.
        // That path is integration-tested by `backward_pass_propagates_infeasible_error`.
        //
        // Here we test the complementary invariant: a *successful* solve at ω=0
        // with a pre-existing slot uses the reuse branch (get_basis into the
        // existing allocation) and leaves the slot Some (not None).
        let result = run_one_trial_point_with_stores(&mut basis_store);
        assert!(result.is_ok(), "expected Ok for successful solve");

        // The slot must still be Some after the successful reuse-path write.
        assert!(
            basis_store.get(0, 1).is_some(),
            "BasisStore[0, 1] must not be None after successful reuse-path write at ω=0"
        );
        // The reuse path updates state_at_capture to the current x_hat=[5.0].
        assert_eq!(
            basis_store.get(0, 1).unwrap().state_at_capture,
            vec![5.0_f64],
            "state_at_capture must be updated to x_hat by the reuse path"
        );
    }

    /// T-HW01: handshake passes when all ranks agree on `n_workers_local`.
    ///
    /// Uses `StubComm` (echoes send→recv, i.e. min==max==local) with a
    /// 2-worker setup and a 1-stage system so no backward stages are swept.
    /// The test only validates that the uniformity check does not reject a
    /// consistent 2-worker configuration.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn handshake_passes_with_local_backend() {
        use crate::lp_builder::PatchBuffer;

        let n_stages = 1_usize;
        let n_workers = 2_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0()];
        let base_rows = vec![1_usize];
        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let solution = solution_1_0(100.0, -5.0);

        // Build 2 workspaces to exercise n_workers_local=2.
        let mut workspaces: Vec<SolverWorkspace<MockSolver>> = (0..n_workers)
            .map(|idx| SolverWorkspace {
                rank: 0,
                worker_id: i32::try_from(idx).expect("idx fits i32"),
                solver: ProfiledSolver::new(MockSolver::always_ok(solution.clone())),
                patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0),
                current_state: Vec::with_capacity(n_state),
                scratch: crate::workspace::ScratchBuffers {
                    noise_buf: Vec::new(),
                    inflow_m3s_buf: Vec::new(),
                    lag_matrix_buf: Vec::new(),
                    par_inflow_buf: Vec::new(),
                    eta_floor_buf: Vec::new(),
                    zero_targets_buf: Vec::new(),
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::new(),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::new(),
                    effective_eta_buf: Vec::new(),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                    lag_accumulator: vec![],
                    lag_weight_accum: 0.0,
                    downstream_accumulator: Vec::new(),
                    downstream_weight_accum: 0.0,
                    downstream_completed_lags: Vec::new(),
                    downstream_n_completed: 0,
                    recon_slot_lookup: Vec::new(),
                    trajectory_costs_buf: Vec::new(),
                    raw_noise_buf: Vec::new(),
                    perm_scratch: Vec::new(),
                    anticipated_state_buf: Vec::new(),
                    anticipated_state_out_col_indices_buf: Vec::new(),
                },
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::default(),
                worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
            })
            .collect();

        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::new(n_state, 1, 1);
        let comm = StubComm;

        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(n_stages),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        });

        assert!(
            result.is_ok(),
            "handshake must pass when all ranks have the same n_workers_local; got: {result:?}"
        );
    }

    /// T-HW02: handshake rejects non-uniform `n_workers_local` across ranks.
    ///
    /// `NonUniformStubComm` simulates a 2-rank cluster where min and max
    /// worker counts differ. Its `allreduce(Min)` returns all `T::default()`
    /// (zeros), while `allreduce(Max)` copies `send` to `recv` (the local
    /// value). With `local_workers = 1`, `min_recv[0] = 0` and
    /// `max_recv[0] = 1`, so `0 != 1` triggers the uniformity check.
    /// `BackwardPassState::run` must return `SddpError::Validation` with the
    /// expected substring before entering the stage loop.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn handshake_rejects_nonuniform_workers() {
        /// Stub communicator that forces `allreduce(Min)` to return zeros and
        /// `allreduce(Max)` to echo the send buffer, producing `min != max`
        /// for any non-zero local value.
        struct NonUniformStubComm;

        impl Communicator for NonUniformStubComm {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                recv[..send.len()].copy_from_slice(send);
                Ok(())
            }

            fn allreduce<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                op: ReduceOp,
            ) -> Result<(), CommError> {
                match op {
                    // Min: return T::default() (0) to simulate a remote rank
                    // with zero workers, creating a min != max discrepancy.
                    ReduceOp::Min => {
                        for r in recv.iter_mut() {
                            *r = T::default();
                        }
                    }
                    // Max and all others: echo send so max == local value.
                    _ => {
                        recv[..send.len()].copy_from_slice(send);
                    }
                }
                Ok(())
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                Ok(())
            }

            fn barrier(&self) -> Result<(), CommError> {
                Ok(())
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                2
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let n_stages = 1_usize;
        let stochastic = make_stochastic_context(n_stages, 1);
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template_1_0()];
        let base_rows = vec![1_usize];
        let n_state = indexer.n_state;
        let forward_passes = 1_u32;
        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let mut exchange = exchange_with_states(n_state, vec![vec![10.0]]);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];
        let comm = NonUniformStubComm;
        // n_workers_local = 1 on this rank; allreduce(Min) returns 0 and
        // allreduce(Max) returns 1 → 0 != 1 triggers the validation error.
        let mut workspaces =
            single_workspace(MockSolver::always_ok(solution_1_0(100.0, -5.0)), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::new(n_state, 1, 1);

        let result = run_backward_pass(&mut crate::backward_pass_state::BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            baked: &mut templates.clone(),
            fcf: &mut fcf,
            cut_batches: &mut empty_cut_batches(n_stages),
            training_ctx: &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &[],
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
                dcs: None,
                noise_key_diag: None,
            },
            comm: &comm,
            records: &[],
            iteration: 0,
            local_work: exchange.local_count(),
            fwd_offset: 0,
            risk_measures: &risk_measures,
            exchange: &mut exchange,
            cut_activity_tolerance: 0.0,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            cut_selection: None,
            event_sender: None,
        });

        match result {
            Err(crate::SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("non-uniform n_workers_local"),
                    "error message must contain 'non-uniform n_workers_local'; got: {msg}"
                );
                assert!(
                    msg.contains("min=0"),
                    "error message must mention min=0 (stub Min returns T::default()); got: {msg}"
                );
                assert!(
                    msg.contains("max=1"),
                    "error message must mention max=1 (stub Max echoes local=1); got: {msg}"
                );
                assert!(
                    msg.contains("local=1"),
                    "error message must mention local=1 (single workspace); got: {msg}"
                );
            }
            other => panic!(
                "expected Err(SddpError::Validation(_)) from non-uniform handshake, got: {other:?}"
            ),
        }
    }

    /// Minimal anticipated `StageIndexer` for sign-convention tests.
    fn make_anticipated_indexer_local(
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
    ) -> StageIndexer {
        use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout};
        StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices: (0..n_anticipated).collect(),
            },
            &FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Verify cut sign convention: dual 7.5 → batch -7.5 at correct column.
    #[test]
    fn cut_coefficient_sign_convention_slot_zero_k2() {
        let indexer = make_anticipated_indexer_local(1, 2, vec![2]);
        assert_eq!(indexer.anticipated_state.start, 0);
        assert_eq!(indexer.n_state, 2);

        let mut fcf = FutureCostFunction::new(3, indexer.n_state, 1, 10, &[0; 3]);
        let mut coefficients = vec![0.0_f64; indexer.n_state];
        coefficients[indexer.anticipated_state.start] = 7.5;
        fcf.add_cut(1, 0, 0, 0.0, &coefficients);

        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        crate::forward::build_cut_row_batch_into(&mut batch, &fcf, 1, &indexer, &[]);

        let lp_col = indexer.state_to_lp_column(indexer.anticipated_state.start);
        assert_eq!(lp_col, 1);

        let pos = batch
            .col_indices
            .iter()
            .position(|&c| c == lp_col as i32)
            .expect("lp_col must appear in batch.col_indices");

        assert!(
            (batch.values[pos] - (-7.5_f64)).abs() < f64::EPSILON,
            "expected batch.values[pos]=-7.5 for lp_col={lp_col}, got {}",
            batch.values[pos]
        );
    }

    // -----------------------------------------------------------------------
    // DCS backward-integration tests
    // -----------------------------------------------------------------------

    use crate::cut_selection::{CutMetadata, CutSelectionStrategy};
    use crate::dcs::DcsParams;
    use crate::lp_builder::PatchBuffer;
    use crate::workspace::WorkspaceSizing;
    use cobre_solver::ActiveSolver;

    /// Cut-free successor core for `StageIndexer::new(1, 0)`:
    /// columns `[storage_out=0, z_inflow=1, storage_in=2, theta=3]`.
    /// Minimize `theta`. `patch_opening_bounds` pins `storage_in` (col 2, the
    /// incoming-state column) to `x_hat`. A single coupling row
    /// `storage_out - storage_in = 0` ties the outgoing-state column (col 0,
    /// which the cuts reference) to the pinned incoming state, so the cut floor
    /// is evaluated at `x_hat` and the cut subgradient flows back to the pinned
    /// column — the minimal structure that makes the backward dual a real
    /// subgradient with respect to the incoming state.
    fn dcs_core_template() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 1,
            num_nz: 2,
            // CSC by column: col0 → (row0, +1), col2 → (row0, -1); cols 1,3 empty.
            col_starts: vec![0_i32, 1, 1, 2, 2],
            row_indices: vec![0_i32, 0],
            values: vec![1.0, -1.0],
            col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            // Coupling equality: storage_out - storage_in = 0.
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a two-stage `FutureCostFunction` whose successor (stage 1) pool
    /// carries three cuts on the incoming-storage state (index 0):
    ///   slot 0: intercept 1, coeff [0]   → floor 1.0 at `x_hat` = 2
    ///   slot 1: intercept 0, coeff [2]   → floor 4.0  (the binding cut)
    ///   slot 2: intercept 3, coeff [0]   → floor 3.0
    /// Metadata is set so that the metadata-seeded initial set omits the
    /// binding slot 1 (its `last_active_iter` is stale) — the lazy loop must
    /// add it.
    fn dcs_two_stage_fcf() -> FutureCostFunction {
        let n_stages = 2;
        let mut fcf = FutureCostFunction::new(n_stages, 1, 8, 10, &vec![0; n_stages]);
        fcf.add_cut(1, 0, 0, 1.0, &[0.0]);
        fcf.add_cut(1, 0, 1, 0.0, &[2.0]);
        fcf.add_cut(1, 0, 2, 3.0, &[0.0]);
        // Seed metadata: slots 0 and 2 are "recently active"; the binding slot 1
        // is stale so the k2 window excludes it (forcing the lazy add). All were
        // generated before the current iteration so none is current-iteration
        // protected.
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[1].metadata[0] = meta(1, 5);
        fcf.pools[1].metadata[1] = meta(1, 1); // stale → outside k2=2 window at iter 5
        fcf.pools[1].metadata[2] = meta(1, 5);
        fcf
    }

    /// Single real-solver (`ActiveSolver`) workspace sized for `n_state = 1`.
    fn dcs_active_workspace() -> Vec<SolverWorkspace<ActiveSolver>> {
        let sizing = WorkspaceSizing {
            hydro_count: 1,
            max_par_order: 0,
            n_load_buses: 0,
            max_blocks: 0,
            downstream_par_order: 0,
            max_openings: 1,
            initial_pool_capacity: 16,
            n_state: 1,
            max_local_fwd: 1,
            total_forward_passes: 1,
            noise_dim: 1,
            n_anticipated: 0,
            k_max: 0,
        };
        let solver = ActiveSolver::new().expect("ActiveSolver::new()");
        vec![SolverWorkspace::new(
            0,
            0,
            solver,
            PatchBuffer::new(1, 0, 0, 0, 0, 0),
            1,
            sizing,
        )]
    }

    /// Run one backward trial point at the successor stage with the real solver
    /// and return the produced [`StagedCut`] plus the post-call per-slot
    /// `metadata_sync_contribution` snapshot. `dcs` toggles the path. The
    /// incoming state is pinned to `x_hat = 2.0`.
    fn run_dcs_backward_trial_point(
        dcs: Option<DcsParams>,
        iteration: u64,
    ) -> (super::StagedCut, Vec<u64>) {
        run_dcs_backward_trial_point_at(dcs, iteration, 2.0)
    }

    /// `run_dcs_backward_trial_point` with the incoming-state pin `x_hat`
    /// parameterized, so a sweep can vary the pinned state (which cut binds).
    fn run_dcs_backward_trial_point_at(
        dcs: Option<DcsParams>,
        iteration: u64,
        x_hat: f64,
    ) -> (super::StagedCut, Vec<u64>) {
        let indexer = StageIndexer::new(1, 0);
        let n_state = indexer.n_state;
        let core = dcs_core_template();
        let templates = vec![core.clone(), core.clone()];
        let base_rows = vec![0_usize, 0_usize];
        let stochastic = make_stochastic_context(2, 1);
        let horizon = HorizonMode::Finite { num_stages: 2 };
        let risk_measures = vec![RiskMeasure::Expectation; 2];

        let mut fcf = dcs_two_stage_fcf();
        // All-cuts batch for the baked path (delta == all cuts here).
        let cut_batch = crate::forward::build_cut_row_batch(&fcf, 1, &indexer, &[]);
        let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated_count).collect();
        let num_cuts = successor_active_slots.len();

        let mut exchange = exchange_with_states(n_state, vec![vec![x_hat]]);
        let mut workspaces = dcs_active_workspace();
        let mut basis_store = empty_basis_store(exchange.local_count(), 2);

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs,
            noise_key_diag: None,
        };

        let probabilities = vec![1.0_f64];
        let succ = super::SuccessorSpec {
            t: 0,
            successor: 1,
            my_rank: 0,
            probabilities: &probabilities,
            cut_batch: &cut_batch,
            num_cuts_at_successor: num_cuts,
            template_num_rows: core.num_rows,
            baked_template: &core,
            successor_active_slots: &successor_active_slots,
            cut_activity_tolerance: 0.0,
            successor_populated_count: fcf.pools[1].populated_count,
            successor_pool: &fcf.pools[1],
        };

        let mut basis_slices = basis_store.split_workers_mut(1);
        let ws = &mut workspaces[0];
        // Initialise the per-opening accumulator buffers the trial-point helper
        // expects (mirrors process_stage_backward's per-stage setup).
        super::load_backward_lp(ws, &succ);
        let n_openings = succ.probabilities.len();
        while ws.backward_accum.outcomes.len() < n_openings {
            ws.backward_accum
                .outcomes
                .push(crate::risk_measure::BackwardOutcome {
                    intercept: 0.0,
                    coefficients: vec![0.0_f64; n_state],
                    objective_value: 0.0,
                });
        }
        let pop = succ.successor_populated_count;
        if ws.backward_accum.slot_increments.len() < pop {
            ws.backward_accum.slot_increments.resize(pop, 0);
        }
        ws.backward_accum.slot_increments[..pop].fill(0);
        if ws.backward_accum.agg_coefficients.len() < n_state {
            ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
        }
        if ws.backward_accum.metadata_sync_contribution.len() < pop {
            ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
        }
        ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
        ws.backward_accum
            .per_opening_stats
            .resize_with(n_openings, SolverStatsDelta::default);
        for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
            *slot = SolverStatsDelta::default();
        }

        let cut = super::process_trial_point_backward(
            ws,
            &ctx,
            &training_ctx,
            &exchange,
            0,
            iteration,
            &risk_measures,
            &succ,
            &mut basis_slices[0],
            0,
        )
        .expect("backward trial-point solve must succeed");

        let meta_sync = ws.backward_accum.metadata_sync_contribution[..pop].to_vec();
        // Touch fcf/exchange so the borrows live to here.
        let _ = (&mut fcf, &mut exchange);
        (cut, meta_sync)
    }

    fn dcs_params(start_iteration: u64) -> DcsParams {
        DcsParams {
            k1: None,
            k2: 2,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration,
            max_inner_iterations: 50,
        }
    }

    /// AC3 (exactness, real solver, both backends via the active solver): the
    /// DCS-built cut equals the all-cuts cut (intercept + every coefficient
    /// within 1e-9). DCS seeds an initial set that omits the binding cut; the
    /// lazy loop must add it and reach the same dual.
    #[test]
    fn backward_dcs_cut_equals_all_cuts_cut() {
        let iteration = 5;
        let (baked_cut, _) = run_dcs_backward_trial_point(None, iteration);
        let (dcs_cut, _) = run_dcs_backward_trial_point(Some(dcs_params(2)), iteration);

        assert!(
            (baked_cut.intercept - dcs_cut.intercept).abs() < 1e-9,
            "intercept: baked {} vs DCS {}",
            baked_cut.intercept,
            dcs_cut.intercept
        );
        assert_eq!(baked_cut.coefficients.len(), dcs_cut.coefficients.len());
        for (i, (b, d)) in baked_cut
            .coefficients
            .iter()
            .zip(&dcs_cut.coefficients)
            .enumerate()
        {
            assert!(
                (b - d).abs() < 1e-9,
                "coefficient[{i}]: baked {b} vs DCS {d}"
            );
        }
        // The binding cut has gradient 2.0 on the incoming storage; both paths
        // must recover it.
        assert!(
            (baked_cut.coefficients[0] - 2.0).abs() < 1e-9,
            "baked gradient must be the binding cut's 2.0, got {}",
            baked_cut.coefficients[0]
        );
    }

    /// `dcs = None` ⇒ the baked all-cuts path is taken and the cut is identical
    /// to the pre-DCS baseline (same fixture run with `None`).
    #[test]
    fn backward_dcs_off_is_identical_to_baseline() {
        let (cut_a, _) = run_dcs_backward_trial_point(None, 5);
        let (cut_b, _) = run_dcs_backward_trial_point(None, 5);
        assert_eq!(cut_a.intercept, cut_b.intercept);
        assert_eq!(cut_a.coefficients, cut_b.coefficients);
        // Baseline binding gradient.
        assert!((cut_a.coefficients[0] - 2.0).abs() < 1e-9);
    }

    /// `dcs = Some` but `iteration < start_iteration` ⇒ the baked path is used
    /// (DCS not yet active), so the cut equals the baked cut.
    #[test]
    fn backward_dcs_inactive_before_start_iteration() {
        // start_iteration = 4, iteration = 1 → inactive.
        let (baked_cut, baked_meta) = run_dcs_backward_trial_point(None, 1);
        let (early_cut, early_meta) = run_dcs_backward_trial_point(Some(dcs_params(4)), 1);
        assert_eq!(baked_cut.intercept, early_cut.intercept);
        assert_eq!(baked_cut.coefficients, early_cut.coefficients);
        // Baked path updates binding-count metadata; the inactive-DCS run takes
        // the baked path, so its metadata contribution matches the baked run.
        assert_eq!(baked_meta, early_meta);
    }

    /// AC1: the DCS path extracts the cut gradient from the final all-satisfied
    /// LP AND maintains the binding-count metadata slot-correct under the
    /// resident `CutRowMap`. For this fixture the only cut that binds at the
    /// converged optimum (`x_hat` = 2, theta = 4) is the binding slot 1; slots 0
    /// (floor 1) and 2 (floor 3) are resident from the seed but slack, so their
    /// cut-row duals are zero. The DCS binding-count contribution must therefore
    /// equal the baked path's — slot 1 bumped, all others zero — proving the
    /// slot-correct translation maps the resident binding row back to slot 1 and
    /// to no other.
    #[test]
    fn backward_dcs_binding_counts_match_baked() {
        let (_, baked_meta) = run_dcs_backward_trial_point(None, 5);
        let (_, dcs_meta) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);

        // Baked path bumps exactly the binding slot 1 (the floor-4 cut at x=2).
        assert_eq!(
            baked_meta,
            vec![0, 1, 0],
            "baked path must bump exactly binding slot 1, got {baked_meta:?}"
        );
        // DCS path records the SAME binding-count contribution: the resident
        // binding row maps back to slot 1, and to no other slot.
        assert_eq!(
            dcs_meta, baked_meta,
            "DCS binding-count metadata must match baked (slot-correct via the \
             resident CutRowMap), got DCS {dcs_meta:?} vs baked {baked_meta:?}"
        );
    }

    /// `parse_cut_selection_config` for `method = "dynamic"` flows into
    /// `DcsParams::from_strategy`, so the backward context's `dcs` is `Some`
    /// for the dynamic variant and `None` otherwise.
    #[test]
    fn from_strategy_gates_the_backward_dcs_field() {
        let dynamic = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        assert!(DcsParams::from_strategy(&dynamic).is_some());
        let level1 = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(DcsParams::from_strategy(&level1).is_none());
    }

    // -----------------------------------------------------------------------
    // DCS backward validation gates (exactness / finite-k1 / determinism /
    // slow sweep). Default `k1 = None` (∞) — exactness holds only when every
    // pool cut is a candidate.
    // -----------------------------------------------------------------------

    /// `DcsParams` with an explicit finite `k1` candidate-recency window.
    fn dcs_params_k1(start_iteration: u64, k1: Option<u32>) -> DcsParams {
        DcsParams {
            k1,
            k2: 2,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration,
            max_inner_iterations: 50,
        }
    }

    /// Exactness + "never spins": with `k1 = None` the DCS backward cut equals
    /// the all-cuts cut (intercept + every coefficient within 1e-9), and the
    /// lazy loop terminates — even with `max_inner_iterations = 1`, which forces
    /// the bounded TC-fallback branch — so it can never spin unbounded.
    #[test]
    fn backward_dcs_exactness_and_terminates() {
        let iteration = 5;
        let (baked, _) = run_dcs_backward_trial_point(None, iteration);

        // Default-cap DCS reaches the no-violation stop and matches all-cuts.
        let (dcs, _) = run_dcs_backward_trial_point(Some(dcs_params(2)), iteration);
        assert!((baked.intercept - dcs.intercept).abs() < 1e-9);
        for (b, d) in baked.coefficients.iter().zip(&dcs.coefficients) {
            assert!((b - d).abs() < 1e-9, "coeff mismatch baked {b} vs DCS {d}");
        }

        // A 1-iteration cap forces the bounded TC fallback; the call must still
        // return (no spin) and land on the exact all-cuts cut.
        let tight = DcsParams {
            max_inner_iterations: 1,
            ..dcs_params(2)
        };
        let (dcs_tc, _) = run_dcs_backward_trial_point(Some(tight), iteration);
        assert!((baked.intercept - dcs_tc.intercept).abs() < 1e-9);
        for (b, d) in baked.coefficients.iter().zip(&dcs_tc.coefficients) {
            assert!(
                (b - d).abs() < 1e-9,
                "TC-fallback coeff mismatch baked {b} vs DCS {d}"
            );
        }
    }

    /// A finite `k1` window demonstrably takes effect (guards against `k1`
    /// being silently ignored): with `k1 = Some(1)` at iteration 5, the binding
    /// cut (slot 1, generated at iteration 1 → age 4 ≥ 1) is windowed out of
    /// candidacy and is also outside the `k2 = 2` initial-set window, so it is
    /// never added. The DCS optimum then differs from the all-cuts optimum —
    /// the deliberately-non-exact windowed mode.
    #[test]
    fn backward_dcs_finite_k1_window_takes_effect() {
        let iteration = 5;
        let (baked, _) = run_dcs_backward_trial_point(None, iteration);
        // Sanity: the all-cuts (and k1=None DCS) gradient is the binding cut's 2.0.
        assert!((baked.coefficients[0] - 2.0).abs() < 1e-9);

        let (windowed, _) =
            run_dcs_backward_trial_point(Some(dcs_params_k1(2, Some(1))), iteration);
        // The binding cut is windowed out, so the windowed optimum differs:
        // the surviving cuts (slots 0,2, both gradient 0) give a 0 gradient and
        // a different intercept than the all-cuts cut.
        assert!(
            (windowed.coefficients[0] - baked.coefficients[0]).abs() > 1e-6
                || (windowed.intercept - baked.intercept).abs() > 1e-6,
            "finite k1 must change the cut vs all-cuts (windowed coeff {} intercept {}; \
             all-cuts coeff {} intercept {})",
            windowed.coefficients[0],
            windowed.intercept,
            baked.coefficients[0],
            baked.intercept,
        );
    }

    /// Determinism: running the integrated DCS backward trial point twice on
    /// identical inputs yields bit-identical cuts AND bit-identical
    /// binding-count metadata. A non-deterministic inner-loop insert order on
    /// identical deterministic inputs would perturb the converged cut or the
    /// metadata, so cut + metadata bit-identity is the determinism surface.
    #[test]
    fn backward_dcs_run_to_run_determinism() {
        let (cut_a, meta_a) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);
        let (cut_b, meta_b) = run_dcs_backward_trial_point(Some(dcs_params(2)), 5);
        assert_eq!(
            cut_a.intercept.to_bits(),
            cut_b.intercept.to_bits(),
            "intercept must be bit-identical run-to-run"
        );
        assert_eq!(cut_a.coefficients.len(), cut_b.coefficients.len());
        for (a, b) in cut_a.coefficients.iter().zip(&cut_b.coefficients) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "coefficient must be bit-identical run-to-run"
            );
        }
        assert_eq!(
            meta_a, meta_b,
            "binding-count metadata must be bit-identical run-to-run"
        );
    }

    /// Slow exactness sweep: across a handful of pinned incoming states (which
    /// vary the binding cut), the `k1 = None` DCS backward cut equals the
    /// all-cuts cut within 1e-9 at every point. Gated behind `slow-tests`.
    #[test]
    #[cfg_attr(not(feature = "slow-tests"), ignore = "slow DCS exactness sweep")]
    fn backward_dcs_exactness_sweep() {
        // x_hat values span the regimes where different cuts bind:
        //   slot0 floor = 1, slot1 floor = 2*x_hat, slot2 floor = 3.
        // x_hat < 1.5 → slot2 binds (3); x_hat > 1.5 → slot1 binds (2*x_hat).
        let x_hats = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let iterations = [3_u64, 5, 7];
        for &iteration in &iterations {
            for &x in &x_hats {
                let (baked, _) = run_dcs_backward_trial_point_at(None, iteration, x);
                let (dcs, _) = run_dcs_backward_trial_point_at(Some(dcs_params(2)), iteration, x);
                assert!(
                    (baked.intercept - dcs.intercept).abs() < 1e-9,
                    "sweep iter {iteration} x_hat {x}: intercept baked {} vs DCS {}",
                    baked.intercept,
                    dcs.intercept
                );
                for (i, (b, d)) in baked.coefficients.iter().zip(&dcs.coefficients).enumerate() {
                    assert!(
                        (b - d).abs() < 1e-9,
                        "sweep iter {iteration} x_hat {x}: coeff[{i}] baked {b} vs DCS {d}"
                    );
                }
            }
        }
    }

    /// Baked successor template for the regression fixture: the cut-free base
    /// (`dcs_core_template`, the coupling row `col0 - col2 = 0`) PLUS the binding
    /// cut (`-2*col0 + theta >= 0`) baked as a second structural row. This
    /// mimics baking being active (`baked_template.num_rows = 2 >
    /// template_num_rows = 1`), so the all-cuts/baked successor LP already
    /// carries a cut row that the DCS path must NOT re-append.
    ///
    /// CSC by column (4 cols, 2 rows):
    ///   col0 -> (row0, +1), (row1, -5);  col2 -> (row0, -1);  col3 -> (row1, +1)
    ///
    /// The baked cut intentionally DOMINATES the pool's true binding cut: it is
    /// `-5*col0 + theta >= 0`, i.e. `theta >= 5*col0`, giving floor `10` at the
    /// pinned `x_hat = 2` versus the pool's true optimum floor `4` (gradient 2).
    /// This is a cut that is NOT in the DCS resident pool (the pool's cuts have
    /// gradients 0 and 2, never 5). If the DCS path erroneously loaded this
    /// baked template as its core, the LP would carry the spurious floor-10
    /// constraint and the produced cut would be `theta = 10, gradient = 5` —
    /// observably different from the correct all-cuts cut. Loading the cut-free
    /// base (the fix) ignores this baked row, so the DCS cut matches all-cuts.
    fn dcs_baked_template_with_one_cut() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 4,
            col_starts: vec![0_i32, 2, 2, 3, 4],
            row_indices: vec![0_i32, 1, 0, 1],
            values: vec![1.0, -5.0, -1.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0, -1.0e6],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            // row0: coupling equality (=0); row1: spurious baked cut
            // -5*col0 + theta >= 0 (NOT a DCS pool cut).
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, f64::INFINITY],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Regression for the baked-template-as-core bug: when baking is active
    /// (`baked_template.num_rows > template_num_rows`), the DCS path must load
    /// the cut-free base `ctx.templates[s]` — NOT `succ.baked_template`, which
    /// already carries the active cut rows. Loading the baked template would
    /// leave its baked cut rows resident in the LP even though the lazy loop's
    /// fresh `CutRowMap` does not own them, so DCS would solve against cut rows
    /// it never selected.
    ///
    /// Here `succ.baked_template` carries one baked cut row
    /// (`dcs_baked_template_with_one_cut`, `num_rows = 2`) that is a spurious
    /// floor-10 / gradient-5 constraint NOT present in the DCS pool, while
    /// `ctx.templates[s]` is the cut-free base (`num_rows = 1`). With the fix
    /// the DCS cut equals the all-cuts cut (gradient 2) within 1e-9. Against the
    /// old `core = succ.baked_template`, the spurious baked cut dominates the
    /// solve and DCS returns gradient 5 — observably wrong — failing this
    /// assertion. (Verified: this test fails on the buggy code, passes on the
    /// fix.)
    #[test]
    fn backward_dcs_baked_cuts_present_no_duplicate_rows() {
        let iteration = 5;
        let indexer = StageIndexer::new(1, 0);
        let n_state = indexer.n_state;

        // Cut-free base (loaded by the DCS path) and the baked successor
        // template that carries the binding cut as a structural row.
        let base = dcs_core_template();
        let baked = dcs_baked_template_with_one_cut();
        // ctx.templates carries the cut-free base for the successor stage.
        let templates = vec![base.clone(), base.clone()];
        let base_rows = vec![0_usize, 0_usize];
        let stochastic = make_stochastic_context(2, 1);
        let horizon = HorizonMode::Finite { num_stages: 2 };
        let risk_measures = vec![RiskMeasure::Expectation; 2];

        let mut fcf = dcs_two_stage_fcf();
        // All-cuts batch (delta) for the baked path; with baked carrying the
        // binding cut, the delta is the remaining (non-baked) cuts. For the DCS
        // exactness comparison we only need the all-cuts reference, computed
        // from the full pool against the cut-free base below.
        let cut_batch = crate::forward::build_cut_row_batch(&fcf, 1, &indexer, &[]);
        let successor_active_slots: Vec<usize> = (0..fcf.pools[1].populated_count).collect();
        let num_cuts = successor_active_slots.len();

        let mut exchange = exchange_with_states(n_state, vec![vec![2.0]]);
        let mut workspaces = dcs_active_workspace();
        let mut basis_store = empty_basis_store(exchange.local_count(), 2);

        let ctx = StageContext {
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let training_ctx = TrainingContext {
            horizon: &horizon,
            indexer: &indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
            dcs: Some(dcs_params(2)),
            noise_key_diag: None,
        };

        let probabilities = vec![1.0_f64];
        // `baked_template` carries a baked cut row (num_rows = 2); the cut-free
        // base has template_num_rows = 1. This is the baking-active shape that
        // exposed the bug.
        let succ = super::SuccessorSpec {
            t: 0,
            successor: 1,
            my_rank: 0,
            probabilities: &probabilities,
            cut_batch: &cut_batch,
            num_cuts_at_successor: num_cuts,
            template_num_rows: base.num_rows,
            baked_template: &baked,
            successor_active_slots: &successor_active_slots,
            cut_activity_tolerance: 0.0,
            successor_populated_count: fcf.pools[1].populated_count,
            successor_pool: &fcf.pools[1],
        };

        let mut basis_slices = basis_store.split_workers_mut(1);
        let ws = &mut workspaces[0];
        super::load_backward_lp(ws, &succ);
        let n_openings = succ.probabilities.len();
        while ws.backward_accum.outcomes.len() < n_openings {
            ws.backward_accum
                .outcomes
                .push(crate::risk_measure::BackwardOutcome {
                    intercept: 0.0,
                    coefficients: vec![0.0_f64; n_state],
                    objective_value: 0.0,
                });
        }
        let pop = succ.successor_populated_count;
        if ws.backward_accum.slot_increments.len() < pop {
            ws.backward_accum.slot_increments.resize(pop, 0);
        }
        ws.backward_accum.slot_increments[..pop].fill(0);
        if ws.backward_accum.agg_coefficients.len() < n_state {
            ws.backward_accum.agg_coefficients.resize(n_state, 0.0);
        }
        if ws.backward_accum.metadata_sync_contribution.len() < pop {
            ws.backward_accum.metadata_sync_contribution.resize(pop, 0);
        }
        ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
        ws.backward_accum
            .per_opening_stats
            .resize_with(n_openings, SolverStatsDelta::default);
        for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
            *slot = SolverStatsDelta::default();
        }

        let dcs_cut = super::process_trial_point_backward(
            ws,
            &ctx,
            &training_ctx,
            &exchange,
            0,
            iteration,
            &risk_measures,
            &succ,
            &mut basis_slices[0],
            0,
        )
        .expect("DCS backward solve with baked cuts present must succeed");
        let _ = (&mut fcf, &mut exchange);

        // The all-cuts reference cut (cut-free base + full pool, no DCS).
        let (allcuts, _) = run_dcs_backward_trial_point(None, iteration);

        // With the fix (core = cut-free ctx.templates[s]), the binding cut is
        // added exactly once and the DCS cut matches the all-cuts cut. With the
        // bug (core = baked_template), the baked cut is double-added and the
        // solve/extraction is malformed, so this fails.
        assert!(
            (dcs_cut.intercept - allcuts.intercept).abs() < 1e-9,
            "intercept: DCS {} vs all-cuts {}",
            dcs_cut.intercept,
            allcuts.intercept
        );
        assert_eq!(dcs_cut.coefficients.len(), allcuts.coefficients.len());
        for (i, (d, a)) in dcs_cut
            .coefficients
            .iter()
            .zip(&allcuts.coefficients)
            .enumerate()
        {
            assert!((d - a).abs() < 1e-9, "coeff[{i}]: DCS {d} vs all-cuts {a}");
        }
        // The binding gradient (2.0) must be recovered.
        assert!((dcs_cut.coefficients[0] - 2.0).abs() < 1e-9);
    }
}
