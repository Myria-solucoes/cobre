//! Backward pass execution for the SDDP training loop.
//!
//! `run_backward_pass` sweeps stages in reverse order (`T-2` down to `0`),
//! evaluating the cost-to-go at each trial point **assigned to this rank**
//! during the forward pass. For each trial point, the backward pass iterates
//! over every opening from the fixed opening tree, extracts LP duals to form
//! Benders cut coefficients, and aggregates per-opening outcomes via
//! [`RiskMeasure::aggregate_cut`](crate::RiskMeasure::aggregate_cut) to produce
//! one cut per trial point per stage. Each aggregated cut is inserted into the
//! [`crate::FutureCostFunction`].
//!
//! Although [`ExchangeBuffers`](crate::ExchangeBuffers) contains trial points from all ranks (after
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
//! reduced cost (the dual of the `lb == ub` pin — equal to the
//! fixing-row dual by KKT stationarity). State fixing uses column bounds, not
//! equality rows; the gradient is a reduced cost,
//! unscaled by `col_scale` (not `row_scale`). The convention is
//! `coefficients = reduced_cost` (raw, no sign flip at extraction). Negation is
//! applied later when building the LP cut row in
//! `cut::row::build_cut_row_batch_into`:
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
//! `dQ_t/dx_ant[slot, plant]`. Because the fishing constraint is always active
//! for every anticipated plant, every slot at every stage participates in the
//! dual chain: the fishing constraint is emitted at every stage
//! unconditionally. The dual on the Cat 6
//! state-fixing row at slot `s` flows back to the predecessor's LP column via
//! `state_to_lp_column`'s branch decision (Less / Equal / Greater), which maps
//! slot `K_p-1` to the decision column and slot `i < K_p-1` to slot `i+1`.
//! See the `StateLayout::state_to_lp_column` rustdoc for the slot-to-column
//! branch mapping that drives the dual chain.
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
//! parallelised across [`SolverWorkspace`](crate::workspace::SolverWorkspace) instances using rayon's
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

use cobre_solver::RowBatch;

use crate::{cut::pool::CutPool, solver_stats::SolverStatsDelta};

mod duals_extraction;
mod lp_setup;
mod outcome_aggregation;
mod trial_point;

#[cfg(test)]
mod tests;

// Re-export the crate-internal backward surface so `crate::backward::X`,
// `cobre_sddp::backward::X`, and `training::backward::X` resolve verbatim. The
// dense grouped consumer `use crate::{ backward::{BackwardResult,
// StageOpeningSolver, StageWorkerOpeningDelta, StagedCut, SuccessorSpec,
// process_trial_point_backward} }` in `training/backward_pass_state.rs` reaches
// `StageOpeningSolver` + `process_trial_point_backward` here (the four data types
// are defined directly in this module). `load_backward_lp` and
// `resolve_backward_basis` have no non-test crate consumer — `trial_point` reaches
// them through the `super::lp_setup::` path — so they are re-exported only under
// `#[cfg(test)]` for the in-module tests' `super::load_backward_lp` /
// `super::resolve_backward_basis` call sites.
pub(crate) use trial_point::{StageOpeningSolver, process_trial_point_backward};

#[cfg(test)]
pub(crate) use lp_setup::{load_backward_lp, resolve_backward_basis};

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

    /// Range into the producing worker's `agg_arena` holding this cut's
    /// aggregated coefficients (length = `n_state`). The arena is owned by the
    /// rayon worker that produced this cut; the FCF merge resolves the slice
    /// after the parallel region returns via
    /// `workspaces[w].backward_accum.agg_arena[coefficients_range]`.
    pub(crate) coefficients_range: std::ops::Range<usize>,

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
