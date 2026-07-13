//! Backward pass execution for the SDDP training loop.
//!
//! Sweeps stages in reverse, evaluating the cost-to-go at each trial point
//! assigned to this rank, extracting LP duals into Benders cut coefficients, and
//! aggregating per-opening outcomes via
//! [`RiskMeasure::aggregate_cut`](crate::RiskMeasure::aggregate_cut) into one cut
//! per trial point per stage inserted into the [`crate::FutureCostFunction`].
//!
//! Each rank processes only its own forward-pass assignments to avoid generating
//! duplicate cuts; cut synchronization (`allgatherv`) distributes them to all
//! ranks afterward.
//!
//! ## Stage indexing convention
//!
//! The backward pass generates a cut **at stage `t`** by solving the LP
//! **at stage `t + 1`** (the successor) under each opening noise vector from
//! that successor stage. The opening tree provides noise at `t + 1`.
//!
//! ## Cut coefficient formula
//!
//! `coefficients = reduced_cost` (raw, no sign flip at extraction); the LP cut row
//! negates it in `cut::row::build_cut_row_batch_into`. The full subgradient
//! contract (`pi[i] = reduced_cost[col_i] / col_scale[col_i]`, divided not
//! multiplied) lives in `duals_extraction` and sddp.md "Benders cut sign &
//! subgradient extraction".
//!
//! ### Anticipated-ring cut gradient flow
//!
//! Anticipated-ring slots resolve by identity (`state_to_lp_column`, the
//! `transit_buckets_out` convention): the in-LP ring's definition rows — a
//! plain shift for slot `i < K_p-1`, the delivery-decision deposit for plant
//! `p`'s own newest slot `K_p-1` — resolve the ring transition, so cuts apply
//! directly against the outgoing `anticipated_slots_out` column. The fishing
//! constraint is emitted at every stage unconditionally, so every slot
//! participates in the dual chain. See the `StateLayout::state_to_lp_column`
//! rustdoc.
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
//! The outer per-stage loop is sequential (stage `t` depends on cuts generated at
//! stage `t+1`); the inner trial-point loop is parallelised across
//! [`SolverWorkspace`](crate::workspace::SolverWorkspace) instances with static
//! scenario partitioning. Each worker generates cuts into a thread-local
//! `StagedCut` buffer, sorted by `trial_point_idx` after the parallel region for
//! deterministic FCF insertion regardless of thread completion order.

use cobre_solver::RowBatch;
use cobre_solver::StageTemplate;

use crate::{cut::pool::CutPool, indexer::CutStateProjection, solver_stats::SolverStatsDelta};

use std::ops::Range;

mod duals_extraction;
mod lp_setup;
mod outcome_aggregation;
mod trial_point;

#[cfg(test)]
mod tests;

pub(crate) use trial_point::{StageOpeningSolver, process_trial_point_backward};

#[cfg(test)]
pub(crate) use lp_setup::{load_backward_lp, patch_opening_bounds, resolve_backward_basis};

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
    /// Total cuts generated during the backward pass, summed across all ranks
    /// (rank-count invariant — the globally-replicated pool's per-iteration growth).
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
    pub(crate) coefficients_range: Range<usize>,

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
    /// Delta batch when freeze is active, full active-cut batch otherwise.
    pub(crate) cut_batch: &'a RowBatch,
    /// Total number of active cuts at the successor stage for dual extraction.
    /// Includes both frozen and delta cuts contiguous after `template_num_rows`.
    pub(crate) num_cuts_at_successor: usize,
    /// Base row count of the successor template (excludes cuts).
    pub(crate) template_num_rows: usize,
    /// Frozen LP template for the successor stage. Always populated — freeze
    /// is complete before the backward pass begins.
    pub(crate) frozen_template: &'a StageTemplate,
    /// Ordered slot indices of the active cuts at the successor stage.
    pub(crate) successor_active_slots: &'a [usize],
    /// Minimum dual multiplier for a cut to count as binding.
    pub(crate) cut_activity_tolerance: f64,
    /// Populated count of the successor's cut pool.
    pub(crate) successor_populated_count: usize,
    /// Cut pool at the successor stage for binding-activity tracking.
    pub(crate) successor_pool: &'a CutPool,
    /// Cut-state projection for the pool this stage's cut is inserted into (pool
    /// `t`, sized from `stages[t+1].state_config`): the LP incoming-state columns
    /// dual extraction reads and the dimension every per-stage backward buffer is
    /// sized to. `n_state()` equals `successor_pool.state_dimension`.
    pub(crate) cut_state: &'a CutStateProjection,
}
