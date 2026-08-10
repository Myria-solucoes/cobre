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
//! same-slot carry (`out − in = 0`) for a not-yet-due slot, the
//! delivery-decision deposit into the slot of a decision's own delivery target
//! (`delivery_stage mod k_max`) — resolve the ring transition, so cuts apply
//! directly against the outgoing `commit_out` column. The fishing
//! constraint is emitted at every stage unconditionally, so every slot
//! participates in the dual chain. See the `StateSpace::state_to_lp_column`
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
//! `StagedCut` buffer, sorted by `trial_state_idx` after the parallel region for
//! deterministic FCF insertion regardless of thread completion order.

use cobre_solver::{RowBatch, StageTemplate};

use crate::{
    cut::pool::CutPool,
    indexer::CutStateProjection,
    setup::node_graph::{NodeId, NodeOpenings, NodePos, StageIdx},
    solver_stats::SolverStatsDelta,
};

use std::ops::Range;

mod by_node;
mod by_scenario;
mod duals_extraction;
mod lp_setup;
mod outcome_aggregation;
mod replicated;

#[cfg(test)]
mod tests;

pub(crate) use by_node::{
    OpeningOutcome, by_node_block_count, by_node_finish, hardest_first_block_order,
    identity_block_order, merge_block_pivots, process_stage_backward_by_node, resolve_block_size,
};
pub(crate) use by_scenario::{StageOpeningSolver, process_by_scenario_backward};
pub(crate) use lp_setup::fill_external_opening_noise;
pub(crate) use replicated::{ReplicatedScratch, run_backward_node_replicated};

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
/// staged cuts are sorted by `trial_state_idx` and merged into the FCF in
/// deterministic order regardless of thread completion order.
pub(crate) struct StagedCut {
    /// Local trial-point index within `0..local_work`. Used for deterministic
    /// merge ordering after the parallel region.
    pub(crate) trial_state_idx: usize,

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

/// Node-level backward-pass arguments at one cut-generating node, shared across
/// every successor child. The per-child successor data lives in
/// [`SuccessorOutcomes`]; a child is priced against ITS OWN LP, never child 0's.
pub(crate) struct SuccessorSpec<'a> {
    /// Stage index being cut (the stage whose cost-to-go we are computing).
    pub(crate) t: StageIdx,
    /// Successor stage index (`t + 1`), where the LPs are solved. Every child of
    /// the node sits at this one stage (`t -> t+1`, no stage-skipping — validation
    /// rule 39), so it is node-level, not per-child.
    pub(crate) successor: StageIdx,
    /// This rank's MPI rank index (used to address exchange buffer state).
    pub(crate) my_rank: usize,
    /// Product weights `P(n→m) · q_{m,ψ}` over the current node's successor
    /// outcome set `O(n) = {(m, ψ): m ∈ n⁺, ψ ∈ Ω_m}`, flattened in canonical
    /// order (ascending child node id, then within-child ω) — the same order the
    /// [`SuccessorOutcomes`] entries and their `outcome_range`s follow.
    pub(crate) probabilities: &'a [f64],
    /// Cut-state projection for the pool this stage's cut is inserted into
    /// (the pool id resolved from the node graph for the node at stage `t`,
    /// sized from its successor's `state_config`): the LP incoming-state
    /// columns dual extraction reads. `n_slots()` equals every successor pool's
    /// `state_dimension` (all children share one stage and `state_config`).
    pub(crate) cut_state: &'a CutStateProjection,
}

/// Owned per-child metadata for one successor outcome. Borrow-free so it lives in a
/// reused per-node buffer (no hot-path allocation); the pool-indexed LP data
/// (frozen template, delta cut batch, active slots, cut pool) is resolved on demand
/// by [`SuccessorOutcomes::child`].
pub(crate) struct SuccessorEntry {
    /// Child's canonical node position (`NodeGraph::nodes` index) — the warm-start
    /// `BasisStore` node-axis key. On the chain degeneracy equal to the successor stage.
    pub(crate) successor_node: NodePos,
    /// Child's declared node id (`NodeGraph::node_ids[successor_node]`).
    pub(crate) successor_node_id: NodeId,
    /// Child's pool id (`FutureCostFunction::pools` index).
    pub(crate) pool_id: usize,
    /// Total active cuts at the child for dual extraction — the child's OWN frozen
    /// pool rows plus the child's OWN delta (mixing pools corrupts warm-start slots).
    pub(crate) num_cuts_at_successor: usize,
    /// Populated count of the child's cut pool.
    pub(crate) populated_count: usize,
    /// This child's slice of the shared active-slots buffer.
    pub(crate) active_slots: Range<usize>,
    /// Base offset of this child's cut pool's region in the concatenated
    /// binding-metadata buffers (`slot_increments`/`metadata_sync_contribution`) —
    /// so each child's binding metadata lands in ITS OWN pool's slots, never child
    /// 0's. Two children never share a non-empty pool (a non-leaf owns its own pool;
    /// leaves share an empty terminal pool), so per-child offsets separate distinct
    /// pools while shared empty-pool children stay collision-free (`populated == 0`).
    pub(crate) metadata_offset: usize,
    /// Child's Ω view (source + offset + len).
    pub(crate) openings: NodeOpenings,
    /// This child's contiguous slice of the flattened outcome/weight vector.
    pub(crate) outcome_range: Range<usize>,
}

/// A fully-resolved successor child: [`SuccessorEntry`] metadata plus the
/// pool-indexed LP data resolved from the node's arrays. The leaf backward helpers
/// read per-child data from this, so pricing a child against another child's LP —
/// the child-0 collapse — is unrepresentable.
pub(crate) struct SuccessorChild<'a> {
    /// Child's canonical node position — the warm-start `BasisStore` node-axis key.
    pub(crate) successor_node: NodePos,
    /// Child's declared node id — tags the captured basis and the solve target.
    pub(crate) successor_node_id: NodeId,
    /// Child's pool id.
    pub(crate) pool_id: usize,
    /// Child's frozen LP template (`frozen[pool_id]`).
    pub(crate) frozen_template: &'a StageTemplate,
    /// Child's delta cut rows to append to its LP (`cut_batches[pool_id]`).
    pub(crate) cut_batch: &'a RowBatch,
    /// Total active cuts at the child (frozen pool rows + delta).
    pub(crate) num_cuts_at_successor: usize,
    /// Base row count of the successor stage template (node-level; excludes cuts).
    pub(crate) template_num_rows: usize,
    /// Minimum dual multiplier for a cut to count as binding (node-level).
    pub(crate) cut_activity_tolerance: f64,
    /// Ordered slot indices of the child's active cuts.
    pub(crate) successor_active_slots: &'a [usize],
    /// Populated count of the child's cut pool.
    pub(crate) populated_count: usize,
    /// Base offset of the child's pool region in the binding-metadata buffers.
    pub(crate) metadata_offset: usize,
    /// Child's cut pool (binding-activity tracking, basis-capture metadata).
    pub(crate) successor_pool: &'a CutPool,
    /// Child's Ω view (source + offset + len).
    pub(crate) openings: NodeOpenings,
    /// This child's contiguous slice of the flattened outcome/weight vector.
    pub(crate) outcome_range: Range<usize>,
}

/// The current node's reified successor outcome set: one [`SuccessorEntry`] per
/// child, canonical order (ascending child node id, then within-child ω — the order
/// `assemble_outcome_weights` produces, so the weight slices align with the outcome
/// arena by construction), over the node's pool-indexed LP arrays. A chain is the
/// one-element case — chain byte-parity is preserved by that degeneracy, never by a
/// graph-shape predicate.
pub(crate) struct SuccessorOutcomes<'a> {
    entries: &'a [SuccessorEntry],
    active_slots_buf: &'a [usize],
    frozen: &'a [StageTemplate],
    cut_batches: &'a [RowBatch],
    pools: &'a [CutPool],
    template_num_rows: usize,
    cut_activity_tolerance: f64,
}

impl<'a> SuccessorOutcomes<'a> {
    /// Build the view over the reused metadata buffer and the node's pool-indexed
    /// arrays.
    pub(crate) fn new(
        entries: &'a [SuccessorEntry],
        active_slots_buf: &'a [usize],
        frozen: &'a [StageTemplate],
        cut_batches: &'a [RowBatch],
        pools: &'a [CutPool],
        template_num_rows: usize,
        cut_activity_tolerance: f64,
    ) -> Self {
        Self {
            entries,
            active_slots_buf,
            frozen,
            cut_batches,
            pools,
            template_num_rows,
            cut_activity_tolerance,
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        self.entries.len()
    }

    /// Total flattened outcomes `Σ_(m∈successors)|Ω_m|`; must equal `probabilities.len()`.
    pub(crate) fn total_outcomes(&self) -> usize {
        self.entries.iter().map(|e| e.outcome_range.len()).sum()
    }

    /// Total length of the per-worker binding-metadata buffers: the sum of every
    /// child's cut-pool populated count, so each child's pool occupies its own
    /// non-overlapping slot region (`metadata_offset..metadata_offset + populated`).
    pub(crate) fn total_metadata_len(&self) -> usize {
        self.entries.iter().map(|e| e.populated_count).sum()
    }

    /// Resolve child `i`'s full LP bundle from the pool-indexed arrays.
    pub(crate) fn child(&self, i: usize) -> SuccessorChild<'a> {
        let e = &self.entries[i];
        SuccessorChild {
            successor_node: e.successor_node,
            successor_node_id: e.successor_node_id,
            pool_id: e.pool_id,
            frozen_template: &self.frozen[e.pool_id],
            cut_batch: &self.cut_batches[e.pool_id],
            num_cuts_at_successor: e.num_cuts_at_successor,
            template_num_rows: self.template_num_rows,
            cut_activity_tolerance: self.cut_activity_tolerance,
            successor_active_slots: &self.active_slots_buf[e.active_slots.clone()],
            populated_count: e.populated_count,
            metadata_offset: e.metadata_offset,
            successor_pool: &self.pools[e.pool_id],
            openings: e.openings,
            outcome_range: e.outcome_range.clone(),
        }
    }
}
