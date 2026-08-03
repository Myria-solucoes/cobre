//! Backward pass state management and entry point.
//!
//! [`BackwardPassState`] owns pre-allocated scratch buffers reused each iteration.
//! [`BackwardPassInputs`] bundles per-call borrowed inputs (no allocation on hot path).

use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{TrainingEvent, WorkerPhaseTimings, WorkerTimingPhase};
use cobre_io::config::BackwardScheduler;
use cobre_solver::ActiveProfile;
use cobre_solver::{RowBatch, SolverInterface, SolverStatistics, StageTemplate};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::risk_measure::BackwardOutcome;
use crate::setup::node_graph::{NodeGraph, backward_cut_levels};
use crate::{
    backward::{
        BackwardResult, StageOpeningSolver, StageWorkerOpeningDelta, StagedCut, SuccessorSpec,
        by_node_block_count, by_node_finish, hardest_first_block_order, identity_block_order,
        merge_block_pivots, outcome_stride, process_by_scenario_backward,
        process_stage_backward_by_node, resolve_block_size, solve_replicated_outcome_slice,
    },
    config::CutManagementConfig,
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut_sync::{CutSyncBuffers, OutcomeExchangeScratch},
    error::SddpError,
    forward::build_delta_cut_row_batch_into,
    rank_reconcile::reconcile_result,
    risk_measure::RiskMeasure,
    solve::partition,
    solver_phase::Phase,
    solver_stats::{
        SolverStatsDelta, StageWorkerStatsBuffer, WORKER_STATS_ENTRY_STRIDE,
        pack_worker_opening_stats, unpack_worker_opening_stats,
    },
    state_exchange::ExchangeBuffers,
    training_session::{rank_distribution::RankDistribution, runtime::RuntimeHandles},
    trajectory::TrajectoryRecord,
    visited_states::VisitedStatesArchive,
    workspace::{BasisStore, BasisStoreSliceMut, ByNodeScratch, SolverWorkspace, WorkspacePool},
};

/// Per-iteration argument bundle for [`BackwardPassState::run`].
///
/// Groups all borrowed inputs that vary between calls: exchange buffers,
/// trajectory records, risk measures, cut-sync state, and the event sender.
/// Owned scratch buffers live on [`BackwardPassState`] and are not repeated
/// here.
pub struct BackwardPassInputs<'a, S: SolverInterface + Send, C: Communicator> {
    /// Solver workspaces (one per rayon worker thread).
    pub workspaces: &'a mut [SolverWorkspace<S>],
    /// Basis warm-start store (one slot per `(scenario, stage)` pair).
    pub basis_store: &'a mut BasisStore,
    /// Stage-level LP context (templates, row counts, noise scales).
    pub ctx: &'a StageContext<'a>,
    /// Frozen LP templates including pre-appended prior-iteration cuts.
    pub frozen: &'a [StageTemplate],
    /// Future-cost function — receives new cuts after each stage.
    pub fcf: &'a mut FutureCostFunction,
    /// Per-stage delta cut row batches (reused scratch, resized per stage).
    pub cut_batches: &'a mut [RowBatch],
    /// Study-level training context (horizon, indexer, stochastic model).
    pub training_ctx: &'a TrainingContext<'a>,
    /// MPI communicator.
    pub comm: &'a C,

    /// Exchange buffers for gathering trial-point states via `allgatherv`.
    pub exchange: &'a mut ExchangeBuffers,

    /// Forward-pass trajectory records used to populate `exchange` per stage.
    ///
    /// Length is `max_local_fwd * num_stages` — rank-uniform, NOT this rank's
    /// `local_work`; a rank drawing zero trial points still passes full-length
    /// padding, which `real_total_scenarios` discards after the gather. Re-slicing
    /// to `local_work * num_stages` (as `ForwardPassInputs` correctly does) empties
    /// it on exactly those ranks and desynchronises the per-stage `allgatherv`.
    pub records: &'a [TrajectoryRecord],

    /// Pre-allocated cut synchronisation buffers for per-stage `allgatherv`.
    pub cut_sync_bufs: &'a mut CutSyncBuffers,

    /// Optional visited-states archive for dominated cut selection.
    pub visited_archive: Option<&'a mut VisitedStatesArchive>,

    /// Optional event channel for emitting [`TrainingEvent::WorkerTiming`] events.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,

    /// Per-stage risk measures (length = `num_stages`).
    pub risk_measures: &'a [RiskMeasure],

    /// Minimum dual multiplier for a cut to count as binding.
    pub cut_activity_tolerance: f64,

    /// Current training iteration index (1-based), used for cut metadata.
    pub iteration: u64,

    /// Number of trial points assigned to this rank for the backward pass.
    pub local_work: usize,

    /// Global offset for this rank's trial points (`rank * fwd_per_rank`).
    pub fwd_offset: usize,
}

impl<'a, S: SolverInterface + Send, C: Communicator> BackwardPassInputs<'a, S, C> {
    /// Construct inputs from the fields of a `TrainingSession`, minus `bwd_state`.
    // Rationale: the arguments are disjoint mutable borrows into `TrainingSession` fields that
    // Rust NLL cannot split from a single `&mut self`; bundling them into a context struct would
    // just move the arity to the struct literal without resolving the borrow-splitting requirement.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_session_fields(
        fwd_pool: &'a mut WorkspacePool<S>,
        basis_store: &'a mut BasisStore,
        ctx: &'a StageContext<'a>,
        frozen_templates: &'a [StageTemplate],
        cut_batches: &'a mut [RowBatch],
        records: &'a [TrajectoryRecord],
        fcf: &'a mut FutureCostFunction,
        exchange: &'a mut ExchangeBuffers,
        cut_sync_bufs: &'a mut CutSyncBuffers,
        visited_archive: &'a mut Option<VisitedStatesArchive>,
        training_ctx: &'a TrainingContext<'a>,
        comm: &'a C,
        cut_mgmt: &'a CutManagementConfig,
        ranks: &RankDistribution,
        runtime: &'a RuntimeHandles,
        iteration: u64,
    ) -> Self {
        Self {
            workspaces: &mut fwd_pool.workspaces,
            basis_store,
            ctx,
            frozen: frozen_templates,
            fcf,
            cut_batches,
            training_ctx,
            comm,
            exchange,
            records,
            cut_sync_bufs,
            visited_archive: visited_archive.as_mut(),
            event_sender: runtime.event_sender(),
            risk_measures: &cut_mgmt.risk_measures,
            cut_activity_tolerance: cut_mgmt.cut_activity_tolerance,
            iteration,
            local_work: ranks.my_actual_fwd,
            fwd_offset: ranks.my_fwd_offset,
        }
    }
}

/// Owned scratch buffers for the backward pass, allocated once and reused.
///
/// `BackwardPassState` is constructed once by `TrainingSession::new` and stored
/// as a field on `TrainingSession`. The buffers are pre-sized from the study
/// dimensions and reused across every iteration via `clear()` / `resize()` /
/// `fill()`. No allocation occurs on the hot path.
///
/// Per-iteration inputs (exchange buffers, records, risk measures, etc.) are
/// passed via [`BackwardPassInputs`] at each `run()` call.
pub struct BackwardPassState {
    /// Uniform opening probabilities.
    pub(crate) probabilities_buf: Vec<f64>,

    /// Successor active cut slot indices.
    pub(crate) successor_active_slots_buf: Vec<usize>,

    /// Per-slot binding increment aggregation, source of the `allreduce(Sum)` that
    /// synchronises cut binding metadata across MPI ranks.
    pub(crate) metadata_sync_buf: Vec<u64>,

    /// Receive buffer for the `allreduce(Sum)` aggregating per-slot binding
    /// increments across MPI ranks.
    pub(crate) global_increments_buf: Vec<u64>,

    /// Packs real (non-padded) gathered state vectors when archiving visited
    /// states for dominated cut selection.
    pub(crate) real_states_buf: Vec<f64>,

    /// Per-(worker, opening) gather buffer for backward-pass solver stats.
    ///
    /// Shape: `n_workers_local × max_openings`. Reset at the start of each stage.
    pub(crate) stage_worker_stats_buf: StageWorkerStatsBuffer,

    /// MPI send buffer for the per-`(worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    pub(crate) bwd_stats_send_buf: Vec<f64>,

    /// MPI receive buffer for the per-`(rank, worker, opening)` stats `allgatherv`.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE`.
    pub(crate) bwd_stats_recv_buf: Vec<f64>,

    /// Per-rank element counts for the `allgatherv` of backward stats.
    pub(crate) bwd_stats_counts: Vec<usize>,

    /// Displacement array for the `allgatherv` of backward stats.
    pub(crate) bwd_stats_displs: Vec<usize>,

    /// Unpack destination buffer for the per-`(rank, worker, opening)` stats.
    ///
    /// Length: `n_ranks * n_workers_local * bwd_max_openings` `SolverStatsDelta` entries.
    pub(crate) bwd_stats_unpack_buf: Vec<SolverStatsDelta>,

    // ── Per-iteration scratch (reused across stages within one `run()` call) ──
    /// Staging buffer for cuts produced by one stage's parallel trial-point
    /// loop, each paired with the index `w` of the worker that produced it.
    ///
    /// The worker index resolves the cut's `coefficients_range` against
    /// `workspaces[w].backward_accum.agg_arena` at merge time. Cleared at the
    /// start of each stage and grown monotonically.
    pub(crate) staged_cuts_buf: Vec<(usize, StagedCut)>,

    /// Per-worker solver statistics snapshot taken **before** the stage's parallel
    /// region.
    pub(crate) worker_stats_before: Vec<SolverStatistics>,

    /// Per-worker solver statistics snapshot taken **after** the stage's parallel
    /// region.
    pub(crate) worker_stats_after: Vec<SolverStatistics>,

    /// Per-worker solver-statistics delta for this stage (after − before).
    pub(crate) worker_deltas: Vec<SolverStatsDelta>,

    /// Per-worker total work time (solve + load + set-bounds + basis-set) for
    /// load-imbalance decomposition.
    pub(crate) worker_totals: Vec<f64>,

    /// Cross-rank error-reconciliation scratch, reused each stage by the
    /// pre-`sync_packed_records` reconcile so that reconciliation never allocates.
    pub(crate) reconcile_scratch: [i32; 1],

    /// Resolved backward-phase solver profile applied at [`Self::run`] entry.
    /// Defaults to `Phase::Backward.profile()`; override with
    /// [`Self::set_profile`] before the first `run()` call.
    profile: ActiveProfile,

    /// Backward-pass work-unit scheduler, carrying the opening-block size
    /// when the `ByNode` method is selected. Defaults to
    /// [`BackwardScheduler::ByScenario`] (byte-neutral with the pre-scheduler
    /// path); override with [`Self::set_scheduler`] before the first `run()`
    /// call.
    scheduler: BackwardScheduler,

    /// Whether the by-node scheduler claims hardest-`(stage,
    /// block)`-first using [`Self::by_node_scratch`]'s
    /// `block_pivots_prev` row, or the canonical ascending block order.
    /// Defaults to `true`; override with [`Self::set_hardest_first_claim_order`]
    /// before the first `run()` call.
    hardest_first_claim_order: bool,

    /// Maximum local forward-pass count across the run; with
    /// `bwd_max_openings` and `n_state`, sizes [`Self::by_node_scratch`] when
    /// [`Self::set_scheduler`] resolves `ByNode`.
    max_local_fwd: usize,

    /// Maximum opening count across all stages; see [`Self::max_local_fwd`].
    bwd_max_openings: usize,

    /// State dimension; see [`Self::max_local_fwd`].
    n_state: usize,

    /// Number of stages in the study; with `bwd_max_openings`, sizes
    /// [`Self::by_node_scratch`]'s per-`(stage, block-index)` pivot accumulator
    /// when [`Self::set_scheduler`] resolves `ByNode`.
    num_stages: usize,

    /// Pre-allocated by-node scheduler scratch (per-`(m, ω)` outcome
    /// arena + aggregation buffers). Empty until [`Self::set_scheduler`] sizes
    /// it for `BackwardScheduler::ByNode` (sddp.md "By-node
    /// scheduler is warm-start-only").
    by_node_scratch: ByNodeScratch,

    /// Per-level node-compute results, reused across levels (mem-swapped out in
    /// `run_one_backward_level` so the level loop allocates no per-level buffer).
    level_nodes_scratch: Vec<NodeCompute>,

    /// Per-level trial-point-distributed pool list for the batched cut exchange,
    /// reused across levels alongside `level_nodes_scratch`.
    level_pools_scratch: Vec<usize>,

    /// Per-level routing of this rank's own trial points to the level's
    /// cut-generating nodes, CSR values: `routed_trials_scratch[
    /// routed_offsets_scratch[i]..routed_offsets_scratch[i + 1]]` are the
    /// ascending trial-point indices whose stage visit resolves to `level[i]`'s
    /// pool. Reused across levels alongside `level_nodes_scratch`.
    routed_trials_scratch: Vec<usize>,

    /// CSR offsets into `routed_trials_scratch`: one entry per level node plus a
    /// trailing bound (`offsets[i]..offsets[i + 1]` bounds level node `i`).
    routed_offsets_scratch: Vec<usize>,
}

impl BackwardPassState {
    /// Allocate all scratch buffers sized for the given study dimensions.
    ///
    /// # Parameters
    ///
    /// - `n_workers_local`: number of rayon worker threads on this rank.
    /// - `n_ranks`: total MPI rank count.
    /// - `bwd_max_openings`: maximum opening count across all stages.
    /// - `real_states_capacity`: capacity hint for `real_states_buf`
    ///   (`real_total_scenarios * n_state`).
    /// - `max_local_fwd`: maximum local forward-pass count across the run.
    /// - `n_state`: state dimension.
    /// - `num_stages`: number of stages in the study.
    ///
    /// `max_local_fwd`, `n_state`, and `num_stages`, together with
    /// `bwd_max_openings`, size `Self::by_node_scratch` when
    /// [`Self::set_scheduler`] later resolves `ByNode`; `by_node_scratch`
    /// starts empty regardless of the resolved scheduler.
    #[must_use]
    pub fn new(
        n_workers_local: usize,
        n_ranks: usize,
        bwd_max_openings: usize,
        real_states_capacity: usize,
        max_local_fwd: usize,
        n_state: usize,
        num_stages: usize,
    ) -> Self {
        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;
        Self {
            probabilities_buf: Vec::new(),
            successor_active_slots_buf: Vec::new(),
            metadata_sync_buf: Vec::new(),
            global_increments_buf: Vec::new(),
            real_states_buf: Vec::with_capacity(real_states_capacity),
            stage_worker_stats_buf: StageWorkerStatsBuffer::new(n_workers_local, bwd_max_openings),
            bwd_stats_send_buf: vec![0.0; send_stride],
            bwd_stats_recv_buf: vec![0.0; n_ranks * send_stride],
            bwd_stats_counts: vec![send_stride; n_ranks],
            bwd_stats_displs: (0..n_ranks).map(|r| r * send_stride).collect(),
            bwd_stats_unpack_buf: vec![
                SolverStatsDelta::default();
                n_ranks * n_workers_local * bwd_max_openings
            ],
            staged_cuts_buf: Vec::new(),
            worker_stats_before: Vec::with_capacity(n_workers_local),
            worker_stats_after: Vec::with_capacity(n_workers_local),
            worker_deltas: Vec::with_capacity(n_workers_local),
            worker_totals: Vec::with_capacity(n_workers_local),
            reconcile_scratch: [0_i32; 1],
            profile: Phase::Backward.profile(),
            scheduler: BackwardScheduler::default(),
            hardest_first_claim_order: true,
            max_local_fwd,
            bwd_max_openings,
            n_state,
            num_stages,
            by_node_scratch: ByNodeScratch::default(),
            level_nodes_scratch: Vec::new(),
            level_pools_scratch: Vec::new(),
            routed_trials_scratch: Vec::new(),
            routed_offsets_scratch: Vec::new(),
        }
    }

    /// Overrides the backward-phase solver profile applied at [`Self::run`]
    /// entry (default: `Phase::Backward.profile()`). Call before `run()`.
    pub fn set_profile(&mut self, profile: ActiveProfile) {
        self.profile = profile;
    }

    /// Overrides the backward-pass scheduler applied at [`Self::run`] entry
    /// (default: [`BackwardScheduler::ByScenario`]). Call before `run()`.
    ///
    /// Sizes `Self::by_node_scratch` once, here, from the dimensions passed to
    /// [`Self::new`]: the full `max_local_fwd * bwd_max_openings` shape for
    /// `ByNode`, empty (zero opening-block footprint) for `ByScenario` —
    /// never on the hot path.
    pub fn set_scheduler(&mut self, scheduler: BackwardScheduler) {
        self.scheduler = scheduler;
        self.by_node_scratch = match scheduler {
            BackwardScheduler::ByNode { .. } => ByNodeScratch::sized(
                self.max_local_fwd,
                self.bwd_max_openings,
                self.n_state,
                self.num_stages,
            ),
            BackwardScheduler::ByScenario {} => ByNodeScratch::default(),
        };
    }

    /// Overrides whether the by-node scheduler claims work
    /// hardest-`(stage, block)`-first using the previous iteration's
    /// mean pivots (default `true`). `false` forces the canonical ascending
    /// block order — the byte-neutrality gate's off leg. Call before `run()`.
    pub fn set_hardest_first_claim_order(&mut self, enabled: bool) {
        self.hardest_first_claim_order = enabled;
    }

    /// Execute the backward pass for one training iteration on this rank.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
    /// feasible solution during the backward sweep. Returns
    /// `Err(SddpError::Solver(_))` for all other terminal LP solver failures.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any of the following debug preconditions are violated:
    ///
    /// - `inputs.ctx.templates.len() != num_stages`
    /// - `inputs.ctx.base_rows.len() != num_stages`
    /// - `inputs.risk_measures.len() != num_stages`
    /// - `inputs.frozen.len() != n_pools`
    pub fn run<S, C: Communicator>(
        &mut self,
        inputs: &mut BackwardPassInputs<'_, S, C>,
    ) -> Result<BackwardResult, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_ctx = inputs.training_ctx;
        let num_stages = training_ctx.horizon.num_stages();

        debug_assert_eq!(inputs.ctx.templates.len(), num_stages);
        debug_assert_eq!(inputs.ctx.base_rows.len(), num_stages);
        debug_assert_eq!(inputs.risk_measures.len(), num_stages);
        debug_assert_eq!(
            inputs.frozen.len(),
            training_ctx.node_graph.n_pools,
            "frozen.len() must equal n_pools"
        );

        let start = Instant::now();
        let solves_before: u64 = inputs
            .workspaces
            .iter()
            .map(|ws| ws.solver.statistics().solve_count)
            .sum();

        // `set_profile` is delta-tracked: it issues solver-option calls only for
        // fields that differ from the solver's current state.
        let backward_profile = self.profile;
        for ws in inputs.workspaces.iter_mut() {
            ws.solver.set_profile(&backward_profile);
            debug_assert!(
                ws.solver.current_profile() == &backward_profile,
                "solver profile must equal the profile passed to set_profile"
            );
            ws.worker_timing_buf = WorkerPhaseTimings::default();
        }
        // The opening-block scheduler's per-(stage, block-index) pivot accumulator is iteration-local:
        // swapped here (once per `run()` call), never per stage. the hardest-first order reads the
        // swapped-in `block_pivots_prev` (last iteration's fully-merged means);
        // reading `block_pivots` during the sweep instead is the
        // wrong-but-compiling alternative — it is reset-then-partially-filled,
        // yielding zeros or a half-filled row.
        std::mem::swap(
            &mut self.by_node_scratch.block_pivots,
            &mut self.by_node_scratch.block_pivots_prev,
        );
        self.by_node_scratch.block_pivots.fill((0, 0));

        #[allow(clippy::cast_precision_loss)]
        let params = StageDerivedParams {
            my_rank: inputs.comm.rank(),
            n_workers_local: inputs.workspaces.len(),
            n_ranks: inputs.comm.size(),
            bwd_max_openings: self.bwd_stats_send_buf.len()
                / inputs.workspaces.len().max(1)
                / WORKER_STATS_ENTRY_STRIDE,
            n_workers: inputs.workspaces.len() as f64,
        };

        // Verify all ranks agree on n_workers_local. A mismatch silently
        // corrupts the per-worker stats allgatherv buffer; surface it as a
        // typed error before the exchange.
        let local_workers = u64::try_from(inputs.workspaces.len())
            .map_err(|_| SddpError::Validation("workspaces.len() exceeds u64::MAX".into()))?;
        let send = [local_workers];
        let mut min_recv = [0_u64; 1];
        let mut max_recv = [0_u64; 1];
        inputs
            .comm
            .allreduce(&send, &mut min_recv, ReduceOp::Min)
            .map_err(SddpError::Communication)?;
        inputs
            .comm
            .allreduce(&send, &mut max_recv, ReduceOp::Max)
            .map_err(SddpError::Communication)?;
        if min_recv[0] != max_recv[0] {
            return Err(SddpError::Validation(format!(
                "non-uniform n_workers_local across MPI ranks: \
                 local={local_workers}, min={}, max={}; all ranks must \
                 run with the same --threads value",
                min_recv[0], max_recv[0],
            )));
        }

        let mut cuts_generated: usize = 0;
        let mut stage_stats: Vec<(usize, Vec<StageWorkerOpeningDelta>)> =
            Vec::with_capacity(num_stages.saturating_sub(1));
        let mut state_exchange_ms: u64 = 0;
        let mut cut_batch_build_ms: u64 = 0;
        let mut setup_ms: u64 = 0;
        let mut imbalance_ms: u64 = 0;
        let mut scheduling_ms: u64 = 0;
        let mut cut_sync_ms: u64 = 0;

        // Reverse-topological cut-sharing sweep: descending-stage levels, nodes
        // within a level processed independently (nested per-node risk makes
        // siblings barrier-free). Absent `nodes[]` every level is one node
        // (== stage), reducing to the reversed stage loop byte-for-byte.
        let levels = backward_cut_levels(training_ctx.node_graph);
        for level in &levels {
            let out = run_one_backward_level(self, inputs, level, &params)?;
            cuts_generated += out.cuts_generated;
            state_exchange_ms += out.state_exchange_ms;
            cut_batch_build_ms += out.cut_batch_build_ms;
            setup_ms += out.setup_ms;
            imbalance_ms += out.imbalance_ms;
            scheduling_ms += out.scheduling_ms;
            cut_sync_ms += out.cut_sync_ms;
            stage_stats.extend(out.stage_entries);
        }

        if let Some(sender) = inputs.event_sender {
            for ws in inputs.workspaces.iter() {
                let _ = sender.send(TrainingEvent::WorkerTiming {
                    rank: ws.rank,
                    worker_id: ws.worker_id,
                    iteration: inputs.iteration,
                    phase: WorkerTimingPhase::Backward,
                    timings: ws.worker_timing_buf,
                });
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let solves_after: u64 = inputs
            .workspaces
            .iter()
            .map(|ws| ws.solver.statistics().solve_count)
            .sum();

        Ok(BackwardResult {
            cuts_generated,
            elapsed_ms,
            lp_solves: solves_after - solves_before,
            stage_stats,
            state_exchange_time_ms: state_exchange_ms,
            cut_batch_build_time_ms: cut_batch_build_ms,
            setup_time_ms: setup_ms,
            load_imbalance_ms: imbalance_ms,
            scheduling_overhead_ms: scheduling_ms,
            cut_sync_time_ms: cut_sync_ms,
        })
    }

    /// Synchronise per-slot cut binding metadata across MPI ranks for one pool via
    /// one `allreduce(Sum)` over `metadata_sync_contribution`.
    ///
    /// The reduction is bounded to `populated_count`, not the full pool capacity:
    /// slots in `[populated_count, capacity)` are structurally zero on every rank.
    /// `populated_count` is rank-invariant (cuts are added identically on every
    /// rank), so all ranks reduce over the same length.
    fn sync_stage_metadata<C: Communicator>(
        &mut self,
        pool: usize,
        iteration: u64,
        workspaces: &[SolverWorkspace<impl SolverInterface>],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<(), SddpError> {
        let populated_count = fcf.pools[pool].populated();
        if populated_count == 0 {
            return Ok(());
        }
        self.metadata_sync_buf.clear();
        self.metadata_sync_buf.resize(populated_count, 0u64);
        for ws in workspaces {
            for (slot, &inc) in ws
                .backward_accum
                .metadata_sync_contribution
                .iter()
                .enumerate()
                .take(populated_count)
            {
                self.metadata_sync_buf[slot] += inc;
            }
        }
        self.global_increments_buf.clear();
        self.global_increments_buf.resize(populated_count, 0u64);
        comm.allreduce(
            &self.metadata_sync_buf,
            &mut self.global_increments_buf,
            ReduceOp::Sum,
        )
        .map_err(SddpError::from)?;
        for (slot, &inc) in self.global_increments_buf.iter().enumerate() {
            if inc > 0 {
                fcf.pools[pool].record_binding(slot, inc, iteration);
            }
        }
        Ok(())
    }

    /// Decompose one stage's parallel overhead into setup, load-imbalance, and
    /// scheduling components, returning `(setup_ms, imbalance_ms, scheduling_ms)`.
    ///
    /// Deltas are taken against the before-snapshot in `self.worker_stats_before`.
    fn collect_stage_timing_stats<S: SolverInterface + Send>(
        &mut self,
        parallel_wall_ms: u64,
        n_workers: f64,
        workspaces: &mut [SolverWorkspace<S>],
    ) -> (u64, u64, u64) {
        self.worker_stats_after.clear();
        self.worker_stats_after
            .extend(workspaces.iter().map(|w| w.solver.statistics()));
        self.worker_deltas.clear();
        self.worker_deltas.extend(
            self.worker_stats_before
                .iter()
                .zip(&self.worker_stats_after)
                .map(|(before, after)| SolverStatsDelta::from_snapshots(before, after)),
        );
        let stage_setup_ms: f64 = self
            .worker_deltas
            .iter()
            .map(|d| d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms)
            .sum();
        for (ws, delta) in workspaces.iter_mut().zip(&self.worker_deltas) {
            ws.worker_timing_buf.bwd_setup_ms +=
                delta.load_model_time_ms + delta.set_bounds_time_ms + delta.basis_set_time_ms;
        }
        self.worker_totals.clear();
        self.worker_totals
            .extend(self.worker_deltas.iter().map(|d| {
                d.solve_time_ms + d.load_model_time_ms + d.set_bounds_time_ms + d.basis_set_time_ms
            }));
        let max_worker_ms = self.worker_totals.iter().copied().fold(0.0_f64, f64::max);
        let avg_worker_ms = if self.worker_totals.is_empty() {
            0.0_f64
        } else {
            self.worker_totals.iter().sum::<f64>() / n_workers
        };
        let stage_imbalance_ms = (max_worker_ms - avg_worker_ms).max(0.0);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let stage_scheduling_ms = (parallel_wall_ms as f64 - max_worker_ms).max(0.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (
                stage_setup_ms as u64,
                stage_imbalance_ms as u64,
                stage_scheduling_ms as u64,
            )
        }
    }

    /// Pack local per-worker solver statistics, `allgatherv` across all MPI ranks,
    /// unpack the result, and return per-`(rank, worker_id, opening)` delta entries.
    ///
    /// Returns one `StageWorkerOpeningDelta` per `(rank, worker, opening)` triple
    /// where `delta.lp_solves > 0` or `omega == 0` (the ω=0 sentinel always
    /// appears so that downstream stage-stats consumers can detect "stage visited").
    fn gather_stage_solver_stats<C: Communicator>(
        &mut self,
        n_openings: usize,
        n_ranks: usize,
        n_workers_local: usize,
        bwd_max_openings: usize,
        workspaces: &[SolverWorkspace<impl SolverInterface>],
        comm: &C,
    ) -> Result<Vec<StageWorkerOpeningDelta>, SddpError> {
        self.stage_worker_stats_buf.reset();
        for ws in workspaces {
            debug_assert_eq!(
                ws.backward_accum.per_opening_stats.len(),
                n_openings,
                "per_opening_stats length must equal n_openings on every worker"
            );
            #[allow(clippy::cast_sign_loss)]
            let wid = ws.worker_id as usize;
            for omega in 0..n_openings {
                self.stage_worker_stats_buf.set(
                    wid,
                    omega,
                    ws.backward_accum.per_opening_stats[omega].clone(),
                );
            }
        }
        pack_worker_opening_stats(
            &mut self.bwd_stats_send_buf,
            self.stage_worker_stats_buf.as_slice(),
            n_workers_local,
            bwd_max_openings,
        );
        comm.allgatherv(
            &self.bwd_stats_send_buf,
            &mut self.bwd_stats_recv_buf,
            &self.bwd_stats_counts,
            &self.bwd_stats_displs,
        )
        .map_err(SddpError::Communication)?;
        debug_assert_eq!(
            self.bwd_stats_recv_buf.len(),
            n_ranks * n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE,
            "recv buffer length must equal n_ranks * n_workers_local * bwd_max_openings * STRIDE"
        );
        unpack_worker_opening_stats(
            &self.bwd_stats_recv_buf,
            &mut self.bwd_stats_unpack_buf,
            n_ranks * n_workers_local,
            bwd_max_openings,
        );
        let mut entries: Vec<StageWorkerOpeningDelta> =
            Vec::with_capacity(n_ranks * n_workers_local * n_openings);
        for r in 0..n_ranks {
            let rank_i32 = i32::try_from(r).map_err(|_| {
                SddpError::Validation(format!(
                    "MPI rank count {r} overflows i32 (max {})",
                    i32::MAX
                ))
            })?;
            for w in 0..n_workers_local {
                let wid_i32 = i32::try_from(w).map_err(|_| {
                    SddpError::Validation(format!(
                        "worker count {w} overflows i32 (max {})",
                        i32::MAX
                    ))
                })?;
                for omega in 0..n_openings {
                    let flat = (r * n_workers_local + w) * bwd_max_openings + omega;
                    let delta = &self.bwd_stats_unpack_buf[flat];
                    if delta.lp_solves > 0 || omega == 0 {
                        entries.push((rank_i32, wid_i32, omega, delta.clone()));
                    }
                }
            }
        }
        Ok(entries)
    }
}

/// Test/tooling accessors for `BackwardPassState::by_node_scratch` — never called
/// from production hot-path code.
#[cfg(any(test, feature = "test-support"))]
impl BackwardPassState {
    /// `.capacity()` of the opening-block scratch arena — `0` under `ByScenario`, else the
    /// `max_local_fwd * bwd_max_openings` shape `ByNodeScratch::sized`
    /// allocated at [`Self::set_scheduler`].
    #[must_use]
    pub fn by_node_scratch_arena_capacity(&self) -> usize {
        self.by_node_scratch.arena.capacity()
    }

    /// Read-only view of the opening-block scratch arena, for sizing assertions
    /// (`.len()`, each entry's `coefficients.len()`).
    #[must_use]
    pub fn by_node_scratch_arena(&self) -> &[BackwardOutcome] {
        &self.by_node_scratch.arena
    }

    /// Per-`(stage, block-index)` mean `simplex_iterations` pivot from the
    /// by-node scheduler's rank-local accumulator. Outer index is
    /// the backward pass's successor stage (`t + 1`); inner index is the
    /// block index. `None` where no opening was solved this iteration
    /// (count == 0); empty under `BackwardScheduler::ByScenario`.
    #[must_use]
    pub fn block_pivot_means(&self) -> Vec<Vec<Option<f64>>> {
        let stride = self.by_node_scratch.n_blocks_max;
        if stride == 0 {
            return Vec::new();
        }
        self.by_node_scratch
            .block_pivots
            .chunks(stride)
            .map(|stage_row| {
                stage_row
                    .iter()
                    .map(|&(sum, count)| {
                        #[allow(clippy::cast_precision_loss)]
                        (count > 0).then(|| sum as f64 / count as f64)
                    })
                    .collect()
            })
            .collect()
    }
}

/// Iteration-constant values derived once from `BackwardPassInputs` at the start of `run`.
///
/// Passed to `compute_one_backward_node` to avoid recomputing them on every node
/// and to keep the argument count of that helper within budget.
struct StageDerivedParams {
    /// This rank's MPI rank index.
    my_rank: usize,
    /// Number of rayon workers on this rank.
    n_workers_local: usize,
    /// Total MPI rank count.
    n_ranks: usize,
    /// Maximum opening count across all stages (stride for stats buffers).
    bwd_max_openings: usize,
    /// `n_workers_local as f64`, pre-cast for load-imbalance arithmetic.
    n_workers: f64,
}

/// Aggregate output of one cut-sharing level, accumulated in [`BackwardPassState::run`].
struct LevelOutput {
    /// Global cuts added across every node in the level (rank-count invariant).
    cuts_generated: usize,
    /// Per-node `(successor stage, per-(rank, worker, opening) deltas)`.
    stage_entries: Vec<(usize, Vec<StageWorkerOpeningDelta>)>,
    /// Time in the level's one `exchange.exchange()`, in ms.
    state_exchange_ms: u64,
    /// `build_delta_cut_row_batch_into` time summed over the level's nodes, in ms.
    cut_batch_build_ms: u64,
    /// Aggregate non-solve setup time summed over the level's nodes, in ms.
    setup_ms: u64,
    /// Load-imbalance component summed over the level's nodes, in ms.
    imbalance_ms: u64,
    /// Scheduling overhead summed over the level's nodes, in ms.
    scheduling_ms: u64,
    /// Time in the level's one batched cut-sync `allgatherv`, in ms.
    cut_sync_ms: u64,
}

/// Per-node result of [`compute_one_backward_node`], consumed by the level's
/// batched cut exchange and per-node metadata/timing/stats epilogue.
struct NodeCompute {
    /// Successor stage index (`node.stage + 1`), the `stage_stats` key.
    successor_stage: usize,
    /// The node's own cut pool (where its cut was inserted).
    pool_id: usize,
    /// The successor's cut pool (binding-metadata sync target).
    successor_pool_id: usize,
    /// Opening count of the node's successor outcome set (stats stride).
    n_openings: usize,
    /// `build_delta_cut_row_batch_into` time for this node, in ms.
    cut_batch_build_ms: u64,
    /// Aggregate non-solve setup time for this node, in ms.
    setup_ms: u64,
    /// Load-imbalance component for this node, in ms.
    imbalance_ms: u64,
    /// Scheduling overhead for this node, in ms.
    scheduling_ms: u64,
}

/// A node's trial-state axis is a singleton when its backward processes exactly
/// one trial point yet more than one scenario reached it — the dedup collapse an
/// enumerated forward produces. A lone sampled trajectory (`n_scenarios == 1`)
/// is NOT a singleton group: nothing collapsed.
///
/// Reserved: the current forward emits one trial point per scenario
/// (`n_trial_states == n_scenarios`), so this is never satisfiable and the
/// by-node/replicated path it would gate is inert — see
/// [`run_backward_node_replicated`].
#[allow(dead_code)]
fn trial_state_axis_is_singleton(n_trial_states: usize, n_scenarios: usize) -> bool {
    n_trial_states == 1 && n_scenarios > 1
}

/// Resolve the effective backward thread scheduler. A singleton-trial-state
/// group auto-selects the by-node scheduler regardless of config
/// — the by-scenario scheduler would serialize it to one work unit — but an
/// active Dynamic Cut Selection iteration always forces the by-scenario path
/// (its cut-free lazy core is incompatible with the by-node frozen-LP
/// load). A non-singleton group keeps its configured scheduler unchanged.
///
/// Reserved: the singleton-group selector is inert until the forward emits a
/// deduped trial-state count — see [`run_backward_node_replicated`].
#[allow(dead_code)]
fn resolve_backward_scheduler(
    singleton: bool,
    dcs_active: bool,
    configured: BackwardScheduler,
) -> BackwardScheduler {
    if dcs_active {
        return BackwardScheduler::ByScenario {};
    }
    if singleton {
        return match configured {
            BackwardScheduler::ByNode { .. } => configured,
            BackwardScheduler::ByScenario {} => BackwardScheduler::ByNode { block_size: None },
        };
    }
    configured
}

/// Flatten node `node_pos`'s successor outcome set
/// `O(n) = {(m, ψ): m ∈ n⁺, ψ ∈ Ω_m}` into `out`, canonical order (ascending
/// child node id — `node_graph.successors[node_pos]`'s own invariant — then
/// within-child ω): `CVaR`'s tail weighting is index-order-sensitive (sddp.md
/// "Backward opening order is warm-start-only"). Delegates to the shared
/// [`crate::setup::node_graph::assemble_outcome_weights`] primitive — the
/// single owner of this fill, shared with
/// `lower_bound::assemble_outcome_weights`.
fn assemble_successor_outcome_weights(node_graph: &NodeGraph, node_pos: usize, out: &mut Vec<f64>) {
    crate::setup::node_graph::assemble_outcome_weights(
        node_graph,
        &node_graph.successors[node_pos],
        out,
    );
}

/// Group this rank's own local trial points (`0..local_work`) by the
/// cut-generating node each visited at `level_stage`, into CSR form:
/// `routed[offsets[i]..offsets[i + 1]]` are the ascending trial-point indices
/// whose visit resolves to `level[i]`'s pool. A cut-generating node always owns
/// its own pool, so grouping by visited node *is* per-pool routing — each cut
/// anchors at its own node's sampled incoming states (Σ over a level's pools of
/// their trial counts = `local_work`).
///
/// A single-node level carries exactly one node at its stage (a mid-horizon
/// leaf beside it is rejected at setup), so every trial point that reached the
/// stage visited that node: route all with no per-record lookup. This is the
/// chain and terminal-fan case, and reproduces the pre-routing `0..local_work`
/// slice byte-for-byte. Only a multi-node level reads `TrajectoryRecord::node_id`
/// to split siblings that own distinct pools.
fn build_trial_routing(
    node_graph: &NodeGraph,
    level: &[usize],
    level_stage: usize,
    records: &[TrajectoryRecord],
    num_stages: usize,
    local_work: usize,
    routed: &mut Vec<usize>,
    offsets: &mut Vec<usize>,
) {
    routed.clear();
    offsets.clear();
    offsets.push(0);
    if level.len() == 1 {
        routed.extend(0..local_work);
        offsets.push(local_work);
        return;
    }
    for &node_pos in level {
        let node_id = node_graph.node_ids[node_pos];
        for m in 0..local_work {
            if records[m * num_stages + level_stage].node_id == node_id {
                routed.push(m);
            }
        }
        offsets.push(routed.len());
    }
    debug_assert_eq!(
        routed.len(),
        local_work,
        "every local trial point must route to exactly one cut-generating node in the level \
         (a mid-horizon leaf would be rejected at setup); routed {} of {local_work}",
        routed.len()
    );
}

/// Execute one reverse-topological cut-sharing level: one state exchange for the
/// level, each node's backward cut computed independently over ONLY the trial
/// points routed to its pool, ONE batched cut exchange over the level's
/// trial-point-distributed pools, then per-node metadata/timing/stats. Absent
/// `nodes[]` a level is one node (== stage) and every trial routes to it, so
/// this is byte-for-byte the reversed stage loop (one state exchange, one cut
/// exchange, one metadata/stats gather).
fn run_one_backward_level<S: SolverInterface + Send, C: Communicator>(
    state: &mut BackwardPassState,
    inputs: &mut BackwardPassInputs<'_, S, C>,
    level: &[usize],
    params: &StageDerivedParams,
) -> Result<LevelOutput, SddpError> {
    let training_ctx = inputs.training_ctx;
    let num_stages = training_ctx.horizon.num_stages();
    // Every node in a level shares one stage; the stage-keyed state records
    // cover them all in ONE `allgatherv` — one exchange per level, never per node
    // (sddp.md "Per-level exchange in the backward pass").
    let level_stage = training_ctx.node_graph.nodes[level[0]].stage;

    let exch_start = Instant::now();
    inputs
        .exchange
        .exchange(inputs.records, level_stage, num_stages, inputs.comm)?;
    #[allow(clippy::cast_possible_truncation)]
    let state_exchange_ms = exch_start.elapsed().as_millis() as u64;

    if let Some(ref mut archive) = inputs.visited_archive {
        let total_fwd = inputs.exchange.real_total_scenarios();
        inputs
            .exchange
            .pack_real_states_into(&mut state.real_states_buf);
        // Archive by NODE position, not `level_stage` used as a node index:
        // sibling nodes at one stage own distinct pools and cut regions, so each
        // reads back its own archive (`states_for_node`). The exchange gathers the
        // level's states without per-node split, so every node in a multi-node
        // level receives the level's states — a conservative over-inclusion that
        // never drops a binding cut. On a chain the level is one node, giving the
        // former single stage-keyed bucket byte-for-byte.
        for &node_pos in level {
            archive.archive_gathered_states(node_pos, &state.real_states_buf, total_fwd);
        }
    }

    // Route this rank's trial points to the level's pools once (a mem-swapped
    // local so the borrow does not conflict with `&mut state` below); a
    // cut-generating node then anchors its cut ONLY at the states its own pool
    // was visited with.
    let mut routed_trials = std::mem::take(&mut state.routed_trials_scratch);
    let mut routed_offsets = std::mem::take(&mut state.routed_offsets_scratch);
    build_trial_routing(
        training_ctx.node_graph,
        level,
        level_stage,
        inputs.records,
        num_stages,
        inputs.local_work,
        &mut routed_trials,
        &mut routed_offsets,
    );

    // Reuse the per-level buffers across levels (mem-swapped out so the loop
    // allocates nothing per level; the empty placeholder left on `state` frees
    // the mutable-borrow conflict with `compute_one_backward_node`).
    let mut nodes_out = std::mem::take(&mut state.level_nodes_scratch);
    nodes_out.clear();
    let mut cut_batch_build_ms = 0u64;
    for (level_idx, &node_pos) in level.iter().enumerate() {
        let trial_points = &routed_trials[routed_offsets[level_idx]..routed_offsets[level_idx + 1]];
        let nc = compute_one_backward_node(state, inputs, node_pos, trial_points, params)?;
        cut_batch_build_ms += nc.cut_batch_build_ms;
        nodes_out.push(nc);
    }

    // ONE batched cut exchange over the level's pools (not one collective per
    // node): all the level's nodes' cuts go out in a single `allgatherv`.
    let sync_start = Instant::now();
    let mut level_pools = std::mem::take(&mut state.level_pools_scratch);
    level_pools.clear();
    level_pools.extend(nodes_out.iter().map(|nc| nc.pool_id));
    let (n_local_total, remote_total) = inputs.cut_sync_bufs.sync_level_records(
        &level_pools,
        inputs.fcf,
        inputs.iteration,
        inputs.comm,
    )?;
    // Global cuts added across the level: every rank's local share plus every
    // peer's; the local count scales with rank count while the global total is
    // rank-count invariant.
    let cuts_generated = n_local_total + remote_total;
    #[allow(clippy::cast_possible_truncation)]
    let cut_sync_ms = sync_start.elapsed().as_millis() as u64;

    let mut stage_entries = Vec::with_capacity(nodes_out.len());
    let mut setup_ms = 0u64;
    let mut imbalance_ms = 0u64;
    let mut scheduling_ms = 0u64;
    for nc in &nodes_out {
        state.sync_stage_metadata(
            nc.successor_pool_id,
            inputs.iteration,
            inputs.workspaces,
            inputs.fcf,
            inputs.comm,
        )?;
        setup_ms += nc.setup_ms;
        imbalance_ms += nc.imbalance_ms;
        scheduling_ms += nc.scheduling_ms;
        let entries = state.gather_stage_solver_stats(
            nc.n_openings,
            params.n_ranks,
            params.n_workers_local,
            params.bwd_max_openings,
            inputs.workspaces,
            inputs.comm,
        )?;
        stage_entries.push((nc.successor_stage, entries));
    }

    state.level_nodes_scratch = nodes_out;
    state.level_pools_scratch = level_pools;
    state.routed_trials_scratch = routed_trials;
    state.routed_offsets_scratch = routed_offsets;

    Ok(LevelOutput {
        cuts_generated,
        stage_entries,
        state_exchange_ms,
        cut_batch_build_ms,
        setup_ms,
        imbalance_ms,
        scheduling_ms,
        cut_sync_ms,
    })
}

/// Compute one node's backward cut over `trial_points` — the trial-point
/// indices routed to this node's pool — building the successor cut batch,
/// solving (the replicated aggregation at a singleton-trial-state group, else
/// the by-node or trial-point path), inserting the local cut(s), reconciling,
/// and computing this node's timing. The level driver owns the state exchange,
/// the routing, the batched cut exchange, and the metadata/stats gather.
// RATIONALE: sequences the per-node solve phases whose intermediate state each
// next phase reads; splitting further would thread every disjoint sub-field as
// &mut without gaining clarity.
#[allow(clippy::too_many_lines)]
fn compute_one_backward_node<S: SolverInterface + Send, C: Communicator>(
    state: &mut BackwardPassState,
    inputs: &mut BackwardPassInputs<'_, S, C>,
    node_pos: usize,
    trial_points: &[usize],
    params: &StageDerivedParams,
) -> Result<NodeCompute, SddpError> {
    let training_ctx = inputs.training_ctx;
    let ctx = inputs.ctx;
    let cut_state = training_ctx.state;
    let node_stage = training_ctx.node_graph.nodes[node_pos].stage;
    let successor_stage = node_stage + 1;
    // The node's own pool, and its first canonical successor's node position and
    // pool — the basis node-axis key and the binding-metadata target. Absent
    // `nodes[]` node position == stage, so these equal `node_stage` / `+1`.
    let successor_node = training_ctx.node_graph.successors[node_pos][0].child;
    let pool_id = training_ctx.node_graph.nodes[node_pos].pool_id;
    let successor_pool_id = training_ctx.node_graph.nodes[successor_node].pool_id;
    let cut_state_projection = &training_ctx.cut_state_layouts[pool_id];

    state.worker_stats_before.clear();
    state
        .worker_stats_before
        .extend(inputs.workspaces.iter().map(|w| w.solver.statistics()));

    assemble_successor_outcome_weights(
        training_ctx.node_graph,
        node_pos,
        &mut state.probabilities_buf,
    );
    let n_openings = state.probabilities_buf.len();

    let batch_start = Instant::now();
    let template_num_rows = ctx.templates[successor_stage].num_rows;
    // Rendering the successor's cuts uses pool `successor_pool_id`'s projection —
    // NOT `cut_state_projection` (pool `pool_id`'s, used only for extraction).
    let successor_cut_layout = &training_ctx.cut_state_layouts[successor_pool_id];
    build_delta_cut_row_batch_into(
        &mut inputs.cut_batches[successor_pool_id],
        inputs.fcf,
        successor_pool_id,
        cut_state,
        successor_cut_layout,
        &ctx.templates[successor_stage].col_scale,
        inputs.iteration,
    );
    // The successor LP is loaded per POOL: its frozen rows and the delta cut
    // batch both belong to pool `successor_pool_id`. `num_cuts_at_successor`
    // MUST count the frozen pool's own cut rows (`frozen[successor_pool_id]`)
    // plus that same pool's delta — mixing a frozen pool's row count with a
    // different pool's delta count corrupts warm-start slot reconstruction.
    let frozen_tmpl = &inputs.frozen[successor_pool_id];
    let num_cuts_at_successor =
        (frozen_tmpl.num_rows - template_num_rows) + inputs.cut_batches[successor_pool_id].num_rows;
    #[allow(clippy::cast_possible_truncation)]
    let cut_batch_build_ms = batch_start.elapsed().as_millis() as u64;

    state.successor_active_slots_buf.clear();
    state.successor_active_slots_buf.extend(
        inputs
            .fcf
            .active_cuts(successor_pool_id)
            .map(|(slot, _, _)| slot),
    );

    let succ_spec = SuccessorSpec {
        t: node_stage,
        successor: successor_stage,
        successor_node,
        successor_node_id: training_ctx.node_graph.node_ids[successor_node],
        my_rank: params.my_rank,
        probabilities: &state.probabilities_buf,
        cut_batch: &inputs.cut_batches[successor_pool_id],
        num_cuts_at_successor,
        template_num_rows,
        frozen_template: frozen_tmpl,
        successor_active_slots: &state.successor_active_slots_buf,
        cut_activity_tolerance: inputs.cut_activity_tolerance,
        successor_populated_count: inputs.fcf.pools[successor_pool_id].populated(),
        successor_pool: &inputs.fcf.pools[successor_pool_id],
        cut_state: cut_state_projection,
    };

    // The by-node scheduler needs the frozen-LP path (`load_backward_lp`),
    // incompatible with DCS's cut-free lazy core, so an active DCS iteration always
    // falls back to the by-scenario path (sddp.md "By-node scheduler is
    // warm-start-only"). A singleton-trial-state group would auto-select the
    // by-node path via `resolve_backward_scheduler` and take the replicated
    // aggregation, but no such node is trainable yet — see
    // `trial_state_axis_is_singleton`.
    let use_by_node = matches!(state.scheduler, BackwardScheduler::ByNode { .. })
        && training_ctx
            .dcs
            .filter(|p| p.is_active(inputs.iteration))
            .is_none();

    let process_start = Instant::now();
    let (local_solve, parallel_wall_ms): (Result<usize, SddpError>, u64) = if use_by_node {
        let configured_block_size = match state.scheduler {
            BackwardScheduler::ByNode { block_size } => block_size,
            BackwardScheduler::ByScenario {} => None,
        };
        let block_size = resolve_block_size(n_openings, configured_block_size);
        let n_blocks = by_node_block_count(n_openings, block_size);
        if state.hardest_first_claim_order {
            let row_start = successor_stage * state.by_node_scratch.n_blocks_max;
            hardest_first_block_order(
                &state.by_node_scratch.block_pivots_prev[row_start..row_start + n_blocks],
                n_blocks,
                &mut state.by_node_scratch.block_order,
            );
        } else {
            identity_block_order(n_blocks, &mut state.by_node_scratch.block_order);
        }
        let worker_out = process_stage_backward_by_node(
            inputs.workspaces,
            ctx,
            training_ctx,
            trial_points,
            inputs.exchange,
            inputs.fwd_offset,
            inputs.iteration,
            &succ_spec,
            &*inputs.basis_store,
            block_size,
            &state.by_node_scratch.block_order[..n_blocks],
        );
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = process_start.elapsed().as_millis() as u64;
        // Telemetry-only merge (sddp.md "by-node scheduler is
        // warm-start-only" — disjoint from the per-`(m, ω)` arena scatter and
        // ascending-m aggregation `by_node_finish` performs below).
        merge_block_pivots(
            inputs.workspaces.iter().map(|ws| {
                (
                    ws.backward_accum.block_pivot_sum.as_slice(),
                    ws.backward_accum.block_pivot_count.as_slice(),
                )
            }),
            n_blocks,
            successor_stage,
            &mut state.by_node_scratch,
        );
        let result = by_node_finish(
            worker_out,
            &*inputs.workspaces,
            trial_points,
            n_openings,
            cut_state_projection.n_slots(),
            &state.probabilities_buf,
            &inputs.risk_measures[node_stage],
            inputs.fcf,
            pool_id,
            inputs.iteration,
            inputs.fwd_offset,
            &mut state.by_node_scratch,
        );
        (result, elapsed_ms)
    } else {
        let basis_slices = inputs
            .basis_store
            .split_workers_mut(params.n_workers_local.max(1));
        let worker_staged = process_stage_backward(
            inputs.workspaces,
            ctx,
            training_ctx,
            trial_points,
            inputs.exchange,
            inputs.fwd_offset,
            inputs.iteration,
            inputs.risk_measures,
            &succ_spec,
            basis_slices,
        );
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = process_start.elapsed().as_millis() as u64;

        state.staged_cuts_buf.clear();
        let mut worker_failure: Option<SddpError> = None;
        for worker_result in worker_staged {
            match worker_result {
                Ok((w, cuts)) => state
                    .staged_cuts_buf
                    .extend(cuts.into_iter().map(|cut| (w, cut))),
                Err(e) => {
                    worker_failure = Some(e);
                    break;
                }
            }
        }

        let result = if let Some(e) = worker_failure {
            Err(e)
        } else {
            // `trial_state_idx` is the SOLE sort key: globally unique across workers
            // (disjoint contiguous partitions), so the merge order is identical regardless
            // of worker index.
            state
                .staged_cuts_buf
                .sort_by_key(|(_, cut)| cut.trial_state_idx);
            debug_assert_eq!(state.staged_cuts_buf.len(), trial_points.len());
            for (w, cut) in &state.staged_cuts_buf {
                let range = cut.coefficients_range.clone();
                let arena = &inputs.workspaces[*w].backward_accum.agg_arena;
                debug_assert!(
                    range.len() == cut_state_projection.n_slots() && range.end <= arena.len(),
                    "coefficients_range must span exactly the pool's cut n_state and lie within the worker arena"
                );
                inputs.fcf.add_cut(
                    pool_id,
                    inputs.iteration,
                    cut.forward_pass_index,
                    cut.intercept,
                    &arena[range],
                );
            }
            Ok(state.staged_cuts_buf.len())
        };
        (result, elapsed_ms)
    };

    // Reconcile the divergent backward solve outcome BEFORE the level's sync
    // collectives (mirrors the forward-phase precedent): ranks solve disjoint
    // trial points, so a failure on a strict subset makes every rank return Err
    // here rather than let a healthy rank block in the batched cut sync /
    // metadata sync / stats gather while a peer skipped them.
    reconcile_result(local_solve, inputs.comm, &mut state.reconcile_scratch)?;

    let (setup_ms, imbalance_ms, scheduling_ms) =
        state.collect_stage_timing_stats(parallel_wall_ms, params.n_workers, inputs.workspaces);

    Ok(NodeCompute {
        successor_stage,
        pool_id,
        successor_pool_id,
        n_openings,
        cut_batch_build_ms,
        setup_ms,
        imbalance_ms,
        scheduling_ms,
    })
}

/// Scalars for [`run_backward_node_replicated`].
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct ReplicatedAggregate {
    /// Cut-state dimension (subgradient length per outcome).
    n_state: usize,
    /// Total outcomes in the node's successor outcome set.
    n_openings: usize,
    /// Total MPI rank count (the opening-split partition denominator).
    n_ranks: usize,
    /// This rank's MPI rank index (the singleton trial point's exchange slot).
    my_rank: usize,
    /// The node's own cut pool.
    pool_id: usize,
    /// Global offset of this rank's trial points.
    fwd_offset: usize,
    /// Current training iteration.
    iteration: u64,
}

/// Reserved: opening-split replicated aggregation for a singleton-trial-state
/// node — a declared dedup/recombining node whose scenarios collapse onto one
/// trial state. Ranks split the successor outcome set, allgather per-outcome
/// `(objective, subgradient)` in canonical `(m, ψ)` order, and every rank runs
/// the identical flat `aggregate_cut_into` and appends the identical cut, so the
/// batched cut exchange is skipped for the pool (a naive exchange would multiply
/// the cut by the rank count).
///
/// Inert until the forward pass emits a DEDUPED trial-state count distinct from
/// the scenario count [`ExchangeBuffers::real_total_scenarios`]: the sampled
/// walk resolves a node per trajectory, but no per-node trial-state count
/// exists yet, so [`trial_state_axis_is_singleton`] is never satisfiable from
/// the scenario count alone. When that count lands, wire it into
/// `compute_one_backward_node`'s scheduler resolution ([`resolve_backward_scheduler`])
/// so this path runs and the pool is excluded from the level's batched exchange.
/// The allgather + aggregation math is pinned by the backward-pass unit tests.
///
/// # Errors
///
/// Propagates the slice solve's `SddpError` and the outcome `allgatherv`'s
/// `SddpError::Validation`/`SddpError::Communication`.
#[allow(dead_code, clippy::too_many_arguments)]
fn run_backward_node_replicated<S: SolverInterface + Send, C: Communicator>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    exchange: &ExchangeBuffers,
    comm: &C,
    fcf: &mut FutureCostFunction,
    risk_measure: &RiskMeasure,
    probabilities: &[f64],
    succ: &SuccessorSpec<'_>,
    agg: ReplicatedAggregate,
) -> Result<usize, SddpError> {
    let ReplicatedAggregate {
        n_state,
        n_openings,
        n_ranks,
        my_rank,
        pool_id,
        fwd_offset,
        iteration,
    } = agg;

    let (o_start, o_end) = partition(n_openings, n_ranks, my_rank);
    let mut outcome_counts = vec![0usize; n_ranks];
    for (r, c) in outcome_counts.iter_mut().enumerate() {
        let (s0, e0) = partition(n_openings, n_ranks, r);
        *c = e0 - s0;
    }

    let mut slice = Vec::new();
    solve_replicated_outcome_slice(
        ws,
        ctx,
        training_ctx,
        exchange,
        succ,
        0,
        o_start,
        o_end,
        fwd_offset,
        iteration,
        &mut slice,
    )?;

    let mut ex = OutcomeExchangeScratch::default();
    let full = ex.allgather_outcomes(&slice, n_state, &outcome_counts, comm)?;

    let x_hat = exchange.state_at(my_rank, 0);
    let stride = outcome_stride(n_state);
    let mut outcomes: Vec<BackwardOutcome> = Vec::with_capacity(n_openings);
    for o in 0..n_openings {
        let base = o * stride;
        let coefficients = full[base + 1..base + 1 + n_state].to_vec();
        let objective = full[base];
        let intercept = objective
            - coefficients
                .iter()
                .zip(x_hat)
                .map(|(c, x)| c * x)
                .sum::<f64>();
        outcomes.push(BackwardOutcome {
            intercept,
            coefficients,
            objective_value: objective,
        });
    }

    let mut intercept = 0.0_f64;
    let mut coeffs = vec![0.0_f64; n_state];
    let mut risk_scratch = crate::risk_measure::RiskMeasureScratch::default();
    risk_measure.aggregate_cut_into(
        &outcomes,
        probabilities,
        &mut intercept,
        &mut coeffs,
        &mut risk_scratch,
    );
    fcf.add_cut(pool_id, iteration, 0, intercept, &coeffs);
    Ok(1)
}

/// Evaluate this node's routed `trial_points` for a single backward stage,
/// returning staged cuts (one per trial point).
// RATIONALE: 10 args are individually-borrowed slices passed through the rayon closure
// boundary. Bundling them into a struct would require either cloning or an `Arc`, both of
// which conflict with the zero-allocation HPC constraint for backward-pass hot code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_stage_backward<S: SolverInterface + Send>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    trial_points: &[usize],
    exchange: &ExchangeBuffers,
    fwd_offset: usize,
    iteration: u64,
    risk_measures: &[RiskMeasure],
    succ: &SuccessorSpec<'_>,
    basis_slices: Vec<BasisStoreSliceMut<'_>>,
) -> Vec<Result<(usize, Vec<StagedCut>), SddpError>> {
    let n_openings = succ.probabilities.len();
    // Per-stage cut dimension (pool `t`'s `CutStateProjection`). Buffers reused
    // across stages are resized to EXACTLY this each stage, never grown to a
    // per-worker max: a reduced stage after a full one must shrink, or
    // `write_opening_outcome`'s `copy_from_slice` reads stale full-length data.
    let cut_n_state = succ.cut_state.n_slots();
    let pop = succ.successor_populated_count;

    // Opening-solve strategy, chosen once per stage. The decision depends only on
    // `iteration`, so it is constant across workers and trial points.
    let opening_solver = StageOpeningSolver::from_dcs_params(
        training_ctx
            .dcs
            .filter(|params| params.is_active(iteration)),
    );

    workspaces
        .par_iter_mut()
        .zip(basis_slices.into_par_iter())
        .enumerate()
        .map(|(w, (ws, mut basis_slice))| {
            // Pre-allocate per-stage buffers. This touches only `ws.backward_accum`,
            // never `ws.solver`: the LP load is issued per trial point via
            // `opening_solver.prepare`, not here.
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
            ws.backward_accum
                .agg_coefficients
                .resize(cut_n_state, 0.0_f64);
            if ws.backward_accum.metadata_sync_contribution.len() < pop {
                ws.backward_accum
                    .metadata_sync_contribution
                    .resize(pop, 0u64);
            }
            ws.backward_accum.metadata_sync_contribution[..pop].fill(0);
            ws.backward_accum
                .per_opening_stats
                .resize_with(n_openings, SolverStatsDelta::default);
            for slot in &mut ws.backward_accum.per_opening_stats[..n_openings] {
                *slot = SolverStatsDelta::default();
            }

            // Worker `w` processes exactly the routed trial points whose GLOBAL
            // scenario index falls in its own basis-slice window, so
            // `basis_slice.get(m, node)` is in-bounds by construction. A node's
            // routed subset is scattered across the global scenario axis, so an
            // even split of `trial_points.len()` would misalign with the
            // contiguous per-worker basis window and index out of the slice. On a
            // single-node level the window tiling reproduces the pre-routing
            // `partition(local_work, n_workers, w)` assignment exactly.
            let (w_start, w_end) = basis_slice.scenario_window();
            let owns = move |m: usize| w_start <= m && m < w_end;
            let count_w = trial_points.iter().filter(|&&m| owns(m)).count();
            // Grow-only arena (one `cut_n_state` slot per owned trial point);
            // content is overwritten per trial point before read, so no zero-fill.
            // Stride is `cut_n_state` — must match the `coefficients_range` length
            // `process_by_scenario_backward` writes.
            let arena_len = count_w * cut_n_state;
            if ws.backward_accum.agg_arena.len() < arena_len {
                ws.backward_accum.agg_arena.resize(arena_len, 0.0_f64);
            }
            ws.backward_accum.staged_cuts_buf.clear();
            let worker_stage_wall_start = Instant::now();
            // Snapshot the cumulative lazy-scoring accumulator; the delta below
            // attributes this stage's scoring to the backward phase (the accumulator
            // is never reset, so a snapshot-delta is the only correct attribution).
            let scoring_seconds_before = ws.backward_accum.dcs_solve.scoring_time_seconds;

            // `compacted` is the trial point's index within the node's FULL routed
            // subset (the cut slot's per-pool position); `local_i` is its row in
            // this worker's own arena. They diverge only when a peer worker owns
            // earlier routed points — `local_i` skips those, `compacted` counts them.
            let mut local_i = 0usize;
            for (compacted, &m) in trial_points.iter().enumerate() {
                if !owns(m) {
                    continue;
                }
                // Reset so the cold-head solve cannot depend on which trial points
                // the worker solved before it (determinism across thread/rank counts).
                // No-op for HiGHS; for CLP recreates the model (`Clp_loadProblem`
                // leaves rim/pricing/steepest-edge state stale). MUST fire ONCE PER
                // TRIAL POINT, not per opening — the within-trial-point warm-start
                // chain across openings lives in `process_by_scenario_backward`.
                ws.solver.reset_solver_state();

                opening_solver.prepare(ws, ctx, succ, iteration);
                ws.backward_accum.slot_increments[..pop].fill(0);
                // Call before the push to avoid a simultaneous mutable borrow of
                // `staged_cuts_buf` (push receiver) and `ws` (function argument).
                let arena_offset = local_i * cut_n_state;
                let cut = process_by_scenario_backward(
                    ws,
                    ctx,
                    training_ctx,
                    exchange,
                    fwd_offset,
                    iteration,
                    risk_measures,
                    succ,
                    &mut basis_slice,
                    &opening_solver,
                    m,
                    compacted,
                    arena_offset,
                )?;
                ws.backward_accum.staged_cuts_buf.push(cut);
                local_i += 1;
            }

            ws.worker_timing_buf.backward_wall_ms +=
                worker_stage_wall_start.elapsed().as_secs_f64() * 1_000.0;
            // The forward and backward folds write the same physical `scoring_ms`
            // field on disjoint phase emissions, so the per-iteration total is
            // recovered by summing across both phases' events.
            ws.worker_timing_buf.scoring_ms += (ws.backward_accum.dcs_solve.scoring_time_seconds
                - scoring_seconds_before)
                * 1_000.0;

            // `drain(..)` leaves capacity intact for the next stage; `w` rides
            // alongside so the merge resolves each cut's `coefficients_range`
            // against that worker's arena.
            Ok((w, ws.backward_accum.staged_cuts_buf.drain(..).collect()))
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::scenario::{InflowModel, SamplingScheme};
    use cobre_solver::{
        Basis, LpSolution, ProfiledSolver, RowBatch, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };

    use crate::{
        context::{StageContext, TrainingContext},
        cut::FutureCostFunction,
        cut_sync::CutSyncBuffers,
        horizon_mode::HorizonMode,
        indexer::StateSpace,
        inflow_method::InflowNonNegativityMethod,
        risk_measure::{BackwardOutcome, RiskMeasure},
        setup::node_graph::{
            NodeOpenings, NodeRuntime, NodeSuccessor, OpeningSource, max_successor_outcome_count,
        },
        solver_stats::WORKER_STATS_ENTRY_STRIDE,
        state_exchange::ExchangeBuffers,
        test_support::{
            all_enabled_cut_state_layouts, state_layout, study_dims, trial_state_records,
        },
        trajectory::TrajectoryRecord,
        workspace::{
            BackwardAccumulators, BasisStore, CapturedBasis, ScratchBuffers, SolverWorkspace,
        },
    };

    // ── test stubs ──────────────────────────────────────────────────────────

    struct StubComm;

    impl Communicator for StubComm {
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
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv[..send.len()].copy_from_slice(send);
            Ok(())
        }
        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            Ok(())
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
        fn abort(&self, code: i32) -> ! {
            std::process::exit(code)
        }
    }

    struct MockSolver {
        solution: LpSolution,
        call_count: usize,
        current_num_rows: usize,
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
                call_count: 0,
                current_num_rows: base_rows,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn name(&self) -> &'static str {
            "mock"
        }
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.current_num_rows = template.num_rows;
            self.buf_primal = self.solution.primal.clone();
            self.buf_dual = self.solution.dual.clone();
            self.buf_reduced_costs = self.solution.reduced_costs.clone();
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn add_rows(&mut self, cuts: &RowBatch) {
            self.current_num_rows += cuts.num_rows;
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            self.call_count += 1;
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            *out = Basis::new(0, 0);
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
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
            col_upper: vec![f64::INFINITY; 3],
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
        LpSolution {
            objective,
            primal: vec![0.0, 0.0, 0.0],
            dual: vec![dual_storage],
            reduced_costs: vec![0.0; 3],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn single_workspace(solver: MockSolver, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
        use crate::lp_builder::PatchBuffer;
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver: ProfiledSolver::new(solver),
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                ncs_col_lower_active_buf: Vec::new(),
                ncs_col_upper_active_buf: Vec::new(),
                last_ncs_col_start: usize::MAX,
                ncs_col_upper_extract_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: Vec::new(),
                lag_weight_accum: Vec::new(),
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                recon_slot_lookup: Vec::new(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                current_node_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: WorkerPhaseTimings::default(),
        }]
    }

    fn empty_basis_store(num_scenarios: usize, num_stages: usize) -> BasisStore {
        BasisStore::new(num_scenarios, num_stages)
    }

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

    fn make_stochastic_context(
        n_stages: usize,
        branching_factor: usize,
    ) -> cobre_stochastic::StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::{
            Bus, DeficitSegment, EntityId, SystemBuilder,
            scenario::{CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile},
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
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
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
        hydro.declare_mirror_unit_group(EntityId(0));

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
        let inflow_models: Vec<_> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

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

    // ── tests ───────────────────────────────────────────────────────────────

    #[test]
    fn backward_pass_state_new_sizes_buffers_correctly() {
        let n_workers_local = 2_usize;
        let n_ranks = 3_usize;
        let bwd_max_openings = 5_usize;
        let real_states_capacity = 10_usize;

        let state = BackwardPassState::new(
            n_workers_local,
            n_ranks,
            bwd_max_openings,
            real_states_capacity,
            7,
            4,
            3,
        );

        let send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE;

        // Empty/zero-sized on construction (grown lazily):
        assert!(state.probabilities_buf.is_empty());
        assert!(state.successor_active_slots_buf.is_empty());
        assert!(state.metadata_sync_buf.is_empty());
        assert!(state.global_increments_buf.is_empty());

        // Pre-sized:
        assert_eq!(state.bwd_stats_send_buf.len(), send_stride);
        assert_eq!(state.bwd_stats_recv_buf.len(), n_ranks * send_stride);
        assert_eq!(state.bwd_stats_counts.len(), n_ranks);
        assert!(state.bwd_stats_counts.iter().all(|&c| c == send_stride));
        assert_eq!(state.bwd_stats_displs.len(), n_ranks);
        assert_eq!(state.bwd_stats_displs[0], 0);
        assert_eq!(state.bwd_stats_displs[1], send_stride);
        assert_eq!(state.bwd_stats_displs[2], 2 * send_stride);
        assert_eq!(
            state.bwd_stats_unpack_buf.len(),
            n_ranks * n_workers_local * bwd_max_openings
        );
    }

    /// Verify that `BackwardPassState::run` on a minimal 2-stage, 1-hydro,
    /// 1-opening study produces a non-empty `BackwardResult` with the expected
    /// cut count and preserves result parity with the equivalent
    /// `run_backward_pass` shim call.
    ///
    /// Setup mirrors `two_stage_system_two_trial_states_generates_two_cuts_at_stage_0`
    /// in `backward.rs`, which documents the expected arithmetic:
    ///
    /// - `MockSolver` returns `objective=100.0`, `dual[0]=-5.0`
    /// - Two trial points with states `[10.0]` and `[20.0]`
    /// - Expected: 2 cuts at stage 0, 0 cuts at stage 1
    #[test]
    fn backward_pass_state_run_preserves_one_stage_scenario_result() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        // Production carries a separate frozen-template buffer alongside
        // `ctx.templates`; mirror that here so `frozen` does not alias the
        // `&templates` borrow held by `ctx`.
        let frozen_templates = templates.clone();
        let base_rows = vec![1_usize; n_stages];
        let n_state = state.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(n_state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims,
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        // Capture local_count before mutably borrowing exchange inside the struct literal.
        let local_count = exchange.local_count();
        let mut state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            n_state,
            n_stages,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let result = state
            .run(&mut inputs)
            .expect("backward pass must not error");

        // 2 trial points × 1 stage with a successor → 2 cuts at stage 0.
        assert_eq!(
            result.cuts_generated, 2,
            "expected 2 cuts from 2 trial points"
        );
        assert_eq!(
            fcf.active_cuts(0).count(),
            2,
            "stage 0 must hold exactly 2 active cuts"
        );
        assert_eq!(
            fcf.active_cuts(1).count(),
            0,
            "stage 1 (last stage) must have no cuts"
        );
        // BackwardResult must be non-empty (stage_stats populated for stage 0).
        assert!(
            !result.stage_stats.is_empty(),
            "stage_stats must be non-empty after a successful backward pass"
        );
    }

    /// `records`, with `leaf_node_ids[m]` written into trial point `m`'s
    /// LAST-stage record (the leaf visit); every earlier stage keeps the
    /// placeholder `0` — mirroring [`trial_state_records`] except for that one
    /// per-trial override.
    fn trial_state_records_with_leaf_ids(
        states: &[Vec<f64>],
        n_stages: usize,
        leaf_node_ids: &[i32],
    ) -> Vec<TrajectoryRecord> {
        states
            .iter()
            .zip(leaf_node_ids)
            .flat_map(|(state, &leaf_node_id)| {
                (0..n_stages).map(move |t| TrajectoryRecord {
                    primal: Vec::new(),
                    dual: Vec::new(),
                    stage_cost: 0.0,
                    node_id: if t + 1 == n_stages { leaf_node_id } else { 0 },
                    state: state.clone(),
                })
            })
            .collect()
    }

    /// Runs `BackwardPassState::run` once over a declared K-fan and returns
    /// `(cuts_generated, root_pool_active_cuts, leaf_pool_active_cuts)`.
    #[allow(clippy::too_many_arguments)]
    fn run_backward_over_k_fan(
        node_graph: &NodeGraph,
        stochastic: &cobre_stochastic::StochasticContext,
        state: &StateSpace,
        templates: &[StageTemplate],
        base_rows: &[usize],
        n_stages: usize,
        records: &[TrajectoryRecord],
    ) -> (usize, usize, usize) {
        let frozen_templates = templates.to_vec();
        let n_state = state.n_state;
        let trial_count = records.len() / n_stages;
        let forward_passes = u32::try_from(trial_count).expect("trial_count fits u32");
        let bwd_max_openings = max_successor_outcome_count(node_graph).max(1);

        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            n_state,
            forward_passes,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut exchange = ExchangeBuffers::new(n_state, trial_count, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates,
            base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph,
            horizon: &horizon,
            state,
            cut_state_layouts: &all_enabled_cut_state_layouts(state, n_stages),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state_machine = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            n_state,
            n_stages,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let result = state_machine
            .run(&mut inputs)
            .expect("backward pass over a declared K-fan must not error");

        let root_pool = node_graph.nodes[0].pool_id;
        let leaf_pool = node_graph.nodes[node_graph.successors[0][0].child].pool_id;
        (
            result.cuts_generated,
            fcf.active_cuts(root_pool).count(),
            fcf.active_cuts(leaf_pool).count(),
        )
    }

    /// Trial states exchanged to the backward sweep route to the K-fan's
    /// shared leaf pool the same way whether each trial's stage-1 record
    /// carries the SPECIFIC leaf it visited or one stage-uniform id — the
    /// exchange and cut generation are keyed by trial-point position and the
    /// declared graph's own (leaf-sharing) pool structure, never by the
    /// per-record `node_id`. Two runs differing ONLY in that field must
    /// produce byte-identical cut counts.
    #[test]
    fn backward_pass_state_run_over_k_fan_is_invariant_to_per_trial_leaf_node_id() {
        use cobre_core::temporal::{Node, PolicyGraph, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                realization_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let study_stage_ids = [0_i32, 1_i32];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![node(0, 0), node(1, 1), node(2, 1), node(3, 1)],
            transitions: vec![
                transition(0, 1, 1.0 / 3.0),
                transition(0, 2, 1.0 / 3.0),
                transition(0, 3, 1.0 / 3.0),
            ],
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared K-fan graph must build");
        let leaf_node_ids: Vec<i32> = node_graph.successors[0]
            .iter()
            .map(|s| node_graph.node_ids[s.child])
            .collect();
        assert_eq!(leaf_node_ids.len(), 3, "the K-fan must declare 3 leaves");

        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let trial_states = vec![vec![10.0], vec![20.0], vec![30.0]];

        let hetero_records =
            trial_state_records_with_leaf_ids(&trial_states, n_stages, &leaf_node_ids);
        let homogeneous_leaf_ids = vec![leaf_node_ids[0]; trial_states.len()];
        let homo_records =
            trial_state_records_with_leaf_ids(&trial_states, n_stages, &homogeneous_leaf_ids);

        let hetero = run_backward_over_k_fan(
            &node_graph,
            &stochastic,
            &state,
            &templates,
            &base_rows,
            n_stages,
            &hetero_records,
        );
        let homo = run_backward_over_k_fan(
            &node_graph,
            &stochastic,
            &state,
            &templates,
            &base_rows,
            n_stages,
            &homo_records,
        );

        assert_eq!(
            hetero, homo,
            "(cuts_generated, root_pool_active_cuts, leaf_pool_active_cuts) must be identical \
             whether stage-1 records carry each trial's own visited leaf or one stage-uniform \
             leaf id"
        );
        assert_eq!(
            hetero.0,
            trial_states.len(),
            "3 trial points must each produce one cut at the root pool"
        );
    }

    /// `records` with `stage1_ids[m]` written into trial point `m`'s STAGE-1
    /// record — the interior-node visit a two-interior-node level routes on.
    /// Stage 0 carries the root id `0`; every other stage a placeholder `0`
    /// (routing never reads a single-node level's records).
    fn trial_state_records_with_stage1_ids(
        states: &[Vec<f64>],
        n_stages: usize,
        stage1_ids: &[i32],
    ) -> Vec<TrajectoryRecord> {
        states
            .iter()
            .zip(stage1_ids)
            .flat_map(|(state, &stage1_id)| {
                (0..n_stages).map(move |t| TrajectoryRecord {
                    primal: Vec::new(),
                    dual: Vec::new(),
                    stage_cost: 0.0,
                    node_id: if t == 1 { stage1_id } else { 0 },
                    state: state.clone(),
                })
            })
            .collect()
    }

    /// `n_workers` `MockSolver` workspaces, worker ids `0..n_workers`, for the
    /// multi-worker routing regression (`split_workers_mut` splits the basis store
    /// across exactly these).
    fn workspaces_n(count: usize, n_state: usize) -> Vec<SolverWorkspace<MockSolver>> {
        (0..count)
            .map(|i| {
                let mut ws =
                    single_workspace(MockSolver::always_ok(solution_1_0(100.0, -5.0)), n_state);
                let mut w = ws.remove(0);
                w.worker_id = i as i32;
                w
            })
            .collect()
    }

    /// Runs `BackwardPassState::run` once over a declared 3-stage binary tree
    /// (root → 2 interior nodes → 4 leaves) under `scheduler` with `n_workers`
    /// worker threads, returning `(cuts_generated, per_pool_active_cuts)` with the
    /// second entry indexed by pool id.
    #[allow(clippy::too_many_arguments)]
    fn run_backward_over_binary_tree(
        node_graph: &NodeGraph,
        stochastic: &cobre_stochastic::StochasticContext,
        state: &StateSpace,
        templates: &[StageTemplate],
        base_rows: &[usize],
        n_stages: usize,
        records: &[TrajectoryRecord],
        scheduler: BackwardScheduler,
        n_workers: usize,
    ) -> (usize, Vec<usize>) {
        // Frozen overlay is per POOL: pool `p`'s base is `templates[pool_stage[p]]`
        // (all templates identical here, so this just spans every pool, including
        // the leaf pool at position >= n_stages).
        let frozen_templates: Vec<StageTemplate> = (0..node_graph.n_pools)
            .map(|p| templates[node_graph.pool_stage[p]].clone())
            .collect();
        let n_state = state.n_state;
        let trial_count = records.len() / n_stages;
        let forward_passes = u32::try_from(trial_count).expect("trial_count fits u32");
        let bwd_max_openings = node_graph
            .successors
            .iter()
            .map(|succs| {
                succs
                    .iter()
                    .map(|s| node_graph.nodes[s.child].openings.len)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(1);

        let mut fcf = FutureCostFunction::new(
            node_graph.n_pools,
            n_state,
            forward_passes,
            10,
            &vec![0; node_graph.n_pools],
        );
        let mut exchange = ExchangeBuffers::new(n_state, trial_count, 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let comm = StubComm;
        let mut workspaces = workspaces_n(n_workers, n_state);
        // The basis store's second axis is NODE count (keyed by node position),
        // not stage count — the two coincide only on a chain.
        let mut basis_store = empty_basis_store(exchange.local_count(), node_graph.nodes.len());
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        // Cut-batch scratch is pool-indexed (backward writes `cut_batches[successor_pool_id]`).
        let mut cut_batches = empty_cut_batches(node_graph.n_pools);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates,
            base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph,
            horizon: &horizon,
            state,
            // `cut_state_layouts` is keyed by POOL id (a cut-generating node
            // owns its own pool), so it is sized by `n_pools`, not `n_stages` —
            // the two coincide only on a chain.
            cut_state_layouts: &all_enabled_cut_state_layouts(state, node_graph.n_pools),
            study_dims: &study_dims,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state_machine = BackwardPassState::new(
            n_workers,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            n_state,
            n_stages,
        );
        state_machine.set_scheduler(scheduler);

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let result = state_machine
            .run(&mut inputs)
            .expect("backward pass over a declared binary tree must not error");

        let per_pool: Vec<usize> = (0..node_graph.n_pools)
            .map(|p| fcf.active_cuts(p).count())
            .collect();
        (result.cuts_generated, per_pool)
    }

    /// Cut-generation routing at a two-INTERIOR-node level: each cut-generating
    /// node anchors its cut ONLY at the trial states routed to its own pool. Two
    /// forwards that partition trajectories differently across the siblings must
    /// produce DIFFERENT per-pool cut sets — a trajectory switching sibling moves
    /// its cut from one pool to the other. This is the mirror of
    /// `backward_pass_state_run_over_k_fan_is_invariant_to_per_trial_leaf_node_id`
    /// (where a single cut-generating node makes routing a no-op); under per-stage
    /// aggregation both partitions would wrongly give each pool `F` cuts. Both
    /// backward schedulers must route identically.
    #[test]
    fn backward_pass_state_routes_trial_states_to_the_visited_nodes_pool() {
        use cobre_core::temporal::{Node, PolicyGraph, PolicyGraphType, Transition};
        use cobre_io::StageIdResolver;

        use crate::setup::node_graph::build_node_graph;

        fn node(id: i32, stage_id: i32) -> Node {
            Node {
                id,
                stage_id,
                realization_id: None,
                label: None,
            }
        }
        fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
            Transition {
                source_id,
                target_id,
                probability,
                annual_discount_rate_override: None,
            }
        }

        let n_stages = 3_usize;
        let branching = 2_usize;
        let stochastic = make_stochastic_context(n_stages, branching);
        let study_stage_ids = [0_i32, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        // Root (stage 0) → two INTERIOR nodes (stage 1, distinct pools) → four
        // leaves (stage 2, one shared pool). backward_cut_levels == [[1, 2], [0]].
        let graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            nodes: vec![
                node(0, 0),
                node(1, 1),
                node(2, 1),
                node(3, 2),
                node(4, 2),
                node(5, 2),
                node(6, 2),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            season_map: None,
        };
        let node_graph = build_node_graph(&graph, n_stages, &resolver, &stochastic)
            .expect("declared binary-tree graph must build");

        let root_pool = node_graph.nodes[0].pool_id;
        let pool1 = node_graph.nodes[1].pool_id;
        let pool2 = node_graph.nodes[2].pool_id;
        assert_ne!(
            pool1, pool2,
            "two interior siblings at one level must own distinct pools"
        );

        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let base_rows = vec![1_usize; n_stages];
        let trial_states = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];
        let f = trial_states.len();

        // Config A: 3 trajectories visit node 1, 1 visits node 2 (at stage 1).
        let ids_a = [1_i32, 1, 1, 2];
        // Config B: the mirror partition — 1 visits node 1, 3 visit node 2.
        let ids_b = [1_i32, 2, 2, 2];

        // The by-scenario path exercises routing on a multi-successor,
        // multi-pool level. The by-node scheduler is not run here: it indexes a
        // single successor's `solve_order` and does not yet handle a node's
        // flattened multi-successor opening set (orthogonal to routing; its
        // routing threading is covered on chains by the by-node determinism
        // gates in `tests/mpi_wire.rs`).
        let scheduler = BackwardScheduler::ByScenario {};
        let records_a = trial_state_records_with_stage1_ids(&trial_states, n_stages, &ids_a);
        let records_b = trial_state_records_with_stage1_ids(&trial_states, n_stages, &ids_b);

        // Both worker counts. `n_workers == 2` is the regression: a node's routed
        // subset is scattered across the global scenario axis, so a worker must be
        // handed only the routed points inside its own contiguous basis window —
        // splitting `trial_points.len()` evenly instead indexes a routed `m` out of
        // another worker's basis slice (a deterministic OOB panic). Routing is
        // worker-count invariant, so the per-pool counts match across both legs.
        for n_workers in [1_usize, 2] {
            let (gen_a, pools_a) = run_backward_over_binary_tree(
                &node_graph,
                &stochastic,
                &state,
                &templates,
                &base_rows,
                n_stages,
                &records_a,
                scheduler,
                n_workers,
            );
            let (_gen_b, pools_b) = run_backward_over_binary_tree(
                &node_graph,
                &stochastic,
                &state,
                &templates,
                &base_rows,
                n_stages,
                &records_b,
                scheduler,
                n_workers,
            );

            assert_eq!(
                pools_a[root_pool], f,
                "the root (single-node level) anchors a cut at every trajectory's stage-0 state"
            );
            assert_eq!(
                pools_a[pool1], 3,
                "node 1's pool: the 3 trajectories that visited it"
            );
            assert_eq!(
                pools_a[pool2], 1,
                "node 2's pool: the 1 trajectory that visited it"
            );
            assert_eq!(
                pools_a[pool1] + pools_a[pool2],
                f,
                "Σ over the level's pools of their trial counts must equal F"
            );
            assert_eq!(
                gen_a,
                2 * f,
                "F cuts at the two-interior-node level plus F at the root"
            );

            assert_eq!(pools_b[root_pool], f);
            assert_eq!(pools_b[pool1], 1);
            assert_eq!(pools_b[pool2], 3);

            assert_ne!(
                pools_a[pool1], pools_b[pool1],
                "routing is load-bearing: repartitioning trajectories across the siblings must \
                 move cuts between the sibling pools (per-stage aggregation would give each F)"
            );
        }
    }

    /// A profile installed via `set_profile` before `run()` is the one
    /// `ProfiledSolver::current_profile()` reports afterwards — the resolved
    /// profile reaches the solver, not just the stored default.
    #[test]
    fn backward_pass_state_set_profile_reaches_current_profile_after_run() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state_layout_fixture = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let base_rows = vec![1_usize; n_stages];
        let n_state = state_layout_fixture.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(n_state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state_layout_fixture,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state_layout_fixture, n_stages),
            study_dims: &study_dims,
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        let local_count = exchange.local_count();
        let mut bwd_state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            n_state,
            n_stages,
        );
        let resolved =
            Phase::Backward.resolve_profile(Some(&cobre_io::config::PhaseSolverProfileConfig {
                dual_edge_weight: Some(cobre_io::config::DualEdgeWeight::SteepestEdge),
                scale: Some(cobre_io::config::ScaleStrategy::SolverScaling),
                price: Some(cobre_io::config::PriceStrategy::Row),
                primal_feasibility_tolerance: Some(1e-7),
                dual_feasibility_tolerance: None,
                presolve: None,
                simplex_update_limit: None,
                cost_perturbation: None,
                refactor_error_tolerance: None,
                factor_pivot_threshold: None,
                use_warm_start: None,
                steepest_edge_devex_fallback_threshold: None,
            }));
        bwd_state.set_profile(resolved);

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let _ = bwd_state
            .run(&mut inputs)
            .expect("backward pass must not error");

        assert_eq!(
            inputs.workspaces[0].solver.current_profile(),
            &resolved,
            "the profile installed via set_profile must be the one stored on \
             current_profile after run()"
        );
    }

    /// Verify that `state_duals_buf` on the per-worker `BackwardAccumulators`
    /// is correctly sized after the backward pass completes.
    ///
    /// After `BackwardPassState::run` returns, the single worker's
    /// `backward_accum.state_duals_buf` must hold exactly `state.n_state`
    /// entries — the last opening's unscaled duals that were written during
    /// the final trial-point/opening iteration.
    ///
    /// This guards against buffer re-use bugs where the fill loop writes the
    /// wrong number of entries across consecutive openings.
    #[test]
    fn backward_pass_state_duals_buf_len_equals_n_state_after_run() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let base_rows = vec![1_usize; n_stages];
        let n_state = state.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 10, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(n_state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let solution = solution_1_0(100.0, -5.0);
        let comm = StubComm;
        let mut workspaces = single_workspace(MockSolver::always_ok(solution), n_state);
        let mut basis_store = empty_basis_store(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &crate::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state, n_stages),
            study_dims: &study_dims,
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
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let bwd_max_openings = n_openings;
        let local_count = exchange.local_count();
        let mut state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            n_state,
            local_count,
            n_state,
            n_stages,
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let _ = state
            .run(&mut inputs)
            .expect("backward pass must not error");

        assert_eq!(
            inputs.workspaces[0].backward_accum.state_duals_buf.len(),
            n_state,
            "state_duals_buf must have length n_state after backward pass"
        );
    }

    /// AC2: a cut generated at iteration `g` that binds at a later
    /// iteration `i > g` under DCS must have its `last_active_iter` advanced to
    /// `i` by the (unchanged) per-stage metadata sync, fed by the DCS-maintained
    /// binding-count contribution.
    ///
    /// This drives the real [`BackwardPassState::sync_stage_metadata`] with a
    /// per-worker `metadata_sync_contribution` shaped exactly as the DCS backward
    /// path produces it (see `backward::tests::backward_dcs_binding_counts_match_frozen`,
    /// which proves the DCS path bumps exactly the binding slot): slot 1 bound at
    /// iteration `i`, slot 0 did not. Before the sync `last_active_iter == g` for
    /// both slots; after, the binding slot 1 advances to `i` (even though its
    /// `iteration_generated` is the older `g`), while the non-binding slot 0
    /// stays frozen at `g` (metadata staleness, unrelated to the frozen-template
    /// LP mode). This is the §3.1 clause-1 prerequisite the seed reads.
    #[test]
    fn dcs_binding_contribution_advances_last_active_iter() {
        use crate::cut_selection::CutMetadata;

        let g = 1_u64; // generation iteration
        let i = 5_u64; // binding (current) iteration, i > g
        let successor = 1_usize;
        let n_state = 1_usize;
        let n_stages = 2_usize;
        let n_openings = 1_usize;

        // FCF with two cuts at the successor stage, both generated at `g` and
        // last active at `g` (stale). Slot 1 is the one that will bind at `i`.
        let mut fcf = FutureCostFunction::new(n_stages, n_state, 8, 10, &vec![0; n_stages]);
        fcf.add_cut(successor, g, 0, 1.0, &[0.0]);
        fcf.add_cut(successor, g, 1, 0.0, &[2.0]);
        let meta = |generated: u64, last: u64| CutMetadata {
            iteration_generated: generated,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: last,
        };
        fcf.pools[successor].set_metadata_for_test(0, meta(g, g));
        fcf.pools[successor].set_metadata_for_test(1, meta(g, g));
        let pop = fcf.pools[successor].populated();

        // Pre-state: both slots frozen at generation iteration `g`.
        assert_eq!(fcf.pools[successor].metadata(0).last_active_iter, g);
        assert_eq!(fcf.pools[successor].metadata(1).last_active_iter, g);

        // One worker whose DCS binding-count contribution bumps only slot 1 (the
        // resident binding cut), matching what the DCS path emits at iteration i.
        let mut workspaces =
            single_workspace(MockSolver::always_ok(solution_1_0(0.0, 0.0)), n_state);
        let contrib = &mut workspaces[0].backward_accum.metadata_sync_contribution;
        contrib.clear();
        contrib.resize(pop, 0);
        contrib[1] = 1;

        let comm = StubComm;
        let mut state = BackwardPassState::new(1, 1, n_openings, n_state, 1, n_state, n_stages);

        state
            .sync_stage_metadata(successor, i, &workspaces, &mut fcf, &comm)
            .expect("metadata sync must succeed");

        // Slot 1 bound at `i`: last_active_iter advances from g to i; active_count
        // accrues the increment. Slot 0 did not bind: it stays frozen at g.
        assert_eq!(
            fcf.pools[successor].metadata(1).last_active_iter,
            i,
            "binding slot 1 must advance to iteration {i}"
        );
        assert_eq!(fcf.pools[successor].metadata(1).active_count, 1);
        assert_eq!(
            fcf.pools[successor].metadata(0).last_active_iter,
            g,
            "non-binding slot 0 must stay frozen at its generation iteration {g}"
        );
        assert_eq!(fcf.pools[successor].metadata(0).active_count, 0);
    }

    // ── successor outcome set assembly ──────────────────────────────────────

    /// A single node fanning into two children (ascending id `1`, `2`) with
    /// distinct within-node opening counts and distinct out-edge
    /// probabilities: child `1` carries 2 openings at `P(0→1) = 0.25`, child
    /// `2` carries 2 openings at `P(0→2) = 0.75`.
    fn k_fan_node_graph() -> NodeGraph {
        NodeGraph {
            node_ids: vec![0, 1, 2],
            nodes: vec![
                NodeRuntime {
                    stage: 0,
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: 1,
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 2,
                        q: 0.5,
                    },
                },
                NodeRuntime {
                    stage: 1,
                    pool_id: 2,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 2,
                        q: 0.5,
                    },
                },
            ],
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: 1,
                        probability: 0.25,
                    },
                    NodeSuccessor {
                        child: 2,
                        probability: 0.75,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ],
            n_pools: 3,
            pool_stage: vec![0, 1, 1],
        }
    }

    #[test]
    fn assemble_successor_outcome_weights_k_fan_canonical_order_and_product_weights() {
        let node_graph = k_fan_node_graph();
        let mut buf = Vec::new();
        assemble_successor_outcome_weights(&node_graph, 0, &mut buf);
        // Ascending child id (1 then 2), then within-child ω: P(0→1)·q_{1,ψ} =
        // 0.25·0.5 = 0.125 (twice), P(0→2)·q_{2,ψ} = 0.75·0.5 = 0.375 (twice).
        assert_eq!(buf, vec![0.125, 0.125, 0.375, 0.375]);
    }

    #[test]
    fn assemble_successor_outcome_weights_chain_degenerate_reduces_to_pinned_uniform_bit_pattern() {
        let n = 7_usize;
        let q = 1.0_f64 / (n as f64);
        let node_graph = NodeGraph {
            node_ids: vec![0, 1],
            nodes: vec![
                NodeRuntime {
                    stage: 0,
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: 1,
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: n,
                        q,
                    },
                },
            ],
            successors: vec![
                vec![NodeSuccessor {
                    child: 1,
                    probability: 1.0,
                }],
                Vec::new(),
            ],
            n_pools: 2,
            pool_stage: vec![0, 1],
        };
        let mut buf = Vec::new();
        assemble_successor_outcome_weights(&node_graph, 0, &mut buf);
        assert_eq!(buf.len(), n);
        let expected_bits = q.to_bits();
        for &w in &buf {
            assert_eq!(
                w.to_bits(),
                expected_bits,
                "chain weight (P=1.0 times q) must be the exact pinned \
                 1.0/(n as f64) bit pattern, not a re-derived value"
            );
        }
    }

    /// AC: analytical product-weighted cut regression. `aggregate_cut`'s
    /// result over the assembled weights must equal the closed-form flat sum
    /// `Σ_(m,ψ) P(n→m)·q_{m,ψ}·outcome`, with the outcome vector ordered by
    /// ascending child node id then ω.
    #[test]
    fn analytical_product_weighted_cut_regression_k_fan() {
        let node_graph = k_fan_node_graph();
        let mut probabilities = Vec::new();
        assemble_successor_outcome_weights(&node_graph, 0, &mut probabilities);

        // One outcome per (child, ψ) in the same canonical order as `probabilities`:
        // child 1's two openings, then child 2's two.
        let outcomes = vec![
            BackwardOutcome {
                intercept: 10.0,
                coefficients: vec![1.0, 0.0],
                objective_value: 10.0,
            },
            BackwardOutcome {
                intercept: 20.0,
                coefficients: vec![2.0, 0.0],
                objective_value: 20.0,
            },
            BackwardOutcome {
                intercept: 30.0,
                coefficients: vec![0.0, 3.0],
                objective_value: 30.0,
            },
            BackwardOutcome {
                intercept: 40.0,
                coefficients: vec![0.0, 4.0],
                objective_value: 40.0,
            },
        ];

        let (intercept, coefficients) =
            RiskMeasure::Expectation.aggregate_cut(&outcomes, &probabilities);

        let expected_intercept: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.intercept)
            .sum();
        let expected_c0: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.coefficients[0])
            .sum();
        let expected_c1: f64 = probabilities
            .iter()
            .zip(&outcomes)
            .map(|(p, o)| p * o.coefficients[1])
            .sum();

        assert!((intercept - expected_intercept).abs() < 1e-12);
        assert!((coefficients[0] - expected_c0).abs() < 1e-12);
        assert!((coefficients[1] - expected_c1).abs() < 1e-12);
    }

    /// AC: `CVaR` tie-break order. Two tied-objective outcomes with distinct
    /// product weights: `CVaR`'s greedy allocation processes tied entries in
    /// their array (= canonical child-id-then-ω) index order — `child 1`
    /// (index 0) is allocated its full upper bound first, `child 2` (index 1)
    /// receives only the remainder. A `realization_id`- or declaration-ordered
    /// successor set (child 2 processed first) would allocate the opposite
    /// way and produce a different intercept.
    #[test]
    fn cvar_aggregation_tie_break_follows_canonical_child_order() {
        let node_graph = NodeGraph {
            node_ids: vec![0, 1, 2],
            nodes: vec![
                NodeRuntime {
                    stage: 0,
                    pool_id: 0,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: 1,
                    pool_id: 1,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
                NodeRuntime {
                    stage: 1,
                    pool_id: 2,
                    openings: NodeOpenings {
                        source: OpeningSource::Generated,
                        offset: 0,
                        len: 1,
                        q: 1.0,
                    },
                },
            ],
            successors: vec![
                vec![
                    NodeSuccessor {
                        child: 1,
                        probability: 0.3,
                    },
                    NodeSuccessor {
                        child: 2,
                        probability: 0.7,
                    },
                ],
                Vec::new(),
                Vec::new(),
            ],
            n_pools: 3,
            pool_stage: vec![0, 1, 1],
        };
        let mut probabilities = Vec::new();
        assemble_successor_outcome_weights(&node_graph, 0, &mut probabilities);
        assert_eq!(probabilities, vec![0.3, 0.7]);

        // Tied objective_value: the sort key carries no information, so the
        // greedy allocation's processing order is decided purely by array
        // (= canonical) index.
        let outcomes = vec![
            BackwardOutcome {
                intercept: 10.0,
                coefficients: vec![],
                objective_value: 100.0,
            },
            BackwardOutcome {
                intercept: 20.0,
                coefficients: vec![],
                objective_value: 100.0,
            },
        ];
        let risk = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };
        let (intercept, _) = risk.aggregate_cut(&outcomes, &probabilities);

        // mu[0] = min(0.3/0.5, 1.0) = 0.6, remaining = 0.4;
        // mu[1] = min(0.7/0.5, 0.4) = 0.4.
        // intercept = 0.6*10.0 + 0.4*20.0 = 14.0 — NOT 20.0, which is what a
        // child-2-processed-first (declaration/realization_id) order would give.
        assert!(
            (intercept - 14.0).abs() < 1e-12,
            "expected canonical-order CVaR tie-break intercept 14.0, got {intercept}"
        );
    }

    // ── by-node auto-selection ─────────────────────────────────────────

    #[test]
    fn by_node_auto_selection_singleton_picks_by_node_else_keeps_configured() {
        use cobre_io::config::BackwardScheduler;

        // Singleton trial-state axis: one trial point, more than one scenario
        // collapsed onto it.
        assert!(trial_state_axis_is_singleton(1, 4));
        // A chain stage (one trial point per scenario) is NOT a singleton, and a
        // lone sampled trajectory (one scenario) is not one either.
        assert!(!trial_state_axis_is_singleton(4, 4));
        assert!(!trial_state_axis_is_singleton(1, 1));

        // Singleton ⇒ by-node regardless of the configured scheduler.
        assert!(matches!(
            resolve_backward_scheduler(true, false, BackwardScheduler::ByScenario {}),
            BackwardScheduler::ByNode { .. }
        ));
        assert!(matches!(
            resolve_backward_scheduler(true, false, BackwardScheduler::ByNode { block_size: None }),
            BackwardScheduler::ByNode { .. }
        ));
        // Non-singleton (chain) ⇒ configured scheduler unchanged.
        assert!(matches!(
            resolve_backward_scheduler(false, false, BackwardScheduler::ByScenario {}),
            BackwardScheduler::ByScenario {}
        ));
        assert!(matches!(
            resolve_backward_scheduler(
                false,
                false,
                BackwardScheduler::ByNode { block_size: None }
            ),
            BackwardScheduler::ByNode { .. }
        ));
        // An active DCS iteration forces the by-scenario path even at a singleton.
        assert!(matches!(
            resolve_backward_scheduler(true, true, BackwardScheduler::ByNode { block_size: None }),
            BackwardScheduler::ByScenario {}
        ));
    }

    // ── per-node basis isolation + chain byte-address ──────────────────

    #[test]
    fn per_node_basis_isolation_and_chain_byte_address() {
        // K-fan: node 0's two distinct successor nodes (positions 1 and 2) share
        // one `BasisStoreSliceMut` sized by node count.
        let ng = k_fan_node_graph();
        let node_a = ng.successors[0][0].child;
        let node_b = ng.successors[0][1].child;
        assert_ne!(node_a, node_b);

        let mut store = BasisStore::new(1, ng.nodes.len());
        {
            let mut slices = store.split_workers_mut(1);
            let slice = &mut slices[0];
            *slice.get_mut(0, node_a) =
                Some(CapturedBasis::new(2, 1, 0, 0, 0, ng.node_ids[node_a]));
            assert!(
                slice.get(0, node_a).is_some(),
                "basis saved at successor node A resolves at node A"
            );
            assert!(
                slice.get(0, node_b).is_none(),
                "successor node B resolves nothing — per-node isolation by successor node"
            );
        }

        // Synthesized chain: successor node position == successor stage, so the
        // backward basis key `(m, successor_node)` is the pre-change
        // `(m, successor_stage)` flat slot `[m*num_nodes + node]` byte-for-byte
        // (num_nodes == num_stages on the chain).
        let n_stages = 4usize;
        let stochastic = make_stochastic_context(n_stages, 3);
        let chain = crate::test_support::chain_node_graph(&stochastic);
        assert_eq!(chain.nodes.len(), n_stages);
        for t in 0..n_stages - 1 {
            assert_eq!(chain.nodes[t].stage, t);
            assert_eq!(
                chain.successors[t][0].child,
                t + 1,
                "chain successor node position must equal the successor stage"
            );
        }
        let chain_store = BasisStore::new(2, chain.nodes.len());
        assert_eq!(
            chain_store.num_nodes(),
            n_stages,
            "chain node-axis stride equals num_stages, so (m, successor_node) == (m, successor_stage)"
        );
    }

    // ── no graph-shape dispatch on the backward path ──────────────

    #[test]
    fn backward_engine_has_no_graph_shape_dispatch() {
        // Chain parity is degeneracy, not dispatch: no backward engine path
        // branches on a graph-shape predicate to reach preserved legacy code.
        let sources: [(&str, &str); 6] = [
            (
                "backward_pass_state",
                include_str!("backward_pass_state.rs"),
            ),
            ("backward/mod", include_str!("backward/mod.rs")),
            (
                "backward/by_scenario",
                include_str!("backward/by_scenario.rs"),
            ),
            ("backward/by_node", include_str!("backward/by_node.rs")),
            ("backward/lp_setup", include_str!("backward/lp_setup.rs")),
            (
                "backward/replicated",
                include_str!("backward/replicated.rs"),
            ),
        ];
        // Assemble the forbidden tokens from fragments so this test's own source
        // (folded into the first `include_str!`) does not self-match.
        let us = "_";
        let forbidden = [
            format!("is{us}chain"),
            format!("nodes.is{us}empty()"),
            format!("graph.is{us}none()"),
        ];
        for (name, src) in sources {
            for pat in &forbidden {
                assert!(
                    !src.contains(pat.as_str()),
                    "backward engine file `{name}` contains shape-dispatch predicate `{pat}`"
                );
            }
        }
    }
}
