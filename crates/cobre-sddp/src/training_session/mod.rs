//! Training session state container and iteration loop.
//!
//! [`TrainingSession`] owns all scratch buffers for a single [`crate::training::train`] call.
//! No hot-path allocations; forward and backward passes encapsulated in their own state structs.

pub(crate) mod iteration_scratch;
pub(crate) mod rank_distribution;
pub(crate) mod results;
pub(crate) mod runtime;
use self::iteration_scratch::IterationScratch;
use self::rank_distribution::RankDistribution;
use self::results::TrainingResults;
use self::runtime::RuntimeHandles;

use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::Instant;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::{StageRowSelectionRecord, TrainingEvent};
use cobre_solver::SolverInterface;

use crate::{
    SddpError, TrainingConfig,
    backward_pass_state::{BackwardPassInputs, BackwardPassState},
    context::{StageContext, TrainingContext},
    convergence::ConvergenceMonitor,
    cut::fcf::FutureCostFunction,
    cut_selection::CutActivityUpdates,
    cut_sync::CutSyncBuffers,
    forward::{build_cut_row_batch_into, sync_forward},
    forward_pass_state::{ForwardPassInputs, ForwardPassState},
    lower_bound::evaluate_lower_bound,
    lower_bound::{LbEvalScratchBundle, LbEvalSpec},
    solver_stats::{
        SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, aggregate_solver_statistics,
        pack_delta_scalars, unpack_delta_scalars,
    },
    state_exchange::ExchangeBuffers,
    stopping_rule::RULE_GRACEFUL_SHUTDOWN,
    training::{TrainingOutcome, TrainingResult, broadcast_basis_cache},
    workspace::{BasisStore, WorkspacePool, WorkspaceSizing},
};

// ---------------------------------------------------------------------------
// emit helper (mirrors training.rs emit)
// ---------------------------------------------------------------------------

#[inline]
fn emit(sender: Option<&Sender<TrainingEvent>>, event: TrainingEvent) {
    if let Some(s) = sender {
        let _ = s.send(event);
    }
}

// ---------------------------------------------------------------------------
// IterationOutcome
// ---------------------------------------------------------------------------

/// Result of a single training iteration.
///
/// Returned by [`TrainingSession::run_iteration`] to let the outer orchestrator
/// in [`crate::training::train`] decide whether to continue, stop, or handle
/// an error.
#[derive(Debug)]
pub(crate) enum IterationOutcome {
    /// The iteration completed normally; the loop should continue.
    Continue,
    /// A stopping rule triggered; the loop should break.
    Converged,
    /// An external shutdown flag was observed; the loop should break.
    Shutdown,
}

// ---------------------------------------------------------------------------
// TrainingSession
// ---------------------------------------------------------------------------

/// Owns all per-training-run scratch state for one call to `train`.
///
/// ## Lifetime `'a`
///
/// The session borrows `solver`, `fcf`, `stage_ctx`, `training_ctx`, and `comm`
/// for the duration of the training run. All methods that call into the
/// forward / backward passes require only `&mut self` because the session IS
/// the holder of all required references.
pub(crate) struct TrainingSession<'a, S: SolverInterface + Send, C: Communicator> {
    // ── Borrowed inputs (live for 'a) ─────────────────────────────────────
    solver: &'a mut S,
    fcf: &'a mut FutureCostFunction,
    stage_ctx: &'a StageContext<'a>,
    training_ctx: &'a TrainingContext<'a>,
    comm: &'a C,

    // ── Training configuration (the training configuration for this run) ──
    config: TrainingConfig,

    // ── Runtime handles (per-invocation hooks; set once in new) ───────────
    runtime: RuntimeHandles,

    // ── Rank math (constant for the run) ──────────────────────────────────
    ranks: RankDistribution,

    // ── Per-run scratch buffers (owned; reused across iterations) ─────────
    fwd_pool: WorkspacePool<S>,
    basis_store: BasisStore,
    exchange_bufs: ExchangeBuffers,
    cut_sync_bufs: CutSyncBuffers,
    visited_archive: Option<crate::visited_states::VisitedStatesArchive>,
    scratch: IterationScratch,
    convergence_monitor: ConvergenceMonitor,

    // ── Forward-pass owned scratch ────────────────────────────────────────
    fwd_state: ForwardPassState,

    // ── Backward-pass owned scratch ───────────────────────────────────────
    bwd_state: BackwardPassState,

    // ── Result accumulators (updated each iteration; finalized in finalize()) ─
    results: TrainingResults,
}

impl<'a, S, C: Communicator> TrainingSession<'a, S, C>
where
    S: SolverInterface<Profile = cobre_solver::HighsProfile> + Send,
{
    /// Allocate all per-training-run scratch and emit the `TrainingStarted` event.
    ///
    /// Performs exactly what the 400-712 prelude in the original `train()` did:
    /// rank math, `fwd_pool` construction and pre-sizing, `basis_store`,
    /// `patch_buf`, `convergence_monitor`, `exchange_bufs`, `cut_sync_bufs`,
    /// `visited_archive`, result accumulators, `cut_batches`, `lb_cut_batch`,
    /// `baked_templates`, `bake_row_batches`, `lb_cut_row_map`,
    /// `bwd_state` allocation, and the `TrainingStarted` event emission.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Solver(e)` if the workspace pool cannot be constructed.
    // RATIONALE: `i32::try_from(my_rank).expect(...)` is safe because MPI
    // rank counts are bounded by `i32::MAX` by the MPI specification. A
    // `?`-propagating variant would return `Result` for an invariant the
    // caller cannot recover from and the runtime guarantees.
    #[allow(clippy::expect_used)]
    pub(crate) fn new(
        solver: &'a mut S,
        mut config: TrainingConfig,
        fcf: &'a mut FutureCostFunction,
        stage_ctx: &'a StageContext<'a>,
        training_ctx: &'a TrainingContext<'a>,
        comm: &'a C,
        solver_factory: impl Fn() -> Result<S, cobre_solver::SolverError>,
    ) -> Result<Self, SddpError> {
        // ── Rank math ─────────────────────────────────────────────────────
        let horizon = training_ctx.horizon;
        let indexer = training_ctx.indexer;
        let num_stages = horizon.num_stages();
        let total_forward_passes = config.loop_config.forward_passes as usize;
        let ranks = RankDistribution::new(comm, num_stages, total_forward_passes, indexer.n_state);

        // ── Workspace pool ────────────────────────────────────────────────
        let n_threads = config.loop_config.n_fwd_threads.max(1);
        let mut fwd_pool = WorkspacePool::try_new(
            ranks.fwd_rank,
            n_threads,
            ranks.n_state,
            WorkspaceSizing {
                hydro_count: indexer.hydro_count,
                max_par_order: indexer.max_par_order,
                n_load_buses: stage_ctx.n_load_buses,
                max_blocks: config.loop_config.max_blocks,
                downstream_par_order: stage_ctx.downstream_par_order,
                max_openings: (0..ranks.num_stages)
                    .map(|t| training_ctx.stochastic.opening_tree().n_openings(t))
                    .max()
                    .unwrap_or(0),
                initial_pool_capacity: fcf.pools[0].capacity,
                n_state: ranks.n_state,
                max_local_fwd: ranks.max_local_fwd,
                total_forward_passes,
                noise_dim: training_ctx.stochastic.dim(),
                n_anticipated: indexer.n_anticipated,
                k_max: indexer.k_max,
            },
            solver_factory,
        )
        .map_err(SddpError::Solver)?;
        // Pre-size ws.scratch_basis to the largest template the reconstruction
        // path might ever populate. reconstruct_basis runs unconditionally on any
        // forward/backward apply with a stored basis; pre-sizing here is what
        // keeps the hot path allocation-free.
        let max_cols = stage_ctx
            .templates
            .iter()
            .map(|t| t.num_cols)
            .max()
            .unwrap_or(0);
        let max_rows = stage_ctx
            .templates
            .iter()
            .map(|t| t.num_rows)
            .max()
            .unwrap_or(0);
        fwd_pool.resize_scratch_bases(max_cols, max_rows);

        // ── Basis store ───────────────────────────────────────────────────
        // Per-scenario, per-stage basis store. Sized for the maximum local
        // forward passes so that scenario indices are stable across iterations.
        let basis_store = BasisStore::new(ranks.max_local_fwd, ranks.num_stages);

        // ── Exchange + cut-sync buffers ────────────────────────────────────
        let actual_per_rank = ranks.actual_per_rank(total_forward_passes);
        let exchange_bufs = ExchangeBuffers::with_actual_counts(
            ranks.n_state,
            ranks.max_local_fwd,
            ranks.num_ranks,
            &actual_per_rank,
        );
        let cut_sync_bufs = CutSyncBuffers::with_distribution(
            ranks.n_state,
            ranks.max_local_fwd,
            ranks.num_ranks,
            total_forward_passes,
        );

        // ── Visited-states archive ────────────────────────────────────────
        // All cut-selection strategies require the visited-states archive: the
        // unified value-evaluation kernel evaluates every populated cut at the
        // trial points stored here. Event export also needs the archive when
        // `export_states` is enabled.
        let needs_archive =
            config.cut_management.cut_selection.is_some() || config.events.export_states;
        let visited_archive = if needs_archive {
            Some(crate::visited_states::VisitedStatesArchive::new(
                ranks.num_stages,
                ranks.n_state,
                config.loop_config.max_iterations,
                total_forward_passes,
            ))
        } else {
            None
        };

        // ── Extract runtime handles; leave the rest on `config` ──────────
        // event_sender and shutdown_flag are moved out via .take() so that
        // `config.events` remains in a valid state and `config` can be stored
        // by value. export_states is Copy — it is read directly.
        let event_sender = config.events.event_sender.take();
        let shutdown_flag = config.events.shutdown_flag.take();
        let export_states = config.events.export_states;

        // ── Convergence monitor ────────────────────────────────────────────
        let convergence_monitor =
            ConvergenceMonitor::new(config.loop_config.stopping_rules.clone());

        // ── TrainingStarted event ─────────────────────────────────────────
        // Emit before moving the three locals into RuntimeHandles so that
        // the local `event_sender` binding is still available here.
        #[allow(clippy::cast_possible_truncation)]
        emit(
            event_sender.as_ref(),
            TrainingEvent::TrainingStarted {
                case_name: String::new(),
                stages: ranks.num_stages as u32,
                hydros: indexer.hydro_count as u32,
                thermals: 0,
                ranks: ranks.num_ranks as u32,
                #[allow(clippy::cast_possible_truncation)]
                threads_per_rank: n_threads as u32,
                timestamp: String::new(),
            },
        );

        // ── Runtime handles ────────────────────────────────────────────────
        let runtime = RuntimeHandles::new(event_sender, shutdown_flag, export_states);

        // ── Result accumulators ────────────────────────────────────────────
        let results = TrainingResults::new(config.loop_config.start_iteration);

        // ── Iteration scratch (reused buffers for forward/backward/cut/lb) ──
        let max_pool_capacity = fcf.pools.iter().map(|p| p.capacity).max().unwrap_or(0);
        let scratch = IterationScratch::new(
            ranks.max_local_fwd,
            ranks.num_stages,
            ranks.n_state,
            fcf.pools[0].capacity,
            max_pool_capacity,
            stage_ctx.templates[0].num_rows,
            indexer.hydro_count,
            indexer.max_par_order,
            indexer.n_anticipated,
            indexer.k_max,
            stage_ctx,
        );

        // ── Forward-pass scratch buffers ──────────────────────────────────
        // All allocated once in ForwardPassState::new; reused across iterations.
        let n_workers_local = fwd_pool.workspaces.len();
        let fwd_state =
            ForwardPassState::new(n_workers_local, ranks.num_stages, ranks.max_local_fwd);

        // ── Backward-pass scratch buffers ─────────────────────────────────
        // All allocated once in BackwardPassState::new; reused across iterations.
        let bwd_max_openings = (0..ranks.num_stages)
            .map(|t| training_ctx.stochastic.opening_tree().n_openings(t))
            .max()
            .unwrap_or(0);
        let n_ranks = comm.size();
        let real_states_capacity = exchange_bufs.real_total_scenarios() * ranks.n_state;
        let bwd_state = BackwardPassState::new(
            n_workers_local,
            n_ranks,
            bwd_max_openings,
            real_states_capacity,
        );

        Ok(Self {
            solver,
            fcf,
            stage_ctx,
            training_ctx,
            comm,
            config,
            runtime,
            ranks,
            fwd_pool,
            basis_store,
            exchange_bufs,
            cut_sync_bufs,
            visited_archive,
            scratch,
            convergence_monitor,
            fwd_state,
            bwd_state,
            results,
        })
    }

    /// Returns the range of iteration indices this session should run.
    ///
    /// The outer orchestrator in `train()` iterates over this range and passes
    /// each index to [`run_iteration`](Self::run_iteration).
    pub(crate) fn iteration_range(&self) -> std::ops::RangeInclusive<u64> {
        (self.config.loop_config.start_iteration + 1)..=self.config.loop_config.max_iterations
    }

    /// Execute one training iteration.
    ///
    /// Performs: forward pass, forward sync, backward pass, cut selection,
    /// template baking, lower bound evaluation, convergence update, event
    /// emission, and iteration record construction.
    ///
    /// Returns:
    /// - `Ok(Continue)` — iteration completed normally; outer loop continues.
    /// - `Ok(Converged)` — a stopping rule triggered; outer loop should break.
    /// - `Ok(Shutdown)` — external shutdown flag was observed; outer loop should break.
    /// - `Err(e)` — mid-iteration failure; caller must call `finalize_with_error(e)`.
    ///
    /// # Errors
    ///
    /// Propagates `SddpError` from forward pass, sync, backward pass, or lower
    /// bound evaluation failures.
    pub(crate) fn run_iteration(&mut self, iteration: u64) -> Result<IterationOutcome, SddpError> {
        // Check external shutdown flag before starting.
        if let Some(flag) = self.runtime.shutdown_flag.as_ref()
            && flag.load(Ordering::Relaxed)
        {
            self.convergence_monitor.set_shutdown();
        }

        let iter_start = Instant::now();

        // ── Forward pass + sync ────────────────────────────────────────────
        let (forward_result, sync_result, fwd_solve_time_ms) = self.run_forward_phase(iteration)?;

        // ── Backward pass ──────────────────────────────────────────────────
        let (backward_result, bwd_solve_time_ms) = self.run_backward_phase(iteration)?;

        // ── Cut selection + baking ─────────────────────────────────────────
        self.run_cut_management(iteration);

        // ── Lower bound evaluation ─────────────────────────────────────────
        let (lb, lb_lp_solves, lb_wall_ms, lb_solve_time_ms) = self.run_lower_bound(iteration)?;

        // ── Convergence update ─────────────────────────────────────────────
        let (should_stop, rule_results) = self.convergence_monitor.update(lb, &sync_result);

        self.results.final_lb = self.convergence_monitor.lower_bound();
        self.results.final_ub = self.convergence_monitor.upper_bound();
        self.results.final_ub_std = self.convergence_monitor.upper_bound_std();
        self.results.final_gap = self.convergence_monitor.gap();

        emit(
            self.runtime.event_sender(),
            TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound: self.results.final_lb,
                upper_bound: self.results.final_ub,
                upper_bound_std: self.convergence_monitor.upper_bound_std(),
                gap: self.results.final_gap,
                rules_evaluated: rule_results.clone(),
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        let wall_time_ms = self.results.start_time.elapsed().as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let iteration_time_ms = iter_start.elapsed().as_millis() as u64;

        emit(
            self.runtime.event_sender(),
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound: self.results.final_lb,
                upper_bound: self.results.final_ub,
                gap: self.results.final_gap,
                wall_time_ms,
                iteration_time_ms,
                forward_ms: forward_result.elapsed_ms,
                backward_ms: backward_result.elapsed_ms,
                lp_solves: forward_result.lp_solves + backward_result.lp_solves + lb_lp_solves,
                solve_time_ms: fwd_solve_time_ms + bwd_solve_time_ms + lb_solve_time_ms,
                lower_bound_eval_ms: lb_wall_ms,
                fwd_setup_time_ms: forward_result.setup_time_ms,
                fwd_load_imbalance_ms: forward_result.load_imbalance_ms,
                fwd_scheduling_overhead_ms: forward_result.scheduling_overhead_ms,
            },
        );

        self.results.completed_iterations = iteration;

        if should_stop {
            self.results.termination_reason = rule_results
                .iter()
                .find(|r| r.triggered)
                .map_or_else(|| "unknown".to_string(), |r| r.rule_name.to_string());

            // Distinguish shutdown from convergence using the triggered rule name.
            if self.results.termination_reason == RULE_GRACEFUL_SHUTDOWN {
                return Ok(IterationOutcome::Shutdown);
            }
            return Ok(IterationOutcome::Converged);
        }

        Ok(IterationOutcome::Continue)
    }

    /// Assemble and return the successful `TrainingOutcome`.
    ///
    /// Emits the `TrainingFinished` event with `termination_reason`, broadcasts
    /// the basis cache, and returns a fully-assembled `TrainingOutcome`.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Communication` if `broadcast_basis_cache` fails.
    pub(crate) fn finalize(self) -> Result<TrainingOutcome, SddpError> {
        #[allow(clippy::cast_possible_truncation)]
        let total_time_ms = (self.results.start_time.elapsed().as_millis() as u64).max(1);

        let baked_templates = self.scratch.baked_templates;
        let visited_archive = self.visited_archive;
        let TrainingResults {
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            termination_reason,
            solver_stats_log,
            ..
        } = self.results;

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::TrainingFinished {
                reason: termination_reason.clone(),
                iterations: completed_iterations,
                final_lb,
                final_ub,
                total_time_ms,
                total_rows: self.fcf.total_active_cuts() as u64,
            },
        );

        let basis_cache =
            broadcast_basis_cache(&self.basis_store, self.ranks.num_stages, self.comm)?;

        Ok(TrainingOutcome {
            result: TrainingResult::new(
                final_lb,
                final_ub,
                final_ub_std,
                final_gap,
                completed_iterations,
                termination_reason,
                total_time_ms,
                basis_cache,
                solver_stats_log,
                visited_archive,
                Some(baked_templates),
            ),
            error: None,
        })
    }

    /// Emit `TrainingFinished` with `reason = "error"` and return a partial
    /// `TrainingOutcome` carrying the original error.
    ///
    /// Replaces the inline `on_error!` macro. Consumes `self` so the accumulated
    /// state (solver stats, visited archive, baked templates) is moved — not
    /// cloned — into the outcome.
    ///
    /// # Errors
    ///
    /// Returns `Err(comm_err)` if `broadcast_basis_cache` itself fails (matching
    /// the original `on_error!` behavior at line 644).
    pub(crate) fn finalize_with_error(self, err: SddpError) -> Result<TrainingOutcome, SddpError> {
        let baked_templates = self.scratch.baked_templates;
        let visited_archive = self.visited_archive;
        let TrainingResults {
            final_lb,
            final_ub,
            final_ub_std,
            final_gap,
            completed_iterations,
            solver_stats_log,
            start_time,
            ..
        } = self.results;

        #[allow(clippy::cast_possible_truncation)]
        let total_time_ms = (start_time.elapsed().as_millis() as u64).max(1);

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::TrainingFinished {
                reason: "error".to_string(),
                iterations: completed_iterations,
                final_lb,
                final_ub,
                total_time_ms,
                #[allow(clippy::cast_possible_truncation)]
                total_rows: self.fcf.total_active_cuts() as u64,
            },
        );

        let basis_cache =
            broadcast_basis_cache(&self.basis_store, self.ranks.num_stages, self.comm)?;

        Ok(TrainingOutcome {
            result: TrainingResult::new(
                final_lb,
                final_ub,
                final_ub_std,
                final_gap,
                completed_iterations,
                "error".to_string(),
                total_time_ms,
                basis_cache,
                solver_stats_log,
                visited_archive,
                Some(baked_templates),
            ),
            error: Some(err),
        })
    }

    // ── Private phase helpers ──────────────────────────────────────────────

    /// Run the forward pass and forward synchronisation for one iteration.
    ///
    /// Returns `(forward_result, sync_result, fwd_solve_time_ms)`.
    fn run_forward_phase(
        &mut self,
        iteration: u64,
    ) -> Result<
        (
            crate::forward::ForwardResult,
            crate::forward::SyncResult,
            f64,
        ),
        SddpError,
    > {
        let fwd_stats_before = aggregate_solver_statistics(
            self.fwd_pool
                .workspaces
                .iter()
                .map(|w| w.solver.statistics()),
        );

        // Borrow fwd_state independently so the remaining fields can be
        // passed to the factory without a whole-struct borrow conflict.
        let fwd = &mut self.fwd_state;
        let mut inputs = ForwardPassInputs::from_session_fields(
            &mut self.fwd_pool,
            &mut self.basis_store,
            self.stage_ctx,
            &mut self.scratch,
            self.fcf,
            self.training_ctx,
            &self.config.cut_management,
            &self.ranks,
            &self.runtime,
            iteration,
        );
        let forward_result = fwd.run(&mut inputs)?;

        let fwd_solve_time_ms = {
            let fwd_stats_after = aggregate_solver_statistics(
                self.fwd_pool
                    .workspaces
                    .iter()
                    .map(|w| w.solver.statistics()),
            );
            SolverStatsDelta::from_snapshots(&fwd_stats_before, &fwd_stats_after).solve_time_ms
        };

        // Aggregate per-stage forward solver stats across MPI ranks before
        // logging. Without this, only rank 0's slice (its share of the
        // forward-pass workers) would be written to `iterations.parquet`,
        // because `write_training_outputs` is rank-0-gated. Backward writes
        // per-`(rank, worker_id)` rows via its own allgatherv path; forward
        // emits one row per stage with global totals so the parquet's
        // `lp_solves` column matches the true global LP count.
        //
        // `retry_level_histogram` and `basis_reconstructions` are not packed by
        // `pack_delta_scalars`; they are absent from the aggregated forward
        // rows. Backward retains the histogram via its per-worker rows.
        let num_stages = forward_result.stage_stats.len();
        let global_forward_stage_stats = if num_stages == 0 {
            Vec::new()
        } else {
            let mut local_buf = vec![0.0_f64; num_stages * SOLVER_STATS_DELTA_SCALAR_FIELDS];
            for (i, delta) in forward_result.stage_stats.iter().enumerate() {
                let packed = pack_delta_scalars(delta);
                let slot = &mut local_buf[i * SOLVER_STATS_DELTA_SCALAR_FIELDS..]
                    [..SOLVER_STATS_DELTA_SCALAR_FIELDS];
                slot.copy_from_slice(&packed);
            }
            let mut global_buf = vec![0.0_f64; local_buf.len()];
            self.comm
                .allreduce(&local_buf, &mut global_buf, ReduceOp::Sum)
                .map_err(SddpError::Communication)?;
            global_buf
                .chunks_exact(SOLVER_STATS_DELTA_SCALAR_FIELDS)
                .map(|chunk| {
                    #[allow(clippy::expect_used)]
                    let arr: [f64; SOLVER_STATS_DELTA_SCALAR_FIELDS] = chunk.try_into().expect(
                        "chunks_exact yields slices of exactly SOLVER_STATS_DELTA_SCALAR_FIELDS",
                    );
                    unpack_delta_scalars(&arr)
                })
                .collect::<Vec<SolverStatsDelta>>()
        };

        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        for (stage_idx, delta) in global_forward_stage_stats.iter().enumerate() {
            let mut entry = SolverStatsDelta::default();
            delta.clone_into_reuse(&mut entry);
            self.results.solver_stats_log.push((
                iteration,
                "forward",
                i32::try_from(stage_idx).unwrap_or(i32::MAX),
                -1,
                self.ranks.fwd_rank,
                -1,
                entry,
            ));
        }

        let local_n = forward_result.scenario_costs.len();
        let local_cost_sum: f64 = forward_result.scenario_costs.iter().sum();
        emit(
            self.runtime.event_sender(),
            TrainingEvent::ForwardPassComplete {
                iteration,
                scenarios: self.config.loop_config.forward_passes,
                #[allow(clippy::cast_precision_loss)]
                ub_mean: if local_n > 0 {
                    local_cost_sum / local_n as f64
                } else {
                    0.0
                },
                ub_std: 0.0,
                elapsed_ms: forward_result.elapsed_ms,
            },
        );

        let sync_result = sync_forward(
            &forward_result,
            self.comm,
            self.ranks.num_total_forward_passes,
        )?;

        emit(
            self.runtime.event_sender(),
            TrainingEvent::ForwardSyncComplete {
                iteration,
                global_ub_mean: sync_result.global_ub_mean,
                global_ub_std: sync_result.global_ub_std,
                sync_time_ms: sync_result.sync_time_ms,
            },
        );

        Ok((forward_result, sync_result, fwd_solve_time_ms))
    }

    /// Run the backward pass for one iteration.
    ///
    /// Returns `(backward_result, bwd_solve_time_ms)`.
    // RATIONALE: `i32::try_from(*omega).expect(...)` is safe because opening
    // indices derive from `branching_factor: u16` and never approach
    // `i32::MAX`. The expect documents an invariant of the study tree, not
    // a recoverable runtime error.
    #[allow(clippy::expect_used)]
    fn run_backward_phase(
        &mut self,
        iteration: u64,
    ) -> Result<(crate::backward::BackwardResult, f64), SddpError> {
        // Borrow bwd_state independently so the remaining fields can be
        // passed to the factory without a whole-struct borrow conflict.
        // `IterationScratch` is split into named sub-field borrows so the
        // disjoint `cut_selection_scratch` slot can be co-borrowed mutably
        // alongside `cut_batches` (each is a distinct field of `self.scratch`).
        let bwd = &mut self.bwd_state;
        let mut inputs = BackwardPassInputs::from_session_fields(
            &mut self.fwd_pool,
            &mut self.basis_store,
            self.stage_ctx,
            &mut self.scratch.baked_templates,
            &mut self.scratch.cut_batches,
            &self.scratch.records,
            &mut self.scratch.cut_selection_scratch,
            self.fcf,
            &mut self.exchange_bufs,
            &mut self.cut_sync_bufs,
            &mut self.visited_archive,
            self.training_ctx,
            self.comm,
            &self.config.cut_management,
            &self.ranks,
            &self.runtime,
            iteration,
        );
        let backward_result = bwd.run(&mut inputs)?;
        // `selection_records` length is bounded by `num_stages - 1` (one entry
        // per stage where the in-backward hook ran). Reading the length here
        // also keeps the field visible to the dead-code lint until the
        // rerouting consumer lands.
        debug_assert!(
            backward_result.selection_records.len() <= self.ranks.num_stages.saturating_sub(1),
            "selection_records length must not exceed num_stages - 1"
        );

        let bwd_solve_time_ms = {
            let agg = SolverStatsDelta::aggregate(
                backward_result
                    .stage_stats
                    .iter()
                    .flat_map(|(_, entries)| entries.iter().map(|(_, _, _, d)| d)),
            );
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            for (stage_idx, entries) in &backward_result.stage_stats {
                for (rank, worker_id, omega, delta) in entries {
                    let mut entry = SolverStatsDelta::default();
                    delta.clone_into_reuse(&mut entry);
                    self.results.solver_stats_log.push((
                        iteration,
                        "backward",
                        *stage_idx as i32,
                        i32::try_from(*omega)
                            .expect("opening index is bounded well below i32::MAX"),
                        *rank,
                        *worker_id,
                        entry,
                    ));
                }
            }
            agg.solve_time_ms
        };

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::BackwardPassComplete {
                iteration,
                rows_generated: backward_result.cuts_generated as u32,
                stages_processed: self.ranks.num_stages.saturating_sub(1) as u32,
                elapsed_ms: backward_result.elapsed_ms,
                state_exchange_time_ms: backward_result.state_exchange_time_ms,
                row_batch_build_time_ms: backward_result.cut_batch_build_time_ms,
                setup_time_ms: backward_result.setup_time_ms,
                load_imbalance_ms: backward_result.load_imbalance_ms,
                scheduling_overhead_ms: backward_result.scheduling_overhead_ms,
            },
        );

        #[allow(clippy::cast_possible_truncation)]
        emit(
            self.runtime.event_sender(),
            TrainingEvent::PolicySyncComplete {
                iteration,
                rows_distributed: backward_result.cuts_generated as u32,
                rows_active: self.fcf.total_active_cuts() as u32,
                rows_removed: 0,
                sync_time_ms: backward_result.cut_sync_time_ms,
            },
        );

        Ok((backward_result, bwd_solve_time_ms))
    }

    /// Apply cut selection, budget enforcement, bitmap shift, and template baking.
    ///
    /// All operations are O(active cuts) and perform no heap allocation when the
    /// cut pools have not grown since the previous iteration.
    // RATIONALE: run_cut_management sequences 5 interleaved mutation phases on
    // &mut self (selection, bitmap shift, budget cap, template bake, statistics).
    // Each phase reads state written by the prior phase; splitting into helpers
    // would require passing all 5 fields individually as &mut refs under NLL.
    #[allow(clippy::too_many_lines)]
    fn run_cut_management(&mut self, iteration: u64) {
        // Step 4a: Strategy-based cut selection.
        // sel_state holds (per_stage, rows_deactivated, selection_time_ms,
        // stages_processed) when step 4a ran; None otherwise.
        let mut sel_state: Option<(Vec<StageRowSelectionRecord>, u32, u64, u32)> = None;

        if let Some(strategy) = self.config.cut_management.cut_selection.as_ref()
            && strategy.should_run(iteration)
        {
            let sel_start = Instant::now();
            let num_sel_stages = self.ranks.num_stages.saturating_sub(1);
            let mut rows_deactivated = 0u32;
            let mut per_stage = Vec::with_capacity(num_sel_stages);

            // Stage 0 is exempt: its cuts are never the "successor" in the
            // backward pass, so their binding activity is never updated.
            // Stage T-1 (the terminal stage) is also exempt: the backward
            // pass sweeps T-2 down to 0 (see `backward.rs` module docs),
            // so the terminal-stage pool receives no cuts to select among.
            // `num_sel_stages = num_stages - 1` (via the saturating_sub
            // above) and the parallel loop ranges over `1..num_sel_stages`,
            // which expands to stages 1..=T-2 — naturally excluding
            // stage T-1 from selection.
            #[allow(clippy::cast_possible_truncation)]
            {
                let pool0 = &self.fcf.pools[0];
                let active_0 = pool0.active_count() as u32;
                per_stage.push(StageRowSelectionRecord {
                    stage: 0,
                    rows_populated: pool0.populated_count as u32,
                    rows_active_before: active_0,
                    rows_deactivated: 0,
                    rows_reactivated: 0,
                    rows_active_after: active_0,
                    selection_time_ms: 0.0,
                    budget_evicted: None,
                    active_after_budget: None,
                    rows_in_lp: pool0.cuts_in_lp() as u32,
                });
            }

            let archive_ref = self.visited_archive.as_ref();
            // Sequential selection over interior stages 1..=T-2. Stage 0 was
            // recorded above; stage T-1 (terminal) is skipped because the
            // backward pass produces no cuts at stage T-1, so the pool is
            // empty and selection would be a no-op.
            //
            // The outer loop is sequential (not `into_par_iter`) because
            // the new m-block kernel uses fold-local v_block allocations
            // and shares the same `worker_scratch` slice across all stages;
            // concurrent stages would race for the same scratch slots.
            // Design §4.2 confirms the 24-48 m-block tasks per stage
            // saturate a 96-core machine via the inner parallelism alone.
            let pools = &self.fcf.pools;
            let worker_scratch = self.scratch.cut_selection_scratch_mut();
            #[allow(clippy::cast_possible_truncation)]
            let deactivations: Vec<(usize, CutActivityUpdates, f64)> = (1..num_sel_stages)
                .map(|stage| {
                    let pool = &pools[stage];
                    let states = archive_ref.map_or(&[] as &[f64], |a| a.states_for_stage(stage));
                    let start = Instant::now();
                    let deact = strategy.select_for_stage_with_scratch(
                        pool,
                        states,
                        iteration,
                        stage as u32,
                        worker_scratch,
                    );
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    (stage, deact, elapsed_ms)
                })
                .collect();

            #[allow(clippy::cast_possible_truncation)]
            for (stage, deact, stage_sel_time_ms) in deactivations {
                let pool = &self.fcf.pools[stage];
                let populated = pool.populated_count as u32;
                let active_before = pool.active_count() as u32;
                let n_deact = deact.updates.len() as u32;
                let n_reactivated = deact.reactivations.len() as u32;
                rows_deactivated += n_deact;

                // Apply both deactivations and reactivations from the
                // unified kernel output.
                self.fcf.pools[stage].apply_updates(&deact);

                let active_after = self.fcf.pools[stage].active_count() as u32;
                let rows_in_lp = self.fcf.pools[stage].cuts_in_lp() as u32;
                per_stage.push(StageRowSelectionRecord {
                    stage: stage as u32,
                    rows_populated: populated,
                    rows_active_before: active_before,
                    rows_deactivated: n_deact,
                    rows_reactivated: n_reactivated,
                    rows_active_after: active_after,
                    selection_time_ms: stage_sel_time_ms,
                    budget_evicted: None,
                    active_after_budget: None,
                    rows_in_lp,
                });
            }

            // Trim the visited-states archive to the active strategy's
            // sliding window now that selection has consumed the current
            // contents. Done AFTER the per-stage application loop so the
            // immutable borrow held by `deactivations` is fully released.
            //
            // Trim runs AFTER selection so the kernel evaluates against
            // the full accumulated archive (up to `2 * check_frequency`
            // iterations of states); we then shrink back to a
            // steady-state bound of `check_frequency` iterations.
            // Selection benefits from seeing more visited states than the
            // post-trim window — better selection quality — at the cost
            // of a temporary ~2x memory peak just before this trim runs.
            // The bound documented on
            // [`VisitedStatesArchive::trim_to_window`] is the
            // post-trim (steady-state) size.
            if let Some(ref mut archive) = self.visited_archive {
                let check_freq = match strategy {
                    crate::cut_selection::CutSelectionStrategy::Level1 {
                        check_frequency, ..
                    }
                    | crate::cut_selection::CutSelectionStrategy::Lml1 {
                        check_frequency, ..
                    }
                    | crate::cut_selection::CutSelectionStrategy::Dominated {
                        check_frequency,
                        ..
                    } => *check_frequency,
                };
                archive.trim_to_window(check_freq);
            }

            #[allow(clippy::cast_possible_truncation)]
            let selection_time_ms = sel_start.elapsed().as_millis() as u64;
            #[allow(clippy::cast_possible_truncation)]
            let stages_processed_sel = num_sel_stages as u32;

            sel_state = Some((
                per_stage,
                rows_deactivated,
                selection_time_ms,
                stages_processed_sel,
            ));
        }

        // Step 4b: Budget enforcement (every iteration when budget is set).
        //
        // Runs unconditionally when `budget` is Some — not gated by
        // `check_frequency`. The budget is a hard cap that must be maintained
        // at all times.
        if let Some(b) = self.config.cut_management.budget {
            let budget_start = Instant::now();
            let mut total_evicted = 0u32;
            for stage in 0..self.ranks.num_stages {
                #[allow(clippy::cast_possible_truncation)]
                let result = self.fcf.pools[stage].enforce_budget(
                    b,
                    iteration,
                    self.config.loop_config.forward_passes,
                );
                total_evicted += result.evicted_count;
                // Annotate per-stage records with post-budget counts.
                if let Some((ref mut per_stage, _, _, _)) = sel_state
                    && let Some(rec) = per_stage.get_mut(stage)
                {
                    rec.budget_evicted = Some(result.evicted_count);
                    rec.active_after_budget = Some(result.active_after);
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            let enforcement_time_ms = budget_start.elapsed().as_millis() as u64;
            emit(
                self.runtime.event_sender(),
                #[allow(clippy::cast_possible_truncation)]
                TrainingEvent::PolicyBudgetEnforcementComplete {
                    iteration,
                    rows_evicted: total_evicted,
                    stages_processed: self.ranks.num_stages as u32,
                    enforcement_time_ms,
                },
            );
        }

        // Emit PolicySelectionComplete now that all per-stage annotation is done.
        if let Some((per_stage, rows_deactivated, selection_time_ms, stages_processed)) = sel_state
        {
            emit(
                self.runtime.event_sender(),
                TrainingEvent::PolicySelectionComplete {
                    iteration,
                    rows_deactivated,
                    stages_processed,
                    selection_time_ms,
                    allgatherv_time_ms: 0,
                    per_stage,
                },
            );
        }

        // Shift the sliding-window binding bitmap left by 1 on every populated
        // cut slot. This ages the activity record so the next iteration's bit 0
        // starts clear. Placed AFTER cut selection and BEFORE template baking.
        for pool in &mut self.fcf.pools {
            for m in pool.metadata.iter_mut().take(pool.populated_count) {
                m.active_window <<= 1;
            }
        }

        // Step 4c: Template baking.
        // Rebuild per-stage baked templates from the current active cut set.
        let bake_start = Instant::now();
        let mut total_rows_baked: u64 = 0;
        let indexer = self.training_ctx.indexer;
        for t in 0..self.ranks.num_stages {
            build_cut_row_batch_into(
                &mut self.scratch.bake_row_batches[t],
                self.fcf,
                t,
                indexer,
                &self.stage_ctx.templates[t].col_scale,
            );
            #[allow(clippy::cast_possible_truncation)]
            {
                total_rows_baked += self.scratch.bake_row_batches[t].num_rows as u64;
            }
            cobre_solver::bake_rows_into_template(
                &self.stage_ctx.templates[t],
                &self.scratch.bake_row_batches[t],
                &mut self.scratch.baked_templates[t],
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let bake_time_ms = bake_start.elapsed().as_millis() as u64;
        emit(
            self.runtime.event_sender(),
            #[allow(clippy::cast_possible_truncation)]
            TrainingEvent::PolicyTemplateBakeComplete {
                iteration,
                stages_processed: self.ranks.num_stages as u32,
                total_rows_baked,
                bake_time_ms,
            },
        );
    }

    /// Evaluate the lower bound and push the solver stats entry.
    ///
    /// Returns `(lb_value, lb_lp_solves, lb_wall_ms, lb_solve_time_ms)`.
    fn run_lower_bound(&mut self, iteration: u64) -> Result<(f64, u64, u64, f64), SddpError> {
        let lb_wall_start = Instant::now();
        let lb_stats_before = self.solver.statistics();

        let lb_spec = LbEvalSpec {
            template: &self.stage_ctx.templates[0],
            base_row: self.stage_ctx.base_rows[0],
            noise_scale: self.stage_ctx.noise_scale,
            n_hydros: self.stage_ctx.n_hydros,
            opening_tree: self.training_ctx.stochastic.opening_tree(),
            risk_measure: &self.config.cut_management.risk_measures[0],
            stochastic: Some(self.training_ctx.stochastic),
            n_load_buses: self.stage_ctx.n_load_buses,
            ncs_max_gen: self.stage_ctx.ncs_max_gen,
            ncs_allow_curtailment: self.stage_ctx.ncs_allow_curtailment,
            block_count: self.stage_ctx.block_counts_per_stage[0],
            ncs_generation: self.training_ctx.indexer.ncs_generation.clone(),
            inflow_method: self.training_ctx.inflow_method,
        };
        let mut lb_bundle = LbEvalScratchBundle::from_scratch_fields(
            &mut self.scratch.patch_buf,
            &mut self.scratch.lb_cut_batch,
            Some(&mut self.scratch.lb_cut_row_map),
            &mut self.scratch.lb_scratch,
        );
        let lb = evaluate_lower_bound(
            self.solver,
            self.fcf,
            self.training_ctx.initial_state,
            self.training_ctx.indexer,
            &mut lb_bundle,
            &lb_spec,
            self.comm,
        )?;

        let lb_stats_after = self.solver.statistics();
        let lb_lp_solves = lb_stats_after.solve_count - lb_stats_before.solve_count;
        let lb_delta = SolverStatsDelta::from_snapshots(&lb_stats_before, &lb_stats_after);
        let lb_solve_time_ms = lb_delta.solve_time_ms;
        self.results.solver_stats_log.push((
            iteration,
            "lower_bound",
            -1,
            -1,
            self.ranks.fwd_rank,
            -1,
            lb_delta,
        ));
        #[allow(clippy::cast_possible_truncation)]
        let lb_wall_ms = lb_wall_start.elapsed().as_millis() as u64;

        Ok((lb, lb_lp_solves, lb_wall_ms, lb_solve_time_ms))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use chrono::NaiveDate;
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::{
        Bus, EntityId, SystemBuilder, TrainingEvent, WorkerTimingPhase,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::{
        ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
    };

    use super::{IterationOutcome, TrainingSession};
    use crate::{
        StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
        config::{CutManagementConfig, EventConfig, LoopConfig},
        context::{StageContext, TrainingContext},
        cut::fcf::FutureCostFunction,
        error::SddpError,
        horizon_mode::HorizonMode,
        indexer::StageIndexer,
        inflow_method::InflowNonNegativityMethod,
        risk_measure::RiskMeasure,
    };

    // ── Shared helpers (mirrors training.rs test helpers) ──────────────────

    fn minimal_template(_n_state: usize) -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    struct MockSolver {
        objectives: Vec<f64>,
        call_count: usize,
    }

    impl MockSolver {
        fn with_fixed(objective: f64) -> Self {
            Self {
                objectives: vec![objective],
                call_count: 0,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::HighsProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::HighsProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            let obj = self.objectives[call % self.objectives.len()];
            Ok(cobre_solver::SolutionView {
                objective: obj,
                primal: &[0.0, 0.0, 0.0, 0.0],
                dual: &[0.0, 0.0],
                reduced_costs: &[0.0, 0.0, 0.0, 0.0],
                iterations: 0,
                solve_time_seconds: 0.0,
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

    struct StubComm;

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            recv[..send.len()].clone_from_slice(send);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            recv.clone_from_slice(send);
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

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    fn make_stochastic_context(n_stages: usize, n_openings: usize) -> StochasticContext {
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![cobre_core::DeficitSegment {
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
                branching_factor: n_openings,
                noise_method: NoiseMethod::Saa,
            },
        };

        let stages: Vec<Stage> = (0..n_stages).map(make_stage).collect();

        let inflow_models: Vec<_> = (0..n_stages)
            .map(|i| cobre_core::scenario::InflowModel {
                hydro_id: EntityId(1),
                stage_id: i as i32,
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

    fn make_stages(n_stages: usize) -> Vec<Stage> {
        (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: cobre_core::temporal::StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: cobre_core::temporal::StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect()
    }

    fn make_fcf(
        n_stages: usize,
        n_state: usize,
        forward_passes: u32,
        max_iter: u64,
    ) -> FutureCostFunction {
        FutureCostFunction::new(
            n_stages,
            n_state,
            forward_passes,
            max_iter,
            &vec![0; n_stages],
        )
    }

    fn iteration_limit_rules(limit: u64) -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit }],
            mode: StoppingMode::Any,
        }
    }

    fn make_config(
        forward_passes: u32,
        max_iterations: u64,
        limit: u64,
        n_stages: usize,
    ) -> TrainingConfig {
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes,
                max_iterations,
                start_iteration: 0,
                n_fwd_threads: 1,
                max_blocks: 1,
                stopping_rules: iteration_limit_rules(limit),
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
                warm_start_cuts: 0,
                risk_measures: vec![RiskMeasure::Expectation; n_stages],
            },
            events: EventConfig {
                event_sender: None,
                checkpoint_interval: None,
                shutdown_flag: None,
                export_states: false,
            },
        }
    }

    fn make_stage_ctx<'a>(
        templates: &'a [StageTemplate],
        base_rows: &'a [usize],
        block_counts: &'a [usize],
    ) -> StageContext<'a> {
        StageContext {
            templates,
            base_rows,
            noise_scale: &[],
            n_hydros: 0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: block_counts,
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        }
    }

    fn make_training_ctx<'a>(
        horizon: &'a HorizonMode,
        indexer: &'a StageIndexer,
        stochastic: &'a StochasticContext,
        initial_state: &'a [f64],
        stages: &'a [Stage],
    ) -> TrainingContext<'a> {
        TrainingContext {
            horizon,
            indexer,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic,
            initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            recent_accum_seed: &[],
            recent_weight_seed: 0.0,
        }
    }

    // ── Test: training_session_new_preallocates_all_buffers ────────────────

    /// Verify that `TrainingSession::new` pre-allocates all scratch buffers to
    /// their expected sizes before the first iteration.
    #[test]
    fn training_session_new_preallocates_all_buffers() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);
        let config = make_config(1, 10, 1, n_stages);
        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        // forward_passes=1, num_ranks=1 → max_local_fwd=1
        let max_local_fwd = 1usize;
        assert_eq!(
            session.scratch.records.len(),
            max_local_fwd * n_stages,
            "records must be pre-sized to max_local_fwd * num_stages"
        );
        assert_eq!(
            session.scratch.cut_batches.len(),
            n_stages,
            "cut_batches must have one RowBatch per stage"
        );
        assert_eq!(
            session.scratch.baked_templates.len(),
            n_stages,
            "baked_templates must have one per stage"
        );
        // send_stride = n_workers_local * bwd_max_openings * WORKER_STATS_ENTRY_STRIDE
        // n_fwd_threads=1 → n_workers_local=1; max_openings=1 for this fixture
        let expected_send_stride = crate::solver_stats::WORKER_STATS_ENTRY_STRIDE;
        assert_eq!(
            session.bwd_state.bwd_stats_send_buf.len(),
            expected_send_stride,
            "bwd_stats_send_buf must equal send_stride"
        );
    }

    // ── Test: training_session_finalize_emits_training_finished ───────────

    /// Verify that `finalize()` emits exactly one `TrainingFinished` event.
    #[test]
    fn training_session_finalize_emits_training_finished() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 1, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        // finalize without running any iterations
        let outcome = session.finalize().unwrap();

        assert!(outcome.error.is_none(), "no error expected");
        assert_eq!(outcome.result.iterations, 0);

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        let last = events.last().unwrap();
        assert!(
            matches!(last, TrainingEvent::TrainingFinished { iterations: 0, .. }),
            "last event must be TrainingFinished with iterations=0, got: {last:?}"
        );
    }

    // ── Test: training_session_finalize_with_error_emits_training_finished_with_error_reason ──

    /// Verify that `finalize_with_error` emits `TrainingFinished` with `reason = "error"`.
    #[test]
    fn training_session_finalize_with_error_emits_training_finished_with_error_reason() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 1, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let outcome = session
            .finalize_with_error(SddpError::Validation("test error".to_string()))
            .unwrap();

        assert!(outcome.error.is_some(), "expected error in outcome");
        assert_eq!(outcome.result.reason, "error");

        let events: Vec<TrainingEvent> = rx.try_iter().collect();
        // Events: TrainingStarted + TrainingFinished(reason="error")
        let last = events.last().unwrap();
        assert!(
            matches!(last, TrainingEvent::TrainingFinished { .. }),
            "last event must be TrainingFinished"
        );
        if let TrainingEvent::TrainingFinished { reason, .. } = last {
            assert_eq!(reason, "error");
        }
    }

    // ── Test: training_session_run_iteration_returns_continue_when_not_converged ──

    /// Verify that `run_iteration` returns `Continue` when stopping rules have not triggered.
    #[test]
    fn training_session_run_iteration_returns_continue_when_not_converged() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);
        // iteration_limit=5, so iteration 1 should return Continue
        let config = make_config(1, 10, 5, n_stages);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let result = session.run_iteration(1).unwrap();
        assert!(
            matches!(result, IterationOutcome::Continue),
            "expected Continue when limit is 5, got: {result:?}"
        );
    }

    // ── Test: training_session_run_iteration_returns_converged_when_gap_closes ──

    /// Verify that `run_iteration` eventually returns `Converged` when a stopping
    /// rule triggers.
    #[test]
    fn training_session_run_iteration_returns_converged_when_gap_closes() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);
        // iteration_limit=1 so the first iteration triggers convergence
        let config = make_config(1, 10, 1, n_stages);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        let mut last_outcome = IterationOutcome::Continue;
        for iter in session.iteration_range() {
            last_outcome = session.run_iteration(iter).unwrap();
            if !matches!(last_outcome, IterationOutcome::Continue) {
                break;
            }
        }
        assert!(
            matches!(last_outcome, IterationOutcome::Converged),
            "expected Converged after iteration limit triggers, got: {last_outcome:?}"
        );
    }

    // ── Test: training_session_run_iteration_emits_correct_event_sequence ───

    /// Verify that one call to `run_iteration(1)` followed by `finalize()` emits
    /// exactly 11 events in the correct order:
    ///
    /// 1  × `TrainingStarted`   (emitted by `new`)
    /// 9  × per-iteration events (emitted by `run_iteration`):
    ///        `WorkerTiming(Forward)`, `ForwardPassComplete`, `ForwardSyncComplete`,
    ///        `WorkerTiming(Backward)`, `BackwardPassComplete`, `PolicySyncComplete`,
    ///        `PolicyTemplateBakeComplete`, `ConvergenceUpdate`, `IterationSummary`
    /// 1  × `TrainingFinished`  (emitted by `finalize`)
    ///
    /// Total = 11 events.
    #[test]
    fn training_session_run_iteration_emits_correct_event_sequence() {
        let n_stages = 2;
        let indexer = StageIndexer::new(1, 0);
        let templates = vec![minimal_template(indexer.n_state); n_stages];
        let base_rows = vec![2usize; n_stages];
        let initial_state = vec![0.0_f64; indexer.n_state];
        let stochastic = make_stochastic_context(n_stages, 1);
        let stages = make_stages(n_stages);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let mut fcf = make_fcf(n_stages, indexer.n_state, 1, 10);

        let (tx, rx) = mpsc::channel::<TrainingEvent>();
        let mut config = make_config(1, 10, 10, n_stages);
        config.events.event_sender = Some(tx);

        let mut solver = MockSolver::with_fixed(100.0);
        let comm = StubComm;
        let block_counts = vec![1usize; n_stages];
        let stage_ctx = make_stage_ctx(&templates, &base_rows, &block_counts);
        let training_ctx =
            make_training_ctx(&horizon, &indexer, &stochastic, &initial_state, &stages);

        let mut session = TrainingSession::new(
            &mut solver,
            config,
            &mut fcf,
            &stage_ctx,
            &training_ctx,
            &comm,
            || Ok(MockSolver::with_fixed(100.0)),
        )
        .unwrap();

        // Run exactly one iteration (does not trigger the stopping rule since limit=10).
        let outcome = session.run_iteration(1).unwrap();
        assert!(
            matches!(outcome, IterationOutcome::Continue),
            "expected Continue for iteration 1 with limit=10, got: {outcome:?}"
        );

        // Finalize emits TrainingFinished.
        session.finalize().unwrap();

        let events: Vec<TrainingEvent> = rx.try_iter().collect();

        // ── Count assertion ────────────────────────────────────────────────
        // 1 (TrainingStarted) + 9 (per-iteration) + 1 (TrainingFinished) = 11
        assert_eq!(
            events.len(),
            11,
            "expected 11 events for 1 iteration, got {} ({events:?})",
            events.len()
        );

        // ── Order assertion ────────────────────────────────────────────────
        assert!(
            matches!(events[0], TrainingEvent::TrainingStarted { .. }),
            "events[0] must be TrainingStarted, got: {:?}",
            events[0]
        );

        // Per-iteration block (events[1..=9])
        assert!(
            matches!(
                events[1],
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Forward,
                    ..
                }
            ),
            "events[1] must be WorkerTiming(Forward), got: {:?}",
            events[1]
        );
        assert!(
            matches!(events[2], TrainingEvent::ForwardPassComplete { .. }),
            "events[2] must be ForwardPassComplete, got: {:?}",
            events[2]
        );
        assert!(
            matches!(events[3], TrainingEvent::ForwardSyncComplete { .. }),
            "events[3] must be ForwardSyncComplete, got: {:?}",
            events[3]
        );
        assert!(
            matches!(
                events[4],
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Backward,
                    ..
                }
            ),
            "events[4] must be WorkerTiming(Backward), got: {:?}",
            events[4]
        );
        assert!(
            matches!(events[5], TrainingEvent::BackwardPassComplete { .. }),
            "events[5] must be BackwardPassComplete, got: {:?}",
            events[5]
        );
        assert!(
            matches!(events[6], TrainingEvent::PolicySyncComplete { .. }),
            "events[6] must be PolicySyncComplete, got: {:?}",
            events[6]
        );
        assert!(
            matches!(events[7], TrainingEvent::PolicyTemplateBakeComplete { .. }),
            "events[7] must be PolicyTemplateBakeComplete, got: {:?}",
            events[7]
        );
        assert!(
            matches!(events[8], TrainingEvent::ConvergenceUpdate { .. }),
            "events[8] must be ConvergenceUpdate, got: {:?}",
            events[8]
        );
        assert!(
            matches!(events[9], TrainingEvent::IterationSummary { .. }),
            "events[9] must be IterationSummary, got: {:?}",
            events[9]
        );

        assert!(
            matches!(events[10], TrainingEvent::TrainingFinished { .. }),
            "events[10] must be TrainingFinished, got: {:?}",
            events[10]
        );
    }
}
