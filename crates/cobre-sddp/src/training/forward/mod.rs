//! Forward pass execution for SDDP training.
//!
//! [`run_forward_pass`] simulates scenario trajectories via stage LPs with the
//! current Future Cost Function. Outputs [`TrajectoryRecord`]s and [`ForwardResult`]
//! for the backward pass and synchronisation step. Parallelised across workers
//! with deterministic scenario assignment.
//!
//! The per-scenario hot loop lives in
//! `forward_pass_state::ForwardPassState::run`. All scratch is owned by the
//! workspace and reused across scenarios; `run_forward_stage` reuses the
//! `ws.scratch.unscaled_primal` buffer via `mem::take`/restore (taken out before
//! the solve so it can be filled while the `view` borrow of `ws` is live, then
//! restored after the last read), the same pattern as `simulation/pipeline.rs`.
//!
//! ## Submodule layout
//!
//! - [`sync_forward`] (`stats_aggregation`) — cross-rank `allgatherv` plus the
//!   canonical-order Welford summation for rank-count-invariant upper-bound stats.
//! - [`build_delta_cut_row_batch_into`] (`delta_cut_batch`) — delta-cut `RowBatch`
//!   construction for baked-template appends.
//! - `write_capture_metadata` (`basis_capture`) — captured-basis metadata after a
//!   stage solve.
//! - `run_forward_stage` (`stage_solve`) — the per-(scenario, stage) LP-solve
//!   kernel carrying the hot-path scratch reuse.
//! - [`build_sampler_from_ctx`] (`sampler`) — `ForwardSampler` construction.
//!
//! This `mod.rs` owns the result structs ([`ForwardResult`], [`SyncResult`]), the
//! per-call parameter bundles ([`ForwardPassBatch`], `StageKey`), and the thin
//! [`run_forward_pass`] shim that delegates to `ForwardPassState::run`. The
//! `pub use` block below re-exports every submodule item so the
//! `crate::forward::`, `cobre_sddp::forward::`, and `training::forward::` paths
//! resolve verbatim.

use std::sync::mpsc::Sender;

use cobre_core::TrainingEvent;
use cobre_solver::{SolverInterface, StageTemplate};

use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    cut::pool::CutPool,
    dcs::DcsParams,
    error::SddpError,
    solver_stats::SolverStatsDelta,
    trajectory::TrajectoryRecord,
    workspace::{BasisStore, SolverWorkspace},
};

mod basis_capture;
mod delta_cut_batch;
mod sampler;
mod stage_solve;
mod stats_aggregation;

#[cfg(test)]
mod tests;

pub use delta_cut_batch::build_delta_cut_row_batch_into;
pub use sampler::build_sampler_from_ctx;
pub use stats_aggregation::sync_forward;

// `pub(crate)` re-exports match the items' own visibility: `write_capture_metadata`
// and `run_forward_stage` are crate-internal, reached through `crate::forward::`
// (and `super::`) by `training/backward.rs`, `training/forward_pass_state.rs`, and
// the in-file DCS tests. A `pub` re-export would be a private-in-public error.
pub(crate) use basis_capture::write_capture_metadata;
pub(crate) use stage_solve::run_forward_stage;

/// Local statistics from one rank's forward pass.
///
/// Carries the individual per-scenario trajectory costs in global scenario
/// index order (scenario 0 first, scenario N-1 last). The synchronisation
/// step gathers these costs from all ranks via `allgatherv` and performs
/// canonical-order summation to produce bit-identical statistics regardless
/// of the number of MPI ranks or intra-rank worker threads.
///
/// Does not contain lower bound estimate (evaluated separately after backward pass).
#[derive(Debug, Clone)]
#[must_use]
pub struct ForwardResult {
    /// Per-scenario trajectory costs in global scenario index order.
    ///
    /// Length equals the number of scenarios solved on this rank.
    /// Scenario `m` (local index) appears at position `m`.
    pub scenario_costs: Vec<f64>,

    /// Wall-clock time in milliseconds for this rank's forward pass.
    pub elapsed_ms: u64,

    /// Number of LP solves performed during this forward pass.
    pub lp_solves: u64,

    /// Aggregate non-solve work inside the parallel region summed across all
    /// workers, in milliseconds.
    ///
    /// Computed as the sum across all workers of
    /// `load_model_time_ms + set_bounds_time_ms + basis_set_time_ms`.
    pub setup_time_ms: u64,

    /// Load-imbalance component of parallel overhead, in milliseconds.
    ///
    /// Computed as `max_worker_total_ms - avg_worker_total_ms`, where
    /// `worker_total_ms = solve + load_model + set_bounds + basis_set`
    /// for each worker.  Measures how much the slowest worker exceeds the average.
    pub load_imbalance_ms: u64,

    /// True rayon scheduling overhead, in milliseconds.
    ///
    /// Computed as `parallel_wall_ms - max_worker_total_ms`, clamped to zero.
    /// Represents rayon barrier, thread wake-up, and work-stealing dispatch costs
    /// after accounting for all measured per-worker work.
    pub scheduling_overhead_ms: u64,

    /// Per-stage solver-counter deltas, summed across all workers and
    /// scenarios this rank processed. Length equals the number of
    /// stages in the study. Element `t` is the rank-local sum over
    /// `(scenario × this_rank)` of the per-stage `SolverStatsDelta`
    /// produced by `run_stage_solve` for stage `t`.
    pub stage_stats: Vec<SolverStatsDelta>,
}

/// Global upper bound statistics from forward synchronisation step.
#[derive(Debug, Clone)]
#[must_use]
pub struct SyncResult {
    /// Sample mean of total trajectory costs across all ranks.
    pub global_ub_mean: f64,

    /// Bessel-corrected sample standard deviation of total trajectory costs.
    pub global_ub_std: f64,

    /// 95% confidence interval half-width: `1.96 * std / sqrt(N)`.
    pub ci_95_half_width: f64,

    /// Wall-clock time in milliseconds for the forward synchronization call.
    pub sync_time_ms: u64,
}

/// Bundled scalar parameters for one forward pass invocation.
///
/// Groups the per-iteration, per-rank scalar arguments that are forwarded
/// from [`crate::train`] into [`run_forward_pass`].
pub struct ForwardPassBatch<'a> {
    /// Number of forward-pass scenarios assigned to this rank.
    pub local_forward_passes: usize,
    /// Total forward passes across all MPI ranks. Used for LHS stratification
    /// in the sampler (`total_scenarios` field of `SampleRequest`) and for
    /// sizing the LHS permutation scratch buffer. Must equal the study-level
    /// `forward_passes` parameter, NOT the per-rank local count.
    pub total_forward_passes: usize,
    /// Current training iteration (0-based; used for seed derivation).
    pub iteration: u64,
    /// Global index of this rank's first forward pass for seed derivation.
    pub fwd_offset: usize,
    /// Optional channel for emitting [`TrainingEvent::WorkerTiming`] events.
    ///
    /// When `Some`, one event is emitted per rayon worker after the parallel
    /// region completes, carrying the accumulated wall and setup timings for
    /// the forward phase. When `None` (the default for tests), no events are
    /// emitted.
    pub event_sender: Option<&'a Sender<TrainingEvent>>,
}

/// Per-stage solve context for one (stage, scenario) pair in the forward pass.
///
/// Passed to [`run_forward_stage`] to bundle scalar and slice parameters and
/// keep the argument count within the clippy `too_many_arguments` threshold.
pub(crate) struct StageKey<'a> {
    /// 0-based stage index.
    pub(crate) t: usize,
    /// 0-based global scenario index (rank offset + local scenario index).
    pub(crate) m: usize,
    /// Local scenario index within this worker's partition.
    pub(crate) local_m: usize,
    /// Total number of stages in the horizon.
    pub(crate) num_stages: usize,
    /// Current training iteration (used in error context).
    pub(crate) iteration: u64,
    /// Raw noise sample for this (stage, scenario) pair.
    pub(crate) raw_noise: &'a [f64],
    /// Total LP row count, equal to `baked[t].num_rows`
    /// (the baked template absorbs all active cut rows as structural
    /// rows). Used to pre-allocate basis storage and avoid
    /// per-scenario heap reallocation. Consumed by
    /// `run_forward_stage`, the captured-basis metadata computation, and
    /// `CapturedBasis::new`.
    pub(crate) basis_row_capacity: usize,
    /// True when the last study stage (`T-1`) has at least one warm-start
    /// (boundary) cut.  When true, the theta column at the terminal stage is
    /// NOT zeroed out so that the boundary cuts can contribute to the LP
    /// objective.  Computed once per forward pass from
    /// `fcf.pools[num_stages - 1].warm_start_count > 0` and reused for every
    /// (scenario, stage) pair in the pass.
    pub(crate) terminal_has_boundary_cuts: bool,
    /// Reference to the cut pool for stage `t`. Used by
    /// `write_capture_metadata` to record slot identities for the next
    /// iteration's warm-start, and forwarded into `StageInputs`.
    pub(crate) pool: &'a CutPool,
    /// Dynamic Cut Selection hyperparameters for this stage solve, `Some` only
    /// when the dynamic method is configured AND active at this iteration
    /// (`iteration >= start_iteration`). When `Some`, the stage is solved
    /// lazily against the cut pool from the cut-free base template; when `None`,
    /// the baked all-cuts path is used unchanged.
    pub(crate) dcs: Option<DcsParams>,
}

/// Execute the forward pass for one training iteration on this rank.
///
/// Simulates this rank's share of forward-pass scenarios through the full
/// stage horizon, solving the stage LP at each `(scenario, stage)` pair.
/// Pre-allocated [`TrajectoryRecord`]s in `records` are populated in-place.
///
/// ## Argument layout
///
/// - `workspaces` — one [`SolverWorkspace`] per worker thread. Scenarios are
///   statically partitioned across workspaces; each workspace owns its solver,
///   patch buffer, and current-state buffer exclusively.
/// - `basis_store` — per-scenario, per-stage basis store pre-allocated by the
///   caller. The store is split into disjoint sub-views before the parallel
///   region; each worker writes the optimal basis for its own scenarios.
/// - `ctx` — per-stage LP layout and noise scaling parameters bundled into a
///   single [`crate::context::StageContext`]. Contains: stage templates, base
///   row indices, noise scale factors, hydro and load-bus counts, load-balance
///   row starts, load-bus index mapping, and per-stage block counts.
/// - `fcf` — Future Cost Function carrying the current Benders cut pools.
/// - `stochastic` — pre-built stochastic pipeline (tree, seed, dim).
/// - `local_forward_passes` — number of forward-pass scenarios assigned to this
///   rank (the caller splits the user's total across MPI ranks).
/// - `iteration` — current training iteration (0-based counter used for seed
///   derivation).
/// - `horizon` — horizon mode determining the stage count.
/// - `initial_state` — starting state for every scenario (length `n_state`).
/// - `records` — pre-allocated output slice of length
///   `local_forward_passes * num_stages`.
/// - `indexer` — LP column/row layout map for this stage.
/// - `fwd_offset` — global index of this rank's first forward pass. Used for
///   deterministic seed derivation (`global_scenario = fwd_offset + m`).
/// - `inflow_method` — inflow non-negativity treatment. Controls whether
///   slack columns are present in the LP for absorbing negative inflow.
///
/// ## Record layout
///
/// `records[scenario * num_stages + stage]` holds the LP solution for scenario
/// `scenario` at 0-based stage `stage`.
///
/// ## Error handling
///
/// On `SolverError::Infeasible`, returns `SddpError::Infeasible` with the
/// 0-based stage and local scenario indices. On any other `SolverError`,
/// returns `SddpError::Solver`. On error, `records` may be partially
/// populated.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible { .. })` when a stage LP has no
/// feasible solution. Returns `Err(SddpError::Solver(_))` for all other
/// terminal LP solver failures.
///
/// # Panics (debug builds only)
///
/// Panics if any of the following debug preconditions are violated:
///
/// - `records.len() != local_forward_passes * num_stages`
/// - `initial_state.len() != indexer.n_state`
/// - `ctx.templates.len() != num_stages`
/// - `ctx.base_rows.len() != num_stages`
///
/// Structurally independent parameters: `workspaces` / `basis_store` are per-rank mutable,
/// `ctx`/`baked`/`fcf` are per-stage read/write state, `training_ctx` is study-level,
/// `batch` is per-iteration config, `records` is the trajectory output.
/// Thin shim: delegates to `ForwardPassState::run`.
///
/// Callers in the test suite pass arguments in the old free-function style;
/// the shim constructs a temporary `ForwardPassState` and
/// `ForwardPassInputs` and calls `run` on them.
/// Production callers use `TrainingSession::run_forward_phase` which drives
/// `fwd_state.run(...)` directly.
pub fn run_forward_pass<S>(
    workspaces: &mut [SolverWorkspace<S>],
    basis_store: &mut BasisStore,
    ctx: &StageContext<'_>,
    baked: &[StageTemplate],
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    batch: &ForwardPassBatch<'_>,
    records: &mut [TrajectoryRecord],
) -> Result<ForwardResult, SddpError>
where
    S: SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
{
    use crate::forward_pass_state::{ForwardPassInputs, ForwardPassState};
    let n_workers = workspaces.len().max(1);
    let num_stages = training_ctx.horizon.num_stages();
    let mut state = ForwardPassState::new(n_workers, num_stages, batch.local_forward_passes);
    let mut inputs = ForwardPassInputs {
        workspaces,
        basis_store,
        ctx,
        baked,
        fcf,
        training_ctx,
        records,
        local_forward_passes: batch.local_forward_passes,
        total_forward_passes: batch.total_forward_passes,
        iteration: batch.iteration,
        fwd_offset: batch.fwd_offset,
        event_sender: batch.event_sender,
    };
    state.run(&mut inputs)
}
