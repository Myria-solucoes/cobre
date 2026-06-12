//! Training phase for `cobre run`.
//!
//! Runs the SDDP training loop, collects events, aggregates solver stats across
//! MPI ranks, and prints the training summary.

use std::sync::mpsc;

use cobre_comm::{Communicator, ReduceOp};
use cobre_core::TrainingEvent;
use cobre_sddp::StudySetup;
use cobre_solver::ActiveSolver;

use crate::error::CliError;
use crate::summary::TrainingSummary;

use super::{RunContext, check_stats_overflow};

/// Output of [`run_training_phase`]: result, training output, and optional error.
pub(super) struct TrainingPhaseResult {
    pub(super) result: cobre_sddp::TrainingResult,
    pub(super) output: cobre_io::TrainingOutput,
    /// Mid-iteration training error, if any. Partial results are still valid.
    pub(super) error: Option<cobre_sddp::SddpError>,
}

/// Single owner of the globally-reduced training solve-stats values that both the
/// printed [`TrainingSummary`] and the persisted [`cobre_io::MetadataTrainingSolveStats`]
/// read from. Constructed once from the `allreduce(Sum)` receive buffer; the two
/// downstream literals must read identical values, so they share this one source.
struct GlobalTrainingStats {
    first_try: u64,
    retried: u64,
    failed: u64,
    forward_solve_seconds: f64,
    backward_solve_seconds: f64,
}

/// Run training and collect results, events, and summary stats.
// Rationale: training orchestration is inherently sequential — solver init,
// progress thread launch, the blocking train() call, event collection,
// MPI allreduces for LP-solve counts and solver stats, summary construction,
// and output routing all depend on each other's results in order. Splitting
// would require threading shared mutable state (solver, channel endpoints,
// allreduce buffers) across function boundaries with no correctness benefit.
#[allow(clippy::too_many_lines)]
pub(super) fn run_training_phase(
    ctx: &RunContext<impl Communicator>,
    setup: &mut StudySetup,
) -> Result<TrainingPhaseResult, CliError> {
    let solver_factory = ActiveSolver::new;

    let mut solver = ActiveSolver::new().map_err(|e| CliError::Solver {
        message: format!(
            "{} initialisation failed: {e}",
            cobre_solver::active_solver_name()
        ),
    })?;

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let quiet_rx: Option<mpsc::Receiver<TrainingEvent>>;
    let progress_handle = if ctx.quiet {
        quiet_rx = Some(event_rx);
        None
    } else {
        quiet_rx = None;
        Some(crate::progress::run_progress_thread(
            event_rx,
            ctx.render_mode,
            setup.loop_params.max_iterations,
            ctx.term_width,
        ))
    };

    let training_outcome = match setup.train(
        &mut solver,
        &ctx.comm,
        ctx.n_threads,
        solver_factory,
        Some(event_tx),
        None,
    ) {
        Ok(outcome) => outcome,
        Err(e) => {
            if let Some(handle) = progress_handle {
                let _ = handle.join();
            }
            return Err(CliError::from(e));
        }
    };
    let training_result = training_outcome.result;

    let events: Vec<TrainingEvent> = match (progress_handle, quiet_rx) {
        (Some(handle), _) => handle.join(),
        (None, Some(rx)) => rx.try_iter().collect(),
        (None, None) => Vec::new(),
    };
    let mut training_output = setup.build_training_output(&training_result, &events);

    let local_lp_solves: u64 = training_output
        .convergence_records
        .iter()
        .map(|r| u64::from(r.lp_solves))
        .sum();
    let mut global_lp_solves = [0u64];
    ctx.comm
        .allreduce(&[local_lp_solves], &mut global_lp_solves, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("LP solve count allreduce error: {e}"),
        })?;
    let global_lp_solves = global_lp_solves[0];

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-training barrier error: {e}"),
    })?;

    // Aggregate solver stats from the stats log and allreduce across ranks.
    // Every rank's backward entries cover *all* ranks (allgatherv in backward.rs
    // populates the full set so rank 0 can write them to parquet). Filter to this
    // rank's own contribution here so the subsequent allreduce(Sum) produces
    // correct global totals instead of multiplying backward by world_size.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let my_rank = ctx.comm.rank() as i32;
    let (
        local_first_try,
        local_retried,
        local_failed,
        local_forward_solve_s,
        local_backward_solve_s,
    ) = aggregate_solver_stats(&training_result.solver_stats_log, my_rank);

    // Guard the three u64 counters cast to f64 in the training allreduce
    // buffer below.
    let training_guard_delta = cobre_sddp::SolverStatsDelta {
        lp_successes: local_first_try.saturating_add(local_retried),
        first_try_successes: local_first_try,
        lp_failures: local_failed,
        ..cobre_sddp::SolverStatsDelta::default()
    };
    check_stats_overflow(&training_guard_delta)?;

    #[allow(clippy::cast_precision_loss)]
    let send_stats = [
        local_first_try as f64,
        local_retried as f64,
        local_failed as f64,
        local_forward_solve_s,
        local_backward_solve_s,
    ];
    let mut recv_stats = [0.0_f64; 5];
    ctx.comm
        .allreduce(&send_stats, &mut recv_stats, ReduceOp::Sum)
        .map_err(|e| CliError::Internal {
            message: format!("training solver stats allreduce error: {e}"),
        })?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let global_stats = GlobalTrainingStats {
        first_try: recv_stats[0] as u64,
        retried: recv_stats[1] as u64,
        failed: recv_stats[2] as u64,
        forward_solve_seconds: recv_stats[3],
        backward_solve_seconds: recv_stats[4],
    };

    // Pull iter-1 gap from the in-memory convergence records. Used in the
    // summary line "Gap: X% (started at Y%)". The records are not yet
    // persisted to parquet at this point in the run flow, so reading them
    // from disk would return None; the in-memory copy is authoritative.
    let initial_gap_percent = training_output
        .convergence_records
        .first()
        .and_then(|r| r.gap_percent);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let parallelism = (ctx.n_threads as u32).saturating_mul(ctx.comm.size() as u32);

    // Print training summary on rank 0.
    let training_summary = TrainingSummary {
        iterations: training_result.iterations,
        converged: training_output.converged,
        converged_at: if training_output.converged {
            Some(training_result.iterations)
        } else {
            None
        },
        reason: training_result.reason.clone(),
        lower_bound: training_result.final_lb,
        upper_bound: training_result.final_ub,
        upper_bound_std: training_result.final_ub_std,
        gap_percent: training_result.final_gap * 100.0,
        total_rows_active: training_output.cut_stats.total_active,
        total_rows_generated: training_output.cut_stats.total_generated,
        rows_in_lp_total: training_output.cut_stats.rows_in_lp_total,
        rows_in_lp_solve_count: training_output.cut_stats.rows_in_lp_solve_count,
        rows_in_lp_max: training_output.cut_stats.rows_in_lp_max,
        num_stages: u32::try_from(setup.num_stages()).unwrap_or(u32::MAX),
        total_lp_solves: global_lp_solves,
        total_time_ms: training_result.total_time_ms,
        total_first_try: Some(global_stats.first_try),
        total_retried: Some(global_stats.retried),
        total_failed: Some(global_stats.failed),
        total_forward_solve_seconds: Some(global_stats.forward_solve_seconds),
        total_backward_solve_seconds: Some(global_stats.backward_solve_seconds),
        parallelism: Some(parallelism),
        initial_gap_percent,
    };
    if !ctx.quiet && ctx.is_root {
        crate::summary::print_training_summary(&ctx.stderr, &training_summary);
    }

    // Route the MPI-aggregated training solve totals into the output carrier so
    // the cobre-io writer persists them in `training/metadata.json`. The carrier
    // is constructed by cobre-sddp with these stats left at their defaults;
    // `final_upper_bound_std` is already populated upstream and is left untouched.
    training_output.training_solve_stats = cobre_io::MetadataTrainingSolveStats {
        total_lp_solves: Some(global_lp_solves),
        first_try: Some(global_stats.first_try),
        retried: Some(global_stats.retried),
        failed: Some(global_stats.failed),
        forward_solve_seconds: Some(global_stats.forward_solve_seconds),
        backward_solve_seconds: Some(global_stats.backward_solve_seconds),
        parallelism: Some(parallelism),
    };

    Ok(TrainingPhaseResult {
        result: training_result,
        output: training_output,
        error: training_outcome.error,
    })
}

/// Aggregate this rank's own contribution from the training stats log.
///
/// Backward entries are replicated across ranks (allgatherv in `backward.rs`
/// populates the full set on every rank so rank 0 can write them to parquet).
/// Filtering by the entry's originating rank yields the per-rank local totals
/// that can then be correctly summed across ranks via `allreduce(Sum)`.
/// Forward and `lower_bound` entries carry this rank's MPI rank, so they pass
/// through unchanged.
fn aggregate_solver_stats(
    stats_log: &[cobre_sddp::SolverStatsLogEntry],
    my_rank: i32,
) -> (u64, u64, u64, f64, f64) {
    let mut first_try = 0u64;
    let mut retried = 0u64;
    let mut failed = 0u64;
    let mut forward_solve_ms = 0.0_f64;
    let mut backward_solve_ms = 0.0_f64;
    for entry in stats_log {
        if entry.rank != my_rank {
            continue;
        }
        let delta = &entry.delta;
        first_try += delta.first_try_successes;
        retried += delta.lp_successes.saturating_sub(delta.first_try_successes);
        failed += delta.lp_failures;
        match entry.phase {
            "forward" => forward_solve_ms += delta.solve_time_ms,
            "backward" => backward_solve_ms += delta.solve_time_ms,
            _ => {}
        }
    }
    (
        first_try,
        retried,
        failed,
        forward_solve_ms / 1000.0,
        backward_solve_ms / 1000.0,
    )
}
