//! Criterion micro-benchmark for the per-stage `StageWorkerStatsBuffer` gather.
//!
//! Locks in the zero-allocation property of the gather loop.
//! Target: < 100 microseconds per stage at production sizing
//! (`n_workers = 10`, `n_openings = 20`).
//!
//! Run with: `cargo bench --bench backward_stats_gather`

#![allow(missing_docs)]

use std::sync::mpsc;

use cobre_core::{TrainingEvent, WorkerTimingPhase};
use cobre_solver::{
    SolverInterface,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

use cobre_sddp::solver_stats::{SolverStatsDelta, StageWorkerStatsBuffer};

/// Trivial profile type for the bench mock (satisfies the trait's
/// `Copy + PartialEq + Default + Send` associated-type bounds).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct BenchProfile;

/// Minimal `SolverInterface` impl holding a fixed `SolverStatistics` snapshot
/// with a non-empty `retry_level_histogram`, used solely to exercise the
/// `statistics_into` copy path.
struct BenchStatsMockSolver {
    stats: SolverStatistics,
}

impl SolverInterface for BenchStatsMockSolver {
    type Profile = BenchProfile;

    fn apply_profile(&mut self, _profile: &BenchProfile) {}
    fn load_model(&mut self, _template: &StageTemplate) {}
    fn add_rows(&mut self, _rows: &RowBatch) {}
    fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
    fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        Err(SolverError::InternalError {
            message: "bench mock".to_string(),
            error_code: None,
        })
    }
    fn get_basis(&mut self, _out: &mut Basis) {}
    fn statistics(&self) -> SolverStatistics {
        self.stats.clone()
    }
    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&self.stats);
    }
    fn name(&self) -> &'static str {
        "BenchStatsMock"
    }
    fn solver_name_version(&self) -> String {
        "BenchStatsMockSolver 0.0.0".to_string()
    }
    fn set_primal_feasibility_tolerance(&mut self, _value: f64) {}
    fn set_dual_feasibility_tolerance(&mut self, _value: f64) {}
    fn set_simplex_iteration_limit_profile(&mut self, _value: u32) {}
    fn set_ipm_iteration_limit_profile(&mut self, _value: u32) {}
}

fn bench_gather(c: &mut Criterion) {
    let n_workers = 10;
    let n_openings = 20;
    let mut buf = StageWorkerStatsBuffer::new(n_workers, n_openings);
    let deltas: Vec<SolverStatsDelta> = (0..n_workers * n_openings)
        .map(|_| SolverStatsDelta::default())
        .collect();
    c.bench_function("backward_stats_gather 10x20", |b| {
        b.iter(|| {
            buf.reset();
            for w in 0..n_workers {
                for k in 0..n_openings {
                    buf.set(w, k, black_box(deltas[w * n_openings + k].clone()));
                }
            }
            black_box(buf.as_slice().len());
        });
    });
}

/// Per-worker `WorkerTiming` event construction + send micro-benchmark.
/// Locks in the zero-heap-allocation property of the per-worker
/// emission path: the `[f64; 16]` payload is stack-resident, the channel send
/// of the variant is amortised across iterations.
///
/// The `mpsc::Sender::send` does internally allocate a node; the focus here is
/// that the **event construction** path itself never introduces a `Vec` or
/// `Box` allocation. Inspecting the generated assembly (or running under
/// `dhat`) on this loop will show only the channel-internal allocations.
#[allow(clippy::expect_used)]
fn bench_worker_timing_emit(c: &mut Criterion) {
    let n_workers: i32 = 10;
    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    c.bench_function("worker_timing_emit 10_workers", |b| {
        b.iter(|| {
            for w in 0..n_workers {
                let timings = black_box(cobre_core::WorkerPhaseTimings::default());
                let event = TrainingEvent::WorkerTiming {
                    rank: 0,
                    worker_id: w,
                    iteration: 1,
                    phase: WorkerTimingPhase::Backward,
                    timings,
                };
                tx.send(black_box(event)).expect("channel open");
            }
        });
    });
    // Drain receiver so the channel does not grow unboundedly between runs.
    drop(tx);
    while rx.try_recv().is_ok() {}
}

/// Per-opening `statistics_into` snapshot micro-benchmark.
///
/// Locks in the steady-state zero-allocation property of the backward hot
/// loop's solver-statistics snapshot: a reusable [`SolverStatistics`] buffer is
/// filled via `statistics_into` on every iteration, which `resize`s + copies the
/// histogram in place rather than cloning a fresh `Vec`. After the first call the
/// buffer's `retry_level_histogram` capacity is stable, so no further heap
/// allocation occurs (verifiable by running this loop under `dhat`: only the
/// one-time pre-loop allocation appears).
fn bench_statistics_snapshot(c: &mut Criterion) {
    let solver = BenchStatsMockSolver {
        stats: SolverStatistics {
            solve_count: 100,
            success_count: 95,
            failure_count: 5,
            total_iterations: 4096,
            retry_count: 7,
            total_solve_time_seconds: 2.5,
            basis_consistency_failures: 1,
            first_try_successes: 88,
            basis_offered: 60,
            load_model_count: 3,
            total_load_model_time_seconds: 0.4,
            total_set_bounds_time_seconds: 0.2,
            total_basis_set_time_seconds: 0.1,
            basis_reconstructions: 12,
            retry_level_histogram: vec![1, 0, 2, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        },
    };

    // Reusable buffer: pre-warm its histogram capacity so the timed loop is
    // strictly the steady-state (zero-alloc) path.
    let mut buf = SolverStatistics::default();
    solver.statistics_into(&mut buf);

    c.bench_function("backward_statistics_snapshot", |b| {
        b.iter(|| {
            solver.statistics_into(black_box(&mut buf));
            black_box(buf.solve_count);
        });
    });
}

criterion_group!(
    benches,
    bench_gather,
    bench_worker_timing_emit,
    bench_statistics_snapshot
);
criterion_main!(benches);
