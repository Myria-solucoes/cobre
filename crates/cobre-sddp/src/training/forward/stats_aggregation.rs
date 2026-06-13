//! Forward-pass upper-bound statistics aggregation.
//!
//! Owns `sync_forward`: the cross-rank `allgatherv` plus the canonical-order
//! `WelfordAccumulator` summation that yields bit-identical upper-bound
//! statistics regardless of MPI rank count. The Welford summation is a
//! determinism contract — iterating the gathered costs in one fixed global
//! order is what makes the result rank-count-invariant.

use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::WelfordAccumulator;

use super::{ForwardResult, SyncResult};
use crate::error::SddpError;
// Rationale: `run_forward_pass` is imported solely so the `[run_forward_pass]`
// intra-doc link in `sync_forward`'s rustdoc resolves — the function it documents
// lives in the parent `mod.rs`. The `sync_forward` body is a byte-for-byte
// verbatim move, so its doc link cannot be requalified here; bringing the symbol
// into scope is the resolution that leaves the moved region untouched.
#[allow(unused_imports)]
use super::run_forward_pass;

/// Aggregate local forward pass statistics across all MPI ranks.
///
/// Performs an `allgatherv` collective operation to produce global upper bound
/// statistics from the per-rank [`ForwardResult`] produced by [`run_forward_pass`]:
///
/// - `allgatherv` of `local.scenario_costs` gathers every rank's per-scenario
///   costs into a single canonical-order buffer (rank 0's costs first, then
///   rank 1's, …). All ranks receive the full cost vector.
/// - Statistics are computed via sequential summation in canonical global
///   scenario index order, eliminating floating-point non-associativity from
///   different rank-count groupings.
///
/// From the gathered costs, the following statistics are computed (per SS3.1a):
/// - `mean = global_cost_sum / N`
/// - `variance = (global_cost_sum_sq - N * mean^2) / (N - 1)` when `N > 1`
/// - `variance = 0.0` when `N <= 1` (Bessel correction edge case)
/// - `std_dev = max(0, variance).sqrt()` (guard against negative variance from
///   floating-point catastrophic cancellation)
/// - `ci_95 = 1.96 * std_dev / sqrt(N)`
///
/// The lower bound is **not** computed here. It is evaluated separately after
/// the backward pass adds new cuts to the FCF.
///
/// In single-rank mode (`comm.size() == 1`), `LocalBackend.allgatherv` is an
/// identity copy. No special-casing is needed — the result equals the local values.
///
/// ## Arguments
///
/// - `local` — the [`ForwardResult`] from the calling rank's forward pass.
/// - `comm` — the communicator used for collective operations.
/// - `total_forward_passes` — the total number of forward passes across all
///   ranks. Used to compute per-rank counts and displacements arithmetically
///   without a preliminary communication round.
///
/// # Errors
///
/// Returns `Err(SddpError::Communication(_))` if the `allgatherv` call fails.
/// The `From<CommError>` conversion on `SddpError` is applied automatically
/// via the `?` operator. No partial results are produced on error.
pub fn sync_forward<C: Communicator>(
    local: &ForwardResult,
    comm: &C,
    total_forward_passes: usize,
) -> Result<SyncResult, SddpError> {
    let start = Instant::now();

    let num_ranks = comm.size();
    let my_rank = comm.rank();

    // Compute per-rank counts and displacements arithmetically from the total.
    // Each rank r receives `partition(total_forward_passes, num_ranks, r)` scenarios.
    let base = total_forward_passes / num_ranks;
    let remainder = total_forward_passes % num_ranks;
    let counts: Vec<usize> = (0..num_ranks)
        .map(|r| base + usize::from(r < remainder))
        .collect();
    let mut displs = vec![0usize; num_ranks];
    for r in 1..num_ranks {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    // Allocate the global cost buffer and gather all per-scenario costs.
    let global_n = counts.iter().sum::<usize>();
    debug_assert_eq!(
        global_n, total_forward_passes,
        "counts sum {global_n} != total_forward_passes {total_forward_passes}",
    );
    let mut global_costs = vec![0.0_f64; global_n];

    // Validate that this rank's local cost vector matches the expected count.
    debug_assert_eq!(
        local.scenario_costs.len(),
        counts[my_rank],
        "rank {my_rank}: scenario_costs length {} != expected count {}",
        local.scenario_costs.len(),
        counts[my_rank],
    );

    comm.allgatherv(&local.scenario_costs, &mut global_costs, &counts, &displs)?;

    // Canonical-order single-pass statistics. All ranks iterate global_costs in
    // the same order, producing bit-identical statistics regardless of rank count.
    // Welford's online algorithm is used instead of the two-pass naive formula to
    // avoid catastrophic cancellation when sum_sq ≈ n * mean^2.
    // MPI Welford merge is not used here because the full gathered array is
    // already available — a single sequential pass suffices.
    let mut welford = WelfordAccumulator::new();
    for &c in &global_costs {
        welford.update(c);
    }
    let mean = welford.mean();
    let (std_dev, ci_95) = if global_n > 1 {
        (welford.sample_std_dev(), welford.sample_ci_95_half_width())
    } else {
        (0.0_f64, 0.0_f64)
    };

    #[allow(clippy::cast_possible_truncation)]
    let sync_time_ms = start.elapsed().as_millis() as u64;

    Ok(SyncResult {
        global_ub_mean: mean,
        global_ub_std: std_dev,
        ci_95_half_width: ci_95,
        sync_time_ms,
    })
}
