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
// Rationale: imported solely so the `[run_forward_pass]` intra-doc link in
// `sync_forward`'s rustdoc resolves; the function lives in the parent `mod.rs`.
#[allow(unused_imports)]
use super::run_forward_pass;

/// Aggregate one rank's forward-pass statistics across all MPI ranks.
///
/// `allgatherv`s `local.scenario_costs` into a canonical-order global buffer,
/// then computes mean/std/CI by sequential summation in that fixed order so the
/// result is bit-identical regardless of rank count. In single-rank mode
/// `LocalBackend.allgatherv` is an identity copy, needing no special case.
///
/// The lower bound is **not** computed here; it is evaluated after the backward
/// pass adds new cuts to the FCF.
///
/// # Errors
///
/// Returns `Err(SddpError::Communication(_))` if the `allgatherv` call fails.
/// No partial results are produced on error.
pub fn sync_forward<C: Communicator>(
    local: &ForwardResult,
    comm: &C,
    total_forward_passes: usize,
) -> Result<SyncResult, SddpError> {
    let start = Instant::now();

    let num_ranks = comm.size();
    let my_rank = comm.rank();

    // Per-rank counts/displacements derived arithmetically from the total, so no
    // preliminary communication round is needed.
    let base = total_forward_passes / num_ranks;
    let remainder = total_forward_passes % num_ranks;
    let counts: Vec<usize> = (0..num_ranks)
        .map(|r| base + usize::from(r < remainder))
        .collect();
    let mut displs = vec![0usize; num_ranks];
    for r in 1..num_ranks {
        displs[r] = displs[r - 1] + counts[r - 1];
    }

    let global_n = counts.iter().sum::<usize>();
    debug_assert_eq!(
        global_n, total_forward_passes,
        "counts sum {global_n} != total_forward_passes {total_forward_passes}",
    );
    let mut global_costs = vec![0.0_f64; global_n];

    debug_assert_eq!(
        local.scenario_costs.len(),
        counts[my_rank],
        "rank {my_rank}: scenario_costs length {} != expected count {}",
        local.scenario_costs.len(),
        counts[my_rank],
    );

    comm.allgatherv(&local.scenario_costs, &mut global_costs, &counts, &displs)?;

    // Single canonical-order pass: every rank iterates global_costs in the same
    // order, so the statistics are bit-identical regardless of rank count.
    // Welford's online algorithm, not the two-pass naive formula, avoids
    // catastrophic cancellation when sum_sq ≈ n * mean^2; the full gathered array
    // is in hand, so no MPI Welford merge is needed.
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
