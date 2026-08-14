//! Forward-pass upper-bound statistics aggregation.
//!
//! Owns `sync_forward`: the cross-rank `allgatherv` plus the per-source
//! assembly of the upper bound. Under a sampled forward it is the canonical-order
//! `WelfordAccumulator` summation (sample mean + 95% CI); under an enumerated
//! forward it is the exact `Σ wᵢ·cᵢ` compensated reduction
//! ([`weighted_cost_reduction`]). Both iterate the gathered costs in one fixed
//! global order — the determinism contract that makes the result
//! rank-count-invariant.

use std::time::Instant;

use cobre_comm::Communicator;
use cobre_core::WelfordAccumulator;

use super::{ForwardResult, SyncResult};
use crate::error::SddpError;
// Rationale: imported solely so the `[run_forward_pass]` intra-doc link in
// `sync_forward`'s rustdoc resolves; the function lives in the parent `mod.rs`.
#[allow(unused_imports)]
use super::run_forward_pass;

/// Which upper-bound estimator [`sync_forward`] assembles from the gathered
/// forward-pass costs.
#[derive(Clone, Copy, Debug)]
pub enum ForwardBound<'a> {
    /// Sampled forward: Welford sample mean, standard deviation, and 95% CI
    /// half-width.
    Statistical,
    /// Enumerated forward: exact `Σ wᵢ·cᵢ` over `path_weights` (canonical order);
    /// standard deviation and CI half-width are `0` — a deduplicated enumeration
    /// carries no sampling distribution.
    Exact {
        /// Per-path probability weights, one per gathered cost, in canonical order.
        path_weights: &'a [f64],
    },
}

/// Compensated (Neumaier) `Σ wᵢ·cᵢ` over paired cost/weight slices in slice-index
/// order.
///
/// The index order is fixed and decoupled from solve/gather order, so the
/// reduction is bit-identical across rank and thread counts; Neumaier
/// compensation (not a naive running sum) holds accuracy when the weighted terms
/// span wide magnitudes.
pub(crate) fn weighted_cost_reduction(costs: &[f64], weights: &[f64]) -> f64 {
    debug_assert_eq!(
        costs.len(),
        weights.len(),
        "weighted_cost_reduction: costs ({}) and weights ({}) must align 1:1",
        costs.len(),
        weights.len(),
    );
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (&cost, &weight) in costs.iter().zip(weights.iter()) {
        let term = weight * cost;
        let t = sum + term;
        if sum.abs() >= term.abs() {
            compensation += (sum - t) + term;
        } else {
            compensation += (term - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// Aggregate one rank's forward-pass statistics across all MPI ranks.
///
/// `allgatherv`s `local.scenario_costs` into a canonical-order global buffer,
/// then assembles the upper bound per `bound`: [`ForwardBound::Statistical`]
/// computes the Welford sample mean/std/CI, [`ForwardBound::Exact`] the
/// probability-weighted `Σ wᵢ·cᵢ`. Both reduce in the fixed global order, so the
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
    bound: ForwardBound<'_>,
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

    let (mean, std_dev, ci_95) = match bound {
        ForwardBound::Statistical => {
            // Single canonical-order pass: every rank iterates global_costs in the
            // same order, so the statistics are bit-identical regardless of rank
            // count. Welford's online algorithm, not the two-pass naive formula,
            // avoids catastrophic cancellation when sum_sq ≈ n * mean^2; the full
            // gathered array is in hand, so no MPI Welford merge is needed.
            let mut welford = WelfordAccumulator::new();
            for &c in &global_costs {
                welford.update(c);
            }
            let mean = welford.mean();
            if global_n > 1 {
                (
                    mean,
                    welford.sample_std_dev(),
                    welford.sample_ci_95_half_width(),
                )
            } else {
                (mean, 0.0_f64, 0.0_f64)
            }
        }
        ForwardBound::Exact { path_weights } => {
            // Exact probability-weighted bound; std/CI are 0 because a deduplicated
            // enumeration has no sampling distribution — routing it through the
            // Welford accumulator as S samples would be a category error.
            let ub_exact = weighted_cost_reduction(&global_costs, path_weights);
            (ub_exact, 0.0_f64, 0.0_f64)
        }
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
