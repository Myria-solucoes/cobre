//! MPI aggregation and `SimulationSummary` computation.
//!
//! [`aggregate_simulation`] gathers per-scenario cost data across ranks and
//! computes the final [`SimulationSummary`], identical on all ranks.
//!
//! `allgatherv` is used rather than `gatherv` (the `Communicator` trait has no
//! `gatherv`): every rank receives all data and computes stats locally, avoiding
//! a subsequent broadcast.

use cobre_comm::Communicator;

use crate::risk_measure::RiskMeasure;
use crate::simulation::{
    config::SimulationConfig,
    error::SimulationError,
    types::{ScenarioCategoryCosts, SimulationSummary},
};

/// Gathered per-scenario `(scenario_id, total_cost, weight)`, canonical
/// order — `weight` is `Some` only under [`SimulationWeighting::Census`].
/// Surfaced by [`aggregate_simulation`] for
/// `simulation/scenario_summary.parquet`.
pub type GatheredScenarioCosts = Vec<(u32, f64, Option<f64>)>;

/// Which weighting [`aggregate_simulation`] applies to the gathered
/// per-scenario costs, mirroring
/// [`ForwardBound`](crate::training::forward::ForwardBound) at
/// `sync_forward`'s call site.
#[derive(Debug, Clone, Copy)]
pub enum SimulationWeighting<'a> {
    /// Monte Carlo sampling: every gathered scenario carries the sample-mean
    /// weight `1.0 / n`.
    Uniform,
    /// A declared census: per-scenario leaf-path probabilities, aligned to
    /// the canonical gathered order — one weight per admitted leaf path, from
    /// `setup/mod.rs`'s `resolve_enumerated_simulation_count`.
    Census {
        /// Per-scenario probability weights, canonical gathered order.
        /// Must sum to `1.0` within `1e-9`.
        weights: &'a [f64],
    },
}

/// Aggregate per-scenario cost data across all MPI ranks into a
/// [`SimulationSummary`] that is identical on all ranks.
///
/// `weighting` selects the reduction:
/// [`SimulationWeighting::Uniform`] is the sampled Monte-Carlo estimator
/// (`mean_cost = Σ cᵢ/n`); [`SimulationWeighting::Census`] is the exact
/// probability-weighted expectation (`mean_cost = Σ wᵢ·cᵢ`) over a declared
/// census's leaf-path probabilities.
///
/// Also returns the gathered canonical-order per-scenario `(scenario_id,
/// total_cost, weight)` rows — `weight` is `Some` only under `Census` — for
/// `simulation/scenario_summary.parquet`.
///
/// # Errors
///
/// Returns `Err(SimulationError::IoError { message })` if any collective
/// operation (`allgatherv`) fails.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::Phase;
/// use cobre_sddp::simulation::aggregation::{SimulationWeighting, aggregate_simulation};
/// use cobre_sddp::simulation::{ScenarioCategoryCosts, SimulationConfig};
///
/// let zero_cats = ScenarioCategoryCosts {
///     resource_cost: 0.0,
///     recourse_cost: 0.0,
///     violation_cost: 0.0,
///     regularization_cost: 0.0,
///     imputed_cost: 0.0,
/// };
/// let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = vec![
///     (0, 100.0, zero_cats),
/// ];
/// let config = SimulationConfig {
///     n_scenarios: 1,
///     io_channel_capacity: 1,
///     profile: Phase::Simulation.profile(),
/// };
/// let comm = LocalBackend;
///
/// let (summary, gathered) =
///     aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform).unwrap();
/// assert_eq!(summary.n_scenarios, 1);
/// assert_eq!(summary.mean_cost, 100.0);
/// assert_eq!(summary.std_cost, 0.0);
/// assert_eq!(gathered, vec![(0, 100.0, None)]);
/// ```
pub fn aggregate_simulation<C: Communicator>(
    local_costs: &[(u32, f64, ScenarioCategoryCosts)],
    config: &SimulationConfig,
    comm: &C,
    weighting: SimulationWeighting<'_>,
) -> Result<(SimulationSummary, GatheredScenarioCosts), SimulationError> {
    let num_ranks = comm.size();
    let n_local = local_costs.len();

    // Per-rank scenario counts give the displacement layout for the data gather.
    #[allow(clippy::cast_possible_truncation)]
    let counts_send = [n_local as u64];
    let mut counts_recv = vec![0u64; num_ranks];
    let counts_counts = vec![1usize; num_ranks];
    let counts_displs: Vec<usize> = (0..num_ranks).collect();
    comm.allgatherv(
        &counts_send,
        &mut counts_recv,
        &counts_counts,
        &counts_displs,
    )
    .map_err(|e| SimulationError::IoError {
        message: format!("allgatherv(counts) failed: {e}"),
    })?;

    let (cost_displs, total_gathered) = compute_displs_and_total(&counts_recv);
    let cost_send: Vec<f64> = local_costs.iter().map(|(_, c, _)| *c).collect();
    let mut cost_recv = vec![0.0_f64; total_gathered];
    let cost_counts: Vec<usize> = counts_recv
        .iter()
        .map(|&c| usize::try_from(c).unwrap_or(usize::MAX))
        .collect();
    comm.allgatherv(&cost_send, &mut cost_recv, &cost_counts, &cost_displs)
        .map_err(|e| SimulationError::IoError {
            message: format!("allgatherv(costs) failed: {e}"),
        })?;

    debug_assert_eq!(
        total_gathered, config.n_scenarios as usize,
        "gathered scenario count must match configured n_scenarios"
    );

    // `assign_scenarios` (simulation/extraction.rs) hands each rank a
    // contiguous, ascending scenario_id range in rank order, exactly the
    // layout this allgatherv's counts/displs already reproduce — so gathered
    // position IS canonical scenario_id, with no second collective needed.
    #[cfg(debug_assertions)]
    {
        let base = cost_displs[comm.rank()];
        for (i, (id, _, _)) in local_costs.iter().enumerate() {
            debug_assert_eq!(
                usize::try_from(*id).unwrap_or(usize::MAX),
                base + i,
                "aggregate_simulation: scenario_id must equal its canonical gathered position"
            );
        }
    }

    let n = total_gathered;
    let weights = resolve_weights(weighting, n);

    let mean_cost = RiskMeasure::Expectation.evaluate_risk(&cost_recv, &weights);
    let std_cost = compute_std(&cost_recv, mean_cost);

    let is_census = matches!(weighting, SimulationWeighting::Census { .. });
    let gathered: GatheredScenarioCosts = cost_recv
        .iter()
        .zip(&weights)
        .enumerate()
        .map(|(i, (&cost, &w))| {
            #[allow(clippy::cast_possible_truncation)]
            let scenario_id = i as u32;
            (scenario_id, cost, is_census.then_some(w))
        })
        .collect();

    Ok((
        SimulationSummary {
            mean_cost,
            std_cost,
            #[allow(clippy::cast_possible_truncation)]
            n_scenarios: total_gathered as u32,
        },
        gathered,
    ))
}

// ── Private helpers ────────────────────────────────────────────────────────────

/// Prefix-sum displacements from per-rank counts: `(displs, total)`, where
/// `displs[r]` is rank `r`'s start offset in the receive buffer.
fn compute_displs_and_total(counts_recv: &[u64]) -> (Vec<usize>, usize) {
    let mut displs = Vec::with_capacity(counts_recv.len());
    let mut offset = 0usize;
    for &c in counts_recv {
        displs.push(offset);
        offset += usize::try_from(c).unwrap_or(usize::MAX);
    }
    (displs, offset)
}

/// Resolve `weighting` into a per-scenario weight vector of length `n`,
/// canonical gathered order.
fn resolve_weights(weighting: SimulationWeighting<'_>, n: usize) -> Vec<f64> {
    match weighting {
        SimulationWeighting::Uniform => {
            #[allow(clippy::cast_precision_loss)]
            let w = 1.0 / (n as f64);
            vec![w; n]
        }
        SimulationWeighting::Census { weights } => {
            debug_assert_eq!(
                weights.len(),
                n,
                "Census weight vector length ({}) must match gathered scenario count ({n})",
                weights.len()
            );
            debug_assert!(
                (weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "Census weights must sum to 1.0 within 1e-9"
            );
            weights.to_vec()
        }
    }
}

/// Sample standard deviation (Bessel-corrected) of `costs` around `mean`.
///
/// Returns `0.0` for `costs.len() <= 1` (no variance with a single
/// observation, or none).
fn compute_std(costs: &[f64], mean: f64) -> f64 {
    let n = costs.len();
    if n <= 1 {
        return 0.0;
    }

    let sum_sq_diff: f64 = costs.iter().map(|&c| (c - mean) * (c - mean)).sum();
    #[allow(clippy::cast_precision_loss)]
    let variance = sum_sq_diff / (n as f64 - 1.0);
    variance.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]

    use cobre_comm::LocalBackend;

    use super::{SimulationWeighting, aggregate_simulation};
    use crate::simulation::{config::SimulationConfig, types::ScenarioCategoryCosts};

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn zero_cats() -> ScenarioCategoryCosts {
        ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        }
    }

    fn make_config(n: u32) -> SimulationConfig {
        SimulationConfig {
            n_scenarios: n,
            io_channel_capacity: 1,
            profile: crate::solve::solver_phase::Phase::Simulation.profile(),
        }
    }

    // ── AC1: Uniform weighted mean matches RiskMeasure::Expectation ───────────

    #[test]
    fn aggregate_uniform_mean_matches_risk_measure_expectation() {
        let local_costs = vec![
            (0u32, 100.0, zero_cats()),
            (1u32, 200.0, zero_cats()),
            (2u32, 150.0, zero_cats()),
        ];
        let config = make_config(3);
        let comm = LocalBackend;

        let (summary, _gathered) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();

        let expected = crate::risk_measure::RiskMeasure::Expectation
            .evaluate_risk(&[100.0, 200.0, 150.0], &[1.0 / 3.0; 3]);
        assert_eq!(summary.mean_cost, expected);
        assert!(
            (summary.mean_cost - 150.0).abs() < 1e-9,
            "mean_cost {} not within 1e-9 of the old sum/n value 150.0",
            summary.mean_cost
        );
        assert_eq!(summary.n_scenarios, 3);
    }

    // ── AC3: Census weighting seam on synthetic weights ────────────────────────

    #[test]
    fn aggregate_census_weighted_mean() {
        let local_costs = vec![(0u32, 10.0, zero_cats()), (1u32, 30.0, zero_cats())];
        let config = make_config(2);
        let comm = LocalBackend;
        let weights = [0.5, 0.5];

        let (summary, gathered) = aggregate_simulation(
            &local_costs,
            &config,
            &comm,
            SimulationWeighting::Census { weights: &weights },
        )
        .unwrap();

        assert_eq!(summary.mean_cost, 20.0);
        assert_eq!(
            gathered,
            vec![(0, 10.0, Some(0.5)), (1, 30.0, Some(0.5))],
            "gathered rows must carry the census weight, canonical scenario-id order"
        );
    }

    #[test]
    fn aggregate_uniform_gathered_rows_carry_no_weight() {
        let local_costs = vec![(0u32, 100.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let (_summary, gathered) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();

        assert_eq!(gathered, vec![(0, 100.0, None)]);
    }

    // ── AC5: struct-shape / no hard-coded 0.0 ──────────────────────────────────

    #[test]
    fn aggregate_summary_carries_exactly_three_fields() {
        let local_costs = vec![(0u32, 999.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let (summary, _gathered) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();

        // Exhaustive destructure: a field added to `SimulationSummary` without a
        // matching test update fails to compile here.
        let crate::simulation::types::SimulationSummary {
            mean_cost,
            std_cost,
            n_scenarios,
        } = summary;
        assert_eq!(mean_cost, 999.0);
        assert_eq!(std_cost, 0.0);
        assert_eq!(n_scenarios, 1);
    }

    // ── AC2 (part): removed symbols do not exist ───────────────────────────────
    //
    // A grep-asserted inspection test: `compute_cvar`, `CVAR_ALPHA`,
    // `compute_local_min_max`, `pack_category_costs`, `compute_category_stats`,
    // `N_CATEGORIES`, and `CATEGORY_NAMES` are not just unreferenced but ABSENT
    // from the crate source — the actual `cvar`/`cvar_alpha` field removal is
    // pinned by the struct-shape test above, which would fail to compile if
    // either field returned.
    #[test]
    fn removed_cvar_and_category_symbols_are_absent_from_source() {
        // Declaration-form needles (`fn X(`/`const X`), assembled at runtime
        // via `format!` so this test's own needle text never appears
        // contiguously in the source it inspects — `include_str!` brings in
        // the whole file verbatim, including this test.
        let source = include_str!("aggregation.rs");
        let needles = [
            format!("fn compute_{}(", "cvar"),
            format!("const CVAR_{}", "ALPHA"),
            format!("fn compute_local_{}(", "min_max"),
            format!("fn pack_{}(", "category_costs"),
            format!("fn compute_{}(", "category_stats"),
            format!("const N_{}", "CATEGORIES"),
            format!("const {}_NAMES", "CATEGORY"),
        ];
        for needle in &needles {
            assert!(
                !source.contains(needle.as_str()),
                "removed declaration {needle} must not reappear in aggregation.rs"
            );
        }
    }

    // ── AC4 (unit-level repeat check; the rank/thread-shape gate lives in
    //    tests/mpi_wire.rs::simulation_aggregation_determinism) ────────────────

    #[test]
    fn aggregate_mean_std_bit_identical_across_repeated_calls() {
        let local_costs = vec![
            (0u32, 100.0, zero_cats()),
            (1u32, 200.0, zero_cats()),
            (2u32, 150.0, zero_cats()),
            (3u32, 400.0, zero_cats()),
        ];
        let config = make_config(4);
        let comm = LocalBackend;

        let (single_rank, _) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();
        let (repeat, _) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();

        assert_eq!(
            single_rank.mean_cost.to_bits(),
            repeat.mean_cost.to_bits(),
            "mean_cost must be bit-identical across repeated identical-shape runs"
        );
        assert_eq!(
            single_rank.std_cost.to_bits(),
            repeat.std_cost.to_bits(),
            "std_cost must be bit-identical across repeated identical-shape runs"
        );
    }

    #[test]
    fn aggregate_single_scenario_std_zero() {
        let local_costs = vec![(0u32, 999.0, zero_cats())];
        let config = make_config(1);
        let comm = LocalBackend;

        let (summary, _gathered) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();
        assert_eq!(summary.std_cost, 0.0);
        assert_eq!(summary.mean_cost, 999.0);
    }

    #[test]
    fn aggregate_std_five_costs_bessel_corrected() {
        let local_costs: Vec<(u32, f64, ScenarioCategoryCosts)> = (0u32..5)
            .map(|i| (i, f64::from(i + 1) * 100.0, zero_cats()))
            .collect();
        let config = make_config(5);
        let comm = LocalBackend;

        let (summary, _gathered) =
            aggregate_simulation(&local_costs, &config, &comm, SimulationWeighting::Uniform)
                .unwrap();
        let expected_std = 25000.0_f64.sqrt();
        assert!(
            (summary.std_cost - expected_std).abs() < 1e-9,
            "expected std={expected_std}, got {}",
            summary.std_cost
        );
    }
}
