//! Non-uniform-block extraction-correctness assertions.
//!
//! These tests pin the simulation read-path against the equipment-base /
//! cost-range bug that surfaces only when a stage's block count differs from
//! stage 0's. Both D33 (`[1, 3, 2]`, no anticipated thermals) and D34
//! (`[1, 3, 2]` plus an anticipated thermal) declare such a schedule, so any
//! family read with the global stage-0 base/length misreads the interior stages.
//!
//! Two assertions, each FAILS against the pre-fix stage-0-base/length read and
//! PASSES once extraction resolves the per-stage `StageGeometry`:
//!
//! 1. **Per-block equipment shape** — at every stage the simulation must emit one
//!    hydro record per (block, hydro) pair, i.e. exactly `n_blks(stage)` records
//!    per hydro. A wrong per-stage stride/base does not change the record *count*
//!    but does change which column each record reads; this assertion pins the
//!    count as a coarse structural guard and the reconciliation below pins the
//!    *values*.
//! 2. **Cost-breakdown reconciliation** — `Σ(cost categories)` must equal
//!    `immediate_cost` at every stage, including the non-uniform interior stages.
//!    Pre-fix, `compute_cost_result` sums the stage-0 ranges (wrong base AND
//!    length) for the interior stages, so the breakdown no longer reconciles to
//!    the solved objective; post-fix it sums the stage-correct ranges and
//!    reconciles exactly. D33/D34 declare no generic constraints and no NCS, so
//!    every cost category is an objective·primal·scale sum and the breakdown is
//!    expected to reconcile to the LP objective to within floating-point round-off.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "highs")]

use std::path::Path;
use std::sync::mpsc;

use cobre_core::{TrainingEvent, scenario::ScenarioSource};
use cobre_sddp::{
    SimulationScenarioResult, StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use cobre_solver::highs::HighsSolver;

mod common;
use common::StubComm;

// ---------------------------------------------------------------------------
// Single-rank stub communicator (mirrors the parity harness).
// ---------------------------------------------------------------------------

fn case_dir(suffix: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(suffix)
}

fn train_and_simulate(suffix: &str) -> Vec<SimulationScenarioResult> {
    let dir = case_dir(suffix);
    let config_path = dir.join("config.json");

    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(&dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 1;

    let params =
        StudyParams::from_config(&config_with_sim).expect("StudyParams::from_config must succeed");
    let construction = params.into_construction_config();

    let sentinel = Path::new("config.json");
    let training_source = config_with_sim
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config_with_sim
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let mut setup = StudySetup::from_broadcast_params(
        &system,
        stochastic,
        construction,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
    let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            1,
            HighsSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "{suffix}: training error");
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, HighsSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.baked_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    scenario_results
}

/// The non-uniform block schedule shared by D33 and D34: the interior stages
/// differ from stage 0's single block, which is the bug trigger.
const BLOCK_COUNTS: [usize; 3] = [1, 3, 2];

/// Assert one hydro record per (block, hydro) pair at every stage: a wrong
/// per-stage stride does not change the count, but the count is the coarse shape
/// guard the value reconciliation below complements.
fn assert_per_block_equipment_shape(scenarios: &[SimulationScenarioResult], label: &str) {
    let scenario = scenarios.first().expect("one simulation scenario");
    for stage in &scenario.stages {
        let s = stage.stage_id as usize;
        let expected_blocks = BLOCK_COUNTS[s];
        // One hydro in both fixtures; per-block branch emits `n_blks` records.
        let block_ids: Vec<u32> = stage.hydros.iter().filter_map(|h| h.block_id).collect();
        assert_eq!(
            block_ids.len(),
            expected_blocks,
            "{label}: stage {s} must emit {expected_blocks} per-block hydro records",
        );
        // Cross-check the two spillage read paths against each other. The
        // per-record `spillage_cost` (a Part-1 per-block `grid.flat` read of the
        // spillage column) summed across blocks must equal the cost result's
        // `spillage_cost` (a Part-2 `range_sum` over the spillage range). D33/D34
        // declare no diversion, so the cost result's `spillage_cost` is the pure
        // spillage range_sum (diversion contributes 0). Pre-fix, at the
        // non-uniform interior stages the per-block base and the range base/length
        // disagree (one strides off stage-0's block width per block, the other
        // sums stage-0's range), so the two paths read different columns and the
        // cross-check diverges; post-fix both read the stage-correct columns and
        // agree to round-off.
        let per_record_spillage_cost: f64 = stage.hydros.iter().map(|h| h.spillage_cost).sum();
        let cost = stage.costs.first().expect("one cost record per stage");
        let scale = cost.spillage_cost.abs().max(1.0);
        let abs_err = (per_record_spillage_cost - cost.spillage_cost).abs();
        assert!(
            abs_err <= 1e-6 * scale,
            "{label}: stage {s} per-block spillage cost {per_record_spillage_cost} \
             disagrees with the cost-result spillage_cost {} (abs err {abs_err}); \
             the per-block column read and the range_sum addressed different \
             columns at a non-uniform-block stage",
            cost.spillage_cost,
        );
        for h in &stage.hydros {
            assert!(
                h.spillage_m3s.is_finite(),
                "{label}: stage {s} spillage must be finite",
            );
        }
    }
}

/// Reconciliation invariant: the sum of every cost-breakdown category equals
/// `immediate_cost` at every stage. The interior stages (1, 2) have block counts
/// differing from stage 0's, so this fails pre-fix (stage-0 ranges misbook the
/// cost) and passes post-fix (stage-correct ranges).
fn assert_cost_reconciliation(scenarios: &[SimulationScenarioResult], label: &str) {
    let scenario = scenarios.first().expect("one simulation scenario");
    for stage in &scenario.stages {
        let s = stage.stage_id as usize;
        let cost = stage.costs.first().expect("one cost record per stage");

        // `hydro_violation_cost` is the total of the per-constraint violation
        // costs (outflow/turbined/generation/evaporation/withdrawal); summing it
        // here together with the standalone categories covers every priced column
        // exactly once. `generic_violation_cost` and `curtailment_cost` use a
        // non-range formula, but D33/D34 declare neither generic constraints nor
        // NCS, so both are zero and do not perturb the reconciliation.
        let breakdown = cost.thermal_cost
            + cost.anticipated_thermal_cost
            + cost.contract_cost
            + cost.deficit_cost
            + cost.excess_cost
            + cost.storage_violation_cost
            + cost.filling_target_cost
            + cost.hydro_violation_cost
            + cost.inflow_penalty_cost
            + cost.generic_violation_cost
            + cost.spillage_cost
            + cost.turbined_cost
            + cost.curtailment_cost
            + cost.exchange_cost
            + cost.pumping_cost;

        // Relative tolerance scaled by the immediate cost magnitude: the
        // breakdown and `immediate_cost` are the same objective·primal·scale sum
        // grouped differently, so they agree to floating-point round-off.
        let scale = cost.immediate_cost.abs().max(1.0);
        let abs_err = (breakdown - cost.immediate_cost).abs();
        assert!(
            abs_err <= 1e-6 * scale,
            "{label}: stage {s} cost breakdown {breakdown} does not reconcile to \
             immediate_cost {} (abs err {abs_err}); a stage-0 equipment range was \
             summed at a non-uniform-block stage",
            cost.immediate_cost,
        );
    }
}

/// Pin the reported `anticipated_decision_mw` at the active interior delivery
/// stage. The decision column's base is the per-stage `thermal.end`
/// (`n_blks`-dependent); reading it off the global stage-0 base lands on a
/// thermal-generation column at a non-uniform stage, so the reported decision MW
/// equals one of the thermal's per-block `generation_mw` values instead of the
/// distinct decision primal. Post-fix the decision is a single per-plant scalar
/// (identical across blocks) distinct from every per-block generation; pre-fix it
/// collapses onto a generation column. D34's anticipated thermal (`K = 1`)
/// commits at stage 0 and re-commits at stage 1 (3 blocks ≠ stage 0's 1), so
/// stage 1 is the bug-exposing active delivery stage.
fn assert_anticipated_decision_mw(scenarios: &[SimulationScenarioResult], label: &str) {
    let scenario = scenarios.first().expect("one simulation scenario");
    // Stage 1 is the active interior stage with a block count differing from
    // stage 0's (3 vs 1); the anticipated K=1 thermal has a live decision there.
    let active_stage = 1u32;
    let stage = scenario
        .stages
        .iter()
        .find(|s| s.stage_id == active_stage)
        .expect("D34 has an interior stage 1");
    assert_eq!(
        BLOCK_COUNTS[active_stage as usize], 3,
        "stage 1 must carry 3 blocks (≠ stage 0's 1) to exercise the bug",
    );

    let antic: Vec<_> = stage.thermals.iter().filter(|t| t.is_anticipated).collect();
    assert!(
        !antic.is_empty(),
        "{label}: stage {active_stage} must report an anticipated thermal",
    );

    // The decision is a per-plant-per-stage scalar: identical across all block
    // records of the same anticipated thermal, present, finite, and positive.
    let decision = antic[0]
        .anticipated_decision_mw
        .expect("active anticipated decision must be Some at the delivery stage");
    assert!(
        decision.is_finite() && decision > 0.0,
        "{label}: anticipated_decision_mw must be a positive finite scalar, got {decision}",
    );
    for t in &antic {
        let d = t
            .anticipated_decision_mw
            .expect("anticipated decision present for every block record at the active stage");
        assert!(
            (d - decision).abs() <= 1e-9,
            "{label}: anticipated_decision_mw must be identical across blocks of the \
             same thermal (per-plant scalar), got {d} vs {decision}",
        );
        // The decisive base-correctness check: a stage-0-based read lands on a
        // thermal-generation column at this non-uniform stage, so the reported
        // decision would equal one of the per-block generation values. The
        // stage-correct decision column is distinct from every per-block
        // generation of the same thermal.
        assert!(
            (d - t.generation_mw).abs() > 1e-6,
            "{label}: anticipated_decision_mw {d} coincides with this thermal's \
             per-block generation_mw {} at stage {active_stage} — the decision was \
             read off the global stage-0 base onto a generation column",
            t.generation_mw,
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d33_non_uniform_blocks_per_block_equipment_shape() {
    let scenarios = train_and_simulate("d33-per-stage-block-counts");
    assert_per_block_equipment_shape(&scenarios, "D33");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d33_non_uniform_blocks_cost_reconciles() {
    let scenarios = train_and_simulate("d33-per-stage-block-counts");
    assert_cost_reconciliation(&scenarios, "D33");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d34_anticipated_non_uniform_blocks_per_block_equipment_shape() {
    let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
    assert_per_block_equipment_shape(&scenarios, "D34");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d34_anticipated_non_uniform_blocks_cost_reconciles() {
    // D34 exercises the anticipated-decision column range too: its base is the
    // per-stage `thermal.end`, so the reconciliation here additionally pins the
    // anticipated-decision range repoint at the interior delivery stages.
    let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
    assert_cost_reconciliation(&scenarios, "D34");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d34_anticipated_decision_mw_reads_stage_correct_column() {
    let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
    assert_anticipated_decision_mw(&scenarios, "D34");
}
