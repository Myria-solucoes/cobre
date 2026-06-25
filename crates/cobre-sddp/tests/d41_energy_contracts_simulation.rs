//! Energy-contract gating and dispatch must reach the simulation output.
//!
//! The parity hash does NOT digest contract-output columns, so a gating bug — a
//! commissioning-dormant contract that dispatches, a wrong-signed load-balance
//! term that flips import/export, a mis-stamped `contract_id`, a stage override
//! that fails to change dispatch — would still hash-match and ship silently. This
//! test asserts the dispatch values directly through the full train+simulate
//! pipeline; they are the only numerical guard.
//!
//! The commissioning gate is `entry <= stage.id && stage.id < exit`. Under the
//! dense layout a dormant contract keeps its column but is pinned to `[0, 0]`,
//! emitting a ZERO row rather than being absent.
//!
//! D41 topology: 3 stages, single 730h block, one bus, two hydros H0->H1, one
//! thermal at $50/MWh, deficit at $1000/MWh. Two contracts on bus 0: import id 0
//! (price $200/MWh, always active, stage-2 `min_mw` + `price_per_mwh` override)
//! and export id 1 (price -$150/MWh revenue, commissioned at stage 1 only).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    // The single test drives one train+simulate run and reads every per-stage
    // contract row from it; AC1-AC5 cannot be split without re-running the case.
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::simulation::accumulate_category_costs;
use cobre_sddp::simulation::types::ScenarioCategoryCosts;
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::ActiveSolver;

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

#[test]
fn energy_contract_gating_reaches_simulation_output() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d41-energy-contracts");

    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    // The shipped case disables simulation (the parity harness trains only);
    // enable one scenario so the contract extraction path runs.
    config.simulation = cobre_io::config::SimulationConfig {
        enabled: true,
        num_scenarios: 1,
        io_channel_capacity: 8,
        ..cobre_io::config::SimulationConfig::default()
    };

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup::new");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            None,
            &outcome.result.basis_cache,
        )
        .expect("simulate must not return Err");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(
        scenario_results.len(),
        1,
        "exactly one deterministic scenario result",
    );
    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        3,
        "one stage record per study stage (horizon 0, 1, 2)",
    );

    // Each stage keeps a dense row per contract (single block): import id 0 then
    // export id 1, regardless of commissioning.
    for (t, stage) in scenario.stages.iter().enumerate() {
        assert_eq!(
            stage.contracts.len(),
            2,
            "stage {t} keeps the dense column for both contracts, got {:?}",
            stage.contracts,
        );
    }

    let import_row = |t: usize| {
        scenario.stages[t]
            .contracts
            .iter()
            .find(|r| r.contract_id == 0)
            .unwrap_or_else(|| panic!("stage {t} must have an import (id 0) row"))
    };
    let export_row = |t: usize| {
        scenario.stages[t]
            .contracts
            .iter()
            .find(|r| r.contract_id == 1)
            .unwrap_or_else(|| panic!("stage {t} must have an export (id 1) row"))
    };

    // AC1 — commissioning gate, dormant zero: export id 1 has entry_stage_id = 1,
    // so stage 0 (t < E) is dormant: zero power, operative_state_code == 1.
    let exp0 = export_row(0);
    assert_eq!(
        exp0.power_mw, 0.0,
        "stage 0 is before the export entry window: dormant column pinned to 0, got {}",
        exp0.power_mw,
    );
    assert_eq!(
        exp0.operative_state_code, 1,
        "dormant export row still carries operative_state_code 1",
    );

    // AC2 — commissioning gate, active dispatch + post-exit zero: export id 1 has
    // exit_stage_id = 2, active only at stage 1 (E <= t < X). It is the surplus
    // stage, so the optimizer sells the revenue export at a positive power.
    let exp1 = export_row(1);
    assert!(
        exp1.power_mw > 0.0,
        "stage 1 is inside the export window and a surplus stage: export must dispatch, got {}",
        exp1.power_mw,
    );
    assert!(
        exp1.total_cost < 0.0,
        "an active export (negative price) nets negative total_cost (revenue), got {}",
        exp1.total_cost,
    );
    let exp2 = export_row(2);
    assert_eq!(
        exp2.power_mw, 0.0,
        "stage 2 is at/after the export exit boundary (decommissioned): pinned to 0, got {}",
        exp2.power_mw,
    );

    // R2 shortage — import id 0 is pulled at stage 0 (high load, scarce hydro,
    // thermal capped): import at $200 beats the $1000 deficit.
    let imp0 = import_row(0);
    assert!(
        imp0.power_mw > 0.0,
        "stage 0 is a shortage stage: import must be pulled, got {}",
        imp0.power_mw,
    );
    assert!(
        (imp0.price_per_mwh - 200.0).abs() < 1e-6,
        "stage 0 import uses the base price 200.0, got {}",
        imp0.price_per_mwh,
    );

    // AC4 — take-or-pay floor binds: the contract_bounds override sets import id 0
    // min_mw = 10.0 at stage 2, where importing is uneconomic (hydro+thermal cover
    // the load and the override price is 999.0). The floor forces power_mw == 10.0.
    let imp2 = import_row(2);
    assert!(
        (imp2.power_mw - 10.0).abs() < 1e-6,
        "stage 2 import take-or-pay floor (min_mw = 10.0) must bind, got {}",
        imp2.power_mw,
    );

    // AC3 — stage override changes price and cost: the same override sets import id
    // 0 price_per_mwh = 999.0 at stage 2, differing from the base 200.0 used at
    // stage 0, which visibly changes total_cost relative to the non-overridden
    // stage 0 import row.
    assert!(
        (imp2.price_per_mwh - 999.0).abs() < 1e-6,
        "stage 2 import price override (999.0) must be reflected, got {}",
        imp2.price_per_mwh,
    );
    assert!(
        (imp2.price_per_mwh - imp0.price_per_mwh).abs() > 1e-6,
        "override price (999.0) must differ from base price (200.0)",
    );
    let block_hours = 730.0;
    assert!(
        (imp2.total_cost - 999.0 * 10.0 * block_hours).abs() < 1.0,
        "stage 2 import total_cost = price * power * hours = 999 * 10 * 730, got {}",
        imp2.total_cost,
    );
    assert!(
        (imp2.total_cost - imp0.total_cost).abs() > 1.0,
        "overridden stage 2 import total_cost must differ from non-overridden stage 0",
    );

    // AC5 — cost-breakdown invariant with an active contract: the five macro
    // categories sum to immediate_cost at the export-active stage 1.
    let cost = &scenario.stages[1].costs[0];
    let mut accum = ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    };
    accumulate_category_costs(cost, &mut accum);
    let macro_sum = accum.resource_cost
        + accum.recourse_cost
        + accum.violation_cost
        + accum.regularization_cost
        + accum.imputed_cost;
    assert!(
        (macro_sum - cost.immediate_cost).abs() < 1.0,
        "Sigma(macro categories) ({macro_sum}) must equal immediate_cost ({}) with an active contract",
        cost.immediate_cost,
    );
}
