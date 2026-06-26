//! Focused semantic regression test for the signed evaporation-outflow variable.
//!
//! Runs D17 (`d17-evaporation-mixed-sign`) end-to-end and asserts that the
//! simulation output reflects the mixed-sign monthly evaporation coefficients
//! defined in the fixture:
//!
//! - Stages 0, 1, 2 land on Oct/Nov/Dec where `coefficients_mm` is negative
//!   (net rainfall input on the lake surface); the simulated
//!   `evaporation_m3s` must be **negative**.
//! - Stage 3 lands on January where `coefficients_mm` is positive (true
//!   evaporation loss); the simulated `evaporation_m3s` must be **positive**.
//! - In every stage the symmetric `[-q_max, +q_max]` magnitude bound must
//!   absorb the linearized target, so neither the positive nor the negative
//!   evaporation violation slack should fire.
//!
//! The companion `parity_hash_d17` test guards bit-for-bit byte stability of
//! the same case; this test guards the **sign semantics** explicitly so a
//! future regression that flips the sign convention is caught by an
//! actionable assertion rather than an opaque hash diff.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
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
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d17_evaporation_is_signed_per_month() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic/d17-evaporation-mixed-sign");

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

    let sentinel = Path::new("config.json");
    let training_source = config_with_sim
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config_with_sim
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let params =
        StudyParams::from_config(&config_with_sim).expect("StudyParams::from_config must succeed");
    let construction = params.into_construction_config();

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
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "D17: expected no training error");
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
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
    let mut scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    assert_eq!(
        scenario_results.len(),
        1,
        "expected exactly one simulation scenario for D17, got {}",
        scenario_results.len()
    );
    let scenario = scenario_results.remove(0);
    assert_eq!(
        scenario.stages.len(),
        4,
        "expected 4 stages in D17 simulation result, got {}",
        scenario.stages.len()
    );

    for stage in &scenario.stages {
        assert_eq!(
            stage.hydros.len(),
            1,
            "stage {} should have exactly one hydro record",
            stage.stage_id
        );
        let h = &stage.hydros[0];
        let evap = h
            .evaporation_m3s
            .expect("evaporation must be modeled for D17");

        match stage.stage_id {
            0..=2 => assert!(
                evap < 0.0,
                "stage {} (Oct/Nov/Dec) expected net-rainfall evaporation_m3s < 0, got {evap}",
                stage.stage_id
            ),
            3 => assert!(
                evap > 0.0,
                "stage {} (Jan) expected true-evaporation evaporation_m3s > 0, got {evap}",
                stage.stage_id
            ),
            other => panic!("unexpected stage_id in D17 simulation: {other}"),
        }

        assert_eq!(
            h.evaporation_violation_pos_m3s, 0.0,
            "stage {}: positive evaporation slack must not fire, got {}",
            stage.stage_id, h.evaporation_violation_pos_m3s
        );
        assert_eq!(
            h.evaporation_violation_neg_m3s, 0.0,
            "stage {}: negative evaporation slack must not fire, got {}",
            stage.stage_id, h.evaporation_violation_neg_m3s
        );
    }
}
