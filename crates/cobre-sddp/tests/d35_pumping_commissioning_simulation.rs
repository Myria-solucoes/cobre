//! Pumping-station commissioning gating must reach the simulation output.
//!
//! ## What the parity hash cannot see
//!
//! The declaration-order parity baseline hashes hydro storage / water values
//! and cuts; it does NOT hash pumping-output columns. The local→system index
//! mapping `extract_pumping_stations` applies — turning the active-local column
//! position `p_local` back into a `pumping_station_id` via
//! `geometry.active_pumping_indices` — is therefore invisible to the hash. A
//! gating bug that emitted a pumping row at an omitted stage, or stamped the
//! wrong station id, would still hash-match. This test exercises that mapping
//! directly through the full train+simulate pipeline.
//!
//! ## Fixture (`d35-pumping-commissioning`)
//!
//! One station (id 0, source H1 → destination H0) with
//! `entry_stage_id = 1`, `exit_stage_id = 2` over a 3-stage horizon (0, 1, 2),
//! single block each. The commissioning gate
//! (`entry <= stage.id && stage.id < exit`) makes the station ACTIVE only at
//! stage 1 and OMITTED at stages 0 and 2. Because `flow.min_m3s = 2.0` forces a
//! transfer on every active solve, stage 1 must pump at least 2.0 m³/s.
//!
//! ## Assertions
//!
//! - Stage 1 emits exactly one pumping row: `station_id` 0, `pumped_flow_m3s >= 2.0`.
//! - Stages 0 and 2 emit NO pumping rows (the station contributes no column there).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::ActiveSolver;

/// Single-rank communicator stub that faithfully copies data through the
/// collectives, so the pipeline runs without MPI.
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

/// Train the commissioning-gated pumping case, simulate one deterministic
/// scenario, and assert the pumping output is present only at the active stage.
#[test]
fn pumping_commissioning_window_gates_simulation_output() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d35-pumping-commissioning");

    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    // The shipped case disables simulation (the parity harness trains only).
    // Enable one deterministic simulation scenario so the pumping extraction
    // path runs; `StudySetup::new` reads `n_scenarios` from this config.
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
        .expect("simulate must not return Err (omitting the forced transfer only relaxes 0/2)");

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

    // Stage 0: omitted (stage.id 0 < entry 1) — no pumping column, no row.
    assert!(
        scenario.stages[0].pumping_stations.is_empty(),
        "stage 0 is before the entry window: no pumping rows, got {:?}",
        scenario.stages[0].pumping_stations,
    );

    // Stage 2: omitted (stage.id 2 == exit 2, gate is `stage.id < exit`) —
    // no pumping column, no row.
    assert!(
        scenario.stages[2].pumping_stations.is_empty(),
        "stage 2 is at the exit boundary (decommissioned): no pumping rows, got {:?}",
        scenario.stages[2].pumping_stations,
    );

    // Stage 1: active — exactly one (station, block) row mapped to station id 0,
    // pumping at least the forced minimum of 2.0 m³/s.
    let stage1 = &scenario.stages[1].pumping_stations;
    assert_eq!(
        stage1.len(),
        1,
        "stage 1 is active with a single station and a single block: one pumping row",
    );
    let row = &stage1[0];
    assert_eq!(
        row.pumping_station_id, 0,
        "active-local index 0 must map back to system station id 0",
    );
    assert_eq!(row.stage_id, 1, "row must be stamped with stage 1");
    assert!(
        row.pumped_flow_m3s >= 2.0 - 1e-6,
        "the forced minimum flow (2.0 m³/s) must bind at the active stage; got {}",
        row.pumped_flow_m3s,
    );
    // Power consumption is the pumped flow scaled by the station's
    // consumption rate (0.5 MW per m³/s); the negative-injection sign lives in
    // the LP coupling, the output reports the positive magnitude.
    assert!(
        (row.power_consumption_mw - row.pumped_flow_m3s * 0.5).abs() < 1e-6,
        "power consumption must equal pumped_flow * 0.5; got {} for flow {}",
        row.power_consumption_mw,
        row.pumped_flow_m3s,
    );
}
