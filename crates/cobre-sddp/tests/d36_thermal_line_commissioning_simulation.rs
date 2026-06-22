//! Thermal and line commissioning gating must reach the simulation output.
//!
//! ## What the parity hash cannot see
//!
//! The declaration-order parity baseline hashes hydro storage / water values
//! and cuts; it does NOT hash thermal-generation or line-flow columns. The
//! zero-influence convention — a commissioning-dormant thermal pins BOTH bounds
//! to `[0, 0]` (so its `min_generation_mw` must-run floor is zeroed too) and a
//! dormant line pins `col_upper` to 0 on both directions — is therefore
//! invisible to the hash except through dispatch coupling. A gating bug that let
//! a windowed-out thermal run (or made the LP infeasible by zeroing only the
//! cap), or let a dormant line carry flow, could still hash-match. This test
//! exercises those paths directly through the full train+simulate pipeline.
//!
//! ## Fixture (`d36-thermal-line-commissioning`)
//!
//! Two buses (B0, B1), two hydros (H0 on B0, H1 on B1), one thermal (T0 on B0,
//! `min_mw = 10` must-run, `entry_stage_id = 1`, `exit_stage_id = 2`), and one
//! line (`B0_B1`, `entry_stage_id = 1`, no exit) over a 3-stage horizon (0, 1, 2),
//! single block each. The commissioning gate (`entry <= stage.id && stage.id <
//! exit`) makes:
//!
//! - T0 ACTIVE only at stage 1, DORMANT at stages 0 and 2.
//! - the line DORMANT at stage 0, ACTIVE at stages 1 and 2.
//!
//! ## Assertions
//!
//! - Stage 0: T0 emits a thermal row with `generation_mw == 0` (dormant: BOTH
//!   bounds pinned to `[0, 0]`, so the 10 MW must-run floor is dropped and the
//!   LP stays feasible). The line emits an exchange row with both flows `== 0`
//!   (dormant: caps pinned to 0).
//! - Stage 1: T0's must-run floor binds — `generation_mw >= 10`. The line is
//!   active (caps restored to 15 MW), so it may carry flow.
//! - Stage 2: T0 emits `generation_mw == 0` again (decommissioned at the exit
//!   boundary `stage.id == exit`). The line is still active (no exit).

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

/// Train the commissioning-gated thermal+line case, simulate one deterministic
/// scenario, and assert the thermal/line output is gated by the windows.
#[test]
fn thermal_line_commissioning_window_gates_simulation_output() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d36-thermal-line-commissioning");

    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    // The shipped case disables simulation (the parity harness trains only).
    // Enable one deterministic simulation scenario so the thermal/line
    // extraction paths run; `StudySetup::new` reads `n_scenarios` from this.
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
        .expect(
            "simulate must not return Err (a windowed-out must-run thermal must stay feasible)",
        );

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

    // ── Thermal gating ────────────────────────────────────────────────────────

    // Stage 0: T0 dormant (stage.id 0 < entry 1). BOTH bounds pinned to [0, 0],
    // so the 10 MW must-run floor is dropped and generation is exactly 0 — the
    // LP stays feasible despite the declared floor.
    let t_stage0 = &scenario.stages[0].thermals;
    assert_eq!(
        t_stage0.len(),
        1,
        "stage 0 keeps the dense thermal column: one (zeroed) row, got {t_stage0:?}",
    );
    assert_eq!(
        t_stage0[0].thermal_id, 0,
        "dormant row maps to system thermal 0"
    );
    assert_eq!(
        t_stage0[0].generation_mw, 0.0,
        "stage 0 is before entry: the must-run floor must be zeroed, got {}",
        t_stage0[0].generation_mw,
    );

    // Stage 2: T0 dormant (stage.id 2 == exit 2, gate is `stage.id < exit`).
    let t_stage2 = &scenario.stages[2].thermals;
    assert_eq!(
        t_stage2.len(),
        1,
        "stage 2 keeps the dense thermal column: one (zeroed) row, got {t_stage2:?}",
    );
    assert_eq!(
        t_stage2[0].generation_mw, 0.0,
        "stage 2 is at the exit boundary (decommissioned): must-run floor zeroed, got {}",
        t_stage2[0].generation_mw,
    );

    // Stage 1: T0 active — the 10 MW must-run floor binds.
    let t_stage1 = &scenario.stages[1].thermals;
    assert_eq!(t_stage1.len(), 1, "stage 1 is active: one thermal row");
    assert_eq!(
        t_stage1[0].thermal_id, 0,
        "active row maps to system thermal 0"
    );
    assert_eq!(t_stage1[0].stage_id, 1, "row stamped with stage 1");
    assert!(
        t_stage1[0].generation_mw >= 10.0 - 1e-6,
        "the must-run floor (10 MW) must bind at the active stage; got {}",
        t_stage1[0].generation_mw,
    );

    // ── Line gating ─────────────────────────────────────────────────────────────

    // Stage 0: line dormant (stage.id 0 < entry 1). Both directional caps pinned
    // to 0, so flow is exactly 0 in both directions.
    let l_stage0 = &scenario.stages[0].exchanges;
    assert_eq!(
        l_stage0.len(),
        1,
        "stage 0 keeps the dense line column: one (zeroed) row, got {l_stage0:?}",
    );
    assert_eq!(l_stage0[0].line_id, 0, "dormant row maps to system line 0");
    assert_eq!(
        l_stage0[0].direct_flow_mw, 0.0,
        "stage 0 line dormant: direct flow must be 0, got {}",
        l_stage0[0].direct_flow_mw,
    );
    assert_eq!(
        l_stage0[0].reverse_flow_mw, 0.0,
        "stage 0 line dormant: reverse flow must be 0, got {}",
        l_stage0[0].reverse_flow_mw,
    );

    // Stage 1: line active (entry 1 <= stage.id 1, no exit). Caps restored to
    // 15 MW; B1's 40 MW load exceeds H1's 20 MW cap, so B0 exports across the
    // line — direct flow is positive (and within the restored cap).
    let l_stage1 = &scenario.stages[1].exchanges;
    assert_eq!(l_stage1.len(), 1, "stage 1 line active: one exchange row");
    assert_eq!(l_stage1[0].line_id, 0, "active row maps to system line 0");
    assert!(
        l_stage1[0].direct_flow_mw > 1e-6,
        "stage 1 line active: B0 must export to the deficit-prone B1, got {}",
        l_stage1[0].direct_flow_mw,
    );
    assert!(
        l_stage1[0].direct_flow_mw <= 15.0 + 1e-6,
        "stage 1 direct flow must respect the restored 15 MW cap, got {}",
        l_stage1[0].direct_flow_mw,
    );

    // Stage 2: line still active (no exit), so its caps remain restored.
    let l_stage2 = &scenario.stages[2].exchanges;
    assert_eq!(l_stage2.len(), 1, "stage 2 line active: one exchange row");
    assert!(
        l_stage2[0].direct_flow_mw <= 15.0 + 1e-6 && l_stage2[0].reverse_flow_mw <= 15.0 + 1e-6,
        "stage 2 flows must respect the active 15 MW caps, got direct={} reverse={}",
        l_stage2[0].direct_flow_mw,
        l_stage2[0].reverse_flow_mw,
    );
}
