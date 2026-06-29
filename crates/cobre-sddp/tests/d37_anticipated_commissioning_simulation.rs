//! Anticipated-thermal commissioning-window gating must reach the simulation
//! output, and warm-start must survive the dormancy boundary.
//!
//! ## What this case exercises that no other anticipated case does
//!
//! `d37-anticipated-commissioning` is the only deterministic case combining an
//! anticipated thermal with a commissioning window. The anticipated thermal T1
//! (`K=2`, cheap, `max 150 MW`) carries window `[entry=2, exit=4)` over a
//! 6-stage horizon (ids 0..6) with a per-stage-varying block schedule (1/3/2/3/1/2).
//!
//! The decision gate (`StateLayout::is_anticipated_decision_active`) is the
//! conjunction of the strict horizon clause (`t + K < n_stages`) and the
//! operation-window clause keyed on the DELIVERY stage (`commissioning_active(2,
//! 4, id(t + 2))`):
//!
//! - `t = 0` → delivers at id 2 ∈ `[2, 4)` → ACTIVE (the pre-entry decision,
//!   priced at `entry − K`),
//! - `t = 1` → delivers at id 3 ∈ `[2, 4)` → ACTIVE,
//! - `t = 2` → delivers at id 4 ∉ `[2, 4)` → INACTIVE (post-exit drain begins),
//! - `t = 3` → delivers at id 5 ∉ `[2, 4)` → INACTIVE,
//! - `t = 4, 5` → `t + K ≥ 6` → horizon-INACTIVE.
//!
//! The GENERATION column (operation window) zeroes T1's generation outside
//! `[entry, exit)`: T1 generates only at stages 2 and 3.
//!
//! ## What the parity hash cannot see
//!
//! The parity baseline hashes hydro storage/water/cuts/convergence, NOT thermal
//! generation, the anticipated decision, or the ring buffer. A gating bug that
//! let T1 run outside its window, or never delivered the pre-entry commitment, or
//! left the ring buffer un-drained, could still hash-match. This test exercises
//! those paths directly through train + simulate.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{
    SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

fn case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d37-anticipated-commissioning")
}

fn build_setup(case_dir: &Path) -> (StudySetup, cobre_io::config::Config) {
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    // The shipped case disables simulation (parity trains only); enable one
    // deterministic scenario so the thermal extraction paths run.
    config.simulation = cobre_io::config::SimulationConfig {
        enabled: true,
        num_scenarios: 1,
        io_channel_capacity: 8,
        ..cobre_io::config::SimulationConfig::default()
    };

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup::new");
    (setup, config)
}

const T1_ID: i32 = 1;
const ENTRY: usize = 2;
const EXIT: usize = 4;
const K: usize = 2;
const N_STAGES: usize = 6;
const TOL: f64 = 1e-6;

/// Locate the anticipated thermal (T1) result at a given stage.
fn t1_at(
    scenario: &cobre_sddp::simulation::SimulationScenarioResult,
    stage: usize,
) -> &cobre_sddp::simulation::SimulationThermalResult {
    scenario.stages[stage]
        .thermals
        .iter()
        .find(|t| t.thermal_id == T1_ID)
        .unwrap_or_else(|| panic!("T1 (id={T1_ID}) missing at stage {stage}"))
}

/// Train the windowed anticipated case, simulate one deterministic scenario, and
/// assert the commissioning window gates the generation, the pre-entry decision
/// delivers at `entry`, and the ring buffer drains after `exit`.
#[test]
fn anticipated_commissioning_window_gates_simulation_output() {
    let dir = case_dir();
    let (mut setup, _config) = build_setup(&dir);

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
    assert!(
        outcome.result.iterations >= 1,
        "training must run at least one iteration",
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
        .expect("simulate must not return Err (a windowed anticipated thermal must stay feasible)");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");
    assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        N_STAGES,
        "one record per study stage"
    );

    // (i) GENERATION == 0 at every stage outside [entry, exit).
    for stage in 0..N_STAGES {
        if !(ENTRY..EXIT).contains(&stage) {
            let g = t1_at(scenario, stage).generation_mw;
            assert!(
                g.abs() <= TOL,
                "stage {stage} is outside the operation window [{ENTRY}, {EXIT}): \
                 T1 generation must be 0, got {g}",
            );
        }
    }

    // (ii) Generation present at the first operating stage `entry`. This proves
    // the pre-entry decision at `entry − K` delivered: the only way T1 can
    // generate at stage `entry` is for the commitment placed K stages earlier to
    // have matured here (the generation column is in-window, the decision column
    // at `entry` itself delivers at `entry + K`, outside the window).
    let g_entry = t1_at(scenario, ENTRY).generation_mw;
    assert!(
        g_entry > TOL,
        "T1 must generate at the first operating stage {ENTRY} (the pre-entry \
         decision at stage {} delivered here); got {g_entry}",
        ENTRY - K,
    );

    // (iii) The ring buffer drains to 0 within K stages after exit. The committed
    // MW (slot 0 of the ring buffer) is the matured commitment; after the last
    // in-window delivery (stage EXIT-1 = 3) no new in-window decision is made, so
    // by stage EXIT + K - 1 = 5 the buffer has shifted all residual commitments
    // out and reads 0.
    let drain_stage = EXIT + K - 1;
    let committed_drain = t1_at(scenario, drain_stage)
        .anticipated_committed_mw
        .expect("anticipated thermal must report committed MW");
    assert!(
        committed_drain.abs() <= TOL,
        "the ring buffer must drain to 0 within K={K} stages after exit={EXIT}: \
         committed MW at stage {drain_stage} must be 0, got {committed_drain}",
    );

    // The decision column is inactive at every post-exit / horizon-edge stage:
    // the simulation read returns None there (the column is pinned to [0, 0]).
    for stage in ENTRY..N_STAGES {
        let decision = t1_at(scenario, stage).anticipated_decision_mw;
        assert!(
            decision.is_none(),
            "T1 decision must be inactive (None) at stage {stage} (delivery at \
             {} is outside [{ENTRY}, {EXIT}) or beyond the horizon); got {decision:?}",
            stage + K,
        );
    }
    // Conversely, the pre-entry decision stages (0 and 1, delivering at 2 and 3)
    // are active and report a decision value.
    for stage in 0..ENTRY {
        let decision = t1_at(scenario, stage).anticipated_decision_mw;
        assert!(
            decision.is_some(),
            "T1 decision must be active at the pre-entry stage {stage} (delivers \
             at {} ∈ [{ENTRY}, {EXIT})); got None",
            stage + K,
        );
    }
}

/// Warm-start regression: the windowed anticipated study must reconstruct every
/// stored basis with zero rejections across the dormancy boundary.
///
/// The relocated `anticipated_state_out` column lives in the stage-invariant
/// state region; the commissioning window toggles which stages emit an active
/// `anticipated_state_out_def` row, so the per-stage row/column counts change at
/// the entry/exit boundaries. `reconstruct_basis` matches stored cut rows to
/// current LP rows by `CutPool` slot identity (never by absolute index), so the
/// dormancy-driven count change must remain transparent to the warm start.
/// `basis_consistency_failures` counts every rejected offered basis; it must stay
/// at zero.
#[test]
fn anticipated_commissioning_warm_start_zero_basis_rejections() {
    let dir = case_dir();
    let (mut setup, _config) = build_setup(&dir);
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
    let result = &outcome.result;
    // Warm-start reconstruction runs on every iteration after the first, so the
    // regression needs at least two iterations to exercise `reconstruct_basis`
    // across the dormancy boundary. The shipped config's iteration limit governs
    // the exact count; a deterministic case may converge before the cap.
    assert!(
        result.iterations >= 2,
        "warm-start regression needs >= 2 iterations to exercise reconstruct_basis \
         on iteration 2+, got {}",
        result.iterations,
    );

    let total_rejections =
        SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
            .basis_consistency_failures;
    assert_eq!(
        total_rejections, 0,
        "windowed anticipated warm-start: expected 0 basis rejections across the \
         dormancy boundary, got {total_rejections} (reconstruct_basis must match \
         cut rows by CutPool slot identity, not by absolute index)",
    );
}
