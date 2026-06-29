//! End-to-end integration test for the multi-resolution stage-transition pipeline
//! on the D30 case (6 monthly stages followed by 4 quarterly stages).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_core::temporal::SeasonCycleType;
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn d30_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d30-multi-resolution-monthly-quarterly")
}

/// Uses `config.training_scenario_source()`, not [`ScenarioSource::default()`]:
/// the default silently falls back to `InSample` and bypasses the PAR(1) pipeline.
fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> StudySetup {
    let system = cobre_io::load_case(case_dir).expect("load_case");
    let config_path = case_dir.join("config.json");
    let source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source");
    let prep =
        prepare_stochastic(system, case_dir, config, 42, &source).expect("prepare_stochastic");
    let hydro_models =
        prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
    StudySetup::new(&prep.system, config, prep.stochastic, hydro_models).expect("StudySetup::new")
}

// ---------------------------------------------------------------------------
// Test: structural properties, training, and simulation
// ---------------------------------------------------------------------------

#[test]
fn multi_resolution_structural_properties_and_training() {
    let case_dir = d30_case_dir();
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config");

    // ── 1. Structural properties after load ──────────────────────────────────

    let system = cobre_io::load_case(&case_dir).expect("load_case D30");

    // Pre-study stages carry negative IDs; the study count is the non-negative set.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    assert_eq!(
        study_stages.len(),
        10,
        "D30: expected 10 study stages; got {}",
        study_stages.len()
    );

    for (t, stage) in study_stages.iter().enumerate().take(6) {
        assert_eq!(
            stage.scenario_config.branching_factor, 5,
            "D30: stage {t} (monthly) must have num_scenarios == 5"
        );
    }

    for (t, stage) in study_stages.iter().enumerate().skip(6) {
        assert_eq!(
            stage.scenario_config.branching_factor, 5,
            "D30: stage {t} (quarterly) must have num_scenarios == 5"
        );
    }

    let season_map = system
        .policy_graph()
        .season_map
        .as_ref()
        .expect("D30 must have season_map");
    assert_eq!(
        season_map.cycle_type,
        SeasonCycleType::Custom,
        "D30: season map must have Custom cycle type"
    );
    assert_eq!(
        season_map.seasons.len(),
        16,
        "D30: Custom season map must have 16 seasons (12 monthly + 4 quarterly); got {}",
        season_map.seasons.len()
    );

    for expected_id in 0..12_usize {
        assert!(
            season_map.seasons.iter().any(|s| s.id == expected_id),
            "D30: monthly season id {expected_id} must be present in the season map"
        );
    }

    for expected_id in 12..16_usize {
        assert!(
            season_map.seasons.iter().any(|s| s.id == expected_id),
            "D30: quarterly season id {expected_id} must be present in the season map"
        );
    }

    // ── 2. Build setup and verify noise group IDs ─────────────────────────────

    let mut setup = build_setup(&case_dir, &config);

    let groups = &setup.stage_data.noise_group_ids;
    assert_eq!(
        groups.len(),
        10,
        "D30: noise_group_ids must have 10 elements (one per study stage); got {}",
        groups.len()
    );

    // Monthly stages 0-5 each pair a unique season_id with a unique year, so each
    // gets its own group ID — hence 6 distinct values.
    let monthly_groups: Vec<u32> = groups[..6].to_vec();
    let mut unique_monthly: Vec<u32> = monthly_groups.clone();
    unique_monthly.sort_unstable();
    unique_monthly.dedup();
    assert_eq!(
        unique_monthly.len(),
        6,
        "D30: monthly stages 0-5 must have 6 distinct noise group IDs; got {monthly_groups:?}"
    );

    // ── 3. Downstream lag transition fields ───────────────────────────────────

    let stage_ctx = setup.stage_ctx();
    let lag_transitions = stage_ctx.stage_lag_transitions;

    assert_eq!(
        lag_transitions.len(),
        10,
        "D30: stage_lag_transitions must have 10 entries (one per study stage); got {}",
        lag_transitions.len()
    );

    // downstream_par_order == 1 → the pre-transition window is the 1*3 = 3 monthly
    // stages (3, 4, 5) immediately before the quarterly boundary.
    for (t, transition) in lag_transitions.iter().enumerate().take(6).skip(3) {
        assert!(
            transition.accumulate_downstream,
            "D30: monthly stage {t} (index in pre-transition window) must have              accumulate_downstream == true; got false. Full transition: {transition:?}"
        );
    }

    // Stages 0-2 are outside the window: no downstream accumulation.
    for (t, transition) in lag_transitions.iter().enumerate().take(3) {
        assert!(
            !transition.accumulate_downstream,
            "D30: monthly stage {t} (outside pre-transition window) must have              accumulate_downstream == false; got true. Full transition: {transition:?}"
        );
    }

    assert!(
        lag_transitions[6].rebuild_from_downstream,
        "D30: first quarterly stage (index 6) must have rebuild_from_downstream == true; \
         got false. Full transition: {:?}",
        lag_transitions[6]
    );

    for (t, transition) in lag_transitions.iter().enumerate().skip(7) {
        assert!(
            !transition.rebuild_from_downstream,
            "D30: quarterly stage {t} (not first) must have rebuild_from_downstream == false;              got true. Full transition: {transition:?}"
        );
    }

    // ── 4. Train ──────────────────────────────────────────────────────────────

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("solver");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("D30: train must not return Err");

    assert!(
        outcome.error.is_none(),
        "D30: training outcome must have no error; got: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations >= 1,
        "D30: must complete at least 1 iteration; got {}",
        outcome.result.iterations
    );
    assert!(
        outcome.result.final_lb > 0.0,
        "D30: lower bound must be positive; got {}",
        outcome.result.final_lb
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "D30: lower bound must be finite; got {}",
        outcome.result.final_lb
    );

    // ── 5. Simulate ───────────────────────────────────────────────────────────

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("D30: workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    // Drain the channel in a background thread to avoid blocking simulate().
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
        .expect("D30: simulate must not return Err");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(
        scenario_results.len(),
        1,
        "D30: expected 1 simulation scenario result; got {}",
        scenario_results.len()
    );
}
