//! Integration test for simulation-only round-trip: the FCF reconstructed from
//! a written-then-reloaded policy checkpoint must evaluate identically.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::Path;

use cobre_core::scenario::ScenarioSource;
use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
use cobre_sddp::{
    FutureCostFunction, StudySetup, build_basis_cache_from_checkpoint,
    hydro_models::prepare_hydro_models,
    policy_export::{
        build_active_indices, build_stage_basis_records, build_stage_cut_records,
        build_stage_cuts_payloads, convert_basis_cache,
    },
    setup::prepare_stochastic,
};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

#[test]
fn simulation_only_fcf_round_trip() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let training_result = outcome.result;

    let original_active_cuts = setup.fcf.total_active_cuts();
    assert!(original_active_cuts > 0, "training should produce cuts");

    let n_stages = setup.stage_data.stage_templates.templates.len();
    let state_dim = setup.fcf.state_dimension;

    let test_state: Vec<f64> = vec![50.0; state_dim];
    let mut original_evals = Vec::with_capacity(n_stages);
    for stage in 0..n_stages {
        original_evals.push(setup.fcf.evaluate_at_state(stage, &test_state));
    }

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");

    let fcf = &setup.fcf;
    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
    let stage_cuts =
        build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices, &stage_manifests);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(&training_result);
    let stage_bases =
        build_stage_basis_records(fcf, &training_result, &basis_col_u8, &basis_row_u8);

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let metadata = cobre_io::PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-29T00:00:00Z".to_string(),
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dim as u32,
        num_stages: n_stages as u32,
        max_iterations: setup.loop_params.max_iterations as u32,
        forward_passes: setup.loop_params.forward_passes,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: 42,
        total_visited_states: 0,
    };

    write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
        .expect("write checkpoint");

    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

    assert_eq!(
        checkpoint.metadata.state_dimension, state_dim as u32,
        "state_dimension must round-trip"
    );
    assert_eq!(
        checkpoint.metadata.num_stages, n_stages as u32,
        "num_stages must round-trip"
    );
    assert_eq!(
        checkpoint.stage_cuts.len(),
        n_stages,
        "stage_cuts count must match"
    );

    let loaded_fcf =
        FutureCostFunction::from_deserialized(&checkpoint.stage_cuts).expect("from_deserialized");

    assert_eq!(
        loaded_fcf.total_active_cuts(),
        original_active_cuts,
        "active cut count must match after round-trip"
    );

    for (stage, &expected_eval) in original_evals.iter().enumerate().take(n_stages) {
        let loaded_eval = loaded_fcf.evaluate_at_state(stage, &test_state);
        assert_eq!(
            loaded_eval, expected_eval,
            "evaluate_at_state mismatch at stage {stage}"
        );
    }

    let loaded_basis_cache = build_basis_cache_from_checkpoint(
        n_stages,
        &checkpoint.stage_bases,
        &checkpoint.stage_cuts,
    );
    assert_eq!(
        loaded_basis_cache.len(),
        n_stages,
        "basis cache length must match"
    );
    let has_basis = loaded_basis_cache.iter().any(Option::is_some);
    assert!(has_basis, "at least one stage should have basis data");
}
