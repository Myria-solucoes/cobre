//! Warm-start training: a run resumed from saved cuts reuses them
//! (`warm_start_count > 0`) and yields a lower bound no worse than fresh training.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
use cobre_sddp::{
    FutureCostFunction, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
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

fn d01_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch")
}

fn write_test_checkpoint(
    policy_dir: &Path,
    setup: &StudySetup,
    result: &cobre_sddp::TrainingResult,
    seed: u64,
) {
    use cobre_sddp::policy_export::{
        build_active_indices, build_stage_basis_records, build_stage_cut_records,
        build_stage_cuts_payloads, convert_basis_cache,
    };
    let fcf = &setup.fcf;
    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);
    let (basis_col, basis_row) = convert_basis_cache(result);
    let stage_bases = build_stage_basis_records(fcf, result, &basis_col, &basis_row);
    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let metadata = cobre_io::PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-29T00:00:00Z".to_string(),
        completed_iterations: result.iterations as u32,
        final_lower_bound: result.final_lb,
        best_upper_bound: Some(result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: fcf.pools.len() as u32,
        max_iterations: setup.loop_params.max_iterations as u32,
        forward_passes: setup.loop_params.forward_passes,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: seed,
        total_visited_states: 0,
    };
    write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
        .expect("write checkpoint");
}

fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> StudySetup {
    let system = cobre_io::load_case(case_dir).expect("load_case");
    let prep = prepare_stochastic(system, case_dir, config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic");
    let hydro_models =
        prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
    StudySetup::new(&prep.system, config, prep.stochastic, hydro_models).expect("StudySetup::new")
}

#[test]
fn resume_training_from_checkpoint() {
    let case_dir = d01_case_dir();
    let config_path = case_dir.join("config.json");
    let config_full = cobre_io::parse_config(&config_path).expect("config must parse");

    let mut config_phase1 = config_full.clone();
    config_phase1.training.stopping_rules =
        Some(vec![cobre_io::config::StoppingRuleConfig::IterationLimit {
            limit: 5,
        }]);

    let mut setup_phase1 = build_setup(&case_dir, &config_phase1);
    let comm = StubComm;
    let mut solver_phase1 = ActiveSolver::new().expect("ActiveSolver");
    let outcome_phase1 = setup_phase1
        .train(&mut solver_phase1, &comm, 1, ActiveSolver::new, None, None)
        .expect("train phase1");
    assert!(outcome_phase1.error.is_none());
    let result_phase1 = outcome_phase1.result;
    assert_eq!(
        result_phase1.iterations, 5,
        "phase 1 must complete exactly 5 iterations"
    );
    let lb_phase1 = result_phase1.final_lb;

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");
    write_test_checkpoint(&policy_dir, &setup_phase1, &result_phase1, 42);

    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");
    let mut setup_phase2 = build_setup(&case_dir, &config_full);

    let warm_fcf = FutureCostFunction::new_with_warm_start(
        &checkpoint.stage_cuts,
        setup_phase2.loop_params.forward_passes,
        setup_phase2.loop_params.max_iterations.saturating_add(1),
    )
    .expect("warm-start FCF");
    setup_phase2.replace_fcf(warm_fcf);
    setup_phase2.set_start_iteration(u64::from(checkpoint.metadata.completed_iterations));

    let mut solver_phase2 = ActiveSolver::new().expect("ActiveSolver");
    let outcome_phase2 = setup_phase2
        .train(&mut solver_phase2, &comm, 1, ActiveSolver::new, None, None)
        .expect("train phase2");
    assert!(outcome_phase2.error.is_none());
    let result_phase2 = outcome_phase2.result;

    assert_eq!(
        result_phase2.iterations, 10,
        "resumed run must report 10 total iterations (not 5 delta)"
    );

    assert!(
        result_phase2.final_lb >= lb_phase1 - 1e-6,
        "resumed LB ({}) must be >= phase-1 LB ({})",
        result_phase2.final_lb,
        lb_phase1
    );
}

#[test]
fn warm_start_training_preserves_cuts_and_trains_further() {
    let case_dir = d01_case_dir();
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let mut setup_fresh = build_setup(&case_dir, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver");
    let fresh_outcome = setup_fresh
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");
    assert!(fresh_outcome.error.is_none());
    let fresh_result = fresh_outcome.result;
    let fresh_lb = fresh_result.final_lb;
    let fresh_active = setup_fresh.fcf.total_active_cuts();

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let policy_dir = tmpdir.path().join("policy");
    write_test_checkpoint(&policy_dir, &setup_fresh, &fresh_result, 42);

    let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");
    let mut setup_warm = build_setup(&case_dir, &config);

    // Capacity must match from_broadcast_params's saturating_add(1) or the slot
    // layout diverges.
    let warm_fcf = FutureCostFunction::new_with_warm_start(
        &checkpoint.stage_cuts,
        setup_warm.loop_params.forward_passes,
        setup_warm.loop_params.max_iterations.saturating_add(1),
    )
    .expect("warm-start FCF");
    setup_warm.replace_fcf(warm_fcf);

    let warm_start_count = setup_warm.fcf.pools[0].warm_start_count;
    assert!(warm_start_count > 0, "warm_start_count should be > 0");
    assert_eq!(
        setup_warm.fcf.total_active_cuts(),
        fresh_active,
        "warm-start FCF should have same active cuts as fresh training"
    );

    let mut solver_warm = ActiveSolver::new().expect("ActiveSolver");
    let warm_outcome = setup_warm
        .train(&mut solver_warm, &comm, 1, ActiveSolver::new, None, None)
        .expect("warm-start train");
    assert!(warm_outcome.error.is_none());
    let warm_result = warm_outcome.result;

    assert!(
        warm_result.final_lb >= fresh_lb - 1e-6,
        "warm-start LB ({}) should be >= fresh LB ({})",
        warm_result.final_lb,
        fresh_lb
    );

    let total_active_after = setup_warm.fcf.total_active_cuts();
    assert!(
        total_active_after > fresh_active,
        "warm-start training should produce more total cuts"
    );
}
