//! Terminal-stage boundary cuts: a consumer study loads boundary cuts from a
//! source checkpoint; they must inject into the terminal pool and the lower
//! bound must not degrade versus the same study without them.

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
use cobre_io::output::policy::write_policy_checkpoint;
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

fn d01_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d01-thermal-dispatch")
}

/// Write a policy checkpoint to `policy_dir` from the given setup and training result.
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
    let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
    let stage_cuts =
        build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices, &stage_manifests);
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

fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> (StudySetup, cobre_core::System) {
    let system = cobre_io::load_case(case_dir).expect("load_case");
    let prep = prepare_stochastic(system, case_dir, config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic");
    let hydro_models =
        prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
    let setup = StudySetup::new(&prep.system, config, prep.stochastic, hydro_models)
        .expect("StudySetup::new");
    (setup, prep.system)
}

#[test]
fn boundary_cuts_improve_terminal_stage_objective() {
    let case_dir = d01_case_dir();
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config");

    // 5 iterations for speed.
    let mut config_5iter = config.clone();
    config_5iter.training.stopping_rules =
        Some(vec![cobre_io::config::StoppingRuleConfig::IterationLimit {
            limit: 5,
        }]);

    // Run A: source study, produces the checkpoint.
    let (mut setup_a, _system_a) = build_setup(&case_dir, &config_5iter);
    let comm = StubComm;
    let mut solver_a = ActiveSolver::new().expect("solver");
    let outcome_a = setup_a
        .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
        .expect("train A");
    assert!(outcome_a.error.is_none());

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let source_policy_dir = tmpdir.path().join("source_policy");
    write_test_checkpoint(&source_policy_dir, &setup_a, &outcome_a.result, 42);

    let num_stages = setup_a.fcf.pools.len();
    let source_stage = (num_stages - 2) as u32; // terminal stage has no backward-pass cuts

    // Run B: consumer baseline, no boundary cuts.
    let (mut setup_b, _system_b) = build_setup(&case_dir, &config_5iter);
    let mut solver_b = ActiveSolver::new().expect("solver");
    let outcome_b = setup_b
        .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
        .expect("train B");
    assert!(outcome_b.error.is_none());
    let lb_no_boundary = outcome_b.result.final_lb;

    // Run C: consumer with boundary cuts.
    let (mut setup_c, system_c) = build_setup(&case_dir, &config_5iter);
    let state_dim = setup_c.fcf.state_dimension as u32;
    let current_manifest = setup_c.build_terminal_entity_manifest(&system_c);
    let mut warnings: Vec<String> = Vec::new();
    let boundary_records = cobre_sddp::load_boundary_cuts(
        &source_policy_dir,
        source_stage,
        state_dim,
        &current_manifest,
        &mut |msg| warnings.push(msg.to_string()),
    )
    .expect("load_boundary_cuts");
    assert!(
        !boundary_records.is_empty(),
        "source stage must have cuts after training"
    );
    cobre_sddp::inject_boundary_cuts(&mut setup_c, &boundary_records);

    let terminal_pool = &setup_c.fcf.pools[num_stages - 1];
    assert!(
        terminal_pool.warm_start_count > 0,
        "terminal pool must have boundary cuts"
    );
    assert!(
        terminal_pool.populated_count >= terminal_pool.warm_start_count as usize,
        "populated_count must include boundary cuts"
    );

    let mut solver_c = ActiveSolver::new().expect("solver");
    let outcome_c = setup_c
        .train(&mut solver_c, &comm, 1, ActiveSolver::new, None, None)
        .expect("train C");
    assert!(outcome_c.error.is_none());
    let lb_with_boundary = outcome_c.result.final_lb;

    // The boundary-cut contract: valid cuts only tighten, never degrade, the LB.
    assert!(
        lb_with_boundary >= lb_no_boundary - 1e-6,
        "boundary LB ({lb_with_boundary}) must be >= baseline LB ({lb_no_boundary})"
    );
}
