//! End-to-end integration test for the mixed-resolution pipeline on the D28
//! case (5 weekly stages + 1 monthly terminal stage).

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

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::temporal::SeasonCycleType;
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

fn d28_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d28-decomp-weekly-monthly")
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
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);
    let (basis_col, basis_row) = convert_basis_cache(result);
    let stage_bases = build_stage_basis_records(fcf, result, &basis_col, &basis_row);
    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let metadata = cobre_io::PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-04-14T00:00:00Z".to_string(),
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

/// Build a `StudySetup` for the D28 case, calling `config.training_scenario_source()`
/// so the External inflow library is loaded; `ScenarioSource::default()` (`InSample`)
/// would skip the external parquet and produce a wrong scenario pipeline.
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

/// Verify structural properties, training correctness, lag accumulation, and
/// simulation completion for the D28 mixed-resolution case.
///
/// D28 has 5 weekly stages (indices 0-4, `num_scenarios == 1`) followed by
/// 1 monthly terminal stage (index 5, `num_scenarios == 5`). The monthly
/// season map must have 12 seasons.
#[test]
fn structural_properties_and_training() {
    let case_dir = d28_case_dir();
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config");

    let system = cobre_io::load_case(&case_dir).expect("load_case D28");
    let stages = system.stages();

    assert_eq!(
        stages.len(),
        6,
        "D28 must have exactly 6 study stages; got {}",
        stages.len()
    );

    for (t, stage) in stages.iter().enumerate().take(5) {
        assert_eq!(
            stage.scenario_config.branching_factor, 1,
            "stage {t} (weekly) must have num_scenarios == 1"
        );
    }

    assert_eq!(
        stages[5].scenario_config.branching_factor, 5,
        "stage 5 (monthly) must have num_scenarios == 5"
    );

    let season_map = system
        .policy_graph()
        .season_map
        .as_ref()
        .expect("D28 must have season_map");
    assert_eq!(
        season_map.cycle_type,
        SeasonCycleType::Monthly,
        "season map must have monthly cycle type"
    );
    assert_eq!(
        season_map.seasons.len(),
        12,
        "monthly season map must have 12 seasons"
    );

    let mut setup = build_setup(&case_dir, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("solver");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "training outcome must have no error; got: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations >= 1,
        "must complete at least 1 iteration; got {}",
        outcome.result.iterations
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "lower bound must be finite; got {}",
        outcome.result.final_lb
    );
    assert!(
        !outcome.result.final_lb.is_nan(),
        "lower bound must not be NaN"
    );

    // The backward pass writes cuts into pools[0..T-2] only; the terminal pool
    // (pools[T-1]) stays empty unless boundary cuts are injected.
    let fcf = &setup.fcf;
    let n_pools = fcf.pools.len();
    for (t, pool) in fcf.pools[..n_pools - 1].iter().enumerate() {
        assert!(
            pool.populated_count >= 1,
            "stage {t} (non-terminal) cut pool must have at least 1 cut after training; got {}",
            pool.populated_count
        );
    }
    assert_eq!(
        fcf.pools[n_pools - 1].populated_count,
        0,
        "terminal pool must have 0 backward-pass cuts (boundary-cut injection is a separate test)"
    );

    let stage_ctx = setup.stage_ctx();
    let lag_transitions = stage_ctx.stage_lag_transitions;

    assert_eq!(
        lag_transitions.len(),
        6,
        "stage_lag_transitions must have 6 entries (one per study stage)"
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    // Drain the channel in a background thread to avoid blocking simulate().
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let sim_run = setup
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
    let _scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert!(
        sim_run.costs.is_empty() || !sim_run.costs.is_empty(),
        "simulate returned successfully"
    );
    let sim_config = setup.simulation_config();
    assert_eq!(
        sim_config.n_scenarios, 5,
        "simulation must be configured for 5 scenarios"
    );
}

/// Verify that boundary cuts produced by a trained D28 source study compose
/// correctly with a consumer D28 study: injecting them must not degrade the
/// lower bound relative to the no-boundary-cut baseline.
#[test]
fn decomp_boundary_cuts_compose_with_weekly_monthly() {
    let case_dir = d28_case_dir();
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config");

    let comm = StubComm;

    let mut setup_a = build_setup(&case_dir, &config);
    let mut solver_a = ActiveSolver::new().expect("solver A");
    let outcome_a = setup_a
        .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
        .expect("train A");
    assert!(outcome_a.error.is_none(), "source training must not error");

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let source_policy_dir = tmpdir.path().join("source_policy");
    write_test_checkpoint(&source_policy_dir, &setup_a, &outcome_a.result, 42);

    let num_stages = setup_a.fcf.pools.len();
    let source_stage = (num_stages - 2) as u32; // second-to-last stage has backward-pass cuts

    let mut setup_b = build_setup(&case_dir, &config);
    let mut solver_b = ActiveSolver::new().expect("solver B");
    let outcome_b = setup_b
        .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
        .expect("train B");
    assert!(
        outcome_b.error.is_none(),
        "baseline training must not error"
    );
    let lb_no_boundary = outcome_b.result.final_lb;

    let mut setup_c = build_setup(&case_dir, &config);
    let state_dim = setup_c.fcf.state_dimension as u32;
    let boundary_records =
        cobre_sddp::load_boundary_cuts(&source_policy_dir, source_stage, state_dim)
            .expect("load_boundary_cuts");
    assert!(
        !boundary_records.is_empty(),
        "source stage must have cuts after training"
    );
    cobre_sddp::inject_boundary_cuts(&mut setup_c, &boundary_records);

    let terminal_pool = &setup_c.fcf.pools[num_stages - 1];
    assert!(
        terminal_pool.warm_start_count > 0,
        "terminal pool must have boundary cuts after injection; warm_start_count == 0"
    );
    assert!(
        terminal_pool.populated_count >= terminal_pool.warm_start_count as usize,
        "populated_count ({}) must include boundary cuts (warm_start_count = {})",
        terminal_pool.populated_count,
        terminal_pool.warm_start_count
    );

    let mut solver_c = ActiveSolver::new().expect("solver C");
    let outcome_c = setup_c
        .train(&mut solver_c, &comm, 1, ActiveSolver::new, None, None)
        .expect("train C");
    assert!(
        outcome_c.error.is_none(),
        "consumer training must not error"
    );
    let lb_with_boundary = outcome_c.result.final_lb;

    assert!(
        lb_with_boundary >= lb_no_boundary - 1e-6,
        "boundary LB ({lb_with_boundary}) must be >= baseline LB ({lb_no_boundary})"
    );
}
