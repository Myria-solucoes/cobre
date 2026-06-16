//! Self-reproducibility regression test.
//!
//! Runs the SDDP training + simulation pipeline **twice in the same process**
//! on the D02 single-hydro fixture and asserts that the parity hash of run-1
//! equals the parity hash of run-2.
//!
//! Distinct from `parity_hash_d01_d15.rs` (which compares against committed
//! SHA-256 baselines on disk) and `cut_subgradient_parity.rs` (which verifies
//! the KKT identity for a single isolated LP solve). This test guards against
//! future non-determinism vectors — HashMap iteration order, parallel
//! floating-point reductions, RNG re-seeding drift, scheduler ordering — that
//! would surface as a hash that drifts between consecutive runs of the same
//! `(seed, config, input)`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{TrainingEvent, scenario::ScenarioSource};
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use cobre_solver::ActiveSolver;

mod common;

use crate::common::parity_hash::compute_parity_hash;

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

fn d02_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic/d02-single-hydro")
}

/// Drive one full train + simulate pass and return the parity hash.
///
/// Mirrors the body of `run_case` in `parity_hash_d01_d15.rs` but skips the
/// baseline read/write: this test compares two in-process invocations against
/// each other, not against committed hashes.
fn run_d02_once() -> String {
    let dir = d02_case_dir();
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

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            1,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

    let mut convergence_updates: Vec<(u64, f64, f64, f64, f64)> = event_rx
        .into_iter()
        .filter_map(|ev| {
            if let TrainingEvent::ConvergenceUpdate {
                iteration,
                lower_bound,
                upper_bound,
                upper_bound_std,
                gap,
                ..
            } = ev
            {
                Some((iteration, lower_bound, upper_bound, upper_bound_std, gap))
            } else {
                None
            }
        })
        .collect();
    convergence_updates.sort_by_key(|&(iter, ..)| iter);

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
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    compute_parity_hash(&convergence_updates, &setup, scenario_results)
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn d02_self_reproducibility() {
    let hash_run1 = run_d02_once();
    let hash_run2 = run_d02_once();
    assert_eq!(
        hash_run1, hash_run2,
        "self-reproducibility violation on d02-single-hydro:\n  \
         run-1 hash: {hash_run1}\n  \
         run-2 hash: {hash_run2}"
    );
}
