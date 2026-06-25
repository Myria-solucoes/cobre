//! CLP parity-hash harness for the deterministic cases enumerated in `case_dir`.
//!
//! Drives the CLP backend over the same semantic-field whitelist owned by
//! [`compute_parity_hash`], digesting each case's train + simulate output. With
//! `COBRE_PARITY_REGEN=1` the test **writes** the baseline; otherwise it
//! **verifies** against the committed baseline.
//!
//! CLP's simplex legitimately reaches **different-but-valid** optima, so its
//! digests differ from other backends': the CLP baselines are an independent set
//! under `tests/fixtures/parity_baselines_clp/`. The committed `*.sha256` files
//! are machine/CI-canonical — regenerated on the canonical environment to assert
//! run-to-run reproducibility there, not bit-for-bit reproduction on arbitrary
//! machines.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
#![cfg(feature = "clp")]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{TrainingEvent, scenario::ScenarioSource};
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use cobre_solver::clp::ClpSolver;

mod common;

use crate::common::parity_hash::compute_parity_hash;

// ---------------------------------------------------------------------------
// Stub communicator (single-rank)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Baseline file path
// ---------------------------------------------------------------------------

fn baseline_path(case: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/parity_baselines_clp")
        .join(format!("{case}.sha256"))
}

// ---------------------------------------------------------------------------
// Baseline read/write
// ---------------------------------------------------------------------------

/// Read the baseline file for `case` and compare to `hash`, or write the
/// baseline when `COBRE_PARITY_REGEN=1`.
///
/// Returns `Ok(())` on match or successful write; `Err(msg)` on mismatch or
/// missing baseline.
fn read_or_regen_baseline(case: &str, hash: &str) -> Result<(), String> {
    let path = baseline_path(case);

    if std::env::var("COBRE_PARITY_REGEN").as_deref() == Ok("1") {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create baseline dir: {e}"))?;
        }
        std::fs::write(&path, format!("{hash}\n"))
            .map_err(|e| format!("cannot write baseline for {case}: {e}"))?;
        eprintln!("REGEN: wrote baseline for {case}: {hash}");
        return Ok(());
    }

    if !path.exists() {
        return Err(format!(
            "baseline file for {case} is missing at {}; \
             run with COBRE_PARITY_REGEN=1 to generate it",
            path.display()
        ));
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read baseline for {case}: {e}"))?;
    let expected = raw.trim();

    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("baseline file {case} is malformed: {expected:?}"));
    }

    if expected == hash {
        eprintln!("OK: parity hash for {case} matched baseline {hash}");
        Ok(())
    } else {
        Err(format!(
            "parity hash mismatch for {case}:\n  expected (baseline): {expected}\n  actual:              {hash}"
        ))
    }
}

// ---------------------------------------------------------------------------
// Case runner
// ---------------------------------------------------------------------------

/// Map a D-case label (e.g. `"D01"`) to its fixture directory path.
fn case_dir(label: &str) -> std::path::PathBuf {
    let suffix = match label {
        "D01" => "d01-thermal-dispatch",
        "D02" => "d02-single-hydro",
        "D03" => "d03-two-hydro-cascade",
        "D04" => "d04-transmission",
        "D05" => "d05-fpha-constant-head",
        "D06" => "d06-fpha-variable-head",
        "D07" => "d07-fpha-computed",
        "D08" => "d08-evaporation",
        "D09" => "d09-multi-deficit",
        "D10" => "d10-inflow-nonnegativity",
        "D11" => "d11-water-withdrawal",
        "D13" => "d13-generic-constraint",
        "D14" => "d14-block-factors",
        "D15" => "d15-non-controllable-source",
        "D17" => "d17-evaporation-mixed-sign",
        "D18" => "d18-ncs-commissioning-window",
        "D31" => "d31-backwater-reference-volume",
        "D32" => "d32-reversible-plant",
        "D33" => "d33-per-stage-block-counts",
        "D34" => "d34-anticipated-varying-blocks",
        "D35" => "d35-pumping-commissioning",
        "D36" => "d36-thermal-line-commissioning",
        "D37" => "d37-anticipated-commissioning",
        "D38" => "d38-dead-volume-filling",
        "D39" => "d39-prefilling-upstream-of-filling",
        "D40" => "d40-filling-cascade",
        other => panic!("unknown case label: {other}"),
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join(suffix)
}

/// Run the full train + simulate pipeline for a D-case and assert parity.
fn run_case(label: &str) {
    let dir = case_dir(label);
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
    let mut solver = ClpSolver::new().expect("ClpSolver::new must succeed");

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(&mut solver, &comm, 1, ClpSolver::new, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: expected no training error"
    );
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
        .create_workspace_pool(&comm, 1, ClpSolver::new)
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

    let hash = compute_parity_hash(&convergence_updates, &setup, scenario_results);

    read_or_regen_baseline(label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
}

// ---------------------------------------------------------------------------
// Individual test functions — one per case enumerated in `case_dir`
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d01() {
    run_case("D01");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d02() {
    run_case("D02");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d03() {
    run_case("D03");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d04() {
    run_case("D04");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d05() {
    run_case("D05");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d06() {
    run_case("D06");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d07() {
    run_case("D07");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d08() {
    run_case("D08");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d09() {
    run_case("D09");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d10() {
    run_case("D10");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d11() {
    run_case("D11");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d13() {
    run_case("D13");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d14() {
    run_case("D14");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d15() {
    run_case("D15");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d17() {
    run_case("D17");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d18() {
    run_case("D18");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d31() {
    run_case("D31");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d32() {
    run_case("D32");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d33() {
    run_case("D33");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d34() {
    run_case("D34");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d35() {
    run_case("D35");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d36() {
    run_case("D36");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d37() {
    run_case("D37");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d38() {
    run_case("D38");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d39() {
    run_case("D39");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn parity_hash_d40() {
    run_case("D40");
}
