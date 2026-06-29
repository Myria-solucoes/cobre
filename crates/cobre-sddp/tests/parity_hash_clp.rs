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

use cobre_core::TrainingEvent;
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use cobre_solver::clp::ClpSolver;

mod common;
use common::StubComm;

use crate::common::parity_hash::compute_parity_hash;

// ---------------------------------------------------------------------------
// Stub communicator (single-rank)
// ---------------------------------------------------------------------------

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

/// Map a D-case label (e.g. `"D06"`) to its fixture directory path.
fn case_dir(label: &str) -> std::path::PathBuf {
    let suffix = match label {
        "D06" => "d06-fpha-variable-head",
        "D15" => "d15-non-controllable-source",
        "D30" => "d30-multi-resolution-monthly-quarterly",
        "D34" => "d34-anticipated-varying-blocks",
        "D41" => "d41-energy-contracts",
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

    let prep_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");
    let pr = prepare_stochastic(system, &dir, &config, 42, &prep_source)
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

    let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(&mut solver, &comm, 1, ClpSolver::new, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: expected no training error"
    );
    let result = outcome.result;

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

    let hash = compute_parity_hash(&setup, scenario_results);

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
fn parity_hash_d06() {
    run_case("D06");
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
fn parity_hash_d30() {
    run_case("D30");
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
fn parity_hash_d41() {
    run_case("D41");
}
