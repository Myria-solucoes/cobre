//! Shared parity-hash computation for the integration test harnesses — the sole
//! owner of the hash whitelist and its byte layout.
//!
//! The hash is deterministic: every field is little-endian, stages ascending,
//! scenarios sorted by `scenario_id`, hydro/thermal records by
//! `(block_id, id)`. Fields 4–6 are not redundant with storage/dual: each is an
//! `n_blks`-dependent read kept specifically to surface an extraction/cost/base
//! bug a uniform-block case cannot detect — do not drop them as duplicate.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-stage, per-cut: `stage_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`
//! 2. Primal trajectory (`storage_final_hm3`) per scenario per stage.
//! 3. Dual trajectory (`water_value_per_hm3`) per scenario per stage.
//! 4. Per-block equipment (`spillage_m3s`) — base shifts off stage 0's block
//!    width under a non-uniform schedule (the simulation-extraction base bug).
//! 5. Cost breakdown (`spillage_cost`) — a `range_sum` whose base AND length
//!    shift under a non-uniform schedule (the cost-breakdown bug).
//! 6. Anticipated decision (`anticipated_decision_mw`) — base is the per-stage
//!    `thermal.end` (`n_blks`-dependent). `None` hashes as a 0-flag, so only
//!    anticipated cases (D34) move this field.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cobre_core::TrainingEvent;
use cobre_sddp::{
    SimulationScenarioResult, StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use sha2::{Digest, Sha256};

use super::StubComm;

/// Compute the SHA-256 parity hash over the module-doc whitelist.
pub fn compute_parity_hash(
    setup: &StudySetup,
    mut scenario_results: Vec<SimulationScenarioResult>,
) -> String {
    let mut hasher = Sha256::new();

    // Active cuts in ascending stage order, then active_cuts() slot order — fixed
    // iteration order is what makes the cut digest declaration-order-stable.
    let fcf = &setup.fcf;
    let num_stages = fcf.pools.len();
    for stage in 0..num_stages {
        for (_slot, intercept, coefficients) in fcf.active_cuts(stage) {
            hasher.update((stage as u32).to_le_bytes());
            hasher.update(intercept.to_le_bytes());
            hasher.update((coefficients.len() as u32).to_le_bytes());
            for &c in coefficients {
                hasher.update(c.to_le_bytes());
            }
        }
    }

    // Sort into canonical order so the digest is independent of how the pipeline
    // emitted scenarios/stages/records.
    scenario_results.sort_by_key(|r| r.scenario_id);

    for scenario in &mut scenario_results {
        scenario.stages.sort_by_key(|s| s.stage_id);

        for stage in &mut scenario.stages {
            stage.hydros.sort_by_key(|h| (h.block_id, h.hydro_id));

            let num_primals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_primals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.storage_final_hm3.to_le_bytes());
            }

            let num_duals = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_duals.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.water_value_per_hm3.to_le_bytes());
            }

            let num_equipment = stage.hydros.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_equipment.to_le_bytes());
            for h in &stage.hydros {
                hasher.update(h.spillage_m3s.to_le_bytes());
            }

            let num_costs = stage.costs.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_costs.to_le_bytes());
            for c in &stage.costs {
                hasher.update(c.spillage_cost.to_le_bytes());
            }

            stage.thermals.sort_by_key(|t| (t.block_id, t.thermal_id));
            let num_thermals = stage.thermals.len() as u32;
            hasher.update(stage.stage_id.to_le_bytes());
            hasher.update(num_thermals.to_le_bytes());
            for t in &stage.thermals {
                // Hash a presence flag + value so `None` maps to a fixed (0, 0.0):
                // dropping the flag would collide `None` with `Some(0.0)` and break
                // the encoding's injectivity.
                let (flag, value) = t.anticipated_decision_mw.map_or((0u8, 0.0), |v| (1u8, v));
                hasher.update(flag.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
        }
    }

    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Golden-case harness — shared by the per-backend `parity_hash_*` modules
// ---------------------------------------------------------------------------

/// Path to the `<case>.sha256` baseline under the given backend subdir of
/// `tests/fixtures/`.
fn baseline_path(baseline_subdir: &str, case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(baseline_subdir)
        .join(format!("{case}.sha256"))
}

/// Read the baseline for `case` and compare to `hash`, or write it when
/// `COBRE_PARITY_REGEN=1`.
///
/// Returns `Ok(())` on match or successful write; `Err(msg)` on mismatch or a
/// missing/malformed baseline.
fn read_or_regen_baseline(baseline_subdir: &str, case: &str, hash: &str) -> Result<(), String> {
    let path = baseline_path(baseline_subdir, case);

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

/// Map a golden-case label (e.g. `"D06"`) to its deterministic fixture directory.
///
/// The five labels are deliberately non-sequential: they are the
/// feature-combined subset of the contiguous `d01..` deterministic suite whose
/// union spans the cross-feature LP / scaling / cut / basis interactions that
/// hide dormant bugs (the tier-1 golden-parity contract). Each pins a distinct
/// piece of machinery:
///
/// - `D06` (`d06-fpha-variable-head`) — variable-head FPHA average-storage
///   coefficient on both incoming and outgoing storage columns.
/// - `D15` (`d15-non-controllable-source`) — NCS availability factor and the
///   lower-bound-patches-NCS contract.
/// - `D30` (`d30-multi-resolution-monthly-quarterly`) — multi-resolution
///   monthly/quarterly decomposition.
/// - `D34` (`d34-anticipated-varying-blocks`) — anticipated dispatch under
///   non-uniform per-stage block counts.
/// - `D41` (`d41-energy-contracts`) — energy import/export contract columns.
///
/// The simpler `d01..d05` single-feature cases are covered at the behavioral
/// tier (`deterministic.rs`); promoting a case into this golden set needs
/// justification.
fn case_dir(label: &str) -> PathBuf {
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

/// Run the full train + simulate pipeline for a golden case under the given LP
/// solver and assert its parity hash against the committed baseline (or write the
/// baseline when `COBRE_PARITY_REGEN=1`).
///
/// Generic over the solver so both backend `parity_hash_*` modules share one
/// body; `make_solver` supplies fresh worker solvers exactly as the production
/// training/simulation paths do. `baseline_subdir` selects the per-backend
/// baseline directory under `tests/fixtures/`.
pub fn run_golden_case<S, F>(baseline_subdir: &str, label: &str, make_solver: F)
where
    S: cobre_solver::SolverInterface<Profile = cobre_solver::ActiveProfile> + Send,
    F: Fn() -> Result<S, cobre_solver::SolverError> + Copy,
{
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
    let mut solver = make_solver().expect("solver construction must succeed");

    let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(&mut solver, &comm, 1, make_solver, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: expected no training error"
    );
    let result = outcome.result;

    let mut pool = setup
        .create_workspace_pool(&comm, 1, make_solver)
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
            result.frozen_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    let hash = compute_parity_hash(&setup, scenario_results);

    read_or_regen_baseline(baseline_subdir, label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
}
