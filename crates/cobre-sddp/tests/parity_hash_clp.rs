//! CLP parity hash harness for the deterministic cases enumerated in `case_dir`
//! (the D01–D17 core plus the D31 backwater reference-volume case; gaps in the
//! index reflect retired/absent cases).
//!
//! This harness drives the CLP backend and reads/writes its own baseline set
//! under `tests/fixtures/parity_baselines_clp/`. It shares the same field
//! whitelist and the same shared [`compute_parity_hash`] helper used by the
//! other backend's harness.
//!
//! The CLP simplex legitimately reaches **different-but-valid** optimal
//! vertices, so its semantic-field digests differ from those produced by other
//! LP backends. The CLP baselines are therefore an independent set, captured
//! and verified solely against this CLP-specific directory.
//!
//! The committed `*.sha256` files are machine/CI-canonical artifacts: they are
//! regenerated on the canonical environment and assert run-to-run
//! reproducibility there, not bit-for-bit reproduction on arbitrary machines.
//!
//! Computes a SHA-256 digest over a whitelist of semantic fields from each
//! case's training + simulation output. On first run with `COBRE_PARITY_REGEN=1`
//! the test **writes** the baseline files; on subsequent runs it **verifies**
//! against the committed baselines.
//!
//! ## Hash whitelist (in fixed order)
//!
//! 1. Per-iteration convergence data: `iteration_u64_le || lower_bound_f64_le
//!    || upper_bound_f64_le || upper_bound_std_f64_le || gap_f64_le`
//!    Captured from [`TrainingEvent::ConvergenceUpdate`] events (one per
//!    completed iteration, ordered 1..=N).
//!
//! 2. Per-stage, per-cut: `stage_u32_le || intercept_f64_le ||
//!    coefficient_count_u32_le || coefficient_f64_le[]`
//!    Iterated over stages 0..num_stages, then active cuts within each stage
//!    in the slot order reported by [`FutureCostFunction::active_cuts`].
//!
//! 3. Simulation primal trajectory per scenario per stage:
//!    `stage_u32_le || num_primals_u32_le || primal_f64_le[]`
//!    Scenarios sorted ascending by `scenario_id`; stages by `stage_id`.
//!    Primals = `storage_final_hm3` for each hydro at each stage, sorted by
//!    `(block_id, hydro_id)`.  For pure-thermal cases the primal vector is
//!    empty (`num_primals = 0`).
//!
//! 4. Simulation dual trajectory per scenario per stage:
//!    `stage_u32_le || num_duals_u32_le || dual_f64_le[]`
//!    Same ordering.  Duals = `water_value_per_hm3` for each hydro record,
//!    sorted by `(block_id, hydro_id)`.
//!
//! ## Field-name translation
//!
//! The generic "primal trajectory" and "dual trajectory" concepts map onto
//! concrete struct fields as follows:
//! - `SimulationHydroResult::storage_final_hm3`  → primal state variable
//! - `SimulationHydroResult::water_value_per_hm3` → dual of storage balance
//! - `TrainingEvent::ConvergenceUpdate::upper_bound_std` → `upper_bound_std_f64_le`
//!
//! ## Timing exclusion
//!
//! No field ending in `_ms`, containing `elapsed`, or containing `wall` is
//! included in the hash. Timing fields are allowed to drift between runs.

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
// Stub communicator (single-rank, copied from deterministic.rs)
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
        // Regeneration mode: write the baseline.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create baseline dir: {e}"))?;
        }
        std::fs::write(&path, format!("{hash}\n"))
            .map_err(|e| format!("cannot write baseline for {case}: {e}"))?;
        eprintln!("REGEN: wrote baseline for {case}: {hash}");
        return Ok(());
    }

    // Verification mode.
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

    // Validate the baseline is a well-formed 64-char hex string.
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
        // NCS commissioning window: NCS0 enters service at stage 1
        // (`entry_stage_id`) and NCS1 leaves at stage 2 (`exit_stage_id`), so the
        // active NCS set — and therefore the per-stage NCS column base — differs
        // across stages 0/1/2 instead of collapsing to a uniform set. Exercises
        // the per-stage `identify_active_ncs` path that a uniform-NCS case cannot.
        "D18" => "d18-ncs-commissioning-window",
        // Cascade case whose downstream plant declares a `reference_volume`,
        // shifting the upstream computed-FPHA plant's backwater family.
        "D31" => "d31-backwater-reference-volume",
        // Reversible plant: a single pumping station lifts water from the
        // downstream reservoir (H1) back up to the upstream one (H0) every
        // block. The `flow.min_m3s > 0` lower bound forces a non-degenerate
        // transfer (and the matching power draw) on every solve, so the pumping
        // column actually participates in the LP.
        "D32" => "d32-reversible-plant",
        // Per-stage varying block counts: stages declare `blocks` arrays of
        // different length (1 / 3 / 2), so each stage's per-block equipment
        // column stride (turbine/spillage/thermal/bus) diverges from stage 0's.
        // A uniform-block case (every stage shares one block count, as in D14)
        // cannot detect equipment columns read off the wrong stage's block
        // width, because all stages share the same stride. The `load_factors`
        // block-factor array is sized to each stage's own count, exercising the
        // per-stage block-indexed input path.
        "D33" => "d33-per-stage-block-counts",
        // Anticipated thermals under a per-stage-varying block schedule
        // (1 / 3 / 2). The cheap anticipated thermal (K=1) commits at stage 0
        // and stage 1, maturing at the interior delivery stages 1 (3 blocks)
        // and 2 (2 blocks) — both with block counts differing from stage 0's
        // single block. The matured-anticipated cut coefficient is therefore
        // harvested at an off-stage-0 block count, which is exactly the
        // condition that distinguishes the `anticipated_state_out` column
        // resolved through the stage-invariant state region (correct) from one
        // resolved through an `n_blks`-dependent control-region offset (the
        // class of bug a uniform-block anticipated case cannot detect, because
        // every stage shares the same per-block stride). Pairs anticipated
        // thermals (absent from D33) with the non-uniform block schedule
        // (absent from every anticipated test).
        "D34" => "d34-anticipated-varying-blocks",
        // Pumping commissioning gating: the reversible plant of D32 carries an
        // `entry_stage_id`/`exit_stage_id` window so it is active only at the
        // interior stage 1 (omitted at stages 0 and 2). Because `flow.min_m3s > 0`
        // forces a transfer on every active solve, gating removes that forced
        // transfer at the two omitted stages — a station active at every stage
        // (as in D32) cannot detect a window that is parsed but never applied to
        // the LP. The omitted stages contribute no pumping column, so their
        // storage trajectories and water values diverge from the always-active
        // D32 baseline.
        "D35" => "d35-pumping-commissioning",
        // Thermal + line commissioning gating: a plain (non-anticipated) thermal
        // with a must-run floor (`min_mw > 0`) carries an
        // `entry_stage_id`/`exit_stage_id` window so it is active only at stage 1
        // (dormant at 0 and 2), and a line carries an `entry_stage_id` so it is
        // dormant at stage 0 and active at stages 1 and 2. Under the dense
        // (zero-influence) layout a dormant thermal pins BOTH bounds to `[0, 0]`
        // (the must-run floor zeroes too, so the windowed-out plant cannot make
        // the LP infeasible) and a dormant line pins `col_upper` to 0 on both
        // directions. The parity hash covers hydro storage/water/cuts/convergence,
        // not thermal/line output directly, so it reflects the gating only through
        // the dispatch coupling: with T0 off and the inter-bus line cut at stage 0,
        // B1's load must lean on H1 + deficit instead of imported B0 power,
        // shifting both reservoirs' trajectories relative to an always-active case.
        "D36" => "d36-thermal-line-commissioning",
        // Anticipated thermal under a commissioning window AND a per-stage-varying
        // block schedule: a K=2 anticipated thermal with window `[entry=2, exit=4)`
        // over a 6-stage horizon. The decision gate is active only when the
        // DELIVERY stage `t + 2` lands in `[2, 4)`, so the pre-entry decision at
        // stage 0 delivers at the first operating stage 2 and the ring buffer drains
        // to 0 within K stages after exit. The only case combining an anticipated
        // thermal with a commissioning window, exercising the shifted decision gate
        // on top of the D34 n_blks-dependent decision-base surface.
        "D37" => "d37-anticipated-commissioning",
        // Mid-cascade dead-volume filling: a cascade
        // `H1 → H2 (filling) → H3 (real fed downstream)` plus an off-cascade
        // control H4. H2 carries `entry_stage_id = 4` with
        // `filling { start_stage_id = 2 }`, so it is PreFilling at stages 0–1,
        // Filling at 2–3, and Operating at 4–5. During PreFilling the dam is absent
        // from the LP, so its inflow short-circuits onto its real downstream H3's
        // water-balance row instead of into a sink. Block counts change at BOTH
        // phase boundaries (schedule 1/1/3/2/3/1), so the per-stage geometry and
        // per-stage `τ` are read across phase transitions that also straddle
        // block-count changes. The parity hash reflects the short-circuit and the
        // phase transitions through the cascade coupling on storage/water/cuts.
        "D38" => "d38-dead-volume-filling",
        other => panic!("unknown case label: {other}"),
    };
    // Integration tests run from the crate root; fixtures live at
    // ../../examples/deterministic/<suffix> relative to the crate.
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

    // Enable simulation so we get per-scenario stage results.
    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    // Use a small fixed scenario count for determinism and speed.
    config_with_sim.simulation.num_scenarios = 1;

    // The `hydro_energy_productivity.parquet` override is already folded into
    // `hydro_models.productivity_override` by the caller's
    // `prepare_hydro_models` invocation, so this helper does no parquet I/O.

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

    // Set up event channel to capture per-iteration convergence data.
    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();

    let outcome = setup
        .train(&mut solver, &comm, 1, ClpSolver::new, Some(event_tx), None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{label}: expected no training error"
    );
    let result = outcome.result;

    // Collect ConvergenceUpdate events and sort by iteration number.
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

    // Run simulation to collect per-scenario stage results.
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

    // Compute parity hash and compare/write baseline.
    let hash = compute_parity_hash(&convergence_updates, &setup, scenario_results);

    read_or_regen_baseline(label, &hash).unwrap_or_else(|msg| panic!("{msg}"));
}

// ---------------------------------------------------------------------------
// Individual test functions — D01–D17, with D12 and D16 absent
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
