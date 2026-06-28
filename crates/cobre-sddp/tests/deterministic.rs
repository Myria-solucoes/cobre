//! Deterministic test suite for the SDDP pipeline.
//!
//! Each case is fully deterministic — constant load (zero variance), no
//! stochastic inflows — so the optimal cost is hand-computable, which is what
//! licenses the exact cost asserts and tight convergence bounds below.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::{
    PolicyCheckpointMetadata, PolicyCutRecord, StageCutsPayload, write_policy_checkpoint,
};
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
};
use cobre_solver::ActiveSolver;
use cobre_solver::SolverInterface;

mod common;

/// Single-rank communicator whose `allgatherv`/`allreduce` copy send→recv
/// verbatim, so the pipeline runs without MPI.
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

/// Build a [`StudySetup`] for a case directory. Does no parquet I/O — the
/// caller's `prepare_hydro_models` already folded any override into
/// `hydro_models`, which is why `_case_dir` is unused.
fn build_setup_for_case(
    _case_dir: &Path,
    config: &cobre_io::Config,
    system: &cobre_core::System,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: cobre_sddp::PrepareHydroModelsResult,
) -> StudySetup {
    let sentinel = std::path::Path::new("config.json");
    let training_source = config
        .training_scenario_source(sentinel)
        .expect("training_scenario_source must parse");
    let simulation_source = config
        .simulation_scenario_source(sentinel)
        .expect("simulation_scenario_source must parse");

    let params = StudyParams::from_config(config).expect("StudyParams::from_config must succeed");
    let construction = params.into_construction_config();

    StudySetup::from_broadcast_params(
        system,
        stochastic,
        construction,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .expect("StudySetup::from_broadcast_params must build")
}

/// Train a case (`StubComm`, seed 42, 1 thread) and return the result plus the
/// live solver, so callers can inspect `solver.statistics()` after training.
fn run_deterministic_with_solver(case_dir: &Path) -> (cobre_sddp::TrainingResult, ActiveSolver) {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    (outcome.result, solver)
}

/// Train a case (`StubComm`, `ActiveSolver`, seed 42, 1 thread) and return the result.
fn run_deterministic(case_dir: &Path) -> cobre_sddp::TrainingResult {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    outcome.result
}

/// Train with 1-scenario simulation enabled, then simulate, returning the
/// training result, per-scenario results, and the aggregate summary.
fn run_with_simulation(
    case_dir: &Path,
) -> (
    cobre_sddp::TrainingResult,
    Vec<cobre_sddp::SimulationScenarioResult>,
    cobre_sddp::SimulationSummary,
) {
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 1;

    let mut setup = build_setup_for_case(
        case_dir,
        &config_with_sim,
        &system,
        stochastic,
        hydro_models,
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

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
    let summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    (result, scenario_results, summary)
}

fn assert_cost(actual: f64, expected: f64, tolerance: f64, case_name: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{case_name}: expected cost {expected}, got {actual} (diff={diff} > tolerance={tolerance})"
    );
}

/// Write a single-row `hydro_energy_productivity.parquet` supplying a per-hydro
/// `ρ_eq` override (`stage_id = NULL`, applies to all stages). Lets FPHA cases
/// without VHA geometry pass the FPHA correctness gate without changing LP economics.
fn write_energy_productivity_override(
    dest: &std::path::Path,
    hydro_id: i32,
    equivalent_productivity_mw_per_m3s: f64,
) {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new(
            "equivalent_productivity_mw_per_m3s",
            DataType::Float64,
            true,
        ),
        Field::new("reference_volume_hm3", DataType::Float64, true),
        Field::new("reference_outflow_m3s", DataType::Float64, true),
        Field::new(
            "specific_productivity_mw_per_m3s_per_m",
            DataType::Float64,
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![hydro_id])),
            Arc::new(Int32Array::from(vec![None::<i32>])),
            Arc::new(Float64Array::from(vec![equivalent_productivity_mw_per_m3s])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
        ],
    )
    .expect("valid RecordBatch for hydro_energy_productivity override");

    let file = std::fs::File::create(dest).expect("create hydro_energy_productivity.parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter for override");
    writer.write(&batch).expect("write override batch");
    writer.close().expect("close override writer");
}

/// Expected total cost for D02 (single hydro, 2 stages, deterministic inflows).
/// Derivation: κ=2.628, S₀=100hm³, inflows=[40,10]m³/s, demand=80MW.
/// Terminal stage turbines at 50m³/s capacity. Backward pass shows optimal
/// storage at stage boundary = 105.12 hm³.
///
/// Analytical thermal cost is `23_635_000 / 9 ≈ 2_626_111.11 $`. With
/// `turbined_cost = 0.01 $/MWh` applied to every hydro's turbine column
/// (see `fill_turbine_columns` in `lp::builder::columns`), the
/// deterministic LB adds a fixed regularization contribution of
/// `5_785 / 9 ≈ 642.78 $` (= 0.01 · 730 · (25_000/657 + 50) summed across the
/// two stages' turbined flows). Total = `23_640_785 / 9 ≈ 2_626_753.89 $`.
pub const D02_EXPECTED_COST: f64 = 23_640_785.0 / 9.0;

/// Expected total cost for D05 (FPHA constant-head, same physical setup as D02).
///
/// D05's `penalties.json` sets `turbined_cost = 0.0`, so the universal
/// turbined-cost regularization (which lifts D02 by `5_785 / 9 ≈ 642.78 $`)
/// contributes nothing here. The LB collapses to the pre-regularization
/// analytical value `23_635_000 / 9 ≈ 2_626_111.11 $` used to validate the
/// FPHA constant-head encoding against D02's hand-computed cost.
pub const D05_EXPECTED_COST: f64 = 23_635_000.0 / 9.0;

/// Two-stage pure thermal dispatch. Optimal cost is hand-computable.
///
/// ## Case setup
///
/// - 1 bus, 2 thermal plants (merit order), deterministic load 20 MW,
///   2 stages each with 730 hours, no hydro.
///
/// ## Expected cost derivation
///
/// - T0: capacity 15 MW at $5/MWh → dispatched at full capacity
/// - T1: capacity 15 MW at $10/MWh → dispatched at 5 MW to cover residual load
/// - Cost per stage = (15 × 5.0 + 5 × 10.0) × 730 = 125.0 × 730 = 91,250 $
/// - Total (2 stages) = 2 × 91,250 = **182,500 $**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d01_thermal_dispatch() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 182_500.0, 1e-6, "D01");
    assert!(
        result.iterations <= 10,
        "D01: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D01: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage hydrothermal dispatch. Optimal cost is hand-computed via LP.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0: 100 MW at $50/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 40.0 m3/s (stage 0) and 10.0 m3/s (stage 1)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages, 730 h each, no discounting
///
/// ## Expected cost
///
/// See [`D02_EXPECTED_COST`] for the full derivation. The optimal cost is
/// 23 635 000 / 9 ≈ 2 626 111.111... $, achieved by:
/// - Stage 0: turb₀ = 100/2.628 ≈ 38.05 m3/s, gen_th₀ ≈ 41.95 MW
/// - Stage 1: turb₁ = 50 m3/s (full capacity), gen_th₁ = 30 MW
/// - Storage at end of stage 0: exactly 40·2.628 = 105.12 hm3
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d02_single_hydro() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D02");
    assert!(
        result.iterations <= 10,
        "D02: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D02: gap={:.2e}",
        result.final_gap
    );
}

/// Three-stage cascade hydrothermal dispatch (2 hydros in series).
/// Combined capacity 70 MW < demand 75 MW. Cascade coupling: H1 receives H0's
/// discharge. Terminal stages: full capacity (thermal = 5 MW). Stage 0 binding
/// storage constraints yield thermal ≈ 28.09 MW.
///
/// Analytical thermal + deficit cost is `4_171_000 / 3 ≈ 1_390_333.33`. With
/// `turbined_cost` applied to every hydro's turbine column,
/// the deterministic LB adds a fixed regularization contribution that lifts
/// the total by `+1_364.4333…` (= turbined_cost × turbined MWh summed over
/// stages and blocks for both hydros).
pub const D03_EXPECTED_COST: f64 = 1_391_697.766_666_667_3;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d03_two_hydro_cascade() {
    let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D03_EXPECTED_COST, 1e-4, "D03");
    assert!(
        result.iterations <= 10,
        "D03: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D03: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage 2-bus transmission dispatch with line export limit.
/// B0 has excess hydro capacity and exports via 15 MW line to B1 (deficit region).
/// H0 covers B0 demand + export (thermal = 0). B1 has 5 MW unmet deficit/stage.
/// H1 depleted to minimize loss. Analytical thermal+deficit cost is
/// `5_263_443_883 / 657 ≈ 8_011_330.11 $`. With the universal `turbined_cost`
/// regularization applied to both hydros' turbine columns, the LB lifts to
/// `5_264_062_704 / 657 ≈ 8_012_272.00 $` (delta `≈ 941.89 $`).
pub const D04_EXPECTED_COST: f64 = 5_264_062_704.0 / 657.0;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d04_transmission() {
    let case_dir = Path::new("../../examples/deterministic/d04-transmission");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D04_EXPECTED_COST, 1e-4, "D04");
    assert!(
        result.iterations <= 10,
        "D04: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D04: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage FPHA hydrothermal dispatch with single hyperplane (constant-productivity equivalent).
/// Identical to D02 except H0 uses FPHA model with one hyperplane per stage encoding
/// `gen = 1.0 × turbined_flow` (γ₀=0, γᵥ=0, γ_q=1.0, γ_s=0.0).
/// LP must match D02 exactly; cost tolerance 1e-6.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d05_fpha_constant_head() {
    let case_dir = Path::new("../../examples/deterministic/d05-fpha-constant-head");

    // ρ_eq = 1.0 is LP-neutral here: D05's FPHA hyperplane encodes
    // gen_h = 1.0 × turbined_m3s, matching the override scalar.
    write_energy_productivity_override(
        &case_dir.join("system/hydro_energy_productivity.parquet"),
        0,
        1.0,
    );

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D05_EXPECTED_COST, 1e-6, "D05");
    assert!(
        result.iterations <= 10,
        "D05: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D05: gap={:.2e}",
        result.final_gap
    );
}

/// Expected total cost for D06 (variable-head FPHA, 2 planes per stage).
///
/// ## Case setup
///
/// Same physical system as D02/D05: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0–200 hm3), demand 80 MW, 2 stages × 730 h,
/// initial storage 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 uses 2 precomputed FPHA hyperplanes (stage_id = null, valid for all stages):
/// - Plane 0: γ₀=0.0, γᵥ=0.002, γ_q=0.8, γ_s=0.0, κ_scale=1.0
/// - Plane 1: γ₀=0.0, γᵥ=0.001, γ_q=0.95, γ_s=0.0, κ_scale=1.0
///
/// ## FPHA constraint formulation in cobre-sddp
///
/// The LP builder implements the FPHA constraint using the **average** of
/// incoming and outgoing storage, not solely outgoing storage:
///
/// ```text
/// gen_h ≤ γ₀ + γᵥ/2·V_in + γᵥ/2·V_out + γ_q·q + γ_s·s
/// ```
///
/// This encodes the average forebay head over the stage interval.
/// V_in is fixed by the Benders storage-fixing row at the reference point
/// (previous iteration's trial value or the initial condition).
///
/// ## Analytical derivation (κ = 730·3600/10⁶ = 657/250 = 2.628 hm3 per m3/s)
///
/// ### Stage 1 (terminal, no future value)
///
/// With V_in1 = V_out0, inflow = 10 m3/s, and the turbine cap at 50 m3/s:
///
/// Setting q₁ = 50 m3/s (max) requires V_out0 ≥ 40·κ = 105.12 hm3 so that
/// V_out1 = V_out0 + (10 − 50)·κ ≥ 0.
///
/// At V_out0 = 105.12 hm3 = 40·κ, V_out1 = 0, V_in1 = 40·κ:
/// - Plane 0 bound: 0.002/2·(40·κ + 0) + 0.8·50 = 657/6250 + 40 = 250657/6250 ≈ 40.105 MW
/// - Plane 1 bound: 0.001/2·(40·κ + 0) + 0.95·50 ≈ 47.553 MW
/// - Binding plane: 0 → gen_h1 = 250657/6250 ≈ 40.105 MW
/// - gen_th1 = 80 − 250657/6250 = 249343/6250 ≈ 39.895 MW
/// - Stage 1 cost = (249343/6250) × 730 × 50 = 36404078/25 = 1,456,163.12 $
///
/// ### Stage 0
///
/// The SDDP backward pass places a Benders cut on V_out0. The shadow price
/// of the water-balance constraint drives the optimiser to raise storage
/// from 100 to exactly 40·κ = 105.12 hm3, the minimum needed for q₁ = 50.
///
/// At optimum: q₀ = 25000/657 ≈ 38.0518 m3/s, sp₀ = 0,
/// V_out0 = 2628/25 hm3, V_in0 = 100 hm3.
///
/// Plane 0 binds (with average-storage FPHA):
/// - gen_h0 = 0.002/2·(100 + 2628/25) + 0.8·(25000/657) = 62921137/2053125 ≈ 30.6465 MW
/// - gen_th0 = 80 − gen_h0 = 101328863/2053125 ≈ 49.3535 MW
/// - Stage 0 cost = (101328863/2053125) × 730 × 50 = 405315452/225 ≈ 1,801,402.01 $
///
/// ### Total cost
///
/// Total = 405315452/225 + 36404078/25
///       = 405315452/225 + 327636702/225
///       = **732952154/225 ≈ 3,257,565.1289 $**
///
/// This differs from D02_EXPECTED_COST (≈ 2,626,111.11 $) and D05_EXPECTED_COST
/// because the variable-head FPHA constraints reduce per-unit generation:
/// at q₀ ≈ 38.05 m3/s and mean storage ≈ (100+105.12)/2 hm3, plane 0 gives
/// only ≈ 30.65 MW, and in stage 1 the partial head (from V_in1=105.12, V_out1=0)
/// raises gen_h1 slightly above 40 MW, lowering thermal cost slightly vs D02.
pub const D06_EXPECTED_COST: f64 = 732_952_154.0 / 225.0;

/// Two-stage FPHA hydrothermal dispatch with 2 variable-head hyperplanes.
///
/// H0 uses 2 precomputed hyperplanes encoding storage-dependent generation
/// (γᵥ > 0). Plane 0 (γᵥ=0.002, γ_q=0.8) and plane 1 (γᵥ=0.001, γ_q=0.95)
/// together approximate the concave production function. Cost differs from D02/D05
/// because head variation reduces per-m3/s generation at typical storage levels.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d06_fpha_variable_head() {
    let case_dir = Path::new("../../examples/deterministic/d06-fpha-variable-head");

    // ρ_eq value is irrelevant to D06 economics — assertions depend only on
    // FPHA hyperplane evaluation, not on ρ_eq.
    write_energy_productivity_override(
        &case_dir.join("system/hydro_energy_productivity.parquet"),
        0,
        1.0,
    );

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D06_EXPECTED_COST, 1e-4, "D06");
    assert!(
        result.iterations <= 10,
        "D06: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D06: gap={:.2e}",
        result.final_gap
    );
    assert!(
        (result.final_lb - D02_EXPECTED_COST).abs() > 1.0,
        "D06: cost must differ from D02 (variable head changes economics)"
    );
}

/// Two-stage FPHA hydrothermal dispatch with computed hyperplanes from VHA geometry.
///
/// ## Case setup
///
/// Same physical system as D02/D05/D06: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0–200 hm3), demand 80 MW, 2 stages × 730 h,
/// initial storage 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 uses `"source": "computed"` in `hydro_production_models.json`. The fitting
/// pipeline reads the VHA curve from `system/hydro_geometry.parquet`, evaluates
/// the production function φ(V, q) using the tailrace, hydraulic losses, and
/// efficiency fields in `hydros.json`, and fits FPHA hyperplanes automatically.
///
/// ## Geometry
///
/// VHA curve: 5 breakpoints over 0–200 hm3. Forebay heights 350–400 m (well above
/// the constant tailrace at 300 m), giving a net head of ~50–100 m. Factor losses
/// of 3% and constant efficiency of 92% are applied.
///
/// No exact cost is asserted: the computed fitting pipeline uses a different
/// discretization grid and plane count than D06's precomputed planes.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d07_fpha_computed() {
    let case_dir = Path::new("../../examples/deterministic/d07-fpha-computed");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_gap.abs() < 1e-6,
        "D07: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.iterations <= 10,
        "D07: iterations={}",
        result.iterations
    );
    assert!(
        result.final_lb > 0.0,
        "D07: final_lb={} must be positive",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// D-case energy-output correctness sweep (d02, d03)
// ---------------------------------------------------------------------------

/// Conversion factor from hm³·MW/(m³/s) to MWh (= 10⁶ / 3600).
///
/// `stored_energy_mwh = (volume_hm3 − V_min) × ρ_acum × ENERGY_FACTOR`
const ENERGY_FACTOR: f64 = 1.0e6 / 3600.0;

/// Expected `ρ_eq` and `ρ_acum` for D02 (single hydro, `constant_productivity = 1.0`,
/// no downstream).
///
/// `ρ_eq = 1.0` MW/(m³/s) — read directly from `hydros.json`.
/// `ρ_acum = 1.0` — H0 has no downstream, so accumulated = own `ρ_eq`.
const D02_RHO_EQ: f64 = 1.0;
const D02_RHO_ACUM: f64 = 1.0;

/// Stage-0 initial storage for D02 H0 \[hm³\] from `initial_conditions.json`.
const D02_H0_V_INIT: f64 = 100.0;

/// Deterministic stage-0 inflow for D02 H0 \[m³/s\].
///
/// From the D02 scenario parquet (std = 0): mean = 40.0 m³/s.
const D02_H0_STAGE0_INFLOW: f64 = 40.0;

/// Deterministic stage-1 inflow for D02 H0 \[m³/s\].
///
/// From the D02 scenario parquet (std = 0): mean = 10.0 m³/s.
const D02_H0_STAGE1_INFLOW: f64 = 10.0;

/// Expected `ρ_eq` and `ρ_acum` for D03 H0.
///
/// `ρ_eq(H0) = 1.0` — from `hydros.json`, `constant_productivity`.
/// `ρ_acum(H0) = 2.0` — H0 is upstream of H1; accumulated = ρ_eq(H0) + ρ_eq(H1).
const D03_H0_RHO_EQ: f64 = 1.0;
const D03_H0_RHO_ACUM: f64 = 2.0;

/// Expected `ρ_eq` and `ρ_acum` for D03 H1.
///
/// `ρ_eq(H1) = 1.0` — from `hydros.json`, `constant_productivity`.
/// `ρ_acum(H1) = 1.0` — H1 has no downstream.
const D03_H1_RHO_EQ: f64 = 1.0;
const D03_H1_RHO_ACUM: f64 = 1.0;

/// Stage-0 initial storage for D03 H0 \[hm³\] from `initial_conditions.json`.
const D03_H0_V_INIT: f64 = 80.0;

/// Stage-0 initial storage for D03 H1 \[hm³\] from `initial_conditions.json`.
const D03_H1_V_INIT: f64 = 50.0;

/// Verify the natural-inflow-energy and stored-energy columns in
/// `simulation/hydros` for D02 and D03. Both use `ConstantProductivity` hydros
/// (bypassing the FPHA gate), so `ρ_eq` and `ρ_acum` are directly computable
/// from `hydros.json` without running the LP.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d_case_energy_outputs() {
    const TOL: f64 = 1e-6;
    const V_MIN: f64 = 0.0; // both cases have min_storage_hm3 = 0

    // ── D02: single hydro, 2 stages ──────────────────────────────────────────
    {
        let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
        let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

        assert_eq!(scenario_results.len(), 1, "D02: expected 1 scenario");
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), 2, "D02: expected 2 stages");

        for stage_result in &scenario.stages {
            let stage = stage_result.stage_id as usize;
            assert_eq!(
                stage_result.hydros.len(),
                1,
                "D02 stage {stage}: expected 1 hydro result"
            );
            let h = &stage_result.hydros[0];

            let diff_rho_eq = (h.equivalent_productivity_mw_per_m3s - D02_RHO_EQ).abs();
            assert!(
                diff_rho_eq <= TOL,
                "D02 stage {stage} H0: equivalent_productivity_mw_per_m3s = {} (expected {D02_RHO_EQ}, diff = {diff_rho_eq})",
                h.equivalent_productivity_mw_per_m3s,
            );

            let diff_rho_acum = (h.accumulated_productivity_mw_per_m3s - D02_RHO_ACUM).abs();
            assert!(
                diff_rho_acum <= TOL,
                "D02 stage {stage} H0: accumulated_productivity_mw_per_m3s = {} (expected {D02_RHO_ACUM}, diff = {diff_rho_acum})",
                h.accumulated_productivity_mw_per_m3s,
            );

            let expected_energy = h.accumulated_productivity_mw_per_m3s * h.incremental_inflow_m3s;
            let diff_energy = (h.incremental_inflow_energy_mw - expected_energy).abs();
            assert!(
                diff_energy <= TOL,
                "D02 stage {stage} H0: incremental_inflow_energy_mw = {} (ρ_acum × inflow = {expected_energy}, diff = {diff_energy})",
                h.incremental_inflow_energy_mw,
            );

            let expected_earm = (h.storage_initial_hm3 - V_MIN)
                * h.accumulated_productivity_mw_per_m3s
                * ENERGY_FACTOR;
            let diff_earm = (h.stored_energy_initial_mwh - expected_earm).abs();
            assert!(
                diff_earm <= TOL,
                "D02 stage {stage} H0: stored_energy_initial_mwh = {} (expected {expected_earm}, diff = {diff_earm})",
                h.stored_energy_initial_mwh,
            );
        }

        // Stage 0 against hand-fixed constants (deterministic from inputs),
        // not just field-consistency.
        let h0_stage0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D02: H0 missing from stage 0");

        let diff_inflow0 =
            (h0_stage0.incremental_inflow_energy_mw - D02_RHO_ACUM * D02_H0_STAGE0_INFLOW).abs();
        assert!(
            diff_inflow0 <= TOL,
            "D02 stage 0 H0: incremental_inflow_energy_mw = {} (expected {}, diff = {diff_inflow0})",
            h0_stage0.incremental_inflow_energy_mw,
            D02_RHO_ACUM * D02_H0_STAGE0_INFLOW,
        );

        let expected_earm0 = (D02_H0_V_INIT - V_MIN) * D02_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm0 = (h0_stage0.stored_energy_initial_mwh - expected_earm0).abs();
        assert!(
            diff_earm0 <= TOL,
            "D02 stage 0 H0: stored_energy_initial_mwh = {} (expected {expected_earm0}, diff = {diff_earm0})",
            h0_stage0.stored_energy_initial_mwh,
        );

        let h0_stage1 = scenario.stages[1]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D02: H0 missing from stage 1");

        let diff_inflow1 =
            (h0_stage1.incremental_inflow_energy_mw - D02_RHO_ACUM * D02_H0_STAGE1_INFLOW).abs();
        assert!(
            diff_inflow1 <= TOL,
            "D02 stage 1 H0: incremental_inflow_energy_mw = {} (expected {}, diff = {diff_inflow1})",
            h0_stage1.incremental_inflow_energy_mw,
            D02_RHO_ACUM * D02_H0_STAGE1_INFLOW,
        );
    }

    // ── D03: two-hydro cascade, 3 stages ─────────────────────────────────────
    {
        let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
        let (_result, scenario_results, _summary) = run_with_simulation(case_dir);

        assert_eq!(scenario_results.len(), 1, "D03: expected 1 scenario");
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), 3, "D03: expected 3 stages");

        for stage_result in &scenario.stages {
            let stage = stage_result.stage_id as usize;
            assert_eq!(
                stage_result.hydros.len(),
                2,
                "D03 stage {stage}: expected 2 hydro results"
            );

            for h in &stage_result.hydros {
                let (expected_rho_eq, expected_rho_acum) = if h.hydro_id == 0 {
                    (D03_H0_RHO_EQ, D03_H0_RHO_ACUM)
                } else {
                    (D03_H1_RHO_EQ, D03_H1_RHO_ACUM)
                };

                let diff_rho_eq = (h.equivalent_productivity_mw_per_m3s - expected_rho_eq).abs();
                assert!(
                    diff_rho_eq <= TOL,
                    "D03 stage {stage} H{}: equivalent_productivity_mw_per_m3s = {} (expected {expected_rho_eq}, diff = {diff_rho_eq})",
                    h.hydro_id,
                    h.equivalent_productivity_mw_per_m3s,
                );

                let diff_rho_acum =
                    (h.accumulated_productivity_mw_per_m3s - expected_rho_acum).abs();
                assert!(
                    diff_rho_acum <= TOL,
                    "D03 stage {stage} H{}: accumulated_productivity_mw_per_m3s = {} (expected {expected_rho_acum}, diff = {diff_rho_acum})",
                    h.hydro_id,
                    h.accumulated_productivity_mw_per_m3s,
                );

                let expected_energy =
                    h.accumulated_productivity_mw_per_m3s * h.incremental_inflow_m3s;
                let diff_energy = (h.incremental_inflow_energy_mw - expected_energy).abs();
                assert!(
                    diff_energy <= TOL,
                    "D03 stage {stage} H{}: incremental_inflow_energy_mw = {} (ρ_acum × inflow = {expected_energy}, diff = {diff_energy})",
                    h.hydro_id,
                    h.incremental_inflow_energy_mw,
                );

                let expected_earm = (h.storage_initial_hm3 - V_MIN)
                    * h.accumulated_productivity_mw_per_m3s
                    * ENERGY_FACTOR;
                let diff_earm = (h.stored_energy_initial_mwh - expected_earm).abs();
                assert!(
                    diff_earm <= TOL,
                    "D03 stage {stage} H{}: stored_energy_initial_mwh = {} (expected {expected_earm}, diff = {diff_earm})",
                    h.hydro_id,
                    h.stored_energy_initial_mwh,
                );
            }
        }

        // Stage 0 against hand-fixed constants (deterministic from initial conditions).
        let h0_s0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 0)
            .expect("D03: H0 missing from stage 0");
        let h1_s0 = scenario.stages[0]
            .hydros
            .iter()
            .find(|h| h.hydro_id == 1)
            .expect("D03: H1 missing from stage 0");

        let expected_earm_h0 = (D03_H0_V_INIT - V_MIN) * D03_H0_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm_h0 = (h0_s0.stored_energy_initial_mwh - expected_earm_h0).abs();
        assert!(
            diff_earm_h0 <= TOL,
            "D03 stage 0 H0: stored_energy_initial_mwh = {} (expected {expected_earm_h0}, diff = {diff_earm_h0})",
            h0_s0.stored_energy_initial_mwh,
        );

        let expected_earm_h1 = (D03_H1_V_INIT - V_MIN) * D03_H1_RHO_ACUM * ENERGY_FACTOR;
        let diff_earm_h1 = (h1_s0.stored_energy_initial_mwh - expected_earm_h1).abs();
        assert!(
            diff_earm_h1 <= TOL,
            "D03 stage 0 H1: stored_energy_initial_mwh = {} (expected {expected_earm_h1}, diff = {diff_earm_h1})",
            h1_s0.stored_energy_initial_mwh,
        );
    }
}

/// Expected total cost for D08 (single hydro with linearized evaporation, 2 stages).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: constant productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW,
/// storage 0–200 hm3), demand 80 MW, 2 stages × 730 h, initial storage
/// 100 hm3, deterministic inflows 40/10 m3/s.
///
/// H0 has evaporation enabled: `coefficients_mm = [100.0; 12]` (uniform
/// across all 12 calendar months). No `reference_volumes_hm3` — default
/// midpoint (100 hm3) is used.
///
/// ## Geometry (hydro_geometry.parquet)
///
/// Linear VHA: (0 hm3 → 0.5 km²), (100 hm3 → 1.0 km²), (200 hm3 → 1.5 km²).
/// Uniform slope da/dv = 0.005 km²/hm3.
///
/// ## Evaporation coefficient derivation
///
/// Midpoint reference_volume = (0 + 200) / 2 = 100 hm3.
/// a_ref = 1.0 km², da/dv = 0.005 km²/hm3.
/// stage_hours = 730 h → mm_km2_to_m3s = 1 / (3.6 × 730) = 1/2628.
/// monthly_evaporation_mm = 100 mm.
/// volume_slope_m3s_per_hm3 = (1/2628) × 100 × 0.005 = 1/5256.
/// intercept_m3s            = (1/2628) × 100 × 1.0 − (1/5256) × 100 = 25/1314.
///
/// ## Water-balance model (α = κ × volume_slope_m3s_per_hm3 / 2 = 1/4000)
///
/// Substituting evaporation_outflow = intercept_m3s + volume_slope_m3s_per_hm3/2 × (V_out + V_in) into the LP:
///   V_out × (1 + α) = V_in × (1 − α) + κ × (q_in − q_turb) − κ × intercept_m3s
///
/// ## Expected cost derivation (κ = 657/250 hm3/(m3/s))
///
/// The gradient of total cost w.r.t. turb₀ is proportional to
/// `−1 + (1−α)/(1+α) < 0`, so increasing turb₀ decreases total cost.
/// The optimal stage-0 policy is therefore turb₀ = 50 m3/s (full capacity).
///
/// ### Stage 0 (turb₀ = 50, V_in₀ = 100 hm3)
///
/// V_out₀ = [100×(1−α) + κ×(40 − 50) − κ×intercept_m3s] / (1+α)
///         = 294580/4001 ≈ 73.627 hm3.
/// gen_h₀ = 50 MW, gen_th₀ = 30 MW.
/// Stage 0 cost = 30 × 50 × 730 = 1,095,000 $.
///
/// ### Stage 1 (terminal, V_in₁ = V_out₀ = 294580/4001 hm3)
///
/// turb₁_max = V_in₁ × (1−α) / κ + 10 − intercept_m3s
///           = 399452585/10514628 ≈ 37.990 m3/s  (<50, so binding).
/// At turb₁ = turb₁_max: V_out₁ = 0.
///
/// gen_th₁ = 80 − turb₁_max = 441717655/10514628 ≈ 42.010 MW.
/// Stage 1 cost = gen_th₁ × 50 × 730 = 55214706875/36009 ≈ 1,533,358.52 $.
///
/// ### Total cost
///
/// Total = 1,095,000 + 55214706875/36009
///       = 39439545000/36009 + 55214706875/36009
///       = **94644561875/36009 ≈ 2,628,358.52 $**
///
/// D08 cost > D02 cost (≈ 2,626,111.11 $): evaporation consumes additional water
/// in the reservoir, leaving less for stage-1 generation and requiring more
/// thermal dispatch.
pub const D08_EXPECTED_COST: f64 = 94_644_561_875.0 / 36_009.0;

#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d08_evaporation() {
    let case_dir = Path::new("../../examples/deterministic/d08-evaporation");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D08_EXPECTED_COST, 1e-4, "D08");
    assert!(
        result.iterations <= 10,
        "D08: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D08: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D08: cost {:.6} must exceed D02 cost {:.6}",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Expected total cost for D09 (multi-segment deficit, 2 stages).
///
/// ## Case setup
///
/// - 1 bus (B0) with 2 deficit segments:
///   - Segment 0: depth_mw=10.0, cost=$500/MWh (first 10 MW of deficit)
///   - Segment 1: depth_mw=null (unlimited), cost=$5000/MWh (remaining deficit)
/// - 1 thermal (T0): capacity 30 MW at $10/MWh
/// - Deterministic load: 50 MW per stage
/// - 2 stages, 730 hours each, no hydro
///
/// ## Expected cost derivation
///
/// With supply = 30 MW (thermal at full capacity) and demand = 50 MW:
///   deficit = 20 MW per stage
///   - First 10 MW of deficit → segment 0: 10 × $500/MWh
///   - Next 10 MW of deficit → segment 1: 10 × $5000/MWh
///
/// Cost per stage = (30 × 10 + 10 × 500 + 10 × 5000) × 730
///                = (300 + 5000 + 50000) × 730
///                = 55300 × 730
///                = 40,369,000 $
/// Total (2 stages) = 2 × 40,369,000 = **80,738,000 $**
pub const D09_EXPECTED_COST: f64 = 80_738_000.0;

/// Two-stage pure thermal dispatch with 2-segment tiered deficit pricing.
///
/// ## Case setup
///
/// - 1 bus with 2 deficit segments: [10 MW @ $500/MWh, unlimited @ $5000/MWh]
/// - 1 thermal (T0): 30 MW at $10/MWh, deterministic load 50 MW
/// - 2 stages × 730 h, no hydro
///
/// ## Expected cost
///
/// See [`D09_EXPECTED_COST`] for the derivation. With 20 MW deficit per stage
/// split across both segments, total cost is 80,738,000 $.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d09_multi_deficit() {
    let case_dir = Path::new("../../examples/deterministic/d09-multi-deficit");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D09_EXPECTED_COST, 1e-6, "D09");
    assert!(
        result.iterations <= 10,
        "D09: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D09: gap={:.2e}",
        result.final_gap
    );
}

/// Expected total cost for D10 (inflow non-negativity penalty, 2 stages).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus (B0), 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: constant productivity 1.0 MW/(m3/s), max 50 m3/s, storage
/// 0–200 hm3), demand 80 MW, 2 stages × 730 h, initial storage 100 hm3.
///
/// Inflows: stage 0 = 40 m3/s (positive), stage 1 = -5 m3/s (negative).
/// Config: `inflow_non_negativity: {method: "penalty", penalty_cost: 500.0}`.
///
/// ## Penalty cost unit (verified from `lp::builder::template`)
///
/// From `build_stage_templates` in `lp::builder::template`:
/// ```text
/// let obj_coeff = penalty_cost * total_stage_hours;
/// objective[col] = obj_coeff;
/// ```
/// The inflow slack column `sigma_inf_h` has objective coefficient
/// `penalty_cost × total_stage_hours = 500 × 730 = 365,000 $/m3/s`.
///
/// The LP column for sigma has lower bound 0 (not max(0, -inflow)).  The LP
/// freely chooses sigma = 0 when it is cheaper to reduce turbining instead.
///
/// ## Cost derivation (κ = 730·3600/10⁶ = 657/250 hm³/(m³/s))
///
/// ### Stage 1 optimal sigma
///
/// Stage 1 water balance:
///   V_out1 = V_in1 + (sigma − 5 − q1) × κ ≥ 0.
///
/// KKT: increasing sigma by 1 m3/s costs 365,000 $ but adds κ hm3 of water
/// (which at most saves 50 × 730/κ = 36,500 $/m3/s via turbining).
/// Since 365,000 >> 36,500, the optimizer always sets sigma = 0.
///
/// ### Stage 1 dispatch (sigma = 0, effective inflow = −5 m3/s)
///
/// With sigma = 0 the balance is:
///   V_out1 = V_in1 + (−5 − q1) × κ ≥ 0  →  q1 ≤ V_in1/κ − 5.
///
/// Breakpoints:
///   If V_in1 ≥ 55·κ (= 144.54 hm3): q1 = 50, gen_th1 = 30, cost1 = 1,095,000.
///   If 5·κ ≤ V_in1 < 55·κ:           q1 = V_in1/κ − 5.
///
/// Since q0 ≤ 50 and inflow0 = 40: V_out0 ≥ 100 − 10·κ ≈ 73.7 hm3 > 5·κ = 13.1 hm3.
/// So sigma = 0 is always feasible (no infeasibility risk for any q0 ≤ 50).
///
/// ### Stage 0 optimal policy
///
/// Water balance: V_out0 = 100 + (40 − q0) × κ.
///
/// Case A (q0 ≤ 15145/657, V_out0 ≥ 55·κ):
///   Total thermal = 36500 × (110 − q0). Decreases with q0; optimal at boundary.
///
/// Case B (q0 > 15145/657, V_out0 < 55·κ):
///   q1 = V_out0/κ − 5 = 35 + 25000/657 − q0.
///   Total thermal = 36500 × [(80 − q0) + (45 − 25000/657 + q0)]
///                 = 36500 × (125 − 25000/657) — constant in q0.
///
/// At the boundary (Case A → B): total thermal is continuous.
/// The optimizer is indifferent in Case B and the SDDP converges to the
/// same constant total thermal cost for any q0 in that range.
///
/// ### Total cost (no penalty)
///
/// Total = 36500 × (125 − 25000/657)
///       = 36500 × (82125 − 25000)/657
///       = 36500 × 57125/657
///       = **2,085,062,500/657 = 28,562,500/9 ≈ 3,173,611.11 $**
///
/// D10 cost > D02 cost (≈ 2,626,111.11 $): the negative inflow in stage 1
/// effectively reduces available water (turbining is limited to V_in1/κ − 5
/// instead of V_in1/κ + 10), requiring more thermal dispatch in stage 1.
pub const D10_EXPECTED_COST: f64 = 28_562_500.0 / 9.0;

/// Two-stage hydrothermal dispatch with inflow non-negativity penalty.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $50/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s, storage 0–200 hm3)
/// - Deterministic inflows: 40.0 m3/s (stage 0), -5.0 m3/s (stage 1)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h, `inflow_non_negativity: {method: "penalty", penalty_cost: 500.0}`
///
/// ## Expected cost
///
/// See [`D10_EXPECTED_COST`] for the full derivation.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d10_inflow_nonnegativity() {
    let case_dir = Path::new("../../examples/deterministic/d10-inflow-nonnegativity");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D10_EXPECTED_COST, 1e-4, "D10");
    assert!(
        result.iterations <= 10,
        "D10: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D10: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D10: cost {:.6} must exceed D02 cost {:.6}",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Expected total cost for D11 (single hydro with water withdrawal, 2 stages).
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 30.0 m3/s per stage
/// - Water withdrawal: 10 m3/s per stage (via `constraints/hydro_bounds.parquet`)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h, `inflow_non_negativity: {method: "none"}`
///
/// ## How withdrawal enters the water balance
///
/// The LP water balance for each stage is:
///   V_out = V_in + κ × (inflow − withdrawal − turbine − spill)
///         = V_in + κ × (30 − 10 − turbine − spill)
///         = V_in + κ × (20 − turbine − spill)
///
/// This is equivalent to a case with 20 m3/s net inflow and no withdrawal.
///
/// ## Expected cost derivation (κ = 730 × 3600 / 10⁶ = 657/250 hm³/(m³/s))
///
/// Let q0 = turbined flow in stage 0.
///
/// Stage 1 water balance: V_out1 = V_in1 + κ × (20 − q1) ≥ 0
///   → q1 ≤ V_in1/κ + 20
///
/// Stage 0 storage: V_out0 = 100 + κ × (20 − q0)
///
/// ### Case A: q0 ≤ 18430/657 (so V_out0 ≥ 30κ, stage 1 can run at full capacity)
///
/// When V_out0 ≥ 30κ, q1 = 50 and gen_th1 = 30 MW.
/// Total thermal = (80 − q0) + 30 = 110 − q0, which decreases with q0.
/// Optimal at q0 = 18430/657 ≈ 28.054 m3/s (boundary).
///
/// ### Case B: q0 > 18430/657 (V_out0 < 30κ, stage 1 is storage-limited)
///
/// q1 = V_out0/κ + 20 = (100 + κ×(20−q0))/κ + 20 = 100/κ + 40 − q0
/// gen_th1 = 80 − q1 = 40 − 100/κ + q0
///
/// Total thermal = (80 − q0) + (40 − 100/κ + q0) = 120 − 100/κ
///   = 120 − 25000/657 = 53840/657 MW (constant — independent of q0)
///
/// The objective is constant in Case B, so SDDP converges to:
///   Total cost = (53840/657) × 100 × 730
///              = 53840 × 73000 / 657
///              = **3,930,320,000 / 657 ≈ 5,982,222.22 $**
///
/// D11 cost > D02 cost (≈ 2,626,111.11 $) because the withdrawal reduces net
/// inflow from 30 to 20 m3/s, leaving less water for generation across both
/// stages and requiring significantly more thermal dispatch.
///
/// The pinned value adds the universal `turbined_cost` regularization
/// (`≈ 569.78 $`) to the analytical thermal+deficit cost
/// `3_930_320_000 / 657 ≈ 5_982_222.22 $`, yielding `3_930_694_344 / 657`.
pub const D11_WATER_WITHDRAWAL_EXPECTED_COST: f64 = 3_930_694_344.0 / 657.0;

/// Two-stage hydrothermal dispatch with water withdrawal applied via hydro bounds.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 100 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m3/s), max 50 m3/s / 50 MW, storage 0–200 hm3)
/// - Deterministic inflows: 30.0 m3/s per stage
/// - Water withdrawal: 10 m3/s per stage (from `constraints/hydro_bounds.parquet`)
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 100.0 hm3
/// - 2 stages × 730 h
///
/// ## Expected cost
///
/// See [`D11_WATER_WITHDRAWAL_EXPECTED_COST`] for the full derivation. The 10 m3/s
/// withdrawal reduces effective net inflow from 30 to 20 m3/s, increasing thermal dispatch.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d11_water_withdrawal() {
    let case_dir = Path::new("../../examples/deterministic/d11-water-withdrawal");
    let result = run_deterministic(case_dir);
    assert_cost(
        result.final_lb,
        D11_WATER_WITHDRAWAL_EXPECTED_COST,
        1e-4,
        "D11",
    );
    assert!(
        result.iterations <= 10,
        "D11: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D11: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb > D02_EXPECTED_COST,
        "D11: cost {:.6} must exceed D02 cost {:.6} (withdrawal increases thermal dispatch)",
        result.final_lb,
        D02_EXPECTED_COST
    );
}

/// Warm-start verification for the D02 system: after training,
/// `SolverStatistics.basis_consistency_failures` must be zero. A non-zero count
/// means silent cold-start fallbacks that degrade performance without an error.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d11_warm_start_verification() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D11");
    assert!(
        result.final_gap.abs() < 1e-6,
        "D11: gap={:.2e}",
        result.final_gap
    );

    let stats = solver.statistics();
    assert_eq!(
        stats.basis_consistency_failures, 0,
        "D11: expected 0 basis rejections, got {}",
        stats.basis_consistency_failures
    );
}

/// Checkpoint round-trip for the D02 system: exercises the full FlatBuffers
/// persistence pipeline (train → write → read → simulate from the loaded FCF).
///
/// ## Why the simulation cost should equal the training LB
///
/// D02 has deterministic (zero-variance) inflows and load. With
/// `num_scenarios = 1`, the single simulation scenario uses the same inflow
/// realization as the training forward pass, and the converged FCF is the true
/// value function — so the simulation cost equals the training LB.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d12_checkpoint_round_trip() {
    let case_dir = Path::new("../../examples/deterministic/d02-single-hydro");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 1;

    let mut setup = StudySetup::new(&system, &config_with_sim, stochastic, hydro_models)
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let result = outcome.result;

    assert_cost(result.final_lb, D02_EXPECTED_COST, 1e-4, "D12-train");
    assert!(
        result.final_gap.abs() < 1e-6,
        "D12: gap={:.2e}",
        result.final_gap
    );

    // Required side effect before StageCutsPayload extraction below.
    let _training_output = setup.build_training_output(&result, &[]);

    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let policy_dir = tmp.path().join("policy");

    let fcf = &setup.fcf;

    let cut_records_per_stage: Vec<Vec<PolicyCutRecord<'_>>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .map(|slot| {
                    let meta = &pool.metadata[slot];
                    PolicyCutRecord {
                        cut_id: slot as u64,
                        slot_index: slot as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[slot],
                        coefficients: &pool.coefficients
                            [slot * pool.state_dimension..(slot + 1) * pool.state_dimension],
                        is_active: pool.active[slot],
                    }
                })
                .collect()
        })
        .collect();

    let active_indices_per_stage: Vec<Vec<u32>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .filter(|&slot| pool.active[slot])
                .map(|slot| slot as u32)
                .collect()
        })
        .collect();

    let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = fcf
        .pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: pool.state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &cut_records_per_stage[stage_idx],
            active_cut_indices: &active_indices_per_stage[stage_idx],
            populated_count: pool.populated_count as u32,
        })
        .collect();

    let n_stages = fcf.pools.len();
    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let policy_metadata = PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-03-16T00:00:00Z".to_string(),
        completed_iterations: result.iterations as u32,
        final_lower_bound: result.final_lb,
        best_upper_bound: Some(result.final_ub),
        state_dimension: fcf.state_dimension as u32,
        num_stages: n_stages as u32,
        max_iterations: 100,
        forward_passes: 1,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: 42,
        total_visited_states: 0,
    };

    write_policy_checkpoint(
        &policy_dir,
        &stage_cuts_payloads,
        &[],
        &policy_metadata,
        &[],
    )
    .expect("write_policy_checkpoint must succeed");

    let checkpoint =
        cobre_io::read_policy_checkpoint(&policy_dir).expect("read_policy_checkpoint must succeed");

    assert_eq!(
        checkpoint.metadata.num_stages, 2,
        "D12: checkpoint must have 2 stages"
    );
    assert_eq!(
        checkpoint.metadata.state_dimension, 1,
        "D12: checkpoint must have state_dimension == 1 (one hydro = one storage state)"
    );
    assert!(
        !checkpoint.stage_cuts.is_empty(),
        "D12: checkpoint must contain at least one stage_cuts entry"
    );

    let metadata_path = policy_dir.join("metadata.json");
    assert!(metadata_path.is_file(), "D12: metadata.json must exist");

    let stage_bin_path = policy_dir.join("cuts/stage_000.bin");
    assert!(
        stage_bin_path.is_file(),
        "D12: cuts/stage_000.bin must exist"
    );

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
    let _scenario_results = drain_handle.join().expect("drain thread must not panic");

    let sim_config = setup.simulation_config();
    let summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    assert_eq!(
        summary.n_scenarios, 1,
        "D12: simulation must produce exactly 1 scenario"
    );

    assert_cost(summary.mean_cost, D02_EXPECTED_COST, 1e-2, "D12-sim");
}

/// Generic constraint capping thermal dispatch, forcing deficit.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal T0: capacity 30 MW at $50/MWh, deterministic load 20 MW,
///   2 stages each with 730 hours, no hydro.
/// - 1 generic constraint: `thermal_generation(0) <= 10 MW`
///   with slack penalty $5000/MWh (slack is more expensive than deficit at $1000/MWh).
/// - Deficit cost: $1000/MWh (from buses.json).
///
/// ## Expected cost derivation
///
/// The optimizer will dispatch T0 = 10 MW (at the constraint cap) and leave
/// 10 MW of deficit, since deficit ($1000/MWh) is cheaper than violating
/// the generic constraint ($5000/MWh).
///
/// Cost per stage:
///   thermal: 10 MW × $50/MWh × 730 h = $365,000
///   deficit: 10 MW × $1000/MWh × 730 h = $7,300,000
///   total:   $7,665,000
///
/// Total (2 stages) = 2 × $7,665,000 = **$15,330,000**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d13_generic_constraint() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d13-generic-constraint");

    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Int32Array::new_null(2)), // block_id null = all blocks
            Arc::new(Float64Array::from(vec![10.0, 10.0])),
        ],
    )
    .expect("RecordBatch");

    let bounds_path = constraints_dir.join("generic_constraint_bounds.parquet");
    let file = std::fs::File::create(&bounds_path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 15_330_000.0, 1e-2, "D13");
    assert!(
        result.iterations <= 10,
        "D13: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D13: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage thermal dispatch with per-block load factors.
///
/// ## Case setup
///
/// - 1 bus, 2 thermal plants (merit order: T0 at $5/MWh cap 15 MW, T1 at
///   $10/MWh cap 15 MW), deterministic load 20 MW mean, 2 stages each with
///   2 blocks (block 0: 400 hours, block 1: 330 hours), load factors [0.8, 1.2]
///
/// ## Expected cost derivation
///
/// - Block 0: load = 20 * 0.8 = 16 MW.  T0=15 MW, T1=1 MW.
///   cost = (15*5 + 1*10) * 400 = 85 * 400 = 34,000
/// - Block 1: load = 20 * 1.2 = 24 MW.  T0=15 MW, T1=9 MW.
///   cost = (15*5 + 9*10) * 330 = 165 * 330 = 54,450
/// - Cost per stage = 34,000 + 54,450 = 88,450
/// - Total (2 stages) = 2 * 88,450 = **176,900**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d14_block_factors() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d14-block-factors");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));

    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![20.0, 20.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");

    let load_path = scenarios_dir.join("load_seasonal_stats.parquet");
    let file = std::fs::File::create(&load_path).expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    // Empty inflow stats: D14 has no hydros.
    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::new_empty(Arc::clone(&inflow_schema));
    let inflow_path = scenarios_dir.join("inflow_seasonal_stats.parquet");
    let file = std::fs::File::create(&inflow_path).expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 176_900.0, 1e-4, "D14");
    assert!(
        result.iterations <= 10,
        "D14: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D14: gap={:.2e}",
        result.final_gap
    );
}

/// Two-stage thermal + NCS dispatch.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0 at $10/MWh, cap 100 MW), 1 NCS (curtailment_cost
///   $0.001/MWh, bus 0, max_generation_mw 100 MW), deterministic load 80 MW,
///   2 stages each with 1 block of 730 hours.
/// - NCS available generation = 50 MW per stage (from non_controllable_stats.parquet,
///   mean=0.5, std=0.0 — availability factor 0.5 * 100 MW = 50 MW, deterministic,
///   exercises the stochastic NCS pipeline).
///
/// ## Expected cost derivation
///
/// - NCS generates at full 50 MW (incentivized by negative objective coeff).
/// - Thermal covers remaining 30 MW.
/// - Thermal cost per stage: 30 * 10 * 730 = 219,000
/// - NCS curtailment cost per stage: 0.001 * 50 * 730 = 36.5 (regularization,
///   the LP objective adds -0.001 * block_hours * g_ncs).
///   The NCS contribution to objective = -0.001 * 730 * 50 = -36.5
/// - Total objective per stage = 219,000 + (-36.5) = 218,963.5
/// - Total (2 stages) = 2 * 218,963.5 = **437,927.0**
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d15_non_controllable_source() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d15-non-controllable-source");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));

    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![80.0, 80.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");

    let load_path = scenarios_dir.join("load_seasonal_stats.parquet");
    let file = std::fs::File::create(&load_path).expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    // Empty inflow stats: D15 has no hydros.
    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::new_empty(Arc::clone(&inflow_schema));
    let inflow_path = scenarios_dir.join("inflow_seasonal_stats.parquet");
    let file = std::fs::File::create(&inflow_path).expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    // NCS availability is a factor: mean 0.5 × max 100 MW = 50 MW. std 0 drives
    // the stochastic NCS pipeline with zero noise (deterministic).
    let ncs_schema = Arc::new(Schema::new(vec![
        Field::new("ncs_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean", DataType::Float64, false),
        Field::new("std", DataType::Float64, false),
    ]));

    let ncs_batch = RecordBatch::try_new(
        Arc::clone(&ncs_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![0.5, 0.5])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("non_controllable_stats RecordBatch");

    let ncs_path = scenarios_dir.join("non_controllable_stats.parquet");
    let file = std::fs::File::create(&ncs_path).expect("create non_controllable_stats parquet");
    let mut writer = ArrowWriter::try_new(file, ncs_schema, None).expect("ArrowWriter");
    writer
        .write(&ncs_batch)
        .expect("write non_controllable_stats batch");
    writer.close().expect("close non_controllable_stats writer");

    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 437_927.0, 1e-2, "D15");
    assert!(
        result.iterations <= 10,
        "D15: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D15: gap={:.2e}",
        result.final_gap
    );
}

/// D18 per-stage NCS commissioning dormancy varies across the window.
///
/// The fixture declares two NCS sources on a 3-stage horizon: NCS0 with
/// `entry_stage_id = 1` (commissions at stage 1) and NCS1 with
/// `exit_stage_id = 2` (decommissions at stage 2). Under the dense layout both
/// NCS keep a column at every stage; the per-stage dormancy mask computed by
/// `StudySetup::ncs_stochastic_dormant_for_test` (applying the commissioning
/// predicate to the stored windows, keyed by stochastic slot) marks which is
/// commissioning-dormant:
///
/// - stage 0 → exactly one slot dormant (NCS0 not yet commissioned)
/// - stage 1 → no slot dormant (both active)
/// - stage 2 → exactly one slot dormant (NCS1 decommissioned)
///
/// and the dormant slot at stage 0 differs from the dormant slot at stage 2.
/// This guards the per-stage NCS patch path: a uniform-NCS case (no dormancy at
/// any stage, as in D15) cannot detect an availability bound zeroed onto the
/// wrong stage's columns. Asserting the dormancy genuinely varies across stages
/// confirms the fixture exercises that path rather than collapsing to a uniform
/// (no-dormancy) mask.
///
/// This assertion only builds the stage templates — it does not train — so it
/// runs under the default `cargo test` profile, unconditionally.
#[test]
fn d18_ncs_commissioning_active_set_varies() {
    let case_dir = Path::new("../../examples/deterministic/d18-ncs-commissioning-window");
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let setup = build_setup_for_case(
        case_dir,
        &config,
        &system,
        prepare_result.stochastic,
        hydro_models,
    );

    let dormant = setup.ncs_stochastic_dormant_for_test();
    assert_eq!(dormant.len(), 3, "D18: expected 3 study stages");

    // The dense column count is the full NCS count (2) at every stage; the
    // dormancy mask, not an active-subset list, carries the per-stage variance.
    let dormant_count = |s: usize| dormant[s].iter().filter(|&&d| d).count();
    assert_eq!(dormant_count(0), 1, "D18 stage 0: exactly one NCS dormant");
    assert_eq!(dormant_count(1), 0, "D18 stage 1: both NCS active");
    assert_eq!(dormant_count(2), 1, "D18 stage 2: exactly one NCS dormant");

    // The load-bearing per-stage property: the dormant set is not uniform across
    // the horizon — stage 0 differs from the fully-commissioned stage, and the
    // dormant slot at stage 0 (NCS0 pre-entry) differs from stage 2 (NCS1 exit).
    assert_ne!(
        dormant[0], dormant[1],
        "D18: dormancy must differ between stage 0 and the commissioned stage"
    );
    assert_ne!(
        dormant[0], dormant[2],
        "D18: the dormant NCS at stage 0 (pre-entry) must differ from stage 2 (post-exit)"
    );
}

/// D33 per-stage block count varies across the horizon.
///
/// The fixture declares three study stages whose `blocks` arrays have different
/// length — 1 / 3 / 2 — so each stage's per-block equipment column stride
/// (turbine/spillage/thermal/bus) diverges from stage 0's. `block_hours_per_stage[t]`
/// carries one entry per block of stage `t`, so its length equals that stage's
/// `blocks.len()` — the per-stage block count threaded through the pipeline.
///
/// This guards the per-stage block-count path: a uniform-block case (every stage
/// shares one block count, as in D14) cannot detect equipment columns read off
/// the wrong stage's block width, because all stages share the same stride.
/// Asserting the block count genuinely differs across stages confirms the
/// fixture exercises that path rather than collapsing to a uniform width.
///
/// This assertion only builds the stage templates — it does not train — so it
/// runs under the default `cargo test` profile, unconditionally.
#[test]
fn d33_per_stage_block_count_varies() {
    let case_dir = Path::new("../../examples/deterministic/d33-per-stage-block-counts");
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let setup = build_setup_for_case(
        case_dir,
        &config,
        &system,
        prepare_result.stochastic,
        hydro_models,
    );

    // `block_hours_per_stage[t].len()` is the block count of stage `t`
    // (`compute_noise_scale` collects one `duration_hours` per block of the
    // stage), which equals `block_counts_per_stage[t]` threaded through the
    // pipeline.
    let block_counts: Vec<usize> = setup
        .stage_data
        .stage_templates
        .block_hours_per_stage
        .iter()
        .map(Vec::len)
        .collect();
    assert_eq!(block_counts, vec![1, 3, 2], "D33: per-stage block counts");

    // The load-bearing per-stage property: the block count is not uniform across
    // the horizon, so the per-block equipment stride genuinely differs per stage.
    let distinct: std::collections::BTreeSet<usize> = block_counts.iter().copied().collect();
    assert!(
        distinct.len() >= 2,
        "D33: per-stage block count must take at least two distinct values, got {block_counts:?}"
    );
    assert_ne!(
        block_counts[0], block_counts[1],
        "D33: block count must differ between stage 0 and stage 1"
    );
}

/// D18: NCS commissioning window — three-stage thermal + NCS dispatch.
///
/// ## Case setup
///
/// - 1 bus, 1 thermal (T0 at $10/MWh, cap 100 MW), deterministic load 80 MW,
///   3 stages each with 1 block of 730 hours, finite horizon, no discount.
/// - Two NCS sources (bus 0, max 100 MW, curtailment_cost $0.001/MWh),
///   availability factor 0.5 (= 50 MW each, std 0 deterministic). NCS0
///   `entry_stage_id = 1`, NCS1 `exit_stage_id = 2`.
///
/// ## Expected cost derivation
///
/// Stages are fully decoupled (no storage, no discount, finite horizon), so the
/// converged lower bound is the sum of per-stage optima. The NCS objective term
/// is `-curtailment_cost × block_hours × g_ncs = -0.73 × g_ncs` (per MW
/// dispatched); thermal is `+7300 × g_thermal`; excess is `+7.3 × g_excess`.
///
/// - Stage 0 (NCS1 active, 50 MW): NCS 50 + thermal 30.
///   `50 × (-0.73) + 30 × 7300 = -36.5 + 219_000 = 218_963.5`.
/// - Stage 1 (NCS0 + NCS1, 100 MW available, load 80): dispatching NCS beyond
///   the 80 MW load would force excess (`+7.3/MW`) against only `-0.73/MW` of
///   NCS incentive, so the optimum dispatches exactly 80 MW of NCS and zero
///   thermal: `80 × (-0.73) = -58.4`.
/// - Stage 2 (NCS0 active, 50 MW): NCS 50 + thermal 30 = `218_963.5`.
///
/// Total = `218_963.5 - 58.4 + 218_963.5 = 437_868.6`.
///
/// The case must stay feasible at stage 0, where NCS0 is not yet commissioned —
/// the unlimited deficit segment plus the 100 MW thermal cap guarantee it.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d18_ncs_commissioning_window() {
    let case_dir = Path::new("../../examples/deterministic/d18-ncs-commissioning-window");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, 437_868.6, 1e-2, "D18");
    assert!(
        result.iterations <= 10,
        "D18: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-4,
        "D18: gap={:.2e}",
        result.final_gap
    );
}

/// D16: PAR(1) lag-shift deterministic test.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 3 stages, constant productivity = 1.0 MW/(m3/s).
/// No storage (min=max=0). Max turbined = 200 m3/s, max generation = 200 MW.
/// Load = 200 MW constant. Deficit cost = 1000 $/MWh. Block = 730 hours.
///
/// ## PAR(1) model
///
/// psi = 0.5 at all stages, mean = 100 m3/s, std ~ 0 (deterministic).
/// Initial lag (past_inflows) = 200 m3/s.
///
/// ## Expected inflows with correct lag shift
///
/// Z_0 = 100 + 0.5 * (200 - 100) = 150 m3/s
/// Z_1 = 100 + 0.5 * (150 - 100) = 125 m3/s
/// Z_2 = 100 + 0.5 * (125 - 100) = 112.5 m3/s
///
/// ## Expected cost
///
/// Deficit per stage = 200 - Z_t MW.
/// Cost = sum_t[ deficit_t * 1000 * 730 ]
///      = (50 + 75 + 87.5) * 730000
///      = 212.5 * 730000
///      = 155_125_000
///
/// Without lag shift (bug): every stage sees lag=200, Z_t=150 for all t.
/// Cost = 3 * 50 * 730000 = 109_500_000. Fails with correct cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d16_par1_lag_shift() {
    let case_dir = Path::new("../../examples/deterministic/d16-par1-lag-shift");
    let result = run_deterministic(case_dir);
    assert!(
        result.final_lb > 0.0,
        "D16: lower bound must be positive, got {}",
        result.final_lb
    );
    // The lag shift makes inflows decrease across stages (Z = 150, 125, 112.5),
    // producing higher deficits than the no-shift bug (Z = 150 at every stage).
    // Pinned value adds the universal `turbined_cost` regularization (`≈ 2_828.75 $`)
    // to the deficit-only cost `7_756_250.0`.
    assert_cost(result.final_lb, 7_759_078.749_993_78, 1.0, "D16");
}

/// Regression guard for the model-persistence optimization.
///
/// Runs D01 (2 stages, 2 thermals, deterministic) and verifies that the
/// solver's `load_model_count` is consistent with per-stage loading, NOT
/// per-scenario loading. With model persistence, `load_model` is called
/// once per stage per iteration (forward + backward + lower bound), not
/// once per (scenario, stage) pair.
///
/// Numerical equivalence is verified by the `d01_thermal_dispatch` test;
/// this test additionally checks the call count invariant.
#[test]
fn model_persistence_regression_d01() {
    use cobre_solver::SolverInterface;

    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, 182_500.0, 1e-6, "D01-persistence");

    let stats = solver.statistics();
    let n_stages = 2_u64;
    let forward_passes = 2_u64;
    let iterations = result.iterations;

    let without_persistence_forward = n_stages * forward_passes * iterations;
    let with_persistence_forward = n_stages * iterations;

    // load_model_count (forward + backward + LB) must stay below the
    // per-scenario forward-only count, confirming per-stage persistence is active.
    assert!(
        stats.load_model_count < without_persistence_forward,
        "model persistence regression: load_model_count ({}) should be < {} (per-scenario forward-only count), \
         expected ~{} for persisted forward",
        stats.load_model_count,
        without_persistence_forward,
        with_persistence_forward
    );
}

// ---------------------------------------------------------------------------
// Incremental cut management integration tests
// ---------------------------------------------------------------------------

/// Verify the LB solver's incremental cut management reduces `load_model_count`
/// compared to the full-rebuild baseline.
///
/// Runs D03 (3-stage, 2-hydro cascade) which runs for 10 iterations. The LB
/// solver uses a dedicated CutRowMap and only calls `load_model` once (first
/// iteration). Forward + backward still call `load_model` per stage per
/// iteration.
///
/// Expected load_model breakdown with incremental LB:
/// - Forward: n_stages * iterations = 3 * 10 = 30
/// - Backward: (n_stages - 1) * iterations = 2 * 10 = 20
/// - LB: 1 (first iteration only, incremental thereafter)
/// - Total ~51
///
/// Without incremental LB, the total would be 30 + 20 + 10 = 60.
/// We verify that load_model_count is strictly less than the non-incremental total.
#[test]
fn incremental_lb_reduces_load_model_count() {
    use cobre_solver::SolverInterface;

    let case_dir = Path::new("../../examples/deterministic/d03-two-hydro-cascade");
    let (result, solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, D03_EXPECTED_COST, 1e-4, "D03-incremental");

    let stats = solver.statistics();
    let n_stages = 3_u64;
    let iterations = result.iterations;

    // Non-incremental LB would call load_model once per iteration; incremental
    // calls it only on the first. Total budget = forward + backward + LB.
    let non_incremental_lb = iterations;
    let forward_count = n_stages * iterations;
    let backward_count = (n_stages - 1) * iterations;
    let total_without_incremental = forward_count + backward_count + non_incremental_lb;

    assert!(
        stats.load_model_count < total_without_incremental,
        "incremental LB should reduce load_model_count: got {} >= {} (non-incremental total), \
         iterations={}, n_stages={}",
        stats.load_model_count,
        total_without_incremental,
        iterations,
        n_stages
    );

    // The LB solver does 1 load_model instead of `iterations`, so the reduction
    // is at least (iterations - 1).
    let expected_savings = iterations.saturating_sub(1);
    let actual_savings = total_without_incremental - stats.load_model_count;
    assert!(
        actual_savings >= expected_savings,
        "LB incremental savings should be >= {} (iterations - 1), got {} savings \
         (total_without={}, actual={})",
        expected_savings,
        actual_savings,
        total_without_incremental,
        stats.load_model_count
    );
}

/// Bit-for-bit equivalence spot check for the incremental cut-management path:
/// D01's converged trace must match its expected cost. All D-cases share this
/// pipeline, so D01 passing exercises the same incremental LB management.
#[test]
fn incremental_bit_for_bit_d01_trace() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let (result, _solver) = run_deterministic_with_solver(case_dir);

    assert_cost(result.final_lb, 182_500.0, 1e-6, "D01-trace");

    assert!(
        result.final_gap.abs() < 1e-6,
        "D01-trace: gap={:.2e} should be < 1e-6",
        result.final_gap
    );
}

/// Multi-hydro PAR(2) regression test with inflow truncation.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 200 MW at $50/MWh), 2 hydros:
///   - H0: constant productivity 1.0 MW/(m3/s), max turbined 100 m3/s,
///     storage 0–200 hm3, PAR(2) with psi = [0.5, 0.3], mean = 40 m3/s
///   - H1: constant productivity 0.8 MW/(m3/s), max turbined 80 m3/s,
///     storage 0–150 hm3, PAR(2) with psi = [0.4, 0.2], mean = 25 m3/s
/// - Deterministic load: 100 MW per stage
/// - Initial storage: H0 = 100 hm3, H1 = 75 hm3
/// - Past inflows: H0 = [50, 45] m3/s, H1 = [30, 28] m3/s
/// - 3 stages × 730 h, `inflow_non_negativity: {method: "truncation"}`
///
/// ## What this tests
///
/// With 2 hydros and PAR(2), the lag state indices are:
///   - `inflow_lags.start + 0*2 + 0` = hydro 0, lag 0
///   - `inflow_lags.start + 0*2 + 1` = hydro 0, lag 1  (BUG: was hydro 1, lag 0)
///   - `inflow_lags.start + 1*2 + 0` = hydro 1, lag 0
///   - `inflow_lags.start + 1*2 + 1` = hydro 1, lag 1  (BUG: was hydro 0, lag 1)
///
/// If the hydro-major/lag-major bug regressed, the wrong lag values would be
/// used in PAR evaluation, producing a different optimal cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d19_multi_hydro_par_truncation() {
    let case_dir = Path::new("../../examples/deterministic/d19-multi-hydro-par");
    let result = run_deterministic(case_dir);

    // The system with PAR(2) truncation and 2 hydros must produce a positive cost.
    assert!(
        result.final_lb > 0.0,
        "D19: lower bound must be positive, got {}",
        result.final_lb
    );
    // No iteration bound: the 2-hydro × 2-lag state space need not converge
    // within 50 iterations. The LB match is the lag-major indexing regression check.
    assert_cost(result.final_lb, D19_EXPECTED_COST, 1.0, "D19");
}

/// Expected lower bound for D19 (2-hydro PAR(2) with truncation, 3 stages).
///
/// Empirical, not hand-computable (2-hydro × 2-lag state space). D19's 3 study
/// stages all carry `season_id=0` in 2024, so `precompute_noise_groups` assigns
/// one group ID and `generate_opening_tree` makes stages 1 and 2 share stage 0's
/// correlated noise draws — a different but still deterministic optimal cost.
///
/// A lag-major/hydro-major indexing regression reads the wrong lag for each
/// hydro in PAR evaluation, producing a different cost.
pub const D19_EXPECTED_COST: f64 = 1_334_655.175_543_562_7;

/// Operational violation slacks: 1 hydro with active min_outflow, max_outflow,
/// min_turbined, and min_generation bounds.
///
/// ## Case setup (D20)
///
/// - 1 bus, 1 hydro, 0 thermals, 1 block (730h), 2 stages, deterministic.
/// - Hydro: min_outflow=40, max_outflow=50, min_turbined=30, min_generation=20,
///   max_turbined=50, productivity=1.0, max_storage=200, initial_storage=10.
/// - Inflows: stage 0 = 40 m3/s, stage 1 = 10 m3/s (zero std_dev).
/// - Penalty costs: all 4 operational violations = 5000 $/MWh, deficit = 1000 $/MWh.
///
/// ## Expected behaviour
///
/// With low initial storage (10 hm3), the hydro cannot sustain 40 m3/s
/// min_outflow at either stage. At stage 0, total available water is
/// 10 + 40*2.628 = 115.12 hm3, but the optimizer splits water across
/// stages, leading to outflow below 40 m3/s. At stage 1, even less water
/// is available, forcing min_outflow and min_turbined violation slacks.
///
/// The expected cost is recorded empirically and locked for regression.
/// Simulation is also run to verify non-zero operational violation slacks.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d20_operational_violations() {
    let case_dir = Path::new("../../examples/deterministic/d20-operational-violations");
    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D20: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D20: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D20_EXPECTED_COST, 1e-2, "D20");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(summary.mean_cost, D20_EXPECTED_COST, 1e-2, "D20-sim");

    // Verify operational violation slacks are non-zero in at least one stage.
    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    // Verify operational violation slacks fire in at least one stage.
    let mut found_outflow_below = false;
    let mut found_turbine_below = false;
    for stage_result in &scenario.stages {
        for hydro_result in &stage_result.hydros {
            if hydro_result.outflow_slack_below_m3s > 1e-10 {
                found_outflow_below = true;
            }
            if hydro_result.turbined_slack_m3s > 1e-10 {
                found_turbine_below = true;
            }
        }
    }
    assert!(
        found_outflow_below,
        "D20: expected non-zero outflow_slack_below_m3s"
    );
    assert!(
        found_turbine_below,
        "D20: expected non-zero turbined_slack_m3s"
    );
}

/// Empirical lower bound, initial_storage=10 hm3 (includes the universal
/// `turbined_cost` regularization).
pub const D20_EXPECTED_COST: f64 = 195_744_837.222_222_24;

/// LP consistency test: cost consistency between outflow violation slacks
/// and `hydro_violation_cost`. 1 hydro (min_outflow=50 m3/s), 1 thermal,
/// inflow=10 m3/s (insufficient), initial_storage=5 hm3, penalty=5000.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d21_min_outflow_regression() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d21-min-outflow-regression");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));
    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![20.0, 20.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("load_seasonal_stats.parquet"))
        .expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::try_new(
        Arc::clone(&inflow_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![10.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("inflow RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("inflow_seasonal_stats.parquet"))
        .expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    // Train + simulate.
    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D21: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D21: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D21_EXPECTED_COST, 1e-2, "D21");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(
        summary.mean_cost,
        result.final_lb,
        1e-2,
        "D21-sim-vs-training",
    );

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    let found_outflow_below = scenario
        .stages
        .iter()
        .flat_map(|s| &s.hydros)
        .any(|h| h.outflow_slack_below_m3s > 1e-10);
    assert!(
        found_outflow_below,
        "D21: expected non-zero outflow_slack_below_m3s"
    );

    // Per-block formulation: slack is in m3/s, objective = penalty * block_hours.
    let penalty = 5000.0_f64;
    let hours = 730.0_f64;

    let mut total_hydro_violation_cost = 0.0;
    for (s, stage_result) in scenario.stages.iter().enumerate() {
        assert_eq!(stage_result.hydros.len(), 1);
        assert_eq!(stage_result.costs.len(), 1);
        let slack_m3s = stage_result.hydros[0].outflow_slack_below_m3s;
        let stage_violation_cost = stage_result.costs[0].hydro_violation_cost;
        total_hydro_violation_cost += stage_violation_cost;

        if slack_m3s > 1e-10 {
            let expected_cost = slack_m3s * penalty * hours;
            let cost_diff = (stage_violation_cost - expected_cost).abs();
            assert!(
                cost_diff < 1e-2,
                "D21 stage {s}: hydro_violation_cost={stage_violation_cost}, \
                 expected={expected_cost}, diff={cost_diff}"
            );
        }
    }
    assert!(
        total_hydro_violation_cost > 0.0,
        "D21: hydro_violation_cost must be positive"
    );

    // Decomposed cost fields. D21 has only min-outflow violations, so every
    // other violation component must be zero and the below-cost must carry the
    // whole hydro_violation_cost.
    for (s, stage_result) in scenario.stages.iter().enumerate() {
        let cost = &stage_result.costs[0];

        let component_sum = cost.outflow_violation_below_cost
            + cost.outflow_violation_above_cost
            + cost.turbined_violation_cost
            + cost.generation_violation_cost
            + cost.evaporation_violation_cost
            + cost.withdrawal_violation_cost;
        assert!(
            (cost.hydro_violation_cost - component_sum).abs() < 1e-6,
            "D21 stage {s}: sum invariant failed: hydro_violation_cost={}, component_sum={}",
            cost.hydro_violation_cost,
            component_sum
        );

        assert!(
            cost.outflow_violation_above_cost.abs() < 1e-10,
            "D21 stage {s}: outflow_above should be 0, got {}",
            cost.outflow_violation_above_cost
        );
        assert!(
            cost.turbined_violation_cost.abs() < 1e-10,
            "D21 stage {s}: turbined should be 0, got {}",
            cost.turbined_violation_cost
        );
        assert!(
            cost.generation_violation_cost.abs() < 1e-10,
            "D21 stage {s}: generation should be 0, got {}",
            cost.generation_violation_cost
        );
        assert!(
            cost.evaporation_violation_cost.abs() < 1e-10,
            "D21 stage {s}: evaporation should be 0, got {}",
            cost.evaporation_violation_cost
        );
        assert!(
            cost.withdrawal_violation_cost.abs() < 1e-10,
            "D21 stage {s}: withdrawal should be 0, got {}",
            cost.withdrawal_violation_cost
        );

        let slack_m3s = stage_result.hydros[0].outflow_slack_below_m3s;
        if slack_m3s > 1e-10 {
            let expected_below_cost = slack_m3s * penalty * hours;
            assert!(
                (cost.outflow_violation_below_cost - expected_below_cost).abs() < 1e-2,
                "D21 stage {s}: outflow_violation_below_cost={}, expected={}",
                cost.outflow_violation_below_cost,
                expected_below_cost
            );
        }
    }

    let found_below_cost = scenario
        .stages
        .iter()
        .flat_map(|s| &s.costs)
        .any(|c| c.outflow_violation_below_cost > 1e-10);
    assert!(
        found_below_cost,
        "D21: expected non-zero outflow_violation_below_cost in at least one stage"
    );
}

/// Empirical lower bound, initial_storage=5 hm3, inflow=10 m3/s (includes the
/// universal `turbined_cost` regularization).
pub const D21_EXPECTED_COST: f64 = 285_716_271.0;

/// D22: Multi-block per-block min outflow regression test.
///
/// 1 hydro (min_outflow=30 m3/s), 1 thermal, 3 blocks per stage (200h, 300h, 230h),
/// inflow=10 m3/s. Violation of 20 m3/s in every block at penalty 5000 $/m3/s.
/// Validates that per-block constraints prevent the optimizer from concentrating
/// flow into one block while starving others.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d22_per_block_min_outflow() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d22-per-block-min-outflow");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));
    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![20.0, 20.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("load_seasonal_stats.parquet"))
        .expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::try_new(
        Arc::clone(&inflow_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![10.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("inflow RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("inflow_seasonal_stats.parquet"))
        .expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    let (result, scenario_results, summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D22: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D22: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );
    assert_cost(result.final_lb, D22_EXPECTED_COST, 1e-2, "D22");
    assert_eq!(summary.n_scenarios, 1);
    assert_cost(
        summary.mean_cost,
        result.final_lb,
        1e-2,
        "D22-sim-vs-training",
    );

    let scenario = &scenario_results[0];
    let block_hours = [200.0_f64, 300.0, 230.0];
    let penalty = 5000.0_f64;

    for (s, stage_result) in scenario.stages.iter().enumerate() {
        // 1 hydro × 3 blocks = 3 rows.
        assert_eq!(
            stage_result.hydros.len(),
            3,
            "D22 stage {s}: expected 3 per-block hydro rows"
        );

        for (b, hr) in stage_result.hydros.iter().enumerate() {
            // inflow=10 < min_outflow=30, so every block violates.
            assert!(
                hr.outflow_slack_below_m3s > 1e-6,
                "D22 stage {s} block {b}: outflow_slack_below_m3s should be > 0, got {}",
                hr.outflow_slack_below_m3s
            );
        }

        // Slack differs per block: penalty ∝ block_hours, so the optimizer
        // concentrates outflow into the longer (more expensive) blocks.
        assert_eq!(stage_result.costs.len(), 1);
        let total_violation_cost = stage_result.costs[0].hydro_violation_cost;
        let expected_total: f64 = stage_result
            .hydros
            .iter()
            .enumerate()
            .map(|(b, hr)| hr.outflow_slack_below_m3s * penalty * block_hours[b])
            .sum();
        assert!(
            (total_violation_cost - expected_total).abs() < 1e-2,
            "D22 stage {s}: hydro_violation_cost={total_violation_cost}, expected={expected_total}"
        );
    }
}

/// Empirical lower bound, multi-block (3 blocks) per-block min outflow
/// (includes the universal `turbined_cost` regularization).
pub const D22_EXPECTED_COST: f64 = 140_376_826.555_555_58;

/// D23: Bidirectional withdrawal -- over-withdrawal activation.
///
/// ## Case setup
///
/// - 1 bus (B0), 1 thermal (T0: 200 MW at $100/MWh), 1 hydro (H0: constant
///   productivity 1.0 MW/(m³/s), max turbine 20 m³/s, reservoir 0-10 hm³)
/// - Deterministic inflows: 50.0 m³/s per stage (high, to create water excess)
/// - Water withdrawal target: 5 m³/s per stage
/// - Deterministic load: 80.0 MW per stage
/// - Initial storage: 5.0 hm³
/// - 2 stages x 730 h, `inflow_non_negativity: none`
///
/// ## Penalty structure (asymmetric)
///
/// - `water_withdrawal_violation_pos_cost`: 1.0 (cheap over-withdrawal)
/// - `water_withdrawal_violation_neg_cost`: 10,000.0 (expensive under-withdrawal)
/// - `spillage_cost`: 1,000.0 (expensive spillage)
///
/// ## Why over-withdrawal activates
///
/// kappa = 730 * 3600 / 1e6 = 2.628 hm³/(m³/s)
///
/// Water excess per stage: inflow (50) - withdrawal_target (5) - max_turbine (20) = 25 m³/s
/// Storage fill from excess: 25 * 2.628 = 65.7 hm³ >> max_storage (10 hm³)
///
/// The solver must shed excess water. Two options:
/// 1. Spill: cost = 1,000 * 730 = 730,000 per m³/s
/// 2. Over-withdraw: cost = 1.0 * 730 = 730 per m³/s
///
/// Over-withdrawal is ~1000x cheaper, so the solver strongly prefers `ww_pos > 0`.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d23_bidirectional_withdrawal() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d23-bidirectional-withdrawal");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));
    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![80.0, 80.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("load_seasonal_stats.parquet"))
        .expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::try_new(
        Arc::clone(&inflow_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![50.0, 50.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("inflow RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("inflow_seasonal_stats.parquet"))
        .expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    let constraints_dir = case_dir.join("constraints");
    std::fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    let bounds_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("water_withdrawal_m3s", DataType::Float64, false),
    ]));
    let bounds_batch = RecordBatch::try_new(
        Arc::clone(&bounds_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![5.0, 5.0])),
        ],
    )
    .expect("bounds RecordBatch");
    let file = std::fs::File::create(constraints_dir.join("hydro_bounds.parquet"))
        .expect("create bounds parquet");
    let mut writer = ArrowWriter::try_new(file, bounds_schema, None).expect("ArrowWriter");
    writer.write(&bounds_batch).expect("write bounds batch");
    writer.close().expect("close bounds writer");

    // Train + simulate.
    let (result, scenario_results, _summary) = run_with_simulation(case_dir);

    assert!(
        result.iterations <= 20,
        "D23: iterations={} (expected <= 20)",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D23: gap={:.2e} (expected < 1e-6)",
        result.final_gap
    );

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), 2);

    let mut found_ww_pos = false;
    for stage_result in &scenario.stages {
        for hydro_result in &stage_result.hydros {
            if hydro_result.water_withdrawal_violation_pos_m3s > 1e-10 {
                found_ww_pos = true;
            }
            // Under-withdrawal must not activate: its neg cost is high.
            assert!(
                hydro_result.water_withdrawal_violation_neg_m3s < 1e-10,
                "D23: unexpected under-withdrawal violation: {}",
                hydro_result.water_withdrawal_violation_neg_m3s
            );
        }
    }
    assert!(
        found_ww_pos,
        "D23: expected non-zero water_withdrawal_violation_pos_m3s (over-withdrawal)"
    );

    // Water balance identity, per stage:
    // V_out = V_in + kappa * (inflow - ww_target + ww_neg - ww_pos - turbined - spillage)
    let kappa = 730.0 * 3600.0 / 1e6; // hm3 per (m3/s)
    let ww_target = 5.0;
    let inflow = 50.0;

    for (s, stage_result) in scenario.stages.iter().enumerate() {
        assert_eq!(stage_result.hydros.len(), 1);
        let h = &stage_result.hydros[0];

        let net_flow = inflow - ww_target + h.water_withdrawal_violation_neg_m3s
            - h.water_withdrawal_violation_pos_m3s
            - h.turbined_m3s
            - h.spillage_m3s;
        let expected_v_out = h.storage_initial_hm3 + kappa * net_flow;
        let diff = (h.storage_final_hm3 - expected_v_out).abs();
        assert!(
            diff < 1e-6,
            "D23 stage {s}: water balance mismatch: V_out={}, expected={expected_v_out}, diff={diff}",
            h.storage_final_hm3
        );
    }
}

/// Verify per-block bus balance: generation + deficit - excess + net_exchange =
/// load. Catches LP-extraction mismatches where a bus-balance coefficient
/// (e.g. productivity) diverges between the LP builder and the extraction pipeline.
///
/// # Limitations
///
/// Entity result structs carry no `bus_id`, so this sums generation across all
/// buses — accurate only for single-bus systems. Multi-bus systems with exchange
/// lines would need a bus-entity mapping for per-bus balance.
fn assert_bus_balance(stage: &cobre_sddp::SimulationStageResult, tolerance: f64, label: &str) {
    let mut block_ids: Vec<u32> = stage.buses.iter().filter_map(|b| b.block_id).collect();
    block_ids.sort_unstable();
    block_ids.dedup();

    for &block_id in &block_ids {
        let hydro_gen: f64 = stage
            .hydros
            .iter()
            .filter(|h| h.block_id == Some(block_id))
            .map(|h| h.generation_mw)
            .sum();
        let thermal_gen: f64 = stage
            .thermals
            .iter()
            .filter(|t| t.block_id == Some(block_id))
            .map(|t| t.generation_mw)
            .sum();
        let ncs_gen: f64 = stage
            .non_controllables
            .iter()
            .filter(|n| n.block_id == Some(block_id))
            .map(|n| n.generation_mw)
            .sum();
        let deficit: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.deficit_mw)
            .sum();
        let excess: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.excess_mw)
            .sum();
        // Net exchange sign: direct flow enters the bus, reverse flow leaves.
        let net_exchange: f64 = stage
            .exchanges
            .iter()
            .filter(|e| e.block_id == Some(block_id))
            .map(|e| e.direct_flow_mw - e.reverse_flow_mw)
            .sum();
        let load: f64 = stage
            .buses
            .iter()
            .filter(|b| b.block_id == Some(block_id))
            .map(|b| b.load_mw)
            .sum();

        let supply = hydro_gen + thermal_gen + ncs_gen + deficit - excess + net_exchange;
        let mismatch = (supply - load).abs();
        assert!(
            mismatch < tolerance,
            "{label} stage {} block {block_id}: bus balance mismatch: \
             supply={supply:.6} (hydro={hydro_gen:.6} + thermal={thermal_gen:.6} \
             + ncs={ncs_gen:.6} + deficit={deficit:.6} - excess={excess:.6} \
             + exchange={net_exchange:.6}) vs load={load:.6}, diff={mismatch:.2e}",
            stage.stage_id
        );
    }
}

/// Expected cost for D24: per-stage productivity override (rho_0=0.8, rho_1=1.2).
///
/// ## Case setup
///
/// Same physical system as D02: 1 bus, 1 thermal (T0: 100 MW at $50/MWh),
/// 1 hydro (H0: max 50 m3/s, storage 0-200 hm3, max_generation 50 MW),
/// demand 80 MW, 2 stages x 730 h, initial storage 100 hm3, deterministic
/// inflows 40/10 m3/s.
///
/// Unlike D02, H0 uses per-stage productivity overrides via
/// `hydro_production_models.json`: stage 0 rho=0.8, stage 1 rho=1.2.
/// The entity-level `productivity_mw_per_m3s = 1.0` must NOT be used.
///
/// ## Expected cost derivation
///
/// kappa = 730 * 3600 / 1e6 = 657/250 = 2.628 hm3/(m3/s * stage).
///
/// Key constraint interaction: with rho_1=1.2, the generation cap (50 MW)
/// limits effective turbining: gen_h1 = 1.2 * q1 <= 50 => q1 <= 125/3 m3/s.
/// In stage 0, rho_0=0.8 leaves q0 <= 50 (gen_h0 = 0.8*50 = 40 < 50 MW).
///
/// Since rho_1 > rho_0, water is more valuable in stage 1. The optimizer
/// stores water in stage 0 to use it at higher productivity in stage 1.
/// The optimal V1 = (125/3 - 10) * kappa = 4161/50 = 83.22 hm3, making
/// V2 = 0 with q1 = 125/3 m3/s (generation cap binding).
///
/// Stage 0: q0 = 30475/657 m3/s, gen_h0 = 0.8 * q0 = 24380/657 MW,
///   g_th0 = 80 - 24380/657 = 28180/657 MW.
///   Cost_0 = 50 * 730 * 28180/657 = 14090000/9.
///
/// Stage 1: q1 = 125/3 m3/s, gen_h1 = 1.2 * 125/3 = 50 MW (cap),
///   g_th1 = 80 - 50 = 30 MW, V2 = 0 hm3.
///   Cost_1 = 50 * 730 * 30 = 1095000.
///
/// Total = 14090000/9 + 1095000 = 23945000/9 ~ 2660555.56 $.
///
/// ## Bug detection
///
/// If the bug were present (using entity rho=1.0 instead of the overrides),
/// the cost would equal D02: 23635000/9 ~ 2626111.11 $. The difference
/// (~ $34444) is well above the 1e-4 tolerance, so the test catches the bug.
///
/// The pinned value adds D02's universal `turbined_cost` regularization
/// (`5_785 / 9 ≈ 642.78 $`, same single-hydro turbined flows) to the analytical
/// thermal cost `23_945_000 / 9`, yielding `23_950_785 / 9`.
pub const D24_EXPECTED_COST: f64 = 23_950_785.0 / 9.0;

/// D24: Productivity override — per-stage productivity from `hydro_production_models.json`.
///
/// Same physical system as D02 except H0 has per-stage productivity overrides:
/// stage 0 -> rho = 0.8, stage 1 -> rho = 1.2 (entity model says 1.0).
///
/// This test catches the bus balance productivity mismatch bug: if the LP uses
/// the entity-level productivity (1.0) instead of the per-stage override, the
/// optimal cost would differ.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d24_productivity_override() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let case_dir = Path::new("../../examples/deterministic/d24-productivity-override");

    let scenarios_dir = case_dir.join("scenarios");
    std::fs::create_dir_all(&scenarios_dir).expect("create scenarios dir");

    let inflow_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ]));
    let inflow_batch = RecordBatch::try_new(
        Arc::clone(&inflow_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![40.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("inflow RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("inflow_seasonal_stats.parquet"))
        .expect("create inflow parquet");
    let mut writer = ArrowWriter::try_new(file, inflow_schema, None).expect("ArrowWriter");
    writer.write(&inflow_batch).expect("write inflow batch");
    writer.close().expect("close inflow writer");

    let load_schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ]));
    let load_batch = RecordBatch::try_new(
        Arc::clone(&load_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Int32Array::from(vec![0, 1])),
            Arc::new(Float64Array::from(vec![80.0, 80.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0])),
        ],
    )
    .expect("load RecordBatch");
    let file = std::fs::File::create(scenarios_dir.join("load_seasonal_stats.parquet"))
        .expect("create load parquet");
    let mut writer = ArrowWriter::try_new(file, load_schema, None).expect("ArrowWriter");
    writer.write(&load_batch).expect("write load batch");
    writer.close().expect("close load writer");

    let (result, scenario_results, _summary) = run_with_simulation(case_dir);
    assert_cost(result.final_lb, D24_EXPECTED_COST, 1e-4, "D24");
    assert!(
        result.iterations <= 10,
        "D24: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D24: gap={:.2e}",
        result.final_gap
    );
    assert!(
        (result.final_lb - D02_EXPECTED_COST).abs() > 1.0,
        "D24: cost must differ from D02 (per-stage overrides change economics)"
    );

    assert_eq!(
        scenario_results.len(),
        1,
        "D24: expected 1 simulation scenario"
    );
    for stage in &scenario_results[0].stages {
        assert_bus_balance(stage, 1e-3, "D24");
    }
}

// ---------------------------------------------------------------------------
// D25: Discount rate
// ---------------------------------------------------------------------------

/// Expected cost for D25 (single hydro, 2 stages, 12% annual discount rate).
///
/// Same physical system as D02 but with `annual_discount_rate: 0.12`.
/// The one-step discount factor for stage 0 (31-day January) is:
///
/// `d_0 = 1 / (1.12)^(31/365.25) ≈ 0.9904`
///
/// The discount factor multiplies the theta (future cost) coefficient in
/// the stage-0 LP objective, reducing the present value of future costs.
/// This shifts the optimal dispatch toward less water conservation, yielding
/// a lower total present-value cost than the undiscounted D02 case.
/// Includes the universal `turbined_cost` regularization (`≈ 640.12 $`).
const D25_EXPECTED_COST: f64 = 2_612_094.703_543_594_6;

/// D25: Two-stage single-hydro with 12% annual discount rate.
///
/// Verifies that the discounted SDDP lower bound converges to the correct
/// present-value cost, and that it is strictly less than D02's undiscounted LB.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d25_discount_rate() {
    let case_dir = Path::new("../../examples/deterministic/d25-discount-rate");
    let result = run_deterministic(case_dir);
    assert_cost(result.final_lb, D25_EXPECTED_COST, 1e-4, "D25");
    assert!(
        result.iterations <= 10,
        "D25: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D25: gap={:.2e}",
        result.final_gap
    );
    assert!(
        result.final_lb < D02_EXPECTED_COST,
        "D25: discounted LB ({}) must be < undiscounted D02 LB ({})",
        result.final_lb,
        D02_EXPECTED_COST,
    );
}

/// D25: Verify simulation discount factors match expected cumulative factors.
///
/// Runs training + simulation on the D25 case and asserts that:
/// - Stage 0 cumulative discount factor = 1.0 (always)
/// - Stage 1 cumulative discount factor = d_0 = 1/(1.12)^(31/365.25)
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d25_simulation_discount_factors() {
    let case_dir = Path::new("../../examples/deterministic/d25-discount-rate");
    let (result, scenarios, _summary) = run_with_simulation(case_dir);

    assert_cost(result.final_lb, D25_EXPECTED_COST, 1e-4, "D25-sim");

    assert_eq!(scenarios.len(), 1, "D25: expected 1 simulation scenario");
    let stages = &scenarios[0].stages;
    assert_eq!(stages.len(), 2, "D25: expected 2 stages");

    // Stage 0 cumulative discount factor is always 1.0.
    let df0 = stages[0].costs[0].discount_factor;
    assert!(
        (df0 - 1.0).abs() < 1e-12,
        "D25: stage 0 discount_factor expected 1.0, got {df0}"
    );

    let d0 = 1.0_f64 / 1.12_f64.powf(31.0 / 365.25);
    let df1 = stages[1].costs[0].discount_factor;
    assert!(
        (df1 - d0).abs() < 1e-10,
        "D25: stage 1 discount_factor expected {d0}, got {df1}"
    );
}

// ---------------------------------------------------------------------------
// D26: Estimated PAR(2) — regression guard for the forward-prediction fix
// ---------------------------------------------------------------------------

/// D26 expected lower bound — regression guard that PAR prediction uses the
/// forward (not backward) lag. Includes the universal `turbined_cost`
/// regularization (`≈ 17_830.06 $`).
pub const D26_EXPECTED_COST: f64 = 50_625_314.970_196_81;

/// D26: PAR(2) estimation from inflow history (regression guard for forward-prediction fix).
/// Exercises full PAR(p) pipeline with PACF order selection and Yule-Walker fitting.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d26_estimated_par2() {
    let case_dir = Path::new("../../examples/deterministic/d26-estimated-par2");
    let result = run_deterministic(case_dir);

    assert!(
        result.final_lb > 0.0,
        "D26: lower bound must be positive, got {}",
        result.final_lb
    );
    assert_cost(result.final_lb, D26_EXPECTED_COST, 1.0, "D26");
    assert!(
        result.iterations <= 100,
        "D26: must converge within 100 iterations, got {}",
        result.iterations
    );
}

/// D26: Verify PACF order selection picks AR order 2.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d26_estimated_par2_order_selection() {
    use cobre_sddp::prepare_stochastic;

    let case_dir = Path::new("../../examples/deterministic/d26-estimated-par2");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let prepare_result =
        prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");

    let report = prepare_result
        .estimation_report
        .expect("estimation report must be Some");

    assert_eq!(report.entries.len(), 1, "expected 1 hydro entry");

    let (hydro_id, entry) = report.entries.iter().next().expect("entry exists");
    assert_eq!(
        entry.selected_order, 2,
        "expected AR order 2 for hydro {hydro_id}, got {}",
        entry.selected_order
    );
}

// ---------------------------------------------------------------------------
// D27: Per-stage thermal cost override
// ---------------------------------------------------------------------------

/// Expected cost for D27 (2-thermal system, stage-varying costs).
///
/// ## Case setup
///
/// - 1 bus (B0), 2 thermals, no hydro, deterministic load 100 MW.
/// - T1 (id=0): base cost 50 $/MWh, capacity 0-60 MW.
/// - T2 (id=1): base cost 80 $/MWh, capacity 0-80 MW.
/// - `thermal_bounds.parquet` overrides T1 cost at stage 1 to 120 $/MWh.
/// - 2 stages × 730 h each.
///
/// ## Expected cost derivation
///
/// Stage 0 (T1 at 50 $/MWh, T2 at 80 $/MWh — T1 dispatched first):
/// - T1 at full capacity: 60 MW × 50 $/MWh × 730 h = 2 190 000 $
/// - T2 covers residual: 40 MW × 80 $/MWh × 730 h = 2 336 000 $
/// - Stage 0 cost = 4 526 000 $
///
/// Stage 1 (T1 at 120 $/MWh via override, T2 at 80 $/MWh — T2 dispatched first):
/// - T2 at full capacity: 80 MW × 80 $/MWh × 730 h = 4 672 000 $
/// - T1 covers residual: 20 MW × 120 $/MWh × 730 h = 1 752 000 $
/// - Stage 1 cost = 6 424 000 $
///
/// Total = 4 526 000 + 6 424 000 = **10 950 000 $**
///
/// Compared to the uniform-cost baseline (T1 at 50 $/MWh in both stages):
/// - Uniform total = 2 × 4 526 000 = 9 052 000 $
/// - D27 total must be strictly greater, confirming the override is applied.
pub const D27_EXPECTED_COST: f64 = 10_950_000.0;

/// D27: Per-stage thermal cost override via `constraints/thermal_bounds.parquet`.
///
/// Uses pre-committed parquet fixtures (scenarios + constraints) to verify that
/// the LP objective coefficients use the resolved per-stage cost rather than the
/// entity base cost.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d27_per_stage_thermal_cost() {
    let case_dir = Path::new("../../examples/deterministic/d27-per-stage-thermal-cost");

    let result = run_deterministic(case_dir);

    assert_cost(result.final_lb, D27_EXPECTED_COST, 1e-4, "D27");
    assert!(
        result.iterations <= 10,
        "D27: iterations={}",
        result.iterations
    );
    assert!(
        result.final_gap.abs() < 1e-6,
        "D27: gap={:.2e}",
        result.final_gap
    );

    // Exceeding the uniform-cost baseline confirms the override is applied and
    // reorders stage-1 dispatch (see D27_EXPECTED_COST for the baseline).
    let uniform_baseline = 9_052_000.0_f64;
    assert!(
        result.final_lb > uniform_baseline,
        "D27: per-stage cost override must increase total cost vs uniform baseline \
         ({} > {})",
        result.final_lb,
        uniform_baseline
    );
}

/// D28: Mixed-resolution case (5 weekly + 1 monthly stages).
///
/// Smoke test that verifies the full pipeline loads and trains without error
/// on a case with:
/// - Non-uniform `num_scenarios` (1 per weekly stage, 5 for the monthly stage)
/// - `season_definitions` with monthly cycle (12 seasons)
/// - External inflow scenario source
/// - `recent_observations` in initial conditions
///
/// The test only checks that training completes at least 1 iteration; no
/// expected cost is asserted here
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d28_decomp_weekly_monthly_loads_and_trains() {
    let case_dir = Path::new("../../examples/deterministic/d28-decomp-weekly-monthly");

    let result = run_deterministic(case_dir);

    assert!(
        result.iterations > 0,
        "D28: must complete at least 1 iteration"
    );
}

/// D29: weekly stages with PAR(1) noise-group sharing.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 4 weekly stages in January 2024 (all season_id=0),
/// PAR(1) with psi=0.5, OutOfSample noise, inflow_lags=true.
///
/// ## What this tests
///
/// - All 4 weekly stages share the same noise group ID (group 0).
/// - Training with noise sharing completes without error.
/// - Simulation completes with sensible costs.
///
/// This is the end-to-end verification that noise group precomputation,
/// ForwardSampler integration, opening tree integration, and setup wiring
/// compose correctly for the noise-group-sharing workflow.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d29_weekly_par_noise_sharing() {
    let case_dir = Path::new("../../examples/deterministic/d29-weekly-par-noise-sharing");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Build the training scenario source from the config so the seed and
    // OutOfSample scheme are propagated to the forward-pass noise generator.
    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &training_source)
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let groups = &setup.stage_data.noise_group_ids;
    assert_eq!(groups.len(), 4, "expected 4 study stages");
    assert!(
        groups.iter().all(|&g| g == groups[0]),
        "all weekly stages in the same month must share the same group ID, got {groups:?}"
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "D29: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;

    assert!(
        result.iterations > 0,
        "D29: must complete at least 1 iteration"
    );
    assert!(
        result.final_lb > 0.0,
        "D29: lower bound must be positive, got {}",
        result.final_lb
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let _local_costs = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            result.baked_templates.as_deref(),
            &result.basis_cache,
        )
        .expect("simulation must succeed");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(
        scenario_results.len(),
        1,
        "D29: expected 1 simulation scenario result"
    );
}

/// D30: monthly-to-quarterly multi-resolution stage transition.
///
/// ## System
///
/// 1 bus, 1 hydro (H0), 6 monthly stages (Jan-Jun 2024, season_id 0-5) followed
/// by 4 quarterly stages (Q3 2024 – Q2 2025, season_id 12-15). Custom SeasonMap
/// with 12 monthly + 4 quarterly season definitions. PAR(1) with psi=0.5,
/// OutOfSample noise, inflow_lags=true for all stages.
///
/// ## What this tests
///
/// - Case loads and trains without error on a Custom-cycle multi-resolution study.
/// - Training completes at least 1 iteration with a positive lower bound.
///
/// Full structural and downstream-lag-transition assertions are in the dedicated
/// `multi_resolution_integration.rs` test file, which verifies composition
/// correctness including noise group IDs, accumulate_downstream flags,
/// rebuild_from_downstream, and simulation.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d30_multi_resolution_loads_and_trains() {
    let case_dir = Path::new("../../examples/deterministic/d30-multi-resolution-monthly-quarterly");

    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Use the config's OutOfSample training source so PAR noise is correctly seeded.
    let training_source = config
        .training_scenario_source(&config_path)
        .expect("training_scenario_source must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(system, case_dir, &config, 42, &training_source)
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "D30: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;

    assert!(
        result.iterations > 0,
        "D30: must complete at least 1 iteration"
    );
    assert!(
        result.final_lb > 0.0,
        "D30: lower bound must be positive, got {}",
        result.final_lb
    );
}

// ── integration test ──────────────────────────────────────────────

/// The baked-template simulation path must produce bit-for-bit identical
/// per-scenario costs to the legacy fallback path (rel error == 0.0, within
/// 1e-12), confirming `bake_rows_into_template` builds a mathematically
/// equivalent LP to `load_model + add_rows`. Trained on D01, which runs >= 2
/// iterations so `baked_templates` is `Some`.
#[test]
fn baked_vs_fallback_simulation_costs_are_identical() {
    let case_dir = Path::new("../../examples/deterministic/d01-thermal-dispatch");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("config must parse");

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let pr = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &cobre_core::scenario::ScenarioSource::default(),
    )
    .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut config_with_sim = config.clone();
    config_with_sim.simulation.enabled = true;
    config_with_sim.simulation.num_scenarios = 4;

    let mut setup = StudySetup::new(&system, &config_with_sim, stochastic, hydro_models)
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "expected no training error");
    let training_result = outcome.result;

    assert!(
        training_result.baked_templates.is_some(),
        "D01 training must produce baked templates (requires >= 2 iterations)"
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (tx_baked, rx_baked) = mpsc::sync_channel(io_capacity);
    let drain_baked = std::thread::spawn(move || rx_baked.into_iter().collect::<Vec<_>>());

    let baked_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &tx_baked,
            None,
            training_result.baked_templates.as_deref(),
            &training_result.basis_cache,
        )
        .expect("baked-path simulate must return Ok");
    drop(tx_baked);
    drop(drain_baked.join().expect("drain thread must not panic"));

    let (tx_fallback, rx_fallback) = mpsc::sync_channel(io_capacity);
    let drain_fallback = std::thread::spawn(move || rx_fallback.into_iter().collect::<Vec<_>>());

    let fallback_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &tx_fallback,
            None,
            None, // force legacy path
            &training_result.basis_cache,
        )
        .expect("fallback-path simulate must return Ok");
    drop(tx_fallback);
    drop(drain_fallback.join().expect("drain thread must not panic"));

    assert_eq!(
        baked_run.costs.len(),
        fallback_run.costs.len(),
        "both runs must return the same number of scenarios"
    );

    for ((b_id, b_cost, _), (f_id, f_cost, _)) in
        baked_run.costs.iter().zip(fallback_run.costs.iter())
    {
        assert_eq!(b_id, f_id, "scenario IDs must match between runs");
        let rel_err = if b_cost.abs() > 1e-10 {
            (b_cost - f_cost).abs() / b_cost.abs()
        } else {
            (b_cost - f_cost).abs()
        };
        assert!(
            rel_err < 1e-12,
            "scenario {b_id}: baked cost {b_cost} != fallback cost {f_cost} (rel_err={rel_err})"
        );
    }
}

/// D43: storage-only (lag-dropped) cut convergence.
///
/// ## System
///
/// 1 bus, 1 hydro (H0, constant productivity 0.8 MW/(m3/s), 0–500 hm3 storage),
/// 1 thermal (300 MW @ 100 $/MWh), 4 monthly stages, load 220 MW constant.
/// PACF with `max_order=2` fits a PAR(2) model (global `n_state = N*(1+L) = 3`).
///
/// ## Storage-only stage
///
/// Stage 2 sets `state_variables.inflow_lags = false`. Pool `t` is sized by stage
/// `t+1`'s config, so stage 2's storage-only config sizes **pool 1** at cut
/// dimension `N = 1` (the AR lags are dropped from that pool's cuts) while the
/// lag-enabled pools stay at `N*(1+L) = 3`. The reduced cut, rendered through the
/// per-pool outgoing projection, touches only the storage column.
///
/// ## Convergence
///
/// The lag-dropped cut is an approximation change; this case proves it still
/// closes the LB/UB gap. The run terminates on `bound_stalling` (the LB
/// stabilizes — it does not plateau below the UB), with LB == UB to machine
/// precision.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn d43_storage_only_cut_converges() {
    let case_dir = Path::new("../../examples/deterministic/d43-storage-only-cut");

    // Pool dimensions: stage 2's storage-only config sizes pool 1 down to N,
    // while the lag-enabled pools carry storage + lags.
    let config = cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
    let pr = prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");
    let system = pr.system;
    let stochastic = pr.stochastic;
    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");
    let mut setup = build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models);
    let n_hydros = setup.stage_state().hydro_count;
    let global_n_state = setup.stage_state().n_state;
    assert!(
        global_n_state > n_hydros,
        "D43: PACF must fit PAR(p>0) so the lag drop is non-trivial (n_state={global_n_state}, N={n_hydros})",
    );
    // ONLY pool 1 is reduced (sized by stage 2's storage-only config); every
    // sibling stays at the full global dimension. Pinning all four catches both an
    // under-reduction (pool 1 not dropped) and an over-eager reduction (a sibling
    // wrongly dropped).
    assert_eq!(
        setup.fcf.pools[1].state_dimension, n_hydros,
        "D43: storage-only pool 1 must have cut dimension N (lags dropped)",
    );
    for t in [0usize, 2, 3] {
        assert_eq!(
            setup.fcf.pools[t].state_dimension, global_n_state,
            "D43: lag-enabled pool {t} must stay at the full global dimension",
        );
    }

    // Convergence: the storage-only cut closes the LB/UB gap and the run stops on
    // bound stalling (not the iteration cap), proving the LB does not plateau.
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(outcome.error.is_none(), "D43: expected no training error");
    let result = outcome.result;
    assert_eq!(
        result.reason, "bound_stalling",
        "D43: must converge via bound_stalling, not exhaust the iteration cap (reason={})",
        result.reason,
    );
    assert!(
        result.final_gap < 1e-6,
        "D43: LB/UB gap must close (gap={})",
        result.final_gap,
    );
    assert_cost(result.final_lb, 11_658_487.253_236_46, 1.0, "D43");

    // The cut(s) stored in the reduced pool carry the reduced length: a cut on a
    // storage-only pool has exactly N coefficients, not the global n_state. This is
    // what discriminates the per-stage-dimension-aware storage from a revert to
    // full-dimension (the bound alone is neutral — an exact reduction stores
    // numerically the same cut a full pool would).
    let reduced_pool = &setup.fcf.pools[1];
    assert!(
        reduced_pool.active_count() > 0,
        "D43: the reduced pool must hold at least one trained cut to inspect",
    );
    for (slot, _intercept, coefficients) in reduced_pool.active_cuts() {
        assert_eq!(
            coefficients.len(),
            n_hydros,
            "D43: stored cut at pool-1 slot {slot} must have reduced length N={n_hydros}",
        );
    }
}
