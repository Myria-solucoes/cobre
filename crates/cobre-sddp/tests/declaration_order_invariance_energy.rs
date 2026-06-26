//! Declaration-order invariance: the five energy columns
//! (`equivalent_productivity_mw_per_m3s`, `accumulated_productivity_mw_per_m3s`,
//! `incremental_inflow_energy_mw`, `stored_energy_initial_mwh`,
//! `stored_energy_final_mwh`) must be bit-for-bit identical regardless of the order
//! in which hydro plants are declared in the input JSON.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{
    StudySetup, aggregate_simulation,
    hydro_models::prepare_hydro_models,
    setup::{StudyParams, prepare_stochastic},
    simulation::SimulationScenarioResult,
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

fn build_setup_for_case(
    case_dir: &Path,
    config: &cobre_io::Config,
    system: &cobre_core::System,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: cobre_sddp::PrepareHydroModelsResult,
) -> StudySetup {
    let _ = case_dir; // productivity override rides on hydro_models, not case_dir
    let sentinel = Path::new("config.json");
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

fn run_with_simulation(case_dir: &Path) -> Vec<SimulationScenarioResult> {
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
    let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
        .expect("aggregate_simulation must succeed");

    scenario_results
}

/// Copies only regular files; symlinks and other special files are skipped.
fn copy_dir_recursive(src_dir: &Path, dst_dir: &Path) {
    std::fs::create_dir_all(dst_dir)
        .expect("staging temp dir must be writable: create_dir_all failed");

    for entry in std::fs::read_dir(src_dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", src_dir.display()))
    {
        let entry = entry.expect("directory entry must be readable");
        let src_path = entry.path();
        let dst_path = dst_dir.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else if src_path.is_file() {
            std::fs::copy(&src_path, &dst_path).unwrap_or_else(|e| {
                panic!(
                    "failed to copy {} to {}: {e}",
                    src_path.display(),
                    dst_path.display()
                )
            });
        }
    }
}

fn reverse_hydros_json(case_dir: &Path) {
    let hydros_path = case_dir.join("system").join("hydros.json");
    let content = std::fs::read_to_string(&hydros_path)
        .expect("staging temp dir must be writable: read hydros.json failed");

    let mut value: serde_json::Value =
        serde_json::from_str(&content).expect("hydros.json must be valid JSON");

    let hydros_arr = value
        .get_mut("hydros")
        .and_then(serde_json::Value::as_array_mut)
        .expect("hydros.json must contain a top-level \"hydros\" array");

    hydros_arr.reverse();

    let pretty = serde_json::to_string_pretty(&value)
        .expect("hydros.json round-trip serialization must succeed");

    std::fs::write(&hydros_path, pretty)
        .expect("staging temp dir must be writable: write hydros.json failed");
}

/// Tuple types match the field types in [`cobre_sddp::SimulationHydroResult`].
type RecordKey = (i32, u32, Option<u32>);

/// Raw float bits for bit-exact comparison, positionally parallel to
/// [`ENERGY_COLUMN_NAMES`]: `[equivalent_productivity, accumulated_productivity,
/// incremental_inflow_energy, stored_energy_initial, stored_energy_final]`.
type EnergyBits = [u64; 5];

/// Sorting by (`hydro_id`, `stage_id`, `block_id`) is essential: the two runs emit
/// hydro records in different slice order, so comparison must key on logical entity
/// ID, not slice position.
fn collect_energy_records(
    scenario_results: &[SimulationScenarioResult],
) -> Vec<(RecordKey, EnergyBits)> {
    let mut records: Vec<(RecordKey, EnergyBits)> = scenario_results
        .iter()
        .flat_map(|s| s.stages.iter())
        .flat_map(|stage| {
            stage.hydros.iter().map(|h| {
                let key: RecordKey = (h.hydro_id, h.stage_id, h.block_id);
                let bits: EnergyBits = [
                    h.equivalent_productivity_mw_per_m3s.to_bits(),
                    h.accumulated_productivity_mw_per_m3s.to_bits(),
                    h.incremental_inflow_energy_mw.to_bits(),
                    h.stored_energy_initial_mwh.to_bits(),
                    h.stored_energy_final_mwh.to_bits(),
                ];
                (key, bits)
            })
        })
        .collect();

    records.sort_by_key(|(k, _)| *k);
    records
}

/// Column names parallel to the [`EnergyBits`] array layout, used in
/// diagnostic messages.
const ENERGY_COLUMN_NAMES: [&str; 5] = [
    "equivalent_productivity_mw_per_m3s",
    "accumulated_productivity_mw_per_m3s",
    "incremental_inflow_energy_mw",
    "stored_energy_initial_mwh",
    "stored_energy_final_mwh",
];

/// Declaration-order invariance is a hard project rule: the energy columns must be
/// bit-for-bit identical for `hydros.json` in original (`[H0, H1]`) versus reversed
/// (`[H1, H0]`) order, end-to-end through train + simulate.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn declaration_order_invariance_d03_energy_columns() {
    let d03_src: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("cobre-sddp crate must have a parent directory")
        .parent()
        .expect("crates/ must have a parent directory")
        .join("examples/deterministic/d03-two-hydro-cascade");

    assert!(
        d03_src.is_dir(),
        "D03 fixture must exist at {}",
        d03_src.display()
    );

    let tmp_original = tempfile::tempdir().expect("original temp dir must be created");
    let tmp_reversed = tempfile::tempdir().expect("reversed temp dir must be created");

    copy_dir_recursive(&d03_src, tmp_original.path());
    copy_dir_recursive(&d03_src, tmp_reversed.path());

    reverse_hydros_json(tmp_reversed.path());

    let results_original = run_with_simulation(tmp_original.path());
    let results_reversed = run_with_simulation(tmp_reversed.path());

    let records_original = collect_energy_records(&results_original);
    let records_reversed = collect_energy_records(&results_reversed);

    assert_eq!(
        records_original.len(),
        records_reversed.len(),
        "both runs must produce the same number of (hydro, stage, block) records: \
         original={}, reversed={}",
        records_original.len(),
        records_reversed.len()
    );

    assert!(
        !records_original.is_empty(),
        "simulation must produce at least one hydro result record"
    );

    for ((key_o, bits_o), (key_r, bits_r)) in records_original.iter().zip(records_reversed.iter()) {
        assert_eq!(
            key_o, key_r,
            "sorted record keys must match: original={key_o:?}, reversed={key_r:?}"
        );

        let (hydro_id, stage_id, block_id) = key_o;

        for col in 0..5 {
            assert_eq!(
                bits_o[col], bits_r[col],
                "declaration-order invariance violated: \
                 hydro_id={hydro_id}, stage_id={stage_id}, block_id={block_id:?}, \
                 column=\"{}\": \
                 original={:#018x}, reversed={:#018x}",
                ENERGY_COLUMN_NAMES[col], bits_o[col], bits_r[col]
            );
        }
    }
}
