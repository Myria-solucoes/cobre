//! Integration test: end-to-end run with anticipated thermals.
//!
//! Verifies that `simulation/thermals/` carries non-null, semantically
//! correct values in the three columns `is_anticipated`,
//! `anticipated_decision_mw`, and `anticipated_committed_mw` after a
//! full training + simulation run on a fixture with one regular and one
//! anticipated thermal plant.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]

use std::fs;
use std::path::Path;
use std::process::Command;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array};
use assert_cmd::prelude::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Config with simulation enabled (1 scenario, 2 training iterations).
const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    },
    "simulation": { "enabled": true, "num_scenarios": 1 }
}"#;

const PENALTIES_JSON: &str = r#"{
    "bus": {
        "deficit_segments": [
            { "depth_mw": 500.0, "cost": 1000.0 },
            { "depth_mw": null,  "cost": 5000.0 }
        ],
        "excess_cost": 100.0
    },
    "line": { "exchange_cost": 2.0 },
    "hydro": {
        "spillage_cost": 0.01,
        "turbined_cost": 0.05,
        "diversion_cost": 0.1,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 50000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
    },
    "non_controllable_source": { "curtailment_cost": 0.005 }
}"#;

/// 2-stage finite horizon. Anticipated thermal with `lead_stages=1` places
/// a decision at stage 0 that matures at stage 1.
const STAGES_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_scenarios": 2
        }
    ]
}"#;

/// Anticipated thermal id=2 has `lead_stages=1`, so `values_mw` must have
/// exactly one entry — the prior commitment before the study start.
const INITIAL_CONDITIONS_JSON: &str = r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "values_mw": [0.0] }
    ]
}"#;
const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

/// Two thermals: id=1 regular, id=2 anticipated with `lead_stages=1`.
/// IDs are ascending (declaration-order invariance rule: anticipated id > regular id).
const THERMALS_JSON: &str = r#"{
    "thermals": [
        {
            "id": 1,
            "name": "REGULAR",
            "bus_id": 1,
            "cost_per_mwh": 30.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        },
        {
            "id": 2,
            "name": "ANTICIPATED",
            "bus_id": 1,
            "cost_per_mwh": 25.0,
            "generation": { "min_mw": 0.0, "max_mw": 100.0 },
            "anticipated_config": { "lead_stages": 1 }
        }
    ]
}"#;

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

/// Reads all record batches from a Parquet file and concatenates them into
/// a single collection of column arrays, indexed by column name.
///
/// Returns `(stage_ids, thermal_ids, is_anticipated, anticipated_committed_mw,
/// anticipated_decision_mw)` extracted from every row.
struct ThermalRows {
    stage_ids: Vec<i32>,
    thermal_ids: Vec<i32>,
    is_anticipated: Vec<bool>,
    anticipated_committed_mw: Vec<Option<f64>>,
    anticipated_decision_mw: Vec<Option<f64>>,
}

fn read_thermals_parquet(path: &Path) -> ThermalRows {
    let file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("failed to open thermals parquet at {}: {e}", path.display()));

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("failed to build ParquetRecordBatchReaderBuilder");
    let reader = builder.build().expect("failed to build reader");

    let mut stage_ids: Vec<i32> = Vec::new();
    let mut thermal_ids: Vec<i32> = Vec::new();
    let mut is_anticipated: Vec<bool> = Vec::new();
    let mut anticipated_committed_mw: Vec<Option<f64>> = Vec::new();
    let mut anticipated_decision_mw: Vec<Option<f64>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.expect("failed to read record batch");
        let schema = batch.schema();

        let stage_col_idx = schema
            .index_of("stage_id")
            .expect("thermals schema must have stage_id column");
        let thermal_col_idx = schema
            .index_of("thermal_id")
            .expect("thermals schema must have thermal_id column");
        let is_ant_col_idx = schema
            .index_of("is_anticipated")
            .expect("thermals schema must have is_anticipated column");
        let committed_col_idx = schema
            .index_of("anticipated_committed_mw")
            .expect("thermals schema must have anticipated_committed_mw column");
        let decision_col_idx = schema
            .index_of("anticipated_decision_mw")
            .expect("thermals schema must have anticipated_decision_mw column");

        let stage_col = batch
            .column(stage_col_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("stage_id must be Int32Array");
        let thermal_col = batch
            .column(thermal_col_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("thermal_id must be Int32Array");
        let is_ant_col = batch
            .column(is_ant_col_idx)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("is_anticipated must be BooleanArray");
        let committed_col = batch
            .column(committed_col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("anticipated_committed_mw must be Float64Array");
        let decision_col = batch
            .column(decision_col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("anticipated_decision_mw must be Float64Array");

        for i in 0..batch.num_rows() {
            stage_ids.push(stage_col.value(i));
            thermal_ids.push(thermal_col.value(i));
            is_anticipated.push(is_ant_col.value(i));
            anticipated_committed_mw.push(if committed_col.is_null(i) {
                None
            } else {
                Some(committed_col.value(i))
            });
            anticipated_decision_mw.push(if decision_col.is_null(i) {
                None
            } else {
                Some(decision_col.value(i))
            });
        }
    }

    ThermalRows {
        stage_ids,
        thermal_ids,
        is_anticipated,
        anticipated_committed_mw,
        anticipated_decision_mw,
    }
}

/// End-to-end run with two thermals (id=1 regular, id=2 anticipated with `lead_stages=1`).
///
/// Asserts on the three columns populated by the anticipated thermal extraction:
/// - Regular thermal at every stage: `is_anticipated=false`, both optional columns null.
/// - Anticipated thermal at stage 0: `is_anticipated=true`, `anticipated_decision_mw` non-null
///   and `>= 0.0`, `anticipated_committed_mw` null (decision placed, not yet matured).
/// - Anticipated thermal at stage 1: `is_anticipated=true`, `anticipated_committed_mw`
///   non-null and `>= 0.0`.
#[test]
fn cli_run_populates_anticipated_thermal_columns() {
    // Sanity: IDs are ascending (declaration-order invariance).
    let regular_id: i32 = 1;
    let anticipated_id: i32 = 2;
    assert!(
        regular_id < anticipated_id,
        "regular_id ({regular_id}) must be less than anticipated_id ({anticipated_id})"
    );

    let tmp = TempDir::new().expect("create tempdir");
    let case = tmp.path().join("case");
    let output = tmp.path().join("output");
    fs::create_dir_all(&case).expect("create case dir");

    write_file(&case, "config.json", CONFIG_JSON);
    write_file(&case, "penalties.json", PENALTIES_JSON);
    write_file(&case, "stages.json", STAGES_JSON);
    write_file(&case, "initial_conditions.json", INITIAL_CONDITIONS_JSON);
    write_file(&case, "system/buses.json", BUSES_JSON);
    write_file(&case, "system/lines.json", LINES_JSON);
    write_file(&case, "system/hydros.json", HYDROS_JSON);
    write_file(&case, "system/thermals.json", THERMALS_JSON);

    cobre()
        .args([
            "run",
            case.to_str().expect("case path is valid UTF-8"),
            "--output",
            output.to_str().expect("output path is valid UTF-8"),
            "--threads",
            "1",
        ])
        .assert()
        .success();

    // Simulation writes one parquet per scenario. With num_scenarios=1, the file
    // is at simulation/thermals/scenario_id=0000/data.parquet.
    let parquet_path = output.join("simulation/thermals/scenario_id=0000/data.parquet");
    assert!(
        parquet_path.exists(),
        "simulation/thermals/scenario_id=0000/data.parquet must exist at {}",
        parquet_path.display()
    );

    let rows = read_thermals_parquet(&parquet_path);

    assert!(
        !rows.stage_ids.is_empty(),
        "thermals.parquet must contain at least one row"
    );

    // AC-3 and AC-7: Regular thermal — is_anticipated=false, both optional columns null.
    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != regular_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            !rows.is_anticipated[row_idx],
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             is_anticipated=false"
        );
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_decision_mw=null"
        );
        assert!(
            rows.anticipated_committed_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_committed_mw=null"
        );
    }

    // Verify we actually saw rows for the regular thermal.
    let regular_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == regular_id)
        .count();
    assert!(
        regular_row_count > 0,
        "no rows found for regular thermal id={regular_id} in thermals.parquet"
    );

    // AC-6: Anticipated thermal — is_anticipated=true for all rows.
    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != anticipated_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            rows.is_anticipated[row_idx],
            "row {row_idx}: anticipated thermal (id={anticipated_id}, stage={stage}) must have \
             is_anticipated=true"
        );
    }

    // Verify we actually saw rows for the anticipated thermal.
    let anticipated_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == anticipated_id)
        .count();
    assert!(
        anticipated_row_count > 0,
        "no rows found for anticipated thermal id={anticipated_id} in thermals.parquet"
    );

    // AC-4: Anticipated thermal at stage 0 — anticipated_decision_mw non-null and >= 0.0;
    //        anticipated_committed_mw non-null under always-active fishing (reads slot 0
    //        of the seeded ring buffer regardless of K vs stage_idx).
    let stage_0_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 0)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_0_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 0"
    );

    for &row_idx in &stage_0_ant_rows {
        let decision = rows.anticipated_decision_mw[row_idx];
        assert!(
            decision.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 must have non-null \
             anticipated_decision_mw"
        );
        let v = decision.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_decision_mw at stage 0 must be >= 0.0 and finite, \
             got {v}"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 under always-active fishing \
             must have non-null anticipated_committed_mw (reads slot 0)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 0 must be >= 0.0 and finite, got {c}"
        );
    }

    // AC-5: Anticipated thermal at stage 1 — anticipated_committed_mw non-null and >= 0.0
    //        (matured delivery: stage 1 >= K=1).
    let stage_1_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 1)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_1_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 1"
    );

    for &row_idx in &stage_1_ant_rows {
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 1 must have non-null \
             anticipated_committed_mw (matured delivery for K=1)"
        );
        let v = committed.unwrap();
        // F3-006: bounded magnitude — must lie in the thermal's [min, max]
        // generation envelope. ANTICIPATED has max_mw=100.0 in the fixture.
        assert!(
            v.is_finite() && (0.0..=100.0).contains(&v),
            "row {row_idx}: anticipated_committed_mw at stage 1 must be finite and in \
             [0.0, 100.0] (the plant's generation envelope), got {v}"
        );
    }

    // F3-006: ring-buffer transport invariant. The decision placed at stage 0
    // must equal the committed value matured at stage 1, bit-for-bit (single
    // anticipated plant, single scenario, single block — no aggregation).
    // Catches future regressions of the F1-001 class (simulation pipeline
    // failing to shift the anticipated ring buffer).
    assert_eq!(
        stage_0_ant_rows.len(),
        stage_1_ant_rows.len(),
        "stage-0 anticipated rows must pair 1:1 with stage-1 rows in a single-scenario, \
         single-block fixture"
    );
    for (&stage_0_idx, &stage_1_idx) in stage_0_ant_rows.iter().zip(&stage_1_ant_rows) {
        let decision = rows.anticipated_decision_mw[stage_0_idx]
            .expect("stage-0 decision must be Some (asserted above)");
        let committed = rows.anticipated_committed_mw[stage_1_idx]
            .expect("stage-1 committed must be Some (asserted above)");
        assert_eq!(
            decision.to_bits(),
            committed.to_bits(),
            "ring-buffer transport: stage-0 decision {decision} must equal stage-1 \
             committed {committed} bit-for-bit (the F1-001 simulation shift contract)"
        );
    }
}
