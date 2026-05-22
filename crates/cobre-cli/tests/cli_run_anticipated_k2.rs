//! Integration test: end-to-end run with K=2 anticipated thermals.
//!
//! Verifies that `simulation/thermals/` carries non-null, semantically
//! correct values in the three columns `is_anticipated`,
//! `anticipated_decision_mw`, and `anticipated_committed_mw` after a
//! full training + simulation run on a 3-stage fixture with one regular and
//! one anticipated thermal plant (`lead_stages=2`).
//!
//! Also verifies that `training/dictionaries/state_dictionary.json` is
//! written with exactly `K_max * n_anticipated = 2 * 1 = 2` entries of
//! type `"anticipated_state"` referencing `entity_id=2` at `slot_index` 0 and 1.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

/// 3-stage finite horizon. Anticipated thermal with `lead_stages=2` places
/// a decision at stage 0 that matures at stage 2; at stage 1 (horizon boundary:
/// t + `K_i` = 1 + 2 = 3 = `n_stages`) the decision is still active.
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
        },
        {
            "id": 2,
            "start_date": "2024-03-01",
            "end_date": "2024-04-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 2
        }
    ]
}"#;

/// Anticipated thermal id=2 has `lead_stages=2`, so `values_mw` must have
/// exactly two entries — the prior commitments before the study start.
const INITIAL_CONDITIONS_JSON: &str = r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "values_mw": [0.0, 0.0] }
    ]
}"#;

const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

/// Two thermals: id=1 regular, id=2 anticipated with `lead_stages=2`.
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
            "anticipated_config": { "lead_stages": 2 }
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

#[test]
#[allow(clippy::too_many_lines)]
fn cli_run_k2_populates_anticipated_columns_and_state_dictionary() {
    let regular_id: i32 = 1;
    let anticipated_id: i32 = 2;
    assert!(regular_id < anticipated_id);

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

    // Regular thermal — is_anticipated=false, both optional columns null.
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

    // Anticipated thermal — is_anticipated=true for all rows.
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

    // Anticipated thermal at stage 0 — anticipated_decision_mw non-null and >= 0.0,
    // anticipated_committed_mw null (decision placed, K=2 not yet reached: 2 <= 0 is false).
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
        assert!(
            rows.anticipated_committed_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 0 (K=2, delivery requires 2 <= 0) must \
             have anticipated_committed_mw=null"
        );
    }

    // Anticipated thermal at stage 1 — anticipated_decision_mw non-null and >= 0.0 (horizon-
    // boundary active: t + K_i = 1 + 2 = 3 = n_stages), anticipated_committed_mw null
    // (delivery requires K_i <= stage_index = 2 <= 1 is false).
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
        let decision = rows.anticipated_decision_mw[row_idx];
        assert!(
            decision.is_some(),
            "row {row_idx}: anticipated thermal at stage 1 must have non-null \
             anticipated_decision_mw (horizon-boundary active: t + K_i = 3 = n_stages)"
        );
        let v = decision.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_decision_mw at stage 1 must be >= 0.0 and finite, \
             got {v}"
        );
        assert!(
            rows.anticipated_committed_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 1 (delivery requires K_i=2 <= \
             stage_index=1, which is false) must have anticipated_committed_mw=null"
        );
    }

    // Anticipated thermal at stage 2 — anticipated_decision_mw null (t + K_i = 2 + 2 = 4 >
    // n_stages=3), anticipated_committed_mw non-null and >= 0.0 (delivery: K_i=2 <= 2).
    let stage_2_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 2)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_2_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 2"
    );

    for &row_idx in &stage_2_ant_rows {
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 2 must have anticipated_decision_mw=null \
             (t + K_i = 4 > n_stages=3)"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 2 must have non-null \
             anticipated_committed_mw (delivery: K_i=2 <= stage_index=2)"
        );
        let v = committed.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 2 must be >= 0.0 and finite, \
             got {v}"
        );
    }

    // Assert training/dictionaries/state_dictionary.json exists and contains exactly
    // K_max * n_anticipated = 2 * 1 = 2 entries with "type": "anticipated_state",
    // both referencing entity_id=2 with slot_index in {0, 1} in slot-major order.
    let state_dict_path = output.join("training/dictionaries/state_dictionary.json");
    assert!(
        state_dict_path.exists(),
        "training/dictionaries/state_dictionary.json must exist at {}",
        state_dict_path.display()
    );

    let state_dict_str = fs::read_to_string(&state_dict_path).unwrap_or_else(|e| {
        panic!(
            "failed to read state_dictionary.json at {}: {e}",
            state_dict_path.display()
        )
    });

    let state_dict: serde_json::Value = serde_json::from_str(&state_dict_str).unwrap_or_else(|e| {
        panic!(
            "failed to parse state_dictionary.json at {}: {e}",
            state_dict_path.display()
        )
    });

    let state_vars = state_dict
        .get("state_variables")
        .and_then(serde_json::Value::as_array)
        .expect("state_dictionary.json must have a top-level \"state_variables\" array");

    let anticipated_entries: Vec<&serde_json::Value> = state_vars
        .iter()
        .filter(|entry| entry["type"].as_str() == Some("anticipated_state"))
        .collect();

    assert_eq!(
        anticipated_entries.len(),
        2,
        "state_dictionary.json must contain exactly 2 \"anticipated_state\" entries \
         (K_max=2, n_anticipated=1), found {}",
        anticipated_entries.len()
    );

    for entry in &anticipated_entries {
        assert_eq!(
            entry["entity_id"].as_i64(),
            Some(i64::from(anticipated_id)),
            "each anticipated_state entry must reference entity_id={anticipated_id}, \
             got {:?}",
            entry["entity_id"]
        );
    }

    // Verify slot-major order: slot_index values must be {0, 1}.
    let slot_indices: std::collections::HashSet<u64> = anticipated_entries
        .iter()
        .filter_map(|entry| entry["slot_index"].as_u64())
        .collect();
    assert_eq!(
        slot_indices,
        [0u64, 1u64]
            .into_iter()
            .collect::<std::collections::HashSet<u64>>(),
        "anticipated_state entries must cover slot_index values {{0, 1}}, found {slot_indices:?}"
    );
}
