//! Integration test: end-to-end run with K=2 anticipated thermals on a 3-stage
//! fixture (one regular, one anticipated thermal with `lead_stages=2`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::Path;
use std::process::Command;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array};
use assert_cmd::prelude::*;
use cobre_io::{EntitySlot, deserialize_stage_cuts};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

/// Raw `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

const CONFIG_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    },
    "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 1 } }
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
            "num_openings": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 2
        },
        {
            "id": 2,
            "start_date": "2024-03-01",
            "end_date": "2024-04-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 2
        }
    ]
}"#;

/// Anticipated thermal id=2 has `lead_stages=2`, so its commitment windows
/// must tile exactly two leading delivery stages — the prior commitments
/// before the study start, covering stage 0's `[2024-01-01, 2024-02-01)` and
/// stage 1's `[2024-02-01, 2024-03-01)`. Coverage (`StageCalendar::coverage`)
/// is computed on each stage's own real calendar span, not its declared
/// block-hours, so the windows mirror `STAGES_JSON`'s `start_date`/`end_date`
/// exactly (stage 1's real span is 29 days despite its 672 declared hours).
const INITIAL_CONDITIONS_JSON: &str = r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 },
        { "thermal_id": 2, "start_date": "2024-02-01", "end_date": "2024-03-01", "value_mw": 0.0 }
    ]
}"#;

const BUSES_JSON: &str =
    r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

/// Two thermals: id=1 regular, id=2 anticipated with `lead_stages=2`.
/// IDs are ascending (declaration-order invariance rule: anticipated id > regular id).
const THERMALS_JSON: &str = r#"{
    "thermals": [
        {
            "id": 1,
            "name": "REGULAR",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 30.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        },
        {
            "id": 2,
            "name": "ANTICIPATED",
            "operational_start_date": "2024-01-01",
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
fn cli_run_k2_populates_anticipated_columns_and_manifest() {
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

    // One parquet per scenario; num_scenarios=1 gives the single scenario_id=0000 file.
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

    let regular_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == regular_id)
        .count();
    assert!(
        regular_row_count > 0,
        "no rows found for regular thermal id={regular_id} in thermals.parquet"
    );

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

    let anticipated_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == anticipated_id)
        .count();
    assert!(
        anticipated_row_count > 0,
        "no rows found for anticipated thermal id={anticipated_id} in thermals.parquet"
    );

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
             must have non-null anticipated_committed_mw (reads slot 0 of the ring buffer)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 0 must be >= 0.0 and finite, got {c}"
        );
    }

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
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 1 must have \
             anticipated_decision_mw=null (horizon-boundary inactive: \
             t + K_i = 3 >= n_stages = 3 under the strict predicate)"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 1 under always-active fishing \
             must have non-null anticipated_committed_mw (reads slot 0 of the ring buffer \
             regardless of K_i vs stage_index)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 1 must be >= 0.0 and finite, got {c}"
        );
    }

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

    // The per-slot state identity now lives in the embedded entity manifest of
    // each policy cut file, not a separate dictionary sidecar. Read stage 0's
    // manifest and assert it carries the two anticipated ring slots for plant 2.
    let cuts_path = output.join("policy/cuts/000.bin");
    assert!(
        cuts_path.exists(),
        "policy/cuts/000.bin must exist at {}",
        cuts_path.display()
    );

    let cuts_bytes = fs::read(&cuts_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cuts_path.display()));
    let stage_cuts = deserialize_stage_cuts(&cuts_bytes)
        .unwrap_or_else(|e| panic!("failed to decode {}: {e:?}", cuts_path.display()));

    // No hydros in this case, so every manifest slot is an anticipated ring slot.
    let anticipated_slots: Vec<&EntitySlot> = stage_cuts
        .entity_manifest
        .iter()
        .filter(|slot| slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE)
        .collect();

    assert_eq!(
        anticipated_slots.len(),
        2,
        "stage 0 manifest must contain exactly 2 anticipated ring slots \
         (K_max=2, n_anticipated=1), found {}",
        anticipated_slots.len()
    );

    for slot in &anticipated_slots {
        assert_eq!(
            slot.entity_id, anticipated_id,
            "each anticipated slot must reference entity_id={anticipated_id}, got {}",
            slot.entity_id
        );
    }

    let ring_subindices: std::collections::HashSet<u32> =
        anticipated_slots.iter().map(|slot| slot.subindex).collect();
    assert_eq!(
        ring_subindices,
        [0u32, 1u32]
            .into_iter()
            .collect::<std::collections::HashSet<u32>>(),
        "anticipated slots must cover ring subindex values {{0, 1}}, found {ring_subindices:?}"
    );

    // `was_active` encodes the active/dormant distinction the sidecar's
    // `lead_stages` split previously served: plant 2 has no commissioning
    // window and K_i = K_max = 2, so both ring slots are active at stage 0.
    for slot in &anticipated_slots {
        assert!(
            slot.was_active,
            "anticipated slot (subindex {}) must be active at stage 0",
            slot.subindex
        );
    }
}
