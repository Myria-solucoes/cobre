//! End-to-end integration tests for `cobre_io::load_case`.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use cobre_core::AnticipatedConfig;
use cobre_core::EntityId;
use cobre_io::{deserialize_system, load_case, serialize_system};
use tempfile::TempDir;

// ── test_minimal_valid_case ────────────────────────────────────────────────────

#[test]
fn test_minimal_valid_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for minimal case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        1,
        "minimal case should have exactly 1 bus"
    );
    assert_eq!(system.n_hydros(), 0, "minimal case should have 0 hydros");
    assert_eq!(
        system.n_thermals(),
        0,
        "minimal case should have 0 thermals"
    );
    assert_eq!(system.n_lines(), 0, "minimal case should have 0 lines");
    assert_eq!(
        system.n_stages(),
        1,
        "minimal case should have exactly 1 stage"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be found by O(1) lookup"
    );
}

// ── test_multi_entity_case ────────────────────────────────────────────────────

#[test]
fn test_multi_entity_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for multi-entity case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        2,
        "multi-entity case should have exactly 2 buses"
    );
    assert_eq!(
        system.n_hydros(),
        1,
        "multi-entity case should have exactly 1 hydro"
    );
    assert_eq!(
        system.n_thermals(),
        1,
        "multi-entity case should have exactly 1 thermal"
    );
    assert_eq!(
        system.n_lines(),
        1,
        "multi-entity case should have exactly 1 line"
    );
    assert_eq!(
        system.n_stages(),
        2,
        "multi-entity case should have exactly 2 stages"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be accessible via O(1) lookup"
    );
    assert!(
        system.bus(EntityId(2)).is_some(),
        "bus with id=2 should be accessible via O(1) lookup"
    );
}

// ── test_missing_required_file ────────────────────────────────────────────────

#[test]
fn test_missing_required_file() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    std::fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("buses"),
                "error display should mention 'buses', got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when buses.json is missing, got Ok"),
    }
}

// ── test_malformed_json ───────────────────────────────────────────────────────

#[test]
fn test_malformed_json() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    helpers::write_file(
        dir.path(),
        "system/hydros.json",
        "{ this is not valid json }",
    );

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            // Variant is implementation-defined (ParseError or ConstraintError wrapping the parse); assert only Err.
            let display = err.to_string();
            assert!(
                !display.is_empty(),
                "error display should be non-empty for malformed JSON"
            );
        }
        Ok(_) => panic!("expected Err for malformed hydros.json, got Ok"),
    }
}

// ── test_referential_integrity_violation ─────────────────────────────────────

#[test]
fn test_referential_integrity_violation() {
    let dir = TempDir::new().unwrap();
    helpers::make_referential_violation_case(&dir);

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("999") || display.contains("bus") || display.contains("Bus"),
                "error display should mention the invalid bus reference (999), got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when hydro references non-existent bus_id=999, got Ok"),
    }
}

// ── test_inflow_history_wired_into_system ─────────────────────────────────────

/// The seasonal-stats + AR-coefficient parquets bypass the estimation path, but
/// `stages.json` still needs `season_definitions`: the loading pipeline derives
/// every order-bearing model's `residual_std_ratio` from the periodic-ACF
/// closure (`populate_derived_residual_ratios`), which requires a resolvable
/// season for any AR order > 0, regardless of whether estimation ran.
#[test]
fn test_inflow_history_wired_into_system() {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // `make_multi_entity_case`'s default `stages.json` carries no
    // `season_definitions`; this case's `inflow_ar_coefficients.parquet` below
    // has real AR order > 0, so it needs a resolvable season per stage.
    std::fs::write(
        dir.path().join("stages.json"),
        r#"{
    "season_definitions": {
        "cycle_type": "monthly",
        "seasons": [
            { "id": 0, "month_start": 1, "label": "January" },
            { "id": 1, "month_start": 2, "label": "February" },
            { "id": 2, "month_start": 3, "label": "March" },
            { "id": 3, "month_start": 4, "label": "April" },
            { "id": 4, "month_start": 5, "label": "May" },
            { "id": 5, "month_start": 6, "label": "June" },
            { "id": 6, "month_start": 7, "label": "July" },
            { "id": 7, "month_start": 8, "label": "August" },
            { "id": 8, "month_start": 9, "label": "September" },
            { "id": 9, "month_start": 10, "label": "October" },
            { "id": 10, "month_start": 11, "label": "November" },
            { "id": 11, "month_start": 12, "label": "December" }
        ]
    },
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 }
        ]
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 10
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_scenarios": 10
        }
    ]
}"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        r#"{ "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 2, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.85 }] },
            { "hydro_id": 3, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.8 }] }
        ] }"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 },
              "unit_groups": [
                { "id": 0, "name": "H1", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 200.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0 }
              ] },
            { "id": 2, "name": "H2", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 },
              "unit_groups": [
                { "id": 0, "name": "H2", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 100.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0 }
              ] },
            { "id": 3, "name": "H3", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 },
              "unit_groups": [
                { "id": 0, "name": "H3", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 80.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0 }
              ] }
        ] }"#,
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();

    {
        let stats_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean_m3s", DataType::Float64, false),
            Field::new("std_m3s", DataType::Float64, false),
        ]));
        let stats_batch = RecordBatch::try_new(
            Arc::clone(&stats_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Float64Array::from(vec![
                    150.0, 120.0, 80.0, 70.0, 50.0, 45.0,
                ])),
                Arc::new(Float64Array::from(vec![20.0, 15.0, 10.0, 8.0, 6.0, 5.0])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_seasonal_stats.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, stats_batch.schema(), None).unwrap();
        writer.write(&stats_batch).unwrap();
        writer.close().unwrap();
    }

    // Both stats AND coefficients must be present to skip estimation.
    {
        let ar_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("residual_std_ratio", DataType::Float64, false),
        ]));
        let ar_batch = RecordBatch::try_new(
            Arc::clone(&ar_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1, 1])),
                Arc::new(Float64Array::from(vec![0.3, 0.25, 0.4, 0.35, 0.2, 0.15])),
                Arc::new(Float64Array::from(vec![0.95, 0.92, 0.90, 0.88, 0.93, 0.91])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_ar_coefficients.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, ar_batch.schema(), None).unwrap();
        writer.write(&ar_batch).unwrap();
        writer.close().unwrap();
    }

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(864);
    let mut start_dates: Vec<i32> = Vec::with_capacity(864);
    let mut end_dates: Vec<i32> = Vec::with_capacity(864);
    let mut values: Vec<f64> = Vec::with_capacity(864);

    for hid in 1_i32..=3 {
        for year in 2000_i32..=2023 {
            for month in 1_u32..=12 {
                let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
                let end = start.checked_add_months(chrono::Months::new(1)).unwrap();
                hydro_ids.push(hid);
                start_dates.push(i32::try_from((start - epoch).num_days()).unwrap());
                end_dates.push(i32::try_from((end - epoch).num_days()).unwrap());
                values.push(f64::from(hid) * 100.0 + f64::from(month));
            }
        }
    }

    let history_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let history_batch = RecordBatch::try_new(
        Arc::clone(&history_schema),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(dir.path().join("scenarios/inflow_history.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, history_batch.schema(), None).unwrap();
    writer.write(&history_batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for inflow_history case: {e}"));

    assert_eq!(
        system.inflow_history().len(),
        864,
        "system.inflow_history() must have 864 rows (3 hydros × 24 years × 12 months)"
    );
    for row in system.inflow_history() {
        assert!(
            row.value_m3s.is_finite(),
            "every inflow_history row must have a finite value_m3s"
        );
    }
}

// ── test_external_scenarios_wired_into_system ─────────────────────────────────

#[test]
fn test_external_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    std::fs::write(
        dir.path().join("system/hydro_production_models.json"),
        r#"{ "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 2, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.85 }] },
            { "hydro_id": 3, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null,
                "model": "constant_productivity", "productivity_mw_per_m3s": 0.8 }] }
        ] }"#,
    )
    .unwrap();

    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 },
              "unit_groups": [
                { "id": 0, "name": "H1", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 200.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0 }
              ] },
            { "id": 2, "name": "H2", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 },
              "unit_groups": [
                { "id": 0, "name": "H2", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 100.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0 }
              ] },
            { "id": 3, "name": "H3", "operational_start_date": "2024-01-01", "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity",
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 },
              "unit_groups": [
                { "id": 0, "name": "H3", "bus_id": 1,
                  "min_generation_mw": 0.0, "max_generation_mw": 80.0,
                  "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0 }
              ] }
        ] }"#,
    )
    .unwrap();

    let mut stage_ids: Vec<i32> = Vec::with_capacity(30);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(30);
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(30);
    let mut values: Vec<f64> = Vec::with_capacity(30);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..5 {
            for hid in 1_i32..=3 {
                stage_ids.push(stage_id);
                scenario_ids.push(scenario_id);
                hydro_ids.push(hid);
                values.push(
                    f64::from(stage_id) * 1000.0 + f64::from(scenario_id) * 10.0 + f64::from(hid),
                );
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file = std::fs::File::create(
        dir.path()
            .join("scenarios/external_inflow_scenarios.parquet"),
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_inflow_scenarios case: {e}"));

    assert_eq!(
        system.external_scenarios().len(),
        30,
        "system.external_scenarios() must have 30 rows (2 stages × 5 scenarios × 3 hydros)"
    );
    for row in system.external_scenarios() {
        assert!(
            row.value_m3s.is_finite(),
            "every external_scenarios row must have a finite value_m3s"
        );
    }
}

// ── test_inflow_history_absent_returns_empty ──────────────────────────────────

#[test]
fn test_inflow_history_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.inflow_history().is_empty(),
        "system.inflow_history() must be empty when inflow_history.parquet is absent"
    );
}

// ── test_external_scenarios_absent_returns_empty ──────────────────────────────

#[test]
fn test_external_scenarios_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.external_scenarios().is_empty(),
        "system.external_scenarios() must be empty when external_inflow_scenarios.parquet is absent"
    );
}

// ── test_external_load_scenarios_wired_into_system ───────────────────────────

#[test]
fn test_external_load_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let mut stage_ids: Vec<i32> = Vec::with_capacity(6);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(6);
    let mut bus_ids: Vec<i32> = Vec::with_capacity(6);
    let mut values: Vec<f64> = Vec::with_capacity(6);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..3 {
            stage_ids.push(stage_id);
            scenario_ids.push(scenario_id);
            bus_ids.push(1);
            values.push(100.0 + f64::from(stage_id) * 50.0 + f64::from(scenario_id) * 10.0);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("bus_id", DataType::Int32, false),
        Field::new("value_mw", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(bus_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file = std::fs::File::create(dir.path().join("scenarios/external_load_scenarios.parquet"))
        .unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_load_scenarios case: {e}"));

    assert_eq!(
        system.external_load_scenarios().len(),
        6,
        "system.external_load_scenarios() must have 6 rows (2 stages × 3 scenarios × 1 bus)"
    );
    for row in system.external_load_scenarios() {
        assert!(
            row.value_mw.is_finite(),
            "every external_load_scenarios row must have a finite value_mw"
        );
    }
}

// ── test_external_ncs_scenarios_wired_into_system ─────────────────────────────

#[test]
fn test_external_ncs_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // The referential validator rejects any ncs_id absent from the NCS registry, so register 1 and 2.
    helpers::write_file(
        dir.path(),
        "system/non_controllable_sources.json",
        r#"{
    "non_controllable_sources": [
        { "id": 1, "name": "NCS_A", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 50.0 },
        { "id": 2, "name": "NCS_B", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 30.0 }
    ]
}"#,
    );

    let mut stage_ids: Vec<i32> = Vec::with_capacity(16);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(16);
    let mut ncs_ids: Vec<i32> = Vec::with_capacity(16);
    let mut values: Vec<f64> = Vec::with_capacity(16);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..4 {
            for ncs_id in 1_i32..=2 {
                stage_ids.push(stage_id);
                scenario_ids.push(scenario_id);
                ncs_ids.push(ncs_id);
                // availability factor in (0, 1]
                values.push(0.5 + 0.1 * f64::from(scenario_id) + 0.05 * f64::from(ncs_id - 1));
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("ncs_id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(ncs_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file =
        std::fs::File::create(dir.path().join("scenarios/external_ncs_scenarios.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_ncs_scenarios case: {e}"));

    assert_eq!(
        system.external_ncs_scenarios().len(),
        16,
        "system.external_ncs_scenarios() must have 16 rows (2 stages × 4 scenarios × 2 NCS)"
    );
    for row in system.external_ncs_scenarios() {
        assert!(
            row.value.is_finite(),
            "every external_ncs_scenarios row must have a finite value"
        );
    }
}

// ── test_lead_time_single_decider_on_disk_load ────────────────────────────────

/// Pins the on-disk `LeadTime` load path (`pipeline.rs`'s `k_max` computation
/// for `resolve_bounds`): a single-decider `LeadTime(744.0)` thermal on a
/// uniform 3×744h calendar must load via `load_case` with no panic, and the
/// anticipated plant's resolved bounds must be correct at every study stage
/// (all of which are legitimate delivery-stage indices for the ring),
/// demonstrating a correct study, not merely a non-panicking one.
#[test]
fn test_lead_time_single_decider_on_disk_load() {
    // SystemBuilder sorts thermals by (operational_start_date, id); both
    // declared thermals share the same date, so ascending id puts T_ANT
    // (id=2) at index 0.
    const THERMAL_IDX_ANT: usize = 0;

    let dir = TempDir::new().unwrap();
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "stages.json",
        r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.0,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 },
            { "source_id": 1, "target_id": 2, "probability": 1.0 }
        ]
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_scenarios": 1
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_scenarios": 1
        },
        {
            "id": 2,
            "start_date": "2024-03-01",
            "end_date": "2024-04-01",
            "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
            "num_scenarios": 1
        }
    ]
}"#,
    );
    helpers::write_file(
        root,
        "initial_conditions.json",
        r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "values_mw": [0.0] }
    ]
}"#,
    );
    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(
        root,
        "system/thermals.json",
        r#"{
    "thermals": [
        {
            "id": 2,
            "name": "T_ANT",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 10.0,
            "generation": { "min_mw": 0.0, "max_mw": 100.0 },
            "anticipated_config": { "lead_time_hours": 744.0 }
        },
        {
            "id": 3,
            "name": "T_BACKUP",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 100.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        }
    ]
}"#,
    );

    let system = load_case(root).unwrap_or_else(|e| {
        panic!("load_case must succeed for a valid single-decider LeadTime case, got: {e}")
    });

    assert_eq!(system.n_stages(), 3, "case declares 3 study stages");
    assert_eq!(system.n_thermals(), 2, "case declares 2 thermals");

    let ant = &system.thermals()[THERMAL_IDX_ANT];
    assert_eq!(
        ant.anticipated_config,
        Some(AnticipatedConfig::LeadTime(744.0)),
        "loaded thermal must carry the LeadTime(744.0) config"
    );

    // Every study stage is a legitimate delivery-stage index for the
    // delivery-anchored ring: the resolved bounds must be correct there, not
    // merely absent a panic. The k_max=0 contribution this LeadTime plant
    // makes to the padded thermal-bounds axis pads no region a real study
    // ever reads.
    for stage in 0..system.n_stages() {
        let b = system.bounds().thermal_bounds(THERMAL_IDX_ANT, stage);
        assert!(
            (b.min_generation_mw - 0.0).abs() < f64::EPSILON,
            "stage {stage}: min_generation_mw mismatch, got {}",
            b.min_generation_mw
        );
        assert!(
            (b.max_generation_mw - 100.0).abs() < f64::EPSILON,
            "stage {stage}: max_generation_mw mismatch, got {}",
            b.max_generation_mw
        );
        assert!(
            (b.cost_per_mwh - 10.0).abs() < f64::EPSILON,
            "stage {stage}: cost_per_mwh mismatch, got {}",
            b.cost_per_mwh
        );
    }
}

// ── test_postcard_round_trip ──────────────────────────────────────────────────

#[test]
fn test_postcard_round_trip() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let original = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case should succeed for minimal case, got: {e}"));

    let bytes = serialize_system(&original)
        .unwrap_or_else(|e| panic!("serialize_system should succeed, got: {e}"));

    assert!(!bytes.is_empty(), "serialized bytes should be non-empty");

    let deserialized = deserialize_system(&bytes)
        .unwrap_or_else(|e| panic!("deserialize_system should succeed, got: {e}"));

    assert_eq!(
        deserialized.n_buses(),
        original.n_buses(),
        "bus count must match after postcard round-trip"
    );
    assert!(
        deserialized.bus(EntityId(1)).is_some(),
        "O(1) bus lookup must work after index rebuild on deserialized System"
    );

    assert_eq!(
        deserialized.n_hydros(),
        original.n_hydros(),
        "hydro count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_thermals(),
        original.n_thermals(),
        "thermal count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_lines(),
        original.n_lines(),
        "line count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_stages(),
        original.n_stages(),
        "stage count must match after round-trip"
    );
}
