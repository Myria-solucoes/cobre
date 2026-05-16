//! Integration test for the energy-conversion soft consistency warning.
//!
//! Bootstraps the embedded `1dtoy` template, mutates `system/hydros.json` to
//! supply a `specific_productivity_mw_per_m3s_per_m` that diverges from the
//! implied value `ρ_eq_stored / h_eq`, writes a minimal
//! `system/hydro_geometry.parquet`, runs `cobre run`, and asserts the
//! `[energy-conversion] WARN:` prefix appears on stderr.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use assert_cmd::prelude::*;
use parquet::arrow::ArrowWriter;
use predicates::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Write a single-row VHA parquet at `path` for hydro `hydro_id` with two
/// volume points (necessary for `ForebayTable::new` to construct) producing
/// a flat `height_m` of 100.0 across the operating range.
fn write_flat_geometry(path: &Path, hydro_id: i32) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("volume_hm3", DataType::Float64, false),
        Field::new("height_m", DataType::Float64, false),
        Field::new("area_km2", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![hydro_id, hydro_id])),
            Arc::new(Float64Array::from(vec![0.0_f64, 1_000.0_f64])),
            Arc::new(Float64Array::from(vec![100.0_f64, 100.0_f64])),
            Arc::new(Float64Array::from(vec![1.0_f64, 1.0_f64])),
        ],
    )
    .unwrap();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn run_emits_energy_conversion_warning_on_divergent_specific_productivity() {
    let dir = TempDir::new().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    cobre()
        .args(["init", "--template", "1dtoy", dir_str])
        .assert()
        .success();

    let hydros_path = dir.path().join("system/hydros.json");
    let hydros: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&hydros_path).unwrap()).unwrap();
    let mut hydros = hydros;
    let hydro_array = hydros["hydros"].as_array_mut().unwrap();
    assert_eq!(hydro_array.len(), 1, "1dtoy template should have one hydro");
    let hydro_id = hydro_array[0]["id"].as_i64().unwrap();
    let hydro_id_i32: i32 = hydro_id.try_into().unwrap();
    // Implied ρ_esp = productivity_mw_per_m3s / h_fore = 1.0 / 100.0 = 0.01.
    // Supplied = 0.0050 → relative divergence 50%, well above the 5% tolerance.
    hydro_array[0].as_object_mut().unwrap().insert(
        "specific_productivity_mw_per_m3s_per_m".to_string(),
        serde_json::json!(0.0050),
    );
    fs::write(&hydros_path, serde_json::to_string_pretty(&hydros).unwrap()).unwrap();

    write_flat_geometry(
        &dir.path().join("system/hydro_geometry.parquet"),
        hydro_id_i32,
    );

    let out = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            dir_str,
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("[energy-conversion] WARN:"));
}
