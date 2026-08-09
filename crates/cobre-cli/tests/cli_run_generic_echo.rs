//! Integration test: `cobre run` emits `generic_constraints/resolved_echo.parquet`
//! for a study with generic constraints, and emits nothing for one without.
//!
//! The echo is a training-side sidecar; the written file must carry the 13-column
//! echo schema and the resolved interval a reader can compare against the deck.

#![allow(clippy::expect_used, clippy::panic, clippy::float_cmp)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::{Array, Float64Array, StringArray};
use assert_cmd::prelude::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn case_dir(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR");
    root.join("examples/deterministic").join(name)
}

fn run_case(case: &Path, out: &Path) {
    cobre()
        .args([
            "run",
            case.to_str().expect("case path is valid UTF-8"),
            "--output",
            out.to_str().expect("temp path is valid UTF-8"),
            "--quiet",
        ])
        .assert()
        .success();
}

#[test]
fn cli_writes_generic_constraint_echo_for_d13() {
    let case = case_dir("d13-generic-constraint");
    assert!(
        case.join("config.json").is_file(),
        "d13 fixture must exist at {}",
        case.display()
    );

    let out = TempDir::new().expect("create temp output dir");
    run_case(&case, out.path());

    let echo_path = out
        .path()
        .join("generic_constraints")
        .join("resolved_echo.parquet");
    assert!(
        echo_path.is_file(),
        "resolved_echo.parquet must exist at {} (d13 declares a generic constraint)",
        echo_path.display()
    );

    let file = fs::File::open(&echo_path).expect("open resolved_echo.parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("build parquet reader");
    assert_eq!(
        builder.schema().fields().len(),
        13,
        "echo must carry the 13-column schema"
    );
    let reader = builder.build().expect("build reader");

    let mut saw_cap_at_ten = false;
    let mut total_rows = 0usize;
    for batch_result in reader {
        let batch = batch_result.expect("read record batch");
        total_rows += batch.num_rows();
        let schema = batch.schema();

        let shape_col = batch
            .column(
                schema
                    .index_of("derived_shape")
                    .expect("derived_shape column"),
            )
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("derived_shape must be Utf8");
        let upper_col = batch
            .column(schema.index_of("bound_upper").expect("bound_upper column"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("bound_upper must be Float64");

        for i in 0..batch.num_rows() {
            if shape_col.value(i) == "cap" && !upper_col.is_null(i) && upper_col.value(i) == 10.0 {
                saw_cap_at_ten = true;
            }
        }
    }

    assert!(total_rows > 0, "d13 echo must contain at least one row");
    assert!(
        saw_cap_at_ten,
        "d13's `thermal_generation(0) <= 10` must echo a `cap` row with bound_upper = 10.0"
    );
}

#[test]
fn cli_writes_no_echo_without_generic_constraints() {
    let case = case_dir("d01-thermal-dispatch");
    assert!(
        case.join("config.json").is_file(),
        "d01 fixture must exist at {}",
        case.display()
    );

    let out = TempDir::new().expect("create temp output dir");
    run_case(&case, out.path());

    assert!(
        !out.path().join("generic_constraints").exists(),
        "a study with no generic constraints must write no generic_constraints/ directory"
    );
}
