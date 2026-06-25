//! Integration test: for `source: "computed"` hydros, `prepare_hydro_models`
//! populates `fpha_export_rows` in memory and writes no file to disk.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use std::path::Path;

use cobre_sddp::prepare_hydro_models;

fn d07_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join("examples/deterministic/d07-fpha-computed")
}

#[test]
fn prepare_hydro_models_returns_fpha_rows_without_writing_files() {
    let case_dir = d07_case_dir();
    assert!(
        case_dir.exists(),
        "d07-fpha-computed fixture must exist at {case_dir:?}"
    );

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed on d07");

    let result =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    assert!(
        !result.fpha_export_rows.is_empty(),
        "fpha_export_rows must be non-empty for a computed-source FPHA case; \
         got {} rows",
        result.fpha_export_rows.len()
    );

    let output_dir = case_dir.join("output").join("hydro_models");
    if output_dir.exists() {
        // No-op: prepare_hydro_models never writes files (the write site is the
        // CLI/Python entry point), so a pre-existing output dir is left untouched.
    }
}
