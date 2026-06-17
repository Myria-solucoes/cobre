//! Integration test: the CLI writes `hydro_models/evaporation_models.parquet`
//! whose coefficient rows equal the resolved evaporation models.
//!
//! Runs the `cobre` binary on the D08 evaporation fixture, reads the written
//! `evaporation_models.parquet`, and compares each row's `(intercept_m3s,
//! volume_slope_m3s_per_hm3, reference_volume_hm3, stage_id, source)` against
//! the rows the in-process resolver
//! produces via `cobre_sddp::build_evaporation_model_rows`. Because the CLI and
//! Python write sites both serialize the identical builder output, asserting the
//! written file matches the builder output establishes CLI⇄Python file parity
//! without launching a Python interpreter (the builder is the single source both
//! call sites consume).

#![allow(clippy::expect_used, clippy::panic, clippy::float_cmp)]

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Absolute path to the D08 evaporation fixture, resolved relative to this
/// crate's manifest directory (two levels below the repo root).
fn d08_case_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR");
    root.join("examples/deterministic/d08-evaporation")
}

#[test]
fn cli_writes_evaporation_models_matching_resolver() {
    let case = d08_case_dir();
    assert!(
        case.join("config.json").is_file(),
        "D08 fixture must exist at {}",
        case.display()
    );

    // Write outputs to a temp dir so the committed `output/` tree is untouched.
    let out = TempDir::new().expect("create temp output dir");

    cobre()
        .args([
            "run",
            case.to_str().expect("D08 path is valid UTF-8"),
            "--output",
            out.path().to_str().expect("temp path is valid UTF-8"),
            "--quiet",
        ])
        .assert()
        .success();

    // The CLI writes the evaporation sidecar beside the FPHA export.
    let evaporation_path = out
        .path()
        .join("hydro_models")
        .join("evaporation_models.parquet");
    assert!(
        evaporation_path.is_file(),
        "evaporation_models.parquet must exist at {} (D08 models evaporation)",
        evaporation_path.display()
    );

    // Parse the written file. The reader returns rows sorted by
    // `(hydro_id, stage_id)`, which matches the builder's canonical order.
    let written =
        cobre_io::parse_evaporation_models(&evaporation_path).expect("parse evaporation_models");
    assert!(
        !written.is_empty(),
        "D08 must produce at least one evaporation row"
    );

    // Re-resolve the same models in-process and build the expected rows. This is
    // the exact source both the CLI and Python write sites serialize, so a
    // row-for-row match proves the persisted file equals the resolver output.
    let system = cobre_io::load_case(&case).expect("load D08 system");
    let result = cobre_sddp::prepare_hydro_models(&system, &case, false)
        .expect("prepare hydro models for D08");
    let expected = cobre_sddp::build_evaporation_model_rows(&result, &system);

    assert_eq!(
        written.len(),
        expected.len(),
        "written row count must equal the resolver row count"
    );

    // D08 has two study stages, so the single evaporation hydro yields two
    // per-stage rows (stage_id 0 and 1) — exercising the multi-stage path.
    assert_eq!(
        written.len(),
        2,
        "D08 (one evaporation hydro, two study stages) must yield two per-stage rows"
    );

    for (got, want) in written.iter().zip(&expected) {
        assert_eq!(got.hydro_id, want.hydro_id, "hydro_id must match");
        assert_eq!(got.stage_id, want.stage_id, "stage_id must match");
        assert_eq!(
            got.intercept_m3s.to_bits(),
            want.intercept_m3s.to_bits(),
            "intercept_m3s must be bit-identical to the resolver output"
        );
        assert_eq!(
            got.volume_slope_m3s_per_hm3.to_bits(),
            want.volume_slope_m3s_per_hm3.to_bits(),
            "volume_slope_m3s_per_hm3 must be bit-identical to the resolver output"
        );
        assert_eq!(
            got.reference_volume_hm3.to_bits(),
            want.reference_volume_hm3.to_bits(),
            "reference_volume_hm3 must be bit-identical to the resolver output"
        );
        assert_eq!(got.source, want.source, "source tag must match");
    }

    // Both per-stage rows carry an explicit stage_id (the multi-stage shape).
    assert_eq!(written[0].stage_id, Some(0), "first row tags stage 0");
    assert_eq!(written[1].stage_id, Some(1), "second row tags stage 1");
}
