//! Integration tests for the `cobre validate` subcommand.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

// ── fixture helpers ───────────────────────────────────────────────────────────

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 10,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 100 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    }
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
            "num_scenarios": 50
        }
    ]
}"#;

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

/// Build a minimal valid case directory in `dir`.
fn make_valid_case(dir: &TempDir) {
    let root = dir.path();
    write_file(root, "config.json", CONFIG_JSON);
    write_file(root, "penalties.json", PENALTIES_JSON);
    write_file(root, "stages.json", STAGES_JSON);
    write_file(root, "initial_conditions.json", INITIAL_CONDITIONS_JSON);
    write_file(root, "system/buses.json", BUSES_JSON);
    write_file(root, "system/lines.json", LINES_JSON);
    write_file(root, "system/hydros.json", HYDROS_JSON);
    write_file(root, "system/thermals.json", THERMALS_JSON);
}

#[test]
fn valid_case_exits_0() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn valid_case_stdout_contains_buses_count() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("buses,"));
}

#[test]
fn missing_buses_json_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn missing_buses_json_stdout_contains_error() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("error"));
}

#[test]
fn missing_buses_json_stdout_mentions_file() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("buses.json"));
}

/// stderr must NOT carry the "run `cobre validate`" hint — that would point the
/// user back at the very command they just ran.
#[test]
fn validate_failure_report_in_stdout_not_stderr() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();
    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("error"))
        .stderr(predicate::str::contains("buses.json").not())
        .stderr(predicate::str::contains("run `cobre validate`").not());
}

/// The offending path must surface exactly once: the relative prefix and the
/// embedded `SchemaError` display both carry it, so the two must be deduped.
#[test]
fn validate_schema_failure_path_appears_once() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    write_file(
        dir.path(),
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1" }, { "id": 1, "name": "BUS_2" }] }"#,
    );

    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "expected validation failure");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let occurrences = stdout.matches("system/buses.json").count();
    assert_eq!(
        occurrences, 1,
        "offending path must appear exactly once, found {occurrences} in: {stdout:?}"
    );
}

#[test]
fn nonexistent_path_exits_2() {
    cobre()
        .args(["validate", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn nonexistent_path_stderr_mentions_path() {
    cobre()
        .args(["validate", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("nonexistent"));
}

#[test]
fn valid_case_piped_stdout_has_no_ansi_escapes() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    // `console` strips ANSI codes when stdout is not a terminal, as it is here.
    let output = cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains('\x1b'),
        "stdout should contain no ANSI escape sequences when piped, got: {stdout:?}"
    );
}

// These tests cover failures that pass the six-layer IO pipeline but are caught
// only by the three additional phases (StudyParams::from_config, prepare_stochastic,
// prepare_hydro_models_from_artifacts) that `validate` exercises.

/// An unknown `cut_selection` key is a hard schema error under
/// `deny_unknown_fields`, not silently ignored.
#[test]
fn removed_cut_selection_field_fails_validate() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);

    let removed_field_config = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 },
            "cut_selection": { "basis_activity_window": 100 }
        }
    }"#;
    write_file(dir.path(), "config.json", removed_field_config);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure();
}

/// An FPHA hydro with no `hydro_production_models.json` entry slips past the
/// IO pipeline (the Layer-4 dimensional check skips FPHA hydros when
/// `fpha_hyperplanes.parquet` is absent) and is rejected only at Phase 10 by
/// `prepare_hydro_models_from_artifacts` → `determine_source`.
#[test]
fn fpha_hydro_without_production_models_json_fails_validate() {
    let dir = TempDir::new().unwrap();

    write_file(dir.path(), "config.json", CONFIG_JSON);
    write_file(dir.path(), "penalties.json", PENALTIES_JSON);
    write_file(dir.path(), "stages.json", STAGES_JSON);
    write_file(
        dir.path(),
        "initial_conditions.json",
        INITIAL_CONDITIONS_JSON,
    );
    write_file(dir.path(), "system/buses.json", BUSES_JSON);
    write_file(dir.path(), "system/lines.json", LINES_JSON);
    write_file(dir.path(), "system/thermals.json", THERMALS_JSON);

    let fpha_hydros_json = r#"{
        "hydros": [
            {
                "id": 1,
                "name": "UHE_FPHA",
                "bus_id": 1,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 500.0
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "fpha",
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 100.0,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0
                }
            }
        ]
    }"#;
    write_file(dir.path(), "system/hydros.json", fpha_hydros_json);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn fpha_hydro_without_production_models_json_stdout_mentions_file() {
    let dir = TempDir::new().unwrap();

    write_file(dir.path(), "config.json", CONFIG_JSON);
    write_file(dir.path(), "penalties.json", PENALTIES_JSON);
    write_file(dir.path(), "stages.json", STAGES_JSON);
    write_file(
        dir.path(),
        "initial_conditions.json",
        INITIAL_CONDITIONS_JSON,
    );
    write_file(dir.path(), "system/buses.json", BUSES_JSON);
    write_file(dir.path(), "system/lines.json", LINES_JSON);
    write_file(dir.path(), "system/thermals.json", THERMALS_JSON);

    let fpha_hydros_json = r#"{
        "hydros": [
            {
                "id": 1,
                "name": "UHE_FPHA",
                "bus_id": 1,
                "downstream_id": null,
                "reservoir": {
                    "min_storage_hm3": 0.0,
                    "max_storage_hm3": 500.0
                },
                "outflow": {
                    "min_outflow_m3s": 0.0,
                    "max_outflow_m3s": null
                },
                "generation": {
                    "model": "fpha",
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 100.0,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0
                }
            }
        ]
    }"#;
    write_file(dir.path(), "system/hydros.json", fpha_hydros_json);

    cobre()
        .args(["validate", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stdout(predicate::str::contains("hydro_production_models.json"));
}
