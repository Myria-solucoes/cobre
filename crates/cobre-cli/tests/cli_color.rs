//! Integration tests for the `--color` global flag and `COBRE_COLOR` / `FORCE_COLOR` env vars.
//!
//! Each test spawns a subprocess so that env var overrides and `console`'s global
//! `colors_enabled_stderr` state are completely isolated from the test process.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

// Minimal valid-case fixture, duplicated (not shared) with cli_run.rs to keep
// this test module self-contained.

const CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 1,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
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

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

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

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[test]
fn color_always_flag_forces_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            "--color",
            "always",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        // Auto-detection would disable color on a piped stderr subprocess; the
        // 256-color orange busbar escape is present only when color is forced on.
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// `--quiet` suppresses the `indicatif` progress bar, whose cursor-movement
/// sequences (`\x1b[1A`, `\x1b[2K`) are structural, not color-related, and would
/// otherwise survive `--color never` and defeat the no-ANSI assertion.
#[test]
fn color_never_flag_suppresses_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            "--color",
            "never",
            "--quiet",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[").not());
}

/// Accepted because the `--color` arg is declared with `global = true`.
#[test]
fn color_always_global_flag_before_subcommand_is_accepted() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "--color",
            "always",
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

#[test]
fn cobre_color_env_always_forces_ansi() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("COBRE_COLOR", "always")
        .env_remove("FORCE_COLOR")
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

#[test]
fn force_color_env_forces_ansi() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("FORCE_COLOR", "1")
        // COBRE_COLOR outranks FORCE_COLOR; unset it so the FORCE_COLOR branch runs.
        .env_remove("COBRE_COLOR")
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// An invalid value falls back to auto-detection, which disables color on a
/// non-TTY subprocess; the only meaningful assertion is success (no panic).
#[test]
fn cobre_color_env_invalid_value_is_silently_ignored() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    cobre()
        .env("COBRE_COLOR", "invalid-value")
        .env_remove("FORCE_COLOR")
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success();
}
