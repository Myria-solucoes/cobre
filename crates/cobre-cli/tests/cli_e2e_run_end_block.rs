//! End-to-end determinism test: `cobre run` on the committed `examples/1dtoy`
//! case (seed 42, all `in_sample`). Asserts deterministic strings in the live
//! run's stderr end-block, the shape and ordering of the Time-split block, and
//! golden LP-solve counts and mean cost read from the written metadata files.
//!
//! The golden numbers are stable because the case is deterministic, seed-pinned,
//! and input-committed; they move only when the committed case changes or the
//! solver/algorithm produces a different optimum.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn run_ok(args: &[&str]) -> (String, String) {
    let output = cobre().args(args).assert().success().get_output().clone();
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy")
}

fn assert_ordered(haystack: &str, needle_a: &str, needle_b: &str) {
    let pos_a = haystack
        .find(needle_a)
        .unwrap_or_else(|| panic!("expected to find {needle_a:?} in stderr"));
    let pos_b = haystack
        .find(needle_b)
        .unwrap_or_else(|| panic!("expected to find {needle_b:?} in stderr"));
    assert!(
        pos_a < pos_b,
        "expected {needle_a:?} (at {pos_a}) before {needle_b:?} (at {pos_b})"
    );
}

fn find_line<'a>(haystack: &'a str, label: &str) -> &'a str {
    haystack
        .lines()
        .find(|l| l.contains(label))
        .unwrap_or_else(|| panic!("expected a {label} Time-split line in run stderr"))
}

const EXPECTED_MEAN_COST: f64 = 9_679_385.922_404_844;

#[test]
fn run_produces_deterministic_end_block_and_metadata() {
    let case = case_dir();
    assert!(
        case.is_dir(),
        "committed example case must exist at {}",
        case.display()
    );

    let out = TempDir::new().unwrap();
    let out_path = out.path();

    let (_, run_stderr) = run_ok(&[
        "run",
        case.to_str().unwrap(),
        "--output",
        out_path.to_str().unwrap(),
    ]);

    for rel in [
        "training/metadata.json",
        "simulation/metadata.json",
        "training/hydro_models.json",
        "training/model_provenance.json",
    ] {
        let path = out_path.join(rel);
        assert!(
            path.is_file(),
            "run must write {rel} (at {})",
            path.display()
        );
    }

    for needle in [
        "Lower bound:  1.55955e7",
        "Upper bound:  5.79592e5",
        "LP solves:    5632",
        "Expected cost: 9.67939e6",
        "1 constant",
        "0 linearized, 1 without",
        "user_stats_white_noise",
    ] {
        assert!(
            run_stderr.contains(needle),
            "live run stderr must contain {needle:?}"
        );
    }

    let forward_line = find_line(&run_stderr, "Forward");
    let backward_line = find_line(&run_stderr, "Backward");
    let serial_line = find_line(&run_stderr, "Serial");

    for (label, line) in [
        ("Forward", forward_line),
        ("Backward", backward_line),
        ("Serial", serial_line),
    ] {
        assert!(
            line.contains('%'),
            "{label} Time-split line must carry a wall value with a percentage, got: {line:?}"
        );
    }
    assert!(
        forward_line.contains("solve") && forward_line.contains("wait"),
        "Forward line must show the solve/wait decomposition, got: {forward_line:?}"
    );
    assert!(
        backward_line.contains("solve") && backward_line.contains("wait"),
        "Backward line must show the solve/wait decomposition, got: {backward_line:?}"
    );
    assert_ordered(&run_stderr, "Forward", "Backward");
    assert_ordered(&run_stderr, "Backward", "Serial");

    let sim_metadata_content =
        std::fs::read_to_string(out_path.join("simulation/metadata.json")).unwrap();
    let sim_metadata: serde_json::Value = serde_json::from_str(&sim_metadata_content).unwrap();

    let mean_cost = sim_metadata["cost"]["mean_cost"]
        .as_f64()
        .expect(".cost.mean_cost must be present and non-null");
    let rel_err = (mean_cost - EXPECTED_MEAN_COST).abs() / EXPECTED_MEAN_COST;
    assert!(
        rel_err < 1e-3,
        "simulation metadata .cost.mean_cost = {mean_cost} is not within 1e-3 of \
         {EXPECTED_MEAN_COST} (relative error {rel_err})"
    );

    let training_metadata_content =
        std::fs::read_to_string(out_path.join("training/metadata.json")).unwrap();
    let training_metadata: serde_json::Value =
        serde_json::from_str(&training_metadata_content).unwrap();

    assert_eq!(
        training_metadata["solve_stats"]["total_lp_solves"].as_u64(),
        Some(5632),
        "training metadata .solve_stats.total_lp_solves must be 5632"
    );
    assert_eq!(
        sim_metadata["solve_stats"]["total_lp_solves"].as_u64(),
        Some(400),
        "simulation metadata .solve_stats.total_lp_solves must be 400"
    );
}
