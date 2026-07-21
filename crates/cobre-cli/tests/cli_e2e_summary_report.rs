//! Cross-binary end-to-end parity test: `run` -> `summary` -> `report` on the
//! deterministic `examples/1dtoy` quickstart case (seed 42, all `in_sample`).
//!
//! Timing tokens are intentionally NOT asserted: `summary` recomputes the
//! `Time split` percentages and `complete in Xs` durations from persisted
//! metadata rather than the live clock, so those tokens differ between the
//! live run and the reconstruction. Only content bit-identical across both
//! surfaces is compared.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

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

/// Expected simulation mean cost for `examples/1dtoy` (seed 42, deterministic).
const EXPECTED_MEAN_COST: f64 = 14_532_064.352_935_942;

#[test]
fn run_then_summary_and_report_preserve_the_deterministic_end_block() {
    let case = case_dir();
    assert!(
        case.is_dir(),
        "committed example case must exist at {}",
        case.display()
    );

    let out = TempDir::new().unwrap();
    let out_path = out.path();

    // No --quiet, so the end-block lands on stderr.
    let (_, run_stderr) = run_ok(&[
        "run",
        case.to_str().unwrap(),
        "--output",
        out_path.to_str().unwrap(),
    ]);

    // The four end-block source artifacts that `summary` and `report` consume.
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

    let (_, summary_stderr) = run_ok(&["summary", out_path.to_str().unwrap()]);

    assert_ordered(&summary_stderr, "Execution", "Hydro models");
    assert_ordered(&summary_stderr, "Hydro models", "Model provenance");
    assert_ordered(&summary_stderr, "Model provenance", "Training complete in");
    assert_ordered(
        &summary_stderr,
        "Training complete in",
        "Simulation complete",
    );

    // Deterministic content that must appear bit-identical in both the live run
    // and the summary reconstruction.
    for needle in [
        "Lower bound:  1.55955e7",
        "Upper bound:  5.79592e5",
        "LP solves:    5632",
        "Expected cost: 1.45321e7",
        "1 constant",
        "0 linearized, 1 without",
        // Provenance Inflow line; the hydro-production sub-section is
        // zero-suppressed for constant production (n_fpha=0).
        "user_stats_white_noise",
    ] {
        assert!(
            run_stderr.contains(needle),
            "live run stderr must contain {needle:?}"
        );
        assert!(
            summary_stderr.contains(needle),
            "summary stderr must contain {needle:?}"
        );
    }

    // Time split (training): three-line Forward/Backward/Serial format.
    // Asserted against `run_stderr` only — `cobre summary` reconstructs
    // `TrainingSummary` from `metadata.json`, which does not persist
    // per-iteration phase-wall timing, so this block never appears in
    // `summary_stderr` (see `build_training_summary`).
    let forward_line = run_stderr
        .lines()
        .find(|l| l.contains("Forward"))
        .unwrap_or_else(|| panic!("expected a Forward Time-split line in run stderr"));
    let backward_line = run_stderr
        .lines()
        .find(|l| l.contains("Backward"))
        .unwrap_or_else(|| panic!("expected a Backward Time-split line in run stderr"));
    let serial_line = run_stderr
        .lines()
        .find(|l| l.contains("Serial"))
        .unwrap_or_else(|| panic!("expected a Serial Time-split line in run stderr"));

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
        "Forward line must show the solve/wait decomposition for a local, \
         per-worker-data-present run, got: {forward_line:?}"
    );
    assert!(
        backward_line.contains("solve") && backward_line.contains("wait"),
        "Backward line must show the solve/wait decomposition for a local, \
         per-worker-data-present run, got: {backward_line:?}"
    );
    assert_ordered(&run_stderr, "Forward", "Backward");
    assert_ordered(&run_stderr, "Backward", "Serial");

    let (report_stdout, _) = run_ok(&["report", out_path.to_str().unwrap()]);
    let value: serde_json::Value = serde_json::from_str(&report_stdout).unwrap();

    let nested = value["simulation"]["cost"]["mean_cost"].as_f64();
    assert!(
        nested.is_some(),
        ".simulation.cost.mean_cost must be present and non-null"
    );
    let mean_cost = nested.unwrap();
    let rel_err = (mean_cost - EXPECTED_MEAN_COST).abs() / EXPECTED_MEAN_COST;
    assert!(
        rel_err < 1e-3,
        ".simulation.cost.mean_cost = {mean_cost} is not within 1e-3 of \
         {EXPECTED_MEAN_COST} (relative error {rel_err})"
    );

    assert_eq!(
        value["cost"]["mean_cost"].as_f64(),
        nested,
        ".cost.mean_cost must equal .simulation.cost.mean_cost"
    );

    // `report` reads these from persisted metadata; `summary`'s "LP solves:"
    // line re-reads the convergence parquet. For a fresh run both reflect the
    // same solves, so the golden counts (5632 training, 400 simulation) lock
    // the metadata-vs-display relationship.
    assert_eq!(
        value["training"]["solve_stats"]["total_lp_solves"].as_u64(),
        Some(5632),
        ".training.solve_stats.total_lp_solves must equal the summary display count"
    );
    assert_eq!(
        value["simulation"]["solve_stats"]["total_lp_solves"].as_u64(),
        Some(400),
        ".simulation.solve_stats.total_lp_solves must equal the summary display count"
    );
}
