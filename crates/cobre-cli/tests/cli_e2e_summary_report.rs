//! Cross-binary end-to-end parity test: `run` -> `summary` -> `report`.
//!
//! This test proves the CLI-only e2e contract for the deterministic
//! `examples/1dtoy` quickstart case:
//!
//! 1. A real `cobre run` against `examples/1dtoy` produces a results directory
//!    and emits the live end-block on stderr.
//! 2. `cobre summary` reconstructs the same five-section end-block from the
//!    persisted results directory, reproducing the deterministic numbers
//!    bit-for-bit (lower/upper bounds, LP-solve counts, hydro-model and
//!    provenance lines).
//! 3. `cobre report` exposes the simulation expected cost as machine-readable
//!    JSON on stdout, under both `.simulation.cost.mean_cost` and the top-level
//!    `.cost.mean_cost` convenience key.
//!
//! The case is fully deterministic (seed 42, all `in_sample` scenario schemes),
//! runs in well under a second on a debug build, and is therefore an ungated
//! integration test (no `slow-tests` gate).
//!
//! Timing tokens are intentionally NOT asserted: `summary` recomputes the
//! `Time split` percentages and `complete in Xs` durations from persisted
//! metadata rather than the live clock, so those tokens differ between the
//! live run and the reconstruction (measured: `9%` vs `12%`, `0.2s` vs `0.1s`).
//! Only the deterministic numeric/text content that is bit-identical across both
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

/// The compiled `cobre` binary under test.
fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

/// Run `cobre` with `args`, assert success, and return its decoded
/// `(stdout, stderr)`.
fn run_ok(args: &[&str]) -> (String, String) {
    let output = cobre().args(args).assert().success().get_output().clone();
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

/// Resolve the committed `examples/1dtoy` case directory relative to this
/// crate's manifest. The `cobre-cli` manifest dir is `crates/cobre-cli`; two
/// parents up is the workspace root.
fn case_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/1dtoy")
}

/// Assert that `needle_a` appears strictly before `needle_b` in `haystack`.
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

    // ── 1. Live run (no --quiet) so the end-block lands on stderr ───────────
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

    // ── 2. Summary reconstruction of the end-block from the results dir ─────
    let (_, summary_stderr) = run_ok(&["summary", out_path.to_str().unwrap()]);

    // ── 3. All five sections present in live top-to-bottom order ────────────
    assert_ordered(&summary_stderr, "Execution", "Hydro models");
    assert_ordered(&summary_stderr, "Hydro models", "Model provenance");
    assert_ordered(&summary_stderr, "Model provenance", "Training complete in");
    assert_ordered(
        &summary_stderr,
        "Training complete in",
        "Simulation complete",
    );

    // ── 4. Deterministic content appears in BOTH the live run and summary ───
    //
    // This is the "diff vs live end-block is empty" parity metric, applied to
    // the time-stripped deterministic lines. Timing tokens (`Time split`
    // percentages, `complete in Xs`) are deliberately excluded because
    // `summary` recomputes them from persisted metadata, not the live clock.
    for needle in [
        // Training end-block.
        "Lower bound:  1.55955e7",
        "Upper bound:  5.79592e5",
        "LP solves:    5632",
        // Simulation end-block.
        "Expected cost: 1.45321e7",
        // Hydro-models section.
        "1 constant",
        "0 linearized, 1 without",
        // Model-provenance section (Inflow line; the hydro-production
        // sub-section is zero-suppressed for constant production, n_fpha=0).
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

    // ── 5. Report exposes the simulation cost as machine-readable JSON ──────
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

    // The top-level convenience key must mirror the nested cost exactly.
    assert_eq!(
        value["cost"]["mean_cost"].as_f64(),
        nested,
        ".cost.mean_cost must equal .simulation.cost.mean_cost"
    );

    // ── 6. Metadata LP-solve counts match the summary "LP solves" display ──
    //
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
