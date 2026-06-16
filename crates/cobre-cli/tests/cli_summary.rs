//! Integration tests for the `cobre summary` subcommand.
//!
//! These tests build a hand-crafted output-directory fixture (the four metadata
//! files) in a tempdir, then run the real `cobre` binary against it and assert
//! on the live end-block written to stderr, in top-to-bottom order:
//!
//!   Execution topology → Hydro models → Model provenance → Fit quality →
//!   Training → Simulation
//!
//! Fit quality is conditional: it renders only when the training metadata carries
//! `production_fit_deviation` (computed-FPHA runs).
//!
//! The fixture is written with the public `cobre_io` writers (training/simulation
//! metadata) and `serde_json` (the two optional sidecars), so the JSON shapes
//! stay in sync with the real schemas without needing a full live run.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use cobre_io::{
    DeviationSummary, DeviationWorstEntry, DistributionInfo, HostLayout, MetadataBounds,
    MetadataConfiguration, MetadataConvergence, MetadataCost, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    MetadataTrainingSolveStats, SimulationMetadata, TrainingMetadata, write_simulation_metadata,
    write_training_metadata,
};
use predicates::prelude::*;
use serde_json::json;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn local_distribution() -> DistributionInfo {
    DistributionInfo {
        backend: "local".to_string(),
        world_size: 1,
        ranks_participated: 1,
        num_nodes: 1,
        threads_per_rank: 1,
        mpi_library: None,
        mpi_standard: None,
        thread_level: None,
        slurm_job_id: None,
        hosts: vec![HostLayout {
            hostname: "fixture-host".to_string(),
            ranks: vec![0],
        }],
    }
}

fn write_training_fixture(dir: &Path) {
    write_training_fixture_with_deviation(dir, None);
}

fn write_training_fixture_with_deviation(dir: &Path, deviation: Option<DeviationSummary>) {
    let metadata = TrainingMetadata {
        cobre_version: "0.0.0-test".to_string(),
        hostname: "fixture-host".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.7.0".to_string()),
        started_at: "2026-01-17T08:00:00Z".to_string(),
        completed_at: "2026-01-17T12:30:00Z".to_string(),
        duration_seconds: 16_200.0,
        status: "complete".to_string(),
        configuration: MetadataConfiguration {
            seed: Some(42),
            max_iterations: Some(100),
            forward_passes: Some(192),
            stopping_mode: "any".to_string(),
            policy_mode: "fresh".to_string(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: 12,
            num_hydros: 1,
            num_thermals: 1,
            num_buses: 1,
            num_lines: 0,
        },
        iterations: MetadataIterations {
            completed: 10,
            converged_at: Some(10),
        },
        convergence: MetadataConvergence {
            achieved: true,
            final_gap_percent: Some(0.45),
            termination_reason: "gap_tolerance".to_string(),
        },
        row_pool: MetadataRowPool {
            total_generated: 100,
            total_active: 90,
            peak_active: 100,
            cuts_active: 90,
            rows_in_lp_total: 0,
            rows_in_lp_solve_count: 0,
            rows_in_lp_max: 0,
        },
        bounds: MetadataBounds {
            final_lower_bound: 48_500.0,
            final_upper_bound: Some(49_000.0),
            final_upper_bound_std: Some(250.0),
        },
        solve_stats: MetadataTrainingSolveStats {
            total_lp_solves: Some(840),
            first_try: Some(800),
            retried: Some(38),
            failed: Some(2),
            forward_solve_seconds: Some(1.5),
            backward_solve_seconds: Some(4.5),
            parallelism: Some(1),
        },
        setup: None,
        production_fit_deviation: deviation,
        distribution: local_distribution(),
    };

    std::fs::create_dir_all(dir.join("training")).unwrap();
    write_training_metadata(&dir.join("training/metadata.json"), &metadata).unwrap();
}

fn write_simulation_fixture(dir: &Path) {
    let metadata = SimulationMetadata {
        cobre_version: "0.0.0-test".to_string(),
        hostname: "fixture-host".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.7.0".to_string()),
        started_at: "2026-01-17T12:30:00Z".to_string(),
        completed_at: "2026-01-17T12:30:12Z".to_string(),
        duration_seconds: 12.5,
        status: "complete".to_string(),
        scenarios: MetadataScenarios {
            total: 192,
            completed: 192,
            failed: 0,
        },
        cost: Some(MetadataCost {
            mean_cost: 1.0e6,
            std_cost: 2.0e4,
            cvar: 1.2e6,
            cvar_alpha: 0.95,
        }),
        solve_stats: MetadataSimulationSolveStats {
            total_lp_solves: Some(5_000),
            first_try: Some(4_900),
            retried: Some(90),
            failed: Some(10),
            solve_seconds: Some(3.0),
            parallelism: Some(1),
        },
        distribution: local_distribution(),
    };

    std::fs::create_dir_all(dir.join("simulation")).unwrap();
    write_simulation_metadata(&dir.join("simulation/metadata.json"), &metadata).unwrap();
}

fn write_hydro_models_fixture(dir: &Path) {
    let summary = json!({
        "n_constant": 1,
        "n_fpha": 0,
        "total_planes": 0,
        "fpha_details": [],
        "n_evaporation": 0,
        "n_no_evaporation": 1,
        "n_user_supplied_ref": 0,
        "n_default_midpoint_ref": 0
    });
    std::fs::create_dir_all(dir.join("training")).unwrap();
    std::fs::write(
        dir.join("training/hydro_models.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();
}

fn write_provenance_fixture(dir: &Path) {
    let report = json!({
        "inflow": {
            "estimation_path": "full_estimation",
            "seasonal_stats_source": "estimated",
            "ar_coefficients_source": "estimated",
            "correlation_source": "estimated",
            "opening_tree_source": "estimated",
            "n_hydros": 1,
            "ar_method": "AIC",
            "ar_max_order": 2,
            "white_noise_fallbacks": []
        },
        "hydro_production": {
            "n_fpha_computed_from_geometry": 0,
            "n_fpha_precomputed_hyperplanes": 0,
            "n_evaporation_ref_user_supplied": 0,
            "n_evaporation_ref_default_midpoint": 0
        }
    });
    std::fs::create_dir_all(dir.join("training")).unwrap();
    std::fs::write(
        dir.join("training/model_provenance.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
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

#[test]
fn summary_prints_all_five_sections_in_live_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    write_hydro_models_fixture(path);
    write_provenance_fixture(path);
    write_simulation_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // All five section headers present, in live top-to-bottom order.
    assert_ordered(&stderr, "Execution", "Hydro models");
    assert_ordered(&stderr, "Hydro models", "Model provenance");
    assert_ordered(&stderr, "Model provenance", "Training complete in");
    assert_ordered(&stderr, "Training complete in", "Simulation complete");
}

#[test]
fn summary_renders_fit_quality_from_metadata_deviation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    // A computed-FPHA run persists `production_fit_deviation`; the read-back path
    // must surface it as a Fit quality block between Model provenance and Training.
    let deviation = DeviationSummary {
        n_entries: 3,
        mean_abs: 12.0,
        max_abs: 40.0,
        worst_relative: 0.032,
        worst_entry: Some(DeviationWorstEntry {
            entity_id: 12,
            stage_id: 1,
            relative: 0.032,
            mean_abs: 12.0,
            max_abs: 40.0,
        }),
    };
    write_training_fixture_with_deviation(path, Some(deviation));
    write_hydro_models_fixture(path);
    write_provenance_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // The Fit quality block renders, carries the worst-deviation line, and sits
    // after Model provenance and before Training in the live block order.
    assert!(stderr.contains("Fit quality"));
    assert!(stderr.contains("Worst FPHA dev: 3.2% relative (12/1)"));
    assert_ordered(&stderr, "Model provenance", "Fit quality");
    assert_ordered(&stderr, "Fit quality", "Training complete in");
}

#[test]
fn summary_skips_fit_quality_when_deviation_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    // The default fixture sets `production_fit_deviation: None` (no computed FPHA).
    write_training_fixture(path);
    write_hydro_models_fixture(path);
    write_provenance_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // No deviation in metadata → the Fit quality block is skipped entirely.
    assert!(!stderr.contains("Fit quality"));
    assert!(stderr.contains("Model provenance"));
    assert!(stderr.contains("Training complete in"));
}

#[test]
fn summary_skips_model_provenance_when_sidecar_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    write_hydro_models_fixture(path);
    // model_provenance.json deliberately omitted.
    write_simulation_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // The provenance section is skipped, but the surrounding sections remain.
    assert!(!stderr.contains("Model provenance"));
    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Hydro models"));
    assert!(stderr.contains("Training complete in"));
    assert!(stderr.contains("Simulation complete"));
}

#[test]
fn summary_skips_hydro_models_when_sidecar_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    // hydro_models.json deliberately omitted.
    write_provenance_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(!stderr.contains("Hydro models"));
    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Model provenance"));
    assert!(stderr.contains("Training complete in"));
}

#[test]
fn summary_only_training_required_minimal_dir() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    // Only the required training metadata; all optional sidecars absent.
    write_training_fixture(path);

    let output = cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Execution topology and Training always render from the required metadata.
    assert!(stderr.contains("Execution"));
    assert!(stderr.contains("Training complete in"));
    // Every optional section is skipped.
    assert!(!stderr.contains("Hydro models"));
    assert!(!stderr.contains("Model provenance"));
    assert!(!stderr.contains("Simulation complete"));
}

#[test]
fn summary_malformed_provenance_sidecar_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();

    write_training_fixture(path);
    std::fs::create_dir_all(path.join("training")).unwrap();
    // A present-but-corrupt sidecar must surface as an error, not be skipped.
    std::fs::write(
        path.join("training/model_provenance.json"),
        "{ this is not valid json",
    )
    .unwrap();

    cobre()
        .args(["summary", path.to_str().unwrap()])
        .assert()
        .failure()
        .code(predicate::eq(4));
}
