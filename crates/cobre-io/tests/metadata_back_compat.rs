//! Frozen-fixture back-compatibility and forward-roundtrip tests for the
//! extended output metadata structs.
//!
//! These tests guard the additive-evolution contract for `training/metadata.json`
//! and `simulation/metadata.json`: new fields must be `#[serde(default)]` so that
//! output directories produced by an older cobre still deserialize cleanly.
//!
//! The legacy fixtures here are **hand-frozen JSON string literals** that omit the
//! newer keys (`bounds`, `solve_stats`, `cost`, and `distribution.hosts`). They are
//! deliberately NOT built by serializing a current struct — a struct-built fixture
//! always carries every field and therefore could never catch a field that was
//! accidentally made required.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use cobre_io::{
    DistributionInfo, HostLayout, MetadataBounds, MetadataConfiguration, MetadataConvergence,
    MetadataCost, MetadataIterations, MetadataProblemDimensions, MetadataRowPool,
    MetadataScenarios, MetadataSimulationSolveStats, MetadataTrainingSolveStats,
    SimulationMetadata, TrainingMetadata, read_simulation_metadata, read_training_metadata,
    write_simulation_metadata, write_training_metadata,
};

// ── Frozen legacy fixtures ─────────────────────────────────────────────────────
//
// These literals represent the PRE-CHANGE shapes of the metadata files. They must
// remain verbatim and must NOT contain the keys `bounds`, `solve_stats`, `cost`,
// or `hosts`. The dedicated `*_omits_new_keys` tests below assert this property so
// that an accidental edit cannot quietly defeat the back-compat guarantee.

/// Legacy `training/metadata.json` as produced before the `bounds`, `solve_stats`,
/// and `distribution.hosts` fields existed.
const LEGACY_TRAINING_JSON: &str = r#"{
  "cobre_version": "0.1.0",
  "hostname": "legacy-host",
  "solver": "highs",
  "solver_version": "1.8.0",
  "started_at": "2026-01-17T08:00:00Z",
  "completed_at": "2026-01-17T12:30:00Z",
  "duration_seconds": 16200.0,
  "status": "complete",
  "configuration": {
    "seed": 42,
    "max_iterations": 100,
    "forward_passes": 192,
    "stopping_mode": "any",
    "policy_mode": "fresh"
  },
  "problem_dimensions": {
    "num_stages": 12,
    "num_hydros": 160,
    "num_thermals": 200,
    "num_buses": 5,
    "num_lines": 8
  },
  "iterations": {
    "completed": 100,
    "converged_at": 95
  },
  "convergence": {
    "achieved": true,
    "final_gap_percent": 0.45,
    "termination_reason": "bound_stalling"
  },
  "row_pool": {
    "total_generated": 1250000,
    "total_active": 980000,
    "peak_active": 1100000
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1
  }
}"#;

/// Legacy `simulation/metadata.json` as produced before the `cost`, `solve_stats`,
/// and `distribution.hosts` fields existed.
const LEGACY_SIM_JSON: &str = r#"{
  "cobre_version": "0.1.0",
  "hostname": "legacy-host",
  "solver": "highs",
  "solver_version": "1.8.0",
  "started_at": "2026-01-17T13:00:00Z",
  "completed_at": "2026-01-17T13:15:00Z",
  "duration_seconds": 900.0,
  "status": "complete",
  "scenarios": {
    "total": 100,
    "completed": 100,
    "failed": 0
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1
  }
}"#;

/// Multi-node `DistributionInfo` fixture with a two-host `hosts` array. Frozen to
/// pin the on-disk shape of per-host rank assignments.
const MULTI_NODE_DISTRIBUTION_JSON: &str = r#"{
  "backend": "mpi",
  "world_size": 8,
  "ranks_participated": 8,
  "num_nodes": 2,
  "threads_per_rank": 4,
  "mpi_library": "Open MPI v4.1.6",
  "mpi_standard": "MPI 4.0",
  "thread_level": "Funneled",
  "hosts": [
    { "hostname": "node01", "ranks": [0, 1, 2, 3] },
    { "hostname": "node02", "ranks": [4, 5, 6, 7] }
  ]
}"#;

// ── Back-compat: legacy JSON deserializes with documented defaults ──────────────

#[test]
fn legacy_training_json_deserializes_with_defaults() {
    let decoded: TrainingMetadata = serde_json::from_str(LEGACY_TRAINING_JSON)
        .expect("legacy training metadata must still deserialize");

    // `bounds` defaults to zeroed lower bound and absent upper bounds.
    assert_eq!(decoded.bounds.final_lower_bound, 0.0);
    assert_eq!(decoded.bounds.final_upper_bound, None);
    assert_eq!(decoded.bounds.final_upper_bound_std, None);

    // `solve_stats` defaults to all-`None`.
    assert_eq!(decoded.solve_stats.total_lp_solves, None);
    assert_eq!(decoded.solve_stats.first_try, None);
    assert_eq!(decoded.solve_stats.retried, None);
    assert_eq!(decoded.solve_stats.failed, None);
    assert_eq!(decoded.solve_stats.forward_solve_seconds, None);
    assert_eq!(decoded.solve_stats.backward_solve_seconds, None);
    assert_eq!(decoded.solve_stats.parallelism, None);

    // `distribution.hosts` defaults to an empty vector.
    assert!(decoded.distribution.hosts.is_empty());

    // Sanity: pre-existing fields still load correctly.
    assert_eq!(decoded.iterations.completed, 100);
    assert_eq!(decoded.row_pool.total_generated, 1_250_000);
}

#[test]
fn legacy_simulation_json_deserializes_with_defaults() {
    let decoded: SimulationMetadata = serde_json::from_str(LEGACY_SIM_JSON)
        .expect("legacy simulation metadata must still deserialize");

    // `cost` defaults to `None`.
    assert!(decoded.cost.is_none());

    // `solve_stats` defaults to all-`None`.
    assert_eq!(decoded.solve_stats.total_lp_solves, None);
    assert_eq!(decoded.solve_stats.first_try, None);
    assert_eq!(decoded.solve_stats.retried, None);
    assert_eq!(decoded.solve_stats.failed, None);
    assert_eq!(decoded.solve_stats.solve_seconds, None);
    assert_eq!(decoded.solve_stats.parallelism, None);

    // `distribution.hosts` defaults to an empty vector.
    assert!(decoded.distribution.hosts.is_empty());

    // Sanity: pre-existing fields still load correctly.
    assert_eq!(decoded.scenarios.total, 100);
    assert_eq!(decoded.scenarios.completed, 100);
}

// ── Guard: the legacy fixtures genuinely omit the new keys ──────────────────────

#[test]
fn legacy_training_fixture_omits_new_keys() {
    assert!(
        !LEGACY_TRAINING_JSON.contains("bounds"),
        "legacy training fixture must omit the `bounds` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("solve_stats"),
        "legacy training fixture must omit the `solve_stats` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("hosts"),
        "legacy training fixture must omit the `hosts` key to exercise back-compat"
    );
}

#[test]
fn legacy_simulation_fixture_omits_new_keys() {
    assert!(
        !LEGACY_SIM_JSON.contains("cost"),
        "legacy simulation fixture must omit the `cost` key to exercise back-compat"
    );
    assert!(
        !LEGACY_SIM_JSON.contains("solve_stats"),
        "legacy simulation fixture must omit the `solve_stats` key to exercise back-compat"
    );
    assert!(
        !LEGACY_SIM_JSON.contains("hosts"),
        "legacy simulation fixture must omit the `hosts` key to exercise back-compat"
    );
}

// ── Multi-node hosts fixture ────────────────────────────────────────────────────

#[test]
fn multi_node_distribution_deserializes_with_correct_rank_lists() {
    let decoded: DistributionInfo = serde_json::from_str(MULTI_NODE_DISTRIBUTION_JSON)
        .expect("multi-node distribution fixture must deserialize");

    assert_eq!(decoded.backend, "mpi");
    assert_eq!(decoded.world_size, 8);
    assert_eq!(decoded.num_nodes, 2);
    assert_eq!(decoded.hosts.len(), 2);

    assert_eq!(decoded.hosts[0].hostname, "node01");
    assert_eq!(decoded.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.hosts[1].hostname, "node02");
    assert_eq!(decoded.hosts[1].ranks, vec![4, 5, 6, 7]);
}

// ── Forward roundtrip: fully-populated metadata survives write → read ────────────

fn fully_populated_distribution() -> DistributionInfo {
    DistributionInfo {
        backend: "mpi".to_string(),
        world_size: 8,
        ranks_participated: 8,
        num_nodes: 2,
        threads_per_rank: 4,
        mpi_library: Some("Open MPI v4.1.6".to_string()),
        mpi_standard: Some("MPI 4.0".to_string()),
        thread_level: Some("Funneled".to_string()),
        slurm_job_id: Some("123456".to_string()),
        hosts: vec![
            HostLayout {
                hostname: "node01".to_string(),
                ranks: vec![0, 1, 2, 3],
            },
            HostLayout {
                hostname: "node02".to_string(),
                ranks: vec![4, 5, 6, 7],
            },
        ],
    }
}

fn fully_populated_training_metadata() -> TrainingMetadata {
    TrainingMetadata {
        cobre_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
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
            num_hydros: 160,
            num_thermals: 200,
            num_buses: 5,
            num_lines: 8,
        },
        iterations: MetadataIterations {
            completed: 100,
            converged_at: Some(95),
        },
        convergence: MetadataConvergence {
            achieved: true,
            final_gap_percent: Some(0.45),
            termination_reason: "bound_stalling".to_string(),
        },
        row_pool: MetadataRowPool {
            total_generated: 1_250_000,
            total_active: 980_000,
            peak_active: 1_100_000,
            cuts_active: 980_000,
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
            total_lp_solves: Some(84_000),
            first_try: Some(80_000),
            retried: Some(3_800),
            failed: Some(200),
            forward_solve_seconds: Some(123.5),
            backward_solve_seconds: Some(456.75),
            parallelism: Some(8),
        },
        distribution: fully_populated_distribution(),
    }
}

fn fully_populated_simulation_metadata() -> SimulationMetadata {
    SimulationMetadata {
        cobre_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
        started_at: "2026-01-17T13:00:00Z".to_string(),
        completed_at: "2026-01-17T13:15:00Z".to_string(),
        duration_seconds: 900.0,
        status: "complete".to_string(),
        scenarios: MetadataScenarios {
            total: 100,
            completed: 100,
            failed: 0,
        },
        cost: Some(MetadataCost {
            mean_cost: 12_345.6,
            std_cost: 200.0,
            cvar: 13_000.0,
            cvar_alpha: 0.95,
        }),
        solve_stats: MetadataSimulationSolveStats {
            total_lp_solves: Some(50_000),
            first_try: Some(48_000),
            retried: Some(1_900),
            failed: Some(100),
            solve_seconds: Some(321.0),
            parallelism: Some(8),
        },
        distribution: fully_populated_distribution(),
    }
}

#[test]
fn training_metadata_new_fields_survive_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_training_metadata();

    write_training_metadata(&path, &original).expect("write must succeed");
    let decoded = read_training_metadata(&path).expect("read must succeed");

    // `bounds.*` survive.
    assert_eq!(
        decoded.bounds.final_lower_bound,
        original.bounds.final_lower_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound,
        original.bounds.final_upper_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound_std,
        original.bounds.final_upper_bound_std
    );

    // training `solve_stats.*` survive.
    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.forward_solve_seconds,
        original.solve_stats.forward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.backward_solve_seconds,
        original.solve_stats.backward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );

    // `distribution.hosts` survive with their rank lists.
    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);
}

#[test]
fn simulation_metadata_new_fields_survive_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_simulation_metadata();

    write_simulation_metadata(&path, &original).expect("write must succeed");
    let decoded = read_simulation_metadata(&path).expect("read must succeed");

    // `cost.*` survive.
    let original_cost = original.cost.expect("fixture must populate cost");
    let decoded_cost = decoded.cost.expect("cost must survive roundtrip");
    assert_eq!(decoded_cost.mean_cost, original_cost.mean_cost);
    assert_eq!(decoded_cost.std_cost, original_cost.std_cost);
    assert_eq!(decoded_cost.cvar, original_cost.cvar);
    assert_eq!(decoded_cost.cvar_alpha, original_cost.cvar_alpha);

    // simulation `solve_stats.*` survive.
    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.solve_seconds,
        original.solve_stats.solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );

    // `distribution.hosts` survive with their rank lists.
    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);
}
