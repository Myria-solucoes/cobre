//! Integration coverage for the `BoundaryInjection` load path's per-family
//! identity reconciliation: storage and inflow-lag reconcile by identity and
//! REJECT on a missing core slot; a `FullFcf`-typed manifest check stays on
//! the exact-match path, unaffected by wiring `reconcile` into
//! `load_boundary_cuts`.
//!
//! Extended with transit-bucket (identity match copies, a miss defaults to
//! `0.0`), sentinel-dated-anticipated default-`0.0`, and dated-anticipated
//! graceful-rejection coverage as those families gain their own reconcile
//! arms.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use cobre_io::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest, ManifestEdge,
    ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock, StageCutsPayload,
    write_policy_checkpoint,
};
use cobre_sddp::{
    BoundaryInjection, FullFcf, PolicyStageManifest, load_boundary_cuts, validate_policy_load,
};

/// `EntityType::HydroStorage` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_STORAGE: u8 = 0;
/// `EntityType::HydroInflowLag` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_INFLOW_LAG: u8 = 1;
/// `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;
/// `EntityType::HydroTransitBucket` discriminant from `schemas/policy.fbs`.
const ENTITY_TYPE_HYDRO_TRANSIT_BUCKET: u8 = 3;

fn storage_slot(id: i32) -> EntitySlot {
    EntitySlot {
        entity_type: ENTITY_TYPE_HYDRO_STORAGE,
        entity_id: id,
        subindex: 0,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
    EntitySlot {
        entity_type: ENTITY_TYPE_HYDRO_INFLOW_LAG,
        entity_id: id,
        subindex: lag_depth,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn transit_bucket_slot(downstream_hydro_id: i32, lag: u32) -> EntitySlot {
    EntitySlot {
        entity_type: ENTITY_TYPE_HYDRO_TRANSIT_BUCKET,
        entity_id: downstream_hydro_id,
        subindex: lag,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn sentinel_anticipated_slot(thermal_id: i32, ring_slot: u32) -> EntitySlot {
    EntitySlot {
        entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
        entity_id: thermal_id,
        subindex: ring_slot,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn dated_anticipated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
    EntitySlot {
        entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
        entity_id: thermal_id,
        subindex: ring_slot,
        was_active: true,
        delivery_date,
    }
}

fn producer_block() -> ProducerBlock {
    ProducerBlock {
        completed_iterations: 10,
        final_lower_bound: 0.0,
        best_upper_bound: None,
        max_iterations: 50,
        forward_passes: 1,
        warm_start_cuts: 0,
        warm_start_counts: vec![],
        rng_seed: 0,
        total_visited_states: 0,
        training_block_mode: "parallel".to_string(),
        training_block_mode_per_stage: vec![],
        cost_scale_factor: None,
    }
}

/// A 1-stage chain graph manifest (node id == stage id == pool id).
fn single_stage_manifest() -> GraphManifest {
    GraphManifest {
        n_pools: 1,
        nodes: vec![ManifestNode {
            id: 0,
            stage_id: 0,
            pool_id: 0,
        }],
        edges: Vec::<ManifestEdge>::new(),
    }
}

/// Write a single-stage checkpoint whose one cut carries `coefficients`, one
/// per `manifest` slot in the same order.
fn write_checkpoint(dir: &std::path::Path, manifest: &[EntitySlot], coefficients: &[f64]) {
    let state_dimension = u32::try_from(coefficients.len()).expect("small coefficient count");
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: 1.0,
        coefficients,
        is_active: true,
    };
    let cuts = vec![cut];
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension,
        capacity: 1,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: manifest,
    };
    let metadata = PolicyCheckpointMetadata {
        format_version: FORMAT_VERSION,
        cobre_version: "0.14.0".to_string(),
        created_at: "2026-08-11T00:00:00Z".to_string(),
        num_stages: 1,
        graph_manifest: single_stage_manifest(),
        producer: producer_block(),
    };
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// Given a `BoundaryInjection` load whose source matches the current
/// terminal manifest's storage and inflow-lag slots by identity, when the
/// load runs, then it succeeds and the coefficients land at their
/// identity-matched positions.
#[test]
fn boundary_injection_storage_lag_identity_match_succeeds() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), inflow_lag_slot(1, 1)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), inflow_lag_slot(1, 1)];
    let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, None, 1_000_000.0, &mut |_| {})
        .expect("an identity-matching boundary must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 20.0],
        "coefficients must land at their identity-matched positions"
    );
}

/// Given a current terminal storage slot for a hydro the boundary source
/// never prices, when the `BoundaryInjection` load runs, then it rejects,
/// naming the unpriced hydro.
#[test]
fn boundary_injection_different_hydro_source_rejects_naming_hydro() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(7)];
    write_checkpoint(tmp.path(), &manifest, &[10.0]);

    let current = vec![storage_slot(42)];
    let result = load_boundary_cuts(tmp.path(), 0, 1, &current, None, 1_000_000.0, &mut |_| {});

    let msg = result
        .expect_err("an unpriced hydro must reject")
        .to_string();
    assert!(
        msg.contains("42"),
        "error must name the unpriced hydro 42: {msg}"
    );
}

/// A `FullFcf`-typed manifest check over the same differently-shaped pair
/// stays on the exact-match path: `validate_policy_load::<FullFcf>` still
/// hard-rejects the per-slot mismatch that `BoundaryInjection` now reconciles
/// (and, for this exact case, still rejects, but with a different message) —
/// unaffected by wiring `reconcile` into `load_boundary_cuts`.
#[test]
fn full_fcf_manifest_check_unaffected_by_boundary_reconcile_wiring() {
    let source_slots = vec![storage_slot(7)];
    let current_slots = vec![storage_slot(42)];
    let empty_graph = GraphManifest::default();
    let source = PolicyStageManifest {
        state_dimension: 1,
        num_stages: 1,
        n_pools: 1,
        slots: &source_slots,
        graph: &empty_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: 1,
        num_stages: 1,
        n_pools: 1,
        slots: &current_slots,
        graph: &empty_graph,
    };

    let full_fcf_result = validate_policy_load::<FullFcf>(&source, &current);
    assert!(
        full_fcf_result.is_err(),
        "FullFcf's exact per-slot match must still hard-reject: {full_fcf_result:?}"
    );

    let boundary_result = validate_policy_load::<BoundaryInjection>(&source, &current);
    assert!(
        boundary_result.is_ok(),
        "BoundaryInjection defers slot identity to reconcile::build_rebind, not this check: \
         {boundary_result:?}"
    );
}

/// Given a current terminal manifest with a transit-bucket slot and a source
/// carrying none (the NEWAVE-shaped case), when the `BoundaryInjection` load
/// runs, then it succeeds and the transit coefficient defaults to `0.0`,
/// while the storage slot still loads its identity-matched coefficient.
#[test]
fn boundary_injection_transit_bucket_defaults_to_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), storage_slot(2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
    let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, None, 1_000_000.0, &mut |_| {})
        .expect("a transit-only-target load must succeed via the default-zero arm");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the transit slot must default to 0.0"
    );
}

/// Given a current terminal manifest with a sentinel-dated anticipated slot
/// and a source carrying no counterpart, when the `BoundaryInjection` load
/// runs, then it succeeds and the anticipated coefficient defaults to `0.0`,
/// while the storage slot still loads its identity-matched coefficient.
#[test]
fn boundary_injection_sentinel_anticipated_defaults_to_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), storage_slot(2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), sentinel_anticipated_slot(9, 0)];
    let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, None, 1_000_000.0, &mut |_| {})
        .expect("a sentinel-anticipated-target load must succeed via the default-zero arm");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the sentinel-dated anticipated slot must default to 0.0"
    );
}

/// Given a source already shaped identically to the current terminal
/// manifest (storage, inflow-lag, and a sentinel-dated anticipated slot),
/// when the `BoundaryInjection` load runs, then the injected coefficients
/// equal the source cut's own coefficients bit-for-bit (`f64::to_bits`) — the
/// superset property: reconciling a target-shaped source never regresses
/// today's exact-match load.
#[test]
fn boundary_injection_target_shaped_source_reconciles_bit_identically() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        inflow_lag_slot(1, 1),
        sentinel_anticipated_slot(9, 0),
    ];
    let coefficients = vec![10.5, -3.25, 0.0];
    write_checkpoint(tmp.path(), &manifest, &coefficients);

    let current = manifest.clone();
    let cuts = load_boundary_cuts(tmp.path(), 0, 3, &current, None, 1_000_000.0, &mut |_| {})
        .expect("a target-shaped source must load");

    assert_eq!(cuts.len(), 1);
    for (actual, expected) in cuts[0].coefficients.iter().zip(coefficients.iter()) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "reconciling a target-shaped source must reproduce its coefficients bit-for-bit: \
             {actual} != {expected}"
        );
    }
}

/// Given a current terminal manifest with a dated (non-sentinel) anticipated
/// slot — the shape a post-horizon commitment lane, dated at every terminal
/// stage, produces — when the `BoundaryInjection` load runs, then it returns
/// a clean `Err`, never a panic.
#[test]
fn boundary_injection_dated_anticipated_target_rejects_gracefully() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), storage_slot(2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), dated_anticipated_slot(9, 0, 20_260_301)];
    let result = load_boundary_cuts(tmp.path(), 0, 2, &current, None, 1_000_000.0, &mut |_| {});

    let msg = result
        .expect_err("a dated anticipated target slot must reject cleanly, never panic")
        .to_string();
    assert!(
        msg.contains("post-horizon anticipated date reconciliation is not yet supported"),
        "error must name the not-yet-supported dated fan-out reconciliation: {msg}"
    );
}

/// Given a current terminal manifest with a transit-bucket slot and a source
/// carrying a matching transit-bucket slot at the same identity (a
/// Cobre-to-Cobre boundary with matching transit arcs), when the
/// `BoundaryInjection` load runs, then the transit coefficient is copied from
/// the source verbatim, never zeroed — distinct from
/// `boundary_injection_transit_bucket_defaults_to_zero`'s no-source-transit
/// (NEWAVE) case, which still defaults to `0.0` on a miss.
#[test]
fn boundary_injection_transit_bucket_copies_on_identity_match() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), transit_bucket_slot(2, 1)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 30.0]);

    let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
    let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, None, 1_000_000.0, &mut |_| {})
        .expect("a matching transit-bucket boundary must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 30.0],
        "the transit slot must copy its identity-matched source coefficient, not zero it"
    );
}
