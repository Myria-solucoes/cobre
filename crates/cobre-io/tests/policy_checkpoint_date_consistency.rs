//! Integration tests for the intra-artifact date-consistency validation that
//! `read_policy_checkpoint` runs after decode.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

use cobre_io::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    PolicyCheckpointMetadata, ProducerBlock, StageCutsPayload, StateFamily, read_policy_checkpoint,
    write_policy_checkpoint,
};

fn slot(entity_type: u8, entity_id: i32, subindex: u32, delivery_date: i32) -> EntitySlot {
    EntitySlot {
        entity_type,
        entity_id,
        subindex,
        was_active: true,
        delivery_date,
    }
}

fn metadata() -> PolicyCheckpointMetadata {
    PolicyCheckpointMetadata {
        format_version: FORMAT_VERSION,
        cobre_version: "0.13.0".to_string(),
        created_at: "2026-08-11T00:00:00Z".to_string(),
        num_stages: 1,
        graph_manifest: GraphManifest::default(),
        producer: ProducerBlock {
            completed_iterations: 1,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 1,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        },
    }
}

/// Write a one-pool checkpoint fixture whose only payload is `manifest`.
fn write_fixture(dir: &std::path::Path, pool_id: u32, manifest: &[EntitySlot]) {
    let payload = StageCutsPayload {
        stage_id: pool_id,
        state_dimension: manifest.len() as u32,
        capacity: 0,
        warm_start_count: 0,
        cuts: &[],
        active_cut_indices: &[],
        populated_count: 0,
        entity_manifest: manifest,
        cost_scale_factor: 1_000_000.0,
        node_id: -1,
        graph_stage_id: -1,
    };
    write_policy_checkpoint(dir, &[payload], &[], &metadata(), &[]).expect("fixture must write");
}

// ── a malformed delivery_date is rejected, naming the slot ──────────────

#[test]
fn malformed_month_delivery_date_rejected_naming_slot() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [slot(
        StateFamily::AnticipatedThermalState.code(),
        7,
        0,
        20_261_305,
    )];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path()).expect_err("month-13 date must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("subindex 0"), "must name the subindex: {msg}");
    assert!(
        msg.contains("20261305"),
        "must name the malformed date: {msg}"
    );
}

// ── decreasing HydroTransitBucket dates are rejected, naming pool+subindex ──

#[test]
fn hydro_transit_bucket_decreasing_dates_rejected_naming_pool_and_subindex() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        slot(StateFamily::HydroTransitBucket.code(), 42, 0, 20_260_601),
        slot(StateFamily::HydroTransitBucket.code(), 42, 1, 20_260_501),
    ];
    write_fixture(dir.path(), 3, &manifest);

    let err = read_policy_checkpoint(dir.path())
        .expect_err("decreasing HydroTransitBucket dates must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 3"), "must name the pool: {msg}");
    assert!(
        msg.contains("subindex 1"),
        "must name the offending subindex: {msg}"
    );
}

// ── a well-formed, monotone checkpoint reads Ok with an unchanged manifest ──

#[test]
fn well_formed_monotone_checkpoint_accepted_manifest_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        slot(
            StateFamily::HydroStorage.code(),
            1,
            0,
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        ),
        slot(StateFamily::HydroTransitBucket.code(), 42, 0, 20_260_501),
        slot(StateFamily::HydroTransitBucket.code(), 42, 1, 20_260_601),
    ];
    write_fixture(dir.path(), 1, &manifest);

    let checkpoint = read_policy_checkpoint(dir.path()).expect("well-formed checkpoint must read");
    assert_eq!(checkpoint.stage_cuts.len(), 1);
    let read_back = &checkpoint.stage_cuts[0].entity_manifest;
    assert_eq!(read_back.len(), manifest.len());
    for (got, want) in read_back.iter().zip(manifest.iter()) {
        assert_eq!(got.entity_type, want.entity_type);
        assert_eq!(got.entity_id, want.entity_id);
        assert_eq!(got.subindex, want.subindex);
        assert_eq!(got.delivery_date, want.delivery_date);
    }
}

// ── a fully sentinel-dated legacy checkpoint reads Ok ───────────────────

#[test]
fn fully_sentinel_legacy_checkpoint_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        slot(
            StateFamily::HydroStorage.code(),
            1,
            0,
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        ),
        slot(
            StateFamily::AnticipatedThermalState.code(),
            7,
            0,
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        ),
        slot(
            StateFamily::AnticipatedThermalState.code(),
            7,
            1,
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        ),
        slot(
            StateFamily::HydroTransitBucket.code(),
            42,
            0,
            ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        ),
    ];
    write_fixture(dir.path(), 0, &manifest);

    read_policy_checkpoint(dir.path())
        .expect("a fully sentinel-dated legacy checkpoint must be accepted");
}

// ── Anti-false-reject guard: a modular-residue subindex sequence is exempt ──

#[test]
fn anticipated_thermal_state_non_monotone_dates_accepted() {
    let dir = tempfile::tempdir().unwrap();
    // At a stage where `t mod k_max != 0`, a correctly-produced modular-residue
    // subindex sequence decreases (subindex 0 delivers later than subindex 1).
    // Rejecting this would be a false positive on real data — see
    // `StateFamily::HydroTransitBucket`'s doc in `checkpoint.rs`.
    let manifest = [
        slot(
            StateFamily::AnticipatedThermalState.code(),
            1,
            0,
            20_260_601,
        ),
        slot(
            StateFamily::AnticipatedThermalState.code(),
            1,
            1,
            20_260_501,
        ),
    ];
    write_fixture(dir.path(), 1, &manifest);

    read_policy_checkpoint(dir.path())
        .expect("a modular-residue subindex family must be exempt from ordering checks");
}
