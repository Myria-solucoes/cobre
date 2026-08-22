//! Integration coverage for the `BoundaryInjection` load path's per-family
//! identity reconciliation: storage and inflow-lag reconcile by identity and
//! REJECT on a missing core slot; a `FullFcf`-typed manifest check stays on
//! the exact-match path, unaffected by wiring `reconcile` into
//! `load_boundary_cuts`.
//!
//! Also covers transit-bucket (identity match copies, a miss defaults to
//! `0.0`), sentinel-dated-anticipated default-`0.0`, and the dated-anticipated
//! date-driven fan-out (`÷H_M` `Blend`, coverage-`Renormalize`) — including a
//! ring-sourced post-study slot reconciling bit-identically to the
//! pre-switchover shape, and an in-study ring slot (no resolved interval)
//! defaulting to `Zero`.

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

use chrono::NaiveDate;
use cobre_io::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest, ManifestEdge,
    ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock, StageCutsPayload,
    StateFamily, write_policy_checkpoint,
};
use cobre_sddp::{
    BoundaryInjection, FullFcf, PolicyStageManifest, load_boundary_cuts, validate_policy_load,
};

fn storage_slot(id: i32) -> EntitySlot {
    EntitySlot {
        entity_type: StateFamily::HydroStorage.code(),
        entity_id: id,
        subindex: 0,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
    EntitySlot {
        entity_type: StateFamily::HydroInflowLag.code(),
        entity_id: id,
        subindex: lag_depth,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn transit_bucket_slot(downstream_hydro_id: i32, lag: u32) -> EntitySlot {
    EntitySlot {
        entity_type: StateFamily::HydroTransitBucket.code(),
        entity_id: downstream_hydro_id,
        subindex: lag,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn sentinel_anticipated_slot(thermal_id: i32, ring_slot: u32) -> EntitySlot {
    EntitySlot {
        entity_type: StateFamily::AnticipatedThermalState.code(),
        entity_id: thermal_id,
        subindex: ring_slot,
        was_active: true,
        delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
    }
}

fn dated_anticipated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
    EntitySlot {
        entity_type: StateFamily::AnticipatedThermalState.code(),
        entity_id: thermal_id,
        subindex: ring_slot,
        was_active: true,
        delivery_date,
    }
}

/// All-`None` delivery intervals aligned to `len` — for tests whose
/// manifests carry no live-dated anticipated slot.
fn no_intervals(len: usize) -> Vec<Option<(NaiveDate, NaiveDate)>> {
    vec![None; len]
}

fn ymd(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("valid calendar date")
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
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
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
    let result = load_boundary_cuts(
        tmp.path(),
        0,
        1,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    );

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
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a transit-only-target load must succeed via the default-zero arm");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the transit slot must default to 0.0"
    );
}

/// Given a boundary SOURCE that prices a transit-bucket arc the current study's
/// topology does not model (same state dimension, mismatched bucket identity),
/// when the `BoundaryInjection` load runs, then the source coupling is dropped
/// (its coefficient discarded) BUT the load succeeds and emits a warning naming
/// the dropped family and slot — the previously-silent C17 source-drop is now
/// visible. The target's own transit slot still defaults to `0.0`.
#[test]
fn boundary_injection_dropped_source_transit_coupling_warns_and_loads() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), transit_bucket_slot(2, 1)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), transit_bucket_slot(9, 1)];
    let mut warnings: Vec<String> = Vec::new();
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |w| warnings.push(w.to_string()),
    )
    .expect("a dropped source coupling must warn, never reject");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the target's own unmatched transit slot still defaults to 0.0"
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("transit-bucket") && w.contains("id 2")),
        "the dropped source transit-bucket coupling must be surfaced: {warnings:?}"
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
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
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
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        3,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
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
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a matching transit-bucket boundary must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 30.0],
        "the transit slot must copy its identity-matched source coefficient, not zero it"
    );
}

/// One source month (April 2026, `π_M = 300.0`) and the four `[start, end)`
/// week-lane intervals that exactly tile it — shared by the fan-out matrix and
/// interior-weeks conservation tests below.
fn april_2026_weekly_intervals() -> [(NaiveDate, NaiveDate); 4] {
    [
        (ymd(2026, 4, 1), ymd(2026, 4, 8)),
        (ymd(2026, 4, 8), ymd(2026, 4, 15)),
        (ymd(2026, 4, 15), ymd(2026, 4, 22)),
        (ymd(2026, 4, 22), ymd(2026, 5, 1)),
    ]
}

/// Given a source manifest with one monthly anticipated slot for thermal `9`
/// (`π_M = 300.0`, April 2026) and a target manifest with 4 post-horizon lane
/// slots for thermal `9` whose `[start, end)` intervals tile that month
/// exactly, when `load_boundary_cuts` runs end-to-end, then each lane
/// coefficient lands at `π_M · overlap/H_M` — the `÷H_M` distribute.
#[test]
fn boundary_injection_dated_anticipated_fan_out_matrix() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 0, 20_260_401),
        sentinel_anticipated_slot(9, 1),
        sentinel_anticipated_slot(9, 2),
        sentinel_anticipated_slot(9, 3),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 100, 20_260_401),
        dated_anticipated_slot(9, 101, 20_260_401),
        dated_anticipated_slot(9, 102, 20_260_401),
        dated_anticipated_slot(9, 103, 20_260_401),
    ];
    let intervals: Vec<Option<(NaiveDate, NaiveDate)>> = std::iter::once(None)
        .chain(weeks.iter().map(|&w| Some(w)))
        .collect();

    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        5,
        &current,
        &intervals,
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a fully-covered dated anticipated fan-out must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients[0], 10.0,
        "the storage slot still copies its identity-matched coefficient"
    );

    let h_m = 30.0 * 24.0;
    let expected_widths_days = [7.0, 7.0, 7.0, 9.0];
    for (lane, &width_days) in expected_widths_days.iter().enumerate() {
        let expected = 300.0 * (width_days * 24.0) / h_m;
        let actual = cuts[0].coefficients[lane + 1];
        assert!(
            (actual - expected).abs() < expected.abs() * 1e-9,
            "lane {lane}: coefficient {actual} != expected {expected} (π_M · overlap/H_M)"
        );
    }
}

/// Given the same one-source-month → 4-target-lane fan-out as
/// `boundary_injection_dated_anticipated_fan_out_matrix`, when the fanned lane
/// coefficients are summed, then the sum reproduces the source's own monthly
/// valuation `π_M` — the interior-weeks conservation invariant: a
/// constant per-week commitment `K` sees `Σ_w β_w · K = π_M · K` for any `K`,
/// so the coefficient-level identity `Σ_w β_w == π_M` holds with `K` divided
/// out.
#[test]
fn boundary_injection_dated_anticipated_interior_weeks_conservation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        dated_anticipated_slot(9, 0, 20_260_401),
        sentinel_anticipated_slot(9, 1),
        sentinel_anticipated_slot(9, 2),
        sentinel_anticipated_slot(9, 3),
    ];
    let source_coeff = 300.0;
    write_checkpoint(tmp.path(), &source_manifest, &[source_coeff, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        dated_anticipated_slot(9, 100, 20_260_401),
        dated_anticipated_slot(9, 101, 20_260_401),
        dated_anticipated_slot(9, 102, 20_260_401),
        dated_anticipated_slot(9, 103, 20_260_401),
    ];
    let intervals: Vec<Option<(NaiveDate, NaiveDate)>> = weeks.iter().map(|&w| Some(w)).collect();

    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        4,
        &current,
        &intervals,
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a fully-covered dated anticipated fan-out must load");

    assert_eq!(cuts.len(), 1);
    let summed: f64 = cuts[0].coefficients.iter().sum();
    assert!(
        (summed - source_coeff).abs() < source_coeff.abs() * 1e-9,
        "Σ_w β_w = {summed} must reproduce the source monthly valuation π_M = {source_coeff}"
    );
}

/// Given the same one-source-month → 4-target-lane fan-out as
/// `boundary_injection_dated_anticipated_fan_out_matrix`, when `.report()` is
/// read, then the anticipated family's `fan_out`/`straddling`/`default_zero`
/// match the fully-covered fan-out and the coverage line renders the
/// expected shape.
#[test]
fn boundary_injection_report_fan_out_matrix_coverage() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 0, 20_260_401),
        sentinel_anticipated_slot(9, 1),
        sentinel_anticipated_slot(9, 2),
        sentinel_anticipated_slot(9, 3),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 100, 20_260_401),
        dated_anticipated_slot(9, 101, 20_260_401),
        dated_anticipated_slot(9, 102, 20_260_401),
        dated_anticipated_slot(9, 103, 20_260_401),
    ];
    let intervals: Vec<Option<(NaiveDate, NaiveDate)>> = std::iter::once(None)
        .chain(weeks.iter().map(|&w| Some(w)))
        .collect();

    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        5,
        &current,
        &intervals,
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a fully-covered dated anticipated fan-out must load");

    let report = cuts.report();
    assert!(report.reconciled);
    assert_eq!(
        report.anticipated.fan_out, 4,
        "4 fully-covered target lanes"
    );
    assert_eq!(report.anticipated.straddling, 0);
    assert_eq!(report.anticipated.default_zero, 0);
    assert_eq!(
        report.anticipated_coverage.source_month_count, 1,
        "one source month (April 2026)"
    );

    let lines = report.detail_lines();
    assert!(
        lines.iter().any(|l| l
            == "anticipated: 1 source months fanned to 4 target slots (0 straddling, \
                overlap-blended), 0 months defaulted"),
        "coverage line must match the expected shape: {lines:?}"
    );
}

/// Given a target-shaped source (storage + lag + sentinel anticipated) that
/// loads bit-identically, when `.report()` is read, then every family
/// reports only `copy`, `fan_out == 0` everywhere — matching the superset
/// guarantee.
#[test]
fn boundary_injection_report_target_shaped_superset_is_copy_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        inflow_lag_slot(1, 1),
        sentinel_anticipated_slot(9, 0),
    ];
    let coefficients = vec![10.5, -3.25, 0.0];
    write_checkpoint(tmp.path(), &manifest, &coefficients);

    let current = manifest.clone();
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        3,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a target-shaped source must load");

    let report = cuts.report();
    assert!(report.reconciled);
    assert_eq!(report.storage.copy, 1);
    assert_eq!(report.inflow_lag.copy, 1);
    assert_eq!(
        report.anticipated.copy, 0,
        "sentinel Zero is excluded, not copy"
    );
    assert_eq!(report.anticipated.fan_out, 0);
    assert_eq!(report.transit_bucket.fan_out, 0);
    assert_eq!(report.other_identity.fan_out, 0);
}

/// Given a boundary checkpoint with an empty manifest, when `.report()` is
/// read, then `reconciled == false` and the render states a dimension-only
/// load.
#[test]
fn boundary_injection_report_empty_manifest_is_unreconciled() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_checkpoint(tmp.path(), &[], &[10.0, 20.0]);

    let current = vec![storage_slot(1), storage_slot(2)];
    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &no_intervals(current.len()),
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("an absent manifest must still load cuts");

    let report = cuts.report();
    assert!(!report.reconciled);
    assert!(
        report.summary_line().contains("dimension-only"),
        "must state a dimension-only load: {}",
        report.summary_line()
    );
    assert!(report.detail_lines().is_empty());
}

/// Given a monthly-priced source (April 2026, `π_M = 300.0`) and a current
/// terminal manifest whose ring-residue `subindex` values (100/101) tag a
/// post-study-targeted provenance, when `load_boundary_cuts` fans one
/// fully-covered lane and one lane straddling into unpriced (post-study) time,
/// then the coefficients equal the `Blend` (`π_M · overlap/H_M`, the same value
/// `boundary_injection_dated_anticipated_fan_out_matrix` asserts for a 7-day
/// lane) and `Renormalize` (`π_M · H_w/H_M`) values — reconcile keys on
/// `delivery_date`, never the ring `subindex`, so ring provenance reconciles
/// identically to the retired lane shape.
#[test]
fn boundary_injection_ring_sourced_post_study_fan_out_matches_pre_switchover() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 0, 20_260_401),
        sentinel_anticipated_slot(9, 1),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0]);

    let current = vec![
        storage_slot(1),
        dated_anticipated_slot(9, 100, 20_260_401),
        dated_anticipated_slot(9, 101, 20_260_401),
    ];
    let intervals = vec![
        None,
        Some((ymd(2026, 4, 1), ymd(2026, 4, 8))),
        Some((ymd(2026, 4, 22), ymd(2026, 5, 8))),
    ];

    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        3,
        &current,
        &intervals,
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("a ring-sourced post-study dated fan-out must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients[0], 10.0,
        "the storage slot still copies its identity-matched coefficient"
    );

    let h_m = 30.0 * 24.0;
    let expected_blend = 300.0 * (7.0 * 24.0) / h_m;
    assert!(
        (cuts[0].coefficients[1] - expected_blend).abs() < expected_blend.abs() * 1e-9,
        "fully-covered ring lane must Blend to π_M · overlap/H_M = {expected_blend}, got {}",
        cuts[0].coefficients[1]
    );
    let expected_renorm = 300.0 * (16.0 * 24.0) / h_m;
    assert!(
        (cuts[0].coefficients[2] - expected_renorm).abs() < expected_renorm.abs() * 1e-9,
        "straddling ring lane must Renormalize to π_M · H_w/H_M = {expected_renorm}, got {}",
        cuts[0].coefficients[2]
    );
}

/// Given a current terminal manifest carrying a dated ring slot whose
/// `delivery_date` resolves NO interval — an in-study ring slot (a
/// within-horizon delivery, e.g. a `K = 0` self-delivery) — when
/// `load_boundary_cuts` runs, then that slot reconciles to `0.0`, not a reject
/// and not a blend, even though the source carries a month whose date would
/// overlap it if it were wrongly fanned out. The integration mirror of the unit
/// pin `build_rebind_dated_in_study_ring_slot_with_no_interval_yields_zero`.
#[test]
fn boundary_injection_dated_in_study_ring_slot_with_no_interval_yields_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![storage_slot(1), dated_anticipated_slot(9, 0, 20_260_401)];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0]);

    let current = vec![storage_slot(1), dated_anticipated_slot(9, 0, 20_260_401)];
    let intervals = vec![None, None];

    let cuts = load_boundary_cuts(
        tmp.path(),
        0,
        2,
        &current,
        &intervals,
        &[],
        None,
        1_000_000.0,
        &mut |_| {},
    )
    .expect("an in-study ring slot with no interval must load, defaulting to 0.0");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "a dated ring slot with no resolved interval is an in-study delivery: it defaults \
         to 0.0, never fans out against the overlapping source month"
    );
}
