//! Integration tests for the standalone `post_study_stages.json` boundary input:
//! the `System` flow (present → `Some`, absent → `None`, additive), the semantic
//! rejection paths (contiguity, first-start, coverage, missing bound, empty
//! intersection), and declaration-order invariance — all through the public
//! `cobre_io::load_case` pipeline.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cobre_io::load_case;
use tempfile::TempDir;

mod helpers;
use helpers::{make_minimal_case, write_file};

/// `initial_conditions.json` carrying one `future_anticipated_deliveries` window.
fn initial_conditions_with_delivery(
    delivery_start: &str,
    delivery_end: &str,
    min: f64,
    max: f64,
) -> String {
    format!(
        r#"{{
          "storage": [],
          "filling_storage": [],
          "future_anticipated_deliveries": [
            {{ "thermal_id": 86, "delivery_start": "{delivery_start}", "delivery_end": "{delivery_end}", "min_mw": {min}, "max_mw": {max} }}
          ]
        }}"#
    )
}

// ── AC: present → System.post_study_stages Some, both collections sorted ──────

#[test]
fn test_present_carries_sorted_post_study_stages_onto_system() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // Stages and bounds declared out of canonical order; the reader sorts both.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-02-01", "duration_hours": 696.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System), got: {e}"));
    let ps = system
        .post_study_stages()
        .expect("post_study_stages must be Some when the file is present");

    assert_eq!(ps.stages.len(), 2);
    assert_eq!(
        ps.stages[0].start_date,
        chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        "stages must be ascending by start_date"
    );
    assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
    assert_eq!(ps.thermal_bounds[1].post_study_stage_index, 1);
}

// ── AC: absent → None, System is otherwise unchanged (additive/inert) ─────────

#[test]
fn test_absent_yields_none_and_is_additive() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System), got: {e}"));
    assert!(
        system.post_study_stages().is_none(),
        "absent post_study_stages.json must leave System.post_study_stages None"
    );
    // The rest of the System is unaffected — the input is inert when absent.
    assert_eq!(system.n_buses(), 1);
    assert_eq!(system.n_stages(), 1);
}

// ── AC: first post-study start != study end → Err naming the mismatch ─────────

#[test]
fn test_first_start_not_study_end_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-03-01", "duration_hours": 744.0 } ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("study horizon end") && msg.contains("2024-02-01"),
        "error should name the first-start / study-end mismatch, got: {msg}"
    );
}

// ── AC: non-contiguous post-study stages → Err ────────────────────────────────

#[test]
fn test_non_contiguous_stages_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // stage 0 ends 2024-03-01 (Feb has 29 days in 2024), but stage 1 starts
    // 2024-04-01 — a one-month gap.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("not date-contiguous"),
        "error should name the gap, got: {msg}"
    );
}

// ── AC: future_anticipated_deliveries but no post_study_stages → Err ──────────

#[test]
fn test_delivery_without_post_study_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "initial_conditions.json",
        &initial_conditions_with_delivery("2024-02-01", "2024-03-01", 0.0, 350.0),
    );
    // No post_study_stages.json written.

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("no post_study_stages.json") && msg.contains("Thermal 86"),
        "error should name the unanchored delivery, got: {msg}"
    );
}

// ── AC: delivery window not covered exactly (over-reach) → Err ────────────────

#[test]
fn test_delivery_overreaches_post_study_horizon_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "initial_conditions.json",
        &initial_conditions_with_delivery("2024-02-01", "2024-04-01", 0.0, 350.0),
    );
    // Only one post-study stage [2024-02-01, 2024-03-01): the delivery reaches past it.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("outside the post-study horizon") && msg.contains("2024-02-01"),
        "error should name the over-reaching delivery, got: {msg}"
    );
}

// ── AC: covered stage has no PostStudyThermalBound{t, j} → Err naming (t, j) ──

#[test]
fn test_missing_bound_for_covered_stage_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "initial_conditions.json",
        &initial_conditions_with_delivery("2024-02-01", "2024-03-01", 0.0, 350.0),
    );
    // Delivery covers stage index 0, but the only bound is for index 1.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("no thermal_bounds entry")
            && msg.contains("thermal_id 86")
            && msg.contains("post_study_stage_index 0"),
        "error should name the missing (thermal, stage) cell, got: {msg}"
    );
}

// ── AC: empty commitment∩capability intersection → Err ───────────────────────

#[test]
fn test_empty_commitment_capability_intersection_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // Commitment interval [400, 500] cannot intersect capability [0, 350].
    write_file(
        dir.path(),
        "initial_conditions.json",
        &initial_conditions_with_delivery("2024-02-01", "2024-03-01", 400.0, 500.0),
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("do not intersect") && msg.contains("Thermal 86"),
        "error should name the unsatisfiable commitment, got: {msg}"
    );
}

// ── A valid covered delivery loads cleanly ────────────────────────────────────

#[test]
fn test_valid_covered_delivery_loads() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "initial_conditions.json",
        &initial_conditions_with_delivery("2024-02-01", "2024-03-01", 0.0, 300.0),
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("a covered, intersecting delivery must load, got: {e}"));
    assert!(system.post_study_stages().is_some());
}

// ── AC: declaration-order invariance through System ───────────────────────────

#[test]
fn test_declaration_order_invariance_through_system() {
    let forward = r#"{
      "stages": [
        { "start_date": "2024-02-01", "duration_hours": 696.0 },
        { "start_date": "2024-03-01", "duration_hours": 744.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 },
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 }
      ]
    }"#;
    let reversed = r#"{
      "stages": [
        { "start_date": "2024-03-01", "duration_hours": 744.0 },
        { "start_date": "2024-02-01", "duration_hours": 696.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;

    let dir1 = TempDir::new().unwrap();
    make_minimal_case(&dir1);
    write_file(dir1.path(), "post_study_stages.json", forward);

    let dir2 = TempDir::new().unwrap();
    make_minimal_case(&dir2);
    write_file(dir2.path(), "post_study_stages.json", reversed);

    let s1 = load_case(dir1.path()).unwrap();
    let s2 = load_case(dir2.path()).unwrap();
    assert_eq!(
        s1.post_study_stages(),
        s2.post_study_stages(),
        "post_study_stages must be identical regardless of declaration order"
    );
}
