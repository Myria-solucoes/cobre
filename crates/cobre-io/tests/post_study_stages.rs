//! Integration tests for the standalone `post_study_stages.json` boundary input,
//! the sole post-horizon surface: the `System` flow (present → `Some`, absent →
//! `None`, additive), the semantic rejection paths (contiguity, first-start,
//! Rule 1 missing bound cell, Rule 2 E2 no-carrier, lead-exceeds-horizon), and
//! declaration-order invariance — all through the public `cobre_io::load_case`
//! pipeline. Post-study deliveries are driven by an anticipated thermal whose
//! `LeadTime` reaches the post-study calendar, not by a declared window.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use cobre_io::load_case;
use tempfile::TempDir;

mod helpers;
use helpers::{make_minimal_case, write_file};

/// Write a `system/thermals.json` declaring thermal 86 as an anticipated
/// `LeadTime` plant on bus 1.
fn write_anticipated_thermal_86(root: &Path, lead_time_hours: f64) {
    write_file(
        root,
        "system/thermals.json",
        &format!(
            r#"{{
          "thermals": [
            {{
                "id": 86,
                "name": "T_ANT",
                "operational_start_date": "2024-01-01",
                "bus_id": 1,
                "cost_per_mwh": 10.0,
                "generation": {{ "min_mw": 0.0, "max_mw": 300.0 }},
                "anticipated_config": {{ "lead_time_hours": {lead_time_hours} }}
            }}
          ]
        }}"#
        ),
    );
}

/// Write an `initial_conditions.json` whose single `past_anticipated_commitments`
/// window tiles the leading study stage (id 0, `[2024-01-01, 2024-02-01)`) at
/// 0 MW — the bijection every anticipated thermal needs.
fn write_ic_with_past_commitment(root: &Path) {
    write_file(
        root,
        "initial_conditions.json",
        r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 86, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }
          ]
        }"#,
    );
}

// ── present → System.post_study_stages Some, both collections sorted ──────

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

// ── absent → None, System is otherwise unchanged (additive/inert) ─────────

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

// ── first post-study start != study end → Err naming the mismatch ─────────

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

// ── non-contiguous post-study stages → Err ────────────────────────────────

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

// ── anticipated lead exceeds the horizon with no post_study → Err ─────────

#[test]
fn test_lead_exceeds_horizon_without_post_study_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 1440h (60 days) exceeds the single 744h study stage; with no
    // post_study_stages.json declared, the plant can never deliver.
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_past_commitment(dir.path());
    // No post_study_stages.json written.

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("can never deliver within the study horizon") && msg.contains("Thermal 86"),
        "error should name the lead-exceeds-horizon rejection, got: {msg}"
    );
}

// ── lead reaches a post-study stage with no bound cell → Err (Rule 1) ──────

#[test]
fn test_missing_post_study_bound_cell_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 744h: decided at study stage 0, delivered into post-study stage 0.
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_past_commitment(dir.path());
    // Post-study stage 0 is reached, but no bound cell is declared for it.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("no thermal_bounds entry")
            && msg.contains("thermal_id 86")
            && msg.contains("post_study_stage_index 0"),
        "error should name the missing (thermal, stage) bound cell, got: {msg}"
    );
}

// ── a pre-study-decided post-study delivery has no carrier → Err (Rule 2/E2) ─

#[test]
fn test_pre_study_decided_post_study_delivery_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 1440h: post-study stage 0 (Feb) is decided pre-study (no carrier),
    // while post-study stage 1 (Mar) is decided in-study, so the lead-cap is
    // satisfied and the E2 rejection is isolated to stage 0.
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_past_commitment(dir.path());
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("has no carrier") && msg.contains("Thermal 86"),
        "error should name the pre-study-decided post-study delivery, got: {msg}"
    );
}

// ── A valid reached delivery loads cleanly ────────────────────────────────────

#[test]
fn test_valid_covered_delivery_loads() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_past_commitment(dir.path());
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
        .unwrap_or_else(|e| panic!("a reached, bounded delivery must load, got: {e}"));
    assert!(system.post_study_stages().is_some());
}

// ── declaration-order invariance through System ───────────────────────────

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
