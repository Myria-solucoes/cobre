//! Integration tests for the `future_anticipated_deliveries` windowed input —
//! the right-boundary counterpart to `past_anticipated_commitments`.

#![allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]

use chrono::NaiveDate;
use cobre_core::EntityId;
use cobre_io::LoadError;
use cobre_io::parse_initial_conditions;
use std::io::Write;
use tempfile::NamedTempFile;

/// Write a string to a temp file and return the file handle (keeps it alive).
fn write_json(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ── array present → parsed with correct dates and [min_mw, max_mw] ────────

#[test]
fn test_future_anticipated_deliveries_present_parses_correctly() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 86, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;
    let f = write_json(json);
    let ic = parse_initial_conditions(f.path()).unwrap();

    assert_eq!(ic.future_anticipated_deliveries.len(), 1);
    let entry = &ic.future_anticipated_deliveries[0];
    assert_eq!(entry.thermal_id, EntityId(86));
    assert_eq!(entry.delivery_start, date(2026, 9, 5));
    assert_eq!(entry.delivery_end, date(2026, 9, 12));
    assert_eq!(entry.min_mw, 0.0);
    assert_eq!(entry.max_mw, 350.0);
}

// ── no future_anticipated_deliveries key → empty vector, no error ─────────

#[test]
fn test_future_anticipated_deliveries_absent_defaults_to_empty() {
    let json = r#"{ "storage": [], "filling_storage": [] }"#;
    let f = write_json(json);
    let ic = parse_initial_conditions(f.path()).unwrap();
    assert!(
        ic.future_anticipated_deliveries.is_empty(),
        "absent future_anticipated_deliveries must default to empty vec"
    );
}

// ── max_mw < min_mw → Err naming (thermal_id, delivery_start) ─────────────

#[test]
fn test_future_anticipated_deliveries_max_less_than_min_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 86, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 350.0, "max_mw": 0.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { field, message, .. } => {
            assert!(
                field.contains("future_anticipated_deliveries["),
                "got: {field}"
            );
            assert!(
                message.contains("thermal_id 86"),
                "message should name thermal_id, got: {message}"
            );
            assert!(
                message.contains("2026-09-05"),
                "message should name delivery_start, got: {message}"
            );
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

// ── delivery_end <= delivery_start → Err ───────────────────────────────────

#[test]
fn test_future_anticipated_deliveries_end_before_start_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 7, "delivery_start": "2026-09-12", "delivery_end": "2026-09-05", "min_mw": 0.0, "max_mw": 100.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { message, .. } => {
            assert!(
                message.contains("end_date must be after start_date"),
                "got: {message}"
            );
            assert!(
                message.contains("thermal_id 7"),
                "message should name thermal_id, got: {message}"
            );
            assert!(
                message.contains("2026-09-12"),
                "message should name delivery_start, got: {message}"
            );
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

#[test]
fn test_future_anticipated_deliveries_end_equals_start_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 7, "delivery_start": "2026-09-05", "delivery_end": "2026-09-05", "min_mw": 0.0, "max_mw": 100.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { message, .. } => {
            assert!(
                message.contains("end_date must be after start_date"),
                "got: {message}"
            );
            assert!(
                message.contains("thermal_id 7"),
                "message should name thermal_id, got: {message}"
            );
            assert!(
                message.contains("2026-09-05"),
                "message should name delivery_start, got: {message}"
            );
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

// ── bad date format → Err naming the offending field ──────────────────────

#[test]
fn test_future_anticipated_deliveries_invalid_delivery_start_format_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 86, "delivery_start": "2026/09/05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { field, .. } => {
            assert!(field.contains("delivery_start"), "got: {field}");
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

#[test]
fn test_future_anticipated_deliveries_invalid_delivery_end_format_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 86, "delivery_start": "2026-09-05", "delivery_end": "not-a-date", "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { field, .. } => {
            assert!(field.contains("delivery_end"), "got: {field}");
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

// ── duplicate (thermal_id, delivery_start) → Err ───────────────────────────

#[test]
fn test_future_anticipated_deliveries_duplicate_key_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 5, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 5, "delivery_start": "2026-09-05", "delivery_end": "2026-09-19", "min_mw": 0.0, "max_mw": 200.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { field, message, .. } => {
            assert!(
                field.contains("future_anticipated_deliveries["),
                "got: {field}"
            );
            assert!(message.contains("duplicate"), "got: {message}");
            assert!(message.contains("thermal_id 5"), "got: {message}");
            assert!(message.contains("2026-09-05"), "got: {message}");
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

// ── two overlapping windows for the same thermal_id → Err naming overlap ──

#[test]
fn test_future_anticipated_deliveries_overlapping_windows_rejected() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 7, "delivery_start": "2026-09-05", "delivery_end": "2026-09-19", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 7, "delivery_start": "2026-09-12", "delivery_end": "2026-09-26", "min_mw": 0.0, "max_mw": 100.0 }
      ]
    }"#;
    let f = write_json(json);
    let err = parse_initial_conditions(f.path()).unwrap_err();
    match &err {
        LoadError::SchemaError { message, .. } => {
            assert!(message.contains("overlapping"), "got: {message}");
            assert!(message.contains("thermal_id 7"), "got: {message}");
        }
        other => panic!("expected SchemaError, got: {other:?}"),
    }
}

/// Adjacent (non-overlapping) windows for the same thermal are accepted —
/// `delivery_start == previous delivery_end` is the boundary, not a conflict.
#[test]
fn test_future_anticipated_deliveries_adjacent_windows_accepted() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 7, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 7, "delivery_start": "2026-09-12", "delivery_end": "2026-09-19", "min_mw": 0.0, "max_mw": 100.0 }
      ]
    }"#;
    let f = write_json(json);
    let result = parse_initial_conditions(f.path());
    assert!(
        result.is_ok(),
        "adjacent windows (start == prev_end) must be accepted, got: {result:?}"
    );
}

// ── min_mw == max_mw pins a fixed commitment ───────────────────────────────────

#[test]
fn test_future_anticipated_deliveries_min_equals_max_accepted() {
    let json = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 3, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 150.0, "max_mw": 150.0 }
      ]
    }"#;
    let f = write_json(json);
    let ic = parse_initial_conditions(f.path()).unwrap();
    assert_eq!(ic.future_anticipated_deliveries.len(), 1);
    assert_eq!(ic.future_anticipated_deliveries[0].min_mw, 150.0);
    assert_eq!(ic.future_anticipated_deliveries[0].max_mw, 150.0);
}

// ── declaration-order invariance ───────────────────────────────────────────

#[test]
fn test_future_anticipated_deliveries_declaration_order_invariance() {
    let json_forward = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 1, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 1, "delivery_start": "2026-09-12", "delivery_end": "2026-09-19", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 2, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 50.0 }
      ]
    }"#;
    let json_reversed = r#"{
      "storage": [],
      "filling_storage": [],
      "future_anticipated_deliveries": [
        { "thermal_id": 2, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 50.0 },
        { "thermal_id": 1, "delivery_start": "2026-09-12", "delivery_end": "2026-09-19", "min_mw": 0.0, "max_mw": 100.0 },
        { "thermal_id": 1, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 100.0 }
      ]
    }"#;
    let f1 = write_json(json_forward);
    let f2 = write_json(json_reversed);
    let ic1 = parse_initial_conditions(f1.path()).unwrap();
    let ic2 = parse_initial_conditions(f2.path()).unwrap();

    assert_eq!(
        ic1.future_anticipated_deliveries, ic2.future_anticipated_deliveries,
        "results must be identical regardless of input ordering"
    );
    assert_eq!(ic1.future_anticipated_deliveries[0].thermal_id, EntityId(1));
    assert_eq!(
        ic1.future_anticipated_deliveries[0].delivery_start,
        date(2026, 9, 5)
    );
    assert_eq!(ic1.future_anticipated_deliveries[1].thermal_id, EntityId(1));
    assert_eq!(
        ic1.future_anticipated_deliveries[1].delivery_start,
        date(2026, 9, 12)
    );
    assert_eq!(ic1.future_anticipated_deliveries[2].thermal_id, EntityId(2));
}
