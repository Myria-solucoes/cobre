//! Schema-conformance tests for `schemas/policy.fbs`, gated behind the
//! `flatc-conformance` feature (requires `flatc` on `$PATH` or via `FLATC`).
//!
//! The hand-rolled writer/reader encode field positions via the `*_FIELD_*: u16`
//! slot constants; the schema encodes them via `(id: N)` attributes. These two
//! views must agree exactly, or every schema consumer is corrupted. The tests
//! check both directions — writer→`flatc -t` (catches wrong/unknown slot writes)
//! and `flatc -b`→reader (catches a slot the reader expects at another offset);
//! changing one side without the other fails at least one check.
//!
//! Missing `flatc` panics rather than silently passing: invoking the feature is
//! an explicit request to run these checks.
//!
//! ```bash
//! cargo test -p cobre-io --features flatc-conformance --test flatbuffers_schema_conformance
//! ```

#![cfg(feature = "flatc-conformance")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreadable_literal
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cobre_io::{
    ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCutRecord, STAGE_STATES_NODE_ID_SENTINEL, StageCutsReadResult,
    StageStatesPayload, StageStatesReadResult, deserialize_stage_basis, deserialize_stage_cuts,
    deserialize_stage_states, serialize_stage_basis, serialize_stage_cuts, serialize_stage_states,
};
use serde_json::{Value, json};
use tempfile::TempDir;

/// # Panics
/// When `flatc` is not found on `$PATH` or via the `FLATC` env var.
fn flatc_command() -> Command {
    let exe = std::env::var_os("FLATC").unwrap_or_else(|| "flatc".into());
    let mut probe = Command::new(&exe);
    probe.arg("--version");
    let probe_output = probe.output().unwrap_or_else(|err| {
        panic!(
            "the `flatc-conformance` feature requires `flatc` on PATH (or via the FLATC env var); \
             tried `{}`: {err}",
            Path::new(&exe).display()
        )
    });
    assert!(
        probe_output.status.success(),
        "`flatc --version` failed; stderr = {}",
        String::from_utf8_lossy(&probe_output.stderr)
    );
    Command::new(exe)
}

fn schema_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/policy.fbs")
}

/// `--root-type` must be namespace-qualified: flatc rejects the unqualified
/// short name (e.g. `StageCuts`) with `unknown root type`.
fn qualified(root_type: &str) -> String {
    format!("Cobre.IO.Policy.{root_type}")
}

fn flatc_decode(buf: &[u8], root_type: &str) -> Value {
    let dir = TempDir::new().expect("create tempdir");
    let bin_path = dir.path().join("buf.bin");
    std::fs::write(&bin_path, buf).expect("write bin");

    let status = flatc_command()
        .arg("-t")
        .arg("--strict-json")
        .arg("--raw-binary")
        .arg("--root-type")
        .arg(qualified(root_type))
        .arg("-o")
        .arg(dir.path())
        .arg(schema_path())
        .arg("--")
        .arg(&bin_path)
        .status()
        .expect("run flatc -t");
    assert!(
        status.success(),
        "flatc -t failed for root {root_type}; buffer contains a slot or layout the schema does \
         not match"
    );

    let json_path = dir.path().join("buf.json");
    let json_bytes = std::fs::read(&json_path).expect("read flatc JSON");
    serde_json::from_slice(&json_bytes).expect("parse flatc JSON")
}

fn flatc_encode(json: &Value, root_type: &str) -> Vec<u8> {
    let dir = TempDir::new().expect("create tempdir");
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(json).unwrap()).expect("write JSON");

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified(root_type))
        .arg("-o")
        .arg(dir.path())
        .arg(schema_path())
        .arg(&json_path)
        .status()
        .expect("run flatc -b");
    assert!(
        status.success(),
        "flatc -b failed for root {root_type}; the JSON does not satisfy the schema"
    );

    let bin_path = dir.path().join("doc.bin");
    std::fs::read(&bin_path).expect("read flatc-produced bin")
}

fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("flatc JSON missing field `{key}`; got: {v}"))
}

fn as_u64(v: &Value, key: &str) -> u64 {
    get(v, key)
        .as_u64()
        .unwrap_or_else(|| panic!("field `{key}` is not a u64: {v}"))
}

fn as_f64(v: &Value, key: &str) -> f64 {
    get(v, key)
        .as_f64()
        .unwrap_or_else(|| panic!("field `{key}` is not a f64: {v}"))
}

fn as_bool(v: &Value, key: &str) -> bool {
    get(v, key)
        .as_bool()
        .unwrap_or_else(|| panic!("field `{key}` is not a bool: {v}"))
}

/// flatc `-t` omits scalar fields equal to their default, so manifest assertions
/// must treat an absent field as its default rather than panicking.
fn u64_or(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).map_or(default, |f| {
        f.as_u64()
            .unwrap_or_else(|| panic!("field `{key}` is not a u64: {v}"))
    })
}

fn i64_or(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).map_or(default, |f| {
        f.as_i64()
            .unwrap_or_else(|| panic!("field `{key}` is not an i64: {v}"))
    })
}

fn bool_or(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).map_or(default, |f| {
        f.as_bool()
            .unwrap_or_else(|| panic!("field `{key}` is not a bool: {v}"))
    })
}

/// flatc renders the `EntityType` enum by name; map the raw byte to the name the
/// schema declares so writer→flatc assertions can compare against it. An absent
/// `entity_type` (the `0`/`HydroStorage` default) is treated as `HydroStorage`.
fn entity_type_name(byte: u8) -> &'static str {
    match byte {
        0 => "HydroStorage",
        1 => "HydroInflowLag",
        2 => "AnticipatedThermalState",
        3 => "HydroTransitBucket",
        other => panic!("unexpected entity_type byte {other}"),
    }
}

fn assert_manifest_json_matches(manifest_json: &Value, expected: &[EntitySlot]) {
    let arr = manifest_json
        .as_array()
        .expect("entity_manifest must be an array");
    assert_eq!(arr.len(), expected.len(), "entity_manifest length");
    for (i, (obj, slot)) in arr.iter().zip(expected).enumerate() {
        let ty = obj
            .get("entity_type")
            .and_then(Value::as_str)
            .unwrap_or("HydroStorage");
        assert_eq!(
            ty,
            entity_type_name(slot.entity_type),
            "slot {i} entity_type"
        );
        assert_eq!(
            i64_or(obj, "entity_id", 0),
            i64::from(slot.entity_id),
            "slot {i} entity_id"
        );
        assert_eq!(
            u64_or(obj, "subindex", 0),
            u64::from(slot.subindex),
            "slot {i} subindex"
        );
        assert_eq!(
            bool_or(obj, "was_active", false),
            slot.was_active,
            "slot {i} was_active"
        );
        // flatc omits a scalar equal to its default (0); an absent field is 0.
        assert_eq!(
            i64_or(obj, "delivery_date", 0),
            i64::from(slot.delivery_date),
            "slot {i} delivery_date"
        );
    }
}

/// Manifest fixture exercising every `entity_type` byte and a `-1` id, so the
/// writer→flatc and flatc→reader paths cover every field including a signed id.
fn conformance_manifest() -> Vec<EntitySlot> {
    vec![
        EntitySlot {
            entity_type: 2,
            entity_id: 7,
            subindex: 1,
            was_active: true,
            delivery_date: 20240501,
        },
        EntitySlot {
            entity_type: 1,
            entity_id: -1,
            subindex: 3,
            was_active: false,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        },
        EntitySlot {
            entity_type: 3,
            entity_id: 42,
            subindex: 2,
            was_active: true,
            delivery_date: 20200101,
        },
    ]
}

// ─── StageCuts ───────────────────────────────────────────────────────────────

#[test]
fn stage_cuts_writer_matches_schema() {
    let coeffs_a = [1.0, 2.0, 3.0, 4.0];
    let coeffs_b = [-0.5, 0.25, 0.125, 0.0625];
    let cuts = [
        PolicyCutRecord {
            cut_id: 7,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 100.5,
            coefficients: &coeffs_a,
            is_active: true,
        },
        PolicyCutRecord {
            cut_id: 8,
            slot_index: 1,
            iteration: 1,
            forward_pass_index: 1,
            intercept: -42.0,
            coefficients: &coeffs_b,
            is_active: false,
        },
    ];
    let active = [0_u32];
    let manifest = conformance_manifest();
    let buf = serialize_stage_cuts(3, 4, 16, 0, &cuts, &active, 2, &manifest);

    let json = flatc_decode(&buf, "StageCuts");

    assert_eq!(as_u64(&json, "stage_id"), 3);
    assert_eq!(as_u64(&json, "state_dimension"), 4);
    assert_eq!(as_u64(&json, "capacity"), 16);
    assert_eq!(as_u64(&json, "warm_start_count"), 0);
    assert_eq!(as_u64(&json, "populated_count"), 2);
    assert_eq!(get(&json, "active_cut_indices"), &json!([0]));
    assert_manifest_json_matches(get(&json, "entity_manifest"), &manifest);

    let cuts_json = get(&json, "cuts").as_array().expect("cuts is an array");
    assert_eq!(cuts_json.len(), 2);

    let c0 = &cuts_json[0];
    assert_eq!(as_u64(c0, "piece_id"), 7);
    assert_eq!(as_u64(c0, "slot_index"), 0);
    assert_eq!(as_u64(c0, "iteration"), 1);
    assert_eq!(as_u64(c0, "forward_pass_index"), 0);
    assert!((as_f64(c0, "intercept") - 100.5).abs() < 1e-12);
    assert_eq!(get(c0, "coefficients"), &json!([1.0, 2.0, 3.0, 4.0]));
    assert!(as_bool(c0, "is_active"));

    let c1 = &cuts_json[1];
    assert_eq!(as_u64(c1, "piece_id"), 8);
    assert_eq!(as_u64(c1, "slot_index"), 1);
    assert!((as_f64(c1, "intercept") - (-42.0)).abs() < 1e-12);
    assert!(!as_bool(c1, "is_active"));
}

#[test]
fn stage_cuts_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 5,
        "state_dimension": 3,
        "capacity": 8,
        "warm_start_count": 2,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 99,
                "slot_index": 0,
                "iteration": 4,
                "forward_pass_index": 2,
                "intercept": 12.5,
                "coefficients": [0.1, 0.2, 0.3],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "AnticipatedThermalState", "entity_id": 7, "subindex": 1, "was_active": true},
            {"entity_type": "HydroInflowLag", "entity_id": -1, "subindex": 3, "was_active": false},
            {"entity_type": "HydroTransitBucket", "entity_id": 42, "subindex": 2, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageCuts");
    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 5);
    assert_eq!(result.state_dimension, 3);
    assert_eq!(result.capacity, 8);
    assert_eq!(result.warm_start_count, 2);
    assert_eq!(result.populated_count, 1);
    assert_eq!(result.cuts.len(), 1);

    let cut: &OwnedPolicyCutRecord = &result.cuts[0];
    assert_eq!(cut.cut_id, 99);
    assert_eq!(cut.slot_index, 0);
    assert_eq!(cut.iteration, 4);
    assert_eq!(cut.forward_pass_index, 2);
    assert!((cut.intercept - 12.5).abs() < 1e-12);
    assert_eq!(cut.coefficients, vec![0.1, 0.2, 0.3]);
    assert!(cut.is_active);

    assert_eq!(result.entity_manifest.len(), 3);
    let m0 = &result.entity_manifest[0];
    assert_eq!(m0.entity_type, 2, "AnticipatedThermalState byte");
    assert_eq!(m0.entity_id, 7);
    assert_eq!(m0.subindex, 1);
    assert!(m0.was_active);
    let m1 = &result.entity_manifest[1];
    assert_eq!(m1.entity_type, 1, "HydroInflowLag byte");
    assert_eq!(m1.entity_id, -1);
    assert_eq!(m1.subindex, 3);
    assert!(!m1.was_active);
    let m2 = &result.entity_manifest[2];
    assert_eq!(m2.entity_type, 3, "HydroTransitBucket byte");
    assert_eq!(m2.entity_id, 42);
    assert_eq!(m2.subindex, 2);
    assert!(m2.was_active);
}

/// Reduced-dimension manifest: storage-only (two `HydroStorage` slots, zero lag
/// slots), `state_dimension = 2` where the all-enabled analogue would be larger.
/// This is the reduced-per-stage case at the conformance level — a manifest
/// shorter than the all-enabled count must round-trip in both directions.
fn reduced_storage_manifest() -> Vec<EntitySlot> {
    vec![
        EntitySlot {
            entity_type: 0,
            entity_id: 1,
            subindex: 0,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        },
        EntitySlot {
            entity_type: 0,
            entity_id: 2,
            subindex: 0,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        },
    ]
}

#[test]
fn stage_cuts_reduced_dimension_writer_matches_schema() {
    let coeffs = [3.5, -1.25];
    let cuts = [PolicyCutRecord {
        cut_id: 42,
        slot_index: 0,
        iteration: 2,
        forward_pass_index: 0,
        intercept: 7.0,
        coefficients: &coeffs,
        is_active: true,
    }];
    let active = [0_u32];
    let manifest = reduced_storage_manifest();
    let buf = serialize_stage_cuts(1, 2, 8, 0, &cuts, &active, 1, &manifest);

    let json = flatc_decode(&buf, "StageCuts");

    assert_eq!(as_u64(&json, "state_dimension"), 2);
    assert_eq!(as_u64(&json, "populated_count"), 1);

    let manifest_json = get(&json, "entity_manifest");
    assert_eq!(
        manifest_json
            .as_array()
            .expect("entity_manifest must be an array")
            .len(),
        2,
        "reduced manifest must have 2 storage slots"
    );
    assert_manifest_json_matches(manifest_json, &manifest);

    let decoded_cut = &get(&json, "cuts").as_array().expect("cuts array")[0];
    assert_eq!(get(decoded_cut, "coefficients"), &json!([3.5, -1.25]));
}

#[test]
fn stage_cuts_reduced_dimension_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 8,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 42,
                "slot_index": 0,
                "iteration": 2,
                "forward_pass_index": 0,
                "intercept": 7.0,
                "coefficients": [3.5, -1.25],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true},
            {"entity_type": "HydroStorage", "entity_id": 2, "subindex": 0, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageCuts");
    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.state_dimension, 2);
    assert_eq!(result.entity_manifest.len(), 2);
    for (i, slot) in result.entity_manifest.iter().enumerate() {
        assert_eq!(slot.entity_type, 0, "slot {i} HydroStorage byte");
        assert_eq!(slot.subindex, 0, "slot {i} subindex");
        assert!(slot.was_active, "slot {i} was_active");
    }
    assert_eq!(result.entity_manifest[0].entity_id, 1);
    assert_eq!(result.entity_manifest[1].entity_id, 2);
}

// ─── StageBasis ──────────────────────────────────────────────────────────────

#[test]
fn stage_basis_writer_matches_schema() {
    let cols = [0_u8, 1, 2, 3];
    let rows = [4_u8, 5, 6, 0, 1];
    let record = PolicyBasisRecord {
        stage_id: 11,
        iteration: 23,
        column_status: &cols,
        row_status: &rows,
        num_cut_rows: 2,
    };
    let buf = serialize_stage_basis(&record);

    let json = flatc_decode(&buf, "StageBasis");

    assert_eq!(as_u64(&json, "stage_id"), 11);
    assert_eq!(as_u64(&json, "iteration"), 23);
    assert_eq!(as_u64(&json, "num_columns"), 4);
    assert_eq!(as_u64(&json, "num_rows"), 5);
    assert_eq!(as_u64(&json, "num_cut_rows"), 2);
    assert_eq!(get(&json, "column_status"), &json!([0, 1, 2, 3]));
    assert_eq!(get(&json, "row_status"), &json!([4, 5, 6, 0, 1]));
}

#[test]
fn stage_basis_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 7,
        "iteration": 3,
        "num_columns": 3,
        "num_rows": 4,
        "column_status": [1, 2, 3],
        "row_status": [0, 1, 1, 2],
        "num_cut_rows": 1
    });
    let buf = flatc_encode(&document, "StageBasis");
    let result: OwnedPolicyBasisRecord =
        deserialize_stage_basis(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 7);
    assert_eq!(result.iteration, 3);
    assert_eq!(result.column_status, vec![1, 2, 3]);
    assert_eq!(result.row_status, vec![0, 1, 1, 2]);
    assert_eq!(result.num_cut_rows, 1);
}

// ─── StageStates ─────────────────────────────────────────────────────────────

#[test]
fn stage_states_writer_matches_schema() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let manifest = conformance_manifest();
    let payload = StageStatesPayload {
        stage_id: 2,
        node_id: 5,
        state_dimension: 3,
        count: 2,
        data: &data,
        entity_manifest: &manifest,
    };
    let buf = serialize_stage_states(&payload);

    let json = flatc_decode(&buf, "StageStates");

    assert_eq!(as_u64(&json, "stage_id"), 2);
    assert_eq!(as_u64(&json, "state_dimension"), 3);
    assert_eq!(as_u64(&json, "count"), 2);
    assert_eq!(get(&json, "data"), &json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_manifest_json_matches(get(&json, "entity_manifest"), &manifest);
    assert_eq!(i64_or(&json, "node_id", -1), 5);
}

#[test]
fn stage_states_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 9,
        "node_id": 12,
        "state_dimension": 2,
        "count": 3,
        "data": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 4, "subindex": 0, "was_active": true},
            {"entity_type": "HydroStorage", "entity_id": 5, "subindex": 0, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageStates");
    let result: StageStatesReadResult =
        deserialize_stage_states(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 9);
    assert_eq!(result.node_id, 12);
    assert_eq!(result.state_dimension, 2);
    assert_eq!(result.count, 3);
    assert_eq!(result.data, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

    assert_eq!(result.entity_manifest.len(), 2);
    assert_eq!(result.entity_manifest[0].entity_type, 0);
    assert_eq!(result.entity_manifest[0].entity_id, 4);
    assert!(result.entity_manifest[0].was_active);
    assert_eq!(result.entity_manifest[1].entity_id, 5);
}

/// Forward-compat: a buffer written against a pre-`id:5` `StageStates` schema
/// (no `node_id` field) must deserialize with `node_id` at
/// [`STAGE_STATES_NODE_ID_SENTINEL`], never a bare `0` (a valid node id).
/// Mirrors `pre_delivery_date_entity_slot_reads_as_sentinel`'s rewritten-schema
/// approach: flatc cannot emit a field the schema lacks, so the buffer is built
/// from a schema that stops at `entity_manifest (id: 4)`.
#[test]
fn pre_node_id_stage_states_reads_as_sentinel() {
    let schema_pre_node_id = "
namespace Cobre.IO.Policy;

file_identifier \"CBVF\";

enum EntityType : byte {
  HydroStorage = 0,
  HydroInflowLag = 1,
  AnticipatedThermalState = 2,
  HydroTransitBucket = 3,
}

table EntitySlot {
  entity_type:EntityType (id: 0);
  entity_id:int32 (id: 1);
  subindex:uint32 (id: 2);
  was_active:bool (id: 3);
}

table StageStates {
  stage_id:uint32 (id: 0);
  state_dimension:uint32 (id: 1);
  count:uint32 (id: 2);
  data:[float64] (id: 3);
  entity_manifest:[EntitySlot] (id: 4);
}
";
    let dir = TempDir::new().unwrap();
    let pre_node_id_schema = dir.path().join("pre_node_id.fbs");
    std::fs::write(&pre_node_id_schema, schema_pre_node_id).unwrap();

    let document = json!({
        "stage_id": 3,
        "state_dimension": 1,
        "count": 1,
        "data": [1.0],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true}
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified("StageStates"))
        .arg("-o")
        .arg(dir.path())
        .arg(&pre_node_id_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on pre-node_id schema");
    assert!(status.success(), "flatc -b on pre-node_id schema failed");
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result =
        deserialize_stage_states(&buf).expect("hand-rolled reader must accept pre-node_id buffer");
    assert_eq!(result.stage_id, 3);
    assert_eq!(
        result.node_id, STAGE_STATES_NODE_ID_SENTINEL,
        "a pre-node_id buffer must read back node_id as the sentinel, not 0"
    );
}

// ─── Schema neutrality (0.14 value-function artifact) ────────────────────────

/// The published schema speaks affine-piece vocabulary: no standalone `Cut`
/// table, no `state_at_generation` field, no live (non-deprecated)
/// `domination_count` slot, and the header names the `src/output/policy/`
/// directory module. A text inspection of `schemas/policy.fbs`.
#[test]
fn schema_carries_no_algorithm_vocabulary() {
    let schema = std::fs::read_to_string(schema_path()).expect("read policy.fbs");
    assert!(
        !schema.contains("table Cut "),
        "the schema must not declare a `Cut` table (renamed to AffinePiece)"
    );
    assert!(
        schema.contains("table AffinePiece"),
        "the schema must declare the neutral `AffinePiece` table"
    );
    assert!(
        !schema.contains("state_at_generation"),
        "the deleted `state_at_generation` field must not appear"
    );
    // A reclaimed id-4 field named `intercept` — not a live `domination_count`.
    assert!(
        !schema.contains("domination_count:uint32 (id: 4);"),
        "no live domination_count slot may survive the id-4 reclaim"
    );
    assert!(
        schema.contains("src/output/policy/"),
        "the header must name the src/output/policy/ directory module"
    );
}

/// The 0.14 `EntitySlot` change: id 4 is a burned `deprecated` placeholder, the
/// `YYYYMMDD` `delivery_date` sits at id 5, and the artifact declares the
/// `CBVF` `file_identifier` plus a `file_extension`. A text inspection of
/// `schemas/policy.fbs`.
#[test]
fn schema_burns_delivery_anchor_and_declares_file_identifier() {
    let schema = std::fs::read_to_string(schema_path()).expect("read policy.fbs");
    assert!(
        schema.contains("delivery_anchor:int32 (id: 4, deprecated);"),
        "EntitySlot id 4 must be the burned, deprecated delivery_anchor placeholder"
    );
    assert!(
        schema.contains("delivery_date:int32 (id: 5);"),
        "the YYYYMMDD delivery_date must sit at id 5"
    );
    assert!(
        schema.contains("file_identifier \"CBVF\";"),
        "the schema must declare the CBVF file_identifier"
    );
    assert!(
        schema.contains("file_extension"),
        "the schema must declare a file_extension"
    );
}

// ─── EntitySlot delivery_date (id: 5) round-trip + forward-compat ─────────────

/// §E6 round-trip: an `EntitySlot` carrying a non-sentinel `delivery_date`
/// survives the hand-rolled writer → hand-rolled reader path, and the same
/// buffer decodes through `flatc` with the date at slot id 5.
#[test]
fn entity_slot_delivery_date_round_trips() {
    let coeffs = [1.0, 2.0];
    let cuts = [PolicyCutRecord {
        cut_id: 1,
        slot_index: 0,
        iteration: 1,
        forward_pass_index: 0,
        intercept: 3.0,
        coefficients: &coeffs,
        is_active: true,
    }];
    let manifest = vec![
        EntitySlot {
            entity_type: 2,
            entity_id: 7,
            subindex: 1,
            was_active: true,
            delivery_date: 20240501,
        },
        EntitySlot {
            entity_type: 0,
            entity_id: 1,
            subindex: 0,
            was_active: true,
            delivery_date: ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        },
    ];
    let buf = serialize_stage_cuts(3, 2, 8, 0, &cuts, &[0], 1, &manifest);

    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume its own buffer");
    assert_eq!(result.entity_manifest.len(), 2);
    assert_eq!(result.entity_manifest[0].delivery_date, 20240501);
    assert_eq!(
        result.entity_manifest[1].delivery_date,
        ENTITY_SLOT_DELIVERY_DATE_SENTINEL
    );

    let json = flatc_decode(&buf, "StageCuts");
    let arr = get(&json, "entity_manifest")
        .as_array()
        .expect("entity_manifest is an array")
        .clone();
    assert_eq!(i64_or(&arr[0], "delivery_date", 0), i64::from(20240501));
    assert_eq!(
        i64_or(&arr[1], "delivery_date", 0),
        i64::from(ENTITY_SLOT_DELIVERY_DATE_SENTINEL)
    );
}

/// §E6 forward-compat (the schema-evolution half of the reject-role for the
/// `FlatBuffers` `policy/codec.rs` row; the identifier rejection is the codec
/// unit test): a buffer written against a pre-`id:5` `EntitySlot` schema (no
/// `delivery_date` field) must deserialize with every slot's date at the
/// sentinel and no error. flatc cannot emit a field the schema lacks, so the
/// buffer is built from a rewritten schema that stops at `was_active (id: 3)`;
/// that schema still declares the `CBVF` `file_identifier`, so the buffer clears
/// the read-path identifier gate that a genuine pre-0.14 buffer would fail.
#[test]
fn pre_delivery_date_entity_slot_reads_as_sentinel() {
    let schema_pre_delivery_date = "
namespace Cobre.IO.Policy;

file_identifier \"CBVF\";

enum EntityType : byte {
  HydroStorage = 0,
  HydroInflowLag = 1,
  AnticipatedThermalState = 2,
  HydroTransitBucket = 3,
}

table EntitySlot {
  entity_type:EntityType (id: 0);
  entity_id:int32 (id: 1);
  subindex:uint32 (id: 2);
  was_active:bool (id: 3);
}

table AffinePiece {
  piece_id:uint64 (id: 0);
  slot_index:uint32 (id: 1);
  iteration:uint32 (id: 2);
  forward_pass_index:uint32 (id: 3);
  intercept:float64 (id: 4);
  coefficients:[float64] (id: 5);
  is_active:bool (id: 6);
  reserved_7:[float64] (id: 7, deprecated);
}

table StageCuts {
  stage_id:uint32 (id: 0);
  state_dimension:uint32 (id: 1);
  capacity:uint32 (id: 2);
  warm_start_count:uint32 (id: 3);
  cuts:[AffinePiece] (id: 4);
  active_cut_indices:[uint32] (id: 5);
  populated_count:uint32 (id: 6);
  entity_manifest:[EntitySlot] (id: 7);
}
";
    let dir = TempDir::new().unwrap();
    let pre_delivery_date_schema = dir.path().join("pre_delivery_date.fbs");
    std::fs::write(&pre_delivery_date_schema, schema_pre_delivery_date).unwrap();

    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 4,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 1,
                "slot_index": 0,
                "iteration": 1,
                "forward_pass_index": 0,
                "intercept": 1.0,
                "coefficients": [1.0, 2.0],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true},
            {"entity_type": "AnticipatedThermalState", "entity_id": 7, "subindex": 1, "was_active": true}
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified("StageCuts"))
        .arg("-o")
        .arg(dir.path())
        .arg(&pre_delivery_date_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on pre-delivery_date schema");
    assert!(
        status.success(),
        "flatc -b on pre-delivery_date schema failed"
    );
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result = deserialize_stage_cuts(&buf)
        .expect("hand-rolled reader must accept pre-delivery_date buffer");
    assert_eq!(result.entity_manifest.len(), 2);
    for (i, slot) in result.entity_manifest.iter().enumerate() {
        assert_eq!(
            slot.delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "pre-delivery_date slot {i} must read back as the sentinel"
        );
    }
}
