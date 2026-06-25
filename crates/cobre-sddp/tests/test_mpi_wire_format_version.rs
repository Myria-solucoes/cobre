//! MPI wire-format version guards: `CapturedBasis::try_from_broadcast_payload`
//! and `deserialize_cut` must return `SddpError::Validation` on a corrupted
//! version byte/field — a stale version must be rejected, never silently
//! decoded as corrupt data. Exercised through the public API, no MPI spawn.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

use cobre_sddp::{
    SddpError,
    cut::wire::{CUT_WIRE_VERSION, cut_wire_size, deserialize_cut, serialize_cut},
    workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis},
};

// ---------------------------------------------------------------------------
// basis wire-format version guard
// ---------------------------------------------------------------------------

#[test]
fn basis_try_from_broadcast_payload_rejects_wrong_version() {
    let num_cols = 2;
    let num_rows = 3;
    let base_row_count = 1;
    let cut_slot_capacity = 2;
    let n_state = 1;

    let mut original = CapturedBasis::new(
        num_cols,
        num_rows,
        base_row_count,
        cut_slot_capacity,
        n_state,
    );
    // Minimal valid data so the Some-path sentinel (i32_buf[0] == 1) is written.
    original.basis.col_status.extend_from_slice(&[0_i32, 1_i32]);
    original
        .basis
        .row_status
        .extend_from_slice(&[0_i32, 1_i32, 0_i32]);
    original.cut_row_slots.push(0_u32);
    original.cut_row_slots.push(1_u32);
    original.state_at_capture.push(42.0_f64);

    let mut i32_buf: Vec<i32> = Vec::new();
    let mut f64_buf: Vec<f64> = Vec::new();
    original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

    // i32_buf layout (owned by to_broadcast_payload): [sentinel, version, ...] —
    // the version field is at index 1, which this test corrupts below.
    assert_eq!(i32_buf[0], 1, "sentinel must be 1 (Some path)");
    assert_eq!(
        i32_buf[1], BASIS_BROADCAST_WIRE_VERSION,
        "version field must equal BASIS_BROADCAST_WIRE_VERSION before corruption"
    );

    i32_buf[1] = 2;

    let mut i32_cursor = 0_usize;
    let mut f64_cursor = 0_usize;
    let result = CapturedBasis::try_from_broadcast_payload(
        0,
        &i32_buf,
        &mut i32_cursor,
        &f64_buf,
        &mut f64_cursor,
    );

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("unsupported wire version 2"),
                "error must contain 'unsupported wire version 2'; got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) containing 'unsupported wire version 2', \
             got: {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------------
// Cut wire-format version guard
// ---------------------------------------------------------------------------

#[test]
fn deserialize_cut_rejects_wrong_version() {
    let n_state = 2;
    let record_size = cut_wire_size(n_state);
    let mut buf = vec![0u8; record_size];

    serialize_cut(
        &mut buf,
        /* slot_index */ 0,
        /* iteration */ 1,
        /* forward_pass_index */ 0,
        /* intercept */ 99.0,
        /* coefficients */ &[1.0, 2.0],
    );

    assert_eq!(
        buf[0], CUT_WIRE_VERSION,
        "serialize_cut must write the current version byte"
    );

    buf[0] = 99_u8;

    let result = deserialize_cut(&buf, n_state);

    match result {
        Err(SddpError::Validation(ref msg)) => {
            assert!(
                msg.contains("99"),
                "error must reference the corrupted version 99; got: {msg}"
            );
        }
        other => panic!(
            "expected Err(SddpError::Validation(_)) referencing the corrupted version, \
             got: {other:?}"
        ),
    }
}
