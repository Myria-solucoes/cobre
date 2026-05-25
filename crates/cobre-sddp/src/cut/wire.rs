//! Wire format for MPI cut exchange.
//!
//! During the SDDP backward pass, each MPI rank broadcasts its newly generated
//! cuts and activity updates to all other ranks via `allgatherv`. Because the
//! coefficient count (`n_state`) is a runtime value, `allgatherv` is called
//! with `T = u8` and records are packed into a contiguous byte buffer.
//!
//! ## Wire format version 2 — two record types
//!
//! Version 2 payloads contain two record types interleaved in a single buffer.
//! Every record starts with a version byte (offset 0, must equal
//! `CUT_WIRE_VERSION = 2`) and a record tag byte (offset 13, either
//! `RECORD_TAG_CUT = 0` or `RECORD_TAG_ACTIVITY = 1`).
//!
//! ### `CutRecord` (tag = 0)
//!
//! Total size: `25 + n_state * 8` bytes.
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0       1   version             (u8 = 2)
//!  1- 4    4   slot_index          (u32, native-endian)
//!  5- 8    4   iteration           (u32, native-endian)
//!  9-12    4   forward_pass_index  (u32, native-endian)
//! 13       1   record_tag          (u8 = 0)
//! 14-16    3   padding             (zeroed; future use)
//! 17-24    8   intercept           (f64, native-endian)
//! 25 ...   8*n coefficients[0..n]  (f64 each, native-endian)
//! ```
//!
//! ### `ActivityUpdateRecord` (tag = 1)
//!
//! Total size: 25 bytes (padded to match the `CutRecord` minimum stride so the
//! receiver loop can iterate with a uniform stride at `n_state = 0`).
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0       1   version             (u8 = 2)
//!  1- 4    4   slot_index          (u32, native-endian)
//!  5- 8    4   stage_index         (u32, native-endian)
//!  9-12    4   reserved            (zeroed)
//! 13       1   record_tag          (u8 = 1)
//! 14       1   activity_change_tag (u8: 0=Deactivate, 1=Reactivate)
//! 15-24   10   reserved            (zeroed)
//! ```
//!
//! ## Version compatibility
//!
//! Wire version 1 receivers fed a version 2 payload return
//! `SddpError::Validation`. Wire version 2 receivers fed a version 1 payload
//! return `SddpError::Validation`. No compatibility shim is provided.
//!
//! ## Functions
//!
//! - [`cut_wire_size`] — compute the byte size for one cut record.
//! - [`activity_update_wire_size`] — byte size for one activity update record.
//! - [`max_wire_record_size`] — maximum record size (cut records are larger).
//! - [`serialize_cut`] — write one cut record into a byte buffer.
//! - [`deserialize_cut`] — read one cut record from a byte buffer.
//! - [`serialize_activity_update`] — write one activity update record.
//! - [`deserialize_activity_update`] — read one activity update record.
//! - [`serialize_records_to_buffer`] — pack cuts and activity updates into a buffer.
//! - [`deserialize_records_from_buffer_into`] — unpack mixed records (no allocation).
//! - [`deserialize_cuts_from_buffer`] — unpack cut-only records from a buffer.
//! - [`deserialize_cuts_from_buffer_into`] — unpack cuts into caller-provided buffers.

use crate::{SddpError, cut_selection::ActivityChange};

// ---------------------------------------------------------------------------
// Wire version and record-tag constants
// ---------------------------------------------------------------------------

/// Wire format version byte. Bump when the payload layout changes
/// in a backward-incompatible way.
pub const CUT_WIRE_VERSION: u8 = 2;

/// Record-tag value identifying a cut record (offset 13 in every record).
pub const RECORD_TAG_CUT: u8 = 0;

/// Record-tag value identifying an activity update record (offset 13).
pub const RECORD_TAG_ACTIVITY: u8 = 1;

/// Activity-tag value encoding `ActivityChange::Deactivate` (offset 14 in activity records).
pub const ACTIVITY_TAG_DEACTIVATE: u8 = 0;

/// Activity-tag value encoding `ActivityChange::Reactivate` (offset 14 in activity records).
pub const ACTIVITY_TAG_REACTIVATE: u8 = 1;

// ---------------------------------------------------------------------------
// CutWireHeader
// ---------------------------------------------------------------------------

/// Parsed header from a [`cut wire record`](self).
///
/// This struct holds the decoded header fields of a cut wire record.
/// It is a plain Rust struct (not `#[repr(C)]`); byte conversion is handled
/// explicitly by [`serialize_cut`] and [`deserialize_cut`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutWireHeader {
    /// Deterministic slot index in the target [`CutPool`].
    ///
    /// [`CutPool`]: crate::cut::CutPool
    pub slot_index: u32,

    /// Training iteration counter when this cut was generated.
    pub iteration: u32,

    /// Forward pass index within the iteration when this cut was generated.
    pub forward_pass_index: u32,

    /// Intercept of the Benders cut (`α` in `α + β · x`).
    pub intercept: f64,
}

// ---------------------------------------------------------------------------
// ActivityUpdateRecord
// ---------------------------------------------------------------------------

/// Parsed fields of an activity update wire record.
///
/// Activity updates are carried alongside cut records in the version-2 wire
/// format. Each record occupies 25 bytes on the wire; the `change` field is
/// encoded as a single byte at offset 14 (`ACTIVITY_TAG_DEACTIVATE` or
/// `ACTIVITY_TAG_REACTIVATE`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityUpdateRecord {
    /// Deterministic slot index of the cut whose activity is being changed.
    pub slot_index: u32,
    /// Stage index (0-based) that this update belongs to.
    pub stage_index: u32,
    /// Direction of the activity change.
    pub change: ActivityChange,
}

// ---------------------------------------------------------------------------
// cut_wire_size
// ---------------------------------------------------------------------------

/// Return the byte size of one cut wire record with `n_state` coefficients.
///
/// The layout is a 25-byte fixed header (1 version byte + 24 bytes of fields)
/// followed by `n_state * 8` bytes for the coefficient array:
///
/// ```
/// use cobre_sddp::cut::wire::cut_wire_size;
///
/// assert_eq!(cut_wire_size(0), 25);
/// assert_eq!(cut_wire_size(1), 33);
/// assert_eq!(cut_wire_size(9), 97);
/// assert_eq!(cut_wire_size(2080), 16665);
/// ```
#[inline]
#[must_use]
pub fn cut_wire_size(n_state: usize) -> usize {
    25 + n_state * 8
}

/// Return the byte size of one activity update wire record (always 25).
///
/// Activity update records have no trailing coefficient bytes and occupy a
/// fixed 25 bytes on the wire, matching the minimum stride of a cut record at
/// `n_state = 0`.
#[inline]
#[must_use]
pub fn activity_update_wire_size() -> usize {
    25
}

/// Return the maximum record size for the given `n_state`.
///
/// Cut records are always the larger of the two record types, so this is
/// equivalent to [`cut_wire_size`].
#[inline]
#[must_use]
pub fn max_wire_record_size(n_state: usize) -> usize {
    cut_wire_size(n_state)
}

/// Serialize one cut record into `buf` starting at offset 0.
///
/// Writes the version byte (`CUT_WIRE_VERSION`) at offset 0, then the header
/// as three `u32` values (12 bytes) at offsets 1–12, then the record tag
/// `RECORD_TAG_CUT` at offset 13 followed by 3 zero padding bytes at offsets
/// 14–16, then one `f64` intercept (8 bytes) at offsets 17–24. Coefficients
/// follow immediately as native-endian `f64` bytes starting at offset 25.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(coefficients.len())`.
pub fn serialize_cut(
    buf: &mut [u8],
    slot_index: u32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: &[f64],
) {
    debug_assert!(
        buf.len() >= cut_wire_size(coefficients.len()),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(coefficients.len())
    );

    buf[0] = CUT_WIRE_VERSION;
    buf[1..5].copy_from_slice(&slot_index.to_ne_bytes());
    buf[5..9].copy_from_slice(&iteration.to_ne_bytes());
    buf[9..13].copy_from_slice(&forward_pass_index.to_ne_bytes());
    buf[13] = RECORD_TAG_CUT;
    buf[14] = 0;
    buf[15] = 0;
    buf[16] = 0;
    buf[17..25].copy_from_slice(&intercept.to_ne_bytes());

    for (i, &coeff) in coefficients.iter().enumerate() {
        let start = 25 + i * 8;
        buf[start..start + 8].copy_from_slice(&coeff.to_ne_bytes());
    }
}

/// Deserialize one cut record from `buf`, expecting `n_state` coefficients.
///
/// Reads the version byte at offset 0 and returns an error if it does not
/// match [`CUT_WIRE_VERSION`]. Reads the record tag at offset 13 and returns
/// an error if it does not equal [`RECORD_TAG_CUT`]. Then reads the 24-byte
/// header from fixed offsets starting at 1 and recovers `n_state` `f64`
/// values starting at offset 25.
///
/// After the length `debug_assert`, all slice-to-array conversions use direct
/// fixed-length indexing, which is infallible for the exact sizes used here.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if:
/// - The version byte does not equal [`CUT_WIRE_VERSION`]. The message
///   contains `"cut_wire: unsupported version"`.
/// - The record tag at offset 13 does not equal [`RECORD_TAG_CUT`]. The
///   message contains `"cut_wire: expected cut record"`.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(n_state)`.
pub fn deserialize_cut(buf: &[u8], n_state: usize) -> Result<(CutWireHeader, Vec<f64>), SddpError> {
    debug_assert!(
        buf.len() >= cut_wire_size(n_state),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(n_state)
    );

    let version = buf[0];
    if version != CUT_WIRE_VERSION {
        return Err(SddpError::Validation(format!(
            "cut_wire: unsupported version {version}"
        )));
    }

    let record_tag = buf[13];
    if record_tag != RECORD_TAG_CUT {
        return Err(SddpError::Validation(format!(
            "cut_wire: expected cut record (tag {RECORD_TAG_CUT}), got tag {record_tag}"
        )));
    }

    // All slice-to-array conversions below are infallible: the debug_assert
    // above guarantees buf.len() >= 25 + n_state*8, so the fixed offsets 1..5,
    // 5..9, 9..13, and 17..25 are all within bounds.
    let slot_index = u32::from_ne_bytes([buf[1], buf[2], buf[3], buf[4]]);
    let iteration = u32::from_ne_bytes([buf[5], buf[6], buf[7], buf[8]]);
    let forward_pass_index = u32::from_ne_bytes([buf[9], buf[10], buf[11], buf[12]]);
    let intercept = f64::from_ne_bytes([
        buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23], buf[24],
    ]);

    let header = CutWireHeader {
        slot_index,
        iteration,
        forward_pass_index,
        intercept,
    };

    let coefficients: Vec<f64> = (0..n_state)
        .map(|i| {
            let s = 25 + i * 8;
            f64::from_ne_bytes([
                buf[s],
                buf[s + 1],
                buf[s + 2],
                buf[s + 3],
                buf[s + 4],
                buf[s + 5],
                buf[s + 6],
                buf[s + 7],
            ])
        })
        .collect();

    Ok((header, coefficients))
}

/// Serialize multiple cuts into a freshly allocated contiguous byte buffer.
///
/// Each element of `cuts` is a tuple `(slot_index, iteration,
/// forward_pass_index, intercept, coefficients)`.  All cuts must have the
/// same `n_state` coefficient count; `n_state` is passed explicitly so the
/// caller controls the layout without iterating over the slice.
///
/// Returns a `Vec<u8>` of length `cuts.len() * cut_wire_size(n_state)`.
///
/// # Allocation
///
/// This function allocates `cuts.len() * cut_wire_size(n_state)` bytes on
/// every call. It is intended for off-hot-path use: tests, policy export, and
/// one-shot serialization. The production MPI hot path uses
/// `CutSyncBuffers::pack_local_cuts_into` which writes into a pre-allocated
/// buffer instead.
///
/// # Panics (debug builds only)
///
/// Panics if any coefficient slice has length != `n_state`.
#[cold]
#[must_use]
pub fn serialize_cuts_to_buffer(cuts: &[(u32, u32, u32, f64, &[f64])], n_state: usize) -> Vec<u8> {
    let record_size = cut_wire_size(n_state);
    let mut buf = vec![0u8; cuts.len() * record_size];

    for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
        cuts.iter().enumerate()
    {
        debug_assert!(
            coefficients.len() == n_state,
            "cut {i} coefficient length {} != n_state {n_state}",
            coefficients.len()
        );
        let start = i * record_size;
        serialize_cut(
            &mut buf[start..start + record_size],
            slot_index,
            iteration,
            forward_pass_index,
            intercept,
            coefficients,
        );
    }

    buf
}

/// Deserialize all cuts from a contiguous byte buffer.
///
/// The buffer must contain a whole number of cut records: its length must be
/// `0` or a multiple of `cut_wire_size(n_state)`. Returns a `Vec` of
/// `(header, coefficients)` pairs in the same order they appear in the buffer.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if any cut record contains an
/// unrecognised version byte.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of `cut_wire_size(n_state)` (when
/// `n_state > 0`).
pub fn deserialize_cuts_from_buffer(
    buf: &[u8],
    n_state: usize,
) -> Result<Vec<(CutWireHeader, Vec<f64>)>, SddpError> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }

    let record_size = cut_wire_size(n_state);
    assert!(
        buf.len() % record_size == 0,
        "buffer length {} is not a multiple of record size {record_size}",
        buf.len()
    );

    let n_cuts = buf.len() / record_size;
    (0..n_cuts)
        .map(|i| {
            let start = i * record_size;
            deserialize_cut(&buf[start..start + record_size], n_state)
        })
        .collect()
}

/// Deserialize all cuts from a contiguous byte buffer into caller-provided
/// pre-allocated scratch buffers.
///
/// On return, `headers_out` contains one [`CutWireHeader`] per cut record and
/// `coefficients_flat_out` contains all coefficients concatenated in order:
/// cut 0's `n_state` values, then cut 1's, and so on (flat `SoA` layout).
///
/// Both output buffers are cleared at the start of each call so they can be
/// reused across iterations without releasing their heap allocation.
///
/// The buffer must contain a whole number of cut records: its length must be
/// `0` or a multiple of `cut_wire_size(n_state)`.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if any cut record contains an
/// unrecognised version byte. On error, the output buffers are in an
/// unspecified partial state.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of `cut_wire_size(n_state)` (when
/// `n_state > 0`).
pub fn deserialize_cuts_from_buffer_into(
    buf: &[u8],
    n_state: usize,
    headers_out: &mut Vec<CutWireHeader>,
    coefficients_flat_out: &mut Vec<f64>,
) -> Result<(), SddpError> {
    headers_out.clear();
    coefficients_flat_out.clear();

    if buf.is_empty() {
        return Ok(());
    }

    let record_size = cut_wire_size(n_state);
    assert!(
        buf.len() % record_size == 0,
        "buffer length {} is not a multiple of record size {record_size}",
        buf.len()
    );

    let n_cuts = buf.len() / record_size;
    headers_out.reserve(n_cuts);
    coefficients_flat_out.reserve(n_cuts * n_state);

    for i in 0..n_cuts {
        let start = i * record_size;
        let (header, coefficients) = deserialize_cut(&buf[start..start + record_size], n_state)?;
        headers_out.push(header);
        coefficients_flat_out.extend_from_slice(&coefficients);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Activity update serialization
// ---------------------------------------------------------------------------

/// Serialize one activity update record into `buf` starting at offset 0.
///
/// The record is always 25 bytes (see [`activity_update_wire_size`]).
///
/// Layout:
/// - byte 0: [`CUT_WIRE_VERSION`]
/// - bytes 1–4: `slot_index` (u32, native-endian)
/// - bytes 5–8: `stage_index` (u32, native-endian)
/// - bytes 9–12: zeros (reserved)
/// - byte 13: [`RECORD_TAG_ACTIVITY`]
/// - byte 14: [`ACTIVITY_TAG_DEACTIVATE`] or [`ACTIVITY_TAG_REACTIVATE`]
/// - bytes 15–24: zeros (reserved)
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if `buf.len() < 25`.
pub fn serialize_activity_update(
    buf: &mut [u8],
    rec: &ActivityUpdateRecord,
) -> Result<usize, SddpError> {
    if buf.len() < 25 {
        return Err(SddpError::Validation(format!(
            "cut_wire: activity update buffer too small: {} < 25",
            buf.len()
        )));
    }
    buf[0] = CUT_WIRE_VERSION;
    buf[1..5].copy_from_slice(&rec.slot_index.to_ne_bytes());
    buf[5..9].copy_from_slice(&rec.stage_index.to_ne_bytes());
    buf[9] = 0;
    buf[10] = 0;
    buf[11] = 0;
    buf[12] = 0;
    buf[13] = RECORD_TAG_ACTIVITY;
    buf[14] = match rec.change {
        ActivityChange::Deactivate => ACTIVITY_TAG_DEACTIVATE,
        ActivityChange::Reactivate => ACTIVITY_TAG_REACTIVATE,
    };
    buf[15] = 0;
    buf[16] = 0;
    buf[17] = 0;
    buf[18] = 0;
    buf[19] = 0;
    buf[20] = 0;
    buf[21] = 0;
    buf[22] = 0;
    buf[23] = 0;
    buf[24] = 0;
    Ok(25)
}

/// Deserialize one activity update record from `buf`.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if:
/// - byte 0 != [`CUT_WIRE_VERSION`]
/// - byte 13 != [`RECORD_TAG_ACTIVITY`]
/// - byte 14 is neither [`ACTIVITY_TAG_DEACTIVATE`] nor
///   [`ACTIVITY_TAG_REACTIVATE`]
pub fn deserialize_activity_update(buf: &[u8]) -> Result<ActivityUpdateRecord, SddpError> {
    if buf.len() < 25 {
        return Err(SddpError::Validation(format!(
            "cut_wire: activity update buffer too small: {} < 25",
            buf.len()
        )));
    }
    let version = buf[0];
    if version != CUT_WIRE_VERSION {
        return Err(SddpError::Validation(format!(
            "cut_wire: unsupported version {version}"
        )));
    }
    let record_tag = buf[13];
    if record_tag != RECORD_TAG_ACTIVITY {
        return Err(SddpError::Validation(format!(
            "cut_wire: expected activity record (tag {RECORD_TAG_ACTIVITY}), got tag {record_tag}"
        )));
    }
    let slot_index = u32::from_ne_bytes([buf[1], buf[2], buf[3], buf[4]]);
    let stage_index = u32::from_ne_bytes([buf[5], buf[6], buf[7], buf[8]]);
    let change = match buf[14] {
        ACTIVITY_TAG_DEACTIVATE => ActivityChange::Deactivate,
        ACTIVITY_TAG_REACTIVATE => ActivityChange::Reactivate,
        other => {
            return Err(SddpError::Validation(format!(
                "cut_wire: unknown activity change tag {other}"
            )));
        }
    };
    Ok(ActivityUpdateRecord {
        slot_index,
        stage_index,
        change,
    })
}

// ---------------------------------------------------------------------------
// Mixed-record buffer serialization / deserialization
// ---------------------------------------------------------------------------

/// Pack cuts and activity updates into a caller-provided byte buffer.
///
/// Each cut record occupies `cut_wire_size(n_state)` bytes; each activity
/// update record occupies [`activity_update_wire_size`] (25) bytes. Records
/// are written in order: all cuts first, then all activities.
///
/// Returns the total number of bytes written.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if `buf` is too small to hold
/// all records.
pub fn serialize_records_to_buffer(
    buf: &mut [u8],
    cuts: &[(CutWireHeader, &[f64])],
    activities: &[ActivityUpdateRecord],
    n_state: usize,
) -> Result<usize, SddpError> {
    let cut_size = cut_wire_size(n_state);
    let act_size = activity_update_wire_size();
    let total = cuts.len() * cut_size + activities.len() * act_size;
    if buf.len() < total {
        return Err(SddpError::Validation(format!(
            "cut_wire: buffer too small for records: {} < {total}",
            buf.len()
        )));
    }
    let mut offset = 0usize;
    for (header, coefficients) in cuts {
        serialize_cut(
            &mut buf[offset..offset + cut_size],
            header.slot_index,
            header.iteration,
            header.forward_pass_index,
            header.intercept,
            coefficients,
        );
        offset += cut_size;
    }
    for rec in activities {
        serialize_activity_update(&mut buf[offset..offset + act_size], rec)?;
        offset += act_size;
    }
    Ok(offset)
}

/// Unpack a mixed buffer of cut records and activity update records.
///
/// Walks `buf` record by record, dispatching on byte 13 (the record tag):
/// - `RECORD_TAG_CUT` (0): consumes `cut_wire_size(n_state)` bytes, appends
///   to `out_cuts`.
/// - `RECORD_TAG_ACTIVITY` (1): consumes 25 bytes, appends to `out_activities`.
/// - Any other value: returns `Err(SddpError::Validation(_))`.
///
/// Both output vectors are cleared at the start of each call so they can be
/// reused across iterations without releasing their heap allocation.
///
/// Returns `(n_cuts, n_activities)` decoded.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if any record contains an
/// unrecognised version byte, an unknown record tag, or an unknown activity
/// change tag.
pub fn deserialize_records_from_buffer_into(
    buf: &[u8],
    n_state: usize,
    out_cuts: &mut Vec<(CutWireHeader, Vec<f64>)>,
    out_activities: &mut Vec<ActivityUpdateRecord>,
) -> Result<(usize, usize), SddpError> {
    out_cuts.clear();
    out_activities.clear();

    let cut_size = cut_wire_size(n_state);
    let act_size = activity_update_wire_size();
    let mut pos = 0usize;

    while pos < buf.len() {
        // Every record is at least 25 bytes; peek at byte 13 for the tag.
        if pos + 25 > buf.len() {
            return Err(SddpError::Validation(format!(
                "cut_wire: truncated record at byte offset {pos}"
            )));
        }
        let record_tag = buf[pos + 13];
        match record_tag {
            RECORD_TAG_CUT => {
                if pos + cut_size > buf.len() {
                    return Err(SddpError::Validation(format!(
                        "cut_wire: truncated cut record at byte offset {pos}"
                    )));
                }
                let (header, coefficients) = deserialize_cut(&buf[pos..pos + cut_size], n_state)?;
                out_cuts.push((header, coefficients));
                pos += cut_size;
            }
            RECORD_TAG_ACTIVITY => {
                let rec = deserialize_activity_update(&buf[pos..pos + act_size])?;
                out_activities.push(rec);
                pos += act_size;
            }
            other => {
                return Err(SddpError::Validation(format!(
                    "cut_wire: unknown record tag {other} at byte offset {pos}"
                )));
            }
        }
    }

    Ok((out_cuts.len(), out_activities.len()))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation, // loop indices are small constants in tests
        clippy::cast_precision_loss,      // usize cast to f64 is intentional in tests
        clippy::cast_lossless,            // i32→f64 is lossless but clippy prefers From
        clippy::unwrap_used,              // unwrap is acceptable in tests
        clippy::expect_used,              // expect is acceptable in tests
    )]

    use super::{
        ACTIVITY_TAG_DEACTIVATE, ACTIVITY_TAG_REACTIVATE, ActivityUpdateRecord, CUT_WIRE_VERSION,
        CutWireHeader, RECORD_TAG_ACTIVITY, RECORD_TAG_CUT, activity_update_wire_size,
        cut_wire_size, deserialize_activity_update, deserialize_cut, deserialize_cuts_from_buffer,
        deserialize_cuts_from_buffer_into, deserialize_records_from_buffer_into,
        serialize_activity_update, serialize_cut, serialize_cuts_to_buffer,
        serialize_records_to_buffer,
    };
    use crate::{SddpError, cut_selection::ActivityChange};

    #[test]
    fn cut_wire_size_zero_state_returns_25() {
        assert_eq!(cut_wire_size(0), 25);
    }

    #[test]
    fn cut_wire_size_one_state_returns_33() {
        assert_eq!(cut_wire_size(1), 33);
    }

    #[test]
    fn cut_wire_size_three_hydro_ar2_returns_97() {
        // 3-hydro AR(2) system: n_state = 9 → 25 + 9 * 8 = 97
        assert_eq!(cut_wire_size(9), 97);
    }

    #[test]
    fn cut_wire_size_production_scale_returns_16665() {
        // Production-scale: n_state = 2080 → 25 + 2080 * 8 = 16665
        assert_eq!(cut_wire_size(2080), 16665);
    }

    #[test]
    fn round_trip_all_fields_match_exactly() {
        let n_state = 3;
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 5, 3, 7, 42.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.slot_index, 5);
        assert_eq!(header.iteration, 3);
        assert_eq!(header.forward_pass_index, 7);
        assert_eq!(header.intercept, 42.0);
        assert_eq!(recovered, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn round_trip_verifies_bit_for_bit_coefficient_integrity() {
        // Use values that are not exactly representable in f64 to verify
        // that round-trip preserves the bit pattern exactly.
        let n_state = 4;
        let val = 1.0_f64 / 3.0;
        let coefficients = [val, -val, val * 2.0, f64::MIN_POSITIVE];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 1, 10, 2, f64::MAX, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.intercept.to_bits(), f64::MAX.to_bits());
        for (orig, got) in coefficients.iter().zip(&recovered) {
            assert_eq!(orig.to_bits(), got.to_bits(), "coefficient mismatch");
        }
    }

    #[test]
    fn byte_offsets_match_wire_format_spec() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(3)];

        serialize_cut(&mut buf, 5, 3, 7, 42.0, &coefficients);

        // version at offset 0
        assert_eq!(buf[0], CUT_WIRE_VERSION, "version at offset 0");
        // slot_index at offset 1-4
        assert_eq!(
            u32::from_ne_bytes(buf[1..5].try_into().unwrap()),
            5u32,
            "slot_index at offset 1"
        );
        // iteration at offset 5-8
        assert_eq!(
            u32::from_ne_bytes(buf[5..9].try_into().unwrap()),
            3u32,
            "iteration at offset 5"
        );
        // forward_pass_index at offset 9-12
        assert_eq!(
            u32::from_ne_bytes(buf[9..13].try_into().unwrap()),
            7u32,
            "forward_pass_index at offset 9"
        );
        // padding at offset 13-16 must be zero
        assert_eq!(&buf[13..17], &[0u8; 4], "padding at offset 13 must be zero");
        // intercept at offset 17-24
        assert_eq!(
            f64::from_ne_bytes(buf[17..25].try_into().unwrap()),
            42.0_f64,
            "intercept at offset 17"
        );
        // first coefficient at offset 25
        assert_eq!(
            f64::from_ne_bytes(buf[25..33].try_into().unwrap()),
            1.0_f64,
            "coefficient[0] at offset 25"
        );
    }

    #[test]
    fn round_trip_production_scale_n_state_2080() {
        let n_state = 2080;
        let coefficients: Vec<f64> = (0..n_state).map(|i| i as f64 * 0.001).collect();
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 100, 50, 3, 999.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.slot_index, 100);
        assert_eq!(header.iteration, 50);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, 999.0);
        assert_eq!(recovered.len(), n_state);
        for (i, (orig, got)) in coefficients.iter().zip(&recovered).enumerate() {
            assert_eq!(orig.to_bits(), got.to_bits(), "mismatch at coefficient {i}");
        }
    }

    #[test]
    fn edge_case_n_state_zero_header_only_25_bytes() {
        let mut buf = vec![0u8; cut_wire_size(0)];
        assert_eq!(buf.len(), 25);

        serialize_cut(&mut buf, 1, 2, 3, -1.0, &[]);
        let (header, coefficients) = deserialize_cut(&buf, 0).unwrap();

        assert_eq!(header.slot_index, 1);
        assert_eq!(header.iteration, 2);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, -1.0);
        assert!(coefficients.is_empty());
    }

    #[test]
    fn edge_case_n_state_one_produces_33_byte_record() {
        let mut buf = vec![0u8; cut_wire_size(1)];
        assert_eq!(buf.len(), 33);

        // Use 2.5 (exactly representable in f64) as a non-PI coefficient.
        let coeff = 2.5_f64;
        serialize_cut(&mut buf, 0, 0, 0, 7.0, &[coeff]);
        let (header, coefficients) = deserialize_cut(&buf, 1).unwrap();

        assert_eq!(header.intercept, 7.0);
        assert_eq!(coefficients.len(), 1);
        assert_eq!(coefficients[0].to_bits(), coeff.to_bits());
    }

    #[test]
    fn padding_bytes_at_offset_13_to_16_are_zero() {
        let mut buf = vec![0xFFu8; cut_wire_size(2)]; // Pre-fill with 0xFF
        serialize_cut(&mut buf, 1, 1, 1, 1.0, &[1.0, 2.0]);
        assert_eq!(&buf[13..17], &[0u8; 4], "padding bytes must be zero");
    }

    #[test]
    fn multi_cut_five_cuts_round_trip_all_match() {
        let n_state = 3;
        let coefficients: Vec<[f64; 3]> = (0..5u32).map(|i| [f64::from(i); 3]).collect();
        let cuts: Vec<(u32, u32, u32, f64, &[f64])> = coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (idx, idx * 2, idx, f64::from(idx) * 10.0, c.as_slice())
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        assert_eq!(buf.len(), 5 * cut_wire_size(n_state));

        let recovered = deserialize_cuts_from_buffer(&buf, n_state).unwrap();
        assert_eq!(recovered.len(), 5);

        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx, "slot_index mismatch at cut {i}");
            assert_eq!(header.iteration, idx * 2, "iteration mismatch at cut {i}");
            assert_eq!(
                header.forward_pass_index, idx,
                "forward_pass_index mismatch at cut {i}"
            );
            assert_eq!(
                header.intercept,
                f64::from(idx) * 10.0,
                "intercept mismatch at cut {i}"
            );
            for (j, &c) in coeffs.iter().enumerate() {
                assert_eq!(c, f64::from(idx), "coefficient[{j}] mismatch at cut {i}");
            }
        }
    }

    #[test]
    fn multi_cut_ten_cuts_round_trip_order_preserved() {
        let n_state = 2;
        let all_coefficients: Vec<Vec<f64>> = (0..10u32)
            .map(|i| vec![f64::from(i), -f64::from(i)])
            .collect();
        let cuts: Vec<(u32, u32, u32, f64, &[f64])> = all_coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (idx, 0u32, idx, f64::from(idx), c.as_slice())
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        let recovered = deserialize_cuts_from_buffer(&buf, n_state).unwrap();

        assert_eq!(recovered.len(), 10);
        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx);
            assert_eq!(coeffs[0].to_bits(), f64::from(idx).to_bits());
            assert_eq!(coeffs[1].to_bits(), (-f64::from(idx)).to_bits());
        }
    }

    #[test]
    fn deserialize_cuts_from_empty_buffer_returns_empty_vec() {
        let result = deserialize_cuts_from_buffer(&[], 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cut_wire_header_derives_debug_clone_copy_partialeq() {
        let h = CutWireHeader {
            slot_index: 1,
            iteration: 2,
            forward_pass_index: 3,
            intercept: 4.0,
        };
        let cloned = h;
        assert_eq!(h, cloned);
        let debug_str = format!("{h:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn deserialize_cuts_from_buffer_into_populates_buffers() {
        // Serialize 3 cuts with n_state=2, then verify that
        // deserialize_cuts_from_buffer_into produces values bit-for-bit
        // identical to those from deserialize_cuts_from_buffer.
        let n_state = 2usize;
        let cuts_data: &[(u32, u32, u32, f64, &[f64])] = &[
            (0, 1, 0, 10.0, &[1.0, 2.0]),
            (1, 2, 1, 20.0, &[3.0, 4.0]),
            (2, 3, 2, 30.0, &[5.0, 6.0]),
        ];
        let buf = serialize_cuts_to_buffer(cuts_data, n_state);

        // New path: into pre-allocated buffers.
        let mut headers_out: Vec<CutWireHeader> = Vec::new();
        let mut coefficients_flat_out: Vec<f64> = Vec::new();
        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();

        assert_eq!(headers_out.len(), 3, "must produce exactly 3 headers");
        assert_eq!(
            coefficients_flat_out.len(),
            3 * n_state,
            "flat coefficient buffer must have 3 * n_state entries"
        );

        // Old path: allocating reference.
        let reference = deserialize_cuts_from_buffer(&buf, n_state).unwrap();
        assert_eq!(reference.len(), 3);

        // Values must be bit-for-bit identical.
        for (i, (ref_header, ref_coeffs)) in reference.iter().enumerate() {
            assert_eq!(headers_out[i], *ref_header, "header mismatch at cut {i}");
            let start = i * n_state;
            for j in 0..n_state {
                assert_eq!(
                    coefficients_flat_out[start + j].to_bits(),
                    ref_coeffs[j].to_bits(),
                    "coefficient[{j}] mismatch at cut {i}"
                );
            }
        }
    }

    #[test]
    fn deserialize_cuts_from_buffer_into_reuses_capacity() {
        // Call twice with the same buffers and verify that after the second
        // call the capacity is at least as large as after the first (proving
        // the Vec allocation is retained between calls).
        let n_state = 3usize;
        let cuts_data: &[(u32, u32, u32, f64, &[f64])] = &[
            (0, 1, 0, 1.0, &[1.0, 2.0, 3.0]),
            (1, 1, 1, 2.0, &[4.0, 5.0, 6.0]),
            (2, 1, 2, 3.0, &[7.0, 8.0, 9.0]),
        ];
        let buf = serialize_cuts_to_buffer(cuts_data, n_state);

        let mut headers_out: Vec<CutWireHeader> = Vec::new();
        let mut coefficients_flat_out: Vec<f64> = Vec::new();

        // First call: buffers grow to hold 3 cuts.
        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();
        let cap_headers_after_first = headers_out.capacity();
        let cap_coeffs_after_first = coefficients_flat_out.capacity();

        assert!(
            cap_headers_after_first >= 3,
            "headers capacity must be >= 3 after first call, got {cap_headers_after_first}"
        );

        // Second call: buffers are cleared then re-populated without
        // releasing the previous allocation.
        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();

        assert!(
            headers_out.capacity() >= cap_headers_after_first,
            "headers capacity must not shrink between calls"
        );
        assert!(
            coefficients_flat_out.capacity() >= cap_coeffs_after_first,
            "coefficients capacity must not shrink between calls"
        );
        assert_eq!(
            headers_out.len(),
            3,
            "second call must still produce 3 headers"
        );
    }

    // ── New tests for AC1–AC3, AC6 ────────────────────────────────────────────

    #[test]
    fn serialize_cut_writes_version_at_offset_zero() {
        let n_state = 3;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 5, 3, 7, 42.0, &[1.0, 2.0, 3.0]);
        assert_eq!(
            buf[0], CUT_WIRE_VERSION,
            "version byte at offset 0 must equal CUT_WIRE_VERSION"
        );
        // AC6: padding at new offset 13-16 is preserved as zeroed
        assert_eq!(
            &buf[13..17],
            &[0u8; 4],
            "padding at offset 13-16 must be zero"
        );
    }

    #[test]
    fn deserialize_cut_rejects_wrong_version() {
        let n_state = 3;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 5, 3, 7, 42.0, &[1.0, 2.0, 3.0]);

        // Overwrite the version byte with wire version 1 (the old format).
        buf[0] = 1_u8;

        let result = deserialize_cut(&buf, n_state);
        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("unsupported version"),
                    "error message must contain 'unsupported version', got: {msg}"
                );
            }
            other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
        }
    }

    #[test]
    fn cut_wire_size_matches_25_plus_n_state_times_8_spec() {
        // AC3: assert the four canonical sizes.
        assert_eq!(cut_wire_size(0), 25);
        assert_eq!(cut_wire_size(1), 33);
        assert_eq!(cut_wire_size(9), 97);
        assert_eq!(cut_wire_size(2080), 16665);
    }

    // ── New tests for version-2 wire format ──────────────────────────────────

    #[test]
    fn wire_version_2_constant_value() {
        assert_eq!(CUT_WIRE_VERSION, 2);
    }

    #[test]
    fn serialize_cut_writes_record_tag_zero_at_offset_13() {
        let n_state = 2;
        let mut buf = vec![0xFFu8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 7, 1, 3, 5.0, &[1.0, 2.0]);
        assert_eq!(
            buf[13], RECORD_TAG_CUT,
            "byte 13 must equal RECORD_TAG_CUT after serialize_cut"
        );
    }

    #[test]
    fn serialize_activity_update_round_trips() {
        let rec = ActivityUpdateRecord {
            slot_index: 42,
            stage_index: 7,
            change: ActivityChange::Reactivate,
        };
        let mut buf = vec![0u8; activity_update_wire_size()];
        let written = serialize_activity_update(&mut buf, &rec).unwrap();
        assert_eq!(written, 25);

        assert_eq!(buf[0], CUT_WIRE_VERSION, "version byte at offset 0");
        assert_eq!(buf[13], RECORD_TAG_ACTIVITY, "record tag at offset 13");
        assert_eq!(
            buf[14], ACTIVITY_TAG_REACTIVATE,
            "activity tag at offset 14"
        );

        let recovered = deserialize_activity_update(&buf).unwrap();
        assert_eq!(recovered, rec);
    }

    #[test]
    fn deserialize_cut_rejects_wire_version_1() {
        let n_state = 2;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        // Write a structurally valid cut record but stamp it as wire version 1.
        buf[0] = 1; // old wire version
        buf[1..5].copy_from_slice(&10u32.to_ne_bytes()); // slot_index
        buf[5..9].copy_from_slice(&1u32.to_ne_bytes()); // iteration
        buf[9..13].copy_from_slice(&0u32.to_ne_bytes()); // forward_pass_index
        buf[13] = RECORD_TAG_CUT;
        buf[17..25].copy_from_slice(&1.0f64.to_ne_bytes()); // intercept

        let result = deserialize_cut(&buf, n_state);
        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("wire") || msg.contains("version") || msg.contains("unsupported"),
                    "error message must describe a wire version problem, got: {msg}"
                );
            }
            other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
        }
    }

    #[test]
    fn mixed_payload_round_trips_cut_and_activity_records() {
        let n_state = 2usize;

        // Build 2 cuts.
        let cut0_hdr = CutWireHeader {
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 10.0,
        };
        let cut0_coeffs = [1.5_f64, 2.5];
        let cut1_hdr = CutWireHeader {
            slot_index: 1,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 20.0,
        };
        let cut1_coeffs = [3.5_f64, 4.5];
        let cuts: &[(CutWireHeader, &[f64])] =
            &[(cut0_hdr, &cut0_coeffs), (cut1_hdr, &cut1_coeffs)];

        // Build 3 activity updates.
        let activities = [
            ActivityUpdateRecord {
                slot_index: 0,
                stage_index: 5,
                change: ActivityChange::Deactivate,
            },
            ActivityUpdateRecord {
                slot_index: 1,
                stage_index: 5,
                change: ActivityChange::Reactivate,
            },
            ActivityUpdateRecord {
                slot_index: 2,
                stage_index: 6,
                change: ActivityChange::Deactivate,
            },
        ];

        // Allocate buffer for 2 cuts + 3 activities.
        let cut_size = cut_wire_size(n_state);
        let act_size = activity_update_wire_size();
        let total = 2 * cut_size + 3 * act_size;
        let mut buf = vec![0u8; total];

        let written = serialize_records_to_buffer(&mut buf, cuts, &activities, n_state).unwrap();
        assert_eq!(written, total);

        // Deserialize and verify.
        let mut out_cuts: Vec<(CutWireHeader, Vec<f64>)> = Vec::new();
        let mut out_acts: Vec<ActivityUpdateRecord> = Vec::new();
        let (n_cuts, n_acts) =
            deserialize_records_from_buffer_into(&buf, n_state, &mut out_cuts, &mut out_acts)
                .unwrap();

        assert_eq!(n_cuts, 2, "must decode 2 cut records");
        assert_eq!(n_acts, 3, "must decode 3 activity records");

        // Verify cut content.
        assert_eq!(out_cuts[0].0, cut0_hdr);
        assert_eq!(out_cuts[0].1.len(), n_state);
        assert_eq!(out_cuts[0].1[0].to_bits(), cut0_coeffs[0].to_bits());
        assert_eq!(out_cuts[0].1[1].to_bits(), cut0_coeffs[1].to_bits());
        assert_eq!(out_cuts[1].0, cut1_hdr);
        assert_eq!(out_cuts[1].1[0].to_bits(), cut1_coeffs[0].to_bits());
        assert_eq!(out_cuts[1].1[1].to_bits(), cut1_coeffs[1].to_bits());

        // Verify activity content.
        for (got, expected) in out_acts.iter().zip(activities.iter()) {
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn activity_update_deactivate_tag_byte_is_zero() {
        assert_eq!(ACTIVITY_TAG_DEACTIVATE, 0u8);
        assert_eq!(ACTIVITY_TAG_REACTIVATE, 1u8);
        assert_eq!(RECORD_TAG_CUT, 0u8);
        assert_eq!(RECORD_TAG_ACTIVITY, 1u8);
    }
}
