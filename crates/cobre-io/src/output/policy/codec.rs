//! `FlatBuffers` serializers, wire-format helpers, and deserializers for policy data.
//!
//! ## `FlatBuffers` schema
//!
//! The canonical wire-format description is `schemas/policy.fbs` in this
//! crate (namespace `Cobre.IO.Policy`, tables `StageCuts`, `Cut`,
//! `StageBasis`, `StageStates`). The `*_FIELD_*: u16` slot constants below
//! mirror the `(id: N)` attributes in the schema via the formula
//! `slot = (id + 2) * 2`. They MUST stay in sync; the `flatc-conformance`
//! cargo feature gates a round-trip test in
//! `tests/flatbuffers_schema_conformance.rs` that fails when they diverge.
//!
//! ## Safe raw-byte parsing
//!
//! Reader functions use **safe raw byte parsing** of the `FlatBuffers` wire
//! format instead of the generated `Table::get` API (which is `unsafe fn`).
//! This is required because the workspace forbids `unsafe_code`.

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use super::super::error::OutputError;
use super::records::{
    OwnedPolicyBasisRecord, OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCutRecord,
    StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
};

// ── FlatBuffers vtable slot offsets ──────────────────────────────────────────
//
// Each constant pairs with one `(id: N)` attribute in `schemas/policy.fbs`
// via the formula slot = (id + 2) * 2 (the +2 accounts for the two vtable
// header fields). Editing either side without the other breaks the
// `flatc-conformance` round-trip; the slot 12 gap on `Cut` is the historical
// `domination_count` field, marked `deprecated` in the schema and intentionally
// never reused.

const CUT_FIELD_CUT_ID: u16 = 4;
const CUT_FIELD_SLOT_INDEX: u16 = 6;
const CUT_FIELD_ITERATION: u16 = 8;
const CUT_FIELD_FORWARD_PASS_IDX: u16 = 10;
const CUT_FIELD_INTERCEPT: u16 = 14;
const CUT_FIELD_COEFFICIENTS: u16 = 16;
const CUT_FIELD_STATE_AT_GENERATION: u16 = 18;
const CUT_FIELD_IS_ACTIVE: u16 = 20;

const STAGE_CUTS_FIELD_STAGE_ID: u16 = 4;
const STAGE_CUTS_FIELD_STATE_DIMENSION: u16 = 6;
const STAGE_CUTS_FIELD_CAPACITY: u16 = 8;
const STAGE_CUTS_FIELD_WARM_START_COUNT: u16 = 10;
const STAGE_CUTS_FIELD_CUTS: u16 = 12;
const STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES: u16 = 14;
const STAGE_CUTS_FIELD_POPULATED_COUNT: u16 = 16;

const BASIS_FIELD_STAGE_ID: u16 = 4;
const BASIS_FIELD_ITERATION: u16 = 6;
const BASIS_FIELD_NUM_COLUMNS: u16 = 8;
const BASIS_FIELD_NUM_ROWS: u16 = 10;
const BASIS_FIELD_COLUMN_STATUS: u16 = 12;
const BASIS_FIELD_ROW_STATUS: u16 = 14;
const BASIS_FIELD_NUM_CUT_ROWS: u16 = 16;

const STATES_FIELD_STAGE_ID: u16 = 4;
const STATES_FIELD_STATE_DIMENSION: u16 = 6;
const STATES_FIELD_COUNT: u16 = 8;
const STATES_FIELD_DATA: u16 = 10;

// ── Helper: build a single cut table ─────────────────────────────────────────

/// Build a single cut table inside `builder` and return its offset.
///
/// All nested objects (coefficient vector, `state_at_generation` vector) must be
/// created before the table `start_table`/`end_table` pair, per the `FlatBuffers`
/// requirement that nested objects precede the enclosing table in the buffer.
fn build_cut_table(
    builder: &mut FlatBufferBuilder<'_>,
    cut: &PolicyCutRecord<'_>,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let coefficients_vec = builder.create_vector(cut.coefficients);
    let state_at_gen_vec = builder.create_vector::<f64>(&[]);

    let tab = builder.start_table();

    builder.push_slot_always::<u64>(CUT_FIELD_CUT_ID, cut.cut_id);
    builder.push_slot_always::<u32>(CUT_FIELD_SLOT_INDEX, cut.slot_index);
    builder.push_slot_always::<u32>(CUT_FIELD_ITERATION, cut.iteration);
    builder.push_slot_always::<u32>(CUT_FIELD_FORWARD_PASS_IDX, cut.forward_pass_index);
    builder.push_slot_always::<f64>(CUT_FIELD_INTERCEPT, cut.intercept);
    builder.push_slot_always(CUT_FIELD_COEFFICIENTS, coefficients_vec);
    builder.push_slot_always(CUT_FIELD_STATE_AT_GENERATION, state_at_gen_vec);
    builder.push_slot_always::<bool>(CUT_FIELD_IS_ACTIVE, cut.is_active);

    builder.end_table(tab)
}

// ── Serializers ───────────────────────────────────────────────────────────────

/// Serialize all cuts for one stage into a `FlatBuffers` buffer.
///
/// Produces a buffer containing a root `StageCuts` table. The buffer is ready
/// for writing directly to a `.bin` policy file. Field layout matches the
/// `StageCuts` and `Cut` tables in `schemas/policy.fbs`.
///
/// The function is infallible: the `FlatBuffers` builder API only allocates and
/// writes, never returns errors. Any I/O error is the caller's responsibility.
///
/// # Parameters
///
/// - `stage_id` — stage index (0-based) stored in the root table.
/// - `state_dimension` — number of state variables; determines coefficient vector
///   length per cut.
/// - `capacity` — total preallocated cut slots in the pool.
/// - `warm_start_count` — number of slots `[0..warm_start_count)` loaded from a
///   prior policy.
/// - `cuts` — slice of cut records to serialize; length equals `populated_count`.
/// - `active_cut_indices` — indices of cuts currently active in the LP.
/// - `populated_count` — number of filled slots in the pool.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, serialize_stage_cuts};
///
/// let cut = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);
/// assert!(!buf.is_empty());
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_cuts(
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    warm_start_count: u32,
    cuts: &[PolicyCutRecord<'_>],
    active_cut_indices: &[u32],
    populated_count: u32,
) -> Vec<u8> {
    // Pre-size the builder to avoid reallocation.
    // Each cut occupies roughly: vtable overhead (32 B) + scalar fields (48 B)
    // + coefficient vector (state_dimension * 8 B) + state_at_generation (4 B empty).
    // Plus the StageCuts wrapper and two u32 index vectors.
    let estimated = 64
        + cuts.len() * (96usize + state_dimension as usize * std::mem::size_of::<f64>())
        + std::mem::size_of_val(active_cut_indices);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let cut_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = cuts
        .iter()
        .map(|c| build_cut_table(&mut builder, c))
        .collect();

    // Create the cuts vector from the collected offsets.
    let cuts_vec = builder.create_vector(&cut_offsets);

    // Create the active_cut_indices vector.
    let active_vec = builder.create_vector(active_cut_indices);

    // Build the root StageCuts table.
    let root = builder.start_table();

    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STAGE_ID, stage_id);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STATE_DIMENSION, state_dimension);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_CAPACITY, capacity);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_WARM_START_COUNT, warm_start_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_CUTS, cuts_vec);
    builder.push_slot_always(STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES, active_vec);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_POPULATED_COUNT, populated_count);

    let root_offset = builder.end_table(root);
    builder.finish_minimal(root_offset);

    builder.finished_data().to_vec()
}

/// Serialize one stage's solver basis into a `FlatBuffers` buffer.
///
/// Produces a buffer containing a root `StageBasis` table. The buffer is ready
/// for writing directly to a `.bin` policy file under `basis/`. Field layout
/// matches the `StageBasis` table in `schemas/policy.fbs`.
///
/// The `num_columns` and `num_rows` fields are inferred from the status slice
/// lengths and do not need to be supplied separately.
///
/// The function is infallible: the `FlatBuffers` builder API only allocates and
/// writes, never returns errors.
///
/// # Parameters
///
/// - `record` — a reference to the basis record to serialize.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyBasisRecord, serialize_stage_basis};
///
/// let record = PolicyBasisRecord {
///     stage_id: 0,
///     iteration: 5,
///     column_status: &[0, 1, 2],
///     row_status: &[1, 1, 0, 0],
///     num_cut_rows: 2,
/// };
/// let buf = serialize_stage_basis(&record);
/// assert!(!buf.is_empty());
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_basis(record: &PolicyBasisRecord<'_>) -> Vec<u8> {
    // Pre-size: scalar fields (~32 B) + two byte vectors + headers.
    let estimated =
        64 + std::mem::size_of_val(record.column_status) + std::mem::size_of_val(record.row_status);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    // Create nested vectors before opening the table.
    let col_vec = builder.create_vector(record.column_status);
    let row_vec = builder.create_vector(record.row_status);

    let root = builder.start_table();

    builder.push_slot_always::<u32>(BASIS_FIELD_STAGE_ID, record.stage_id);
    builder.push_slot_always::<u32>(BASIS_FIELD_ITERATION, record.iteration);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_COLUMNS, record.column_status.len() as u32);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_ROWS, record.row_status.len() as u32);
    builder.push_slot_always(BASIS_FIELD_COLUMN_STATUS, col_vec);
    builder.push_slot_always(BASIS_FIELD_ROW_STATUS, row_vec);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_CUT_ROWS, record.num_cut_rows);

    let root_offset = builder.end_table(root);
    builder.finish_minimal(root_offset);

    builder.finished_data().to_vec()
}

/// Serialize one stage's visited states into a `FlatBuffers` buffer.
///
/// Produces a buffer containing a root `StageStates` table with fields
/// `stage_id`, `state_dimension`, `count`, and `data` (a flat `[f64]`
/// vector). The buffer is ready for writing directly to a `.bin` policy
/// file under `states/`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_states(payload: &StageStatesPayload<'_>) -> Vec<u8> {
    let estimated = 64 + std::mem::size_of_val(payload.data);
    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let data_vec = builder.create_vector(payload.data);

    let root = builder.start_table();
    builder.push_slot_always::<u32>(STATES_FIELD_STAGE_ID, payload.stage_id);
    builder.push_slot_always::<u32>(STATES_FIELD_STATE_DIMENSION, payload.state_dimension);
    builder.push_slot_always::<u32>(STATES_FIELD_COUNT, payload.count);
    builder.push_slot_always(STATES_FIELD_DATA, data_vec);

    let root_offset = builder.end_table(root);
    builder.finish_minimal(root_offset);

    builder.finished_data().to_vec()
}

// ── Safe FlatBuffers wire-format helpers ─────────────────────────────────────
//
// All helpers return `Option` so callers can propagate truncation / corruption
// errors without panicking. The `resolve_*` functions follow the FlatBuffers
// specification exactly:
//
//   Buffer layout (finish_minimal):
//     bytes[0..4]  = u32 LE root_offset — byte offset from position 0 to root table
//     ...builder data (written right-to-left)...
//     vtable  = [u16 vtable_size][u16 table_size][u16 field0][u16 field1]...
//     table   = [i32 soffset_to_vtable][...inline field data...]
//
//   soffset_to_vtable at table_pos:
//     vtable_pos = table_pos - (i32 at table_pos)
//
//   Field data for field with vtable slot `slot`:
//     field_data_offset_from_table_start = u16 at vtable[slot]
//     (0 means field absent)
//     actual data at: table_pos + field_data_offset_from_table_start
//
//   Nested table / vector fields store a u32 forward uoffset at their data position:
//     nested_pos = field_data_pos + u32_at(field_data_pos)

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> Option<u16> {
    let bytes = buf.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline]
fn read_i32_le(buf: &[u8], offset: usize) -> Option<i32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> Option<u32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline]
fn read_u64_le(buf: &[u8], offset: usize) -> Option<u64> {
    let bytes = buf.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

#[inline]
fn read_f64_le(buf: &[u8], offset: usize) -> Option<f64> {
    read_u64_le(buf, offset).map(f64::from_bits)
}

#[inline]
fn read_bool_byte(buf: &[u8], offset: usize) -> Option<bool> {
    buf.get(offset).map(|&b| b != 0)
}

/// Resolve the root table position from a finished `FlatBuffers` buffer.
///
/// Returns the byte offset of the root table within `buf`.
fn resolve_root(buf: &[u8]) -> Option<usize> {
    let offset = read_u32_le(buf, 0)? as usize;
    // The root offset must point inside the buffer (at minimum for the soffset).
    if offset.checked_add(4)? > buf.len() {
        return None;
    }
    Some(offset)
}

/// Resolve the vtable position for the table at `table_pos`.
///
/// Returns the byte offset of the vtable within `buf`.
fn resolve_vtable_pos(buf: &[u8], table_pos: usize) -> Option<usize> {
    let soffset = read_i32_le(buf, table_pos)?;
    // vtable_pos = table_pos - soffset (soffset is signed; positive = vtable before table).
    // Avoid lossy `as i64` casts (clippy::cast_possible_wrap / cast_possible_truncation).
    let vtable_pos = if soffset >= 0 {
        // Vtable precedes the table: table_pos - soffset (as a non-negative offset).
        table_pos.checked_sub(u32::try_from(soffset).ok()? as usize)?
    } else {
        // Vtable follows the table: table_pos + |soffset|.
        let abs = u32::try_from(soffset.wrapping_neg()).ok()? as usize;
        table_pos.checked_add(abs)?
    };
    if vtable_pos.checked_add(4)? > buf.len() {
        return None;
    }
    Some(vtable_pos)
}

/// Read the data offset for field slot `slot` from the vtable at `vtable_pos`.
///
/// Returns `None` if the slot is beyond the vtable, or `Some(0)` if the field
/// is absent (the `FlatBuffers` convention for optional fields).
fn field_data_offset(buf: &[u8], vtable_pos: usize, slot: u16) -> Option<u16> {
    let vtable_size = read_u16_le(buf, vtable_pos)?;
    let slot_pos = vtable_pos.checked_add(slot as usize)?;
    if slot_pos.checked_add(2)? > vtable_pos.checked_add(vtable_size as usize)? {
        // Slot is past end of vtable — field was added in a later schema version.
        return Some(0);
    }
    read_u16_le(buf, slot_pos)
}

/// Resolve the absolute position of field `slot` data in a table at `table_pos`.
///
/// Returns `None` if the field is absent (vtable offset is 0) or if the buffer
/// is truncated.
fn field_pos(buf: &[u8], table_pos: usize, vtable_pos: usize, slot: u16) -> Option<usize> {
    let data_off = field_data_offset(buf, vtable_pos, slot)?;
    if data_off == 0 {
        return None; // field absent
    }
    table_pos.checked_add(data_off as usize)
}

/// Follow a `uoffset` stored at `pos` to reach a nested table or vector.
///
/// `FlatBuffers` stores forward offsets: the referenced object is at
/// `pos + u32_at(pos)`. The offset is relative to the position of the u32 itself.
fn follow_uoffset(buf: &[u8], pos: usize) -> Option<usize> {
    let off = read_u32_le(buf, pos)?;
    pos.checked_add(off as usize)
}

/// Read a `f32` vector stored at `vec_pos` and return its elements as `f64`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 4` bytes.
/// This function is not used currently but kept for completeness.
// Rationale: the f32 vector reader is the symmetric counterpart to `read_f64_vector`; retaining
// it keeps the codec complete against the full FlatBuffers type palette and avoids re-deriving
// the safe byte-level parsing pattern from scratch when an f32 field is added to the schema.
#[allow(dead_code)]
fn read_f32_vector_as_f64(buf: &[u8], vec_pos: usize) -> Option<Vec<f64>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let pos = data_start + i * 4;
        let bits = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        out.push(f64::from(f32::from_bits(bits)));
    }
    Some(out)
}

/// Read a `f64` vector stored at `vec_pos`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 8` bytes.
fn read_f64_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<f64>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(8)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let pos = data_start + i * 8;
        out.push(read_f64_le(buf, pos)?);
    }
    Some(out)
}

/// Read a `u8` vector stored at `vec_pos`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 1` bytes.
fn read_u8_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u8>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    if data_end > buf.len() {
        return None;
    }
    Some(buf[data_start..data_end].to_vec())
}

/// Read a vector of nested tables stored at `vec_pos`.
///
/// Returns a `Vec` of absolute buffer positions, one per element. Each element
/// stores a `u32` uoffset from its own position to the nested table.
fn read_table_vector_positions(buf: &[u8], vec_pos: usize) -> Option<Vec<usize>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut positions = Vec::with_capacity(len);
    for i in 0..len {
        let elem_pos = data_start + i * 4;
        let nested_pos = follow_uoffset(buf, elem_pos)?;
        positions.push(nested_pos);
    }
    Some(positions)
}

// ── Deserializers ─────────────────────────────────────────────────────────────

/// Deserialize a `StageCuts` `FlatBuffers` buffer into an owned [`StageCutsReadResult`].
///
/// Reads the root `StageCuts` table and each nested cut record table using safe
/// raw byte parsing of the `FlatBuffers` wire format. No `unsafe` code is used.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, serialize_stage_cuts, deserialize_stage_cuts};
///
/// let cut = PolicyCutRecord {
///     cut_id: 7,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 1,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(2, 3, 100, 0, &[cut], &[0], 1);
/// let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
/// assert_eq!(result.stage_id, 2);
/// assert_eq!(result.cuts.len(), 1);
/// assert_eq!(result.cuts[0].cut_id, 7);
/// assert_eq!(result.cuts[0].coefficients, &[1.0, 2.0, 3.0]);
/// ```
pub fn deserialize_stage_cuts(buf: &[u8]) -> Result<StageCutsReadResult, OutputError> {
    let ctx = "stage_cuts";

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    // Read scalar fields from StageCuts root table.
    let stage_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let state_dimension = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_STATE_DIMENSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let capacity = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CAPACITY)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let warm_start_count = field_pos(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_WARM_START_COUNT,
    )
    .and_then(|p| read_u32_le(buf, p))
    .unwrap_or(0);

    let populated_count = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_POPULATED_COUNT)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    // Read the cuts vector of nested tables.
    let cuts = if let Some(cuts_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CUTS)
    {
        let vec_pos = follow_uoffset(buf, cuts_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for cuts vector"))?;

        let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "cuts vector header truncated or corrupt")
        })?;

        let mut out = Vec::with_capacity(nested_positions.len());
        for (idx, &cut_table_pos) in nested_positions.iter().enumerate() {
            let cut = deserialize_cut_table(buf, cut_table_pos).ok_or_else(|| {
                OutputError::serialization(ctx, format!("cut table {idx} truncated or corrupt"))
            })?;
            out.push(cut);
        }
        out
    } else {
        Vec::new()
    };

    Ok(StageCutsReadResult {
        stage_id,
        state_dimension,
        capacity,
        warm_start_count,
        populated_count,
        cuts,
    })
}

/// Deserialize a single cut record nested table at `cut_table_pos`.
fn deserialize_cut_table(buf: &[u8], cut_table_pos: usize) -> Option<OwnedPolicyCutRecord> {
    let vtable_pos = resolve_vtable_pos(buf, cut_table_pos)?;

    let cut_id = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_CUT_ID)
        .and_then(|p| read_u64_le(buf, p))
        .unwrap_or(0);

    let slot_index = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_SLOT_INDEX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let iteration = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_ITERATION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let forward_pass_index = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_FORWARD_PASS_IDX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let intercept = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_INTERCEPT)
        .and_then(|p| read_f64_le(buf, p))
        .unwrap_or(0.0);

    let coefficients = if let Some(coeff_field_pos) =
        field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_COEFFICIENTS)
    {
        let vec_pos = follow_uoffset(buf, coeff_field_pos)?;
        read_f64_vector(buf, vec_pos)?
    } else {
        Vec::new()
    };

    let is_active = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_IS_ACTIVE)
        .and_then(|p| read_bool_byte(buf, p))
        .unwrap_or(false);

    Some(OwnedPolicyCutRecord {
        cut_id,
        slot_index,
        iteration,
        forward_pass_index,
        intercept,
        coefficients,
        is_active,
    })
}

/// Deserialize a `StageBasis` `FlatBuffers` buffer into an owned [`OwnedPolicyBasisRecord`].
///
/// Reads the root `StageBasis` table using safe raw byte parsing. No `unsafe` code is used.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyBasisRecord, serialize_stage_basis, deserialize_stage_basis};
///
/// let record = PolicyBasisRecord {
///     stage_id: 0,
///     iteration: 5,
///     column_status: &[0, 1, 2],
///     row_status: &[1, 1, 0, 0],
///     num_cut_rows: 2,
/// };
/// let buf = serialize_stage_basis(&record);
/// let owned = deserialize_stage_basis(&buf).expect("round-trip must succeed");
/// assert_eq!(owned.stage_id, 0);
/// assert_eq!(owned.column_status, &[0, 1, 2]);
/// assert_eq!(owned.row_status, &[1, 1, 0, 0]);
/// ```
pub fn deserialize_stage_basis(buf: &[u8]) -> Result<OwnedPolicyBasisRecord, OutputError> {
    let ctx = "stage_basis";

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let stage_id = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let iteration = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_ITERATION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let column_status = if let Some(col_field_pos) =
        field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_COLUMN_STATUS)
    {
        let vec_pos = follow_uoffset(buf, col_field_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "invalid uoffset for column_status vector")
        })?;
        read_u8_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "column_status vector truncated"))?
    } else {
        Vec::new()
    };

    let row_status = if let Some(row_field_pos) =
        field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_ROW_STATUS)
    {
        let vec_pos = follow_uoffset(buf, row_field_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "invalid uoffset for row_status vector")
        })?;
        read_u8_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "row_status vector truncated"))?
    } else {
        Vec::new()
    };

    let num_cut_rows = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_NUM_CUT_ROWS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    Ok(OwnedPolicyBasisRecord {
        stage_id,
        iteration,
        column_status,
        row_status,
        num_cut_rows,
    })
}

/// Deserialize one stage's visited states from a `FlatBuffers` buffer.
///
/// Parses the `StageStates` root table produced by [`serialize_stage_states`]
/// and returns an owned [`StageStatesReadResult`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated or
/// has an invalid wire format.
pub fn deserialize_stage_states(buf: &[u8]) -> Result<StageStatesReadResult, OutputError> {
    let ctx = "stage_states";

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let stage_id = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let state_dimension = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_STATE_DIMENSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let count = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_COUNT)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let data = if let Some(data_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STATES_FIELD_DATA)
    {
        let vec_pos = follow_uoffset(buf, data_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for data vector"))?;
        read_f64_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "data vector truncated"))?
    } else {
        Vec::new()
    };

    Ok(StageStatesReadResult {
        stage_id,
        state_dimension,
        count,
        data,
    })
}

/// Read all `*.bin` files from `dir`, deserialize each with `deser_fn`, and return a `Vec`.
///
/// Files are enumerated via [`std::fs::read_dir`]. The returned `Vec` is unsorted —
/// callers should sort by the appropriate `stage_id` field after this call.
///
/// If `dir` exists but contains no `.bin` files, an empty `Vec` is returned.
pub(super) fn read_sorted_bin_files<T, F>(
    dir: &std::path::Path,
    ctx: &str,
    deser_fn: F,
) -> Result<Vec<T>, OutputError>
where
    F: Fn(&[u8]) -> Result<T, OutputError>,
{
    let entries = std::fs::read_dir(dir).map_err(|e| OutputError::io(dir, e))?;

    let mut results = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| OutputError::io(dir, e))?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if !name.ends_with(".bin") {
            continue;
        }
        let file_path = entry.path();
        let bytes = std::fs::read(&file_path).map_err(|e| OutputError::io(&file_path, e))?;
        let record = deser_fn(&bytes).map_err(|e| {
            // Re-wrap with file context for better diagnostics.
            OutputError::serialization(
                ctx,
                format!("failed to deserialize {}: {e}", file_path.display()),
            )
        })?;
        results.push(record);
    }
    Ok(results)
}
