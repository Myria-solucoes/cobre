//! `FlatBuffers` serializers, wire-format helpers, and deserializers for policy data.
//!
//! Wire layout follows the policy schema specification (spec SS3.2).
//!
//! This module is the sole owner of the policy `FlatBuffers` byte layout. The
//! `*_FIELD_*: u16` slot constants below mirror the `(id: N)` attributes in
//! `schemas/policy.fbs` via `slot = (id + 2) * 2` and MUST stay in sync; the
//! `flatc-conformance` feature gates the round-trip test in
//! `tests/flatbuffers_schema_conformance.rs` that fails when they diverge.
//!
//! Reader functions parse raw bytes rather than the generated `Table::get` API
//! (an `unsafe fn`) because the workspace forbids `unsafe_code`.

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use super::super::error::OutputError;
use super::records::{
    CheckpointManifest, ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION,
    GraphManifest, ManifestEdge, ManifestNode, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCutRecord, ProducerBlock, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL,
    STAGE_CUTS_NODE_ID_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, StageCutsPayload,
    StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
};

use std::path::Path;

/// `FlatBuffers` `file_identifier` for every policy artifact (`schemas/policy.fbs`).
/// Written by `finish(root, Some(POLICY_FILE_IDENTIFIER))` and required on read:
/// a buffer lacking it (a pre-0.14 `finish_minimal` artifact) is rejected.
const POLICY_FILE_IDENTIFIER: &str = "CBVF";

// ── FlatBuffers vtable slot offsets ──────────────────────────────────────────
//
// slot = (id + 2) * 2. Two slots are permanently burned and never read or
// written here — reusing either would diverge the hand-written layout from the
// schema's `deprecated` placeholder: `AffinePiece` id 7 (slot 18), the former
// `state_at_generation`; and `EntitySlot` id 4 (slot 12), the former
// month-integer `delivery_anchor` replaced by `delivery_date` at id 5 (slot 14).

const CUT_FIELD_CUT_ID: u16 = 4;
const CUT_FIELD_SLOT_INDEX: u16 = 6;
const CUT_FIELD_ITERATION: u16 = 8;
const CUT_FIELD_FORWARD_PASS_IDX: u16 = 10;
const CUT_FIELD_INTERCEPT: u16 = 12;
const CUT_FIELD_COEFFICIENTS: u16 = 14;
const CUT_FIELD_IS_ACTIVE: u16 = 16;

const ENTITY_SLOT_FIELD_ENTITY_TYPE: u16 = 4;
const ENTITY_SLOT_FIELD_ENTITY_ID: u16 = 6;
const ENTITY_SLOT_FIELD_SUBINDEX: u16 = 8;
const ENTITY_SLOT_FIELD_WAS_ACTIVE: u16 = 10;
const ENTITY_SLOT_FIELD_DELIVERY_DATE: u16 = 14;

const STAGE_CUTS_FIELD_STAGE_ID: u16 = 4;
const STAGE_CUTS_FIELD_STATE_DIMENSION: u16 = 6;
const STAGE_CUTS_FIELD_CAPACITY: u16 = 8;
const STAGE_CUTS_FIELD_WARM_START_COUNT: u16 = 10;
const STAGE_CUTS_FIELD_CUTS: u16 = 12;
const STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES: u16 = 14;
const STAGE_CUTS_FIELD_POPULATED_COUNT: u16 = 16;
const STAGE_CUTS_FIELD_ENTITY_MANIFEST: u16 = 18;
const STAGE_CUTS_FIELD_COST_SCALE_FACTOR: u16 = 20;
const STAGE_CUTS_FIELD_NODE_ID: u16 = 22;
const STAGE_CUTS_FIELD_GRAPH_STAGE_ID: u16 = 24;

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
const STATES_FIELD_ENTITY_MANIFEST: u16 = 12;
const STATES_FIELD_NODE_ID: u16 = 14;

// CheckpointManifest ids start fresh at 0; its own vtable is distinct from the
// AffinePiece/EntitySlot vtables, so its id 4 / id 7 collide with no burned slot.
const MANIFEST_FIELD_FORMAT_VERSION: u16 = 4;
const MANIFEST_FIELD_COBRE_VERSION: u16 = 6;
const MANIFEST_FIELD_CREATED_AT: u16 = 8;
const MANIFEST_FIELD_NUM_STAGES: u16 = 10;
const MANIFEST_FIELD_N_POOLS: u16 = 12;
const MANIFEST_FIELD_NODES: u16 = 14;
const MANIFEST_FIELD_EDGES: u16 = 16;
const MANIFEST_FIELD_COMPLETED_ITERATIONS: u16 = 18;
const MANIFEST_FIELD_FINAL_LOWER_BOUND: u16 = 20;
const MANIFEST_FIELD_BEST_UPPER_BOUND: u16 = 22;
const MANIFEST_FIELD_MAX_ITERATIONS: u16 = 24;
const MANIFEST_FIELD_FORWARD_PASSES: u16 = 26;
const MANIFEST_FIELD_WARM_START_CUTS: u16 = 28;
const MANIFEST_FIELD_WARM_START_COUNTS: u16 = 30;
const MANIFEST_FIELD_RNG_SEED: u16 = 32;
const MANIFEST_FIELD_TOTAL_VISITED_STATES: u16 = 34;
const MANIFEST_FIELD_TRAINING_BLOCK_MODE: u16 = 36;
const MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE: u16 = 38;
const MANIFEST_FIELD_COST_SCALE_FACTOR: u16 = 40;

const MANIFEST_NODE_FIELD_ID: u16 = 4;
const MANIFEST_NODE_FIELD_STAGE_ID: u16 = 6;
const MANIFEST_NODE_FIELD_POOL_ID: u16 = 8;

const MANIFEST_EDGE_FIELD_SOURCE_ID: u16 = 4;
const MANIFEST_EDGE_FIELD_TARGET_ID: u16 = 6;
const MANIFEST_EDGE_FIELD_PROBABILITY: u16 = 8;

/// The coefficient vector must be created before the `start_table`/`end_table`
/// pair — `FlatBuffers` requires nested objects to precede the enclosing table
/// in the buffer.
fn build_cut_table(
    builder: &mut FlatBufferBuilder<'_>,
    piece: &PolicyCutRecord<'_>,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let coefficients_vec = builder.create_vector(piece.coefficients);

    let tab = builder.start_table();

    builder.push_slot_always::<u64>(CUT_FIELD_CUT_ID, piece.cut_id);
    builder.push_slot_always::<u32>(CUT_FIELD_SLOT_INDEX, piece.slot_index);
    builder.push_slot_always::<u32>(CUT_FIELD_ITERATION, piece.iteration);
    builder.push_slot_always::<u32>(CUT_FIELD_FORWARD_PASS_IDX, piece.forward_pass_index);
    builder.push_slot_always::<f64>(CUT_FIELD_INTERCEPT, piece.intercept);
    builder.push_slot_always(CUT_FIELD_COEFFICIENTS, coefficients_vec);
    builder.push_slot_always::<bool>(CUT_FIELD_IS_ACTIVE, piece.is_active);

    builder.end_table(tab)
}

/// Build one `EntitySlot` nested table. Has no inner vector, so — unlike
/// [`build_cut_table`] — nothing precedes the `start_table`/`end_table` pair.
fn build_entity_slot_table(
    builder: &mut FlatBufferBuilder<'_>,
    slot: &EntitySlot,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<u8>(ENTITY_SLOT_FIELD_ENTITY_TYPE, slot.entity_type);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_ENTITY_ID, slot.entity_id);
    builder.push_slot_always::<u32>(ENTITY_SLOT_FIELD_SUBINDEX, slot.subindex);
    builder.push_slot_always::<bool>(ENTITY_SLOT_FIELD_WAS_ACTIVE, slot.was_active);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_DELIVERY_DATE, slot.delivery_date);

    builder.end_table(tab)
}

/// Build one `ManifestNode` nested table. No inner vector, so nothing precedes
/// the `start_table`/`end_table` pair.
fn build_manifest_node_table(
    builder: &mut FlatBufferBuilder<'_>,
    node: &ManifestNode,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<i32>(MANIFEST_NODE_FIELD_ID, node.id);
    builder.push_slot_always::<i32>(MANIFEST_NODE_FIELD_STAGE_ID, node.stage_id);
    builder.push_slot_always::<u32>(MANIFEST_NODE_FIELD_POOL_ID, node.pool_id);

    builder.end_table(tab)
}

/// Build one `ManifestEdge` nested table.
fn build_manifest_edge_table(
    builder: &mut FlatBufferBuilder<'_>,
    edge: &ManifestEdge,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<i32>(MANIFEST_EDGE_FIELD_SOURCE_ID, edge.source_id);
    builder.push_slot_always::<i32>(MANIFEST_EDGE_FIELD_TARGET_ID, edge.target_id);
    builder.push_slot_always::<f64>(MANIFEST_EDGE_FIELD_PROBABILITY, edge.probability);

    builder.end_table(tab)
}

/// Reject a buffer whose leading `file_identifier` is not [`POLICY_FILE_IDENTIFIER`].
///
/// `finish(root, Some(id))` writes the 4-byte identifier at bytes `4..8` (right
/// after the 4-byte root uoffset), so a pre-0.14 `finish_minimal` buffer carries
/// none and is rejected here before any field is decoded.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] when the identifier is absent or wrong.
fn check_file_identifier(buf: &[u8], ctx: &str) -> Result<(), OutputError> {
    if buf.get(4..8) == Some(POLICY_FILE_IDENTIFIER.as_bytes()) {
        Ok(())
    } else {
        Err(OutputError::serialization(
            ctx,
            format!(
                "missing FlatBuffers file_identifier {POLICY_FILE_IDENTIFIER:?}; not a 0.14+ policy artifact"
            ),
        ))
    }
}

// ── Serializers ───────────────────────────────────────────────────────────────

/// Serialize all cuts for one stage into a root `StageCuts` `FlatBuffers` buffer,
/// ready to write directly to a `.bin` policy file.
///
/// Infallible: the builder only allocates and writes. Any I/O error is the
/// caller's responsibility.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, StageCutsPayload, serialize_stage_cuts};
///
/// let piece = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(&StageCutsPayload {
///     stage_id: 0,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[piece],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
///     cost_scale_factor: 1_000_000.0,
///     node_id: 0,
///     graph_stage_id: 0,
/// });
/// assert!(!buf.is_empty());
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_cuts(payload: &StageCutsPayload<'_>) -> Vec<u8> {
    let estimated = 64
        + payload.cuts.len()
            * (96usize + payload.state_dimension as usize * std::mem::size_of::<f64>())
        + std::mem::size_of_val(payload.active_cut_indices)
        + payload.entity_manifest.len() * 32usize;

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let cut_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .cuts
        .iter()
        .map(|c| build_cut_table(&mut builder, c))
        .collect();
    let manifest_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .entity_manifest
        .iter()
        .map(|s| build_entity_slot_table(&mut builder, s))
        .collect();

    let cuts_vec = builder.create_vector(&cut_offsets);
    let active_vec = builder.create_vector(payload.active_cut_indices);
    let manifest_vec = builder.create_vector(&manifest_offsets);

    let root = builder.start_table();

    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STAGE_ID, payload.stage_id);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STATE_DIMENSION, payload.state_dimension);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_CAPACITY, payload.capacity);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_WARM_START_COUNT, payload.warm_start_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_CUTS, cuts_vec);
    builder.push_slot_always(STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES, active_vec);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_POPULATED_COUNT, payload.populated_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_ENTITY_MANIFEST, manifest_vec);
    builder.push_slot_always::<f64>(
        STAGE_CUTS_FIELD_COST_SCALE_FACTOR,
        payload.cost_scale_factor,
    );
    builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_NODE_ID, payload.node_id);
    builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_GRAPH_STAGE_ID, payload.graph_stage_id);

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder.finished_data().to_vec()
}

/// Serialize one stage's solver basis into a root `StageBasis` `FlatBuffers`
/// buffer, ready to write directly to a `.bin` policy file under `basis/`.
///
/// `num_columns` and `num_rows` are inferred from the status slice lengths, not
/// supplied separately. Infallible: the builder only allocates and writes.
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
    let estimated =
        64 + std::mem::size_of_val(record.column_status) + std::mem::size_of_val(record.row_status);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    // Nested vectors must be created before opening the table.
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
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder.finished_data().to_vec()
}

/// Serialize one stage's visited states into a root `StageStates` `FlatBuffers`
/// buffer, ready to write directly to a `.bin` policy file under `states/`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_states(payload: &StageStatesPayload<'_>) -> Vec<u8> {
    let estimated =
        64 + std::mem::size_of_val(payload.data) + payload.entity_manifest.len() * 32usize;
    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let manifest_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .entity_manifest
        .iter()
        .map(|s| build_entity_slot_table(&mut builder, s))
        .collect();

    let data_vec = builder.create_vector(payload.data);
    let manifest_vec = builder.create_vector(&manifest_offsets);

    let root = builder.start_table();
    builder.push_slot_always::<u32>(STATES_FIELD_STAGE_ID, payload.stage_id);
    builder.push_slot_always::<u32>(STATES_FIELD_STATE_DIMENSION, payload.state_dimension);
    builder.push_slot_always::<u32>(STATES_FIELD_COUNT, payload.count);
    builder.push_slot_always(STATES_FIELD_DATA, data_vec);
    builder.push_slot_always(STATES_FIELD_ENTITY_MANIFEST, manifest_vec);
    builder.push_slot_always::<i32>(STATES_FIELD_NODE_ID, payload.node_id);

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder.finished_data().to_vec()
}

/// Serialize a [`CheckpointManifest`] into a root `CheckpointManifest`
/// `FlatBuffers` buffer.
///
/// The two `Option<f64>` provenance fields (`best_upper_bound`,
/// `cost_scale_factor`) are written only when `Some`, so absence round-trips as
/// `None` rather than a spurious `0.0`. Infallible: the builder only allocates
/// and writes.
#[must_use]
pub fn serialize_checkpoint_manifest(manifest: &CheckpointManifest) -> Vec<u8> {
    let graph = &manifest.graph_manifest;
    let producer = &manifest.producer;

    let estimated = 128
        + graph.nodes.len() * 32
        + graph.edges.len() * 40
        + manifest.cobre_version.len()
        + manifest.created_at.len()
        + producer.training_block_mode.len()
        + producer.warm_start_counts.len() * std::mem::size_of::<u32>()
        + producer
            .training_block_mode_per_stage
            .iter()
            .map(|s| s.len() + 8)
            .sum::<usize>();

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let node_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = graph
        .nodes
        .iter()
        .map(|n| build_manifest_node_table(&mut builder, n))
        .collect();
    let edge_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = graph
        .edges
        .iter()
        .map(|e| build_manifest_edge_table(&mut builder, e))
        .collect();

    let cobre_version = builder.create_string(&manifest.cobre_version);
    let created_at = builder.create_string(&manifest.created_at);
    let training_block_mode = builder.create_string(&producer.training_block_mode);
    let per_stage_offsets: Vec<WIPOffset<&str>> = producer
        .training_block_mode_per_stage
        .iter()
        .map(|s| builder.create_string(s))
        .collect();

    let nodes_vec = builder.create_vector(&node_offsets);
    let edges_vec = builder.create_vector(&edge_offsets);
    let warm_start_counts_vec = builder.create_vector(producer.warm_start_counts.as_slice());
    let per_stage_vec = builder.create_vector(&per_stage_offsets);

    let root = builder.start_table();

    builder.push_slot_always::<u32>(MANIFEST_FIELD_FORMAT_VERSION, manifest.format_version);
    builder.push_slot_always(MANIFEST_FIELD_COBRE_VERSION, cobre_version);
    builder.push_slot_always(MANIFEST_FIELD_CREATED_AT, created_at);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_NUM_STAGES, manifest.num_stages);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_N_POOLS, graph.n_pools);
    builder.push_slot_always(MANIFEST_FIELD_NODES, nodes_vec);
    builder.push_slot_always(MANIFEST_FIELD_EDGES, edges_vec);
    builder.push_slot_always::<u32>(
        MANIFEST_FIELD_COMPLETED_ITERATIONS,
        producer.completed_iterations,
    );
    builder.push_slot_always::<f64>(MANIFEST_FIELD_FINAL_LOWER_BOUND, producer.final_lower_bound);
    if let Some(best) = producer.best_upper_bound {
        builder.push_slot_always::<f64>(MANIFEST_FIELD_BEST_UPPER_BOUND, best);
    }
    builder.push_slot_always::<u32>(MANIFEST_FIELD_MAX_ITERATIONS, producer.max_iterations);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_FORWARD_PASSES, producer.forward_passes);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_WARM_START_CUTS, producer.warm_start_cuts);
    builder.push_slot_always(MANIFEST_FIELD_WARM_START_COUNTS, warm_start_counts_vec);
    builder.push_slot_always::<u64>(MANIFEST_FIELD_RNG_SEED, producer.rng_seed);
    builder.push_slot_always::<u64>(
        MANIFEST_FIELD_TOTAL_VISITED_STATES,
        producer.total_visited_states,
    );
    builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE, training_block_mode);
    builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE, per_stage_vec);
    if let Some(csf) = producer.cost_scale_factor {
        builder.push_slot_always::<f64>(MANIFEST_FIELD_COST_SCALE_FACTOR, csf);
    }

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder.finished_data().to_vec()
}

// ── Safe FlatBuffers wire-format helpers ─────────────────────────────────────
//
// All helpers return `Option` so callers can propagate truncation / corruption
// errors without panicking. The `resolve_*` functions follow the FlatBuffers
// specification exactly:
//
//   Buffer layout (finish with file_identifier):
//     bytes[0..4]  = u32 LE root_offset — byte offset from position 0 to root table
//     bytes[4..8]  = 4-byte file_identifier ("CBVF"), checked before any decode
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
//
//   Vector at vec_pos: [u32 length][length × element_size bytes of element data].

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

fn resolve_root(buf: &[u8]) -> Option<usize> {
    let offset = read_u32_le(buf, 0)? as usize;
    if offset.checked_add(4)? > buf.len() {
        return None;
    }
    Some(offset)
}

fn resolve_vtable_pos(buf: &[u8], table_pos: usize) -> Option<usize> {
    let soffset = read_i32_le(buf, table_pos)?;
    // soffset is signed: positive = vtable precedes the table, negative = follows.
    let vtable_pos = if soffset >= 0 {
        table_pos.checked_sub(u32::try_from(soffset).ok()? as usize)?
    } else {
        let abs = u32::try_from(soffset.wrapping_neg()).ok()? as usize;
        table_pos.checked_add(abs)?
    };
    if vtable_pos.checked_add(4)? > buf.len() {
        return None;
    }
    Some(vtable_pos)
}

/// `Some(0)` means the field is absent — the `FlatBuffers` optional-field
/// convention. A slot past the vtable end is a field added in a later schema
/// version (forward compatibility): treat it as absent, not as an error.
fn field_data_offset(buf: &[u8], vtable_pos: usize, slot: u16) -> Option<u16> {
    let vtable_size = read_u16_le(buf, vtable_pos)?;
    let slot_pos = vtable_pos.checked_add(slot as usize)?;
    if slot_pos.checked_add(2)? > vtable_pos.checked_add(vtable_size as usize)? {
        return Some(0);
    }
    read_u16_le(buf, slot_pos)
}

fn field_pos(buf: &[u8], table_pos: usize, vtable_pos: usize, slot: u16) -> Option<usize> {
    let data_off = field_data_offset(buf, vtable_pos, slot)?;
    if data_off == 0 {
        return None; // field absent
    }
    table_pos.checked_add(data_off as usize)
}

/// `FlatBuffers` uoffsets are forward and self-relative: the referenced nested
/// table or vector is at `pos + u32_at(pos)`, not `0 + u32_at(pos)`.
fn follow_uoffset(buf: &[u8], pos: usize) -> Option<usize> {
    let off = read_u32_le(buf, pos)?;
    pos.checked_add(off as usize)
}

/// Read a `f32` vector stored at `vec_pos` and return its elements as `f64`.
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

fn read_u8_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u8>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    if data_end > buf.len() {
        return None;
    }
    Some(buf[data_start..data_end].to_vec())
}

fn read_u32_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u32>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_u32_le(buf, data_start + i * 4)?);
    }
    Some(out)
}

/// Read a length-prefixed `FlatBuffers` UTF-8 string at `str_pos`.
fn read_string(buf: &[u8], str_pos: usize) -> Option<String> {
    let len = read_u32_le(buf, str_pos)? as usize;
    let data_start = str_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    let bytes = buf.get(data_start..data_end)?;
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}

/// Returns one absolute buffer position per element; each element stores a `u32`
/// uoffset from its own position to the nested table (self-relative, not from 0).
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

/// Read an `entity_manifest` table-vector at vtable `slot`, mirroring the cuts
/// read block. An absent field yields an empty `Vec` (graceful absence).
fn read_entity_manifest(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<EntitySlot>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos).ok_or_else(|| {
        OutputError::serialization(ctx, "invalid uoffset for entity_manifest vector")
    })?;
    let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
        OutputError::serialization(ctx, "entity_manifest vector header truncated or corrupt")
    })?;

    let mut out = Vec::with_capacity(nested_positions.len());
    for (idx, &slot_table_pos) in nested_positions.iter().enumerate() {
        let entry = deserialize_entity_slot_table(buf, slot_table_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("entity_slot table {idx} truncated or corrupt"))
        })?;
        out.push(entry);
    }
    Ok(out)
}

fn deserialize_entity_slot_table(buf: &[u8], slot_table_pos: usize) -> Option<EntitySlot> {
    let vtable_pos = resolve_vtable_pos(buf, slot_table_pos)?;

    let entity_type = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_ENTITY_TYPE,
    )
    .and_then(|p| buf.get(p).copied())
    .unwrap_or(0);

    let entity_id = field_pos(buf, slot_table_pos, vtable_pos, ENTITY_SLOT_FIELD_ENTITY_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(0);

    let subindex = field_pos(buf, slot_table_pos, vtable_pos, ENTITY_SLOT_FIELD_SUBINDEX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let was_active = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_WAS_ACTIVE,
    )
    .and_then(|p| read_bool_byte(buf, p))
    .unwrap_or(false);

    // Absent in a pre-`id:5` buffer (FlatBuffers graceful absence): default to
    // the sentinel, not zero — zero is a valid calendar date.
    let delivery_date = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_DELIVERY_DATE,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(ENTITY_SLOT_DELIVERY_DATE_SENTINEL);

    Some(EntitySlot {
        entity_type,
        entity_id,
        subindex,
        was_active,
        delivery_date,
    })
}

// The next five helpers (`read_string_field` through `read_manifest_edges`)
// share one absence contract: a field missing from the vtable yields an empty
// `String`/`Vec`, never an error (graceful absence).

/// Read a string field at vtable `slot`.
fn read_string_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<String, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(String::new());
    };
    let str_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for string field"))?;
    read_string(buf, str_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "string field truncated or not UTF-8"))
}

/// Read a `[uint32]` field at vtable `slot`.
fn read_u32_vector_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<u32>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for uint32 vector"))?;
    read_u32_vector(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "uint32 vector truncated or corrupt"))
}

/// Read a `[string]` field at vtable `slot`.
fn read_string_vector_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<String>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for string vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "string vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &str_pos) in positions.iter().enumerate() {
        let entry = read_string(buf, str_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("string vector element {idx} truncated"))
        })?;
        out.push(entry);
    }
    Ok(out)
}

/// Read a `[ManifestNode]` field at vtable `slot`.
fn read_manifest_nodes(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<ManifestNode>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for nodes vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "nodes vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &node_pos) in positions.iter().enumerate() {
        let node = deserialize_manifest_node_table(buf, node_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("manifest node table {idx} truncated"))
        })?;
        out.push(node);
    }
    Ok(out)
}

/// Read a `[ManifestEdge]` field at vtable `slot`.
fn read_manifest_edges(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<ManifestEdge>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for edges vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "edges vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &edge_pos) in positions.iter().enumerate() {
        let edge = deserialize_manifest_edge_table(buf, edge_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("manifest edge table {idx} truncated"))
        })?;
        out.push(edge);
    }
    Ok(out)
}

fn deserialize_manifest_node_table(buf: &[u8], node_table_pos: usize) -> Option<ManifestNode> {
    let vtable_pos = resolve_vtable_pos(buf, node_table_pos)?;

    let id = field_pos(buf, node_table_pos, vtable_pos, MANIFEST_NODE_FIELD_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(0);
    let stage_id = field_pos(
        buf,
        node_table_pos,
        vtable_pos,
        MANIFEST_NODE_FIELD_STAGE_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let pool_id = field_pos(buf, node_table_pos, vtable_pos, MANIFEST_NODE_FIELD_POOL_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    Some(ManifestNode {
        id,
        stage_id,
        pool_id,
    })
}

fn deserialize_manifest_edge_table(buf: &[u8], edge_table_pos: usize) -> Option<ManifestEdge> {
    let vtable_pos = resolve_vtable_pos(buf, edge_table_pos)?;

    let source_id = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_SOURCE_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let target_id = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_TARGET_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let probability = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_PROBABILITY,
    )
    .and_then(|p| read_f64_le(buf, p))
    .unwrap_or(0.0);

    Some(ManifestEdge {
        source_id,
        target_id,
        probability,
    })
}

// ── Deserializers ─────────────────────────────────────────────────────────────

/// Deserialize a `StageCuts` `FlatBuffers` buffer into an owned [`StageCutsReadResult`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, StageCutsPayload, serialize_stage_cuts, deserialize_stage_cuts};
///
/// let piece = PolicyCutRecord {
///     cut_id: 7,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 1,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(&StageCutsPayload {
///     stage_id: 2,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[piece],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
///     cost_scale_factor: 1_000_000.0,
///     node_id: 2,
///     graph_stage_id: 2,
/// });
/// let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
/// assert_eq!(result.stage_id, 2);
/// assert_eq!(result.cuts.len(), 1);
/// assert_eq!(result.cuts[0].cut_id, 7);
/// assert_eq!(result.cuts[0].coefficients, &[1.0, 2.0, 3.0]);
/// assert_eq!(result.cost_scale_factor, Some(1_000_000.0));
/// ```
pub fn deserialize_stage_cuts(buf: &[u8]) -> Result<StageCutsReadResult, OutputError> {
    let ctx = "stage_cuts";
    check_file_identifier(buf, ctx)?;

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

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

    let cuts = if let Some(cuts_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CUTS)
    {
        let vec_pos = follow_uoffset(buf, cuts_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for cuts vector"))?;

        let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "cuts vector header truncated or corrupt")
        })?;

        let mut out = Vec::with_capacity(nested_positions.len());
        for (idx, &piece_table_pos) in nested_positions.iter().enumerate() {
            let piece = deserialize_cut_table(buf, piece_table_pos).ok_or_else(|| {
                OutputError::serialization(
                    ctx,
                    format!("affine-piece table {idx} truncated or corrupt"),
                )
            })?;
            out.push(piece);
        }
        out
    } else {
        Vec::new()
    };

    let entity_manifest = read_entity_manifest(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_ENTITY_MANIFEST,
        ctx,
    )?;

    // Absent in a pre-`id:8` buffer (FlatBuffers graceful absence): cost_scale_factor
    // stays `None` (distinct from a real `0.0`), the two ids default to the sentinel
    // (never a bare `0` — `0` is a valid node/stage id).
    let cost_scale_factor = field_pos(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_COST_SCALE_FACTOR,
    )
    .and_then(|p| read_f64_le(buf, p));

    let node_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_NODE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL);

    let graph_stage_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_GRAPH_STAGE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL);

    Ok(StageCutsReadResult {
        stage_id,
        state_dimension,
        capacity,
        warm_start_count,
        populated_count,
        cuts,
        entity_manifest,
        cost_scale_factor,
        node_id,
        graph_stage_id,
    })
}

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
    check_file_identifier(buf, ctx)?;

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

/// Deserialize one stage's visited states from a `StageStates` `FlatBuffers` buffer.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated or
/// has an invalid wire format.
pub fn deserialize_stage_states(buf: &[u8]) -> Result<StageStatesReadResult, OutputError> {
    let ctx = "stage_states";
    check_file_identifier(buf, ctx)?;

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

    let entity_manifest = read_entity_manifest(
        buf,
        table_pos,
        vtable_pos,
        STATES_FIELD_ENTITY_MANIFEST,
        ctx,
    )?;

    // Absent in a pre-`id:5` buffer (FlatBuffers graceful absence): default to
    // the sentinel, never a bare 0 — 0 is a valid node id.
    let node_id = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_NODE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_STATES_NODE_ID_SENTINEL);

    Ok(StageStatesReadResult {
        stage_id,
        node_id,
        state_dimension,
        count,
        data,
        entity_manifest,
    })
}

/// Deserialize a `CheckpointManifest` `FlatBuffers` buffer into an owned
/// [`CheckpointManifest`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] when the buffer lacks the `CBVF`
/// identifier, is truncated or corrupt, or carries a `format_version` other than
/// [`FORMAT_VERSION`] (an absent version field reads as `0` and is likewise
/// rejected), so a stale-version manifest is refused before any consumer reads it.
pub fn deserialize_checkpoint_manifest(buf: &[u8]) -> Result<CheckpointManifest, OutputError> {
    let ctx = "checkpoint_manifest";
    check_file_identifier(buf, ctx)?;

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let format_version = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FORMAT_VERSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    if format_version != FORMAT_VERSION {
        return Err(OutputError::serialization(
            ctx,
            format!(
                "unsupported checkpoint manifest format_version {format_version}; expected {FORMAT_VERSION}"
            ),
        ));
    }

    let cobre_version = read_string_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_COBRE_VERSION,
        ctx,
    )?;
    let created_at = read_string_field(buf, table_pos, vtable_pos, MANIFEST_FIELD_CREATED_AT, ctx)?;

    let num_stages = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_NUM_STAGES)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let n_pools = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_N_POOLS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let nodes = read_manifest_nodes(buf, table_pos, vtable_pos, MANIFEST_FIELD_NODES, ctx)?;
    let edges = read_manifest_edges(buf, table_pos, vtable_pos, MANIFEST_FIELD_EDGES, ctx)?;

    let completed_iterations = field_pos(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_COMPLETED_ITERATIONS,
    )
    .and_then(|p| read_u32_le(buf, p))
    .unwrap_or(0);

    // A plain-f64 provenance field: absent reads as 0.0, matching the intercept
    // read. The two Option<f64> fields below stay None when absent, never 0.0.
    let final_lower_bound = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FINAL_LOWER_BOUND)
        .and_then(|p| read_f64_le(buf, p))
        .unwrap_or(0.0);
    let best_upper_bound = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_BEST_UPPER_BOUND)
        .and_then(|p| read_f64_le(buf, p));

    let max_iterations = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_MAX_ITERATIONS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let forward_passes = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FORWARD_PASSES)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let warm_start_cuts = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_WARM_START_CUTS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let warm_start_counts = read_u32_vector_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_WARM_START_COUNTS,
        ctx,
    )?;

    let rng_seed = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_RNG_SEED)
        .and_then(|p| read_u64_le(buf, p))
        .unwrap_or(0);
    let total_visited_states = field_pos(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TOTAL_VISITED_STATES,
    )
    .and_then(|p| read_u64_le(buf, p))
    .unwrap_or(0);

    let training_block_mode = read_string_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TRAINING_BLOCK_MODE,
        ctx,
    )?;
    let training_block_mode_per_stage = read_string_vector_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE,
        ctx,
    )?;

    let cost_scale_factor = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_COST_SCALE_FACTOR)
        .and_then(|p| read_f64_le(buf, p));

    Ok(CheckpointManifest {
        format_version,
        cobre_version,
        created_at,
        num_stages,
        graph_manifest: GraphManifest {
            n_pools,
            nodes,
            edges,
        },
        producer: ProducerBlock {
            completed_iterations,
            final_lower_bound,
            best_upper_bound,
            max_iterations,
            forward_passes,
            warm_start_cuts,
            warm_start_counts,
            rng_seed,
            total_visited_states,
            training_block_mode,
            training_block_mode_per_stage,
            cost_scale_factor,
        },
    })
}

/// Read all `*.bin` files from `dir`, deserialize each with `deser_fn`, and return a `Vec`.
///
/// The returned `Vec` is unsorted — callers must sort by `stage_id` after this call
/// (`read_dir` order is not guaranteed; sorting upholds declaration-order invariance).
pub(super) fn read_sorted_bin_files<T, F>(
    dir: &Path,
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
            OutputError::serialization(
                ctx,
                format!("failed to deserialize {}: {e}", file_path.display()),
            )
        })?;
        results.push(record);
    }
    Ok(results)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The reject-old-version half of the dual-owned wire-format checklist: a
    /// buffer without the `CBVF` `file_identifier` (a pre-0.14 `finish_minimal`
    /// artifact) is rejected by the read path before any field is decoded, so no
    /// stale month-integer anchor is ever decoded as a `YYYYMMDD` date.
    #[test]
    fn deserialize_rejects_buffer_without_cbvf_identifier() {
        let coeffs = [1.0_f64, 2.0];
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 3.0,
            coefficients: &coeffs,
            is_active: true,
        };
        let mut buf = serialize_stage_cuts(&StageCutsPayload {
            stage_id: 0,
            state_dimension: 2,
            capacity: 8,
            warm_start_count: 0,
            cuts: &[cut],
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: -1,
            graph_stage_id: -1,
        });
        assert_eq!(
            buf.get(4..8),
            Some(POLICY_FILE_IDENTIFIER.as_bytes()),
            "a freshly written buffer must carry the CBVF identifier"
        );

        // Strip the identifier to mimic a pre-0.14 `finish_minimal` buffer.
        buf[4..8].copy_from_slice(&[0, 0, 0, 0]);
        let err = deserialize_stage_cuts(&buf)
            .expect_err("a buffer without the CBVF identifier must be rejected");
        assert!(
            err.to_string().contains(POLICY_FILE_IDENTIFIER),
            "rejection must name the expected identifier: {err}"
        );
    }
}
