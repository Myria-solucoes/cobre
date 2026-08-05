//! Input and output record types for value-function artifact serialization.
//!
//! Input types (`PolicyCutRecord`, `PolicyBasisRecord`, `StageStatesPayload`,
//! `StageCutsPayload`) borrow from caller-owned buffers; owned output types
//! (`Owned*`, `*ReadResult`, `PolicyCheckpoint`) own their vectors. All use
//! generic names to maintain infrastructure crate genericity; conversion from
//! algorithm-specific types is the calling crate's responsibility. Field names
//! correspond to the tables in `schemas/policy.fbs`.

/// Current on-disk value-function artifact format version.
///
/// [`PolicyCheckpointMetadata::format_version`] must equal this;
/// [`crate::read_policy_checkpoint`] rejects any other value — and absence —
/// with a named error before parsing any payload, so a pre-marker artifact is
/// cleanly rejected, never read positionally.
pub const FORMAT_VERSION: u32 = 1;

/// Sentinel [`EntitySlot::delivery_date`] value for a slot with no
/// delivery/arrival calendar semantics; also the value a reader yields when the
/// field is absent from a pre-`id:5` buffer (forward-compatible default).
pub const ENTITY_SLOT_DELIVERY_DATE_SENTINEL: i32 = i32::MIN;

/// One per-slot entity-identity record for a state-vector dimension.
///
/// `entity_type` is the raw `EntityType` enum byte from `schemas/policy.fbs`
/// (`0`/`1`/`2`); the dimension-class meaning of each value is owned by the
/// calling crate, not interpreted here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntitySlot {
    /// Raw `EntityType` enum byte.
    pub entity_type: u8,
    /// Owning entity's id; `int32` because a sentinel id can be `-1`.
    pub entity_id: i32,
    /// Secondary index within the owning entity (per-type meaning is the caller's).
    pub subindex: u32,
    /// Whether the owning entity was operationally active at this slot's stage.
    pub was_active: bool,
    /// Canonical absolute delivery/arrival calendar date for this slot, encoded
    /// `YYYYMMDD` (`year * 10000 + month * 100 + day`);
    /// [`ENTITY_SLOT_DELIVERY_DATE_SENTINEL`] when the slot has no delivery
    /// semantics. Which calendar date maps to a slot is the calling crate's
    /// responsibility, as with `subindex`.
    pub delivery_date: i32,
}

/// One affine-piece record for value-function artifact serialization.
///
/// `'a` borrows the coefficient slice without copying (vectors can be large).
#[derive(Debug, Clone)]
pub struct PolicyCutRecord<'a> {
    /// Unique identifier for this piece across all iterations.
    pub cut_id: u64,
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients, length must equal `state_dimension`.
    ///
    /// Positional only: index `i` is the i-th state-vector dimension, whose
    /// identity is carried by slot `i` of the co-located [`EntitySlot`] manifest
    /// (`entity_manifest`); no labels are stored inline.
    pub coefficients: &'a [f64],
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
}

/// One stage's solver basis for value-function artifact serialization.
#[derive(Debug, Clone)]
pub struct PolicyBasisRecord<'a> {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: &'a [u8],
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: &'a [u8],
    /// Number of trailing rows in `row_status` that correspond to affine-piece rows.
    pub num_cut_rows: u32,
}

/// Payload for writing per-stage visited states to a value-function artifact.
///
/// The `data` slice contains the flat state vectors (row-major, each of length
/// `state_dimension`). The total number of stored states is `count`.
#[derive(Debug, Clone)]
pub struct StageStatesPayload<'a> {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Length of each state vector.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer: `count * state_dimension` f64 elements.
    pub data: &'a [f64],
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
}

/// Per-pool affine-piece data payload for [`crate::write_policy_checkpoint`],
/// grouping the arguments of [`crate::serialize_stage_cuts`].
#[derive(Debug)]
pub struct StageCutsPayload<'a> {
    /// Pool id (0-based) — the storage-unit key naming this payload's file
    /// `cuts/<pool>.bin`. Equals the stage index on a chain.
    pub stage_id: u32,
    /// Number of state variables; determines coefficient vector length per piece.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Number of slots `[0..warm_start_count)` loaded from a prior artifact.
    pub warm_start_count: u32,
    /// Slice of affine-piece records to serialize; length equals `populated_count`.
    pub cuts: &'a [PolicyCutRecord<'a>],
    /// Indices of pieces currently active in the LP.
    pub active_cut_indices: &'a [u32],
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
}

/// One node of the value-function artifact's graph manifest: its declared id,
/// the stage it sits at, and the pool whose payload carries its affine pieces.
///
/// `pool_id` **is** the node → pool map: a node references one pool, and a
/// reader resolves node `id`'s pieces as `pool_id`'s payload (leaf nodes sharing
/// a pool all name the same `pool_id`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestNode {
    /// Declared node id.
    pub id: i32,
    /// Stage id this node sits at.
    pub stage_id: i32,
    /// Pool whose payload holds this node's affine pieces.
    pub pool_id: u32,
}

/// One directed edge of the graph manifest, with its transition probability.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestEdge {
    /// Source node id.
    pub source_id: i32,
    /// Target node id.
    pub target_id: i32,
    /// Transition probability `P(source -> target)`.
    pub probability: f64,
}

/// The graph manifest: the node list (each node carrying its own node → pool
/// assignment), the edge list, and the pool-set size.
///
/// This is the identity source the positional format never had — a reader
/// resolves a node's payload through it (node `n`'s pieces are `pool(n)`'s
/// payload), rather than trusting a filename.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GraphManifest {
    /// Number of distinct pools (the pool-set size).
    pub n_pools: u32,
    /// Every node, in canonical order, each with its stage and pool.
    pub nodes: Vec<ManifestNode>,
    /// Every directed edge with its transition probability.
    pub edges: Vec<ManifestEdge>,
}

/// Producer-namespaced metadata: everything specific to how the artifact was
/// produced (the training algorithm's own recorded state), segregated from the
/// neutral core so a reader that does not know the producer can still read the
/// core from the core's own vocabulary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProducerBlock {
    /// Number of training iterations completed at write time.
    pub completed_iterations: u32,
    /// Lower bound value after the final completed iteration.
    pub final_lower_bound: f64,
    /// Last iteration's upper bound, if available (the final value, not a
    /// min-tracked best).
    pub best_upper_bound: Option<f64>,
    /// Maximum number of iterations configured for the run.
    pub max_iterations: u32,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Number of pieces loaded from a previous artifact at run start.
    pub warm_start_cuts: u32,
    /// Per-pool warm-start piece counts, in pool-id order.
    ///
    /// When non-empty, supersedes [`warm_start_cuts`] for per-pool accuracy.
    ///
    /// [`warm_start_cuts`]: Self::warm_start_cuts
    #[serde(default)]
    pub warm_start_counts: Vec<u32>,
    /// RNG seed used by the scenario sampler.
    ///
    /// Per-draw seeds are derived from `(rng_seed, iteration, scenario, stage)`,
    /// so resume needs only the seed — no accumulated RNG state is persisted.
    pub rng_seed: u64,
    /// Total visited states across all nodes.
    #[serde(default)]
    pub total_visited_states: u64,
    /// Block mode the artifact was trained under: the shared lowercase mode when
    /// every study stage agrees, else `"mixed"`.
    #[serde(default)]
    pub training_block_mode: String,
    /// Per-study-stage training block modes, in study-stage order.
    ///
    /// Populated only for mixed-mode studies.
    #[serde(default)]
    pub training_block_mode_per_stage: Vec<String>,
    /// Objective cost-scale factor the writing study resolved
    /// (`modeling.cost_scale_factor`) — the provenance marker that makes piece
    /// `coefficients`/`intercept` scale-independent at rest (canonical
    /// currency units, not the writer's internal scaled cost space).
    ///
    /// Absent when unmarked; a missing marker is interpreted as
    /// scaled-at-`1_000_000.0`, the constant every unmarked artifact was
    /// unconditionally written under.
    #[serde(default)]
    pub cost_scale_factor: Option<f64>,
}

/// Value-function artifact metadata for resume and warm-start.
///
/// Serialized to JSON (not `FlatBuffers`) because it is small, human-readable,
/// and may be edited by operators. The neutral core (`format_version`,
/// provenance, the stage/pool/graph descriptors) describes the artifact itself;
/// the algorithm's own recorded state lives under the namespaced [`producer`]
/// block, so a reader that does not know the producer reads the core from the
/// core's own vocabulary.
///
/// [`producer`]: Self::producer
///
/// # Examples
///
/// ```
/// use cobre_io::{
///     FORMAT_VERSION, GraphManifest, PolicyCheckpointMetadata, ProducerBlock,
/// };
///
/// let meta = PolicyCheckpointMetadata {
///     format_version: FORMAT_VERSION,
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     num_stages: 60,
///     graph_manifest: GraphManifest::default(),
///     producer: ProducerBlock {
///         completed_iterations: 50,
///         final_lower_bound: 1234.56,
///         best_upper_bound: Some(1300.0),
///         max_iterations: 200,
///         forward_passes: 4,
///         warm_start_cuts: 0,
///         warm_start_counts: vec![],
///         rng_seed: 42,
///         total_visited_states: 0,
///         training_block_mode: "parallel".to_string(),
///         training_block_mode_per_stage: vec![],
///         cost_scale_factor: None,
///     },
/// };
/// let json = serde_json::to_string_pretty(&meta).unwrap();
/// assert!(json.contains("producer"));
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyCheckpointMetadata {
    /// On-disk format version; must equal [`FORMAT_VERSION`] on read. Defaults to
    /// `0` when absent (a pre-marker artifact), which the reader rejects.
    #[serde(default)]
    pub format_version: u32,
    /// Cobre crate version that wrote this artifact.
    pub cobre_version: String,
    /// ISO 8601 timestamp when the artifact was written.
    pub created_at: String,
    /// Number of stages the graph manifest spans.
    pub num_stages: u32,
    /// Graph manifest: node list, edge list, node → pool map, and pool-set size.
    #[serde(default)]
    pub graph_manifest: GraphManifest,
    /// Producer-namespaced metadata (the training algorithm's own state).
    pub producer: ProducerBlock,
}

// ── Owned output types for deserialization ───────────────────────────────────

/// Owned version of [`PolicyCutRecord`] returned by [`crate::deserialize_stage_cuts`].
///
/// Unlike [`PolicyCutRecord<'a>`], this type owns its `coefficients` vector so it
/// can be returned from a deserialization function that does not borrow from the
/// input buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyCutRecord {
    /// Unique identifier for this piece across all iterations.
    pub cut_id: u64,
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients; positional per the [`PolicyCutRecord::coefficients`] contract.
    pub coefficients: Vec<f64>,
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
}

/// Owned version of [`PolicyBasisRecord`] returned by [`crate::deserialize_stage_basis`].
///
/// Unlike [`PolicyBasisRecord<'a>`], this type owns its status byte vectors so it
/// can be returned from a deserialization function.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyBasisRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: Vec<u8>,
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: Vec<u8>,
    /// Number of trailing rows in `row_status` that correspond to affine-piece rows.
    pub num_cut_rows: u32,
}

/// Stage-level metadata and affine-piece records returned by [`crate::deserialize_stage_cuts`].
///
/// Contains the stage-level fields stored in the `StageCuts` root table plus the
/// vector of deserialized affine-piece records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageCutsReadResult {
    /// Pool id (0-based), as written by the pool-keyed payload.
    pub stage_id: u32,
    /// Number of state variables; equals the length of each piece's `coefficients` vector.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Number of slots loaded from a prior artifact.
    pub warm_start_count: u32,
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Deserialized affine-piece records.
    pub cuts: Vec<OwnedPolicyCutRecord>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
}

/// Owned version of [`StageStatesPayload`] returned by [`crate::deserialize_stage_states`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageStatesReadResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Length of each state vector.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer (owned).
    pub data: Vec<f64>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
}

/// Complete deserialized value-function artifact returned by [`crate::read_policy_checkpoint`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyCheckpoint {
    /// Metadata read from `metadata.json`.
    pub metadata: PolicyCheckpointMetadata,
    /// Per-pool affine-piece collections, sorted by pool id.
    pub stage_cuts: Vec<StageCutsReadResult>,
    /// Per-stage solver bases, sorted by `stage_id`.
    pub stage_bases: Vec<OwnedPolicyBasisRecord>,
    /// Per-stage visited states, sorted by `stage_id`.
    ///
    /// Empty when the artifact was written without visited states.
    pub stage_states: Vec<StageStatesReadResult>,
}
