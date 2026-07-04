//! Input and output record types for policy checkpoint serialization.
//!
//! Input types (`PolicyCutRecord`, `PolicyBasisRecord`, `StageStatesPayload`,
//! `StageCutsPayload`) borrow from caller-owned buffers; owned output types
//! (`Owned*`, `*ReadResult`, `PolicyCheckpoint`) own their vectors. All use
//! generic names to maintain infrastructure crate genericity; conversion from
//! algorithm-specific types is the calling crate's responsibility. Field names
//! correspond to the tables in `schemas/policy.fbs`.

/// Sentinel [`EntitySlot::delivery_anchor`] value for a slot with no
/// delivery/arrival calendar semantics; also the value a reader yields when the
/// field is absent from a pre-`id:4` buffer (forward-compatible default).
pub const ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL: i32 = i32::MIN;

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
    /// Canonical absolute delivery/arrival calendar anchor for this slot;
    /// [`ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL`] when the slot has no delivery
    /// semantics. The calendar encoding is the calling crate's responsibility,
    /// as with `subindex`.
    pub delivery_anchor: i32,
}

/// One cut record for policy checkpoint serialization.
///
/// `'a` borrows the coefficient slice without copying (vectors can be large).
#[derive(Debug, Clone)]
pub struct PolicyCutRecord<'a> {
    /// Unique identifier for this cut across all iterations.
    pub cut_id: u64,
    /// LP row position (required for checkpoint reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this cut.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed cut intercept: `alpha - beta' * x_hat`.
    pub intercept: f64,
    /// Gradient coefficients, length must equal `state_dimension`.
    ///
    /// Positional only: index `i` is the i-th state-vector dimension, whose
    /// identity is carried by slot `i` of the co-located [`EntitySlot`] manifest
    /// (`entity_manifest`); no labels are stored inline.
    pub coefficients: &'a [f64],
    /// Whether this cut is currently active in the LP.
    pub is_active: bool,
}

/// One stage's solver basis for policy checkpoint serialization.
///
/// `'a` borrows the status arrays without copying.
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
    /// Number of trailing rows in `row_status` that correspond to cut rows.
    pub num_cut_rows: u32,
}

/// Payload for writing per-stage visited states to a policy checkpoint.
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

/// Per-stage cut data payload for [`crate::write_policy_checkpoint`], grouping the
/// arguments of [`crate::serialize_stage_cuts`]. `'a` borrows slices without copying.
#[derive(Debug)]
pub struct StageCutsPayload<'a> {
    /// Stage index (0-based), used as the file name index in `cuts/stage_NNN.bin`.
    pub stage_id: u32,
    /// Number of state variables; determines coefficient vector length per cut.
    pub state_dimension: u32,
    /// Total preallocated cut slots in the pool.
    pub capacity: u32,
    /// Number of slots `[0..warm_start_count)` loaded from a prior policy.
    pub warm_start_count: u32,
    /// Slice of cut records to serialize; length equals `populated_count`.
    pub cuts: &'a [PolicyCutRecord<'a>],
    /// Indices of cuts currently active in the LP.
    pub active_cut_indices: &'a [u32],
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
}

/// Policy metadata for checkpoint resume and warm-start.
///
/// Serialized to JSON (not `FlatBuffers`) because it is small, human-readable, and
/// may be edited by operators.
///
/// # Examples
///
/// ```
/// use cobre_io::PolicyCheckpointMetadata;
///
/// let meta = PolicyCheckpointMetadata {
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     completed_iterations: 50,
///     final_lower_bound: 1234.56,
///     best_upper_bound: Some(1300.0),
///     state_dimension: 160,
///     num_stages: 60,
///     max_iterations: 200,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     warm_start_counts: vec![],
///     rng_seed: 42,
///     total_visited_states: 0,
///     training_block_mode: "parallel".to_string(),
///     training_block_mode_per_stage: vec![],
/// };
/// let json = serde_json::to_string_pretty(&meta).unwrap();
/// assert!(json.contains("completed_iterations"));
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyCheckpointMetadata {
    /// Cobre crate version that wrote this checkpoint.
    pub cobre_version: String,
    /// ISO 8601 timestamp when the checkpoint was written.
    pub created_at: String,
    /// Number of training iterations completed at checkpoint time.
    pub completed_iterations: u32,
    /// Lower bound value after the final completed iteration.
    pub final_lower_bound: f64,
    /// Best upper bound observed during training, if available.
    pub best_upper_bound: Option<f64>,
    /// Number of state variables (determines cut coefficient vector length).
    pub state_dimension: u32,
    /// Number of stages in the planning horizon.
    pub num_stages: u32,
    /// Maximum number of iterations configured for the run.
    pub max_iterations: u32,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Number of cuts loaded from a previous policy at run start.
    pub warm_start_cuts: u32,
    /// Per-stage warm-start cut counts (one per stage, 0-based).
    ///
    /// When non-empty, supersedes [`warm_start_cuts`] for per-stage accuracy.
    /// Empty in old checkpoints; fall back to broadcasting [`warm_start_cuts`].
    ///
    /// [`warm_start_cuts`]: Self::warm_start_cuts
    #[serde(default)]
    pub warm_start_counts: Vec<u32>,
    /// RNG seed used by the scenario sampler.
    ///
    /// Per-draw seeds are derived from `(rng_seed, iteration, scenario, stage)`,
    /// so resume needs only the seed — no accumulated RNG state is persisted.
    pub rng_seed: u64,
    /// Total visited states across all stages.
    ///
    /// Absent in checkpoints written before this field was added;
    /// defaults to `0` on deserialization.
    #[serde(default)]
    pub total_visited_states: u64,
    /// Block mode the policy was trained under: the shared lowercase mode when
    /// every study stage agrees, else `"mixed"`.
    ///
    /// Empty in checkpoints written before this field was added; an empty value
    /// reads as "unknown / pre-field policy".
    #[serde(default)]
    pub training_block_mode: String,
    /// Per-study-stage training block modes, in study-stage order.
    ///
    /// Populated only for mixed-mode studies (empty when all stages share one
    /// mode, and in pre-field checkpoints).
    #[serde(default)]
    pub training_block_mode_per_stage: Vec<String>,
}

// ── Owned output types for deserialization ───────────────────────────────────

/// Owned version of [`PolicyCutRecord`] returned by [`crate::deserialize_stage_cuts`].
///
/// Unlike [`PolicyCutRecord<'a>`], this type owns its `coefficients` vector so it
/// can be returned from a deserialization function that does not borrow from the
/// input buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyCutRecord {
    /// Unique identifier for this cut across all iterations.
    pub cut_id: u64,
    /// LP row position (required for checkpoint reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this cut.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed cut intercept.
    pub intercept: f64,
    /// Gradient coefficients; positional per the [`PolicyCutRecord::coefficients`] contract.
    pub coefficients: Vec<f64>,
    /// Whether this cut is currently active in the LP.
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
    /// Number of trailing rows in `row_status` that correspond to cut rows.
    pub num_cut_rows: u32,
}

/// Stage-level metadata and cut records returned by [`crate::deserialize_stage_cuts`].
///
/// Contains the stage-level fields stored in the `StageCuts` root table plus the
/// vector of deserialized cut records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageCutsReadResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Number of state variables; equals the length of each cut's `coefficients` vector.
    pub state_dimension: u32,
    /// Total preallocated cut slots in the pool.
    pub capacity: u32,
    /// Number of slots loaded from a prior policy.
    pub warm_start_count: u32,
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Deserialized cut records.
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

/// Complete deserialized policy checkpoint returned by [`crate::read_policy_checkpoint`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyCheckpoint {
    /// Policy metadata read from `metadata.json`.
    pub metadata: PolicyCheckpointMetadata,
    /// Per-stage cut pools, sorted by `stage_id`.
    pub stage_cuts: Vec<StageCutsReadResult>,
    /// Per-stage solver bases, sorted by `stage_id`.
    pub stage_bases: Vec<OwnedPolicyBasisRecord>,
    /// Per-stage visited states, sorted by `stage_id`.
    ///
    /// Empty for checkpoints written before this field was added.
    pub stage_states: Vec<StageStatesReadResult>,
}
