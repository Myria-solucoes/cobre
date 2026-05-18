//! Input and output record types for policy checkpoint serialization.
//!
//! Input types (`PolicyCutRecord`, `PolicyBasisRecord`, `StageStatesPayload`,
//! `StageCutsPayload`) borrow their data from caller-owned buffers. Owned output
//! types (`OwnedPolicyCutRecord`, `OwnedPolicyBasisRecord`, `StageCutsReadResult`,
//! `StageStatesReadResult`, `PolicyCheckpoint`) are returned from deserialization
//! and own their vectors.

/// One cut record for policy checkpoint serialization.
///
/// Conversion from algorithm-specific cut pool structures is handled by the calling
/// algorithm crate. This type uses generic names to maintain infrastructure crate
/// genericity. The lifetime parameter `'a` allows borrowing the coefficient slice
/// without copying (coefficient vectors can reach 2,080 `f64` values at production
/// scale).
///
/// Field names correspond to the `Cut` table in `schemas/policy.fbs`.
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
    /// Gradient coefficient vector, length must equal `state_dimension`.
    pub coefficients: &'a [f64],
    /// Whether this cut is currently active in the LP.
    pub is_active: bool,
}

/// One stage's solver basis for policy checkpoint serialization.
///
/// Conversion from solver-specific basis structures is handled by the calling crate.
/// The lifetime parameter `'a` allows borrowing the status arrays without copying.
///
/// Field names correspond to the `StageBasis` table in `schemas/policy.fbs`.
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
}

/// Per-stage cut data payload for [`crate::write_policy_checkpoint`].
///
/// Groups all fields required by [`crate::serialize_stage_cuts`] into a single struct so
/// the checkpoint writer can iterate over stages without unpacking individual
/// arguments at each call site. The lifetime parameter `'a` allows borrowing
/// coefficient slices and index arrays without copying.
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
}

/// Policy metadata for checkpoint resume and warm-start.
///
/// Serialized to JSON (not `FlatBuffers`) because it is small, human-readable, and
/// may be edited by operators. The `serde::Serialize` derive enables
/// `serde_json::to_string_pretty` in the checkpoint writer.
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
    /// The noise sampling architecture derives per-draw seeds from
    /// `(rng_seed, iteration, scenario, stage)` via SipHash-1-3. This
    /// makes noise at any given iteration deterministic from the seed
    /// alone — no accumulated RNG state is needed for resume. A resumed
    /// training run with the same `rng_seed` and `forward_passes` will
    /// produce identical noise sequences at each iteration.
    pub rng_seed: u64,
    /// Total visited states across all stages.
    ///
    /// Absent in checkpoints written before this field was added;
    /// defaults to `0` on deserialization.
    #[serde(default)]
    pub total_visited_states: u64,
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
    /// Gradient coefficient vector, length equals `state_dimension` of the stage.
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
