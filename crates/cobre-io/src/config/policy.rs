//! Policy directory and checkpointing configuration types for `config.json → policy`.

use serde::{Deserialize, Serialize};

use std::fmt;

/// Policy initialization mode (`config.json → policy.mode`).
///
/// Controls whether the training phase starts from scratch, warm-starts from
/// a prior policy's rows, or resumes a checkpointed training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PolicyMode {
    /// Start training from an empty future-cost function.
    Fresh,
    /// Load rows from a prior policy checkpoint and continue training.
    WarmStart,
    /// Resume a previously interrupted training run from its checkpoint.
    Resume,
}

impl std::fmt::Display for PolicyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyMode::Fresh => f.write_str("fresh"),
            PolicyMode::WarmStart => f.write_str("warm_start"),
            PolicyMode::Resume => f.write_str("resume"),
        }
    }
}

/// Boundary-row configuration for terminal-stage FCF coupling.
///
/// When present, the solver loads rows from a source Cobre policy
/// checkpoint and injects them as fixed boundary conditions at the
/// terminal stage of the current study.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BoundaryPolicy {
    /// Path to the source policy checkpoint directory.
    pub path: String,
    /// 0-based stage index in the source checkpoint to load rows from.
    pub source_stage: u32,
}

/// Policy directory settings (`config.json → policy`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PolicyConfig {
    /// Directory for policy data (rows, states, vertices, basis).
    pub path: String,

    /// Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`.
    pub mode: PolicyMode,

    /// Checkpoint settings.
    pub checkpointing: CheckpointingConfig,

    /// Optional boundary-row policy for terminal-stage coupling.
    #[serde(default)]
    pub boundary: Option<BoundaryPolicy>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            checkpointing: CheckpointingConfig::default(),
            boundary: None,
        }
    }
}

/// Checkpoint settings (`config.json → policy.checkpointing`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CheckpointingConfig {
    /// Enable periodic checkpointing.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// First iteration to write a checkpoint.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between checkpoints.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Include LP basis in checkpoints for warm-start.
    #[serde(default)]
    pub store_basis: Option<bool>,

    /// Compress checkpoint files.
    #[serde(default)]
    pub compress: Option<bool>,
}
