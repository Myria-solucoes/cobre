//! Filesystem write and read entry points for policy checkpoints.
//!
//! `metadata.json` is written last so its presence is the commit signal of a
//! complete checkpoint.

use std::path::Path;

use super::super::error::OutputError;
use super::codec::{
    deserialize_stage_basis, deserialize_stage_cuts, deserialize_stage_states,
    read_sorted_bin_files, serialize_stage_basis, serialize_stage_cuts, serialize_stage_states,
};
use super::records::{
    OwnedPolicyBasisRecord, PolicyBasisRecord, PolicyCheckpoint, PolicyCheckpointMetadata,
    StageCutsPayload, StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
};

/// Write a complete policy checkpoint to `path`.
///
/// ## Directory layout produced
///
/// ```text
/// path/
///   metadata.json
///   cuts/
///     stage_000.bin
///     stage_001.bin
///     ...
///   basis/
///     stage_000.bin   (only when stage_bases is non-empty)
///     stage_001.bin
///     ...
/// ```
///
/// `metadata.json` is written **last**, only after every `.bin` write succeeds:
/// its absence is how the caller detects an incomplete checkpoint. Partially
/// written files are not cleaned up. An empty `stage_bases` writes no basis files
/// (the `basis/` directory is still created).
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation or file write failed.
/// - [`OutputError::SerializationError`] — JSON serialization of metadata failed.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{
///     write_policy_checkpoint, PolicyBasisRecord, PolicyCheckpointMetadata, PolicyCutRecord,
///     StageCutsPayload,
/// };
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let coefficients = [1.0_f64, 2.0, 3.0];
/// let cut = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 0,
///     iteration: 1,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &coefficients,
///     is_active: true,
/// };
/// let stage_cuts = [StageCutsPayload {
///     stage_id: 0,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[cut],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
/// }];
/// let metadata = PolicyCheckpointMetadata {
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     completed_iterations: 1,
///     final_lower_bound: 42.0,
///     best_upper_bound: None,
///     state_dimension: 3,
///     num_stages: 1,
///     max_iterations: 100,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     warm_start_counts: vec![0],
///     rng_seed: 0,
///     total_visited_states: 0,
///     training_block_mode: "parallel".to_string(),
///     training_block_mode_per_stage: vec![],
/// };
/// write_policy_checkpoint(Path::new("/tmp/policy"), &stage_cuts, &[], &metadata, &[])?;
/// # Ok(())
/// # }
/// ```
pub fn write_policy_checkpoint(
    path: &Path,
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    metadata: &PolicyCheckpointMetadata,
    stage_states: &[StageStatesPayload<'_>],
) -> Result<(), OutputError> {
    let cuts_dir = path.join("cuts");
    std::fs::create_dir_all(&cuts_dir).map_err(|e| OutputError::io(&cuts_dir, e))?;

    let basis_dir = path.join("basis");
    std::fs::create_dir_all(&basis_dir).map_err(|e| OutputError::io(&basis_dir, e))?;

    for payload in stage_cuts {
        let filename = format!("stage_{:03}.bin", payload.stage_id);
        let file_path = cuts_dir.join(&filename);
        let buf = serialize_stage_cuts(
            payload.stage_id,
            payload.state_dimension,
            payload.capacity,
            payload.warm_start_count,
            payload.cuts,
            payload.active_cut_indices,
            payload.populated_count,
            payload.entity_manifest,
        );
        std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
    }

    for record in stage_bases {
        let filename = format!("stage_{:03}.bin", record.stage_id);
        let file_path = basis_dir.join(&filename);
        let buf = serialize_stage_basis(record);
        std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
    }

    if !stage_states.is_empty() {
        let states_dir = path.join("states");
        std::fs::create_dir_all(&states_dir).map_err(|e| OutputError::io(&states_dir, e))?;

        for payload in stage_states {
            let filename = format!("stage_{:03}.bin", payload.stage_id);
            let file_path = states_dir.join(&filename);
            let buf = serialize_stage_states(payload);
            std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
        }
    }

    // Write metadata.json LAST — its presence is the commit signal.
    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| OutputError::serialization("policy_metadata", e.to_string()))?;
    let meta_path = path.join("metadata.json");
    std::fs::write(&meta_path, json.as_bytes()).map_err(|e| OutputError::io(&meta_path, e))?;

    Ok(())
}

/// Read a complete policy checkpoint from `path`.
///
/// Per-stage results are sorted by `stage_id` in the returned [`PolicyCheckpoint`].
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory or file read failed.
/// - [`OutputError::SerializationError`] — JSON or `FlatBuffers` parse failure.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::read_policy_checkpoint;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let checkpoint = read_policy_checkpoint(Path::new("/tmp/policy"))?;
/// println!("metadata: {} stages", checkpoint.metadata.num_stages);
/// println!("stages loaded: {}", checkpoint.stage_cuts.len());
/// # Ok(())
/// # }
/// ```
pub fn read_policy_checkpoint(path: &Path) -> Result<PolicyCheckpoint, OutputError> {
    let meta_path = path.join("metadata.json");
    let meta_bytes = std::fs::read(&meta_path).map_err(|e| OutputError::io(&meta_path, e))?;
    let metadata: PolicyCheckpointMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| OutputError::serialization("policy_metadata", e.to_string()))?;

    let cuts_dir = path.join("cuts");
    let mut stage_cuts: Vec<StageCutsReadResult> =
        read_sorted_bin_files(&cuts_dir, "stage_cuts", deserialize_stage_cuts)?;
    stage_cuts.sort_by_key(|r| r.stage_id);

    let basis_dir = path.join("basis");
    let mut stage_bases: Vec<OwnedPolicyBasisRecord> =
        read_sorted_bin_files(&basis_dir, "stage_basis", deserialize_stage_basis)?;
    stage_bases.sort_by_key(|r| r.stage_id);

    let states_dir = path.join("states");
    let stage_states: Vec<StageStatesReadResult> = if states_dir.is_dir() {
        let mut ss = read_sorted_bin_files(&states_dir, "stage_states", deserialize_stage_states)?;
        ss.sort_by_key(|r| r.stage_id);
        ss
    } else {
        Vec::new()
    };

    Ok(PolicyCheckpoint {
        metadata,
        stage_cuts,
        stage_bases,
        stage_states,
    })
}
