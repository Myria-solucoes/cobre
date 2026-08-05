//! Filesystem write and read entry points for value-function artifacts.
//!
//! `metadata.json` is written last so its presence is the commit signal of a
//! complete artifact, and it carries the `format_version` marker the reader
//! checks first — a pre-marker artifact is cleanly rejected before any payload
//! is parsed.

use std::path::Path;

use super::super::error::OutputError;
use super::codec::{
    deserialize_stage_basis, deserialize_stage_cuts, deserialize_stage_states,
    read_sorted_bin_files, serialize_stage_basis, serialize_stage_cuts, serialize_stage_states,
};
use super::records::{
    FORMAT_VERSION, OwnedPolicyBasisRecord, PolicyBasisRecord, PolicyCheckpoint,
    PolicyCheckpointMetadata, StageCutsPayload, StageCutsReadResult, StageStatesPayload,
    StageStatesReadResult,
};

/// One `.bin` payload file name, keyed by the payload's own id (the pool id for
/// `cuts/`, the stage id for `basis/`/`states/`). Zero-padded for a stable
/// on-disk sort; the reader derives identity from inside each buffer, never from
/// this name.
fn bin_file_name(id: u32) -> String {
    format!("{id:03}.bin")
}

/// Write a complete value-function artifact to `path`.
///
/// ## Directory layout produced
///
/// ```text
/// path/
///   metadata.json
///   cuts/
///     000.bin        (one per pool, keyed by pool id; a shared leaf pool once)
///     001.bin
///     ...
///   basis/
///     000.bin        (only when stage_bases is non-empty)
///     001.bin
///     ...
/// ```
///
/// `metadata.json` is written **last**, only after every `.bin` write succeeds:
/// its absence is how the caller detects an incomplete artifact. Partially
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
///     write_policy_checkpoint, FORMAT_VERSION, GraphManifest, PolicyBasisRecord,
///     PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock, StageCutsPayload,
/// };
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let coefficients = [1.0_f64, 2.0, 3.0];
/// let piece = PolicyCutRecord {
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
///     cuts: &[piece],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
/// }];
/// let metadata = PolicyCheckpointMetadata {
///     format_version: FORMAT_VERSION,
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     num_stages: 1,
///     graph_manifest: GraphManifest::default(),
///     producer: ProducerBlock {
///         completed_iterations: 1,
///         final_lower_bound: 42.0,
///         best_upper_bound: None,
///         max_iterations: 100,
///         forward_passes: 4,
///         warm_start_cuts: 0,
///         warm_start_counts: vec![0],
///         rng_seed: 0,
///         total_visited_states: 0,
///         training_block_mode: "parallel".to_string(),
///         training_block_mode_per_stage: vec![],
///         cost_scale_factor: None,
///     },
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
        let file_path = cuts_dir.join(bin_file_name(payload.stage_id));
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
        let file_path = basis_dir.join(bin_file_name(record.stage_id));
        let buf = serialize_stage_basis(record);
        std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
    }

    if !stage_states.is_empty() {
        let states_dir = path.join("states");
        std::fs::create_dir_all(&states_dir).map_err(|e| OutputError::io(&states_dir, e))?;

        for payload in stage_states {
            let file_path = states_dir.join(bin_file_name(payload.stage_id));
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

/// Read a complete value-function artifact from `path`.
///
/// `metadata.json` is read first and its `format_version` is checked against
/// [`FORMAT_VERSION`] **before any `.bin` payload is parsed**: an absent or
/// mismatched version is a named [`OutputError::SerializationError`], so a
/// pre-marker artifact is cleanly rejected rather than read positionally.
///
/// Per-pool/-stage results are sorted by `stage_id` in the returned
/// [`PolicyCheckpoint`].
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory or file read failed.
/// - [`OutputError::SerializationError`] — JSON or `FlatBuffers` parse failure,
///   or a `format_version` that is absent or not [`FORMAT_VERSION`].
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
    #[derive(serde::Deserialize)]
    struct FormatVersionProbe {
        #[serde(default)]
        format_version: u32,
    }

    let meta_path = path.join("metadata.json");
    let meta_bytes = std::fs::read(&meta_path).map_err(|e| OutputError::io(&meta_path, e))?;
    // Check the `format_version` marker FIRST — off a minimal probe that ignores
    // every other field — so a pre-marker 0.13 artifact (which also lacks the
    // newer required fields) is rejected by the marker with a named error, never
    // an opaque missing-field parse error, and before any `.bin` is parsed.
    let probe: FormatVersionProbe = serde_json::from_slice(&meta_bytes)
        .map_err(|e| OutputError::serialization("policy_metadata", e.to_string()))?;
    if probe.format_version != FORMAT_VERSION {
        return Err(OutputError::serialization(
            "policy_metadata",
            format!(
                "unsupported value-function artifact format_version: expected {FORMAT_VERSION}, \
                 found {} (a pre-marker or other-version artifact is not readable by this build)",
                probe.format_version
            ),
        ));
    }

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
