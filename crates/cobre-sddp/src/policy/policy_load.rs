//! Policy loading and compatibility validation.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction

use cobre_io::PolicyCheckpointMetadata;
use cobre_solver::Basis;

use crate::SddpError;
use crate::cut::pool::CutPool;
use crate::setup::StudySetup;
use crate::workspace::CapturedBasis;

/// Resolve the per-stage warm-start cut counts from a loaded policy checkpoint.
///
/// Returns a `Vec<u32>` of length `num_stages` for [`FutureCostFunction::new`].
/// An empty `metadata.warm_start_counts` (old checkpoint format) broadcasts the
/// scalar `warm_start_cuts` to all stages.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `warm_start_counts.len() != num_stages`.
///
/// [`FutureCostFunction::new`]: crate::FutureCostFunction::new
// Rationale: kept as the validated entry point for the planned checkpoint-migration
// tool (and exercised by this module's tests); the active path consumes
// `metadata.warm_start_counts` directly.
#[allow(dead_code)]
pub(crate) fn resolve_warm_start_counts(
    metadata: &PolicyCheckpointMetadata,
    num_stages: usize,
) -> Result<Vec<u32>, SddpError> {
    if metadata.warm_start_counts.is_empty() {
        Ok(vec![metadata.warm_start_cuts; num_stages])
    } else if metadata.warm_start_counts.len() != num_stages {
        Err(SddpError::Validation(format!(
            "warm_start_counts length mismatch: checkpoint has {}, current system has {} stages",
            metadata.warm_start_counts.len(),
            num_stages,
        )))
    } else {
        Ok(metadata.warm_start_counts.clone())
    }
}

/// Validate that a loaded policy checkpoint is compatible with the current
/// system configuration.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `state_dimension` or `num_stages`
/// do not match the current system configuration.
pub fn validate_policy_compatibility(
    metadata: &PolicyCheckpointMetadata,
    current_state_dimension: u32,
    current_num_stages: u32,
) -> Result<(), SddpError> {
    if metadata.state_dimension != current_state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {}",
            metadata.state_dimension, current_state_dimension
        )));
    }

    if metadata.num_stages != current_num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            metadata.num_stages, current_num_stages
        )));
    }

    Ok(())
}

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<CapturedBasis>>`, one entry per stage; stages without a
/// matching record get `None` (`u8` status codes widen to `i32`).
///
/// # Cut-slot reconstruction
///
/// `row_status` is `[template rows…, cut rows…]`, the trailing `num_cut_rows` in
/// capture-time [`CutPool::active_cuts`](crate::cut::pool::CutPool::active_cuts)
/// order (active slots, increasing). Slot identity is recovered by matching each
/// basis record to its [`StageCutsReadResult`](cobre_io::StageCutsReadResult) by
/// `stage_id` and taking the active records' `slot_index` in increasing order, so
/// `reconstruct_basis` preserves stored cut-row statuses across cut-set churn.
///
/// # Graceful fallback
///
/// When the derived active-slot count ≠ `num_cut_rows` (cut selection deactivated
/// cuts between capture and export) or no cut record matches, fall back to safe
/// all-template behavior (empty `cut_row_slots`; every cut row reconstructs
/// BASIC). This changes only the warm-start solve path, never the optimum.
#[must_use]
pub fn build_basis_cache_from_checkpoint(
    num_stages: usize,
    stage_bases: &[cobre_io::OwnedPolicyBasisRecord],
    stage_cuts: &[cobre_io::StageCutsReadResult],
) -> Vec<Option<CapturedBasis>> {
    let mut cache: Vec<Option<CapturedBasis>> = vec![None; num_stages];
    for record in stage_bases {
        let stage = record.stage_id as usize;
        if stage >= num_stages {
            continue;
        }
        let col_status: Vec<i32> = record.column_status.iter().map(|&c| i32::from(c)).collect();
        let row_status: Vec<i32> = record.row_status.iter().map(|&r| i32::from(r)).collect();

        let num_cut = record.num_cut_rows as usize;
        let active_slots: Option<Vec<u32>> = stage_cuts
            .iter()
            .find(|sc| sc.stage_id == record.stage_id)
            .map(|sc| {
                sc.cuts
                    .iter()
                    .filter(|c| c.is_active)
                    .map(|c| c.slot_index)
                    .collect()
            });

        let (base_row_count, cut_row_slots) = match active_slots {
            Some(slots) if slots.len() == num_cut && num_cut <= row_status.len() => {
                (row_status.len() - num_cut, slots)
            }
            _ => (row_status.len(), Vec::new()),
        };
        debug_assert_eq!(
            cut_row_slots.len(),
            row_status.len() - base_row_count,
            "build_basis_cache_from_checkpoint: cut_row_slots length must equal the trailing \
             cut-row count for the CapturedBasis invariant",
        );

        cache[stage] = Some(CapturedBasis {
            basis: Basis {
                col_status,
                row_status,
            },
            base_row_count,
            cut_row_slots,
            state_at_capture: Vec::new(),
        });
    }
    cache
}

/// Positional identity of one state-vector slot; `was_active` is excluded —
/// adding it would reject a cut whose entity merely changed activity across
/// studies.
fn slot_identity(slot: &cobre_io::EntitySlot) -> (u8, i32, u32) {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// Load boundary cuts from the `source_stage` of a source Cobre policy checkpoint.
///
/// Per-slot `(entity_type, entity_id, subindex)` identity is matched against
/// `current_manifest` slot-for-slot and a mismatch is REJECTED: the
/// `state_dimension` check alone passes a different entity (or lag) occupying the
/// same slot, silently attaching a cut's coefficient to the wrong state variable.
/// Only `state_dimension` must match — `num_stages` may differ (a monthly source
/// vs. a weekly+monthly current study); per-slot matching compares the source
/// stage's manifest to the current TERMINAL-stage manifest, both length
/// `state_dimension`.
///
/// `current_manifest` is built via
/// [`StudySetup::build_terminal_entity_manifest`](crate::StudySetup::build_terminal_entity_manifest)
/// (single owner of identity resolution, shared with the checkpoint writer). An
/// empty manifest (pre-manifest checkpoint) leaves the `state_dimension` check
/// standing and warns. A `was_active == false` boundary slot whose current
/// counterpart is active is a non-fatal divergence: warn, load the cut anyway.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read
/// - `source_stage` does not exist in the checkpoint
/// - The source stage's state dimension does not match `current_state_dimension`
/// - A populated boundary manifest disagrees with `current_manifest` in length or
///   in any slot's `(entity_type, entity_id, subindex)`
pub fn load_boundary_cuts(
    boundary_path: &std::path::Path,
    source_stage: u32,
    current_state_dimension: u32,
    current_manifest: &[cobre_io::EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) -> Result<Vec<cobre_io::OwnedPolicyCutRecord>, SddpError> {
    let checkpoint =
        cobre_io::output::policy::read_policy_checkpoint(boundary_path).map_err(|e| {
            SddpError::Validation(format!(
                "failed to read boundary policy checkpoint at {}: {e}",
                boundary_path.display()
            ))
        })?;

    let stage_result = checkpoint
        .stage_cuts
        .iter()
        .find(|sr| sr.stage_id == source_stage)
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "boundary policy: source_stage {} not found in checkpoint \
                 (available stages: {:?})",
                source_stage,
                checkpoint
                    .stage_cuts
                    .iter()
                    .map(|sr| sr.stage_id)
                    .collect::<Vec<_>>()
            ))
        })?;

    if stage_result.state_dimension != current_state_dimension {
        return Err(SddpError::Validation(format!(
            "boundary policy state_dimension mismatch: source stage {} has {}, \
             current study has {}",
            source_stage, stage_result.state_dimension, current_state_dimension
        )));
    }

    let boundary_manifest = &stage_result.entity_manifest;
    if boundary_manifest.is_empty() || current_manifest.is_empty() {
        on_warning(&format!(
            "boundary policy: entity manifest absent (boundary slots: {}, current slots: {}); \
             slot identity could not be verified, relying on state_dimension={} alone",
            boundary_manifest.len(),
            current_manifest.len(),
            current_state_dimension
        ));
        return Ok(stage_result.cuts.clone());
    }

    if boundary_manifest.len() != current_manifest.len() {
        return Err(SddpError::Validation(format!(
            "boundary policy manifest length mismatch: source stage {} has {} slots, \
             current study terminal stage has {}",
            source_stage,
            boundary_manifest.len(),
            current_manifest.len()
        )));
    }

    for (i, (boundary, current)) in boundary_manifest.iter().zip(current_manifest).enumerate() {
        if slot_identity(boundary) != slot_identity(current) {
            return Err(SddpError::Validation(format!(
                "boundary policy entity-identity mismatch at slot {i} (source stage {source_stage}): \
                 boundary (entity_type={}, entity_id={}, subindex={}) != \
                 current (entity_type={}, entity_id={}, subindex={}); \
                 the boundary cut coefficient at this slot would attach to the wrong state variable",
                boundary.entity_type,
                boundary.entity_id,
                boundary.subindex,
                current.entity_type,
                current.entity_id,
                current.subindex
            )));
        }
        if !boundary.was_active && current.was_active {
            on_warning(&format!(
                "boundary policy: slot {i} (entity_type={}, entity_id={}, subindex={}) was dormant \
                 at the boundary stage but is active in the current study; loading its boundary cut",
                current.entity_type, current.entity_id, current.subindex
            ));
        }
    }

    Ok(stage_result.cuts.clone())
}

/// Inject boundary cuts into the terminal stage of the study's FCF.
///
/// Replaces the terminal stage's [`CutPool`] with one pre-populated from
/// `boundary_records`, retaining capacity for new training cuts. The resulting
/// nonzero `warm_start_count` is what makes the forward pass treat the terminal
/// stage as boundary-loaded (`terminal_has_boundary_cuts`) and skip theta zeroing.
pub fn inject_boundary_cuts(
    setup: &mut StudySetup,
    boundary_records: &[cobre_io::OwnedPolicyCutRecord],
) {
    let fcf = &mut setup.fcf;
    let terminal_idx = fcf.pools.len() - 1;
    let state_dimension = fcf.state_dimension;
    let forward_passes = fcf.forward_passes;
    let existing_capacity = fcf.pools[terminal_idx].capacity;
    let existing_warm_start = fcf.pools[terminal_idx].warm_start_count as usize;
    let training_capacity = existing_capacity.saturating_sub(existing_warm_start);
    let max_iterations = if forward_passes > 0 {
        #[allow(clippy::cast_possible_truncation)]
        let m = (training_capacity / forward_passes as usize) as u64;
        m
    } else {
        0
    };
    fcf.pools[terminal_idx] = CutPool::new_with_warm_start(
        state_dimension,
        forward_passes,
        max_iterations,
        boundary_records,
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use cobre_io::{EntitySlot, PolicyCheckpointMetadata, StageCutsPayload};

    use super::{load_boundary_cuts, resolve_warm_start_counts, validate_policy_compatibility};

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Discard warnings: a `&mut dyn FnMut(&str)` for tests asserting only the
    /// `Result`.
    fn ignore_warnings() -> impl FnMut(&str) {
        |_| {}
    }

    /// Write a minimal policy checkpoint to `dir` with `n_stages` stages each
    /// having `state_dimension` state variables and the supplied cut intercepts,
    /// with no entity manifest (the pre-manifest checkpoint shape).
    ///
    /// Each stage gets `cuts.len()` cuts with coefficients all set to 1.0.
    fn write_minimal_checkpoint(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
    ) {
        write_checkpoint_with_manifest(dir, n_stages, state_dimension, cut_intercepts, &[]);
    }

    /// Like [`write_minimal_checkpoint`] but attaches `manifest` to every stage's
    /// cuts payload (an empty `manifest` reproduces the pre-manifest shape).
    fn write_checkpoint_with_manifest(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
        manifest: &[EntitySlot],
    ) {
        let state_dim = state_dimension as usize;
        let coefficients = vec![1.0_f64; state_dim];
        let n_cuts = cut_intercepts.len();

        let cut_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = (0..n_stages)
            .map(|_| {
                cut_intercepts
                    .iter()
                    .enumerate()
                    .map(|(i, &intercept)| cobre_io::PolicyCutRecord {
                        cut_id: i as u64,
                        slot_index: i as u32,
                        iteration: i as u32,
                        forward_pass_index: 0,
                        intercept,
                        coefficients: &coefficients,
                        is_active: true,
                    })
                    .collect()
            })
            .collect();

        let active_indices: Vec<Vec<u32>> = (0..n_stages)
            .map(|_| (0..n_cuts as u32).collect())
            .collect();

        let payloads: Vec<StageCutsPayload<'_>> = (0..n_stages as usize)
            .map(|s| StageCutsPayload {
                stage_id: s as u32,
                state_dimension,
                capacity: n_cuts as u32,
                warm_start_count: 0,
                cuts: &cut_records[s],
                active_cut_indices: &active_indices[s],
                populated_count: n_cuts as u32,
                entity_manifest: manifest,
            })
            .collect();

        let metadata = PolicyCheckpointMetadata {
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-14T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension,
            num_stages: n_stages,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
        };

        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    // ── load_boundary_cuts tests ──────────────────────────────────────────────

    /// Given a valid checkpoint with 12 stages and `state_dimension=10`, when
    /// `load_boundary_cuts` is called for stage 2 with matching dimension,
    /// then it returns `Ok` with the cuts from that stage.
    #[test]
    fn load_boundary_cuts_valid_stage() {
        let tmp = tempfile::tempdir().unwrap();
        let intercepts = vec![10.0, 20.0, 30.0];
        write_minimal_checkpoint(tmp.path(), 12, 10, &intercepts);

        let cuts = load_boundary_cuts(tmp.path(), 2, 10, &[], &mut ignore_warnings()).unwrap();

        assert_eq!(cuts.len(), 3, "should return all 3 cuts from stage 2");
        let returned_intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(
            returned_intercepts, intercepts,
            "intercepts should match written values"
        );
        for cut in &cuts {
            assert_eq!(
                cut.coefficients.len(),
                10,
                "each cut should have state_dimension=10 coefficients"
            );
        }
    }

    /// Given a checkpoint without stage 99, when `load_boundary_cuts` is called
    /// for stage 99, then it returns `Err(SddpError::Validation)` with a message
    /// containing `"source_stage"` and `"99"`.
    #[test]
    fn load_boundary_cuts_missing_stage_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(tmp.path(), 99, 10, &[], &mut ignore_warnings());

        assert!(result.is_err(), "should fail for missing stage");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("source_stage"),
            "error should mention 'source_stage': {msg}"
        );
        assert!(
            msg.contains("99"),
            "error should include the missing stage index: {msg}"
        );
    }

    /// Given a checkpoint with `state_dimension=10`, when `load_boundary_cuts` is
    /// called with `current_state_dimension=5`, then it returns
    /// `Err(SddpError::Validation)` with a message containing `"state_dimension"`.
    #[test]
    fn load_boundary_cuts_state_dimension_mismatch_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(tmp.path(), 0, 5, &[], &mut ignore_warnings());

        assert!(result.is_err(), "should fail for dimension mismatch");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "error should mention 'state_dimension': {msg}"
        );
    }

    /// Given a non-existent path, when `load_boundary_cuts` is called, then it
    /// returns `Err(SddpError::Validation)` with a message describing the failure.
    #[test]
    fn load_boundary_cuts_nonexistent_path_returns_error() {
        let result = load_boundary_cuts(
            std::path::Path::new("/nonexistent/path/to/policy"),
            0,
            10,
            &[],
            &mut ignore_warnings(),
        );

        assert!(result.is_err(), "should fail for non-existent path");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to read boundary policy checkpoint"),
            "error should describe the IO failure: {msg}"
        );
    }

    /// Build a 2-slot storage manifest with the given hydro ids, both active.
    fn storage_manifest(id0: i32, id1: i32) -> Vec<EntitySlot> {
        vec![
            EntitySlot {
                entity_type: 0,
                entity_id: id0,
                subindex: 0,
                was_active: true,
            },
            EntitySlot {
                entity_type: 0,
                entity_id: id1,
                subindex: 0,
                was_active: true,
            },
        ]
    }

    /// Given a checkpoint whose source-stage manifest matches the current study's
    /// terminal manifest slot-for-slot, `load_boundary_cuts` returns `Ok` with the
    /// source cuts and emits no warning.
    #[test]
    fn load_boundary_cuts_matching_manifest_loads_without_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut |m| {
            warnings.push(m.to_string());
        })
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot match must emit no warning: {warnings:?}"
        );
    }

    /// Given a boundary slot 0 with `entity_id` 7 but a current slot 0 with
    /// `entity_id` 9 (same `state_dimension`), `load_boundary_cuts` rejects with a
    /// `Validation` error naming slot `0`, `7`, and `9`.
    #[test]
    fn load_boundary_cuts_entity_id_mismatch_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(7, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(9, 2);
        let result = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut ignore_warnings());

        assert!(result.is_err(), "entity_id mismatch at slot 0 must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(
            msg.contains("entity_id=7"),
            "error must name the boundary id 7: {msg}"
        );
        assert!(
            msg.contains("entity_id=9"),
            "error must name the current id 9: {msg}"
        );
    }

    /// Given a boundary slot 1 typed `HydroInflowLag` (type 1) but a current slot 1
    /// typed `HydroStorage` (type 0), `load_boundary_cuts` rejects naming slot 1 and
    /// the differing entity types.
    #[test]
    fn load_boundary_cuts_type_mismatch_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let mut boundary = storage_manifest(1, 2);
        boundary[1].entity_type = 1; // HydroInflowLag
        boundary[1].subindex = 1;
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(1, 2);
        let result = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut ignore_warnings());

        assert!(result.is_err(), "type mismatch at slot 1 must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 1"), "error must name slot 1: {msg}");
        assert!(
            msg.contains("entity_type=1"),
            "error must name the boundary type 1: {msg}"
        );
        assert!(
            msg.contains("entity_type=0"),
            "error must name the current type 0: {msg}"
        );
    }

    /// Given a boundary checkpoint with an empty manifest and a matching
    /// `current_state_dimension`, `load_boundary_cuts` returns `Ok` (no hard fail on
    /// absence) and surfaces an "identity could not be verified" warning.
    #[test]
    fn load_boundary_cuts_absent_manifest_loads_with_warning() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &[]);

        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut |m| {
            warnings.push(m.to_string());
        })
        .unwrap();

        assert_eq!(cuts.len(), 2, "absent manifest must still load cuts");
        assert_eq!(warnings.len(), 1, "absence must surface one warning");
        assert!(
            warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            warnings[0]
        );
    }

    /// Given a boundary slot whose identity matches the current study but whose
    /// `was_active` is `false` while the current study treats it as active,
    /// `load_boundary_cuts` returns `Ok` (cut loaded) and surfaces a `was_active`
    /// divergence warning.
    #[test]
    fn load_boundary_cuts_was_active_divergence_warns_and_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let mut boundary = storage_manifest(1, 2);
        boundary[1].was_active = false; // dormant at the boundary stage
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut |m| {
            warnings.push(m.to_string());
        })
        .unwrap();

        assert_eq!(cuts.len(), 2, "was_active divergence must still load cuts");
        assert_eq!(warnings.len(), 1, "divergence must surface one warning");
        assert!(
            warnings[0].contains("dormant") && warnings[0].contains("slot 1"),
            "warning must flag slot 1's dormancy divergence: {}",
            warnings[0]
        );
    }

    fn sample_metadata() -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            cobre_version: "0.2.2".to_string(),
            created_at: "2026-03-29T00:00:00Z".to_string(),
            completed_iterations: 50,
            final_lower_bound: 1234.56,
            best_upper_bound: Some(1300.0),
            state_dimension: 10,
            num_stages: 12,
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 42,
            total_visited_states: 0,
        }
    }

    #[test]
    fn compatible_metadata_passes() {
        let meta = sample_metadata();
        assert!(validate_policy_compatibility(&meta, 10, 12).is_ok());
    }

    #[test]
    fn state_dimension_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 8, 12);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include policy value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
    }

    #[test]
    fn num_stages_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 10, 24);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include policy value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");
    }

    #[test]
    fn both_dimensions_mismatched_returns_err() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 8, 24);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "should report state_dimension mismatch first: {msg}"
        );
    }

    // ── resolve_warm_start_counts tests ───────────────────────────────────────

    fn meta_with_counts(
        warm_start_cuts: u32,
        warm_start_counts: Vec<u32>,
    ) -> PolicyCheckpointMetadata {
        #[allow(clippy::cast_possible_truncation)]
        let num_stages: u32 = if warm_start_counts.is_empty() {
            3
        } else {
            warm_start_counts.len() as u32
        };
        PolicyCheckpointMetadata {
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-01T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 2,
            num_stages,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts,
            warm_start_counts,
            rng_seed: 0,
            total_visited_states: 0,
        }
    }

    #[test]
    fn resolve_warm_start_counts_new_format_returns_per_stage_counts() {
        let meta = meta_with_counts(10, vec![10, 8, 6]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![10u32, 8, 6]);
    }

    #[test]
    fn resolve_warm_start_counts_old_format_broadcasts_scalar() {
        // Empty warm_start_counts: fall back to warm_start_cuts broadcast.
        let meta = meta_with_counts(5, vec![]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![5u32, 5, 5]);
    }

    #[test]
    fn resolve_warm_start_counts_old_format_zero_scalar_broadcasts_zeros() {
        let meta = meta_with_counts(0, vec![]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![0u32, 0, 0]);
    }

    #[test]
    fn resolve_warm_start_counts_wrong_length_returns_validation_error() {
        // warm_start_counts has 2 entries but num_stages is 3 — corrupted checkpoint.
        let meta = meta_with_counts(5, vec![5, 5]);
        let result = resolve_warm_start_counts(&meta, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("warm_start_counts length mismatch"),
            "error message should mention length mismatch: {msg}"
        );
        assert!(msg.contains('2'), "should include vector length: {msg}");
        assert!(msg.contains('3'), "should include num_stages: {msg}");
    }

    #[test]
    fn resolve_warm_start_counts_single_stage_new_format() {
        let meta = meta_with_counts(7, vec![7]);
        let counts = resolve_warm_start_counts(&meta, 1).unwrap();
        assert_eq!(counts, vec![7u32]);
    }

    #[test]
    fn resolve_warm_start_counts_zero_stages_old_format_returns_empty() {
        let meta = meta_with_counts(5, vec![]);
        let counts = resolve_warm_start_counts(&meta, 0).unwrap();
        assert!(counts.is_empty());
    }
}
