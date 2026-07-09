//! Policy loading and compatibility validation.
//!
//! [`validate_policy_load`] is the single entry point for compatibility
//! validation; every load path (full-FCF warm-start/resume/simulation-only and
//! boundary-cut injection) routes through it.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction

use cobre_io::PolicyCheckpointMetadata;
use cobre_solver::{Basis, BasisStatus};

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

/// Per-side state layout fed to [`validate_policy_load`]: one manifest for the
/// loaded policy (`source`) and one for the study being trained or simulated
/// (`current`). The caller builds both — one from checkpoint metadata and its
/// entity manifest, the other from the live [`StudySetup`].
#[derive(Debug, Clone, Copy)]
pub struct PolicyStageManifest<'a> {
    /// Length of the state vector (one entry per reservoir/lag/bucket dimension).
    pub state_dimension: u32,
    /// Number of stages in the study.
    pub num_stages: u32,
    /// Per-slot entity identity, in state-vector order.
    pub slots: &'a [cobre_io::EntitySlot],
}

/// Which policy-load path is being validated; selects the `num_stages` rule in
/// [`validate_policy_load`]'s check matrix (`state_dimension` and per-slot
/// identity are checked unconditionally on both variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyLoadKind {
    /// Full future-cost-function load (warm-start, resume, simulation-only):
    /// `num_stages` must match `current` exactly.
    FullFcf,
    /// Single-stage boundary-cut injection into the terminal pool: `num_stages`
    /// is unchecked (a monthly source may feed a weekly+monthly current study).
    ///
    /// Reserved for the boundary-coupling family-fill payload (not yet wired):
    /// a future field on this variant will carry the entity-family re-indexing
    /// rules that replace today's positional cut copy.
    BoundaryInjection,
}

/// Non-fatal findings from [`validate_policy_load`]; warnings never abort the load.
#[derive(Debug, Clone, Default)]
pub struct CompatibilityReport {
    /// Human-readable warning messages, in emission order.
    pub warnings: Vec<String>,
}

/// Validate that `source`'s state layout is compatible with `current`'s, per
/// `kind`'s check matrix: `state_dimension` equality and per-slot identity
/// (delegated to [`compare_manifest_slot_identity`]) are hard-rejected on both
/// variants; `num_stages` equality is hard-rejected only on
/// [`PolicyLoadKind::FullFcf`]. `col_scale`/scaling is never a compatibility
/// dimension. This is the single entry point for policy-load validation; every
/// load path calls it.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a `state_dimension` mismatch, a
/// `num_stages` mismatch under [`PolicyLoadKind::FullFcf`], or a per-slot
/// identity mismatch (see [`compare_manifest_slot_identity`]).
pub fn validate_policy_load(
    source: &PolicyStageManifest<'_>,
    current: &PolicyStageManifest<'_>,
    kind: PolicyLoadKind,
) -> Result<CompatibilityReport, SddpError> {
    if source.state_dimension != current.state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {}",
            source.state_dimension, current.state_dimension
        )));
    }

    if kind == PolicyLoadKind::FullFcf && source.num_stages != current.num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            source.num_stages, current.num_stages
        )));
    }

    let mut warnings = Vec::new();
    compare_manifest_slot_identity(source.slots, current.slots, &mut |msg| {
        warnings.push(msg.to_string());
    })?;

    Ok(CompatibilityReport { warnings })
}

/// Decode a checkpoint status vector, preferring the lossless `canonical` codes
/// and falling back to the legacy `HiGHS`-code-space `legacy` codes only when
/// `canonical` is empty (a file written before the canonical fields existed).
fn decode_status_vector(canonical: &[u8], legacy: &[u8]) -> Vec<BasisStatus> {
    if canonical.is_empty() {
        legacy
            .iter()
            .map(|&c| BasisStatus::from_highs_code(i32::from(c)))
            .collect()
    } else {
        canonical
            .iter()
            .map(|&c| BasisStatus::from_discriminant_code(c))
            .collect()
    }
}

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<CapturedBasis>>`, one entry per stage; stages without a
/// matching record get `None`. A record's canonical status vectors decode via
/// `from_discriminant_code` (lossless, the mirror of `convert_basis_cache`'s
/// export-side `to_discriminant_code`); an old file carrying only the legacy
/// `HiGHS`-code-space vectors falls back to `from_highs_code`, preserving that
/// format's `Superbasic`→`Nonbasic`/`Fixed`→`Lower` fold.
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
        let col_status = decode_status_vector(&record.col_status_canonical, &record.column_status);
        let row_status = decode_status_vector(&record.row_status_canonical, &record.row_status);

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

/// Compare two entity manifests slot-for-slot by `slot_identity`.
///
/// `source` (a loaded policy's manifest) and `current` (the current study's
/// terminal manifest) describe the same-length state vector of two studies. A
/// per-slot `(entity_type, entity_id, subindex)` mismatch means a cut coefficient
/// would attach to the wrong state variable and is REJECTED. `was_active` is
/// excluded from `slot_identity` (a slot whose entity merely changed activity is
/// still the same state variable); a `source`-dormant slot now active only warns.
/// An empty manifest on either side (a pre-manifest checkpoint) cannot be
/// verified: warn and return `Ok`, leaving the caller's `state_dimension` check
/// standing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `source` and `current` differ in length
/// or in any slot's `(entity_type, entity_id, subindex)`.
pub fn compare_manifest_slot_identity(
    source: &[cobre_io::EntitySlot],
    current: &[cobre_io::EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) -> Result<(), SddpError> {
    if source.is_empty() || current.is_empty() {
        on_warning(&format!(
            "entity manifest absent (source slots: {}, current slots: {}); slot identity \
             could not be verified, relying on state_dimension alone",
            source.len(),
            current.len(),
        ));
        return Ok(());
    }

    if source.len() != current.len() {
        return Err(SddpError::Validation(format!(
            "entity manifest length mismatch: source has {} slots, current study has {}",
            source.len(),
            current.len()
        )));
    }

    for (i, (src, cur)) in source.iter().zip(current).enumerate() {
        if slot_identity(src) != slot_identity(cur) {
            return Err(SddpError::Validation(format!(
                "entity-identity mismatch at slot {i}: \
                 source (entity_type={}, entity_id={}, subindex={}) != \
                 current (entity_type={}, entity_id={}, subindex={}); \
                 the cut coefficient at this slot would attach to the wrong state variable",
                src.entity_type,
                src.entity_id,
                src.subindex,
                cur.entity_type,
                cur.entity_id,
                cur.subindex
            )));
        }
        if !src.was_active && cur.was_active {
            on_warning(&format!(
                "slot {i} (entity_type={}, entity_id={}, subindex={}) was dormant in the source \
                 policy but is active in the current study; loading its cut",
                cur.entity_type, cur.entity_id, cur.subindex
            ));
        }
    }

    Ok(())
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
/// Delegates to [`validate_policy_load`] with [`PolicyLoadKind::BoundaryInjection`].
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

    let source = PolicyStageManifest {
        state_dimension: stage_result.state_dimension,
        num_stages: checkpoint.metadata.num_stages,
        slots: &stage_result.entity_manifest,
    };
    let current = PolicyStageManifest {
        state_dimension: current_state_dimension,
        num_stages: checkpoint.metadata.num_stages,
        slots: current_manifest,
    };
    let report = validate_policy_load(&source, &current, PolicyLoadKind::BoundaryInjection)?;
    for warning in &report.warnings {
        on_warning(warning);
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
    #[allow(clippy::cast_possible_truncation)]
    let max_iterations = if forward_passes > 0 {
        (training_capacity / forward_passes as usize) as u64
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

    use super::{
        PolicyLoadKind, PolicyStageManifest, compare_manifest_slot_identity, load_boundary_cuts,
        resolve_warm_start_counts, validate_policy_load,
    };
    use crate::SddpError;

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
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
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
        vec![storage_slot(id0), storage_slot(id1)]
    }

    /// A single active `HydroStorage` slot (`entity_type 0`, `subindex 0`).
    fn storage_slot(id: i32) -> EntitySlot {
        EntitySlot {
            entity_type: 0,
            entity_id: id,
            subindex: 0,
            was_active: true,
            delivery_anchor: cobre_io::ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL,
        }
    }

    /// A single active `HydroTransitBucket` slot (`entity_type 3`): `id` is the
    /// downstream hydro, `lag` the maturity subindex.
    fn transit_bucket_slot(id: i32, lag: u32) -> EntitySlot {
        EntitySlot {
            entity_type: 3,
            entity_id: id,
            subindex: lag,
            was_active: true,
            delivery_anchor: cobre_io::ENTITY_SLOT_DELIVERY_ANCHOR_SENTINEL,
        }
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

    /// A manifest carrying a `HydroTransitBucket` slot (`entity_type 3`, the
    /// downstream hydro id, the maturity lag as `subindex`) round-trips: written to a
    /// checkpoint and reloaded against a slot-for-slot matching current manifest,
    /// `load_boundary_cuts` returns `Ok` with no warning — the bucket slot passes
    /// `slot_identity`.
    #[test]
    fn load_boundary_cuts_matching_transit_bucket_manifest_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut |m| {
            warnings.push(m.to_string());
        })
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching bucket manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot bucket match must emit no warning: {warnings:?}"
        );
    }

    /// A policy exported WITHOUT travel-time buckets (slot 1 typed `HydroStorage`)
    /// loaded by bucket-aware code whose terminal manifest types slot 1 as
    /// `HydroTransitBucket` (type 3) is rejected: the per-slot identity check names
    /// slot 1 and the differing types. Asserts the `load_boundary_cuts` rejection
    /// primitive fires on the bucket-vs-non-bucket case (the end-to-end force-on
    /// wiring lands separately).
    #[test]
    fn load_boundary_cuts_missing_transit_bucket_slot_identity_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let result = load_boundary_cuts(tmp.path(), 0, 2, &current, &mut ignore_warnings());

        assert!(
            result.is_err(),
            "bucket-vs-non-bucket at slot 1 must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 1"), "error must name slot 1: {msg}");
        assert!(
            msg.contains("entity_type=3"),
            "error must name the current bucket type 3: {msg}"
        );
        assert!(
            msg.contains("entity_type=0"),
            "error must name the boundary storage type 0: {msg}"
        );
    }

    /// A policy exported WITHOUT travel-time buckets has a smaller `state_dimension`
    /// than the bucket-aware current study, so `load_boundary_cuts` rejects on the
    /// `state_dimension` guard before per-slot identity. Pairs with the force-on
    /// wiring that lands separately.
    #[test]
    fn load_boundary_cuts_no_transit_bucket_export_dimension_mismatch_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &storage_manifest(1, 2));

        let current = vec![storage_slot(1), storage_slot(2), transit_bucket_slot(2, 1)];
        let result = load_boundary_cuts(tmp.path(), 0, 3, &current, &mut ignore_warnings());

        assert!(
            result.is_err(),
            "no-bucket export vs bucket study must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "error must cite state_dimension: {msg}"
        );
    }

    // ── compare_manifest_slot_identity tests ──────────────────────────────────

    /// Two same-length manifests differing only at slot 0's `entity_id` (7 vs 9)
    /// are rejected, naming slot `0` and both ids.
    #[test]
    fn compare_manifest_slot_identity_same_dim_different_id_rejects() {
        let source = storage_manifest(7, 2);
        let current = storage_manifest(9, 2);

        let result = compare_manifest_slot_identity(&source, &current, &mut ignore_warnings());

        assert!(result.is_err(), "different entity_id at slot 0 must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(msg.contains("entity_id=7"), "error must name id 7: {msg}");
        assert!(msg.contains("entity_id=9"), "error must name id 9: {msg}");
    }

    /// An empty `source` manifest cannot be verified: warn once and return `Ok`.
    #[test]
    fn compare_manifest_slot_identity_empty_source_warns_and_oks() {
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&[], &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "empty manifest must not hard-fail");
        assert_eq!(
            warnings.len(),
            1,
            "absence must surface exactly one warning"
        );
        assert!(
            warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            warnings[0]
        );
    }

    /// Identical manifests pass with no warning.
    #[test]
    fn compare_manifest_slot_identity_identical_oks_without_warning() {
        let source = storage_manifest(1, 2);
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&source, &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "identical manifests must pass");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot match must emit no warning: {warnings:?}"
        );
    }

    /// A `source`-dormant slot whose current counterpart is active warns but
    /// loads (`Ok`).
    #[test]
    fn compare_manifest_slot_identity_was_active_divergence_warns_and_oks() {
        let mut source = storage_manifest(1, 2);
        source[1].was_active = false;
        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();

        let result = compare_manifest_slot_identity(&source, &current, &mut |m| {
            warnings.push(m.to_string());
        });

        assert!(result.is_ok(), "was_active divergence must not hard-fail");
        assert_eq!(warnings.len(), 1, "divergence must surface one warning");
        assert!(
            warnings[0].contains("dormant") && warnings[0].contains("slot 1"),
            "warning must flag slot 1's dormancy divergence: {}",
            warnings[0]
        );
    }

    /// The full-FCF terminal-manifest shape: a checkpoint terminal manifest
    /// `[storage(7), storage(2)]` vs a current terminal manifest
    /// `[storage(9), storage(2)]` at equal `state_dimension` is rejected with a
    /// `Validation` error naming slot `0` — the same guard
    /// `load_and_validate_checkpoint` applies after the dims/`num_stages` check.
    #[test]
    fn compare_manifest_full_fcf_terminal_entity_swap_rejects() {
        let checkpoint_terminal = storage_manifest(7, 2);
        let current_terminal = storage_manifest(9, 2);

        let result = compare_manifest_slot_identity(
            &checkpoint_terminal,
            &current_terminal,
            &mut ignore_warnings(),
        );

        assert!(
            matches!(result, Err(SddpError::Validation(_))),
            "same-dimension terminal entity swap must be a Validation error"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(
            msg.contains("entity_id=7") && msg.contains("entity_id=9"),
            "error must name both diverging ids: {msg}"
        );
    }

    // ── validate_policy_load tests ────────────────────────────────────────────

    /// Identical `state_dimension`, `num_stages`, and slot-for-slot matching
    /// manifests pass `FullFcf` with no warnings.
    #[test]
    fn validate_policy_load_full_fcf_identical_oks_without_warning() {
        let slots = storage_manifest(1, 2);
        let source = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &slots,
        };

        let report = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf).unwrap();

        assert!(
            report.warnings.is_empty(),
            "identical manifests must emit no warning: {:?}",
            report.warnings
        );
    }

    /// A `state_dimension` mismatch is a hard reject on `FullFcf`.
    #[test]
    fn validate_policy_load_full_fcf_state_dimension_mismatch_rejects() {
        let slots = storage_manifest(1, 2);
        let source = PolicyStageManifest {
            state_dimension: 10,
            num_stages: 12,
            slots: &slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 8,
            num_stages: 12,
            slots: &slots,
        };

        let result = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf);

        assert!(result.is_err(), "state_dimension mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include source value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
    }

    /// A `num_stages` mismatch is a hard reject on `FullFcf` but the identical
    /// inputs pass `BoundaryInjection` (unchecked there).
    #[test]
    fn validate_policy_load_num_stages_mismatch_rejects_full_fcf_oks_boundary() {
        let slots = storage_manifest(1, 2);
        let source = PolicyStageManifest {
            state_dimension: 10,
            num_stages: 12,
            slots: &slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 10,
            num_stages: 24,
            slots: &slots,
        };

        let full_fcf_result = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf);
        assert!(
            full_fcf_result.is_err(),
            "num_stages mismatch must reject FullFcf"
        );
        let msg = full_fcf_result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include source value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");

        let boundary_result =
            validate_policy_load(&source, &current, PolicyLoadKind::BoundaryInjection);
        assert!(
            boundary_result.is_ok(),
            "num_stages is unchecked under BoundaryInjection: {boundary_result:?}"
        );
    }

    /// Both `state_dimension` (10 vs 8) and `num_stages` (12 vs 24) mismatch under
    /// `FullFcf`; the `state_dimension` guard fires first, so the error names
    /// `state_dimension`, not `num_stages`.
    #[test]
    fn validate_policy_load_full_fcf_both_dimensions_mismatched_rejects() {
        let slots = storage_manifest(1, 2);
        let source = PolicyStageManifest {
            state_dimension: 10,
            num_stages: 12,
            slots: &slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 8,
            num_stages: 24,
            slots: &slots,
        };

        let result = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf);

        assert!(result.is_err(), "both-dimension mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "should report state_dimension mismatch first: {msg}"
        );
    }

    /// A per-slot identity mismatch is a hard reject under `FullFcf`, naming the
    /// mismatched slot and both diverging `entity_id`s.
    #[test]
    fn validate_policy_load_full_fcf_slot_mismatch_rejects() {
        let source_slots = storage_manifest(7, 2);
        let current_slots = storage_manifest(9, 2);
        let source = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &source_slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &current_slots,
        };

        let result = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf);

        assert!(result.is_err(), "slot identity mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
        assert!(
            msg.contains("entity_id=7"),
            "error must name the source id 7: {msg}"
        );
        assert!(
            msg.contains("entity_id=9"),
            "error must name the current id 9: {msg}"
        );
    }

    /// A per-slot identity mismatch is a hard reject under `BoundaryInjection`
    /// too — slot identity is checked on both kinds, unlike `num_stages`.
    #[test]
    fn validate_policy_load_boundary_injection_slot_mismatch_rejects() {
        let source_slots = storage_manifest(7, 2);
        let current_slots = storage_manifest(9, 2);
        let source = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &source_slots,
        };
        let current = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 6,
            slots: &current_slots,
        };

        let result = validate_policy_load(&source, &current, PolicyLoadKind::BoundaryInjection);

        assert!(result.is_err(), "slot identity mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("slot 0"), "error must name slot 0: {msg}");
    }

    /// An empty manifest on either side cannot be verified by slot identity:
    /// `validate_policy_load` falls back to the `state_dimension` check alone,
    /// returning `Ok` with one warning.
    #[test]
    fn validate_policy_load_full_fcf_empty_manifest_oks_with_warning() {
        let current_slots = storage_manifest(1, 2);
        let source = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &[],
        };
        let current = PolicyStageManifest {
            state_dimension: 2,
            num_stages: 12,
            slots: &current_slots,
        };

        let report = validate_policy_load(&source, &current, PolicyLoadKind::FullFcf).unwrap();

        assert_eq!(report.warnings.len(), 1, "absence must surface one warning");
        assert!(
            report.warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            report.warnings[0]
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
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
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

    // ── basis-status export+load round-trip (§E6) ─────────────────────────────

    /// Every one of the seven `BasisStatus` variants survives the full
    /// export→disk→load path losslessly — including the CLP-only `Superbasic`
    /// and `Fixed`, which the legacy `HiGHS`-code path folded onto `Nonbasic`/
    /// `Lower`. Encode via `convert_basis_cache`, round-trip through the codec's
    /// canonical fields, decode via `build_basis_cache_from_checkpoint`.
    #[test]
    fn basis_status_round_trips_through_export_and_load_for_every_variant() {
        use cobre_io::{PolicyBasisRecord, deserialize_stage_basis, serialize_stage_basis};
        use cobre_solver::{Basis, BasisStatus};

        use super::build_basis_cache_from_checkpoint;
        use crate::TrainingResult;
        use crate::policy_export::convert_basis_cache;
        use crate::workspace::CapturedBasis;

        let col_status = vec![
            BasisStatus::Lower,
            BasisStatus::Basic,
            BasisStatus::Upper,
            BasisStatus::Zero,
            BasisStatus::Nonbasic,
            BasisStatus::Superbasic,
            BasisStatus::Fixed,
        ];
        // Reversed so the column and row vectors are checked independently.
        let mut row_status = col_status.clone();
        row_status.reverse();

        let captured = CapturedBasis {
            basis: Basis {
                col_status: col_status.clone(),
                row_status: row_status.clone(),
            },
            base_row_count: row_status.len(),
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
        };
        let training_result = TrainingResult::new(
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            "test".to_string(),
            0,
            vec![Some(captured)],
            Vec::new(),
            None,
            None,
        );

        let (col_u8, row_u8) = convert_basis_cache(&training_result);

        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &[],
            row_status: &[],
            num_cut_rows: 0,
            col_status_canonical: &col_u8[0],
            row_status_canonical: &row_u8[0],
        };
        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("codec round-trip must succeed");

        let cache = build_basis_cache_from_checkpoint(1, std::slice::from_ref(&owned), &[]);
        let recovered = cache[0].as_ref().expect("stage 0 basis must be present");

        assert_eq!(
            recovered.basis.col_status, col_status,
            "every column variant must round-trip losslessly"
        );
        assert_eq!(
            recovered.basis.row_status, row_status,
            "every row variant must round-trip losslessly, including Superbasic/Fixed"
        );
    }

    /// Backward compatibility: an old-file basis record carrying ONLY the legacy
    /// `HiGHS`-code-space vectors (canonical fields empty) loads through the
    /// `from_highs_code` fallback. A legacy byte outside the `HiGHS` `0..=4` range
    /// folds to `Nonbasic` (never `Superbasic`/`Fixed`, which the canonical
    /// decoder would yield), proving the legacy branch is the one selected.
    #[test]
    fn legacy_only_basis_record_loads_via_highs_fold() {
        use cobre_solver::BasisStatus;

        use super::build_basis_cache_from_checkpoint;

        let legacy = cobre_io::OwnedPolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: vec![0, 1, 2, 3, 4, 6],
            row_status: vec![4, 3, 2, 1, 0],
            num_cut_rows: 0,
            col_status_canonical: Vec::new(),
            row_status_canonical: Vec::new(),
        };

        let cache = build_basis_cache_from_checkpoint(1, std::slice::from_ref(&legacy), &[]);
        let recovered = cache[0].as_ref().expect("stage 0 basis must be present");

        assert_eq!(
            recovered.basis.col_status,
            vec![
                BasisStatus::Lower,
                BasisStatus::Basic,
                BasisStatus::Upper,
                BasisStatus::Zero,
                BasisStatus::Nonbasic,
                BasisStatus::Nonbasic,
            ],
            "legacy bytes decode via from_highs_code; the out-of-range 6 folds to Nonbasic"
        );
        assert_eq!(
            recovered.basis.row_status,
            vec![
                BasisStatus::Nonbasic,
                BasisStatus::Zero,
                BasisStatus::Upper,
                BasisStatus::Basic,
                BasisStatus::Lower,
            ],
            "legacy row bytes decode via from_highs_code"
        );
    }
}
