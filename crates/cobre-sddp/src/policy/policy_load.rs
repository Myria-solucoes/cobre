//! Policy loading and compatibility validation.
//!
//! [`validate_policy_load`] is the single entry point for compatibility
//! validation; every load path (full-FCF warm-start/resume/simulation-only and
//! boundary-cut injection) routes through it and returns a [`PolicyLoadProof`]
//! kind-typed to [`FullFcf`] or [`BoundaryInjection`] — the only way to obtain
//! one, so [`FutureCostFunction::new_with_warm_start`],
//! [`FutureCostFunction::from_deserialized`], and [`inject_boundary_cuts`]
//! cannot compile against unvalidated data.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction
//! [`FutureCostFunction::new_with_warm_start`]: crate::FutureCostFunction::new_with_warm_start
//! [`FutureCostFunction::from_deserialized`]: crate::FutureCostFunction::from_deserialized

use cobre_io::EntitySlot;
use cobre_io::OwnedPolicyBasisRecord;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyCheckpointMetadata;
use cobre_io::StageCutsReadResult;
use cobre_io::read_policy_checkpoint;
use cobre_solver::{Basis, BasisStatus};

use crate::SddpError;
use crate::cut::pool::CutPool;
use crate::setup::StudySetup;
use crate::workspace::CapturedBasis;

use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;

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

/// The constant a policy checkpoint written before `cost_scale_factor`
/// provenance was recorded (§5 Option A) was unconditionally scaled at — the
/// pre-this-feature hard-coded `COST_SCALE_FACTOR`. A checkpoint whose
/// `metadata.cost_scale_factor` is `None` is interpreted under this constant.
pub const LEGACY_COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Rescale one stage's cut records (§5 Option A) from their at-rest
/// representation into the LOADING study's internal scaled cost space.
///
/// - **Marked** (`source_cost_scale_factor: Some(s)`): the checkpoint holds
///   canonical currency units — export multiplied every value by the writing
///   study's own `s` ([`crate::policy_export::scale_cut_records_for_export`]).
///   Every value here is divided by `loading_cost_scale_factor`, UNCONDITIONALLY
///   — even when `s` equals `loading_cost_scale_factor` — since the file
///   already carries one export-side rounding; a second division is the
///   accepted same-factor ULP drift (§5 Option A trade-off), never special-cased
///   away.
/// - **Legacy** (`None`): the checkpoint holds the writing study's OWN internal
///   scaled values under the hard-coded historical constant
///   [`LEGACY_COST_SCALE_FACTOR`] — no export-side multiply ever ran, because
///   Option A did not exist yet. When `loading_cost_scale_factor ==
///   LEGACY_COST_SCALE_FACTOR` (the overwhelmingly common case: every existing
///   policy directory read at the still-default factor) this is an exact
///   no-op, bit-identical to pre-this-feature behavior — skipping it is a
///   correctness requirement, not an optimization: a legacy checkpoint loaded
///   at the default factor is NOT part of the one-time re-baseline the ticket
///   scopes to newly-written Option-A files. Otherwise every value is
///   multiplied by `LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor`.
pub(crate) fn rescale_cut_records_for_load(
    records: &mut [OwnedPolicyCutRecord],
    source_cost_scale_factor: Option<f64>,
    loading_cost_scale_factor: f64,
) {
    if source_cost_scale_factor.is_some() {
        for cut in records {
            cut.intercept /= loading_cost_scale_factor;
            for c in &mut cut.coefficients {
                *c /= loading_cost_scale_factor;
            }
        }
        return;
    }

    if loading_cost_scale_factor == LEGACY_COST_SCALE_FACTOR {
        return;
    }
    let ratio = LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor;
    for cut in records {
        cut.intercept *= ratio;
        for c in &mut cut.coefficients {
            *c *= ratio;
        }
    }
}

/// [`rescale_cut_records_for_load`] applied to every stage of a full policy
/// checkpoint — the [`FullFcf`] load path (training warm-start/resume and
/// simulation-only runs both route through this before the records reach
/// [`crate::FutureCostFunction::from_deserialized`] /
/// [`crate::FutureCostFunction::new_with_warm_start`]).
pub fn rescale_checkpoint_cuts_for_load(
    stage_cuts: &mut [StageCutsReadResult],
    source_cost_scale_factor: Option<f64>,
    loading_cost_scale_factor: f64,
) {
    for stage in stage_cuts {
        rescale_cut_records_for_load(
            &mut stage.cuts,
            source_cost_scale_factor,
            loading_cost_scale_factor,
        );
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
    pub slots: &'a [EntitySlot],
}

mod sealed {
    pub trait Sealed {}
}

/// Selects [`validate_policy_load`]'s check matrix (`state_dimension` and
/// per-slot identity are checked unconditionally for every kind). Sealed so
/// [`FullFcf`] and [`BoundaryInjection`] are the only implementors, making a
/// [`PolicyLoadProof<K>`] a proof of validation under exactly one real kind —
/// never a third, uncatalogued one.
pub trait PolicyLoadKind: sealed::Sealed {
    /// Whether `num_stages` equality is hard-rejected for this kind.
    const CHECK_NUM_STAGES: bool;
}

/// Full future-cost-function load (warm-start, resume, simulation-only):
/// `num_stages` must match `current` exactly.
#[derive(Debug, Clone, Copy)]
pub struct FullFcf;

impl sealed::Sealed for FullFcf {}
impl PolicyLoadKind for FullFcf {
    const CHECK_NUM_STAGES: bool = true;
}

/// Single-stage boundary-cut injection into the terminal pool: `num_stages`
/// is unchecked (a monthly source may feed a weekly+monthly current study).
#[derive(Debug, Clone, Copy)]
pub struct BoundaryInjection;

impl sealed::Sealed for BoundaryInjection {}
impl PolicyLoadKind for BoundaryInjection {
    const CHECK_NUM_STAGES: bool = false;
}

/// Unforgeable, kind-typed evidence that [`validate_policy_load`] accepted a
/// `source`/`current` pair for load kind `K`. The private marker field means a
/// struct literal cannot be written outside this module, so a consumer
/// requiring `&PolicyLoadProof<K>` cannot compile against unvalidated data —
/// and a proof typed to the wrong `K` cannot substitute, since `K` is a
/// distinct type per kind.
#[derive(Debug)]
pub struct PolicyLoadProof<K: PolicyLoadKind> {
    /// Human-readable warning messages, in emission order.
    pub warnings: Vec<String>,
    _kind: PhantomData<K>,
}

/// Validate that `source`'s state layout is compatible with `current`'s, per
/// `K`'s check matrix: `state_dimension` equality and per-slot identity
/// (delegated to [`compare_manifest_slot_identity`]) are hard-rejected for
/// every kind; `num_stages` equality is hard-rejected only when
/// `K::CHECK_NUM_STAGES` ([`FullFcf`]). `col_scale`/scaling is never a
/// compatibility dimension. This is the single entry point for policy-load
/// validation — its success is the only way to construct a
/// [`PolicyLoadProof<K>`], so every load path routes through it.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a `state_dimension` mismatch, a
/// `num_stages` mismatch under [`FullFcf`], or a per-slot identity mismatch
/// (see [`compare_manifest_slot_identity`]).
pub fn validate_policy_load<K: PolicyLoadKind>(
    source: &PolicyStageManifest<'_>,
    current: &PolicyStageManifest<'_>,
) -> Result<PolicyLoadProof<K>, SddpError> {
    if source.state_dimension != current.state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {}",
            source.state_dimension, current.state_dimension
        )));
    }

    if K::CHECK_NUM_STAGES && source.num_stages != current.num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            source.num_stages, current.num_stages
        )));
    }

    let mut warnings = Vec::new();
    compare_manifest_slot_identity(source.slots, current.slots, &mut |msg| {
        warnings.push(msg.to_string());
    })?;

    Ok(PolicyLoadProof {
        warnings,
        _kind: PhantomData,
    })
}

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<CapturedBasis>>`, one entry per stage; stages without a
/// matching record get `None`. `u8` status codes decode via `from_discriminant_code`,
/// the mirror of `convert_basis_cache`'s export-side `to_discriminant_code`; a
/// pre-existing checkpoint (bytes `0..=4`) decodes identically, since that range
/// means the same in the canonical and `HiGHS` code spaces.
///
/// # Cut-slot reconstruction
///
/// `row_status` is `[template rows…, cut rows…]`, the trailing `num_cut_rows` in
/// capture-time [`CutPool::active_cuts`](crate::cut::pool::CutPool::active_cuts)
/// order (active slots, increasing). Slot identity is recovered by matching each
/// basis record to its [`StageCutsReadResult`] by
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
    stage_bases: &[OwnedPolicyBasisRecord],
    stage_cuts: &[StageCutsReadResult],
) -> Vec<Option<CapturedBasis>> {
    let mut cache: Vec<Option<CapturedBasis>> = vec![None; num_stages];
    for record in stage_bases {
        let stage = record.stage_id as usize;
        if stage >= num_stages {
            continue;
        }
        let col_status: Vec<BasisStatus> = record
            .column_status
            .iter()
            .map(|&c| BasisStatus::from_discriminant_code(c))
            .collect();
        let row_status: Vec<BasisStatus> = record
            .row_status
            .iter()
            .map(|&r| BasisStatus::from_discriminant_code(r))
            .collect();

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
fn slot_identity(slot: &EntitySlot) -> (u8, i32, u32) {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// Compare two entity manifests slot-for-slot by `slot_identity`.
///
/// `source` (a loaded policy's manifest) and `current` (the current study's
/// terminal manifest) describe the same-length state vector of two studies. A
/// per-slot `(entity_type, entity_id, subindex)` mismatch means a cut coefficient
/// would attach to the wrong state variable and is REJECTED. `was_active` is
/// excluded from `slot_identity`, so a `source`-dormant slot now active only warns.
/// An empty manifest on either side (a pre-manifest checkpoint) cannot be
/// verified: warn and return `Ok`, leaving the caller's `state_dimension` check
/// standing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `source` and `current` differ in length
/// or in any slot's `(entity_type, entity_id, subindex)`.
pub fn compare_manifest_slot_identity(
    source: &[EntitySlot],
    current: &[EntitySlot],
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
/// Per-slot matching compares the source stage's manifest to the current
/// TERMINAL-stage manifest (both length `state_dimension`); `num_stages` may
/// differ. `current_manifest` is built via
/// [`StudySetup::build_terminal_entity_manifest`](crate::StudySetup::build_terminal_entity_manifest)
/// (single owner of identity resolution, shared with the checkpoint writer). An
/// empty manifest (pre-manifest checkpoint) leaves the `state_dimension` check
/// standing and warns; a `was_active == false` boundary slot whose current
/// counterpart is active warns and loads. Delegates to [`validate_policy_load`]
/// typed to [`BoundaryInjection`] for the per-slot identity reject, and wraps
/// the result in a [`ValidatedBoundaryCuts`] — the sole constructor
/// [`inject_boundary_cuts`] accepts.
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
    boundary_path: &Path,
    source_stage: u32,
    current_state_dimension: u32,
    current_manifest: &[EntitySlot],
    loading_cost_scale_factor: f64,
    on_warning: &mut dyn FnMut(&str),
) -> Result<ValidatedBoundaryCuts, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
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
    let proof = validate_policy_load::<BoundaryInjection>(&source, &current)?;
    for warning in &proof.warnings {
        on_warning(warning);
    }

    let mut records = stage_result.cuts.clone();
    rescale_cut_records_for_load(
        &mut records,
        checkpoint.metadata.cost_scale_factor,
        loading_cost_scale_factor,
    );

    Ok(ValidatedBoundaryCuts { records })
}

/// Boundary cut records that passed [`validate_policy_load`]'s
/// [`BoundaryInjection`] check matrix. The private field means the only way to
/// obtain one is [`load_boundary_cuts`], so [`inject_boundary_cuts`] cannot
/// compile against a bare, unvalidated `Vec<OwnedPolicyCutRecord>`. Derefs to
/// `[OwnedPolicyCutRecord]` for read access.
#[derive(Debug, Clone)]
pub struct ValidatedBoundaryCuts {
    records: Vec<OwnedPolicyCutRecord>,
}

impl Deref for ValidatedBoundaryCuts {
    type Target = [OwnedPolicyCutRecord];

    fn deref(&self) -> &Self::Target {
        &self.records
    }
}

/// Inject boundary cuts into the terminal stage of the study's FCF.
///
/// Replaces the terminal stage's [`CutPool`] with one pre-populated from
/// `boundary_cuts`, retaining capacity for new training cuts. The resulting
/// nonzero `warm_start_count` is what makes the forward pass treat the terminal
/// stage as boundary-loaded (`terminal_has_boundary_cuts`) and skip theta zeroing.
///
/// `boundary_cuts` must come from [`load_boundary_cuts`] — its private field
/// means a bare `Vec<OwnedPolicyCutRecord>`/slice cannot substitute, so an
/// unvalidated boundary load cannot compile.
///
/// ```compile_fail
/// use cobre_sddp::{StudySetup, inject_boundary_cuts};
///
/// fn call_with_bare_records(
///     setup: &mut StudySetup,
///     records: &[cobre_io::OwnedPolicyCutRecord],
/// ) {
///     inject_boundary_cuts(setup, records); // bare records, not ValidatedBoundaryCuts
/// }
/// ```
pub fn inject_boundary_cuts(setup: &mut StudySetup, boundary_cuts: &ValidatedBoundaryCuts) {
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
        boundary_cuts,
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use cobre_io::{EntitySlot, PolicyCheckpointMetadata, StageCutsPayload};

    use super::{
        BoundaryInjection, FullFcf, PolicyStageManifest, compare_manifest_slot_identity,
        load_boundary_cuts, resolve_warm_start_counts, validate_policy_load,
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
            cost_scale_factor: None,
        };

        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// Write a single-stage checkpoint whose one cut has the given at-rest
    /// `intercept`/`coefficients` (written byte-for-byte, no transform applied
    /// here) and `metadata.cost_scale_factor` set to `cost_scale_factor`, for
    /// [`load_boundary_cuts`] round-trip tests across differing loading
    /// factors.
    fn write_checkpoint_with_scale(
        dir: &std::path::Path,
        stage_id: u32,
        intercept: f64,
        coefficients: &[f64],
        cost_scale_factor: Option<f64>,
    ) {
        let state_dimension = coefficients.len() as u32;
        let cut = cobre_io::PolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active: true,
        };
        let cuts = vec![cut];
        let payload = StageCutsPayload {
            stage_id,
            state_dimension,
            capacity: 1,
            warm_start_count: 0,
            cuts: &cuts,
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &[],
        };
        let metadata = PolicyCheckpointMetadata {
            cobre_version: "0.11.0".to_string(),
            created_at: "2026-07-20T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension,
            num_stages: stage_id + 1,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor,
        };
        cobre_io::write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).unwrap();
    }

    /// Behavioral: [`load_boundary_cuts`] on a MARKED checkpoint (§5 Option A,
    /// canonical currency units at rest) loaded at a series of differing
    /// `loading_cost_scale_factor` values recovers `at_rest / loading_factor`
    /// for every value, matching [`rescale_cut_records_for_load`]'s contract at
    /// the file-I/O boundary — not just as a pure-function unit test.
    #[test]
    fn load_boundary_cuts_across_differing_loading_factors() {
        let at_rest_intercept = 1_234_000.0;
        let at_rest_coefficients = [10_000.0, -25_000.0];

        for loading_factor in [500_000.0, 1_000_000.0, 2_500_000.0, 1e10] {
            let tmp = tempfile::tempdir().unwrap();
            write_checkpoint_with_scale(
                tmp.path(),
                0,
                at_rest_intercept,
                &at_rest_coefficients,
                Some(1_000_000.0),
            );

            let cuts = load_boundary_cuts(
                tmp.path(),
                0,
                2,
                &[],
                loading_factor,
                &mut ignore_warnings(),
            )
            .unwrap();

            assert_eq!(cuts.len(), 1);
            let expected_intercept = at_rest_intercept / loading_factor;
            assert!(
                (cuts[0].intercept - expected_intercept).abs()
                    < expected_intercept.abs().max(1.0) * 1e-9,
                "loading_factor={loading_factor}: intercept {} != expected {expected_intercept}",
                cuts[0].intercept
            );
            for (c, &at_rest) in cuts[0].coefficients.iter().zip(&at_rest_coefficients) {
                let expected = at_rest / loading_factor;
                assert!(
                    (c - expected).abs() < expected.abs().max(1.0) * 1e-9,
                    "loading_factor={loading_factor}: coefficient {c} != expected {expected}"
                );
            }
        }
    }

    /// Behavioral: a legacy (no-marker) boundary checkpoint loaded at the
    /// default factor is bit-exact; loaded at a non-default factor is
    /// rescaled by `LEGACY_COST_SCALE_FACTOR / loading_factor`.
    #[test]
    fn load_boundary_cuts_legacy_checkpoint_migration() {
        let raw_intercept = 5.0;
        let raw_coefficients = [1.0, 2.0];

        let tmp_default = tempfile::tempdir().unwrap();
        write_checkpoint_with_scale(
            tmp_default.path(),
            0,
            raw_intercept,
            &raw_coefficients,
            None,
        );
        let cuts_default = load_boundary_cuts(
            tmp_default.path(),
            0,
            2,
            &[],
            LEGACY_COST_SCALE_FACTOR,
            &mut ignore_warnings(),
        )
        .unwrap();
        assert_eq!(
            cuts_default[0].intercept.to_bits(),
            raw_intercept.to_bits(),
            "legacy checkpoint at default factor must be bit-exact"
        );

        let tmp_nondefault = tempfile::tempdir().unwrap();
        write_checkpoint_with_scale(
            tmp_nondefault.path(),
            0,
            raw_intercept,
            &raw_coefficients,
            None,
        );
        let loading_factor = 2_000_000.0;
        let cuts_nondefault = load_boundary_cuts(
            tmp_nondefault.path(),
            0,
            2,
            &[],
            loading_factor,
            &mut ignore_warnings(),
        )
        .unwrap();
        let ratio = LEGACY_COST_SCALE_FACTOR / loading_factor;
        assert!((cuts_nondefault[0].intercept - raw_intercept * ratio).abs() < 1e-9);
    }

    // ── §5 Option A: rescale_cut_records_for_load unit tests ─────────────────

    use super::{LEGACY_COST_SCALE_FACTOR, rescale_cut_records_for_load};
    use crate::policy_export::scale_cut_records_for_export;
    use cobre_io::OwnedPolicyCutRecord;

    fn owned_cut(intercept: f64, coefficients: Vec<f64>) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active: true,
        }
    }

    /// AC: a legacy checkpoint (`source_cost_scale_factor: None`) loaded at the
    /// still-default [`LEGACY_COST_SCALE_FACTOR`] is a bit-exact no-op — the
    /// correctness requirement backing "no re-baseline except the §5-A
    /// policy-load-hashed set".
    #[test]
    fn legacy_no_marker_at_default_factor_is_bit_exact_noop() {
        let mut records = vec![owned_cut(42.5, vec![1.0, -2.5, 3.75])];
        let original = records.clone();

        rescale_cut_records_for_load(&mut records, None, LEGACY_COST_SCALE_FACTOR);

        assert_eq!(
            records[0].intercept.to_bits(),
            original[0].intercept.to_bits()
        );
        for (a, b) in records[0]
            .coefficients
            .iter()
            .zip(&original[0].coefficients)
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "legacy default-factor load must be bit-exact"
            );
        }
    }

    /// AC: a legacy checkpoint loaded at a NON-default factor is interpreted as
    /// scaled-at-[`LEGACY_COST_SCALE_FACTOR`] and rescaled by
    /// `LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor`.
    #[test]
    fn legacy_no_marker_at_non_default_factor_rescales_by_ratio() {
        let mut records = vec![owned_cut(10.0, vec![2.0, 4.0])];
        let loading_factor = 2_000_000.0;

        rescale_cut_records_for_load(&mut records, None, loading_factor);

        let ratio = LEGACY_COST_SCALE_FACTOR / loading_factor;
        assert!((records[0].intercept - 10.0 * ratio).abs() < 1e-9);
        assert!((records[0].coefficients[0] - 2.0 * ratio).abs() < 1e-9);
        assert!((records[0].coefficients[1] - 4.0 * ratio).abs() < 1e-9);
    }

    /// AC: a marked checkpoint (`Some(s)`) is ALWAYS divided by
    /// `loading_cost_scale_factor` — even when `s` equals the loading factor —
    /// never special-cased to a no-op. The `source_cost_scale_factor` VALUE is
    /// irrelevant once the file holds canonical currency units; only its
    /// presence (marked vs. legacy) selects the code path.
    #[test]
    fn marked_checkpoint_always_divides_regardless_of_source_value() {
        let loading_factor = 1_000_000.0;
        let mut with_matching_source = vec![owned_cut(100.0, vec![50.0])];
        let mut with_different_source = vec![owned_cut(100.0, vec![50.0])];

        rescale_cut_records_for_load(
            &mut with_matching_source,
            Some(loading_factor),
            loading_factor,
        );
        rescale_cut_records_for_load(&mut with_different_source, Some(42.0), loading_factor);

        assert_eq!(
            with_matching_source[0].intercept.to_bits(),
            with_different_source[0].intercept.to_bits(),
            "the source factor's VALUE must not affect the loaded result"
        );
        assert!((with_matching_source[0].intercept - 100.0 / loading_factor).abs() < 1e-12);
    }

    /// §5 Option A transform property: export (multiply by `S`) then load at
    /// the SAME factor (divide by `S`) recovers the original value within 1
    /// ULP per value — the accepted same-factor round-trip drift (1e6 is not a
    /// power of two, so two roundings do not cancel exactly).
    #[test]
    fn export_then_load_same_factor_round_trips_within_one_ulp() {
        let cost_scale_factor = 1_000_000.0;
        let originals = [vec![1.0_f64, -3.5, 1e-6, 123_456.789]];
        let intercepts = [7.25_f64];

        let internal_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = vec![
            originals
                .iter()
                .zip(&intercepts)
                .map(|(coeffs, &intercept)| cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept,
                    coefficients: coeffs,
                    is_active: true,
                })
                .collect(),
        ];

        let exported = scale_cut_records_for_export(&internal_records, cost_scale_factor);
        let mut round_tripped = exported[0].clone();
        rescale_cut_records_for_load(
            &mut round_tripped,
            Some(cost_scale_factor),
            cost_scale_factor,
        );

        let original_intercept = intercepts[0];
        let ulp_intercept = (round_tripped[0].intercept - original_intercept).abs();
        assert!(
            ulp_intercept <= original_intercept.abs() * f64::EPSILON * 4.0,
            "intercept round-trip drift {ulp_intercept} exceeds a few ULP of {original_intercept}"
        );
        for (rt, orig) in round_tripped[0].coefficients.iter().zip(&originals[0]) {
            let drift = (rt - orig).abs();
            let tol = (orig.abs().max(1.0)) * f64::EPSILON * 4.0;
            assert!(
                drift <= tol,
                "coefficient round-trip drift {drift} exceeds tolerance {tol} for original {orig}"
            );
        }
    }

    /// §5 Option A transform property: cross-factor linearity — exporting at
    /// `S_train` then loading at `S_prime` recovers `original * (S_train /
    /// S_prime)` (the net two-rounding transform), for `S_prime != S_train`.
    #[test]
    fn export_then_load_cross_factor_is_linear() {
        let s_train = 1_000_000.0;
        let s_prime = 4_000_000.0;
        let original = [vec![2.0_f64, -0.5]];
        let intercept = 9.0_f64;

        let internal_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> =
            vec![vec![cobre_io::PolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                iteration: 0,
                forward_pass_index: 0,
                intercept,
                coefficients: &original[0],
                is_active: true,
            }]];

        let exported = scale_cut_records_for_export(&internal_records, s_train);
        let mut loaded = exported[0].clone();
        rescale_cut_records_for_load(&mut loaded, Some(s_train), s_prime);

        let ratio = s_train / s_prime;
        assert!((loaded[0].intercept - intercept * ratio).abs() < 1e-9);
        for (l, o) in loaded[0].coefficients.iter().zip(&original[0]) {
            assert!((l - o * ratio).abs() < 1e-9);
        }
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

        let cuts = load_boundary_cuts(tmp.path(), 2, 10, &[], 1_000_000.0, &mut ignore_warnings())
            .unwrap();

        assert_eq!(cuts.len(), 3, "should return all 3 cuts from stage 2");
        let returned_intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(
            returned_intercepts, intercepts,
            "intercepts should match written values"
        );
        for cut in cuts.iter() {
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

        let result =
            load_boundary_cuts(tmp.path(), 99, 10, &[], 1_000_000.0, &mut ignore_warnings());

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

        let result = load_boundary_cuts(tmp.path(), 0, 5, &[], 1_000_000.0, &mut ignore_warnings());

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
            1_000_000.0,
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
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, 1_000_000.0, &mut |m| {
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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, 1_000_000.0, &mut |m| {
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
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, 1_000_000.0, &mut |m| {
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
        let cuts = load_boundary_cuts(tmp.path(), 0, 2, &current, 1_000_000.0, &mut |m| {
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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            3,
            &current,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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

        let report = validate_policy_load::<FullFcf>(&source, &current).unwrap();

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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let full_fcf_result = validate_policy_load::<FullFcf>(&source, &current);
        assert!(
            full_fcf_result.is_err(),
            "num_stages mismatch must reject FullFcf"
        );
        let msg = full_fcf_result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include source value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");

        let boundary_result = validate_policy_load::<BoundaryInjection>(&source, &current);
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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let result = validate_policy_load::<BoundaryInjection>(&source, &current);

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

        let report = validate_policy_load::<FullFcf>(&source, &current).unwrap();

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
            cost_scale_factor: None,
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
    /// and `Fixed`, which the old `to_highs_code` path folded onto `Nonbasic`/
    /// `Lower`. Encode via `convert_basis_cache`, round-trip through the codec,
    /// decode via `build_basis_cache_from_checkpoint`.
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
            column_status: &col_u8[0],
            row_status: &row_u8[0],
            num_cut_rows: 0,
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

    /// Pre-existing-file compatibility: a checkpoint written by the pre-canonical
    /// writer stored `HiGHS` codes (bytes `0..=4`) directly in `column_status`/
    /// `row_status`. Those same bytes must load to exactly the statuses the old
    /// `HiGHS`-space decode produced, since `0..=4` means the same in the canonical
    /// and `HiGHS` code spaces.
    #[test]
    fn pre_existing_checkpoint_bytes_load_to_highs_space_statuses() {
        use cobre_io::{PolicyBasisRecord, deserialize_stage_basis, serialize_stage_basis};
        use cobre_solver::BasisStatus;

        use super::build_basis_cache_from_checkpoint;

        // HiGHS codes 0..=4, exactly what the pre-canonical writer stored on disk.
        let col_bytes: [u8; 5] = [0, 1, 2, 3, 4];
        let row_bytes: [u8; 5] = [4, 3, 2, 1, 0];

        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &col_bytes,
            row_status: &row_bytes,
            num_cut_rows: 0,
        };
        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("codec round-trip must succeed");

        let cache = build_basis_cache_from_checkpoint(1, std::slice::from_ref(&owned), &[]);
        let recovered = cache[0].as_ref().expect("stage 0 basis must be present");

        let expected_col: Vec<BasisStatus> = col_bytes
            .iter()
            .map(|&c| BasisStatus::from_highs_code(i32::from(c)))
            .collect();
        let expected_row: Vec<BasisStatus> = row_bytes
            .iter()
            .map(|&c| BasisStatus::from_highs_code(i32::from(c)))
            .collect();
        assert_eq!(
            recovered.basis.col_status, expected_col,
            "old-file bytes must load to the old HiGHS-space column statuses"
        );
        assert_eq!(
            recovered.basis.row_status, expected_row,
            "old-file bytes must load to the old HiGHS-space row statuses"
        );
    }
}
