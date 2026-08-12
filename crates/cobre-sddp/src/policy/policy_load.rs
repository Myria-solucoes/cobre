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

use chrono::NaiveDate;
use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
use cobre_io::EntitySlot;
use cobre_io::GraphManifest;
use cobre_io::OwnedPolicyBasisRecord;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyCheckpointMetadata;
use cobre_io::StageCutsReadResult;
use cobre_io::read_policy_checkpoint;
use cobre_solver::{Basis, BasisStatus};

use crate::SddpError;
use crate::cut::pool::CutPool;
use crate::policy::policy_export::{
    ENTITY_TYPE_ANTICIPATED_THERMAL_STATE, ENTITY_TYPE_HYDRO_INFLOW_LAG,
};
use crate::policy::reconcile::{
    BoundaryReconciliationReport, RebindOp, build_rebind, build_reconciliation_report,
    decode_month_anchor, overlap_hours, rebind_cut,
};
use crate::setup::{NodeId, NodePos, StudySetup, TypedVec};
use crate::workspace::CapturedBasis;

use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;

/// Resolve the per-POOL warm-start cut counts from a loaded policy checkpoint.
///
/// Returns a `Vec<u32>` of length `n_pools` for [`FutureCostFunction::new`].
/// An empty `metadata.warm_start_counts` (old checkpoint format) broadcasts the
/// scalar `warm_start_cuts` to every pool.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `warm_start_counts.len() != n_pools`.
/// `warm_start_counts` is per-pool, not per-stage — validate against the
/// pool count, which differs from the stage count on a branching graph.
///
/// [`FutureCostFunction::new`]: crate::FutureCostFunction::new
// Rationale: kept as the validated entry point for the planned checkpoint-migration
// tool (and exercised by this module's tests); the active path consumes
// `metadata.warm_start_counts` directly.
#[allow(dead_code)]
pub(crate) fn resolve_warm_start_counts(
    metadata: &PolicyCheckpointMetadata,
    n_pools: usize,
) -> Result<Vec<u32>, SddpError> {
    if metadata.producer.warm_start_counts.is_empty() {
        Ok(vec![metadata.producer.warm_start_cuts; n_pools])
    } else if metadata.producer.warm_start_counts.len() != n_pools {
        Err(SddpError::Validation(format!(
            "warm_start_counts length mismatch: checkpoint has {}, current system has {} pools",
            metadata.producer.warm_start_counts.len(),
            n_pools,
        )))
    } else {
        Ok(metadata.producer.warm_start_counts.clone())
    }
}

/// The constant every unmarked policy checkpoint (no `cost_scale_factor`
/// provenance) was unconditionally scaled at. A checkpoint whose
/// `metadata.cost_scale_factor` is `None` is interpreted under this constant.
pub const LEGACY_COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Rescale one stage's cut records from their at-rest
/// representation into the LOADING study's internal scaled cost space.
///
/// - **Marked** (`source_cost_scale_factor: Some(s)`): the checkpoint holds
///   canonical currency units — export multiplied every value by the writing
///   study's own `s` ([`crate::policy_export::scale_cut_records_for_export`]).
///   Every value here is divided by `loading_cost_scale_factor`, UNCONDITIONALLY
///   — even when `s` equals `loading_cost_scale_factor` — since the file
///   already carries one export-side rounding; a second division is the
///   accepted same-factor ULP drift, never special-cased
///   away.
/// - **Legacy** (`None`): the checkpoint holds the writing study's OWN internal
///   scaled values under [`LEGACY_COST_SCALE_FACTOR`] — legacy files carry
///   no export-side multiply. When `loading_cost_scale_factor ==
///   LEGACY_COST_SCALE_FACTOR` (the overwhelmingly common case: every existing
///   policy directory read at the still-default factor) this is an exact,
///   bit-identical no-op — a correctness requirement, not an optimization: a
///   legacy checkpoint at the default factor must load bit-identically, never
///   re-baselined. Otherwise every value is
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

/// `rescale_cut_records_for_load` applied to every stage of a full policy
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
    /// Number of storage pools (the pool-set size) — the pool-count analogue of
    /// `num_stages`, checked only under [`FullFcf`].
    pub n_pools: u32,
    /// Per-slot entity identity, in state-vector order.
    pub slots: &'a [EntitySlot],
    /// Graph manifest (node list, edges, node → pool map). Checked for identity
    /// only under [`FullFcf`]; a [`BoundaryInjection`] load ignores it.
    pub graph: &'a GraphManifest,
}

mod sealed {
    pub trait Sealed {}
}

/// Selects [`validate_policy_load`]'s check matrix (`state_dimension` is
/// checked unconditionally for every kind). Sealed so [`FullFcf`] and
/// [`BoundaryInjection`] are the only implementors, making a
/// [`PolicyLoadProof<K>`] a proof of validation under exactly one real kind —
/// never a third, uncatalogued one.
pub trait PolicyLoadKind: sealed::Sealed {
    /// Whether `num_stages` equality is hard-rejected for this kind.
    const CHECK_NUM_STAGES: bool;
    /// Whether the pool count and graph-manifest identity are hard-rejected for
    /// this kind. The pool-count analogue of [`Self::CHECK_NUM_STAGES`].
    const CHECK_N_POOLS: bool;
    /// Whether per-slot identity is checked here as an EXACT positional match
    /// ([`compare_manifest_slot_identity`]). `false` means the kind reconciles
    /// slot identity by its own mechanism instead, confined to its own load
    /// path (currently only [`BoundaryInjection`], via
    /// [`crate::policy::reconcile::build_rebind`] in [`load_boundary_cuts`]).
    const CHECK_SLOT_IDENTITY_EXACT: bool;
}

/// Full future-cost-function load (warm-start, resume, simulation-only):
/// `num_stages`, the pool count, the graph manifest, and per-slot identity
/// must match `current` exactly.
#[derive(Debug, Clone, Copy)]
pub struct FullFcf;

impl sealed::Sealed for FullFcf {}
impl PolicyLoadKind for FullFcf {
    const CHECK_NUM_STAGES: bool = true;
    const CHECK_N_POOLS: bool = true;
    const CHECK_SLOT_IDENTITY_EXACT: bool = true;
}

/// Single-stage boundary-cut injection into the terminal pool: `num_stages`,
/// the pool count, and the graph manifest are unchecked (a monthly source may
/// feed a weekly+monthly current study on a different graph); per-slot
/// identity is RECONCILED, not exact-matched — [`load_boundary_cuts`] wires
/// [`crate::policy::reconcile::build_rebind`]/`rebind_cut` in after this
/// validation succeeds.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryInjection;

impl sealed::Sealed for BoundaryInjection {}
impl PolicyLoadKind for BoundaryInjection {
    const CHECK_NUM_STAGES: bool = false;
    const CHECK_N_POOLS: bool = false;
    const CHECK_SLOT_IDENTITY_EXACT: bool = false;
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
/// `K`'s check matrix: `state_dimension` equality is hard-rejected for every
/// kind; `num_stages` equality is hard-rejected only when
/// `K::CHECK_NUM_STAGES` ([`FullFcf`]). Per-slot identity is an EXACT
/// positional match (delegated to [`compare_manifest_slot_identity`]) only
/// when `K::CHECK_SLOT_IDENTITY_EXACT` ([`FullFcf`]); [`BoundaryInjection`]
/// skips it here — its slot identity is RECONCILED separately, by
/// [`crate::policy::reconcile::build_rebind`] in [`load_boundary_cuts`], never
/// by this lower-level manifest check. `col_scale`/scaling is never a
/// compatibility dimension. This is the single entry point for policy-load
/// validation — its success is the only way to construct a
/// [`PolicyLoadProof<K>`], so every load path routes through it.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a `state_dimension` mismatch, a
/// `num_stages` mismatch under [`FullFcf`], or (under [`FullFcf`] only) a
/// per-slot identity mismatch (see [`compare_manifest_slot_identity`]).
pub fn validate_policy_load<K: PolicyLoadKind>(
    source: &PolicyStageManifest<'_>,
    current: &PolicyStageManifest<'_>,
) -> Result<PolicyLoadProof<K>, SddpError> {
    if source.state_dimension != current.state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {} (a lag-state \
             depth mismatch, e.g. state_space.inflow_lag_depth, is a common cause)",
            source.state_dimension, current.state_dimension
        )));
    }

    if K::CHECK_NUM_STAGES && source.num_stages != current.num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            source.num_stages, current.num_stages
        )));
    }

    if K::CHECK_N_POOLS && source.n_pools != current.n_pools {
        return Err(SddpError::Validation(format!(
            "policy n_pools mismatch: policy has {}, current system has {}",
            source.n_pools, current.n_pools
        )));
    }

    let mut warnings = Vec::new();
    if K::CHECK_SLOT_IDENTITY_EXACT {
        compare_manifest_slot_identity(source.slots, current.slots, &mut |msg| {
            warnings.push(msg.to_string());
        })?;
    }

    if K::CHECK_N_POOLS {
        compare_graph_manifest_identity(source.graph, current.graph)?;
    }

    Ok(PolicyLoadProof {
        warnings,
        _kind: PhantomData,
    })
}

/// Compare two graph manifests for structural identity: pool-set size, per-node
/// `(id, stage_id, pool_id)`, and per-edge `(source_id, target_id)`.
///
/// A [`FullFcf`] resume/warm-start continues the SAME node topology, so a
/// divergence means the loaded value function attaches to a different graph and
/// is REJECTED. An empty manifest on either side (a graph-less artifact — e.g.
/// one authored from raw records) cannot be verified: silently skip, leaving the
/// `state_dimension`/`num_stages`/`n_pools` checks standing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a pool-count, node, or edge divergence.
pub fn compare_graph_manifest_identity(
    source: &GraphManifest,
    current: &GraphManifest,
) -> Result<(), SddpError> {
    if source.nodes.is_empty() || current.nodes.is_empty() {
        return Ok(());
    }

    if source.n_pools != current.n_pools {
        return Err(SddpError::Validation(format!(
            "graph manifest n_pools mismatch: source has {}, current study has {}",
            source.n_pools, current.n_pools
        )));
    }

    if source.nodes.len() != current.nodes.len() {
        return Err(SddpError::Validation(format!(
            "graph manifest node-count mismatch: source has {} nodes, current study has {}",
            source.nodes.len(),
            current.nodes.len()
        )));
    }

    for (i, (src, cur)) in source.nodes.iter().zip(&current.nodes).enumerate() {
        if (src.id, src.stage_id, src.pool_id) != (cur.id, cur.stage_id, cur.pool_id) {
            return Err(SddpError::Validation(format!(
                "graph manifest node {i} mismatch: source (id={}, stage_id={}, pool_id={}) != \
                 current (id={}, stage_id={}, pool_id={})",
                src.id, src.stage_id, src.pool_id, cur.id, cur.stage_id, cur.pool_id
            )));
        }
    }

    if source.edges.len() != current.edges.len() {
        return Err(SddpError::Validation(format!(
            "graph manifest edge-count mismatch: source has {} edges, current study has {}",
            source.edges.len(),
            current.edges.len()
        )));
    }

    for (i, (src, cur)) in source.edges.iter().zip(&current.edges).enumerate() {
        if (src.source_id, src.target_id) != (cur.source_id, cur.target_id) {
            return Err(SddpError::Validation(format!(
                "graph manifest edge {i} mismatch: source ({} -> {}) != current ({} -> {})",
                src.source_id, src.target_id, cur.source_id, cur.target_id
            )));
        }
    }

    Ok(())
}

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<CapturedBasis>>` of length `n_nodes` (`== node_ids.len()`),
/// one entry per canonical node position; nodes without a matching record get
/// `None`. Each basis record is keyed by its own node ordinal (its `stage_id`),
/// so leaves sharing a pool land in distinct node slots — no `>= num_stages`
/// drop, no cross-node collision, both of which the pre-branching per-stage
/// sizing produced once `n_nodes > n_stages`. `u8` status codes decode via
/// `from_discriminant_code`, the mirror of `convert_basis_cache`'s export-side
/// `to_discriminant_code`; a pre-existing checkpoint (bytes `0..=4`) decodes
/// identically, since that range means the same in the canonical and `HiGHS`
/// code spaces.
///
/// # Cut-slot reconstruction
///
/// `row_status` is `[template rows…, cut rows…]`, the trailing `num_cut_rows` in
/// capture-time [`CutPool::active_cuts`](crate::cut::pool::CutPool::active_cuts)
/// order (active slots, increasing). A node's cut records live in its OWN pool's
/// [`StageCutsReadResult`] (`sc.stage_id == node_pools[node]`), never the record
/// whose pool id happens to equal the node ordinal — the two diverge once
/// `n_pools != n_nodes` on a branching graph. Slot identity is recovered from
/// that pool's active records' `slot_index` in increasing order, so
/// `reconstruct_basis` preserves stored cut-row statuses across cut-set churn.
///
/// # Graceful fallback
///
/// When the derived active-slot count ≠ `num_cut_rows` (cut selection deactivated
/// cuts between capture and export) or no cut record matches, fall back to safe
/// all-template behavior (empty `cut_row_slots`; every cut row reconstructs
/// BASIC). This changes only the warm-start solve path, never the optimum.
///
/// `node_ids` / `node_pools` are the CURRENT study's `NodeGraph::node_ids` and
/// `NodeGraph::node_pool_ids` (both length `n_nodes`): a resume/warm-start
/// continues the SAME node topology, so each reconstructed basis's `node_id` is
/// `node_ids[node]` and its owning pool is `node_pools[node]` — never a value
/// recovered from the checkpoint itself (the checkpoint wire carries no node id).
#[must_use]
pub fn build_basis_cache_from_checkpoint(
    stage_bases: &[OwnedPolicyBasisRecord],
    stage_cuts: &[StageCutsReadResult],
    node_ids: &TypedVec<NodePos, NodeId>,
    node_pools: &TypedVec<NodePos, usize>,
) -> Vec<Option<CapturedBasis>> {
    let n_nodes = node_ids.len();
    let mut cache: Vec<Option<CapturedBasis>> = vec![None; n_nodes];
    for record in stage_bases {
        // Wire boundary: the checkpoint's `stage_id` field is a legacy name for
        // what the node-native engine writes/reads as a node position — convert
        // to `NodePos` immediately, never carry the raw wire int past this line.
        let node = NodePos(record.stage_id as usize);
        if node.0 >= n_nodes {
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
        let pool = node_pools[node];
        let active_slots: Option<Vec<u32>> = stage_cuts
            .iter()
            .find(|sc| sc.stage_id as usize == pool)
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

        cache[node.0] = Some(CapturedBasis {
            basis: Basis {
                col_status,
                row_status,
            },
            base_row_count,
            cut_row_slots,
            state_at_capture: Vec::new(),
            node_id: node_ids[node],
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

/// Whether `source`/`current` can be identity-verified at all: an empty
/// manifest on either side (a pre-manifest checkpoint) cannot be verified —
/// warn once and answer `false`, leaving the caller's `state_dimension` check
/// as the sole compatibility guard. Shared by [`compare_manifest_slot_identity`]
/// (the [`FullFcf`] exact-match path) and [`load_boundary_cuts`]'s reconcile
/// path, so the two report the identical warning on an absent manifest.
fn manifest_identity_verifiable(
    source: &[EntitySlot],
    current: &[EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) -> bool {
    if source.is_empty() || current.is_empty() {
        on_warning(&format!(
            "entity manifest absent (source slots: {}, current slots: {}); slot identity \
             could not be verified, relying on state_dimension alone",
            source.len(),
            current.len(),
        ));
        return false;
    }
    true
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
    if !manifest_identity_verifiable(source, current, on_warning) {
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

/// The deepest inflow-lag slot a manifest carries a cut coefficient on — the
/// 1-based `HydroInflowLag` subindex (`policy_export::build_stage_entity_manifest`
/// emits `lag + 1`), `0` when the manifest carries no lag slot.
fn boundary_cut_lag_depth(manifest: &[EntitySlot]) -> u32 {
    manifest
        .iter()
        .filter(|slot| slot.entity_type == ENTITY_TYPE_HYDRO_INFLOW_LAG)
        .map(|slot| slot.subindex)
        .max()
        .unwrap_or(0)
}

/// Load boundary cuts from the `source_stage` of a source Cobre policy checkpoint.
///
/// Compares the source stage's manifest against the current TERMINAL-stage
/// manifest (`current_manifest`, built via
/// [`StudySetup::build_terminal_entity_manifest`](crate::StudySetup::build_terminal_entity_manifest));
/// `num_stages` may differ. `state_dimension`/`num_stages` compatibility
/// routes through [`validate_policy_load`] typed to [`BoundaryInjection`];
/// per-slot identity is then RECONCILED, never exact-matched, via
/// [`crate::policy::reconcile::build_rebind`]/`rebind_cut` — storage and
/// inflow-lag reject a target slot with no source counterpart, naming the
/// offending hydro or lag depth (the entity is never relaxed); a live, dated
/// anticipated target slot fans out against the source's own priced
/// anticipated months by calendar overlap
/// (`target_delivery_intervals`, aligned 1:1 with `current_manifest` — built
/// via
/// [`StudySetup::build_terminal_anticipated_delivery_intervals`](crate::StudySetup::build_terminal_anticipated_delivery_intervals)).
/// An empty manifest on either side skips reconciliation and warns, relying
/// on `state_dimension` alone; a `was_active == false` boundary slot whose
/// current counterpart is active warns and loads. Wraps the result in a
/// [`ValidatedBoundaryCuts`] — the sole constructor [`inject_boundary_cuts`]
/// accepts — carrying a [`crate::policy::reconcile::build_reconciliation_report`]
/// tally on the reconcile path, or the `reconciled: false` default on the
/// skipped path; read via [`ValidatedBoundaryCuts::report`].
///
/// `declared_inflow_lag_depth` is `config.state_space.inflow_lag_depth`; when
/// `Some`, a boundary cut referencing inflow-lag state deeper than the declared
/// depth is rejected before the manifest checks, so the lag-depth-specific
/// message wins over the generic `state_dimension` reject.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read
/// - `source_stage` does not exist in the checkpoint
/// - The boundary cut references inflow-lag state deeper than a declared
///   `inflow_lag_depth`
/// - The source stage's state dimension does not match `current_state_dimension`
/// - A target storage or inflow-lag slot has no source counterpart under
///   identity reconciliation
///
/// A dated anticipated target slot with no resolved `target_delivery_intervals`
/// entry is NOT an error: it is an in-study ring slot (a within-horizon
/// delivery) the boundary does not price, resolved to `Zero` (see
/// `reconcile::resolve_anticipated`).
pub fn load_boundary_cuts(
    boundary_path: &Path,
    source_stage: u32,
    current_state_dimension: u32,
    current_manifest: &[EntitySlot],
    target_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
    declared_inflow_lag_depth: Option<u32>,
    loading_cost_scale_factor: f64,
    on_warning: &mut dyn FnMut(&str),
) -> Result<ValidatedBoundaryCuts, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;

    // Resolve `source_stage` to its pool THROUGH the graph manifest (a node
    // references one pool), rejecting a multi-node stage: boundary injection
    // requires a single-node source and the frozen `policy.boundary` config
    // offers no node selector. A graph-less artifact (empty manifest) falls back
    // to keying by `source_stage` directly — the chain identity where
    // stage == pool.
    let manifest = &checkpoint.metadata.graph_manifest;
    let resolved_pool: Option<u32> = if manifest.nodes.is_empty() {
        Some(source_stage)
    } else {
        let source_stage_i32 = i32::try_from(source_stage).map_err(|_| {
            SddpError::Validation(format!(
                "boundary policy: source_stage {source_stage} overflows the stage id space"
            ))
        })?;
        let mut at_stage = manifest
            .nodes
            .iter()
            .filter(|n| n.stage_id == source_stage_i32);
        let first = at_stage.next();
        if at_stage.next().is_some() {
            return Err(SddpError::Validation(format!(
                "boundary policy: source_stage {source_stage} names a multi-node stage; boundary \
                 injection requires a single-node source"
            )));
        }
        first.map(|n| n.pool_id)
    };

    let stage_result = resolved_pool
        .and_then(|pool| checkpoint.stage_cuts.iter().find(|sr| sr.stage_id == pool))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "boundary policy: source_stage {} not found in checkpoint \
                 (available pools: {:?})",
                source_stage,
                checkpoint
                    .stage_cuts
                    .iter()
                    .map(|sr| sr.stage_id)
                    .collect::<Vec<_>>()
            ))
        })?;

    if let Some(declared) = declared_inflow_lag_depth {
        let depth = boundary_cut_lag_depth(&stage_result.entity_manifest);
        if depth > declared {
            return Err(SddpError::Validation(format!(
                "lag-state depth too shallow for boundary policy stage {source_stage}: the loaded \
                 cuts reference inflow-lag state to depth {depth}, exceeding the declared \
                 state_space.inflow_lag_depth = {declared}; inflow_lag_depth must cover the deepest \
                 lag any loaded cut references so the lag state holds the conditioning history the \
                 recombination claim depends on — raise state_space.inflow_lag_depth to at least \
                 {depth}"
            )));
        }
    }

    // `BoundaryInjection` checks neither `n_pools` nor the graph manifest, so
    // these unchecked fields carry placeholders.
    let empty_graph = GraphManifest::default();
    let source = PolicyStageManifest {
        state_dimension: stage_result.state_dimension,
        num_stages: checkpoint.metadata.num_stages,
        n_pools: 0,
        slots: &stage_result.entity_manifest,
        graph: &empty_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: current_state_dimension,
        num_stages: checkpoint.metadata.num_stages,
        n_pools: 0,
        slots: current_manifest,
        graph: &empty_graph,
    };
    let proof = validate_policy_load::<BoundaryInjection>(&source, &current)?;
    for warning in &proof.warnings {
        on_warning(warning);
    }

    let mut records = stage_result.cuts.clone();
    rescale_cut_records_for_load(
        &mut records,
        checkpoint.metadata.producer.cost_scale_factor,
        loading_cost_scale_factor,
    );

    let report = if manifest_identity_verifiable(
        &stage_result.entity_manifest,
        current_manifest,
        on_warning,
    ) {
        let rebind = build_rebind(
            &stage_result.entity_manifest,
            current_manifest,
            target_delivery_intervals,
        )?;
        warn_on_dormant_source_now_active(
            &rebind,
            &stage_result.entity_manifest,
            current_manifest,
            on_warning,
        );
        for record in &mut records {
            record.coefficients = rebind_cut(record, &rebind);
        }
        build_reconciliation_report(
            &stage_result.entity_manifest,
            current_manifest,
            target_delivery_intervals,
            &rebind,
        )
    } else {
        BoundaryReconciliationReport::default()
    };

    Ok(ValidatedBoundaryCuts { records, report })
}

/// Auto-resolve an absent `policy.boundary.source_stage`: decode each source
/// pool's live anticipated `delivery_date` months
/// ([`decode_month_anchor`]) and pick the pool whose months overlap
/// `current_terminal_delivery_intervals` — the same `target_delivery_intervals`
/// axis [`load_boundary_cuts`] fans coefficients onto (built by
/// [`StudySetup::build_terminal_anticipated_delivery_intervals`](crate::StudySetup::build_terminal_anticipated_delivery_intervals)).
/// The winning candidate is a POOL id (`checkpoint.stage_cuts`'s own
/// `stage_id` field names a pool, not a graph stage — see
/// [`cobre_io::StageCutsPayload`]'s doc); [`load_boundary_cuts`]'s
/// `source_stage` parameter speaks graph-stage-id, so this resolver maps the
/// winning pool back to its owning node's `stage_id` via
/// `graph_manifest.nodes` before returning, through [`resolve_stage_id_for_pool`]
/// (an empty manifest — the chain-degenerate artifact — has `pool_id ==
/// stage`, so the mapping is the identity there). Returning the pool id
/// unmapped is the reproduced defect this guards: threaded back into
/// `load_boundary_cuts` as `source_stage` on a branching graph, it can match
/// an unrelated node whose OWN `stage_id` happens to equal the winning pool's
/// numeric value, silently resolving through that node to a different pool.
///
/// This only PICKS a candidate; it is not a new trust boundary — a
/// calendar-matched pool from an incompatible source still rejects once
/// [`load_boundary_cuts`] reconciles it by storage/lag identity.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read.
/// - More than one source pool's decoded months overlap the current terminal
///   window (ambiguous; names the count and advises an explicit
///   `source_stage`).
/// - No source pool carries any decodable live anticipated `delivery_date`
///   (a pre-dated-slot checkpoint, or a pure storage/lag boundary; advises
///   re-exporting or an explicit `source_stage`).
/// - Every source pool that does carry decodable months has none overlapping
///   the current terminal window (advises an explicit `source_stage`).
/// - A live anticipated source slot's `delivery_date` fails to decode to a
///   real calendar month.
/// - The winning pool is owned by more than one graph node (sibling fan nodes
///   sharing a pool — genuinely ambiguous, mirroring `load_boundary_cuts`'s
///   own single-node-source restriction) or by none (an internal
///   inconsistency: the pool came from `checkpoint.stage_cuts` itself).
pub fn resolve_boundary_source_stage(
    boundary_path: &Path,
    current_terminal_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
) -> Result<u32, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;

    let target_intervals: Vec<(NaiveDate, NaiveDate)> = current_terminal_delivery_intervals
        .iter()
        .copied()
        .flatten()
        .collect();

    let mut any_decodable = false;
    let mut candidates: Vec<u32> = Vec::new();
    for stage in &checkpoint.stage_cuts {
        let months = decode_pool_anticipated_months(&stage.entity_manifest)?;
        if months.is_empty() {
            continue;
        }
        any_decodable = true;
        let overlaps = months.iter().any(|&month| {
            target_intervals
                .iter()
                .any(|&iv| overlap_hours(month, iv) > 0.0)
        });
        if overlaps {
            candidates.push(stage.stage_id);
        }
    }

    if candidates.len() > 1 {
        return Err(SddpError::Validation(format!(
            "ambiguous: {} source pools match the terminal date; set \
             policy.boundary.source_stage explicitly",
            candidates.len()
        )));
    }
    if let Some(&pool) = candidates.first() {
        return resolve_stage_id_for_pool(&checkpoint.metadata.graph_manifest, pool);
    }
    if any_decodable {
        return Err(SddpError::Validation(
            "no source pool's anticipated delivery calendar aligns with the current study's \
             terminal delivery window; set policy.boundary.source_stage explicitly"
                .to_string(),
        ));
    }
    Err(SddpError::Validation(
        "this boundary predates dated slots; re-export it, or set policy.boundary.source_stage \
         explicitly"
            .to_string(),
    ))
}

/// Map a chosen source POOL id back to [`load_boundary_cuts`]'s `source_stage`
/// vocabulary: the `stage_id` of the graph node that owns it, resolved via
/// `manifest.nodes` — the exact inverse of `load_boundary_cuts`'s own
/// `source_stage -> pool` lookup. An empty `manifest` (the chain-degenerate
/// artifact, no `nodes[]`) has `pool_id == stage` by construction, so `pool`
/// is returned unchanged.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if more than one node owns `pool`
/// (sibling fan nodes sharing a pool — genuinely ambiguous, mirroring
/// `load_boundary_cuts`'s own single-node-source restriction) or if no node
/// owns it (an internal inconsistency: `pool` was drawn from
/// `checkpoint.stage_cuts`, so the manifest is missing a pool it wrote).
fn resolve_stage_id_for_pool(manifest: &GraphManifest, pool: u32) -> Result<u32, SddpError> {
    if manifest.nodes.is_empty() {
        return Ok(pool);
    }
    let mut owners = manifest.nodes.iter().filter(|n| n.pool_id == pool);
    let Some(owner) = owners.next() else {
        return Err(SddpError::Validation(format!(
            "boundary policy: pool {pool} has no owning node in the graph manifest \
             (internal inconsistency)"
        )));
    };
    if owners.next().is_some() {
        return Err(SddpError::Validation(format!(
            "boundary policy: pool {pool} is shared by more than one graph node; set \
             policy.boundary.source_stage explicitly"
        )));
    }
    u32::try_from(owner.stage_id).map_err(|_| {
        SddpError::Validation(format!(
            "boundary policy: pool {pool}'s owning node has a negative stage id {}",
            owner.stage_id
        ))
    })
}

/// Decode `manifest`'s live (non-sentinel) `AnticipatedThermalState` slots'
/// `delivery_date` anchors into `[month_start, month_end)` spans, via
/// [`decode_month_anchor`] — the per-pool candidate set
/// [`resolve_boundary_source_stage`] matches against the current study's
/// terminal delivery window.
fn decode_pool_anticipated_months(
    manifest: &[EntitySlot],
) -> Result<Vec<(NaiveDate, NaiveDate)>, SddpError> {
    manifest
        .iter()
        .filter(|slot| {
            slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
                && slot.delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        })
        .map(|slot| decode_month_anchor(slot.delivery_date).map(|(start, end, _)| (start, end)))
        .collect()
}

/// Mirrors [`compare_manifest_slot_identity`]'s dormant-to-active divergence
/// warning for the reconciled (`BoundaryInjection`) path: for each
/// identity-matched (`Copy`) target slot, warn when the source slot was
/// dormant but the current slot is active.
fn warn_on_dormant_source_now_active(
    rebind: &[RebindOp],
    source: &[EntitySlot],
    current: &[EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) {
    for (j, op) in rebind.iter().enumerate() {
        if let RebindOp::Copy(pos) = op
            && !source[*pos].was_active
            && current[j].was_active
        {
            on_warning(&format!(
                "slot {j} (entity_type={}, entity_id={}, subindex={}) was dormant in the source \
                 policy but is active in the current study; loading its cut",
                current[j].entity_type, current[j].entity_id, current[j].subindex
            ));
        }
    }
}

/// Boundary cut records that passed [`validate_policy_load`]'s
/// [`BoundaryInjection`] check matrix. The private fields mean the only way to
/// obtain one is [`load_boundary_cuts`], so [`inject_boundary_cuts`] cannot
/// compile against a bare, unvalidated `Vec<OwnedPolicyCutRecord>`. Derefs to
/// `[OwnedPolicyCutRecord]` for read access; carries the load's
/// [`BoundaryReconciliationReport`], read via [`Self::report`].
#[derive(Debug, Clone)]
pub struct ValidatedBoundaryCuts {
    records: Vec<OwnedPolicyCutRecord>,
    report: BoundaryReconciliationReport,
}

impl ValidatedBoundaryCuts {
    /// The reconciliation report [`load_boundary_cuts`] built for this load.
    #[must_use]
    pub fn report(&self) -> &BoundaryReconciliationReport {
        &self.report
    }
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
    use chrono::NaiveDate;
    use cobre_io::{
        EntitySlot, GraphManifest, ManifestEdge, ManifestNode, PolicyCheckpointMetadata,
        ProducerBlock, StageCutsPayload,
    };

    use super::{
        BoundaryInjection, FullFcf, NodeId, NodePos, PolicyStageManifest, TypedVec,
        compare_manifest_slot_identity, load_boundary_cuts, resolve_boundary_source_stage,
        resolve_warm_start_counts, validate_policy_load,
    };
    use crate::SddpError;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// All-`None` delivery intervals aligned to `len` — for
    /// [`load_boundary_cuts`] tests whose manifests carry no live-dated
    /// anticipated slot.
    fn no_intervals(len: usize) -> Vec<Option<(NaiveDate, NaiveDate)>> {
        vec![None; len]
    }

    /// Discard warnings: a `&mut dyn FnMut(&str)` for tests asserting only the
    /// `Result`.
    fn ignore_warnings() -> impl FnMut(&str) {
        |_| {}
    }

    /// A minimal producer block for artifact-writing test helpers.
    fn producer_block() -> ProducerBlock {
        ProducerBlock {
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        }
    }

    /// A 1:1 chain graph manifest over `n_stages` nodes (node id == stage id ==
    /// pool id) — the shape a chain-degenerate study writes.
    fn chain_manifest(n_stages: u32) -> GraphManifest {
        let nodes = (0..n_stages)
            .map(|t| ManifestNode {
                id: i32::try_from(t).unwrap(),
                stage_id: i32::try_from(t).unwrap(),
                pool_id: t,
            })
            .collect();
        let edges = (0..n_stages.saturating_sub(1))
            .map(|t| ManifestEdge {
                source_id: i32::try_from(t).unwrap(),
                target_id: i32::try_from(t + 1).unwrap(),
                probability: 1.0,
            })
            .collect();
        GraphManifest {
            n_pools: n_stages,
            nodes,
            edges,
        }
    }

    /// A shared empty graph manifest for `validate_policy_load` unit tests that
    /// exercise only the `state_dimension`/`num_stages`/slot-identity checks —
    /// an empty graph makes the graph-identity check a silent no-op.
    static EMPTY_GRAPH: std::sync::LazyLock<GraphManifest> =
        std::sync::LazyLock::new(GraphManifest::default);

    /// Build a [`PolicyStageManifest`] with `n_pools == num_stages` and the shared
    /// [`EMPTY_GRAPH`] (graph identity skipped) — for the pure validate tests.
    fn psm(state_dimension: u32, num_stages: u32, slots: &[EntitySlot]) -> PolicyStageManifest<'_> {
        PolicyStageManifest {
            state_dimension,
            num_stages,
            n_pools: num_stages,
            slots,
            graph: &EMPTY_GRAPH,
        }
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
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-14T00:00:00Z".to_string(),
            num_stages: n_stages,
            graph_manifest: chain_manifest(n_stages),
            producer: producer_block(),
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
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.11.0".to_string(),
            created_at: "2026-07-20T00:00:00Z".to_string(),
            num_stages: stage_id + 1,
            graph_manifest: chain_manifest(stage_id + 1),
            producer: ProducerBlock {
                cost_scale_factor,
                ..producer_block()
            },
        };
        cobre_io::write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).unwrap();
    }

    /// Behavioral: [`load_boundary_cuts`] on a MARKED checkpoint (canonical
    /// currency units at rest) loaded at a series of differing
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
                &[],
                None,
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
            &[],
            None,
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
            &[],
            None,
            loading_factor,
            &mut ignore_warnings(),
        )
        .unwrap();
        let ratio = LEGACY_COST_SCALE_FACTOR / loading_factor;
        assert!((cuts_nondefault[0].intercept - raw_intercept * ratio).abs() < 1e-9);
    }

    // ── rescale_cut_records_for_load unit tests ──────────────────────────────

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

    /// A legacy checkpoint (`source_cost_scale_factor: None`) loaded at the
    /// still-default [`LEGACY_COST_SCALE_FACTOR`] is a bit-exact no-op — the
    /// requirement that a legacy policy at the default factor never
    /// re-baselines.
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

    /// A legacy checkpoint loaded at a NON-default factor is interpreted as
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

    /// A marked checkpoint (`Some(s)`) is ALWAYS divided by
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

    /// Export/load transform property: export (multiply by `S`) then load at
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

    /// Export/load transform property: cross-factor linearity — exporting at
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

        let cuts = load_boundary_cuts(
            tmp.path(),
            2,
            10,
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        )
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

        let result = load_boundary_cuts(
            tmp.path(),
            99,
            10,
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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

        let result = load_boundary_cuts(
            tmp.path(),
            0,
            5,
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
            &[],
            None,
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
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
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
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    /// A single active `HydroInflowLag` slot (`entity_type 1`): `id` is the
    /// hydro, `lag_depth` the 1-based lag (as `build_stage_entity_manifest` emits).
    fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
        EntitySlot {
            entity_type: 1,
            entity_id: id,
            subindex: lag_depth,
            was_active: true,
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    /// A boundary cut carrying inflow-lag coefficients to depth 12, loaded against
    /// a study declaring `inflow_lag_depth = 6`, is rejected before the manifest
    /// checks with a message naming the boundary depth (12), the declared depth
    /// (6), and the fix (raise to at least 12) — the recombination-soundness gate.
    #[test]
    fn load_boundary_cuts_lag_depth_exceeds_declared_rejects_naming_depths_and_fix() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            Some(6),
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(
            result.is_err(),
            "a boundary cut deeper than the declared inflow_lag_depth must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("12"),
            "must name the boundary-cut depth 12: {msg}"
        );
        assert!(msg.contains('6'), "must name the declared depth 6: {msg}");
        assert!(
            msg.contains("inflow_lag_depth"),
            "must name the config field to raise: {msg}"
        );
        assert!(
            msg.contains("at least 12"),
            "must instruct raising to at least 12: {msg}"
        );
    }

    /// A boundary cut at depth 12 loaded against `inflow_lag_depth = 12` clears the
    /// depth gate; a slot-for-slot matching manifest then loads.
    #[test]
    fn load_boundary_cuts_lag_depth_within_declared_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), inflow_lag_slot(1, 12)];
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            Some(12),
            1_000_000.0,
            &mut ignore_warnings(),
        )
        .unwrap();

        assert_eq!(
            cuts.len(),
            2,
            "a boundary within the declared depth must load"
        );
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
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot match must emit no warning: {warnings:?}"
        );
    }

    /// Given a current terminal storage slot for hydro `9`, absent from a
    /// boundary source that prices hydros `7` and `2` only,
    /// `load_boundary_cuts` reconciles by identity and rejects naming the
    /// unpriced hydro `9`.
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
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(result.is_err(), "an unpriced hydro must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains('9'),
            "error must name the unpriced current hydro 9: {msg}"
        );
    }

    /// Given a current terminal storage slot for hydro `2`, but the boundary
    /// source prices hydro `2` only as an inflow-lag slot (no storage
    /// counterpart), `load_boundary_cuts` reconciles by identity and rejects
    /// naming the unpriced storage hydro `2` — the differently-typed slot the
    /// source happens to carry at the same raw position is irrelevant under
    /// identity matching.
    #[test]
    fn load_boundary_cuts_storage_slot_absent_from_differently_typed_source_rejects() {
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
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(
            result.is_err(),
            "a storage slot with no identity counterpart must reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains('2'),
            "error must name the unpriced storage hydro 2: {msg}"
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
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
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
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
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
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching bucket manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot bucket match must emit no warning: {warnings:?}"
        );
    }

    /// A policy exported WITHOUT travel-time buckets loaded by bucket-aware
    /// code whose terminal manifest has a `HydroTransitBucket` (type 3) slot
    /// at the same `state_dimension` succeeds: a target transit bucket with no
    /// source match defaults to `0.0` (distinct from storage/lag's
    /// reject-on-miss; a matching source transit slot is instead copied). The
    /// storage slot still loads its identity-matched coefficient.
    #[test]
    fn load_boundary_cuts_missing_transit_bucket_slot_identity_defaults_to_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        )
        .expect(
            "a transit-bucket slot with no source counterpart must default to zero, not reject",
        );

        assert_eq!(cuts.len(), 2, "both cuts must still load");
        for cut in cuts.iter() {
            assert_eq!(
                cut.coefficients,
                vec![1.0, 0.0],
                "the storage slot copies its matched coefficient (the fixture's uniform 1.0); the \
                 transit slot defaults to 0.0"
            );
        }
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
            &no_intervals(current.len()),
            None,
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

    // ── load_boundary_cuts manifest resolution (0.14 pool-keyed artifact) ─────

    /// Write a single-pool checkpoint whose one cut pool is keyed by `pool_id`
    /// (the payload `stage_id` field) and whose graph manifest declares the nodes
    /// in `graph_nodes` (`(node_id, stage_id, pool_id)`) — so `load_boundary_cuts`
    /// must resolve `source_stage` to `pool_id` THROUGH the manifest, not by a
    /// stage == pool coincidence.
    fn write_checkpoint_pool_keyed(
        dir: &std::path::Path,
        pool_id: u32,
        graph_nodes: &[(i32, i32, u32)],
        intercepts: &[f64],
    ) {
        let coefficients = vec![1.0_f64, 1.0];
        let cuts: Vec<cobre_io::PolicyCutRecord<'_>> = intercepts
            .iter()
            .enumerate()
            .map(|(i, &intercept)| cobre_io::PolicyCutRecord {
                cut_id: i as u64,
                slot_index: i as u32,
                iteration: 0,
                forward_pass_index: 0,
                intercept,
                coefficients: &coefficients,
                is_active: true,
            })
            .collect();
        let active: Vec<u32> = (0..intercepts.len() as u32).collect();
        let payload = StageCutsPayload {
            stage_id: pool_id,
            state_dimension: 2,
            capacity: intercepts.len() as u32,
            warm_start_count: 0,
            cuts: &cuts,
            active_cut_indices: &active,
            populated_count: intercepts.len() as u32,
            entity_manifest: &[],
        };
        let nodes = graph_nodes
            .iter()
            .map(|&(id, stage_id, pool_id)| ManifestNode {
                id,
                stage_id,
                pool_id,
            })
            .collect();
        let n_pools = graph_nodes
            .iter()
            .map(|&(_, _, p)| p + 1)
            .max()
            .unwrap_or(0);
        let metadata = PolicyCheckpointMetadata {
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.14.0".to_string(),
            created_at: "2026-08-04T00:00:00Z".to_string(),
            num_stages: 6,
            graph_manifest: GraphManifest {
                n_pools,
                nodes,
                edges: vec![],
            },
            producer: producer_block(),
        };
        cobre_io::write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).unwrap();
    }

    /// A `source_stage` naming a stage with more than one node is rejected: the
    /// frozen `policy.boundary` config offers no node selector, so a multi-node
    /// source is a named `SddpError::Validation` (no remedy field).
    #[test]
    fn load_boundary_cuts_multi_node_source_stage_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        // Two nodes at stage 3 (ids 30, 31) → stage 3 is multi-node.
        write_checkpoint_pool_keyed(tmp.path(), 2, &[(30, 3, 2), (31, 3, 2)], &[10.0, 20.0]);

        let result = load_boundary_cuts(
            tmp.path(),
            3,
            2,
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(result.is_err(), "a multi-node source_stage must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("multi-node"),
            "rejection must name the stage as multi-node: {msg}"
        );
        assert!(
            msg.contains("source_stage 3"),
            "rejection must name the stage: {msg}"
        );
    }

    /// A single-node `source_stage` resolves THROUGH the manifest to that node's
    /// pool, even when the pool id differs from the stage id (node at stage 5 →
    /// pool 2): the cuts loaded are pool 2's.
    #[test]
    fn load_boundary_cuts_single_node_resolves_through_manifest_to_pool() {
        let tmp = tempfile::tempdir().unwrap();
        // One node at stage 5 mapped to pool 2 (stage != pool).
        write_checkpoint_pool_keyed(tmp.path(), 2, &[(50, 5, 2)], &[10.0, 20.0, 30.0]);

        let cuts = load_boundary_cuts(
            tmp.path(),
            5,
            2,
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        )
        .expect("single-node source must resolve through the manifest");

        assert_eq!(
            cuts.len(),
            3,
            "must load all of pool 2's cuts, resolved from stage 5 via the manifest"
        );
        let intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(intercepts, vec![10.0, 20.0, 30.0]);
    }

    // ── resolve_boundary_source_stage tests ───────────────────────────────────

    fn ymd(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("valid calendar date")
    }

    /// A single `AnticipatedThermalState` slot (`entity_type 2`), dated at
    /// `delivery_date` (`YYYYMM01`).
    fn anticipated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
        EntitySlot {
            entity_type: 2,
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date,
        }
    }

    /// Write a checkpoint with one pool per `manifests` entry (pool id ==
    /// index, a chain-degenerate graph), each pool holding one cut sized to
    /// its own manifest — for [`resolve_boundary_source_stage`] tests
    /// exercising distinct per-pool anticipated calendars.
    fn write_checkpoint_with_pool_manifests(dir: &std::path::Path, manifests: &[Vec<EntitySlot>]) {
        let n_pools = manifests.len() as u32;
        let coefficients_per_pool: Vec<Vec<f64>> =
            manifests.iter().map(|m| vec![1.0_f64; m.len()]).collect();
        let cuts_per_pool: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = coefficients_per_pool
            .iter()
            .map(|coeffs| {
                vec![cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept: 0.0,
                    coefficients: coeffs,
                    is_active: true,
                }]
            })
            .collect();
        let active_indices = [0_u32];
        let payloads: Vec<StageCutsPayload<'_>> = manifests
            .iter()
            .enumerate()
            .map(|(pool, manifest)| StageCutsPayload {
                stage_id: pool as u32,
                state_dimension: manifest.len() as u32,
                capacity: 1,
                warm_start_count: 0,
                cuts: &cuts_per_pool[pool],
                active_cut_indices: &active_indices,
                populated_count: 1,
                entity_manifest: manifest,
            })
            .collect();
        let metadata = PolicyCheckpointMetadata {
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.14.0".to_string(),
            created_at: "2026-08-11T00:00:00Z".to_string(),
            num_stages: n_pools,
            graph_manifest: chain_manifest(n_pools),
            producer: producer_block(),
        };
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// A checkpoint whose pool 1 alone carries an anticipated slot dated
    /// inside the current terminal window resolves to pool 1.
    #[test]
    fn resolve_boundary_source_stage_unique_match_returns_pool_index() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![
            vec![storage_slot(1)],
            vec![anticipated_slot(9, 0, 20_260_301)],
        ];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![None, Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target)
            .expect("a unique calendar match must resolve");

        assert_eq!(resolved, 1, "pool 1 carries the matching anticipated month");
    }

    /// Two pools both carrying an anticipated slot dated inside the current
    /// terminal window are ambiguous: rejects naming the candidate count and
    /// advising an explicit `source_stage`.
    #[test]
    fn resolve_boundary_source_stage_ambiguous_multiple_pools_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![
            vec![anticipated_slot(9, 0, 20_260_301)],
            vec![anticipated_slot(9, 0, 20_260_301)],
        ];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "must name the ambiguity: {msg}");
        assert!(msg.contains('2'), "must name the candidate count 2: {msg}");
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
    }

    /// A source with no anticipated slot at all, on any pool, has no
    /// decodable delivery date to match against: rejects with the
    /// re-export hint.
    #[test]
    fn resolve_boundary_source_stage_fully_sentinel_dated_source_rejects_with_reexport_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![storage_slot(1)], vec![storage_slot(1)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("re-export"),
            "must advise re-exporting the boundary: {msg}"
        );
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
    }

    /// A source pool DOES carry a decodable anticipated month, but it falls
    /// outside the current terminal window: rejects advising an explicit
    /// `source_stage`, distinct from both the ambiguous and the
    /// sentinel-dated fallback.
    #[test]
    fn resolve_boundary_source_stage_no_overlap_among_decodable_pools_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![anticipated_slot(9, 0, 20_260_301)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 6, 1), ymd(2026, 7, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
        assert!(
            !msg.contains("ambiguous") && !msg.contains("re-export"),
            "a decodable-but-non-overlapping source is neither ambiguous nor sentinel-dated: {msg}"
        );
    }

    /// A non-existent boundary path surfaces the same IO-failure message
    /// [`load_boundary_cuts`] reports.
    #[test]
    fn resolve_boundary_source_stage_nonexistent_path_returns_error() {
        let err =
            resolve_boundary_source_stage(std::path::Path::new("/nonexistent/path/to/policy"), &[])
                .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("failed to read boundary policy checkpoint"),
            "error should describe the IO failure: {msg}"
        );
    }

    /// The resolver only PICKS a candidate by calendar: a pool that matches
    /// the terminal date but belongs to a storage-identity-incompatible
    /// source still rejects once `load_boundary_cuts` reconciles it — the
    /// resolver is not a new trust boundary.
    #[test]
    fn resolve_boundary_source_stage_wrong_system_still_rejected_by_load_boundary_cuts_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![storage_slot(7), anticipated_slot(9, 0, 20_260_301)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target_intervals = vec![None, Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target_intervals)
            .expect("a unique calendar match must resolve, even from an incompatible system");
        assert_eq!(resolved, 0);

        let current = vec![storage_slot(42), anticipated_slot(9, 100, 20_260_301)];
        let result = load_boundary_cuts(
            tmp.path(),
            resolved,
            current.len() as u32,
            &current,
            &target_intervals,
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(
            result.is_err(),
            "a calendar-matched but identity-incompatible source must still reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("42"),
            "the storage/lag identity reject must name the unpriced hydro 42: {msg}"
        );
    }

    /// Write a checkpoint from explicit `(pool_id, entity_manifest, intercept)`
    /// pools under an explicit `graph` (never the chain-degenerate default) —
    /// for tests where a pool id must diverge from its owning node's `stage_id`.
    fn write_checkpoint_with_explicit_graph(
        dir: &std::path::Path,
        pools: &[(u32, Vec<EntitySlot>, f64)],
        graph: GraphManifest,
    ) {
        let coefficients_per_pool: Vec<Vec<f64>> = pools
            .iter()
            .map(|(_, manifest, _)| vec![1.0_f64; manifest.len()])
            .collect();
        let cuts_per_pool: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = pools
            .iter()
            .zip(&coefficients_per_pool)
            .map(|((_, _, intercept), coeffs)| {
                vec![cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept: *intercept,
                    coefficients: coeffs,
                    is_active: true,
                }]
            })
            .collect();
        let active_indices = [0_u32];
        let payloads: Vec<StageCutsPayload<'_>> = pools
            .iter()
            .zip(&cuts_per_pool)
            .map(|((pool_id, manifest, _), cuts)| StageCutsPayload {
                stage_id: *pool_id,
                state_dimension: manifest.len() as u32,
                capacity: 1,
                warm_start_count: 0,
                cuts,
                active_cut_indices: &active_indices,
                populated_count: 1,
                entity_manifest: manifest,
            })
            .collect();
        let num_stages = graph
            .nodes
            .iter()
            .map(|n| n.stage_id)
            .max()
            .and_then(|m| u32::try_from(m + 1).ok())
            .unwrap_or(0);
        let metadata = PolicyCheckpointMetadata {
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.14.0".to_string(),
            created_at: "2026-08-11T00:00:00Z".to_string(),
            num_stages,
            graph_manifest: graph,
            producer: producer_block(),
        };
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// The reproduced defect: the resolver's winning POOL id (2) numerically
    /// coincides with an unrelated DECOY node's `stage_id`, whose OWN pool (5)
    /// carries different cuts. Returning the raw pool id (the pre-fix
    /// behavior) makes `load_boundary_cuts` resolve `source_stage` through
    /// the decoy node instead, silently loading the decoy's cuts (999.0)
    /// rather than the calendar-winning pool's (777.0). The fix maps the
    /// winning pool back through the manifest to its OWN owning node's
    /// `stage_id` (7 — distinct from every pool id and from the decoy's
    /// stage id), so `load_boundary_cuts` follows the correct node.
    #[test]
    fn resolve_boundary_source_stage_maps_winning_pool_to_its_owning_node_stage_id() {
        let tmp = tempfile::tempdir().unwrap();
        let winning_pool = 2;
        let decoy_pool = 5;

        let winning_manifest = vec![anticipated_slot(9, 0, 20_260_301)];
        let decoy_manifest = vec![storage_slot(1)];

        let graph = GraphManifest {
            n_pools: 6,
            nodes: vec![
                // Decoy: `stage_id` equals the winning pool's numeric value,
                // but owns a DIFFERENT pool.
                ManifestNode {
                    id: 100,
                    stage_id: winning_pool as i32,
                    pool_id: decoy_pool,
                },
                // Correct: owns the winning pool, at an unrelated stage id.
                ManifestNode {
                    id: 101,
                    stage_id: 7,
                    pool_id: winning_pool,
                },
            ],
            edges: vec![],
        };

        write_checkpoint_with_explicit_graph(
            tmp.path(),
            &[
                (winning_pool, winning_manifest, 777.0),
                (decoy_pool, decoy_manifest, 999.0),
            ],
            graph,
        );

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target)
            .expect("a unique calendar match must resolve");

        assert_eq!(
            resolved, 7,
            "must return the winning pool's OWNING NODE's stage id (7), not the raw pool id (2)"
        );

        let cuts = load_boundary_cuts(
            tmp.path(),
            resolved,
            1,
            &[],
            &no_intervals(1),
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        )
        .expect("the resolved stage id must thread correctly through load_boundary_cuts");

        assert_eq!(
            cuts.len(),
            1,
            "must load the winning pool's one cut, not the decoy's"
        );
        assert_eq!(
            cuts[0].intercept, 777.0,
            "must load the CORRECT pool's cut (777.0), never the decoy's (999.0)"
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
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);

        let report = validate_policy_load::<FullFcf>(&source, &current).unwrap();

        assert!(
            report.warnings.is_empty(),
            "identical manifests must emit no warning: {:?}",
            report.warnings
        );
    }

    /// A `state_dimension` mismatch is a hard reject on `FullFcf`, and its message
    /// names lag depth as a probable cause so an `inflow_lag_depth`-driven mismatch
    /// is legible.
    #[test]
    fn validate_policy_load_full_fcf_state_dimension_mismatch_rejects() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(8, 12, &slots);

        let result = validate_policy_load::<FullFcf>(&source, &current);

        assert!(result.is_err(), "state_dimension mismatch must reject");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include source value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
        assert!(
            msg.contains("inflow_lag_depth"),
            "message must name lag depth as a probable cause: {msg}"
        );
    }

    /// A `num_stages` mismatch is a hard reject on `FullFcf` but the identical
    /// inputs pass `BoundaryInjection` (unchecked there).
    #[test]
    fn validate_policy_load_num_stages_mismatch_rejects_full_fcf_oks_boundary() {
        let slots = storage_manifest(1, 2);
        let source = psm(10, 12, &slots);
        let current = psm(10, 24, &slots);

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
        let source = psm(10, 12, &slots);
        let current = psm(8, 24, &slots);

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
        let source = psm(2, 12, &source_slots);
        let current = psm(2, 12, &current_slots);

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

    /// `validate_policy_load` itself no longer checks per-slot identity under
    /// `BoundaryInjection` (unlike `FullFcf`): a same-`state_dimension`
    /// identity mismatch it would have hard-rejected now passes here, because
    /// slot identity is reconciled separately, by `reconcile::build_rebind`,
    /// in `load_boundary_cuts` — confined to that load path, not this
    /// lower-level manifest check.
    #[test]
    fn validate_policy_load_boundary_injection_does_not_check_slot_identity() {
        let source_slots = storage_manifest(7, 2);
        let current_slots = storage_manifest(9, 2);
        let source = psm(2, 12, &source_slots);
        let current = psm(2, 6, &current_slots);

        let result = validate_policy_load::<BoundaryInjection>(&source, &current);

        assert!(
            result.is_ok(),
            "BoundaryInjection defers slot identity to reconcile::build_rebind: {result:?}"
        );
    }

    /// An empty manifest on either side cannot be verified by slot identity:
    /// `validate_policy_load` falls back to the `state_dimension` check alone,
    /// returning `Ok` with one warning.
    #[test]
    fn validate_policy_load_full_fcf_empty_manifest_oks_with_warning() {
        let current_slots = storage_manifest(1, 2);
        let source = psm(2, 12, &[]);
        let current = psm(2, 12, &current_slots);

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
            format_version: cobre_io::FORMAT_VERSION,
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-01T00:00:00Z".to_string(),
            num_stages,
            graph_manifest: chain_manifest(num_stages),
            producer: ProducerBlock {
                warm_start_cuts,
                warm_start_counts,
                ..producer_block()
            },
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
        assert!(msg.contains('3'), "should include the pool count: {msg}");
    }

    /// `warm_start_counts` is a per-POOL field: on a branching graph the pool
    /// count differs from the stage count, so a checkpoint whose array length
    /// matches `n_pools` (not `num_stages`) must be accepted, and a checkpoint
    /// whose array length matches `num_stages` (not `n_pools`) must be
    /// rejected — asserting the check reads the caller's `n_pools` argument,
    /// never `metadata.num_stages`.
    #[test]
    fn resolve_warm_start_counts_validates_against_n_pools_not_num_stages() {
        // A 2-stage, 3-pool graph (e.g. one root pool + two branch pools
        // sharing a stage) — `meta_with_counts` derives `num_stages` from the
        // counts vector's own length, so override it after construction to
        // decouple the two counts.
        let mut meta = meta_with_counts(5, vec![10, 8, 6]);
        meta.num_stages = 2;

        let n_pools = 3;
        let counts = resolve_warm_start_counts(&meta, n_pools)
            .expect("length matches n_pools (3), not num_stages (2): must accept");
        assert_eq!(counts, vec![10u32, 8, 6]);

        // A caller that mistakenly passes the STAGE count (2) instead of the
        // pool count (3) must be rejected, not silently accepted against the
        // wrong axis.
        let mistaken_num_stages = 2;
        let result = resolve_warm_start_counts(&meta, mistaken_num_stages);
        let msg = result
            .expect_err("array length (3) disagrees with the passed count (2): must reject")
            .to_string();
        assert!(
            msg.contains("pools"),
            "the rejection must name pools, not stages: {msg}"
        );
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
            node_id: NodeId(0),
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

        let cache = build_basis_cache_from_checkpoint(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
        );
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

        let cache = build_basis_cache_from_checkpoint(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
        );
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

    // ── branching-graph basis-cache keying ────────────────────────────────────

    use cobre_io::{OwnedPolicyBasisRecord, StageCutsReadResult};

    /// An active cut record at LP slot `slot`.
    fn active_cut(slot: u32) -> OwnedPolicyCutRecord {
        OwnedPolicyCutRecord {
            cut_id: u64::from(slot),
            slot_index: slot,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 0.0,
            coefficients: vec![0.0],
            is_active: true,
        }
    }

    /// A pool-keyed cut collection (`stage_id` is the pool id) holding the given
    /// active slots.
    fn pool_cuts(pool: u32, slots: &[u32]) -> StageCutsReadResult {
        StageCutsReadResult {
            stage_id: pool,
            state_dimension: 1,
            capacity: 8,
            warm_start_count: 0,
            populated_count: slots.len() as u32,
            cuts: slots.iter().copied().map(active_cut).collect(),
            entity_manifest: Vec::new(),
        }
    }

    /// A node-keyed basis record (`stage_id` is the node ordinal) with
    /// `num_cut` trailing cut rows over `3` template rows.
    fn node_basis(node: u32, num_cut: usize) -> OwnedPolicyBasisRecord {
        OwnedPolicyBasisRecord {
            stage_id: node,
            iteration: 0,
            column_status: vec![1_u8, 1_u8],
            row_status: vec![1_u8; 3 + num_cut],
            num_cut_rows: num_cut as u32,
        }
    }

    /// A branching (K-fan) checkpoint where `n_nodes (7) > num_stages`: nodes
    /// 0/1/2 are interior (pools 0/1/2), leaves 3..=6 share pool 3. The cache is
    /// sized by `n_nodes`, every node lands in its own slot (no `>= num_stages`
    /// drop, no cross-node collision), each carries its own `node_id`, and the
    /// cut-slot reconstruction resolves through the node's OWN pool
    /// (`node_pools[node]`) — never the pool whose id equals the node ordinal,
    /// which for leaves 4/5/6 names no pool at all.
    #[test]
    fn build_basis_cache_from_checkpoint_keys_branching_graph_by_node() {
        use super::build_basis_cache_from_checkpoint;

        let node_ids: TypedVec<NodePos, NodeId> = vec![10, 11, 12, 13, 14, 15, 16]
            .into_iter()
            .map(NodeId)
            .collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2, 3, 3, 3, 3].into();
        let stage_cuts = vec![
            pool_cuts(0, &[0]),
            pool_cuts(1, &[1]),
            pool_cuts(2, &[2]),
            pool_cuts(3, &[5, 7]),
        ];
        let stage_bases = vec![
            node_basis(0, 1),
            node_basis(1, 1),
            node_basis(2, 1),
            node_basis(3, 2),
            node_basis(4, 2),
            node_basis(5, 2),
            node_basis(6, 2),
        ];

        let cache =
            build_basis_cache_from_checkpoint(&stage_bases, &stage_cuts, &node_ids, &node_pools);

        assert_eq!(cache.len(), 7, "cache is sized by n_nodes, not num_stages");
        for (node, slot) in cache.iter().enumerate() {
            let cb = slot
                .as_ref()
                .unwrap_or_else(|| panic!("node {node} must not be dropped"));
            assert_eq!(
                cb.node_id,
                node_ids[NodePos(node)],
                "node {node} must carry its own node_id (no cross-node collision)"
            );
        }

        // Interior node 2 (pool 2) recovers pool 2's single active slot.
        assert_eq!(cache[2].as_ref().unwrap().cut_row_slots, vec![2_u32]);
        // Every leaf (nodes 3..=6) recovers the SHARED pool 3's active slots,
        // keyed by node_pools[node] == 3 — a node-ordinal key would drop 4/5/6.
        for (node, slot) in cache.iter().enumerate().skip(3) {
            assert_eq!(
                slot.as_ref().unwrap().cut_row_slots,
                vec![5_u32, 7_u32],
                "leaf node {node} must recover shared pool 3's active slots"
            );
        }
    }

    /// Chain degeneracy: `node_pools` is the identity (`node_pools[t] == t`), so
    /// node-keyed sizing and pool-keyed cut matching reduce exactly to the
    /// pre-branching per-stage behavior — one basis per stage, keyed by ordinal.
    #[test]
    fn build_basis_cache_from_checkpoint_chain_is_identity_keyed() {
        use super::build_basis_cache_from_checkpoint;

        let node_ids: TypedVec<NodePos, NodeId> = vec![0, 1, 2].into_iter().map(NodeId).collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2].into();
        let stage_cuts = vec![pool_cuts(0, &[0]), pool_cuts(1, &[3]), pool_cuts(2, &[9])];
        let stage_bases = vec![node_basis(0, 1), node_basis(1, 1), node_basis(2, 1)];

        let cache =
            build_basis_cache_from_checkpoint(&stage_bases, &stage_cuts, &node_ids, &node_pools);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].as_ref().unwrap().cut_row_slots, vec![0_u32]);
        assert_eq!(cache[1].as_ref().unwrap().cut_row_slots, vec![3_u32]);
        assert_eq!(cache[2].as_ref().unwrap().cut_row_slots, vec![9_u32]);
    }
}
