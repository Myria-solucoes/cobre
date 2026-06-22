//! Stage-indexed data sub-struct extracted from [`super::StudySetup`].

use cobre_core::{Stage, temporal::StageLagTransition};

use crate::{
    indexer::{StateLayout, StudyDimensions},
    lp_builder::StageTemplates,
    scaling_report::ScalingReport,
    simulation::EntityCounts,
};

/// All per-stage and stage-indexed data owned by [`super::StudySetup`].
///
/// Groups the per-stage and stage-indexed fields that describe the study's
/// temporal and stage structure. Constructed once during
/// [`super::StudySetup::from_broadcast_params`] and borrowed for hot-path
/// context construction.
#[derive(Debug)]
pub struct StageData {
    /// LP skeleton templates, one per study stage.
    pub stage_templates: StageTemplates,

    /// Canonical stage-invariant state-vector layout (role (a)): the state /
    /// cut column ranges and the two layout-derived caches
    /// (`state_to_lp_column_map`, `nonzero_state_indices`).
    ///
    /// Built once in [`super::build_wired_indexer`] from the study dimensions and
    /// per-hydro effective lag counts. This is the single role-(a) owner: it is
    /// borrowed into `TrainingContext::state` and resolves every role-(a) read on
    /// the hot path — the state-fixing patch
    /// (`PatchBuffer::fill_col_state_patches`), the cut-row build and dual
    /// extraction (`cut::row`, `cut::dcs`, the delta-cut and `duals_extraction`
    /// consumers), and the simulation extraction's state columns. The role-(b)
    /// equipment geometry lives per stage on [`crate::lp_builder::StageGeometry`].
    pub(crate) state: StateLayout,

    /// Single owner of the study-invariant, non-state LP shape: the non-state
    /// entity counts, the optional-column presence flags, and the
    /// anticipated-thermal identity list.
    ///
    /// Built once in [`super::build_wired_indexer`] alongside [`Self::state`].
    /// Borrowed into `TrainingContext::study_dims` and
    /// `StageExtractionSpec::study_dims`; every reader of one of these facts
    /// resolves it here. The state-defining dims live on [`Self::state`] and the
    /// per-stage `n_blks` lives on the per-stage geometry, so neither is carried
    /// here.
    pub(crate) study_dims: StudyDimensions,

    /// Study stages (id >= 0) in index order.
    ///
    /// Borrowed by `TrainingContext` so that
    /// [`cobre_stochastic::build_forward_sampler`] can read per-stage noise
    /// methods when constructing an `OutOfSample` sampler.
    pub(crate) stages: Vec<Stage>,

    /// Entity IDs and productivities for all dispatch entities.
    pub(crate) entity_counts: EntityCounts,

    /// Per-station pumping power-consumption rate \[MW/(m³/s)\], ID-sorted to
    /// match `entity_counts.pumping_station_ids`.
    ///
    /// Threaded into the simulation extraction pipeline to compute
    /// `power_consumption_mw = pumped_flow_m3s * consumption_mw_per_m3s` for each
    /// pumping row. Built from `system.pumping_stations()`, which returns the
    /// stations in canonical ID order — the same order `pumping_station_ids` is
    /// built in — so the two slices are positionally aligned.
    pub(crate) pumping_consumption_mw_per_m3s: Vec<f64>,

    /// Number of blocks per stage.
    pub(crate) block_counts_per_stage: Vec<usize>,

    /// Precomputed lag accumulation weights and period-finalization flags,
    /// one entry per study stage. Indexed by stage: `stage_lag_transitions[t]`.
    ///
    /// Computed once at setup time by
    /// [`crate::lag_transition::precompute_stage_lag_transitions`].
    pub(crate) stage_lag_transitions: Vec<StageLagTransition>,

    /// Pre-computed noise group assignments for noise sharing via noise-group sharing.
    ///
    /// Stages with the same `(season_id, year)` share a noise group. Computed at
    /// setup time by `precompute_noise_groups`.
    pub noise_group_ids: Vec<u32>,

    /// LP scaling report captured during template build.
    pub scaling_report: ScalingReport,
}
