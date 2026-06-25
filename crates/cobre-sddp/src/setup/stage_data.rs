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

    /// Canonical stage-invariant state-vector layout (role (a)): the state / cut
    /// column ranges and the layout-derived caches.
    ///
    /// The single role-(a) owner — resolves every role-(a) read on the hot path.
    /// Role-(b) equipment geometry lives per stage on
    /// [`crate::lp_builder::StageGeometry`].
    pub(crate) state: StateLayout,

    /// Single owner of the study-invariant, non-state LP shape: the non-state
    /// entity counts, the optional-column presence flags, and the
    /// anticipated-thermal identity list.
    ///
    /// The state-defining dims live on [`Self::state`] and the per-stage `n_blks`
    /// lives on the per-stage geometry, so neither is carried here.
    pub(crate) study_dims: StudyDimensions,

    /// Study stages (id >= 0) in index order.
    pub(crate) stages: Vec<Stage>,

    /// Entity IDs and productivities for all dispatch entities.
    pub(crate) entity_counts: EntityCounts,

    /// Per-station pumping power-consumption rate \[MW/(m³/s)\], ID-sorted to
    /// match `entity_counts.pumping_station_ids`.
    ///
    /// Built from `system.pumping_stations()` in canonical ID order — the same
    /// order `pumping_station_ids` is built in — so the two slices are
    /// positionally aligned.
    pub(crate) pumping_consumption_mw_per_m3s: Vec<f64>,

    /// Per-stage RESOLVED contract price \[$/`MWh`\]: one inner `Vec` per study
    /// stage, ID-sorted to match `entity_counts.contract_ids`. The resolved,
    /// possibly stage-overridden `contract_bounds(c, t).price_per_mwh`.
    pub(crate) contract_prices_per_stage: Vec<Vec<f64>>,

    /// Direction per contract, ID-sorted to match `entity_counts.contract_ids`
    /// (`true` = import). Stage-invariant.
    pub(crate) contract_is_import: Vec<bool>,

    /// Number of blocks per stage.
    pub(crate) block_counts_per_stage: Vec<usize>,

    /// Precomputed lag accumulation weights and period-finalization flags,
    /// one entry per study stage.
    pub(crate) stage_lag_transitions: Vec<StageLagTransition>,

    /// Noise group assignments: stages with the same `(season_id, year)` share a
    /// noise group.
    pub noise_group_ids: Vec<u32>,

    /// LP scaling report captured during template build.
    pub scaling_report: ScalingReport,
}
