//! Context structs for reducing parameter count in hot-path functions.

use cobre_core::{Stage, scenario::SamplingScheme, temporal::StageLagTransition};
use cobre_solver::StageTemplate;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    dcs::DcsParams,
    horizon_mode::HorizonMode,
    indexer::{CutStateProjection, StateLayout, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lag_transition::DisaggregationWeight,
    lp_builder::StageGeometry,
};

/// Immutable per-stage LP layout and noise scaling parameters.
///
/// Read-only parameters shared by the forward pass, backward pass, and
/// simulation pipeline. Slice fields are indexed by study stage `t` unless
/// noted; per-stage NCS/anticipated slices are empty when the study lacks that
/// entity class.
pub struct StageContext<'a> {
    /// Stage LP templates.
    pub templates: &'a [StageTemplate],
    /// Row index of the first water-balance row in each stage template.
    pub base_rows: &'a [usize],
    /// Per-stage equipment geometry.
    ///
    /// `geometry_per_stage[t]` holds the stage-correct column and row ranges for
    /// stage `t`; a single global stage-0 geometry would carry `n_blks`-striped
    /// bases that misread any stage with a differing block count. Empty `&[]` in
    /// tests driving a sub-path without a stage table; the reader then falls back
    /// to the all-empty `StageGeometry::default`.
    pub geometry_per_stage: &'a [StageGeometry],
    /// Noise scaling factors, layout: `[stage * n_hydros + hydro]`.
    pub noise_scale: &'a [f64],
    /// Number of hydro plants with LP variables.
    pub n_hydros: usize,
    /// Number of buses with stochastic load noise.
    pub n_load_buses: usize,
    /// Row index of the first load-balance row in each stage template.
    pub load_balance_row_starts: &'a [usize],
    /// Bus indices for stochastic load mapping.
    pub load_bus_indices: &'a [usize],
    /// Number of blocks per stage.
    pub block_counts_per_stage: &'a [usize],
    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage]` is the column index of the first NCS generation
    /// column at that stage. The base shifts per stage under mid-horizon
    /// commissioning or varying block counts, so the forward/backward bound patch
    /// strides from this per-stage base, never a single global stage-0 NCS base
    /// (which would address the wrong columns for non-uniform geometries).
    pub ncs_col_starts: &'a [usize],
    /// Full-system NCS column count, identical at every stage (the dense layout
    /// keeps a dormant NCS's column). Consumed by the lower-bound
    /// `ncs_generation` range.
    pub n_ncs: usize,
    /// Stage-invariant stochastic-slot → dense NCS column index map.
    ///
    /// `ncs_stochastic_dense_col[slot]` is the dense column position of the
    /// stochastic NCS at `slot` (id-sorted `StochasticContext::ncs_entity_ids`
    /// order, the order `transform_ncs_noise` emits its bound buffers). The NCS
    /// bound patch strides the per-opening cap onto
    /// `ncs_col_starts[stage] + ncs_stochastic_dense_col[slot] * n_blks + blk`.
    /// Length equals `n_stochastic_ncs`.
    pub ncs_stochastic_dense_col: &'a [usize],
    /// Stage-invariant commissioning window per stochastic NCS slot.
    ///
    /// `ncs_stochastic_windows[slot] = (entry, exit)`, id-sorted to match
    /// [`Self::ncs_stochastic_dense_col`]. The forward, backward, and lower-bound
    /// patch sites force a `[0, 0]` cap (the zero-influence convention)
    /// identically for a slot dormant at the current stage. Length equals
    /// `n_stochastic_ncs`.
    pub ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    /// Stage-invariant commissioning window per anticipated thermal,
    /// anticipated-local order (matching
    /// `StudyDimensions::anticipated_thermal_indices`).
    ///
    /// `anticipated_windows[i] = (entry, exit)`. The simulation
    /// anticipated-decision read gates on the operation-window clause keyed to the
    /// DELIVERY stage's `stage.id`, the same predicate the LP builder used.
    pub anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    /// `stage.id` for each study stage index (`study_stage_ids[t] = stage.id`).
    ///
    /// The anticipated decision gate keys its operation-window clause on the
    /// DELIVERY stage's `stage.id` (not the index), mapping the delivery index
    /// `t + K_i` to its id through this slice.
    pub study_stage_ids: &'a [i32],
    /// Maximum generation (MW) per stochastic NCS entity, sorted by entity ID.
    pub ncs_max_gen: &'a [f64],
    /// Per-stochastic-NCS curtailment policy, aligned 1:1 with
    /// [`Self::ncs_max_gen`]. `true` = the LP dispatches in `[0, max × α
    /// × factor]`; `false` = must-run, pinned `col_lower = col_upper = max × α
    /// × factor` for every scenario.
    pub ncs_allow_curtailment: &'a [bool],
    /// One-step discount factor for the transition departing each stage:
    /// `1 / (1 + r)^(Dt / 365.25)` for annual rate `r` and stage duration `Dt`
    /// in days. All `1.0` when no discount.
    pub discount_factors: &'a [f64],
    /// Cumulative discount factor for present-value costing:
    /// `cumulative_discount_factors[t]` is the product of all one-step factors
    /// for transitions preceding stage `t`. `[0] == 1.0` always.
    pub cumulative_discount_factors: &'a [f64],
    /// Precomputed per-stage lag accumulation weights and period-finalization
    /// flags. Used by the forward pass and simulation pipeline.
    pub stage_lag_transitions: &'a [StageLagTransition],
    /// Noise group IDs for noise-group sharing, indexed by stage array index.
    /// Stages sharing a group ID share a noise draw in the opening tree and
    /// forward pass; uniform monthly studies give each stage a unique ID (no
    /// sharing).
    pub noise_group_ids: &'a [u32],
    /// PAR order for the downstream (coarser-resolution) model.
    ///
    /// `0` for uniform-resolution studies — all downstream accumulation paths in
    /// `accumulate_and_shift_lag_state` are skipped. Non-zero for a
    /// monthly-to-quarterly transition. Sizes the downstream scratch buffers and
    /// is passed as `par_order` to `crate::noise::DownstreamAccumState`.
    pub downstream_par_order: usize,
    /// Precomputed Regime A day-weighted disaggregation weights, one entry per
    /// study stage. Read via [`Self::disaggregation_weight_at`], never indexed
    /// directly — empty in call sites (mostly tests) not exercising the
    /// feature.
    pub disaggregation_weights: &'a [DisaggregationWeight],
    /// Per-stage m3s→hm3 flow-to-volume scale (ζ), from
    /// `crate::lp_builder::StageTemplates::zeta_per_stage`. Read only when a
    /// boundary stage's [`DisaggregationWeight::next_day_weight`] is positive.
    pub zeta_s: &'a [f64],
}

impl StageContext<'_> {
    /// Returns the noise group ID for stage index `t`.
    #[inline]
    #[must_use]
    pub fn noise_group_id_at(&self, t: usize) -> u32 {
        if self.noise_group_ids.is_empty() {
            #[allow(clippy::cast_possible_truncation)]
            return t as u32;
        }
        debug_assert!(
            t < self.noise_group_ids.len(),
            "stage index {t} out of bounds for noise_group_ids (len={})",
            self.noise_group_ids.len()
        );
        self.noise_group_ids[t]
    }

    /// Returns the day-weighted disaggregation weight for stage index `t`.
    ///
    /// Falls back to `DisaggregationWeight::interior` (a no-op:
    /// `next_day_weight == 0.0`, `next_period == None`) when
    /// `disaggregation_weights` is empty — the same empty-slice fallback shape
    /// [`Self::noise_group_id_at`] uses, covering call sites that do not
    /// exercise Regime A disaggregation.
    #[inline]
    #[must_use]
    pub fn disaggregation_weight_at(&self, t: usize) -> DisaggregationWeight {
        if self.disaggregation_weights.is_empty() {
            return DisaggregationWeight::interior(0);
        }
        debug_assert!(
            t < self.disaggregation_weights.len(),
            "stage index {t} out of bounds for disaggregation_weights (len={})",
            self.disaggregation_weights.len()
        );
        self.disaggregation_weights[t]
    }
}

/// Immutable algorithm-level configuration for the training loop.
///
/// Read-only parameters shared by the training loop, forward pass, backward
/// pass, and simulation pipeline. The external-scenario `Option` libraries are
/// `Some` exactly when their entity class's `SamplingScheme` selects them.
pub struct TrainingContext<'a> {
    /// Horizon mode (finite/infinite) determining stage count.
    pub horizon: &'a HorizonMode,
    /// Single owner of the state-vector layout: the state column ranges,
    /// `n_state`, the resolvers, and the mask. Every hot-path state-column read
    /// resolves through this handle.
    pub state: &'a StateLayout,
    /// Per-pool cut-state projection, indexed by pool `t` (paired 1:1 with
    /// `FutureCostFunction::pools`). The backward pass reads
    /// `cut_state_layouts[t]` when solving stage `t+1` to size pool `t`'s
    /// extracted subgradient and every per-stage backward buffer. Empty on the
    /// non-training paths (simulation, lower-bound eval), which never extract
    /// cuts.
    pub cut_state_layouts: &'a [CutStateProjection],
    /// Single owner of the study-invariant, non-state LP shape (non-state entity
    /// counts, optional-column presence flags, anticipated-thermal identity
    /// list). Nested contexts reach it transitively as `training_ctx.study_dims`.
    pub study_dims: &'a StudyDimensions,
    /// Inflow non-negativity enforcement strategy.
    pub inflow_method: &'a InflowNonNegativityMethod,
    /// Stochastic context providing noise generation and PAR model.
    pub stochastic: &'a StochasticContext,
    /// Initial state vector for stage 0.
    pub initial_state: &'a [f64],
    /// Forward-pass noise source scheme for the inflow entity class.
    pub inflow_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the load entity class.
    pub load_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the NCS entity class.
    pub ncs_scheme: SamplingScheme,
    /// Study stages (id >= 0) in index order; required by [`cobre_stochastic::build_forward_sampler`].
    pub stages: &'a [Stage],
    /// Pre-standardized historical inflow windows library.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
    /// Per-hydro accumulated `value_m3s * hours` seed from pre-study
    /// `RecentObservation` data, copied into `ws.scratch.lag_accumulator` at
    /// every trajectory start. Empty slice when there are no observations (the
    /// zero-fill path is taken instead).
    pub recent_accum_seed: &'a [f64],
    /// Fraction of the lag period covered by pre-study observations, set into
    /// `ws.scratch.lag_weight_accum` at every trajectory start. `0.0` when there
    /// are no observations.
    pub recent_weight_seed: f64,
    /// Dynamic Cut Selection hyperparameters, `Some` only when the dynamic
    /// cut-selection method is configured. When `Some` and the iteration is at or
    /// past `start_iteration`, the backward pass solves each stage LP lazily;
    /// otherwise the frozen all-cuts path is used.
    pub dcs: Option<DcsParams>,
}
