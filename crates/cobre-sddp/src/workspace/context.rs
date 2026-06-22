//! Context structs for reducing parameter count in hot-path functions.

use cobre_core::{Stage, scenario::SamplingScheme, temporal::StageLagTransition};
use cobre_solver::StageTemplate;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    dcs::DcsParams,
    horizon_mode::HorizonMode,
    indexer::{StateLayout, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lp_builder::StageGeometry,
    noise_key_diag::NoiseKeyDiag,
};

/// Immutable per-stage LP layout and noise scaling parameters.
///
/// Groups the read-only parameters shared by the forward pass, backward pass,
/// and simulation pipeline. Constructed once at algorithm startup and passed
/// by reference to all hot-path functions.
pub struct StageContext<'a> {
    /// Stage LP templates (one per study stage).
    pub templates: &'a [StageTemplate],
    /// Row index of the first water-balance row in each stage template.
    pub base_rows: &'a [usize],
    /// Per-stage equipment geometry (one entry per study stage).
    ///
    /// `geometry_per_stage[t]` holds the stage-correct column and row ranges for
    /// stage `t`, sourced from that stage's `StageLayout`. A single global
    /// stage-0 geometry would carry `n_blks`-striped bases that misread any stage
    /// with a differing block count. The hot patch path reads the per-stage row bases
    /// (e.g. `z_inflow_row_start`) through this table, mirroring the sibling
    /// per-stage slices (`base_rows`, `load_balance_row_starts`, `ncs_col_starts`).
    /// Empty `&[]` in tests that drive a sub-path without a real stage table; the
    /// reader falls back to the all-empty `StageGeometry::default` in that case,
    /// matching how the sibling per-stage slices default.
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
    /// column at that stage. The NCS column base shifts per stage when a source
    /// commissions/decommissions mid-horizon or block counts vary, so the
    /// forward/backward bound patch strides from this per-stage base — never a
    /// single global stage-0 NCS base, which would address the wrong columns for
    /// non-uniform geometries. NCS presence is a `has_ncs` flag on
    /// [`StudyDimensions`]; there is no global NCS column base. Empty when the
    /// study has no NCS columns (the patch is guarded off in that case).
    pub ncs_col_starts: &'a [usize],
    /// NCS column count — the full system NCS count, identical at every stage.
    ///
    /// Under the dense layout every NCS keeps a column at every stage, so the count
    /// is a single scalar, not a per-stage slice (a dormant NCS keeps its column).
    /// Consumed by the lower-bound `ncs_generation` range; `0` when the study has
    /// no NCS.
    pub n_ncs: usize,
    /// Stage-invariant stochastic-slot → dense NCS column index map.
    ///
    /// `ncs_stochastic_dense_col[slot]` is the system NCS index (the dense column
    /// position) of the stochastic NCS at `slot` (id-sorted
    /// `StochasticContext::ncs_entity_ids` order, the order `transform_ncs_noise`
    /// emits its bound buffers). Under the dense layout every NCS keeps a column at
    /// every stage, so this map is identical at every stage; the NCS bound patch
    /// strides the per-opening cap onto
    /// `ncs_col_starts[stage] + ncs_stochastic_dense_col[slot] * n_blks + blk`.
    /// Length equals `n_stochastic_ncs`; empty when the study has no stochastic
    /// NCS. Built once at setup in `StudySetup`.
    pub ncs_stochastic_dense_col: &'a [usize],
    /// Stage-invariant commissioning window per stochastic NCS slot.
    ///
    /// `ncs_stochastic_windows[slot] = (entry, exit)` for the stochastic NCS at
    /// `slot` (id-sorted to match [`Self::ncs_stochastic_dense_col`] and the
    /// `transform_ncs_noise` buffer order). The NCS bound patch sites compute
    /// dormancy inline at each stage via the shared
    /// `commissioning_active` predicate
    /// and force a `[0, 0]` cap for a dormant slot — the zero-influence convention,
    /// applied identically across the forward, backward, and lower-bound patch
    /// sites. Length equals `n_stochastic_ncs`; empty when the study has no
    /// stochastic NCS. Built once at setup in `StudySetup`. Carrying the windows,
    /// not a per-stage dormancy mask, keeps activity out of per-stage storage.
    pub ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    /// Stage-invariant commissioning window per anticipated thermal.
    ///
    /// `anticipated_windows[i] = (entry, exit)` for the i-th anticipated thermal,
    /// in anticipated-local order (matching `StudyDimensions::anticipated_thermal_indices`).
    /// Threaded into the simulation
    /// [`StageExtractionSpec`](crate::simulation::extraction::StageExtractionSpec)
    /// so the anticipated-decision read gates on the same horizon-∧-operation-window
    /// predicate the LP builder used; the gate keys its operation-window clause on
    /// the DELIVERY stage's `stage.id`. Carrying the windows, not a per-stage
    /// activity mask, keeps activity out of per-stage storage — the same convention
    /// as [`Self::ncs_stochastic_windows`]. Empty when there are no anticipated
    /// thermals.
    pub anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    /// Study-stage commissioning id for each study stage index
    /// (`study_stage_ids[t] = stage.id`).
    ///
    /// The anticipated decision gate keys its operation-window clause on the
    /// DELIVERY stage's `stage.id` (not the stage index), so it maps the delivery
    /// stage index `t + K_i` to its id through this slice. Length equals the number
    /// of study stages.
    pub study_stage_ids: &'a [i32],
    /// Maximum generation (MW) per stochastic NCS entity, sorted by entity ID.
    /// Length equals the number of stochastic NCS entities. Empty when none exist.
    pub ncs_max_gen: &'a [f64],
    /// Per-stochastic-NCS curtailment policy, aligned 1:1 with
    /// [`Self::ncs_max_gen`]. `true` = the LP can dispatch in `[0, max × α
    /// × factor]`; `false` = must-run, the LP pins
    /// `col_lower = col_upper = max × α × factor` for every scenario.
    pub ncs_allow_curtailment: &'a [bool],
    /// One-step discount factor for the transition departing each stage.
    ///
    /// `discount_factors[t] = 1 / (1 + r)^(Dt / 365.25)` where `r` is the
    /// annual discount rate and `Dt` is the stage duration in days.
    /// Length equals the number of study stages. All `1.0` when no discount.
    pub discount_factors: &'a [f64],
    /// Cumulative discount factor at each stage for present-value costing.
    ///
    /// `cumulative_discount_factors[t]` is the product of all one-step discount
    /// factors for transitions preceding stage `t`. `[0] == 1.0` always.
    /// Length equals the number of study stages.
    pub cumulative_discount_factors: &'a [f64],
    /// Precomputed lag accumulation weights and period-finalization flags, one
    /// entry per study stage. Indexed by stage: `stage_lag_transitions[t]`.
    ///
    /// Populated by `crate::lag_transition::precompute_stage_lag_transitions`
    /// at setup time. Used by the forward pass and simulation pipeline.
    pub stage_lag_transitions: &'a [StageLagTransition],
    /// Noise group IDs for noise-group sharing, indexed by stage array index.
    ///
    /// Stages with the same group ID share the same noise draw in the opening
    /// tree and forward pass. For uniform monthly studies every stage has a
    /// unique group ID, so no sharing is triggered. Populated from
    /// `StudySetup::stage_data.noise_group_ids`
    /// at setup time. Length equals the number of study stages.
    pub noise_group_ids: &'a [u32],
    /// PAR order for the downstream (coarser-resolution) model.
    ///
    /// `0` for uniform-resolution studies — all downstream accumulation code paths
    /// in `accumulate_and_shift_lag_state` are skipped. Non-zero for studies with a
    /// monthly-to-quarterly transition (e.g., hybrid resolution studies).
    ///
    /// Used to size the downstream scratch buffers in the forward-pass workspace pool
    /// (`train`) and to pass as `par_order` to `crate::noise::DownstreamAccumState`.
    /// Set from `crate::setup::StudySetup::downstream_par_order` at setup time.
    pub downstream_par_order: usize,
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
}

/// Immutable algorithm-level configuration for the training loop.
///
/// Groups the read-only parameters shared by the training loop, forward pass,
/// backward pass, and simulation pipeline. Constructed once by the caller of
/// [`train`](crate::train) and passed by reference to all pass functions.
pub struct TrainingContext<'a> {
    /// Horizon mode (finite/infinite) determining stage count.
    pub horizon: &'a HorizonMode,
    /// Stage-invariant state-vector layout (role (a)): the single owner of the
    /// state column ranges, `n_state`, the resolvers, and the mask.
    ///
    /// Every hot-path role-(a) read resolves through this handle: the
    /// state-fixing column patch (`PatchBuffer::fill_col_state_patches`), the
    /// cut-row build and dual extraction (`cut::row`, `cut::dcs`,
    /// `duals_extraction`, the delta-cut consumer), and the lower-bound /
    /// simulation-extraction state reads.
    pub state: &'a StateLayout,
    /// Single owner of the study-invariant, non-state LP shape (non-state entity
    /// counts, optional-column presence flags, anticipated-thermal identity list).
    ///
    /// Nested contexts that already borrow this `TrainingContext` (the forward
    /// worker spec, the session) reach it transitively as
    /// `training_ctx.study_dims` rather than carrying an independent handle.
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
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::Historical`, `None` otherwise.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    ///
    /// `Some` when `load_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    ///
    /// `Some` when `ncs_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
    /// Per-hydro accumulated `value_m3s * hours` seed values from pre-study
    /// `RecentObservation` data.
    ///
    /// Copied into `ws.scratch.lag_accumulator` at every trajectory start
    /// instead of zero-filling. Empty slice when there are no observations
    /// (backward-compatible: the zero-fill path is taken instead).
    pub recent_accum_seed: &'a [f64],
    /// Fraction of the lag period covered by pre-study observations.
    ///
    /// Set into `ws.scratch.lag_weight_accum` at every trajectory start.
    /// `0.0` when there are no observations.
    pub recent_weight_seed: f64,
    /// Dynamic Cut Selection hyperparameters, `Some` only when the configured
    /// cut-selection method is the dynamic variant; `None` for every other
    /// method (and when cut selection is disabled). When `Some` and the current
    /// iteration is at or past `start_iteration`, the backward pass solves each
    /// stage LP lazily; otherwise the baked all-cuts path is used.
    pub dcs: Option<DcsParams>,
    /// Throwaway, env-gated backward `noise_key` diagnostic table.
    ///
    /// `Some` only when the `COBRE_W1_DIAG` environment variable was set at
    /// setup; `None` otherwise (the default), in which case the backward pass
    /// performs zero key computation and the solve path is byte-identical. When
    /// `Some`, the backward pass looks up the precomputed per-(stage, ω) noise
    /// key by canonical ω and emits it alongside the opening's
    /// `simplex_iterations`. See [`crate::noise_key_diag`].
    pub noise_key_diag: Option<&'a NoiseKeyDiag>,
}
