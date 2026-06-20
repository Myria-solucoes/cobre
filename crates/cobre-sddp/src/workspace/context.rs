//! Context structs for reducing parameter count in hot-path functions.

use cobre_core::{Stage, scenario::SamplingScheme, temporal::StageLagTransition};
use cobre_solver::StageTemplate;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    dcs::DcsParams, horizon_mode::HorizonMode, indexer::StageIndexer,
    inflow_method::InflowNonNegativityMethod, noise_key_diag::NoiseKeyDiag,
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
    /// non-uniform geometries. The `StageIndexer` carries only `has_ncs` (a
    /// presence flag), not a column base. Empty when the study has no NCS columns
    /// (the patch is guarded off in that case).
    pub ncs_col_starts: &'a [usize],
    /// Per-stage stochastic-slot → active-local-column map for NCS bound patching.
    ///
    /// `ncs_active_slot_to_local[stage][slot]` is `Some(ncs_local)` when the
    /// stochastic NCS at `slot` (id-sorted `StochasticContext::ncs_entity_ids`
    /// order, the order `transform_ncs_noise` emits its bound buffers) is active
    /// at that stage; `ncs_local` is its column position within
    /// `active_ncs_indices[stage]`, so its LP column block is
    /// `ncs_col_starts[stage] + ncs_local * n_blks + blk`. `None` means the slot
    /// is inactive at that stage and contributes no column and no bound entry.
    /// The forward/backward NCS bound patch strides by `ncs_local`, never by the
    /// raw stochastic slot — a strict-subset stage allocates only
    /// `active_ncs_indices[stage].len()` NCS columns, so addressing all
    /// `n_stochastic_ncs` slots would overrun the block. Empty when the study has
    /// no stochastic NCS columns (the patch is guarded off in that case).
    /// Built once at setup as [`crate::setup::StudySetup::ncs_active_slot_to_local`].
    pub ncs_active_slot_to_local: &'a [Vec<Option<usize>>],
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
    /// Stage indexer providing state layout and entity counts.
    pub indexer: &'a StageIndexer,
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
