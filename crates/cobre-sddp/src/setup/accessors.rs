//! Accessor methods and context builders for [`StudySetup`].

use cobre_core::scenario::SamplingScheme;

use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    energy_conversion::EnergyConversionSet,
    indexer::{StageIndexer, StateLayout},
    simulation::SimulationConfig,
    workspace::CapturedBasis,
};

use super::StudySetup;

impl StudySetup {
    // -------------------------------------------------------------------------
    // Mutation setters — remain `pub` (called from cobre-cli or cobre-python)
    // -------------------------------------------------------------------------

    /// Replace the FCF with a pre-loaded policy.
    pub fn replace_fcf(&mut self, fcf: FutureCostFunction) {
        self.fcf = fcf;
    }

    /// Set the starting iteration for resumed training.
    pub fn set_start_iteration(&mut self, iteration: u64) {
        self.loop_params.start_iteration = iteration;
    }

    /// Seed the per-stage warm-start basis cache for warm-start / resume
    /// training.
    ///
    /// `cache` carries one entry per stage (as built by
    /// [`build_basis_cache_from_checkpoint`](crate::build_basis_cache_from_checkpoint)
    /// from the checkpoint's stored solver bases). [`StudySetup::train`] takes
    /// this out of `self` and replicates each stage's basis across every
    /// forward-pass worker so iteration 1's cut-loaded LPs warm-start.
    ///
    /// Leave unset (the default `None`) for a fresh start.
    pub fn set_warm_start_basis_cache(&mut self, cache: Vec<Option<CapturedBasis>>) {
        self.warm_start_basis_cache = Some(cache);
    }

    /// Enable state archiving for export.
    pub fn set_export_states(&mut self, export: bool) {
        self.events.export_states = export;
    }

    /// Set the active-cut budget cap per stage.
    pub fn set_budget(&mut self, budget: Option<u32>) {
        self.cut_management.budget = budget;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Context builders — span multiple sub-structs
    // ─────────────────────────────────────────────────────────────────────

    /// Return the pre-computed [`EnergyConversionSet`] for this study.
    ///
    /// Provides `ρ_eq`, `V_ref`, `Q_ref`, and `ρ_acum` (accumulated cascade
    /// productivity) for every `(hydro, stage)` pair. Consumed by the
    /// energy-balance LP constraints and inflow-energy / stored-energy extraction.
    #[must_use]
    pub fn energy_conversion(&self) -> &EnergyConversionSet {
        &self.energy_conversion
    }

    /// Return a reference to the simulation configuration.
    #[must_use]
    pub fn simulation_config(&self) -> &SimulationConfig {
        &self.simulation_config
    }

    /// Return a reference to the per-stage role-(b) LP geometry descriptor.
    ///
    /// Provides the equipment / slack column and constraint-row ranges for every
    /// entity class (turbine, spillage, thermal, anticipated decision, deficit,
    /// …) plus the entity-count scalars and presence flags, so callers can locate
    /// equipment primal entries without hard-coding offsets. The state-vector
    /// concern (storage / lag / anticipated-state columns, `theta`, `n_state`)
    /// lives on [`Self::stage_state`].
    ///
    /// The same descriptor applies to every stage whose block count equals stage
    /// 0's; the per-stage layout is the authority where block counts vary.
    #[must_use]
    pub fn stage_indexer(&self) -> &StageIndexer {
        &self.stage_data.indexer
    }

    /// Return a reference to the stage-invariant role-(a) state-vector layout.
    ///
    /// Provides the state-region column ranges (`storage`, `inflow_lags`,
    /// `storage_in`, `anticipated_state`, `anticipated_state_out`, `z_inflow`),
    /// the `theta` future-cost column, `n_state`, the state-vector dimension
    /// scalars (`hydro_count`, `max_par_order`, `n_anticipated`, `k_max`,
    /// `anticipated_lead_stages`), and the cut resolvers / mask. These offsets are
    /// pure functions of `(N, L, A, k_max)`, so the single layout resolves onto
    /// the correct column at every stage regardless of per-stage block counts.
    #[must_use]
    pub fn stage_state(&self) -> &StateLayout {
        &self.stage_data.state
    }

    /// Number of stages in the planning horizon.
    ///
    /// Used by the CLI summary to express the pool-level active-row total on a
    /// per-stage basis, so it is directly comparable to the per-solve
    /// rows-in-LP metric reported for Dynamic Cut Selection.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.methodology.horizon.num_stages()
    }

    /// Construct a [`StageContext`] borrowing from this setup.
    #[must_use]
    pub fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.stage_data.stage_templates.templates,
            base_rows: &self.stage_data.stage_templates.base_rows,
            noise_scale: &self.stage_data.stage_templates.noise_scale,
            n_hydros: self.stage_data.stage_templates.n_hydros,
            n_load_buses: self.stage_data.stage_templates.n_load_buses,
            load_balance_row_starts: &self.stage_data.stage_templates.load_balance_row_starts,
            load_bus_indices: &self.stage_data.stage_templates.load_bus_indices,
            block_counts_per_stage: &self.stage_data.block_counts_per_stage,
            ncs_col_starts: &self.stage_data.stage_templates.ncs_col_starts,
            ncs_active_slot_to_local: &self.ncs_active_slot_to_local,
            ncs_max_gen: &self.ncs_max_gen,
            ncs_allow_curtailment: &self.ncs_allow_curtailment,
            discount_factors: self.stage_data.stage_templates.discount_factors(),
            cumulative_discount_factors: self
                .stage_data
                .stage_templates
                .cumulative_discount_factors(),
            stage_lag_transitions: &self.stage_data.stage_lag_transitions,
            noise_group_ids: &self.stage_data.noise_group_ids,
            downstream_par_order: self.downstream_par_order,
        }
    }

    /// Construct a [`TrainingContext`] borrowing from this setup. Test-only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn training_ctx(&self) -> TrainingContext<'_> {
        let tr = &self.scenario_libraries.training;
        TrainingContext {
            horizon: &self.methodology.horizon,
            indexer: &self.stage_data.indexer,
            state: &self.stage_data.state,
            study_dims: &self.stage_data.study_dims,
            inflow_method: &self.methodology.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: tr.inflow_scheme,
            load_scheme: tr.load_scheme,
            ncs_scheme: tr.ncs_scheme,
            stages: &self.stage_data.stages,
            historical_library: tr.historical.as_ref(),
            external_inflow_library: tr.external_inflow.as_ref(),
            external_load_library: tr.external_load.as_ref(),
            external_ncs_library: tr.external_ncs.as_ref(),
            recent_accum_seed: &self.recent_observation_seed.accum_seed,
            recent_weight_seed: self.recent_observation_seed.weight_seed,
            dcs: self
                .cut_management
                .cut_selection
                .as_ref()
                .and_then(crate::dcs::DcsParams::from_strategy),
            noise_key_diag: None,
        }
    }

    /// Build simulation [`TrainingContext`] with simulation-specific schemes and libraries.
    ///
    /// Reuses training libraries when simulation schemes match. Selects per-class
    /// libraries in this order: simulation-specific, then training (shared).
    #[must_use]
    pub(crate) fn simulation_ctx(&self) -> TrainingContext<'_> {
        let tr = &self.scenario_libraries.training;
        let sim = &self.scenario_libraries.simulation;

        let historical_library =
            sim.historical
                .as_ref()
                .or(if sim.inflow_scheme == SamplingScheme::Historical {
                    tr.historical.as_ref()
                } else {
                    None
                });
        let external_inflow_library =
            sim.external_inflow
                .as_ref()
                .or(if sim.inflow_scheme == SamplingScheme::External {
                    tr.external_inflow.as_ref()
                } else {
                    None
                });
        let external_load_library =
            sim.external_load
                .as_ref()
                .or(if sim.load_scheme == SamplingScheme::External {
                    tr.external_load.as_ref()
                } else {
                    None
                });
        let external_ncs_library =
            sim.external_ncs
                .as_ref()
                .or(if sim.ncs_scheme == SamplingScheme::External {
                    tr.external_ncs.as_ref()
                } else {
                    None
                });

        TrainingContext {
            horizon: &self.methodology.horizon,
            indexer: &self.stage_data.indexer,
            state: &self.stage_data.state,
            study_dims: &self.stage_data.study_dims,
            inflow_method: &self.methodology.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: sim.inflow_scheme,
            load_scheme: sim.load_scheme,
            ncs_scheme: sim.ncs_scheme,
            stages: &self.stage_data.stages,
            historical_library,
            external_inflow_library,
            external_load_library,
            external_ncs_library,
            recent_accum_seed: &self.recent_observation_seed.accum_seed,
            recent_weight_seed: self.recent_observation_seed.weight_seed,
            // When the dynamic cut-selection method is configured, simulation
            // solves each stage lazily against the cut pool (`Some` only for the
            // dynamic variant); otherwise it uses the baked all-cuts path.
            dcs: self
                .cut_management
                .cut_selection
                .as_ref()
                .and_then(crate::dcs::DcsParams::from_strategy),
            // The backward `noise_key` diagnostic does not apply to simulation.
            noise_key_diag: None,
        }
    }
}
