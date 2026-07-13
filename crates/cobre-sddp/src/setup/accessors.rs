//! Accessor methods and context builders for [`StudySetup`].

use cobre_core::System;
use cobre_core::commissioning::commissioning_active;
use cobre_core::scenario::SamplingScheme;
use cobre_io::EntitySlot;

use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    energy_conversion::EnergyConversionSet,
    indexer::StateLayout,
    simulation::SimulationConfig,
    workspace::CapturedBasis,
};

use super::StudySetup;
use crate::dcs::DcsParams;
use crate::policy_export::build_stage_entity_manifest;

impl StudySetup {
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
    /// [`build_basis_cache_from_checkpoint`](crate::build_basis_cache_from_checkpoint)).
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

    /// Return the pre-computed [`EnergyConversionSet`] for this study.
    #[must_use]
    pub fn energy_conversion(&self) -> &EnergyConversionSet {
        &self.energy_conversion
    }

    /// Return a reference to the simulation configuration.
    #[must_use]
    pub fn simulation_config(&self) -> &SimulationConfig {
        &self.simulation_config
    }

    /// Return a reference to the stage-invariant role-(a) state-vector layout.
    ///
    /// State-region offsets are pure functions of `(N, L, A, k_max)`, so the
    /// single layout resolves onto the correct column at every stage regardless
    /// of per-stage block counts.
    #[must_use]
    pub fn stage_state(&self) -> &StateLayout {
        &self.stage_data.state
    }

    /// Build the per-slot entity-identity manifest for the terminal stage's cut
    /// pool — the stage a boundary policy injects into.
    ///
    /// Delegates to
    /// [`build_stage_entity_manifest`](crate::policy_export::build_stage_entity_manifest),
    /// the single owner of identity resolution shared with the checkpoint writer,
    /// against the terminal stage's projection. The caller passes the result to
    /// [`load_boundary_cuts`](crate::load_boundary_cuts) so a boundary cut whose
    /// slot identity diverges from the current study is rejected rather than
    /// silently mis-loaded.
    ///
    /// `system` is passed explicitly because [`StudySetup`] does not own it.
    #[must_use]
    pub fn build_terminal_entity_manifest(&self, system: &System) -> Vec<EntitySlot> {
        let terminal_idx = self.stage_data.cut_state_layouts.len() - 1;
        build_stage_entity_manifest(
            system,
            &self.stage_data.state,
            &self.stage_data.cut_state_layouts[terminal_idx],
            self.study_stage_ids[terminal_idx],
        )
    }

    /// Number of stages in the planning horizon.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.methodology.horizon.num_stages()
    }

    /// Per-stage stochastic-NCS dormancy mask, reconstructed for the out-of-crate
    /// `tests/` harness (the dormancy mask is not stored — the patch path applies
    /// the commissioning predicate inline to `ncs_stochastic_windows`).
    ///
    /// Outer index is the study stage; inner is the stochastic slot (id-sorted
    /// `StochasticContext::ncs_entity_ids` order). `true` marks a
    /// commissioning-dormant slot whose dense NCS column is zeroed at that stage.
    #[must_use]
    pub fn ncs_stochastic_dormant_for_test(&self) -> Vec<Vec<bool>> {
        self.stage_data
            .stages
            .iter()
            .map(|stage| {
                self.ncs_stochastic_windows
                    .iter()
                    .map(|&(entry, exit)| !commissioning_active(entry, exit, stage.id))
                    .collect()
            })
            .collect()
    }

    /// Construct a [`StageContext`] borrowing from this setup.
    #[must_use]
    pub fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.stage_data.stage_templates.templates,
            base_rows: &self.stage_data.stage_templates.base_rows,
            geometry_per_stage: &self.stage_data.stage_templates.geometry_per_stage,
            noise_scale: &self.stage_data.stage_templates.noise_scale,
            n_hydros: self.stage_data.stage_templates.n_hydros,
            n_load_buses: self.stage_data.stage_templates.n_load_buses,
            load_balance_row_starts: &self.stage_data.stage_templates.load_balance_row_starts,
            load_bus_indices: &self.stage_data.stage_templates.load_bus_indices,
            block_counts_per_stage: &self.stage_data.block_counts_per_stage,
            ncs_col_starts: &self.stage_data.stage_templates.ncs_col_starts,
            n_ncs: self.stage_data.stage_templates.n_ncs,
            ncs_stochastic_dense_col: &self.ncs_stochastic_dense_col,
            ncs_stochastic_windows: &self.ncs_stochastic_windows,
            anticipated_windows: &self.anticipated_windows,
            study_stage_ids: &self.study_stage_ids,
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
            state: &self.stage_data.state,
            cut_state_layouts: &self.stage_data.cut_state_layouts,
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
                .and_then(DcsParams::from_strategy),
        }
    }

    /// Build simulation [`TrainingContext`] with simulation-specific schemes and libraries.
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
            state: &self.stage_data.state,
            // Simulation renders stored cuts into frozen templates and the DCS LP
            // (it does not extract), so the per-pool projection threads through here.
            cut_state_layouts: &self.stage_data.cut_state_layouts,
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
            dcs: self
                .cut_management
                .cut_selection
                .as_ref()
                .and_then(DcsParams::from_strategy),
        }
    }
}
