//! Study setup struct that owns all precomputed state for a solve run.
//!
//! [`StudySetup`] centralises orchestration from CLI/Python entry points, built
//! from a validated [`System`] and [`cobre_io::Config`].
//!
//! **Ownership**: `StudySetup` owns all data; callers borrow for `TrainingContext`
//! and `StageContext` construction. The [`StochasticContext`] lifetime matches setup.
//!
//! **Not included**: MPI communication (in CLI/Python), solver instances (caller-created),
//! progress bars, event channels (caller-managed).
//!
//! ## Example
//!
//! ```rust,no_run
//! use cobre_sddp::setup::StudySetup;
//! use cobre_sddp::hydro_models::PrepareHydroModelsResult;
//! use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
//!
//! # fn example(system: &cobre_core::System, config: &cobre_io::Config)
//! #     -> Result<(), cobre_sddp::SddpError> {
//! let stochastic = build_stochastic_context(system, 42, None, &[], &[], OpeningTreeInputs::default(), ClassSchemes { inflow: None, load: None, ncs: None })?;
//! let hydro_models = PrepareHydroModelsResult::default_from_system(system);
//! let setup = StudySetup::new(system, config, stochastic, hydro_models)?;
//! assert!(!setup.stage_data.stage_templates.templates.is_empty());
//! # Ok(())
//! # }
//! ```

mod accessors;
pub(crate) mod bucket_seed;
pub(crate) mod bucket_topology;
pub(crate) mod methodology_config;
mod orchestration;
pub mod params;
pub(crate) mod scenario_libraries;
pub mod scenario_library_set;
pub mod stage_data;
pub mod stochastic_pipeline;
pub(crate) mod template_postprocess;

pub use params::{
    ConstructionConfig, DEFAULT_FORWARD_PASSES, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, StudyParams,
};
pub use scenario_library_set::{PhaseLibraries, ScenarioLibraries};
pub use stage_data::StageData;
pub use stochastic_pipeline::{
    PrepareStochasticResult, build_ncs_factor_entries, load_load_factors_for_stochastic,
    prepare_stochastic,
};

use std::path::Path;

use cobre_core::{
    EntityId, Stage, System,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::build_hydro_reference_volumes_resolved;
use cobre_stochastic::{
    ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext, SweepDirection,
};

use crate::{
    config::{CutManagementConfig, EventParams},
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversionSet, build_energy_conversion_set},
    error::SddpError,
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    indexer::{CutStateProjection, StateLayout, StudyDimensions},
    lp_builder::build_stage_templates,
    risk_measure::RiskMeasure,
    simulation::EntityCounts,
    stopping_rule::{StoppingRule, StoppingRuleSet},
    workspace::CapturedBasis,
};

// ---------------------------------------------------------------------------
// StudySetup
// ---------------------------------------------------------------------------

/// All precomputed study state built once before training and simulation.
///
/// Constructed by [`StudySetup::new`] from a validated [`System`] and
/// [`cobre_io::Config`]. Owns all data so it can be held across async
/// boundaries (e.g., Python GIL release) without lifetime issues.
///
/// Callers build `TrainingContext` and `StageContext` by borrowing
/// from `StudySetup`.
///
/// Commissioning windows (NCS, anticipated) are carried as per-slot
/// `(entry, exit)` pairs rather than per-stage activity masks, so the per-stage
/// patch sites compute dormancy inline and activity stays out of per-stage
/// storage.
#[derive(Debug)]
pub struct StudySetup {
    /// Stage-indexed data: LP templates, indexer, stages, entity counts, blocks,
    /// lag transitions, noise groups, and scaling report.
    pub stage_data: stage_data::StageData,

    /// Stochastic context holding sampling distributions, libraries, and provenance.
    pub stochastic: StochasticContext,
    /// Future cost function (cut pool) updated by the backward pass during training.
    pub fcf: FutureCostFunction,
    pub(crate) initial_state: Vec<f64>,

    /// Pre-computed hydro production models (FPHA, turbine curves, etc.).
    pub hydro_models: PrepareHydroModelsResult,
    pub(crate) ncs_entity_ids_per_stage: Vec<Vec<i32>>,
    /// Stage-invariant stochastic-slot → dense NCS column index map (slot in
    /// `StochasticContext::ncs_entity_ids` id-sorted order).
    ///
    /// The NCS bound patch sites stride the per-opening cap onto
    /// `ncs_col_starts[s] + ncs_stochastic_dense_col[slot] * n_blks_s + blk`.
    /// Length equals `n_stochastic_ncs`; empty when the study has no stochastic NCS.
    pub(crate) ncs_stochastic_dense_col: Vec<usize>,
    /// Stage-invariant `(entry_stage_id, exit_stage_id)` per stochastic NCS slot
    /// (id-sorted to match `ncs_stochastic_dense_col` and the `transform_ncs_noise`
    /// buffer order).
    ///
    /// The dormant-slot `[0, 0]` cap MUST stay identical across the forward,
    /// backward, and lower-bound patch sites — the `evaluate_lower_bound`
    /// "patch NCS per opening" contract; a divergence understates the bound (D15).
    /// Length equals `n_stochastic_ncs`; empty when no stochastic NCS.
    pub(crate) ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)>,
    /// Max generation \[MW\] per stochastic NCS entity, sorted by entity ID.
    pub(crate) ncs_max_gen: Vec<f64>,
    /// Whether each stochastic NCS entity may be curtailed, aligned 1:1 with
    /// [`Self::ncs_max_gen`]. `false` = must-run: the patch sites pin
    /// `col_lower = col_upper` (not `[0, cap]`), and non-simulated must-run
    /// generation is pre-netted from load.
    pub(crate) ncs_allow_curtailment: Vec<bool>,

    /// Stage-invariant `(entry_stage_id, exit_stage_id)` per anticipated thermal,
    /// in anticipated-local order matching
    /// `stage_data.study_dims.anticipated_thermal_indices`.
    ///
    /// Threaded into the simulation
    /// [`StageExtractionSpec`](crate::simulation::extraction::StageExtractionSpec)
    /// so the anticipated-decision read gates on the same
    /// `StateLayout::is_anticipated_decision_active` predicate the LP builder used,
    /// keying its operation-window clause on the DELIVERY stage's `stage.id`. Empty
    /// when there are no anticipated thermals.
    pub(crate) anticipated_windows: Vec<(Option<i32>, Option<i32>)>,

    /// `study_stage_ids[t] = stage.id` per study stage index; the simulation
    /// context borrows it to map a delivery stage index to its commissioning id
    /// for the `anticipated_windows` gate.
    pub(crate) study_stage_ids: Vec<i32>,

    /// Sampling schemes and pre-built libraries for training and simulation phases.
    pub scenario_libraries: ScenarioLibraries,
    /// Iteration-loop parameters projected from [`crate::config::LoopConfig`].
    ///
    /// `n_fwd_threads` is excluded (derived at runtime) and supplied as a per-call
    /// argument to [`StudySetup::train`].
    pub loop_params: crate::config::LoopParams,

    /// Simulation pipeline parameters, stored directly as [`crate::simulation::SimulationConfig`].
    pub simulation_config: crate::simulation::SimulationConfig,

    /// Relative path to the policy output directory (e.g. `"training/policy"`).
    pub policy_path: String,

    /// Two-stage cut management pipeline configuration.
    pub(crate) cut_management: CutManagementConfig,

    /// Pure-data event flags (output-side).
    ///
    /// Runtime handles (`event_sender`, `shutdown_flag`) and deferred fields
    /// (`checkpoint_interval`) are excluded and supplied per-call in
    /// [`StudySetup::train`].
    pub(crate) events: EventParams,

    /// Stochastic numerical methodology parameters (`horizon`, `inflow_method`).
    pub(crate) methodology: methodology_config::MethodologyConfig,

    /// Lag accumulator seed from `initial_conditions.recent_observations`, applied
    /// at every trajectory start in the forward pass and simulation pipeline instead
    /// of zero-filling. All-zero (a plain zero reset) when `recent_observations` is
    /// empty.
    pub(crate) recent_observation_seed: crate::lag_transition::RecentObservationSeed,

    /// PAR order of the downstream (coarser) resolution model. Non-zero only when
    /// the study includes stages with `season_id >= 12` (a monthly-to-quarterly
    /// transition); zero for uniform-resolution studies. Sizes the downstream
    /// scratch buffers via `WorkspaceSizing`.
    pub(crate) downstream_par_order: usize,

    /// Energy-conversion scalars (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_acum`) per
    /// `(hydro, stage)`, consumed by the energy-balance LP constraints and
    /// inflow-/stored-energy extraction.
    pub(crate) energy_conversion: EnergyConversionSet,

    /// `V_min` (`min_storage_hm3`) per hydro, in declaration order; threaded into
    /// the simulation pipeline for stored-energy calculations.
    pub(crate) hydro_min_storage_hm3: Vec<f64>,

    /// Water travel-time in-transit bucket topology: canonical column order,
    /// global bucket count, and per-stage reachability mask. Empty
    /// (`b_total == 0`) when the system declares no travel-time arc.
    // Voice 4: no read site consumes this yet — the state layout reads
    // `b_total`/`column_order` to size and order the bucket block. The
    // `#[allow(dead_code)]` refires once that reader lands.
    #[allow(dead_code)]
    pub(crate) bucket_topology: bucket_topology::BucketTopology,

    /// Per-stage warm-start basis cache for warm-start / resume training.
    ///
    /// Populated by the CLI / Python paths via
    /// [`StudySetup::set_warm_start_basis_cache`] from the checkpoint's stored
    /// solver bases; [`StudySetup::train`] seeds it into the session's
    /// [`BasisStore`](crate::workspace::BasisStore) so iteration 1's LPs warm-start.
    /// `None` for a fresh start, leaving fresh-mode behavior untouched.
    pub(crate) warm_start_basis_cache: Option<Vec<Option<CapturedBasis>>>,
}

impl StudySetup {
    /// Build all precomputed study state from a validated system and config.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] — if `build_stage_templates` succeeds but
    ///   the template list is empty ("system has no study stages").
    /// - [`SddpError::Solver`] — propagated from `build_stage_templates`
    ///   on LP construction failure.
    /// - [`SddpError::Validation`] — if `parse_cut_selection_config` returns
    ///   an invalid config string.
    pub fn new(
        system: &System,
        config: &cobre_io::Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,
    ) -> Result<Self, SddpError> {
        let params = StudyParams::from_config(config)?;
        // Sentinel: the scenario-source resolvers use the path only for error
        // messages and the historical-years look-up, neither exercised here with a
        // validated Config.
        let sentinel_path = Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        let simulation_source = config
            .simulation_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        let config = params.into_construction_config();
        Self::from_broadcast_params(
            system,
            stochastic,
            config,
            hydro_models,
            &training_source,
            &simulation_source,
        )
    }

    /// Build all precomputed study state from pre-resolved broadcast parameters.
    ///
    /// This constructor accepts the scalar fields already extracted from either a
    /// [`cobre_io::Config`] (on rank 0) or a broadcast config struct (on non-root
    /// ranks), performing the expensive computation steps that cannot be serialised.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] — if `build_stage_templates` succeeds but
    ///   the template list is empty ("system has no study stages").
    /// - [`SddpError::Solver`] — propagated from `build_stage_templates` on LP
    ///   construction failure.
    // Rationale (too_many_lines): a single linear pass building the `StudySetup`
    // literal from per-entity prep blocks; splitting it would scatter the
    // construction the literal reads.
    #[allow(clippy::too_many_lines)]
    pub fn from_broadcast_params(
        system: &System,
        mut stochastic: StochasticContext,
        config: ConstructionConfig,
        hydro_models: PrepareHydroModelsResult,
        training_source: &ScenarioSource,
        simulation_source: &ScenarioSource,
    ) -> Result<Self, SddpError> {
        let ConstructionConfig {
            seed,
            forward_passes,
            stopping_rule_set,
            n_scenarios,
            io_channel_capacity,
            policy_path,
            inflow_method,
            cut_selection,
            cut_activity_tolerance,
            budget,
            export_states,
            scalar_parameters,
        } = config;

        // Keys are a pure function of the synced tree + fixed σ, so every rank
        // computes the identical permutation and cuts stay bit-identical across
        // thread/rank counts (canonical-ω aggregation is order-independent).
        let solve_order_keys =
            crate::stochastic::noise_key::build_noise_key_table(system, &stochastic)?;
        stochastic
            .set_solve_order(&solve_order_keys, SweepDirection::Descending)
            .map_err(|e| SddpError::Validation(e.to_string()))?;

        let EnergyAndTemplates {
            energy_conversion,
            stage_templates,
            scaling_report,
        } = build_energy_and_templates(
            system,
            inflow_method,
            &stochastic,
            &hydro_models,
            &scalar_parameters,
        )?;

        // Computed here (not inside `build_wired_indexer`) so the one
        // `BucketTopology` this constructor derives from `system` also seeds the
        // `StudySetup.bucket_topology` field below, with no second call.
        let bucket_topology = bucket_topology::build_bucket_topology(system);

        let (state_layout, study_dims) = build_wired_indexer(
            system,
            &stage_templates,
            inflow_method,
            &stochastic,
            &bucket_topology,
        );

        let mut initial_state = build_initial_state(system, &study_dims, &state_layout);
        splice_bucket_seed(&mut initial_state, &state_layout, system, &bucket_topology);

        let n_stages = stage_templates.templates.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);

        let cut_state_layouts = build_cut_state_layouts(system, &state_layout, n_stages);
        let pool_state_dimensions: Vec<usize> = cut_state_layouts
            .iter()
            .map(CutStateProjection::n_state)
            .collect();
        let fcf = FutureCostFunction::new_per_stage(
            &pool_state_dimensions,
            state_layout.n_state,
            forward_passes,
            fcf_capacity_iterations,
            &vec![0; n_stages],
        );

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        // Rejects a degenerate single-stage problem (`num_stages < 2`, no
        // predecessor to generate cuts for); reachable since the empty case rejected
        // above still leaves `n_stages == 1` possible.
        horizon.validate()?;

        let risk_measures = build_risk_measures(system);

        let NcsEntityData {
            entity_counts,
            ncs_entity_ids_per_stage,
            ncs_stochastic_dense_col,
            ncs_stochastic_windows,
            ncs_max_gen,
            ncs_allow_curtailment,
        } = build_ncs_entity_data(system, &stage_templates, &stochastic)?;
        let pumping_consumption_mw_per_m3s = build_pumping_consumption(system);
        let contract_prices_per_stage = build_contract_prices_per_stage(system, n_stages);
        let contract_is_import = build_contract_is_import(system);

        let block_counts_per_stage: Vec<usize> = stage_templates
            .block_hours_per_stage
            .iter()
            .map(Vec::len)
            .collect();
        let max_blocks = block_counts_per_stage.iter().copied().max().unwrap_or(0);

        let stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let study_stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();
        let anticipated_windows = build_anticipated_windows(system);

        let LagData {
            stage_lag_transitions,
            noise_group_ids,
            recent_observation_seed,
            downstream_par_order,
        } = precompute_lag_data(system, &stages, &stochastic);

        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

        let scenario_libraries = build_scenario_libraries(
            system,
            &stages,
            &hydro_ids,
            &stochastic,
            &stage_lag_transitions,
            training_source,
            simulation_source,
            forward_passes,
        )?;

        let hydro_min_storage_hm3: Vec<f64> =
            system.hydros().iter().map(|h| h.min_storage_hm3).collect();

        Ok(Self {
            stage_data: stage_data::StageData {
                stage_templates,
                state: state_layout,
                study_dims,
                cut_state_layouts,
                stages,
                entity_counts,
                pumping_consumption_mw_per_m3s,
                contract_prices_per_stage,
                contract_is_import,
                block_counts_per_stage,
                stage_lag_transitions,
                noise_group_ids,
                scaling_report,
            },
            stochastic,
            fcf,
            initial_state,
            hydro_models,
            ncs_entity_ids_per_stage,
            ncs_stochastic_dense_col,
            ncs_stochastic_windows,
            ncs_max_gen,
            ncs_allow_curtailment,
            anticipated_windows,
            study_stage_ids,
            scenario_libraries,
            loop_params: crate::config::LoopParams {
                seed,
                forward_passes,
                max_iterations,
                start_iteration: 0,
                max_blocks,
                stopping_rules: stopping_rule_set,
            },
            simulation_config: crate::simulation::SimulationConfig {
                n_scenarios,
                io_channel_capacity,
            },
            policy_path,
            cut_management: CutManagementConfig {
                cut_selection,
                budget,
                cut_activity_tolerance,
                warm_start_cuts: 0,
                risk_measures,
            },
            events: EventParams { export_states },
            methodology: methodology_config::MethodologyConfig {
                horizon,
                inflow_method,
            },
            recent_observation_seed,
            downstream_par_order,
            energy_conversion,
            hydro_min_storage_hm3,
            bucket_topology,
            warm_start_basis_cache: None,
        })
    }
}

// ---------------------------------------------------------------------------
// from_broadcast_params sub-phase helpers
// ---------------------------------------------------------------------------

/// Grouped output of [`build_ncs_entity_data`].
struct NcsEntityData {
    entity_counts: EntityCounts,
    ncs_entity_ids_per_stage: Vec<Vec<i32>>,
    ncs_stochastic_dense_col: Vec<usize>,
    ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)>,
    ncs_max_gen: Vec<f64>,
    ncs_allow_curtailment: Vec<bool>,
}

/// Build entity counts and the dense NCS column/window maps from the system.
///
/// `ncs_stochastic_dense_col`, `ncs_stochastic_windows`, `ncs_max_gen`, and
/// `ncs_allow_curtailment` are aligned 1:1 in stochastic NCS-entity (slot) order;
/// see [`StudySetup::ncs_stochastic_dense_col`] and
/// [`StudySetup::ncs_stochastic_windows`] for what each carries.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a stochastic NCS entity has no match
/// in the system's `non_controllable_sources`.
fn build_ncs_entity_data(
    system: &System,
    stage_templates: &crate::lp_builder::StageTemplates,
    stochastic: &StochasticContext,
) -> Result<NcsEntityData, SddpError> {
    let entity_counts = build_entity_counts(system);

    let n_study = stage_templates.templates.len();

    // Every stage repeats the full id-sorted NCS list, so a dormant NCS still
    // occupies its slot and reports a zero row rather than being absent.
    let ncs_entity_ids_per_stage: Vec<Vec<i32>> =
        vec![entity_counts.non_controllable_ids.clone(); n_study];

    let stoch_ncs_ids = stochastic.ncs_entity_ids();

    // Bridge each slot to its dense column via entity id (not a direct index) so the
    // map stays correct when only a subset of NCS are stochastic or the orders
    // diverge. Keyed on the id-sorted slot order, not entity declaration order.
    let mut ncs_stochastic_dense_col: Vec<usize> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> =
        Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_max_gen: Vec<f64> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_allow_curtailment: Vec<bool> = Vec::with_capacity(stoch_ncs_ids.len());
    for slot_id in stoch_ncs_ids {
        let dense_col = entity_counts
            .non_controllable_ids
            .iter()
            .position(|&id| id == slot_id.0)
            .ok_or_else(|| {
                SddpError::Validation(format!(
                    "stochastic NCS entity {slot_id:?} not found in system non_controllable_sources"
                ))
            })?;
        let ncs = system
            .non_controllable_sources()
            .iter()
            .find(|n| n.id == *slot_id)
            .ok_or_else(|| {
                SddpError::Validation(format!(
                    "stochastic NCS entity {slot_id:?} not found in system non_controllable_sources"
                ))
            })?;
        ncs_stochastic_dense_col.push(dense_col);
        ncs_stochastic_windows.push((ncs.entry_stage_id, ncs.exit_stage_id));
        ncs_max_gen.push(ncs.max_generation_mw);
        ncs_allow_curtailment.push(ncs.allow_curtailment);
    }

    Ok(NcsEntityData {
        entity_counts,
        ncs_entity_ids_per_stage,
        ncs_stochastic_dense_col,
        ncs_stochastic_windows,
        ncs_max_gen,
        ncs_allow_curtailment,
    })
}

/// Grouped output of [`build_energy_and_templates`].
struct EnergyAndTemplates {
    energy_conversion: EnergyConversionSet,
    stage_templates: crate::lp_builder::StageTemplates,
    scaling_report: crate::scaling_report::ScalingReport,
}

/// Build the energy-conversion set, the resolved parameter table, and the
/// post-processed stage LP templates.
///
/// The energy-conversion set and resolved parameter table are built before the
/// LP templates so the builder can resolve `CoefficientRef::Parameter` values.
/// The resolved parameter table is consumed only by `build_stage_templates`, so
/// it is not returned. The stage-to-season mapping uses `season_id.unwrap_or(0)`
/// so stages without a season collapse to season 0, consistent with every other
/// season-indexed lookup.
///
/// # Errors
///
/// - [`SddpError::Validation`] — on energy-conversion / resolved-parameter
///   construction failure, or when the post-processed template list is empty.
/// - [`SddpError::Solver`] — propagated from `build_stage_templates`.
fn build_energy_and_templates(
    system: &System,
    inflow_method: crate::InflowNonNegativityMethod,
    stochastic: &StochasticContext,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[cobre_core::ScalarParameter],
) -> Result<EnergyAndTemplates, SddpError> {
    let n_stages_pre = system.stages().iter().filter(|s| s.id >= 0).count();
    let stage_to_season: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| i32::try_from(s.season_id.unwrap_or(0)).unwrap_or(0))
        .collect();
    // Single source of truth for `reference_volume_hm3`, identical to the source the
    // FPHA backwater path uses, so the productivity reference and the backwater level
    // never drift.
    let reference_volume_fractions =
        build_hydro_reference_volumes_resolved(&hydro_models.reference_volumes_hm3, 0.0);
    let energy_conversion = build_energy_conversion_set(
        system.hydros(),
        n_stages_pre,
        system.cascade(),
        &reference_volume_fractions,
        // Feeds the FPHA ρ_eq derivation only for plants with no parquet override
        // (the override still wins when present). Per-rank, never broadcast, so every
        // rank sees the same map.
        &hydro_models.vha_geometry_by_hydro,
        Some(&hydro_models.productivity_override),
        Some(&hydro_models.production),
    )
    .map_err(|e| SddpError::Validation(e.to_string()))?;
    let resolved_parameters = crate::resolved_parameters::build_resolved_parameters(
        scalar_parameters,
        &energy_conversion,
        &hydro_models.productivity_override,
        system.hydros(),
        &stage_to_season,
        n_stages_pre,
    )
    .map_err(|e| SddpError::Validation(e.to_string()))?;

    let mut stage_templates = build_stage_templates(
        system,
        inflow_method,
        stochastic.par(),
        stochastic.normal(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &resolved_parameters,
    )?;

    let scaling_report = template_postprocess::postprocess_templates(&mut stage_templates, system);

    if stage_templates.templates.is_empty() {
        return Err(SddpError::Validation(
            "system has no study stages".to_string(),
        ));
    }

    Ok(EnergyAndTemplates {
        energy_conversion,
        stage_templates,
        scaling_report,
    })
}

/// Build the canonical [`StateLayout`] and the [`StudyDimensions`] from the
/// (representative) stage-0 LP layout, the PAR effective lag counts, and the
/// bucket topology.
fn build_wired_indexer(
    system: &System,
    stage_templates: &crate::lp_builder::StageTemplates,
    inflow_method: crate::InflowNonNegativityMethod,
    stochastic: &StochasticContext,
    bucket_topology: &bucket_topology::BucketTopology,
) -> (StateLayout, StudyDimensions) {
    let stage_templates_ref = &stage_templates.templates;
    let has_inflow_penalty =
        inflow_method.has_slack_columns() && stage_templates_ref[0].n_hydro > 0;

    let max_deficit_segments = system
        .buses()
        .iter()
        .map(|b| b.deficit_segments.len())
        .max()
        .unwrap_or(0);

    let mut anticipated_thermal_indices: Vec<usize> = Vec::new();
    let mut anticipated_lead_stages: Vec<usize> = Vec::new();
    for (t_idx, thermal) in system.thermals().iter().enumerate() {
        if let Some(cfg) = thermal.anticipated_config.as_ref() {
            anticipated_thermal_indices.push(t_idx);
            anticipated_lead_stages.push(usize::try_from(cfg.lead_stages).unwrap_or(usize::MAX));
        }
    }
    let n_anticipated = anticipated_thermal_indices.len();
    let k_max: usize = anticipated_lead_stages.iter().copied().max().unwrap_or(0);
    let hydro_count = stage_templates_ref[0].n_hydro;
    let max_par_order = stage_templates_ref[0].max_par_order;

    // Single owner of the study-invariant, non-state LP shape. `has_ncs` only flags
    // presence; the per-(ncs, block) column base is read per stage from
    // `StageContext::ncs_col_starts`, never a global handle. `n_blks` is deliberately
    // absent — it is per-stage, owned by the per-stage geometry, never study-global.
    let study_dims = crate::indexer::StudyDimensions {
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: system.buses().len(),
        max_deficit_segments,
        has_ncs: !stage_templates.ncs_col_starts.is_empty(),
        has_inflow_penalty,
        has_withdrawal: hydro_count > 0,
        has_operational_violations: hydro_count != 0,
        anticipated_thermal_indices,
        n_pumping: system.n_pumping_stations(),
    };

    // Per-hydro lag-state-slot count for the cut sparse mask: `max_par_order` (the
    // widened psi stride) when PAR(p)-A annual is active, else the classical AR
    // order. `par.order(h)` here would silently truncate the cut row's coefficients
    // on the annual-`ψ̂/12` lag slots and produce over-estimating cuts.
    let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
        let par = stochastic.par();
        (0..par.n_hydros())
            .map(|h| par.effective_lag_count(h))
            .collect()
    } else {
        vec![0; hydro_count]
    };

    // `StateLayout` is the sole role-(a) owner; its constructor finalizes the
    // nonzero mask unconditionally, so every study (storage-only or pure-thermal)
    // has a finalized mask for the single-path mask-driven cut-row loop.
    let state = StateLayout::new(
        hydro_count,
        max_par_order,
        bucket_topology.b_total,
        bucket_topology.column_order.clone(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_counts,
    );

    (state, study_dims)
}

/// Build the per-pool [`CutStateProjection`], one per stage (pool) `t`, projecting
/// the global [`StateLayout`] onto the cut-state dimensions each pool carries.
///
/// Pool `t` is sized by `stages[t + 1].state_config` — the cost-to-go this
/// stage's **successor** generates for it (pool `t` is populated by the backward
/// pass when it solves stage `t + 1`'s LP and reads stage `t + 1`'s
/// incoming-state reduced costs). Sizing pool `t` from `stages[t].state_config`
/// is the off-by-one that compiles but stores cuts at the wrong dimension.
/// Stage 0's config never sizes a pool (it has no predecessor pool).
///
/// The terminal pool `n_stages - 1` has no successor stage, so the `t + 1` rule
/// does not apply: it is sized by the **full global `n_state`**. With
/// `config.policy.boundary` set, the injected boundary cuts come from the
/// external study and are validated and rebuilt against `fcf.state_dimension`
/// (the global `n_state`) by `load_boundary_cuts` / `inject_boundary_cuts`, so
/// the global dimension is exactly the size injection requires — never a DECOMP
/// stage's reduced config. (Per-slot identity reconciliation between a
/// differently-scoped boundary manifest and the local layout is out of scope
/// here.)
fn build_cut_state_layouts(
    system: &cobre_core::System,
    state_layout: &StateLayout,
    n_stages: usize,
) -> Vec<CutStateProjection> {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    (0..n_stages)
        .map(|t| {
            if t + 1 < n_stages {
                CutStateProjection::new(state_layout, study_stages[t + 1].state_config)
            } else {
                CutStateProjection::new(state_layout, FULL_STATE_CONFIG)
            }
        })
        .collect()
}

/// The all-dimensions cut-state config, sizing a pool to the full global
/// `n_state`. Used for the terminal pool (no successor stage to govern it).
const FULL_STATE_CONFIG: cobre_core::temporal::StageStateConfig =
    cobre_core::temporal::StageStateConfig {
        storage: true,
        inflow_lags: true,
    };

/// Grouped output of [`precompute_lag_data`].
struct LagData {
    stage_lag_transitions: Vec<cobre_core::temporal::StageLagTransition>,
    noise_group_ids: Vec<u32>,
    recent_observation_seed: crate::lag_transition::RecentObservationSeed,
    downstream_par_order: usize,
}

/// Precompute per-stage lag accumulation weights, noise-group ids, the
/// recent-observation seed, and the downstream PAR order.
fn precompute_lag_data(
    system: &System,
    stages: &[Stage],
    stochastic: &StochasticContext,
) -> LagData {
    let noop_season_map;
    let season_map_ref = if let Some(sm) = system.policy_graph().season_map.as_ref() {
        sm
    } else {
        // No season map: all stages produce zero-weight no-op transitions.
        noop_season_map = cobre_core::temporal::SeasonMap {
            cycle_type: cobre_core::temporal::SeasonCycleType::Monthly,
            seasons: Vec::new(),
        };
        &noop_season_map
    };
    // Proxy: the global `max_par_order` stands in for the quarterly PAR order until a
    // separate quarterly stochastic context exists.
    let has_quarterly_stages = stages
        .iter()
        .any(|s| s.season_id.is_some_and(|id| id >= 12));
    let downstream_par_order = if has_quarterly_stages {
        stochastic.par().max_order()
    } else {
        0
    };
    let stage_lag_transitions = crate::lag_transition::precompute_stage_lag_transitions(
        stages,
        season_map_ref,
        downstream_par_order,
    );
    let noise_group_ids = crate::lag_transition::precompute_noise_groups(stages);

    let recent_observation_seed = if stages.is_empty() {
        crate::lag_transition::RecentObservationSeed::zero(system.hydros().len())
    } else {
        crate::lag_transition::compute_recent_observation_seed(
            &system.initial_conditions().recent_observations,
            &stages[0],
            season_map_ref,
            system.hydros(),
        )
    };

    LagData {
        stage_lag_transitions,
        noise_group_ids,
        recent_observation_seed,
        downstream_par_order,
    }
}

/// Build the training and simulation [`ScenarioLibraries`].
///
/// Each phase's per-class library (`historical`, `external_inflow`,
/// `external_load`, `external_ncs`) is constructed only when that class uses
/// the matching sampling scheme. Simulation-specific libraries are built only
/// when the simulation scheme differs from the training scheme; when identical,
/// the simulation phase stores `None` and `simulation_ctx()` falls back to the
/// training library references.
///
/// # Errors
///
/// Propagates [`SddpError`] from the individual library builders on validation
/// or padding failure.
// Rationale: eight disjoint read-only inputs drive one cohesive setup phase; a
// bundle struct would only relocate the arity without improving clarity.
#[allow(clippy::too_many_arguments)]
fn build_scenario_libraries(
    system: &System,
    stages: &[Stage],
    hydro_ids: &[EntityId],
    stochastic: &StochasticContext,
    stage_lag_transitions: &[cobre_core::temporal::StageLagTransition],
    training_source: &ScenarioSource,
    simulation_source: &ScenarioSource,
    forward_passes: u32,
) -> Result<ScenarioLibraries, SddpError> {
    let inflow_scheme = training_source.inflow_scheme;
    let load_scheme = training_source.load_scheme;
    let ncs_scheme = training_source.ncs_scheme;
    let sim_inflow_scheme = simulation_source.inflow_scheme;
    let sim_load_scheme = simulation_source.load_scheme;
    let sim_ncs_scheme = simulation_source.ncs_scheme;

    let training_historical: Option<HistoricalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::Historical {
            Some(scenario_libraries::build_historical_inflow_library(
                system.inflow_history(),
                hydro_ids,
                stages,
                stochastic.par(),
                system.policy_graph().season_map.as_ref(),
                &system.initial_conditions().past_inflows,
                stage_lag_transitions,
                training_source.historical_years.as_ref(),
                forward_passes,
            )?)
        } else {
            None
        };

    let training_external_inflow: Option<ExternalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_inflow_library(
                system.external_scenarios(),
                hydro_ids,
                stages,
                stochastic.par(),
                &system.initial_conditions().past_inflows,
                stage_lag_transitions,
                forward_passes,
            )?)
        } else {
            None
        };

    let training_external_load: Option<ExternalScenarioLibrary> =
        if load_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_load_library(
                system.external_load_scenarios(),
                system.load_models(),
                stages,
                forward_passes,
            )?)
        } else {
            None
        };

    let training_external_ncs: Option<ExternalScenarioLibrary> =
        if ncs_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_ncs_library(
                system.external_ncs_scenarios(),
                system.ncs_models(),
                stages,
                forward_passes,
            )?)
        } else {
            None
        };

    let simulation_historical: Option<HistoricalScenarioLibrary> =
        if sim_inflow_scheme == SamplingScheme::Historical && sim_inflow_scheme != inflow_scheme {
            Some(scenario_libraries::build_historical_inflow_library(
                system.inflow_history(),
                hydro_ids,
                stages,
                stochastic.par(),
                system.policy_graph().season_map.as_ref(),
                &system.initial_conditions().past_inflows,
                stage_lag_transitions,
                simulation_source.historical_years.as_ref(),
                forward_passes,
            )?)
        } else {
            None
        };

    let simulation_external_inflow: Option<ExternalScenarioLibrary> =
        if sim_inflow_scheme == SamplingScheme::External && sim_inflow_scheme != inflow_scheme {
            Some(scenario_libraries::build_external_inflow_library(
                system.external_scenarios(),
                hydro_ids,
                stages,
                stochastic.par(),
                &system.initial_conditions().past_inflows,
                stage_lag_transitions,
                forward_passes,
            )?)
        } else {
            None
        };

    let simulation_external_load: Option<ExternalScenarioLibrary> =
        if sim_load_scheme == SamplingScheme::External && sim_load_scheme != load_scheme {
            Some(scenario_libraries::build_external_load_library(
                system.external_load_scenarios(),
                system.load_models(),
                stages,
                forward_passes,
            )?)
        } else {
            None
        };

    let simulation_external_ncs: Option<ExternalScenarioLibrary> =
        if sim_ncs_scheme == SamplingScheme::External && sim_ncs_scheme != ncs_scheme {
            Some(scenario_libraries::build_external_ncs_library(
                system.external_ncs_scenarios(),
                system.ncs_models(),
                stages,
                forward_passes,
            )?)
        } else {
            None
        };

    Ok(ScenarioLibraries {
        training: PhaseLibraries {
            inflow_scheme,
            load_scheme,
            ncs_scheme,
            historical: training_historical,
            external_inflow: training_external_inflow,
            external_load: training_external_load,
            external_ncs: training_external_ncs,
        },
        simulation: PhaseLibraries {
            inflow_scheme: sim_inflow_scheme,
            load_scheme: sim_load_scheme,
            ncs_scheme: sim_ncs_scheme,
            historical: simulation_historical,
            external_inflow: simulation_external_inflow,
            external_load: simulation_external_load,
            external_ncs: simulation_external_ncs,
        },
    })
}

/// Return the maximum iteration budget from the stopping rule set.
///
/// Used for FCF pre-sizing. If no iteration limit is present, returns
/// [`DEFAULT_MAX_ITERATIONS`].
fn max_iterations_from_rules(rules: &StoppingRuleSet) -> u64 {
    rules
        .rules
        .iter()
        .filter_map(|r| {
            if let StoppingRule::IterationLimit { limit } = r {
                Some(*limit)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
}

/// Build the per-study-stage risk measures from the system's stage risk configs.
///
/// One entry per study stage (`id >= 0`), in stage-index order, matching the
/// `block_counts_per_stage` / template ordering the cut-management pipeline
/// indexes by stage.
fn build_risk_measures(system: &System) -> Vec<RiskMeasure> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| RiskMeasure::from(s.risk_config))
        .collect()
}

/// Build [`EntityCounts`] from the loaded system.
fn build_entity_counts(system: &System) -> EntityCounts {
    EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: vec![0.0; system.hydros().len()],
        thermal_ids: system.thermals().iter().map(|t| t.id.0).collect(),
        line_ids: system.lines().iter().map(|l| l.id.0).collect(),
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: system.pumping_stations().iter().map(|p| p.id.0).collect(),
        contract_ids: system.contracts().iter().map(|c| c.id.0).collect(),
        non_controllable_ids: system
            .non_controllable_sources()
            .iter()
            .map(|n| n.id.0)
            .collect(),
    }
}

/// Build the per-station pumping power-consumption rates \[MW/(m³/s)\].
///
/// ID-sorted parallel to `EntityCounts::pumping_station_ids` (both derive from the
/// canonical ID-ordered `system.pumping_stations()` slice), so a row's position
/// matches its station ID's position in `pumping_station_ids`.
fn build_pumping_consumption(system: &System) -> Vec<f64> {
    system
        .pumping_stations()
        .iter()
        .map(|p| p.consumption_mw_per_m3s)
        .collect()
}

/// Build the per-stage RESOLVED contract prices \[$/`MWh`\].
///
/// Outer index is the study-stage index `t` (0-based, matching the
/// `contract_bounds` stage axis); each inner `Vec` is ID-sorted parallel to
/// `system.contracts()` — the same order `EntityCounts::contract_ids` is built in —
/// carrying `contract_bounds(c, t).price_per_mwh`. Empty inner `Vec`s for a
/// contract-free system.
fn build_contract_prices_per_stage(system: &System, n_stages: usize) -> Vec<Vec<f64>> {
    let bounds = system.bounds();
    let n_contracts = system.contracts().len();
    (0..n_stages)
        .map(|t| {
            (0..n_contracts)
                .map(|c| bounds.contract_bounds(c, t).price_per_mwh)
                .collect()
        })
        .collect()
}

/// Build the per-contract direction flags (`true` = import).
///
/// ID-sorted parallel to `system.contracts()` — the same order
/// `EntityCounts::contract_ids` is built in — so extraction's running per-direction
/// slot count reproduces the LP builder's `fill_contract_columns` slot assignment.
fn build_contract_is_import(system: &System) -> Vec<bool> {
    system
        .contracts()
        .iter()
        .map(|c| c.contract_type == cobre_core::ContractType::Import)
        .collect()
}

/// Build the per-plant commissioning windows for the anticipated thermals.
///
/// In anticipated-local declaration order — the same order
/// `anticipated_thermal_indices` and the LP-builder `anticipated_windows` use, so
/// the simulation decision gate reads the matching window per index. Empty when
/// there are no anticipated thermals.
fn build_anticipated_windows(system: &System) -> Vec<(Option<i32>, Option<i32>)> {
    system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| (t.entry_stage_id, t.exit_stage_id))
        .collect()
}

/// Build the initial state vector from the system's initial conditions.
///
/// Layout `[storage(0..N), lags(N..N*(1+L))]` (N hydros, L = max PAR order),
/// storage in canonical ID order. Lag slots come from
/// `initial_conditions.past_inflows`: `values_m3s[l]` with index 0 = lag 1 (most
/// recent), index L-1 = lag L (oldest). Storage-only when `max_par_order == 0`.
fn build_initial_state(
    system: &System,
    study_dims: &StudyDimensions,
    layout: &StateLayout,
) -> Vec<f64> {
    let mut state = vec![0.0_f64; layout.n_state];
    let hydros = system.hydros();
    let ic = system.initial_conditions();

    for hs in &ic.storage {
        // Both hydros() and ic.storage are sorted by hydro_id.
        if let Ok(idx) = hydros.binary_search_by_key(&hs.hydro_id.0, |h| h.id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    for hs in &ic.filling_storage {
        // ic.filling_storage is sorted by hydro_id (binary_search requires it). The
        // seed writes the same coordinate the PreFilling pin
        // (`fill_prefilling_shortcircuit`) freezes to `[seed, seed]`; do not merge
        // the two collections or re-index the column — a separate index would
        // silently desync from that pin.
        if let Ok(idx) = hydros.binary_search_by_key(&hs.hydro_id.0, |h| h.id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    if layout.max_par_order > 0 {
        let n_h = layout.hydro_count;
        for pi in &ic.past_inflows {
            if let Ok(idx) = hydros.binary_search_by_key(&pi.hydro_id.0, |h| h.id.0) {
                let n_lags = pi.values_m3s.len().min(layout.max_par_order);
                for lag in 0..n_lags {
                    let slot = layout.inflow_lags.start + lag * n_h + idx;
                    state[slot] = pi.values_m3s[lag];
                }
            }
        }
    }

    // Anticipated-state ring buffer, slot-major:
    // `state[anticipated_state.start + slot * n_anticipated + local_idx]`. Padding
    // slots `[K_i, k_max)` must stay zero — the ring-buffer logic in `noise.rs` and
    // `indexer.rs` assumes it.
    if layout.n_anticipated > 0 && layout.k_max > 0 {
        debug_assert_eq!(
            study_dims.anticipated_thermal_indices.len(),
            layout.n_anticipated,
            "anticipated_thermal_indices length must equal n_anticipated",
        );
        let thermals = system.thermals();
        let n_ant = layout.n_anticipated;
        let ant_start = layout.anticipated_state.start;
        for history in &ic.past_anticipated_commitments {
            // thermals() and past_anticipated_commitments are both sorted by
            // thermal_id (binary_search requires it).
            let Ok(global_idx) = thermals.binary_search_by_key(&history.thermal_id.0, |t| t.id.0)
            else {
                // Defense-in-depth — the cobre-io validator rejects an unknown ID in
                // production; matches the existing `past_inflows` skip behavior.
                continue;
            };
            // O(n) over the small `n_anticipated` list, not a map.
            let Some(local_idx) = study_dims
                .anticipated_thermal_indices
                .iter()
                .position(|&g| g == global_idx)
            else {
                // Not an anticipated plant (`anticipated_config: None`) — skip.
                continue;
            };
            // Clamp to K_i, not k_max: over-long input would otherwise corrupt the
            // padding slots.
            let k_i = layout.anticipated_lead_stages[local_idx];
            let n_slots = history.values_mw.len().min(k_i);
            for slot in 0..n_slots {
                let off = ant_start + slot * n_ant + local_idx;
                state[off] = history.values_mw[slot];
            }
            // Padding slots `[K_i, k_max)` must stay 0.0 — a non-zero value corrupts
            // the ring buffer and causes LP infeasibility.
            #[allow(clippy::float_cmp)]
            for slot in k_i..layout.k_max {
                let off = ant_start + slot * n_ant + local_idx;
                debug_assert_eq!(
                    state[off], 0.0,
                    "padding slot must be zero: plant local_idx={local_idx}, slot={slot}, K_i={k_i}, k_max={}",
                    layout.k_max
                );
            }
        }
    }

    state
}

/// Write the travel-time bucket seed into `state`'s declared `buckets_out`
/// slots — the same index space [`StateLayout::state_to_lp_incoming_column`]
/// remaps to the pinned `buckets_in` LP column, so no separate pin wiring is
/// needed beyond this splice.
fn splice_bucket_seed(
    state: &mut [f64],
    layout: &StateLayout,
    system: &System,
    topology: &bucket_topology::BucketTopology,
) {
    let seed = bucket_seed::build_initial_bucket_state(system, topology);
    debug_assert_eq!(seed.len(), layout.b_total);
    for (b, &value) in seed.iter().enumerate() {
        state[layout.buckets_out.start + b] = value;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
