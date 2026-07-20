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

use cobre_core::ContractType::Import;
use cobre_core::temporal::SeasonCycleType::Monthly;
use cobre_core::temporal::SeasonMap;
use cobre_core::temporal::StageLagTransition;
use cobre_core::temporal::StageStateConfig;
use cobre_io::Config;
use cobre_io::config::BackwardScheduler;
use cobre_solver::ActiveProfile;
use cobre_stochastic::par::RecentObservationSeed;
use cobre_stochastic::par::lag_transition::compute_recent_observation_seed;
use cobre_stochastic::par::lag_transition::derive_downstream_par_order;
use cobre_stochastic::par::lag_transition::precompute_noise_groups;
use cobre_stochastic::par::lag_transition::precompute_stage_lag_transitions;

use crate::StageTemplates;
use crate::config::LoopParams;
use crate::resolved_parameters::build_resolved_parameters;
use crate::scaling_report::ScalingReport;
use crate::simulation::SimulationConfig;
use crate::solve::solver_phase::{Phase, validate_phase_solver_config};
use crate::stochastic::noise_key::build_noise_key_table;
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
    prepare_stochastic, study_stage_noise_group_ids,
};

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;

use cobre_core::{
    AnticipatedConfig, EntityId, Hydro, Stage, StageId, System, Thermal,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::build_hydro_reference_volumes_resolved;
use cobre_stochastic::par::precompute::PrecomputedPar;
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
    indexer::{CutStateProjection, StateSpace, StudyDimensions},
    lead_time::{AnticipatedResolution, LeadTime, PointResolution, SpreadResolution},
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
    /// `is_anticipated_decision_active` predicate the LP builder used,
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
    pub loop_params: LoopParams,

    /// Simulation pipeline parameters, stored directly as [`crate::simulation::SimulationConfig`].
    pub simulation_config: SimulationConfig,

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

    /// Resolved backward-pass solver profile (`training.solver.backward`, layered
    /// over the current per-phase constant — see
    /// [`crate::solve::solver_phase::Phase::resolve_profile`]). Threaded into
    /// [`StudySetup::train`].
    pub(crate) backward_profile: ActiveProfile,

    /// Resolved forward-pass solver profile (`training.solver.forward`).
    pub(crate) forward_profile: ActiveProfile,

    /// Backward-pass scheduler (`training.backward_scheduler`), threaded into
    /// [`StudySetup::train`] alongside [`Self::backward_profile`].
    pub(crate) backward_scheduler: BackwardScheduler,

    /// Opening-block size override for `backward_scheduler = opening_block`
    /// (`training.opening_block_size`).
    pub(crate) opening_block_size: Option<NonZeroUsize>,

    /// PN opening-block-scheduler claim-order override, threaded into
    /// [`StudySetup::train`] alongside [`Self::backward_scheduler`]. No
    /// `training.*` config field resolves this yet — a reserved test-support
    /// seam; production always resolves `true` (see
    /// [`crate::solve::solver_phase::SolverProfiles::lpt_claim_order`]).
    pub(crate) lpt_claim_order: bool,

    /// Stochastic numerical methodology parameters (`horizon`, `inflow_method`).
    pub(crate) methodology: methodology_config::MethodologyConfig,

    /// Lag accumulator seed from `initial_conditions.recent_observations`, applied
    /// at every trajectory start in the forward pass and simulation pipeline instead
    /// of zero-filling. All-zero (a plain zero reset) when `recent_observations` is
    /// empty.
    pub(crate) recent_observation_seed: RecentObservationSeed,

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
    /// global bucket count, per-stage reachability mask, and the three
    /// resolved arc tables (stage-clock weights, chronological spread,
    /// arrival density) — the single derivation site for all of them. Empty
    /// (`n_buckets == 0`) when the system declares no travel-time arc.
    // Voice 4: every field is consumed via the constructor's threaded LOCAL
    // (state-layout sizing, the LP builder's arc-table threading, the bucket
    // IC seed) before this STORED field is set below; no post-construction
    // reader exists yet. `#[allow(dead_code)]` refires once one lands.
    #[allow(dead_code)]
    pub(crate) transit_bucket_topology: bucket_topology::TransitBucketTopology,

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
        config: &Config,
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
    /// - [`SddpError::Validation`] — a per-phase solver profile config sets a
    ///   field the compiled backend does not support (see
    ///   `validate_phase_solver_config`).
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
            training_solver_backward,
            training_solver_forward,
            simulation_solver,
            backward_opening_order,
            backward_scheduler,
            opening_block_size,
        } = config;

        // Fail fast on a backend-unsupported field before any template exists;
        // validation runs on every rank (`from_broadcast_params` is the shared
        // setup path), so it is deterministic across the run.
        validate_phase_solver_config(training_solver_backward.as_ref(), Phase::Backward)?;
        validate_phase_solver_config(training_solver_forward.as_ref(), Phase::Forward)?;
        validate_phase_solver_config(simulation_solver.as_ref(), Phase::Simulation)?;

        // `resolve_profile` is a pure function of the (identically broadcast)
        // config, so every rank resolving independently is sufficient — the
        // resolved `ActiveProfile` itself never needs to go on the wire.
        let backward_profile = Phase::Backward.resolve_profile(training_solver_backward.as_ref());
        let forward_profile = Phase::Forward.resolve_profile(training_solver_forward.as_ref());
        let simulation_profile = Phase::Simulation.resolve_profile(simulation_solver.as_ref());

        // Keys are a pure function of the synced tree + fixed σ, so every rank
        // computes the identical permutation and cuts stay bit-identical across
        // thread/rank counts (canonical-ω aggregation is order-independent).
        let solve_order_keys = build_noise_key_table(system, &stochastic, backward_opening_order)?;
        stochastic
            .set_solve_order(&solve_order_keys, SweepDirection::Descending)
            .map_err(|e| SddpError::Validation(e.to_string()))?;

        // Computed here (not inside `build_energy_and_templates`) so the one
        // `TransitBucketTopology` this constructor derives from `system` also seeds the
        // `StudySetup.transit_bucket_topology` field below, with no second call.
        let transit_bucket_topology = bucket_topology::build_transit_bucket_topology(system);

        // Resolved before the LP templates: none of the state dimensions depend on
        // the built LP, and `build_stage_templates` needs the finished `StateSpace`
        // threaded in as a parameter (the single role-(a) owner — see
        // `resolve_state_layout`).
        let (state_layout, hydro_count, anticipated_thermal_indices) =
            resolve_state_layout(system, stochastic.par(), &transit_bucket_topology)?;

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
            &state_layout,
            &transit_bucket_topology.per_stage_mask,
            &transit_bucket_topology.arc_stage_weights,
            &transit_bucket_topology.arc_spread_chrono,
            &transit_bucket_topology.arc_arrival_density,
        )?;

        let study_dims = build_study_dimensions(
            system,
            &stage_templates,
            inflow_method,
            hydro_count,
            anticipated_thermal_indices,
        );

        let mut initial_state = build_initial_state(system, &study_dims, &state_layout);
        splice_transit_bucket_seed(
            &mut initial_state,
            &state_layout,
            system,
            &transit_bucket_topology,
        );

        let n_stages = stage_templates.templates.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);

        let cut_state_layouts = build_cut_state_layouts(system, &state_layout, n_stages);
        let pool_state_dimensions: Vec<usize> = cut_state_layouts
            .iter()
            .map(CutStateProjection::n_slots)
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
            downstream_par_order,
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
            loop_params: LoopParams {
                seed,
                forward_passes,
                max_iterations,
                start_iteration: 0,
                max_blocks,
                stopping_rules: stopping_rule_set,
            },
            simulation_config: SimulationConfig {
                n_scenarios,
                io_channel_capacity,
                profile: simulation_profile,
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
            backward_profile,
            forward_profile,
            backward_scheduler,
            opening_block_size,
            lpt_claim_order: true,
            methodology: methodology_config::MethodologyConfig {
                horizon,
                inflow_method,
            },
            recent_observation_seed,
            downstream_par_order,
            energy_conversion,
            hydro_min_storage_hm3,
            transit_bucket_topology,
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
    stage_templates: &StageTemplates,
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
    stage_templates: StageTemplates,
    scaling_report: ScalingReport,
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
// Rationale (too_many_arguments): each of the three arc-table parameters threads
// the single setup-owned derivation (`build_transit_bucket_topology`) into
// `build_stage_templates`, mirroring the existing `per_stage_mask` thread; a
// wrapper struct used at this one call site would rename the coupling, not
// remove it.
#[allow(clippy::too_many_arguments)]
fn build_energy_and_templates(
    system: &System,
    inflow_method: crate::InflowNonNegativityMethod,
    stochastic: &StochasticContext,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[cobre_core::ScalarParameter],
    state_layout: &StateSpace,
    per_stage_mask: &[Vec<usize>],
    arc_stage_weights: &HashMap<usize, Vec<Vec<f64>>>,
    arc_spread_chrono: &HashMap<usize, Vec<Option<SpreadResolution>>>,
    arc_arrival_density: &HashMap<usize, Vec<Option<Vec<f64>>>>,
) -> Result<EnergyAndTemplates, SddpError> {
    let study_stage_ids: Vec<StageId> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| StageId(s.id))
        .collect();
    let n_stages_pre = study_stage_ids.len();
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
        &study_stage_ids,
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
    let resolved_parameters = build_resolved_parameters(
        scalar_parameters,
        &energy_conversion,
        &hydro_models.productivity_override,
        system.hydros(),
        &stage_to_season,
        &study_stage_ids,
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
        state_layout,
        per_stage_mask,
        arc_stage_weights,
        arc_spread_chrono,
        arc_arrival_density,
    )?;

    let scaling_report =
        template_postprocess::postprocess_templates(&mut stage_templates, system, state_layout);

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

/// Resolve every anticipated thermal's delivery-anchored commitment and
/// construct the single role-(a) [`StateSpace`] — before stage templates
/// exist, since none of the state dimensions depend on the built LP.
///
/// The returned `hydro_count` and `anticipated_thermal_indices` are the exact
/// values the layout was built from; [`build_study_dimensions`] takes them as
/// parameters instead of re-deriving them from the built templates.
///
/// # Errors
///
/// - [`SddpError::Validation`] — a `LeadTime` anticipated plant's resolution
///   fans out (`AnticipatedResolution::max_fanout > 1`); per-delivery-stage
///   fan-out simulation output is not yet supported.
pub(crate) fn resolve_state_layout(
    system: &System,
    par_lp: &PrecomputedPar,
    transit_bucket_topology: &bucket_topology::TransitBucketTopology,
) -> Result<(StateSpace, usize, Vec<usize>), SddpError> {
    let anticipated_thermal_indices: Vec<usize> = system
        .thermals()
        .iter()
        .enumerate()
        .filter_map(|(t_idx, thermal)| thermal.anticipated_config.is_some().then_some(t_idx))
        .collect();
    let n_anticipated = anticipated_thermal_indices.len();

    // Single resolve_point consumer: map each anticipated plant's config to a
    // delivery-anchored PointResolution and derive the constant-lead K_i the
    // still-live ring machinery reads (the resolve_point decider contract). A
    // second resolve_point call site is forbidden — this resolution threads onto
    // the state layout instead.
    let (anticipated_resolution, anticipated_lead_stages) = resolve_anticipated_commitments(system);
    debug_assert_eq!(anticipated_lead_stages.len(), n_anticipated);

    // TODO(anticipated-fanout-output): the coupled output extractor is
    // compute_anticipated_decision_mw
    if anticipated_resolution.max_fanout > 1 {
        let plant_id = first_fanned_plant_id(
            system,
            &anticipated_thermal_indices,
            &anticipated_resolution,
        );
        debug_assert!(
            plant_id.is_some(),
            "max_fanout > 1 must locate the fanning plant"
        );
        return Err(SddpError::Validation(format!(
            "anticipated thermal {}: LeadTime fan-out (a coarse decision stage anchoring \
             several delivery stages) — per-delivery-stage fan-out simulation output is \
             not yet supported",
            plant_id.map_or(EntityId(-1), |id| id)
        )));
    }

    // Ring depth: the delivery-anchored max_t K_i(t), clamped up to the
    // constant-lead machinery's per-plant K_i so its slot indexing stays in range.
    // A LeadStages plant's depth is bounded by ℓ, so this equals the pre-anchor
    // max(lead_stages) and the ring sizing is byte-for-byte unchanged.
    let k_max: usize = anticipated_resolution
        .k_max
        .max(anticipated_lead_stages.iter().copied().max().unwrap_or(0));

    let hydro_count = system.hydros().len();
    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0)
        .max(par_lp.max_order());

    // Per-hydro lag-state-slot count for the cut sparse mask: `max_par_order` (the
    // widened psi stride) when PAR(p)-A annual is active, else the classical AR
    // order. `par.order(h)` here would silently truncate the cut row's coefficients
    // on the annual-`ψ̂/12` lag slots and produce over-estimating cuts. Falls back
    // to the dense `max_par_order` stride for a hydro `par_lp` omits (`h >=
    // par_lp.n_hydros()`) — production's `par_lp` always covers every system
    // hydro, so the fallback is inert there; a hydro-free `PrecomputedPar` test
    // fixture paired with a hydro-bearing system relies on it to satisfy the
    // `StateSpace::new` length contract.
    let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
        (0..hydro_count)
            .map(|h| {
                if h < par_lp.n_hydros() {
                    par_lp.effective_lag_count(h)
                } else {
                    max_par_order
                }
            })
            .collect()
    } else {
        vec![0; hydro_count]
    };

    // `StateSpace` is the sole role-(a) owner; its constructor finalizes the
    // nonzero mask unconditionally, so every study (storage-only or pure-thermal)
    // has a finalized mask for the single-path mask-driven cut-row loop.
    let mut state = StateSpace::new(
        hydro_count,
        max_par_order,
        transit_bucket_topology.n_buckets,
        transit_bucket_topology.column_order.clone(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_counts,
    );
    state.set_anticipated_resolution(anticipated_resolution);

    Ok((state, hydro_count, anticipated_thermal_indices))
}

/// Build the study-invariant, non-state [`StudyDimensions`] from the system
/// and the post-processed stage templates.
///
/// `hydro_count` and `anticipated_thermal_indices` are threaded from
/// [`resolve_state_layout`] — the same values its [`StateSpace`] was built
/// from — so the only per-stage template field this reads is
/// `ncs_col_starts`, the one dimension genuinely derived from the built LP.
fn build_study_dimensions(
    system: &System,
    stage_templates: &StageTemplates,
    inflow_method: crate::InflowNonNegativityMethod,
    hydro_count: usize,
    anticipated_thermal_indices: Vec<usize>,
) -> StudyDimensions {
    let has_inflow_penalty = inflow_method.has_slack_columns() && hydro_count > 0;

    let max_deficit_segments = system
        .buses()
        .iter()
        .map(|b| b.deficit_segments.len())
        .max()
        .unwrap_or(0);

    // Single owner of the study-invariant, non-state LP shape. `has_ncs` only flags
    // presence; the per-(ncs, block) column base is read per stage from
    // `StageContext::ncs_col_starts`, never a global handle. `n_blks` is deliberately
    // absent — it is per-stage, owned by the per-stage geometry, never study-global.
    StudyDimensions {
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
    }
}

/// The first (canonical-order) anticipated plant whose `LeadTime` resolution
/// fans out — `|genuine C(t)| > 1` at some decision stage `t` — or `None` if
/// none does. Shares the exact per-plant/per-stage predicate
/// [`AnticipatedResolution::max_fanout`] maxes over, so `Some(_)` iff
/// `resolution.max_fanout > 1`; `anticipated_thermal_indices` and
/// `resolution.per_plant` are both in canonical (anticipated-local) order, so
/// the first match is declaration-order-invariant.
fn first_fanned_plant_id(
    system: &System,
    anticipated_thermal_indices: &[usize],
    resolution: &AnticipatedResolution,
) -> Option<EntityId> {
    resolution
        .per_plant
        .iter()
        .enumerate()
        .find_map(|(local_idx, point)| {
            let fans_out =
                (0..point.decision_sets.len()).any(|t| point.genuine_decisions_at(t).count() > 1);
            fans_out.then(|| system.thermals()[anticipated_thermal_indices[local_idx]].id)
        })
}

/// Resolve every anticipated thermal's delivery-anchored point commitment and
/// derive the constant-lead per-plant `K_i` the still-live ring machinery reads.
///
/// The sole `resolve_point` consumer (via [`AnticipatedResolution::resolve`]).
/// Warn-free: [`resolve_anticipated_commitments`] wraps this with the setup-time
/// `K = 0` advisory; [`crate::lp_builder::build_stage_templates`] calls this core
/// directly to attach an identical resolution onto its own `StateSpace` — the
/// same accepted redundant-but-deterministic recompute this crate already
/// applies to the bucket topology, not a second advisory emission. Returns the
/// per-plant resolution and the anticipated-local constant leads: a
/// `LeadStages(ℓ)` plant keeps `ℓ` byte-for-byte; a `LeadTime` plant (gated off
/// the load path until the in-LP ring lands) takes its per-plant max depth as a
/// `k_max`-consistent placeholder, so its LP path is not yet correct.
pub(crate) fn resolve_anticipated_commitments_core(
    system: &System,
) -> (AnticipatedResolution, Vec<usize>) {
    let anticipated_thermals: Vec<&Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .collect();
    let leads: Vec<LeadTime> = anticipated_thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|cfg| match cfg {
            AnticipatedConfig::LeadStages(l) => LeadTime::Stages(*l),
            AnticipatedConfig::LeadTime(h) => LeadTime::Time(*h),
        })
        .collect();
    if leads.is_empty() {
        return (AnticipatedResolution::default(), Vec::new());
    }

    let durations = bucket_topology::study_stage_durations(system);
    let n_stages = durations.len();
    let resolution = AnticipatedResolution::resolve(&leads, &durations, n_stages);

    let lead_stages: Vec<usize> = leads
        .iter()
        .zip(&resolution.per_plant)
        .map(|(lead, point)| match lead {
            LeadTime::Stages(l) => {
                let l = usize::try_from(*l).unwrap_or(usize::MAX);
                // LeadStages byte-identity anchor: c(m)=m−ℓ ⇒ depth ≤ ℓ and each
                // in-horizon C(t) is the singleton {t+ℓ}.
                debug_assert!(
                    point.depth.iter().all(|&d| d <= l),
                    "LeadStages depth must be bounded by ℓ"
                );
                debug_assert!(
                    leadstages_decision_sets_are_singletons(point, l, n_stages),
                    "LeadStages c(m)=m−ℓ ⇒ each in-horizon C(t)={{t+ℓ}}"
                );
                l
            }
            LeadTime::Time(_) => point.depth.iter().copied().max().unwrap_or(0),
        })
        .collect();

    (resolution, lead_stages)
}

/// [`resolve_anticipated_commitments_core`] plus the setup-time `K = 0`
/// advisory ([`warn_on_sub_stage_lead`]) — the single owner of that advisory.
/// Every other caller (e.g. [`crate::lp_builder::build_stage_templates`]) uses
/// the core directly so the advisory never double-emits.
pub(crate) fn resolve_anticipated_commitments(
    system: &System,
) -> (AnticipatedResolution, Vec<usize>) {
    let (resolution, lead_stages) = resolve_anticipated_commitments_core(system);
    let anticipated_thermals: Vec<&Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .collect();
    warn_on_sub_stage_lead(&anticipated_thermals, &resolution);
    (resolution, lead_stages)
}

/// Emit a per-stage setup-time advisory (exclude-with-advisory, never a
/// hard error) for every `K = 0` sub-stage-lead delivery a `LeadTime` plant's
/// calendar resolves to (`PointResolution::self_delivered_stages`): names the
/// plant, the stage, and the effective `lead_stages == 0` alternative.
/// `LeadStages` plants never trigger it (a positive stage-count lead never
/// resolves `c(m) = m`). Called once from [`resolve_anticipated_commitments`]
/// at setup/load time — the established `tracing::warn!` advisory channel
/// (mirrors `StudyParams::from_config`'s budget-below-forward-passes warning);
/// never from a per-scenario/per-trajectory function (log-spam rule).
fn warn_on_sub_stage_lead(thermals: &[&Thermal], resolution: &AnticipatedResolution) {
    for (thermal, point) in thermals.iter().zip(&resolution.per_plant) {
        for stage in point.self_delivered_stages() {
            tracing::warn!(
                "anticipated thermal {} ({}): stage {stage} resolves to a K=0 sub-stage \
                 lead (lead_stages == 0 at this stage); no anticipation binds and this \
                 plant's generation dispatches as ordinary, unconstrained thermal output",
                thermal.id,
                thermal.name,
            );
        }
    }
}

/// Whether every in-horizon delivery stage's decision set is the singleton
/// `{t+ℓ}` — the `LeadStages` byte-identity anchor. Edge stages (`t+ℓ ≥
/// n_stages`) carry empty sets and are skipped.
fn leadstages_decision_sets_are_singletons(
    point: &PointResolution,
    lead: usize,
    n_stages: usize,
) -> bool {
    point.decision_sets.iter().enumerate().all(|(t, set)| {
        t.checked_add(lead)
            .filter(|&m| m < n_stages)
            .is_none_or(|m| set.as_slice() == [m])
    })
}

/// Build the per-pool [`CutStateProjection`], one per stage (pool) `t`, projecting
/// the global [`StateSpace`] onto the cut-state dimensions each pool carries.
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
    state_layout: &StateSpace,
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
const FULL_STATE_CONFIG: StageStateConfig = StageStateConfig {
    storage: true,
    inflow_lags: true,
};

/// Grouped output of [`precompute_lag_data`].
struct LagData {
    stage_lag_transitions: Vec<StageLagTransition>,
    noise_group_ids: Vec<u32>,
    recent_observation_seed: RecentObservationSeed,
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
        noop_season_map = SeasonMap {
            cycle_type: Monthly,
            seasons: Vec::new(),
        };
        &noop_season_map
    };
    // Proxy: the global `max_par_order` stands in for the quarterly PAR order until a
    // separate quarterly stochastic context exists.
    let downstream_par_order = derive_downstream_par_order(stages, stochastic.par().max_order());
    let stage_lag_transitions =
        precompute_stage_lag_transitions(stages, season_map_ref, downstream_par_order);
    // Both outputs derive from `stages`, so they cannot disagree about which
    // stages are in scope; `study_stage_noise_group_ids` re-derives that scope
    // from `System` and is for callers that have no filtered slice.
    let noise_group_ids = precompute_noise_groups(stages);

    let recent_observation_seed = if stages.is_empty() {
        RecentObservationSeed::zero(system.hydros().len())
    } else {
        compute_recent_observation_seed(
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
fn build_scenario_libraries(
    system: &System,
    stages: &[Stage],
    hydro_ids: &[EntityId],
    stochastic: &StochasticContext,
    stage_lag_transitions: &[StageLagTransition],
    training_source: &ScenarioSource,
    simulation_source: &ScenarioSource,
    forward_passes: u32,
    downstream_par_order: usize,
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
                downstream_par_order,
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
                downstream_par_order,
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
                downstream_par_order,
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
                downstream_par_order,
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
        .map(|c| c.contract_type == Import)
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

/// Map each entity's declared numeric ID to its position in a canonically
/// ordered slice (`System::hydros()` / `System::thermals()`).
///
/// Canonical order sorts by `(operational_start_date, id)`
/// (`cobre_core::system::builder::sort_canonical`), which is id-ascending only
/// when every entity shares one operational start date. A staggered-
/// commissioning system (filling reservoirs, future-entry plants) breaks that
/// coincidence, so any id-keyed initial-condition lookup MUST resolve through
/// this map — `binary_search_by_key` over the canonical slice itself silently
/// returns `Err` (or the wrong index) for an out-of-id-order entry, dropping
/// its seed to the default `0.0`.
fn id_to_position<T>(entities: &[T], id_of: impl Fn(&T) -> i32) -> HashMap<i32, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, e)| (id_of(e), idx))
        .collect()
}

/// Build the initial state vector from the system's initial conditions.
///
/// Layout `[storage(0..N), lags(N..N*(1+L))]` (N hydros, L = max PAR order),
/// storage indexed by each hydro's position in `system.hydros()`'s canonical
/// order. Lag slots come from `initial_conditions.past_inflows`:
/// `values_m3s[l]` with index 0 = lag 1 (most recent), index L-1 = lag L
/// (oldest). Storage-only when `max_par_order == 0`.
fn build_initial_state(
    system: &System,
    study_dims: &StudyDimensions,
    layout: &StateSpace,
) -> Vec<f64> {
    let mut state = vec![0.0_f64; layout.n_state];
    let hydros = system.hydros();
    let hydro_positions = id_to_position(hydros, |h: &Hydro| h.id.0);
    let ic = system.initial_conditions();

    for hs in &ic.storage {
        if let Some(&idx) = hydro_positions.get(&hs.hydro_id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    for hs in &ic.filling_storage {
        // The seed writes the same coordinate the PreFilling pin
        // (`fill_prefilling_shortcircuit`) freezes to `[seed, seed]`; do not merge
        // the two collections or re-index the column — a separate index would
        // silently desync from that pin.
        if let Some(&idx) = hydro_positions.get(&hs.hydro_id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    if layout.max_par_order > 0 {
        let n_h = layout.hydro_count;
        for pi in &ic.past_inflows {
            if let Some(&idx) = hydro_positions.get(&pi.hydro_id.0) {
                let n_lags = pi.values_m3s.len().min(layout.max_par_order);
                for lag in 0..n_lags {
                    let slot = layout.inflow_lags.start + lag * n_h + idx;
                    state[slot] = pi.values_m3s[lag];
                }
            }
        }
    }

    // Anticipated ring, slot-major: `state[anticipated_slots_out.start + slot *
    // n_anticipated + local_idx]`. This IS the state-vector numbering
    // (`StateSpace::state_to_lp_column`'s identity domain), the same
    // `anticipated_slots_out` position every other outgoing-state read uses —
    // never `anticipated_state` (the relocated, incoming-only pinned block).
    // Padding slots `[K_i, k_max)` must stay zero — the in-LP ring's row/column
    // fill in `lp/builder` assumes it.
    if layout.n_anticipated > 0 && layout.k_max > 0 {
        debug_assert_eq!(
            study_dims.anticipated_thermal_indices.len(),
            layout.n_anticipated,
            "anticipated_thermal_indices length must equal n_anticipated",
        );
        let thermals = system.thermals();
        let thermal_positions = id_to_position(thermals, |t: &Thermal| t.id.0);
        let n_ant = layout.n_anticipated;
        let ant_start = layout.anticipated_slots_out.start;
        for history in &ic.past_anticipated_commitments {
            let Some(&global_idx) = thermal_positions.get(&history.thermal_id.0) else {
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

/// Write the travel-time bucket seed into `state`'s declared `transit_buckets_out`
/// slots — the same index space [`StateSpace::state_to_lp_incoming_column`]
/// remaps to the pinned `transit_buckets_in` LP column, so no separate pin wiring is
/// needed beyond this splice.
fn splice_transit_bucket_seed(
    state: &mut [f64],
    layout: &StateSpace,
    system: &System,
    topology: &bucket_topology::TransitBucketTopology,
) {
    let seed = bucket_seed::build_initial_transit_bucket_state(system, topology);
    debug_assert_eq!(seed.len(), layout.n_buckets);
    for (b, &value) in seed.iter().enumerate() {
        state[layout.transit_buckets_out.start + b] = value;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
