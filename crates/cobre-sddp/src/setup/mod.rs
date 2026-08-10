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

use chrono::NaiveDate;
use cobre_core::ContractType::Import;
use cobre_core::temporal::SeasonCycleType::Monthly;
use cobre_core::temporal::SeasonMap;
use cobre_core::temporal::StageLagTransition;
use cobre_core::temporal::StageStateConfig;
use cobre_io::Config;
use cobre_io::config::BackwardScheduler;
use cobre_solver::ActiveProfile;
use cobre_stochastic::DerivedInflowSeeds;
use cobre_stochastic::derive_inflow_seeds;
use cobre_stochastic::noise_entity_order;
use cobre_stochastic::par::lag_transition::derive_downstream_par_order;
use cobre_stochastic::par::lag_transition::precompute_noise_groups;
use cobre_stochastic::par::lag_transition::precompute_stage_lag_transitions;
use cobre_stochastic::season_cast::{DatedWindow, StageCalendar, post_study_calendar_stages};

use crate::StageTemplates;
use crate::config::LoopParams;
use crate::resolved_parameters::{ResolvedParameters, build_resolved_parameters};
use crate::scaling_report::ScalingReport;
use crate::simulation::SimulationConfig;
use crate::solve::solver_phase::{Phase, validate_phase_solver_config};
use crate::stochastic::noise_key::build_noise_key_table;
mod accessors;
pub(crate) mod bucket_topology;
pub(crate) mod methodology_config;
pub mod node_graph;
mod orchestration;
pub mod params;
pub(crate) mod scenario_libraries;
pub mod scenario_library_set;
pub mod stage_data;
pub mod stochastic_pipeline;
pub(crate) mod template_postprocess;

pub use node_graph::{
    EnumeratedPlan, NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor,
    OpeningSource, StageIdx, Traversal, TypedVec,
};
pub use params::{
    ConstructionConfig, DEFAULT_COST_SCALE_FACTOR, DEFAULT_FORWARD_PASSES, DEFAULT_MAX_ITERATIONS,
    DEFAULT_SEED, SimulationEnumeratedRequest, StudyParams,
};
pub use scenario_library_set::{PhaseLibraries, ScenarioLibraries};
pub use stage_data::StageData;
pub use stochastic_pipeline::{
    PrepareStochasticResult, build_ncs_factor_entries, load_load_factors_for_stochastic,
    prepare_stochastic, study_stage_noise_group_ids,
};

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{
    AnticipatedConfig, EntityId, HorizonGraph, Hydro, PostStudyStages, PostStudyThermalBound,
    Stage, StageId, System, Thermal,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::StageIdResolver;
use cobre_io::build_hydro_reference_volumes_resolved;
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::{
    ClassSchemes, ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext,
    SweepDirection,
};

use crate::{
    config::{CutManagementConfig, EventParams},
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversionSet, build_energy_conversion_set},
    error::SddpError,
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    indexer::{CutStateProjection, HydroCellIndex, StateSpace, StudyDimensions},
    lead_time::{
        AnticipatedResolution, LeadTime, PointResolution, SpreadResolution,
        resolve_future_delivery_decider,
    },
    lp_builder::{M3S_TO_HM3, build_stage_templates},
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

    /// Resolved `(parameter_id, stage)` coefficients; consumed by the LP builder
    /// and the generic-constraint echo.
    pub(crate) resolved_parameters: ResolvedParameters,

    /// Setup-side resolved post-study boundary artifacts
    /// Sampling schemes and pre-built libraries for training and simulation phases.
    pub scenario_libraries: ScenarioLibraries,

    /// The runtime node graph (F7): node identity/order, the `node → pool`
    /// map, and per-node Ω views/out-edges. Absent `nodes[]` this is the
    /// byte-exact chain degeneracy. Reached through
    /// [`crate::context::TrainingContext::node_graph`] on the hot path.
    pub node_graph: node_graph::NodeGraph,
    /// Iteration-loop parameters projected from [`crate::config::LoopConfig`].
    ///
    /// `n_fwd_threads` is excluded (derived at runtime) and supplied as a per-call
    /// argument to [`StudySetup::train`].
    pub loop_params: LoopParams,

    /// Simulation pipeline parameters, stored directly as [`crate::simulation::SimulationConfig`].
    pub simulation_config: SimulationConfig,

    /// Whether simulation's scenario source is a declared census
    /// (`simulation.selection = enumerated`) or Monte Carlo sampling —
    /// resolved once the node graph exists, mirroring
    /// [`Self::simulation_config`]'s `n_scenarios`. The caller reads this to
    /// select [`crate::simulation::SimulationWeighting::Census`] vs
    /// [`crate::simulation::SimulationWeighting::Uniform`] for
    /// `aggregate_simulation`.
    pub simulation_enumerated: SimulationEnumeratedRequest,

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

    /// Backward-pass scheduler (`training.parallelism.backward_scheduler`,
    /// carrying the opening-block size), threaded into [`StudySetup::train`]
    /// alongside [`Self::backward_profile`].
    pub(crate) backward_scheduler: BackwardScheduler,

    /// Opening-block-scheduler claim-order override, threaded into
    /// [`StudySetup::train`] alongside [`Self::backward_scheduler`]. No
    /// `training.*` config field resolves this yet — a reserved test-support
    /// seam; production always resolves `true` (see
    /// [`crate::solve::solver_phase::SolverProfiles::hardest_first_claim_order`]).
    pub(crate) hardest_first_claim_order: bool,

    /// Stochastic numerical methodology parameters (`horizon`, `inflow_method`).
    pub(crate) methodology: methodology_config::MethodologyConfig,

    /// Derived per-hydro PAR lag-slot and accumulator seeds ([`derive_inflow_seeds`]),
    /// applied to the stage-0 lag block and to every trajectory start in the
    /// forward pass and simulation pipeline instead of zero-filling. All-zero
    /// when the derivation has no resolvable data.
    pub(crate) derived_inflow_seeds: DerivedInflowSeeds,

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
    // Every field is consumed via the constructor's threaded LOCAL
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
            training_enumerated,
            stopping_rule_set,
            n_scenarios,
            simulation_enumerated,
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
            backward_scheduler,
            cost_scale_factor,
            inflow_lag_depth,
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
        let solve_order_keys = build_noise_key_table(system, &stochastic)?;
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
        let (state_layout, hydro_count, anticipated_thermal_indices) = resolve_state_layout(
            system,
            stochastic.par(),
            &transit_bucket_topology,
            inflow_lag_depth,
        )?;

        // The sole `derive_inflow_seeds` call site: every consumer (the lag block
        // below, `StudySetup::derived_inflow_seeds`) reads this one value — do not
        // add a second call. Computed locally on every rank from the already-
        // broadcast `system` rather than carried over the wire: the derivation is
        // a pure function of `system`, so every rank derives a bit-identical seed
        // with no extra broadcast.
        let noop_season_map = SeasonMap {
            cycle_type: Monthly,
            seasons: Vec::new(),
        };
        let season_map_ref = system
            .policy_graph()
            .season_map
            .as_ref()
            .unwrap_or(&noop_season_map);
        let derived_inflow_seeds = match system.stages().iter().find(|s| s.id >= 0) {
            None => DerivedInflowSeeds::zero(system.hydros().len(), state_layout.max_par_order),
            Some(first_stage) => derive_inflow_seeds(
                system.inflow_history(),
                &system.initial_conditions().recent_observations,
                system.hydros(),
                first_stage,
                season_map_ref,
                state_layout.max_par_order,
            ),
        };

        // Built here, before the stage templates: `TemplateBuildCtx` reads it during
        // `StageLayout::new`, and the SAME value (never rebuilt or cloned) is stored
        // on `StageData` below.
        let hydro_cell_index = HydroCellIndex::build(system.hydros());

        let EnergyAndTemplates {
            energy_conversion,
            stage_templates,
            scaling_report,
            resolved_parameters,
        } = build_energy_and_templates(
            system,
            inflow_method,
            &stochastic,
            &hydro_models,
            &scalar_parameters,
            &state_layout,
            cost_scale_factor,
            &transit_bucket_topology.per_stage_mask,
            &transit_bucket_topology.arc_stage_weights,
            &transit_bucket_topology.arc_spread_chrono,
            &transit_bucket_topology.arc_arrival_density,
            &hydro_cell_index,
        )?;

        let study_dims = build_study_dimensions(
            system,
            &stage_templates,
            inflow_method,
            hydro_count,
            anticipated_thermal_indices,
        );

        let mut initial_state = build_initial_state(
            system,
            &study_dims,
            &state_layout,
            &derived_inflow_seeds.lag_values,
        );
        splice_transit_bucket_seed(
            &mut initial_state,
            &state_layout,
            system,
            &transit_bucket_topology,
        );

        let n_stages = stage_templates.templates.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);

        let stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let study_stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();

        let LagData {
            stage_lag_transitions,
            noise_group_ids,
            downstream_par_order,
        } = precompute_lag_data(system, &stages, &stochastic, season_map_ref);

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
            &derived_inflow_seeds.lag_values,
            state_layout.max_par_order,
            &derived_inflow_seeds.accum,
            &derived_inflow_seeds.weight,
        )?;

        // G1: binds after `build_scenario_libraries` — an `External`-bound
        // node's Ω addresses the standardized library's raw scenario axis,
        // so binding earlier would race the library's own standardization.
        // Also binds BEFORE the FCF / cut_state_layouts construction below: the
        // pool axis they use is resolved through this graph's `node → pool` map.
        let stage_id_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let node_graph = node_graph::build_node_graph(
            system.policy_graph(),
            n_stages,
            &stage_id_resolver,
            &stochastic,
        )?;

        reject_scenario_id_under_sampled_selection(&node_graph, training_enumerated)?;
        {
            let prov = stochastic.provenance();
            reject_insample_class_under_external_nodes(
                &node_graph,
                (prov.inflow_scheme, stochastic.n_hydros()),
                (prov.load_scheme, stochastic.n_load_buses()),
                (prov.ncs_scheme, stochastic.n_stochastic_ncs()),
            )?;
        }

        // Resolves any `enumerated`-declared phase's actual count now that the
        // graph exists — config load could only signal the request, never the
        // count. `forward_passes`/`n_scenarios` carry a `sampled`-shaped
        // placeholder until this point when enumerated was requested.
        warn_on_enumeration_asymmetry(
            training_enumerated,
            matches!(
                simulation_enumerated,
                SimulationEnumeratedRequest::Enumerated
            ),
        );
        let forward_passes = if training_enumerated {
            resolve_enumerated_training_count(&node_graph)?
        } else {
            forward_passes
        };
        let n_scenarios = match simulation_enumerated {
            SimulationEnumeratedRequest::Enumerated => {
                resolve_enumerated_simulation_count(&node_graph)?
            }
            SimulationEnumeratedRequest::Sampled => n_scenarios,
        };

        // Resolved AFTER the guard-checked counts above (`resolve_enumerated_training_count`
        // has already run the enumerated admissibility guards for a `true`
        // `training_enumerated`), so this resolution cannot fail — it is the
        // typed reification of what the two calls above already validated.
        let traversal =
            node_graph::Traversal::resolve(&node_graph, training_enumerated, forward_passes);

        let cut_state_layouts = build_cut_state_layouts(system, &state_layout, &node_graph);
        let pool_state_dimensions: Vec<usize> = cut_state_layouts
            .iter()
            .map(CutStateProjection::n_slots)
            .collect();
        // Cut-RECEIPT stride selected through the resolved traversal. The
        // `Sampled` arm keeps `pool_cut_stride` — the mean+σ statistical margin
        // capped at `forward_passes`, one candidate cut per TRIAL POINT — and
        // NEVER `forward_solve_counts`, the enumerated engine's node-deduplicated
        // per-pool FORWARD-SOLVE count, which under-reserves a branched pool's
        // slots (the backward still produces one cut per trial point, so the next
        // trial collides with a still-active slot — `CutPool::add_cut`'s
        // double-insert panic). The `Enumerated` arm sizes at the node-native cut
        // count, `enumerated_pool_cut_stride`: exactly 1 per non-leaf node
        // (in-degree 1, one distinct incoming state, one cut per iteration) and 0
        // for the shared leaf pool — NOT the sampled bound, which would keep the
        // per-pool capacity/basis/broadcast/checkpoint reservation the node-native
        // backward never fills.
        let visit_bounds = match &traversal {
            node_graph::Traversal::Sampled { forward_passes } => {
                node_graph.pool_cut_stride(*forward_passes)
            }
            node_graph::Traversal::Enumerated(_) => {
                node_graph::enumerated_pool_cut_stride(&node_graph)
            }
        };
        let fcf = FutureCostFunction::new_per_pool(
            &pool_state_dimensions,
            state_layout.n_state,
            forward_passes,
            fcf_capacity_iterations,
            &vec![0; node_graph.n_pools],
            &visit_bounds,
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
        let block_counts_per_stage: Vec<usize> = stage_templates
            .block_hours_per_stage
            .iter()
            .map(Vec::len)
            .collect();
        let max_blocks = block_counts_per_stage.iter().copied().max().unwrap_or(0);

        let pumping_consumption_mw_per_m3s = build_pumping_consumption(system);
        let contract_prices_per_stage =
            build_contract_prices_per_stage(system, n_stages, &block_counts_per_stage);
        let contract_is_import = build_contract_is_import(system);

        let anticipated_windows = build_anticipated_windows(system);

        admission_gate(&risk_measures, &stopping_rule_set, training_enumerated)?;

        let hydro_min_storage_hm3: Vec<f64> =
            system.hydros().iter().map(|h| h.min_storage_hm3).collect();

        Ok(Self {
            stage_data: stage_data::StageData {
                stage_templates,
                state: state_layout,
                study_dims,
                hydro_cell_index,
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
            resolved_parameters,
            scenario_libraries,
            node_graph,
            loop_params: LoopParams {
                seed,
                forward_passes,
                training_enumerated,
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
            simulation_enumerated,
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
            hardest_first_claim_order: true,
            methodology: methodology_config::MethodologyConfig {
                horizon,
                inflow_method,
            },
            derived_inflow_seeds,
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
        let not_found = || {
            SddpError::Validation(format!(
                "stochastic NCS entity {slot_id:?} not found in system non_controllable_sources"
            ))
        };
        let dense_col = entity_counts
            .non_controllable_ids
            .iter()
            .position(|&id| id == slot_id.0)
            .ok_or_else(not_found)?;
        let ncs = system
            .non_controllable_sources()
            .iter()
            .find(|n| n.id == *slot_id)
            .ok_or_else(not_found)?;
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
    resolved_parameters: ResolvedParameters,
}

/// Build the energy-conversion set, the resolved parameter table, and the
/// post-processed stage LP templates.
///
/// The energy-conversion set and resolved parameter table are built before the
/// LP templates so the builder can resolve `CoefficientRef::Parameter` values.
/// The resolved parameter table feeds `build_stage_templates` and is returned
/// for the generic-constraint echo. Seasonless stages collapse to season 0,
/// consistent with every other season-indexed lookup.
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
    cost_scale_factor: f64,
    per_stage_mask: &[Vec<usize>],
    arc_stage_weights: &HashMap<usize, Vec<Vec<f64>>>,
    arc_spread_chrono: &HashMap<usize, Vec<Option<SpreadResolution>>>,
    arc_arrival_density: &HashMap<usize, Vec<Option<Vec<f64>>>>,
    hydro_cell_index: &HydroCellIndex,
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
    let stage_block_counts: Vec<usize> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .collect();
    let resolved_parameters = build_resolved_parameters(
        scalar_parameters,
        &energy_conversion,
        &hydro_models.productivity_override,
        system.hydros(),
        &stage_to_season,
        &study_stage_ids,
        &stage_block_counts,
        n_stages_pre,
        cost_scale_factor,
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
        hydro_cell_index,
        stochastic
            .provenance()
            .load_scheme
            .unwrap_or(SamplingScheme::InSample),
    )?;

    let scaling_report = template_postprocess::postprocess_templates(
        &mut stage_templates,
        system,
        state_layout,
        cost_scale_factor,
    );

    if stage_templates.templates.is_empty() {
        return Err(SddpError::Validation(
            "system has no study stages".to_string(),
        ));
    }

    Ok(EnergyAndTemplates {
        energy_conversion,
        stage_templates,
        scaling_report,
        resolved_parameters,
    })
}

/// `L_state = max(computed_order, declared_depth)` — the single widening
/// every lag-state-slot source (`resolve_state_layout`'s dense stride and
/// per-hydro activeness mask, `build_opening_tree_library`,
/// `rebuild_historical_library_non_root`) applies in lockstep so a declared
/// `state_space.inflow_lag_depth` never truncates on one source while
/// widening another. `None` leaves `computed_order` unchanged.
#[must_use]
pub fn widen_lag_state_depth(computed_order: usize, declared_depth: Option<u32>) -> usize {
    declared_depth.map_or(computed_order, |d| computed_order.max(d as usize))
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
    inflow_lag_depth: Option<u32>,
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
            plant_id.unwrap_or(EntityId(-1))
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
    let max_par_order: usize = widen_lag_state_depth(
        system
            .inflow_models()
            .iter()
            .filter(|m| m.stage_id >= 0)
            .map(|m| m.ar_coefficients.len())
            .max()
            .unwrap_or(0)
            .max(par_lp.max_order()),
        inflow_lag_depth,
    );

    // Per-hydro lag-state-slot count for the cut sparse mask: `max_par_order` (the
    // widened psi stride) when PAR(p)-A annual is active, else the classical AR
    // order, each further raised to `inflow_lag_depth` via `widen_lag_state_depth`
    // — the same `L_state = max(AR order, declared depth)` formula `max_par_order`
    // above applies, so a declared depth widens every hydro's activeness mask in
    // lockstep with the dense stride. `par.order(h)` here would silently truncate
    // the cut row's coefficients on the annual-`ψ̂/12` lag slots and produce
    // over-estimating cuts. Falls back to the dense (already-widened) `max_par_order`
    // stride for a hydro `par_lp` omits (`h >= par_lp.n_hydros()`) — production's
    // `par_lp` always covers every system hydro, so the fallback is inert there; a
    // hydro-free `PrecomputedPar` test fixture paired with a hydro-bearing system
    // relies on it to satisfy the `StateSpace::new` length contract.
    let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
        (0..hydro_count)
            .map(|h| {
                if h < par_lp.n_hydros() {
                    widen_lag_state_depth(par_lp.effective_lag_count(h), inflow_lag_depth)
                } else {
                    max_par_order
                }
            })
            .collect()
    } else {
        vec![0; hydro_count]
    };

    let commitment_hold_windows =
        resolve_commitment_hold_windows(system, &anticipated_thermal_indices);

    // `StateSpace` is the sole role-(a) owner; its constructor finalizes the
    // nonzero mask unconditionally, so every study (storage-only or pure-thermal)
    // has a finalized mask for the single-path mask-driven cut-row loop.
    let mut state = StateSpace::new_with_commitment_hold_windows(
        hydro_count,
        max_par_order,
        transit_bucket_topology.n_buckets,
        transit_bucket_topology.column_order.clone(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_counts,
        commitment_hold_windows.n_commitment,
        commitment_hold_windows.decider_stage,
        commitment_hold_windows.thermal_id,
        commitment_hold_windows.min_max,
        commitment_hold_windows.dest_stage,
    );
    state.set_anticipated_resolution(anticipated_resolution);

    Ok((state, hydro_count, anticipated_thermal_indices))
}

/// [`resolve_commitment_hold_windows`]'s return shape: every field is
/// survivor-indexed and parallel, length [`Self::n_commitment`], in
/// [`StateSpace::commit_out`]'s post-horizon-lane order — never a raw
/// `future_anticipated_deliveries` position, which diverges from this index
/// the moment any window is dropped.
#[derive(Debug, Default, PartialEq)]
struct CommitmentHoldWindows {
    n_commitment: usize,
    /// Survivor `w`'s in-study decider stage.
    decider_stage: Vec<usize>,
    /// Survivor `w`'s owning thermal id.
    thermal_id: Vec<EntityId>,
    /// Survivor `w`'s `(min_mw, max_mw)` commitment interval.
    min_max: Vec<(f64, f64)>,
    /// Survivor `w`'s resolved post-study destination stage index.
    dest_stage: Vec<usize>,
}

/// Resolve every declared post-horizon delivery window
/// ([`cobre_core::FutureAnticipatedDelivery`]) to its in-study decider stage
/// ([`resolve_future_delivery_decider`], spec #1 §4.3 step 2) and its resolved
/// post-study destination stage ([`StageCalendar::resolve_window`] against the
/// post-study calendar), in canonical `(anticipated thermal system position,
/// delivery_start)` order — canonical thermal order, not raw `thermal_id`
/// order, keeps the block's column order declaration-invariant under
/// staggered commissioning the same way [`build_initial_state`]'s
/// id-position-map contract does.
///
/// A window whose thermal has no physical lead time
/// (`AnticipatedConfig::LeadStages`, which never consults the calendar) or
/// whose resolved decider precedes the study horizon
/// (`resolve_future_delivery_decider` returning `None`) is dropped with a
/// setup-time advisory, never a hard error — mirrors
/// [`warn_on_sub_stage_lead`]'s exclude-with-advisory convention. The returned
/// [`CommitmentHoldWindows`] is entirely survivor-indexed; a dropped window
/// leaves no trace in it. `future_anticipated_deliveries`' own sorting
/// invariant (`(thermal_id, delivery_start)` ascending) is what keeps the
/// per-thermal filter below yielding each thermal's windows in
/// `delivery_start` order — the order
/// [`crate::policy_export::build_stage_entity_manifest`] derives a window's
/// local index within its thermal from.
fn resolve_commitment_hold_windows(
    system: &System,
    anticipated_thermal_indices: &[usize],
) -> CommitmentHoldWindows {
    let Some(start_0) = study_start_date(system) else {
        return CommitmentHoldWindows::default();
    };
    let stage_lengths_hours = bucket_topology::study_stage_durations(system);
    let thermals = system.thermals();
    let ic = system.initial_conditions();

    let post_study_calendar_stages_vec = system
        .post_study_stages()
        .map(|post_study| post_study_calendar_stages(&post_study.stages))
        .unwrap_or_default();
    let post_study_calendar = StageCalendar::new(&post_study_calendar_stages_vec);

    let mut decider_stage = Vec::new();
    let mut thermal_id = Vec::new();
    let mut min_max = Vec::new();
    let mut dest_stage = Vec::new();
    for &t_idx in anticipated_thermal_indices {
        let thermal = &thermals[t_idx];
        let entries: Vec<_> = ic
            .future_anticipated_deliveries
            .iter()
            .filter(|d| d.thermal_id == thermal.id)
            .collect();
        if entries.is_empty() {
            continue;
        }

        let Some(AnticipatedConfig::LeadTime(delta_hours)) = thermal.anticipated_config else {
            for entry in &entries {
                tracing::warn!(
                    "anticipated thermal {} ({}): future_anticipated_deliveries window \
                     [{}, {}) dropped — anticipated_config has no physical lead time \
                     (LeadStages never consults the calendar) to resolve a post-horizon \
                     in-study decider",
                    thermal.id,
                    thermal.name,
                    entry.delivery_start,
                    entry.delivery_end,
                );
            }
            continue;
        };

        for entry in entries {
            let window_end_hours = hours_between(entry.delivery_end, start_0);
            if let Some((decider, _kind)) =
                resolve_future_delivery_decider(delta_hours, &stage_lengths_hours, window_end_hours)
            {
                let window = DatedWindow {
                    start_date: entry.delivery_start,
                    end_date: entry.delivery_end,
                };
                let resolved_dest = post_study_calendar.resolve_window(&window);
                debug_assert!(
                    resolved_dest.is_some(),
                    "a surviving future_anticipated_deliveries window must resolve to a \
                     post-study stage — cobre-io's coverage validator rejects any window it \
                     cannot cover exactly before setup ever runs this resolution"
                );
                decider_stage.push(decider);
                thermal_id.push(thermal.id);
                min_max.push((entry.min_mw, entry.max_mw));
                dest_stage.push(resolved_dest.unwrap_or(0));
            } else {
                tracing::warn!(
                    "anticipated thermal {} ({}): future_anticipated_deliveries window \
                     [{}, {}) dropped — out of the lead's reach (its decider precedes \
                     the study horizon)",
                    thermal.id,
                    thermal.name,
                    entry.delivery_start,
                    entry.delivery_end,
                );
            }
        }
    }

    let n_commitment = decider_stage.len();
    CommitmentHoldWindows {
        n_commitment,
        decider_stage,
        thermal_id,
        min_max,
        dest_stage,
    }
}

/// Per-`(thermal, post-study stage)` cost/bounds lookup — [`PostStudyStages::
/// thermal_bounds`] verbatim, never rebuilt into a nondeterministic-
/// iteration-order map.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyThermalLookup {
    bounds: Vec<PostStudyThermalBound>,
}

impl PostStudyThermalLookup {
    fn new(bounds: Vec<PostStudyThermalBound>) -> Self {
        debug_assert!(
            bounds.is_sorted_by_key(|b| (b.thermal_id, b.post_study_stage_index)),
            "PostStudyStages::thermal_bounds must already be canonically sorted by \
             (thermal_id, post_study_stage_index) — the cobre-io parser's own invariant"
        );
        Self { bounds }
    }

    /// `(cost_per_mwh, min_mw, max_mw)` declared for `(thermal_id,
    /// post_study_stage_index)`; `None` when undeclared.
    #[must_use]
    pub(crate) fn lookup(
        &self,
        thermal_id: EntityId,
        post_study_stage_index: usize,
    ) -> Option<(f64, f64, f64)> {
        self.bounds
            .binary_search_by_key(&(thermal_id, post_study_stage_index), |b| {
                (b.thermal_id, b.post_study_stage_index)
            })
            .ok()
            .map(|i| {
                let b = &self.bounds[i];
                (b.cost_per_mwh, b.min_mw, b.max_mw)
            })
    }
}

/// Setup-side resolved post-study boundary artifacts
/// ([`System::post_study_stages`]), built once so the LP builder
/// (`TemplateBuildCtx`/`StageLayout`) and `policy_export` read them without
/// re-deriving the calendar walk, the discount continuation, or the
/// per-thermal lookup. Every field is empty without `post_study_stages` —
/// inert: a study with no post-horizon commitment leaves the rest of setup
/// unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyResolved {
    /// Post-study stage `j`'s own duration in hours
    /// (`PostStudyStage::duration_hours` verbatim).
    pub(crate) total_hours: Vec<f64>,
    /// Cumulative discount factor continued past the study horizon — the exact
    /// values [`template_postprocess::compute_cumulative_discount_factors`]
    /// would hold for these stages had the horizon been extended to cover them
    /// (the study's last cumulative factor bridged by the last study stage's own
    /// one-step factor, then multiplied stage-by-stage).
    pub(crate) cumulative_discount_factors: Vec<f64>,
    /// Per-`(thermal, post-study stage)` cost/bounds lookup.
    pub(crate) thermal_bounds: PostStudyThermalLookup,
}

/// Resolve [`System::post_study_stages`] into the three setup-side artifacts:
/// post-study `total_hours`, the discount continuation, and the per-thermal
/// cost/bounds lookup. `None`/empty `post_study` returns
/// [`PostStudyResolved::default`] — inert.
///
/// `last_real_cumulative` and `last_real_per_stage` are the study's own last
/// cumulative and per-stage discount factors — [`crate::StageTemplates::
/// cumulative_discount_factors`]/[`crate::StageTemplates::discount_factors`]'s
/// last entries, or (`crate::lp_builder::build_stage_templates`'s own
/// `TemplateBuildCtx` build) the identical values computed from the same
/// `compute_per_stage_discount_factors`/`compute_cumulative_discount_factors`
/// pair before those output slices exist. The first post-study cumulative
/// factor bridges the horizon by the LAST STUDY stage's own one-step factor
/// (`last_real_cumulative * last_real_per_stage`), NEVER the first post-study
/// stage's (`* per_stage_post[0]`): the continuation must equal what
/// `cumulative_discount_factors` would hold had the horizon been extended to
/// cover the post-study stages.
pub(crate) fn resolve_post_study_artifacts(
    post_study: Option<&PostStudyStages>,
    pg: &HorizonGraph,
    last_real_cumulative: f64,
    last_real_per_stage: f64,
) -> PostStudyResolved {
    let Some(post_study) = post_study else {
        return PostStudyResolved::default();
    };
    if post_study.stages.is_empty() {
        return PostStudyResolved::default();
    }

    let total_hours: Vec<f64> = post_study.stages.iter().map(|s| s.duration_hours).collect();

    let calendar_stages = post_study_calendar_stages(&post_study.stages);
    // `PostStudyStage` declares no rate-override field (unlike a dispatched
    // `Stage`); a `HorizonGraph` carrying only `annual_discount_rate` keeps this
    // call from resolving a synthetic post-study stage id against a REAL study
    // stage's override in `pg.stage_discount_rate_overrides`.
    let rate_graph = HorizonGraph {
        annual_discount_rate: pg.annual_discount_rate,
        ..HorizonGraph::default()
    };
    let calendar_stage_refs: Vec<&Stage> = calendar_stages.iter().collect();
    let per_stage_post =
        template_postprocess::compute_per_stage_discount_factors(&calendar_stage_refs, &rate_graph);

    let mut cumulative_discount_factors = Vec::with_capacity(per_stage_post.len());
    let mut cumulative = last_real_cumulative * last_real_per_stage;
    for &factor in &per_stage_post {
        cumulative_discount_factors.push(cumulative);
        cumulative *= factor;
    }

    let thermal_bounds = PostStudyThermalLookup::new(post_study.thermal_bounds.clone());

    PostStudyResolved {
        total_hours,
        cumulative_discount_factors,
        thermal_bounds,
    }
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

/// Build the per-pool [`CutStateProjection`], one per pool id, projecting the
/// global [`StateSpace`] onto the cut-state dimensions each pool carries.
///
/// Pool `p`, owned by a non-leaf node `n` (`n.pool_id == p`), is sized by its
/// successor's `state_config` — the cost-to-go node `n`'s successor generates
/// for it (pool `p` is populated by the backward pass when it solves the
/// successor's LP and reads the successor's incoming-state reduced costs).
/// Every edge in the node graph goes `t -> t+1` (asserted in
/// `node_graph::build_declared_node_graph`), so all of `n`'s successors sit at
/// one stage and agree on that stage's `state_config` — the dimension is
/// well-defined by construction, no heterogeneity rule needed. Sizing pool `p`
/// from node `n`'s OWN stage's config instead of its successor's is the
/// off-by-one that compiles but stores cuts at the wrong dimension.
///
/// A leaf node has no successor, so the `successor.state_config` rule does not
/// apply to its pool (the trailing shared leaf pool on a declared graph; the
/// terminal pool `n_stages - 1` on a chain): it is sized by the **full global
/// `n_state`**. With `config.policy.boundary` set, the injected boundary cuts
/// come from the external study and are validated and rebuilt against
/// `fcf.state_dimension` (the global `n_state`) by `load_boundary_cuts` /
/// `inject_boundary_cuts`, so the global dimension is exactly the size
/// injection requires — never a DECOMP stage's reduced config. (Per-slot
/// identity reconciliation between a differently-scoped boundary manifest and
/// the local layout is out of scope here.)
///
/// On the chain degeneracy (`nodes[]` absent), `node_graph.n_pools ==
/// n_stages` and `node.pool_id == t`, so this reduces byte-for-byte to the
/// pre-node-native per-stage projection.
fn build_cut_state_layouts(
    system: &System,
    state_layout: &StateSpace,
    node_graph: &NodeGraph,
) -> Vec<CutStateProjection> {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    // Every pool defaults to the full-dimension projection — the correct value
    // for a leaf-owned pool (no successor) — then non-leaf nodes overwrite
    // their own (disjoint) pool id with the successor-sized projection below.
    let mut layouts =
        vec![CutStateProjection::new(state_layout, FULL_STATE_CONFIG); node_graph.n_pools];
    for (pos, node) in node_graph.nodes.iter_indexed() {
        let Some(succ) = node_graph.successors[pos].first() else {
            continue;
        };
        let config = study_stages[node_graph.nodes[succ.child].stage.0].state_config;
        layouts[node.pool_id] = CutStateProjection::new(state_layout, config);
    }
    layouts
}

/// The all-dimensions cut-state config, sizing a pool to the full global
/// `n_state`. Used for a leaf-owned pool (no successor to govern it) — the
/// terminal pool on a chain.
const FULL_STATE_CONFIG: StageStateConfig = StageStateConfig {
    storage: true,
    inflow_lags: true,
};

/// Grouped output of [`precompute_lag_data`].
struct LagData {
    stage_lag_transitions: Vec<StageLagTransition>,
    noise_group_ids: Vec<u32>,
    downstream_par_order: usize,
}

/// Precompute per-stage lag accumulation weights, noise-group ids, and the
/// downstream PAR order. `season_map_ref` is the caller's already-resolved
/// no-op-fallback season map (see the `from_broadcast_params` hoist).
fn precompute_lag_data(
    system: &System,
    stages: &[Stage],
    stochastic: &StochasticContext,
    season_map_ref: &SeasonMap,
) -> LagData {
    // Proxy: the global `max_par_order` stands in for the quarterly PAR order until a
    // separate quarterly stochastic context exists.
    let downstream_par_order = derive_downstream_par_order(
        stages,
        stochastic.par().max_order(),
        system.policy_graph().season_map.as_ref(),
    );
    let stage_lag_transitions =
        precompute_stage_lag_transitions(stages, season_map_ref, downstream_par_order);
    // Both outputs derive from `stages`, so they cannot disagree about which
    // stages are in scope; `study_stage_noise_group_ids` re-derives that scope
    // from `System` and is for callers that have no filtered slice.
    let noise_group_ids = precompute_noise_groups(stages);

    LagData {
        stage_lag_transitions,
        noise_group_ids,
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
// Rationale: mirrors build_historical_inflow_library/build_external_inflow_library's
// own arity; a context struct would just relocate the arity, not reduce it.
#[allow(clippy::too_many_arguments)]
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
    derived_lag_values: &[f64],
    l_state: usize,
    derived_accum: &[f64],
    derived_weight: &[f64],
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
                derived_lag_values,
                l_state,
                derived_accum,
                derived_weight,
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
                derived_lag_values,
                l_state,
                derived_accum,
                derived_weight,
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
                load_scheme,
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
                derived_lag_values,
                l_state,
                derived_accum,
                derived_weight,
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
                derived_lag_values,
                l_state,
                derived_accum,
                derived_weight,
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
                sim_load_scheme,
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

    let libraries = ScenarioLibraries {
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
    };

    assert_external_library_widths(system, &libraries, training_source)?;
    Ok(libraries)
}

/// G2 (rule 49): every standardized external library's `n_entities()` matches its
/// `noise_entity_order` block width. Reuses [`noise_entity_order`] — the single
/// owner of the three-block entity order — rather than re-deriving a class's
/// entity count a third time; a mismatch is a hard [`SddpError::Validation`]
/// naming the class and both widths. Runs at setup because the standardized
/// libraries exist only after [`build_scenario_libraries`]. `training_source`
/// resolves the same [`ClassSchemes`] every `noise_entity_order` caller in the
/// setup path passes, so training and simulation phases agree on membership.
fn assert_external_library_widths(
    system: &System,
    libraries: &ScenarioLibraries,
    training_source: &ScenarioSource,
) -> Result<(), SddpError> {
    let schemes = ClassSchemes {
        inflow: Some(training_source.inflow_scheme),
        load: Some(training_source.load_scheme),
        ncs: Some(training_source.ncs_scheme),
    };
    let order = noise_entity_order(system, &schemes);
    let check = |library: Option<&ExternalScenarioLibrary>, block_width: usize| {
        library.map_or(Ok(()), |lib| {
            if lib.n_entities() == block_width {
                Ok(())
            } else {
                Err(SddpError::Validation(format!(
                    "external {} library width mismatch: n_entities() = {} but the \
                     noise_entity_order block width is {block_width}",
                    lib.entity_class(),
                    lib.n_entities(),
                )))
            }
        })
    };
    for phase in [&libraries.training, &libraries.simulation] {
        check(phase.external_inflow.as_ref(), order.hydro_ids.len())?;
        check(phase.external_load.as_ref(), order.load_bus_ids.len())?;
        check(phase.external_ncs.as_ref(), order.ncs_entity_ids.len())?;
    }
    Ok(())
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

// ---------------------------------------------------------------------------
// Admission gate
// ---------------------------------------------------------------------------

/// The setup-time admission gate: the permanent arms that survive the
/// node-native collapse, evaluated once from
/// [`StudySetup::from_broadcast_params`]. Absent the gated features (no `gap`
/// stopping rule, or enumerated forwards + an expectation measure at every
/// stage) it returns `Ok(())` unconditionally, so a default study is
/// byte-neutral.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a `gap` stopping rule is present under
/// any stage's effective non-expectation risk measure, or under sampled forward
/// selection.
fn admission_gate(
    risk_measures: &[RiskMeasure],
    stopping_rules: &StoppingRuleSet,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    reject_gap_under_effective_risk_aversion(risk_measures, stopping_rules)?;
    reject_gap_under_sampled_selection(stopping_rules, training_enumerated)
}

/// Reject a `gap` stopping rule under an effective non-expectation risk measure:
/// the exact upper bound a `gap` rule compares the lower bound against is
/// defined only under expectation, so pairing a `gap` rule with an effectively
/// risk-averse measure at any stage is inadmissible. No `gap` rule present ⇒
/// `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the rule, the offending stage's
/// measure, and the admitting condition (an expectation measure).
fn reject_gap_under_effective_risk_aversion(
    risk_measures: &[RiskMeasure],
    stopping_rules: &StoppingRuleSet,
) -> Result<(), SddpError> {
    if !stopping_rules.rules.iter().any(rule_is_gap) {
        return Ok(());
    }
    for (stage, measure) in risk_measures.iter().enumerate() {
        if is_effective_non_expectation(measure) {
            return Err(SddpError::Validation(format!(
                "gap stopping rule is inadmissible under the effective non-expectation \
                 risk measure at stage {stage} ({measure:?}); a gap rule admits only an \
                 expectation risk measure at every stage"
            )));
        }
    }
    Ok(())
}

/// Reject a `gap` stopping rule under sampled forward selection: the exact upper
/// bound a `gap` rule compares the lower bound against is produced only by the
/// enumerated engine; under sampled forwards the upper bound is a noisy
/// statistical estimate, so their difference is not a valid gap. No `gap` rule
/// present, or enumerated forwards ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the rule, the offending selection
/// (sampled), and the admitting condition (enumerated forwards).
fn reject_gap_under_sampled_selection(
    stopping_rules: &StoppingRuleSet,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if training_enumerated {
        return Ok(());
    }
    if stopping_rules.rules.iter().any(rule_is_gap) {
        return Err(SddpError::Validation(
            "gap stopping rule is inadmissible under sampled forward selection; the upper \
             bound is then a statistical estimate, not the exact bound a gap rule requires — \
             a gap rule admits only enumerated forward selection"
                .to_string(),
        ));
    }
    Ok(())
}

/// Whether `rule` is the `gap` stopping-rule variant. Total match (every variant
/// named, `Gap` destructured with no `..`) so a new field on
/// [`StoppingRule::Gap`] or a new [`StoppingRule`] variant must be dispositioned
/// here rather than silently falling through.
fn rule_is_gap(rule: &StoppingRule) -> bool {
    match rule {
        StoppingRule::Gap {
            tolerance: _,
            relative_tolerance: _,
        } => true,
        StoppingRule::IterationLimit { .. }
        | StoppingRule::TimeLimit { .. }
        | StoppingRule::BoundStalling { .. }
        | StoppingRule::GracefulShutdown => false,
    }
}

/// Whether `measure` is *effectively* non-expectation (risk-averse).
/// `CVaR { lambda: 0 }` is documented-equivalent to `Expectation` (its convex
/// weight on the tail is zero), so only a positive risk-aversion weight counts.
/// `RiskMeasure` is destructured exhaustively (every field named, no `..`) so a
/// new `CVaR` field or a new variant must be dispositioned here.
fn is_effective_non_expectation(measure: &RiskMeasure) -> bool {
    match measure {
        RiskMeasure::Expectation => false,
        RiskMeasure::CVaR { alpha: _, lambda } => *lambda > 0.0,
    }
}

/// Advisory (never a reject) for an asymmetric enumeration declaration: when
/// exactly one phase declares `enumerated` scenario selection, one census-only
/// capability is unavailable. Names both phases and the specific missing
/// capability — the exact lower bound (needs enumerated training) or the
/// weighted census simulation statistics (needs enumerated simulation) — never
/// a generic "census required". Symmetric declarations warn nothing.
fn warn_on_enumeration_asymmetry(training_enumerated: bool, simulation_enumerated: bool) {
    match (training_enumerated, simulation_enumerated) {
        (true, false) => tracing::warn!(
            "training declares enumerated scenario selection but simulation declares \
             sampled: the exact lower bound from exhaustive training enumeration is \
             available, but the weighted census simulation statistics are not, since \
             simulation samples its scenarios"
        ),
        (false, true) => tracing::warn!(
            "simulation declares enumerated scenario selection but training declares \
             sampled: the weighted census simulation statistics are available, but the \
             exact lower bound is not, since training samples its scenarios"
        ),
        (true, true) | (false, false) => {}
    }
}

/// Shared enumerated admissibility guard, called by both
/// [`resolve_enumerated_training_count`] and
/// [`resolve_enumerated_simulation_count`] so the two enumerated axes cannot
/// admit different graph shapes: derives the graph's path count via
/// [`node_graph::enumerated_scenario_count`] (propagating its `K^T` u64
/// overflow guard unchanged), rejects a non-singleton within-node opening set
/// via [`reject_within_node_opening_enumeration`], rejects a recombination
/// join via [`reject_recombining_node_enumeration`] — the two preconditions
/// exact node-dedup traversal needs, not merely a fence — then narrows the
/// result to `u32`. `axis` and `count_noun` phrase only the caller's own
/// overflow message (e.g. `("training", "forward-pass")`,
/// `("simulation", "scenario")`).
///
/// # Errors
///
/// Propagates [`node_graph::enumerated_scenario_count`]'s overflow
/// [`SddpError::Validation`]; returns [`SddpError::Validation`] when a node
/// carries more than one opening, when a node has two or more predecessors (a
/// recombination join), or when the derived count exceeds `u32`.
fn enumerated_admissible_count(
    node_graph: &NodeGraph,
    axis: &str,
    count_noun: &str,
) -> Result<u32, SddpError> {
    let derived = node_graph::enumerated_scenario_count(node_graph)?;
    reject_within_node_opening_enumeration(node_graph)?;
    reject_recombining_node_enumeration(node_graph)?;
    u32::try_from(derived).map_err(|_| {
        SddpError::Validation(format!(
            "{axis} enumerated scenario selection derived {derived} paths from the policy \
             graph, exceeding the u32 {count_noun} count the engine addresses"
        ))
    })
}

/// Resolve the `enumerated`-declared TRAINING forward-pass count once the node
/// graph exists, via the shared guard [`enumerated_admissible_count`]: any
/// derived count `>= 1` executes — the enumerated all-paths forward engine is
/// the consumer.
///
/// # Errors
///
/// See [`enumerated_admissible_count`].
fn resolve_enumerated_training_count(node_graph: &NodeGraph) -> Result<u32, SddpError> {
    enumerated_admissible_count(node_graph, "training", "forward-pass")
}

/// Resolve the `enumerated`-declared SIMULATION scenario count once the node
/// graph exists, via the shared guard [`enumerated_admissible_count`]: any
/// derived count `>= 1` executes — the node-native census simulation engine is
/// the consumer, weighting each resolved leaf path through
/// [`node_graph::Traversal::simulation_weighting`].
///
/// # Errors
///
/// See [`enumerated_admissible_count`].
fn resolve_enumerated_simulation_count(node_graph: &NodeGraph) -> Result<u32, SddpError> {
    enumerated_admissible_count(node_graph, "simulation", "scenario")
}

/// The first node pinning an [`OpeningSource::External`] scenario column, in
/// canonical position order — the shared trigger condition
/// [`reject_scenario_id_under_sampled_selection`] and
/// [`reject_insample_class_under_external_nodes`] both gate on.
fn find_external_bound_node(node_graph: &NodeGraph) -> Option<(NodePos, &NodeRuntime)> {
    node_graph
        .nodes
        .iter_indexed()
        .find(|(_, n)| n.openings.source == OpeningSource::External)
}

/// Reject a node carrying a scenario pointer under sampled forward selection: a
/// node's `scenario_id` (surfaced as an `External` opening) selects a
/// deterministic external-library column, which only the enumerated forward
/// engine consumes. Under sampled forwards every node draws its openings by hash,
/// so a declared pointer would be validated at load and then silently ignored;
/// an explicit rejection closes that footgun. Enumerated selection, or a graph
/// carrying no external-bound node, ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id, its
/// stage, and the admitting condition (enumerated forward selection).
fn reject_scenario_id_under_sampled_selection(
    node_graph: &NodeGraph,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if training_enumerated {
        return Ok(());
    }
    if let Some((pos, node)) = find_external_bound_node(node_graph) {
        return Err(SddpError::Validation(format!(
            "node {} (stage {}) declares a scenario_id but training uses sampled forward \
             selection; scenario_id requires enumerated selection",
            node_graph.node_ids[pos], node.stage
        )));
    }
    Ok(())
}

/// Reject a non-empty in-sample class alongside an external-column node graph. An
/// [`OpeningSource::External`] node pins a scenario column that only the external
/// libraries carry; a class with real entities drawing under
/// [`SamplingScheme::InSample`] instead reads the generated opening tree at that
/// column offset, silently sampling a wrong opening (or, where the tree lacks that
/// column, tripping the sampler's opening-range assert). The mixed config is
/// unsupported: for an external-column graph every non-empty class must draw
/// external. A zero-entity class draws nothing and is exempt (the degenerate
/// no-entity class an all-external study still carries).
///
/// Takes each class's `(scheme, entity_count)` directly so it is unit-testable
/// without a [`StochasticContext`].
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending class and the
/// admitting condition (all non-empty classes external).
fn reject_insample_class_under_external_nodes(
    node_graph: &NodeGraph,
    inflow: (Option<SamplingScheme>, usize),
    load: (Option<SamplingScheme>, usize),
    ncs: (Option<SamplingScheme>, usize),
) -> Result<(), SddpError> {
    let Some((pos, node)) = find_external_bound_node(node_graph) else {
        return Ok(());
    };
    for (class, (scheme, count)) in [("inflow", inflow), ("load", load), ("ncs", ncs)] {
        if count > 0 && scheme == Some(SamplingScheme::InSample) {
            return Err(SddpError::Validation(format!(
                "node {} (stage {}) pins an external scenario column, but the {class} class draws \
                 {count} entities under in-sample selection; an external-column node graph admits \
                 only all-external non-empty classes (a zero-entity class is exempt) — set the \
                 {class} class to external selection",
                node_graph.node_ids[pos], node.stage
            )));
        }
    }
    Ok(())
}

/// Reject an `enumerated` graph whose branching is expressed as within-node
/// openings rather than structurally as distinct nodes: every enumerated axis
/// (training's forward engine, the census simulation driver) solves each node
/// once per distinct incoming state and does not enumerate a node's own
/// opening set, so a `|Ω_n| > 1` node would be sampled at a single realization
/// while the exact bound weights it as if fully enumerated. Declare the
/// branching structurally (one realization per node) or use sampled
/// selection.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id, its
/// stage, and its opening count.
fn reject_within_node_opening_enumeration(node_graph: &NodeGraph) -> Result<(), SddpError> {
    if let Some((pos, node)) = node_graph
        .nodes
        .iter_indexed()
        .find(|(_, n)| n.openings.len > 1)
    {
        return Err(SddpError::Validation(format!(
            "enumerated scenario selection requires a singleton within-node opening set at \
             every node, but node id {} (stage {}) carries {} openings; within-node weighted \
             opening enumeration is not yet wired — declare the branching structurally (one \
             realization per node) or use sampled selection",
            node_graph.node_ids[pos], node.stage, node.openings.len
        )));
    }
    Ok(())
}

/// Reject an `enumerated` graph carrying a recombination join — a node reached
/// from two or more predecessor nodes (in-degree ≥ 2, counting how many
/// successor edges name it as a child). Every enumerated axis reconstructs
/// each visited node's incoming state through the single-predecessor
/// [`node_graph::NodeGraph::build_parent_map`] (via [`EnumeratedPlan`]); a multi-parent
/// node would, in a release build, be solved once under one arbitrarily
/// chosen parent's outgoing state while paths arriving through its other
/// parent silently read that wrong state — an invalid exact bound, not a
/// compile error. This setup-time guard precedes and makes release-active
/// `build_parent_map`'s single-predecessor `debug_assert`. Sampled selection is
/// unaffected: it carries each trajectory's own incoming state and resolves
/// recombination natively.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id and its
/// stage (sibling to the within-node-opening rejection above).
fn reject_recombining_node_enumeration(node_graph: &NodeGraph) -> Result<(), SddpError> {
    let mut in_degree: TypedVec<NodePos, usize> = vec![0usize; node_graph.nodes.len()].into();
    for succ in node_graph.successors.iter().flatten() {
        in_degree[succ.child] += 1;
    }
    if let Some(pos) = in_degree.iter().position(|&d| d >= 2).map(NodePos) {
        return Err(SddpError::Validation(format!(
            "enumerated scenario selection requires a single-predecessor (tree) policy graph, \
             but node id {} (stage {}) is reached from {} predecessor nodes (a recombination \
             join); per-prefix state reconstruction for a multi-parent node is not yet wired — \
             use sampled selection, which handles recombination, or declare a non-recombining \
             graph (sibling requirement: a singleton within-node opening set at every node)",
            node_graph.node_ids[pos], node_graph.nodes[pos].stage, in_degree[pos]
        )));
    }
    Ok(())
}

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

/// Build the per-stage RESOLVED contract prices \[$/`MWh`\], per block.
///
/// Outer index is the study-stage index `t` (0-based, matching
/// [`ResolvedBounds`](cobre_core::ResolvedBounds)'s contract stage axis); each
/// inner slice is flat with the per-stage stride `block_counts_per_stage[t]` —
/// index `c * n_blks + blk`, `c` ID-sorted parallel to `system.contracts()`
/// (the same order `EntityCounts::contract_ids` is built in) — carrying
/// `contract_bounds_at_block(c, t, blk).price_per_mwh`. Empty inner slices for
/// a contract-free system or a zero-block stage.
fn build_contract_prices_per_stage(
    system: &System,
    n_stages: usize,
    block_counts_per_stage: &[usize],
) -> Vec<Vec<f64>> {
    let bounds = system.bounds();
    let n_contracts = system.contracts().len();
    (0..n_stages)
        .map(|t| {
            let n_blks = block_counts_per_stage[t];
            (0..n_contracts)
                .flat_map(|c| {
                    (0..n_blks)
                        .map(move |blk| bounds.contract_bounds_at_block(c, t, blk).price_per_mwh)
                })
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
/// order. Lag slots come from `derived_lag_values` (entity-major,
/// `derived_lag_values[pos * L + lag]`, lag 0 = most recent) — already
/// pre-ordered by canonical hydro position at its single derivation site
/// ([`derive_inflow_seeds`]), so `pos` here needs no id lookup. Storage-only
/// when `max_par_order == 0`.
fn build_initial_state(
    system: &System,
    study_dims: &StudyDimensions,
    layout: &StateSpace,
    derived_lag_values: &[f64],
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
        let l = layout.max_par_order;
        for idx in 0..n_h {
            for lag in 0..l {
                let slot = layout.inflow_lags.start + lag * n_h + idx;
                state[slot] = derived_lag_values[idx * l + lag];
            }
        }
    }

    // Anticipated ring, slot-major: `state[commit_out.start + slot *
    // n_anticipated + local_idx]`. This IS the state-vector numbering
    // (`StateSpace::state_to_lp_column`'s identity domain), the same
    // `commit_out` position every other outgoing-state read uses — never
    // `commit_in` (the relocated, incoming-only pinned block).
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
        let ant_start = layout.commit_out.start;
        let study_stages: &[Stage] = match system.stages().iter().position(|s| s.id >= 0) {
            Some(idx) => &system.stages()[idx..],
            None => &[],
        };
        let calendar = StageCalendar::new(study_stages);
        for history in &ic.past_anticipated_commitments {
            let Some(&global_idx) = thermal_positions.get(&history.thermal_id.0) else {
                // Defense-in-depth — the cobre-io validator rejects an unknown ID in
                // production.
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
            // Clamp to K_i, not k_max: a window resolving beyond it would otherwise
            // corrupt the padding slots.
            let k_i = layout.anticipated_lead_stages[local_idx];
            let window = DatedWindow {
                start_date: history.start_date,
                end_date: history.end_date,
            };
            // coverage's whole-day-hours arithmetic keeps a full-coverage ratio
            // bit-exact (mirrors StageCalendar::covers_exactly).
            #[allow(clippy::float_cmp)]
            for (slot, fraction) in calendar.coverage(&window).into_iter().enumerate().take(k_i) {
                if fraction == 1.0 {
                    let off = ant_start + slot * n_ant + local_idx;
                    state[off] = history.value_mw;
                }
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

/// Unroll every declared arc's `past_defluences` windows into the stage-0
/// incoming bucket seed, in [`bucket_topology::TransitBucketTopology::column_order`]
/// order. Runs single-threaded in that canonical order — never a
/// rank-count-dependent parallel reduction.
///
/// Each window `[start_date, end_date)` for upstream hydro `i` contributes
/// `k_d · D_i` (`D_i` the width-scaled volume, `k_d` from
/// [`StageCalendar::hour_window_shares`] anchored at
/// `e_off = start_0 − end_date`, width `end_date − start_date`) into every
/// bucket it reaches. A hydro may carry multiple, non-contiguous windows; each
/// is `filter`ed and deposited independently — never `find`, which would
/// silently keep only the first window and drop the rest, understating the
/// seed with no error.
///
/// `cobre-io`'s `validate_travel_time` coverage gate guarantees every declared
/// arc's windows cover `[start_0 − t_v, start_0)` before this runs; there is no
/// fallback for incomplete coverage.
fn build_initial_transit_bucket_state(
    system: &System,
    topology: &bucket_topology::TransitBucketTopology,
) -> Vec<f64> {
    let mut seed = vec![0.0_f64; topology.n_buckets];
    if topology.n_buckets == 0 {
        return seed;
    }

    let Some(start_0) = study_start_date(system) else {
        debug_assert!(
            false,
            "n_buckets > 0 implies build_transit_bucket_topology sized a depth from a non-empty \
             study calendar, so at least one study stage must exist here"
        );
        return seed;
    };
    let study_stages: &[Stage] = match system.stages().iter().position(|s| s.id >= 0) {
        Some(idx) => &system.stages()[idx..],
        None => &[],
    };
    let calendar = StageCalendar::new(study_stages);
    let ic = system.initial_conditions();
    let hydros = system.hydros();

    let mut start = 0_usize;
    for &depth in &topology.per_plant_depth {
        let plant_id = hydros[topology.column_order[start].0].id;

        for upstream in hydros {
            let Some(t_v) = upstream.travel_time_hours.filter(|&t| t > 0.0) else {
                continue;
            };
            if upstream.downstream_id != Some(plant_id) {
                continue;
            }

            for window in ic
                .past_defluences
                .iter()
                .filter(|w| w.hydro_id == upstream.id)
            {
                debug_assert!(
                    window.end_date <= start_0,
                    "past_defluences window must end at or before start_0 ({start_0}); \
                     cobre-io's validate_travel_time row-5b gate guarantees this"
                );
                let e_off = hours_between(start_0, window.end_date);
                let width = hours_between(window.end_date, window.start_date);
                let volume = width * M3S_TO_HM3 * window.value_m3s;

                let k = calendar.hour_window_shares(t_v, e_off, width);
                for (transit_bucket_offset, &k_val) in k.iter().enumerate().take(depth) {
                    if k_val != 0.0 {
                        seed[start + transit_bucket_offset] += k_val * volume;
                    }
                }
            }
        }

        start += depth;
    }

    debug_assert_eq!(seed.len(), topology.n_buckets);
    seed
}

/// The first study stage's (`id >= 0`, lowest `id`) start date — `start_0`, the
/// anchor every `past_defluences` window's `(e_off, width)` measures against.
/// `None` only when the system declares no study stages.
fn study_start_date(system: &System) -> Option<NaiveDate> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .min_by_key(|s| s.id)
        .map(|s| s.start_date)
}

/// Hours of wall clock between `earlier` and `later` (`later − earlier`),
/// positive when `earlier` precedes `later`.
// Rationale: pre-study spans are on the order of years, far under f64's
// exact-integer range; a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
fn hours_between(later: NaiveDate, earlier: NaiveDate) -> f64 {
    (later - earlier).num_hours() as f64
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
    let seed = build_initial_transit_bucket_state(system, topology);
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

#[cfg(test)]
mod post_study_resolution_tests {
    use super::{PostStudyResolved, resolve_post_study_artifacts, template_postprocess};
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId, HorizonGraph, PostStudyStage, PostStudyStages, PostStudyThermalBound,
    };
    use cobre_stochastic::season_cast::post_study_calendar_stages;

    fn two_stage_post_study() -> PostStudyStages {
        PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 11, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 720.0,
                },
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 12, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 744.0,
                },
            ],
            thermal_bounds: vec![
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 0,
                    cost_per_mwh: 210.0,
                    min_mw: 0.0,
                    max_mw: 350.0,
                },
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 1,
                    cost_per_mwh: 220.0,
                    min_mw: 0.0,
                    max_mw: 300.0,
                },
            ],
        }
    }

    #[test]
    fn post_study_absent_returns_default() {
        let resolved = resolve_post_study_artifacts(None, &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn post_study_with_no_stages_returns_default() {
        let empty = PostStudyStages {
            stages: Vec::new(),
            thermal_bounds: Vec::new(),
        };
        let resolved =
            resolve_post_study_artifacts(Some(&empty), &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn total_hours_matches_declared_duration() {
        let post_study = two_stage_post_study();
        let resolved =
            resolve_post_study_artifacts(Some(&post_study), &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved.total_hours, vec![720.0, 744.0]);
    }

    #[test]
    fn continued_cumulative_discount_is_seed_at_zero_rate() {
        let post_study = two_stage_post_study();
        let resolved =
            resolve_post_study_artifacts(Some(&post_study), &HorizonGraph::default(), 0.9, 1.0);
        assert_eq!(resolved.cumulative_discount_factors, vec![0.9, 0.9]);
    }

    #[test]
    fn continued_cumulative_discount_matches_extended_horizon() {
        let post_study = two_stage_post_study();
        let pg = HorizonGraph {
            annual_discount_rate: 0.08,
            ..HorizonGraph::default()
        };
        // A synthetic two-stage study whose per-stage one-step factors are
        // `[0.95, 0.93]`: its last cumulative factor is `0.95` (the product of
        // the stages strictly before the last) and its last per-stage factor is
        // `0.93`.
        let study_per_stage = [0.95_f64, 0.93_f64];
        let last_real_cumulative = study_per_stage[0];
        let last_real_per_stage = study_per_stage[1];

        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &pg,
            last_real_cumulative,
            last_real_per_stage,
        );

        // Ground truth: extend the horizon with the post-study stages, take the
        // cumulative product over the whole thing, and read off the post-study
        // tail. A resolver that bridged by `per_stage_post[0]` instead of the
        // last study factor would diverge here.
        let calendar_stages = post_study_calendar_stages(&post_study.stages);
        let calendar_stage_refs: Vec<_> = calendar_stages.iter().collect();
        let per_stage_post =
            template_postprocess::compute_per_stage_discount_factors(&calendar_stage_refs, &pg);
        let mut extended_per_stage = study_per_stage.to_vec();
        extended_per_stage.extend_from_slice(&per_stage_post);
        let extended_cumulative =
            template_postprocess::compute_cumulative_discount_factors(&extended_per_stage);

        assert_eq!(
            resolved.cumulative_discount_factors,
            extended_cumulative[study_per_stage.len()..].to_vec()
        );
    }

    #[test]
    fn thermal_bound_lookup_returns_declared_triple() {
        let post_study = two_stage_post_study();
        let resolved =
            resolve_post_study_artifacts(Some(&post_study), &HorizonGraph::default(), 1.0, 1.0);

        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 0),
            Some((210.0, 0.0, 350.0))
        );
        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 1),
            Some((220.0, 0.0, 300.0))
        );
        assert_eq!(resolved.thermal_bounds.lookup(EntityId(2), 0), None);
    }
}
