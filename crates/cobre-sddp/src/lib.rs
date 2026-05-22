//! SDDP solver for hydrothermal dispatch.
//!
//! Implements the SDDP algorithm: forward/backward passes, Benders cuts, risk measures,
//! convergence monitoring, and policy simulation. Parallelized via rayon (intra-rank)
//! and ferrompi (inter-rank).

// Relax strict production lints for test builds (normal in test contexts).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
    )
)]

// Module visibility policy (audit reference: HI-1, 2026-05-18):
//
// - `pub mod` modules are accessed by name from outside the crate.
//   `setup` and `policy_export` are reached by qualified path from
//   `cobre-cli` / `cobre-python` / examples; the others are reached by
//   qualified path from this crate's own integration tests in `tests/`
//   (which compile as separate crates and so need pub visibility).
//   Downstream crates SHOULD prefer the curated re-exports below; the
//   `pub mod` namespaces are not a semver-stable API.
// - `pub(crate)` modules are pure internals — never named from outside
//   this crate (verified by grep against `tests/`, `examples/`, and the
//   other workspace crates).
pub(crate) mod backward;
pub(crate) mod backward_pass_state;
pub mod basis_reconstruct;
pub mod config;
pub mod context;
pub mod convergence;
pub(crate) mod conversion;
pub mod cut;
pub mod cut_selection;
pub mod cut_sync;
pub mod energy_conversion;
pub mod error;
pub mod estimation;
pub mod forward;
pub(crate) mod forward_pass_state;
pub(crate) mod fpha_fitting;
pub(crate) mod generic_constraints;
pub mod horizon_mode;
pub mod hydro_models;
pub mod indexer;
pub mod inflow_method;
pub mod lag_transition;
pub mod lower_bound;
pub mod lp_builder;
pub(crate) mod noise;
pub mod orchestration;
pub mod policy_export;
pub(crate) mod policy_load;
pub(crate) mod provenance;
pub mod resolved_parameters;
pub mod risk_measure;
pub(crate) mod scaling_report;
pub mod setup;
pub mod simulation;
pub mod solver_stats;
pub(crate) mod stage_solve;
pub(crate) mod state_exchange;
pub(crate) mod stochastic_summary;
pub mod stopping_rule;
pub mod training;
pub(crate) mod training_output;
pub(crate) mod training_session;
pub(crate) mod trajectory;
pub mod validate_phases;
pub(crate) mod visited_states;
pub mod workspace;

// ── basis_reconstruct ─────────────────────────────────────────────────────────
pub use basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW;
// ── config ────────────────────────────────────────────────────────────────────
pub use config::TrainingConfig;
// ── convergence ───────────────────────────────────────────────────────────────
pub use convergence::ConvergenceMonitor;
// ── cut ───────────────────────────────────────────────────────────────────────
pub use cut::wire::{cut_wire_size, deserialize_cut, serialize_cut, CutWireHeader};
pub use cut::{CutPool, FutureCostFunction};
// ── cut_selection ─────────────────────────────────────────────────────────────
pub use cut_selection::CutSelectionStrategy;
// ── cut_sync ──────────────────────────────────────────────────────────────────
pub use cut_sync::CutSyncBuffers;
// ── energy_conversion ─────────────────────────────────────────────────────────
pub use energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
// ── error ─────────────────────────────────────────────────────────────────────
pub use error::SddpError;
// ── estimation ────────────────────────────────────────────────────────────────
pub use estimation::{estimate_from_history, EstimationPath, EstimationReport};
// ── forward ───────────────────────────────────────────────────────────────────
pub use forward::{build_cut_row_batch_into, SyncResult};
// ── hydro_models ──────────────────────────────────────────────────────────────
pub use hydro_models::{
    build_hydro_model_summary, prepare_hydro_models, FphaHydroDetail, HydroModelSummary,
    PrepareHydroModelsResult, ProductionModelSource,
};
// ── indexer ───────────────────────────────────────────────────────────────────
pub use indexer::{EquipmentCounts, FphaColumnLayout, StageIndexer};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
pub use lp_builder::{build_stage_templates, StageTemplates};
// ── policy_load ───────────────────────────────────────────────────────────────
pub use policy_load::{
    build_basis_cache_from_checkpoint, inject_boundary_cuts, load_boundary_cuts,
    validate_policy_compatibility,
};
// ── provenance ────────────────────────────────────────────────────────────────
pub use provenance::{build_provenance_report, ModelProvenanceReport, ProvenanceSource};
// ── risk_measure ──────────────────────────────────────────────────────────────
pub use risk_measure::{BackwardOutcome, RiskMeasure};
// ── setup ─────────────────────────────────────────────────────────────────────
pub use setup::{
    prepare_stochastic, PrepareStochasticResult, StudyParams, StudySetup, DEFAULT_MAX_ITERATIONS,
    DEFAULT_SEED,
};
// ── simulation ────────────────────────────────────────────────────────────────
pub use simulation::{
    aggregate_simulation, simulate, ScenarioCategoryCosts, SimulationError, SimulationHydroResult,
    SimulationScenarioResult, SimulationStageResult, SimulationSummary,
};
// ── solver_stats ──────────────────────────────────────────────────────────────
pub use solver_stats::{
    pack_delta_scalars, pack_scenario_stats, unpack_delta_scalars, unpack_scenario_stats,
    SolverStatsDelta, SOLVER_STATS_DELTA_SCALAR_FIELDS,
};
// ── stochastic_summary ────────────────────────────────────────────────────────
pub use stochastic_summary::{
    build_stochastic_summary, estimation_report_to_fitting_report,
    inflow_models_to_annual_component_rows, inflow_models_to_ar_rows, inflow_models_to_stats_rows,
    ArOrderSummary, StochasticSource, StochasticSummary,
};
// ── stopping_rule ─────────────────────────────────────────────────────────────
pub use stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};
// ── training ──────────────────────────────────────────────────────────────────
pub use training::{train, TrainingOutcome, TrainingResult};
// ── training_output ───────────────────────────────────────────────────────────
pub use training_output::build_training_output;
// ── resolved_parameters ───────────────────────────────────────────────────────
pub use resolved_parameters::{
    build_resolved_parameters, deserialize_resolved_parameters, serialize_resolved_parameters,
    ResolvedParameters, ResolvedParametersError,
};
// ── state_exchange ────────────────────────────────────────────────────────────
pub use state_exchange::ExchangeBuffers;
// ── trajectory ────────────────────────────────────────────────────────────────
pub use trajectory::TrajectoryRecord;
// ── workspace ─────────────────────────────────────────────────────────────────
pub use workspace::{CapturedBasis, BASIS_BROADCAST_WIRE_VERSION};
