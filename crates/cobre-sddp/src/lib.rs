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
pub mod dcs;
pub mod energy_conversion;
pub mod error;
pub mod estimation;
pub mod forward;
pub(crate) mod forward_pass_state;
pub(crate) mod fpha_fitting;
pub(crate) mod gemm;
pub(crate) mod generic_constraints;
pub mod horizon_mode;
pub mod hydro_models;
pub mod indexer;
pub mod inflow_method;
pub mod lag_transition;
pub mod lower_bound;
pub mod lp_builder;
pub(crate) mod noise;
pub mod noise_key_diag;
pub mod orchestration;
pub mod policy_export;
pub(crate) mod policy_load;
pub(crate) mod provenance;
pub mod resolved_parameters;
pub mod risk_measure;
pub(crate) mod scaling_report;
pub mod setup;
pub mod simulation;
pub mod solver_phase;
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

// ── config ────────────────────────────────────────────────────────────────────
pub use config::TrainingConfig;
// ── convergence ───────────────────────────────────────────────────────────────
pub use convergence::ConvergenceMonitor;
// ── cut ───────────────────────────────────────────────────────────────────────
pub use cut::wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut};
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
pub use estimation::{EstimationPath, EstimationReport, estimate_from_history};
// ── forward ───────────────────────────────────────────────────────────────────
pub use forward::{SyncResult, build_cut_row_batch_into};
// ── hydro_models ──────────────────────────────────────────────────────────────
pub use hydro_models::{
    FphaHydroDetail, HydroModelSummary, PrepareHydroModelsResult, ProductionModelSource,
    build_hydro_model_summary, prepare_hydro_models,
};
// ── indexer ───────────────────────────────────────────────────────────────────
pub use indexer::{EquipmentCounts, FphaColumnLayout, StageIndexer};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
pub use lp_builder::{StageTemplates, build_stage_templates};
// ── policy_load ───────────────────────────────────────────────────────────────
pub use policy_load::{
    build_basis_cache_from_checkpoint, inject_boundary_cuts, load_boundary_cuts,
    validate_policy_compatibility,
};
// ── provenance ────────────────────────────────────────────────────────────────
pub use provenance::{
    HydroProductionProvenance, InflowProvenance, ModelProvenanceReport, ProvenanceSource,
    build_provenance_report,
};
// ── risk_measure ──────────────────────────────────────────────────────────────
pub use risk_measure::{BackwardOutcome, RiskMeasure};
// ── setup ─────────────────────────────────────────────────────────────────────
pub use setup::{
    DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, PrepareStochasticResult, StudyParams, StudySetup,
    prepare_stochastic,
};
// ── simulation ────────────────────────────────────────────────────────────────
pub use simulation::{
    ScenarioCategoryCosts, SimulationError, SimulationHydroResult, SimulationScenarioResult,
    SimulationStageResult, SimulationSummary, aggregate_simulation, simulate,
};
// ── solver_phase ─────────────────────────────────────────────────────────────
pub use solver_phase::Phase;
#[cfg(feature = "highs")]
pub use solver_phase::{BACKWARD_PROFILE, FORWARD_PROFILE, SIMULATION_PROFILE};
// ── solver_stats ──────────────────────────────────────────────────────────────
pub use solver_stats::{
    SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, pack_delta_scalars, pack_scenario_stats,
    unpack_delta_scalars, unpack_scenario_stats,
};
// ── stochastic_summary ────────────────────────────────────────────────────────
pub use stochastic_summary::{
    ArOrderSummary, StochasticSource, StochasticSummary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
// ── stopping_rule ─────────────────────────────────────────────────────────────
pub use stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};
// ── training ──────────────────────────────────────────────────────────────────
pub use training::{TrainingOutcome, TrainingResult, train};
// ── training_output ───────────────────────────────────────────────────────────
pub use training_output::build_training_output;
// ── resolved_parameters ───────────────────────────────────────────────────────
pub use resolved_parameters::{
    ResolvedParameters, ResolvedParametersError, build_resolved_parameters,
    deserialize_resolved_parameters, serialize_resolved_parameters,
};
// ── state_exchange ────────────────────────────────────────────────────────────
pub use state_exchange::ExchangeBuffers;
// ── trajectory ────────────────────────────────────────────────────────────────
pub use trajectory::TrajectoryRecord;
// ── workspace ─────────────────────────────────────────────────────────────────
pub use workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis};

/// Enable or disable the in-backward cut-selection hook.
///
/// Default is `false`. The hook installed inside the backward sweep
/// reads this toggle and skips execution unless enabled. Flip once at
/// process startup before training begins; the expected usage is a
/// single flip, not a runtime mode switch. Uses `Ordering::Relaxed`.
pub fn set_inside_backward_enabled(enabled: bool) {
    crate::backward_pass_state::IN_BACKWARD_ENABLED
        .store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Read the current value of the in-backward cut-selection hook toggle.
#[must_use]
pub fn is_inside_backward_enabled() -> bool {
    crate::backward_pass_state::IN_BACKWARD_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
