//! SDDP solver for hydrothermal dispatch.
//!
//! Implements the SDDP algorithm: forward/backward passes, Benders cuts, risk measures,
//! convergence monitoring, and policy simulation. Parallelized via rayon (intra-rank)
//! and ferrompi (inter-rank).

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

// `pub mod` modules are reached by qualified path from downstream crates or
// from integration tests in `tests/` (separate crates needing pub visibility);
// the `pub mod` namespaces are not a semver-stable API — prefer the curated
// re-exports below. `pub(crate)` modules are crate internals.
pub mod config;
pub mod convergence;
pub mod cut;
pub mod error;
pub(crate) mod gemm;
pub mod horizon_mode;
pub(crate) mod hull;
pub mod lead_time;
pub mod lp;
pub mod policy;
pub mod production;
pub mod setup;
pub mod simulation;
pub mod solve;
pub mod solver_stats;
pub mod stochastic;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod training;
pub mod validate_phases;
pub mod workspace;

// Re-export shim: exposes `workspace::context` at `cobre_sddp::context` for the
// `crate::context::` call sites.
pub use workspace::context;

// Re-export shims for `solve::solver_phase` / `solve::stage_solve` at their
// crate-root paths. `stage_solve` stays `pub(crate)` — no external consumer.
pub use solve::solver_phase;
pub(crate) use solve::stage_solve;

// Re-export shim: exposes `risk_measure` / `stopping_rule` at their crate-root
// paths for integration tests that import them by qualified path.
pub use convergence::{risk_measure, stopping_rule};

// Re-export shims exposing the `cut/` cluster modules at their crate-root paths
// for external (bench/test) and internal `crate::`-prefixed consumers.
pub use cut::{basis_reconstruct, cut_selection, cut_sync, dcs};

// Re-export shims exposing the `lp/` cluster at crate-root paths: `indexer` and
// `lp_builder` (alias of `lp::builder`) for external and internal consumers;
// `generic_constraints` stays `pub(crate)`.
pub use lp::builder as lp_builder;
pub(crate) use lp::generic_constraints;
pub use lp::indexer;

// Re-export shim exposing the `policy/` cluster modules at crate-root paths for
// raw-path callers the curated re-exports below do not cover.
pub use policy::{orchestration, policy_export, resolved_parameters, scaling_report};

// Re-export shim exposing the `production/` cluster modules at crate-root paths
// for raw-path callers the curated re-exports below do not cover.
// `hydro_models::prepare_hydro_models_from_artifacts` is intentionally absent
// from the curated re-export — this shim is its sole resolution path.
// `fpha_fitting` stays `pub(crate)`.
pub(crate) use production::fpha_fitting;
pub use production::{energy_conversion, hydro_models};

// Re-export shim exposing the `stochastic/` cluster modules at crate-root paths
// for raw-path callers the curated re-exports below do not cover.
// `lag_transition::precompute_stage_lag_transitions` is intentionally absent
// from the curated re-export — this shim is its sole resolution path. `noise`
// and `stochastic_summary` stay `pub(crate)`.
pub use stochastic::{estimation, inflow_method, lag_transition};
pub(crate) use stochastic::{noise, stochastic_summary};

// Re-export shim exposing the `training/` pass modules at crate-root paths for
// the internal `crate::`-prefixed references the curated re-exports below do not
// cover. `forward` and `lower_bound` stay `pub`; the rest are `pub(crate)`.
pub(crate) use training::{
    backward, backward_pass_state, forward_pass_state, rank_reconcile, state_exchange, trajectory,
    visited_states,
};
pub use training::{forward, lower_bound};

// Re-export shim: aliases `training::session` at `crate::training_session` for
// the `crate::training_session::` references in `training/forward_pass_state.rs`.
pub(crate) use training::session as training_session;

// ── config ────────────────────────────────────────────────────────────────────
pub use config::TrainingConfig;
// ── convergence ───────────────────────────────────────────────────────────────
pub use convergence::convergence::ConvergenceMonitor;
// ── cut ───────────────────────────────────────────────────────────────────────
pub use cut::wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut};
pub use cut::{CutPool, FutureCostFunction};
// ── cut_selection ─────────────────────────────────────────────────────────────
pub use cut::cut_selection::CutSelectionStrategy;
// ── cut_sync ──────────────────────────────────────────────────────────────────
pub use cut::cut_sync::CutSyncBuffers;
// ── cut::row ──────────────────────────────────────────────────────────────────
pub use cut::row::build_cut_row_batch_into;
// ── energy_conversion ─────────────────────────────────────────────────────────
pub use production::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
// ── error ─────────────────────────────────────────────────────────────────────
pub use error::SddpError;
// ── estimation ────────────────────────────────────────────────────────────────
pub use stochastic::estimation::{EstimationPath, EstimationReport, estimate_from_history};
// ── forward ───────────────────────────────────────────────────────────────────
pub use training::forward::SyncResult;
// ── hydro_models ──────────────────────────────────────────────────────────────
pub use production::hydro_models::{
    FphaFitDeviationEntry, FphaHydroDetail, HydroFitTimings, HydroModelSummary,
    PrepareHydroModelsResult, ProductionModelSource, build_deviation_summary,
    build_evaporation_model_rows, build_fpha_deviation_point_rows, build_hydro_model_summary,
    prepare_hydro_models,
};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use stochastic::inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
#[cfg(any(test, feature = "test-support"))]
pub use lp::builder::build_stage_templates_resolving_layout;
pub use lp::builder::{StageTemplates, build_stage_templates};
// ── policy_load ───────────────────────────────────────────────────────────────
pub use policy::policy_load::{
    CompatibilityReport, PolicyLoadKind, PolicyStageManifest, build_basis_cache_from_checkpoint,
    compare_manifest_slot_identity, inject_boundary_cuts, load_boundary_cuts, validate_policy_load,
};
// ── provenance ────────────────────────────────────────────────────────────────
pub use policy::provenance::{
    HydroProductionProvenance, InflowProvenance, ModelProvenanceReport, ProvenanceSource,
    build_provenance_report,
};
// ── risk_measure ──────────────────────────────────────────────────────────────
pub use convergence::risk_measure::{BackwardOutcome, RiskMeasure};
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
pub use solve::solver_phase::Phase;
#[cfg(feature = "highs")]
pub use solve::solver_phase::{BACKWARD_PROFILE, FORWARD_PROFILE, SIMULATION_PROFILE};
// ── solver_stats ──────────────────────────────────────────────────────────────
pub use solver_stats::{
    SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, SolverStatsLogEntry, delta_to_stats_row,
    pack_delta_scalars, pack_scenario_stats, solver_stats_log_to_rows, unpack_delta_scalars,
    unpack_scenario_stats,
};
// ── stochastic_summary ────────────────────────────────────────────────────────
pub use stochastic::stochastic_summary::{
    ArOrderSummary, StochasticSource, StochasticSummary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
// ── stopping_rule ─────────────────────────────────────────────────────────────
pub use convergence::stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};
// ── training ──────────────────────────────────────────────────────────────────
pub use training::training::{TrainingOutcome, TrainingResult, train};
// ── training_output ───────────────────────────────────────────────────────────
pub use training::training_output::build_training_output;
// ── resolved_parameters ───────────────────────────────────────────────────────
pub use policy::resolved_parameters::{
    ResolvedParameters, ResolvedParametersError, build_resolved_parameters,
    deserialize_resolved_parameters, serialize_resolved_parameters,
};
// ── state_exchange ────────────────────────────────────────────────────────────
pub use training::state_exchange::ExchangeBuffers;
// ── trajectory ────────────────────────────────────────────────────────────────
pub use training::trajectory::TrajectoryRecord;
// ── workspace ─────────────────────────────────────────────────────────────────
pub use workspace::workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis};
