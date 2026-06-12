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

// Module visibility policy:
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
pub mod convergence;
pub mod cut;
pub mod cut_selection;
pub mod cut_sync;
pub mod dcs;
pub mod error;
pub mod forward;
pub(crate) mod forward_pass_state;
pub(crate) mod gemm;
pub(crate) mod generic_constraints;
pub mod horizon_mode;
pub mod indexer;
pub mod lower_bound;
pub mod lp_builder;
pub mod policy;
pub mod production;
pub mod setup;
pub mod simulation;
pub mod solve;
pub mod solver_stats;
pub(crate) mod state_exchange;
pub mod stochastic;
pub mod training;
pub(crate) mod training_output;
pub(crate) mod training_session;
pub(crate) mod trajectory;
pub mod validate_phases;
pub(crate) mod visited_states;
pub mod workspace;

// Crate-root submodule shim: re-exposes the inner `context` file module (now
// `workspace::context` after the `workspace/` cluster move) at the pre-move
// `cobre_sddp::context` path, so the in-crate `crate::context::{StageContext,
// TrainingContext}` call sites resolve verbatim without per-site edits.
pub use workspace::context;

// Crate-root submodule shim: re-exposes the inner `solver_phase` / `stage_solve`
// file modules (now `solve::solver_phase` / `solve::stage_solve` after the
// `solve/` cluster move) at their pre-move crate-root paths, so the in-crate
// `crate::solver_phase::Phase` use in `simulation/state.rs` and the
// `crate::stage_solve::{fill_unscaled, fill_unscaled_dual, StageInputs,
// run_stage_solve}` uses in `forward.rs`, `backward.rs`, and
// `simulation/pipeline.rs` resolve verbatim without per-site edits.
// `stage_solve` keeps `pub(crate)` visibility — it has no external raw-path
// consumer.
pub use solve::solver_phase;
pub(crate) use solve::stage_solve;

// Crate-root submodule shim: preserves the pre-move
// `cobre_sddp::risk_measure::` / `cobre_sddp::stopping_rule::` raw paths
// verbatim for the integration tests in `tests/conformance.rs`, which import
// these submodules by qualified path.
pub use convergence::{risk_measure, stopping_rule};

// Crate-root submodule shim: preserves the pre-`policy/`-relocation raw
// `cobre_sddp::<module>::` / `crate::<module>::` paths verbatim so consumers
// resolve without edits. Each re-exported module has raw-path callers that the
// curated re-exports above do not cover:
//   - `orchestration` — production callers `write_checkpoint`,
//     `CheckpointParams`, `export_stochastic_artifacts` in
//     `cobre-cli/src/commands/run/{outputs,setup}.rs` and
//     `cobre-python/src/run.rs`.
//   - `policy_export` — `tests/{boundary_cuts,decomp_integration,warm_start}.rs`
//     plus the intra-crate `crate::policy_export::` use in `orchestration`.
//   - `resolved_parameters` — `crate::resolved_parameters::` paths in
//     `lp_builder/{layout,patch,matrix,template}.rs` and `setup`.
//   - `scaling_report` — `crate::scaling_report::` paths in
//     `setup/{template_postprocess,mod}.rs`.
pub use policy::{orchestration, policy_export, resolved_parameters, scaling_report};

// Crate-root submodule shim: preserves the pre-`production/`-relocation raw
// `cobre_sddp::<module>::` / `crate::<module>::` paths verbatim so consumers
// resolve without edits. Each re-exported module has raw-path callers that the
// curated re-exports above do not cover:
//   - `energy_conversion` — `cobre_sddp::energy_conversion::` in the integration
//     tests `tests/{scalar_parameters_declaration_order,simulation_pipeline_integration}.rs`,
//     plus intra-crate `crate::energy_conversion::` uses.
//   - `hydro_models` — `cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts`
//     in `cobre-cli/src/commands/{run/setup,validate}.rs` and
//     `cobre-python/src/{io,run}.rs` (this symbol is intentionally NOT in the
//     curated re-export above; the shim is its sole resolution path), plus
//     intra-crate `crate::hydro_models::` uses.
//   - `fpha_fitting` — `crate::fpha_fitting::FphaFittingError` in the non-moved
//     `error.rs` plus intra-cluster uses in `hydro_models` and
//     `energy_conversion`; `pub(crate)` keeps its crate-private visibility.
pub(crate) use production::fpha_fitting;
pub use production::{energy_conversion, hydro_models};

// Crate-root submodule shim: preserves the pre-`stochastic/`-relocation raw
// `cobre_sddp::<module>::` / `crate::<module>::` paths verbatim so consumers
// resolve without edits. Every one of the six modules has raw-path callers in
// non-moved files that the curated re-exports below do not cover:
//   - `estimation` — `crate::estimation::` in `error.rs`,
//     `policy/{orchestration,provenance}.rs`, and
//     `setup/stochastic_pipeline.rs`.
//   - `inflow_method` — `crate::inflow_method::` in `indexer.rs`, `forward.rs`,
//     and `lp_builder/{template,matrix}.rs`, `simulation/pipeline.rs`.
//   - `lag_transition` — `cobre_sddp::lag_transition::precompute_stage_lag_transitions`
//     in `cobre-cli/src/commands/run/setup.rs` (this symbol is intentionally NOT
//     in the curated re-export; the shim is its sole resolution path), plus
//     `crate::lag_transition::` uses in `workspace/context.rs` and
//     `setup/{stochastic_pipeline,stage_data,mod}.rs`.
//   - `noise_key_diag` — `crate::noise_key_diag::` in `setup/mod.rs`.
//   - `noise` — `crate::noise::` in `workspace/context.rs`, `forward.rs`, and
//     `simulation/pipeline.rs`; `pub(crate)` keeps its crate-private visibility.
//   - `stochastic_summary` — `crate::stochastic_summary::` in
//     `policy/orchestration.rs`; `pub(crate)` keeps its crate-private visibility.
pub use stochastic::{estimation, inflow_method, lag_transition, noise_key_diag};
pub(crate) use stochastic::{noise, stochastic_summary};

// ── config ────────────────────────────────────────────────────────────────────
pub use config::TrainingConfig;
// ── convergence ───────────────────────────────────────────────────────────────
pub use convergence::convergence::ConvergenceMonitor;
// ── cut ───────────────────────────────────────────────────────────────────────
pub use cut::wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut};
pub use cut::{CutPool, FutureCostFunction};
// ── cut_selection ─────────────────────────────────────────────────────────────
pub use cut_selection::CutSelectionStrategy;
// ── cut_sync ──────────────────────────────────────────────────────────────────
pub use cut_sync::CutSyncBuffers;
// ── cut::row ──────────────────────────────────────────────────────────────────
pub use cut::row::build_cut_row_batch_into;
// ── energy_conversion ─────────────────────────────────────────────────────────
pub use production::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
// ── error ─────────────────────────────────────────────────────────────────────
pub use error::SddpError;
// ── estimation ────────────────────────────────────────────────────────────────
pub use stochastic::estimation::{EstimationPath, EstimationReport, estimate_from_history};
// ── forward ───────────────────────────────────────────────────────────────────
pub use forward::SyncResult;
// ── hydro_models ──────────────────────────────────────────────────────────────
pub use production::hydro_models::{
    FphaHydroDetail, HydroModelSummary, PrepareHydroModelsResult, ProductionModelSource,
    build_hydro_model_summary, prepare_hydro_models,
};
// ── indexer ───────────────────────────────────────────────────────────────────
pub use indexer::{EquipmentCounts, FphaColumnLayout, StageIndexer};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use stochastic::inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
pub use lp_builder::{StageTemplates, build_stage_templates};
// ── policy_load ───────────────────────────────────────────────────────────────
pub use policy::policy_load::{
    build_basis_cache_from_checkpoint, inject_boundary_cuts, load_boundary_cuts,
    validate_policy_compatibility,
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
pub use training::{TrainingOutcome, TrainingResult, train};
// ── training_output ───────────────────────────────────────────────────────────
pub use training_output::build_training_output;
// ── resolved_parameters ───────────────────────────────────────────────────────
pub use policy::resolved_parameters::{
    ResolvedParameters, ResolvedParametersError, build_resolved_parameters,
    deserialize_resolved_parameters, serialize_resolved_parameters,
};
// ── state_exchange ────────────────────────────────────────────────────────────
pub use state_exchange::ExchangeBuffers;
// ── trajectory ────────────────────────────────────────────────────────────────
pub use trajectory::TrajectoryRecord;
// ── workspace ─────────────────────────────────────────────────────────────────
pub use workspace::workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis};
