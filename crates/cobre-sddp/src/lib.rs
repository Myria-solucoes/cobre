//! SDDP solver for hydrothermal dispatch.
//!
//! Implements the SDDP algorithm: forward/backward passes, Benders cuts, risk measures,
//! convergence monitoring, and policy simulation. Parallelized via rayon (intra-rank)
//! and ferrompi (inter-rank).

// Relax strict production lints for test builds.
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
pub mod config;
pub mod convergence;
pub mod cut;
pub mod error;
pub(crate) mod gemm;
pub mod horizon_mode;
pub mod lp;
pub mod policy;
pub mod production;
pub mod setup;
pub mod simulation;
pub mod solve;
pub mod solver_stats;
pub mod stochastic;
pub mod training;
pub mod validate_phases;
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
// run_stage_solve}` uses in `training/forward.rs`, `training/backward.rs`, and
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

// Crate-root submodule shims for the `cut/` cluster move (`cut_selection`,
// `cut_sync`, `dcs`, `basis_reconstruct` were flat at the crate root; they now
// live under `cut/`). Each shim re-exposes a moved submodule at its pre-move
// crate-root path so consumers resolve verbatim without per-site edits:
//
//   - `cut_selection` — `cobre_sddp::cut_selection::` in `benches/cut_selection_kernel.rs`
//     and the integration tests `tests/{cut_selection_kernel_perf,cut_selection_determinism_realistic}.rs`,
//     the internal `crate::cut_selection::` call sites across `training/backward.rs`,
//     `training/backward_pass_state.rs`, `config.rs`, `cut/pool.rs`, `cut/dcs.rs`,
//     `training/forward.rs`, `simulation/pipeline.rs`, `training/training.rs`, and
//     `training/session/mod.rs`, plus the `crate::cut_selection::CutMetadata`
//     intra-doc links in `cut/mod.rs`.
//   - `cut_sync` — the internal `crate::cut_sync::CutSyncBuffers` references
//     (reached via grouped `use crate::{ … }` imports) in `training/backward.rs`,
//     `training/backward_pass_state.rs`, and `training/session/mod.rs`.
//   - `dcs` — `cobre_sddp::dcs::` in `benches/dcs_batched_scoring.rs` plus the
//     internal `crate::dcs::` call sites across `training/backward.rs`,
//     `training/forward.rs`, `setup/{accessors,mod}.rs`, `simulation/pipeline.rs`,
//     and `workspace/workspace.rs`.
//   - `basis_reconstruct` — `cobre_sddp::basis_reconstruct::` in
//     `tests/hybrid_reconstruction.rs` (plus the doc-comment intra-doc links in
//     `tests/basis_reconstruct_churn.rs` and `cobre-python/src/run.rs`), and the
//     internal `crate::basis_reconstruct::` call sites in `cut/dcs.rs`,
//     `training/forward.rs`, and `workspace/workspace.rs`.
pub use cut::{basis_reconstruct, cut_selection, cut_sync, dcs};

// Crate-root submodule shims for the `lp/` cluster move (`indexer` /
// `generic_constraints` / `lp_builder` were flat at the crate root; they now
// live under `lp/`). Each shim re-exposes a moved submodule at its pre-move
// crate-root path so consumers resolve verbatim without per-site edits:
//
// - `indexer` — covers the external raw-path consumers (`cobre_sddp::indexer::`
//   in 4 integration tests + 2 benches) AND the internal `crate::indexer::`
//   call sites across 13 files (including `lp/generic_constraints.rs` itself).
// - `lp_builder` (alias of `lp::builder`) — keeps the 27 internal
//   `crate::lp_builder::Symbol` references across 11 files resolving to the
//   moved subtree without editing those files.
// - `generic_constraints` — keeps `pub(crate)` visibility; covers the internal
//   `crate::generic_constraints::` references in `lp/builder/matrix.rs` and
//   `lp/builder/layout.rs`, which moved into the cluster and still reach the
//   constraint-lowering module by its pre-move crate-root path.
pub use lp::builder as lp_builder;
pub(crate) use lp::generic_constraints;
pub use lp::indexer;

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
//   - `inflow_method` — `crate::inflow_method::` in `indexer.rs`, `training/forward.rs`,
//     and `lp_builder/{template,matrix}.rs`, `simulation/pipeline.rs`.
//   - `lag_transition` — `cobre_sddp::lag_transition::precompute_stage_lag_transitions`
//     in `cobre-cli/src/commands/run/setup.rs` (this symbol is intentionally NOT
//     in the curated re-export; the shim is its sole resolution path), plus
//     `crate::lag_transition::` uses in `workspace/context.rs` and
//     `setup/{stochastic_pipeline,stage_data,mod}.rs`.
//   - `noise_key_diag` — `crate::noise_key_diag::` in `setup/mod.rs`.
//   - `noise` — `crate::noise::` in `workspace/context.rs`, `training/forward.rs`, and
//     `simulation/pipeline.rs`; `pub(crate)` keeps its crate-private visibility.
//   - `stochastic_summary` — `crate::stochastic_summary::` in
//     `policy/orchestration.rs`; `pub(crate)` keeps its crate-private visibility.
pub use stochastic::{estimation, inflow_method, lag_transition, noise_key_diag};
pub(crate) use stochastic::{noise, stochastic_summary};

// Crate-root submodule shim: preserves the pre-`training/`-relocation raw
// `crate::<module>::` paths verbatim for the internal references that the
// curated re-exports below do not cover, so the consumer files resolve without
// per-site edits. The pass files moved into `training/`; their pre-move
// crate-root paths are re-exposed here:
//   - `forward` — `crate::forward::{ForwardResult, SyncResult}` in
//     `training/session/mod.rs`, the `crate::forward::sync_forward` use there and
//     its intra-doc links in `convergence/convergence.rs`, and the
//     `crate::forward::build_delta_cut_row_batch_into` intra-doc link in
//     `cut/row.rs`. Keeps `pub` to match its pre-move visibility.
//   - `backward` — `crate::backward::StagedCut` in `workspace/workspace.rs`,
//     `crate::backward::BackwardResult` in `training/session/mod.rs`, and the
//     `crate::backward::extract_duals_from_view` intra-doc link in `cut/row.rs`.
//   - `backward_pass_state` — `crate::backward_pass_state::BackwardPass{State,Inputs}`
//     in `training/backward.rs` and the doc reference in `cut/cut_selection.rs`.
//   - `forward_pass_state` — `crate::forward_pass_state::{ForwardPassInputs,
//     ForwardPassState}` in `training/forward.rs`.
//   - `lower_bound` — `crate::lower_bound::evaluate_lower_bound` intra-doc links
//     in `convergence/convergence.rs`. Keeps `pub` to match its pre-move visibility.
//   - `state_exchange` — no current `crate::state_exchange::` consumer; re-exported
//     for symmetry with the other pass modules.
//   - `trajectory` — `crate::trajectory::TrajectoryRecord` in `training/forward.rs`
//     and `training/state_exchange.rs`.
//   - `visited_states` — `crate::visited_states::VisitedStatesArchive` in
//     `training/training.rs`, `training/session/mod.rs`, and `policy/policy_export.rs`.
pub(crate) use training::{
    backward, backward_pass_state, forward_pass_state, state_exchange, trajectory, visited_states,
};
pub use training::{forward, lower_bound};

// Crate-root submodule shim: aliases the nested `training/session/` subtree at
// the pre-move `crate::training_session::` path so the
// `crate::training_session::{iteration_scratch, rank_distribution, runtime}`
// references in `training/forward_pass_state.rs` resolve verbatim.
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
    FphaHydroDetail, HydroModelSummary, PrepareHydroModelsResult, ProductionModelSource,
    build_hydro_model_summary, prepare_hydro_models,
};
// ── indexer ───────────────────────────────────────────────────────────────────
pub use lp::indexer::{EquipmentCounts, FphaColumnLayout, StageIndexer};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use stochastic::inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
pub use lp::builder::{StageTemplates, build_stage_templates};
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
