//! Shared output-writing helpers consumed by both the CLI binary and the
//! Python bindings.
//!
//! Before this module existed, the CLI (`crates/cobre-cli/src/policy_io.rs`
//! and `crates/cobre-cli/src/commands/run.rs::export_stochastic_artifacts`)
//! and the Python bindings (`crates/cobre-python/src/run.rs`) maintained two
//! near-identical implementations of the policy-checkpoint writer and the
//! stochastic-artifacts exporter. The duplication was the most likely source
//! of CLI/Python parity drift in future schema changes (CLAUDE.md treats
//! Python parity as non-negotiable).
//!
//! Both functions now live here so there is exactly one implementation. The
//! callers thread their own diagnostics (CLI uses `console::Term`; Python
//! uses `eprintln!`) via the `on_warning` callback on
//! [`export_stochastic_artifacts`].

use std::path::Path;

use cobre_io::output::policy::{PolicyCheckpointMetadata, write_policy_checkpoint};
use cobre_io::output::{
    OutputError, write_correlation_json, write_fitting_report, write_inflow_annual_component,
    write_inflow_ar_coefficients, write_inflow_seasonal_stats, write_load_seasonal_stats,
    write_noise_openings,
};
use cobre_io::scenarios::LoadSeasonalStatsRow;
use cobre_stochastic::StochasticContext;

use crate::estimation::EstimationReport;
use crate::policy_export::{
    build_active_indices, build_stage_basis_records, build_stage_cut_records,
    build_stage_cuts_payloads, build_stage_states_payloads, convert_basis_cache,
};
use crate::stochastic_summary::{
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
use crate::{FutureCostFunction, TrainingResult};

use cobre_core::System;
use cobre_core::scenario::LoadModel;

// ── Policy checkpoint ─────────────────────────────────────────────────────────

/// Inputs to [`write_checkpoint`] that depend on the surrounding run rather
/// than on the training result itself.
///
/// Construct one of these from the caller's parsed config; the field types
/// match the underlying `PolicyCheckpointMetadata` widths to keep the cast
/// site here.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointParams {
    /// Maximum iteration count from the active stopping rules. Stored in
    /// the checkpoint metadata so subsequent runs can resume with matching
    /// limits.
    pub max_iterations: u64,
    /// Number of forward passes per iteration. Stored for resume-validation.
    pub forward_passes: u32,
    /// Random seed used for noise generation. Stored so resumed runs are
    /// reproducible.
    pub seed: u64,
    /// When `false`, the writer omits visited-state payloads even if the
    /// archive is populated. Controlled by the `exports.states` config flag.
    pub export_states: bool,
}

/// Write the trained policy (cuts, bases, visited states, metadata) to
/// `policy_dir` as `FlatBuffers` files.
///
/// This is the single implementation shared by the CLI and the Python
/// bindings — both call this function so the on-disk format and write
/// ordering cannot drift between them.
///
/// # Errors
///
/// Propagates [`OutputError`] from
/// [`cobre_io::output::policy::write_policy_checkpoint`] if any of the
/// `FlatBuffers` files cannot be written.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn write_checkpoint(
    policy_dir: &Path,
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    params: &CheckpointParams,
) -> Result<(), OutputError> {
    let n_stages = fcf.pools.len();
    let state_dimension = fcf.state_dimension;

    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(training_result);
    let stage_bases = build_stage_basis_records(fcf, training_result, &basis_col_u8, &basis_row_u8);

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
    let metadata = PolicyCheckpointMetadata {
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: cobre_io::now_iso8601(),
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dimension as u32,
        num_stages: n_stages as u32,
        max_iterations: params.max_iterations as u32,
        forward_passes: params.forward_passes,
        warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
        warm_start_counts,
        rng_seed: params.seed,
        total_visited_states: training_result
            .visited_archive
            .as_ref()
            .map_or(0, |a| (0..a.num_stages()).map(|t| a.count(t) as u64).sum()),
    };

    let stage_states = if params.export_states {
        build_stage_states_payloads(training_result.visited_archive.as_ref())
    } else {
        Vec::new()
    };

    write_policy_checkpoint(
        policy_dir,
        &stage_cuts,
        &stage_bases,
        &metadata,
        &stage_states,
    )
}

// ── Stochastic artifacts ──────────────────────────────────────────────────────

/// Write all applicable stochastic preprocessing artifacts to
/// `{output_dir}/stochastic/`.
///
/// Called when `exports.stochastic` is `true` in `config.json`. Each writer
/// invocation is independent: a failure produces a one-line message routed
/// through the caller-supplied `on_warning` callback and does not prevent
/// the remaining files (or training) from proceeding.
///
/// Files written:
/// - `noise_openings.parquet` — always
/// - `inflow_seasonal_stats.parquet` — always
/// - `inflow_ar_coefficients.parquet` — always
/// - `inflow_annual_component.parquet` — always
/// - `correlation.json` — always
/// - `load_seasonal_stats.parquet` — only when any load model has `std_mw > 0`
/// - `fitting_report.json` — only when `estimation_report` is `Some`
///
/// The CLI and the Python bindings call this with their own warning sinks
/// (`console::Term::write_line` and `eprintln!` respectively); the writer
/// itself never touches stderr.
pub fn export_stochastic_artifacts(
    output_dir: &Path,
    stochastic: &StochasticContext,
    system: &System,
    estimation_report: Option<&EstimationReport>,
    on_warning: &mut dyn FnMut(&str),
) {
    let stochastic_dir = output_dir.join("stochastic");

    if let Err(e) = write_noise_openings(
        &stochastic_dir.join("noise_openings.parquet"),
        stochastic.opening_tree(),
    ) {
        on_warning(&format!("noise_openings: {e}"));
    }

    let stats_rows = inflow_models_to_stats_rows(system.inflow_models());
    if let Err(e) = write_inflow_seasonal_stats(
        &stochastic_dir.join("inflow_seasonal_stats.parquet"),
        &stats_rows,
    ) {
        on_warning(&format!("inflow_seasonal_stats: {e}"));
    }

    let ar_rows = inflow_models_to_ar_rows(system.inflow_models());
    if let Err(e) = write_inflow_ar_coefficients(
        &stochastic_dir.join("inflow_ar_coefficients.parquet"),
        &ar_rows,
    ) {
        on_warning(&format!("inflow_ar_coefficients: {e}"));
    }

    let annual_rows = inflow_models_to_annual_component_rows(system.inflow_models());
    if let Err(e) = write_inflow_annual_component(
        &stochastic_dir.join("inflow_annual_component.parquet"),
        &annual_rows,
    ) {
        on_warning(&format!("inflow_annual_component: {e}"));
    }

    if let Err(e) = write_correlation_json(
        &stochastic_dir.join("correlation.json"),
        system.correlation(),
    ) {
        on_warning(&format!("correlation: {e}"));
    }

    let has_stochastic_load = system
        .load_models()
        .iter()
        .any(|m: &LoadModel| m.std_mw > 0.0);
    if has_stochastic_load {
        let load_rows: Vec<LoadSeasonalStatsRow> = system
            .load_models()
            .iter()
            .map(|m| LoadSeasonalStatsRow {
                bus_id: m.bus_id,
                stage_id: m.stage_id,
                mean_mw: m.mean_mw,
                std_mw: m.std_mw,
            })
            .collect();
        if let Err(e) = write_load_seasonal_stats(
            &stochastic_dir.join("load_seasonal_stats.parquet"),
            &load_rows,
        ) {
            on_warning(&format!("load_seasonal_stats: {e}"));
        }
    }

    if let Some(report) = estimation_report {
        let fitting = estimation_report_to_fitting_report(report);
        if let Err(e) = write_fitting_report(&stochastic_dir.join("fitting_report.json"), &fitting)
        {
            on_warning(&format!("fitting_report: {e}"));
        }
    }
}
