//! Shared output-writing helpers consumed by both the CLI binary and the
//! Python bindings.
//!
//! Both front ends call this one implementation of the policy-checkpoint writer
//! and the stochastic-artifacts exporter, so their outputs cannot drift (Python
//! parity is a hard rule). Callers thread their own diagnostics via the
//! `on_warning` callback on [`export_stochastic_artifacts`].

use std::path::Path;

use cobre_io::EntitySlot;
use cobre_io::output::policy::{
    FORMAT_VERSION, PolicyCheckpointMetadata, ProducerBlock, write_policy_checkpoint,
};
use cobre_io::output::{
    OutputError, write_correlation_json, write_fitting_report, write_inflow_annual_component,
    write_inflow_ar_coefficients, write_inflow_seasonal_stats, write_load_seasonal_stats,
    write_noise_openings,
};
use cobre_io::scenarios::LoadSeasonalStatsRow;
use cobre_io::scenarios::estimation::EstimationReport;
use cobre_stochastic::StochasticContext;

use crate::TrainingResult;
use crate::policy_export::{
    borrow_cut_records, build_active_indices, build_stage_basis_records, build_stage_cut_records,
    build_stage_cuts_payloads, build_stage_entity_manifest, build_stage_states_payloads,
    convert_basis_cache, scale_cut_records_for_export,
};
use crate::setup::{NodePos, StudySetup};
use crate::stochastic_summary::{
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};

use cobre_core::{BlockMode, System};

// ── Policy checkpoint ─────────────────────────────────────────────────────────

fn block_mode_label(mode: BlockMode) -> &'static str {
    match mode {
        BlockMode::Parallel => "parallel",
        BlockMode::Chronological => "chronological",
    }
}

fn training_block_provenance(modes: &[BlockMode]) -> (String, Vec<String>) {
    match modes.first() {
        None => (String::new(), Vec::new()),
        Some(&first) if modes.iter().all(|&m| m == first) => {
            (block_mode_label(first).to_string(), Vec::new())
        }
        Some(_) => (
            "mixed".to_string(),
            modes
                .iter()
                .map(|&m| block_mode_label(m).to_string())
                .collect(),
        ),
    }
}

/// Run-derived inputs to [`write_checkpoint`] (independent of the training
/// result). Stored in the checkpoint metadata for resume validation and
/// reproducibility; field types match `PolicyCheckpointMetadata` widths to keep
/// the casts at this site.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointParams {
    /// Maximum iteration count from the active stopping rules.
    pub max_iterations: u64,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Random seed used for noise generation.
    pub seed: u64,
    /// When `false`, omit visited-state payloads even if the archive is
    /// populated. Controlled by the `exports.states` config flag.
    pub export_states: bool,
}

/// Write the trained policy (cuts, bases, visited states, metadata) to
/// `policy_dir` as `FlatBuffers` files.
///
/// This is the single implementation shared by the CLI and the Python
/// bindings — both call this function so the on-disk format and write
/// ordering (including the per-slot entity manifest) cannot drift between them.
///
/// `system` is passed explicitly because [`StudySetup`] does not own it.
///
/// # Errors
///
/// Propagates [`OutputError`] from
/// [`cobre_io::output::policy::write_policy_checkpoint`] if any of the
/// `FlatBuffers` files cannot be written.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn write_checkpoint(
    policy_dir: &Path,
    setup: &StudySetup,
    system: &System,
    training_result: &TrainingResult,
    params: &CheckpointParams,
) -> Result<(), OutputError> {
    let fcf = &setup.fcf;
    // `n_pools` sizes the pool-indexed vectors below (`fcf.pools`,
    // `cut_state_layouts`, `stage_manifests`); `n_stages` (the true study stage
    // count, from `setup.num_stages()` — NOT `fcf.pools.len()`, which counts
    // pools, equal to the stage count only on the chain degeneracy) is the
    // checkpoint metadata's own field.
    let n_pools = fcf.pools.len();
    let n_stages = setup.num_stages();

    let global_layout = setup.stage_state();
    let stage_manifests: Vec<Vec<EntitySlot>> = (0..n_pools)
        .map(|p| {
            // `p` is a pool ordinal; its owning stage resolves through
            // `pool_stage` — indexing `study_stage_ids` by `p` is OOB once
            // `n_pools > n_stages` on a branching graph (see `NodeGraph::pool_stage`).
            let stage_id = setup.study_stage_ids[setup.node_graph.pool_stage[p].0];
            build_stage_entity_manifest(
                system,
                global_layout,
                &setup.stage_data.cut_state_layouts[p],
                stage_id,
            )
        })
        .collect();

    let cost_scale_factor = setup.stage_data.stage_templates.cost_scale_factor;
    let stage_records_internal = build_stage_cut_records(fcf);
    let stage_records_owned =
        scale_cut_records_for_export(&stage_records_internal, cost_scale_factor);
    let stage_records = borrow_cut_records(&stage_records_owned);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts =
        build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices, &stage_manifests);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(training_result);
    let stage_bases = build_stage_basis_records(
        fcf,
        training_result,
        &setup.node_graph,
        &basis_col_u8,
        &basis_row_u8,
    );

    let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();

    let study_modes: Vec<BlockMode> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.block_mode)
        .collect();
    let (training_block_mode, training_block_mode_per_stage) =
        training_block_provenance(&study_modes);

    let metadata = PolicyCheckpointMetadata {
        format_version: FORMAT_VERSION,
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: cobre_io::now_iso8601(),
        num_stages: n_stages as u32,
        graph_manifest: setup.build_graph_manifest(),
        producer: ProducerBlock {
            completed_iterations: training_result.iterations as u32,
            final_lower_bound: training_result.final_lb,
            best_upper_bound: Some(training_result.final_ub),
            max_iterations: params.max_iterations as u32,
            forward_passes: params.forward_passes,
            warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
            warm_start_counts,
            rng_seed: params.seed,
            total_visited_states: training_result.visited_archive.as_ref().map_or(0, |a| {
                (0..a.num_nodes()).map(|t| a.count(NodePos(t)) as u64).sum()
            }),
            training_block_mode,
            training_block_mode_per_stage,
            cost_scale_factor: Some(setup.stage_data.stage_templates.cost_scale_factor),
        },
    };

    let stage_states = if params.export_states {
        build_stage_states_payloads(
            training_result.visited_archive.as_ref(),
            &stage_manifests,
            &setup.node_graph,
        )
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

    let has_stochastic_load = system.load_models().iter().any(|m| m.std_mw > 0.0);
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

#[cfg(test)]
mod tests {
    use super::{BlockMode, training_block_provenance};

    #[test]
    fn uniform_parallel_study_summarizes_without_per_stage_list() {
        let (summary, per_stage) =
            training_block_provenance(&[BlockMode::Parallel, BlockMode::Parallel]);
        assert_eq!(summary, "parallel");
        assert!(per_stage.is_empty());
    }

    #[test]
    fn uniform_chronological_study_summarizes_without_per_stage_list() {
        let (summary, per_stage) =
            training_block_provenance(&[BlockMode::Chronological, BlockMode::Chronological]);
        assert_eq!(summary, "chronological");
        assert!(per_stage.is_empty());
    }

    #[test]
    fn mixed_study_reports_mixed_summary_and_full_per_stage_list() {
        let (summary, per_stage) = training_block_provenance(&[
            BlockMode::Parallel,
            BlockMode::Chronological,
            BlockMode::Parallel,
        ]);
        assert_eq!(summary, "mixed");
        assert_eq!(per_stage, vec!["parallel", "chronological", "parallel"]);
    }
}
