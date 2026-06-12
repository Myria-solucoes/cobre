//! Output-writing phase for `cobre run`.
//!
//! Owns the training and simulation output writers and their argument carriers.
//! These functions are the Python-parity surface mirrored against
//! `crates/cobre-python/src/run.rs`.

use std::path::Path;

use console::Term;

use cobre_core::System;
use cobre_sddp::StudySetup;

use crate::error::CliError;

/// Arguments for [`write_training_outputs`].
pub(super) struct WriteTrainingArgs<'a> {
    pub(super) output_dir: &'a Path,
    pub(super) system: &'a System,
    pub(super) config: &'a cobre_io::Config,
    pub(super) training_output: &'a cobre_io::TrainingOutput,
    pub(super) setup: &'a StudySetup,
    pub(super) training_result: &'a cobre_sddp::TrainingResult,
    pub(super) output_ctx: &'a cobre_io::OutputContext,
    pub(super) hydro_models: &'a cobre_sddp::PrepareHydroModelsResult,
    pub(super) quiet: bool,
    pub(super) stderr: &'a Term,
}

/// Write training artifacts: policy checkpoint, training results, solver stats,
/// and row-selection records. Called immediately after training completes, before
/// simulation starts.
pub(super) fn write_training_outputs(args: &WriteTrainingArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing training outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    let policy_dir = args.output_dir.join(&args.setup.policy_path);
    cobre_sddp::orchestration::write_checkpoint(
        &policy_dir,
        &args.setup.fcf,
        args.training_result,
        &cobre_sddp::orchestration::CheckpointParams {
            max_iterations: args.setup.loop_params.max_iterations,
            forward_passes: args.setup.loop_params.forward_passes,
            seed: args.setup.loop_params.seed,
            export_states: args.config.exports.states,
        },
    )
    .map_err(CliError::from)?;

    cobre_io::write_training_results(
        args.output_dir,
        args.training_output,
        args.system,
        args.config,
        args.output_ctx,
    )
    .map_err(CliError::from)?;

    if !args.hydro_models.fpha_export_rows.is_empty() {
        let fpha_path = args
            .output_dir
            .join("hydro_models")
            .join("fpha_hyperplanes.parquet");
        cobre_io::output::write_fpha_hyperplanes(&fpha_path, &args.hydro_models.fpha_export_rows)
            .map_err(CliError::from)?;
    }

    // Write training solver stats to training/solver/iterations.parquet.
    if !args.training_result.solver_stats_log.is_empty() {
        let rows = cobre_sddp::solver_stats_log_to_rows(&args.training_result.solver_stats_log);
        cobre_io::write_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    // Write per-stage row-selection records to training/cut_selection/iterations.parquet.
    if !args.training_output.cut_selection_records.is_empty() {
        let parquet_config = cobre_io::ParquetWriterConfig::default();
        cobre_io::write_row_selection_records(
            args.output_dir,
            &args.training_output.cut_selection_records,
            &parquet_config,
        )
        .map_err(CliError::from)?;
    }

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        crate::summary::print_output_path(args.stderr, args.output_dir, write_secs);
    }

    Ok(())
}

/// Arguments for [`write_simulation_outputs`].
pub(super) struct WriteSimulationArgs<'a> {
    pub(super) output_dir: &'a Path,
    pub(super) sim_output: &'a cobre_io::SimulationOutput,
    pub(super) sim_solver_stats: &'a [(u32, cobre_sddp::SolverStatsDelta)],
    pub(super) output_ctx: &'a cobre_io::OutputContext,
    pub(super) quiet: bool,
    pub(super) stderr: &'a Term,
}

/// Write simulation artifacts: simulation results manifest and solver stats.
/// Called after simulation completes.
pub(super) fn write_simulation_outputs(args: &WriteSimulationArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing simulation outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    cobre_io::write_simulation_results(args.output_dir, args.sim_output, args.output_ctx)
        .map_err(CliError::from)?;

    // Write simulation solver stats to simulation/solver/iterations.parquet.
    // Simulation has no opening dimension and no per-worker dimension yet;
    // opening, rank, and worker_id are all None.
    if !args.sim_solver_stats.is_empty() {
        let rows: Vec<cobre_io::SolverStatsRow> = args
            .sim_solver_stats
            .iter()
            .map(|(scenario_id, delta)| {
                cobre_sddp::delta_to_stats_row(
                    *scenario_id,
                    "simulation",
                    -1,
                    None,
                    None,
                    None,
                    delta,
                )
            })
            .collect();
        cobre_io::write_simulation_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

    if !args.quiet {
        let write_secs = write_start.elapsed().as_secs_f64();
        crate::summary::print_output_path(args.stderr, args.output_dir, write_secs);
    }

    Ok(())
}
