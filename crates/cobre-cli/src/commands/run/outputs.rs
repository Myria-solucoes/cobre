//! Output-writing phase for `cobre run`.
//!
//! Python-parity contract: every file written here must also be written by
//! `cobre-python`'s `run.rs`; the two writers must stay mirrored.

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

/// Write the training artifacts (policy checkpoint, results, sidecars, stats).
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

    // No evaporation-modeled hydro writes no file (FPHA "if-any" behavior);
    // mirror on the Python side: `write_evaporation_models_if_any`.
    let evaporation_rows = cobre_sddp::build_evaporation_model_rows(args.hydro_models, args.system);
    if !evaporation_rows.is_empty() {
        let evaporation_path = args
            .output_dir
            .join("hydro_models")
            .join("evaporation_models.parquet");
        cobre_io::output::write_evaporation_models(&evaporation_path, &evaporation_rows)
            .map_err(CliError::from)?;
    }

    // Off by default, so a default run writes no file and stays byte-identical;
    // mirror on the Python side: `write_fpha_deviation_points_if_any`.
    if args.config.exports.fpha_deviation_points {
        let deviation_point_rows = cobre_sddp::build_fpha_deviation_point_rows(args.hydro_models);
        if !deviation_point_rows.is_empty() {
            let deviation_points_path = args
                .output_dir
                .join("hydro_models")
                .join("fpha_deviation_points.parquet");
            cobre_io::output::write_fpha_deviation_points(
                &deviation_points_path,
                deviation_point_rows,
            )
            .map_err(CliError::from)?;
        }
    }

    if !args.training_result.solver_stats_log.is_empty() {
        let rows = cobre_sddp::solver_stats_log_to_rows(&args.training_result.solver_stats_log);
        cobre_io::write_solver_stats(args.output_dir, &rows).map_err(CliError::from)?;
    }

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

/// Write the simulation artifacts (results manifest and solver stats).
pub(super) fn write_simulation_outputs(args: &WriteSimulationArgs<'_>) -> Result<(), CliError> {
    if !args.quiet {
        use std::io::Write;
        let _ = args.stderr.write_line("Writing simulation outputs...");
        let _ = std::io::stderr().flush();
    }
    let write_start = std::time::Instant::now();

    cobre_io::write_simulation_results(args.output_dir, args.sim_output, args.output_ctx)
        .map_err(CliError::from)?;

    // Simulation has no opening/rank/worker dimension; those fields are all None.
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
