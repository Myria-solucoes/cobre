//! Case-load, communicator setup, broadcast, and pre-training phases for `cobre run`.
//!
//! Rank 0 loads from disk; system and config are broadcast to all ranks, which
//! then build `StudySetup` from the shared data.

use std::path::{Path, PathBuf};

use console::Term;

use cobre_comm::{Communicator, TopologyProvider, create_communicator};
use cobre_core::System;
use cobre_sddp::{
    EstimationReport, PrepareHydroModelsResult, PrepareStochasticResult, StudySetup,
    build_hydro_model_summary, prepare_hydro_models, prepare_stochastic,
    setup::{ConstructionConfig, build_ncs_factor_entries, load_load_factors_for_stochastic},
};
use cobre_stochastic::{
    OpeningTreeInputs, build_stochastic_context, context::OpeningTree,
    provenance::ComponentProvenance,
};

use crate::error::CliError;

use crate::commands::broadcast::{
    BroadcastConfig, BroadcastOpeningTree, broadcast_value, stopping_rules_from_broadcast,
};

use super::{RunArgs, RunContext};

pub(super) fn resolve_thread_count(cli_threads: Option<u32>) -> usize {
    if let Some(n) = cli_threads {
        return n as usize;
    }
    if let Ok(val) = std::env::var("COBRE_THREADS")
        && let Ok(n) = val.parse::<usize>()
        && n > 0
    {
        return n;
    }
    1
}

/// Values loaded on rank 0 by [`load_case_and_config`]. The trailing
/// [`cobre_io::SetupTimings`] leaves `broadcast_seconds` zero;
/// [`broadcast_and_build_setup`] fills it after the broadcast region runs.
type LoadedCase = (
    PrepareStochasticResult,
    PrepareHydroModelsResult,
    BroadcastConfig,
    cobre_io::Config,
    Vec<cobre_core::ScalarParameter>,
    cobre_io::SetupTimings,
);

/// Load case and config on rank 0, capturing errors for MPI collective participation.
fn load_case_and_config(
    args: &RunArgs,
    quiet: bool,
    stderr: &Term,
) -> Result<LoadedCase, CliError> {
    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "case directory does not exist",
            ),
            context: args.case_dir.display().to_string(),
        });
    }
    if !quiet {
        let _ = stderr.write_line(&format!("Loading case: {}", args.case_dir.display()));
    }
    // Single load: downstream consumers reuse the artifacts returned here instead
    // of re-reading the same files from disk.
    let mut timings = cobre_io::SetupTimings::default();

    let load_start = std::time::Instant::now();
    let cobre_io::LoadedCase { system, artifacts } =
        cobre_io::load_case_with_artifacts(&args.case_dir)?;
    let config_path = args.case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path)?;
    timings.load_seconds = load_start.elapsed().as_secs_f64();

    let bcast = BroadcastConfig::from_config(&config)?;
    let seed = bcast.seed;

    let stochastic_start = std::time::Instant::now();
    let prepared = prepare_stochastic(
        system,
        &args.case_dir,
        &config,
        seed,
        &bcast.training_source,
    )
    .map_err(CliError::from)?;
    timings.stochastic_fit_seconds = stochastic_start.elapsed().as_secs_f64();

    let mut hydro_timings = cobre_sddp::HydroFitTimings::default();
    let hydro_models = cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts(
        &prepared.system,
        &artifacts,
        config.exports.fpha_deviation_points,
        Some(&mut hydro_timings),
    )
    .map_err(CliError::from)?;
    timings.production_fit_seconds = hydro_timings.production_fit_seconds;
    timings.evaporation_fit_seconds = hydro_timings.evaporation_fit_seconds;

    Ok((
        prepared,
        hydro_models,
        bcast,
        config,
        artifacts.scalar_parameters,
        timings,
    ))
}

/// Output of [`broadcast_and_build_setup`]. `root_*` fields are `Some` only on
/// rank 0 (used for output writing); the flag fields are broadcast from rank 0.
pub(super) struct LoadBroadcastResult {
    pub(super) system: System,
    pub(super) setup: StudySetup,
    pub(super) root_config: Option<cobre_io::Config>,
    pub(super) root_estimation_report: Option<EstimationReport>,
    pub(super) root_estimation_path: Option<cobre_sddp::EstimationPath>,
    pub(super) training_enabled: bool,
    pub(super) policy_mode: cobre_io::PolicyMode,
    /// `None` on non-root ranks, which reconstruct setup independently and never
    /// write metadata.
    pub(super) setup_timings: Option<cobre_io::SetupTimings>,
}

/// Set up the communicator, terminal, rayon pool, and resolve the output directory.
pub(super) fn setup_communicator(
    args: &RunArgs,
) -> Result<RunContext<impl Communicator>, CliError> {
    let comm = create_communicator()?;
    let is_root = comm.rank() == 0;
    let quiet = args.quiet || !is_root;

    // mpiexec pipes rank 0's stderr without a PTY, so force colors on; console
    // would otherwise disable them on the non-TTY pipe.
    let mpi_active = comm.size() > 1;
    if mpi_active && is_root && !args.quiet {
        console::set_colors_enabled_stderr(true);
    }

    let stderr = Term::stderr();

    // Gather topology while the concrete backend type is still in scope.
    let topology = comm.topology().clone();

    let configured_threads = resolve_thread_count(args.threads);
    let actual_threads = match rayon::ThreadPoolBuilder::new()
        .num_threads(configured_threads)
        .build_global()
    {
        Ok(()) => configured_threads,
        Err(err) => {
            let actual = rayon::current_num_threads();
            tracing::warn!(
                configured = configured_threads,
                actual,
                %err,
                "rayon global thread pool init failed; using existing pool",
            );
            actual
        }
    };
    if actual_threads == 0 {
        return Err(CliError::Internal {
            message: "rayon reported zero active threads — unexpected state".to_string(),
        });
    }

    let solver_version = cobre_solver::active_solver_version();

    if !quiet {
        crate::banner::print_banner(&stderr);
        crate::summary::print_execution_topology(
            &stderr,
            &topology,
            actual_threads,
            cobre_solver::active_solver_name(),
            Some(&solver_version),
        );
    }

    let output_dir: PathBuf = args
        .output
        .clone()
        .unwrap_or_else(|| args.case_dir.join("output"));
    let term_width = crate::progress::resolve_term_width();
    let render_mode = crate::progress::RenderMode::auto();

    Ok(RunContext {
        comm,
        is_root,
        quiet,
        n_threads: actual_threads,
        output_dir,
        term_width,
        stderr,
        render_mode,
        topology,
        solver_version,
    })
}

/// Load the case on rank 0, broadcast system/config/tree, and build `StudySetup` on all ranks.
// Rationale: one MPI coordination seam — splitting it would scatter the ordered
// broadcast-receive-build sequence across callsites.
#[allow(clippy::too_many_lines)]
pub(super) fn broadcast_and_build_setup(
    ctx: &RunContext<impl Communicator>,
    args: &RunArgs,
) -> Result<LoadBroadcastResult, CliError> {
    // Kept out of the broadcast tuple: timings are never broadcast — only rank 0
    // writes metadata.
    let mut root_setup_timings: Option<cobre_io::SetupTimings> = None;
    let (
        raw_system,
        raw_bcast_config,
        mut root_config,
        root_stochastic,
        root_estimation_report,
        root_estimation_path,
        raw_bcast_tree,
        root_hydro_models,
        raw_scalar_parameters,
        load_err,
    ) = if ctx.is_root {
        match load_case_and_config(args, ctx.quiet, &ctx.stderr) {
            Ok((prepared, hydro_models, bcast, config, scalar_parameters, timings)) => {
                root_setup_timings = Some(timings);
                let bcast_tree = if prepared.stochastic.provenance().opening_tree
                    == ComponentProvenance::UserSupplied
                {
                    let t = prepared.stochastic.opening_tree();
                    Some(BroadcastOpeningTree {
                        data: t.data().to_vec(),
                        openings_per_stage: t.openings_per_stage_slice().to_vec(),
                        dim: t.dim(),
                    })
                } else {
                    None
                };
                let cobre_sddp::PrepareStochasticResult {
                    system,
                    stochastic,
                    estimation_report,
                    estimation_path,
                } = prepared;
                let bcast_params: Vec<cobre_io::BroadcastScalarParameter> = scalar_parameters
                    .iter()
                    .map(cobre_io::BroadcastScalarParameter::from)
                    .collect();
                (
                    Some(system),
                    Some(bcast),
                    Some(config),
                    Some(stochastic),
                    estimation_report,
                    Some(estimation_path),
                    Some(bcast_tree),
                    Some(hydro_models),
                    Some(bcast_params),
                    None,
                )
            }
            Err(e) => (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(e),
            ),
        }
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };
    let broadcast_start = std::time::Instant::now();
    let system_result = broadcast_value(raw_system, &ctx.comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &ctx.comm);
    let scalar_parameters_result = broadcast_value(raw_scalar_parameters, &ctx.comm);

    let tree_result = broadcast_value(raw_bcast_tree, &ctx.comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    let seed = bcast_config.seed;

    // Non-root reconstruction must reproduce rank 0's factor entries (load, NCS)
    // and forward seed exactly, for MPI reproducibility.
    let stochastic = if ctx.is_root {
        drop(tree_result);
        root_stochastic.ok_or_else(|| CliError::Internal {
            message: "stochastic context missing on rank 0 after successful load".to_string(),
        })?
    } else {
        let user_tree: Option<OpeningTree> =
            tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        reconstruct_stochastic_context_non_root(
            &system,
            &bcast_config,
            user_tree,
            seed,
            &args.case_dir,
        )?
    };

    let hydro_models = if ctx.is_root {
        root_hydro_models.ok_or_else(|| CliError::Internal {
            message: "hydro models missing on rank 0 after successful load".to_string(),
        })?
    } else {
        // Deviation-points opt-in is `false` regardless of the run-level export
        // flag: non-root ranks never reach the write site (only rank 0 writes).
        prepare_hydro_models(&system, &args.case_dir, false).map_err(|e| CliError::Internal {
            message: format!("hydro model preprocessing error on non-root rank: {e}"),
        })?
    };

    let training_enabled = bcast_config.training_enabled;
    let policy_mode = bcast_config.policy_mode;
    let scalar_parameters: Vec<cobre_core::ScalarParameter> = scalar_parameters_result?
        .into_iter()
        .map(cobre_core::ScalarParameter::from)
        .collect();
    let setup = build_study_setup(
        &system,
        &mut bcast_config,
        stochastic,
        hydro_models,
        scalar_parameters,
    )?;

    if let Some(timings) = root_setup_timings.as_mut() {
        timings.broadcast_seconds = broadcast_start.elapsed().as_secs_f64();
    }

    Ok(LoadBroadcastResult {
        system,
        setup,
        root_config: root_config.take(),
        root_estimation_report,
        root_estimation_path,
        training_enabled,
        policy_mode,
        setup_timings: root_setup_timings,
    })
}

/// Reconstruct the non-root `StochasticContext` from broadcast parameters.
fn reconstruct_stochastic_context_non_root(
    system: &System,
    bcast_config: &BroadcastConfig,
    user_tree: Option<OpeningTree>,
    seed: u64,
    case_dir: &Path,
) -> Result<cobre_stochastic::StochasticContext, CliError> {
    let training_src = &bcast_config.training_source;
    let forward_seed = training_src.seed.map(i64::unsigned_abs);

    let load_factor_entries =
        load_load_factors_for_stochastic(case_dir).map_err(|e| CliError::Internal {
            message: format!("load factor error on non-root rank: {e}"),
        })?;
    let load_block_pairs: Vec<Vec<cobre_stochastic::normal::precompute::BlockFactorPair>> =
        load_factor_entries
            .iter()
            .map(|e| {
                e.block_factors
                    .iter()
                    .map(|bf| (bf.block_id, bf.factor))
                    .collect()
            })
            .collect();
    let load_entity_factors: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
        load_factor_entries
            .iter()
            .zip(load_block_pairs.iter())
            .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
            .collect();

    let ncs_raw = build_ncs_factor_entries(system);
    let ncs_entity_factors: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
        ncs_raw
            .iter()
            .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
            .collect();

    let opening_tree_library = rebuild_historical_library_non_root(system, training_src)?;

    build_stochastic_context(
        system,
        seed,
        forward_seed,
        &load_entity_factors,
        &ncs_entity_factors,
        OpeningTreeInputs {
            user_tree,
            historical_library: opening_tree_library.as_ref(),
            external_scenario_counts: None,
            // None is safe: the auto-generated tree is broadcast from rank 0, so
            // independent per-stage noise here never reaches the result.
            // TODO(noise-group-non-root-saa-tree): wire noise_group_ids for non-root SAA tree generation
            noise_group_ids: None,
        },
        cobre_stochastic::ClassSchemes {
            inflow: Some(training_src.inflow_scheme),
            load: Some(training_src.load_scheme),
            ncs: Some(training_src.ncs_scheme),
        },
    )
    .map_err(|e| CliError::Internal {
        message: format!("stochastic context error: {e}"),
    })
}

/// Build the non-root `HistoricalScenarioLibrary` from broadcast parameters.
fn rebuild_historical_library_non_root(
    system: &System,
    training_src: &cobre_core::scenario::ScenarioSource,
) -> Result<Option<cobre_stochastic::HistoricalScenarioLibrary>, CliError> {
    use cobre_core::temporal::NoiseMethod;

    // Mirrors `prepare_stochastic` on rank 0: build the library when any stage
    // uses HistoricalResiduals.
    let needs_historical_tree = system
        .stages()
        .iter()
        .any(|s| s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals);

    if needs_historical_tree {
        let study_stages: Vec<_> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let hydro_ids: Vec<cobre_core::EntityId> = system.hydros().iter().map(|h| h.id).collect();
        let cycle_len = system
            .policy_graph()
            .season_map
            .as_ref()
            .map(|sm| sm.seasons.len());
        let par = cobre_stochastic::PrecomputedPar::build(
            system.inflow_models(),
            &study_stages,
            &hydro_ids,
            cycle_len,
        )
        .map_err(|e| CliError::Internal {
            message: format!("PAR build error on non-root rank: {e}"),
        })?;
        let max_order = par.max_order();
        let user_pool = training_src.historical_years.as_ref();
        let window_years = cobre_stochastic::discover_historical_windows(
            system.inflow_history(),
            &hydro_ids,
            &study_stages,
            max_order,
            user_pool,
            system.policy_graph().season_map.as_ref(),
            1,
        )
        .map_err(|e| CliError::Internal {
            message: format!("historical window discovery error on non-root rank: {e}"),
        })?;
        let mut lib = cobre_stochastic::HistoricalScenarioLibrary::new(
            window_years.len(),
            study_stages.len(),
            hydro_ids.len(),
            max_order,
            window_years.clone(),
        );
        // past_inflows seeds the η-inversion chain from the same x₀ as the
        // forward pass. Compute stage_lag_transitions via the production helper,
        // not the in-function uniform-monthly fallback, which silently misroutes
        // non-monthly study grids.
        let noop_season_map;
        let season_map_for_transitions: &cobre_core::temporal::SeasonMap =
            if let Some(sm) = system.policy_graph().season_map.as_ref() {
                sm
            } else {
                noop_season_map = cobre_core::temporal::SeasonMap {
                    cycle_type: cobre_core::temporal::SeasonCycleType::Monthly,
                    seasons: Vec::new(),
                };
                &noop_season_map
            };
        let stage_lag_transitions = cobre_sddp::lag_transition::precompute_stage_lag_transitions(
            &study_stages,
            season_map_for_transitions,
            max_order,
        );
        cobre_stochastic::standardize_historical_windows(
            &mut lib,
            system.inflow_history(),
            &hydro_ids,
            &study_stages,
            &par,
            &window_years,
            system.policy_graph().season_map.as_ref(),
            &system.initial_conditions().past_inflows,
            &stage_lag_transitions,
        );
        Ok(Some(lib))
    } else {
        Ok(None)
    }
}

/// Construct `StudySetup` on all ranks from broadcast parameters.
// Rationale: by-value because `StudySetup` construction moves these in; a
// reference would force an internal clone.
#[allow(clippy::needless_pass_by_value)]
fn build_study_setup(
    system: &System,
    bcast_config: &mut BroadcastConfig,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    scalar_parameters: Vec<cobre_core::ScalarParameter>,
) -> Result<StudySetup, CliError> {
    let stopping_rule_set = stopping_rules_from_broadcast(bcast_config);
    let cut_selection = bcast_config.cut_selection.take();
    let config = ConstructionConfig {
        seed: bcast_config.seed,
        forward_passes: bcast_config.forward_passes,
        stopping_rule_set,
        n_scenarios: bcast_config.n_scenarios,
        io_channel_capacity: usize::try_from(bcast_config.io_channel_capacity).unwrap_or(64),
        policy_path: bcast_config.policy_path.clone(),
        inflow_method: bcast_config.inflow_method,
        cut_selection,
        cut_activity_tolerance: bcast_config.cut_activity_tolerance,
        budget: bcast_config.budget,
        export_states: bcast_config.export_states,
        scalar_parameters,
    };
    StudySetup::from_broadcast_params(
        system,
        stochastic,
        config,
        hydro_models,
        &bcast_config.training_source,
        &bcast_config.simulation_source,
    )
    .map_err(CliError::from)
}

pub(super) fn run_pre_training(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&cobre_io::Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<cobre_sddp::EstimationPath>,
    setup_timings: Option<&cobre_io::SetupTimings>,
) -> Result<(), CliError> {
    // Renders before the Hydro models block so the setup timings sit with the
    // other setup summaries.
    if ctx.is_root
        && !ctx.quiet
        && let Some(timings) = setup_timings
    {
        crate::summary::print_setup_summary(&ctx.stderr, timings);
    }

    // Built regardless of `quiet`: it also feeds the persisted sidecar consumed
    // by `cobre summary`, not just the optional print.
    if ctx.is_root {
        let hydro_summary = build_hydro_model_summary(&setup.hydro_models, system);
        if !ctx.quiet {
            crate::summary::print_hydro_model_summary(&ctx.stderr, &hydro_summary);
        }
        let hydro_models_path = ctx.output_dir.join("training/hydro_models.json");
        cobre_io::write_hydro_model_summary(&hydro_models_path, &hydro_summary).map_err(|e| {
            CliError::Internal {
                message: format!("failed to write hydro model summary: {e}"),
            }
        })?;
    }

    if ctx.is_root
        && let Some(path) = root_estimation_path
    {
        let provenance = cobre_sddp::build_provenance_report(
            path,
            root_estimation_report,
            setup.stochastic.provenance(),
            system.hydros().len(),
            &setup.hydro_models.provenance,
        );
        if !ctx.quiet {
            crate::summary::print_provenance_summary(&ctx.stderr, &provenance);
        }
        let provenance_path = ctx.output_dir.join("training/model_provenance.json");
        cobre_io::write_provenance_report(&provenance_path, &provenance).map_err(|e| {
            CliError::Internal {
                message: format!("failed to write provenance report: {e}"),
            }
        })?;
    }

    if ctx.is_root && root_config.is_some_and(|c| c.exports.stochastic) {
        if !ctx.quiet {
            let _ = ctx.stderr.write_line("Exporting stochastic artifacts...");
        }
        let stderr = &ctx.stderr;
        let quiet = ctx.quiet;
        let mut on_warning = |msg: &str| {
            if !quiet {
                let _ = stderr.write_line(&format!("warning: stochastic export failed ({msg})"));
            }
        };
        cobre_sddp::orchestration::export_stochastic_artifacts(
            &ctx.output_dir,
            &setup.stochastic,
            system,
            root_estimation_report,
            &mut on_warning,
        );
    }

    if ctx.is_root {
        let scaling_path = ctx.output_dir.join("training/scaling_report.json");
        cobre_io::write_scaling_report(&scaling_path, &setup.stage_data.scaling_report).map_err(
            |e| CliError::Internal {
                message: format!("failed to write scaling report: {e}"),
            },
        )?;
    }

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

    Ok(())
}
