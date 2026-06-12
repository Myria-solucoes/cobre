//! Policy load/warm-start/resume phase for `cobre run`.
//!
//! Applies warm-start, resume, or boundary cuts before training, and loads a
//! trained policy from disk for simulation-only mode.

use std::path::Path;

use cobre_comm::Communicator;
use cobre_core::System;
use cobre_sddp::StudySetup;

use crate::error::CliError;

use super::RunContext;

/// Load a policy checkpoint from disk and optionally validate its compatibility.
///
/// The `policy_dir` must already exist.
fn load_and_validate_checkpoint(
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&cobre_io::Config>,
) -> Result<cobre_io::PolicyCheckpoint, CliError> {
    let checkpoint = cobre_io::output::policy::read_policy_checkpoint(policy_dir).map_err(|e| {
        CliError::Internal {
            message: format!("failed to read policy checkpoint: {e}"),
        }
    })?;

    if let Some(config) = root_config
        && config.policy.validate_compatibility
    {
        // Rationale: the cast cannot truncate — `n_stages` is the validated study
        // horizon (a `u16`-scale stage count), far below `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
        let state_dim =
            u32::try_from(setup.fcf.state_dimension).map_err(|e| CliError::Internal {
                message: format!("state_dimension overflows u32: {e}"),
            })?;
        cobre_sddp::validate_policy_compatibility(&checkpoint.metadata, state_dim, n_stages)
            .map_err(CliError::from)?;
    }

    Ok(checkpoint)
}

/// Build the warm-start FCF from a loaded checkpoint and seed the basis cache.
///
/// Shared by the warm-start and resume paths: replaces the setup's FCF with one
/// reserving an extra final-iteration slot, then seeds the warm-start basis store.
fn load_checkpoint_into_setup(
    checkpoint: &cobre_io::PolicyCheckpoint,
    setup: &mut StudySetup,
) -> Result<(), CliError> {
    // Reserve one extra slot for cuts added in the final iteration.
    let warm_fcf = cobre_sddp::FutureCostFunction::new_with_warm_start(
        &checkpoint.stage_cuts,
        setup.loop_params.forward_passes,
        setup.loop_params.max_iterations.saturating_add(1),
    )
    .map_err(CliError::from)?;
    setup.replace_fcf(warm_fcf);
    // Seed the warm-start basis store from the checkpoint's stored
    // bases so iteration 1's cut-loaded LPs warm-start. Skip when the
    // checkpoint carries no bases (written without `store_basis`):
    // the store stays empty and iteration 1 cold-starts.
    if !checkpoint.stage_bases.is_empty() {
        let basis_cache = cobre_sddp::build_basis_cache_from_checkpoint(
            setup.stage_data.stage_templates.templates.len(),
            &checkpoint.stage_bases,
            &checkpoint.stage_cuts,
        );
        setup.set_warm_start_basis_cache(basis_cache);
    }
    Ok(())
}

/// Apply warm-start or resume policy before training, if requested.
pub(super) fn apply_training_policy(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    root_config: Option<&cobre_io::Config>,
    policy_mode: cobre_io::PolicyMode,
) -> Result<(), CliError> {
    match policy_mode {
        cobre_io::PolicyMode::WarmStart => {
            let policy_dir = ctx.output_dir.join(&setup.policy_path);
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot warm-start \
                         without a prior policy.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior policy for warm-start training...");
            }
            let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;
            load_checkpoint_into_setup(&checkpoint, setup)?;
            if ctx.is_root && !ctx.quiet {
                let warm_count = setup.fcf.pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Warm-start: loaded {warm_count} cuts per stage from prior policy."
                ));
            }
        }
        cobre_io::PolicyMode::Resume => {
            let policy_dir = ctx.output_dir.join(&setup.policy_path);
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot resume \
                         without a prior checkpoint.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior checkpoint for resume training...");
            }
            let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;
            let completed = u64::from(checkpoint.metadata.completed_iterations);
            if completed >= setup.loop_params.max_iterations && ctx.is_root && !ctx.quiet {
                let _ = ctx.stderr.write_line(&format!(
                    "WARNING: Checkpoint already completed {completed} iterations \
                     (max_iterations = {}). No additional training will occur.",
                    setup.loop_params.max_iterations
                ));
            }
            load_checkpoint_into_setup(&checkpoint, setup)?;
            setup.set_start_iteration(completed);
            if ctx.is_root && !ctx.quiet {
                let warm_count = setup.fcf.pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Resume: loaded {warm_count} cuts per stage, \
                     resuming from iteration {completed}."
                ));
            }
        }
        cobre_io::PolicyMode::Fresh => {}
    }

    // Boundary cuts — orthogonal to policy mode. Runs after the match block so
    // that warm-start and boundary cuts compose correctly: warm-start replaces the
    // entire FCF first, then boundary cuts overwrite only the terminal pool.
    if let Some(bp) = root_config.and_then(|c| c.policy.boundary.as_ref()) {
        let boundary_path = ctx.output_dir.join(&bp.path);
        // Rationale: the cast cannot truncate — `state_dimension` counts FCF
        // state variables (one per reservoir/lag), bounded by the validated study
        // dimensions and far below `u32::MAX`.
        #[allow(clippy::cast_possible_truncation)]
        let state_dim = setup.fcf.state_dimension as u32;
        let boundary_records =
            cobre_sddp::load_boundary_cuts(&boundary_path, bp.source_stage, state_dim)
                .map_err(CliError::from)?;
        cobre_sddp::inject_boundary_cuts(setup, &boundary_records);
        if ctx.is_root && !ctx.quiet {
            let _ = ctx.stderr.write_line(&format!(
                "Boundary cuts: loaded {} cuts from stage {} of {}",
                boundary_records.len(),
                bp.source_stage,
                boundary_path.display()
            ));
        }
    }

    Ok(())
}

/// Load a policy checkpoint and build a synthetic `TrainingResult` for simulation-only mode.
pub(super) fn load_policy_for_simulation(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    root_config: Option<&cobre_io::Config>,
) -> Result<cobre_sddp::TrainingResult, CliError> {
    if ctx.is_root && !ctx.quiet {
        let _ = ctx
            .stderr
            .write_line("Training disabled. Loading policy for simulation-only mode...");
    }

    let policy_dir = ctx.output_dir.join(&setup.policy_path);
    if !policy_dir.exists() {
        return Err(CliError::Internal {
            message: format!(
                "Policy directory not found: {}. Cannot run simulation-only \
                 mode without a trained policy.",
                policy_dir.display()
            ),
        });
    }

    let checkpoint = load_and_validate_checkpoint(&policy_dir, system, setup, root_config)?;

    let loaded_fcf = cobre_sddp::FutureCostFunction::from_deserialized(&checkpoint.stage_cuts)
        .map_err(CliError::from)?;
    setup.replace_fcf(loaded_fcf);

    let basis_cache = cobre_sddp::build_basis_cache_from_checkpoint(
        setup.stage_data.stage_templates.templates.len(),
        &checkpoint.stage_bases,
        &checkpoint.stage_cuts,
    );

    Ok(cobre_sddp::TrainingResult::new(
        checkpoint.metadata.final_lower_bound,
        checkpoint
            .metadata
            .best_upper_bound
            .unwrap_or(f64::INFINITY),
        0.0,
        0.0,
        checkpoint.metadata.completed_iterations.into(),
        "loaded from checkpoint".to_string(),
        0,
        basis_cache,
        Vec::new(),
        None,
        // Baked templates are not stored in policy checkpoints. simulate() re-bakes all stage
        // templates at startup from the FCF row pool when baked_templates is None.
        None,
    ))
}
