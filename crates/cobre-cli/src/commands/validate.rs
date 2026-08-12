//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the six-layer validation pipeline followed by the pre-solver
//! preparation phases and prints a structured diagnostic report to stdout —
//! or, with `--json`, a single machine-readable JSON object: the boundary
//! reconciliation outcome on success, or an `error` object naming the first
//! failing phase. Stdout under `--json` is always one such object or empty,
//! never human report text. No banner or progress bar — the output is the
//! deliverable.
//!
//! ## Validation contract
//!
//! If `cobre validate <CASE_DIR>` exits 0, then `cobre run <CASE_DIR>` will not
//! fail in any phase before the solver begins iterating. The pre-solver
//! phases exercised here are:
//!
//! 1. [`cobre_sddp::StudyParams::from_config`] — validates `config.json` fields
//!    that are only checked at algorithm startup and surfaces deprecation
//!    warnings for fields that are scheduled for removal.
//! 2. [`cobre_sddp::prepare_stochastic`] — runs PAR estimation from inflow
//!    history, loads user opening trees, and builds the stochastic context.
//! 3. [`cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts`] — resolves
//!    production and evaporation models from the pre-parsed artifact bundle.
//! 4. When `config.policy.boundary` is configured, [`cobre_sddp::StudySetup::new`]
//!    plus [`cobre_sddp::load_boundary_cuts`] — builds the study and reconciles
//!    the boundary policy against its terminal manifest, without solving.

use std::path::{Path, PathBuf};

use clap::Args;
use cobre_core::System;
use cobre_io::{BoundaryPolicy, Config, LoadError, validate_case_with_artifacts};
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::validate_phases::{PrepPhase, prep_phase_metadata};
use cobre_sddp::{
    BoundaryReconciliationReport, PrepareHydroModelsResult, SddpError, StudyParams, StudySetup,
    load_boundary_cuts, prepare_stochastic, resolve_boundary_source_stage,
};
use cobre_stochastic::StochasticContext;
use console::{Term, style};
use serde::Serialize;

use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,

    /// Emit the boundary reconciliation outcome as a single JSON object to
    /// stdout instead of the human-readable report.
    #[arg(long)]
    pub json: bool,
}

/// `cobre validate --json`'s stdout payload. On success, populates
/// `configured` (`report` stays `None` — an explicit absent marker, not a
/// crash — when `configured` is `Some(false)`). On a failure that aborts
/// before boundary status is ever resolved, `configured`/`report` stay
/// `None` and `error` is populated instead — the two outcomes never overlap.
#[derive(Debug, Serialize)]
struct ValidateBoundaryOutput {
    /// Whether `policy.boundary` is configured in this case's `config.json`.
    configured: Option<bool>,
    /// The reconciliation report when `configured` is `Some(true)`.
    report: Option<BoundaryReconciliationReport>,
    /// The failing phase and message, populated only on an early abort.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ValidateErrorOutput>,
}

/// One `cobre validate --json` early-abort failure: `phase` is
/// [`prep_phase_metadata`]'s stable kind string (or `CaseValidationError` for
/// the six-layer IO pipeline, which precedes any [`PrepPhase`]) — the same
/// string programmatic callers already filter on; `message` is the
/// human-readable detail.
#[derive(Debug, Serialize)]
struct ValidateErrorOutput {
    phase: String,
    message: String,
}

impl ValidateBoundaryOutput {
    fn success(report: Option<BoundaryReconciliationReport>) -> Self {
        Self {
            configured: Some(report.is_some()),
            report,
            error: None,
        }
    }

    fn error(phase: &str, message: &str) -> Self {
        Self {
            configured: None,
            report: None,
            error: Some(ValidateErrorOutput {
                phase: phase.to_string(),
                message: message.to_string(),
            }),
        }
    }
}

fn format_constraint_description(
    term: &Term,
    description: &str,
    warning_count: usize,
    path: &Path,
) {
    let error_lines: Vec<&str> = description.lines().collect();
    let _ = term.write_line(&format!(
        "Validation: {} errors, {} warnings in {}",
        error_lines.len(),
        warning_count,
        path.display()
    ));
    for line in error_lines {
        let _ = term.write_line(&format!("{} {line}", style("error:").red().bold()));
    }
}

/// Compute a pre-solver preparation error's stable phase kind (the same
/// string [`prep_phase_metadata`] exposes for programmatic filtering) and its
/// `"file_label: message"` report string — shared by the human stdout render
/// and the `--json` error object, so the two never drift apart.
fn describe_prep_error(phase: PrepPhase, err: &SddpError) -> (&'static str, String) {
    let (kind, file_label) = prep_phase_metadata(phase, err);
    (kind, format!("{file_label}: {err}"))
}

/// Print a pre-solver preparation error's `report` line to `term`.
fn print_prep_error(term: &Term, report: &str, case_dir: &Path) {
    let _ = term.write_line(&format!(
        "Validation: 1 errors, 0 warnings in {}",
        case_dir.display()
    ));
    let _ = term.write_line(&format!("{} {report}", style("error:").red().bold()));
}

/// Handle a pre-solver preparation-phase failure: print the human report to
/// `stdout_sink` (`None` under `--json`), emit the `--json` error object when
/// `json`, and return the [`CliError`] for the caller to propagate.
fn prep_error_to_cli_error(
    stdout_sink: Option<&Term>,
    json: bool,
    phase: PrepPhase,
    err: &SddpError,
    case_dir: &Path,
) -> Result<CliError, CliError> {
    let (kind, report) = describe_prep_error(phase, err);
    if let Some(term) = stdout_sink {
        print_prep_error(term, &report, case_dir);
    }
    if json {
        emit_validate_json(&ValidateBoundaryOutput::error(kind, &report))?;
    }
    Ok(CliError::Validation {
        report,
        already_rendered: true,
    })
}

/// Print a boundary-reconciliation error to `term` and return the
/// `"policy.boundary: message"` string for the caller to embed in a
/// [`CliError`].
fn format_boundary_error(term: &Term, err: &SddpError, case_dir: &Path) -> String {
    let message = err.to_string();
    let _ = term.write_line(&format!(
        "Validation: 1 errors, 0 warnings in {}",
        case_dir.display()
    ));
    let _ = term.write_line(&format!(
        "{} policy.boundary: {message}",
        style("error:").red().bold()
    ));
    format!("policy.boundary: {message}")
}

/// Build a `StudySetup` from the parsed config and reconcile
/// `config.policy.boundary` against its terminal manifest, without solving.
/// `stdout` is `None` under `--json`, suppressing every advisory/warning line
/// this prints in human mode.
fn reconcile_boundary(
    case_dir: &Path,
    config: &Config,
    bp: &BoundaryPolicy,
    system: &System,
    stochastic: StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    stdout: Option<&Term>,
) -> Result<BoundaryReconciliationReport, SddpError> {
    let setup = StudySetup::new(system, config, stochastic, hydro_models)?;
    let boundary_path = case_dir.join("output").join(&bp.path);

    // Rationale: the cast cannot truncate — `state_dimension` counts FCF
    // state variables (one per reservoir/lag), bounded by the validated study
    // dimensions and far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;
    let current_manifest = setup.build_terminal_entity_manifest(system);
    let target_delivery_intervals = setup.build_terminal_anticipated_delivery_intervals(system);

    let source_stage = if let Some(idx) = bp.source_stage {
        idx
    } else {
        let resolved = resolve_boundary_source_stage(&boundary_path, &target_delivery_intervals)?;
        if let Some(term) = stdout {
            let _ = term.write_line(&format!(
                "Boundary source_stage resolved to {resolved} (no explicit \
                 policy.boundary.source_stage configured)."
            ));
        }
        resolved
    };

    let mut on_warning = |msg: &str| {
        if let Some(term) = stdout {
            let _ = term.write_line(&format!("{} {msg}", style("warning:").yellow().bold()));
        }
    };
    let boundary_cuts = load_boundary_cuts(
        &boundary_path,
        source_stage,
        state_dim,
        &current_manifest,
        &target_delivery_intervals,
        config.state_space.inflow_lag_depth,
        setup.stage_data.stage_templates.cost_scale_factor,
        &mut on_warning,
    )?;

    Ok(boundary_cuts.report().clone())
}

/// Reconcile `config.policy.boundary` when configured, mapping a reject into
/// a [`CliError::Validation`] (pre-rendered to `stdout` in human mode, per the
/// module's exit-0 contract). Returns `Ok(None)` when no boundary is
/// configured — no `StudySetup` work runs.
fn run_boundary_check(
    case_dir: &Path,
    config: &Config,
    system: &System,
    stochastic: StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    stdout: Option<&Term>,
) -> Result<Option<BoundaryReconciliationReport>, CliError> {
    let Some(bp) = config.policy.boundary.as_ref() else {
        return Ok(None);
    };

    match reconcile_boundary(
        case_dir,
        config,
        bp,
        system,
        stochastic,
        hydro_models,
        stdout,
    ) {
        Ok(report) => Ok(Some(report)),
        Err(err) => {
            let Some(term) = stdout else {
                return Err(CliError::from(err));
            };
            let report_msg = format_boundary_error(term, &err, case_dir);
            Err(CliError::Validation {
                report: report_msg,
                already_rendered: true,
            })
        }
    }
}

/// Serialize `output` as `cobre validate --json`'s single stdout JSON object
/// (the `cobre report` convention).
fn emit_validate_json(output: &ValidateBoundaryOutput) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(output).map_err(|e| CliError::Internal {
        message: format!("failed to serialize validate output: {e}"),
    })?;
    println!("{json}");
    Ok(())
}

/// Execute the `validate` subcommand, printing a structured diagnostic report
/// (with any pipeline warnings) to stdout. Honors the module's validation contract:
/// exit 0 implies `cobre run` will not fail before the solver begins iterating.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the case directory fails validation,
/// [`CliError::Io`] on filesystem errors, or [`CliError::Internal`] for
/// unexpected parse or schema failures.
#[allow(clippy::needless_pass_by_value)]
pub fn execute(args: ValidateArgs) -> Result<(), CliError> {
    let stdout = Term::stdout();
    let stdout_sink = (!args.json).then_some(&stdout);

    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("case directory not found: {}", args.case_dir.display()),
            ),
            context: args.case_dir.display().to_string(),
        });
    }

    // _with_artifacts returns the pre-parsed CaseArtifacts the hydro-models phase
    // needs, avoiding a second disk read.
    let (loaded, report) = match validate_case_with_artifacts(&args.case_dir) {
        Ok(result) => result,
        Err(LoadError::IoError { path, source }) => {
            return Err(CliError::Io {
                source,
                context: path.display().to_string(),
            });
        }
        Err(LoadError::ConstraintError { description }) => {
            // Warnings are not available when errors abort the pipeline, so report 0.
            if let Some(term) = stdout_sink {
                format_constraint_description(term, &description, 0, &args.case_dir);
            }
            if args.json {
                emit_validate_json(&ValidateBoundaryOutput::error(
                    "CaseValidationError",
                    &description,
                ))?;
            }
            return Err(CliError::Validation {
                report: description,
                already_rendered: true,
            });
        }
        Err(other) => {
            return Err(CliError::Internal {
                message: other.to_string(),
            });
        }
    };

    let system = loaded.system;
    let artifacts = loaded.artifacts;

    let config_path = args.case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).map_err(CliError::from)?;

    let study_params = match StudyParams::from_config(&config) {
        Ok(p) => p,
        Err(ref err) => {
            return Err(prep_error_to_cli_error(
                stdout_sink,
                args.json,
                PrepPhase::Config,
                err,
                &args.case_dir,
            )?);
        }
    };

    let seed = study_params.seed;

    // config_path is a sentinel here: training_scenario_source uses it only for
    // historical-years look-up and error messages.
    let training_source = config
        .training_scenario_source(&config_path)
        .map_err(CliError::from)?;

    // The most expensive step (PAR estimation, opening trees); validate runs it
    // anyway so an exit-0 guarantees full parity with `run`.
    let prepared = match prepare_stochastic(system, &args.case_dir, &config, seed, &training_source)
    {
        Ok(p) => p,
        Err(ref err) => {
            return Err(prep_error_to_cli_error(
                stdout_sink,
                args.json,
                PrepPhase::Stochastic,
                err,
                &args.case_dir,
            )?);
        }
    };

    // Reuses the already-parsed artifacts bundle to avoid re-reading disk.
    let hydro_models =
        match prepare_hydro_models_from_artifacts(&prepared.system, &artifacts, false, None) {
            Ok(hm) => hm,
            Err(ref err) => {
                return Err(prep_error_to_cli_error(
                    stdout_sink,
                    args.json,
                    PrepPhase::HydroModels,
                    err,
                    &args.case_dir,
                )?);
            }
        };

    if !args.json {
        let _ = stdout.write_line(&format!(
            "Valid case: {} buses, {} hydros, {} thermals, {} lines",
            prepared.system.n_buses(),
            prepared.system.n_hydros(),
            prepared.system.n_thermals(),
            prepared.system.n_lines(),
        ));
        if report.warning_count > 0 {
            let _ = stdout.write_line(&format!(
                "Validation: 0 errors, {} warnings in {}",
                report.warning_count,
                args.case_dir.display()
            ));
            for entry in &report.warnings {
                let location = if let Some(entity) = &entry.entity {
                    format!("{} ({})", entry.file, entity)
                } else {
                    entry.file.clone()
                };
                let _ = stdout.write_line(&format!(
                    "{} {location}: {}",
                    style("warning:").yellow().bold(),
                    entry.message
                ));
            }
        }
    }

    let boundary_report = run_boundary_check(
        &args.case_dir,
        &config,
        &prepared.system,
        prepared.stochastic,
        hydro_models,
        stdout_sink,
    )?;

    if args.json {
        emit_validate_json(&ValidateBoundaryOutput::success(boundary_report))?;
    } else if let Some(report) = &boundary_report {
        for line in report.diagnostic_lines() {
            let _ = stdout.write_line(&line);
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::fmt::Write as _;

    use cobre_io::{ReportEntry, ValidationReport};

    fn format_report_to_string(report: &ValidationReport, path: &Path) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Validation: {} errors, {} warnings in {}",
            report.error_count,
            report.warning_count,
            path.display()
        );
        for entry in &report.errors {
            let _ = writeln!(out, "error: {}", format_entry(entry));
        }
        for entry in &report.warnings {
            let _ = writeln!(out, "warning: {}", format_entry(entry));
        }
        out
    }

    fn format_entry(entry: &ReportEntry) -> String {
        if let Some(entity) = &entry.entity {
            format!("{}: {} ({})", entry.file, entry.message, entity)
        } else {
            format!("{}: {}", entry.file, entry.message)
        }
    }

    fn make_report() -> ValidationReport {
        ValidationReport {
            error_count: 1,
            warning_count: 1,
            errors: vec![ReportEntry {
                kind: "FileNotFound".to_string(),
                file: "system/hydros.json".to_string(),
                entity: Some("hydro_42".to_string()),
                message: "required file is missing".to_string(),
            }],
            warnings: vec![ReportEntry {
                kind: "UnusedEntity".to_string(),
                file: "system/thermals.json".to_string(),
                entity: None,
                message: "thermal has zero capacity".to_string(),
            }],
        }
    }

    use super::*;

    #[test]
    fn format_report_contains_error_label() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("error:"),
            "expected 'error:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_warning_label() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("warning:"),
            "expected 'warning:' in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_file_path() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("system/hydros.json"),
            "expected file path in output, got: {output}"
        );
    }

    #[test]
    fn format_report_contains_error_message() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("required file is missing"),
            "expected error message in output, got: {output}"
        );
    }

    #[test]
    fn format_report_summary_header_present() {
        let path = PathBuf::from("/case/dir");
        let output = format_report_to_string(&make_report(), &path);
        assert!(
            output.contains("1 errors") && output.contains("1 warnings"),
            "expected summary header with counts, got: {output}"
        );
    }

    #[test]
    fn format_entry_with_entity() {
        let entry = ReportEntry {
            kind: "FileNotFound".to_string(),
            file: "system/buses.json".to_string(),
            entity: Some("bus_01".to_string()),
            message: "missing required field".to_string(),
        };
        let result = format_entry(&entry);
        assert!(result.contains("system/buses.json"), "{result}");
        assert!(result.contains("missing required field"), "{result}");
        assert!(result.contains("bus_01"), "{result}");
    }

    #[test]
    fn format_entry_without_entity() {
        let entry = ReportEntry {
            kind: "FileNotFound".to_string(),
            file: "system/buses.json".to_string(),
            entity: None,
            message: "missing required field".to_string(),
        };
        let result = format_entry(&entry);
        assert!(result.contains("system/buses.json"), "{result}");
        assert!(result.contains("missing required field"), "{result}");
        assert!(!result.contains("(None)"), "{result}");
    }
}
