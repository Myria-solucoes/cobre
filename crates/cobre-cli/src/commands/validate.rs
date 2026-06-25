//! `cobre validate <CASE_DIR>` subcommand.
//!
//! Runs the six-layer validation pipeline followed by the three pre-solver
//! preparation phases and prints a structured diagnostic report to stdout.
//! No banner or progress bar — the output is the deliverable.
//!
//! ## Validation contract
//!
//! If `cobre validate <CASE_DIR>` exits 0, then `cobre run <CASE_DIR>` will not
//! fail in any phase before the solver begins iterating. The three pre-solver
//! phases exercised here are:
//!
//! 1. [`cobre_sddp::StudyParams::from_config`] — validates `config.json` fields
//!    that are only checked at algorithm startup and surfaces deprecation
//!    warnings for fields that are scheduled for removal.
//! 2. [`cobre_sddp::prepare_stochastic`] — runs PAR estimation from inflow
//!    history, loads user opening trees, and builds the stochastic context.
//! 3. [`cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts`] — resolves
//!    production and evaporation models from the pre-parsed artifact bundle.

use std::path::{Path, PathBuf};

use clap::Args;
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::validate_phases::{PrepPhase, prep_phase_metadata};
use cobre_sddp::{StudyParams, prepare_stochastic};
use console::{Term, style};

use crate::error::CliError;

/// Arguments for the `cobre validate` subcommand.
#[derive(Debug, Args)]
#[command(about = "Validate a case directory and print a structured diagnostic report")]
pub struct ValidateArgs {
    /// Path to the case directory to validate.
    pub case_dir: PathBuf,
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

/// Print a pre-solver preparation error to `term` and return the
/// `"file_label: message"` string for the caller to embed in a [`CliError`].
fn format_prep_error(
    term: &Term,
    phase: PrepPhase,
    err: &cobre_sddp::SddpError,
    case_dir: &Path,
) -> String {
    let (_kind, file_label) = prep_phase_metadata(phase, err);
    let message = err.to_string();
    let _ = term.write_line(&format!(
        "Validation: 1 errors, 0 warnings in {}",
        case_dir.display()
    ));
    let _ = term.write_line(&format!(
        "{} {file_label}: {message}",
        style("error:").red().bold()
    ));
    format!("{file_label}: {message}")
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
    let (loaded, report) = match cobre_io::validate_case_with_artifacts(&args.case_dir) {
        Ok(result) => result,
        Err(cobre_io::LoadError::IoError { path, source }) => {
            return Err(CliError::Io {
                source,
                context: path.display().to_string(),
            });
        }
        Err(cobre_io::LoadError::ConstraintError { description }) => {
            // Warnings are not available when errors abort the pipeline, so report 0.
            format_constraint_description(&stdout, &description, 0, &args.case_dir);
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
            let report_msg = format_prep_error(&stdout, PrepPhase::Config, err, &args.case_dir);
            return Err(CliError::Validation {
                report: report_msg,
                already_rendered: true,
            });
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
            let report_msg = format_prep_error(&stdout, PrepPhase::Stochastic, err, &args.case_dir);
            return Err(CliError::Validation {
                report: report_msg,
                already_rendered: true,
            });
        }
    };

    // Reuses the already-parsed artifacts bundle to avoid re-reading disk.
    if let Err(ref err) =
        prepare_hydro_models_from_artifacts(&prepared.system, &artifacts, false, None)
    {
        let report_msg = format_prep_error(&stdout, PrepPhase::HydroModels, err, &args.case_dir);
        return Err(CliError::Validation {
            report: report_msg,
            already_rendered: true,
        });
    }

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
