//! # cobre
//!
//! Command-line interface for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! Provides commands for running optimization studies, validating input data,
//! and inspecting results from the terminal.
//!
//! ## Subcommands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `cobre run <CASE_DIR>` | Load, train, simulate, and write results |
//! | `cobre validate <CASE_DIR>` | Validate a case directory |
//! | `cobre report <RESULTS_DIR>` | Query results and print to stdout |
//! | `cobre summary <OUTPUT_DIR>` | Display the post-run summary from a completed output directory |
//! | `cobre schema export` | Export JSON Schema files for all input types |
//! | `cobre version` | Print version and build information |

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod banner;
mod commands;
mod error;
mod progress;
mod summary;
mod templates;

use clap::{Parser, Subcommand, ValueEnum};

use commands::{
    init::{self, InitArgs},
    report::{self, ReportArgs},
    run::{self, RunArgs},
    schema::{self, SchemaArgs},
    summary::{self as summary_cmd, SummaryArgs},
    validate::{self, ValidateArgs},
    version,
};

/// Controls when ANSI color/style escapes are emitted on stderr.
///
/// Selected solely by the `--color <WHEN>` CLI flag (no environment override).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ColorWhen {
    /// Enable color when stderr is connected to a TTY (default).
    Auto,
    /// Always emit ANSI color escapes, even when stderr is not a TTY.
    Always,
    /// Never emit ANSI color escapes.
    Never,
}

/// Apply the resolved color setting to the `console` crate's global stderr flag.
///
/// Must be called before any output is written to stderr so that the banner,
/// progress bars, and error messages all honour the chosen setting. `Auto` leaves
/// the `console` crate's TTY auto-detection in place.
pub(crate) fn resolve_color(cli_color: ColorWhen) {
    match cli_color {
        ColorWhen::Always => console::set_colors_enabled_stderr(true),
        ColorWhen::Never => console::set_colors_enabled_stderr(false),
        ColorWhen::Auto => {}
    }
}

/// Open infrastructure for power system computation.
#[derive(Debug, Parser)]
#[command(
    name = "cobre",
    about = "Open infrastructure for power system computation"
)]
struct Cli {
    /// Control ANSI color output on stderr.
    ///
    /// `always` forces color on (useful under `mpiexec` which pipes stderr through
    /// a non-TTY). `never` disables all color. `auto` lets the terminal detection
    /// decide.
    #[arg(long, global = true, default_value = "auto")]
    color: ColorWhen,

    #[command(subcommand)]
    command: Command,
}

/// Top-level subcommands for the `cobre` binary.
#[derive(Debug, Subcommand)]
enum Command {
    /// Scaffold a new case directory from an embedded template.
    Init(InitArgs),
    /// Load a case directory, train a policy, and run simulation.
    Run(RunArgs),
    /// Validate a case directory and print a structured diagnostic report.
    Validate(ValidateArgs),
    /// Query results from a completed run and print them to stdout.
    Report(ReportArgs),
    /// Display the post-run summary from a completed output directory.
    Summary(SummaryArgs),
    /// Manage JSON Schema files for case directory input types.
    Schema(SchemaArgs),
    /// Print version, solver backend, and build information.
    Version,
}

fn main() {
    let cli = Cli::parse();

    resolve_color(cli.color);

    // The default `warn` level surfaces library deprecation notices on stderr
    // while staying quiet for ordinary runs; `RUST_LOG` overrides it. Init
    // errors are ignored — the subscriber can only install once, and a CLI that
    // cannot subscribe must still run its command.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let result = match cli.command {
        Command::Init(args) => init::execute(args),
        Command::Run(ref args) => run::execute(args),
        Command::Validate(args) => validate::execute(args),
        Command::Report(args) => report::execute(args),
        Command::Summary(args) => summary_cmd::execute(args),
        Command::Schema(args) => schema::execute(args),
        Command::Version => version::execute(),
    };

    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            e.format_error(&console::Term::stderr());
            std::process::exit(e.exit_code());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ColorWhen, resolve_color};

    #[test]
    fn test_resolve_color_always_enables_color() {
        resolve_color(ColorWhen::Always);
        assert!(
            console::colors_enabled_stderr(),
            "Always must set colors_enabled_stderr to true"
        );
    }

    #[test]
    fn test_resolve_color_never_disables_color() {
        resolve_color(ColorWhen::Never);
        assert!(
            !console::colors_enabled_stderr(),
            "Never must set colors_enabled_stderr to false"
        );
    }
}
