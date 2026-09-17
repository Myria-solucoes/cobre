//! # cobre
//!
//! Command-line interface for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! Provides commands for running optimization studies, validating input data,
//! and inspecting results from the terminal.

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
    run::{self, RunArgs},
    schema::{self, SchemaArgs},
    validate::{self, ValidateArgs},
    version,
};

/// Controls when ANSI color/style escapes are emitted on stderr (no environment override).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ColorWhen {
    /// Enable color when stderr is connected to a TTY (default).
    Auto,
    /// Always emit ANSI color escapes, even when stderr is not a TTY.
    Always,
    /// Never emit ANSI color escapes.
    Never,
}

/// Must be called before stderr output; `Auto` leaves TTY auto-detection in place.
pub(crate) fn resolve_color(cli_color: ColorWhen) {
    match cli_color {
        ColorWhen::Always => console::set_colors_enabled_stderr(true),
        ColorWhen::Never => console::set_colors_enabled_stderr(false),
        ColorWhen::Auto => {}
    }
}

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
    /// Manage JSON Schema files for case directory input types.
    Schema(SchemaArgs),
    /// Print version, solver backend, and build information.
    Version,
}

fn main() {
    let cli = Cli::parse();

    resolve_color(cli.color);

    // A subscriber can only install once, so a failed init must not stop the command.
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
