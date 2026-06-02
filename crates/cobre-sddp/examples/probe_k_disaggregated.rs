//! Probe: measure populated cut counts per stage after a single SDDP
//! iteration with today's cut-selection kernel.
//!
//! The probe is a one-off diagnostic that runs **one** training iteration
//! against a user-supplied study config and prints, for every stage,
//! the number of populated cuts (`K`) and the number of active cuts (`A`)
//! held in the future cost function pool. The output drives sizing decisions
//! for downstream verification work (disaggregated determinism fixtures
//! and production-scale benchmarks).
//!
//! ## Why one iteration
//!
//! Today's kernel makes a multi-iteration disaggregated run prohibitively
//! expensive — a single iteration's selection cost is bearable (~5–10 s on
//! a 64-stage disaggregated case) and is enough to populate the pool to its
//! iter-1 value, which is what the design's `K` estimate is derived from.
//!
//! ## Usage
//!
//! ```text
//! cargo build --release -p cobre-sddp --example probe_k_disaggregated
//! ./target/release/examples/probe_k_disaggregated <study-config-path>
//! ```
//!
//! Where `<study-config-path>` is the path to a `config.json` belonging to
//! a disaggregated study case directory (the probe derives the case
//! directory by taking the parent of the config path, mirroring the CLI's
//! `cobre run <CASE_DIR>` convention).
//!
//! ## Output
//!
//! One line per stage on stdout:
//!
//! ```text
//! stage=<t> populated_count=<K> active_count=<A>
//! ```
//!
//! Followed by a trailing summary line:
//!
//! ```text
//! summary D=<D> M=<M> max_K=<max> mean_K=<mean> min_K=<min>
//! ```
//!
//! Where `D` is the state dimension, `M` is the configured trial-point
//! count (forward passes per iteration), and `max/mean/min_K` are the
//! summary statistics across stages.
//!
//! ## Exit codes
//!
//! - `0`: success.
//! - `2`: bad arguments (missing positional or `--help`).
//! - `3`: training completed but no pools were populated (defensive safety net).
//! - other: non-zero on any error during loading or training.
//!
//! ## Scope
//!
//! Not registered as a workspace test or CI target — this is a one-shot
//! diagnostic, not a regression check.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stdout
)]

use std::process::ExitCode;

#[cfg(feature = "highs")]
use std::path::{Path, PathBuf};

#[cfg(feature = "highs")]
use cobre_comm::LocalBackend;
#[cfg(feature = "highs")]
use cobre_core::scenario::ScenarioSource;
#[cfg(feature = "highs")]
use cobre_io::{config::StoppingRuleConfig, parse_config};
#[cfg(feature = "highs")]
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
#[cfg(feature = "highs")]
use cobre_solver::highs::HighsSolver;

/// Force the probe to run exactly one SDDP iteration.
#[cfg(feature = "highs")]
const PROBE_MAX_ITERATIONS: u32 = 1;

#[cfg(not(feature = "highs"))]
fn main() -> ExitCode {
    eprintln!(
        "probe_k_disaggregated example requires the `highs` feature; rebuild with --features highs"
    );
    ExitCode::from(2)
}

#[cfg(feature = "highs")]
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("usage: probe_k_disaggregated <study-config-path>");
        return ExitCode::from(2);
    }
    let config_path = PathBuf::from(&args[1]);

    if let Err(code) = run_probe(&config_path) {
        return code;
    }
    ExitCode::SUCCESS
}

/// Drive the single-iteration probe. Returns an [`ExitCode`] on error so the
/// caller can propagate it; returns `Ok(())` on success.
#[cfg(feature = "highs")]
fn run_probe(config_path: &Path) -> Result<(), ExitCode> {
    // The CLI accepts a case directory and looks for `config.json` inside it;
    // we accept the config path directly and derive the case directory as
    // the parent of the config file (matching the CLI's `<case_dir>/config.json`
    // convention).
    let case_dir = match config_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => {
            eprintln!(
                "error: cannot derive case directory from config path '{}'",
                config_path.display()
            );
            return Err(ExitCode::from(2));
        }
    };

    if !config_path.exists() {
        eprintln!(
            "error: config file does not exist: {}",
            config_path.display()
        );
        return Err(ExitCode::from(2));
    }

    let mut config = parse_config(config_path).map_err(|e| {
        eprintln!("error: failed to parse config: {e}");
        ExitCode::from(2)
    })?;

    // Force a single SDDP iteration regardless of what the config file
    // specifies. Overriding the stopping rule (rather than mutating
    // `setup.loop_params.max_iterations` after construction) ensures the
    // FCF cut pool is sized for one iteration's worth of cuts instead of
    // the full configured budget — significantly less memory pressure on
    // the host running the probe.
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: PROBE_MAX_ITERATIONS,
    }]);

    let system = cobre_io::load_case(&case_dir).map_err(|e| {
        eprintln!("error: load_case failed: {e}");
        ExitCode::from(1)
    })?;

    let prepared = prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
        .map_err(|e| {
        eprintln!("error: prepare_stochastic failed: {e}");
        ExitCode::from(1)
    })?;
    let system = prepared.system;
    let stochastic = prepared.stochastic;

    let hydro_models = prepare_hydro_models(&system, &case_dir).map_err(|e| {
        eprintln!("error: prepare_hydro_models failed: {e}");
        ExitCode::from(1)
    })?;

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models).map_err(|e| {
        eprintln!("error: StudySetup::new failed: {e}");
        ExitCode::from(1)
    })?;

    // Belt-and-suspenders: even though we already capped the stopping rule,
    // explicitly force `loop_params.max_iterations` to the probe's value so
    // any future code path that reads it directly cannot accidentally run
    // a multi-iteration probe.
    setup.loop_params.max_iterations = u64::from(PROBE_MAX_ITERATIONS);

    let comm = LocalBackend;
    let mut solver = HighsSolver::new().map_err(|e| {
        eprintln!("error: HiGHS init failed: {e}");
        ExitCode::from(1)
    })?;

    eprintln!(
        "probe_k_disaggregated: starting 1-iteration run (case_dir={}, D={}, M={})",
        case_dir.display(),
        setup.fcf.state_dimension,
        setup.loop_params.forward_passes,
    );

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .map_err(|e| {
            eprintln!("error: training failed: {e}");
            ExitCode::from(1)
        })?;
    if let Some(err) = outcome.error {
        eprintln!("error: training reported mid-iteration failure: {err}");
        return Err(ExitCode::from(1));
    }

    match print_pool_report(&setup) {
        PoolReport::Ok => Ok(()),
        PoolReport::EmptyPools => Err(ExitCode::from(3)),
    }
}

/// Outcome of [`print_pool_report`]: either the report printed normally, or
/// every stage pool was empty (caller should exit with code 3).
#[cfg(feature = "highs")]
enum PoolReport {
    Ok,
    EmptyPools,
}

/// Format per-stage `populated_count` / `active_count` lines and the trailing
/// summary line. Returns [`PoolReport::EmptyPools`] if every pool is empty
/// (which should not happen at disaggregated scale).
#[cfg(feature = "highs")]
fn print_pool_report(setup: &StudySetup) -> PoolReport {
    let pools = &setup.fcf.pools;
    let d = setup.fcf.state_dimension;
    let m = setup.loop_params.forward_passes;

    let mut max_k: usize = 0;
    let mut min_k: usize = usize::MAX;
    let mut total_k: u64 = 0;

    for (t, pool) in pools.iter().enumerate() {
        let k = pool.populated_count;
        let a = pool.active_count();
        println!("stage={t} populated_count={k} active_count={a}");
        if k > max_k {
            max_k = k;
        }
        if k < min_k {
            min_k = k;
        }
        total_k = total_k.saturating_add(k as u64);
    }

    if max_k == 0 {
        eprintln!(
            "error: every stage pool is empty after iter 1 — expected disaggregated scale to populate at least one cut per stage"
        );
        return PoolReport::EmptyPools;
    }

    // `pools.is_empty()` cannot hold (max_K == 0 path above would have caught it),
    // so the divisor is non-zero.
    let n_stages = pools.len() as u64;
    #[allow(clippy::cast_precision_loss)]
    let mean_k = total_k as f64 / n_stages as f64;
    // If pools is empty min_K would still be usize::MAX — guard against that
    // for cleaner output, although the max_K == 0 branch above already exits.
    let min_k_out = if min_k == usize::MAX { 0 } else { min_k };

    println!("summary D={d} M={m} max_K={max_k} mean_K={mean_k:.2} min_K={min_k_out}");

    PoolReport::Ok
}
