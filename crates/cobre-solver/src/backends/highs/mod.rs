//! `HiGHS` LP solver backend implementing [`SolverInterface`](crate::SolverInterface).
//!
//! This module provides [`HighsSolver`], which wraps the `HiGHS` C API through
//! the FFI layer in `ffi` and implements the full
//! [`SolverInterface`](crate::SolverInterface) contract for iterative LP solving
//! in power system optimization.
//!
//! # Thread Safety
//!
//! [`HighsSolver`] is `Send` but not `Sync`. The underlying `HiGHS` handle is
//! exclusively owned; transferring ownership to a worker thread is safe.
//! Concurrent access from multiple threads is not permitted (`HiGHS`
//! Implementation SS6.3).
//!
//! # Configuration
//!
//! The constructor applies performance-tuned defaults (`HiGHS` Implementation
//! SS4.1): dual simplex, presolve enabled, no parallelism, suppressed output, and
//! tight feasibility tolerances. These defaults are optimised for repeated
//! solves of small-to-medium LPs. Per-run parameters (time limit, iteration
//! limit) are not set here -- those are applied by the caller before each solve.
//!
//! # Submodule layout
//!
//! - `config` — the [`HighsProfile`] tuning value type plus the typed
//!   `OptionValue` / `DefaultOption` table and `default_options()` (leaf;
//!   no dependency on the other submodules).
//! - `solver` — the [`HighsSolver`] struct, the `unsafe impl Send`, the
//!   lifecycle/solve primitives, the warm-start `solve_inner` orchestration,
//!   `Drop`, the [`highs_version`] free function, and the `test-support`
//!   accessors.
//! - `retry` — the `RetryOutcome` and the 12-level warm-start retry-escalation
//!   ladder (a further `impl HighsSolver` block).
//! - `interface` — `impl SolverInterface for HighsSolver`.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_solver::highs::Symbol` module path resolve to the
//! same item regardless of which submodule owns it.

mod config;
mod interface;
mod retry;
mod solver;
#[cfg(test)]
mod tests;

pub use config::HighsProfile;
pub use solver::{HighsSolver, highs_version};
