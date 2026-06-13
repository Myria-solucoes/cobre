//! CLP LP solver backend.
//!
//! This module provides [`ClpSolver`], which wraps the CLP C API through the
//! FFI layer in [`crate::ffi::clp`]. It defines the solver type, its associated
//! [`ClpProfile`] value type, the handle lifecycle, and the reusable solution
//! buffers, plus the complete [`crate::SolverInterface`] implementation.
//!
//! # Thread Safety
//!
//! [`ClpSolver`] is `Send` but not `Sync`. The underlying CLP handle is
//! exclusively owned; transferring ownership to a worker thread is safe.
//! Concurrent access from multiple threads is not permitted.
//!
//! # Configuration
//!
//! Configuration is wired through [`ClpProfile`] and the profile-setter FFI
//! calls. The profile defaults mirror `HighsProfile` where the options
//! overlap (tight feasibility tolerances, scaling off because the cobre
//! prescaler already conditions the matrix), and use CLP-native values
//! otherwise (perturbation off, iteration limit deferred to the per-call
//! heuristic).
//!
//! # Submodule layout
//!
//! - `config` — the [`ClpAlgorithm`] selector and the [`ClpProfile`] tuning
//!   value type plus its [`Default`] (leaf; no dependency on the other
//!   submodules).
//! - `solver` — the [`ClpSolver`] struct, the `unsafe impl Send`, the
//!   `CLP_BASIS_*` cold-basis status codes, the lifecycle/hot-start primitives,
//!   the `normalize_row_dual` / `i32_from_usize` free helpers, `Drop`, and the
//!   [`clp_version`] free function.
//! - `retry` — the `LADDER_RUNGS` ladder-size constant, the `EscalationOutcome`,
//!   and the cold-solve escalation ladder (a further `impl ClpSolver` block).
//! - `interface` — `impl SolverInterface for ClpSolver`.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_solver::clp::Symbol` module path resolve to the same
//! item regardless of which submodule owns it.

mod config;
mod interface;
mod retry;
mod solver;
#[cfg(test)]
mod tests;

pub use config::{ClpAlgorithm, ClpProfile};
pub use solver::{ClpSolver, clp_version};

// `pub(crate)` because the escalation-count contract is shared by `interface`'s
// `solve` (charges this many attempts to `retry_count` on ladder exhaustion) and
// the regression test asserting the exhausted-ladder `retry_count`; this
// re-export lets the in-file `tests` import resolve `LADDER_RUNGS` through the
// directory module path. The sole consumer of *this re-export* is the
// `#[cfg(test)]` `tests` module (`interface` reads the const directly via
// `super::retry`), so a non-test build sees no consumer; the `cfg(not(test))`
// allow suppresses the spurious `unused_imports` while keeping the lint live
// under `--cfg test`.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use retry::LADDER_RUNGS;
