//! Shared LP-solve seam used by the training passes (`forward`, `backward`) and
//! by `simulation`.
//!
//! This directory module owns exactly the solve-side code that is shared across
//! all three hot-path drivers, kept OUTSIDE any single pass module so neither a
//! training pass nor simulation owns code the others depend on:
//!
//! - [`solver_phase`] — the SDDP phase enum, the backend-agnostic profile trait,
//!   and the per-phase solver-profile constants.
//! - [`stage_solve`] — the single unified per-stage LP-solve entry point
//!   ([`stage_solve::run_stage_solve`]) wrapping the 3-driver
//!   `enforce_basic_count_invariant` seam. Kept as ONE file: fragmenting the
//!   invariant-enforcement seam would let a driver bypass it.
//! - [`partition`] — the deterministic static work partition every driver uses
//!   to split scenarios across workers.

pub(crate) mod partition;
pub mod solver_phase;
pub(crate) mod stage_solve;

// Re-exported at this module so internal `crate::solve::solver_phase::Phase`
// resolves and so the crate-root re-export in `lib.rs` can re-point here.
// `partition` stays crate-private — the re-export carries its `pub(crate)`
// visibility so `crate::solve::partition` resolves at every internal call site.
pub(crate) use partition::partition;
pub use solver_phase::Phase;
#[cfg(feature = "highs")]
pub use solver_phase::{BACKWARD_PROFILE, FORWARD_PROFILE, SIMULATION_PROFILE};
