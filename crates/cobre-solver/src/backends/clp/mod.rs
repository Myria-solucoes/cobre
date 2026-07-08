//! CLP LP solver backend: [`ClpSolver`] wrapping the CLP C API through the FFI
//! layer in `crate::ffi::clp`.
//!
//! # Thread Safety
//!
//! [`ClpSolver`] is `Send` but not `Sync`: the CLP handle is exclusively owned,
//! so transferring ownership to a worker thread is safe; concurrent access is
//! not permitted.

mod config;
mod interface;
mod retry;
mod solver;
#[cfg(test)]
mod tests;

pub use config::{ClpAlgorithm, ClpProfile};
pub use solver::{ClpSolver, clp_version};

// Rationale: the sole consumer of this re-export is the `#[cfg(test)]` `tests`
// module (`interface` reads the const directly via `super::retry`), so a
// non-test build sees no consumer; the allow suppresses the spurious
// `unused_imports` while keeping the lint live under `--cfg test`.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use retry::LADDER_RUNGS;
