//! Concrete LP/MIP solver backends behind the crate's public vocabulary.
//!
//! Each backend implements the same [`crate::SolverInterface`] contract:
//!
//! - `highs` — the default `HiGHS` backend (`highs` feature).
//! - `clp` — the optional CLP/`CoinUtils` backend (`clp` feature).
//! - [`profiled`] — a backend-agnostic wrapper that records per-solve
//!   statistics around any inner backend.
//!
//! Exactly one of `highs`/`clp` is selected at compile time (enabling both is a
//! compile error in the crate root); `profiled` is always present because it is
//! generic over the active backend.

#[cfg(feature = "highs")]
pub mod highs;

#[cfg(feature = "clp")]
pub mod clp;

pub mod profiled;
