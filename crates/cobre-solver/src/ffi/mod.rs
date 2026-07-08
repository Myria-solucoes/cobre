//! Raw FFI boundary to the LP solver C wrapper layers.
//!
//! - [`highs`] — `cobre_highs_*` bindings to `csrc/highs_wrapper.h`.
//! - `clp` — `cobre_clp_*` bindings to `csrc/clp_wrapper.h` (`clp` feature).
//!
//! `HiGHS` symbols are re-exported flat so `crate::ffi::cobre_highs_*` resolves
//! without qualifying through [`highs`]. Use the safe wrappers in the backend
//! modules rather than calling these bindings directly.
//!
//! `pub(crate)` throughout: not nameable from outside `cobre-solver`. The
//! `test_support` module (`lib.rs`, `test-support` feature) is the sole
//! sanctioned external escape hatch, bridging a named subset as thin
//! pass-throughs for integration tests.

pub(crate) mod highs;

#[cfg(feature = "clp")]
pub(crate) mod clp;

#[cfg(feature = "highs")]
pub(crate) use highs::*;
