//! Raw FFI boundary to the LP solver C wrapper layers.
//!
//! - [`highs`] — `cobre_highs_*` bindings to `csrc/highs_wrapper.h`.
//! - `clp` — `cobre_clp_*` bindings to `csrc/clp_wrapper.h` (`clp` feature).
//!
//! `HiGHS` symbols are re-exported flat so `crate::ffi::cobre_highs_*` resolves
//! without qualifying through [`highs`]. Use the safe wrappers in the backend
//! modules rather than calling these bindings directly.

pub mod highs;

#[cfg(feature = "clp")]
pub mod clp;

pub use highs::*;
