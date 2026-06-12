//! Raw FFI boundary to the LP solver C wrapper layers.
//!
//! This directory isolates the crate's entire `unsafe` C-binding surface. Each
//! backend's bindings live in its own module:
//!
//! - [`highs`] — `cobre_highs_*` bindings to `csrc/highs_wrapper.h`.
//! - [`clp`] — `cobre_clp_*` bindings to `csrc/clp_wrapper.h` (`clp` feature).
//!
//! The `HiGHS` symbols are re-exported flat from this module so that
//! `crate::ffi::cobre_highs_*` continues to resolve at every call site without
//! qualifying through [`highs`]. Use the safe wrappers in the backend modules
//! rather than calling these bindings directly.

pub mod highs;

#[cfg(feature = "clp")]
pub mod clp;

pub use highs::*;
