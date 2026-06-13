//! Resolved and scenario data model.
//!
//! Groups the temporal structure ([`temporal`]), per-stage resolved bounds and
//! penalties ([`resolved`]), scenario sources ([`scenario`]), and the supporting
//! parameter ([`parameters`]) and penalty-resolution ([`penalty`]) types.

pub mod parameters;
pub mod penalty;
pub mod resolved;
pub mod scenario;
pub mod temporal;
