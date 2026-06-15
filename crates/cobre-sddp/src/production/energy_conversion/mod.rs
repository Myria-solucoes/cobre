//! Energy-conversion preprocessing: derives the per-`(hydro, stage)` scalars
//! `ρ_eq` (equivalent productivity, MW per m³/s), `V_ref` (reference reservoir
//! volume, hm³), `Q_ref` (reference turbined flow, m³/s), and `ρ_acum`
//! (accumulated cascade productivity, MW per m³/s).
//!
//! The output [`EnergyConversionSet`] is consumed by simulation extraction
//! and by the energy-balance constraints that compute natural inflow energy and stored energy.
//!
//! ## Derivation logic
//!
//! This module owns both the data layout and the builder that derives every
//! scalar:
//!
//! - non-FPHA `ρ_eq` derivation populates `equivalent_productivity_mw_per_m3s`
//!   from the model's stored scalar.
//! - FPHA `ρ_eq` derivation evaluates `ρ_esp · h_eq(V_ref, Q_ref)` against the
//!   reservoir VHA curve.
//! - `ρ_acum` derivation walks the cascade in topological order summing
//!   downstream equivalent productivities.
//!
//! ## Submodule layout
//!
//! - `types` — the data layout: the per-`(hydro, stage)` cell `EnergyConversion`,
//!   the indexed grid `EnergyConversionSet`, and the `EnergyConversionError` enum.
//! - `productivity_override` — the user-supplied `HydroEnergyProductivityOverride`
//!   lookup table and its `build_hydro_energy_productivity_override` constructor
//!   (kept together because the constructor populates the struct's private fields).
//! - `builder` — `build_energy_conversion_set` and the private FPHA/cascade helpers.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_sddp::energy_conversion::Symbol` module path resolve to
//! the same item regardless of which submodule owns it.

mod builder;
mod productivity_override;
mod types;

pub use builder::build_energy_conversion_set;
pub use productivity_override::{
    HydroEnergyProductivityOverride, build_hydro_energy_productivity_override,
};
pub use types::{EnergyConversion, EnergyConversionError, EnergyConversionSet};
