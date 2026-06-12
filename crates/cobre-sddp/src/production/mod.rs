//! Production-modeling cluster for the SDDP solver.
//!
//! This directory groups the preprocessing that turns reservoir geometry and
//! plant data into the per-`(hydro, stage)` production representations the LP
//! consumes — FPHA hyperplane fitting, resolved production/evaporation models,
//! energy-conversion scalars, and the simulation write-payload conversions.
//!
//! ## Contents
//!
//! - [`hydro_models`] — resolved production functions
//!   ([`ProductionModelSet`](hydro_models::ProductionModelSet),
//!   [`ResolvedProductionModel`](hydro_models::ResolvedProductionModel)),
//!   evaporation models, and the [`prepare_hydro_models`](hydro_models::prepare_hydro_models)
//!   preprocessing entry point.
//! - [`fpha_fitting`] — FPHA hyperplane fitting from the Volume-Height-Area curve.
//!   Owns the static FPHA-geometry validation
//!   ([`FphaFittingError`](fpha_fitting::FphaFittingError)); the runtime FPHA
//!   average-storage coefficient lives in `lp_builder` (see `.claude/rules/sddp.md`).
//! - [`energy_conversion`] — derives the `ρ_eq`, `V_ref`, `Q_ref`, and `ρ_acum`
//!   scalars ([`EnergyConversionSet`](energy_conversion::EnergyConversionSet))
//!   consumed by simulation extraction and the energy-balance constraints.
//! - [`conversion`] — field-for-field conversions from `Simulation*Result` to the
//!   `cobre-io` write-payload record types.

pub mod energy_conversion;
pub(crate) mod fpha_fitting;
pub mod hydro_models;

// Carries `pub(crate)` from the pre-cluster declaration: this conversion layer
// is a pure intra-crate internal, never named from outside the crate, so the
// move into `production/` must not widen its visibility.
pub(crate) mod conversion;
