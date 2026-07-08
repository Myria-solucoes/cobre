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
//! - `fpha_fitting` — FPHA hyperplane fitting from the Volume-Height-Area curve.
//!   Owns the static FPHA-geometry validation
//!   (`FphaFittingError`); the runtime FPHA
//!   average-storage coefficient lives in [`crate::lp::builder`].
//! - [`energy_conversion`] — derives the `ρ_eq`, `V_ref`, `Q_ref`, and `ρ_acum`
//!   scalars ([`EnergyConversionSet`](energy_conversion::EnergyConversionSet))
//!   consumed by simulation extraction and the energy-balance constraints.
//! - [`stage_key`] — the [`StageId`](stage_key::StageId)/[`StudyPos`](stage_key::StudyPos)
//!   newtypes distinguishing a domain stage identifier from a study-horizon
//!   position, and [`CalendarMonth`](stage_key::CalendarMonth) distinguishing
//!   the calendar month from a season-cycle index.
//! - `conversion` — field-for-field conversions from `Simulation*Result` to the
//!   `cobre-io` write-payload record types.

pub(crate) mod conversion;
pub mod energy_conversion;
pub(crate) mod fpha_fitting;
pub mod hydro_models;
pub mod stage_key;
