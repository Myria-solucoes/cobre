//! Inflow/stochastic-estimation cluster for the SDDP solver.
//!
//! Turns raw historical inflow data and per-stage noise samples into the PAR(p)
//! parameters, lag-accumulation state, and patched LP right-hand sides the
//! solver consumes.
//!
//! ## Contents
//!
//! - [`estimation`] — automatic PAR(p) parameter estimation from historical
//!   inflow observations; bridges `cobre-io` case loading and `cobre-stochastic`
//!   fitting.
//! - `noise` — the noise-transformation hot path that writes patched RHS values
//!   into the stage LP before each solve. Owns the NCS availability-factor
//!   contract (`transform_ncs_noise`, `compute_effective_eta`).
//! - [`noise_key`] — σ-weighted noise keys for the backward solve order.
//! - [`lag_transition`] — precomputation of per-stage lag accumulation weights
//!   and period-finalization flags.
//! - [`inflow_method`] — the closed
//!   [`InflowNonNegativityMethod`](inflow_method::InflowNonNegativityMethod)
//!   enum for negative PAR(p) realisations.
//! - `stochastic_summary` — reporting types and builder for the
//!   stochastic-preprocessing outcome.

pub mod estimation;
pub mod inflow_method;
pub mod lag_transition;
pub mod noise_key;

// `pub(crate)`: intra-crate internals, never named from outside the crate.
pub(crate) mod noise;
pub(crate) mod stochastic_summary;
