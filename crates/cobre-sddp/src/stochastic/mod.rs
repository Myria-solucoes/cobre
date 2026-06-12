//! Inflow/stochastic-estimation cluster for the SDDP solver.
//!
//! This directory groups the preprocessing and hot-path machinery that turns
//! raw historical inflow data and per-stage noise samples into the PAR(p)
//! parameters, lag-accumulation state, and patched LP right-hand sides the
//! solver consumes.
//!
//! ## Contents
//!
//! - [`estimation`] — automatic PAR(p) parameter estimation from historical
//!   inflow observations ([`EstimationPath`](estimation::EstimationPath),
//!   [`estimate_from_history`](estimation::estimate_from_history)); bridges
//!   `cobre-io` case loading and `cobre-stochastic` fitting.
//! - [`noise`] — the noise-transformation hot path
//!   ([`transform_inflow_noise`](noise::transform_inflow_noise),
//!   [`transform_load_noise`](noise::transform_load_noise)) that writes patched
//!   RHS values into the stage LP before each solve. Owns the NCS
//!   availability-factor contract `A_r = max_gen · clamp(mean + std·η, 0, 1)`
//!   ([`transform_ncs_noise`](noise::transform_ncs_noise),
//!   [`compute_effective_eta`](noise::compute_effective_eta); see
//!   `.claude/rules/sddp.md`).
//! - [`noise_key_diag`] — env-gated backward-pass diagnostic pairing a
//!   σ-weighted aggregate noise key with per-opening warm-resolve pivot counts.
//! - [`lag_transition`] — precomputation of per-stage lag accumulation weights
//!   and period-finalization flags
//!   ([`precompute_stage_lag_transitions`](lag_transition::precompute_stage_lag_transitions)),
//!   keeping calendar arithmetic out of inner solver loops.
//! - [`inflow_method`] — the closed
//!   [`InflowNonNegativityMethod`](inflow_method::InflowNonNegativityMethod)
//!   enum dispatched when handling negative PAR(p) realisations in the LP.
//! - [`stochastic_summary`] — the
//!   [`StochasticSummary`](stochastic_summary::StochasticSummary) reporting
//!   types and builder for the stochastic-preprocessing outcome.

pub mod estimation;
pub mod inflow_method;
pub mod lag_transition;
pub mod noise_key_diag;

// Carries `pub(crate)` from the pre-cluster declarations: the noise hot path and
// the summary builder are pure intra-crate internals, never named from outside
// the crate, so the move into `stochastic/` must not widen their visibility.
pub(crate) mod noise;
pub(crate) mod stochastic_summary;
