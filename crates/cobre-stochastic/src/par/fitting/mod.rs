//! Periodic Yule-Walker estimation and seasonal statistics for PAR model fitting.
//!
//! This module provides the core primitives for fitting Periodic Autoregressive
//! models:
//!
//! 1. [`periodic_autocorrelation`] — computes the periodic normalised
//!    autocorrelation `rho(p, k)` with population divisor and cross-year
//!    lag adjustment.
//! 2. [`build_periodic_yw_matrix`] — constructs the non-Toeplitz periodic
//!    Yule-Walker matrix for a given season and AR order.
//! 3. [`periodic_pacf`] — computes the periodic PACF via progressive matrix
//!    solves for order selection.
//! 4. [`estimate_periodic_ar_coefficients`] — solves the periodic YW system
//!    at the selected order to produce AR coefficients and residual std ratio.
//! 5. [`estimate_seasonal_stats`] — computes seasonal means and
//!    population-divisor (1/N) standard deviations from historical
//!    observations, grouped by `(entity, season)` pair.
//! 6. [`estimate_ar_coefficients`] — produces white-noise (order-0) estimates
//!    for all `(entity, season)` pairs; used by the PACF path when
//!    `max_order == 0`.
//! 7. [`estimate_correlation`] — computes the Pearson correlation matrix of
//!    PAR model residuals across entities, returning a [`CorrelationModel`]
//!    suitable for downstream spectral decomposition.
//!
//! ## Periodic Yule-Walker equations
//!
//! For a periodic AR(p) process the Yule-Walker system is non-Toeplitz because
//! lags cross season boundaries. [`build_periodic_yw_matrix`] assembles the
//! correct per-season matrix and [`estimate_periodic_ar_coefficients`] solves it
//! via LU factorisation with partial pivoting.
//!
//! ## Submodules
//!
//! The fitter is split into cohesive function families:
//!
//! - `seasonal_stats` — seasonal mean/std estimation, history classification,
//!   and the date-to-season lookup.
//! - `ar_coefficients` — white-noise AR estimates and the shared season-lookup
//!   data preparation.
//! - `correlation` — cross-entity residual correlation matrices.
//! - `order_selection` — AIC and PACF AR-order selection.
//! - `yw_matrices` — periodic autocorrelation, the Yule-Walker matrix builders,
//!   the cross-correlation primitives, and the dense linear solver.
//! - `partitioned_covariance` — the conditional FACP via partitioned covariance.
//! - `periodic_ar` — periodic Yule-Walker coefficient estimators.
//! - `annual` — annual-component seasonal statistics.
//! - `estimation` — top-level AR coefficient estimation with order selection,
//!   contribution validation, and the estimation report.

mod annual;
mod ar_coefficients;
mod correlation;
mod estimation;
mod order_selection;
mod partitioned_covariance;
mod periodic_ar;
mod seasonal_stats;
mod yw_matrices;

pub use annual::{AnnualSeasonalStats, estimate_annual_seasonal_stats};
pub use ar_coefficients::{
    ArCoefficientEstimate, estimate_ar_coefficients, estimate_ar_coefficients_with_season_map,
};
pub use correlation::{estimate_correlation, estimate_correlation_with_season_map};
pub use estimation::{
    ArEstimationConfig, ContributionReduction, ContributionValidationResult, EstimationReport,
    HydroEstimationEntry, ReductionReason, StdRatioDivergence, build_estimation_report,
    estimate_ar_coefficients_with_selection,
};
pub use order_selection::{
    AicSelectionResult, PacfSelectionResult, periodic_pacf, select_order_aic, select_order_pacf,
    select_order_pacf_annual,
};
pub use partitioned_covariance::conditional_facp_partitioned;
pub use periodic_ar::{
    PeriodicYwAnnualResult, PeriodicYwResult, estimate_periodic_ar_annual_coefficients,
    estimate_periodic_ar_coefficients,
};
pub use seasonal_stats::{
    HistoryClass, SeasonalStats, classify_history, estimate_seasonal_stats,
    estimate_seasonal_stats_with_season_map, find_season_for_date,
};
pub use yw_matrices::{
    build_extended_periodic_yw_matrix, build_periodic_yw_matrix, build_periodic_yw_matrix_into,
    cross_correlation_a_z_neg1, cross_correlation_z_a, periodic_autocorrelation,
    solve_linear_system,
};

// Private re-imports so the in-module `mod tests` can reach the items it
// references through `super::` without rewriting any test import. These names
// are accessible to the child test module (which can see private items of its
// ancestor) but introduce no new public path.
#[cfg(test)]
use cobre_core::scenario::CorrelationGroup;
#[cfg(test)]
use correlation::compute_pearson_correlation_matrix;
#[cfg(test)]
use partitioned_covariance::assemble_partitioned_covariance;
#[cfg(test)]
use yw_matrices::BUILD_PERIODIC_YW_MATRIX_CALL_COUNT;

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::doc_markdown
)]
mod tests;
