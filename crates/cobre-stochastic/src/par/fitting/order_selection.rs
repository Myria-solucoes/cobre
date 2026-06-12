//! AR-order selection via the Akaike Information Criterion and the periodic
//! partial autocorrelation function.

use super::yw_matrices::{build_periodic_yw_matrix_into, solve_linear_system};

// ---------------------------------------------------------------------------
// AIC-based AR order selection
// ---------------------------------------------------------------------------

/// Result of AIC-based AR order selection.
///
/// Produced by [`select_order_aic`]. Contains the selected AR order and the
/// AIC value for each candidate order from 0 (white noise) through `p_max`.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AicSelectionResult {
    /// Selected AR order (1-based index into `sigma2_per_order`).
    ///
    /// `0` when no AR order improves over white noise (white noise is optimal
    /// or `sigma2_per_order` is empty).
    pub selected_order: usize,
    /// AIC value for each candidate order `0..=p_max`.
    ///
    /// `aic_values[0]` is the AIC for order 0 (white noise baseline = `0.0`).
    /// `aic_values[k]` is the AIC for AR order `k`, for `k >= 1`.
    pub aic_values: Vec<f64>,
}

/// Select the AR order that minimises the Akaike Information Criterion (AIC).
///
/// For each candidate order `p` in `1..=p_max`, the AIC is:
///
/// ```text
/// AIC(p) = N * ln(σ²_p) + 2p
/// ```
///
/// where `N = n_observations` and `σ²_p = sigma2_per_order[p-1]`.
///
/// The white-noise baseline (order 0) has `AIC(0) = 0.0` by convention
/// (`σ²_0 = 1.0` in the normalised Yule-Walker formulation, so
/// `N * ln(1) + 0 = 0`).
///
/// On ties the lower order wins (parsimony). Non-positive `sigma2` values
/// (which can arise from near-singular Levinson-Durbin truncation) are
/// excluded by assigning `AIC = f64::INFINITY`.
///
/// # Parameters
///
/// - `sigma2_per_order` — prediction error variances from
///   `LevinsonDurbinResult::sigma2_per_order`. `sigma2_per_order[k]`
///   corresponds to AR order `k+1`. Length = `p_max`.
/// - `n_observations` — number of historical observations for this season (`N_m`).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_aic;
///
/// // A variance drop at order 1 that outweighs the penalty selects order 1.
/// let result = select_order_aic(&[0.3], 100);
/// assert_eq!(result.selected_order, 1);
/// assert_eq!(result.aic_values.len(), 2);
///
/// // Empty sigma2 always selects white noise.
/// let result = select_order_aic(&[], 50);
/// assert_eq!(result.selected_order, 0);
/// assert_eq!(result.aic_values, vec![0.0]);
/// ```
pub fn select_order_aic(sigma2_per_order: &[f64], n_observations: usize) -> AicSelectionResult {
    let p_max = sigma2_per_order.len();

    let mut aic_values = Vec::with_capacity(p_max + 1);
    aic_values.push(0.0_f64); // order 0: N*ln(1) + 0 = 0

    #[allow(clippy::cast_precision_loss)]
    let n = n_observations as f64;

    for (k, &sigma2) in sigma2_per_order.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let order = (k + 1) as f64;
        let aic = if sigma2 <= 0.0 {
            f64::INFINITY
        } else {
            n * sigma2.ln() + 2.0 * order
        };
        aic_values.push(aic);
    }

    // Find the index of the minimum AIC. Use `enumerate` with a fold so that
    // ties naturally resolve to the first (lower-order) occurrence.
    let selected_order = aic_values
        .iter()
        .enumerate()
        .fold(
            (0usize, f64::INFINITY),
            |(best_idx, best_val), (idx, &val)| {
                if val < best_val {
                    (idx, val)
                } else {
                    (best_idx, best_val)
                }
            },
        )
        .0;

    AicSelectionResult {
        selected_order,
        aic_values,
    }
}

// ---------------------------------------------------------------------------
// PACF-based AR order selection
// ---------------------------------------------------------------------------

/// Result of PACF-based AR order selection.
///
/// Produced by [`select_order_pacf`]. Contains the selected AR order, the
/// PACF values for each lag, and the significance threshold used.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct PacfSelectionResult {
    /// Selected AR order.
    ///
    /// The maximum lag `k` where `|parcor[k-1]| > threshold`.
    /// `0` when no lag exceeds the significance threshold.
    pub selected_order: usize,
    /// PACF values (partial autocorrelation coefficients) for lags `1..=p_max`.
    ///
    /// `pacf_values[k]` is the PACF at lag `k+1`. Same as
    /// `LevinsonDurbinResult::parcor`.
    pub pacf_values: Vec<f64>,
    /// Significance threshold: `z_alpha / sqrt(n_observations)`.
    pub threshold: f64,
}

/// Select the AR order using partial autocorrelation function (PACF)
/// significance testing.
///
/// For each lag `k` in `1..=p_max`, tests whether the partial
/// autocorrelation coefficient (reflection coefficient from Levinson-Durbin)
/// exceeds the significance threshold `z_alpha / sqrt(N)`. Selects the
/// **maximum** lag with a significant PACF value.
///
/// If no lag exceeds the threshold, order 0 is selected (white noise).
///
/// # Parameters
///
/// - `parcor` -- partial autocorrelation coefficients from
///   `LevinsonDurbinResult::parcor`. `parcor[k]` is the PACF at lag `k+1`.
/// - `n_observations` -- number of historical observations for this season.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf;
///
/// // PACF at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // No significant PACF values -> order 0.
/// let result = select_order_pacf(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf(
    parcor: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    // Find the maximum lag with |PACF| > threshold.
    let selected_order = parcor
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    PacfSelectionResult {
        selected_order,
        pacf_values: parcor.to_vec(),
        threshold,
    }
}

/// Select the AR order for a PAR(p)-A model using the conditional FACP
/// significance test, with two extensions over the classical PACF rule.
///
/// The classical rule is a single 95% confidence interval on the number
/// of historical years (`z_alpha / sqrt(N)`, no lag-dependent deflation):
/// the largest lag whose `|FACP|` exceeds the threshold is attributed
/// as `p_m`. The classical rule is silent on the cases where (a) lag 1
/// is exactly zero or (b) no lag is significant; the rules below cover
/// those cases:
///
/// 1. **Structural-zero short-circuit at lag 1.** If
///    `conditional_facp[0] == 0.0` exactly, the model is forced to
///    order 0. A structural zero at lag 1 indicates a degenerate (Z, A)
///    bucket — typically a single-observation season or a numerically
///    singular partitioned-covariance solve — and the convention refuses
///    to fit any auto-regressive structure on top of it. Structural
///    zeros at higher lags do **not** trigger the short-circuit; the
///    convention proceeds with the AR(1) base whenever lag 1 itself is
///    non-degenerate (e.g., `[+0.37, 0, 0, 0, 0, 0]` -> order 1, not 0).
/// 2. **Minimum order of 1 when lag 1 is non-zero.** If the conditional
///    FACP at lag 1 is not a structural zero, the selected order is
///    `max(1, max_significant_lag)` — the model defaults to AR(1)
///    whenever no lag exceeds the threshold but lag 1 is well defined.
///
/// The Maceira-Damazio iterative order-reduction step is **not** applied
/// here; the order returned by this function is the tentative
/// pre-validation order. The reduction runs across all seasons of the
/// periodic cycle and is the caller's responsibility within the
/// PAR(p)-A estimation pipeline.
///
/// `pacf_values[k]` in the returned struct is the conditional FACP at lag
/// `k+1`, conditioned on the intermediate standardised annual noise series
/// `Z` and the previous annual innovation `A_{t-1}`.
///
/// # Parameters
///
/// - `conditional_facp` -- conditional FACP coefficients from
///   [`conditional_facp_partitioned`]. `conditional_facp[k]` is the
///   conditional FACP at lag `k+1`.
/// - `n_observations` -- number of historical observations for the given
///   (hydro, season) pair.
/// - `z_alpha` -- z-score for the desired confidence level (e.g., `1.96`
///   for 95% two-sided).
///
/// # Examples
///
/// ```
/// use cobre_stochastic::par::fitting::select_order_pacf_annual;
///
/// // Conditional FACP at lag 1 = 0.5 exceeds 1.96/sqrt(100) = 0.196; lag 2 = 0.1 does not.
/// let result = select_order_pacf_annual(&[0.5, 0.1], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Lag 1 is non-zero (just small) -> min-order-1 rule kicks in.
/// let result = select_order_pacf_annual(&[0.05, 0.03], 100, 1.96);
/// assert_eq!(result.selected_order, 1);
///
/// // Structural zero at lag 1 -> order 0 (degenerate bucket).
/// let result = select_order_pacf_annual(&[0.0, 0.5], 100, 1.96);
/// assert_eq!(result.selected_order, 0);
/// ```
pub fn select_order_pacf_annual(
    conditional_facp: &[f64],
    n_observations: usize,
    z_alpha: f64,
) -> PacfSelectionResult {
    #[allow(clippy::cast_precision_loss)]
    let threshold = if n_observations > 0 {
        z_alpha / (n_observations as f64).sqrt()
    } else {
        f64::INFINITY
    };

    // Rule 1 — Structural-zero short-circuit at lag 1.
    // A structural zero at lag 1 (FACP exactly 0.0 from a degenerate
    // bucket) forces white-noise selection.
    if conditional_facp.first().copied() == Some(0.0) {
        return PacfSelectionResult {
            selected_order: 0,
            pacf_values: conditional_facp.to_vec(),
            threshold,
        };
    }

    // Find the maximum lag with |conditional FACP| > threshold.
    let max_significant = conditional_facp
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, p)| p.abs() > threshold)
        .map_or(0, |(k, _)| k + 1);

    // Rule 2 — Min-order-1 when lag 1 is non-zero (not a structural zero).
    // When no lag exceeds the significance threshold but lag 1 is well
    // defined, the model defaults to AR(1) rather than a pure white-noise
    // (order-0) fit.
    let selected_order = match conditional_facp.first() {
        Some(&p1) if p1 != 0.0 => max_significant.max(1),
        _ => max_significant,
    };

    PacfSelectionResult {
        selected_order,
        pacf_values: conditional_facp.to_vec(),
        threshold,
    }
}

// ---------------------------------------------------------------------------
// Periodic PACF
// ---------------------------------------------------------------------------

/// Compute the periodic PACF for a given season up to `max_order`.
///
/// For each candidate order k (`1..=max_order`), builds the periodic Yule-Walker
/// matrix of dimension k, solves `R * phi = rhs`, and extracts the last
/// coefficient `phi[k-1]` as `PACF(k)`.
///
/// This is the correct periodic PACF computation that accounts for the
/// non-Toeplitz covariance structure of periodic autoregressive processes.
/// It replaces the stationary Levinson-Durbin reflection coefficients which
/// assume a Toeplitz (stationary) covariance matrix.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `max_order` -- maximum lag to compute PACF for.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A `Vec<f64>` of length <= `max_order`. Entry `k` is `PACF(k+1)`.
/// The vector may be shorter than `max_order` if a system at some order
/// is singular (remaining orders are skipped).
#[must_use]
pub fn periodic_pacf(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> Vec<f64> {
    let mut pacf_values = Vec::with_capacity(max_order);

    // Reuse two scratch buffers across the loop to avoid allocating a new
    // Vec pair per order k. build_periodic_yw_matrix_into resizes them in
    // place (no-alloc when capacity already covers the new size).
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_buf: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        build_periodic_yw_matrix_into(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            &mut matrix_buf,
            &mut rhs_buf,
        );

        match solve_linear_system(&mut matrix_buf, &mut rhs_buf, k) {
            Some(phi) => pacf_values.push(phi[k - 1]),
            None => break, // Singular matrix, stop.
        }
    }

    pacf_values
}
