//! Periodic Yule-Walker AR coefficient estimators (classical and PAR-A).

use super::yw_matrices::{
    build_extended_periodic_yw_matrix, build_periodic_yw_matrix, solve_linear_system,
};

// ---------------------------------------------------------------------------
// Periodic YW coefficient estimation
// ---------------------------------------------------------------------------

/// Result of periodic Yule-Walker AR coefficient estimation for one
/// (entity, season) pair.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwResult {
    /// Standardised AR coefficients `phi_1..phi_p`.
    pub coefficients: Vec<f64>,
    /// Residual std ratio: `sigma_residual / sigma_sample`.
    /// In `(0, 1]` for valid models; 1.0 for order-0.
    pub residual_std_ratio: f64,
    /// Prediction error variance at each intermediate order `1..=selected_order`.
    /// Used for diagnostic reporting. `sigma2_per_order[k-1]` is the variance
    /// for AR(k).
    pub sigma2_per_order: Vec<f64>,
}

/// Estimate AR coefficients by solving the periodic Yule-Walker system at the
/// given order.
///
/// Also computes prediction error variances at each intermediate order
/// (`1..=selected_order`) for diagnostic reporting compatibility.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `selected_order` -- the AR order to fit (from PACF-based selection).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A [`PeriodicYwResult`] with the fitted coefficients, residual std ratio,
/// and prediction error variances. Returns order-0 result (empty coefficients,
/// ratio 1.0) when `selected_order == 0` or when the system is singular.
pub fn estimate_periodic_ar_coefficients(
    season: usize,
    selected_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> PeriodicYwResult {
    let zero_result = PeriodicYwResult {
        coefficients: Vec::new(),
        residual_std_ratio: 1.0,
        sigma2_per_order: Vec::new(),
    };

    if selected_order == 0 {
        return zero_result;
    }

    let mut sigma2_per_order = Vec::with_capacity(selected_order);
    let mut final_coefficients = Vec::new();

    for k in 1..=selected_order {
        // Build and solve the periodic YW system at order k.
        // Save the original RHS for sigma2 computation (solve modifies in-place).
        // Build the matrix once; clone rhs (O(k)) before the in-place solve.
        let (mut matrix, mut rhs) = build_periodic_yw_matrix(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
        let rhs_orig: Vec<f64> = rhs.clone();

        match solve_linear_system(&mut matrix, &mut rhs, k) {
            Some(phi) => {
                // sigma2(k) = 1 - sum_{j=0}^{k-1} phi[j] * rhs_original[j]
                let sigma2_k: f64 = 1.0
                    - phi
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(p, r)| p * r)
                        .sum::<f64>();
                sigma2_per_order.push(sigma2_k);

                if k == selected_order {
                    final_coefficients = phi;
                }
            }
            None => {
                // Singular matrix: fall back to order-0 result.
                return zero_result;
            }
        }
    }

    // Compute residual_std_ratio = sqrt(sigma2(selected_order)).
    let sigma2_final = *sigma2_per_order.last().unwrap_or(&1.0);
    let residual_std_ratio = if sigma2_final > 0.0 {
        sigma2_final.sqrt().clamp(f64::EPSILON, 1.0)
    } else {
        1.0 // Numerical issue: fall back.
    };

    PeriodicYwResult {
        coefficients: final_coefficients,
        residual_std_ratio,
        sigma2_per_order,
    }
}

// ---------------------------------------------------------------------------
// Extended periodic YW coefficient estimation (PAR-A)
// ---------------------------------------------------------------------------

/// Result of the extended periodic Yule-Walker solve for the PAR-A model.
///
/// Returned by [`estimate_periodic_ar_annual_coefficients`]. The three fields
/// are **standardised** (dimensionless), matching the convention for the classical
/// `PeriodicYwResult::coefficients`. Unit conversion to runtime coefficients
/// `(φ̂_j, ψ̂)` happens at `PrecomputedPar::build` time, not here.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwAnnualResult {
    /// Standardised AR coefficients `φ_1..φ_p` (dimensionless, direct Yule-Walker
    /// output). Empty when `selected_order == 0`.
    pub coefficients: Vec<f64>,
    /// Standardised annual coefficient `ψ` (dimensionless, direct Yule-Walker
    /// output).
    pub annual_coefficient: f64,
    /// Residual std ratio `σ_residual / σ_seasonal` in `(0, 1]`.
    pub residual_std_ratio: f64,
}

/// Estimate PAR-A coefficients `(φ_1..φ_p, ψ)` by solving the extended
/// periodic Yule-Walker system.
///
/// Builds the `(selected_order + 1) × (selected_order + 1)` system via
/// [`build_extended_periodic_yw_matrix`] and solves it via
/// [`solve_linear_system`].
///
/// ## Singular-system fallback
///
/// When the system is singular (the solver returns `None`), the function
/// returns `PeriodicYwAnnualResult { coefficients: vec![], annual_coefficient:
/// 0.0, residual_std_ratio: 1.0 }`, matching the classical fallback in
/// [`estimate_periodic_ar_coefficients`].
///
/// ## Order-0 case
///
/// When `selected_order == 0`, the function solves the 1×1 system that yields
/// only `ψ`. The returned `coefficients` is empty.
///
/// ## Residual std ratio
///
/// `sigma2 = 1 - Σ_i (solution[i] · rhs_orig[i])` over all `selected_order + 1`
/// solution entries. `residual_std_ratio = sqrt(sigma2).clamp(f64::EPSILON, 1.0)`
/// when `sigma2 > 0`, else `1.0`.
pub fn estimate_periodic_ar_annual_coefficients(
    season: usize,
    selected_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> PeriodicYwAnnualResult {
    let zero_result = PeriodicYwAnnualResult {
        coefficients: Vec::new(),
        annual_coefficient: 0.0,
        residual_std_ratio: 1.0,
    };

    let (mut matrix, mut rhs) = build_extended_periodic_yw_matrix(
        season,
        selected_order,
        n_seasons,
        observations_by_season,
        stats_by_season,
        z_year_starts,
        annual_observations_by_season,
        annual_stats_by_season,
        a_year_starts,
    );

    let dim = selected_order + 1;
    let rhs_orig: Vec<f64> = rhs.clone();

    let Some(solution) = solve_linear_system(&mut matrix, &mut rhs, dim) else {
        return zero_result;
    };

    // Split: first `selected_order` entries are AR coefficients; last is ψ.
    let coefficients: Vec<f64> = solution[..selected_order].to_vec();
    let annual_coefficient: f64 = solution[selected_order];

    // sigma2 = 1 - sum(solution[i] * rhs_orig[i]) for all i in 0..dim.
    let sigma2: f64 = 1.0
        - solution
            .iter()
            .zip(rhs_orig.iter())
            .map(|(s, r)| s * r)
            .sum::<f64>();

    let residual_std_ratio = if sigma2 > 0.0 {
        sigma2.sqrt().clamp(f64::EPSILON, 1.0)
    } else {
        1.0
    };

    PeriodicYwAnnualResult {
        coefficients,
        annual_coefficient,
        residual_std_ratio,
    }
}
