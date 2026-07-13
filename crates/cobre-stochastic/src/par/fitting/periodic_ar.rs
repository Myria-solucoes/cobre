//! Periodic Yule-Walker AR coefficient estimators (classical and PAR-A).

use super::yw_matrices::{
    build_extended_periodic_yw_matrix, build_periodic_yw_matrix_into, solve_linear_system,
};

/// Result of periodic Yule-Walker AR coefficient estimation for one
/// (entity, season) pair.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwResult {
    /// Standardised AR coefficients `phi_1..phi_p`.
    pub coefficients: Vec<f64>,
    /// Residual std ratio `sigma_residual / sigma_sample`, in `(0, 1]`; 1.0 for
    /// order-0.
    pub residual_std_ratio: f64,
    /// Prediction error variance per intermediate order; `sigma2_per_order[k-1]`
    /// is the variance for AR(k).
    pub sigma2_per_order: Vec<f64>,
}

/// Estimate AR coefficients by solving the periodic Yule-Walker system at
/// `selected_order`, also recording the prediction error variance per
/// intermediate order.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `selected_order` -- the AR order to fit.
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// Returns the order-0 result (empty coefficients, ratio 1.0) when
/// `selected_order == 0` or the system is singular.
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

    // Reused across the loop so build_periodic_yw_matrix_into resizes in place
    // rather than allocating a Vec pair per order k.
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_buf: Vec<f64> = Vec::new();

    for k in 1..=selected_order {
        build_periodic_yw_matrix_into(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            &mut matrix_buf,
            &mut rhs_buf,
        );
        // Clone only the rhs (O(k), not the O(k²) matrix): sigma2 needs the
        // pre-solve rhs, which solve_linear_system overwrites.
        let rhs_orig: Vec<f64> = rhs_buf.clone();

        match solve_linear_system(&mut matrix_buf, &mut rhs_buf, k) {
            Some(phi) => {
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
            None => return zero_result,
        }
    }

    let sigma2_final = *sigma2_per_order.last().unwrap_or(&1.0);
    // sigma2 = residual_std_ratio² = 1 − Σ φ_ℓ ρ(ℓ): the fraction of the season's
    // (unit) standardized variance left unexplained by the AR fit. sigma2 ≤ 0 means
    // Σ φ_ℓ ρ(ℓ) ≥ 1 — a degenerate near-unit-root fit where the AR part explains
    // ~all seasonal variance. We fall back to ratio = 1.0 (σ = s_m, maximum noise)
    // rather than 0: injecting full seasonal noise is conservative (never collapses
    // inflow to a deterministic value), at the cost of not tripping the r² < 0.01
    // overfit warning in validation.rs.
    let residual_std_ratio = if sigma2_final > 0.0 {
        sigma2_final.sqrt().clamp(f64::EPSILON, 1.0)
    } else {
        1.0
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
/// All fields are **standardised** (dimensionless), as for
/// `PeriodicYwResult::coefficients`; unit conversion to runtime coefficients
/// `(φ̂_j, ψ̂)` happens at `PrecomputedPar::build`, not here.
#[must_use]
#[derive(Debug, Clone)]
pub struct PeriodicYwAnnualResult {
    /// Standardised AR coefficients `φ_1..φ_p`; empty when `selected_order == 0`.
    pub coefficients: Vec<f64>,
    /// Standardised annual coefficient `ψ`.
    pub annual_coefficient: f64,
    /// Residual std ratio `σ_residual / σ_seasonal` in `(0, 1]`.
    pub residual_std_ratio: f64,
}

/// Estimate PAR-A coefficients `(φ_1..φ_p, ψ)` by solving the
/// `(selected_order + 1)`-dimensional extended periodic Yule-Walker system.
///
/// `selected_order == 0` solves the 1×1 system yielding only `ψ` (empty
/// `coefficients`). A singular system returns the zero fallback
/// (`coefficients: vec![]`, `annual_coefficient: 0.0`, `residual_std_ratio: 1.0`),
/// matching [`estimate_periodic_ar_coefficients`].
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

    // Solution layout: first `selected_order` entries are AR coefficients, last is ψ.
    let coefficients: Vec<f64> = solution[..selected_order].to_vec();
    let annual_coefficient: f64 = solution[selected_order];

    let sigma2: f64 = 1.0
        - solution
            .iter()
            .zip(rhs_orig.iter())
            .map(|(s, r)| s * r)
            .sum::<f64>();

    // sigma2 ≤ 0: degenerate near-unit-root fit (Σ solution·ρ ≥ 1); fall back to
    // ratio = 1.0 (max noise, conservative) as in `estimate_periodic_ar_coefficients`.
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
