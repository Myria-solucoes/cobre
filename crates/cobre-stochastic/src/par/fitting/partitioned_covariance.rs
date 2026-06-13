//! Conditional FACP for the PAR-A model via the partitioned-covariance approach.

use super::yw_matrices::{
    cross_correlation_a_z_neg1, cross_correlation_z_a, periodic_autocorrelation,
    solve_linear_system,
};

/// Partitioned covariance matrices for one `(season, k)` evaluation.
///
/// Stores the three sub-matrices used in the conditional FACP formula
/// `Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21`. Sizes depend on the lag `k`:
/// - `sigma_11`: 2×2 (always four entries).
/// - `sigma_12`: 2×k row-major (the conditioning set has `k` elements).
/// - `sigma_22`: k×k row-major symmetric matrix.
// Rationale: the sigma_11/sigma_12/sigma_22 names mirror the partitioned-covariance
// notation (Σ_11, Σ_12, Σ_22); dropping the common prefix to silence the lint would
// break the correspondence with the algorithm documentation.
#[allow(clippy::struct_field_names)]
pub(crate) struct PartitionedCov {
    /// 2×2 auto-covariance of `(Z_t, Z_{t−k})`, row-major.
    pub(super) sigma_11: [f64; 4],
    /// 2×k cross-covariance between `(Z_t, Z_{t−k})` and the conditioning
    /// set `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})`, row-major.
    pub(super) sigma_12: Vec<f64>,
    /// k×k auto-covariance of the conditioning set, row-major.
    pub(super) sigma_22: Vec<f64>,
}

/// Assemble the three sub-matrices of the partitioned covariance for the
/// conditional FACP at lag `k` from `season`.
///
/// # Matrix layout
///
/// The conditioning set at lag `k` is:
/// `(Z_{t−1}, Z_{t−2}, …, Z_{t−k+1}, A_{t−1})` — that is, `k−1` lagged
/// `Z` values followed by the annual component at season `m−1`.
///
/// **`sigma_11`** (2×2): auto-covariance of `(Z_t, Z_{t−k})`.
/// - `[0,0] = 1` (unit variance of standardised `Z_t`)
/// - `[0,1] = [1,0] = ρ^{season}(k)`
/// - `[1,1] = 1`
///
/// **`sigma_22`** (k×k): auto-covariance of the conditioning set.
/// - Rows/cols `i,j < k−1`: `ρ^{season−1}(|i−j|)` — periodic autocorrelation
///   of the `Z` block at season `m−1`. Diagonal entries are `1.0`.
/// - Row/col `k−1` (the `A_{t−1}` entry):
///   - Off-diagonal: `cross_correlation_z_a(season−1, i, …)` for `i < k−1`.
///   - Diagonal `[k−1, k−1] = 1.0` (unit variance of standardised `A_{t−1}`).
///
/// **`sigma_12`** (2×k): cross-covariance between `(Z_t, Z_{t−k})` and the
/// conditioning set.
/// - Row 0 (from `Z_t`), column `j < k−1`: `ρ^{season}(j+1)`.
/// - Row 0, column `k−1`: `cross_correlation_a_z_neg1(season−1, …)`.
/// - Row 1 (from `Z_{t−k}`), column `j < k−1`: `ρ^{season−k}(k−1−j)`.
/// - Row 1, column `k−1`: `cross_correlation_z_a(season−1, k−1, …)`.
pub(crate) fn assemble_partitioned_covariance(
    season: usize,
    k: usize,
    n_seasons: usize,
    obs_z: &[&[f64]],
    stats_z: &[(f64, f64)],
    z_year_starts: &[i32],
    obs_a: &[&[f64]],
    stats_a: &[(f64, f64)],
    a_year_starts: &[i32],
) -> PartitionedCov {
    let prev_season = (season + n_seasons - 1) % n_seasons;

    // Σ_11: 2×2 auto-covariance of (Z_t, Z_{t−k}).
    // Row-major: [0,0]=1, [0,1]=ρ^season(k), [1,0]=ρ^season(k), [1,1]=1.
    let rho_k = periodic_autocorrelation(season, k, n_seasons, obs_z, stats_z);
    let sigma_11 = [1.0, rho_k, rho_k, 1.0];

    // Σ_22: k×k auto-covariance of conditioning set.
    // Conditioning set: (Z_{t−1}, …, Z_{t−k+1}, A_{t−1})  (k elements).
    let mut sigma_22 = vec![0.0_f64; k * k];

    // Z-block: rows/cols 0..k−1, all against season prev_season.
    // Diagonal is always 1.0 (unit variance). Off-diagonal: symmetric.
    for i in 0..k.saturating_sub(1) {
        sigma_22[i * k + i] = 1.0;
        let ref_month = (prev_season + n_seasons - i % n_seasons) % n_seasons;
        for j in (i + 1)..k.saturating_sub(1) {
            let lag = j - i;
            let rho = periodic_autocorrelation(ref_month, lag, n_seasons, obs_z, stats_z);
            sigma_22[i * k + j] = rho;
            sigma_22[j * k + i] = rho;
        }
    }

    // Cross-terms between the Z-block and A_{t−1} (column/row k−1).
    // sigma_22[i, k−1] = Corr(Z_{t−1−i}, A_{t−1}).
    // Z_{t−1−i} is `i` steps older than A_{t−1}, so lag = i.
    // for i in 0..k−1 (and symmetrically sigma_22[k−1, i]).
    // Note: for k=1 there is no Z-block (k.saturating_sub(1)=0), so this
    // loop body is never entered.
    for i in 0..k.saturating_sub(1) {
        let lag = i;
        let rho = cross_correlation_z_a(
            prev_season,
            lag,
            n_seasons,
            obs_z,
            stats_z,
            z_year_starts,
            obs_a,
            stats_a,
            a_year_starts,
        );
        sigma_22[i * k + (k - 1)] = rho;
        sigma_22[(k - 1) * k + i] = rho;
    }

    // Diagonal entry for A_{t−1}: unit variance by construction.
    sigma_22[(k - 1) * k + (k - 1)] = 1.0;

    // Σ_12: 2×k cross-covariance between (Z_t, Z_{t−k}) and conditioning set.
    let mut sigma_12 = vec![0.0_f64; 2 * k];

    // Row 0 (Z_t) with Z-block of conditioning set.
    for (j, entry) in sigma_12[..k.saturating_sub(1)].iter_mut().enumerate() {
        let rho = periodic_autocorrelation(season, j + 1, n_seasons, obs_z, stats_z);
        *entry = rho;
    }
    // Row 0 (Z_t) with A_{t−1}.
    sigma_12[k - 1] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    // Row 1 (Z_{t−k}) with Z-block of conditioning set.
    // Position j (0..k−2) of this row is Corr(Z_{t−k}, Z_{t−1−j}).
    // Z_{t−1−j} is the **newer** of the two (since j ≤ k−2 < k−1 < k), so
    // ρ_periodic must be anchored at its season with lag = (k−1) − j.
    for (j, entry) in sigma_12[k..k + k.saturating_sub(1)].iter_mut().enumerate() {
        let ref_season = (season + n_seasons - 1 - j) % n_seasons;
        let lag = k.saturating_sub(1).saturating_sub(j);
        let rho = periodic_autocorrelation(ref_season, lag, n_seasons, obs_z, stats_z);
        *entry = rho;
    }
    // Row 1 (Z_{t−k}) with A_{t−1}.
    sigma_12[k + (k - 1)] = cross_correlation_z_a(
        prev_season,
        k - 1,
        n_seasons,
        obs_z,
        stats_z,
        z_year_starts,
        obs_a,
        stats_a,
        a_year_starts,
    );

    PartitionedCov {
        sigma_11,
        sigma_12,
        sigma_22,
    }
}

/// Compute the conditional FACP for the PAR-A model up to `max_order`.
///
/// For each candidate lag `k` (`1..=max_order`), the conditioning set is
/// `(Z_{t−1}, …, Z_{t−k+1}, A_{t−1})` — the `k−1` intermediate standardised
/// inflow values plus the standardised annual component at season `m−1`. The
/// conditional correlation is obtained from the partitioned-covariance formula:
///
/// ```text
/// Σ̄ = Σ_11 − Σ_12 · Σ_22⁻¹ · Σ_21
/// ```
///
/// where `Σ_21 = Σ_12ᵀ`. The conditional FACP at lag `k` is
/// `Σ̄[0,1] / √(Σ̄[0,0] · Σ̄[1,1])`, clamped to `[−1, 1]`.
///
/// # Parameters
///
/// - `season` — 0-based target season (the "current" season `m`).
/// - `max_order` — maximum lag to evaluate. Returns `Vec::new()` when zero.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — periodic inflow series `Z`, grouped by season.
///   Entry `[s][y]` is the standardised observation for season `s` in year `y`.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   Used by the cross-correlation helpers to align `A` and `Z` by absolute
///   PDF year rather than by bucket index — required for monthly data where
///   `A` buckets start one year later than `Z` for most seasons.
/// - `annual_observations_by_season` — annual component `A`, grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// A `Vec<f64>` of length `≤ max_order`. Entry `i` is the conditional FACP at
/// lag `i+1`, clamped to `[−1.0, 1.0]`.
///
/// The vector is shorter than `max_order` when `Σ_22` is singular at some lag
/// `k` — the loop breaks early and entries for lags `≥ k` are omitted.
/// When `Σ̄[0,0] · Σ̄[1,1] ≤ 0` (numerical degeneracy), the affected entry is
/// recorded as `0.0`.
#[must_use]
pub fn conditional_facp_partitioned(
    season: usize,
    max_order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> Vec<f64> {
    if max_order == 0 {
        return Vec::new();
    }

    let mut facp_values = Vec::with_capacity(max_order);

    // Reusable scratch buffers for solve_linear_system calls.
    // The Σ_22 matrix and RHS column are cloned per solve; the buffers below
    // hold the cloned working copies so we avoid re-allocating each iteration.
    let mut matrix_buf: Vec<f64> = Vec::new();
    let mut rhs_col: Vec<f64> = Vec::new();

    for k in 1..=max_order {
        let cov = assemble_partitioned_covariance(
            season,
            k,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );

        // Solve Σ_22 · X = Σ_21 column-by-column (2 columns, one per row of
        // Σ_12). Σ_21 = Σ_12ᵀ, so column c of Σ_21 = row c of Σ_12.
        // X has shape k×2; store solutions as two Vec<f64> of length k.
        let mut x_cols: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut singular = false;

        for (col_idx, x_col) in x_cols.iter_mut().enumerate() {
            // Copy Σ_22 into working buffer (solve_linear_system modifies in-place).
            matrix_buf.clear();
            matrix_buf.extend_from_slice(&cov.sigma_22);

            // Column col_idx of Σ_21 = row col_idx of Σ_12 (k entries).
            rhs_col.clear();
            for row in 0..k {
                rhs_col.push(cov.sigma_12[col_idx * k + row]);
            }

            if let Some(sol) = solve_linear_system(&mut matrix_buf, &mut rhs_col, k) {
                *x_col = sol;
            } else {
                singular = true;
                break;
            }
        }

        if singular {
            // Singular Σ_22 — stop the loop and return results so far.
            break;
        }

        // Compute Σ̄ = Σ_11 − Σ_12 · X  (2×2 result).
        //
        // Σ_12 is 2×k, X is k×2.
        // [Σ_12 · X][r, c] = sum_{j=0}^{k-1} Σ_12[r,j] * X[j,c]
        //                  = sum_{j=0}^{k-1} sigma_12[r*k + j] * x_cols[c][j]
        //
        // sigma_bar is stored row-major: [0,0], [0,1], [1,0], [1,1].
        let mut sigma_bar = cov.sigma_11;
        for r in 0..2 {
            for c in 0..2 {
                let correction: f64 = (0..k).map(|j| cov.sigma_12[r * k + j] * x_cols[c][j]).sum();
                sigma_bar[r * 2 + c] -= correction;
            }
        }

        // Extract conditional FACP: sigma_bar[0,1] / sqrt(sigma_bar[0,0] * sigma_bar[1,1]).
        // Guard against non-positive product (numerical degeneracy).
        let denom_sq = sigma_bar[0] * sigma_bar[3];
        let facp = if denom_sq <= 0.0 {
            0.0
        } else {
            (sigma_bar[1] / denom_sq.sqrt()).clamp(-1.0, 1.0)
        };

        facp_values.push(facp);
    }

    facp_values
}
