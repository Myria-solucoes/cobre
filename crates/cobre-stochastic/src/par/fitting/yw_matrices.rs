//! Periodic Yule-Walker matrix construction and the dense linear solver.
//!
//! Provides the periodic autocorrelation primitive, the classical and extended
//! Yule-Walker matrix builders, the annual cross-correlation primitives, and the
//! small Gaussian-elimination solver shared across the fitter.

// ---------------------------------------------------------------------------
// Periodic autocorrelation
// ---------------------------------------------------------------------------

/// Compute the periodic normalised autocorrelation `rho(p, k)` for a given
/// reference season `p` and lag `k`.
///
/// The periodic autocorrelation differs from a stationary autocorrelation
/// in that the reference season determines both the "current" observations
/// and their seasonal statistics, while the lag determines the "lagged"
/// observations and their statistics.
///
/// Uses population divisor (1/N) and cross-year lag adjustment.
///
/// # Parameters
///
/// - `ref_season` -- 0-based season index of the reference month `p`.
/// - `lag` -- lag in seasonal periods (1-based: lag=1 means one season back).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season index.
///   `observations_by_season[s]` contains all historical values for season `s`,
///   in chronological order.
/// - `stats_by_season` -- `(mean, std)` for each season, indexed by season.
///
/// # Returns
///
/// The normalised autocorrelation value `rho(ref_season, lag)`, clamped to [-1, 1].
/// Returns 0.0 when either the reference or lagged season has zero std,
/// or when insufficient paired observations exist.
#[must_use]
pub fn periodic_autocorrelation(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> f64 {
    // Lag 0 is the identity: rho(m, 0) = 1.0 by normalisation.
    if lag == 0 {
        return 1.0;
    }

    // Compute lagged season index.
    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;

    let (mu_ref, std_ref) = stats_by_season[ref_season];
    let (mu_lag, std_lag) = stats_by_season[lag_season];

    // Zero-std guard: autocorrelation is undefined.
    if std_ref.abs() < f64::EPSILON || std_lag.abs() < f64::EPSILON {
        return 0.0;
    }

    let ref_obs = observations_by_season[ref_season];
    let lag_obs = observations_by_season[lag_season];

    // Cross-year lag adjustment: the number of year boundaries crossed by
    // a lag of `k` seasons determines how many observations must be dropped.
    //
    // A lag that stays within the same calendar year (lag_season < ref_season
    // and lag < n_seasons) crosses 0 boundaries. Otherwise, the number of
    // full years spanned is `(lag + n_seasons - 1) / n_seasons` when the lag
    // crosses into an earlier calendar position, or `lag / n_seasons` when
    // it wraps full cycles.
    //
    // Approach: for lag `k` within one cycle (k < n_seasons),
    // detect cross-year when lag_season >= ref_season. For larger lags,
    // additional years are spanned. The total drop count equals the number
    // of year boundaries crossed.
    let years_crossed = if lag < n_seasons {
        usize::from(lag_season >= ref_season)
    } else {
        // Full years from the lag.
        lag / n_seasons
    };

    let ref_start = years_crossed;
    let n_pairs = ref_obs
        .len()
        .saturating_sub(years_crossed)
        .min(lag_obs.len());

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with population divisor (1/N) over the year-aligned
    // valid pairs. The convention uses N = n_pairs here for Z⊗Z
    // autocorrelations. The max-bucket-size convention used by
    // cross_correlation_z_a / cross_correlation_a_z_neg1 only applies to
    // Z⊗A cross-terms because those buckets have inherently different
    // lengths (A excludes the first year of Z by construction).
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (ref_obs[ref_start + i] - mu_ref) * (lag_obs[i] - mu_lag);
    }
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= n_pairs as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_ref * std_lag);
    rho.clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Periodic Yule-Walker matrix
// ---------------------------------------------------------------------------

/// Build the periodic Yule-Walker matrix and right-hand side for a given
/// season and AR order.
///
/// Solves the **forward prediction** problem: predict `z_m` from
/// `{z_{m-1}, ..., z_{m-p}}`. The matrix uses rows 1..p of the extended
/// `(order+1) x (order+1)` covariance matrix, and the RHS uses column 0.
///
/// The matrix has dimension `order x order`. Entry `R[i,j]` is the
/// periodic autocorrelation `rho(season - (i+1), |j - i|)`, where the
/// reference month shifts per row (starting one step before `season`).
/// The matrix is symmetric but NOT Toeplitz because the autocorrelation
/// function varies with the reference period.
///
/// The right-hand side vector `rhs[i] = rho(season, i+1)` is anchored at
/// the target season with lags 1..p (column 0 of the extended matrix).
///
/// # Parameters
///
/// - `season` -- 0-based target season for the YW system.
/// - `order` -- AR order (determines matrix dimension: `order x order`).
/// - `n_seasons` -- total number of seasons in the periodic cycle.
/// - `observations_by_season` -- observations grouped by season, chronological order.
/// - `stats_by_season` -- `(mean, std)` for each season.
///
/// # Returns
///
/// A tuple `(matrix, rhs)` where:
/// - `matrix` is a flat `Vec<f64>` of length `order * order` in row-major layout.
///   Entry `R[i][j]` is at index `i * order + j`.
/// - `rhs` is a `Vec<f64>` of length `order`.
///
/// Returns `(vec![], vec![])` when `order == 0`.
#[must_use]
pub fn build_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
) -> (Vec<f64>, Vec<f64>) {
    #[cfg(test)]
    BUILD_PERIODIC_YW_MATRIX_CALL_COUNT.with(|c| {
        *c.borrow_mut() += 1;
    });

    if order == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut matrix = vec![0.0_f64; order * order];
    let mut rhs = vec![0.0_f64; order];

    // Fill the matrix: R[i][j] = rho(season - (i+1), |j - i|).
    // Diagonal is always 1.0 (rho(m, 0) = 1 for any m).
    // The inner `(i + 1) % n_seasons` prevents underflow when order > n_seasons.
    for i in 0..order {
        matrix[i * order + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix[i * order + j] = rho;
            matrix[j * order + i] = rho; // symmetric
        }
    }

    // Fill the RHS: rhs[i] = rho(season, i+1).
    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }

    (matrix, rhs)
}

/// Write the periodic Yule-Walker matrix and RHS into caller-supplied flat
/// buffers, resizing them to `order * order` and `order` respectively.
///
/// This variant avoids allocating new `Vec`s on each call, which matters when
/// `periodic_pacf` builds systems of increasing dimension in a loop.
///
/// # Parameters
///
/// - `season` -- 0-based target season.
/// - `order` -- AR order (`matrix_out` will hold `order * order` entries).
/// - `n_seasons` -- total number of seasons.
/// - `observations_by_season` -- observations grouped by season.
/// - `stats_by_season` -- `(mean, std)` per season.
/// - `matrix_out` -- caller-allocated flat buffer; resized and overwritten.
/// - `rhs_out` -- caller-allocated buffer; resized and overwritten.
///
/// No-op when `order == 0` (buffers are cleared to empty).
pub fn build_periodic_yw_matrix_into(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    matrix_out: &mut Vec<f64>,
    rhs_out: &mut Vec<f64>,
) {
    if order == 0 {
        matrix_out.clear();
        rhs_out.clear();
        return;
    }

    matrix_out.resize(order * order, 0.0_f64);
    // Zero-fill: entries are written below, but the symmetric fill relies on
    // the upper triangle being populated before mirroring.
    matrix_out.fill(0.0);
    rhs_out.resize(order, 0.0_f64);

    for i in 0..order {
        matrix_out[i * order + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix_out[i * order + j] = rho;
            matrix_out[j * order + i] = rho;
        }
    }

    for (i, rhs_entry) in rhs_out.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }
}

// ---------------------------------------------------------------------------
// Extended periodic Yule-Walker matrix (annual component)
// ---------------------------------------------------------------------------

/// Compute the cross-correlation between the annual component series `A` at
/// season `ref_season` (with lag 0 meaning the same observation index) and the
/// periodic series `Z` at season `ref_season - lag` (wrapping), for non-negative
/// lag values.
///
/// This corresponds to `ρ_{Z,A}^{ref_season}(lag)` from the PAR-A extended
/// Yule-Walker system. The cross-correlation is defined as:
///
/// ```text
/// ρ_{Z,A}^{ref}(k) =
///   E[(A_i − μ^A_{ref}) · (Z_{i−k} − μ_{ref−k})] / (σ^A_{ref} · σ_{ref−k})
/// ```
///
/// where the expectation is approximated with a population (1/N) divisor.
///
/// # Cross-year alignment
///
/// Two distinct year offsets compose:
///
/// 1. **Bucket year offset** (`year_diff`). The `A` and `Z` buckets can have
///    different starting PDF years per season. For monthly data starting on
///    January, `Z` starts at year `Y0` for every season but `A` starts at
///    `Y0 + 1` for seasons 0..10 and `Y0` for season 11 (because the rolling
///    12-month window needs a full year of look-back). Calling code passes
///    `z_year_starts` and `a_year_starts` (one entry per season) so that the
///    pairing aligns by absolute PDF year, not by bucket index.
///
/// 2. **Lag year wrap** (`pdf_year_back_shift`). When stepping back `lag` months
///    from `ref_season`, if the lagged season index wraps (`lag_season >
///    ref_season` for `lag < n_seasons`, or `lag / n_seasons` for larger lags),
///    `Z`'s PDF year is one (or more) earlier than `A`'s.
///
/// **Lag-0 special case**: when `lag == 0`, `A` and `Z` refer to the same
/// season and the same regression year, so `pdf_year_back_shift = 0`
/// unconditionally — without this guard the `lag_season >= ref_season` branch
/// would falsely cross a year boundary.
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `lag` — non-negative integer lag; `lag = 0` pairs each `A_i` with `Z_i`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   When all entries are equal, the legacy by-index pairing is recovered.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation, clamped to `[-1.0, 1.0]`.
/// Returns `0.0` when either standard deviation is below [`f64::EPSILON`],
/// or when insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_z_a(
    ref_season: usize,
    lag: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let lag_season = (ref_season + n_seasons - lag % n_seasons) % n_seasons;
    let (mu_z, std_z) = stats_by_season[lag_season];

    // Zero-std guard: cross-correlation is undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[lag_season];

    // Lag-direction year wrap: how many year boundaries the lag traverses
    // backward from ref_season to lag_season.
    let pdf_year_back_shift = if lag == 0 {
        0
    } else if lag < n_seasons {
        usize::from(lag_season >= ref_season)
    } else {
        lag / n_seasons
    };

    // Bucket year offset between A's first PDF year and Z's first PDF year.
    let year_diff = i64::from(a_year_starts[ref_season]) - i64::from(z_year_starts[lag_season]);
    #[allow(clippy::cast_possible_wrap)]
    let shift = year_diff - pdf_year_back_shift as i64;

    // Pairing is `(a_obs[a_start + k], z_obs[z_start + k])` for k = 0..n_pairs.
    // shift > 0 ⇒ skip extra Z entries at start (Z starts earlier); shift < 0 ⇒
    // skip extra A entries at start (A starts earlier).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let (a_start, z_start) = if shift >= 0 {
        (0_usize, shift as usize)
    } else {
        ((-shift) as usize, 0_usize)
    };

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with the max-bucket-size population divisor.
    //
    // Sum runs over the year-aligned valid pairs, but the divisor is the
    // **maximum bucket size** (typically the Z bucket = total study-window
    // years). This is equivalent to padding the missing-A years with the
    // sample mean (their cross-product contribution is zero) while keeping
    // the observed σ̂_A computed over the genuinely populated entries.
    // Using n_pairs (the strict-pair count) here would systematically
    // overstate ρ̂(Z, A) by a factor of `max_len / n_pairs` and tilt
    // downstream partitioned-covariance FACPs across the threshold
    // boundary.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Compute the "lag = -1" cross-correlation between the annual component `A`
/// at season `ref_season` and the periodic series `Z` at the **next** season
/// `(ref_season + 1) % n_seasons`.
///
/// This corresponds to `ρ_{Z,A}^{ref_season}(-1)` — equivalently,
/// `ρ_{A,Z}^{ref_season}(+1)` with the arguments reversed. In the PAR-A
/// extended Yule-Walker RHS (eq. 15), this entry pairs `A_{t-1}` (season `m-1`)
/// with `Z_t` (season `m`), so `Z` is one step **ahead** of `A`.
///
/// # Cross-year alignment
///
/// Two distinct year offsets compose:
///
/// 1. **Bucket year offset** (`year_diff`). `A` and `Z` buckets can have
///    different starting PDF years per season (see [`cross_correlation_z_a`]
///    docs). Pairing aligns by absolute PDF year using `z_year_starts` and
///    `a_year_starts`.
///
/// 2. **Lag year wrap forward**. When `z_season = (ref_season + 1) % n_seasons`
///    wraps to 0, `Z` belongs to the next regression year relative to `A`.
///
/// # Parameters
///
/// - `ref_season` — 0-based season index for the `A` series.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
/// - `annual_observations_by_season` — `A` observations grouped by season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// The normalised cross-correlation, clamped to `[-1.0, 1.0]`.
/// Returns `0.0` when either standard deviation is below [`f64::EPSILON`],
/// or when insufficient paired observations exist.
#[must_use]
pub fn cross_correlation_a_z_neg1(
    ref_season: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> f64 {
    let (mu_a, std_a) = annual_stats_by_season[ref_season];
    let z_season = (ref_season + 1) % n_seasons;
    let (mu_z, std_z) = stats_by_season[z_season];

    // Zero-std guard: cross-correlation is undefined.
    if std_a.abs() < f64::EPSILON || std_z.abs() < f64::EPSILON {
        return 0.0;
    }

    let a_obs = annual_observations_by_season[ref_season];
    let z_obs = observations_by_season[z_season];

    // Z is one PDF month after A. When (ref_season + 1) wraps to 0, the
    // regression year of Z is one greater than A's.
    let pdf_year_forward_shift = usize::from(z_season == 0);

    let year_diff = i64::from(a_year_starts[ref_season]) - i64::from(z_year_starts[z_season]);
    #[allow(clippy::cast_possible_wrap)]
    let shift = year_diff + pdf_year_forward_shift as i64;

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let (a_start, z_start) = if shift >= 0 {
        (0_usize, shift as usize)
    } else {
        ((-shift) as usize, 0_usize)
    };

    let n_pairs = a_obs
        .len()
        .saturating_sub(a_start)
        .min(z_obs.len().saturating_sub(z_start));

    // Insufficient data guard.
    if n_pairs == 0 {
        return 0.0;
    }

    // Cross-covariance with the max-bucket-size population divisor — see
    // [`cross_correlation_z_a`] for the rationale on dividing by the larger
    // bucket size rather than `n_pairs`.
    let mut gamma = 0.0_f64;
    for i in 0..n_pairs {
        gamma += (a_obs[a_start + i] - mu_a) * (z_obs[z_start + i] - mu_z);
    }
    let denom_n = a_obs.len().max(z_obs.len());
    #[allow(clippy::cast_precision_loss)]
    {
        gamma /= denom_n as f64;
    }

    // Normalise and clamp.
    let rho = gamma / (std_a * std_z);
    rho.clamp(-1.0, 1.0)
}

/// Build the extended periodic Yule-Walker matrix and right-hand side for the
/// PAR-A model, augmenting the classical `order × order` system with one
/// extra row and column for the annual component coefficient `ψ`.
///
/// The returned matrix has dimension `(order+1) × (order+1)`. Its layout is:
///
/// ```text
/// [ classical_yw_matrix  |  cross_col ]
/// [ cross_row            |  1.0       ]
/// ```
///
/// where:
/// - The top-left `order × order` block is the classical periodic YW matrix
///   (same as [`build_periodic_yw_matrix`] for `season` and `order`).
/// - The top-right column (indices `i * (order+1) + order` for `i < order`) and
///   the bottom-left row (indices `order * (order+1) + j` for `j < order`) are
///   filled symmetrically with
///   `cross_correlation_z_a((season + n_seasons − 1) % n_seasons, i, …)`.
/// - The bottom-right diagonal entry `(order, order)` is `1.0`.
/// - `rhs[0..order]` mirrors the classical YW right-hand side.
/// - `rhs[order]` is `cross_correlation_a_z_neg1((season + n_seasons − 1) % n_seasons, …)`.
///
/// All entries are normalised correlations, so the matrix is symmetric and has
/// unit diagonal. The solution vector `[φ_1, …, φ_p, ψ]` contains
/// **standardised** coefficients. Unit conversion is applied later by the
/// caller.
///
/// # Parameters
///
/// - `season` — 0-based target season.
/// - `order` — AR order `p`; the returned matrix has dimension `(order+1) × (order+1)`.
/// - `n_seasons` — total number of seasons in the periodic cycle.
/// - `observations_by_season` — `Z` observations grouped by season, chronological.
/// - `stats_by_season` — `(mean, std)` for each `Z` season.
/// - `z_year_starts` — first PDF year of each `Z` bucket, indexed by season.
///   Threaded through to the cross-correlation helpers for absolute-year
///   alignment between `A` and `Z` (see [`cross_correlation_z_a`] docs).
/// - `annual_observations_by_season` — annual component `A` observations grouped by
///   season.
/// - `annual_stats_by_season` — `(mean, std)` for each `A` season.
/// - `a_year_starts` — first PDF year of each `A` bucket, indexed by season.
///
/// # Returns
///
/// A tuple `(matrix, rhs)` where:
/// - `matrix` is a flat `Vec<f64>` of length `(order+1)²` in row-major layout.
///   Entry `R[i][j]` is at index `i * (order+1) + j`.
/// - `rhs` is a `Vec<f64>` of length `order+1`.
///
/// When `order == 0`, returns `(vec![1.0], vec![rhs_annual_neg1])` — a 1×1
/// system that solves directly for `ψ`.
#[must_use]
pub fn build_extended_periodic_yw_matrix(
    season: usize,
    order: usize,
    n_seasons: usize,
    observations_by_season: &[&[f64]],
    stats_by_season: &[(f64, f64)],
    z_year_starts: &[i32],
    annual_observations_by_season: &[&[f64]],
    annual_stats_by_season: &[(f64, f64)],
    a_year_starts: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let dim = order + 1;
    let prev_season = (season + n_seasons - 1) % n_seasons;

    let mut matrix = vec![0.0_f64; dim * dim];
    let mut rhs = vec![0.0_f64; dim];

    // Fill the top-left order × order block (classical periodic YW structure).
    // Diagonal entries are 1.0 (autocorrelation at lag 0).
    // Off-diagonal: R[i][j] = periodic_autocorrelation(ref_month, |j-i|, ...)
    // where ref_month = (season - (i+1)) % n_seasons.
    for i in 0..order {
        matrix[i * dim + i] = 1.0;
        let ref_month = (season + n_seasons - (i + 1) % n_seasons) % n_seasons;
        for j in (i + 1)..order {
            let lag = j - i;
            let rho = periodic_autocorrelation(
                ref_month,
                lag,
                n_seasons,
                observations_by_season,
                stats_by_season,
            );
            matrix[i * dim + j] = rho;
            matrix[j * dim + i] = rho; // symmetric
        }
    }

    // Fill the classical RHS entries: rhs[i] = rho(season, i+1).
    for (i, rhs_entry) in rhs.iter_mut().enumerate().take(order) {
        *rhs_entry = periodic_autocorrelation(
            season,
            i + 1,
            n_seasons,
            observations_by_season,
            stats_by_season,
        );
    }

    // Fill the right column and bottom row (annual extension).
    // Entry (i, order) = (order, i) = cross_correlation_z_a(prev_season, i, ...).
    for i in 0..order {
        let rho = cross_correlation_z_a(
            prev_season,
            i,
            n_seasons,
            observations_by_season,
            stats_by_season,
            z_year_starts,
            annual_observations_by_season,
            annual_stats_by_season,
            a_year_starts,
        );
        matrix[i * dim + order] = rho;
        matrix[order * dim + i] = rho; // symmetric
    }

    // Bottom-right diagonal entry for the annual component.
    matrix[order * dim + order] = 1.0;

    // Annual RHS entry: rho_{A,Z}^{prev_season}(-1).
    rhs[order] = cross_correlation_a_z_neg1(
        prev_season,
        n_seasons,
        observations_by_season,
        stats_by_season,
        z_year_starts,
        annual_observations_by_season,
        annual_stats_by_season,
        a_year_starts,
    );

    (matrix, rhs)
}

// ---------------------------------------------------------------------------
// Small matrix solver
// ---------------------------------------------------------------------------

/// Solve a dense linear system `A * x = b` via Gaussian elimination with
/// partial pivoting.
///
/// Designed for small systems (n <= 10) arising from Yule-Walker equations
/// in PAR model fitting. For these sizes, the O(n^3) cost is negligible.
///
/// # Parameters
///
/// - `a` -- flat row-major matrix of dimension `n x n` (length `n * n`).
///   **Modified in place** during elimination.
/// - `b` -- right-hand side vector of length `n`. **Modified in place**.
/// - `n` -- system dimension.
///
/// # Returns
///
/// `Some(x)` where `x` is the solution vector of length `n`, or `None` if the
/// matrix is singular (pivot magnitude below `f64::EPSILON`).
pub fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n, "matrix must have n*n elements");
    debug_assert_eq!(b.len(), n, "rhs must have n elements");

    if n == 0 {
        return Some(Vec::new());
    }

    // Forward elimination with partial pivoting.
    for k in 0..n {
        // Find pivot: row with largest |a[row][k]| in rows k..n-1.
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for row in (k + 1)..n {
            let val = a[row * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Singularity check.
        if max_val < f64::EPSILON {
            return None;
        }

        // Swap rows k and max_row in both a and b.
        if max_row != k {
            for col in 0..n {
                a.swap(k * n + col, max_row * n + col);
            }
            b.swap(k, max_row);
        }

        // Eliminate below.
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for k in (0..n).rev() {
        let mut sum = b[k];
        for j in (k + 1)..n {
            sum -= a[k * n + j] * x[j];
        }
        x[k] = sum / a[k * n + k];
    }

    Some(x)
}

// Thread-local call counter used by
// `estimate_periodic_ar_coefficients_calls_build_once_per_order` to verify
// that the single-call behaviour is in effect (exactly one
// `build_periodic_yw_matrix` call per loop iteration, not two).
#[cfg(test)]
thread_local! {
    pub(super) static BUILD_PERIODIC_YW_MATRIX_CALL_COUNT: std::cell::RefCell<usize> =
        const { std::cell::RefCell::new(0) };
}
