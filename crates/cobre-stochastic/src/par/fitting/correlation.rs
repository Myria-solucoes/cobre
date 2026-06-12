//! Cross-entity residual correlation estimation for fitted PAR models.

use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use rayon::prelude::*;

use cobre_core::{
    EntityId,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry,
    },
    temporal::{SeasonMap, Stage},
};

use super::ar_coefficients::{ArCoefficientEstimate, SeasonLookups, build_season_lookups};
use super::seasonal_stats::{SeasonalStats, find_season_for_date};
use crate::StochasticError;

// ---------------------------------------------------------------------------
// Correlation estimation
// ---------------------------------------------------------------------------

/// Estimate the cross-entity residual correlation matrix from historical observations.
///
/// After a PAR(p) model is fitted (via [`estimate_seasonal_stats`] and
/// [`estimate_ar_coefficients`]), this function computes the standardized
/// innovation residuals for each entity at each time step and derives
/// the Pearson correlation between each pair of entities. The result is
/// a [`CorrelationModel`] with a single `"default"` profile containing
/// all entities, which the assembly pipeline can use as an automatic
/// fallback when no explicit correlation file is provided.
///
/// ## Residual formula
///
/// The standardized residual (innovation) at time step `t` for entity `i`
/// in season `m` is:
///
/// ```text
/// ε_t = z_t − Σ_{l=1}^{p} ψ*_{m,l} · z_{t−l}
/// ```
///
/// where `z_t = (a_t − μ_m) / s_m` is the standardized observation and
/// `ψ*_{m,l}` is the standardized AR coefficient for lag `l` in season `m`.
/// Only time steps where all `p` lagged standardized observations are
/// available contribute residuals.
///
/// ## Pearson correlation
///
/// For each pair `(i, j)` the function computes:
///
/// ```text
/// r_{ij} = cov(ε_i, ε_j) / (std(ε_i) · std(ε_j))
/// ```
///
/// using Bessel-corrected (N−1) estimators over the subset of time steps
/// where both entities have valid residuals. When fewer than 2 overlapping
/// steps exist, `r_{ij}` is set to 0.0.
///
/// ## Output structure
///
/// Returns a [`CorrelationModel`] with:
/// - `method: "spectral"`
/// - A single profile named `"default"` with a single group containing all
///   entities in canonical `hydro_ids` order and the estimated correlation matrix.
/// - An empty `schedule` (the single profile applies to all stages).
///
/// The function does **not** enforce positive-semidefiniteness. The downstream
/// spectral decomposition handles rank-deficient and non-PD matrices naturally.
///
/// # Parameters
///
/// - `observations` — flat slice of `(entity_id, date, value)` triples,
///   sorted by `(entity_id, date)`.
/// - `ar_estimates` — output of [`estimate_ar_coefficients`].
/// - `seasonal_stats` — output of [`estimate_seasonal_stats`].
/// - `stages` — all study stages with `season_id` assignments.
/// - `hydro_ids` — canonical sorted entity IDs; determines matrix row/column order.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when `seasonal_stats` is empty but
///   `hydro_ids` is non-empty (inconsistent inputs).
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::{
///     estimate_seasonal_stats, estimate_ar_coefficients, estimate_correlation,
/// };
///
/// fn stage(id: i32, y0: i32, m0: u32, y1: i32, m1: u32, season: usize) -> Stage {
///     Stage {
///         index: 0,
///         id,
///         start_date: NaiveDate::from_ymd_opt(y0, m0, 1).unwrap(),
///         end_date: NaiveDate::from_ymd_opt(y1, m1, 1).unwrap(),
///         season_id: Some(season),
///         blocks: vec![Block { index: 0, name: "S".to_string(), duration_hours: 744.0 }],
///         block_mode: BlockMode::Parallel,
///         state_config: StageStateConfig { storage: true, inflow_lags: false },
///         risk_config: StageRiskConfig::Expectation,
///         scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
///     }
/// }
///
/// // Build a single-season study over 5 years.
/// let stages_vec: Vec<Stage> = (2000..2005_i32)
///     .map(|y| stage(y - 1999, y, 1, y, 2, 0))
///     .collect();
/// let hydro_ids = vec![EntityId::from(1)];
/// let obs: Vec<(EntityId, NaiveDate, f64)> = (2000..2005_i32)
///     .map(|y| (EntityId::from(1), NaiveDate::from_ymd_opt(y, 1, 15).unwrap(), 100.0 + y as f64))
///     .collect();
/// let stats = estimate_seasonal_stats(&obs, &stages_vec, &hydro_ids).unwrap();
/// let estimates = estimate_ar_coefficients(&obs, &stats, &stages_vec, &hydro_ids, 0).unwrap();
/// let corr = estimate_correlation(&obs, &estimates, &stats, &stages_vec, &hydro_ids).unwrap();
/// assert!(corr.profiles.contains_key("default"));
/// assert_eq!(corr.profiles["default"].groups[0].matrix.len(), 1);
/// ```
pub fn estimate_correlation(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
) -> Result<CorrelationModel, StochasticError> {
    estimate_correlation_with_season_map(
        observations,
        ar_estimates,
        seasonal_stats,
        stages,
        hydro_ids,
        None,
    )
}

/// Minimum number of paired observations required per season for a per-season
/// correlation matrix. Seasons below this threshold fall back to the pooled
/// (all-season) matrix via the "default" profile.
const MIN_CORRELATION_PAIRS: usize = 30;

/// Estimate correlation with an optional [`SeasonMap`] fallback.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when seasonal stats are
/// empty but hydros are present, or when residual computation fails.
pub fn estimate_correlation_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    ar_estimates: &[ArCoefficientEstimate],
    seasonal_stats: &[SeasonalStats],
    stages: &[Stage],
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<CorrelationModel, StochasticError> {
    // Trivial case: no hydros.
    if hydro_ids.is_empty() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile { groups: Vec::new() },
        );
        return Ok(CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: Vec::new(),
        });
    }

    if seasonal_stats.is_empty() {
        return Err(StochasticError::InsufficientData {
            context: "seasonal_stats is empty but hydro_ids is non-empty; \
                      cannot estimate correlation without seasonal statistics"
                .to_string(),
        });
    }

    let lookups = build_season_lookups(observations, seasonal_stats, stages, season_map);

    let ar_lookup: HashMap<(EntityId, usize), &ArCoefficientEstimate> = ar_estimates
        .iter()
        .map(|e| ((e.hydro_id, e.season_id), e))
        .collect();

    let per_season_residuals = compute_hydro_residuals(&lookups, &ar_lookup, hydro_ids, season_map);

    // Warn about potentially degenerate hydros (informational only; spectral
    // decomposition handles near-zero eigenvalues naturally).
    warn_degenerate_hydros(&lookups, hydro_ids, &per_season_residuals);

    let pooled_residuals = flatten_residuals(&per_season_residuals);
    let pooled_matrix = compute_pearson_correlation_matrix(&pooled_residuals);
    let seasonal_matrices = compute_seasonal_matrices(&per_season_residuals, lookups.n_seasons);

    Ok(assemble_seasonal_correlation_model(
        hydro_ids,
        &pooled_matrix,
        &seasonal_matrices,
        stages,
        lookups.n_seasons,
    ))
}

/// Compute standardized AR innovation residuals for each hydro, partitioned by season.
///
/// For each hydro and each observation date where the AR model has sufficient
/// lag history, computes:
///   `epsilon_t = z_t - sum(psi_{m,l} * z_{t-l})` for l in 1..=p
///
/// Returns one `HashMap<season_id, HashMap<NaiveDate, f64>>` per hydro (indexed
/// by position in `hydro_ids`). Each residual is placed under its `season_id`
/// key instead of being pooled into a flat date-keyed map.
fn compute_hydro_residuals(
    lookups: &SeasonLookups<'_>,
    ar_lookup: &HashMap<(EntityId, usize), &ArCoefficientEstimate>,
    hydro_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Vec<HashMap<usize, HashMap<NaiveDate, f64>>> {
    // Each hydro's residual map is computed independently from immutable shared
    // inputs, so the outer per-hydro loop is embarrassingly parallel. Each task
    // owns a local map and `collect()` reassembles results in canonical
    // `hydro_ids` index order, making thread scheduling irrelevant to the output
    // ordering (deterministic). The inner per-observation arithmetic — including
    // the sequential `ar_sum` accumulation over lags — is reproduced verbatim so
    // every output element is bit-identical to a single-threaded pass.
    hydro_ids
        .par_iter()
        .map(|&hydro_id| {
            let mut residuals: HashMap<usize, HashMap<NaiveDate, f64>> = HashMap::new();

            let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
                return residuals;
            };
            let Some(date_index) = lookups.entity_date_index.get(&hydro_id) else {
                return residuals;
            };

            for &(date, value) in all_obs {
                let Some(season_id) = find_season_for_date(&lookups.stage_index, date)
                    .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
                else {
                    continue;
                };

                let Some(stats_m) = lookups.stats_lookup.get(&(hydro_id, season_id)) else {
                    continue;
                };

                let z_t = if stats_m.std == 0.0 {
                    0.0
                } else {
                    (value - stats_m.mean) / stats_m.std
                };

                let ar_est = ar_lookup.get(&(hydro_id, season_id));
                let ar_order = ar_est.map_or(0, |e| e.coefficients.len());
                let ar_coeffs = ar_est.map_or(&[] as &[f64], |e| &e.coefficients);

                let Some(&pos) = date_index.get(&date) else {
                    continue;
                };
                if pos < ar_order {
                    continue;
                }

                let mut ar_sum = 0.0_f64;
                let mut lag_ok = true;

                for lag in 1..=ar_order {
                    let n_s = lookups.n_seasons.max(1);
                    let lag_season = season_id.wrapping_add(n_s).wrapping_sub(lag % n_s) % n_s;

                    let Some(stats_lag) = lookups.stats_lookup.get(&(hydro_id, lag_season)) else {
                        lag_ok = false;
                        break;
                    };

                    let (_, lagged_value) = all_obs[pos - lag];
                    let z_lag = if stats_lag.std == 0.0 {
                        0.0
                    } else {
                        (lagged_value - stats_lag.mean) / stats_lag.std
                    };
                    ar_sum += ar_coeffs[lag - 1] * z_lag;
                }

                if !lag_ok {
                    continue;
                }

                let epsilon = z_t - ar_sum;
                residuals
                    .entry(season_id)
                    .or_default()
                    .insert(date, epsilon);
            }

            residuals
        })
        .collect()
}

/// Flatten per-season residuals into a single date-keyed map per hydro.
///
/// Dates are unique per hydro per season (each observation maps to exactly one
/// season), so no collisions occur when merging season maps.
fn flatten_residuals(
    per_season: &[HashMap<usize, HashMap<NaiveDate, f64>>],
) -> Vec<HashMap<NaiveDate, f64>> {
    per_season
        .iter()
        .map(|seasons| {
            let mut flat = HashMap::new();
            for date_map in seasons.values() {
                flat.extend(date_map.iter().map(|(&d, &v)| (d, v)));
            }
            flat
        })
        .collect()
}

/// Emit diagnostic warnings for hydros that exhibit degenerate statistical
/// properties. These are informational only — the spectral decomposition handles
/// near-zero eigenvalues naturally and no hydros are excluded.
fn warn_degenerate_hydros(
    lookups: &SeasonLookups<'_>,
    hydro_ids: &[EntityId],
    per_season_residuals: &[HashMap<usize, HashMap<NaiveDate, f64>>],
) {
    for (hidx, &hydro_id) in hydro_ids.iter().enumerate() {
        let Some(all_obs) = lookups.entity_obs.get(&hydro_id) else {
            continue;
        };

        // Partition raw observations by season.
        let mut obs_by_season: HashMap<usize, Vec<f64>> = HashMap::new();
        for &(date, value) in all_obs {
            if let Some(season_id) = find_season_for_date(&lookups.stage_index, date) {
                obs_by_season.entry(season_id).or_default().push(value);
            }
        }

        for season_id in 0..lookups.n_seasons {
            let Some(vals) = obs_by_season.get(&season_id) else {
                continue;
            };
            if vals.len() < 2 {
                continue;
            }

            // Warn: >50% negative observations.
            #[allow(clippy::cast_precision_loss)]
            let neg_frac = vals.iter().filter(|&&v| v < 0.0).count() as f64 / vals.len() as f64;
            if neg_frac > 0.5 {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_id,
                    negative_fraction = neg_frac,
                    "hydro has majority negative observations in season \
                     (included in correlation; spectral decomposition handles this)"
                );
            }

            // Warn: constant series (all values identical).
            let first = vals[0];
            if vals.iter().all(|&v| (v - first).abs() < f64::EPSILON) {
                tracing::warn!(
                    hydro_id = hydro_id.0,
                    season = season_id,
                    value = first,
                    "hydro has constant series in season \
                     (included in correlation; near-zero eigenvalue expected)"
                );
                // Near-zero residual variance is implied by a constant series;
                // skip the residual check for this season.
                continue;
            }

            // Warn: near-zero residual variance.
            if let Some(residuals) = per_season_residuals
                .get(hidx)
                .and_then(|m| m.get(&season_id))
                .filter(|r| r.len() >= 2)
            {
                let r_vals: Vec<f64> = residuals.values().copied().collect();
                #[allow(clippy::cast_precision_loss)]
                let r_mean = r_vals.iter().sum::<f64>() / r_vals.len() as f64;
                #[allow(clippy::cast_precision_loss)]
                let r_std = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f64>()
                    / (r_vals.len() - 1) as f64)
                    .sqrt();
                if r_std < 1e-8 {
                    tracing::warn!(
                        hydro_id = hydro_id.0,
                        season = season_id,
                        residual_std = r_std,
                        "hydro has near-zero residual variance in season \
                         (included in correlation; near-zero eigenvalue expected)"
                    );
                }
            }
        }
    }
}

/// Compute per-season Pearson correlation matrices.
///
/// For each season, extracts the residuals belonging to that season from each
/// hydro and computes the Pearson correlation matrix. Seasons where the minimum
/// number of paired observations across all hydro pairs is below
/// [`MIN_CORRELATION_PAIRS`] are excluded from the result and fall back to the
/// pooled (all-season) matrix via the `"default"` profile.
///
/// All hydros participate regardless of degenerate status. Rank-deficient
/// matrices (more hydros than observations) are acceptable because the downstream
/// spectral decomposition handles them naturally.
fn compute_seasonal_matrices(
    per_season_residuals: &[HashMap<usize, HashMap<NaiveDate, f64>>],
    n_seasons: usize,
) -> HashMap<usize, Vec<f64>> {
    let mut result = HashMap::new();
    let n_hydros = per_season_residuals.len();

    for season_id in 0..n_seasons {
        // Extract per-hydro residuals for this season (all hydros, no exclusions).
        let season_residuals: Vec<HashMap<NaiveDate, f64>> = per_season_residuals
            .iter()
            .map(|hydro_seasons| hydro_seasons.get(&season_id).cloned().unwrap_or_default())
            .collect();

        // Determine minimum paired observation count across all hydro pairs.
        let min_pairs = if n_hydros <= 1 {
            season_residuals.first().map_or(0, HashMap::len)
        } else {
            (0..n_hydros)
                .flat_map(|i| (i + 1..n_hydros).map(move |j| (i, j)))
                .map(|(i, j)| {
                    season_residuals[i]
                        .keys()
                        .filter(|d| season_residuals[j].contains_key(*d))
                        .count()
                })
                .min()
                .unwrap_or(0)
        };

        if min_pairs < MIN_CORRELATION_PAIRS {
            continue;
        }

        result.insert(
            season_id,
            compute_pearson_correlation_matrix(&season_residuals),
        );
    }

    result
}

/// Compute the `n_hydros` x `n_hydros` Pearson correlation matrix from residuals.
///
/// For each pair `(i, j)`, collects time steps where both have residuals,
/// then applies the standard Pearson formula with Bessel correction (N-1).
///
/// Pairs are sorted by their join key (`NaiveDate`) before accumulation. The
/// date is the unique observation identifier shared by both residual series, so
/// sorting on it yields an iteration order that is independent of (a) the
/// nondeterministic `HashMap` traversal order and (b) which of the two series is
/// the row (`i`) versus the column (`j`). The latter is what makes the result
/// invariant to the declaration order of the input series: reversing the slice
/// swaps the row/column roles of every pair but leaves the date-sorted summation
/// order — and therefore every floating-point partial sum — bit-identical.
pub(super) fn compute_pearson_correlation_matrix(
    hydro_residuals: &[HashMap<NaiveDate, f64>],
) -> Vec<f64> {
    let n = hydro_residuals.len();
    // Flat row-major N×N matrix: entry (i, j) is at index i * n + j.
    let mut matrix = vec![0.0_f64; n * n];

    for i in 0..n {
        matrix[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let r_i = &hydro_residuals[i];
            let r_j = &hydro_residuals[j];

            let mut pairs: Vec<(NaiveDate, f64, f64)> = r_i
                .iter()
                .filter_map(|(&date, &ei)| r_j.get(&date).map(|&ej| (date, ei, ej)))
                .collect();

            if pairs.len() < 2 {
                matrix[i * n + j] = 0.0;
                matrix[j * n + i] = 0.0;
                continue;
            }

            // Sort by the join key (date) so the accumulation order depends only
            // on the shared observation dates — never on HashMap traversal order
            // or on which series is i vs j. Dates are unique within a series, so
            // this is a total order with no ties.
            pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            #[allow(clippy::cast_precision_loss)]
            let np = pairs.len() as f64;
            let mean_i = pairs.iter().map(|(_, ei, _)| ei).sum::<f64>() / np;
            let mean_j = pairs.iter().map(|(_, _, ej)| ej).sum::<f64>() / np;

            let mut cov = 0.0_f64;
            let mut var_i = 0.0_f64;
            let mut var_j = 0.0_f64;
            for (_, ei, ej) in &pairs {
                let di = ei - mean_i;
                let dj = ej - mean_j;
                cov += di * dj;
                var_i += di * di;
                var_j += dj * dj;
            }

            let denom = np - 1.0;
            let std_i = (var_i / denom).sqrt();
            let std_j = (var_j / denom).sqrt();

            let rho = if std_i < f64::EPSILON || std_j < f64::EPSILON {
                0.0
            } else {
                (cov / denom) / (std_i * std_j)
            };

            let rho = rho.clamp(-1.0, 1.0);
            matrix[i * n + j] = rho;
            matrix[j * n + i] = rho;
        }
    }

    matrix
}

/// Assemble a multi-profile [`CorrelationModel`] from hydro IDs, a pooled
/// matrix, per-season matrices, the stage list, and the total season count.
///
/// The pooled matrix is always inserted as the `"default"` profile.
/// When `n_seasons <= 1`, only the default profile is produced (backward
/// compatibility with the single-season path).
/// For the multi-season case, each season that passed the minimum-sample check
/// gets a `"season_XX"` profile (zero-padded to the width of the largest season
/// index), and the schedule maps each stage to its season's profile name.
fn assemble_seasonal_correlation_model(
    hydro_ids: &[EntityId],
    pooled_matrix: &[f64],
    seasonal_matrices: &HashMap<usize, Vec<f64>>,
    stages: &[Stage],
    n_seasons: usize,
) -> CorrelationModel {
    let n = hydro_ids.len();

    let entities: Vec<CorrelationEntity> = hydro_ids
        .iter()
        .map(|&id| CorrelationEntity {
            entity_type: "inflow".to_string(),
            id,
        })
        .collect();

    let mut profiles = BTreeMap::new();

    // Convert a flat row-major N×N Vec<f64> to the AoS Vec<Vec<f64>> layout
    // required by CorrelationGroup.matrix (defined in cobre-core).
    let flat_to_aos = |flat: &[f64]| -> Vec<Vec<f64>> {
        (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
    };

    // Always include the pooled matrix as "default".
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "default".to_string(),
                entities: entities.clone(),
                matrix: flat_to_aos(pooled_matrix),
            }],
        },
    );

    // Single-season: return early with no per-season profiles and empty schedule.
    if n_seasons <= 1 {
        return CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: Vec::new(),
        };
    }

    // Multi-season: add per-season profiles.
    let width = format!("{}", n_seasons.saturating_sub(1)).len();
    for (&season_id, matrix) in seasonal_matrices {
        let name = format!("season_{season_id:0>width$}");
        profiles.insert(
            name,
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "default".to_string(),
                    entities: entities.clone(),
                    matrix: flat_to_aos(matrix),
                }],
            },
        );
    }

    // Build schedule: map each stage to its season profile name, omitting
    // stages whose season has no per-season profile (they fall back to "default").
    let mut schedule: Vec<CorrelationScheduleEntry> = stages
        .iter()
        .filter_map(|stage| {
            let season_id = stage.season_id?;
            if seasonal_matrices.contains_key(&season_id) {
                Some(CorrelationScheduleEntry {
                    stage_id: stage.id,
                    profile_name: format!("season_{season_id:0>width$}"),
                })
            } else {
                None
            }
        })
        .collect();
    schedule.sort_by_key(|e| e.stage_id);

    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule,
    }
}
