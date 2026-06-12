//! Numeric PAR(p) parameter-fitting core: order selection, contribution-based
//! reduction, and report assembly.
//!
//! This module sits one level above the per-season `fitting/` primitives it
//! drives ([`periodic_pacf`], [`conditional_facp_partitioned`],
//! [`estimate_periodic_ar_coefficients`], …). It owns the order-selection and
//! iterative reduction chain that turns raw `(entity, date, value)` observations
//! plus seasonal stats into a `Vec<ArCoefficientEstimate>` together with an
//! [`EstimationReport`] of the reductions applied.
//!
//! The two public entry points are [`estimate_ar_coefficients_with_selection`]
//! (the classical and annual dispatch) and [`build_estimation_report`]. The core
//! fails only with [`StochasticError`]; it never touches case-loading or row
//! types — that orchestration lives in the I/O shell that calls into this module.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{Datelike, NaiveDate};
use cobre_core::EntityId;
use cobre_core::scenario::AnnualComponent;

use crate::StochasticError;
use crate::par::contribution::{
    check_negative_contributions, compute_contributions, find_max_valid_order, has_negative_phi1,
};
use crate::par::fitting::{
    AnnualSeasonalStats, ArCoefficientEstimate, SeasonalStats, conditional_facp_partitioned,
    estimate_annual_seasonal_stats, estimate_periodic_ar_annual_coefficients,
    estimate_periodic_ar_coefficients, find_season_for_date, periodic_pacf, select_order_pacf,
    select_order_pacf_annual,
};

/// Reason for an AR order reduction.
///
/// Distinguishes the three mechanisms that can reduce a season's AR order
/// during the estimation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionReason {
    /// Coefficient exceeds the magnitude-bound safety threshold.
    MagnitudeBound,
    /// First AR coefficient (`phi_1`) is negative, contradicting
    /// hydrological persistence.
    Phi1Negative,
    /// Contribution analysis detected negative entries at one or more lags.
    NegativeContribution,
}

impl ReductionReason {
    /// Convert to a stable string representation for diagnostic output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MagnitudeBound => "magnitude_bound",
            Self::Phi1Negative => "phi1_negative",
            Self::NegativeContribution => "negative_contribution",
        }
    }
}

/// A single contribution-based order reduction event.
///
/// Records that a season's AR order was reduced because the contribution
/// analysis detected negative entries, indicating potential model instability.
#[derive(Debug, Clone)]
pub struct ContributionReduction {
    /// Season where the reduction occurred.
    pub season_id: usize,
    /// Order before reduction (from AIC or previous iteration).
    pub original_order: usize,
    /// Order after reduction (the maximum valid order from contributions).
    pub reduced_order: usize,
    /// Contribution values at the original order that triggered the reduction.
    pub contributions: Vec<f64>,
    /// The mechanism that triggered this reduction.
    pub reason: ReductionReason,
}

/// Per-hydro AIC diagnostic data captured during AIC-based AR order selection.
///
/// Holds the selected order, fitted AR coefficients, and any contribution-based
/// order reductions for each season at the selected order.
#[derive(Debug, Clone)]
pub struct HydroEstimationEntry {
    /// The selected AR order for this hydro plant (maximum across all seasons).
    ///
    /// This is the maximum of the per-season selected orders, which determines
    /// the coefficient vector length in the output.
    pub selected_order: u32,
    /// Fitted AR lag coefficients, one inner vector per season sorted by `season_id` ascending.
    ///
    /// Each inner vector holds the coefficients at the selected order for
    /// that season. Seasons where estimation was skipped (zero std, insufficient
    /// observations) have an empty coefficient vector.
    pub coefficients: Vec<Vec<f64>>,
    /// Records of contribution-based order reductions applied during fitting.
    ///
    /// Each entry documents a season where the initial order (from PACF or fixed
    /// selection) was reduced due to negative contributions. Empty when no
    /// reductions were needed.
    pub contribution_reductions: Vec<ContributionReduction>,
}

/// Computation-side summary of the AR estimation pipeline.
///
/// Contains one [`HydroEstimationEntry`] per hydro plant that was fitted,
/// keyed by [`EntityId`] for canonical deterministic ordering.
#[must_use]
#[derive(Debug, Clone)]
pub struct EstimationReport {
    /// Per-hydro diagnostic entries, keyed by entity ID.
    pub entries: BTreeMap<EntityId, HydroEstimationEntry>,
    /// The order selection method used (e.g., `"AIC"`, `"PACF"`, `"fixed"`).
    pub method: String,
    /// Hydro IDs that have user-provided stats but no estimated AR
    /// coefficients, resulting in white-noise fallback (empty AR, ratio=1.0).
    /// Only populated by the partial-estimation path; empty for other paths.
    pub white_noise_fallbacks: Vec<EntityId>,
    /// Warnings for hydros where consecutive-season std ratios diverge
    /// significantly between user-provided and history-estimated profiles.
    /// Only populated by the partial-estimation path; empty for other paths.
    pub std_ratio_warnings: Vec<StdRatioDivergence>,
}

/// Advisory diagnostic for a (hydro, season pair) where the cross-season
/// standard deviation ratio diverges significantly between the user-provided
/// profile and the history-estimated profile.
///
/// Produced by the partial-estimation path (P9 diagnostic) when
/// `max(user_ratio / est_ratio, est_ratio / user_ratio) > 2.0` for any
/// consecutive season pair `(season_a, season_b)`.
#[derive(Debug, Clone)]
pub struct StdRatioDivergence {
    /// The hydro plant for which the divergence was detected.
    pub hydro_id: EntityId,
    /// Index of the first season in the consecutive pair.
    pub season_a: usize,
    /// Index of the second season in the consecutive pair (wraps around).
    pub season_b: usize,
    /// `std[season_a] / std[season_b]` from the user-provided profile.
    pub user_ratio: f64,
    /// `std[season_a] / std[season_b]` from the history-estimated profile.
    pub estimated_ratio: f64,
    /// `max(user_ratio / estimated_ratio, estimated_ratio / user_ratio)`.
    pub divergence: f64,
}

/// Result of validating an AR order via contribution analysis.
///
/// Captures whether the current order is stable (all contributions non-negative),
/// the maximum valid order if not, and the computed contribution values for
/// diagnostic reporting.
#[derive(Debug, Clone)]
pub struct ContributionValidationResult {
    /// Whether the current order passed contribution validation.
    pub valid: bool,
    /// Maximum valid order (same as `current_order` if valid, less otherwise).
    pub max_valid_order: usize,
    /// Computed contribution values for the current order.
    pub contributions: Vec<f64>,
}

/// Validate an AR order for a single (entity, season) pair via contribution analysis.
///
/// Computes the recursively-composed contributions for the given season at the
/// current order, then checks for negative entries. Returns a result indicating
/// whether the order is stable and, if not, the maximum valid order.
///
/// When `current_order == 0`, returns immediately with `valid: true` and no
/// contributions (an order-0 model has no autoregressive dependence to validate).
fn validate_order_contributions(
    season_id: usize,
    n_seasons: usize,
    current_order: usize,
    all_season_coefficients: &[Vec<f64>],
    std_by_season: &[f64],
) -> ContributionValidationResult {
    if current_order == 0 {
        return ContributionValidationResult {
            valid: true,
            max_valid_order: 0,
            contributions: Vec::new(),
        };
    }

    let coeff_refs: Vec<&[f64]> = all_season_coefficients.iter().map(Vec::as_slice).collect();
    let contributions = compute_contributions(
        season_id,
        n_seasons,
        current_order,
        &coeff_refs,
        std_by_season,
    );

    let valid = !check_negative_contributions(&contributions);
    let max_valid_order = if valid {
        current_order
    } else {
        find_max_valid_order(&contributions)
    };

    ContributionValidationResult {
        valid,
        max_valid_order,
        contributions,
    }
}

/// Configuration parameters for AR coefficient estimation.
pub struct ArEstimationConfig<'a> {
    /// Maximum AR order considered before order selection.
    pub max_order: usize,
    /// Optional per-coefficient magnitude safety bound.
    pub max_coeff_magnitude: Option<f64>,
    /// Season map for calendar-based date-to-season fallback.
    pub season_map: Option<&'a cobre_core::temporal::SeasonMap>,
    /// When `true`, the PAR-A path (conditional FACP + extended YW) is used.
    /// When `false` (the default), the classical PACF path is used.
    pub use_annual_component: bool,
}

/// Estimate AR coefficients, dispatching to the classical or PAR-A path.
///
/// When `cfg.use_annual_component` is `false` (the default), delegates to
/// [`estimate_ar_with_pacf`] (classical periodic Yule-Walker + PACF).
/// When `cfg.use_annual_component` is `true`, delegates to
/// [`estimate_ar_with_pacf_annual`] (extended YW with rolling 12-month average).
///
/// # Errors
///
/// Propagates [`StochasticError`] from the underlying fitting primitives.
pub fn estimate_ar_coefficients_with_selection(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    cfg: &ArEstimationConfig<'_>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    if cfg.use_annual_component {
        estimate_ar_with_pacf_annual(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            cfg.max_order,
            cfg.season_map,
            cfg.max_coeff_magnitude,
        )
    } else {
        estimate_ar_with_pacf(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            cfg.max_order,
            cfg.season_map,
            cfg.max_coeff_magnitude,
        )
    }
}

/// PACF-based AR order selection using periodic Yule-Walker method.
///
/// Selects the AR order using the periodic partial autocorrelation function
/// (PACF) significance test, then estimates coefficients via the periodic
/// Yule-Walker matrix solve. The PACF threshold uses a 95% confidence
/// interval (`z_alpha = 1.96`).
///
/// The periodic approach correctly accounts for the non-Toeplitz covariance
/// structure of periodic autoregressive processes.
fn estimate_ar_with_pacf(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&cobre_core::temporal::SeasonMap>,
    max_coeff_magnitude: Option<f64>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    if max_order == 0 {
        // Order-0: produce white-noise estimates for all (entity, season) pairs.
        let estimates = crate::par::fitting::estimate_ar_coefficients_with_season_map(
            observations,
            seasonal_stats,
            stages,
            hydro_ids,
            0,
            season_map,
        )?;
        let report = EstimationReport {
            entries: BTreeMap::new(),
            method: "PACF".to_string(),
            white_noise_fallbacks: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };
        return Ok((estimates, report));
    }

    let (stage_index, stats_map, n_seasons) = build_pacf_stage_lookups(stages, seasonal_stats);
    let group_obs = group_observations_by_season(observations, hydro_ids, &stage_index, season_map);

    // 95% confidence z-score for the PACF significance threshold.
    let z_alpha = 1.96_f64;

    let mut estimates = estimate_all_hydro_ar_coefficients(
        hydro_ids, &group_obs, &stats_map, n_seasons, max_order, z_alpha,
    );

    let reductions = iterative_pacf_reduction(
        &mut estimates,
        n_seasons,
        hydro_ids,
        &group_obs,
        &stats_map,
        &PacfReductionParams {
            initial_max_order: max_order,
            z_alpha,
            max_coeff_magnitude,
        },
    );

    let report = build_estimation_report(&estimates, n_seasons, &reductions, "PACF");
    Ok((estimates, report))
}

/// PAR-A path: extended periodic Yule-Walker with rolling 12-month annual component.
///
/// Mirrors [`estimate_ar_with_pacf`] with three key differences:
///
/// 1. Calls [`estimate_annual_seasonal_stats`] to obtain per-(hydro, season)
///    `(μ^A_m, σ^A_m)`.
/// 2. Builds rolling-window `A_t` groupings (same windows Part A consumed),
///    organised by target season.
/// 3. Per (hydro, season): calls [`conditional_facp_partitioned`] →
///    [`select_order_pacf_annual`] → [`estimate_periodic_ar_annual_coefficients`].
///
/// Every returned [`ArCoefficientEstimate`] has `annual: Some(AnnualComponent { .. })`
/// when the `(hydro, season)` pair has at least one rolling-window `A_t` observation;
/// seasons with no rolling-window data fall through to the classical PAR(p) path with
/// `annual: None`.
/// The estimation report uses `method = "PACF_ANNUAL"`.
///
/// # Errors
///
/// Propagates `StochasticError::InsufficientData` from
/// [`estimate_annual_seasonal_stats`] when any hydro has fewer than 13
/// chronological observations (no rolling window can be formed).
// Rationale: the function encodes a single cohesive PACF estimation pipeline —
// seasonal stats, stage indexing, Z-score grouping, order reduction, and report
// assembly — whose six numbered phases share intermediate look-up tables; splitting
// into sub-functions would require threading those tables as additional arguments
// and would obscure the sequential data-flow contract of the pipeline.
#[allow(clippy::too_many_lines)]
fn estimate_ar_with_pacf_annual(
    observations: &[(EntityId, NaiveDate, f64)],
    seasonal_stats: &[SeasonalStats],
    stages: &[cobre_core::temporal::Stage],
    hydro_ids: &[EntityId],
    max_order: usize,
    season_map: Option<&cobre_core::temporal::SeasonMap>,
    max_coeff_magnitude: Option<f64>,
) -> Result<(Vec<ArCoefficientEstimate>, EstimationReport), StochasticError> {
    // ── 1. Compute (μ^A_m, σ^A_m) per (hydro, season). ─────────────────────
    let annual_stats: Vec<AnnualSeasonalStats> =
        estimate_annual_seasonal_stats(observations, stages, hydro_ids, season_map)?;

    // Build a fast lookup: (hydro_id, season_id) → &AnnualSeasonalStats.
    let annual_stats_map: HashMap<(EntityId, usize), &AnnualSeasonalStats> = annual_stats
        .iter()
        .map(|s| ((s.hydro_id, s.season_id), s))
        .collect();

    // ── 2. Build stage index for date-to-season mapping. ─────────────────────
    let (stage_index, stats_map, n_seasons) = build_pacf_stage_lookups(stages, seasonal_stats);

    // ── 3. Group Z observations by (hydro, season) + per-bucket year start. ──
    let group_obs = group_observations_by_season(observations, hydro_ids, &stage_index, season_map);
    let group_z_year_starts: HashMap<(EntityId, usize), i32> = {
        let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
        let mut starts: HashMap<(EntityId, usize), i32> = HashMap::new();
        for &(entity_id, date, _value) in observations {
            if !entity_set.contains(&entity_id) {
                continue;
            }
            let Some(season_id) = find_season_for_date(&stage_index, date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            else {
                continue;
            };
            let y = date.year();
            starts
                .entry((entity_id, season_id))
                .and_modify(|cur| {
                    if y < *cur {
                        *cur = y;
                    }
                })
                .or_insert(y);
        }
        starts
    };

    // ── 4. Build rolling-window A_t groups by (hydro, season) + year start. ──
    //
    // Reproduce the same chronological grouping as `estimate_annual_seasonal_stats`
    // so that `annual_observations_by_season[s]` aligns with `obs_by_season[s]`.
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();

    // Sort observations per entity chronologically.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if entity_set.contains(&entity_id) {
            entity_obs.entry(entity_id).or_default().push((date, value));
        }
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    // For each entity, build rolling A_t values grouped by target season.
    //
    // Indexing convention (must match `estimate_annual_seasonal_stats`):
    // A_{t-1} = mean(z[t-12..t-1]) is stored under the season of its own
    // PDF time-index (t-1), i.e., the season of `group[i + 11]` when t = i + 12.
    // YW callers retrieve it via `prev_season = (m - 1) mod n_seasons`.
    //
    // The year of target_date is the PDF year of A_{t-1} for that bucket.
    // Tracking the minimum across all entries gives the bucket's first PDF
    // year — needed by `cross_correlation_z_a` to align A and Z by absolute
    // year rather than by bucket index.
    let mut annual_group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    let mut annual_group_year_starts: HashMap<(EntityId, usize), i32> = HashMap::new();
    for &entity_id in hydro_ids {
        let Some(group) = entity_obs.get(&entity_id) else {
            continue;
        };
        for i in 0..group.len().saturating_sub(12) {
            let target_date = group[i + 11].0;
            let Some(season_id) = find_season_for_date(&stage_index, target_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(target_date)))
            else {
                continue;
            };
            let mean_a: f64 = group[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
            annual_group_obs
                .entry((entity_id, season_id))
                .or_default()
                .push(mean_a);
            let y = target_date.year();
            annual_group_year_starts
                .entry((entity_id, season_id))
                .and_modify(|cur| {
                    if y < *cur {
                        *cur = y;
                    }
                })
                .or_insert(y);
        }
    }

    // ── 5. Per (hydro, season): conditional FACP → order → extended YW. ─────
    let z_alpha = 1.96_f64;
    let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();

    for &hydro_id in hydro_ids {
        // Collect Z and A observations + stats indexed by season.
        let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        let mut annual_obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
        let mut annual_stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
        let mut z_year_starts: Vec<i32> = vec![0; n_seasons];
        let mut a_year_starts: Vec<i32> = vec![0; n_seasons];

        for season in 0..n_seasons {
            if let Some(obs) = group_obs.get(&(hydro_id, season)) {
                obs_by_season[season].clone_from(obs);
            }
            if let Some(ann_obs) = annual_group_obs.get(&(hydro_id, season)) {
                annual_obs_by_season[season].clone_from(ann_obs);
            }
            if let Some(stats) = stats_map.get(&(hydro_id, season)) {
                stats_by_season[season] = (stats.mean, stats.std);
            }
            if let Some(ann_stats) = annual_stats_map.get(&(hydro_id, season)) {
                annual_stats_by_season[season] = (ann_stats.mean_m3s, ann_stats.std_m3s);
            }
            if let Some(&y) = group_z_year_starts.get(&(hydro_id, season)) {
                z_year_starts[season] = y;
            }
            if let Some(&y) = annual_group_year_starts.get(&(hydro_id, season)) {
                a_year_starts[season] = y;
            }
        }

        let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
        let annual_obs_refs: Vec<&[f64]> = annual_obs_by_season.iter().map(Vec::as_slice).collect();

        for season in 0..n_seasons {
            // The Yule-Walker equation for current month `season` couples
            // Z_t (at season `season`) with A_{t-1}, whose PDF time-index is
            // at the previous season. Annual stats and observations for that
            // A are stored under `prev_season` (see indexing convention in
            // `estimate_annual_seasonal_stats`).
            let prev_season = (season + n_seasons - 1) % n_seasons;
            let n_obs = obs_by_season[season].len();
            let n_ann_obs = annual_obs_by_season[prev_season].len();
            let stats_s = stats_by_season[season];
            let annual_stats_s = annual_stats_by_season[prev_season];

            // White-noise fallback: zero std, too few observations, or no annual obs.
            if stats_s.1 == 0.0 || n_obs < 2 || n_ann_obs == 0 || annual_stats_s.1 == 0.0 {
                estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id: season,
                    coefficients: Vec::new(),
                    residual_std_ratio: 1.0,
                    annual: annual_stats_map.get(&(hydro_id, prev_season)).map(|s| {
                        AnnualComponent {
                            coefficient: 0.0,
                            mean_m3s: s.mean_m3s,
                            std_m3s: s.std_m3s,
                        }
                    }),
                });
                continue;
            }

            // Conditional FACP → order selection → extended YW solve.
            let facp_values = conditional_facp_partitioned(
                season,
                max_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &z_year_starts,
                &annual_obs_refs,
                &annual_stats_by_season,
                &a_year_starts,
            );
            let pacf_result = select_order_pacf_annual(&facp_values, n_obs, z_alpha);
            let yw_result = estimate_periodic_ar_annual_coefficients(
                season,
                pacf_result.selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &z_year_starts,
                &annual_obs_refs,
                &annual_stats_by_season,
                &a_year_starts,
            );

            // Look up annual stats for the AnnualComponent triple. The YW
            // solver matches Z_t (season `season`) with A_{t-1} (PDF time at
            // `prev_season`); precompute applies the standardised ψ via
            // `psi_hat = ψ · σ_m / σ_a`, where σ_a must be the std of A_{t-1}
            // — i.e., the entry stored at `prev_season`.
            let (ann_mean, ann_std) = annual_stats_by_season[prev_season];
            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id: season,
                coefficients: yw_result.coefficients,
                residual_std_ratio: yw_result.residual_std_ratio,
                annual: Some(AnnualComponent {
                    coefficient: yw_result.annual_coefficient,
                    mean_m3s: ann_mean,
                    std_m3s: ann_std,
                }),
            });
        }
    }

    // Magnitude, φ_1, and iterative contribution pre-passes. Mirrors
    // `iterative_pacf_reduction` for the PAR-A path. The contribution
    // recursion runs on the φ vector only; ψ is preserved through reductions
    // and refreshed via re-solves of the extended Yule-Walker system at the
    // new ceiling.
    let reductions = apply_annual_prepass_reductions(
        &mut estimates,
        n_seasons,
        hydro_ids,
        &group_obs,
        &group_z_year_starts,
        &annual_group_obs,
        &annual_group_year_starts,
        &stats_map,
        &annual_stats_map,
        max_order,
        z_alpha,
        max_coeff_magnitude,
    );

    let report = build_estimation_report(&estimates, n_seasons, &reductions, "PACF_ANNUAL");
    Ok((estimates, report))
}

/// Apply magnitude-bound, `phi_1`, and contribution pre-passes for the PAR-A path.
///
/// Mirrors [`iterative_pacf_reduction`] for the PAR-A flow:
///
/// 1. Per-coefficient magnitude bound (drops the season's AR coefficients when
///    any `|φ| > threshold`; ψ is preserved).
/// 2. `φ_1 ≥ 0` guard (drops AR coefficients when `φ_1 < 0`; ψ preserved).
/// 3. Iterative contribution-based reduction via [`reduce_entity_orders_annual`].
///
/// **Contribution check scope.** The order `pm` refers to the
/// autoregressive components alone (the φ vector); the annual term ψ is a
/// separate parameter that is preserved across AR-order reductions. The
/// contribution recursion here operates on the φ coefficients (length p),
/// exactly as in the classical PAR(p) path. When a season's contributions
/// go negative, the AR ceiling is reduced and the extended Yule-Walker
/// system is re-solved at the new order — both φ and ψ are updated, but
/// the AR portion is what shrinks.
///
/// Returns a map of reductions for report building.
// Rationale: the function threads four independent paired look-up tables (regular and annual
// variants of observations and year-start maps) plus three independent stat maps and two
// scalar controls; bundling them into a struct would just displace the arity to the struct
// literal at each of the two call sites with no clarity gain.
#[allow(clippy::too_many_arguments)]
fn apply_annual_prepass_reductions(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_z_year_starts: &HashMap<(EntityId, usize), i32>,
    annual_group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    annual_group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    annual_stats_map: &HashMap<(EntityId, usize), &AnnualSeasonalStats>,
    initial_max_order: usize,
    z_alpha: f64,
    max_coeff_magnitude: Option<f64>,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    if let Some(threshold) = max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
                est.residual_std_ratio = 1.0;
            }
        }
    }

    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
            est.residual_std_ratio = 1.0;
        }
    }

    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for &hydro_id in hydro_ids {
        let Some(indices) = hydro_indices.get(&hydro_id) else {
            continue;
        };
        reduce_entity_orders_annual(
            estimates,
            n_seasons,
            hydro_id,
            indices,
            group_obs,
            group_z_year_starts,
            annual_group_obs,
            annual_group_year_starts,
            stats_map,
            annual_stats_map,
            initial_max_order,
            z_alpha,
            &mut all_reductions,
        );
    }

    all_reductions
}

/// Detect the seasons whose recursively-composed AR contributions have turned
/// negative at their current order, recording a `NegativeContribution`
/// reduction for each and returning the failing season ids.
///
/// Rationale: this is the single genuinely-shared sub-block of the two
/// iterative reduction loops ([`reduce_entity_orders`] and
/// [`reduce_entity_orders_annual`]). Both call it at the top of each loop
/// iteration with the same locals — the regular path and the annual path build
/// `all_coeffs`/`std_by_season`/`frozen` differently beforehand, but the
/// detection logic itself is byte-identical, so it is owned here once. The
/// per-season re-solve bodies that follow legitimately diverge (regular vs
/// annual Yule-Walker primitives) and stay in their respective callers.
fn detect_failing_seasons(
    estimates: &[ArCoefficientEstimate],
    indices: &[usize],
    frozen: &[bool],
    n_seasons: usize,
    all_coeffs: &[Vec<f64>],
    std_by_season: &[f64],
    hydro_id: EntityId,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) -> Vec<usize> {
    let mut failing_seasons: Vec<usize> = Vec::new();
    for &idx in indices {
        let season_id = estimates[idx].season_id;
        if frozen[season_id] || estimates[idx].coefficients.is_empty() {
            continue;
        }
        let current_order = estimates[idx].coefficients.len();
        let result = validate_order_contributions(
            season_id,
            n_seasons,
            current_order,
            all_coeffs,
            std_by_season,
        );
        if !result.valid {
            all_reductions
                .entry(hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id,
                    original_order: estimates[idx].coefficients.len(),
                    reduced_order: result.max_valid_order,
                    contributions: result.contributions,
                    reason: ReductionReason::NegativeContribution,
                });
            failing_seasons.push(season_id);
        }
    }
    failing_seasons
}

/// Run the iterative contribution-based order reduction for one entity in the
/// PAR-A path.
///
/// Mirrors [`reduce_entity_orders`] for the classical path: maintains
/// per-season `max_orders` ceilings, checks the recursively-composed AR
/// contributions (φ-only, length p) at each season, and reduces the ceiling
/// by 1 whenever any contribution turns negative. After each reduction, the
/// extended Yule-Walker system is re-solved (`conditional_facp_partitioned` →
/// `select_order_pacf_annual` → `estimate_periodic_ar_annual_coefficients`)
/// at the new ceiling so both φ and ψ are refreshed; the recorded
/// `AnnualComponent` triple is updated to match.
///
/// When the ceiling reaches 0 the AR coefficients are dropped (ψ retained
/// via a final order-0 YW solve so the constant term remains consistent
/// with the per-season annual stats).
// Rationale: the arguments are four paired look-up tables (regular/annual
// observations and year-starts) plus two stat maps, a hydro key, season count,
// and two scalar controls — all independently sourced by the caller; no context
// struct spans them. The length reflects a per-entity iterative contribution
// loop that re-solves the annual Yule-Walker system at each ceiling reduction
// and cannot be decomposed without threading the mutable `estimates` slice
// across helpers. The one sub-block shared with the regular path — failing-
// season detection — is extracted into `detect_failing_seasons`; the per-season
// re-solve body that follows is annual-specific and legitimately stays here.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn reduce_entity_orders_annual(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_id: EntityId,
    indices: &[usize],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    group_z_year_starts: &HashMap<(EntityId, usize), i32>,
    annual_group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    annual_group_year_starts: &HashMap<(EntityId, usize), i32>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    annual_stats_map: &HashMap<(EntityId, usize), &AnnualSeasonalStats>,
    initial_max_order: usize,
    z_alpha: f64,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    let mut annual_obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
    let mut annual_stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
    let mut z_year_starts: Vec<i32> = vec![0; n_seasons];
    let mut a_year_starts: Vec<i32> = vec![0; n_seasons];
    for season in 0..n_seasons {
        if let Some(obs) = group_obs.get(&(hydro_id, season)) {
            obs_by_season[season].clone_from(obs);
        }
        if let Some(ann_obs) = annual_group_obs.get(&(hydro_id, season)) {
            annual_obs_by_season[season].clone_from(ann_obs);
        }
        if let Some(s) = stats_map.get(&(hydro_id, season)) {
            stats_by_season[season] = (s.mean, s.std);
        }
        if let Some(s) = annual_stats_map.get(&(hydro_id, season)) {
            annual_stats_by_season[season] = (s.mean_m3s, s.std_m3s);
        }
        if let Some(&y) = group_z_year_starts.get(&(hydro_id, season)) {
            z_year_starts[season] = y;
        }
        if let Some(&y) = annual_group_year_starts.get(&(hydro_id, season)) {
            a_year_starts[season] = y;
        }
    }
    let std_by_season: Vec<f64> = stats_by_season.iter().map(|&(_, s)| s).collect();

    let mut max_orders: Vec<usize> = vec![initial_max_order; n_seasons];
    let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    for &idx in indices {
        let est = &estimates[idx];
        if est.season_id < n_seasons {
            all_coeffs[est.season_id].clone_from(&est.coefficients);
        }
    }

    // A season is "frozen" when its AR component has been driven to zero.
    // ψ is still permitted to update via order-0 re-fits.
    let mut frozen: Vec<bool> = vec![false; n_seasons];
    for &idx in indices {
        let sid = estimates[idx].season_id;
        if estimates[idx].coefficients.is_empty() {
            frozen[sid] = true;
        }
    }

    let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
    let annual_obs_refs: Vec<&[f64]> = annual_obs_by_season.iter().map(Vec::as_slice).collect();

    loop {
        // Detect failing seasons (negative contribution among the φ entries).
        let failing_seasons = detect_failing_seasons(
            estimates,
            indices,
            &frozen,
            n_seasons,
            &all_coeffs,
            &std_by_season,
            hydro_id,
            all_reductions,
        );
        if failing_seasons.is_empty() {
            break;
        }

        let mut any_reselected = false;
        for &season_id in &failing_seasons {
            if max_orders[season_id] == 0 {
                continue;
            }
            max_orders[season_id] -= 1;

            // Re-solve at the (possibly reduced) ceiling. When the ceiling has
            // dropped to 0 we still solve the 1×1 extended YW so ψ is refreshed
            // for the new (AR-empty) configuration.
            let stats_s = stats_by_season[season_id];
            if stats_s.1 == 0.0
                || obs_by_season[season_id].len() < 2
                || annual_obs_by_season[season_id].is_empty()
                || annual_stats_by_season[season_id].1 == 0.0
            {
                // No data to refit — drop AR entirely and freeze.
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        estimates[idx].residual_std_ratio = 1.0;
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
                continue;
            }

            let n_obs = obs_by_season[season_id].len();
            let selected_order = if max_orders[season_id] == 0 {
                0
            } else {
                let facp = conditional_facp_partitioned(
                    season_id,
                    max_orders[season_id],
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );
                select_order_pacf_annual(&facp, n_obs, z_alpha).selected_order
            };
            let yw_result = estimate_periodic_ar_annual_coefficients(
                season_id,
                selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
                &z_year_starts,
                &annual_obs_refs,
                &annual_stats_by_season,
                &a_year_starts,
            );

            // Annual stats live at prev_season under the storage convention
            // (PDF time-index of A_{t-1}).
            let prev_season = (season_id + n_seasons - 1) % n_seasons;
            let (ann_mean, ann_std) = annual_stats_by_season[prev_season];
            for &idx in indices {
                if estimates[idx].season_id == season_id {
                    estimates[idx]
                        .coefficients
                        .clone_from(&yw_result.coefficients);
                    estimates[idx].residual_std_ratio = yw_result.residual_std_ratio;
                    estimates[idx].annual = Some(cobre_core::scenario::AnnualComponent {
                        coefficient: yw_result.annual_coefficient,
                        mean_m3s: ann_mean,
                        std_m3s: ann_std,
                    });
                    all_coeffs[season_id].clone_from(&yw_result.coefficients);
                }
            }

            if max_orders[season_id] == 0 || yw_result.coefficients.is_empty() {
                frozen[season_id] = true;
                continue;
            }

            // Re-check φ_1 after the new YW solve. φ_1 < 0 is treated the same
            // as in the initial prepass: drop AR (ψ retained from the order-0
            // refit below).
            if has_negative_phi1(&all_coeffs[season_id]) {
                let original_order = all_coeffs[season_id].len();
                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::Phi1Negative,
                    });
                let yw0 = estimate_periodic_ar_annual_coefficients(
                    season_id,
                    0,
                    n_seasons,
                    &obs_refs,
                    &stats_by_season,
                    &z_year_starts,
                    &annual_obs_refs,
                    &annual_stats_by_season,
                    &a_year_starts,
                );
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        estimates[idx].residual_std_ratio = yw0.residual_std_ratio;
                        estimates[idx].annual = Some(cobre_core::scenario::AnnualComponent {
                            coefficient: yw0.annual_coefficient,
                            mean_m3s: ann_mean,
                            std_m3s: ann_std,
                        });
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
            } else {
                any_reselected = true;
            }
        }
        if !any_reselected {
            break;
        }
    }
}

/// Stage index entry: `(start_date, end_date, stage_id, season_id)`.
type StageSeasonEntry = (chrono::NaiveDate, chrono::NaiveDate, i32, usize);

/// Return type of [`build_pacf_stage_lookups`]:
/// `(stage_index, stats_map, n_seasons)`.
type PacfStageLookups<'a> = (
    Vec<StageSeasonEntry>,
    HashMap<(EntityId, usize), &'a SeasonalStats>,
    usize,
);

/// Build the stage-season index, stats map, and season count for PACF estimation.
///
/// Returns `(stage_index, stats_map, n_seasons)` where `stage_index` is sorted
/// by start date and `stats_map` keys are `(EntityId, season_id)`.
fn build_pacf_stage_lookups<'a>(
    stages: &[cobre_core::temporal::Stage],
    seasonal_stats: &'a [SeasonalStats],
) -> PacfStageLookups<'a> {
    let mut stage_index = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect::<Vec<_>>();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);

    let stage_id_to_season: HashMap<i32, usize> = stage_index
        .iter()
        .map(|&(_, _, stage_id, season_id)| (stage_id, season_id))
        .collect();

    let stats_map: HashMap<(EntityId, usize), &SeasonalStats> = seasonal_stats
        .iter()
        .filter_map(|s| {
            let season_id = stage_id_to_season.get(&s.stage_id).copied()?;
            Some(((s.entity_id, season_id), s))
        })
        .collect();

    let n_seasons = stage_index
        .iter()
        .map(|&(_, _, _, season_id)| season_id + 1)
        .max()
        .unwrap_or(0);

    (stage_index, stats_map, n_seasons)
}

/// Group raw observations by `(EntityId, season_id)` for PACF fitting.
fn group_observations_by_season(
    observations: &[(EntityId, NaiveDate, f64)],
    hydro_ids: &[EntityId],
    stage_index: &[(chrono::NaiveDate, chrono::NaiveDate, i32, usize)],
    season_map: Option<&cobre_core::temporal::SeasonMap>,
) -> HashMap<(EntityId, usize), Vec<f64>> {
    let entity_set: HashSet<EntityId> = hydro_ids.iter().copied().collect();
    let mut group_obs: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }
        let Some(season_id) = find_season_for_date(stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
        else {
            continue;
        };
        group_obs
            .entry((entity_id, season_id))
            .or_default()
            .push(value);
    }
    group_obs
}

/// Build initial AR coefficient estimates for all hydros using periodic PACF + YW.
fn estimate_all_hydro_ar_coefficients(
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    n_seasons: usize,
    max_order: usize,
    z_alpha: f64,
) -> Vec<ArCoefficientEstimate> {
    let mut estimates: Vec<ArCoefficientEstimate> = Vec::new();
    for &hydro_id in hydro_ids {
        let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
        let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
        for season in 0..n_seasons {
            if let Some(obs) = group_obs.get(&(hydro_id, season)) {
                obs_by_season[season].clone_from(obs);
            }
            if let Some(stats) = stats_map.get(&(hydro_id, season)) {
                stats_by_season[season] = (stats.mean, stats.std);
            }
        }
        let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
        for season in 0..n_seasons {
            let stats_s = stats_by_season[season];
            if stats_s.1 == 0.0 || obs_by_season[season].len() < 2 {
                estimates.push(ArCoefficientEstimate {
                    hydro_id,
                    season_id: season,
                    coefficients: Vec::new(),
                    residual_std_ratio: 1.0,
                    annual: None,
                });
                continue;
            }
            let n_obs = obs_by_season[season].len();
            let pacf_values =
                periodic_pacf(season, max_order, n_seasons, &obs_refs, &stats_by_season);
            let pacf_result = select_order_pacf(&pacf_values, n_obs, z_alpha);
            let yw_result = estimate_periodic_ar_coefficients(
                season,
                pacf_result.selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
            );
            estimates.push(ArCoefficientEstimate {
                hydro_id,
                season_id: season,
                coefficients: yw_result.coefficients,
                residual_std_ratio: yw_result.residual_std_ratio,
                annual: None,
            });
        }
    }
    estimates
}

/// PAR-fit knobs for `iterative_pacf_reduction` and its helpers.
struct PacfReductionParams {
    initial_max_order: usize,
    z_alpha: f64,
    max_coeff_magnitude: Option<f64>,
}

/// Apply magnitude-bound and `phi_1` pre-passes, recording reductions in `all_reductions`.
fn apply_prepass_reductions(
    estimates: &mut [ArCoefficientEstimate],
    params: &PacfReductionParams,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    if let Some(threshold) = params.max_coeff_magnitude {
        for est in estimates.iter_mut() {
            let has_explosive = est.coefficients.iter().any(|c| c.abs() > threshold);
            if has_explosive {
                let original_order = est.coefficients.len();
                all_reductions
                    .entry(est.hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id: est.season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::MagnitudeBound,
                    });
                est.coefficients.clear();
                est.residual_std_ratio = 1.0;
            }
        }
    }
    for est in estimates.iter_mut() {
        if has_negative_phi1(&est.coefficients) {
            let original_order = est.coefficients.len();
            all_reductions
                .entry(est.hydro_id)
                .or_default()
                .push(ContributionReduction {
                    season_id: est.season_id,
                    original_order,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                });
            est.coefficients.clear();
            est.residual_std_ratio = 1.0;
        }
    }
}

/// Run the iterative PACF order-reduction loop for one entity.
///
/// Mutates `estimates` in-place for the seasons indexed by `indices` and appends
/// `ContributionReduction` records to `all_reductions`.
fn reduce_entity_orders(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_id: EntityId,
    indices: &[usize],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    params: &PacfReductionParams,
    all_reductions: &mut HashMap<EntityId, Vec<ContributionReduction>>,
) {
    let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    let mut stats_by_season: Vec<(f64, f64)> = vec![(0.0, 0.0); n_seasons];
    for season in 0..n_seasons {
        if let Some(obs) = group_obs.get(&(hydro_id, season)) {
            obs_by_season[season].clone_from(obs);
        }
        if let Some(stats) = stats_map.get(&(hydro_id, season)) {
            stats_by_season[season] = (stats.mean, stats.std);
        }
    }
    let std_by_season: Vec<f64> = (0..n_seasons)
        .map(|sid| stats_map.get(&(hydro_id, sid)).map_or(0.0, |s| s.std))
        .collect();
    let mut max_orders: Vec<usize> = vec![params.initial_max_order; n_seasons];
    let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_seasons];
    for &idx in indices {
        let est = &estimates[idx];
        if est.season_id < n_seasons {
            all_coeffs[est.season_id].clone_from(&est.coefficients);
        }
    }
    let mut frozen: Vec<bool> = vec![false; n_seasons];
    for &idx in indices {
        let sid = estimates[idx].season_id;
        if estimates[idx].coefficients.is_empty() {
            frozen[sid] = true;
        }
    }
    let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
    loop {
        let failing_seasons = detect_failing_seasons(
            estimates,
            indices,
            &frozen,
            n_seasons,
            &all_coeffs,
            &std_by_season,
            hydro_id,
            all_reductions,
        );
        if failing_seasons.is_empty() {
            break;
        }
        let mut any_reselected = false;
        for &season_id in &failing_seasons {
            if max_orders[season_id] == 0 {
                continue;
            }
            max_orders[season_id] -= 1;
            if max_orders[season_id] == 0 {
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        estimates[idx].residual_std_ratio = 1.0;
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
                continue;
            }
            let stats_s = stats_by_season[season_id];
            if stats_s.1 == 0.0 || obs_by_season[season_id].len() < 2 {
                frozen[season_id] = true;
                continue;
            }
            let n_obs = obs_by_season[season_id].len();
            let pacf_values = periodic_pacf(
                season_id,
                max_orders[season_id],
                n_seasons,
                &obs_refs,
                &stats_by_season,
            );
            let pacf_result = select_order_pacf(&pacf_values, n_obs, params.z_alpha);
            let yw_result = estimate_periodic_ar_coefficients(
                season_id,
                pacf_result.selected_order,
                n_seasons,
                &obs_refs,
                &stats_by_season,
            );
            for &idx in indices {
                if estimates[idx].season_id == season_id {
                    estimates[idx]
                        .coefficients
                        .clone_from(&yw_result.coefficients);
                    estimates[idx].residual_std_ratio = yw_result.residual_std_ratio;
                    all_coeffs[season_id].clone_from(&yw_result.coefficients);
                }
            }
            if has_negative_phi1(&all_coeffs[season_id]) {
                let original_order = all_coeffs[season_id].len();
                all_reductions
                    .entry(hydro_id)
                    .or_default()
                    .push(ContributionReduction {
                        season_id,
                        original_order,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::Phi1Negative,
                    });
                for &idx in indices {
                    if estimates[idx].season_id == season_id {
                        estimates[idx].coefficients.clear();
                        estimates[idx].residual_std_ratio = 1.0;
                        all_coeffs[season_id].clear();
                        frozen[season_id] = true;
                    }
                }
            } else {
                any_reselected = true;
            }
        }
        if !any_reselected {
            break;
        }
    }
}

/// Iteratively reduce AR orders via PACF re-selection and contribution validation.
///
/// For each entity, maintains per-season `max_order` ceilings. When contribution
/// analysis detects negative contributions for a season, reduces that season's
/// ceiling by 1 and re-runs the full PACF selection + YW estimation + `phi_1`
/// check + contribution validation cycle. Repeats until all seasons pass or
/// their ceilings reach 0.
fn iterative_pacf_reduction(
    estimates: &mut [ArCoefficientEstimate],
    n_seasons: usize,
    hydro_ids: &[EntityId],
    group_obs: &HashMap<(EntityId, usize), Vec<f64>>,
    stats_map: &HashMap<(EntityId, usize), &SeasonalStats>,
    params: &PacfReductionParams,
) -> HashMap<EntityId, Vec<ContributionReduction>> {
    let mut all_reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();

    apply_prepass_reductions(estimates, params, &mut all_reductions);

    let mut hydro_indices: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, est) in estimates.iter().enumerate() {
        hydro_indices.entry(est.hydro_id).or_default().push(idx);
    }

    for &hydro_id in hydro_ids {
        let Some(indices) = hydro_indices.get(&hydro_id) else {
            continue;
        };
        reduce_entity_orders(
            estimates,
            n_seasons,
            hydro_id,
            indices,
            group_obs,
            stats_map,
            params,
            &mut all_reductions,
        );
    }

    all_reductions
}

/// Build an [`EstimationReport`] from AR estimates and contribution validation results.
///
/// This function is infallible: it only reorganises already-computed data.
/// For each hydro plant the selected order is the **maximum** across all
/// seasons. These choices align with how the I/O layer (`FittingReport`)
/// expects a single order per hydro.
// Rationale: the `contribution_reductions` map is always built with the default
// hasher by the in-crate callers and the report consumer; generalising over
// `BuildHasher` would widen the signature for no caller that uses a custom
// hasher, so the implicit-hasher lint is suppressed rather than satisfied.
#[allow(clippy::implicit_hasher)]
pub fn build_estimation_report(
    estimates: &[ArCoefficientEstimate],
    n_seasons: usize,
    contribution_reductions: &HashMap<EntityId, Vec<ContributionReduction>>,
    method: &str,
) -> EstimationReport {
    // Group coefficient vectors by hydro_id (estimates are already sorted by
    // (hydro_id, season_id) from estimate_ar_coefficients).
    let mut hydro_coeffs: BTreeMap<EntityId, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
    for est in estimates {
        hydro_coeffs
            .entry(est.hydro_id)
            .or_default()
            .push((est.season_id, est.coefficients.clone()));
    }

    let mut entries: BTreeMap<EntityId, HydroEstimationEntry> = BTreeMap::new();

    for (hydro_id, mut season_coeffs) in hydro_coeffs {
        // Sort by season_id ascending (should already be sorted, but ensure it).
        season_coeffs.sort_by_key(|(season_id, _)| *season_id);

        // Compute max selected_order as the maximum actual coefficient length
        // across all seasons for this hydro (after all truncations).
        let selected_order = season_coeffs
            .iter()
            .map(|(_, coeffs)| coeffs.len())
            .max()
            .unwrap_or(0);

        // Build per-season coefficient vectors, filling missing seasons with empty vecs.
        // The season_coeffs may not cover all n_seasons if some were skipped.
        let season_map: HashMap<usize, Vec<f64>> = season_coeffs.into_iter().collect();
        let coefficients: Vec<Vec<f64>> = (0..n_seasons)
            .map(|sid| season_map.get(&sid).cloned().unwrap_or_default())
            .collect();

        let reductions = contribution_reductions
            .get(&hydro_id)
            .cloned()
            .unwrap_or_default();

        #[allow(clippy::cast_possible_truncation)]
        entries.insert(
            hydro_id,
            HydroEstimationEntry {
                selected_order: selected_order as u32,
                coefficients,
                contribution_reductions: reductions,
            },
        );
    }

    EstimationReport {
        entries,
        method: method.to_string(),
        white_noise_fallbacks: Vec::new(),
        std_ratio_warnings: Vec::new(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::useless_vec
)]
mod tests;
