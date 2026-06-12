//! Per-season statistics of the rolling 12-month average for the PAR-A model.

use std::collections::{HashMap, HashSet};

use chrono::NaiveDate;

use cobre_core::{
    EntityId,
    temporal::{SeasonMap, Stage},
};

use super::seasonal_stats::{build_stage_index, find_season_for_date};
use crate::StochasticError;

// ---------------------------------------------------------------------------
// Annual seasonal stats helper (PAR-A eqs. 17, 18)
// ---------------------------------------------------------------------------

/// Per-season sample statistics of the rolling 12-month average.
///
/// One entry per season for one entity. Computed by
/// [`estimate_annual_seasonal_stats`] from a chronological observation list.
///
/// The `mean_m3s` and `std_m3s` fields are in the original m³/s units of the
/// observation series; they are **not** standardised. The standardisation
/// happens inside the cross-correlation helpers from
/// [`build_extended_periodic_yw_matrix`], and the unit conversion to `ψ̂`
/// happens at `PrecomputedPar::build` time.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AnnualSeasonalStats {
    /// Entity (e.g., hydro plant) identifier.
    pub hydro_id: EntityId,
    /// Season ID (0-based).
    pub season_id: usize,
    /// Sample mean of the rolling 12-month average for this (entity, season) pair, in m³/s.
    pub mean_m3s: f64,
    /// Population-divisor standard deviation (`1/N` divisor) of the rolling
    /// 12-month average for this (entity, season) pair, in m³/s. Matches
    /// the Maceira-Damazio PAR(p)-A standard-deviation convention.
    pub std_m3s: f64,
}

/// Estimate the per-season sample statistics `(μ^A_m, σ^A_m)` of the rolling
/// 12-month average from chronological observations.
///
/// For each (entity, season) pair, `μ^A_m` is the sample mean of the rolling
/// 12-month average `A_t = (1/12) · Σ_{j=0..11} z[t-j]` values whose target
/// date falls in season `m`. `σ^A_m` is the **population-divisor standard
/// deviation** using divisor `1/N`, matching the Maceira-Damazio PAR(p)-A
/// standard-deviation convention and the workspace-wide convention used by
/// [`estimate_seasonal_stats`]. The PAR(p)-A runtime coefficient is then
/// `ψ̂ = ψ · σ_m / σ^A_m`, which requires `σ^A_m > 0` (enforced by the
/// output validator in `cobre-io`).
///
/// ## Algorithm
///
/// 1. Group observations by `EntityId` and sort each group chronologically.
/// 2. For each entity group, build the rolling 12-month average:
///    `A_{i+12} = (1/12) · Σ_{j=0..11} z[i+j]` for every chronological index `i`
///    such that `i + 12 < group.len()`. The target date is `group[i+12].date`.
/// 3. Group `A_{i+12}` values by the season of the target date (using
///    [`find_season_for_date`] + `season_map` fallback, mirroring
///    [`estimate_seasonal_stats_with_season_map`]).
/// 4. Compute per (entity, season) the sample mean and population-divisor std (1/N).
///
/// Returns rows sorted by `(hydro_id, season_id)` ascending.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when any requested `entity_id`
/// produces zero rolling-window `A_t` values (fewer than 13 observations for
/// that entity). The error names the entity and its observation count.
/// Silent fallback to the classical PAR path is not performed.
pub fn estimate_annual_seasonal_stats(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<AnnualSeasonalStats>, StochasticError> {
    // Build stage index for date-to-season mapping.
    let stage_index = build_stage_index(stages);

    let entity_set: HashSet<EntityId> = entity_ids.iter().copied().collect();

    // Group observations by entity in chronological order.
    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if entity_set.contains(&entity_id) {
            entity_obs.entry(entity_id).or_default().push((date, value));
        }
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    // Per-entity insufficient-data guard: every requested entity must produce
    // at least one rolling-window A_t value, which requires >= 13 observations.
    for &entity_id in entity_ids {
        let n_obs = entity_obs.get(&entity_id).map_or(0, Vec::len);
        if n_obs < 13 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} has {n_obs} observation(s); \
                     at least 13 are required to form one rolling 12-month average"
                ),
            });
        }
    }

    // Build rolling-window A_t values grouped by (entity_id, season_id).
    //
    // Indexing convention: an A value A_{t-1} = mean(z[t-12..t-1]) is stored
    // under the season of its own PDF time-index (t-1), which is the most
    // recent observation in the rolling window — `group[i + 11]`. With this
    // convention, `annual_stats_by_season[s]` contains stats for
    // `A_{t-1}` whose PDF time-index falls in season `s`, equivalently
    // `A_{t-1}` for `t` at season `s + 1`. The Yule-Walker callers
    // (`build_extended_periodic_yw_matrix`, `assemble_partitioned_covariance`)
    // index this map with `prev_season = (m - 1) mod n_seasons` to retrieve
    // the stats for the equation at current season `m`.
    let mut group_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    for &entity_id in entity_ids {
        let Some(group) = entity_obs.get(&entity_id) else {
            continue;
        };

        // For each index i such that i + 11 < group.len() (we still require
        // i + 12 <= group.len() to access the full 12-month window), the
        // rolling-window mean A = (1/12) * sum of z[i..i+12] is stored under
        // the season of group[i + 11].date — i.e., the PDF time-index of
        // A_{t-1} when target month t = i + 12.
        for i in 0..group.len().saturating_sub(12) {
            let target_date = group[i + 11].0;

            let Some(season_id) = find_season_for_date(&stage_index, target_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(target_date)))
            else {
                // Target date not in any stage and no season_map fallback — skip.
                continue;
            };

            // A_{(i+12)-1} = (1/12) * sum of z[i..i+12]; PDF time of this value
            // is i + 11, so it is stored under the season of group[i + 11].
            let mean_a: f64 = group[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
            group_map
                .entry((entity_id, season_id))
                .or_default()
                .push(mean_a);
        }
    }

    // Compute mean and population-divisor std for each (entity, season) group.
    //
    // The Maceira-Damazio PAR(p)-A formulation uses sigma^A_m with the 1/N
    // population divisor. Using the population divisor is required for
    // self-consistent partitioned-covariance FACPs — the sample-vs-population
    // scale factor would otherwise leak through every Z⊗A cross-correlation.
    let mut result: Vec<AnnualSeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, season_id), values) in &group_map {
        let n = values.len();
        #[allow(clippy::cast_precision_loss)]
        let mean_m3s = values.iter().copied().sum::<f64>() / n as f64;
        let std_m3s = if n >= 1 {
            #[allow(clippy::cast_precision_loss)]
            let var = values
                .iter()
                .map(|&v| (v - mean_m3s) * (v - mean_m3s))
                .sum::<f64>()
                / n as f64;
            var.sqrt()
        } else {
            0.0
        };

        result.push(AnnualSeasonalStats {
            hydro_id: *entity_id,
            season_id: *season_id,
            mean_m3s,
            std_m3s,
        });
    }

    // Sort by (hydro_id, season_id) ascending.
    result.sort_unstable_by_key(|s| (s.hydro_id.0, s.season_id));

    Ok(result)
}
