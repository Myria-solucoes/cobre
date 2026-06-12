//! Seasonal mean/std estimation, history classification, and date-to-season
//! lookup primitives for PAR model fitting.

use std::collections::HashMap;

use chrono::NaiveDate;

use cobre_core::{
    EntityId,
    temporal::{SeasonMap, Stage},
};

use crate::StochasticError;

/// Build the season-aware stage index `(start_date, end_date, stage_id, season_id)`.
///
/// Only stages carrying a `season_id` are included; the result is sorted by
/// `start_date` so [`find_season_for_date`] can binary-search it for range
/// containment.
pub(super) fn build_stage_index(stages: &[Stage]) -> Vec<(NaiveDate, NaiveDate, i32, usize)> {
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);
    stage_index
}

// ---------------------------------------------------------------------------
// Seasonal statistics
// ---------------------------------------------------------------------------

/// Seasonal mean and standard deviation for one entity–season pair.
///
/// Produced by [`estimate_seasonal_stats`] and consumed by AR coefficient
/// estimation routines. The caller (typically in a higher-level crate) is
/// responsible for mapping between this type and any crate-specific row type
/// used for storage or serialization.
///
/// The `stage_id` field holds the identifier of the **first** stage whose
/// `season_id` matches the season for this row — it is not a stage index.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalStats {
    /// Entity (e.g., hydro plant) identifier.
    pub entity_id: EntityId,
    /// Identifier of the first stage that belongs to this season.
    pub stage_id: i32,
    /// Sample mean of observed values (m³/s or whatever unit the caller uses).
    pub mean: f64,
    /// Population-divisor standard deviation (1/N divisor), matching the
    /// Maceira-Damazio PAR(p)-A standard-deviation convention.
    pub std: f64,
}

/// Classification of a per-(entity, season) historical observation series.
///
/// Applied inside [`estimate_seasonal_stats`], so the override propagates
/// through both the classical PAR(p) and PAR(p)-A paths. On PAR(p)-A the
/// structural-zero short-circuit at lag 1 of the conditional FACP turns
/// a degenerate `(value, 0)` bucket into an explicit order-0 fit; on
/// classical PAR(p) the same `std = 0` zeroes every periodic
/// autocorrelation, which drives `select_order_pacf` to order 0
/// implicitly. Either way, the bucket cannot inject spurious
/// autoregressive structure into adjacent months' PACFs.
///
/// - [`Default`](HistoryClass::Default): no specific behaviour; standard
///   fitting applies.
/// - [`Constant`](HistoryClass::Constant): every observation equals the
///   same value (or every observation is zero/null). Mean and std are
///   forced to that constant and 0, respectively; AR order is forced to
///   0 and any annual coefficient is suppressed. Common for plants with
///   regulated/transposed flows whose incremental inflow is structurally
///   constant for a given month.
/// - [`ManyNegative`](HistoryClass::ManyNegative): more than 10% of
///   observations are strictly negative — a signal that the upstream
///   incremental construction (the bridge subtracting upstream postos)
///   has produced unphysical values for this month. Detected for
///   diagnostics, but **does not override fitting** — the flag is
///   operator information, not a fit instruction.
/// - [`Saturated`](HistoryClass::Saturated): more than 50% of
///   observations equal the modal value — a flow cap (turbine/reservoir
///   constraint) or a low-flow constant (transposed flow plants).
///   Treated like `Constant` with the cap as the constant. The std=0
///   propagates structural zeros into adjacent months' PACF rows. No
///   P99 condition: low-flow constants (cap=1.0, cap=0.0) classify as
///   saturated just as readily as high caps.
///
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistoryClass {
    /// No specific behaviour — run the standard fit.
    Default,
    /// Every observation is the same value (or all zero/null).
    Constant {
        /// The constant value to use as the seasonal mean.
        value: f64,
    },
    /// More than 10% of observations are strictly negative.
    /// `sample_mean` is the empirical mean over the full series (used as the
    /// fallback constant; std forced to 0).
    ManyNegative {
        /// Empirical mean of the observation series.
        sample_mean: f64,
    },
    /// Saturating cap (>50% of observations at the modal value).
    Saturated {
        /// The cap value (modal value of the series).
        cap: f64,
    },
}

impl HistoryClass {
    /// Returns the override `(mean, std)` that should replace the empirical
    /// stats for fitting purposes.
    ///
    /// `Constant` and `Saturated` force the seasonal mean to the
    /// constant/cap value and the std to 0. The `std = 0` short-circuits
    /// every downstream fitter — PAR(p)-A explicitly, classical PAR(p)
    /// implicitly via zeroed periodic autocorrelations — so order 0 is
    /// the result on either path. `ManyNegative` is purely diagnostic
    /// and returns `None` (the classification does not override fitting
    /// for it). `Default` also returns `None`.
    #[must_use]
    pub fn stats_override(self) -> Option<(f64, f64)> {
        match self {
            HistoryClass::Default | HistoryClass::ManyNegative { .. } => None,
            HistoryClass::Constant { value } => Some((value, 0.0)),
            HistoryClass::Saturated { cap } => Some((cap, 0.0)),
        }
    }

    /// Returns `true` when the classification forces a degenerate fit
    /// (order 0, no AR/annual coefficients). Currently `Constant` and
    /// `Saturated`. `ManyNegative` is diagnostic only, so it returns
    /// `false`.
    #[must_use]
    pub fn is_degenerate(self) -> bool {
        matches!(
            self,
            HistoryClass::Constant { .. } | HistoryClass::Saturated { .. }
        )
    }
}

/// Classify a single (entity, season) observation series per the
/// [`HistoryClass`] taxonomy.
///
/// The classifier runs in priority order
/// `Constant` → `ManyNegative` → `Saturated` → `Default`: constant series
/// take precedence over negative-pathological detection, which in turn
/// takes precedence over saturation. Observations are rounded to the
/// nearest integer for mode counting (matching the precision of the
/// standard historical inflow input format, which stores values in m³/s
/// as integers). The constancy check uses an absolute tolerance of
/// `1e-6` to absorb the float round-trip from parquet.
///
/// Returns `HistoryClass::Constant { value: 0.0 }` for an empty input — the
/// degenerate single-observation case is treated the same as a zero-history
/// series so that downstream fitters short-circuit predictably.
pub fn classify_history(observations: &[f64]) -> HistoryClass {
    if observations.is_empty() {
        return HistoryClass::Constant { value: 0.0 };
    }

    let first = observations[0];
    let const_tol = 1e-6;

    // Constant — every observation matches the first within tolerance.
    if observations.iter().all(|&v| (v - first).abs() < const_tol) {
        return HistoryClass::Constant { value: first };
    }

    // ManyNegative — more than 10% strictly negative.
    let n = observations.len();
    let n_neg = observations.iter().filter(|&&v| v < 0.0).count();
    #[allow(clippy::cast_precision_loss)]
    if (n_neg as f64) / (n as f64) > 0.10 {
        #[allow(clippy::cast_precision_loss)]
        let sample_mean = observations.iter().sum::<f64>() / n as f64;
        return HistoryClass::ManyNegative { sample_mean };
    }

    // Saturated — modal value occupies more than 50% of observations.
    //
    // Round to integer for mode counting (the historical inflow format
    // stores values to 1 m³/s). No P99 guard: low-flow constants
    // (cap=0.0, cap=1.0) classify as saturated just as eagerly as high
    // caps. The driving criterion is structural constancy of the bucket,
    // not magnitude.
    #[allow(clippy::cast_possible_truncation)]
    let mut sorted: Vec<i64> = observations.iter().map(|v| v.round() as i64).collect();
    sorted.sort_unstable();
    // Largest run of equal values gives the mode.
    let mut best_count = 0_usize;
    let mut best_value: i64 = sorted[0];
    let mut run = 1_usize;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] {
            run += 1;
        } else {
            if run > best_count {
                best_count = run;
                best_value = sorted[i - 1];
            }
            run = 1;
        }
    }
    if run > best_count {
        best_count = run;
        best_value = sorted[sorted.len() - 1];
    }
    #[allow(clippy::cast_precision_loss)]
    if (best_count as f64) / (n as f64) > 0.50 {
        return HistoryClass::Saturated {
            cap: best_value as f64,
        };
    }

    HistoryClass::Default
}

/// Estimate seasonal means and standard deviations from historical observations.
///
/// Groups observations by `(entity_id, season_id)` and computes the sample
/// mean and population-divisor (1/N) standard deviation for each group,
/// matching the Maceira-Damazio PAR(p)-A standard-deviation convention.
/// Only entities listed in `entity_ids` are processed; observations for
/// other entities are silently ignored.
///
/// Stages with `season_id = None` are skipped when building the date-to-season
/// mapping. Observations whose date does not fall within any stage's
/// `[start_date, end_date)` range produce an error.
///
/// # Parameters
///
/// - `observations` — flat slice of `(entity_id, date, value)` triples,
///   sorted by `(entity_id, date)` (parser guarantee).
/// - `stages` — all stages in canonical index order. Each stage has
///   `start_date` (inclusive), `end_date` (exclusive), `season_id`, and `id`.
/// - `entity_ids` — canonical sorted list of entity IDs to estimate for.
///
/// # Errors
///
/// - [`StochasticError::InsufficientData`] when a `(entity, season)` group has
///   fewer than 2 observations (a degenerate single-sample bucket has no
///   meaningful std and would propagate zeros into every downstream
///   correlation, so it is rejected up front).
/// - [`StochasticError::InsufficientData`] when an observation date falls
///   outside every stage's date range.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::estimate_seasonal_stats;
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
/// let stages = vec![
///     stage(1, 2020, 1, 2020, 2, 0),
///     stage(2, 2020, 2, 2020, 3, 1),
/// ];
/// let obs = vec![
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(), 100.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 20).unwrap(), 200.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 10).unwrap(), 150.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 20).unwrap(), 250.0),
/// ];
/// let entity_ids = vec![EntityId::from(1)];
/// let stats = estimate_seasonal_stats(&obs, &stages, &entity_ids).unwrap();
/// assert_eq!(stats.len(), 2);
/// assert!((stats[0].mean - 150.0).abs() < 1e-10);
/// ```
pub fn estimate_seasonal_stats(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
) -> Result<Vec<SeasonalStats>, StochasticError> {
    estimate_seasonal_stats_with_season_map(observations, stages, entity_ids, None)
}

/// Estimate seasonal statistics with an optional [`SeasonMap`] fallback.
///
/// When `season_map` is `Some`, historical observation dates that fall outside
/// the study horizon are resolved to a season using the calendar-based cycle
/// definition. This allows PAR estimation from inflow history that predates
/// the study period.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when an observation date
/// cannot be mapped to any season, or when fewer than 2 observations exist
/// for any `(entity, season)` group.
pub fn estimate_seasonal_stats_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<SeasonalStats>, StochasticError> {
    if observations.is_empty() {
        return Ok(Vec::new());
    }

    // Build a set of entity IDs for O(1) membership checks.
    let entity_set: std::collections::HashSet<EntityId> = entity_ids.iter().copied().collect();

    // Build stage index: (start_date, end_date, stage_id, season_id).
    // Only include stages that have a season_id. Sorted by start_date for
    // binary-search based range lookup.
    let stage_index = build_stage_index(stages);

    // Map (entity_id, season_id) -> (observations: Vec<f64>, first_stage_id: i32).
    // The `first_stage_id` is the id of the first stage (lowest stage.id among
    // those with that season_id, determined from the sorted stage_index).
    let mut group_map: HashMap<(EntityId, usize), (Vec<f64>, i32)> = HashMap::new();

    // Build a separate lookup from season_id -> first_stage_id (the stage.id of
    // the first stage with that season_id, where "first" means lowest start_date
    // i.e. first in stage_index order).
    let mut season_first_stage: HashMap<usize, i32> = HashMap::new();
    for &(_, _, stage_id, season_id) in &stage_index {
        season_first_stage.entry(season_id).or_insert(stage_id);
    }

    for &(entity_id, date, value) in observations {
        // Skip entities not in the study set.
        if !entity_set.contains(&entity_id) {
            continue;
        }

        // Try exact stage date containment first (for in-range observations),
        // then fall back to the SeasonMap calendar-based mapping (for historical
        // observations that predate the study horizon).
        let season_id = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            .ok_or_else(|| StochasticError::InsufficientData {
                context: format!(
                    "observation date {date} for entity {entity_id} \
                     does not match any stage date range or season definition"
                ),
            })?;

        // Skip observations whose resolved season has no study stage. A season
        // not present in `season_first_stage` is not lag-reachable for a study
        // with no pre-study stages, so its stats are never consumed; indexing it
        // would panic for partial-year studies whose history spans the full cycle.
        let Some(&first_stage_id) = season_first_stage.get(&season_id) else {
            continue;
        };
        let entry = group_map
            .entry((entity_id, season_id))
            .or_insert_with(|| (Vec::new(), first_stage_id));
        entry.0.push(value);
    }

    // Compute mean and population-divisor std for each group.
    //
    // The Maceira-Damazio PAR(p)-A formulation computes sigma^Z_m with the
    // 1/N population divisor, not the Bessel-corrected 1/(N-1). Using the
    // population divisor is required for self-consistent conditional FACP
    // values and selected AR orders — the sample-vs-population scale
    // factor would otherwise propagate through every cross-correlation.
    let mut result: Vec<SeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, _season_id), (values, stage_id)) in group_map {
        let n = values.len();
        if n < 2 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} season mapped to stage {stage_id} \
                     has {n} observation(s); need at least 2 for std estimation"
                ),
            });
        }

        // n is the number of observations; the cast to f64 is intentional here.
        // In practice, observation counts never exceed ~10^6 (well within the
        // 2^53 exact-integer range of f64), so precision loss cannot occur.
        #[allow(clippy::cast_precision_loss)]
        let mean = values.iter().copied().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        // History-class override: degenerate buckets (constant series,
        // saturated caps) get a forced (constant, 0) pair. The `std = 0`
        // short-circuits both PAR(p)-A (explicit structural-zero rule)
        // and classical PAR(p) (zeroed periodic autocorrelations →
        // order 0). Typical examples are constant low-flow/high-flow
        // buckets and ecological-flow months. The classifier returns
        // `Default` for normal series, in which case we keep the
        // empirical (mean, std) computed above.
        let (final_mean, final_std) = match classify_history(&values).stats_override() {
            Some((override_mean, override_std)) => (override_mean, override_std),
            None => (mean, std),
        };

        result.push(SeasonalStats {
            entity_id,
            stage_id,
            mean: final_mean,
            std: final_std,
        });
    }

    // Sort by (entity_id, stage_id) ascending to match parser convention.
    result.sort_unstable_by_key(|s| (s.entity_id.0, s.stage_id));

    Ok(result)
}

/// Find the `season_id` for `date` by binary-searching `stage_index`.
///
/// `stage_index` must be sorted by `start_date`. Returns `None` when `date`
/// falls outside every stage's `[start_date, end_date)` range.
#[must_use]
pub fn find_season_for_date(
    stage_index: &[(NaiveDate, NaiveDate, i32, usize)],
    date: NaiveDate,
) -> Option<usize> {
    let pos = stage_index.partition_point(|(start, _, _, _)| *start <= date);
    if pos == 0 {
        return None;
    }
    let (_, end_date, _, season_id) = stage_index[pos - 1];
    if date < end_date {
        Some(season_id)
    } else {
        None
    }
}
