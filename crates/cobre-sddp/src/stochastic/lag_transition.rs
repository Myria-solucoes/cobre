//! Precomputation of per-stage lag accumulation weights and period
//! finalization flags from stage date boundaries and season definitions.
//!
//! [`precompute_stage_lag_transitions`] runs once at setup; the resulting
//! per-stage slice is consumed read-only on the hot path, keeping calendar
//! arithmetic out of inner solver loops.

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate, TimeDelta, Weekday};
use cobre_core::{
    entities::hydro::Hydro,
    initial_conditions::RecentObservation,
    temporal::{SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageLagTransition},
};
use cobre_stochastic::PeriodBlendWeight;

/// Pre-computed seed values for the lag accumulator, derived from
/// [`RecentObservation`] data in [`cobre_core::InitialConditions`].
///
/// Applied at every trajectory start (forward pass and simulation pipeline).
/// `weight_seed == 0.0` (no observations or non-Monthly season cycle) is the
/// zero-reset behaviour.
#[derive(Debug, Clone)]
pub struct RecentObservationSeed {
    /// Per-hydro accumulated `value_m3s * observation_hours`; zero for hydros
    /// without observations.
    pub accum_seed: Vec<f64>,
    /// Fraction of the lag period covered by pre-study observations
    /// (`total_observation_hours / total_period_hours`); one scalar because all
    /// observations share the same calendar period.
    pub weight_seed: f64,
}

impl RecentObservationSeed {
    /// Construct an all-zero seed for `hydro_count` hydros.
    #[must_use]
    pub fn zero(hydro_count: usize) -> Self {
        Self {
            accum_seed: vec![0.0_f64; hydro_count],
            weight_seed: 0.0,
        }
    }
}

/// Compute the lag accumulator seed from pre-study [`RecentObservation`] data.
///
/// Only the `Monthly` cycle is implemented; `Weekly`/`Custom`, an empty
/// `recent_obs`, a `None` `first_stage.season_id`, or empty `hydros` all return
/// a zero seed. Unknown `hydro_id` values are silently skipped, matching
/// `build_initial_state`.
pub(crate) fn compute_recent_observation_seed(
    recent_obs: &[RecentObservation],
    first_stage: &Stage,
    season_map: &SeasonMap,
    hydros: &[Hydro],
) -> RecentObservationSeed {
    let hydro_count = hydros.len();
    if recent_obs.is_empty() || hydro_count == 0 {
        return RecentObservationSeed::zero(hydro_count);
    }

    let Some(season_id) = first_stage.season_id else {
        return RecentObservationSeed::zero(hydro_count);
    };

    if !matches!(season_map.cycle_type, SeasonCycleType::Monthly) {
        // TODO(historical-replay-non-monthly): only Monthly seeding is implemented;
        // cobre-io `check_recent_observations_non_monthly_seed_gap` warns at load time.
        return RecentObservationSeed::zero(hydro_count);
    }

    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return RecentObservationSeed::zero(hydro_count);
    };

    let season_month = season_def.month_start;
    let year = find_season_year_monthly(first_stage.start_date, first_stage.end_date, season_month);
    let total_period_hours = month_total_hours(year, season_month);

    let mut accum_seed = vec![0.0_f64; hydro_count];
    let mut per_hydro_hours: HashMap<i32, f64> = HashMap::new();

    // `hydros` is System::hydros()'s canonical `(operational_start_date, id)`
    // order, id-ascending only when every hydro shares one start date; a
    // staggered-commissioning system breaks that coincidence, so the lookup
    // resolves through this position map, never `binary_search_by_key` over
    // `hydros` itself.
    let hydro_positions: HashMap<i32, usize> = hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id.0, idx))
        .collect();

    for obs in recent_obs {
        let Some(&idx) = hydro_positions.get(&obs.hydro_id.0) else {
            continue;
        };
        let obs_days = (obs.end_date - obs.start_date).num_days();
        let obs_hours = f64::from(
            u32::try_from(obs_days)
                .unwrap_or_else(|_| unreachable!("observation days always fit in u32")),
        ) * 24.0;
        accum_seed[idx] += obs.value_m3s * obs_hours;
        *per_hydro_hours.entry(obs.hydro_id.0).or_insert(0.0) += obs_hours;
    }

    // max per-hydro total, not the sum: all hydros observe the same period, so
    // summing would inflate the weight linearly with hydro count.
    let total_obs_hours = per_hydro_hours.values().copied().fold(0.0_f64, f64::max);
    let weight_seed = total_obs_hours / total_period_hours;

    RecentObservationSeed {
        accum_seed,
        weight_seed,
    }
}

/// Compute the exclusive end date of the calendar month identified by
/// `month` (1–12) and `year`.
pub(crate) fn month_exclusive_end(year: i32, month: u32) -> NaiveDate {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1u32)
    } else {
        (year, month + 1)
    };
    NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .unwrap_or_else(|| unreachable!("next-month date is always valid"))
}

/// Returns the total hours in the calendar month identified by `year` and
/// `month` (1–12). Each day is exactly 24 hours (timezone-free calendar dates, no DST).
pub(crate) fn month_total_hours(year: i32, month: u32) -> f64 {
    let first = NaiveDate::from_ymd_opt(year, month, 1)
        .unwrap_or_else(|| unreachable!("month-start date is always valid"));
    let next = month_exclusive_end(year, month);
    let days = u32::try_from((next - first).num_days())
        .unwrap_or_else(|_| unreachable!("days in a month always fit in u32"));
    f64::from(days) * 24.0
}

/// Determine the calendar year whose occurrence of `season_month` overlaps the
/// stage interval `[start_date, end_date)`, in a `Monthly` cycle.
///
/// Candidates are checked in order: `start_date.year()`, then the previous year
/// (a December-season stage starting in January), then the next year as a
/// fallback against unexpected gaps.
pub(crate) fn find_season_year_monthly(
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_month: u32,
) -> i32 {
    let candidate_year = start_date.year();
    let period_start = NaiveDate::from_ymd_opt(candidate_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month is always valid"));
    let period_end = month_exclusive_end(candidate_year, season_month);

    if start_date < period_end && end_date > period_start {
        return candidate_year;
    }

    let prev_year = candidate_year - 1;
    let period_start_prev = NaiveDate::from_ymd_opt(prev_year, season_month, 1)
        .unwrap_or_else(|| unreachable!("season month with previous year is always valid"));
    let period_end_prev = month_exclusive_end(prev_year, season_month);

    if start_date < period_end_prev && end_date > period_start_prev {
        return prev_year;
    }

    candidate_year + 1
}

/// Count the number of days in `[stage_start, stage_end)` that fall within
/// `[period_start, period_end)`. Returns 0 if there is no overlap.
pub(crate) fn days_in_period(
    stage_start: NaiveDate,
    stage_end: NaiveDate,
    period_start: NaiveDate,
    period_end: NaiveDate,
) -> u32 {
    let overlap_start = stage_start.max(period_start);
    let overlap_end = stage_end.min(period_end);
    if overlap_end > overlap_start {
        u32::try_from((overlap_end - overlap_start).num_days())
            .unwrap_or_else(|_| unreachable!("overlap days always fit in u32"))
    } else {
        0
    }
}

/// Concrete `[start, end)` calendar window for one occurrence of a season
/// period, plus its total duration in hours.
struct PeriodWindow {
    start: NaiveDate,
    end: NaiveDate,
    hours: f64,
}

/// Number of real calendar days in `year`-`month` (1–12).
fn days_in_month(year: i32, month: u32) -> u32 {
    let first = NaiveDate::from_ymd_opt(year, month, 1)
        .unwrap_or_else(|| unreachable!("month-start date is always valid"));
    let next = month_exclusive_end(year, month);
    u32::try_from((next - first).num_days())
        .unwrap_or_else(|_| unreachable!("days in a month always fit in u32"))
}

/// Resolve a `Custom` `season_def`'s `[start, end)` range anchored in `year`.
///
/// Mirrors `season_for_date`'s `Custom` arm (`day_start`/`day_end` defaults,
/// `start <= end` wrap-around); `day_end` is clamped to the real month length
/// here because a concrete date must be constructed (the tuple comparison in
/// `season_for_date` needs no such clamp).
fn custom_period_bounds(year: i32, season_def: &SeasonDefinition) -> (NaiveDate, NaiveDate) {
    let month_start = season_def.month_start;
    let day_start = season_def.day_start.unwrap_or(1);
    let month_end = season_def.month_end.unwrap_or(month_start);
    let day_end = season_def.day_end.unwrap_or(31);

    let start_day = day_start.min(days_in_month(year, month_start));
    let start = NaiveDate::from_ymd_opt(year, month_start, start_day)
        .unwrap_or_else(|| unreachable!("clamped custom start date is always valid"));

    let wraps = (month_start, day_start) > (month_end, day_end);
    let end_year = if wraps { year + 1 } else { year };
    let end_day = day_end.min(days_in_month(end_year, month_end));
    let end_inclusive = NaiveDate::from_ymd_opt(end_year, month_end, end_day)
        .unwrap_or_else(|| unreachable!("clamped custom end date is always valid"));

    (start, end_inclusive + TimeDelta::days(1))
}

/// Determine the calendar year whose occurrence of a `Custom` `season_def`
/// overlaps `[start_date, end_date)`. Generalizes
/// `find_season_year_monthly`'s candidate/previous-year/fallback search to a
/// day-level range.
fn find_season_year_custom(
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_def: &SeasonDefinition,
) -> i32 {
    let candidate_year = start_date.year();
    let (period_start, period_end) = custom_period_bounds(candidate_year, season_def);
    if start_date < period_end && end_date > period_start {
        return candidate_year;
    }

    let prev_year = candidate_year - 1;
    let (period_start_prev, period_end_prev) = custom_period_bounds(prev_year, season_def);
    if start_date < period_end_prev && end_date > period_start_prev {
        return prev_year;
    }

    candidate_year + 1
}

/// Resolve `season_def`'s concrete calendar window for `stage`'s occurrence.
///
/// `Monthly` routes through `find_season_year_monthly`/`month_exclusive_end`/
/// `month_total_hours` verbatim. `Weekly` derives the 7-day ISO-week window
/// containing `stage.start_date` directly — `season_for_date`'s week-53→52
/// fold is a season-id label fold, not a window fold, so the physical week
/// stays 7 real days regardless. `Custom` resolves `season_def`'s own range
/// via `find_season_year_custom`/`custom_period_bounds`.
fn period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    stage: &Stage,
) -> PeriodWindow {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = find_season_year_monthly(stage.start_date, stage.end_date, season_month);
            let start = NaiveDate::from_ymd_opt(year, season_month, 1)
                .unwrap_or_else(|| unreachable!("season month is always valid"));
            let end = month_exclusive_end(year, season_month);
            let hours = month_total_hours(year, season_month);
            PeriodWindow { start, end, hours }
        }
        SeasonCycleType::Weekly => {
            let iso_week = stage.start_date.iso_week();
            let start = NaiveDate::from_isoywd_opt(iso_week.year(), iso_week.week(), Weekday::Mon)
                .unwrap_or_else(|| unreachable!("iso week start date is always valid"));
            let end = start + TimeDelta::days(7);
            PeriodWindow {
                start,
                end,
                hours: 7.0 * 24.0,
            }
        }
        SeasonCycleType::Custom => {
            let year = find_season_year_custom(stage.start_date, stage.end_date, season_def);
            let (start, end) = custom_period_bounds(year, season_def);
            let days = u32::try_from((end - start).num_days())
                .unwrap_or_else(|_| unreachable!("custom period day count always fits in u32"));
            PeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            }
        }
    }
}

/// Resolve the period window immediately following `current`, for forward
/// spillover accounting.
///
/// `Monthly` and `Weekly` derive the next window arithmetically (next
/// calendar month; next 7-day span). `Custom` advances to the next
/// `season_def` in id order, wrapping the season list — `season_map.seasons`
/// is sorted by id, so this is the next entry in the list.
fn next_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    current: &PeriodWindow,
) -> Option<PeriodWindow> {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = current.start.year();
            let (next_year, next_month) = if season_month == 12 {
                (year + 1, 1u32)
            } else {
                (year, season_month + 1)
            };
            let start = current.end;
            let end = month_exclusive_end(next_year, next_month);
            let hours = month_total_hours(next_year, next_month);
            Some(PeriodWindow { start, end, hours })
        }
        SeasonCycleType::Weekly => {
            let start = current.end;
            let end = start + TimeDelta::days(7);
            Some(PeriodWindow {
                start,
                end,
                hours: 7.0 * 24.0,
            })
        }
        SeasonCycleType::Custom => {
            let pos = season_map
                .seasons
                .iter()
                .position(|s| s.id == season_def.id)?;
            let next_def = &season_map.seasons[(pos + 1) % season_map.seasons.len()];
            let probe_end = current.end + TimeDelta::days(1);
            let year = find_season_year_custom(current.end, probe_end, next_def);
            let (start, end) = custom_period_bounds(year, next_def);
            let days = u32::try_from((end - start).num_days())
                .unwrap_or_else(|_| unreachable!("custom period day count always fits in u32"));
            Some(PeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            })
        }
    }
}

/// The calendar year identifying which occurrence of `season_def` `stage`
/// belongs to, disambiguating repeated season ids across years.
///
/// `Weekly` uses the ISO week-numbering year (`iso_week().year()`), not the
/// calendar year of the window start: a week's Monday can fall in the prior
/// December (`from_isoywd_opt(2004, 1, Mon)` is 2003-12-29), so the window
/// start's calendar year would misclassify that week as the prior year's.
fn resolved_year(season_map: &SeasonMap, season_def: &SeasonDefinition, stage: &Stage) -> i32 {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            find_season_year_monthly(stage.start_date, stage.end_date, season_def.month_start)
        }
        SeasonCycleType::Weekly => stage.start_date.iso_week().year(),
        SeasonCycleType::Custom => {
            find_season_year_custom(stage.start_date, stage.end_date, season_def)
        }
    }
}

/// An all-zero, non-finalizing [`StageLagTransition`] — the shared absent-case
/// value for a stage with no season or an unresolvable season.
fn noop_transition() -> StageLagTransition {
    StageLagTransition {
        accumulate_weight: 0.0,
        spillover_weight: 0.0,
        finalize_period: false,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// Compute the [`StageLagTransition`] for a single stage from its resolved
/// `season_def`'s period window — the day-weighted accumulate/spillover/
/// finalize arithmetic generalized across `Monthly`/`Weekly`/`Custom` cycles.
pub(crate) fn compute_period_transition(
    stage: &Stage,
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    all_stages: &[Stage],
) -> StageLagTransition {
    let current = period_window(season_map, season_def, stage);

    let days_current = days_in_period(stage.start_date, stage.end_date, current.start, current.end);
    let accumulate_weight = f64::from(days_current) * 24.0 / current.hours;

    let spillover_weight =
        next_period_window(season_map, season_def, &current).map_or(0.0, |next| {
            let days_next = days_in_period(stage.start_date, stage.end_date, next.start, next.end);
            if days_next > 0 {
                f64::from(days_next) * 24.0 / next.hours
            } else {
                0.0
            }
        });

    let year = resolved_year(season_map, season_def, stage);
    let finalize_period = !all_stages
        .iter()
        .skip(stage.index + 1)
        .filter(|s| s.season_id == Some(season_def.id))
        .any(|s| resolved_year(season_map, season_def, s) == year);

    StageLagTransition {
        accumulate_weight,
        spillover_weight,
        finalize_period,
        accumulate_downstream: false,
        downstream_accumulate_weight: 0.0,
        downstream_spillover_weight: 0.0,
        downstream_finalize: false,
        rebuild_from_downstream: false,
    }
}

/// Precompute one [`StageLagTransition`] per stage from stage date boundaries
/// and season definitions; consumed read-only on the forward-pass hot path.
///
/// A `season_id = None` stage, or any input outside a season, produces a
/// fully zeroed no-op transition.
///
/// # Downstream accumulation
///
/// `downstream_par_order > 0` detects a resolution transition (first stage whose
/// `season_id >= 12`, i.e. crosses from the monthly into the quarterly range) and
/// fills downstream fields for the `downstream_par_order * 3` monthly stages
/// before it. Passing `0` leaves every downstream field at its default — the
/// downstream fields are inert unless populated here.
#[must_use]
pub fn precompute_stage_lag_transitions(
    stages: &[Stage],
    season_map: &SeasonMap,
    downstream_par_order: usize,
) -> Vec<StageLagTransition> {
    let mut result: Vec<StageLagTransition> = stages
        .iter()
        .map(|stage| {
            let Some(season_id) = stage.season_id else {
                return noop_transition();
            };

            let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
                return noop_transition();
            };

            compute_period_transition(stage, season_map, season_def, stages)
        })
        .collect();

    if downstream_par_order > 0 {
        compute_downstream_transitions(stages, &mut result, downstream_par_order);
    }

    result
}

/// Day-weighted split of one stage's inflow across the (at most two) calendar
/// periods it overlaps, for monthly→weekly (Regime A) disaggregation.
///
/// An interior stage (fully inside one period) carries `next_period == None`
/// and `anchor_day_weight == 1.0`, so the forward-pass blend collapses to the anchor
/// period's rate and the monthly instruction stream is byte-for-byte unchanged.
/// A boundary stage spanning periods `a` (anchor) and `b` (next) carries the
/// day-share weights and, when an in-study stage carries the next period's draw,
/// `next_period_stage` — the representative stage the forward peek reads the realized
/// rate from. A boundary at the study's trailing edge (no in-study next-period
/// stage) degrades to interior: no realized next-period rate exists to blend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisaggregationWeight {
    /// Anchor period (the stage's `season_id`, the earliest period it overlaps).
    pub anchor_period: usize,
    /// Next period the stage overlaps; `None` when the stage is interior.
    pub next_period: Option<usize>,
    /// Representative in-study stage index carrying `next_period`'s draw;
    /// `None` when interior or the next period lies past the study horizon.
    pub next_period_stage: Option<usize>,
    /// Day-share of the anchor period; `1.0` when interior.
    pub anchor_day_weight: f64,
    /// Day-share of the next period; `0.0` when interior.
    pub next_day_weight: f64,
}

impl DisaggregationWeight {
    /// The interior (no-op) weight for a stage anchored to `anchor_period`.
    ///
    /// `pub(crate)`: also the empty-slice fallback shape returned by
    /// [`crate::context::StageContext::disaggregation_weight_at`] for call sites
    /// (mostly tests) that pass `disaggregation_weights: &[]`.
    #[must_use]
    pub(crate) fn interior(anchor_period: usize) -> Self {
        Self {
            anchor_period,
            next_period: None,
            next_period_stage: None,
            anchor_day_weight: 1.0,
            next_day_weight: 0.0,
        }
    }
}

/// Precompute one [`DisaggregationWeight`] per stage from stage dates and season
/// definitions; consumed read-only on the forward/backward/simulation hot path.
///
/// Reuses the same period-window enumerator as `compute_period_transition`, so
/// the day-share split is the exact inverse of the lag re-aggregation. No new
/// input: the weights are calendar-derived and the two period rates are the
/// trajectory's own drawn values.
#[must_use]
pub fn precompute_disaggregation_weights(
    stages: &[Stage],
    season_map: &SeasonMap,
) -> Vec<DisaggregationWeight> {
    stages
        .iter()
        .map(|stage| {
            let Some(season_id) = stage.season_id else {
                return DisaggregationWeight::interior(0);
            };
            let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
                return DisaggregationWeight::interior(season_id);
            };

            let current = period_window(season_map, season_def, stage);
            let days_a =
                days_in_period(stage.start_date, stage.end_date, current.start, current.end);
            let Some(next) = next_period_window(season_map, season_def, &current) else {
                return DisaggregationWeight::interior(season_id);
            };
            let days_b = days_in_period(stage.start_date, stage.end_date, next.start, next.end);
            if days_b == 0 {
                return DisaggregationWeight::interior(season_id);
            }

            // The realized next-period rate lives on the first study stage that
            // starts inside `next`; without one (trailing-edge boundary) there is
            // no in-study draw to blend, so degrade to interior.
            let Some(next_period_stage) = stages
                .iter()
                .position(|s| s.start_date >= next.start && s.start_date < next.end)
            else {
                return DisaggregationWeight::interior(season_id);
            };

            let total = f64::from(days_a) + f64::from(days_b);
            DisaggregationWeight {
                anchor_period: season_id,
                next_period: stages[next_period_stage].season_id,
                next_period_stage: Some(next_period_stage),
                anchor_day_weight: f64::from(days_a) / total,
                next_day_weight: f64::from(days_b) / total,
            }
        })
        .collect()
}

/// Project [`DisaggregationWeight`]s into `cobre-stochastic`'s crate-generic
/// [`PeriodBlendWeight`] for the η-inversion call sites
/// (`standardize_external_inflow`, `standardize_historical_windows`) —
/// `cobre-stochastic` cannot depend on `cobre-sddp`, so the two crates share
/// only the numeric fields the inverse needs, never the `DisaggregationWeight`
/// type itself.
#[must_use]
pub fn to_period_blend_weights(weights: &[DisaggregationWeight]) -> Vec<PeriodBlendWeight> {
    weights
        .iter()
        .map(|w| PeriodBlendWeight {
            anchor_day_weight: w.anchor_day_weight,
            next_day_weight: w.next_day_weight,
            next_period_stage: w.next_period_stage,
        })
        .collect()
}

/// Populate downstream accumulation fields on the pre-transition window entries
/// in `transitions`.
///
/// The transition is the first stage whose `season_id >= 12` (quarterly range);
/// the window is the `downstream_par_order * 3` monthly stages before it. Weights
/// use quarterly calendar boundaries (months 1–3 → Q1, 4–6 → Q2, 7–9 → Q3,
/// 10–12 → Q4); `downstream_finalize` is set on the last monthly stage of each
/// calendar quarter within the window. No transition / empty window leaves
/// `transitions` unchanged.
fn compute_downstream_transitions(
    stages: &[Stage],
    transitions: &mut [StageLagTransition],
    downstream_par_order: usize,
) {
    let Some(transition_idx) = stages
        .iter()
        .position(|s| s.season_id.is_some_and(|id| id >= 12))
    else {
        return;
    };

    let window_len = downstream_par_order * 3;
    let window_start = transition_idx.saturating_sub(window_len);

    for stage_idx in window_start..transition_idx {
        let stage = &stages[stage_idx];
        let Some(season_id) = stage.season_id else {
            continue;
        };

        // season_id is 0-based (0=Jan … 11=Dec); + 1 makes a 1-based calendar month.
        let month = u32::try_from(season_id % 12 + 1)
            .unwrap_or_else(|_| unreachable!("season_id % 12 always fits in u32"));

        let quarter_start_month: u32 = ((month - 1) / 3) * 3 + 1; // 1, 4, 7, or 10
        let quarter_end_month: u32 = quarter_start_month + 2;

        let year = find_season_year_monthly(stage.start_date, stage.end_date, month);

        let quarter_total_hours: f64 = (quarter_start_month..=quarter_end_month)
            .map(|m| {
                let (y, mo) = if m > 12 {
                    (year + 1, m - 12)
                } else {
                    (year, m)
                };
                month_total_hours(y, mo)
            })
            .sum();

        let quarter_period_start = NaiveDate::from_ymd_opt(year, quarter_start_month, 1)
            .unwrap_or_else(|| unreachable!("quarter start date is always valid"));
        let last_quarter_month_end = month_exclusive_end(year, quarter_end_month);

        let days_current = days_in_period(
            stage.start_date,
            stage.end_date,
            quarter_period_start,
            last_quarter_month_end,
        );
        let downstream_accumulate_weight = f64::from(days_current) * 24.0 / quarter_total_hours;

        let next_quarter_start_month = quarter_end_month + 1; // may be 13 → wrap to next year
        let (next_q_year, next_q_start_month) = if next_quarter_start_month > 12 {
            (year + 1, next_quarter_start_month - 12)
        } else {
            (year, next_quarter_start_month)
        };
        let next_quarter_end_month = next_q_start_month + 2;
        let next_quarter_start = NaiveDate::from_ymd_opt(next_q_year, next_q_start_month, 1)
            .unwrap_or_else(|| unreachable!("next quarter start date is always valid"));
        let (next_q_end_year, next_q_end_month_adj) = if next_quarter_end_month > 12 {
            (next_q_year + 1, next_quarter_end_month - 12)
        } else {
            (next_q_year, next_quarter_end_month)
        };
        let next_quarter_end = month_exclusive_end(next_q_end_year, next_q_end_month_adj);
        let next_quarter_total_hours: f64 = (next_q_start_month..=next_quarter_end_month)
            .map(|m| {
                let (y, mo) = if m > 12 {
                    (next_q_year + 1, m - 12)
                } else {
                    (next_q_year, m)
                };
                month_total_hours(y, mo)
            })
            .sum();
        let days_next = days_in_period(
            stage.start_date,
            stage.end_date,
            next_quarter_start,
            next_quarter_end,
        );
        let downstream_spillover_weight = if days_next > 0 {
            f64::from(days_next) * 24.0 / next_quarter_total_hours
        } else {
            0.0
        };

        let is_last_of_quarter = stages[stage_idx + 1..transition_idx].iter().all(|later| {
            let later_month = later.season_id.map_or(u32::MAX, |id| {
                u32::try_from(id % 12 + 1).unwrap_or(u32::MAX)
            });
            let later_quarter_start = ((later_month.saturating_sub(1)) / 3) * 3 + 1;
            later_quarter_start != quarter_start_month
        });

        transitions[stage_idx].accumulate_downstream = true;
        transitions[stage_idx].downstream_accumulate_weight = downstream_accumulate_weight;
        transitions[stage_idx].downstream_spillover_weight = downstream_spillover_weight;
        transitions[stage_idx].downstream_finalize = is_last_of_quarter;
    }

    // rebuild_from_downstream: at the transition the primary lag state is
    // discarded and rebuilt from the completed quarterly lags in the downstream
    // ring buffer.
    if transition_idx < transitions.len() {
        transitions[transition_idx].rebuild_from_downstream = true;
    }
}

/// Precompute a noise group ID for each study stage, so the forward sampler can
/// draw one noise sample per group and broadcast it (weekly stages sharing
/// monthly PAR noise).
///
/// Stages with `season_id = Some(id)` group by `(id, start_date.year())`,
/// consecutive IDs from 0 in stage-index order of first occurrence; a
/// `season_id = None` stage each receives its own unique ID (no sharing). For a
/// uniform monthly study the result is `[0, 1, …, n-1]`.
#[must_use]
pub fn precompute_noise_groups(stages: &[Stage]) -> Vec<u32> {
    let mut group_map: HashMap<(usize, i32), u32> = HashMap::new();
    let mut next_group_id: u32 = 0;
    let mut result = Vec::with_capacity(stages.len());
    for stage in stages {
        if let Some(season_id) = stage.season_id {
            let key = (season_id, stage.start_date.year());
            let gid = *group_map.entry(key).or_insert_with(|| {
                let id = next_group_id;
                next_group_id += 1;
                id
            });
            result.push(gid);
        } else {
            result.push(next_group_id);
            next_group_id += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
        SeasonMap, Stage, StageRiskConfig, StageStateConfig,
    };

    fn monthly_season_map() -> SeasonMap {
        let seasons: Vec<SeasonDefinition> = (0..12u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Month{}", i + 1),
                month_start: i + 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        }
    }

    fn make_stage(
        index: usize,
        start: NaiveDate,
        end: NaiveDate,
        season_id: Option<usize>,
    ) -> Stage {
        let days = u32::try_from((end - start).num_days()).unwrap();
        Stage {
            index,
            id: i32::try_from(index).unwrap(),
            start_date: start,
            end_date: end,
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: f64::from(days) * 24.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn test_disaggregation_interior_week_uses_month_rate() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 4, 4), d(2026, 4, 11), Some(3)),
            make_stage(1, d(2026, 4, 11), d(2026, 4, 18), Some(3)),
            make_stage(2, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
        ];

        let weights = precompute_disaggregation_weights(&stages, &season_map);
        let interior = &weights[1];

        assert_eq!(interior.next_period, None);
        assert_eq!(interior.next_period_stage, None);
        assert_eq!(interior.anchor_period, 3);
        assert!((interior.anchor_day_weight - 1.0).abs() < 1e-12);
        assert!(interior.next_day_weight.abs() < 1e-12);
    }

    #[test]
    fn test_disaggregation_boundary_week_day_weighted_blend() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
            make_stage(1, d(2026, 4, 25), d(2026, 5, 2), Some(3)),
            make_stage(2, d(2026, 5, 2), d(2026, 6, 1), Some(4)),
        ];

        // Hand-derive the day counts from the public enumerator, never the struct.
        let april = season_map.seasons.iter().find(|s| s.id == 3).unwrap();
        let current = period_window(&season_map, april, &stages[1]);
        let next = next_period_window(&season_map, april, &current).unwrap();
        let days_a = days_in_period(
            stages[1].start_date,
            stages[1].end_date,
            current.start,
            current.end,
        );
        let days_b = days_in_period(
            stages[1].start_date,
            stages[1].end_date,
            next.start,
            next.end,
        );
        assert_eq!((days_a, days_b), (6, 1));
        let total = f64::from(days_a + days_b);

        let weights = precompute_disaggregation_weights(&stages, &season_map);
        let boundary = &weights[1];

        assert_eq!(boundary.anchor_period, 3);
        assert_eq!(boundary.next_period, Some(4));
        assert_eq!(boundary.next_period_stage, Some(2));
        assert!((boundary.anchor_day_weight - f64::from(days_a) / total).abs() < 1e-12);
        assert!((boundary.next_day_weight - f64::from(days_b) / total).abs() < 1e-12);
        // Closed-form cross-check.
        assert!((boundary.anchor_day_weight - 6.0 / 7.0).abs() < 1e-12);
        assert!((boundary.next_day_weight - 1.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_monthly_identity() {
        let season_map = monthly_season_map();
        let stages: Vec<Stage> = (0..12usize)
            .map(|i| {
                let month = u32::try_from(i + 1).unwrap();
                let start = d(2026, month, 1);
                let (ny, nm) = if month == 12 {
                    (2027, 1u32)
                } else {
                    (2026, month + 1)
                };
                let end = d(ny, nm, 1);
                make_stage(i, start, end, Some(i))
            })
            .collect();

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);

        assert_eq!(transitions.len(), 12);
        for (i, t) in transitions.iter().enumerate() {
            assert!(
                (t.accumulate_weight - 1.0).abs() < 1e-10,
                "stage {i}: accumulate_weight expected 1.0, got {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < 1e-10,
                "stage {i}: spillover_weight expected 0.0, got {}",
                t.spillover_weight
            );
            assert!(
                t.finalize_period,
                "stage {i}: finalize_period expected true"
            );
        }
    }

    /// Six-stage mixed weekly+monthly layout from the design doc.
    ///
    /// Stage dates use exclusive-end (`[start, end)`) convention:
    /// - W1: `[2026-03-28, 2026-04-04)` — 3 April days (pre-study March days excluded)
    /// - W2: `[2026-04-04, 2026-04-11)` — 7 April days
    /// - W3: `[2026-04-11, 2026-04-18)` — 7 April days
    /// - W4: `[2026-04-18, 2026-04-25)` — 7 April days
    /// - W5: `[2026-04-25, 2026-05-02)` — 6 April days + 1 May day (spillover)
    /// - M2: `[2026-05-02, 2026-06-01)` — 30 May days
    ///
    /// April = 720 h; May = 744 h.
    #[test]
    fn test_pmo_apr_2026_rv0_trace() {
        let season_map = monthly_season_map();

        let stages = vec![
            make_stage(0, d(2026, 3, 28), d(2026, 4, 4), Some(3)),
            make_stage(1, d(2026, 4, 4), d(2026, 4, 11), Some(3)),
            make_stage(2, d(2026, 4, 11), d(2026, 4, 18), Some(3)),
            make_stage(3, d(2026, 4, 18), d(2026, 4, 25), Some(3)),
            make_stage(4, d(2026, 4, 25), d(2026, 5, 2), Some(3)),
            make_stage(5, d(2026, 5, 2), d(2026, 6, 1), Some(4)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 6);

        let april_hours = 30.0 * 24.0;
        let may_hours = 31.0 * 24.0;
        let tol = 1e-6;

        let w1 = transitions[0];
        assert!(
            (w1.accumulate_weight - 3.0 * 24.0 / april_hours).abs() < tol,
            "W1 accumulate_weight: expected {}, got {}",
            3.0 * 24.0 / april_hours,
            w1.accumulate_weight
        );
        assert!(
            w1.spillover_weight.abs() < tol,
            "W1 spillover_weight must be 0"
        );
        assert!(!w1.finalize_period, "W1 must not finalize");

        let w2 = transitions[1];
        assert!(
            (w2.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W2 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w2.accumulate_weight
        );
        assert!(
            w2.spillover_weight.abs() < tol,
            "W2 spillover_weight must be 0"
        );
        assert!(!w2.finalize_period, "W2 must not finalize");

        let w3 = transitions[2];
        assert!(
            (w3.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W3 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w3.accumulate_weight
        );
        assert!(
            w3.spillover_weight.abs() < tol,
            "W3 spillover_weight must be 0"
        );
        assert!(!w3.finalize_period, "W3 must not finalize");

        let w4 = transitions[3];
        assert!(
            (w4.accumulate_weight - 7.0 * 24.0 / april_hours).abs() < tol,
            "W4 accumulate_weight: expected {}, got {}",
            7.0 * 24.0 / april_hours,
            w4.accumulate_weight
        );
        assert!(
            w4.spillover_weight.abs() < tol,
            "W4 spillover_weight must be 0"
        );
        assert!(!w4.finalize_period, "W4 must not finalize");

        let w5 = transitions[4];
        assert!(
            (w5.accumulate_weight - 6.0 * 24.0 / april_hours).abs() < tol,
            "W5 accumulate_weight: expected {}, got {}",
            6.0 * 24.0 / april_hours,
            w5.accumulate_weight
        );
        assert!(
            (w5.spillover_weight - 1.0 * 24.0 / may_hours).abs() < tol,
            "W5 spillover_weight: expected {}, got {}",
            1.0 * 24.0 / may_hours,
            w5.spillover_weight
        );
        assert!(w5.finalize_period, "W5 must finalize");

        let m2 = transitions[5];
        assert!(
            (m2.accumulate_weight - 30.0 * 24.0 / may_hours).abs() < tol,
            "M2 accumulate_weight: expected {}, got {}",
            30.0 * 24.0 / may_hours,
            m2.accumulate_weight
        );
        assert!(
            m2.spillover_weight.abs() < tol,
            "M2 spillover_weight must be 0"
        );
        assert!(m2.finalize_period, "M2 must finalize");
    }

    // -----------------------------------------------------------------------
    // Test 3: single stage straddling a month boundary
    // -----------------------------------------------------------------------

    /// Stage `[2026-01-28, 2026-02-04)` with `season_id=0` (January).
    ///
    /// "Jan 28 to Feb 3" in inclusive notation equals `[Jan 28, Feb 04)` in
    /// Cobre exclusive-end convention.  That gives 4 January days (28–31) and
    /// 3 February days (01–03).
    ///
    /// January 2026: 31 days = 744 h.
    /// February 2026: 28 days = 672 h (not a leap year).
    #[test]
    fn test_boundary_straddling_week() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 1, 28), d(2026, 2, 4), Some(0));
        let stages = vec![stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        let jan_hours = 31.0 * 24.0;
        let feb_hours = 28.0 * 24.0;
        let tol = 1e-10;

        assert!(
            (t.accumulate_weight - 4.0 * 24.0 / jan_hours).abs() < tol,
            "accumulate_weight: expected {}, got {}",
            4.0 * 24.0 / jan_hours,
            t.accumulate_weight
        );
        assert!(
            (t.spillover_weight - 3.0 * 24.0 / feb_hours).abs() < tol,
            "spillover_weight: expected {}, got {}",
            3.0 * 24.0 / feb_hours,
            t.spillover_weight
        );
        assert!(t.finalize_period, "single stage must finalize its period");
    }

    // -----------------------------------------------------------------------
    // Test 4: stage with season_id = None produces no-op
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_season_id_produces_noop() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 1, 1), d(2026, 2, 1), None);
        let stages = vec![stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        assert_eq!(t.accumulate_weight, 0.0);
        assert_eq!(t.spillover_weight, 0.0);
        assert!(!t.finalize_period);
    }

    // -----------------------------------------------------------------------
    // Test 5: two consecutive monthly stages each finalise their own period
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_stage_per_month_finalizes() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 2);
        assert!(
            transitions[0].finalize_period,
            "January stage must finalize"
        );
        assert!(
            transitions[1].finalize_period,
            "February stage must finalize"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: four weekly stages in January — only the last finalises
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_weekly_stages_only_last_finalizes() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 1, 8), Some(0)),
            make_stage(1, d(2026, 1, 8), d(2026, 1, 15), Some(0)),
            make_stage(2, d(2026, 1, 15), d(2026, 1, 22), Some(0)),
            make_stage(3, d(2026, 1, 22), d(2026, 1, 29), Some(0)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 4);

        let jan_hours = 31.0 * 24.0;
        let tol = 1e-10;

        for (i, t) in transitions.iter().enumerate().take(3) {
            assert!(
                !t.finalize_period,
                "stage {i}: finalize_period must be false"
            );
            assert!(
                (t.accumulate_weight - 7.0 * 24.0 / jan_hours).abs() < tol,
                "stage {i}: accumulate_weight wrong: {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < tol,
                "stage {i}: spillover_weight must be 0"
            );
        }

        let w4 = transitions[3];
        assert!(w4.finalize_period, "W4 must be the finalising stage");
        assert!(
            (w4.accumulate_weight - 7.0 * 24.0 / jan_hours).abs() < tol,
            "W4 accumulate_weight wrong: {}",
            w4.accumulate_weight
        );
    }

    // -----------------------------------------------------------------------
    // Weekly / Custom period-window generalization (compute_period_transition)
    // -----------------------------------------------------------------------

    fn weekly_season_map() -> SeasonMap {
        let seasons: Vec<SeasonDefinition> = (0..52u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Week{}", i + 1),
                month_start: 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons,
        }
    }

    /// Four consecutive real ISO weeks of January 2024 (2024-01-01 is a Monday,
    /// ISO week 1), each stage exactly spanning its own week.
    #[test]
    fn test_weekly_cycle_per_week_finalizes() {
        let season_map = weekly_season_map();
        let stages = vec![
            make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0)),
            make_stage(1, d(2024, 1, 8), d(2024, 1, 15), Some(1)),
            make_stage(2, d(2024, 1, 15), d(2024, 1, 22), Some(2)),
            make_stage(3, d(2024, 1, 22), d(2024, 1, 29), Some(3)),
        ];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 4);

        let tol = 1e-10;
        for (i, t) in transitions.iter().enumerate() {
            assert!(
                (t.accumulate_weight - 1.0).abs() < tol,
                "stage {i}: accumulate_weight expected 1.0, got {}",
                t.accumulate_weight
            );
            assert!(
                t.spillover_weight.abs() < tol,
                "stage {i}: spillover_weight expected 0.0, got {}",
                t.spillover_weight
            );
            assert!(
                t.finalize_period,
                "stage {i}: finalize_period expected true (weekly PAR finalizes every stage)"
            );
        }
    }

    /// ISO week 53 of 2026 spans `[2026-12-28, 2027-01-04)`; `season_for_date`
    /// folds it to season id 51 (the same id as ISO week 52), but the physical
    /// window stays the real 7-day week-53 span.
    #[test]
    fn test_weekly_iso_week_53_folds() {
        let season_map = weekly_season_map();
        assert_eq!(season_map.season_for_date(d(2026, 12, 28)), Some(51));

        let stage = make_stage(0, d(2026, 12, 28), d(2027, 1, 4), Some(51));
        let transitions = precompute_stage_lag_transitions(&[stage], &season_map, 0);
        assert_eq!(transitions.len(), 1);

        let t = transitions[0];
        let tol = 1e-10;
        assert!(
            (t.accumulate_weight - 1.0).abs() < tol,
            "accumulate_weight expected 1.0 (the full physical week-53 span), got {}",
            t.accumulate_weight
        );
        assert!(
            t.spillover_weight.abs() < tol,
            "spillover_weight expected 0.0, got {}",
            t.spillover_weight
        );
        assert!(t.finalize_period, "single stage must finalize its period");
    }

    /// d30-style multi-resolution `Custom` map: a monthly definition (June) and
    /// a quarterly definition (Q3) in the SAME `season_map`. Each stage must
    /// weight against its OWN level's period hours, never a flattened cycle.
    fn custom_multi_resolution_season_map() -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                SeasonDefinition {
                    id: 5,
                    label: "June".to_string(),
                    month_start: 6,
                    day_start: Some(1),
                    month_end: Some(6),
                    day_end: Some(30),
                },
                SeasonDefinition {
                    id: 12,
                    label: "Q3".to_string(),
                    month_start: 7,
                    day_start: Some(1),
                    month_end: Some(9),
                    day_end: Some(30),
                },
            ],
        }
    }

    #[test]
    fn test_custom_multi_resolution_each_stage_in_own_level() {
        let season_map = custom_multi_resolution_season_map();

        // June stage spans its whole month (30 days); the Q3 stage spans only
        // the first 30 days of the 92-day quarter — a sub-window, so its
        // weight can only match if it used the QUARTER's hours, not July's.
        let june_stage = make_stage(0, d(2024, 6, 1), d(2024, 7, 1), Some(5));
        let q3_partial_stage = make_stage(1, d(2024, 7, 1), d(2024, 7, 31), Some(12));
        let stages = vec![june_stage, q3_partial_stage];

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert_eq!(transitions.len(), 2);

        let tol = 1e-10;

        let june_hours = 30.0 * 24.0;
        let expected_june_weight = 30.0 * 24.0 / june_hours;
        assert!(
            (transitions[0].accumulate_weight - expected_june_weight).abs() < tol,
            "June stage must weight against its own month's hours ({june_hours}): expected {expected_june_weight}, got {}",
            transitions[0].accumulate_weight
        );
        assert!(transitions[0].finalize_period, "June stage must finalize");

        let q3_hours = 92.0 * 24.0;
        let expected_q3_weight = 30.0 * 24.0 / q3_hours;
        assert!(
            (transitions[1].accumulate_weight - expected_q3_weight).abs() < tol,
            "Q3 stage must weight against its own quarter's hours ({q3_hours}): expected {expected_q3_weight}, got {}",
            transitions[1].accumulate_weight
        );
        assert!(transitions[1].finalize_period, "Q3 stage must finalize");

        let flattened_to_july_weight = 30.0 * 24.0 / (31.0 * 24.0);
        assert!(
            (transitions[1].accumulate_weight - flattened_to_july_weight).abs() > 1e-3,
            "Q3 stage must not collapse to a monthly-level weight"
        );
    }

    /// d30-shaped multi-resolution disaggregation: a fine-level (monthly,
    /// June) boundary stage — the study's last monthly-resolution stage before
    /// the decomposition transitions to coarse (quarterly, Q3) resolution —
    /// must source its next-period rate from the COARSE stage directly above
    /// it (the level the decomposition switches to), never from a flattened
    /// global cycle. Reuses `custom_multi_resolution_season_map` (June + Q3)
    /// verbatim — no new season map shape.
    #[test]
    fn test_disaggregation_multi_resolution_sources_from_level_above() {
        let season_map = custom_multi_resolution_season_map();

        // June (fine) overlaps 5 days into its own month plus 2 days into Q3
        // (coarse); Q3 (the study's next stage, at coarse resolution) starts
        // where June's overlap ends.
        let june_boundary_stage = make_stage(0, d(2024, 6, 26), d(2024, 7, 3), Some(5));
        let q3_stage = make_stage(1, d(2024, 7, 3), d(2024, 10, 1), Some(12));
        let stages = vec![june_boundary_stage, q3_stage];

        // Hand-derive the day counts from the public enumerator, never the struct.
        let june_def = season_map.seasons.iter().find(|s| s.id == 5).unwrap();
        let current = period_window(&season_map, june_def, &stages[0]);
        let next = next_period_window(&season_map, june_def, &current).unwrap();
        let days_a = days_in_period(
            stages[0].start_date,
            stages[0].end_date,
            current.start,
            current.end,
        );
        let days_b = days_in_period(
            stages[0].start_date,
            stages[0].end_date,
            next.start,
            next.end,
        );
        assert_eq!((days_a, days_b), (5, 2));
        let total = f64::from(days_a + days_b);

        let weights = precompute_disaggregation_weights(&stages, &season_map);
        let boundary = &weights[0];

        assert_eq!(
            boundary.anchor_period, 5,
            "anchor must be June (fine level)"
        );
        assert_eq!(
            boundary.next_period,
            Some(12),
            "next period must be Q3 (the coarse level directly above), never a \
             flattened global cycle"
        );
        assert_eq!(
            boundary.next_period_stage,
            Some(1),
            "representative stage must be the coarse Q3 stage"
        );
        assert!((boundary.anchor_day_weight - f64::from(days_a) / total).abs() < 1e-12);
        assert!((boundary.next_day_weight - f64::from(days_b) / total).abs() < 1e-12);
        // Closed-form cross-check.
        assert!((boundary.anchor_day_weight - 5.0 / 7.0).abs() < 1e-12);
        assert!((boundary.next_day_weight - 2.0 / 7.0).abs() < 1e-12);

        // The coarse Q3 stage itself is interior to Q3 — no further boundary.
        assert_eq!(weights[1].next_period, None);
        assert!((weights[1].anchor_day_weight - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Tests for compute_recent_observation_seed
    // -----------------------------------------------------------------------

    use cobre_core::{
        EntityId,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        initial_conditions::RecentObservation,
    };

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        }
    }

    fn make_observation(
        hydro_id: i32,
        y: i32,
        m1: u32,
        d1: u32,
        m2: u32,
        d2: u32,
        val: f64,
    ) -> RecentObservation {
        RecentObservation {
            hydro_id: EntityId(hydro_id),
            start_date: d(y, m1, d1),
            end_date: d(y, m2, d2),
            value_m3s: val,
        }
    }

    // April 2026: 30 days = 720 h.
    const APRIL_2026_HOURS: f64 = 720.0;

    /// Test 7: empty `recent_observations` — zero seed.
    #[test]
    fn test_seed_empty_observations_returns_zero() {
        let season_map = monthly_season_map();
        // First study stage: April 4 → May 2 (season_id = 3 → April).
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];

        let seed = compute_recent_observation_seed(&[], &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed.len(), 1);
        assert_eq!(seed.accum_seed[0], 0.0);
        assert_eq!(seed.weight_seed, 0.0);
    }

    /// Test 8: one observation for one hydro, 3 days (April 1–4) at 500.0 m3/s.
    ///
    /// Expected: `accum_seed[0] == 500.0 * 72.0`, `weight_seed == 72.0 / 720.0`.
    #[test]
    fn test_seed_one_observation_one_hydro() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        let obs = vec![make_observation(0, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let expected_accum = 500.0 * 72.0;
        let expected_weight = 72.0 / APRIL_2026_HOURS;
        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - expected_accum).abs() < tol,
            "accum_seed[0]: expected {expected_accum}, got {}",
            seed.accum_seed[0]
        );
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 9: two observations for the same hydro (rv2 pattern: Apr 1–4 at 500.0 and
    /// Apr 4–11 at 480.0) → additive accumulation.
    ///
    /// `accum_seed[0] == 500.0 * 72.0 + 480.0 * 168.0`
    /// `weight_seed == (72.0 + 168.0) / 720.0`
    #[test]
    fn test_seed_two_observations_same_hydro_additive() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 11), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 500.0),
            make_observation(0, 2026, 4, 4, 4, 11, 480.0),
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let expected_accum = 500.0 * 72.0 + 480.0 * 168.0;
        let expected_weight = (72.0 + 168.0) / APRIL_2026_HOURS;
        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - expected_accum).abs() < tol,
            "accum_seed[0]: expected {expected_accum}, got {}",
            seed.accum_seed[0]
        );
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 10: observations for two different hydros → each slot is independent.
    #[test]
    fn test_seed_two_observations_different_hydros_independent() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0), make_hydro(1)];
        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 500.0), // hydro 0: 3 days
            make_observation(1, 2026, 4, 1, 4, 4, 300.0), // hydro 1: 3 days
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - 500.0 * 72.0).abs() < tol,
            "accum_seed[0]: expected {}, got {}",
            500.0 * 72.0,
            seed.accum_seed[0]
        );
        assert!(
            (seed.accum_seed[1] - 300.0 * 72.0).abs() < tol,
            "accum_seed[1]: expected {}, got {}",
            300.0 * 72.0,
            seed.accum_seed[1]
        );
        // Both hydros observe the same 3-day (72 h) calendar window, so the
        // weight must reflect that single window's coverage — not doubled by
        // hydro count. The correct weight is max(72, 72) / total_period_hours.
        let expected_weight = 72.0 / APRIL_2026_HOURS;
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight}, got {}",
            seed.weight_seed
        );
    }

    /// Test 10b: regression — weight must not scale with hydro count.
    ///
    /// Four hydros each provide a 72-hour observation in a 720-hour (April)
    /// stage. The correct weight is 72/720 = 0.10, not 4*72/720 = 0.40.
    #[test]
    fn test_seed_weight_independent_of_hydro_count() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0), make_hydro(1), make_hydro(2), make_hydro(3)];
        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 100.0), // hydro 0: 3 days = 72 h
            make_observation(1, 2026, 4, 1, 4, 4, 200.0), // hydro 1: 3 days = 72 h
            make_observation(2, 2026, 4, 1, 4, 4, 300.0), // hydro 2: 3 days = 72 h
            make_observation(3, 2026, 4, 1, 4, 4, 400.0), // hydro 3: 3 days = 72 h
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let tol = 1e-10;
        // Each hydro's accumulator is independent.
        assert!((seed.accum_seed[0] - 100.0 * 72.0).abs() < tol, "accum[0]");
        assert!((seed.accum_seed[1] - 200.0 * 72.0).abs() < tol, "accum[1]");
        assert!((seed.accum_seed[2] - 300.0 * 72.0).abs() < tol, "accum[2]");
        assert!((seed.accum_seed[3] - 400.0 * 72.0).abs() < tol, "accum[3]");
        // Weight must equal 72/720, not 4*72/720.
        let expected_weight = 72.0 / APRIL_2026_HOURS;
        assert!(
            (seed.weight_seed - expected_weight).abs() < tol,
            "weight_seed: expected {expected_weight} (= 72/720), got {} (= {}*72/720 would be the buggy value)",
            seed.weight_seed,
            hydros.len(),
        );
    }

    /// Test 11: observation for unknown `hydro_id` — silently skipped, zero seed.
    #[test]
    fn test_seed_unknown_hydro_id_silently_skipped() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(0)];
        let obs = vec![make_observation(99, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed.len(), 1);
        assert_eq!(seed.accum_seed[0], 0.0, "unknown hydro_id must be skipped");
        assert_eq!(
            seed.weight_seed, 0.0,
            "weight must be 0 when all hydros unknown"
        );
    }

    /// Test 12: first stage has `season_id` = None — zero seed returned.
    #[test]
    fn test_seed_no_season_id_returns_zero() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), None);
        let hydros = vec![make_hydro(0)];
        let obs = vec![make_observation(0, 2026, 4, 1, 4, 4, 500.0)];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        assert_eq!(seed.accum_seed[0], 0.0);
        assert_eq!(seed.weight_seed, 0.0);
    }

    /// Test 13 (regression): `hydros` in canonical `(operational_start_date,
    /// id)` order can be id-DESCENDING — here hydro id=1's earlier
    /// commissioning date sorts it before hydro id=0. Each observation must
    /// still land in its OWN hydro's accumulator slot, resolved through an
    /// id->position map rather than `binary_search_by_key` over `hydros`
    /// (which requires id-ascending order and silently drops the id=1
    /// observation under this staggered ordering).
    #[test]
    fn test_seed_correct_under_staggered_commissioning_dates() {
        let season_map = monthly_season_map();
        let stage = make_stage(0, d(2026, 4, 4), d(2026, 5, 2), Some(3));

        let mut hydro_1_earlier = make_hydro(1);
        hydro_1_earlier.operational_start_date = d(2024, 1, 1);
        let mut hydro_0_later = make_hydro(0);
        hydro_0_later.operational_start_date = d(2025, 6, 1);
        // Canonical order: hydro id=1 (earlier date) at position 0, hydro
        // id=0 (later date) at position 1 — id-descending, not id-ascending.
        let hydros = vec![hydro_1_earlier, hydro_0_later];

        let obs = vec![
            make_observation(0, 2026, 4, 1, 4, 4, 500.0),
            make_observation(1, 2026, 4, 1, 4, 4, 300.0),
        ];

        let seed = compute_recent_observation_seed(&obs, &stage, &season_map, &hydros);

        let tol = 1e-10;
        assert!(
            (seed.accum_seed[0] - 300.0 * 72.0).abs() < tol,
            "hydro id=1 (canonical position 0) accum_seed should be {}, got {}",
            300.0 * 72.0,
            seed.accum_seed[0]
        );
        assert!(
            (seed.accum_seed[1] - 500.0 * 72.0).abs() < tol,
            "hydro id=0 (canonical position 1) accum_seed should be {}, got {}",
            500.0 * 72.0,
            seed.accum_seed[1]
        );
    }

    #[test]
    fn test_noise_groups_monthly_unique() {
        let stages: Vec<Stage> = (0..12usize)
            .map(|i| {
                let month = u32::try_from(i + 1).unwrap();
                let start = d(2024, month, 1);
                let (ny, nm) = if month == 12 {
                    (2025, 1u32)
                } else {
                    (2024, month + 1)
                };
                let end = d(ny, nm, 1);
                make_stage(i, start, end, Some(i))
            })
            .collect();

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 12);
        let expected: Vec<u32> = (0..12u32).collect();
        assert_eq!(groups, expected);
    }

    #[test]
    fn test_noise_groups_weekly_shared() {
        let stages_s0: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 1, day_start);
                let end = d(2024, 1, day_end);
                make_stage(i, start, end, Some(0))
            })
            .collect();
        let stages_s1: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 2, day_start);
                let end = d(2024, 2, day_end);
                make_stage(i + 4, start, end, Some(1))
            })
            .collect();

        let mut all_stages = stages_s0;
        all_stages.extend(stages_s1);

        let groups = precompute_noise_groups(&all_stages);

        assert_eq!(groups.len(), 8);
        assert!(groups[0..4].iter().all(|&g| g == 0));
        assert!(groups[4..8].iter().all(|&g| g == 1));
    }

    #[test]
    fn test_noise_groups_mixed_weekly_monthly() {
        let weekly: Vec<Stage> = (0..4usize)
            .map(|i| {
                let day_start = u32::try_from(i * 7 + 1).unwrap();
                let day_end = u32::try_from(i * 7 + 8).unwrap();
                let start = d(2024, 1, day_start);
                let end = d(2024, 1, day_end);
                make_stage(i, start, end, Some(0))
            })
            .collect();
        let monthly = make_stage(4, d(2024, 1, 1), d(2024, 2, 1), Some(0));

        let mut stages = weekly;
        stages.push(monthly);

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 5);
        assert!(
            groups.iter().all(|&g| g == 0),
            "all stages must share group 0"
        );
    }

    #[test]
    fn test_noise_groups_none_season_id() {
        let stages: Vec<Stage> = (0..3usize)
            .map(|i| {
                let start = d(2024, 1, u32::try_from(i + 1).unwrap());
                let end = d(2024, 1, u32::try_from(i + 2).unwrap());
                make_stage(i, start, end, None)
            })
            .collect();

        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], 0);
        assert_eq!(groups[1], 1);
        assert_eq!(groups[2], 2);
    }

    /// Test 5: same `season_id` but different years must produce different groups.
    #[test]
    fn test_noise_groups_cross_year() {
        // Two weekly stages: season_id=0, year 2024 and year 2025.
        let stage_2024 = make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0));
        let stage_2025 = make_stage(1, d(2025, 1, 1), d(2025, 1, 8), Some(0));

        let stages = vec![stage_2024, stage_2025];
        let groups = precompute_noise_groups(&stages);

        assert_eq!(groups.len(), 2);
        assert_ne!(
            groups[0], groups[1],
            "different years must yield different groups"
        );
        assert_eq!(groups[0], 0);
        assert_eq!(groups[1], 1);
    }
}
