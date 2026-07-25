//! Cycle-generic season-period-window arithmetic (`Monthly`/`Weekly`/`Custom`).
//!
//! Resolves the concrete calendar `[start, end)` window and total duration for
//! a stage's season-period occurrence, and the next chronological occurrence.

use chrono::{Datelike, NaiveDate, TimeDelta, Weekday};
use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap, Stage};

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
    f64::from(days_in_month(year, month)) * 24.0
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

/// Concrete `[start, end)` calendar window for one occurrence of a season
/// period, plus its total duration in hours.
pub struct SeasonPeriodWindow {
    /// Inclusive window start.
    pub start: NaiveDate,
    /// Exclusive window end.
    pub end: NaiveDate,
    /// Total window duration in hours.
    pub hours: f64,
}

/// Number of real calendar days in `year`-`month` (1–12).
pub(crate) fn days_in_month(year: i32, month: u32) -> u32 {
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
pub(crate) fn custom_period_bounds(
    year: i32,
    season_def: &SeasonDefinition,
) -> (NaiveDate, NaiveDate) {
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
pub(crate) fn find_season_year_custom(
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
#[must_use]
pub fn season_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    stage: &Stage,
) -> SeasonPeriodWindow {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = find_season_year_monthly(stage.start_date, stage.end_date, season_month);
            let start = NaiveDate::from_ymd_opt(year, season_month, 1)
                .unwrap_or_else(|| unreachable!("season month is always valid"));
            let end = month_exclusive_end(year, season_month);
            let hours = month_total_hours(year, season_month);
            SeasonPeriodWindow { start, end, hours }
        }
        SeasonCycleType::Weekly => {
            let iso_week = stage.start_date.iso_week();
            let start = NaiveDate::from_isoywd_opt(iso_week.year(), iso_week.week(), Weekday::Mon)
                .unwrap_or_else(|| unreachable!("iso week start date is always valid"));
            let end = start + TimeDelta::days(7);
            SeasonPeriodWindow {
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
            SeasonPeriodWindow {
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
#[must_use]
pub fn next_season_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    current: &SeasonPeriodWindow,
) -> Option<SeasonPeriodWindow> {
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
            Some(SeasonPeriodWindow { start, end, hours })
        }
        SeasonCycleType::Weekly => {
            let start = current.end;
            let end = start + TimeDelta::days(7);
            Some(SeasonPeriodWindow {
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
            Some(SeasonPeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            })
        }
    }
}

/// Resolve the period window immediately preceding `current`, the exact
/// inverse of [`next_season_period_window`].
#[must_use]
pub fn previous_season_period_window(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    current: &SeasonPeriodWindow,
) -> Option<SeasonPeriodWindow> {
    match season_map.cycle_type {
        SeasonCycleType::Monthly => {
            let season_month = season_def.month_start;
            let year = current.start.year();
            let (prev_year, prev_month) = if season_month == 1 {
                (year - 1, 12u32)
            } else {
                (year, season_month - 1)
            };
            let start = NaiveDate::from_ymd_opt(prev_year, prev_month, 1)
                .unwrap_or_else(|| unreachable!("previous month is always valid"));
            let end = current.start;
            let hours = month_total_hours(prev_year, prev_month);
            Some(SeasonPeriodWindow { start, end, hours })
        }
        SeasonCycleType::Weekly => {
            let end = current.start;
            let start = end - TimeDelta::days(7);
            Some(SeasonPeriodWindow {
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
            let len = season_map.seasons.len();
            let prev_def = &season_map.seasons[(pos + len - 1) % len];
            let current_year = find_season_year_custom(current.start, current.end, season_def);
            let target_year = if pos == 0 {
                current_year - 1
            } else {
                current_year
            };
            let (start, end) = custom_period_bounds(target_year, prev_def);
            let days = u32::try_from((end - start).num_days())
                .unwrap_or_else(|_| unreachable!("custom period day count always fits in u32"));
            Some(SeasonPeriodWindow {
                start,
                end,
                hours: f64::from(days) * 24.0,
            })
        }
    }
}

/// Walk [`previous_season_period_window`] `k` times from `anchor` (`k == 0` returns `anchor`).
#[must_use]
pub fn nth_previous_occurrence(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    anchor: &SeasonPeriodWindow,
    k: usize,
) -> Option<SeasonPeriodWindow> {
    let mut window = SeasonPeriodWindow {
        start: anchor.start,
        end: anchor.end,
        hours: anchor.hours,
    };
    let mut current_id = season_def.id;
    for _ in 0..k {
        let current_def = season_map.seasons.iter().find(|s| s.id == current_id)?;
        window = previous_season_period_window(season_map, current_def, &window)?;
        current_id = season_map.season_for_date(window.start)?;
    }
    Some(window)
}

/// The calendar year identifying which occurrence of `season_def` `stage`
/// belongs to, disambiguating repeated season ids across years.
///
/// `Weekly` uses the ISO week-numbering year (`iso_week().year()`), not the
/// calendar year of the window start: a week's Monday can fall in the prior
/// December (`from_isoywd_opt(2004, 1, Mon)` is 2003-12-29), so the window
/// start's calendar year would misclassify that week as the prior year's.
pub(crate) fn resolved_year(
    season_map: &SeasonMap,
    season_def: &SeasonDefinition,
    stage: &Stage,
) -> i32 {
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

/// One hydro's realized-inflow window over `[start_date, end_date)`.
pub struct RealizedWindow {
    /// Inclusive window start.
    pub start_date: NaiveDate,
    /// Exclusive window end (`end_date > start_date`).
    pub end_date: NaiveDate,
    /// Realized inflow value in m3/s over the window.
    pub value_m3s: f64,
}

/// Day-weighted mean value and period-coverage fraction produced by [`cast`].
pub struct Projection {
    /// Coverage-normalized day-weighted mean of the overlapping windows' values.
    pub value: f64,
    /// Fraction of the period's hours covered by the input windows.
    pub coverage: f64,
}

/// Hours of calendar overlap between `[window_start, window_end)` and
/// `[period_start, period_end)`; zero for a window that does not intersect
/// the period.
fn overlap_hours(
    window_start: NaiveDate,
    window_end: NaiveDate,
    period_start: NaiveDate,
    period_end: NaiveDate,
) -> f64 {
    let overlap_start = window_start.max(period_start);
    let overlap_end = window_end.min(period_end);
    let days = u32::try_from((overlap_end - overlap_start).num_days().max(0))
        .unwrap_or_else(|_| unreachable!("overlap day count always fits in u32"));
    f64::from(days) * 24.0
}

/// Day-weighted projection of `windows` onto one `period` occurrence: the
/// coverage-normalized mean value over the overlap, and the covered fraction
/// of `period`.
///
/// `windows` must be non-overlapping — overlapping input double-counts shared
/// days into both `value` and `coverage`.
#[must_use]
pub fn cast(windows: &[RealizedWindow], period: &SeasonPeriodWindow) -> Projection {
    let mut overlap = 0.0_f64;
    let mut weighted_value = 0.0_f64;
    for window in windows {
        let hours = overlap_hours(window.start_date, window.end_date, period.start, period.end);
        overlap += hours;
        weighted_value += window.value_m3s * hours;
    }

    let value = if overlap > 0.0 {
        weighted_value / overlap
    } else {
        0.0
    };
    let coverage = overlap / period.hours;

    Projection { value, coverage }
}

/// Sorted, deduplicated day boundaries from `record` and `conditioning`
/// window edges — the maximal segments over which the covering window (if
/// any) in either layer cannot change.
fn window_boundaries(record: &[RealizedWindow], conditioning: &[RealizedWindow]) -> Vec<NaiveDate> {
    let mut points: Vec<NaiveDate> = record
        .iter()
        .chain(conditioning)
        .flat_map(|w| [w.start_date, w.end_date])
        .collect();
    points.sort_unstable();
    points.dedup();
    points
}

/// Value of the window in `windows` (assumed non-overlapping within the
/// slice) that covers `day` under the `[start_date, end_date)` convention.
fn covering_value(windows: &[RealizedWindow], day: NaiveDate) -> Option<f64> {
    windows
        .iter()
        .find(|w| w.start_date <= day && day < w.end_date)
        .map(|w| w.value_m3s)
}

/// Day-resolved, non-overlapping merge of `record` and `conditioning`:
/// `conditioning` shadows `record` on every day both cover (the conditioning
/// value replaces the record value; the two never blend), `record` alone
/// covers a day neither shadows, and a day covered by neither is absent from
/// the output. The result is sorted by `start_date` and satisfies `cast`'s
/// non-overlapping precondition.
#[must_use]
pub fn merge_layered_windows(
    record: &[RealizedWindow],
    conditioning: &[RealizedWindow],
) -> Vec<RealizedWindow> {
    let cut_points = window_boundaries(record, conditioning);
    let mut merged = Vec::with_capacity(cut_points.len().saturating_sub(1));

    for segment in cut_points.windows(2) {
        let (segment_start, segment_end) = (segment[0], segment[1]);
        let value = covering_value(conditioning, segment_start)
            .or_else(|| covering_value(record, segment_start));
        if let Some(value_m3s) = value {
            merged.push(RealizedWindow {
                start_date: segment_start,
                end_date: segment_end,
                value_m3s,
            });
        }
    }

    merged
}

/// `[start, end)` day-spans where `conditioning` overrode `record` (both
/// cover the day), coalesced into contiguous spans; empty when the two
/// layers share no days.
#[must_use]
pub fn shadowed_spans(
    record: &[RealizedWindow],
    conditioning: &[RealizedWindow],
) -> Vec<(NaiveDate, NaiveDate)> {
    let cut_points = window_boundaries(record, conditioning);
    let mut spans: Vec<(NaiveDate, NaiveDate)> = Vec::new();

    for segment in cut_points.windows(2) {
        let (segment_start, segment_end) = (segment[0], segment[1]);
        let shadowed = covering_value(record, segment_start).is_some()
            && covering_value(conditioning, segment_start).is_some();
        if shadowed {
            match spans.last_mut() {
                Some(last) if last.1 == segment_start => last.1 = segment_end,
                _ => spans.push((segment_start, segment_end)),
            }
        }
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::{
        RealizedWindow, SeasonCycleType, SeasonDefinition, SeasonMap, SeasonPeriodWindow, Stage,
        cast, merge_layered_windows, nth_previous_occurrence, season_period_window, shadowed_spans,
    };
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    #[test]
    fn test_season_period_window_monthly_april_stage() {
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
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        let season_def = &season_map.seasons[3];

        let start = NaiveDate::from_ymd_opt(2026, 4, 4).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 5, 2).unwrap();
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: start,
            end_date: end,
            season_id: Some(3),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 28.0 * 24.0,
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
        };

        let window = season_period_window(&season_map, season_def, &stage);

        assert_eq!(window.hours, 30.0 * 24.0);
        assert_eq!(window.start, NaiveDate::from_ymd_opt(2026, 4, 1).unwrap());
        assert_eq!(window.end, NaiveDate::from_ymd_opt(2026, 5, 1).unwrap());
    }

    #[test]
    fn test_previous_occurrence_monthly_walks_back() {
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
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        let season_def = &season_map.seasons[3];

        let start = NaiveDate::from_ymd_opt(2026, 4, 4).unwrap();
        let end = NaiveDate::from_ymd_opt(2026, 5, 2).unwrap();
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: start,
            end_date: end,
            season_id: Some(3),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 28.0 * 24.0,
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
        };
        let anchor = season_period_window(&season_map, season_def, &stage);

        let march = nth_previous_occurrence(&season_map, season_def, &anchor, 1)
            .expect("k=1 must resolve to March 2026");
        assert_eq!(march.start, NaiveDate::from_ymd_opt(2026, 3, 1).unwrap());
        assert_eq!(march.end, NaiveDate::from_ymd_opt(2026, 4, 1).unwrap());
        assert_eq!(march.hours, 31.0 * 24.0);

        let february = nth_previous_occurrence(&season_map, season_def, &anchor, 2)
            .expect("k=2 must resolve to February 2026");
        assert_eq!(february.start, NaiveDate::from_ymd_opt(2026, 2, 1).unwrap());
        assert_eq!(february.end, NaiveDate::from_ymd_opt(2026, 3, 1).unwrap());
        assert_eq!(february.hours, 28.0 * 24.0);

        let january = nth_previous_occurrence(&season_map, season_def, &anchor, 3)
            .expect("k=3 must resolve to January 2026");
        assert_eq!(january.start, NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
        assert_eq!(january.end, NaiveDate::from_ymd_opt(2026, 2, 1).unwrap());
        assert_eq!(january.hours, 31.0 * 24.0);
    }

    #[test]
    fn test_nth_previous_occurrence_custom_year_wrap() {
        let seasons = vec![
            SeasonDefinition {
                id: 5,
                label: "June".to_string(),
                month_start: 6,
                day_start: None,
                month_end: None,
                day_end: None,
            },
            SeasonDefinition {
                id: 12,
                label: "Q3".to_string(),
                month_start: 7,
                day_start: None,
                month_end: Some(9),
                day_end: None,
            },
        ];
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons,
        };
        let season_def = &season_map.seasons[1];

        let anchor = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
            hours: 92.0 * 24.0,
        };

        let june_2024 = nth_previous_occurrence(&season_map, season_def, &anchor, 1)
            .expect("k=1 must resolve to June 2024");
        assert_eq!(
            june_2024.start,
            NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()
        );
        assert_eq!(june_2024.end, NaiveDate::from_ymd_opt(2024, 7, 1).unwrap());

        let q3_2023 = nth_previous_occurrence(&season_map, season_def, &anchor, 2)
            .expect("k=2 must resolve to Q3 2023, not Q3 2024");
        assert_eq!(q3_2023.start, NaiveDate::from_ymd_opt(2023, 7, 1).unwrap());
        assert_eq!(q3_2023.end, NaiveDate::from_ymd_opt(2023, 10, 1).unwrap());
    }

    #[test]
    fn test_nth_previous_occurrence_k_zero_is_identity() {
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
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        let season_def = &season_map.seasons[3];
        let anchor = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            hours: 30.0 * 24.0,
        };

        let result = nth_previous_occurrence(&season_map, season_def, &anchor, 0)
            .expect("k=0 must return anchor");

        assert_eq!(result.start, anchor.start);
        assert_eq!(result.end, anchor.end);
        assert_eq!(result.hours, anchor.hours);
    }

    fn april_2026() -> SeasonPeriodWindow {
        SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            hours: 30.0 * 24.0,
        }
    }

    #[test]
    fn test_cast_full_coverage_returns_window_value() {
        let period = april_2026();
        let windows = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];

        let projection = cast(&windows, &period);

        assert_eq!(projection.value, 480.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_cast_partition_day_weighted_mean() {
        let period = april_2026();
        let windows = [
            RealizedWindow {
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
                value_m3s: 500.0,
            },
            RealizedWindow {
                start_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                value_m3s: 480.0,
            },
        ];

        let projection = cast(&windows, &period);

        assert_eq!(projection.value, (500.0 * 72.0 + 480.0 * 648.0) / 720.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_cast_straddling_window_split_once() {
        let april = april_2026();
        let may = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
            hours: 31.0 * 24.0,
        };
        let straddling = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 25).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 5, 8).unwrap(),
            value_m3s: 350.0,
        }];

        let april_projection = cast(&straddling, &april);
        let may_projection = cast(&straddling, &may);

        assert_eq!(april_projection.coverage, 144.0 / 720.0);
        assert_eq!(may_projection.coverage, 168.0 / 744.0);
        assert_eq!(april_projection.value, 350.0);
        assert_eq!(may_projection.value, 350.0);
    }

    #[test]
    fn test_straddling_window_contributes_each_day_once() {
        let april = april_2026();
        let may = SeasonPeriodWindow {
            start: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            end: NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
            hours: 31.0 * 24.0,
        };
        let window_start = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
        let window_end = NaiveDate::from_ymd_opt(2026, 5, 8).unwrap();
        let value = 350.0;
        let straddling = [RealizedWindow {
            start_date: window_start,
            end_date: window_end,
            value_m3s: value,
        }];
        let window_total_hours =
            f64::from(u32::try_from((window_end - window_start).num_days()).unwrap()) * 24.0;

        let april_projection = cast(&straddling, &april);
        let may_projection = cast(&straddling, &may);

        let april_overlap_hours = april_projection.coverage * april.hours;
        let may_overlap_hours = may_projection.coverage * may.hours;
        assert_eq!(
            april_overlap_hours + may_overlap_hours,
            window_total_hours,
            "each day of the straddling window must contribute to exactly one \
             period: overlap hours must sum to the window's total, no double-count"
        );

        let april_partial = april_overlap_hours * april_projection.value;
        let may_partial = may_overlap_hours * may_projection.value;
        assert_eq!(
            april_partial + may_partial,
            window_total_hours * value,
            "the two per-period partial values must sum to the window's \
             day-weighted total"
        );
    }

    #[test]
    fn test_cast_out_of_period_window_contributes_zero() {
        let period = april_2026();
        let march_window = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            value_m3s: 999.0,
        }];

        let projection = cast(&march_window, &period);

        assert_eq!(projection.value, 0.0);
        assert_eq!(projection.coverage, 0.0);
    }

    #[test]
    fn test_merge_no_conditioning_preserves_record_cast() {
        let period = april_2026();
        let record = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];

        let merged = merge_layered_windows(&record, &[]);
        let projection = cast(&merged, &period);

        assert_eq!(projection.value, 480.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_merge_partial_shadow_blends_by_day() {
        let period = april_2026();
        let record = [RealizedWindow {
            start_date: period.start,
            end_date: period.end,
            value_m3s: 480.0,
        }];
        let conditioning = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
            value_m3s: 500.0,
        }];

        let merged = merge_layered_windows(&record, &conditioning);
        let projection = cast(&merged, &period);

        assert_eq!(projection.value, (500.0 * 72.0 + 480.0 * 648.0) / 720.0);
        assert_eq!(projection.coverage, 1.0);
    }

    #[test]
    fn test_shadowed_spans_reports_overlap() {
        let record = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            value_m3s: 480.0,
        }];
        let conditioning = [RealizedWindow {
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4).unwrap(),
            value_m3s: 500.0,
        }];

        let spans = shadowed_spans(&record, &conditioning);

        assert_eq!(
            spans,
            vec![(
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 4, 4).unwrap()
            )]
        );
    }

    mod cast_proptests {
        use proptest::prelude::*;

        use super::super::{RealizedWindow, SeasonPeriodWindow, cast};
        use chrono::{NaiveDate, TimeDelta};

        proptest! {
            #[test]
            fn cast_partition_matches_day_weighted_mean(
                segments in prop::collection::vec((1_u32..=15, -1000.0_f64..1000.0_f64), 1..=12),
            ) {
                let period_start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
                let mut windows = Vec::with_capacity(segments.len());
                let mut cursor = period_start;
                let mut weighted_hours_sum = 0.0_f64;
                let mut total_hours = 0.0_f64;
                for &(days, value) in &segments {
                    let end = cursor + TimeDelta::days(i64::from(days));
                    let hours = f64::from(days) * 24.0;
                    windows.push(RealizedWindow {
                        start_date: cursor,
                        end_date: end,
                        value_m3s: value,
                    });
                    weighted_hours_sum += value * hours;
                    total_hours += hours;
                    cursor = end;
                }
                let period = SeasonPeriodWindow {
                    start: period_start,
                    end: cursor,
                    hours: total_hours,
                };
                let expected_value = weighted_hours_sum / total_hours;

                let projection = cast(&windows, &period);

                prop_assert!((projection.value - expected_value).abs() < 1e-9);
                prop_assert!((projection.coverage - 1.0).abs() < 1e-9);
            }
        }
    }

    mod merge_proptests {
        use proptest::prelude::*;

        use super::super::{RealizedWindow, SeasonPeriodWindow, cast, merge_layered_windows};
        use chrono::{NaiveDate, TimeDelta};

        proptest! {
            #[test]
            fn merge_disjoint_windows_matches_concatenated_cast(
                segments in prop::collection::vec((any::<bool>(), 1_u32..=10, -500.0_f64..500.0_f64), 1..=16),
            ) {
                let period_start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
                let mut record = Vec::new();
                let mut conditioning = Vec::new();
                let mut cursor = period_start;
                let mut total_hours = 0.0_f64;
                for &(is_conditioning, days, value) in &segments {
                    let end = cursor + TimeDelta::days(i64::from(days));
                    let window = RealizedWindow {
                        start_date: cursor,
                        end_date: end,
                        value_m3s: value,
                    };
                    if is_conditioning {
                        conditioning.push(window);
                    } else {
                        record.push(window);
                    }
                    total_hours += f64::from(days) * 24.0;
                    cursor = end;
                }
                let period = SeasonPeriodWindow {
                    start: period_start,
                    end: cursor,
                    hours: total_hours,
                };

                let merged = merge_layered_windows(&record, &conditioning);
                let mut concatenated = Vec::with_capacity(record.len() + conditioning.len());
                concatenated.extend(record);
                concatenated.extend(conditioning);

                let merged_projection = cast(&merged, &period);
                let concatenated_projection = cast(&concatenated, &period);

                prop_assert!((merged_projection.value - concatenated_projection.value).abs() < 1e-9);
                prop_assert!(
                    (merged_projection.coverage - concatenated_projection.coverage).abs() < 1e-9
                );
            }
        }
    }
}
