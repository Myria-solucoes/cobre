//! Precomputation of per-stage lag accumulation weights and period
//! finalization flags from stage date boundaries and season definitions.
//!
//! [`precompute_stage_lag_transitions`] runs once at setup; the resulting
//! per-stage slice is consumed read-only on the hot path, keeping calendar
//! arithmetic out of inner solver loops.

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};
use cobre_core::{
    entities::hydro::Hydro,
    initial_conditions::RecentObservation,
    temporal::{SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageLagTransition},
    window_period_overlaps,
};

use crate::season_cast::{
    find_season_year_monthly, month_total_hours, next_season_period_window, resolved_year,
    season_period_window,
};

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
pub fn compute_recent_observation_seed(
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

/// Overlap hours between `stage`'s calendar span and a single period of
/// `period_hours` duration starting at `period_start`, via
/// [`window_period_overlaps`] framed with the origin at `period_start`.
///
/// A negative offset (the stage starts before `period_start`) is the intended
/// pre-period straddle case — `window_period_overlaps` clamps it, so this
/// does not guard or reject it.
fn single_period_overlap_hours(stage: &Stage, period_start: NaiveDate, period_hours: f64) -> f64 {
    let start_days = i32::try_from((stage.start_date - period_start).num_days())
        .unwrap_or_else(|_| unreachable!("stage-to-period day offset always fits in i32"));
    let window_start_hours = f64::from(start_days) * 24.0;

    let width_days = u32::try_from((stage.end_date - stage.start_date).num_days())
        .unwrap_or_else(|_| unreachable!("stage width in days always fits in u32"));
    let window_width_hours = f64::from(width_days) * 24.0;

    window_period_overlaps(window_start_hours, window_width_hours, &[period_hours])
        .first()
        .copied()
        .unwrap_or(0.0)
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
    let current = season_period_window(season_map, season_def, stage);

    let accumulate_weight =
        single_period_overlap_hours(stage, current.start, current.hours) / current.hours;

    let spillover_weight = next_season_period_window(season_map, season_def, &current)
        .map_or(0.0, |next| {
            single_period_overlap_hours(stage, next.start, next.hours) / next.hours
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

/// Derives the `downstream_par_order` gate consumed by
/// [`precompute_stage_lag_transitions`] and by η-inversion
/// (`standardize_historical_windows`): `par_max_order` once any stage crosses
/// into the quarterly range (`season_id >= 12`), gated off only for `Weekly`
/// — an ISO week number reaching 12 has nothing to do with quarters, while a
/// `Monthly` or `Custom` cycle's `season_id >= 12` is a deliberate quarterly
/// convention; a `None` `season_map` also leaves the ring inert (`0`). Every
/// call site that needs this gate — training/simulation lag transitions and
/// both the rank-0 and non-root opening-tree builds — routes through this one
/// function; an independent re-derivation risks the two opening-tree sides
/// diverging across MPI ranks.
#[must_use]
pub fn derive_downstream_par_order(
    stages: &[Stage],
    par_max_order: usize,
    season_map: Option<&SeasonMap>,
) -> usize {
    let cycle_admits_ring =
        season_map.is_some_and(|sm| !matches!(sm.cycle_type, SeasonCycleType::Weekly));
    let has_quarterly_stages = stages
        .iter()
        .any(|s| s.season_id.is_some_and(|id| id >= 12));
    if has_quarterly_stages && cycle_admits_ring {
        par_max_order
    } else {
        0
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

        let downstream_accumulate_weight =
            single_period_overlap_hours(stage, quarter_period_start, quarter_total_hours)
                / quarter_total_hours;

        let next_quarter_start_month = quarter_end_month + 1; // may be 13 → wrap to next year
        let (next_q_year, next_q_start_month) = if next_quarter_start_month > 12 {
            (year + 1, next_quarter_start_month - 12)
        } else {
            (year, next_quarter_start_month)
        };
        let next_quarter_end_month = next_q_start_month + 2;
        let next_quarter_start = NaiveDate::from_ymd_opt(next_q_year, next_q_start_month, 1)
            .unwrap_or_else(|| unreachable!("next quarter start date is always valid"));
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

        let downstream_spillover_weight =
            single_period_overlap_hours(stage, next_quarter_start, next_quarter_total_hours)
                / next_quarter_total_hours;

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

    // -----------------------------------------------------------------------
    // derive_downstream_par_order gate (Monthly-only convention)
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_downstream_par_order_weekly_returns_zero() {
        let season_map = weekly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 1, 8), Some(0)),
            make_stage(1, d(2026, 1, 8), d(2026, 1, 15), Some(1)),
            make_stage(2, d(2026, 1, 15), d(2026, 1, 22), Some(2)),
            make_stage(3, d(2026, 1, 22), d(2026, 1, 29), Some(12)),
        ];

        let derived = derive_downstream_par_order(&stages, 1, Some(&season_map));
        assert_eq!(
            derived, 0,
            "a Weekly season cycle must never activate the quarterly ring, even \
             when a stage crosses season_id >= 12"
        );

        let transitions = precompute_stage_lag_transitions(&stages, &season_map, derived);
        assert!(
            transitions.iter().all(|t| !t.rebuild_from_downstream),
            "the ring must stay provably inert: no stage rebuilds from downstream"
        );
    }

    #[test]
    fn test_derive_downstream_par_order_monthly_quarterly_unchanged() {
        let season_map = monthly_season_map();
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
            make_stage(2, d(2026, 3, 1), d(2026, 4, 1), Some(2)),
            make_stage(3, d(2026, 4, 1), d(2026, 7, 1), Some(12)),
        ];

        let derived = derive_downstream_par_order(&stages, 1, Some(&season_map));
        assert_eq!(
            derived, 1,
            "a Monthly season cycle crossing season_id >= 12 must activate the \
             quarterly ring at par_max_order"
        );
    }

    #[test]
    fn test_derive_downstream_par_order_no_season_map_returns_zero() {
        let stages = vec![
            make_stage(0, d(2026, 1, 1), d(2026, 2, 1), Some(0)),
            make_stage(1, d(2026, 2, 1), d(2026, 3, 1), Some(1)),
            make_stage(2, d(2026, 3, 1), d(2026, 4, 1), Some(2)),
            make_stage(3, d(2026, 4, 1), d(2026, 7, 1), Some(12)),
        ];

        let derived = derive_downstream_par_order(&stages, 1, None);
        assert_eq!(derived, 0, "a None season_map must leave the ring inert");
    }

    #[test]
    fn test_derive_downstream_par_order_custom_quarterly_stays_active() {
        let season_map = custom_multi_resolution_season_map();
        let june_stage = make_stage(0, d(2024, 6, 1), d(2024, 7, 1), Some(5));
        let q3_stage = make_stage(1, d(2024, 7, 1), d(2024, 10, 1), Some(12));
        let stages = vec![june_stage, q3_stage];

        let derived = derive_downstream_par_order(&stages, 1, Some(&season_map));
        assert_eq!(
            derived, 1,
            "a Custom season cycle crossing season_id >= 12 must keep the \
             quarterly ring active at par_max_order — only Weekly is gated off"
        );
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

    // -----------------------------------------------------------------------
    // Migration-equivalence: window_period_overlaps route vs the
    // pre-migration days_in_period * 24.0 route
    // -----------------------------------------------------------------------

    /// Private copy of the pre-migration `days_in_period` formula, kept only
    /// to assert bit-equality against the `window_period_overlaps` route the
    /// production code now uses exclusively.
    fn reference_days_in_period(
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

    fn reference_weight(
        stage_start: NaiveDate,
        stage_end: NaiveDate,
        period_start: NaiveDate,
        period_end: NaiveDate,
        period_hours: f64,
    ) -> f64 {
        let days = reference_days_in_period(stage_start, stage_end, period_start, period_end);
        f64::from(days) * 24.0 / period_hours
    }

    #[test]
    fn test_migration_equivalence_window_period_overlaps_bit_identical_to_days_in_period() {
        // (a) stage straddling a monthly boundary: accumulate + nonzero spillover.
        let stage_monthly = make_stage(0, d(2026, 1, 28), d(2026, 2, 4), Some(0));
        let jan_start = d(2026, 1, 1);
        let jan_end = d(2026, 2, 1);
        let jan_hours = 31.0 * 24.0;
        let feb_start = d(2026, 2, 1);
        let feb_end = d(2026, 3, 1);
        let feb_hours = 28.0 * 24.0;

        let accumulate_new =
            single_period_overlap_hours(&stage_monthly, jan_start, jan_hours) / jan_hours;
        let accumulate_ref = reference_weight(
            stage_monthly.start_date,
            stage_monthly.end_date,
            jan_start,
            jan_end,
            jan_hours,
        );
        assert_eq!(
            accumulate_new, accumulate_ref,
            "monthly accumulate weight must be bit-identical"
        );

        let spillover_new =
            single_period_overlap_hours(&stage_monthly, feb_start, feb_hours) / feb_hours;
        let spillover_ref = reference_weight(
            stage_monthly.start_date,
            stage_monthly.end_date,
            feb_start,
            feb_end,
            feb_hours,
        );
        assert!(
            spillover_ref > 0.0,
            "fixture must exercise nonzero spillover"
        );
        assert_eq!(
            spillover_new, spillover_ref,
            "monthly spillover weight must be bit-identical"
        );

        // (b) Weekly full-week stage.
        let stage_weekly = make_stage(0, d(2024, 1, 1), d(2024, 1, 8), Some(0));
        let week_start = d(2024, 1, 1);
        let week_end = d(2024, 1, 8);
        let week_hours = 7.0 * 24.0;
        let weekly_new =
            single_period_overlap_hours(&stage_weekly, week_start, week_hours) / week_hours;
        let weekly_ref = reference_weight(
            stage_weekly.start_date,
            stage_weekly.end_date,
            week_start,
            week_end,
            week_hours,
        );
        assert_eq!(
            weekly_new, weekly_ref,
            "weekly full-week weight must be bit-identical"
        );

        // (c) Custom sub-window stage (the D30 shape): a Q3 stage spanning
        // only the first 30 days of a 92-day quarter.
        let stage_custom = make_stage(1, d(2024, 7, 1), d(2024, 7, 31), Some(12));
        let q3_start = d(2024, 7, 1);
        let q3_end = d(2024, 10, 1);
        let q3_hours = 92.0 * 24.0;
        let custom_new = single_period_overlap_hours(&stage_custom, q3_start, q3_hours) / q3_hours;
        let custom_ref = reference_weight(
            stage_custom.start_date,
            stage_custom.end_date,
            q3_start,
            q3_end,
            q3_hours,
        );
        assert_eq!(
            custom_new, custom_ref,
            "custom sub-window weight must be bit-identical"
        );

        // (d) quarterly downstream window: a stage straddling the Q3/Q4
        // boundary, mirroring compute_downstream_transitions's own weighting.
        let stage_quarterly = make_stage(0, d(2026, 9, 25), d(2026, 10, 4), Some(8));
        let q3_2026_start = d(2026, 7, 1);
        let q3_2026_end = d(2026, 10, 1);
        let q3_2026_hours = 92.0 * 24.0;
        let q4_2026_start = d(2026, 10, 1);
        let q4_2026_end = d(2027, 1, 1);
        let q4_2026_hours = 92.0 * 24.0;

        let downstream_accumulate_new =
            single_period_overlap_hours(&stage_quarterly, q3_2026_start, q3_2026_hours)
                / q3_2026_hours;
        let downstream_accumulate_ref = reference_weight(
            stage_quarterly.start_date,
            stage_quarterly.end_date,
            q3_2026_start,
            q3_2026_end,
            q3_2026_hours,
        );
        assert_eq!(
            downstream_accumulate_new, downstream_accumulate_ref,
            "quarterly downstream accumulate weight must be bit-identical"
        );

        let downstream_spillover_new =
            single_period_overlap_hours(&stage_quarterly, q4_2026_start, q4_2026_hours)
                / q4_2026_hours;
        let downstream_spillover_ref = reference_weight(
            stage_quarterly.start_date,
            stage_quarterly.end_date,
            q4_2026_start,
            q4_2026_end,
            q4_2026_hours,
        );
        assert!(
            downstream_spillover_ref > 0.0,
            "fixture must exercise nonzero downstream spillover"
        );
        assert_eq!(
            downstream_spillover_new, downstream_spillover_ref,
            "quarterly downstream spillover weight must be bit-identical"
        );
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
