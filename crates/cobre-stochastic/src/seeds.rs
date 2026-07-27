//! Derivation of per-hydro PAR lag-slot and accumulator seeds from the
//! layered `inflow_history` record and `recent_observations` conditioning
//! series ([`crate::season_cast`]).

use cobre_core::{Hydro, InflowHistoryRow, RecentObservation, SeasonMap, Stage};

use crate::season_cast::{
    RealizedWindow, cast, merge_layered_windows, nth_previous_occurrence, season_period_window,
};

/// Per-hydro PAR lag-slot and accumulator seeds, indexed by canonical hydro
/// position — the iteration order [`derive_inflow_seeds`] walks `hydros` in.
#[derive(Debug)]
pub struct DerivedInflowSeeds {
    /// Lag-slot seeds, entity-major: `lag_values[pos * l_state + lag]`, lag
    /// index `0` is lag 1 (most recent). Length `n_hydros * l_state`.
    pub lag_values: Vec<f64>,
    /// Per-hydro accumulator seed (`p.value * p.coverage`) for the first
    /// stage's in-progress occurrence. Length `n_hydros`.
    pub accum: Vec<f64>,
    /// Per-hydro coverage fraction seed (`p.coverage`). Length `n_hydros`.
    pub weight: Vec<f64>,
}

impl DerivedInflowSeeds {
    /// An all-zero seed for `n_hydros` hydros and `l_state` lag slots.
    #[must_use]
    pub fn zero(n_hydros: usize, l_state: usize) -> Self {
        Self {
            lag_values: vec![0.0_f64; n_hydros * l_state],
            accum: vec![0.0_f64; n_hydros],
            weight: vec![0.0_f64; n_hydros],
        }
    }
}

/// Derive [`DerivedInflowSeeds`] from the layered `record`/`conditioning`
/// inflow series, one hydro at a time in `hydros`' canonical order (the loop
/// index is the returned position; no id-to-position lookup is built).
///
/// `conditioning` shadows `record` day-wise via [`merge_layered_windows`]
/// before every [`cast`]. Returns [`DerivedInflowSeeds::zero`] when `hydros`
/// is empty, `first_stage.season_id` is `None`, or that id is absent from
/// `season_map.seasons`.
#[must_use]
pub fn derive_inflow_seeds(
    record: &[InflowHistoryRow],
    conditioning: &[RecentObservation],
    hydros: &[Hydro],
    first_stage: &Stage,
    season_map: &SeasonMap,
    l_state: usize,
) -> DerivedInflowSeeds {
    let n_hydros = hydros.len();
    if n_hydros == 0 {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    }

    let Some(season_id) = first_stage.season_id else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let in_progress = season_period_window(season_map, season_def, first_stage);

    let mut seeds = DerivedInflowSeeds::zero(n_hydros, l_state);

    for (pos, hydro) in hydros.iter().enumerate() {
        let record_windows: Vec<RealizedWindow> = record
            .iter()
            .filter(|row| row.hydro_id == hydro.id)
            .map(|row| RealizedWindow {
                start_date: row.start_date,
                end_date: row.end_date,
                value_m3s: row.value_m3s,
            })
            .collect();
        let conditioning_windows: Vec<RealizedWindow> = conditioning
            .iter()
            .filter(|obs| obs.hydro_id == hydro.id)
            .map(|obs| RealizedWindow {
                start_date: obs.start_date,
                end_date: obs.end_date,
                value_m3s: obs.value_m3s,
            })
            .collect();
        let merged = merge_layered_windows(&record_windows, &conditioning_windows);

        let in_progress_projection = cast(&merged, &in_progress);
        seeds.accum[pos] = in_progress_projection.value * in_progress_projection.coverage;
        seeds.weight[pos] = in_progress_projection.coverage;

        for k in 1..=l_state {
            let Some(occurrence) = nth_previous_occurrence(season_map, season_def, &in_progress, k)
            else {
                continue;
            };
            seeds.lag_values[pos * l_state + (k - 1)] = cast(&merged, &occurrence).value;
        }
    }

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
            StageRiskConfig, StageStateConfig,
        },
    };

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
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

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: d(2020, 1, 1),
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

    #[test]
    #[allow(clippy::float_cmp)] // full-coverage windows make cast's weighted-value/overlap ratio bit-exact
    fn test_derive_inflow_seeds_full_coverage_matches_positional_lags() {
        let season_map = monthly_season_map();
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(1)];
        let record = vec![
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 100.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 2, 1),
                end_date: d(2026, 3, 1),
                value_m3s: 200.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 1, 1),
                end_date: d(2026, 2, 1),
                value_m3s: 300.0,
            },
        ];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 3);

        assert_eq!(seeds.lag_values[0], 100.0);
        assert_eq!(seeds.lag_values[1], 200.0);
        assert_eq!(seeds.lag_values[2], 300.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // whole-day-hours coverage ratio and its product with rate are bit-exact
    fn test_derive_inflow_seeds_partial_inprogress_accumulator() {
        let season_map = monthly_season_map();
        let first_stage = make_stage(0, d(2026, 4, 11), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(1)];
        let rate = 500.0;
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: d(2026, 4, 1),
            end_date: d(2026, 4, 11),
            value_m3s: rate,
        }];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 0);

        assert!(seeds.lag_values.is_empty());
        let covered_hours = 10.0 * 24.0;
        let period_hours = 30.0 * 24.0;
        let coverage = covered_hours / period_hours;
        assert_eq!(seeds.weight[0], coverage);
        assert_eq!(seeds.accum[0], rate * coverage);
    }

    #[test]
    #[allow(clippy::float_cmp)] // derivation and the hand-computed merge_layered_windows call share the same operand order, so the ratios are bit-identical
    fn test_derive_inflow_seeds_conditioning_shadows_record() {
        let season_map = monthly_season_map();
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(7)];
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(7),
            start_date: d(2026, 3, 1),
            end_date: d(2026, 4, 1),
            value_m3s: 100.0,
        }];
        let conditioning = vec![RecentObservation {
            hydro_id: EntityId(7),
            start_date: d(2026, 3, 10),
            end_date: d(2026, 3, 20),
            value_m3s: 999.0,
        }];

        let record_windows = vec![RealizedWindow {
            start_date: d(2026, 3, 1),
            end_date: d(2026, 4, 1),
            value_m3s: 100.0,
        }];
        let conditioning_windows = vec![RealizedWindow {
            start_date: d(2026, 3, 10),
            end_date: d(2026, 3, 20),
            value_m3s: 999.0,
        }];
        let hand_merged = merge_layered_windows(&record_windows, &conditioning_windows);

        let season_def = season_map.seasons.iter().find(|s| s.id == 3).unwrap();
        let in_progress = season_period_window(&season_map, season_def, &first_stage);
        let march = nth_previous_occurrence(&season_map, season_def, &in_progress, 1).unwrap();
        let expected_lag1 = cast(&hand_merged, &march).value;

        let seeds = derive_inflow_seeds(
            &record,
            &conditioning,
            &hydros,
            &first_stage,
            &season_map,
            1,
        );

        assert_eq!(seeds.lag_values[0], expected_lag1);
        assert_ne!(expected_lag1, 100.0);
    }

    #[test]
    fn test_derive_inflow_seeds_guard_conditions_return_zero() {
        let season_map = monthly_season_map();
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(1)];
        let l_state = 2;

        let empty_hydros: Vec<Hydro> = Vec::new();
        let seeds =
            derive_inflow_seeds(&[], &[], &empty_hydros, &first_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(0, l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);

        let no_season_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), None);
        let seeds = derive_inflow_seeds(&[], &[], &hydros, &no_season_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(hydros.len(), l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);

        let unresolvable_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(99));
        let seeds =
            derive_inflow_seeds(&[], &[], &hydros, &unresolvable_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(hydros.len(), l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);
    }

    #[test]
    #[allow(clippy::float_cmp)] // full-coverage window whole-day-hours arithmetic keeps the ratio bit-exact
    fn test_derive_inflow_seeds_weekly_cycle_populates_lags() {
        let season_map = weekly_season_map();
        let first_stage = make_stage(2, d(2024, 1, 15), d(2024, 1, 22), Some(2));
        let hydros = vec![make_hydro(9)];
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(9),
            start_date: d(2024, 1, 8),
            end_date: d(2024, 1, 15),
            value_m3s: 250.0,
        }];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 1);

        assert_eq!(seeds.lag_values[0], 250.0);
    }

    /// Regression: `hydros` in canonical `(operational_start_date, id)` order
    /// can be id-DESCENDING — here hydro id=1's earlier commissioning date
    /// sorts it before hydro id=0. Each hydro's own record must still land in
    /// its OWN canonical-position slot: the derivation resolves position by
    /// iterating `hydros` directly (`hydros.iter().enumerate()`), never by
    /// `binary_search_by_key` (which requires id-ascending order and would
    /// silently misattribute a record under this staggered ordering).
    #[test]
    #[allow(clippy::float_cmp)] // full-coverage window whole-day-hours arithmetic keeps the value bit-exact
    fn test_seed_correct_under_staggered_commissioning_dates() {
        let season_map = monthly_season_map();
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));

        let mut hydro_1_earlier = make_hydro(1);
        hydro_1_earlier.operational_start_date = d(2024, 1, 1);
        let mut hydro_0_later = make_hydro(0);
        hydro_0_later.operational_start_date = d(2025, 6, 1);
        // Canonical order: hydro id=1 (earlier date) at position 0, hydro
        // id=0 (later date) at position 1 — id-descending, not id-ascending.
        let hydros = vec![hydro_1_earlier, hydro_0_later];

        let record = vec![
            InflowHistoryRow {
                hydro_id: EntityId(0),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 500.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 300.0,
            },
        ];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 1);

        assert_eq!(
            seeds.lag_values[0], 300.0,
            "hydro id=1 (canonical position 0) lag should be its own record value"
        );
        assert_eq!(
            seeds.lag_values[1], 500.0,
            "hydro id=0 (canonical position 1) lag should be its own record value"
        );
    }
}
