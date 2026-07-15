//! Layer 5 — Semantic validation: hydro, thermal, stage, penalty, and scenario rules.
//!
//! Validates all domain-specific business rules after Layers 2-4 have
//! ensured schema correctness, referential integrity, and dimensional
//! consistency.
//!
//! ## Layer 5a rules (hydro and thermal domain) — `validate_semantic_hydro_thermal`
//!
//! | # | Rule                                              | Source file                           | `ErrorKind`            |
//! |---|---------------------------------------------------|---------------------------------------|------------------------|
//! | 1 | Hydro cascade graph must be acyclic               | `system/hydros.json`                  | `CycleDetected`        |
//! | 2 | `min_storage_hm3 <= max_storage_hm3`              | `system/hydros.json`                  | `InvalidValue`         |
//! | 3 | `min_turbined_m3s <= max_turbined_m3s`            | `system/hydros.json`                  | `InvalidValue`         |
//! | 4 | `min_outflow_m3s <= max_outflow_m3s` (when Some)  | `system/hydros.json`                  | `InvalidValue`         |
//! | 5 | `min_generation_mw <= max_generation_mw` (hydro)  | `system/hydros.json`                  | `InvalidValue`         |
//! | 6 | `entry_stage_id < exit_stage_id` (when both Some) | all six entity types                  | `InvalidValue`         |
//! | 7 | Filling `start_stage_id` in study stage set       | `system/hydros.json`                  | `InvalidValue`         |
//! | 7a| Filling guards (hard): `filling ⟹ entry_stage_id` (a bare window without filling is valid), `start_stage_id < entry_stage_id`, seed in `[0, min_storage_hm3)`, no `exit_stage_id` on a filling hydro, seed `== 0` when `start_stage_id > 0` | `system/hydros.json` | `InvalidValue` |
//! | 7b| `entry_stage_id >= horizon` on a filling hydro (fills throughout, never operates within this study) | `system/hydros.json` | `ModelQuality` (warning) |
//! | 8 | Geometry `volume_hm3` strictly increasing         | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! | 9 | Geometry `height_m` non-decreasing                | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! |10 | Geometry `area_km2` non-decreasing                | `system/hydro_geometry.parquet`       | `BusinessRuleViolation`|
//! |11 | FPHA: at least 1 plane per (hydro, stage)         | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |12 | FPHA: `gamma_v >= 0`, `gamma_s <= 0`              | `system/fpha_hyperplanes.parquet`     | `BusinessRuleViolation`|
//! |13 | `min_generation_mw <= max_generation_mw` (thermal)| `system/thermals.json`                | `InvalidValue`         |
//! |14 | Anticipated thermal `lead_stages` within study horizon and lifecycle bounds | `system/thermals.json` | `BusinessRuleViolation` |
//! |15 | Anticipated thermals bijection with `past_anticipated_commitments` entries  | `initial_conditions.json` | `BusinessRuleViolation` |
//! |16 | Thermal `thermal_bounds.parquet` override `stage_id` within `[0, n_stages)` | `constraints/thermal_bounds.parquet` | `BusinessRuleViolation` |
//! |17 | `anticipated_decision(N)` in generic constraint targets an anticipated thermal | `constraints/generic_constraints.json` | `BusinessRuleViolation` |
//! |18 | `thermal_generation(N)` in generic constraint when `N` is anticipated (warn) | `constraints/generic_constraints.json` | `SemanticAmbiguity` (warning) |
//! |19 | Pumping `source_hydro_id != destination_hydro_id`  | `system/pumping_stations.json`        | `InvalidValue`         |
//! |20 | Per-block storage reference resolves to a real boundary (parallel `K>1` interior / out-of-range block rejected) | `constraints/generic_constraints.json` | `BusinessRuleViolation` |
//! |21 | `travel_time_hours` negative or non-finite                             | `system/hydros.json`                  | `InvalidValue`         |
//! |22 | `travel_time_hours == 0.0` — treated as undeclared, no arc created     | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |23 | Declared arc: `max_t(t_v/h_t)` below a smallness threshold             | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |24 | Declared arc: `t_v` exceeds the remaining study horizon at some stage  | `system/hydros.json`                  | `ModelQuality` (warning) |
//! |25 | Declared arc: `past_defluences` history shorter than the required pre-study depth (derived-from-`past_inflows` fallback logs a caveat instead) | `initial_conditions.json` | `BusinessRuleViolation` (or `ModelQuality` warning) |
//! |26 | 2+ declared arcs into one downstream plant with differing `travel_time_hours`, while any study stage is `Chronological` | `system/hydros.json` | `NotImplemented` |
//! |27 | `recent_observations` present but the season cycle is not `Monthly` — mid-period PAR lag seeding silently skipped | `initial_conditions.json` | `ModelQuality` (warning) |
//! |28 | `lead_stages` anticipated active window spans a stage-cadence transition (adjacent unequal stage durations); `lead_time` is the physically-anchored alternative | `system/thermals.json` | `ModelQuality` (warning) |
//! |29 | Study supplies an inflow annual component (`inflow_annual_components` non-empty) while `season_map.cycle_type` is not `Monthly` — PAR(p)-A is monthly-exclusive by design | `scenarios/inflow_annual_component.parquet` | `BusinessRuleViolation` |
//!
//! ## Layer 5b rules (stages, penalties, and scenario domain) — `validate_semantic_stages_penalties_scenarios`
//!
//! | #  | Rule                                                                    | Source file                                    | `ErrorKind`              |
//! |----|-------------------------------------------------------------------------|------------------------------------------------|--------------------------|
//! | 1  | Every transition `source_id`/`target_id` must refer to an existing stage| `stages.json`                                  | `InvalidValue`           |
//! | 2  | Outgoing transition probabilities sum to 1.0 (±1e-6) per source stage  | `stages.json`                                  | `InvalidValue`           |
//! | 3  | Cyclic graph: `annual_discount_rate > 0.0`                              | `stages.json`                                  | `InvalidValue`           |
//! | 4  | Every `Block.duration_hours > 0.0`                                      | `stages.json`                                  | `InvalidValue`           |
//! | 5  | CVaR: `alpha` in (0, 1], `lambda` in [0, 1]                             | `stages.json`                                  | `InvalidValue`           |
//! | 6  | `max(deficit_segment_costs) > filling_target_violation_cost`            | `penalties.json`                               | `ModelQuality` (warning) |
//! | 7  | `storage_violation_below_cost > max(deficit_segment_costs)`             | `penalties.json`                               | `ModelQuality` (warning) |
//! | 8  | `max(deficit_segment_costs) > max(constraint_violation_costs)`          | `penalties.json`                               | `ModelQuality` (warning) |
//! | 9  | `min(constraint_violation_costs) > max(resource_costs)`                 | `penalties.json`                               | `ModelQuality` (warning) |
//! |10  | `min(resource_costs) > 0`                                               | `penalties.json`                               | `ModelQuality` (warning) |
//! |11  | FPHA hydros: `turbined_cost >= 0`                                  | `penalties.json`                               | `BusinessRuleViolation`  |
//! |12  | `std_m3s >= 0.0`; warn when `== 0.0` (deterministic inflow)            | `scenarios/inflow_seasonal_stats.parquet`      | `ModelQuality` (warning) |
//! |13  | *(retired — number never reused)* | — | — |
//! |14  | Correlation matrix symmetry (`matrix[i][j] == matrix[j][i]` ±1e-9)     | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |15  | Correlation matrix diagonal entries equal 1.0 (±1e-9)                  | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |16  | Correlation off-diagonal entries in [-1.0, 1.0]                        | `scenarios/correlation.json`                   | `BusinessRuleViolation`  |
//! |17  | Each `block_factors[j].block_id` matches a `Block.index` in its stage  | `scenarios/load_factors.json`                  | `BusinessRuleViolation`  |
//! |18  | Load-factors entry for `(bus_id, stage_id)` with `std_mw == 0.0`       | `scenarios/load_factors.json`                  | `ModelQuality` (warning) |
//! |19  | `season_definitions` required in `stages.json` when estimating          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |20  | Minimum observations per `(hydro, season)` group for estimation         | `scenarios/inflow_history.parquet`             | `ModelQuality` (warning) |
//! |21  | All hydros in `hydros.json` must have observations in history           | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |22  | `inflow_lags: true` with PAR order > 0 requires non-empty `past_inflows` | `initial_conditions.json`                      | `BusinessRuleViolation`  |
//! |23  | Each hydro with PAR order `p` must have a `past_inflows` entry with `values_m3s.len() >= p` | `initial_conditions.json` | `BusinessRuleViolation`  |
//! |24  | All hydro IDs in `past_inflows` must exist in the hydro registry        | `initial_conditions.json`                      | `BusinessRuleViolation`  |
//! |25  | Sobol stages: `branching_factor` should be a power of 2                 | `stages.json`                                  | `ModelQuality` (warning) |
//! |26  | `simulation.sampling_scheme.type` must be a known scheme string          | `config.json`                                  | `InvalidValue`           |
//! |27  | Every stage `season_id` must reference a season defined in `season_definitions` | `stages.json`                        | `BusinessRuleViolation`  |
//! |28  | Season with zero observations when inflow scheme is not External         | `stages.json`                                  | `ModelQuality` (warning) |
//! |29  | All stages sharing a `season_id` must have compatible durations (within 7d) | `stages.json`                        | `BusinessRuleViolation`  |
//! |30  | Season defined in `season_definitions` but not referenced by any stage   | `stages.json`                                  | `ModelQuality` (warning) |
//! |31  | Observation resolution must not be finer than season resolution          | `scenarios/inflow_history.parquet`             | `BusinessRuleViolation`  |
//! |32  | Each `season_id` in `past_inflows[i].season_ids` must exist in `SeasonMap` | `initial_conditions.json`                    | `BusinessRuleViolation`  |
//! |33  | Filling schedule reaches the dead volume: `Σ ζ_s·rate_s >= min_storage − seed` | `system/hydros.json`               | `BusinessRuleViolation`  |
//! |34  | PAR order > 0 but every study stage has `inflow_lags == false` (inflow-lag state omitted) | `stages.json`        | `ModelQuality` (warning) |
//! |35  | User-supplied `inflow_ar_coefficients.parquet` must pass the periodic-ACF closure stationarity gate (external-input path only; annual-aware; season resolved via `resolve_stage_seasons`'s `season_map`-or-fallback) | `scenarios/inflow_ar_coefficients.parquet` | `InvalidValue` (or `BusinessRuleViolation` when a stage's season is genuinely unresolvable) |

use super::{ValidationContext, schema::ParsedData};

mod constraints;
mod correlation;
mod hydro;
mod pumping;
mod scenarios;
mod season;
mod sobol;
mod stages;
mod thermal;
mod travel_time;

#[cfg(test)]
mod test_support;

pub(crate) fn validate_semantic_hydro_thermal(data: &ParsedData, ctx: &mut ValidationContext) {
    hydro::check_cascade_acyclic(data, ctx);
    hydro::check_hydro_bounds(data, ctx);
    hydro::check_lifecycle_consistency(data, ctx);
    hydro::check_lifecycle_consistency_remaining(data, ctx);
    hydro::check_filling_config(data, ctx);
    hydro::check_filling_guards(data, ctx);
    hydro::check_geometry_monotonicity(data, ctx);
    hydro::check_evaporation_geometry_coverage(data, ctx);
    hydro::check_fpha_constraints(data, ctx);
    thermal::check_thermal_generation_bounds(data, ctx);
    thermal::check_anticipated_thermals(data, ctx);
    thermal::check_anticipated_cadence_transition(data, ctx);
    thermal::check_thermal_bounds_override_stage_range(data, ctx);
    thermal::check_anticipated_decision_target_is_anticipated(data, ctx);
    thermal::warn_thermal_generation_on_anticipated_thermal(data, ctx);
    constraints::check_per_block_storage_interior_reference(data, ctx);
    pumping::check_pumping_semantics(data, ctx);
    travel_time::validate_travel_time(data, ctx);
    travel_time::check_recent_observations_non_monthly_seed_gap(data, ctx);
    travel_time::check_annual_component_monthly_only(data, ctx);
}

// ── validate_semantic_stages_penalties_scenarios ──────────────────────────────

/// Performs Layer 5b semantic validation: stage structure, penalty ordering,
/// and scenario model rules. Every violation is collected into `ctx` before
/// returning — no rule short-circuits another.
///
/// # Conditional checks
///
/// Rule 12 is only checked when `data.inflow_seasonal_stats` is non-empty.
/// Rules 14-16 are only checked when `data.correlation` is `Some`.
/// Rules 17-18 are only checked when `data.load_factors` is non-empty.
pub(crate) fn validate_semantic_stages_penalties_scenarios(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    stages::check_stage_structure(data, ctx);
    stages::check_inflow_lags_vs_par_order(data, ctx);
    sobol::check_sobol_power_of_2(data, ctx);
    scenarios::check_penalty_ordering(data, ctx);
    scenarios::check_filling_sufficiency(data, ctx);
    scenarios::check_fpha_penalty_rule(data, ctx);
    scenarios::check_scenario_models(data, ctx);
    scenarios::check_par_stationarity(data, ctx);
    correlation::check_correlation_matrices(data, ctx);
    correlation::check_correlation_same_type(data, ctx);
    scenarios::check_external_scheme_has_files(data, ctx);
    scenarios::check_load_factor_consistency(data, ctx);
    scenarios::check_estimation_prerequisites(data, ctx);
    scenarios::check_past_inflows_coverage(data, ctx);
    scenarios::check_past_inflows_season_ids(data, ctx);
    season::check_season_id_consistency(data, ctx);
    season::check_observation_season_alignment(data, ctx);
}

// ── Tolerances ────────────────────────────────────────────────────────────────

const PROB_TOLERANCE: f64 = 1e-6;

const CORR_TOLERANCE: f64 = 1e-9;
