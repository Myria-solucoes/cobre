//! Shared test helpers for semantic-validation unit tests.
//!
//! This module is compiled only under `#[cfg(test)]` and is visible to all
//! sibling modules (`hydro`, `thermal`, `stages`, `sobol`, `scenarios`,
//! `correlation`, `season`) via `pub(super)`.
//!
//! Helpers that are only used by a single sibling module live inside that
//! module's own `mod tests` block to keep the blast radius small.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use cobre_core::{
    EntityId,
    entities::{Bus, Hydro, HydroGenerationModel, HydroPenalties, Line, Thermal},
    initial_conditions::InitialConditions,
    penalty::GlobalPenaltyDefaults,
    temporal::{
        BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    },
};

use crate::{
    config::Config,
    extensions::{FphaHyperplaneRow, HydroGeometryRow},
    stages::StagesData,
    validation::{ValidationContext, schema::ParsedData},
};

// ── Penalty helpers ───────────────────────────────────────────────────────────

/// Build a minimal valid `HydroPenalties` with all fields set to `v`.
pub(super) fn penalties_all(v: f64) -> HydroPenalties {
    HydroPenalties {
        spillage_cost: v,
        diversion_cost: v,
        turbined_cost: v,
        storage_violation_below_cost: v,
        filling_target_violation_cost: v,
        turbined_violation_below_cost: v,
        outflow_violation_below_cost: v,
        outflow_violation_above_cost: v,
        generation_violation_below_cost: v,
        evaporation_violation_cost: v,
        water_withdrawal_violation_cost: v,
        water_withdrawal_violation_pos_cost: v,
        water_withdrawal_violation_neg_cost: v,
        evaporation_violation_pos_cost: v,
        evaporation_violation_neg_cost: v,
        inflow_nonnegativity_cost: 1000.0,
    }
}

/// Minimal `GlobalPenaltyDefaults` required to fill `ParsedData`.
pub(super) fn minimal_global_penalties() -> GlobalPenaltyDefaults {
    use cobre_core::entities::DeficitSegment;
    GlobalPenaltyDefaults {
        bus_deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1.0,
        }],
        bus_excess_cost: 1.0,
        line_exchange_cost: 1.0,
        hydro: HydroPenalties {
            spillage_cost: 1.0,
            turbined_cost: 1.0,
            diversion_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
            water_withdrawal_violation_pos_cost: 1.0,
            water_withdrawal_violation_neg_cost: 1.0,
            evaporation_violation_pos_cost: 1.0,
            evaporation_violation_neg_cost: 1.0,
            inflow_nonnegativity_cost: 1000.0,
        },
        ncs_curtailment_cost: 1.0,
    }
}

// ── Entity builders ───────────────────────────────────────────────────────────

/// Build a minimal valid `Hydro` using default sensible values.
pub(super) fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
    Hydro {
        id: EntityId::from(id),
        name: format!("Hydro {id}"),
        bus_id: EntityId::from(1),
        downstream_id: downstream_id.map(EntityId::from),
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 1000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1000.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: penalties_all(1.0),
    }
}

/// Build a minimal valid `Thermal`.
pub(super) fn make_thermal(id: i32, min_mw: f64, max_mw: f64) -> Thermal {
    Thermal {
        id: EntityId::from(id),
        name: format!("Thermal {id}"),
        bus_id: EntityId::from(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 100.0,
        min_generation_mw: min_mw,
        max_generation_mw: max_mw,
        anticipated_config: None,
    }
}

/// Build one study stage with the given `id`.
pub(super) fn make_stage(id: i32) -> Stage {
    Stage {
        id,
        index: 0,
        start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![],
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

/// Build a minimal valid `StagesData` with the given stage IDs.
pub(super) fn make_stages(ids: Vec<i32>) -> StagesData {
    StagesData {
        stages: ids.into_iter().map(make_stage).collect(),
        policy_graph: PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            season_map: None,
        },
    }
}

// ── Layer 5a data builder (hydro + thermal) ───────────────────────────────────

/// Build a minimal `ParsedData` with the provided hydros, thermals, stages,
/// geometry, and FPHA rows.  All other fields are empty/minimal.
#[allow(clippy::too_many_arguments)]
pub(super) fn make_data(
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    lines: Vec<Line>,
    stages: StagesData,
    hydro_geometry: Vec<HydroGeometryRow>,
    fpha_hyperplanes: Vec<FphaHyperplaneRow>,
) -> ParsedData {
    ParsedData {
        config: minimal_config(),
        penalties: minimal_global_penalties(),
        stages,
        initial_conditions: InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
        },
        buses: vec![Bus {
            id: EntityId::from(1),
            name: "BUS_1".to_string(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        }],
        thermals,
        hydros,
        lines,
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry,
        production_models: vec![],
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes,
        scalar_parameters: vec![],
        inflow_history: vec![],
        inflow_seasonal_stats: vec![],
        inflow_ar_coefficients: vec![],
        inflow_annual_components: vec![],
        external_scenarios: vec![],
        external_load_scenarios: vec![],
        external_ncs_scenarios: vec![],
        load_seasonal_stats: vec![],
        load_factors: vec![],
        correlation: None,
        non_controllable_factors: vec![],
        ncs_models: vec![],
        thermal_bounds: vec![],
        hydro_bounds: vec![],
        line_bounds: vec![],
        pumping_bounds: vec![],
        contract_bounds: vec![],
        exchange_factors: vec![],
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
    }
}

// ── Layer 5b data builders (stages + penalties + scenarios) ──────────────────

/// Build a minimal valid `ParsedData` for Layer 5b tests.
/// All hydro penalties satisfy the ordering hierarchy by default.
pub(super) fn make_data_5b(
    hydros: Vec<Hydro>,
    stages: StagesData,
    buses: Vec<Bus>,
    inflow_stats: Vec<crate::scenarios::InflowSeasonalStatsRow>,
    inflow_ar: Vec<crate::scenarios::InflowArCoefficientRow>,
    correlation: Option<cobre_core::scenario::CorrelationModel>,
) -> ParsedData {
    ParsedData {
        config: minimal_config(),
        penalties: minimal_global_penalties(),
        stages,
        initial_conditions: InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
        },
        buses,
        thermals: vec![],
        hydros,
        lines: vec![],
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry: vec![],
        production_models: vec![],
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes: vec![],
        scalar_parameters: vec![],
        inflow_history: vec![],
        inflow_seasonal_stats: inflow_stats,
        inflow_ar_coefficients: inflow_ar,
        inflow_annual_components: vec![],
        external_scenarios: vec![],
        external_load_scenarios: vec![],
        external_ncs_scenarios: vec![],
        load_seasonal_stats: vec![],
        load_factors: vec![],
        correlation,
        non_controllable_factors: vec![],
        ncs_models: vec![],
        thermal_bounds: vec![],
        hydro_bounds: vec![],
        line_bounds: vec![],
        pumping_bounds: vec![],
        contract_bounds: vec![],
        exchange_factors: vec![],
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
    }
}

/// Build a hydro with penalties satisfying the ordering hierarchy.
/// filling (1000) > storage_viol (500) > constraint_viol (50) > resource (1)
pub(super) fn make_hydro_ordered_penalties(id: i32) -> Hydro {
    let mut h = make_hydro(id, None);
    h.penalties = HydroPenalties {
        filling_target_violation_cost: 1000.0,
        storage_violation_below_cost: 500.0,
        turbined_violation_below_cost: 50.0,
        outflow_violation_below_cost: 50.0,
        outflow_violation_above_cost: 50.0,
        generation_violation_below_cost: 50.0,
        evaporation_violation_cost: 50.0,
        water_withdrawal_violation_cost: 50.0,
        water_withdrawal_violation_pos_cost: 50.0,
        water_withdrawal_violation_neg_cost: 50.0,
        evaporation_violation_pos_cost: 50.0,
        evaporation_violation_neg_cost: 50.0,
        spillage_cost: 1.0,
        diversion_cost: 1.0,
        turbined_cost: 2.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    h
}

/// Build a minimal valid `StagesData` with the given stage IDs and a
/// `FiniteHorizon` policy graph with valid transitions.
pub(super) fn make_stages_5b(ids: Vec<i32>) -> StagesData {
    StagesData {
        stages: ids.into_iter().map(make_stage).collect(),
        policy_graph: PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            season_map: None,
        },
    }
}

/// Build a bus with a single deficit segment at the given cost.
pub(super) fn make_bus_with_deficit(id: i32, cost_per_mwh: f64) -> Bus {
    use cobre_core::entities::DeficitSegment;
    Bus {
        id: EntityId::from(id),
        name: format!("Bus {id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh,
        }],
        excess_cost: 100.0,
    }
}

// ── Geometry and FPHA row builders ────────────────────────────────────────────

/// Build a minimal `FphaHyperplaneRow` with the given parameters.
pub(super) fn make_fpha_row(
    hydro_id: i32,
    stage_id: Option<i32>,
    plane_id: i32,
) -> FphaHyperplaneRow {
    FphaHyperplaneRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        plane_id,
        gamma_0: 100.0,
        gamma_v: 0.5, // valid: > 0
        gamma_q: 0.8,
        gamma_s: -0.02, // valid: <= 0
        kappa: 1.0,
        valid_v_min_hm3: None,
        valid_v_max_hm3: None,
        valid_q_max_m3s: None,
    }
}

/// Build a minimal `HydroGeometryRow`.
pub(super) fn make_geom_row(
    hydro_id: i32,
    volume_hm3: f64,
    height_m: f64,
    area_km2: f64,
) -> HydroGeometryRow {
    HydroGeometryRow {
        hydro_id: EntityId::from(hydro_id),
        volume_hm3,
        height_m,
        area_km2,
    }
}

// ── Correlation helpers ───────────────────────────────────────────────────────

/// Build a valid 2x2 symmetric correlation group.
pub(super) fn make_corr_group(
    name: &str,
    matrix: Vec<Vec<f64>>,
) -> cobre_core::scenario::CorrelationGroup {
    use cobre_core::scenario::CorrelationEntity;
    cobre_core::scenario::CorrelationGroup {
        name: name.to_string(),
        entities: vec![
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(1),
            },
            CorrelationEntity {
                entity_type: "inflow".to_string(),
                id: EntityId::from(2),
            },
        ],
        matrix,
    }
}

/// Build a `CorrelationModel` with a single "default" profile containing the
/// given group.
pub(super) fn make_correlation(
    group: cobre_core::scenario::CorrelationGroup,
) -> cobre_core::scenario::CorrelationModel {
    use cobre_core::scenario::CorrelationProfile;
    use std::collections::BTreeMap;
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![group],
        },
    );
    cobre_core::scenario::CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

// ── Config helpers ────────────────────────────────────────────────────────────

/// Minimal `Config` required to fill `ParsedData`.
pub(super) fn minimal_config() -> Config {
    // Use the same JSON fragment that schema.rs tests use for config.json.
    let json = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        }
    }"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), json).unwrap();
    crate::config::parse_config(tmp.path()).unwrap()
}

/// Build a `Config` with `training.scenario_source.inflow.scheme = "external"`.
pub(super) fn config_with_training_external_inflow() -> Config {
    let json = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ],
            "scenario_source": {
                "seed": 42,
                "inflow": { "scheme": "external" }
            }
        }
    }"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), json).unwrap();
    crate::config::parse_config(tmp.path()).unwrap()
}

/// Build a `Config` with `simulation.scenario_source.load.scheme = "external"`.
pub(super) fn config_with_simulation_external_load() -> Config {
    let json = r#"{
        "training": {
            "forward_passes": 10,
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100 }
            ]
        },
        "simulation": {
            "scenario_source": {
                "seed": 7,
                "load": { "scheme": "external" }
            }
        }
    }"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), json).unwrap();
    crate::config::parse_config(tmp.path()).unwrap()
}

// ── Season / estimation data builders ────────────────────────────────────────

/// Build a monthly `SeasonMap` with 12 seasons (January=0 .. December=11).
pub(super) fn make_monthly_season_map() -> cobre_core::temporal::SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};
    let seasons = (0..12u32)
        .map(|m| SeasonDefinition {
            id: m as usize,
            label: format!("Month{m}"),
            month_start: m + 1,
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

/// Build `n_obs` `InflowHistoryRow` records for `hydro_id`, one per calendar
/// month starting from January 2000.
pub(super) fn make_history_rows(
    hydro_id: i32,
    n_obs: usize,
) -> Vec<crate::scenarios::InflowHistoryRow> {
    let mut rows = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        let date = chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap();
        rows.push(crate::scenarios::InflowHistoryRow {
            hydro_id: EntityId::from(hydro_id),
            date,
            value_m3s: 100.0,
        });
    }
    rows
}

/// Build a `StagesData` whose stages cover `n_months` monthly periods
/// starting from January 2000, each with `season_id = month_index % 12`.
/// The policy graph includes a `SeasonMap` when `with_season_map` is `true`.
pub(super) fn make_stages_with_seasons(n_months: usize, with_season_map: bool) -> StagesData {
    let mut stages = Vec::with_capacity(n_months);
    for i in 0..n_months {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        let start_date = chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let (end_year, end_month) = if month == 12 {
            (year + 1, 1u32)
        } else {
            (year, month + 1)
        };
        let end_date = chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();
        let season_id = i % 12;
        stages.push(Stage {
            index: i,
            id: i as i32,
            start_date,
            end_date,
            season_id: Some(season_id),
            blocks: vec![],
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
        });
    }
    let season_map = if with_season_map {
        Some(make_monthly_season_map())
    } else {
        None
    };
    StagesData {
        stages,
        policy_graph: PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            season_map,
        },
    }
}

/// Build `ParsedData` for estimation prerequisite tests.
///
/// `inflow_history` rows are provided directly; `inflow_seasonal_stats` is
/// empty (triggering the estimation path when history is non-empty).
pub(super) fn make_data_estimation(
    hydros: Vec<Hydro>,
    stages: StagesData,
    inflow_history: Vec<crate::scenarios::InflowHistoryRow>,
) -> ParsedData {
    ParsedData {
        config: minimal_config(),
        penalties: minimal_global_penalties(),
        stages,
        initial_conditions: cobre_core::initial_conditions::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
        },
        buses: vec![Bus {
            id: EntityId::from(1),
            name: "BUS_1".to_string(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        }],
        thermals: vec![],
        hydros,
        lines: vec![],
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry: vec![],
        production_models: vec![],
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes: vec![],
        inflow_history,
        inflow_seasonal_stats: vec![], // empty → estimation path active
        inflow_ar_coefficients: vec![],
        inflow_annual_components: vec![],
        external_scenarios: vec![],
        external_load_scenarios: vec![],
        external_ncs_scenarios: vec![],
        load_seasonal_stats: vec![],
        load_factors: vec![],
        correlation: None,
        non_controllable_factors: vec![],
        ncs_models: vec![],
        thermal_bounds: vec![],
        hydro_bounds: vec![],
        line_bounds: vec![],
        pumping_bounds: vec![],
        contract_bounds: vec![],
        exchange_factors: vec![],
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
        scalar_parameters: vec![],
    }
}

/// Build an `InflowArCoefficientRow` with the given hydro_id, stage_id, and lag.
pub(super) fn make_ar_row(
    hydro_id: i32,
    stage_id: i32,
    lag: i32,
) -> crate::scenarios::InflowArCoefficientRow {
    crate::scenarios::InflowArCoefficientRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        lag,
        coefficient: 0.5,
        residual_std_ratio: 0.9,
    }
}

/// Build a `ParsedData` suitable for rules 22-24 (past-inflows coverage) tests.
///
/// `inflow_lags_enabled` controls `state_config.inflow_lags` on stage 0.
/// `past_inflows` is placed directly in `initial_conditions`.
pub(super) fn make_data_past_inflows(
    hydros: Vec<Hydro>,
    inflow_lags_enabled: bool,
    past_inflows: Vec<cobre_core::HydroPastInflows>,
    inflow_ar_coefficients: Vec<crate::scenarios::InflowArCoefficientRow>,
) -> ParsedData {
    use cobre_core::EntityId as EId;
    let stage_0_start = chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let stage_0 = Stage {
        id: 0,
        index: 0,
        start_date: stage_0_start,
        end_date: stage_0_start
            .checked_add_months(chrono::Months::new(1))
            .unwrap_or(stage_0_start),
        season_id: None,
        blocks: vec![],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: inflow_lags_enabled,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };
    ParsedData {
        config: minimal_config(),
        penalties: minimal_global_penalties(),
        stages: StagesData {
            stages: vec![stage_0],
            policy_graph: PolicyGraph {
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                season_map: None,
            },
        },
        initial_conditions: cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows,
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
        },
        buses: vec![Bus {
            id: EId::from(1),
            name: "BUS_1".to_string(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        }],
        thermals: vec![],
        hydros,
        lines: vec![],
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry: vec![],
        production_models: vec![],
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes: vec![],
        inflow_history: vec![],
        // Populate a sentinel entry so the estimation path (rules 19-21) is
        // inactive. Rules 22-24 are independent of the estimation path.
        inflow_seasonal_stats: vec![crate::scenarios::InflowSeasonalStatsRow {
            hydro_id: EId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }],
        inflow_ar_coefficients,
        inflow_annual_components: vec![],
        external_scenarios: vec![],
        external_load_scenarios: vec![],
        external_ncs_scenarios: vec![],
        load_seasonal_stats: vec![],
        load_factors: vec![],
        correlation: None,
        non_controllable_factors: vec![],
        ncs_models: vec![],
        thermal_bounds: vec![],
        hydro_bounds: vec![],
        line_bounds: vec![],
        pumping_bounds: vec![],
        contract_bounds: vec![],
        exchange_factors: vec![],
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
        scalar_parameters: vec![],
    }
}

/// Build a `ParsedData` like `make_data_past_inflows` but with a `SeasonMap`
/// containing seasons with IDs `0..num_seasons`.
pub(super) fn make_data_past_inflows_with_season_map(
    hydros: Vec<Hydro>,
    past_inflows: Vec<cobre_core::HydroPastInflows>,
    inflow_ar_coefficients: Vec<crate::scenarios::InflowArCoefficientRow>,
    num_seasons: usize,
) -> ParsedData {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition, SeasonMap};

    let seasons = (0..num_seasons)
        .map(|i| SeasonDefinition {
            id: i,
            label: format!("Season{i}"),
            month_start: (i % 12 + 1) as u32,
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    let season_map = SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    };

    let mut data = make_data_past_inflows(hydros, true, past_inflows, inflow_ar_coefficients);
    data.stages.policy_graph.season_map = Some(season_map);
    data
}

// Suppress dead_code warnings: helpers may not be used by all sibling modules
// simultaneously but are kept together here for discoverability.
#[allow(dead_code)]
pub(super) fn _assert_helpers_present(_ctx: &mut ValidationContext) {}
