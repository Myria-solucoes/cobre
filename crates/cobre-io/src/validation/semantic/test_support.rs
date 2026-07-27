//! Shared test helpers for semantic-validation unit tests, `pub(super)` to all
//! sibling modules. Single-use helpers stay in their own module's `mod tests`
//! block to keep the blast radius small.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use chrono::NaiveDate;
use cobre_core::{
    CorrelationGroup, CorrelationModel, EntityId, SeasonMap,
    entities::{Bus, Hydro, HydroGenerationModel, HydroPenalties, HydroUnitGroup, Line, Thermal},
    initial_conditions::InitialConditions,
    penalty::GlobalPenaltyDefaults,
    temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    },
};

use crate::{
    InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow,
    config::Config,
    extensions::{FphaHyperplaneRow, HydroGeometryRow},
    parse_config,
    stages::StagesData,
    validation::schema::ParsedData,
};

// ── Penalty helpers ───────────────────────────────────────────────────────────

/// Build `HydroPenalties` with every field set to `v` except
/// `inflow_nonnegativity_cost`.
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
        hydro: penalties_all(1.0),
        ncs_curtailment_cost: 1.0,
    }
}

// ── Entity builders ───────────────────────────────────────────────────────────

/// Build a minimal valid `Hydro` using default sensible values. Does NOT
/// materialize the implicit unit group — callers that need one call
/// `normalize_unit_groups()` themselves, or route through [`make_data`] /
/// [`make_data_5b`] / [`make_data_estimation`], which normalize at the same
/// boundary `convert_hydros` and `SystemBuilder::build` do (after the
/// hydros' own field values are final).
pub(super) fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
    Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(id),
        name: format!("Hydro {id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId::from(1),
        downstream_id: downstream_id.map(EntityId::from),
        travel_time_hours: None,
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

/// Build a `HydroUnitGroup` with the given id, bus, and four bounds.
pub(super) fn make_unit_group(
    id: i32,
    bus_id: i32,
    min_generation_mw: f64,
    max_generation_mw: f64,
    min_turbined_m3s: f64,
    max_turbined_m3s: f64,
) -> HydroUnitGroup {
    HydroUnitGroup {
        id: EntityId::from(id),
        name: format!("Group {id}"),
        bus_id: EntityId::from(bus_id),
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }
}

/// Build a minimal valid `Thermal`.
pub(super) fn make_thermal(id: i32, min_mw: f64, max_mw: f64) -> Thermal {
    Thermal {
        id: EntityId::from(id),
        name: format!("Thermal {id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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

/// Build a stage with `id` and `n` blocks of equal duration, reusing
/// [`make_stage`] for every other field so the two builders cannot drift.
pub(super) fn make_stage_with_blocks(id: i32, n: usize) -> Stage {
    let mut stage = make_stage(id);
    stage.blocks = (0..n)
        .map(|index| Block {
            index,
            name: format!("B{index}"),
            duration_hours: 720.0 / n as f64,
        })
        .collect();
    stage
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

/// `ParsedData` skeleton: one `BUS_1` bus, every other field empty or `None`.
fn base_parsed_data(stages: StagesData) -> ParsedData {
    ParsedData {
        config: minimal_config(),
        penalties: minimal_global_penalties(),
        stages,
        initial_conditions: InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        },
        buses: vec![Bus {
            id: EntityId::from(1),
            name: "BUS_1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 100.0,
        }],
        thermals: vec![],
        hydros: vec![],
        lines: vec![],
        non_controllable_sources: vec![],
        pumping_stations: vec![],
        energy_contracts: vec![],
        hydro_geometry: vec![],
        production_models: vec![],
        plane_reduction: None,
        hydro_energy_productivity_rows: vec![],
        fpha_hyperplanes: vec![],
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
        generic_constraints: vec![],
        generic_constraint_bounds: vec![],
        penalty_overrides_bus: vec![],
        penalty_overrides_line: vec![],
        penalty_overrides_hydro: vec![],
        penalty_overrides_ncs: vec![],
        ncs_bounds: vec![],
    }
}

/// Build a minimal `ParsedData` with the provided hydros, thermals, stages,
/// geometry, and FPHA rows.  All other fields are empty/minimal.
///
/// Normalizes `hydros` (materializing each plant's implicit unit group when
/// none is declared) here, at the boundary — mirroring where
/// `convert_hydros` and `SystemBuilder::build` normalize in production, after
/// the hydros' own field values are final. Normalizing inside `make_hydro`
/// instead would snapshot stale bounds for any caller that mutates a plant
/// maximum afterward.
pub(super) fn make_data(
    mut hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    lines: Vec<Line>,
    stages: StagesData,
    hydro_geometry: Vec<HydroGeometryRow>,
    fpha_hyperplanes: Vec<FphaHyperplaneRow>,
) -> ParsedData {
    for hydro in &mut hydros {
        hydro.normalize_unit_groups();
    }
    ParsedData {
        thermals,
        hydros,
        lines,
        hydro_geometry,
        fpha_hyperplanes,
        ..base_parsed_data(stages)
    }
}

// ── Layer 5b data builders (stages + penalties + scenarios) ──────────────────

/// Build a minimal valid `ParsedData` for Layer 5b tests.
/// All hydro penalties satisfy the ordering hierarchy by default.
///
/// Normalizes `hydros` here, at the boundary — see [`make_data`]'s doc for
/// why this must not happen inside `make_hydro`.
pub(super) fn make_data_5b(
    mut hydros: Vec<Hydro>,
    stages: StagesData,
    buses: Vec<Bus>,
    inflow_stats: Vec<InflowSeasonalStatsRow>,
    inflow_ar: Vec<InflowArCoefficientRow>,
    correlation: Option<CorrelationModel>,
) -> ParsedData {
    for hydro in &mut hydros {
        hydro.normalize_unit_groups();
    }
    ParsedData {
        buses,
        hydros,
        inflow_seasonal_stats: inflow_stats,
        inflow_ar_coefficients: inflow_ar,
        correlation,
        ..base_parsed_data(stages)
    }
}

/// Build a hydro with penalties satisfying the ordering hierarchy.
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

pub(super) fn make_stages_5b(ids: Vec<i32>) -> StagesData {
    make_stages(ids)
}

/// Build a bus with a single deficit segment at the given cost.
pub(super) fn make_bus_with_deficit(id: i32, cost_per_mwh: f64) -> Bus {
    use cobre_core::entities::DeficitSegment;
    Bus {
        id: EntityId::from(id),
        name: format!("Bus {id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
pub(super) fn make_corr_group(name: &str, matrix: Vec<Vec<f64>>) -> CorrelationGroup {
    use cobre_core::scenario::CorrelationEntity;
    CorrelationGroup {
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
pub(super) fn make_correlation(group: CorrelationGroup) -> CorrelationModel {
    use cobre_core::scenario::CorrelationProfile;
    use std::collections::BTreeMap;
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![group],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

// ── Config helpers ────────────────────────────────────────────────────────────

/// Minimal `Config` required to fill `ParsedData`.
pub(super) fn minimal_config() -> Config {
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
    parse_config(tmp.path()).unwrap()
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
    parse_config(tmp.path()).unwrap()
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
    parse_config(tmp.path()).unwrap()
}

// ── Season / estimation data builders ────────────────────────────────────────

/// Build a monthly `SeasonMap` with 12 seasons (January=0 .. December=11).
pub(super) fn make_monthly_season_map() -> SeasonMap {
    use cobre_core::temporal::{SeasonCycleType, SeasonDefinition};
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
pub(super) fn make_history_rows(hydro_id: i32, n_obs: usize) -> Vec<InflowHistoryRow> {
    let mut rows = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let year = 2000 + (i / 12) as i32;
        let month = (i % 12) as u32 + 1;
        let start_date = NaiveDate::from_ymd_opt(year, month, 15).unwrap();
        rows.push(InflowHistoryRow {
            hydro_id: EntityId::from(hydro_id),
            start_date,
            end_date: start_date.succ_opt().unwrap(),
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
        let (end_year, end_month) = if month == 12 {
            (year + 1, 1u32)
        } else {
            (year, month + 1)
        };
        let mut stage = make_stage(i as i32);
        stage.index = i;
        stage.start_date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        stage.end_date = NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();
        stage.season_id = Some(i % 12);
        stages.push(stage);
    }
    StagesData {
        stages,
        policy_graph: PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            season_map: with_season_map.then(make_monthly_season_map),
        },
    }
}

/// Build `ParsedData` for estimation prerequisite tests.
///
/// `inflow_history` rows are provided directly; `inflow_seasonal_stats` is
/// empty (triggering the estimation path when history is non-empty).
/// Normalizes `hydros` here, at the boundary — see [`make_data`]'s doc for
/// why this must not happen inside `make_hydro`.
pub(super) fn make_data_estimation(
    mut hydros: Vec<Hydro>,
    stages: StagesData,
    inflow_history: Vec<InflowHistoryRow>,
) -> ParsedData {
    for hydro in &mut hydros {
        hydro.normalize_unit_groups();
    }
    ParsedData {
        hydros,
        inflow_history,
        ..base_parsed_data(stages)
    }
}

/// Build an `InflowArCoefficientRow` with the given hydro_id, stage_id, and lag.
pub(super) fn make_ar_row(hydro_id: i32, stage_id: i32, lag: i32) -> InflowArCoefficientRow {
    InflowArCoefficientRow {
        hydro_id: EntityId::from(hydro_id),
        stage_id,
        lag,
        coefficient: 0.5,
    }
}
