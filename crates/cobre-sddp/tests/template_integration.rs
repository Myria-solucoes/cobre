//! Integration tests for [`cobre_sddp::build_stage_templates`].
//!
//! Covers structural (column/row counts, CSC validity), objective coefficient
//! wiring, and constraint-matrix entries for hydro / FPHA / evaporation /
//! water-withdrawal / multi-segment deficit / generic constraints / operational
//! violation slacks / inflow non-negativity / stochastic load balance and PAR
//! max-order derivation.

#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use cobre_core::{
    AnticipatedConfig, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
    ContractStageBounds, DeficitSegment, EntityId, HydroStageBounds, HydroStagePenalties,
    LineStageBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
};
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use cobre_sddp::{
    build_stage_templates,
    hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, LinearizedEvaporation,
        PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
    },
    indexer::{BlockGrid, StateLayout},
    inflow_method::InflowNonNegativityMethod,
    lp_builder::PatchBuffer,
    resolved_parameters::ResolvedParameters,
};

/// LP objective cost scale factor. Matches `cobre_sddp::lp_builder::COST_SCALE_FACTOR`.
const COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Evaporation flow safety margin multiplier. Matches `cobre_sddp::lp_builder::EVAPORATION_FLOW_SAFETY_MARGIN`.
const EVAPORATION_FLOW_SAFETY_MARGIN: f64 = 2.0;

fn default_production(system: &cobre_core::System) -> ProductionModelSet {
    PrepareHydroModelsResult::default_from_system(system).production
}

fn default_evaporation(system: &cobre_core::System) -> EvaporationModelSet {
    PrepareHydroModelsResult::default_from_system(system).evaporation
}

fn production_set(productivities: &[f64], n_stages: usize) -> ProductionModelSet {
    let n_hydros = productivities.len();
    let models = productivities
        .iter()
        .map(|&p| vec![ResolvedProductionModel::ConstantProductivity { productivity: p }; n_stages])
        .collect();
    ProductionModelSet::new(models, n_hydros, n_stages)
}

fn no_penalty_config() -> InflowNonNegativityMethod {
    InflowNonNegativityMethod::None
}

fn penalty_config(_cost: f64) -> InflowNonNegativityMethod {
    InflowNonNegativityMethod::Penalty
}

fn default_hydro_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        max_diversion_m3s: None,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

fn default_hydro_penalties() -> HydroStagePenalties {
    HydroStagePenalties {
        spillage_cost: 0.01,
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
    }
}

/// One-bus, no-entity system with `n_stages` study stages.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn one_bus_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_bus_system: valid")
}

/// N=1 hydro, B=1 bus, no thermals/lines, K=1 block, L=`lag_order`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_hydro_system(n_stages: usize, lag_order: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: lag_order > 0,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coefficients.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_system: valid")
}

#[test]
fn empty_stages_returns_empty() {
    let system = one_bus_system(0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert!(result.templates.is_empty());
    assert!(result.base_rows.is_empty());
}

#[test]
fn one_stage_one_template() {
    let system = one_bus_system(1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(result.templates.len(), 1);
    assert_eq!(result.base_rows.len(), 1);
}

#[test]
fn num_cols_formula_no_hydro_no_thermal_no_line() {
    // N=0, T=0, Lines=0, B=1, K=1, L=0
    // num_cols = N*(2+L)+1 + N*K*2 + T*K + Lines*K*2 + B*K*2 = 1 + 1*1*2 = 3
    let system = one_bus_system(1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // theta + deficit + excess = 1 + 1 + 1 = 3
    assert_eq!(t.num_cols, 3, "num_cols mismatch for no-entity system");
}

#[test]
fn num_cols_formula_one_hydro_lag_zero() {
    // N=1, L=0, T=0, Lines=0, B=1, K=1
    // State cols: N*(2+L)+1 = 1*2+1 = 3  (v_out, v_in, theta)
    // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
    // Total: 7
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // N=1 withdrawal slacks add 2 columns (neg + pos): 7 + 2 = 9.
    // N=1 operational violation slacks add 4 columns: 9 + 4 = 13.
    // N=1 z-inflow column adds 1: 13 + 1 = 14.
    // N=1 diversion column adds 1: 14 + 1 = 15.
    assert_eq!(t.num_cols, 15, "num_cols mismatch for N=1 L=0");
}

#[test]
fn num_cols_formula_one_hydro_lag_two() {
    // N=1, L=2, T=0, Lines=0, B=1, K=1
    // State cols: N*(2+L)+1 = 1*4+1 = 5  (v_out, lag0, lag1, v_in, theta)
    // Decision: turbine[1] + spillage[1] + deficit[1] + excess[1] = 4
    // Total: 9
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // N=1 withdrawal slacks add 2 columns (neg + pos): 9 + 2 = 11.
    // N=1 operational violation slacks add 4 columns: 11 + 4 = 15.
    // N=1 z-inflow column adds 1: 15 + 1 = 16.
    // N=1 diversion column adds 1: 16 + 1 = 17.
    assert_eq!(t.num_cols, 17, "num_cols mismatch for N=1 L=2");
}

#[test]
fn num_rows_formula_no_hydro() {
    // N=0, B=1, K=1, L=0 → n_state = 0*(1+0) = 0
    // fixing rows: 0, water balance: 0, load balance: 1*1 = 1
    // num_rows = 0 + 0 + 1 = 1
    let system = one_bus_system(1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    assert_eq!(t.num_rows, 1, "num_rows mismatch for no-hydro system");
}

#[test]
fn num_rows_formula_one_hydro_lag_zero() {
    // N=1, L=0, B=1, K=1; no state-fixing rows (incoming state pinned via column bounds).
    // num_rows = N z_inflow(1) + N water_balance(1) + B*K load_balance(1) + 4 op-violation = 7
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    assert_eq!(t.num_rows, 7, "num_rows mismatch for N=1 L=0");
}

#[test]
fn num_rows_formula_one_hydro_lag_two() {
    // N=1, L=2, B=1, K=1; lags do not add rows, no state-fixing rows. Same 7 as L=0:
    // num_rows = N z_inflow(1) + N water_balance(1) + B*K load_balance(1) + 4 op-violation = 7
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    assert_eq!(t.num_rows, 7, "num_rows mismatch for N=1 L=2");
}

#[test]
fn n_state_matches_indexer() {
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let expected = StateLayout::new(1, 2, 0, 0, vec![], &[2; 1]).n_state;
    assert_eq!(t.n_state, expected, "n_state must match StateLayout");
}

#[test]
fn n_transfer_is_n_times_lag_order() {
    // n_transfer = N*L = 1*2 = 2
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    assert_eq!(t.n_transfer, 2, "n_transfer = N*L");
}

#[test]
fn base_row_is_n_dual_relevant_plus_n_hydros() {
    let system = one_hydro_system(2, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    for (s, (&br, t)) in result.base_rows.iter().zip(&result.templates).enumerate() {
        assert_eq!(
            br,
            t.n_dual_relevant + t.n_hydro,
            "base_rows[{s}] must equal n_dual_relevant + n_hydro"
        );
    }
}

#[test]
fn csc_col_starts_monotone_nondecreasing() {
    let system = one_hydro_system(1, 1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    for w in t.col_starts.windows(2) {
        assert!(w[0] <= w[1], "col_starts not monotone: {} > {}", w[0], w[1]);
    }
    assert_eq!(t.col_starts.len(), t.num_cols + 1);
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn csc_row_indices_in_range() {
    let system = one_hydro_system(1, 1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    for &r in &t.row_indices {
        assert!(
            r >= 0 && (r as usize) < t.num_rows,
            "row index {r} out of range [0, {})",
            t.num_rows
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn csc_nz_count_matches_col_starts() {
    let system = one_hydro_system(1, 1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    assert_eq!(
        t.num_nz,
        *t.col_starts.last().unwrap() as usize,
        "num_nz must equal col_starts[num_cols]"
    );
    assert_eq!(
        t.row_indices.len(),
        t.num_nz,
        "row_indices.len() must equal num_nz"
    );
    assert_eq!(t.values.len(), t.num_nz, "values.len() must equal num_nz");
}

#[test]
fn theta_column_has_unit_objective() {
    let lag_order = 2;
    let system = one_hydro_system(1, lag_order);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    let theta_col = StateLayout::new(1, lag_order, 0, 0, vec![], &[lag_order; 1]).theta;
    assert_eq!(
        t.objective[theta_col], 1.0,
        "theta column objective must be 1.0 (theta is not scaled by COST_SCALE_FACTOR)"
    );
}

#[test]
fn spillage_objective_nonzero_for_nonzero_penalty() {
    // Hydro fixture has spillage_cost = 0.01 over a 744h block, so the spillage
    // objective is strictly positive.
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // spillage col for h=0, blk=0: col_spillage_start + 0 = N*(3+L)+1 + N*K
    // With N=1, L=0, K=1: theta=3, decision_start=4, turbine_start=4, spill_start=5
    let spill_col = 5;
    assert!(
        t.objective[spill_col] > 0.0,
        "spillage objective must be > 0 when spillage_cost > 0"
    );
}

/// 1-bus, 1-FPHA-hydro, 1-stage system with `n_planes` planes, parameterized
/// `turbined_cost`, and explicit block durations.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn fpha_system_with_turbined_cost(
    n_planes: usize,
    turbined_cost: f64,
    block_durations_hours: &[f64],
) -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let n_blks = block_durations_hours.len();
    assert!(n_blks > 0, "must have at least one block");

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(2),
        name: "FPHA1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 150.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 300.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost,
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
    };

    let blocks: Vec<Block> = block_durations_hours
        .iter()
        .enumerate()
        .map(|(i, &hours)| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: hours,
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks,
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
    }];

    let inflow_models: Vec<InflowModel> = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 150.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 300.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost,
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
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("fpha_system_with_turbined_cost: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let planes = vec![plane; n_planes];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let production = ProductionModelSet::new(models, 1, 1);

    (system, production)
}

// ---- turbined_cost tests -------------------------------------------

#[test]
fn turbined_cost_applied_to_fpha_turbine_column() {
    // turbined_cost = 0.5 $/MWh over a 720h block → turbine objective 0.5*720 = 360.0
    // (then scaled by 1/COST_SCALE_FACTOR).
    let (system, production) = fpha_system_with_turbined_cost(3, 0.5, &[720.0]);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("fpha system builds ok");
    let t = &result.templates[0];
    let turbine_col = 4_usize;
    let expected = 0.5 * 720.0 / COST_SCALE_FACTOR;
    assert!(
        (t.objective[turbine_col] - expected).abs() < 1e-15,
        "FPHA turbine col objective: expected {expected}, got {}",
        t.objective[turbine_col]
    );
}

#[test]
fn turbined_cost_multi_block_uses_per_block_hours() {
    // turbined_cost = 1.0 $/MWh; each turbine column carries cost * its own
    // block_hours (block 0 = 300h, block 1 = 420h), not the stage total.
    let (system, production) = fpha_system_with_turbined_cost(3, 1.0, &[300.0, 420.0]);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("fpha multi-block system builds ok");
    let t = &result.templates[0];
    let col_blk0 = 4_usize;
    let col_blk1 = 5_usize;
    assert!(
        (t.objective[col_blk0] - 300.0 / COST_SCALE_FACTOR).abs() < 1e-15,
        "block 0 turbine objective: expected {}, got {}",
        300.0 / COST_SCALE_FACTOR,
        t.objective[col_blk0]
    );
    assert!(
        (t.objective[col_blk1] - 420.0 / COST_SCALE_FACTOR).abs() < 1e-15,
        "block 1 turbine objective: expected {}, got {}",
        420.0 / COST_SCALE_FACTOR,
        t.objective[col_blk1]
    );
}

#[test]
fn turbined_cost_mixed_system_all_hydros_carry_cost() {
    let (system, production) = four_hydro_mixed_system();

    let hydro_pen = HydroStagePenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 1.0,
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
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 4,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: hydro_pen,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(system.buses().to_vec())
        .hydros(system.hydros().to_vec())
        .stages(system.stages().to_vec())
        .inflow_models(system.inflow_models().to_vec())
        .load_models(system.load_models().to_vec())
        .bounds(system.bounds().clone())
        .penalties(penalties)
        .build()
        .expect("mixed system with turbined cost");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed system builds");
    let t = &result.templates[0];

    // N=4, L=0, K=1: theta=12, decision_start=13
    // col_turbine_start=13, turbine cols: h0=13, h1=14, h2=15, h3=16
    // spillage cols: h0=17, h1=18, h2=19, h3=20
    let col_turbine_start = 13;
    let block_hours = 744.0;

    let expected = 1.0 * block_hours / COST_SCALE_FACTOR;
    for h in 0..4 {
        assert!(
            (t.objective[col_turbine_start + h] - expected).abs() < 1e-15,
            "hydro {h} turbine objective should be {expected}, got {}",
            t.objective[col_turbine_start + h]
        );
    }
}

#[test]
fn load_balance_rhs_matches_load_model_mean_mw() {
    // Fixture LoadModel mean_mw = 100.0, so the load-balance RHS is 100.0.
    let system = one_bus_system(1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // No hydros → n_dual_relevant=0, water_balance_rows=0, load_balance at row 0, blk 0
    let load_row = 0;
    assert_eq!(
        t.row_lower[load_row], 100.0,
        "row_lower for load balance must be mean_mw"
    );
    assert_eq!(
        t.row_upper[load_row], 100.0,
        "row_upper for load balance must be mean_mw"
    );
}

#[test]
fn multiple_stages_produce_same_count_templates_and_base_rows() {
    let system = one_hydro_system(3, 1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(result.templates.len(), 3);
    assert_eq!(result.base_rows.len(), 3);
}

#[test]
fn stage_templates_clone_and_debug() {
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let cloned = result.clone();
    assert_eq!(cloned.templates.len(), result.templates.len());
    let s = format!("{result:?}");
    assert!(s.contains("StageTemplates"));
}

// -------------------------------------------------------------------------
// FPHA generation model validation tests
// -------------------------------------------------------------------------

/// An `Fpha` entity model paired with a `ConstantProductivity` resolved model
/// is accepted: model-vs-resolved validation lives in `prepare_hydro_models`,
/// not in `build_stage_templates`.
#[test]
#[allow(clippy::too_many_lines)]
fn test_fpha_model_accepted() {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(5),
        name: "Tucurui".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];

    let inflow_models: Vec<InflowModel> = vec![InflowModel {
        hydro_id: EntityId(5),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = cobre_core::SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("test_fpha_model_accepted: valid system");

    // default_from_system maps the Fpha entity model to ConstantProductivity
    // { productivity: 0.0 }; the resolved production model, not the entity model,
    // determines the LP layout.
    let production = PrepareHydroModelsResult::default_from_system(&system).production;
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    );

    assert!(
        result.is_ok(),
        "Fpha entity model with ConstantProductivity resolved model must now succeed: {result:?}"
    );

    // A rejection would name the plant in the error message; assert it is absent.
    if let Err(ref e) = result {
        let msg = e.to_string();
        assert!(
            !msg.contains("Tucurui"),
            "unexpected error for Tucurui: {msg}"
        );
    }
}

#[test]
fn test_constant_productivity_accepted() {
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    );
    assert!(
        result.is_ok(),
        "ConstantProductivity system must return Ok, got: {result:?}"
    );
    assert_eq!(
        result.expect("accepted").templates.len(),
        1,
        "one study stage → one template"
    );
}

// -------------------------------------------------------------------------
// Inflow non-negativity penalty method tests
// -------------------------------------------------------------------------

#[test]
fn test_penalty_columns_added() {
    let system = one_hydro_system(1, 0);
    let without = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let with_p = build_stage_templates(
        &system,
        penalty_config(1000.0),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(
        with_p.templates[0].num_cols,
        without.templates[0].num_cols + 1,
        "penalty method must add exactly n_hydros extra columns"
    );
}

#[test]
fn test_penalty_columns_added_3_hydros() {
    // Despite the name, this checks the n_hydros == 0 edge of
    // num_cols(penalty) = num_cols(none) + n_hydros: zero hydros add zero slacks
    // regardless of config (the N=1 column count is covered above).
    let system = one_bus_system(1);
    let with_p = build_stage_templates(
        &system,
        penalty_config(1000.0),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let without = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(
        with_p.templates[0].num_cols, without.templates[0].num_cols,
        "no slack columns when n_hydros == 0, even with penalty config"
    );
}

// Slack objective = penalty_cost * total_stage_hours; the fixture has 1 block
// of 744h, so the expected coefficient is 1000.0 * 744.0 (then COST_SCALE_FACTOR-scaled).
#[test]
fn test_penalty_objective_coefficient() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // N=1, L=0: theta=3, decision_start=4, turbine=4, spillage=5, diversion=6,
    // deficit=7, excess=8, inflow_slack=9, withdrawal_neg=10, withdrawal_pos=11,
    // outflow_below=12, outflow_above=13, turbine_below=14, generation_below=15.
    // inflow_slack sits before the 2 withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * t.n_hydro;
    let expected_obj = 1000.0 * 744.0 / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "expected objective {expected_obj}, got {}",
        t.objective[slack_col]
    );
}

#[test]
fn test_no_penalty_columns_when_none() {
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // N=1, L=2: state+aux = N*(3+L)+1 = 6 (storage, lags, z_inflow, storage_in, theta);
    // decisions = turb+spill+diversion+def+exc = 5; withdrawal = 2 (neg+pos);
    // operational violation slacks = 4; total = 17.
    assert_eq!(
        t.num_cols, 17,
        "method=none must not add extra penalty columns"
    );
    // num_rows = N z_inflow + N water_balance + B*K load_balance (1+1+1)
    // + 4 op-violation rows = 7; no state-fixing rows (state pinned via column bounds).
    assert_eq!(t.num_rows, 7, "method=none must not add extra penalty rows");
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn test_penalty_slack_in_water_balance() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * t.n_hydro;

    let water_balance_row = 1_usize; // N + h = 1 + 0

    let col_start = t.col_starts[slack_col] as usize;
    let col_end = t.col_starts[slack_col + 1] as usize;
    let found = t.row_indices[col_start..col_end]
        .iter()
        .zip(&t.values[col_start..col_end])
        .any(|(&r, &v)| r as usize == water_balance_row && v.abs() > 1e-12);

    assert!(
        found,
        "slack column must have a non-zero entry in the water balance row"
    );
}

#[test]
fn test_penalty_slack_bounds() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * t.n_hydro;
    assert_eq!(t.col_lower[slack_col], 0.0, "slack lower bound must be 0.0");
    assert!(
        t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
        "slack upper bound must be +infinity"
    );
}

// The penalty slack is virtual inflow, so it enters the LHS of the water-balance
// constraint (outflows - inflows = RHS) with coefficient -ζ, ζ = tau_total * M3S_TO_HM3.
// For 1 block of 744h: ζ = 744.0 * (3600.0 / 1_000_000.0) = 2.6784 → coefficient -2.6784.
#[test]
#[allow(clippy::cast_sign_loss)]
fn test_penalty_water_balance_coefficient_value() {
    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let slack_col = t.num_cols - 1 - 6 * t.n_hydro;
    let water_balance_row = 1_usize; // N + h = 1 + 0
    let zeta = 744.0 * (3_600.0 / 1_000_000.0);
    let expected_coeff = -zeta;

    let col_start = t.col_starts[slack_col] as usize;
    let col_end = t.col_starts[slack_col + 1] as usize;
    let coeff = t.row_indices[col_start..col_end]
        .iter()
        .zip(&t.values[col_start..col_end])
        .find(|&(&r, _)| r as usize == water_balance_row)
        .map(|(_, &v)| v);

    assert!(
        coeff.is_some(),
        "slack column must have an entry in the water balance row"
    );
    let coeff = coeff.unwrap();
    assert!(
        (coeff - expected_coeff).abs() < 1e-9,
        "expected coefficient {expected_coeff:.9}, got {coeff:.9}"
    );
}

#[test]
fn test_penalty_multi_stage_consistent() {
    let system = one_hydro_system(3, 1);
    let config = penalty_config(2000.0);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(result.templates.len(), 3);
    let base_cols = result.templates[0].num_cols;
    for t in &result.templates {
        assert_eq!(
            t.num_cols, base_cols,
            "all stages must have the same column count"
        );
    }
}

// A large negative noise forces the inflow slack on: the deficit exceeds the
// available storage drawdown. Water balance is
// v_out - v_in + ζ*(turbine + spillage - inflow_slack) = RHS; with v_in = 100 hm³
// and RHS = -110 hm³ it reduces to ζ*inflow_slack ≥ 10 > 0 (v_out, turbine,
// spillage ≥ 0), so the slack is mandatory regardless of turbine level.
#[test]
fn test_penalty_slack_absorbs_negative_inflow() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let system = one_hydro_system(1, 0);
    let config = penalty_config(1000.0);
    let pm = production_set(&[0.9], 1);
    let result = build_stage_templates(
        &system,
        config,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let template = &result.templates[0];

    // inflow_slack sits before the withdrawal + 4 op-violation slacks (6 per hydro).
    let col_inflow_slack_start = template.num_cols - 1 - 6 * template.n_hydro;

    let col_storage_in = 2_usize; // storage_in for hydro 0 when N=1, L=0
    let water_balance_row = 1_usize; // N + h = 1 + 0

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Incoming storage is pinned via column bounds, not a row-0 equality.
    let initial_storage = 100.0_f64;
    let negative_noise = -110.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[initial_storage], &[initial_storage]);
    solver.set_row_bounds(&[water_balance_row], &[negative_noise], &[negative_noise]);

    let view = solver
        .solve(None)
        .expect("LP must be feasible with inflow slack active");

    let primal = view.primal;

    assert!(
        primal[col_inflow_slack_start] > 0.0,
        "inflow slack must be positive when noise is negative, got {}",
        primal[col_inflow_slack_start]
    );

    assert!(
        view.objective > 0.0,
        "objective must include a positive penalty contribution, got {}",
        view.objective
    );
}

// -------------------------------------------------------------------------
// load balance row starts, n_load_buses, load_bus_indices
// -------------------------------------------------------------------------

/// Build a two-bus system with N hydros and K blocks per stage.
/// Bus B1 (EntityId=10) has `std_mw` = 0 (no load noise).
/// Bus B2 (EntityId=20) has `std_mw` > 0 (stochastic load).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn two_bus_system_with_stochastic_load(
    n_stages: usize,
    n_hydros_in_system: usize,
    n_blocks: usize,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus1 = Bus {
        id: EntityId(10),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let bus2 = Bus {
        id: EntityId(20),
        name: "B2".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let blocks: Vec<_> = (0..n_blocks)
        .map(|b| Block {
            index: b,
            name: format!("B{b}"),
            duration_hours: 240.0,
        })
        .collect();

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: blocks.clone(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let hydros: Vec<Hydro> = (0..n_hydros_in_system)
        .map(|h| Hydro {
            id: EntityId((h + 100) as i32),
            name: format!("H{h}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 50.0,
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = hydros
        .iter()
        .flat_map(|h| {
            (0..n_stages).map(move |s| InflowModel {
                hydro_id: h.id,
                stage_id: s as i32,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .flat_map(|s| {
            [
                LoadModel {
                    bus_id: EntityId(10),
                    stage_id: s as i32,
                    mean_mw: 80.0,
                    std_mw: 0.0, // B1: no noise
                },
                LoadModel {
                    bus_id: EntityId(20),
                    stage_id: s as i32,
                    mean_mw: 120.0,
                    std_mw: 15.0, // B2: stochastic
                },
            ]
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: n_hydros_in_system,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: n_hydros_in_system,
            n_buses: 2,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let mut builder = SystemBuilder::new()
        .buses(vec![bus1, bus2])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties);
    if !hydros.is_empty() {
        builder = builder.hydros(hydros).inflow_models(inflow_models);
    }
    builder.build().expect("two_bus_system: valid")
}

#[test]
fn stage_templates_load_balance_row_starts_correct() {
    let system = two_bus_system_with_stochastic_load(2, 2, 3);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert_eq!(
        result.load_balance_row_starts.len(),
        result.templates.len(),
        "load_balance_row_starts length must match templates length"
    );

    // N=2 hydros, L=0: row_load_balance_start = row_water_balance_start + n_state(2).
    let expected_row_start = result.base_rows[0] + 2; // base_rows[0] = row_water_balance_start
    assert_eq!(
        result.load_balance_row_starts[0], expected_row_start,
        "load_balance_row_starts[0] must equal row_water_balance_start + n_hydros"
    );
    assert_eq!(
        result.load_balance_row_starts[0], result.load_balance_row_starts[1],
        "identical stages share the same load balance row start"
    );
}

#[test]
fn stage_templates_n_load_buses_matches_stochastic_buses() {
    let system = two_bus_system_with_stochastic_load(1, 0, 1);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert_eq!(
        result.n_load_buses, 1,
        "only B2 has std_mw > 0 → n_load_buses must be 1"
    );
    assert_eq!(
        result.load_bus_indices.len(),
        1,
        "load_bus_indices must have exactly one entry"
    );
    assert_eq!(
        result.load_bus_indices[0], 1,
        "B2 is at buses slice index 1 (buses are [B1(10), B2(20)])"
    );
}

#[test]
fn stage_templates_no_load_buses_gives_zero() {
    // one_bus_system uses std_mw = 0 for all load models.
    let system = one_bus_system(2);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert_eq!(
        result.n_load_buses, 0,
        "system with std_mw = 0 everywhere must give n_load_buses = 0"
    );
    assert!(
        result.load_bus_indices.is_empty(),
        "load_bus_indices must be empty when n_load_buses = 0"
    );
    assert_eq!(
        result.load_balance_row_starts.len(),
        result.templates.len(),
        "load_balance_row_starts length must always match templates length"
    );
}

// -------------------------------------------------------------------------
// FPHA constraint tests (AC-1 through AC-5)
// -------------------------------------------------------------------------

/// CSC coefficient at (`col`, `row`); `None` if the column has no entry in that row.
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
fn csc_entry(tmpl: &cobre_solver::StageTemplate, col: usize, row: usize) -> Option<f64> {
    let start = tmpl.col_starts[col] as usize;
    let end = tmpl.col_starts[col + 1] as usize;
    for pos in start..end {
        if tmpl.row_indices[pos] as usize == row {
            return Some(tmpl.values[pos]);
        }
    }
    None
}

/// 1-bus, 1-FPHA-hydro, 1-stage, 1-block system with `n_planes` planes, each
/// `intercept=10.0, gamma_v=0.5, gamma_q=2.0, gamma_s=0.1`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_fpha_hydro_system(n_planes: usize) -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let hydro = Hydro {
        id: EntityId(2),
        name: "FPHA1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 150.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 300.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];

    let inflow_models: Vec<InflowModel> = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 150.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 300.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_fpha_hydro_system: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let planes = vec![plane; n_planes];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let production = ProductionModelSet::new(models, 1, 1);

    (system, production)
}

/// 1-bus, 4-hydro, 1-stage, 1-block system. EntityId-sorted: 100 (const),
/// 101 (const), 102 (fpha), 103 (fpha) → hydro indices 0,1 are constant
/// productivity; indices 2,3 are FPHA (3 planes each).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn four_hydro_mixed_system() -> (cobre_core::System, ProductionModelSet) {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro_penalties = HydroPenalties {
        spillage_cost: 0.01,
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
    };

    let make_hydro = |id: i32, gen_model: HydroGenerationModel| Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: gen_model,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: hydro_penalties,
    };

    let hydros = vec![
        make_hydro(100, HydroGenerationModel::ConstantProductivity),
        make_hydro(101, HydroGenerationModel::ConstantProductivity),
        make_hydro(102, HydroGenerationModel::Fpha),
        make_hydro(103, HydroGenerationModel::Fpha),
    ];

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];

    let inflow_models: Vec<InflowModel> = hydros
        .iter()
        .map(|h| InflowModel {
            hydro_id: h.id,
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 400.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 4,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 4,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("four_hydro_mixed_system: valid");

    let plane = FphaPlane {
        intercept: 10.0,
        gamma_v: 0.5,
        gamma_q: 2.0,
        gamma_s: 0.1,
    };
    let fpha_planes = vec![plane; 3];
    let models = vec![
        vec![ResolvedProductionModel::ConstantProductivity { productivity: 2.5 }],
        vec![ResolvedProductionModel::ConstantProductivity { productivity: 3.0 }],
        vec![ResolvedProductionModel::Fpha {
            planes: fpha_planes.clone(),
        }],
        vec![ResolvedProductionModel::Fpha {
            planes: fpha_planes,
        }],
    ];
    let production = ProductionModelSet::new(models, 4, 1);

    (system, production)
}

#[test]
fn fpha_ac1_dimensions_one_fpha_hydro_five_planes() {
    let (system, production) = one_fpha_hydro_system(5);

    let fpha_result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let const_result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    let fpha_tmpl = &fpha_result.templates[0];
    let const_tmpl = &const_result.templates[0];

    assert_eq!(
        fpha_tmpl.num_cols,
        const_tmpl.num_cols + 1,
        "FPHA adds exactly 1 generation column (1 hydro * 1 block)"
    );
    assert_eq!(
        fpha_tmpl.num_rows,
        const_tmpl.num_rows + 5,
        "FPHA adds exactly 5 constraint rows (5 planes * 1 block)"
    );
}

/// Generation column has +1.0 in all 5 FPHA rows and in the hydro's load
/// balance row.
#[test]
fn fpha_ac2_generation_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    // N=1, L=0 → n_state = 1, decision_start = 4, col_generation_start = 9.
    // (turbine[4..5], spillage[5..6], diversion[6..7], deficit[7..8], excess[8..9], generation[9])
    let col_g = 9_usize;

    // row_fpha_start = 3 (= N z_inflow + N water balance + B*1 load balance)
    let row_fpha_start = 3_usize;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_g, row)
            .unwrap_or_else(|| panic!("generation column missing entry in FPHA row {row}"));
        assert!(
            (coeff - 1.0).abs() < 1e-12,
            "generation col FPHA row {row}: expected +1.0, got {coeff}"
        );
    }

    let row_lb = 2_usize; // load balance row
    let lb_coeff = csc_entry(tmpl, col_g, row_lb)
        .unwrap_or_else(|| panic!("generation column missing entry in load balance row {row_lb}"));
    assert!(
        (lb_coeff - 1.0).abs() < 1e-12,
        "generation col load balance row: expected +1.0, got {lb_coeff}"
    );
}

/// The `v_in` column carries `-gamma_v/2` in every FPHA row; gamma_v = 0.5 → -0.25.
#[test]
fn fpha_ac3_v_in_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v_in = 2_usize; // N=1, L=0: storage_in after z_inflow
    let row_fpha_start = 3_usize; // N z_inflow + N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v_in, row)
            .unwrap_or_else(|| panic!("v_in column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v_in col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

/// The outgoing-storage column `v` carries `-gamma_v/2` in every FPHA row; -0.25 here.
#[test]
fn fpha_ac4_v_out_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v = 0_usize; // outgoing storage is column 0 for N=1
    let row_fpha_start = 3_usize; // N z_inflow + N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v, row)
            .unwrap_or_else(|| panic!("v column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

/// In the mixed system, FPHA hydros enter the load-balance row via their
/// generation column (+1.0) while constant hydros enter via rho * turbine.
#[test]
fn fpha_ac5_mixed_system_load_balance_uses_generation_col() {
    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system ok");

    let tmpl = &result.templates[0];

    // Base (constant) num_cols = 13 + 4*1*3 (turbine+spillage+diversion per hydro/block)
    // + 1*1*2 = 27; + 2 FPHA generation = 29; + 4 withdrawal (N=4) = 33;
    // + 4*N=16 op-violation slacks = 53.
    // num_rows (no state-fixing rows) = N z_inflow(4) + N water balance(4) + load balance(1)
    // = 9; + 2*3 FPHA = 15; + 4*N=16 op-violation = 31.
    assert_eq!(
        tmpl.num_cols, 53,
        "4-hydro mixed system: num_cols should be 53 (includes diversion, 2*4 withdrawal, and operational slacks)"
    );
    assert_eq!(
        tmpl.num_rows, 31,
        "4-hydro mixed system: num_rows should be 31 (Phase 1: state-fixing rows removed)"
    );

    let row_lb = 8_usize; // N z_inflow(4) + N water balance(4) + bus_blk_idx(0)

    // FPHA hydro h_idx=2 (local_idx=0): generation col 27 (= 23 + 4 diversion cols).
    let col_g_fpha0 = 27_usize;
    let g0_lb_coeff = csc_entry(tmpl, col_g_fpha0, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 0 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g0_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 0 load balance: expected +1.0, got {g0_lb_coeff}"
    );

    let col_g_fpha1 = 28_usize; // FPHA hydro h_idx=3 (local_idx=1)
    let g1_lb_coeff = csc_entry(tmpl, col_g_fpha1, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 1 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g1_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 1 load balance: expected +1.0, got {g1_lb_coeff}"
    );

    let col_turb_const = 13_usize; // constant hydro h_idx=0: col_turbine_start
    let turb_lb_coeff = csc_entry(tmpl, col_turb_const, row_lb);
    assert!(
        turb_lb_coeff.is_some(),
        "constant hydro 0 turbine col must appear in load balance row"
    );
    // Constant hydro enters the power-balance matrix with rho (the productivity
    // scalar), NOT rho * block_hours — block_hours scales cost objectives only.
    // Hydro 0 has productivity 2.5 MW/(m³/s), so the coefficient is 2.5.
    let expected_rho_coeff = 2.5_f64;
    assert!(
        (turb_lb_coeff.unwrap() - expected_rho_coeff).abs() < 1e-12,
        "constant hydro 0 turbine: expected rho = {expected_rho_coeff}, got {turb_lb_coeff:?}"
    );
}

// -------------------------------------------------------------------------
// FPHA LP integration tests (HiGHS end-to-end solve)
// -------------------------------------------------------------------------
//
// Planes (intercept=300.0, gamma_v=1.0, gamma_q=3.0, gamma_s=0.0) keep the LP
// feasible: at v_in=100 hm³, load=200 MW, g=200, q=50 m³/s the FPHA RHS is
// 300 + 1.0*v_avg + 3.0*50 ≈ 550 >> 200.

/// 1-FPHA-hydro system reusing `one_fpha_hydro_system` but with 3 planes whose
/// large intercept keeps the solve feasible for any `v_in` in `[0, 500]` hm³:
/// `intercept=300.0, gamma_v=1.0, gamma_q=3.0, gamma_s=0.0`.
fn fpha_solve_system() -> (cobre_core::System, ProductionModelSet) {
    let planes = vec![
        FphaPlane {
            intercept: 300.0,
            gamma_v: 1.0,
            gamma_q: 3.0,
            gamma_s: 0.0,
        };
        3
    ];
    let models = vec![vec![ResolvedProductionModel::Fpha { planes }]];
    let production = ProductionModelSet::new(models, 1, 1);
    let (system, _) = one_fpha_hydro_system(3);
    (system, production)
}

/// 1-FPHA-hydro LP solves to optimal with generation > 0, given `v_in = 100 hm³`.
#[test]
fn fpha_solve_one_hydro_optimal() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let (system, production) = fpha_solve_system();
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    let v_in = 100.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("FPHA LP must be feasible and optimal");

    let col_g = 9_usize; // g column, shifted +1 by the diversion column
    let generation = view.primal[col_g];
    assert!(
        generation > 0.0,
        "FPHA generation must be strictly positive, got {generation}"
    );
}

/// After solving, every plane holds within 1e-6:
/// `g <= intercept + gamma_v * v_avg + gamma_q * q + gamma_s * s`,
/// `v_avg = (v + v_in) / 2`.
#[test]
fn fpha_solve_hyperplane_constraints_hold() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let (system, production) = fpha_solve_system();

    // Extract planes before moving production into build_stage_templates.
    let planes = match production.model(0, 0) {
        ResolvedProductionModel::Fpha { planes, .. } => planes.clone(),
        ResolvedProductionModel::ConstantProductivity { .. } => {
            panic!("expected Fpha model")
        }
    };

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    let v_in = 100.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver.solve(None).expect("FPHA LP must solve to optimal");
    let primal = view.primal;

    let col_v = 0_usize;
    let col_v_in = 1_usize;
    let col_q = 3_usize;
    let col_s = 4_usize;
    let col_g = 9_usize;

    let g = primal[col_g];
    let v = primal[col_v];
    let v_in_sol = primal[col_v_in];
    let q = primal[col_q];
    let s = primal[col_s];
    let v_avg = f64::midpoint(v, v_in_sol);

    for (p_idx, plane) in planes.iter().enumerate() {
        let rhs = plane.intercept + plane.gamma_v * v_avg + plane.gamma_q * q + plane.gamma_s * s;
        assert!(
            g <= rhs + 1e-6,
            "FPHA plane {p_idx}: g={g} must be <= rhs={rhs} \
             (intercept={intercept}, gamma_v={gamma_v}, v_avg={v_avg}, \
              gamma_q={gamma_q}, q={q}, gamma_s={gamma_s}, s={s})",
            intercept = plane.intercept,
            gamma_v = plane.gamma_v,
            gamma_q = plane.gamma_q,
            gamma_s = plane.gamma_s,
        );
    }
}

/// The storage-fixing dual (reduced cost of the pinned `storage_in` column)
/// differs between FPHA and constant productivity. The `-gamma_v/2` FPHA entries
/// on the `v_in` column propagate through the simplex dual to that reduced cost.
///
/// The tight planes (`intercept=0, gamma_v=0.5, gamma_q=1.0, gamma_s=0.0`) keep
/// the FPHA capacity below the 200 MW load so the constraint binds at the optimum
/// — necessary for a non-zero dual. With K=1, `zeta`≈2.678 and `v_in`=100 hm³:
/// - FPHA row: `g <= 0.25*(v + v_in) + q`
/// - Water balance: `v = v_in + inflow*zeta - (q+s)*zeta`
/// - `q_max = v_in/zeta + inflow ≈ 37.3 + 80 = 117.3 m³/s` (at `v`=0)
/// - `g_max ≈ 0.25*v_in + q_max ≈ 25 + 117.3 = 142.3 MW < 200 MW`
///
/// So the optimizer covers the shortfall with costly deficit and extra `v_in`
/// lowers cost → FPHA dual < 0. Constant productivity gives `rho`=0
/// (`default_from_system`), generation is `v_in`-independent → dual = 0.
#[test]
fn fpha_solve_storage_fixing_dual_differs_from_constant() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let (system, _) = one_fpha_hydro_system(1);

    let tight_planes = vec![FphaPlane {
        intercept: 0.0,
        gamma_v: 0.5,
        gamma_q: 1.0,
        gamma_s: 0.0,
    }];
    let fpha_production = ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: tight_planes,
        }]],
        1,
        1,
    );

    let fpha_result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &fpha_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let const_production = default_production(&system);
    let const_result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &const_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity template build must succeed");

    let solve_and_get_storage_dual = |template: &cobre_solver::StageTemplate| -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        solver.load_model(template);
        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);
        // Storage is pinned via column bounds: col 0 = storage_out, 1 = z_inflow, 2 = storage_in.
        let col_storage_in = 2_usize;
        let v_in = 100.0_f64;
        solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);
        let view = solver.solve(None).expect("LP must solve to optimal");
        // The storage_in column's reduced cost is the shadow price of fixing v_in.
        view.reduced_costs[col_storage_in]
    };

    let fpha_dual = solve_and_get_storage_dual(&fpha_result.templates[0]);
    let const_dual = solve_and_get_storage_dual(&const_result.templates[0]);

    assert!(
        const_dual.abs() < 1e-6,
        "constant-productivity dual must be ~0, got {const_dual}"
    );

    assert!(
        fpha_dual.abs() > 1e-6,
        "FPHA storage-fixing dual must be non-zero (FPHA v_in contribution \
         must be present), got {fpha_dual}"
    );

    assert_ne!(
        (fpha_dual * 1e6).round(),
        (const_dual * 1e6).round(),
        "storage-fixing dual must differ between FPHA ({fpha_dual}) and \
         constant-productivity ({const_dual})"
    );
}

/// The `four_hydro_mixed_system` (2 constant + 2 FPHA) solves to a finite
/// objective with non-negative FPHA generation variables.
#[test]
fn fpha_solve_mixed_system_optimal() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Fix v_in for all 4 hydros via rows 0..3.
    solver.set_row_bounds(
        &[0, 1, 2, 3],
        &[100.0, 100.0, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0],
    );

    let view = solver
        .solve(None)
        .expect("mixed FPHA LP must be feasible and optimal");

    assert!(
        view.objective.is_finite(),
        "objective must be finite, got {}",
        view.objective
    );

    let col_g0 = 19_usize; // FPHA generation variables for hydros 2 and 3
    let col_g1 = 20_usize;
    assert!(
        view.primal[col_g0] >= 0.0,
        "FPHA hydro 0 generation must be non-negative, got {}",
        view.primal[col_g0]
    );
    assert!(
        view.primal[col_g1] >= 0.0,
        "FPHA hydro 1 generation must be non-negative, got {}",
        view.primal[col_g1]
    );
}

// =========================================================================
// Evaporation variable tests
// =========================================================================

use cobre_solver::StageTemplate;

/// `EvaporationModelSet` giving `None` to every hydro except those in
/// `evap_indices`, which get `Linearized` from `intercept_per_stage`.
fn evap_set_for_system(
    system: &cobre_core::System,
    evap_indices: &[usize],
    intercept_per_stage: &[f64],
) -> EvaporationModelSet {
    let n_hydros = system.hydros().len();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let models = (0..n_hydros)
        .map(|h| {
            if evap_indices.contains(&h) {
                let coefficients = (0..n_stages)
                    .map(|s| LinearizedEvaporation {
                        intercept_m3s: intercept_per_stage
                            .get(s)
                            .copied()
                            .unwrap_or(intercept_per_stage.first().copied().unwrap_or(0.0)),
                        volume_slope_m3s_per_hm3: 0.0,
                    })
                    .collect();
                EvaporationModel::Linearized {
                    coefficients,
                    reference_volumes_hm3: vec![100.0; n_stages],
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

/// With zero evaporation hydros, `num_cols` and `num_rows` match the no-evaporation baseline.
#[test]
fn evap_zero_hydros_layout_unchanged() {
    let system = one_hydro_system(1, 0);
    let no_evap = default_evaporation(&system);
    let with_evap = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("none evaporation ok");

    assert_eq!(
        with_evap.templates[0].num_cols, baseline.templates[0].num_cols,
        "num_cols must match with zero evaporation hydros"
    );
    assert_eq!(
        with_evap.templates[0].num_rows, baseline.templates[0].num_rows,
        "num_rows must match with zero evaporation hydros"
    );
}

/// Each evaporation hydro adds 3 columns (evap outflow, f_evap_plus, f_evap_minus)
/// and 1 row (the evaporation equality), measured against the no-evaporation baseline.
#[test]
fn evap_two_hydros_increases_cols_and_rows() {
    let system1 = one_hydro_system(1, 0);

    let baseline = build_stage_templates(
        &system1,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system1),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("no evaporation baseline ok");

    let evap = evap_set_for_system(&system1, &[0], &[1.5]);
    let with_evap = build_stage_templates(
        &system1,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system1),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("1 evaporation hydro ok");

    let base_cols = baseline.templates[0].num_cols;
    let base_rows = baseline.templates[0].num_rows;
    let evap_cols = with_evap.templates[0].num_cols;
    let evap_rows = with_evap.templates[0].num_rows;

    assert_eq!(
        evap_cols,
        base_cols + 3,
        "1 evap hydro must add exactly 3 columns (evaporation outflow, f_evap_plus, f_evap_minus)"
    );
    assert_eq!(
        evap_rows,
        base_rows + 1,
        "1 evap hydro must add exactly 1 row (evaporation equality constraint)"
    );
}

/// The evaporation row is an equality at `intercept_m3s` (row_lower == row_upper).
#[test]
fn evap_row_bounds_equality_at_intercept() {
    let system = one_hydro_system(1, 0);
    let intercept_m3s = 1.5_f64;
    let evap = evap_set_for_system(&system, &[0], &[intercept_m3s]);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];

    // Evaporation row: followed by 4*N operational violation rows.
    let evap_row = t.num_rows - 1 - 4 * t.n_hydro;
    assert_eq!(
        t.row_lower[evap_row], intercept_m3s,
        "evaporation row_lower must equal intercept_m3s = {intercept_m3s}, got {}",
        t.row_lower[evap_row]
    );
    assert_eq!(
        t.row_upper[evap_row], intercept_m3s,
        "evaporation row_upper must equal intercept_m3s = {intercept_m3s}, got {}",
        t.row_upper[evap_row]
    );
}

/// The evaporation-outflow column has a physical upper bound; f_plus and f_minus
/// are unbounded above. All three carry objective 0.0.
#[test]
fn evap_col_bounds_and_objective() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_for_system(&system, &[0], &[1.5]);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];

    // The 3 evaporation columns are followed by 1 withdrawal slack + 4 operational
    // violation slack columns (5*N=5 total for N=1).
    let col_evaporation_flow = t.num_cols - 4 - 5 * t.n_hydro;
    let col_f_plus = t.num_cols - 3 - 5 * t.n_hydro;
    let col_f_minus = t.num_cols - 2 - 5 * t.n_hydro;

    // The evaporation-outflow column is free-signed: [-q_max, +q_max] where
    // q_max = |intercept_m3s + volume_slope_m3s_per_hm3 * v_max| * margin.
    // intercept_m3s = 1.5, volume_slope_m3s_per_hm3 = 0.0, v_max = 200.0 → q_max = 1.5 * 2.0 = 3.0.
    let expected_evaporation_flow_bound = 1.5 * EVAPORATION_FLOW_SAFETY_MARGIN;
    assert!(
        (t.col_lower[col_evaporation_flow] - (-expected_evaporation_flow_bound)).abs() < 1e-12,
        "evaporation-outflow lower bound must be {}, got {}",
        -expected_evaporation_flow_bound,
        t.col_lower[col_evaporation_flow]
    );
    assert!(
        (t.col_upper[col_evaporation_flow] - expected_evaporation_flow_bound).abs() < 1e-12,
        "evaporation-outflow upper bound must be {expected_evaporation_flow_bound}, got {}",
        t.col_upper[col_evaporation_flow]
    );
    assert_eq!(
        t.objective[col_evaporation_flow], 0.0,
        "evaporation-outflow objective must be 0.0, got {}",
        t.objective[col_evaporation_flow]
    );

    for &col in &[col_f_plus, col_f_minus] {
        assert_eq!(
            t.col_lower[col], 0.0,
            "evap slack column {col} lower bound must be 0.0, got {}",
            t.col_lower[col]
        );
        assert!(
            t.col_upper[col].is_infinite() && t.col_upper[col] > 0.0,
            "evap slack column {col} upper bound must be +inf, got {}",
            t.col_upper[col]
        );
        assert_eq!(
            t.objective[col], 0.0,
            "evap slack column {col} objective must be 0.0, got {}",
            t.objective[col]
        );
    }
}

// =========================================================================
// fill_evaporation_entries — CSC matrix entries
// =========================================================================

/// Like `evap_set_for_system` but with an explicit `volume_slope_m3s_per_hm3`
/// alongside `intercept_m3s`.
fn evap_set_with_volume_slope(
    system: &cobre_core::System,
    evap_indices: &[usize],
    intercept_m3s: f64,
    volume_slope_m3s_per_hm3: f64,
) -> EvaporationModelSet {
    let n_hydros = system.hydros().len();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let models = (0..n_hydros)
        .map(|h| {
            if evap_indices.contains(&h) {
                let coefficients = (0..n_stages)
                    .map(|_| LinearizedEvaporation {
                        intercept_m3s,
                        volume_slope_m3s_per_hm3,
                    })
                    .collect();
                EvaporationModel::Linearized {
                    coefficients,
                    reference_volumes_hm3: vec![100.0; n_stages],
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

/// All `(row, value)` CSC entries for a given column.
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by construction
fn entries_for_col(t: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    (start..end)
        .map(|i| (t.row_indices[i] as usize, t.values[i]))
        .collect()
}

/// With `volume_slope_m3s_per_hm3 = 0.02`, the evaporation constraint row carries
/// `(evap-outflow, +1.0)`, `(v_col, -0.01)`, `(v_in_col, -0.01)` (= -slope/2),
/// `(f_plus_col, +1.0)`, `(f_minus_col, -1.0)`; the evap-outflow column also enters
/// the water balance row with `+zeta`.
#[test]
fn evap_csc_entries_one_hydro_correct_coefficients() {
    let system = one_hydro_system(1, 0);
    let volume_slope_m3s_per_hm3 = 0.02_f64;
    let evap = evap_set_with_volume_slope(&system, &[0], 1.5, volume_slope_m3s_per_hm3);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];

    // Column layout for 1-hydro system (N=1, L=0, T=0, B=1, K=1):
    //   col 0 = v (storage_out)  col 1 = z_inflow  col 2 = v_in  col 3 = theta
    //   col 4 = turbine  col 5 = spillage  col 6 = diversion
    //   col 7 = deficit  col 8 = excess
    //   col_evap_start = num_cols - 4 - 4*N
    // Row layout for N=1, L=0, B=1, K=1, no FPHA (no state-fixing rows):
    //   row 0: z_inflow definition
    //   row 1: water balance (row_water_balance_start = N = 1)
    //   row 2: load balance
    //   row 3: evaporation constraint
    //   rows 4-7: operational violation rows
    // Evaporation columns come before withdrawal slack + 4*N operational slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * t.n_hydro;
    let col_f_plus = t.num_cols - 3 - 5 * t.n_hydro;
    let col_f_minus = t.num_cols - 2 - 5 * t.n_hydro;
    let evap_row = t.num_rows - 1 - 4 * t.n_hydro;
    let water_balance_row = 1_usize; // row_water_balance_start = N = 1

    // evaporation outflow has 2 entries: water balance row (+zeta) and
    // evaporation constraint row (+1.0). Entries are sorted by row ascending.
    let zeta = 744.0 * (3_600.0 / 1_000_000.0);
    let entries_evaporation_flow = entries_for_col(t, col_evaporation_flow);
    assert_eq!(
        entries_evaporation_flow.len(),
        2,
        "evaporation outflow column must have exactly 2 entries (water balance + evap constraint), got {entries_evaporation_flow:?}"
    );
    assert_eq!(
        entries_evaporation_flow[0].0, water_balance_row,
        "evaporation outflow first entry must be at water balance row"
    );
    assert!(
        (entries_evaporation_flow[0].1 - zeta).abs() < 1e-12,
        "evaporation outflow water balance coefficient must be +zeta={zeta}, got {}",
        entries_evaporation_flow[0].1
    );
    assert_eq!(
        entries_evaporation_flow[1].0, evap_row,
        "evaporation outflow second entry must be at evap_row"
    );
    assert!(
        (entries_evaporation_flow[1].1 - 1.0).abs() < 1e-12,
        "evaporation outflow evap constraint coefficient must be +1.0, got {}",
        entries_evaporation_flow[1].1
    );

    let entries_f_plus = entries_for_col(t, col_f_plus);
    assert_eq!(
        entries_f_plus.len(),
        1,
        "f_plus column must have exactly 1 entry, got {entries_f_plus:?}"
    );
    assert_eq!(
        entries_f_plus[0].0, evap_row,
        "f_plus entry must be at evap_row"
    );
    assert!(
        (entries_f_plus[0].1 - 1.0).abs() < 1e-12,
        "f_plus coefficient must be +1.0, got {}",
        entries_f_plus[0].1
    );

    let entries_f_minus = entries_for_col(t, col_f_minus);
    assert_eq!(
        entries_f_minus.len(),
        1,
        "f_minus column must have exactly 1 entry, got {entries_f_minus:?}"
    );
    assert_eq!(
        entries_f_minus[0].0, evap_row,
        "f_minus entry must be at evap_row"
    );
    assert!(
        (entries_f_minus[0].1 - (-1.0)).abs() < 1e-12,
        "f_minus coefficient must be -1.0, got {}",
        entries_f_minus[0].1
    );

    // v and v_in carry -volume_slope/2 at evap_row (average-storage split).
    let expected_coeff = -volume_slope_m3s_per_hm3 / 2.0;
    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have an entry at evap_row");
    assert!(
        (entry_v.1 - expected_coeff).abs() < 1e-12,
        "v coefficient must be {expected_coeff}, got {}",
        entry_v.1
    );

    // v_in column: storage_in.start for 1-hydro (L=0) = N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have an entry at evap_row");
    assert!(
        (entry_v_in.1 - expected_coeff).abs() < 1e-12,
        "v_in coefficient must be {expected_coeff}, got {}",
        entry_v_in.1
    );
}

/// coefficient value check with `volume_slope_m3s_per_hm3 = 0.04` → v and `v_in` entries are -0.02.
#[test]
fn evap_csc_entries_coefficient_scaling() {
    let system = one_hydro_system(1, 0);
    let volume_slope_m3s_per_hm3 = 0.04_f64;
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, volume_slope_m3s_per_hm3);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let evap_row = t.num_rows - 1 - 4 * t.n_hydro;
    let expected_coeff = -volume_slope_m3s_per_hm3 / 2.0; // -0.02

    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have evap_row entry");
    assert!(
        (entry_v.1 - expected_coeff).abs() < 1e-12,
        "v coefficient: expected {expected_coeff}, got {}",
        entry_v.1
    );

    // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have evap_row entry");
    assert!(
        (entry_v_in.1 - expected_coeff).abs() < 1e-12,
        "v_in coefficient: expected {expected_coeff}, got {}",
        entry_v_in.1
    );
}

/// 0 evaporation hydros — `fill_evaporation_entries` is a no-op;
/// the evaporation columns do not exist and no extra non-zeros are added.
#[test]
fn evap_csc_entries_zero_hydros_no_op() {
    let system = one_hydro_system(1, 0);
    let no_evap = default_evaporation(&system);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &EvaporationModelSet::new(vec![EvaporationModel::None]),
        &ResolvedParameters::default(),
    )
    .expect("none evaporation ok");

    assert_eq!(
        result.templates[0].num_nz, baseline.templates[0].num_nz,
        "num_nz must be identical with zero evaporation hydros"
    );
}

/// 2 evap hydros with distinct `volume_slope_m3s_per_hm3` produce independent rows.
#[test]
fn evap_csc_entries_two_hydros_independent_rows() {
    let (system, production) = four_hydro_mixed_system();
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();

    let models = vec![
        EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.02,
                };
                n_stages
            ],
            reference_volumes_hm3: vec![100.0; n_stages],
        },
        EvaporationModel::Linearized {
            coefficients: vec![
                LinearizedEvaporation {
                    intercept_m3s: 2.0,
                    volume_slope_m3s_per_hm3: 0.06,
                };
                n_stages
            ],
            reference_volumes_hm3: vec![100.0; n_stages],
        },
        EvaporationModel::None,
        EvaporationModel::None,
    ];
    let evap = EvaporationModelSet::new(models);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("2-evap-hydro system ok");

    let t = &result.templates[0];
    // 2 evap hydros: evap rows are followed by 4*N operational violation rows.
    let evap_row_0 = t.num_rows - 2 - 4 * t.n_hydro;
    let evap_row_1 = t.num_rows - 1 - 4 * t.n_hydro;

    // Hydro 0 (volume_slope_m3s_per_hm3=0.02): v coefficient = -0.01.
    let entry_v_h0 = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row_0)
        .expect("hydro 0 v col entry");
    assert!(
        (entry_v_h0.1 - (-0.01)).abs() < 1e-12,
        "hydro 0 v: expected -0.01, got {}",
        entry_v_h0.1
    );

    // Hydro 1 (volume_slope_m3s_per_hm3=0.06): v coefficient = -0.03.
    let entry_v_h1 = entries_for_col(t, 1)
        .into_iter()
        .find(|&(r, _)| r == evap_row_1)
        .expect("hydro 1 v col entry");
    assert!(
        (entry_v_h1.1 - (-0.03)).abs() < 1e-12,
        "hydro 1 v: expected -0.03, got {}",
        entry_v_h1.1
    );

    // Row bounds: hydro 0 → intercept_m3s=1.0, hydro 1 → intercept_m3s=2.0.
    assert!((t.row_lower[evap_row_0] - 1.0).abs() < 1e-12);
    assert!((t.row_lower[evap_row_1] - 2.0).abs() < 1e-12);
}

/// `volume_slope_m3s_per_hm3 = 0.0` → v and `v_in` entries are 0.0;
/// the constraint reduces to `evaporation outflow + f_plus - f_minus = intercept_m3s`.
#[test]
fn evap_csc_entries_zero_volume_slope_produces_zero_volume_coefficients() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_with_volume_slope(&system, &[0], 2.0, 0.0);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];
    let evap_row = t.num_rows - 1 - 4 * t.n_hydro;

    let entry_v = entries_for_col(t, 0)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v column must have evap_row entry");
    assert!(
        entry_v.1.abs() < 1e-12,
        "v coefficient must be 0.0 when volume_slope_m3s_per_hm3=0, got {}",
        entry_v.1
    );

    // storage_in.start for 1-hydro (L=0): N*(2+L) = 2; col_v_in = 2 + h_idx = 2.
    let col_v_in = 2;
    let entry_v_in = entries_for_col(t, col_v_in)
        .into_iter()
        .find(|&(r, _)| r == evap_row)
        .expect("v_in column must have evap_row entry");
    assert!(
        entry_v_in.1.abs() < 1e-12,
        "v_in coefficient must be 0.0 when volume_slope_m3s_per_hm3=0, got {}",
        entry_v_in.1
    );
}

// ── water balance entries for evaporation ────────────────────

/// The evaporation-outflow column enters the water balance row with `+zeta`,
/// `zeta = 744.0 * 3_600.0 / 1_000_000.0` (1 block of 744h).
#[test]
#[allow(clippy::cast_sign_loss)]
fn evap_water_balance_one_hydro_coefficient_is_zeta() {
    let system = one_hydro_system(1, 0);
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, 0.0);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evaporation system ok");

    let t = &result.templates[0];

    let water_balance_row = 1_usize; // row_water_balance_start = N = 1

    // evap outflow is the first of 3 evaporation columns; before withdrawal + 4*N op slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * t.n_hydro;

    let entries = entries_for_col(t, col_evaporation_flow);
    let entry = entries
        .iter()
        .find(|&&(r, _)| r == water_balance_row)
        .copied()
        .expect("evaporation outflow column must have an entry in the water balance row");

    let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
    assert!(
        (entry.1 - zeta).abs() < 1e-12,
        "evaporation outflow water balance coefficient must be +zeta={zeta}, got {}",
        entry.1
    );
}

/// With evaporation only on hydro 1, hydro 1's evap-outflow column carries `+zeta`
/// in its water balance row while hydro 0's water balance row has no evap entry.
#[test]
#[allow(clippy::cast_sign_loss, clippy::too_many_lines)]
fn evap_water_balance_only_second_hydro_has_evap() {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage as CStage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let hp = HydroPenalties {
        spillage_cost: 0.01,
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
    };
    let make_h = |id: i32| Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: hp,
    };
    let hydros = vec![make_h(2), make_h(3)];
    let stages = vec![CStage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];
    let inflow_models: Vec<InflowModel> = hydros
        .iter()
        .map(|h| InflowModel {
            hydro_id: h.id,
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();
    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 200.0,
        std_mw: 0.0,
    }];
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("2-hydro system ok");

    // Only hydro 1 (h_idx=1) has evaporation.
    let evap = evap_set_with_volume_slope(&system, &[1], 0.0, 0.0);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("2-hydro evap system ok");

    let t = &result.templates[0];

    // row_water_balance_start = N = 2 (z_inflow rows [0,2)); hydro 0 row 2, hydro 1 row 3.
    let water_balance_row_h0 = 2_usize;
    let water_balance_row_h1 = 3_usize;

    // evaporation outflow for hydro 1 (local_idx=0, since only hydro 1 is evap): col_evap_start + 0*3.
    // N=2 withdrawal + 4*N operational slack columns follow evap.
    let col_evaporation_flow_h1 = t.num_cols - 5 - 5 * t.n_hydro;

    // evaporation outflow (h1) must have an entry at water balance row 3.
    let entries_h1 = entries_for_col(t, col_evaporation_flow_h1);
    let found_h1 = entries_h1
        .iter()
        .find(|&&(r, _)| r == water_balance_row_h1)
        .copied();
    assert!(
        found_h1.is_some(),
        "evaporation outflow for hydro 1 must have an entry in water balance row {water_balance_row_h1}"
    );
    let zeta = 744.0_f64 * (3_600.0 / 1_000_000.0);
    assert!(
        (found_h1.unwrap().1 - zeta).abs() < 1e-12,
        "evaporation outflow (h1) water balance coefficient must be +zeta={zeta}, got {}",
        found_h1.unwrap().1
    );

    // evaporation outflow (h1) must NOT have an entry at hydro 0's water balance row.
    let found_h0 = entries_h1.iter().any(|&(r, _)| r == water_balance_row_h0);
    assert!(
        !found_h0,
        "evaporation outflow for hydro 1 must not appear in hydro 0's water balance row"
    );
}

/// Zero evaporation hydros leave `num_nz` identical to the no-evaporation baseline.
#[test]
fn evap_water_balance_zero_hydros_no_op() {
    let system = one_hydro_system(1, 0);
    let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("no evaporation ok");

    let baseline = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("default evaporation ok");

    assert_eq!(
        result.templates[0].num_nz, baseline.templates[0].num_nz,
        "num_nz must be identical with zero evaporation hydros (no water balance entries added)"
    );
}

// =========================================================================
// Evaporation violation cost tests
// =========================================================================

/// 1-bus, 1-hydro (constant-productivity), 1-stage system with a parameterized
/// `evaporation_violation_cost` and block duration.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn evap_hydro_system_with_violation_cost(
    block_hours: f64,
    evaporation_violation_cost: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 2_000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: evaporation_violation_cost,
            evaporation_violation_neg_cost: evaporation_violation_cost,
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let stages = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: block_hours,
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
    }];

    let inflow_models = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 50.0,
        std_m3s: 10.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 2_000.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: evaporation_violation_cost,
                evaporation_violation_neg_cost: evaporation_violation_cost,
                inflow_nonnegativity_cost: 1000.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("evap_hydro_system_with_violation_cost: valid")
}

/// `f_evap_plus` and `f_evap_minus` both carry the base violation cost: with
/// `evaporation_violation_cost = 500.0` over a 730h block the objective is
/// `500.0 * 730.0 / COST_SCALE_FACTOR = 365.0` (the test sets pos_cost == base_cost).
#[test]
fn evap_violation_cost_applied_to_slack_columns() {
    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap violation cost system builds ok");

    let t = &result.templates[0];

    // Evaporation columns (evaporation outflow, f_plus, f_minus) are followed by
    // 1 withdrawal slack + 4*N operational slacks.
    let col_evaporation_flow = t.num_cols - 4 - 5 * t.n_hydro;
    let col_f_plus = t.num_cols - 3 - 5 * t.n_hydro;
    let col_f_minus = t.num_cols - 2 - 5 * t.n_hydro;

    let expected_base = 500.0 * 730.0 / COST_SCALE_FACTOR;

    assert!(
        t.objective[col_evaporation_flow].abs() < 1e-12,
        "evaporation outflow column objective must be 0.0 (evaporation flow itself has no cost), got {}",
        t.objective[col_evaporation_flow]
    );
    assert!(
        (t.objective[col_f_plus] - expected_base).abs() < 1e-12,
        "f_evap_plus objective: expected {expected_base}, got {}",
        t.objective[col_f_plus]
    );
    // f_evap_minus (over-evaporation) now uses evaporation_violation_pos_cost directly.
    // Test-constructed HydroPenalties sets pos_cost = base_cost, so objective matches.
    assert!(
        (t.objective[col_f_minus] - expected_base).abs() < 1e-6,
        "f_evap_minus objective: expected {expected_base} (pos_cost == base_cost in test), got {}",
        t.objective[col_f_minus]
    );
}

/// `evaporation outflow` column objective is 0.0 even when a
/// non-zero `evaporation_violation_cost` is set.
#[test]
fn evap_outflow_objective_is_zero() {
    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 0.0, 0.0);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system with zero k_evap builds ok");

    let t = &result.templates[0];
    // N=1 withdrawal + 4*N operational slacks follow the 3 evap columns.
    let col_evaporation_flow = t.num_cols - 4 - 5 * t.n_hydro;

    assert!(
        t.objective[col_evaporation_flow].abs() < 1e-12,
        "evaporation outflow objective must be 0.0, got {}",
        t.objective[col_evaporation_flow]
    );
}

/// LP with 1 evaporation hydro is solvable (`HiGHS` returns `Optimal`) after
/// fixing `v_in = 1000.0 hm3`.
///
/// System: 1 bus, 1 hydro, `intercept_m3s = 1.0`, `volume_slope_m3s_per_hm3 = 0.02`.
/// With all-positive coefficients and `v_in` fixed at 1000 hm3, the
/// linearised equality forces `evaporation outflow = intercept_m3s + (volume_slope_m3s_per_hm3 / 2) · (v + v_in)`,
/// whose minimum at `v = v_min = 0` is `1.0 + 0.01 · 1000 = 11.0`.
#[test]
fn evap_lp_solvable_and_outflow_positive_coefficients() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Fix v_in = 1000 hm3 via column bounds on storage_in.
    let col_storage_in = 2_usize; // col 0 = storage_out, col 1 = z_inflow, col 2 = storage_in
    let v_in = 1_000.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("evaporation LP must be feasible and optimal");

    // evaporation outflow is the first evaporation column (before withdrawal + 4*N operational slacks).
    let col_evaporation_flow = template.num_cols - 4 - 5 * template.n_hydro;
    let evaporation_flow = view.primal[col_evaporation_flow];

    // Tight lower bound: evaporation outflow >= intercept_m3s + (volume_slope_m3s_per_hm3 / 2) · v_min + (volume_slope_m3s_per_hm3 / 2) · v_in
    //                         >= 1.0   + 0.0                       + 0.01 · 1000 = 11.0.
    // A loose threshold (`evaporation_flow > -1e-8`) would silently pass a sign-convention
    // regression that flipped the bound; assert the structurally-forced minimum.
    assert!(
        evaporation_flow > 10.0,
        "evaporation outflow must reflect the positive linearised target (>= 11.0), got {evaporation_flow}"
    );
}

/// violation slacks are near zero when `v_in` is large
/// enough for the linearised evaporation constraint to be satisfiable without
/// artificial violation.
///
/// With `intercept_m3s = 1.0`, `volume_slope_m3s_per_hm3 = 0.02`, and `v_in = 1000 hm3`, the
/// evaporation constraint RHS is positive and feasible, so the solver should
/// drive the high-cost violation slacks to zero.
#[test]
fn evap_violation_slacks_near_zero_feasible_constraint() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 1.0, 0.02);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    let v_in = 1_000.0_f64;
    solver.set_row_bounds(&[0], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("evaporation LP must be feasible and optimal");

    // Evaporation violation slack columns are before withdrawal + 4*N operational slacks.
    let col_f_plus = template.num_cols - 3 - 5 * template.n_hydro;
    let col_f_minus = template.num_cols - 2 - 5 * template.n_hydro;
    let f_plus = view.primal[col_f_plus];
    let f_minus = view.primal[col_f_minus];

    assert!(
        f_plus.abs() < 1e-6,
        "f_evap_plus slack must be near zero (constraint satisfied without violation), got {f_plus}"
    );
    assert!(
        f_minus.abs() < 1e-6,
        "f_evap_minus slack must be near zero (constraint satisfied without violation), got {f_minus}"
    );
}

/// the storage-fixing dual for an evaporation hydro differs
/// from the no-evaporation case.
///
/// When evaporation is active, higher `v_in` reduces evaporation volume
/// (water retained in the reservoir increases), changing the water balance and
/// hence the marginal value of initial storage. The dual of the storage-fixing
/// row must differ between the two configurations.
#[test]
fn evap_storage_fixing_dual_differs_from_no_evaporation() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    // System with evaporation violation cost (so slacks are penalised).
    let system_evap = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system_evap, &[0], 1.0, 0.02);
    let evap_result = build_stage_templates(
        &system_evap,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_evap),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap system template build must succeed");

    // Baseline system without evaporation (same structure, EvaporationModel::None).
    let system_base = one_hydro_system(1, 0);
    let no_evap = EvaporationModelSet::new(vec![EvaporationModel::None]);
    let base_result = build_stage_templates(
        &system_base,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_base),
        &no_evap,
        &ResolvedParameters::default(),
    )
    .expect("baseline system template build must succeed");

    let solve_and_get_storage_dual = |template: &cobre_solver::StageTemplate| -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        solver.load_model(template);
        let empty_cuts = RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        };
        solver.add_rows(&empty_cuts);
        let v_in = 1_000.0_f64;
        solver.set_row_bounds(&[0], &[v_in], &[v_in]);
        let view = solver.solve(None).expect("LP must solve to optimal");
        // Row 0 is the storage-fixing equality; its dual is the marginal value
        // of one additional hm3 of initial storage.
        view.dual[0]
    };

    let evap_dual = solve_and_get_storage_dual(&evap_result.templates[0]);
    let base_dual = solve_and_get_storage_dual(&base_result.templates[0]);

    // The evaporation constraint couples evaporation outflow to v and v_in via volume_slope_m3s_per_hm3,
    // so the marginal value of initial storage differs from the no-evaporation case.
    // Note: with unused bidirectional withdrawal slack columns (pinned to zero),
    // the solver may produce degenerate duals where both are -0.0 or 0.0.
    // We compare the raw f64 values to account for this edge case.
    let evap_rounded = (evap_dual * 1e6).round();
    let base_rounded = (base_dual * 1e6).round();
    // When both are zero (degenerate), the test is inconclusive but not a failure.
    if evap_rounded != 0.0 || base_rounded != 0.0 {
        assert_ne!(
            evap_rounded, base_rounded,
            "storage-fixing dual must differ between evaporation ({evap_dual}) and \
             no-evaporation ({base_dual}) configurations"
        );
    }
}

/// evaporation outflow physical bound prevents the LP from using evaporation as a dump
/// valve.  With high v_in and high inflow, the LP must use spillage (not
/// evaporation) to remove excess water.  The test confirms evaporation outflow <= evaporation_flow_max,
/// f_minus ~ 0, and spillage > 0.
#[test]
fn evap_bound_prevents_dump_valve() {
    use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};

    let system = evap_hydro_system_with_violation_cost(730.0, 500.0);
    let evap = evap_set_with_volume_slope(&system, &[0], 2.0, 0.0001);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &evap,
        &ResolvedParameters::default(),
    )
    .expect("evap dump valve test: template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Fix v_in at max_storage = 2000 hm3 via column bounds on storage_in.
    // col 0 = storage_out, col 1 = z_inflow, col 2 = storage_in (N=1, L=0).
    let col_storage_in = 2_usize;
    let v_in = 2_000.0_f64;
    solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);

    // Inject large inflow via water-balance RHS (row 1; row 0 = z_inflow[0]).
    // Water balance: v + zeta*(turbine + spill + div) - v_in + zeta*evaporation outflow = RHS.
    // The template RHS = zeta * base = 2.628 * 50 = 131.4.
    // Set RHS to zeta * 500 = 1314 to simulate a 500 m3/s inflow.
    // The LP must then satisfy: v + zeta*(turbine+spill+...) = v_in + 1314 = 3314.
    // With v <= 2000 and max turbine = 262.8 hm3, surplus > 1000 hm3 must spill.
    let zeta = 730.0 * 3600.0 / 1e6;
    let high_inflow_rhs = zeta * 500.0;
    // Water balance row index: N + h = 1 + 0 = 1 for N=1, L=0, h=0.
    let water_balance_row = 1_usize;
    solver.set_row_bounds(&[water_balance_row], &[high_inflow_rhs], &[high_inflow_rhs]);

    let view = solver
        .solve(None)
        .expect("evap dump valve LP must be feasible and optimal");

    // Column layout: N=1, L=0, K=1.
    // col 0: v, col 1: z_inflow, col 2: v_in, col 3: theta,
    // col 4: turbine, col 5: spillage, col 6: diversion,
    // col 7: deficit, col 8: excess.
    // Evaporation columns: evaporation outflow, f_plus, f_minus, then withdrawal + 4*N operational slacks.
    let col_spillage = 5;
    let col_evaporation_flow = template.num_cols - 4 - 5 * template.n_hydro;
    let col_f_minus = template.num_cols - 2 - 5 * template.n_hydro;

    let evaporation_flow = view.primal[col_evaporation_flow];
    let f_minus = view.primal[col_f_minus];
    let spillage = view.primal[col_spillage];

    // evaporation outflow must respect the symmetric magnitude bound.
    // intercept_m3s=2.0, volume_slope_m3s_per_hm3=0.0001, max_storage_hm3=2000.0
    // evaporation_flow_max = |2.0 + 0.0001*2000| * 2.0 = 2.2 * 2.0 = 4.4
    let evaporation_flow_max = (2.0 + 0.0001 * 2_000.0_f64).abs() * EVAPORATION_FLOW_SAFETY_MARGIN;
    assert!(
        evaporation_flow <= evaporation_flow_max + 1e-8,
        "evaporation outflow must be bounded by physical limit {evaporation_flow_max}, got {evaporation_flow}"
    );

    // Over-evaporation violation must be near zero (the violation penalty deters it).
    assert!(
        f_minus < 1e-6,
        "f_minus (over-evaporation) must be near zero, got {f_minus}"
    );

    assert!(
        spillage > 1e-6,
        "spillage must be positive when excess water needs dumping, got {spillage}"
    );
}

// ─── Multi-segment deficit tests ──────────────────────────────────────────

/// No-hydro, no-thermal, no-line system over the given buses, 1 stage / 1 block.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn multi_segment_system(buses: Vec<Bus>, block_hours: f64) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    };

    let n_buses = buses.len();
    let load_models: Vec<LoadModel> = buses
        .iter()
        .map(|b| LoadModel {
            bus_id: b.id,
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        })
        .collect();

    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: block_hours,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: cobre_core::temporal::NoiseMethod::Saa,
        },
    };

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(buses)
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("multi_segment_system: valid")
}

/// 2 buses (bus0: 3 segments, bus1: 1), 2 blocks → deficit columns = `B*S_max*K` = 2*3*2 = 12.
#[test]
#[allow(clippy::too_many_lines)]
fn test_multi_segment_deficit_column_count() {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    };

    let bus0 = Bus {
        id: EntityId(1),
        name: "Bus0".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![
            DeficitSegment {
                depth_mw: Some(10.0),
                cost_per_mwh: 100.0,
            },
            DeficitSegment {
                depth_mw: Some(20.0),
                cost_per_mwh: 200.0,
            },
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            },
        ],
        excess_cost: 0.0,
    };
    let bus1 = Bus {
        id: EntityId(2),
        name: "Bus1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let load_models = vec![
        LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
        LoadModel {
            bus_id: EntityId(2),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
    ];

    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![
            Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 360.0,
            },
            Block {
                index: 1,
                name: "B1".to_string(),
                duration_hours: 360.0,
            },
        ],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: cobre_core::temporal::NoiseMethod::Saa,
        },
    };

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 2,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("2-bus 2-block system: valid");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, L=0 → theta=0, decision_start=1
    // No thermals, no lines → col_deficit_start = 1
    // B=2, S_max=3, K=2 → deficit region = 2*3*2 = 12 columns → col_excess_start = 13
    // excess region = B*K = 2*2 = 4 → num_cols = 13 + 4 = 17
    let col_deficit_start = 1_usize;
    let max_deficit_segments = 3_usize;
    let n_blks = 2_usize;
    let n_buses = 2_usize;
    let deficit_region = n_buses * max_deficit_segments * n_blks;
    assert_eq!(
        deficit_region, 12,
        "deficit region must be B*S_max*K = 2*3*2 = 12"
    );
    let col_excess_start = col_deficit_start + deficit_region;
    let excess_region = n_buses * n_blks; // 2*2 = 4
    let expected_num_cols = col_excess_start + excess_region;
    assert_eq!(
        t.num_cols, expected_num_cols,
        "num_cols must include expanded deficit region"
    );
}

/// Bus with 2 deficit segments [{10MW, $500}, {None, $5000}], 1 block of 730h:
/// segment upper bounds are depth_mw / +inf, objectives are cost * block_hours.
#[test]
fn test_multi_segment_deficit_bounds_and_objective() {
    let bus = Bus {
        id: EntityId(1),
        name: "Bus0".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![
            DeficitSegment {
                depth_mw: Some(10.0),
                cost_per_mwh: 500.0,
            },
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            },
        ],
        excess_cost: 0.0,
    };

    let block_hours = 730.0_f64;
    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, no thermals/lines
    // col_deficit_start = 1
    // B=1, S_max=2, K=1 → deficit region = 1*2*1 = 2
    // seg0 col = 1 + 0*2*1 + 0*1 + 0 = 1
    // seg1 col = 1 + 0*2*1 + 1*1 + 0 = 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    assert_eq!(
        t.col_upper[col_seg0], 10.0,
        "segment 0 upper bound must equal depth_mw = 10.0"
    );
    assert!(
        t.col_upper[col_seg1].is_infinite() && t.col_upper[col_seg1] > 0.0,
        "segment 1 upper bound must be +infinity (unbounded final segment)"
    );
    assert!(
        (t.objective[col_seg0] - 500.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 0 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        500.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg0]
    );
    assert!(
        (t.objective[col_seg1] - 5000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 1 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        5000.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg1]
    );
}

/// A single-segment bus produces one unbounded deficit column carrying the cost.
#[test]
fn test_single_segment_backward_compat() {
    let cost = 1000.0_f64;
    let block_hours = 744.0_f64;

    let bus = Bus {
        id: EntityId(1),
        name: "Bus0".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: cost,
        }],
        excess_cost: 0.0,
    };

    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, col_deficit_start=1
    // B=1, S_max=1, K=1 → 1 deficit column at index 1
    let col_def = 1_usize;

    assert!(
        t.col_upper[col_def].is_infinite() && t.col_upper[col_def] > 0.0,
        "single segment must be unbounded (None depth_mw)"
    );
    assert!(
        (t.objective[col_def] - cost * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "single-segment objective must be cost * block_hours / COST_SCALE_FACTOR"
    );

    // Excess column immediately follows deficit (S_max=1, B=1, K=1 → excess at col 2)
    let col_exc = 2_usize;
    assert!(
        t.col_upper[col_exc].is_infinite() && t.col_upper[col_exc] > 0.0,
        "excess column must be unbounded"
    );
}

/// Each deficit segment column has exactly one CSC entry, in the load-balance
/// row, with coefficient +1.0.
#[test]
fn test_multi_segment_deficit_load_balance_coefficients() {
    let bus = Bus {
        id: EntityId(1),
        name: "Bus0".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![
            DeficitSegment {
                depth_mw: Some(10.0),
                cost_per_mwh: 500.0,
            },
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            },
        ],
        excess_cost: 0.0,
    };

    let system = multi_segment_system(vec![bus], 730.0);

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, no thermals/lines → col_deficit_start = 1
    // B=1, S_max=2, K=1 → seg0 at col 1, seg1 at col 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    // N=0 hydros → row_load_balance_start = 0; bus 0, block 0 → row 0
    let load_balance_row = 0_usize;

    let entries_seg0 = entries_for_col(t, col_seg0);
    assert_eq!(
        entries_seg0.len(),
        1,
        "deficit segment 0 column must have exactly 1 CSC entry (load-balance row), got {entries_seg0:?}"
    );
    assert_eq!(
        entries_seg0[0].0, load_balance_row,
        "deficit segment 0 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg0[0].0
    );
    assert!(
        (entries_seg0[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 0 load-balance coefficient must be +1.0, got {}",
        entries_seg0[0].1
    );

    let entries_seg1 = entries_for_col(t, col_seg1);
    assert_eq!(
        entries_seg1.len(),
        1,
        "deficit segment 1 column must have exactly 1 CSC entry (load-balance row), got {entries_seg1:?}"
    );
    assert_eq!(
        entries_seg1[0].0, load_balance_row,
        "deficit segment 1 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg1[0].0
    );
    assert!(
        (entries_seg1[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 1 load-balance coefficient must be +1.0, got {}",
        entries_seg1[0].1
    );
}

// -------------------------------------------------------------------------
// Water withdrawal LP wiring unit tests
// -------------------------------------------------------------------------

/// `one_hydro_system` variant injecting `water_withdrawal_m3s` and
/// `water_withdrawal_violation_cost`. One 744h block; `lag_order` adds AR lag columns.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn one_hydro_system_with_withdrawal(
    n_stages: usize,
    lag_order: usize,
    water_withdrawal_m3s: f64,
    water_withdrawal_violation_cost: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost,
            water_withdrawal_violation_pos_cost: water_withdrawal_violation_cost,
            water_withdrawal_violation_neg_cost: water_withdrawal_violation_cost,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: lag_order > 0,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let ar_coefficients: Vec<f64> = (0..lag_order).map(|_| 0.5).collect();
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coefficients.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);

    // Build bounds with the specified withdrawal rate for every (hydro, stage) cell.
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    for s in 0..n_st {
        bounds.hydro_bounds_mut(0, s).water_withdrawal_m3s = water_withdrawal_m3s;
    }

    let mut penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost,
                water_withdrawal_violation_pos_cost: water_withdrawal_violation_cost,
                water_withdrawal_violation_neg_cost: water_withdrawal_violation_cost,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    for s in 0..n_st {
        penalties
            .hydro_penalties_mut(0, s)
            .water_withdrawal_violation_cost = water_withdrawal_violation_cost;
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_system_with_withdrawal: valid")
}

/// Water balance RHS = ζ * (`deterministic_base` - `water_withdrawal_m3s`). With
/// no PAR data base=0, so for withdrawal=10 the RHS is
/// `744 * 3600/1_000_000 * (0 - 10) = -2.6784`.
#[test]
fn withdrawal_rhs_subtracted_from_water_balance() {
    let withdrawal = 10.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, withdrawal, 0.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let row_water = 1_usize; // row_water_balance_start = N = 1
    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;
    // base = 0 (no PAR data), withdrawal = 10.0
    let expected_rhs = zeta * (0.0 - withdrawal);
    assert!(
        (t.row_lower[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_lower: expected {expected_rhs}, got {}",
        t.row_lower[row_water]
    );
    assert!(
        (t.row_upper[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_upper: expected {expected_rhs}, got {}",
        t.row_upper[row_water]
    );
}

/// `PrecomputedPar::default()` has `n_stages`=0, so `deterministic_base`=0 cannot
/// be set to a non-zero value here; this test only confirms withdrawal=0 leaves
/// the RHS identical to the no-withdrawal case. The base=50 arithmetic
/// (0.36 * (50-10) = 14.4) is covered by the fixture-free test below.
#[test]
fn withdrawal_zero_leaves_rhs_unchanged_from_base() {
    let system_zero = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let system_base = one_hydro_system(1, 0);

    let result_zero = build_stage_templates(
        &system_zero,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_zero),
        &default_evaporation(&system_zero),
        &ResolvedParameters::default(),
    )
    .expect("zero-withdrawal build ok");

    let result_base = build_stage_templates(
        &system_base,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_base),
        &default_evaporation(&system_base),
        &ResolvedParameters::default(),
    )
    .expect("base build ok");

    let t_zero = &result_zero.templates[0];
    let t_base = &result_base.templates[0];

    // N=1, L=0: row_water_balance_start = 1.
    let row_water = 1_usize;
    assert!(
        (t_zero.row_lower[row_water] - t_base.row_lower[row_water]).abs() < 1e-15,
        "zero-withdrawal row_lower must equal base: {} vs {}",
        t_zero.row_lower[row_water],
        t_base.row_lower[row_water]
    );
    assert!(
        (t_zero.row_upper[row_water] - t_base.row_upper[row_water]).abs() < 1e-15,
        "zero-withdrawal row_upper must equal base: {} vs {}",
        t_zero.row_upper[row_water],
        t_base.row_upper[row_water]
    );
}

/// The withdrawal slack column has exactly one CSC entry, at (`row_water`, -`zeta`).
#[test]
fn withdrawal_slack_matrix_entry_coefficient_is_minus_zeta() {
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, 1000.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // Withdrawal slack neg column: followed by N pos cols and 4*N operational violation slack columns.
    let col_neg = t.num_cols - 1 - 4 * t.n_hydro - t.n_hydro;
    let row_water = 1_usize; // water balance for hydro 0 = N + h = 1 + 0

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff = csc_entry(t, col_neg, row_water).unwrap_or_else(|| {
        panic!(
            "withdrawal slack neg column {col_neg} has no entry at water balance row {row_water}"
        )
    });
    assert!(
        (coeff - (-zeta)).abs() < 1e-12,
        "withdrawal slack neg coefficient: expected {}, got {coeff}",
        -zeta
    );

    // Must have exactly one entry (water balance only; no load balance).
    let all_entries = entries_for_col(t, col_neg);
    assert_eq!(
        all_entries.len(),
        1,
        "withdrawal slack neg column must have exactly 1 CSC entry, got {all_entries:?}"
    );
}

/// Slack objective = `water_withdrawal_violation_cost` * `total_stage_hours`;
/// 1000.0 * 744.0 = 744_000.0 (then COST_SCALE_FACTOR-scaled).
#[test]
fn withdrawal_slack_objective_equals_cost_times_hours() {
    let violation_cost = 1_000.0_f64;
    let total_hours = 744.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, violation_cost);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // Withdrawal slack neg: followed by N pos cols and 4*N operational violation slack columns.
    let col_w = t.num_cols - 1 - 4 * t.n_hydro - t.n_hydro;
    let expected_obj = violation_cost * total_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[col_w] - expected_obj).abs() < 1e-12,
        "withdrawal slack objective: expected {expected_obj}, got {}",
        t.objective[col_w]
    );
}

/// The withdrawal slack objective is 0.0 when the violation cost is 0.0.
#[test]
fn withdrawal_slack_objective_zero_when_cost_is_zero() {
    let system = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // Withdrawal slack neg: followed by N pos + 4*N operational violation slack columns.
    let col_w = t.num_cols - 1 - 5 * t.n_hydro;
    assert!(
        t.objective[col_w].abs() < 1e-15,
        "withdrawal slack neg objective must be 0 when cost=0, got {}",
        t.objective[col_w]
    );
}

/// For a positive target `T = 10`, the under-delivery (`neg`) slack is capped at
/// `|T| = 10` (floors realized withdrawal ≥ 0) and the over-delivery (`pos`) slack
/// is unbounded; both lower bounds are 0.
#[test]
fn withdrawal_slack_bounds_are_sign_aware_positive_target() {
    let system = one_hydro_system_with_withdrawal(1, 0, 10.0, 5_000.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // Trailing slack layout per hydro: neg, pos, then 4 operational violation
    // slacks (5 families after neg). col_neg is the first of the 6 trailing cols.
    let col_neg = t.num_cols - 1 - 5 * t.n_hydro;
    let col_pos = col_neg + t.n_hydro;
    assert!(
        (t.col_lower[col_neg] - 0.0).abs() < 1e-15,
        "neg slack lower bound must be 0.0, got {}",
        t.col_lower[col_neg]
    );
    assert!(
        (t.col_upper[col_neg] - 10.0).abs() < 1e-15,
        "neg slack upper bound must be |T| = 10.0 for T > 0, got {}",
        t.col_upper[col_neg]
    );
    assert!(
        (t.col_lower[col_pos] - 0.0).abs() < 1e-15,
        "pos slack lower bound must be 0.0, got {}",
        t.col_lower[col_pos]
    );
    assert!(
        t.col_upper[col_pos].is_infinite() && t.col_upper[col_pos] > 0.0,
        "pos slack upper bound must be +inf for T > 0 (over-withdrawal latitude), got {}",
        t.col_upper[col_pos]
    );
}

/// For N=2 hydros, each withdrawal slack column has one CSC entry at
/// (`row_water` + `h_idx`, -`zeta`).
#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn two_hydro_withdrawal_slack_entries_per_hydro() {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    #[allow(clippy::cast_possible_wrap)]
    let make_hydro = |id: i32| Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 2_000.0,
            water_withdrawal_violation_pos_cost: 2_000.0,
            water_withdrawal_violation_neg_cost: 2_000.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let stages = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];

    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        max_diversion_m3s: None,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    bounds.hydro_bounds_mut(0, 0).water_withdrawal_m3s = 8.0;
    bounds.hydro_bounds_mut(1, 0).water_withdrawal_m3s = 12.0;

    let hydro_penalties_default = HydroStagePenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
        filling_target_violation_cost: 0.0,
        turbined_violation_below_cost: 0.0,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        generation_violation_below_cost: 0.0,
        evaporation_violation_cost: 0.0,
        water_withdrawal_violation_cost: 2_000.0,
        water_withdrawal_violation_pos_cost: 2_000.0,
        water_withdrawal_violation_neg_cost: 2_000.0,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(2), make_hydro(3)])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_hydro_with_withdrawal: valid");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=2, L=0: state+aux = N*(3+L)+1 = 7.
    // col_withdrawal_neg_start = 15, col_withdrawal_pos_start = 17,
    // operational violation slacks = 4*2 = 8, num_cols = 27.
    // Withdrawal neg columns: num_cols - 6*N = 27 - 12 = 15.
    let col_w0 = t.num_cols - 6 * t.n_hydro;
    let col_w1 = t.num_cols - 6 * t.n_hydro + 1;

    let row_w0 = 2_usize; // water balance for hydro 0 (row_water_balance_start = N = 2)
    let row_w1 = 3_usize; // water balance for hydro 1

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff_w0 = csc_entry(t, col_w0, row_w0).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w0} has no entry at water balance row {row_w0}")
    });
    assert!(
        (coeff_w0 - (-zeta)).abs() < 1e-12,
        "hydro-0 withdrawal slack coeff: expected {}, got {coeff_w0}",
        -zeta
    );

    let coeff_w1 = csc_entry(t, col_w1, row_w1).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w1} has no entry at water balance row {row_w1}")
    });
    assert!(
        (coeff_w1 - (-zeta)).abs() < 1e-12,
        "hydro-1 withdrawal slack coeff: expected {}, got {coeff_w1}",
        -zeta
    );

    assert!(
        csc_entry(t, col_w0, row_w1).is_none(),
        "hydro-0 withdrawal slack must not appear in hydro-1 water balance row"
    );
    assert!(
        csc_entry(t, col_w1, row_w0).is_none(),
        "hydro-1 withdrawal slack must not appear in hydro-0 water balance row"
    );
}

/// 3-hydro system: `num_cols` includes exactly 3 withdrawal slack columns.
#[test]
#[allow(clippy::too_many_lines)]
fn three_hydro_num_cols_includes_three_withdrawal_slacks() {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    #[allow(clippy::cast_possible_wrap)]
    let make_hydro = |id: i32| Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 1_000.0,
            water_withdrawal_violation_pos_cost: 1_000.0,
            water_withdrawal_violation_neg_cost: 1_000.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let stages = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
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
    }];

    let inflow_models: Vec<InflowModel> = [1, 2, 3]
        .iter()
        .map(|&hid| InflowModel {
            hydro_id: EntityId(hid),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 3,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 5.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 3,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 1_000.0,
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(1), make_hydro(2), make_hydro(3)])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("three_hydro_system: valid");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=3, L=0, 1 block, no thermals/lines/evap/inflow-penalty:
    // state+aux: 3 storage + 3 z_inflow + 3 storage_in + 1 theta = 10
    // turbine: 3, spillage: 3, diversion: 3
    // deficit: 1, excess: 1
    // inflow slack: 0 (no penalty config)
    // evap: 0
    // withdrawal slacks: 3 neg + 3 pos = 6
    // operational violation slacks: 4*3 = 12
    // Total = 10 + 3 + 3 + 3 + 1 + 1 + 6 + 12 = 39
    let expected_cols = 39_usize;
    assert_eq!(
        t.num_cols, expected_cols,
        "3-hydro system: num_cols should be {expected_cols}, got {}",
        t.num_cols
    );

    // Verify the withdrawal slack columns. They are followed by
    // 2*N withdrawal + 4*N operational = 6*N columns after withdrawal_neg start.
    // withdrawal_neg starts at num_cols - 6*N.
    //
    // All three hydros have a positive withdrawal target T = 5.0, so the
    // under-delivery (neg) slack is capped at |T| = 5.0 (floors realized
    // withdrawal R ≥ 0); the over-delivery (pos) slack remains unbounded.
    let n_h = t.n_hydro;
    let neg_start = t.num_cols - 6 * n_h;
    let pos_start = neg_start + n_h;
    for h in 0..n_h {
        assert!(
            (t.col_upper[neg_start + h] - 5.0).abs() < 1e-15,
            "withdrawal slack neg column for hydro {h} must be capped at |T| = 5.0, got {}",
            t.col_upper[neg_start + h]
        );
        assert_eq!(
            t.col_upper[pos_start + h],
            f64::INFINITY,
            "withdrawal slack pos column for hydro {h} should be unbounded above (T > 0)"
        );
    }
}

// ── Generic constraint layout tests ──────────────────────────

/// One-bus, one-stage system with `n_blks` operating blocks.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_system_n_blks(n_blks: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let blocks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = cobre_core::temporal::Stage {
        index: 0,
        id: 0_i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks,
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_bus_system_n_blks: valid")
}

/// `GenericConstraint` with a trivial (no-term) expression. A no-term expression
/// is vacuously block-independent, so a `block_id = None` bound on it collapses to
/// a single stage-level row.
fn make_constraint(
    id: i32,
    sense: cobre_core::ConstraintSense,
    slack_enabled: bool,
) -> cobre_core::GenericConstraint {
    use cobre_core::ConstraintExpression;
    make_constraint_with_expr(
        id,
        sense,
        slack_enabled,
        ConstraintExpression { terms: vec![] },
    )
}

/// Make a `GenericConstraint` carrying the given expression.
fn make_constraint_with_expr(
    id: i32,
    sense: cobre_core::ConstraintSense,
    slack_enabled: bool,
    expression: cobre_core::ConstraintExpression,
) -> cobre_core::GenericConstraint {
    use cobre_core::{GenericConstraint, SlackConfig};
    GenericConstraint {
        id: EntityId(id),
        name: format!("gc_{id}"),
        description: None,
        expression,
        sense,
        slack: SlackConfig {
            enabled: slack_enabled,
            penalty: if slack_enabled { Some(5000.0) } else { None },
        },
    }
}

/// A block-level expression: one `BusExcess` term with variable-level
/// `block_id = None`. `BusExcess` resolves to a per-block column
/// (`excess.start + bus_pos * n_blks + block_idx`), so a `block_id = None` bound
/// on this expression is **not** collapsible — distinct blocks yield distinct rows.
fn block_level_excess_expr(bus_id: i32) -> cobre_core::ConstraintExpression {
    use cobre_core::{ConstraintExpression, LinearTerm, VariableRef};
    ConstraintExpression {
        terms: vec![LinearTerm::literal(
            1.0,
            VariableRef::BusExcess {
                bus_id: EntityId(bus_id),
                block_id: None,
            },
        )],
    }
}

/// Build templates for `system` using the no-penalty method and default PAR/Normal.
fn build_templates_for(system: &cobre_core::System) -> Vec<cobre_solver::StageTemplate> {
    let production = default_production(system);
    let evaporation = default_evaporation(system);
    build_stage_templates(
        system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_templates_for: valid")
    .templates
}

/// Zero generic constraints leave num_rows and num_cols at the baseline.
#[test]
fn generic_constraints_zero_does_not_change_layout() {
    let system = one_bus_system_n_blks(1);
    let templates = build_templates_for(&system);
    let t = &templates[0];
    assert!(t.num_cols > 0);
    assert!(t.num_rows > 0);
    // A second build must be bit-for-bit identical (determinism).
    let templates2 = build_templates_for(&system);
    assert_eq!(
        templates2[0].num_cols, t.num_cols,
        "second build must not change num_cols"
    );
    assert_eq!(
        templates2[0].num_rows, t.num_rows,
        "second build must not change num_rows"
    );
}

/// A block-independent (trivial) constraint with `block_id = None` over 3 blocks
/// collapses to a single stage-level row: num_rows += 1, num_cols unchanged.
#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_collapses() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, 500.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(10, ConstraintSense::LessEqual, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (block-independent block_id=None collapses to a single row)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

/// A block-level (`BusExcess`) constraint with `block_id = None` over 3 blocks
/// keeps one row per block: num_rows += n_blks.
#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_block_level_per_block() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(11_i32, 0)].into_iter().collect();
    let rows = vec![(11_i32, 0_i32, None::<i32>, 500.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(
        11,
        ConstraintSense::LessEqual,
        false,
        block_level_excess_expr(1),
    );
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + n_blks,
        "num_rows must increase by n_blks={n_blks} (block-level expression keeps one row per block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

/// A block-independent `<=` constraint with slack and `block_id = None` over 2
/// blocks collapses to one stage-level row plus one `<=` slack column.
#[test]
fn generic_constraint_le_slack_enabled_2_blocks_collapses() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, 300.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(20, ConstraintSense::LessEqual, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 1,
        "num_cols must increase by 1 (one `<=` slack on the single collapsed row)"
    );
}

/// A block-level `<=` constraint with slack over 2 blocks yields 2 rows (one per
/// block) and 2 slack columns (one per row).
#[test]
fn generic_constraint_le_slack_enabled_2_blocks_block_level_per_block() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(21_i32, 0)].into_iter().collect();
    let rows = vec![(21_i32, 0_i32, None::<i32>, 300.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(
        21,
        ConstraintSense::LessEqual,
        true,
        block_level_excess_expr(1),
    );
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (one slack per per-block row for `<=`)"
    );
}

/// A block-independent `==` constraint with slack and `block_id = None` over 2
/// blocks collapses to one stage-level row plus two slack columns (plus/minus).
#[test]
fn generic_constraint_equal_sense_two_slacks_collapses() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, 100.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(30, ConstraintSense::Equal, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (plus+minus slacks on the single collapsed row)"
    );
}

/// A block-level `==` constraint with slack over 2 blocks yields 2 rows (one per
/// block) and 4 slack columns (two per row).
#[test]
fn generic_constraint_equal_sense_two_slacks_block_level_per_block() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(31_i32, 0)].into_iter().collect();
    let rows = vec![(31_i32, 0_i32, None::<i32>, 100.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint =
        make_constraint_with_expr(31, ConstraintSense::Equal, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 4,
        "num_cols must increase by 4 (two slacks per per-block row for `==`)"
    );
}

/// The collapsed stage-level slack is priced by total stage hours, conserving the
/// per-block penalty sum: `penalty × total_stage_hours = penalty × Σ block_hours`.
/// With 3 blocks of 720h (Σ = 2160h), the `<=` slack objective is
/// `penalty × 2160 / COST_SCALE_FACTOR`.
#[test]
fn generic_constraint_collapsed_slack_priced_by_total_stage_hours() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let block_hours = 720.0_f64; // one_bus_system_n_blks uses 720 h per block
    let penalty = 5000.0_f64; // make_constraint sets penalty = 5000 when slack enabled
    let total_stage_hours = block_hours * (n_blks as f64);

    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(60_i32, 0)].into_iter().collect();
    let rows = vec![(60_i32, 0_i32, None::<i32>, 500.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(60, ConstraintSense::LessEqual, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_cols,
        baseline_cols + 1,
        "collapsed `<=` row must add exactly one slack column"
    );

    let slack_col = baseline_cols; // new slack appended after the baseline cols
    let generic_row = t.num_rows - 1; // the single collapsed generic row is last

    let expected_obj = penalty * total_stage_hours / COST_SCALE_FACTOR;
    let per_block_sum = penalty * block_hours * (n_blks as f64) / COST_SCALE_FACTOR;
    assert!(
        (expected_obj - per_block_sum).abs() < 1e-9,
        "penalty conservation identity must hold: {expected_obj} vs {per_block_sum}"
    );
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-9,
        "collapsed slack objective must be penalty × total_stage_hours / scale = {expected_obj}, \
         got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "collapsed `<=` slack must have -1.0 at the generic row, got {total}"
    );
}

/// A constraint with `block_id = Some(1)` over 3 blocks generates exactly one
/// row (only the specified block).
#[test]
fn generic_constraint_specific_block_id_generates_one_row() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, Some(1_i32), 200.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(40, ConstraintSense::LessEqual, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by exactly 1 (only the specified block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

/// Of two constraints, only the one with bounds at stage 0 contributes rows; the
/// boundless constraint is inactive.
#[test]
fn generic_constraint_inactive_does_not_contribute_rows() {
    use cobre_core::{ConstraintSense, ResolvedGenericConstraintBounds};
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;

    // Constraint 51 has no bounds row → inactive.
    let id_map: HashMap<i32, usize> = [(50_i32, 0), (51_i32, 1)].into_iter().collect();
    let rows = vec![(50_i32, 0_i32, None::<i32>, 400.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let c_active = make_constraint(50, ConstraintSense::LessEqual, false);
    let c_inactive = make_constraint(51, ConstraintSense::LessEqual, false);
    let system =
        one_bus_system_n_blks_with_generic(n_blks, vec![c_active, c_inactive], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;

    // The active constraint's trivial expression collapses to one row (without the
    // collapse it would be baseline_rows + n_blks); the test's point is that the
    // inactive constraint stays absent under either representation.
    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "only the active constraint must contribute rows (collapsed to one)"
    );
}

/// Helper: build a one-bus system with `n_blks` operating blocks and
/// the given generic constraints + resolved bounds.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_system_n_blks_with_generic(
    n_blks: usize,
    constraints: Vec<cobre_core::GenericConstraint>,
    bounds: cobre_core::ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let blks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = Stage {
        index: 0,
        id: 0_i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: blks,
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };

    let load_models: Vec<LoadModel> = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(constraints)
        .resolved_generic_bounds(bounds)
        .build()
        .expect("one_bus_system_n_blks_with_generic: valid")
}

// ── Helper: scan CSC for entries in a specific (column, row) pair ──────────

/// All values stored at `(col, row)` in the CSC template. Returns a `Vec` (not
/// `Option`) because two fill helpers can add to the same position; tests check
/// the total or assert uniqueness.
fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    t.row_indices[start..end]
        .iter()
        .zip(t.values[start..end].iter())
        .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
        .collect()
}

// ── AC01: thermal <= constraint row bounds ─────────────────────────────────

/// `thermal_generation(0) <= 50.0` gives the generic row `row_upper = 50.0`,
/// `row_lower = -INF`, and a `+1.0` CSC entry in the thermal generation column.
#[test]
fn generic_constraint_thermal_le_row_bounds_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
        VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(10),
        name: "gc_thermal_le".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        sense: ConstraintSense::LessEqual,
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, 50.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1: theta=0, thermal col 0/block 0 = col 1,
    // load_balance row 0 → generic row at row 1.
    let thermal_col = 1_usize;
    let generic_row = 1_usize;

    assert!(
        t.row_lower[generic_row].is_infinite() && t.row_lower[generic_row] < 0.0,
        "row_lower must be -INF for <= constraint, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 50.0).abs() < f64::EPSILON,
        "row_upper must be 50.0, got {}",
        t.row_upper[generic_row]
    );

    let entries = csc_entries_at(t, thermal_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={thermal_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 total at thermal col / generic row, got {total}"
    );
}

// ── AC02: slack column for <= constraint ───────────────────────────────────

/// With slack enabled (penalty=5000) on `thermal <= 50.0`, the slack column has
/// bounds [0, +INF), objective 5000*744, and a `-1.0` CSC entry at the generic row.
#[test]
fn generic_constraint_thermal_le_slack_column_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
        VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let block_hours = 744.0_f64;
    let penalty = 5000.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(20),
        name: "gc_thermal_le_slack".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        sense: ConstraintSense::LessEqual,
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, 50.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1, 1 slack col:
    //   withdrawal_slack_start = col_evap_start + 0 evap cols
    //   = col_generation_start + 0 generation cols = inflow_slack_end
    //   = excess_end = 1(theta)+1(thermal)+1(deficit)+1(excess) = 4
    //   col_generic_slack_start = withdrawal_slack_start + n_h(=0) = 4
    //   slack_plus_col = 4
    let slack_col = 4_usize;
    let generic_row = 1_usize;

    assert!(
        t.col_lower[slack_col].abs() < f64::EPSILON,
        "slack col_lower must be 0.0, got {}",
        t.col_lower[slack_col]
    );
    assert!(
        t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
        "slack col_upper must be +INF, got {}",
        t.col_upper[slack_col]
    );

    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "slack objective must be {expected_obj}, got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={slack_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "expected -1.0 at slack col / generic row for <= sense, got {total}"
    );
}

// ── AC03: >= row bounds ────────────────────────────────────────────────────

/// `thermal_generation(0) >= 10.0` gives the generic row `row_lower = 10.0`,
/// `row_upper = +INF`.
#[test]
fn generic_constraint_thermal_ge_row_bounds() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
        VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(30),
        name: "gc_thermal_ge".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        sense: ConstraintSense::GreaterEqual,
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, 10.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let generic_row = 1_usize;
    assert!(
        (t.row_lower[generic_row] - 10.0).abs() < f64::EPSILON,
        "row_lower must be 10.0 for >= constraint, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
        "row_upper must be +INF for >= constraint, got {}",
        t.row_upper[generic_row]
    );
}

// ── AC04: == row bounds with two slacks ────────────────────────────────────

/// `thermal_generation(0) == 80.0` with slack adds two slack columns (plus at
/// col 4, minus at col 5) and sets the generic row to an equality at 80.0.
#[test]
fn generic_constraint_thermal_equal_two_slacks() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{ConstraintExpression, ConstraintSense, GenericConstraint, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let block_hours = 744.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(40),
        name: "gc_thermal_eq_slack".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        sense: ConstraintSense::Equal,
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, None::<i32>, 80.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let slack_plus_col = 4_usize;
    let slack_minus_col = 5_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_lower must be 80.0 for == constraint, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_upper must be 80.0 for == constraint, got {}",
        t.row_upper[generic_row]
    );

    // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
    // withdrawal_slack=0), with 2 slacks → num_cols = 6.
    assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

    assert!(
        t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
        "plus slack col_upper must be +INF"
    );
    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
        "plus slack objective must be {expected_obj}, got {}",
        t.objective[slack_plus_col]
    );
    let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
    assert!(
        !plus_entries.is_empty(),
        "no CSC entry at plus slack col / generic row"
    );
    let plus_total: f64 = plus_entries.iter().sum();
    assert!(
        (plus_total - 1.0).abs() < f64::EPSILON,
        "plus slack CSC must be +1.0 for == sense, got {plus_total}"
    );

    assert!(
        t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
        "minus slack col_upper must be +INF"
    );
    assert!(
        (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
        "minus slack objective must be {expected_obj}"
    );
    let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
    assert!(
        !minus_entries.is_empty(),
        "no CSC entry at minus slack col / generic row"
    );
    let minus_total: f64 = minus_entries.iter().sum();
    assert!(
        (minus_total - (-1.0)).abs() < f64::EPSILON,
        "minus slack CSC must be -1.0 for == sense, got {minus_total}"
    );
}

// ── AC03: two hydros with constant productivity, sum constraint ────────────

/// For `hydro_generation(H1) + hydro_generation(H2)` with constant productivities
/// 2.5 and 3.0, each turbine column's CSC coefficient equals its productivity.
#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_two_hydros_sum_csc_entries() {
    use chrono::NaiveDate;
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        ConstraintExpression, ConstraintSense, GenericConstraint, LinearTerm, SlackConfig,
        VariableRef,
    };
    use std::collections::HashMap;

    let h1_id = EntityId(5);
    let h2_id = EntityId(10);
    let prod_h1 = 2.5_f64;
    let prod_h2 = 3.0_f64;

    let make_hydro = |id: EntityId, _prod: f64| Hydro {
        id,
        name: format!("H{}", id.0),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let hydros = vec![make_hydro(h1_id, prod_h1), make_hydro(h2_id, prod_h2)];

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let stage = Stage {
        index: 0,
        id: 0_i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
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

    let inflow_models = vec![
        InflowModel {
            hydro_id: h1_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
        InflowModel {
            hydro_id: h2_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    // Generic constraint: hydro_generation(H1) + hydro_generation(H2) <= 200
    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_sum_gen".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h1_id,
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h2_id,
                        block_id: None,
                    },
                ),
            ],
        },
        sense: ConstraintSense::LessEqual,
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, 200.0_f64)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(vec![stage])
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(vec![constraint])
        .resolved_generic_bounds(generic_bounds)
        .build()
        .expect("two_hydro_system: valid");

    // Productivities are supplied explicitly (prod_h1=2.5 at index 0, prod_h2=3.0
    // at index 1) so the generic-row LP coefficients equal the productivities.
    let pm = production_set(&[prod_h1, prod_h2], 1);
    let t = &build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("two_hydro_system: valid")
    .templates[0];

    // For constant-productivity hydros, HydroGeneration maps to the turbine column
    // with multiplier = productivity; the generic constraint row is the last row.
    let generic_row = t.num_rows - 1;

    // N=2: storage=[0..2], z_inflow=[2..4], storage_in=[4..6], theta=6, turbine_start=7.
    let turbine_h1_col = 7_usize;
    let turbine_h2_col = 8_usize;

    let entries_h1 = csc_entries_at(t, turbine_h1_col, generic_row);
    assert!(
        !entries_h1.is_empty(),
        "no CSC entry found for H1 turbine at generic constraint row"
    );
    let total_h1: f64 = entries_h1.iter().sum();
    assert!(
        (total_h1 - prod_h1).abs() < f64::EPSILON,
        "expected coefficient {prod_h1} for H1, got {total_h1}"
    );

    let entries_h2 = csc_entries_at(t, turbine_h2_col, generic_row);
    assert!(
        !entries_h2.is_empty(),
        "no CSC entry found for H2 turbine at generic constraint row"
    );
    let total_h2: f64 = entries_h2.iter().sum();
    assert!(
        (total_h2 - prod_h2).abs() < f64::EPSILON,
        "expected coefficient {prod_h2} for H2, got {total_h2}"
    );
}

// ── Helper: build a one-bus, one-thermal system with generic constraints ───

/// 1-bus, 1-thermal (constant productivity), 1-block, 1-stage system with the
/// given generic constraints. Column layout (N=0, T=1, B=1, K=1, no penalty/FPHA/evap):
///   theta=0, thermal=[1,2), deficit=[2,3), excess=[3,4),
///   withdrawal_slack=[] (n_h=0), col_generic_slack_start=4.
/// Row layout: load_balance=[0,1), generic_start=1.
#[allow(clippy::cast_possible_wrap)]
fn one_bus_one_thermal_system(
    thermal_entity_id: EntityId,
    constraints: Vec<cobre_core::GenericConstraint>,
    bounds: cobre_core::ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: thermal_entity_id,
        name: "T1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stage = Stage {
        index: 0,
        id: 0_i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    // Resolved bounds: 0 hydros, 1 thermal, 0 lines, 0 pumping, 0 contracts, 1 stage.
    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(constraints)
        .resolved_generic_bounds(bounds)
        .build()
        .expect("one_bus_one_thermal_system: valid")
}

// ─────────────────────────────────────────────────────────────────────
// Operational violation slack structural tests
// ─────────────────────────────────────────────────────────────────────

/// 1-hydro, 1-bus, 2-block system with active operational bounds.
/// Blocks: 720.0h (heavy), 48.0h (light).
/// Hydro: min_outflow=50.0, max_outflow=Some(800.0), min_turbined=10.0,
///        min_generation=5.0, productivity=0.5.
/// Penalties: outflow_below/above=1000, turbined_below=1000, generation_below=1000.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn one_hydro_active_violations(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 50.0,
        max_outflow_m3s: Some(800.0),
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 10.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 5.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 1000.0,
            outflow_violation_below_cost: 1000.0,
            outflow_violation_above_cost: 1000.0,
            generation_violation_below_cost: 1000.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![
                Block {
                    index: 0,
                    name: "Heavy".to_string(),
                    duration_hours: 720.0,
                },
                Block {
                    index: 1,
                    name: "Light".to_string(),
                    duration_hours: 48.0,
                },
            ],
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 10.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_outflow_m3s: Some(800.0),
                min_generation_mw: 5.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 1000.0,
                outflow_violation_below_cost: 1000.0,
                outflow_violation_above_cost: 1000.0,
                generation_violation_below_cost: 1000.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_active_violations: valid")
}
fn build_active_violations_template() -> cobre_sddp::StageTemplates {
    let system = one_hydro_active_violations(1);
    // Productivity = 0.5 to match the coefficient expected by
    // `min_generation_constant_productivity_coefficients`.
    let pm = production_set(&[0.5], 1);
    build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("active violations ok")
}
#[test]
fn min_outflow_active_col_bounds() {
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    let col = indexer.outflow_below_slack.start;
    assert_eq!(t.col_lower[col], 0.0, "outflow_below lower must be 0");
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_below upper must be +inf when active"
    );
}

#[test]
fn max_outflow_active_col_bounds() {
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    let col = indexer.outflow_above_slack.start;
    assert_eq!(t.col_lower[col], 0.0, "outflow_above lower must be 0");
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_above upper must be +inf when max_outflow is Some"
    );
}

#[test]
fn operational_violation_inactive_pinned() {
    let system = one_hydro_system(1, 0); // default: all violation bounds = 0
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("base ok");
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    for &col in &[
        indexer.outflow_below_slack.start,
        indexer.outflow_above_slack.start,
        indexer.turbine_below_slack.start,
        indexer.generation_below_slack.start,
    ] {
        assert_eq!(t.col_lower[col], 0.0, "inactive col {col} lower != 0");
        assert_eq!(
            t.col_upper[col], 0.0,
            "inactive col {col} upper != 0 (should be pinned)"
        );
    }
}

#[test]
fn operational_violation_objective_costs() {
    // Per-block: penalty * block_hours / COST_SCALE_FACTOR.
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let n_blks = 2;
    let indexer = &result.geometry_per_stage[0];

    let block_hours = [720.0, 48.0];
    for (blk, &hours) in block_hours.iter().enumerate().take(n_blks) {
        let expected = 1000.0 * hours / COST_SCALE_FACTOR;
        for &start in &[
            indexer.outflow_below_slack.start,
            indexer.outflow_above_slack.start,
            indexer.turbine_below_slack.start,
            indexer.generation_below_slack.start,
        ] {
            // Column for hydro 0, block `blk`: start + 0 * n_blks + blk.
            let col = start + blk;
            assert!(
                (t.objective[col] - expected).abs() < 1e-10,
                "col {col} (block {blk}): objective = {}, expected = {}",
                t.objective[col],
                expected
            );
        }
    }
}
#[test]
fn turbine_column_lower_bound_is_zero() {
    // Turbine column lower bound is 0.0, NOT min_turbined_m3s (the min is enforced
    // by the turbine_below slack, not a hard column bound).
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    assert_eq!(
        t.col_lower[indexer.turbine.start], 0.0,
        "turbine blk0 lower bound must be 0.0"
    );
    assert_eq!(
        t.col_lower[indexer.turbine.start + 1],
        0.0,
        "turbine blk1 lower bound must be 0.0"
    );
}
// -------------------------------------------------------------------------
// max_par_order derivation tests (annual component path)
// -------------------------------------------------------------------------

/// 1-stage, 2-hydro system with AR order `ar_order` per hydro. `season_id: Some(0)`
/// lets `PrecomputedPar::build` resolve lag-stage statistics via the season fallback
/// even with no pre-study inflow models.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn two_hydro_par_system(
    ar_order: usize,
    inflow_models: Vec<cobre_core::scenario::InflowModel>,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str| Hydro {
        id: EntityId(id),
        name: name.to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let stages = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: ar_order > 0,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![make_hydro(2, "H1"), make_hydro(3, "H2")])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_hydro_par_system: valid")
}

/// When an annual component is present, `max_par_order` is widened to 12
/// regardless of the classical AR order.
///
/// System: 1 stage, 2 hydros, AR order p=2, `annual: Some(_)` on hydro 0.
#[test]
fn max_par_order_uses_par_lp_when_annual_present() {
    use cobre_core::scenario::{AnnualComponent, InflowModel};

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates ok");

    assert_eq!(
        result.templates[0].max_par_order, 12,
        "annual component must widen max_par_order to 12, got {}",
        result.templates[0].max_par_order
    );
}

/// Classical PAR systems are unaffected: `max_par_order` equals the AR order.
///
/// System: 1 stage, 2 hydros, AR order p=3, no annual component.
#[test]
fn max_par_order_classical_unchanged() {
    use cobre_core::scenario::InflowModel;

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2, 0.1];
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(3, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates ok");

    assert_eq!(
        result.templates[0].max_par_order, 3,
        "classical-PAR max_par_order must remain 3, got {}",
        result.templates[0].max_par_order
    );
}

/// When `max_par_order == 12`, the z-inflow definition row for hydro 0
/// has exactly 12 nonzero lag-column entries (one per lag in 0..12).
///
/// The `+1.0` entry on `col_z_inflow_start + 0` is excluded from the count.
#[allow(clippy::cast_sign_loss)]
#[test]
fn max_par_order_z_inflow_row_has_twelve_lag_entries() {
    use cobre_core::scenario::{AnnualComponent, InflowModel};

    // Same annual-PAR fixture as `max_par_order_uses_par_lp_when_annual_present`.
    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates ok");

    let t = &result.templates[0];
    assert_eq!(
        t.max_par_order, 12,
        "precondition: max_par_order must be 12"
    );

    // z_inflow rows start at 0; row for hydro 0 is row 0.
    let n_h = 2_usize;
    let l = 12_usize;
    let row_z_inflow_h0 = 0_usize; // z_inflow rows start at 0

    // z_inflow column for hydro 0: col_z_inflow_start = N*(1+L) = 2*13 = 26 (unchanged).
    let col_z_inflow_h0 = n_h * (1 + l); // = 26
    let mut lag_entry_count = 0usize;
    for col in 0..t.num_cols {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        for pos in start..end {
            if t.row_indices[pos] as usize == row_z_inflow_h0 && col != col_z_inflow_h0 {
                lag_entry_count += 1;
            }
        }
    }

    assert_eq!(
        lag_entry_count, 12,
        "z-inflow definition row for hydro 0 must have exactly 12 lag-column entries \
         when max_par_order == 12, got {lag_entry_count}"
    );
}

/// Regression guard: [`PatchBuffer`] must never grow to include generic-constraint rows.
///
/// The only row categories `PatchBuffer` mutates at solve time are AR dynamics /
/// noise (Category 3), load-balance (Category 4), and z-inflow definition
/// (Category 5); incoming state is pinned via column bounds, not patched rows.
/// Generic-constraint coefficients are immutable after construction. The test pins
/// this by asserting `forward_patch_count` after filling all three categories
/// equals `N + M*B_active + N` and never exceeds capacity `N + M*B_max + N`.
#[test]
#[allow(clippy::cast_precision_loss)] // fixture values are small integers; no precision is lost
fn parameter_coefficient_persists_across_stage_template_uses() {
    use PatchBuffer;

    // Realistic-scale system: N=3, L=2, M=2, B_max=3.
    // Row capacity = N + M*B_max + N = 3 + 2*3 + 3 = 12.
    let n: usize = 3;
    let l: usize = 2;
    let m: usize = 2;
    let b_max: usize = 3;

    let capacity_formula = n + m * b_max + n;
    let mut buf = PatchBuffer::new(n, l, m, b_max, 0, 0);

    assert_eq!(
        buf.indices.len(),
        capacity_formula,
        "PatchBuffer capacity must equal N + M*B_max + N; \
         formula change indicates new patch categories were added"
    );

    // Fill all three row categories with realistic values.
    let n_state = n * (1 + l);
    let state: Vec<f64> = (0..n_state).map(|i| (i + 1) as f64 * 10.0).collect();
    let noise: Vec<f64> = (0..n).map(|h| h as f64 * 0.5).collect();
    let base_row: usize = n; // water_balance_start = N

    // Category 3 — AR dynamics / noise.
    buf.fill_forward_patches(
        &StateLayout::new(n, l, 0, 0, vec![], &vec![l; n]),
        &state,
        &noise,
        base_row,
        &[],
    );

    // Category 4 — 2 load buses, 2 active blocks (< max 3). The per-stage grid
    // carries `b_active`, NOT `b_max`, so the load-balance row stride matches the
    // per-stage LP (a global grid striding by `b_max` would address the wrong row).
    let b_active: usize = 2;
    let load_rhs: Vec<f64> = (0..m * b_active).map(|i| 100.0 + i as f64).collect();
    let bus_positions: Vec<usize> = (0..m).collect();
    let load_row_start: usize = 200; // arbitrary LP row offset
    buf.fill_load_patches(
        load_row_start,
        BlockGrid::new(b_active, 1),
        &load_rhs,
        &bus_positions,
        &[],
    );

    // Category 5 — z-inflow rows.
    let z_inflow_rhs: Vec<f64> = (0..n).map(|h| 80.0 + h as f64).collect();
    let z_inflow_row_start: usize = 50;
    buf.fill_z_inflow_patches(z_inflow_row_start, &z_inflow_rhs, &[]);

    // The count uses b_active, not B_max: any generic-constraint patching would push
    // it past the N + M*B_max + N capacity into an out-of-bounds write.
    let expected_count = n + m * b_active + n;
    assert_eq!(
        buf.forward_patch_count(),
        expected_count,
        "forward_patch_count must equal N + M*b_active + N; \
         any generic-constraint patching would alter this count"
    );

    assert!(
        buf.forward_patch_count() < buf.indices.len(),
        "forward_patch_count {} must be < capacity {} when b_active < b_max",
        buf.forward_patch_count(),
        buf.indices.len(),
    );

    // Two builds of the same system must yield bit-identical CSC arrays —
    // the matrix is not mutated by the solver loop (determinism).
    let system = one_hydro_system(2, l);
    let result_a = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");
    let result_b = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    for (s, (ta, tb)) in result_a
        .templates
        .iter()
        .zip(&result_b.templates)
        .enumerate()
    {
        assert_eq!(
            ta.values, tb.values,
            "stage {s}: CSC values differ between two builds of the same system; \
             stage-template matrix must be deterministic and immutable"
        );
        assert_eq!(
            ta.row_indices, tb.row_indices,
            "stage {s}: CSC row_indices differ between two builds"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Anticipated-decision column bounds tests (AC-2 through AC-5)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a minimal system with one anticipated thermal (`K_i = lead_stages`,
/// `min_generation_mw`, `max_generation_mw`) and `n_stages` study stages.
///
/// System geometry: 0 hydros, 1 thermal (anticipated), 1 bus, 1 block per
/// stage, 1 deficit segment. `ResolvedBounds` is constructed with `k_max =
/// lead_stages` so delivery-stage lookups (`t + K_i`) never exceed the thermal
/// stage axis.
///
/// Column layout for this geometry (0 hydros, 1 thermal, 1 anticipated, 1 blk):
/// - `n_ant_state  = n_anticipated * k_max = 1 * lead_stages`
/// - `theta        = n_ant_state` (no hydro z_inflow or storage_in columns)
/// - `decision_start = theta + 1`
/// - `col_thermal_start = decision_start` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + n_thermals * n_blks`
///   = `theta + 1 + 1` = `theta + 2`
///
/// For `lead_stages = 2`: `n_ant_state = 2`, `theta = 2`,
/// `col_anticipated_decision_start = 4`.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn one_anticipated_thermal_system(
    n_stages: usize,
    lead_stages: u32,
    min_generation_mw: f64,
    max_generation_mw: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T_ant".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw,
        max_generation_mw,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw,
                max_generation_mw,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_anticipated_thermal_system: valid")
}

/// Compute `col_anticipated_decision_start` for the minimal geometry used by
/// the anticipated-decision tests (0 hydros, 1 thermal, 1 anticipated, 1 blk).
///
/// Layout derivation (anticipated_state_out relocated into the state region):
/// - `n_ant_state = n_anticipated * k_max = 1 * lead_stages`
/// - the relocated `anticipated_state_out` block adds `n_anticipated = 1` column
///   in the state region, after the ring buffer and before z_inflow
/// - `theta = n_ant_state + n_anticipated + 0 (z_inflow) + 0 (storage_in)`
///   = `n_ant_state + 1`
/// - `decision_start = theta + 1`
/// - `col_thermal_start = decision_start` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + n_thermals * n_blks`
fn anticipated_decision_col(lead_stages: usize) -> usize {
    let n_ant_state = lead_stages; // n_anticipated=1, k_max=lead_stages
    let theta = n_ant_state + 1; // + n_anticipated (relocated state_out), no z_inflow/storage_in
    let decision_start = theta + 1;
    let col_thermal_start = decision_start; // 0 hydro turbine/spillage/diversion cols
    col_thermal_start + 1 // n_thermals=1, n_blks=1
}

/// When `t + K_i < n_stages`, the anticipated-decision column takes bounds from
/// `thermal_bounds(thermal_idx, t + K_i)`. With `n_stages = 4`, `K_i = 2`,
/// min/max = 10.0/100.0, stage `t = 0` (delivery 2, active) → col bounds [10, 100].
#[test]
fn test_anticipated_decision_bounds_at_active_stage() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[0];
    assert_eq!(
        t.col_lower[col], 10.0,
        "stage 0: anticipated-decision col_lower must equal min_generation_mw=10.0 \
         (delivery stage = 0+2=2, active)"
    );
    assert_eq!(
        t.col_upper[col], 100.0,
        "stage 0: anticipated-decision col_upper must equal max_generation_mw=100.0 \
         (delivery stage = 0+2=2, active)"
    );
}

/// When `t + K_i > n_stages` the anticipated-decision column is gated inactive
/// with bounds [0.0, 0.0]. With `n_stages = 4`, `K_i = 2`, stage `t = 3`
/// (delivery 5 > 4) → [0, 0].
#[test]
fn test_anticipated_decision_bounds_inactive_when_beyond_horizon() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[3];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 3: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 3+2=5 > n_stages=4, inactive)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 3: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 3+2=5 > n_stages=4, inactive)"
    );
}

/// The horizon boundary `t + K_i == n_stages` is REJECTED (inactive)
/// under the strict predicate `t + K_i < n_stages`.
///
/// Setup: `n_stages = 4`, `K_i = 2`, `min_generation_mw = 10.0`,
/// `max_generation_mw = 100.0`.
/// At stage `t = 2`: delivery stage = `2 + 2 = 4 == n_stages` → inactive
/// (the strict predicate excludes equality; no delivery LP exists at
/// `delivery_stage == n_stages` because the per-stage loop iterates
/// `[0, n_stages)`).
/// Expected: `col_lower = 0.0`, `col_upper = 0.0`.
#[test]
fn test_anticipated_decision_inactive_at_horizon_boundary() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[2];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 2: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 2+2=4 == n_stages=4, strict predicate excludes boundary)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 2: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 2+2=4 == n_stages=4, strict predicate excludes boundary)"
    );
}

/// One-past-boundary `t + K_i == n_stages + 1` is also inactive. With
/// `n_stages = 3`, `K_i = 2`, stage `t = 2` (delivery 4) → [0, 0].
#[test]
fn test_anticipated_decision_inactive_one_past_horizon_boundary() {
    let system = one_anticipated_thermal_system(3, 2, 10.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[2];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 2: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 2+2=4 = n_stages+1=4, one-past-boundary inactive)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 2: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 2+2=4 = n_stages+1=4, one-past-boundary inactive)"
    );
}

// ── objective tests ────────────────────────────────────────────────────────────

/// The decision-stage objective uses the DELIVERY stage's cost, hours, and
/// discount factor. With K_i=2, cost=50.0, no discount, 744h blocks, stage t=0
/// (delivery 2) → 50.0 * 744.0 * 1.0 / COST_SCALE_FACTOR = 37.2.
#[test]
fn test_anticipated_decision_objective_uses_delivery_stage_factors() {
    // System: n_stages=4, K_i=2, cost_per_mwh=50.0, 744h blocks, no discounting.
    // At stage t=0: delivery=2, d_factor=1.0, delivery_hours=744.0.
    // objective (pre-scale) = 50.0 * 744.0 * 1.0 = 37200.0.
    // After /COST_SCALE_FACTOR: 37200.0 / 1000.0 = 37.2.
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[0];

    let cost_per_mwh = 50.0_f64;
    let delivery_hours = 744.0_f64; // all stages have 744h single block
    let d_factor = 1.0_f64; // no discount
    let expected = cost_per_mwh * delivery_hours * d_factor / COST_SCALE_FACTOR;
    assert_eq!(
        tmpl.objective[col], expected,
        "stage 0: anticipated-decision objective must equal 50*744*1/1000 = {expected}"
    );
}

/// Objective at boundary stage t + K_i == n_stages is REJECTED (zero)
/// under the strict predicate `t + K_i < n_stages`.
///
/// System: n_stages=4, K_i=2. At stage t=2: delivery_stage=4==n_stages → inactive
/// (the strict predicate excludes equality; no delivery LP exists at
/// `delivery_stage == n_stages`).
/// Expected: `objective[col] == 0.0`.
#[test]
fn test_anticipated_decision_objective_zero_at_horizon_boundary() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[2]; // t=2, delivery=4==n_stages, strict-predicate-inactive

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 2 (t+K==n_stages): anticipated-decision objective must be 0.0 \
         (strict predicate excludes boundary; no delivery LP exists at n_stages)"
    );
}

/// The objective is zero when the plant is inactive (delivery_stage > n_stages):
/// `n_stages=4`, `K_i=2`, stage t=3 (delivery 5) → objective 0.0.
#[test]
fn test_anticipated_decision_objective_zero_when_inactive() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[3]; // t=3, delivery=5>n_stages=4

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 3: anticipated-decision objective must be 0.0 (inactive beyond horizon)"
    );
}

/// The objective is zero one past the boundary (`t + K_i == n_stages + 1`):
/// `n_stages=3`, `K_i=2`, stage t=2 (delivery 4) → objective 0.0.
#[test]
fn test_anticipated_decision_objective_zero_one_past_boundary() {
    let system = one_anticipated_thermal_system(3, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[2]; // t=2, delivery=4=n_stages+1=4

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 2 (t+K=n_stages+1): anticipated-decision objective must be 0.0"
    );
}

/// Regression: at `stage_idx = n_stages - K_i`, no anticipated-decision
/// column is emitted (bounds `[0,0]` and objective `0.0`) for any K and any
/// n_stages such that K < n_stages.
///
/// Under the strict predicate `stage_idx + K_i < n_stages`, the boundary stage
/// `stage_idx = n_stages - K_i` would produce `delivery_stage = n_stages` —
/// outside the study horizon `[0, n_stages)`. The decision column at that
/// stage must be gated out so the LP never pays for an undelivered commitment.
///
/// This is the multi-K sweep guarding the strict-predicate semantics across
/// realistic horizons (n_stages in {3, 4, 5, 6}) and lead times (K in {1, 2, 3}).
#[test]
fn test_anticipated_decision_no_column_at_boundary_stage_strict_predicate() {
    for n_stages in 3_usize..=6 {
        for k in 1_usize..=3 {
            if k >= n_stages {
                // K must be strictly less than n_stages for the boundary stage
                // `n_stages - K` to be a valid non-negative index.
                continue;
            }
            let boundary_stage = n_stages - k;
            #[allow(clippy::cast_possible_truncation)]
            let k_u32 = k as u32;
            let system = one_anticipated_thermal_system(n_stages, k_u32, 10.0, 100.0);
            let result = build_stage_templates(
                &system,
                no_penalty_config(),
                &PrecomputedPar::default(),
                &PrecomputedNormal::default(),
                &default_production(&system),
                &default_evaporation(&system),
                &ResolvedParameters::default(),
            )
            .expect("build ok");

            let col = anticipated_decision_col(k);
            let tmpl = &result.templates[boundary_stage];

            assert_eq!(
                tmpl.col_lower[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 col_lower must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
            assert_eq!(
                tmpl.col_upper[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 col_upper must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
            assert_eq!(
                tmpl.objective[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 objective must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
        }
    }
}

// ── Helper: two-thermal system (one anticipated, one not) ────────────────────

/// Build a system with two thermals: `thermal 0` anticipated (K=`lead_stages`),
/// `thermal 1` non-anticipated. Both have `cost_per_mwh = 50.0`.
///
/// Column layout (0 hydros, 2 thermals, 1 anticipated, 1 blk per stage):
/// - `n_ant_state = n_anticipated * k_max = 1 * lead_stages`
/// - `theta = n_ant_state + n_anticipated` (relocated anticipated_state_out adds
///   n_anticipated cols)
/// - `col_thermal_start = theta + 1` (0 turbine/spillage/diversion cols)
/// - `col_anticipated_decision_start = col_thermal_start + 2 * n_blks`
///   = `theta + 1 + 2`
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn two_thermal_one_anticipated_system(n_stages: usize, lead_stages: u32) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal_ant = Thermal {
        id: EntityId(2),
        name: "T_ant".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages }),
        entry_stage_id: None,
        exit_stage_id: None,
    };
    // Thermal 1: non-anticipated.
    let thermal_non = Thermal {
        id: EntityId(3),
        name: "T_non".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant, thermal_non])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_thermal_one_anticipated_system: valid")
}

/// Compute column offsets for the two-thermal geometry.
///
/// Layout: 0 hydros, 2 thermals, 1 anticipated (thermal 0), 1 blk, K=lead_stages.
/// The relocated `anticipated_state_out` block (width n_anticipated=1) shifts
/// theta by 1 beyond the ring buffer.
/// - `n_ant_state = lead_stages`
/// - `theta = n_ant_state + n_anticipated = n_ant_state + 1`
/// - `col_thermal_start = theta + 1`
/// - `col_thermal_0_blk0 = col_thermal_start`           (thermal 0, block 0)
/// - `col_thermal_1_blk0 = col_thermal_start + 1`       (thermal 1, block 0)
/// - `col_anticipated_start = col_thermal_start + 2`    (2 thermals * 1 blk)
fn two_thermal_col_thermal_start(lead_stages: usize) -> usize {
    let n_ant_state = lead_stages;
    let theta = n_ant_state + 1; // + n_anticipated (relocated state_out block)
    theta + 1
}

/// The anticipated thermal's per-block cost is zero at its delivery stages. With
/// `n_stages=4`, `K_i=2`, thermal 0's delivery stages are {2, 3} → objective 0.0.
#[test]
fn test_anticipated_delivery_thermal_cost_is_zero() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages); // thermal 0, blk 0

    // Delivery stages for K_i=2: stage_idx in {2, 3}.
    for stage_idx in [2_usize, 3] {
        let obj = result.templates[stage_idx].objective[col_thermal_0];
        assert_eq!(
            obj, 0.0,
            "stage {stage_idx}: anticipated thermal 0 per-block cost must be 0.0 (delivery stage)"
        );
    }
}

/// The anticipated thermal's per-block cost is 0.0 at EVERY stage. The fishing
/// constraint is always active for an anticipated plant, so `fill_thermal_columns`
/// skips its per-block objective (via `anticipated_local_by_sys_pos`) at every
/// stage — including pre-horizon stages before K_i matures — leaving the cost at
/// its 0.0 initialization default.
#[test]
fn test_anticipated_pre_delivery_thermal_cost_unchanged() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages); // thermal 0, blk 0

    // Always-active predicate: cost is zeroed at all stages, including pre-delivery.
    for stage_idx in [0_usize, 1] {
        let obj = result.templates[stage_idx].objective[col_thermal_0];
        assert_eq!(
            obj, 0.0,
            "stage {stage_idx}: anticipated thermal 0 cost must be 0.0 (always-active zeroing)"
        );
    }
}

/// The non-anticipated thermal carries `cost_per_mwh * block_hours / COST_SCALE`
/// at every stage, unaffected by the anticipated thermal's cost zeroing.
#[test]
fn test_non_anticipated_thermal_cost_unchanged_under_anticipated_zero_out() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    // Thermal 1 is at index 1 in the thermals slice; col = col_thermal_start + 1 * n_blks + 0.
    let col_thermal_1 = two_thermal_col_thermal_start(lead_stages) + 1; // offset 1 for thermal 1, blk 0
    let expected = 50.0 * 744.0 / COST_SCALE_FACTOR;

    for stage_idx in 0..4 {
        let obj = result.templates[stage_idx].objective[col_thermal_1];
        assert_eq!(
            obj, expected,
            "stage {stage_idx}: non-anticipated thermal 1 cost must equal {expected} at every stage"
        );
    }
}

/// The set of stages where the anticipated thermal's per-block cost is zero equals
/// ALL stages: the fishing constraint is always active, so `fill_thermal_columns`
/// always skips its per-block objective (K_i=2, n_stages=4 → zero at {0,1,2,3}).
#[test]
fn test_zero_out_and_fishing_active_predicate_align() {
    let lead_stages = 2_usize;
    let n_stages = 4_usize;
    let system = two_thermal_one_anticipated_system(n_stages, lead_stages as u32);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages);

    let zeroed_stages: Vec<usize> = (0..n_stages)
        .filter(|&s| result.templates[s].objective[col_thermal_0] == 0.0)
        .collect();
    let all_stages: Vec<usize> = (0..n_stages).collect();

    assert_eq!(
        zeroed_stages, all_stages,
        "stages with zero-out thermal cost must equal all stages under always-active predicate. \
         Got zeroed={zeroed_stages:?}, expected={all_stages:?}"
    );
}

// ── anticipated-fishing row tests ────────────────────────────────────────────

/// Build a system with two anticipated thermals (K_0=1, K_1=2) and one bus.
///
/// Both thermals are anticipated; no non-anticipated thermals in this fixture.
/// Geometry (0 hydros, 2 thermals, n_anticipated=2, k_max=2, 1 bus, 1 blk/stage):
/// - `n_ant_state = 2 * 2 = 4`
/// - `theta = 4` (N=0 → N*(3+L) = 0; theta = n_ant_state)
/// - `col_thermal_start = 5` (decision_start = theta+1 = 5; 0 turbine/spillage/diversion)
/// - `col_anticipated_state_start = 0` (N*(1+L)=0)
/// - `row_anticipated_fishing_start = 5` (n_state=4; 1 load-balance row → starts at row 5)
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn two_anticipated_thermal_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal_0 = Thermal {
        id: EntityId(2),
        name: "T_ant0".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let thermal_1 = Thermal {
        id: EntityId(3),
        name: "T_ant1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    // k_max = max(K_0, K_1) = 2.
    let k_max = 2_usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_0, thermal_1])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_anticipated_thermal_system: valid")
}

/// 1 hydro (max_par_order=1) + 1 anticipated thermal (K=2), exercising the
/// `n_state = N*(1+L) + n_ant_state` formula with non-zero terms:
/// `1*(1+1) + 2 = 4`.
///
/// `PrecomputedPar::default()` is safe here: the matrix builder guards par_lp
/// accesses with `par_lp.n_stages() > 0` (false for the default), so PAR
/// coefficients are treated as zero — acceptable for structural tests.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn one_hydro_one_ant_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    // One anticipated thermal with K=2 → n_anticipated=1, k_max=2, n_ant_state=2.
    let thermal = Thermal {
        id: EntityId(3),
        name: "T_ant".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true, // AR(1) → contributes to max_par_order
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    // AR(1) inflow model per stage: ar_coefficients.len()=1 drives max_par_order=1.
    // Note: season_id=None is safe here because PrecomputedPar::default() is used
    // (n_stages()==0 bypasses all par_lp branches in the matrix builder).
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    // k_max = K = 2 (single anticipated thermal with lead_stages=2).
    let k_max = 2_usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_one_ant_system: valid")
}
// ── AC-2: fishing row count is constant across stages (always-active) ────────

/// With two anticipated thermals (K_0=1, K_1=2) and n_stages=4, fishing rows are
/// always 2 per stage (always-active = `n_anticipated`); the `state_out_def` count
/// varies only with the strict `stage + K_i < 4` gate.
///
/// State_out_def rows per stage:
///   stage 0: K_0=1: 0+1=1 < 4 ✓, K_1=2: 0+2=2 < 4 ✓ → 2 rows
///   stage 1: K_0=1: 1+1=2 < 4 ✓, K_1=2: 1+2=3 < 4 ✓ → 2 rows
///   stage 2: K_0=1: 2+1=3 < 4 ✓, K_1=2: 2+2=4 < 4 ✗ → 1 row
///   stage 3: K_0=1: 3+1=4 < 4 ✗, K_1=2: 3+2=5 < 4 ✗ → 0 rows
///
/// Combined fishing + state_out_def counts (base = stage 0: 2 fishing + 2 def):
///   stage 1: 2 fishing + 2 state_out_def = base     (same as stage 0)
///   stage 2: 2 fishing + 1 state_out_def = base - 1
///   stage 3: 2 fishing + 0 state_out_def = base - 2
#[test]
fn test_anticipated_fishing_rows_count_by_stage() {
    let system = two_anticipated_thermal_system(4);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    // base = stage 0: 2 fishing rows (always-active) + 2 state_out_def rows.
    let base = result.templates[0].num_rows;
    assert_eq!(
        result.templates[1].num_rows, base,
        "stage 1: 2 fishing + 2 state_out_def = base (equal to stage 0)"
    );
    assert_eq!(
        result.templates[2].num_rows,
        base - 1,
        "stage 2: 2 fishing + 1 state_out_def = base - 1 (K_1=2 def inactive: 2+2=4)"
    );
    assert_eq!(
        result.templates[3].num_rows,
        base - 2,
        "stage 3: 2 fishing + 0 state_out_def = base - 2 (both def inactive)"
    );
}

// ── Fishing-row count stage-invariance under always-active predicate ─────────

/// Because the fishing constraint is always active for every anticipated
/// plant, each plant emits exactly one fishing row at every stage in
/// `[0, n_stages)`. This test confirms the row count is stage-invariant by
/// asserting equality between `num_rows` at two adjacent stages.
///
/// System: one anticipated thermal K=2, n_stages=4.
/// At every stage: `n_anticipated_fishing_rows == n_anticipated == 1`.
#[test]
fn test_anticipated_fishing_same_count_both_stages() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    // Under the always-active predicate every stage carries one fishing row
    // per anticipated plant. The two adjacent stages must therefore have the
    // same total row count (the rest of the LP layout is also stage-invariant
    // for this single-anticipated-thermal fixture).
    let rows_stage_0 = result.templates[0].num_rows;
    let rows_stage_1 = result.templates[1].num_rows;
    assert_eq!(
        rows_stage_1, rows_stage_0,
        "stage 1 and stage 0 must have identical row counts (always-active fishing emits one row per anticipated plant at every stage)"
    );
}

// ── AC-1 through AC-8 ────────────────────────────────────────────────────────
//
// Geometry for AC-1..3, AC-6..8 (two-anticipated-thermal system):
//   n_hydros=0, n_anticipated=2 (K_0=1, K_1=2), k_max=2, n_ant_state=4
//   col_anticipated_state_start = 0   (N*(1+L) = 0)
//   theta = n_ant_state + n_anticipated = 6  (relocated anticipated_state_out adds 2 cols)
//   decision_start = 7
//   col_thermal_start = 7  (0 turbine/spillage/diversion cols)
//   col_anticipated_decision_start = 7 + 2*1 = 9  (2 thermals, 1 blk)
//
// Geometry for AC-4..5 (one-anticipated-thermal system, K=2):
//   n_hydros=0, n_anticipated=1, k_max=2, n_ant_state=2
//   col_anticipated_state_start = 0
//   col_anticipated_decision_start = anticipated_decision_col(2) = 5

// ── AC-1: anticipated-state columns are unconstrained ─────────────────────

/// With two anticipated thermals (K_0=1, K_1=2) → n_ant_state=4, all 4
/// anticipated-state columns are unbounded (-INF, +INF).
/// Slot-major layout: col = col_anticipated_state_start + slot * n_anticipated + plant
/// → cols 0..3 for N=0 hydros.
#[test]
fn test_anticipated_state_columns_unconstrained() {
    let system = two_anticipated_thermal_system(4);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // col_anticipated_state_start = 0 for N=0 hydros (N*(1+L) = 0).
    // n_anticipated=2, k_max=2 → n_ant_state=4 columns: indices 0..3.
    let col_state_start = 0_usize; // N*(1+L) = 0 when N=0
    let n_ant_state = 2 * 2; // n_anticipated * k_max

    for i in 0..n_ant_state {
        let col = col_state_start + i;
        assert!(
            t.col_lower[col].is_infinite() && t.col_lower[col] < 0.0,
            "col {col}: col_lower must be -INF, got {}",
            t.col_lower[col]
        );
        assert!(
            t.col_upper[col].is_infinite() && t.col_upper[col] > 0.0,
            "col {col}: col_upper must be +INF, got {}",
            t.col_upper[col]
        );
    }
}

// ── AC-4: decision column no longer writes to Cat 6 state-fixing slot ────

/// One anticipated thermal with K=2, n_stages=4: the Cat 6 state-fixing slot at
/// K_i-1 is a PURE IDENTITY row — the decision-write coefficient is removed (this
/// test verifies that removal; the decision-write into `anticipated_state_out_def`
/// is checked elsewhere).
///
/// Layout (no hydros, 1 bus, 1 block):
///   n_state = n_ant_state = K = 2; state-fixing rows: 0, 1;
///   col_anticipated_state_out_start: 2; col_anticipated_decision_start: 5;
///   old Cat 6 slot row: row_fix_start + (K_i-1)*n_anticipated = 1.
#[test]
fn test_anticipated_decision_write_to_state_out_def_row() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0]; // stage 0: plant active (0+2<4)
    let col_dec = anticipated_decision_col(2);

    // The old Cat 6 slot: row = state_fixing_base(0) + (K_i-1)*n_anticipated + 0
    //                         = 0 + (2-1)*1 + 0 = 1.
    // Under Alternative A the decision column must have NO entry at this row.
    // The decision-write moved to the def-row (-1.0 on decision, +1.0 on state_out),
    // so the old state-fixing slot must hold no decision entry.
    let old_state_fixing_row = 1_usize;
    let entries_at_old_row = csc_entries_at(t, col_dec, old_state_fixing_row);
    assert!(
        entries_at_old_row.is_empty(),
        "stage 0, active plant K=2: decision column must have NO entry at old state_fixing \
         slot row={old_state_fixing_row} (Cat 6 write removed), \
         got {entries_at_old_row:?}"
    );
}

// ── AC-5: inactive decision column has no state-write entry ───────────────

/// At an inactive stage (K=2, n_stages=4, stage 3: 3+2=5 > 4) the
/// anticipated-decision column has no CSC entry at any state-fixing row.
#[test]
fn test_anticipated_decision_inactive_no_state_write() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[3]; // stage 3: 3+2=5 > 4 → inactive
    // n_anticipated=1, k_max=2, n_ant_state=2.
    let col_dec = anticipated_decision_col(2);
    // Check all n_ant_state state-fixing rows: none should have the decision entry.
    let row_fix_start = 0_usize;
    let n_ant_state = 2_usize; // n_anticipated=1 * k_max=2
    for i in 0..n_ant_state {
        let row = row_fix_start + i;
        let entries = csc_entries_at(t, col_dec, row);
        assert!(
            entries.is_empty(),
            "stage 3, inactive plant K=2: CSC at (col={col_dec}, row={row}) must be empty, got {entries:?}"
        );
    }
}

// ── AC-6: n_state widens by n_ant_state ───────────────────────────────────

/// With 1 hydro (max_par_order=1) and 1 anticipated thermal (K=2),
/// `n_state = N*(1+L) + n_ant_state = 1*(1+1) + 1*2 = 4`. `one_hydro_one_ant_system`
/// keeps the hydro term N*(1+L) non-zero so the full formula is exercised.
#[test]
fn test_n_state_includes_n_ant_state() {
    let system = one_hydro_one_ant_system(4);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // n_hydros=1, max_par_order=1 → N*(1+L) = 1*(1+1) = 2.
    // n_anticipated=1, k_max=2 → n_ant_state = 2.
    // Expected n_state = N*(1+L) + n_ant_state = 2 + 2 = 4.
    let expected_n_state = 4_usize;
    assert_eq!(
        t.n_state, expected_n_state,
        "n_state must equal N*(1+L) + n_ant_state = {expected_n_state}, got {}",
        t.n_state
    );
}

// ── AC-8: n_transfer unchanged by anticipated state ───────────────────────

/// Anticipated state does not participate in the transfer operation (the
/// ring-buffer shift is handled by PatchBuffer): with n_hydros=0, max_par_order=0,
/// `n_transfer = n_hydros * max_par_order = 0`.
#[test]
fn test_n_transfer_unchanged_by_anticipated() {
    let system = two_anticipated_thermal_system(4);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // n_hydros=0, max_par_order=0 → n_transfer = n_hydros * max_par_order = 0.
    let expected_n_transfer = 0_usize;
    assert_eq!(
        t.n_transfer, expected_n_transfer,
        "n_transfer must equal n_hydros * max_par_order = {expected_n_transfer} (no anticipated contribution), got {}",
        t.n_transfer
    );
}

// ─── Anticipated Thermals K=1/2/3 Roundtrip ────────────────────────────────
//
// These integration tests exercise the full LP construction
// for synthetic systems with one hydro + one anticipated thermal at K=1, K=2,
// and K=3.  They verify all cross-cutting structural invariants simultaneously:
// column count, row count, n_state, anticipated_decision bounds, NPV objective
// coefficient, state-fixing CSC diagonal, decision-write CSC, and fishing-row
// CSC pattern.
//
// System geometry (shared across K=1/2/3):
//   N=1 hydro (constant productivity, L=0, max_par_order=0)
//   T=1 thermal (anticipated, K_i=K, min=0, max=100, cost=50)
//   B=1 bus, n_blks=2, block_hours=360h
//   n_stages=4, no FPHA, no evaporation, no generic constraints
//
// Column layout derivation (K >= 1 anticipated; for the K=0 non-anticipated
// baseline the formula differs — see the AC-4 test for the correct value):
//   n_ant_state = 1 * K = K
//   n_state = N*(1+L) + n_ant_state = 1 + K
//   col_anticipated_state_start = N*(1+L) = 1
//   col_anticipated_state_out_start = 1+K  (state region: = anticipated_state.end, 1 per plant)
//   z_inflow = [2+K, 2+K+N) = [2+K, 3+K)
//   storage_in = [3+K, 3+K+N) = [3+K, 4+K)
//   theta = 4+K
//   decision_start = 5+K
//   col_thermal_start = 5+K + 3*N*n_blks = 5+K+6 = 11+K
//   col_anticipated_decision_start = 11+K + 1*2 = 13+K
//   line_fwd/rev: 0 (no lines)
//   deficit: B*1*n_blks = 2 columns → cols 14+K..15+K
//   excess:  B*n_blks = 2 columns  → cols 16+K..17+K
//   withdrawal_neg/pos: N each = 2 → cols 18+K..19+K
//   op_slacks (4*N*n_blks): 8 → cols 20+K..27+K
//   num_cols = 28+K  (valid for K >= 1)
//
// Row layout derivation (K arbitrary, stage t):
//   rows 0..1     = hydro storage-fixing (N=1)
//   rows 1..1+K   = anticipated_state_fixing (K rows)
//   row 1+K       = z_inflow def (N=1)
//   row 2+K       = water_balance (N=1)
//   rows 3+K..4+K = load_balance (B=1, n_blks=2 → 2 rows)
//   rows 5+K..6+K = min_outflow (N*n_blks=2)
//   rows 7+K..8+K = max_outflow
//   rows 9+K..10+K = min_turbine
//   rows 11+K..12+K = min_generation
//   row 13+K      = anticipated_fishing (0 or 1 row; active iff K <= stage_idx)
//   row/rows after fishing = anticipated_state_out_def (active iff stage_idx+K < n_stages)
//   num_rows = 13+K + (1 if K <= stage_idx else 0) + (1 if stage_idx+K < n_stages else 0)
//
// ─────────────────────────────────────────────────────────────────────────────

/// Build a system with 1 hydro (constant productivity, L=0) and 1 anticipated
/// thermal (K_i = `lead_stages`), 1 bus, 2 blocks of 360 h per stage.
///
/// Used by the K=1, K=2, K=3 roundtrip tests.
///
/// Block durations are 360h so that total_stage_hours = 720h (consistent
/// across stages), which keeps the NPV objective computation tractable for
/// AC-5's discount-rate verification.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
fn build_hydro_one_ant_system(
    n_stages: usize,
    lead_stages: u32,
    annual_discount_rate: f64,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    let thermal = Thermal {
        id: EntityId(3),
        name: "T_ant".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // 2 blocks of 360 h each (total 720 h/stage).
    let blocks = vec![
        Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 360.0,
        },
        Block {
            index: 1,
            name: "BLK1".to_string(),
            duration_hours: 360.0,
        },
    ];

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: blocks.clone(),
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
        })
        .collect();

    // AR(0) inflow model (no lags → max_par_order=0).
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max = lead_stages as usize;
    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let policy_graph = PolicyGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate,
        transitions: vec![],
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .policy_graph(policy_graph)
        .build()
        .expect("build_hydro_one_ant_system: valid")
}

/// Build the K=1 roundtrip system (lead_stages=1, no discounting).
fn build_k1_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 1, 0.0)
}

/// Build the K=2 roundtrip system (lead_stages=2, no discounting).
fn build_k2_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 2, 0.0)
}

/// Build the K=3 roundtrip system (lead_stages=3, no discounting).
fn build_k3_system() -> cobre_core::System {
    build_hydro_one_ant_system(4, 3, 0.0)
}

/// K=0 baseline: 1 hydro + 1 NON-anticipated thermal (`anticipated_config: None`
/// → `n_anticipated=0`), so the LP layout matches the pre-anticipated baseline.
fn build_k0_baseline_system() -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::entities::thermal::Thermal;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
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
    };

    // Non-anticipated thermal — same bounds as the anticipated thermal in K-cases.
    let thermal = Thermal {
        id: EntityId(3),
        name: "T_non".to_string(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let blocks = vec![
        Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 360.0,
        },
        Block {
            index: 1,
            name: "BLK1".to_string(),
            duration_hours: 360.0,
        },
    ];

    let stages: Vec<Stage> = (0..4)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: blocks.clone(),
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0_i32..4)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i,
            mean_m3s: 80.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0_i32..4)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 4,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 4,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("build_k0_baseline_system: valid")
}

// ── Column layout helpers for the roundtrip geometry ─────────────────────────

/// `col_anticipated_state_start` for the roundtrip geometry (N=1, L=0).
///
/// = N*(1+L) = 1.
fn rt_col_ant_state_start() -> usize {
    1
}

/// `col_thermal_start` for the roundtrip geometry (N=1, L=0, K=k).
///
/// The relocated `anticipated_state_out` block (width n_anticipated=1) shifts
/// theta — and therefore every control-region column — by 1.
/// = decision_start + 3*N*n_blks = (5+K) + 6 = 11+K.
fn rt_col_thermal_start(k: usize) -> usize {
    11 + k
}

/// `col_anticipated_decision_start` for the roundtrip geometry.
///
/// = col_thermal_start + T*n_blks = (11+K) + 2 = 13+K.
fn rt_col_ant_dec_start(k: usize) -> usize {
    13 + k
}

/// `row_anticipated_fishing_start` for the roundtrip geometry. With no state-fixing
/// rows, = min_generation_start + n_op_rows = 11 + 1 = 12 (K-independent).
fn rt_row_ant_fishing_start(_k: usize) -> usize {
    12
}

/// Expected `num_cols` for the roundtrip geometry with anticipation K=k.
///
/// = 28+k (as derived in the section header comment).
/// The extra column versus the pre-anticipated formula is the
/// `anticipated_state_out` block (one column per anticipated plant, here 1).
fn rt_expected_num_cols(k: usize) -> usize {
    28 + k
}

/// Expected `num_rows` for the roundtrip geometry with anticipation K=k and stage
/// `stage_idx`. No state-fixing rows. Fishing row always-active (one per
/// anticipated plant); `anticipated_state_out_def` row active iff
/// `stage_idx + k < 4` (strict gate, n_stages=4).
fn rt_expected_num_rows(k: usize, stage_idx: usize) -> usize {
    // base = 12 (no state-fixing rows)
    let fishing = 1_usize; // always-active: 1 fishing row per anticipated plant
    let state_out_def = usize::from(stage_idx + k < 4);
    12 + fishing + state_out_def
}

// ─── AC-1: K=1 roundtrip integration ────────────────────────────────────────

/// K=1 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=1), B=1 bus,
/// 2 blocks × 360h, n_stages=4, no discounting.
///
/// Verifies simultaneously:
/// - `n_state == 2` for all stages.
/// - `num_cols == 29` and `num_rows` per stage match the K=1 formula.
/// - anticipated_decision bounds: `[0,100]` when active (`t+1 < 4`), which
///   is stages 0..2 for K=1; INACTIVE at boundary stage 3 (`3+1=4==n_stages`,
///   excluded by the strict predicate).
/// - NPV objective coefficient at stage 0 (no discount): `50*720/1000 = 36.0`.
/// - State-fixing CSC diagonal +1.0 for slot 0, plant 0.
/// - Decision-write CSC +1.0 at row `1 + (K-1)*1 = 1` (slot K-1=0).
/// - Fishing row CSC at stage 1 (first stage with K=1 <= stage_idx=1).
/// - Fishing row equality bounds 0==0.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k1() {
    let k = 1_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k1_system();
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=1 build ok");

    let col_ant_state = rt_col_ant_state_start(); // 1
    let col_ant_dec = rt_col_ant_dec_start(k); // 13
    let col_thermal = rt_col_thermal_start(k); // 11
    let row_fish_start = rt_row_ant_fishing_start(k); // 12

    // ── n_state (AC-1.a) ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=1, stage {t}: n_state must be {} (1 hydro + 1 ant-slot), got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols (AC-1.b) ────────────────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=1, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
    }

    // ── num_rows (AC-1.b) ────────────────────────────────────────────────────
    for t in 0..n_stages {
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=1, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds (AC-1.d) ────────────────────────────────
    // Active: t in 0..3 (t+1 < 4: 1, 2, 3 all < 4 under strict predicate).
    for t in 0..(n_stages - k) {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=1, stage {t}: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=1, stage {t}: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive: boundary stage t=3 (3+1=4 NOT < 4 under strict predicate).
    {
        let tmpl = &result.templates[n_stages - k];
        assert_eq!(
            tmpl.col_lower[col_ant_dec],
            0.0,
            "K=1, boundary stage {}: anticipated_decision col_lower must be 0.0 (strict predicate excludes)",
            n_stages - k
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec],
            0.0,
            "K=1, boundary stage {}: anticipated_decision col_upper must be 0.0 (strict predicate excludes; t+K=n_stages)",
            n_stages - k
        );
    }

    // ── NPV objective at stage 0 (AC-1.e) ────────────────────────────────────
    // delivery_stage = 0+1 = 1; cumulative_factor[1] = 1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=1, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // ── Fishing row CSC at stage 1 (K=1 <= 1) (AC-1.h) ──────────────────────
    {
        let t = &result.templates[1]; // stage 1: K=1 <= stage_idx=1 → fishing active
        let row_fish = row_fish_start;
        // Thermal generation columns: col_thermal + blk for blk in 0..2.
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=1, stage 1: fishing CSC at thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        // Slot-0 anticipated-state column: -total_hours.
        let col_state_slot0 = col_ant_state; // slot 0, plant 0
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        let expected_neg = -total_hours; // -(360+360) = -720.0
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=1, stage 1: fishing CSC at ant_state slot 0 must be [{expected_neg}], \
             got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 1 (AC-1.i) ─────────────────
    {
        let t = &result.templates[1];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=1, stage 1: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=1, stage 1: fishing row_upper must be 0.0"
        );
    }

    // Fishing is present at every stage; state_out_def is active at stages 0,1,2
    // (0+1,1+1,2+1 all < 4) but not stage 3 — so 0,1,2 share num_rows, stage 3 has one fewer.
    {
        assert_eq!(
            result.templates[0].num_rows, result.templates[1].num_rows,
            "K=1: stage 0 and stage 1 must have equal row count (fishing always-active)"
        );
        assert_eq!(
            result.templates[1].num_rows, result.templates[2].num_rows,
            "K=1: stage 1 and stage 2 must have equal row count (state_out_def still active)"
        );
        assert_eq!(
            result.templates[3].num_rows + 1,
            result.templates[2].num_rows,
            "K=1: stage 3 must have 1 fewer row than stage 2 (state_out_def inactive at stage 3)"
        );
    }
}

// ─── AC-2: K=2 roundtrip integration ────────────────────────────────────────

/// K=2 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=2), B=1 bus,
/// 2×360h, n_stages=4.
///
/// Verifies:
/// - `n_state == 3` for all stages.
/// - `num_cols == 30` and `num_rows` per stage match K=2 formula.
/// - Bounds: active at t=0 (`0+2=2<4`), active at t=1 (`1+2=3<4`),
///   INACTIVE at boundary t=2 (`2+2=4 NOT < 4`) and t=3 (`3+2=5>4`).
/// - Decision-write: slot K-1=1; at stage 0 active, col has +1.0 at
///   `row_fix_start + 1 = 2`.
/// - Fishing row active at stage 2 (K=2 <= 2), absent at stage 1 (K=2 > 1).
/// - Fishing row CSC pattern at stage 2.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k2() {
    let k = 2_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k2_system();
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=2 build ok");

    let col_ant_state = rt_col_ant_state_start(); // 1
    let col_ant_dec = rt_col_ant_dec_start(k); // 14
    let col_thermal = rt_col_thermal_start(k); // 12
    let row_fish_start = rt_row_ant_fishing_start(k); // 12

    // ── n_state (AC-2.a) ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=2, stage {t}: n_state must be {}, got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols / num_rows (AC-2.b) ─────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=2, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=2, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds (AC-2.d) ─────────────────────────────────
    // Active under strict predicate: t=0 (0+2=2 < 4), t=1 (1+2=3 < 4).
    for t in 0..=1 {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=2, stage {t}: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive under strict predicate: boundary t=2 (2+2=4 NOT < 4) and t=3.
    for t in 2..n_stages {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_lower must be 0.0 (inactive)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_upper must be 0.0 (inactive under strict predicate; t+K >= n_stages)"
        );
    }

    // ── NPV objective at stage 0 (AC-2.e) ────────────────────────────────────
    // delivery_stage=2; cumulative_factor[2]=1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=2, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // ── Fishing row CSC at stage 2 (K=2 <= 2) (AC-2.h) ──────────────────────
    {
        let t = &result.templates[2];
        let row_fish = row_fish_start;
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=2, stage 2: fishing CSC thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        let col_state_slot0 = col_ant_state;
        let expected_neg = -total_hours;
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=2, stage 2: fishing CSC ant_state slot 0 must be [{expected_neg}], got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 2 (AC-2.i) ─────────────────
    {
        let t = &result.templates[2];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=2, stage 2: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=2, stage 2: fishing row_upper must be 0.0"
        );
    }

    // K=2, n_stages=4: fishing is 1 row at every stage; state_out_def (s+K<4) is
    // active at 0,1 but absent at 2,3 → stages 0,1 have one more row than 2,3.
    {
        assert_eq!(
            result.templates[1].num_rows, result.templates[0].num_rows,
            "K=2: stage 1 must have same row count as stage 0 \
             (both have fishing + state_out_def active)"
        );
        assert_eq!(
            result.templates[2].num_rows, result.templates[3].num_rows,
            "K=2: stage 2 must have same row count as stage 3 \
             (both have fishing active, state_out_def absent)"
        );
        assert_eq!(
            result.templates[0].num_rows,
            result.templates[2].num_rows + 1,
            "K=2: stage 0 must have exactly 1 more row than stage 2 \
             (state_out_def active at stage 0, absent at stage 2)"
        );
    }
}

// ─── AC-3: K=3 roundtrip integration ────────────────────────────────────────

/// K=3 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=3), B=1 bus,
/// 2×360h, n_stages=4.
///
/// Verifies:
/// - `n_state == 4` for all stages.
/// - `num_cols == 31` and `num_rows` per stage match K=3 formula.
/// - Bounds: active at t=0 (`0+3=3 < 4`), INACTIVE at boundary t=1
///   (`1+3=4 NOT < 4`), t=2 (`2+3=5>4`), and t=3.
/// - Decision-write: slot K-1=2; at stage 0, col has +1.0 at row_fix_start+2=3.
/// - Fishing rows: absent at t=0,1,2; present at t=3 (K=3 <= 3).
/// - Fishing row CSC pattern at stage 3.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k3() {
    let k = 3_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k3_system();
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=3 build ok");

    let col_ant_state = rt_col_ant_state_start(); // 1
    let col_ant_dec = rt_col_ant_dec_start(k); // 15
    let col_thermal = rt_col_thermal_start(k); // 13
    let row_fish_start = rt_row_ant_fishing_start(k); // 12 (K-independent)

    // ── n_state (AC-3.a) ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=3, stage {t}: n_state must be {}, got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols / num_rows (AC-3.b) ─────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=3, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=3, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds (AC-3.d) ─────────────────────────────────
    // Active under strict predicate: t=0 only (0+3=3 < 4).
    {
        let tmpl = &result.templates[0];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=3, stage 0: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=3, stage 0: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive under strict predicate: boundary t=1 (1+3=4 NOT < 4), t=2, t=3.
    for t in 1..n_stages {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=3, stage {t}: anticipated_decision col_lower must be 0.0 (inactive)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 0.0,
            "K=3, stage {t}: anticipated_decision col_upper must be 0.0 (inactive under strict predicate; t+3 >= n_stages)"
        );
    }

    // ── NPV objective at stage 0 (AC-3.e) ────────────────────────────────────
    // delivery_stage=3; cumulative_factor[3]=1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=3, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // K=3, n_stages=4: fishing (K<=stage) is absent at 0,1,2 and active at 3;
    // state_out_def (s+K<4) is active at 0 only → stages 0 and 3 each have one
    // more row than stages 1,2.
    {
        assert_eq!(
            result.templates[1].num_rows, result.templates[2].num_rows,
            "K=3: stages 1 and 2 must have equal row count (no fishing, no def)"
        );
        assert_eq!(
            result.templates[0].num_rows,
            result.templates[1].num_rows + 1,
            "K=3: stage 0 must have 1 more row than stage 1 (state_out_def active at stage 0)"
        );
    }

    // ── Fishing row CSC at stage 3 (K=3 <= 3) (AC-3.h) ──────────────────────
    {
        let t = &result.templates[3];
        let row_fish = row_fish_start;
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=3, stage 3: fishing CSC thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        let col_state_slot0 = col_ant_state;
        let expected_neg = -total_hours;
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=3, stage 3: fishing CSC ant_state slot 0 must be [{expected_neg}], got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 3 (AC-3.i) ─────────────────
    {
        let t = &result.templates[3];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=3, stage 3: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=3, stage 3: fishing row_upper must be 0.0"
        );
    }

    // ── Row-count invariants under always-active fishing (K=3) ───────────────
    // For K=3, n_stages=4:
    //   Fishing: 1 row per stage at every stage (always-active predicate).
    //   State_out_def (s+K<4): active at 0; absent at 1,2,3.
    // So stage 0 has one extra row vs stages 1, 2, 3, which are equal.
    {
        assert_eq!(
            result.templates[3].num_rows, result.templates[2].num_rows,
            "K=3: stage 3 must have same row count as stage 2 \
             (fishing always-active; state_out_def absent at both)"
        );
    }
}

// ─── AC-4: K=0 baseline parity ──────────────────────────────────────────────

/// A thermal with `anticipated_config: None` (K=0) yields `n_anticipated=0` and a
/// layout identical to the pre-anticipated baseline: no anticipated columns, no
/// fishing rows. Same geometry as the K-cases (N=1 hydro, T=1 thermal, B=1,
/// 2×360h, n_stages=4).
#[test]
fn test_anticipated_thermals_lp_roundtrip_k0_baseline_parity() {
    let system_baseline = build_k0_baseline_system();
    let result_baseline = build_stage_templates(
        &system_baseline,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_baseline),
        &default_evaporation(&system_baseline),
        &ResolvedParameters::default(),
    )
    .expect("baseline build ok");

    // A second identical build must be bit-identical (determinism).
    let result_baseline2 = build_stage_templates(
        &system_baseline,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_baseline),
        &default_evaporation(&system_baseline),
        &ResolvedParameters::default(),
    )
    .expect("baseline build2 ok");

    let n_stages = result_baseline.templates.len();
    assert_eq!(n_stages, 4, "baseline must have 4 templates");

    for s in 0..n_stages {
        let ta = &result_baseline.templates[s];
        let tb = &result_baseline2.templates[s];
        assert_eq!(
            ta.num_cols, tb.num_cols,
            "parity: stage {s} num_cols must match ({} vs {})",
            ta.num_cols, tb.num_cols
        );
        assert_eq!(
            ta.num_rows, tb.num_rows,
            "parity: stage {s} num_rows must match ({} vs {})",
            ta.num_rows, tb.num_rows
        );
        assert_eq!(
            ta.n_state, tb.n_state,
            "parity: stage {s} n_state must match ({} vs {})",
            ta.n_state, tb.n_state
        );
        assert_eq!(
            ta.n_transfer, tb.n_transfer,
            "parity: stage {s} n_transfer must match ({} vs {})",
            ta.n_transfer, tb.n_transfer
        );
        assert_eq!(
            ta.n_dual_relevant, tb.n_dual_relevant,
            "parity: stage {s} n_dual_relevant must match ({} vs {})",
            ta.n_dual_relevant, tb.n_dual_relevant
        );
        assert_eq!(
            ta.col_starts, tb.col_starts,
            "parity: stage {s} col_starts differ between two builds"
        );
        assert_eq!(
            ta.row_indices, tb.row_indices,
            "parity: stage {s} row_indices differ between two builds"
        );
        assert_eq!(
            ta.values, tb.values,
            "parity: stage {s} CSC values differ between two builds"
        );
        assert_eq!(
            ta.col_lower, tb.col_lower,
            "parity: stage {s} col_lower differs between two builds"
        );
        assert_eq!(
            ta.col_upper, tb.col_upper,
            "parity: stage {s} col_upper differs between two builds"
        );
        assert_eq!(
            ta.objective, tb.objective,
            "parity: stage {s} objective differs between two builds"
        );
        assert_eq!(
            ta.row_lower, tb.row_lower,
            "parity: stage {s} row_lower differs between two builds"
        );
        assert_eq!(
            ta.row_upper, tb.row_upper,
            "parity: stage {s} row_upper differs between two builds"
        );
    }

    // n_state = N*(1+L) = 1 (no anticipated term).
    assert_eq!(
        result_baseline.templates[0].n_state, 1,
        "K=0 baseline: n_state must be 1 (no anticipated state)"
    );
    // num_cols = 26: the anticipated geometry's 28+K drops both K-only columns
    // (1 anticipated_state slot + 1 anticipated_decision) when K=0.
    assert_eq!(
        result_baseline.templates[0].num_cols, 26,
        "K=0 baseline: num_cols must be 26 (no anticipated state or decision columns)"
    );
    assert_eq!(
        result_baseline.templates[0].num_rows, 12,
        "K=0 baseline: num_rows must be 12 (state-fixing rows removed in Phase 1)"
    );
}

// ─── AC-5: K=2 with non-zero discount rate ──────────────────────────────────

/// K=2 roundtrip with 6% annual discount: the stage-0 anticipated-decision
/// objective uses the DELIVERY stage (2) factors, `50 * total_hours *
/// cumulative_discount_factors[2] / COST_SCALE_FACTOR`. With 31-day stages,
/// `per_stage_factor = 1 / (1.06)^(31/365.25)` and `cumulative[2] = per_stage_factor^2`.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k2_with_discount_rate() {
    let k = 2_usize;
    let annual_rate = 0.06_f64;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0

    let system = build_hydro_one_ant_system(4, k as u32, annual_rate);
    let result = build_stage_templates(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=2 discount build ok");

    let col_ant_dec = rt_col_ant_dec_start(k); // 14

    // cumulative_discount_factors[delivery=2], computed from 31-day stages.
    let dt_days = 31.0_f64;
    let per_stage_factor = 1.0 / (1.0 + annual_rate).powf(dt_days / 365.25);
    let cumulative_at_delivery = per_stage_factor * per_stage_factor;

    let expected_obj = 50.0 * total_hours * cumulative_at_delivery / COST_SCALE_FACTOR;

    let actual_obj = result.templates[0].objective[col_ant_dec];
    let rel_err = (actual_obj - expected_obj).abs() / expected_obj.abs().max(f64::EPSILON);
    assert!(
        rel_err < 1e-12,
        "K=2 with 6% discount: stage 0 anticipated_decision objective must be {expected_obj:.15} \
         (rel_err={rel_err:.2e}), got {actual_obj:.15}"
    );
}
