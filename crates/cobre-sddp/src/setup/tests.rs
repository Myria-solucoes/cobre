use super::StudySetup;
use crate::hydro_models::{PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StateLayout;

use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractStageBounds, HydroStageBounds,
    HydroStagePenalties, LineStageBounds, LineStagePenalties, NcsStagePenalties,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedPenalties,
    ThermalStageBounds,
};
use cobre_core::{
    EntityId, HydroPastInflows, InitialConditions, SystemBuilder,
    entities::{
        bus::{Bus, DeficitSegment},
        hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        thermal::{AnticipatedConfig, Thermal},
    },
    scenario::{InflowModel, LoadModel, SamplingScheme},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
    RawClassConfigEntry, RawScenarioSourceConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

/// Build a minimal system with 1 bus, 1 thermal, 1 hydro, and `n_stages`
/// study stages (each with 1 block). All bounds and penalties are set to
/// sensible non-zero defaults so `build_stage_templates` succeeds.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn minimal_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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
            hydro_id: EntityId(3),
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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("minimal_system: valid")
}

/// Variant of [`minimal_system`] whose single hydro is FPHA without any
/// VHA rows or `specific_productivity_mw_per_m3s_per_m`, so the energy
/// conversion gate must reject it.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn minimal_fpha_misconfigured_system(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: EntityId(3),
        name: "H_FPHA_BAD".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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
            hydro_id: EntityId(3),
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
            n_thermals: 1,
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
                water_withdrawal_m3s: 0.0,
            },
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
                storage_violation_below_cost: 500.0,
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

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("minimal_fpha_misconfigured_system: valid")
}

/// Build a minimal valid [`Config`] with a single iteration-limit stopping rule.
fn minimal_config(forward_passes: u32, max_iterations: u32) -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            forward_passes: Some(forward_passes),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: max_iterations,
            }]),
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Build a minimal valid [`Config`] with the given per-class scheme overrides.
///
/// `inflow_scheme`, `load_scheme`, and `ncs_scheme` are optional strings
/// matching the JSON schema values (`"in_sample"`, `"historical"`, `"external"`,
/// `"out_of_sample"`). `None` leaves the class defaulting to `in_sample`.
fn minimal_config_with_schemes(
    forward_passes: u32,
    max_iterations: u32,
    inflow_scheme: Option<&str>,
    load_scheme: Option<&str>,
    ncs_scheme: Option<&str>,
) -> Config {
    // A seed is required when any class uses a non-in-sample scheme.
    let needs_seed = inflow_scheme.is_some_and(|s| s != "in_sample")
        || load_scheme.is_some_and(|s| s != "in_sample")
        || ncs_scheme.is_some_and(|s| s != "in_sample");
    let scenario_source = RawScenarioSourceConfig {
        seed: if needs_seed { Some(42) } else { None },
        historical_years: None,
        inflow: inflow_scheme.map(|s| RawClassConfigEntry {
            scheme: s.to_string(),
        }),
        load: load_scheme.map(|s| RawClassConfigEntry {
            scheme: s.to_string(),
        }),
        ncs: ncs_scheme.map(|s| RawClassConfigEntry {
            scheme: s.to_string(),
        }),
    };
    let mut config = minimal_config(forward_passes, max_iterations);
    config.training.scenario_source = Some(scenario_source);
    config
}

/// Given a minimal valid system (1 hydro, 1 thermal, 1 bus, 2 stages),
/// when `StudySetup::new()` is called, then it returns `Ok` and
/// `stage_templates()` returns a non-empty slice.
#[test]
fn new_minimal_valid_system_returns_ok() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    );
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    let setup = result.unwrap();
    assert!(!setup.stage_data.stage_templates.templates.is_empty());
}

/// Given a system with zero study stages, when `StudySetup::new()` is
/// called, then it returns `Err` containing the substring "no study stages".
#[test]
fn new_zero_stages_returns_validation_error() {
    let system = minimal_system(0);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    );
    assert!(result.is_err(), "expected Err, got Ok");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("no study stages"),
        "error message should contain 'no study stages': {msg}"
    );
}

/// Given a valid `StudySetup`, accessor methods return the expected values.
#[test]
fn accessor_methods_return_expected_values() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 50);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert_eq!(setup.stage_data.stage_templates.templates.len(), n_stages);
    assert_eq!(setup.stage_data.stage_templates.base_rows.len(), n_stages);

    assert_eq!(setup.loop_params.seed, 42);
    assert_eq!(setup.loop_params.forward_passes, 2);
    assert_eq!(setup.loop_params.max_iterations, 50);
    assert_eq!(setup.simulation_config.n_scenarios, 0); // simulation disabled by default
    assert_eq!(setup.policy_path, "./policy");

    assert_eq!(setup.stage_data.block_counts_per_stage.len(), n_stages);
    assert!(setup.loop_params.max_blocks > 0);

    assert_eq!(setup.methodology.horizon.num_stages(), n_stages);

    assert_eq!(setup.cut_management.risk_measures.len(), n_stages);

    assert_eq!(setup.fcf.pools.len(), n_stages);

    assert_eq!(setup.stage_data.entity_counts.hydro_ids.len(), 1);
    assert_eq!(setup.stage_data.entity_counts.thermal_ids.len(), 1);
}

/// FCF is accessible mutably via `fcf_mut()`.
#[test]
fn fcf_mut_allows_cut_insertion() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let n_state = setup.stage_data.state.n_state;
    let coefficients = vec![1.0_f64; n_state];
    setup.fcf.add_cut(0, 0, 0, 42.0, &coefficients);
    assert_eq!(setup.fcf.total_active_cuts(), 1);
}

/// `inflow_method()` reflects the config setting.
#[test]
fn inflow_method_reflects_config() {
    use crate::InflowNonNegativityMethod;

    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert!(
        !matches!(
            setup.methodology.inflow_method,
            InflowNonNegativityMethod::None
        ),
        "expected penalty or truncation method"
    );
}

/// `cut_selection()` returns `None` when disabled in config (default).
#[test]
fn cut_selection_none_when_disabled() {
    let system = minimal_system(2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert!(
        setup.cut_management.cut_selection.is_none(),
        "cut_selection should be None when disabled"
    );
}

#[test]
fn stage_ctx_fields_match_study_setup() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let ctx = setup.stage_ctx();

    assert_eq!(
        ctx.templates.len(),
        setup.stage_data.stage_templates.templates.len(),
        "templates length mismatch"
    );
    assert_eq!(
        ctx.base_rows.len(),
        setup.stage_data.stage_templates.base_rows.len(),
        "base_rows length mismatch"
    );
    assert_eq!(
        ctx.noise_scale.len(),
        setup.stage_data.stage_templates.noise_scale.len(),
        "noise_scale length mismatch"
    );
    assert_eq!(
        ctx.n_hydros,
        setup.stage_data.entity_counts.hydro_ids.len(),
        "n_hydros mismatch"
    );
    assert_eq!(
        ctx.block_counts_per_stage.len(),
        setup.stage_data.block_counts_per_stage.len(),
        "block_counts_per_stage length mismatch"
    );
}

#[test]
fn training_ctx_fields_match_study_setup() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let ctx = setup.training_ctx();

    assert_eq!(
        ctx.horizon.num_stages(),
        setup.methodology.horizon.num_stages(),
        "horizon num_stages mismatch"
    );
    assert_eq!(
        ctx.state.n_state, setup.stage_data.state.n_state,
        "indexer n_state mismatch"
    );
    assert_eq!(
        ctx.initial_state.len(),
        setup.initial_state.len(),
        "initial_state length mismatch"
    );
}

#[test]
fn simulation_ctx_propagates_dynamic_dcs_from_setup() {
    let n_stages = 3;
    let system = minimal_system(n_stages);
    let mut config = minimal_config(2, 10);
    // A Dynamic strategy here is what makes `simulation_ctx()` populate `dcs`.
    config.training.cut_selection = RowSelectionConfig {
        selection: Some(cobre_io::config::SelectionMethod::Dynamic {
            start_iteration: 2,
            seed_window: 5,
            candidate_recency: None,
            max_added_per_round: 10,
            violation_tolerance: 1e-10,
        }),
        ..RowSelectionConfig::default()
    };
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let ctx = setup.simulation_ctx();

    // The dynamic method with default fields maps to the spec defaults
    // (k1 = None, k2 = 5, nadic = 10, epsilon_viol = 1e-10, start_iteration = 2).
    let expected = crate::dcs::DcsParams {
        k1: None,
        k2: 5,
        nadic: 10,
        epsilon_viol: 1e-10,
        start_iteration: 2,
        max_inner_iterations: crate::dcs::DcsParams::default().max_inner_iterations,
    };
    assert_eq!(
        ctx.dcs,
        Some(expected),
        "simulation_ctx().dcs must carry the configured dynamic DcsParams, got {:?}",
        ctx.dcs
    );
}

/// Given a 1-hydro, 1-thermal, 1-bus, 2-stage system with an iteration
/// limit of 3, when `train()` is called, then it completes successfully
/// with `result.iterations <= 3`.
#[test]
fn train_completes_within_iteration_limit() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    let result = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    assert!(
        result.result.iterations <= 3,
        "expected iterations <= 3, got {}",
        result.result.iterations
    );
    assert!(
        result.result.iterations >= 1,
        "expected at least 1 iteration, got {}",
        result.result.iterations
    );
}

/// After `train()` completes, at least one cut should be populated in the
/// FCF cut pool for stage 0.
#[test]
fn train_generates_cuts_in_fcf() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    assert!(
        setup.fcf.pools[0].populated_count > 0,
        "expected at least one cut in FCF pool[0] after training"
    );
}

/// `simulation_config()` returns a `SimulationConfig` whose fields match
/// the values extracted from the `Config` at construction time.
#[test]
fn simulation_config_reflects_setup_fields() {
    use cobre_io::config::SimulationConfig as IoSimulationConfig;

    let mut config = minimal_config(1, 5);
    config.simulation = IoSimulationConfig {
        enabled: true,
        num_scenarios: 50,
        io_channel_capacity: 16,
        ..IoSimulationConfig::default()
    };

    let system = minimal_system(2);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let sim_cfg = setup.simulation_config();
    assert_eq!(sim_cfg.n_scenarios, setup.simulation_config.n_scenarios);
    assert_eq!(
        sim_cfg.io_channel_capacity,
        setup.simulation_config.io_channel_capacity
    );
}

/// `create_workspace_pool()` with `n_threads = 2` returns a pool whose
/// `workspaces.len()` equals 2.
#[test]
fn create_workspace_pool_returns_correct_size() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 3);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let comm = LocalBackend;
    let pool = setup
        .create_workspace_pool(&comm, 2, ActiveSolver::new)
        .expect("workspace pool");

    assert_eq!(pool.workspaces.len(), 2);
}

/// `build_training_output()` with a non-empty `TrainingResult` and empty
/// events produces a `TrainingOutput` whose `convergence_records` is
/// non-empty (one record per `result.iterations`).
#[test]
fn build_training_output_non_empty() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let system = minimal_system(2);
    let config = minimal_config(1, 2);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");

    let (event_tx, event_rx) = std::sync::mpsc::channel();
    let result = setup
        .train(
            &mut solver,
            &comm,
            1,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train");

    let events: Vec<cobre_core::TrainingEvent> = event_rx.try_iter().collect();

    let output = setup.build_training_output(&result.result, &events);
    assert!(
        !output.convergence_records.is_empty(),
        "convergence_records must be non-empty after training"
    );
}

/// Given a trained `StudySetup` with `n_scenarios > 0`, calling `simulate()`
/// returns `Ok(costs)` with `costs.len() > 0`.
#[test]
fn simulate_after_train_returns_nonempty_costs() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    let mut config = minimal_config(1, 3);
    config.simulation = cobre_io::config::SimulationConfig {
        enabled: true,
        num_scenarios: 3,
        io_channel_capacity: 8,
        ..cobre_io::config::SimulationConfig::default()
    };

    let system = minimal_system(2);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let mut setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    // Train first so the FCF has cuts.
    let comm = LocalBackend;
    let mut solver = ActiveSolver::new().expect("solver");
    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train");

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("sim pool");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let sim_result = setup
        .simulate(&mut pool.workspaces, &comm, &result_tx, None, None, &[])
        .expect("simulate");

    // Drop the sender so the drain thread terminates.
    drop(result_tx);
    let _results = drain_handle.join().expect("drain thread");

    assert!(
        !sim_result.costs.is_empty(),
        "simulate must return at least one cost entry"
    );
    assert_eq!(
        sim_result.solver_stats.len(),
        sim_result.costs.len(),
        "one solver stats entry per scenario"
    );
}

/// Given a config with no overrides, `StudyParams::from_config` returns the
/// default values for all fields.
#[test]
fn study_params_from_config_defaults() {
    use super::{DEFAULT_FORWARD_PASSES, DEFAULT_SEED, StudyParams};
    use crate::stopping_rule::StoppingMode;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, TrainingConfig,
        TrainingSolverConfig, UpperBoundEvaluationConfig,
    };

    let config = Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: None,
            forward_passes: None,
            stopping_rules: None,
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    };

    let params = StudyParams::from_config(&config).expect("from_config");

    assert_eq!(
        params.seed, DEFAULT_SEED,
        "seed should default to DEFAULT_SEED"
    );
    assert_eq!(
        params.forward_passes, DEFAULT_FORWARD_PASSES,
        "forward_passes should default to DEFAULT_FORWARD_PASSES"
    );
    assert_eq!(
        params.stopping_rule_set.rules.len(),
        1,
        "expected exactly 1 default stopping rule"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[0],
            crate::stopping_rule::StoppingRule::IterationLimit { .. }
        ),
        "default rule should be IterationLimit"
    );
    assert!(
        matches!(params.stopping_rule_set.mode, StoppingMode::Any),
        "default stopping mode should be Any"
    );
    assert_eq!(
        params.n_scenarios, 0,
        "n_scenarios should be 0 when simulation disabled"
    );
    assert!(
        params.cut_selection.is_none(),
        "cut_selection should be None by default"
    );
}

/// Given a config with explicit values for all fields, `StudyParams::from_config`
/// extracts them correctly.
#[test]
fn study_params_from_config_explicit() {
    use super::StudyParams;
    use crate::stopping_rule::{StoppingMode, StoppingRule};
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };

    let config = Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(1234),
            forward_passes: Some(5),
            stopping_rules: Some(vec![
                StoppingRuleConfig::IterationLimit { limit: 50 },
                StoppingRuleConfig::TimeLimit { seconds: 60.0 },
            ]),
            stopping_mode: "all".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig {
            path: "./my_policy".to_string(),
            ..PolicyConfig::default()
        },
        simulation: IoSimulationConfig {
            enabled: true,
            num_scenarios: 200,
            ..IoSimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    };

    let params = StudyParams::from_config(&config).expect("from_config");

    // Seed: i64::unsigned_abs(1234) == 1234
    assert_eq!(params.seed, 1234, "seed mismatch");
    assert_eq!(params.forward_passes, 5, "forward_passes mismatch");
    assert_eq!(
        params.stopping_rule_set.rules.len(),
        2,
        "stopping rule count mismatch"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[0],
            StoppingRule::IterationLimit { limit: 50 }
        ),
        "first rule should be IterationLimit(50)"
    );
    assert!(
        matches!(
            params.stopping_rule_set.rules[1],
            StoppingRule::TimeLimit { seconds } if (seconds - 60.0).abs() < 1e-9
        ),
        "second rule should be TimeLimit(60.0)"
    );
    assert!(
        matches!(params.stopping_rule_set.mode, StoppingMode::All),
        "stopping mode should be All"
    );
    assert_eq!(params.n_scenarios, 200, "n_scenarios mismatch");
    assert_eq!(params.policy_path, "./my_policy", "policy_path mismatch");
}

/// Build a minimal case directory with required structural files present so
/// that `validate_structure` does not fail. The optional estimation and
/// opening tree files are NOT created here; tests add them as needed.
fn write_minimal_case_dir(root: &std::path::Path) {
    use std::fs;

    fs::create_dir_all(root.join("system")).unwrap();
    fs::write(root.join("config.json"), b"{}").unwrap();
    fs::write(root.join("penalties.json"), b"{}").unwrap();
    fs::write(root.join("stages.json"), b"{}").unwrap();
    fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
    fs::write(root.join("system/buses.json"), b"{}").unwrap();
    fs::write(root.join("system/lines.json"), b"{}").unwrap();
    fs::write(root.join("system/hydros.json"), b"{}").unwrap();
    fs::write(root.join("system/thermals.json"), b"{}").unwrap();
}

/// Build a minimal [`cobre_io::Config`] with no estimation or seed overrides.
fn minimal_prepare_config() -> cobre_io::Config {
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, TrainingConfig,
        TrainingSolverConfig, UpperBoundEvaluationConfig,
    };

    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: None,
            forward_passes: None,
            stopping_rules: None,
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Given a case directory with no `inflow_history.parquet` and no
/// `scenarios/noise_openings.parquet`, `prepare_stochastic` returns
/// `estimation_report = None` and a stochastic context with generated provenance.
#[test]
fn prepare_stochastic_no_history_no_tree_returns_none_report_and_generated_provenance() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use cobre_stochastic::provenance::ComponentProvenance;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let system = minimal_system(2);
    let config = minimal_prepare_config();
    let seed = 42_u64;

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, seed, &source)
        .expect("prepare_stochastic should succeed with no optional files");

    assert!(
        result.estimation_report.is_none(),
        "estimation_report must be None when no inflow_history.parquet is present"
    );
    assert_eq!(
        result.stochastic.provenance().opening_tree,
        ComponentProvenance::Generated,
        "opening_tree provenance must be Generated when no user tree is supplied"
    );
}

/// Given a case directory with `inflow_seasonal_stats.parquet` present
/// alongside `inflow_history.parquet`, estimation is skipped and
/// `estimation_report` is `None`.
#[test]
fn prepare_stochastic_with_stats_file_present_skips_estimation() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use std::fs;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    // No `inflow_history.parquet`: the estimation-skip path does not need it, and
    // its absence avoids parse errors (`validate_structure` checks existence,
    // not content).
    fs::create_dir_all(root.join("scenarios")).unwrap();
    fs::write(root.join("scenarios/inflow_seasonal_stats.parquet"), b"").unwrap();
    fs::write(root.join("scenarios/inflow_ar_coefficients.parquet"), b"").unwrap();

    let system = minimal_system(2);
    let config = minimal_prepare_config();
    let seed = 42_u64;

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, seed, &source)
        .expect("prepare_stochastic should succeed when stats file is present");

    assert!(
        result.estimation_report.is_none(),
        "estimation_report must be None when inflow_seasonal_stats.parquet is present"
    );
}

/// Given a case directory with no `scenarios/noise_openings.parquet`,
/// `load_user_opening_tree_inner` returns `None`.
///
/// This is tested indirectly via `prepare_stochastic` by checking that the
/// returned stochastic context does not claim `UserSupplied` provenance.
#[test]
fn prepare_stochastic_no_opening_tree_gives_non_user_supplied_provenance() {
    use super::prepare_stochastic;
    use cobre_core::scenario::ScenarioSource;
    use cobre_stochastic::provenance::ComponentProvenance;
    use tempfile::TempDir;

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let system = minimal_system(2);
    let config = minimal_prepare_config();

    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, 0, &source)
        .expect("prepare_stochastic must succeed with no opening tree file");

    assert_ne!(
        result.stochastic.provenance().opening_tree,
        ComponentProvenance::UserSupplied,
        "opening_tree provenance must not be UserSupplied when file is absent"
    );
}

/// Given a system with `NoiseMethod::HistoricalResiduals` on all stages and
/// sufficient inflow history, when `prepare_stochastic` is called, then it
/// returns `Ok` and the resulting stochastic context has
/// `opening_tree().n_stages()` equal to the number of study stages.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn test_prepare_stochastic_historical_residuals_noise_method() {
    use super::prepare_stochastic;
    use chrono::NaiveDate;
    use cobre_core::{
        scenario::{InflowHistoryRow, ScenarioSource},
        system::SystemBuilder,
    };
    use tempfile::TempDir;

    let n_stages = 2usize;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    // Stages with HistoricalResiduals noise method; branching_factor=2 so
    // each stage selects 2 historical windows as openings.
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 2,
                noise_method: NoiseMethod::HistoricalResiduals,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
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

    // Historical inflow data: 1990 and 1991 cover 12 months each — 2 valid windows.
    let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
        .flat_map(|year| {
            (1u32..=12).map(move |month| InflowHistoryRow {
                hydro_id: EntityId(3),
                date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
            })
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
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
                water_withdrawal_m3s: 0.0,
            },
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
                storage_violation_below_cost: 500.0,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .inflow_history(inflow_history)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("test system: valid");

    let dir = TempDir::new().unwrap();
    let root = dir.path();
    write_minimal_case_dir(root);

    let config = minimal_prepare_config();
    let source = ScenarioSource {
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: None,
        historical_years: None,
    };
    let result = prepare_stochastic(system, root, &config, 42, &source)
        .expect("prepare_stochastic must succeed with HistoricalResiduals noise method");

    assert_eq!(
        result.stochastic.opening_tree().n_stages(),
        n_stages,
        "opening_tree must have n_stages == {n_stages}"
    );
}

/// Given a system with no FPHA and no evaporation data, `default_from_system`
/// returns a result where all hydros use constant productivity and no evaporation.
#[test]
fn default_from_system_gives_constant_and_no_evaporation() {
    use crate::hydro_models::{EvaporationModel, ProductionModelSource, ResolvedProductionModel};

    let system = minimal_system(2);
    let result = PrepareHydroModelsResult::default_from_system(&system);

    assert_eq!(
        result.provenance.production_sources.len(),
        system.hydros().len(),
        "production_sources length must equal n_hydros"
    );
    for (_, source) in &result.provenance.production_sources {
        assert_eq!(
            *source,
            ProductionModelSource::DefaultConstant,
            "all hydros must use DefaultConstant"
        );
    }

    assert_eq!(
        result.provenance.evaporation_sources.len(),
        system.hydros().len(),
        "evaporation_sources length must equal n_hydros"
    );
    assert!(
        !result.evaporation.has_evaporation(),
        "default result must have no evaporation"
    );

    let model = result.production.model(0, 0);
    assert!(
        matches!(model, ResolvedProductionModel::ConstantProductivity { .. }),
        "default production model must be ConstantProductivity"
    );

    let evap = result.evaporation.model(0);
    assert!(
        matches!(evap, EvaporationModel::None),
        "default evaporation model must be None"
    );
}

/// Given a valid `StudySetup`, `hydro_models()` returns the stored result.
#[test]
fn hydro_models_accessor_returns_stored_result() {
    use crate::hydro_models::ProductionModelSource;

    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);

    let setup = StudySetup::new(&system, &config, stochastic, hydro_result).expect("setup");

    let models = &setup.hydro_models;
    assert_eq!(
        models.provenance.production_sources.len(),
        system.hydros().len(),
        "hydro_models() must return the stored result (provenance length mismatch)"
    );
    for (_, source) in &models.provenance.production_sources {
        assert_eq!(
            *source,
            ProductionModelSource::DefaultConstant,
            "stored result must preserve DefaultConstant provenance"
        );
    }
}

/// Given a valid `StudySetup`, `energy_conversion()` returns a set with
/// the correct dimensions and a non-zero accumulated productivity where
/// expected (the system hydro has `ρ_eq=2.5`, and no downstream, so
/// `ρ_acum=2.5` at every stage).
#[test]
fn energy_conversion_accessor_returns_built_set() {
    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    // Build a PrepareHydroModelsResult with productivity=2.5 for the single hydro.
    // `default_from_system` uses 0.0 as a placeholder; here we supply the
    // specific value that the assertion checks against.
    let n_study_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let hydro_models_result = {
        let mut result = PrepareHydroModelsResult::default_from_system(&system);
        let pm = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::ConstantProductivity {
                    productivity: 2.5
                };
                n_study_stages
            ]],
            1,
            n_study_stages,
        );
        result.production = pm;
        result
    };

    let setup = StudySetup::new(&system, &config, stochastic, hydro_models_result).expect("setup");

    let ec = setup.energy_conversion();
    assert_eq!(ec.n_hydros(), system.hydros().len());
    // The minimal system has 2 study stages and 1 hydro (ConstantProductivity,
    // productivity=2.5, no downstream). ρ_acum must equal ρ_eq = 2.5.
    for s in 0..ec.n_stages() {
        assert!(
            (ec.accumulated_productivity(0, s) - 2.5).abs() < f64::EPSILON,
            "stage {s}: expected ρ_acum=2.5, got {}",
            ec.accumulated_productivity(0, s)
        );
    }
}

/// Given a system whose single hydro is FPHA but lacks VHA geometry and
/// `specific_productivity_mw_per_m3s_per_m`, `StudySetup::new` must
/// propagate the energy-conversion gate failure as an error whose chain
/// contains `EnergyConversionError::FphaMissingEquivalentProductivity`.
#[test]
fn study_setup_propagates_fpha_missing_equivalent_productivity() {
    let system = minimal_fpha_misconfigured_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let err = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect_err("setup must reject misconfigured FPHA hydro");

    let msg = err.to_string();
    assert!(
        msg.contains("cannot derive ρ_eq"),
        "error must come from FphaMissingEquivalentProductivity Display; got: {msg}"
    );
    assert!(
        msg.contains("H_FPHA_BAD"),
        "error must mention the offending hydro by name; got: {msg}"
    );
}

/// Build the role-(a) `StateLayout` for lag tests: N hydros, L lags.
fn layout_for_lag_test(hydro_count: usize, max_par_order: usize) -> StateLayout {
    crate::test_support::state_layout(hydro_count, max_par_order)
}

/// Build the role-(a) `StateLayout` matching [`counts_with_anticipated`]
/// (1 hydro, 0 lags, `n_anticipated` plants with the given per-plant K).
fn layout_with_anticipated(n_anticipated: usize, k_values: &[usize]) -> StateLayout {
    let k_max = k_values.iter().copied().max().unwrap_or(0);
    crate::test_support::state_layout_full(1, 0, n_anticipated, k_max, k_values.to_vec())
}

/// Build a 2-hydro system (IDs 1 and 2) with `n_stages` study stages and
/// PAR order 2 AR coefficients on all stages, with `inflow_lags: true`.
///
/// Provides `past_inflows` in `initial_conditions` with the given values
/// for both hydros.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn minimal_system_2_hydros_with_past_inflows(
    n_stages: usize,
    h1_past: Vec<f64>,
    h2_past: Vec<f64>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str| Hydro {
        id: EntityId(id),
        name: name.to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    let n_st = n_stages.max(1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                ((i % 12 + 1) % 12 + 1) as u32,
                1,
            )
            .unwrap(),
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .flat_map(|i| {
            [1_i32, 2].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
                annual: None,
            })
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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
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
            n_hydros: 2,
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

    let past_inflows = vec![
        cobre_core::HydroPastInflows {
            hydro_id: EntityId(1),
            values_m3s: h1_past,
            season_ids: None,
        },
        cobre_core::HydroPastInflows {
            hydro_id: EntityId(2),
            values_m3s: h2_past,
            season_ids: None,
        },
    ];

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![])
        .hydros(vec![make_hydro(1, "H1"), make_hydro(2, "H2")])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows,
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("minimal_system_2_hydros_with_past_inflows: valid")
}

/// Given 2 hydros (IDs 1, 2), `max_par_order`=2, and `past_inflows` set,
/// `build_initial_state` populates lag slots correctly.
///
/// Hydro idx 0 (id=1): lag 0 = 600.0, lag 1 = 500.0
/// Hydro idx 1 (id=2): lag 0 = 200.0, lag 1 = 100.0
#[test]
fn build_initial_state_populates_lags_from_past_inflows() {
    use super::build_initial_state;

    let system =
        minimal_system_2_hydros_with_past_inflows(1, vec![600.0, 500.0], vec![200.0, 100.0]);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    // State layout: storage(0..2), lags(2..6) in lag-major order.
    // Lag-major: slot = s + lag * N + h, where N = 2.
    // lag0_h0 = 600.0 at s+0, lag0_h1 = 200.0 at s+1,
    // lag1_h0 = 500.0 at s+2, lag1_h1 = 100.0 at s+3.
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - 600.0).abs() < 1e-10,
        "lag0 hydro 0: expected 600.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1: expected 200.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0: expected 500.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1: expected 100.0, got {}",
        state[s + 3]
    );
    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
}

/// Given no `past_inflows` entries, all lag slots remain 0.0.
#[test]
fn build_initial_state_empty_past_inflows_leaves_zero_lags() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 3);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    let s = layout.inflow_lags.start;
    for l in 0..3 {
        assert!(
            state[s + l].abs() < 1e-10,
            "lag slot {l} should be 0.0 when past_inflows is empty, got {}",
            state[s + l]
        );
    }
}

/// Given `past_inflows` only for a hydro not in the system, lag slots
/// for the system's hydros remain 0.0.
#[test]
fn build_initial_state_unknown_hydro_in_past_inflows_stays_zero() {
    use super::build_initial_state;

    // minimal_system cannot override IC, so its past_inflows is empty and both
    // lag slots stay 0.0 — the same outcome as past_inflows for an unknown hydro.
    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 2);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    let s = layout.inflow_lags.start;
    assert!(
        state[s].abs() < 1e-10,
        "lag 0 should be 0.0 when past_inflows is absent, got {}",
        state[s]
    );
    assert!(
        state[s + 1].abs() < 1e-10,
        "lag 1 should be 0.0 when past_inflows is absent, got {}",
        state[s + 1]
    );
}

/// Build a 2-hydro system (IDs 1, 2) where hydro 2 carries a `FillingConfig`
/// (a filling hydro) and hydro 1 is operating, with caller-supplied
/// `initial_conditions`.
///
/// Mirrors [`minimal_system_2_hydros_with_past_inflows`] but exposes the full
/// `InitialConditions` so a test can populate `storage` and `filling_storage`
/// independently. `start_stage_id` sets hydro 2's filling start stage (0 =
/// mid-filling seed; >0 = empty pit).
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn filling_system_2_hydros(
    n_stages: usize,
    start_stage_id: i32,
    initial_conditions: cobre_core::InitialConditions,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let make_hydro = |id: i32, name: &str, filling: Option<cobre_core::FillingConfig>| Hydro {
        id: EntityId(id),
        name: name.to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
        // A filling hydro requires `entry_stage_id` (the operating-handoff
        // stage) to be `Some`; the system builder rejects `filling` without
        // it. Operating hydros leave it `None`.
        entry_stage_id: filling.as_ref().map(|f| f.start_stage_id + 1),
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
        filling,
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

    let n_st = n_stages.max(1);
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(
                if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                ((i % 12 + 1) % 12 + 1) as u32,
                1,
            )
            .unwrap(),
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .flat_map(|i| {
            [1_i32, 2].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.5, 0.3],
                residual_std_ratio: 0.8,
                annual: None,
            })
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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
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
            n_hydros: 2,
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

    let filling = Some(cobre_core::FillingConfig {
        start_stage_id,
        filling_min_rate_m3s: 50.0,
    });

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![])
        .hydros(vec![
            make_hydro(1, "H1", None),
            make_hydro(2, "H2_FILL", filling),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("filling_system_2_hydros: valid")
}

/// Given a `filling_storage` entry with a non-zero mid-fill seed for the
/// filling hydro (id=2, system index 1), `build_initial_state` writes the
/// seed at the hydro's storage coordinate.
#[test]
fn build_initial_state_seeds_filling_storage() {
    use super::build_initial_state;

    let seed = 120.0_f64;
    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: seed,
        }],
        past_inflows: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    // Hydro id=2 is at system index 1; its storage coordinate is state[1].
    assert!(
        (state[1] - seed).abs() < 1e-10,
        "filling hydro storage coordinate should carry the seed {seed}, got {}",
        state[1]
    );
    // The operating hydro (id=1) had no storage IC, so its coordinate is 0.
    assert!(
        state[0].abs() < 1e-10,
        "non-seeded operating hydro storage should be 0.0, got {}",
        state[0]
    );
}

/// Given a filling hydro whose `start_stage_id > 0` so `filling_storage`
/// holds `value_hm3 == 0.0` (empty pit), `build_initial_state`
/// leaves the coordinate at 0.0 and fires no debug-assert.
#[test]
fn build_initial_state_filling_empty_pit_is_zero() {
    use super::build_initial_state;

    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: 0.0,
        }],
        past_inflows: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 1, ic);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    assert!(
        state[1].abs() < 1e-10,
        "empty-pit filling hydro storage should be 0.0, got {}",
        state[1]
    );
}

/// Given a `filling_storage` entry whose `hydro_id` matches no hydro in the
/// system, `build_initial_state` returns without panic and every state slot
/// is unchanged from the no-filling baseline.
#[test]
fn build_initial_state_unknown_filling_hydro_skipped() {
    use super::build_initial_state;

    let layout = layout_for_lag_test(2, 2);
    let study_dims = crate::test_support::study_dims();

    let baseline_ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let baseline_system = filling_system_2_hydros(1, 0, baseline_ic);
    let baseline = build_initial_state(&baseline_system, &study_dims, &layout);

    let ic = cobre_core::InitialConditions {
        storage: vec![],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(99),
            value_hm3: 150.0,
        }],
        past_inflows: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let state = build_initial_state(&system, &study_dims, &layout);

    assert_eq!(
        state, baseline,
        "an unknown filling hydro_id must be silently skipped, leaving the \
             no-filling baseline state unchanged"
    );
}

/// Given both `storage` (operating hydro id=1) and `filling_storage`
/// (filling hydro id=2) populated alongside `past_inflows`,
/// `build_initial_state` seeds both storage coordinates and the AR-lag slots
/// identically to the operating-only path (filling hydros share the lag
/// path).
#[test]
fn build_initial_state_mixed_operating_and_filling_seeds() {
    use super::build_initial_state;

    let operating_seed = 175.0_f64;
    let filling_seed = 90.0_f64;
    let past_inflows = vec![
        cobre_core::HydroPastInflows {
            hydro_id: EntityId(1),
            values_m3s: vec![600.0, 500.0],
            season_ids: None,
        },
        cobre_core::HydroPastInflows {
            hydro_id: EntityId(2),
            values_m3s: vec![200.0, 100.0],
            season_ids: None,
        },
    ];
    let ic = cobre_core::InitialConditions {
        storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(1),
            value_hm3: operating_seed,
        }],
        filling_storage: vec![cobre_core::HydroStorage {
            hydro_id: EntityId(2),
            value_hm3: filling_seed,
        }],
        past_inflows,
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let system = filling_system_2_hydros(1, 0, ic);
    let layout = layout_for_lag_test(2, 2);

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    // Both storage coordinates carry their respective seeds.
    assert!(
        (state[0] - operating_seed).abs() < 1e-10,
        "operating hydro storage should be {operating_seed}, got {}",
        state[0]
    );
    assert!(
        (state[1] - filling_seed).abs() < 1e-10,
        "filling hydro storage should be {filling_seed}, got {}",
        state[1]
    );

    // AR-lag slots are seeded by the shared `past_inflows` path, identical to
    // the operating-only case (lag-major: slot = s + lag * N + h, N = 2).
    let s = layout.inflow_lags.start;
    assert!(
        (state[s] - 600.0).abs() < 1e-10,
        "lag0 hydro 0: expected 600.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1: expected 200.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0: expected 500.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1: expected 100.0, got {}",
        state[s + 3]
    );
}

/// Integration test: `StudySetup::new` with `past_inflows` in the system's
/// initial conditions produces `initial_state()` with non-zero lag values.
#[test]
fn study_setup_initial_state_has_nonzero_lags_from_past_inflows() {
    let system =
        minimal_system_2_hydros_with_past_inflows(3, vec![600.0, 500.0], vec![200.0, 100.0]);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup with past_inflows");

    let state = &setup.initial_state;

    // With 2 hydros (N=2) and max_par_order=2 (L=2), lag slots start at N=2.
    // Lag-major layout: slot = lag_start + lag * N + h.
    // lag0_h0 = 600.0 at [2], lag0_h1 = 200.0 at [3],
    // lag1_h0 = 500.0 at [4], lag1_h1 = 100.0 at [5].
    let n_hydros = 2;
    let lag_start = n_hydros;
    assert!(
        (state[lag_start] - 600.0).abs() < 1e-10,
        "lag0 hydro 0 should be 600.0 via StudySetup, got {}",
        state[lag_start]
    );
    assert!(
        (state[lag_start + 1] - 200.0).abs() < 1e-10,
        "lag0 hydro 1 should be 200.0 via StudySetup, got {}",
        state[lag_start + 1]
    );
    assert!(
        (state[lag_start + 2] - 500.0).abs() < 1e-10,
        "lag1 hydro 0 should be 500.0 via StudySetup, got {}",
        state[lag_start + 2]
    );
    assert!(
        (state[lag_start + 3] - 100.0).abs() < 1e-10,
        "lag1 hydro 1 should be 100.0 via StudySetup, got {}",
        state[lag_start + 3]
    );
}

/// Given `max_par_order`=0, no lag slots exist; state is storage-only.
#[test]
fn build_initial_state_no_lags_state_is_storage_only() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 0);

    // n_state = N*(1+L) = 1*(1+0) = 1
    assert_eq!(layout.n_state, 1);
    assert!(
        layout.inflow_lags.is_empty(),
        "inflow_lags range should be empty for L=0"
    );

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    assert_eq!(state.len(), 1, "state length must equal n_state=1");
}

// -----------------------------------------------------------------------
// build_initial_state — anticipated_state seed
// -----------------------------------------------------------------------

/// Build the `GeometryDims` for N=1 hydro, L=0 lags, and the given
/// anticipated-thermal metadata.
///
/// This gives a non-zero `anticipated_state` block in the state vector. The
/// anticipated `build_initial_state` tests derive their non-state
/// `StudyDimensions` from these dims (via `study_dims_for`), so the geometry
/// and the study shape stay aligned from one source.
fn counts_with_anticipated(
    n_anticipated: usize,
    k_values: &[usize],
    thermal_indices: &[usize],
) -> crate::test_support::GeometryDims {
    let k_max = k_values.iter().copied().max().unwrap_or(0);
    crate::test_support::GeometryDims {
        hydro_count: 1,
        n_thermals: n_anticipated, // at least cover the anticipated plants
        n_buses: 1,
        n_blks: 1,
        n_anticipated,
        k_max,
        anticipated_thermal_indices: thermal_indices.to_vec(),
        ..Default::default()
    }
}

/// Build a 1-bus / 1-hydro system whose `thermals` list contains N
/// anticipated thermals with the given `lead_stages` values.  Thermal IDs
/// are assigned as `EntityId(10 + i as i32)` so they are distinct from the
/// bus (ID 1) and the hydro (ID 3).  `past_anticipated_commitments` is set
/// to `past_commits` (must be pre-sorted by `thermal_id`).
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn system_with_anticipated_thermals(
    k_values: &[u32],
    past_commits: Vec<cobre_core::AnticipatedCommitmentHistory>,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    // Build N anticipated thermals. IDs are 10, 11, 12, … so they are
    // always above the hydro ID (3) and can be easily identified.
    let thermals: Vec<Thermal> = k_values
        .iter()
        .enumerate()
        .map(|(i, &k)| Thermal {
            id: EntityId(10 + i as i32),
            name: format!("AT{i}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(k)),
            entry_stage_id: None,
            exit_stage_id: None,
        })
        .collect();

    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    let n_stages = 2_usize;
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
            hydro_id: EntityId(3),
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

    let k_max_bounds = k_values.iter().copied().max().unwrap_or(0) as usize;
    let n_thermals = k_values.len();

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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max_bounds,
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
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
        .thermals(thermals)
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(cobre_core::InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: past_commits,
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("system_with_anticipated_thermals: valid")
}

/// AC-1: A system with `n_anticipated == 0` produces an unchanged state
/// vector (length `n_state`, `anticipated_slots_out` block is empty).
///
/// Regression guard: confirms zero-anticipated path is unaffected.
#[test]
fn build_initial_state_no_anticipated_state_unchanged() {
    use super::build_initial_state;

    let system = minimal_system(2);
    let layout = layout_for_lag_test(1, 0);

    // n_anticipated == 0; anticipated_slots_out range is 0..0.
    assert_eq!(layout.n_anticipated, 0);
    assert!(layout.anticipated_slots_out.is_empty());

    let state = build_initial_state(&system, &crate::test_support::study_dims(), &layout);

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    // All slots are 0.0 — storage IC is empty in minimal_system.
    assert!(
        state.iter().all(|&v| v == 0.0),
        "all state slots must be 0.0 when no anticipated thermals and no ICs set"
    );
}

/// AC-2: `n_anticipated == 1`, `k_max == 2`, `K_0 == 2` and
/// `past_anticipated_commitments` has one entry with `values_mw: [50.0, 75.0]`.
///
/// Expected slot-major layout (`n_ant=1`):
///   slot 0: `ant_start + 0*1 + 0 = ant_start`   → 50.0
///   slot 1: `ant_start + 1*1 + 0 = ant_start+1`  → 75.0
#[test]
fn build_initial_state_single_anticipated_thermal_k2() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    // Thermal ID 10 is the first (and only) anticipated plant.
    // The system thermals() sorts by ID, so global_idx == 0 for ID 10.
    let past_commits = vec![AnticipatedCommitmentHistory {
        thermal_id: EntityId(10),
        values_mw: vec![50.0, 75.0],
    }];
    let system = system_with_anticipated_thermals(&[2], past_commits);

    // indexer: 1 hydro, 0 lags, 1 anticipated thermal (global idx 0), k_max=2.
    let layout = layout_with_anticipated(1, &[2]);

    let state = build_initial_state(
        &system,
        &crate::test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.anticipated_slots_out.start;
    assert!(
        (state[ant_start] - 50.0).abs() < 1e-10,
        "slot 0 expected 50.0, got {}",
        state[ant_start]
    );
    assert!(
        (state[ant_start + 1] - 75.0).abs() < 1e-10,
        "slot 1 expected 75.0, got {}",
        state[ant_start + 1]
    );
}

/// AC-3: `n_anticipated == 2`, `k_max == 3`, `K_0 == 2`, `K_1 == 3`.
///
/// Slot-major layout with `n_ant=2`:
///
/// - (slot 0, plant 0): `ant_start + 0*2+0` → 10.0
/// - (slot 0, plant 1): `ant_start + 0*2+1` → 100.0
/// - (slot 1, plant 0): `ant_start + 1*2+0` → 20.0
/// - (slot 1, plant 1): `ant_start + 1*2+1` → 200.0
/// - (slot 2, plant 0): `ant_start + 2*2+0` → 0.0  (padding: `K_0=2 < k_max=3`)
/// - (slot 2, plant 1): `ant_start + 2*2+1` → 300.0
#[test]
fn build_initial_state_two_anticipated_thermals_mixed_k() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    // Thermal IDs 10 (K=2) and 11 (K=3); sorted ascending so global order
    // in system.thermals() is idx 0 → ID 10, idx 1 → ID 11.
    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            values_mw: vec![10.0, 20.0],
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            values_mw: vec![100.0, 200.0, 300.0],
        },
    ];
    let system = system_with_anticipated_thermals(&[2, 3], past_commits);

    // indexer: 1 hydro, 0 lags, 2 anticipated thermals
    //   anticipated_thermal_indices = [0, 1]  (global idxs in thermals())
    //   anticipated_lead_stages     = [2, 3]
    //   k_max                       = 3
    let layout = layout_with_anticipated(2, &[2, 3]);

    let state = build_initial_state(
        &system,
        &crate::test_support::study_dims_for(&counts_with_anticipated(2, &[2, 3], &[0, 1])),
        &layout,
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    // n_ant = 2, k_max = 3.  Slot-major offsets from ant_start:
    //   (slot, plant) → offset = slot * n_ant + plant
    //   (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3, (2,0)→4, (2,1)→5
    let s = layout.anticipated_slots_out.start;

    assert!(
        (state[s] - 10.0).abs() < 1e-10,
        "slot 0 plant 0: expected 10.0, got {}",
        state[s]
    );
    assert!(
        (state[s + 1] - 100.0).abs() < 1e-10,
        "slot 0 plant 1: expected 100.0, got {}",
        state[s + 1]
    );
    assert!(
        (state[s + 2] - 20.0).abs() < 1e-10,
        "slot 1 plant 0: expected 20.0, got {}",
        state[s + 2]
    );
    assert!(
        (state[s + 3] - 200.0).abs() < 1e-10,
        "slot 1 plant 1: expected 200.0, got {}",
        state[s + 3]
    );
    assert!(
        state[s + 4].abs() < 1e-10,
        "slot 2 plant 0 (K_0=2 padding): expected 0.0, got {}",
        state[s + 4]
    );
    assert!(
        (state[s + 5] - 300.0).abs() < 1e-10,
        "slot 2 plant 1: expected 300.0, got {}",
        state[s + 5]
    );
}

/// AC-4: `n_anticipated == 1`, `k_max == 2`, but `past_anticipated_commitments`
/// is empty.  All `anticipated_state` slots must remain 0.0 (no panic).
#[test]
fn build_initial_state_empty_past_commitments_leaves_zeros() {
    use super::build_initial_state;

    let system = system_with_anticipated_thermals(&[2], vec![]);

    let layout = layout_with_anticipated(1, &[2]);

    let state = build_initial_state(
        &system,
        &crate::test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.anticipated_slots_out.start;
    let ant_end = layout.anticipated_slots_out.end;
    for (i, &v) in state[ant_start..ant_end].iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "anticipated_state slot {i} expected 0.0, got {v}"
        );
    }
}

/// AC-5: `past_anticipated_commitments` contains a `thermal_id` that does
/// not match any anticipated thermal.  The function silently ignores it and
/// all `anticipated_state` slots remain 0.0 (no panic).
#[test]
fn build_initial_state_unknown_thermal_id_silently_skipped() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    // System has one anticipated thermal (ID 10).
    // past_anticipated_commitments references ID 99999 — not in the system.
    let past_commits = vec![AnticipatedCommitmentHistory {
        thermal_id: EntityId(99999),
        values_mw: vec![42.0, 43.0],
    }];
    let system = system_with_anticipated_thermals(&[2], past_commits);

    let layout = layout_with_anticipated(1, &[2]);

    let state = build_initial_state(
        &system,
        &crate::test_support::study_dims_for(&counts_with_anticipated(1, &[2], &[0])),
        &layout,
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let ant_start = layout.anticipated_slots_out.start;
    let ant_end = layout.anticipated_slots_out.end;
    for (i, &v) in state[ant_start..ant_end].iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "anticipated_state slot {i} expected 0.0 for unknown ID, got {v}"
        );
    }
}

/// AC-6 (happy path with padding slot): two anticipated plants with
/// `K_0 = 1` and `K_1 = 2`, so `k_max = 2`. The plant-0 ring-buffer column
/// has one valid slot (slot 0) and one padding slot (slot 1).
///
/// `past_anticipated_commitments` carries `[100.0]` for plant 0 and
/// `[50.0, 75.0]` for plant 1, each of length `K_i` exactly (the contract
/// the cobre-io validator enforces in production).
///
/// Expected layout (`n_ant = 2`, slot-major):
///   - `ant_start + 0*2 + 0` (slot 0, plant 0) -> 100.0  (seed)
///   - `ant_start + 0*2 + 1` (slot 0, plant 1) ->  50.0  (seed)
///   - `ant_start + 1*2 + 0` (slot 1, plant 0) ->   0.0  (padding; `K_0=1` < `k_max=2`)
///   - `ant_start + 1*2 + 1` (slot 1, plant 1) ->  75.0  (seed)
///
/// The padding-slot `debug_assert!` must not fire because the `.min(k_i)`
/// clamp prevents writing past slot `K_0=1` on plant 0.
#[test]
fn build_initial_state_anticipated_seed_padding_slot_stays_zero() {
    use super::build_initial_state;
    use cobre_core::AnticipatedCommitmentHistory;

    let past_commits = vec![
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(10),
            values_mw: vec![100.0],
        },
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(11),
            values_mw: vec![50.0, 75.0],
        },
    ];
    let system = system_with_anticipated_thermals(&[1, 2], past_commits);
    // n_anticipated=2, k_values=[1, 2] -> k_max=2.
    let layout = layout_with_anticipated(2, &[1, 2]);

    let state = build_initial_state(
        &system,
        &crate::test_support::study_dims_for(&counts_with_anticipated(2, &[1, 2], &[0, 1])),
        &layout,
    );

    assert_eq!(
        state.len(),
        layout.n_state,
        "state length must equal n_state"
    );
    let s = layout.anticipated_slots_out.start;
    let n_ant = layout.n_anticipated;
    assert_eq!(n_ant, 2);
    assert_eq!(layout.k_max, 2);

    // slot 0, plant 0 -> 100.0
    assert!(
        (state[s] - 100.0).abs() < 1e-10,
        "slot 0 plant 0 expected 100.0, got {}",
        state[s]
    );
    // slot 0, plant 1 -> 50.0
    assert!(
        (state[s + 1] - 50.0).abs() < 1e-10,
        "slot 0 plant 1 expected 50.0, got {}",
        state[s + 1]
    );
    // slot 1, plant 0 -> 0.0 (padding for K_0=1 < k_max=2). This is the
    // invariant the new debug_assert! protects.
    assert!(
        state[s + 2].abs() < 1e-10,
        "padding slot 1 plant 0 expected 0.0, got {}",
        state[s + 2]
    );
    // slot 1, plant 1 -> 75.0
    assert!(
        (state[s + 3] - 75.0).abs() < 1e-10,
        "slot 1 plant 1 expected 75.0, got {}",
        state[s + 3]
    );
}

/// Given a `System` with `inflow_scheme = InSample`, when `StudySetup::new()`
/// is called, then `historical_library()` returns `None`.
#[test]
fn historical_library_none_for_insample() {
    let system = minimal_system(2);
    let config = minimal_config(1, 5);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert!(
        setup.scenario_libraries.training.historical.is_none(),
        "historical_library must be None for InSample scheme"
    );
    assert!(
        setup.scenario_libraries.training.external_inflow.is_none(),
        "external_inflow_library must be None for InSample scheme"
    );
    assert!(
        setup.scenario_libraries.training.external_load.is_none(),
        "external_load_library must be None for InSample load scheme"
    );
    assert!(
        setup.scenario_libraries.training.external_ncs.is_none(),
        "external_ncs_library must be None for InSample ncs scheme"
    );
}

/// Build a system that has `inflow_scheme = Historical` and the inflow
/// history needed to discover at least one window.
///
/// The system has 1 hydro, 1 bus, 1 thermal, 2 monthly stages (`season_id`
/// `Some(0)` and `Some(1)`), and historical data covering years 1990-1991.
/// With `max_par_order = 0` (no AR coefficients), a window is valid if
/// we have observations for both study months. Year 1990 covers months 0-1
/// so season 0 and 1 are available under year 1990.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]
fn system_with_historical_inflow(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::{scenario::InflowHistoryRow, system::SystemBuilder};

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
            storage_violation_below_cost: 500.0,
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

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    // Monthly stages: season_id = month index (0-based).
    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
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

    // Historical inflow data: 1990 and 1991 cover 12 months each.
    // With n_stages <= 2 and max_par_order = 0, year 1990 and 1991 are
    // both valid windows (study months are in Jan-Feb = seasons 0-1).
    let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
        .flat_map(|year| {
            (1u32..=12).map(move |month| InflowHistoryRow {
                hydro_id: EntityId(3),
                date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
            })
        })
        .collect();

    let n_st = n_stages.max(1);

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .inflow_history(inflow_history)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_historical_inflow: valid")
}

/// Given a `System` with `inflow_scheme = Historical` and valid inflow history,
/// when `StudySetup::new()` is called, then `historical_library()` returns
/// `Some` and `n_windows() > 0`.
#[test]
fn historical_library_built_when_scheme_is_historical() {
    let system = system_with_historical_inflow(2);
    let config = minimal_config_with_schemes(1, 5, Some("historical"), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::Historical),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let lib = setup
        .scenario_libraries
        .training
        .historical
        .as_ref()
        .expect("expected Some(HistoricalScenarioLibrary) for Historical scheme");
    assert!(
        lib.n_windows() > 0,
        "expected at least one historical window, got 0"
    );
    assert_eq!(
        lib.n_stages(),
        2,
        "expected n_stages == 2 matching the system's study stages"
    );
    assert_eq!(lib.n_hydros(), 1, "expected n_hydros == 1");
}

/// Given a `System` with `inflow_scheme = External` and valid external
/// inflow rows, when `StudySetup::new()` is called, then
/// `external_inflow_library()` returns `Some` and `n_entities() > 0`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
fn external_inflow_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::ExternalScenarioRow;
    use cobre_core::{scenario::InflowModel as CoreInflowModel, system::SystemBuilder};

    // 3 scenarios × 1 hydro (ID 3, from minimal_system) × 2 stages.
    let hydro_id = EntityId(3);
    let mut external_rows: Vec<ExternalScenarioRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_rows.push(ExternalScenarioRow {
                stage_id,
                scenario_id,
                hydro_id,
                value_m3s: 80.0 + scenario_id as f64 * 5.0,
            });
        }
    }

    // minimal_system has no seam to inject external rows, so rebuild the system
    // directly to carry them.
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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
    let stages: Vec<Stage> = (0..2usize)
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
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
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
            n_stages: 2,
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
                water_withdrawal_m3s: 0.0,
            },
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_scenarios(external_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external inflow: valid");

    let config = minimal_config_with_schemes(1, 5, Some("external"), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::External),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let lib = setup
        .scenario_libraries
        .training
        .external_inflow
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External inflow scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external inflow library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "inflow");
}

/// Given a `System` with `load_scheme = External` and valid external load
/// rows, when `StudySetup::new()` is called, then
/// `external_load_library()` returns `Some` and `n_entities() > 0`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
fn external_load_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::ExternalLoadRow;
    use cobre_core::{scenario::InflowModel as CoreInflowModel, system::SystemBuilder};

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    let stages: Vec<Stage> = (0..2usize)
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
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 10.0,
        })
        .collect();

    // External load rows: 3 scenarios × 1 bus × 2 stages.
    let mut external_load_rows: Vec<ExternalLoadRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_load_rows.push(ExternalLoadRow {
                stage_id,
                scenario_id,
                bus_id: EntityId(1),
                value_mw: 90.0 + scenario_id as f64 * 10.0,
            });
        }
    }

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
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
                water_withdrawal_m3s: 0.0,
            },
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .external_load_scenarios(external_load_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external load: valid");

    let config = minimal_config_with_schemes(1, 5, None, Some("external"), None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::External),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let lib = setup
        .scenario_libraries
        .training
        .external_load
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External load scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external load library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "load");
}

/// Given a `System` with `ncs_scheme = External` and valid external NCS
/// rows, when `StudySetup::new()` is called, then
/// `external_ncs_library()` returns `Some` and `n_entities() > 0`.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
fn external_ncs_library_built_when_scheme_is_external() {
    use chrono::NaiveDate;
    use cobre_core::scenario::InflowModel as CoreInflowModel;
    use cobre_core::{
        NonControllableSource,
        scenario::{ExternalNcsRow, NcsModel},
        system::SystemBuilder,
    };

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    let ncs_id = EntityId(4);
    let ncs_source = NonControllableSource {
        id: ncs_id,
        name: "Wind1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 0.01,
    };

    let stages: Vec<Stage> = (0..2usize)
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
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<CoreInflowModel> = (0..2usize)
        .map(|i| CoreInflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let ncs_models: Vec<NcsModel> = (0..2usize)
        .map(|i| NcsModel {
            ncs_id,
            stage_id: i as i32,
            mean: 0.8,
            std: 0.1,
        })
        .collect();

    // External NCS rows: 3 scenarios × 1 NCS × 2 stages.
    let mut external_ncs_rows: Vec<ExternalNcsRow> = Vec::new();
    for stage_id in 0i32..2 {
        for scenario_id in 0i32..3 {
            external_ncs_rows.push(ExternalNcsRow {
                stage_id,
                scenario_id,
                ncs_id,
                value: 0.7 + scenario_id as f64 * 0.1,
            });
        }
    }

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
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
                water_withdrawal_m3s: 0.0,
            },
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 1,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .non_controllable_sources(vec![ncs_source])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(ncs_models)
        .external_ncs_scenarios(external_ncs_rows)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system with external NCS: valid");

    let config = minimal_config_with_schemes(1, 5, None, None, Some("external"));
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::External),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let lib = setup
        .scenario_libraries
        .training
        .external_ncs
        .as_ref()
        .expect("expected Some(ExternalScenarioLibrary) for External NCS scheme");
    assert!(
        lib.n_entities() > 0,
        "expected n_entities > 0 in external NCS library"
    );
    assert_eq!(lib.n_stages(), 2);
    assert_eq!(lib.n_scenarios(), 3);
    assert_eq!(lib.entity_class(), "ncs");
}

/// Given a `System` with `inflow_scheme = Historical` but a user pool
/// that references a year with no data, when `StudySetup::new()` is
/// called, then it returns `Err` with a message about windows.
#[test]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
fn historical_library_fails_when_no_valid_windows() {
    use cobre_core::system::SystemBuilder;

    // Historical scheme with empty inflow_history guarantees zero candidate
    // years in discovery.
    use chrono::NaiveDate;
    use cobre_core::scenario::InflowModel;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };
    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

    let stages: Vec<Stage> = (0..2usize)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
            season_id: Some(i % 12),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 720.0,
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..2usize)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..2usize)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
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
            n_stages: 2,
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
                water_withdrawal_m3s: 0.0,
            },
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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

    // Historical scheme but NO inflow_history data — discovery must fail.
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("system: valid");

    let config = minimal_config_with_schemes(1, 5, Some("historical"), None, None);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::Historical),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let result = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    );

    assert!(result.is_err(), "expected Err when no historical data");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("window") || err_msg.contains("historical"),
        "error should mention windows or historical, got: {err_msg}"
    );
}

/// Given a `Config` with training inflow scheme `InSample` and simulation
/// inflow scheme `OutOfSample`, when `StudySetup::new()` is called, then
/// `training_ctx().inflow_scheme` is `InSample` and
/// `simulation_ctx().inflow_scheme` is `OutOfSample`.
#[test]
fn test_simulate_uses_simulation_scheme() {
    let system = minimal_system(2);

    // Training: InSample (default). Simulation: OutOfSample.
    let mut config = minimal_config(1, 5);
    config.simulation.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(99),
        historical_years: None,
        inflow: Some(RawClassConfigEntry {
            scheme: "out_of_sample".to_string(),
        }),
        load: None,
        ncs: None,
    });

    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let train_ctx = setup.training_ctx();
    assert_eq!(
        train_ctx.inflow_scheme,
        SamplingScheme::InSample,
        "training context must use InSample inflow scheme"
    );

    let sim_ctx = setup.simulation_ctx();
    assert_eq!(
        sim_ctx.inflow_scheme,
        SamplingScheme::OutOfSample,
        "simulation context must use OutOfSample inflow scheme"
    );
}

/// Given a `Config` with training inflow scheme `InSample` and simulation
/// inflow scheme `Historical`, when `StudySetup::new()` is called on a
/// system that has inflow history, then `training_ctx().historical_library`
/// is `None` and `simulation_ctx().historical_library` is `Some`.
#[test]
fn test_sim_historical_library_built_when_sim_scheme_is_historical() {
    let system = system_with_historical_inflow(2);

    // Training: InSample. Simulation: Historical.
    let mut config = minimal_config(1, 5);
    config.simulation.scenario_source = Some(RawScenarioSourceConfig {
        seed: Some(42),
        historical_years: None,
        inflow: Some(RawClassConfigEntry {
            scheme: "historical".to_string(),
        }),
        load: None,
        ncs: None,
    });

    // The stochastic context is built for the training scheme (InSample).
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert!(
        setup.training_ctx().historical_library.is_none(),
        "training context must NOT have a historical library when scheme is InSample"
    );
    assert!(
        setup.simulation_ctx().historical_library.is_some(),
        "simulation context must have a historical library when sim scheme is Historical"
    );
}

/// Build a minimal system identical to [`minimal_system`] except that the single
/// thermal carries the given `anticipated_config` and each study stage's single
/// block runs for the corresponding `stage_hours` entry. `k_max_bounds` sets
/// `BoundsCountsSpec::k_max` so the thermal stage-bounds axis is wide enough for
/// delivery-stage padding.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn minimal_system_with_anticipated(
    stage_hours: &[f64],
    anticipated_config: AnticipatedConfig,
    k_max_bounds: usize,
) -> cobre_core::System {
    use chrono::NaiveDate;

    let n_stages = stage_hours.len();

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(anticipated_config),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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
                duration_hours: stage_hours[i],
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
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: k_max_bounds,
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
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("minimal_system_with_anticipated: valid")
}

/// [`minimal_system_with_anticipated`] with `n_stages` uniform 744 h stages and a
/// `LeadStages(lead_stages)` thermal — the pre-delivery-anchor fixture.
fn minimal_system_with_anticipated_lead_stages(
    n_stages: usize,
    lead_stages: u32,
) -> cobre_core::System {
    minimal_system_with_anticipated(
        &vec![744.0; n_stages],
        AnticipatedConfig::LeadStages(lead_stages),
        lead_stages as usize,
    )
}

/// Given a `StudySetup::new` call on a system with one anticipated thermal
/// (`K_i = 2`), when the test inspects the resulting indexer metadata, then
/// `n_anticipated == 1`, `k_max == 2`, and `anticipated_lead_stages == [2]`.
#[test]
fn setup_wires_anticipated_metadata_into_indexer() {
    let system = minimal_system_with_anticipated_lead_stages(2, 2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert_eq!(
        setup.stage_data.state.n_anticipated, 1,
        "expected n_anticipated == 1"
    );
    assert_eq!(setup.stage_data.state.k_max, 2, "expected k_max == 2");
    assert_eq!(
        setup.stage_data.state.anticipated_lead_stages,
        vec![2],
        "expected anticipated_lead_stages == [2]"
    );
}

/// AC#1: a `LeadStages(2)` plant on a 5-stage uniform study resolves to a
/// per-plant depth whose max is `2` with singleton in-horizon decision sets
/// `{t+2}`, and the resulting `k_max` and `state_dimension` equal the
/// pre-delivery-anchor values (`k_max == 2`, `n_state == 3`).
#[test]
fn setup_leadstages_resolution_preserves_k_max_and_state_dimension() {
    let system = minimal_system_with_anticipated_lead_stages(5, 2);
    let (resolution, lead_stages) = super::resolve_anticipated_commitments(&system);

    assert_eq!(lead_stages, vec![2], "LeadStages keeps the constant ℓ == 2");
    let point = &resolution.per_plant[0];
    assert_eq!(
        point.depth.iter().copied().max(),
        Some(2),
        "per-plant depth max == ℓ == 2"
    );
    assert_eq!(point.decision_sets[0], vec![2], "C(0) == {{2}}");
    assert_eq!(point.decision_sets[1], vec![3], "C(1) == {{3}}");
    assert_eq!(point.decision_sets[2], vec![4], "C(2) == {{4}}");
    assert_eq!(resolution.k_max, 2, "delivery-anchored ring depth == 2");

    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");
    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert_eq!(setup.stage_data.state.k_max, 2, "k_max unchanged");
    // n_state = N*(1+L) + A*k_max = 1*(1+0) + 1*2 = 3 (no PAR lags, one hydro).
    assert_eq!(
        setup.stage_data.state.n_state, 3,
        "state_dimension unchanged"
    );
    assert_eq!(
        setup
            .stage_data
            .state
            .anticipated_resolution
            .per_plant
            .len(),
        1,
        "resolution threaded onto StageData.state"
    );
}

/// AC#2 (hand-derived): a `LeadTime(720.0)` plant on the weekly-then-monthly PMO
/// calendar `[168,168,168,168,720,720]` resolves via the end-anchored
/// `resolve_point` decider contract to `decider ==
/// [None,None,None,None,Some(3),Some(4)]`, `C(3) == {4}`, `C(4) == {5}`, and
/// `depth == [0,0,0,1,1,0]` (ring depth 1).
#[test]
fn test_anticipated_resolve_point_pmo_calendar() {
    let system = minimal_system_with_anticipated(
        &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0],
        AnticipatedConfig::LeadTime(720.0),
        6,
    );
    let (resolution, _) = super::resolve_anticipated_commitments(&system);
    let point = &resolution.per_plant[0];

    assert_eq!(
        point.decider,
        vec![None, None, None, None, Some(3), Some(4)]
    );
    assert_eq!(point.decision_sets[3], vec![4]);
    assert_eq!(point.decision_sets[4], vec![5]);
    assert_eq!(point.depth, vec![0, 0, 0, 1, 1, 0]);
    assert_eq!(resolution.k_max, 1);
}

/// AC#3 (hand-derived): a `LeadTime(720.0)` plant on the monthly-then-weekly
/// fan-out calendar `[720,168,168,168,168,168]` resolves to a coarse decider 0
/// committing four fine delivery stages — `C(0) == {1,2,3,4}` (|C(0)| == 4) —
/// with `depth == [4,4,3,2,1,0]` and ring depth 4.
#[test]
fn test_anticipated_resolve_point_fanout_calendar() {
    let system = minimal_system_with_anticipated(
        &[720.0, 168.0, 168.0, 168.0, 168.0, 168.0],
        AnticipatedConfig::LeadTime(720.0),
        6,
    );
    let (resolution, _) = super::resolve_anticipated_commitments(&system);
    let point = &resolution.per_plant[0];

    assert_eq!(point.decision_sets[0], vec![1, 2, 3, 4]);
    assert_eq!(point.decision_sets[0].len(), 4);
    assert_eq!(point.depth, vec![4, 4, 3, 2, 1, 0]);
    assert_eq!(resolution.k_max, 4);
}

/// Assert the canonical `StageData.state` role-(a) layout is internally
/// consistent and finalized, and that it agrees with a reference
/// [`StateLayout`] built independently from the same state-vector dimensions.
///
/// The role-(a) concern lives solely on `StateLayout`; the role-(b)
/// equipment geometry lives per stage on `StageGeometry`, so there is no
/// state half there to compare against. This checks that
/// `build_wired_indexer`'s `StateLayout` finalizes both caches and reproduces
/// a fresh `StateLayout::new` over the same `(N, L, A, k_max, leads)`
/// byte-for-byte — the property the single-owner extraction guarantees.
fn assert_state_layout_finalized(state: &StateLayout) {
    assert_eq!(
        state.state_to_lp_column_map.len(),
        state.n_state,
        "state_to_lp_column_map must be finalized to n_state length"
    );
    let reference = StateLayout::new(
        state.hydro_count,
        state.max_par_order,
        state.n_buckets,
        state.transit_bucket_column_order.clone(),
        state.n_anticipated,
        state.k_max,
        state.anticipated_lead_stages.clone(),
        &vec![state.max_par_order; state.hydro_count],
    );
    assert_eq!(
        state.state_to_lp_column_map, reference.state_to_lp_column_map,
        "state_to_lp_column_map must match a fresh StateLayout::new"
    );
    assert_eq!(state.n_state, reference.n_state, "n_state must match");
    assert_eq!(state.theta, reference.theta, "theta column must match");
    assert_eq!(state.storage, reference.storage, "storage range must match");
    assert_eq!(
        state.inflow_lags, reference.inflow_lags,
        "inflow_lags range must match"
    );
    assert_eq!(
        state.transit_buckets_out, reference.transit_buckets_out,
        "transit_buckets_out range must match"
    );
    assert_eq!(
        state.anticipated_state, reference.anticipated_state,
        "anticipated_state range must match"
    );
    assert_eq!(
        state.anticipated_slots_out, reference.anticipated_slots_out,
        "anticipated_slots_out range must match"
    );
    assert_eq!(
        state.z_inflow, reference.z_inflow,
        "z_inflow range must match"
    );
    assert_eq!(
        state.storage_in, reference.storage_in,
        "storage_in range must match"
    );
    assert_eq!(
        state.transit_buckets_in, reference.transit_buckets_in,
        "transit_buckets_in range must match"
    );
}

/// AC#1 (uniform): `StageData.state` finalized by `build_wired_indexer` is
/// internally consistent and finalized for a storage+lag study.
#[test]
fn stage_data_state_matches_indexer_role_a_uniform() {
    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert_state_layout_finalized(&setup.stage_data.state);
}

/// Build a 2-hydro cascade with hydro 2 (upstream) declaring a travel-time
/// arc into hydro 1 (downstream) — mirrors [`minimal_system`]'s single-hydro
/// template, extended to a cascade so `bucket_topology.n_buckets > 0`.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements
)]
fn system_with_travel_time_arc(n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(3),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let make_hydro =
        |id: i32, name: &str, downstream_id: Option<i32>, travel_time_hours: Option<f64>| Hydro {
            id: EntityId(id),
            name: name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: downstream_id.map(EntityId),
            travel_time_hours,
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
        .flat_map(|i| {
            [1_i32, 2].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
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
            storage_violation_below_cost: 500.0,
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

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 1,
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
            n_hydros: 2,
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
        .hydros(vec![
            make_hydro(1, "H1_downstream", None, None),
            make_hydro(2, "H2_upstream", Some(1), Some(2000.0)),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        })
        .build()
        .expect("system_with_travel_time_arc: valid")
}

/// A declared travel-time arc (`n_buckets > 0`) must size `StageData.state` (the
/// role-(a) `StateLayout` `build_wired_indexer` stores) and the actual LP
/// template (sized by `build_stage_templates`'s own, independently-recomputed
/// `StateLayout`) to the SAME `n_state` — the cross-construction agreement the
/// contained `template.rs` recompute fix keeps intact.
#[test]
fn setup_state_and_stage_template_agree_on_n_state_with_declared_arc() {
    let system = system_with_travel_time_arc(6);
    let setup = setup_from_system(&system);

    assert!(
        setup.stage_data.state.n_buckets > 0,
        "fixture must declare a real travel-time arc"
    );
    assert_eq!(
        setup.stage_data.stage_templates.templates[0].n_state, setup.stage_data.state.n_state,
        "template.rs's independently-recomputed StateLayout must agree with \
         StageData.state on n_state"
    );
}

/// Geometry byte-identity: the production stage-0 `StageGeometry` (built by
/// `StageGeometry::from_layout` and stored in `StageTemplates::geometry_per_stage[0]`)
/// is byte-identical to an independent `test_support::geometry` build from
/// the same equipment dimensions (the role-(b) analogue of
/// `assert_state_layout_finalized`). A divergence means the per-stage geometry
/// the `StageLayout` produces drifted from the column/row arithmetic the
/// fixture reproduces.
#[test]
fn stage_data_geometry_role_b_matches_reference_build() {
    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let geometry = &setup.stage_data.stage_templates.geometry_per_stage[0];
    let study_dims = &setup.stage_data.study_dims;
    // Rebuild the reference geometry independently from the single-owner
    // `study_dims`, so a divergence between it and the production geometry fails
    // the test.
    let dims = crate::test_support::GeometryDims {
        hydro_count: geometry.water_balance.len(),
        max_par_order: 0, // role-(b) ranges do not depend on L
        n_thermals: study_dims.n_thermals,
        n_lines: study_dims.n_lines,
        n_buses: study_dims.n_buses,
        n_blks: geometry.n_blks,
        has_inflow_penalty: study_dims.has_inflow_penalty,
        max_deficit_segments: study_dims.max_deficit_segments,
        n_anticipated: study_dims.anticipated_thermal_indices.len(),
        k_max: 0,
        anticipated_thermal_indices: study_dims.anticipated_thermal_indices.clone(),
    };
    let reference = crate::test_support::geometry(
        &dims,
        geometry.fpha_hydro_indices.clone(),
        // minimal_system has no FPHA planes; mirror the built geometry's
        // FPHA-hydro count with a placeholder plane count per hydro.
        &vec![1usize; geometry.fpha_hydro_indices.len()],
        geometry.evap_hydro_indices.clone(),
    );

    // Role-(b) equipment/slack column ranges.
    assert_eq!(reference.turbine, geometry.turbine, "turbine range");
    assert_eq!(reference.spillage, geometry.spillage, "spillage range");
    assert_eq!(reference.diversion, geometry.diversion, "diversion range");
    assert_eq!(reference.thermal, geometry.thermal, "thermal range");
    assert_eq!(
        reference.anticipated_decision, geometry.anticipated_decision,
        "anticipated_decision range"
    );
    assert_eq!(reference.line_fwd, geometry.line_fwd, "line_fwd range");
    assert_eq!(reference.line_rev, geometry.line_rev, "line_rev range");
    assert_eq!(reference.deficit, geometry.deficit, "deficit range");
    assert_eq!(reference.excess, geometry.excess, "excess range");
    assert_eq!(
        reference.withdrawal_slack_neg, geometry.withdrawal_slack_neg,
        "withdrawal_slack_neg range"
    );
    assert_eq!(
        reference.withdrawal_slack_pos, geometry.withdrawal_slack_pos,
        "withdrawal_slack_pos range"
    );
    // Role-(b) constraint row ranges + surviving stride scalars.
    assert_eq!(
        reference.water_balance, geometry.water_balance,
        "water_balance"
    );
    assert_eq!(
        reference.load_balance, geometry.load_balance,
        "load_balance"
    );
    assert_eq!(
        reference.z_inflow_row_start, geometry.z_inflow_row_start,
        "z_inflow_row_start"
    );
    assert_eq!(reference.n_blks, geometry.n_blks, "n_blks");
}

/// AC#1 (anticipated): `StageData.state` is byte-identical to the indexer's
/// role-(a) when anticipated thermals are present (`K_i = 2`), exercising
/// the `anticipated_slots_out` / `anticipated_state` ranges and the
/// anticipated entries of the nonzero mask.
#[test]
fn stage_data_state_matches_indexer_role_a_anticipated() {
    let system = minimal_system_with_anticipated_lead_stages(2, 2);
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    assert_eq!(setup.stage_data.state.n_anticipated, 1, "fixture sanity");
    assert_state_layout_finalized(&setup.stage_data.state);
}

/// AC#2 (cut-row byte-identity at the cut-path repoint): the production cut
/// row built by `build_cut_row_batch` reading role (a) from `StageData.state`
/// (a [`StateLayout`]) is byte-identical to an independent reference loop that
/// reads the same `StateLayout` (`theta`, `nonzero_state_indices`,
/// `lp_column_for_state`). This is the substitutability guarantee the cut-path
/// repoint relies on: after repointing the production builder onto
/// `StateLayout`, the mask, `theta`, and `lp_column_for_state` reads resolve
/// to the same LP columns and the same negated-scaled coefficients.
#[test]
fn cut_row_from_state_matches_reference_loop() {
    use crate::cut::FutureCostFunction;
    use crate::cut::row::build_cut_row_batch;

    let system = minimal_system(3);
    let config = minimal_config(2, 10);
    let stochastic = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");

    let setup = StudySetup::new(
        &system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(&system),
    )
    .expect("setup");

    let state = &setup.stage_data.state;
    let n_state = state.n_state;
    assert!(n_state > 0, "fixture must have a non-empty state vector");

    // `u32::try_from` + `f64::from` keeps the indices lossless without a
    // cast-precision lint relaxation.
    let mut fcf = FutureCostFunction::new(2, n_state, 1, 4, &[0; 2]);
    let coefficients: Vec<f64> = (0..n_state)
        .map(|j| 1.0 + f64::from(u32::try_from(j).expect("state index fits u32")))
        .collect();
    fcf.add_cut(0, 0, 0, 7.5, &coefficients);

    let from_production = build_cut_row_batch(
        &fcf,
        0,
        state,
        &crate::test_support::cut_state_projection(state),
        &[],
    );

    // Independent mirror of `build_cut_row_batch_into`'s mask-driven body over the
    // same `StateLayout` role-(a) reads; a disagreement means the cut-path repoint
    // changed the emitted row.
    let mut from_state = cobre_solver::RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    let theta_col = state.theta;
    let mask = &state.nonzero_state_indices;
    for (_slot, intercept, coeffs) in fcf.active_cuts(0) {
        from_state.row_starts.push(0);
        for &j in mask {
            let lp_col = state.lp_column_for_state(j);
            from_state
                .col_indices
                .push(i32::try_from(lp_col).expect("col fits i32"));
            from_state.values.push(-coeffs[j]);
        }
        from_state
            .col_indices
            .push(i32::try_from(theta_col).expect("theta fits i32"));
        from_state.values.push(1.0);
        from_state.row_lower.push(intercept);
        from_state.row_upper.push(f64::INFINITY);
    }
    from_state
        .row_starts
        .push(i32::try_from(mask.len() + 1).expect("nnz fits i32"));
    from_state.num_rows = 1;

    assert_eq!(
        from_production.row_starts, from_state.row_starts,
        "row_starts must be byte-identical"
    );
    assert_eq!(
        from_production.col_indices, from_state.col_indices,
        "col_indices must be byte-identical"
    );
    assert_eq!(
        from_production.values, from_state.values,
        "values must be byte-identical"
    );
    assert_eq!(
        from_production.row_lower, from_state.row_lower,
        "row_lower must be byte-identical"
    );
    assert_eq!(
        from_production.row_upper, from_state.row_upper,
        "row_upper must be byte-identical"
    );
    assert_eq!(
        from_production.num_rows, from_state.num_rows,
        "num_rows must match"
    );
}

// ── per-stage cut-pool sizing (build_cut_state_layouts) ───────────────────

/// Build a PAR(2) study (`max_par_order = 2`, so `L > 0`) with 1 hydro, 1
/// thermal, 1 bus, and one stage per entry in `state_configs`. Each stage
/// takes the matching `StageStateConfig`; AR(2) coefficients plus pre-study
/// inflow models at stage ids -1/-2 give the PAR builder the lag statistics
/// it needs, so the global `StateLayout` has `n_state = N*(1 + 2)`.
#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
fn par2_system_with_state_configs(state_configs: &[StageStateConfig]) -> cobre_core::System {
    use chrono::NaiveDate;

    const PHI_1: f64 = 0.5;
    const PHI_2: f64 = 0.2;
    const N_SEASONS: usize = 12;
    let hydro_id = EntityId(3);

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: hydro_id,
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 200.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
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

    let n_stages = state_configs.len();
    let stages: Vec<Stage> = state_configs
        .iter()
        .enumerate()
        .map(|(i, &state_config)| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(i % N_SEASONS),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let mut inflow_models: Vec<InflowModel> = Vec::new();
    for pre_id in [-2_i32, -1_i32] {
        inflow_models.push(InflowModel {
            hydro_id,
            stage_id: pre_id,
            mean_m3s: 1000.0,
            std_m3s: 200.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        });
    }
    for i in 0..n_stages {
        inflow_models.push(InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 1000.0,
            std_m3s: 200.0,
            ar_coefficients: vec![PHI_1, PHI_2],
            residual_std_ratio: 0.7,
            annual: None,
        });
    }

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
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
            n_stages: n_stages.max(1),
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 500.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
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
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_stages.max(1),
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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

    let initial_conditions = InitialConditions {
        past_inflows: vec![HydroPastInflows {
            hydro_id,
            values_m3s: vec![1000.0, 1000.0],
            season_ids: None,
        }],
        ..InitialConditions::default()
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("par2_system_with_state_configs: valid")
}

/// Construct a [`StudySetup`] from `system` with a single-iteration config.
fn setup_from_system(system: &cobre_core::System) -> StudySetup {
    let config = minimal_config(1, 10);
    let stochastic = build_stochastic_context(
        system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("stochastic context");
    StudySetup::new(
        system,
        &config,
        stochastic,
        PrepareHydroModelsResult::default_from_system(system),
    )
    .expect("setup")
}

/// The `t + 1` governance rule: with `stages[1].state_config.inflow_lags =
/// false` and all other stages enabling lags, only `pools[0]` (governed by
/// stage 1) drops the lag dimensions; every other pool keeps the full global
/// `n_state`. With N=1 hydro, L=2, A=0: storage-only is `N = 1`,
/// `n_state = N*(1 + L) = 3`.
#[test]
fn cut_pool_sizing_t_plus_1_reduces_pool_zero_for_lagless_successor() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let storage_only = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    // Stage 1 is lag-less; it governs pool 0 (its predecessor).
    let system = par2_system_with_state_configs(&[lags, storage_only, lags, lags]);
    let setup = setup_from_system(&system);

    let global_n_state = setup.stage_data.state.n_state;
    assert_eq!(global_n_state, 3, "N=1, L=2 → n_state = N*(1+L) = 3");
    assert_eq!(setup.fcf.pools.len(), 4);

    // pool 0 ← stages[1] (inflow_lags=false) → storage-only = N + A*k_max = 1.
    assert_eq!(
        setup.fcf.pools[0].state_dimension, 1,
        "pool 0 is governed by stage 1's lag-less config (the t+1 rule)"
    );
    // pools 1, 2 ← stages[2], stages[3] (lags) → global n_state.
    assert_eq!(setup.fcf.pools[1].state_dimension, global_n_state);
    assert_eq!(setup.fcf.pools[2].state_dimension, global_n_state);
    // Terminal pool 3 ← full global n_state (no successor stage).
    assert_eq!(setup.fcf.pools[3].state_dimension, global_n_state);
}

/// Result-neutrality: when every stage enables all dimensions (the default
/// for every shipped case), every pool's `state_dimension` equals the global
/// `StateLayout::n_state`. This is the bit-identical-to-today guarantee.
#[test]
fn cut_pool_sizing_all_enabled_matches_global_n_state() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let system = par2_system_with_state_configs(&[lags, lags, lags, lags]);
    let setup = setup_from_system(&system);

    let global_n_state = setup.stage_data.state.n_state;
    assert_eq!(global_n_state, 3);
    assert_eq!(setup.fcf.pools.len(), 4);
    for (t, pool) in setup.fcf.pools.iter().enumerate() {
        assert_eq!(
            pool.state_dimension, global_n_state,
            "all-enabled pool {t} must equal the global n_state"
        );
    }
    // The FCF's global field is unchanged from today's single-value model.
    assert_eq!(setup.fcf.state_dimension, global_n_state);
}

/// One [`CutStateProjection`] is stored per pool and is reachable, with each
/// layout's `n_state()` equal to its pool's `state_dimension` (the pairing
/// the backward pass relies on to extract duals at the right dimension).
#[test]
fn cut_state_layouts_stored_one_per_pool_and_reachable() {
    let lags = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    let storage_only = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    let system = par2_system_with_state_configs(&[lags, storage_only, lags, lags]);
    let setup = setup_from_system(&system);

    assert_eq!(
        setup.stage_data.cut_state_layouts.len(),
        setup.fcf.pools.len(),
        "exactly one CutStateProjection per pool",
    );
    for (t, layout) in setup.stage_data.cut_state_layouts.iter().enumerate() {
        assert_eq!(
            layout.n_state(),
            setup.fcf.pools[t].state_dimension,
            "cut_state_layouts[{t}].n_state() must match pool {t} dimension",
        );
    }

    // Pool `t` is sized by stage `t+1`'s config. The `storage_only` config sits
    // at stage index 1, so it sizes pool 0: its cut dimension drops the AR lags
    // to exactly the hydro (storage) count N, while pool 1 (sized by stage 2's
    // lag-enabled config) carries storage + lags (N*(1+L) > N).
    let n_hydros = setup.stage_data.state.hydro_count;
    assert_eq!(
        setup.fcf.pools[0].state_dimension, n_hydros,
        "storage-only pool 0 must have cut dimension N (lags dropped)",
    );
    assert!(
        setup.fcf.pools[1].state_dimension > n_hydros,
        "lag-enabled pool 1 must carry storage + lags (dimension > N)",
    );
}

// ---------------------------------------------------------------------------
// K = 0 sub-stage lead (`c(m) = m`) — exclude-with-advisory (D4)
// ---------------------------------------------------------------------------

/// Minimal WARN-capturing `tracing::Subscriber`, mirroring
/// `params::tests::WarnRecorder` (the established setup-time advisory-test
/// pattern for this crate).
struct WarnRecorder {
    messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl WarnRecorder {
    fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                messages: std::sync::Arc::clone(&messages),
            },
            messages,
        )
    }
}

impl tracing::Subscriber for WarnRecorder {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }

    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            struct MessageVisitor(String);
            impl tracing::field::Visit for MessageVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }

                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "message" {
                        self.0 = value.to_string();
                    }
                }
            }
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.messages.lock().unwrap().push(visitor.0);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Hand-derived: a `LeadTime(720.0)` plant on the uniform 31-day-month
/// calendar `[744, 744, 744, 744]` h resolves `c(m) = m` at every delivery
/// stage (the 720h lead is shorter than each 744h stage, so `end_m - 720`
/// always lands inside stage `m`'s own window) — the `K = 0` sub-stage-lead
/// degeneracy, `depth == [0, 0, 0, 0]` and `resolution.k_max == 0` (never an
/// underflow).
#[test]
fn test_anticipated_resolve_point_k0_uniform_calendar() {
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0, 744.0, 744.0],
        AnticipatedConfig::LeadTime(720.0),
        0,
    );
    let (resolution, lead_stages) = super::resolve_anticipated_commitments(&system);
    let point = &resolution.per_plant[0];

    assert_eq!(
        point.decider,
        vec![Some(0), Some(1), Some(2), Some(3)],
        "every delivery stage self-delivers (K=0)"
    );
    assert_eq!(point.depth, vec![0, 0, 0, 0]);
    assert_eq!(resolution.k_max, 0, "ring depth collapses to 0");
    assert_eq!(resolution.max_fanout, 0, "no genuine fan-out either");
    assert_eq!(
        point.self_delivered_stages().collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "every stage is a self-delivery"
    );
    // The still-live constant-lead machinery's placeholder is the per-plant
    // max depth — 0 here, never underflowed downstream.
    assert_eq!(lead_stages, vec![0]);
}

/// `resolve_anticipated_commitments` emits one `tracing::WARN` event per
/// `K = 0` self-delivered stage (D4: exclude-with-advisory, never a hard
/// error), naming the thermal, the stage, and the `lead_stages == 0`
/// stage-count alternative. Setup/load-time only (this is a direct,
/// non-per-trajectory call).
#[test]
fn resolve_anticipated_commitments_warns_on_k0_sub_stage_lead() {
    let system = minimal_system_with_anticipated(
        &[744.0, 744.0, 744.0, 744.0],
        AnticipatedConfig::LeadTime(720.0),
        0,
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let _ = super::resolve_anticipated_commitments(&system);
    });
    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("lead_stages == 0"))
        .map(std::string::String::as_str)
        .collect();
    assert_eq!(
        relevant.len(),
        4,
        "expected one advisory per self-delivered stage (4 stages), got: {recorded:?}"
    );
    for (stage, msg) in relevant.iter().enumerate() {
        assert!(
            msg.contains(&format!("stage {stage}")),
            "advisory {stage} must name its stage; got: {msg}"
        );
        assert!(
            msg.contains("T1"),
            "advisory {stage} must name the plant; got: {msg}"
        );
    }
}

/// A `LeadStages` plant never triggers the `K = 0` advisory: a positive
/// stage-count lead never resolves `c(m) = m`.
#[test]
fn resolve_anticipated_commitments_leadstages_never_warns() {
    let system = minimal_system_with_anticipated_lead_stages(5, 2);

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let _ = super::resolve_anticipated_commitments(&system);
    });
    let recorded = messages.lock().unwrap();
    assert!(
        recorded.iter().all(|msg| !msg.contains("lead_stages == 0")),
        "LeadStages must never trigger the K=0 advisory; got: {recorded:?}"
    );
}
