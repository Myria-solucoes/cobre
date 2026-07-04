#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::doc_overindented_list_items,
    clippy::similar_names
)]

use chrono::NaiveDate;
use cobre_core::{
    AnticipatedConfig, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
    ContractStageBounds, ContractType, DeficitSegment, EnergyContract, EntityId, Hydro,
    HydroGenerationModel, HydroPenalties, HydroStageBounds, HydroStagePenalties, LineStageBounds,
    LineStagePenalties, LoadModel, NcsStagePenalties, NoiseMethod, PenaltiesCountsSpec,
    PenaltiesDefaults, PumpingStageBounds, PumpingStation, ResolvedBounds, ResolvedPenalties,
    ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, SystemBuilder, Thermal,
    ThermalStageBounds,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::PrepareHydroModelsResult;
use crate::inflow_method::InflowNonNegativityMethod;
use crate::resolved_parameters::ResolvedParameters;

use super::super::test_support::state_layout_for;

// ── Fixtures ─────────────────────────────────────────────────────────────

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

/// Build a one-bus system with exactly the thermals provided.
///
/// Uses one study stage with a single block of 744 hours and no hydros.
fn system_with_thermals(thermals: Vec<Thermal>) -> cobre_core::System {
    let n_thermals = thermals.len();
    let n_stages = 1_usize;

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

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    // k_max for the anticipated thermals in this system
    let k_max = thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|c| c.lead_stages().unwrap() as usize)
        .max()
        .unwrap_or(0);

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max,
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
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_thermals: valid system")
}

/// Build empty [`ResolvedParameters`] (no parameters).
fn empty_resolved_params() -> ResolvedParameters {
    ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    }
}

/// All-zero per-plant [`HydroPenalties`] for fixture hydros.
fn hydro_penalties_zero() -> HydroPenalties {
    HydroPenalties {
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
        inflow_nonnegativity_cost: 0.0,
    }
}

/// Minimal independent (no-downstream) hydro for pumping-station refs.
fn fixture_hydro(id: i32) -> Hydro {
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
        max_turbined_m3s: 50.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 45.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: hydro_penalties_zero(),
    }
}

/// Build a one-bus, two-hydro system with the supplied pumping stations.
///
/// `SystemBuilder::build` sorts every entity Vec by `id.0`, so passing
/// stations out of declaration order exercises the canonical-ordering
/// guarantee that `build_template_build_ctx` relies on when threading the
/// slice into `ctx.pumping_stations`/`pumping_pos`. The two hydros and bus
/// exist solely to satisfy pumping-station reference validation.
fn system_with_pumping_stations(stations: Vec<PumpingStation>) -> cobre_core::System {
    let n_pumping = stations.len();
    let n_hydros = 2_usize;
    let n_stages = 1_usize;

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

    let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping,
            n_contracts: 0,
            n_stages,
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
                max_flow_m3s: 100.0,
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
            n_hydros,
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
        .hydros(hydros)
        .pumping_stations(stations)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_pumping_stations: valid system")
}

/// Build a pumping station with the given id (bus/hydro refs fixed to the
/// fixture entities; flow window and consumption are non-degenerate).
fn fixture_pumping_station(id: i32) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: format!("P{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        source_hydro_id: EntityId(1),
        destination_hydro_id: EntityId(2),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s: 0.5,
        min_flow_m3s: 0.0,
        max_flow_m3s: 100.0,
    }
}

// ── Pumping data threaded into TemplateBuildCtx ────────────────────────────

/// Stations declared out of ID order are exposed ID-sorted on the ctx, and
/// `pumping_pos` maps each station id to its slot in that sorted slice.
///
/// Declaration order `[30, 10, 20]` must become `[10, 20, 30]` on the ctx
/// (the canonical sort applied by `SystemBuilder::build`), with
/// `pumping_pos = {10->0, 20->1, 30->2}`.
#[test]
fn build_template_build_ctx_pumping_stations_id_sorted_and_pos_mapped() {
    let stations = vec![
        fixture_pumping_station(30),
        fixture_pumping_station(10),
        fixture_pumping_station(20),
    ];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    let ids: Vec<i32> = ctx.pumping_stations.iter().map(|p| p.id.0).collect();
    assert_eq!(
        ids,
        vec![10, 20, 30],
        "ctx.pumping_stations must be ID-sorted regardless of declaration order"
    );

    assert_eq!(
        ctx.pumping_pos.len(),
        3,
        "pumping_pos has one entry per station"
    );
    assert_eq!(ctx.pumping_pos[&EntityId(10)], 0);
    assert_eq!(ctx.pumping_pos[&EntityId(20)], 1);
    assert_eq!(ctx.pumping_pos[&EntityId(30)], 2);

    // The position map must agree with the slot order of the sorted slice.
    for (slot, station) in ctx.pumping_stations.iter().enumerate() {
        assert_eq!(
            ctx.pumping_pos[&station.id], slot,
            "pumping_pos[{:?}] must equal its slot in the sorted slice",
            station.id
        );
    }
}

/// `ctx.n_pumping` equals `pumping_stations.len()` and the resolved-bounds
/// station count, and that count is the source `StageLayout` reserves from.
#[test]
fn build_template_build_ctx_n_pumping_matches_slice_and_bounds() {
    let stations = vec![fixture_pumping_station(7), fixture_pumping_station(3)];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    assert_eq!(
        ctx.n_pumping,
        ctx.pumping_stations.len(),
        "n_pumping == slice len"
    );
    assert_eq!(ctx.n_pumping, 2, "two stations were declared");
    assert_eq!(
        ctx.n_pumping,
        ctx.resolved.bounds.n_pumping(),
        "ctx.n_pumping must agree with the resolved-bounds station count"
    );

    // The re-pointed ctx count flows through to the layout: StageLayout reads
    // its `n_pumping` from `ctx.n_pumping`. (The block-major column reservation
    // itself is pinned by the layout-module test
    // `pumping_layout_reserves_block_major_columns`.)
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let state = state_layout_for(&ctx);
    let layout = super::super::layout::StageLayout::new(&ctx, &state, stage, 0);
    assert_eq!(
        layout.n_pumping, ctx.n_pumping,
        "StageLayout.n_pumping must equal the ctx-sourced count"
    );
}

/// `build_stage_templates` records the layout-owned pumping column base for
/// every stage: `pumping_col_starts[t]` equals
/// `StageLayout::new(..).col_pumping_start`, and the scalar `n_pumping`
/// equals `StageLayout::new(..).n_pumping` (constant across stages under the
/// dense layout).
///
/// This pins the threading contract the simulation extraction pipeline reads
/// from: the column base is sourced from the layout, the sole owner of the
/// pumping-flow column base.
#[test]
fn build_stage_templates_records_layout_pumping_col_start_per_stage() {
    let stations = vec![fixture_pumping_station(5), fixture_pumping_station(2)];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let templates = super::build_stage_templates(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");

    // Rebuild the ctx once so each stage's StageLayout can be reconstructed
    // and compared against the recorded per-stage value.
    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    assert_eq!(templates.pumping_col_starts.len(), study_stages.len());
    assert_eq!(
        templates.n_pumping, 2,
        "two stations were declared; the dense count is a scalar"
    );
    for (t, stage) in study_stages.iter().enumerate() {
        let state = state_layout_for(&ctx);
        let layout = super::super::layout::StageLayout::new(&ctx, &state, stage, t);
        assert_eq!(
            templates.pumping_col_starts[t], layout.col_pumping_start,
            "stage {t}: pumping_col_starts must equal layout.col_pumping_start"
        );
        assert_eq!(
            templates.n_pumping, layout.n_pumping,
            "stage {t}: scalar n_pumping must equal layout.n_pumping",
        );
    }
}

// ── Contract data threaded into TemplateBuildCtx and StageGeometry ─────────

/// Build an energy contract with the given id and direction (bus fixed to the
/// fixture bus; a non-degenerate price/power window).
fn fixture_contract(id: i32, contract_type: ContractType) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name: format!("C{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 100.0,
        min_mw: 0.0,
        max_mw: 500.0,
    }
}

/// Build a one-bus, two-hydro system with the supplied contracts and a single
/// `n_blks`-block study stage. `n_contracts` matches the slice so the
/// resolved-bounds count check holds; the two hydros and bus exist solely to
/// satisfy contract bus-reference validation and give the layout pumping-end
/// anchor a non-trivial column prefix.
fn system_with_contracts(contracts: Vec<EnergyContract>, n_blks: usize) -> cobre_core::System {
    let n_contracts = contracts.len();
    let n_hydros = 2_usize;
    let n_stages = 1_usize;

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

    let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 372.0,
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts,
            n_stages,
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
                max_mw: 500.0,
                price_per_mwh: 100.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
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
        .hydros(hydros)
        .contracts(contracts)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_contracts: valid system")
}

/// One import + one export contract (declared out of ID order) are exposed
/// ID-sorted on the ctx; `contract_pos` maps each id to its slot, and the
/// per-direction counts are derived by `contract_type`.
#[test]
fn build_template_build_ctx_contracts_counted_and_pos_mapped() {
    let contracts = vec![
        fixture_contract(20, ContractType::Export),
        fixture_contract(10, ContractType::Import),
    ];
    let system = system_with_contracts(contracts, 1);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    assert_eq!(ctx.contracts.len(), 2);
    let ids: Vec<i32> = ctx.contracts.iter().map(|c| c.id.0).collect();
    assert_eq!(
        ids,
        vec![10, 20],
        "ctx.contracts must be ID-sorted regardless of declaration order"
    );
    assert_eq!(ctx.n_contract_import, 1);
    assert_eq!(ctx.n_contract_export, 1);
    assert_eq!(ctx.contract_pos[&EntityId(10)], 0);
    assert_eq!(ctx.contract_pos[&EntityId(20)], 1);
    for (slot, contract) in ctx.contracts.iter().enumerate() {
        assert_eq!(
            ctx.contract_pos[&contract.id], slot,
            "contract_pos[{:?}] must equal its slot in the sorted slice",
            contract.id
        );
    }
}

/// With `n_blks == 2` and one import + one export contract, `from_layout`
/// populates each contract column range with `n_contracts * n_blks` columns:
/// import follows pumping, export follows import.
#[test]
fn stage_geometry_from_layout_populates_contract_ranges() {
    let contracts = vec![
        fixture_contract(10, ContractType::Import),
        fixture_contract(20, ContractType::Export),
    ];
    let system = system_with_contracts(contracts, 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let state = state_layout_for(&ctx);
    let layout = super::super::layout::StageLayout::new(&ctx, &state, stage, 0);
    let geometry = super::StageGeometry::from_layout(&layout, stage.block_mode);

    assert_eq!(geometry.contract_import.len(), 2, "1 import * 2 blocks");
    assert_eq!(geometry.contract_export.len(), 2, "1 export * 2 blocks");
    assert_eq!(
        geometry.contract_import.start, layout.col_contract_import_start,
        "import range anchored at the layout import-block start"
    );
    assert_eq!(
        geometry.contract_export.start, geometry.contract_import.end,
        "export block immediately follows the import block"
    );
}

/// A contract-free system yields empty contract ranges anchored at the
/// pumping-end column (`start..start`, not `0..0`), leaving the prior column
/// layout byte-identical (parity-neutral).
#[test]
fn stage_geometry_from_layout_empty_contracts_are_pumping_end_anchored() {
    let system = system_with_contracts(vec![], 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let state = state_layout_for(&ctx);
    let layout = super::super::layout::StageLayout::new(&ctx, &state, stage, 0);
    let col_pumping_end = layout.col_pumping_start + layout.n_pumping * layout.n_blks;
    let geometry = super::StageGeometry::from_layout(&layout, stage.block_mode);

    assert!(geometry.contract_import.is_empty());
    assert!(geometry.contract_export.is_empty());
    assert_eq!(
        geometry.contract_import.start, col_pumping_end,
        "empty import range anchors at the pumping-end column, not 0"
    );
    assert_eq!(
        geometry.contract_export.start, col_pumping_end,
        "empty export range anchors at the pumping-end column, not 0"
    );
}

/// A resolved-bounds count divergence (one contract entity, `n_contracts: 0`
/// in the bounds table) trips the `debug_assert_eq!` in
/// `build_template_build_ctx`.
#[test]
#[should_panic(expected = "resolved-bounds")]
fn build_template_build_ctx_contract_count_divergence_panics() {
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
    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
    }];
    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];
    // n_contracts: 0 in the bounds table while one contract entity exists.
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
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .contracts(vec![fixture_contract(1, ContractType::Import)])
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("valid system; the count mismatch is caught downstream");
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let _ = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );
}

// ── AC-1 ─────────────────────────────────────────────────────────────────

/// AC-1: `build_template_build_ctx` populates anticipated metadata for a
/// system with `T_a`(K=2), `T_b`(no anticipated), `T_c`(K=3).
///
/// Expected: `n_anticipated`=2, `k_max`=3, `anticipated_lead_stages`=[2,3],
/// `anticipated_thermal_indices`=[0,2].
#[test]
fn build_template_build_ctx_populates_anticipated_metadata() {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T_a".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        },
        Thermal {
            id: EntityId(2),
            name: "T_b".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 20.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
        Thermal {
            id: EntityId(3),
            name: "T_c".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 30.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
        },
    ];
    let system = system_with_thermals(thermals);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    assert_eq!(ctx.n_anticipated, 2, "n_anticipated");
    assert_eq!(ctx.k_max, 3, "k_max");
    assert_eq!(
        ctx.anticipated_lead_stages,
        vec![2, 3],
        "anticipated_lead_stages"
    );
    assert_eq!(
        ctx.anticipated_thermal_indices,
        vec![0, 2],
        "anticipated_thermal_indices"
    );
}

// ── AC-2 ─────────────────────────────────────────────────────────────────

/// AC-2: `build_template_build_ctx` returns zeroed metadata when no
/// thermal has `anticipated_config`.
#[test]
fn build_template_build_ctx_zero_anticipated_when_none() {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
        Thermal {
            id: EntityId(2),
            name: "T2".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 20.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
    ];
    let system = system_with_thermals(thermals);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    assert_eq!(ctx.n_anticipated, 0, "n_anticipated");
    assert_eq!(ctx.k_max, 0, "k_max");
    assert!(
        ctx.anticipated_lead_stages.is_empty(),
        "anticipated_lead_stages"
    );
    assert!(
        ctx.anticipated_thermal_indices.is_empty(),
        "anticipated_thermal_indices"
    );
}

// ── Real declaration-order-invariance probe ──

/// Build a 5-stage 3-thermal system used by the order-invariance probe.
///
/// Three thermals (canonical EntityId order, since `SystemBuilder::build`
/// sorts by `EntityId`):
/// - `id=1`: anticipated K=2, max=120 MW, cost=50 $/MWh
/// - `id=2`: anticipated K=3, max=80 MW, cost=40 $/MWh
/// - `id=3`: standard thermal (no anticipation), max=200 MW, cost=500 $/MWh
///
/// `ResolvedBounds` is populated with per-thermal stage costs/limits matching
/// the per-thermal declarations (the default `BoundsDefaults::thermal` is uniform,
/// so a probe that relied on defaults would be trivial — distinct per-thermal
/// stage data is required to expose any latent order-dependence in the LP fill).
///
/// `n_stages = 5` ensures both anticipated decisions are active at `stage_idx=0`
/// (strict gate `t + K_i < n_stages` -> `2 < 5` and `3 < 5`).
fn anticipated_invariance_system() -> cobre_core::System {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T_ant_k2".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 120.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        },
        Thermal {
            id: EntityId(2),
            name: "T_ant_k3".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 40.0,
            min_generation_mw: 0.0,
            max_generation_mw: 80.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
        },
        Thermal {
            id: EntityId(3),
            name: "T_backup".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 500.0,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            anticipated_config: None,
        },
    ];

    let n_thermals = thermals.len();
    let n_stages = 5_usize;
    let k_max = 3_usize;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
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
        .map(|s| LoadModel {
            bus_id: EntityId(1),
            stage_id: s as i32,
            mean_mw: 150.0,
            std_mw: 0.0,
        })
        .collect();

    let mut resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max,
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

    // Per-thermal stage bounds (distinct so a permutation actually changes
    // the LP coefficients). The bounds table is indexed [thermal_idx][stage_idx]
    // with a stage axis of length `n_stages + k_max` (delivery-stage padding).
    let stage_axis_len = resolved_bounds.thermal_stage_axis_len();
    for t_idx in 0..n_thermals {
        for s_idx in 0..stage_axis_len {
            let tb = resolved_bounds.thermal_bounds_mut(t_idx, s_idx);
            match t_idx {
                0 => {
                    tb.max_generation_mw = 120.0;
                    tb.cost_per_mwh = 50.0;
                }
                1 => {
                    tb.max_generation_mw = 80.0;
                    tb.cost_per_mwh = 40.0;
                }
                2 => {
                    tb.max_generation_mw = 200.0;
                    tb.cost_per_mwh = 500.0;
                }
                _ => unreachable!("only 3 thermals"),
            }
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
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
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("anticipated_invariance_system: valid system")
}

/// Compare two `StageTemplate`s for bit-for-bit equivalence after applying
/// the swap-(0,1) permutation on anticipated-decision columns,
/// anticipated-state ring-buffer columns/rows (slot-major), and
/// anticipated-fishing rows.
///
/// `dec_start_a` / `dec_start_b`: column index of `col_anticipated_decision_start`
/// in each template.
/// `state_start_a` / `state_start_b`: column index of
/// `col_anticipated_state_start` in each template.
/// `n_ant`: number of anticipated plants (must be 2 for this swap).
/// `k_max`: ring-buffer slots per plant.
/// `fish_start_a` / `fish_start_b`: row index of `row_anticipated_fishing_start`.
/// `n_fish_rows`: number of active fishing rows at this stage (0..=n_ant).
///
/// Strategy: build the column-permutation `col_perm` such that
/// `tpl_a.column[col_perm[j]]` corresponds to `tpl_b.column[j]`, and the row
/// permutation `row_perm` analogously. Then assert that the permuted dense
/// LP (bounds, objective, full coefficient matrix) matches `tpl_b` bitwise.
///
/// Uses dense matrix expansion for clarity; the templates are tiny
/// (`num_cols ~ 20-50`, `num_rows ~ 10-20`) so the O(n^2) memory cost is fine.
#[allow(clippy::too_many_arguments)]
fn assert_lp_equivalence_after_anticipated_swap(
    tpl_a: &cobre_solver::StageTemplate,
    tpl_b: &cobre_solver::StageTemplate,
    dec_start_a: usize,
    dec_start_b: usize,
    state_start_a: usize,
    state_start_b: usize,
    slots_out_start_a: usize,
    slots_out_start_b: usize,
    n_ant: usize,
    k_max: usize,
    fish_start_a: usize,
    fish_start_b: usize,
    n_fish_rows: usize,
    def_row_start_a: usize,
    def_row_start_b: usize,
    n_def_rows: usize,
    slot_def_row_start_a: usize,
    slot_def_row_start_b: usize,
    slot_row_pos_a: &[Option<usize>],
    slot_row_pos_b: &[Option<usize>],
    stage_idx: usize,
) {
    assert_eq!(
        tpl_a.num_cols, tpl_b.num_cols,
        "stage {stage_idx}: num_cols"
    );
    assert_eq!(
        tpl_a.num_rows, tpl_b.num_rows,
        "stage {stage_idx}: num_rows"
    );
    assert_eq!(tpl_a.num_nz, tpl_b.num_nz, "stage {stage_idx}: num_nz");
    assert_eq!(n_ant, 2, "this helper requires n_ant == 2");

    // Build column permutation: `col_perm[j] = i` means tpl_a column `i`
    // corresponds to tpl_b column `j`. Identity outside the anticipated regions.
    let mut col_perm: Vec<usize> = (0..tpl_a.num_cols).collect();
    // Swap the two anticipated_decision columns.
    col_perm[dec_start_b] = dec_start_a + 1;
    col_perm[dec_start_b + 1] = dec_start_a;
    // Swap anticipated_state (incoming) columns at each ring-buffer slot.
    // Slot-major layout: column for slot `s`, plant `p` = state_start + s *
    // n_ant + p.
    for s in 0..k_max {
        col_perm[state_start_b + s * n_ant] = state_start_a + s * n_ant + 1;
        col_perm[state_start_b + s * n_ant + 1] = state_start_a + s * n_ant;
    }
    // Swap anticipated_slots_out (outgoing) columns at each ring-buffer slot.
    for s in 0..k_max {
        col_perm[slots_out_start_b + s * n_ant] = slots_out_start_a + s * n_ant + 1;
        col_perm[slots_out_start_b + s * n_ant + 1] = slots_out_start_a + s * n_ant;
    }

    // Build row permutation: identity outside anticipated_fishing and
    // anticipated_state_out_def rows. State fixing now uses column bounds
    // (no row equalities), so there are no state-fixing rows to permute.
    let mut row_perm: Vec<usize> = (0..tpl_a.num_rows).collect();
    if n_fish_rows == 2 {
        row_perm[fish_start_b] = fish_start_a + 1;
        row_perm[fish_start_b + 1] = fish_start_a;
    }
    // Under the always-active fishing predicate every anticipated plant
    // emits exactly one fishing row at every stage; this branch handles the
    // historical case of a partial active set still encountered in legacy
    // fixtures that pre-date the predicate flip. The SAME plant is active
    // in both LPs — but at LOCAL index 0 in one and
    // LOCAL index 1 in the other. The fishing-row index differs but corresponds
    // to the same plant's constraint. The mapping is still a single-row swap
    // when applicable.
    if n_fish_rows == 1 {
        // The single active fishing row in tpl_a corresponds to the single
        // active fishing row in tpl_b (same plant, different local index).
        row_perm[fish_start_b] = fish_start_a;
    }
    // Anticipated-state-out definition rows: one per active plant (strict gate).
    // When both plants are active, swap rows 0 and 1 (plant order changes).
    // When only one plant is active, the single def row maps identity-wise
    // (the active plant appears at local index 0 in both ctx_a and ctx_b).
    if n_def_rows == 2 {
        row_perm[def_row_start_b] = def_row_start_a + 1;
        row_perm[def_row_start_b + 1] = def_row_start_a;
    }
    if n_def_rows == 1 {
        row_perm[def_row_start_b] = def_row_start_a;
    }
    // Anticipated-ring interior-slot definition rows: the general per-(slot,
    // plant) analogue of the def-row swap above. `slot_row_pos_b`'s global
    // index `slot * n_ant + plant` corresponds to `slot_row_pos_a`'s
    // `slot * n_ant + (1 - plant)` (the same plant-swap the column loops
    // apply); both must agree on reachability (Some/None) since swapping
    // local labels never changes which physical (slot, plant) pair is
    // in-horizon.
    for (g_b, pos_b) in slot_row_pos_b.iter().enumerate() {
        let Some(pos_b) = *pos_b else { continue };
        let slot = g_b / n_ant;
        let plant = g_b % n_ant;
        let g_a = slot * n_ant + (1 - plant);
        let pos_a = slot_row_pos_a[g_a].unwrap_or_else(|| {
            panic!(
                "stage {stage_idx}: slot_row_pos_a[{g_a}] must be reachable to \
                 match slot_row_pos_b[{g_b}]"
            )
        });
        row_perm[slot_def_row_start_b + pos_b] = slot_def_row_start_a + pos_a;
    }

    // Dense bound/objective comparison: tpl_a[col_perm[j]] == tpl_b[j].
    for j in 0..tpl_a.num_cols {
        let a = col_perm[j];
        assert_eq!(
            tpl_a.col_lower[a].to_bits(),
            tpl_b.col_lower[j].to_bits(),
            "stage {stage_idx}: col_lower mismatch at permuted col {j} <- {a}"
        );
        assert_eq!(
            tpl_a.col_upper[a].to_bits(),
            tpl_b.col_upper[j].to_bits(),
            "stage {stage_idx}: col_upper mismatch at permuted col {j} <- {a}"
        );
        assert_eq!(
            tpl_a.objective[a].to_bits(),
            tpl_b.objective[j].to_bits(),
            "stage {stage_idx}: objective mismatch at permuted col {j} <- {a}"
        );
    }
    for i in 0..tpl_a.num_rows {
        let ra = row_perm[i];
        assert_eq!(
            tpl_a.row_lower[ra].to_bits(),
            tpl_b.row_lower[i].to_bits(),
            "stage {stage_idx}: row_lower mismatch at permuted row {i} <- {ra}"
        );
        assert_eq!(
            tpl_a.row_upper[ra].to_bits(),
            tpl_b.row_upper[i].to_bits(),
            "stage {stage_idx}: row_upper mismatch at permuted row {i} <- {ra}"
        );
    }

    // Dense matrix comparison: expand CSC to dense, apply permutation,
    // assert bit-equality. Tiny LPs (~50x20) so the O(n^2) cost is fine.
    let dense_a = csc_to_dense(tpl_a);
    let dense_b = csc_to_dense(tpl_b);
    for i in 0..tpl_a.num_rows {
        for j in 0..tpl_a.num_cols {
            let va = dense_a[row_perm[i]][col_perm[j]];
            let vb = dense_b[i][j];
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "stage {stage_idx}: coefficient mismatch at row {i} col {j} \
                     (permuted from row {} col {} in tpl_a)",
                row_perm[i],
                col_perm[j],
            );
        }
    }
}

/// Expand a CSC `StageTemplate` to a dense `Vec<Vec<f64>>`.
fn csc_to_dense(tpl: &cobre_solver::StageTemplate) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0_f64; tpl.num_cols]; tpl.num_rows];
    for j in 0..tpl.num_cols {
        let start = tpl.col_starts[j] as usize;
        let end = tpl.col_starts[j + 1] as usize;
        for k in start..end {
            let row = tpl.row_indices[k] as usize;
            dense[row][j] = tpl.values[k];
        }
    }
    dense
}

/// **Invariance probe** — direct LP-construction layer test.
///
/// Verifies that the LP templates produced by [`build_single_stage_template`]
/// are equivalent under a permutation of the `anticipated_thermal_indices` /
/// `anticipated_lead_stages` arrays.
///
/// ## Why this test exists
///
/// A full-`System` declaration-order test is a tautology: building two
/// `System`s with thermals declared in different orders proves only
/// **determinism** (same canonical input -> same output), because
/// `SystemBuilder::build()` sorts by `EntityId` so both present identical
/// canonical input downstream — not **invariance** (different declaration
/// orders -> same canonical result). That canonicalization is covered
/// directly by the `cobre-core` proptest
/// `build_canonical_order_invariant_under_input_permutation`.
///
/// The Cobre hard rule on declaration-order invariance requires bit-for-bit
/// identical results regardless of input entity ordering. This unit test
/// targets the **actual** code path that the canonical sort masks: it
/// directly constructs a `TemplateBuildCtx` with a permuted (yet internally
/// consistent) pair of `(anticipated_thermal_indices, anticipated_lead_stages)`
/// arrays and confirms that the resulting LP coefficients are equivalent
/// (modulo the expected swap of the anticipated-decision columns,
/// anticipated-state ring-buffer columns/rows, and anticipated-fishing rows).
///
/// ## Method
///
/// 1. Build a system with two anticipated thermals (K=2 and K=3) plus one
///    standard backup thermal, with **distinct** per-thermal stage costs
///    and bounds (uniform defaults would trivially pass).
/// 2. Call `build_template_build_ctx` to obtain `ctx_a` with the canonical
///    ordering `anticipated_thermal_indices = [0, 1]`,
///    `anticipated_lead_stages = [2, 3]`.
/// 3. Manually construct `ctx_b` by swapping both arrays in lockstep:
///    `anticipated_thermal_indices = [1, 0]`,
///    `anticipated_lead_stages = [3, 2]`.
/// 4. Build single-stage templates for both contexts at stages 0, 2, and 3.
///    Under the always-active fishing predicate every anticipated plant
///    emits one fishing row at every stage, so all sampled stages carry
///    the same number of fishing rows; the anticipated-decision active set
///    is independent of the fishing predicate and still depends on
///    `t + K_i < T` at each stage.
/// 5. Assert LP equivalence under the canonical swap permutation
///    (column swap on anticipated_decision and slot-major state, row swap
///    on state-fixing and fishing rows when both plants are present).
#[test]
fn lp_template_invariant_under_anticipated_index_permutation() {
    let system = anticipated_invariance_system();
    // Canonical sort places thermals as [id=1, id=2, id=3].
    assert_eq!(system.thermals().len(), 3);
    assert_eq!(system.thermals()[0].id.0, 1);
    assert_eq!(system.thermals()[1].id.0, 2);
    assert_eq!(system.thermals()[2].id.0, 3);

    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    };

    let (ctx_a, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    );

    // Sanity: ctx_a uses canonical ordering.
    assert_eq!(ctx_a.n_anticipated, 2);
    assert_eq!(ctx_a.k_max, 3);
    assert_eq!(ctx_a.anticipated_thermal_indices, vec![0, 1]);
    assert_eq!(ctx_a.anticipated_lead_stages, vec![2, 3]);

    // Construct ctx_b: a clone of ctx_a with the two anticipated arrays
    // swapped in lockstep. Both arrays must be permuted by the same
    // permutation to preserve the (thermal_idx, K_i) pairing.
    let ctx_b = super::super::layout::TemplateBuildCtx {
        hydros: ctx_a.hydros,
        thermals: ctx_a.thermals,
        lines: ctx_a.lines,
        buses: ctx_a.buses,
        load_models: ctx_a.load_models,
        cascade: ctx_a.cascade,
        resolved: super::super::layout::ResolvedTables {
            bounds: ctx_a.resolved.bounds,
            penalties: ctx_a.resolved.penalties,
            resolved_generic_bounds: ctx_a.resolved.resolved_generic_bounds,
            resolved_load_factors: ctx_a.resolved.resolved_load_factors,
            resolved_exchange_factors: ctx_a.resolved.resolved_exchange_factors,
            resolved_ncs_bounds: ctx_a.resolved.resolved_ncs_bounds,
            resolved_ncs_factors: ctx_a.resolved.resolved_ncs_factors,
            resolved_parameters: ctx_a.resolved.resolved_parameters,
        },
        hydro_pos: ctx_a.hydro_pos.clone(),
        thermal_pos: ctx_a.thermal_pos.clone(),
        line_pos: ctx_a.line_pos.clone(),
        bus_pos: ctx_a.bus_pos.clone(),
        par_lp: ctx_a.par_lp,
        production_models: ctx_a.production_models,
        evaporation_models: ctx_a.evaporation_models,
        generic_constraints: ctx_a.generic_constraints,
        non_controllable_sources: ctx_a.non_controllable_sources,
        pumping_stations: ctx_a.pumping_stations,
        pumping_pos: ctx_a.pumping_pos.clone(),
        n_pumping: ctx_a.n_pumping,
        contracts: ctx_a.contracts,
        contract_pos: ctx_a.contract_pos.clone(),
        n_contract_import: ctx_a.n_contract_import,
        n_contract_export: ctx_a.n_contract_export,
        diversion_upstream: ctx_a.diversion_upstream.clone(),
        n_hydros: ctx_a.n_hydros,
        n_thermals: ctx_a.n_thermals,
        n_lines: ctx_a.n_lines,
        n_buses: ctx_a.n_buses,
        max_par_order: ctx_a.max_par_order,
        n_anticipated: ctx_a.n_anticipated,
        k_max: ctx_a.k_max,
        // The swap: lockstep permutation [0,1] -> [1,0] on both arrays.
        anticipated_lead_stages: vec![
            ctx_a.anticipated_lead_stages[1],
            ctx_a.anticipated_lead_stages[0],
        ],
        anticipated_thermal_indices: vec![
            ctx_a.anticipated_thermal_indices[1],
            ctx_a.anticipated_thermal_indices[0],
        ],
        // Swap the windows in lockstep with the index/lead permutation so the
        // per-plant window stays aligned with its anticipated-local position.
        anticipated_windows: vec![ctx_a.anticipated_windows[1], ctx_a.anticipated_windows[0]],
        study_stage_ids: ctx_a.study_stage_ids.clone(),
        has_penalty: ctx_a.has_penalty,
        cumulative_discount_factors: ctx_a.cumulative_discount_factors.clone(),
        total_hours_per_stage: ctx_a.total_hours_per_stage.clone(),
        filling_v_target: ctx_a.filling_v_target.clone(),
        arc_spread_k: ctx_a.arc_spread_k.clone(),
        arc_spread_chrono: ctx_a.arc_spread_chrono.clone(),
        per_stage_mask: ctx_a.per_stage_mask.clone(),
    };

    // Sanity: ctx_b really has the swapped ordering.
    assert_eq!(ctx_b.anticipated_thermal_indices, vec![1, 0]);
    assert_eq!(ctx_b.anticipated_lead_stages, vec![3, 2]);

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    // Test multiple stages to cover the active-decision boundary while
    // the always-active fishing predicate keeps the fishing-row count
    // constant at every stage (one row per anticipated plant).
    for stage_idx in [0_usize, 2, 3] {
        let stage = study_stages[stage_idx];

        let state_a = state_layout_for(&ctx_a);
        let state_b = state_layout_for(&ctx_b);

        let tpl_a = super::build_single_stage_template(&ctx_a, &state_a, stage, stage_idx).template;
        let tpl_b = super::build_single_stage_template(&ctx_b, &state_b, stage, stage_idx).template;

        // Reconstruct the layout for tpl_a / tpl_b to find the
        // anticipated_decision / anticipated_state / fishing row offsets.
        // Both templates use the same num_cols/num_rows (the layout depends
        // only on n_anticipated and k_max, both unchanged by the swap).
        let layout_a = super::super::layout::StageLayout::new(&ctx_a, &state_a, stage, stage_idx);
        let layout_b = super::super::layout::StageLayout::new(&ctx_b, &state_b, stage, stage_idx);

        // Layout offsets must be identical (they depend only on counts,
        // not on the contents of the anticipated arrays).
        assert_eq!(
            layout_a.anticipated.col_anticipated_decision_start,
            layout_b.anticipated.col_anticipated_decision_start,
            "stage {stage_idx}: dec_start"
        );
        assert_eq!(
            layout_a.col_anticipated_state_start(),
            layout_b.col_anticipated_state_start(),
            "stage {stage_idx}: state_start"
        );
        assert_eq!(
            layout_a.anticipated.row_anticipated_fishing_start,
            layout_b.anticipated.row_anticipated_fishing_start,
            "stage {stage_idx}: fish_start"
        );
        assert_eq!(
            layout_a.anticipated.n_anticipated_fishing_rows,
            layout_b.anticipated.n_anticipated_fishing_rows,
            "stage {stage_idx}: n_fish_rows"
        );

        assert_lp_equivalence_after_anticipated_swap(
            &tpl_a,
            &tpl_b,
            layout_a.anticipated.col_anticipated_decision_start,
            layout_b.anticipated.col_anticipated_decision_start,
            layout_a.col_anticipated_state_start(),
            layout_b.col_anticipated_state_start(),
            layout_a.anticipated.col_anticipated_slots_out_start,
            layout_b.anticipated.col_anticipated_slots_out_start,
            ctx_a.n_anticipated,
            ctx_a.k_max,
            layout_a.anticipated.row_anticipated_fishing_start,
            layout_b.anticipated.row_anticipated_fishing_start,
            layout_a.anticipated.n_anticipated_fishing_rows,
            layout_a.anticipated.row_anticipated_state_out_def_start,
            layout_b.anticipated.row_anticipated_state_out_def_start,
            layout_a.anticipated.n_anticipated_state_out_def_rows,
            layout_a.anticipated.row_anticipated_slot_definition_start,
            layout_b.anticipated.row_anticipated_slot_definition_start,
            &layout_a.anticipated.anticipated_slot_row_pos,
            &layout_b.anticipated.anticipated_slot_row_pos,
            stage_idx,
        );
    }
}

// ── StageTemplates::empty ──────────────────────────────────────────────────

/// `StageTemplates::empty(n)` yields every per-stage collection empty and
/// records `n_hydros == n`. This pins the all-empty shape the empty-study
/// early return relies on.
#[test]
fn stage_templates_empty_is_all_empty_with_n_hydros() {
    let n = 7_usize;
    let empty = super::StageTemplates::empty(n);

    assert_eq!(empty.n_hydros, n, "empty(n).n_hydros must equal n");
    assert_eq!(empty.n_load_buses, 0, "n_load_buses must be 0");

    assert!(empty.templates.is_empty(), "templates");
    assert!(empty.base_rows.is_empty(), "base_rows");
    assert!(empty.noise_scale.is_empty(), "noise_scale");
    assert!(empty.zeta_per_stage.is_empty(), "zeta_per_stage");
    assert!(
        empty.block_hours_per_stage.is_empty(),
        "block_hours_per_stage"
    );
    assert!(
        empty.load_balance_row_starts.is_empty(),
        "load_balance_row_starts"
    );
    assert!(empty.load_bus_indices.is_empty(), "load_bus_indices");
    assert!(
        empty.generic_constraint_row_entries.is_empty(),
        "generic_constraint_row_entries"
    );
    assert!(empty.ncs_col_starts.is_empty(), "ncs_col_starts");
    assert_eq!(empty.n_ncs, 0, "n_ncs");
    assert!(empty.pumping_col_starts.is_empty(), "pumping_col_starts");
    assert_eq!(empty.n_pumping, 0, "n_pumping");
    assert!(empty.diversion_upstream.is_empty(), "diversion_upstream");
    assert!(
        empty.hydro_productivities_per_stage.is_empty(),
        "hydro_productivities_per_stage"
    );
    assert!(empty.discount_factors().is_empty(), "discount_factors");
    assert!(
        empty.cumulative_discount_factors().is_empty(),
        "cumulative_discount_factors"
    );
}

// ── discount-factor placeholder is replaced by the public path ─────────────

/// Build a 3-stage thermals-only system carrying a non-zero global annual
/// discount rate. Empty `transitions` means every stage falls back to the
/// global rate, so the postprocessed per-stage factors are all < 1.0 and the
/// cumulative vector compounds below the 1.0 placeholder.
fn discounted_multi_stage_system() -> cobre_core::System {
    use cobre_core::{PolicyGraph, PolicyGraphType};

    let n_stages = 3_usize;
    let thermals = vec![Thermal {
        id: EntityId(1),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 10.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        anticipated_config: None,
    }];
    let n_thermals = thermals.len();

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

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
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
        .map(|s| LoadModel {
            bus_id: EntityId(1),
            stage_id: s as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 10.0,
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

    // Non-zero global rate with no per-transition overrides: every stage
    // discounts at the global rate.
    let policy_graph = PolicyGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.10,
        transitions: Vec::new(),
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .policy_graph(policy_graph)
        .build()
        .expect("discounted_multi_stage_system: valid system")
}

/// Any `StageTemplates` produced by the public build + postprocess path has
/// `cumulative_discount_factors().len() == templates.len()` and is no longer
/// the all-`1.0` placeholder once a non-zero discount rate is in effect.
///
/// The discount fields are private, so an external caller can only ever observe
/// the postprocessed (discounted) values, never the placeholder
/// `build_stage_templates` leaves behind.
#[test]
fn postprocessed_stage_templates_carry_discounted_factors() {
    let system = discounted_multi_stage_system();
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let mut templates = super::build_stage_templates(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");

    // Run the postprocess step that installs the real discount factors —
    // exactly the public path StudySetup drives.
    let _report =
        crate::setup::template_postprocess::postprocess_templates(&mut templates, &system);

    let cumulative = templates.cumulative_discount_factors();
    assert_eq!(
        cumulative.len(),
        templates.templates.len(),
        "cumulative_discount_factors length must equal templates.len() after postprocess"
    );
    assert_eq!(
        cumulative[0], 1.0,
        "cumulative_discount_factors[0] is the present value (1.0)"
    );
    // With a 10% annual rate the later cumulative factors compound strictly
    // below the 1.0 placeholder, so the postprocessed vector is provably not
    // the placeholder the builder hands back.
    assert!(
        cumulative.iter().any(|&d| d < 1.0),
        "postprocessed cumulative factors must drop below the 1.0 placeholder, got {cumulative:?}"
    );
    assert!(
        cumulative[cumulative.len() - 1] < 1.0,
        "the final cumulative factor must be discounted below 1.0, got {}",
        cumulative[cumulative.len() - 1]
    );
}

// ── Operational-violation RHS & matrix-coefficient verification ──────────
//
// These verify the LP-builder's row bounds (RHS) and CSC matrix coefficients
// at the four operational-violation constraint-row families. They locate the
// rows via `StageLayout`'s op-violation row ranges directly — the row ranges
// the per-stage layout owns and the public `StageGeometry` does not expose —
// so this is the correct layer for them. The RHS/coefficient values are
// produced by `fill_operational_violation_rows` / `fill_operational_violation_entries`
// through `build_single_stage_template`.

use super::super::layout::StageLayout;
use super::COST_SCALE_FACTOR;
use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use cobre_core::System;
use cobre_solver::StageTemplate;

/// One-hydro system with all operational-violation bounds active (min/max
/// outflow, min turbine, min generation > 0), two blocks per stage, and
/// `1000.0` violation penalties — the fixture the operational-violation
/// builder tests exercise.
fn one_hydro_active_violations(n_stages: usize) -> System {
    use cobre_core::scenario::{InflowModel, LoadModel};

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

    let hydro = Hydro {
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
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

/// Get CSC entries for column `col` of a built `StageTemplate` as
/// `(row, value)` pairs.
#[allow(clippy::cast_sign_loss)]
fn csc_entries_for_col(t: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    (start..end)
        .map(|nz| (t.row_indices[nz] as usize, t.values[nz]))
        .collect()
}

/// Build the active-violations stage-0 `StageLayout` (the owner of the
/// op-violation row/column ranges) and the matching `StageTemplate` (RHS,
/// bounds, objective, CSC) from one shared `TemplateBuildCtx`, so the row
/// ranges and the template the tests query agree by construction.
///
/// Productivity is `0.5` so the per-block min-generation row carries a
/// `0.5` turbine coefficient (asserted by
/// [`relocated_min_generation_constant_productivity_coefficients`]).
fn build_active_violations_layout_and_template() -> (StageLayout<'static>, StageTemplate) {
    let system = Box::leak(Box::new(one_hydro_active_violations(1)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::ConstantProductivity {
            productivity: 0.5,
        }]],
        1,
        1,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    }));

    let (ctx, _, _) = super::build_template_build_ctx(
        system,
        InflowNonNegativityMethod::None,
        par_lp,
        production,
        &hydro_models.evaporation,
        resolved_params,
    );
    let ctx = Box::leak(Box::new(ctx));
    let state = Box::leak(Box::new(state_layout_for(ctx)));
    let stage = &system.stages()[0];

    // `build_single_stage_template` and `StageLayout::new` are deterministic
    // functions of the same `(ctx, state, stage, 0)`, so the template and the
    // layout agree on every row/column offset.
    let template = super::build_single_stage_template(ctx, state, stage, 0).template;
    let layout = StageLayout::new(ctx, state, stage, 0);
    (layout, template)
}

#[test]
fn relocated_operational_violation_row_counts() {
    // 1 hydro, 2 blocks => 4 operational violation rows of length 2 each.
    let (layout, t) = build_active_violations_layout_and_template();

    // 4 row ranges each contain n_hydros * n_blks = 1 * 2 = 2 rows.
    assert_eq!(layout.min_outflow_rows.len(), 2);
    assert_eq!(layout.max_outflow_rows.len(), 2);
    assert_eq!(layout.min_turbine_rows.len(), 2);
    assert_eq!(layout.min_generation_rows.len(), 2);

    // All constraint rows are within the template's range.
    assert!(
        layout.min_generation_rows.end <= t.num_rows,
        "operational violation rows exceed num_rows"
    );
}

#[test]
fn relocated_min_outflow_row_bounds() {
    // Per-block: RHS in rate units (m3/s), not volume.
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 50.0; // min_outflow_m3s

    // Both blocks get the same RHS.
    for blk in 0..2 {
        let row = layout.min_outflow_rows.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_outflow row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_outflow row_upper must be +inf"
        );
    }
}

#[test]
fn relocated_max_outflow_row_bounds() {
    // Per-block: RHS in rate units (m3/s).
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_upper = 800.0; // max_outflow_m3s

    for blk in 0..2 {
        let row = layout.max_outflow_rows.start + blk;
        assert_eq!(
            t.row_lower[row],
            f64::NEG_INFINITY,
            "max_outflow row_lower must be -inf"
        );
        assert!(
            (t.row_upper[row] - expected_upper).abs() < 1e-10,
            "max_outflow row_upper (block {blk}) = {}, expected {}",
            t.row_upper[row],
            expected_upper
        );
    }
}

#[test]
fn relocated_min_turbine_row_bounds() {
    // Per-block: RHS in rate units (m3/s).
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 10.0; // min_turbined_m3s

    for blk in 0..2 {
        let row = layout.min_turbine_rows.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_turbine row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_turbine row_upper must be +inf"
        );
    }
}

#[test]
fn relocated_min_generation_row_bounds() {
    // Per-block: RHS in rate units (MW), not MWh.
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 5.0; // min_generation_mw

    for blk in 0..2 {
        let row = layout.min_generation_rows.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_generation row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_generation row_upper must be +inf"
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn relocated_min_outflow_matrix_coefficients() {
    // Per-block min outflow: q + s + d + slack = 1.0 per block-row.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.min_outflow_rows.start + blk;

        // Turbine column for this block: coefficient 1.0
        let entries = csc_entries_for_col(&t, layout.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine blk{blk} entry for min_outflow row: {v:?}"
        );

        // Spillage column for this block: coefficient 1.0
        let entries = csc_entries_for_col(&t, layout.spillage.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "spillage blk{blk} entry for min_outflow row: {v:?}"
        );

        // Slack column for this block: coefficient 1.0
        let entries = csc_entries_for_col(&t, layout.outflow_below_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "outflow_below slack blk{blk}: {v:?}"
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn relocated_max_outflow_matrix_slack_is_negative() {
    // Per-block max outflow row: slack coefficient = -1.0.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.max_outflow_rows.start + blk;
        let entries = csc_entries_for_col(&t, layout.outflow_above_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - (-1.0)).abs() < 1e-15,
            "outflow_above slack blk{blk} must be -1.0, got {v:?}"
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn relocated_min_turbine_matrix_only_turbine_cols() {
    // Per-block min turbine: only turbine columns (no spillage), coefficient 1.0.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.min_turbine_rows.start + blk;

        // Turbine column: coefficient 1.0
        let entries = csc_entries_for_col(&t, layout.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine blk{blk} min_turbine: {v:?}"
        );

        // Spillage should NOT appear in min_turbine row.
        let entries_spill = csc_entries_for_col(&t, layout.spillage.start + blk);
        let v_spill = entries_spill.iter().find(|e| e.0 == row);
        assert!(
            v_spill.is_none(),
            "spillage should not appear in min_turbine row (blk {blk})"
        );

        // Slack = +1.0
        let entries = csc_entries_for_col(&t, layout.turbine_below_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine_below slack blk{blk}: {v:?}"
        );
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn relocated_min_generation_constant_productivity_coefficients() {
    // Per-block constant productivity: coefficient = rho = 0.5 per block-row.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;
    let rho = 0.5;

    for blk in 0..n_blks {
        let row = layout.min_generation_rows.start + blk;

        // Turbine column: coefficient = rho
        let entries = csc_entries_for_col(&t, layout.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - rho).abs() < 1e-10,
            "turbine blk{blk} min_gen coeff: {v:?}, expected {rho}"
        );

        // Slack: +1.0
        let entries_s = csc_entries_for_col(&t, layout.generation_below_slack.start + blk);
        let vs = entries_s.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            vs.is_some() && (vs.unwrap() - 1.0).abs() < 1e-15,
            "generation_below slack blk{blk}: {vs:?}"
        );
    }
}

#[test]
fn relocated_operational_violation_rows_outside_dual_relevant() {
    // n_dual_relevant is always 0 (state pinning uses column bounds). All
    // operational violation rows must be placed beyond this range.
    let (layout, t) = build_active_violations_layout_and_template();

    assert_eq!(
        t.n_dual_relevant, 0,
        "n_dual_relevant is always 0 with column-bound state pinning"
    );

    // All 4 operational violation row ranges must start beyond n_dual_relevant.
    assert!(
        layout.min_outflow_rows.start > t.n_dual_relevant,
        "min_outflow row {} must be > n_dual_relevant {}",
        layout.min_outflow_rows.start,
        t.n_dual_relevant
    );
    assert!(
        layout.max_outflow_rows.start > t.n_dual_relevant,
        "max_outflow row {} must be > n_dual_relevant {}",
        layout.max_outflow_rows.start,
        t.n_dual_relevant
    );
    assert!(
        layout.min_turbine_rows.start > t.n_dual_relevant,
        "min_turbine row {} must be > n_dual_relevant {}",
        layout.min_turbine_rows.start,
        t.n_dual_relevant
    );
    assert!(
        layout.min_generation_rows.start > t.n_dual_relevant,
        "min_generation row {} must be > n_dual_relevant {}",
        layout.min_generation_rows.start,
        t.n_dual_relevant
    );
}

#[test]
fn relocated_diagnostic_template_operational_violation_correctness() {
    let (layout, t) = build_active_violations_layout_and_template();

    // Operational-violation presence: the non-empty min-outflow slack range
    // is the evidence the families are present when hydros exist.
    assert!(
        !layout.outflow_below_slack.is_empty(),
        "operational-violation slack columns must be present when hydros exist"
    );

    // Per-block formulation: RHS is in rate units (m3/s or MW), not volume/energy.
    // Block 0 column at `.start`, block 1 at `.start + 1`.
    let block_hours_0 = 720.0;

    // Min outflow row (block 0): row_lower = 50.0 m3/s
    let row = layout.min_outflow_rows.start;
    assert!(
        (t.row_lower[row] - 50.0).abs() < 1e-10,
        "min_outflow row_lower = {}, expected 50.0 (rate units m3/s)",
        t.row_lower[row],
    );
    assert_eq!(
        t.row_upper[row],
        f64::INFINITY,
        "min_outflow row_upper must be +inf for >= constraint"
    );

    // Column bounds: outflow_below_slack block 0.
    let col = layout.outflow_below_slack.start;
    assert_eq!(
        t.col_lower[col], 0.0,
        "outflow_below_slack col_lower must be 0"
    );
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_below_slack col_upper must be +inf when min_outflow > 0"
    );

    // Objective: penalty * block_hours (block 0).
    let expected_objective = 1000.0 * block_hours_0 / COST_SCALE_FACTOR;
    assert!(
        t.objective[col] > 0.0,
        "outflow_below_slack objective must be positive (penalty), got {}",
        t.objective[col]
    );
    assert!(
        (t.objective[col] - expected_objective).abs() < 1e-10,
        "outflow_below_slack objective = {}, expected {} (= 1000 * {} / {})",
        t.objective[col],
        expected_objective,
        block_hours_0,
        COST_SCALE_FACTOR
    );

    let col_above = layout.outflow_above_slack.start;
    assert_eq!(t.col_upper[col_above], f64::INFINITY);
    assert!(t.objective[col_above] > 0.0);

    let col_turb = layout.turbine_below_slack.start;
    assert_eq!(t.col_upper[col_turb], f64::INFINITY);
    assert!(t.objective[col_turb] > 0.0);

    let col_gen = layout.generation_below_slack.start;
    assert_eq!(t.col_upper[col_gen], f64::INFINITY);
    assert!(t.objective[col_gen] > 0.0);

    // Min turbine row (block 0): row_lower = 10.0 m3/s
    let min_turb_row = layout.min_turbine_rows.start;
    assert!(
        (t.row_lower[min_turb_row] - 10.0).abs() < 1e-10,
        "min_turbine row_lower = {}, expected 10.0 (rate units m3/s)",
        t.row_lower[min_turb_row],
    );

    // Min generation row (block 0): row_lower = 5.0 MW
    let min_gen_row = layout.min_generation_rows.start;
    assert!(
        (t.row_lower[min_gen_row] - 5.0).abs() < 1e-10,
        "min_generation row_lower = {}, expected 5.0 (rate units MW)",
        t.row_lower[min_gen_row],
    );

    // Max outflow row (block 0): row_upper = 800.0 m3/s
    let max_outflow_row = layout.max_outflow_rows.start;
    assert!(
        (t.row_upper[max_outflow_row] - 800.0).abs() < 1e-10,
        "max_outflow row_upper = {}, expected 800.0 (rate units m3/s)",
        t.row_upper[max_outflow_row],
    );
}

// ── build_filling_v_target backward fold ─────────────────────────────────

use cobre_core::FillingConfig;
use std::collections::BTreeMap as VTargetMap;

/// A single non-cascade hydro carrying a `FillingConfig`
/// (`start_stage_id`/`entry_stage_id`), used by the `build_filling_v_target`
/// fold tests. All other fields are inert.
fn vtarget_filling_hydro(id: i32, start: i32, entry: i32) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: Some(entry),
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 50.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 45.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: Some(FillingConfig {
            start_stage_id: start,
            filling_min_rate_m3s: 0.0,
        }),
        penalties: hydro_penalties_zero(),
    }
}

/// A `ResolvedBounds` table for one hydro across `n_stages` stages, with every
/// stage's `min_storage_hm3` and `filling_min_rate_m3s` set to the given values.
fn vtarget_bounds(n_stages: usize, min_storage: f64, rate: f64) -> ResolvedBounds {
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
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
    for stage_idx in 0..n_stages {
        let hb = bounds.hydro_bounds_mut(0, stage_idx);
        hb.min_storage_hm3 = min_storage;
        hb.filling_min_rate_m3s = rate;
    }
    bounds
}

/// Identity `stage_id → stage_idx` map for `n_stages` study stages.
fn vtarget_id_map(n_stages: usize) -> VTargetMap<i32, usize> {
    (0..n_stages).map(|i| (i as i32, i)).collect()
}

/// The AC: `start = 2`, `entry = 4`, `min_storage = 60`, per-stage ζ = 2.592
/// (`total_hours = 720`, `M3S_TO_HM3 = 0.0036`), `rate = 5`. The backward fold
/// pins `V_target[3] = 60` (the dead-volume anchor at L = entry − 1) and
/// `V_target[2] = 60 − 2.592·5 = 47.04` (one stage of minimum accumulation
/// below the anchor). No `V_target` is emitted at PreFilling (ids 0, 1) or
/// Operating (id ≥ 4).
#[test]
fn build_filling_v_target_backward_fold_ac_values() {
    let n_stages = 5;
    let hydros = vec![vtarget_filling_hydro(1, 2, 4)];
    let bounds = vtarget_bounds(n_stages, 60.0, 5.0);
    // ζ_t = total_hours[t]·M3S_TO_HM3 = 720·0.0036 = 2.592 at every stage.
    let total_hours = vec![720.0; n_stages];
    let v_target =
        super::build_filling_v_target(&hydros, &bounds, &total_hours, &vtarget_id_map(n_stages));

    // L = entry − 1 = 3: anchored at the dead volume.
    assert!(
        (v_target[&(0, 3)] - 60.0).abs() < 1e-9,
        "V_target[3] == min_storage == 60.0, got {}",
        v_target[&(0, 3)]
    );
    // Early Filling stage 2: 60 − 2.592·5 = 47.04.
    assert!(
        (v_target[&(0, 2)] - 47.04).abs() < 1e-9,
        "V_target[2] == 60 − 2.592·5 == 47.04, got {}",
        v_target[&(0, 2)]
    );
    // No entry outside the Filling window {2, 3}.
    assert!(
        !v_target.contains_key(&(0, 0)),
        "no V_target at PreFilling id 0"
    );
    assert!(
        !v_target.contains_key(&(0, 1)),
        "no V_target at PreFilling id 1"
    );
    assert!(
        !v_target.contains_key(&(0, 4)),
        "no V_target at Operating id 4"
    );
    // Exactly the two Filling-stage entries.
    assert_eq!(v_target.len(), 2, "exactly one V_target per Filling stage");
}

/// Over-provisioned schedule: a fill rate large enough that the backward fold's
/// unclipped value would exceed `min_storage` is impossible (non-negative rate
/// only lowers it); the contract is that EVERY `V_target[t] ≤ min_storage`. With
/// a wide Filling window (ids 1..=5, entry = 6) and a high rate, every earliest
/// floor sits strictly below the dead volume and the clip never raises one above
/// it. The clip is verified to hold at every Filling stage.
#[test]
fn build_filling_v_target_clips_at_min_storage_when_over_provisioned() {
    let n_stages = 7;
    let min_storage = 30.0;
    let hydros = vec![vtarget_filling_hydro(1, 1, 6)]; // Filling ids {1,2,3,4,5}.
    // A high rate (50 m³/s over ζ = 2.592 ⇒ 129.6 hm³/stage) far exceeds the
    // 30 hm³ dead volume, so the unclipped earliest floors go deeply negative.
    let bounds = vtarget_bounds(n_stages, min_storage, 50.0);
    let total_hours = vec![720.0; n_stages];
    let v_target =
        super::build_filling_v_target(&hydros, &bounds, &total_hours, &vtarget_id_map(n_stages));

    for stage_id in 1..=5 {
        let v = v_target[&(0, stage_id)];
        assert!(
            v <= min_storage + 1e-12,
            "V_target[{stage_id}] = {v} must not exceed the dead volume {min_storage}"
        );
    }
    // The anchor (last Filling stage, id 5) is exactly the dead volume.
    assert!(
        (v_target[&(0, 5)] - min_storage).abs() < 1e-9,
        "V_target[L] == min_storage (the clip is a no-op at the anchor)"
    );
    // The earliest floor is strictly below (deep accumulation requirement).
    assert!(
        v_target[&(0, 1)] < v_target[&(0, 5)],
        "earliest floor strictly below the anchor"
    );
}

/// A zero fill rate makes the trajectory FLAT: every Filling stage's floor
/// equals `min_storage` (the design's `rate == 0 ⇒ V_target[t] == V_target[t+1]`
/// degenerate case). The clip is a no-op throughout.
#[test]
fn build_filling_v_target_flat_when_rate_is_zero() {
    let n_stages = 5;
    let hydros = vec![vtarget_filling_hydro(1, 1, 4)]; // Filling ids {1,2,3}.
    let bounds = vtarget_bounds(n_stages, 45.0, 0.0);
    let total_hours = vec![720.0; n_stages];
    let v_target =
        super::build_filling_v_target(&hydros, &bounds, &total_hours, &vtarget_id_map(n_stages));
    for stage_id in 1..=3 {
        assert!(
            (v_target[&(0, stage_id)] - 45.0).abs() < 1e-9,
            "flat trajectory: V_target[{stage_id}] == min_storage == 45.0"
        );
    }
}

/// A non-filling hydro (no `FillingConfig`) yields an EMPTY map — the
/// parity-neutrality contract for the precompute itself.
#[test]
fn build_filling_v_target_empty_for_non_filling() {
    let n_stages = 3;
    let mut h = vtarget_filling_hydro(1, 1, 2);
    h.filling = None;
    h.entry_stage_id = None;
    let hydros = vec![h];
    let bounds = vtarget_bounds(n_stages, 50.0, 5.0);
    let total_hours = vec![720.0; n_stages];
    let v_target =
        super::build_filling_v_target(&hydros, &bounds, &total_hours, &vtarget_id_map(n_stages));
    assert!(
        v_target.is_empty(),
        "non-filling hydro ⇒ empty V_target map"
    );
}

/// Assert two `StageTemplate`s are byte-identical: CSC structure
/// (`col_starts`, `row_indices`, `values`), bounds (`col_lower`/`col_upper`,
/// `row_lower`/`row_upper`), `objective`, and the full dense matrix — every
/// `f64` compared by `to_bits()` so it is true bit-identity, not approximate.
fn assert_templates_byte_identical(tpl_a: &StageTemplate, tpl_b: &StageTemplate) {
    assert_eq!(tpl_a.num_cols, tpl_b.num_cols, "num_cols");
    assert_eq!(tpl_a.num_rows, tpl_b.num_rows, "num_rows");
    assert_eq!(tpl_a.num_nz, tpl_b.num_nz, "num_nz");
    assert_eq!(tpl_a.n_state, tpl_b.n_state, "n_state");

    assert_eq!(tpl_a.col_starts, tpl_b.col_starts, "col_starts");
    assert_eq!(tpl_a.row_indices, tpl_b.row_indices, "row_indices");

    let bits = |xs: &[f64]| xs.iter().map(|v| v.to_bits()).collect::<Vec<u64>>();
    assert_eq!(bits(&tpl_a.values), bits(&tpl_b.values), "values");
    assert_eq!(bits(&tpl_a.col_lower), bits(&tpl_b.col_lower), "col_lower");
    assert_eq!(bits(&tpl_a.col_upper), bits(&tpl_b.col_upper), "col_upper");
    assert_eq!(bits(&tpl_a.objective), bits(&tpl_b.objective), "objective");
    assert_eq!(bits(&tpl_a.row_lower), bits(&tpl_b.row_lower), "row_lower");
    assert_eq!(bits(&tpl_a.row_upper), bits(&tpl_b.row_upper), "row_upper");

    let dense_a = csc_to_dense(tpl_a);
    let dense_b = csc_to_dense(tpl_b);
    for i in 0..tpl_a.num_rows {
        for j in 0..tpl_a.num_cols {
            assert_eq!(
                dense_a[i][j].to_bits(),
                dense_b[i][j].to_bits(),
                "dense coefficient mismatch at row {i} col {j}"
            );
        }
    }
}

/// One-bus, one-hydro FPHA system whose single stage carries `n_blks` blocks
/// under `block_mode`. The FPHA generation rows put the average-storage `γᵥ/2`
/// coefficient on both the incoming and outgoing storage columns, so the
/// byte-identity check actually exercises the storage-bearing rows.
fn one_hydro_block_system(block_mode: BlockMode, n_blks: usize) -> System {
    use cobre_core::scenario::{InflowModel, LoadModel};

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

    let hydro = fixture_hydro(2);

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 300.0 + 100.0 * f64::from(u32::try_from(b).unwrap_or(0)),
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks,
        block_mode,
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
        mean_m3s: 80.0,
        std_m3s: 20.0,
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

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_block_system: valid")
}

/// Build the full stage-0 `StageTemplate` for the one-hydro FPHA study under
/// `block_mode` with `n_blks` blocks, using a single FPHA plane so the
/// generation row carries the average-storage anchor.
fn block_template(block_mode: BlockMode, n_blks: usize) -> StageTemplate {
    use crate::hydro_models::FphaPlane;

    let system = one_hydro_block_system(block_mode, n_blks);
    let par_lp = PrecomputedPar::default();
    let production = ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 1.0,
                gamma_v: 0.2,
                gamma_q: 0.5,
                gamma_s: 0.05,
            }],
        }]],
        1,
        1,
    );
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let resolved_params = ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    };

    let (ctx, _, _) = super::build_template_build_ctx(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &production,
        &hydro_models.evaporation,
        &resolved_params,
    );
    let state = state_layout_for(&ctx);
    let stage = &system.stages()[0];
    super::build_single_stage_template(&ctx, &state, stage, 0).template
}

/// `K = 1` chronological build collapses to the parallel LP: the
/// `storage_internal` interior-column family is empty, there is one water row,
/// and FPHA rides the single incoming/outgoing storage pair — so the two
/// templates are byte-identical (§9 contract). This anchors the layout half of
/// the chronological feature against any regression that perturbs the `K = 1`
/// column/row/value layout.
#[test]
fn chronological_k1_byte_identical_to_parallel() {
    let parallel = block_template(BlockMode::Parallel, 1);
    let chronological = block_template(BlockMode::Chronological, 1);
    assert_templates_byte_identical(&parallel, &chronological);
}

/// `theta` and `n_state` are pure functions of `(N, L, A, k_max)` and are
/// `n_blks`-free by construction (`StateLayout::new` never sees `block_mode` or
/// `n_blks`): per-block storage lives strictly in the control region, never in
/// the state region (§2). Building a chronological `K ≥ 2` stage therefore
/// changes neither — only the control-region column count grows, by exactly
/// `n_h * (n_blks − 1)` interior storage columns.
#[test]
fn theta_and_n_state_invariant_to_block_mode() {
    let hydro_count = 1_usize;
    let max_par_order = 0_usize;
    let n_anticipated = 0_usize;
    let k_max = 0_usize;
    let n_blks = 3_usize;

    let state = crate::test_support::state_layout_full(
        hydro_count,
        max_par_order,
        n_anticipated,
        k_max,
        vec![],
    );
    let parallel_theta = state.theta;
    let parallel_n_state = state.n_state;

    let parallel = block_template(BlockMode::Parallel, n_blks);
    let chronological = block_template(BlockMode::Chronological, n_blks);

    assert_eq!(
        parallel.n_state, parallel_n_state,
        "parallel template n_state must equal StateLayout n_state"
    );
    assert_eq!(
        chronological.n_state, parallel_n_state,
        "chronological build must not change n_state"
    );
    assert_eq!(
        parallel_theta, state.theta,
        "building chronological must not change theta"
    );

    assert_eq!(
        chronological.num_cols,
        parallel.num_cols + hydro_count * (n_blks - 1),
        "chronological adds exactly n_h*(n_blks-1) interior storage columns"
    );
}

/// Build the one-hydro FPHA study's stage-0 `StageLayout` AND `StageTemplate` from
/// one shared `TemplateBuildCtx`, so the column/row accessors the chronological
/// water tests call (`block_storage_col`, `turbine_col`, `row_water_balance_start`)
/// agree with the template they query by construction.
fn block_layout_and_template(
    block_mode: BlockMode,
    n_blks: usize,
) -> (StageLayout<'static>, StageTemplate, Vec<f64>) {
    use crate::hydro_models::FphaPlane;

    let system = Box::leak(Box::new(one_hydro_block_system(block_mode, n_blks)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 1.0,
                gamma_v: 0.2,
                gamma_q: 0.5,
                gamma_s: 0.05,
            }],
        }]],
        1,
        1,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    }));

    let (ctx, _, _) = super::build_template_build_ctx(
        system,
        InflowNonNegativityMethod::None,
        par_lp,
        production,
        &hydro_models.evaporation,
        resolved_params,
    );
    let ctx = Box::leak(Box::new(ctx));
    let state = Box::leak(Box::new(state_layout_for(ctx)));
    let stage = &system.stages()[0];

    let template = super::build_single_stage_template(ctx, state, stage, 0).template;
    let layout = StageLayout::new(ctx, state, stage, 0);
    let tau: Vec<f64> = stage
        .blocks
        .iter()
        .map(|b| b.duration_hours * super::super::M3S_TO_HM3)
        .collect();
    (layout, template, tau)
}

/// AC#1: a chronological `K = 2` Operating hydro emits two chained rows; block
/// `k`'s row carries `+1.0` on `Sᵏ`, `−1.0` on `Sᵏ⁻¹`, and `+τ_k` on that block's
/// turbine column.
#[test]
fn chronological_water_balance_chained_rows() {
    let (layout, t, tau) = block_layout_and_template(BlockMode::Chronological, 2);
    let h = 0_usize;
    let row0 = layout.row_water_balance_start() + h * 2;
    let row1 = layout.row_water_balance_start() + h * 2 + 1;

    let entry = |col: usize, row: usize| -> f64 {
        let es = csc_entries_for_col(&t, col);
        let vals: Vec<f64> = es
            .iter()
            .filter(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(vals.len(), 1, "exactly one entry at (col {col}, row {row})");
        vals[0]
    };

    assert_eq!(entry(layout.block_storage_col(h, 1), row0), 1.0);
    assert_eq!(entry(layout.block_storage_col(h, 0), row0), -1.0);
    assert_eq!(entry(layout.turbine_col(h, 0), row0), tau[0]);

    assert_eq!(entry(layout.block_storage_col(h, 2), row1), 1.0);
    assert_eq!(entry(layout.block_storage_col(h, 1), row1), -1.0);
    assert_eq!(entry(layout.turbine_col(h, 1), row1), tau[1]);
}

/// AC#2: summing the `K` chronological water rows coefficient-wise (over every
/// column) reproduces the parallel single-row build. The interior `Sⁱ` terms cancel
/// to `+1.0` on `Sᴷ` and `−1.0` on `S⁰`, and every `τ_k`-scaled flow term sums to
/// its `ζ`-scaled parallel coefficient (`Σ_k τ_k = ζ`).
#[test]
fn chronological_water_balance_telescopes_to_parallel() {
    let n_blks = 3_usize;
    let h = 0_usize;
    let (par_layout, par_t, _) = block_layout_and_template(BlockMode::Parallel, n_blks);
    let (chr_layout, chr_t, _) = block_layout_and_template(BlockMode::Chronological, n_blks);

    let dense_par = csc_to_dense(&par_t);
    let dense_chr = csc_to_dense(&chr_t);

    // Interior storage columns are shifted into chronological's control region, so
    // parallel and chronological do NOT share control-region column indices; compare
    // per SEMANTIC column via each layout's accessors.
    let par_row = par_layout.row_water_balance_start() + h;
    let chr_sum = |chr_col: usize| -> f64 {
        (0..n_blks)
            .map(|k| dense_chr[chr_layout.row_water_balance_start() + h * n_blks + k][chr_col])
            .sum()
    };
    let assert_telescopes = |par_col: usize, chr_col: usize, label: &str| {
        let summed = chr_sum(chr_col);
        let expected = dense_par[par_row][par_col];
        assert!(
            (summed - expected).abs() < 1e-12,
            "{label}: chronological telescoped sum {summed} != parallel {expected}"
        );
    };

    // Storage endpoints: Sᴷ (outgoing) telescopes to +1, S⁰ (incoming) to −1.
    assert_telescopes(
        h,
        chr_layout.block_storage_col(h, n_blks),
        "outgoing storage Sᴷ",
    );
    assert_telescopes(
        par_layout.col_storage_in_start() + h,
        chr_layout.block_storage_col(h, 0),
        "incoming storage S⁰",
    );

    // Per-block flow columns share their semantic accessor (same block index); each
    // block's τ_k sum reproduces the parallel ζ-scaled flow coefficient.
    for blk in 0..n_blks {
        assert_telescopes(
            par_layout.turbine_col(h, blk),
            chr_layout.turbine_col(h, blk),
            "turbine",
        );
        assert_telescopes(
            par_layout.spillage_col(h, blk),
            chr_layout.spillage_col(h, blk),
            "spillage",
        );
        assert_telescopes(
            par_layout.diversion_col(h, blk),
            chr_layout.diversion_col(h, blk),
            "diversion",
        );
    }

    // Withdrawal slacks: parallel applies ±ζ once; chronological's per-block ±τ_k sum
    // recovers ±ζ.
    assert_telescopes(
        par_layout.col_withdrawal_neg_start() + h,
        chr_layout.col_withdrawal_neg_start() + h,
        "withdrawal neg",
    );
    assert_telescopes(
        par_layout.col_withdrawal_pos_start() + h,
        chr_layout.col_withdrawal_pos_start() + h,
        "withdrawal pos",
    );

    // Interior boundaries Sⁱ (chronological-only) appear +1 in row i−1 and −1 in
    // row i, so they net to zero across the K rows.
    for k in 1..n_blks {
        let summed = chr_sum(chr_layout.block_storage_col(h, k));
        assert!(
            summed.abs() < 1e-12,
            "interior boundary S{k} must cancel across the K rows, got {summed}"
        );
    }

    // The telescoped RHS recovers the parallel RHS: Σ_k τ_k·(base − withdrawal) =
    // ζ·(base − withdrawal).
    let chr_rhs_sum: f64 = (0..n_blks)
        .map(|k| chr_t.row_lower[chr_layout.row_water_balance_start() + h * n_blks + k])
        .sum();
    let par_rhs = par_t.row_lower[par_row];
    assert!(
        (chr_rhs_sum - par_rhs).abs() < 1e-12,
        "telescoped RHS {chr_rhs_sum} != parallel RHS {par_rhs}"
    );
}

/// `StageGeometry::block_storage_col` resolves the same column as the source
/// `StageLayout::block_storage_col` for every boundary `k ∈ 0..=K` and every hydro,
/// so the discarded-`StageLayout` accessor is faithfully mirrored onto the per-stage
/// geometry the simulation read-path carries.
#[test]
fn stage_geometry_block_storage_col_matches_layout() {
    let n_blks = 3_usize;
    let (layout, _, _) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let geometry = super::StageGeometry::from_layout(&layout, BlockMode::Chronological);
    let storage_in_start = layout.col_storage_in_start();

    for h in 0..layout.n_h {
        for k in 0..=n_blks {
            assert_eq!(
                geometry.block_storage_col(h, k, storage_in_start),
                layout.block_storage_col(h, k),
                "block_storage_col mismatch at hydro {h}, boundary {k}"
            );
        }
    }
}

/// AC#3: a chronological `K = 1` build's water-balance row is byte-identical to the
/// parallel build's — the single chained row IS the parallel row (`τ_1 = ζ`, no
/// interior boundary). The full-template anchor is
/// [`chronological_k1_byte_identical_to_parallel`]; this isolates the water row.
#[test]
fn chronological_k1_water_row_byte_identical() {
    let parallel = block_template(BlockMode::Parallel, 1);
    let (chrono_layout, chrono, _tau) = block_layout_and_template(BlockMode::Chronological, 1);
    let row = chrono_layout.row_water_balance_start();

    let dense_par = csc_to_dense(&parallel);
    let dense_chr = csc_to_dense(&chrono);
    assert_eq!(parallel.num_cols, chrono.num_cols, "K=1 column count");
    for col in 0..parallel.num_cols {
        assert_eq!(
            dense_par[row][col].to_bits(),
            dense_chr[row][col].to_bits(),
            "K=1 water row coefficient mismatch at col {col}"
        );
    }
    assert_eq!(
        parallel.row_lower[row].to_bits(),
        chrono.row_lower[row].to_bits(),
        "K=1 water row_lower"
    );
    assert_eq!(
        parallel.row_upper[row].to_bits(),
        chrono.row_upper[row].to_bits(),
        "K=1 water row_upper"
    );
}

/// §9 contract "D06 preserved per block": at `K ≥ 2` each per-block FPHA plane row
/// carries `−γᵥ/2` on BOTH block-local storage columns `block_storage_col(h, k−1)`
/// (`Sᵏ⁻¹`) and `block_storage_col(h, k)` (`Sᵏ`) — the FPHA average-storage rule
/// applied to the block's own `(Sᵏ⁻¹, Sᵏ)` pair. Placing it on the outgoing column
/// alone (or on the stage endpoints instead of the block boundaries) understates the
/// head term and is the wrong-but-compiling alternative D06 pins against.
#[test]
fn chronological_d06_gamma_v_on_both_block_columns() {
    let n_blks = 2_usize;
    let (layout, t, _tau) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let h = 0_usize;
    // `block_template`/`block_layout_and_template` fix the single FPHA plane's
    // gamma_v; the row coefficient is `−gamma_v/2` after the objective-neutral
    // matrix fill (matrix values are not cost-scaled).
    let half_gamma_v = -0.2 / 2.0;

    let entry = |col: usize, row: usize| -> f64 {
        csc_entries_for_col(&t, col)
            .iter()
            .filter(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .sum()
    };

    for k in 1..=n_blks {
        let blk = k - 1;
        let row = layout.row_fpha_start() + blk;
        assert_eq!(
            entry(layout.block_storage_col(h, k - 1), row),
            half_gamma_v,
            "block {k}: −γᵥ/2 on Sᵏ⁻¹ (D06 both-columns)"
        );
        assert_eq!(
            entry(layout.block_storage_col(h, k), row),
            half_gamma_v,
            "block {k}: −γᵥ/2 on Sᵏ (D06 both-columns)"
        );
    }
}

/// §5 cross-mode cut-row byte-comparability invariant: for a chronological `K ≥ 2`
/// FPHA study, the matrix-derived column scale at an interior `block_storage_col(h,
/// k)` equals the endpoint storage-column scale. Identical state-column scaling
/// across the storage family is what keeps rendered cut rows (`−coeff·col_scale[col]`)
/// byte-comparable; a divergent interior scale would silently desynchronise the cut
/// rendering.
#[test]
fn chronological_interior_storage_scale_matches_endpoint() {
    let n_blks = 3_usize;
    let (layout, t, _tau) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let h = 0_usize;

    let col_scale = super::super::compute_col_scale(t.num_cols, &t.col_starts, &t.values);

    // The outgoing endpoint Sᴷ (`block_storage_col(h, K)`) is the reference storage
    // column whose scale every interior boundary must match.
    let endpoint_scale = col_scale[layout.block_storage_col(h, n_blks)];
    for k in 1..n_blks {
        let interior_col = layout.block_storage_col(h, k);
        assert_eq!(
            col_scale[interior_col].to_bits(),
            endpoint_scale.to_bits(),
            "interior boundary S{k} scale must equal the endpoint Sᴷ scale (§5 \
             byte-comparability); divergence signals FPHA/evap coefficients differ \
             between interior and endpoint storage columns"
        );
    }
}

// ── Filling / PreFilling block anchors (D38–D42, σ_fill on Sᴷ) ─────────────

const FILL_N_STAGES: usize = 5;
const FILL_ENTRY_ID: i32 = 4;
const FILL_PRE_START_ID: i32 = 3;
const FILL_MIN_STORAGE_HM3: f64 = 60.0;
const FILL_RATE_M3S: f64 = 5.0;
const FILL_PRE_HYDRO_ID: i32 = 2;
const FILL_FILL_HYDRO_ID: i32 = 3;

/// A `FILL_N_STAGES`-stage, one-bus cascade under `block_mode` with `n_blks` blocks.
/// H2 (`FILL_PRE_HYDRO_ID`) is a filling hydro whose `start_stage_id` sits at
/// `FILL_PRE_START_ID`, so at stage id 0 it is `PreFilling`; H3
/// (`FILL_FILL_HYDRO_ID`) is a filling hydro with `start_stage_id = 0`, so at stage
/// id 0 it is `Filling`. Both share `entry = FILL_ENTRY_ID`. A backup thermal and a
/// bus deficit segment keep the LP feasible regardless of the frozen filling storage.
fn filling_block_system(block_mode: BlockMode, n_blks: usize) -> System {
    use cobre_core::scenario::{InflowModel, LoadModel};

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

    let filling_hydro = |id: i32, downstream: Option<i32>, start: i32| Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: downstream.map(EntityId),
        travel_time_hours: None,
        entry_stage_id: Some(FILL_ENTRY_ID),
        exit_stage_id: None,
        min_storage_hm3: FILL_MIN_STORAGE_HM3,
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
        filling: Some(FillingConfig {
            start_stage_id: start,
            filling_min_rate_m3s: FILL_RATE_M3S,
        }),
        penalties: hydro_penalties_zero(),
    };

    let hydros = vec![
        filling_hydro(
            FILL_PRE_HYDRO_ID,
            Some(FILL_FILL_HYDRO_ID),
            FILL_PRE_START_ID,
        ),
        filling_hydro(FILL_FILL_HYDRO_ID, None, 0),
    ];

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 360.0 + 24.0 * f64::from(u32::try_from(b).unwrap_or(0)),
        })
        .collect();

    let stages: Vec<Stage> = (0..FILL_N_STAGES)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1).unwrap(),
            season_id: Some(0),
            blocks: blocks.clone(),
            block_mode,
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

    let inflow_models: Vec<InflowModel> = (0..FILL_N_STAGES)
        .flat_map(|i| {
            [FILL_PRE_HYDRO_ID, FILL_FILL_HYDRO_ID].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..FILL_N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: FILL_N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: FILL_MIN_STORAGE_HM3,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: FILL_RATE_M3S,
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
    for h_idx in [0_usize, 1] {
        for stage_idx in 0..FILL_N_STAGES {
            let hb = bounds.hydro_bounds_mut(h_idx, stage_idx);
            hb.min_storage_hm3 = FILL_MIN_STORAGE_HM3;
            hb.filling_min_rate_m3s = FILL_RATE_M3S;
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: FILL_N_STAGES,
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
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("filling_block_system: valid filling cascade")
}

/// Build the stage-0 `StageLayout` and `StageTemplate` for [`filling_block_system`]
/// under `block_mode` with `n_blks` blocks, alongside the resolved `filling_v_target`
/// fold the σ_fill row RHS reads. Stage 0 places H2 in `PreFilling` and H3 in
/// `Filling`. `Box::leak` matches [`block_layout_and_template`]: the ctx/state must
/// outlive the borrowed `StageLayout`.
fn filling_block_layout_and_template(
    block_mode: BlockMode,
    n_blks: usize,
) -> (
    StageLayout<'static>,
    StageTemplate,
    std::collections::BTreeMap<(usize, i32), f64>,
) {
    let system = Box::leak(Box::new(filling_block_system(block_mode, n_blks)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 };
                FILL_N_STAGES
            ];
            2
        ],
        2,
        FILL_N_STAGES,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    }));

    let (ctx, _, _) = super::build_template_build_ctx(
        system,
        InflowNonNegativityMethod::None,
        par_lp,
        production,
        &hydro_models.evaporation,
        resolved_params,
    );
    let ctx = Box::leak(Box::new(ctx));
    let state = Box::leak(Box::new(state_layout_for(ctx)));
    let stage = &system.stages()[0];

    let template = super::build_single_stage_template(ctx, state, stage, 0).template;
    let layout = StageLayout::new(ctx, state, stage, 0);
    (layout, template, ctx.filling_v_target.clone())
}

/// §9 contract "D38–D42 preserved per block": at `K ≥ 2` a `PreFilling` hydro's `K`
/// water rows are frozen identities (`Sᵏ − Sᵏ⁻¹ = 0`) and its spillage AND turbine
/// columns are frozen `[0, 0]` on every block (no dam, no machinery). A `Filling`
/// hydro at the same stage keeps its per-block spillage FREE (the D40 over-dam relief
/// valve) while its turbine stays frozen `[0, 0]` (no machinery until entry).
#[test]
fn chronological_prefilling_d38_d42_per_block() {
    let n_blks = 2_usize;
    let (layout, t, _v_target) =
        filling_block_layout_and_template(BlockMode::Chronological, n_blks);

    // Positional indices after the SystemBuilder id-sort: H2 → 0 (PreFilling at
    // stage 0), H3 → 1 (Filling at stage 0).
    let h_pre = 0_usize;
    let h_fill = 1_usize;

    let entry = |col: usize, row: usize| -> f64 {
        csc_entries_for_col(&t, col)
            .iter()
            .filter(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .sum()
    };

    for k in 1..=n_blks {
        let blk = k - 1;
        let row = layout.row_water_balance_start() + h_pre * n_blks + blk;
        assert_eq!(
            entry(layout.block_storage_col(h_pre, k), row),
            1.0,
            "PreFilling block {k}: +1 on Sᵏ (frozen identity, D38/D39/D42)"
        );
        assert_eq!(
            entry(layout.block_storage_col(h_pre, k - 1), row),
            -1.0,
            "PreFilling block {k}: −1 on Sᵏ⁻¹ (frozen identity, D38/D39/D42)"
        );
        assert_eq!(
            t.row_lower[row], 0.0,
            "PreFilling block {k}: frozen-identity RHS lower == 0"
        );
        assert_eq!(
            t.row_upper[row], 0.0,
            "PreFilling block {k}: frozen-identity RHS upper == 0"
        );

        let spill_pre = layout.spillage_col(h_pre, blk);
        assert_eq!(
            (t.col_lower[spill_pre], t.col_upper[spill_pre]),
            (0.0, 0.0),
            "PreFilling block {k}: spillage frozen [0,0] (no dam to spill from, D38/D39/D42)"
        );
        let turb_pre = layout.turbine_col(h_pre, blk);
        assert_eq!(
            (t.col_lower[turb_pre], t.col_upper[turb_pre]),
            (0.0, 0.0),
            "PreFilling block {k}: turbine frozen [0,0] (no machinery, D38/D39/D42)"
        );

        // A Filling hydro's spillage is the legitimate D40 relief valve: free upward.
        let spill_fill = layout.spillage_col(h_fill, blk);
        assert_eq!(
            t.col_lower[spill_fill], 0.0,
            "Filling block {k}: spillage lower == 0"
        );
        assert_eq!(
            t.col_upper[spill_fill],
            f64::INFINITY,
            "Filling block {k}: spillage FREE (D40 relief valve), not frozen"
        );
    }
}

/// §9 contract "Filling-phase target on `Sᴷ`": at `K ≥ 2` a `Filling`-phase hydro's
/// `σ_fill` row references the stage-final storage `block_storage_col(h, K)` (= `Sᴷ`,
/// which aliases the outgoing endpoint `h`), its `V_target` fold value is UNCHANGED
/// from the parallel build (`build_filling_v_target` is keyed `(hydro, stage)` and
/// `ζ`-scaled, so the τ_k-replaces-ζ water change leaves it untouched), and its
/// per-block spillage stays FREE (D40).
#[test]
fn chronological_filling_target_on_final_storage() {
    let n_blks = 2_usize;
    let (chr_layout, chr_t, chr_v_target) =
        filling_block_layout_and_template(BlockMode::Chronological, n_blks);
    let (_par_layout, _par_t, par_v_target) =
        filling_block_layout_and_template(BlockMode::Parallel, n_blks);

    // H3 is the Filling hydro at stage 0 (positional index 1 after the id-sort).
    let h_fill = 1_usize;
    assert_eq!(
        chr_layout.filling_target_hydro_indices,
        vec![h_fill],
        "exactly the Filling hydro H3 emits a σ_fill target at stage 0"
    );

    // The σ_fill row places +1 on the OUTGOING storage column, which in chronological
    // mode is the stage-final Sᴷ (block_storage_col aliases the outgoing endpoint to
    // the dense hydro index h_fill).
    let sk_col = chr_layout.block_storage_col(h_fill, n_blks);
    assert_eq!(
        sk_col, h_fill,
        "block_storage_col(h, K) aliases the outgoing endpoint (= dense hydro index)"
    );
    let row = chr_layout.row_filling_target_start();
    let entry = |col: usize| -> f64 {
        csc_entries_for_col(&chr_t, col)
            .iter()
            .filter(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .sum()
    };
    assert_eq!(
        entry(sk_col),
        1.0,
        "σ_fill row references Sᴷ (block_storage_col(h, K)), the stage-final storage"
    );
    assert_eq!(
        entry(chr_layout.col_filling_target_start()),
        1.0,
        "σ_fill row carries +1 on its σ_fill slack column"
    );

    // The ζ-scaled V_target fold is mode-independent: build_filling_v_target never
    // sees block_mode, so replacing ζ with per-block τ_k in the water rows must not
    // move the target RHS.
    let stage0_id = 0_i32;
    let chr_target = chr_v_target[&(h_fill, stage0_id)];
    let par_target = par_v_target[&(h_fill, stage0_id)];
    assert_eq!(
        chr_target.to_bits(),
        par_target.to_bits(),
        "Filling V_target fold must be byte-identical across modes (keyed (hydro, \
         stage), ζ-scaled)"
    );
    assert_eq!(
        chr_t.row_lower[row].to_bits(),
        chr_target.to_bits(),
        "σ_fill row RHS (≥ lower) equals the V_target fold value"
    );

    // Per-block spillage stays the free D40 relief valve, not frozen.
    for blk in 0..n_blks {
        let spill = chr_layout.spillage_col(h_fill, blk);
        assert_eq!(
            (chr_t.col_lower[spill], chr_t.col_upper[spill]),
            (0.0, f64::INFINITY),
            "Filling block {blk}: per-block spillage FREE (D40), not frozen"
        );
    }
}
