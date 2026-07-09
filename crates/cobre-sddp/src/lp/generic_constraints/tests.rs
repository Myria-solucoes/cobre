#![allow(
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op
)]

use std::collections::{BTreeMap, HashMap};

use cobre_core::entities::{HydroGenerationModel, HydroPenalties};
use cobre_core::{
    CascadeTopology, ContractType, EnergyContract, EntityId, Hydro, PumpingStation, VariableRef,
};

use super::{
    CascadeRefs, ContractRefs, ElementKind, GenericResolverGeom, PumpingRefs, block_col_range,
    contract_family_slot, resolve_variable_ref, variable_ref_is_block_independent,
};
use crate::hydro_models::{FphaPlane, ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StateLayout;
use crate::lp_builder::StageGeometry;

// ── Test helpers ──────────────────────────────────────────────────────────

/// Build the [`StateLayout`] matching [`make_indexer`]'s state dimensions
/// (N=4 hydros, L=0 lags, no anticipated thermals), so the role-(a) storage /
/// z-inflow columns the resolver reads through the handle equal the indexer's.
fn make_state() -> StateLayout {
    StateLayout::new(4, 0, 0, Vec::new(), 0, 0, vec![], &[0, 0, 0, 0])
}

/// Build the [`StateLayout`] matching [`make_indexer_with_anticipated`]'s state
/// dimensions (N=0 hydros, L=0, A=1 anticipated plant, k_max=2, K=[2]).
fn make_state_anticipated() -> StateLayout {
    StateLayout::new(0, 0, 0, Vec::new(), 1, 2, vec![2], &[])
}

/// Build the [`GenericResolverGeom`] view from a test [`StageGeometry`] (role
/// (b)), a [`StateLayout`] (role (a)), the deficit-segment stride, and the
/// anticipated-thermal identity list. Mirrors the production view built in
/// `entries.rs`, sourcing each role-(b) range from the geometry's equivalent
/// field so the resolver tests exercise the same offsets production does.
fn make_geom<'a>(
    indexer: &'a StageGeometry,
    state: &'a StateLayout,
    max_deficit_segments: usize,
    anticipated_thermal_indices: &[usize],
) -> GenericResolverGeom<'a> {
    make_geom_with_contracts(
        indexer,
        state,
        max_deficit_segments,
        anticipated_thermal_indices,
        &indexer.contract_import,
        &indexer.contract_export,
    )
}

/// Like [`make_geom`], but with explicit contract column ranges. The
/// `geometry` fixture hardcodes both contract ranges to `0..0`, so the contract
/// resolution / `block_col_range` tests inject their own non-empty ranges here.
fn make_geom_with_contracts<'a>(
    indexer: &'a StageGeometry,
    state: &'a StateLayout,
    max_deficit_segments: usize,
    anticipated_thermal_indices: &[usize],
    contract_import: &'a std::ops::Range<usize>,
    contract_export: &'a std::ops::Range<usize>,
) -> GenericResolverGeom<'a> {
    // Production builds the `anticipated_local_by_sys_pos` reverse map on the
    // per-stage `StageLayout`; the non-state anticipated identity list lives on
    // `StudyDimensions`, so the test passes it in here. Reconstruct the
    // equivalent reverse map and leak it so the borrowed `GenericResolverGeom`
    // field has a `'a`-compatible referent without threading an owner through
    // every call site.
    let reverse: std::collections::HashMap<usize, usize> = anticipated_thermal_indices
        .iter()
        .enumerate()
        .map(|(local, &sys_pos)| (sys_pos, local))
        .collect();
    let reverse: &'a std::collections::HashMap<usize, usize> = Box::leak(Box::new(reverse));
    GenericResolverGeom {
        state,
        storage_internal_start: indexer.storage_internal_start,
        turbine: &indexer.turbine,
        spillage: &indexer.spillage,
        diversion: &indexer.diversion,
        thermal: &indexer.thermal,
        line_fwd: &indexer.line_fwd,
        line_rev: &indexer.line_rev,
        excess: &indexer.excess,
        contract_import,
        contract_export,
        generation: &indexer.generation,
        deficit: &indexer.deficit,
        max_deficit_segments,
        n_blks: indexer.n_blks,
        evap_indices: &indexer.evap_indices,
        evap_hydro_indices: &indexer.evap_hydro_indices,
        fpha_hydro_indices: &indexer.fpha_hydro_indices,
        anticipated_decision_start: indexer.anticipated_decision.start,
        anticipated_local_by_sys_pos: reverse,
    }
}

/// Minimal `Hydro` carrying only the `id`/`downstream_id` that
/// [`CascadeTopology::build`] reads; every other field is an inert default.
/// Mirrors the `make_hydro` helper in `cobre-core`'s cascade tests.
fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
    let zero_penalties = HydroPenalties {
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
    };
    Hydro {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        downstream_id: downstream_id.map(EntityId),
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 1.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties,
    }
}

/// An empty cascade (no upstream links) for the resolver paths that ignore it.
fn empty_cascade() -> CascadeTopology {
    CascadeTopology::build(&[])
}

/// Build a `StageGeometry` with equipment for tests.
///
/// N=4 hydros (2 FPHA at positions 0, 2), L=0, T=2 thermals, Ln=1 line, B=2 buses, K=3 blocks.
/// S=2 max deficit segments.
///
/// Column layout:
///   storage:   [0, 4)         = 0..4
///   lags:      [4, 4*(1+0))   = 4..4   (L=0, empty)
///   z_inflow:  [4*(1+0), 4*(2+0)) = 4..8
///   storage_in:[4*(2+0), 4*(3+0)) = 8..12
///   theta = N*(3+L) = 4*(3+0) = 12
///   decision_start = 13
///   turbine:    [13, 13+4*3) = 13..25   (4 hydros * 3 blocks)
///   spillage:   [25, 25+4*3) = 25..37
///   diversion:  [37, 37+4*3) = 37..49  (4 hydros * 3 blocks)
///   thermal:    [49, 49+2*3) = 49..55  (2 thermals * 3 blocks)
///   line_fwd:   [55, 55+1*3) = 55..58  (1 line * 3 blocks)
///   line_rev:   [58, 58+1*3) = 58..61
///   deficit:    [61, 61+2*2*3) = 61..73 (2 buses * 2 segs * 3 blocks)
///   excess:     [73, 73+2*3) = 73..79  (2 buses * 3 blocks)
///   generation: [79, 79+2*3) = 79..85  (2 FPHA hydros * 3 blocks)
///   evap: none
///   withdrawal_slack_neg: [85, 89)  withdrawal_slack_pos: [89, 93) (4 hydros)
///
/// Storage: 0..4
fn make_indexer() -> StageGeometry {
    // N=4, L=0, T=2, Ln=1, B=2, K=3, no penalty, 2 FPHA hydros at positions 0 and 2
    // (local FPHA indices 0 and 1), each with 3 planes.
    crate::test_support::geometry(
        &crate::test_support::GeometryDims {
            hydro_count: 4,
            n_thermals: 2,
            n_lines: 1,
            n_buses: 2,
            n_blks: 3,
            max_deficit_segments: 2,
            ..Default::default()
        },
        vec![0, 2],
        &[3, 3],
        vec![],
    )
}

/// Build a `ProductionModelSet` for 4 hydros and 2 stages.
///
/// - Hydro 0: FPHA at all stages
/// - Hydro 1: ConstantProductivity(2.5) at all stages
/// - Hydro 2: FPHA at all stages
/// - Hydro 3: ConstantProductivity(1.0) at all stages
fn make_production_models() -> ProductionModelSet {
    let fpha_plane = FphaPlane {
        intercept: 0.0,
        gamma_v: 0.1,
        gamma_q: 0.5,
        gamma_s: 0.0,
    };
    let fpha_model = || ResolvedProductionModel::Fpha {
        planes: vec![fpha_plane],
    };
    let models: Vec<Vec<ResolvedProductionModel>> = vec![
        vec![fpha_model(), fpha_model()], // hydro 0 — FPHA
        vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
            ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
        ], // hydro 1 — constant
        vec![fpha_model(), fpha_model()], // hydro 2 — FPHA
        vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
        ], // hydro 3 — constant
    ];
    ProductionModelSet::new(models, 4, 2)
}

fn make_hydro_pos() -> BTreeMap<EntityId, usize> {
    // Hydros with EntityId 10, 20, 30, 40 at system positions 0, 1, 2, 3
    [
        (EntityId(10), 0),
        (EntityId(20), 1),
        (EntityId(30), 2),
        (EntityId(40), 3),
    ]
    .into_iter()
    .collect()
}

fn make_thermal_pos() -> BTreeMap<EntityId, usize> {
    // Thermals with EntityId 5 and 6 at positions 0 and 1
    [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect()
}

fn make_bus_pos() -> BTreeMap<EntityId, usize> {
    // Buses with EntityId 100, 200 at positions 0, 1
    [(EntityId(100), 0), (EntityId(200), 1)]
        .into_iter()
        .collect()
}

fn make_line_pos() -> BTreeMap<EntityId, usize> {
    // Line with EntityId 50 at position 0
    [(EntityId(50), 0)].into_iter().collect()
}

fn call(
    var_ref: VariableRef,
    block_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    hydro_pos: &BTreeMap<EntityId, usize>,
    thermal_pos: &BTreeMap<EntityId, usize>,
    bus_pos: &BTreeMap<EntityId, usize>,
    line_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    // Paths under test here ignore the cascade context; pass an empty one.
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    call_with_cascade(
        var_ref,
        block_idx,
        geom,
        production_models,
        hydro_pos,
        thermal_pos,
        bus_pos,
        line_pos,
        &cascade,
        &diversion_upstream,
    )
}

/// Like [`call`], but threads an explicit cascade topology and
/// diversion-into map for the `HydroInflow` total-inflow tests.
fn call_with_cascade(
    var_ref: VariableRef,
    block_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    hydro_pos: &BTreeMap<EntityId, usize>,
    thermal_pos: &BTreeMap<EntityId, usize>,
    bus_pos: &BTreeMap<EntityId, usize>,
    line_pos: &BTreeMap<EntityId, usize>,
    cascade: &CascadeTopology,
    diversion_upstream: &HashMap<EntityId, Vec<usize>>,
) -> Vec<(usize, f64)> {
    let positions = super::EntityPositionMaps {
        hydro: hydro_pos,
        thermal: thermal_pos,
        bus: bus_pos,
        line: line_pos,
    };
    let cascade_refs = CascadeRefs {
        cascade,
        diversion_upstream,
    };
    // Non-pumping paths ignore the pumping context; pass an empty one (no
    // stations), so the PumpingFlow/PumpingPower lookup misses and yields [].
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let pumping_refs = PumpingRefs {
        col_pumping_start: 0,
        pumping_stations: &no_stations,
        pumping_pos: &empty_pumping_pos,
    };
    // Non-contract paths ignore the contract context; pass an empty one, so the
    // ContractImport/ContractExport lookup misses and yields [].
    let no_contracts: Vec<EnergyContract> = Vec::new();
    let empty_contract_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let contract_refs = ContractRefs {
        contracts: &no_contracts,
        contract_pos: &empty_contract_pos,
    };
    resolve_variable_ref(
        &var_ref,
        block_idx,
        0, // stage_idx = 0
        geom,
        production_models,
        &positions,
        &cascade_refs,
        &pumping_refs,
        &contract_refs,
    )
}

/// Resolve a `PumpingFlow`/`PumpingPower` ref with an explicit pumping
/// context (column start, block stride, station slice, position map).
///
/// Threads real pumping data the way the production `fill_pumping_water_entries`
/// caller does — sourcing
/// `col_pumping_start` from a `StageLayout`-style reserved range — so the
/// pumping arms exercise their real column arithmetic and consumption-rate
/// coefficient instead of the empty fixture used by [`call`].
// Mirrors the production resolver's argument surface it exercises; bundling
// into a struct would diverge the test from the real call shape.
#[allow(clippy::too_many_arguments)]
fn call_pumping(
    var_ref: VariableRef,
    block_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    col_pumping_start: usize,
    n_blks: usize,
    pumping_stations: &[PumpingStation],
    pumping_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let empty: BTreeMap<EntityId, usize> = BTreeMap::new();
    let positions = super::EntityPositionMaps {
        hydro: &empty,
        thermal: &empty,
        bus: &empty,
        line: &empty,
    };
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    let cascade_refs = CascadeRefs {
        cascade: &cascade,
        diversion_upstream: &diversion_upstream,
    };
    let pumping_refs = PumpingRefs {
        col_pumping_start,
        pumping_stations,
        pumping_pos,
    };
    // The pumping column stride is now sourced from the geometry's `BlockGrid`,
    // so the fixture's declared `n_blks` must match `geom.n_blks` for the
    // asserted columns to hold; pin that invariant rather than silently
    // diverging if a future fixture sets a mismatched stride.
    assert_eq!(n_blks, geom.n_blks);
    let no_contracts: Vec<EnergyContract> = Vec::new();
    let empty_contract_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let contract_refs = ContractRefs {
        contracts: &no_contracts,
        contract_pos: &empty_contract_pos,
    };
    resolve_variable_ref(
        &var_ref,
        block_idx,
        0, // stage_idx = 0
        geom,
        production_models,
        &positions,
        &cascade_refs,
        &pumping_refs,
        &contract_refs,
    )
}

/// Resolve a `ContractImport`/`ContractExport` ref with an explicit contract
/// context (the id-sorted contract slice and its id→slot map), threaded the way
/// the production `fill_generic_constraint_entries` caller does so the contract
/// arms exercise their real per-family-slot column arithmetic. The column bases
/// ride on `geom.contract_import`/`contract_export`.
fn call_contract(
    var_ref: VariableRef,
    block_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    contracts: &[EnergyContract],
    contract_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let empty: BTreeMap<EntityId, usize> = BTreeMap::new();
    let positions = super::EntityPositionMaps {
        hydro: &empty,
        thermal: &empty,
        bus: &empty,
        line: &empty,
    };
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    let cascade_refs = CascadeRefs {
        cascade: &cascade,
        diversion_upstream: &diversion_upstream,
    };
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let pumping_refs = PumpingRefs {
        col_pumping_start: 0,
        pumping_stations: &no_stations,
        pumping_pos: &empty_pumping_pos,
    };
    let contract_refs = ContractRefs {
        contracts,
        contract_pos,
    };
    resolve_variable_ref(
        &var_ref,
        block_idx,
        0, // stage_idx = 0
        geom,
        production_models,
        &positions,
        &cascade_refs,
        &pumping_refs,
        &contract_refs,
    )
}

/// An energy contract carrying only the `id`/`bus_id`/`contract_type` the
/// resolver and load-balance fill read; every other field is an inert value.
fn make_contract(id: i32, contract_type: ContractType) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 0.0,
        min_mw: 0.0,
        max_mw: 1.0,
    }
}

/// A pumping station carrying a `consumption_mw_per_m3s` rate; every other
/// field is an inert value the resolver does not read.
fn make_pumping_station(id: i32, consumption_mw_per_m3s: f64) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: String::new(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        source_hydro_id: EntityId(0),
        destination_hydro_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s,
        min_flow_m3s: 0.0,
        max_flow_m3s: 1.0,
    }
}

// ── ThermalGeneration tests ───────────────────────────────────────────────

/// `ThermalGeneration` column arithmetic across the `block_id`/position axes
/// the per-arm coverage requires: one `block_id = None`, one `block_id = Some`,
/// and one `position != 0`. All resolve through `resolve_block_variable` with
/// `block_col_range(geom, ElementKind::Thermal).start = 49`, `n_blks = 3`.
#[test]
fn thermal_generation_column_arithmetic() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    // (case_name, thermal_id, block_id, block_idx, expected_col)
    let cases: [(&str, EntityId, Option<usize>, usize, usize); 3] = [
        ("none_block_1", EntityId(5), None, 1, 49 + 0 * 3 + 1),
        ("some_block_2", EntityId(5), Some(2), 2, 49 + 0 * 3 + 2),
        ("second_thermal", EntityId(6), None, 0, 49 + 1 * 3 + 0),
    ];

    for (case_name, thermal_id, block_id, block_idx, expected_col) in cases {
        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id,
                block_id,
            },
            block_idx,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        assert_eq!(
            result,
            vec![(expected_col, 1.0)],
            "thermal_generation case `{case_name}`",
        );
    }
}

// ── HydroStorage tests ────────────────────────────────────────────────────

/// HydroStorage returns stage-level storage column.
///
/// storage.start = 0, hydro_pos[EntityId(10)] = 0
/// Expected column = 0 + 0 = 0, regardless of block_idx.
#[test]
fn hydro_storage_stage_level_ignores_block() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    for block_idx in [0, 1, 2] {
        let result = call(
            VariableRef::HydroStorage {
                hydro_id: EntityId(10),
            },
            block_idx,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        // storage.start = 0, pos = 0 → column 0
        assert_eq!(result, vec![(0, 1.0)], "block_idx={block_idx}");
    }

    // Hydro at position 2 (EntityId 30)
    let result2 = call(
        VariableRef::HydroStorage {
            hydro_id: EntityId(30),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    // storage.start = 0, pos = 2 → column 2
    assert_eq!(result2, vec![(2, 1.0)]);
}

// ── HydroOutflow tests ────────────────────────────────────────────────────

/// HydroOutflow returns 2 entries (turbine + spillage).
///
/// hydro_pos[EntityId(40)] = 3 (position 3), block_id=None, block_idx=0
/// turbine.start = 13, spillage.start = 25, n_blks = 3
/// Expected: [(13 + 3*3 + 0, 1.0), (25 + 3*3 + 0, 1.0)] = [(22, 1.0), (34, 1.0)]
#[test]
fn hydro_outflow_expands_to_turbine_and_spillage() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroOutflow {
            hydro_id: EntityId(40),
            block_id: None,
        },
        0, // block_idx
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    let turbine_col = 13 + 3 * 3 + 0; // 22
    let spillage_col = 25 + 3 * 3 + 0; // 34
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (turbine_col, 1.0));
    assert_eq!(result[1], (spillage_col, 1.0));
}

/// HydroOutflow with block_id=Some(1) at block_idx=0: should use the explicit block.
#[test]
fn hydro_outflow_block_id_some_uses_explicit_block() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroOutflow {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0, // block_idx is irrelevant when block_id = Some
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    // hydro pos=0, turbine.start=13, spillage.start=25, block=1, n_blks=3
    assert_eq!(result, vec![(13 + 0 * 3 + 1, 1.0), (25 + 0 * 3 + 1, 1.0)]);
}

// ── HydroGeneration tests ─────────────────────────────────────────────────

/// HydroGeneration for constant-productivity hydro returns
/// turbine column with productivity multiplier.
///
/// hydro_pos[EntityId(20)] = 1 → constant productivity 2.5
/// turbine.start = 13, n_blks = 3, block_idx = 0
/// Expected: [(13 + 1*3 + 0, 2.5)] = [(16, 2.5)]
#[test]
fn hydro_generation_constant_productivity_maps_to_turbine() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(20),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    // hydro pos=1, turbine.start=13, n_blks=3, block=0, productivity=2.5
    assert_eq!(result, vec![(13 + 1 * 3 + 0, 2.5)]);
}

/// HydroGeneration for FPHA hydro returns generation column.
///
/// hydro_pos[EntityId(10)] = 0 → FPHA (local FPHA index = 0)
/// generation.start = 79, n_blks = 3, block_idx = 0
/// Expected: [(79 + 0*3 + 0, 1.0)] = [(79, 1.0)]
#[test]
fn hydro_generation_fpha_maps_to_generation_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(10),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    // FPHA local index 0, generation.start=79, n_blks=3, block=0
    assert_eq!(result, vec![(79 + 0 * 3 + 0, 1.0)]);
}

/// HydroGeneration for FPHA hydro at position 2 (second FPHA hydro, local index 1).
///
/// hydro_pos[EntityId(30)] = 2 → FPHA (local FPHA index = 1)
/// generation.start = 79, n_blks = 3, block_idx = 2
/// Expected: [(79 + 1*3 + 2, 1.0)] = [(84, 1.0)]
#[test]
fn hydro_generation_fpha_second_hydro_block_2() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroGeneration {
            hydro_id: EntityId(30),
            block_id: None,
        },
        2,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    // FPHA local index 1, generation.start=79, n_blks=3, block=2
    assert_eq!(result, vec![(79 + 1 * 3 + 2, 1.0)]);
}

// ── HydroEvaporation tests ────────────────────────────────────────────────

/// HydroEvaporation maps to the evaporation-outflow column for the matching evaporation hydro.
///
/// Use a dedicated indexer with evaporation hydros to test this path.
///
/// N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, evap hydro at pos 0.
/// theta = 2*(3+0) = 6
/// turbine:    [7, 9)
/// spillage:   [9, 11)
/// diversion: [11, 13)
/// deficit:   [13, 14)
/// excess:    [14, 15)
/// evap cols: [15, 18)  → evaporation_flow=15, f_evap_plus=16, f_evap_minus=17
#[test]
fn hydro_evaporation_maps_to_evaporation_flow_col() {
    let evap_indexer = crate::test_support::geometry(
        &crate::test_support::GeometryDims {
            hydro_count: 2,
            n_buses: 1,
            n_blks: 1,
            ..Default::default()
        },
        vec![],
        &[],
        vec![0],
    );

    let prod_models = ProductionModelSet::new(
        vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
        ],
        2,
        1,
    );

    let hpos: BTreeMap<EntityId, usize> =
        [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
    let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    let positions = super::EntityPositionMaps {
        hydro: &hpos,
        thermal: &tpos,
        bus: &bpos,
        line: &lpos,
    };
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    let cascade_refs = CascadeRefs {
        cascade: &cascade,
        diversion_upstream: &diversion_upstream,
    };
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let pumping_refs = PumpingRefs {
        col_pumping_start: 0,
        pumping_stations: &no_stations,
        pumping_pos: &empty_pumping_pos,
    };
    let state = StateLayout::new(2, 0, 0, Vec::new(), 0, 0, vec![], &[0, 0]);
    let geom = make_geom(&evap_indexer, &state, 1, &[]);
    let no_contracts: Vec<EnergyContract> = Vec::new();
    let empty_contract_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let contract_refs = ContractRefs {
        contracts: &no_contracts,
        contract_pos: &empty_contract_pos,
    };
    let result = resolve_variable_ref(
        &VariableRef::HydroEvaporation {
            hydro_id: EntityId(10),
            block_id: None,
        },
        0,
        0, // stage_idx
        &geom,
        &prod_models,
        &positions,
        &cascade_refs,
        &pumping_refs,
        &contract_refs,
    );

    assert_eq!(result, vec![(15, 1.0)]);
}

/// HydroEvaporation for hydro that has no evaporation model returns empty vec.
#[test]
fn hydro_evaporation_no_evap_model_returns_empty() {
    let evap_indexer = crate::test_support::geometry(
        &crate::test_support::GeometryDims {
            hydro_count: 2,
            n_buses: 1,
            n_blks: 1,
            ..Default::default()
        },
        vec![],
        &[],
        vec![0],
    );

    let prod_models = ProductionModelSet::new(
        vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
        ],
        2,
        1,
    );

    let hpos: BTreeMap<EntityId, usize> =
        [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
    let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    // Hydro 20 (pos=1) has no evaporation in evap_hydro_indices=[0]
    let positions = super::EntityPositionMaps {
        hydro: &hpos,
        thermal: &tpos,
        bus: &bpos,
        line: &lpos,
    };
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    let cascade_refs = CascadeRefs {
        cascade: &cascade,
        diversion_upstream: &diversion_upstream,
    };
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let pumping_refs = PumpingRefs {
        col_pumping_start: 0,
        pumping_stations: &no_stations,
        pumping_pos: &empty_pumping_pos,
    };
    let state = StateLayout::new(2, 0, 0, Vec::new(), 0, 0, vec![], &[0, 0]);
    let geom = make_geom(&evap_indexer, &state, 1, &[]);
    let no_contracts: Vec<EnergyContract> = Vec::new();
    let empty_contract_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let contract_refs = ContractRefs {
        contracts: &no_contracts,
        contract_pos: &empty_contract_pos,
    };
    let result = resolve_variable_ref(
        &VariableRef::HydroEvaporation {
            hydro_id: EntityId(20),
            block_id: None,
        },
        0,
        0,
        &geom,
        &prod_models,
        &positions,
        &cascade_refs,
        &pumping_refs,
        &contract_refs,
    );

    assert!(result.is_empty());
}

/// At `K = 3`, `HydroEvaporation{None}` resolves to the single block-0 column (one
/// entry, NOT a sum over blocks); `Some(k)` selects distinct per-block columns and
/// an out-of-range block resolves to empty.
#[test]
fn hydro_evaporation_none_resolves_block_zero_not_sum() {
    let evap_indexer = crate::test_support::geometry(
        &crate::test_support::GeometryDims {
            hydro_count: 2,
            n_buses: 1,
            n_blks: 3,
            ..Default::default()
        },
        vec![],
        &[],
        vec![0],
    );

    let prod_models = ProductionModelSet::new(
        vec![
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
        ],
        2,
        1,
    );

    let hpos: BTreeMap<EntityId, usize> =
        [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
    let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    let positions = super::EntityPositionMaps {
        hydro: &hpos,
        thermal: &tpos,
        bus: &bpos,
        line: &lpos,
    };
    let cascade = empty_cascade();
    let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    let cascade_refs = CascadeRefs {
        cascade: &cascade,
        diversion_upstream: &diversion_upstream,
    };
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let pumping_refs = PumpingRefs {
        col_pumping_start: 0,
        pumping_stations: &no_stations,
        pumping_pos: &empty_pumping_pos,
    };
    let state = StateLayout::new(2, 0, 0, Vec::new(), 0, 0, vec![], &[0, 0]);
    let geom = make_geom(&evap_indexer, &state, 3, &[]);
    let no_contracts: Vec<EnergyContract> = Vec::new();
    let empty_contract_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let contract_refs = ContractRefs {
        contracts: &no_contracts,
        contract_pos: &empty_contract_pos,
    };

    let resolve = |block_id: Option<usize>| {
        resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
                block_id,
            },
            0,
            0,
            &geom,
            &prod_models,
            &positions,
            &cascade_refs,
            &pumping_refs,
            &contract_refs,
        )
    };

    let none = resolve(None);
    assert_eq!(
        none.len(),
        1,
        "None resolves to one column, not a K-block sum"
    );
    assert_eq!(none, resolve(Some(0)), "None resolves to block 0");
    assert_ne!(resolve(Some(0)), resolve(Some(1)));
    assert_ne!(resolve(Some(1)), resolve(Some(2)));
    assert!(
        resolve(Some(3)).is_empty(),
        "out-of-range block resolves to empty"
    );
}

// ── Pumping tests ─────────────────────────────────────────────────────────
//
// Shared layout: two stations id 10 (p_idx 0, consumption 2.5 MW/(m³/s)) and
// id 20 (p_idx 1, consumption 0.75), n_blks = 3, col_pumping_start = 100.
// Block-major column = col_pumping_start + p_idx * n_blks + blk.

const PUMP_COL_START: usize = 100;
const PUMP_N_BLKS: usize = 3;

/// Two pumping stations and the matching `pumping_pos`, in ID-sorted slot order.
fn make_pumping_fixture() -> (Vec<PumpingStation>, BTreeMap<EntityId, usize>) {
    let stations = vec![
        make_pumping_station(10, 2.5),
        make_pumping_station(20, 0.75),
    ];
    let pumping_pos: BTreeMap<EntityId, usize> =
        [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
    (stations, pumping_pos)
}

/// `PumpingFlow{station, Some(blk)}` → the block-major flow column × 1.0.
///
/// Station id 20 at p_idx 1, blk 2: col = 100 + 1*3 + 2 = 105, coeff 1.0.
#[test]
fn pumping_flow_resolves_to_flow_column_with_unit_coeff() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let (stations, ppos) = make_pumping_fixture();

    let result = call_pumping(
        VariableRef::PumpingFlow {
            station_id: EntityId(20),
            block_id: Some(2),
        },
        0, // block_idx — overridden by block_id = Some(2)
        &geom,
        &prod,
        PUMP_COL_START,
        PUMP_N_BLKS,
        &stations,
        &ppos,
    );

    assert_eq!(result, vec![(PUMP_COL_START + 1 * PUMP_N_BLKS + 2, 1.0)]);
}

/// `PumpingPower{station, Some(blk)}` → the SAME flow column × consumption.
///
/// Station id 10 at p_idx 0, blk 1: col = 100 + 0*3 + 1 = 101, coeff = 2.5.
/// The column is identical to `PumpingFlow` for the same (station, blk) — the
/// power term aliases the flow column, it is not a separate column.
#[test]
fn pumping_power_resolves_to_flow_column_with_consumption_coeff() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let (stations, ppos) = make_pumping_fixture();

    let blk = 1;
    let power = call_pumping(
        VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: Some(blk),
        },
        0,
        &geom,
        &prod,
        PUMP_COL_START,
        PUMP_N_BLKS,
        &stations,
        &ppos,
    );
    let flow = call_pumping(
        VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: Some(blk),
        },
        0,
        &geom,
        &prod,
        PUMP_COL_START,
        PUMP_N_BLKS,
        &stations,
        &ppos,
    );

    let expected_col = PUMP_COL_START + 0 * PUMP_N_BLKS + blk;
    assert_eq!(power, vec![(expected_col, 2.5)]);
    // Same column as flow — PumpingPower must alias, not allocate a new column.
    assert_eq!(power[0].0, flow[0].0);
}

/// `PumpingFlow{station, None}` with `block_idx = k` resolves the single
/// column for block `k` (`eff_blk = block_id.unwrap_or(block_idx)`), so the
/// caller's per-block loop yields one `(col, 1.0)` entry per block in order.
#[test]
fn pumping_flow_none_resolves_per_block() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let (stations, ppos) = make_pumping_fixture();

    // The caller iterates blocks and calls the resolver once per block; collect
    // those per-block resolutions to confirm one (col, 1.0) entry per block.
    let per_block: Vec<(usize, f64)> = (0..PUMP_N_BLKS)
        .map(|blk| {
            let r = call_pumping(
                VariableRef::PumpingFlow {
                    station_id: EntityId(10),
                    block_id: None,
                },
                blk, // block_idx supplies the effective block
                &geom,
                &prod,
                PUMP_COL_START,
                PUMP_N_BLKS,
                &stations,
                &ppos,
            );
            assert_eq!(r.len(), 1);
            r[0]
        })
        .collect();

    assert_eq!(
        per_block,
        vec![
            (PUMP_COL_START + 0, 1.0),
            (PUMP_COL_START + 1, 1.0),
            (PUMP_COL_START + 2, 1.0),
        ]
    );
}

/// `PumpingPower{station, None}` resolves to the per-block column × consumption.
///
/// Station id 20 at p_idx 1, consumption 0.75: per-block cols 103, 104, 105.
#[test]
fn pumping_power_none_resolves_per_block_with_consumption() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let (stations, ppos) = make_pumping_fixture();

    let per_block: Vec<(usize, f64)> = (0..PUMP_N_BLKS)
        .map(|blk| {
            let r = call_pumping(
                VariableRef::PumpingPower {
                    station_id: EntityId(20),
                    block_id: None,
                },
                blk,
                &geom,
                &prod,
                PUMP_COL_START,
                PUMP_N_BLKS,
                &stations,
                &ppos,
            );
            assert_eq!(r.len(), 1);
            r[0]
        })
        .collect();

    assert_eq!(
        per_block,
        vec![
            (PUMP_COL_START + 1 * PUMP_N_BLKS + 0, 0.75),
            (PUMP_COL_START + 1 * PUMP_N_BLKS + 1, 0.75),
            (PUMP_COL_START + 1 * PUMP_N_BLKS + 2, 0.75),
        ]
    );
}

/// Unknown station id resolves to `vec![]` for both pumping variants.
#[test]
fn pumping_unknown_station_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let (stations, ppos) = make_pumping_fixture();

    for var_ref in [
        VariableRef::PumpingFlow {
            station_id: EntityId(999),
            block_id: None,
        },
        VariableRef::PumpingPower {
            station_id: EntityId(999),
            block_id: Some(0),
        },
    ] {
        let result = call_pumping(
            var_ref,
            0,
            &geom,
            &prod,
            PUMP_COL_START,
            PUMP_N_BLKS,
            &stations,
            &ppos,
        );
        assert!(
            result.is_empty(),
            "unknown station must return empty vec, got: {result:?} for {var_ref:?}"
        );
    }
}

/// `n_pumping == 0` (no stations) resolves to `vec![]` — the empty `pumping_pos`
/// lookup misses before `col_pumping_start` is ever used.
#[test]
fn pumping_no_stations_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let no_stations: Vec<PumpingStation> = Vec::new();
    let empty_pos: BTreeMap<EntityId, usize> = BTreeMap::new();

    for var_ref in [
        VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: Some(0),
        },
        VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: None,
        },
    ] {
        let result = call_pumping(
            var_ref,
            0,
            &geom,
            &prod,
            PUMP_COL_START,
            PUMP_N_BLKS,
            &no_stations,
            &empty_pos,
        );
        assert!(
            result.is_empty(),
            "n_pumping == 0 must return empty vec, got: {result:?} for {var_ref:?}"
        );
    }
}

/// `PumpingFlow` and `PumpingPower` are block-DEPENDENT — per-block columns,
/// so the single-row collapse must NOT apply (they stay in the `false` arm).
#[test]
fn pumping_variants_are_block_dependent() {
    assert!(!variable_ref_is_block_independent(
        &VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: None,
        }
    ));
    assert!(!variable_ref_is_block_independent(
        &VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: None,
        }
    ));
}

/// `HydroStorage`, `HydroEvaporation`, and `AnticipatedDecision` are
/// block-INDEPENDENT — stage-level stock variables whose resolver ignores
/// `block_idx`, so the single-row collapse is sound. This is the `true`-arm
/// counterpart to `pumping_variants_are_block_dependent` /
/// `hydro_inflow_is_block_dependent`: dropping any of these three from the
/// `true` branch of `variable_ref_is_block_independent` would silently expand
/// a per-stage stock variable into per-block rows.
#[test]
fn block_independent_kinds_classify_true() {
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        }
    ));
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroEvaporation {
            hydro_id: EntityId(10),
            block_id: None,
        }
    ));
    assert!(variable_ref_is_block_independent(
        &VariableRef::AnticipatedDecision {
            thermal_id: EntityId(6),
        }
    ));
}

// ── Contract resolution tests ─────────────────────────────────────────────

/// `contract_family_slot` counts only same-direction contracts before `c_sys`.
/// Slice order: import(10), export(20), import(30), export(40) → import slots
/// 0,1 and export slots 0,1.
#[test]
fn contract_family_slot_counts_per_direction() {
    let contracts = vec![
        make_contract(10, ContractType::Import),
        make_contract(20, ContractType::Export),
        make_contract(30, ContractType::Import),
        make_contract(40, ContractType::Export),
    ];
    assert_eq!(
        contract_family_slot(&contracts, 0),
        (ContractType::Import, 0)
    );
    assert_eq!(
        contract_family_slot(&contracts, 1),
        (ContractType::Export, 0)
    );
    assert_eq!(
        contract_family_slot(&contracts, 2),
        (ContractType::Import, 1)
    );
    assert_eq!(
        contract_family_slot(&contracts, 3),
        (ContractType::Export, 1)
    );
}

/// AC: `ContractImport { contract_id, block_id: Some(0) }` resolves to exactly
/// one `(column, 1.0)` pair addressing the contract's block-0 column.
///
/// Two imports + one export: import base 200 (2 imports * n_blks=3 → cols
/// 200..206), export base 206 (1 export * 3 → 206..209). The second import
/// (id 30, per-family slot 1) at block 0 is `grid.flat(200, 1, 0) = 203`.
#[test]
fn contract_import_resolves_to_column_with_unit_coefficient() {
    let indexer = make_indexer();
    let state = make_state();
    let import_range = 200..206;
    let export_range = 206..209;
    let geom = make_geom_with_contracts(&indexer, &state, 2, &[], &import_range, &export_range);
    let prod = make_production_models();
    let contracts = vec![
        make_contract(10, ContractType::Import),
        make_contract(30, ContractType::Import),
        make_contract(20, ContractType::Export),
    ];
    let contract_pos: BTreeMap<EntityId, usize> = contracts
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    let result = call_contract(
        VariableRef::ContractImport {
            contract_id: EntityId(30),
            block_id: Some(0),
        },
        0,
        &geom,
        &prod,
        &contracts,
        &contract_pos,
    );

    // import base 200, per-family slot 1, n_blks 3, block 0 → 200 + 1*3 + 0
    assert_eq!(result, vec![(203, 1.0)]);
}

/// AC: a `ContractExport` ref resolves to exactly one `(column, 1.0)` pair on
/// the export family. The variable's own coefficient is `+1.0`; the
/// injection/withdrawal sign is owned by the load-balance fill, not here.
///
/// Same fixture as the import test: export id 20 is per-family slot 0,
/// `grid.flat(206, 0, 2) = 208`.
#[test]
fn contract_export_resolves_to_column_with_unit_coefficient() {
    let indexer = make_indexer();
    let state = make_state();
    let import_range = 200..206;
    let export_range = 206..209;
    let geom = make_geom_with_contracts(&indexer, &state, 2, &[], &import_range, &export_range);
    let prod = make_production_models();
    let contracts = vec![
        make_contract(10, ContractType::Import),
        make_contract(30, ContractType::Import),
        make_contract(20, ContractType::Export),
    ];
    let contract_pos: BTreeMap<EntityId, usize> = contracts
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    let result = call_contract(
        VariableRef::ContractExport {
            contract_id: EntityId(20),
            block_id: Some(2),
        },
        0,
        &geom,
        &prod,
        &contracts,
        &contract_pos,
    );

    // export base 206, per-family slot 0, n_blks 3, block 2 → 206 + 0*3 + 2
    assert_eq!(result, vec![(208, 1.0)]);
}

/// An unknown contract id misses `contract_pos` and resolves to empty — the
/// defense-in-depth fallback past referential validation.
#[test]
fn contract_unknown_id_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let import_range = 200..203;
    let export_range = 203..203;
    let geom = make_geom_with_contracts(&indexer, &state, 2, &[], &import_range, &export_range);
    let prod = make_production_models();
    let contracts = vec![make_contract(10, ContractType::Import)];
    let contract_pos: BTreeMap<EntityId, usize> = [(EntityId(10), 0)].into_iter().collect();

    let result = call_contract(
        VariableRef::ContractImport {
            contract_id: EntityId(99),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &contracts,
        &contract_pos,
    );

    assert!(result.is_empty());
}

// ── Stub entity tests ─────────────────────────────────────────────────────

/// NonControllableGeneration returns empty vec.
#[test]
fn non_controllable_generation_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::NonControllableGeneration {
            source_id: EntityId(7),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(result.is_empty());
}

/// `HydroWithdrawal` resolves to an empty vec: withdrawal carries no LP
/// decision column (a schedule fixed by bounds, not a decision variable), so
/// a generic-constraint term referencing it contributes no `(column,
/// coefficient)` pair — the deliberate stub contract documented above the
/// `resolve_variable_ref` stub arm. `EntityId(999)` is in no `hydro_pos`
/// entry, confirming the empty return is unconditional, not a missing-id
/// fall-through.
#[test]
fn hydro_withdrawal_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroWithdrawal {
            hydro_id: EntityId(999),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(
        result,
        Vec::<(usize, f64)>::new(),
        "HydroWithdrawal must resolve to no column (no-LP-column stub contract)"
    );
}

/// `NonControllableCurtailment` resolves to an empty vec: non-controllable
/// sources carry no decision column, so a generic-constraint term referencing
/// curtailment contributes no `(column, coefficient)` pair — the same
/// deliberate stub contract as `NonControllableGeneration`. `EntityId(999)` is
/// in no position map, confirming the empty return is unconditional, not a
/// missing-id fall-through.
#[test]
fn non_controllable_curtailment_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::NonControllableCurtailment {
            source_id: EntityId(999),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(
        result,
        Vec::<(usize, f64)>::new(),
        "NonControllableCurtailment must resolve to no column (no-LP-column stub contract)"
    );
}

// ── Missing entity ID test ─────────────────────────────────────────────────

/// missing entity ID returns empty vec (defense-in-depth).
#[test]
fn missing_entity_id_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    // EntityId(999) is not in thermal_pos
    let result = call(
        VariableRef::ThermalGeneration {
            thermal_id: EntityId(999),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(result.is_empty());
}

// ── BusDeficit tests ──────────────────────────────────────────────────────

/// BusDeficit with S=2 deficit segments returns 2 column entries.
///
/// bus_pos[EntityId(100)] = 0, deficit.start = 61, max_deficit_segments = 2,
/// n_blks = 3, block_idx = 0
/// Expected: [(61 + 0*2*3 + 0*3 + 0, 1.0), (61 + 0*2*3 + 1*3 + 0, 1.0)]
///         = [(61, 1.0), (64, 1.0)]
#[test]
fn bus_deficit_returns_one_entry_per_segment() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::BusDeficit {
            bus_id: EntityId(100),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    // deficit.start=61, b_pos=0, S=2, n_blks=3, blk=0
    // seg0: 61 + 0*2*3 + 0*3 + 0 = 61
    // seg1: 61 + 0*2*3 + 1*3 + 0 = 64
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (61, 1.0));
    assert_eq!(result[1], (64, 1.0));
}

/// BusDeficit for second bus (position 1) at block 1.
///
/// bus_pos[EntityId(200)] = 1, deficit.start = 61, S = 2, n_blks = 3, blk = 1
/// seg0: 61 + 1*2*3 + 0*3 + 1 = 61 + 6 + 0 + 1 = 68
/// seg1: 61 + 1*2*3 + 1*3 + 1 = 61 + 6 + 3 + 1 = 71
#[test]
fn bus_deficit_second_bus_block_1() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::BusDeficit {
            bus_id: EntityId(200),
            block_id: None,
        },
        1,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], (68, 1.0));
    assert_eq!(result[1], (71, 1.0));
}

// ── BusExcess tests ───────────────────────────────────────────────────────

/// BusExcess maps to the excess column for the bus.
///
/// bus_pos[EntityId(100)] = 0, excess.start = 73, n_blks = 3, block = 2
/// Expected: [(73 + 0*3 + 2, 1.0)] = [(75, 1.0)]
#[test]
fn bus_excess_maps_to_excess_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::BusExcess {
            bus_id: EntityId(100),
            block_id: None,
        },
        2,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(73 + 0 * 3 + 2, 1.0)]);
}

// ── LineDirect / LineReverse tests ────────────────────────────────────────

/// LineDirect maps to line_fwd column.
///
/// line_pos[EntityId(50)] = 0, line_fwd.start = 55, n_blks = 3, block = 1
/// Expected: [(55 + 0*3 + 1, 1.0)] = [(56, 1.0)]
#[test]
fn line_direct_maps_to_fwd_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::LineDirect {
            line_id: EntityId(50),
            block_id: None,
        },
        1,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(55 + 0 * 3 + 1, 1.0)]);
}

/// LineReverse maps to line_rev column.
///
/// line_pos[EntityId(50)] = 0, line_rev.start = 58, n_blks = 3, block = 0
/// Expected: [(58 + 0*3 + 0, 1.0)] = [(58, 1.0)]
#[test]
fn line_reverse_maps_to_rev_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::LineReverse {
            line_id: EntityId(50),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(58, 1.0)]);
}

// ── LineExchange tests ──────────────────────────────────────────────────────

/// LineExchange maps to both line_fwd and line_rev columns with opposite signs.
///
/// line_pos[EntityId(50)] = 0, line_fwd.start = 55, line_rev.start = 58,
/// n_blks = 3, block = 1
/// Expected: [(55 + 0*3 + 1, 1.0), (58 + 0*3 + 1, -1.0)] = [(56, 1.0), (59, -1.0)]
#[test]
fn line_exchange_maps_to_fwd_and_rev_columns() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(50),
            block_id: None,
        },
        1,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(56, 1.0), (59, -1.0)]);
}

/// LineExchange with explicit block_id overrides current block_idx.
///
/// block_idx = 2 but block_id = Some(0)
/// Expected: [(55, 1.0), (58, -1.0)]
#[test]
fn line_exchange_with_explicit_block() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(50),
            block_id: Some(0),
        },
        2,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(55, 1.0), (58, -1.0)]);
}

/// LineExchange with unknown line ID returns empty vec.
#[test]
fn line_exchange_unknown_id_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::LineExchange {
            line_id: EntityId(999),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(result.is_empty());
}

// ── AnticipatedDecision tests ─────────────────────────────────────────────

/// Build a `StageGeometry` with 2 thermals where thermal at system position 1
/// is anticipated (local anticipated index 0).
///
/// N=0 hydros, T=2 thermals (pos 0 = regular, pos 1 = anticipated), Ln=0,
/// B=1 bus, K=2 blocks, n_anticipated=1, k_max=2.
///
/// Column layout (no hydros, no FPHA, no evap):
///   storage:               [0, 0)    empty
///   lags:                  [0, 0)    empty
///   anticipated_slots_out: [0, 0 + 2*1) = [0, 2)  (k_max=2, n_anticipated=1, outgoing)
///   z_inflow:              [2, 2)    empty
///   storage_in:            [2, 2)    empty
///   anticipated_state:     [2, 4)    (incoming, relocated)
///   theta:                 4
///   decision_start:        5
///   thermal:               [5, 5 + 2*2) = [5, 9)  (T=2, K=2)
///   anticipated_decision:  [9, 9+1) = [9, 10)   (n_anticipated=1)
///   line_fwd: [10, 10) empty
///   line_rev: [10, 10) empty
///   deficit: [10, 10+1*1*2) = [10, 12)  (B=1, S=1, K=2)
///   excess:  [12, 12+1*2) = [12, 14)
fn make_indexer_with_anticipated() -> StageGeometry {
    crate::test_support::geometry(
        &crate::test_support::GeometryDims {
            n_thermals: 2,
            n_buses: 1,
            n_blks: 2,
            n_anticipated: 1,
            k_max: 2,
            anticipated_thermal_indices: vec![1], // sys pos 1 is anticipated
            ..Default::default()
        },
        vec![],
        &[],
        vec![],
    )
}

/// AC-12: `AnticipatedDecision` for an anticipated thermal maps to the
/// correct stage-level column: `anticipated_decision.start + local_idx`.
///
/// Using `make_indexer_with_anticipated`:
/// - Thermal EntityId(6) at sys_pos=1, which is anticipated_thermal_indices[0].
/// - anticipated_decision.start = 9, local_idx = 0.
/// - Expected column = 9 + 0 = 9.
#[test]
fn anticipated_decision_maps_to_correct_column() {
    let indexer = make_indexer_with_anticipated();
    let state = make_state_anticipated();
    // `make_indexer_with_anticipated` places the anticipated plant at system
    // position 1 (local index 0), matching the reverse map the production
    // `StageLayout` builds.
    let geom = make_geom(&indexer, &state, 1, &[1]);
    let prod = ProductionModelSet::new(vec![], 0, 1);
    let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let tpos: BTreeMap<EntityId, usize> =
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    // Verify anticipated_decision.start is as expected.
    assert_eq!(
        indexer.anticipated_decision.start, 9,
        "anticipated_decision.start should be 9, got {}",
        indexer.anticipated_decision.start
    );

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(6), // sys_pos=1, local anticipated idx=0
        },
        0, // block_idx is ignored for stage-level variable
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(
        result,
        vec![(9, 1.0)],
        "AnticipatedDecision(6) should resolve to column 9 (anticipated_decision.start + 0)"
    );
}

/// AC-12 (block-independence): `AnticipatedDecision` is stage-level — the
/// returned column is the same regardless of `block_idx`.
#[test]
fn anticipated_decision_ignores_block_idx() {
    let indexer = make_indexer_with_anticipated();
    let state = make_state_anticipated();
    // `make_indexer_with_anticipated` places the anticipated plant at system
    // position 1 (local index 0).
    let geom = make_geom(&indexer, &state, 1, &[1]);
    let prod = ProductionModelSet::new(vec![], 0, 1);
    let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let tpos: BTreeMap<EntityId, usize> =
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    for block_idx in [0, 1] {
        let result = call(
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(6),
            },
            block_idx,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        assert_eq!(
            result,
            vec![(9, 1.0)],
            "AnticipatedDecision must be stage-level (block_idx={block_idx} should not change column)"
        );
    }
}

/// AC-13: `AnticipatedDecision` for a regular (non-anticipated) thermal
/// returns an empty vec (defense-in-depth).
///
/// Thermal EntityId(5) at sys_pos=0 is NOT in anticipated_thermal_indices.
#[test]
fn anticipated_decision_non_anticipated_thermal_returns_empty() {
    let indexer = make_indexer_with_anticipated();
    let state = make_state_anticipated();
    // Anticipated plant is at system position 1; querying a different thermal
    // must miss the populated reverse map and return empty.
    let geom = make_geom(&indexer, &state, 1, &[1]);
    let prod = ProductionModelSet::new(vec![], 0, 1);
    let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let tpos: BTreeMap<EntityId, usize> =
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5), // sys_pos=0, NOT anticipated
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(
        result.is_empty(),
        "AnticipatedDecision for non-anticipated thermal must return empty vec, got: {result:?}"
    );
}

/// AC-14: `AnticipatedDecision` for an unknown entity ID returns empty vec.
#[test]
fn anticipated_decision_unknown_entity_returns_empty() {
    let indexer = make_indexer_with_anticipated();
    let state = make_state_anticipated();
    // Anticipated plant is at system position 1 (populated reverse map).
    let geom = make_geom(&indexer, &state, 1, &[1]);
    let prod = ProductionModelSet::new(vec![], 0, 1);
    let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let tpos: BTreeMap<EntityId, usize> =
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
    let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    let result = call(
        VariableRef::AnticipatedDecision {
            thermal_id: EntityId(999), // unknown
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(
        result.is_empty(),
        "AnticipatedDecision for unknown entity must return empty vec, got: {result:?}"
    );
}

// ── block_col_range tests ─────────────────────────────────────────────────

/// Each equipment/line family maps to its matching `StageGeometry` range, and
/// the two contract families map to their `geom.contract_import` /
/// `contract_export` ranges. This pins the family↔range pairing the resolver's
/// `col_start` reads depend on.
#[test]
fn block_col_range_maps_each_family_to_its_geometry_range() {
    let idx = make_indexer();
    let state = make_state();
    let import_range = 200..206;
    let export_range = 206..209;
    let geom = make_geom_with_contracts(&idx, &state, 2, &[], &import_range, &export_range);

    assert_eq!(block_col_range(&geom, ElementKind::Turbine), idx.turbine);
    assert_eq!(block_col_range(&geom, ElementKind::Spillage), idx.spillage);
    assert_eq!(
        block_col_range(&geom, ElementKind::Diversion),
        idx.diversion
    );
    assert_eq!(block_col_range(&geom, ElementKind::Thermal), idx.thermal);
    assert_eq!(block_col_range(&geom, ElementKind::LineFwd), idx.line_fwd);
    assert_eq!(block_col_range(&geom, ElementKind::LineRev), idx.line_rev);
    assert_eq!(block_col_range(&geom, ElementKind::Excess), idx.excess);

    assert_eq!(
        block_col_range(&geom, ElementKind::ContractImport),
        import_range
    );
    assert_eq!(
        block_col_range(&geom, ElementKind::ContractExport),
        export_range
    );
}

// ── HydroTurbined / HydroSpillage tests ───────────────────────────────────

/// HydroTurbined maps to turbine column.
#[test]
fn hydro_turbined_maps_to_turbine_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    // hydro pos=1 (EntityId 20), turbine.start=13, n_blks=3, block=2
    let result = call(
        VariableRef::HydroTurbined {
            hydro_id: EntityId(20),
            block_id: None,
        },
        2,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(13 + 1 * 3 + 2, 1.0)]);
}

/// HydroSpillage maps to spillage column.
#[test]
fn hydro_spillage_maps_to_spillage_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    // hydro pos=3 (EntityId 40), spillage.start=25, n_blks=3, block=1
    let result = call(
        VariableRef::HydroSpillage {
            hydro_id: EntityId(40),
            block_id: None,
        },
        1,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(25 + 3 * 3 + 1, 1.0)]);
}

/// HydroDiversion maps to the diversion column.
///
/// Routes through `resolve_block_variable` with
/// `block_col_range(geom, ElementKind::Diversion).start = 37`. For hydro
/// pos=1 (EntityId 20), n_blks=3, block=2 the flat block-major address is
/// `37 + 1*3 + 2 = 42` with the unit coefficient.
#[test]
fn diversion_maps_to_diversion_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroDiversion {
            hydro_id: EntityId(20),
            block_id: None,
        },
        2,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert_eq!(result, vec![(37 + 1 * 3 + 2, 1.0)]);
}

// ── HydroInflow tests ──────────────────────────────────────────────────────

/// Cascade for the total-inflow tests: EntityId(10) and EntityId(20) both
/// flow into EntityId(40), so `upstream(40) = [10, 20]` (ID-sorted). The
/// three hydros map to system positions 0, 1, 3 via `make_hydro_pos`.
fn make_inflow_cascade() -> CascadeTopology {
    CascadeTopology::build(&[
        make_hydro(10, Some(40)),
        make_hydro(20, Some(40)),
        make_hydro(30, None),
        make_hydro(40, None),
    ])
}

/// AC: a two-upstream hydro (no diversion-into) resolves at block `k` to the
/// incremental `z_inflow` column plus, in canonical upstream order, each
/// upstream plant's turbine + spillage column. All coefficients `+1.0`.
///
/// Target EntityId(40) at pos_h=3; upstream [10, 20] at pos 0, 1.
/// z_inflow.start=4 → (4+3, 1.0)=(7, 1.0); turbine.start=13, spillage.start=25,
/// n_blks=3, k=2.
#[test]
fn hydro_inflow_two_upstream_canonical_order() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();
    let cascade = make_inflow_cascade();
    let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

    let blk = 2;
    let result = call_with_cascade(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0, // block_idx — overridden by block_id = Some(blk)
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
        &cascade,
        &div,
    );

    let z_col = 4 + 3; // z_inflow.start + pos_h
    let turb = 13; // turbine.start
    let spill = 25; // spillage.start
    let nb = 3; // n_blks
    assert_eq!(
        result,
        vec![
            (z_col, 1.0),
            (turb + 0 * nb + blk, 1.0),  // upstream 10 turbine
            (spill + 0 * nb + blk, 1.0), // upstream 10 spillage
            (turb + 1 * nb + blk, 1.0),  // upstream 20 turbine
            (spill + 1 * nb + blk, 1.0), // upstream 20 spillage
        ]
    );
}

/// AC: `block_id = None` with `block_idx = k` matches `block_id = Some(k)`
/// (the resolver uses `eff_blk = block_id.unwrap_or(block_idx)`).
#[test]
fn hydro_inflow_none_matches_some_block_idx() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();
    let cascade = make_inflow_cascade();
    let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

    let blk = 2;
    let none_result = call_with_cascade(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: None,
        },
        blk, // block_idx supplies the effective block
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
        &cascade,
        &div,
    );
    let some_result = call_with_cascade(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
        &cascade,
        &div,
    );

    assert_eq!(none_result, some_result);
}

/// AC: a hydro that also has a plant diverting into it gets the diversion
/// column appended after the upstream terms.
///
/// `diversion_upstream[40] = [2]` (system index 2). diversion.start=37,
/// n_blks=3, k=1 → (37 + 2*3 + 1, 1.0) = (44, 1.0).
#[test]
fn hydro_inflow_diversion_into_appends_diversion_column() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();
    let cascade = make_inflow_cascade();
    let div: HashMap<EntityId, Vec<usize>> = [(EntityId(40), vec![2])].into_iter().collect();

    let blk = 1;
    let result = call_with_cascade(
        VariableRef::HydroInflow {
            hydro_id: EntityId(40),
            block_id: Some(blk),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
        &cascade,
        &div,
    );

    let z_col = 4 + 3;
    let turb = 13; // turbine.start
    let spill = 25; // spillage.start
    let div_start = 37; // diversion.start
    let nb = 3; // n_blks
    assert_eq!(
        result,
        vec![
            (z_col, 1.0),
            (turb + 0 * nb + blk, 1.0),
            (spill + 0 * nb + blk, 1.0),
            (turb + 1 * nb + blk, 1.0),
            (spill + 1 * nb + blk, 1.0),
            (div_start + 2 * nb + blk, 1.0), // diversion-into, system index 2
        ]
    );
}

/// AC: a headwater hydro (no upstream, no diversion-into) resolves to exactly
/// the incremental `z_inflow` column.
///
/// EntityId(30) at pos=2 is a headwater in `make_inflow_cascade`.
/// z_inflow.start=4 → (6, 1.0).
#[test]
fn hydro_inflow_headwater_resolves_to_z_inflow_only() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();
    let cascade = make_inflow_cascade();
    let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

    for block_idx in [0, 1, 2] {
        let result = call_with_cascade(
            VariableRef::HydroInflow {
                hydro_id: EntityId(30),
                block_id: None,
            },
            block_idx,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
            &cascade,
            &div,
        );
        assert_eq!(result, vec![(6, 1.0)], "block_idx={block_idx}");
    }
}

/// AC: `hydro_count == 0` (empty `z_inflow`) resolves to `vec![]`.
///
/// `make_indexer_with_anticipated` has no hydros, so `z_inflow` is empty and
/// `z_inflow.start` is meaningless; the resolver must short-circuit to `[]`.
#[test]
fn hydro_inflow_empty_when_no_hydros() {
    let indexer = make_indexer_with_anticipated();
    let state = make_state_anticipated();
    // Keep `study_dims` faithful to the indexer's anticipated plant at
    // system position 1, though this test exercises only the hydro path.
    let geom = make_geom(&indexer, &state, 1, &[1]);
    let prod = ProductionModelSet::new(vec![], 0, 1);
    let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let bpos: BTreeMap<EntityId, usize> = BTreeMap::new();
    let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

    assert!(
        state.z_inflow.is_empty(),
        "z_inflow must be empty with hydro_count == 0"
    );

    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(0),
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(
        result.is_empty(),
        "HydroInflow with hydro_count == 0 must return empty vec, got: {result:?}"
    );
}

/// AC: an unknown `hydro_id` resolves to `vec![]`.
#[test]
fn hydro_inflow_unknown_id_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_geom(&indexer, &state, 2, &[]);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let result = call(
        VariableRef::HydroInflow {
            hydro_id: EntityId(999), // unknown
            block_id: None,
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );

    assert!(
        result.is_empty(),
        "HydroInflow for unknown id must return empty vec, got: {result:?}"
    );
}

/// AC: `HydroInflow` is block-DEPENDENT — its upstream releases are per-block
/// columns, so the single-row collapse must NOT apply.
#[test]
fn hydro_inflow_is_block_dependent() {
    assert!(!variable_ref_is_block_independent(
        &VariableRef::HydroInflow {
            hydro_id: EntityId(0),
            block_id: None,
        }
    ));
}

// ── Per-block storage boundary tests ──────────────────────────────────────
//
// A K=3 chronological geom: `make_indexer` (n_blks=3) + `make_state` (N=4,
// storage=[0,4), storage_in.start=8), overriding `storage_internal_start` to a
// non-zero interior anchor so the interior-boundary formula is exercised (the
// `test_support::geometry` fixture hardcodes it to 0). Interior stride is
// `n_blks - 1 = 2`; for hydro pos 0 the boundaries are S⁰=8, S¹=12, S²=13, S³=0.

const STORAGE_INTERNAL_START: usize = 12;

/// Build the K=3 chronological geom with a non-zero `storage_internal_start`.
fn make_chronological_geom<'a>(
    indexer: &'a StageGeometry,
    state: &'a StateLayout,
) -> GenericResolverGeom<'a> {
    let mut geom = make_geom(indexer, state, 2, &[]);
    geom.storage_internal_start = STORAGE_INTERNAL_START;
    geom
}

/// Resolve every boundary (`k = 0`, interior, `k = K`) of both variants on a K=3
/// chronological geom against the mirrored `block_storage_col`.
#[test]
fn hydro_storage_boundary_resolves_each_boundary() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_chronological_geom(&indexer, &state);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    // Hydro EntityId(10) at pos 0; K = 3.
    // Initial{0} = S⁰ = storage_in.start + 0 = 8.
    let initial_0 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(0),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    assert_eq!(initial_0, vec![(geom.block_storage_col(0, 0), 1.0)]);
    assert_eq!(initial_0, vec![(state.storage_in.start + 0, 1.0)]);

    // Initial{1} = S¹ = storage_internal_start + 0 = 12 (interior boundary).
    let initial_1 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    assert_eq!(initial_1, vec![(geom.block_storage_col(0, 1), 1.0)]);
    assert_eq!(initial_1, vec![(STORAGE_INTERNAL_START, 1.0)]);

    // Final{2} = S³ = Sᴷ = storage.start + 0 = 0 (K=3, last block).
    let final_2 = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(2),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    assert_eq!(final_2, vec![(geom.block_storage_col(0, 3), 1.0)]);
    assert_eq!(final_2, vec![(state.storage.start + 0, 1.0)]);
}

/// `HydroStorageFinal{K-1}` resolves to the SAME column as `HydroStorage` (Sᴷ).
#[test]
fn hydro_storage_final_last_block_equals_hydro_storage() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_chronological_geom(&indexer, &state);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let final_last = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(2),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    let storage = call(
        VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    assert_eq!(final_last, storage);
    assert_eq!(final_last, vec![(state.storage.start + 0, 1.0)]);
}

/// A block's final boundary is the next block's initial: `Final{0}` and
/// `Initial{1}` both resolve to the interior column `S¹`.
#[test]
fn hydro_storage_final_shares_interior_column_with_next_initial() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_chronological_geom(&indexer, &state);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    let final_0 = call(
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(10),
            block_id: Some(0),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    let initial_1 = call(
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(10),
            block_id: Some(1),
        },
        0,
        &geom,
        &prod,
        &hpos,
        &tpos,
        &bpos,
        &lpos,
    );
    assert_eq!(final_0, initial_1);
    assert_eq!(final_0, vec![(STORAGE_INTERNAL_START, 1.0)]);
}

/// `block_id = None` resolves to the fixed stage endpoint — `S⁰` for initial,
/// `Sᴷ` for final — independent of the caller's `block_idx`.
#[test]
fn hydro_storage_boundary_none_resolves_stage_endpoint() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_chronological_geom(&indexer, &state);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    for blk in 0..3 {
        let initial = call(
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(10),
                block_id: None,
            },
            blk,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        assert_eq!(initial, vec![(geom.block_storage_col(0, 0), 1.0)]);

        let final_ = call(
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(10),
                block_id: None,
            },
            blk,
            &geom,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        assert_eq!(final_, vec![(geom.block_storage_col(0, 3), 1.0)]);
    }
}

/// A `hydro_pos` miss resolves to an empty vec for both boundary variants.
#[test]
fn hydro_storage_boundary_unknown_id_returns_empty() {
    let indexer = make_indexer();
    let state = make_state();
    let geom = make_chronological_geom(&indexer, &state);
    let prod = make_production_models();
    let hpos = make_hydro_pos();
    let tpos = make_thermal_pos();
    let bpos = make_bus_pos();
    let lpos = make_line_pos();

    for var_ref in [
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId(999),
            block_id: Some(1),
        },
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId(999),
            block_id: None,
        },
    ] {
        let result = call(var_ref, 0, &geom, &prod, &hpos, &tpos, &bpos, &lpos);
        assert!(result.is_empty(), "unknown id must resolve to empty vec");
    }
}

/// Both storage boundary variants resolve to a fixed column (a stage endpoint or a
/// named boundary), so they are block-INDEPENDENT (`true`) for `None` and `Some`
/// alike, like the stage-final alias `HydroStorage`.
#[test]
fn storage_boundary_variants_are_block_independent() {
    for block_id in [None, Some(1)] {
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroStorageInitial {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroStorageFinal {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
    }
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        }
    ));
}
