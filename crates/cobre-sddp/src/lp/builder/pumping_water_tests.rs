#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, CascadeTopology, CoefficientRef,
    ConstraintExpression, ConstraintSense, ContractStageBounds, DeficitSegment, EntityId,
    GenericConstraint, Hydro, HydroGenerationModel, HydroPenalties, HydroStageBounds,
    LineStageBounds, LinearTerm, NoiseMethod, PumpingStageBounds, PumpingStation, ResolvedBounds,
    ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
    ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, SlackConfig,
    Stage, StageRiskConfig, StageStateConfig, ThermalStageBounds, VariableRef,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{EvaporationModel, EvaporationModelSet, ProductionModelSet};
use crate::resolved_parameters::ResolvedParameters;

use super::M3S_TO_HM3;
use super::columns::{ColumnBufs, fill_pumping_columns};
use super::entries::{
    LpMatrixBuffers, assemble_csc, build_stage_matrix_entries, fill_generic_constraint_entries,
    fill_load_balance_entries, fill_pumping_water_entries,
};
use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};

const N_STAGES: usize = 1;

/// Minimal independent (no-downstream) constant-productivity hydro.
fn fixture_hydro(id: i32) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        bus_id: EntityId(1),
        downstream_id: None,
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
        penalties: zero_hydro_penalties(),
    }
}

fn zero_hydro_penalties() -> HydroPenalties {
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

fn default_bounds_defaults() -> BoundsDefaults {
    BoundsDefaults {
        hydro: HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
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
    }
}

/// Build a two-block `Stage` with distinct block durations so a τ that
/// confuses the block index is observable.
fn two_block_stage() -> Stage {
    Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![
            Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 300.0,
            },
            Block {
                index: 1,
                name: "BLK1".to_string(),
                duration_hours: 444.0,
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
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// A single bus with one unbounded deficit segment, on `EntityId(1)` (the bus
/// the fixture hydros and `station` helper reference).
fn fixture_bus(id: i32) -> Bus {
    Bus {
        id: EntityId(id),
        name: format!("B{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 1.0,
    }
}

/// A pumping station on `EntityId(1)` with explicit source/destination/flow
/// data and the default `consumption_mw_per_m3s` (0.5).
fn station(id: i32, source: i32, destination: i32, min_flow: f64, max_flow: f64) -> PumpingStation {
    station_full(id, source, destination, min_flow, max_flow, 1, 0.5)
}

/// A pumping station with an explicit `bus_id` and consumption rate so the
/// power-coupling tests can place a station on an unmapped bus and observe a
/// distinct coefficient.
fn station_full(
    id: i32,
    source: i32,
    destination: i32,
    min_flow: f64,
    max_flow: f64,
    bus_id: i32,
    consumption_mw_per_m3s: f64,
) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: format!("P{id}"),
        bus_id: EntityId(bus_id),
        source_hydro_id: EntityId(source),
        destination_hydro_id: EntityId(destination),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s,
        min_flow_m3s: min_flow,
        max_flow_m3s: max_flow,
    }
}

/// Owns the data backing a two-hydro `TemplateBuildCtx` carrying pumping
/// stations. Hydros and stations are stored in canonical (ID-sorted) order;
/// `hydro_pos`/`pumping_pos` are derived from those sorted slices, exactly as
/// `SystemBuilder::build` produces them in production.
struct PumpFixtures {
    hydros: Vec<Hydro>,
    stations: Vec<PumpingStation>,
    buses: Vec<Bus>,
    hydro_pos: HashMap<EntityId, usize>,
    pumping_pos: HashMap<EntityId, usize>,
    bus_pos: HashMap<EntityId, usize>,
    par_lp: PrecomputedPar,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_exchange_factors: ResolvedExchangeFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
    /// Generic constraints whose expressions the LP builder resolves. Empty by
    /// default; the end-to-end test sets a pumping-referencing constraint so the
    /// `PumpingFlow`/`PumpingPower` resolver arms run through the real caller.
    generic_constraints: Vec<GenericConstraint>,
}

impl PumpFixtures {
    /// Build a fixture with a single bus (`EntityId(1)`) — the bus the fixture
    /// hydros and the `station` helper reference — so the load-balance row is
    /// present and the pumping-power coupling is exercised.
    fn new(hydros: Vec<Hydro>, stations: Vec<PumpingStation>) -> Self {
        Self::new_with_buses(hydros, stations, vec![fixture_bus(1)])
    }

    /// Build a fixture from hydros, stations, and buses supplied in arbitrary
    /// declaration order. All three are sorted by `id.0` (the canonical
    /// operation `SystemBuilder::build` performs) before deriving position maps
    /// and the pumping bounds table, so the resulting ctx is
    /// declaration-order-invariant.
    fn new_with_buses(
        mut hydros: Vec<Hydro>,
        mut stations: Vec<PumpingStation>,
        mut buses: Vec<Bus>,
    ) -> Self {
        hydros.sort_by_key(|h| h.id.0);
        stations.sort_by_key(|s| s.id.0);
        buses.sort_by_key(|b| b.id.0);

        let hydro_pos: HashMap<EntityId, usize> =
            hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
        let pumping_pos: HashMap<EntityId, usize> = stations
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();
        let bus_pos: HashMap<EntityId, usize> =
            buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: hydros.len(),
                n_thermals: 0,
                n_lines: 0,
                n_pumping: stations.len(),
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &default_bounds_defaults(),
        );
        // Distinct per-station bounds so a column/bound mismatch is observable.
        for (p_idx, s) in stations.iter().enumerate() {
            for stage_idx in 0..N_STAGES {
                *bounds.pumping_bounds_mut(p_idx, stage_idx) = PumpingStageBounds {
                    min_flow_m3s: s.min_flow_m3s,
                    max_flow_m3s: s.max_flow_m3s,
                };
            }
        }

        let production_models = ProductionModelSet::new(
            vec![
                vec![
                    crate::hydro_models::ResolvedProductionModel::ConstantProductivity {
                        productivity: 1.0,
                    };
                    N_STAGES
                ];
                hydros.len()
            ],
            hydros.len(),
            N_STAGES,
        );
        let evaporation_models =
            EvaporationModelSet::new(vec![EvaporationModel::None; hydros.len()]);

        Self {
            hydros,
            stations,
            buses,
            hydro_pos,
            pumping_pos,
            bus_pos,
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            bounds,
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_exchange_factors: ResolvedExchangeFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
            },
            production_models,
            evaporation_models,
            generic_constraints: Vec::new(),
        }
    }

    /// Attach a generic constraint (and its active-at-stage-0 bound) so the
    /// LP builder resolves the constraint's expression against the pumping
    /// columns. Used by the end-to-end resolver-integration test.
    fn with_generic_constraint(mut self, constraint: GenericConstraint, bound: f64) -> Self {
        let constraint_id = constraint.id.0;
        let id_map: HashMap<i32, usize> = [(constraint_id, 0)].into_iter().collect();
        let rows = (0..N_STAGES).map(|s| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            (constraint_id, s as i32, None, bound)
        });
        self.resolved_generic_bounds =
            ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
        self.generic_constraints = vec![constraint];
        self
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &self.buses,
            load_models: &[],
            cascade: &self.cascade,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_exchange_factors: &self.resolved_exchange_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: self.hydro_pos.clone(),
            thermal_pos: HashMap::new(),
            line_pos: HashMap::new(),
            bus_pos: self.bus_pos.clone(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &self.generic_constraints,
            non_controllable_sources: &[],
            pumping_stations: &self.stations,
            pumping_pos: self.pumping_pos.clone(),
            n_pumping: self.stations.len(),
            diversion_upstream: HashMap::new(),
            n_hydros: self.hydros.len(),
            n_thermals: 0,
            n_lines: 0,
            n_buses: self.buses.len(),
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            has_penalty: false,
            cumulative_discount_factors: vec![1.0; N_STAGES],
            total_hours_per_stage: vec![744.0; N_STAGES],
        }
    }
}

/// Column bounds = `[min_flow, max_flow]` and zero objective for every
/// `(station, block)` pumping column.
#[test]
fn pumping_columns_get_flow_bounds_and_zero_cost() {
    let stations = vec![station(10, 1, 2, 5.0, 80.0), station(20, 2, 1, 0.0, 30.0)];
    let fixtures = PumpFixtures::new(vec![fixture_hydro(1), fixture_hydro(2)], stations);
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    // Lower/upper start at a NaN sentinel so any column the helper fails to
    // bound is visible; objective starts at the production default (0.0), so
    // the post-fill zero assertion proves the helper writes no cost.
    let mut col_lower = vec![f64::NAN; layout.num_cols];
    let mut col_upper = vec![f64::NAN; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];
    let mut bufs = ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };

    fill_pumping_columns(&ctx, 0, &layout, &mut bufs);

    let n_blks = layout.n_blks;
    for (p_idx, s) in ctx.pumping_stations.iter().enumerate() {
        for blk in 0..n_blks {
            let col = layout.col_pumping_start + p_idx * n_blks + blk;
            assert_eq!(
                bufs.col_lower[col], s.min_flow_m3s,
                "station {p_idx} blk {blk}: lower bound must be min_flow"
            );
            assert_eq!(
                bufs.col_upper[col], s.max_flow_m3s,
                "station {p_idx} blk {blk}: upper bound must be max_flow"
            );
            assert_eq!(
                bufs.objective[col], 0.0,
                "station {p_idx} blk {blk}: objective must be zero"
            );
        }
    }
}

/// Source water row gains `+tau_h`, destination water row gains `−tau_h`,
/// with `tau_h == block.duration_hours * M3S_TO_HM3` per block.
#[test]
fn pumping_water_entries_source_plus_tau_destination_minus_tau() {
    // Station id 10: source hydro id 1 (pos 0), destination hydro id 2 (pos 1).
    let fixtures = PumpFixtures::new(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station(10, 1, 2, 0.0, 50.0)],
    );
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

    let n_blks = layout.n_blks;
    let source_pos = ctx.hydro_pos[&EntityId(1)];
    let dest_pos = ctx.hydro_pos[&EntityId(2)];
    let row_source = layout.row_water_balance_start() + source_pos;
    let row_dest = layout.row_water_balance_start() + dest_pos;

    for blk in 0..n_blks {
        let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
        let col = layout.col_pumping_start + blk;
        assert_eq!(
            col_entries[col],
            vec![(row_source, tau_h), (row_dest, -tau_h)],
            "blk {blk}: source +tau_h then destination -tau_h"
        );
    }
}

/// A station whose `source_hydro_id` is absent from `hydro_pos` skips only
/// the source entry — the destination side is still written, no panic.
#[test]
fn pumping_water_entries_missing_source_skips_only_source() {
    // Source hydro id 99 does NOT exist; destination hydro id 2 (pos 1) does.
    let fixtures = PumpFixtures::new(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station(10, 99, 2, 0.0, 50.0)],
    );
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

    let n_blks = layout.n_blks;
    let dest_pos = ctx.hydro_pos[&EntityId(2)];
    let row_dest = layout.row_water_balance_start() + dest_pos;
    for blk in 0..n_blks {
        let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
        let col = layout.col_pumping_start + blk;
        assert_eq!(
            col_entries[col],
            vec![(row_dest, -tau_h)],
            "blk {blk}: only the destination -tau_h entry survives"
        );
    }
}

/// A station whose `destination_hydro_id` is absent skips only the
/// destination entry — the source side is still written, no panic.
#[test]
fn pumping_water_entries_missing_destination_skips_only_destination() {
    // Source hydro id 1 (pos 0) exists; destination hydro id 99 does NOT.
    let fixtures = PumpFixtures::new(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station(10, 1, 99, 0.0, 50.0)],
    );
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

    let n_blks = layout.n_blks;
    let source_pos = ctx.hydro_pos[&EntityId(1)];
    let row_source = layout.row_water_balance_start() + source_pos;
    for blk in 0..n_blks {
        let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
        let col = layout.col_pumping_start + blk;
        assert_eq!(
            col_entries[col],
            vec![(row_source, tau_h)],
            "blk {blk}: only the source +tau_h entry survives"
        );
    }
}

/// The pumping flow column enters its bus load-balance row with
/// `−consumption_mw_per_m3s` per block — a negative injection, the same sign a
/// line carries into its source bus, NOT the `+1.0` of generation.
#[test]
fn pumping_power_enters_bus_row_with_negative_consumption() {
    // Station id 10 on bus id 1 (pos 0), consumption 0.75 MW per m³/s.
    let fixtures = PumpFixtures::new_with_buses(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station_full(10, 1, 2, 0.0, 50.0, 1, 0.75)],
        vec![fixture_bus(1)],
    );
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

    let n_blks = layout.n_blks;
    let b_idx = ctx.bus_pos[&EntityId(1)];
    for blk in 0..n_blks {
        let row = layout.row_load_balance_start() + b_idx * n_blks + blk;
        let col = layout.col_pumping_start + blk;
        assert!(
            col_entries[col].contains(&(row, -0.75)),
            "blk {blk}: pumping column {col} must carry (row {row}, -0.75); got {:?}",
            col_entries[col]
        );
        // The flow column carries the bus-power coupling and nothing else from
        // the load-balance fill (no positive generation-style entry).
        assert_eq!(
            col_entries[col],
            vec![(row, -0.75)],
            "blk {blk}: pumping column must carry only the negative-injection entry"
        );
    }
}

/// A station whose `bus_id` is absent from `bus_pos` writes no load-balance
/// entry and does not panic.
#[test]
fn pumping_power_missing_bus_skips_without_panic() {
    // Station on bus id 99, which is NOT among the fixture buses (only id 1).
    let fixtures = PumpFixtures::new_with_buses(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station_full(10, 1, 2, 0.0, 50.0, 99, 0.5)],
        vec![fixture_bus(1)],
    );
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

    let n_blks = layout.n_blks;
    for blk in 0..n_blks {
        let col = layout.col_pumping_start + blk;
        assert!(
            col_entries[col].is_empty(),
            "blk {blk}: station on an unmapped bus must write no load-balance entry"
        );
    }
}

/// With no pumping stations, `fill_load_balance_entries` produces exactly the
/// same entries it would without the pumping loop — the pumping path is inert.
/// A 1-bus, 2-hydro system with one declared station is the baseline; removing
/// the station must leave every column's load-balance entries unchanged.
#[test]
fn no_pumping_stations_leaves_load_balance_entries_identical() {
    let build = |stations: Vec<PumpingStation>| {
        let fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            stations,
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);
        // Truncate to the non-pumping column region: with zero stations the
        // layout has no pumping columns, so compare the shared prefix that both
        // layouts share (generation/thermal/line/deficit/excess columns are all
        // indexed before the pumping block).
        (layout.col_pumping_start, col_entries)
    };

    let (pump_start_empty, entries_empty) = build(vec![]);
    let (_pump_start_one, entries_one) = build(vec![station(10, 1, 2, 0.0, 50.0)]);

    // Every column before the pumping block must carry identical load-balance
    // entries whether or not a station is present.
    for col in 0..pump_start_empty {
        assert_eq!(
            entries_empty[col], entries_one[col],
            "load-balance entries for column {col} must be pumping-independent"
        );
    }
}

/// Build the full CSC for a 2-reservoir + 1-bus system twice with the hydro,
/// station, and bus declarations supplied in two DIFFERENT input orders, and
/// assert the assembled CSC arrays are byte-identical. Determinism is a hard
/// rule: the canonical ID-sort plus the per-column row-sort must erase all
/// trace of the input declaration order.
///
/// Two stations with opposite source/destination orientation are declared so
/// the assertion is load-bearing on the pumping path: a single station would
/// pass even if the pumping iteration were declaration-order-dependent (there
/// is nothing to scramble), whereas permuting two stations exercises the
/// per-column row-sort that decouples declaration order from CSC layout. Both
/// stations sit on the single bus, so the `−consumption_mw_per_m3s` bus-power
/// entries are part of the assembled CSC and the assertion covers them too.
#[test]
fn csc_byte_identical_under_permuted_declaration_order() {
    let assemble = |hydros: Vec<Hydro>, stations: Vec<PumpingStation>| {
        let fixtures = PumpFixtures::new(hydros, stations);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);
        let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
        // Mirror the production per-column row-sort (see build_single_stage_template).
        for col in &mut entries {
            col.sort_unstable_by_key(|&(row, _)| row);
        }
        assemble_csc(&entries)
    };

    // Two reservoirs (ids 1, 2) and two stations moving water in opposite
    // directions (10: 1 → 2, 20: 2 → 1), both on bus 1 with DISTINCT
    // consumption rates so a permutation that mislabels which station's
    // `−consumption_mw_per_m3s` lands on which pumping column would be caught.
    // Order A declares both ascending.
    let csc_a = assemble(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![
            station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
            station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
        ],
    );
    // Order B declares the identical entities with both hydros and both
    // stations reversed.
    let csc_b = assemble(
        vec![fixture_hydro(2), fixture_hydro(1)],
        vec![
            station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
            station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
        ],
    );

    assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
    assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
    assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
}

/// End-to-end: a generic constraint referencing `pumping_flow` and
/// `pumping_power` resolves to the REAL pumping column(s) through the
/// resolver's sole caller (`fill_generic_constraint_entries`), and the
/// constraint participates in the LP — its row carries CSC entries on the
/// pumping columns. The `block_id = None` expression is block-dependent, so it
/// expands to one generic row per block; each row's two terms (flow ×1.0 and
/// power ×consumption) alias the SAME pumping column for that block, so the
/// summed coefficient at `(pumping_col, generic_row)` is `1.0 + consumption`.
#[test]
fn b6b_generic_constraint_resolves_pumping_columns_in_lp() {
    let consumption = 0.5_f64;
    let constraint_id = EntityId(7);
    let station_id = EntityId(10);

    // pumping_flow(10) + pumping_power(10) <= 40 (block_id = None on both).
    let constraint = GenericConstraint {
        id: constraint_id,
        name: "gc_pump".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm {
                    coefficient: CoefficientRef::Literal(1.0),
                    scale: 1.0,
                    variable: VariableRef::PumpingFlow {
                        station_id,
                        block_id: None,
                    },
                },
                LinearTerm {
                    coefficient: CoefficientRef::Literal(1.0),
                    scale: 1.0,
                    variable: VariableRef::PumpingPower {
                        station_id,
                        block_id: None,
                    },
                },
            ],
        },
        sense: ConstraintSense::LessEqual,
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let fixtures = PumpFixtures::new(
        vec![fixture_hydro(1), fixture_hydro(2)],
        vec![station_full(station_id.0, 1, 2, 0.0, 50.0, 1, consumption)],
    )
    .with_generic_constraint(constraint, 40.0);
    let ctx = fixtures.make_ctx();
    let stage = two_block_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    // Block-dependent expression with block_id = None expands to one generic
    // row per block, so the constraint participates as `n_blks` rows.
    let n_blks = layout.n_blks;
    assert_eq!(
        layout.n_generic_rows, n_blks,
        "block-dependent pumping constraint must expand to one row per block"
    );

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];
    let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
    let mut row_upper = vec![f64::INFINITY; layout.num_rows];
    let mut buffers = LpMatrixBuffers {
        col_entries: &mut col_entries,
        col_upper: &mut col_upper,
        objective: &mut objective,
        row_lower: &mut row_lower,
        row_upper: &mut row_upper,
    };

    fill_generic_constraint_entries(&ctx, &stage, 0, &layout, &mut buffers);

    // Each generic row `blk` lands on the station's flow column for that block,
    // with the flow (1.0) and power (consumption) terms aliasing the SAME column.
    // p_idx = 0 (the only station), so col = col_pumping_start + blk.
    for blk in 0..n_blks {
        let row = layout.row_generic_start + blk;
        let col = layout.col_pumping_start + blk;
        let summed: f64 = col_entries[col]
            .iter()
            .filter(|&&(r, _)| r == row)
            .map(|&(_, v)| v)
            .sum();
        assert_eq!(
            summed,
            1.0 + consumption,
            "blk {blk}: pumping column {col} must carry flow(1.0) + power({consumption}) on generic row {row}"
        );
        // The row bound proves the constraint participates with the right sense.
        assert_eq!(row_upper[row], 40.0, "blk {blk}: <= row upper bound");
        assert_eq!(row_lower[row], f64::NEG_INFINITY, "blk {blk}: <= row lower");
    }
}
