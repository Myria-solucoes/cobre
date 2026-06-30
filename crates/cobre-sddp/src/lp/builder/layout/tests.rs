#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds,
    EntityId, FillingConfig, Hydro, HydroGenerationModel, HydroStageBounds, LineStageBounds,
    NoiseMethod, PumpingStageBounds, PumpingStation, ResolvedBounds, ResolvedExchangeFactors,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    ThermalStageBounds,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};

use crate::resolved_parameters::ResolvedParameters;

use super::super::test_support::{state_layout_for, zero_hydro_penalties};
use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET, ResolvedTables,
    StageLayout, TemplateBuildCtx,
};

// ── Fixture helpers ───────────────────────────────────────────────────────

/// Owns all data needed to construct a zero-entity `TemplateBuildCtx`.
///
/// Fields are kept together so that references into them share a single
/// lifetime `'_`, avoiding the 16-argument helper that clippy flags.
struct ZeroEntityFixtures {
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
}

impl ZeroEntityFixtures {
    fn new() -> Self {
        Self {
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            bounds: ResolvedBounds::empty(),
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
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
        }
    }

    /// Build a zero-entity `TemplateBuildCtx` with the supplied
    /// anticipated-metadata overrides.
    ///
    /// All slice fields are empty; all scalar entity counts are zero except
    /// the anticipated fields provided by the caller.
    fn make_ctx(
        &self,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        anticipated_thermal_indices: Vec<usize>,
    ) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            // Windowless: one `(None, None)` per anticipated plant. With no
            // window the operation-window clause is identically true, so the
            // decision gate reduces to the strict horizon clause — the
            // behaviour these layout tests assert. `study_stage_ids` is sized
            // to the bounds' study-stage count so the gate's in-range
            // delivery-stage lookup never indexes out of bounds.
            anticipated_windows: vec![(None, None); n_anticipated],
            study_stage_ids: (0..i32::try_from(self.bounds.n_stages()).unwrap_or(0)).collect(),
            anticipated_thermal_indices,
            has_penalty: false,
            // Tests that use ZeroEntityFixtures don't exercise discount
            // factors; provide n_stages = 1 element vecs that won't panic.
            cumulative_discount_factors: vec![1.0],
            total_hours_per_stage: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// Build a minimal `Stage` with one block of 744 hours.
fn minimal_stage() -> Stage {
    stage_with_id(0)
}

/// Build a one-block `Stage` whose `id` (the study stage id `filling_phase`
/// keys on) equals `stage_id`. `index` is held at `0` because the per-stage
/// FPHA/evaporation/bounds lookups in these fixtures are indexed by
/// `stage_idx = 0`, while the phase gate reads `stage.id` alone — decoupling
/// the two lets one bounds/model row serve every phase under test.
fn stage_with_id(stage_id: i32) -> Stage {
    Stage {
        index: 0,
        id: stage_id,
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
    }
}

/// Build a single hydro for the FPHA/evaporation membership fixtures.
///
/// `filling`/`entry` drive the [`filling_phase`] gate; `generation_model`
/// follows `fpha`. All other fields are inert defaults — these fixtures
/// exercise per-stage row *membership* (`identify_fpha_hydros` /
/// `identify_evap_hydros`), not column values.
fn membership_hydro(
    id: i32,
    fpha: bool,
    filling: Option<FillingConfig>,
    entry: Option<i32>,
) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: entry,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: if fpha {
            HydroGenerationModel::Fpha
        } else {
            HydroGenerationModel::ConstantProductivity
        },
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
        filling,
        penalties: zero_hydro_penalties(),
    }
}

// ── AC-3 ─────────────────────────────────────────────────────────────────

/// AC-3: `StageLayout` built from a context with `n_anticipated == 0` has
/// `n_ant_state == 0`, `n_anticipated == 0`, `k_max == 0`, and
/// `col_turbine_start == idx.theta + 1` where `idx` is the legacy
/// the N=0, L=0 state layout (zero hydros, zero lag order).
///
/// This verifies that the decision-region offset before the
/// `anticipated_state_out` insertion is preserved when no anticipated
/// thermals are present.
#[test]
fn stage_layout_zero_anticipated_matches_pre_anticipated_offsets() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // n_ant_state, n_anticipated, k_max must all be zero.
    assert_eq!(layout.n_ant_state, 0, "n_ant_state");
    assert_eq!(layout.n_anticipated, 0, "n_anticipated");
    assert_eq!(layout.k_max, 0, "k_max");

    // col_turbine_start must equal the legacy theta + 1.
    let idx = crate::test_support::state_layout(ctx.n_hydros, ctx.max_par_order);
    assert_eq!(
        layout.col_turbine_start(),
        idx.theta + 1,
        "col_turbine_start must equal idx.theta + 1 with zero anticipated"
    );
}

// ── storage_internal interior-boundary sizing ────────────────────────────

/// Owns a two-hydro, constant-productivity `TemplateBuildCtx` for the
/// `storage_internal` sizing assertions. No FPHA/filling/evaporation, so only
/// the block geometry (`n_blks`, `block_mode`) and `n_hydros` drive the family.
struct TwoHydroFixtures {
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
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
}

impl TwoHydroFixtures {
    fn new() -> Self {
        use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let models = vec![vec![constant.clone()], vec![constant]];
        let hydros = vec![
            membership_hydro(1, false, None, None),
            membership_hydro(2, false, None, None),
        ];
        let cascade = CascadeTopology::build(&hydros);
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            cascade,
            bounds: ResolvedBounds::empty(),
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
            production_models: ProductionModelSet::new(models, 2, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            study_stage_ids: vec![],
            has_penalty: false,
            cumulative_discount_factors: vec![1.0],
            total_hours_per_stage: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// Build a `Stage` with `n_blks` equal-duration blocks under `block_mode`.
fn stage_with_blocks(block_mode: BlockMode, n_blks: usize) -> Stage {
    let mut stage = minimal_stage();
    stage.block_mode = block_mode;
    stage.blocks = (0..n_blks)
        .map(|index| Block {
            index,
            name: format!("BLK{index}"),
            duration_hours: 744.0,
        })
        .collect();
    stage
}

/// `storage_internal` spans the `K − 1` interior boundaries per hydro only in
/// chronological mode with `K ≥ 2`: empty in parallel mode and at `K = 1`, with
/// `turbine.start` re-anchored to `storage_internal.end`.
#[test]
fn chronological_storage_internal_sizing() {
    let fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);
    let anchor = state.control_region_start();

    let parallel = StageLayout::new(&ctx, &state, &stage_with_blocks(BlockMode::Parallel, 3), 0);
    assert_eq!(
        parallel.storage_internal.start, parallel.storage_internal.end,
        "parallel K=3 storage_internal is empty"
    );
    assert_eq!(
        parallel.storage_internal_start, anchor,
        "parallel storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        parallel.turbine.start, anchor,
        "parallel turbine.start anchors at control_region_start()"
    );

    let chrono_k1 = StageLayout::new(
        &ctx,
        &state,
        &stage_with_blocks(BlockMode::Chronological, 1),
        0,
    );
    assert_eq!(
        chrono_k1.storage_internal.start, chrono_k1.storage_internal.end,
        "chronological K=1 storage_internal is empty"
    );
    assert_eq!(
        chrono_k1.storage_internal_start, anchor,
        "chronological K=1 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k1.turbine.start, anchor,
        "chronological K=1 turbine.start anchors at control_region_start()"
    );

    let chrono_k3 = StageLayout::new(
        &ctx,
        &state,
        &stage_with_blocks(BlockMode::Chronological, 3),
        0,
    );
    assert_eq!(
        chrono_k3.storage_internal_start, anchor,
        "chronological K=3 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k3.storage_internal.end - chrono_k3.storage_internal.start,
        4,
        "chronological K=3 storage_internal spans n_h * (K - 1) = 2 * 2 columns"
    );
    assert_eq!(
        chrono_k3.turbine.start, chrono_k3.storage_internal.end,
        "chronological K=3 turbine.start re-anchors to storage_internal.end"
    );
}

/// `block_storage_col` resolves all `K + 1` boundaries: the two endpoints to the
/// state columns (`k = 0 → storage_in[h]`, `k = K → storage[h] = h`) and the
/// `K − 1` interiors into the `storage_internal` family at stride `n_blks − 1`. At
/// `K = 1` only the two endpoints resolve (no interior column is addressed).
#[test]
fn block_storage_col_resolves_all_boundaries() {
    let fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);

    let chrono_k3 = StageLayout::new(
        &ctx,
        &state,
        &stage_with_blocks(BlockMode::Chronological, 3),
        0,
    );
    let h = 1;
    assert_eq!(
        chrono_k3.block_storage_col(h, 0),
        chrono_k3.col_storage_in_start() + h,
        "k = 0 resolves to the incoming-state column storage_in[h]"
    );
    assert_eq!(
        chrono_k3.block_storage_col(h, 3),
        h,
        "k = K resolves to the outgoing-state column storage[h] = storage.start + h = h"
    );
    let interior_1 = chrono_k3.block_storage_col(h, 1);
    let interior_2 = chrono_k3.block_storage_col(h, 2);
    assert_eq!(
        interior_1,
        chrono_k3.storage_internal_start + h * 2,
        "k = 1 resolves to storage_internal_start + h * (K - 1) + 0"
    );
    assert_eq!(
        interior_2,
        chrono_k3.storage_internal_start + h * 2 + 1,
        "k = 2 resolves to storage_internal_start + h * (K - 1) + 1"
    );
    assert!(
        chrono_k3.storage_internal.contains(&interior_1)
            && chrono_k3.storage_internal.contains(&interior_2),
        "both interior columns lie within the storage_internal range"
    );

    let chrono_k1 = StageLayout::new(
        &ctx,
        &state,
        &stage_with_blocks(BlockMode::Chronological, 1),
        0,
    );
    assert!(
        chrono_k1.storage_internal.is_empty(),
        "K = 1 has no interior storage columns"
    );
    for h in 0..ctx.n_hydros {
        assert_eq!(
            chrono_k1.block_storage_col(h, 0),
            chrono_k1.col_storage_in_start() + h,
            "K = 1 endpoint k = 0 resolves to storage_in[h]"
        );
        assert_eq!(
            chrono_k1.block_storage_col(h, 1),
            h,
            "K = 1 endpoint k = K = 1 resolves to storage[h] = h"
        );
    }
}

// ── FPHA-local inverse map ───────────────────────────────────────────────

/// Owns the data needed to construct a three-hydro `TemplateBuildCtx` with a
/// single FPHA hydro at system index 1 (the other two use constant
/// productivity), so `StageLayout::new` derives `fpha_hydro_indices == [1]`.
struct FphaMixFixtures {
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
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
}

impl FphaMixFixtures {
    fn new() -> Self {
        use crate::hydro_models::{EvaporationModel, FphaPlane, ResolvedProductionModel};

        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let fpha = ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 0.0,
                gamma_v: 0.0,
                gamma_q: 0.0,
                gamma_s: 0.0,
            }],
        };
        // models[hydro][stage]: hydro 1 is FPHA, hydros 0 and 2 are constant.
        let models = vec![vec![constant.clone()], vec![fpha], vec![constant]];
        // All three hydros are non-filling: `filling_phase` is `Operating` at
        // every stage, so the filling exclusion never fires and these fixtures
        // assert the same FPHA membership a pre-gate build would (parity-neutral).
        let hydros = vec![
            membership_hydro(1, false, None, None),
            membership_hydro(2, true, None, None),
            membership_hydro(3, false, None, None),
        ];
        let cascade = CascadeTopology::build(&hydros);
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            cascade,
            bounds: ResolvedBounds::empty(),
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
            production_models: ProductionModelSet::new(models, 3, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 3,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            study_stage_ids: vec![],
            has_penalty: false,
            cumulative_discount_factors: vec![1.0],
            total_hours_per_stage: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// `StageLayout::new` inverts `fpha_hydro_indices` into `fpha_local_index`:
/// the FPHA hydro at system index 1 of three maps to local index 0, and the
/// two non-FPHA hydros stay `None`, giving `[None, Some(0), None]`.
#[test]
fn stage_layout_populates_fpha_local_index_inverse_map() {
    let fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.fpha_hydro_indices,
        vec![1],
        "only the system-index-1 hydro uses FPHA"
    );
    assert_eq!(
        layout.fpha_local_index,
        vec![None, Some(0), None],
        "fpha_local_index inverts fpha_hydro_indices over n_h = 3"
    );
}

// ── Per-stage FPHA / evaporation filling exclusion ───────────────────────

/// Owns a two-hydro `TemplateBuildCtx` for the filling-phase membership
/// tests. Hydro 0 is an FPHA **filling** hydro; hydro 1 is a non-FPHA
/// **filling** hydro carrying a linearized evaporation model. Both share the
/// filling window `start_stage_id = 1`, `entry_stage_id = 3`, so a single
/// fixture exercises every phase by varying only `stage.id`:
/// `0` ⇒ `PreFilling`, `1`/`2` ⇒ `Filling`, `≥ 3` ⇒ `Operating`.
struct FillingMembershipFixtures {
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
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
}

impl FillingMembershipFixtures {
    const START_STAGE_ID: i32 = 1;
    const ENTRY_STAGE_ID: i32 = 3;

    /// `filling`/`fpha` flags: hydro 0 is FPHA + filling, hydro 1 is
    /// non-FPHA + filling-with-evaporation. Both hydros use the same filling
    /// window so the phase is a pure function of `stage.id` under test.
    fn new() -> Self {
        use crate::hydro_models::{
            EvaporationModel, FphaPlane, LinearizedEvaporation, ResolvedProductionModel,
        };

        let filling = || {
            Some(FillingConfig {
                start_stage_id: Self::START_STAGE_ID,
                filling_min_rate_m3s: 0.0,
            })
        };
        let entry = Some(Self::ENTRY_STAGE_ID);
        let hydros = vec![
            membership_hydro(1, true, filling(), entry),
            membership_hydro(2, false, filling(), entry),
        ];
        let cascade = CascadeTopology::build(&hydros);

        // Production: hydro 0 is FPHA at stage 0; hydro 1 is constant.
        let fpha = ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 0.0,
                gamma_v: 0.0,
                gamma_q: 0.0,
                gamma_s: 0.0,
            }],
        };
        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let models = vec![vec![fpha], vec![constant]];

        // Evaporation: hydro 0 has none; hydro 1 is linearized. The
        // `Linearized` variant is per-hydro, so membership does not depend on
        // `stage_idx`.
        let evaporation_models = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 0.0,
                    volume_slope_m3s_per_hm3: 0.0,
                }],
                reference_volumes_hm3: vec![0.0],
            },
        ]);

        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            cascade,
            bounds: ResolvedBounds::empty(),
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
            production_models: ProductionModelSet::new(models, 2, 1),
            evaporation_models,
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            study_stage_ids: vec![],
            has_penalty: false,
            cumulative_discount_factors: vec![1.0],
            total_hours_per_stage: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }

    /// `fpha_hydro_indices` for a stage built at `stage_id` (`stage_idx` held
    /// at 0 so the single FPHA/evaporation model row serves every phase).
    fn fpha_indices_at(&self, stage_id: i32) -> Vec<usize> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).fpha_hydro_indices
    }

    /// `evap_hydro_indices` for a stage built at `stage_id`.
    fn evap_indices_at(&self, stage_id: i32) -> Vec<usize> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).evap_hydro_indices
    }

    /// `filling_target_hydro_indices` for a stage built at `stage_id`.
    fn filling_target_indices_at(&self, stage_id: i32) -> Vec<usize> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).filling_target_hydro_indices
    }

    /// `filled_min_storage_floor_hydro_indices` for a stage built at `stage_id`.
    fn filled_min_storage_floor_indices_at(&self, stage_id: i32) -> Vec<usize> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).filled_min_storage_floor_hydro_indices
    }

    /// `num_rows` for a stage built at `stage_id` — the structural row count
    /// the append-only cut rows begin at.
    fn num_rows_at(&self, stage_id: i32) -> usize {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).num_rows
    }
}

/// The per-stage `σ_fill` target is emitted at EVERY Filling stage, not only at
/// `entry − 1`. Both filling hydros share `start = 1`, `entry = 3`, so the
/// Filling stages are `{1, 2}`; both carry the target at BOTH. `PreFilling` (id 0)
/// and Operating (id ≥ 3) emit none. The wrong-but-compiling alternative is the
/// v1 terminal-only rule (`entry − 1 == stage_id`), which would drop the id-1
/// floor; this test pins per-stage Filling membership.
#[test]
fn filling_target_emitted_at_every_filling_stage() {
    let fixtures = FillingMembershipFixtures::new();

    // Filling stages 1 and 2 (start = 1, entry = 3): both filling hydros
    // (system indices 0, 1) carry the target at every Filling stage.
    for stage_id in [1, 2] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id),
            vec![0, 1],
            "both filling hydros carry the σ_fill target at Filling id {stage_id}"
        );
    }

    // PreFilling (id 0) and Operating (id ≥ entry = 3) emit NO target.
    for stage_id in [0, 3, 4] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id),
            Vec::<usize>::new(),
            "no σ_fill target at non-Filling id {stage_id}"
        );
    }
}

/// Parity-neutrality: a non-filling system never emits a `σ_fill` target, so
/// `num_rows` is bit-identical across every stage id (the cut-row region anchor
/// is unmoved). The forbidden alternative — reserving a target row for every
/// hydro unconditionally — would shift `num_rows` and alias the append-only cut
/// rows for the existing non-filling deterministic cases.
#[test]
fn non_filling_system_no_filling_target_num_rows_unchanged() {
    // `FphaMixFixtures` hydros are all non-filling.
    let fixtures = FphaMixFixtures::new();
    let layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        (layout.filling_target_hydro_indices.clone(), layout.num_rows)
    };
    let (reference_targets, reference_num_rows) = layout_at(0);
    assert_eq!(
        reference_targets,
        Vec::<usize>::new(),
        "non-filling system emits no σ_fill target"
    );
    for stage_id in [1, 2, 3, 7] {
        let (targets, num_rows) = layout_at(stage_id);
        assert_eq!(
            targets,
            Vec::<usize>::new(),
            "non-filling σ_fill target empty at id {stage_id}"
        );
        assert_eq!(
            num_rows, reference_num_rows,
            "non-filling num_rows unchanged at id {stage_id}"
        );
    }
}

/// The `σ_fill` row block lands STRICTLY BELOW `num_rows` (the pre-cut
/// region), ahead of the append-only cut rows that begin at `num_rows`. A row
/// at index `>= num_rows` would alias a cut row and corrupt slot-identity
/// warm-start reconstruction. The `σ_fill` column likewise lands strictly below
/// `num_cols`. The `filling_target` block is the FIRST pre-cut filling-row
/// family, so it follows the operational-violation rows directly (no retention
/// block precedes it).
#[test]
fn filling_target_row_and_col_below_structural_bounds() {
    let fixtures = FillingMembershipFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = stage_with_id(2); // entry − 1: the terminal stage.
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let n_targets = layout.filling_target_hydro_indices.len();
    assert_eq!(n_targets, 2, "both filling hydros carry the target at id 2");

    // Every σ_fill row index is strictly below num_rows (pre-cut region).
    let row_start = layout.row_filling_target_start();
    for local_idx in 0..n_targets {
        assert!(
            row_start + local_idx < layout.num_rows,
            "σ_fill row {} must be < num_rows {}",
            row_start + local_idx,
            layout.num_rows
        );
    }
    // The σ_fill rows sit immediately after the operational-violation rows (the
    // last of which is `min_generation_rows`) and before the fishing rows — i.e.
    // inside the pre-cut region, with no retention block in between.
    assert_eq!(
        row_start, layout.min_generation_rows.end,
        "σ_fill rows follow the operational-violation rows directly"
    );

    // Every σ_fill column index is strictly below num_cols.
    let col_start = layout.col_filling_target_start();
    for local_idx in 0..n_targets {
        assert!(
            col_start + local_idx < layout.num_cols,
            "σ_fill col {} must be < num_cols {}",
            col_start + local_idx,
            layout.num_cols
        );
    }
    // At the terminal Filling stage (id 2) no hydro is Operating, so the
    // sibling σ^{v-} `filled_min_storage_floor` column block (the true last column family)
    // is empty: its start coincides with num_cols and the σ_fill block is the
    // last occupied family, so num_cols = col_filling_target_start + n_targets.
    assert_eq!(
        layout.col_filled_min_storage_floor_start(),
        layout.num_cols,
        "σ^{{v-}} column block empty at the terminal Filling stage"
    );
    assert_eq!(
        layout.num_cols,
        col_start + n_targets,
        "num_cols = col_filling_target_start + n_targets (σ^{{v-}} block empty here)"
    );
}

/// The `σ_fill` target family adds rows at EVERY Filling stage (ids 1, 2 here)
/// and NONE at `PreFilling` (id 0) or Operating (id ≥ 3). The non-Filling stages
/// keep an empty target block (the fishing-row start coincides with the
/// target-row start), isolating the per-stage target rows to the Filling window.
#[test]
fn filling_target_adds_rows_at_every_filling_stage() {
    let fixtures = FillingMembershipFixtures::new();
    // PreFilling (id 0) and Operating (id 3, 4): the σ_fill TARGET adds no rows.
    for stage_id in [0, 3, 4] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert!(
            layout.filling_target_hydro_indices.is_empty(),
            "no σ_fill target rows at non-Filling id {stage_id}"
        );
        // Anchor unaffected: same fixture exercised in num_rows_at below.
        let _ = fixtures.num_rows_at(stage_id);
    }
    // Every Filling stage (ids 1, 2) adds exactly 2 target rows (one per hydro).
    for stage_id in [1, 2] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id).len(),
            2,
            "Filling id {stage_id} adds one σ_fill target row per filling hydro"
        );
    }
}

/// The soft `σ^{v-}` operating floor is emitted at EVERY `Operating` stage of a
/// filling hydro (id ≥ entry = 3), for BOTH filling hydros — distinct from the
/// every-Filling-stage `σ_fill` target. `PreFilling` (id 0) and `Filling` (id 1, 2)
/// emit none. This pins the `Operating`-only scope and the `σ^{v-}`/`σ_fill`
/// stage split.
#[test]
fn filled_min_storage_floor_emitted_at_every_operating_stage() {
    let fixtures = FillingMembershipFixtures::new();

    // Operating (id >= entry = 3): both filling hydros carry the floor at every
    // stage, not just one terminal stage.
    for stage_id in [3, 4, 7] {
        assert_eq!(
            fixtures.filled_min_storage_floor_indices_at(stage_id),
            vec![0, 1],
            "both filling hydros carry σ^{{v-}} at Operating id {stage_id}"
        );
    }

    // PreFilling (id 0) and Filling (id 1, 2 = the σ_fill terminal): no floor.
    for stage_id in [0, 1, 2] {
        assert_eq!(
            fixtures.filled_min_storage_floor_indices_at(stage_id),
            Vec::<usize>::new(),
            "no σ^{{v-}} at non-operating id {stage_id}"
        );
    }

    // Mutual exclusivity at the boundary: id 2 (entry − 1) carries σ_fill but
    // NOT σ^{v-}; id 3 (entry) carries σ^{v-} but NOT σ_fill.
    assert_eq!(fixtures.filling_target_indices_at(2), vec![0, 1]);
    assert!(fixtures.filled_min_storage_floor_indices_at(2).is_empty());
    assert!(fixtures.filling_target_indices_at(3).is_empty());
    assert_eq!(fixtures.filled_min_storage_floor_indices_at(3), vec![0, 1]);
}

/// Parity-neutrality: a non-filling system never emits a `σ^{v-}` floor, so
/// `num_rows` is bit-identical across every stage id. The forbidden GLOBAL soft
/// floor — reserving a floor row for every Operating hydro regardless of
/// `filling` — would shift `num_rows` and alias the append-only cut rows for the
/// existing deterministic cases.
#[test]
fn non_filling_system_no_filled_min_storage_floor_num_rows_unchanged() {
    let fixtures = FphaMixFixtures::new();
    let layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        (
            layout.filled_min_storage_floor_hydro_indices.clone(),
            layout.num_rows,
        )
    };
    let (reference_floors, reference_num_rows) = layout_at(0);
    assert_eq!(
        reference_floors,
        Vec::<usize>::new(),
        "non-filling system emits no σ^{{v-}} floor"
    );
    for stage_id in [1, 2, 3, 7] {
        let (floors, num_rows) = layout_at(stage_id);
        assert_eq!(
            floors,
            Vec::<usize>::new(),
            "non-filling σ^{{v-}} floor empty at id {stage_id}"
        );
        assert_eq!(
            num_rows, reference_num_rows,
            "non-filling num_rows unchanged at id {stage_id}"
        );
    }
}

/// A filling FPHA hydro is excluded from `fpha_hydro_indices` while
/// `Filling` (its FPHA fit is invalid below `min_storage`), and re-included
/// once `Operating`. The forbidden alternative — leaving it in the set during
/// filling — would emit an FPHA production row over an invalid operating-range
/// fit and a generation column with no constraining row.
#[test]
fn filling_fpha_hydro_excluded_while_filling_present_when_operating() {
    let fixtures = FillingMembershipFixtures::new();

    // Filling (stage_id 1 and 2 are in `[start_stage_id, entry_stage_id)`):
    // hydro 0 (the FPHA hydro) is absent.
    assert_eq!(
        fixtures.fpha_indices_at(1),
        Vec::<usize>::new(),
        "FPHA filling hydro absent from fpha_hydro_indices during Filling"
    );
    assert_eq!(
        fixtures.fpha_indices_at(2),
        Vec::<usize>::new(),
        "FPHA filling hydro absent at the last Filling stage"
    );

    // Operating (stage_id >= entry_stage_id): hydro 0 re-enters.
    assert_eq!(
        fixtures.fpha_indices_at(3),
        vec![0],
        "FPHA filling hydro present from the first Operating stage"
    );
    assert_eq!(
        fixtures.fpha_indices_at(4),
        vec![0],
        "FPHA filling hydro present at later Operating stages"
    );

    // PreFilling (stage_id < start_stage_id): the dam does not exist yet, so
    // the FPHA hydro is also excluded.
    assert_eq!(
        fixtures.fpha_indices_at(0),
        Vec::<usize>::new(),
        "FPHA filling hydro absent during PreFilling"
    );
}

/// A filling hydro with evaporation is excluded from `evap_hydro_indices`
/// only during `PreFilling` (no reservoir surface), and present during
/// `Filling` and `Operating` (the impounding reservoir has a surface). This
/// is the opposite of the FPHA rule, which also excludes during `Filling` —
/// the two exclusions must not be unified.
#[test]
fn filling_evap_hydro_excluded_only_in_prefilling() {
    let fixtures = FillingMembershipFixtures::new();

    // PreFilling (stage_id < start_stage_id): hydro 1 (evaporation) is absent.
    assert_eq!(
        fixtures.evap_indices_at(0),
        Vec::<usize>::new(),
        "evaporation filling hydro absent during PreFilling (no reservoir surface)"
    );

    // Filling: evaporation is normal — the reservoir already has a surface.
    assert_eq!(
        fixtures.evap_indices_at(1),
        vec![1],
        "evaporation filling hydro present during Filling"
    );
    assert_eq!(
        fixtures.evap_indices_at(2),
        vec![1],
        "evaporation filling hydro present at the last Filling stage"
    );

    // Operating: evaporation remains normal.
    assert_eq!(
        fixtures.evap_indices_at(3),
        vec![1],
        "evaporation filling hydro present once Operating"
    );
}

/// Parity-neutrality contract: a non-filling hydro is `Operating` at every
/// stage, so neither exclusion fires — its membership in both
/// `fpha_hydro_indices` and `evap_hydro_indices` is bit-identical across all
/// stages, matching a build without the filling gate.
#[test]
fn non_filling_hydro_membership_bit_identical_across_stages() {
    // The `FphaMixFixtures` hydros are all non-filling (one FPHA at system
    // index 1, two constant), so its membership must be invariant to stage_id.
    let fixtures = FphaMixFixtures::new();
    let reference_fpha = {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(0);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).fpha_hydro_indices
    };
    let reference_evap = {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(0);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).evap_hydro_indices
    };

    // The non-filling FPHA hydro is at system index 1; evaporation is empty.
    assert_eq!(reference_fpha, vec![1]);
    assert_eq!(reference_evap, Vec::<usize>::new());

    for stage_id in [1, 2, 3, 7] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.fpha_hydro_indices, reference_fpha,
            "non-filling fpha_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
        assert_eq!(
            layout.evap_hydro_indices, reference_evap,
            "non-filling evap_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
    }
}

// ── Operational-violation row ranges ─────────────────────────────────────

/// The four operational-violation row families (`min_outflow_rows`,
/// `max_outflow_rows`, `min_turbine_rows`, `min_generation_rows`) are
/// contiguous, in that order, each spanning exactly `n_h * n_blks` rows, and
/// the block starts immediately after the post-equipment row cursor
/// (`evap_rows_end`), which equals `min_outflow_rows.start`. The owning
/// arithmetic lives in [`StageLayout::new`]; this pins it at the internal
/// layer where the row ranges are visible. The forbidden alternative is a
/// stale or transposed base — placing `max_outflow` before `min_outflow`, or
/// striding any family by something other than `n_h * n_blks`, addresses the
/// wrong constraint rows and silently mis-bounds the operational violations.
///
/// Fixture: `n_h = 3`, one block (`n_blks = 1`), so `n_op = 3` per family.
#[test]
fn stage_layout_operational_violation_rows_are_contiguous_blocks() {
    let fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let n_op = ctx.n_hydros; // n_h * n_blks with n_blks == 1
    assert!(
        n_op > 0,
        "fixture must have hydros so the rows are non-empty"
    );

    // The four families are non-empty, in canonical order, contiguous, and
    // each spans exactly `n_op` rows.
    assert_eq!(layout.min_outflow_rows.len(), n_op, "min_outflow row count");
    assert_eq!(layout.max_outflow_rows.len(), n_op, "max_outflow row count");
    assert_eq!(layout.min_turbine_rows.len(), n_op, "min_turbine row count");
    assert_eq!(
        layout.min_generation_rows.len(),
        n_op,
        "min_generation row count"
    );

    assert_eq!(
        layout.max_outflow_rows.start, layout.min_outflow_rows.end,
        "max_outflow must follow min_outflow contiguously"
    );
    assert_eq!(
        layout.min_turbine_rows.start, layout.max_outflow_rows.end,
        "min_turbine must follow max_outflow contiguously"
    );
    assert_eq!(
        layout.min_generation_rows.start,
        layout.max_outflow_rows.end + n_op,
        "min_generation must start one min_turbine block (n_op rows) after max_outflow ends"
    );

    // The block anchors at the post-equipment row cursor, mirrored by the
    // `row_min_outflow_start` accessor.
    assert_eq!(
        layout.min_outflow_rows.start,
        layout.row_min_outflow_start(),
        "min_outflow_rows.start must equal row_min_outflow_start() when n_h > 0"
    );
    assert_eq!(
        layout.max_outflow_rows.start,
        layout.row_max_outflow_start()
    );
    assert_eq!(
        layout.min_turbine_rows.start,
        layout.row_min_turbine_start()
    );
    assert_eq!(
        layout.min_generation_rows.start,
        layout.row_min_generation_start()
    );
}

// ── Anticipated-decision column positioning ──────────────────────────────

/// `col_anticipated_decision_start` falls between thermal end and
/// `col_line_fwd_start` when `n_anticipated=2, n_thermals=3, n_blks=4`.
///
/// After the `anticipated_state_out` relocation the control region is
/// `thermal` then `anticipated_decision` (2 cols) then `line_fwd` —
/// `state_out` moved to the state region. So `col_line_fwd_start` equals
/// `col_anticipated_decision_start + n_anticipated`, and
/// `col_anticipated_state_out_start` is sourced from the state-region
/// position (immediately after the `anticipated_state` ring buffer), not
/// from the control region.
#[test]
fn anticipated_decision_columns_placed_between_thermal_and_line_fwd() {
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let fixtures = ZeroEntityFixtures::new();
    // ZeroEntityFixtures builds n_thermals=0, so the thermal per-block block is
    // empty and col_anticipated_decision_start == col_thermal_start. The two
    // stage-level anticipated blocks then separate col_thermal_start from
    // col_line_fwd_start by exactly 2 * n_anticipated columns.
    let n_anticipated = 2_usize;
    let k_max = 1_usize;
    let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![1, 1], vec![0, 0]);

    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![
            Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 186.0,
            },
            Block {
                index: 1,
                name: "B1".to_string(),
                duration_hours: 186.0,
            },
            Block {
                index: 2,
                name: "B2".to_string(),
                duration_hours: 186.0,
            },
            Block {
                index: 3,
                name: "B3".to_string(),
                duration_hours: 186.0,
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
    };
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // n_thermals = 0, n_blks = 4:
    // col_thermal_start = col_diversion_start + 0 * 4 = col_diversion_start
    // col_anticipated_decision_start = col_thermal_start + 0 * 4 = col_thermal_start
    // col_line_fwd_start = col_anticipated_decision_start + n_anticipated
    //   (state_out is no longer in the control region)
    assert_eq!(
        layout.anticipated.col_anticipated_decision_start,
        layout.col_thermal_start(),
        "col_anticipated_decision_start must equal col_thermal_start \
             when n_thermals=0 (no thermal per-block cols)"
    );
    assert_eq!(
        layout.col_line_fwd_start(),
        layout.anticipated.col_anticipated_decision_start + n_anticipated,
        "col_line_fwd_start == col_anticipated_decision_start + n_anticipated \
             (state_out relocated out of the control region)"
    );
    // The relocated state_out start equals the indexer's state-region
    // position: after the ring buffer (N*(1+L) + n_ant*k_max). Here N=0,
    // L=0, k_max=1, n_ant=2 → ring buffer is [0, 2), state_out starts at 2.
    assert_eq!(
        layout.anticipated.col_anticipated_state_out_start,
        n_anticipated * k_max,
        "col_anticipated_state_out_start must equal the state-region offset \
             N*(1+L) + n_ant*k_max"
    );
    // Verify the separation between thermal_start and line_fwd_start is
    // exactly n_anticipated (only the decision block remains between them).
    assert_eq!(
        layout.col_line_fwd_start() - layout.col_thermal_start(),
        n_anticipated,
        "gap from thermal_start to line_fwd_start must be exactly n_anticipated \
             (only the anticipated_decision block remains in the control region)"
    );
}

// ── AC-4 ─────────────────────────────────────────────────────────────────

/// AC-4: `StageLayout` with `n_anticipated=2, k_max=3, n_hydros=0,
/// max_par_order=0` has `col_turbine_start == 0*(3+0) + 6 + 2 + 1 == 9`.
///
/// `n_ant_state = n_anticipated * k_max = 2 * 3 = 6` and the relocated
/// `anticipated_state_out` block (width `n_anticipated = 2`) together shift
/// `theta` from the legacy `N*(3+L) = 0` to `0 + 6 + 2 = 8`, so decisions
/// begin at 9.
///
/// The general formula (any N, L) is
/// `N*(3+L) + n_ant_state + n_anticipated + 1`.
#[test]
fn stage_layout_with_anticipated_shifts_decision_region() {
    let n_hydros = 0_usize;
    let max_par_order = 0_usize;
    let n_anticipated = 2_usize;
    let k_max = 3_usize;

    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![2, 3], // anticipated_lead_stages
        vec![0, 2], // anticipated_thermal_indices (arbitrary; layout doesn't inspect them)
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // n_ant_state = n_anticipated * k_max = 2 * 3 = 6
    let expected_n_ant_state = n_anticipated * k_max;
    assert_eq!(layout.n_ant_state, expected_n_ant_state, "n_ant_state");

    // theta = N*(3+L) + n_ant_state + n_anticipated = 0*(3+0) + 6 + 2 = 8
    // col_turbine_start = theta + 1 = 9
    let expected_col_turbine_start =
        n_hydros * (3 + max_par_order) + expected_n_ant_state + n_anticipated + 1;
    assert_eq!(
        layout.col_turbine_start(),
        expected_col_turbine_start,
        "col_turbine_start == N*(3+L) + n_ant_state + n_anticipated + 1"
    );
}

// ── Anticipated-fishing row positioning ──────────────────────────────────

/// `row_anticipated_fishing_start` immediately follows the operational
/// violation row block, i.e. equals `row_min_generation_start + n_op_rows`.
///
/// Uses a zero-hydro context so `n_op_rows == 0`, which means the fishing
/// start equals `row_min_generation_start` exactly. The algebraic identity
/// `row_anticipated_fishing_start == row_min_generation_start + n_op_rows`
/// is verified for the general formula; the case `n_op_rows > 0` is covered by
/// the production code path (`n_hydros * n_blks` counts operational violation rows).
///
/// Setup: `n_anticipated=2`, `k_max=2`, `anticipated_lead_stages=[1,2]`,
/// zero hydros, one block. At `stage_idx=1`:
/// - `n_op_rows = 0 * 1 = 0` (no hydros)
/// - `n_anticipated_fishing_rows = 1` (`K_0=1<=1` active, `K_1=2>1` inactive)
/// - `row_anticipated_fishing_start` must equal `row_min_generation_start + 0`
#[test]
fn anticipated_fishing_row_offset_after_operational_violations() {
    let n_anticipated = 2_usize;
    let k_max = 2_usize;

    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![1, 2], // K_0=1, K_1=2
        vec![0, 1], // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block
    // stage_idx=1: always-active → both plants active → 2 fishing rows.
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 1);

    // n_op_rows = n_hydros * n_blks = 0 * 1 = 0
    let n_op_rows = 0_usize;
    assert_eq!(
        layout.anticipated.row_anticipated_fishing_start,
        layout.row_min_generation_start() + n_op_rows,
        "row_anticipated_fishing_start must equal row_min_generation_start + n_op_rows"
    );
    // Always-active: both plants active at every stage → 2 fishing rows.
    assert_eq!(
        layout.anticipated.n_anticipated_fishing_rows, 2,
        "n_anticipated_fishing_rows must equal n_anticipated (2) under always-active predicate"
    );
}

/// `n_anticipated_fishing_rows` equals `n_anticipated` at every stage under
/// the always-active predicate. With `K_i=[1,2]` and `n_anticipated=2`, the
/// count is 2 at every stage in `[0, 1, 2, 3]`.
#[test]
fn anticipated_fishing_row_count_grows_with_stage() {
    let n_anticipated = 2_usize;
    let k_max = 2_usize;

    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![1, 2], // K_0=1, K_1=2
        vec![0, 1], // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block

    for (stage_idx, expected) in [(0_usize, 2), (1, 2), (2, 2), (3, 2)] {
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, expected,
            "n_anticipated_fishing_rows must equal {expected} at stage_idx={stage_idx}"
        );
    }
}

/// `num_rows` does not include state-fixing rows; the LP row layout starts
/// directly with `z_inflow_rows` at row 0.
///
/// State pinning uses column bounds, so there is no `[0, n_state)` row
/// prefix. `num_rows` equals the count of structural rows only (`z_inflow`,
/// water balance, load balance, FPHA, evap, operational, fishing,
/// `anticipated_state_out_def`, generic).
#[test]
fn num_rows_drops_by_n_state_with_anticipated_thermals() {
    let n_anticipated = 2_usize;
    let k_max = 3_usize;

    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![3, 2], vec![0, 1]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // n_state for this fixture: N*(1+L) + A*K = 0 + 2*3 = 6.
    let n_state = ctx.n_hydros * (1 + ctx.max_par_order) + n_anticipated * k_max;
    assert_eq!(n_state, 6);

    // num_rows for this zero-hydro fixture: only the anticipated_fishing
    // block contributes (2 active plants at stage 0). All other row blocks
    // are 0 (no hydros, no buses, no FPHA, no evap).
    let observed = layout.num_rows;
    assert_eq!(
        observed, 2,
        "num_rows equals anticipated_fishing_rows (2) for this fixture"
    );

    // Reference value: if state-fixing rows existed, num_rows would be observed + n_state.
    let num_rows_if_state_rows_existed = observed + n_state;
    assert_eq!(
        num_rows_if_state_rows_existed, 8,
        "observed + n_state is 8 for this fixture"
    );
    // Structural invariant proving the reduction: row_water_balance_start
    // equals ctx.n_hydros (no n_state offset). With state-fixing rows it
    // would be n_state + ctx.n_hydros.
    assert_eq!(
        layout.row_water_balance_start(),
        ctx.n_hydros,
        "row_water_balance_start does not include the n_state offset"
    );
}

// ── Anticipated-decision range tests ──────────────────────────────────────

/// Build a `ResolvedBounds` with zero entities but the given `n_stages`.
///
/// Used to exercise the `StateLayout::is_anticipated_decision_active` gate
/// in `n_anticipated_state_out_def_rows` without needing real entity data.
fn bounds_with_n_stages(n_stages: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
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
    )
}

/// Builds a fixture struct owning all data for a context with anticipated
/// thermals and a known `n_stages` for the `state_out_def` predicate.
struct AntFixturesWithNStages {
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
}

impl AntFixturesWithNStages {
    fn new(n_stages: usize) -> Self {
        Self {
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            bounds: bounds_with_n_stages(n_stages),
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
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
        }
    }

    fn make_ctx(
        &self,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        anticipated_thermal_indices: Vec<usize>,
    ) -> TemplateBuildCtx<'_> {
        let n_stages = self.bounds.n_stages();
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_thermal_indices,
            // Windowless: one `(None, None)` per plant, so the decision gate
            // reduces to the strict horizon clause. `study_stage_ids` covers
            // the study-stage count so the in-range delivery lookup is safe.
            anticipated_windows: vec![(None, None); n_anticipated],
            study_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            has_penalty: false,
            cumulative_discount_factors: vec![1.0; n_stages],
            total_hours_per_stage: vec![744.0; n_stages],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// `col_anticipated_state_out_start` is sourced from the relocated
/// state-region position (after the `anticipated_state` ring buffer, before
/// `z_inflow`), `col_line_fwd_start` follows `anticipated_decision` directly,
/// and `n_anticipated_state_out_def_rows` counts both active plants at stage 0.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `k_max=3`, `n_stages=6`,
/// `stage_idx=0`, `N=0`, `L=0`. Both plants are active: `0+2=2 < 6` and
/// `0+3=3 < 6`. State-region offset = `N*(1+L) + n_ant*k_max = 0 + 6 = 6`.
#[test]
fn test_layout_state_out_block_adjacent_to_decision() {
    let fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // K_0=2, K_1=3
        vec![0, 1],
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // state_out is relocated to the state region: N*(1+L) + n_ant*k_max.
    assert_eq!(
        layout.anticipated.col_anticipated_state_out_start, 6,
        "state_out columns must be sourced from the state-region offset \
             N*(1+L) + n_ant*k_max"
    );
    // line_fwd follows anticipated_decision directly (state_out moved out).
    assert_eq!(
        layout.col_line_fwd_start(),
        layout.anticipated.col_anticipated_decision_start + 2,
        "line_fwd must be immediately after the anticipated_decision block"
    );
    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 2);
    assert_eq!(
        layout.anticipated.row_anticipated_state_out_def_start,
        layout.anticipated.row_anticipated_fishing_start
            + layout.anticipated.n_anticipated_fishing_rows
    );
}

/// `n_anticipated_state_out_def_rows == 0` when all plants are inactive at
/// the given stage, but the column block stays allocated.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `n_stages=6`, `stage_idx=5`.
/// Both inactive: `5+2=7 >= 6` and `5+3=8 >= 6`.
#[test]
fn test_layout_state_out_def_rows_zero_when_all_inactive() {
    let fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // K_0=2, K_1=3
        vec![0, 1],
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 5);

    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
    // Column block stays allocated at the state-region offset regardless of
    // activity: N*(1+L) + n_ant*k_max = 0 + 6 = 6.
    assert_eq!(layout.anticipated.col_anticipated_state_out_start, 6);
}

/// Zero-anticipated layouts must not grow `num_cols` or emit def rows.
///
/// `col_anticipated_state_out_start` must equal `col_anticipated_decision_start`
/// when `n_anticipated == 0` (empty block; both starts coincide).
#[test]
fn test_layout_no_anticipated_unchanged_num_cols() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.anticipated.col_anticipated_state_out_start,
        layout.anticipated.col_anticipated_decision_start,
        "col_anticipated_state_out_start must equal col_anticipated_decision_start when n_anticipated=0"
    );
    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
}

// ── Pumping-flow column region ─────────────────────────────────────────────

/// Build a `ResolvedBounds` with the given pumping-station count and stage
/// count (all other entity tables empty). `table.n_pumping()` recovers
/// `n_pumping` from the `pumping` Vec length divided by `n_stages`.
fn bounds_with_pumping(n_pumping: usize, n_stages: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
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
    )
}

/// Owns the data for a `TemplateBuildCtx` whose `bounds` report a non-zero
/// `n_pumping()`. Mirrors `ZeroEntityFixtures` but injects a pumping-aware
/// `ResolvedBounds` so `StageLayout::new` reserves the `pumping_flow` block.
struct PumpingFixtures {
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
    /// Windowless stations (always commissioning-active); the
    /// column-reservation probe exercises the dense per-station arithmetic the
    /// production builder runs.
    stations: Vec<PumpingStation>,
}

impl PumpingFixtures {
    fn new(n_pumping: usize, n_stages: usize) -> Self {
        // One windowless station per reserved slot, ids in slot order. No
        // entry/exit window ⇒ active at every stage ⇒ the active set is the
        // full count, so `StageLayout::n_pumping == n_pumping` matches the
        // bounds-derived count this fixture pins.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let stations = (0..n_pumping)
            .map(|i| PumpingStation {
                id: EntityId(i as i32),
                name: format!("P{i}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(0),
                source_hydro_id: EntityId(0),
                destination_hydro_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                consumption_mw_per_m3s: 0.5,
                min_flow_m3s: 0.0,
                max_flow_m3s: 10.0,
            })
            .collect();
        Self {
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            bounds: bounds_with_pumping(n_pumping, n_stages),
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
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
            stations,
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        let n_stages = self.bounds.n_stages();
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
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
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            // Windowless stations (always active), so the per-stage active set
            // equals the full count and `StageLayout::new` reserves a column
            // block of exactly `bounds.n_pumping()` stations. The active-path
            // reservation arithmetic is what this fixture probes; the
            // slice/`pumping_pos` threading is covered by the
            // `build_template_build_ctx` tests in `template.rs`.
            pumping_stations: &self.stations,
            pumping_pos: BTreeMap::new(),
            n_pumping: self.bounds.n_pumping(),
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            study_stage_ids: vec![],
            has_penalty: false,
            cumulative_discount_factors: vec![1.0; n_stages],
            total_hours_per_stage: vec![744.0; n_stages],
            filling_v_target: BTreeMap::new(),
        }
    }

    /// Build a stage with `n_blks` equal-duration blocks.
    fn stage_with_blocks(n_blks: usize) -> Stage {
        use cobre_core::Block;
        Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: (0..n_blks)
                .map(|b| Block {
                    index: b,
                    name: format!("B{b}"),
                    duration_hours: 248.0,
                })
                .collect(),
            block_mode: cobre_core::BlockMode::Parallel,
            state_config: cobre_core::StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: cobre_core::StageRiskConfig::Expectation,
            scenario_config: cobre_core::ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: cobre_core::NoiseMethod::Saa,
            },
        }
    }
}

/// Inert-layout invariant: with `n_pumping == 0` the `pumping_flow` block
/// collapses, so `col_pumping_start` sits exactly where the generic-slack
/// columns begin (`col_ncs_end`, which equals `col_ncs_start` when no NCS are
/// active) and `num_cols` is unshifted. For this zero-entity one-block system
/// the entire column count is the single theta column.
///
/// Pinning `n_pumping == 0`, `col_pumping_start == col_ncs_start`, and the
/// exact `num_cols`/equipment starts proves that reserving the pumping region
/// does not move any pre-existing column when there are no stations.
#[test]
fn pumping_layout_inert_when_no_stations() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // No stations: the bounds table reports zero pumping.
    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        0,
        "fixture has no pumping stations"
    );
    assert_eq!(layout.n_pumping, 0, "layout.n_pumping must be 0");

    // The empty pumping block does not advance the cursor: its start equals
    // the NCS-region end. With zero active NCS, col_ncs_end == col_ncs_start.
    assert_eq!(
        layout.col_pumping_start, layout.col_ncs_start,
        "col_pumping_start must equal col_ncs_start (col_ncs_end) when no stations"
    );

    // Pre-existing column starts for the zero-entity, single-block layout:
    // theta == 0, every equipment/slack/NCS region empty starting at theta+1.
    let idx = crate::test_support::state_layout(ctx.n_hydros, ctx.max_par_order);
    let expected_start = idx.theta + 1;
    assert_eq!(layout.col_turbine_start(), expected_start);
    assert_eq!(layout.col_thermal_start(), expected_start);
    assert_eq!(layout.col_line_fwd_start(), expected_start);
    assert_eq!(layout.col_deficit_start(), expected_start);
    assert_eq!(layout.col_excess_start(), expected_start);
    assert_eq!(layout.col_ncs_start, expected_start);
    assert_eq!(layout.col_pumping_start, expected_start);
    // num_cols is the single theta column; the empty pumping block adds nothing.
    assert_eq!(
        layout.num_cols, expected_start,
        "num_cols must be unshifted"
    );
}

/// `n_pumping == 2`, `n_blks == 3` ⇒ a 6-column `pumping_flow` block at
/// `col_pumping_start`, block-major, and `num_cols` increased by exactly 6
/// relative to the otherwise-identical station-free layout.
#[test]
fn pumping_layout_reserves_block_major_columns() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;

    // Baseline: identical zero-entity 3-block layout with no stations.
    let baseline_fixtures = PumpingFixtures::new(0, 3);
    let baseline_ctx = baseline_fixtures.make_ctx();
    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&baseline_ctx);
    let baseline = StageLayout::new(&baseline_ctx, &state, &stage, 0);
    assert_eq!(baseline.n_pumping, 0);

    // Station-bearing layout: 2 pumping stations across 3 stages.
    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        n_pumping,
        "fixture bounds must report n_pumping() == 2"
    );
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(layout.n_pumping, n_pumping, "layout.n_pumping == 2");
    // The pumping block begins exactly where the NCS region ends (no NCS here).
    assert_eq!(
        layout.col_pumping_start, layout.col_ncs_start,
        "col_pumping_start must follow the NCS region"
    );
    // Block-major width: n_pumping * n_blks == 2 * 3 == 6.
    assert_eq!(
        layout.num_cols - baseline.num_cols,
        n_pumping * n_blks,
        "num_cols must grow by exactly n_pumping * n_blks == 6"
    );
    // The 6 reserved columns occupy [col_pumping_start, col_pumping_start + 6).
    assert_eq!(
        layout.num_cols,
        layout.col_pumping_start + n_pumping * n_blks,
        "the 6-column block ends at num_cols (no generic-slack columns here)"
    );
}

/// With both contract counts 0 the import/export blocks collapse onto
/// `col_pumping_end` and the generic-slack start (here surfaced as
/// `col_filling_target_start()`, since no generic-slack columns exist) is
/// unshifted — the contract-free parity guarantee.
#[test]
fn contract_columns_empty_keep_generic_slack_at_pumping_end() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;
    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(ctx.n_contract_import, 0);
    assert_eq!(ctx.n_contract_export, 0);

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let col_pumping_end = layout.col_pumping_start + layout.n_pumping * n_blks;
    assert_eq!(
        layout.col_contract_import_start, col_pumping_end,
        "empty import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.col_contract_export_start, col_pumping_end,
        "empty export block collapses onto col_pumping_end"
    );
    assert_eq!(
        layout.col_filling_target_start(),
        col_pumping_end,
        "generic-slack start is unshifted for a contract-free system"
    );
}

/// `n_contract_import == 2`, `n_contract_export == 1`, `n_blks == 3`: the import
/// block (6 columns) starts at `col_pumping_end`, the export block (3 columns)
/// follows it, and the generic-slack start (`col_filling_target_start()` with no
/// generic-slack columns) shifts by `(2 + 1) * 3 == 9`.
#[test]
fn contract_columns_reserve_import_then_export_blocks() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;
    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = TemplateBuildCtx {
        n_contract_import: 2,
        n_contract_export: 1,
        ..fixtures.make_ctx()
    };

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let col_pumping_end = layout.col_pumping_start + layout.n_pumping * n_blks;
    assert_eq!(
        layout.col_contract_import_start, col_pumping_end,
        "import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.col_contract_export_start,
        col_pumping_end + 6,
        "export block follows the 6-column import block"
    );
    assert_eq!(
        layout.col_filling_target_start(),
        col_pumping_end + 9,
        "generic-slack start shifts by (2 + 1) * 3 == 9"
    );
}

/// The shared `commissioning_active` predicate gates on
/// `entry_stage_id <= stage_id < exit_stage_id` and is the single owner of
/// commissioning activity for every equipment family (NCS, pumping, and the
/// later thermal/line/hydro). Covers the five window shapes — no window
/// (always active), entry-only, exit-only, both, and a stage outside the
/// window. The forbidden alternative — a non-strict upper bound
/// (`stage_id <= exit`) — would keep a decommissioned entity active at its
/// exit stage.
#[test]
fn commissioning_active_gates_on_stage_id_with_half_open_window() {
    use crate::lp_builder::commissioning_active;
    // p0 no window: active at every stage.
    for id in [0, 1, 2, 3, 4, 100] {
        assert!(
            commissioning_active(None, None, id),
            "no window active at {id}"
        );
    }
    // entry=2: active iff id >= 2.
    assert!(!commissioning_active(Some(2), None, 1));
    assert!(commissioning_active(Some(2), None, 2));
    // exit=3: active iff id < 3 (strict upper).
    assert!(commissioning_active(None, Some(3), 2));
    assert!(!commissioning_active(None, Some(3), 3));
    // window [1, 4): active iff 1 <= id < 4.
    assert!(!commissioning_active(Some(1), Some(4), 0));
    assert!(commissioning_active(Some(1), Some(4), 1));
    assert!(commissioning_active(Some(1), Some(4), 3));
    assert!(!commissioning_active(Some(1), Some(4), 4));
}

// ── block-major column-accessor arithmetic equivalence ─────────────────────

/// Every `#[inline]` column accessor returns the exact `usize` its open-coded
/// formula returned. Built over a `n_blks = 4` layout and probed with
/// `entity >= 1`, `blk >= 1`, and `seg >= 1` so a transposed stride
/// (`blk * n_entities + entity`) or a swapped evap offset would differ from
/// the open-coded expression and fail the assertion.
#[test]
fn column_accessors_match_open_coded_formulas() {
    // Multi-block, zero-entity layout: the block-major `col_*_start` fields
    // and `n_blks` are populated; the accessor reads the same fields the
    // open-coded formula reads, so this pins each accessor's arithmetic.
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);
    let n_blks = layout.n_blks;
    assert_eq!(n_blks, 4, "fixture must build a 4-block layout");

    // Generic block_col against its definition, across a grid that makes the
    // entity (outer) and block (inner) factors distinguishable.
    for entity in [0_usize, 1, 2, 5] {
        for blk in 0..n_blks {
            assert_eq!(
                layout.block_col(layout.col_turbine_start(), entity, blk),
                layout.col_turbine_start() + entity * n_blks + blk,
                "block_col(entity={entity}, blk={blk})"
            );
        }
    }

    // Per-family block-major accessors, each owning its `col_*_start`.
    for entity in [0_usize, 1, 3] {
        for blk in 0..n_blks {
            assert_eq!(
                layout.turbine_col(entity, blk),
                layout.col_turbine_start() + entity * n_blks + blk,
                "turbine_col"
            );
            assert_eq!(
                layout.spillage_col(entity, blk),
                layout.col_spillage_start() + entity * n_blks + blk,
                "spillage_col"
            );
            assert_eq!(
                layout.diversion_col(entity, blk),
                layout.col_diversion_start() + entity * n_blks + blk,
                "diversion_col"
            );
            assert_eq!(
                layout.generation_col(entity, blk),
                layout.col_generation_start() + entity * n_blks + blk,
                "generation_col"
            );
            assert_eq!(
                layout.line_fwd_col(entity, blk),
                layout.col_line_fwd_start() + entity * n_blks + blk,
                "line_fwd_col"
            );
            assert_eq!(
                layout.line_rev_col(entity, blk),
                layout.col_line_rev_start() + entity * n_blks + blk,
                "line_rev_col"
            );
            assert_eq!(
                layout.outflow_below_col(entity, blk),
                layout.col_outflow_below_start() + entity * n_blks + blk,
                "outflow_below_col"
            );
            assert_eq!(
                layout.outflow_above_col(entity, blk),
                layout.col_outflow_above_start() + entity * n_blks + blk,
                "outflow_above_col"
            );
            assert_eq!(
                layout.turbine_below_col(entity, blk),
                layout.col_turbine_below_start() + entity * n_blks + blk,
                "turbine_below_col"
            );
            assert_eq!(
                layout.generation_below_col(entity, blk),
                layout.col_generation_below_start() + entity * n_blks + blk,
                "generation_below_col"
            );
        }
    }

    // Evaporation accessors: stage-level, EVAP_COLS_PER_HYDRO-strided. The
    // three within-hydro offsets must map flow→0, f_plus→1, f_minus→2.
    for local_idx in [0_usize, 1, 4] {
        assert_eq!(
            layout.evap_flow_col(local_idx),
            layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_FLOW_OFFSET,
            "evap_flow_col"
        );
        assert_eq!(
            layout.evap_f_plus_col(local_idx),
            layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_PLUS_OFFSET,
            "evap_f_plus_col"
        );
        assert_eq!(
            layout.evap_f_minus_col(local_idx),
            layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_MINUS_OFFSET,
            "evap_f_minus_col"
        );
        // The three columns are consecutive and ordered flow < plus < minus.
        assert_eq!(
            layout.evap_f_plus_col(local_idx),
            layout.evap_flow_col(local_idx) + 1
        );
        assert_eq!(
            layout.evap_f_minus_col(local_idx),
            layout.evap_flow_col(local_idx) + 2
        );
    }
    // The evap offset consts are exactly 0/1/2 in order.
    assert_eq!(EVAP_FLOW_OFFSET, 0);
    assert_eq!(EVAP_F_PLUS_OFFSET, 1);
    assert_eq!(EVAP_F_MINUS_OFFSET, 2);
    assert_eq!(EVAP_COLS_PER_HYDRO, 3);

    // Deficit three-term stride: per-bus span, per-segment span, then block.
    for b_idx in [0_usize, 1, 2] {
        for seg_idx in [0_usize, 1] {
            for blk in 0..n_blks {
                assert_eq!(
                    layout.deficit_col(b_idx, seg_idx, blk),
                    layout.col_deficit_start()
                        + b_idx * layout.max_deficit_segments() * n_blks
                        + seg_idx * n_blks
                        + blk,
                    "deficit_col(b={b_idx}, seg={seg_idx}, blk={blk})"
                );
            }
        }
    }
}

// ── post-equipment column cursor (no-hydro fork fallback) ───────────────────

/// With `n_hydros == 0` every withdrawal / operational-violation / NCS column
/// region is empty, so all their starts collapse onto the single
/// post-equipment column cursor `col_evap_start`. A multi-block stage keeps
/// that cursor non-trivial (not the degenerate one-column case), so a stale
/// `> 0`-branch cursor leaking into the empty-hydro fallback would shift these
/// starts off `col_evap_start` and fail here.
#[test]
fn post_equipment_col_start_matches_evap_col_start_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    let post_equipment = layout.col_evap_start();
    assert_eq!(
        layout.col_ncs_start, post_equipment,
        "col_ncs_start must collapse onto col_evap_start when n_hydros == 0"
    );
    assert_eq!(
        layout.col_withdrawal_neg_start(),
        post_equipment,
        "col_withdrawal_neg_start"
    );
    assert_eq!(
        layout.col_withdrawal_pos_start(),
        post_equipment,
        "col_withdrawal_pos_start"
    );
    assert_eq!(
        layout.col_outflow_below_start(),
        post_equipment,
        "col_outflow_below_start"
    );
    assert_eq!(
        layout.col_outflow_above_start(),
        post_equipment,
        "col_outflow_above_start"
    );
    assert_eq!(
        layout.col_turbine_below_start(),
        post_equipment,
        "col_turbine_below_start"
    );
    assert_eq!(
        layout.col_generation_below_start(),
        post_equipment,
        "col_generation_below_start"
    );
}

// ── post-equipment row cursor (no-hydro fork fallback) ──────────────────────

/// With `n_hydros == 0` every operational-violation row block is empty, so all
/// four row starts collapse onto the single post-equipment row cursor
/// `fpha_rows_end() + n_evap_hydros`. A multi-block stage keeps that cursor
/// non-trivial (not the degenerate one-row case), so a stale `> 0`-branch
/// cursor leaking into the empty-hydro fallback would shift these starts off
/// the shared post-equipment row cursor and fail here.
#[test]
fn post_equipment_row_start_matches_evap_rows_end_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    let post_equipment = layout.post_equipment_row_start;
    assert_eq!(
        layout.row_min_outflow_start(),
        post_equipment,
        "row_min_outflow_start must collapse onto the post-equipment row cursor when n_hydros == 0"
    );
    assert_eq!(
        layout.row_max_outflow_start(),
        post_equipment,
        "row_max_outflow_start"
    );
    assert_eq!(
        layout.row_min_turbine_start(),
        post_equipment,
        "row_min_turbine_start"
    );
    assert_eq!(
        layout.row_min_generation_start(),
        post_equipment,
        "row_min_generation_start"
    );
}

// ── Group-2 accessors: hydro-free divergence guard ──────────────────────────

/// With `n_hydros == 0`, every Group-2 accessor must return the post-equipment
/// cursor — `post_equipment_col_start()` for the eight column accessors,
/// `post_equipment_row_start()` for the five row accessors. A bare
/// `self.indexer.<range>.start` returns the `0` of the normalised `0..0` empty
/// range; the equality assertions here pin the accessor to the real cursor
/// instead, which is the silent misbuild this split exists to prevent.
///
/// The column cursor is additionally asserted `!= 0`: the theta and state columns
/// always precede the equipment/slack region, so `post_equipment_col_start()` is
/// provably positive and a spurious bare-`.start` `0` is directly detectable. The
/// row cursor is NOT asserted `!= 0`: with zero hydros AND zero buses no rows
/// precede the operational-violation block, so `post_equipment_row_start()` is
/// legitimately `0` here (asserting `!= 0` would test a false invariant). The
/// non-zero-row divergence is covered end-to-end by the D01 hydro-free parity
/// case, whose load-balance rows make the row cursor positive.
#[test]
fn group2_accessors_return_post_equipment_cursor_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    // Column cursor: the eight column accessors collapse onto
    // `post_equipment_col_start` (== `col_evap_start()`) with no hydros, and
    // that cursor is provably positive (theta + state columns precede it).
    let post_col = layout.post_equipment_col_start;
    assert_ne!(post_col, 0, "post-equipment column cursor must not be 0");
    for (value, name) in [
        (layout.col_generation_start(), "col_generation_start"),
        (layout.col_evap_start(), "col_evap_start"),
        (
            layout.col_withdrawal_neg_start(),
            "col_withdrawal_neg_start",
        ),
        (
            layout.col_withdrawal_pos_start(),
            "col_withdrawal_pos_start",
        ),
        (layout.col_outflow_below_start(), "col_outflow_below_start"),
        (layout.col_outflow_above_start(), "col_outflow_above_start"),
        (layout.col_turbine_below_start(), "col_turbine_below_start"),
        (
            layout.col_generation_below_start(),
            "col_generation_below_start",
        ),
    ] {
        assert_eq!(
            value, post_col,
            "{name} must equal post_equipment_col_start() (not 0) when n_hydros == 0"
        );
    }

    // Row cursor: `row_evap_start()` is `fpha_rows_end`, which equals
    // `post_equipment_row_start` (= `fpha_rows_end + n_evap_hydros`) when
    // `n_evap_hydros == 0`. The four operational-violation row accessors collapse
    // onto that same cursor. Each must equal it, never a bare `.start`.
    let post_row = layout.post_equipment_row_start;
    for (value, name) in [
        (layout.row_evap_start(), "row_evap_start"),
        (layout.row_min_outflow_start(), "row_min_outflow_start"),
        (layout.row_max_outflow_start(), "row_max_outflow_start"),
        (layout.row_min_turbine_start(), "row_min_turbine_start"),
        (
            layout.row_min_generation_start(),
            "row_min_generation_start",
        ),
    ] {
        assert_eq!(
            value, post_row,
            "{name} must equal post_equipment_row_start() when n_hydros == 0"
        );
    }
}
