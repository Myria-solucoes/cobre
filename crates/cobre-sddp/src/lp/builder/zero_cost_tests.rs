#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds,
    HydroStageBounds, LineStageBounds, NoiseMethod, PumpingStageBounds, ResolvedBounds,
    ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
    ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage,
    StageRiskConfig, StageStateConfig, ThermalStageBounds,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
use crate::resolved_parameters::ResolvedParameters;

use super::columns::{
    ColumnBufs, fill_anticipated_state_out_columns, zero_anticipated_delivery_thermal_cost,
};
use super::entries::{
    build_stage_matrix_entries, fill_anticipated_fishing_entries,
    fill_anticipated_state_out_def_entries,
};
use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
use super::rows::{fill_anticipated_fishing_rows, fill_anticipated_state_out_def_rows};

/// Build a minimal two-block `Stage` at the given index.
fn two_block_stage(index: usize) -> Stage {
    Stage {
        index,
        id: index as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![
            Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 372.0,
            },
            Block {
                index: 1,
                name: "BLK1".to_string(),
                duration_hours: 372.0,
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

/// Owns data for a context with anticipated thermals and zero other entities.
struct AntFixtures {
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

impl AntFixtures {
    /// Build a minimal `ResolvedBounds` with zero entities and the given `n_stages`.
    ///
    /// `n_stages` must exceed the queried `stage_idx` (and each plant's
    /// delivery stage `stage_idx + K_i`) so the anticipated layout the
    /// fixture builds places every plant inside the study horizon
    /// `[0, n_stages)`.
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
            },
        )
    }

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

    fn make_ctx(
        &self,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        anticipated_thermal_indices: Vec<usize>,
        n_thermals: usize,
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
            hydro_pos: HashMap::new(),
            thermal_pos: HashMap::new(),
            line_pos: HashMap::new(),
            bus_pos: HashMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: HashMap::new(),
            n_pumping: 0,
            diversion_upstream: HashMap::new(),
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_thermal_indices,
            has_penalty: false,
            cumulative_discount_factors: vec![1.0],
            total_hours_per_stage: vec![744.0],
        }
    }
}

/// Zeroes cost for every anticipated plant (the fishing constraint is
/// always active, one plant per iteration).
/// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants zeroed.
#[test]
fn zero_anticipated_delivery_thermal_cost_zeroes_all_plants() {
    let mut fixtures = AntFixtures::new();
    // Override bounds so every plant's delivery stage `stage_idx + K_i`
    // falls inside the horizon `[0, n_stages)` when the layout is built.
    fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        5,          // k_max (max of [1, 5])
        vec![1, 5], // anticipated_lead_stages: K_0 = 1, K_1 = 5
        vec![0, 1], // anticipated_thermal_indices: positions in ctx.thermals
        2,          // n_thermals
    );
    let stage = two_block_stage(2);
    let layout = StageLayout::new(&ctx, &stage, 2);

    // Allocate objective buffer at full column width, pre-filled with a
    // sentinel so any un-zeroed entry is visible in the assertions.
    let mut objective = vec![42.0_f64; layout.num_cols];
    let mut col_lower = vec![0.0_f64; layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut bufs = ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };

    zero_anticipated_delivery_thermal_cost(&ctx, &layout, &mut bufs);

    let n_blks = layout.n_blks;
    // Plant 0 (K_0=1): zeroed under always-active predicate.
    let thermal_idx_0 = ctx.anticipated_thermal_indices[0];
    for blk in 0..n_blks {
        let col = layout.col_thermal_start() + thermal_idx_0 * n_blks + blk;
        assert_eq!(
            bufs.objective[col], 0.0,
            "plant 0 must be zeroed at col {col}",
        );
    }
    // Plant 1 (K_1=5): also zeroed under always-active predicate.
    let thermal_idx_1 = ctx.anticipated_thermal_indices[1];
    for blk in 0..n_blks {
        let col = layout.col_thermal_start() + thermal_idx_1 * n_blks + blk;
        assert_eq!(
            bufs.objective[col], 0.0,
            "plant 1 must be zeroed at col {col}",
        );
    }
}

/// Fills rows for all anticipated plants (always-active predicate).
/// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants are active,
/// so exactly two fishing rows are written.
#[test]
fn fishing_rows_fill_all_plants() {
    let mut fixtures = AntFixtures::new();
    fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
    let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
    let stage = two_block_stage(2);
    let layout = StageLayout::new(&ctx, &stage, 2);

    // Always-active: both plants active at stage 2 → two fishing rows.
    assert_eq!(
        layout.anticipated.n_anticipated_fishing_rows, 2,
        "expected n_anticipated_fishing_rows == 2, got {}",
        layout.anticipated.n_anticipated_fishing_rows
    );

    let mut row_lower = vec![f64::NAN; layout.num_rows];
    let mut row_upper = vec![f64::NAN; layout.num_rows];

    fill_anticipated_fishing_rows(&ctx, &layout, &mut row_lower, &mut row_upper);

    // Both plants write a row with (0.0, 0.0) bounds.
    for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        assert_eq!(
            row_lower[row], 0.0,
            "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
            row_lower[row]
        );
        assert_eq!(
            row_upper[row], 0.0,
            "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
            row_upper[row]
        );
    }
}

/// Always-active at `stage_idx = 0`: with `K = [1, 5]` and `n_anticipated = 2`,
/// both plants are active even before their lead time elapses.
/// Asserts `layout.anticipated.n_anticipated_fishing_rows == 2`, that both rows
/// are filled with `(0.0, 0.0)` bounds, and that the anticipated-state
/// slot-0 column carries the `-block_hours_total` coupling for both plants.
#[test]
fn fishing_rows_always_active_stage_zero() {
    let mut fixtures = AntFixtures::new();
    fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
    let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
    let stage = two_block_stage(0);
    let layout = StageLayout::new(&ctx, &stage, 0);

    // Always-active: both plants are active at stage 0 → two fishing rows.
    assert_eq!(
        layout.anticipated.n_anticipated_fishing_rows, 2,
        "expected n_anticipated_fishing_rows == 2 at stage 0, got {}",
        layout.anticipated.n_anticipated_fishing_rows
    );

    let mut row_lower = vec![f64::NAN; layout.num_rows];
    let mut row_upper = vec![f64::NAN; layout.num_rows];

    fill_anticipated_fishing_rows(&ctx, &layout, &mut row_lower, &mut row_upper);

    // Both plants write equality rows with (0.0, 0.0) bounds.
    for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        assert_eq!(
            row_lower[row], 0.0,
            "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
            row_lower[row]
        );
        assert_eq!(
            row_upper[row], 0.0,
            "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
            row_upper[row]
        );
    }

    // CSC coupling: anticipated_state slot-0 column carries (row, -block_hours_total)
    // for each plant under the always-active predicate.
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_anticipated_fishing_entries(&ctx, &stage, &layout, &mut col_entries);

    let block_hours_total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
    let expected_neg = -block_hours_total;
    for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        let col_state = layout.col_anticipated_state_start() + local_idx;
        let state_couplings: Vec<&(usize, f64)> = col_entries[col_state]
            .iter()
            .filter(|(r, _)| *r == row)
            .collect();
        assert_eq!(
            state_couplings.len(),
            1,
            "anticipated_state col {col_state} must carry exactly 1 coupling \
             at fishing row {row} (plant local_idx={local_idx})"
        );
        let (_, coeff) = state_couplings[0];
        assert!(
            (coeff - expected_neg).abs() < 1e-12,
            "anticipated_state col {col_state} fishing-row coupling: \
             expected {expected_neg}, got {coeff} (plant local_idx={local_idx})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Fixture helper for anticipated state-out tests
// ─────────────────────────────────────────────────────────────────────────

/// Build a minimal fixture for anticipated state-out tests:
/// `n_anticipated = 2`, `K = [2, 3]`, `n_stages = 6`.
///
/// `ResolvedBounds` is constructed with the correct `n_stages` so that
/// `ctx.resolved.bounds.n_stages()` returns 6, which is required by
/// `fill_anticipated_state_out_columns`, `fill_anticipated_state_out_def_rows`,
/// and `fill_anticipated_state_out_def_entries`.
fn build_anticipated_ctx_n_stages_6() -> (AntFixtures, Stage) {
    let mut fixtures = AntFixtures::new();
    // Override bounds with a 6-stage table (zero entities, k_max = 3).
    fixtures.bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 6,
            k_max: 3,
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
        },
    );
    let stage = two_block_stage(0);
    (fixtures, stage)
}

// ─────────────────────────────────────────────────────────────────────────
// Tests for fill_anticipated_state_out_columns
// ─────────────────────────────────────────────────────────────────────────

/// Active plants (`stage_idx + K_i < n_stages`) get `[-INF, +INF]` bounds;
/// inactive plants (`stage_idx + K_i >= n_stages`) get `[0, 0]` bounds.
///
/// Fixture: `n_anticipated=2`, `K=[2, 3]`, `n_stages=6`.
/// Stage 0: both plants active  (0+2=2 < 6, 0+3=3 < 6) → `[-INF, +INF]`.
/// Stage 5: both plants inactive (5+2=7 >= 6, 5+3=8 >= 6) → `[0, 0]`.
#[test]
fn test_fill_anticipated_state_out_columns_active_and_inactive() {
    let (fixtures, _) = build_anticipated_ctx_n_stages_6();
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // anticipated_lead_stages: K=[2,3]
        vec![0, 1], // anticipated_thermal_indices
        0,          // n_thermals
    );

    // Stage 0: both plants active.
    let stage0 = two_block_stage(0);
    let layout0 = StageLayout::new(&ctx, &stage0, 0);
    let mut col_lower = vec![0.0_f64; layout0.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout0.num_cols];
    let mut objective = vec![0.0_f64; layout0.num_cols];
    let mut bufs = ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };
    fill_anticipated_state_out_columns(&ctx, 0, &layout0, &mut bufs);
    for i in 0..2 {
        let col = layout0.anticipated.col_anticipated_state_out_start + i;
        assert_eq!(
            col_lower[col],
            f64::NEG_INFINITY,
            "stage 0, plant {i}: col_lower expected -INF, got {}",
            col_lower[col]
        );
        assert_eq!(
            col_upper[col],
            f64::INFINITY,
            "stage 0, plant {i}: col_upper expected +INF, got {}",
            col_upper[col]
        );
    }

    // Stage 5: both plants inactive.
    let stage5 = two_block_stage(5);
    let layout5 = StageLayout::new(&ctx, &stage5, 5);
    let mut col_lower5 = vec![0.0_f64; layout5.num_cols];
    let mut col_upper5 = vec![f64::INFINITY; layout5.num_cols];
    let mut objective5 = vec![0.0_f64; layout5.num_cols];
    let mut bufs5 = ColumnBufs {
        col_lower: &mut col_lower5,
        col_upper: &mut col_upper5,
        objective: &mut objective5,
    };
    fill_anticipated_state_out_columns(&ctx, 5, &layout5, &mut bufs5);
    assert_eq!(
        layout5.anticipated.n_anticipated_state_out_def_rows, 0,
        "stage 5 inactive: expected no def rows, got {}",
        layout5.anticipated.n_anticipated_state_out_def_rows,
    );
    for i in 0..2 {
        let col = layout5.anticipated.col_anticipated_state_out_start + i;
        assert_eq!(
            col_lower5[col], 0.0,
            "stage 5, plant {i}: col_lower expected 0.0, got {}",
            col_lower5[col]
        );
        assert_eq!(
            col_upper5[col], 0.0,
            "stage 5, plant {i}: col_upper expected 0.0, got {}",
            col_upper5[col]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests for fill_anticipated_state_out_def_rows
// ─────────────────────────────────────────────────────────────────────────

/// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active
/// (0+2 < 6, 0+3 < 6), so `n_anticipated_state_out_def_rows == 2` and
/// both definition rows must have equality bounds `[0.0, 0.0]`.
#[test]
fn test_fill_anticipated_state_out_def_rows_two_active_plants() {
    let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
    let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.anticipated.n_anticipated_state_out_def_rows, 2,
        "expected n_anticipated_state_out_def_rows == 2, got {}",
        layout.anticipated.n_anticipated_state_out_def_rows
    );

    let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
    let mut row_upper = vec![f64::INFINITY; layout.num_rows];
    fill_anticipated_state_out_def_rows(&ctx, 0, &layout, &mut row_lower, &mut row_upper);

    for k in 0..2 {
        let row = layout.anticipated.row_anticipated_state_out_def_start + k;
        assert_eq!(
            row_lower[row], 0.0,
            "def row {k}: row_lower expected 0.0, got {}",
            row_lower[row]
        );
        assert_eq!(
            row_upper[row], 0.0,
            "def row {k}: row_upper expected 0.0, got {}",
            row_upper[row]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests for fill_anticipated_state_out_def_entries
// ─────────────────────────────────────────────────────────────────────────

/// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active.
/// For each active plant `i`, the CSC entry list must contain:
/// - `(def_row_i, +1.0)` on `col_anticipated_state_out_start + i`
/// - `(def_row_i, -1.0)` on `col_anticipated_decision_start + i`
#[test]
fn test_fill_anticipated_state_out_def_entries_two_active_plants() {
    let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
    let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    fill_anticipated_state_out_def_entries(&ctx, 0, &layout, &mut col_entries);

    for k in 0..2 {
        let row = layout.anticipated.row_anticipated_state_out_def_start + k;
        let col_state_out = layout.anticipated.col_anticipated_state_out_start + k;
        let col_decision = layout.anticipated.col_anticipated_decision_start + k;

        assert!(
            col_entries[col_state_out]
                .iter()
                .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
            "plant {k}: expected (+1.0) entry at (col_state_out={col_state_out}, row={row}), \
             got {:?}",
            col_entries[col_state_out]
        );
        assert!(
            col_entries[col_decision]
                .iter()
                .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
            "plant {k}: expected (-1.0) entry at (col_decision={col_decision}, row={row}), \
             got {:?}",
            col_entries[col_decision]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// State-fixing diagonals must be absent from the CSC output
// ─────────────────────────────────────────────────────────────────────────

/// Asserts `build_stage_matrix_entries` produces no state-fixing
/// diagonals in the CSC output.
///
/// Coverage strategy: storage-fixing and lag-fixing diagonals are
/// guaranteed absent by structural deletion of their for-loops in
/// `fill_state_and_water_entries` (verified by C1+C2 grep — the
/// functions/loops emitting those entries no longer exist in the
/// source). Anticipated-state-fixing diagonals are checked dynamically:
/// the test builds a fixture with `n_anticipated = 2, k_max = 3` and
/// asserts every `(slot, plant)` column at
/// `col_anticipated_state_start + slot*A + plant` has no entry at
/// row `slot*A + plant` (the diagonal entry that existed in the
/// pre-cutover layout, before state pinning moved to column bounds).
///
/// The storage/lag assertions are included as zero-iteration loops in
/// this fixture (`n_hydros` = 0) so the test documents the intent and
/// would catch a future regression in any fixture that adds hydros.
#[test]
fn state_fixing_diagonals_absent_from_csc() {
    let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // anticipated_lead_stages
        vec![0, 1], // anticipated_thermal_indices
        0,          // n_thermals
    );

    let layout = StageLayout::new(&ctx, &stage, 0);
    let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

    let a = ctx.n_anticipated;
    let k = ctx.k_max;
    for slot in 0..k {
        for plant in 0..a {
            let col = layout.col_anticipated_state_start() + slot * a + plant;
            let diag_row = slot * a + plant;
            let has_diag = col_entries[col]
                .iter()
                .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
            assert!(
                !has_diag,
                "anticipated-state-fixing diagonal (row={diag_row}, val=1.0) must be absent \
                 from col {col} (slot={slot}, plant={plant})"
            );
        }
    }

    // Storage-fixing and lag-fixing diagonal absence assertions. With
    // n_hydros = 0 in this fixture these loops execute zero iterations,
    // but the structure documents intent and the same assertion shape
    // catches a regression in any future fixture with non-zero hydros.
    let n_h = ctx.n_hydros;
    let lag_order = ctx.max_par_order;
    for h in 0..n_h {
        let col = layout.col_storage_in_start() + h;
        let has_diag = col_entries[col]
            .iter()
            .any(|&(r, v)| r == h && (v - 1.0).abs() < 1e-15);
        assert!(
            !has_diag,
            "storage-fixing diagonal (row={h}, val=1.0) must be absent from col {col}"
        );
    }
    for lag in 0..lag_order {
        for h in 0..n_h {
            let col = layout.col_inflow_lags_start() + lag * n_h + h;
            let diag_row = n_h + lag * n_h + h;
            let has_diag = col_entries[col]
                .iter()
                .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
            assert!(
                !has_diag,
                "lag-fixing diagonal (row={diag_row}, val=1.0) must be absent from col {col} \
                 (lag={lag}, h={h})"
            );
        }
    }
}
