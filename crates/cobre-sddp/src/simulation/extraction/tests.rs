#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use std::collections::HashMap;

use super::{
    EntityCounts, HydroReverseLookup, SolutionView, StageExtractionSpec, accumulate_category_costs,
    assign_scenarios, extract_contracts, extract_pumping_stations, extract_stage_result,
    extract_stub_collections,
};
use crate::indexer::StudyDimensions;
use crate::lp_builder::StageGeometry;
use crate::simulation::types::{
    ScenarioCategoryCosts, SimulationContractResult, SimulationCostResult,
};

// -------------------------------------------------------------------------
// HydroReverseLookup per-stage membership
// -------------------------------------------------------------------------

/// Pins that `build_per_stage` reads each stage's own `fpha_hydro_indices`, not
/// a single global stage-0 list: hydro 1 is FPHA only at stage 1.
#[test]
fn build_per_stage_resolves_stage_varying_fpha_membership() {
    let n_hydros = 2;

    // Stage 0: only hydro 0 is FPHA. Stage 1: hydros 0 and 1 are both FPHA,
    // so hydro 1's FPHA-local slot is 1. The `generation` range is the stage's
    // FPHA-generation column block (one column per FPHA hydro here, n_blks = 1);
    // its base differs per stage but only the membership lists drive the lookup.
    let geom_stage0 = StageGeometry {
        fpha_hydro_indices: vec![0],
        generation: 100..101,
        ..StageGeometry::default()
    };
    let geom_stage1 = StageGeometry {
        fpha_hydro_indices: vec![0, 1],
        generation: 200..202,
        ..StageGeometry::default()
    };
    let geometry_per_stage = vec![geom_stage0, geom_stage1];

    let hydro_per_stage = HydroReverseLookup::build_per_stage(&geometry_per_stage, n_hydros);
    assert_eq!(hydro_per_stage.len(), 2);

    // Hydro 1 is absent from stage 0's FPHA list and present at FPHA-local slot
    // 1 in stage 1's — the membership genuinely differs by stage.
    assert_eq!(hydro_per_stage[0].fpha[1], None);
    assert_eq!(hydro_per_stage[1].fpha[1], Some(1));

    // Hydro 0 is FPHA at both stages, always at FPHA-local slot 0.
    assert_eq!(hydro_per_stage[0].fpha[0], Some(0));
    assert_eq!(hydro_per_stage[1].fpha[0], Some(0));

    // The stage-1 FPHA-local slot for hydro 1 is the index an extraction read at
    // stage 1 uses into `geometry[1].generation`: slot 1 addresses column
    // `generation.start + 1` (= 201), the column the solved primal occupies. A
    // stage-0 lookup would have reported `None` and skipped this read entirely.
    let stage1_local = hydro_per_stage[1].fpha[1].expect("hydro 1 is FPHA at stage 1");
    assert_eq!(geometry_per_stage[1].generation.start + stage1_local, 201);
}

// -------------------------------------------------------------------------
// Filling-slack (σ_fill / σ^{v-}) reverse-lookup membership
// -------------------------------------------------------------------------

/// Pins that the SPARSE `σ_fill` / `σ^{v-}` families resolve through a
/// system→slot reverse map (filling hydro `Some(slot)`, others `None`) and are
/// INDEPENDENT: a hydro can own a `σ^{v-}` column without a `σ_fill` column.
#[test]
fn filling_reverse_lookup_resolves_sparse_membership() {
    let n_hydros = 3;
    // Hydro 2 owns the (single) terminal σ_fill column at this stage; hydro 0
    // owns the σ^{v-} operating-floor column. Hydro 1 owns neither — the common
    // non-filling case. The column ranges are sparse: one column each.
    let geom = StageGeometry {
        filling_target_hydro_indices: vec![2],
        filling_target_col: 500..501,
        filled_min_storage_floor_hydro_indices: vec![0],
        filled_min_storage_floor_col: 600..601,
        ..StageGeometry::default()
    };

    let lookup = HydroReverseLookup::build(&geom, n_hydros);

    // σ_fill: only hydro 2, at slot 0 ⇒ column 500. Others absent.
    assert_eq!(lookup.filling_target[2], Some(0));
    assert_eq!(lookup.filling_target[0], None);
    assert_eq!(lookup.filling_target[1], None);
    let target_local = lookup.filling_target[2].expect("hydro 2 owns a σ_fill column");
    assert_eq!(geom.filling_target_col.start + target_local, 500);

    // σ^{v-}: only hydro 0, at slot 0 ⇒ column 600. Others absent. Independent of
    // the σ_fill family — hydro 0 has a floor column but no target column.
    assert_eq!(lookup.filled_min_storage_floor[0], Some(0));
    assert_eq!(lookup.filled_min_storage_floor[1], None);
    assert_eq!(lookup.filled_min_storage_floor[2], None);
    let floor_local = lookup.filled_min_storage_floor[0].expect("hydro 0 owns a σ^{v-} column");
    assert_eq!(geom.filled_min_storage_floor_col.start + floor_local, 600);
}

/// `read_filling_slack_primal` returns the solved primal at `start + local_idx`
/// for a present slot and `0.0` for an absent slot (the sparse-family default).
#[test]
fn read_filling_slack_primal_present_and_absent() {
    let primal = vec![0.0, 0.0, 7.5, 11.0];
    let range = 2..4;
    // Slot 0 ⇒ column 2 ⇒ 7.5; slot 1 ⇒ column 3 ⇒ 11.0.
    assert_eq!(
        super::read_filling_slack_primal(&primal, &range, Some(0)),
        7.5
    );
    assert_eq!(
        super::read_filling_slack_primal(&primal, &range, Some(1)),
        11.0
    );
    // Absent ⇒ 0.0 regardless of what the primal vector holds.
    assert_eq!(super::read_filling_slack_primal(&primal, &range, None), 0.0);
}

/// End-to-end (no-turbine / stage-aggregate branch): a filling hydro whose
/// `σ_fill` slack BINDS surfaces the non-zero primal in
/// `filling_target_violation_hm3`, while a non-filling hydro stays `0.0`. The
/// `geom(2, 1)` fixture has an empty `turbine` range, so `extract_hydros` takes
/// the no-turbine branch — exercising that read site directly.
#[test]
fn extract_reads_binding_filling_target_slack_no_turbine_branch() {
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let ec = zero_energy_conversion(2, 1);

    // make_primal_2_1 lays out columns [0..9) with theta at 8. Append the single
    // σ_fill slack column at index 9 carrying the binding violation (hm³).
    let sigma_fill = 4.25_f64;
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 0.0);
    primal.push(sigma_fill); // column 9 = σ_fill for hydro 0
    let dual = vec![0.0; 8];

    // Hydro 0 is the lone filling hydro at its terminal Filling stage; hydro 1 is
    // non-filling. The σ_fill column block is the single column [9, 10).
    let geom = StageGeometry {
        filling_target_hydro_indices: vec![0],
        filling_target_col: 9..10,
        ..crate::test_support::geom(2, 1)
    };

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geom,
            state: &state,
            n_blks: geom.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // No-turbine branch ⇒ one row per hydro, in hydro_ids order.
    assert_eq!(result.hydros.len(), 2);
    // Hydro 0 (filling, slack binds) reports the σ_fill primal — not 0.0.
    assert_eq!(result.hydros[0].filling_target_violation_hm3, sigma_fill);
    // Hydro 1 (non-filling, absent from the family) reports 0.0 (parity-neutral).
    assert_eq!(result.hydros[1].filling_target_violation_hm3, 0.0);
    // Neither hydro owns a σ^{v-} column here, so both floor fields stay 0.0.
    assert_eq!(result.hydros[0].storage_violation_below_hm3, 0.0);
    assert_eq!(result.hydros[1].storage_violation_below_hm3, 0.0);
}

/// End-to-end (per-block / turbine branch): a filling hydro whose `σ^{v-}`
/// operating-floor slack BINDS surfaces the non-zero primal in
/// `storage_violation_below_hm3` on every per-block row (the slack is
/// stage-level), while a hydro absent from the family stays `0.0`. A non-empty
/// `turbine` range routes `extract_hydros` through the per-block branch.
#[test]
fn extract_reads_binding_filled_min_storage_floor_slack_per_block_branch() {
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let ec = zero_energy_conversion(2, 1);

    // n_blks = 1, two hydros. Columns [0..9) per make_primal_2_1 (theta at 8).
    // Turbine columns at [9, 11) (one per hydro, n_blks=1), spillage at [11, 13),
    // and the single σ^{v-} slack column for hydro 1 at index 13.
    let sigma_floor = 2.75_f64;
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 0.0);
    primal.extend_from_slice(&[
        1.0,
        2.0, // turbine[0], turbine[1]
        0.0,
        0.0,         // spillage[0], spillage[1]
        sigma_floor, // column 13 = σ^{v-} for hydro 1
    ]);
    let dual = vec![0.0; 8];
    let objective_coeffs = vec![0.0; primal.len()];

    // Hydro 1 owns the lone σ^{v-} column at this Operating stage; hydro 0 is
    // absent from the floor family. turbine/spillage are dense (one col/hydro).
    let geom = StageGeometry {
        turbine: 9..11,
        spillage: 11..13,
        n_blks: 1,
        filled_min_storage_floor_hydro_indices: vec![1],
        filled_min_storage_floor_col: 13..14,
        ..crate::test_support::geom(2, 1)
    };

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geom,
            state: &state,
            n_blks: 1,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Per-block branch ⇒ one row per (hydro, block); n_blks = 1 ⇒ one row each.
    assert_eq!(result.hydros.len(), 2);
    let hydro0 = result
        .hydros
        .iter()
        .find(|h| h.hydro_id == 10)
        .expect("hydro 10 present");
    let hydro1 = result
        .hydros
        .iter()
        .find(|h| h.hydro_id == 20)
        .expect("hydro 20 present");
    // Hydro 1 (filling-floor, slack binds) reports the σ^{v-} primal.
    assert_eq!(hydro1.storage_violation_below_hm3, sigma_floor);
    // Hydro 0 (absent from the floor family) reports 0.0.
    assert_eq!(hydro0.storage_violation_below_hm3, 0.0);
    // No σ_fill column here, so both target fields stay 0.0.
    assert_eq!(hydro0.filling_target_violation_hm3, 0.0);
    assert_eq!(hydro1.filling_target_violation_hm3, 0.0);
}

// -------------------------------------------------------------------------
// assign_scenarios
// -------------------------------------------------------------------------

#[test]
fn assign_scenarios_uneven_rank0() {
    // Acceptance criterion: n=10, rank=0, world=3 → 0..4
    // 10 % 3 = 1 fat rank → rank 0 gets ceil(10/3) = 4 scenarios
    assert_eq!(assign_scenarios(10, 0, 3), 0..4);
}

#[test]
fn assign_scenarios_uneven_rank2() {
    // Acceptance criterion: n=10, rank=2, world=3 → 7..10
    assert_eq!(assign_scenarios(10, 2, 3), 7..10);
}

#[test]
fn assign_scenarios_single_rank() {
    // Acceptance criterion: n=7, rank=0, world=1 → 0..7
    assert_eq!(assign_scenarios(7, 0, 1), 0..7);
}

#[test]
fn assign_scenarios_uneven_rank1() {
    // Derived from acceptance criteria: rank 1 is a lean rank with 3 scenarios
    // starting at offset 4 (end of rank 0's fat slice).
    assert_eq!(assign_scenarios(10, 1, 3), 4..7);
}

#[test]
fn assign_scenarios_exact_division() {
    // n=9, world=3: 9 % 3 = 0 fat ranks → all lean (3 each)
    assert_eq!(assign_scenarios(9, 0, 3), 0..3);
    assert_eq!(assign_scenarios(9, 1, 3), 3..6);
    assert_eq!(assign_scenarios(9, 2, 3), 6..9);
}

#[test]
fn assign_scenarios_zero_scenarios() {
    // Every rank gets an empty range.
    assert_eq!(assign_scenarios(0, 0, 1), 0..0);
    assert_eq!(assign_scenarios(0, 0, 4), 0..0);
    assert_eq!(assign_scenarios(0, 3, 4), 0..0);
}

#[test]
fn assign_scenarios_more_ranks_than_scenarios() {
    // n=2, world=5: ranks 0-1 get 1 scenario each; ranks 2-4 get empty.
    assert_eq!(assign_scenarios(2, 0, 5), 0..1);
    assert_eq!(assign_scenarios(2, 1, 5), 1..2);
    assert_eq!(assign_scenarios(2, 2, 5), 2..2);
    assert_eq!(assign_scenarios(2, 3, 5), 2..2);
    assert_eq!(assign_scenarios(2, 4, 5), 2..2);
}

#[test]
fn assign_scenarios_sum_equals_n_scenarios() {
    // Property test: for various (n, world_size) pairs, total assigned = n.
    for (n, world_size) in [(0_u32, 1_usize), (1, 1), (10, 3), (9, 3), (2, 5), (100, 7)] {
        let total: u32 = (0..world_size)
            .map(|rank| {
                let r = assign_scenarios(n, rank, world_size);
                r.end - r.start
            })
            .sum();
        assert_eq!(
            total, n,
            "total assigned {total} != n_scenarios {n} for world_size={world_size}"
        );
    }
}

// -------------------------------------------------------------------------
// extract_stage_result
// -------------------------------------------------------------------------

/// Build a zero-valued [`crate::energy_conversion::EnergyConversionSet`] for tests
/// that do not assert on energy fields.
fn zero_energy_conversion(
    n_hydros: usize,
    n_stages: usize,
) -> crate::energy_conversion::EnergyConversionSet {
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; n_hydros],
        vec![vec![0.0_f64; n_stages]; n_hydros],
        n_hydros,
        n_stages,
    )
}

fn make_entity_counts_2_hydros() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a primal vector for a stage geometry with `hydro_count=2`, `max_par_order=1`.
///
/// Layout: storage\[0..2\], `inflow_lags`\[2..4\], `storage_in`\[4..6\], theta=6
fn make_primal_2_1(
    storage: [f64; 2],
    lags: [f64; 2],
    storage_in: [f64; 2],
    theta: f64,
) -> Vec<f64> {
    // Layout: storage(2), lags(2), z_inflow(2), storage_in(2), theta(1)
    vec![
        storage[0],
        storage[1],
        lags[0],
        lags[1],
        0.0, // z_inflow[0]
        0.0, // z_inflow[1]
        storage_in[0],
        storage_in[1],
        theta,
    ]
}

#[test]
fn extract_costs_has_one_entry_matching_stage_id() {
    // Acceptance criterion: costs contains exactly one entry whose stage_id
    // matches the input stage and whose future_cost == primal[state.theta].
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        3,
    );

    assert_eq!(result.costs.len(), 1);
    assert_eq!(result.costs[0].stage_id, 3);
    // future_cost = primal[theta] * COST_SCALE_FACTOR = 999.5 * 1_000_000 = 999_500_000.0
    assert_eq!(result.costs[0].future_cost, 999_500_000.0);
}

#[test]
fn extract_cost_splits_objective_correctly() {
    // objective = immediate_cost + future_cost
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let theta_val = 300.0;
    let objective = 800.0;
    let primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], theta_val);
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // All cost fields are in original units: multiply by COST_SCALE_FACTOR.
    assert_eq!(cost.future_cost, theta_val * 1_000_000.0);
    assert_eq!(cost.immediate_cost, (objective - theta_val) * 1_000_000.0);
    assert_eq!(cost.total_cost, objective * 1_000_000.0);
}

#[test]
fn extract_hydro_storage_values_from_primal() {
    // Hydro h=0: storage[0]=100, storage_in[4]=90
    // Hydro h=1: storage[1]=200, storage_in[5]=180
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 2);
    assert_eq!(result.hydros[0].hydro_id, 10);
    assert_eq!(result.hydros[0].storage_initial_hm3, 90.0);
    assert_eq!(result.hydros[0].storage_final_hm3, 100.0);

    assert_eq!(result.hydros[1].hydro_id, 20);
    assert_eq!(result.hydros[1].storage_initial_hm3, 180.0);
    assert_eq!(result.hydros[1].storage_final_hm3, 200.0);
}

#[test]
fn extract_inflow_lag_values_from_primal() {
    // inflow_lags[2]=50.0 for hydro 0 lag 0, [3]=60.0 for hydro 1 lag 0
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.inflow_lags.len(), 2); // 2 hydros × 1 lag each
    // Hydro 10, lag 0 → primal[2] = 50.0
    assert_eq!(result.inflow_lags[0].hydro_id, 10);
    assert_eq!(result.inflow_lags[0].lag_index, 0);
    assert_eq!(result.inflow_lags[0].inflow_m3s, 50.0);
    // Hydro 20, lag 0 → primal[3] = 60.0
    assert_eq!(result.inflow_lags[1].hydro_id, 20);
    assert_eq!(result.inflow_lags[1].lag_index, 0);
    assert_eq!(result.inflow_lags[1].inflow_m3s, 60.0);
}

#[test]
fn extract_no_lags_when_max_par_order_zero() {
    // Stage geometry (N=2, L=0): no inflow_lag columns → empty inflow_lags vec.
    let indexer = crate::test_support::geom(2, 0);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 0);
    // Layout: storage[0..2], z_inflow[2..4], storage_in[4..6], theta=6
    let primal = vec![100.0, 200.0, 0.0, 0.0, 90.0, 180.0, 500.0];
    let dual = vec![];
    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 600.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        2,
    );

    assert!(result.inflow_lags.is_empty());
    assert_eq!(result.hydros[0].incremental_inflow_m3s, 0.0);
}

#[test]
fn extract_stage_id_propagates_to_all_results() {
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 10.0);
    let dual = vec![0.0; 4];
    let stage_id = 7_u32;
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 110.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        stage_id,
    );

    assert_eq!(result.stage_id, stage_id);
    assert_eq!(result.costs[0].stage_id, stage_id);
    assert!(result.hydros.iter().all(|h| h.stage_id == stage_id));
    assert!(result.thermals.iter().all(|t| t.stage_id == stage_id));
    assert!(result.exchanges.iter().all(|e| e.stage_id == stage_id));
    assert!(result.buses.iter().all(|b| b.stage_id == stage_id));
    assert!(result.inflow_lags.iter().all(|l| l.stage_id == stage_id));
}

#[test]
fn extract_equipment_zero_when_indexer_has_no_equipment_ranges() {
    // When the geometry has no equipment, equipment ranges are empty and
    // all equipment result fields default to zero — backward-compatible behaviour.
    let indexer = crate::test_support::geom(2, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(2, 1);
    let primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], 0.0);
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Thermal — one entry per thermal entity, all zero.
    assert_eq!(result.thermals.len(), 1);
    assert_eq!(result.thermals[0].generation_mw, 0.0);
    assert_eq!(result.thermals[0].generation_cost, 0.0);
    assert_eq!(result.thermals[0].block_id, None);

    // Exchange — one entry per line entity, all zero.
    assert_eq!(result.exchanges.len(), 1);
    assert_eq!(result.exchanges[0].direct_flow_mw, 0.0);
    assert_eq!(result.exchanges[0].block_id, None);

    // Bus — one entry per bus entity, all zero.
    assert_eq!(result.buses.len(), 1);
    assert_eq!(result.buses[0].deficit_mw, 0.0);
    assert_eq!(result.buses[0].spot_price, 0.0);
    assert_eq!(result.buses[0].block_id, None);
}

/// Verify that equipment columns are read from the primal vector when the
/// indexer was built via `with_equipment`.
///
/// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus, K=1 block:
///
/// ```text
/// theta = N*(3+L) = 2*(3+1) = 8
/// turbine:   [9, 11)   h0→9, h1→10
/// spillage:  [11, 13)  h0→11, h1→12
/// diversion: [13, 15)  h0→13, h1→14
/// thermal:   [15, 16)  t0→15
/// line_fwd:  [16, 17)  l0→16
/// line_rev:  [17, 18)  l0→17
/// deficit:   [18, 19)  b0→18
/// excess:    [19, 20)  b0→19
/// ```
#[test]
fn extract_equipment_reads_primal_when_with_equipment() {
    // N=2, L=1, T=1, Ln=1, B=1, K=1
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_thermal_indices: vec![],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(2, 1, 0, 0, vec![]);
    // theta = 8, equipment starts at 9
    assert_eq!(state.theta, 8);
    assert_eq!(indexer.turbine, 9..11);
    assert_eq!(indexer.spillage, 11..13);
    assert_eq!(indexer.diversion, 13..15);
    assert_eq!(indexer.thermal, 15..16);
    assert_eq!(indexer.line_fwd, 16..17);
    assert_eq!(indexer.line_rev, 17..18);
    assert_eq!(indexer.deficit, 18..19);
    assert_eq!(indexer.excess, 19..20);

    // Build a primal vector sized to include withdrawal_slack columns.
    // storage[0..2]=100,200  inflow_lags[2..4]=50,60  z_inflow[4..6]=0,0
    // storage_in[6..8]=90,180  theta[8]=500
    // turbine[9..11]=30.0,40.0   spillage[11..13]=5.0,0.0
    // diversion[13..15]=0.0,0.0
    // thermal[15]=80.0   line_fwd[16]=15.0   line_rev[17]=0.0
    // deficit[18]=10.0   excess[19]=2.0   withdrawal_slack[20..22]=0.0,0.0
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 100.0; // storage h0
    primal[1] = 200.0; // storage h1
    primal[2] = 50.0; // lag h0
    primal[3] = 60.0; // lag h1
    // primal[4..6] = z_inflow (zeros)
    primal[6] = 90.0; // storage_in h0
    primal[7] = 180.0; // storage_in h1
    primal[8] = 500.0; // theta
    primal[9] = 30.0; // turbine h0 b0
    primal[10] = 40.0; // turbine h1 b0
    primal[11] = 5.0; // spillage h0 b0
    primal[12] = 0.0; // spillage h1 b0
    // primal[13..15] = diversion (zeros)
    primal[15] = 80.0; // thermal t0 b0
    primal[16] = 15.0; // line_fwd l0 b0
    primal[17] = 0.0; // line_rev l0 b0
    primal[18] = 10.0; // deficit b0 b0
    primal[19] = 2.0; // excess b0 b0

    // Objective coefficients: thermal cost=50/MWh, spillage cost=0.1, deficit=1000, excess=50
    let mut obj = vec![0.0_f64; n_cols];
    obj[8] = 1.0; // theta (objective = 1)
    obj[11] = 0.1; // spillage h0 penalty
    obj[15] = 50.0; // thermal cost per MW
    obj[16] = 5.0; // line_fwd cost per MW
    obj[18] = 1000.0; // deficit cost per MW
    obj[19] = 50.0; // excess cost per MW

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    // Dual vector: water_value reads from dual[water_balance.start + h],
    // load balance from dual[load_balance.start + b*K + blk].
    // N=2, L=1: z_inflow rows [0,2), water_balance=[2,4), load_balance=[4,5).
    let mut dual = vec![0.0_f64; 5];
    dual[2] = -120.0; // water value h0 ($/hm³) — at water_balance.start+0
    dual[3] = -95.0; // water value h1 ($/hm³) — at water_balance.start+1
    dual[4] = 108_000.0; // raw load balance dual ($/MW); 150 $/MWh × 720 h

    // Build row_lower for the load balance row. load_balance.start=4, K=1, B=1.
    let mut row_lower = vec![0.0_f64; 5]; // must be >= load_balance.end = 5
    row_lower[4] = 75.0; // load = 75 MW for bus 100
    let block_hours = [720.0_f64]; // one block, 30-day month
    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 600.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &block_hours,
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Hydro: one entry per (hydro, block), block_id = Some(0)
    assert_eq!(result.hydros.len(), 2); // 2 hydros × 1 block
    assert_eq!(result.hydros[0].block_id, Some(0));
    assert_eq!(result.hydros[0].turbined_m3s, 30.0);
    assert_eq!(result.hydros[0].spillage_m3s, 5.0);
    // spillage_cost = 5.0 * 0.1 * COST_SCALE_FACTOR = 500_000.0
    assert!((result.hydros[0].spillage_cost - 500_000.0).abs() < 1e-12); // 5.0 * 0.1 * 1_000_000

    // Hydro generation = turbined * productivity (1.0)
    assert_eq!(result.hydros[0].generation_mw, 30.0); // 30 * 1.0
    assert_eq!(result.hydros[1].generation_mw, 40.0); // 40 * 1.0

    // Hydro h=1 (no spillage)
    assert_eq!(result.hydros[1].block_id, Some(0));
    assert_eq!(result.hydros[1].turbined_m3s, 40.0);
    assert_eq!(result.hydros[1].spillage_m3s, 0.0);

    // Thermal: one entry per (thermal, block), block_id = Some(0)
    assert_eq!(result.thermals.len(), 1);
    assert_eq!(result.thermals[0].generation_mw, 80.0);
    // generation_cost = 80 * 50 * COST_SCALE_FACTOR = 4_000_000_000.0
    assert!((result.thermals[0].generation_cost - 4_000_000_000.0).abs() < 1e-3); // 80 * 50 * 1_000_000
    assert_eq!(result.thermals[0].block_id, Some(0));

    // Exchange: one entry per (line, block)
    assert_eq!(result.exchanges.len(), 1);
    assert_eq!(result.exchanges[0].direct_flow_mw, 15.0);
    assert_eq!(result.exchanges[0].reverse_flow_mw, 0.0);
    // exchange_cost = 15 * 5 * COST_SCALE_FACTOR = 75_000_000.0
    assert!((result.exchanges[0].exchange_cost - 75_000_000.0).abs() < 1e-3); // 15 * 5 * 1_000_000
    assert_eq!(result.exchanges[0].block_id, Some(0));

    // Bus: one entry per (bus, block)
    assert_eq!(result.buses.len(), 1);
    assert_eq!(result.buses[0].load_mw, 75.0); // from row_lower
    assert_eq!(result.buses[0].deficit_mw, 10.0);
    assert_eq!(result.buses[0].excess_mw, 2.0);
    assert_eq!(result.buses[0].block_id, Some(0));
    // spot_price = dual * COST_SCALE_FACTOR / hrs = 108_000 * 1_000_000 / 720 = 150_000_000.0 $/MWh
    assert!((result.buses[0].spot_price - 150_000_000.0).abs() < 1e-3); // 108_000 * 1_000_000 / 720

    // water_value = dual[water_balance.start+h] * COST_SCALE_FACTOR
    assert!((result.hydros[0].water_value_per_hm3 - (-120_000_000.0)).abs() < 1e-3);
    assert!((result.hydros[1].water_value_per_hm3 - (-95_000_000.0)).abs() < 1e-3);

    // Cost breakdown — all values multiplied by COST_SCALE_FACTOR = 1_000_000.
    let cost = &result.costs[0];
    assert!((cost.thermal_cost - 4_000_000_000.0).abs() < 1e-3); // 80 * 50 * 1_000_000
    assert!((cost.spillage_cost - 500_000.0).abs() < 1e-6); // 5 * 0.1 * 1_000_000
    assert!((cost.deficit_cost - 10_000_000_000.0).abs() < 1e-3); // 10 * 1000 * 1_000_000
    assert!((cost.excess_cost - 100_000_000.0).abs() < 1e-3); // 2 * 50 * 1_000_000
    assert!((cost.exchange_cost - 75_000_000.0).abs() < 1e-3); // 15 * 5 * 1_000_000
}

/// Verify that `is_anticipated` is set to `true` for thermals whose global
/// index appears in `anticipated_thermal_indices`, and `false` for all others.
///
/// Setup: 2 thermals (ids 10 and 20), 1 block. Thermal at global index 1
/// (id=20) is anticipated. The per-block branch is exercised by using
/// `n_blks=1` with a non-empty thermal range.
#[test]
fn extract_thermals_marks_anticipated_thermals_when_indices_nonempty() {
    // N=0 hydros, T=2 thermals, B=0 buses, K=1 block, n_anticipated=1 (index 1)
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![1],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);
    // With N=0, L=0, A=1, K_max=1 (anticipated_state_out relocated to the
    // state region after the ring buffer):
    //   anticipated_state     = [0, 1)  (A*K_max = 1 slot)
    //   anticipated_state_out = [1, 2)  (A = 1 column)
    //   theta = N*(3+L) + A*K_max + A = 0 + 1 + 1 = 2
    //   thermal = [theta+1, theta+1+T*K) = [3, 5)  — t0→3, t1→4
    assert_eq!(state.theta, 2);
    assert_eq!(indexer.thermal, 3..5);

    // n_cols must cover at least anticipated_decision.end (last column used)
    let n_cols = indexer.anticipated_decision.end.max(5);
    let mut primal = vec![0.0_f64; n_cols];
    primal[2] = 0.0; // theta
    primal[3] = 50.0; // thermal t0 b0 (gen_mw)
    primal[4] = 30.0; // thermal t1 b0 (gen_mw)

    let mut obj = vec![0.0_f64; n_cols];
    obj[3] = 10.0; // thermal t0 cost coefficient
    obj[4] = 20.0; // thermal t1 cost coefficient

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            n_stages: 2,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // Per-block path: 2 thermals × 1 block = 2 entries
    assert_eq!(result.thermals.len(), 2);

    // Thermal 10 at global index 0: NOT anticipated
    assert_eq!(result.thermals[0].thermal_id, 10);
    assert!(
        !result.thermals[0].is_anticipated,
        "thermal 10 should not be anticipated"
    );
    assert_eq!(result.thermals[0].anticipated_committed_mw, None);
    assert_eq!(result.thermals[0].anticipated_decision_mw, None);

    // Thermal 20 at global index 1: IS anticipated
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert!(
        result.thermals[1].is_anticipated,
        "thermal 20 should be anticipated"
    );
    // Always-active fishing: committed_mw reads slot 0 of anticipated_state
    // (primal[anticipated_state.start + 0] = 0.0 in this zero-initialised fixture).
    assert_eq!(result.thermals[1].anticipated_committed_mw, Some(0.0));
    // With stage_index=0, n_stages=2, K_i=1: t+K_i=1 <= n_stages -> decision is active.
    // primal[anticipated_decision.start + 0] defaults to 0.0 in this fixture.
    assert_eq!(result.thermals[1].anticipated_decision_mw, Some(0.0));
}

/// Verify that `is_anticipated` is `false` for every thermal when
/// `anticipated_thermal_indices` is empty (no anticipated thermals configured).
#[test]
fn extract_thermals_marks_no_thermals_anticipated_when_indices_empty() {
    // N=0 hydros, T=2 thermals, B=0 buses, K=1 block, n_anticipated=0
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_thermal_indices: vec![],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 0, 0, vec![]);

    let n_cols = indexer.generation_below_slack.end.max(3);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            n_stages: 2,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Per-block path: 2 thermals × 1 block = 2 entries, both non-anticipated
    assert_eq!(result.thermals.len(), 2);
    assert!(
        !result.thermals[0].is_anticipated,
        "thermal 10 should not be anticipated"
    );
    assert!(
        !result.thermals[1].is_anticipated,
        "thermal 20 should not be anticipated"
    );
}

// -------------------------------------------------------------------------
// Tests for compute_anticipated_decision_mw
// -------------------------------------------------------------------------

/// Shared fixture builder for the anticipated-decision tests.
///
/// N=0 hydros, T=2 thermals (IDs 10 and 20), B=0 buses, K=1 block,
/// `n_anticipated=1` (global index 1, ID 20), `k_max=2`, `K_i=2`.
///
/// Layout:
///   `n_ant_state = 1*2 = 2`  →  theta = 2
///   thermal = [3, 5)         →  `anticipated_decision.start = 5`
/// The `GeometryDims` both [`make_anticipated_decision_indexer_k2`] and the
/// matching `study_dims` derive from, so the role-(b) geometry and the
/// non-state `StudyDimensions` stay aligned from one source.
fn anticipated_decision_counts_k2() -> crate::test_support::GeometryDims {
    crate::test_support::GeometryDims {
        n_thermals: 2,
        n_blks: 1,
        n_anticipated: 1,
        k_max: 2,
        anticipated_thermal_indices: vec![1],
        ..Default::default()
    }
}

fn make_anticipated_decision_indexer_k2() -> StageGeometry {
    crate::test_support::geometry(&anticipated_decision_counts_k2(), vec![], &[], vec![])
}

/// Returns a primal vector sized to cover `anticipated_decision.end`, with
/// the decision column for local index 0 set to `sentinel`.
fn make_primal_with_decision_sentinel(geometry: &StageGeometry, sentinel: f64) -> Vec<f64> {
    let n_cols = geometry.anticipated_decision.end.max(geometry.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[geometry.anticipated_decision.start] = sentinel;
    primal
}

/// Single anticipated thermal (local index 0, global index 1, ID 20),
/// `K_i=2`, `stage_index=0`, `n_stages=3`.  `t+K_i = 2 <= 3` — decision is active.
///
/// Expects `anticipated_decision_mw == Some(123.5)` for the record with ID 20.
#[test]
fn extract_thermals_reads_anticipated_decision_when_in_horizon() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = crate::test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 2 thermals × 1 block = 2 records.  Thermal 20 is at index 1.
    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw,
        Some(123.5),
        "expected Some(123.5) for thermal 20 at stage 0 with K_i=2, n_stages=3"
    );
}

/// Same fixture, `stage_index=1`.  `t+K_i = 1+2 = 3 == n_stages` (boundary
/// is INACTIVE under the strict predicate `<`).
///
/// Expects `anticipated_decision_mw == None` — the LP has [0,0] bounds at
/// this boundary column and the extraction predicate must match.
#[test]
fn extract_thermals_emits_none_at_horizon_boundary() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = crate::test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 1,
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw, None,
        "expected None for thermal 20 at boundary stage 1 (K_i=2, n_stages=3): \
             t+K_i = 3 >= n_stages = 3 under the strict predicate"
    );
}

/// Same fixture, `stage_index=2`.  `t+K_i = 2+2 = 4 > 3 = n_stages` — one past
/// the boundary.  Decision column is structurally absent.
///
/// Expects `anticipated_decision_mw == None`.
#[test]
fn extract_thermals_emits_none_one_past_horizon_boundary() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = crate::test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2,
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw, None,
        "expected None for thermal 20 at stage 2 with K_i=2, n_stages=3 (one past boundary)"
    );
}

/// Two thermals (IDs 10 and 20), only thermal 20 is anticipated (global
/// index 1, local index 0).  `K_i=1`, `stage_index=0`, `n_stages=2`.
///
/// Asserts that the record for thermal 10 (the regular thermal) has
/// `anticipated_decision_mw == None`.
#[test]
fn extract_thermals_emits_none_for_non_anticipated_thermals() {
    // N=0, T=2, n_blks=1, n_anticipated=1 (index 1), k_max=1, K_i=1
    // Layout: n_ant_state=1, theta=1, thermal=[2,4), anticipated_decision.start=4
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![1],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[indexer.anticipated_decision.start] = 123.5;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            n_stages: 2,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[0].thermal_id, 10);
    assert_eq!(
        result.thermals[0].anticipated_decision_mw, None,
        "thermal 10 is not anticipated; expected None"
    );
}

/// `n_blks=4`, one anticipated thermal (ID 30, global index 0, local index
/// 0).  `K_i=1`, `stage_index=0`, `n_stages=2`.  The decision column is
/// stage-level (not per-block), so the same value must appear in all 4
/// block records.
///
/// Layout: N=0, T=1, `n_blks=4`, `n_anticipated=1`, `k_max=1`:
///   `n_ant_state=1`, theta=1, thermal=[2,6), `anticipated_decision.start=6`
#[test]
fn extract_thermals_anticipated_decision_is_per_block_invariant() {
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 4,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![0],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[indexer.anticipated_decision.start] = 123.5;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![30],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            n_stages: 2,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 1 thermal × 4 blocks = 4 records, all for thermal 30.
    assert_eq!(
        result.thermals.len(),
        4,
        "expected 4 block records (1 thermal × 4 blocks)"
    );
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(rec.thermal_id, 30);
        assert_eq!(
            rec.anticipated_decision_mw,
            Some(123.5),
            "block {blk}: expected Some(123.5) for per-block invariance"
        );
    }
}

// Tests for compute_anticipated_committed_mw (consolidated helper that
// reads slot 0 of the anticipated_state ring buffer in both branches).
// -------------------------------------------------------------------------

/// Build a `StageGeometry` for the anticipated-committed tests.
///
/// N=0 hydros, T=1 thermal (ID 10, global index 0), `n_blks=3`, `n_anticipated=1`
/// (global index 0, local index 0), `k_max=2`, `K_i=2`.
///
/// Layout (`anticipated_state_out` relocated to the state region):
///   `n_ant_state = 1*2 = 2`, plus the A=1 `state_out` column → `theta = 3`
///   `anticipated_state`     = `[0, 2)`
///   `anticipated_state_out` = `[2, 3)`
///   `thermal` = `[4, 7)`          (1 thermal * 3 blocks)
///   `anticipated_decision.start = 7`
/// The `GeometryDims` both
/// [`make_anticipated_committed_indexer_k2_3blks`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn anticipated_committed_counts_k2_3blks() -> crate::test_support::GeometryDims {
    crate::test_support::GeometryDims {
        n_thermals: 1,
        n_blks: 3,
        n_anticipated: 1,
        k_max: 2,
        anticipated_thermal_indices: vec![0],
        ..Default::default()
    }
}

fn make_anticipated_committed_indexer_k2_3blks() -> StageGeometry {
    crate::test_support::geometry(
        &anticipated_committed_counts_k2_3blks(),
        vec![],
        &[],
        vec![],
    )
}

/// AC-2: Per-block branch, K=2, `stage_index=2` (delivery stage).
///
/// The committed value is the slot-0 entry of the `anticipated_state` ring
/// buffer (per-plant, per-stage scalar), NOT the per-block thermal
/// generation. To guard against the regression where the helper
/// returned `primal[thermal_col]` (the per-block generation) instead of
/// `primal[anticipated_state.start + local_idx]`, this fixture uses three
/// distinct per-block generation values (50/60/70 MW) and a distinct
/// slot-0 value (42.0 MW). The fix must read 42.0 for every block; the
/// bug would read 50/60/70.
///
/// Per the fishing constraint
/// `sum_blk h_blk * g_blk = h_total * s_{i,0}`, a real LP would couple
/// these — but this is a synthetic primal vector, so the values are
/// independent and the test isolates which column the helper reads.
#[test]
fn extract_thermals_per_block_committed_at_delivery_stage() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    // thermal = [4, 7): col 4 = block 0, col 5 = block 1, col 6 = block 2
    assert_eq!(indexer.thermal.start, 4);
    // anticipated_state = [0, 2): col 0 = slot 0, col 1 = slot 1
    assert_eq!(state.anticipated_state.start, 0);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 42.0; // ant_state slot 0 (the committed scalar)
    primal[1] = 99.0; // ant_state slot 1 (unrelated; must not be read)
    primal[3] = 0.0; // theta
    primal[4] = 50.0; // thermal 10, block 0
    primal[5] = 60.0; // thermal 10, block 1
    primal[6] = 70.0; // thermal 10, block 2
    let obj = vec![0.0_f64; n_cols];

    // Also verify the helper directly for AC-1 coverage.
    let study_dims = crate::test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let lookup = super::ThermalReverseLookup::build(&study_dims, 1);
    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        state: &state,
        n_blks: indexer.n_blks,
        entity_counts: &EntityCounts {
            hydro_ids: vec![],
            hydro_productivities: vec![],
            thermal_ids: vec![10],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        },
        inflow_m3s_per_hydro: &[],
        block_hours: &[1.0, 1.0, 1.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &zero_energy_conversion(0, 3),
        hydro_min_storage_hm3: &[],
        stage_index: 2,
        n_stages: 3,
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &obj,
        row_lower: &[],
    };

    // Direct helper call: thermal_local=0. The helper must read slot 0 of
    // anticipated_state (col 0), NOT a per-block thermal column.
    assert_eq!(
        super::compute_anticipated_committed_mw(&view, &spec, &lookup, 0),
        Some(42.0),
        "helper: expected slot-0 value 42.0, NOT a per-block thermal value"
    );

    // Full extraction path.
    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2,
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 1 thermal * 3 blocks = 3 records. Every block carries the same
    // (per-stage) committed scalar from slot 0, NOT its own generation.
    // On the buggy code path this assertion fails with the
    // per-block thermal values [50, 60, 70].
    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(
            rec.anticipated_committed_mw,
            Some(42.0),
            "block {blk}: must read slot-0 ant_state (42.0), not per-block gen"
        );
        // Sanity: generation_mw is still the per-block thermal column,
        // and the per-block values are distinct from 42.0 — so the
        // regression would surface as committed_mw == generation_mw.
        assert_ne!(
            rec.anticipated_committed_mw,
            Some(rec.generation_mw),
            "block {blk}: committed_mw must NOT alias generation_mw"
        );
    }
}

/// AC-3: Per-block branch, K=2, `stage_index=1` (pre-delivery under the
/// legacy maturity-gate, but the always-active fishing predicate reads
/// slot 0 of `anticipated_state` regardless of `K_i` vs `stage_index`).
/// With a zero-initialised `primal[anticipated_state.start + 0]`,
/// expects every block to read `Some(0.0)`.
#[test]
fn extract_thermals_per_block_committed_slot0_when_seed_zero() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let study_dims = crate::test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[3] = 50.0; // thermal 10, block 0
    primal[4] = 60.0; // thermal 10, block 1
    primal[5] = 70.0; // thermal 10, block 2
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 1,
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(
            rec.anticipated_committed_mw,
            Some(0.0),
            "block {blk}: always-active reads slot 0 = 0.0 regardless of stage"
        );
    }
}

/// AC-4: Per-block branch, K=2, `stage_index=2` (boundary: `k_i == stage_index`).
/// Expects every block to have `anticipated_committed_mw == Some(_)`.
#[test]
fn extract_thermals_per_block_committed_at_first_delivery_boundary() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let study_dims = crate::test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[3] = 50.0;
    primal[4] = 60.0;
    primal[5] = 70.0;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2, // k_i == stage_index: boundary acceptance
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert!(
            rec.anticipated_committed_mw.is_some(),
            "block {blk}: expected Some(_) at first delivery boundary (k_i == stage_index)"
        );
    }
}

/// AC-5: Two thermals, only thermal at global index 1 is anticipated.
/// Thermal at global index 0 must have `anticipated_committed_mw == None` for every block.
#[test]
fn extract_thermals_per_block_committed_none_for_non_anticipated() {
    // N=0, T=2, n_blks=3, n_anticipated=1 (global index 1), k_max=2, K_i=2
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 3,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 2,
        anticipated_thermal_indices: vec![1],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 2, vec![2]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    // thermal 0 (non-anticipated): blocks at thermal.start + 0*3 + [0,1,2]
    primal[indexer.thermal.start] = 10.0;
    primal[indexer.thermal.start + 1] = 20.0;
    primal[indexer.thermal.start + 2] = 30.0;
    // thermal 1 (anticipated): blocks at thermal.start + 1*3 + [0,1,2]
    primal[indexer.thermal.start + 3] = 40.0;
    primal[indexer.thermal.start + 4] = 50.0;
    primal[indexer.thermal.start + 5] = 60.0;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2, // delivery stage for thermal 20
            n_stages: 3,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 2 thermals * 3 blocks = 6 records; thermal 10 is first 3.
    assert_eq!(result.thermals.len(), 6);
    for blk in 0..3 {
        assert_eq!(result.thermals[blk].thermal_id, 10);
        assert_eq!(
            result.thermals[blk].anticipated_committed_mw, None,
            "block {blk}: non-anticipated thermal 10 must have None"
        );
    }
    // Thermal 20 (anticipated) at delivery stage must have Some(_).
    for blk in 0..3 {
        assert!(
            result.thermals[3 + blk].anticipated_committed_mw.is_some(),
            "block {blk}: anticipated thermal 20 must have Some(_) at delivery"
        );
    }
}

/// AC-6: No-block branch, K=1, `stage_index=1`, `n_stages=2` (delivery).
/// Expects `anticipated_committed_mw == Some(0.0)`.
#[test]
fn extract_thermals_no_block_committed_at_delivery_is_zero() {
    // N=0, T=1, n_blks=0 (no-block branch), n_anticipated=1, k_max=1, K_i=1
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 0,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![0],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);
    assert!(
        indexer.thermal.is_empty(),
        "n_blks=0 must yield empty thermal range"
    );

    // Also verify stage-level helper directly.
    let lookup = super::ThermalReverseLookup::build(&study_dims, 1);
    let spec_delivery = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        state: &state,
        n_blks: indexer.n_blks,
        entity_counts: &EntityCounts {
            hydro_ids: vec![],
            hydro_productivities: vec![],
            thermal_ids: vec![10],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        },
        inflow_m3s_per_hydro: &[],
        block_hours: &[],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &zero_energy_conversion(0, 2),
        hydro_min_storage_hm3: &[],
        stage_index: 1,
        n_stages: 2,
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };
    // In the no-block branch the fishing-constraint LHS sum vanishes; Category 6
    // pins slot 0 to incoming (0.0 here), so the helper returns Some(0.0) — same
    // observable as before the fix, but via slot-0 of anticipated_state.
    let n_cols_helper = indexer.anticipated_decision.end.max(1);
    let primal_helper = vec![0.0_f64; n_cols_helper];
    let dual_helper: Vec<f64> = vec![];
    let view_helper = SolutionView {
        primal: &primal_helper,
        dual: &dual_helper,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    assert_eq!(
        super::compute_anticipated_committed_mw(&view_helper, &spec_delivery, &lookup, 0),
        Some(0.0),
        "consolidated helper: expected Some(0.0) at delivery stage (slot-0 = incoming = 0.0)"
    );

    // Full extraction path.
    let n_cols = indexer.anticipated_decision.end.max(1);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 1, // delivery stage
            n_stages: 2,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 1);
    assert_eq!(
        result.thermals[0].anticipated_committed_mw,
        Some(0.0),
        "no-block delivery: expected Some(0.0)"
    );
}

/// AC-7: No-block branch, K=1, `stage_index=0`, `n_stages=2`. Pre-delivery
/// under the legacy maturity-gate, but the always-active fishing predicate
/// reads slot 0 of `anticipated_state` regardless. Expects `Some(0.0)`
/// (zero-initialised slot 0).
#[test]
fn extract_thermals_no_block_committed_reads_slot0_when_seed_zero() {
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 0,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![0],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(1);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0, // before delivery
            n_stages: 2,
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 1);
    assert_eq!(
        result.thermals[0].anticipated_committed_mw,
        Some(0.0),
        "no-block always-active: reads slot 0 = 0.0 regardless of stage"
    );
}

/// Regression guard: verify that using a pre-built
/// [`ThermalReverseLookup`] via `extract_stage_result_with_lookups`
/// produces bit-for-bit identical results to the standard
/// [`extract_stage_result`] path (which builds the lookup internally).
///
/// If the two paths ever diverge, this test catches it immediately.
#[test]
fn extract_stage_result_prebuilt_lookup_matches_standard_path() {
    use super::{HydroReverseLookup, ThermalReverseLookup, extract_stage_result_with_lookups};

    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 2,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        k_max: 1,
        anticipated_thermal_indices: vec![0],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let state = crate::test_support::state_layout_full(0, 0, 1, 1, vec![1]);

    let n_cols = indexer
        .anticipated_decision
        .end
        .max(indexer.thermal.end)
        .max(state.anticipated_state.end)
        .max(state.theta + 1);
    let mut primal = vec![0.0_f64; n_cols];
    // Slot 0 of anticipated_state = committed MW scalar.
    primal[state.anticipated_state.start] = 37.5;
    // Anticipated decision column.
    primal[indexer.anticipated_decision.start] = 80.0;
    // Thermal columns: block 0, block 1.
    if !indexer.thermal.is_empty() {
        primal[indexer.thermal.start] = 40.0;
        primal[indexer.thermal.start + 1] = 50.0;
    }
    let obj = vec![0.01_f64; n_cols];
    let ec = zero_energy_conversion(0, 3);
    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![42],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &obj,
        row_lower: &[],
    };
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        state: &state,
        n_blks: indexer.n_blks,
        entity_counts: &counts,
        inflow_m3s_per_hydro: &[],
        block_hours: &[1.0, 1.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[],
        stage_index: 2, // delivery stage (k_i=1, stage_index=2 > k_i)
        n_stages: 3,
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };

    // Standard path: builds the lookup internally on every call.
    let result_standard = extract_stage_result(&view, &spec, 2);

    // Pre-built path: lookup built once, reused across calls (hot-path pattern).
    let thermal_lookup = ThermalReverseLookup::build(&study_dims, counts.thermal_ids.len());
    let hydro_lookup = HydroReverseLookup::build(spec.geometry, counts.hydro_ids.len());
    let result_prebuilt =
        extract_stage_result_with_lookups(&view, &spec, 2, &hydro_lookup, &thermal_lookup);

    // Verify bit-for-bit equality on the anticipated fields.
    assert_eq!(
        result_standard.thermals.len(),
        result_prebuilt.thermals.len(),
        "thermal result count must match"
    );
    for (std_t, pre_t) in result_standard
        .thermals
        .iter()
        .zip(result_prebuilt.thermals.iter())
    {
        assert_eq!(
            std_t.is_anticipated, pre_t.is_anticipated,
            "is_anticipated must match"
        );
        assert_eq!(
            std_t.anticipated_committed_mw.map(f64::to_bits),
            pre_t.anticipated_committed_mw.map(f64::to_bits),
            "anticipated_committed_mw bits must match"
        );
        assert_eq!(
            std_t.anticipated_decision_mw.map(f64::to_bits),
            pre_t.anticipated_decision_mw.map(f64::to_bits),
            "anticipated_decision_mw bits must match"
        );
        assert_eq!(
            std_t.generation_mw.to_bits(),
            pre_t.generation_mw.to_bits(),
            "generation_mw bits must match"
        );
    }
}

#[test]
fn extract_optional_entity_types_are_empty_when_absent() {
    let indexer = crate::test_support::geom(1, 0);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);
    let primal = vec![50.0, 0.0, 40.0, 200.0]; // storage, z_inflow, storage_in, theta
    let dual = vec![];
    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 250.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert!(result.pumping_stations.is_empty());
    assert!(result.contracts.is_empty());
    assert!(result.non_controllables.is_empty());
    assert!(result.generic_violations.is_empty());
}

// -------------------------------------------------------------------------
// accumulate_category_costs
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn make_cost(
    thermal: f64,
    contract: f64,
    deficit: f64,
    excess: f64,
    storage_violation: f64,
    filling: f64,
    hydro_violation: f64,
    inflow_penalty: f64,
    generic_violation: f64,
    spillage: f64,
    fpha: f64,
    curtailment: f64,
    exchange: f64,
    pumping: f64,
) -> SimulationCostResult {
    SimulationCostResult {
        stage_id: 0,
        block_id: None,
        total_cost: 0.0,
        immediate_cost: 0.0,
        future_cost: 0.0,
        discount_factor: 1.0,
        thermal_cost: thermal,
        // Held at zero so the `resource_cost` expectations (thermal + contract) stay exact.
        anticipated_thermal_cost: 0.0,
        contract_cost: contract,
        deficit_cost: deficit,
        excess_cost: excess,
        storage_violation_cost: storage_violation,
        filling_target_cost: filling,
        hydro_violation_cost: hydro_violation,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        turbined_violation_cost: 0.0,
        generation_violation_cost: 0.0,
        evaporation_violation_cost: 0.0,
        withdrawal_violation_cost: 0.0,
        inflow_penalty_cost: inflow_penalty,
        generic_violation_cost: generic_violation,
        spillage_cost: spillage,
        turbined_cost: fpha,
        curtailment_cost: curtailment,
        exchange_cost: exchange,
        pumping_cost: pumping,
    }
}

fn zero_accum() -> ScenarioCategoryCosts {
    ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    }
}

#[test]
fn accumulate_single_stage_all_categories() {
    let cost = make_cost(
        400.0, 100.0, // resource: 500
        50.0, 10.0, // recourse: 60
        20.0, 30.0, 5.0, 3.0, 2.0, // violation: 60
        1.0, 4.0, 7.0, 8.0,  // regularization: 20
        60.0, // imputed: 60
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.resource_cost, 500.0);
    assert_eq!(accum.recourse_cost, 60.0);
    assert_eq!(accum.violation_cost, 60.0);
    assert_eq!(accum.regularization_cost, 20.0);
    assert_eq!(accum.imputed_cost, 60.0);
}

#[test]
fn accumulate_two_consecutive_stages_sums_correctly() {
    let cost1 = make_cost(
        100.0, 0.0, // resource
        10.0, 0.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        0.0, 0.0, 0.0, 0.0, // regularization
        5.0, // imputed
    );
    let cost2 = make_cost(
        200.0, 50.0, // resource
        20.0, 5.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        0.0, 0.0, 0.0, 0.0,  // regularization
        10.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost1, &mut accum);
    accumulate_category_costs(&cost2, &mut accum);

    assert_eq!(accum.resource_cost, 100.0 + 200.0 + 50.0);
    assert_eq!(accum.recourse_cost, 10.0 + 20.0 + 5.0);
    assert_eq!(accum.violation_cost, 0.0);
    assert_eq!(accum.regularization_cost, 0.0);
    assert_eq!(accum.imputed_cost, 5.0 + 10.0);
}

#[test]
fn accumulate_all_zeros_leaves_accum_unchanged() {
    let cost = make_cost(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let mut accum = ScenarioCategoryCosts {
        resource_cost: 1.0,
        recourse_cost: 2.0,
        violation_cost: 3.0,
        regularization_cost: 4.0,
        imputed_cost: 5.0,
    };
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.resource_cost, 1.0);
    assert_eq!(accum.recourse_cost, 2.0);
    assert_eq!(accum.violation_cost, 3.0);
    assert_eq!(accum.regularization_cost, 4.0);
    assert_eq!(accum.imputed_cost, 5.0);
}

#[test]
fn accumulate_violation_all_five_components() {
    let cost = make_cost(
        0.0, 0.0, // resource
        0.0, 0.0, // recourse
        1.0, 2.0, 3.0, 4.0, 5.0, // violation: 15
        0.0, 0.0, 0.0, 0.0, // regularization
        0.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.violation_cost, 15.0);
}

#[test]
fn accumulate_regularization_all_four_components() {
    let cost = make_cost(
        0.0, 0.0, // resource
        0.0, 0.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        2.0, 3.0, 4.0, 5.0, // regularization: 14
        0.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.regularization_cost, 14.0);
}

// ── test_slack_extraction_in_simulation ──────────────────────────────────

/// Verify that `inflow_nonnegativity_slack_m3s` is read from the primal
/// solution when `has_inflow_penalty == true`.
///
/// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus,
/// K=1 block, with penalty method active:
///
/// theta=6, `turbine`=[7,9), `spillage`=[9,11), `thermal`=[11,12),
/// `line_fwd`=[12,13), `line_rev`=[13,14), `deficit`=[14,15), `excess`=[15,16),
/// `inflow_slack`=[16,18)
#[test]
fn test_slack_extraction_with_penalty_active() {
    // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=true
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: true,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_thermal_indices: vec![],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(2, 1, 0, 0, vec![]);

    assert!(
        study_dims.has_inflow_penalty,
        "has_inflow_penalty must be true"
    );
    assert!(
        !indexer.inflow_slack.is_empty(),
        "inflow_slack must be non-empty"
    );

    // Primal vector: base columns + inflow slack + withdrawal slack columns
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];

    // Fill base values
    primal[0] = 100.0; // storage h0
    primal[1] = 200.0; // storage h1
    primal[2] = 50.0; // lag h0
    primal[3] = 60.0; // lag h1
    primal[4] = 90.0; // storage_in h0
    primal[5] = 180.0; // storage_in h1
    primal[6] = 500.0; // theta

    // Inflow slack values: hydro 0 has slack 7.5, hydro 1 has slack 0.0
    primal[indexer.inflow_slack.start] = 7.5; // slack h0
    primal[indexer.inflow_slack.start + 1] = 0.0; // slack h1

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 4];
    let row_lower = vec![0.0_f64; indexer.load_balance.end.max(1)];

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // turbine is non-empty → per-(hydro, block) results, so 2 hydros × 1 block = 2 entries
    assert_eq!(result.hydros.len(), 2);

    // Slack for hydro 0 must equal the primal slack column value
    assert!(
        (result.hydros[0].inflow_nonnegativity_slack_m3s - 7.5).abs() < 1e-12,
        "hydro 0 slack should be 7.5, got {}",
        result.hydros[0].inflow_nonnegativity_slack_m3s
    );

    // Slack for hydro 1 must be 0.0
    assert_eq!(
        result.hydros[1].inflow_nonnegativity_slack_m3s, 0.0,
        "hydro 1 slack should be 0.0"
    );
}

/// Verify that `inflow_nonnegativity_slack_m3s` is zero when the penalty
/// method is inactive (`has_inflow_penalty == false`).
#[test]
fn test_slack_extraction_without_penalty_is_zero() {
    // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=false
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_thermal_indices: vec![],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(2, 1, 0, 0, vec![]);
    assert!(
        !study_dims.has_inflow_penalty,
        "has_inflow_penalty must be false"
    );

    let n_cols = indexer.generation_below_slack.end; // includes withdrawal_slack columns
    let primal = vec![1.0_f64; n_cols]; // all ones
    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 4];
    let row_lower = vec![0.0_f64; indexer.load_balance.end.max(1)];

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    for (h, hr) in result.hydros.iter().enumerate() {
        assert_eq!(
            hr.inflow_nonnegativity_slack_m3s, 0.0,
            "hydro {h} slack must be 0.0 when penalty is inactive"
        );
    }
}

/// Verify that the fallback path (no equipment ranges) also reads slack
/// when `has_inflow_penalty == true`.
#[test]
fn test_slack_extraction_fallback_path_with_penalty() {
    // Zero blocks (turbine.is_empty()) with
    // has_inflow_penalty=true — exercises the empty-equipment-range extraction path.
    // N=2, L=1, T=0, Ln=0, B=0, K=0, has_inflow_penalty=true
    let eq_counts = crate::test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 0,
        n_blks: 0,
        has_inflow_penalty: true,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_thermal_indices: vec![],
    };
    let indexer = crate::test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = crate::test_support::study_dims_for(&eq_counts);
    let state = crate::test_support::state_layout_full(2, 1, 0, 0, vec![]);

    // turbine is empty (n_blks=0) → fallback path
    assert!(
        indexer.turbine.is_empty(),
        "turbine must be empty to trigger fallback"
    );
    assert!(
        study_dims.has_inflow_penalty,
        "has_inflow_penalty must be true"
    );

    // Layout: storage[0..2], lags[2..4], storage_in[4..6], theta=6,
    //         inflow_slack=[7..9), withdrawal_slack=[9..11)
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 150.0; // storage h0
    primal[1] = 250.0; // storage h1
    primal[2] = 55.0; // lag h0
    primal[3] = 65.0; // lag h1
    primal[4] = 140.0; // storage_in h0
    primal[5] = 240.0; // storage_in h1
    primal[6] = 0.0; // theta
    primal[indexer.inflow_slack.start] = 3.0; // slack h0
    primal[indexer.inflow_slack.start + 1] = 0.0; // slack h1

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 4];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Fallback: one entry per hydro (block_id = None)
    assert_eq!(result.hydros.len(), 2);
    assert!(
        (result.hydros[0].inflow_nonnegativity_slack_m3s - 3.0).abs() < 1e-12,
        "hydro 0 fallback slack should be 3.0, got {}",
        result.hydros[0].inflow_nonnegativity_slack_m3s
    );
    assert_eq!(result.hydros[1].inflow_nonnegativity_slack_m3s, 0.0);
}

// ── FPHA and Evaporation extraction tests ────────────────────────────────

/// Build a `StageGeometry` with 2 hydros (h0 = FPHA, h1 = constant-productivity),
/// 1 block, no thermals/lines/buses.
///
/// Column layout:
/// ```text
/// N=2, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[0], planes=[2]
/// theta = N*(3+L) = 2*(3+0) = 6
/// turbine:    [7, 9)    h0→7, h1→8
/// spillage:   [9, 11)   h0→9, h1→10
/// diversion: [11, 13)   h0→11, h1→12
/// generation:[13, 14)   fpha h0 b0 → 13
/// ```
/// The `GeometryDims` both [`make_indexer_2h_1fpha_1blk`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn counts_2h_1fpha_1blk() -> crate::test_support::GeometryDims {
    crate::test_support::GeometryDims {
        hydro_count: 2,
        n_blks: 1,
        ..Default::default()
    }
}

fn make_indexer_2h_1fpha_1blk() -> StageGeometry {
    // h0 is FPHA (system index 0), h1 is constant-productivity (system index 1)
    crate::test_support::geometry(&counts_2h_1fpha_1blk(), vec![0], &[2], vec![])
}

/// Acceptance criterion: FPHA hydro's `generation_mw` equals the LP generation
/// variable (not turbined * productivity = 0).
#[test]
fn fpha_generation_read_from_lp_column() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    // generation.start should be after turbine(7..9) + spillage(9..11) + diversion(11..13) = 13
    // generation[0] = generation.start + 0 * 1 + 0 = 13
    assert_eq!(indexer.generation.start, 13, "generation starts at 13");
    assert_eq!(indexer.fpha_hydro_indices, vec![0]);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 50.0; // storage h0
    primal[1] = 80.0; // storage h1
    // primal[2..4] = z_inflow (zeros)
    primal[4] = 45.0; // storage_in h0
    primal[5] = 75.0; // storage_in h1
    primal[6] = 0.0; // theta
    primal[7] = 20.0; // turbine h0 b0 (not used for FPHA gen)
    primal[8] = 30.0; // turbine h1 b0
    primal[9] = 0.0; // spillage h0 b0
    primal[10] = 0.0; // spillage h1 b0
    // primal[11..13] = diversion (zeros)
    primal[13] = 75.0; // FPHA generation h0 b0 — acceptance criterion value

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5], // FPHA has 0.0, constant has 1.5
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // 2 hydros × 1 block = 2 entries
    assert_eq!(result.hydros.len(), 2);

    // FPHA hydro (h0, block 0): generation from LP column 13 = 75.0
    assert!(
        (result.hydros[0].generation_mw - 75.0).abs() < 1e-12,
        "FPHA generation_mw should be 75.0, got {}",
        result.hydros[0].generation_mw
    );

    // Constant-productivity hydro (h1, block 0): generation = turbined * productivity
    // turbine h1 b0 = primal[8] = 30.0, productivity = 1.5 → 45.0
    assert!(
        (result.hydros[1].generation_mw - 45.0).abs() < 1e-12,
        "constant-productivity generation_mw should be 45.0, got {}",
        result.hydros[1].generation_mw
    );
}

/// Acceptance criterion: both FPHA and constant-productivity hydros report
/// the `equivalent_productivity_mw_per_m3s` produced by the supplied
/// `EnergyConversionSet`. With a zero-valued set the field is `0.0` for
/// every hydro regardless of generation model.
#[test]
fn fpha_productivity_placeholder_zero() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Both hydros report the values supplied by the zero-valued EnergyConversionSet.
    assert_eq!(
        result.hydros[0].equivalent_productivity_mw_per_m3s, 0.0,
        "FPHA hydro must carry placeholder 0.0 for equivalent_productivity_mw_per_m3s"
    );
    assert_eq!(
        result.hydros[1].equivalent_productivity_mw_per_m3s, 0.0,
        "constant-productivity hydro must carry placeholder 0.0 for equivalent_productivity_mw_per_m3s"
    );
}

/// Build a `StageGeometry` with 1 hydro that has evaporation, 1 block.
///
/// Column layout:
/// ```text
/// N=1, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[], evap=[0]
/// theta = 1*(3+0) = 3
/// turbine:    [4, 5)   h0→4
/// spillage:   [5, 6)   h0→5
/// diversion:  [6, 7)   h0→6
/// evap:       [7, 10)  evaporation_flow→7, f_plus→8, f_minus→9
/// ```
/// The `GeometryDims` both [`make_indexer_1h_evap_1blk`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn counts_1h_evap_1blk() -> crate::test_support::GeometryDims {
    crate::test_support::GeometryDims {
        hydro_count: 1,
        n_blks: 1,
        ..Default::default()
    }
}

fn make_indexer_1h_evap_1blk() -> StageGeometry {
    crate::test_support::geometry(&counts_1h_evap_1blk(), vec![], &[], vec![0])
}

/// Acceptance criterion: `evaporation_m3s` equals the LP evaporation-outflow variable value.
#[test]
fn evaporation_read_from_lp_column() {
    let indexer = make_indexer_1h_evap_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_1h_evap_1blk());
    let state = crate::test_support::state_layout(1, 0);
    assert_eq!(indexer.evap_hydro_indices, vec![0]);
    let ei = &indexer.evap_indices[0];
    assert_eq!(ei.evaporation_flow_col, 7);
    assert_eq!(ei.f_evap_plus_col, 8);
    assert_eq!(ei.f_evap_minus_col, 9);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 200.0; // storage h0
    // primal[1] = z_inflow h0 (zero)
    primal[2] = 190.0; // storage_in h0
    primal[3] = 0.0; // theta
    primal[4] = 10.0; // turbine h0 b0
    primal[5] = 0.0; // spillage h0 b0
    // primal[6] = diversion h0 b0 (zero)
    primal[7] = 3.5; // evaporation outflow — acceptance criterion value

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 1];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    assert_eq!(
        result.hydros[0].evaporation_m3s,
        Some(3.5),
        "evaporation_m3s should be Some(3.5)"
    );
    assert!(
        result.hydros[0].evaporation_violation_pos_m3s.abs() < 1e-12,
        "evaporation_violation_pos_m3s should be 0.0"
    );
}

/// Acceptance criterion: directional evaporation violations are extracted
/// separately from the LP's `f_evap_plus` (under-evaporation / neg) and
/// `f_evap_minus` (over-evaporation / pos) columns.
#[test]
fn evaporation_violation_is_sum_of_slacks() {
    let indexer = make_indexer_1h_evap_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_1h_evap_1blk());
    let state = crate::test_support::state_layout(1, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 200.0;
    // primal[1] = z_inflow h0 (zero)
    primal[2] = 190.0; // storage_in h0
    // primal[3] = theta = 0
    // primal[6] = diversion h0 b0 (zero)
    primal[7] = 2.0; // evaporation outflow
    primal[8] = 0.5; // f_evap_plus (under-evaporation -> neg)
    primal[9] = 0.0; // f_evap_minus (over-evaporation -> pos)

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; 1];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // f_evap_plus (primal[8] = 0.5) maps to under-evaporation (neg).
    assert!(
        (result.hydros[0].evaporation_violation_neg_m3s - 0.5).abs() < 1e-12,
        "evaporation_violation_neg_m3s should be 0.5, got {}",
        result.hydros[0].evaporation_violation_neg_m3s
    );
    // f_evap_minus (primal[9] = 0.0) maps to over-evaporation (pos).
    assert!(
        result.hydros[0].evaporation_violation_pos_m3s.abs() < 1e-12,
        "evaporation_violation_pos_m3s should be 0.0, got {}",
        result.hydros[0].evaporation_violation_pos_m3s
    );
}

/// Acceptance criterion: `turbined_cost` equals the sum of primal * obj\_coeff
/// over turbine columns (every hydro, every block), multiplied by
/// `COST_SCALE_FACTOR`.
///
/// Setup: 2 hydros, 1 block. h0 turbine column: primal=30.0,
/// `objective_coeff`=0.01 → `scaled_cost`=0.3 → unscaled=300.0.
#[test]
fn turbined_cost_in_compute_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    // turbine.start = 7 (h0 b0)
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta (at N*(3+L) = 2*3 = 6)

    // h0 turbine column 7: primal=30.0
    primal[7] = 30.0;

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    // h0 turbine column 7: objective_coeff=0.01
    obj[7] = 0.01;

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.3, // theta + fpha cost
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        (cost.turbined_cost - 300_000.0).abs() < 1e-9,
        "turbined_cost should be 300_000.0 (30.0 * 0.01 * COST_SCALE_FACTOR), got {}",
        cost.turbined_cost
    );
}

/// Verify that per-component cost breakdown sums to `immediate_cost` under
/// identity `col_scale`.
///
/// Setup: 2 hydros, 1 block. h0 turbine col 7: primal=30, `obj_coeff`=0.01.
/// Objective = `theta_scaled` + `turbine_scaled` = 500 + (30 * 0.01) = 500.3.
/// `immediate_cost` = (500.3 - 500) * `1_000_000` = `300_000`.
/// Per-component sum = `turbined_cost` = 30 * 0.01 * `1_000_000` = `300_000` (matches).
#[test]
fn cost_breakdown_sums_to_immediate_identity_scale() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    obj[7] = 0.01; // turbined cost (scaled)
    // objective in scaled space = theta_coeff * theta + turbine_coeff * turbine
    //                           = 1.0 * 500 + 0.01 * 30 = 500.3
    let objective_val = 500.3_f64;

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // immediate_cost = (obj - theta) * K = (500.3 - 500.0) * 1_000_000 = 300_000.0
    assert!(
        (cost.immediate_cost - 300_000.0).abs() < 1.0,
        "immediate_cost should be 300_000.0, got {}",
        cost.immediate_cost
    );

    // Per-component sum: only turbined_cost is non-zero.
    let component_sum = cost.thermal_cost
        + cost.deficit_cost
        + cost.excess_cost
        + cost.exchange_cost
        + cost.spillage_cost
        + cost.generic_violation_cost
        + cost.turbined_cost
        + cost.curtailment_cost;

    assert!(
        (component_sum - cost.immediate_cost).abs() < 1.0,
        "per-component cost sum ({component_sum}) must equal immediate_cost ({})",
        cost.immediate_cost
    );
    assert_eq!(
        cost.contract_cost, 0.0,
        "contract-free stage books zero contract_cost"
    );
}

/// `contract_cost` for an active import contract equals `price * power * hours`.
/// The scaled LP objective coeff is `price * hours / COST_SCALE_FACTOR`; the
/// extractor recovers the original-unit cost. Exercises the geometry-range
/// cost path in `compute_cost_result` (`contract_ids: vec![]`), NOT the
/// `extract_contracts` primal read — `d41_energy_contracts_simulation` covers
/// that end to end.
#[test]
fn contract_cost_active_import_equals_price_power_hours_via_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    let base = indexer.generation_below_slack.end;
    let contract_col = base;
    let geometry = StageGeometry {
        contract_import: contract_col..contract_col + 1,
        contract_export: contract_col + 1..contract_col + 1,
        ..indexer
    };

    let n_cols = contract_col + 1;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[contract_col] = 40.0; // import power (MW)

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0;
    // scaled objective coeff = price * hours / COST_SCALE_FACTOR = 200 * 730 / 1e6
    obj[contract_col] = 200.0 * 730.0 / 1_000_000.0;
    // objective = theta + contract = 500 + 40 * 0.146 = 505.84
    let objective_val = 500.0 + 40.0 * (200.0 * 730.0 / 1_000_000.0);

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];
    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            state: &state,
            n_blks: geometry.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        (cost.contract_cost - 200.0 * 40.0 * 730.0).abs() < 1.0,
        "contract_cost should be 200 * 40 * 730, got {}",
        cost.contract_cost
    );
}

/// Σ(macro categories) == `immediate_cost` when a contract is active — the
/// load-bearing invariant. An export contract (negative price) nets negative
/// `contract_cost` (revenue), so the sum still matches `immediate_cost`.
/// Exercises the geometry-range cost path in `compute_cost_result`
/// (`contract_ids: vec![]`), NOT the `extract_contracts` primal read —
/// `d41_energy_contracts_simulation` covers that end to end.
#[test]
fn cost_breakdown_sums_to_immediate_with_active_export_contract_via_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    let base = indexer.generation_below_slack.end;
    let export_col = base;
    let geometry = StageGeometry {
        contract_import: export_col..export_col,
        contract_export: export_col..export_col + 1,
        ..indexer
    };

    let n_cols = export_col + 1;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column
    primal[export_col] = 30.0; // export power (MW)

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0;
    obj[7] = 0.01; // turbined cost (scaled)
    // export price < 0 (revenue): scaled coeff = -150 * 730 / 1e6
    obj[export_col] = -150.0 * 730.0 / 1_000_000.0;
    let objective_val = 500.0 + 30.0 * 0.01 + 30.0 * (-150.0 * 730.0 / 1_000_000.0);

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];
    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            state: &state,
            n_blks: geometry.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        cost.contract_cost < 0.0,
        "export contract nets negative contract_cost (revenue), got {}",
        cost.contract_cost
    );
    assert!(
        (cost.contract_cost - (-150.0 * 30.0 * 730.0)).abs() < 1.0,
        "contract_cost should be -150 * 30 * 730, got {}",
        cost.contract_cost
    );

    let mut accum = ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    };
    accumulate_category_costs(cost, &mut accum);
    let macro_sum = accum.resource_cost
        + accum.recourse_cost
        + accum.violation_cost
        + accum.regularization_cost
        + accum.imputed_cost;
    assert!(
        (macro_sum - cost.immediate_cost).abs() < 1.0,
        "Σ(macro categories) ({macro_sum}) must equal immediate_cost ({}) with an active contract",
        cost.immediate_cost
    );
}

/// Verify that per-component costs are correctly unscaled when non-trivial
/// `col_scale` is applied.
///
/// With `col_scale` = 2.0 on the h0 turbine column:
/// - `obj_coeff` in template = `c_orig` * `col_scale` / K = 0.005 * 2.0 = 0.01
/// - After unscaling: cost = primal * `obj_coeff` / `col_scale` * K = 30 * 0.01 / 2.0 * `1_000_000` = `150_000`.
#[test]
fn cost_unscaled_by_col_scale() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    // c_orig / K = 0.005.  With col_scale = 2.0: obj_coeff = 0.005 * 2.0 = 0.01.
    obj[7] = 0.01;

    // Build col_scale: all 1.0 except column 7 = 2.0.
    let mut col_scale = vec![1.0_f64; n_cols];
    col_scale[7] = 2.0;

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.3, // scaled space: theta + fpha_scaled
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &col_scale,
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // turbined_cost = primal * obj_coeff / col_scale * K
    //                    = 30 * 0.01 / 2.0 * 1_000_000 = 150_000.0
    assert!(
        (cost.turbined_cost - 150_000.0).abs() < 1e-6,
        "turbined_cost should be 150_000.0 (unscaled by col_scale=2.0), got {}",
        cost.turbined_cost
    );
}

/// Verify that `compute_cost_result` decomposes `hydro_violation_cost` into
/// the 6 per-constraint components, and that the sum invariant holds.
#[test]
fn hydro_violation_cost_decomposition() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = crate::test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = crate::test_support::state_layout(2, 0);
    // Layout (see make_indexer_2h_1fpha_1blk):
    //   withdrawal_slack_neg:   14..16
    //   withdrawal_slack_pos:   16..18
    //   outflow_below_slack:    18..20
    //   outflow_above_slack:    20..22
    //   turbine_below_slack:    22..24
    //   generation_below_slack: 24..26
    assert_eq!(indexer.withdrawal_slack_neg, 14..16);
    assert_eq!(indexer.withdrawal_slack_pos, 16..18);
    assert_eq!(indexer.outflow_below_slack, 18..20);
    assert_eq!(indexer.outflow_above_slack, 20..22);
    assert_eq!(indexer.turbine_below_slack, 22..24);
    assert_eq!(indexer.generation_below_slack, 24..26);
    assert!(study_dims.has_operational_violations);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    let mut obj = vec![0.0_f64; n_cols];

    // Set theta so objective algebra works.
    primal[state.theta] = 0.0;

    // Assign known primal (slack) and objective (penalty) values per
    // constraint type. Each slack * penalty gives a known cost contribution.
    // COST_SCALE_FACTOR = 1_000_000.0 is applied inside the extraction.

    // outflow_below: h0=2.0 * 10.0, h1=3.0 * 10.0  => (20+30) * 1_000_000 = 50_000_000
    primal[18] = 2.0;
    obj[18] = 10.0;
    primal[19] = 3.0;
    obj[19] = 10.0;

    // outflow_above: h0=1.0 * 5.0, h1=0.0  => 5 * 1_000_000 = 5_000_000
    primal[20] = 1.0;
    obj[20] = 5.0;

    // turbine_below: h0=4.0 * 8.0, h1=0.0  => 32 * 1_000_000 = 32_000_000
    primal[22] = 4.0;
    obj[22] = 8.0;

    // generation_below: h0=0.0, h1=6.0 * 3.0  => 18 * 1_000_000 = 18_000_000
    primal[25] = 6.0;
    obj[25] = 3.0;

    // withdrawal (neg): h0=0.5 * 20.0, h1=0.0  => 10 * 1_000_000 = 10_000_000
    primal[14] = 0.5;
    obj[14] = 20.0;

    // withdrawal (pos): h0=0.0, h1=0.3 * 15.0  => 4.5 * 1_000_000 = 4_500_000
    primal[17] = 0.3;
    obj[17] = 15.0;

    // Total objective for the LP (sum of primal * obj):
    let total_obj: f64 = primal.iter().zip(obj.iter()).map(|(p, o)| p * o).sum();

    let dual = vec![0.0_f64; 2];
    let row_lower = vec![0.0_f64; 1];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: total_obj,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];

    // Expected values (primal * obj * COST_SCALE_FACTOR):
    let expected_outflow_below = (2.0 * 10.0 + 3.0 * 10.0) * 1_000_000.0;
    let expected_outflow_above = (1.0 * 5.0) * 1_000_000.0;
    let expected_turbined = (4.0 * 8.0) * 1_000_000.0;
    let expected_generation = (6.0 * 3.0) * 1_000_000.0;
    let expected_withdrawal = (0.5 * 20.0 + 0.3 * 15.0) * 1_000_000.0;
    let expected_evaporation = 0.0; // no evaporation hydros

    assert!(
        (cost.outflow_violation_below_cost - expected_outflow_below).abs() < 1e-6,
        "outflow_below: expected {expected_outflow_below}, got {}",
        cost.outflow_violation_below_cost
    );
    assert!(
        (cost.outflow_violation_above_cost - expected_outflow_above).abs() < 1e-6,
        "outflow_above: expected {expected_outflow_above}, got {}",
        cost.outflow_violation_above_cost
    );
    assert!(
        (cost.turbined_violation_cost - expected_turbined).abs() < 1e-6,
        "turbined: expected {expected_turbined}, got {}",
        cost.turbined_violation_cost
    );
    assert!(
        (cost.generation_violation_cost - expected_generation).abs() < 1e-6,
        "generation: expected {expected_generation}, got {}",
        cost.generation_violation_cost
    );
    assert!(
        (cost.evaporation_violation_cost - expected_evaporation).abs() < 1e-6,
        "evaporation: expected {expected_evaporation}, got {}",
        cost.evaporation_violation_cost
    );
    assert!(
        (cost.withdrawal_violation_cost - expected_withdrawal).abs() < 1e-6,
        "withdrawal: expected {expected_withdrawal}, got {}",
        cost.withdrawal_violation_cost
    );

    // Sum invariant: hydro_violation_cost == sum of all 6 components.
    let component_sum = cost.outflow_violation_below_cost
        + cost.outflow_violation_above_cost
        + cost.turbined_violation_cost
        + cost.generation_violation_cost
        + cost.evaporation_violation_cost
        + cost.withdrawal_violation_cost;
    assert!(
        (cost.hydro_violation_cost - component_sum).abs() < 1e-6,
        "hydro_violation_cost ({}) must equal sum of components ({component_sum})",
        cost.hydro_violation_cost
    );
}

/// Build an [`EnergyConversionSet`] for the (1 hydro, 1 stage) case with
/// explicit `ρ_eq` and `ρ_acum` values.
fn one_hydro_energy_set(
    rho_eq: f64,
    rho_acum: f64,
) -> crate::energy_conversion::EnergyConversionSet {
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    let cell = EnergyConversion {
        equivalent_productivity_mw_per_m3s: rho_eq,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(vec![vec![cell; 1]; 1], vec![vec![rho_acum; 1]; 1], 1, 1)
}

fn make_entity_counts_1_hydro() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![10],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a primal vector for the N=1, L=1 state layout:
/// `[storage, lag, z_inflow, storage_in, theta]` (length 5).
fn make_primal_1_1(storage: f64, storage_in: f64, theta: f64) -> Vec<f64> {
    vec![storage, 0.0, 0.0, storage_in, theta]
}

#[test]
fn stored_energy_initial_uses_v_min_offset() {
    // storage_initial = 110, V_min = 100, ρ_acum = 4.0 →
    // stored_energy_initial = (110 - 100) * 4 * 1e6 / 3600 ≈ 11_111.111…
    let indexer = crate::test_support::geom(1, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 1);
    let primal = make_primal_1_1(120.0, 110.0, 0.0);
    let dual = vec![0.0; 2];
    let ec = one_hydro_energy_set(0.9, 4.0);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    let h = &result.hydros[0];
    assert!(
        (h.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
        "equivalent_productivity should be 0.9, got {}",
        h.equivalent_productivity_mw_per_m3s
    );
    assert!(
        (h.accumulated_productivity_mw_per_m3s - 4.0).abs() < 1e-12,
        "accumulated_productivity should be 4.0, got {}",
        h.accumulated_productivity_mw_per_m3s
    );
    // storage_initial_hm3 = 110.0 (from primal storage_in), V_min = 100.0.
    let expected = (110.0_f64 - 100.0) * 4.0 * 1.0e6 / 3600.0;
    assert!(
        (h.stored_energy_initial_mwh - expected).abs() < 1e-6,
        "stored_energy_initial: expected {expected}, got {}",
        h.stored_energy_initial_mwh
    );
}

#[test]
fn incremental_inflow_energy_uses_rho_acum() {
    // ρ_acum = 4.0, incremental_inflow = 50.0 →
    // incremental_inflow_energy = 4.0 * 50.0 = 200.0 (exactly).
    let indexer = crate::test_support::geom(1, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 1);
    let primal = make_primal_1_1(120.0, 110.0, 0.0);
    let dual = vec![0.0; 2];
    let ec = one_hydro_energy_set(0.9, 4.0);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[50.0],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    assert!(
        (result.hydros[0].incremental_inflow_energy_mw - 200.0).abs() < 1e-12,
        "incremental_inflow_energy should be 200.0, got {}",
        result.hydros[0].incremental_inflow_energy_mw
    );
}

#[test]
fn stage_path_propagates_productivity_values() {
    // The per-stage path (block_hours empty) must read ρ_eq and ρ_acum
    // from the supplied EnergyConversionSet and surface them on the
    // result. The per-block path shares the same HydroStageContext,
    // so by construction it cannot disagree.
    let indexer = crate::test_support::geom(1, 1);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 1);
    let primal = make_primal_1_1(120.0, 110.0, 0.0);
    let dual = vec![0.0; 2];
    let ec = one_hydro_energy_set(0.85, 3.5);

    let stage_result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &[],
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            state: &state,
            n_blks: indexer.n_blks,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[10.0],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_start: 0,
            n_ncs: 0,
            ncs_entity_ids: &[],
            ncs_col_upper: &[],
            pumping_col_start: 0,
            n_pumping: 0,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_is_import: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            n_stages: 1,
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let rho_eq = stage_result.hydros[0].equivalent_productivity_mw_per_m3s;
    let rho_acum = stage_result.hydros[0].accumulated_productivity_mw_per_m3s;
    assert!((rho_eq - 0.85).abs() < 1e-12, "stage rho_eq = {rho_eq}");
    assert!(
        (rho_acum - 3.5).abs() < 1e-12,
        "stage rho_acum = {rho_acum}"
    );
}

// -------------------------------------------------------------------------
// extract_pumping_stations
// -------------------------------------------------------------------------

/// `EntityCounts` carrying a single pumping station (id `42`) and no other
/// entities, for the pumping-extraction tests.
fn entity_counts_one_pumping_station() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![42],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a `StageExtractionSpec` whose only meaningful fields are the
/// pumping inputs; all other fields are inert defaults.  The spec's `n_blks`
/// is taken from `geometry.n_blks`. Under the dense layout
/// `extract_pumping_stations` indexes the station id list directly by the
/// system column position, so the caller need only supply matching
/// `pumping_col_start`/`n_pumping`.
// Rationale: this helper assembles a single inert-default StageExtractionSpec
// for the pumping tests; its parameter list mirrors the spec's borrowed
// inputs, so bundling them would obscure rather than clarify.
#[allow(clippy::too_many_arguments)]
fn pumping_only_spec<'a>(
    study_dims: &'a StudyDimensions,
    geometry: &'a StageGeometry,
    state: &'a crate::indexer::StateLayout,
    entity_counts: &'a EntityCounts,
    pumping_col_start: usize,
    n_pumping: usize,
    consumption: &'a [f64],
    ec: &'a crate::energy_conversion::EnergyConversionSet,
    diversion: &'a HashMap<cobre_core::EntityId, Vec<usize>>,
) -> StageExtractionSpec<'a> {
    StageExtractionSpec {
        study_dims,
        geometry,
        state,
        n_blks: geometry.n_blks,
        entity_counts,
        inflow_m3s_per_hydro: &[],
        block_hours: &[],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start,
        n_pumping,
        pumping_consumption_mw_per_m3s: consumption,
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: ec,
        hydro_min_storage_hm3: &[],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    }
}

/// AC-1: one station, two blocks, distinct flow primals
/// (`[7.0, 3.0]`) and a non-unit `consumption_mw_per_m3s = 0.5`.
///
/// The pumping columns are placed at `pumping_col_start = 4` in the primal,
/// block-major (`col + p_idx * n_blks + blk`).  Each row's
/// `pumped_flow_m3s` is read directly from `view.primal` (already unscaled —
/// no `col_scale` division) and `power_consumption_mw = flow * consumption`.
#[test]
fn extract_pumping_two_blocks_reads_per_block_flow_and_power() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_one_pumping_station();
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let consumption = [0.5_f64];

    // Primal layout: filler columns 0..4, then pumping block at 4..6:
    //   col 4 = (station 0, block 0) = 7.0
    //   col 5 = (station 0, block 1) = 3.0
    let primal = vec![0.0, 0.0, 0.0, 0.0, 7.0, 3.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = StageGeometry {
        n_blks: 2,
        ..StageGeometry::default()
    };
    let study_dims = crate::test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        4,
        1,
        &consumption,
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 9);

    assert_eq!(rows.len(), 2, "one station x two blocks = two rows");

    assert_eq!(rows[0].stage_id, 9);
    assert_eq!(rows[0].pumping_station_id, 42);
    assert_eq!(rows[0].block_id, Some(0));
    assert_eq!(rows[0].pumped_flow_m3s, 7.0);
    assert_eq!(rows[0].power_consumption_mw, 3.5); // 7.0 * 0.5
    assert_eq!(rows[0].pumping_cost, 0.0);
    assert_eq!(rows[0].operative_state_code, 1);

    assert_eq!(rows[1].block_id, Some(1));
    assert_eq!(rows[1].pumped_flow_m3s, 3.0);
    assert_eq!(rows[1].power_consumption_mw, 1.5); // 3.0 * 0.5
    assert_eq!(rows[1].pumping_station_id, 42);
}

/// AC-2: zero pumping stations yields an empty `Vec` (the block loop never
/// runs), independent of `n_blks`.
#[test]
fn extract_pumping_zero_stations_is_empty() {
    let mut indexer = crate::test_support::geom(0, 0);
    let state = crate::test_support::state_layout(0, 0);
    indexer.n_blks = 2;
    let entity_counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = StageGeometry::default();
    let study_dims = crate::test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        0,
        0,
        &[],
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 0);
    assert!(rows.is_empty(), "no stations => no rows");
}

/// `n_blks == 0` also yields an empty `Vec` even with an active station.
#[test]
fn extract_pumping_zero_blocks_is_empty() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_one_pumping_station();
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let consumption = [0.5_f64];
    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = StageGeometry::default();
    let study_dims = crate::test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        0,
        1,
        &consumption,
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 0);
    assert!(rows.is_empty(), "n_blks == 0 => no rows");
}

// -------------------------------------------------------------------------
// extract_contracts
// -------------------------------------------------------------------------

fn entity_counts_contracts(contract_ids: Vec<i32>) -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids,
        non_controllable_ids: vec![],
    }
}

/// Build a `StageExtractionSpec` whose only meaningful fields are the contract
/// inputs; all other fields are inert defaults. `n_blks` is `geometry.n_blks`.
// Rationale: mirrors `pumping_only_spec`; the parameter list IS the spec's
// borrowed contract inputs, so bundling them would obscure rather than clarify.
#[allow(clippy::too_many_arguments)]
fn contract_only_spec<'a>(
    study_dims: &'a StudyDimensions,
    geometry: &'a StageGeometry,
    state: &'a crate::indexer::StateLayout,
    entity_counts: &'a EntityCounts,
    block_hours: &'a [f64],
    contract_prices: &'a [f64],
    contract_is_import: &'a [bool],
    ec: &'a crate::energy_conversion::EnergyConversionSet,
    diversion: &'a HashMap<cobre_core::EntityId, Vec<usize>>,
) -> StageExtractionSpec<'a> {
    StageExtractionSpec {
        study_dims,
        geometry,
        state,
        n_blks: geometry.n_blks,
        entity_counts,
        inflow_m3s_per_hydro: &[],
        block_hours,
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices,
        contract_is_import,
        diversion_upstream: diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: ec,
        hydro_min_storage_hm3: &[],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    }
}

/// AC: an import contract column holding `40.0` with `block_hours = 730` and
/// resolved price `200` yields `power_mw == 40.0`,
/// `total_cost == 200 * 40 * 730`, and `operative_state_code == 1`.
#[test]
fn extract_contract_import_reads_primal_and_cost() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![7]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // Import block at column 4..5 (one import, one block); export block empty at 5..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 40.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..5,
        contract_export: 5..5,
        ..StageGeometry::default()
    };
    let study_dims = crate::test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [200.0_f64];
    let is_import = [true];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &is_import,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 9);

    assert_eq!(rows.len(), 1, "one contract x one block = one row");
    assert_eq!(rows[0].stage_id, 9);
    assert_eq!(rows[0].contract_id, 7);
    assert_eq!(rows[0].block_id, Some(0));
    assert_eq!(rows[0].power_mw, 40.0);
    assert_eq!(rows[0].price_per_mwh, 200.0);
    assert_eq!(rows[0].total_cost, 200.0 * 40.0 * 730.0);
    assert_eq!(rows[0].operative_state_code, 1);
}

/// AC: an export contract column holding `30.0` with resolved price `-150` and
/// `block_hours = 730` yields a negative `total_cost` (export revenue), addressed
/// from the export family base.
#[test]
fn extract_contract_export_yields_negative_cost() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![8]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // No imports (empty 4..4); one export at column 4..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 30.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..4,
        contract_export: 4..5,
        ..StageGeometry::default()
    };
    let study_dims = crate::test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [-150.0_f64];
    let is_import = [false];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &is_import,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 0);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].contract_id, 8);
    assert_eq!(rows[0].power_mw, 30.0);
    assert_eq!(rows[0].total_cost, -150.0 * 30.0 * 730.0);
    assert!(rows[0].total_cost < 0.0, "export revenue is negative");
    assert_eq!(rows[0].operative_state_code, 1);
}

/// AC: a dormant contract whose column is pinned to `0.0` yields `power_mw == 0.0`,
/// `total_cost == 0.0`, and still `operative_state_code == 1` (never a
/// commissioning flag).
#[test]
fn extract_contract_dormant_zero_row_keeps_state_code_1() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![3]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // Import column pinned to 0.0 (dormant) at 4..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..5,
        contract_export: 5..5,
        ..StageGeometry::default()
    };
    let study_dims = crate::test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [200.0_f64];
    let is_import = [true];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &is_import,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 0);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].power_mw, 0.0);
    assert_eq!(rows[0].total_cost, 0.0);
    assert_eq!(rows[0].operative_state_code, 1);
}

/// AC: a contract-free system yields an empty contracts vector from
/// `extract_stub_collections` (parity-neutral with the prior zero-placeholder
/// behaviour, which also emitted no real dispatch).
#[test]
fn extract_stub_collections_contract_free_is_empty() {
    let state = crate::test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        ..StageGeometry::default()
    };
    let study_dims = crate::test_support::study_dims();
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &[730.0],
        &[],
        &[],
        &ec,
        &diversion,
    );

    let (_inflow_lags, _pumping, contracts): (Vec<_>, Vec<_>, Vec<SimulationContractResult>) =
        extract_stub_collections(&view, &spec, 0);
    assert!(contracts.is_empty(), "no contracts => empty vector");
}

// -------------------------------------------------------------------------
// Per-block chronological storage and evaporation
// -------------------------------------------------------------------------

/// One evaporating hydro, one operating block, `state_layout(1, 0)` gives
/// `storage_in.start == 2` and `storage.start == 0`.
fn entity_counts_1_hydro() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![7],
        hydro_productivities: vec![0.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a single-hydro `K`-block `StageGeometry` in the given mode.
///
/// Control-region layout for `state_layout(1, 0)` (`control_region_start == 4`):
/// interior storage `S¹ … Sᴷ⁻¹` at `[4, 4 + (K−1))`, then turbine `[t0, t0 + K)`,
/// spillage `[t0 + K, t0 + 2K)`, then `K` evaporation triples. In parallel mode the
/// interior family is empty and turbine begins at 4.
fn single_hydro_block_geometry(block_mode: cobre_core::BlockMode, k: usize) -> StageGeometry {
    use crate::indexer::EvaporationIndices;
    let n_interior = match block_mode {
        cobre_core::BlockMode::Chronological => k - 1,
        cobre_core::BlockMode::Parallel => 0,
    };
    let storage_internal_start = 4;
    let turbine_start = storage_internal_start + n_interior;
    let spillage_start = turbine_start + k;
    let evap_start = spillage_start + k;
    let evap_indices: Vec<EvaporationIndices> = (0..k)
        .map(|b| {
            let base = evap_start + b * 3;
            EvaporationIndices {
                evaporation_flow_col: base,
                f_evap_plus_col: base + 1,
                f_evap_minus_col: base + 2,
                evap_row: b,
            }
        })
        .collect();
    StageGeometry {
        turbine: turbine_start..spillage_start,
        spillage: spillage_start..evap_start,
        n_blks: k,
        storage_internal_start,
        block_mode,
        evap_indices,
        evap_hydro_indices: vec![0],
        ..StageGeometry::default()
    }
}

/// Chronological block `b` reports `(Sᵇ, Sᵇ⁺¹)` read through the accessor, and the
/// boundary chain's endpoints coincide with the state columns (`S⁰`/`Sᴷ`).
#[test]
fn extract_chronological_per_block_storage() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(cobre_core::BlockMode::Chronological, k);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    // Boundary values: S⁰=10 (storage_in col 2), S¹=20 (col 4), S²=30 (col 5),
    // S³=Sᴷ=40 (storage col 0). Turbine cols [6,9), evap triples at [12, ...).
    let n_cols = geom.spillage.end + k * 3;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ (outgoing state)
    primal[2] = 10.0; // S⁰ (incoming state)
    primal[4] = 20.0; // S¹ (interior)
    primal[5] = 30.0; // S² (interior)
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        state: &state,
        n_blks: k,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);

    let boundaries = [10.0, 20.0, 30.0, 40.0];
    for b in 0..k {
        let row = &result.hydros[b];
        assert_eq!(row.block_id, Some(b as u32));
        assert_eq!(
            row.storage_initial_hm3, boundaries[b],
            "block {b} incoming == Sᵇ"
        );
        assert_eq!(
            row.storage_final_hm3,
            boundaries[b + 1],
            "block {b} outgoing == Sᵇ⁺¹"
        );
    }

    // Endpoints coincide with the state region.
    assert_eq!(
        result.hydros[0].storage_initial_hm3,
        primal[state.storage_in.start]
    );
    assert_eq!(
        result.hydros[k - 1].storage_final_hm3,
        primal[state.storage.start]
    );
}

/// Parallel mode: every block row reports the stage-level `(S⁰, Sᴷ)` state pair,
/// `.to_bits()`-identical to the direct state-column reads.
#[test]
fn extract_parallel_per_block_storage_byte_identical() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(cobre_core::BlockMode::Parallel, k);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.spillage.end + k * 3;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ (outgoing state)
    primal[2] = 10.0; // S⁰ (incoming state)
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        state: &state,
        n_blks: k,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);
    let s0 = primal[state.storage_in.start];
    let sk = primal[state.storage.start];
    for row in &result.hydros {
        assert_eq!(row.storage_initial_hm3.to_bits(), s0.to_bits());
        assert_eq!(row.storage_final_hm3.to_bits(), sk.to_bits());
    }
}

/// Chronological per-block stored energy derives from each block's own boundary:
/// `(S − V_min) · ρ_acum · ENERGY_FACTOR`, with `V_min` / `ρ_acum` block-invariant.
#[test]
fn extract_chronological_per_block_stored_energy() {
    use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
    let k = 3_usize;
    let geom = single_hydro_block_geometry(cobre_core::BlockMode::Chronological, k);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);

    let rho_acum = 2.0_f64;
    let v_min = 5.0_f64;
    let ec = EnergyConversionSet::new(
        vec![vec![EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        }]],
        vec![vec![rho_acum]],
        1,
        1,
    );

    let n_cols = geom.spillage.end + k * 3;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ
    primal[2] = 10.0; // S⁰
    primal[4] = 20.0; // S¹
    primal[5] = 30.0; // S²
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        state: &state,
        n_blks: k,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[v_min],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    let boundaries = [10.0, 20.0, 30.0, 40.0];
    let expected = |s: f64| -> f64 {
        (s - v_min) * rho_acum * super::ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S
    };
    for b in 0..k {
        let row = &result.hydros[b];
        assert!((row.stored_energy_initial_mwh - expected(boundaries[b])).abs() < 1e-9);
        assert!((row.stored_energy_final_mwh - expected(boundaries[b + 1])).abs() < 1e-9);
    }
}

/// Chronological block `b` reads its OWN evaporation triple
/// `evap_indices[local * n_blks + b]`, not block 0's.
#[test]
fn extract_chronological_per_block_evaporation() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(cobre_core::BlockMode::Chronological, k);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.spillage.end + k * 3;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0;
    primal[2] = 10.0;
    primal[4] = 20.0;
    primal[5] = 30.0;
    // Distinct evaporation values per block: block b flow = 1.0 + b, neg = 0.1 + b,
    // pos = 0.2 + b, laid out block-major at evap_indices[b].
    for b in 0..k {
        let ei = &geom.evap_indices[b];
        primal[ei.evaporation_flow_col] = 1.0 + b as f64;
        primal[ei.f_evap_plus_col] = 0.1 + b as f64;
        primal[ei.f_evap_minus_col] = 0.2 + b as f64;
    }
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        state: &state,
        n_blks: k,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    for b in 0..k {
        let row = &result.hydros[b];
        assert_eq!(row.evaporation_m3s, Some(1.0 + b as f64), "block {b} flow");
        assert!(
            (row.evaporation_violation_neg_m3s - (0.1 + b as f64)).abs() < 1e-12,
            "block {b} neg violation reads its own triple"
        );
        assert!(
            (row.evaporation_violation_pos_m3s - (0.2 + b as f64)).abs() < 1e-12,
            "block {b} pos violation reads its own triple"
        );
    }
}

/// Parallel mode: every block row's evaporation fields are `.to_bits()`-identical to
/// the block-0 read (byte-identical to the pre-change behavior).
#[test]
fn extract_parallel_per_block_evaporation_byte_identical() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(cobre_core::BlockMode::Parallel, k);
    let study_dims = crate::test_support::study_dims();
    let state = crate::test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.spillage.end + k * 3;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0;
    primal[2] = 10.0;
    for b in 0..k {
        let ei = &geom.evap_indices[b];
        primal[ei.evaporation_flow_col] = 1.0 + b as f64;
        primal[ei.f_evap_plus_col] = 0.1 + b as f64;
        primal[ei.f_evap_minus_col] = 0.2 + b as f64;
    }
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        state: &state,
        n_blks: k,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_start: 0,
        n_ncs: 0,
        ncs_entity_ids: &[],
        ncs_col_upper: &[],
        pumping_col_start: 0,
        n_pumping: 0,
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_is_import: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        n_stages: 1,
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    // Block 0's triple is the parallel stage-level read.
    let flow0 = 1.0_f64;
    let neg0 = 0.1_f64;
    let pos0 = 0.2_f64;
    for row in &result.hydros {
        assert_eq!(row.evaporation_m3s.map(f64::to_bits), Some(flow0.to_bits()));
        assert_eq!(row.evaporation_violation_neg_m3s.to_bits(), neg0.to_bits());
        assert_eq!(row.evaporation_violation_pos_m3s.to_bits(), pos0.to_bits());
    }
}
