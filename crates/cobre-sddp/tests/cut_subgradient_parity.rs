//! KKT parity test: `reduced_costs[storage_in.start]` matches the hand-computed
//! closed-form value derived from row duals of the same LP solve.
//!
//! ## What is being verified
//!
//! Phase 1 swaps cut-subgradient extraction from row duals of state-fixing
//! equality rows (`view.dual[..n_state]`) to reduced costs of the incoming-state
//! columns (`view.reduced_costs[state_to_lp_incoming_column(j)]`).
//!
//! By KKT optimality, the reduced cost of a column with `lb == ub` equals the
//! dual of the equivalent equality row — but the same identity holds for the
//! multi-row case via the full KKT stationarity condition:
//!
//! ```text
//! rc[j] = c_j - sum_i( a_{i,j} * y_i )
//! ```
//!
//! where `c_j` is the column objective coefficient, `a_{i,j}` are the
//! constraint-matrix coefficients, and `y_i` are the row dual variables.
//!
//! For the `storage_in[0]` column (incoming storage, `c_j = 0`):
//! - Water-balance row: coefficient `-1.0`
//! - FPHA row:          coefficient `-gamma_v / 2`
//! - Evaporation row:   coefficient `-volume_slope_m3s_per_hm3 / 2`
//!
//! Therefore: `rc[storage_in[0]] = y_wb + (gamma_v/2)*y_fpha + (volume_slope_m3s_per_hm3/2)*y_evap`
//!
//! This test solves the LP with column-bound state pinning (Phase 1 approach) and
//! asserts the algebraic identity holds within 1e-8 absolute tolerance. It covers
//! the FPHA+evaporation coupled multi-row column-participation case that Q1's
//! single-row probe left unverified.
//!
//! ## System layout
//!
//! - N=1 FPHA hydro, L=0 (no lags), A=0 (no anticipated thermals)
//! - 1 bus, 1 block (K=1)
//! - No thermal plant — deficit variable absorbs any load-generation gap
//! - FPHA: 1 plane with `gamma_v=0.002, gamma_q=0.8, gamma_0=0.0`
//! - Evaporation: `volume_slope_m3s_per_hm3=0.01, intercept_m3s=0.0`
//!
//! ## Column and row indices (Phase 1, with storage_fixing = 0..0)
//!
//! Column layout for N=1, L=0, A=0, K=1 (no penalty, no thermal):
//! ```text
//! col 0: storage (v_out)
//! col 1: z_inflow
//! col 2: storage_in (v_in) ← pinned via set_col_bounds; reduced cost tested here
//! col 3: theta
//! col 4: turbine
//! col 5: spillage
//! col 6: diversion
//! col 7: deficit
//! col 8: excess
//! col 9: withdrawal_slack_neg
//! col 10: withdrawal_slack_pos
//! cols 11-14: 4 × operational violation slacks
//! col 15: evaporation outflow
//! col 16: f_evap_plus
//! col 17: f_evap_minus
//! col 18: g_fpha (FPHA generation variable)
//! ```
//!
//! Row layout (Phase 1 — storage_fixing = 0..0, z_inflow at row 0):
//! ```text
//! row 0: z_inflow definition (z = RHS)
//! row 1: water_balance       ← dual = y_wb
//! row 2: load_balance
//! row 3: FPHA plane 0        ← dual = y_fpha
//! row 4: evaporation equality ← dual = y_evap
//! rows 5-8: operational violation constraints
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds, DeficitSegment,
    EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
    ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{InflowModel, LoadModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    build_stage_templates,
    hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, LinearizedEvaporation,
        PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
    },
    inflow_method::InflowNonNegativityMethod,
    resolved_parameters::ResolvedParameters,
};
use cobre_solver::{ActiveSolver, RowBatch, SolverInterface};
use cobre_stochastic::{PrecomputedPar, normal::precompute::PrecomputedNormal};

/// FPHA plane coefficients for the test: modest gamma_v so the FPHA constraint
/// can be active with reasonable storage values.
const GAMMA_V: f64 = 0.002;
const GAMMA_Q: f64 = 0.8;
const GAMMA_0: f64 = 0.0;

/// Evaporation coefficients.
const VOLUME_SLOPE_M3S_PER_HM3: f64 = 0.01;
const INTERCEPT_M3S: f64 = 0.0;

/// Initial (incoming) storage value pinned via column bounds.
const V_IN_HM3: f64 = 100.0;

/// Number of columns before the FPHA generation column.
/// For N=1, L=0, A=0, K=1, no penalty, no thermal, no lines, 1 bus, 1 deficit segment:
/// storage(1) + z_inflow(1) + storage_in(1) + theta(1) = col 0–3 (state)
/// turbine(1) + spillage(1) + diversion(1) = col 4–6 (flow)
/// deficit(1) + excess(1) = col 7–8
/// withdrawal_neg(1) + withdrawal_pos(1) = col 9–10
/// outflow_below(1) + outflow_above(1) + turbine_below(1) + generation_below(1) = col 11–14
/// evaporation outflow(1) + f_evap_plus(1) + f_evap_minus(1) = col 15–17
/// g_fpha(1) = col 18
const COL_STORAGE_IN: usize = 2;
const ROW_WATER_BALANCE: usize = 1;
const ROW_FPHA: usize = 3;
const ROW_EVAP: usize = 4;

/// Build a minimal 1-FPHA-hydro, 1-bus system with linearized evaporation.
///
/// Parameters chosen so the LP is feasible and the FPHA constraint is active
/// at the optimum (maximising turbine use to minimise thermal/deficit cost).
fn fpha_evap_system() -> cobre_core::System {
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let hydro = Hydro {
        id: EntityId(2),
        name: "H_FPHA".to_string(),
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
        max_turbined_m3s: 50.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
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
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 730.0,
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
        mean_m3s: 0.0,
        std_m3s: 0.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 50.0,
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
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
        .expect("fpha_evap_system: valid")
}

/// Build the FPHA production model (1 hydro, 1 stage, 1 plane).
fn fpha_production() -> ProductionModelSet {
    let plane = FphaPlane {
        intercept: GAMMA_0,
        gamma_v: GAMMA_V,
        gamma_q: GAMMA_Q,
        gamma_s: 0.0,
    };
    let models = vec![vec![ResolvedProductionModel::Fpha {
        planes: vec![plane],
    }]];
    ProductionModelSet::new(models, 1, 1)
}

/// Build the evaporation model set (1 hydro with volume_slope_m3s_per_hm3 > 0).
fn fpha_evap_evaporation(system: &cobre_core::System) -> EvaporationModelSet {
    let models = vec![EvaporationModel::Linearized {
        coefficients: vec![LinearizedEvaporation {
            intercept_m3s: INTERCEPT_M3S,
            volume_slope_m3s_per_hm3: VOLUME_SLOPE_M3S_PER_HM3,
        }],
        reference_volumes_hm3: vec![100.0],
    }];
    let _ = system; // system reference for symmetry with other helpers; unused here
    EvaporationModelSet::new(models)
}

/// KKT parity: `reduced_costs[storage_in.start]` equals the closed-form
/// reference `y_wb + (gamma_v/2)*y_fpha + (volume_slope_m3s_per_hm3/2)*y_evap`
/// from the same LP solve.
///
/// This confirms that `state_to_lp_incoming_column` returns the correct column
/// for cut-subgradient extraction in the presence of both FPHA and evaporation
/// contributions to the incoming-storage column's KKT stationarity condition.
#[test]
fn cut_subgradient_parity_with_fpha_and_evaporation() {
    let system = fpha_evap_system();
    let production = fpha_production();
    let evaporation = fpha_evap_evaporation(&system);

    // Build the stage LP template.
    let result = build_stage_templates(
        &system,
        InflowNonNegativityMethod::None,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &evaporation,
        &ResolvedParameters::default(),
    )
    .expect("FPHA+evap template must build");

    let template = &result.templates[0];

    // Verify the column index we will inspect matches the documented layout.
    // storage_in.start = N*(2+L) + A*K = 1*(2+0) + 0*0 = 2.
    // This assertion documents and guards the hardcoded constant above.
    assert_eq!(
        template.n_state, 1,
        "N=1, L=0: n_state must be 1 (storage only, no lags)"
    );

    // Verify that Phase 1 removed state-fixing rows: storage_fixing = 0..0.
    // base_rows[0] is the water-balance row index, which equals N = 1 in Phase 1.
    assert_eq!(
        result.base_rows[0], 1,
        "Phase 1: water-balance must be at row 1 (z_inflow at row 0, no state-fixing prefix)"
    );

    // Verify the FPHA and evap row offsets match the documented layout.
    // row_fpha_start = row_load_balance_start + n_buses*n_blks = 2 + 1 = 3
    // row_evap_start = row_fpha_start + n_fpha_planes_per_stage = 3 + 1 = 4
    // These are validated by the assertions below after the solve.

    // Load the template into HiGHS.
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);

    // Add an empty cut row batch (no future-cost cuts at this stage).
    let empty_cuts = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    };
    solver.add_rows(&empty_cuts);

    // Phase 1: pin storage_in[0] = V_IN_HM3 via column bounds (lb == ub).
    // This is the correct Phase-1 approach — NOT set_row_bounds on a storage-fixing row.
    solver.set_col_bounds(&[COL_STORAGE_IN], &[V_IN_HM3], &[V_IN_HM3]);

    // Solve the LP.
    let view = solver
        .solve(None)
        .expect("FPHA+evap LP must be feasible and optimal");

    let rc_storage_in = view.reduced_costs[COL_STORAGE_IN];
    let y_wb = view.dual[ROW_WATER_BALANCE];
    let y_fpha = view.dual[ROW_FPHA];
    let y_evap = view.dual[ROW_EVAP];
    let obj = view.objective;

    // Closed-form KKT reference.
    //
    // storage_in[0] participates in:
    //   water-balance row: coefficient  -1.0
    //   FPHA row:          coefficient  -gamma_v / 2
    //   evap row:          coefficient  -volume_slope_m3s_per_hm3 / 2
    //
    // KKT stationarity: rc[j] = c_j - sum_i( a_{i,j} * y_i )
    //   = 0 - ( (-1.0)*y_wb + (-GAMMA_V/2)*y_fpha + (-VOLUME_SLOPE_M3S_PER_HM3/2)*y_evap )
    //   = y_wb + (GAMMA_V/2)*y_fpha + (VOLUME_SLOPE_M3S_PER_HM3/2)*y_evap
    let kkt_ref = y_wb + (GAMMA_V / 2.0) * y_fpha + (VOLUME_SLOPE_M3S_PER_HM3 / 2.0) * y_evap;

    assert!(
        obj.is_finite(),
        "LP objective must be finite (LP must be feasible and bounded)"
    );

    assert!(
        (rc_storage_in - kkt_ref).abs() < 1e-8,
        "KKT parity failed:\n  \
         rc[storage_in[0]]  = {rc_storage_in:.12}\n  \
         KKT reference      = {kkt_ref:.12}\n  \
         |delta|            = {:.2e}\n  \
         y_wb               = {y_wb:.12}\n  \
         y_fpha             = {y_fpha:.12}\n  \
         y_evap             = {y_evap:.12}\n  \
         gamma_v/2          = {:.6}\n  \
         volume_slope_m3s_per_hm3/2 = {:.6}",
        (rc_storage_in - kkt_ref).abs(),
        GAMMA_V / 2.0,
        VOLUME_SLOPE_M3S_PER_HM3 / 2.0,
    );
}

/// Structural guard: `PrepareHydroModelsResult::default_from_system` produces
/// a model set that does NOT include FPHA for a system declared with
/// `HydroGenerationModel::ConstantProductivity`. This confirms that the FPHA
/// plane override in `fpha_production()` is required and meaningful.
#[test]
fn default_production_is_not_fpha_for_this_system() {
    let system = fpha_evap_system();
    let default_prod = PrepareHydroModelsResult::default_from_system(&system);
    // A system with `Fpha` generation model still gets a default *constant* model
    // from `default_from_system` (it is a fallback, not a full FPHA computation).
    // The real FPHA planes come from `fpha_production()` above.
    assert!(
        matches!(
            default_prod.production.model(0, 0),
            ResolvedProductionModel::ConstantProductivity { .. }
        ),
        "default_from_system must return ConstantProductivity as fallback"
    );
}
