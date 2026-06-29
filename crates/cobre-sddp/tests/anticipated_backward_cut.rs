//! Analytical verification of backward-pass cut-coefficient extraction for an
//! anticipated thermal across lead_stages K = 1, 2, 3. Each K's closed-form
//! derivation lives on its test function.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use cobre_core::entities::{
    bus::{Bus, DeficitSegment},
    thermal::{AnticipatedConfig, Thermal},
};
use cobre_core::scenario::{LoadModel, SamplingScheme};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, InitialConditions,
    LineStageBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::{StudySetup, hydro_models::PrepareHydroModelsResult};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

mod common;
use common::StubComm;

// ---------------------------------------------------------------------------
// Numeric constants shared across all K (single source of truth).
// ---------------------------------------------------------------------------

const BLOCK_HOURS: f64 = 1.0;
const C_REG: f64 = 100.0;
const C_ANT: f64 = 50.0;

// Every non-theta objective coefficient is divided by this, so duals and the
// stored cut live in scaled cost units.
const COST_SCALE_FACTOR: f64 = 1_000_000.0;

const TOL: f64 = 1e-6;

// System::build sorts thermals by EntityId ascending; with reg_id < ant_id (R7),
// thermal_idx 0 is the regular thermal and 1 is the anticipated thermal.
const THERMAL_IDX_REG: usize = 0;
const THERMAL_IDX_ANT: usize = 1;

// ---------------------------------------------------------------------------
// Per-K fixture table
// ---------------------------------------------------------------------------

/// Per-K parameters for the anticipated backward-cut fixtures. Each `#[test]`
/// builds an independent `System` from one entry, so entity IDs need only be
/// disjoint within an entry (bus 1, reg, ant); cross-K reuse is harmless.
struct BackwardCutFixture {
    n_stages: usize,
    k_max: usize,
    /// Per-stage load, MW (length `n_stages`).
    loads_mw: &'static [f64],
    max_gen_reg: f64,
    max_gen_ant: f64,
    reg_id: EntityId,
    ant_id: EntityId,
    reg_start_date: (i32, u32, u32),
    ant_start_date: (i32, u32, u32),
    /// Anticipated ring-buffer seeds, MW (length `k_max`).
    seeds_mw: &'static [f64],
    iterations: usize,
    expected_coeff: f64,
}

const FIXTURE_K1: BackwardCutFixture = BackwardCutFixture {
    n_stages: 2,
    k_max: 1,
    loads_mw: &[10.0, 20.0],
    max_gen_reg: 50.0,
    max_gen_ant: 30.0,
    reg_id: EntityId(3),
    ant_id: EntityId(4),
    reg_start_date: (2024, 1, 4),
    ant_start_date: (2024, 1, 5),
    seeds_mw: &[10.0],
    iterations: 1,
    expected_coeff: -C_REG / COST_SCALE_FACTOR,
};

const FIXTURE_K2: BackwardCutFixture = BackwardCutFixture {
    n_stages: 3,
    k_max: 2,
    loads_mw: &[5.0, 10.0, 30.0],
    max_gen_reg: 100.0,
    max_gen_ant: 50.0,
    reg_id: EntityId(2),
    ant_id: EntityId(4),
    reg_start_date: (2024, 1, 3),
    ant_start_date: (2024, 1, 5),
    seeds_mw: &[0.0, 0.0],
    iterations: 5,
    expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
};

const FIXTURE_K3: BackwardCutFixture = BackwardCutFixture {
    n_stages: 4,
    k_max: 3,
    loads_mw: &[5.0, 10.0, 15.0, 30.0],
    max_gen_reg: 100.0,
    max_gen_ant: 50.0,
    reg_id: EntityId(2),
    ant_id: EntityId(5),
    reg_start_date: (2024, 1, 3),
    ant_start_date: (2024, 1, 6),
    seeds_mw: &[0.0, 0.0, 0.0],
    iterations: 5,
    expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
};

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

fn build_system(fixture: &BackwardCutFixture) -> cobre_core::System {
    use chrono::NaiveDate;

    let date = |d: (i32, u32, u32)| NaiveDate::from_ymd_opt(d.0, d.1, d.2).expect("valid date");

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        // Deficit cost set safely above c_reg so the LP never prefers shedding load.
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let thermal_reg = Thermal {
        id: fixture.reg_id,
        name: "T_reg".to_string(),
        operational_start_date: date(fixture.reg_start_date),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: fixture.max_gen_reg,
        cost_per_mwh: C_REG,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let thermal_ant = Thermal {
        id: fixture.ant_id,
        name: "T_ant".to_string(),
        operational_start_date: date(fixture.ant_start_date),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: fixture.max_gen_ant,
        cost_per_mwh: C_ANT,
        anticipated_config: Some(AnticipatedConfig {
            lead_stages: fixture.k_max as u32,
        }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    assert!(
        thermal_reg.id.0 < thermal_ant.id.0,
        "R7: T_reg.id ({}) must be strictly less than T_ant.id ({}) so that \
         System::build's sort_by_key aligns thermal_idx with the bounds table",
        thermal_reg.id.0,
        thermal_ant.id.0,
    );

    let stages: Vec<Stage> = (0..fixture.n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: BLOCK_HOURS,
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

    let load_models: Vec<LoadModel> = (0..fixture.n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: fixture.loads_mw[i],
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: fixture.n_stages,
            k_max: fixture.k_max,
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
    );

    // K-padded axis: fill_anticipated_columns reads delivery cells at
    // stage_idx + K_i, so overrides must cover the n_stages + k_max range.
    let thermal_axis = fixture.n_stages + fixture.k_max;
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(THERMAL_IDX_REG, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: fixture.max_gen_reg,
            cost_per_mwh: C_REG,
        };
        *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: fixture.max_gen_ant,
            cost_per_mwh: C_ANT,
        };
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: fixture.n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
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
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    // Anticipated ring-buffer seeds; per R6 any feasible choice yields the same cut.
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: fixture.ant_id,
            values_mw: fixture.seeds_mw.to_vec(),
        }],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_reg, thermal_ant])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system: valid")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

fn build_config(iterations: usize) -> Config {
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
            forward_passes: Some(1),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: iterations as u32,
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

// ---------------------------------------------------------------------------
// Setup builder
// ---------------------------------------------------------------------------

fn build_setup(system: cobre_core::System, config: &Config) -> StudySetup {
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
    .expect("build_stochastic_context");

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    StudySetup::new(&system, config, stochastic, hydro_models).expect("StudySetup::new")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Backward-pass cut coefficient for an anticipated thermal with `lead_stages = 1`
/// in a 2-stage system.
///
/// One anticipated thermal (K=1, cost c_ant) and one regular thermal (cost c_reg) at a
/// single bus; loads D_0, D_1; one one-hour block per stage; max_par_order = 0 so
/// anticipated_state.start = 0. The LP-builder divides every non-theta objective
/// coefficient by COST_SCALE_FACTOR (call it K), so the stored cut lives in scaled units.
///
/// Stage-1 LP (the anticipated decision column d_ant carries scaled cost c_reg/K):
///
/// ```text
///   min  (c_reg/K) gt_reg + (c_reg/K) d_ant + theta
///   s.t. gt_reg + gt_ant = D_1            (load balance)
///        gt_ant - x_state = 0             (fishing, K=1)
///        x_state + d_ant = x_hat          (state-fixing, dual pi)
///        theta >= 0
/// ```
///
/// At the box optimum d_ant = 0, Q_scaled(x_hat) = (c_reg/K)(D_1 - x_hat), so the
/// state-fixing dual is pi = -c_reg/K. With coefficients = dual (no sign flip), the
/// coefficient is -c_reg/K and the intercept is Q_scaled(x_hat) - pi*x_hat = (c_reg/K)*D_1.
#[test]
fn two_stage_k1_anticipated_cut_coefficient_matches_analytical() {
    const K_MAX: usize = FIXTURE_K1.k_max;
    const D_1: f64 = FIXTURE_K1.loads_mw[1];
    const EXPECTED_COEFFICIENT: f64 = FIXTURE_K1.expected_coeff;
    const EXPECTED_INTERCEPT: f64 = C_REG * D_1 / COST_SCALE_FACTOR;

    let system = build_system(&FIXTURE_K1);
    let config = build_config(FIXTURE_K1.iterations);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            FIXTURE_K1.iterations,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error
    );

    let pool0 = &setup.fcf.pools[0];
    let active_count = pool0.active_count();
    assert_eq!(
        active_count, 1,
        "AC-2: stage 0 FCF must contain exactly one active cut; got {active_count}",
    );

    let state = setup.stage_state();
    let ant_state_idx = state.anticipated_state.start;
    assert_eq!(
        state.n_anticipated, 1,
        "fixture must have exactly one anticipated thermal",
    );
    assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
    assert_eq!(
        ant_state_idx, 0,
        "with n_hydros=0 and max_par_order=0, anticipated_state.start must be 0; got {ant_state_idx}",
    );

    let (slot, intercept, coefficients) = setup
        .fcf
        .active_cuts(0)
        .next()
        .expect("AC-3: exactly one active cut must be retrievable from stage 0 pool");
    assert_eq!(
        coefficients.len(),
        state.anticipated_state.end,
        "coefficient slice length must equal n_state",
    );

    let actual_coeff = coefficients[ant_state_idx];
    assert!(
        (actual_coeff - EXPECTED_COEFFICIENT).abs() < TOL,
        "AC-3 / AC-5: cut coefficient at anticipated_state index {ant_state_idx} \
         (slot={slot}, n_state={n_state}) does not match analytical value: \
         actual = {actual_coeff}, expected = {EXPECTED_COEFFICIENT} (= -c_reg/K = -{C_REG}/{COST_SCALE_FACTOR})",
        n_state = coefficients.len(),
    );

    assert!(
        (intercept - EXPECTED_INTERCEPT).abs() < TOL,
        "AC-4: cut intercept does not match analytical value: actual = {intercept}, \
         expected = {EXPECTED_INTERCEPT} (= c_reg * D_1 / K = {C_REG} * {D_1} / {COST_SCALE_FACTOR})",
    );
}

/// Backward-pass cut-coefficient propagation for an anticipated thermal with
/// `lead_stages = 2` in a 3-stage system.
///
/// One anticipated thermal (K=2) and one regular thermal at a single bus; loads
/// D_0, D_1, D_2; one one-hour block per stage; zero seeds; max_par_order = 0 so
/// anticipated_state.start = 0. Fishing rows are emitted at every stage in 0..n_stages.
///
/// The stage-0 FCF cut is generated by backward t=0 (solving stage 1's LP), which carries
/// the FCF cut produced earlier in the same sweep by backward t=1 (solving stage 2). Both
/// stage-1 state-fixing-row duals equal -c_reg/COST_SCALE_FACTOR * BLOCK_HOURS: slot 0 is
/// the same-stage fishing-equality dual (fishing is always active for every anticipated
/// plant); slot 1 flows through the baked stage-1 FCF cut, whose
/// +c_reg/COST_SCALE_FACTOR * BLOCK_HOURS coefficient on x_state slot 1 originates from
/// stage 2's slot-0 fishing dual, routed via the Less-branch ring-buffer shift in
/// state_to_lp_column. So the stored stage-0 cut carries -0.0001 at both state slots.
///
/// iterations = 5: backward t=0 consumes the cut just added to FCF[1] within the same
/// iteration, so the propagated stage-2 subgradient reaches FCF[0] (the slot-1 cut) by
/// iteration 1; the remaining iterations are margin and do not move the asserted cut.
#[test]
fn three_stage_k2_anticipated_cut_coefficient_propagates_correctly() {
    const K_MAX: usize = FIXTURE_K2.k_max;
    const N_ITERATIONS: usize = FIXTURE_K2.iterations;
    const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K2.expected_coeff;
    const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K2.expected_coeff;

    let system = build_system(&FIXTURE_K2);
    let config = build_config(FIXTURE_K2.iterations);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            FIXTURE_K2.iterations,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error,
    );

    // ── AC-2: at least one active cut at stage 0 FCF ──────────────────
    let pool0 = &setup.fcf.pools[0];
    let active_count = pool0.active_count();
    assert!(
        active_count >= 1,
        "AC-2: stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
    );

    // Locate the anticipated_state indices inside the state vector.
    // For n_hydros = 0 and max_par_order = 0 the block starts at 0, with
    // layout `start + slot * n_anticipated + plant`. Here n_anticipated = 1
    // and plant = 0, so slot 0 lives at `start + 0` and slot 1 at `start + 1`.
    let state = setup.stage_state();
    let ant_state_start = state.anticipated_state.start;
    let slot0_idx = ant_state_start; // slot 0, plant 0
    let slot1_idx = ant_state_start + 1; // slot 1, plant 0
    assert_eq!(
        state.n_anticipated, 1,
        "fixture must have exactly one anticipated thermal",
    );
    assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
    assert_eq!(
        ant_state_start, 0,
        "with n_hydros=0 and max_par_order=0, anticipated_state.start must \
         be 0; got {ant_state_start}",
    );

    // ── AC-3 / AC-4: select the iteration-1 cut explicitly.
    // `active_cuts(stage)` yields `(slot, intercept, &[coeffs])` where `slot`
    // encodes `warm_start_count + (iteration - iteration_base) * forward_passes
    // + forward_pass_index` (per CutPool::slot_index). With dense packing
    // (iteration_base = start_iteration + 1 = 1) and forward_passes = 1, the
    // iteration-1 cut lands at slot 0. The analytical match is this FIRST cut:
    // once iteration 1's cut is baked into stage 0's template, the iteration-2
    // forward trial point shifts to a regime where stage 2's subproblem is
    // insensitive to the propagated state (the FCF tangent is exact at the
    // visited point), so iterations 2-5 add zero-subgradient cuts with intercept
    // c_ant*D_1/K = 0.5. The closed-form derivation applies to the iteration-1
    // cut; select it explicitly rather than taking the most-recent one.
    let analytical = setup
        .fcf
        .active_cuts(0)
        .find(|(slot, _, _)| *slot == 0)
        .expect(
            "AC-3: iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool",
        );
    let (slot, _intercept, coefficients) = analytical;

    assert_eq!(
        coefficients.len(),
        state.anticipated_state.end,
        "coefficient slice length must equal n_state (= anticipated_state.end \
         in this no-hydro fixture); got len={}, expected={}",
        coefficients.len(),
        state.anticipated_state.end,
    );

    // ── AC-3: coefficient at slot 1 ─────────────────────────────────────────
    // Expected: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = -0.0001.
    // Source: dual flowing through the baked stage-1 FCF cut, which carries
    // coefficient +c_reg/COST_SCALE*BLOCK_HOURS on x_state[slot=1]_1.
    // The coefficient originates from stage 2's slot-0 fishing dual, routed
    // via the Less-branch ring-buffer shift in state_to_lp_column.
    let actual_coeff_slot1 = coefficients[slot1_idx];
    assert!(
        (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
        "AC-3 / AC-5: stage 0 cut coefficient at anticipated_state slot 1 \
         (state-vector index {slot1_idx}) does not match analytical value: \
         actual = {actual_coeff_slot1}, expected = {EXPECTED_COEFF_SLOT1} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: Less-branch dual flowing through the stage-1 FCF cut \
         (indexer.rs:state_to_lp_column). \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
        n_state = coefficients.len(),
    );

    // ── AC-4: coefficient at slot 0 ─────────────────────────────────────────
    // Expected: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = -0.0001.
    // Source: dual of the same-stage fishing equality at stage 1, which is
    // active because the fishing constraint is always active for every
    // anticipated plant. Both slots carry identical magnitude via different
    // propagation paths.
    let actual_coeff_slot0 = coefficients[slot0_idx];
    assert!(
        (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
        "AC-4 / AC-5: stage 0 cut coefficient at anticipated_state slot 0 \
         (state-vector index {slot0_idx}) does not match analytical value: \
         actual = {actual_coeff_slot0}, expected = {EXPECTED_COEFF_SLOT0} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: same-stage fishing equality dual at stage 1; the fishing \
         constraint is always active for every anticipated plant. \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
        n_state = coefficients.len(),
    );
}

/// Backward-pass cut-coefficient propagation for an anticipated thermal with
/// `lead_stages = 3` in a 4-stage system.
///
/// One anticipated thermal (K=3, cost $50/MWh, max 50 MW), one regular thermal
/// (cost $100/MWh, max 100 MW), loads 5, 10, 15, 30 MW, zero seeds, one-hour blocks.
/// Fishing rows are emitted at every stage in 0..n_stages. All three stage-0 slots receive
/// -c_reg/COST_SCALE_FACTOR via distinct paths:
/// - slot 0: direct fishing dual at stage 1 (solving stage 2);
/// - slot 1: stage-2 fishing dual via one Less-branch shift through stage-1's FCF cut;
/// - slot 2: stage-3 fishing dual via two successive Less-branch shifts (stage-2 then
///   stage-1 FCF cuts), reaching slot 2 at stage 0.
///
/// See state_to_lp_column for the full algebraic chain.
#[test]
fn four_stage_k3_anticipated_cut_coefficient_propagates_correctly() {
    const K_MAX: usize = FIXTURE_K3.k_max;
    const N_ITERATIONS: usize = FIXTURE_K3.iterations;
    const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K3.expected_coeff;
    const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K3.expected_coeff;
    const EXPECTED_COEFF_SLOT2: f64 = FIXTURE_K3.expected_coeff;

    let system = build_system(&FIXTURE_K3);
    let config = build_config(FIXTURE_K3.iterations);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            FIXTURE_K3.iterations,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error,
    );

    let pool0 = &setup.fcf.pools[0];
    let active_count = pool0.active_count();
    assert!(
        active_count >= 1,
        "AC-1: stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
    );

    // Anticipated-state layout is `start + slot * n_anticipated + plant`; with
    // n_anticipated = 1, plant = 0 the slots are consecutive from `start`.
    let state = setup.stage_state();
    let ant_state_start = state.anticipated_state.start;
    let slot0_idx = ant_state_start;
    let slot1_idx = ant_state_start + 1;
    let slot2_idx = ant_state_start + 2;
    assert_eq!(
        state.n_anticipated, 1,
        "fixture must have exactly one anticipated thermal",
    );
    assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
    assert_eq!(
        ant_state_start, 0,
        "with n_hydros=0 and max_par_order=0, anticipated_state.start must \
         be 0; got {ant_state_start}",
    );

    // The analytical match is the iteration-1 cut (slot 0 under dense packing,
    // per CutPool::slot_index): its three-stage propagation chain completes at
    // backward t=0. Later iterations add cuts at trial points with a different
    // active basis.
    let analytical = setup
        .fcf
        .active_cuts(0)
        .find(|(slot, _, _)| *slot == 0)
        .expect("iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool");
    let (_slot, _intercept, coefficients) = analytical;

    assert_eq!(
        coefficients.len(),
        state.anticipated_state.end,
        "coefficient slice length must equal n_state (= anticipated_state.end \
         in this no-hydro fixture); got len={}, expected={}",
        coefficients.len(),
        state.anticipated_state.end,
    );

    let actual_coeff_slot2 = coefficients[slot2_idx];
    assert!(
        (actual_coeff_slot2 - EXPECTED_COEFF_SLOT2).abs() < TOL,
        "AC-2: slot 2 coefficient {actual_coeff_slot2} != {EXPECTED_COEFF_SLOT2} \
         (stage-3 fishing dual via two FCF baked cuts and successive Less-branch shifts)",
    );

    let actual_coeff_slot1 = coefficients[slot1_idx];
    assert!(
        (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
        "AC-3: slot 1 coefficient {actual_coeff_slot1} != {EXPECTED_COEFF_SLOT1} \
         (stage-2 fishing dual via one Less-branch shift through stage-1 FCF cut)",
    );

    let actual_coeff_slot0 = coefficients[slot0_idx];
    assert!(
        (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
        "AC-4: slot 0 coefficient {actual_coeff_slot0} != {EXPECTED_COEFF_SLOT0} \
         (stage-1 fishing equality dual under always-active predicate)",
    );
}
