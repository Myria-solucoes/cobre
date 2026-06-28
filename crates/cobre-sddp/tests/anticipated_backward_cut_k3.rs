//! Analytical verification of backward-pass cut-coefficient propagation for an
//! anticipated thermal with `lead_stages = 3` in a 4-stage system under the
//! Alternative-A layout and the always-active fishing predicate.
//!
//! ## Closed-form derivation
//!
//! Fixture: one anticipated thermal (K=3, cost $50/MWh, max 50 MW),
//! one regular thermal (cost $100/MWh, max 100 MW), loads [5, 10, 15, 30] MW,
//! seeds [0, 0, 0], one-hour blocks per stage.
//!
//! Fishing rows are emitted at every stage [0, n_stages) under the always-active
//! predicate. The slot `K-1 = 2` state-fixing row is pure identity; decision
//! coupling moves to the `anticipated_state_out` definition row
//! (`state_to_lp_column` Equal branch).
//!
//! ## Propagation chain: stage 3 → stage 2 → stage 1 → stage 0
//!
//! All three slots at stage 0 receive coefficient `-c_reg / COST_SCALE_FACTOR`
//! via distinct paths:
//! - **Slot 0**: Direct fishing dual at stage 1 (stage-1 solving stage-2).
//! - **Slot 1**: Stage-2 fishing dual routed via one Less-branch shift through
//!   stage-1's baked FCF cut.
//! - **Slot 2**: Stage-3 fishing dual routed via two successive Less-branch shifts
//!   (stage-2 FCF cut, then stage-1 FCF cut) reaching slot 2 at stage 0.
//!
//! See `state_to_lp_column` for the complete algebraic chain.

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

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
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

// ---------------------------------------------------------------------------
// Numeric fixture (single source of truth).
// ---------------------------------------------------------------------------

const N_STAGES: usize = 4;
const K_MAX: usize = 3;
const BLOCK_HOURS: f64 = 1.0;
const N_ITERATIONS: u32 = 5;

const D_0: f64 = 5.0;
const D_1: f64 = 10.0;
const D_2: f64 = 15.0;
const D_3: f64 = 30.0;
const C_REG: f64 = 100.0;
const C_ANT: f64 = 50.0;
const MAX_GEN_REG: f64 = 100.0;
const MAX_GEN_ANT: f64 = 50.0;

// Duals live in scaled cost units: the LP-builder divides every non-theta
// objective coefficient by COST_SCALE_FACTOR, and cut storage preserves that
// scaling end-to-end (forward.rs consumes them unrescaled).
const COST_SCALE_FACTOR: f64 = 1_000_000.0;

const EXPECTED_COEFF_SLOT2: f64 = -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS;
const EXPECTED_COEFF_SLOT1: f64 = -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS;
const EXPECTED_COEFF_SLOT0: f64 = -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS;
const TOL: f64 = 1e-6;

// `System::build()` sorts thermals by `EntityId::0` ascending; the regular
// thermal must sort before the anticipated one so thermal_idx aligns with the
// bounds table (regular id 2 < anticipated id 5).
const THERMAL_IDX_REG: usize = 0;
const THERMAL_IDX_ANT: usize = 1;
const REGULAR_ID: EntityId = EntityId(2);
const ANTICIPATED_ID: EntityId = EntityId(5);

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator (per-file copy: test independence over a
// shared module).
// ---------------------------------------------------------------------------

struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        // Deficit cost set safely above c_reg so the LP never prefers shedding load.
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let thermal_reg = Thermal {
        id: REGULAR_ID,
        name: "T_reg".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_REG,
        cost_per_mwh: C_REG,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let thermal_ant = Thermal {
        id: ANTICIPATED_ID,
        name: "T_ant".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_ANT,
        cost_per_mwh: C_ANT,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 3 }),
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

    let stages: Vec<Stage> = (0..N_STAGES)
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

    let load_models: Vec<LoadModel> = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: match i {
                0 => D_0,
                1 => D_1,
                2 => D_2,
                _ => D_3,
            },
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
            n_stages: N_STAGES,
            k_max: K_MAX,
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

    // K-padded axis: `fill_anticipated_columns` reads delivery cells at
    // stage_idx + K_i, so overrides must cover the n_stages + k_max range.
    let thermal_axis = N_STAGES + K_MAX;
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(THERMAL_IDX_REG, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GEN_REG,
            cost_per_mwh: C_REG,
        };
        *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GEN_ANT,
            cost_per_mwh: C_ANT,
        };
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
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

    // Zero seeds so any propagated cut coefficient is attributable to in-study
    // decisions, not the ring-buffer history.
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: ANTICIPATED_ID,
            values_mw: vec![0.0, 0.0, 0.0],
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

fn build_config() -> Config {
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
                limit: N_ITERATIONS,
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

#[test]
fn four_stage_k3_anticipated_cut_coefficient_propagates_correctly() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(
            &mut solver,
            &comm,
            N_ITERATIONS as usize,
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
