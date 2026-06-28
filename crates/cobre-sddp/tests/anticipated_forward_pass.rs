//! End-to-end integration test verifying the anticipated-state ring-buffer
//! evolution across the full forward pass for a 5-stage K=2 system.
//!
//! Ring-buffer shift semantics: at the end of stage `t`, `shift_anticipated_state`
//! shifts each plant's slots down (`slot[s] <- incoming[s+1]`) and writes the new
//! decision into the highest slot (`slot[K-1] <- decision_primal`), so slot 0 at
//! stage `t+1` equals slot 1 at stage `t`.

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
    hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    thermal::{AnticipatedConfig, Thermal},
};
use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, HydroStorage,
    InitialConditions, LineStageBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
    ThermalStageBounds,
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
// StubComm — single-rank communicator for testing
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

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build a 5-stage system with one anticipated thermal (K=2, seeded
/// `[100.0, 50.0]`) and one backup thermal that alone covers the 150 MW load, so
/// the LP is always feasible.
fn build_system() -> cobre_core::System {
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

    let anticipated_id = EntityId(2);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let thermal_backup = Thermal {
        id: EntityId(4),
        name: "T_backup".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 500.0,
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

    let n_stages = 5_usize;
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
            mean_mw: 150.0,
            std_mw: 0.0,
        })
        .collect();

    let k_max: usize = 2;
    let n_st = n_stages;

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
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
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

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![100.0, 50.0],
        }],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant, thermal_backup])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
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

/// Construct a [`StudySetup`] in-process, building the stochastic context
/// directly so the test stays hermetic (no external scenario files).
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
// Integration test
// ---------------------------------------------------------------------------

/// Verify that the anticipated-state ring buffer evolves correctly across all
/// 5 stages of the forward pass.
///
/// The block layout is slot-major, plant-minor; with `n_anticipated=1`,
/// `k_max=2` it is `[slot0_plant0, slot1_plant0]`. `state_at_capture` is filled
/// by two paths, which is why slots 0 and 1 carry the same values:
///
/// - **Stage 0**: forward pass writes the post-shift outgoing state of stage 0.
/// - **Stages 1..=4**: backward pass stores the forward outgoing of stage `t-1`
///   as its trial point `x_hat` — so `basis_cache[1]` (trial point of stage 1)
///   equals `basis_cache[0]` (forward outgoing of stage 0).
#[test]
fn five_stage_k2_anticipated_state_ring_buffer_evolution() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    // One iteration populates basis_cache for every stage.
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "AC-1: training error must be None; got {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations >= 1,
        "AC-1: at least 1 iteration must complete; got {}",
        outcome.result.iterations
    );

    let state = setup.stage_state();
    let n_ant = state.n_anticipated;
    let k_max = state.k_max;
    assert_eq!(n_ant, 1, "fixture must have exactly 1 anticipated thermal");
    assert_eq!(k_max, 2, "fixture must have k_max = 2");

    let ant_start = state.anticipated_state.start;

    let basis_cache = &outcome.result.basis_cache;
    assert_eq!(
        basis_cache.len(),
        5,
        "basis_cache must have one entry per study stage"
    );

    // AC-2: forward pass stores the post-shift outgoing state. Seed [100.0, 50.0]
    // shifts to slot 0 = 50.0 (no other code path produces this exact value),
    // slot 1 = LP decision d_0 ∈ [0, 100].
    let s0 = basis_cache[0]
        .as_ref()
        .expect("AC-2: stage 0 basis must be captured")
        .state_at_capture
        .as_slice();
    assert!(
        (s0[ant_start] - 50.0).abs() < 1e-9,
        "AC-2: stage 0 slot 0 must equal seeded slot 1 = 50.0; got {}",
        s0[ant_start]
    );
    assert!(
        (-0.01..=100.01).contains(&s0[ant_start + 1]),
        "AC-2: stage 0 slot 1 (= d_0) must lie in [0, 100]; got {}",
        s0[ant_start + 1]
    );

    // AC-3: backward pass for stage 1 stores the forward outgoing of stage 0 as
    // its trial point x_hat, so basis_cache[1] equals basis_cache[0].
    let s1 = basis_cache[1]
        .as_ref()
        .expect("AC-3: stage 1 basis must be captured")
        .state_at_capture
        .as_slice();
    assert!(
        (s1[ant_start] - s0[ant_start]).abs() < 1e-9,
        "AC-3: stage 1 slot 0 ({}) must equal stage 0 slot 0 ({}) — both carry \
         the post-shift outgoing state of forward stage 0",
        s1[ant_start],
        s0[ant_start],
    );
    assert!(
        (s1[ant_start + 1] - s0[ant_start + 1]).abs() < 1e-9,
        "AC-3: stage 1 slot 1 ({}) must equal stage 0 slot 1 ({}) — both carry \
         d_0 from the forward pass",
        s1[ant_start + 1],
        s0[ant_start + 1],
    );

    // AC-4: for t≥2, basis_cache[t] holds the forward outgoing of stage t-1, so
    // the forward shift gives s_curr slot 0 == s_prev slot 1.
    for t in 2..5_usize {
        let s_curr = basis_cache[t]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-4: stage {t} basis must be captured"))
            .state_at_capture
            .as_slice();
        let s_prev = basis_cache[t - 1]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-4: stage {} basis must be captured", t - 1))
            .state_at_capture
            .as_slice();

        assert!(
            (s_curr[ant_start] - s_prev[ant_start + 1]).abs() < 1e-9,
            "AC-4: stage {t} slot 0 ({}) must equal stage {} slot 1 ({})",
            s_curr[ant_start],
            t - 1,
            s_prev[ant_start + 1],
        );
    }

    // AC-5: every captured slot 1 (the anticipated decision) stays within the
    // thermal's dispatch bounds [0.0, 100.0].
    for t in 0..5_usize {
        let s_t = basis_cache[t]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-5: stage {t} basis must be captured"))
            .state_at_capture
            .as_slice();
        let decision = s_t[ant_start + 1];
        assert!(
            (-0.01..=100.01).contains(&decision),
            "AC-5: stage {t} slot 1 must lie in [0, 100]; got {decision}",
        );
    }
}
