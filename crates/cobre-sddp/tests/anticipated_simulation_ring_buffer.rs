//! Regression: the simulation pipeline (`solve_simulation_stage`) must call
//! `shift_anticipated_state` once per stage, like the inflow-lag ring buffer.
//!
//! Without the shift, the next stage's `ws.current_state` carries the post-solve
//! primal of the unbounded `anticipated_state` columns — `incoming - decision`,
//! the residual the Category 6 fixing rows leave — instead of the shifted
//! ring-buffer state, so the fishing constraint at delivery stage `K` reads a
//! never-advanced slot 0. The contract this pins: with the anticipated thermal
//! cheaper than the backup, the matured commitment equals the in-study decision
//! made `K` stages earlier,
//!
//! `anticipated_committed_mw(t = K) == anticipated_decision_mw(t = 0)`,
//!
//! not the seeded `past_anticipated_commitments` a missing shift would surface.

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

use std::sync::mpsc;

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

/// Build a deterministic K-stage system: one bus, one mostly-inactive hydro, a
/// cheap anticipated thermal (id 2) plus an expensive backup (id 4), constant
/// 150 MW load, and `past_anticipated_commitments` seeded with `past_commitments_mw`.
///
/// The non-zero seed is intentional: building the resolved `System` directly via
/// `SystemBuilder::new()` bypasses the `cobre-io` parse-and-validate pipeline, so
/// the semantic validator that rejects non-zero `values_mw` does not fire — that
/// rule applies to JSON through `load_case`, not to directly-constructed `System`.
fn build_system(k: usize, past_commitments_mw: Vec<f64>, n_stages: usize) -> cobre_core::System {
    use chrono::NaiveDate;

    assert_eq!(
        past_commitments_mw.len(),
        k,
        "past_commitments_mw length must equal K (lead_stages)",
    );

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 5000.0,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K lead stages, very cheap so the optimal policy
    // saturates anticipated dispatch at the load level.
    let anticipated_id = EntityId(2);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 10.0,
        anticipated_config: Some(AnticipatedConfig {
            lead_stages: k as u32,
        }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Backup thermal: very expensive so the LP prefers anticipated dispatch
    // whenever possible.
    let thermal_backup = Thermal {
        id: EntityId(4),
        name: "T_backup".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 5000.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Hydro: small to keep the model deterministic in the thermal regime.
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        downstream_id: None,
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
            mean_m3s: 0.0,
            std_m3s: 0.0,
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

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
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

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
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
    // ResolvedBounds::new applies one default to ALL thermals; without these
    // per-thermal overrides the cheap anticipated (index 0) and expensive backup
    // (index 1) are indistinguishable and decision_at(t) collapses to zero,
    // neutering the regression. The override must span the padding region
    // `[n_stages, n_stages + k_max)` too — `fill_anticipated_columns` reads the
    // delivery-stage axis there, so the decision column needs its cost out to K.
    let thermal_axis = n_stages + k;
    for s in 0..thermal_axis {
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // anticipated
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 100.0;
        bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // backup
        bounds.thermal_bounds_mut(1, s).max_generation_mw = 200.0;
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
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

    // Seed the anticipated ring buffer with NON-ZERO past commitments. This
    // is the lever that exposes the bug: if the simulation pipeline failed
    // to shift the buffer, slot 0 at stage K would still report the seed
    // values rather than the in-study decision.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: past_commitments_mw,
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

/// Build a [`Config`] for training + 1-scenario deterministic simulation.
fn build_config(training_iters: u32) -> Config {
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
                limit: training_iters,
            }]),
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig {
            enabled: true,
            num_scenarios: 1,
            io_channel_capacity: 8,
            ..IoSimulationConfig::default()
        },
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
// Test
// ---------------------------------------------------------------------------

/// K=1 case of the module-doc invariant: stage-1 matured commitment equals the
/// stage-0 decision, not the non-zero seed a missing ring-buffer shift surfaces.
#[test]
fn simulation_ring_buffer_shifts_anticipated_state_k1() {
    let k: usize = 1;
    let n_stages: usize = 5;
    // Seed (7 MW) has no relationship to load/bounds, so the LP will not pick it
    // as a decision — a seed equal to a plausible decision would let the buggy and
    // fixed paths agree and neuter the test.
    let seed: Vec<f64> = vec![7.0];

    let system = build_system(k, seed.clone(), n_stages);
    let config = build_config(50);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 50, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let sim_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            None,
            &outcome.result.basis_cache,
        )
        .expect("simulate must not return Err");

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(
        sim_run.costs.len(),
        1,
        "simulation must produce exactly one scenario cost",
    );
    assert_eq!(
        scenario_results.len(),
        1,
        "simulation must stream exactly one scenario result",
    );

    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        n_stages,
        "scenario must contain one stage record per study stage",
    );

    let anticipated_thermal_id: i32 = 2;
    let decision_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_decision_mw)
    };
    let committed_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_committed_mw)
    };

    // Under always-active fishing, stage-0 committed reads the seed at slot 0.
    let d0 =
        decision_at(0).expect("anticipated_decision_mw must exist at stage 0 (t + K < n_stages)");
    let c0 = committed_at(0)
        .expect("anticipated_committed_mw must be Some at stage 0 under always-active fishing");
    assert!(
        (c0 - 7.0).abs() < 1e-6,
        "committed_at(0) must equal the K=1 seed values_mw[0]=7.0; got {c0}",
    );
    assert!(
        d0.abs() > 1e-6,
        "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
    );

    let c1 = committed_at(1).expect("committed at stage 1 must exist (K <= 1)");
    assert!(
        (c1 - d0).abs() < 1e-6,
        "REGRESSION (ring-buffer shift): stage 1 committed ({c1}) must equal \
         stage 0 decision ({d0}). On the buggy code path the ring buffer was \
         never shifted in simulation, so stage 1's Cat 6 RHS carried the \
         residual `seed - d_0` (negative when d_0 > seed) and the LP was \
         infeasible. With the shift, Cat 6 RHS = d_0 and gt_anticipated at \
         stage 1 saturates at d_0 (cost zeroed at delivery).",
    );
}

/// K=2 case of the module-doc invariant: the two pre-horizon stages read the seed
/// slots, and from stage 2 the matured commitment equals the decision made K=2
/// stages earlier (two shifts carry it into slot 0), never the seed, never zero.
#[test]
fn simulation_ring_buffer_shifts_anticipated_state_k2() {
    let k: usize = 2;
    let n_stages: usize = 6;
    // Seed slots are distinct from d_0 (which saturates near the thermal max of
    // 100), so neither slot can coincide with a decision and mask a missing shift.
    let seed: Vec<f64> = vec![50.0, 30.0];

    let system = build_system(k, seed.clone(), n_stages);
    let config = build_config(10);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("workspace pool must build");
    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

    let _sim_run = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            None,
            &outcome.result.basis_cache,
        )
        .expect("simulate must not return Err");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

    assert_eq!(scenario_results.len(), 1);
    let scenario = &scenario_results[0];
    assert_eq!(scenario.stages.len(), n_stages);

    let anticipated_thermal_id: i32 = 2;
    let decision_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_decision_mw)
    };
    let committed_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_committed_mw)
    };

    // Pre-horizon: stage 0 reads seed slot 0; stage 1 reads slot 1 after the
    // stage-0 shift moves it into slot 0.
    let c0 =
        committed_at(0).expect("committed_at(0) must be Some under always-active fishing with K=2");
    assert!(
        (c0 - 50.0).abs() < 1e-6,
        "committed_at(0) must equal K=2 seed values_mw[0]=50.0; got {c0}",
    );
    let c1 =
        committed_at(1).expect("committed_at(1) must be Some under always-active fishing with K=2");
    assert!(
        (c1 - 30.0).abs() < 1e-6,
        "committed_at(1) must equal K=2 seed values_mw[1]=30.0 (shifted to slot 0); got {c1}",
    );

    let d0 = decision_at(0).expect("decision at stage 0 must exist (0 + K < n_stages)");

    assert!(
        d0.abs() > 1e-6,
        "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
    );

    let c2 = committed_at(2).expect("committed at stage 2 must exist (K <= 2)");
    assert!(
        (c2 - d0).abs() < 1e-6,
        "REGRESSION (ring-buffer shift, K=2): stage 2 committed ({c2}) must \
         equal stage 0 decision ({d0}). On the buggy code path the ring \
         buffer was never shifted in simulation, so stage 2's Cat 6 RHS \
         carried a stale residual instead of the d_0 that the two shifts \
         (end of stage 0, end of stage 1) propagated into slot 0.",
    );
}
