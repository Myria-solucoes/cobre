//! Regression test: `d_t` must commit to load level for every active
//! anticipated-decision stage in a K=2 fixture.
//!
//! ## Bug being guarded against
//!
//! In a 6-stage K=2 study with a 500x cost asymmetry (anticipated thermal at
//! $10/MWh vs backup at $5000/MWh), the LP should commit the anticipated
//! thermal to exactly `load = 150 MW` at every stage `t` where `t + K < n_stages`
//! (i.e. `t in {0, 1, 2, 3}`). Over-committing to `max_gen = 200 MW` costs an
//! extra 50 MW × $10/MWh × 744 h = $372k/stage with zero offsetting benefit
//! (excess generation is free), so the LP optimum is `d_t = load = 150 MW`,
//! not `max_gen = 200 MW`.
//!
//! Empirical inspection on pre-fix HEAD shows that `d_0 = 150` (approximately
//! correct) but `d_1 = d_2 = d_3 = 0` (wrong). The LP is discarding the benefit
//! at all intermediate active stages. This is the symptom of the cut-coefficient
//! corruption in `state_to_lp_column`'s `Less` branch: the Benders cut
//! gradients for anticipated-state columns are mapped to the wrong LP column
//! index for `k >= 2`, so the policy at those stages receives no incentive
//! to commit anticipated capacity.
//!
//! ## What this test asserts
//!
//! After training a deterministic single-scenario simulation for 10 iterations:
//!
//! - For `t in {0, 1, 2, 3}`: `anticipated_decision_mw` exists (is `Some`) and
//!   commits to load level `150.0 ± 1e-3 MW`.
//! - For `t in {4, 5}`: `anticipated_decision_mw` is `None` (the
//!   strict-boundary predicate `t + K < n_stages` excludes these).
//!
//! ## Economic reasoning
//!
//! Load = 150 MW < max_gen = 200 MW. Excess generation cost = $0.
//! LP optimum: `d_t = load = 150 MW` at every active stage.
//! Committing 200 MW instead would cost an extra 50 MW × $10/MWh × 744 h
//! = $372k/stage with no benefit — the excess is simply discarded for free.
//! The 500x cost asymmetry matters only to ensure the anticipated thermal
//! dispatches at all (reaching load level), not to saturate at max_gen.

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
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator for testing
// ---------------------------------------------------------------------------

/// Single-rank communicator stub for testing.
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

/// Build a 6-stage K=2 system with:
/// - 1 bus (deficit cost 5000 $/MWh, excess cost $0)
/// - 1 trivial hydro (1 hm³ max storage, zero inflow, max_gen 1 MW) — keeps
///   the model in the thermal regime without adding a hydro state variable
///   that complicates interpretation.
/// - 1 anticipated thermal (K=2, cost 10 $/MWh, max 200 MW) — id=2
/// - 1 backup thermal (cost 5000 $/MWh, max 500 MW) — id=4
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments = [(id=2, [0.0, 0.0])]` — zero seeds so
///   the test isolates the in-horizon bug, not any seeding artefact.
///
/// The LP optimum is `d_t = load = 150 MW` at every active stage. Committing
/// more (200 MW) costs extra at $10/MWh with no benefit — excess is free.
/// The 500x cost ratio (10 vs 5000) ensures the anticipated thermal reaches
/// load level, not that it saturates at max_gen.
fn build_system_k2() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 2;
    let n_stages: usize = 6;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 5000.0,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K=2 lead stages, very cheap so the policy
    // dispatches to load level. Max 200 MW > load 150 MW, but d_t = 150
    // is the optimum since over-commitment costs $10/MWh with zero benefit.
    let anticipated_id = EntityId(2);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant".to_string(),
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

    // Backup thermal: very expensive so the LP avoids it wherever anticipated
    // dispatch is available.
    let thermal_backup = Thermal {
        id: EntityId(4),
        name: "T_backup".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 5000.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Trivial hydro: 1 hm³ cap, zero inflow, max_gen 1 MW. Present to
    // satisfy `n_hydros = 1` requirements of `ResolvedBounds` while keeping
    // the system firmly in the thermal regime.
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
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

    // Zero inflow — keeps the model deterministic and purely thermal.
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
            filling_inflow_m3s: 0.0,
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

    // Build resolved bounds with default values, then apply per-thermal
    // overrides. `ResolvedBounds::new` accepts a single default for ALL
    // thermals; the per-thermal costs must be patched afterwards so the
    // objective properly distinguishes the cheap anticipated thermal from
    // the expensive backup.
    //
    // The padding region `[n_stages, n_stages + k)` is the delivery-stage
    // axis read by `fill_anticipated_decision_objective`; it must also
    // carry the per-thermal cost so the decision column's objective
    // coefficient is non-zero.
    let thermal_axis = n_stages + k;
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
    for s in 0..thermal_axis {
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // anticipated: cheap
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 200.0;
        bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // backup: expensive
        bounds.thermal_bounds_mut(1, s).max_generation_mw = 500.0;
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

    // Zero seeds: isolates the in-horizon bug. Pre-horizon commitments play
    // no role here — what we are testing is whether the cut coefficients
    // correctly signal the value of anticipated dispatch at stages t >= 1.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![0.0, 0.0],
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
        .expect("build_system_k2: valid system")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a [`Config`] for 10-iteration training and 1-scenario deterministic
/// simulation.
///
/// Ten iterations is sufficient to expose the bug: with a 500x cost ratio,
/// any policy that even partially explores the anticipated commitment creates
/// cuts whose gradients at the anticipated-state columns reveal the
/// corruption at `t >= 1`.
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
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
        energy: cobre_io::EnergyConfig::default(),
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
    .expect("build_stochastic_context: must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    StudySetup::new(&system, config, stochastic, hydro_models)
        .expect("StudySetup::new: must succeed")
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Assert that `anticipated_decision_mw` commits to load level (150 MW) for
/// every active decision stage in a K=2, 6-stage fixture.
///
/// With load = 150 MW < max_gen = 200 MW and excess generation cost = $0,
/// the LP optimum is `d_t = load = 150 MW` at every active stage `t` where
/// `t + K < n_stages`. Committing 200 MW instead costs an extra
/// 50 MW × $10/MWh × 744 h = $372k/stage with no benefit.
///
/// **Expected behaviour (post-fix)**: `d_t ≈ 150.0` for `t in {0, 1, 2, 3}`.
///
/// **Pre-fix behaviour (historical context)**: `d_0 ≈ 150.0` (approximately
/// correct at stage 0) but `d_1 = d_2 = d_3 ≈ 0`. The cut coefficients for
/// the anticipated-state columns are corrupted for `k >= 2` in
/// `state_to_lp_column`'s `Less` branch, so the policy at those stages
/// receives no incentive to commit anticipated capacity, forcing backup to
/// dispatch at $5000/MWh.
#[test]
fn d_t_commits_to_load_for_every_active_stage_k2() {
    let k: usize = 2;
    let n_stages: usize = 6;
    // Active decision stages: t + K < n_stages  =>  t in {0, 1, 2, 3}.
    let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
    // Inactive (boundary) stages where the strict predicate excludes decision.
    let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();

    let system = build_system_k2();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new: must succeed");

    // Train the policy for 10 iterations.
    let outcome = setup
        .train(&mut solver, &comm, 10, HighsSolver::new, None, None)
        .expect("training error: train() must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: training returned an error: {:?}",
        outcome.error,
    );

    // Run a single deterministic simulation to observe the policy decisions.
    let mut pool = setup
        .create_workspace_pool(&comm, 1, HighsSolver::new)
        .expect("workspace pool error: create_workspace_pool must succeed");
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
        .expect("simulation error: simulate() must not return Err");
    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");

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

    // The anticipated thermal has entity id=2 (see build_system_k2).
    let anticipated_thermal_id: i32 = 2;
    let decision_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_decision_mw)
    };

    // ── Active stages: decision must exist and commit to load = 150 MW ──
    //
    // With the strict-boundary predicate `t + K < n_stages`, stages 0..3
    // are active. Load = 150 MW < max_gen = 200 MW; excess is free, so
    // d_t = load = 150 MW is the optimum — over-committing costs extra with
    // no benefit. On pre-fix code, the cut-coefficient bug causes d_t = 0
    // for t in {1, 2, 3}, which this test guards against post-fix.
    let load_mw = 150.0_f64;
    let tol = 1e-3_f64;
    for t in &active_stages {
        let d_t = decision_at(*t).unwrap_or_else(|| {
            panic!("anticipated_decision_mw must be Some at active stage t={t} (t + K < n_stages)")
        });
        assert!(
            (d_t - load_mw).abs() < tol,
            "d_t at stage {t} must saturate at load=150 MW: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Pre-fix behaviour: d_t ≈ 0 for t >= 1 due to cut-coefficient \
             corruption in state_to_lp_column (Less branch), forcing backup \
             at $5000/MWh.",
            delta = (d_t - load_mw).abs(),
        );
    }

    // ── Inactive stages: decision must be None (strict-boundary predicate) ──
    //
    // For t in {4, 5}: t + K >= n_stages, so the anticipated-decision
    // variable does not exist in the LP. The simulation must report None.
    for t in &inactive_stages {
        assert!(
            decision_at(*t).is_none(),
            "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages; strict-boundary predicate excludes this stage)",
        );
    }
}
