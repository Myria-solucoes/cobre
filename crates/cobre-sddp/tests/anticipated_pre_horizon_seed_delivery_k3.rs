//! Integration test: pre-horizon seed delivery across three pre-horizon stages
//! with K=3 and the always-active fishing predicate.
//!
//! ## What this test asserts
//!
//! With `K = 3`, `n_stages = 6`, a single anticipated thermal (id=52), and
//! `past_anticipated_commitments.values_mw = [50.0, 30.0, 10.0]`, the LP must:
//!
//! 1. Deliver `committed_at(0) == 50.0 MW` — the always-active fishing
//!    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
//!    buffer, which holds the 50.0 MW seed (`values_mw[0]`). The cost-zeroing
//!    predicate zeros the per-block objective for this column so the LP
//!    accepts the delivery at zero additional cost.
//!
//! 2. Deliver `committed_at(1) == 30.0 MW` — `shift_anticipated_state`
//!    (`noise.rs:253`) moves slot 1 (`values_mw[1] = 30.0`) into slot 0 at
//!    the start of stage 1. Stage 1's always-active fishing equality then
//!    reads slot 0 = 30.0 MW. This is one of the two K=3-specific assertions
//!    that the K=1 and K=2 delivery tests cannot reach: K=3 has three
//!    pre-horizon stages, with two ring-buffer shifts between them.
//!
//! 3. Deliver `committed_at(2) == 10.0 MW` — after two ring-buffer shifts,
//!    slot 0 holds `values_mw[2] = 10.0`. Stage 2's always-active fishing
//!    equality delivers it at zero LP cost. This is the deepest pre-horizon
//!    delivery assertion in the entire anticipated test suite.
//!
//! 4. Satisfy `committed_at(t) ≈ decision_at(t-3)` for t ∈ {3, 4, 5} — the
//!    K=3 ring-buffer matures decisions three stages after they are committed.
//!    With K=3, the decision written at stage t occupies slot `K-1 = 2` in
//!    the outgoing state, which shifts into slot 0 after three forward steps,
//!    at which point the fishing equality delivers it. This is the t-3 offset
//!    (compare: K=1 uses t-1, K=2 uses t-2).
//!
//! 5. Saturate `decision_at(t) > 0` and `≤ max_gen + 1e-6` for t ∈ {0, 1, 2}
//!    (stages where `t + K < n_stages`, i.e., `t + 3 < 6`, giving t ∈ {0,1,2})
//!    — the anticipated thermal costs $10/MWh vs the backup's $5000/MWh, and
//!    the per-block cost of the decision column is non-zero only at the
//!    decision stage, so the LP commits a non-trivial amount to avoid future
//!    backup dispatch.
//!
//! 6. Satisfy the analytical cost bound:
//!    - Stage 0: seed delivers 50 MW; backup covers 100 MW
//!      × $5000/MWh × 744 h = $372,000,000.
//!    - Stage 1: shifted seed delivers 30 MW; backup covers 120 MW
//!      × $5000/MWh × 744 h = $446,400,000.
//!    - Stage 2: doubly-shifted seed delivers 10 MW; backup covers 140 MW
//!      × $5000/MWh × 744 h = $520,800,000.
//!    - Stages 3–5 delivery: anticipated delivers ≥ 150 MW load (zeroed cost).
//!    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
//!    - Tolerance: $1,000.
//!    - Total upper bound: $1,343,665,000.
//!
//! ## Legacy behaviour
//!
//! The old predicate `K_i > stage_idx` had fishing active at stages 0, 1, and 2.
//! However, the ring-buffer shifts between three pre-horizon stages were not
//! explicitly exercised. This test closes that gap for the K=3 case, including
//! both shifts: stages 0→1 and 1→2.
//!
//! ## Entity IDs
//!
//! IDs 51 (hydro), 52 (anticipated thermal), 53 (backup thermal) are chosen
//! to be distinct from all existing anticipated tests:
//! - `anticipated_simulation_ring_buffer.rs` uses IDs 2/3/4
//! - `anticipated_numerical_reconciliation_k2.rs` uses IDs 5/6/7
//! - `anticipated_d_t_saturation_k2.rs` uses IDs 2/3/4
//! - `anticipated_d_t_saturation_k3.rs` uses IDs 3/4/5
//! - `anticipated_pre_horizon_seed_delivery_k1.rs` uses IDs 30/31/32
//! - `anticipated_pre_horizon_seed_delivery_k2.rs` uses IDs 41/42/43
//! - `anticipated_backward_cut_k3.rs` uses IDs 2/5
//!
//! The 50-series ensures no cross-test entity confusion in nextest.

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
    Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
    StageRiskConfig, StageStateConfig,
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
// Analytical cost bound constants (documented in module doc comment above)
// ---------------------------------------------------------------------------

/// Stage-0 backup cost: backup carries 150 - 50 = 100 MW at $5000/MWh × 744 h.
/// The seed (50 MW) is delivered by the anticipated thermal at zero LP cost
/// (cost-zeroing predicate). Only the remaining 100 MW requires backup.
const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0; // $372,000,000

/// Stage-1 backup cost: after the first ring-buffer shift, slot 0 holds 30 MW.
/// Backup carries 150 - 30 = 120 MW at $5000/MWh × 744 h = $446,400,000.
/// The shifted seed (30 MW) is delivered at zero LP cost via cost-zeroing.
const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 30.0) * 744.0 * 5000.0; // $446,400,000

/// Stage-2 backup cost: after two ring-buffer shifts, slot 0 holds 10 MW.
/// Backup carries 150 - 10 = 140 MW at $5000/MWh × 744 h = $520,800,000.
/// The doubly-shifted seed (10 MW) is delivered at zero LP cost via cost-zeroing.
/// This is the deepest pre-horizon delivery in the test suite.
const STAGE_2_BACKUP_COST_USD: f64 = (150.0 - 10.0) * 744.0 * 5000.0; // $520,800,000

/// Upper bound on active-decision cost: 3 decision stages (t ∈ {0,1,2},
/// where t + K < n_stages ⟺ t + 3 < 6) × 200 MW (max_gen) × $10/MWh × 744 h.
/// The LP may commit less than max_gen after 1 iteration, but will not exceed
/// max_gen.
const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0; // $4,464,000

/// Absolute cost tolerance matching `anticipated_numerical_reconciliation_k2`.
const COST_TOLERANCE_USD: f64 = 1_000.0;

/// Total upper bound: stage-0 backup + stage-1 backup + stage-2 backup +
/// decision cost ceiling + tolerance.
const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
    + STAGE_1_BACKUP_COST_USD
    + STAGE_2_BACKUP_COST_USD
    + MAX_DECISION_COST_USD
    + COST_TOLERANCE_USD;

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

fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 3;
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

    // Anticipated thermal: K=3 lead stages, very cheap so the LP saturates
    // commitment at max_gen = 200 MW at every active decision stage.
    let anticipated_id = EntityId(52);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant_seed_k3".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 10.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 3 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Backup thermal: expensive so the LP uses it only when anticipated
    // capacity is unavailable (stage 0: 50 MW seed → backup 100 MW;
    // stage 1: 30 MW shifted seed → backup 120 MW;
    // stage 2: 10 MW doubly-shifted seed → backup 140 MW).
    let thermal_backup = Thermal {
        id: EntityId(53),
        name: "T_backup_seed_k3".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 5000.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Trivial hydro: present to satisfy `n_hydros = 1` in ResolvedBounds;
    // zero inflow and 1 MW max_gen keep the system firmly in the thermal
    // regime.
    let hydro = Hydro {
        id: EntityId(51),
        name: "H1_seed_k3".to_string(),
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

    // Zero inflow keeps the model deterministic.
    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(51),
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

    // The padding region [n_stages, n_stages + k) = [6, 9) is the delivery-stage
    // axis read by `fill_anticipated_columns`; it must carry per-thermal
    // costs so the decision column's objective coefficient is non-zero. Without
    // the padding, decision cost defaults to 0 and the LP commits everywhere,
    // masking the AC5 regression assertion.
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
    // Thermal index 0 = anticipated (id=52): cheap at $10/MWh, max 200 MW.
    // Thermal index 1 = backup (id=53): expensive at $5000/MWh, max 500 MW.
    // These per-thermal overrides ensure the LP distinguishes them. Without
    // the override the LP has no cost incentive to commit anticipated capacity
    // and decision_at(t) collapses to zero, masking the regression assertion.
    // The thermal_axis range [0, n_stages + k) = [0, 9) covers the delivery-stage
    // padding; omitting the padding causes the decision cost coefficient to
    // default to 0 and the LP commits everywhere, hiding the AC5 regression.
    for s in 0..thermal_axis {
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 200.0;
        bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
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

    // Seed the anticipated ring buffer with [50.0, 30.0, 10.0] MW.
    //
    // With the always-active fishing predicate, stage 0 reads slot 0 (= 50.0 MW)
    // via the fishing equality and delivers it at zero LP cost (cost-zeroing
    // predicate). Stage 1 reads slot 0 after the first ring-buffer shift
    // (`noise.rs:253 shift_anticipated_state`) moves slot 1 (= 30.0 MW) into
    // slot 0; stage 1's fishing equality delivers 30.0 MW at zero cost.
    // Stage 2 reads slot 0 after the second ring-buffer shift moves slot 1
    // (now holding 10.0 MW, having been shifted from slot 2) into slot 0;
    // stage 2's fishing equality delivers 10.0 MW at zero cost.
    //
    // Distinct values (50.0 vs 30.0 vs 10.0) catch slot-swap bugs that
    // identical values would silently mask across two pre-horizon shifts.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(51),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![50.0, 30.0, 10.0],
        }],
        recent_observations: vec![],
    };

    // discount_rate = 0.0 so all discount factors are 1.0 — the analytical
    // cost derivation is exact.
    let policy_graph = PolicyGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![],
        season_map: None,
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
        .policy_graph(policy_graph)
        .build()
        .expect("build_system: valid system")
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 5 }]),
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

#[test]
fn pre_horizon_seed_delivers_three_pre_horizon_stages_k3() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    // Step 1: train for 5 iterations so cuts sharpen and stage-2 decisions reach max_gen.
    // (This covers stage-5 delivery at zero backup cost, satisfying AC6 bound.)
    let outcome = setup
        .train(&mut solver, &comm, 5, ActiveSolver::new, None, None)
        .expect("train: must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    // Step 2: run a single deterministic simulation.
    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool: must succeed");
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
        .expect("simulate: must not return Err");

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
        6,
        "scenario must contain one record per study stage (n_stages=6)",
    );

    // Step 3: define accessors for the anticipated thermal (id=52).
    let anticipated_thermal_id: i32 = 52;
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

    // ── AC1: committed_at(0) == Some(50.0) within 1e-6 MW ──────────────────
    let c0 = committed_at(0).expect(
        "AC1 FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 50 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
    );
    assert!(
        (c0 - 50.0).abs() < 1e-6,
        "AC1 FAIL: committed_at(0) = {c0} MW, expected 50.0 MW (values_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 50.0 MW of the ring buffer.",
    );

    // ── AC2: committed_at(1) == Some(30.0) within 1e-6 MW ──────────────────
    // (First ring-buffer shift: slot 1 → slot 0 between pre-horizon stages 0→1)
    let c1 = committed_at(1).expect(
        "AC2 FAIL: committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
    assert!(
        (c1 - 30.0).abs() < 1e-6,
        "AC2 FAIL: committed_at(1) = {c1} MW, expected 30.0 MW (values_mw[1]). \
         `shift_anticipated_state` (noise.rs:253) must move slot 1 (30.0 MW) \
         into slot 0 at the start of stage 1, and the fishing equality must read \
         that value. If the result is 50.0 MW, the first ring-buffer shift is not \
         moving slot 1 into slot 0 between pre-horizon stages 0 and 1.",
    );

    // ── AC3: committed_at(2) == Some(10.0) within 1e-6 MW ──────────────────
    // (Deepest pre-horizon delivery: second ring-buffer shift between stages 1→2.
    //  Slot 1 (now 10.0 MW) → slot 0)
    let c2 = committed_at(2).expect(
        "AC3 FAIL: committed_at(2) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 2. \
         If committed_at(2) is None, the fishing constraint is absent for stage 2.",
    );
    assert!(
        (c2 - 10.0).abs() < 1e-6,
        "AC3 FAIL: committed_at(2) = {c2} MW, expected 10.0 MW (values_mw[2]). \
         After two ring-buffer shifts, slot 0 must hold 10.0 MW. \
         If the result is 30.0 MW, the second ring-buffer shift (between stages 1 \
         and 2) is not moving slot 1 (10.0 MW) into slot 0 correctly. \
         If the result is 50.0 MW, neither shift has occurred.",
    );

    // ── AC4: committed_at(t) ≈ decision_at(t-3) for t ∈ {3,4,5} ───────────
    // (K=3 ring-buffer: decisions mature 3 stages after being made)
    for t in 3..6_usize {
        let ct = committed_at(t).unwrap_or_else(|| {
            panic!(
                "AC4 FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                t - 3,
            )
        });
        let d_prev3 = decision_at(t - 3).unwrap_or_else(|| {
            panic!(
                "AC4 FAIL: decision_at({}) is None (needed to check K=3 \
                 ring-buffer invariant at stage {t})",
                t - 3,
            )
        });
        assert!(
            (ct - d_prev3).abs() < 1e-6,
            "AC4 FAIL (K=3 ring-buffer shift): committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev3} MW (within 1e-6 MW). \
             With K=3, decisions mature three stages later (slot K-1=2 shifts into \
             slot 0 after three forward steps). The ring buffer is not correctly \
             propagating in-study decisions.",
            t - 3,
        );
    }

    // ── AC5: decision_at(t) non-zero and bounded for t ∈ {0,1,2} ───────────
    // (Active-decision stages: t + 3 < 6; LP saturates on cost ratio)
    for t in 0..3_usize {
        let dt = decision_at(t).unwrap_or_else(|| {
            panic!(
                "AC5 FAIL: decision_at({t}) is None; anticipated thermal id=52 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 3 < 6)",
            )
        });
        assert!(
            dt.abs() > 1e-6,
            "AC5 FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
            t + 3,
        );
        assert!(
            dt <= 200.0 + 1e-6,
            "AC5 FAIL: decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
        );
    }

    // ── AC6: observed_total ≤ EXPECTED_TOTAL_UPPER_BOUND_USD ────────────────
    // (Use immediate_cost, not total_cost which includes theta approximation artefact.
    //  If no seeds delivered: 3 × 150 MW × 744 h × $5000 = $1.674B >> bound → fails)
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
        .sum();

    assert!(
        observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
        "AC6 FAIL: observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         STAGE_2_BACKUP_COST_USD=${STAGE_2_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If none of the seeds are delivered, 3 pre-horizon stages use backup \
         for all 150 MW: 3 × 150 MW × 744 h × $5000/MWh = $1,674,000,000 >> \
         $1,343,665,000.",
    );
}
