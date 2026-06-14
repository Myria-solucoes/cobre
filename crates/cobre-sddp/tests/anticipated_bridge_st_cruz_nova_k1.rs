//! Integration test: ST.CRUZ NOVA bridge-parity fixture with K=1 pre-horizon
//! seed delivery.
//!
//! ## What this test asserts
//!
//! With `K = 1`, `n_stages = 5`, a single anticipated thermal (id=61, named
//! `ST_CRUZ_NOVA`), and `past_anticipated_commitments.values_mw = [204.5647]`,
//! the LP must:
//!
//! 1. Deliver `committed_at(0) ≈ 204.5647 MW` — the always-active fishing
//!    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
//!    buffer, which holds the 204.5647 MW seed. The cost-zeroing predicate
//!    zeros the per-block objective for this column so the LP accepts the
//!    delivery at zero additional cost.
//!
//! 2. Commit `decision_at(t) > 1e-6` for `t ∈ {0, 1, 2, 3}` — the anticipated
//!    thermal costs $10/MWh vs the backup's $5000/MWh, and the per-block cost
//!    of the decision column is non-zero only at the decision stage (not zeroed),
//!    so the LP commits a non-trivial anticipated amount to avoid backup at
//!    future stages.
//!
//! 3. Satisfy `committed_at(t) ≈ decision_at(t-1)` for `t ∈ {1, 2, 3, 4}` —
//!    the ring-buffer shift invariant from `anticipated_simulation_ring_buffer`.
//!
//! 4. Satisfy the analytical cost bound:
//!    - Stage 0: anticipated delivers seed (204.5647 MW, zero cost) + backup
//!      covers remaining (250.0 - 204.5647) MW at $5000/MWh × 744 h
//!      ≈ $169,019,316.
//!    - Stages 1–4 delivery: anticipated covers ≥ load (zeroed cost).
//!    - Decision cost ≤ 4 × 350 MW × $10/MWh × 744 h = $10,416,000.
//!    - Total ≤ $169,019,316 + $10,416,000 + $1,000 (tolerance).
//!
//! ## Parameter derivation
//!
//! The fixture parameters mirror the NEWAVE example case `example/newave_rodada`
//! from the `cobre-bridge` repository. The bridge findings document at
//! `~/git/cobre-bridge/docs/findings/cobre-anticipated-thermal-pre-horizon-limitation.md`
//! records that thermal ST.CRUZ NOVA (NEWAVE code 86, GNL configuration) has
//! lag 1 (K=1) and an aggregated MW value of 204.5647 MW, computed by
//! block-fraction weighting of per-patamar MW values (227.86, 238.37, 173.51)
//! against September 2024 block fractions (0.2333, 0.2833, 0.4834).
//!
//! The always-active fishing predicate is implemented at
//! `crates/cobre-sddp/src/indexer.rs:1555`. The cost-zeroing predicate for
//! delivery-stage columns is applied in `fill_anticipated_decision_objective`.
//!
//! ## Legacy behaviour (before always-active fishing)
//!
//! With the old predicate `K_i > stage_idx`, stage 0 had fishing *inactive*.
//! `committed_at(0)` returned `None` rather than `Some(204.5647)`. The seed
//! was never delivered to the LP; instead the backup had to cover all 250 MW,
//! producing a stage-0 cost of $930,000,000 — well above this test's bound.
//!
//! ## Entity IDs
//!
//! IDs 60 (hydro), 61 (anticipated thermal ST_CRUZ_NOVA), 62 (backup thermal)
//! are chosen to be distinct from all existing anticipated tests:
//! - `anticipated_simulation_ring_buffer.rs` uses IDs 2/3/4
//! - `anticipated_numerical_reconciliation_k2.rs` uses IDs 5/6/7
//! - `anticipated_d_t_saturation_k2.rs` and `_k3.rs` use IDs 2/3/4 and 3/4/5
//! - `anticipated_pre_horizon_seed_delivery_k1.rs` uses IDs 30/31/32
//! - `anticipated_pre_horizon_seed_delivery_k2.rs` uses IDs 41/42/43
//! - `anticipated_pre_horizon_seed_delivery_k3.rs` uses IDs 51/52/53
//!
//! The 60-series ensures no cross-test entity confusion in nextest.

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

const STAGE_0_BACKUP_COST_USD: f64 = (250.0 - 204.5647) * 744.0 * 5000.0;
const MAX_DECISION_COST_USD: f64 = 4.0 * 350.0 * 744.0 * 10.0;
const COST_TOLERANCE_USD: f64 = 1_000.0;
const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
    STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

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

    fn broadcast<T: CommData>(&self, _: &mut [T], _: usize) -> Result<(), CommError> {
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

/// Build a 5-stage K=1 system for the ST.CRUZ NOVA bridge-parity seed delivery
/// test:
///
/// - 1 bus (deficit cost $5000/MWh, excess cost $0)
/// - 1 trivial hydro (id=60, 1 hm³ max, zero inflow) — keeps the model in the
///   thermal regime without adding hydro uncertainty.
/// - 1 anticipated thermal (id=61, `ST_CRUZ_NOVA`, K=1, cost $10/MWh,
///   max_gen 350 MW)
/// - 1 backup thermal (id=62, `T_backup`, cost $5000/MWh, max_gen 500 MW)
/// - Load 250 MW constant across all stages
/// - `past_anticipated_commitments = [(id=61, [204.5647])]` — the bridge seed
///
/// **Note on construction**: this function constructs the resolved
/// `cobre_core::System` directly via `SystemBuilder::new()`, bypassing the
/// `cobre-io` parse-and-validate pipeline. The 204.5647 MW seed is within the
/// `[0.0, max_generation_mw]` bounds and would also pass the semantic
/// bounds-check through `load_case`; the fixture builds the system directly to
/// keep the test self-contained and avoid filesystem I/O.
///
/// **Cost asymmetry**: anticipated ($10/MWh) vs backup ($5000/MWh) gives a
/// 500× ratio so the LP prefers anticipated dispatch to avoid expensive backup.
///
/// **Discount rate = 0.0** is set explicitly on `PolicyGraph` so all discount
/// factors collapse to 1.0, making the analytical cost derivation exact.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 1;
    let n_stages: usize = 5;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 5000.0,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K=1 lead stage, very cheap so the LP prefers
    // anticipated commitment over backup at every active decision stage.
    let anticipated_id = EntityId(61);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "ST_CRUZ_NOVA".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 350.0,
        cost_per_mwh: 10.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Backup thermal: expensive so the LP uses it only when anticipated
    // capacity cannot cover the full load (stage 0: seed covers 204.5647 MW,
    // backup handles the remaining 45.4353 MW).
    let thermal_backup = Thermal {
        id: EntityId(62),
        name: "T_backup".to_string(),
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
        id: EntityId(60),
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
            start_date: NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
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
            hydro_id: EntityId(60),
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
            mean_mw: 250.0,
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

    // The padding region [n_stages, n_stages + k) is the delivery-stage axis
    // read by `fill_anticipated_decision_objective`; it must carry per-thermal
    // costs so the decision column's objective coefficient is non-zero.
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
                max_generation_mw: 350.0,
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
    // Thermal index 0 = anticipated (id=61, ST_CRUZ_NOVA): cheap at $10/MWh,
    // max 350 MW.
    // Thermal index 1 = backup (id=62, T_backup): expensive at $5000/MWh,
    // max 500 MW.
    // These per-thermal overrides ensure the LP distinguishes them. Without
    // the override the LP has no cost incentive to commit anticipated capacity
    // and decision_at(t) collapses to zero, masking the regression assertion.
    for s in 0..thermal_axis {
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 350.0;
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

    // Seed the anticipated ring buffer with 204.5647 MW at slot 0.
    //
    // With the always-active fishing predicate (see indexer.rs:1555), stage 0
    // reads slot 0 (= 204.5647 MW) via the fishing equality and delivers it at
    // zero LP cost (cost-zeroing predicate). The backup thermal covers the
    // remaining (250.0 - 204.5647) MW at $5000/MWh.
    //
    // This value mirrors the block-fraction-weighted aggregate for ST.CRUZ NOVA
    // from the bridge findings document at
    // ~/git/cobre-bridge/docs/findings/cobre-anticipated-thermal-pre-horizon-limitation.md.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(60),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![204.5647],
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

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a [`Config`] for 1-iteration training and 1-scenario simulation.
///
/// One iteration is sufficient to demonstrate that the seed value reaches the
/// LP: the fishing equality at stage 0 pins the anticipated thermal to
/// 204.5647 MW regardless of cut quality, and the cost-zeroing predicate
/// removes any double-counting. The cost bound is deliberately generous to
/// accommodate a loose 1-iteration cut approximation.
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
    .expect("build_stochastic_context: must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    StudySetup::new(&system, config, stochastic, hydro_models)
        .expect("StudySetup::new: must succeed")
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Verify that the ST.CRUZ NOVA pre-horizon seed (204.5647 MW) is delivered at
/// stage 0 via the always-active fishing predicate, and that the ring-buffer
/// shift propagates in-study decisions correctly for stages 1–4.
///
/// ## Fixture
///
/// - K=1, n_stages=5
/// - Anticipated thermal id=61 (`ST_CRUZ_NOVA`): cost $10/MWh, max_gen 350 MW
/// - Backup thermal id=62 (`T_backup`): cost $5000/MWh, max_gen 500 MW
/// - Load: 250 MW constant across all stages
/// - Seed: `past_anticipated_commitments = [(id=61, [204.5647])]`
///
/// ## Assertions
///
/// **AC-delivery** — `committed_at(0) ≈ 204.5647 MW` within 1e-3 MW: the
/// always-active fishing equality at stage 0 pins the anticipated thermal to
/// slot 0 = 204.5647 MW of the ring buffer.
///
/// **AC-decision-nonzero** — `decision_at(t) > 1e-6` for `t ∈ {0,1,2,3}`: the
/// LP commits a non-trivial anticipated amount to avoid $5000/MWh backup.
///
/// **AC-ring-buffer** — `committed_at(t) ≈ decision_at(t-1)` for
/// `t ∈ {1,2,3,4}`: ring-buffer shift invariant.
///
/// **AC-cost-bound** — `observed_total ≤ EXPECTED_TOTAL_UPPER_BOUND_USD`:
/// stage-0 backup cost (≈$169M) + decision cost ceiling ($10.4M) + $1000
/// tolerance.
///
/// ## What a regression looks like
///
/// If the always-active fishing predicate is missing (legacy `K_i > stage_idx`):
/// - `committed_at(0)` returns `None` instead of `Some(204.5647)` →
///   AC-delivery fails.
/// - Stage-0 backup must carry all 250 MW → stage-0 cost ≈ $927M ≫ $179.4M →
///   AC-cost-bound fails.
///
/// If the cost-zeroing predicate is missing:
/// - The LP objective double-counts the 204.5647 MW seed delivery →
///   AC-cost-bound fails.
#[test]
fn pre_horizon_seed_delivers_at_stage_zero_st_cruz_nova_k1() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    // Step 1: train the policy for 1 iteration.
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
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
        5,
        "scenario must contain one record per study stage (n_stages=5)",
    );

    // Step 3: define accessors for the anticipated thermal (id=61).
    let anticipated_thermal_id: i32 = 61;
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

    // ── AC-delivery: committed_at(0) ≈ 204.5647 MW within 1e-3 MW ──────────
    //
    // The always-active fishing equality (see indexer.rs:1555) pins the
    // anticipated thermal's generation at stage 0 to slot 0 of the ring buffer
    // = 204.5647 MW. Before the always-active predicate, the condition
    // `K_i > stage_idx` was false at stage 0 with K=1, so fishing was inactive
    // and committed_at(0) returned None.
    //
    // Tolerance is 1e-3 MW per the bridge findings doc, which specifies that
    // the aggregated value is accurate to four decimal places.
    let c0 = committed_at(0).expect(
        "AC-delivery FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 204.5647 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
    );
    assert!(
        (c0 - 204.5647).abs() < 1e-3,
        "AC-delivery FAIL: committed_at(0) = {c0} MW, expected 204.5647 MW within \
         1e-3 MW tolerance. The fishing equality at stage 0 must pin the anticipated \
         thermal to slot 0 = 204.5647 MW of the ring buffer.",
    );

    // ── AC-decision-nonzero: decision_at(t) > 1e-6 for t ∈ {0,1,2,3} ───────
    //
    // For active-decision stages t ∈ {0,1,2,3} (t + K < n_stages = t + 1 < 5),
    // the LP commits a non-trivial amount because the anticipated thermal is
    // 500× cheaper than backup. After 1 iteration the cut may be loose, but the
    // LP should still commit a non-zero amount.
    for t in 0..4_usize {
        let dt = decision_at(t).unwrap_or_else(|| {
            panic!(
                "AC-decision-nonzero FAIL: decision_at({t}) is None; anticipated \
                 thermal id=61 was not found in stage {t} thermals or \
                 anticipated_decision_mw is absent (stage {t} is an active-decision \
                 stage: {t} + 1 < 5)",
            )
        });
        assert!(
            dt.abs() > 1e-6,
            "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
        );
        assert!(
            dt <= 350.0 + 1e-6,
            "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW exceeds \
             max_gen=350 MW. This indicates a bounds violation in the LP.",
        );
    }

    // ── AC-ring-buffer: committed_at(t) ≈ decision_at(t-1) for t ∈ {1,2,3,4}
    //
    // The ring-buffer shift invariant: after the shift at the end of stage t-1,
    // slot 0 holds the in-study decision from stage t-1. Stage t's fishing
    // equality then pins the anticipated thermal generation to that value.
    for t in 1..5_usize {
        let ct = committed_at(t).unwrap_or_else(|| {
            panic!(
                "AC-ring-buffer FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                t - 1,
            )
        });
        let d_prev = decision_at(t - 1).unwrap_or_else(|| {
            panic!(
                "AC-ring-buffer FAIL: decision_at({}) is None (needed to check \
                 ring-buffer invariant at stage {t})",
                t - 1,
            )
        });
        assert!(
            (ct - d_prev).abs() < 1e-6,
            "AC-ring-buffer FAIL: committed_at({t}) = {ct} MW should equal \
             decision_at({}) = {d_prev} MW (within 1e-6 MW). The ring buffer is \
             not correctly propagating in-study decisions.",
            t - 1,
        );
    }

    // ── AC-cost-bound: observed_total ≤ EXPECTED_TOTAL_UPPER_BOUND_USD ───────
    //
    // Sum per-stage `immediate_cost` (= LP objective minus theta).
    // Use `immediate_cost`, NOT `total_cost`; the latter includes the theta
    // approximation artefact.
    //
    // Upper bound derivation:
    //   Stage 0: seed (204.5647 MW) at zero cost + backup
    //            (250.0 - 204.5647) MW × 744 h × $5000/MWh ≈ $169,019,316.
    //   Stages 1–4 delivery: anticipated covers ≥ load (zeroed cost), no backup.
    //   Decision cost ≤ 4 × 350 MW × 744 h × $10/MWh = $10,416,000.
    //   Tolerance = $1,000.
    //   Total upper bound ≈ $179,436,316.
    //
    // If the seed is not delivered (regression), stage-0 backup covers 250 MW
    // instead of 45.4353 MW: stage-0 cost ≈ $927M ≫ this bound → AC-cost-bound
    // fails.
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
        .sum();

    assert!(
        observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
        "AC-cost-bound FAIL: observed_total = ${observed_total:.2} exceeds upper \
         bound ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         250 MW instead of 45.4353 MW, producing ~$927M >> this bound.",
    );
}
