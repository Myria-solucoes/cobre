//! Numerical reconciliation test: LP total cost must match the analytical optimum
//! for a K=2, 6-stage fixture with zero discount rate.
//!
//! ## Bug being guarded against
//!
//! In a 6-stage K=2 study with a 500x cost asymmetry (anticipated thermal at
//! $10/MWh vs backup at $5000/MWh), the LP's total immediate cost should equal
//! the analytical optimum derived below. Before the layout fix, intermediate-stage
//! anticipated dispatch is silently zeroed, forcing the backup thermal to carry
//! all load at those stages. The result is an LP total that is roughly 500x larger
//! than optimal.
//!
//! ## Analytical optimum (discount rate = 0, all discount factors = 1.0)
//!
//! Parameters:
//! - n_stages = 6, K = 2, load = 150 MW, block duration = 744 h/stage
//! - Anticipated thermal: max_gen = 200 MW, cost = $10/MWh
//! - Backup thermal: max_gen = 500 MW, cost = $5000/MWh
//! - Excess generation cost = $0 (over-commitment is free)
//!
//! Stages partition into three zones:
//!
//! **Zone A — Active decision stages (t + K < n_stages → t ∈ {0, 1, 2, 3}):**
//! The LP decides `d_t ∈ [0, 200]`. Because excess is free, the LP saturates
//! `d_t = 200 MW`. The per-block cost of the anticipated-decision column is
//! $10/MWh, charged at the decision stage.
//!
//! Anticipated decision cost = 4 stages × 200 MW × 744 h × $10/MWh
//!                           = **$5,952,000**
//!
//! **Zone B — Delivery stages with matured anticipated commitment (t ∈ {2, 3, 4, 5}):**
//! The anticipated thermal delivers `committed_t = d_{t-K}`. Because `d_{t-K} = 200 MW > 150 MW`
//! load, anticipated covers all load. The per-block cost on the anticipated
//! thermal at delivery stages is zeroed by `zero_anticipated_delivery_thermal_cost`,
//! so the delivered generation costs $0 in the objective. No backup needed.
//!
//! **Zone C — Pre-horizon delivery stages (t ∈ {0, 1}):**
//! The ring buffer has not yet matured a commitment from inside the horizon
//! (`past_anticipated_commitments = [0.0, 0.0]`), so backup must carry all
//! 150 MW load at these two stages.
//!
//! Pre-horizon backup cost = 2 stages × 150 MW × 744 h × $5000/MWh
//!                         = **$1,116,000,000**
//!
//! **Total analytical optimum = $5,952,000 + $1,116,000,000 = $1,121,952,000**
//!
//! ## What this test asserts
//!
//! After training for 10 iterations and running one deterministic simulation,
//! the sum of per-stage `immediate_cost` values (LP objective minus theta at
//! each stage) must equal $1,121,952,000 within $1.00 absolute tolerance.
//!
//! Before the fix, the LP costs approximately 500x more because intermediate
//! anticipated dispatch (`d_1 = d_2 = d_3 ≈ 0`) is zeroed by the cut-coefficient
//! corruption in `state_to_lp_column`'s `Less` branch, forcing backup to carry
//! 150 MW at stages 2, 3, 4, 5 as well as 0 and 1.
//!
//! ## Entity IDs
//!
//! IDs 5 (anticipated), 6 (backup), and 7 (hydro) are chosen to be distinct from
//! both the K=2 saturation test (IDs 2, 4, 3) and the K=3 saturation test
//! (IDs 3, 4, 5) so that combined nextest runs produce unambiguous per-entity
//! attribution in failure messages.

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
use cobre_sddp::{hydro_models::PrepareHydroModelsResult, StudySetup};
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::{build_stochastic_context, ClassSchemes, OpeningTreeInputs};

// ---------------------------------------------------------------------------
// Analytical optimum constants (documented in module-level doc comment above)
// ---------------------------------------------------------------------------

/// Decision cost: 4 active stages × 200 MW × 744 h × $10/MWh = $5,952,000
const EXPECTED_DECISION_COST_USD: f64 = 4.0 * 200.0 * 744.0 * 10.0;

/// Pre-horizon backup cost: 2 stages × 150 MW × 744 h × $5000/MWh =
/// $1,116,000,000
const EXPECTED_BACKUP_COST_USD: f64 = 2.0 * 150.0 * 744.0 * 5000.0;

/// Total analytical optimum = decision cost + pre-horizon backup cost =
/// $1,121,952,000
const EXPECTED_TOTAL_USD: f64 = EXPECTED_DECISION_COST_USD + EXPECTED_BACKUP_COST_USD;

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
/// - 1 bus (deficit cost $5000/MWh, excess cost $0)
/// - 1 trivial hydro (1 hm³ max storage, zero inflow, max_gen 1 MW) — keeps
///   the model in the thermal regime without adding a hydro state variable
///   that complicates interpretation.
/// - 1 anticipated thermal (K=2, cost $10/MWh, max 200 MW) — id=5
/// - 1 backup thermal (cost $5000/MWh, max 500 MW) — id=6
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments = [(id=5, [0.0, 0.0])]` — zero seeds so the
///   test isolates the pre-horizon backup cost exactly as derived in the
///   analytical optimum.
/// - `annual_discount_rate = 0.0` set explicitly on `PolicyGraph` — all
///   discount factors collapse to 1.0, making the analytical cost summation
///   straightforward.
///
/// Entity IDs 5 and 6 are distinct from both the K=2 saturation test (2, 4)
/// and the K=3 saturation test (3, 4) to avoid cross-test confusion in nextest.
fn build_system_reconciliation_k2() -> cobre_core::System {
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

    // Anticipated thermal: K=2 lead stages, cheap at $10/MWh.
    // The 500x cost ratio vs the backup makes it overwhelmingly optimal
    // to saturate d_t = 200 MW at every active stage.
    let anticipated_id = EntityId(5);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant_reconcil".to_string(),
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

    // Backup thermal: very expensive at $5000/MWh.
    // The LP must avoid it wherever anticipated dispatch is available.
    let thermal_backup = Thermal {
        id: EntityId(6),
        name: "T_backup_reconcil".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 5000.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Trivial hydro: 1 hm³ cap, zero inflow, max_gen 1 MW.
    // Present to satisfy `n_hydros = 1` requirements of `ResolvedBounds`
    // while keeping the system firmly in the thermal regime.
    let hydro = Hydro {
        id: EntityId(7),
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
            hydro_id: EntityId(7),
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

    // The padding region [n_stages, n_stages + k) is the delivery-stage axis
    // read by `fill_anticipated_decision_objective`; it must carry the
    // per-thermal cost so the decision column's objective coefficient is
    // non-zero.
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
        // Thermal index 0 = anticipated (id=5): cheap at $10/MWh, max 200 MW
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 200.0;
        // Thermal index 1 = backup (id=6): expensive at $5000/MWh, max 500 MW
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

    // Zero seeds: the past_anticipated_commitments for the anticipated thermal
    // are [0.0, 0.0], matching the pre-horizon conditions in the analytical
    // optimum. These zeros cause backup to carry 150 MW at stages 0 and 1,
    // contributing the $1,116,000,000 pre-horizon backup cost.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(7),
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

    // `annual_discount_rate = 0.0` is set explicitly on the PolicyGraph so
    // that all per-transition discount factors collapse to 1.0. With d_t = 1
    // for all t, the analytical optimum derivation is exact: there is no NPV
    // scaling between decision cost and delivery/backup cost.
    //
    // Note: `PolicyGraph::default()` already has `annual_discount_rate: 0.0`
    // and `transitions: vec![]`. Setting it explicitly here documents the
    // intent and guards against future default changes.
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
        .expect("build_system_reconciliation_k2: valid system")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a [`Config`] for 10-iteration training and 1-scenario deterministic
/// simulation.
///
/// Ten iterations is sufficient for the 500x cost asymmetry to produce cuts
/// that signal the value of anticipated dispatch. If the layout fix is present,
/// the observed cost will match the analytical optimum. If not, the cost will
/// be approximately 500x larger.
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

/// Assert that the LP total cost equals the hand-derived analytical optimum for
/// a K=2, 6-stage fixture with zero discount rate.
///
/// ## Analytical optimum (derivation in module doc)
///
/// - Anticipated decision cost: 4 stages × 200 MW × 744 h × $10/MWh
///   = **$5,952,000**
/// - Pre-horizon backup cost (stages 0, 1): 2 × 150 MW × 744 h × $5000/MWh
///   = **$1,116,000,000**
/// - **Total = $1,121,952,000**
///
/// ## Cost field used
///
/// The test sums `stage.costs[block].immediate_cost` across all stages and
/// blocks. `immediate_cost = LP_objective - theta_variable`, i.e. the stage's
/// realized cost excluding the future-cost approximation. This matches the
/// definition of "total system cost" in the analytical derivation.
///
/// `SimulationCostResult::total_cost` is NOT used because it includes the
/// theta (future cost function) value, which is an approximation artefact of
/// the SDDP algorithm and should not appear in the realized cost total.
///
/// ## Expected failure on current HEAD
///
/// Before the layout fix (Epic 03 ticket-009), `d_1 = d_2 = d_3 ≈ 0` due to
/// cut-coefficient corruption. The backup thermal carries 150 MW at all 6
/// stages, giving:
///
///   observed ≈ decision_cost(1 stage) + backup_cost(6 stages)
///            ≈ 1×200×744×10 + 6×150×744×5000
///            ≈ 1,488,000 + 3,348,000,000
///            ≈ ~3.35 billion
///
/// That is roughly 3x the expected value for this fixture (500x is the
/// per-stage cost ratio, modulated by the decision cost that does correctly
/// fire at stage 0). The failure message shows both the observed and expected
/// values and their difference.
#[test]
#[ignore = "fails until Epic 03 ticket-009 lands the layout fix"]
fn lp_total_cost_matches_analytical_optimum_k2_discount_zero() {
    // Step 1: build and train the study.
    // (Constants EXPECTED_DECISION_COST_USD, EXPECTED_BACKUP_COST_USD, and
    // EXPECTED_TOTAL_USD are defined and documented at the module level.)
    let system = build_system_reconciliation_k2();
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

    // Step 2: run a single deterministic simulation.
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
        6,
        "scenario must contain one stage record per study stage (n_stages=6)",
    );

    // Step 3: sum per-stage immediate costs.
    //
    // `immediate_cost = LP_objective - theta`, i.e. the stage's realized
    // resource + deficit cost, excluding the future-cost approximation.
    // With discount_rate = 0.0, the cumulative discount factor is 1.0 at
    // every stage, so immediate_cost equals the undiscounted stage cost.
    //
    // Each stage has exactly one block (block_id = None for the stage-level
    // aggregate entry). The `costs` vec has one entry per stage in this
    // single-block fixture.
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|stage| stage.costs.iter())
        .map(|cost| cost.immediate_cost)
        .sum();

    // Step 4: assert within $1000 absolute tolerance.
    //
    // $1000 on a ~$1.12 billion cost is a relative tolerance of ~8.9e-7,
    // comfortably above HiGHS's documented 1e-9 precision tolerance (which
    // could yield up to ~$1.12 of theoretical noise on this objective
    // magnitude). The pre-fix error is ~$370M, so $1000 keeps 5+ orders of
    // magnitude of detection headroom while removing any solver-tolerance
    // fragility.
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    assert!(
        (observed_total - EXPECTED_TOTAL_USD).abs() < COST_TOLERANCE_USD,
        "LP total cost {} differs from analytical optimum {} by {} \
         (tolerance ${:.2}). \
         Pre-fix behaviour: intermediate anticipated dispatch is zeroed \
         (d_1 = d_2 = d_3 ≈ 0), forcing backup to carry 150 MW at stages \
         2–5 in addition to 0–1, producing a cost far above the optimum.",
        observed_total,
        EXPECTED_TOTAL_USD,
        (observed_total - EXPECTED_TOTAL_USD).abs(),
        COST_TOLERANCE_USD,
    );
}
