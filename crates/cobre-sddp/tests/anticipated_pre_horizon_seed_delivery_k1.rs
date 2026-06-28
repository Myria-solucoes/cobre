//! Integration test: pre-horizon seed delivery at stage 0 with K=1 and the
//! always-active fishing predicate.
//!
//! The always-active fishing equality at stage 0 pins the anticipated thermal to
//! slot 0 of the ring buffer (the 100 MW seed) and the cost-zeroing predicate
//! accepts that delivery at zero LP cost. The 30-series entity IDs (hydro 30,
//! anticipated thermal 31, backup 32) avoid collision with the other anticipated
//! integration tests (which use the 2..7 range).

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
// Analytical cost bound constants
// ---------------------------------------------------------------------------

/// Stage-0 backup cost: the seed (100 MW) is delivered at zero LP cost, so backup
/// carries only 150 - 100 = 50 MW at $5000/MWh × 744 h = $186,000,000.
const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 100.0) * 744.0 * 5000.0;

/// Active-decision cost ceiling: 4 decision stages × 200 MW (max_gen) × $10/MWh
/// × 744 h = $5,952,000. The LP may commit less if cuts are loose, never more.
const MAX_DECISION_COST_USD: f64 = 4.0 * 200.0 * 744.0 * 10.0;

/// Absolute cost tolerance matching `anticipated_numerical_reconciliation_k2`.
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

/// Build a 5-stage K=1 system for the pre-horizon seed delivery test (anticipated
/// thermal id=31, backup id=32, trivial hydro id=30, load 150 MW, seed
/// `past_anticipated_commitments = [(id=31, [100.0])]`).
///
/// **Non-zero seed**: constructing `System` directly via `SystemBuilder` bypasses
/// the `cobre-io` validator that rejects non-zero `values_mw`. That is intentional
/// — the non-zero seed is the deliberate fixture; the rejection rule applies only
/// to JSON input through `load_case`.
///
/// **Cost asymmetry** ($10/MWh anticipated vs $5000/MWh backup) saturates
/// anticipated dispatch at max_gen; **discount_rate = 0.0** collapses all discount
/// factors to 1.0 so the analytical cost derivation is exact.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 1;
    let n_stages: usize = 5;

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

    let anticipated_id = EntityId(31);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant_seed_k1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 10.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let thermal_backup = Thermal {
        id: EntityId(32),
        name: "T_backup_seed_k1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 500.0,
        cost_per_mwh: 5000.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Zero inflow and 1 MW max_gen keep the system firmly in the thermal regime.
    let hydro = Hydro {
        id: EntityId(30),
        name: "H1_seed_k1".to_string(),
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
            hydro_id: EntityId(30),
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

    // The padding region [n_stages, n_stages + k) is the delivery-stage axis read
    // by `fill_anticipated_columns`; it must carry per-thermal costs so the
    // decision column's objective coefficient is non-zero.
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
    // Thermal index 0 = anticipated (id=31), index 1 = backup (id=32). Without
    // these per-thermal cost overrides the LP has no incentive to commit
    // anticipated capacity and decision_at(t) collapses to zero, masking the
    // regression.
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

    // Seed the anticipated ring buffer with 100.0 MW at slot 0: stage 0 reads it
    // via the always-active fishing equality and delivers it at zero LP cost.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(30),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![100.0],
        }],
        recent_observations: vec![],
    };

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
/// One iteration suffices: the fishing equality pins the seed regardless of cut
/// quality, so the deliberately generous cost bound absorbs a loose 1-iteration
/// cut approximation.
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

/// Verify that the pre-horizon seed (100 MW) is delivered at stage 0 via the
/// always-active fishing predicate, and that the ring-buffer shift propagates
/// in-study decisions correctly for stages 1–4. The four assertions (committed
/// seed at stage 0, decision saturation, ring-buffer shift, and the cost upper
/// bound) are derived inline at each AC block below.
#[test]
fn pre_horizon_seed_delivers_at_stage_zero_k1() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train: must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

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

    let anticipated_thermal_id: i32 = 31;
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

    // ── AC1: committed_at(0) == Some(100.0) within 1e-6 MW ─────────────────
    let c0 = committed_at(0).expect(
        "AC1 FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 100 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
    );
    assert!(
        (c0 - 100.0).abs() < 1e-6,
        "AC1 FAIL: committed_at(0) = {c0} MW, expected 100.0 MW (the seed). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 100.0 MW of the ring buffer.",
    );

    // ── AC2: decision_at(t) non-zero and saturates near 200 MW ─────────────
    // Active-decision stages are t ∈ {0,1,2,3} (t + K < n_stages, i.e. t + 1 < 5).
    for t in 0..4_usize {
        let dt = decision_at(t).unwrap_or_else(|| {
            panic!(
                "AC2 FAIL: decision_at({t}) is None; anticipated thermal id=31 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 1 < 5)",
            )
        });
        assert!(
            dt.abs() > 1e-6,
            "AC2 FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
        );
        assert!(
            dt <= 200.0 + 1e-6,
            "AC2 FAIL: decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
        );
    }

    // ── AC3: committed_at(t) ≈ decision_at(t-1) for t ∈ {1,2,3,4} ─────────
    // Ring-buffer shift invariant: after the shift at the end of stage t-1, slot 0
    // holds the in-study decision from t-1, which stage t's fishing equality pins.
    for t in 1..5_usize {
        let ct = committed_at(t).unwrap_or_else(|| {
            panic!(
                "AC3 FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                t - 1,
            )
        });
        let d_prev = decision_at(t - 1).unwrap_or_else(|| {
            panic!(
                "AC3 FAIL: decision_at({}) is None (needed to check ring-buffer \
                 invariant at stage {t})",
                t - 1,
            )
        });
        assert!(
            (ct - d_prev).abs() < 1e-6,
            "AC3 FAIL (ring-buffer shift): committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev} MW (within 1e-6 MW). \
             The ring buffer is not correctly propagating in-study decisions.",
            t - 1,
        );
    }

    // ── AC4: observed_total ≤ EXPECTED_TOTAL_UPPER_BOUND_USD ────────────────
    // Sum per-stage `immediate_cost` (LP objective minus theta), NOT `total_cost`
    // — the latter includes the theta approximation artefact. The bound is derived
    // on the cost-bound constants above.
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
        .sum();

    assert!(
        observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
        "AC4 FAIL: observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         150 MW instead of 50 MW, producing ~$558M >> $191.95M.",
    );
}
