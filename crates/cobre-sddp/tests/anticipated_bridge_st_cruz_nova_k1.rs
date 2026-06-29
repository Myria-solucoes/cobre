//! Integration test: ST.CRUZ NOVA bridge-parity fixture with K=1 pre-horizon
//! seed delivery.
//!
//! ## Analytical cost bound
//!
//! With `K = 1`, `n_stages = 5`, a single anticipated thermal (id=61,
//! `ST_CRUZ_NOVA`), and `past_anticipated_commitments.values_mw = [204.5647]`:
//!
//! - Stage 0: anticipated delivers seed (204.5647 MW, zero cost) + backup
//!   covers remaining (250.0 - 204.5647) MW at $5000/MWh × 744 h ≈ $169,019,316.
//! - Stages 1–4 delivery: anticipated covers ≥ load (zeroed cost).
//! - Decision cost ≤ 4 × 350 MW × $10/MWh × 744 h = $10,416,000.
//! - Total ≤ $169,019,316 + $10,416,000 + $1,000 (tolerance).
//!
//! The 204.5647 MW seed is the block-fraction-weighted aggregate of ST.CRUZ NOVA
//! per-block MW values (227.86, 238.37, 173.51) against September 2024 block
//! fractions (0.2333, 0.2833, 0.4834); the 1e-3 MW tolerance reflects its
//! four-decimal accuracy.
//!
//! The fishing constraint is always active for every anticipated plant, so a
//! fishing row is emitted at every stage. The anticipated plant's delivery-stage
//! per-block thermal cost is skipped in `fill_thermal_columns` (the plant is
//! detected via `anticipated_local_by_sys_pos`), so those columns are consumed
//! at zero cost.
//!
//! The 60-series entity IDs are distinct from the other anticipated tests so
//! combined nextest runs give unambiguous per-entity failure attribution.

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
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::sync::mpsc;

use cobre_core::entities::{
    bus::DeficitSegment,
    hydro::{HydroGenerationModel, HydroPenalties},
    thermal::AnticipatedConfig,
};
use cobre_core::scenario::{InflowModel, LoadModel};
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
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

// ---------------------------------------------------------------------------
// Analytical cost bound constants (documented in module doc comment above)
// ---------------------------------------------------------------------------

const STAGE_0_BACKUP_COST_USD: f64 = (250.0 - 204.5647) * 744.0 * 5000.0;
const MAX_DECISION_COST_USD: f64 = 4.0 * 350.0 * 744.0 * 10.0;
const COST_TOLERANCE_USD: f64 = 1_000.0;
const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
    STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build the 5-stage K=1 ST.CRUZ NOVA fixture.
///
/// Builds the resolved `cobre_core::System` directly via `SystemBuilder::new()`,
/// bypassing the `cobre-io` parse-and-validate pipeline, to keep the test
/// self-contained; the 204.5647 MW seed is within bounds and would also pass
/// `load_case`.
///
/// The anticipated ($10/MWh) vs backup ($5000/MWh) 500× ratio makes the LP
/// prefer anticipated dispatch. `annual_discount_rate = 0.0` collapses all
/// discount factors to 1.0, making the analytical cost derivation exact.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 1;
    let n_stages: usize = 5;

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let anticipated_id = EntityId(61);
    let thermal_ant = make_thermal(
        anticipated_id,
        ThermalSpec {
            name: "ST_CRUZ_NOVA".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 350.0,
            cost_per_mwh: 10.0,
            anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let thermal_backup = make_thermal(
        EntityId(62),
        ThermalSpec {
            name: "T_backup".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 5000.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    // Trivial hydro keeps the model in the thermal regime; present only so
    // `n_hydros = 1` exercises the hydro state path without adding uncertainty.
    let hydro = make_hydro(
        EntityId(60),
        HydroSpec {
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
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
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
                    ..Default::default()
                },
            )
        })
        .collect();

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

    // The padding region [n_stages, n_stages + k) is the delivery-stage axis
    // read by `fill_anticipated_columns`; it must carry per-thermal
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
    // SystemBuilder sorts by EntityId: index 0 = anticipated (id=61), index 1 =
    // backup (id=62). Without these per-thermal overrides the LP has no cost
    // incentive to commit anticipated capacity, so decision_at(t) collapses to
    // zero and masks the regression assertion.
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

    // Slot 0 = 204.5647 MW seed (block-fraction-weighted aggregate; see module
    // doc). The always-active fishing equality reads it at stage 0 and delivers
    // it at zero LP cost, leaving backup to cover the remainder.
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

/// One training iteration suffices: the stage-0 fishing equality pins the seed
/// delivery regardless of cut quality, and the cost bound is deliberately
/// generous to absorb the loose 1-iteration cut.
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
// Test
// ---------------------------------------------------------------------------

/// Verify that the ST.CRUZ NOVA pre-horizon seed (204.5647 MW) is delivered at
/// stage 0 via the always-active fishing predicate, and that the ring-buffer
/// shift propagates in-study decisions correctly for stages 1–4.
#[test]
fn pre_horizon_seed_delivers_at_stage_zero_st_cruz_nova_k1() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup_in_code(system, &config);
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
    // After the shift ending stage t-1, slot 0 holds that stage's decision, and
    // stage t's fishing equality pins generation to it.
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
    // Sum `immediate_cost` (LP objective minus theta), NOT `total_cost`; the
    // latter includes the theta approximation artefact. Bound derived in the
    // module doc.
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
