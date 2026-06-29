//! Numerical reconciliation test: LP total cost must match the analytical optimum
//! for a K=2, 6-stage fixture with zero discount rate.
//!
//! A regression that silently zeroes intermediate-stage anticipated dispatch
//! forces backup to carry that load, inflating the LP total by hundreds of times.
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
//! The LP decides `d_t ∈ [0, 200]`. Because load = 150 MW < max_gen = 200 MW
//! and excess is free, the LP commits `d_t = load = 150 MW` — not max_gen.
//! Over-committing to 200 MW costs 50 MW × $10/MWh × 744 h = $372k/stage extra
//! with no benefit. The per-block cost of the anticipated-decision column is
//! $10/MWh, charged at the decision stage.
//!
//! Anticipated decision cost = 4 stages × 150 MW × 744 h × $10/MWh
//!                           = **$4,464,000**
//!
//! **Zone B — Delivery stages with matured anticipated commitment (t ∈ {2, 3, 4, 5}):**
//! The anticipated thermal delivers `committed_t = d_{t-K} = 150 MW = load`.
//! Per-block cost on the anticipated thermal at delivery stages is skipped in
//! `fill_thermal_columns` (never written; the anticipated thermal is detected
//! via `anticipated_local_by_sys_pos`), so delivered generation costs $0
//! in the objective. No backup needed since 150 MW = load exactly.
//!
//! **Zone C — Pre-horizon stages (t ∈ {0, 1}):**
//! The always-active fishing predicate pins the anticipated thermal to seed
//! slot 0 = 0 MW (`past_anticipated_commitments = [0.0, 0.0]`). The LP must
//! dispatch backup at $5000/MWh to meet the 150 MW load. The cost-zeroing
//! predicate is also always-active, so the anticipated thermal column has
//! objective 0 — but its column upper bound is fishing-pinned to 0, leaving
//! backup as the sole feasible source.
//!
//! Pre-horizon backup cost = 2 stages × 150 MW × 744 h × $5000/MWh
//!                         = **$1,116,000,000**
//!
//! **Total analytical optimum = $4,464,000 + $1,116,000,000 = $1,120,464,000**
//!
//! The 5/6/7 entity IDs are distinct from the K=2 and K=3 saturation tests so
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
// Analytical optimum constants (documented in module-level doc comment above)
// ---------------------------------------------------------------------------

/// Active decision cost: 4 stages × 150 MW (load, not max_gen — over-committing
/// is wasted when excess is free) × 744 h × $10/MWh = $4,464,000.
const EXPECTED_DECISION_COST_USD: f64 = 4.0 * 150.0 * 744.0 * 10.0;

/// Pre-horizon backup cost: 2 stages × 150 MW × 744 h × $5000/MWh = $1,116,000,000.
/// At t∈{0,1} the always-active fishing equality pins the anticipated thermal to
/// the zero seed (slot 0 = 0 MW), leaving backup as the sole feasible source for
/// the 150 MW load. See module doc Zone C.
const EXPECTED_PRE_HORIZON_BACKUP_COST_USD: f64 = 2.0 * 150.0 * 744.0 * 5000.0;

/// Total = active decision cost (stages 2..=5) + pre-horizon backup cost
/// (stages 0, 1) = $4,464,000 + $1,116,000,000 = $1,120,464,000.
const EXPECTED_TOTAL_USD: f64 = EXPECTED_DECISION_COST_USD + EXPECTED_PRE_HORIZON_BACKUP_COST_USD;

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build the 6-stage K=2 reconciliation fixture.
///
/// The trivial hydro keeps the model in the thermal regime; it exists only so
/// `n_hydros = 1` is satisfied without adding a hydro state variable that
/// complicates interpretation. `annual_discount_rate = 0.0` collapses all
/// discount factors to 1.0, making the analytical cost summation exact.
fn build_system_reconciliation_k2() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 2;
    let n_stages: usize = 6;

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

    let anticipated_id = EntityId(5);
    let thermal_ant = make_thermal(
        anticipated_id,
        ThermalSpec {
            name: "T_ant_reconcil".to_string(),
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
            ..Default::default()
        },
    );

    let thermal_backup = make_thermal(
        EntityId(6),
        ThermalSpec {
            name: "T_backup_reconcil".to_string(),
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

    let hydro = make_hydro(
        EntityId(7),
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
                    ..Default::default()
                },
            )
        })
        .collect();

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
    // read by `fill_anticipated_columns`; it must carry the
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
    // SystemBuilder sorts by EntityId: index 0 = anticipated (id=5), index 1 =
    // backup (id=6).
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

    // Zero seeds: slot 0 carries 0 MW at both pre-horizon stages (Zone C).
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

    // Set explicitly (not relying on `PolicyGraph::default()`) so a future
    // default change cannot silently introduce NPV scaling into the analytical
    // derivation.
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

/// Ten training iterations let the 500x cost asymmetry produce cuts that signal
/// the value of anticipated dispatch, driving the observed cost to the optimum.
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
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Assert that the LP total cost equals the hand-derived analytical optimum
/// ($1,120,464,000; derivation in module doc) for a K=2, 6-stage fixture with
/// zero discount rate.
#[test]
fn lp_total_cost_matches_analytical_optimum_k2_discount_zero() {
    let system = build_system_reconciliation_k2();
    let config = build_config();
    let mut setup = build_setup_in_code(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new: must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
        .expect("training error: train() must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: training returned an error: {:?}",
        outcome.error,
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
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

    // Sum `immediate_cost` (LP objective minus theta), excluding the future-cost
    // approximation — the realized cost the analytical optimum is derived for.
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|stage| stage.costs.iter())
        .map(|cost| cost.immediate_cost)
        .sum();

    // $1000 sits comfortably above HiGHS's 1e-9 precision yet far below the
    // ~$370M pre-fix error, giving 5+ orders of magnitude of detection headroom.
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    assert!(
        (observed_total - EXPECTED_TOTAL_USD).abs() < COST_TOLERANCE_USD,
        "LP total cost {} differs from analytical optimum {} by {} \
         (tolerance ${:.2}). \
         Pre-fix behaviour: intermediate anticipated dispatch is zeroed \
         (d_1 = d_2 = d_3 ≈ 0), forcing the LP to backfill with backup at \
         $5000/MWh — producing a cost gap of approximately $370M.",
        observed_total,
        EXPECTED_TOTAL_USD,
        (observed_total - EXPECTED_TOTAL_USD).abs(),
        COST_TOLERANCE_USD,
    );

    // The named cost categories must sum to `immediate_cost` at every stage,
    // including the anticipated commitment fuel as `anticipated_thermal_cost`.
    // `hydro_violation_cost` already aggregates its six sub-components and
    // `spillage_cost` already includes diversion — sum the aggregates, not the
    // parts, or the total double-counts.
    const RECONCILE_TOLERANCE_USD: f64 = 1.0;
    let mut saw_nonzero_anticipated = false;
    for stage in &scenario.stages {
        for cost in &stage.costs {
            let category_sum = cost.thermal_cost
                + cost.anticipated_thermal_cost
                + cost.contract_cost
                + cost.deficit_cost
                + cost.excess_cost
                + cost.storage_violation_cost
                + cost.filling_target_cost
                + cost.hydro_violation_cost
                + cost.inflow_penalty_cost
                + cost.generic_violation_cost
                + cost.spillage_cost
                + cost.turbined_cost
                + cost.curtailment_cost
                + cost.exchange_cost
                + cost.pumping_cost;
            assert!(
                (category_sum - cost.immediate_cost).abs() < RECONCILE_TOLERANCE_USD,
                "stage {}: Σ(named cost categories) = {} must equal immediate_cost = {} \
                 (diff {}); the anticipated commitment fuel must be attributed to \
                 anticipated_thermal_cost, not left as an unattributed remainder",
                cost.stage_id,
                category_sum,
                cost.immediate_cost,
                (category_sum - cost.immediate_cost).abs(),
            );
            if cost.anticipated_thermal_cost.abs() > RECONCILE_TOLERANCE_USD {
                saw_nonzero_anticipated = true;
            }
        }
    }
    // Zone A (decision stages t∈{0,1,2,3}) must book a positive
    // anticipated_thermal_cost; otherwise the new field is dead and the
    // reconciliation above would pass trivially.
    assert!(
        saw_nonzero_anticipated,
        "expected a non-zero anticipated_thermal_cost at the decision stages; \
         got zero everywhere (the GNL fuel was not attributed)",
    );
}
