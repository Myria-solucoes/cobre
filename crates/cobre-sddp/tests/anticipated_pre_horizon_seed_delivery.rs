//! Pre-horizon seed-delivery integration tests for an anticipated thermal across
//! lead_stages K = 1, 2, 3. Each test trains a small in-code study, runs a
//! one-scenario simulation, and asserts that the matured ring-buffer seeds are
//! delivered at the early stages, that anticipated decisions saturate within
//! bounds, that the ring-buffer shift maps committed_at(t) ≈ decision_at(t−K),
//! and that the observed cost stays under a per-K analytical upper bound. Each
//! K's derivation and cost bound live on its test function.

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

mod common;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};
use common::run_simulation;

// ---------------------------------------------------------------------------
// Per-K fixture table
// ---------------------------------------------------------------------------

/// Per-K parameters for the pre-horizon seed-delivery fixtures. Each `#[test]`
/// builds an independent `System` from one entry; the entity IDs sit in disjoint
/// decades (30s / 40s / 50s), themselves disjoint from the 2..7 range the sibling
/// anticipated tests use, so a combined nextest run attributes failures
/// unambiguously per entity.
struct SeedDeliveryFixture {
    n_stages: usize,
    /// Anticipated `lead_stages` K — also the ring-buffer depth `k_max` and the
    /// `kN` suffix in each entity name.
    k: usize,
    bus_id: EntityId,
    hydro_id: EntityId,
    anticipated_id: EntityId,
    backup_id: EntityId,
    /// Anticipated ring-buffer seeds, MW (length `k`).
    seeds_mw: &'static [f64],
    iterations: usize,
}

const FIXTURE_K1: SeedDeliveryFixture = SeedDeliveryFixture {
    n_stages: 5,
    k: 1,
    bus_id: EntityId(1),
    hydro_id: EntityId(30),
    anticipated_id: EntityId(31),
    backup_id: EntityId(32),
    seeds_mw: &[100.0],
    iterations: 1,
};

const FIXTURE_K2: SeedDeliveryFixture = SeedDeliveryFixture {
    n_stages: 5,
    k: 2,
    bus_id: EntityId(1),
    hydro_id: EntityId(41),
    anticipated_id: EntityId(42),
    backup_id: EntityId(43),
    seeds_mw: &[80.0, 50.0],
    iterations: 5,
};

const FIXTURE_K3: SeedDeliveryFixture = SeedDeliveryFixture {
    n_stages: 6,
    k: 3,
    bus_id: EntityId(1),
    hydro_id: EntityId(51),
    anticipated_id: EntityId(52),
    backup_id: EntityId(53),
    seeds_mw: &[50.0, 30.0, 10.0],
    iterations: 5,
};

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build the `System` for one seed-delivery fixture: one anticipated thermal at
/// `fixture.anticipated_id`, one backup at `fixture.backup_id`, a trivial hydro at
/// `fixture.hydro_id`, load 150 MW, ring-buffer seed `fixture.seeds_mw`.
///
/// Constructing `System` directly via `SystemBuilder` bypasses the `cobre-io`
/// validator that rejects non-zero `values_mw`: the non-zero seed is the
/// deliberate fixture; the rejection rule applies only to JSON input through
/// `load_case`. The $10/MWh anticipated vs $5000/MWh backup asymmetry saturates
/// anticipated dispatch at max_gen, and `annual_discount_rate = 0.0` collapses
/// every discount factor to 1.0 so each test's analytical cost derivation is exact.
fn build_system(fixture: &SeedDeliveryFixture) -> cobre_core::System {
    use chrono::NaiveDate;

    let k = fixture.k;
    let n_stages = fixture.n_stages;

    let bus = make_bus(
        fixture.bus_id,
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

    let thermal_ant = make_thermal(
        fixture.anticipated_id,
        ThermalSpec {
            name: format!("T_ant_seed_k{k}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
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
        fixture.backup_id,
        ThermalSpec {
            name: format!("T_backup_seed_k{k}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 5000.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    // Zero inflow and 1 MW max_gen keep the system firmly in the thermal regime.
    let hydro = make_hydro(
        fixture.hydro_id,
        HydroSpec {
            name: format!("H1_seed_k{k}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
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
            hydro_id: fixture.hydro_id,
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
            bus_id: fixture.bus_id,
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
    // Thermal index 0 = anticipated (cheap); index 1 = backup (expensive). Without
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

    // Seed the anticipated ring buffer; distinct seed values catch slot-swap bugs
    // that identical values would mask across the pre-horizon shifts.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: fixture.hydro_id,
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: fixture.anticipated_id,
            values_mw: fixture.seeds_mw.to_vec(),
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

fn build_config(iterations: usize) -> Config {
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
                limit: iterations as u32,
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
// Train + simulate + drain helper
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Pre-horizon seed delivery at stage 0 with K=1 and the always-active fishing
/// predicate. The always-active fishing equality at stage 0 pins the anticipated
/// thermal to slot 0 of the ring buffer (the 100 MW seed) and the cost-zeroing
/// predicate accepts that delivery at zero LP cost.
///
/// Verifies that the 100 MW seed is delivered at stage 0 and that the ring-buffer
/// shift propagates in-study decisions for stages 1–4 (committed seed at stage 0,
/// decision saturation, ring-buffer shift, cost upper bound — derived inline at
/// each AC block below).
///
/// Cost bound: stage-0 backup carries 150 − 100 = 50 MW × $5000/MWh × 744 h =
/// $186,000,000 (the 100 MW seed delivers at zero LP cost); the active-decision
/// ceiling is 4 decision stages × 200 MW × $10/MWh × 744 h = $5,952,000 (the LP
/// may commit less if cuts are loose, never more); plus a $1,000 tolerance.
#[test]
fn pre_horizon_seed_delivers_at_stage_zero_k1() {
    // Cost bound: see this test's doc comment. Tolerance matches
    // anticipated_numerical_reconciliation_k2.
    const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 100.0) * 744.0 * 5000.0;
    const MAX_DECISION_COST_USD: f64 = 4.0 * 200.0 * 744.0 * 10.0;
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
        STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

    let system = build_system(&FIXTURE_K1);
    let config = build_config(FIXTURE_K1.iterations);
    let mut setup = build_setup_in_code(system, &config);

    // One iteration suffices: the fishing equality pins the seed regardless of cut
    // quality; the generous cost bound absorbs a loose 1-iteration cut.
    let scenario_results = run_simulation(&mut setup, FIXTURE_K1.iterations);

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

/// Pre-horizon seed delivery across two pre-horizon stages with K=2 and the
/// always-active fishing predicate.
///
/// With `K = 2`, `n_stages = 5`, a single anticipated thermal (id=42), and
/// `past_anticipated_commitments.values_mw = [80.0, 50.0]`, the LP must:
///
/// 1. Deliver `committed_at(0) == 80.0 MW` — the always-active fishing
///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
///    buffer, which holds the 80.0 MW seed (`values_mw[0]`). The cost-zeroing
///    predicate zeros the per-block objective for this column so the LP
///    accepts the delivery at zero additional cost.
///
/// 2. Deliver `committed_at(1) == 50.0 MW` — `shift_anticipated_state`
///    moves slot 1 (`values_mw[1] = 50.0`) into slot 0 at the start of stage 1.
///    Stage 1's always-active fishing equality then reads slot 0 = 50.0 MW. This
///    is the K=2-specific assertion that the K=1 delivery test cannot reach: K=1
///    has only one pre-horizon stage, so there is no ring-buffer shift between
///    two pre-horizon stages to exercise.
///
/// 3. Satisfy `committed_at(t) ≈ decision_at(t-2)` for t ∈ {2,3,4} — the
///    K=2 ring-buffer matures decisions two stages after they are made. With
///    K=2, the decision written at stage t occupies slot `K-1 = 1` in the
///    outgoing state, which shifts into slot 0 after two forward steps, at
///    which point the fishing equality delivers it. This is the t-2 offset
///    (compare: K=1 delivery test uses t-1 offset).
///
/// 4. Saturate `decision_at(t) ≈ 200 MW` (max_gen) for t ∈ {0,1,2} (stages
///    where `t + K < n_stages`, i.e., `t + 2 < 5`) — the anticipated thermal
///    costs $10/MWh vs the backup's $5000/MWh, and the per-block cost of the
///    decision column is non-zero only at the decision stage, so the LP
///    saturates commitment to avoid future backup dispatch.
///
/// 5. Satisfy the analytical cost bound:
///    - Stage 0: seed delivers 80 MW; backup covers 70 MW
///      × $5000/MWh × 744 h = $260,400,000.
///    - Stage 1: shifted seed delivers 50 MW; backup covers 100 MW
///      × $5000/MWh × 744 h = $372,000,000.
///    - Stages 2–4 delivery: anticipated covers ≥ 150 MW load (zeroed cost).
///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
///    - Total ≤ $636,865,000.
#[test]
fn pre_horizon_seed_delivers_pre_horizon_stages_k2() {
    // Cost bound: see this test's doc comment. Tolerance matches
    // anticipated_numerical_reconciliation_k2.
    const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 80.0) * 744.0 * 5000.0;
    const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
    const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
        + STAGE_1_BACKUP_COST_USD
        + MAX_DECISION_COST_USD
        + COST_TOLERANCE_USD;

    let system = build_system(&FIXTURE_K2);
    let config = build_config(FIXTURE_K2.iterations);
    let mut setup = build_setup_in_code(system, &config);

    // 5 iterations: after 1, stage-1 decisions are too loose to satisfy the AC5
    // cost bound.
    let scenario_results = run_simulation(&mut setup, FIXTURE_K2.iterations);

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

    let anticipated_thermal_id: i32 = 42;
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

    // ── AC1: committed_at(0) == Some(80.0) within 1e-6 MW ──────────────────
    let c0 = committed_at(0).expect(
        "AC1 FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 80 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
    );
    assert!(
        (c0 - 80.0).abs() < 1e-6,
        "AC1 FAIL: committed_at(0) = {c0} MW, expected 80.0 MW (values_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 80.0 MW of the ring buffer.",
    );

    // ── AC2: committed_at(1) == Some(50.0) within 1e-6 MW ──────────────────
    // (K=2-specific: tests ring-buffer shift between pre-horizon stages 0→1)
    let c1 = committed_at(1).expect(
        "AC2 FAIL: committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
    assert!(
        (c1 - 50.0).abs() < 1e-6,
        "AC2 FAIL: committed_at(1) = {c1} MW, expected 50.0 MW (values_mw[1]). \
         `shift_anticipated_state` (noise.rs:253) must move slot 1 (50.0 MW) \
         into slot 0 at the start of stage 1, and the fishing equality must read \
         that value. If the result is 80.0 MW, the ring-buffer shift is not \
         moving slot 1 into slot 0 between pre-horizon stages.",
    );

    // ── AC3: committed_at(t) ≈ decision_at(t-2) for t ∈ {2,3,4} ───────────
    // (K=2 ring-buffer: decisions mature 2 stages after being made)
    for t in 2..5_usize {
        let ct = committed_at(t).unwrap_or_else(|| {
            panic!(
                "AC3 FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                t - 2,
            )
        });
        let d_prev2 = decision_at(t - 2).unwrap_or_else(|| {
            panic!(
                "AC3 FAIL: decision_at({}) is None (needed to check K=2 \
                 ring-buffer invariant at stage {t})",
                t - 2,
            )
        });
        assert!(
            (ct - d_prev2).abs() < 1e-6,
            "AC3 FAIL (K=2 ring-buffer shift): committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev2} MW (within 1e-6 MW). \
             With K=2, decisions mature two stages later (slot K-1=1 shifts into \
             slot 0 after two forward steps). The ring buffer is not correctly \
             propagating in-study decisions.",
            t - 2,
        );
    }

    // ── AC4: decision_at(t) non-zero and bounded for t ∈ {0,1,2} ───────────
    // (Active-decision stages: t + 2 < 5; LP saturates on cost ratio)
    for t in 0..3_usize {
        let dt = decision_at(t).unwrap_or_else(|| {
            panic!(
                "AC4 FAIL: decision_at({t}) is None; anticipated thermal id=42 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 2 < 5)",
            )
        });
        assert!(
            dt.abs() > 1e-6,
            "AC4 FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
            t + 2,
        );
        assert!(
            dt <= 200.0 + 1e-6,
            "AC4 FAIL: decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
        );
    }

    // ── AC5: observed_total ≤ EXPECTED_TOTAL_UPPER_BOUND_USD ────────────────
    // (Use immediate_cost, not total_cost which includes theta approximation artefact.
    //  If no seeds delivered: 2 × 150 MW × 744 h × $5000 = $1.116B >> bound → fails)
    let observed_total: f64 = scenario
        .stages
        .iter()
        .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
        .sum();

    assert!(
        observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
        "AC5 FAIL: observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If neither seed is delivered, stage-0+1 backup covers 300 MW total \
         producing ~$1,116M >> $636.865M.",
    );
}

/// Pre-horizon seed delivery across three pre-horizon stages with K=3 and the
/// always-active fishing predicate.
///
/// With `K = 3`, `n_stages = 6`, a single anticipated thermal (id=52), and
/// `past_anticipated_commitments.values_mw = [50.0, 30.0, 10.0]`, the LP must:
///
/// 1. Deliver `committed_at(0) == 50.0 MW` — the always-active fishing
///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
///    buffer, which holds the 50.0 MW seed (`values_mw[0]`). The cost-zeroing
///    predicate zeros the per-block objective for this column so the LP
///    accepts the delivery at zero additional cost.
///
/// 2. Deliver `committed_at(1) == 30.0 MW` — `shift_anticipated_state`
///    moves slot 1 (`values_mw[1] = 30.0`) into slot 0 at
///    the start of stage 1. Stage 1's always-active fishing equality then
///    reads slot 0 = 30.0 MW. This is one of the two K=3-specific assertions
///    that the K=1 and K=2 delivery tests cannot reach: K=3 has three
///    pre-horizon stages, with two ring-buffer shifts between them.
///
/// 3. Deliver `committed_at(2) == 10.0 MW` — after two ring-buffer shifts,
///    slot 0 holds `values_mw[2] = 10.0`. Stage 2's always-active fishing
///    equality delivers it at zero LP cost. This is the deepest pre-horizon
///    delivery assertion in the entire anticipated test suite.
///
/// 4. Satisfy `committed_at(t) ≈ decision_at(t-3)` for t ∈ {3, 4, 5} — the
///    K=3 ring-buffer matures decisions three stages after they are committed.
///    With K=3, the decision written at stage t occupies slot `K-1 = 2` in
///    the outgoing state, which shifts into slot 0 after three forward steps,
///    at which point the fishing equality delivers it. This is the t-3 offset
///    (compare: K=1 uses t-1, K=2 uses t-2).
///
/// 5. Saturate `decision_at(t) > 0` and `≤ max_gen + 1e-6` for t ∈ {0, 1, 2}
///    (stages where `t + K < n_stages`, i.e., `t + 3 < 6`, giving t ∈ {0,1,2})
///    — the anticipated thermal costs $10/MWh vs the backup's $5000/MWh, and
///    the per-block cost of the decision column is non-zero only at the
///    decision stage, so the LP commits a non-trivial amount to avoid future
///    backup dispatch.
///
/// 6. Satisfy the analytical cost bound:
///    - Stage 0: seed delivers 50 MW; backup covers 100 MW
///      × $5000/MWh × 744 h = $372,000,000.
///    - Stage 1: shifted seed delivers 30 MW; backup covers 120 MW
///      × $5000/MWh × 744 h = $446,400,000.
///    - Stage 2: doubly-shifted seed delivers 10 MW; backup covers 140 MW
///      × $5000/MWh × 744 h = $520,800,000.
///    - Stages 3–5 delivery: anticipated delivers ≥ 150 MW load (zeroed cost).
///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
///    - Tolerance: $1,000.
///    - Total upper bound: $1,343,665,000.
#[test]
fn pre_horizon_seed_delivers_three_pre_horizon_stages_k3() {
    // Cost bound: see this test's doc comment. Tolerance matches
    // anticipated_numerical_reconciliation_k2.
    const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
    const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 30.0) * 744.0 * 5000.0;
    const STAGE_2_BACKUP_COST_USD: f64 = (150.0 - 10.0) * 744.0 * 5000.0;
    const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
        + STAGE_1_BACKUP_COST_USD
        + STAGE_2_BACKUP_COST_USD
        + MAX_DECISION_COST_USD
        + COST_TOLERANCE_USD;

    let system = build_system(&FIXTURE_K3);
    let config = build_config(FIXTURE_K3.iterations);
    let mut setup = build_setup_in_code(system, &config);

    // 5 iterations let cuts sharpen so stage-2 decisions reach max_gen, covering
    // stage-5 delivery at zero backup cost (the AC6 bound).
    let scenario_results = run_simulation(&mut setup, FIXTURE_K3.iterations);

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
    // Sum immediate_cost, NOT total_cost — total_cost includes the theta
    // approximation artefact that would break this bound.
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
