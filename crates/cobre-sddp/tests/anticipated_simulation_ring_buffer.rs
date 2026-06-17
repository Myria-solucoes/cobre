//! Regression test: simulation pipeline must shift the anticipated-state
//! ring buffer between stages.
//!
//! ## Bug being guarded against
//!
//! Prior to the fix, `solve_simulation_stage` in
//! `crates/cobre-sddp/src/simulation/pipeline.rs` advanced the inflow-lag
//! ring buffer between stages but never called `shift_anticipated_state`.
//! Because the `anticipated_state` LP columns are unbounded
//! `[-inf, +inf]` and the Category 6 fixing rows enforce
//! `state_col + decision_col = incoming_state`, the post-solve primal
//! value at the `anticipated_state` columns equals `incoming - decision`
//! (a residual that has no physical meaning) rather than the shifted
//! ring-buffer state.
//!
//! Without the shift, this corrupted value propagated to the next
//! stage's `ws.current_state`, and the fishing constraint at the
//! delivery stage `K` read slot 0 of a never-shifted buffer. The
//! `anticipated_committed_mw` output therefore reflected stale or zero
//! values rather than the in-study decisions.
//!
//! ## What this test asserts
//!
//! The fixture deterministically pins the anticipated decision at each
//! stage to a non-trivial value by making the anticipated thermal much
//! cheaper than the backup thermal. With a deterministic single-scenario
//! simulation, every stage's `anticipated_decision_mw` is well-defined,
//! and the post-shift ring-buffer evolution implies:
//!
//! `anticipated_committed_mw(t = K) == anticipated_decision_mw(t = 0)`
//!
//! i.e. the freshly-committed MW at stage 0 must equal the matured
//! commitment at stage K (where K is the lead time).
//!
//! On the buggy code this invariant fails because the ring buffer never
//! advances; the fishing constraint at stage K reads the seeded
//! `past_anticipated_commitments` (or zero) rather than the in-study
//! decision.

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

/// Build a deterministic K-stage system with:
/// - 1 bus (deficit cost 5000 $/MWh)
/// - 1 hydro (small capacity, mostly inactive)
/// - 1 anticipated thermal (K lead stages, cost 10 $/MWh, max 200 MW) — id=2
/// - 1 backup thermal (cost 5000 $/MWh, max 500 MW) — id=4
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments` seeded with the given `past_commitments_mw`
///
/// **Note on non-zero seeds**: this function constructs the resolved
/// `cobre_core::System` directly via `SystemBuilder::new()`, bypassing the
/// `cobre-io` parse-and-validate pipeline. The semantic validator that rejects
/// non-zero `values_mw` entries therefore does NOT fire here. That is
/// intentional: the non-zero seed is a deliberate test fixture for the
/// ring-buffer shift mechanic (see the test doc below), not a user-supplied
/// pre-horizon commitment. The validator's rejection rule applies to JSON input
/// through `load_case`; unit tests that construct `System` directly are exempt.
///
/// With K=1 and cheap anticipated cost, the optimal policy commits the full
/// load (150 MW) one stage in advance from stage 0 onward, so:
/// - `anticipated_decision_mw(t)` should converge to 150 for t in 0..n-1
/// - `anticipated_committed_mw(t)` at stage t==K must equal the decision
///   made at stage 0 (a fresh in-study commitment), NOT any seeded value.
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
    // Per-thermal cost overrides: ResolvedBounds::new uses one default
    // for ALL thermals, but the LP must distinguish the cheap anticipated
    // thermal (thermal index 0) from the expensive backup (thermal index
    // 1). Without these per-thermal costs, the policy has no reason to
    // commit anticipated capacity and decision_at(t) collapses to zero —
    // which would obscure the regression assertion.
    //
    // The padding region `[n_stages, n_stages + k_max)` is the delivery-
    // stage axis read by `fill_anticipated_decision_objective`; it must
    // also carry the per-thermal cost so the decision column has a
    // non-zero objective coefficient.
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

/// Verify that the simulation forward walk advances the anticipated-state
/// ring buffer once per stage.
///
/// With K=1 and a non-zero seed `past_anticipated_commitments = [50.0]`,
/// the LP at stage 1 sees a fishing constraint whose RHS comes from slot 0
/// of the anticipated-state vector at the start of stage 1. After the
/// ring-buffer shift performed at the end of stage 0, slot 0 must hold
/// the LP's stage-0 decision (the freshly-made commitment) — not the
/// seeded 50.0 and not the residual `incoming - decision` value left in
/// the unbounded `anticipated_state` columns.
///
/// Concretely: with the anticipated thermal at $10/MWh and the backup
/// thermal at $5000/MWh, the optimal policy is to commit the full 150 MW
/// of load via the anticipated thermal at every active stage. Therefore:
///
/// 1. `anticipated_decision_mw(t = 0)` saturates near 150 MW.
/// 2. `anticipated_committed_mw(t = 1)` equals the decision from t = 0
///    (150 MW), NOT the seed 50.0.
///
/// On the buggy code path (no shift), `anticipated_committed_mw(t = 1)`
/// would equal the seed 50.0 (because slot 0 was never advanced), and the
/// equality assertion `committed(1) == decision(0)` would fail.
#[test]
fn simulation_ring_buffer_shifts_anticipated_state_k1() {
    let k: usize = 1;
    let n_stages: usize = 5;
    // Use a seed value the LP is structurally unlikely to reproduce
    // as a decision. Setting the seed to 7 MW (a value with no special
    // relationship to load/bounds) maximises the chance that
    // decision_at(t) != seed[0]. Even if the LP happens to also pick
    // d_t = 7, the relationship would be coincidental rather than a
    // fixed-point of the buggy code.
    let seed: Vec<f64> = vec![7.0];

    let system = build_system(k, seed.clone(), n_stages);
    let config = build_config(50);
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    // Train the policy.
    let outcome = setup
        .train(&mut solver, &comm, 50, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    // Run the simulation (1 deterministic scenario).
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

    // Locate the anticipated thermal (id=2) in each stage's thermal vec.
    // Helper: pull the anticipated decision and committed values for a stage.
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

    // ── Sanity: stage 0 has a decision; under always-active fishing the
    // committed value reads the seed at slot 0 (`values_mw[0] = 7.0`).
    let d0 =
        decision_at(0).expect("anticipated_decision_mw must exist at stage 0 (t + K < n_stages)");
    let c0 = committed_at(0)
        .expect("anticipated_committed_mw must be Some at stage 0 under always-active fishing");
    assert!(
        (c0 - 7.0).abs() < 1e-6,
        "committed_at(0) must equal the K=1 seed values_mw[0]=7.0; got {c0}",
    );
    // With the cheap anticipated thermal (cost 10 $/MWh) saving expensive
    // backup dispatch (5000 $/MWh) at the K=1 delivery stage, any policy
    // that even partially explores the trade-off commits a non-trivial
    // amount at stage 0.
    assert!(
        d0.abs() > 1e-6,
        "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
    );

    // ── Regression assertion: stage-1 matured commitment must equal d_0 ──
    //
    // Trace:
    //
    // - With the FIX: end-of-stage-0 outgoing state slot 0 = d_0 (the
    //   shift writes d_0 into slot 0). Stage 1 sees Cat 6 RHS = d_0 > 0.
    //   The fishing constraint `gt_anticipated = state_col` paired with
    //   Cat 6 `state_col + d_1 = d_0` and `state_col >= 0` gives
    //   `gt_anticipated_stage_1 = d_0 - d_1`. With cheap thermal cost
    //   zeroed at delivery, the LP picks d_1 = 0 and gt = d_0.
    //
    // - On the BUGGY code path: end-of-stage-0 outgoing state slot 0 =
    //   the LP primal of the unbounded `anticipated_state` column, which
    //   equals `seed[0] - d_0`. With seed[0]=7 and d_0=100, the residual
    //   is -93. Stage 1's Cat 6 RHS = -93. Combined with state_col >= 0
    //   from fishing, the LP is INFEASIBLE: state_col + d_1 = -93 cannot
    //   be satisfied with both state_col >= 0 and d_1 >= 0. The
    //   simulation returns `Err(LpInfeasible{stage_id: 1, ..})`.
    //
    // So on the fixed path: simulation succeeds AND committed_at(1) == d_0.
    // On the buggy path: simulation errors out at stage 1.
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

/// Same invariant, K=2 case. With K=2 and a two-element seed
/// `[100.0, 50.0]`, the matured commitment at stage 0 is the seed slot 0
/// (100.0) and at stage 1 is the seed slot 1 (50.0). From stage 2 onward,
/// the matured commitment must equal the in-study decision made K=2
/// stages earlier — never the seed values, never zero.
#[test]
fn simulation_ring_buffer_shifts_anticipated_state_k2() {
    let k: usize = 2;
    let n_stages: usize = 6;
    // Pick seed values that the LP will NOT reproduce as `d_0`. With the
    // anticipated thermal max=100 and the LP preferring full commitment
    // when fishing is inactive at stage 0, d_0 saturates near 100 — which
    // is distinct from both seed slots (50 and 30 respectively).
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

    // With K=2 and always-active fishing, slot 0 of the ring buffer is
    // populated at every stage. Pre-horizon stages read seed values:
    //   stage 0 -> values_mw[0] = 50.0 (initial slot 0)
    //   stage 1 -> values_mw[1] = 30.0 (after the stage-0 ring-buffer shift)
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

    // The non-zero seed and the cheap anticipated thermal should produce
    // a non-trivial stage-0 commitment under any reasonable policy.
    assert!(
        d0.abs() > 1e-6,
        "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
    );

    // ── Regression assertion: stage-2 matured commitment must equal d_0 ──
    //
    // With K=2, the ring buffer shift at the end of stage 0 places d_0
    // in slot K-1=1, and the shift at the end of stage 1 moves it into
    // slot 0. Stage 2 then reads slot 0 (via the fishing constraint), so
    // committed_at(2) = d_0.
    //
    // On the buggy path the simulation pipeline never shifted, so stage 2
    // would read the stale residual from the unbounded LP column at
    // stage 1. With seed[1]=50 and stage-1 LP picking some state value,
    // the residual is typically negative (when d picks above seed) and
    // the LP at stage 2 becomes infeasible (Cat 6 RHS < 0 paired with
    // fishing forcing state_col >= 0).
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
