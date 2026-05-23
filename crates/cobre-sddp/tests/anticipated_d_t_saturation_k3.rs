//! Regression test: `d_t` must saturate at `max_generation_mw` for every
//! active anticipated-decision stage in a K=3 fixture.
//!
//! ## Bug being guarded against
//!
//! In an 8-stage K=3 study with a 500x cost asymmetry (anticipated thermal at
//! $10/MWh vs backup at $5000/MWh), the LP should commit the anticipated
//! thermal to its maximum (`200 MW`) at every stage `t` where `t + K < n_stages`
//! (i.e. `t in {0, 1, 2, 3, 4}`), because over-commitment is free (excess
//! generation cost = $0) while any backup dispatch is ruinously expensive.
//!
//! The K=3 propagation chain is longer than K=2: cuts at stage `t` propagate
//! to predecessor's slot 1, which is `state_col[slot 2]`, whose value at K=3
//! is `incoming_slot_2 - d_{t-1}` if the decision-write coefficient is at
//! slot K-1 = 2. This variant exercises the multi-step propagation through
//! all three slot positions, confirming the fix is general and not K=2-specific.
//!
//! ## What this test asserts
//!
//! After training a deterministic single-scenario simulation for 15 iterations:
//!
//! - For `t in {0, 1, 2, 3, 4}`: `anticipated_decision_mw` exists (is `Some`) and
//!   saturates at `200.0 ± 1e-3 MW`.
//! - For `t in {5, 6, 7}`: `anticipated_decision_mw` is `None` (the
//!   strict-boundary predicate `t + K < n_stages` excludes these).
//! - For `t in {3, 4, 5, 6, 7}` (stages where `t >= K`): the matured
//!   `anticipated_committed_mw` at stage `t` equals the decision made K=3
//!   stages earlier, i.e. `committed_at(t) ≈ decision_at(t - 3)`.
//!
//! This test is gated behind `#[ignore]` until the layout fix in Epic 03
//! ticket-009 lands. It will transition from ignored to passing once that
//! fix corrects the column-index corruption.

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
use cobre_sddp::{hydro_models::PrepareHydroModelsResult, StudySetup};
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::{build_stochastic_context, ClassSchemes, OpeningTreeInputs};

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

/// Build an 8-stage K=3 system with:
/// - 1 bus (deficit cost 5000 $/MWh, excess cost $0)
/// - 1 trivial hydro (1 hm³ max storage, zero inflow, max_gen 1 MW) — keeps
///   the model in the thermal regime without adding a hydro state variable
///   that complicates interpretation.
/// - 1 anticipated thermal (K=3, cost 10 $/MWh, max 200 MW) — id=3
/// - 1 backup thermal (cost 5000 $/MWh, max 500 MW) — id=4
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments = [(id=3, [0.0, 0.0, 0.0])]` — zero seeds
///   so the test isolates the in-horizon bug, not any seeding artefact.
///
/// The 500x cost ratio (10 vs 5000) makes it overwhelmingly optimal to
/// saturate anticipated dispatch at every active stage (`t + K < n_stages`).
/// Over-generation is free (excess_cost = $0), so the LP has no incentive
/// to cap anticipated commits below `max_generation_mw = 200`.
///
/// IDs 3 and 4 are chosen deliberately to differ from the K=2 sibling test
/// (which uses IDs 2 and 4) so that combined nextest runs produce
/// unambiguous per-entity attribution in failure messages.
fn build_system_k3() -> cobre_core::System {
    use chrono::NaiveDate;

    let k: usize = 3;
    let n_stages: usize = 8;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 5000.0,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K=3 lead stages, very cheap so the policy
    // saturates anticipated dispatch to max_generation_mw.
    let anticipated_id = EntityId(3);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant_k3".to_string(),
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
        name: "T_backup_k3".to_string(),
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
        id: EntityId(5),
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
            hydro_id: EntityId(5),
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
            hydro_id: EntityId(5),
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![0.0, 0.0, 0.0],
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
        .expect("build_system_k3: valid system")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a [`Config`] for 15-iteration training and 1-scenario deterministic
/// simulation.
///
/// Fifteen iterations is sufficient to expose the bug at K=3: with a 500x
/// cost ratio and a three-slot propagation chain, any policy that even
/// partially explores the anticipated commitment creates cuts whose gradients
/// at the anticipated-state columns reveal the corruption for `t >= 1`.
/// K=3 needs one extra step for cut propagation through three slot positions
/// versus K=2; 15 is a safe upper bound.
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 15 }]),
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

/// Assert that `anticipated_decision_mw` saturates at `max_generation_mw`
/// for every active decision stage in a K=3, 8-stage fixture, and that the
/// ring-buffer shift correctly propagates each decision to its delivery stage.
///
/// With the anticipated thermal at $10/MWh and the backup at $5000/MWh,
/// the value of pre-committing 200 MW (max) at every stage `t` where
/// `t + K < n_stages` is enormous compared to relying on the backup. The
/// excess generation cost is $0, so the LP has no reason to cap below 200 MW.
///
/// **Expected behaviour (post-fix)**:
/// - `d_t ≈ 200.0` for `t in {0, 1, 2, 3, 4}`.
/// - `anticipated_decision_mw` is `None` for `t in {5, 6, 7}`.
/// - `committed_at(t) ≈ decision_at(t - 3)` for `t in {3, 4, 5, 6, 7}`.
///
/// **Observed behaviour (current HEAD)**:
/// - `d_0 ≈ 200.0` but `d_1 = d_2 = d_3 = d_4 ≈ 0`.
/// - The ring-buffer shift assertion may also fail if committed values
///   do not track the zero-valued decisions.
///
/// The cut coefficients for the anticipated-state columns are corrupted for
/// `k >= 2` in `state_to_lp_column`'s `Less` branch, so the policy at those
/// stages receives no incentive to commit. At K=3 the corruption propagates
/// through all three slot positions, making it a stricter test than K=2.
#[test]
#[ignore = "fails until Epic 03 ticket-009 lands the layout fix"]
fn d_t_saturates_at_max_gen_for_every_active_stage_k3() {
    let k: usize = 3;
    let n_stages: usize = 8;
    // Active decision stages: t + K < n_stages  =>  t in {0, 1, 2, 3, 4}.
    let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
    // Inactive (boundary) stages where the strict predicate excludes decision.
    let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();
    // Stages where the ring buffer has fully matured: t >= K => t in {3..7}.
    let committed_stages: Vec<usize> = (0..n_stages).filter(|&t| t >= k).collect();

    let system = build_system_k3();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new: must succeed");

    // Train the policy for 15 iterations.
    let outcome = setup
        .train(&mut solver, &comm, 15, HighsSolver::new, None, None)
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

    // The anticipated thermal has entity id=3 (see build_system_k3).
    let anticipated_thermal_id: i32 = 3;
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

    // ── Active stages: decision must exist and saturate at max_generation_mw ──
    //
    // With the strict-boundary predicate `t + K < n_stages`, stages 0..4
    // are active. The 500x cost asymmetry and zero excess cost make
    // d_t = 200 overwhelmingly optimal at every active stage. On the current
    // HEAD the cut-coefficient bug causes d_t = 0 for t in {1, 2, 3, 4}.
    let max_gen_mw = 200.0_f64;
    let tol = 1e-3_f64;
    for t in &active_stages {
        let d_t = decision_at(*t).unwrap_or_else(|| {
            panic!(
                "anticipated_decision_mw must be Some at active stage t={t} \
                 (t + K < n_stages, K=3)",
            )
        });
        assert!(
            (d_t - max_gen_mw).abs() < tol,
            "d_t at stage {t} must saturate at max_generation_mw={max_gen_mw}: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Current HEAD produces d_t ≈ 0 for t >= 1 due to \
             cut-coefficient corruption in state_to_lp_column (Less branch). \
             At K=3 the corruption spans all three slot positions.",
            delta = (d_t - max_gen_mw).abs(),
        );
    }

    // ── Inactive stages: decision must be None (strict-boundary predicate) ──
    //
    // For t in {5, 6, 7}: t + K >= n_stages, so the anticipated-decision
    // variable does not exist in the LP. The simulation must report None.
    for t in &inactive_stages {
        assert!(
            decision_at(*t).is_none(),
            "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages, K=3; strict-boundary predicate excludes this stage)",
        );
    }

    // ── Ring-buffer shift: committed_at(t) == decision_at(t - K) ──
    //
    // For t in {3, 4, 5, 6, 7} (all stages where t >= K=3), the matured
    // commitment `anticipated_committed_mw` at stage t must equal the
    // decision `anticipated_decision_mw` that was made K=3 stages earlier.
    //
    // Trace for post-fix code:
    //   - The ring-buffer shift at end-of-stage-(t-3) places d_{t-3} into
    //     slot K-1=2. After two more shifts (end of stage t-2 and t-1) it
    //     reaches slot 0, where the fishing constraint reads it at stage t.
    //     So committed_at(t) = d_{t-3}.
    //
    // Trace for buggy code path:
    //   - On current HEAD the ring buffer faithfully propagates the bug's
    //     d_{t-3} ≈ 0 into committed_at(t), so the loop passes trivially
    //     (|0 - 0| = 0 < tol). The saturation assertions above are the
    //     sole pre-fix change-detectors. This loop's job is post-fix
    //     regression coverage: it would catch a future bug where d_t is
    //     correct but propagation breaks.
    for t in &committed_stages {
        let c_t = committed_at(*t).unwrap_or_else(|| {
            panic!(
                "anticipated_committed_mw must be Some at stage t={t} \
                 (t >= K=3; fishing constraint is active)",
            )
        });
        let d_prev = decision_at(*t - k).unwrap_or_else(|| {
            panic!(
                "anticipated_decision_mw must be Some at stage t-K={prev} \
                 (used to verify ring-buffer shift at delivery stage t={t})",
                prev = *t - k,
            )
        });
        assert!(
            (c_t - d_prev).abs() < tol,
            "ring-buffer shift invariant violated at t={t}: \
             committed_at({t}) = {c_t:.6} MW but decision_at({prev}) = {d_prev:.6} MW \
             (delta = {delta:.6} MW, tol = {tol} MW, K=3). \
             The ring buffer should propagate d_{{t-3}} through three slot \
             positions so it reaches slot 0 at stage t.",
            prev = *t - k,
            delta = (c_t - d_prev).abs(),
        );
    }
}
