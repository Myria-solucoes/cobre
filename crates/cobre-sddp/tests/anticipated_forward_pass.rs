//! End-to-end integration test verifying the anticipated-state ring-buffer
//! evolution across the full forward pass for a 5-stage K=2 system.
//!
//! The ring-buffer shift semantics: at the end of stage `t`,
//! `shift_anticipated_state` (`crates/cobre-sddp/src/noise.rs`) shifts each
//! plant's slots down (`slot[s] <- incoming[s+1]`) and writes the new
//! decision into the highest slot (`slot[K-1] <- decision_primal`). Slot 0
//! at stage `t+1` therefore equals slot 1 at stage `t`. The basis-cache
//! capture sequence across forward + backward passes is described inline
//! in the per-stage assertions below.

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

/// Build a 5-stage system with:
/// - 1 bus (deficit cost 500 $/MWh)
/// - 1 hydro (storage 200 hm³, max_gen 250 MW, inflow mean 80 m³/s)
/// - 1 anticipated thermal (K=2, cost 50 $/MWh, max_gen 100 MW) — id=2
/// - 1 backup standard thermal (cost 500 $/MWh, max_gen 200 MW) — id=4
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments = [(id=2, [100.0, 50.0])]`
///
/// The LP is always feasible: backup thermal alone covers 150 MW.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K=2 lead stages, cost 50 $/MWh, max 100 MW.
    let anticipated_id = EntityId(2);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        cost_per_mwh: 50.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Backup standard thermal: cost 500 $/MWh (high, always feasible).
    let thermal_backup = Thermal {
        id: EntityId(4),
        name: "T_backup".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 500.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Hydro: constant-productivity, large storage so spill is unlikely.
    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        bus_id: EntityId(1),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
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

    let n_stages = 5_usize;
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
            mean_m3s: 80.0,
            std_m3s: 20.0,
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

    // k_max=2 for the anticipated thermal bounds layout.
    let k_max: usize = 2;
    let n_st = n_stages;

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
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

    // 2 thermals (anticipated + backup), k_max=2.
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
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

    // Seed the anticipated ring buffer: slot 0 = 100.0 MW, slot 1 = 50.0 MW.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![100.0, 50.0],
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

/// Build a minimal [`Config`] for 1 forward pass and 1 iteration.
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
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

// ---------------------------------------------------------------------------
// Setup builder
// ---------------------------------------------------------------------------

/// Construct a [`StudySetup`] in-process from the given system and config.
///
/// Uses [`build_stochastic_context`] directly (no external scenario files),
/// which keeps the test hermetic.
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
// Integration test
// ---------------------------------------------------------------------------

/// Verify that the anticipated-state ring buffer evolves correctly across all
/// 5 stages of the forward pass.
///
/// ## Ring-buffer layout and capture semantics
///
/// The layout is slot-major, plant-minor.  With `n_anticipated=1` and `k_max=2`
/// the anticipated block is `[slot0_plant0, slot1_plant0]`.
///
/// `CapturedBasis::state_at_capture` is populated by two separate code paths:
///
/// - **Stage 0**: the forward pass writes the post-shift outgoing state of
///   stage 0.  With initial seed `[100.0, 50.0]` and decision `d_0`, the
///   outgoing state is `[50.0, d_0]`.
///
/// - **Stages 1..=4**: the backward pass overwrites each slot.  For stage `s`,
///   the backward pass exchanges the forward outgoing state from stage `s-1`
///   (used as the linearisation trial point `x_hat`) and stores that state
///   in `basis_store[(m, s)]`.  Concretely:
///
///   | basis_cache slot | stored by        | value equals              |
///   |-----------------|------------------|---------------------------|
///   | 0               | forward pass     | forward outgoing of t=0   |
///   | 1               | backward pass    | forward outgoing of t=0   |
///   | 2               | backward pass    | forward outgoing of t=1   |
///   | t (≥1)          | backward pass    | forward outgoing of t-1   |
///
///   This means `basis_cache[1]` and `basis_cache[0]` carry the same
///   anticipated-state values (both equal the post-shift state of stage 0).
///
/// ## Invariants tested
///
/// - **AC-2**: `basis_cache[0][ant_start] == 50.0` — seed slot 1 shifted into
///   slot 0 of the outgoing state.
/// - **AC-3**: `basis_cache[1][ant_start..ant_start+2]` equals
///   `basis_cache[0][ant_start..ant_start+2]` — the backward pass for stage 1
///   stores the same trial point as the forward outgoing of stage 0.
/// - **AC-4**: for `t` in `2..5`, `basis_cache[t][ant_start]` equals
///   `basis_cache[t-1][ant_start+1]` — the backward-to-backward shift
///   invariant: each slot 0 comes from the previous capture's slot 1.
/// - **AC-5**: every `basis_cache[t][ant_start+1]` lies in `[0.0, 100.0]`.
#[test]
fn five_stage_k2_anticipated_state_ring_buffer_evolution() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    // Run 1 training iteration — enough to populate basis_cache for all stages.
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");

    // -----------------------------------------------------------------------
    // AC-1: training completes without error and runs at least 1 iteration.
    // -----------------------------------------------------------------------
    assert!(
        outcome.error.is_none(),
        "AC-1: training error must be None; got {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations >= 1,
        "AC-1: at least 1 iteration must complete; got {}",
        outcome.result.iterations
    );

    // Retrieve the stage indexer to locate the anticipated_state slice.
    let indexer = setup.stage_indexer();
    let n_ant = indexer.n_anticipated;
    let k_max = indexer.k_max;
    assert_eq!(n_ant, 1, "fixture must have exactly 1 anticipated thermal");
    assert_eq!(k_max, 2, "fixture must have k_max = 2");

    let ant_start = indexer.anticipated_state.start;

    let basis_cache = &outcome.result.basis_cache;
    assert_eq!(
        basis_cache.len(),
        5,
        "basis_cache must have one entry per study stage"
    );

    // -----------------------------------------------------------------------
    // AC-2: stage 0 state_at_capture reflects the seeded past_anticipated_commitments.
    //
    // The forward pass stores the POST-SHIFT outgoing state at stage 0.
    // Incoming seed is [100.0, 50.0]; after the ring-buffer shift:
    //   slot 0 = seeded slot 1 = 50.0   (shifts down by one)
    //   slot 1 = LP decision d_0 ∈ [0, 100]
    //
    // The 50.0 in slot 0 is direct evidence that the seed was consumed:
    // no other code path produces this exact value in the anticipated block.
    // -----------------------------------------------------------------------
    let s0 = basis_cache[0]
        .as_ref()
        .expect("AC-2: stage 0 basis must be captured")
        .state_at_capture
        .as_slice();
    assert!(
        (s0[ant_start] - 50.0).abs() < 1e-9,
        "AC-2: stage 0 slot 0 must equal seeded slot 1 = 50.0; got {}",
        s0[ant_start]
    );
    assert!(
        (-0.01..=100.01).contains(&s0[ant_start + 1]),
        "AC-2: stage 0 slot 1 (= d_0) must lie in [0, 100]; got {}",
        s0[ant_start + 1]
    );

    // -----------------------------------------------------------------------
    // AC-3: stage 1 state_at_capture equals stage 0 state_at_capture.
    //
    // The backward pass for stage 1 uses the forward outgoing state of stage 0
    // as its trial-point x_hat and writes it into basis_cache[1].  This is
    // identical to basis_cache[0] (both hold the post-shift state of stage 0).
    //
    // Invariant: basis_cache[1][ant_start..ant_start+2]
    //            == basis_cache[0][ant_start..ant_start+2]
    // -----------------------------------------------------------------------
    let s1 = basis_cache[1]
        .as_ref()
        .expect("AC-3: stage 1 basis must be captured")
        .state_at_capture
        .as_slice();
    assert!(
        (s1[ant_start] - s0[ant_start]).abs() < 1e-9,
        "AC-3: stage 1 slot 0 ({}) must equal stage 0 slot 0 ({}) — both carry \
         the post-shift outgoing state of forward stage 0",
        s1[ant_start],
        s0[ant_start],
    );
    assert!(
        (s1[ant_start + 1] - s0[ant_start + 1]).abs() < 1e-9,
        "AC-3: stage 1 slot 1 ({}) must equal stage 0 slot 1 ({}) — both carry \
         d_0 from the forward pass",
        s1[ant_start + 1],
        s0[ant_start + 1],
    );

    // -----------------------------------------------------------------------
    // AC-4: for t in 2..=4, slot 0 at stage t equals slot 1 at stage t-1
    //        (the backward-pass shift invariant across consecutive stages).
    //
    // basis_cache[t] for t≥2 stores the forward outgoing state of stage t-1.
    // basis_cache[t-1] stores the forward outgoing state of stage t-2.
    // The forward shift produces:
    //   outgoing_of(t-1)[slot0] = outgoing_of(t-2)[slot1]
    // which is exactly s_curr[ant_start] == s_prev[ant_start+1].
    // -----------------------------------------------------------------------
    for t in 2..5_usize {
        let s_curr = basis_cache[t]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-4: stage {t} basis must be captured"))
            .state_at_capture
            .as_slice();
        let s_prev = basis_cache[t - 1]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-4: stage {} basis must be captured", t - 1))
            .state_at_capture
            .as_slice();

        assert!(
            (s_curr[ant_start] - s_prev[ant_start + 1]).abs() < 1e-9,
            "AC-4: stage {t} slot 0 ({}) must equal stage {} slot 1 ({})",
            s_curr[ant_start],
            t - 1,
            s_prev[ant_start + 1],
        );
    }

    // -----------------------------------------------------------------------
    // AC-5: every captured slot 1 (the anticipated decision field) lies
    //        within the anticipated thermal's dispatch bounds [0.0, 100.0].
    // -----------------------------------------------------------------------
    for t in 0..5_usize {
        let s_t = basis_cache[t]
            .as_ref()
            .unwrap_or_else(|| panic!("AC-5: stage {t} basis must be captured"))
            .state_at_capture
            .as_slice();
        let decision = s_t[ant_start + 1];
        assert!(
            (-0.01..=100.01).contains(&decision),
            "AC-5: stage {t} slot 1 must lie in [0, 100]; got {decision}",
        );
    }
}
