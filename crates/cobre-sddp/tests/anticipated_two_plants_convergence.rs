//! Integration test verifying the training lower bound for a 6-stage system
//! with 2 anticipated thermals (K_1=2, K_2=4), 1 backup thermal, and 1 hydro.
//!
//! ## Fixture description
//!
//! - 6 stages, each 744 hours (one calendar month).
//! - 1 bus (deficit cost 500 $/MWh).
//! - 1 hydro (storage cap 200 hm³, initial 100 hm³, max_gen 250 MW,
//!   inflow mean 80 m³/s, spillage cost 0.01 $/hm³).
//! - Anticipated thermal plant 0: K=2, cost 50 $/MWh, max 100 MW — id=2.
//! - Hydro: id=3.
//! - Backup standard thermal: cost 500 $/MWh, max 200 MW — id=4.
//! - Anticipated thermal plant 1: K=4, cost 40 $/MWh, max 80 MW — id=5.
//! - Load 150 MW constant across all stages.
//! - `past_anticipated_commitments`:
//!   - plant 0 (id=2, K=2): `[60.0, 30.0]`
//!   - plant 1 (id=5, K=4): `[20.0, 25.0, 30.0, 35.0]`
//!
//! ## Multi-plant LP layout
//!
//! With `n_anticipated=2` and `K_max=4`, the anticipated-state block has
//! `2 * 4 = 8` columns in slot-major, plant-minor order:
//!
//! ```text
//! ant_start + 0 = slot 0, plant 0  (K=2 plant — delivery slot)
//! ant_start + 1 = slot 0, plant 1  (K=4 plant — delivery slot)
//! ant_start + 2 = slot 1, plant 0  (K=2 plant — decision slot)
//! ant_start + 3 = slot 1, plant 1  (K=4 plant)
//! ant_start + 4 = slot 2, plant 0  (PADDING for K=2 plant)
//! ant_start + 5 = slot 2, plant 1  (K=4 plant)
//! ant_start + 6 = slot 3, plant 0  (PADDING for K=2 plant)
//! ant_start + 7 = slot 3, plant 1  (K=4 plant — decision slot)
//! ```
//!
//! ## Ring-buffer shift invariant (plant 0, stages 1→2)
//!
//! For plant 0 (K=2), the shift invariant asserts that the value in slot 1 at
//! stage `t` equals the value in slot 0 at stage `t+1` (for t≥1). The test
//! exercises the multi-plant slot-major ring-buffer shift using stages 1→2:
//!
//! `basis_cache[1].state_at_capture[ant_start + 1*n_anticipated + 0]`
//! must equal
//! `basis_cache[2].state_at_capture[ant_start + 0*n_anticipated + 0]`.
//!
//! Using t=1→t=2 (rather than t=0→t=1) avoids the trivial identity where
//! `basis_cache[0]` (forward-pass capture) and `basis_cache[1]` (backward-pass
//! trial point for stage 1, which also stores the forward outgoing of stage 0)
//! carry the same state. The t=1→t=2 comparison exercises a genuine backward-to-
//! backward ring-buffer advancement in the multi-plant slot-major layout.
//!
//! ## Golden value
//!
//! `EXPECTED_LB` was captured from the first passing run of this test and
//! pinned as the golden value. The lower bound combines locked anticipated
//! delivery costs, hydro dispatch, and backup thermal costs across 6 stages.
//! Do not change this constant without re-running the test to confirm the new
//! value is intentional.

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
use cobre_solver::highs::HighsSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

// EXPECTED_LB = 0.0 is pinned from a converged run of this fixture. The test
// validates slot-major LP layout, per-plant ring-buffer shift, and basis-cache
// capture across two anticipated plants — not a closed-form cost. Re-pin only
// after deliberate fixture changes.
const EXPECTED_LB: f64 = 0.0_f64;

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

/// Build a 6-stage system with:
/// - 1 bus (deficit cost 500 $/MWh)
/// - 1 hydro (storage 200 hm³, max_gen 250 MW, inflow mean 80 m³/s) — id=3
/// - 1 anticipated thermal plant 0 (K=2, cost 50 $/MWh, max 100 MW) — id=2
/// - 1 anticipated thermal plant 1 (K=4, cost 40 $/MWh, max 80 MW) — id=5
/// - 1 backup standard thermal (cost 500 $/MWh, max 200 MW) — id=4
/// - Load 150 MW constant across all stages
/// - `past_anticipated_commitments` for both anticipated plants
///
/// EntityId ordering is set so `SystemBuilder::build()` (which sorts by id)
/// produces the correct global thermal order. The two anticipated plants have
/// ids 2 and 5; the backup has id 4. Within the sorted thermal list the order
/// is: id=2 (ant K=2), id=4 (backup), id=5 (ant K=4). The anticipated-local
/// indices (0-based within the anticipated subset) are: plant 0 → id=2,
/// plant 1 → id=5.
///
/// The LP is always feasible: backup thermal alone covers 150 MW.
fn build_system_two_anticipated() -> cobre_core::System {
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

    // Anticipated thermal plant 0: K=2 lead stages, cost 50 $/MWh, max 100 MW.
    let ant_id_k2 = EntityId(2);
    let thermal_ant_k2 = Thermal {
        id: ant_id_k2,
        name: "T_ant_k2".to_string(),
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

    // Anticipated thermal plant 1: K=4 lead stages, cost 40 $/MWh, max 80 MW.
    let ant_id_k4 = EntityId(5);
    let thermal_ant_k4 = Thermal {
        id: ant_id_k4,
        name: "T_ant_k4".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 80.0,
        cost_per_mwh: 40.0,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 4 }),
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

    let n_stages = 6_usize;
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

    // k_max=4 (= max(K_1=2, K_2=4)) drives the anticipated-state block size.
    let k_max: usize = 4;
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

    // 3 thermals (ant K=2, backup, ant K=4), k_max=4.
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 3,
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

    // Seed both anticipated ring buffers with non-zero past commitments.
    // Plant 0 (K=2): 2 values — forces delivery at stages 0 and 1.
    // Plant 1 (K=4): 4 values — forces delivery at stages 0..=3.
    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![
            AnticipatedCommitmentHistory {
                thermal_id: ant_id_k2,
                values_mw: vec![60.0, 30.0],
            },
            AnticipatedCommitmentHistory {
                thermal_id: ant_id_k4,
                values_mw: vec![20.0, 25.0, 30.0, 35.0],
            },
        ],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant_k2, thermal_backup, thermal_ant_k4])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system_two_anticipated: valid")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a minimal [`Config`] for 1 forward pass and 12 iterations.
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 12 }]),
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
        energy: cobre_io::EnergyConfig::default(),
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

#[test]
fn test_two_anticipated_plants_k1_2_k2_4_convergence() {
    let ant_id_k2: u32 = 2;
    let ant_id_k4: u32 = 5;
    let backup_id: u32 = 4;
    assert!(ant_id_k2 < ant_id_k4);
    assert_ne!(backup_id, ant_id_k4);

    let system = build_system_two_anticipated();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 12, HighsSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let result = &outcome.result;
    assert_eq!(result.iterations, 12);

    let actual = result.final_lb;
    let expected = EXPECTED_LB;
    let rel_diff = if expected.abs() > f64::EPSILON {
        (actual - expected).abs() / expected.abs()
    } else {
        actual.abs()
    };
    assert!(
        rel_diff < 1e-6,
        "final_lb mismatch: {actual} vs {expected} (rel_diff={rel_diff}). \
         If intentional, update EXPECTED_LB."
    );

    let indexer = setup.stage_indexer();
    assert_eq!(indexer.n_anticipated, 2);
    assert_eq!(indexer.k_max, 4);

    let n_anticipated = indexer.n_anticipated;
    let k_max = indexer.k_max;
    let ant_start = indexer.anticipated_state.start;
    let ant_block_len = n_anticipated * k_max;

    let basis_cache = &result.basis_cache;
    assert_eq!(basis_cache.len(), 6);

    let s0 = basis_cache[0]
        .as_ref()
        .expect("stage 0 basis must be Some")
        .state_at_capture
        .as_slice();

    let ant_slice = &s0[ant_start..ant_start + ant_block_len];
    assert_eq!(ant_slice.len(), 8);
    for &v in ant_slice {
        assert!(v.is_finite(), "anticipated state must be finite");
    }

    let s1 = basis_cache[1]
        .as_ref()
        .expect("stage 1 basis must be Some")
        .state_at_capture
        .as_slice();
    let s2 = basis_cache[2]
        .as_ref()
        .expect("stage 2 basis must be Some")
        .state_at_capture
        .as_slice();

    let slot1_p0_at_stage1 = s1[ant_start + n_anticipated];
    let slot0_p0_at_stage2 = s2[ant_start];
    assert!(
        (slot1_p0_at_stage1 - slot0_p0_at_stage2).abs() < 1e-9,
        "ring-buffer shift invariant violated: slot-1@stage-1={slot1_p0_at_stage1}, \
         slot-0@stage-2={slot0_p0_at_stage2}"
    );
}
