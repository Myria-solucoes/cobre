//! Canonical-sort determinism check for anticipated thermal plants.
//!
//! ## Scope and limitations
//!
//! This integration test verifies that `SystemBuilder::build()` produces a
//! deterministic canonical thermal ordering (by `EntityId`) regardless of the
//! declaration order in the input, and that training results are bit-for-bit
//! identical across two declaration orderings.
//!
//! **It is NOT a declaration-order invariance probe.** Because
//! `SystemBuilder::build()` sorts thermals by `EntityId`, both configurations
//! present **identical canonical input** to every downstream layer — so this
//! test is structurally a tautology: same input -> same output. It catches a
//! regression in `SystemBuilder`'s canonical sort but does NOT exercise the
//! LP-builder code paths that consume `anticipated_thermal_indices` directly.
//!
//! The genuine LP-level order-invariance probe is the test
//! `lp_template_invariant_under_anticipated_index_permutation` in the
//! `#[cfg(test)]` block of the `lp::builder::template` module. That probe
//! constructs two `TemplateBuildCtx`s with permuted
//! `anticipated_thermal_indices`/`anticipated_lead_stages` arrays and asserts LP
//! equivalence under the canonical column/row swap permutation — the direct
//! structural check this test does not perform.
//!
//! ## Fixture description
//!
//! - 6 stages, each 744 hours (one calendar month).
//! - 1 bus (deficit cost 500 $/MWh).
//! - 1 hydro (storage cap 200 hm³, initial 100 hm³, max_gen 250 MW,
//!   inflow mean 80 m³/s, spillage cost 0.01 $/hm³) — id=3.
//! - Anticipated thermal plant 0: K=2, cost 50 $/MWh, max 100 MW — id=2.
//! - Backup standard thermal: cost 500 $/MWh, max 200 MW — id=4.
//! - Anticipated thermal plant 1: K=4, cost 40 $/MWh, max 80 MW — id=5.
//! - Load 150 MW constant across all stages.
//!
//! ## Canonical-sort behavior
//!
//! Config A declares thermals as `[id=4 (backup), id=2 (ant K=2), id=5 (ant K=4)]`.
//! Config B declares thermals as `[id=5 (ant K=4), id=2 (ant K=2), id=4 (backup)]`.
//!
//! `SystemBuilder::build()` sorts by `EntityId`, so the canonical sorted order
//! — `[id=2, id=4, id=5]` — is identical in both runs. The LP column layout
//! is therefore deterministic: the anticipated-state block uses slot-major,
//! plant-minor indexing relative to the anticipated-local sort order
//! (id=2 -> local 0, id=5 -> local 1). A regression in `SystemBuilder`'s
//! canonical sort would surface here as different `final_lb` values; deeper
//! LP-builder regressions that depend on `anticipated_thermal_indices`
//! ordering are caught by the LP-construction unit test referenced above.

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
// System builder — parameterised by thermal declaration order
// ---------------------------------------------------------------------------

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

/// Build a 6-stage system with the three thermals declared in `thermal_order`.
///
/// `thermal_order` must be a permutation of `[EntityId(4), EntityId(2),
/// EntityId(5)]`. `SystemBuilder::build()` will sort by `EntityId` regardless
/// of the order passed here, so the canonical post-sort thermal list is always
/// `[id=2, id=4, id=5]`.
fn build_system_with_thermal_order(thermal_order: &[EntityId; 3]) -> cobre_core::System {
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

    // Build the thermal list in the requested declaration order.
    let thermals: Vec<Thermal> = thermal_order
        .iter()
        .map(|id| match id.0 {
            2 => thermal_ant_k2.clone(),
            4 => thermal_backup.clone(),
            5 => thermal_ant_k4.clone(),
            other => panic!("unexpected thermal EntityId: {other}"),
        })
        .collect();

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

    let k_max: usize = 4;
    let n_st = n_stages;

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
        .thermals(thermals)
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system_with_thermal_order: valid")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a minimal [`Config`] for 1 forward pass and 8 iterations.
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]),
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

/// Verify that `SystemBuilder::build()` produces deterministic canonical
/// thermal ordering and that training results are bit-for-bit identical
/// across two declaration orderings of the same 3 thermals.
///
/// **This is a determinism check, not an invariance probe.** See the
/// module-level docstring for the scope-and-limitations discussion and
/// the pointer to the genuine LP-level invariance probe that
/// exercises the `anticipated_thermal_indices` consumer paths directly.
///
/// What this test catches:
/// - A regression in `SystemBuilder::build()` that broke the canonical
///   sort would produce different `final_lb` values here.
/// - A change to the LP construction that is sensitive to which canonical
///   sort `SystemBuilder` uses (e.g., a switch from EntityId to name)
///   would produce different cached state bitwise.
///
/// What this test does NOT catch:
/// - Order-dependence within the LP builder itself when given different
///   permutations of `anticipated_thermal_indices`. Both configurations
///   feed identical canonical input to the LP builder, so any order-
///   dependent code path downstream of `SystemBuilder` is exercised
///   identically in both runs.
#[test]
fn declaration_order_invariance_anticipated_thermals() {
    // Config A: regular-first, then anticipated ascending.
    let order_a = [EntityId(4), EntityId(2), EntityId(5)];
    // Config B: reversed — anticipated descending, then regular.
    let order_b = [EntityId(5), EntityId(2), EntityId(4)];

    let system_a = build_system_with_thermal_order(&order_a);
    let system_b = build_system_with_thermal_order(&order_b);

    // After SystemBuilder::build(), both systems must have thermals sorted by
    // EntityId: [id=2, id=4, id=5]. Assert this explicitly so a regression in
    // the builder is caught here rather than silently producing wrong results.
    assert!(
        system_a
            .thermals()
            .windows(2)
            .all(|w| w[0].id.0 < w[1].id.0)
    );
    assert!(
        system_b
            .thermals()
            .windows(2)
            .all(|w| w[0].id.0 < w[1].id.0)
    );

    let config = build_config();

    let mut setup_a = build_setup(system_a, &config);
    let mut setup_b = build_setup(system_b, &config);

    let comm = StubComm;

    let mut solver_a = ActiveSolver::new().expect("ActiveSolver::new");
    let mut solver_b = ActiveSolver::new().expect("ActiveSolver::new");

    let outcome_a = setup_a
        .train(&mut solver_a, &comm, 8, ActiveSolver::new, None, None)
        .expect("train");

    let outcome_b = setup_b
        .train(&mut solver_b, &comm, 8, ActiveSolver::new, None, None)
        .expect("train");

    assert!(
        outcome_a.error.is_none(),
        "config A error: {:?}",
        outcome_a.error
    );
    assert!(
        outcome_b.error.is_none(),
        "config B error: {:?}",
        outcome_b.error
    );

    let result_a = &outcome_a.result;
    let result_b = &outcome_b.result;

    assert_eq!(result_a.iterations, result_b.iterations);

    let lb_a = result_a.final_lb;
    let lb_b = result_b.final_lb;
    assert_eq!(lb_a.to_bits(), lb_b.to_bits(), "final_lb differs");

    let basis_cache_a = &result_a.basis_cache;
    let basis_cache_b = &result_b.basis_cache;
    assert_eq!(basis_cache_a.len(), basis_cache_b.len());

    let state = setup_a.stage_state();
    let ant_start = state.anticipated_state.start;
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;
    let ant_block_len = n_anticipated * k_max;

    assert_eq!(n_anticipated, 2);
    assert_eq!(k_max, 4);

    let n_stages = basis_cache_a.len();

    for t in 0..n_stages {
        let entry_a = basis_cache_a[t].as_ref().expect("basis_a");
        let entry_b = basis_cache_b[t].as_ref().expect("basis_b");

        let state_a = entry_a.state_at_capture.as_slice();
        let state_b = entry_b.state_at_capture.as_slice();

        assert_eq!(state_a.len(), state_b.len());

        let ant_slice_a = &state_a[ant_start..ant_start + ant_block_len];
        let ant_slice_b = &state_b[ant_start..ant_start + ant_block_len];

        for slot in 0..k_max {
            for plant in 0..n_anticipated {
                let idx = slot * n_anticipated + plant;
                let va = ant_slice_a[idx];
                let vb = ant_slice_b[idx];

                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "stage={t} slot={slot} plant={plant}"
                );
            }
        }
    }
}
