//! Slow-gated convergence test for a medium anticipated-thermal case (fixture
//! built by `build_system`).
//!
//! Verifies that `SamplingScheme::OutOfSample` converges to a lower bound
//! within 10% relative tolerance of the `SamplingScheme::InSample` lower
//! bound when anticipated thermals are present. The 10% tolerance (relaxed from
//! 5%) accommodates the anticipated ring-buffer variance source that is absent
//! in the pure-hydro convergence test.

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
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::entities::{
    bus::{Bus, DeficitSegment},
    hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    thermal::{AnticipatedConfig, Thermal},
};
use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme, ScenarioSource};
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
use cobre_sddp::{
    InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
    TrainingOutcome, hydro_models::PrepareHydroModelsResult, setup::ConstructionConfig,
};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

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

/// Build a 4-stage system with:
/// - 1 bus (deficit cost 500 $/MWh)
/// - 1 hydro (storage 200 hm³, initial 100 hm³, max_gen 250 MW,
///   inflow mean 80 m³/s, std 20 m³/s) — id=3
/// - 1 anticipated thermal (K=2, cost 50 $/MWh, max 100 MW) — id=2
/// - 1 backup standard thermal (cost 500 $/MWh, max 200 MW) — id=4
/// - Load 220 MW constant across all stages
/// - `branching_factor=5`, `NoiseMethod::Saa`
/// - `past_anticipated_commitments = [(id=2, [40.0, 20.0])]`
///
/// The LP is always feasible: backup thermal alone covers 220 MW.
fn build_system(branching_factor: usize) -> cobre_core::System {
    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let anticipated_id = EntityId(2);
    let thermal_ant = Thermal {
        id: anticipated_id,
        name: "T_ant".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        cost_per_mwh: 500.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let hydro = Hydro {
        id: EntityId(3),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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

    let n_stages = 4_usize;
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
                branching_factor,
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
            mean_mw: 220.0,
            std_mw: 0.0,
        })
        .collect();

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

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: vec![40.0, 20.0],
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
// Training runner
// ---------------------------------------------------------------------------

/// Run the SDDP training pipeline programmatically.
///
/// Builds a `StudySetup` using the low-level `from_broadcast_params` path
/// (identical to `forward_sampler_integration.rs::run_programmatic`) so that
/// both the `InSample` and `OutOfSample` sampling schemes can be exercised
/// without going through the file-based config/IO path.
fn run_training(
    system: &cobre_core::System,
    sampling_scheme: SamplingScheme,
    forward_seed: Option<i64>,
    inflow_method: InflowNonNegativityMethod,
) -> TrainingOutcome {
    const FORWARD_PASSES: u32 = 10;
    const MAX_ITERATIONS: u64 = 30;

    let tree_seed: u64 = 42;
    // `forward_seed` for `build_stochastic_context` is the unsigned abs of the
    // signed seed from `ScenarioSource`, matching the conversion in
    // `forward_sampler_integration.rs::run_programmatic`.
    let fwd_seed_u64 = forward_seed.map(i64::unsigned_abs);

    let stochastic = build_stochastic_context(
        system,
        tree_seed,
        fwd_seed_u64,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_stochastic_context must succeed");

    let hydro_models = PrepareHydroModelsResult::default_from_system(system);

    let stopping_rule_set = StoppingRuleSet {
        rules: vec![StoppingRule::IterationLimit {
            limit: MAX_ITERATIONS,
        }],
        mode: StoppingMode::Any,
    };

    let config = ConstructionConfig {
        seed: tree_seed,
        forward_passes: FORWARD_PASSES,
        stopping_rule_set,
        n_scenarios: 0,
        io_channel_capacity: 0,
        policy_path: String::new(),
        inflow_method,
        cut_selection: None,
        cut_activity_tolerance: 0.0,
        budget: None,
        export_states: false,
        scalar_parameters: Vec::new(),
    };

    let source = ScenarioSource {
        inflow_scheme: sampling_scheme,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        seed: forward_seed,
        historical_years: None,
    };

    let mut setup = StudySetup::from_broadcast_params(
        system,
        stochastic,
        config,
        hydro_models,
        &source,
        &source,
    )
    .expect("StudySetup::from_broadcast_params must succeed");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok")
}

// ---------------------------------------------------------------------------
// Slow-gated convergence test
// ---------------------------------------------------------------------------

/// Verify that `SamplingScheme::OutOfSample` converges to a lower bound within
/// 10% relative tolerance of the `SamplingScheme::InSample` lower bound on a
/// 4-stage K=2 anticipated-thermal system with `branching_factor=5` SAA noise.
///
/// The 10% tolerance (relaxed from the 5% used in the pure-hydro convergence
/// test) accommodates the additional variance introduced by the anticipated
/// ring-buffer's interaction with the out-of-sample noise trajectories.
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
#[test]
fn anticipated_k2_insample_vs_out_of_sample_convergence() {
    const BRANCHING_FACTOR: usize = 5;
    const RELATIVE_TOLERANCE: f64 = 0.10;

    let system_insample = build_system(BRANCHING_FACTOR);
    let system_oos = build_system(BRANCHING_FACTOR);

    let result_insample = run_training(
        &system_insample,
        SamplingScheme::InSample,
        None,
        InflowNonNegativityMethod::None,
    );

    assert!(result_insample.error.is_none(), "InSample error");
    let lb_insample = result_insample.result.final_lb;
    assert!(lb_insample > 0.0 && lb_insample.is_finite());

    let result_oos = run_training(
        &system_oos,
        SamplingScheme::OutOfSample,
        Some(42),
        InflowNonNegativityMethod::Truncation,
    );

    assert!(result_oos.error.is_none(), "OutOfSample error");
    let lb_oos = result_oos.result.final_lb;
    assert!(lb_oos > 0.0 && lb_oos.is_finite());

    let relative_error = (lb_oos - lb_insample).abs() / lb_insample.abs().max(1e-10);
    assert!(
        relative_error < RELATIVE_TOLERANCE,
        "convergence exceeded tolerance: {:.2}% vs {:.0}%",
        relative_error * 100.0,
        RELATIVE_TOLERANCE * 100.0,
    );
}
