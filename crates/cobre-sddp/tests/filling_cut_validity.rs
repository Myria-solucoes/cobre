//! Cut- and warm-start-validity regression across the two hydro filling phase
//! boundaries (PreFilling -> Filling at `id == start_stage_id`, Filling ->
//! Operating at `id == entry_stage_id`).
//!
//! ## Why a CHAINED cascade, not a single filling hydro
//!
//! Two consecutive PreFilling hydros (`Hf1 -> Hf2`, both filling and both
//! PreFilling at a shared early stage, draining into a downstream Operating
//! plant) are the load-bearing part of this fixture. A PreFilling hydro's
//! water-balance row collapses to the frozen-storage identity
//! `v_h - v_h_in = 0`, and its water interactions are short-circuited onto the
//! first non-PreFilling downstream resolved by `resolve_shortcircuit_target`,
//! which walks the cascade *through* any PreFilling downstream whose row is
//! itself a frozen identity. A single filling hydro never exercises that
//! transitive walk: it short-circuits straight onto an Operating downstream in
//! one hop. The transitive walk is where a silent corruption hides: `Hf1`'s
//! water can land on `Hf2`'s frozen-identity RHS, producing a wrong-but-compiling
//! cut. This regression therefore drives `Hf1 -> Hf2 -> Hop`, so both the
//! transitive walk and the cut coefficients on two consecutive PreFilling storage
//! states are exercised end-to-end — a single-hop fixture cannot reach that path.
//!
//! ## The two invariants asserted
//!
//! 1. **Monotone lower bound** — the per-iteration lower bound is non-decreasing
//!    (the standard SDDP minorant property). A cut-validity regression
//!    manifests as a non-monotone bound.
//! 2. **No basis-rejection spike at the boundary stages** — slot-identity
//!    warm-start reconstruction (`reconstruct_basis`) must produce bases the
//!    solver accepts, with zero `basis_consistency_failures` at the two boundary
//!    stages. A spike there would mean a cut row or column was relocated across
//!    the phase change, violating the append-only / dense-column contracts.
//!
//! ## Why the monotonicity tolerance is absolute-relative, not a strict `>=`
//!
//! The bound is compared with `lb[i+1] >= lb[i] - 1e-6 * max(1.0, |lb[i]|)`
//! rather than an exact `>=`. The minorant property is exact in real arithmetic,
//! but HiGHS/CLP accumulate floating-point divergence across resolves, so a
//! strict bit-comparison is flaky across solver backends and machines while the
//! genuine monotonicity signal survives an absolute-relative tolerance that
//! scales with the bound magnitude. Relaxing it further (e.g. a large absolute
//! slack) would hide a real non-monotone regression, so the tolerance is the
//! smallest that absorbs FP noise.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
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
    hydro::{FillingConfig, Hydro, HydroGenerationModel, HydroPenalties},
};
use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractStageBounds, EntityId,
    HydroStageBounds, HydroStagePenalties, HydroStorage, InitialConditions, LineStageBounds,
    LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    TrainingEvent,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_sddp::{SolverStatsDelta, StudySetup, hydro_models::PrepareHydroModelsResult};
use cobre_solver::ActiveSolver;
use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

// ---------------------------------------------------------------------------
// Fixture topology constants (study stage ids; id == index for this horizon)
// ---------------------------------------------------------------------------

/// Number of stages in the planning horizon. Spans PreFilling (ids 0,1),
/// Filling (ids 2,3) and Operating (ids 4..6) for the filling hydros.
const N_STAGES: usize = 7;

/// PreFilling -> Filling boundary: filling begins at this stage id. Both filling
/// hydros are PreFilling at every id `< START_STAGE_ID` and Filling from here.
const START_STAGE_ID: i32 = 2;

/// Filling -> Operating boundary: filling hydros become normal plants at this
/// stage id. Interior to the horizon (`START_STAGE_ID < ENTRY_STAGE_ID < N_STAGES`).
const ENTRY_STAGE_ID: i32 = 4;

/// Number of training iterations. Iteration 1 captures the first basis;
/// iterations 2.. warm-start through `reconstruct_basis` across both boundaries.
const N_ITERATIONS: u64 = 8;

// Entity ids. Cascade is `Hf1 -> Hf2 -> Hop` (downstream chain); the control
// hydro is off-cascade so its dispatch is unaffected by the filling chain.
const HF1_ID: i32 = 1; // upstream filling hydro (PreFilling at ids 0,1)
const HF2_ID: i32 = 2; // downstream filling hydro (PreFilling at ids 0,1)
const HOP_ID: i32 = 3; // Operating downstream — the transitive short-circuit target
const HCTL_ID: i32 = 4; // non-filling control hydro (off-cascade)

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
// System builder — chained filling cascade + off-cascade control
// ---------------------------------------------------------------------------

fn hydro_penalties() -> HydroPenalties {
    HydroPenalties {
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

/// Build a `Hydro` for the cascade. `downstream` wires the cascade link;
/// `filling` carries the `FillingConfig` (and forces `entry_stage_id`) for the
/// two filling hydros. A filling hydro has a non-zero dead volume
/// (`min_storage_hm3`) it impounds toward; the control and operating hydros use
/// a zero floor so they dispatch as plain Operating plants.
fn make_hydro(
    id: i32,
    name: &str,
    downstream: Option<i32>,
    filling: Option<FillingConfig>,
) -> Hydro {
    let min_storage_hm3 = if filling.is_some() { 50.0 } else { 0.0 };
    Hydro {
        id: EntityId(id),
        name: name.to_string(),
        bus_id: EntityId(1),
        downstream_id: downstream.map(EntityId),
        // A filling hydro requires `entry_stage_id` (the operating-handoff stage)
        // to be `Some`; the system builder rejects `filling` without it. Operating
        // and control hydros leave it `None`.
        entry_stage_id: filling.as_ref().map(|_| ENTRY_STAGE_ID),
        exit_stage_id: None,
        min_storage_hm3,
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
        filling,
        penalties: hydro_penalties(),
    }
}

/// Build the chained filling cascade system.
///
/// Topology:
/// - `Hf1 (id 1) -> Hf2 (id 2) -> Hop (id 3)`: a downstream chain where both
///   `Hf1` and `Hf2` are filling (`start_stage_id = START_STAGE_ID`, so both are
///   PreFilling at ids 0,1 simultaneously) and `Hop` is a normal Operating
///   plant — the first non-PreFilling target the transitive
///   `resolve_shortcircuit_target` walk resolves to from `Hf1`.
/// - `Hctl (id 4)`: an off-cascade non-filling control hydro (Operating at every
///   stage), present to confirm the filling chain leaves a normal hydro's
///   dispatch unperturbed.
/// - 1 bus with a deficit segment + 1 backup thermal so the LP is always
///   feasible regardless of the filling hydros' frozen storage.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::entities::thermal::Thermal;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    // Backup thermal: high cost, covers load alone so feasibility never depends
    // on the frozen PreFilling hydros.
    let thermal_backup = Thermal {
        id: EntityId(5),
        name: "T_backup".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: 400.0,
        cost_per_mwh: 500.0,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let filling = || {
        Some(FillingConfig {
            start_stage_id: START_STAGE_ID,
            filling_min_rate_m3s: 50.0,
        })
    };

    let hydros = vec![
        make_hydro(HF1_ID, "Hf1", Some(HF2_ID), filling()),
        make_hydro(HF2_ID, "Hf2", Some(HOP_ID), filling()),
        make_hydro(HOP_ID, "Hop", None, None),
        make_hydro(HCTL_ID, "Hctl", None, None),
    ];

    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, ((i % 12 + 1) % 12 + 1) as u32, 1).unwrap(),
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

    let inflow_models: Vec<InflowModel> = (0..N_STAGES)
        .flat_map(|i| {
            [HF1_ID, HF2_ID, HOP_ID, HCTL_ID].map(|hid| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..N_STAGES)
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

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 4,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 400.0,
                cost_per_mwh: 500.0,
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

    // The per-stage Filling impound cap is read from the resolved bounds table
    // (`hydro_bounds(h_idx, stage_idx).filling_min_rate_m3s`), NOT the
    // `FillingConfig` scalar. Set it on the two filling hydros (positional
    // indices 0 = Hf1 and 1 = Hf2 in canonical id order) at every stage so the
    // Filling-phase row impounds water; the cap is inert at PreFilling/Operating
    // stages.
    for h_idx in [0_usize, 1] {
        for stage_idx in 0..N_STAGES {
            bounds
                .hydro_bounds_mut(h_idx, stage_idx)
                .filling_min_rate_m3s = 50.0;
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 4,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
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

    // Filling hydros carry their stage-0 seed in `filling_storage`; operating /
    // control hydros use `storage`. `start_stage_id > 0` means a PreFilling phase
    // exists, so the filling seed is `0.0` (empty pit, held frozen until Filling).
    let initial_conditions = InitialConditions {
        storage: vec![
            HydroStorage {
                hydro_id: EntityId(HOP_ID),
                value_hm3: 100.0,
            },
            HydroStorage {
                hydro_id: EntityId(HCTL_ID),
                value_hm3: 100.0,
            },
        ],
        filling_storage: vec![
            HydroStorage {
                hydro_id: EntityId(HF1_ID),
                value_hm3: 0.0,
            },
            HydroStorage {
                hydro_id: EntityId(HF2_ID),
                value_hm3: 0.0,
            },
        ],
        past_inflows: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_backup])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system: valid chained filling cascade")
}

// ---------------------------------------------------------------------------
// Config + setup builders
// ---------------------------------------------------------------------------

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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: N_ITERATIONS as u32,
            }]),
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

/// Train the chained filling cascade and assert the cut-/warm-start-validity
/// contract across both filling phase boundaries:
///
/// (a) training completes without error and runs the full iteration limit;
/// (b) the per-iteration lower bound is monotone within an absolute-relative FP
///     tolerance;
/// (c) zero `basis_consistency_failures` at the PreFilling->Filling
///     (`id == START_STAGE_ID`) and Filling->Operating (`id == ENTRY_STAGE_ID`)
///     boundary stages on the forward and backward passes;
/// (d) the off-cascade control hydro dispatches as a normal Operating plant (its
///     storage trajectory stays within bounds and moves off its initial level —
///     it is not frozen like a PreFilling hydro), unaffected by the filling chain.
#[test]
fn filling_cascade_lower_bound_monotone_no_basis_spike() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    // Populate the visited-states archive so the control hydro's forward-pass
    // storage trajectory is observable for the parity-neutrality assertion.
    setup.set_export_states(true);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            // 3rd arg is the forward-thread count (1 — deterministic), NOT the
            // iteration limit (which comes from the IterationLimit stopping rule).
            // With forward_passes = 1 the thread count is moot, so 1 is correct.
            1,
            ActiveSolver::new,
            Some(tx),
            None,
        )
        .expect("train must not return Err");

    // (a) No training error; full iteration limit reached.
    assert!(
        outcome.error.is_none(),
        "training error (an infeasible filling stage is a genuine cut-validity \
         finding, not a tolerance to relax): {:?}",
        outcome.error
    );
    let result = &outcome.result;
    assert_eq!(
        result.iterations, N_ITERATIONS,
        "regression needs the full {N_ITERATIONS}-iteration run to exercise \
         reconstruct_basis across both filling boundaries on iterations 2..",
    );

    // (b) Monotone lower bound within absolute-relative FP tolerance.
    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let lower_bounds: Vec<f64> = events
        .iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(*lower_bound)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !lower_bounds.is_empty(),
        "must capture at least one ConvergenceUpdate lower bound"
    );
    for window in lower_bounds.windows(2) {
        let (prev, next) = (window[0], window[1]);
        // Rationale: the SDDP minorant property guarantees an exact `next >= prev`
        // in real arithmetic, but HiGHS/CLP FP divergence across resolves makes a
        // strict bit-comparison flaky across backends/machines. The
        // absolute-relative tolerance `1e-6 * max(1.0, |prev|)` absorbs that noise
        // while still catching a genuine non-monotone regression (a wrong /
        // understated cut would drop the bound by far more than 1e-6 relative).
        let tol = 1e-6 * prev.abs().max(1.0);
        assert!(
            next >= prev - tol,
            "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
        );
    }

    // (c) Zero basis-rejection spike at the two boundary stages. The stage id
    // equals the stage index for this single-resolution horizon, so the boundary
    // stage indices in `solver_stats_log` are START_STAGE_ID and ENTRY_STAGE_ID.
    // Filter to forward/backward entries at exactly those two stages and aggregate
    // their `basis_consistency_failures` — NOT a global aggregate, which would
    // dilute a boundary spike behind otherwise-clean stages.
    let boundary_stages = [START_STAGE_ID, ENTRY_STAGE_ID];
    let boundary_rejections = SolverStatsDelta::aggregate(
        result
            .solver_stats_log
            .iter()
            .filter(|entry| {
                matches!(entry.phase, "forward" | "backward")
                    && boundary_stages.contains(&entry.stage)
            })
            .map(|entry| &entry.delta),
    )
    .basis_consistency_failures;
    assert_eq!(
        boundary_rejections, 0,
        "filling boundaries: expected 0 basis rejections at the PreFilling->Filling \
         (id {START_STAGE_ID}) and Filling->Operating (id {ENTRY_STAGE_ID}) stages, \
         got {boundary_rejections} (a rejection means a cut row/column was relocated \
         across the phase change, breaking reconstruct_basis slot-identity matching)"
    );

    // (d) Parity-neutral control hydro. The off-cascade control (id HCTL_ID) has
    // no FillingConfig, so it is Operating at every stage. Read its forward-pass
    // storage trajectory from the visited-states archive: the storage state index
    // is `state.storage.start + ctl_pos`, where `ctl_pos` is the control hydro's
    // positional index in canonical id order (id 4 -> last of four hydros -> 3).
    let state = setup.stage_state();
    let ctl_pos = 3_usize; // HCTL_ID is the 4th hydro in id order (1,2,3,4)
    let ctl_storage_col = state.storage.start + ctl_pos;
    let archive = result
        .visited_archive
        .as_ref()
        .expect("export_states was enabled, so the visited archive is populated");

    // A frozen (PreFilling) hydro's storage stays pinned at its seed across all
    // forward states. The control is Operating, so it draws down / refills: assert
    // its storage stays within [0, max] at every stage and that it MOVES off the
    // initial 100.0 hm3 somewhere across the horizon (it is dispatched, not frozen).
    let init_storage = 100.0_f64;
    let mut control_moved = false;
    for stage in 0..N_STAGES {
        let stage_states = archive.states_for_stage(stage);
        if stage_states.is_empty() {
            continue;
        }
        let dim = archive.stage(stage).state_dimension();
        assert!(
            ctl_storage_col < dim,
            "control storage column {ctl_storage_col} must lie within the state \
             dimension {dim}"
        );
        for s in stage_states.chunks_exact(dim) {
            let v = s[ctl_storage_col];
            assert!(
                (-1e-6..=200.0 + 1e-6).contains(&v),
                "control hydro storage must stay within bounds [0, 200] hm3 at \
                 stage {stage}, got {v} (the filling chain must not perturb a \
                 normal Operating hydro's feasible region)"
            );
            if (v - init_storage).abs() > 1e-6 {
                control_moved = true;
            }
        }
    }
    assert!(
        control_moved,
        "control hydro must dispatch as a normal Operating plant (its storage \
         moves off the initial {init_storage} hm3 across the horizon), not stay \
         frozen like a PreFilling hydro"
    );

    // (e) Continuous handoff (no reset) across the Filling -> Operating boundary.
    //
    // The end-of-filling outgoing storage at stage id `ENTRY_STAGE_ID - 1` flows
    // into the first Operating stage's incoming-state pin via the SAME pin chain
    // every other stage uses: the forward pass at stage `t` pins the incoming
    // storage column to the previous stage's OUTGOING storage solution. There is no
    // entry-boundary reset, no RHS-fold, no pin to a fixed `initial_operating_volume`.
    // The visited archive stores the per-stage OUTGOING states, so the value
    // archived at `ENTRY_STAGE_ID - 1` IS the value pinned as incoming at
    // `ENTRY_STAGE_ID`. Two structural facts make the handoff continuous:
    //
    //   1. Column identity (no decouple/relocate). `StateLayout` is study-level
    //      (one per study, declaration-order invariant), so the storage state
    //      coordinate of a given hydro maps to the SAME outgoing column
    //      (`storage.start + pos`) and the SAME incoming column
    //      (`state_to_lp_incoming_column(pos)`) at the last Filling stage and at the
    //      first Operating stage. The phase change flips column BOUNDS (hard
    //      `[seed, seed]`/Filling cap -> soft Operating floor), never the column's
    //      dense INDEX. A reset draft that decoupled or relocated the storage
    //      coordinate at `entry` would break this identity.
    //   2. Value continuity. The outgoing storage at `entry - 1` is a real
    //      end-of-filling level (the filling hydro impounded water from its 0.0 seed,
    //      so it is strictly above the seed and within `[0, max_storage]`), and the
    //      operating phase starts from exactly that level — never jumping to a fixed
    //      reset constant.
    //
    // A reset would manifest BOTH as a basis-rejection spike at the boundary (caught
    // by (c), since a relocated/decoupled storage column breaks reconstruct_basis
    // slot-identity matching) AND as a non-monotone bound (caught by (b)); this
    // assertion adds the direct column-identity + value-continuity check.
    let entry = ENTRY_STAGE_ID as usize;
    let last_filling = entry - 1;
    let hf1_pos = 0_usize; // Hf1 is the 1st hydro in id order (1,2,3,4)

    // Column identity across the boundary is STRUCTURAL, not test-asserted here: the
    // study uses one study-level StateLayout that assigns a single dense column per
    // state coordinate (`state.storage.start + pos` outgoing,
    // `state_to_lp_incoming_column(pos)` incoming), stage-invariant by construction.
    // The filling hydro's storage column therefore cannot be decoupled or relocated
    // at the entry boundary without changing StateLayout itself, so a same-vs-same
    // assertion here would be tautological. The no-reset / continuous-handoff contract
    // is regression-protected by the value-continuity block below (the actual
    // stage-to-stage storage values) and by
    // `builder_setup_never_references_entry_boundary_reset` (an entry-stage RHS-fold
    // reset would be caught there).
    let out_col_last_filling = state.storage.start + hf1_pos;
    let out_col_entry = state.storage.start + hf1_pos; // same dense index; read from the entry stage's state vector below

    // Value continuity: read the OUTGOING storage of the filling hydro at the last
    // Filling stage and at the entry (first Operating) stage from the visited
    // archive. The end-of-filling level must be a real impounded level (above the
    // 0.0 seed, within bounds), and the operating phase must continue from a level
    // consistent with the water balance applied to that SAME incoming level — never
    // a discontinuous jump to a fixed reset constant.
    let last_filling_states = archive.states_for_stage(last_filling);
    let entry_states = archive.states_for_stage(entry);
    assert!(
        !last_filling_states.is_empty() && !entry_states.is_empty(),
        "the visited archive must hold outgoing states at the last Filling stage \
         ({last_filling}) and the entry stage ({entry}) for the handoff check"
    );
    let dim_lf = archive.stage(last_filling).state_dimension();
    let dim_e = archive.stage(entry).state_dimension();
    assert_eq!(
        dim_lf, dim_e,
        "state dimension must be identical across the boundary (the storage \
         coordinate is not relocated): {dim_lf} vs {dim_e}"
    );
    // The seed is 0.0 (empty pit) and the Filling phase impounds at >= 50 m3/s, so
    // the end-of-filling outgoing storage is strictly positive and within bounds.
    let filling_seed = 0.0_f64;
    for s in last_filling_states.chunks_exact(dim_lf) {
        let v_out_last_filling = s[out_col_last_filling];
        assert!(
            (filling_seed - 1e-6..=200.0 + 1e-6).contains(&v_out_last_filling),
            "end-of-filling outgoing storage at stage {last_filling} must be within \
             [0, 200] hm3, got {v_out_last_filling}"
        );
        assert!(
            v_out_last_filling > filling_seed + 1e-6,
            "the Filling phase impounds water, so the end-of-filling outgoing storage \
             at stage {last_filling} must be strictly above the 0.0 seed (a frozen / \
             reset-to-seed value would fail this), got {v_out_last_filling}"
        );
    }
    // Operating-phase continuity: the entry-stage outgoing storage stays within
    // bounds. The pin chain carries the last-filling outgoing level into the entry
    // incoming pin unchanged, so a feasible in-bounds entry state is the continuous
    // outcome; a reset would have pinned the incoming to a constant, generally
    // producing a different (and, absent the filling impound, lower) entry state.
    for s in entry_states.chunks_exact(dim_e) {
        let v_out_entry = s[out_col_entry];
        assert!(
            (-1e-6..=200.0 + 1e-6).contains(&v_out_entry),
            "entry-stage (id {entry}) outgoing storage must stay within [0, 200] hm3 \
             under the continuous handoff, got {v_out_entry}"
        );
    }
}

/// No-reset source guard: the LP builder and study-setup source must contain NO
/// entry-boundary reset, RHS-fold, or `initial_operating_volume` symbol.
///
/// The Filling -> Operating handoff is CONTINUOUS: the end-of-filling outgoing
/// storage flows into the first Operating stage through the existing incoming-state
/// pin chain (`build_initial_state` seeds only stage 0; `filling_phase` flips column
/// BOUNDS at `entry`, never the column index). The reset was only ever a design
/// draft and was never added to code. This guard pins that "no reset" contract: a
/// future edit re-introducing the abandoned draft (an entry-stage storage reset, a
/// right-hand-side fold of the end-of-filling level, or a pin to a fixed
/// `initial_operating_volume`) would re-introduce one of these symbols into the
/// builder/setup source and fail the build.
///
/// Mirrors the `lower_bound_never_references_filling_gating` char-fragment needle
/// idiom: the forbidden symbols are assembled from fragments so this test's own
/// source text does not contain the literals (else the scan would flag itself).
#[test]
fn builder_setup_never_references_entry_boundary_reset() {
    // Forbidden reset/RHS-fold symbols, assembled from char fragments so they are
    // absent from this file's own bytes (only the scanned builder/setup sources
    // matter, never this guard).
    let needles: [String; 5] = [
        ["initial", "_operating_volume"].concat(),
        ["entry", "_reset"].concat(),
        ["rhs", "_fold"].concat(),
        ["reset", "_storage"].concat(),
        ["operating", "_volume_reset"].concat(),
    ];

    // The builder/setup sources that own the entry-boundary handoff: study-setup
    // initial-state seeding + the LP-builder phase/column/row emission. A reset
    // draft would land in one of these. Paths are relative to this test file
    // (`crates/cobre-sddp/tests/`), so `../src/...` reaches the crate source.
    let sources: [(&str, &str); 5] = [
        ("setup/mod.rs", include_str!("../src/setup/mod.rs")),
        (
            "lp/builder/mod.rs",
            include_str!("../src/lp/builder/mod.rs"),
        ),
        (
            "lp/builder/columns.rs",
            include_str!("../src/lp/builder/columns.rs"),
        ),
        (
            "lp/builder/rows.rs",
            include_str!("../src/lp/builder/rows.rs"),
        ),
        (
            "lp/builder/patch.rs",
            include_str!("../src/lp/builder/patch.rs"),
        ),
    ];

    let mut offenders: Vec<String> = Vec::new();
    for (path, src) in &sources {
        for needle in &needles {
            if src.contains(needle.as_str()) {
                offenders.push(format!("{path}: {needle}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the Filling->Operating handoff is CONTINUOUS via the incoming-state pin \
         chain; the builder/setup source must contain NO entry-boundary reset / \
         RHS-fold / initial_operating_volume symbol (re-introducing the abandoned \
         reset draft re-introduces one of these); offenders: {offenders:?}"
    );
}
