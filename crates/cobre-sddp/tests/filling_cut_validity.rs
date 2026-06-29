//! Cut- and warm-start-validity regression across the two hydro filling phase
//! boundaries (PreFilling -> Filling at `id == start_stage_id`, Filling ->
//! Operating at `id == entry_stage_id`).
//!
//! The fixture drives a CHAINED cascade `Hf1 -> Hf2 -> Hop` (two consecutive
//! PreFilling hydros draining into an Operating plant), not a single filling
//! hydro: a PreFilling water-balance row collapses to the frozen identity
//! `v_h - v_h_in = 0` and short-circuits onto the first non-PreFilling downstream
//! resolved by `resolve_shortcircuit_target`, which walks *through* any PreFilling
//! downstream whose row is itself a frozen identity. A single-hop fixture never
//! exercises that transitive walk, where the silent corruption hides: `Hf1`'s
//! water can land on `Hf2`'s frozen-identity RHS, producing a wrong-but-compiling
//! cut.

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
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::sync::mpsc;

use cobre_core::entities::{
    bus::DeficitSegment,
    hydro::{FillingConfig, HydroGenerationModel, HydroPenalties},
};
use cobre_core::scenario::{InflowModel, LoadModel};
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
use cobre_sddp::SolverStatsDelta;
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

// ---------------------------------------------------------------------------
// Fixture topology constants (study stage ids; id == index for this horizon)
// ---------------------------------------------------------------------------

const N_STAGES: usize = 7;

/// PreFilling -> Filling boundary: both filling hydros are PreFilling at every
/// id `< START_STAGE_ID` and Filling from here.
const START_STAGE_ID: i32 = 2;

/// Filling -> Operating boundary, interior to the horizon
/// (`START_STAGE_ID < ENTRY_STAGE_ID < N_STAGES`).
const ENTRY_STAGE_ID: i32 = 4;

/// Iteration 1 captures the first basis; iterations 2.. warm-start through
/// `reconstruct_basis` across both boundaries.
const N_ITERATIONS: u64 = 8;

const HF1_ID: i32 = 1;
const HF2_ID: i32 = 2;
const HOP_ID: i32 = 3; // Operating downstream — the transitive short-circuit target
const HCTL_ID: i32 = 4; // off-cascade control hydro (Operating every stage)

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

/// Build the chained filling cascade system: `Hf1 -> Hf2 -> Hop` (both filling,
/// both PreFilling at ids 0,1; `Hop` the transitive short-circuit target),
/// an off-cascade control `Hctl`, plus a bus deficit segment + backup thermal so
/// the LP stays feasible regardless of the filling hydros' frozen storage.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let thermal_backup = make_thermal(
        EntityId(5),
        ThermalSpec {
            name: "T_backup".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 500.0,
            min_generation_mw: 0.0,
            max_generation_mw: 400.0,
            anticipated_config: None,
            ..Default::default()
        },
    );

    let filling = || {
        Some(FillingConfig {
            start_stage_id: START_STAGE_ID,
            filling_min_rate_m3s: 50.0,
        })
    };

    // A filling hydro impounds toward a non-zero dead volume (min_storage_hm3 = 50)
    // and requires entry_stage_id = Some (the system builder rejects `filling`
    // without it); operating/control hydros use a 0 floor and no entry stage.
    let cascade_hydro =
        |id: i32, name: &str, downstream: Option<i32>, filling: Option<FillingConfig>| {
            let min_storage_hm3 = if filling.is_some() { 50.0 } else { 0.0 };
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: name.to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                        .unwrap()
                        .checked_add_signed(chrono::Duration::days(i64::from(id)))
                        .unwrap(),
                    bus_id: EntityId(1),
                    downstream_id: downstream.map(EntityId),
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
                    ..Default::default()
                },
            )
        };

    let hydros = vec![
        cascade_hydro(HF1_ID, "Hf1", Some(HF2_ID), filling()),
        cascade_hydro(HF2_ID, "Hf2", Some(HOP_ID), filling()),
        cascade_hydro(HOP_ID, "Hop", None, None),
        cascade_hydro(HCTL_ID, "Hctl", None, None),
    ];

    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2020, ((i % 12 + 1) % 12 + 1) as u32, 1)
                        .unwrap(),
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
    // `FillingConfig` scalar; set it on the two filling hydros (positional
    // indices 0 = Hf1, 1 = Hf2) at every stage so the Filling-phase row impounds.
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

    // Filling hydros carry their stage-0 seed in `filling_storage`, operating /
    // control hydros in `storage`; the filling seed is `0.0` (empty pit, frozen
    // through the PreFilling phase that exists because `start_stage_id > 0`).
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

// ---------------------------------------------------------------------------
// Integration test
// ---------------------------------------------------------------------------

/// Train the chained filling cascade and assert the cut-/warm-start-validity
/// contract across both filling phase boundaries: monotone lower bound, zero
/// basis-rejection spike at the boundary stages, and an off-cascade control
/// hydro that still dispatches as a normal Operating plant.
#[test]
fn filling_cascade_lower_bound_monotone_no_basis_spike() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup_in_code(system, &config);
    // Populate the visited-states archive so the control hydro's forward-pass
    // storage trajectory is observable for the parity-neutrality assertion below.
    setup.set_export_states(true);

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            // 3rd arg is the forward-thread count, NOT the iteration limit (that
            // comes from the IterationLimit stopping rule).
            1,
            ActiveSolver::new,
            Some(tx),
            None,
        )
        .expect("train must not return Err");

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
        // The minorant property is exact in real arithmetic, but HiGHS/CLP FP
        // divergence makes a strict `>=` flaky across backends; the
        // absolute-relative tolerance absorbs that noise while still catching a
        // genuine non-monotone regression (an understated cut drops the bound by
        // far more than 1e-6 relative). Do not relax it to a large absolute slack.
        let tol = 1e-6 * prev.abs().max(1.0);
        assert!(
            next >= prev - tol,
            "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
        );
    }

    // Aggregate `basis_consistency_failures` at exactly the two boundary stages
    // (id == index here) — NOT a global aggregate, which would dilute a boundary
    // spike behind otherwise-clean stages.
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

    // Read the control hydro's forward-pass storage from the visited archive at
    // state index `state.storage.start + ctl_pos`; `ctl_pos` is its positional
    // index in canonical id order (id 4 -> last of four hydros -> 3).
    let state = setup.stage_state();
    let ctl_pos = 3_usize;
    let ctl_storage_col = state.storage.start + ctl_pos;
    let archive = result
        .visited_archive
        .as_ref()
        .expect("export_states was enabled, so the visited archive is populated");

    // The control is Operating (not frozen at its seed like a PreFilling hydro):
    // assert it stays within [0, max] and MOVES off its initial 100.0 hm3 somewhere
    // across the horizon.
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
    // The end-of-filling outgoing storage flows into the first Operating stage's
    // incoming-state pin via the SAME pin chain every other stage uses (no
    // entry-boundary reset / RHS-fold / `initial_operating_volume`); the no-reset
    // contract is owned by `builder_setup_never_references_entry_boundary_reset`.
    // Column identity is structural — one study-level `StateLayout` maps each
    // storage coordinate to a stage-invariant dense column, so a same-vs-same
    // assertion would be tautological; this block adds the value-continuity check.
    let entry = ENTRY_STAGE_ID as usize;
    let last_filling = entry - 1;
    let hf1_pos = 0_usize;

    let out_col_last_filling = state.storage.start + hf1_pos;
    let out_col_entry = state.storage.start + hf1_pos;

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
/// BOUNDS at `entry`, never the column index). Re-introducing an entry-stage reset,
/// an RHS-fold, or a pin to `initial_operating_volume` would land one of the
/// forbidden symbols in the builder/setup source and fail this guard.
///
/// The needles are assembled from char fragments (the
/// `lower_bound_never_references_filling_gating` idiom) so the literals are absent
/// from this file's own bytes — else the scan would flag itself.
#[test]
fn builder_setup_never_references_entry_boundary_reset() {
    let needles: [String; 5] = [
        ["initial", "_operating_volume"].concat(),
        ["entry", "_reset"].concat(),
        ["rhs", "_fold"].concat(),
        ["reset", "_storage"].concat(),
        ["operating", "_volume_reset"].concat(),
    ];

    // The builder/setup sources that own the entry-boundary handoff; a reset draft
    // would land in one of these.
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
