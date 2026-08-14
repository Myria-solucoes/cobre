//! Terminal boundary-FCF pricing of the post-horizon carry lanes on the
//! unified hold geometry: a loaded boundary cut carrying coefficient `β` on a
//! post-horizon `commit_out`/`commit_in` lane reaches the terminal `θ`, which
//! responds to the carried committed MW by `β·x`, and the backward pass
//! propagates that `β` inward to the deciding stage's cut. The one shared
//! boundary prices every terminal fan leaf against its OWN lane state
//! (the shared-fan injection topology, extended to the lanes). `β`
//! values the carried STATE; the delivery-anchored fuel is booked separately on
//! the decision column — the two are disjoint columns, so they compose without
//! double-counting.
//!
//! The fixture is deliberately minimal: zero hydros, one bus, one anticipated
//! thermal declared `LeadTime`, two study stages, and one
//! `future_anticipated_deliveries` window resolving into one declared
//! `post_study_stages` stage. The lead is long enough that the thermal's own
//! stage-1 self-delivery keeps `k_max == 1` (the in-study ring sizes
//! non-degenerately, avoiding the untested `k_max == 0` path) yet the
//! post-horizon window's decider lands at stage 0 (`DeciderKind::Trunk`, NOT
//! the terminal) — so the terminal's own lane row is a pure CARRY (`out = in`),
//! and pinning the incoming lane column via `patch_backward_opening_for_probe`
//! controls the terminal's priced committed MW directly. With `hydro_count = 0`
//! and `max_par_order = 0`, the state vector is exactly `[in-study ring slot,
//! post-horizon lane]` (`n_state == 2`), so a boundary priced ONLY on the lane
//! slot isolates the pricing arithmetic to the lane.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;

use chrono::NaiveDate;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, StageStateConfig, Transition};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, FutureAnticipatedDelivery,
    HorizonGraph, HydroBlockBounds, HydroStageBounds, InitialConditions, LineBlockBounds,
    PostStudyStage, PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds,
    System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::{
    FORMAT_VERSION, GraphManifest, ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord,
    ProducerBlock, StageCutsPayload, write_policy_checkpoint,
};
use cobre_sddp::indexer::{CutStateProjection, StateDim};
use cobre_sddp::setup::{NodeGraph, NodeId, NodePos, StageIdx};
use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
use cobre_sddp::workspace::SolverWorkspace;
use cobre_sddp::{inject_boundary_cuts, load_boundary_cuts};
use cobre_solver::{
    ActiveSolver, FreezeScratch, RowBatch, SolverInterface, StageTemplate,
    freeze_rows_into_template,
};

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// `LeadTime` delta (hours), ~60 days. Resolves the thermal's own stage-1
/// self-delivery to a stage-0 decider (`k_max == 1`), while the post-horizon
/// window (delivering in the first post-study month) resolves to a stage-0
/// `DeciderKind::Trunk` decider — the terminal is a carry, not the latch.
const DELTA_HOURS: f64 = 1450.0;
/// Boundary cut intercept `α`, positive so the single injected cut binds `θ`
/// above its zero floor for every committed MW `x >= 0`.
const ALPHA: f64 = 5.0;
/// Boundary cut coefficient `β` on the post-horizon lane slot (zero on the
/// in-study ring slot), so `θ = α + β·x` isolates the lane.
const BETA: f64 = 3.0;
/// Post-study fuel `$/MWh` at the resolved cell — booked on the decision
/// column, disjoint from the terminal `β·state` valuation this file pins.
const COST_PER_MWH: f64 = 37.5;
/// Post-study stage duration \[h\]; the study carries a `0.0` discount rate.
const POST_STUDY_HOURS: f64 = 720.0;
const TOL: f64 = 1e-6;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Study end date (`stage 1` end) — the first `post_study_stages` stage and the
/// post-horizon delivery window both anchor here, making the segment
/// date-contiguous with the horizon (the first-post-study-start == study-end
/// rule).
fn study_end() -> NaiveDate {
    study_start() + chrono::TimeDelta::days(61)
}

/// Two stages matching their real calendar span exactly (`StageCalendar`
/// coverage): stage 0 January (744h), stage 1 the following 30 days (720h).
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    let stage0_end = start + chrono::TimeDelta::days(31);
    vec![
        make_stage(
            0,
            StageSpec {
                start_date: start,
                end_date: stage0_end,
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S0".to_string(),
                    duration_hours: 744.0,
                }],
                ..Default::default()
            },
        ),
        make_stage(
            1,
            StageSpec {
                start_date: stage0_end,
                end_date: study_end(),
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S1".to_string(),
                    duration_hours: 720.0,
                }],
                ..Default::default()
            },
        ),
    ]
}

/// The one declared post-horizon window: the 30-day month beginning at the
/// study end, resolving (via [`DELTA_HOURS`]) to a stage-0 decider and (via the
/// post-study calendar below) to post-study destination stage 0.
fn future_delivery() -> FutureAnticipatedDelivery {
    FutureAnticipatedDelivery {
        thermal_id: THERMAL_ID,
        delivery_start: study_end(),
        delivery_end: study_end() + chrono::TimeDelta::days(30),
        min_mw: 0.0,
        max_mw: 100.0,
    }
}

/// The one declared post-study stage: the EXACT span [`future_delivery`]
/// targets, so `StageCalendar::resolve_window` covers it at destination index 0.
fn post_study_stages() -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: study_end(),
            duration_hours: POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: COST_PER_MWH,
            min_mw: 0.0,
            max_mw: 100.0,
        }],
    }
}

fn bounds() -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 1.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    )
}

fn penalties() -> cobre_core::resolved::ResolvedPenalties {
    use cobre_core::resolved::{
        BusStagePenalties, HydroStagePenalties, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, ResolvedPenalties,
    };
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.0,
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
                inflow_nonnegativity_cost: 0.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// Two-leaf terminal fan: root (id 0, stage 0) branches into leaves (ids 1 and
/// 2, stage 1) — both terminal, sharing ONE pool (`build_node_graph`'s
/// leaf-sharing rule).
fn two_leaf_fan_graph() -> HorizonGraph {
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ],
        nodes: vec![
            PolicyNode {
                id: 0,
                stage_id: 0,
                scenario_id: None,
                label: None,
            },
            PolicyNode {
                id: 1,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
            PolicyNode {
                id: 2,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
        ],
        stage_discount_rate_overrides: std::collections::HashMap::new(),
        season_map: None,
    }
}

/// `with_window` toggles the single post-horizon commitment window plus its
/// covering `post_study_stages`; `fanned` toggles the two-leaf terminal fan
/// (chain otherwise).
fn build_system(with_window: bool, fanned: bool) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            bus_id: BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(DELTA_HOURS)),
            ..Default::default()
        },
    );
    let initial_conditions = InitialConditions {
        future_anticipated_deliveries: if with_window {
            vec![future_delivery()]
        } else {
            Vec::new()
        },
        ..Default::default()
    };

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages())
        .bounds(bounds())
        .penalties(penalties())
        .initial_conditions(initial_conditions);
    if with_window {
        builder = builder.post_study_stages(Some(post_study_stages()));
    }
    if fanned {
        builder = builder.policy_graph(two_leaf_fan_graph());
    }
    builder.build().expect("fixture System must build")
}

fn config() -> cobre_io::config::Config {
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StateSpaceConfig, StoppingMode, StoppingRuleConfig,
        TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        state_space: StateSpaceConfig::default(),
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            // Unscaled: the priced `θ` and the propagated cut coefficient
            // compare directly against `α`/`β` with no COST_SCALE_FACTOR back-out.
            cost_scale_factor: Some(1.0),
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: SimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// State-vector index of the (sole) post-horizon lane: the trailing
/// `n_commitment` block of `commit_out`, after the leading in-study ring.
fn lane_state_index(setup: &cobre_sddp::StudySetup) -> usize {
    let state = setup.stage_state();
    state.commit_out.end - state.n_commitment
}

/// Write a synthetic single-cut boundary checkpoint carrying `intercept` and
/// the explicit per-slot `coefficients`. No entity manifest (`&[]`): the
/// loader's identity check short-circuits with a warning (the established probe
/// pattern — `shared_boundary_terminal_fan_probe.rs`), so this test controls
/// only the state dimension, not entity-identity matching.
fn write_synthetic_boundary(
    dir: &Path,
    state_dimension: u32,
    intercept: f64,
    coefficients: &[f64],
) {
    let cuts = vec![PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept,
        coefficients,
        is_active: true,
    }];
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension,
        capacity: 1,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
    };
    let metadata = PolicyCheckpointMetadata {
        format_version: FORMAT_VERSION,
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: "2026-08-10T00:00:00Z".to_string(),
        num_stages: 1,
        graph_manifest: GraphManifest {
            n_pools: 1,
            nodes: vec![ManifestNode {
                id: 100,
                stage_id: 0,
                pool_id: 0,
            }],
            edges: vec![],
        },
        producer: ProducerBlock {
            completed_iterations: 0,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 0,
            forward_passes: 0,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: Some(1.0),
        },
    };
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// Load a lane-only boundary (`β` on [`lane_state_index`], zero elsewhere,
/// intercept `α`) and inject it into `setup`'s terminal pool.
fn inject_lane_boundary(setup: &mut cobre_sddp::StudySetup, dir: &Path) {
    let state_dimension = setup.fcf.state_dimension as u32;
    let lane = lane_state_index(setup);
    let mut coefficients = vec![0.0_f64; state_dimension as usize];
    coefficients[lane] = BETA;
    write_synthetic_boundary(dir, state_dimension, ALPHA, &coefficients);

    let boundary_cuts =
        load_boundary_cuts(dir, 0, state_dimension, &[], &[], None, 1.0, &mut |_msg| {})
            .expect("boundary cut must load");
    inject_boundary_cuts(setup, &boundary_cuts);
}

/// Build the terminal pool's FROZEN LP template: the base structural template
/// plus one literal row per active cut in the pool, exactly as production
/// freezes a pool once per iteration (`freeze_rows_into_template`).
/// `solve_stage_for_probe`'s `pool` argument only drives basis reconstruction
/// on a stored warm start — it never appends cut rows to a cold solve, so
/// pricing an injected cut requires freezing it into the loaded model first.
fn freeze_terminal_template(setup: &cobre_sddp::StudySetup, pool_id: usize) -> StageTemplate {
    let state = setup.stage_state();
    let terminal_stage = setup.num_stages() - 1;
    let ctx = setup.stage_ctx();
    let base = &ctx.templates[terminal_stage];

    // The terminal pool projects the full global state, so the all-enabled
    // projection reproduces exactly the coefficients the boundary cut carries.
    let cut_state = CutStateProjection::new(
        state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );

    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    cobre_sddp::build_cut_row_batch_into(
        &mut batch,
        &setup.fcf,
        pool_id,
        state,
        &cut_state,
        &base.col_scale,
    );

    let mut frozen = StageTemplate::empty();
    let mut scratch = FreezeScratch::new();
    freeze_rows_into_template(base, &batch, &mut frozen, &mut scratch);
    frozen
}

/// Solve the terminal stage's frozen `template` against `pool` with the state
/// vector pinned to `pinned_state` (length `n_state`), returning `θ`'s primal
/// value. `raw_noise = &[]`: `hydro_count == 0`, so the noise transform has no
/// dimension to iterate.
fn terminal_theta(
    setup: &cobre_sddp::StudySetup,
    template: &StageTemplate,
    pool: &cobre_sddp::CutPool,
    node_id: NodeId,
    pinned_state: &[f64],
) -> f64 {
    let comm = StubComm;
    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let theta_col = setup.stage_state().theta;
    let terminal_stage = setup.num_stages() - 1;

    ws.solver.reset_solver_state();
    ws.solver.load_model(template);
    patch_backward_opening_for_probe(
        ws,
        &ctx,
        &training_ctx,
        StageIdx(terminal_stage),
        pinned_state,
        &[],
    )
    .expect("StageSolvePrep::run must not error on the minimal fixture");

    let view = solve_stage_for_probe(ws, &ctx, pool, None, StageIdx(terminal_stage), 0, node_id)
        .expect("terminal stage solve must not error");
    view.primal[theta_col]
}

/// Pin the state vector with `x` on the post-horizon lane and `0.0` on every
/// other component (the in-study ring slot carries a zero boundary coefficient,
/// so its pinned value never reaches `θ`).
fn lane_pin(setup: &cobre_sddp::StudySetup, x: f64) -> Vec<f64> {
    let mut pin = vec![0.0_f64; setup.stage_state().n_state];
    pin[lane_state_index(setup)] = x;
    pin
}

/// Every leaf `NodePos` in `graph` — a node with no successors.
fn leaf_positions(graph: &NodeGraph) -> Vec<NodePos> {
    graph
        .nodes
        .iter_indexed()
        .filter(|&(pos, _)| graph.successors[pos].is_empty())
        .map(|(pos, _)| pos)
        .collect()
}

// ── θ prices the carried committed MW by β·x, and β propagates inward ──

/// The terminal value responds to the outstanding committed MW on the lane by
/// exactly `β·Δx`: with the single injected cut binding (`α > 0`), `θ = α + β·x`
/// on the lane, so `θ(x1) − θ(x0) = β·(x1 − x0)` — closed-form for the
/// single-lane single-coefficient case. Repeating the solve at the same pin
/// reproduces `θ` bit-for-bit (run-to-run reproducibility).
#[test]
fn boundary_cut_prices_the_committed_mw_by_beta_times_x() {
    let mut setup = build_setup_in_code(build_system(true, false), &config());
    assert_eq!(
        setup.fcf.state_dimension, 2,
        "fixture must be [in-study ring slot, post-horizon lane]"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_lane_boundary(&mut setup, &tmp.path().join("boundary"));

    let terminal_pool_id = setup.fcf.pools.len() - 1;
    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let pool = &setup.fcf.pools[terminal_pool_id];

    let x0 = 0.0;
    let x1 = 2.0;
    let theta0 = terminal_theta(&setup, &template, pool, NodeId(1), &lane_pin(&setup, x0));
    let theta1 = terminal_theta(&setup, &template, pool, NodeId(1), &lane_pin(&setup, x1));

    assert!(
        theta0 > 0.0 && theta1 > 0.0,
        "both solves must land on the injected cut's binding regime (theta0={theta0}, \
         theta1={theta1})"
    );

    let expected_delta = BETA * (x1 - x0);
    let actual_delta = theta1 - theta0;
    assert!(
        (actual_delta - expected_delta).abs() < TOL,
        "theta must respond to the committed MW by beta*x: expected delta {expected_delta}, \
         got {actual_delta}"
    );
    // The closed form is exact: theta = alpha + beta*x on the lane.
    assert!(
        (theta0 - ALPHA).abs() < TOL && (theta1 - (ALPHA + BETA * x1)).abs() < TOL,
        "theta must equal alpha + beta*x: theta0={theta0} (alpha={ALPHA}), \
         theta1={theta1} (alpha+beta*x1={})",
        ALPHA + BETA * x1
    );

    let theta1_again = terminal_theta(&setup, &template, pool, NodeId(1), &lane_pin(&setup, x1));
    assert_eq!(
        theta1.to_bits(),
        theta1_again.to_bits(),
        "the priced terminal value must reproduce bit-for-bit across a repeated solve"
    );
}

/// The backward pass propagates the boundary coefficient inward: with the
/// lane-only boundary injected into the terminal pool, one training iteration
/// leaves the deciding stage's generated cut carrying a nonzero coefficient on
/// the post-horizon lane slot. Because the terminal row is a pure carry
/// (`out = in`) with unit `col_scale` and unit `cost_scale`, the propagated
/// coefficient is exactly `β` (analytical).
#[test]
fn backward_pass_propagates_boundary_coefficient_to_the_deciding_stage() {
    let mut setup = build_setup_in_code(build_system(true, false), &config());
    // The window decides at stage 0 (Trunk); pool 0 is the FCF the deciding
    // stage carries, populated by the backward pass solving the terminal.
    let deciding_pool = 0;
    assert_eq!(
        setup.stage_ctx().geometry_per_stage[deciding_pool]
            .commitment_decision
            .len(),
        1,
        "the post-horizon window must decide at stage 0"
    );
    let lane = lane_state_index(&setup);

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_lane_boundary(&mut setup, &tmp.path().join("boundary"));

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error
    );

    let (_slot, _intercept, coefficients) = setup
        .fcf
        .active_cuts(deciding_pool)
        .next()
        .expect("the deciding stage's pool must carry a generated cut after training");
    let lane_coeff = coefficients[lane];
    assert!(
        lane_coeff.abs() > TOL,
        "the deciding stage's cut must carry a nonzero coefficient on the post-horizon lane; \
         got {lane_coeff}"
    );
    assert!(
        (lane_coeff - BETA).abs() < TOL,
        "the carried terminal boundary must propagate the coefficient unchanged (out = in, \
         unit scale): expected {BETA}, got {lane_coeff}"
    );
}

// ── Fuel-exclusive composition: β prices the state, the decision books fuel ──

/// The terminal `β·state` valuation and the delivery-anchored fuel
/// booking live on DIFFERENT LP columns, so they compose additively with no
/// double-count: at the deciding stage the decision column carries the fuel in
/// its objective while the lane's `commit_out` column carries no objective at
/// all (β reaches it through the boundary cut ROW, never the objective).
#[test]
fn terminal_valuation_and_decision_fuel_are_disjoint_columns() {
    let setup = build_setup_in_code(build_system(true, false), &config());
    let deciding_stage = 0;

    let decision_range = setup.stage_ctx().geometry_per_stage[deciding_stage]
        .commitment_decision
        .clone();
    assert_eq!(
        decision_range.len(),
        1,
        "one decision column at the decider"
    );
    let decision_col = decision_range.start;

    let lane = lane_state_index(&setup);
    let lane_out_col = setup
        .stage_state()
        .state_to_lp_column(StateDim::new(lane))
        .get();

    assert_ne!(
        decision_col, lane_out_col,
        "the fuel-bearing decision column and the beta-priced lane state column must be distinct"
    );

    let template = &setup.stage_ctx().templates[deciding_stage];
    assert!(
        template.objective[decision_col].abs() > TOL,
        "the decision column must book the delivery-anchored fuel (nonzero objective); got {}",
        template.objective[decision_col]
    );
    assert_eq!(
        template.objective[lane_out_col], 0.0,
        "the lane's commit_out column must carry no fuel term — beta prices it via the cut row"
    );
}

// ── one shared boundary prices each fanned leaf by its own lane state ──

/// Extends the shared-boundary fan to the `commit_out`/`commit_in`
/// lanes: both terminal fan leaves resolve to ONE shared pool carrying the
/// single injected boundary, and each leaf's terminal value is priced by ITS
/// OWN committed MW from that one shared `β` — `θ = α + β·x_leaf`, distinct
/// across leaves committing distinct MW.
#[test]
fn shared_boundary_prices_each_fanned_leaf_by_its_own_lane_state() {
    let mut setup = build_setup_in_code(build_system(true, true), &config());

    let leaves = leaf_positions(&setup.node_graph);
    assert_eq!(
        leaves.len(),
        2,
        "fixture must declare a 2-leaf terminal fan"
    );
    let leaf_pool_ids: Vec<usize> = leaves
        .iter()
        .map(|&pos| setup.node_graph.nodes[pos].pool_id)
        .collect();
    assert_eq!(
        leaf_pool_ids[0], leaf_pool_ids[1],
        "both terminal fan leaves must share ONE pool"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_lane_boundary(&mut setup, &tmp.path().join("boundary"));

    let shared_pool_id = leaf_pool_ids[0];
    assert_eq!(
        shared_pool_id,
        setup.fcf.pools.len() - 1,
        "the shared leaf pool is the terminal pool the boundary overwrites"
    );
    let pool = &setup.fcf.pools[shared_pool_id];
    let injected: Vec<(f64, Vec<f64>)> = pool
        .active_cuts()
        .map(|(_, intercept, coeffs)| (intercept, coeffs.to_vec()))
        .collect();
    let lane = lane_state_index(&setup);
    let mut expected_coeffs = vec![0.0_f64; setup.fcf.state_dimension];
    expected_coeffs[lane] = BETA;
    assert_eq!(
        injected,
        vec![(ALPHA, expected_coeffs)],
        "the shared pool must carry exactly the one injected lane boundary"
    );

    let template = freeze_terminal_template(&setup, shared_pool_id);
    let x_leaf1 = 1.5;
    let x_leaf2 = 4.0;
    let theta_leaf1 = terminal_theta(
        &setup,
        &template,
        pool,
        NodeId(1),
        &lane_pin(&setup, x_leaf1),
    );
    let theta_leaf2 = terminal_theta(
        &setup,
        &template,
        pool,
        NodeId(2),
        &lane_pin(&setup, x_leaf2),
    );

    for (label, x, theta) in [
        ("leaf1", x_leaf1, theta_leaf1),
        ("leaf2", x_leaf2, theta_leaf2),
    ] {
        let expected = ALPHA + BETA * x;
        assert!(
            (theta - expected).abs() < TOL,
            "{label}: the shared beta must price this leaf by its OWN committed MW: \
             expected {expected}, got {theta}"
        );
    }
    assert!(
        (theta_leaf1 - theta_leaf2).abs() > TOL,
        "leaves committing different MW must be priced to different terminal values"
    );
}

// ── a declared post-horizon window solves the first stage, no walk-off ──

/// End-to-end sizing pin: a study declaring a post-horizon commitment window
/// solves its first stage without a `fill_col_state_patches` Vec walk-off. One
/// training iteration exercises the forward pass's stage-0 solve through the
/// full patch pipeline (`StageSolvePrep::run` → `fill_col_state_patches`), which
/// panics on an undersized `+ n_commitment` buffer; a clean `Ok` proves the
/// buffer covers the post-horizon lane end-to-end.
#[test]
fn post_horizon_commitment_first_stage_solves_without_patch_walk_off() {
    let mut setup = build_setup_in_code(build_system(true, false), &config());
    assert_eq!(
        setup.stage_state().n_commitment,
        1,
        "the fixture must declare exactly one post-horizon commitment window"
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("first stage must solve without a fill_col_state_patches panic");
    assert!(
        outcome.error.is_none(),
        "training over the post-horizon-commitment study must complete; got {:?}",
        outcome.error
    );
}

// ── Inert/bootstrap: no boundary → θ bounded at zero, lane columns present ──

/// Without an injected boundary the terminal carries no future cost, so `θ` is
/// bounded at zero regardless of the committed MW — yet the post-horizon lane
/// columns are structurally present (kept open at every stage, per the
/// keep-live contract), ready to be priced once a boundary is injected.
#[test]
fn no_boundary_leaves_theta_zero_with_lane_columns_present() {
    let setup = build_setup_in_code(build_system(true, false), &config());

    assert_eq!(
        setup.stage_state().n_commitment,
        1,
        "the lane must exist in the state layout"
    );
    assert_eq!(
        setup.fcf.state_dimension, 2,
        "the post-horizon lane column must be present in the state dimension"
    );

    let terminal_pool_id = setup.fcf.pools.len() - 1;
    let pool = &setup.fcf.pools[terminal_pool_id];
    assert_eq!(pool.populated(), 0, "no boundary cut must be loaded");

    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let theta = terminal_theta(&setup, &template, pool, NodeId(1), &lane_pin(&setup, 2.0));
    assert!(
        theta.abs() < 1e-9,
        "with no boundary cut, theta must be 0 regardless of the committed MW: got {theta}"
    );
}
