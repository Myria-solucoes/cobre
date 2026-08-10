//! Terminal FCF pricing of the commitment block (ticket-006): the block's
//! state columns join the cut-state projection so a loaded boundary cut's
//! coefficient `β` reaches the terminal `θ` — `θ` responds to the committed
//! MW pinned on the block's incoming column, and the shared boundary prices
//! every terminal fan node against its OWN block state from the ONE shared
//! `β` (ticket-004's shared-fan injection topology).
//!
//! The fixture is deliberately minimal: zero hydros, one anticipated thermal
//! declared `LeadTime` with a physical lead shorter than either stage's
//! duration (so its OWN in-horizon delivery schedule is `K = 0`
//! self-delivered — no regular ring, `k_max == 0`) plus one
//! `future_anticipated_deliveries` window whose decider resolves to stage 0
//! (non-terminal — the terminal's own commitment-block row is therefore a
//! CARRY, `out = in`, so pinning the incoming column via
//! `patch_backward_opening_for_probe` controls the terminal's committed MW
//! directly). With `hydro_count = 0`, `n_buckets = 0`, and `k_max = 0`, the
//! block is the STUDY's entire state vector (`n_state == 1`), isolating the
//! pricing arithmetic from every other state family.

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
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, Transition};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, FutureAnticipatedDelivery,
    HorizonGraph, HydroBlockBounds, HydroStageBounds, InitialConditions, LineBlockBounds,
    PumpingBlockBounds, ResolvedBounds, System, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
};
use cobre_io::{
    FORMAT_VERSION, GraphManifest, ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord,
    ProducerBlock, StageCutsPayload, write_policy_checkpoint,
};
use cobre_sddp::setup::{NodeId, NodePos, StageIdx};
use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
use cobre_sddp::workspace::SolverWorkspace;
use cobre_sddp::{inject_boundary_cuts, load_boundary_cuts};
use cobre_solver::{ActiveSolver, SolverInterface};

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// Delta well under either stage's 720h duration: every WITHIN-horizon
/// delivery is `K = 0` self-delivered (`c(m) = m`), so the regular ring never
/// activates (`k_max == 0`) — isolating the terminal commitment block as the
/// study's only state dimension.
const DELTA_HOURS: f64 = 10.0;
/// `EntityType::AnticipatedThermalState` discriminant (`schemas/policy.fbs`):
/// the commitment block reuses this type (ratified, no new discriminant).
const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Two 720h stages (id 0 non-terminal, id 1 terminal).
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    (0..2)
        .map(|i| {
            let s = start + chrono::TimeDelta::days(30 * i as i64);
            make_stage(
                i,
                StageSpec {
                    start_date: s,
                    end_date: s + chrono::TimeDelta::days(30),
                    blocks: vec![cobre_core::temporal::Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 720.0,
                    }],
                    ..Default::default()
                },
            )
        })
        .collect()
}

/// One declared post-horizon window, decider resolved to stage 0 (Trunk, not
/// terminal): `delivery_end = start + 3 days` (72h), `target = 72 -
/// DELTA_HOURS = 62`, inside `(boundaries[0]=0, boundaries[1]=720)`.
fn future_delivery() -> FutureAnticipatedDelivery {
    let start = study_start();
    FutureAnticipatedDelivery {
        thermal_id: THERMAL_ID,
        delivery_start: start + chrono::TimeDelta::days(2),
        delivery_end: start + chrono::TimeDelta::days(3),
        min_mw: 0.0,
        max_mw: 100.0,
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
            k_max: 0,
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

/// Two-leaf terminal fan: root (id 0, stage 0) branches into leaves (ids 1
/// and 2, stage 1) — both terminal, sharing ONE pool
/// (`build_node_graph`'s leaf-sharing rule, confirmed by ticket-004).
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

/// `with_window` toggles the single post-horizon commitment window;
/// `fanned` toggles the two-leaf terminal fan (chain otherwise).
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

/// Write a synthetic single-cut boundary checkpoint carrying coefficient
/// `beta` on the block's (sole) state dimension and intercept `alpha`. No
/// entity manifest (`&[]`): the loader's identity check short-circuits with
/// a warning rather than an error (the established probe pattern —
/// `shared_boundary_terminal_fan_probe.rs`), so this test controls only the
/// dimension the ticket-006 wiring is about, not entity-identity matching
/// (covered separately by the `policy_export` slot-identity unit tests).
fn write_synthetic_boundary(dir: &Path, state_dimension: u32, alpha: f64, beta: f64) {
    let coefficients = vec![beta; state_dimension as usize];
    let cuts = vec![PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: alpha,
        coefficients: &coefficients,
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

/// Solve the terminal stage's LP with the block's (sole) state dimension
/// pinned to `x`, against `pool`, and return `theta`'s primal value.
/// `raw_noise = &[]`: `hydro_count == 0` so the noise transform has no
/// dimension to iterate.
/// Build the terminal pool's FROZEN LP template: the base structural
/// template plus one literal row per active cut in `pool`, exactly as
/// production freezes a pool once per iteration
/// (`training/session/iteration_scratch.rs`, `freeze_rows_into_template`).
/// `solve_stage_for_probe`'s `pool` argument only drives BASIS
/// reconstruction on a stored warm start — it never appends cut rows to a
/// cold solve, so pricing an injected cut requires freezing it into the
/// loaded model first.
fn freeze_terminal_template(
    setup: &cobre_sddp::StudySetup,
    pool_id: usize,
) -> cobre_solver::StageTemplate {
    let state = setup.stage_state();
    let terminal_stage = setup.num_stages() - 1;
    let ctx = setup.stage_ctx();
    let base = &ctx.templates[terminal_stage];

    let cut_state = cobre_sddp::indexer::CutStateProjection::new(
        state,
        cobre_core::temporal::StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );

    let mut batch = cobre_solver::RowBatch {
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

    let mut frozen = cobre_solver::StageTemplate::empty();
    let mut scratch = cobre_solver::FreezeScratch::new();
    cobre_solver::freeze_rows_into_template(base, &batch, &mut frozen, &mut scratch);
    frozen
}

/// Patch the terminal stage's block dimension to `x` and solve the FROZEN
/// `template` (see [`freeze_terminal_template`]), returning theta's primal
/// value.
fn terminal_theta(
    setup: &cobre_sddp::StudySetup,
    template: &cobre_solver::StageTemplate,
    pool: &cobre_sddp::CutPool,
    node_id: NodeId,
    x: f64,
) -> f64 {
    let comm = StubComm;
    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let state = setup.stage_state();
    let terminal_stage = setup.num_stages() - 1;

    ws.solver.reset_solver_state();
    ws.solver.load_model(template);
    patch_backward_opening_for_probe(ws, &ctx, &training_ctx, StageIdx(terminal_stage), &[x], &[])
        .expect("StageSolvePrep::run must not error on the minimal fixture");

    let view = solve_stage_for_probe(ws, &ctx, pool, None, StageIdx(terminal_stage), 0, node_id)
        .expect("terminal stage solve must not error");
    view.primal[state.theta]
}

/// Every leaf `NodePos` in `graph` — a node with no successors.
fn leaf_positions(graph: &cobre_sddp::setup::NodeGraph) -> Vec<NodePos> {
    graph
        .nodes
        .iter_indexed()
        .filter(|&(pos, _)| graph.successors[pos].is_empty())
        .map(|(pos, _)| pos)
        .collect()
}

// -- AC1: the block joins the projection and widens fcf.state_dimension --

#[test]
fn commitment_block_widens_the_terminal_projection_and_fcf_dimension() {
    let cfg = config();
    let setup_without = build_setup_in_code(build_system(false, false), &cfg);
    let setup_with = build_setup_in_code(build_system(true, false), &cfg);
    let system_without = build_system(false, false);
    let system_with = build_system(true, false);

    assert_eq!(
        setup_with.fcf.state_dimension,
        setup_without.fcf.state_dimension + 1,
        "a declared commitment window must widen fcf.state_dimension by exactly \
         the block's width (1)"
    );

    let manifest_without = setup_without.build_terminal_entity_manifest(&system_without);
    let manifest_with = setup_with.build_terminal_entity_manifest(&system_with);
    assert_eq!(
        manifest_with.len(),
        manifest_without.len() + 1,
        "the terminal manifest must grow by exactly one slot"
    );
    let block_slot = manifest_with.last().expect("manifest must be non-empty");
    assert_eq!(
        block_slot.entity_type, ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
        "the commitment-block slot reuses the anticipated-thermal-state type"
    );
    assert_eq!(block_slot.entity_id, THERMAL_ID.0);
}

// -- AC2: theta responds to beta*x, K-normalized (read back, relative tolerance) --

#[test]
fn boundary_cut_prices_the_committed_mw_by_beta_times_x() {
    let cfg = config();
    let system = build_system(true, false);
    let mut setup = build_setup_in_code(system, &cfg);

    let state_dimension = setup.fcf.state_dimension as u32;
    assert_eq!(
        state_dimension, 1,
        "fixture must isolate the block as n_state"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("boundary_source");
    let alpha = 5.0;
    let beta_written = 3.0;
    write_synthetic_boundary(&source_dir, state_dimension, alpha, beta_written);

    let mut warnings: Vec<String> = Vec::new();
    let boundary_cuts = load_boundary_cuts(
        &source_dir,
        0,
        state_dimension,
        &[],
        None,
        1.0,
        &mut |msg| warnings.push(msg.to_string()),
    )
    .expect("boundary cut must load");
    inject_boundary_cuts(&mut setup, &boundary_cuts);

    let terminal_pool_id = setup.fcf.pools.len() - 1;
    let pool = &setup.fcf.pools[terminal_pool_id];
    let (_, _, beta_internal) = pool
        .active_cuts()
        .next()
        .expect("exactly one active cut after injection");
    let beta_internal = beta_internal[0];

    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let node_id = NodeId(1);
    let x0 = 0.0;
    let x1 = 2.0;
    let theta0 = terminal_theta(&setup, &template, pool, node_id, x0);
    let theta1 = terminal_theta(&setup, &template, pool, node_id, x1);

    assert!(
        theta0 > 0.0 && theta1 > 0.0,
        "both solves must land on the injected cut's binding regime (theta0={theta0}, \
         theta1={theta1})"
    );

    let expected_delta = beta_internal * (x1 - x0);
    let actual_delta = theta1 - theta0;
    let rel_err = (actual_delta - expected_delta).abs() / expected_delta.abs().max(1.0);
    assert!(
        rel_err < 1e-6,
        "theta must respond to the committed MW by beta*x (K-normalized): expected delta \
         {expected_delta}, got {actual_delta} (beta_internal={beta_internal}, rel_err={rel_err})"
    );
}

// -- AC3: no boundary loaded => theta = 0 on the block dimension, columns still present --

#[test]
fn no_boundary_leaves_theta_zero_but_block_columns_present() {
    let cfg = config();
    let system = build_system(true, false);
    let setup = build_setup_in_code(system, &cfg);

    assert_eq!(
        setup.fcf.state_dimension, 1,
        "the block column must still be present in the state dimension"
    );

    let terminal_pool_id = setup.fcf.pools.len() - 1;
    let pool = &setup.fcf.pools[terminal_pool_id];
    assert_eq!(pool.populated(), 0, "no boundary cut must be loaded");

    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let theta = terminal_theta(&setup, &template, pool, NodeId(1), 2.0);
    assert!(
        theta.abs() < 1e-9,
        "with no boundary cut, theta must be 0 regardless of the committed MW: got {theta}"
    );
}

// -- AC4: fanned terminal, one shared beta, two nodes reflect their own beta*x --

#[test]
fn shared_boundary_prices_each_fanned_node_by_its_own_committed_mw() {
    let cfg = config();
    let system = build_system(true, true);
    let mut setup = build_setup_in_code(system, &cfg);

    let leaves = leaf_positions(&setup.node_graph);
    assert_eq!(
        leaves.len(),
        2,
        "fixture must declare a genuine 2-leaf terminal fan"
    );
    let leaf_pool_ids: Vec<usize> = leaves
        .iter()
        .map(|&pos| setup.node_graph.nodes[pos].pool_id)
        .collect();
    assert_eq!(
        leaf_pool_ids[0], leaf_pool_ids[1],
        "both terminal fan leaves must share ONE pool"
    );

    let state_dimension = setup.fcf.state_dimension as u32;
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_dir = tmp.path().join("boundary_source_fan");
    let alpha = 5.0;
    let beta_written = 3.0;
    write_synthetic_boundary(&source_dir, state_dimension, alpha, beta_written);

    let mut warnings: Vec<String> = Vec::new();
    let boundary_cuts = load_boundary_cuts(
        &source_dir,
        0,
        state_dimension,
        &[],
        None,
        1.0,
        &mut |msg| warnings.push(msg.to_string()),
    )
    .expect("boundary cut must load");
    inject_boundary_cuts(&mut setup, &boundary_cuts);

    let shared_pool_id = leaf_pool_ids[0];
    let pool = &setup.fcf.pools[shared_pool_id];
    let (_, _, beta_internal) = pool
        .active_cuts()
        .next()
        .expect("exactly one active cut after injection");
    let beta_internal = beta_internal[0];

    let template = freeze_terminal_template(&setup, shared_pool_id);
    let theta_baseline = terminal_theta(&setup, &template, pool, NodeId(1), 0.0);

    let x_leaf1 = 1.5;
    let x_leaf2 = -1.0;
    let theta_leaf1 = terminal_theta(&setup, &template, pool, NodeId(1), x_leaf1);
    let theta_leaf2 = terminal_theta(&setup, &template, pool, NodeId(2), x_leaf2);

    for (label, x, theta) in [
        ("leaf1", x_leaf1, theta_leaf1),
        ("leaf2", x_leaf2, theta_leaf2),
    ] {
        let expected_delta = beta_internal * x;
        let actual_delta = theta - theta_baseline;
        let rel_err = (actual_delta - expected_delta).abs() / expected_delta.abs().max(1.0);
        assert!(
            rel_err < 1e-6,
            "{label}: theta must reflect its own committed MW from the SHARED beta: expected \
             delta {expected_delta}, got {actual_delta} (rel_err={rel_err})"
        );
    }
    assert!(
        (theta_leaf1 - theta_leaf2).abs() > 1e-6,
        "leaf1 and leaf2 committed different MW, so their priced terminal values must differ"
    );
}
