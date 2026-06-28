//! End-to-end integration tests for generic constraints referencing
//! `anticipated_decision(N)`.
//!
//! ## AC-15: Constrained training with `anticipated_decision <= 20.0`
//!
//! A 4-stage, K=2, no-hydro deterministic fixture (1 anticipated thermal, 1
//! backup thermal, no hydro). Load = 50 MW. Backup cost = 100 $/MWh, anticipated
//! cost = 10 $/MWh. Past commitments are both zero so stage-0 anticipated decision
//! delivers at stage 2, and stage-1 decision delivers at stage 3.
//!
//! Without constraint: optimal decision is `d_ant_0 = 50 MW`, eliminating all
//! backup cost at stage 2. Constrained to `d_ant_0 ≤ 20 MW`, backup must cover
//! 30 MW at stage 2, raising the lower bound.
//!
//! Assertions:
//! - Training completes without error in both constrained and baseline runs.
//! - Constrained final LB is strictly greater than baseline LB, proving the
//!   constraint is economically binding.
//!
//! ## AC-16: Semantic-validator rejects constraint on non-anticipated thermal
//!
//! Same fixture topology, but the `generic_constraints.json` references thermal id=3
//! (the backup thermal, which is NOT anticipated). The `cobre_io::validate_case`
//! pipeline is invoked on a temp case directory. The test asserts that loading
//! fails with a `BusinessRuleViolation` error whose message contains the
//! substring "not an anticipated thermal".

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

use std::path::Path;

use chrono::NaiveDate;
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ConstraintExpression, ConstraintSense, ContractStageBounds, EntityId, GenericConstraint,
    HydroStageBounds, HydroStagePenalties, InitialConditions, LineStageBounds, LineStagePenalties,
    LinearTerm, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds,
    ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties, SlackConfig, SystemBuilder,
    ThermalStageBounds, VariableRef,
    entities::{
        bus::{Bus, DeficitSegment},
        thermal::{AnticipatedConfig, Thermal},
    },
    scenario::{LoadModel, SamplingScheme},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
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
// Fixture parameters (AC-15 closed-form derivation)
// ---------------------------------------------------------------------------

/// Number of study stages.
const N_STAGES: usize = 4;
/// Anticipated thermal lead time (stages).
const K_MAX: usize = 2;
/// Block duration (hours). 1 hour keeps cost magnitudes small and derivable.
const BLOCK_HOURS: f64 = 1.0;
/// Constant deterministic load (MW).
const LOAD_MW: f64 = 50.0;
/// Anticipated thermal capacity (MW).
const ANT_MAX_MW: f64 = 100.0;
/// Anticipated thermal cost ($/MWh). Must be less than BACKUP_COST.
const ANT_COST: f64 = 10.0;
/// Backup thermal capacity (MW).
const BACKUP_MAX_MW: f64 = 200.0;
/// Backup thermal cost ($/MWh).
const BACKUP_COST: f64 = 100.0;
/// Deficit cost ($/MWh). Well above BACKUP_COST so deficit is never optimal.
const DEFICIT_COST: f64 = 1000.0;
/// Constraint upper bound on anticipated_decision (MW). Strictly below LOAD_MW
/// so the unconstrained optimum (d_ant = LOAD_MW) is infeasible under the constraint.
const CONSTRAINT_BOUND_MW: f64 = 20.0;

/// EntityId of the anticipated thermal.
const ANT_THERMAL_ID: EntityId = EntityId(2);
/// EntityId of the backup thermal. Non-anticipated.
const BACKUP_THERMAL_ID: EntityId = EntityId(3);
/// EntityId of the bus.
const BUS_ID: EntityId = EntityId(1);

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build the `N_STAGES`-stage no-hydro system (1 bus, 1 anticipated thermal, 1
/// backup thermal) with optional generic constraints + resolved stage bounds.
/// Always feasible: the backup thermal alone covers `LOAD_MW`.
fn build_system(
    generic_constraints: Vec<GenericConstraint>,
    generic_bounds: ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    let bus = Bus {
        id: BUS_ID,
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: DEFICIT_COST,
        }],
        excess_cost: 0.0,
    };

    let thermal_ant = Thermal {
        id: ANT_THERMAL_ID,
        name: "T_ant".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: BUS_ID,
        min_generation_mw: 0.0,
        max_generation_mw: ANT_MAX_MW,
        cost_per_mwh: ANT_COST,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let thermal_backup = Thermal {
        id: BACKUP_THERMAL_ID,
        name: "T_backup".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: BUS_ID,
        min_generation_mw: 0.0,
        max_generation_mw: BACKUP_MAX_MW,
        cost_per_mwh: BACKUP_COST,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: BLOCK_HOURS,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: BUS_ID,
            stage_id: i as i32,
            mean_mw: LOAD_MW,
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: K_MAX,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                max_diversion_m3s: None,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
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

    // SystemBuilder sorts thermals by EntityId ascending, so thermal_idx 0 = id=2
    // (anticipated) and thermal_idx 1 = id=3 (backup); these indices feed
    // thermal_bounds_mut below. The thermal stage axis runs N_STAGES + K_MAX to
    // cover delivery-stage lookups in fill_anticipated_columns.
    let thermal_axis = N_STAGES + K_MAX;
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(0, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: ANT_MAX_MW,
            cost_per_mwh: ANT_COST,
        };
        *bounds.thermal_bounds_mut(1, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: BACKUP_MAX_MW,
            cost_per_mwh: BACKUP_COST,
        };
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
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
    );

    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: ANT_THERMAL_ID,
            values_mw: vec![0.0, 0.0],
        }],
        recent_observations: vec![],
    };

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant, thermal_backup])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions);

    if !generic_constraints.is_empty() {
        builder = builder
            .generic_constraints(generic_constraints)
            .resolved_generic_bounds(generic_bounds);
    }

    builder.build().expect("build_system: valid")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a minimal [`Config`] for this fixture.
fn build_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
            },
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            forward_passes: Some(1),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
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
// AC-15: Constrained training lower-bound is strictly worse than unconstrained
// ---------------------------------------------------------------------------

/// AC-15: A 4-stage, K=2, no-hydro deterministic fixture with a generic
/// constraint `anticipated_decision(2) <= 20.0`.
///
/// ## Closed-form expected behaviour
///
/// With K=2 and N=4 stages, anticipated decisions are active at stages 0 and 1:
/// - Stage 0 decision (`d0`) delivers at stage 2.
/// - Stage 1 decision (`d1`) delivers at stage 3.
///
/// Past commitments are zero, so no pre-study deliveries at stages 0 or 1.
///
/// Without constraint: optimal `d0 = d1 = LOAD_MW (50 MW)`, fully covering
/// load at delivery stages with cheap anticipated dispatch, eliminating backup.
///
/// With constraint (`d0, d1 <= 20 MW`): only 20 MW of cheap anticipated
/// dispatch is available at stages 2 and 3; the remaining 30 MW at each
/// delivery stage must use the backup at BACKUP_COST (100 $/MWh), raising LB.
#[test]
fn anticipated_decision_constraint_raises_lb() {
    let constraint = GenericConstraint {
        id: EntityId(1),
        name: "cap_ant_decision".to_string(),
        description: Some(format!(
            "Cap anticipated commitment for T_ant at {CONSTRAINT_BOUND_MW} MW"
        )),
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::AnticipatedDecision {
                    thermal_id: ANT_THERMAL_ID,
                },
            )],
        },
        sense: ConstraintSense::LessEqual,
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let config = build_config();
    let comm = StubComm;

    // Constraint id=1 carries a bound at every study stage. At stages 2 and 3 the
    // d_ant column is inactive ([0,0]) so its row has no LP effect, but applying it
    // uniformly is harmless and keeps the setup simple.
    let id_map: std::collections::HashMap<i32, usize> = [(1_i32, 0_usize)].into_iter().collect();
    let raw_bounds: Vec<(i32, i32, Option<i32>, f64)> = (0..N_STAGES as i32)
        .map(|stage_id| (1_i32, stage_id, None::<i32>, CONSTRAINT_BOUND_MW))
        .collect();
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, raw_bounds.into_iter());

    let constrained_system = build_system(vec![constraint], generic_bounds);
    let mut constrained_setup = build_setup(constrained_system, &config);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let constrained_outcome = constrained_setup
        .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
        .expect("constrained train must not return Err");

    assert!(
        constrained_outcome.error.is_none(),
        "constrained training error: {:?}",
        constrained_outcome.error
    );
    let constrained_lb = constrained_outcome.result.final_lb;

    let baseline_system = build_system(vec![], ResolvedGenericConstraintBounds::empty());
    let mut baseline_setup = build_setup(baseline_system, &config);
    let mut baseline_solver = ActiveSolver::new().expect("ActiveSolver::new baseline");

    let baseline_outcome = baseline_setup
        .train(
            &mut baseline_solver,
            &comm,
            10,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("baseline train must not return Err");

    assert!(
        baseline_outcome.error.is_none(),
        "baseline training error: {:?}",
        baseline_outcome.error
    );
    let baseline_lb = baseline_outcome.result.final_lb;

    // The constraint limits d0, d1 to CONSTRAINT_BOUND_MW (20 MW) instead of
    // the optimal LOAD_MW (50 MW). At each delivery stage (2 and 3), 30 MW must
    // use backup at BACKUP_COST $/MWh. So the constrained run costs strictly more.
    assert!(
        constrained_lb > baseline_lb,
        "constrained LB ({constrained_lb:.6}) must be strictly greater than \
         baseline LB ({baseline_lb:.6}) — constraint is not binding"
    );

    let lb_delta = constrained_lb - baseline_lb;
    assert!(
        lb_delta > 1.0,
        "LB delta ({lb_delta:.6}) is too small to confirm economically meaningful binding"
    );
}

// ---------------------------------------------------------------------------
// AC-16: Semantic validator rejects anticipated_decision on non-anticipated thermal
// ---------------------------------------------------------------------------

/// AC-16: A case loaded via `cobre_io::validate_case` with a generic constraint
/// `anticipated_decision(3)` where thermal id=3 is NOT an anticipated thermal.
///
/// The semantic validator (`check_anticipated_decision_target_is_anticipated`)
/// must reject this, returning `Err` with a message containing "not an anticipated
/// thermal".
#[test]
fn anticipated_decision_on_non_anticipated_thermal_rejected_by_validator() {
    use std::fs;

    let tmp = tempfile::tempdir().expect("tempdir");
    let case_dir = tmp.path();

    // ── system/ ───────────────────────────────────────────────────────────────
    let system_dir = case_dir.join("system");
    fs::create_dir_all(&system_dir).expect("create system dir");

    fs::write(
        system_dir.join("buses.json"),
        r#"{"buses":[{"id":1,"name":"B1","deficit_segments":[{"depth_mw":null,"cost":1000.0}]}]}"#,
    )
    .expect("write buses.json");

    fs::write(system_dir.join("hydros.json"), r#"{"hydros":[]}"#).expect("write hydros.json");

    fs::write(system_dir.join("lines.json"), r#"{"lines":[]}"#).expect("write lines.json");

    fs::write(
        system_dir.join("thermals.json"),
        r#"{
  "thermals": [
    {
      "id": 2,
      "name": "T_ant",
      "bus_id": 1,
      "generation": { "min_mw": 0.0, "max_mw": 100.0 },
      "cost_per_mwh": 10.0,
      "anticipated_config": { "lead_stages": 2 }
    },
    {
      "id": 3,
      "name": "T_backup",
      "bus_id": 1,
      "generation": { "min_mw": 0.0, "max_mw": 200.0 },
      "cost_per_mwh": 100.0
    }
  ]
}"#,
    )
    .expect("write thermals.json");

    // ── constraints/ ─────────────────────────────────────────────────────────
    let constraints_dir = case_dir.join("constraints");
    fs::create_dir_all(&constraints_dir).expect("create constraints dir");

    // Constraint references thermal id=3 (non-anticipated) via anticipated_decision.
    // Must be rejected by the semantic validator (rule 17).
    fs::write(
        constraints_dir.join("generic_constraints.json"),
        r#"{
  "constraints": [
    {
      "id": 1,
      "name": "bad_constraint",
      "expression": "anticipated_decision(3)",
      "sense": "<=",
      "slack": { "enabled": false }
    }
  ]
}"#,
    )
    .expect("write generic_constraints.json");

    // Write a minimal constraint-bounds parquet (required by the pipeline when
    // generic_constraints.json is present).
    write_constraint_bounds_parquet(
        &constraints_dir.join("generic_constraint_bounds.parquet"),
        1,    // constraint_id
        0,    // stage_id
        25.0, // bound
    )
    .expect("write generic_constraint_bounds.parquet");

    // ── stages.json ───────────────────────────────────────────────────────────
    fs::write(
        case_dir.join("stages.json"),
        r#"{
  "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0 },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 744 }],
      "num_scenarios": 1
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 672 }],
      "num_scenarios": 1
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 744 }],
      "num_scenarios": 1
    },
    {
      "id": 3,
      "start_date": "2024-04-01",
      "end_date": "2024-05-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 720 }],
      "num_scenarios": 1
    }
  ]
}"#,
    )
    .expect("write stages.json");

    // ── initial_conditions.json ───────────────────────────────────────────────
    // Anticipated thermal (id=2, K=2) requires past_anticipated_commitments.
    fs::write(
        case_dir.join("initial_conditions.json"),
        r#"{
  "storage": [],
  "filling_storage": [],
  "past_anticipated_commitments": [
    { "thermal_id": 2, "values_mw": [0.0, 0.0] }
  ]
}"#,
    )
    .expect("write initial_conditions.json");

    // ── penalties.json ────────────────────────────────────────────────────────
    fs::write(
        case_dir.join("penalties.json"),
        r#"{
  "bus": {
    "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
    "excess_cost": 0.01
  },
  "line": { "exchange_cost": 0.01 },
  "hydro": {
    "spillage_cost": 0.01,
    "turbined_cost": 0.01,
    "diversion_cost": 0.01,
    "storage_violation_below_cost": 10000.0,
    "filling_target_violation_cost": 10000.0,
    "turbined_violation_below_cost": 10000.0,
    "outflow_violation_below_cost": 10000.0,
    "outflow_violation_above_cost": 10000.0,
    "generation_violation_below_cost": 10000.0,
    "evaporation_violation_cost": 10000.0,
    "water_withdrawal_violation_cost": 10000.0
  },
  "non_controllable_source": { "curtailment_cost": 0.005 }
}"#,
    )
    .expect("write penalties.json");

    // ── config.json ───────────────────────────────────────────────────────────
    fs::write(
        case_dir.join("config.json"),
        r#"{
  "training": {
    "forward_passes": 1,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 2 }]
  },
  "simulation": { "enabled": false, "num_scenarios": 1 },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#,
    )
    .expect("write config.json");

    let result = cobre_io::validate_case(case_dir);

    assert!(
        result.is_err(),
        "validate_case should fail when anticipated_decision references a non-anticipated thermal"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("not an anticipated thermal"),
        "error message must contain 'not an anticipated thermal', got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Helper: write a minimal `generic_constraint_bounds.parquet`
// ---------------------------------------------------------------------------

/// Write a single-row constraint-bounds parquet; `block_id` is null (all blocks).
fn write_constraint_bounds_parquet(
    path: &Path,
    constraint_id: i32,
    stage_id: i32,
    bound: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![constraint_id])),
            Arc::new(Int32Array::from(vec![stage_id])),
            Arc::new(Int32Array::new_null(1)),
            Arc::new(Float64Array::from(vec![bound])),
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
