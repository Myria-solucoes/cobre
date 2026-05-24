//! Analytical verification of backward-pass cut-coefficient propagation for an
//! anticipated thermal with `lead_stages = 2` in a 3-stage system.
//!
//! ## Closed-form derivation
//!
//! This fixture extends `anticipated_backward_cut_k1.rs` to a 3-stage system
//! with `K_i = 2`, exercising one stage of in-between ring-buffer propagation
//! between decision (stage 0) and delivery (stage 2).
//!
//! - One anticipated thermal (K=2, cost `c_ant`, max `max_gen_ant`) and one
//!   regular thermal (cost `c_reg`, max `max_gen_reg`) at a single bus.
//! - Loads `D_0 = 5`, `D_1 = 10`, `D_2 = 30` MW; one-hour block per stage.
//! - `max_par_order = 0`, so the anticipated-state block starts at
//!   `indexer.anticipated_state.start = 0`.
//!
//! Stage-2 LP (delivery stage; fishing constraint ACTIVE because `K=2 <= 2`):
//!
//! ```text
//!   min  (c_reg/K) gt_reg + theta
//!   s.t. gt_reg + gt_ant = D_2                           (load balance)
//!        gt_ant - x_anticipated_state[slot=0] = 0        (fishing)
//!        x_anticipated_state[slot=0] + d_ant = x_hat_0   (state-fixing, slot 0)
//!        x_anticipated_state[slot=1] = x_hat_1           (state-fixing, slot 1)
//! ```
//!
//! Substituting: `gt_ant = x_state_0 = x_hat_0 - d_ant`, then
//! `gt_reg = D_2 - x_hat_0 + d_ant`. The dual on the slot-0 state-fixing row
//! is `dQ_2_scaled/dx_hat_0 = -c_reg/K` (per the same reasoning as the K=1
//! test; the `d_ant` decision at stage 2 has `t + K_i = 4` which is out of
//! range, so its objective coefficient is zero in the in-range case — but
//! even if non-zero, the analysis here is identical since `d_ant` is bounded
//! at 0 by the cost gradient).
//!
//! Ring-buffer propagation: the forward pass writes the stage-0 decision into
//! slot `K_i - 1 = 1` at stage 0's outgoing state. After one shift step (stage
//! 1 forward), slot 0 at stage 1's outgoing = slot 1 at stage 1's incoming =
//! stage-0 decision. Therefore the channel that carries the stage-0 decision
//! to stage 2's fishing constraint is `slot 1` at stage 0's outgoing.
//!
//! The stage-0 cut coefficient on the anticipated_state column at slot 1 must
//! therefore equal `-c_reg/K` (propagated through one stage of cuts). The
//! coefficient on slot 0 must be `0` (the slot-0 incoming-state value at
//! stage 0 is consumed by the seed at stage 0 and shifted out before the
//! fishing constraint engages at stage 2).
//!
//! ## Why `iterations = 5`
//!
//! Within a single SDDP iteration the backward pass processes stages in
//! reverse: `t = 1` runs before `t = 0`. The `t = 0` backward immediately
//! consumes the cut just added to `FCF[1]` via the per-stage cut batch, so the
//! propagated stage-2 subgradient reaches `FCF[0]` by the end of iteration 1
//! (the cut at `slot = 1`). Running 5 iterations provides a safety margin
//! against numerical effects from later trial points shifting the active
//! basis, without changing which cut the assertion targets.

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
    thermal::{AnticipatedConfig, Thermal},
};
use cobre_core::scenario::{LoadModel, SamplingScheme};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, InitialConditions,
    LineStageBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingStageBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalStageBounds,
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

// ---------------------------------------------------------------------------
// Numeric fixture (single source of truth).
// ---------------------------------------------------------------------------

const N_STAGES: usize = 3;
const K_MAX: usize = 2;
const BLOCK_HOURS: f64 = 1.0;
const N_ITERATIONS: u32 = 5;

const D_0: f64 = 5.0;
const D_1: f64 = 10.0;
const D_2: f64 = 30.0;
const C_REG: f64 = 100.0;
const C_ANT: f64 = 50.0;
const MAX_GEN_REG: f64 = 100.0;
const MAX_GEN_ANT: f64 = 50.0;

// Cuts are stored in scaled cost units; the LP-builder divides every non-theta
// objective coefficient by `COST_SCALE_FACTOR` (see lp_builder/mod.rs and
// template.rs). Duals therefore live in scaled units too, and the cut storage
// at backward.rs preserves that scaling end-to-end (forward.rs consumes them
// unrescaled).
const COST_SCALE_FACTOR: f64 = 1_000.0;

// Closed-form expected coefficients at stage 0 FCF for the propagated
// stage-0 decision (slot 1) and the irrelevant slot 0, in scaled cost units.
const EXPECTED_COEFF_SLOT1: f64 = -C_REG / COST_SCALE_FACTOR; // = -0.1
const EXPECTED_COEFF_SLOT0: f64 = 0.0;
const TOL: f64 = 1e-6;

// Thermal column ordering inside the built `System`. `System::build()` sorts
// thermals by `EntityId::0` ascending. Per R7, the regular thermal must sort
// before the anticipated thermal — keep regular at id 2 (sorts first) and
// anticipated at id 4 (sorts second).
const THERMAL_IDX_REG: usize = 0;
const THERMAL_IDX_ANT: usize = 1;
const REGULAR_ID: EntityId = EntityId(2);
const ANTICIPATED_ID: EntityId = EntityId(4);

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator (copied from anticipated_backward_cut_k1
// per ticket Patterns guidance: test independence over shared modules).
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
// System builder
// ---------------------------------------------------------------------------

/// Build the 3-stage K=2 analytical system described at the top of the module.
fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        // Deficit cost set safely above c_reg so the LP never prefers shedding load.
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    // thermal_idx 0 — regular thermal (marginal at stage 2).
    let thermal_reg = Thermal {
        id: REGULAR_ID,
        name: "T_reg".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_REG,
        cost_per_mwh: C_REG,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // thermal_idx 1 — anticipated thermal, K=2.
    let thermal_ant = Thermal {
        id: ANTICIPATED_ID,
        name: "T_ant".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_ANT,
        cost_per_mwh: C_ANT,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // R7: regular thermal must sort before anticipated thermal in System::build.
    assert!(
        thermal_reg.id.0 < thermal_ant.id.0,
        "R7: T_reg.id ({}) must be strictly less than T_ant.id ({}) so that \
         System::build's sort_by_key aligns thermal_idx with the bounds table",
        thermal_reg.id.0,
        thermal_ant.id.0,
    );

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
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: match i {
                0 => D_0,
                1 => D_1,
                _ => D_2,
            },
            std_mw: 0.0,
        })
        .collect();

    // No hydros — anticipated/regular thermals only. n_anticipated=1, k_max=2.
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
                filling_inflow_m3s: 0.0,
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

    // Per-thermal per-stage overrides across the full thermal stage axis
    // (length = n_stages + k_max = 5). K-padded delivery cells are read by
    // `fill_anticipated_decision_objective` for stage_idx + K_i lookups.
    let thermal_axis = N_STAGES + K_MAX;
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(THERMAL_IDX_REG, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GEN_REG,
            cost_per_mwh: C_REG,
        };
        *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GEN_ANT,
            cost_per_mwh: C_ANT,
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

    // Seed the anticipated ring buffer slots 0 and 1 with feasible (irrelevant)
    // values. With K=2 the ring has K_MAX = 2 entries: values_mw[0] delivers at
    // study stage 1, values_mw[1] delivers at study stage 2. Per R6 the choice
    // is irrelevant as long as it is feasible.
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: ANTICIPATED_ID,
            values_mw: vec![0.0, 0.0],
        }],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_reg, thermal_ant])
        .stages(stages)
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
                limit: N_ITERATIONS,
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
    .expect("build_stochastic_context");

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    StudySetup::new(&system, config, stochastic, hydro_models).expect("StudySetup::new")
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "ticket-016 re-derives K=2 cut coefficients under always-active fishing"]
fn three_stage_k2_anticipated_cut_coefficient_propagates_correctly() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new");

    // Run N_ITERATIONS = 5 training iterations so the FCF at stage 0 receives
    // a non-zero cut propagated from stage 2 via stage 1.
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            N_ITERATIONS as usize,
            HighsSolver::new,
            None,
            None,
        )
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error,
    );

    // ── AC-2: at least one active cut at stage 0 FCF ──────────────────
    let pool0 = &setup.fcf.pools[0];
    let active_count = pool0.active_count();
    assert!(
        active_count >= 1,
        "AC-2: stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
    );

    // Locate the anticipated_state indices inside the state vector.
    // For n_hydros = 0 and max_par_order = 0 the block starts at 0, with
    // layout `start + slot * n_anticipated + plant`. Here n_anticipated = 1
    // and plant = 0, so slot 0 lives at `start + 0` and slot 1 at `start + 1`.
    let indexer = setup.stage_indexer();
    let ant_state_start = indexer.anticipated_state.start;
    let slot0_idx = ant_state_start; // slot 0, plant 0
    let slot1_idx = ant_state_start + 1; // slot 1, plant 0
    assert_eq!(
        indexer.n_anticipated, 1,
        "fixture must have exactly one anticipated thermal",
    );
    assert_eq!(indexer.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
    assert_eq!(
        ant_state_start, 0,
        "with n_hydros=0 and max_par_order=0, anticipated_state.start must \
         be 0; got {ant_state_start}",
    );

    // ── AC-3 / AC-4: select the most-recent cut (largest forward_pass_index).
    // `active_cuts(stage)` yields `(slot, intercept, &[coeffs])` where `slot`
    // encodes `warm_start_count + iteration * forward_passes + forward_pass_index`
    // (per CutPool::slot_index). The analytical match is at the FIRST cut
    // (slot=1, iteration 1): once iteration 1's cut is baked into stage 0's
    // template, the iteration-2 forward trial point shifts to a regime where
    // stage 2's subproblem is insensitive to the propagated state (the FCF
    // tangent is exact at the visited point), so iterations 2-5 add zero-
    // subgradient cuts with intercept c_ant*D_1/K = 0.5. The closed-form
    // derivation in the module docstring applies to the iteration-1 cut.
    // Select that cut explicitly rather than taking the most-recent one.
    let analytical = setup
        .fcf
        .active_cuts(0)
        .find(|(slot, _, _)| *slot == 1)
        .expect("AC-3: cut at slot=1 (iteration-1) must be present in stage 0 pool");
    let (slot, _intercept, coefficients) = analytical;

    assert_eq!(
        coefficients.len(),
        indexer.anticipated_state.end,
        "coefficient slice length must equal n_state (= anticipated_state.end \
         in this no-hydro fixture); got len={}, expected={}",
        coefficients.len(),
        indexer.anticipated_state.end,
    );

    // ── AC-3: coefficient at slot 1 (propagated stage-0 decision) ──────
    let actual_coeff_slot1 = coefficients[slot1_idx];
    assert!(
        (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
        "AC-3 / AC-5: stage 0 cut coefficient at anticipated_state slot 1 \
         (state-vector index {slot1_idx}) does not match analytical value: \
         actual = {actual_coeff_slot1}, expected = {EXPECTED_COEFF_SLOT1} \
         (= -c_reg/K = -{C_REG}/{COST_SCALE_FACTOR}). \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
        n_state = coefficients.len(),
    );

    // ── AC-4: coefficient at slot 0 (irrelevant; shifted out before fishing) ──
    let actual_coeff_slot0 = coefficients[slot0_idx];
    assert!(
        (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
        "AC-4 / AC-5: stage 0 cut coefficient at anticipated_state slot 0 \
         (state-vector index {slot0_idx}) does not match analytical value: \
         actual = {actual_coeff_slot0}, expected = {EXPECTED_COEFF_SLOT0}. \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
        n_state = coefficients.len(),
    );
}
