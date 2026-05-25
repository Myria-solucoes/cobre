//! Analytical verification of backward-pass cut-coefficient extraction for an
//! anticipated thermal with `lead_stages = 1` in a 2-stage system.
//!
//! ## Closed-form derivation
//!
//! - One anticipated thermal (K=1, cost `c_ant`, max `max_gen_ant`) and one
//!   regular thermal (cost `c_reg`, max `max_gen_reg`) at a single bus.
//! - Load `D_0` at stage 0, `D_1` at stage 1; single one-hour block per stage.
//! - `max_par_order = 0`, so `n_state = K_max = 1` and the anticipated-state
//!   index inside the state vector equals `indexer.anticipated_state.start = 0`.
//!
//! The LP-builder divides every non-theta objective coefficient by
//! `COST_SCALE_FACTOR = K = 1_000` (see `crates/cobre-sddp/src/lp_builder/mod.rs`
//! and `template.rs`). Duals therefore live in scaled units, and the cut
//! storage at `backward.rs::accumulate_opening_outcome` preserves that scaling
//! end-to-end (forward.rs consumes the coefficients unrescaled).
//!
//! Stage-1 LP (delivery stage; the per-block anticipated-thermal generation
//! column is zeroed by `zero_anticipated_delivery_thermal_cost`; the stage-1
//! anticipated-decision column `d_ant` carries a scaled NPV cost
//! `c_reg * block_hours_total / K = c_reg / K` because the delivery stage for
//! that decision (`t + K_i = 2`) is in range):
//!
//! ```text
//!   min  (c_reg/K) gt_reg + (c_reg/K) d_ant + theta
//!   s.t. gt_reg + gt_ant = D_1                          (load balance)
//!        gt_ant − x_anticipated_state[slot=0] = 0       (fishing, K=1 <= 1)
//!        x_anticipated_state[slot=0] + d_ant = x_hat    (state-fixing row, dual = π)
//!        theta >= 0                                     (empty terminal cut pool)
//! ```
//!
//! Substituting `gt_ant = x_state` and `x_state = x_hat − d_ant`:
//! `Q_scaled(x_hat) = (c_reg/K)(D_1 − x_hat + 2 d_ant)`. The coefficient of
//! `d_ant` is positive, so the optimum sets `d_ant = 0` for the chosen box
//! bounds, giving `Q_scaled(x_hat) = (c_reg/K)(D_1 − x_hat)` and
//! `π = dQ_scaled/dx_hat = −c_reg/K`.
//!
//! Cut convention `coefficients = dual` (no sign flip — see
//! `crates/cobre-sddp/src/backward.rs` module-level documentation on cut
//! sign convention and anticipated-state cut gradient flow). The intercept is
//! `α = Q_scaled(x_hat) − π · x_hat = (c_reg/K) D_1`, independent of the trial
//! point.
//!
//! For the chosen numerics (`c_reg = 100`, `D_1 = 20`, `K = 1_000`) the expected
//! scaled-unit values are `coefficient = −0.1` and `intercept = 2.0`.

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

const N_STAGES: usize = 2;
const K_MAX: usize = 1;
const BLOCK_HOURS: f64 = 1.0;

const D_0: f64 = 10.0;
const D_1: f64 = 20.0;
const C_REG: f64 = 100.0;
const C_ANT: f64 = 50.0;
const MAX_GEN_REG: f64 = 50.0;
const MAX_GEN_ANT: f64 = 30.0;
const X0_SEED: f64 = 10.0;

// Cuts are stored in scaled cost units; the LP-builder divides every non-theta
// objective coefficient by `COST_SCALE_FACTOR` (see lp_builder/mod.rs and
// template.rs). Duals therefore live in scaled units too, and the cut storage
// at backward.rs preserves that scaling end-to-end (forward.rs consumes them
// unrescaled).
const COST_SCALE_FACTOR: f64 = 1_000_000.0;

// Closed-form expected values for stage-0 FCF cut at the anticipated_state
// index, in the LP's scaled cost units.
const EXPECTED_COEFFICIENT: f64 = -C_REG / COST_SCALE_FACTOR; // = -0.0001
const EXPECTED_INTERCEPT: f64 = C_REG * D_1 / COST_SCALE_FACTOR; // = 2.0
const TOL: f64 = 1e-6;

// Thermal column ordering inside the built `System`. `System::build()` sorts
// thermals by `EntityId::0` ascending, so `THERMAL_IDX_REG = 0` requires
// `T_reg.id < T_ant.id`. Keep the regular thermal at id 2 (sorts first) and the
// anticipated thermal at id 4 (sorts second).
const THERMAL_IDX_REG: usize = 0;
const THERMAL_IDX_ANT: usize = 1;
const ANTICIPATED_ID: EntityId = EntityId(4);

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator (copied from anticipated_forward_pass.rs
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

/// Build the 2-stage analytical system described at the top of the module.
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

    // thermal_idx 0 — regular thermal (marginal at stage 1).
    let thermal_reg = Thermal {
        id: EntityId(3),
        name: "T_reg".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_REG,
        cost_per_mwh: C_REG,
        anticipated_config: None,
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // thermal_idx 1 — anticipated thermal, K=1.
    let thermal_ant = Thermal {
        id: ANTICIPATED_ID,
        name: "T_ant".to_string(),
        bus_id: EntityId(1),
        min_generation_mw: 0.0,
        max_generation_mw: MAX_GEN_ANT,
        cost_per_mwh: C_ANT,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
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
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: if i == 0 { D_0 } else { D_1 },
            std_mw: 0.0,
        })
        .collect();

    // No hydros — anticipated/regular thermals only. n_anticipated=1, k_max=1.
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
    // (length = n_stages + k_max = 3). Index 2 (the K-padded delivery cell)
    // matters only for plants with K_i > 0; we seed it from the anticipated
    // delivery bounds so `fill_anticipated_decision_objective` reads a
    // well-defined cost at stage_idx + K_i = 0 + 1 = 1, which is in range.
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

    // Seed the anticipated ring buffer slot 0 with X0_SEED = 10.0.
    // With K=1 the ring has K_MAX = 1 slot; this is the matured commitment
    // delivered at stage 0 (fishing inactive at stage 0 because K=1 > 0).
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: ANTICIPATED_ID,
            values_mw: vec![X0_SEED],
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
fn two_stage_k1_anticipated_cut_coefficient_matches_analytical() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new");

    // Run one training iteration — generates exactly one cut at stage 0
    // for the single forward trajectory / opening.
    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error
    );

    // ── AC-2: exactly one cut at stage 0 FCF ──────────────────────────
    let pool0 = &setup.fcf.pools[0];
    let active_count = pool0.active_count();
    assert_eq!(
        active_count, 1,
        "AC-2: stage 0 FCF must contain exactly one active cut; got {active_count}",
    );

    // Locate the anticipated_state index inside the state vector.
    // For n_hydros = 0 and max_par_order = 0 this is `0` by construction.
    let indexer = setup.stage_indexer();
    let ant_state_idx = indexer.anticipated_state.start;
    assert_eq!(
        indexer.n_anticipated, 1,
        "fixture must have exactly one anticipated thermal",
    );
    assert_eq!(indexer.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
    assert_eq!(
        ant_state_idx, 0,
        "with n_hydros=0 and max_par_order=0, anticipated_state.start must be 0; got {ant_state_idx}",
    );

    // ── AC-3 / AC-4: cut coefficient and intercept match closed form ──
    let (slot, intercept, coefficients) = setup
        .fcf
        .active_cuts(0)
        .next()
        .expect("AC-3: exactly one active cut must be retrievable from stage 0 pool");
    assert_eq!(
        coefficients.len(),
        indexer.anticipated_state.end,
        "coefficient slice length must equal n_state",
    );

    let actual_coeff = coefficients[ant_state_idx];
    assert!(
        (actual_coeff - EXPECTED_COEFFICIENT).abs() < TOL,
        "AC-3 / AC-5: cut coefficient at anticipated_state index {ant_state_idx} \
         (slot={slot}, n_state={n_state}) does not match analytical value: \
         actual = {actual_coeff}, expected = {EXPECTED_COEFFICIENT} (= -c_reg/K = -{C_REG}/{COST_SCALE_FACTOR})",
        n_state = coefficients.len(),
    );

    assert!(
        (intercept - EXPECTED_INTERCEPT).abs() < TOL,
        "AC-4: cut intercept does not match analytical value: actual = {intercept}, \
         expected = {EXPECTED_INTERCEPT} (= c_reg * D_1 / K = {C_REG} * {D_1} / {COST_SCALE_FACTOR})",
    );
}
