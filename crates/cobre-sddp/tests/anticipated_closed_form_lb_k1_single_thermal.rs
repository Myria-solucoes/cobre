//! Closed-form lower-bound canary for anticipated thermals.
//!
//! Two-stage, K=1, no hydro, no stochastic noise, single deterministic opening
//! per stage. The lower bound returned by `train` is hand-derivable in five
//! minutes from the LP coefficients written by `lp_builder::matrix`. Any
//! deviation from `EXPECTED_LB` flags a regression in the value-correctness
//! class (cost-shift, sign flip, missing scaling, broken fishing, etc.).
//!
//! ## Why this canary exists
//!
//! Larger anticipated-thermals integration tests (`anticipated_5stage_k2_smoke`,
//! `anticipated_two_plants_smoke`) use structural assertions only — convergence,
//! basis-cache population, ring-buffer invariants. Those tests do not pin
//! an `EXPECTED_LB` constant because there is no closed form for the larger
//! fixtures; any constant pinned there would only certify the converged value
//! is _stable_, not that it is _correct_. This canary supplies the value-
//! correctness signal that the structural tests cannot.
//!
//! ## Fixture
//!
//! - `n_stages = 2`, single 1-hour block per stage.
//! - 1 bus (deficit cost `1000 $/MWh`, excess cost `0 $/MWh`).
//! - 1 anticipated thermal `T_ant` (id=2): `cost = c_a = 10 $/MWh`,
//!   `max_gen = M = 100 MW`, `lead_stages = K = 1`.
//! - 1 backup thermal `T_b` (id=3): `cost = c_b = 100 $/MWh`,
//!   `max_gen = B = 200 MW`. Strict ordering `c_a < c_b` is what makes the
//!   anticipated commitment cheaper at delivery than backup.
//! - Load `D = 50 MW` at both stages.
//! - `past_anticipated_commitments = [(thermal_id=2, values_mw=[0.0])]`.
//!   Per F3-002 the past must be zero so that any non-zero anticipated
//!   delivery observed at stage 1 is attributable to the stage-0 decision.
//! - 1 deterministic opening per stage.
//! - Default `PolicyGraph::annual_discount_rate = 0.0`, so every
//!   `discount_factors[t] = 1.0` and `cumulative_discount_factors[t] = 1.0`.
//!
//! ## Closed-form derivation
//!
//! ### Legacy behaviour (before always-active fishing)
//!
//! Under the pre-flip predicate `is_anticipated_fishing_active` returned TRUE
//! only when `stage_idx >= K_i` (i.e. `stage_idx >= 1` for K=1). Stage 0 had
//! no fishing row, the anticipated thermal dispatched freely at cost `c_a`, and
//! the closed form was `T* = 2 · c_a · D = 1000.0`. That constant is now wrong.
//!
//! ### Always-active fishing (current behaviour)
//!
//! `StageIndexer::is_anticipated_fishing_active` at `indexer.rs:1555` now
//! returns TRUE at every stage, including stage 0. The fishing row
//! `g_a_t − x_state_t = 0` is therefore emitted at stage 0 as well, and
//! `zero_anticipated_delivery_thermal_cost` zeros the per-block cost of the
//! anticipated column at stage 0 (same always-active path). See the K=1
//! sign-chain table in
//! `artifacts/layout-decision.md`
//! §2 for the cut-coefficient sign convention that applies here.
//!
//! Variables (all in MW; subscript denotes stage):
//! - `g_a_t` — per-block anticipated thermal generation at stage `t`.
//! - `g_b_t` — per-block backup thermal generation at stage `t`.
//! - `d_ant_0` — anticipated decision placed at stage 0 (delivery at stage 1).
//! - `θ_0` — stage-0 future-cost approximation (`≥ 0`).
//! - `x_state_t` — anticipated-state slot 0 at stage `t` (free variable).
//!
//! Stage 0 (always-active fishing; decision predicate `t + K_i < n_stages` is
//! `0 + 1 < 2` — TRUE, so `d_ant_0` is active; fishing predicate now TRUE at
//! every stage including 0; per-block anticipated cost zeroed by
//! `zero_anticipated_delivery_thermal_cost`):
//!
//! ```text
//!   min  0 · g_a_0 + c_b · g_b_0 + c_a · d_ant_0 + θ_0
//!   s.t. g_a_0 + g_b_0 + deficit_0 − excess_0 = D       (load balance)
//!        g_a_0 − x_state_0 = 0                          (fishing row, always-active)
//!        x_state_0 = past[0] = 0                        (state-fixing, slot 0; pure identity under Alt-A)
//!        state_out_0 − d_ant_0 = 0                      (state-out definition row; couples decision to next-stage delivery)
//!        θ_0 ≥ 0                                        (no cuts initially)
//!        g_a_0 ∈ [0, M], g_b_0 ∈ [0, B], d_ant_0 ∈ [0, M]
//!        deficit_0 ≥ 0, excess_0 ≥ 0
//! ```
//!
//! Under the Alternative-A layout, the slot-0 state-fixing row is pure
//! identity (it pins `x_state_0` to `past[0] = 0` only; no `d_ant_0` coupling
//! on this row). The decision-vs-state coupling moves to the `state_out`
//! definition row, which lets `d_ant_0` be optimised freely. Fishing then
//! forces `g_a_0 = x_state_0 = 0`, so the load must be covered entirely by
//! `g_b_0 = D` at cost `c_b · D = 5000`. `d_ant_0` is the new commitment;
//! its objective coefficient `c_a` drives the trade-off between paying
//! `c_a · d_ant_0` now and paying `c_b · max(0, D − d_ant_0)` at stage 1.
//!
//! The decision objective coefficient is set by
//! `fill_anticipated_decision_objective` to
//! `c_a · total_hours_per_stage[delivery=1] · cumulative_discount_factors[1] =
//! c_a · 1 · 1 = c_a`.
//!
//! Stage 1 (delivery; fishing `is_anticipated_fishing_active` — TRUE; decision
//! `1 + 1 < 2` — FALSE, so `d_ant_1 ∈ [0,0]` and per-block anticipated cost
//! is zeroed by `zero_anticipated_delivery_thermal_cost`):
//!
//! ```text
//!   min  c_b · g_b_1 + 0 · g_a_1
//!   s.t. g_a_1 + g_b_1 + deficit_1 − excess_1 = D       (load balance)
//!        g_a_1 − x_state_1 = 0                          (fishing row)
//!        x_state_1 + 0 = d_ant_0                        (state-fixing; incoming = d_ant_0)
//!        g_a_1 ∈ [0, M], g_b_1 ∈ [0, B]
//!        deficit_1 ≥ 0, excess_1 ≥ 0
//! ```
//!
//! Substituting `x_state_1 = d_ant_0` and `g_a_1 = x_state_1 = d_ant_0`:
//!
//! - If `d_ant_0 ≤ D`: pick `g_b_1 = D − d_ant_0`, `excess_1 = deficit_1 = 0`.
//!   Stage-1 cost = `c_b · (D − d_ant_0)`.
//! - If `d_ant_0 > D` (≤ M): `g_b_1 = 0`, `excess_1 = d_ant_0 − D`,
//!   `deficit_1 = 0`. Stage-1 cost = `0` (excess cost = 0).
//!
//! The cost-to-go function is therefore
//! `V_1(d_ant_0) = c_b · max(0, D − d_ant_0)`.
//!
//! At convergence `θ_0 = V_1(d_ant_0)`, so the total objective is
//!
//! ```text
//!   T(d_ant_0) = c_b · D + c_a · d_ant_0 + c_b · max(0, D − d_ant_0)
//!              (stage-0: g_b_0=D at cost c_b·D; g_a_0=0 at cost 0·0=0)
//! ```
//!
//! The variable part over `d_ant_0 ∈ [0, D]` is
//!
//! ```text
//!   c_a · d_ant_0 + c_b · D − c_b · d_ant_0
//!     = c_b · D + (c_a − c_b) · d_ant_0
//! ```
//!
//! Since `c_a − c_b = 10 − 100 = −90 < 0`, this is minimised at `d_ant_0 = D`.
//! For `d_ant_0 > D` the term becomes `c_a · d_ant_0`, strictly increasing, so
//! the kink at `d_ant_0 = D` is the unique global minimum.
//!
//! ## Numerical evaluation
//!
//! With `c_a = 10`, `c_b = 100`, `D = 50`, `M = 100`:
//!
//! ```text
//!   T*  = c_b · D                 (stage-0 backup dispatch, g_b_0 = D; g_a_0 = 0)
//!       + c_a · D                 (stage-0 decision, d_ant_0 = D)
//!       + 0                       (stage-1 cost, c_b · (D − D) = 0)
//!       = (c_a + c_b) · D
//!       = (10 + 100) · 50
//!       = 5500.0  USD
//! ```
//!
//! Legacy value under the pre-flip predicate: `2 · c_a · D = 1000.0`. The
//! always-active fishing predicate zeroes `g_a_0`'s per-block cost but forces
//! backup to cover stage-0 load, raising the lower bound by `(c_b − c_a) · D`.
//!
//! Sanity checks (independent traversal of the piecewise total under
//! always-active fishing — stage-0 backup covers D; decision cost = c_a·d_ant_0):
//!
//! - `d_ant_0 = 0`:   `T = c_b·D + 0 + c_b·D       = 5000 + 0    + 5000 = 10000`
//! - `d_ant_0 = D`:   `T = c_b·D + c_a·D + 0       = 5000 + 500  + 0    = 5500` ← optimum
//! - `d_ant_0 = M`:   `T = c_b·D + c_a·M + 0       = 5000 + 1000 + 0    = 6000`
//!
//! The optimum is unambiguous — the kink at `d_ant_0 = D = 50` is the unique
//! global minimum.
//!
//! ## Bit-for-bit determinism
//!
//! The fixture has no stochastic noise (one deterministic opening per stage,
//! `std_mw = 0`, no hydro / no PAR(p) model). HiGHS is run single-threaded
//! with `tree_seed = 42`. The LP-builder's column / row scaling is
//! data-dependent but deterministic. We therefore assert bit-for-bit equality
//! of `final_lb` against `EXPECTED_LB`. If the scaler introduces sub-ULP
//! arithmetic noise on a future libhighs upgrade, switch to the relative
//! tolerance assertion (`rel_diff < 1e-12`).

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
// Closed-form fixture parameters (single source of truth).
// ---------------------------------------------------------------------------

const N_STAGES: usize = 2;
const K_MAX: usize = 1;
const BLOCK_HOURS: f64 = 1.0;

/// Constant load at every stage (MW).
const D_LOAD: f64 = 50.0;

/// Anticipated thermal capacity (MW). Strictly above `D_LOAD` so the optimum
/// lies in the interior of `[0, M]` at the kink `d_ant_0 = D`.
const M_ANT: f64 = 100.0;

/// Backup thermal capacity (MW). Generous so backup never binds capacity.
const B_BACK: f64 = 200.0;

/// Anticipated marginal cost ($/MWh). Strict ordering `C_A < C_B` is what
/// makes the anticipated commitment cheaper-than-backup at the delivery
/// stage; the LP would otherwise be indifferent.
const C_A: f64 = 10.0;

/// Backup marginal cost ($/MWh).
const C_B: f64 = 100.0;

/// Deficit cost ($/MWh). Set well above `C_B` so deficit is never preferred
/// over backup dispatch.
const C_DEFICIT: f64 = 1000.0;

/// Closed-form lower bound under always-active fishing:
/// `T* = (C_A + C_B) · D_LOAD`. See module docs for the full derivation.
/// Numerical value: `(10 + 100) · 50 = 5500.0` USD.
///
/// Legacy value (pre-flip predicate `K_i ≤ stage_idx`): `2 · C_A · D_LOAD = 1000.0`.
/// The always-active fishing predicate at `indexer.rs:1555`
/// (`StageIndexer::is_anticipated_fishing_active`) emits a fishing row at stage 0,
/// forcing `g_a_0 = 0` and requiring backup to cover stage-0 load. See the K=1
/// sign-chain table in
/// `artifacts/layout-decision.md`
/// §2 for the cut-coefficient convention.
///
/// Held to bit-for-bit equality. Sub-ULP noise from a future libhighs upgrade
/// should be rare for this trivial fixture; if it occurs, demote to a 1e-12
/// relative-tolerance check.
const EXPECTED_LB: f64 = (C_A + C_B) * D_LOAD; // = 5500.0

const ANTICIPATED_ID: EntityId = EntityId(2);
const BACKUP_ID: EntityId = EntityId(3);
const BUS_ID: EntityId = EntityId(1);

// SystemBuilder::build() sorts thermals by EntityId ascending, so the
// global thermal indices end up: 0 → id=2 (anticipated), 1 → id=3 (backup).
const THERMAL_IDX_ANT: usize = 0;
const THERMAL_IDX_BACKUP: usize = 1;

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator for testing.
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

fn build_system() -> cobre_core::System {
    use chrono::NaiveDate;

    let bus = Bus {
        id: BUS_ID,
        name: "B1".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: C_DEFICIT,
        }],
        excess_cost: 0.0,
    };

    // Anticipated thermal: K=1, sorts first by EntityId.
    let thermal_ant = Thermal {
        id: ANTICIPATED_ID,
        name: "T_ant".to_string(),
        bus_id: BUS_ID,
        min_generation_mw: 0.0,
        max_generation_mw: M_ANT,
        cost_per_mwh: C_A,
        anticipated_config: Some(AnticipatedConfig { lead_stages: 1 }),
        entry_stage_id: None,
        exit_stage_id: None,
    };

    // Backup standard thermal.
    let thermal_backup = Thermal {
        id: BACKUP_ID,
        name: "T_backup".to_string(),
        bus_id: BUS_ID,
        min_generation_mw: 0.0,
        max_generation_mw: B_BACK,
        cost_per_mwh: C_B,
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
            mean_mw: D_LOAD,
            std_mw: 0.0,
        })
        .collect();

    // n_hydros = 0; anticipated + backup thermals only. k_max = 1.
    // Per-thermal per-stage overrides across the full thermal axis (length =
    // n_stages + k_max = 3) so that `fill_anticipated_decision_objective`
    // reads a well-defined cost at delivery (stage 1).
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

    let thermal_axis = N_STAGES + K_MAX;
    for s in 0..thermal_axis {
        *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: M_ANT,
            cost_per_mwh: C_A,
        };
        *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: B_BACK,
            cost_per_mwh: C_B,
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

    // Past commitment is zero per F3-002 (the strict-zero precondition for
    // the K=1 ring buffer). Any non-zero delivery observed at stage 1 must
    // come from the stage-0 decision.
    let initial_conditions = InitialConditions {
        storage: vec![],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: ANTICIPATED_ID,
            values_mw: vec![0.0],
        }],
        recent_observations: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant, thermal_backup])
        .stages(stages)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system: valid")
}

// ---------------------------------------------------------------------------
// Config and setup builders
// ---------------------------------------------------------------------------

/// Build a deterministic config. Iteration limit = 4 is sufficient to drive
/// the LB to the closed-form value; the first iteration generates the cut at
/// `d_ant_0 = 0`, the second probes the saturated region (`d_ant_0 > D`),
/// and the third confirms the lower envelope at `d_ant_0 = D`. We add a
/// fourth iteration to give the LP one more chance to settle in case of
/// degenerate-vertex selection.
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
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
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

/// Closed-form canary: the LP-derived lower bound for the 2-stage K=1 fixture
/// must equal the hand-derived value `(C_A + C_B) · D_LOAD = 5500.0` under
/// always-active fishing (`StageIndexer::is_anticipated_fishing_active`,
/// `indexer.rs:1555`).
///
/// Bit-for-bit equality is asserted via `f64::to_bits`. The fixture is fully
/// deterministic (single opening, no noise, single-thread HiGHS), so any
/// drift indicates a value-correctness regression.
#[test]
fn anticipated_closed_form_lb_k1_single_thermal() {
    let system = build_system();
    let config = build_config();
    let mut setup = build_setup(system, &config);
    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must not return Err");

    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );

    let result = &outcome.result;
    assert_eq!(result.iterations, 4, "iterations mismatch");

    // Closed-form LB derived in the module-level docstring. Bit-for-bit
    // equality is asserted because the fixture has no source of stochastic
    // noise; any drift indicates a value-correctness regression.
    let actual = result.final_lb;
    assert!(actual.is_finite(), "final_lb must be finite; got {actual}");
    assert_eq!(
        actual.to_bits(),
        EXPECTED_LB.to_bits(),
        "closed-form LB mismatch: actual = {actual}, expected = {EXPECTED_LB}. \
         Under always-active fishing the answer is `(C_A + C_B) · D_LOAD = 5500.0` \
         (stage-0 backup covers load; anticipated column cost zeroed by \
         zero_anticipated_delivery_thermal_cost; see indexer.rs:1555). \
         If a libhighs upgrade introduces sub-ULP arithmetic drift, switch to \
         a 1e-12 relative tolerance check.",
    );

    // Final gap should also have closed to zero for this deterministic
    // fixture: UB and LB coincide on the unique optimum.
    let gap = result.final_gap;
    assert!(
        gap.is_finite() && gap.abs() < 1e-9,
        "final_gap should be approximately zero for a fully deterministic \
         fixture; got {gap}",
    );
}
