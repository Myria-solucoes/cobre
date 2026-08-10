//! Cost semantics of the post-horizon commitment-hold decision column
//! (ticket-009): the decision column books the discounted, delivery-anchored
//! fuel cost (`cost_per_mwh[thermal, dest] * post_study_hours[dest] *
//! post_study_cumulative_discount[dest]`, unscaled) and is bounded by the
//! intersection of the commitment interval and the post-study capability.
//! The terminal boundary FCF (ticket-006) prices the carried state
//! fuel-exclusively, so this booking does not double-count.
//!
//! The fixture is deliberately minimal: zero hydros, one bus, one anticipated
//! thermal declared `LeadTime` over two study stages. Its OWN in-study
//! self-delivery (stage 1, decided at stage 0) keeps `lead_stages >= 1` alive
//! so the in-study ring sizes non-degenerately; the ONE
//! `future_anticipated_deliveries` window (decider stage 1) resolves into the
//! ONE declared `post_study_stages` stage (`dest = 0`), so the decision
//! column's bound/cost is isolated from every other column family.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, FutureAnticipatedDelivery,
    HydroBlockBounds, HydroStageBounds, InitialConditions, LineBlockBounds, PostStudyStage,
    PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds, System,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_solver::{ActiveSolver, SolverInterface};

mod common;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// `LeadTime` delta (hours). Chosen so BOTH resolve in-study: the thermal's
/// own stage-1 self-delivery decides at stage 0 (`744 + 1450 -` overlap keeps
/// `decider(m=1) == 0`, a genuine one-stage-ahead decision — `lead_stages ==
/// 1`, avoiding the degenerate `k_max == 0` ring), and the far post-horizon
/// window (day ~121) decides at stage 1 (`decider(window) == 1`).
const DELTA_HOURS: f64 = 1450.0;
/// The decider stage for both [`future_delivery`] and the decision-column
/// lookups below — the LAST study stage (`DeciderKind::TerminalFan`).
const DECIDER_STAGE: usize = 1;
/// Fuel cost `$/MWh` at the resolved post-study cell.
const COST_PER_MWH: f64 = 37.5;
/// Post-study stage duration \[h\] (`H`); the study carries a `0.0` discount
/// rate, so the post-study cumulative discount factor (`D`) is exactly `1.0`
/// and the analytical objective is `COST_PER_MWH * POST_STUDY_HOURS`.
const POST_STUDY_HOURS: f64 = 720.0;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Two stages matching their real calendar span exactly (required for
/// `StageCalendar` coverage): stage 0 is January (744h), stage 1 is February
/// (696h in a common year, but 2024 is a leap year — 30 days declared here to
/// keep `duration_hours` an exact calendar match).
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    let stage0_end = start + chrono::TimeDelta::days(31);
    let stage1_end = stage0_end + chrono::TimeDelta::days(30);
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
                end_date: stage1_end,
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

/// The one declared post-horizon window: `[2024-04-01, 2024-05-01)`,
/// resolving (via `DELTA_HOURS`) to decider stage [`DECIDER_STAGE`] and (via
/// the post-study calendar below) to post-study destination stage 0.
fn future_delivery(min_mw: f64, max_mw: f64) -> FutureAnticipatedDelivery {
    FutureAnticipatedDelivery {
        thermal_id: THERMAL_ID,
        delivery_start: NaiveDate::from_ymd_opt(2024, 4, 1).expect("valid date"),
        delivery_end: NaiveDate::from_ymd_opt(2024, 5, 1).expect("valid date"),
        min_mw,
        max_mw,
    }
}

/// The one declared post-study stage: `[2024-04-01, 2024-05-01)` — 30 days,
/// `POST_STUDY_HOURS` (720h) — the EXACT span [`future_delivery`] targets, so
/// `StageCalendar::resolve_window` covers it at destination index 0.
fn post_study_stages(min_k: f64, max_k: f64) -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 4, 1).expect("valid date"),
            duration_hours: POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: COST_PER_MWH,
            min_mw: min_k,
            max_mw: max_k,
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

/// `with_commitment` toggles the declared `future_anticipated_deliveries`
/// window and its `post_study_stages` destination; `(min_c, max_c, min_k,
/// max_k)` are read only when `with_commitment` is `true`.
fn build_system(with_commitment: bool, min_c: f64, max_c: f64, min_k: f64, max_k: f64) -> System {
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
        future_anticipated_deliveries: if with_commitment {
            vec![future_delivery(min_c, max_c)]
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
    if with_commitment {
        builder = builder.post_study_stages(Some(post_study_stages(min_k, max_k)));
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
            // Unscaled: the analytical assertions compare `template.objective`
            // directly against `C * H * D` with no COST_SCALE_FACTOR back-out.
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

/// The decider stage's post-horizon decision-column range, resolved through
/// the same `StageGeometry::commitment_decision` the LP builder itself
/// derives — never a hand-computed offset that could drift from the real
/// layout.
fn decision_col(setup: &cobre_sddp::StudySetup, decider_stage: usize) -> usize {
    let range = setup.stage_ctx().geometry_per_stage[decider_stage]
        .commitment_decision
        .clone();
    assert_eq!(
        range.len(),
        1,
        "fixture declares exactly one post-horizon window deciding at this stage"
    );
    range.start
}

// -- Analytical costing + intersection bound (LP structural, no solve) --

mod analytical_costing_and_bounds {
    use super::*;

    #[test]
    fn decision_column_objective_equals_cost_times_hours_times_discount() {
        let system = build_system(true, 20.0, 80.0, 10.0, 50.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        let expected = COST_PER_MWH * POST_STUDY_HOURS * 1.0;
        assert!(
            (template.objective[col] - expected).abs() < 1e-9,
            "unscaled objective must equal C*H*D: expected {expected}, got {}",
            template.objective[col]
        );
    }

    #[test]
    fn decision_column_bound_is_the_intersection_not_either_interval_alone() {
        // Commitment [20, 80], capability [10, 50] -> intersection [20, 50]:
        // strictly narrower than either declared interval on its own.
        let system = build_system(true, 20.0, 80.0, 10.0, 50.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        assert_eq!(
            template.col_lower[col], 20.0,
            "col_lower must be max(min_c, min_k) == max(20, 10)"
        );
        assert_eq!(
            template.col_upper[col], 50.0,
            "col_upper must be min(max_c, max_k) == min(80, 50)"
        );
    }

    /// Swap which side of each interval binds: capability wider on the low
    /// end, commitment wider on the high end -- the intersection is neither
    /// input interval, ruling out "bind to the commitment" or "bind to the
    /// capability" as a wrong-but-plausible simplification.
    #[test]
    fn decision_column_bound_intersection_holds_regardless_of_which_side_binds() {
        let system = build_system(true, 30.0, 60.0, 5.0, 45.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        assert_eq!(template.col_lower[col], 30.0, "max(30, 5) == 30");
        assert_eq!(template.col_upper[col], 45.0, "min(60, 45) == 45");
    }
}

// -- Pinned commitment: min_c == max_c, solved end-to-end --

mod pinned_decision {
    use super::*;

    #[test]
    fn pinned_commitment_within_capability_fixes_the_decision_and_its_contribution() {
        let pinned = 30.0;
        let system = build_system(true, pinned, pinned, 10.0, 50.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        assert_eq!(template.col_lower[col], pinned);
        assert_eq!(template.col_upper[col], pinned);

        let mut solver = ActiveSolver::new().expect("ActiveSolver::new: must succeed");
        solver.load_model(template);
        let view = solver.solve(None).expect("stage LP must solve");

        assert!(
            (view.primal[col] - pinned).abs() < 1e-6,
            "solved primal must equal the pinned commitment: expected {pinned}, got {}",
            view.primal[col]
        );

        let expected_contribution = pinned * COST_PER_MWH * POST_STUDY_HOURS;
        let actual_contribution = view.primal[col] * template.objective[col];
        assert!(
            (actual_contribution - expected_contribution).abs() < 1e-6,
            "objective contribution must equal min_c*C*H*D: expected \
             {expected_contribution}, got {actual_contribution}"
        );
    }
}

// -- Inert without a post-horizon commitment (byte-identity regression guard) --

mod byte_identity_without_commitment {
    use super::*;

    /// Dropping `future_anticipated_deliveries`/`post_study_stages` resolves
    /// zero commitment windows and widens NEITHER `commit_out` nor
    /// `commit_in` by the post-horizon lane's width — the merged
    /// commitment-hold region collapses to exactly the in-study ring's own
    /// width (`n_anticipated * k_max`), matching the pre-ticket layout with
    /// no post-horizon lane at all. This is the structural half of the
    /// inertness claim; `fill_commitment_decision_columns`'s own byte-level
    /// no-op (an empty `commitment_decision_windows` never touches `bufs`) is
    /// pinned directly by the `crates/cobre-sddp/src/lp/builder/columns.rs`
    /// unit test, not re-derived here via a fragile cross-build column-index
    /// comparison (the ONE added lane shifts every later column index by 2,
    /// so raw positional diffing across the two builds would spuriously fail).
    #[test]
    fn no_future_anticipated_deliveries_resolves_zero_commitment_windows() {
        let with_setup = build_setup_in_code(build_system(true, 20.0, 80.0, 10.0, 50.0), &config());
        let without_setup = build_setup_in_code(build_system(false, 0.0, 0.0, 0.0, 0.0), &config());

        assert_eq!(
            without_setup.stage_state().n_commitment,
            0,
            "a study with no future_anticipated_deliveries must resolve zero commitment windows"
        );
        assert_eq!(
            with_setup.stage_state().n_commitment,
            1,
            "the sibling fixture must declare exactly one commitment window"
        );

        let n_ant_state = with_setup.stage_state().n_anticipated * with_setup.stage_state().k_max;
        assert_eq!(
            without_setup.stage_state().commit_out.len(),
            n_ant_state,
            "without a declared window, commit_out must be exactly the in-study ring's width"
        );
        assert_eq!(
            with_setup.stage_state().commit_out.len(),
            n_ant_state + 1,
            "with one declared window, commit_out must be the ring's width plus one lane"
        );

        // The ONE added lane is the sole structural delta, stage-invariant and
        // symmetric across commit_out and commit_in: every stage's num_cols
        // grows by exactly 2 (one outgoing lane column, one incoming) — plus
        // one more, exactly at DECIDER_STAGE, for the sparse decision column
        // itself.
        for stage in 0..2 {
            let without_cols = without_setup.stage_ctx().templates[stage].num_cols;
            let with_cols = with_setup.stage_ctx().templates[stage].num_cols;
            let expected_delta = 2 + usize::from(stage == DECIDER_STAGE);
            assert_eq!(
                with_cols,
                without_cols + expected_delta,
                "stage {stage}: num_cols must grow by exactly {expected_delta}"
            );
        }

        // The decision column itself is sparse: it exists ONLY at the
        // deciding stage, in both builds (empty at stage 0, populated only at
        // DECIDER_STAGE in the "with" build; never populated at all "without").
        for stage in 0..2 {
            let without_range = without_setup.stage_ctx().geometry_per_stage[stage]
                .commitment_decision
                .clone();
            assert_eq!(
                without_range.len(),
                0,
                "stage {stage}: no window is ever declared, so no decision column ever exists"
            );
            let with_range = with_setup.stage_ctx().geometry_per_stage[stage]
                .commitment_decision
                .clone();
            let expected_with_len = usize::from(stage == DECIDER_STAGE);
            assert_eq!(
                with_range.len(),
                expected_with_len,
                "stage {stage}: the decision column exists only at DECIDER_STAGE"
            );
        }
    }
}
