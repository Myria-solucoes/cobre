//! `d_t`-saturation regression tests for an anticipated thermal across lead_stages
//! K = 2, 3. Each test trains a small in-code study, runs a one-scenario
//! simulation, and asserts that the anticipated-decision variable `d_t` commits to
//! the load level at every active stage (`t + K < n_stages`) and is absent at the
//! strict-boundary inactive stages. The K=3 test additionally checks the
//! ring-buffer shift `committed_at(t) ≈ decision_at(t − K)`. Each K's
//! economic-reasoning derivation lives on its test function.

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
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use cobre_core::entities::{
    bus::DeficitSegment,
    hydro::{HydroGenerationModel, HydroPenalties},
    thermal::AnticipatedConfig,
};
use cobre_core::scenario::{InflowModel, LoadModel};
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
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};

mod common;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};
use common::run_simulation;

// ---------------------------------------------------------------------------
// Per-K fixture table
// ---------------------------------------------------------------------------

/// Per-K parameters for a `d_t`-saturation fixture. Each `#[test]` builds an
/// independent `System` from one entry, so an entity id may name different roles
/// across K (id 3 is the K=2 hydro but the K=3 anticipated thermal) without
/// colliding — the ids are local to each built system.
struct SaturationFixture {
    n_stages: usize,
    /// Anticipated `lead_stages` K — also the ring-buffer depth.
    k: usize,
    bus_id: EntityId,
    anticipated_id: EntityId,
    anticipated_name: &'static str,
    backup_id: EntityId,
    backup_name: &'static str,
    hydro_id: EntityId,
    /// Anticipated ring-buffer seeds, MW (length `k`); zero isolates in-horizon
    /// behaviour from any seeding artefact.
    seeds_mw: &'static [f64],
    iterations: usize,
}

const FIXTURE_K2: SaturationFixture = SaturationFixture {
    n_stages: 6,
    k: 2,
    bus_id: EntityId(1),
    anticipated_id: EntityId(2),
    anticipated_name: "T_ant",
    backup_id: EntityId(4),
    backup_name: "T_backup",
    hydro_id: EntityId(3),
    seeds_mw: &[0.0, 0.0],
    iterations: 10,
};

const FIXTURE_K3: SaturationFixture = SaturationFixture {
    n_stages: 8,
    k: 3,
    bus_id: EntityId(1),
    anticipated_id: EntityId(3),
    anticipated_name: "T_ant_k3",
    backup_id: EntityId(4),
    backup_name: "T_backup_k3",
    hydro_id: EntityId(5),
    seeds_mw: &[0.0, 0.0, 0.0],
    iterations: 15,
};

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// Build the thermal `System` for one `d_t`-saturation fixture: one anticipated
/// thermal (cheap, $10/MWh, max 200 MW) at `fixture.anticipated_id`, one backup
/// ($5000/MWh, max 500 MW) at `fixture.backup_id`, and a trivial hydro (1 hm³,
/// zero inflow, 1 MW max_gen) at `fixture.hydro_id` that keeps the model in the
/// thermal regime without adding an interpretable hydro state. Load is 150 MW
/// across all stages; seeds (`fixture.seeds_mw`) are zero to isolate the
/// in-horizon behaviour from any seeding artefact.
fn build_system(fixture: &SaturationFixture) -> cobre_core::System {
    use chrono::NaiveDate;

    let k = fixture.k;
    let n_stages = fixture.n_stages;

    let bus = make_bus(
        fixture.bus_id,
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 5000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let anticipated_id = fixture.anticipated_id;
    let thermal_ant = make_thermal(
        anticipated_id,
        ThermalSpec {
            name: fixture.anticipated_name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            cost_per_mwh: 10.0,
            anticipated_config: Some(AnticipatedConfig {
                lead_stages: k as u32,
            }),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let thermal_backup = make_thermal(
        fixture.backup_id,
        ThermalSpec {
            name: fixture.backup_name.to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            cost_per_mwh: 5000.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        fixture.hydro_id,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: fixture.bus_id,
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
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
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
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
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: fixture.hydro_id,
            stage_id: i as i32,
            mean_m3s: 0.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: fixture.bus_id,
            stage_id: i as i32,
            mean_mw: 150.0,
            std_mw: 0.0,
        })
        .collect();

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
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

    // The per-thermal costs must be patched afterwards (ResolvedBounds::new takes
    // one default for ALL thermals) so the objective distinguishes the cheap
    // anticipated thermal from the expensive backup. The patch must extend over the
    // padding region `[n_stages, n_stages + k)` — the delivery-stage axis read by
    // `fill_anticipated_columns` — or the decision column's objective coefficient
    // stays zero and the regression is masked.
    let thermal_axis = n_stages + k;
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
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
    for s in 0..thermal_axis {
        bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // index 0 = anticipated (cheap)
        bounds.thermal_bounds_mut(0, s).max_generation_mw = 200.0;
        bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // index 1 = backup (expensive)
        bounds.thermal_bounds_mut(1, s).max_generation_mw = 500.0;
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
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
            hydro_id: fixture.hydro_id,
            value_hm3: 0.0,
        }],
        filling_storage: vec![],
        past_inflows: vec![],
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: anticipated_id,
            values_mw: fixture.seeds_mw.to_vec(),
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
        .expect("build_system: valid system")
}

// ---------------------------------------------------------------------------
// Config builder
// ---------------------------------------------------------------------------

/// Build a [`Config`] for `iterations`-iteration training and 1-scenario
/// deterministic simulation. More iterations let cuts sharpen so the cut
/// gradients at the anticipated-state columns drive `d_t` to load level at every
/// active stage; K=3's longer propagation chain needs more than K=2.
fn build_config(iterations: usize) -> Config {
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
                limit: iterations as u32,
            }]),
            stopping_mode: "any".to_string(),
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            scenario_source: None,
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig {
            enabled: true,
            num_scenarios: 1,
            io_channel_capacity: 8,
            ..IoSimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

// ---------------------------------------------------------------------------
// Train + simulate + drain helper
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Assert `anticipated_decision_mw` commits to load level (150 MW) for every
/// active decision stage (`t + K < n_stages`, i.e. `t in {0,1,2,3}`) and is
/// `None` for the boundary stages, in a K=2, 6-stage fixture.
///
/// ## Economic reasoning (pins the asserted optimum)
///
/// In a 6-stage K=2 study with a 500x cost asymmetry (anticipated thermal at
/// $10/MWh vs backup at $5000/MWh), load = 150 MW < max_gen = 200 MW and excess
/// generation costs $0. The LP optimum is therefore `d_t = load = 150 MW` at every
/// stage `t` where `t + K < n_stages` (`t in {0,1,2,3}`): over-committing to 200 MW
/// costs an extra 50 MW × $10/MWh × 744 h = $372k/stage with no offsetting benefit.
/// The 500x asymmetry only forces the anticipated thermal to dispatch at all
/// (reaching load level), not to saturate at max_gen.
///
/// A regression in the anticipated-state cut-coefficient mapping
/// (`state_to_lp_column`'s `Less` branch, for `k >= 2`) drops `d_t` to 0 at the
/// intermediate active stages, forcing backup at $5000/MWh.
#[test]
fn d_t_commits_to_load_for_every_active_stage_k2() {
    let k: usize = 2;
    let n_stages: usize = 6;
    // Active decision stages: t + K < n_stages  =>  t in {0, 1, 2, 3}.
    let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
    let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();

    let system = build_system(&FIXTURE_K2);
    let config = build_config(FIXTURE_K2.iterations);
    let mut setup = build_setup_in_code(system, &config);
    let scenario_results = run_simulation(&mut setup, FIXTURE_K2.iterations);

    assert_eq!(
        scenario_results.len(),
        1,
        "simulation must stream exactly one scenario result",
    );
    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        n_stages,
        "scenario must contain one stage record per study stage",
    );

    // The anticipated thermal has entity id=2 (see FIXTURE_K2).
    let anticipated_thermal_id: i32 = 2;
    let decision_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_decision_mw)
    };

    // ── Active stages: decision must exist and commit to load = 150 MW ──
    let load_mw = 150.0_f64;
    let tol = 1e-3_f64;
    for t in &active_stages {
        let d_t = decision_at(*t).unwrap_or_else(|| {
            panic!("anticipated_decision_mw must be Some at active stage t={t} (t + K < n_stages)")
        });
        assert!(
            (d_t - load_mw).abs() < tol,
            "d_t at stage {t} must saturate at load=150 MW: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Pre-fix behaviour: d_t ≈ 0 for t >= 1 due to cut-coefficient \
             corruption in state_to_lp_column (Less branch), forcing backup \
             at $5000/MWh.",
            delta = (d_t - load_mw).abs(),
        );
    }

    // ── Inactive stages: decision must be None (strict-boundary predicate) ──
    for t in &inactive_stages {
        assert!(
            decision_at(*t).is_none(),
            "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages; strict-boundary predicate excludes this stage)",
        );
    }
}

/// Assert that `anticipated_decision_mw` commits to load level (150 MW) for every
/// active decision stage in a K=3, 8-stage fixture, and that the ring-buffer shift
/// propagates each decision to its delivery stage:
/// - `d_t ≈ 150.0` for `t in {0, 1, 2, 3, 4}`.
/// - `anticipated_decision_mw` is `None` for `t in {5, 6, 7}`.
/// - `committed_at(t) ≈ decision_at(t - 3)` for `t in {3, 4, 5, 6, 7}`.
///
/// ## Bug being guarded against
///
/// In an 8-stage K=3 study with a 500x cost asymmetry (anticipated thermal at
/// $10/MWh vs backup at $5000/MWh), the LP should commit the anticipated
/// thermal to exactly `load = 150 MW` at every stage `t` where `t + K < n_stages`
/// (i.e. `t in {0, 1, 2, 3, 4}`). Over-committing to `max_gen = 200 MW` costs
/// an extra 50 MW × $10/MWh × 744 h = $372k/stage with zero offsetting benefit
/// (excess generation is free), so the LP optimum is `d_t = load = 150 MW`,
/// not `max_gen = 200 MW`.
///
/// The K=3 propagation chain is longer than K=2: cuts at stage `t` propagate
/// to predecessor's slot 1, which is `state_col[slot 2]`, whose value at K=3
/// is `incoming_slot_2 - d_{t-1}` if the decision-write coefficient is at
/// slot K-1 = 2. This variant exercises the multi-step propagation through
/// all three slot positions, confirming the fix is general and not K=2-specific.
///
/// The cut coefficients for the anticipated-state columns are corrupted for
/// `k >= 2` in `state_to_lp_column`'s `Less` branch, so the policy at those
/// stages receives no incentive to commit. At K=3 the corruption propagates
/// through all three slot positions, making it a stricter test than K=2.
#[test]
fn d_t_commits_to_load_for_every_active_stage_k3() {
    let k: usize = 3;
    let n_stages: usize = 8;
    let active_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k < n_stages).collect();
    let inactive_stages: Vec<usize> = (0..n_stages).filter(|&t| t + k >= n_stages).collect();
    let committed_stages: Vec<usize> = (0..n_stages).filter(|&t| t >= k).collect();

    let system = build_system(&FIXTURE_K3);
    let config = build_config(FIXTURE_K3.iterations);
    let mut setup = build_setup_in_code(system, &config);
    let scenario_results = run_simulation(&mut setup, FIXTURE_K3.iterations);

    assert_eq!(
        scenario_results.len(),
        1,
        "simulation must stream exactly one scenario result",
    );
    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        n_stages,
        "scenario must contain one stage record per study stage",
    );

    // The anticipated thermal has entity id=3 (see FIXTURE_K3).
    let anticipated_thermal_id: i32 = 3;
    let decision_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_decision_mw)
    };
    let committed_at = |t: usize| -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == anticipated_thermal_id)
            .and_then(|th| th.anticipated_committed_mw)
    };

    // ── Active stages: decision must exist and commit to load = 150 MW ──
    let load_mw = 150.0_f64;
    let tol = 1e-3_f64;
    for t in &active_stages {
        let d_t = decision_at(*t).unwrap_or_else(|| {
            panic!(
                "anticipated_decision_mw must be Some at active stage t={t} \
                 (t + K < n_stages, K=3)",
            )
        });
        assert!(
            (d_t - load_mw).abs() < tol,
            "d_t at stage {t} must saturate at load=150 MW: \
             got {d_t} (delta = {delta:.6} MW, tol = {tol} MW). \
             Pre-fix behaviour: d_t ≈ 0 for t >= 1 due to cut-coefficient \
             corruption in state_to_lp_column (Less branch). \
             At K=3 the corruption spans all three slot positions.",
            delta = (d_t - load_mw).abs(),
        );
    }

    // ── Inactive stages: decision must be None (strict-boundary predicate) ──
    for t in &inactive_stages {
        assert!(
            decision_at(*t).is_none(),
            "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages, K=3; strict-boundary predicate excludes this stage)",
        );
    }

    // ── Ring-buffer shift: committed_at(t) == decision_at(t - K) ──
    //
    // The shift at end-of-stage-(t-3) places d_{t-3} into slot K-1=2; after two
    // more shifts (end of stage t-2 and t-1) it reaches slot 0, where the fishing
    // constraint reads it at stage t. So committed_at(t) = d_{t-3}.
    for t in &committed_stages {
        let c_t = committed_at(*t).unwrap_or_else(|| {
            panic!(
                "anticipated_committed_mw must be Some at stage t={t} \
                 (t >= K=3; fishing constraint is active)",
            )
        });
        let d_prev = decision_at(*t - k).unwrap_or_else(|| {
            panic!(
                "anticipated_decision_mw must be Some at stage t-K={prev} \
                 (used to verify ring-buffer shift at delivery stage t={t})",
                prev = *t - k,
            )
        });
        assert!(
            (c_t - d_prev).abs() < tol,
            "ring-buffer shift invariant violated at t={t}: \
             committed_at({t}) = {c_t:.6} MW but decision_at({prev}) = {d_prev:.6} MW \
             (delta = {delta:.6} MW, tol = {tol} MW, K=3). \
             The ring buffer should propagate d_{{t-3}} through three slot \
             positions so it reaches slot 0 at stage t.",
            prev = *t - k,
            delta = (c_t - d_prev).abs(),
        );
    }
}
