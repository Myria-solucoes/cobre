//! Consolidated anticipated-scenarios integration tests for `cobre-sddp`.
//!
//! Each anticipated-scenarios domain group lives in its own inner `mod` so the
//! suite links the statically-bound solver once rather than once per file.
//! Per-`mod` scoping isolates each group's consts, helpers, and fixtures.

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

mod common;

mod anticipated_5stage_k2_smoke {
    //! Smoke test for K=2 anticipated thermal dispatch with 5 stages.
    //!
    //! ## Scope
    //!
    //! Uses structural assertions (training completes without error, iteration
    //! count matches the configured limit, lower bound is finite and
    //! non-decreasing across iterations, basis cache is populated at every
    //! stage, and anticipated-state slots at delivery stages hold finite,
    //! non-negative values). It deliberately does NOT pin an `EXPECTED_LB`
    //! constant: there is no closed-form derivation for this fixture, so a
    //! pinned value would only certify the converged value is _stable_, not
    //! that it is _correct_.
    //!
    //! The closed-form value-correctness canary is the
    //! `anticipated_closed_form_lb_k1_single_thermal` test in `anticipated_core.rs`,
    //! which uses a stripped 2-stage K=1 fixture whose lower bound is hand-derivable
    //! in five minutes.
    //! That canary defends the LP/cut math; this smoke test defends multi-stage
    //! state propagation, the K=2 ring-buffer shift, and basis-cache capture.

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, HydroStorage,
        InitialConditions, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::SolverStatsDelta;
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::anticipated_structural_assertions::{
        assert_anticipated_delivery_slots_populated, assert_basis_cache_fully_populated,
        assert_training_converged_structurally,
    };
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// The LP is always feasible: the backup thermal alone covers the load.
    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
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

        let n_stages = 5_usize;
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
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 2;
        let n_st = n_stages;

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

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
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
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: vec![100.0, 50.0],
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
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]),
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

    #[test]
    fn test_anticipated_5stage_k2_analytical_lb() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 8, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        let state = setup.stage_state();
        let anticipated_state_start = state.anticipated_state.start;
        let n_anticipated = state.n_anticipated;

        assert_training_converged_structurally(result, &[], 8);
        assert_basis_cache_fully_populated(result, 5);
        // K=2 strict predicate (t + K < n_stages): decisions at t in {0,1,2} deliver
        // at t+K in {2,3,4}.
        assert_anticipated_delivery_slots_populated(
            result,
            anticipated_state_start,
            n_anticipated,
            &[2, 3, 4],
        );
    }

    /// Warm-start regression: an anticipated-thermal study trained with warm-start
    /// must reconstruct every stored basis with zero rejections.
    ///
    /// `anticipated_state_out` lives in the stage-invariant state region, so the
    /// cut row that targets it shifts `z_inflow`/`storage_in`/`theta` and the whole
    /// control region downstream by `n_anticipated`. The risk this test pins: that
    /// the shift breaks `reconstruct_basis`'s slot-identity matching and starts
    /// producing bases `HiGHS` rejects. It cannot, because `reconstruct_basis`
    /// matches stored cut rows to current LP rows by `CutPool` slot identity — never
    /// by absolute column index — and copies the column block verbatim before
    /// resizing to the (now-wider) column count. `basis_consistency_failures` is the
    /// `SolverStatistics` counter incremented whenever the solver rejects an offered
    /// warm-start basis (`isBasisConsistent` returns false): every reconstructed
    /// basis must be accepted by the solver.
    #[test]
    fn test_anticipated_5stage_k2_warm_start_zero_basis_rejections() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 8, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(
            result.iterations, 8,
            "warm-start regression needs the full 8-iteration run to exercise \
         reconstruct_basis on iterations 2..8"
        );

        // Aggregate basis-rejection telemetry across every phase. On the baked
        // warm-start path `reconstruct_basis` runs once per warm-start solve; a
        // single rejection here would mean the relocated-column LP produced a basis
        // HiGHS could not accept.
        let total_rejections =
            SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
                .basis_consistency_failures;

        assert_eq!(
            total_rejections, 0,
            "anticipated warm-start: expected 0 basis rejections after relocating \
         anticipated_state_out into the state region, got {total_rejections} \
         (reconstruct_basis must match cut rows by CutPool slot identity, not by \
         absolute column index, so the column shift is transparent)"
        );
    }
}

mod anticipated_two_plants_smoke {
    //! Integration test verifying the training lower bound for a 6-stage system
    //! with 2 anticipated thermals (K_1=2, K_2=4), 1 backup thermal, and 1 hydro.
    //!
    //! ## Multi-plant LP layout
    //!
    //! With `n_anticipated=2` and `K_max=4`, the anticipated-state block has
    //! `2 * 4 = 8` columns in slot-major, plant-minor order (the index arithmetic
    //! the assertions below depend on):
    //!
    //! ```text
    //! ant_start + 0 = slot 0, plant 0  (K=2 plant — delivery slot)
    //! ant_start + 1 = slot 0, plant 1  (K=4 plant — delivery slot)
    //! ant_start + 2 = slot 1, plant 0  (K=2 plant — decision slot)
    //! ant_start + 3 = slot 1, plant 1  (K=4 plant)
    //! ant_start + 4 = slot 2, plant 0  (PADDING for K=2 plant)
    //! ant_start + 5 = slot 2, plant 1  (K=4 plant)
    //! ant_start + 6 = slot 3, plant 0  (PADDING for K=2 plant)
    //! ant_start + 7 = slot 3, plant 1  (K=4 plant — decision slot)
    //! ```
    //!
    //! ## Ring-buffer shift invariant (plant 0, stages 1→2)
    //!
    //! The shift invariant asserts slot 1 at stage `t` equals slot 0 at stage `t+1`
    //! (t≥1). Using t=1→t=2 (not t=0→t=1) avoids the trivial identity where
    //! `basis_cache[0]` (forward capture) and `basis_cache[1]` (backward trial point
    //! for stage 1, which also holds the forward outgoing of stage 0) carry the same
    //! state, so it exercises a genuine backward-to-backward ring-buffer advancement.

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, HydroStorage,
        InitialConditions, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // EXPECTED_LB = 0.0 is pinned from a converged run of this fixture. The test
    // validates slot-major LP layout, per-plant ring-buffer shift, and basis-cache
    // capture across two anticipated plants — not a closed-form cost. Re-pin only
    // after deliberate fixture changes.
    const EXPECTED_LB: f64 = 0.0_f64;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 6-stage two-anticipated-plant system. `SystemBuilder::build()` sorts
    /// thermals by id into `[id=2 (ant K=2), id=4 (backup), id=5 (ant K=4)]`, so the
    /// anticipated-local indices the assertions use are plant 0 → id=2, plant 1 → id=5.
    /// The backup thermal alone covers the 150 MW load, so the LP is always feasible.
    fn build_system_two_anticipated() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let ant_id_k2 = EntityId(2);
        let thermal_ant_k2 = make_thermal(
            ant_id_k2,
            ThermalSpec {
                name: "T_ant_k2".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let ant_id_k4 = EntityId(5);
        let thermal_ant_k4 = make_thermal(
            ant_id_k4,
            ThermalSpec {
                name: "T_ant_k4".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 80.0,
                cost_per_mwh: 40.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 4 }),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
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

        let n_stages = 6_usize;
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
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 4;
        let n_st = n_stages;

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

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 3,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
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

        // Seed lengths force locked deliveries: plant 0 at stages 0..=1, plant 1 at
        // stages 0..=3 — the committed costs the lower bound includes.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: ant_id_k2,
                    values_mw: vec![60.0, 30.0],
                },
                AnticipatedCommitmentHistory {
                    thermal_id: ant_id_k4,
                    values_mw: vec![20.0, 25.0, 30.0, 35.0],
                },
            ],
            recent_observations: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant_k2, thermal_backup, thermal_ant_k4])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system_two_anticipated: valid")
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
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 12 }]),
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

    #[test]
    fn test_two_anticipated_plants_k1_2_k2_4_convergence() {
        let ant_id_k2: u32 = 2;
        let ant_id_k4: u32 = 5;
        let backup_id: u32 = 4;
        assert!(ant_id_k2 < ant_id_k4);
        assert_ne!(backup_id, ant_id_k4);

        let system = build_system_two_anticipated();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 12, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(result.iterations, 12);

        let actual = result.final_lb;
        let expected = EXPECTED_LB;
        let rel_diff = if expected.abs() > f64::EPSILON {
            (actual - expected).abs() / expected.abs()
        } else {
            actual.abs()
        };
        assert!(
            rel_diff < 1e-6,
            "final_lb mismatch: {actual} vs {expected} (rel_diff={rel_diff}). \
         If intentional, update EXPECTED_LB."
        );

        let state = setup.stage_state();
        assert_eq!(state.n_anticipated, 2);
        assert_eq!(state.k_max, 4);

        let n_anticipated = state.n_anticipated;
        let k_max = state.k_max;
        let ant_start = state.anticipated_state.start;
        let ant_block_len = n_anticipated * k_max;

        let basis_cache = &result.basis_cache;
        assert_eq!(basis_cache.len(), 6);

        let s0 = basis_cache[0]
            .as_ref()
            .expect("stage 0 basis must be Some")
            .state_at_capture
            .as_slice();

        let ant_slice = &s0[ant_start..ant_start + ant_block_len];
        assert_eq!(ant_slice.len(), 8);
        for &v in ant_slice {
            assert!(v.is_finite(), "anticipated state must be finite");
        }

        let s1 = basis_cache[1]
            .as_ref()
            .expect("stage 1 basis must be Some")
            .state_at_capture
            .as_slice();
        let s2 = basis_cache[2]
            .as_ref()
            .expect("stage 2 basis must be Some")
            .state_at_capture
            .as_slice();

        let slot1_p0_at_stage1 = s1[ant_start + n_anticipated];
        let slot0_p0_at_stage2 = s2[ant_start];
        assert!(
            (slot1_p0_at_stage1 - slot0_p0_at_stage2).abs() < 1e-9,
            "ring-buffer shift invariant violated: slot-1@stage-1={slot1_p0_at_stage1}, \
         slot-0@stage-2={slot0_p0_at_stage2}"
        );
    }
}

mod anticipated_simulation_ring_buffer {
    //! Regression: the simulation pipeline (`solve_simulation_stage`) must call
    //! `shift_anticipated_state` once per stage, like the inflow-lag ring buffer.
    //!
    //! Without the shift, the next stage's `ws.current_state` carries the post-solve
    //! primal of the unbounded `anticipated_state` columns — `incoming - decision`,
    //! the residual the Category 6 fixing rows leave — instead of the shifted
    //! ring-buffer state, so the fishing constraint at delivery stage `K` reads a
    //! never-advanced slot 0. The contract this pins: with the anticipated thermal
    //! cheaper than the backup, the matured commitment equals the in-study decision
    //! made `K` stages earlier,
    //!
    //! `anticipated_committed_mw(t = K) == anticipated_decision_mw(t = 0)`,
    //!
    //! not the seeded `past_anticipated_commitments` a missing shift would surface.

    use std::sync::mpsc;

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractStageBounds, EntityId, HydroStageBounds, HydroStagePenalties, HydroStorage,
        InitialConditions, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build a deterministic K-stage system: one bus, one mostly-inactive hydro, a
    /// cheap anticipated thermal (id 2) plus an expensive backup (id 4), constant
    /// 150 MW load, and `past_anticipated_commitments` seeded with `past_commitments_mw`.
    ///
    /// The non-zero seed is intentional: building the resolved `System` directly via
    /// `SystemBuilder::new()` bypasses the `cobre-io` parse-and-validate pipeline, so
    /// the semantic validator that rejects non-zero `values_mw` does not fire — that
    /// rule applies to JSON through `load_case`, not to directly-constructed `System`.
    fn build_system(
        k: usize,
        past_commitments_mw: Vec<f64>,
        n_stages: usize,
    ) -> cobre_core::System {
        use chrono::NaiveDate;

        assert_eq!(
            past_commitments_mw.len(),
            k,
            "past_commitments_mw length must equal K (lead_stages)",
        );

        let bus = make_bus(
            EntityId(1),
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

        // Anticipated thermal: K lead stages, very cheap so the optimal policy
        // saturates anticipated dispatch at the load level.
        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
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

        // Backup thermal: very expensive so the LP prefers anticipated dispatch
        // whenever possible.
        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // Hydro: small to keep the model deterministic in the thermal regime.
        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
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
                hydro_id: EntityId(3),
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
                bus_id: EntityId(1),
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
        // ResolvedBounds::new applies one default to ALL thermals; without these
        // per-thermal overrides the cheap anticipated (index 0) and expensive backup
        // (index 1) are indistinguishable and decision_at(t) collapses to zero,
        // neutering the regression. The override must span the padding region
        // `[n_stages, n_stages + k_max)` too — `fill_anticipated_columns` reads the
        // delivery-stage axis there, so the decision column needs its cost out to K.
        let thermal_axis = n_stages + k;
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // anticipated
            bounds.thermal_bounds_mut(0, s).max_generation_mw = 100.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // backup
            bounds.thermal_bounds_mut(1, s).max_generation_mw = 200.0;
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

        // Seed the anticipated ring buffer with NON-ZERO past commitments. This
        // is the lever that exposes the bug: if the simulation pipeline failed
        // to shift the buffer, slot 0 at stage K would still report the seed
        // values rather than the in-study decision.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: past_commitments_mw,
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
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    /// Build a [`Config`] for training + 1-scenario deterministic simulation.
    fn build_config(training_iters: u32) -> Config {
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
                    limit: training_iters,
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
    // Test
    // ---------------------------------------------------------------------------

    /// K=1 case of the module-doc invariant: stage-1 matured commitment equals the
    /// stage-0 decision, not the non-zero seed a missing ring-buffer shift surfaces.
    #[test]
    fn simulation_ring_buffer_shifts_anticipated_state_k1() {
        let k: usize = 1;
        let n_stages: usize = 5;
        // Seed (7 MW) has no relationship to load/bounds, so the LP will not pick it
        // as a decision — a seed equal to a plausible decision would let the buggy and
        // fixed paths agree and neuter the test.
        let seed: Vec<f64> = vec![7.0];

        let system = build_system(k, seed.clone(), n_stages);
        let config = build_config(50);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 50, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let sim_run = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            sim_run.costs.len(),
            1,
            "simulation must produce exactly one scenario cost",
        );
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

        let anticipated_thermal_id: i32 = 2;
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

        // Under always-active fishing, stage-0 committed reads the seed at slot 0.
        let d0 = decision_at(0)
            .expect("anticipated_decision_mw must exist at stage 0 (t + K < n_stages)");
        let c0 = committed_at(0)
            .expect("anticipated_committed_mw must be Some at stage 0 under always-active fishing");
        assert!(
            (c0 - 7.0).abs() < 1e-6,
            "committed_at(0) must equal the K=1 seed values_mw[0]=7.0; got {c0}",
        );
        assert!(
            d0.abs() > 1e-6,
            "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
        );

        let c1 = committed_at(1).expect("committed at stage 1 must exist (K <= 1)");
        assert!(
            (c1 - d0).abs() < 1e-6,
            "REGRESSION (ring-buffer shift): stage 1 committed ({c1}) must equal \
         stage 0 decision ({d0}). On the buggy code path the ring buffer was \
         never shifted in simulation, so stage 1's Cat 6 RHS carried the \
         residual `seed - d_0` (negative when d_0 > seed) and the LP was \
         infeasible. With the shift, Cat 6 RHS = d_0 and gt_anticipated at \
         stage 1 saturates at d_0 (cost zeroed at delivery).",
        );
    }

    /// K=2 case of the module-doc invariant: the two pre-horizon stages read the seed
    /// slots, and from stage 2 the matured commitment equals the decision made K=2
    /// stages earlier (two shifts carry it into slot 0), never the seed, never zero.
    #[test]
    fn simulation_ring_buffer_shifts_anticipated_state_k2() {
        let k: usize = 2;
        let n_stages: usize = 6;
        // Seed slots are distinct from d_0 (which saturates near the thermal max of
        // 100), so neither slot can coincide with a decision and mask a missing shift.
        let seed: Vec<f64> = vec![50.0, 30.0];

        let system = build_system(k, seed.clone(), n_stages);
        let config = build_config(10);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let _sim_run = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");
        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(scenario_results.len(), 1);
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), n_stages);

        let anticipated_thermal_id: i32 = 2;
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

        // Pre-horizon: stage 0 reads seed slot 0; stage 1 reads slot 1 after the
        // stage-0 shift moves it into slot 0.
        let c0 = committed_at(0)
            .expect("committed_at(0) must be Some under always-active fishing with K=2");
        assert!(
            (c0 - 50.0).abs() < 1e-6,
            "committed_at(0) must equal K=2 seed values_mw[0]=50.0; got {c0}",
        );
        let c1 = committed_at(1)
            .expect("committed_at(1) must be Some under always-active fishing with K=2");
        assert!(
            (c1 - 30.0).abs() < 1e-6,
            "committed_at(1) must equal K=2 seed values_mw[1]=30.0 (shifted to slot 0); got {c1}",
        );

        let d0 = decision_at(0).expect("decision at stage 0 must exist (0 + K < n_stages)");

        assert!(
            d0.abs() > 1e-6,
            "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
        );

        let c2 = committed_at(2).expect("committed at stage 2 must exist (K <= 2)");
        assert!(
            (c2 - d0).abs() < 1e-6,
            "REGRESSION (ring-buffer shift, K=2): stage 2 committed ({c2}) must \
         equal stage 0 decision ({d0}). On the buggy code path the ring \
         buffer was never shifted in simulation, so stage 2's Cat 6 RHS \
         carried a stale residual instead of the d_0 that the two shifts \
         (end of stage 0, end of stage 1) propagated into slot 0.",
        );
    }
}

mod anticipated_generic_constraint_e2e {
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

    use std::path::Path;

    use chrono::NaiveDate;
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ConstraintExpression, ConstraintSense, ContractStageBounds, EntityId, GenericConstraint,
        HydroStageBounds, HydroStagePenalties, InitialConditions, LineStageBounds,
        LineStagePenalties, LinearTerm, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties,
        SlackConfig, SystemBuilder, ThermalStageBounds, VariableRef,
        entities::{bus::DeficitSegment, thermal::AnticipatedConfig},
        scenario::LoadModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

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
        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: DEFICIT_COST,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANT_THERMAL_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: ANT_MAX_MW,
                cost_per_mwh: ANT_COST,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_THERMAL_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_MAX_MW,
                cost_per_mwh: BACKUP_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let stages: Vec<Stage> = (0..N_STAGES)
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
                        ..Default::default()
                    },
                )
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
        let id_map: std::collections::HashMap<i32, usize> =
            [(1_i32, 0_usize)].into_iter().collect();
        let raw_bounds: Vec<(i32, i32, Option<i32>, f64)> = (0..N_STAGES as i32)
            .map(|stage_id| (1_i32, stage_id, None::<i32>, CONSTRAINT_BOUND_MW))
            .collect();
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, raw_bounds.into_iter());

        let constrained_system = build_system(vec![constraint], generic_bounds);
        let mut constrained_setup = build_setup_in_code(constrained_system, &config);
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
        let mut baseline_setup = build_setup_in_code(baseline_system, &config);
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
        r#"{"buses":[{"id":1,"name":"B1","operational_start_date":"2024-01-02","deficit_segments":[{"depth_mw":null,"cost":1000.0}]}]}"#,
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
      "operational_start_date": "2024-01-03",
      "bus_id": 1,
      "generation": { "min_mw": 0.0, "max_mw": 100.0 },
      "cost_per_mwh": 10.0,
      "anticipated_config": { "lead_stages": 2 }
    },
    {
      "id": 3,
      "name": "T_backup",
      "operational_start_date": "2024-01-04",
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
}

mod d34_anticipated_varying_blocks_shape {
    //! Case-shape assertions for the D34 deterministic fixture.
    //!
    //! D34 is the regression backstop for the relocation of the
    //! `anticipated_state_out` LP column out of the per-block (`n_blks`-dependent)
    //! control region and into the stage-invariant state region. The bug that
    //! relocation fixes can only fire when an anticipated commitment **matures at an
    //! interior stage whose block count differs from stage 0's** — so this fixture
    //! must simultaneously satisfy two shape constraints that no shipped case
    //! combined before:
    //!
    //! 1. at least one anticipated thermal whose `lead_stages` `K_i` matures
    //!    **strictly inside** the horizon (`stage + K_i < n_stages`) at an interior
    //!    delivery stage, and
    //! 2. a per-stage-varying block schedule (block counts differ across stages),
    //!    with the maturation stage landing on an off-stage-0 block count.
    //!
    //! This test pins those two properties so a future edit to the fixture inputs
    //! that silently flattens the block schedule, drops the anticipated thermal, or
    //! pushes the only commitment's maturation outside the horizon is caught here
    //! rather than degrading the `parity_hash_d34` regression to a no-op.

    use std::path::Path;

    fn d34_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join("d34-anticipated-varying-blocks")
    }

    #[test]
    fn d34_combines_anticipated_thermal_with_non_uniform_block_schedule() {
        let system = cobre_io::load_case(&d34_dir()).expect("D34 case must load");

        let block_counts: Vec<usize> = system.stages().iter().map(|s| s.blocks.len()).collect();
        assert_eq!(
            block_counts,
            vec![1, 3, 2],
            "D34 must ship the d33-style non-uniform [1, 3, 2] block schedule; \
         got {block_counts:?}"
        );
        let stage0_blocks = block_counts[0];
        assert!(
            block_counts.iter().any(|&c| c != stage0_blocks),
            "at least one interior stage must differ from stage 0's block count, \
         or the case cannot exercise an off-stage-0 maturation"
        );

        let n_stages = system.stages().len();

        let anticipated: Vec<_> = system
            .thermals()
            .iter()
            .filter_map(|t| t.anticipated_config.map(|cfg| (t.id, cfg.lead_stages)))
            .collect();
        assert!(
            !anticipated.is_empty(),
            "D34 must declare at least one anticipated thermal"
        );

        // A commitment at decision stage `s` matures at `s + K_i` and is active iff
        // `s + K_i < n_stages` (strict). The off-stage-0 maturation requires such a
        // delivery stage to land on an interior block count differing from stage 0's.
        let exercises_off_stage0_maturation = anticipated.iter().any(|&(_, k_i)| {
            let k = k_i as usize;
            (0..n_stages).any(|decision_stage| {
                let delivery_stage = decision_stage + k;
                delivery_stage < n_stages && block_counts[delivery_stage] != stage0_blocks
            })
        });
        assert!(
            exercises_off_stage0_maturation,
            "no anticipated commitment matures strictly inside the horizon at an \
         interior stage whose block count differs from stage 0's; \
         anticipated K_i = {anticipated:?}, block_counts = {block_counts:?}, \
         n_stages = {n_stages} — the case would not exercise the relocated \
         anticipated_state_out column"
        );

        // Pin the K=1 maturation coordinates: decision at stage 0 delivers at stage 1
        // (3 blocks), decision at stage 1 delivers at stage 2 (2 blocks) — both off
        // stage 0's single-block stride.
        let (_, k1) = anticipated
            .iter()
            .find(|&&(_, k)| k == 1)
            .expect("D34's anticipated thermal uses lead_stages = 1");
        assert_eq!(*k1, 1);
        assert_eq!(block_counts[1], 3, "stage-1 delivery lands on 3 blocks");
        assert_eq!(block_counts[2], 2, "stage-2 delivery lands on 2 blocks");
    }
}

mod d37_anticipated_commissioning_simulation {
    //! Anticipated-thermal commissioning-window gating must reach the simulation
    //! output, and warm-start must survive the dormancy boundary.
    //!
    //! ## What this case exercises that no other anticipated case does
    //!
    //! `d37-anticipated-commissioning` is the only deterministic case combining an
    //! anticipated thermal with a commissioning window. The anticipated thermal T1
    //! (`K=2`, cheap, `max 150 MW`) carries window `[entry=2, exit=4)` over a
    //! 6-stage horizon (ids 0..6) with a per-stage-varying block schedule (1/3/2/3/1/2).
    //!
    //! The decision gate (`StateLayout::is_anticipated_decision_active`) is the
    //! conjunction of the strict horizon clause (`t + K < n_stages`) and the
    //! operation-window clause keyed on the DELIVERY stage (`commissioning_active(2,
    //! 4, id(t + 2))`):
    //!
    //! - `t = 0` → delivers at id 2 ∈ `[2, 4)` → ACTIVE (the pre-entry decision,
    //!   priced at `entry − K`),
    //! - `t = 1` → delivers at id 3 ∈ `[2, 4)` → ACTIVE,
    //! - `t = 2` → delivers at id 4 ∉ `[2, 4)` → INACTIVE (post-exit drain begins),
    //! - `t = 3` → delivers at id 5 ∉ `[2, 4)` → INACTIVE,
    //! - `t = 4, 5` → `t + K ≥ 6` → horizon-INACTIVE.
    //!
    //! The GENERATION column (operation window) zeroes T1's generation outside
    //! `[entry, exit)`: T1 generates only at stages 2 and 3.
    //!
    //! ## What the parity hash cannot see
    //!
    //! The parity baseline hashes hydro storage/water/cuts/convergence, NOT thermal
    //! generation, the anticipated decision, or the ring buffer. A gating bug that
    //! let T1 run outside its window, or never delivered the pre-entry commitment, or
    //! left the ring buffer un-drained, could still hash-match. This test exercises
    //! those paths directly through train + simulate.

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{
        SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d37-anticipated-commissioning")
    }

    fn build_setup(case_dir: &Path) -> (StudySetup, cobre_io::config::Config) {
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped case disables simulation (parity trains only); enable one
        // deterministic scenario so the thermal extraction paths run.
        config.simulation = cobre_io::config::SimulationConfig {
            enabled: true,
            num_scenarios: 1,
            io_channel_capacity: 8,
            ..cobre_io::config::SimulationConfig::default()
        };

        let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
        let prepare_result =
            prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let setup =
            StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup::new");
        (setup, config)
    }

    const T1_ID: i32 = 1;
    const ENTRY: usize = 2;
    const EXIT: usize = 4;
    const K: usize = 2;
    const N_STAGES: usize = 6;
    const TOL: f64 = 1e-6;

    /// Locate the anticipated thermal (T1) result at a given stage.
    fn t1_at(
        scenario: &cobre_sddp::simulation::SimulationScenarioResult,
        stage: usize,
    ) -> &cobre_sddp::simulation::SimulationThermalResult {
        scenario.stages[stage]
            .thermals
            .iter()
            .find(|t| t.thermal_id == T1_ID)
            .unwrap_or_else(|| panic!("T1 (id={T1_ID}) missing at stage {stage}"))
    }

    /// Train the windowed anticipated case, simulate one deterministic scenario, and
    /// assert the commissioning window gates the generation, the pre-entry decision
    /// delivers at `entry`, and the ring buffer drains after `exit`.
    #[test]
    fn anticipated_commissioning_window_gates_simulation_output() {
        let dir = case_dir();
        let (mut setup, _config) = build_setup(&dir);

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "training must run at least one iteration",
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect(
                "simulate must not return Err (a windowed anticipated thermal must stay feasible)",
            );

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");
        assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            N_STAGES,
            "one record per study stage"
        );

        // (i) GENERATION == 0 at every stage outside [entry, exit).
        for stage in 0..N_STAGES {
            if !(ENTRY..EXIT).contains(&stage) {
                let g = t1_at(scenario, stage).generation_mw;
                assert!(
                    g.abs() <= TOL,
                    "stage {stage} is outside the operation window [{ENTRY}, {EXIT}): \
                 T1 generation must be 0, got {g}",
                );
            }
        }

        // (ii) Generation present at the first operating stage `entry`. This proves
        // the pre-entry decision at `entry − K` delivered: the only way T1 can
        // generate at stage `entry` is for the commitment placed K stages earlier to
        // have matured here (the generation column is in-window, the decision column
        // at `entry` itself delivers at `entry + K`, outside the window).
        let g_entry = t1_at(scenario, ENTRY).generation_mw;
        assert!(
            g_entry > TOL,
            "T1 must generate at the first operating stage {ENTRY} (the pre-entry \
         decision at stage {} delivered here); got {g_entry}",
            ENTRY - K,
        );

        // (iii) The ring buffer drains to 0 within K stages after exit. The committed
        // MW (slot 0 of the ring buffer) is the matured commitment; after the last
        // in-window delivery (stage EXIT-1 = 3) no new in-window decision is made, so
        // by stage EXIT + K - 1 = 5 the buffer has shifted all residual commitments
        // out and reads 0.
        let drain_stage = EXIT + K - 1;
        let committed_drain = t1_at(scenario, drain_stage)
            .anticipated_committed_mw
            .expect("anticipated thermal must report committed MW");
        assert!(
            committed_drain.abs() <= TOL,
            "the ring buffer must drain to 0 within K={K} stages after exit={EXIT}: \
         committed MW at stage {drain_stage} must be 0, got {committed_drain}",
        );

        // The decision column is inactive at every post-exit / horizon-edge stage:
        // the simulation read returns None there (the column is pinned to [0, 0]).
        for stage in ENTRY..N_STAGES {
            let decision = t1_at(scenario, stage).anticipated_decision_mw;
            assert!(
                decision.is_none(),
                "T1 decision must be inactive (None) at stage {stage} (delivery at \
             {} is outside [{ENTRY}, {EXIT}) or beyond the horizon); got {decision:?}",
                stage + K,
            );
        }
        // Conversely, the pre-entry decision stages (0 and 1, delivering at 2 and 3)
        // are active and report a decision value.
        for stage in 0..ENTRY {
            let decision = t1_at(scenario, stage).anticipated_decision_mw;
            assert!(
                decision.is_some(),
                "T1 decision must be active at the pre-entry stage {stage} (delivers \
             at {} ∈ [{ENTRY}, {EXIT})); got None",
                stage + K,
            );
        }
    }

    /// Warm-start regression: the windowed anticipated study must reconstruct every
    /// stored basis with zero rejections across the dormancy boundary.
    ///
    /// The relocated `anticipated_state_out` column lives in the stage-invariant
    /// state region; the commissioning window toggles which stages emit an active
    /// `anticipated_state_out_def` row, so the per-stage row/column counts change at
    /// the entry/exit boundaries. `reconstruct_basis` matches stored cut rows to
    /// current LP rows by `CutPool` slot identity (never by absolute index), so the
    /// dormancy-driven count change must remain transparent to the warm start.
    /// `basis_consistency_failures` counts every rejected offered basis; it must stay
    /// at zero.
    #[test]
    fn anticipated_commissioning_warm_start_zero_basis_rejections() {
        let dir = case_dir();
        let (mut setup, _config) = build_setup(&dir);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = &outcome.result;
        // Warm-start reconstruction runs on every iteration after the first, so the
        // regression needs at least two iterations to exercise `reconstruct_basis`
        // across the dormancy boundary. The shipped config's iteration limit governs
        // the exact count; a deterministic case may converge before the cap.
        assert!(
            result.iterations >= 2,
            "warm-start regression needs >= 2 iterations to exercise reconstruct_basis \
         on iteration 2+, got {}",
            result.iterations,
        );

        let total_rejections =
            SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
                .basis_consistency_failures;
        assert_eq!(
            total_rejections, 0,
            "windowed anticipated warm-start: expected 0 basis rejections across the \
         dormancy boundary, got {total_rejections} (reconstruct_basis must match \
         cut rows by CutPool slot identity, not by absolute index)",
        );
    }
}
